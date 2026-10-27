#!/usr/bin/env python3
"""eval_mlm.py — fixed-probe masked-prediction metrics for the retrain-v2 pre-registration (C1, C2).

docs/RETRAIN_V2_PREREGISTRATION.md §2-3. Two subcommands:

  build-probe  draw N distinct cells per species from the training mix (fixed seed) and save them
               python finetune/genecompass/eval_mlm.py build-probe \
                   --mix data/training/mixed_species_dataset_v2 --out data/training/eval_probe_v2

  eval         score one model on the probe with fixed masks; writes <out>.json
               python finetune/genecompass/eval_mlm.py eval --model data/models/rat_genecompass_init_v2 \
                   --kind rat --probe data/training/eval_probe_v2 --out results/retrain_v2/mlm_init_v2
               (--kind base for GeneCompass_Base: 2 species, human/mouse cells only)

Masks follow GeneCompass's collator (15% of real tokens; of those 80% <mask>, 10% random token,
10% unchanged; masked values set to -1) but are drawn from a torch.Generator seeded by
(seed, species, batch index), and the random-token range is fixed per species (human/mouse: the
50,558-token GeneCompass vocabulary; rat: 55,275), so every model sees identical inputs.
Metrics are token-weighted over masked positions: gene-ID cross-entropy, value MSE, top-1 accuracy.
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "vendor" / "GeneCompass"))
sys.path.insert(0, str(ROOT / "lib"))

SPECIES = {0: "human", 1: "mouse", 2: "rat"}
GC_VOCAB = 50558          # GeneCompass_Base vocabulary (human + mouse tokens)
PAD_ID, MASK_ID = 0, 1


def build_probe(args):
    from datasets import load_from_disk
    mix = load_from_disk(args.mix)
    sp = np.asarray(mix.data.column("species").to_numpy(zero_copy_only=False)).reshape(-1)
    src = np.asarray(mix.data.column("source_row").to_numpy(zero_copy_only=False)).reshape(-1)
    rng = np.random.default_rng(args.seed)
    rows = []
    for s in sorted(SPECIES):
        idx = np.flatnonzero(sp == s)
        _, first = np.unique(src[idx], return_index=True)        # one mix row per distinct source cell
        distinct = idx[first]
        pick = np.sort(rng.choice(distinct, size=args.n_per_species, replace=False))
        rows.append(pick)
        print(f"{SPECIES[s]}: {len(idx):,} mix rows, {len(distinct):,} distinct cells -> {len(pick)}")
    probe = mix.select(np.concatenate(rows).tolist())
    keep = [c for c in ("input_ids", "values", "length", "species", "source", "source_row",
                        "study_id", "sample_id", "cell_index") if c in probe.column_names]
    probe = probe.select_columns(keep)
    out = Path(args.out)
    if out.exists():
        sys.exit(f"{out} exists; a probe set is built once (pre-registration §5)")
    probe.save_to_disk(str(out))
    from provenance import write_provenance
    write_provenance(out, inputs=[args.mix], params={"seed": args.seed, "n_per_species": args.n_per_species})
    print(f"wrote {out} ({len(probe):,} cells)")


def load_model(model_dir, kind, device):
    from transformers import BertConfig
    from genecompass import BertForMaskedLM
    model_dir = Path(model_dir)
    cfg = json.loads((model_dir / "config.json").read_text())
    cfg.update(warmup_steps=0, emb_warmup_steps=1)
    state = torch.load(model_dir / "pytorch_model.bin", map_location="cpu", weights_only=False)
    knowledges = {
        "promoter": state["bert.embeddings.promoter_knowledge"],
        "co_exp": state["bert.embeddings.co_exp_knowledge"],
        "gene_family": state["bert.embeddings.gene_family_knowledge"],
        "peca_grn": state["bert.embeddings.peca_grn_knowledge"],
    }
    with open(ROOT / "vendor/GeneCompass/prior_knowledge/homologous_hm_token.pickle", "rb") as f:
        knowledges["homologous_gene_human2mouse"] = {v: k for k, v in pickle.load(f).items()}
    model = BertForMaskedLM(BertConfig(**cfg), knowledges=knowledges)
    if kind == "rat":
        model.bert.cls_embedding = torch.nn.Embedding(3, cfg["hidden_size"])
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise RuntimeError(f"unexpected keys: {unexpected[:5]}")
    model.emb_warmup.forward = lambda: model.emb_warmup.alpha   # evaluate as saved; never step warmup
    return model.eval().to(device), float(model.emb_warmup.alpha), missing


def mask_batch(input_ids, values, gen, vocab):
    real = input_ids != PAD_ID
    masked = (torch.rand(input_ids.shape, generator=gen) < 0.15) & real
    replace = (torch.rand(input_ids.shape, generator=gen) < 0.8) & masked
    rand = (torch.rand(input_ids.shape, generator=gen) < 0.5) & masked & ~replace
    rand_ids = torch.randint(vocab, input_ids.shape, generator=gen)
    labels = input_ids.clone(); labels[~masked] = -100
    labels_values = values.clone()
    ids = input_ids.clone(); vals = values.clone()
    ids[replace] = MASK_ID; vals[replace] = -1
    ids[rand] = rand_ids[rand]; vals[rand] = -1
    return ids, vals, labels, labels_values


@torch.no_grad()
def evaluate(args):
    from datasets import load_from_disk
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, alpha, missing = load_model(args.model, args.kind, device)
    probe = load_from_disk(args.probe)
    sp_all = torch.as_tensor(np.asarray(probe[:]["species"])).reshape(-1)
    species = [0, 1] if args.kind == "base" else [0, 1, 2]
    out = {"model": str(args.model), "kind": args.kind, "probe": str(args.probe), "seed": args.seed,
           "emb_warmup_alpha": alpha, "missing_keys": missing, "species": {}}
    for s in species:
        idx = torch.nonzero(sp_all == s).flatten().tolist()
        vocab = 55275 if s == 2 else GC_VOCAB
        ce_sum = se_sum = n_tok = n_hit = 0.0
        for b, start in enumerate(range(0, len(idx), args.batch_size)):
            rows = probe.select(idx[start:start + args.batch_size])[:]      # dict of python lists
            gen = torch.Generator().manual_seed(args.seed * 1000003 + s * 10007 + b)
            input_ids = torch.tensor(rows["input_ids"], dtype=torch.long)
            ids, vals, labels, labels_values = mask_batch(input_ids, torch.tensor(rows["values"], dtype=torch.float),
                                                          gen, vocab)
            spc = torch.full((ids.shape[0], 1), s, dtype=torch.long)
            att = (input_ids != PAD_ID).long()
            o = model(input_ids=ids.to(device), values=vals.to(device), attention_mask=att.to(device),
                      species=spc.to(device), labels=labels.to(device),
                      labels_values=labels_values.to(device), return_dict=True)
            m = labels != -100
            k = int(m.sum())
            ce_sum += float(o.id_loss) * k
            se_sum += float(o.value_loss) * k
            n_hit += int((o.logits.argmax(-1).cpu()[m] == labels[m]).sum())
            n_tok += k
        out["species"][SPECIES[s]] = {"cells": len(idx), "masked_tokens": int(n_tok),
                                      "id_loss": ce_sum / n_tok, "value_loss": se_sum / n_tok,
                                      "top1_acc": n_hit / n_tok}
        print(SPECIES[s], json.dumps(out["species"][SPECIES[s]]))
    path = Path(str(args.out) + ".json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build-probe")
    b.add_argument("--mix", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--n-per-species", type=int, default=2000)
    b.add_argument("--seed", type=int, default=20261008)
    e = sub.add_parser("eval")
    e.add_argument("--model", required=True)
    e.add_argument("--kind", choices=["rat", "base"], required=True)
    e.add_argument("--probe", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--batch-size", type=int, default=8)
    e.add_argument("--seed", type=int, default=20261008)
    args = ap.parse_args()
    build_probe(args) if args.cmd == "build-probe" else evaluate(args)


if __name__ == "__main__":
    main()
