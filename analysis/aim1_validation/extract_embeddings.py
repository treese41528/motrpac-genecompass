#!/usr/bin/env python3
"""
extract_embeddings.py -- Aim-1 validation extractor (GPU).

One forward pass over a tokenized cell dataset with the fine-tuned rat GeneCompass
checkpoint, producing BOTH:
  (1) CLS cell embeddings  [n_cells, 768]   (position 0) -- for V1 (held-out clustering)
  (2) per-gene CONTEXTUAL embeddings = mean over cells of the last-hidden-state at each
      target token's position -- for V2 (homolog similarity, init-free)

The CLS path mirrors finetune/genecompass/embed_cells.py exactly (same model load).
The contextual path accumulates, for every input position j holding a token t in the
--target-tokens set, the hidden state at OUTPUT position j+1 (CLS is prepended at 0, so
input token j -> output j+1). Mean per token at the end; tokens seen in < --min-cells
cells are reported but flagged low-coverage.

Usage (run with the right --species per dataset):
  python extract_embeddings.py --model-dir <ckpt> --dataset <hf_dataset> --species 2 \
      --n-cells 40000 --target-tokens rat_targets.npy --out out_rat \
      [--save-cls]            # CLS only needed for the rat run (V1)
"""
import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(os.environ.get("PIPELINE_ROOT", ".")).resolve()
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "vendor" / "GeneCompass"))


def load_model(model_dir, device):
    """Identical load to embed_cells.py: rebuild knowledges from the state dict,
    patch cls_embedding to 3 species, load strict=False."""
    from transformers import BertConfig
    from genecompass import BertForMaskedLM
    from torch import nn

    model_dir = Path(model_dir)
    with open(model_dir / "config.json") as f:
        cfg = json.load(f)
    state = torch.load(model_dir / "pytorch_model.bin", map_location="cpu", weights_only=False)
    knowledges = {
        "promoter": state["bert.embeddings.promoter_knowledge"],
        "co_exp": state["bert.embeddings.co_exp_knowledge"],
        "gene_family": state["bert.embeddings.gene_family_knowledge"],
        "peca_grn": state["bert.embeddings.peca_grn_knowledge"],
    }
    homolog_path = _ROOT / "vendor" / "GeneCompass" / "prior_knowledge" / "homologous_hm_token.pickle"
    with open(homolog_path, "rb") as f:
        raw = pickle.load(f)
        knowledges["homologous_gene_human2mouse"] = {v: k for k, v in raw.items()}
    cfg["warmup_steps"] = 0
    cfg["emb_warmup_steps"] = 1
    model = BertForMaskedLM(BertConfig(**cfg), knowledges=knowledges)
    model.bert.cls_embedding = nn.Embedding(3, cfg["hidden_size"])
    model.load_state_dict(state, strict=False)
    return model.eval().to(device)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--dataset", required=True, help="HF tokenized dataset (load_from_disk)")
    ap.add_argument("--species", type=int, required=True, help="0=human 1=mouse 2=rat (CLS token + species)")
    ap.add_argument("--n-cells", type=int, default=None, help="order-preserving prefix cap (default all)")
    ap.add_argument("--target-tokens", default=None, help=".npy int array of token ids to extract contextual means for")
    ap.add_argument("--out", required=True, help="output dir")
    ap.add_argument("--save-cls", action="store_true", help="also save CLS cell embeddings (V1; rat run only)")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--min-cells", type=int, default=20, help="coverage floor flagged in the report")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    from datasets import load_from_disk
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    device = args.device
    model = load_model(args.model_dir, device)

    ds = load_from_disk(args.dataset)
    if args.n_cells is not None and args.n_cells < len(ds):
        ds = ds.select(range(args.n_cells))   # order-preserving prefix (caller stratifies upstream)
    n = len(ds)
    print(f"[extract] species={args.species} cells={n} dataset={args.dataset}", flush=True)

    # target-token bookkeeping for contextual means
    tgt = np.load(args.target_tokens).astype(np.int64) if args.target_tokens else np.array([], dtype=np.int64)
    H = model.config.hidden_size
    V = int(model.config.vocab_size)
    ctx_sum = np.zeros((len(tgt), H), dtype=np.float64)
    ctx_cnt = np.zeros(len(tgt), dtype=np.int64)
    tok2row = np.full(V, -1, dtype=np.int64)              # token id -> target row (-1 = not a target)
    if len(tgt):
        tok2row[tgt] = np.arange(len(tgt))
    have_tgt = len(tgt) > 0
    cls_chunks = []

    with torch.no_grad():
        for s in range(0, n, args.batch_size):
            b = ds[s:s + args.batch_size]
            input_ids = torch.tensor(b["input_ids"]).to(device)            # [B, 2048]
            values = torch.tensor(b["values"]).to(device)
            sp = torch.full((input_ids.shape[0], 1), args.species, dtype=torch.long).to(device)
            attn = (input_ids != 0).long()
            outputs = model.bert(input_ids=input_ids, values=values, attention_mask=attn,
                                 species=sp, emb_warmup_alpha=1.0, output_hidden_states=False)
            hs = outputs[0]                                                # [B, 2049, 768] (pos0=CLS)
            if args.save_cls:
                cls_chunks.append(hs[:, 0, :].cpu().numpy())
            if have_tgt:
                ids_np = input_ids.cpu().numpy()                          # [B, 2048] (input token j)
                hs_gene = hs[:, 1:, :].cpu().numpy()                       # [B, 2048, 768] (output pos j+1)
                rows = tok2row[ids_np]                                     # [B, 2048], -1 where not a target
                mask = rows >= 0
                if mask.any():
                    r = rows[mask]                                         # [M]
                    np.add.at(ctx_sum, r, hs_gene[mask].astype(np.float64))
                    np.add.at(ctx_cnt, r, 1)
            if (s // args.batch_size) % 50 == 0:
                print(f"  {min(s + args.batch_size, n)}/{n}", flush=True)

    if args.save_cls:
        cls = np.concatenate(cls_chunks, axis=0).astype(np.float32)
        np.save(out / "cls_embeddings.npy", cls)
        print(f"[extract] saved CLS {cls.shape} -> {out/'cls_embeddings.npy'}", flush=True)

    if have_tgt:
        mean = np.full((len(tgt), H), np.nan, dtype=np.float32)
        ok = ctx_cnt > 0
        mean[ok] = (ctx_sum[ok] / ctx_cnt[ok][:, None]).astype(np.float32)
        np.savez(out / "contextual.npz", tokens=tgt, mean=mean, count=ctx_cnt)
        covered = int((ctx_cnt >= args.min_cells).sum())
        print(f"[extract] contextual: {covered}/{len(tgt)} target tokens with >= {args.min_cells} cells "
              f"(any-coverage {int(ok.sum())}/{len(tgt)}) -> {out/'contextual.npz'}", flush=True)


if __name__ == "__main__":
    main()
