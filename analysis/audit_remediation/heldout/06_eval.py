#!/usr/bin/env python
"""06_eval.py -- masked-token losses of one model on the rat cells of one set, kept per cell.

Implements the measurement of STEP4_PREREG.md V3 ("the masked-token evaluator used for the retrain's criteria
C1 and C2 (finetune/genecompass/eval_mlm.py), same masking and seed"). eval_mlm.py's `eval` command cannot be
run on a rat-only set (it divides by the number of masked human tokens, which is zero) and it reports one
number per species. This script therefore calls eval_mlm.py's own functions and repeats its loop for the rat
cells, line for line:
  eval_mlm.load_model(model, "rat")            the model as saved, 3-row species table, warm-up fixed
  eval_mlm.mask_batch(...)                     15% of real tokens; 80% <mask>, 10% random id, 10% unchanged;
                                               masked input values -1; random ids from the 55,275 rat vocabulary
  torch.Generator().manual_seed(seed * 1000003 + 2 * 10007 + batch_index), seed 20261008, batches of 8 cells in
  dataset order, 32-bit precision, attention mask = non-pad positions, species id 2.
and accumulates the same token-weighted sums (gene-ID cross-entropy, value squared error, top-1 hits). In
addition it keeps, for each cell, the sum of its masked tokens' cross-entropy and squared error (the value
head's output is read with a forward hook on model.cls4value), and the same split by ortholog tier
(eval_mlm.load_tiers(), as --by-tier). No file of eval_mlm.py is changed.

On the in-sample probe (data/training/eval_probe_v2) the totals this script prints for rat must equal those of
`eval_mlm.py eval --by-tier`; 07_tables.py checks that against data/results/retrain_v2/mlm{,_by_tier}/.

Writes <out>.json (eval_mlm.py's layout for the rat block, plus the settings) and <out>.cells.npz.
Usage: 06_eval.py --model DIR --set HF_DATASET_DIR --out PREFIX
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C   # noqa: E402

sys.path.insert(0, str(C.ROOT / "finetune/genecompass"))
import eval_mlm as E   # noqa: E402

S_RAT, RAT_VOCAB = 2, 55275


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--set", required=True, dest="probe")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=8)       # eval_mlm.py's default, used for C1/C2
    ap.add_argument("--seed", type=int, default=20261008)      # eval_mlm.py's default, used for C1/C2
    a = ap.parse_args()
    out_json = C.out_path(str(a.out) + ".json")

    from datasets import load_from_disk
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, alpha, missing = E.load_model(a.model, "rat", device)
    captured = {}
    model.cls4value.register_forward_hook(lambda mod, inp, out: captured.__setitem__("v", out))

    ds = load_from_disk(a.probe)
    sp_all = ds.data.column("species").combine_chunks().flatten().to_numpy()
    assert len(sp_all) == len(ds)
    idx = np.flatnonzero(sp_all == S_RAT)
    tier_of = E.load_tiers()
    tier_names = sorted(set(tier_of.values()) | {"untiered"})
    lut = torch.full((RAT_VOCAB,), tier_names.index("untiered"), dtype=torch.long)
    for t, name in tier_of.items():
        if 0 <= t < RAT_VOCAB:
            lut[t] = tier_names.index(name)
    lut = lut.to(device)
    n, nt = len(idx), len(tier_names)
    cell = {k: torch.zeros(n, dtype=torch.float64, device=device) for k in ("ce", "se", "hit", "n")}
    tier = {k: torch.zeros(n * nt, dtype=torch.float64, device=device) for k in ("ce", "hit", "n")}

    ce_sum = se_sum = n_tok = n_hit = 0.0
    worst = 0.0
    for b, start in enumerate(range(0, n, a.batch_size)):
        sel = idx[start:start + a.batch_size].tolist()
        rows = ds.select(sel)[:]
        gen = torch.Generator().manual_seed(a.seed * 1000003 + S_RAT * 10007 + b)
        input_ids = torch.tensor(rows["input_ids"], dtype=torch.long)
        ids, vals, labels, labels_values = E.mask_batch(input_ids, torch.tensor(rows["values"], dtype=torch.float),
                                                        gen, RAT_VOCAB)
        spc = torch.full((ids.shape[0], 1), S_RAT, dtype=torch.long)
        att = (input_ids != E.PAD_ID).long()
        o = model(input_ids=ids.to(device), values=vals.to(device), attention_mask=att.to(device),
                  species=spc.to(device), labels=labels.to(device),
                  labels_values=labels_values.to(device), return_dict=True)
        m = labels != -100
        k = int(m.sum())
        ce_sum += float(o.id_loss) * k
        se_sum += float(o.value_loss) * k
        n_hit += int((o.logits.argmax(-1).cpu()[m] == labels[m]).sum())
        n_tok += k
        # per token, on the same logits and the value head's output
        md = m.to(device)
        lab = labels.to(device)[md]
        lg = o.logits[md].float()
        ce = torch.nn.functional.cross_entropy(lg, lab, reduction="none").double()
        hit = (lg.argmax(-1) == lab).double()
        pv = captured["v"].squeeze(-1)
        se = ((pv[md] - labels_values.to(device)[md]).float() ** 2).double()
        worst = max(worst, abs(float(ce.mean()) - float(o.id_loss)) / max(float(o.id_loss), 1e-12),
                    abs(float(se.mean()) - float(o.value_loss)) / max(float(o.value_loss), 1e-12))
        row = md.nonzero()[:, 0] + start
        one = torch.ones_like(ce)
        cell["ce"].index_add_(0, row, ce); cell["se"].index_add_(0, row, se)
        cell["hit"].index_add_(0, row, hit); cell["n"].index_add_(0, row, one)
        flat = row * nt + lut[lab]
        tier["ce"].index_add_(0, flat, ce); tier["hit"].index_add_(0, flat, hit); tier["n"].index_add_(0, flat, one)
        if b % 250 == 0:
            print(f"batch {b} / {(n + a.batch_size - 1) // a.batch_size}: id_loss so far {ce_sum / n_tok:.4f}",
                  flush=True)

    if worst > 1e-4:
        raise SystemExit(f"per-token sums disagree with the model's own batch losses (relative {worst:.2e})")
    tn = tier["n"].view(n, nt).sum(0).cpu().numpy()
    tce = tier["ce"].view(n, nt).sum(0).cpu().numpy()
    thit = tier["hit"].view(n, nt).sum(0).cpu().numpy()
    rat = {"cells": int(n), "masked_tokens": int(n_tok), "id_loss": ce_sum / n_tok, "value_loss": se_sum / n_tok,
           "top1_acc": n_hit / n_tok,
           "by_tier": {t: {"masked_tokens": int(tn[i]), "id_loss": float(tce[i] / tn[i]),
                           "top1_acc": float(thit[i] / tn[i])} for i, t in enumerate(tier_names) if tn[i] > 0}}
    out = {"model": str(a.model), "kind": "rat", "probe": str(a.probe), "seed": a.seed, "batch_size": a.batch_size,
           "emb_warmup_alpha": alpha, "missing_keys": missing,
           "mask": "eval_mlm.mask_batch: 15% of real tokens; 80% <mask>, 10% random id (55,275), 10% unchanged",
           "per_token_vs_batch_loss_max_rel_diff": worst,
           "per_cell_sum_check": {"id_loss": float(cell["ce"].sum() / cell["n"].sum()),
                                  "value_loss": float(cell["se"].sum() / cell["n"].sum())},
           "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
           "slurm_job": os.environ.get("SLURM_JOB_ID", ""),
           "species": {"rat": rat}}
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out, indent=2))

    def col(name):
        return np.asarray(ds.data.column(name).combine_chunks().take(idx).to_pylist()) if name in ds.column_names \
            else np.asarray([""] * n)
    length = ds.data.column("length").combine_chunks().flatten().to_numpy()[idx]
    np.savez_compressed(C.out_path(str(a.out) + ".cells.npz"),
                        row=idx, length=length, study_id=col("study_id"), sample_id=col("sample_id"),
                        cell_index=col("cell_index"), tier_names=np.asarray(tier_names),
                        n_masked=cell["n"].cpu().numpy(), ce_sum=cell["ce"].cpu().numpy(),
                        se_sum=cell["se"].cpu().numpy(), n_hit=cell["hit"].cpu().numpy(),
                        tier_n=tier["n"].view(n, nt).cpu().numpy(), tier_ce=tier["ce"].view(n, nt).cpu().numpy(),
                        tier_hit=tier["hit"].view(n, nt).cpu().numpy())
    print("rat", json.dumps({k: v for k, v in rat.items() if k != "by_tier"}))
    print(f"wrote {out_json}")


if __name__ == "__main__":
    main()
