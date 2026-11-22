#!/usr/bin/env python
# coding: utf-8
r"""
build_grn.py -- Module B (Aim 2b) PILOT: per-cell differential trained-vs-control network (a model read-out).

Superseded by build_grn_pooled.py, which pseudobulks each arm and bootstraps; this file is kept because the
pooled and human builders import load_arms() from it. No production table comes from this script.

For one tissue x cell type, split the pseudo-cells into a CONTROL arm (training week 0) and a TRAINED arm
(weeks 1-8), then for each transcription factor (TF) run the in-silico deletion (perturb_cells.perturb_gene)
SEPARATELY in each arm. The per-gene contextual-embedding shift a TF deletion induces is the model's read-out
for that TF and gene in that arm. The DIFFERENTIAL edge is trained - control, taken only on the targets that
both arms report (docs/audit/remediation/STEP3_PREREG.md 8.1; the difference is build_grn_pooled.diff). A
target reported by one arm only gets no edge; the summary table counts them per TF. Output is a ranked edge list.

Arm labels come from augur_input/<tissue>/meta.tsv (columns sample, cell_type, label{control,trained},
week, sex), joined to the tokenized dataset by (sample, cell_type). Use the per-gene TARGET shift as the
edge weight (NOT the whole-cell CLS shift, which is near-noise for a single deletion -- see the smoke
findings). This is a per-cell mechanistic readout (GeneCompass has no cross-cell attention); the arm split
and the target aggregation are what make it a population, exercise-conditioned network.

Usage:
  python finetune/genecompass/build_grn.py \
      --model-dir data/models/rat_genecompass_finetuned_v2/models/rat_phase2_mixed_species/checkpoint-591763 \
      --dataset data/deconvolution/genecompass_input_v2/skmvl/dataset \
      --meta    data/deconvolution/augur_input/skmvl/meta.tsv \
      --cell-type "Skeletal muscle" --out data/deconvolution/grn/skmvl_skm.tsv --device cuda
"""
import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(os.environ.get("PIPELINE_ROOT", ".")).resolve()
sys.path.insert(0, str(_ROOT / "finetune" / "genecompass"))
import perturb_cells as pc   # noqa: E402


def load_arms(dataset_path, meta_path, cell_type):
    """Return (ids_ctrl, vals_ctrl), (ids_trained, vals_trained) for one cell type, split by meta 'label'."""
    from datasets import load_from_disk
    ds = load_from_disk(dataset_path)
    label = {}
    for r in csv.DictReader(open(meta_path), delimiter="\t"):
        label[(r["sample"], r["cell_type"])] = r.get("label", "").strip().lower()
    arms = {"control": ([], []), "trained": ([], [])}
    for r in ds:
        if r["cell_type"] != cell_type:
            continue
        lab = label.get((r["sample"], r["cell_type"]), "")
        if lab in arms:
            arms[lab][0].append(r["input_ids"])
            arms[lab][1].append(r["values"])

    def tens(pair):
        ids, vals = pair
        if not ids:
            return None, None
        return (torch.tensor(np.array(ids), dtype=torch.long),
                torch.tensor(np.array(vals), dtype=torch.float))
    return tens(arms["control"]), tens(arms["trained"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--meta", required=True, help="augur_input/<tissue>/meta.tsv (sample,cell_type,label,...)")
    ap.add_argument("--cell-type", required=True)
    ap.add_argument("--tf-file", default=str(_ROOT / "deconvolution" / "reference" / "rat_tf_ensrnog.tsv"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--min-cells", type=int, default=5, help="min expressing cells per arm to keep a TF")
    ap.add_argument("--top-edges", type=int, default=25, help="top |differential| edges kept per TF")
    ap.add_argument("--max-tfs", type=int, default=None)
    args = ap.parse_args()

    tok = pc.load_token_dict()
    sym = {}
    for r in csv.DictReader(open(_ROOT / "data" / "training" / "ortholog_mappings" / "rat_token_mapping.tsv"),
                            delimiter="\t"):
        sym[r["rat_gene"]] = r.get("rat_symbol", "")
    tfs = [l.split("\t")[0].strip() for i, l in enumerate(open(args.tf_file)) if i > 0 and l.strip()]
    tfs = [g for g in tfs if g in tok]
    if args.max_tfs:
        tfs = tfs[:args.max_tfs]

    (idc, vlc), (idt, vlt) = load_arms(args.dataset, args.meta, args.cell_type)
    nC = 0 if idc is None else idc.shape[0]
    nT = 0 if idt is None else idt.shape[0]
    print(f"[{args.cell_type}] control={nC} trained={nT} cells; candidate TFs={len(tfs)}")
    if nC < args.min_cells or nT < args.min_cells:
        sys.exit(f"too few cells per arm (control={nC}, trained={nT}); need >= {args.min_cells}")
    model = pc.load_model(args.model_dir, args.device)
    # Imported here, not at the top: build_grn_pooled imports load_arms from this module.
    from build_grn_pooled import EDGE_RULE, arm_counts, diff

    edges, summ = [], []
    for gi, g in enumerate(tfs):
        t = tok[g]
        rc = pc.perturb_gene(model, idc, vlc, t, args.device, want_targets=True, batch_size=args.batch_size)
        rt = pc.perturb_gene(model, idt, vlt, t, args.device, want_targets=True, batch_size=args.batch_size)
        if rc.get("n_cells", 0) < args.min_cells or rt.get("n_cells", 0) < args.min_cells:
            continue
        ac, at = rc.get("all_targets", {}), rt.get("all_targets", {})
        # Registration 8.1: the difference exists only for targets both arms report. The earlier line filled
        # a target missing from one arm with 0, which made the "difference" that arm's raw shift.
        dd = diff(ac, at)
        n = arm_counts(ac, at)
        deltas = sorted(((tg, d, ac[tg], at[tg]) for tg, d in dd.items()), key=lambda x: (-abs(x[1]), x[0]))
        for tg, d, c, tr in deltas[:args.top_edges]:
            edges.append((g, sym.get(g, ""), tg, sym.get(tg, ""), round(d, 6),
                          round(c, 6), round(tr, 6), rc["n_cells"], rt["n_cells"]))
        summ.append((g, sym.get(g, ""), rc["n_cells"], rt["n_cells"],
                     round(rc.get("mean_cell_shift", 0), 7), round(rt.get("mean_cell_shift", 0), 7), len(deltas),
                     n["n_targets_control_only"], n["n_targets_trained_only"]))
        if (gi + 1) % 25 == 0:
            print(f"  {gi + 1}/{len(tfs)} TFs processed", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        f.write("tf\ttf_symbol\ttarget\ttarget_symbol\tdelta\tctrl_shift\ttrained_shift\tn_ctrl\tn_trained\n")
        for e in sorted(edges, key=lambda x: (-abs(x[4]), x[0], x[2])):
            f.write("\t".join(str(x) for x in e) + "\n")
    with open(args.out.replace(".tsv", "_tfsummary.tsv"), "w") as f:
        f.write("tf\tsymbol\tn_ctrl\tn_trained\tctrl_cell_shift\ttrained_cell_shift\tn_targets\t"
                "n_targets_control_only\tn_targets_trained_only\n")
        for s in sorted(summ, key=lambda x: (-x[5], x[0])):
            f.write("\t".join(str(x) for x in s) + "\n")
    one_arm = sum(s[7] + s[8] for s in summ)
    print(f"wrote {len(edges)} differential edges from {len(summ)} expressed TFs -> {args.out} "
          f"(edge rule {EDGE_RULE}: {one_arm:,} TF x target pairs reported by one arm only got no edge)")

    # Provenance (audit finding 9.6), with the md5 of the weights file.
    sys.path.insert(0, str(_ROOT / "lib"))
    from provenance import write_provenance
    weights = Path(args.model_dir) / "pytorch_model.bin"
    params = dict(vars(args))
    params.update({"space": "rat", "builder": "per-cell pilot", "edge_rule": EDGE_RULE,
                   "registration": "docs/audit/remediation/STEP3_PREREG.md section 8.1",
                   "weights_file": str(weights), "n_edges": len(edges), "n_tfs": len(summ),
                   "n_pairs_one_arm_only": one_arm})
    write_provenance(Path(args.out).parent, name=f"provenance_{Path(args.out).stem}.json",
                     inputs=[args.dataset, args.meta, args.tf_file, pc.DEFAULT_TOKEN_DICT,
                             Path(args.model_dir) / "config.json"],
                     md5_inputs=[weights, args.tf_file, args.meta], params=params)


if __name__ == "__main__":
    main()
