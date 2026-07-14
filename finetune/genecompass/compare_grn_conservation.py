#!/usr/bin/env python
# coding: utf-8
r"""compare_grn_conservation.py -- Aim 3b step 2: rank-correlate each TF's rat vs human-space
target list to score whether the regulatory logic (not just markers) is conserved.

Inputs (matched by outname):
  rat   : data/deconvolution/grn/<outname>_pooled.tsv        (tf, target[ENSRNOG], delta, z, ...)
  human : data/deconvolution/grn_human/<outname>_pooled.tsv  (tf, target_human_ensg, delta, z, ...)
Rat targets are projected to human ENSG (Stage-3 ortholog map); per TF we Spearman-correlate the
differential target-shift (delta) over the shared human-ENSG targets. A TF is "conserved" if the
paired lists agree beyond chance. Emits per-(cell type x TF) scores + a per-cell-type summary and
the pre-registered anchor-edge check (Mef2c->myogenic, Spi1/Cebpa->myeloid).

Usage:
  python finetune/genecompass/compare_grn_conservation.py \
    --rat-dir data/deconvolution/grn --human-dir data/deconvolution/grn_human \
    --out-dir data/deconvolution/grn_human/conservation
"""
import argparse
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

_ROOT = Path(os.environ.get("PIPELINE_ROOT", ".")).resolve()
ANCHORS = {"Mef2c": "myogenic", "Myf6": "myogenic", "Spi1": "myeloid", "Cebpa": "myeloid"}


def _base(x):
    return str(x).split(".")[0].upper()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rat-dir", default=str(_ROOT / "data/deconvolution/grn"))
    ap.add_argument("--human-dir", default=str(_ROOT / "data/deconvolution/grn_human"))
    ap.add_argument("--out-dir", default=str(_ROOT / "data/deconvolution/grn_human/conservation"))
    ap.add_argument("--ortholog-map", default=str(_ROOT / "data/training/ortholog_mappings/rat_to_human_mapping.pickle"))
    ap.add_argument("--min-shared", type=int, default=8, help="min shared targets to score a TF")
    ap.add_argument("--top-k", type=int, default=0,
                    help="per TF, restrict to its top-k RAT targets by |delta| before intersecting with "
                         "human. 0 = use every shared target (default, and the honest setting).\n"
                         "WHY THIS MATTERS: when the input GRNs were capped at 40 edges/TF, `shared` was "
                         "the intersection of the rat's top-40 and the human's top-40 -- i.e. an edge only "
                         "entered the correlation if it was already strong in BOTH species. That is "
                         "selection for the very agreement the Spearman is meant to measure, and it biases "
                         "conservation upward (it is why n_shared was ~11 and 97%% of rhos were positive). "
                         "With uncapped inputs and --top-k 0 the score is unbiased. Set --top-k 40 only to "
                         "reproduce the old, biased number for comparison.")
    ap.add_argument("--select", default="conf", choices=["conf", "all", "top40", "top200"],
                    help="Which RAT edges enter the correlation. The HUMAN list is NEVER pre-selected, which "
                         "is what makes this unbiased -- selecting on the rat side is the scientific question "
                         "('the edges exercise remodels in rat -- do they agree in human?'), whereas selecting "
                         "on BOTH sides (the old --top-k path) selects for the very agreement being measured.\n"
                         "  conf   (DEFAULT) rat edges with |z|>=2. Unbiased AND undiluted. median rho ~0.45.\n"
                         "  all              every shared target (~1,950). Unbiased but DILUTED by ~1,900 "
                         "near-zero noise targets -- drags median rho to ~0.15.\n"
                         "  top40/top200     rat's top-N by |delta|.\n"
                         "Ignored when --top-k > 0 (that path reproduces the old biased score).")
    ap.add_argument("--out-prefix", default="", help="prefix for output filenames (e.g. 'topk40_')")
    args = ap.parse_args()
    outd = Path(args.out_dir); outd.mkdir(parents=True, exist_ok=True)
    r2h = {_base(k): _base(v) for k, v in pickle.load(open(args.ortholog_map, "rb")).items()}

    rat_dir, hum_dir = Path(args.rat_dir), Path(args.human_dir)
    pairs = []
    for hf in sorted(hum_dir.glob("*_pooled.tsv")):
        rf = rat_dir / hf.name
        if rf.exists():
            pairs.append((hf.name.replace("_pooled.tsv", ""), rf, hf))
    print(f"matched {len(pairs)} cell types with both rat and human GRNs")

    per_tf, per_ct = [], []
    for ct, rf, hf in pairs:
        rat = pd.read_csv(rf, sep="\t"); hum = pd.read_csv(hf, sep="\t")
        rat["t_h"] = rat["target"].map(lambda x: r2h.get(_base(x)))          # rat target -> human ENSG
        rat = rat.dropna(subset=["t_h"])
        hum["t_h"] = hum["target_human_ensg"].map(_base)
        tf_scores = []
        for tf, rg in rat.groupby("tf"):
            hg = hum[hum["tf"] == tf]
            if hg.empty:
                continue
            if args.top_k and args.top_k > 0:
                # OLD, BIASED path: pre-selects BOTH sides, so an edge only enters the correlation if it is
                # already strong in rat AND in human -- selection for the agreement being measured.
                rg = rg.reindex(rg["delta"].abs().sort_values(ascending=False).index).head(args.top_k)
                hg = hg.reindex(hg["delta"].abs().sort_values(ascending=False).index).head(args.top_k)
            else:
                # UNBIASED: select on the RAT side only; the human list is left whole.
                if args.select == "conf":
                    sel = rg[rg["z"].abs() >= 2]
                    if len(sel) >= args.min_shared:
                        rg = sel
                elif args.select == "top40":
                    rg = rg.reindex(rg["delta"].abs().sort_values(ascending=False).index).head(40)
                elif args.select == "top200":
                    rg = rg.reindex(rg["delta"].abs().sort_values(ascending=False).index).head(200)
                # args.select == "all" -> leave rg whole
            rmap = rg.groupby("t_h")["delta"].mean()
            hmap = hg.groupby("t_h")["delta"].mean()
            shared = rmap.index.intersection(hmap.index)
            if len(shared) < args.min_shared:
                continue
            rho, p = spearmanr(rmap.loc[shared].values, hmap.loc[shared].values)
            if not np.isfinite(rho):
                continue
            sym = rg["tf_symbol"].iloc[0]
            row = {"cell_type": ct, "tf": tf, "tf_symbol": sym, "n_shared": int(len(shared)),
                   "spearman": round(float(rho), 4), "p": float(p),
                   "anchor": ANCHORS.get(sym, "")}
            tf_scores.append(row); per_tf.append(row)
        if tf_scores:
            rhos = [r["spearman"] for r in tf_scores]
            per_ct.append({"cell_type": ct, "n_tf_scored": len(tf_scores),
                           "median_spearman": round(float(np.median(rhos)), 4),
                           "frac_pos": round(float(np.mean([r > 0 for r in rhos])), 3),
                           "frac_conserved_rho>=0.3": round(float(np.mean([r >= 0.3 for r in rhos])), 3)})
            print(f"  {ct:32} TFs={len(tf_scores):4}  median rho={np.median(rhos):.3f}  "
                  f"frac rho>=0.3={np.mean([r>=0.3 for r in rhos]):.2f}")

    px = args.out_prefix
    pd.DataFrame(per_tf).to_csv(outd / f"{px}conservation_per_tf.tsv", sep="\t", index=False)
    pd.DataFrame(per_ct).to_csv(outd / f"{px}conservation_per_celltype.tsv", sep="\t", index=False)
    anchors = pd.DataFrame([r for r in per_tf if r["anchor"]])
    if not anchors.empty:
        anchors.to_csv(outd / f"{px}conservation_anchors.tsv", sep="\t", index=False)
        print("\n== pre-registered anchor edges ==")
        print(anchors[["cell_type", "tf_symbol", "anchor", "n_shared", "spearman"]].to_string(index=False))
    print(f"\nwrote {outd}/{px}conservation_per_tf.tsv, {px}conservation_per_celltype.tsv")


if __name__ == "__main__":
    main()
