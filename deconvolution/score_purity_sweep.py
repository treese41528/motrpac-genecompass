#!/usr/bin/env python3
"""score_purity_sweep.py -- Aggregate the focal-type expression accuracy into the
paper's headline view: correlation AS A FUNCTION OF focal-type fraction, binned
to compare directly against BayesPrism Fig. 1h (">0.95 for tumors with >50%
purity", Pearson on DESeq2-VST).

Reads <stage_dir>/scores/z_vst_focal.tsv (per-sample pearson_vst / spearman_raw +
focal_rnafrac, from score_z_vst.R) and reports the median (+IQR) per purity point
and, crucially, the pooled value for the >=0.5-purity samples -- the number that
lines up 1-1 with the paper's >50% claim.

Usage: python deconvolution/score_purity_sweep.py \
    --stage-dir deconvolution/validation/SWEEP_hepato_cross
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage-dir", required=True)
    ap.add_argument("--purity-col", default="focal_rnafrac",
                    choices=["focal_rnafrac", "target_purity"],
                    help="which fraction to bin by (rnafrac = realized RNA "
                         "fraction, the paper's x-axis; target_purity = nominal)")
    args = ap.parse_args()
    sd = Path(args.stage_dir)
    df = pd.read_csv(sd / "scores" / "z_vst_focal.tsv", sep="\t")

    g = df.groupby("target_purity")
    per_point = g.agg(
        n=("pearson_vst", "size"),
        pearson_vst_median=("pearson_vst", "median"),
        pearson_vst_q1=("pearson_vst", lambda s: s.quantile(0.25)),
        pearson_vst_q3=("pearson_vst", lambda s: s.quantile(0.75)),
        spearman_raw_median=("spearman_raw", "median"),
        mean_focal_rnafrac=("focal_rnafrac", "mean"),
    ).reset_index()

    out = sd / "scores"
    per_point.to_csv(out / "purity_sweep_summary.tsv", sep="\t", index=False)

    # the paper-comparable number: samples at >=50% focal fraction
    hi = df[df[args.purity_col] >= 0.5]
    lo = df[df[args.purity_col] < 0.5]

    pd.set_option("display.width", 200)
    print(f"=== {sd.name}: focal-expression accuracy vs purity ===")
    print(per_point.round(3).to_string(index=False))
    print(f"\n--- paper-comparable (binned by {args.purity_col}) ---")
    for name, sub in [(">=50% purity (Fig 1h regime)", hi), ("<50% purity", lo)]:
        if len(sub):
            print(f"  {name:32s} n={len(sub):3d}  "
                  f"median Pearson-VST={sub['pearson_vst'].median():.3f}  "
                  f"median Spearman={sub['spearman_raw'].median():.3f}")
        else:
            print(f"  {name:32s} n=0  (sweep grid did not reach this bin)")
    print(f"\nPaper (Chu 2022 Fig 1h): malignant expression Pearson-VST >0.95 at >50% purity.")
    print(f"Wrote {out}/purity_sweep_summary.tsv")


if __name__ == "__main__":
    main()
