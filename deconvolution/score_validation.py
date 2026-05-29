#!/usr/bin/env python3
"""
score_validation.py -- Score a pseudobulk validation stage (Stage 8 V0/V1):
compare BayesPrism's estimated cell-type fractions against the known ground truth.

BayesPrism's theta estimates the RNA/count fraction, so the primary target is
rnafrac__ (cellfrac__ is reported as a secondary reference). Per cell type we
report Pearson r, Spearman rho, RMSE; plus the overall pooled fit.

Inputs (under --stage-dir):
  mixtures/true_fractions.tsv        truth (cellfrac__/rnafrac__ columns)
  results/estimated_fractions.csv    BayesPrism output (mixtures x cell types)
Output:
  scores/metrics.tsv, scores/merged_long.tsv, and a printed summary.

Usage:  python deconvolution/score_validation.py --stage-dir deconvolution/validation/V0_liver
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage-dir", required=True)
    ap.add_argument("--truth", default="rnafrac", choices=["rnafrac", "cellfrac"],
                    help="ground-truth basis to score against (default rnafrac = BayesPrism's theta)")
    args = ap.parse_args()
    sd = Path(args.stage_dir)

    truth = pd.read_csv(sd / "mixtures" / "true_fractions.tsv", sep="\t")
    est = pd.read_csv(sd / "results" / "estimated_fractions.csv", index_col=0)
    est.index = [f"mix{i+1}" for i in range(len(est))]      # align to truth order

    pref = args.truth + "__"
    true_cols = {c[len(pref):]: c for c in truth.columns if c.startswith(pref)}
    types = [t for t in true_cols if t in est.columns]
    missing = [t for t in true_cols if t not in est.columns]
    if missing:
        print(f"[note] types in truth but not in estimates (set to 0): {missing}")

    rows, long = [], []
    for t in types:
        y = truth[true_cols[t]].to_numpy()
        yhat = est[t].to_numpy()
        n = min(len(y), len(yhat)); y, yhat = y[:n], yhat[:n]
        r = pearsonr(y, yhat)[0] if np.std(yhat) > 0 else np.nan
        rho = spearmanr(y, yhat)[0] if np.std(yhat) > 0 else np.nan
        rmse = float(np.sqrt(np.mean((y - yhat) ** 2)))
        bias = float(np.mean(yhat - y))
        rows.append({"cell_type": t, "n": n, "pearson_r": r, "spearman_rho": rho,
                     "rmse": rmse, "mean_bias": bias,
                     "true_mean": float(y.mean()), "est_mean": float(yhat.mean())})
        for a, b in zip(y, yhat):
            long.append({"cell_type": t, "true": a, "est": b})

    metrics = pd.DataFrame(rows).sort_values("true_mean", ascending=False)
    # overall pooled fit across all type x mixture points
    L = pd.DataFrame(long)
    overall_r = pearsonr(L["true"], L["est"])[0]
    overall_rmse = float(np.sqrt(np.mean((L["true"] - L["est"]) ** 2)))

    out = sd / "scores"; out.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(out / "metrics.tsv", sep="\t", index=False)
    L.to_csv(out / "merged_long.tsv", sep="\t", index=False)

    pd.set_option("display.float_format", lambda v: f"{v:.4f}")
    print(f"\n=== {sd.name}: estimated vs {args.truth} (n={est.shape[0]} mixtures) ===")
    print(metrics.to_string(index=False))
    print(f"\nOVERALL  pearson_r={overall_r:.4f}  rmse={overall_rmse:.4f}")
    print(f"Wrote {out}/metrics.tsv")


if __name__ == "__main__":
    main()
