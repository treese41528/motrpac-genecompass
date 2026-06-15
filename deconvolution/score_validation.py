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

Usage:  python deconvolution/score_validation.py --stage-dir data/deconvolution/validation/V0_liver
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
    ap.add_argument("--drop-dominant", type=int, default=1,
                    help="for the 'separable-compartment' pooled fit, exclude this many "
                    "highest-true_mean types (the dominant parenchyma, which in cross-dataset "
                    "tests collapses or becomes a sink and swamps the pooled overall); default 1")
    ap.add_argument("--est-file", default=None,
                    help="estimates CSV (mixtures x cell types), relative to --stage-dir or "
                    "absolute; default results/estimated_fractions.csv (BayesPrism). Pass e.g. "
                    "results/fractions_music.csv to score an omnideconv method.")
    ap.add_argument("--tag", default="",
                    help="suffix for the scores/ output files (metrics<_tag>.tsv etc.) so "
                    "methods don't overwrite each other; default empty keeps the original names.")
    args = ap.parse_args()
    sd = Path(args.stage_dir)

    truth = pd.read_csv(sd / "mixtures" / "true_fractions.tsv", sep="\t")
    est_path = Path(args.est_file) if args.est_file else sd / "results" / "estimated_fractions.csv"
    if not est_path.is_absolute():
        est_path = sd / est_path
    est = pd.read_csv(est_path, index_col=0)
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

    metrics = pd.DataFrame(rows).sort_values("true_mean", ascending=False).reset_index(drop=True)
    # overall pooled fit across all type x mixture points
    L = pd.DataFrame(long)
    overall_r = pearsonr(L["true"], L["est"])[0]
    overall_rmse = float(np.sqrt(np.mean((L["true"] - L["est"]) ** 2)))

    # --- separable-compartment summary -------------------------------------
    # The pooled overall is dominated by the single most-abundant parenchymal
    # type, which in cross-dataset tests either collapses to ~0 or becomes a
    # sink -- swamping the (often excellent) recovery of every other type. We
    # therefore also report (a) macro per-type r (each type weighted equally)
    # and (b) the pooled fit with the top --drop-dominant type(s) excluded.
    valid_r = metrics["pearson_r"].dropna()
    macro_r = float(valid_r.mean()) if len(valid_r) else float("nan")
    median_r = float(valid_r.median()) if len(valid_r) else float("nan")
    k = max(0, min(args.drop_dominant, len(metrics) - 1))
    dropped_types = list(metrics["cell_type"].iloc[:k]) if k else []
    Ls = L[~L["cell_type"].isin(dropped_types)]
    sep_r = (pearsonr(Ls["true"], Ls["est"])[0]
             if len(Ls) and np.std(Ls["est"]) > 0 else float("nan"))
    sep_rmse = float(np.sqrt(np.mean((Ls["true"] - Ls["est"]) ** 2))) if len(Ls) else float("nan")

    out = sd / "scores"; out.mkdir(parents=True, exist_ok=True)
    sfx = f"_{args.tag}" if args.tag else ""
    metrics.to_csv(out / f"metrics{sfx}.tsv", sep="\t", index=False)
    L.to_csv(out / f"merged_long{sfx}.tsv", sep="\t", index=False)
    overall = pd.DataFrame([
        {"metric": "overall_pooled_pearson", "value": overall_r, "note": "all types pooled"},
        {"metric": "overall_pooled_rmse", "value": overall_rmse, "note": "all types pooled"},
        {"metric": "macro_pearson", "value": macro_r, "note": "mean of per-type r (equal weight)"},
        {"metric": "median_pearson", "value": median_r, "note": "median of per-type r"},
        {"metric": "separable_pooled_pearson", "value": sep_r,
         "note": f"pooled, excl {dropped_types or 'none'}"},
        {"metric": "separable_pooled_rmse", "value": sep_rmse,
         "note": f"pooled, excl {dropped_types or 'none'}"},
    ])
    overall.to_csv(out / f"overall_metrics{sfx}.tsv", sep="\t", index=False)

    pd.set_option("display.float_format", lambda v: f"{v:.4f}")
    print(f"\n=== {sd.name}: {est_path.name} vs {args.truth} (n={est.shape[0]} mixtures) ===")
    print(metrics.to_string(index=False))
    print(f"\nOVERALL (all types pooled)   pearson_r={overall_r:.4f}  rmse={overall_rmse:.4f}")
    print(f"PER-TYPE r  macro(mean)={macro_r:.4f}  median={median_r:.4f}  (n_types={len(valid_r)})")
    print(f"SEPARABLE compartment        pearson_r={sep_r:.4f}  rmse={sep_rmse:.4f}"
          f"   [excluded: {dropped_types or 'none'}]")
    print(f"Wrote {out}/metrics{sfx}.tsv + overall_metrics{sfx}.tsv")


if __name__ == "__main__":
    main()
