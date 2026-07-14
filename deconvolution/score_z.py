#!/usr/bin/env python3
"""score_z.py -- Score BayesPrism's per-cell-type expression Z (pred_z, from
extract_z.R) against the known pseudobulk ground truth (true_z, from
compute_true_z.py), for one validation stage (Stage 8).

Two views per cell type, on the common gene set:
  * aggregated GEP -- sum over mixtures of pred vs true expression -> one profile
    per type. Tests whether BayesPrism recovers each type's expression profile.
  * per-sample    -- per mixture, pred vs true gene vector, over mixtures where the
    type is a non-trivial share of the mixture (>= --min-true-frac); report the
    median over those mixtures.
Both compared via Spearman (rank) and Pearson on log1p (counts are heavy-tailed,
so raw Pearson is dominated by a few high-count genes). log-Pearson is scale-
invariant, so pred (bulk-count space) and true (summed sc counts) are comparable.

Usage: python deconvolution/score_z.py --stage-dir data/deconvolution/validation/V2_GSE137869_male
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


# Shared filename contract. resolve() prefers the current name but falls back to the
# legacy one, so sweep artifacts written before the 2026-07-12 sanitizer fix still load.
from celltype_names import resolve  # noqa: E402


def corr(a, b):
    """Spearman + Pearson-on-log1p of two non-negative vectors."""
    if np.std(a) == 0 or np.std(b) == 0:
        return np.nan, np.nan
    sp_ = spearmanr(a, b)[0]
    lp = pearsonr(np.log1p(a), np.log1p(b))[0]
    return sp_, lp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage-dir", required=True)
    ap.add_argument("--min-true-frac", type=float, default=0.02,
                    help="per-sample: skip mixtures where the type is < this fraction "
                         "of the mixture's total true counts")
    args = ap.parse_args()
    sd = Path(args.stage_dir)
    zt = sd / "mixtures" / "true_z"
    zp = sd / "results" / "pred_z"

    tgenes = [l.strip() for l in open(zt / "genes.txt")]
    ttypes = [l.strip() for l in open(zt / "types.txt")]
    pgenes = [l.strip() for l in open(zp / "genes.txt")]

    # common gene set, indexed into each space
    tg_idx = {g: i for i, g in enumerate(tgenes)}
    pg_idx = {g: i for i, g in enumerate(pgenes)}
    common = [g for g in pgenes if g in tg_idx]
    ti = np.array([tg_idx[g] for g in common])
    print(f"{sd.name}: {len(common)} common genes "
          f"(true {len(tgenes)}, pred {len(pgenes)}); types scored: {ttypes}")

    # true per-type Z (n_mix x n_common); mixture totals for the per-sample filter
    true_z = {t: np.load(resolve(zt, t, "truez__", ".npy"))[:, ti] for t in ttypes}
    mix_total = np.sum([true_z[t].sum(1) for t in ttypes], axis=0)  # per mixture

    rows = []
    for t in ttypes:
        true_m = true_z[t]
        pred = pd.read_csv(resolve(zp, t, "predz__", ".csv"), index_col=0)
        pred = pred.reindex(columns=common).fillna(0.0).to_numpy()  # n_mix x n_common

        agg_sp, agg_lp = corr(true_m.sum(0), pred.sum(0))           # aggregated GEP

        ps = []
        tfrac = np.divide(true_m.sum(1), mix_total,
                          out=np.zeros_like(mix_total), where=mix_total > 0)
        for i in np.where(tfrac >= args.min_true_frac)[0]:
            ps.append(corr(true_m[i], pred[i]))
        ps = np.array(ps) if ps else np.zeros((0, 2))

        rows.append({
            "cell_type": t, "n_genes": len(common),
            "agg_spearman": agg_sp, "agg_logpearson": agg_lp,
            "ps_spearman_median": np.nanmedian(ps[:, 0]) if len(ps) else np.nan,
            "ps_logpearson_median": np.nanmedian(ps[:, 1]) if len(ps) else np.nan,
            "n_samples_scored": len(ps),
        })

    m = pd.DataFrame(rows).sort_values("agg_logpearson", ascending=False)
    out = sd / "scores"
    out.mkdir(parents=True, exist_ok=True)
    m.to_csv(out / "metrics_z.tsv", sep="\t", index=False)
    pd.set_option("display.width", 200)
    print(f"\n=== {sd.name}: Z scoring (pred vs true per-cell-type expression) ===")
    print(m.round(3).to_string(index=False))
    print(f"\nWrote {out}/metrics_z.tsv")


if __name__ == "__main__":
    main()
