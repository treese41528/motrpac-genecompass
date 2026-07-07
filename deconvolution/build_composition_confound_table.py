#!/usr/bin/env python3
"""build_composition_confound_table.py -- DE robustness (Aim 2 hardening).

Formalize the composition-confound pass/fail table over the per-(tissue x cell-type)
blocks. BayesPrism's per-cell-type Z is meant to separate a within-cell EXPRESSION
change from a cell-FRACTION (theta) change, but a cell type whose fraction itself
moves with training can leave residual confound. For every block we place the
within-cell expression response (dose-significant gene count) next to the block's
theta trend over the training weeks (frac_week_slope / frac_week_p from de_summary),
and emit a verdict:

  FLAG_COMPOSITION : theta moves with training (frac_week_p < alpha) AND the block
                     carries a non-trivial expression response -> interpret the
                     expression change as possibly recruitment-driven (down-weight).
  PASS_EXPRESSION  : theta is stable (frac_week_p >= alpha) -> the expression
                     response is not attributable to a fraction shift.
  QUIET            : no meaningful dose-expression response to adjudicate.

Reads:  data/deconvolution/genecompass_input/pseudobulk_de/de_summary.tsv
Writes: .../pseudobulk_de/composition_confound_table.tsv  (all blocks, hotspots first)
Usage:  python deconvolution/build_composition_confound_table.py
"""
import os
from pathlib import Path

import pandas as pd

ROOT = Path(os.environ.get("PIPELINE_ROOT", ".")).resolve()
DE = ROOT / "data/deconvolution/genecompass_input/pseudobulk_de"
ALPHA = 0.05
MIN_EXPR_SIG = 25  # >= this many dose-IHW-significant genes counts as a real response


def verdict(row):
    resp = row["n_sig_dose_IHW"] >= MIN_EXPR_SIG
    theta_moves = pd.notna(row["frac_week_p"]) and row["frac_week_p"] < ALPHA
    if not resp:
        return "QUIET"
    return "FLAG_COMPOSITION" if theta_moves else "PASS_EXPRESSION"


def main():
    df = pd.read_csv(DE / "de_summary.tsv", sep="\t")
    keep = ["tissue", "cell_type", "is_hotspot", "sup_trained_auc", "mean_fraction",
            "n_sig_dose_IHW", "n_sig_interaction", "frac_week_slope", "frac_week_p"]
    out = df[keep].copy()
    for c in ("frac_week_p", "frac_week_slope", "mean_fraction", "sup_trained_auc"):
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out["n_sig_dose_IHW"] = pd.to_numeric(out["n_sig_dose_IHW"], errors="coerce").fillna(0).astype(int)
    out["theta_trend_sig"] = (out["frac_week_p"] < ALPHA).fillna(False)
    out["verdict"] = out.apply(verdict, axis=1)
    # hotspots first, then by verdict severity, then by expression response
    sev = {"FLAG_COMPOSITION": 0, "PASS_EXPRESSION": 1, "QUIET": 2}
    out["_hot"] = out["is_hotspot"].astype(str).str.upper().eq("TRUE")
    out = out.sort_values(["_hot", "verdict", "n_sig_dose_IHW"],
                          key=lambda s: s.map(sev) if s.name == "verdict" else s,
                          ascending=[False, True, False]).drop(columns="_hot")

    outp = DE / "composition_confound_table.tsv"
    out.to_csv(outp, sep="\t", index=False)

    hot = out[out["is_hotspot"].astype(str).str.upper().eq("TRUE")]
    n_flag = int((hot["verdict"] == "FLAG_COMPOSITION").sum())
    print(f"wrote {outp}  ({len(out)} blocks; {len(hot)} hotspots)")
    print(f"hotspots: {n_flag} FLAG_COMPOSITION, "
          f"{int((hot['verdict']=='PASS_EXPRESSION').sum())} PASS_EXPRESSION, "
          f"{int((hot['verdict']=='QUIET').sum())} QUIET")
    print("\n== hotspot composition-confound verdicts ==")
    with pd.option_context("display.width", 160, "display.max_rows", 40,
                           "display.max_colwidth", 24):
        print(hot[["tissue", "cell_type", "mean_fraction", "n_sig_dose_IHW",
                   "frac_week_slope", "frac_week_p", "verdict"]].to_string(index=False))


if __name__ == "__main__":
    main()
