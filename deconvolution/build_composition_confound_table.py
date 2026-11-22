#!/usr/bin/env python3
"""build_composition_confound_table.py -- DE robustness (Aim 2 hardening).

One status per (tissue x cell-type) block on whether its expression response could be a change in the cell
type's FRACTION (theta) instead of a change within the cell type. BayesPrism's per-cell-type Z is meant to
separate the two, but a cell type whose fraction itself moves with training can leave a residual confound.
For every block the dose-significant gene count is placed next to the block's theta trend over the training
weeks (frac_week_slope / frac_week_p from de_summary.tsv, a linear regression of the fraction on week).

Statuses (column `verdict`), assigned in this order:

  NOT_ANALYSED      the DE did not analyse the block (de_summary status is not "ok", e.g. a median library
                    under the analysable-block rule). There is no expression result to judge.
  NO_DOSE_TEST      the block was analysed but has no dose p (before STEP3_PREREG.md 6.1 this was every
                    one-sex block: the Fisher combination needed two sexes).
  QUIET             the dose test ran and fewer than MIN_EXPR_SIG genes are dose-significant: no response
                    to adjudicate.
  THETA_NOT_TESTED  the block has a response but no theta-trend p, so the confound was not tested.
  FLAG_COMPOSITION  the block has a response and theta moves with training (frac_week_p < alpha): read the
                    expression change as possibly driven by composition.
  PASS_EXPRESSION   the block has a response and the theta trend is not significant at alpha.

QUIET counted the first two groups as well until 2026-11-14 (audit finding 7.6: 46 unanalysed and 21 one-sex
blocks among 112 "QUIET" on the 16-tissue default). It now counts only blocks that were tested and quiet.

Reads:  <genecompass_input_dir>/pseudobulk_de/de_summary.tsv  (config; or --de-dir)
Writes: .../pseudobulk_de/composition_confound_table.tsv  (all blocks, hotspots first)
Usage:  python deconvolution/build_composition_confound_table.py [--de-dir DIR]
"""
import argparse
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(os.environ.setdefault("PIPELINE_ROOT", str(Path(__file__).resolve().parents[1]))) / "lib"))
from gene_utils import load_config, resolve_path  # noqa: E402

ALPHA = 0.05
MIN_EXPR_SIG = 25  # >= this many dose-IHW-significant genes counts as a real response
# Order of the statuses in the table (most to least in need of reading) and in the printed counts.
STATUSES = ["FLAG_COMPOSITION", "PASS_EXPRESSION", "THETA_NOT_TESTED", "QUIET", "NO_DOSE_TEST", "NOT_ANALYSED"]


def verdict(row):
    if not row["analysed"]:
        return "NOT_ANALYSED"
    if not row["dose_tested"]:
        return "NO_DOSE_TEST"
    if row["n_sig_dose_IHW"] < MIN_EXPR_SIG:
        return "QUIET"
    if pd.isna(row["frac_week_p"]):
        return "THETA_NOT_TESTED"
    return "FLAG_COMPOSITION" if row["frac_week_p"] < ALPHA else "PASS_EXPRESSION"


def build_table(df):
    """de_summary.tsv (one row per block) -> the composition-confound table, hotspots first."""
    keep = ["tissue", "cell_type", "is_hotspot", "sup_trained_auc", "mean_fraction",
            "n_sig_dose_IHW", "n_sig_interaction", "frac_week_slope", "frac_week_p"]
    if "gate_class" in df.columns:          # hotspot / borderline / none, copied from the gate table by Stage 10
        keep.insert(3, "gate_class")
    out = df[keep].copy()
    for c in ("frac_week_p", "frac_week_slope", "mean_fraction", "sup_trained_auc"):
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out["analysed"] = df["status"].astype(str).eq("ok")
    # Whether the block's dose test ran. A current de_summary.tsv counts the block's dose p in n_dose_tests.
    # An earlier one has no such column; there the test ran exactly in the analysed blocks with two sexes.
    if "n_dose_tests" in df.columns:
        out["dose_tested"] = out["analysed"] & (pd.to_numeric(df["n_dose_tests"], errors="coerce").fillna(0) > 0)
    else:
        two_sex = (pd.to_numeric(df["n_male"], errors="coerce").fillna(0) > 0) & \
                  (pd.to_numeric(df["n_female"], errors="coerce").fillna(0) > 0)
        out["dose_tested"] = out["analysed"] & two_sex
    # The count is kept as written (missing where the block was not analysed); it is not set to 0.
    out["n_sig_dose_IHW"] = pd.to_numeric(out["n_sig_dose_IHW"], errors="coerce").astype("Int64")
    out["theta_trend_sig"] = (out["frac_week_p"] < ALPHA).where(out["frac_week_p"].notna())
    out["verdict"] = out.apply(verdict, axis=1)
    # hotspots first, then by status, then by expression response
    sev = {s: i for i, s in enumerate(STATUSES)}
    out["_hot"] = out["is_hotspot"].astype(str).str.upper().eq("TRUE")
    out["_sev"] = out["verdict"].map(sev)
    out["_n"] = out["n_sig_dose_IHW"].fillna(-1)
    out = out.sort_values(["_hot", "_sev", "_n"], ascending=[False, True, False], kind="stable")
    return out.drop(columns=["_hot", "_sev", "_n", "analysed", "dose_tested"]).reset_index(drop=True)


def status_counts(table):
    """'N STATUS' for every status that occurs, in the order of STATUSES."""
    n = table["verdict"].value_counts()
    return ", ".join(f"{int(n[s])} {s}" for s in STATUSES if s in n.index) or "none"


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser(description="Composition-confound verdict per DE block")
    ap.add_argument("--de-dir", default=str(resolve_path(cfg, cfg["deconvolution"]["genecompass_input_dir"]) / "pseudobulk_de"))
    DE = Path(ap.parse_args().de_dir)
    out = build_table(pd.read_csv(DE / "de_summary.tsv", sep="\t"))

    outp = DE / "composition_confound_table.tsv"
    out.to_csv(outp, sep="\t", index=False)

    hot = out[out["is_hotspot"].astype(str).str.upper().eq("TRUE")]
    print(f"wrote {outp}  ({len(out)} blocks; {len(hot)} hotspots)")
    print(f"all blocks: {status_counts(out)}")
    print(f"hotspots: {status_counts(hot)}")
    print("\n== hotspot composition-confound verdicts ==")
    with pd.option_context("display.width", 160, "display.max_rows", 40,
                           "display.max_colwidth", 24):
        print(hot[["tissue", "cell_type", "mean_fraction", "n_sig_dose_IHW",
                   "frac_week_slope", "frac_week_p", "verdict"]].to_string(index=False))


if __name__ == "__main__":
    main()
