#!/usr/bin/env python
"""calibrate_target_sum.py -- the Stage 9 target sum, by the registered rule.

Rule (docs/RETRAIN_V2_PREREGISTRATION.md:155-164; applied to refs5 by docs/audit/remediation/STEP3_PREREG.md
section 3): one target sum T for every tissue, at which LIVER's pseudo-cells have the corpus's median token value.
Token selection does not depend on T and a token's value is log2(1 + T r), so from LIVER tokenized at any T0,

    T* = T0 * (2^m_corpus - 1) / (2^m_liver(T0) - 1),      T = T* rounded to the nearest 100.

m_corpus is the v2 training corpus's median real-token value, 0.9145306348800659 (every 211th row of
data/training/tokenized_corpus_v2/dataset; data/results/retrain_v2/c5/stage9_calibration/calibration.json).
m_liver(T0) is value_stats.median of the LIVER tokenize_summary.json written by tokenize_pseudocells.py at T0.

This script only reads a tokenize_summary.json and prints T; it changes no file but its own JSON record.

  python deconvolution/calibrate_target_sum.py --tokenize-summary <dir>/tokenize_summary.json --out <record.json>
"""
import argparse
import json
from pathlib import Path

V2_CORPUS_MEDIAN = 0.9145306348800659


def target_sum(m_liver, t0, m_corpus=V2_CORPUS_MEDIAN):
    """T* and T (T* rounded to the nearest 100) by the registered formula."""
    t_star = t0 * (2.0 ** m_corpus - 1.0) / (2.0 ** m_liver - 1.0)
    return t_star, float(round(t_star / 100.0) * 100)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokenize-summary", required=True, help="LIVER tokenize_summary.json, tokenized at T0")
    ap.add_argument("--corpus-median", type=float, default=V2_CORPUS_MEDIAN)
    ap.add_argument("--out", help="write a JSON record here")
    a = ap.parse_args()
    s = json.loads(Path(a.tokenize_summary).read_text())
    t0, m = float(s["target_sum"]), float(s["value_stats"]["median"])
    t_star, t = target_sum(m, t0, a.corpus_median)
    rec = {"tokenize_summary": str(a.tokenize_summary), "T0": t0, "liver_median_at_T0": m,
           "corpus_median": a.corpus_median, "T_star": t_star, "T": t,
           "liver_median_at_T_predicted": __import__("math").log2(1 + (2 ** m - 1) * t / t0),
           "n_pseudocells": s.get("n_pseudocells")}
    print(json.dumps(rec, indent=1))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(rec, indent=1) + "\n")


if __name__ == "__main__":
    main()
