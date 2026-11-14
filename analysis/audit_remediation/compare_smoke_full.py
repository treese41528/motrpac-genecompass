#!/usr/bin/env python
"""compare_smoke_full.py -- do the smoke runs and the full runs of TESTES give the same numbers?

The smoke run of V5 and V6 (docs/audit/remediation/STEP4_V5_V6_DRAFT.md, note N13) ran TESTES with the registered
settings into folders of its own; the full batteries ran it again. For every estimate file of the four TESTES sets
this compares the two runs: byte-identical or not, and the largest absolute difference in any estimated fraction.
It is a run-to-run reproducibility check on one reference, which the audit noted had never been made for
CIBERSORTx (its container draws samples without a seed).

  python analysis/audit_remediation/compare_smoke_full.py
         [--out docs/audit/remediation/record/step4/smoke/smoke_vs_full_TESTES.tsv]
"""
import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PAIRS = [("V5", f"TESTES/{a}", "data/deconvolution/simbu_bench_refs5_smoke", "data/deconvolution/simbu_bench_refs5") for a in "ABC"]
PAIRS += [("V6", "TESTES_holdout", "data/deconvolution/mixture_panel_refs5_smoke", "data/deconvolution/mixture_panel_refs5")]
FILES = {"BayesPrism": "results/estimated_fractions.csv", "DWLS": "results/fractions_dwls.csv",
         "MuSiC": "results/fractions_music.csv", "SCDC": "results/fractions_scdc.csv",
         "Bisque": "results/fractions_bisque.csv", "CIBERSORTx": "results/fractions_cibersortx.csv",
         "mixtures": "mixtures/pseudobulk_counts.mtx", "truth": "mixtures/true_fractions.tsv"}


def md5(p):
    return hashlib.md5(p.read_bytes()).hexdigest()


def compare(a, b):
    if not (a.exists() and b.exists()):
        return {"status": "absent in " + ("both" if not (a.exists() or b.exists()) else "smoke" if not a.exists() else "full")}
    if md5(a) == md5(b):
        return {"status": "byte-identical", "max_abs_difference": 0.0}
    if a.suffix != ".csv":
        return {"status": "different"}
    x, y = pd.read_csv(a, index_col=0), pd.read_csv(b, index_col=0)
    if x.shape != y.shape or list(x.columns) != list(y.columns):
        return {"status": "different shape or cell types"}
    d = np.abs(x.to_numpy() - y.to_numpy())
    return {"status": "different", "max_abs_difference": float(d.max()), "mean_abs_difference": float(d.mean())}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=str(ROOT / "docs/audit/remediation/record/step4/smoke/smoke_vs_full_TESTES.tsv"))
    a = ap.parse_args()
    rows = []
    for battery, s, smoke, full in PAIRS:
        for name, rel in FILES.items():
            rows.append({"battery": battery, "set": s, "file": name, **compare(ROOT / smoke / s / rel, ROOT / full / s / rel)})
    t = pd.DataFrame(rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    t.to_csv(a.out, sep="\t", index=False)
    print(t.to_string(index=False))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
