#!/usr/bin/env python3
"""check_panel_build.py -- checks on a freshly built mixture panel, run BEFORE any set is deconvolved
(docs/audit/remediation/STEP4_PREREG.md V6; STEP4_V5_V6_RESULTS.md note N10).

1. Cross sets. The harmonization of source labels to reference types was reviewed on names alone for the earlier
   panel (MIXTURE_PANEL_REFS4ALL_PREREG.md). A cross set of this panel may be deconvolved only if its
   mixtures/harmonization_map.tsv is the same table as the earlier panel's. If one differs, or has no earlier map
   to compare with, the status is not "ok" and the job scripts refuse to deconvolve cross sets.
2. Every set, descriptive. Whether its mixtures, truth and (holdout) 70% reference are byte-identical to the
   earlier panel's. They should be wherever the production reference did not change, since the builder is seeded.

Writes <panel>/build_check.tsv (one row per set) and <panel>/build_check.status ("ok" or the reason).
Exit status 0 if ok, 3 if not.

  python deconvolution/check_panel_build.py --panel data/deconvolution/mixture_panel_refs5 \
         --previous data/deconvolution/mixture_panel_refs4all
"""
import argparse
import hashlib
import sys
from pathlib import Path

import pandas as pd

FILES = {"mixtures": "mixtures/pseudobulk_counts.mtx", "truth": "mixtures/true_fractions.tsv",
         "mixture_genes": "mixtures/pseudobulk_genes.tsv", "reference": "reference/reference_counts.mtx",
         "reference_cells": "reference/cells_meta.tsv"}


def md5(path, chunk=1 << 24):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def same_file(a, b):
    """'identical', 'different', or 'absent' when either file is missing."""
    if not (a.exists() and b.exists()):
        return "absent"
    return "identical" if md5(a) == md5(b) else "different"


def harmonization(now, prev):
    """Compare two harmonization_map.tsv as tables (source label -> reference type, cells), ignoring row order."""
    if not now.exists():
        return "none"                                   # a holdout set has no map
    if not prev.exists():
        return "no earlier map"
    key = ["cell_type", "ref_type", "n_cells"]
    a = pd.read_csv(now, sep="\t", dtype=str, keep_default_na=False)[key].sort_values(key).reset_index(drop=True)
    b = pd.read_csv(prev, sep="\t", dtype=str, keep_default_na=False)[key].sort_values(key).reset_index(drop=True)
    return "identical" if a.equals(b) else "different"


def check(panel, previous):
    panel, previous = Path(panel), Path(previous)
    sets = sorted([p.name for p in panel.iterdir() if p.is_dir() and p.name.endswith(("_holdout", "_cross"))],
                  key=lambda s: (s.endswith("_cross"), s))
    rows = []
    for s in sets:
        row = {"set": s, "design": "cross" if s.endswith("_cross") else "holdout",
               "harmonization_map": harmonization(panel / s / "mixtures" / "harmonization_map.tsv",
                                                  previous / s / "mixtures" / "harmonization_map.tsv")}
        for name, rel in FILES.items():
            row[f"{name}_vs_previous"] = same_file(panel / s / rel, previous / s / rel)
        rows.append(row)
    t = pd.DataFrame(rows, columns=["set", "design", "harmonization_map"] + [f"{n}_vs_previous" for n in FILES])
    bad = t[(t["design"] == "cross") & (t["harmonization_map"] != "identical")]
    status = "ok" if bad.empty else "cross harmonization not confirmed: " + ", ".join(
        f"{r.set} ({r.harmonization_map})" for r in bad.itertuples())
    return t, status


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--panel", required=True)
    ap.add_argument("--previous", required=True, help="the earlier panel whose cross harmonization was reviewed")
    a = ap.parse_args()
    t, status = check(a.panel, a.previous)
    t.to_csv(Path(a.panel) / "build_check.tsv", sep="\t", index=False)
    (Path(a.panel) / "build_check.status").write_text(status + "\n")
    print(t.to_string(index=False))
    print(f"\nstatus: {status}")
    sys.exit(0 if status == "ok" else 3)


if __name__ == "__main__":
    main()
