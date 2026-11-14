#!/usr/bin/env python3
"""step6_stage13_names_check.py -- Step 6 (records, tests, hygiene).

pipeline/run_stage13.py names each network file with its own sanitiser, _safe(tissue, cell_type). The
project rule is to import safe() from deconvolution/celltype_names.py. This script asks whether the switch
would change a file name: for every cell-type name in the current references
(data/deconvolution/references_v5/*/cells_meta.tsv, column cell_type) it compares the name Stage 13 writes
today with the two candidate replacements built on the shared safe().

  python analysis/audit_remediation/step6_stage13_names_check.py
Out: docs/audit/remediation/record/step6/stage13_network_names.tsv (one row per reference x cell type) and a
printed summary.
"""
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "deconvolution"))
sys.path.insert(0, str(ROOT / "pipeline"))
from celltype_names import safe          # noqa: E402
import run_stage13                       # noqa: E402

REFS = ROOT / "data/deconvolution/references_v5"
OUT = ROOT / "docs/audit/remediation/record/step6/stage13_network_names.tsv"


def main():
    rows = []
    for meta in sorted(REFS.glob("*/cells_meta.tsv")):
        ref = meta.parent.name
        tissue = ref.split("_")[0].lower()
        with open(meta, newline="") as fh:
            types = sorted({r["cell_type"] for r in csv.DictReader(fh, delimiter="\t")})
        for ct in types:
            now = run_stage13._safe(tissue, ct)
            shared = f"{tissue}_{safe(ct)}"
            shared_lower = f"{tissue}_{safe(ct).lower()}"
            rows.append({"reference": ref, "cell_type": ct, "stage13_name_today": now,
                         "with_shared_safe": shared, "with_shared_safe_lowercased": shared_lower,
                         "identical": now == shared, "identical_lowercased": now == shared_lower})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    n = len(rows)
    same, same_l = sum(r["identical"] for r in rows), sum(r["identical_lowercased"] for r in rows)
    print(f"{n} reference x cell-type names in {len({r['reference'] for r in rows})} references")
    print(f"identical to safe():           {same} of {n}")
    print(f"identical to safe().lower():   {same_l} of {n}")
    for r in rows:
        if not r["identical_lowercased"]:
            print(f"  differs: {r['reference']:<32} {r['cell_type']!r:<45} {r['stage13_name_today']}  |  "
                  f"{r['with_shared_safe_lowercased']}")
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
