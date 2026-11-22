#!/usr/bin/env python3
"""step6_purity_sweep_cellfrac_scan.py -- Step 6 (records, tests, hygiene). Found while writing the unit
tests of deconvolution/make_purity_sweep.py; not an audit finding.

generate_purity_sweep() splits the background cells of a mixture by a Dirichlet draw, rounds each share to a
whole number of cells, and repairs the rounding drift by adding it to ONE background type chosen at random.
When the drift is negative and the chosen type has fewer cells than the drift, its count becomes negative
(-1; in one mixture on disk -2). That type is then skipped when cells are drawn, so the mixture holds one or
two cells more than --cells-per-mixture, and true_fractions.tsv records a negative cell fraction for it.

What this does and does not touch. The mixture's counts, its per-type true expression and the RNA fractions
(rnafrac__*, focal_rnafrac) are computed from the cells actually drawn, so they are exact, and the purity-sweep
score (Pearson-VST against focal_rnafrac) is not affected. The cell-fraction columns (cellfrac__*,
focal_cellfrac) of an affected mixture are wrong by one or two cells.

This script only counts: it reads mixtures/true_fractions.tsv of every sweep under the Stage 14 validation
folder of the default configuration and under the research-phase validation folder.

  python analysis/audit_remediation/step6_purity_sweep_cellfrac_scan.py
Out: docs/audit/remediation/record/step6/purity_sweep_negative_cellfrac.tsv
"""
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "lib"))
from gene_utils import load_config, resolve_path   # noqa: E402

OUT = ROOT / "docs/audit/remediation/record/step6/purity_sweep_negative_cellfrac.tsv"


def main():
    cfg = load_config(); dc = cfg["deconvolution"]
    roots = {"deconvolution.validation_dir": resolve_path(cfg, dc["validation_dir"]),
             "deconvolution.pilot_validation_dir": resolve_path(cfg, dc["pilot_validation_dir"])}
    rows = []
    for key, root in roots.items():
        for f in sorted(root.glob("SWEEP_*/mixtures/true_fractions.tsv")):
            with open(f, newline="") as fh:
                rd = list(csv.DictReader(fh, delimiter="\t"))
            cols = [c for c in rd[0] if c.startswith("cellfrac__")]
            neg = [(r["mixture"], c[len("cellfrac__"):], float(r[c])) for r in rd for c in cols if float(r[c]) < 0]
            sums = [sum(float(r[c]) for c in cols) for r in rd]
            rows.append({"config_key": key, "folder": root.name, "sweep": f.parents[1].name,
                         "mixtures": len(rd), "cell_types": len(cols),
                         "mixtures_with_a_negative_cell_fraction": len({m for m, _, _ in neg}),
                         "negative_entries": len(neg),
                         "most_negative": min((v for _, _, v in neg), default=""),
                         "cell_fractions_sum_min": round(min(sums), 6), "cell_fractions_sum_max": round(max(sums), 6),
                         "which": "; ".join(f"{m} {t}" for m, t, _ in neg)})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        w.writeheader(); w.writerows(rows)
    for key in roots:
        mine = [r for r in rows if r["config_key"] == key]
        hit = [r for r in mine if r["negative_entries"]]
        print(f"{key} ({roots[key].name}): {len(mine)} sweeps, {len(hit)} with a negative cell fraction, "
              f"{sum(r['mixtures_with_a_negative_cell_fraction'] for r in hit)} mixtures of "
              f"{sum(r['mixtures'] for r in mine)}")
        for r in hit:
            print(f"    {r['sweep']}: {r['which']} (most negative {r['most_negative']})")
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
