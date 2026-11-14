#!/usr/bin/env python3
"""step6_self_consistency_medians.py -- Step 6 (records, tests, hygiene), audit finding 9.4.

Two different summaries of the rat-human network self-consistency circulate under the name "median":
  (a) the median over every TF x network score (conservation_per_tf.tsv, column spearman);
  (b) the median of the per-network medians (conservation_per_celltype.tsv, column median_spearman).
The audit computed both once. This script re-derives them from the tables on disk, for the 14-tissue refs4
arm (the arm the two quoted figures, 0.310 and 0.297, belong to) and for the arm the default config names.
Read-only.

  python analysis/audit_remediation/step6_self_consistency_medians.py
Out: docs/audit/remediation/record/step6/self_consistency_two_medians.tsv
"""
import csv
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "lib"))
from gene_utils import load_config, resolve_path   # noqa: E402

OUT = ROOT / "docs/audit/remediation/record/step6/self_consistency_two_medians.tsv"


def summarise(label, folder):
    per_tf, per_net = folder / "conservation_per_tf.tsv", folder / "conservation_per_celltype.tsv"
    if not per_tf.exists() or not per_net.exists():
        return {"arm": label, "folder": str(folder.relative_to(ROOT)), "note": "tables not on disk"}
    with open(per_tf, newline="") as fh:
        scores = [float(r["spearman"]) for r in csv.DictReader(fh, delimiter="\t") if r["spearman"] not in ("", "NA", "nan")]
    with open(per_net, newline="") as fh:
        nets = [float(r["median_spearman"]) for r in csv.DictReader(fh, delimiter="\t")]
    return {"arm": label, "folder": str(folder.relative_to(ROOT)), "networks": len(nets), "tf_network_scores": len(scores),
            "median_over_all_tf_network_scores": round(statistics.median(scores), 4),
            "median_of_the_per_network_medians": round(statistics.median(nets), 4),
            "per_network_medians": "; ".join(f"{v:.4f}" for v in sorted(nets)), "note": ""}


def main():
    cfg = load_config()
    default = resolve_path(cfg, cfg["deconvolution"]["grn_human_dir"]) / "conservation"
    rows = [summarise("refs4, 14 tissues (the arm of the two quoted figures)",
                      ROOT / "data/deconvolution/grn_v2_refs4_human/conservation"),
            summarise("the default config's arm when this was run (deconvolution.grn_human_dir)", default)]
    fields = ["arm", "folder", "networks", "tf_network_scores", "median_over_all_tf_network_scores",
              "median_of_the_per_network_medians", "per_network_medians", "note"]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, delimiter="\t", lineterminator="\n")
        w.writeheader(); w.writerows(rows)
    for r in rows:
        print({k: r.get(k) for k in fields if k != "per_network_medians"})
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
