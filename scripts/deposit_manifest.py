#!/usr/bin/env python3
"""md5 manifest of the inputs a rebuild cannot regenerate byte for byte, and must restore from the deposit.

Sets already pinned elsewhere are only pointed to, not repeated:
- the 868 QC h5ads: config/corpus_v2_plan.tsv
- the reference files: config/reference_md5sums.tsv
- the GeneCompass base assets: config/genecompass_assets_md5.tsv
- the external deconvolution references: deconvolution/external_refs_md5.tsv

This script covers the rest: the gene universes, the Stage 1 catalog, including the LLM study analysis (not
deterministic), the cell annotations, the joint labels, the v2 init (built unseeded), and the v2 weights.

    python scripts/deposit_manifest.py --write     # hash everything in DEPOSIT -> config/deposit_md5.tsv (batch job)
    python scripts/deposit_manifest.py --verify    # check a restored deposit against it (exit 1 on any mismatch)
"""
import argparse
import csv
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TABLE = ROOT / "config/deposit_md5.tsv"

# (group, glob relative to the repo root)
DEPOSIT = [
    ("stage2_universe", "data/training/gene_universe/gene_universe.tsv"),
    ("stage2_universe", "data/training/gene_universe/pruned_gene_universe.tsv"),
    ("stage1_catalog", "data/catalog/unified_studies.json"),
    ("stage1_catalog", "data/catalog/llm_study_analysis.json"),
    ("stage1_catalog", "data/catalog/master_catalog.json"),
    ("stage1_catalog", "data/catalog/matrix_analysis.json"),
    ("annotations", "data/training/cell_annotations/**/*"),
    ("annotations", "data/training/cell_annotations_consensus/**/*"),
    ("annotations", "reports/annotations/annotation_inventory.tsv"),
    ("joint_labels", "data/deconvolution/joint_labels/*/cells_final.tsv.gz"),
    ("model_init_v2", "data/models/rat_genecompass_init_v2/*"),
    ("model_v2", "data/models/rat_genecompass_finetuned_v2/models/rat_phase2_mixed_species/models/*"),
]


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def collect():
    for group, pattern in DEPOSIT:
        hits = sorted(p for p in ROOT.glob(pattern) if p.is_file())
        if not hits:
            sys.exit(f"ERROR: nothing matches {pattern}")
        for p in hits:
            yield group, p


def write():
    rows = [(g, str(p.relative_to(ROOT)), p.stat().st_size, md5(p)) for g, p in collect()]
    with open(TABLE, "w") as f:
        f.write("# Deposit manifest: written by scripts/deposit_manifest.py --write. Check a restored deposit with --verify.\n"
                "# The QC h5ads, reference files, GeneCompass assets and external references have their own tables.\n")
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["group", "path", "bytes", "md5"])
        w.writerows(rows)
    print(f"{len(rows)} files, {sum(r[2] for r in rows) / 1e9:.1f} GB -> {TABLE.relative_to(ROOT)}")


def verify():
    rows = list(csv.DictReader((l for l in open(TABLE) if not l.startswith("#")), delimiter="\t"))
    bad = 0
    for r in rows:
        p = ROOT / r["path"]
        if not p.is_file():
            print(f"MISSING  {r['path']}"); bad += 1
        elif p.stat().st_size != int(r["bytes"]) or md5(p) != r["md5"]:
            print(f"DIFFERS  {r['path']}"); bad += 1
    print(f"{len(rows) - bad}/{len(rows)} files match {TABLE.relative_to(ROOT)}")
    return bad == 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--write", action="store_true")
    g.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    if a.write:
        write()
    else:
        sys.exit(0 if verify() else 1)
