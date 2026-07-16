#!/usr/bin/env python3
"""merge_consensus_labels.py -- collapse collinear / duplicate consensus labels in a reference.

This is the principled anti-over-split cleanup, applied when the authors' DEPOSITED per-cell labels
(adopt_authors_labels.py, route A) are NOT publicly accessible (login-gated SCP, or never deposited as a
barcode-level table). It does NOT invent new biology -- it only MERGES labels that name the same underlying
cell type into one basis label, which is exactly what BayesPrism needs: collinear GEPs (near-identical
expression) split across multiple labels cause the dominant-parenchyma collinear collapse, so the split
label fractions are individually noisy while their SUM is stable. Merging restores one stable fraction.

Each tissue's MERGES map is {old_label -> new_label}; any label not in the map is kept as-is. We rewrite
consensus.tsv (cluster -> consensus_label) for the reference's samples; celltypes.tsv (barcode->cluster) is
untouched. Rebuild the reference afterwards to apply it. consensus_source is tagged 'merged_anti_oversplit'.

Usage: python deconvolution/merge_consensus_labels.py --tissue MUSCLE [--dry-run]
Then rebuild: python deconvolution/build_reference.py --study ... --tissue ... --out ...
"""
import argparse, os, sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("PIPELINE_ROOT", str(Path(__file__).resolve().parents[1]))
import build_reference as br

# tissue (reference key) -> {reference_dir, samples, merges}
MERGES = {
  "MUSCLE": {  # GSE137869 Ma 2020: myofiber parenchyma split across 4 collinear labels; FAP/myofibroblast split
    "reference_dir": "data/deconvolution/references_v3/MUSCLE_GSE137869_Y",
    "merges": {
      "Skeletal muscle cells":  "Skeletal myocytes",
      "Skeletal muscle fibers": "Skeletal myocytes",
      "Skeletal muscle":        "Skeletal myocytes",
      "Myocytes":               "Skeletal myocytes",
      "Myofibroblasts":         "Fibroblasts",
    },
  },
}


def samples_of(ref_dir):
    m = pd.read_csv(Path(br.PROJECT) / ref_dir / "cells_meta.tsv", sep="\t")
    return list(pd.unique(m["sample"]))


def apply_merge(tissue, dry):
    spec = MERGES[tissue]
    mp = spec["merges"]
    samples = spec.get("samples") or samples_of(spec["reference_dir"])
    print(f"=== merge {tissue}: {len(samples)} sample(s), {len(mp)} label merges ===")
    for old, new in mp.items():
        print(f"    {old!r} -> {new!r}")
    for s in samples:
        cf = br.CONS_DIR / s / f"{s}_consensus.tsv"
        if not cf.exists():
            print(f"  [{s}] MISSING consensus.tsv -- skip"); continue
        df = pd.read_csv(cf, sep="\t")
        changed = df["consensus_label"].isin(mp)
        n_changed = int(changed.sum())
        df["consensus_label"] = df["consensus_label"].map(lambda x: mp.get(x, x))
        if "consensus_source" not in df:
            df["consensus_source"] = "consensus"
        df.loc[changed, "consensus_source"] = "merged_anti_oversplit"
        after = df.groupby("consensus_label")["n_cells"].sum().sort_values(ascending=False).to_dict() \
            if "n_cells" in df else df["consensus_label"].value_counts().to_dict()
        print(f"  [{s}] {n_changed} cluster-row(s) relabeled | after: "
              + ", ".join(f"{k}={v}" for k, v in list(after.items())[:8]))
        if not dry:
            df.to_csv(cf, sep="\t", index=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tissue", required=True, choices=sorted(MERGES))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    apply_merge(args.tissue, args.dry_run)
    if not args.dry_run:
        print(f"\nrewrote consensus for {args.tissue}; now rebuild the reference and re-deconvolve.")


if __name__ == "__main__":
    main()
