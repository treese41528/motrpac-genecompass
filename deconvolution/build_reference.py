#!/usr/bin/env python3
"""
build_reference.py -- Assemble a BayesPrism single-cell reference from the
annotated rat SC corpus (Stage 8 deconvolution).

Paper-faithful: a single cohesive study, FULL cells with natural counts (no
balancing/subsampling), fine cell types + finer cell states. We do NOT pool
multiple studies into one reference.

Join: barcode -> leiden (per-cell) -> consensus_label (per-cluster)
      cell.type = consensus_label ; cell.state = "{sample}_c{leiden}"

Output (read by run_deconvolution.R; project-local, never /tmp):
  deconvolution/reference/{TISSUE}_{STUDY}/
    reference_counts.mtx   cells x genes integer (MatrixMarket)
    genes.tsv  genes (matrix columns) | cells_meta.tsv  per-cell labels | summary.txt

The loader/assembler/exporter are importable (make_pseudobulk.py reuses them).

Usage (project venv):
  python deconvolution/build_reference.py --study GSE220075 --tissue liver
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.sparse as sp
import anndata as ad
import scanpy as sc

PROJECT = Path("/depot/reese18/apps/motrpac-genecompass")
QC_DIR = Path("/depot/reese18/data/training/preprocessed/qc_matrices")
CT_DIR = PROJECT / "data/training/cell_annotations"
CONS_DIR = PROJECT / "data/training/cell_annotations_consensus"
INVENTORY = PROJECT / "reports/annotations/annotation_inventory.tsv"
UNKNOWN = {"Unknown", "unknown", "", "nan", "NA"}


def select_samples(study, tissue, conditions=None):
    """In-corpus sample IDs for study+tissue, optionally filtered by condition_resolved."""
    inv = pd.read_csv(INVENTORY, sep="\t", dtype=str)
    sel = inv[
        (inv["accession"] == study)
        & (inv["tissue_normalized"].str.lower() == tissue.lower())
        & (inv["in_corpus"].str.lower() == "true")
    ]
    if conditions:
        sel = sel[sel["condition_resolved"].isin(conditions)]
    samples = sorted(sel["sample_id"].tolist())
    if not samples:
        sys.exit(f"ERROR: no in-corpus {tissue} samples for {study} "
                 f"(conditions={conditions}) in {INVENTORY}")
    return samples


def load_sample(sample):
    """AnnData of raw counts over ENSRNOG genes, obs = barcode/sample/leiden/cell_type/cell_state."""
    h5 = QC_DIR / f"{sample}.h5ad"
    ct_tsv = CT_DIR / sample / f"{sample}_celltypes.tsv"
    cons_tsv = CONS_DIR / sample / f"{sample}_consensus.tsv"
    for p in (h5, ct_tsv, cons_tsv):
        if not p.exists():
            print(f"  [skip] {sample}: missing {p.name}")
            return None

    adata = sc.read_h5ad(h5)
    if adata.raw is not None:                       # raw integer counts
        counts, genes = adata.raw.X, np.asarray(adata.raw.var_names)
    else:
        counts, genes = adata.X, np.asarray(adata.var_names)
    counts = sp.csr_matrix(counts)
    if not np.all(counts.data == np.round(counts.data)):
        print(f"  [warn] {sample}: non-integer counts (not raw?)")

    cells = pd.read_csv(ct_tsv, sep="\t", dtype={"barcode": str, "leiden": str})
    cons = pd.read_csv(cons_tsv, sep="\t", dtype={"cluster": str})
    c2lab = dict(zip(cons["cluster"], cons["consensus_label"]))
    cells["cell_type"] = cells["leiden"].map(c2lab)
    cells["cell_state"] = sample + "_c" + cells["leiden"].astype(str)
    cells["sample"] = sample

    bc = pd.Index(adata.obs_names.astype(str))
    cells = cells.set_index("barcode").reindex(bc)
    miss = int(cells["cell_type"].isna().sum())
    if miss:
        print(f"  [warn] {sample}: {miss}/{len(bc)} cells lack annotation match")

    sub = ad.AnnData(X=counts, obs=cells.reset_index().rename(columns={"index": "barcode"}))
    sub.var_names = genes
    print(f"  [ok]  {sample}: {sub.n_obs} cells x {sub.n_vars} genes")
    return sub


def load_study(study, tissue, conditions=None):
    """Concatenate all (loadable) samples of a study/tissue on the shared gene set."""
    samples = select_samples(study, tissue, conditions)
    print(f"{study} / {tissue}: {len(samples)} samples -> {samples}")
    parts = [s for s in (load_sample(x) for x in samples) if s is not None]
    if not parts:
        sys.exit("ERROR: no samples loaded.")
    adata = ad.concat(parts, join="inner", merge="same")
    print(f"concatenated: {adata.n_obs} cells x {adata.n_vars} genes (shared)")
    return adata


def clean_cells(adata, min_state_cells=20, drop_unknown=True):
    """Drop unannotated/Unknown cells and cell states (clusters) with < min_state_cells."""
    keep = adata.obs["cell_type"].notna()
    if drop_unknown:
        keep &= ~adata.obs["cell_type"].isin(UNKNOWN)
    adata = adata[keep].copy()
    vc = adata.obs["cell_state"].value_counts()
    small = vc[vc < min_state_cells].index
    if len(small):
        print(f"dropping {len(small)} states <{min_state_cells} cells ({int(vc[small].sum())} cells)")
        adata = adata[~adata.obs["cell_state"].isin(small)].copy()
    return adata


def export_reference(adata, out_dir):
    """Write reference_counts.mtx (cells x genes int) + genes.tsv + cells_meta.tsv + summary.txt."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    X = sp.csr_matrix(adata.X)
    sio.mmwrite(str(out / "reference_counts.mtx"), X.astype(np.int32))
    pd.Series(adata.var_names).to_csv(out / "genes.tsv", index=False, header=False)
    adata.obs[["barcode", "sample", "leiden", "cell_type", "cell_state"]].to_csv(
        out / "cells_meta.tsv", sep="\t", index=False)
    ct = adata.obs["cell_type"].value_counts()
    with open(out / "summary.txt", "w") as fh:
        fh.write(f"cells={adata.n_obs} genes={adata.n_vars} "
                 f"cell_types={ct.size} cell_states={adata.obs['cell_state'].nunique()}\n\n")
        fh.write(ct.to_string())
    print(f"\n=== reference cell types ({ct.size}) ===")
    print(ct.to_string())
    print(f"\nWrote reference -> {out}/")
    return ct


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", required=True)
    ap.add_argument("--tissue", required=True)
    ap.add_argument("--min-state-cells", type=int, default=20)
    args = ap.parse_args()
    adata = load_study(args.study, args.tissue)
    adata = clean_cells(adata, args.min_state_cells)
    out = PROJECT / "deconvolution/reference" / f"{args.tissue}_{args.study}"
    export_reference(adata, out)


if __name__ == "__main__":
    main()
