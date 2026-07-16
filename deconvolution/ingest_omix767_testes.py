#!/usr/bin/env python3
"""ingest_omix767_testes.py -- stage the OMIX767 rat-testis 10x arms into corpus-format h5ads.

Phase 1 of TESTES ingestion (deconvolution/tissue_references.yaml TESTES / REFERENCE_SELECTION_PLAN.md):
read the staged 10x mtx triplets for the two NORMAL-testosterone arms (C = untreated control, E7W =
Leydig restored; Geyu's C+E7W pool), build cells x genes AnnData over ENSRNOG (the corpus var space),
compute the corpus QC obs (library_size / n_genes_detected / mito_pct), light-QC filter, and write
qc_matrices/OMIX767_sample{C,E7W}.h5ad. Phase 2 (NOT here) = Leiden + annotate_celltypes/sctype +
consensus_annotations + append annotation_inventory rows, then build_reference.py --study OMIX767.
"""
import sys
from pathlib import Path
import numpy as np, scipy.io as sio, scipy.sparse as sp, pandas as pd, anndata as ad

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data/raw/ngdc/datasets/OMIX767/Gene_bc_matrices"
OUT = ROOT / "data/training/preprocessed/qc_matrices"
ARMS = {"C": "C_gene_bc_matrices", "E7W": "E7W_gene_bc_matrices"}
MIN_GENES, MIN_CELLS, MAX_MITO = 200, 3, 20.0   # standard 10x QC

def build(arm, sub):
    d = BASE / sub
    M = sio.mmread(str(d / "matrix.mtx")).tocsr()                    # 10x: genes x cells
    genes = pd.read_csv(d / "genes.tsv", sep="\t", header=None)
    bcs = pd.read_csv(d / "barcodes.tsv", sep="\t", header=None)[0].astype(str).tolist()
    ensrnog = genes[0].astype(str).tolist()
    symbol = genes[1].astype(str).tolist() if genes.shape[1] > 1 else ensrnog
    X = sp.csr_matrix(M.T)                                           # -> cells x genes
    assert X.shape == (len(bcs), len(ensrnog)), (X.shape, len(bcs), len(ensrnog))
    a = ad.AnnData(X=X)
    a.var_names = ensrnog; a.var["symbol"] = symbol
    a.var_names_make_unique()
    a.obs_names = [f"OMIX767_sample{arm}_{b}" for b in bcs]
    # QC metrics (corpus obs contract)
    lib = np.asarray(X.sum(1)).ravel()
    ng = np.asarray((X > 0).sum(1)).ravel()
    mito = np.array([s.lower().startswith(("mt-", "mt.")) for s in symbol])
    mito_counts = np.asarray(X[:, mito].sum(1)).ravel() if mito.any() else np.zeros(X.shape[0])
    mito_pct = np.where(lib > 0, 100.0 * mito_counts / lib, 0.0)
    a.obs["library_size"] = lib; a.obs["n_genes_detected"] = ng; a.obs["mito_pct"] = mito_pct
    a.obs["sample_id"] = f"OMIX767_sample{arm}"; a.obs["arm"] = arm
    n0 = a.n_obs
    a = a[(ng >= MIN_GENES) & (mito_pct <= MAX_MITO)].copy()
    import scanpy as sc
    sc.pp.filter_genes(a, min_cells=MIN_CELLS)
    print(f"  {arm}: {n0} -> {a.n_obs} cells (min_genes={MIN_GENES}, mito<={MAX_MITO}%), {a.n_vars} genes; "
          f"median libsize={np.median(a.obs['library_size']):.0f}")
    out = OUT / f"OMIX767_sample{arm}.h5ad"
    a.write_h5ad(out)
    print(f"     wrote {out}")
    return a.n_obs

if __name__ == "__main__":
    tot = sum(build(arm, sub) for arm, sub in ARMS.items())
    print(f"\nTESTES phase-1 h5ads written: {tot} cells total (C+E7W). "
          f"NEXT: Leiden + annotate + consensus + inventory rows (phase 2).")
