#!/usr/bin/env python3
"""build_lung_pooled.py -- build a POOLED NATIVE adult-rat-lung BayesPrism reference.

Replaces the engineered/in-vitro GSE178405 lung reference (cell isolates / engineered-d7 /
tri-culture / P7-developing -- the root cause of lung being the weakest deconv tissue) with a
native reference pooled from the HEALTHY control arms of three independent adult rat lung
scRNA studies (all already in-corpus + consensus-annotated):
    GSE273062  VeNx (vehicle+normoxia) controls  -> sample0, sample2   (2)
    GSE252844  C3 control (blast-injury study)    -> sample0           (1)
    GSE242310  NOX (normoxia) control             -> sample1           (1)
= 4 healthy native adult lung samples. (GSE247625's 2 healthy are deferred -- mouse-mm10 aligned,
needs re-alignment.) Reuses build_reference machinery: per-study load_study (consensus labels via
barcode->leiden->consensus_label), cross-study OUTER gene join (studies differ in depth), the
'lung' label-scheme (merge Club/Clara/ciliated/NK synonyms; keep immune resolved), clean_cells,
export_reference. Run on a compute node (loads h5ads).
"""
import sys
import anndata as ad
import scanpy as sc
sys.path.insert(0, "deconvolution")
import build_reference as br

HEALTHY = [
    ("GSE273062", ["GSE273062_sample0", "GSE273062_sample2"]),   # VeNx normoxia controls
    ("GSE252844", ["GSE252844_sample0"]),                        # C3 control
    ("GSE242310", ["GSE242310_sample1"]),                        # NOX normoxia control
]
OUT = "data/deconvolution/references/lung_native_pooled"

parts = []
for study, sids in HEALTHY:
    a = br.load_study(study, "lung", sample_ids=sids, gene_join="outer", min_gene_cells=0)
    a.obs["source_study"] = study
    print(f"  {study}: {a.n_obs} cells, {a.obs['cell_type'].nunique()} consensus labels")
    parts.append(a)

adata = ad.concat(parts, join="outer", merge="same", fill_value=0)
print(f"\npooled: {adata.n_obs} cells x {adata.n_vars} genes from {len(HEALTHY)} studies")
before = adata.n_vars
sc.pp.filter_genes(adata, min_cells=10)           # trim the outer-join union's long tail
print(f"min-gene-cells=10: {before} -> {adata.n_vars} genes")

adata.obs["cell_type"] = br.canonicalize_labels(adata.obs["cell_type"], "lung")
adata = br.clean_cells(adata, 20)                 # >=20 cells per (sample x state)
print(f"\nafter clean_cells: {adata.n_obs} cells, {adata.obs['cell_type'].nunique()} cell types")
print("cell-type counts:")
print(adata.obs["cell_type"].value_counts().to_string())
print("cells per source study:")
print(adata.obs["source_study"].value_counts().to_string())

br.export_reference(adata, OUT)
print(f"\nwrote pooled native lung reference -> {OUT}")
