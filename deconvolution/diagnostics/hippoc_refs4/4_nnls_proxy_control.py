"""Control: does the NNLS proxy reproduce BayesPrism on CORTEX (a working snRNA reference)?"""
from pathlib import Path
import numpy as np, pandas as pd, scipy.io as sio
from scipy.optimize import nnls
D = Path("/depot/reese18/apps/motrpac-genecompass/data/deconvolution")
for T, ref in (("CORTEX", "CORTEX_GSE303115"), ("HIPPOC", "HIPPOC_GSE295314")):
    R = D / "references_v4" / ref; RES = D / "results_refs4/motrpac" / T; BU = D / "motrpac_bulk" / T
    genes = pd.read_csv(R / "genes.tsv", header=None)[0].astype(str).values
    meta = pd.read_csv(R / "cells_meta.tsv", sep="\t")
    X = sio.mmread(R / "reference_counts.mtx").tocsr()
    if X.shape[0] == len(genes): X = X.T.tocsr()
    bg = pd.read_csv(BU / "bulk_genes.tsv", header=None, sep="\t")[0].astype(str).values
    B = sio.mmread(BU / "bulk.mtx").tocsr()
    if B.shape[0] == len(bg): B = B.T.tocsr()
    B = pd.DataFrame(B.toarray(), columns=bg)
    bp = pd.read_csv(RES / "pred_z/genes.txt", header=None)[0].astype(str).values
    common = [g for g in bp if g in set(genes) and g in set(bg)]
    X = X[:, pd.Index(genes).get_indexer(common)]
    umi = np.asarray(X.sum(1)).ravel()
    P = pd.DataFrame({t: np.asarray(X[(meta.cell_type == t).values].sum(0)).ravel() for t in sorted(meta.cell_type.unique())}, index=common)
    P = P / P.sum() * 1e6
    Y = B[common]; Y = (Y.T / Y.sum(1)).T * 1e6
    W = np.array([nnls(np.sqrt(P.values), np.sqrt(Y.iloc[i].values))[0] for i in range(len(Y))])
    nn = pd.Series((W / W.sum(1, keepdims=True)).mean(0), index=P.columns)
    q = pd.DataFrame({"cells": meta.cell_type.value_counts(), "median_umi": pd.Series(umi).groupby(meta.cell_type.values).median()})
    out = pd.DataFrame({"BayesPrism": pd.read_csv(RES / "estimated_fractions.csv", index_col=0).mean(), "NNLS_sqrt": nn}).join(q).round(3)
    print(f"\n=== {T}\n", out.sort_values("BayesPrism", ascending=False).to_string())
