"""HIPPOC refs4: NNLS what-ifs on the candidate causes (endothelial ambient soup; nuclear-vs-whole-cell genes;
neuron collinearity). Diagnostic only -- nothing here changes the reference or the registered run."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.optimize import nnls

ROOT = Path("/depot/reese18/apps/motrpac-genecompass")
D = ROOT / "data/deconvolution"
REF = D / "references_v4/HIPPOC_GSE295314"
RES = D / "results_refs4/motrpac/HIPPOC"
BULK = D / "motrpac_bulk/HIPPOC"
pd.set_option("display.width", 250)

genes = pd.read_csv(REF / "genes.tsv", header=None)[0].astype(str).values
meta = pd.read_csv(REF / "cells_meta.tsv", sep="\t")
X = sio.mmread(REF / "reference_counts.mtx").tocsr()
if X.shape[0] == len(genes):
    X = X.T.tocsr()
bg = pd.read_csv(BULK / "bulk_genes.tsv", header=None, sep="\t")[0].astype(str).values
B = sio.mmread(BULK / "bulk.mtx").tocsr()
if B.shape[0] == len(bg):
    B = B.T.tocsr()
B = pd.DataFrame(B.toarray(), columns=bg)
bp_genes = pd.read_csv(RES / "pred_z/genes.txt", header=None)[0].astype(str).values
common = [g for g in bp_genes if g in set(genes) and g in set(bg)]
gi = pd.Index(genes).get_indexer(common)
X = X[:, gi]
Y = B[common]; Y = (Y.T / Y.sum(1)).T * 1e6
umi = np.asarray(X.sum(1)).ravel()
ct = meta.cell_type.values


def profiles(mask=None, merge=None):
    lab = pd.Series(ct)
    if merge:
        lab = lab.replace(merge)
    m = np.ones(len(lab), bool) if mask is None else mask
    P = pd.DataFrame({t: np.asarray(X[(m & (lab == t).values)].sum(0)).ravel()
                      for t in sorted(lab[m].unique())}, index=common)
    return P / P.sum() * 1e6


def fit(P, genes_used=None, drop=()):
    P = P.drop(columns=list(drop))
    g = genes_used if genes_used is not None else P.index
    Pg = P.loc[g]; Pg = Pg / Pg.sum() * 1e6
    Yg = Y[g]; Yg = (Yg.T / Yg.sum(1)).T * 1e6
    W = np.array([nnls(np.sqrt(Pg.values), np.sqrt(Yg.iloc[i].values))[0] for i in range(len(Yg))])
    return pd.Series((W / W.sum(1, keepdims=True)).mean(0), index=Pg.columns)


P0 = profiles()
res = {"0 refs4 as built": fit(P0)}

# E1: endothelial removed
res["E1 no endothelial"] = fit(P0, drop=["Endothelial cells"])

# E2: endothelial profile from its better nuclei only (top half by UMI) -- ambient share falls with UMI?
endo = ct == "Endothelial cells"
cut = np.median(umi[endo])
P2 = profiles(mask=~endo | (umi >= cut))
res["E2 endo top-half UMI"] = fit(P2)

# E3: ambient-corrected endothelial: subtract the reference-wide average (ambient proxy) scaled to the
# fraction that removes the neuronal/oligo marker signal, clip at 0
amb = P0.drop(columns=["Endothelial cells"]).mul(
    pd.Series(meta.cell_type.value_counts()).reindex(P0.columns.drop("Endothelial cells")), axis=1).sum(1)
amb = amb / amb.sum() * 1e6
best = None
for a in np.arange(0, 0.96, 0.05):
    e = (P0["Endothelial cells"] - a * amb).clip(lower=0)
    if e.sum() <= 0:
        break
    best = (a, e / e.sum() * 1e6)
    mk = [g for g in common if P0.loc[g].drop("Endothelial cells").max() > 5 * (P0.loc[g, "Endothelial cells"] + 1)]
P3 = P0.copy()
for a in (0.5, 0.7, 0.9):
    e = (P0["Endothelial cells"] - a * amb).clip(lower=0)
    P3["Endothelial cells"] = e / e.sum() * 1e6
    res[f"E3 endo minus {a:.1f} x ambient"] = fit(P3)

# E4: drop genes the nuclei under-represent relative to the bulk (cytoplasmic/dendritic): top 20% by
# log2(bulk / max reference CPM)
ratio = np.log2((Y.mean() + 1) / (P0.max(1) + 1))
keep = ratio[ratio <= ratio.quantile(0.80)].index
res["E4 drop nuclear-depleted 20%"] = fit(P0, genes_used=keep)
keep40 = ratio[ratio <= ratio.quantile(0.60)].index
res["E4b drop nuclear-depleted 40%"] = fit(P0, genes_used=keep40)

# E5: merge the collinear neurons (Exc + DG + Thal -> Excitatory)
mg = {"Dentate gyrus granule cells": "Excitatory neurons", "Thalamic neurons": "Excitatory neurons"}
res["E5 merge Exc+DG+Thal"] = fit(profiles(merge=mg))

# E6: E1 + E4 + E5 combined (no endothelial, nuclear-depleted genes dropped, neurons merged)
res["E6 E1+E4+E5"] = fit(profiles(merge=mg), genes_used=keep, drop=["Endothelial cells"])

out = pd.DataFrame(res).fillna(0).round(3)
out.loc["ALL NEURONS"] = out.loc[[i for i in out.index if "neuron" in i.lower() or "granule" in i.lower()]].sum()
print(out.to_string())
print("\nnuclear-depleted genes (top 30 by bulk / max-reference ratio):")
idx_sym = pd.read_csv(ROOT / "data/references/biomart/rat_gene_info.tsv", sep="\t", dtype=str)
sym = dict(zip(idx_sym["Gene stable ID"], idx_sym["Gene name"]))
print(", ".join(f"{sym.get(g, g)} {ratio[g]:.1f}" for g in ratio.sort_values(ascending=False).index[:30]))
e = np.where(endo)[0]
mbp = [i for i, g in enumerate(common) if sym.get(g) == "Mbp"]
if mbp:
    xm = np.asarray(X[e][:, mbp].sum(1)).ravel() / umi[e] * 1e6
    print(f"\nendothelial nuclei: Mbp CPM by UMI half -- low {np.median(xm[umi[e] < cut]):.0f}, "
          f"high {np.median(xm[umi[e] >= cut]):.0f}")
