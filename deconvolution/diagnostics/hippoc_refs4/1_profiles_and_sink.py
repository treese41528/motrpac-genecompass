"""HIPPOC refs4 diagnosis: why does BayesPrism give endothelial 0.50 / excitatory 0.003?"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.sparse as sp
from scipy.optimize import nnls
from scipy.stats import spearmanr

ROOT = Path("/depot/reese18/apps/motrpac-genecompass")
sys.path.insert(0, str(ROOT / "deconvolution"))
from joint_reannotate import symbol_index  # noqa: E402

D = ROOT / "data/deconvolution"
REF = D / "references_v4/HIPPOC_GSE295314"
RES = D / "results_refs4/motrpac/HIPPOC"
BULK = D / "motrpac_bulk/HIPPOC"
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)

genes = pd.read_csv(REF / "genes.tsv", header=None)[0].astype(str).values
meta = pd.read_csv(REF / "cells_meta.tsv", sep="\t")
X = sio.mmread(REF / "reference_counts.mtx").tocsr()
if X.shape[0] == len(genes) and X.shape[1] == len(meta):
    X = X.T.tocsr()
assert X.shape == (len(meta), len(genes)), X.shape
types = sorted(meta.cell_type.unique())
print(f"reference: {X.shape[0]} cells x {X.shape[1]} genes, {len(types)} types")

# --- 1. per-type quality and profiles
umi = np.asarray(X.sum(1)).ravel(); ng = np.asarray((X > 0).sum(1)).ravel()
q = pd.DataFrame({"ct": meta.cell_type, "umi": umi, "ngenes": ng}).groupby("ct").agg(
    cells=("umi", "size"), median_umi=("umi", "median"), median_genes=("ngenes", "median"))
prof = pd.DataFrame({t: np.asarray(X[(meta.cell_type == t).values].sum(0)).ravel() for t in types}, index=genes)
cpm = prof / prof.sum() * 1e6

bg = pd.read_csv(BULK / "bulk_genes.tsv", header=None, sep="\t")[0].astype(str).values
B = sio.mmread(BULK / "bulk.mtx").tocsr()
nsamp = len([l for l in open(BULK / "bulk_samples.tsv").read().split() if l])
if B.shape[0] == len(bg):
    B = B.T.tocsr()
B = pd.DataFrame(B.toarray(), columns=bg)
print(f"bulk: {B.shape[0]} samples x {B.shape[1]} genes (bulk_samples.tsv lists {nsamp})")

bp_genes = pd.read_csv(RES / "pred_z/genes.txt", header=None)[0].astype(str).values
print(f"BayesPrism gene set: {len(bp_genes)}")

idx = symbol_index(list(set(genes) | set(bg)))
def gid(sym):
    s = sorted(idx.get(sym.lower(), []))
    return s[0] if s else None

MARK = {"neuron": ["Snap25", "Syt1", "Rbfox3", "Camk2a", "Slc17a7", "Neurod6", "Prox1", "Gad1", "Gad2"],
        "endothelial": ["Cldn5", "Flt1", "Slco1a4", "Itm2a", "Ly6a"],
        "astro": ["Gfap", "Aqp4", "Slc1a3"], "oligo": ["Mbp", "Plp1", "Mog"],
        "microglia": ["Cx3cr1", "P2ry12", "Csf1r"],
        "nuclear/mito": ["Malat1", "Meg3", "Mt-co1", "Mt-nd1", "Mt-cytb"]}
bulk_cpm = (B.T / B.sum(1)).T * 1e6
rows = []
for grp, syms in MARK.items():
    for s in syms:
        g = gid(s)
        r = {"group": grp, "gene": s, "id": g}
        if g in cpm.index:
            r.update(cpm.loc[g].round(0).to_dict())
        r["BULK"] = round(float(bulk_cpm[g].mean()), 0) if g in bulk_cpm.columns else np.nan
        r["in_BP"] = g in set(bp_genes)
        rows.append(r)
mk = pd.DataFrame(rows)
ab = {"Astrocytes": "Ast", "Dentate gyrus granule cells": "DG", "Endothelial cells": "Endo", "Excitatory neurons": "Exc",
      "Inhibitory neurons": "Inh", "Microglia": "Mic", "Mural cells": "Mur", "Oligodendrocyte precursor cells": "OPC",
      "Oligodendrocytes": "Olig", "Thalamic neurons": "Thal", "Vascular leptomeningeal cells": "VLMC"}
print("\n=== 1. per-type quality\n", q.astype(int).to_string())
print("\n=== 2. marker CPM per reference type and in the bulk (mean)\n", mk.rename(columns=ab).to_string(index=False))

# --- 3. independent NNLS on the BayesPrism genes (CPM profiles -> weights = mRNA fractions)
common = [g for g in bp_genes if g in cpm.index and g in bulk_cpm.columns]
P = cpm.loc[common]; P = P / P.sum() * 1e6
Y = bulk_cpm[common]; Y = (Y.T / Y.sum(1)).T * 1e6
def nnls_frac(y, P):
    w, _ = nnls(P.values, y); return w / w.sum()
lin = pd.DataFrame([nnls_frac(Y.iloc[i].values, P) for i in range(len(Y))], columns=P.columns).mean()
# log-scale-free variant: NNLS on sqrt to damp a few huge genes
sq = pd.DataFrame([nnls_frac(np.sqrt(Y.iloc[i].values), np.sqrt(P)) for i in range(len(Y))], columns=P.columns).mean()
# marker genes only: top 50 genes per type by specificity (CPM / max other CPM), expressed >= 50 CPM
spec_genes = []
for t in P.columns:
    other = P.drop(columns=t).max(1)
    s = (P[t] + 1) / (other + 1)
    spec_genes += list(s[P[t] >= 50].sort_values(ascending=False).index[:50])
spec_genes = list(dict.fromkeys(spec_genes))
mkr = pd.DataFrame([nnls_frac(Y.iloc[i][spec_genes].values, P.loc[spec_genes]) for i in range(len(Y))],
                   columns=P.columns).mean()
bp = pd.read_csv(RES / "estimated_fractions.csv", index_col=0)
cmp = pd.DataFrame({"BayesPrism": bp.mean(), "NNLS_lin": lin, "NNLS_sqrt": sq, "NNLS_markers": mkr}).round(3)
print(f"\n=== 3. fractions: BayesPrism vs independent NNLS ({len(common)} genes; markers {len(spec_genes)})\n",
      cmp.sort_values("BayesPrism", ascending=False).to_string())
ylog = np.log1p(Y.mean())
print("\n  Spearman(bulk mean, profile) per type:",
      {ab[t]: round(spearmanr(ylog, np.log1p(P[t])).correlation, 3) for t in P.columns})

# --- 4. per-sample endothelial fraction behaviour
e = bp["Endothelial cells"]; n = bp[["Excitatory neurons", "Inhibitory neurons", "Dentate gyrus granule cells"]].sum(1)
print(f"\n=== 4. per-sample: endothelial min {e.min():.3f} median {e.median():.3f} max {e.max():.3f}; "
      f"corr(endo, Exc+Inh+DG) = {np.corrcoef(e, n)[0,1]:.3f}")
print("  per-sample fraction SD per type:", bp.std().round(3).to_dict())

# --- 5. what does BayesPrism put in the endothelial Z?
zg = pd.read_csv(RES / "pred_z/genes.txt", header=None)[0].astype(str).values
tf = pd.read_csv(RES / "pred_z/celltype_files.tsv", sep="\t")
print("\n  celltype_files columns:", tf.columns.tolist())
Z = {}
for f in sorted((RES / "pred_z").glob("predz__*.csv")):
    z = pd.read_csv(f, index_col=0); Z[f.stem.replace("predz__", "")] = z.clip(lower=0).sum(0)
Zt = pd.DataFrame(Z)
Zt.index = zg if len(Zt) == len(zg) else Zt.index
print("  share of total Z mass per type:", (Zt.sum() / Zt.values.sum()).round(3).to_dict())
endo = Zt["Endothelial_cells"]
rev = {}
for k, v in idx.items():
    for g in v: rev.setdefault(g, k)
top = endo.sort_values(ascending=False).index[:40]
tt = pd.DataFrame({"sym": [rev.get(g, g) for g in top],
                   "Zendo_share_of_gene": (endo[top] / Zt.loc[top].sum(1)).round(2),
                   "Zendo_mass_pct": (endo[top] / endo.sum() * 100).round(2),
                   "ref_Endo_cpm": [round(cpm.loc[g, "Endothelial cells"]) if g in cpm.index else np.nan for g in top],
                   "ref_max_neuron_cpm": [round(cpm.loc[g, ["Excitatory neurons", "Inhibitory neurons",
                                          "Dentate gyrus granule cells"]].max()) if g in cpm.index else np.nan for g in top]})
print("\n=== 5. top 40 genes by endothelial Z mass\n", tt.to_string())
# How endothelial-specific is the endothelial reference profile? fraction of its CPM in genes where it is the top type
top_type = cpm.idxmax(1)
print("\n  endothelial profile CPM mass on genes where Endo is the max type: "
      f"{cpm.loc[top_type == 'Endothelial cells', 'Endothelial cells'].sum() / 1e4:.1f}%")
print("  correlation (log CPM) between reference profiles:")
lc = np.log1p(cpm.loc[common]); print(lc.corr().rename(index=ab, columns=ab).round(2).to_string())
