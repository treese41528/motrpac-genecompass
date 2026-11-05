"""HIPPOC: ambient (soup) correction test. Soup = the barcodes the July reference kept and refs4 dropped
(empty droplets / ambient nuclei). (1) How much does each refs4 type, and the July types, look like the
soup? (2) SoupX-style correction of every refs4 type, then NNLS (with and without the soup as an extra
component). Diagnostic only."""
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.optimize import nnls

D = Path("/depot/reese18/apps/motrpac-genecompass/data/deconvolution")
R4, R3 = D / "references_v4/HIPPOC_GSE295314", D / "references_v3/HIPPOC_GSE295314"
RES, BU = D / "results_refs4/motrpac/HIPPOC", D / "motrpac_bulk/HIPPOC"
pd.set_option("display.width", 250)

genes = pd.read_csv(R4 / "genes.tsv", header=None)[0].astype(str).values
g3 = pd.read_csv(R3 / "genes.tsv", header=None)[0].astype(str).values
assert (genes == g3).all()
m4 = pd.read_csv(R4 / "cells_meta.tsv", sep="\t"); m3 = pd.read_csv(R3 / "cells_meta.tsv", sep="\t")
X4 = sio.mmread(R4 / "reference_counts.mtx").tocsr()
if X4.shape[0] == len(genes): X4 = X4.T.tocsr()
X3 = sio.mmread(R3 / "reference_counts.mtx").tocsr()
if X3.shape[0] == len(genes): X3 = X3.T.tocsr()
k4 = set(zip(m4.barcode, m4["sample"])); junk = np.array([(b, s) not in k4 for b, s in zip(m3.barcode, m3["sample"])])
print(f"July barcodes {len(m3)}, kept in refs4 {(~junk).sum()}, soup (dropped) {junk.sum()}")

bg = pd.read_csv(BU / "bulk_genes.tsv", header=None, sep="\t")[0].astype(str).values
B = sio.mmread(BU / "bulk.mtx").tocsr()
if B.shape[0] == len(bg): B = B.T.tocsr()
B = pd.DataFrame(B.toarray(), columns=bg)
bp = pd.read_csv(RES / "pred_z/genes.txt", header=None)[0].astype(str).values
common = [g for g in bp if g in set(genes) and g in set(bg)]
gi = pd.Index(genes).get_indexer(common)
X4, X3 = X4[:, gi], X3[:, gi]
Y = B[common]; Y = (Y.T / Y.sum(1)).T * 1e6


def prof(X, lab):
    P = pd.DataFrame({t: np.asarray(X[(lab == t)].sum(0)).ravel() for t in sorted(set(lab))}, index=common)
    return P / P.sum() * 1e6


P4 = prof(X4, m4.cell_type.values)
soup = pd.Series(np.asarray(X3[junk].sum(0)).ravel(), index=common); soup = soup / soup.sum() * 1e6
P3 = prof(X3, m3.cell_type.values)
lb = np.log1p(Y.mean()); ls = np.log1p(soup)
print("\n(1) Pearson(log CPM) with the soup / with the bulk mean:")
for name, P in (("refs4", P4), ("July", P3)):
    print(f"  {name}: " + ", ".join(f"{t[:14]} {np.corrcoef(np.log1p(P[t]), ls)[0,1]:.2f}/{np.corrcoef(np.log1p(P[t]), lb)[0,1]:.2f}"
                                    for t in P.columns))
print(f"  soup vs bulk: {np.corrcoef(ls, lb)[0,1]:.2f}")
# July excitatory: what share of its barcodes are soup?
ex = m3.cell_type.values == "Excitatory neurons"
print(f"  July 'Excitatory neurons': {ex.sum()} barcodes, {junk[ex].mean():.1%} of them dropped as junk in refs4")
mic = m3.cell_type.values == "Microglia"
print(f"  July 'Microglia': {mic.sum()} barcodes, {junk[mic].mean():.1%} dropped as junk")


def fit(P, extra=None):
    Q = P.copy()
    if extra is not None:
        Q["SOUP"] = extra
    W = np.array([nnls(np.sqrt(Q.values), np.sqrt(Y.iloc[i].values))[0] for i in range(len(Y))])
    return pd.Series((W / W.sum(1, keepdims=True)).mean(0), index=Q.columns)


# (2) SoupX-style contamination per type: rho_t from genes that are soup-abundant but specific to another
# type (this type's own expression ~0 there): rho_t = sum p_t / sum soup over those genes (an upper bound)
rho = {}
for t in P4.columns:
    other = P4.drop(columns=t).max(1)
    cand = [g for g in soup.sort_values(ascending=False).index[:2000] if other[g] >= 20 * (P4.loc[g, t] + 1) or
            (other[g] > 200 and P4.loc[g, t] < 0.05 * other[g])]
    cand = cand[:100]
    rho[t] = min(0.95, P4.loc[cand, t].sum() / soup[cand].sum()) if cand else 0.0
rho = pd.Series(rho)
print("\n(2) estimated ambient fraction per refs4 type (SoupX-style upper bound):\n  " +
      ", ".join(f"{t} {r:.2f}" for t, r in rho.sort_values(ascending=False).items()))
Pc = pd.DataFrame({t: ((P4[t] - rho[t] * soup) / (1 - rho[t])).clip(lower=0) for t in P4.columns})
Pc = Pc / Pc.sum() * 1e6
out = pd.DataFrame({"BayesPrism": pd.read_csv(RES / "estimated_fractions.csv", index_col=0).mean(),
                    "NNLS as built": fit(P4), "NNLS + soup": fit(P4, soup),
                    "NNLS ambient-corrected": fit(Pc), "NNLS corrected + soup": fit(Pc, soup),
                    "NNLS July ref": fit(P3).reindex(P4.columns.union(P3.columns))}).fillna(0)
mg = {"Dentate gyrus granule cells": "Excitatory neurons", "Thalamic neurons": "Excitatory neurons"}
Pm = Pc.T.groupby(lambda t: mg.get(t, t)).sum().T; Pm = Pm / Pm.sum() * 1e6
out["corrected + neurons merged"] = fit(Pm).reindex(out.index).fillna(0)
out = out.round(3)
nr = [i for i in out.index if "neuron" in i.lower() or "granule" in i.lower()]
out.loc["ALL NEURONS"] = out.loc[nr].sum()
print("\n", out.sort_values("BayesPrism", ascending=False).to_string())

# (3) corrected + vascular merged (Endothelial + Mural + VLMC) + neurons merged; and corrected minus tiny types
vm = {"Mural cells": "Vascular", "Endothelial cells": "Vascular", "Vascular leptomeningeal cells": "Vascular", **mg}
lab = pd.Series(m4.cell_type.values).replace(vm).values
Pv = prof(X4, lab)
rv = {}
for t in Pv.columns:
    other = Pv.drop(columns=t).max(1)
    cand = [g for g in soup.sort_values(ascending=False).index[:2000] if other[g] >= 20 * (Pv.loc[g, t] + 1) or
            (other[g] > 200 and Pv.loc[g, t] < 0.05 * other[g])][:100]
    rv[t] = min(0.95, Pv.loc[cand, t].sum() / soup[cand].sum()) if cand else 0.0
Pvc = pd.DataFrame({t: ((Pv[t] - rv[t] * soup) / (1 - rv[t])).clip(lower=0) for t in Pv.columns}); Pvc = Pvc / Pvc.sum() * 1e6
r3 = pd.DataFrame({"vascular merged, raw": fit(Pv), "vascular merged, corrected": fit(Pvc),
                   "corrected, Mural dropped": fit(Pc.drop(columns=["Mural cells"])).reindex(Pvc.columns.union(Pc.columns))}).fillna(0).round(3)
print("\n(3) ambient fractions (merged):", {k: round(v, 2) for k, v in rv.items()})
print(r3.to_string())
