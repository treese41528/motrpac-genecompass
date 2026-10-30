"""HIPPOC refs4h (REFERENCE_RELABEL_PREREG.md §13): BayesPrism composition with the SoupX-corrected reference
against the registered criteria, the NNLS proxy on the same reference, and the script-5 ambient estimator and
soup/bulk correlations per type, before (refs4) and after correction."""
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.optimize import nnls

D = Path("/depot/reese18/apps/motrpac-genecompass/data/deconvolution")
R4, R3, RH = (D / "references_v4/HIPPOC_GSE295314", D / "references_v3/HIPPOC_GSE295314",
              D / "references_v4/HIPPOC_GSE295314_ambcorr")
RES4, RESH, BU = D / "results_refs4/motrpac/HIPPOC", D / "results_refs4h/motrpac/HIPPOC", D / "motrpac_bulk/HIPPOC"
pd.set_option("display.width", 250)


def load(d, genes_ref=None):
    g = pd.read_csv(d / "genes.tsv", header=None)[0].astype(str).values
    m = pd.read_csv(d / "cells_meta.tsv", sep="\t", dtype=str)
    X = sio.mmread(d / "reference_counts.mtx").tocsr()
    if X.shape[0] == len(g) and X.shape[1] == len(m): X = X.T.tocsr()
    if genes_ref is not None: assert (g == genes_ref).all()
    return g, m, X


genes, m4, X4 = load(R4)
_, mh, XH = load(RH, genes)
_, m3, X3 = load(R3, genes)
assert (mh.barcode.values == m4.barcode.values).all() and (mh["sample"].values == m4["sample"].values).all()
k4 = set(zip(m4.barcode, m4["sample"])); junk = np.array([(b, s) not in k4 for b, s in zip(m3.barcode, m3["sample"])])

print("SoupX per sample:\n" + pd.read_csv(RH / "soupx_rho.tsv", sep="\t").to_string(index=False))
print("\nSoupX per refs4 type:\n" + pd.read_csv(RH / "soupx_per_type.tsv", sep="\t").to_string(index=False))

# (1) composition against the registered criteria
fh = pd.read_csv(RESH / "estimated_fractions.csv", index_col=0)
f4 = pd.read_csv(RES4 / "estimated_fractions.csv", index_col=0)
mg = dict(zip(mh.cell_type_refs4, mh.cell_type))
f4m = f4.T.groupby(lambda t: mg.get(t, t)).sum().T
comp = pd.DataFrame({"refs4 (merged names)": f4m.mean(), "refs4h": fh.mean(), "refs4h sd": fh.std()}).fillna(0).round(3)
print(f"\n(1) BayesPrism mean fraction over {len(fh)} HIPPOC samples\n" + comp.sort_values("refs4h", ascending=False).to_string())
neu = fh.mean().get("Neurons", 0.0); other = fh.mean().drop("Neurons", errors="ignore").max()
verdict = ("WORKS" if neu >= 0.50 and other <= 0.30 else
           "FAILS" if neu < 0.30 or other > 0.40 else "PARTIAL")
print(f"  §13 criterion: Neurons {neu:.3f}, largest non-neuronal {fh.mean().drop('Neurons', errors='ignore').idxmax()} "
      f"{other:.3f} -> {verdict}")

# (2) NNLS proxy on the BayesPrism gene set
bg = pd.read_csv(BU / "bulk_genes.tsv", header=None, sep="\t")[0].astype(str).values
B = sio.mmread(BU / "bulk.mtx").tocsr()
if B.shape[0] == len(bg): B = B.T.tocsr()
B = pd.DataFrame(B.toarray(), columns=bg)
bp = pd.read_csv(RESH / "pred_z/genes.txt", header=None)[0].astype(str).values
common = [g for g in bp if g in set(genes) and g in set(bg)]
gi = pd.Index(genes).get_indexer(common)
Y = B[common]; Y = (Y.T / Y.sum(1)).T * 1e6


def prof(X, lab):
    X = X[:, gi]
    P = pd.DataFrame({t: np.asarray(X[lab == t].sum(0)).ravel() for t in sorted(set(lab))}, index=common)
    return P / P.sum() * 1e6


def fit(P):
    W = np.array([nnls(np.sqrt(P.values), np.sqrt(Y.iloc[i].values))[0] for i in range(len(Y))])
    return pd.Series((W / W.sum(1, keepdims=True)).mean(0), index=P.columns)


PH = prof(XH, mh.cell_type.values); P4m = prof(X4, mh.cell_type.values)
print("\n(2) NNLS proxy (sqrt scale, BayesPrism gene set):\n" +
      pd.DataFrame({"refs4 merged, uncorrected": fit(P4m), "refs4h corrected": fit(PH)}).round(3).to_string())

# (3) ambient estimator (script 5) and correlations, before and after
soup = pd.Series(np.asarray(X3[junk][:, gi].sum(0)).ravel(), index=common); soup = soup / soup.sum() * 1e6
lb, ls = np.log1p(Y.mean()), np.log1p(soup)


def rho_est(P):
    r = {}
    for t in P.columns:
        other = P.drop(columns=t).max(1)
        cand = [g for g in soup.sort_values(ascending=False).index[:2000] if other[g] >= 20 * (P.loc[g, t] + 1) or
                (other[g] > 200 and P.loc[g, t] < 0.05 * other[g])][:100]
        r[t] = min(0.95, P.loc[cand, t].sum() / soup[cand].sum()) if cand else 0.0
    return pd.Series(r)


tab = pd.DataFrame({"ambient before": rho_est(P4m), "ambient after": rho_est(PH),
                    "r(soup) before": [np.corrcoef(np.log1p(P4m[t]), ls)[0, 1] for t in P4m.columns],
                    "r(soup) after": [np.corrcoef(np.log1p(PH[t]), ls)[0, 1] for t in PH.columns],
                    "r(bulk) before": [np.corrcoef(np.log1p(P4m[t]), lb)[0, 1] for t in P4m.columns],
                    "r(bulk) after": [np.corrcoef(np.log1p(PH[t]), lb)[0, 1] for t in PH.columns]}).round(2)
print(f"\n(3) per merged type (soup vs bulk r = {np.corrcoef(ls, lb)[0, 1]:.2f}):\n" + tab.to_string())
