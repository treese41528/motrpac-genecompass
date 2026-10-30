"""HIPPOC: (a) gene-ID mismatch between bulk and reference? (b) choroid plexus in the bulk?"""
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

ROOT = Path("/depot/reese18/apps/motrpac-genecompass")
D = ROOT / "data/deconvolution"
REF = D / "references_v4/HIPPOC_GSE295314"
BULK = D / "motrpac_bulk/HIPPOC"
RES = D / "results_refs4/motrpac/HIPPOC"
info = pd.read_csv(ROOT / "data/references/biomart/rat_gene_info.tsv", sep="\t", dtype=str)
sym = dict(zip(info["Gene stable ID"], info["Gene name"]))

genes = pd.read_csv(REF / "genes.tsv", header=None)[0].astype(str).values
X = sio.mmread(REF / "reference_counts.mtx").tocsr()
if X.shape[0] == len(genes):
    X = X.T.tocsr()
rtot = pd.Series(np.asarray(X.sum(0)).ravel(), index=genes)
rcpm = rtot / rtot.sum() * 1e6
bg = pd.read_csv(BULK / "bulk_genes.tsv", header=None, sep="\t")[0].astype(str).values
B = sio.mmread(BULK / "bulk.mtx").tocsr()
if B.shape[0] == len(bg):
    B = B.T.tocsr()
bt = pd.Series(np.asarray(B.sum(0)).ravel(), index=bg)
bcpm = bt / bt.sum() * 1e6
bp = set(pd.read_csv(RES / "pred_z/genes.txt", header=None)[0].astype(str))

# (a) symbols present in both under DIFFERENT IDs, or a well-expressed bulk gene with ~0 reference counts
print(f"ref genes {len(genes)}, bulk genes {len(bg)}, shared IDs {len(set(genes) & set(bg))}, BayesPrism {len(bp)}")
inbp = [g for g in bp if g in bcpm.index and g in rcpm.index]
z = pd.DataFrame({"sym": [sym.get(g, "") for g in inbp], "bulk_cpm": bcpm[inbp].values, "ref_cpm": rcpm[inbp].values}, index=inbp)
dead = z[(z.bulk_cpm >= 50) & (z.ref_cpm < 1)].sort_values("bulk_cpm", ascending=False)
print(f"\n(a) BayesPrism genes with bulk CPM >= 50 but reference CPM < 1: {len(dead)} "
      f"(bulk CPM mass {dead.bulk_cpm.sum()/1e4:.1f}%)")
print(dead.head(40).round(1).to_string())
# same symbol under another ID in the reference?
s2ref = {}
for g in genes:
    s = sym.get(g)
    if s: s2ref.setdefault(s, []).append(g)
alt = [(g, r.sym, [(a, round(rcpm[a], 1)) for a in s2ref.get(r.sym, []) if a != g]) for g, r in dead.iterrows()]
alt = [a for a in alt if a[2]]
print(f"\n  of those, symbol also present in the reference under another ID: {len(alt)}")
for a in alt[:20]:
    print("  ", a)
print(f"  reference genes with zero counts in all nuclei: {(rtot == 0).sum()}")

# (b) choroid plexus / ependymal / other non-hippocampal markers in the bulk vs the reference
cp = ["Ttr", "Kl", "Folr1", "Clic6", "Kcnj13", "Otx2", "Sostdc1", "Prlr", "Aqp1", "F5", "Enpp2", "Foxj1", "Rarres2",
      "Pmch", "Hcrt", "Avp", "Oxt", "Hbb", "Hba-a1", "Ptgds"]
rows = []
for s in cp:
    for g in s2ref.get(s, []) + [g for g in bg if sym.get(g) == s and g not in s2ref.get(s, [])]:
        rows.append({"sym": s, "id": g, "bulk_cpm": round(bcpm.get(g, np.nan), 1), "ref_cpm": round(rcpm.get(g, np.nan), 1),
                     "in_BP": g in bp})
print("\n(b) choroid plexus / other markers\n", pd.DataFrame(rows).to_string(index=False))
