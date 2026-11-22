#!/usr/bin/env python
"""measure_bulk_id_drift.py -- Step 5, M3 for the MoTrPAC bulk: IDs that pass the liftover unchanged but mean another gene.

prepare_motrpac_bulk.R lifts the bulk's gene IDs (an older rat annotation) to release 115. An ID that already exists
in release 115 passes through unchanged ("direct"). This script checks those IDs the same way the corpus IDs are
checked (classify_id_symbol_conflicts.py): the bulk's own symbol for the ID (MoTrPAC FEATURE_TO_GENE, recorded in
deconvolution/reference/motrpac_bulk_liftover.tsv) against release 115's, and whether the token's ortholog is the
gene the bulk meant. Counts are the bulk's, summed over the 16 deconvolved tissues.

Also lists the pre-registered positive-control genes (deconvolution/reference/posctrl_prereg_refs4.tsv) whose
symbol the bulk carries under an ID that release 115 gives to another gene or does not have.

Usage: measure_bulk_id_drift.py <out_dir>
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import mmread

ROOT = Path(__file__).resolve().parents[2]
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
B = ROOT / "data/references/biomart"
lm = pd.read_csv(ROOT / "deconvolution/reference/motrpac_bulk_liftover.tsv", sep="\t", dtype=str).fillna("")
gi = pd.read_csv(B / "rat_gene_info.tsv", sep="\t", dtype=str).fillna("")
sym115 = dict(zip(gi["Gene stable ID"], gi["Gene name"]))
tm = pd.read_csv(ROOT / "data/training/ortholog_mappings/rat_token_mapping.tsv", sep="\t", dtype=str).fillna("").set_index("rat_gene")
hs = pd.read_csv(B / "human_all_genes.tsv", sep="\t", dtype=str).fillna(""); mm = pd.read_csv(B / "mouse_all_genes.tsv", sep="\t", dtype=str).fillna("")
hsym = dict(zip(hs["Gene stable ID"], hs["Gene name"])); msym = dict(zip(mm["Gene stable ID"], mm["Gene name"]))
human_symbols = {s.upper() for s in hs["Gene name"] if s}
syn = {}
for f in ("human_gene_synonyms.tsv", "mouse_gene_synonyms.tsv"):
    t = pd.read_csv(B / f, sep="\t", dtype=str).fillna("")
    for g, s in zip(t["Gene stable ID"], t["Gene Synonym"]):
        if s:
            syn.setdefault(g, set()).add(s.upper())
PLACEHOLDER = re.compile(r"^(|LOC\d+|RGD\d+|AABR\d+\.\d+|A[CL]\d+\.\d+|AY\d+\.\d+|ENSRNOG\d+|.*Rik|Gm\d+|NEWGENE_\d+|U\d+|Y_RNA|7SK|5S_rRNA|5_8S_rRNA|Metazoa_SRP|SNOR.*|Mir\d.*|mir-\d.*)$", re.I)
MT = set(gi.loc[gi["Chromosome/scaffold name"] == "MT", "Gene stable ID"])


def norm(s):
    return s.upper().replace("-", "")


def classify(gid, old):
    new = sym115.get(gid, "")
    if norm(old) == norm(new) or gid in MT:
        return "same symbol"
    if PLACEHOLDER.match(old) or PLACEHOLDER.match(new):
        return "rename (placeholder)"
    if gid not in tm.index:
        return "not in the token table"
    r = tm.loc[gid]
    names = [(hsym.get(r["human_ortholog"], ""), r["human_ortholog"])] if r["human_ortholog"] else []
    names += [(msym.get(r["mouse_ortholog"], ""), r["mouse_ortholog"])] if r["mouse_ortholog"] else []
    if not names:
        return "alignment lost" if old.upper() in human_symbols else "no ortholog"
    for nm, g in names:
        if norm(nm) == norm(old) or old.upper() in syn.get(g, ()):
            return "same gene, renamed"
    return "other gene"


# bulk counts per lifted ID over the 16 deconvolved tissues
res_root = ROOT / "data/deconvolution/results_refs4all_rebuild/motrpac"
tissues = sorted(p.name for p in res_root.iterdir() if p.is_dir())
tot = {}
for t in tissues:
    bg = [l.strip() for l in open(ROOT / f"data/deconvolution/motrpac_bulk/{t}/bulk_genes.tsv")]
    s = np.asarray(mmread(ROOT / f"data/deconvolution/motrpac_bulk/{t}/bulk.mtx").tocsc().sum(axis=0)).ravel()
    for g, v in zip(bg, s):
        tot[g] = tot.get(g, 0.0) + float(v)
grand = sum(tot.values())

d = lm[lm.method == "direct"].copy()
d["release115_symbol"] = d.lifted_id.map(sym115).fillna("")
d["class"] = [classify(g, s) for g, s in zip(d.lifted_id, d.symbol)]
d["token_tier"] = [tm.loc[g, "tier"] if g in tm.index else "" for g in d.lifted_id]
d["token_ortholog"] = [(hsym.get(tm.loc[g, "human_ortholog"], "") or msym.get(tm.loc[g, "mouse_ortholog"], "")) if g in tm.index else "" for g in d.lifted_id]
d["bulk_counts"] = d.lifted_id.map(tot).fillna(0.0)
d["share_of_bulk_counts"] = d.bulk_counts / grand
d.sort_values("bulk_counts", ascending=False).to_csv(out / "bulk_direct_ids_classified.tsv", sep="\t", index=False)
s = d.groupby("class").agg(ids=("lifted_id", "size"), bulk_counts=("bulk_counts", "sum"))
s["share_of_bulk_counts"] = s.bulk_counts / grand
s.loc["ALL direct"] = [len(d), d.bulk_counts.sum(), d.bulk_counts.sum() / grand]
s.to_csv(out / "bulk_direct_id_classes.tsv", sep="\t")
pd.set_option("display.width", 220)
print(f"liftover methods: {lm.method.value_counts().to_dict()}; bulk counts over {len(tissues)} tissues: {grand:.3g}")
print(s.round(5).to_string())
for k in ("other gene", "alignment lost"):
    print(f"\n{k}: the 20 largest by bulk counts")
    print(d[d["class"] == k].sort_values("bulk_counts", ascending=False).head(20)[
        ["lifted_id", "symbol", "release115_symbol", "token_ortholog", "token_tier", "share_of_bulk_counts"]].round(5).to_string(index=False))

# positive controls: is each control's gene in the bulk under the ID the specification names?
pc = pd.read_csv(ROOT / "deconvolution/reference/posctrl_prereg_refs4.tsv", sep="\t", dtype=str).fillna("")
symcol = "symbol" if "symbol" in pc.columns else pc.columns[2]
idcol = "ensembl" if "ensembl" in pc.columns else None
old_by_sym = {}
for g, sy, lid in zip(lm.old_id, lm.symbol, lm.lifted_id):
    if sy:
        old_by_sym.setdefault(sy.upper(), []).append((g, lid))
rows = []
for sy in sorted(set(pc[symcol])):
    spec_id = pc.loc[pc[symcol] == sy, idcol].iloc[0] if idcol else ""
    hits = old_by_sym.get(sy.upper(), [])
    lifted = sorted({l for _, l in hits if l})
    rows.append({"symbol": sy, "id_in_specification": spec_id, "spec_id_has_bulk_counts": tot.get(spec_id, 0.0) > 0,
                 "bulk_ids_carrying_this_symbol": ",".join(lifted), "same_id": spec_id in lifted,
                 "bulk_counts_under_those_ids": sum(tot.get(l, 0.0) for l in lifted),
                 "release115_symbol_of_those_ids": ",".join(sym115.get(l, "") for l in lifted)})
r = pd.DataFrame(rows); r.to_csv(out / "posctrl_symbol_ids_in_bulk.tsv", sep="\t", index=False)
bad = r[~r.same_id]
print(f"\npositive-control genes: {len(r)}; the bulk carries the symbol under another ID than the specification's, or not at all: {len(bad)}")
print(bad.to_string(index=False))
