#!/usr/bin/env python
"""classify_id_symbol_conflicts.py -- Step 5, M3, second pass: which symbol conflicts change the token a gene gets?

measure_id_symbol_drift.py lists every Ensembl ID whose symbol in a corpus file differs from its release-115
symbol. A different symbol is not yet a different gene: release 115 renamed many loci. What matters for the model
is the token. Each rat ID gets its token from release 115's orthology (data/training/ortholog_mappings/
rat_token_mapping.tsv), so the question for each conflicting ID is whether the ortholog behind its token is the
gene the file meant.

  same gene        the token's human (or mouse) ortholog carries the file's symbol, or the file's symbol is a
                   recorded synonym of it: release 115 only renamed the rat locus
  other gene       the token belongs to the ortholog of a different gene: the file's counts for one gene enter
                   the model as another gene's token
  alignment lost   the ID has no ortholog in release 115 (a new rat-only token), although the file's symbol names
                   a gene that has a human ortholog: the counts are kept but lose their cross-species token
  no ortholog      no ortholog in release 115 and none by the file's symbol either

Usage: classify_id_symbol_conflicts.py <ids_dir>     (the output folder of measure_id_symbol_drift.py)
"""
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
d = Path(sys.argv[1])
conf = pd.read_csv(d / "id_symbol_conflicts.tsv", sep="\t", dtype={"file_symbol": str, "release115_symbol": str}).fillna("")
samples = pd.read_csv(d / "id_symbol_by_sample.tsv", sep="\t")
total = samples.loc[samples.status == "ok", "counts_compared"].sum()

tm = pd.read_csv(ROOT / "data/training/ortholog_mappings/rat_token_mapping.tsv", sep="\t", dtype=str).fillna("").set_index("rat_gene")
B = ROOT / "data/references/biomart"
hs = pd.read_csv(B / "human_all_genes.tsv", sep="\t", dtype=str).fillna("")
mm = pd.read_csv(B / "mouse_all_genes.tsv", sep="\t", dtype=str).fillna("")
hsym = dict(zip(hs["Gene stable ID"], hs["Gene name"])); msym = dict(zip(mm["Gene stable ID"], mm["Gene name"]))
human_symbols = {s.upper() for s in hs["Gene name"] if s}
syn = {}
for f, col in (("human_gene_synonyms.tsv", "Gene Synonym"), ("mouse_gene_synonyms.tsv", "Gene Synonym")):
    t = pd.read_csv(B / f, sep="\t", dtype=str).fillna("")
    for gid, s in zip(t["Gene stable ID"], t[col]):
        if s:
            syn.setdefault(gid, set()).add(s.upper())


gi = pd.read_csv(B / "rat_gene_info.tsv", sep="\t", dtype=str).fillna("")
MT_IDS = set(gi.loc[gi["Chromosome/scaffold name"] == "MT", "Gene stable ID"])
SUFFIX = re.compile(r"_(m?rat|rn\d*|rnor\d*|mratbn[\d.]*)$", re.I)      # multi-species references tag every symbol


def clean(sym):
    """A file's symbol without a species tag (GSE254371 writes Tpt1_mrat)."""
    return SUFFIX.sub("", sym)


def norm(s):
    return clean(s).upper().replace("-", "")


def classify(row):
    g, fsym = row.ensembl_id, clean(row.file_symbol)
    if g in MT_IDS:                       # COX2, MT-Mt-co2, Mt-cox3 ... : naming conventions of the 37 MT genes
        return "same gene", "mitochondrial", row.release115_symbol
    if norm(fsym) == norm(row.release115_symbol):
        return "same gene", "species tag only", row.release115_symbol
    if g not in tm.index:
        return "not in the token table", "", ""
    r = tm.loc[g]
    h, m = r["human_ortholog"], r["mouse_ortholog"]
    names = [(hsym.get(h, ""), h)] if h else []
    names += [(msym.get(m, ""), m)] if m else []
    if not names:
        return ("alignment lost" if fsym.upper() in human_symbols else "no ortholog"), r["tier"], ""
    for nm, gid in names:
        if norm(nm) == norm(fsym) or fsym.upper() in syn.get(gid, ()):
            return "same gene", r["tier"], nm
    return "other gene", r["tier"], names[0][0]


out = conf.copy()
out[["class", "tier", "token_ortholog_symbol"]] = out.apply(lambda r: pd.Series(classify(r)), axis=1)
out["share_of_compared_counts"] = out.counts / total
out.sort_values("counts", ascending=False).to_csv(d / "id_symbol_conflicts_classified.tsv", sep="\t", index=False)
s = out.groupby("class").agg(pairs=("ensembl_id", "size"), ids=("ensembl_id", "nunique"), counts=("counts", "sum"))
s["share_of_compared_counts"] = s.counts / total
rec = out[out.n_studies >= 5].groupby("class").agg(ids_in_5_or_more_studies=("ensembl_id", "nunique"), c=("counts", "sum"))
s["ids_in_5_or_more_studies"] = rec.ids_in_5_or_more_studies.reindex(s.index).fillna(0).astype(int)
s["share_of_counts_in_5_or_more_studies"] = (rec.c / total).reindex(s.index).fillna(0.0)
s.loc["ALL conflicts"] = [len(out), out.ensembl_id.nunique(), out.counts.sum(), out.counts.sum() / total,
                          out[out.n_studies >= 5].ensembl_id.nunique(), out[out.n_studies >= 5].counts.sum() / total]
s.to_csv(d / "id_symbol_conflict_classes.tsv", sep="\t")

# per study: the share of its compared counts in each class (the registered reading is per study)
bs = d / "id_symbol_conflicts_by_study.tsv.gz"
if bs.exists():
    b = pd.read_csv(bs, sep="\t", dtype={"file_symbol": str, "release115_symbol": str}).fillna("")
    key = ["ensembl_id", "file_symbol", "release115_symbol"]
    b = b.merge(out[key + ["class"]], on=key, how="left")
    tot = samples[samples.status == "ok"].groupby("study").counts_compared.sum()
    ps = b.pivot_table(index="study", columns="class", values="counts", aggfunc="sum", fill_value=0.0)
    ps = ps.div(tot.reindex(ps.index), axis=0)
    ps["n_samples"] = samples[samples.status == "ok"].groupby("study").size().reindex(ps.index)
    ps.sort_values("other gene", ascending=False).to_csv(d / "id_symbol_classes_by_study.tsv", sep="\t")
    print("\nper study, share of counts under another gene's token: median %.4f, max %.4f (%s); studies above 0.5%%: %d of %d"
          % (ps["other gene"].median(), ps["other gene"].max(), ps["other gene"].idxmax(), int((ps["other gene"] > 0.005).sum()), len(ps)))
    print("per study, share of counts that lost their cross-species token: median %.4f, max %.4f"
          % (ps["alignment lost"].median(), ps["alignment lost"].max()))
pd.set_option("display.width", 220)
print(s.round(5).to_string())
for k in ("other gene", "alignment lost"):
    t = out[out["class"] == k].sort_values("counts", ascending=False).head(25)
    print(f"\n{k}: the 25 largest by counts")
    print(t[["ensembl_id", "file_symbol", "release115_symbol", "token_ortholog_symbol", "tier", "n_samples", "n_studies",
             "share_of_compared_counts"]].round(5).to_string(index=False))
