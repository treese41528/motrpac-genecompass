#!/usr/bin/env python3
"""Record the ID-space membership of every lifted MoTrPAC-bulk gene across the
single-cell CORPUS (all qc_matrices, not just the filtered built references),
the built references, biomart rel-113, and RGD -- so an old/new Ensembl release
mismatch is detectable and reconcilable BY SYMBOL afterward (counts alone hide it).

Writes deconvolution/reference/idspace_audit/:
  corpus_genes.txt, builtref_genes.txt, rgd_ens.txt   -- the raw ID sets
  id2symbol.tsv                                        -- ID -> symbol (biomart+vocab+rgd)
  lifted_membership.tsv  -- per bulk gene: old_id, lifted_id, method, cur_symbol,
                            in_biomart113, in_corpus_byID, in_corpus_bySymbol, in_builtref
  summary.txt            -- the release checks + per-method tallies
Reuses tmp/corpus_genes.txt if present (the slow 864-h5ad union)."""
import anndata as ad, glob, pandas as pd, re, os, warnings
warnings.filterwarnings("ignore")
OUT = "deconvolution/reference/idspace_audit"; os.makedirs(OUT, exist_ok=True)

# --- single-cell corpus gene union (reuse cache if present) ---
cache = "tmp/corpus_genes.txt"
if os.path.exists(cache) and os.path.getsize(cache) > 0:
    corpus = set(x.strip() for x in open(cache) if x.strip())
    print(f"corpus: reused {cache} -> {len(corpus)} genes")
else:
    fs = sorted(glob.glob("data/training/preprocessed/qc_matrices/*.h5ad")); corpus = set(); bad = 0
    for f in fs:
        try:
            corpus |= set(map(str, ad.read_h5ad(f, backed="r").var_names))
        except Exception:
            bad += 1
    print(f"corpus: built from {len(fs)} h5ads ({bad} unreadable) -> {len(corpus)} genes")

# --- reference / annotation sets ---
builtref = set()
for g in glob.glob("deconvolution/reference/*/genes.tsv"):
    builtref |= set(x.strip() for x in open(g) if x.strip())
bm = pd.read_csv("data/references/biomart/rat_gene_info.tsv", sep="\t")
bm_ids = set(bm["Gene stable ID"].dropna())
voc = pd.read_csv("data/training/ortholog_mappings/rat_token_mapping.tsv", sep="\t")
rgd = pd.read_csv("data/references/biomart/GENES_RAT.txt", sep="\t", comment="#",
                  dtype=str, low_memory=False)
rgd["ens"] = rgd["ENSEMBL_ID"].map(lambda s: (re.search(r"ENSRNOG\d+", s).group(0)
                                              if isinstance(s, str) and re.search(r"ENSRNOG\d+", s) else None))
rgd_ids = set(rgd["ens"].dropna())

# --- ID -> symbol, unioned across biomart + vocab + RGD (release-robust reconciliation) ---
id2sym = {}
for k, v in zip(bm["Gene stable ID"], bm["Gene name"].fillna("")):
    if isinstance(v, str) and v: id2sym.setdefault(k, v.upper())
for k, v in zip(voc["rat_gene"], voc["rat_symbol"].fillna("")):
    if isinstance(v, str) and v: id2sym.setdefault(k, v.upper())
rr = rgd.dropna(subset=["ens"])
for k, v in zip(rr["ens"], rr["SYMBOL"].fillna("")):
    if isinstance(v, str) and v: id2sym.setdefault(k, v.upper())
pd.DataFrame({"id": list(id2sym), "symbol": list(id2sym.values())}).to_csv(
    f"{OUT}/id2symbol.tsv", sep="\t", index=False)
corpus_sym = set(filter(None, (id2sym.get(g) for g in corpus)))

# --- per-gene membership of the lifted bulk ---
m = pd.read_csv("deconvolution/reference/motrpac_bulk_liftover.tsv", sep="\t")
lid = m["lifted_id"]
m["cur_symbol"]         = lid.map(lambda x: id2sym.get(x, "") if isinstance(x, str) else "")
m["in_biomart113"]      = lid.isin(bm_ids)
m["in_corpus_byID"]     = lid.isin(corpus)
m["in_builtref"]        = lid.isin(builtref)
m["in_corpus_bySymbol"] = m["cur_symbol"].map(lambda s: bool(s) and s in corpus_sym)
m.to_csv(f"{OUT}/lifted_membership.tsv", sep="\t", index=False)

# --- raw sets ---
for name, s in [("corpus_genes", corpus), ("builtref_genes", builtref), ("rgd_ens", rgd_ids)]:
    pd.Series(sorted(s)).to_csv(f"{OUT}/{name}.txt", index=False, header=False)

# --- summary + release checks ---
def frac(a, b): return f"{len(a & b)}/{len(a)} ({100 * len(a & b) / max(len(a), 1):.1f}%)"
L = [f"corpus={len(corpus)}  builtref={len(builtref)}  biomart113={len(bm_ids)}  rgd={len(rgd_ids)}",
     f"RELEASE CHECK  corpus in biomart113: {frac(corpus, bm_ids)}   (<<100% => corpus is a different Ensembl release)",
     f"RELEASE CHECK  rgd    in biomart113: {frac(rgd_ids, bm_ids)}",
     f"RELEASE CHECK  corpus in rgd       : {frac(corpus, rgd_ids)}"]
for meth in ["direct", "symbol", "id_history"]:
    sub = m[m.method == meth]; ids = set(sub["lifted_id"].dropna())
    L += [f"\n[{meth}]  {len(ids)} unique lifted IDs / {len(sub)} bulk rows:",
          f"   in biomart113      : {frac(ids, bm_ids)}",
          f"   in corpus (by ID)  : {frac(ids, corpus)}",
          f"   in corpus (by SYM) : {int(sub['in_corpus_bySymbol'].sum())}/{len(sub)}  <- release-robust",
          f"   in built refs      : {frac(ids, builtref)}"]
open(f"{OUT}/summary.txt", "w").write("\n".join(L) + "\n")
print("\n".join(L))
print(f"\nwrote {OUT}/  (lifted_membership.tsv + 4 set files + id2symbol.tsv + summary.txt)")
