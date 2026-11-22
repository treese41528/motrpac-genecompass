#!/usr/bin/env python
"""measure_id_symbol_drift.py -- Step 5, M3: do the corpus's Ensembl IDs still name the genes their files meant?

Audit finding 1.5: an Ensembl ID from an older release is accepted as the same gene whenever the ID exists in
release 115. This script measures how often such an ID carries a different gene symbol in release 115 than in the
file it came from. Read-only. The reading rule is in docs/audit/remediation/STEP5_PREREG.md (M3), written first.

For every kept corpus sample whose source is a 10x matrix keyed by rat Ensembl ID, the features file gives
(ID, symbol). For the IDs that entered the QC file, the file's symbol is compared with release 115's:
  same         equal, ignoring case
  rename       they differ and at least one is a placeholder (LOC..., a clone name, an RGD number, empty)
  different    they differ and both are real names: a possible reassignment
Shares are given over genes and over the sample's counts (column sums of the QC file).

Writes id_symbol_by_sample.tsv, id_symbol_by_study.tsv, id_symbol_overall.tsv and id_symbol_conflicts.tsv (every
"different" pair with the samples and counts it touches).

Usage: measure_id_symbol_drift.py <out_dir> [--jobs N]
"""
import argparse
import csv
import gzip
import json
import os
import re
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

ROOT = Path(os.environ.setdefault("PIPELINE_ROOT", str(Path(__file__).resolve().parents[2])))
sys.path.insert(0, str(ROOT / "lib"))
from gene_utils import load_config, raw_counts_adata, resolve_path   # noqa: E402

PLACEHOLDER = re.compile(r"^(|LOC\d+|RGD\d+|AABR\d+\.\d+|A[CL]\d+\.\d+|AY\d+\.\d+|ENSRNOG\d+|.*Rik|Gm\d+|NEWGENE_\d+|U\d+|Y_RNA|7SK|5S_rRNA|5_8S_rRNA|Metazoa_SRP|SNOR.*|Mir\d.*|mir-\d.*)$", re.I)


def features_file(source):
    """The 10x features/genes table that belongs to a matrix file."""
    p = Path(source)
    d, name = p.parent, p.name
    cands = []
    for a, b in ((".matrix.mtx", ".genes.tsv"), (".matrix.mtx", ".features.tsv"), ("_matrix.mtx", "_genes.tsv"),
                 ("_matrix.mtx", "_features.tsv"), ("matrix.mtx", "genes.tsv"), ("matrix.mtx", "features.tsv")):
        for nm in (name, name.replace(".gz", "")):
            if a in nm:
                base = nm.replace(".gz", "").replace(a, b)
                cands += [d / base, d / (base + ".gz")]
    for c in cands:
        if c.is_file():
            return c
    return None


def read_features(path):
    op = gzip.open if str(path).endswith(".gz") else open
    ids, syms = [], []
    with op(path, "rt") as fh:
        for ln in fh:
            f = ln.rstrip("\n").split("\t")
            if len(f) < 2:
                return None
            ids.append(f[0].split(".")[0]); syms.append(f[1])
    return pd.Series(syms, index=ids)


def classify(a, b):
    if a.lower() == b.lower():
        return "same"
    if PLACEHOLDER.match(a) or PLACEHOLDER.match(b):
        return "rename"
    return "different"


def one(sample_id, source, qc_path, sym115):
    ff = features_file(source)
    if ff is None:
        return {"sample_id": sample_id, "status": "no features file"}, []
    fs = read_features(ff)
    if fs is None:
        return {"sample_id": sample_id, "status": "features file has one column"}, []
    fs = fs[~fs.index.duplicated()]
    a = ad.read_h5ad(qc_path); a, _ = raw_counts_adata(a)
    tot = pd.Series(np.asarray(a.X.sum(axis=0)).ravel().astype(float), index=a.var_names)
    ids = [g for g in tot.index if g in fs.index and g in sym115.index]
    cls = pd.Series([classify(fs[g], sym115[g]) for g in ids], index=ids)
    w = tot[ids]
    row = {"sample_id": sample_id, "study": sample_id.split("_sample")[0], "status": "ok",
           "features_file": str(ff), "n_genes_compared": len(ids), "counts_compared": float(w.sum())}
    for k in ("same", "rename", "different"):
        m = cls == k
        row[f"genes_{k}"] = int(m.sum()); row[f"counts_{k}"] = float(w[m].sum())
    conf = [(g, fs[g], sym115[g], float(w[g])) for g in cls.index[cls == "different"]]
    return row, conf


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("out_dir"); ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    qc_dir = resolve_path(cfg, cfg["paths"]["qc_h5ad_dir"])
    keep = {r["sample_id"] for r in csv.DictReader(open(resolve_path(cfg, cfg["paths"]["qc_file_list"])), delimiter="\t")
            if r["action"].startswith("keep")}
    rep = json.load(open(ROOT / "data/training/preprocessed/preprocessing_report.json"))
    todo, other = [], 0
    for st in rep["studies"]:
        for s in st.get("samples", []):
            if s["sample_id"] in keep:
                if s.get("format") == "mtx" and s.get("gene_id_format") == "ensembl_rat":
                    todo.append((s["sample_id"], s["source"], qc_dir / f"{s['sample_id']}.h5ad"))
                else:
                    other += 1
    gi = pd.read_csv(ROOT / "data/references/biomart/rat_gene_info.tsv", sep="\t", dtype=str).fillna("")
    sym115 = pd.Series(gi["Gene name"].values, index=gi["Gene stable ID"].values); sym115 = sym115[~sym115.index.duplicated()]
    print(f"{len(todo)} kept samples are 10x matrices keyed by rat Ensembl ID; {other} kept samples have another format "
          f"or key and are not measured", flush=True)
    res = Parallel(n_jobs=a.jobs, verbose=5)(delayed(one)(s, src, q, sym115) for s, src, q in todo)
    rows = pd.DataFrame([r for r, _ in res]); rows.to_csv(out / "id_symbol_by_sample.tsv", sep="\t", index=False)
    ok = rows[rows.status == "ok"]
    print(rows.status.value_counts().to_string(), flush=True)
    num = ["n_genes_compared", "counts_compared"] + [f"{p}_{k}" for p in ("genes", "counts") for k in ("same", "rename", "different")]
    st = ok.groupby("study")[num].sum(); st["samples"] = ok.groupby("study").size()
    for k in ("same", "rename", "different"):
        st[f"gene_share_{k}"] = st[f"genes_{k}"] / st.n_genes_compared
        st[f"count_share_{k}"] = st[f"counts_{k}"] / st.counts_compared
    st.sort_values("count_share_different", ascending=False).to_csv(out / "id_symbol_by_study.tsv", sep="\t")
    t = ok[num].sum()
    o = {"samples_measured": len(ok), "samples_not_measurable": int((rows.status != "ok").sum()), "samples_other_format": other,
         **{f"gene_share_{k}": float(t[f"genes_{k}"] / t.n_genes_compared) for k in ("same", "rename", "different")},
         **{f"count_share_{k}": float(t[f"counts_{k}"] / t.counts_compared) for k in ("same", "rename", "different")},
         "max_study_count_share_different": float(st.count_share_different.max()),
         "studies_above_0.5pct_counts_different": int((st.count_share_different > 0.005).sum()), "studies": len(st)}
    pd.Series(o).to_csv(out / "id_symbol_overall.tsv", sep="\t", header=["value"]); print(pd.Series(o).to_string(), flush=True)
    conf, by_study = {}, {}
    for (r, cs) in res:
        for g, fsym, s115, w in cs:
            c = conf.setdefault((g, fsym, s115), [0, 0.0, set()]); c[0] += 1; c[1] += w; c[2].add(r["study"])
            k = (r["study"], g, fsym, s115); by_study[k] = by_study.get(k, 0.0) + w
    pd.DataFrame([{"study": k[0], "ensembl_id": k[1], "file_symbol": k[2], "release115_symbol": k[3], "counts": v}
                  for k, v in by_study.items()]).to_csv(out / "id_symbol_conflicts_by_study.tsv.gz", sep="\t", index=False)
    cf = pd.DataFrame([{"ensembl_id": g, "file_symbol": f, "release115_symbol": s, "n_samples": v[0], "counts": v[1],
                        "n_studies": len(v[2]), "studies": ",".join(sorted(v[2])[:8])} for (g, f, s), v in conf.items()])
    if len(cf):
        cf.sort_values("counts", ascending=False).to_csv(out / "id_symbol_conflicts.tsv", sep="\t", index=False)
        print(cf.sort_values("counts", ascending=False).head(25).to_string(index=False))


if __name__ == "__main__":
    main()
