#!/usr/bin/env python
"""measure_mito_fraction.py -- Step 5, M1: the mitochondrial fraction of every training cell and reference cell.

Audit findings 1.2 and 1.3: no fixed mitochondrial cut was applied to the corpus, and the 3 SD rule on
mitochondrial fraction had no effect in files keyed by Ensembl ID. This script measures what that let through.
Read-only. The decision rules are in docs/audit/remediation/STEP5_PREREG.md (M1) and were written first.

For every QC file the v2 corpus keeps (config/corpus_v2_plan.tsv), each cell's share of raw counts on the 13
mitochondrial protein-coding genes, found by Ensembl ID. Writes
  mito_by_file.tsv        one row per file: cells, cells above each cut, quantiles, the stored obs mito_pct maximum
  mito_by_study.tsv       the same summed per study
  mito_overall.tsv        the corpus totals
  mito_by_reference.tsv   per production reference and cell type (references built from corpus cells)

Usage: measure_mito_fraction.py <out_dir> [--jobs N]
"""
import argparse
import os
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed

ROOT = Path(os.environ.setdefault("PIPELINE_ROOT", str(Path(__file__).resolve().parents[2])))
sys.path.insert(0, str(ROOT / "lib"))
from gene_utils import load_config, qc_corpus_files, raw_counts_adata, resolve_path   # noqa: E402

CUTS = [0.05, 0.10, 0.15, 0.20, 0.50]


def mito_ids():
    gi = pd.read_csv(ROOT / "data/references/biomart/rat_gene_info.tsv", sep="\t", dtype=str).fillna("")
    gi.columns = [c.strip() for c in gi.columns]
    m = gi[(gi["Chromosome/scaffold name"] == "MT") & (gi["Gene type"] == "protein_coding")]
    return sorted(m["Gene stable ID"])


def one_file(path, mt, keep_cells):
    a = ad.read_h5ad(path)
    stored = a.obs["mito_pct"].to_numpy(dtype=float) if "mito_pct" in a.obs else np.full(a.n_obs, np.nan)
    a, used_raw = raw_counts_adata(a)
    X = a.X
    tot = np.asarray(X.sum(axis=1)).ravel().astype(float)
    cols = np.flatnonzero(np.isin(a.var_names, mt))
    mito = np.asarray(X[:, cols].sum(axis=1)).ravel().astype(float)
    frac = np.divide(mito, tot, out=np.zeros_like(mito), where=tot > 0)
    sid = Path(path).stem
    row = {"sample_id": sid, "study": sid.split("_sample")[0], "n_cells": int(a.n_obs), "n_mt_genes_in_file": int(len(cols)),
           "n_mt_genes_with_counts": int((np.asarray(X[:, cols].sum(axis=0)).ravel() > 0).sum()),
           "used_raw_slot": bool(used_raw), "stored_mito_pct_max": float(np.nanmax(stored)) if len(stored) else np.nan,
           "median": float(np.median(frac)), "p90": float(np.quantile(frac, 0.9)), "p99": float(np.quantile(frac, 0.99)),
           "max": float(frac.max())}
    for c in CUTS:
        row[f"n_above_{int(c * 100)}"] = int((frac > c).sum())
    per_cell = None
    if keep_cells:                               # this file feeds a deconvolution reference: keep its cells' fractions
        per_cell = pd.DataFrame({"sample": sid, "barcode": a.obs_names.astype(str), "mito_frac": frac})
    return row, per_cell


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir"); ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    files = qc_corpus_files(cfg)
    mt = mito_ids()
    print(f"{len(files)} kept QC files; {len(mt)} mitochondrial protein-coding genes", flush=True)

    # corpus samples that feed a production reference
    refs = yaml.safe_load(open(ROOT / "deconvolution/tissue_references.yaml"))
    ref_dirs = {}
    for tis, c in refs.items():
        if isinstance(c, dict) and c.get("reference_dir") and c.get("status") == "ready":
            d = ROOT / (c.get("refs4_reference_dir") or c["reference_dir"])      # the builds in use for the default
            if (d / "cells_meta.tsv").exists():
                ref_dirs.setdefault(str(d), []).append(tis)
    ref_samples = set()
    metas = {}
    for d in ref_dirs:
        m = pd.read_csv(Path(d) / "cells_meta.tsv", sep="\t", dtype=str)
        metas[d] = m; ref_samples |= set(m["sample"])
    qc_dir = resolve_path(cfg, cfg["paths"]["qc_h5ad_dir"])
    extra = [qc_dir / f"{s}.h5ad" for s in sorted(ref_samples) if (qc_dir / f"{s}.h5ad").is_file()
             and (qc_dir / f"{s}.h5ad") not in set(files)]
    print(f"{len(ref_samples)} samples feed a reference; {len(extra)} of them are QC files outside the kept corpus list", flush=True)

    res = Parallel(n_jobs=a.jobs, verbose=5)(delayed(one_file)(f, mt, Path(f).stem in ref_samples) for f in files)
    rows = [r for r, _ in res]; cells = [c for _, c in res if c is not None]
    if extra:
        res2 = Parallel(n_jobs=a.jobs)(delayed(one_file)(f, mt, True) for f in extra)
        cells += [c for _, c in res2 if c is not None]
        pd.DataFrame([r for r, _ in res2]).to_csv(out / "mito_by_file_reference_only.tsv", sep="\t", index=False)
    f = pd.DataFrame(rows); f.to_csv(out / "mito_by_file.tsv", sep="\t", index=False)
    num = ["n_cells"] + [f"n_above_{int(c * 100)}" for c in CUTS]
    s = f.groupby("study")[num].sum()
    s["files"] = f.groupby("study").size()
    s["files_with_no_mt_counts"] = f[f.n_mt_genes_with_counts == 0].groupby("study").size().reindex(s.index).fillna(0).astype(int)
    s["files_whose_stored_mito_pct_is_all_zero"] = f[f.stored_mito_pct_max == 0].groupby("study").size().reindex(s.index).fillna(0).astype(int)
    for c in CUTS:
        s[f"share_above_{int(c * 100)}"] = s[f"n_above_{int(c * 100)}"] / s.n_cells
    s.sort_values("n_above_15", ascending=False).to_csv(out / "mito_by_study.tsv", sep="\t")
    tot = f[num].sum()
    o = {"files": len(f), "cells": int(tot.n_cells),
         "files_whose_stored_mito_pct_is_all_zero": int((f.stored_mito_pct_max == 0).sum()),
         "cells_in_those_files": int(f.loc[f.stored_mito_pct_max == 0, "n_cells"].sum()),
         "files_with_no_mt_counts": int((f.n_mt_genes_with_counts == 0).sum())}
    for c in CUTS:
        o[f"cells_above_{int(c * 100)}"] = int(tot[f"n_above_{int(c * 100)}"]); o[f"share_above_{int(c * 100)}"] = float(tot[f"n_above_{int(c * 100)}"] / tot.n_cells)
    pd.Series(o).to_csv(out / "mito_overall.tsv", sep="\t", header=["value"])
    print(pd.Series(o).to_string(), flush=True)

    if cells:
        pc = pd.concat(cells, ignore_index=True)
        rr = []
        for d, tissues in ref_dirs.items():
            m = metas[d].merge(pc, on=["sample", "barcode"], how="left")
            for ct, g in m.groupby("cell_type"):
                k = g.mito_frac.notna()
                rr.append({"reference": Path(d).name, "tissues": ",".join(tissues), "cell_type": ct, "n_cells": len(g),
                           "n_matched_to_a_qc_cell": int(k.sum()),
                           "median_mito": float(g.mito_frac[k].median()) if k.any() else np.nan,
                           **{f"share_above_{int(c * 100)}": float((g.mito_frac[k] > c).mean()) if k.any() else np.nan for c in CUTS}})
        r = pd.DataFrame(rr); r.to_csv(out / "mito_by_reference.tsv", sep="\t", index=False)
        big = r[(r.n_cells >= 100) & (r.share_above_15 >= 0.05)].sort_values("share_above_15", ascending=False)
        print(f"\nreference cell types with at least 100 cells and 5% or more of them above 15%: {len(big)}")
        print(big[["reference", "cell_type", "n_cells", "median_mito", "share_above_15"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
