#!/usr/bin/env python
"""measure_untokenizable_share.py -- Step 5, M4: how much of each tissue's pseudo-cell counts falls on genes that
cannot be tokenized (audit finding 2.6; docs/audit/remediation/STEP5_PREREG.md, M4).

A gene reaches the model only if it has a token and a median (the "eligible" set the tokenizer builds). For every
tissue of a pseudo-cell tree: the genes in the pseudo-cells, how many are eligible, and the share of counts on the
ones that are not: pooled over the tissue's pseudo-cells, and the median and maximum over single pseudo-cells.
Also listed: the not-eligible genes with the most counts. Read-only; no threshold (registered).

Usage: measure_untokenizable_share.py <genecompass_input_root> <out_dir>
"""
import os
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = Path(os.environ.setdefault("PIPELINE_ROOT", str(Path(__file__).resolve().parents[2])))
for sub in ("lib", "pipeline/05_tokenization"):
    sys.path.insert(0, str(ROOT / sub))
from gene_utils import load_config, resolve_path                                        # noqa: E402
from tokenize_corpus import (build_eligible_gene_arrays, build_gene_index, load_median_dict,   # noqa: E402
                             load_token_dict, map_varnames_to_eligible)


def main(gc_root, out):
    cfg = load_config(); paths = cfg["paths"]
    token_dict = load_token_dict(Path(resolve_path(cfg, paths["ortholog_dir"])))        # as tokenize_pseudocells.py
    median_dict = load_median_dict(Path(resolve_path(cfg, paths["median_dir"])))
    eligible_ids, _, _ = build_eligible_gene_arrays(token_dict, median_dict)
    gene_index = build_gene_index(eligible_ids)
    print(f"tokens {len(token_dict):,}; medians {len(median_dict):,}; eligible (both) {len(eligible_ids):,}")
    rows, top = [], []
    for d in sorted(p for p in Path(gc_root).iterdir() if (p / "pseudocells.h5ad").exists()):
        a = ad.read_h5ad(d / "pseudocells.h5ad")
        X = a.X.tocsr() if sp.issparse(a.X) else sp.csr_matrix(a.X)
        col_idx, _ = map_varnames_to_eligible(list(a.var_names), gene_index)
        elig = np.zeros(a.n_vars, bool); elig[np.asarray(col_idx, int)] = True
        tot = np.asarray(X.sum(axis=1)).ravel().astype(float)
        off = np.asarray(X[:, ~elig].sum(axis=1)).ravel().astype(float)
        share = np.divide(off, tot, out=np.zeros_like(off), where=tot > 0)
        names = np.array(a.var_names)[~elig]
        has_tok = np.array([g in token_dict for g in names]); has_med = np.array([g in median_dict for g in names])
        rows.append({"tissue": d.name.upper(), "pseudo_cells": a.n_obs, "genes": a.n_vars, "genes_eligible": int(elig.sum()),
                     "genes_not_eligible": int((~elig).sum()), "not_eligible_no_token": int((~has_tok).sum()) if len(names) else 0,
                     "not_eligible_token_but_no_median": int((has_tok & ~has_med).sum()) if len(names) else 0,
                     "share_of_counts_pooled": float(off.sum() / tot.sum()) if tot.sum() else 0.0,
                     "share_median_pseudo_cell": float(np.median(share)), "share_max_pseudo_cell": float(share.max())})
        if len(names):
            g = np.asarray(X[:, ~elig].sum(axis=0)).ravel()
            for i in np.argsort(-g)[:10]:
                if g[i] > 0:
                    top.append({"tissue": d.name.upper(), "gene": names[i], "counts": float(g[i]), "share_of_tissue_counts": float(g[i] / tot.sum())})
        print(rows[-1], flush=True)
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "untokenizable_share_by_tissue.tsv", sep="\t", index=False)
    pd.DataFrame(top, columns=["tissue", "gene", "counts", "share_of_tissue_counts"]).to_csv(
        out / "untokenizable_top_genes.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main(*sys.argv[1:3])
