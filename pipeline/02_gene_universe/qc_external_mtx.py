#!/usr/bin/env python3
"""qc_external_mtx.py — Stage 2 QC for a 10x matrix that was not part of the Stage 2 run.

Studies staged after the corpus was built (e.g. OMIX767 testis from NGDC) never passed through
preprocess_training_matrices.py, so their gene IDs were never resolved into the corpus gene universe.
OMIX767's genes.tsv carries IDs from an older Ensembl release: 3,242 of its genes (12.9% of counts)
are not in the universe and would be silently dropped downstream.

This script applies the Stage 2 steps to one 10x triplet (filtered matrices only):
  1. resolve genes into the corpus gene universe (Ensembl 115 / GRCr8). Old-release stable IDs are
     NOT trusted on their own: Ensembl reassigns some between releases (OMIX767's ENSRNOG00000000471 is
     B3galt4; in 115 that ID is Rps18 and B3galt4 is ENSRNOG00000071248). So, per source gene:
       a. its symbol, if that symbol names exactly one Ensembl 115 gene (case-insensitive)
       b. else its own ID, if the ID exists in 115 and no gene from step a already claimed it
       c. else dropped
     and only universe genes are kept. Source genes landing on one ID share its symbol (or one took
     the ID by step b) and are summed, as Stage 2 does. Stage 1's gene_resolution.tsv is not used for
     this: it only knows names seen in the original corpus.
  2. mito genes identified by symbol, carried to the resolved IDs
  3. normal_filter -> gene_number_filter -> min_genes_filter -> min_cells_per_sample (Stage 2 functions)
  4. annotate_cell_stats, write <qc_h5ad_dir>/<sample_id>.h5ad (raw counts in X) + <sample_id>.qc.json

Usage:
  python pipeline/02_gene_universe/qc_external_mtx.py \
      --mtx-dir data/raw/ngdc/datasets/OMIX767/Gene_bc_matrices/C_gene_bc_matrices \
      --sample-id OMIX767_grcr8_sampleC
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "lib"))
from gene_utils import BioMartReference, load_config, resolve_path  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "preprocess_training_matrices", Path(__file__).with_name("preprocess_training_matrices.py"))
s2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s2)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--mtx-dir", required=True, help="10x triplet dir (matrix.mtx, barcodes.tsv, genes.tsv)")
    ap.add_argument("--sample-id", required=True, help="output file stem in qc_h5ad_dir")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    config = load_config()
    paths = config["paths"]
    pp = config["gene_universe"].get("preprocessing", {})
    min_genes = int(pp.get("min_genes_per_cell", 200))
    min_cells = int(pp.get("min_cells_per_sample", 4))
    qc_dir = resolve_path(config, paths["qc_h5ad_dir"])
    out = qc_dir / f"{args.sample_id}.h5ad"
    if out.exists() and not args.overwrite:
        sys.exit(f"{out} exists; pass --overwrite to replace it")

    d = Path(args.mtx_dir)
    adata = s2._load_10x_mtx_manual(d / "matrix.mtx", d / "barcodes.tsv", d / "genes.tsv")
    genes = pd.read_csv(d / "genes.tsv", sep="\t", header=None, dtype=str)
    ids, symbols = genes[0].tolist(), (genes[1] if genes.shape[1] > 1 else genes[0]).tolist()

    mapper = s2.GeneUniverseMapper(resolve_path(config, paths["gene_universe_dir"]))
    bm = BioMartReference.from_config(config)
    by_symbol = {}
    for eid, info in bm.gene_info.items():
        by_symbol.setdefault(info["symbol"].lower(), set()).add(eid)
    unique_symbol = {k: next(iter(v)) for k, v in by_symbol.items() if len(v) == 1}

    target, how = [None] * len(ids), [None] * len(ids)
    for i, sym in enumerate(symbols):                                     # a. symbol
        t = unique_symbol.get(sym.lower())
        if t is not None:
            target[i], how[i] = t, "symbol"
    claimed = {t for t in target if t is not None}
    for i, gid in enumerate(ids):                                         # b. own ID
        if target[i] is None and gid in bm.gene_info and gid not in claimed:
            target[i], how[i] = gid, "id"
    for i in range(len(ids)):
        if target[i] is None:
            how[i] = "unresolved"
        elif target[i] not in mapper.universe_ids:
            target[i], how[i] = None, "not_in_universe"

    counts_in = float(adata.X.sum())
    gene_counts = np.asarray(adata.X.sum(0)).ravel()
    resolved_by = {h: {"genes": how.count(h),
                       "counts_pct": round(100 * float(gene_counts[[j for j, x in enumerate(how) if x == h]].sum())
                                           / counts_in, 3)}
                   for h in ("symbol", "id", "unresolved", "not_in_universe")}
    mito_targets = {t for t, sym in zip(target, symbols) if t is not None and s2.identify_mito_genes([sym])}

    keep = [i for i, t in enumerate(target) if t is not None]
    kept_ids = [target[i] for i in keep]
    uniq = sorted(set(kept_ids))
    col = {t: j for j, t in enumerate(uniq)}
    n_merged = len(kept_ids) - len(uniq)
    # summing source genes per target = X[:, keep] @ (indicator source -> target)
    G = sp.csr_matrix((np.ones(len(keep)), (np.arange(len(keep)), [col[t] for t in kept_ids])),
                      shape=(len(keep), len(uniq)))
    X = sp.csr_matrix(adata.X)[:, keep] @ G
    mapped = ad.AnnData(X=sp.csr_matrix(X), obs=adata.obs.copy(), var=pd.DataFrame(index=uniq))
    mstats = {"total_input_genes": len(ids), "mapped": len(keep), "unique_targets": len(set(kept_ids)),
              "source_genes_merged_away": n_merged}
    mito = mito_targets
    n0 = mapped.n_obs
    a = s2.normal_filter(mapped, mito); n1 = a.n_obs
    a = s2.gene_number_filter(a, mapper.core_ids); n2 = a.n_obs
    a = s2.min_genes_filter(a, min_genes=min_genes); n3 = a.n_obs
    if a.n_obs < min_cells:
        sys.exit(f"only {a.n_obs} cells after QC (< {min_cells})")
    s2.annotate_cell_stats(a, mito)
    a.obs["sample_id"] = args.sample_id
    a.obs_names = [f"{args.sample_id}_{b}" for b in a.obs_names]

    report = {
        "sample_id": args.sample_id, "source": str(d),
        "genes_in": len(ids), "resolved_by": resolved_by, "mapping_stats": mstats,
        "genes_out": int(a.n_vars), "mito_genes": sorted(mito),
        "counts_kept_after_mapping_pct": round(100 * float(mapped.X.sum()) / counts_in, 3),
        "cells": {"raw": n0, "normal_filter": n1, "gene_number_filter": n2, "min_genes": n3},
        "params": {"min_genes_per_cell": min_genes, "min_cells_per_sample": min_cells},
    }
    qc_dir.mkdir(parents=True, exist_ok=True)
    a.write_h5ad(out)
    (qc_dir / f"{args.sample_id}.qc.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "mito_genes"}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
