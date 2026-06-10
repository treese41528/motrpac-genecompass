#!/usr/bin/env python3
"""
build_pseudocells.py -- Stage 8: the deconvolution -> GeneCompass connector.

Turns BayesPrism per-cell-type expression (extract_z.R -> pred_z/) into a
"pseudo single-cell" AnnData that the GeneCompass tokenizer (pipeline/05_tokenization)
consumes -- the bridge from MoTrPAC bulk to the fine-tuned rat GeneCompass (grant
Aim 1/2: "deconvolution ... to generate pseudo single-cell profiles from bulk RNA-seq").

Granularity: ONE pseudo-cell per (bulk sample x cell type). Its expression vector is
the portion of that sample's bulk counts BayesPrism attributes to that cell type
(get.exp, sample x gene). This preserves per-sample variation needed for the Aim 2
tissue x sex x timepoint contrasts (aggregating per group would discard it).

Genes are rel-113 ENSRNOG -- the liftover (prepare_motrpac_bulk.R) guarantees the bulk,
and therefore Z, lives in the same vocab the tokenizer keys on. RAW Z (count-scaled) is
stored in .X and layers['z_raw']; per-pseudo-cell library normalization is a tokenization
concern applied downstream (tokenize_pseudocells.py), so this stays a faithful record of
the deconvolution output.

Output:
  <out>/pseudocells.h5ad   obs: pseudocell_id, sample, cell_type, tissue [+ merged meta]
                           var_names = ENSRNOG ; X = raw Z (cells x genes, float32, CSR)
  <out>/summary.txt

Usage (project venv):
  python deconvolution/build_pseudocells.py \
      --pred-z-dir deconvolution/validation/V0_liver/results/pred_z \
      --tissue liver --out deconvolution/genecompass_input/liver
  optional metadata join (real bulk): --meta-tsv pheno.tsv --meta-key viallabel
"""
import argparse
import re
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp


def _safe(s):
    """Mirror extract_z.R's safe(): gsub('[^A-Za-z0-9]+', '_', s) -- the predz filename key."""
    return re.sub(r"[^A-Za-z0-9]+", "_", str(s))


def load_pred_z(pred_z_dir):
    """Read a pred_z/ dir -> (genes, types, {type: DataFrame[samples x genes]})."""
    d = Path(pred_z_dir)
    genes = [g.strip() for g in (d / "genes.txt").read_text().splitlines() if g.strip()]
    types = [t.strip() for t in (d / "types.txt").read_text().splitlines() if t.strip()]
    if not genes or not types:
        sys.exit(f"ERROR: empty genes.txt/types.txt in {d}")
    mats = {}
    for t in types:
        f = d / f"predz__{_safe(t)}.csv"
        if not f.exists():
            sys.exit(f"ERROR: missing {f.name} for cell type {t!r}")
        df = pd.read_csv(f, index_col=0)
        missing = [g for g in genes if g not in df.columns]
        if missing:
            sys.exit(f"ERROR: {f.name} is missing {len(missing)} genes listed in genes.txt")
        mats[t] = df.reindex(columns=genes)        # enforce identical gene axis/order
    return genes, types, mats


def build_pseudocells(pred_z_dir, tissue, meta=None, meta_key="sample"):
    """pred_z/ -> (AnnData of pseudo-cells, n_dropped_zero, n_samples, n_types)."""
    genes, types, mats = load_pred_z(pred_z_dir)
    samples = list(next(iter(mats.values())).index)
    for t, df in mats.items():
        if list(df.index) != samples:
            sys.exit(f"ERROR: sample axis of cell type {t!r} differs from the others")

    blocks, obs_rows = [], []
    for t in types:                                 # stack (sample x gene) blocks, one per type
        blocks.append(mats[t].to_numpy(dtype=np.float32))
        for s in samples:
            obs_rows.append({
                "pseudocell_id": f"{tissue}|{s}|{_safe(t)}",
                "sample": str(s), "cell_type": t, "tissue": tissue,
            })
    X = np.vstack(blocks)
    obs = pd.DataFrame(obs_rows)

    nz = np.asarray(X.sum(axis=1)).ravel() > 0      # drop pseudo-cells with no signal
    n_drop = int((~nz).sum())
    X, obs = X[nz], obs.loc[nz].reset_index(drop=True)

    if meta is not None:
        if meta_key not in meta.columns:
            sys.exit(f"ERROR: --meta-key {meta_key!r} not a column of the metadata tsv")
        meta = meta.copy()
        meta[meta_key] = meta[meta_key].astype(str)
        obs = obs.merge(meta, how="left", left_on="sample", right_on=meta_key,
                        suffixes=("", "_meta"))
        n_unmatched = int(obs[meta_key].isna().sum()) if meta_key in obs else 0
        if n_unmatched:
            print(f"  [warn] {n_unmatched}/{len(obs)} pseudo-cells had no metadata match")

    adata = ad.AnnData(X=sp.csr_matrix(X), obs=obs.reset_index(drop=True))
    adata.var_names = pd.Index(genes, name="ensrnog")
    adata.obs_names = obs["pseudocell_id"].to_numpy()
    adata.layers["z_raw"] = adata.X.copy()
    adata.uns["pseudocell_build"] = {
        "tissue": tissue, "n_samples": len(samples), "n_cell_types": len(types),
        "granularity": "sample_x_celltype", "values": "raw BayesPrism get.exp (count-scaled)",
        "gene_space": "rel-113 ENSRNOG",
    }
    return adata, n_drop, len(samples), len(types)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred-z-dir", required=True, help="extract_z.R output dir (genes.txt/types.txt/predz__*.csv)")
    ap.add_argument("--tissue", required=True)
    ap.add_argument("--out", required=True, help="output dir (writes pseudocells.h5ad + summary.txt)")
    ap.add_argument("--meta-tsv", help="optional per-sample metadata tsv to join into obs (e.g. PHENO: sex/group/timepoint)")
    ap.add_argument("--meta-key", default="sample", help="join column in --meta-tsv matching the bulk sample id (default 'sample')")
    args = ap.parse_args()

    meta = pd.read_csv(args.meta_tsv, sep="\t", dtype=str) if args.meta_tsv else None
    adata, n_drop, n_s, n_t = build_pseudocells(args.pred_z_dir, args.tissue, meta, args.meta_key)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(out / "pseudocells.h5ad")
    ct = adata.obs["cell_type"].value_counts()
    with open(out / "summary.txt", "w") as fh:
        fh.write(f"pseudocells={adata.n_obs} genes={adata.n_vars} "
                 f"(from {n_s} samples x {n_t} cell types; {n_drop} all-zero dropped)\n")
        fh.write(f"tissue={args.tissue} gene_space=rel-113_ENSRNOG values=raw_get.exp\n\n")
        fh.write("pseudo-cells per cell type:\n")
        fh.write(ct.to_string())
    print(f"pseudocells: {adata.n_obs} cells x {adata.n_vars} genes "
          f"({n_s} samples x {n_t} types, {n_drop} all-zero dropped)")
    print(ct.to_string())
    print(f"\nWrote -> {out}/pseudocells.h5ad")


if __name__ == "__main__":
    main()
