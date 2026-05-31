#!/usr/bin/env python3
"""compute_true_z.py -- Ground-truth per-cell-type expression Z for a pseudobulk
validation stage (Stage 8 Z scoring), by replaying make_pseudobulk's mixture
generation with the same seed/args.

Cross mode (V1/V2): rebuilds the source pool exactly as make_pseudobulk does, then
regenerates the mixtures and VERIFIES reproduction two ways before trusting the
ground truth:
  1. per-type pool cell counts == saved mixtures/pool_composition.tsv
  2. regenerated pseudobulk counts == saved mixtures/pseudobulk_counts.mtx
Only if both pass do we write the per-sample true Z (so a wrong seed/filters can
never silently produce a misaligned ground truth).

Writes under <out>/mixtures/true_z/:
  truez__<safe_type>.npy   per type: (n_mix x n_genes) summed counts of that type
  genes.txt                gene order (== pseudobulk_genes.tsv)
  types.txt                benchmarked cell types (original names)
  gep_true.tsv             aggregated GEP per type = pool-mean profile (seed-free)

Usage mirrors make_pseudobulk cross mode, e.g.:
  python deconvolution/compute_true_z.py --source-study GSE137869 --tissue liver \
     --sex male --ref-dir deconvolution/reference/liver_GSE220075 \
     --out deconvolution/validation/V2_GSE137869_male
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.sparse as sp

from build_reference import load_study, clean_cells
from make_pseudobulk import build_harmonization, generate_mixtures


def safe(s):
    return re.sub(r"[^A-Za-z0-9]+", "_", s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-study", required=True)
    ap.add_argument("--tissue", required=True)
    ap.add_argument("--ref-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--conditions", help="comma-sep condition_resolved filter (source)")
    ap.add_argument("--sex", help="sex_resolved filter (e.g. male/female)")
    ap.add_argument("--n-mixtures", type=int, default=50)
    ap.add_argument("--cells-per-mixture", type=int, default=1000)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--min-pool-cells", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    out = Path(args.out)
    mixd = out / "mixtures"
    conds = [c.strip() for c in args.conditions.split(",")] if args.conditions else None
    rng = np.random.default_rng(args.seed)

    # ---- rebuild the cross-mode source pool exactly as make_pseudobulk.main() ----
    ref_meta = pd.read_csv(Path(args.ref_dir) / "cells_meta.tsv", sep="\t")
    ref_types = sorted(ref_meta["cell_type"].unique())
    pool = clean_cells(load_study(args.source_study, args.tissue, conds, args.sex))
    hmap = build_harmonization(sorted(pool.obs["cell_type"].unique()), ref_types)
    pool.obs["ref_type"] = pool.obs["cell_type"].map(hmap)
    pool = pool[pool.obs["ref_type"].notna()].copy()
    comp = pool.obs["ref_type"].value_counts()
    types = [t for t in ref_types if comp.get(t, 0) >= args.min_pool_cells]

    # ---- verify 1: pool composition matches what the stage was built from ----
    saved_comp = pd.read_csv(mixd / "pool_composition.tsv", sep="\t",
                             index_col=0)["n_cells"].astype(int)
    rebuilt = comp.reindex(saved_comp.index).fillna(0).astype(int)
    if not rebuilt.equals(saved_comp):
        print("[FAIL] pool composition mismatch -- wrong study/tissue/conditions/sex.")
        print(pd.DataFrame({"saved": saved_comp, "rebuilt": rebuilt}))
        sys.exit(1)
    print(f"[ok] pool composition matches saved ({int(saved_comp.sum())} cells, "
          f"{len(types)} benchmarked types).")

    # ---- regenerate mixtures + per-sample Z ----
    counts, _truth, z_true = generate_mixtures(
        pool, types, rng, args.n_mixtures, args.cells_per_mixture, args.alpha,
        return_z=True)

    # ---- verify 2: regenerated counts reproduce the saved pseudobulk bit-for-bit ----
    saved = sio.mmread(str(mixd / "pseudobulk_counts.mtx")).tocsr().astype(np.int64)
    if saved.shape != counts.shape or (saved != counts).nnz != 0:
        print(f"[FAIL] regenerated counts differ from saved (seed wrong?). "
              f"saved={saved.shape} regen={counts.shape}")
        sys.exit(2)
    print(f"[ok] regenerated pseudobulk counts match saved (seed={args.seed}).")

    # ---- write ground truth ----
    genes = [l.strip() for l in open(mixd / "pseudobulk_genes.tsv")]
    zdir = mixd / "true_z"
    zdir.mkdir(parents=True, exist_ok=True)
    for t in types:
        np.save(zdir / f"truez__{safe(t)}.npy", z_true[t].astype(np.float64))
    (zdir / "genes.txt").write_text("\n".join(genes) + "\n")
    (zdir / "types.txt").write_text("\n".join(types) + "\n")

    # aggregated GEP per type = pool-mean expression profile (seed-free reference)
    X = sp.csr_matrix(pool.X)
    gep = {t: np.asarray(X[np.where(pool.obs["ref_type"].values == t)[0]]
                         .mean(axis=0)).ravel() for t in types}
    pd.DataFrame(gep, index=genes).to_csv(zdir / "gep_true.tsv", sep="\t")
    print(f"Wrote true Z ({len(types)} types x {len(genes)} genes) -> {zdir}/")


if __name__ == "__main__":
    main()
