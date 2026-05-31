#!/usr/bin/env python3
"""make_purity_sweep.py -- Replicate BayesPrism Fig. 1h on rat data: pseudobulk
mixtures that sweep ONE focal cell type's fraction from low to high purity, so we
can measure its inferred-expression accuracy AS A FUNCTION OF its fraction (the
paper's design) instead of at the balanced Dirichlet mix V0/V1/V2 used.

Paper (Chu 2022, Nat Cancer, Fig. 1h): malignant-cell expression correlation vs
known ground truth, plotted against malignant fraction; reported ">0.95 for
tumors with >50% purity", Pearson on DESeq2-VST values. Our focal type =
Hepatocytes (the dominant parenchymal type, the rat analog of "malignant").

Two designs (mirror the paper + our real task):
  --mode holdout : focal + background drawn from the SAME study as the reference,
      with a disjoint reference split (paper-faithful: focal is in-reference).
  --mode cross   : focal + background drawn from a DIFFERENT study, deconvolved
      against an existing fixed reference (our actual MoTrPAC setting; harder).

For each target purity p in --purity-grid, --reps mixtures are drawn: the focal
type contributes round(p * cells_per_mixture) cells; the remaining (1-p) is split
among the background types by a Dirichlet draw. Ground-truth per-cell-type
expression (true Z) is saved AT GENERATION (no replay needed) -- both as the
focal type's sample x gene CSV (for the R/VST scorer) and as per-type .npy.

Outputs (under --out):
  mixtures/pseudobulk_counts.mtx, pseudobulk_genes.tsv     (samples x genes)
  mixtures/true_fractions.tsv         per mixture: cellfrac__/rnafrac__ per type
  mixtures/sweep_design.tsv           mixture, target_purity, rep, focal_type
  mixtures/true_z/truez__<type>.npy   per-type ground-truth Z (samples x genes)
  mixtures/true_z_focal.csv           focal type true Z (samples x genes; for R)
  mixtures/harmonization_map.tsv      (cross mode) source label -> ref type
  reference/                          (holdout mode) the disjoint reference

Then: run BayesPrism (run_deconvolution.sh) -> extract_z.sh (pred Z) ->
score_z_vst.sh (Pearson-VST + Spearman per sample) -> score_purity_sweep.py
(bin by focal fraction, compare to the paper's >0.95@>50% point).

Usage:
  # paper-faithful (same study as reference atlas)
  python deconvolution/make_purity_sweep.py --mode holdout --study GSE220075 \
      --tissue liver --focal-type Hepatocytes \
      --out deconvolution/validation/SWEEP_hepato_holdout
  # cross-dataset (our real setting)
  python deconvolution/make_purity_sweep.py --mode cross --source-study GSE245240 \
      --tissue liver --conditions Nave --focal-type Hepatocytes \
      --ref-dir deconvolution/reference/liver_GSE220075 \
      --out deconvolution/validation/SWEEP_hepato_cross
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.sparse as sp

from build_reference import load_study, clean_cells, export_reference
from make_pseudobulk import build_harmonization


def generate_purity_sweep(pool, types, focal, rng, purity_grid, reps,
                          cells_per_mix, bg_alpha):
    """Mixtures sweeping the focal type's fraction. Returns
    (counts samples x genes, truth DataFrame, design DataFrame, z_true dict)."""
    if focal not in types:
        sys.exit(f"ERROR: focal type {focal!r} not among usable types {types}")
    background = [t for t in types if t != focal]
    if not background:
        sys.exit("ERROR: need >=1 background type besides the focal type.")
    X = sp.csr_matrix(pool.X)
    n_genes = X.shape[1]
    type_idx = {t: np.where(pool.obs["ref_type"].values == t)[0] for t in types}

    rows, recs, design = [], [], []
    z_true = {t: [] for t in types}
    mix_i = 0
    for p in purity_grid:
        for r in range(reps):
            mix_i += 1
            n_focal = int(round(p * cells_per_mix))
            n_focal = min(max(n_focal, 0), cells_per_mix)
            n_bg_total = cells_per_mix - n_focal
            n_per = {focal: n_focal}
            if n_bg_total > 0:
                bg_frac = rng.dirichlet(np.full(len(background), bg_alpha))
                bg_counts = (bg_frac * n_bg_total).round().astype(int)
                # fix rounding drift so the total is exact
                drift = n_bg_total - int(bg_counts.sum())
                if drift != 0:
                    bg_counts[rng.integers(len(background))] += drift
                for t, n in zip(background, bg_counts):
                    n_per[t] = int(n)
            else:
                for t in background:
                    n_per[t] = 0

            picks = {}
            for t in types:
                n = n_per[t]
                if n <= 0:
                    continue
                pt = type_idx[t]
                picks[t] = rng.choice(pt, size=n, replace=len(pt) < n)

            mix_counts = np.asarray(
                X[np.concatenate(list(picks.values()))].sum(axis=0)).ravel()
            total = mix_counts.sum() or 1
            n_total = int(sum(n_per.values())) or 1

            rec = {"mixture": f"mix{mix_i}"}
            for t in types:
                rec[f"cellfrac__{t}"] = n_per[t] / n_total
            for t in types:
                tvec = (np.asarray(X[picks[t]].sum(axis=0)).ravel()
                        if t in picks else np.zeros(n_genes))
                rec[f"rnafrac__{t}"] = float(tvec.sum()) / total
                z_true[t].append(tvec)
            rows.append(mix_counts)
            recs.append(rec)
            design.append({"mixture": f"mix{mix_i}", "target_purity": p,
                           "rep": r + 1, "focal_type": focal,
                           "focal_cellfrac": n_per[focal] / n_total,
                           "focal_rnafrac": rec[f"rnafrac__{focal}"]})

    counts = sp.csr_matrix(np.vstack(rows).astype(np.int64))
    z_true = {t: np.vstack(v) for t, v in z_true.items()}
    return counts, pd.DataFrame(recs), pd.DataFrame(design), z_true


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["holdout", "cross"], required=True)
    ap.add_argument("--tissue", required=True)
    ap.add_argument("--focal-type", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--study", help="holdout: study to split (ref + mixture pool)")
    ap.add_argument("--source-study", help="cross: mixture source study")
    ap.add_argument("--ref-dir", help="cross: existing reference dir")
    ap.add_argument("--conditions", help="comma-sep condition_resolved filter (source)")
    ap.add_argument("--sex", help="sex_resolved filter (e.g. male/female)")
    ap.add_argument("--purity-grid", default="0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9",
                    help="comma-sep focal-type target fractions to sweep")
    ap.add_argument("--reps", type=int, default=10, help="mixtures per purity point")
    ap.add_argument("--cells-per-mixture", type=int, default=1000)
    ap.add_argument("--bg-alpha", type=float, default=1.0,
                    help="Dirichlet concentration for the background split")
    ap.add_argument("--holdout-frac", type=float, default=0.30)
    ap.add_argument("--min-pool-cells", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    (out / "mixtures").mkdir(parents=True, exist_ok=True)
    conds = [c.strip() for c in args.conditions.split(",")] if args.conditions else None
    grid = [float(x) for x in args.purity_grid.split(",")]

    if args.mode == "holdout":
        assert args.study, "--study required for holdout mode"
        adata = clean_cells(load_study(args.study, args.tissue))
        is_mix = np.zeros(adata.n_obs, dtype=bool)
        for t, idx in adata.obs.groupby("cell_type").indices.items():
            idx = np.array(idx); rng.shuffle(idx)
            k = int(round(len(idx) * args.holdout_frac))
            is_mix[idx[:k]] = True
        ref_adata = clean_cells(adata[~is_mix].copy())
        export_reference(ref_adata, out / "reference")
        ref_types = sorted(ref_adata.obs["cell_type"].unique())
        pool = adata[is_mix].copy()
        pool.obs["ref_type"] = pool.obs["cell_type"]
    else:
        assert args.source_study and args.ref_dir, "--source-study and --ref-dir required for cross"
        ref_meta = pd.read_csv(Path(args.ref_dir) / "cells_meta.tsv", sep="\t")
        ref_types = sorted(ref_meta["cell_type"].unique())
        pool = clean_cells(load_study(args.source_study, args.tissue, conds, args.sex))
        hmap = build_harmonization(sorted(pool.obs["cell_type"].unique()), ref_types)
        pool.obs["ref_type"] = pool.obs["cell_type"].map(hmap)
        (pool.obs.groupby("cell_type")
            .agg(n_cells=("ref_type", "size"), ref_type=("ref_type", "first"))
            .reset_index()
            .to_csv(out / "mixtures" / "harmonization_map.tsv", sep="\t", index=False))
        pool = pool[pool.obs["ref_type"].notna()].copy()

    comp = pool.obs["ref_type"].value_counts()
    comp.to_csv(out / "mixtures" / "pool_composition.tsv", sep="\t", header=["n_cells"])
    types = [t for t in ref_types if comp.get(t, 0) >= args.min_pool_cells]
    print(f"reference types: {ref_types}")
    print(f"usable types (>= {args.min_pool_cells} pool cells): {types}")
    if args.focal_type not in types:
        sys.exit(f"ERROR: focal type {args.focal_type!r} not usable "
                 f"(pool has {comp.get(args.focal_type, 0)} cells).")
    # sanity: can we reach the top of the grid with this focal pool?
    n_focal_needed = int(round(max(grid) * args.cells_per_mixture))
    n_focal_have = int(comp.get(args.focal_type, 0))
    print(f"focal {args.focal_type!r}: pool={n_focal_have} cells; "
          f"max purity {max(grid)} needs {n_focal_needed}/mix "
          f"({'sampled WITH replacement' if n_focal_have < n_focal_needed else 'sampled without replacement'}).")

    counts, truth, design, z_true = generate_purity_sweep(
        pool, types, args.focal_type, rng, grid, args.reps,
        args.cells_per_mixture, args.bg_alpha)

    md = out / "mixtures"
    sio.mmwrite(str(md / "pseudobulk_counts.mtx"), counts)
    pd.Series(list(map(str, pool.var_names))).to_csv(
        md / "pseudobulk_genes.tsv", index=False, header=False)
    truth.to_csv(md / "true_fractions.tsv", sep="\t", index=False)
    design.to_csv(md / "sweep_design.tsv", sep="\t", index=False)

    genes = list(map(str, pool.var_names))
    zdir = md / "true_z"; zdir.mkdir(exist_ok=True)
    safe = lambda s: __import__("re").sub(r"[^A-Za-z0-9]+", "_", s)
    for t in types:
        np.save(zdir / f"truez__{safe(t)}.npy", z_true[t].astype(np.float64))
    (zdir / "genes.txt").write_text("\n".join(genes) + "\n")
    (zdir / "types.txt").write_text("\n".join(types) + "\n")
    # focal true Z as CSV (rows mix1.., cols genes) for the R/VST scorer
    pd.DataFrame(z_true[args.focal_type], columns=genes,
                 index=[f"mix{i+1}" for i in range(counts.shape[0])]).to_csv(
        md / "true_z_focal.csv")

    print(f"\nWrote {counts.shape[0]} mixtures x {counts.shape[1]} genes "
          f"({len(grid)} purity points x {args.reps} reps) -> {md}/")
    print(f"focal true Z -> {md}/true_z_focal.csv ; per-type Z -> {zdir}/")


if __name__ == "__main__":
    main()
