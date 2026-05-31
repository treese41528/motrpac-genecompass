#!/usr/bin/env python3
"""
make_pseudobulk.py -- Build known-truth pseudobulk mixtures to validate the rat
BayesPrism pipeline (Stage 8, validation V0/V1).

Two modes:
  --mode holdout  (V0 self-consistency): split ONE study's cells stratified by
      cell type into a reference split (1-frac) and a mixture split (frac), build
      the reference from the reference split, draw mixtures from the (disjoint)
      mixture split. Reference and mixtures never share cells.
  --mode cross    (V1 cross-dataset):    draw mixtures from a DIFFERENT study and
      deconvolve against an existing reference (--ref-dir). Tests whether the
      reference generalizes across cohorts -- the closest proxy to the real
      MoTrPAC task.

Mixtures are heterogeneous: each of N synthetic samples gets a Dirichlet-drawn
cell-type proportion vector; we sample that many cells (with replacement) and SUM
their raw counts. Ground truth = the RNA/count fraction per cell type (what
BayesPrism's theta estimates), not the cell fraction -- both are recorded.

Label harmonization: a cell's consensus_label maps to a reference cell type by
exact match; fine immune subtypes (T/B/NK/monocyte/DC/...) collapse to the
reference's "*immune*" bucket; anything not mappable to a reference type is
dropped (and reported) -- BayesPrism requires complete enumeration.

Output (project-local):
  <out>/reference/   (holdout mode only) the disjoint reference
  <out>/mixtures/
    pseudobulk_counts.mtx     mixtures x genes integer, genes aligned to reference
    pseudobulk_genes.tsv      gene order of the matrix columns
    true_fractions.tsv        per mixture: rna_frac + cell_frac per cell type
    harmonization_map.tsv     source consensus_label -> reference cell type, n cells
    pool_composition.tsv      mappable cells per reference cell type in the pool

Usage (project venv):
  # V0
  python deconvolution/make_pseudobulk.py --mode holdout --study GSE220075 \
      --tissue liver --holdout-frac 0.30 --out deconvolution/validation/V0_liver
  # V1
  python deconvolution/make_pseudobulk.py --mode cross --source-study GSE285476 \
      --tissue liver --conditions "syngeneic group,healthy control group" \
      --ref-dir deconvolution/reference/liver_GSE220075 \
      --out deconvolution/validation/V1_liver
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.sparse as sp

from build_reference import (PROJECT, load_study, clean_cells, export_reference)

# Substrings (lowercased, no word boundaries -> robust to plurals) that mark a
# cell as immune; collapsed to the reference's coarse immune bucket.
IMMUNE_SUBSTR = ["t cell", "b cell", "nk cell", "nkt", "natural killer", "monocyte",
                 "macrophage", "dendritic", "lymphocyte", "plasma cell", "plasmablast",
                 "mast cell", "neutrophil", "granulocyte", "leukocyte", "myeloid",
                 "cytotoxic", "treg", "regulatory t", "t helper", "th17"]
_STOP = {"hepatic", "cell", "cells"}


def _ref_keywords(ref_types):
    """Distinctive keyword(s) per reference type, e.g. 'Endothelial cells' -> ['endothelial']."""
    kw = {}
    for t in ref_types:
        toks = [w for w in re.split(r"[^a-z0-9]+", t.lower()) if w and w not in _STOP]
        kw[t] = [w[:-1] if w.endswith("s") and len(w) > 3 else w for w in toks]  # singularize
    return kw


def build_harmonization(source_labels, ref_types):
    """source consensus_label -> reference cell type.
    Priority: exact match -> reference-type keyword (handles synonyms/subtypes,
    e.g. 'Hepatic endothelial cells' -> 'Endothelial cells', 'Metabolic hepatocytes'
    -> 'Hepatocytes') -> immune synonym -> coarse immune bucket -> None (dropped)."""
    immune_target = next((t for t in ref_types if "immune" in t.lower()), None)
    ref_set = set(ref_types)
    kw = _ref_keywords(ref_types)
    mapping = {}
    for lab in source_labels:
        ll = str(lab).lower()
        if lab in ref_set:
            mapping[lab] = lab
            continue
        hit = None
        for t, words in kw.items():
            if t == immune_target:                 # immune handled via synonyms below
                continue
            if any(w in ll for w in words):
                hit = t
                break
        if hit is None and immune_target and ("immune" in ll
                                              or any(s in ll for s in IMMUNE_SUBSTR)):
            hit = immune_target
        mapping[lab] = hit                          # None -> unmappable, dropped
    return mapping


def generate_mixtures(pool, types, rng, n_mix, cells_per_mix, alpha, return_z=False):
    """Dirichlet-proportioned pseudobulk. Returns (counts mixtures x genes, true_frac DataFrame).

    Ground truth recorded two ways: cellfrac__ = fraction of CELLS of each type;
    rnafrac__ = fraction of total COUNTS contributed by each type (this is what
    BayesPrism's theta estimates, so it's the primary scoring target).

    If return_z, also returns z_true: dict cell_type -> (n_mix x n_genes) array of
    the summed counts contributed by that type's cells per mixture -- the ground-
    truth per-cell-type expression Z (sum over types == the mixture total). The rng
    draw order is IDENTICAL whether or not return_z, so replaying with the same
    seed reproduces the exact same mixtures (and hence a Z aligned to the saved
    pseudobulk that BayesPrism deconvolved). See deconvolution/compute_true_z.py.
    """
    X = sp.csr_matrix(pool.X)                        # cells x genes
    n_genes = X.shape[1]
    type_idx = {t: np.where(pool.obs["ref_type"].values == t)[0] for t in types}
    rows, recs = [], []
    z_true = {t: np.zeros((n_mix, n_genes)) for t in types} if return_z else None
    for i in range(n_mix):
        frac = rng.dirichlet(np.full(len(types), alpha))
        n_per = (frac * cells_per_mix).round().astype(int)
        if n_per.sum() == 0:
            n_per[rng.integers(len(types))] = cells_per_mix
        picks_by_type = {}
        for t, n in zip(types, n_per):
            if n == 0:
                continue
            pt = type_idx[t]
            picks_by_type[t] = rng.choice(pt, size=int(n), replace=len(pt) < n)
        mix_counts = np.asarray(
            X[np.concatenate(list(picks_by_type.values()))].sum(axis=0)).ravel()
        total = mix_counts.sum() or 1
        n_total = int(n_per.sum())
        rec = {"mixture": f"mix{i+1}"}
        for t, n in zip(types, n_per):
            rec[f"cellfrac__{t}"] = n / n_total
        for t in types:
            tvec = (np.asarray(X[picks_by_type[t]].sum(axis=0)).ravel()
                    if t in picks_by_type else np.zeros(n_genes))
            rec[f"rnafrac__{t}"] = float(tvec.sum()) / total
            if return_z:
                z_true[t][i] = tvec
        rows.append(mix_counts)
        recs.append(rec)
    counts = sp.csr_matrix(np.vstack(rows).astype(np.int64))
    if return_z:
        return counts, pd.DataFrame(recs), z_true
    return counts, pd.DataFrame(recs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["holdout", "cross"], required=True)
    ap.add_argument("--tissue", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--study", help="holdout mode: the study to split")
    ap.add_argument("--source-study", help="cross mode: mixture source study")
    ap.add_argument("--ref-dir", help="cross mode: existing reference dir")
    ap.add_argument("--conditions", help="comma-sep condition_resolved filter (source)")
    ap.add_argument("--sex", help="cross mode: sex_resolved filter (e.g. male/female)")
    ap.add_argument("--holdout-frac", type=float, default=0.30)
    ap.add_argument("--n-mixtures", type=int, default=50)
    ap.add_argument("--cells-per-mixture", type=int, default=1000)
    ap.add_argument("--alpha", type=float, default=1.0, help="Dirichlet concentration")
    ap.add_argument("--min-pool-cells", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    (out / "mixtures").mkdir(parents=True, exist_ok=True)
    conds = [c.strip() for c in args.conditions.split(",")] if args.conditions else None

    # ---- obtain mixture-pool cells + reference cell types/genes ----
    if args.mode == "holdout":
        assert args.study, "--study required for holdout mode"
        adata = clean_cells(load_study(args.study, args.tissue))
        # stratified split by cell_type
        is_mix = np.zeros(adata.n_obs, dtype=bool)
        for t, idx in adata.obs.groupby("cell_type").indices.items():
            idx = np.array(idx); rng.shuffle(idx)
            k = int(round(len(idx) * args.holdout_frac))
            is_mix[idx[:k]] = True
        ref_adata = clean_cells(adata[~is_mix].copy())     # re-apply >=20/state on the split
        export_reference(ref_adata, out / "reference")
        ref_types = sorted(ref_adata.obs["cell_type"].unique())
        ref_genes = list(map(str, ref_adata.var_names))
        pool = adata[is_mix].copy()
        pool.obs["ref_type"] = pool.obs["cell_type"]       # same vocabulary
    else:
        assert args.source_study and args.ref_dir, "--source-study and --ref-dir required for cross"
        ref_meta = pd.read_csv(Path(args.ref_dir) / "cells_meta.tsv", sep="\t")
        ref_types = sorted(ref_meta["cell_type"].unique())
        ref_genes = [l.strip() for l in open(Path(args.ref_dir) / "genes.tsv")]
        pool = clean_cells(load_study(args.source_study, args.tissue, conds, args.sex))
        hmap = build_harmonization(sorted(pool.obs["cell_type"].unique()), ref_types)
        pool.obs["ref_type"] = pool.obs["cell_type"].map(hmap)
        hm = (pool.obs.groupby("cell_type")
              .agg(n_cells=("ref_type", "size"),
                   ref_type=("ref_type", "first")).reset_index())
        hm.to_csv(out / "mixtures" / "harmonization_map.tsv", sep="\t", index=False)
        pool = pool[pool.obs["ref_type"].notna()].copy()
        # No gene alignment: new.prism intersects reference & mixture by name
        # (BayesPrism README FAQ #6), so the mixture keeps its native genes.

    # ---- restrict to reference types with enough pool cells ----
    comp = pool.obs["ref_type"].value_counts()
    comp.to_csv(out / "mixtures" / "pool_composition.tsv", sep="\t", header=["n_cells"])
    types = [t for t in ref_types if comp.get(t, 0) >= args.min_pool_cells]
    dropped = [t for t in ref_types if t not in types]
    print(f"\nreference types: {ref_types}")
    print(f"types usable in pool (>= {args.min_pool_cells} cells): {types}")
    if dropped:
        print(f"  [note] reference types absent/too-rare in mixture pool (not benchmarked): {dropped}")
    if len(types) < 2:
        sys.exit("ERROR: fewer than 2 usable cell types in the mixture pool.")

    # ---- generate + write ----
    counts, truth = generate_mixtures(pool, types, rng, args.n_mixtures,
                                      args.cells_per_mixture, args.alpha)
    sio.mmwrite(str(out / "mixtures" / "pseudobulk_counts.mtx"), counts)
    pd.Series(list(map(str, pool.var_names))).to_csv(
        out / "mixtures" / "pseudobulk_genes.tsv", index=False, header=False)
    truth.to_csv(out / "mixtures" / "true_fractions.tsv", sep="\t", index=False)
    print(f"\nWrote {counts.shape[0]} mixtures x {counts.shape[1]} genes; "
          f"{len(types)} benchmarked types -> {out}/mixtures/")


if __name__ == "__main__":
    main()
