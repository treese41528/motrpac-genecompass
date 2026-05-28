#!/usr/bin/env python3
"""
annotate_sctype.py -- ScType cell-type annotation for QC'd h5ad corpus

Python implementation of ScType (Ianevski et al., Nat Commun 2022) using the
curated ScTypeDB marker database with both positive and negative markers.

Key difference from PanglaoDB/sc.tl.score_genes approach:
  - Uses NEGATIVE markers to penalize implausible assignments
  - Tissue-specific cell type lists (Heart has 16 types, not 178)
  - Score normalized by sqrt(n_markers)
  - Threshold: score < n_cells_in_cluster/4 → "Unknown"

Usage:
  # Step 1: Build marker database (maps ScTypeDB symbols → ENSRNOG)
  python annotate_sctype.py build-markers

  # Step 2: Test on one sample
  python annotate_sctype.py annotate --sample-id GSE240848_sample0 --tissue Heart

  # Step 3: Compare with PanglaoDB results
  python annotate_sctype.py compare --sample-id GSE240848_sample0 --tissue Heart

  # Step 4: Run on all samples (uses tissue hints from annotation inventory)
  python annotate_sctype.py annotate --all --no-umap

  # Step 5: Summarize
  python annotate_sctype.py summarize
"""

import argparse
import json
import logging
import math
import os
import pickle
import sys
import warnings
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scanpy as sc

warnings.filterwarnings("ignore", category=FutureWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path("/depot/reese18/apps/motrpac-genecompass")
QC_H5AD_DIR = PROJECT_ROOT / "data/training/qc_h5ad"
GENE_UNIVERSE = PROJECT_ROOT / "data/training/gene_universe/gene_universe.tsv"
OUTPUT_DIR = PROJECT_ROOT / "data/training/cell_annotations_sctype"
SCTYPE_DB_PATH = PROJECT_ROOT / "data/training/cell_annotations/ScTypeDB_full.xlsx"
MARKER_DB_PATH = OUTPUT_DIR / "sctype_marker_database.pkl"

LEIDEN_RESOLUTION = 0.8

# ScTypeDB tissue → MoTrPAC tissue mapping
SCTYPE_TO_MOTRPAC = {
    "Heart": "heart",
    "Kidney": "kidney",
    "Liver": "liver",
    "Lung": "lung",
    "Muscle": "gastrocnemius",
    "Brain": "cortex",
    "Hippocampus": "hippocampus",
    "Intestine": "small intestine",
    "Spleen": "spleen",
    "Adrenal": "adrenal",
    "Stomach": "colon",  # closest match
    "Thymus": "spleen",  # closest match
    "Immune system": "blood RNA",
    "Pancreas": None,
    "Eye": None,
    "Placenta": None,
}

# MoTrPAC tissue → ScTypeDB tissue(s) mapping
MOTRPAC_TO_SCTYPE = {
    "heart": ["Heart", "Immune system"],
    "kidney": ["Kidney", "Immune system"],
    "liver": ["Liver", "Immune system"],
    "lung": ["Lung", "Immune system"],
    "gastrocnemius": ["Muscle", "Immune system"],
    "cortex": ["Brain", "Immune system"],
    "hippocampus": ["Hippocampus", "Immune system"],
    "hypothalamus": ["Brain", "Immune system"],
    "small intestine": ["Intestine", "Immune system"],
    "colon": ["Intestine", "Immune system"],
    "spleen": ["Spleen", "Immune system"],
    "blood RNA": ["Immune system"],
    "BAT": ["Immune system"],  # no adipose in ScTypeDB
    "WAT-SC": ["Immune system"],
    "adrenal": ["Adrenal", "Immune system"],
}


# ============================================================================
# Step 1: Build ScType marker database
# ============================================================================
def build_marker_database():
    """Parse ScTypeDB_full.xlsx, map symbols to ENSRNOG, save as pickle."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load gene universe for symbol → ENSRNOG mapping
    logger.info("Loading gene universe...")
    gu = pd.read_csv(GENE_UNIVERSE, sep="\t")
    symbol_to_ensrnog = {}
    for _, row in gu.iterrows():
        sym = str(row["symbol"]).strip()
        eid = str(row["ensembl_id"]).strip()
        if sym and eid and sym != "nan":
            symbol_to_ensrnog[sym.upper()] = eid
    logger.info(f"  {len(symbol_to_ensrnog)} symbols in gene universe")

    # Load ScTypeDB
    logger.info(f"Loading ScTypeDB from {SCTYPE_DB_PATH}...")
    df = pd.read_excel(SCTYPE_DB_PATH)
    logger.info(f"  {len(df)} entries across {df['tissueType'].nunique()} tissues")

    # Parse markers per tissue+celltype
    db = {}  # tissue → {celltype → {positive: [ENSRNOG], negative: [ENSRNOG]}}
    total_pos_mapped = 0
    total_neg_mapped = 0
    total_pos_unmapped = 0
    total_neg_unmapped = 0

    for _, row in df.iterrows():
        tissue = str(row["tissueType"]).strip()
        cell_type = str(row["cellName"]).strip()

        if tissue not in db:
            db[tissue] = {}

        # Parse positive markers
        pos_symbols = []
        pos_str = str(row.get("geneSymbolmore1", "")).strip()
        if pos_str and pos_str != "nan":
            pos_symbols = [s.strip().upper() for s in pos_str.split(",") if s.strip()]

        # Parse negative markers
        neg_symbols = []
        neg_str = str(row.get("geneSymbolmore2", "")).strip()
        if neg_str and neg_str != "nan":
            neg_symbols = [s.strip().upper() for s in neg_str.split(",") if s.strip()]

        # Map to ENSRNOG
        pos_ensrnog = []
        neg_ensrnog = []
        for sym in pos_symbols:
            eid = symbol_to_ensrnog.get(sym)
            if eid:
                pos_ensrnog.append(eid)
                total_pos_mapped += 1
            else:
                total_pos_unmapped += 1

        for sym in neg_symbols:
            eid = symbol_to_ensrnog.get(sym)
            if eid:
                neg_ensrnog.append(eid)
                total_neg_mapped += 1
            else:
                total_neg_unmapped += 1

        db[tissue][cell_type] = {
            "positive": sorted(set(pos_ensrnog)),
            "negative": sorted(set(neg_ensrnog)),
            "pos_symbols": pos_symbols,
            "neg_symbols": neg_symbols,
            "n_pos_mapped": len(pos_ensrnog),
            "n_neg_mapped": len(neg_ensrnog),
            "n_pos_total": len(pos_symbols),
            "n_neg_total": len(neg_symbols),
        }

    # Save
    result = {
        "db": db,
        "symbol_to_ensrnog": symbol_to_ensrnog,
        "build_info": {
            "source": str(SCTYPE_DB_PATH),
            "built_at": datetime.now().isoformat(),
            "total_entries": len(df),
            "tissues": sorted(db.keys()),
            "n_tissues": len(db),
            "total_pos_mapped": total_pos_mapped,
            "total_neg_mapped": total_neg_mapped,
            "total_pos_unmapped": total_pos_unmapped,
            "total_neg_unmapped": total_neg_unmapped,
        },
    }

    with open(MARKER_DB_PATH, "wb") as f:
        pickle.dump(result, f)

    logger.info(f"  Positive markers mapped: {total_pos_mapped} ({total_pos_unmapped} unmapped)")
    logger.info(f"  Negative markers mapped: {total_neg_mapped} ({total_neg_unmapped} unmapped)")
    logger.info(f"  Saved to {MARKER_DB_PATH}")

    # Summary per tissue
    logger.info("\n  ScTypeDB coverage:")
    for tissue in sorted(db.keys()):
        celltypes = db[tissue]
        n_types = len(celltypes)
        has_neg = sum(1 for ct in celltypes.values() if ct["negative"])
        logger.info(f"    {tissue:20s}  {n_types:3d} cell types  ({has_neg} with negative markers)")

    return result


# ============================================================================
# Step 2: ScType scoring algorithm
# ============================================================================
def sctype_score(
    adata: sc.AnnData,
    gs_positive: Dict[str, List[str]],
    gs_negative: Dict[str, List[str]],
) -> pd.DataFrame:
    """
    Compute ScType scores for each cell x cell_type.

    Algorithm (from Ianevski et al. 2022):
      1. Take scaled (z-scored) expression matrix
      2. For positive markers: weight by 2 (upweight enrichment)
      3. For negative markers: weight by -1 (penalize expression)
      4. Sum marker contributions per cell per cell type
      5. Normalize by sqrt(n_markers)

    Args:
        adata: AnnData with scaled expression in .X
        gs_positive: {cell_type: [ENSRNOG_ids]} positive markers
        gs_negative: {cell_type: [ENSRNOG_ids]} negative markers

    Returns:
        DataFrame: cell_types x cells, values are ScType scores
    """
    gene_set = set(adata.var_names)
    cell_names = adata.obs_names
    n_cells = len(cell_names)

    # Get expression matrix as dense array (genes x cells for efficiency)
    if hasattr(adata.X, 'toarray'):
        X = adata.X.toarray().T  # genes x cells
    else:
        X = np.array(adata.X).T
    gene_idx = {g: i for i, g in enumerate(adata.var_names)}

    scores = {}

    for ct in gs_positive:
        pos_genes = [g for g in gs_positive.get(ct, []) if g in gene_set]
        neg_genes = [g for g in gs_negative.get(ct, []) if g in gene_set]

        n_markers = len(pos_genes) + len(neg_genes)
        if len(pos_genes) < 2:
            continue

        # Cell-level score
        cell_scores = np.zeros(n_cells)

        # Positive markers: weight by 2
        for g in pos_genes:
            cell_scores += X[gene_idx[g], :] * 2.0

        # Negative markers: weight by -1
        for g in neg_genes:
            cell_scores -= X[gene_idx[g], :]

        # Normalize by sqrt(n_markers)
        cell_scores /= math.sqrt(n_markers)

        scores[ct] = cell_scores

    if not scores:
        return pd.DataFrame()

    return pd.DataFrame(scores, index=cell_names)


def assign_sctype(
    score_df: pd.DataFrame,
    adata: sc.AnnData,
    cluster_key: str = "leiden",
) -> pd.DataFrame:
    """
    Assign cell types per cluster using ScType scoring.

    For each cluster:
      1. Sum ScType scores across all cells in the cluster
      2. Winner = cell type with highest sum
      3. If winner score < n_cells_in_cluster / 4 → "Unknown"

    Returns DataFrame with cluster assignments.
    """
    if score_df.empty:
        return pd.DataFrame(columns=["cluster", "cell_type", "sctype_score",
                                      "second_best", "second_score", "n_cells"])

    clusters = sorted(adata.obs[cluster_key].unique(), key=lambda x: int(x))
    assignments = []

    for cluster in clusters:
        mask = adata.obs[cluster_key].astype(str) == str(cluster)
        n_cells = int(mask.sum())

        if n_cells == 0:
            continue

        # Sum scores across cells in this cluster
        cluster_scores = score_df.loc[mask].sum(axis=0).sort_values(ascending=False)

        best_ct = cluster_scores.index[0]
        best_score = cluster_scores.iloc[0]
        second_ct = cluster_scores.index[1] if len(cluster_scores) > 1 else "none"
        second_score = cluster_scores.iloc[1] if len(cluster_scores) > 1 else 0.0

        # ScType threshold: score < n_cells/4 → Unknown
        if best_score < n_cells / 4:
            assigned = "Unknown"
        else:
            assigned = best_ct

        assignments.append({
            "cluster": cluster,
            "cell_type": assigned,
            "sctype_score": round(float(best_score), 2),
            "second_best": second_ct,
            "second_score": round(float(second_score), 2),
            "margin": round(float(best_score - second_score), 2),
            "n_cells": n_cells,
            "pct_cells": round(100 * n_cells / adata.n_obs, 1),
        })

    return pd.DataFrame(assignments)


# ============================================================================
# Step 3: Annotate a sample
# ============================================================================
def load_marker_db() -> dict:
    if not MARKER_DB_PATH.exists():
        logger.error(f"Marker database not found: {MARKER_DB_PATH}")
        logger.error("Run: python annotate_sctype.py build-markers")
        sys.exit(1)
    with open(MARKER_DB_PATH, "rb") as f:
        return pickle.load(f)


def get_sctype_tissues(tissue_normalized: Optional[str]) -> List[str]:
    """Map a normalized tissue name to ScTypeDB tissue type(s)."""
    if not tissue_normalized:
        return ["Immune system"]  # fallback

    # Direct lookup
    tissues = MOTRPAC_TO_SCTYPE.get(tissue_normalized)
    if tissues:
        return tissues

    # Fuzzy match
    t_lower = tissue_normalized.lower()
    for motrpac, sctype_list in MOTRPAC_TO_SCTYPE.items():
        if motrpac in t_lower or t_lower in motrpac:
            return sctype_list

    # Default: immune system (always relevant)
    return ["Immune system"]


def annotate_sample(
    h5ad_path: Path,
    marker_db: dict,
    output_dir: Path,
    tissue_hint: Optional[str] = None,
    sctype_tissues: Optional[List[str]] = None,
    generate_umap: bool = False,
) -> Optional[dict]:
    """Annotate a single h5ad file using ScType."""
    sample_id = h5ad_path.stem
    sample_outdir = output_dir / sample_id
    sample_outdir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Annotating {sample_id}...")

    try:
        adata = sc.read_h5ad(h5ad_path)
        n_cells = adata.n_obs
        n_genes = adata.n_vars

        if n_cells < 10:
            logger.warning(f"  {sample_id}: Only {n_cells} cells, skipping")
            return None

        # Determine which ScTypeDB tissues to use
        if sctype_tissues is None:
            sctype_tissues = get_sctype_tissues(tissue_hint)
        logger.info(f"  {n_cells:,} cells | ScType tissues: {sctype_tissues}")

        # --- Normalize + cluster (same as PanglaoDB pipeline) ---
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

        n_hvg = min(2000, n_genes)
        sc.pp.highly_variable_genes(adata, n_top_genes=n_hvg, flavor="seurat_v3",
                                     layer=None, span=0.3)
        adata_hvg = adata[:, adata.var["highly_variable"]].copy()
        sc.pp.scale(adata_hvg, max_value=10)
        n_pcs = min(50, n_cells - 1, n_hvg - 1)
        sc.tl.pca(adata_hvg, n_comps=n_pcs)
        adata.obsm["X_pca"] = adata_hvg.obsm["X_pca"]

        n_neighbors = min(15, n_cells - 1)
        sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=min(30, n_pcs))
        sc.tl.leiden(adata, resolution=LEIDEN_RESOLUTION, key_added="leiden")

        n_clusters = adata.obs["leiden"].nunique()
        logger.info(f"  {n_clusters} Leiden clusters")

        # --- Build positive/negative marker dicts for selected tissues ---
        db = marker_db["db"]
        gs_positive = {}
        gs_negative = {}
        for tissue in sctype_tissues:
            if tissue not in db:
                logger.warning(f"  Tissue '{tissue}' not in ScTypeDB")
                continue
            for ct, markers in db[tissue].items():
                # Prefix with tissue to disambiguate same-named types across tissues
                key = f"{ct}" if len(sctype_tissues) == 1 else f"{ct} ({tissue})"
                if key in gs_positive:
                    # Merge markers if same cell type appears in multiple tissues
                    gs_positive[key] = sorted(set(gs_positive[key]) | set(markers["positive"]))
                    gs_negative[key] = sorted(set(gs_negative[key]) | set(markers["negative"]))
                else:
                    gs_positive[key] = markers["positive"]
                    gs_negative[key] = markers["negative"]

        # --- Scale ONLY marker genes (memory-efficient) ---
        # Collect all unique marker genes
        all_marker_genes = set()
        for genes in gs_positive.values():
            all_marker_genes.update(genes)
        for genes in gs_negative.values():
            all_marker_genes.update(genes)
        marker_genes_in_data = sorted(all_marker_genes & set(adata.var_names))
        logger.info(f"  {len(marker_genes_in_data)} marker genes in data (of {len(all_marker_genes)} total)")

        if len(marker_genes_in_data) < 10:
            logger.warning(f"  {sample_id}: Too few marker genes in data")
            return None

        # Subset to marker genes only, then scale
        adata_markers = adata[:, marker_genes_in_data].copy()
        sc.pp.scale(adata_markers, max_value=10)

        logger.info(f"  Scoring {len(gs_positive)} cell types ({sum(1 for v in gs_negative.values() if v)} with negative markers)")

        # --- Score (on scaled marker-gene subset) ---
        score_df = sctype_score(adata_markers, gs_positive, gs_negative)
        if score_df.empty:
            logger.warning(f"  {sample_id}: No cell types scored")
            return None

        # --- Assign (using cluster labels from full adata) ---
        assignments = assign_sctype(score_df, adata, cluster_key="leiden")

        # --- DE markers for validation ---
        try:
            sc.tl.rank_genes_groups(adata, groupby="leiden", method="wilcoxon",
                                     n_genes=10, use_raw=False)
            has_de = True
        except Exception:
            has_de = False

        # --- Write outputs ---
        # 1. Assignments
        assign_path = sample_outdir / f"{sample_id}_sctype_assignments.tsv"
        assignments.to_csv(assign_path, sep="\t", index=False)

        # 2. Score matrix (cluster-level sums)
        cluster_scores = {}
        for cluster in sorted(adata.obs["leiden"].unique(), key=int):
            mask = adata.obs["leiden"].astype(str) == str(cluster)
            cluster_scores[cluster] = score_df.loc[mask].sum(axis=0)
        score_matrix = pd.DataFrame(cluster_scores).T
        score_matrix.index.name = "cluster"
        score_path = sample_outdir / f"{sample_id}_sctype_scores.tsv"
        score_matrix.to_csv(score_path, sep="\t", float_format="%.2f")

        # 3. Top DE markers
        if has_de:
            rev = {v: k for k, v in marker_db["symbol_to_ensrnog"].items()}
            de_rows = []
            cluster_to_ct = dict(zip(assignments["cluster"].astype(str), assignments["cell_type"]))
            for cluster in sorted(adata.obs["leiden"].unique(), key=int):
                try:
                    names = adata.uns["rank_genes_groups"]["names"][cluster]
                    scores_de = adata.uns["rank_genes_groups"]["scores"][cluster]
                    ct = cluster_to_ct.get(str(cluster), "Unknown")
                    for i in range(min(10, len(names))):
                        ensrnog = str(names[i])
                        sym = rev.get(ensrnog, ensrnog)
                        de_rows.append({
                            "cluster": cluster,
                            "assigned_cell_type": ct,
                            "rank": i + 1,
                            "gene_id": ensrnog,
                            "gene_symbol": sym,
                            "score": round(float(scores_de[i]), 4),
                        })
                except (KeyError, IndexError):
                    continue
            if de_rows:
                de_path = sample_outdir / f"{sample_id}_sctype_top_markers.tsv"
                pd.DataFrame(de_rows).to_csv(de_path, sep="\t", index=False)

        # Summary
        ct_counts = assignments.set_index("cluster")["cell_type"].value_counts().to_dict()
        n_unknown = int((assignments["cell_type"] == "Unknown").sum())

        logger.info(f"  Results: {len(ct_counts)} types, {n_unknown} Unknown clusters")
        for _, row in assignments.iterrows():
            flag = " ***" if row["cell_type"] == "Unknown" else ""
            logger.info(f"    Cl {row['cluster']:>3}  {row['cell_type']:35s}  {row['n_cells']:>5} cells  score={row['sctype_score']:>8.1f}{flag}")

        return {
            "sample_id": sample_id,
            "sctype_tissues": sctype_tissues,
            "n_cells": n_cells,
            "n_clusters": n_clusters,
            "n_unknown": n_unknown,
            "assignments": assignments.to_dict("records"),
        }

    except Exception as e:
        logger.error(f"  {sample_id}: Failed - {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return None


# ============================================================================
# Compare: ScType vs PanglaoDB side-by-side
# ============================================================================
def compare_annotations(sample_id: str, output_dir: Path):
    """Compare ScType and PanglaoDB annotations for a sample."""
    panglao_dir = PROJECT_ROOT / "data/training/cell_annotations" / sample_id
    sctype_dir = output_dir / sample_id

    panglao_file = panglao_dir / f"{sample_id}_assignments.tsv"
    sctype_file = sctype_dir / f"{sample_id}_sctype_assignments.tsv"

    if not panglao_file.exists():
        logger.error(f"PanglaoDB results not found: {panglao_file}")
        return
    if not sctype_file.exists():
        logger.error(f"ScType results not found: {sctype_file}")
        return

    panglao = pd.read_csv(panglao_file, sep="\t")
    sctype = pd.read_csv(sctype_file, sep="\t")

    # Merge on cluster
    merged = panglao.merge(sctype, on="cluster", suffixes=("_panglao", "_sctype"))

    logger.info(f"\n{'='*80}")
    logger.info(f"COMPARISON: {sample_id}")
    logger.info(f"{'='*80}")
    logger.info(f"{'Cluster':>7}  {'Cells':>6}  {'PanglaoDB':35s}  {'ScType':35s}  {'Match':>5}")
    logger.info(f"{'-'*7:>7}  {'-'*6:>6}  {'-'*35:35s}  {'-'*35:35s}  {'-'*5:>5}")

    n_match = 0
    n_total = len(merged)
    for _, row in merged.iterrows():
        p_ct = row.get("cell_type_panglao", "?")
        s_ct = row.get("cell_type_sctype", "?")
        cells = row.get("n_cells_panglao", 0)
        match = "YES" if p_ct.lower() == s_ct.lower() else "no"
        if match == "YES":
            n_match += 1
        logger.info(f"  {row['cluster']:>5}  {cells:>6}  {p_ct:35s}  {s_ct:35s}  {match:>5}")

    logger.info(f"\nAgreement: {n_match}/{n_total} clusters ({100*n_match/n_total:.0f}%)")

    # Save comparison
    comp_path = sctype_dir / f"{sample_id}_comparison.tsv"
    merged.to_csv(comp_path, sep="\t", index=False)
    logger.info(f"Saved to {comp_path}")


# ============================================================================
# Summarize
# ============================================================================
def summarize_annotations(output_dir: Path):
    """Aggregate ScType annotations corpus-wide."""
    logger.info("Summarizing ScType annotations...")

    inv_path = PROJECT_ROOT / "reports/annotations/annotation_inventory.json"
    sample_tissue = {}
    if inv_path.exists():
        with open(inv_path) as f:
            inv = json.load(f)
        for s in inv.get("samples", []):
            sample_tissue[s["sample_id"]] = {
                "tissue": s.get("tissue_normalized"),
                "motrpac_tissue": s.get("motrpac_tissue_match"),
                "accession": s.get("accession"),
            }

    all_assignments = []
    all_ct_counts = defaultdict(Counter)

    for sample_dir in sorted(output_dir.iterdir()):
        if not sample_dir.is_dir() or sample_dir.name.startswith("_"):
            continue
        sample_id = sample_dir.name
        assign_file = sample_dir / f"{sample_id}_sctype_assignments.tsv"
        if not assign_file.exists():
            continue

        assignments = pd.read_csv(assign_file, sep="\t")
        meta = sample_tissue.get(sample_id, {})
        motrpac_tissue = meta.get("motrpac_tissue", "unknown")

        for _, row in assignments.iterrows():
            all_assignments.append({
                "sample_id": sample_id,
                "accession": meta.get("accession", "?"),
                "motrpac_tissue": motrpac_tissue,
                "cluster": row["cluster"],
                "cell_type": row["cell_type"],
                "sctype_score": row["sctype_score"],
                "n_cells": row.get("n_cells", 0),
            })
            if motrpac_tissue and motrpac_tissue != "unknown":
                all_ct_counts[motrpac_tissue][row["cell_type"]] += int(row.get("n_cells", 0))

    if not all_assignments:
        logger.warning("No annotations found")
        return

    summary_dir = output_dir / "_corpus_summary"
    summary_dir.mkdir(exist_ok=True)

    pd.DataFrame(all_assignments).to_csv(summary_dir / "sctype_annotation_summary.tsv", sep="\t", index=False)

    ct_tissue_df = pd.DataFrame(all_ct_counts).fillna(0).astype(int)
    ct_tissue_df.index.name = "cell_type"
    ct_tissue_df = ct_tissue_df.loc[ct_tissue_df.sum(axis=1).sort_values(ascending=False).index]
    ct_tissue_df.to_csv(summary_dir / "sctype_cell_type_by_tissue.tsv", sep="\t")

    total_cells = sum(r["n_cells"] for r in all_assignments)
    n_unknown = sum(1 for r in all_assignments if r["cell_type"] == "Unknown")

    logger.info("=" * 60)
    logger.info("SCTYPE ANNOTATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  Samples: {len(set(r['sample_id'] for r in all_assignments))}")
    logger.info(f"  Total cells: {total_cells:,}")
    logger.info(f"  Unknown clusters: {n_unknown}")

    all_ct = Counter()
    for r in all_assignments:
        all_ct[r["cell_type"]] += r["n_cells"]
    logger.info(f"\n  Top cell types:")
    for ct, n in all_ct.most_common(15):
        logger.info(f"    {ct:35s}  {n:>8,} cells")

    logger.info(f"\n  MoTrPAC tissue coverage:")
    for tissue, cts in sorted(all_ct_counts.items()):
        n_cells = sum(cts.values())
        n_types = len(cts)
        logger.info(f"    {tissue:20s}  {n_types:3d} cell types  {n_cells:>8,} cells")


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="ScType cell-type annotation")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("build-markers", help="Build ScType marker database")

    sub_ann = subparsers.add_parser("annotate", help="Annotate samples")
    sub_ann.add_argument("--sample-id", help="Single sample")
    sub_ann.add_argument("--tissue", help="Override ScType tissue (e.g., Heart)")
    sub_ann.add_argument("--all", action="store_true")
    sub_ann.add_argument("--batch-start", type=int)
    sub_ann.add_argument("--batch-end", type=int)
    sub_ann.add_argument("--no-umap", action="store_true")
    sub_ann.add_argument("--output-dir", default=str(OUTPUT_DIR))

    sub_cmp = subparsers.add_parser("compare", help="Compare ScType vs PanglaoDB")
    sub_cmp.add_argument("--sample-id", required=True)
    sub_cmp.add_argument("--tissue", help="Override ScType tissue")
    sub_cmp.add_argument("--output-dir", default=str(OUTPUT_DIR))

    sub_sum = subparsers.add_parser("summarize", help="Corpus-wide summary")
    sub_sum.add_argument("--output-dir", default=str(OUTPUT_DIR))

    args = parser.parse_args()

    if args.command == "build-markers":
        build_marker_database()

    elif args.command == "annotate":
        marker_db = load_marker_db()
        out_dir = Path(args.output_dir)

        # Load tissue hints
        tissue_hints = {}
        inv_path = PROJECT_ROOT / "reports/annotations/annotation_inventory.json"
        if inv_path.exists():
            with open(inv_path) as f:
                inv = json.load(f)
            for s in inv.get("samples", []):
                tissue_hints[s["sample_id"]] = s.get("tissue_normalized")

        if args.sample_id:
            h5ad_path = QC_H5AD_DIR / f"{args.sample_id}.h5ad"
            if not h5ad_path.exists():
                logger.error(f"Not found: {h5ad_path}")
                sys.exit(1)
            sctype_tissues = [args.tissue] if args.tissue else None
            hint = tissue_hints.get(args.sample_id)
            annotate_sample(h5ad_path, marker_db, out_dir,
                           tissue_hint=hint, sctype_tissues=sctype_tissues,
                           generate_umap=not args.no_umap)

        elif args.batch_start is not None and args.batch_end is not None:
            all_h5ad = sorted(QC_H5AD_DIR.glob("*.h5ad"))
            batch = all_h5ad[args.batch_start:args.batch_end]
            logger.info(f"Batch: {len(batch)} files")
            for h5ad_path in batch:
                sid = h5ad_path.stem
                hint = tissue_hints.get(sid)
                annotate_sample(h5ad_path, marker_db, out_dir,
                               tissue_hint=hint, generate_umap=not args.no_umap)

        elif args.all:
            all_h5ad = sorted(QC_H5AD_DIR.glob("*.h5ad"))
            logger.info(f"Annotating {len(all_h5ad)} samples with ScType...")
            for i, h5ad_path in enumerate(all_h5ad):
                sid = h5ad_path.stem
                hint = tissue_hints.get(sid)
                annotate_sample(h5ad_path, marker_db, out_dir,
                               tissue_hint=hint, generate_umap=not args.no_umap)
                if (i + 1) % 50 == 0:
                    logger.info(f"  Progress: {i+1}/{len(all_h5ad)}")

    elif args.command == "compare":
        out_dir = Path(args.output_dir)
        # Run ScType first if not already done
        sctype_file = out_dir / args.sample_id / f"{args.sample_id}_sctype_assignments.tsv"
        if not sctype_file.exists():
            marker_db = load_marker_db()
            h5ad_path = QC_H5AD_DIR / f"{args.sample_id}.h5ad"
            tissue_hints = {}
            inv_path = PROJECT_ROOT / "reports/annotations/annotation_inventory.json"
            if inv_path.exists():
                with open(inv_path) as f:
                    inv = json.load(f)
                for s in inv.get("samples", []):
                    tissue_hints[s["sample_id"]] = s.get("tissue_normalized")
            sctype_tissues = [args.tissue] if args.tissue else None
            hint = tissue_hints.get(args.sample_id)
            annotate_sample(h5ad_path, marker_db, out_dir,
                           tissue_hint=hint, sctype_tissues=sctype_tissues)
        compare_annotations(args.sample_id, out_dir)

    elif args.command == "summarize":
        summarize_annotations(Path(args.output_dir))


if __name__ == "__main__":
    main()