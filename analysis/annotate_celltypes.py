#!/usr/bin/env python3
"""
annotate_celltypes.py -- Marker-based cell-type annotation for QC'd h5ad corpus

Three steps, each runnable independently:

  Step 1: build-markers
    Download PanglaoDB markers, map gene symbols to ENSRNOG IDs via gene_universe.tsv,
    save as pickle for SLURM workers.

  Step 2: annotate
    For each QC'd h5ad: normalize in-memory -> Leiden cluster -> score clusters
    against marker gene sets -> assign cell type. Produces per-sample sidecar files
    and validation outputs.

  Step 3: summarize
    Aggregate all per-sample annotations into a corpus-wide summary with
    tissue coverage, confidence distributions, and collaborator review files.

Usage:
  python annotate_celltypes.py build-markers
  python annotate_celltypes.py annotate [--sample-id GSE137869_sample0] [--all]
  python annotate_celltypes.py summarize
  sbatch slurm/annotate_celltypes.slurm   # array job over all h5ad files

Validation outputs (per sample):
  {sample_id}_celltypes.tsv       - barcode, cluster, cell_type, confidence
  {sample_id}_cluster_scores.tsv  - cluster x cell_type score matrix (for review)
  {sample_id}_top_markers.tsv     - top DE genes per cluster (for review)
  {sample_id}_umap.png            - UMAP colored by assigned cell type

Validation outputs (corpus-wide, from summarize):
  annotation_summary.tsv          - all samples, clusters, assignments, confidence
  cell_type_by_tissue.tsv         - cell type x MoTrPAC tissue count matrix
  low_confidence_flags.tsv        - clusters below confidence threshold for review
  annotation_qc_report.json       - full statistics for programmatic consumption

Normalization note:
  Standard scanpy (normalize_total + log1p base e) for annotation.
  NOT GeneCompass normalization. Marker databases assume standard preprocessing.
  Raw counts in h5ad are never modified on disk.

No GPU required. ~2-5 min per sample on CPU.
"""

import argparse
import csv
import json
import logging
import os
import pickle
import sys
import warnings
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

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
OUTPUT_DIR = PROJECT_ROOT / "data/training/cell_annotations"
MARKER_DB_PATH = OUTPUT_DIR / "marker_database.pkl"
PANGLAODB_URL = "https://panglaodb.se/markers/PanglaoDB_markers_27_Mar_2020.tsv.gz"

# Annotation parameters
LEIDEN_RESOLUTION = 0.8
MIN_MARKER_GENES = 3       # min markers that must be in data for a cell type to be scored
CONFIDENCE_THRESHOLD = 0.1  # scores below this flagged for review
N_TOP_GENES = 10            # top DE genes per cluster for validation


# ============================================================================
# Step 1: Build marker database
# ============================================================================
def build_marker_database():
    """
    Download PanglaoDB markers, map to ENSRNOG IDs, save as pickle.

    Output: marker_database.pkl containing:
      - markers_by_celltype: {cell_type: [ENSRNOG_ids]}
      - markers_by_organ_celltype: {(organ, cell_type): [ENSRNOG_ids]}
      - celltype_metadata: {cell_type: {organ, germ_layer, n_markers}}
      - symbol_to_ensrnog: {symbol: ensrnog_id}
      - unmapped_symbols: set of symbols that couldn't be mapped
      - build_info: metadata about the build
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Load gene universe for symbol -> ENSRNOG mapping ---
    logger.info("Loading gene universe...")
    gu = pd.read_csv(GENE_UNIVERSE, sep="\t")
    # Build symbol -> ensrnog lookup (case-insensitive)
    symbol_to_ensrnog = {}
    for _, row in gu.iterrows():
        sym = str(row["symbol"]).strip()
        eid = str(row["ensembl_id"]).strip()
        if sym and eid and sym != "nan":
            symbol_to_ensrnog[sym.upper()] = eid
    logger.info(f"  {len(symbol_to_ensrnog)} symbols mapped to ENSRNOG")

    # --- Download PanglaoDB markers ---
    logger.info(f"Downloading PanglaoDB markers from {PANGLAODB_URL}...")
    pdb = pd.read_csv(PANGLAODB_URL, sep="\t")
    logger.info(f"  {len(pdb)} total marker entries")
    logger.info(f"  Columns: {list(pdb.columns)}")

    # Filter to mouse+human markers (applicable to rat via orthology)
    pdb_filtered = pdb[pdb["species"].isin(["Mm Hs", "Mm", "Hs"])].copy()
    logger.info(f"  {len(pdb_filtered)} markers after species filter (Mm Hs / Mm / Hs)")

    # --- Map symbols to ENSRNOG ---
    mapped_count = 0
    unmapped_symbols = set()

    markers_by_celltype = defaultdict(set)
    markers_by_organ_celltype = defaultdict(set)
    celltype_organs = defaultdict(set)
    celltype_germ_layers = {}

    for _, row in pdb_filtered.iterrows():
        symbol = str(row["official gene symbol"]).strip().upper()
        cell_type = str(row["cell type"]).strip()
        organ = str(row["organ"]).strip()
        germ_layer = str(row.get("germ layer", "")).strip()
        canonical = row.get("canonical marker", 0)

        ensrnog = symbol_to_ensrnog.get(symbol)
        if ensrnog is None:
            unmapped_symbols.add(symbol)
            continue

        mapped_count += 1
        markers_by_celltype[cell_type].add(ensrnog)
        markers_by_organ_celltype[(organ, cell_type)].add(ensrnog)
        celltype_organs[cell_type].add(organ)
        if germ_layer and germ_layer != "nan":
            celltype_germ_layers[cell_type] = germ_layer

    # Convert sets to sorted lists
    markers_by_celltype = {k: sorted(v) for k, v in markers_by_celltype.items()}
    markers_by_organ_celltype = {k: sorted(v) for k, v in markers_by_organ_celltype.items()}

    # Build metadata
    celltype_metadata = {}
    for ct, genes in markers_by_celltype.items():
        celltype_metadata[ct] = {
            "organs": sorted(celltype_organs.get(ct, set())),
            "germ_layer": celltype_germ_layers.get(ct, "unknown"),
            "n_markers": len(genes),
        }

    # --- Save ---
    db = {
        "markers_by_celltype": markers_by_celltype,
        "markers_by_organ_celltype": markers_by_organ_celltype,
        "celltype_metadata": celltype_metadata,
        "symbol_to_ensrnog": symbol_to_ensrnog,
        "unmapped_symbols": sorted(unmapped_symbols),
        "build_info": {
            "source": PANGLAODB_URL,
            "built_at": datetime.now().isoformat(),
            "gene_universe": str(GENE_UNIVERSE),
            "total_panglaodb_entries": len(pdb),
            "after_species_filter": len(pdb_filtered),
            "mapped_to_ensrnog": mapped_count,
            "unmapped_symbols": len(unmapped_symbols),
            "unique_cell_types": len(markers_by_celltype),
            "unique_organs": len({o for o, _ in markers_by_organ_celltype}),
        },
    }

    with open(MARKER_DB_PATH, "wb") as f:
        pickle.dump(db, f)

    logger.info(f"  Mapped {mapped_count} marker entries to ENSRNOG")
    logger.info(f"  {len(unmapped_symbols)} symbols unmapped (no rat ortholog)")
    logger.info(f"  {len(markers_by_celltype)} cell types in database")
    logger.info(f"  Saved to {MARKER_DB_PATH}")

    # Print coverage summary
    logger.info("\n  Cell types with most markers:")
    for ct, genes in sorted(markers_by_celltype.items(), key=lambda x: -len(x[1]))[:15]:
        organs = ", ".join(celltype_metadata[ct]["organs"][:3])
        logger.info(f"    {ct:30s}  {len(genes):4d} markers  ({organs})")

    return db


# ============================================================================
# Step 2: Annotate a single h5ad file
# ============================================================================
def load_marker_db() -> dict:
    """Load cached marker database."""
    if not MARKER_DB_PATH.exists():
        logger.error(f"Marker database not found: {MARKER_DB_PATH}")
        logger.error("Run: python annotate_celltypes.py build-markers")
        sys.exit(1)
    with open(MARKER_DB_PATH, "rb") as f:
        return pickle.load(f)


def score_clusters(
    adata: sc.AnnData,
    markers_by_celltype: Dict[str, List[str]],
    cluster_key: str = "leiden",
) -> pd.DataFrame:
    """
    Score each cluster against each cell type's marker set.

    Returns a DataFrame: rows=clusters, columns=cell_types, values=mean z-score.
    """
    gene_set = set(adata.var_names)
    clusters = sorted(adata.obs[cluster_key].unique(), key=lambda x: int(x))

    scores = {}
    scored_celltypes = {}

    for ct, markers in markers_by_celltype.items():
        # Filter to markers present in this dataset
        present = [g for g in markers if g in gene_set]
        if len(present) < MIN_MARKER_GENES:
            continue
        scored_celltypes[ct] = present

        # Score genes (adds to adata.obs)
        score_name = f"score_{ct}"
        try:
            sc.tl.score_genes(adata, gene_list=present, score_name=score_name, use_raw=False)
        except Exception:
            continue

        # Mean score per cluster
        cluster_scores = adata.obs.groupby(cluster_key)[score_name].mean()
        scores[ct] = cluster_scores

    if not scores:
        return pd.DataFrame()

    score_df = pd.DataFrame(scores, index=clusters)
    score_df.index.name = "cluster"

    # Clean up temporary score columns from adata.obs
    score_cols = [c for c in adata.obs.columns if c.startswith("score_")]
    adata.obs.drop(columns=score_cols, inplace=True, errors="ignore")

    return score_df


def assign_celltypes(
    score_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Assign cell type to each cluster based on highest score.

    Returns DataFrame with: cluster, cell_type, confidence, second_best, margin
    """
    if score_df.empty:
        return pd.DataFrame(columns=["cluster", "cell_type", "confidence", "second_best", "margin"])

    assignments = []
    for cluster in score_df.index:
        row = score_df.loc[cluster].sort_values(ascending=False)
        best_ct = row.index[0]
        best_score = row.iloc[0]
        second_ct = row.index[1] if len(row) > 1 else "none"
        second_score = row.iloc[1] if len(row) > 1 else 0.0
        margin = best_score - second_score

        assignments.append({
            "cluster": cluster,
            "cell_type": best_ct,
            "confidence": round(float(best_score), 4),
            "second_best": second_ct,
            "second_score": round(float(second_score), 4),
            "margin": round(float(margin), 4),
        })

    return pd.DataFrame(assignments)


def annotate_sample(
    h5ad_path: Path,
    marker_db: dict,
    output_dir: Path,
    generate_umap: bool = True,
    tissue_hint: Optional[str] = None,
) -> Optional[dict]:
    """
    Annotate a single QC'd h5ad file.

    Returns summary dict or None on failure.
    """
    sample_id = h5ad_path.stem
    sample_outdir = output_dir / sample_id
    sample_outdir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Annotating {sample_id}...")

    try:
        # Load raw counts
        adata = sc.read_h5ad(h5ad_path)
        n_cells = adata.n_obs
        n_genes = adata.n_vars

        if n_cells < 10:
            logger.warning(f"  {sample_id}: Only {n_cells} cells, skipping")
            return None

        logger.info(f"  {n_cells:,} cells, {n_genes:,} genes (raw counts)")

        # --- Normalize in-memory (standard scanpy, NOT GeneCompass) ---
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

        # --- HVG selection + PCA + neighbors + clustering ---
        n_hvg = min(2000, n_genes)
        sc.pp.highly_variable_genes(adata, n_top_genes=n_hvg, flavor="seurat_v3",
                                     layer=None, span=0.3)
        adata_hvg = adata[:, adata.var["highly_variable"]].copy()

        sc.pp.scale(adata_hvg, max_value=10)
        n_pcs = min(50, n_cells - 1, n_hvg - 1)
        sc.tl.pca(adata_hvg, n_comps=n_pcs)

        # Transfer PCA to full adata
        adata.obsm["X_pca"] = adata_hvg.obsm["X_pca"]

        n_neighbors = min(15, n_cells - 1)
        sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=min(30, n_pcs))
        sc.tl.leiden(adata, resolution=LEIDEN_RESOLUTION, key_added="leiden")

        n_clusters = adata.obs["leiden"].nunique()
        logger.info(f"  {n_clusters} Leiden clusters (resolution={LEIDEN_RESOLUTION})")

        # --- Score clusters against marker gene sets ---
        markers = marker_db["markers_by_celltype"]

        # If tissue hint available, prefer organ-specific markers
        if tissue_hint:
            organ_markers = {}
            for (organ, ct), genes in marker_db["markers_by_organ_celltype"].items():
                if tissue_hint.lower() in organ.lower():
                    key = f"{ct}"
                    if key in organ_markers:
                        organ_markers[key] = sorted(set(organ_markers[key]) | set(genes))
                    else:
                        organ_markers[key] = genes
            if organ_markers:
                logger.info(f"  Using {len(organ_markers)} organ-specific cell types for '{tissue_hint}'")
                # Merge: organ-specific markers take priority, supplement with global
                merged = dict(markers)
                merged.update(organ_markers)
                markers = merged

        score_df = score_clusters(adata, markers, cluster_key="leiden")

        if score_df.empty:
            logger.warning(f"  {sample_id}: No cell types scored (insufficient marker overlap)")
            return None

        # --- Assign cell types ---
        assignments = assign_celltypes(score_df)

        # Map cluster -> cell type in adata.obs
        cluster_to_ct = dict(zip(
            assignments["cluster"].astype(str),
            assignments["cell_type"]
        ))
        cluster_to_confidence = dict(zip(
            assignments["cluster"].astype(str),
            assignments["confidence"]
        ))

        adata.obs["cell_type"] = adata.obs["leiden"].astype(str).map(cluster_to_ct).fillna("Unknown")
        adata.obs["cell_type_confidence"] = adata.obs["leiden"].astype(str).map(cluster_to_confidence).fillna(0.0)

        # --- Compute top DE genes per cluster (for validation) ---
        try:
            sc.tl.rank_genes_groups(adata, groupby="leiden", method="wilcoxon",
                                     n_genes=N_TOP_GENES, use_raw=False)
            has_de = True
        except Exception as e:
            logger.warning(f"  DE failed: {e}")
            has_de = False

        # ================================================================
        # VALIDATION OUTPUTS
        # ================================================================

        # 1. Per-cell annotation file (barcode -> cell_type)
        celltypes_path = sample_outdir / f"{sample_id}_celltypes.tsv"
        ct_df = adata.obs[["leiden", "cell_type", "cell_type_confidence"]].copy()
        ct_df.index.name = "barcode"
        ct_df.to_csv(celltypes_path, sep="\t")

        # 2. Cluster x cell_type score matrix (for collaborator review)
        scores_path = sample_outdir / f"{sample_id}_cluster_scores.tsv"
        score_df.to_csv(scores_path, sep="\t", float_format="%.4f")

        # 3. Assignment summary with confidence
        assign_path = sample_outdir / f"{sample_id}_assignments.tsv"
        # Add cell counts per cluster
        cluster_counts = adata.obs["leiden"].value_counts()
        assignments["n_cells"] = assignments["cluster"].map(
            lambda c: cluster_counts.get(c, 0)
        )
        assignments["pct_cells"] = (assignments["n_cells"] / n_cells * 100).round(1)
        assignments["low_confidence"] = assignments["confidence"] < CONFIDENCE_THRESHOLD
        assignments.to_csv(assign_path, sep="\t", index=False)

        # 4. Top DE markers per cluster (for manual validation)
        if has_de:
            top_markers_path = sample_outdir / f"{sample_id}_top_markers.tsv"
            de_rows = []
            for cluster in sorted(adata.obs["leiden"].unique(), key=int):
                try:
                    names = adata.uns["rank_genes_groups"]["names"][cluster]
                    scores_de = adata.uns["rank_genes_groups"]["scores"][cluster]
                    pvals = adata.uns["rank_genes_groups"]["pvals_adj"][cluster]
                    ct = cluster_to_ct.get(cluster, "Unknown")
                    for i in range(min(N_TOP_GENES, len(names))):
                        # Map ENSRNOG back to symbol for readability
                        ensrnog = str(names[i])
                        symbol = marker_db.get("symbol_to_ensrnog", {})
                        # Reverse lookup: ensrnog -> symbol
                        rev = {v: k for k, v in marker_db["symbol_to_ensrnog"].items()}
                        sym = rev.get(ensrnog, ensrnog)
                        de_rows.append({
                            "cluster": cluster,
                            "assigned_cell_type": ct,
                            "rank": i + 1,
                            "gene_id": ensrnog,
                            "gene_symbol": sym,
                            "score": round(float(scores_de[i]), 4),
                            "pval_adj": f"{float(pvals[i]):.2e}",
                        })
                except (KeyError, IndexError):
                    continue
            if de_rows:
                pd.DataFrame(de_rows).to_csv(top_markers_path, sep="\t", index=False)

        # 5. UMAP visualization
        if generate_umap and n_cells >= 50:
            try:
                sc.tl.umap(adata)

                # Save UMAP colored by cell type
                umap_path = sample_outdir / f"{sample_id}_umap_celltype.png"
                fig = sc.pl.umap(adata, color="cell_type", title=f"{sample_id} - Cell Types",
                                  show=False, return_fig=True, legend_loc="on data",
                                  legend_fontsize=6, frameon=True)
                fig.savefig(umap_path, dpi=150, bbox_inches="tight")
                import matplotlib.pyplot as plt
                plt.close(fig)

                # Save UMAP colored by cluster (for comparison)
                umap_cluster_path = sample_outdir / f"{sample_id}_umap_cluster.png"
                fig2 = sc.pl.umap(adata, color="leiden", title=f"{sample_id} - Leiden Clusters",
                                   show=False, return_fig=True, legend_loc="on data",
                                   legend_fontsize=6, frameon=True)
                fig2.savefig(umap_cluster_path, dpi=150, bbox_inches="tight")
                plt.close(fig2)

                logger.info(f"  UMAP saved: {umap_path.name}")
            except Exception as e:
                logger.warning(f"  UMAP generation failed: {e}")

        # --- Summary ---
        ct_counts = adata.obs["cell_type"].value_counts().to_dict()
        n_low_conf = int(assignments["low_confidence"].sum())

        summary = {
            "sample_id": sample_id,
            "h5ad_path": str(h5ad_path),
            "n_cells": n_cells,
            "n_genes": n_genes,
            "n_clusters": n_clusters,
            "n_cell_types_assigned": len(set(cluster_to_ct.values())),
            "cell_type_counts": ct_counts,
            "n_low_confidence_clusters": n_low_conf,
            "low_confidence_clusters": assignments[assignments["low_confidence"]][
                ["cluster", "cell_type", "confidence", "n_cells"]
            ].to_dict("records"),
            "output_dir": str(sample_outdir),
            "annotated_at": datetime.now().isoformat(),
        }

        logger.info(f"  {len(ct_counts)} cell types: {dict(list(ct_counts.items())[:5])}...")
        if n_low_conf > 0:
            logger.warning(f"  {n_low_conf} clusters below confidence threshold ({CONFIDENCE_THRESHOLD})")

        return summary

    except Exception as e:
        logger.error(f"  {sample_id}: Failed - {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return None


# ============================================================================
# Step 3: Summarize annotations across corpus
# ============================================================================
def summarize_annotations(output_dir: Path):
    """
    Aggregate per-sample annotations into corpus-wide review files.
    """
    logger.info("Summarizing annotations across corpus...")

    # Load annotation inventory for tissue mapping
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
                "sex": s.get("sex_resolved"),
                "strain": s.get("strain_resolved"),
            }

    # Scan per-sample annotation outputs
    all_assignments = []
    all_ct_counts = defaultdict(Counter)  # motrpac_tissue -> cell_type -> count
    sample_summaries = []
    low_confidence_all = []

    for sample_dir in sorted(output_dir.iterdir()):
        if not sample_dir.is_dir():
            continue
        sample_id = sample_dir.name

        assign_file = sample_dir / f"{sample_id}_assignments.tsv"
        if not assign_file.exists():
            continue

        assignments = pd.read_csv(assign_file, sep="\t")
        meta = sample_tissue.get(sample_id, {})
        motrpac_tissue = meta.get("motrpac_tissue", "unknown")
        accession = meta.get("accession", "?")

        for _, row in assignments.iterrows():
            record = {
                "sample_id": sample_id,
                "accession": accession,
                "motrpac_tissue": motrpac_tissue,
                "tissue_normalized": meta.get("tissue", ""),
                "sex": meta.get("sex", ""),
                "strain": meta.get("strain", ""),
                "cluster": row["cluster"],
                "cell_type": row["cell_type"],
                "confidence": row["confidence"],
                "second_best": row.get("second_best", ""),
                "margin": row.get("margin", 0),
                "n_cells": row.get("n_cells", 0),
                "pct_cells": row.get("pct_cells", 0),
                "low_confidence": row.get("low_confidence", False),
            }
            all_assignments.append(record)

            if motrpac_tissue and motrpac_tissue != "unknown":
                all_ct_counts[motrpac_tissue][row["cell_type"]] += int(row.get("n_cells", 0))

            if row.get("low_confidence", False):
                low_confidence_all.append(record)

        sample_summaries.append({
            "sample_id": sample_id,
            "accession": accession,
            "motrpac_tissue": motrpac_tissue,
            "n_clusters": len(assignments),
            "n_cell_types": assignments["cell_type"].nunique(),
            "n_cells": int(assignments["n_cells"].sum()),
            "n_low_confidence": int(assignments["low_confidence"].sum()),
            "mean_confidence": round(assignments["confidence"].mean(), 4),
        })

    if not all_assignments:
        logger.warning("No annotations found to summarize")
        return

    # --- Write corpus-wide summary ---
    summary_dir = output_dir / "_corpus_summary"
    summary_dir.mkdir(exist_ok=True)

    # 1. All assignments
    assign_all_path = summary_dir / "annotation_summary.tsv"
    pd.DataFrame(all_assignments).to_csv(assign_all_path, sep="\t", index=False)
    logger.info(f"  {assign_all_path}: {len(all_assignments)} cluster assignments")

    # 2. Cell type x MoTrPAC tissue matrix
    ct_tissue_path = summary_dir / "cell_type_by_tissue.tsv"
    ct_tissue_df = pd.DataFrame(all_ct_counts).fillna(0).astype(int)
    ct_tissue_df.index.name = "cell_type"
    ct_tissue_df = ct_tissue_df.loc[ct_tissue_df.sum(axis=1).sort_values(ascending=False).index]
    ct_tissue_df.to_csv(ct_tissue_path, sep="\t")
    logger.info(f"  {ct_tissue_path}: {len(ct_tissue_df)} cell types x {len(ct_tissue_df.columns)} tissues")

    # 3. Low confidence flags
    if low_confidence_all:
        low_conf_path = summary_dir / "low_confidence_flags.tsv"
        pd.DataFrame(low_confidence_all).to_csv(low_conf_path, sep="\t", index=False)
        logger.info(f"  {low_conf_path}: {len(low_confidence_all)} clusters need review")

    # 4. Per-sample summary
    sample_sum_path = summary_dir / "per_sample_summary.tsv"
    pd.DataFrame(sample_summaries).to_csv(sample_sum_path, sep="\t", index=False)

    # 5. QC report (JSON)
    total_cells = sum(s["n_cells"] for s in sample_summaries)
    total_low_conf = sum(s["n_low_confidence"] for s in sample_summaries)
    all_ct = Counter()
    for record in all_assignments:
        all_ct[record["cell_type"]] += record["n_cells"]

    qc_report = {
        "generated_at": datetime.now().isoformat(),
        "n_samples_annotated": len(sample_summaries),
        "n_total_cells": total_cells,
        "n_total_clusters": len(all_assignments),
        "n_unique_cell_types": len(all_ct),
        "n_low_confidence_clusters": total_low_conf,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "leiden_resolution": LEIDEN_RESOLUTION,
        "min_marker_genes": MIN_MARKER_GENES,
        "cell_type_totals": dict(all_ct.most_common()),
        "motrpac_tissue_coverage": {
            tissue: {
                "n_cell_types": len(cts),
                "n_cells": sum(cts.values()),
                "cell_types": dict(cts.most_common(10)),
            }
            for tissue, cts in sorted(all_ct_counts.items())
        },
    }

    qc_path = summary_dir / "annotation_qc_report.json"
    with open(qc_path, "w") as f:
        json.dump(qc_report, f, indent=2)
    logger.info(f"  {qc_path}")

    # --- Print summary ---
    logger.info("=" * 60)
    logger.info("ANNOTATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  Samples annotated: {len(sample_summaries)}")
    logger.info(f"  Total cells: {total_cells:,}")
    logger.info(f"  Unique cell types: {len(all_ct)}")
    logger.info(f"  Low confidence clusters: {total_low_conf}")
    logger.info(f"\n  Top cell types (corpus-wide):")
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
    parser = argparse.ArgumentParser(description="Marker-based cell-type annotation")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Step 1: build markers
    sub_build = subparsers.add_parser("build-markers", help="Build marker database from PanglaoDB")

    # Step 2: annotate
    sub_annotate = subparsers.add_parser("annotate", help="Annotate h5ad files")
    sub_annotate.add_argument("--sample-id", help="Annotate a single sample (e.g., GSE137869_sample0)")
    sub_annotate.add_argument("--all", action="store_true", help="Annotate all h5ad files")
    sub_annotate.add_argument("--batch-start", type=int, help="SLURM array: start index")
    sub_annotate.add_argument("--batch-end", type=int, help="SLURM array: end index")
    sub_annotate.add_argument("--no-umap", action="store_true", help="Skip UMAP generation")
    sub_annotate.add_argument("--output-dir", default=str(OUTPUT_DIR))

    # Step 3: summarize
    sub_summary = subparsers.add_parser("summarize", help="Aggregate annotations corpus-wide")
    sub_summary.add_argument("--output-dir", default=str(OUTPUT_DIR))

    args = parser.parse_args()

    if args.command == "build-markers":
        build_marker_database()

    elif args.command == "annotate":
        marker_db = load_marker_db()
        out_dir = Path(args.output_dir)

        # Load tissue hints from annotation inventory
        tissue_hints = {}
        inv_path = PROJECT_ROOT / "reports/annotations/annotation_inventory.json"
        if inv_path.exists():
            with open(inv_path) as f:
                inv = json.load(f)
            for s in inv.get("samples", []):
                tissue_hints[s["sample_id"]] = s.get("tissue_normalized")

        if args.sample_id:
            # Single sample
            h5ad_path = QC_H5AD_DIR / f"{args.sample_id}.h5ad"
            if not h5ad_path.exists():
                logger.error(f"Not found: {h5ad_path}")
                sys.exit(1)
            hint = tissue_hints.get(args.sample_id)
            annotate_sample(h5ad_path, marker_db, out_dir,
                           generate_umap=not args.no_umap, tissue_hint=hint)

        elif args.batch_start is not None and args.batch_end is not None:
            # SLURM array batch
            all_h5ad = sorted(QC_H5AD_DIR.glob("*.h5ad"))
            batch = all_h5ad[args.batch_start:args.batch_end]
            logger.info(f"Batch: {len(batch)} files (indices {args.batch_start}-{args.batch_end})")
            for h5ad_path in batch:
                sid = h5ad_path.stem
                hint = tissue_hints.get(sid)
                annotate_sample(h5ad_path, marker_db, out_dir,
                               generate_umap=not args.no_umap, tissue_hint=hint)

        elif args.all:
            all_h5ad = sorted(QC_H5AD_DIR.glob("*.h5ad"))
            logger.info(f"Annotating all {len(all_h5ad)} h5ad files...")
            summaries = []
            for i, h5ad_path in enumerate(all_h5ad):
                sid = h5ad_path.stem
                hint = tissue_hints.get(sid)
                s = annotate_sample(h5ad_path, marker_db, out_dir,
                                   generate_umap=not args.no_umap, tissue_hint=hint)
                if s:
                    summaries.append(s)
                if (i + 1) % 50 == 0:
                    logger.info(f"  Progress: {i+1}/{len(all_h5ad)}")

        else:
            logger.error("Specify --sample-id, --all, or --batch-start/--batch-end")
            sys.exit(1)

    elif args.command == "summarize":
        summarize_annotations(Path(args.output_dir))


if __name__ == "__main__":
    main()