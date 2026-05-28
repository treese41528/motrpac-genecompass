#!/usr/bin/env python3
"""
extract_report_data.py -- Pull all deconvolution-relevant data for collaborator report.

Aggregates data from:
  - Consensus annotations (3-method pipeline)
  - PanglaoDB and ScType individual results
  - Annotation inventory (tissue, sex, strain metadata)
  - Consensus label map (LLM rationales)
  - Per-sample DE markers (top genes per cell type per tissue)
  - Forgetting analysis results

Output: reports/annotations/report_data.json (~comprehensive)

Usage:
  python analysis/extract_report_data.py
"""

import json
import logging
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path("/depot/reese18/apps/motrpac-genecompass")

# Input paths
CONSENSUS_SUMMARY = PROJECT_ROOT / "data/training/cell_annotations_consensus/_corpus_summary/consensus_annotations.tsv"
CONSENSUS_QC = PROJECT_ROOT / "data/training/cell_annotations_consensus/_corpus_summary/consensus_qc_report.json"
CONSENSUS_MAP = PROJECT_ROOT / "data/training/cell_annotations_consensus/consensus_label_map.json"
PANGLAO_SUMMARY = PROJECT_ROOT / "data/training/cell_annotations/_corpus_summary/annotation_summary.tsv"
PANGLAO_QC = PROJECT_ROOT / "data/training/cell_annotations/_corpus_summary/annotation_qc_report.json"
SCTYPE_SUMMARY = PROJECT_ROOT / "data/training/cell_annotations_sctype/_corpus_summary/sctype_annotation_summary.tsv"
ANNOTATION_INVENTORY = PROJECT_ROOT / "reports/annotations/annotation_inventory.json"
TISSUE_MAP = PROJECT_ROOT / "reports/annotations/tissue_motrpac_map.json"
FORGETTING_RESULTS = PROJECT_ROOT / "reports/forgetting/forgetting_results.json"
PANGLAO_DIR = PROJECT_ROOT / "data/training/cell_annotations"
SCTYPE_DIR = PROJECT_ROOT / "data/training/cell_annotations_sctype"
MARKER_DB = PROJECT_ROOT / "data/training/cell_annotations/marker_database.pkl"
SCTYPE_DB = PROJECT_ROOT / "data/training/cell_annotations_sctype/sctype_marker_database.pkl"

OUTPUT_PATH = PROJECT_ROOT / "reports/annotations/report_data.json"


def load_json(path):
    if path.exists():
        with open(path) as f:
            return json.load(f)
    logger.warning(f"  Not found: {path}")
    return {}


def main():
    report = {
        "generated_at": datetime.now().isoformat(),
        "project": "MoTrPAC-GeneCompass",
    }

    # ================================================================
    # 1. CORPUS OVERVIEW
    # ================================================================
    logger.info("1. Corpus overview...")
    consensus = pd.read_csv(CONSENSUS_SUMMARY, sep="\t") if CONSENSUS_SUMMARY.exists() else pd.DataFrame()

    if not consensus.empty:
        report["corpus"] = {
            "n_clusters": len(consensus),
            "n_cells": int(consensus["n_cells"].sum()),
            "n_samples": int(consensus["sample_id"].nunique()),
            "n_cell_types": int(consensus["consensus_label"].nunique()),
            "n_motrpac_tissues": int(consensus[consensus["motrpac_tissue"] != "unknown"]["motrpac_tissue"].nunique()),
        }
    logger.info(f"  {report.get('corpus', {}).get('n_cells', 0):,} cells")

    # ================================================================
    # 2. CONSENSUS SOURCE BREAKDOWN
    # ================================================================
    logger.info("2. Consensus source breakdown...")
    if not consensus.empty:
        source_counts = consensus["consensus_source"].value_counts()
        source_cells = consensus.groupby("consensus_source")["n_cells"].sum()
        report["consensus_sources"] = {
            src: {
                "n_clusters": int(source_counts.get(src, 0)),
                "n_cells": int(source_cells.get(src, 0)),
                "pct_clusters": round(100 * source_counts.get(src, 0) / len(consensus), 1),
            }
            for src in source_counts.index
        }

    # ================================================================
    # 3. TOP CELL TYPES (corpus-wide)
    # ================================================================
    logger.info("3. Top cell types...")
    if not consensus.empty:
        ct_cells = consensus.groupby("consensus_label")["n_cells"].sum().sort_values(ascending=False)
        total = ct_cells.sum()
        report["top_cell_types"] = [
            {
                "cell_type": ct,
                "n_cells": int(n),
                "pct": round(100 * n / total, 2),
                "n_clusters": int((consensus["consensus_label"] == ct).sum()),
            }
            for ct, n in ct_cells.head(30).items()
        ]

    # ================================================================
    # 4. PER-TISSUE CELL TYPE COMPOSITION
    # ================================================================
    logger.info("4. Per-tissue composition...")
    motrpac_tissues = [
        "heart", "kidney", "liver", "lung", "gastrocnemius",
        "cortex", "hippocampus", "hypothalamus",
        "blood RNA", "colon", "small intestine", "spleen",
        "BAT", "WAT-SC",
    ]
    tissue_data = {}
    if not consensus.empty:
        for tissue in motrpac_tissues:
            t = consensus[consensus["motrpac_tissue"] == tissue]
            if len(t) == 0:
                tissue_data[tissue] = {"status": "NO COVERAGE"}
                continue

            ct_cells = t.groupby("consensus_label")["n_cells"].sum().sort_values(ascending=False)
            total_cells = ct_cells.sum()
            n_studies = t["accession"].nunique()
            n_samples = t["sample_id"].nunique()

            # Source breakdown for this tissue
            t_sources = t["consensus_source"].value_counts().to_dict()

            # Confidence stats (from PanglaoDB confidence column)
            conf_stats = {}
            if "confidence" in t.columns and t["confidence"].notna().any():
                conf_stats = {
                    "median": round(float(t["confidence"].median()), 3),
                    "mean": round(float(t["confidence"].mean()), 3),
                    "q25": round(float(t["confidence"].quantile(0.25)), 3),
                    "q75": round(float(t["confidence"].quantile(0.75)), 3),
                }

            # Margin stats
            margin_stats = {}
            if "margin" in t.columns and t["margin"].notna().any():
                margin_stats = {
                    "median": round(float(t["margin"].median()), 3),
                    "mean": round(float(t["margin"].mean()), 3),
                }

            tissue_data[tissue] = {
                "n_cells": int(total_cells),
                "n_clusters": int(len(t)),
                "n_cell_types": int(ct_cells.count()),
                "n_studies": int(n_studies),
                "n_samples": int(n_samples),
                "confidence_stats": conf_stats,
                "margin_stats": margin_stats,
                "source_breakdown": {k: int(v) for k, v in t_sources.items()},
                "cell_types": [
                    {
                        "cell_type": ct,
                        "n_cells": int(n),
                        "pct": round(100 * n / total_cells, 1),
                    }
                    for ct, n in ct_cells.items()
                ],
            }

    report["motrpac_tissues"] = tissue_data
    logger.info(f"  {sum(1 for v in tissue_data.values() if v.get('n_cells', 0) > 0)} tissues with data")

    # ================================================================
    # 5. DECONVOLUTION READINESS PER TISSUE
    # ================================================================
    logger.info("5. Deconvolution readiness...")
    MIN_CELLS_PER_TYPE = 50  # minimum for scDEAL
    MIN_CELL_TYPES = 3       # minimum for meaningful deconvolution

    readiness = {}
    for tissue, data in tissue_data.items():
        if data.get("status") == "NO COVERAGE":
            readiness[tissue] = {
                "ready": False,
                "reason": "No single-cell coverage in training corpus",
                "n_cell_types_above_threshold": 0,
            }
            continue

        cell_types = data.get("cell_types", [])
        above_threshold = [ct for ct in cell_types if ct["n_cells"] >= MIN_CELLS_PER_TYPE]
        ready = len(above_threshold) >= MIN_CELL_TYPES

        readiness[tissue] = {
            "ready": ready,
            "n_cell_types_total": len(cell_types),
            "n_cell_types_above_threshold": len(above_threshold),
            "threshold": MIN_CELLS_PER_TYPE,
            "top_types_above_threshold": [
                {"cell_type": ct["cell_type"], "n_cells": ct["n_cells"]}
                for ct in above_threshold[:10]
            ],
            "types_below_threshold": [
                {"cell_type": ct["cell_type"], "n_cells": ct["n_cells"]}
                for ct in cell_types if ct["n_cells"] < MIN_CELLS_PER_TYPE
            ],
        }

    report["deconvolution_readiness"] = readiness
    n_ready = sum(1 for v in readiness.values() if v.get("ready"))
    logger.info(f"  {n_ready}/{len(readiness)} tissues ready for deconvolution")

    # ================================================================
    # 6. METHOD COMPARISON (PanglaoDB vs ScType vs Consensus)
    # ================================================================
    logger.info("6. Method comparison...")
    panglao = pd.read_csv(PANGLAO_SUMMARY, sep="\t") if PANGLAO_SUMMARY.exists() else pd.DataFrame()
    sctype = pd.read_csv(SCTYPE_SUMMARY, sep="\t") if SCTYPE_SUMMARY.exists() else pd.DataFrame()

    comparison = {}
    if not panglao.empty:
        panglao_ct = panglao.groupby("cell_type")["n_cells"].sum().sort_values(ascending=False)
        comparison["panglao"] = {
            "n_cell_types": int(panglao["cell_type"].nunique()),
            "top_5": [{"cell_type": ct, "n_cells": int(n)} for ct, n in panglao_ct.head(5).items()],
            "n_low_confidence": int(panglao["low_confidence"].sum()) if "low_confidence" in panglao.columns else 0,
        }

    if not sctype.empty:
        sctype_ct = sctype.groupby("cell_type")["n_cells"].sum().sort_values(ascending=False)
        n_unknown = int((sctype["cell_type"] == "Unknown").sum())
        comparison["sctype"] = {
            "n_cell_types": int(sctype["cell_type"].nunique()),
            "top_5": [{"cell_type": ct, "n_cells": int(n)} for ct, n in sctype_ct.head(5).items()],
            "n_unknown_clusters": n_unknown,
        }

    if not consensus.empty:
        consensus_ct = consensus.groupby("consensus_label")["n_cells"].sum().sort_values(ascending=False)
        comparison["consensus"] = {
            "n_cell_types": int(consensus["consensus_label"].nunique()),
            "top_5": [{"cell_type": ct, "n_cells": int(n)} for ct, n in consensus_ct.head(5).items()],
        }

    # Stellate cell tracking across methods
    if not panglao.empty and not consensus.empty:
        panglao_stellate = panglao[panglao["cell_type"].str.contains("stellate", case=False, na=False)]
        consensus_stellate = consensus[consensus["consensus_label"].str.contains("stellate", case=False, na=False)]
        comparison["stellate_cell_fix"] = {
            "panglao_total": int(panglao_stellate["n_cells"].sum()),
            "consensus_total": int(consensus_stellate["n_cells"].sum()),
            "reduction_pct": round(100 * (1 - consensus_stellate["n_cells"].sum() / max(panglao_stellate["n_cells"].sum(), 1)), 1),
            "panglao_by_tissue": panglao_stellate.groupby("motrpac_tissue")["n_cells"].sum().sort_values(ascending=False).head(10).to_dict(),
            "consensus_by_tissue": consensus_stellate.groupby("motrpac_tissue")["n_cells"].sum().sort_values(ascending=False).head(10).to_dict(),
        }

    report["method_comparison"] = comparison

    # ================================================================
    # 7. NOTABLE LLM CORRECTIONS
    # ================================================================
    logger.info("7. LLM corrections...")
    consensus_map = load_json(CONSENSUS_MAP)
    corrections = []
    if "map" in consensus_map:
        for key, val in consensus_map["map"].items():
            if val.get("source") == "corrected":
                parts = key.split("|||")
                if len(parts) == 3:
                    corrections.append({
                        "panglao": parts[0],
                        "sctype": parts[1],
                        "tissue": parts[2],
                        "consensus": val["consensus"],
                        "rationale": val.get("rationale", ""),
                    })

    # Sort by most impactful corrections (need cell counts)
    if not consensus.empty and corrections:
        correction_cells = {}
        for _, row in consensus.iterrows():
            if row["consensus_source"] == "corrected":
                key = f"{row.get('panglao_label', '')}|||{row.get('sctype_label', '')}|||{row.get('motrpac_tissue', 'unknown')}"
                correction_cells[key] = correction_cells.get(key, 0) + int(row.get("n_cells", 0))

        for corr in corrections:
            key = f"{corr['panglao']}|||{corr['sctype']}|||{corr['tissue']}"
            corr["n_cells"] = correction_cells.get(key, 0)

        corrections.sort(key=lambda x: x.get("n_cells", 0), reverse=True)

    report["llm_corrections"] = {
        "total": len(corrections),
        "top_corrections": corrections[:30],
    }
    logger.info(f"  {len(corrections)} corrections, top: {corrections[0]['panglao']} → {corrections[0]['consensus']} in {corrections[0]['tissue']}" if corrections else "  No corrections")

    # ================================================================
    # 8. TOP DE MARKERS PER CELL TYPE PER TISSUE
    # ================================================================
    logger.info("8. Top DE markers per cell type per tissue...")
    de_by_tissue_ct = defaultdict(lambda: defaultdict(list))

    if not consensus.empty:
        for tissue in motrpac_tissues:
            t = consensus[consensus["motrpac_tissue"] == tissue]
            if len(t) == 0:
                continue

            # Get representative samples for top cell types
            ct_cells = t.groupby("consensus_label")["n_cells"].sum().sort_values(ascending=False)

            for ct in ct_cells.head(10).index:
                ct_rows = t[t["consensus_label"] == ct].nlargest(1, "n_cells")
                if ct_rows.empty:
                    continue

                sample_id = ct_rows.iloc[0]["sample_id"]
                cluster = str(ct_rows.iloc[0]["cluster"])

                # Try PanglaoDB markers
                markers_file = PANGLAO_DIR / sample_id / f"{sample_id}_top_markers.tsv"
                if not markers_file.exists():
                    markers_file = SCTYPE_DIR / sample_id / f"{sample_id}_sctype_top_markers.tsv"

                if markers_file.exists():
                    try:
                        de = pd.read_csv(markers_file, sep="\t")
                        cluster_de = de[de["cluster"].astype(str) == cluster]
                        if not cluster_de.empty:
                            top5 = cluster_de.nsmallest(5, "rank")
                            genes = []
                            for _, m in top5.iterrows():
                                genes.append({
                                    "symbol": m.get("gene_symbol", m.get("gene_id", "?")),
                                    "score": round(float(m.get("score", 0)), 2),
                                })
                            de_by_tissue_ct[tissue][ct] = genes
                    except Exception:
                        pass

    report["de_markers_by_tissue"] = {
        tissue: {ct: markers for ct, markers in cts.items()}
        for tissue, cts in de_by_tissue_ct.items()
    }
    n_with_markers = sum(len(cts) for cts in de_by_tissue_ct.values())
    logger.info(f"  {n_with_markers} cell type × tissue pairs with DE markers")

    # ================================================================
    # 9. SAMPLE METADATA SUMMARY
    # ================================================================
    logger.info("9. Sample metadata...")
    inventory = load_json(ANNOTATION_INVENTORY)
    if inventory.get("samples"):
        samples = inventory["samples"]
        strains = Counter(s.get("strain_resolved", "unknown") for s in samples)
        sexes = Counter(s.get("sex_resolved", "unknown") for s in samples)
        conditions = Counter(s.get("condition_resolved") or "unknown" for s in samples)

        report["sample_metadata"] = {
            "n_samples": len(samples),
            "strain_distribution": dict(strains.most_common(10)),
            "sex_distribution": dict(sexes.most_common()),
            "condition_distribution": dict(conditions.most_common(10)),
            "pct_with_tissue": round(100 * sum(1 for s in samples if s.get("tissue_resolved")) / len(samples), 1),
            "pct_with_condition": round(100 * sum(1 for s in samples if s.get("condition_resolved")) / len(samples), 1),
        }

    # ================================================================
    # 10. MARKER DATABASE STATS
    # ================================================================
    logger.info("10. Marker database stats...")
    import pickle

    # PanglaoDB
    if MARKER_DB.exists():
        with open(MARKER_DB, "rb") as f:
            pdb = pickle.load(f)
        report["panglao_db"] = pdb.get("build_info", {})

    # ScType
    if SCTYPE_DB.exists():
        with open(SCTYPE_DB, "rb") as f:
            sdb = pickle.load(f)
        report["sctype_db"] = sdb.get("build_info", {})

    # ================================================================
    # 11. TISSUE COVERAGE GAPS
    # ================================================================
    logger.info("11. Coverage gaps...")
    all_18 = [
        "gastrocnemius", "vastus lateralis", "heart", "liver", "kidney",
        "lung", "WAT-SC", "BAT", "adrenal", "blood RNA", "colon",
        "small intestine", "hippocampus", "hypothalamus", "cortex",
        "spleen", "ovary", "testis",
    ]
    covered = set(tissue_data.keys()) - {"unknown"}
    gaps = [t for t in all_18 if tissue_data.get(t, {}).get("status") == "NO COVERAGE" or t not in tissue_data]
    thin = [
        {"tissue": t, "n_cells": tissue_data[t]["n_cells"], "n_samples": tissue_data[t].get("n_samples", 0)}
        for t in covered
        if tissue_data.get(t, {}).get("n_cells", 0) < 50000 and t in tissue_data
    ]

    report["coverage_gaps"] = {
        "missing_tissues": gaps,
        "thin_tissues": sorted(thin, key=lambda x: x["n_cells"]),
    }
    logger.info(f"  Missing: {gaps}")
    logger.info(f"  Thin: {[t['tissue'] for t in thin]}")

    # ================================================================
    # 12. FORGETTING ANALYSIS SUMMARY
    # ================================================================
    logger.info("12. Forgetting analysis...")
    forgetting = load_json(FORGETTING_RESULTS)
    if forgetting:
        report["forgetting_analysis"] = {
            "available": True,
            "n_checkpoints": len(forgetting.get("checkpoints", [])),
            "summary": forgetting.get("summary", {}),
        }
    else:
        report["forgetting_analysis"] = {"available": False}

    # ================================================================
    # 13. ANNOTATION PIPELINE PARAMETERS
    # ================================================================
    report["pipeline_parameters"] = {
        "leiden_resolution": 0.8,
        "min_marker_genes_panglao": 3,
        "confidence_threshold_panglao": 0.1,
        "sctype_unknown_threshold": "n_cells/4",
        "normalization": "normalize_total(target_sum=1e4) + log1p (base e)",
        "hvg_selection": "seurat_v3, n_top_genes=2000",
        "pca_components": "min(50, n_cells-1, n_hvg-1)",
        "n_neighbors": "min(15, n_cells-1)",
        "de_method": "wilcoxon",
        "de_n_genes": 10,
        "consensus_llm_model": "claude-haiku-4-5-20251001",
        "consensus_batch_size": 50,
    }

    # ================================================================
    # 14. PER-TISSUE PANGLAO vs SCTYPE vs CONSENSUS COMPARISON
    # ================================================================
    logger.info("14. Per-tissue method comparison...")
    tissue_method_comparison = {}

    if not panglao.empty and not sctype.empty and not consensus.empty:
        panglao["motrpac_tissue"] = panglao["motrpac_tissue"].fillna("unknown")
        sctype["motrpac_tissue"] = sctype["motrpac_tissue"].fillna("unknown")

        for tissue in motrpac_tissues:
            p = panglao[panglao["motrpac_tissue"] == tissue]
            s = sctype[sctype["motrpac_tissue"] == tissue]
            c = consensus[consensus["motrpac_tissue"] == tissue]

            if len(p) == 0:
                continue

            p_top = p.groupby("cell_type")["n_cells"].sum().sort_values(ascending=False)
            s_top = s.groupby("cell_type")["n_cells"].sum().sort_values(ascending=False)
            c_top = c.groupby("consensus_label")["n_cells"].sum().sort_values(ascending=False)

            tissue_method_comparison[tissue] = {
                "panglao_top5": [{"type": ct, "cells": int(n)} for ct, n in p_top.head(5).items()],
                "sctype_top5": [{"type": ct, "cells": int(n)} for ct, n in s_top.head(5).items()],
                "consensus_top5": [{"type": ct, "cells": int(n)} for ct, n in c_top.head(5).items()],
                "panglao_n_types": int(p_top.count()),
                "sctype_n_types": int(s_top.count()),
                "consensus_n_types": int(c_top.count()),
            }

    report["tissue_method_comparison"] = tissue_method_comparison

    # ================================================================
    # 15. DECONVOLUTION TOOL COMPATIBILITY
    # ================================================================
    logger.info("15. Deconvolution tool compatibility...")
    report["deconvolution_tools"] = {
        "UniCell": {
            "requires": "Pre-trained cell type signatures (human-trained UCDBase, or UCDSelect with custom references)",
            "rat_compatibility": "Requires fine-tuning or custom reference signatures from these annotations",
            "data_needed": "Cell-type-labeled scRNA-seq reference panel per tissue",
            "status": "Annotations now available; reference construction pending",
        },
        "scDEAL": {
            "requires": "Labeled scRNA-seq reference + bulk RNA-seq query",
            "rat_compatibility": "KNN embedding transfer is species-agnostic; uses cluster structure",
            "data_needed": "Cell-type labels + expression data per tissue",
            "status": "Ready — consensus annotations provide labels",
        },
        "Scissor": {
            "requires": "scRNA-seq + bulk RNA-seq + phenotype labels",
            "rat_compatibility": "Fully compatible; identifies phenotype-associated subpopulations",
            "data_needed": "No cell-type labels required (discovers from data)",
            "status": "Ready — MoTrPAC bulk + our scRNA-seq corpus",
            "note": "~40 runs across tissues × timepoints; requires BH FDR correction",
        },
    }

    # ================================================================
    # SAVE
    # ================================================================
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(report, f, indent=2, default=str)

    logger.info(f"\nSaved to {OUTPUT_PATH}")
    logger.info(f"  Total size: {OUTPUT_PATH.stat().st_size / 1024:.0f} KB")

    # Print key stats
    logger.info("\n" + "=" * 60)
    logger.info("KEY STATS FOR REPORT")
    logger.info("=" * 60)
    logger.info(f"  Cells: {report['corpus']['n_cells']:,}")
    logger.info(f"  Samples: {report['corpus']['n_samples']}")
    logger.info(f"  Cell types: {report['corpus']['n_cell_types']}")
    logger.info(f"  MoTrPAC tissues: {report['corpus']['n_motrpac_tissues']}")
    logger.info(f"  Tissues ready for deconvolution: {n_ready}/{len(readiness)}")
    logger.info(f"  Missing tissues: {gaps}")
    logger.info(f"  LLM corrections: {len(corrections)}")
    logger.info(f"  Stellate reduction: {comparison.get('stellate_cell_fix', {}).get('reduction_pct', '?')}%")
    logger.info(f"  DE markers extracted: {n_with_markers} type×tissue pairs")


if __name__ == "__main__":
    main()