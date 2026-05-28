#!/usr/bin/env python3
"""
consensus_annotations.py -- Merge PanglaoDB and ScType cell-type annotations
using LLM-based synonym resolution and tissue-context correction.

Pipeline:
  1. Load PanglaoDB annotation_summary.tsv and ScType sctype_annotation_summary.tsv
  2. Join on (sample_id, cluster)
  3. Extract unique (panglao_label, sctype_label, tissue) triples
  4. Attach top DE markers for representative clusters
  5. Send triples to Claude Haiku for consensus label resolution (~1 API call)
  6. Cache consensus map as JSON
  7. Apply to all 17K clusters
  8. Produce final consensus_annotations.tsv and corpus summaries

Usage:
  # Full pipeline (requires ANTHROPIC_API_KEY)
  python consensus_annotations.py merge

  # Use cached consensus map (no API call)
  python consensus_annotations.py merge --use-cache

  # Just show the unique triples without calling API
  python consensus_annotations.py show-triples

  # Re-run LLM resolution only (regenerate cache)
  python consensus_annotations.py resolve

  # Summarize consensus annotations
  python consensus_annotations.py summarize
"""

import argparse
import json
import logging
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import re
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path("/depot/reese18/apps/motrpac-genecompass")
PANGLAO_DIR = PROJECT_ROOT / "data/training/cell_annotations"
SCTYPE_DIR = PROJECT_ROOT / "data/training/cell_annotations_sctype"
OUTPUT_DIR = PROJECT_ROOT / "data/training/cell_annotations_consensus"
REPORTS_DIR = PROJECT_ROOT / "reports/annotations"

PANGLAO_SUMMARY = PANGLAO_DIR / "_corpus_summary" / "annotation_summary.tsv"
SCTYPE_SUMMARY = SCTYPE_DIR / "_corpus_summary" / "sctype_annotation_summary.tsv"
CONSENSUS_MAP_PATH = OUTPUT_DIR / "consensus_label_map.json"


# ============================================================================
# Step 1: Load and join annotations
# ============================================================================
def load_and_join() -> pd.DataFrame:
    """Load PanglaoDB and ScType summaries, join on (sample_id, cluster)."""

    logger.info("Loading PanglaoDB annotations...")
    if not PANGLAO_SUMMARY.exists():
        logger.error(f"Not found: {PANGLAO_SUMMARY}")
        logger.error("Run: python analysis/annotate_celltypes.py summarize")
        sys.exit(1)
    panglao = pd.read_csv(PANGLAO_SUMMARY, sep="\t")
    logger.info(f"  {len(panglao)} PanglaoDB cluster assignments")

    logger.info("Loading ScType annotations...")
    if not SCTYPE_SUMMARY.exists():
        logger.error(f"Not found: {SCTYPE_SUMMARY}")
        logger.error("Run: python analysis/annotate_sctype.py summarize")
        sys.exit(1)
    sctype = pd.read_csv(SCTYPE_SUMMARY, sep="\t")
    logger.info(f"  {len(sctype)} ScType cluster assignments")

    # Normalize cluster column to string
    panglao["cluster"] = panglao["cluster"].astype(str)
    sctype["cluster"] = sctype["cluster"].astype(str)

    # Join on sample_id + cluster
    merged = panglao.merge(
        sctype[["sample_id", "cluster", "cell_type", "sctype_score"]],
        on=["sample_id", "cluster"],
        how="left",
        suffixes=("_panglao", "_sctype"),
    )

    # Rename for clarity
    merged.rename(columns={
        "cell_type_panglao": "panglao_label",
        "cell_type_sctype": "sctype_label",
    }, inplace=True)

    # Fill missing ScType (samples not yet processed)
    merged["sctype_label"] = merged["sctype_label"].fillna("(not scored)")

    n_both = (merged["sctype_label"] != "(not scored)").sum()
    n_panglao_only = (merged["sctype_label"] == "(not scored)").sum()
    logger.info(f"  Joined: {len(merged)} clusters ({n_both} with both, {n_panglao_only} PanglaoDB only)")

    return merged


# ============================================================================
# Step 2: Extract unique triples
# ============================================================================
def extract_triples(merged: pd.DataFrame) -> pd.DataFrame:
    """
    Extract unique (panglao_label, sctype_label, motrpac_tissue) triples
    with cell counts and representative DE markers.
    """
    # Group by triple
    merged["motrpac_tissue"] = merged["motrpac_tissue"].fillna("unknown")
    triples = merged.groupby(
        ["panglao_label", "sctype_label", "motrpac_tissue"], dropna=False
    ).agg(
        n_clusters=("cluster", "count"),
        n_cells=("n_cells", "sum"),
        mean_confidence=("confidence", "mean"),
        mean_margin=("margin", "mean"),
    ).reset_index()

    triples = triples.sort_values("n_cells", ascending=False).reset_index(drop=True)
    logger.info(f"  {len(triples)} unique (PanglaoDB, ScType, tissue) triples")

    return triples


def attach_de_markers(triples: pd.DataFrame, merged: pd.DataFrame, max_per_triple: int = 1) -> pd.DataFrame:
    """
    For each triple, find a representative cluster and attach its top DE markers.
    """
    de_markers_col = []

    for _, triple in triples.iterrows():
        p_label = triple["panglao_label"]
        s_label = triple["sctype_label"]
        tissue = triple["motrpac_tissue"]

        # Find representative clusters matching this triple
        mask = (
            (merged["panglao_label"] == p_label) &
            (merged["sctype_label"] == s_label) &
            (merged["motrpac_tissue"] == tissue)
        )
        candidates = merged[mask].nlargest(max_per_triple, "n_cells")

        markers_str = ""
        for _, cand in candidates.iterrows():
            sample_id = cand["sample_id"]
            cluster = cand["cluster"]

            # Try PanglaoDB DE markers first
            markers_file = PANGLAO_DIR / sample_id / f"{sample_id}_top_markers.tsv"
            if markers_file.exists():
                try:
                    de = pd.read_csv(markers_file, sep="\t")
                    cluster_de = de[de["cluster"].astype(str) == str(cluster)]
                    if not cluster_de.empty:
                        top_genes = cluster_de.nsmallest(5, "rank")["gene_symbol"].tolist()
                        markers_str = ", ".join(top_genes)
                except Exception:
                    pass

            # Try ScType DE markers if PanglaoDB didn't work
            if not markers_str:
                markers_file = SCTYPE_DIR / sample_id / f"{sample_id}_sctype_top_markers.tsv"
                if markers_file.exists():
                    try:
                        de = pd.read_csv(markers_file, sep="\t")
                        cluster_de = de[de["cluster"].astype(str) == str(cluster)]
                        if not cluster_de.empty:
                            top_genes = cluster_de.nsmallest(5, "rank")["gene_symbol"].tolist()
                            markers_str = ", ".join(top_genes)
                    except Exception:
                        pass

        de_markers_col.append(markers_str)

    triples["de_markers"] = de_markers_col
    n_with_markers = sum(1 for m in de_markers_col if m)
    logger.info(f"  {n_with_markers}/{len(triples)} triples have DE marker context")

    return triples


# ============================================================================
# Step 3: LLM consensus resolution
# ============================================================================
def resolve_consensus_llm(triples: pd.DataFrame) -> Dict[str, dict]:
    """
    Send unique triples to Claude Haiku for consensus label resolution.

    Returns dict keyed by "panglao_label|||sctype_label|||tissue" with:
      - consensus_label: final cell type name
      - source: "panglao", "sctype", "merged", or "corrected"
      - rationale: brief explanation
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        logger.error("ANTHROPIC_API_KEY not set")
        sys.exit(1)

    try:
        import anthropic
    except ImportError:
        logger.error("anthropic package not installed. Run: pip install anthropic --break-system-packages")
        sys.exit(1)

    # Build the input for the LLM
    entries = []
    for _, row in triples.iterrows():
        entry = {
            "panglao": row["panglao_label"],
            "sctype": row["sctype_label"],
            "tissue": row["motrpac_tissue"] if pd.notna(row["motrpac_tissue"]) else "unknown",
            "n_cells": int(row["n_cells"]),
            "de_markers": row.get("de_markers", ""),
        }
        entries.append(entry)

    prompt = f"""You are a single-cell biology expert. I have cell-type annotations from two automated methods (PanglaoDB marker scoring and ScType tissue-specific scoring) for rat scRNA-seq data. For each entry below, provide a single consensus cell type label.

Rules:
1. When both methods identify the same cell type with different names (synonyms), use the most standard/widely-accepted name.
   Examples: "T cytotoxic cells" + "Memory CD8+ T cells" → "CD8+ T cells"; "B cells naive" + "Naive B cells" → "Naive B cells"

2. When PanglaoDB gives a tissue-inappropriate label but ScType gives a tissue-appropriate one, prefer ScType.
   Examples: "Pancreatic stellate cells" in heart + "Stromal cells (Heart)" → "Cardiac fibroblasts"; "Alveolar macrophages" in liver + "Kupffer cells" → "Kupffer cells"

3. When ScType says "Unknown" but PanglaoDB gives a biologically plausible label for that tissue, use PanglaoDB.
   Examples: PanglaoDB "Cardiomyocytes" in heart + ScType "Unknown" → "Cardiomyocytes"

4. When ScType says "(not scored)" (sample wasn't processed by ScType), use PanglaoDB but flag if the label is tissue-inappropriate.

5. When both give plausible but different labels, prefer the more specific one, using DE markers as tiebreaker.

6. Use standardized names: "Fibroblasts" not "Stromal cells", "Kupffer cells" not "Liver macrophages", etc. Include tissue context only when needed for disambiguation (e.g., "Cardiac fibroblasts" vs "Pulmonary fibroblasts").

7. For immune cell types present in any tissue, use the standard immunology name without tissue prefix.

8. Keep labels concise (1-4 words typical, 5 max).

Input entries (JSON array):
{json.dumps(entries, indent=2)}

Return ONLY a JSON array where each element has:
- "panglao": original PanglaoDB label (for matching)
- "sctype": original ScType label (for matching)
- "tissue": tissue context (for matching)
- "consensus": the final consensus label
- "source": one of "panglao", "sctype", "merged", "corrected"
- "rationale": brief explanation (10 words max)

No markdown, no explanation outside the JSON."""

    # Call API
    client = anthropic.Anthropic(api_key=api_key)
    logger.info(f"Calling Claude API to resolve {len(entries)} triples...")

    # Split into batches if needed (Haiku context limit)
    batch_size = 50
    all_results = []

    for i in range(0, len(entries), batch_size):
        batch_entries = entries[i:i + batch_size]
        batch_prompt = prompt.replace(
            json.dumps(entries, indent=2),
            json.dumps(batch_entries, indent=2)
        )

        message = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=8192,
            messages=[{"role": "user", "content": batch_prompt}],
        )

        response_text = message.content[0].text.strip()

        # Parse response
        try:
            # Strip markdown fences if present
            if response_text.startswith("```"):
                response_text = response_text.split("\n", 1)[1]
                response_text = response_text.rsplit("```", 1)[0]
            # Fix common JSON issues from LLM output
            response_text = response_text.strip()
            # Remove trailing commas before ] or }
            response_text = re.sub(r',\s*([}\]])', r'\1', response_text)
            # Truncate at last complete object if array is cut off
            if not response_text.endswith(']'):
                last_brace = response_text.rfind('}')
                if last_brace > 0:
                    response_text = response_text[:last_brace + 1] + ']'
            batch_results = json.loads(response_text)
            all_results.extend(batch_results)
            logger.info(f"  Batch {i // batch_size + 1}: {len(batch_results)} labels resolved")
        except json.JSONDecodeError as e:
            logger.error(f"  Failed to parse LLM response: {e}")
            logger.error(f"  Response (first 500 chars): {response_text[:500]}")
            continue

    # Build lookup dict
    consensus_map = {}
    for r in all_results:
        key = f"{r['panglao']}|||{r['sctype']}|||{r['tissue']}"
        consensus_map[key] = {
            "consensus": r["consensus"],
            "source": r.get("source", "merged"),
            "rationale": r.get("rationale", ""),
        }

    logger.info(f"  Resolved {len(consensus_map)} consensus labels")
    return consensus_map


def save_consensus_map(consensus_map: Dict, triples: pd.DataFrame):
    """Save consensus map with metadata."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output = {
        "generated_at": datetime.now().isoformat(),
        "n_triples": len(consensus_map),
        "model": "claude-haiku-4-5-20251001",
        "map": consensus_map,
    }

    with open(CONSENSUS_MAP_PATH, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"  Cached to {CONSENSUS_MAP_PATH}")


def load_consensus_map() -> Dict[str, dict]:
    """Load cached consensus map."""
    if not CONSENSUS_MAP_PATH.exists():
        logger.error(f"Consensus map not found: {CONSENSUS_MAP_PATH}")
        logger.error("Run: python consensus_annotations.py merge (without --use-cache)")
        sys.exit(1)
    with open(CONSENSUS_MAP_PATH) as f:
        data = json.load(f)
    logger.info(f"  Loaded cached consensus map ({data['n_triples']} triples from {data['generated_at']})")
    return data["map"]


# ============================================================================
# Step 4: Apply consensus labels
# ============================================================================
def apply_consensus(merged: pd.DataFrame, consensus_map: Dict[str, dict]) -> pd.DataFrame:
    """Apply consensus labels to all clusters."""

    consensus_labels = []
    consensus_sources = []
    n_resolved = 0
    n_fallback = 0

    for _, row in merged.iterrows():
        p_label = row["panglao_label"]
        s_label = row["sctype_label"]
        tissue = row["motrpac_tissue"] if pd.notna(row["motrpac_tissue"]) else "unknown"

        key = f"{p_label}|||{s_label}|||{tissue}"
        entry = consensus_map.get(key)

        if entry:
            consensus_labels.append(entry["consensus"])
            consensus_sources.append(entry["source"])
            n_resolved += 1
        else:
            # Fallback: use PanglaoDB label
            consensus_labels.append(p_label)
            consensus_sources.append("fallback_panglao")
            n_fallback += 1

    merged["consensus_label"] = consensus_labels
    merged["consensus_source"] = consensus_sources

    logger.info(f"  Applied: {n_resolved} resolved, {n_fallback} fallback")
    return merged


# ============================================================================
# Step 5: Write outputs
# ============================================================================
def write_outputs(merged: pd.DataFrame):
    """Write consensus annotation files."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_dir = OUTPUT_DIR / "_corpus_summary"
    summary_dir.mkdir(exist_ok=True)

    # 1. Full consensus annotations
    out_cols = [
        "sample_id", "accession", "motrpac_tissue", "tissue_normalized",
        "sex", "strain", "cluster", "n_cells", "pct_cells",
        "panglao_label", "confidence", "margin",
        "sctype_label", "sctype_score",
        "consensus_label", "consensus_source",
    ]
    available_cols = [c for c in out_cols if c in merged.columns]
    consensus_path = summary_dir / "consensus_annotations.tsv"
    merged[available_cols].to_csv(consensus_path, sep="\t", index=False)
    logger.info(f"  {consensus_path}: {len(merged)} clusters")

    # 2. Cell type x tissue matrix
    ct_tissue = defaultdict(Counter)
    for _, row in merged.iterrows():
        tissue = row.get("motrpac_tissue", "unknown")
        if pd.isna(tissue):
            tissue = "unknown"
        ct_tissue[tissue][row["consensus_label"]] += int(row.get("n_cells", 0))

    ct_tissue_df = pd.DataFrame(ct_tissue).fillna(0).astype(int)
    ct_tissue_df.index.name = "cell_type"
    ct_tissue_df = ct_tissue_df.loc[ct_tissue_df.sum(axis=1).sort_values(ascending=False).index]
    ct_matrix_path = summary_dir / "consensus_cell_type_by_tissue.tsv"
    ct_tissue_df.to_csv(ct_matrix_path, sep="\t")
    logger.info(f"  {ct_matrix_path}: {len(ct_tissue_df)} cell types x {len(ct_tissue_df.columns)} tissues")

    # 3. Per-sample consensus (for downstream tools)
    for sample_id, group in merged.groupby("sample_id"):
        sample_dir = OUTPUT_DIR / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        sample_cols = ["cluster", "n_cells", "panglao_label", "sctype_label",
                       "consensus_label", "consensus_source"]
        avail = [c for c in sample_cols if c in group.columns]
        group[avail].to_csv(sample_dir / f"{sample_id}_consensus.tsv", sep="\t", index=False)

    # 4. Source breakdown
    source_counts = merged["consensus_source"].value_counts()
    logger.info(f"\n  Consensus source breakdown:")
    for source, count in source_counts.items():
        pct = 100 * count / len(merged)
        logger.info(f"    {source:20s}  {count:>6} clusters  ({pct:.1f}%)")

    # 5. QC report
    all_ct = Counter()
    for _, row in merged.iterrows():
        all_ct[row["consensus_label"]] += int(row.get("n_cells", 0))

    total_cells = sum(all_ct.values())

    qc = {
        "generated_at": datetime.now().isoformat(),
        "n_clusters": len(merged),
        "n_cells": total_cells,
        "n_unique_cell_types": len(all_ct),
        "n_samples": merged["sample_id"].nunique(),
        "source_breakdown": source_counts.to_dict(),
        "top_cell_types": dict(all_ct.most_common(30)),
        "motrpac_tissue_coverage": {
            tissue: {
                "n_cell_types": len(cts),
                "n_cells": sum(cts.values()),
                "top_types": dict(cts.most_common(10)),
            }
            for tissue, cts in sorted(ct_tissue.items())
            if tissue != "unknown"
        },
    }
    qc_path = summary_dir / "consensus_qc_report.json"
    with open(qc_path, "w") as f:
        json.dump(qc, f, indent=2)
    logger.info(f"  {qc_path}")

    # Print summary
    logger.info("=" * 60)
    logger.info("CONSENSUS ANNOTATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  Clusters: {len(merged):,}")
    logger.info(f"  Cells: {total_cells:,}")
    logger.info(f"  Unique cell types: {len(all_ct)}")
    logger.info(f"\n  Top cell types:")
    for ct, n in all_ct.most_common(15):
        logger.info(f"    {ct:35s}  {n:>8,} cells")

    logger.info(f"\n  MoTrPAC tissue coverage:")
    for tissue, cts in sorted(ct_tissue.items()):
        if tissue == "unknown":
            continue
        n_cells = sum(cts.values())
        n_types = len(cts)
        logger.info(f"    {tissue:20s}  {n_types:3d} cell types  {n_cells:>8,} cells")


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="Consensus cell-type annotation merger")
    subparsers = parser.add_subparsers(dest="command", required=True)

    sub_merge = subparsers.add_parser("merge", help="Full merge pipeline")
    sub_merge.add_argument("--use-cache", action="store_true",
                           help="Use cached consensus map (skip API call)")

    sub_triples = subparsers.add_parser("show-triples",
                                         help="Show unique triples without API call")

    sub_resolve = subparsers.add_parser("resolve",
                                         help="Re-run LLM resolution (regenerate cache)")

    sub_summary = subparsers.add_parser("summarize",
                                         help="Re-summarize from cached consensus")

    args = parser.parse_args()

    if args.command == "show-triples":
        merged = load_and_join()
        triples = extract_triples(merged)
        triples = attach_de_markers(triples, merged)

        logger.info(f"\n{'='*100}")
        logger.info(f"{'PanglaoDB':35s}  {'ScType':35s}  {'Tissue':15s}  {'Cells':>8s}  DE Markers")
        logger.info(f"{'='*100}")
        for _, row in triples.head(50).iterrows():
            p = row["panglao_label"][:35]
            s = row["sctype_label"][:35]
            t = str(row["motrpac_tissue"])[:15] if pd.notna(row["motrpac_tissue"]) else "unknown"
            n = int(row["n_cells"])
            de = row.get("de_markers", "")[:40]
            logger.info(f"  {p:35s}  {s:35s}  {t:15s}  {n:>8,}  {de}")

        logger.info(f"\n  Total unique triples: {len(triples)}")
        logger.info(f"  Total cells covered: {triples['n_cells'].sum():,}")

    elif args.command == "resolve":
        merged = load_and_join()
        triples = extract_triples(merged)
        triples = attach_de_markers(triples, merged)
        consensus_map = resolve_consensus_llm(triples)
        save_consensus_map(consensus_map, triples)

    elif args.command == "merge":
        merged = load_and_join()
        triples = extract_triples(merged)
        triples = attach_de_markers(triples, merged)

        if args.use_cache:
            consensus_map = load_consensus_map()
        else:
            consensus_map = resolve_consensus_llm(triples)
            save_consensus_map(consensus_map, triples)

        merged = apply_consensus(merged, consensus_map)
        write_outputs(merged)

    elif args.command == "summarize":
        merged = load_and_join()
        consensus_map = load_consensus_map()
        merged = apply_consensus(merged, consensus_map)
        write_outputs(merged)


if __name__ == "__main__":
    main()