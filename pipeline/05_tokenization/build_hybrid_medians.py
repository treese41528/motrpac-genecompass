#!/usr/bin/env python3
"""
build_hybrid_medians.py — Stage 5, Pre-Step: Build Hybrid Gene Median Dictionary

Pipeline position:
    Stage 4: Gene Medians  → rat_gene_medians.pickle
    Stage 3: Ortholog Mapping → rat_to_human_mapping.pickle
    Stage 5 Pre-Step: build_hybrid_medians.py   ← THIS SCRIPT
    Stage 5, Step 1: tokenize_corpus.py         (reads hybrid_gene_medians.pickle)

Purpose:
    Construct a per-gene median dictionary that uses human median expression
    values (from GeneCompass's human_gene_median_after_filter.pickle) for
    ortholog-mapped rat genes, and falls back to rat-computed medians for
    Tier 4 rat-specific genes with no human ortholog.

    This hybrid approach is required for value-scale compatibility with the
    GeneCompass pre-trained model:
      - The pre-training corpus was normalized by dividing raw counts directly
        by human gene medians (verified empirically by comparing value
        distributions against randsel_5w_mouse reference corpus).
      - Using rat-specific medians alone produces a compressed distribution
        (std 0.25 vs reference 0.63) with insufficient dynamic range.
      - Using human medians for ortholog-mapped genes (69.3% of eligible genes)
        closely matches the reference distribution (median offset -0.075 vs
        reference, std offset -0.096).

Coverage (from compare_normalization.py on 3,000 rat cells):
    Human medians: 14,812 / 21,379 genes (69.3%)
    Rat medians:    6,567 / 21,379 genes (30.7%)
    Fallback (1.0):     0 genes

Outputs (written to paths.median_dir):
    hybrid_gene_medians.pickle  — {ENSRNOG_base: median_float}
    hybrid_median_provenance.json — per-gene source, coverage stats

Usage:
    python pipeline/05_tokenization/build_hybrid_medians.py
    python pipeline/05_tokenization/build_hybrid_medians.py --dry-run
    python pipeline/05_tokenization/build_hybrid_medians.py -v

Author: Tim Reese Lab / Claude
Date: March 2026
"""

import argparse
import json
import logging
import os
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# Project path bootstrap
# ─────────────────────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(os.environ.get('PIPELINE_ROOT', '.')).resolve()
sys.path.insert(0, str(_PROJECT_ROOT / 'lib'))
from gene_utils import load_config, resolve_path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
)
logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1: LOAD INPUTS
# ═════════════════════════════════════════════════════════════════════════════

def load_rat_medians(median_dir: Path) -> Dict[str, float]:
    """Load rat_gene_medians.pickle → {ENSRNOG_base: float}."""
    p = median_dir / 'rat_gene_medians.pickle'
    if not p.exists():
        raise FileNotFoundError(
            f"rat_gene_medians.pickle not found: {p}\n"
            f"  → Run Stage 4 (python run_stage4.py) first."
        )
    with open(p, 'rb') as f:
        raw = pickle.load(f)
    result = {str(k).strip().split('.')[0].upper(): float(v)
              for k, v in raw.items() if float(v) > 0}
    logger.info(f"Loaded rat_gene_medians.pickle: {len(result):,} genes")
    return result


def load_rat_to_human(ortholog_dir: Path) -> Dict[str, str]:
    """Load rat_to_human_mapping.pickle → {ENSRNOG_base: ENSG}."""
    p = ortholog_dir / 'rat_to_human_mapping.pickle'
    if not p.exists():
        raise FileNotFoundError(
            f"rat_to_human_mapping.pickle not found: {p}\n"
            f"  → Run Stage 3 (python run_stage3.py) first."
        )
    with open(p, 'rb') as f:
        raw = pickle.load(f)
    result = {str(k).strip().split('.')[0].upper(): str(v).strip()
              for k, v in raw.items()}
    logger.info(f"Loaded rat_to_human_mapping.pickle: {len(result):,} mappings")
    return result


def load_human_medians(gc_medians_path: Path) -> Dict[str, float]:
    """Load GeneCompass human_gene_median_after_filter.pickle → {ENSG: float}."""
    if not gc_medians_path.exists():
        raise FileNotFoundError(
            f"human_gene_median_after_filter.pickle not found: {gc_medians_path}"
        )
    with open(gc_medians_path, 'rb') as f:
        raw = pickle.load(f)
    result = {str(k).strip(): float(v) for k, v in raw.items() if float(v) > 0}
    logger.info(f"Loaded human medians: {len(result):,} genes")
    return result


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2: BUILD HYBRID DICTIONARY
# ═════════════════════════════════════════════════════════════════════════════

def build_hybrid_medians(
    rat_medians:   Dict[str, float],
    rat_to_human:  Dict[str, str],
    human_medians: Dict[str, float],
) -> Tuple[Dict[str, float], Dict[str, str]]:
    """Construct hybrid median dictionary and per-gene source provenance.

    Priority:
      1. Human median (via rat→human ortholog mapping) — T1/T2/T3 genes
      2. Rat median — T4 genes (no qualifying human ortholog)
      3. Fallback 1.0 — should not occur given Stage 4 filtering

    Returns:
        hybrid  — {ENSRNOG_base: median_float}
        sources — {ENSRNOG_base: 'human'|'rat'|'fallback'}
    """
    hybrid:  Dict[str, float] = {}
    sources: Dict[str, str]   = {}

    n_human    = 0
    n_rat      = 0
    n_fallback = 0

    for eid, rat_med in rat_medians.items():
        human_id  = rat_to_human.get(eid)
        human_med = human_medians.get(human_id) if human_id else None

        if human_med and human_med > 0.0:
            hybrid[eid]  = human_med
            sources[eid] = 'human'
            n_human += 1
        elif rat_med > 0.0:
            hybrid[eid]  = rat_med
            sources[eid] = 'rat'
            n_rat += 1
        else:
            hybrid[eid]  = 1.0
            sources[eid] = 'fallback'
            n_fallback += 1

    total = len(hybrid)
    logger.info(f"Hybrid median dictionary: {total:,} genes")
    logger.info(f"  Human medians (T1-T3): {n_human:,} ({100*n_human/total:.1f}%)")
    logger.info(f"  Rat medians   (T4):    {n_rat:,} ({100*n_rat/total:.1f}%)")
    logger.info(f"  Fallback 1.0:          {n_fallback:,}")

    return hybrid, sources


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3: WRITE OUTPUTS
# ═════════════════════════════════════════════════════════════════════════════

def write_outputs(
    output_dir:    Path,
    hybrid:        Dict[str, float],
    sources:       Dict[str, str],
    config:        dict,
    t_start:       float,
    dry_run:       bool,
) -> None:
    """Write hybrid_gene_medians.pickle and hybrid_median_provenance.json."""
    import numpy as np

    pickle_path     = output_dir / 'hybrid_gene_medians.pickle'
    provenance_path = output_dir / 'hybrid_median_provenance.json'

    vals    = list(hybrid.values())
    src_cnt = {}
    for s in sources.values():
        src_cnt[s] = src_cnt.get(s, 0) + 1

    provenance = {
        'generated_at':   datetime.utcnow().isoformat() + 'Z',
        'elapsed_s':      round(time.time() - t_start, 1),
        'dry_run':        dry_run,
        'description': (
            'Hybrid gene median dictionary for Stage 5 tokenization. '
            'Human medians used for ortholog-mapped rat genes (T1-T3); '
            'rat medians used for Tier 4 rat-specific genes. '
            'Empirically validated against GeneCompass mouse reference corpus.'
        ),
        'inputs': {
            'rat_gene_medians':               str(output_dir / 'rat_gene_medians.pickle'),
            'rat_to_human_mapping':           str(output_dir.parent / 'ortholog_mappings' / 'rat_to_human_mapping.pickle'),
            'human_gene_median_after_filter': str(resolve_path(config, config['paths']['genecompass_medians'])),
        },
        'coverage': {
            'total_genes':      len(hybrid),
            'human_median':     src_cnt.get('human', 0),
            'rat_median':       src_cnt.get('rat', 0),
            'fallback':         src_cnt.get('fallback', 0),
            'pct_human':        round(100 * src_cnt.get('human', 0) / len(hybrid), 2),
            'pct_rat':          round(100 * src_cnt.get('rat', 0) / len(hybrid), 2),
        },
        'value_stats': {
            'min':    round(float(min(vals)), 6),
            'max':    round(float(max(vals)), 6),
            'mean':   round(float(sum(vals) / len(vals)), 6),
            'median': round(float(sorted(vals)[len(vals) // 2]), 6),
        },
        'normalization_note': (
            'Raw counts are divided directly by these medians (no normalize_total). '
            'Empirical comparison against randsel_5w_mouse: '
            'median offset -0.075, std offset -0.096 vs reference 0.63.'
        ),
        'per_gene_source': sources,
    }

    if dry_run:
        logger.info("DRY RUN — skipping file writes")
        logger.info(f"  Would write: {pickle_path.name}")
        logger.info(f"  Would write: {provenance_path.name}")
        return

    with open(pickle_path, 'wb') as f:
        pickle.dump(hybrid, f, protocol=4)
    logger.info(f"  {pickle_path.name}  ({pickle_path.stat().st_size / 1e6:.2f} MB, "
                f"{len(hybrid):,} genes)")

    with open(provenance_path, 'w') as f:
        json.dump(provenance, f, indent=2)
    logger.info(f"  {provenance_path.name}")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4: MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            'Build hybrid gene median dictionary for Stage 5 tokenization.'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Outputs (written to paths.median_dir):
  hybrid_gene_medians.pickle      — {ENSRNOG_base: median_float}
  hybrid_median_provenance.json   — per-gene source + coverage stats

Run before Stage 5 scatter:
  python pipeline/05_tokenization/build_hybrid_medians.py
  python run_stage5.py
        """,
    )
    parser.add_argument('--dry-run', action='store_true',
                        help='Validate inputs; skip writing outputs')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='DEBUG-level logging')
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    t_start = time.time()
    logger.info("=" * 70)
    logger.info("BUILD HYBRID GENE MEDIANS")
    logger.info("  Human medians (T1-T3) + Rat medians (T4) → hybrid_gene_medians.pickle")
    logger.info("=" * 70)

    try:
        config = load_config()
    except FileNotFoundError:
        logger.error("pipeline_config.yaml not found. Set PIPELINE_ROOT or run from project root.")
        sys.exit(1)

    paths       = config['paths']
    median_dir  = resolve_path(config, paths['median_dir'])
    orth_dir    = resolve_path(config, paths['ortholog_dir'])
    gc_medians  = resolve_path(config, paths['genecompass_medians'])

    # ── Load inputs ──────────────────────────────────────────────────────────
    logger.info("Loading inputs ...")
    try:
        rat_medians   = load_rat_medians(median_dir)
        rat_to_human  = load_rat_to_human(orth_dir)
        human_medians = load_human_medians(gc_medians)
    except FileNotFoundError as exc:
        logger.error(str(exc))
        sys.exit(1)

    # ── Build hybrid dictionary ───────────────────────────────────────────────
    logger.info("Building hybrid median dictionary ...")
    hybrid, sources = build_hybrid_medians(rat_medians, rat_to_human, human_medians)

    # ── Write outputs ─────────────────────────────────────────────────────────
    logger.info("Writing outputs ...")
    write_outputs(
        output_dir = median_dir,
        hybrid     = hybrid,
        sources    = sources,
        config     = config,
        t_start    = t_start,
        dry_run    = args.dry_run,
    )

    elapsed = time.time() - t_start
    logger.info("=" * 70)
    logger.info(f"DONE — {len(hybrid):,} genes in {elapsed:.1f}s")
    logger.info(f"  Output: {median_dir / 'hybrid_gene_medians.pickle'}")
    logger.info("  Next:   python run_stage5.py")
    logger.info("=" * 70)


if __name__ == '__main__':
    main()