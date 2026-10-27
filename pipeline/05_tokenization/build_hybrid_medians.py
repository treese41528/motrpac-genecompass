#!/usr/bin/env python3
"""
build_hybrid_medians.py — Stage 5, Pre-Step: Build Hybrid Gene Median Dictionary

Pipeline position:
    Stage 4: Gene Medians  → rat_gene_medians.pickle
    Stage 3: Ortholog Mapping → rat_to_human_mapping.pickle
    Stage 5 Pre-Step: build_hybrid_medians.py   ← THIS SCRIPT
    Stage 5, Step 1: tokenize_corpus.py         (reads hybrid_gene_medians.pickle)

Purpose:
    Construct a per-gene median dictionary that uses GeneCompass's human median
    (human_gene_median_after_filter.pickle) for every rat gene whose rat->human
    ortholog has one, and the Stage 4 rat median for the remaining genes (T2b,
    T3b, T4 and a few others: every gene without a GeneCompass human median),
    rescaled onto the human-median scale.

    Why this design:
      - GeneCompass's corpora are raw counts divided by a fixed per-gene median,
        then log2(1+x), with no normalize_total. Verified on its human corpus
        (raw counts recovered for 99.8% of values, Feb 2026) and per gene on both
        its human and mouse corpora (Oct 2026). Mouse cells used mouse-specific
        medians on the same scale as the human ones.
      - Rat medians alone compress the values (corpus value median 0.50 vs
        GeneCompass's ~1.01), so genes with a human median keep it. This also
        keeps rat cells and their human re-expression (Stage 12) on the same
        per-gene scaling.
      - Stage 4 rat medians are computed after normalize_total(1e4) and sit ~2.24x
        above GeneCompass's medians on the genes that have both. Left unscaled,
        those genes enter at about half the value of their neighbours (0.50 vs
        0.94 in the same cells), so they are divided by F = median(rat / human)
        over the shared genes (--fallback-rescale auto; decided 2026-10-06).
        Check on 20,370 rat cells: corpus value median 0.886 unscaled -> 0.946
        rescaled, vs GeneCompass 1.007-1.015.

Coverage (March 2026 build):
    Human medians: 14,812 / 21,379 genes (69.3%)
    Rat medians:    6,567 / 21,379 genes (30.7%)
    Fallback (1.0):     0 genes

Outputs (written to paths.median_dir, or --output-dir):
    hybrid_gene_medians.pickle  — {ENSRNOG_base: median_float}
    hybrid_median_provenance.json — per-gene source, coverage stats, fallback scale
    An existing hybrid_gene_medians.pickle is never replaced without --overwrite:
    the current model's corpus and Stage 9 pseudo-cells were built with it.

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

def measure_fallback_scale(
    rat_medians:   Dict[str, float],
    rat_to_human:  Dict[str, str],
    human_medians: Dict[str, float],
) -> Tuple[float, int]:
    """F = median(rat median / human median) over genes that have both; (F, n genes)."""
    import numpy as np
    ratios = [rat_med / human_medians[rat_to_human[eid]]
              for eid, rat_med in rat_medians.items()
              if rat_med > 0.0 and human_medians.get(rat_to_human.get(eid), 0.0) > 0.0]
    if not ratios:
        raise ValueError("No genes with both a rat and a human median: cannot measure the fallback scale")
    return float(np.median(ratios)), len(ratios)


def build_hybrid_medians(
    rat_medians:   Dict[str, float],
    rat_to_human:  Dict[str, str],
    human_medians: Dict[str, float],
    fallback_scale: float = 1.0,
) -> Tuple[Dict[str, float], Dict[str, str]]:
    """Construct hybrid median dictionary and per-gene source provenance.

    Priority:
      1. Human median (via rat→human ortholog mapping) — genes with a GeneCompass human median
      2. Rat median / fallback_scale — every other gene (T2b, T3b, T4 and a few others)
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
            hybrid[eid]  = rat_med / fallback_scale
            sources[eid] = 'rat'
            n_rat += 1
        else:
            hybrid[eid]  = 1.0
            sources[eid] = 'fallback'
            n_fallback += 1

    total = len(hybrid)
    logger.info(f"Hybrid median dictionary: {total:,} genes")
    logger.info(f"  Human medians:                {n_human:,} ({100*n_human/total:.1f}%)")
    logger.info(f"  Rat medians / {fallback_scale:.3f}:        {n_rat:,} ({100*n_rat/total:.1f}%) "
                f"(no GeneCompass human median)")
    logger.info(f"  Fallback 1.0:                 {n_fallback:,}")

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
    input_paths:   Dict[str, str],
    fallback:      Dict[str, object],
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
            'GeneCompass human median for every rat gene whose rat->human ortholog has one; '
            'for every other gene (T2b, T3b, T4 and a few others) the Stage 4 rat median '
            'divided by fallback_scale, which puts it on the human-median scale.'
        ),
        'inputs': input_paths,
        'fallback_rescale': fallback,
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
            'Raw counts are divided directly by these medians (no normalize_total), then '
            'log2(1+x), as in GeneCompass\'s own human and mouse corpora.'
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
    parser.add_argument('--fallback-rescale', choices=['auto', 'none'], default='auto',
                        help="auto (default): divide the rat medians of genes without a GeneCompass "
                             "human median by F = median(rat / human) over shared genes. "
                             "none: the March 2026 behaviour (unscaled rat medians).")
    parser.add_argument('--output-dir', default=None,
                        help='Write here instead of paths.median_dir')
    parser.add_argument('--overwrite', action='store_true',
                        help='Allow replacing an existing hybrid_gene_medians.pickle (off by default)')
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
    logger.info("  Human medians + rescaled rat medians (genes without one) → hybrid_gene_medians.pickle")
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
    output_dir  = Path(args.output_dir).resolve() if args.output_dir else median_dir

    if (output_dir / 'hybrid_gene_medians.pickle').exists() and not args.overwrite and not args.dry_run:
        logger.error(f"{output_dir / 'hybrid_gene_medians.pickle'} already exists. The current model's corpus "
                     f"and Stage 9 pseudo-cells use it: write elsewhere (--output-dir, or a new "
                     f"paths.median_dir), or pass --overwrite if replacing it is really intended.")
        sys.exit(1)

    # ── Load inputs ──────────────────────────────────────────────────────────
    logger.info("Loading inputs ...")
    try:
        rat_medians   = load_rat_medians(median_dir)
        rat_to_human  = load_rat_to_human(orth_dir)
        human_medians = load_human_medians(gc_medians)
    except FileNotFoundError as exc:
        logger.error(str(exc))
        sys.exit(1)

    # ── Fallback scale ────────────────────────────────────────────────────────
    if args.fallback_rescale == 'auto':
        scale, n_shared = measure_fallback_scale(rat_medians, rat_to_human, human_medians)
        logger.info(f"Fallback scale F = {scale:.4f} (median rat/human over {n_shared:,} shared genes)")
    else:
        scale, n_shared = 1.0, 0
        logger.info("Fallback scale: none (rat medians used unscaled, March 2026 behaviour)")

    # ── Build hybrid dictionary ───────────────────────────────────────────────
    logger.info("Building hybrid median dictionary ...")
    hybrid, sources = build_hybrid_medians(rat_medians, rat_to_human, human_medians,
                                           fallback_scale=scale)

    # ── Write outputs ─────────────────────────────────────────────────────────
    logger.info("Writing outputs ...")
    output_dir.mkdir(parents=True, exist_ok=True)
    write_outputs(
        output_dir  = output_dir,
        hybrid      = hybrid,
        sources     = sources,
        config      = config,
        t_start     = t_start,
        dry_run     = args.dry_run,
        input_paths = {
            'rat_gene_medians':               str(median_dir / 'rat_gene_medians.pickle'),
            'rat_to_human_mapping':           str(orth_dir / 'rat_to_human_mapping.pickle'),
            'human_gene_median_after_filter': str(gc_medians),
        },
        fallback    = {'mode': args.fallback_rescale, 'scale': round(scale, 6), 'n_shared_genes': n_shared},
    )

    elapsed = time.time() - t_start
    logger.info("=" * 70)
    logger.info(f"DONE — {len(hybrid):,} genes in {elapsed:.1f}s")
    logger.info(f"  Output: {output_dir / 'hybrid_gene_medians.pickle'}")
    logger.info("  Next:   python run_stage5.py")
    logger.info("=" * 70)


if __name__ == '__main__':
    main()