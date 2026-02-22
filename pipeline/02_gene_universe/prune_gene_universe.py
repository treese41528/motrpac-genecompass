#!/usr/bin/env python3
"""
prune_gene_universe.py — Stage 2, Step 3: Expression-based pruning

Pipeline position:
    Stage 1: Data Harvesting & QC
    Stage 2: Gene Universe
      Step 1: build_gene_universe.py (scan → resolve → gene_universe.tsv)
      Step 2: preprocess_training_matrices.py (cell QC → expression_stats.tsv)
      Step 3: prune_gene_universe.py                             ← THIS SCRIPT
    Stage 3: Ortholog Mapping
    Stage 4: Gene Medians (SLURM)
    Stage 5: Reference Assembly & Corpus Export

Purpose:
    Simple filter. Read gene_universe.tsv (Step 1) + expression_stats.tsv
    (Step 2), remove genes that never appeared in any cell's top-2048
    across the entire corpus. Output the final pruned gene list that
    Stages 3-5 consume.

Design:
    This is intentionally a thin script. All the heavy computation happened
    in Steps 1 and 2. This step just joins two TSV files and filters.

Outputs:
    pruned_gene_universe.tsv  — Final gene list for downstream stages
    pruning_report.json       — Statistics on what was pruned and why
    stage2_manifest.json      — Complete Stage 2 manifest (all 3 steps)

Usage:
    python pipeline/02_gene_universe/prune_gene_universe.py
    python pipeline/02_gene_universe/prune_gene_universe.py --dry-run

Author: Tim Reese Lab / Claude
Date: February 2026
"""

import csv
import hashlib
import json
import logging
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'lib'))
from gene_utils import load_config, resolve_path

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


def _git_hash() -> str:
    try:
        import subprocess
        r = subprocess.run(['git', 'rev-parse', 'HEAD'],
                           capture_output=True, text=True, timeout=5)
        return r.stdout.strip() if r.returncode == 0 else 'unknown'
    except Exception:
        return 'unknown'


def prune_gene_universe(config: dict, dry_run: bool = False):
    """Main: join gene_universe.tsv + expression_stats.tsv, filter, output."""
    t_start = time.time()

    gu_dir = resolve_path(config, config['paths']['gene_universe_dir'])

    universe_path = gu_dir / 'gene_universe.tsv'
    stats_path = gu_dir / 'expression_stats.tsv'
    step1_manifest = gu_dir / 'manifest_step1.json'
    step2_report = gu_dir / 'preprocessing_report.json'

    # ── Validate inputs ──
    if not universe_path.exists():
        logger.error(f"gene_universe.tsv not found: {universe_path}")
        logger.error("Run Step 1 (build_gene_universe.py) first.")
        sys.exit(1)

    if not stats_path.exists():
        logger.error(f"expression_stats.tsv not found: {stats_path}")
        logger.error("Run Step 2 (preprocess_training_matrices.py) first.")
        sys.exit(1)

    if dry_run:
        logger.info("DRY RUN — inputs validated:")
        logger.info(f"  gene_universe.tsv: {universe_path}")
        logger.info(f"  expression_stats.tsv: {stats_path}")
        logger.info("DRY RUN — exiting")
        return

    # ── Load gene_universe.tsv ──
    universe = {}
    with open(universe_path, 'r') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            universe[row['ensembl_id']] = row

    logger.info(f"Loaded gene_universe.tsv: {len(universe):,} genes")

    # ── Load expression_stats.tsv ──
    expr_stats = {}
    with open(stats_path, 'r') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            expr_stats[row['ensembl_id']] = {
                'n_cells_top2048': int(row['n_cells_top2048']),
                'total_expression': float(row['total_expression']),
            }

    logger.info(f"Loaded expression_stats.tsv: {len(expr_stats):,} genes")

    # ── Merge and filter ──
    pruned = []
    removed = []
    biotype_kept = defaultdict(int)
    biotype_removed = defaultdict(int)

    for eid, info in sorted(universe.items()):
        stats = expr_stats.get(eid, {'n_cells_top2048': 0, 'total_expression': 0.0})
        n_top = stats['n_cells_top2048']
        biotype = info.get('biotype', 'unknown')

        if n_top > 0:
            row = dict(info)
            row['n_cells_top2048'] = n_top
            row['total_expression'] = stats['total_expression']
            pruned.append(row)
            biotype_kept[biotype] += 1
        else:
            removed.append({
                'ensembl_id': eid,
                'symbol': info.get('symbol', ''),
                'biotype': biotype,
                'n_studies': info.get('n_studies', 0),
                'reason': 'never_in_top2048',
            })
            biotype_removed[biotype] += 1

    logger.info(f"Pruning: {len(universe):,} → {len(pruned):,} "
                f"(removed {len(removed):,} genes never in any cell's top-2048)")

    # ── Check for genes in expression_stats but NOT in universe ──
    orphan_genes = set(expr_stats.keys()) - set(universe.keys())
    if orphan_genes:
        logger.warning(f"{len(orphan_genes)} genes in expression_stats.tsv "
                       f"but not in gene_universe.tsv (ignored)")

    # ── Output: pruned_gene_universe.tsv ──
    pruned_path = gu_dir / 'pruned_gene_universe.tsv'
    fieldnames = ['ensembl_id', 'symbol', 'biotype', 'n_studies',
                  'n_cells_top2048', 'total_expression']
    # Also keep study_list if present
    if 'study_list' in pruned[0] if pruned else {}:
        fieldnames.append('study_list')

    with open(pruned_path, 'w', newline='') as f:
        w = csv.DictWriter(f, delimiter='\t', fieldnames=fieldnames,
                           extrasaction='ignore')
        w.writeheader()
        w.writerows(pruned)

    logger.info(f"pruned_gene_universe.tsv: {len(pruned):,} genes")

    # ── Output: pruning_report.json ──
    elapsed = time.time() - t_start

    pruning_report = {
        'stage': 'stage2_step3',
        'script': 'prune_gene_universe.py',
        'timestamp': datetime.now().isoformat(),
        'git_hash': _git_hash(),
        'elapsed_seconds': round(elapsed, 1),
        'summary': {
            'universe_size': len(universe),
            'pruned_size': len(pruned),
            'removed_count': len(removed),
            'pruning_rate': round(len(removed) / max(1, len(universe)), 4),
            'orphan_genes': len(orphan_genes),
        },
        'biotype_kept': dict(biotype_kept),
        'biotype_removed': dict(biotype_removed),
        'inputs': {
            'gene_universe.tsv': {'n_genes': len(universe), 'md5': _md5(universe_path)},
            'expression_stats.tsv': {'n_genes': len(expr_stats), 'md5': _md5(stats_path)},
        },
        'outputs': {
            'pruned_gene_universe.tsv': {'n_genes': len(pruned), 'md5': _md5(pruned_path)},
        },
    }

    report_path = gu_dir / 'pruning_report.json'
    with open(report_path, 'w') as f:
        json.dump(pruning_report, f, indent=2)

    # ── Output: stage2_manifest.json (complete Stage 2 summary) ──
    stage2_manifest = {
        'stage': 'stage2',
        'description': 'Gene Universe Construction',
        'timestamp': datetime.now().isoformat(),
        'git_hash': _git_hash(),
        'steps': {
            'step1_gene_universe': {
                'script': 'build_gene_universe.py',
                'output': 'gene_universe.tsv',
                'n_genes': len(universe),
            },
            'step2_preprocessing': {
                'script': 'preprocess_training_matrices.py',
                'output': 'expression_stats.tsv',
                'n_genes_tracked': len(expr_stats),
            },
            'step3_pruning': {
                'script': 'prune_gene_universe.py',
                'output': 'pruned_gene_universe.tsv',
                'n_genes': len(pruned),
                'n_removed': len(removed),
            },
        },
        'final_output': {
            'pruned_gene_universe.tsv': {
                'path': str(pruned_path),
                'n_genes': len(pruned),
                'md5': _md5(pruned_path),
            },
        },
        'downstream': 'Stages 3-5 read ONLY pruned_gene_universe.tsv',
    }

    # Incorporate Step 1 manifest if available
    if step1_manifest.exists():
        with open(step1_manifest) as f:
            s1 = json.load(f)
        stage2_manifest['steps']['step1_gene_universe']['details'] = s1.get('pipeline', {})

    # Incorporate Step 2 report summary if available
    if step2_report.exists():
        with open(step2_report) as f:
            s2 = json.load(f)
        stage2_manifest['steps']['step2_preprocessing']['details'] = s2.get('summary', {})

    manifest_path = gu_dir / 'stage2_manifest.json'
    with open(manifest_path, 'w') as f:
        json.dump(stage2_manifest, f, indent=2)

    # ── Summary ──
    logger.info("=" * 60)
    logger.info("PRUNING SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  Gene universe (in):    {len(universe):,}")
    logger.info(f"  Pruned universe (out): {len(pruned):,}")
    logger.info(f"  Removed:               {len(removed):,}")
    logger.info(f"  Pruning rate:          {len(removed)/max(1,len(universe))*100:.1f}%")
    logger.info("")
    logger.info("  Biotype distribution (kept):")
    for bt, cnt in sorted(biotype_kept.items(), key=lambda x: -x[1]):
        logger.info(f"    {bt:20s} {cnt:>6,}")
    if biotype_removed:
        logger.info("  Biotype distribution (removed):")
        for bt, cnt in sorted(biotype_removed.items(), key=lambda x: -x[1]):
            logger.info(f"    {bt:20s} {cnt:>6,}")
    logger.info(f"  Elapsed:               {elapsed:.1f}s")
    logger.info(f"  Output:                {pruned_path}")
    logger.info("")
    logger.info("Stages 3-5 read: pruned_gene_universe.tsv")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Stage 2, Step 3: Prune gene universe by expression",
    )
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('--dry-run', action='store_true',
                        help='Validate inputs exist, then exit')
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        config = load_config()
    except FileNotFoundError:
        logger.error("pipeline_config.yaml not found. Set PIPELINE_ROOT or run from project root.")
        sys.exit(1)

    if 'paths' not in config:
        logger.error("Config missing 'paths' section")
        sys.exit(1)

    prune_gene_universe(config, dry_run=args.dry_run)


if __name__ == '__main__':
    main()