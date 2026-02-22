#!/usr/bin/env python3
"""
run_stage2.py — Stage 2 Orchestrator: Gene Universe (3-step pipeline)

Runs three steps in sequence:
  Step 1: build_gene_universe.py         (scan raw → resolve → gene_universe.tsv)
  Step 2: preprocess_training_matrices.py (cell QC → h5ad + expression_stats.tsv)
  Step 3: prune_gene_universe.py          (expression pruning → pruned_gene_universe.tsv)

Architecture:
    ┌─────────────────────────────────────────────┐
    │ Step 1: Gene Universe Construction          │
    │ Scan raw matrices (var_names only)          │
    │ Resolve gene IDs: BioMart → RGD             │
    │ Prune: min_studies + biotype                │
    │ OUT: gene_universe.tsv, gene_resolution.tsv │
    └──────────────┬──────────────────────────────┘
                   │
    ┌──────────────▼──────────────────────────────┐
    │ Step 2: Cell QC & Preprocessing             │
    │ Load full matrices, map to gene_universe    │
    │ GeneCompass-exact QC:                       │
    │   normal_filter → gene_number_filter →      │
    │   min_genes → normalize → top-2048 ranking  │
    │ OUT: QC'd h5ad files, expression_stats.tsv  │
    └──────────────┬──────────────────────────────┘
                   │
    ┌──────────────▼──────────────────────────────┐
    │ Step 3: Expression Pruning                  │
    │ Remove genes never in any cell's top-2048   │
    │ OUT: pruned_gene_universe.tsv               │
    │      stage2_manifest.json                   │
    └─────────────────────────────────────────────┘

Key properties:
    - No chicken-and-egg: Step 1 builds gene list from raw var_names.
      Step 2 consumes it. No fallback chains, no pickle dependencies.
    - Single resolution point: Gene ID resolution in Step 1 only.
    - Deterministic: same inputs → same outputs.
    - GeneCompass-exact: Step 2 follows source code order precisely.

Usage:
  python run_stage2.py              # Run all 3 steps
  python run_stage2.py --from 2     # Skip gene universe (already built)
  python run_stage2.py --from 3     # Skip to pruning (preprocessing done)
  python run_stage2.py --dry-run    # Validate only
  python run_stage2.py -v           # Verbose logging

Author: Tim Reese Lab / Claude
Date: February 2026
"""

import argparse
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lib'))
from gene_utils import load_config, resolve_path

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

STAGE_DIR = Path(__file__).resolve().parent / '02_gene_universe'

STEPS = [
    {
        'num': 1,
        'name': 'Build Gene Universe',
        'script': 'build_gene_universe.py',
        'description': 'Scan raw matrices → resolve gene IDs → BioMart gate → prune → gene_universe.tsv',
        'key_output': 'gene_universe.tsv',
        'output_location': 'gene_universe_dir',
    },
    {
        'num': 2,
        'name': 'Preprocess Training Matrices',
        'script': 'preprocess_training_matrices.py',
        'description': 'Load full matrices → GeneCompass-exact cell QC → normalize → top-2048 → expression_stats.tsv',
        'key_output': 'expression_stats.tsv',
        'output_location': 'gene_universe_dir',
    },
    {
        'num': 3,
        'name': 'Prune Gene Universe',
        'script': 'prune_gene_universe.py',
        'description': 'Remove genes never in any cell top-2048 → pruned_gene_universe.tsv',
        'key_output': 'pruned_gene_universe.tsv',
        'output_location': 'gene_universe_dir',
    },
]


def validate_config(config: dict) -> bool:
    """Validate all config sections required by Stage 2."""
    errors = []

    for section in ('gene_universe', 'biomart', 'paths'):
        if section not in config:
            errors.append(f"Missing config section: '{section}'")

    if errors:
        for e in errors:
            logger.error(e)
        return False

    gu = config['gene_universe']
    bm = config['biomart']
    paths = config['paths']

    # Step 1 config
    for key in ('biomart_gate', 'min_biomart_match', 'min_studies',
                'keep_biotypes', 'accession_patterns'):
        if key not in gu:
            errors.append(f"Missing gene_universe.{key}")

    # Step 2 config
    prep = gu.get('preprocessing', {})
    for key in ('min_genes_per_cell', 'min_cells_per_sample'):
        if key not in prep:
            errors.append(f"Missing gene_universe.preprocessing.{key}")

    # BioMart reference
    for key in ('ensembl_release', 'assembly', 'rat_gene_info'):
        if key not in bm:
            errors.append(f"Missing biomart.{key}")

    # Paths
    for key in ('qc_h5ad_dir', 'gene_universe_dir'):
        if key not in paths:
            errors.append(f"Missing paths.{key}")

    # Verify BioMart gene info file exists
    if 'rat_gene_info' in bm:
        gene_info = resolve_path(config, bm['rat_gene_info'])
        if not gene_info.exists():
            errors.append(f"BioMart gene info not found: {gene_info}")

    if errors:
        for e in errors:
            logger.error(e)
        return False

    return True


def validate_inputs(config: dict, from_step: int) -> bool:
    """Validate input files/directories exist for the requested steps."""
    paths = config['paths']
    gu_dir = resolve_path(config, paths['gene_universe_dir'])
    errors = []

    if from_step >= 2:
        # Step 2 needs gene_universe.tsv and gene_resolution.tsv from Step 1
        for fname in ['gene_universe.tsv', 'gene_resolution.tsv']:
            fpath = gu_dir / fname
            if not fpath.exists():
                errors.append(f"{fname} not found: {fpath} (run Step 1 first)")

    if from_step >= 3:
        # Step 3 needs expression_stats.tsv from Step 2
        stats_path = gu_dir / 'expression_stats.tsv'
        if not stats_path.exists():
            errors.append(f"expression_stats.tsv not found: {stats_path} (run Step 2 first)")

    if errors:
        for e in errors:
            logger.error(e)
        return False

    return True


def run_step(step: dict, config: dict, dry_run: bool = False,
             verbose: bool = False) -> bool:
    """Run a single pipeline step as a subprocess."""
    script_path = STAGE_DIR / step['script']

    if not script_path.exists():
        logger.error(f"Script not found: {script_path}")
        return False

    cmd = [sys.executable, str(script_path)]
    if dry_run:
        cmd.append('--dry-run')
    if verbose:
        cmd.append('-v')

    logger.info("=" * 70)
    logger.info(f"STEP {step['num']}: {step['name']}")
    logger.info(f"  Script: {script_path}")
    logger.info(f"  {step['description']}")
    logger.info("=" * 70)

    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(STAGE_DIR.parent.parent))
    elapsed = time.time() - t0

    if result.returncode != 0:
        logger.error(f"Step {step['num']} FAILED (exit code {result.returncode}) "
                      f"after {elapsed:.1f}s")
        return False

    # Report key output
    if not dry_run:
        gu_dir = resolve_path(config, config['paths'].get(step['output_location'], ''))
        key_out = gu_dir / step['key_output']
        if key_out.exists():
            if step['key_output'].endswith('.tsv'):
                with open(key_out) as f:
                    n_rows = sum(1 for _ in f) - 1
                logger.info(f"  Output: {key_out.name} ({n_rows:,} rows)")
            else:
                logger.info(f"  Output: {key_out.name}")

    logger.info(f"Step {step['num']} completed in {elapsed:.1f}s")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Stage 2 Orchestrator: Gene Universe (3-step pipeline)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Steps:
  1  build_gene_universe.py           Scan → resolve → gene_universe.tsv
  2  preprocess_training_matrices.py  Cell QC → h5ad + expression_stats.tsv
  3  prune_gene_universe.py           Expression prune → pruned_gene_universe.tsv

Examples:
  python run_stage2.py              # Run all 3 steps
  python run_stage2.py --from 2     # Skip gene universe building
  python run_stage2.py --from 3     # Skip to pruning
  python run_stage2.py --dry-run    # Validate only
        """,
    )
    parser.add_argument('--from', dest='from_step', type=int, default=1,
                        choices=[s['num'] for s in STEPS],
                        help='Start from this step (default: 1)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Validate config and inputs, pass --dry-run to steps')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Verbose (DEBUG) logging')
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info("=" * 70)
    logger.info("STAGE 2: GENE UNIVERSE (3-step pipeline)")
    logger.info("=" * 70)
    logger.info("")
    logger.info("  Step 1: build_gene_universe.py         → gene_universe.tsv")
    logger.info("  Step 2: preprocess_training_matrices.py → expression_stats.tsv")
    logger.info("  Step 3: prune_gene_universe.py          → pruned_gene_universe.tsv")
    logger.info("")

    # Load config
    try:
        config = load_config()
    except FileNotFoundError:
        logger.error("pipeline_config.yaml not found. "
                      "Set PIPELINE_ROOT or run from project root.")
        sys.exit(1)

    # Validate
    if not validate_config(config):
        logger.error("Config validation failed")
        sys.exit(1)
    logger.info("Config validation passed")

    if not validate_inputs(config, args.from_step):
        logger.error("Input validation failed")
        sys.exit(1)
    logger.info("Input validation passed")

    # Run steps
    t_total = time.time()
    steps_to_run = [s for s in STEPS if s['num'] >= args.from_step]

    if args.from_step > 1:
        logger.info(f"Starting from Step {args.from_step} (skipping earlier steps)")

    for step in steps_to_run:
        ok = run_step(step, config, dry_run=args.dry_run, verbose=args.verbose)
        if not ok:
            logger.error(f"Stage 2 aborted at Step {step['num']}")
            sys.exit(1)

    elapsed_total = time.time() - t_total

    logger.info("=" * 70)
    logger.info(f"STAGE 2 COMPLETE — {len(steps_to_run)} steps in {elapsed_total:.1f}s")
    logger.info("=" * 70)

    # Report final output
    if not args.dry_run:
        gu_dir = resolve_path(config, config['paths']['gene_universe_dir'])
        pruned = gu_dir / 'pruned_gene_universe.tsv'
        if pruned.exists():
            with open(pruned) as f:
                n_genes = sum(1 for _ in f) - 1
            logger.info(f"Final output: {pruned} ({n_genes:,} genes)")
            logger.info("Downstream stages read: pruned_gene_universe.tsv")
        elif args.from_step <= 3:
            logger.warning(f"Expected output not found: {pruned}")


if __name__ == '__main__':
    main()