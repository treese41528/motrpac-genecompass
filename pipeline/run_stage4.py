#!/usr/bin/env python3
"""
run_stage4.py — Stage 4 Orchestrator: Gene Medians (Scatter / Gather)

Runs two steps in sequence:
  Step 1: compute_gene_medians.py   (scatter — all files, single task)
  Step 2: gather_gene_medians.py    (gather — merge → rat_gene_medians.pickle)

Architecture:
    ┌─────────────────────────────────────────────────┐
    │ Stage 3 outputs (inputs to Stage 4)             │
    │ rat_token_mapping.tsv  — token-assigned genes   │
    │ QC'd h5ad files        — raw counts             │
    └──────────────┬──────────────────────────────────┘
                   │
    ┌──────────────▼──────────────────────────────────┐
    │ Step 1: Scatter — compute_gene_medians.py       │
    │ Load each h5ad fully into memory                │
    │ Map var_names → gene universe (strip versions)  │
    │ normalize_total(10000) — ephemeral, in memory   │
    │ Accumulate per-gene:                            │
    │   non-zero values (norm + raw), running stats   │
    │   per-cell lib_size, n_genes_detected           │
    │ OUT: scatter/scatter_NNNN_stats.npz             │
    │      scatter/scatter_NNNN_nonzero.npz           │
    │      scatter/scatter_NNNN_cells.tsv.gz          │
    │      scatter/scatter_NNNN_manifest.json         │
    └──────────────┬──────────────────────────────────┘
                   │
    ┌──────────────▼──────────────────────────────────┐
    │ Step 2: Gather — gather_gene_medians.py         │
    │ Phase 1: Aggregate running stats across tasks   │
    │ Phase 2: Exact medians (gene-batch processing)  │
    │ Phase 3: Write outputs                          │
    │ OUT: rat_gene_medians.pickle   ← PRIMARY        │
    │      gene_median_stats.tsv                      │
    │      gene_median_stats.npz                      │
    │      biotype_summary.json                       │
    │      cell_stats_summary.json                    │
    │      stage4_manifest.json                       │
    └──────────────┬──────────────────────────────────┘
                   │
    ┌──────────────▼──────────────────────────────────┐
    │ Downstream consumers                            │
    │ Stage 5 → rat_gene_medians.pickle (tokenizer)   │
    │ Aim 2  → gene_median_stats.tsv (QC / GRN)      │
    │ Aim 3  → zero_fraction, biotype (translation)   │
    └─────────────────────────────────────────────────┘

On HPC clusters (Gilbreth):
    The SLURM scripts call compute_gene_medians.py and gather_gene_medians.py
    directly as a scatter array + gather job with afterok dependency.
    This orchestrator runs both steps serially in a single process,
    which is appropriate for local testing or non-SLURM environments.

Usage:
  python run_stage4.py              # Run both steps
  python run_stage4.py --from 2     # Skip scatter (already done)
  python run_stage4.py --dry-run    # Validate only
  python run_stage4.py -v           # Verbose logging

Author: Tim Reese Lab / Claude
Date: March 2026
"""

import argparse
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(os.environ.get('PIPELINE_ROOT', '.')).resolve()
sys.path.insert(0, str(_PROJECT_ROOT / 'lib'))
from gene_utils import load_config, resolve_path

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

STAGE_DIR = _PROJECT_ROOT / 'pipeline' / '04_gene_medians'

STEPS = [
    {
        'num': 1,
        'name': 'Scatter — compute per-gene stats',
        'script': 'compute_gene_medians.py',
        'description': 'Load h5ad corpus → normalize → accumulate per-gene stats → scatter/*.npz',
        'key_output': 'scatter',
        'output_location': 'median_dir',
    },
    {
        'num': 2,
        'name': 'Gather — merge → rat_gene_medians.pickle',
        'script': 'gather_gene_medians.py',
        'description': 'Aggregate scatter outputs → exact medians → rat_gene_medians.pickle',
        'key_output': 'rat_gene_medians.pickle',
        'output_location': 'median_dir',
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

def validate_config(config: dict) -> bool:
    errors = []
    for section in ('medians', 'paths', 'biomart'):
        if section not in config:
            errors.append(f"Missing config section: '{section}'")
    if errors:
        for e in errors:
            logger.error(e)
        return False

    paths = config['paths']
    for key in ('qc_h5ad_dir', 'ortholog_dir', 'median_dir'):
        if key not in paths:
            errors.append(f"Missing paths.{key}")
    if errors:
        for e in errors:
            logger.error(e)
        return False

    return True


def validate_inputs(config: dict, from_step: int) -> bool:
    errors = []
    paths = config['paths']

    if from_step >= 1:
        orth_dir = resolve_path(config, paths['ortholog_dir'])
        mapping = orth_dir / 'rat_token_mapping.tsv'
        if not mapping.exists():
            errors.append(
                f"rat_token_mapping.tsv not found: {mapping}\n"
                f"  → Run Stage 3 first (python run_stage3.py)"
            )
        else:
            with open(mapping) as f:
                n_genes = sum(1 for _ in f) - 1
            logger.info(f"  rat_token_mapping.tsv: {n_genes:,} rows [OK]")

        qc_dir = resolve_path(config, paths['qc_h5ad_dir'])
        h5ad_files = list(qc_dir.glob('**/*.h5ad'))
        if not h5ad_files:
            errors.append(
                f"No .h5ad files found in {qc_dir}\n"
                f"  → Run Stage 2 first (python run_stage2.py)"
            )
        else:
            logger.info(f"  QC'd h5ad files: {len(h5ad_files):,} [OK]")

    if from_step >= 2:
        median_dir = resolve_path(config, paths['median_dir'])
        scatter_dir = median_dir / 'scatter'
        manifests = list(scatter_dir.glob('scatter_*_manifest.json')) if scatter_dir.exists() else []
        if not manifests:
            errors.append(
                f"No scatter manifests found in {scatter_dir}\n"
                f"  → Run Step 1 first (python run_stage4.py)"
            )
        else:
            logger.info(f"  Scatter manifests: {len(manifests):,} [OK]")

    if errors:
        for e in errors:
            logger.error(e)
        return False

    return True


# ─────────────────────────────────────────────────────────────────────────────
# STEP RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run_step(step: dict, config: dict, dry_run: bool = False,
             verbose: bool = False) -> bool:
    script_path = STAGE_DIR / step['script']

    if not script_path.exists():
        logger.error(f"Script not found: {script_path}")
        return False

    # Step 1 (scatter) runs as task 0 of 1 — processes all files in one pass
    cmd = [sys.executable, str(script_path)]
    if step['num'] == 1:
        cmd += ['--task-id', '0', '--n-tasks', '1']
    if dry_run:
        cmd.append('--dry-run')
    if verbose:
        cmd.append('-v')

    logger.info("=" * 70)
    logger.info(f"STEP {step['num']}: {step['name']}")
    logger.info(f"  Script: {script_path.name}")
    logger.info(f"  {step['description']}")
    logger.info("=" * 70)

    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(_PROJECT_ROOT))
    elapsed = time.time() - t0

    if result.returncode != 0:
        logger.error(f"Step {step['num']} FAILED (exit code {result.returncode}) "
                     f"after {elapsed:.1f}s")
        return False

    if not dry_run:
        median_dir = resolve_path(config, config['paths']['median_dir'])
        key_out = median_dir / step['key_output']
        if key_out.exists():
            if key_out.suffix == '.pickle':
                logger.info(f"  Output: {key_out.name} "
                            f"({key_out.stat().st_size / 1e6:.1f} MB)")
            else:
                logger.info(f"  Output: {key_out.name}/")

    logger.info(f"Step {step['num']} completed in {elapsed:.1f}s")
    return True


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Stage 4 Orchestrator: Gene Medians (Scatter / Gather)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Steps:
  1  compute_gene_medians.py   Load h5ad corpus → accumulate per-gene stats
  2  gather_gene_medians.py    Merge scatter → exact medians → rat_gene_medians.pickle

On HPC clusters the SLURM scripts call these directly:
  sbatch slurm/pipeline/stage4_scatter.slurm  (array job, step 1 in parallel)
  sbatch slurm/pipeline/stage4_gather.slurm   (step 2, afterok dependency)

Examples:
  python run_stage4.py              # Run both steps (serial)
  python run_stage4.py --from 2     # Scatter done, run gather only
  python run_stage4.py --dry-run    # Validate inputs without running
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
    logger.info("STAGE 4: GENE MEDIANS")
    logger.info("=" * 70)
    logger.info("")
    logger.info("  Step 1: compute_gene_medians.py  → scatter/*.npz")
    logger.info("  Step 2: gather_gene_medians.py   → rat_gene_medians.pickle")
    logger.info("")

    try:
        config = load_config()
    except FileNotFoundError:
        logger.error("pipeline_config.yaml not found. "
                     "Set PIPELINE_ROOT or run from project root.")
        sys.exit(1)

    if not validate_config(config):
        logger.error("Config validation failed")
        sys.exit(1)
    logger.info("Config validation passed")

    if not validate_inputs(config, args.from_step):
        logger.error("Input validation failed")
        sys.exit(1)
    logger.info("Input validation passed")

    t_total = time.time()
    steps_to_run = [s for s in STEPS if s['num'] >= args.from_step]

    if args.from_step > 1:
        logger.info(f"Starting from Step {args.from_step} (skipping earlier steps)")

    for step in steps_to_run:
        ok = run_step(step, config, dry_run=args.dry_run, verbose=args.verbose)
        if not ok:
            logger.error(f"Stage 4 aborted at Step {step['num']}")
            sys.exit(1)

    elapsed_total = time.time() - t_total

    logger.info("=" * 70)
    logger.info(f"STAGE 4 COMPLETE — {len(steps_to_run)} steps in {elapsed_total:.1f}s")
    logger.info("=" * 70)

    if not args.dry_run:
        median_dir = resolve_path(config, config['paths']['median_dir'])
        pickle_path = median_dir / 'rat_gene_medians.pickle'
        if pickle_path.exists():
            logger.info(f"Final output: {pickle_path} "
                        f"({pickle_path.stat().st_size / 1e6:.1f} MB)")
            logger.info("")
            logger.info("Next steps:")
            logger.info("  Stage 5 → Reference Assembly (reads rat_gene_medians.pickle)")


if __name__ == '__main__':
    main()