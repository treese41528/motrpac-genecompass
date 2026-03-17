#!/usr/bin/env python3
"""
run_stage5.py — Stage 5 Orchestrator: Reference Assembly & Tokenization

Runs two steps in sequence:
  Step 1: tokenize_corpus.py   (scatter — all files, single task locally)
  Step 2: assemble_corpus.py   (gather  — Arrow shards → HuggingFace dataset)

Architecture:
    ┌─────────────────────────────────────────────────┐
    │ Stage 4 outputs (inputs to Stage 5)             │
    │ rat_gene_medians.pickle  — per-gene medians     │
    │ Stage 3 outputs                                 │
    │ rat_tokens.pickle        — gene → token_id      │
    │ Stage 2 outputs                                 │
    │ qc_h5ad/                 — 864 QC'd files       │
    └──────────────┬──────────────────────────────────┘
                   │
    ┌──────────────▼──────────────────────────────────┐
    │ Step 1: Scatter — tokenize_corpus.py            │
    │ For each cell in each QC'd h5ad:                │
    │   1. Load raw counts (fully into memory)        │
    │   2. normalize_total(10,000)                    │
    │   3. Divide by per-gene median                  │
    │   4. log1p                                      │
    │   5. Rank non-zero genes descending             │
    │   6. Take top 2,048                             │
    │   7. Map to token IDs                           │
    │ OUT: shards/task_NNNN_shard_MMMM.arrow          │
    │      manifests/task_NNNN_manifest.json          │
    └──────────────┬──────────────────────────────────┘
                   │
    ┌──────────────▼──────────────────────────────────┐
    │ Step 2: Gather — assemble_corpus.py             │
    │ Phase 1: Validate manifests + shard files       │
    │ Phase 2: Assemble → HuggingFace dataset         │
    │ Phase 3: Write corpus_stats.tsv + manifest      │
    │ OUT: dataset/              ← PRIMARY            │
    │      corpus_stats.tsv                           │
    │      stage5_manifest.json                       │
    └──────────────┬──────────────────────────────────┘
                   │
    ┌──────────────▼──────────────────────────────────┐
    │ Downstream consumers                            │
    │ Fine-tuning → dataset/ (GeneCompass pretrain)   │
    │ Aim 2       → study_id column (per-study QC)    │
    │ Aim 3       → full token corpus (translation)   │
    └─────────────────────────────────────────────────┘

On HPC clusters (Gilbreth):
    The SLURM scripts call tokenize_corpus.py and assemble_corpus.py
    directly as a scatter array + gather job with afterok dependency.
    This orchestrator runs both steps serially in a single process,
    which is appropriate for local testing or single-node environments.

Usage:
  python run_stage5.py              # Run both steps
  python run_stage5.py --from 2     # Skip scatter (already done)
  python run_stage5.py --dry-run    # Validate only
  python run_stage5.py -v           # Verbose logging

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

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
)
logger = logging.getLogger(__name__)

STAGE_DIR = _PROJECT_ROOT / 'pipeline' / '05_tokenization'

STEPS = [
    {
        'num':           1,
        'name':          'Scatter — tokenize h5ad corpus → Arrow shards',
        'script':        'tokenize_corpus.py',
        'description':   (
            'Load each h5ad → raw/hybrid_median → log2(1+x) → rank → '
            'top-2048 tokens → shards/*.arrow'
        ),
        'key_output':    'shards',
        'output_section': 'tokenized_corpus_dir',
    },
    {
        'num':           2,
        'name':          'Gather — Arrow shards → HuggingFace dataset',
        'script':        'assemble_corpus.py',
        'description':   (
            'Validate manifests → concatenate shards → '
            'datasets.save_to_disk() → dataset/'
        ),
        'key_output':    'dataset',
        'output_section': 'tokenized_corpus_dir',
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

def validate_config(config: dict) -> bool:
    errors = []
    for section in ('reference_assembly', 'paths', 'biomart', 'medians'):
        if section not in config:
            errors.append(f"Missing config section: '{section}'")

    if errors:
        for e in errors:
            logger.error(e)
        return False

    paths = config['paths']
    required_path_keys = (
        'qc_h5ad_dir',
        'ortholog_dir',
        'median_dir',
        'tokenized_corpus_dir',
    )
    for key in required_path_keys:
        if key not in paths:
            errors.append(f"Missing paths.{key} in config")

    ref = config.get('reference_assembly', {})
    if 'top_n_genes' not in ref:
        errors.append("Missing reference_assembly.top_n_genes in config")

    if errors:
        for e in errors:
            logger.error(e)
        return False

    return True


def validate_inputs(config: dict, from_step: int) -> bool:
    errors = []
    paths  = config['paths']

    if from_step <= 1:
        # Inputs for Step 1 (scatter)
        orth_dir = resolve_path(config, paths['ortholog_dir'])
        tokens   = orth_dir / 'rat_tokens.pickle'
        mapping  = orth_dir / 'rat_token_mapping.tsv'

        if not tokens.exists():
            errors.append(
                f"rat_tokens.pickle not found: {tokens}\n"
                f"  → Run Stage 3 first (python run_stage3.py)"
            )
        else:
            logger.info(f"  rat_tokens.pickle: {tokens.stat().st_size / 1e6:.1f} MB [OK]")

        if not mapping.exists():
            errors.append(
                f"rat_token_mapping.tsv not found: {mapping}\n"
                f"  → Run Stage 3 first (python run_stage3.py)"
            )
        else:
            with open(mapping) as f:
                n_genes = sum(1 for _ in f) - 1  # minus header
            logger.info(f"  rat_token_mapping.tsv: {n_genes:,} rows [OK]")

        median_dir = resolve_path(config, paths['median_dir'])
        medians    = median_dir / 'rat_gene_medians.pickle'
        if not medians.exists():
            errors.append(
                f"rat_gene_medians.pickle not found: {medians}\n"
                f"  → Run Stage 4 first (python run_stage4.py)"
            )
        else:
            logger.info(f"  rat_gene_medians.pickle: {medians.stat().st_size / 1e6:.1f} MB [OK]")

        hybrid = median_dir / 'hybrid_gene_medians.pickle'
        if not hybrid.exists():
            errors.append(
                f"hybrid_gene_medians.pickle not found: {hybrid}\n"
                f"  → Run build_hybrid_medians.py first:\n"
                f"    python pipeline/05_tokenization/build_hybrid_medians.py"
            )
        else:
            logger.info(f"  hybrid_gene_medians.pickle: {hybrid.stat().st_size / 1e6:.1f} MB [OK]")

        qc_dir   = resolve_path(config, paths['qc_h5ad_dir'])
        h5ad_files = list(qc_dir.glob('**/*.h5ad'))
        if not h5ad_files:
            errors.append(
                f"No .h5ad files found in {qc_dir}\n"
                f"  → Run Stage 2 first (python run_stage2.py)"
            )
        else:
            logger.info(f"  QC'd h5ad files: {len(h5ad_files):,} [OK]")

    if from_step <= 2:
        # Inputs for Step 2 (gather) — shards from Step 1
        corpus_dir   = resolve_path(config, paths['tokenized_corpus_dir'])
        manifest_dir = corpus_dir / 'manifests'
        shard_dir    = corpus_dir / 'shards'

        if from_step == 2:
            # Step 1 must have run already
            manifests = list(manifest_dir.glob('task_*_manifest.json')) \
                if manifest_dir.exists() else []
            if not manifests:
                errors.append(
                    f"No task manifests found in {manifest_dir}\n"
                    f"  → Run Step 1 first (python run_stage5.py)"
                )
            else:
                shards = list(shard_dir.glob('*.arrow')) \
                    if shard_dir.exists() else []
                logger.info(
                    f"  Task manifests: {len(manifests):,}, "
                    f"Arrow shards: {len(shards):,} [OK]"
                )

    if errors:
        for e in errors:
            logger.error(e)
        return False

    return True


# ─────────────────────────────────────────────────────────────────────────────
# STEP RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run_step(
    step:     dict,
    config:   dict,
    dry_run:  bool = False,
    verbose:  bool = False,
) -> bool:
    script_path = STAGE_DIR / step['script']

    if not script_path.exists():
        logger.error(f"Script not found: {script_path}")
        return False

    cmd = [sys.executable, str(script_path)]

    # Step 1 (scatter): run as task 0 of 1 — processes all files serially
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

    t0     = time.time()
    result = subprocess.run(cmd, cwd=str(_PROJECT_ROOT))
    elapsed = time.time() - t0

    if result.returncode != 0:
        logger.error(
            f"Step {step['num']} FAILED (exit code {result.returncode}) "
            f"after {elapsed:.1f}s"
        )
        return False

    # Report key output
    if not dry_run:
        corpus_dir = resolve_path(
            config, config['paths']['tokenized_corpus_dir']
        )
        key_out = corpus_dir / step['key_output']
        if key_out.exists():
            if key_out.is_dir():
                n_children = sum(1 for _ in key_out.iterdir())
                logger.info(
                    f"  Output: {step['key_output']}/ "
                    f"({n_children} item(s))"
                )
            else:
                logger.info(
                    f"  Output: {step['key_output']} "
                    f"({key_out.stat().st_size / 1e6:.1f} MB)"
                )

    logger.info(f"Step {step['num']} completed in {elapsed:.1f}s")
    return True


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Stage 5 Orchestrator: Reference Assembly & Tokenization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Steps:
  1  tokenize_corpus.py   Load h5ad corpus → median-normalize → rank → token seqs
  2  assemble_corpus.py   Merge Arrow shards → HuggingFace dataset

On HPC clusters the SLURM scripts call these directly:
  sbatch slurm/pipeline/stage5_tokenize.txt  (array job, step 1 in parallel)
  sbatch slurm/pipeline/stage5_assemble.txt  (step 2, afterok dependency)

Examples:
  python run_stage5.py              # Run both steps (serial, local)
  python run_stage5.py --from 2     # Scatter done on HPC, run gather only
  python run_stage5.py --dry-run    # Validate inputs without running
        """,
    )
    parser.add_argument(
        '--from', dest='from_step', type=int, default=1,
        choices=[s['num'] for s in STEPS],
        help='Start from this step (default: 1)',
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Validate config and inputs; pass --dry-run to each step',
    )
    parser.add_argument(
        '-v', '--verbose', action='store_true',
        help='Verbose (DEBUG) logging',
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info("=" * 70)
    logger.info("STAGE 5: REFERENCE ASSEMBLY & TOKENIZATION")
    logger.info("=" * 70)
    logger.info("")
    logger.info("  Step 1: tokenize_corpus.py  → shards/*.arrow")
    logger.info("  Step 2: assemble_corpus.py  → dataset/")
    logger.info("")

    # ── Config ───────────────────────────────────────────────────────────────
    try:
        config = load_config()
    except FileNotFoundError:
        logger.error(
            "pipeline_config.yaml not found. "
            "Set PIPELINE_ROOT or run from project root."
        )
        sys.exit(1)

    if not validate_config(config):
        logger.error("Config validation failed")
        sys.exit(1)
    logger.info("Config validation passed")

    if not validate_inputs(config, args.from_step):
        logger.error("Input validation failed")
        sys.exit(1)
    logger.info("Input validation passed")

    # ── Summarise key Stage 5 parameters ────────────────────────────────────
    ref_cfg = config.get('reference_assembly', {})
    logger.info(
        f"  top_n_genes={ref_cfg.get('top_n_genes', 2048)}, "
        f"cells_per_shard={ref_cfg.get('cells_per_shard', 50000):,}, "
        f"exclude_studies={ref_cfg.get('exclude_studies', [])}"
    )

    # ── Run steps ────────────────────────────────────────────────────────────
    t_total      = time.time()
    steps_to_run = [s for s in STEPS if s['num'] >= args.from_step]

    if args.from_step > 1:
        logger.info(
            f"Starting from Step {args.from_step} (skipping earlier steps)"
        )

    for step in steps_to_run:
        ok = run_step(
            step, config, dry_run=args.dry_run, verbose=args.verbose
        )
        if not ok:
            logger.error(f"Stage 5 aborted at Step {step['num']}")
            sys.exit(1)

    elapsed_total = time.time() - t_total

    logger.info("=" * 70)
    logger.info(
        f"STAGE 5 COMPLETE — {len(steps_to_run)} step(s) in {elapsed_total:.1f}s"
    )
    logger.info("=" * 70)

    if not args.dry_run:
        corpus_dir  = resolve_path(
            config, config['paths']['tokenized_corpus_dir']
        )
        dataset_dir = corpus_dir / 'dataset'
        if dataset_dir.exists():
            logger.info(f"Final output: {dataset_dir}")
            logger.info("")
            logger.info("Next steps:")
            logger.info("  Fine-tuning  → python vendor/GeneCompass/scripts/pretrain.py \\")
            logger.info(f"    --data_path {dataset_dir}")
            logger.info("  Aim 2        → deconvolution pipeline (UniCell / scDEAL / Scissor)")
            logger.info("  Aim 3        → cross-species translation (GTEx / UK Biobank)")


if __name__ == '__main__':
    main()