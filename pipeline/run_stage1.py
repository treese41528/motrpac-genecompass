#!/usr/bin/env python3
"""
run_stage1.py - Stage 1 Orchestrator: Data Harvesting & QC

Runs all Stage 1 pipeline steps in dependency order:

  1. fetch_biomart_reference_data.py   → reference gene annotations
  2. geo_harvester.py                  → download GEO datasets
  3. arrayexpress_harvester.py         → download ArrayExpress datasets
  4. extract_metadata.py               → master_catalog.json
  5. analyze_matrices.py               → matrix_analysis.json
  6. generate_statistics.py            → statistics.json
  7. llm_study_analyzer.py             → llm_study_analysis.json
  8. combine_data_sources.py           → unified_studies.json
  9. check_matrix_results.py           → QC report
 10. analyze_llm_output.py             → LLM QC report

Usage:
    python run_stage1.py                   # full run
    python run_stage1.py --from 4          # start from step 4
    python run_stage1.py --only 5          # run only step 5
    python run_stage1.py --dry-run         # show plan without executing
    python run_stage1.py --skip-llm        # skip LLM steps (7, 10)
"""

import os
import sys
import subprocess
import argparse
import logging
from pathlib import Path
from datetime import datetime

# --- Config integration ---
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lib'))
from gene_utils import load_config, resolve_path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# STEP DEFINITIONS
# ============================================================================

SCRIPT_DIR = Path(__file__).parent / '01_data_harvesting'


def _catalog_dir(cfg):
    return resolve_path(cfg, cfg.get('harvesting', {}).get('catalog_dir', 'data/catalog'))


STEPS = [
    {
        'num': 1,
        'name': 'Fetch BioMart reference data',
        'script': 'fetch_biomart_reference_data.py',
        'outputs': lambda cfg: [
            resolve_path(cfg, cfg.get('biomart', {}).get('rat_gene_info', 'data/references/biomart/rat_gene_info.tsv')),
        ],
        'is_llm': False,
    },
    {
        'num': 2,
        'name': 'Harvest GEO datasets',
        'script': 'geo_harvester.py',
        'outputs': lambda cfg: [
            resolve_path(cfg, cfg.get('harvesting', {}).get('geo_output_dir', 'data/raw/geo')),
        ],
        'is_llm': False,
    },
    {
        'num': 3,
        'name': 'Harvest ArrayExpress datasets',
        'script': 'arrayexpress_harvester.py',
        'outputs': lambda cfg: [
            resolve_path(cfg, cfg.get('harvesting', {}).get('arrayexpress_output_dir', 'data/raw/arrayexpress')),
        ],
        'is_llm': False,
    },
    {
        'num': 4,
        'name': 'Extract metadata',
        'script': 'extract_metadata.py',
        'outputs': lambda cfg: [
            _catalog_dir(cfg) / 'master_catalog.json',
        ],
        'is_llm': False,
    },
    {
        'num': 5,
        'name': 'Analyze matrices',
        'script': 'analyze_matrices.py',
        'outputs': lambda cfg: [
            _catalog_dir(cfg) / 'matrix_analysis.json',
        ],
        'is_llm': False,
    },
    {
        'num': 6,
        'name': 'Generate statistics',
        'script': 'generate_statistics.py',
        'outputs': lambda cfg: [
            _catalog_dir(cfg) / 'statistics.json',
        ],
        'is_llm': False,
    },
    {
        'num': 7,
        'name': 'LLM study analysis',
        'script': 'llm_study_analyzer.py',
        'outputs': lambda cfg: [
            resolve_path(cfg, cfg.get('harvesting', {}).get('llm_output_file', 'data/catalog/llm_study_analysis.json')),
        ],
        'is_llm': True,
    },
    {
        'num': 8,
        'name': 'Combine data sources',
        'script': 'combine_data_sources.py',
        'outputs': lambda cfg: [
            _catalog_dir(cfg) / 'unified_studies.json',
        ],
        'is_llm': False,
    },
    {
        'num': 9,
        'name': 'Check matrix results (QC)',
        'script': 'check_matrix_results.py',
        'outputs': lambda cfg: [],
        'is_llm': False,
    },
    {
        'num': 10,
        'name': 'Analyze LLM output (QC)',
        'script': 'analyze_llm_output.py',
        'args': lambda cfg: [
            str(resolve_path(cfg, cfg.get('harvesting', {}).get('llm_output_file', 'data/catalog/llm_study_analysis.json')))
        ],
        'outputs': lambda cfg: [],
        'is_llm': True,
    },
]


# ============================================================================
# EXECUTION
# ============================================================================

def check_outputs_exist(step: dict, config: dict) -> bool:
    """Check if all output files/dirs for a step already exist."""
    outputs = step['outputs'](config)
    if not outputs:
        return False
    return all(p.exists() for p in outputs)


def run_step(step: dict, config: dict, dry_run: bool = False) -> bool:
    """Run a single pipeline step. Returns True on success."""
    num = step['num']
    name = step['name']
    script = SCRIPT_DIR / step['script']

    if not script.exists():
        logger.error(f"Step {num}: Script not found: {script}")
        return False

    cmd = [sys.executable, str(script)]

    if 'args' in step:
        cmd.extend(step['args'](config))

    logger.info(f"{'─' * 60}")
    logger.info(f"Step {num}: {name}")
    logger.info(f"  Script: {script.name}")
    logger.info(f"  Command: {' '.join(cmd)}")

    if dry_run:
        outputs = step['outputs'](config)
        for p in outputs:
            exists = '✓ exists' if p.exists() else '✗ missing'
            logger.info(f"  Output: {p} [{exists}]")
        return True

    start = datetime.now()

    try:
        result = subprocess.run(
            cmd,
            cwd=str(SCRIPT_DIR),
            capture_output=False,
            text=True,
        )

        elapsed = (datetime.now() - start).total_seconds()

        if result.returncode != 0:
            logger.error(f"Step {num}: FAILED (exit code {result.returncode}) [{elapsed:.1f}s]")
            return False

        logger.info(f"Step {num}: DONE [{elapsed:.1f}s]")
        return True

    except Exception as e:
        elapsed = (datetime.now() - start).total_seconds()
        logger.error(f"Step {num}: EXCEPTION: {e} [{elapsed:.1f}s]")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Stage 1 Orchestrator: Data Harvesting & QC',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Steps:
  1  Fetch BioMart reference data     6  Generate statistics
  2  Harvest GEO datasets             7  LLM study analysis
  3  Harvest ArrayExpress datasets     8  Combine data sources
  4  Extract metadata                  9  Check matrix results (QC)
  5  Analyze matrices                 10  Analyze LLM output (QC)
"""
    )
    parser.add_argument('--from', dest='start_from', type=int, default=1,
                        help='Start from step N (default: 1)')
    parser.add_argument('--only', type=int, help='Run only step N')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show plan without executing')
    parser.add_argument('--skip-existing', action='store_true',
                        help='Skip steps whose outputs already exist')
    parser.add_argument('--skip-llm', action='store_true',
                        help='Skip LLM steps (7, 10)')
    parser.add_argument('--no-stop-on-fail', action='store_true',
                        help='Continue past failures (default: stop)')

    args = parser.parse_args()

    try:
        config = load_config()
    except FileNotFoundError:
        logger.error("pipeline_config.yaml not found. Run from project root or set PIPELINE_ROOT.")
        sys.exit(1)

    if args.only:
        steps_to_run = [s for s in STEPS if s['num'] == args.only]
        if not steps_to_run:
            logger.error(f"No such step: {args.only}")
            sys.exit(1)
    else:
        steps_to_run = [s for s in STEPS if s['num'] >= args.start_from]

    if args.skip_llm:
        steps_to_run = [s for s in steps_to_run if not s['is_llm']]

    logger.info("=" * 60)
    logger.info("STAGE 1: DATA HARVESTING & QC")
    logger.info(f"Steps: {[s['num'] for s in steps_to_run]}")
    logger.info(f"Mode: {'DRY RUN' if args.dry_run else 'EXECUTE'}")
    logger.info("=" * 60)

    pipeline_start = datetime.now()
    passed = 0
    failed = 0
    skipped = 0

    for step in steps_to_run:
        if args.skip_existing and check_outputs_exist(step, config):
            logger.info(f"Step {step['num']}: {step['name']} — skipped (outputs exist)")
            skipped += 1
            continue

        ok = run_step(step, config, dry_run=args.dry_run)

        if ok:
            passed += 1
        else:
            failed += 1
            if not args.no_stop_on_fail and not args.dry_run:
                logger.error(f"Stopping pipeline at step {step['num']}.")
                break

    elapsed = (datetime.now() - pipeline_start).total_seconds()
    logger.info("=" * 60)
    logger.info("STAGE 1 SUMMARY")
    logger.info(f"  Passed:  {passed}")
    logger.info(f"  Failed:  {failed}")
    logger.info(f"  Skipped: {skipped}")
    logger.info(f"  Time:    {elapsed:.1f}s ({elapsed/60:.1f}min)")
    logger.info("=" * 60)

    sys.exit(1 if failed > 0 else 0)


if __name__ == '__main__':
    main()