#!/usr/bin/env python3
"""
run_stage3.py — Stage 3 Orchestrator: Ortholog Mapping

Runs the ortholog mapping pipeline that assigns each rat gene in the
pruned gene universe to a GeneCompass token ID via tiered ortholog
resolution.

Architecture:
    ┌─────────────────────────────────────────────────┐
    │ Stage 2 outputs (inputs to Stage 3)             │
    │ pruned_gene_universe.tsv  — 22,213 rat genes    │
    │ QC'd h5ad files           — raw counts          │
    └──────────────┬──────────────────────────────────┘
                   │
    ┌──────────────▼──────────────────────────────────┐
    │ Stage 3: Ortholog Mapping                       │
    │                                                 │
    │ Tier 1: Tri-species one2one                     │
    │   rat→human + rat→mouse, h↔m linked in GC       │
    │ Tier 2a: Human-rat one2one                      │
    │ Tier 2b: Mouse-rat one2one                      │
    │ Tier 3a: Human-rat one2many/many2many           │
    │ Tier 3b: Mouse-rat one2many/many2many           │
    │ Tier 4: New rat token (learned in fine-tuning)  │
    │                                                 │
    │ NO identity threshold — GC used none.           │
    │ Ensembl tree-based orthology IS the gate.       │
    │                                                 │
    │ OUT: rat_tokens.pickle                          │
    │      rat_to_human_mapping.pickle                │
    │      rat_to_mouse_mapping.pickle                │
    │      rat_token_mapping.tsv (with confidence)    │
    │      collision_report.tsv                       │
    │      tier_diagnostics.json                      │
    │      mapping_statistics.json                    │
    │      stage3_manifest.json                       │
    └──────────────┬──────────────────────────────────┘
                   │
    ┌──────────────▼──────────────────────────────────┐
    │ Downstream consumers                            │
    │ Stage 4 → rat_to_human_mapping (median source)  │
    │ Stage 5 → rat_tokens (tokenizer)                │
    │ Aim 2  → confidence column (GRN edge flags)     │
    │ Aim 3  → confidence column (translation guard)  │
    └─────────────────────────────────────────────────┘

Usage:
  python run_stage3.py              # Run ortholog mapping
  python run_stage3.py --dry-run    # Validate inputs only
  python run_stage3.py -v           # Verbose logging

Author: Tim Reese Lab / Claude
Date: February 2026
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

# Resolve project root from PIPELINE_ROOT env var or config default,
# then locate lib/ for shared utilities.
_PROJECT_ROOT = Path(os.environ.get('PIPELINE_ROOT', '.')).resolve()
sys.path.insert(0, str(_PROJECT_ROOT / 'lib'))
from gene_utils import load_config, resolve_path

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def validate_config(config: dict) -> bool:
    """Validate all config sections required by Stage 3."""
    errors = []

    for section in ('orthologs', 'biomart', 'paths'):
        if section not in config:
            errors.append(f"Missing config section: '{section}'")

    if errors:
        for e in errors:
            logger.error(e)
        return False

    orth = config['orthologs']
    bm = config['biomart']
    paths = config['paths']

    # Ortholog config
    for key in ('one2one_types', 'multi_types', 'new_token_biotypes'):
        if key not in orth:
            errors.append(f"Missing orthologs.{key}")

    # BioMart ortholog files
    for key in ('rat_human_orthologs', 'rat_mouse_orthologs'):
        if key not in bm:
            errors.append(f"Missing biomart.{key}")

    # Required paths
    for key in ('gene_universe_dir', 'ortholog_dir',
                'genecompass_tokens', 'genecompass_homologs'):
        if key not in paths:
            errors.append(f"Missing paths.{key}")

    if errors:
        for e in errors:
            logger.error(e)
        return False

    return True


def validate_inputs(config: dict) -> bool:
    """Validate all input files exist before running."""
    errors = []
    warnings = []
    paths = config['paths']
    bm = config['biomart']

    # Stage 2 output (required)
    gu_dir = resolve_path(config, paths['gene_universe_dir'])
    pruned = gu_dir / 'pruned_gene_universe.tsv'
    if not pruned.exists():
        errors.append(f"pruned_gene_universe.tsv not found: {pruned}\n"
                      f"  → Run Stage 2 first (python run_stage2.py)")
    else:
        with open(pruned) as f:
            n_genes = sum(1 for _ in f) - 1
        logger.info(f"  pruned_gene_universe.tsv: {n_genes:,} genes [OK]")

    # BioMart ortholog files (required)
    for key, label in [('rat_human_orthologs', 'rat→human'),
                       ('rat_mouse_orthologs', 'rat→mouse')]:
        fpath = resolve_path(config, bm[key])
        if not fpath.exists():
            errors.append(f"BioMart {label} orthologs not found: {fpath}")
        else:
            with open(fpath) as f:
                n_records = sum(1 for _ in f) - 1
            logger.info(f"  {fpath.name}: {n_records:,} records [OK]")

    # GeneCompass files (required)
    gc_vocab = resolve_path(config, paths['genecompass_tokens'])
    gc_homologs = resolve_path(config, paths['genecompass_homologs'])

    if not gc_vocab.exists():
        errors.append(f"GeneCompass vocab not found: {gc_vocab}")
    else:
        logger.info(f"  {gc_vocab.name} [OK]")

    if not gc_homologs.exists():
        errors.append(f"GeneCompass homologs not found: {gc_homologs}")
    else:
        logger.info(f"  {gc_homologs.name} [OK]")

    # Output directory
    out_dir = resolve_path(config, paths['ortholog_dir'])
    if out_dir.exists():
        existing = list(out_dir.glob('*.pickle')) + list(out_dir.glob('*.tsv'))
        if existing:
            warnings.append(f"Output directory already has {len(existing)} files: {out_dir}\n"
                            f"  → Existing files will be overwritten")

    for w in warnings:
        logger.warning(w)
    if errors:
        for e in errors:
            logger.error(e)
        return False

    return True


def run_stage3(config: dict, dry_run: bool = False, verbose: bool = False) -> bool:
    """Run the ortholog mapping script."""
    script_path = _PROJECT_ROOT / 'pipeline' / '03_ortholog_mapping' / 'build_ortholog_mapping.py'

    if not script_path.exists():
        logger.error(f"Script not found: {script_path}")
        return False

    cmd = [sys.executable, str(script_path)]
    if dry_run:
        cmd.append('--dry-run')
    if verbose:
        cmd.append('-v')

    logger.info("=" * 70)
    logger.info("RUNNING: build_ortholog_mapping.py")
    logger.info("  Tier architecture: T1 → T2a → T2b → T3a → T3b → T4")
    logger.info("  Identity threshold: None (GC used none)")
    logger.info("  Disambiguation: highest % identity")
    logger.info("=" * 70)

    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(_PROJECT_ROOT))
    elapsed = time.time() - t0

    if result.returncode != 0:
        logger.error(f"Stage 3 FAILED (exit code {result.returncode}) after {elapsed:.1f}s")
        return False

    logger.info(f"Script completed in {elapsed:.1f}s")
    return True


def print_summary(config: dict):
    """Print post-run summary from Stage 3 outputs."""
    out_dir = resolve_path(config, config['paths']['ortholog_dir'])

    # Read mapping statistics
    stats_path = out_dir / 'mapping_statistics.json'
    if stats_path.exists():
        with open(stats_path) as f:
            stats = json.load(f)

        logger.info("")
        logger.info("=" * 70)
        logger.info("STAGE 3 RESULTS")
        logger.info("=" * 70)
        logger.info(f"  Input genes:       {stats.get('total_input_genes', '?'):,}")
        logger.info(f"  Mapped to tokens:  {stats.get('total_mapped', '?'):,} "
                     f"({stats.get('mapping_rate', '?')}%)")
        logger.info(f"  Pre-trained reuse: {stats.get('pre_trained_tokens_used', '?'):,}")
        logger.info(f"  New tokens:        {stats.get('new_tokens_created', '?'):,}")

        tiers = stats.get('tier_distribution', {})
        pcts = stats.get('tier_percentages', {})
        if tiers:
            logger.info("")
            logger.info("  Tier distribution:")
            for tier in ['T1_tri_species', 'T2a_human_one2one', 'T2b_mouse_one2one',
                         'T3a_human_multi', 'T3b_mouse_multi', 'T4_new_token', 'excluded']:
                cnt = tiers.get(tier, 0)
                pct = pcts.get(tier, 0)
                logger.info(f"    {tier:25s} {cnt:>6,}  ({pct:5.1f}%)")

        conf = stats.get('confidence_distribution', {})
        if conf:
            logger.info("")
            logger.info("  Confidence:")
            for level in ['high', 'medium', 'low']:
                logger.info(f"    {level:10s} {conf.get(level, 0):>6,}")

        collisions = stats.get('token_collisions', {})
        if collisions:
            logger.info("")
            logger.info(f"  Token collisions:  {collisions.get('total_colliding_tokens', 0)}")
            logger.info(f"  Max collision:     {collisions.get('max_collision', 0)}")

    # List output files
    logger.info("")
    logger.info("  Output files:")
    for fpath in sorted(out_dir.iterdir()):
        if fpath.is_file():
            size_kb = fpath.stat().st_size / 1024
            logger.info(f"    {fpath.name:40s} {size_kb:>8.1f} KB")

    logger.info("")
    logger.info("  Downstream handoff:")
    logger.info("    Stage 4 → rat_to_human_mapping.pickle  (determines human vs rat medians)")
    logger.info("    Stage 5 → rat_tokens.pickle            (tokenizer for rat cells)")
    logger.info("    Aim 2  → confidence column             (GRN edge quality flags)")
    logger.info("    Aim 3  → confidence column             (cross-species translation guardrails)")


def main():
    parser = argparse.ArgumentParser(
        description="Stage 3 Orchestrator: Ortholog Mapping",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Tier Architecture:
  T1   Tri-species one2one — rat→human + rat→mouse, human↔mouse linked in GC
  T2a  Human-rat one2one   — rat→human in GC vocab
  T2b  Mouse-rat one2one   — rat→mouse in GC vocab
  T3a  Human-rat multi     — one2many/many2many to human in GC vocab
  T3b  Mouse-rat multi     — one2many/many2many to mouse in GC vocab
  T4   New rat token       — no ortholog, biotype = protein_coding/lncRNA/miRNA

Key: NO identity threshold. GeneCompass used none (min observed: 1.2%).
     Ensembl tree-based orthology inference IS the quality gate.

Examples:
  python run_stage3.py              # Run ortholog mapping
  python run_stage3.py --dry-run    # Validate inputs only
  python run_stage3.py -v           # Verbose logging
        """,
    )
    parser.add_argument('--dry-run', action='store_true',
                        help='Validate config and inputs, pass --dry-run to script')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Verbose (DEBUG) logging')
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info("=" * 70)
    logger.info("STAGE 3: ORTHOLOG MAPPING")
    logger.info("=" * 70)
    logger.info("")
    logger.info("  Map rat genes → GeneCompass token IDs via tiered ortholog resolution")
    logger.info("  Tier priority: T1 → T2a → T2b → T3a → T3b → T4")
    logger.info("  Identity threshold: None (Ensembl orthology IS the gate)")
    logger.info("")

    # Load config
    try:
        config = load_config()
    except FileNotFoundError:
        logger.error("pipeline_config.yaml not found. "
                      "Set PIPELINE_ROOT or run from project root.")
        sys.exit(1)

    # Validate config
    logger.info("Validating config...")
    if not validate_config(config):
        logger.error("Config validation failed")
        sys.exit(1)
    logger.info("Config validation passed")

    # Validate inputs
    logger.info("")
    logger.info("Validating inputs...")
    if not validate_inputs(config):
        logger.error("Input validation failed")
        sys.exit(1)
    logger.info("Input validation passed")

    # Run
    logger.info("")
    t_total = time.time()
    ok = run_stage3(config, dry_run=args.dry_run, verbose=args.verbose)

    if not ok:
        logger.error("Stage 3 aborted")
        sys.exit(1)

    elapsed_total = time.time() - t_total

    # Summary
    if not args.dry_run:
        print_summary(config)

    logger.info("")
    logger.info("=" * 70)
    logger.info(f"STAGE 3 COMPLETE — {elapsed_total:.1f}s")
    logger.info("=" * 70)


if __name__ == '__main__':
    main()