#!/usr/bin/env python
# coding: utf-8
"""
run_stage7.py — Orchestrator for Stage 7 fine-tuning.

Two-step pipeline:
  Step 1: rat_model_init.py — Extend GeneCompass_Base checkpoint for rat vocab
  Step 2: rat_pretrain.py  — Fine-tune on 9.48M rat cells (SLURM submission)

Usage:
  python run_stage7.py              # Run both steps
  python run_stage7.py --from 2     # Skip init (checkpoint already extended)
  python run_stage7.py --dry-run    # Validate prerequisites only
  python run_stage7.py -v           # Verbose logging

Author: Tim Reese Lab
Date: March 2026
"""

import argparse
import logging
import os
import pickle
import subprocess
import sys
from pathlib import Path

import yaml

_PROJECT_ROOT = Path(os.environ.get('PIPELINE_ROOT', '.')).resolve()
sys.path.insert(0, str(_PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

def load_config(config_path: str = None):
    """Load finetuning config — standalone or pipeline_config.yaml."""
    if config_path is None:
        # Default: standalone GeneCompass config
        config_path = _PROJECT_ROOT / 'finetune' / 'genecompass' / 'configs' / 'rat_4gpu.yaml'
    with open(config_path) as f:
        raw = yaml.safe_load(f)

    # Standalone config: flatten nested sections
    if 'model' in raw or 'training' in raw:
        flat = {}
        for section in ('model', 'data', 'training', 'initialization',
                        'logging', 'output', 'distributed'):
            flat.update(raw.get(section, {}))
        return flat

    # Legacy: pipeline_config.yaml with finetuning: section
    return raw.get('finetuning', {})


def resolve(path_str: str) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p
    return _PROJECT_ROOT / p


# ─────────────────────────────────────────────────────────────────────────────
# Prerequisite validation
# ─────────────────────────────────────────────────────────────────────────────

def validate_prerequisites(config: dict) -> bool:
    """Check that all Stage 6 outputs and corpus are in place."""
    errors = []
    ft = config  # Already flattened by load_config()

    # Tokenized corpus
    corpus_dir = resolve(ft.get('dataset_directory',
                                'data/training/tokenized_corpus/dataset'))
    if not corpus_dir.exists():
        errors.append(f"Tokenized corpus not found: {corpus_dir}")
    else:
        # Check for sorted_length.pickle
        sorted_len = corpus_dir / 'sorted_length.pickle'
        if not sorted_len.exists():
            errors.append(f"sorted_length.pickle not found in {corpus_dir}")

    # Stage 6 embeddings
    stage6_dir = resolve(ft.get('stage6_dir', 'data/training/prior_knowledge'))
    for emb_file in ('promoter_embeddings.pkl', 'coexp_embeddings.pkl',
                      'family_embeddings.pkl', 'grn_embeddings.pkl'):
        if not (stage6_dir / emb_file).exists():
            errors.append(f"Stage 6 embedding not found: {stage6_dir / emb_file}")

    # Rat token dictionary
    token_dict = resolve(ft.get('token_dict_path',
                                'data/training/ortholog_mappings/rat_tokens.pickle'))
    if not token_dict.exists():
        errors.append(f"Rat token dict not found: {token_dict}")
    else:
        with open(token_dict, 'rb') as f:
            td = pickle.load(f)
        expected = ft.get('vocab_size', 53032)
        if len(td) != expected:
            errors.append(f"Token dict has {len(td)} entries, expected {expected}")

    # GeneCompass checkpoint
    ckpt = resolve(ft.get('base_checkpoint',
                          'vendor/GeneCompass/pretrained_models/GeneCompass_Base'))
    model_bin = ckpt / 'pytorch_model.bin'
    if not model_bin.exists():
        errors.append(f"GeneCompass checkpoint not found: {model_bin}")

    # Homologous gene mapping
    homolog = resolve(ft.get('homologous_gene_path',
                             'vendor/GeneCompass/prior_knowledge/homologous_hm_token.pickle'))
    if not homolog.exists():
        errors.append(f"Homologous gene mapping not found: {homolog}")

    if errors:
        for e in errors:
            logger.error(f"  ✗ {e}")
        return False

    logger.info("All prerequisites verified ✓")
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Steps
# ─────────────────────────────────────────────────────────────────────────────

def run_model_init(config: dict, dry_run: bool = False) -> bool:
    """Step 1: Extend checkpoint for rat vocabulary."""
    ft = config  # Already flattened by load_config()

    checkpoint = resolve(ft.get('base_checkpoint',
                                'vendor/GeneCompass/pretrained_models/GeneCompass_Base'))
    output_dir = resolve(ft.get('init_checkpoint',
                                'data/models/rat_genecompass_init'))
    token_dict = resolve(ft.get('token_dict_path',
                                'data/training/ortholog_mappings/rat_tokens.pickle'))
    stage6_dir = resolve(ft.get('stage6_dir',
                                'data/training/prior_knowledge'))
    homolog = resolve(ft.get('homologous_gene_path',
                             'vendor/GeneCompass/prior_knowledge/homologous_hm_token.pickle'))

    t4_init = ft.get('t4_word_emb_init', 'random')
    t4_std = ft.get('t4_word_emb_std', 0.02)
    species_std = ft.get('species_noise_std', 0.02)
    coexp_rescale = ft.get('coexp_rescale_factor', 3.9)

    # Check if already done
    if (output_dir / 'pytorch_model.bin').exists():
        logger.info(f"Extended checkpoint already exists: {output_dir}")
        logger.info("Skipping Step 1 (use --force to regenerate)")
        return True

    if dry_run:
        logger.info(f"[DRY RUN] Would extend checkpoint → {output_dir}")
        return True

    cmd = [
        sys.executable,
        str(_PROJECT_ROOT / 'finetune' / 'genecompass' / 'rat_model_init.py'),
        '--checkpoint', str(checkpoint),
        '--rat-token-dict', str(token_dict),
        '--stage6-dir', str(stage6_dir),
        '--homologous-gene-path', str(homolog),
        '--output-dir', str(output_dir),
        '--t4-init', t4_init,
        '--t4-std', str(t4_std),
        '--species-noise-std', str(species_std),
        '--coexp-rescale', str(coexp_rescale),
    ]

    logger.info("Step 1: Extending checkpoint for rat vocabulary...")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        logger.error("Step 1 FAILED")
        return False

    logger.info("Step 1 complete ✓")
    return True


def submit_training(config: dict, dry_run: bool = False) -> bool:
    """Step 2: Submit SLURM job for fine-tuning."""
    slurm_script = _PROJECT_ROOT / 'slurm' / 'stage7.slurm'

    if not slurm_script.exists():
        logger.error(f"SLURM script not found: {slurm_script}")
        return False

    if dry_run:
        logger.info(f"[DRY RUN] Would submit: sbatch {slurm_script}")
        return True

    logger.info("Step 2: Submitting fine-tuning SLURM job...")
    result = subprocess.run(
        ['sbatch', str(slurm_script)],
        capture_output=True, text=True,
    )

    if result.returncode != 0:
        logger.error(f"sbatch failed: {result.stderr}")
        return False

    logger.info(f"SLURM job submitted: {result.stdout.strip()}")
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Stage 7 orchestrator")
    parser.add_argument('--config', default=None, type=str,
                        help='Path to config YAML (default: finetune/genecompass/configs/rat_4gpu.yaml)')
    parser.add_argument('--from', dest='start_step', type=int, default=1,
                        help='Start from step N (1=init, 2=train)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Validate only, do not execute')
    parser.add_argument('--force', action='store_true',
                        help='Force re-execution of completed steps')
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    config = load_config(args.config)

    # Validate prerequisites
    if not validate_prerequisites(config):
        logger.error("Prerequisites not met — aborting")
        sys.exit(1)

    # Step 1: Extend checkpoint
    if args.start_step <= 1:
        if not run_model_init(config, dry_run=args.dry_run):
            sys.exit(1)

    # Step 2: Submit training
    if args.start_step <= 2:
        if not submit_training(config, dry_run=args.dry_run):
            sys.exit(1)

    logger.info("Stage 7 orchestration complete")


if __name__ == '__main__':
    main()