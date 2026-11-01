#!/usr/bin/env python
# coding: utf-8
"""
run_stage7.py — Orchestrator for Stage 7 fine-tuning.

Four steps (all parameters from the training YAML, default finetune/genecompass/configs/rat_finetune_v2.yaml):
  Step 1: rat_model_init.py              — Extend GeneCompass_Base for the rat vocabulary
                                           (--seed from initialization.seed; v2's init was built unseeded)
  Step 2: build_mixed_species_dataset.py — The rat/human/mouse training mix (mixing: section; v2 = 40/45/15, seed 42)
  Step 3: eval_mlm.py build-probe         — The fixed evaluation probe (probe: section; seed 20261008)
  Step 4: scripts/train_stage7.sh         — torchrun training, phase --phase (1, then 2; resumes from the
                                           latest checkpoint). Runs in the foreground; --sbatch SCRIPT submits a
                                           site batch wrapper instead (it should call scripts/train_stage7.sh).

Steps whose output already exists are skipped (init: pytorch_model.bin; mix: state.json; probe: the folder).
Step numbers changed on 2026-11-01 (training was step 2; it is now step 4).

Usage:
  python run_stage7.py --from 1 --to 3   # init, mix, probe (no training)
  python run_stage7.py --from 4 --phase 1   # train phase 1 (then: --from 4 --phase 2)
  python run_stage7.py --dry-run          # Validate prerequisites only
  python run_stage7.py -v                 # Verbose logging

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
        config_path = _PROJECT_ROOT / 'finetune' / 'genecompass' / 'configs' / 'rat_finetune_v2.yaml'
    with open(config_path) as f:
        raw = yaml.safe_load(f)

    # Standalone config: flatten nested sections
    if 'model' in raw or 'training' in raw:
        flat = {}
        for section in ('model', 'data', 'training', 'initialization',
                        'logging', 'output', 'distributed'):
            flat.update(raw.get(section, {}))
        flat['_config_path'] = str(config_path)
        # kept apart so their keys (seed, ...) never collide in the flat dict
        flat['init_seed'] = (raw.get('initialization') or {}).get('seed', 0)
        flat['_mixing'] = raw.get('mixing') or {}
        flat['_probe'] = raw.get('probe') or {}
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
        expected = ft.get('vocab_size', 55275)
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
    t3_priors = ft.get('t3_prior_source', 'ortholog')
    t4_zero_priors = ft.get('t4_zero_priors', ['promoter', 'coexp']) or []

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
        '--t3-priors', t3_priors,
        '--t4-zero-priors', ','.join(t4_zero_priors),
    ]
    # initialization.seed (default 0). v2's init was built without one (not regenerable byte for byte;
    # deposited); null in the YAML reproduces that unseeded behaviour.
    if ft.get('init_seed') is not None:
        cmd += ['--seed', str(int(ft['init_seed']))]

    logger.info("Step 1: Extending checkpoint for rat vocabulary...")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        logger.error("Step 1 FAILED")
        return False

    logger.info("Step 1 complete ✓")
    return True


def build_mix(config: dict, dry_run: bool = False) -> bool:
    """Step 2: interleave the rat corpus with GeneCompass's human and mouse sets."""
    ft, mix = config, config.get('_mixing', {})
    out = resolve(ft.get('mixed_dataset', 'data/training/mixed_species_dataset_v2'))
    if (out / 'state.json').exists():
        logger.info(f"Training mix already exists: {out} (skipping Step 2)")
        return True
    cmd = [
        sys.executable, str(_PROJECT_ROOT / 'finetune' / 'genecompass' / 'build_mixed_species_dataset.py'),
        '--rat-dataset', str(resolve(ft.get('dataset_directory', 'data/training/tokenized_corpus_v2/dataset'))),
        '--human-dataset', str(resolve(ft.get('human_dataset', 'vendor/GeneCompass/data/randsel_500w_human'))),
        '--mouse-dataset', str(resolve(ft.get('mouse_dataset', 'vendor/GeneCompass/data/randsel_500w_mouse'))),
        '--output', str(out),
        '--rat-prob', str(mix.get('rat_prob', 0.40)),
        '--human-prob', str(mix.get('human_prob', 0.45)),
        '--mouse-prob', str(mix.get('mouse_prob', 0.15)),
        '--seed', str(mix.get('seed', 42)),
    ]
    if mix.get('exclude_studies'):
        cmd += ['--exclude-studies', ','.join(mix['exclude_studies'])]
    return _run('Step 2: training mix', cmd, dry_run)


def build_probe(config: dict, dry_run: bool = False) -> bool:
    """Step 3: the fixed evaluation probe drawn from the mix (built once)."""
    ft, probe = config, config.get('_probe', {})
    out = resolve(probe.get('output', 'data/training/eval_probe_v2'))
    if out.exists():
        logger.info(f"Evaluation probe already exists: {out} (skipping Step 3)")
        return True
    cmd = [
        sys.executable, str(_PROJECT_ROOT / 'finetune' / 'genecompass' / 'eval_mlm.py'), 'build-probe',
        '--mix', str(resolve(ft.get('mixed_dataset', 'data/training/mixed_species_dataset_v2'))),
        '--out', str(out),
        '--seed', str(probe.get('seed', 20261008)),
        '--n-per-species', str(probe.get('n_per_species', 2000)),
    ]
    return _run('Step 3: evaluation probe', cmd, dry_run)


def train(config: dict, phase: int, dry_run: bool = False, sbatch_script: str = None) -> bool:
    """Step 4: train one phase with scripts/train_stage7.sh (or submit a site batch wrapper)."""
    env = dict(os.environ, PHASE=str(phase), PIPELINE_ROOT=str(_PROJECT_ROOT))
    if config.get('_config_path'):
        env['CONFIG'] = os.path.relpath(config['_config_path'], _PROJECT_ROOT)
    if config.get('output_directory'):
        env['OUT'] = config['output_directory']
    if sbatch_script:
        cmd = ['sbatch', str(resolve(sbatch_script))]
    else:
        cmd = ['bash', str(_PROJECT_ROOT / 'scripts' / 'train_stage7.sh')]
    logger.info(f"Step 4: training phase {phase}: CONFIG={env.get('CONFIG')} OUT={env.get('OUT')} {' '.join(cmd)}")
    return _run(f'Step 4: training phase {phase}', cmd, dry_run, env=env)


def _run(label: str, cmd: list, dry_run: bool, env: dict = None) -> bool:
    if dry_run:
        logger.info(f"[DRY RUN] {label}: {' '.join(cmd)}")
        return True
    logger.info(f"{label}: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(_PROJECT_ROOT), env=env)
    if result.returncode != 0:
        logger.error(f"{label} FAILED (exit {result.returncode})")
        return False
    logger.info(f"{label} complete ✓")
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Stage 7 orchestrator")
    parser.add_argument('--config', default=None, type=str,
                        help='Path to config YAML (default: finetune/genecompass/configs/rat_finetune_v2.yaml; April: rat_finetune.yaml)')
    parser.add_argument('--from', dest='start_step', type=int, default=1, choices=[1, 2, 3, 4],
                        help='Start from step N (1=init, 2=mix, 3=probe, 4=train)')
    parser.add_argument('--to', dest='end_step', type=int, default=4, choices=[1, 2, 3, 4],
                        help='Stop after step N (default 4)')
    parser.add_argument('--phase', type=int, default=1, choices=[1, 2],
                        help='Training phase for step 4 (run 1, then 2)')
    parser.add_argument('--sbatch', default=None, metavar='SCRIPT',
                        help='Step 4: submit this batch script (with PHASE/CONFIG/OUT in its environment) '
                             'instead of running scripts/train_stage7.sh in the foreground')
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

    steps = {
        1: lambda: run_model_init(config, dry_run=args.dry_run),
        2: lambda: build_mix(config, dry_run=args.dry_run),
        3: lambda: build_probe(config, dry_run=args.dry_run),
        4: lambda: train(config, args.phase, dry_run=args.dry_run, sbatch_script=args.sbatch),
    }
    for n in range(args.start_step, args.end_step + 1):
        if not steps[n]():
            sys.exit(1)

    logger.info("Stage 7 orchestration complete")


if __name__ == '__main__':
    main()