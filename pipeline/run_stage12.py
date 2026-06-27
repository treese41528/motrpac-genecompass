#!/usr/bin/env python3
"""
run_stage12.py -- Stage 12 Orchestrator: cross-species TRANSFER of the rat exercise
response into human GeneCompass embedding space + survival analysis (Aim 3a / Module E).

THE MAIN PURPOSE of the cross-species work. MoTrPAC's invasive multi-tissue exercise
time-course cannot be run in humans; the rat is the proxy and GeneCompass is the transfer
vehicle. This driver re-expresses every rat pseudo-cell AS human (E.1) and then tests
whether the trained-vs-control / ordinal-dose axis SURVIVES the transfer, per cell type (E.2).
One-directional transfer (rat data -> human representation -> analyze); NO human dataset needed.

Per-tissue chain (steps 1-2, like Stage 9), then whole-experiment (steps 3-4, like Stage 10):
  1. transfer_to_human.py            (py, CPU)  rat ENSRNOG pseudo-cells --ortholog--> human ENSG
       --tokenize(human tokens+medians, target-sum 6500, top-2048, species=0)--> dataset/
  2. embed_cells.py --species 0      (py, GPU)  CLS (position-0) embedding from the SAME fine-tuned
       GeneCompass checkpoint -> <human-root>/<label>/embeddings/cell_embeddings.npy
  3. subspace_probe.py --gc-root <human-root>   (py, CPU)  PLS-1 CV: sup_trained_auc + sup_dose_rho
       on the HUMAN-space embeddings (primary E.2 detector) -> <human-root>/subspace_probe.tsv
  4. compare_transfer.py             (py, CPU)  rat vs human per (tissue,cell_type): does the axis
       survive? -> <human-root>/transfer_comparison.tsv + transfer_comparison.md

Usage:
  python pipeline/run_stage12.py                       # all tissues (transfer+embed) -> probe -> compare
  python pipeline/run_stage12.py --labels liver skmvl  # subset for the per-tissue steps
  python pipeline/run_stage12.py --from 3              # re-run only the detection + comparison
  python pipeline/run_stage12.py --label liver --dry-run

HPC: step 2 (embed) needs a GPU node -- submit via slurm/analysis/run_stage12.slurm or run this
orchestrator inside an sbatch job. Full coverage: --n-cells defaults to every pseudo-cell.
"""
import argparse
import glob
import logging
import os
import re
import subprocess
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(os.environ.get('PIPELINE_ROOT') or Path(__file__).resolve().parents[1])
os.environ.setdefault('PIPELINE_ROOT', str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / 'lib'))
from gene_utils import load_config, resolve_path  # noqa: E402

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

DECONV = _PROJECT_ROOT / 'deconvolution'
TRANSFER = _PROJECT_ROOT / 'translation' / 'transfer_to_human.py'
EMBED = _PROJECT_ROOT / 'finetune' / 'genecompass' / 'embed_cells.py'
PROBE = DECONV / 'subspace_probe.py'
COMPARE = _PROJECT_ROOT / 'translation' / 'compare_transfer.py'
PY = os.environ.get('DECONV_PYTHON') or sys.executable

_FALLBACK_N_CELLS = 1_000_000  # >> any tissue's pseudo-cell count -> embeds all (never subsample)


def resolve_model_dir(base: Path) -> Path:
    """Accept a model dir (has config.json) or a parent holding checkpoint-*; pick the latest."""
    if (base / 'config.json').exists():
        return base
    cks = sorted(base.glob('checkpoint-*'), key=lambda p: int(re.sub(r'\D', '', p.name) or 0))
    return cks[-1] if cks else base


def discover_labels(rat_root: Path) -> list:
    """All rat tissue labels (dirs holding pseudocells.h5ad) under the rat gc-root."""
    return sorted(p.parent.name for p in rat_root.glob('*/pseudocells.h5ad'))


def per_tissue_steps(label: str, ctx: dict) -> list:
    out_dir = ctx['human_root'] / label
    return [
        {
            'num': 1, 'label': label,
            'name': f'[{label}] Transfer rat -> human (project + tokenize, species=0)',
            'desc': 'transfer_to_human.py -> <human-root>/<label>/dataset/',
            'cmd':  [PY, str(TRANSFER), '--label', label,
                     '--in-root', str(ctx['rat_root']), '--out-root', str(ctx['human_root']),
                     '--target-sum', str(ctx['target_sum'])],
            'key':  out_dir / 'dataset',
        },
        {
            'num': 2, 'label': label,
            'name': f'[{label}] Embed -- GeneCompass CLS embeddings, species=0 (GPU)',
            'desc': 'embed_cells.py --species 0 -> <human-root>/<label>/embeddings/cell_embeddings.npy',
            'cmd':  [PY, str(EMBED), '--model-dir', str(ctx['model_dir']),
                     '--dataset', str(out_dir / 'dataset'), '--output', str(out_dir / 'embeddings'),
                     '--n-cells', str(ctx['n_cells']), '--species', '0', '--device', ctx['device']],
            'key':  out_dir / 'embeddings' / 'cell_embeddings.npy',
        },
    ]


def global_steps(ctx: dict) -> list:
    hr = ctx['human_root']
    return [
        {
            'num': 3, 'label': None,
            'name': 'Supervised probe on HUMAN-space embeddings (primary E.2 detector)',
            'desc': 'subspace_probe.py --gc-root <human-root> -> subspace_probe.tsv',
            'cmd':  [PY, str(PROBE), '--gc-root', str(hr),
                     '--out', str(hr / 'subspace_probe.tsv'),
                     '--perms', str(ctx['perms']), '--jobs', str(ctx['jobs'])],
            'key':  hr / 'subspace_probe.tsv',
        },
        {
            'num': 4, 'label': None,
            'name': 'Compare rat vs human -- does the exercise axis survive transfer? (E.2 deliverable)',
            'desc': 'compare_transfer.py -> transfer_comparison.tsv + transfer_comparison.md',
            'cmd':  [PY, str(COMPARE), '--rat-corr', str(ctx['rat_corr']),
                     '--human-probe', str(hr / 'subspace_probe.tsv'),
                     '--out', str(hr)],
            'key':  hr / 'transfer_comparison.md',
        },
    ]


def run_step(step: dict, env: dict, dry_run: bool) -> bool:
    logger.info("=" * 70)
    logger.info(f"STEP {step['num']}: {step['name']}")
    logger.info(f"  {step['desc']}")
    logger.info(f"  $ {' '.join(step['cmd'])}")
    logger.info("=" * 70)
    if dry_run:
        logger.info("  [dry-run] not executed")
        return True
    t0 = time.time()
    result = subprocess.run(step['cmd'], cwd=str(_PROJECT_ROOT), env=env)
    elapsed = time.time() - t0
    if result.returncode != 0:
        logger.error(f"Step {step['num']} ({step['label'] or 'global'}) FAILED "
                     f"(exit {result.returncode}) after {elapsed:.1f}s")
        return False
    key = step['key']
    if key.exists():
        rel = key.relative_to(_PROJECT_ROOT) if key.is_relative_to(_PROJECT_ROOT) else key
        logger.info(f"  output: {rel} [OK]")
    else:
        logger.warning(f"  expected output missing: {key}")
    logger.info(f"Step {step['num']} completed in {elapsed:.1f}s")
    return True


def main() -> None:
    p = argparse.ArgumentParser(
        description='Stage 12: rat->human transfer + exercise-axis survival analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument('--labels', nargs='*', default=None,
                   help='tissue labels for the per-tissue steps (default: all rat tissues)')
    p.add_argument('--label', default=None, help='single label (alias for --labels X)')
    p.add_argument('--model-dir', default=None,
                   help='fine-tuned GeneCompass dir or parent with checkpoint-* '
                        '(default: deconvolution.genecompass_model_dir -> latest checkpoint)')
    p.add_argument('--target-sum', type=float, default=6500.0,
                   help='tokenizer normalize_total target; MUST match the rat run (default 6500)')
    p.add_argument('--n-cells', type=int, default=None, help='cells to embed (default: all; never subsample)')
    p.add_argument('--perms', type=int, default=1000, help='subspace_probe permutations')
    p.add_argument('--jobs', type=int, default=8, help='subspace_probe parallel jobs')
    p.add_argument('--device', default='cuda')
    p.add_argument('--from', dest='from_step', type=int, default=1, choices=[1, 2, 3, 4])
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('-v', '--verbose', action='store_true')
    args = p.parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    config = load_config()
    d = config['deconvolution']
    rat_root = resolve_path(config, d['genecompass_input_dir'])
    human_root = Path(str(rat_root) + '_human')
    model_base = Path(args.model_dir).resolve() if args.model_dir \
        else resolve_path(config, d['genecompass_model_dir'])

    labels = args.labels or ([args.label] if args.label else discover_labels(rat_root))
    labels = [l for l in labels if l]
    if not labels:
        p.error(f'no tissue labels found under {rat_root} (need <label>/pseudocells.h5ad)')

    ctx = {
        'rat_root': rat_root, 'human_root': human_root,
        'model_dir': resolve_model_dir(model_base),
        'rat_corr': rat_root / 'corroboration_merged.tsv',
        'target_sum': args.target_sum,
        'n_cells': args.n_cells if args.n_cells is not None else _FALLBACK_N_CELLS,
        'perms': args.perms, 'jobs': args.jobs, 'device': args.device,
    }

    logger.info("=" * 70)
    logger.info("STAGE 12: RAT -> HUMAN TRANSFER + EXERCISE-AXIS SURVIVAL (Aim 3a / Module E)")
    logger.info("=" * 70)
    logger.info(f"  labels   = {', '.join(labels)}")
    logger.info(f"  rat in   = {rat_root}")
    logger.info(f"  human out= {human_root}")
    logger.info(f"  model    = {ctx['model_dir']}")
    logger.info(f"  steps from {args.from_step}: 1-2 per-tissue (transfer+embed), 3-4 whole-experiment (probe+compare)")

    if args.from_step <= 2 and not (ctx['model_dir'] / 'config.json').exists():
        msg = f"model config.json not found under {ctx['model_dir']} (needed for the embed step)  -> pass --model-dir"
        (logger.warning if args.dry_run else logger.error)(msg)
        if not args.dry_run:
            sys.exit(1)

    env = dict(os.environ, PIPELINE_ROOT=str(_PROJECT_ROOT))
    t_total = time.time()

    # steps 1-2: per tissue
    for label in labels:
        for step in per_tissue_steps(label, ctx):
            if step['num'] < args.from_step:
                continue
            if not run_step(step, env, args.dry_run):
                logger.error(f"Stage 12 aborted at step {step['num']} ({label})")
                sys.exit(1)

    # steps 3-4: whole-experiment
    for step in global_steps(ctx):
        if step['num'] < args.from_step:
            continue
        if not run_step(step, env, args.dry_run):
            logger.error(f"Stage 12 aborted at step {step['num']}")
            sys.exit(1)

    logger.info("=" * 70)
    logger.info(f"STAGE 12 COMPLETE in {time.time() - t_total:.1f}s")
    if not args.dry_run:
        logger.info(f"  human embeddings: {human_root}/<label>/embeddings/cell_embeddings.npy")
        logger.info(f"  survival verdict: {human_root / 'transfer_comparison.md'}")
    logger.info("=" * 70)


if __name__ == '__main__':
    main()
