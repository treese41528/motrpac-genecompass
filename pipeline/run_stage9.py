#!/usr/bin/env python3
"""
run_stage9.py — Stage 9 Orchestrator: Tokenize + Embed (pseudo-cells -> embeddings)

The Aim-2 bridge, step 2 of 2. Consumes Stage 8's per-tissue pseudo-cells and
produces 768-d GeneCompass cell embeddings. Drives the existing scripts IN PLACE
(no files moved). Exploration/analysis (Aim-2 gate, probe, Augur, viewer) is NOT
part of the pipeline and is run separately.

Per-tissue chain:
  1. deconvolution/tokenize_pseudocells.py            (py; reuses pipeline/05 tokenizer)
       pseudocells.h5ad -> normalize_total(--target-sum 6500) -> rank -> top-N
       token seqs -> dataset/ (+ tokenize_summary.json)
  2. finetune/genecompass/embed_cells.py              (py; GPU)
       CLS (position-0) embedding from the fine-tuned rat GeneCompass ->
       <gc>/embeddings/cell_embeddings.npy

Input is Stage 8's output dir genecompass_input/<label>/pseudocells.h5ad, so pass
the same --label (or --tissue) you used for Stage 8.

Usage:
  python pipeline/run_stage9.py --label skmgn
  python pipeline/run_stage9.py --tissue SKM-GN                 # label derived as 'skmgn'
  python pipeline/run_stage9.py --label blood --from 2 --device cuda
  python pipeline/run_stage9.py --label liver --dry-run

HPC: step 2 (embed) needs a GPU node. Full coverage: --n-cells defaults to every
pseudo-cell (never subsamples; embed_cells random-subsamples and breaks row
alignment if --n-cells < pseudo-cell count).
"""

import argparse
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
EMBED_SCRIPT = _PROJECT_ROOT / 'finetune' / 'genecompass' / 'embed_cells.py'

# Python for the worker scripts: the interpreter running this orchestrator
# (launch with the project venv active). Override with DECONV_PYTHON.
PY = os.environ.get('DECONV_PYTHON') or sys.executable

_FALLBACK_N_CELLS = 1_000_000  # >> any tissue's pseudo-cell count -> embeds all


def slug(s: str) -> str:
    return re.sub(r'[^a-z0-9]', '', s.lower())


def resolve_model_dir(base: Path) -> Path:
    """Accept either a model dir (has config.json) or a parent holding checkpoint-*;
    in the latter case pick the highest-numbered checkpoint."""
    if (base / 'config.json').exists():
        return base
    cks = sorted(base.glob('checkpoint-*'),
                 key=lambda p: int(re.sub(r'\D', '', p.name) or 0))
    return cks[-1] if cks else base


def pseudocell_count(gc_dir: Path):
    """Read the pseudo-cell count from Stage 8's summary.txt (pseudocells=N)."""
    summ = gc_dir / 'summary.txt'
    if summ.exists():
        m = re.search(r'pseudocells=(\d+)', summ.read_text())
        if m:
            return int(m.group(1))
    return None


def build_steps(ctx: dict) -> list:
    pa = ['--pa-genes', str(ctx['pa_genes'])] if ctx['pa_genes'] else []
    return [
        {
            'num':  1,
            'name': 'Tokenize pseudo-cells',
            'desc': 'tokenize_pseudocells.py -> dataset/ (+ tokenize_summary.json)',
            'cmd':  [PY, str(DECONV / 'tokenize_pseudocells.py'),
                     '--h5ad', str(ctx['h5ad']),
                     '--out', str(ctx['gc_dir']),
                     '--target-sum', str(ctx['target_sum']),
                     '--species', '2'] + pa,
            'key':  ctx['dataset'],
        },
        {
            'num':  2,
            'name': 'Embed — GeneCompass CLS embeddings (GPU)',
            'desc': 'embed_cells.py -> embeddings/cell_embeddings.npy',
            'cmd':  [PY, str(EMBED_SCRIPT),
                     '--model-dir', str(ctx['model_dir']),
                     '--dataset', str(ctx['dataset']),
                     '--output', str(ctx['emb_dir']),
                     '--n-cells', str(ctx['n_cells']),
                     '--species', '2',
                     '--device', ctx['device']],
            'key':  ctx['emb_dir'] / 'cell_embeddings.npy',
        },
    ]


def validate_inputs(ctx: dict, from_step: int) -> list:
    errors = []
    if from_step <= 1 and not ctx['h5ad'].exists():
        errors.append(
            f"pseudocells.h5ad not found: {ctx['h5ad']}\n"
            f"  -> run Stage 8 first (python pipeline/run_stage8.py --tissue ... --ref-dir ...)"
        )
    if from_step == 2 and not ctx['dataset'].exists():
        errors.append(f"tokenized dataset not found: {ctx['dataset']}\n  -> run step 1 first")
    if not (ctx['model_dir'] / 'config.json').exists():
        errors.append(
            f"model config.json not found under: {ctx['model_dir']}\n"
            f"  -> pass --model-dir pointing at the fine-tuned checkpoint"
        )
    return errors


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
        logger.error(f"Step {step['num']} FAILED (exit {result.returncode}) after {elapsed:.1f}s")
        return False
    key = step['key']
    if key.exists():
        logger.info(f"  output: {key.relative_to(_PROJECT_ROOT) if key.is_relative_to(_PROJECT_ROOT) else key} [OK]")
    else:
        logger.warning(f"  expected output missing: {key}")
    logger.info(f"Step {step['num']} completed in {elapsed:.1f}s")
    return True


def main() -> None:
    p = argparse.ArgumentParser(
        description='Stage 9: Tokenize + Embed (pseudo-cells -> embeddings)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument('--label', default=None, help='genecompass_input/<label> dir (Stage 8 output)')
    p.add_argument('--tissue', default=None, help='MoTrPAC code; if --label omitted, label = slug(tissue)')
    p.add_argument('--model-dir', default=None,
                   help='fine-tuned rat GeneCompass dir or a parent with checkpoint-* '
                        '(default: deconvolution.genecompass_model_dir from config -> latest checkpoint)')
    p.add_argument('--n-cells', type=int, default=None,
                   help='cells to embed (default: all pseudo-cells; never subsample)')
    p.add_argument('--target-sum', type=float, default=6500.0,
                   help='tokenizer normalize_total target (default 6500; calibrated to the corpus)')
    p.add_argument('--pa-genes', action='store_true',
                   help='also pass the PA/training-regulated gene priority list to the tokenizer (off by default)')
    p.add_argument('--device', default='cuda')
    p.add_argument('--from', dest='from_step', type=int, default=1, choices=[1, 2])
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('-v', '--verbose', action='store_true')
    args = p.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    label = args.label or (slug(args.tissue) if args.tissue else None)
    if not label:
        p.error('one of --label or --tissue is required')

    config = load_config()
    d = config['deconvolution']

    gc_dir = resolve_path(config, d['genecompass_input_dir']) / label
    model_base = Path(args.model_dir).resolve() if args.model_dir \
        else resolve_path(config, d['genecompass_model_dir'])

    n_cells = args.n_cells
    if n_cells is None:
        cnt = pseudocell_count(gc_dir)
        n_cells = cnt if cnt else _FALLBACK_N_CELLS

    ctx = {
        'label':      label,
        'gc_dir':     gc_dir,
        'h5ad':       gc_dir / 'pseudocells.h5ad',
        'dataset':    gc_dir / 'dataset',
        'emb_dir':    gc_dir / 'embeddings',
        'model_dir':  resolve_model_dir(model_base),
        'pa_genes':   resolve_path(config, d['pa_genes']) if (args.pa_genes and d.get('pa_genes')) else None,
        'target_sum': args.target_sum,
        'n_cells':    n_cells,
        'device':     args.device,
    }

    logger.info("=" * 70)
    logger.info("STAGE 9: TOKENIZE + EMBED (pseudo-cells -> embeddings)")
    logger.info("=" * 70)
    logger.info(f"  label={label}  target_sum={ctx['target_sum']}  n_cells={ctx['n_cells']}  device={ctx['device']}")
    logger.info(f"  in    = {ctx['h5ad']}")
    logger.info(f"  model = {ctx['model_dir']}")
    logger.info(f"  out   = {ctx['emb_dir']}/cell_embeddings.npy")

    errors = validate_inputs(ctx, args.from_step)
    if errors:
        for e in errors:
            (logger.warning if args.dry_run else logger.error)(e)
        if not args.dry_run:
            sys.exit(1)

    env = dict(os.environ, PIPELINE_ROOT=str(_PROJECT_ROOT))
    steps = [s for s in build_steps(ctx) if s['num'] >= args.from_step]
    t_total = time.time()
    for step in steps:
        if not run_step(step, env, args.dry_run):
            logger.error(f"Stage 9 aborted at step {step['num']}")
            sys.exit(1)

    logger.info("=" * 70)
    logger.info(f"STAGE 9 COMPLETE — {len(steps)} step(s) in {time.time() - t_total:.1f}s")
    if not args.dry_run:
        logger.info(f"  embeddings: {ctx['emb_dir'] / 'cell_embeddings.npy'}")
    logger.info("  Exploration (separate, not pipeline): deconvolution/pheno_merge_test.py, "
                "subspace_probe.py, augur_prep.py + run_augur.R, build_umap_viewer.py")
    logger.info("=" * 70)


if __name__ == '__main__':
    main()
