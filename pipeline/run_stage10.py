#!/usr/bin/env python3
"""
run_stage10.py -- Stage 10 Orchestrator: Aim-2 cell-type-resolved analysis
(per-cell-type DE -> positive-control comparison)

The Aim-2 ANALYSIS layer, downstream of Stage 8 (deconvolution -> per-cell-type Z).
Unlike Stage 8/9 this is a WHOLE-EXPERIMENT driver, NOT per-tissue: the exhaustive
DE's global IHW (tissue covariate) and repfdr (8w sex-consistency) pool across ALL
tissues/blocks, and the positive-control comparison is experiment-wide. A tissue
subset is accepted only for debugging.

Chain:
  1. Per-cell-type pseudobulk DE on BayesPrism Z   (R; deconvolution/R/run_pseudobulk_de.sh)
       Vetr-faithful: per-sex limma-trend over factor(week) + per-timepoint contrasts +
       sex*dose interaction + ordinal slope; Fisher sex-combine; global IHW~tissue;
       repfdr 8w sex-consistency; full gene coverage.
       -> genecompass_input/pseudobulk_de/{de_summary.tsv, de_hotspots.tsv, de_methods.tsv,
          <TISSUE>/de__*.tsv}
  2. Pre-registered positive-control comparison    (py; deconvolution/compare_posctrl.py)
       executes the FROZEN spec reference/posctrl_prereg.tsv (Tier A direction / B identity /
       C responsiveness) with the frozen miss-ladder.
       -> .../pseudobulk_de/{posctrl_results.tsv, posctrl_responsiveness.tsv, posctrl_summary.md}

Usage:
  python pipeline/run_stage10.py                      # full experiment (all tissues)
  python pipeline/run_stage10.py --tissues BLOOD SKMVL # debug subset (DE only; comparison still global)
  python pipeline/run_stage10.py --from 2             # re-run only the comparison
  python pipeline/run_stage10.py --dry-run

HPC: step 1 is an R job (limma/IHW/repfdr on existing pred_z; NO run.prism, NO GPU compute,
but this cluster needs a compute node -- submit via slurm/analysis/run_pseudobulk_de.slurm or
run this orchestrator inside an sbatch job). Full gene coverage; never subsamples.
"""
import argparse
import logging
import os
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
DE_WRAPPER = DECONV / 'R' / 'run_pseudobulk_de.sh'
COMPARE = DECONV / 'compare_posctrl.py'
PY = os.environ.get('DECONV_PYTHON') or sys.executable


def build_steps(ctx: dict) -> list:
    return [
        {
            'num': 1,
            'name': 'Per-cell-type pseudobulk DE on Z (R)',
            'desc': 'run_pseudobulk_de.sh -> de_summary.tsv + de_hotspots.tsv + <TISSUE>/de__*.tsv',
            'cmd':  ['bash', str(DE_WRAPPER)] + ctx['tissues'],
            'key':  ctx['de_dir'] / 'de_summary.tsv',
        },
        {
            'num': 2,
            'name': 'Pre-registered positive-control comparison (py)',
            'desc': 'compare_posctrl.py -> posctrl_results.tsv + posctrl_responsiveness.tsv + posctrl_summary.md',
            'cmd':  [PY, str(COMPARE),
                     '--de-dir', str(ctx['de_dir']),
                     '--prereg', str(ctx['prereg']),
                     '--out', str(ctx['de_dir']),
                     '--alpha', str(ctx['alpha']),
                     '--min-fraction', str(ctx['min_fraction']),
                     '--min-nonzero', str(ctx['min_nonzero'])],
            'key':  ctx['de_dir'] / 'posctrl_summary.md',
        },
    ]


def validate_inputs(ctx: dict, from_step: int) -> list:
    errors = []
    if from_step <= 1:
        motrpac = ctx['results_root']
        if not motrpac.exists() or not any(motrpac.glob('*/pred_z/genes.txt')):
            errors.append(
                f"no per-tissue pred_z found under {motrpac}\n"
                f"  -> run Stage 8 deconvolution first (python pipeline/run_stage8.py ...)"
            )
    if from_step == 2 and not (ctx['de_dir'] / 'de_summary.tsv').exists():
        errors.append(f"de_summary.tsv not found under {ctx['de_dir']}\n  -> run step 1 first")
    if not ctx['prereg'].exists():
        errors.append(
            f"frozen pre-registration not found: {ctx['prereg']}\n"
            f"  -> python deconvolution/build_posctrl_prereg.py"
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
        rel = key.relative_to(_PROJECT_ROOT) if key.is_relative_to(_PROJECT_ROOT) else key
        logger.info(f"  output: {rel} [OK]")
    else:
        logger.warning(f"  expected output missing: {key}")
    logger.info(f"Step {step['num']} completed in {elapsed:.1f}s")
    return True


def main() -> None:
    p = argparse.ArgumentParser(
        description='Stage 10: Aim-2 analysis (per-cell-type DE -> positive-control comparison)',
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument('--tissues', nargs='*', default=[],
                   help='optional tissue subset for the DE step (debug only; comparison stays global)')
    p.add_argument('--alpha', type=float, default=0.05)
    p.add_argument('--min-fraction', type=float, default=0.01, help='power floor for the comparison')
    p.add_argument('--min-nonzero', type=int, default=25, help='power floor for the comparison')
    p.add_argument('--from', dest='from_step', type=int, default=1, choices=[1, 2])
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('-v', '--verbose', action='store_true')
    args = p.parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    config = load_config()
    d = config['deconvolution']
    de_dir = resolve_path(config, d['genecompass_input_dir']) / 'pseudobulk_de'
    results_root = resolve_path(config, d['results_dir']) / 'motrpac'
    prereg = resolve_path(config, d['reference_dir']) / 'posctrl_prereg.tsv'

    ctx = {
        'de_dir': de_dir,
        'results_root': results_root,
        'prereg': prereg,
        'tissues': [t.upper() for t in args.tissues],
        'alpha': args.alpha,
        'min_fraction': args.min_fraction,
        'min_nonzero': args.min_nonzero,
    }

    logger.info("=" * 70)
    logger.info("STAGE 10: AIM-2 ANALYSIS (per-cell-type DE -> positive-control comparison)")
    logger.info("=" * 70)
    logger.info(f"  scope = {'tissues ' + ','.join(ctx['tissues']) if ctx['tissues'] else 'ALL tissues (whole experiment)'}")
    logger.info(f"  de_dir = {de_dir}")
    logger.info(f"  prereg = {prereg}")
    if ctx['tissues']:
        logger.warning("  NOTE: tissue subset given -- global IHW/repfdr and the comparison are still "
                       "experiment-wide; a subset DE changes the global FDR pool (use for debugging only).")

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
            logger.error(f"Stage 10 aborted at step {step['num']}")
            sys.exit(1)

    logger.info("=" * 70)
    logger.info(f"STAGE 10 COMPLETE -- {len(steps)} step(s) in {time.time() - t_total:.1f}s")
    if not args.dry_run:
        logger.info(f"  DE:       {de_dir / 'de_summary.tsv'}")
        logger.info(f"  verdict:  {de_dir / 'posctrl_summary.md'}")
    logger.info("=" * 70)


if __name__ == '__main__':
    main()
