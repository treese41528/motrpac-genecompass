#!/usr/bin/env python3
"""
run_stage8.py — Stage 8 Orchestrator: Deconvolution (MoTrPAC bulk -> pseudo-cells)

The Aim-2 bridge, step 1 of 2. Drives the existing deconvolution/ scripts IN
PLACE (no files are moved). Only the production path lives in the pipeline; the
exploration/analysis scripts (Aim-2 gate, subspace probe, Augur, corroboration,
UMAP viewer, validation/purity-sweep, omnideconv) are deliberately NOT part of
it. Stage 9 (run_stage9.py) takes this stage's pseudo-cells -> tokenize -> embed.

Per-tissue chain (each step calls a deconvolution/ script; R via its .sh wrapper):
  1. R/prepare_motrpac_bulk.sh  <TISSUE> <bulk_root>          (R; login node OK)
       lift TRNSCRPT_<TISSUE>_RAW_COUNTS -> <bulk_root>/<TISSUE>/bulk.{mtx,_genes,_samples}
  2. R/run_deconvolution.sh     <ref> <bulk/TISSUE> <res/TISSUE> bulk   (R; COMPUTE node)
       BayesPrism new.prism -> run.prism -> estimated_fractions.csv + bp_result.rds
  3. R/extract_z.sh             <res/TISSUE>/bp_result.rds <res/TISSUE>  (R; light)
       posterior Z -> pred_z/{genes.txt,types.txt,predz__*.csv}
  4. build_pseudocells.py       --pred-z-dir ... --out <gc/label>       (py)
       one pseudo-cell per (sample x cell type) -> pseudocells.h5ad

You supply: --tissue (MoTrPAC bulk code, matches TRNSCRPT_<TISSUE>_RAW_COUNTS.rda)
and --ref-dir (the built SC reference for that tissue; irregular per tissue --
see deconvolution/build_all_references.sh and the survey). Everything else
defaults from config/pipeline_config.yaml [deconvolution].

Usage:
  python pipeline/run_stage8.py --tissue SKM-GN \
      --ref-dir "data/deconvolution/references/skeletal muscle_GSE254371"
  python pipeline/run_stage8.py --tissue BLOOD --ref-dir ... --from 2  # bulk prepared
  python pipeline/run_stage8.py --tissue LIVER --ref-dir ... --dry-run

HPC: steps 1/3/4 are light; step 2 (run.prism) is CPU-heavy -> a compute node.
The per-tissue SLURM job (slurm/, local) remains the production driver; this
orchestrator runs the chain serially for one tissue (local / single-node).
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

DECONV   = _PROJECT_ROOT / 'deconvolution'
DECONV_R = DECONV / 'R'

# Python for the deconvolution worker scripts: the interpreter running this
# orchestrator (launch it with the project venv active, like the other stages).
# Override with DECONV_PYTHON if a worker needs a different interpreter.
PY = os.environ.get('DECONV_PYTHON') or sys.executable


def slug(s: str) -> str:
    return re.sub(r'[^a-z0-9]', '', s.lower())


# ─────────────────────────────────────────────────────────────────────────────
# STEPS
# ─────────────────────────────────────────────────────────────────────────────

def build_steps(ctx: dict) -> list:
    """Per-tissue deconvolution chain. Each 'cmd' calls an existing deconvolution/
    script in place; 'key' is the artifact that proves the step ran."""
    return [
        {
            'num':  1,
            'name': 'Prepare bulk — liftover MoTrPAC counts to current ENSRNOG',
            'desc': 'R/prepare_motrpac_bulk.sh -> bulk.{mtx,_genes,_samples}',
            'cmd':  ['bash', str(DECONV_R / 'prepare_motrpac_bulk.sh'),
                     ctx['tissue'], str(ctx['bulk_root'])],
            'key':  ctx['bulk_dir'] / 'bulk.mtx',
        },
        {
            'num':  2,
            'name': 'Deconvolve — BayesPrism (run.prism is CPU-heavy; compute node)',
            'desc': 'R/run_deconvolution.sh -> estimated_fractions.csv + bp_result.rds',
            'cmd':  ['bash', str(DECONV_R / 'run_deconvolution.sh'),
                     str(ctx['ref_dir']), str(ctx['bulk_dir']), str(ctx['res_dir']), 'bulk'],
            'key':  ctx['bp_rds'],
        },
        {
            'num':  3,
            'name': 'Extract Z — posterior per-cell-type expression',
            'desc': 'R/extract_z.sh -> pred_z/{genes.txt,types.txt,predz__*.csv}',
            'cmd':  ['bash', str(DECONV_R / 'extract_z.sh'),
                     str(ctx['bp_rds']), str(ctx['res_dir'])],
            'key':  ctx['predz_dir'] / 'genes.txt',
        },
        {
            'num':  4,
            'name': 'Build pseudo-cells — one per (sample x cell type)',
            'desc': 'build_pseudocells.py -> pseudocells.h5ad',
            'cmd':  [PY, str(DECONV / 'build_pseudocells.py'),
                     '--pred-z-dir', str(ctx['predz_dir']),
                     '--tissue', ctx['label'],
                     '--out', str(ctx['gc_dir'])]
                    + (['--meta-tsv', str(ctx['sample_pheno'])] if ctx['with_pheno'] else []),
            'key':  ctx['gc_dir'] / 'pseudocells.h5ad',
        },
    ]


# ─────────────────────────────────────────────────────────────────────────────
# VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

def validate_inputs(ctx: dict, from_step: int) -> list:
    """Return a list of error strings (empty == OK)."""
    errors = []

    if from_step <= 1:
        rda = ctx['motrpac_data'] / f"TRNSCRPT_{ctx['tissue']}_RAW_COUNTS.rda"
        if not rda.exists():
            errors.append(
                f"Source bulk not found: {rda}\n"
                f"  -> check --tissue spelling (must match TRNSCRPT_<TISSUE>_RAW_COUNTS.rda) "
                f"and deconvolution.motrpac_bulk_dir"
            )
        else:
            logger.info(f"  source bulk: {rda.name} [OK]")
    else:
        # Skipping prepare: the lifted bulk must already exist
        if not (ctx['bulk_dir'] / 'bulk.mtx').exists():
            errors.append(
                f"Lifted bulk not found: {ctx['bulk_dir'] / 'bulk.mtx'}\n"
                f"  -> run without --from (step 1 builds it)"
            )

    if from_step <= 2:
        for f in ('reference_counts.mtx', 'genes.tsv', 'cells_meta.tsv'):
            if not (ctx['ref_dir'] / f).exists():
                errors.append(
                    f"Reference file missing: {ctx['ref_dir'] / f}\n"
                    f"  -> build it (deconvolution/build_all_references.sh) or fix --ref-dir"
                )

    if from_step == 3 and not ctx['bp_rds'].exists():
        errors.append(
            f"bp_result.rds not found: {ctx['bp_rds']}\n  -> run step 2 first"
        )
    if from_step == 4 and not (ctx['predz_dir'] / 'genes.txt').exists():
        errors.append(
            f"pred_z not found: {ctx['predz_dir']}\n  -> run step 3 first"
        )

    return errors


# ─────────────────────────────────────────────────────────────────────────────
# RUNNER
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description='Stage 8: Deconvolution (MoTrPAC bulk -> pseudo-cells)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument('--tissue', required=True,
                   help='MoTrPAC bulk code (matches TRNSCRPT_<TISSUE>_RAW_COUNTS.rda), e.g. SKM-GN')
    p.add_argument('--ref-dir', required=True,
                   help='built SC reference dir (reference_counts.mtx/genes.tsv/cells_meta.tsv)')
    p.add_argument('--label', default=None,
                   help='output short name for the genecompass_input/<label> dir (default: slug of --tissue)')
    p.add_argument('--bulk-root', default=None, help='override deconvolution.motrpac_bulk_out')
    p.add_argument('--results-dir', default=None, help='override deconvolution.results_dir')
    p.add_argument('--with-pheno', action='store_true',
                   help='join PHENO (sex/group) into pseudocell obs. Off by default: the exploration '
                        'gate does its own row-order join (the mix{i} gotcha); a plain key merge here can mis-join.')
    p.add_argument('--n-cores', type=int, default=None, help='N_CORES for BayesPrism (step 2)')
    p.add_argument('--from', dest='from_step', type=int, default=1, choices=[1, 2, 3, 4],
                   help='start from this step (default 1)')
    p.add_argument('--dry-run', action='store_true', help='validate + print the plan; run nothing')
    p.add_argument('-v', '--verbose', action='store_true')
    args = p.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    config = load_config()
    d = config['deconvolution']

    tissue = args.tissue.upper()
    label = args.label or slug(args.tissue)
    bulk_root = Path(args.bulk_root).resolve() if args.bulk_root else resolve_path(config, d['motrpac_bulk_out'])
    res_root = Path(args.results_dir).resolve() if args.results_dir else resolve_path(config, d['results_dir'])
    res_dir = res_root / tissue

    ctx = {
        'tissue':       tissue,
        'label':        label,
        'ref_dir':      Path(args.ref_dir).resolve(),
        'bulk_root':    bulk_root,
        'bulk_dir':     bulk_root / tissue,
        'res_dir':      res_dir,
        'bp_rds':       res_dir / 'bp_result.rds',
        'predz_dir':    res_dir / 'pred_z',
        'gc_dir':       resolve_path(config, d['genecompass_input_dir']) / label,
        'sample_pheno': resolve_path(config, d['sample_pheno']),
        'motrpac_data': resolve_path(config, d['motrpac_bulk_dir']),
        'with_pheno':   args.with_pheno,
    }

    logger.info("=" * 70)
    logger.info("STAGE 8: DECONVOLUTION (bulk -> pseudo-cells)")
    logger.info("=" * 70)
    logger.info(f"  tissue={tissue}  label={label}")
    logger.info(f"  ref   = {ctx['ref_dir']}")
    logger.info(f"  bulk  = {ctx['bulk_dir']}")
    logger.info(f"  res   = {ctx['res_dir']}")
    logger.info(f"  out   = {ctx['gc_dir']}")

    errors = validate_inputs(ctx, args.from_step)
    if errors:
        for e in errors:
            (logger.warning if args.dry_run else logger.error)(e)
        if not args.dry_run:
            sys.exit(1)

    env = dict(os.environ, PIPELINE_ROOT=str(_PROJECT_ROOT))
    if args.n_cores:
        env['N_CORES'] = str(args.n_cores)

    steps = [s for s in build_steps(ctx) if s['num'] >= args.from_step]
    t_total = time.time()
    for step in steps:
        if not run_step(step, env, args.dry_run):
            logger.error(f"Stage 8 aborted at step {step['num']}")
            sys.exit(1)

    logger.info("=" * 70)
    logger.info(f"STAGE 8 COMPLETE — {len(steps)} step(s) in {time.time() - t_total:.1f}s")
    if not args.dry_run:
        logger.info(f"  pseudo-cells: {ctx['gc_dir'] / 'pseudocells.h5ad'}")
    logger.info(f"  Next: python pipeline/run_stage9.py --label {label} [--model-dir ...]")
    logger.info("=" * 70)


if __name__ == '__main__':
    main()
