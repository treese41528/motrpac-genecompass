#!/usr/bin/env python3
"""
run_stage10.py -- Stage 10 Orchestrator: Aim-2 cell-type-resolved analysis
(per-cell-type DE -> positive-control comparison -> technical-covariate reruns)

The Aim-2 ANALYSIS layer, downstream of Stage 8 (deconvolution -> per-cell-type Z).
Unlike Stage 8/9 this is a WHOLE-EXPERIMENT driver, NOT per-tissue: the DE's pooled IHW
(tissue covariate) and repfdr (8w sex-consistency) fits use ALL tissues/blocks, and the
positive-control comparison is experiment-wide. A tissue subset is accepted only for debugging.

Chain:
  1. Per-cell-type pseudobulk DE on BayesPrism Z   (R; deconvolution/R/run_pseudobulk_de.sh)
       limma-trend per block; per-sex omnibus F over factor(week), combined by Fisher over the
       sexes present; pooled IHW ~ tissue (nominal); 8w sex-consistency (concordance primary,
       repfdr secondary); full gene coverage. The models are listed in de_methods.tsv.
       -> genecompass_input/pseudobulk_de/{de_summary.tsv, de_headline.tsv, de_hotspots.tsv,
          de_methods.tsv, <TISSUE>/de__*.tsv}
  2. Pre-registered positive-control comparison    (py; deconvolution/compare_posctrl.py)
       executes the FROZEN spec (deconvolution.posctrl_prereg, else reference/posctrl_prereg.tsv;
       Tier A direction / B identity / C responsiveness) with the frozen miss-ladder.
       -> .../pseudobulk_de/{posctrl_results.tsv, posctrl_responsiveness.tsv, posctrl_summary.md}
  3. Technical-covariate reruns (only with --covariate-variants; STEP3_PREREG.md 6.4)
       the whole DE again with technical covariates in every model, into sibling folders
       pseudobulk_de_covA (rin, pct_globin) and pseudobulk_de_covB (rin, pct_globin, pct_umi_dup,
       median_5_3_bias), then deconvolution/compare_de_runs.py against the headline folder
       -> .../pseudobulk_de/de_covariate_robustness.tsv

Usage:
  python pipeline/run_stage10.py                      # full experiment (all tissues)
  python pipeline/run_stage10.py --tissues BLOOD SKMVL # debug subset (DE only; comparison still global)
  python pipeline/run_stage10.py --from 2             # re-run only the comparison
  python pipeline/run_stage10.py --hotspots data/deconvolution/genecompass_input_v2_refs4all_rebuild/gate_rcv10/corroboration_merged.tsv
  python pipeline/run_stage10.py --dry-run
  python pipeline/run_stage10.py --sensitivities 100 0   # + the registered DE_MIN_MEDIAN_LIBSIZE sensitivities
  python pipeline/run_stage10.py --covariate-variants A B          # + the two registered covariate reruns
  python pipeline/run_stage10.py --from 3 --covariate-variants A B # only the reruns and their comparison

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
COMPARE_RUNS = DECONV / 'compare_de_runs.py'
PY = os.environ.get('DECONV_PYTHON') or sys.executable

# Technical covariates run_pseudobulk_de.R accepts in DE_TECH_COVARIATES (it reads them from the MoTrPAC
# package's TRNSCRPT_META), and the two registered reruns (STEP3_PREREG.md 6.4): A is the pair of the
# earlier RIN/globin check, B the four covariates of the consortium's own differential analysis.
TECH_COVARIATES = ('rin', 'pct_globin', 'pct_umi_dup', 'median_5_3_bias')
COVARIATE_VARIANTS = {'A': ('rin', 'pct_globin'),
                      'B': ('rin', 'pct_globin', 'pct_umi_dup', 'median_5_3_bias')}


def variant_dir(de_dir: Path, label: str) -> Path:
    """Sibling folder of the headline DE that holds covariate rerun <label>."""
    return Path(str(de_dir) + f'_cov{label}')


def de_env(ctx: dict, base=None) -> dict:
    """Environment of the R step. The wrapper reads the config itself (fatal on failure); the output folder,
    hotspot table, seed, significance level and technical covariates are passed explicitly so that a
    command-line option reaches run_pseudobulk_de.R (DE_ALPHA: finding 7.10, --alpha used to stop here)."""
    env = dict(os.environ if base is None else base,
               PIPELINE_ROOT=str(_PROJECT_ROOT), OUT_DIR=str(ctx['de_dir']), HOTSPOTS=str(ctx['hotspots']),
               DE_SEED=str(ctx['seed']), DE_ALPHA=repr(float(ctx['alpha'])),
               DE_TECH_COVARIATES=','.join(ctx['tech_covariates']))
    env.setdefault('DECONV_PYTHON', PY)   # the wrappers read the config with this interpreter
    return env


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


def covariate_steps(ctx: dict) -> list:
    """Step 3 (STEP3_PREREG.md 6.4): one DE rerun per requested variant into its sibling folder, with the
    variant's covariates in DE_TECH_COVARIATES, then the comparison of every variant with the headline."""
    steps, variants = [], []
    for label in ctx['covariate_variants']:
        out = variant_dir(ctx['de_dir'], label)
        variants += ['--variant', f'{label}={out}']
        steps.append({
            'num': 3,
            'name': f'DE rerun with technical covariates, variant {label}: ' + ', '.join(COVARIATE_VARIANTS[label]),
            'desc': f'run_pseudobulk_de.sh -> {out.name}/',
            'cmd':  ['bash', str(DE_WRAPPER)] + ctx['tissues'],
            'env':  {'OUT_DIR': str(out), 'DE_TECH_COVARIATES': ','.join(COVARIATE_VARIANTS[label])},
            'key':  out / 'de_summary.tsv',
        })
    if steps:
        table = ctx['de_dir'] / 'de_covariate_robustness.tsv'
        steps.append({
            'num': 3,
            'name': 'Headline DE against the covariate reruns (py)',
            'desc': f'compare_de_runs.py -> {table.name}',
            'cmd':  [PY, str(COMPARE_RUNS), '--headline', str(ctx['de_dir'])] + variants + ['--out', str(table)],
            'key':  table,
        })
    return steps


def validate_inputs(ctx: dict, from_step: int) -> list:
    errors = []
    runs_de = from_step <= 1 or bool(ctx['covariate_variants'])
    if runs_de:
        motrpac = ctx['results_root']
        if not motrpac.exists() or not any(motrpac.glob('*/pred_z/genes.txt')):
            errors.append(
                f"no per-tissue pred_z found under {motrpac}\n"
                f"  -> run Stage 8 deconvolution first (python pipeline/run_stage8.py ...)"
            )
        if not ctx['hotspots'].exists():   # the R step would read it as "no hotspots" and carry on
            errors.append(f"hotspot table not found: {ctx['hotspots']}\n"
                          f"  -> run the Aim-2 gate (corroborate_summary.py) first, or pass --hotspots")
    needs_headline = from_step == 2 or (from_step == 3 and ctx['covariate_variants'])
    if needs_headline and not (ctx['de_dir'] / 'de_summary.tsv').exists():
        errors.append(f"de_summary.tsv not found under {ctx['de_dir']}\n  -> run step 1 first")
    if from_step <= 2 and not ctx['prereg'].exists():
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


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Stage 10: Aim-2 analysis (per-cell-type DE -> positive-control comparison)',
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument('--tissues', nargs='*', default=[],
                   help='optional tissue subset for the DE step (debug only; comparison stays global)')
    p.add_argument('--hotspots', default=None,
                   help='Aim-2 gate table. Its is_hotspot column marks the hotspots when present, else '
                        'q_sup_trained < 0.05 '
                        '(default: deconvolution.hotspots_table, else <genecompass_input_dir>/corroboration_merged.tsv)')
    p.add_argument('--alpha', type=float, default=0.05,
                   help='significance level of the DE (IHW and every count; env DE_ALPHA of the R step) and of '
                        'the positive-control comparison')
    p.add_argument('--min-fraction', type=float, default=0.01, help='power floor for the comparison')
    p.add_argument('--min-nonzero', type=int, default=25, help='power floor for the comparison')
    p.add_argument('--sensitivities', type=int, nargs='*', default=[],
                   help='after step 1, rerun the DE with DE_MIN_MEDIAN_LIBSIZE = each value into '
                        '<pseudobulk_de>_minlib<L> (registered: 100 0)')
    p.add_argument('--tech-covariates', nargs='*', default=[], choices=TECH_COVARIATES, metavar='COVARIATE',
                   help='technical covariates added to every model of the DE written to <pseudobulk_de> '
                        f'(env DE_TECH_COVARIATES of the R step; among: {", ".join(TECH_COVARIATES)}). The headline '
                        'DE has none; for the registered reruns use --covariate-variants')
    p.add_argument('--covariate-variants', nargs='*', default=[], choices=sorted(COVARIATE_VARIANTS), metavar='VARIANT',
                   help='step 3: rerun the DE with the registered covariate sets into <pseudobulk_de>_cov<VARIANT> '
                        '(A = rin pct_globin; B = rin pct_globin pct_umi_dup median_5_3_bias) and compare each '
                        'with the headline (STEP3_PREREG.md 6.4; the hotspot verdict needs both A and B)')
    p.add_argument('--seed', type=int, default=None,
                   help='IHW fold seed for the DE step (default: deconvolution.de_seed, else 1 = IHW\'s own default)')
    p.add_argument('--from', dest='from_step', type=int, default=1, choices=[1, 2, 3])
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('-v', '--verbose', action='store_true')
    args = p.parse_args(argv)
    if not 0 < args.alpha < 1:
        p.error('--alpha must be between 0 and 1')
    if args.from_step == 3 and not args.covariate_variants:
        p.error('--from 3 runs only the covariate reruns: name them with --covariate-variants A B')
    if args.tech_covariates and args.covariate_variants:
        p.error('--covariate-variants compares against a headline DE without covariates; '
                'do not combine it with --tech-covariates')
    args.tech_covariates = [c for c in TECH_COVARIATES if c in args.tech_covariates]        # fixed order, no repeats
    args.covariate_variants = [v for v in sorted(COVARIATE_VARIANTS) if v in args.covariate_variants]
    return args


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    config = load_config()
    d = config['deconvolution']
    gc_input = resolve_path(config, d['genecompass_input_dir'])
    de_dir = gc_input / 'pseudobulk_de'
    results_root = resolve_path(config, d['results_dir']) / 'motrpac'
    # The frozen positive-control spec; a profile whose references renamed cell types points at its own
    # renamed copy (deconvolution.posctrl_prereg, e.g. config/profiles/refs_v4.yaml).
    prereg = (resolve_path(config, d['posctrl_prereg']) if d.get('posctrl_prereg')
              else resolve_path(config, d['reference_dir']) / 'posctrl_prereg.tsv')
    hotspots = (Path(args.hotspots).resolve() if args.hotspots
                else resolve_path(config, d['hotspots_table']) if d.get('hotspots_table')
                else gc_input / 'corroboration_merged.tsv')

    seed = args.seed if args.seed is not None else int(d.get('de_seed', 1))
    ctx = {
        'de_dir': de_dir,
        'hotspots': hotspots,
        'results_root': results_root,
        'prereg': prereg,
        'tissues': [t.upper() for t in args.tissues],
        'alpha': args.alpha,
        'seed': seed,
        'tech_covariates': args.tech_covariates,
        'covariate_variants': args.covariate_variants,
        'min_fraction': args.min_fraction,
        'min_nonzero': args.min_nonzero,
    }

    logger.info("=" * 70)
    logger.info("STAGE 10: AIM-2 ANALYSIS (per-cell-type DE -> positive-control comparison)")
    logger.info("=" * 70)
    logger.info(f"  scope = {'tissues ' + ','.join(ctx['tissues']) if ctx['tissues'] else 'ALL tissues (whole experiment)'}")
    logger.info(f"  de_dir = {de_dir}")
    logger.info(f"  hotspots = {hotspots}")
    logger.info(f"  prereg = {prereg}")
    logger.info(f"  IHW seed = {seed}")
    logger.info(f"  alpha = {args.alpha}")
    logger.info(f"  technical covariates = {', '.join(ctx['tech_covariates']) or 'none'}")
    if ctx['covariate_variants']:
        logger.info(f"  covariate reruns = {', '.join(ctx['covariate_variants'])}")
    if ctx['tissues']:
        logger.warning("  NOTE: tissue subset given -- global IHW/repfdr and the comparison are still "
                       "experiment-wide; a subset DE changes the global FDR pool (use for debugging only).")
    if ctx['tech_covariates']:
        logger.warning(f"  NOTE: --tech-covariates given -- {de_dir.name}/ will hold a covariate-adjusted DE, not "
                       "the headline; de_methods.tsv records the covariates.")

    errors = validate_inputs(ctx, args.from_step)
    if errors:
        for e in errors:
            (logger.warning if args.dry_run else logger.error)(e)
        if not args.dry_run:
            sys.exit(1)

    env = de_env(ctx)
    steps = [s for s in build_steps(ctx) if s['num'] >= args.from_step] + covariate_steps(ctx)
    t_total = time.time()
    for step in steps:
        if not run_step(step, dict(env, **step.get('env', {})), args.dry_run):
            logger.error(f"Stage 10 aborted at step {step['num']}")
            sys.exit(1)
        if step['num'] == 1:
            for lib in args.sensitivities:      # registered §6 sensitivities, written beside the primary DE
                out = Path(str(de_dir) + f'_minlib{lib}')
                sens = {'num': 1, 'name': f'DE sensitivity: median library >= {lib}',
                        'desc': f'run_pseudobulk_de.sh -> {out.name}/', 'cmd': step['cmd'],
                        'key': out / 'de_summary.tsv'}
                if not run_step(sens, dict(env, OUT_DIR=str(out), DE_MIN_MEDIAN_LIBSIZE=str(lib)), args.dry_run):
                    logger.error(f"Stage 10 aborted at the minlib {lib} sensitivity")
                    sys.exit(1)

    logger.info("=" * 70)
    logger.info(f"STAGE 10 COMPLETE -- {len(steps)} step(s) in {time.time() - t_total:.1f}s")
    if not args.dry_run:
        if args.from_step <= 2:
            logger.info(f"  DE:       {de_dir / 'de_summary.tsv'}")
            logger.info(f"  verdict:  {de_dir / 'posctrl_summary.md'}")
        if ctx['covariate_variants']:
            logger.info(f"  covariates: {de_dir / 'de_covariate_robustness.tsv'}")
    logger.info("=" * 70)


if __name__ == '__main__':
    main()
