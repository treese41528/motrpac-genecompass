#!/usr/bin/env python3
"""
run_stage14.py -- Stage 14 Orchestrator: Hardening (per-tissue purity sweeps, DE robustness)

The reviewer-facing hardening layer, downstream of Stage 8 (deconvolution references + pred_z)
and Stage 10 (per-cell-type DE). Quantifies the confidence of the dominant-parenchyma Z per
tissue and shows the exercise signal is robust to composition and RNA-quality confounds.

Chain:
  1. Per-tissue expression-Z purity sweeps  (build->BayesPrism->extract Z->Pearson-VST score)
       holdout mixtures sweeping the focal parenchyma 10-95% purity, split from each tissue's
       PRODUCTION reference_dir in deconvolution/tissue_references.yaml (its `purity_sweep:` entries;
       a reference shared by SKMGN/SKMVL is swept once). Each sweep records sweep_spec.json; an existing
       sweep built from a different reference is refused (use --force or a fresh validation_dir).
       -> <deconvolution.validation_dir>/SWEEP_<tag>_holdout/scores/purity_sweep_summary.tsv
       (liver is the paper-faithful original, run via deconvolution/run_purity_sweep.sh)
  2. Composition-confound table              (py; deconvolution/build_composition_confound_table.py)
       per-hotspot: expression dose signal vs the block's theta trend -> PASS/FLAG/QUIET.
       -> .../pseudobulk_de/composition_confound_table.tsv
  3. RIN / %-globin technical-covariate robustness (R; deconvolution/R/de_technical_covariate_robustness.sh)
       re-fit the dose slope +/- RIN + pct_globin (from MotrpacRatTraining6moData::TRNSCRPT_META).
       -> .../pseudobulk_de/rin_globin_robustness.tsv

Step 1 loops per tissue and SKIPS a tissue whose sweep summary already exists (idempotent; --force
re-runs). The heavy per-tissue parallel path is slurm/analysis/run_purity_sweep_{multi,ext}.slurm
(array jobs); this orchestrator is the reproducible sequential entry point. blood + WAT-SC are
deliberately EXCLUDED (no capturable dominant parenchyma -- their theta-dominant type is a
deconvolution artifact with too few reference cells).

Usage:
  python pipeline/run_stage14.py                        # full: 11 sweeps (10 tissues) + both DE checks
  python pipeline/run_stage14.py --from 2               # just the two DE checks (no BayesPrism)
  python pipeline/run_stage14.py --tissues heart kidney # purity-sweep subset
  python pipeline/run_stage14.py --dry-run

HPC: step 1 is CPU-heavy (BayesPrism run.prism, ~16 cores/tissue) -- run inside an sbatch compute
node. Steps 2-3 are light (py / limma; login-node OK). Full coverage; never subsamples.
"""
import argparse
import json
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
R = DECONV / 'R'
PY = os.environ.get('DECONV_PYTHON') or sys.executable
TISSUE_REFS = DECONV / 'tissue_references.yaml'


def purity_configs(path: Path = TISSUE_REFS) -> list:
    """Per-tissue holdout purity-sweep configs DERIVED from tissue_references.yaml: every runnable tissue with
    a `purity_sweep:` list is swept on its PRODUCTION reference_dir (the built reference Stage 8 deconvolves
    against, whatever its builder: corpus, joint labels, pooled lung, deposited labels). A reference shared
    by two tissues (SKMGN/SKMVL muscle) is swept once. Focal = the parenchyma label in that reference."""
    import yaml
    spec = yaml.safe_load(open(path))
    out, seen = [], set()
    for code, cfg in spec.items():
        if not isinstance(cfg, dict) or not cfg.get('purity_sweep') or not cfg.get('reference_dir'):
            continue
        if cfg.get('status') in ('blocked', 'needs-online-data'):
            continue
        for sw in cfg['purity_sweep']:
            key = (cfg['reference_dir'], sw['focal'])
            if key in seen:
                continue
            seen.add(key)
            out.append(dict(sweep=f"SWEEP_{sw['tag']}_holdout", code=code, tissue=code.lower(),
                            ref_dir=cfg['reference_dir'], focal=sw['focal'],
                            cpm=int(sw.get('cells_per_mixture', 1000))))
    return out


GRID = '0.1,0.3,0.5,0.7,0.85,0.95'


def sweep_key(t: str) -> str:
    """map a --tissues token to the sweep-config match (tissue/focal keyword)."""
    return t.strip().lower()


def build_steps(ctx: dict, subset: list) -> list:
    valdir, de_dir = ctx['valdir'], ctx['de_dir']
    want = {sweep_key(t) for t in subset}
    # A token that matches nothing must be an ERROR, not a silent no-op: the tags come from the
    # sweep NAME (SWEEP_hippoc_holdout -> "hippoc"), so a plausible-looking "hippocampus" matches
    # zero sweeps and the run exits 0 having done nothing -- which reads exactly like success.
    purity = purity_configs()
    tags = [c['sweep'].replace('SWEEP_', '').replace('_holdout', '') for c in purity]
    unmatched = [w for w in want if not any(w == t or w in t for t in tags)]
    if unmatched:
        raise SystemExit(
            f"--tissues matched no purity sweep: {sorted(unmatched)}\n"
            f"  known sweeps: {', '.join(sorted(tags))}")
    items = []
    for cfg in purity:
        tag = cfg['sweep'].replace('SWEEP_', '').replace('_holdout', '')
        if want and not (tag in want or any(w in tag for w in want)):
            continue
        S = valdir / cfg['sweep']
        build = [PY, str(DECONV / 'make_purity_sweep.py'), '--mode', 'holdout',
                 '--ref-source', str(_PROJECT_ROOT / cfg['ref_dir']),
                 '--tissue', cfg['tissue'], '--focal-type', cfg['focal'],
                 '--out', str(S), '--purity-grid', GRID, '--reps', '10',
                 '--cells-per-mixture', str(cfg['cpm'])]
        items.append({
            'label': tag, 'out': S / 'scores' / 'purity_sweep_summary.tsv',
            'spec': S / 'sweep_spec.json',
            'spec_data': {'reference_dir': cfg['ref_dir'], 'focal': cfg['focal'], 'cells_per_mixture': cfg['cpm'],
                          'purity_grid': GRID, 'reps': 10},
            'cmds': [
                build,
                ['bash', str(R / 'run_deconvolution.sh'), str(S / 'reference'), str(S / 'mixtures'), str(S / 'results')],
                ['bash', str(R / 'extract_z.sh'), str(S / 'results' / 'bp_result.rds'), str(S / 'results')],
                ['bash', str(R / 'score_z_vst.sh'), str(S), cfg['focal']],
                [PY, str(DECONV / 'score_purity_sweep.py'), '--stage-dir', str(S)],
            ],
        })
    return [
        {'num': 1, 'name': 'Per-tissue expression-Z purity sweeps (BayesPrism)',
         'desc': 'make_purity_sweep -> run_deconvolution -> extract_z -> Pearson-VST score, per tissue',
         'items': items, 'key': valdir},
        {'num': 2, 'name': 'Composition-confound table (py)',
         'desc': 'build_composition_confound_table.py -> pseudobulk_de/composition_confound_table.tsv',
         'cmd': [PY, str(DECONV / 'build_composition_confound_table.py'), '--de-dir', str(de_dir)],
         'key': de_dir / 'composition_confound_table.tsv'},
        {'num': 3, 'name': 'RIN / %-globin technical-covariate robustness (R)',
         'desc': 'de_technical_covariate_robustness.sh -> pseudobulk_de/rin_globin_robustness.tsv',
         'cmd': ['bash', str(R / 'de_technical_covariate_robustness.sh')],
         'key': de_dir / 'rin_globin_robustness.tsv'},
    ]


def _spec_of(it: dict):
    try:
        return json.loads(it['spec'].read_text())
    except (OSError, ValueError):
        return None


def run_step(step: dict, env: dict, dry_run: bool, force: bool) -> bool:
    logger.info("=" * 70)
    logger.info(f"STEP {step['num']}: {step['name']}")
    logger.info(f"  {step['desc']}")
    logger.info("=" * 70)
    items = step.get('items')
    if items is not None:
        stale = [it for it in items if not force and it['out'].exists() and _spec_of(it) != it['spec_data']]
        if stale:
            say = logger.warning if dry_run else logger.error
            for it in stale:
                say(f"  {it['label']}: existing sweep in {it['out'].parents[1]} was not built from "
                             f"{it['spec_data']['reference_dir']} (sweep_spec.json missing or different)")
            say("  -> rerun with --force, or point deconvolution.validation_dir at a fresh folder")
            if not dry_run:
                return False
        todo = [it for it in items if force or not it['out'].exists()]
        logger.info(f"  {len(items)} tissues; {len(todo)} to run, {len(items) - len(todo)} already present"
                    + (" (--force)" if force else ""))
        for it in todo:
            logger.info(f"  -> {it['label']}")
            for cmd in it['cmds']:
                logger.info(f"     $ {' '.join(cmd)}")
                if dry_run:
                    continue
                t0 = time.time()
                if subprocess.run(cmd, cwd=str(_PROJECT_ROOT), env=env).returncode != 0:
                    logger.error(f"  {it['label']} FAILED at: {' '.join(cmd)}")
                    return False
            if not dry_run:
                it['spec'].write_text(json.dumps(it['spec_data'], indent=2) + "\n")
                logger.info(f"     {it['label']}: {'OK' if it['out'].exists() else 'summary MISSING'}")
        return True
    logger.info(f"  $ {' '.join(step['cmd'])}")
    if dry_run:
        logger.info("  [dry-run] not executed")
        return True
    t0 = time.time()
    if subprocess.run(step['cmd'], cwd=str(_PROJECT_ROOT), env=env).returncode != 0:
        logger.error(f"Step {step['num']} FAILED after {time.time() - t0:.1f}s")
        return False
    logger.info(f"  output: {step['key']} [{'OK' if step['key'].exists() else 'MISSING'}]")
    logger.info(f"Step {step['num']} completed in {time.time() - t0:.1f}s")
    return True


def main() -> None:
    p = argparse.ArgumentParser(
        description='Stage 14: Hardening (purity sweeps, composition-confound, RIN/globin robustness)',
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument('--tissues', nargs='*', default=[], help='purity-sweep subset (e.g. heart kidney lung)')
    p.add_argument('--from', dest='from_step', type=int, default=1, choices=[1, 2, 3])
    p.add_argument('--only', type=int, default=None, choices=[1, 2, 3], help='run just this step')
    p.add_argument('--force', action='store_true', help='re-run a tissue sweep even if its summary exists')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('-v', '--verbose', action='store_true')
    args = p.parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    config = load_config()
    d = config['deconvolution']
    gc_input = resolve_path(config, d['genecompass_input_dir'])
    ctx = {'valdir': resolve_path(config, d['validation_dir']),
           'de_dir': gc_input / 'pseudobulk_de'}

    logger.info("=" * 70)
    logger.info("STAGE 14: HARDENING (purity sweeps -> composition-confound -> RIN/globin robustness)")
    logger.info("=" * 70)
    logger.info(f"  validation dir = {ctx['valdir']}")
    logger.info(f"  DE dir = {ctx['de_dir']}")

    errors = []
    if (args.only in (2, 3) or (args.only is None and args.from_step >= 2)) \
            and not (ctx['de_dir'] / 'de_summary.tsv').exists():
        errors.append(f"de_summary.tsv not found under {ctx['de_dir']}\n  -> run Stage 10 first")
    if errors:
        for e in errors:
            (logger.warning if args.dry_run else logger.error)(e)
        if not args.dry_run:
            sys.exit(1)

    # DE_DIR: the RIN/globin R script's DE folder, passed explicitly (see de_technical_covariate_robustness.R).
    env = dict(os.environ, PIPELINE_ROOT=str(_PROJECT_ROOT), DE_DIR=str(ctx['de_dir']))
    env.setdefault('DECONV_PYTHON', PY)   # the R/bash wrappers read the config with this interpreter
    steps = build_steps(ctx, args.tissues)
    steps = [s for s in steps if (s['num'] == args.only if args.only else s['num'] >= args.from_step)]
    t_total = time.time()
    for step in steps:
        if not run_step(step, env, args.dry_run, args.force):
            logger.error(f"Stage 14 aborted at step {step['num']}")
            sys.exit(1)
    logger.info("=" * 70)
    logger.info(f"STAGE 14 COMPLETE -- {len(steps)} step(s) in {time.time() - t_total:.1f}s")
    logger.info("=" * 70)


if __name__ == '__main__':
    main()
