#!/usr/bin/env python3
"""
run_stage11.py -- Stage 11 Orchestrator: the Aim-2 gate (which cell types respond to training?)

Whole-experiment driver over Stage 9's embeddings of ALL tissues (the gate's BH spans every block, so run it
only once every tissue is embedded). The statistic and the multiplicity rule are those of the registered gate
(10 splits x 10,000 permutations, pre-registration §6; REFERENCE_RELABEL_PREREG.md §6 analysable-block rule).
From the Step 3 rerun on (docs/audit/remediation/STEP3_PREREG.md section 5) a hotspot is a call that more
permutations could not reverse: after the first 10,000 permutations the analysable blocks whose class is
still open are sampled in further batches of 10,000, up to 1,000,000, and each block is classified hotspot /
borderline / none from the limits of its ideal p (deconvolution/mc_gate.py). The table's is_hotspot column
(gate_class == hotspot) is what Stages 10, 12 and 13 read; borderline blocks are listed beside it. The
10,000-permutation p and its BH q (the previous method) stay in the table as p_sup_trained_10k and
q_sup_trained_10k.

The gate folder is the folder of deconvolution.hotspots_table (default config:
genecompass_input_v2_refs4all_rebuild/gate_rcv10/); the Augur exports go to deconvolution.augur_input_dir.

Chain:
  1. embed_qc.py            per-tissue embedding QC (silhouette, kNN purity, value scale)        -> stdout
  2. pheno_merge_test.py    sex/group/trained eta^2 vs label-permutation null                   -> <gc>/pheno_merge_test.tsv
  3. subspace_probe.py      supervised PLS-1 probe, R splits x P permutations (primary detector) -> <gate>/subspace_probe.tsv
                            --sequential: further batches for the family's open blocks, interval limits and
                            classes (5.2); sensitivities S1 (sex-stratified) and S2 (8-week against control)
                            for every block (5.3). --scorer loop keeps the original permutation loop.
  4. augur_prep.py          Augur exports + PCA-50-of-Z control probe                           -> <augur>/, <gate>/pca_control.tsv
  5. run_augur.sh           canonical Augur (R; RF AUC per cell type)                           -> <gate>/augur_results.tsv
  6. corroborate_summary.py merge + classes -> hotspots (§6 rule: median library >= 1,000), S1-S4 beside every
                            block, the reading rule, the corroboration figures (5.3, 5.4)       -> <gate>/corroboration_merged.tsv
     (+ the registered sensitivities --min-libsize 100 / 0 -> corroboration_merged_minlib{100,0}.tsv; they use
      the 10,000-permutation p and plain BH)
     Each of the three merges writes its own table, <table>.figures.tsv and provenance_<table>.json.

Usage:
  python pipeline/run_stage11.py                 # full gate (CPU; ~hours with --jobs 48)
  python pipeline/run_stage11.py --from 6        # re-merge only
  python pipeline/run_stage11.py --dry-run
Threads: subspace_probe/augur_prep pin BLAS to 1 thread themselves; parallelism is --jobs processes.
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
PY = os.environ.get('DECONV_PYTHON') or sys.executable


def build_steps(ctx: dict) -> list:
    g, gt, ai = ctx['gc'], ctx['gate'], ctx['augur']
    cv = ['--perms', str(ctx['perms']), '--folds', '5', '--repeats', str(ctx['repeats']), '--jobs', str(ctx['jobs'])]
    # For the probe (step 3 of this chain) only: the sequential sampling of STEP3_PREREG section 5.2 and the
    # permutation scorer. The family is the analysable rule, applied by subspace_probe.py to the configured
    # Stage 8 output (its --results-root default).
    seq = ['--sequential', '--scorer', ctx['scorer']]
    sens = [[PY, str(DECONV / 'corroborate_summary.py'), '--gc-root', str(gt), '--min-libsize', str(m),
             '--out', str(gt / f'corroboration_merged_minlib{m}.tsv')] for m in ctx['sensitivities']]
    return [
        {'num': 1, 'name': 'Embedding QC', 'cmds': [[PY, str(DECONV / 'embed_qc.py'), str(g)]], 'key': None},
        {'num': 2, 'name': 'Phenotype-merge test',
         'cmds': [[PY, str(DECONV / 'pheno_merge_test.py'), '--gc-root', str(g),
                   '--out', str(g / 'pheno_merge_test.tsv')]], 'key': g / 'pheno_merge_test.tsv'},
        {'num': 3, 'name': 'Supervised subspace probe (primary detector)',
         'cmds': [[PY, str(DECONV / 'subspace_probe.py'), '--gc-root', str(g)] + cv + seq
                  + ['--out', str(gt / 'subspace_probe.tsv')]], 'key': gt / 'subspace_probe.tsv'},
        {'num': 4, 'name': 'Augur exports + PCA control probe',
         'cmds': [[PY, str(DECONV / 'augur_prep.py'), '--gc-root', str(g), '--pca', '50'] + cv
                  + ['--out', str(ai), '--ctl-out', str(gt / 'pca_control.tsv')]], 'key': gt / 'pca_control.tsv'},
        {'num': 5, 'name': 'Canonical Augur (R)',
         'cmds': [['bash', str(DECONV / 'run_augur.sh'), str(ai), str(gt / 'augur_results.tsv')]],
         'key': gt / 'augur_results.tsv'},
        {'num': 6, 'name': 'Corroborate -> hotspot classes (§6 family, STEP3 5.2-5.4) + registered sensitivities',
         'cmds': [[PY, str(DECONV / 'corroborate_summary.py'), '--gc-root', str(gt)]] + sens,
         'key': gt / 'corroboration_merged.tsv'},
    ]


def main() -> None:
    p = argparse.ArgumentParser(description='Stage 11: Aim-2 gate', epilog=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--perms', type=int, default=10000,
                   help='permutations of the first batch and of every other test (10,000 = the registered gate)')
    p.add_argument('--repeats', type=int, default=10, help='fold splits (10 = the registered gate)')
    p.add_argument('--scorer', choices=['kernel', 'loop'], default='kernel',
                   help="permutation scorer of the trained probe: 'kernel' (fold kernels; reproduces the loop's "
                        "counts, STEP3_PREREG 5.2) or 'loop' (the original loop)")
    p.add_argument('--jobs', type=int, default=8)
    p.add_argument('--sensitivities', type=int, nargs='*', default=[100, 0],
                   help='extra --min-libsize runs of the merge (registered: 100 0; pass none to skip)')
    p.add_argument('--from', dest='from_step', type=int, default=1, choices=range(1, 7))
    p.add_argument('--only', type=int, default=None, choices=range(1, 7))
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()

    config = load_config()
    d = config['deconvolution']
    gc = resolve_path(config, d['genecompass_input_dir'])
    gate = (resolve_path(config, d['hotspots_table']).parent if d.get('hotspots_table') else gc)
    ctx = {'gc': gc, 'gate': gate, 'augur': resolve_path(config, d['augur_input_dir']),
           'perms': args.perms, 'repeats': args.repeats, 'jobs': args.jobs, 'sensitivities': args.sensitivities,
           'scorer': args.scorer}
    logger.info("STAGE 11: AIM-2 GATE")
    logger.info(f"  embeddings = {gc}\n  gate dir   = {gate}\n  augur in   = {ctx['augur']}\n"
                f"  {args.repeats} splits x {args.perms:,} permutations (first batch; sequential to the cap for "
                f"open blocks), scorer={args.scorer}, jobs={args.jobs}")
    missing = sorted(p_.parent.name for p_ in gc.glob('*/pseudocells.h5ad')
                     if not (p_.parent / 'embeddings' / 'cell_embeddings.npy').exists())
    if missing and (args.only or args.from_step) <= 4:
        msg = f"tissues without embeddings (run Stage 9 first): {missing}"
        (logger.warning if args.dry_run else logger.error)(msg)
        if not args.dry_run:
            sys.exit(1)
    if not args.dry_run:
        gate.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ, PIPELINE_ROOT=str(_PROJECT_ROOT))
    env.setdefault('DECONV_PYTHON', PY)
    steps = [s for s in build_steps(ctx) if (s['num'] == args.only if args.only else s['num'] >= args.from_step)]
    for s in steps:
        logger.info("=" * 70)
        logger.info(f"STEP {s['num']}: {s['name']}")
        for cmd in s['cmds']:
            logger.info(f"  $ {' '.join(cmd)}")
            if args.dry_run:
                continue
            t0 = time.time()
            if subprocess.run(cmd, cwd=str(_PROJECT_ROOT), env=env).returncode != 0:
                logger.error(f"Stage 11 aborted at step {s['num']} after {time.time() - t0:.0f}s")
                sys.exit(1)
        if s['key'] is not None and not args.dry_run:
            logger.info(f"  output: {s['key']} [{'OK' if s['key'].exists() else 'MISSING'}]")
    logger.info("STAGE 11 COMPLETE" + (" (dry run)" if args.dry_run else ""))


if __name__ == '__main__':
    main()
