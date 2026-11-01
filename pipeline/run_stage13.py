#!/usr/bin/env python3
"""
run_stage13.py -- Stage 13 Orchestrator: Mechanism (model-driven regulatory network,
cross-species conservation, pathway enrichment)

The Aim-2b / Aim-3b mechanistic layer, downstream of Stage 9 (embeddings) and Stage 12
(human-space transfer). A WHOLE-EXPERIMENT driver: it loops the validated in-silico
perturbation engine over every abundant hotspot cell type (theta_bar >= 0.01, read from
the Stage-10 de_hotspots.tsv), in both rat and transferred-human embedding space, then
scores cross-species conservation and pathway enrichment.

Chain:
  1. Differential GRN in RAT space         (GPU; finetune/genecompass/build_grn_pooled.py per cell type)
       dose-pooled bootstrap trained-vs-control TF->target network, per abundant hotspot.
       -> <grn_dir>/<tissue_celltype>_pooled.tsv
  2. Differential GRN in HUMAN space        (GPU; finetune/genecompass/build_grn_human.py per cell type)
       same perturbation on the transferred pseudo-cells (human ortholog token, species=0).
       -> <grn_human_dir>/<tissue_celltype>_pooled.tsv
  3. Cross-species conservation             (py; finetune/genecompass/compare_grn_conservation.py)
       per-TF rat-vs-human target-list rank correlation + pre-registered anchor edges.
       -> <grn_human_dir>/conservation/{conservation_per_tf,per_celltype,anchors}.tsv
  4. Pathway / gene-set enrichment          (R;  deconvolution/R/run_gene_set_enrichment.sh)
       fgsea (signed dose statistic) vs MSigDB Hallmark + Reactome over the DE blocks.
       -> <genecompass_input_dir>/pseudobulk_de/enrichment/enrichment_summary.tsv
  5. Parenchyma data-anchored validation    (py; deconvolution/validate_parenchyma_dataanchored.py)
       do the parenchyma DE blocks recover genes that move in the matched bulk? + dominant cell type per tissue.
       -> <genecompass_input_dir>/pseudobulk_de/{parenchyma_dataanchored_validation,dominant_celltype_flags}.tsv

GRN reuse (--reuse-grn-from PROFILE). GRNs exist only for hotspots, and steps 1-2 cost GPU hours. Given a
previous arm's profile (e.g. config/profiles/refs4_14.yaml), steps 1-2 compare that arm's DE hotspot set
(is_hotspot == TRUE in its pseudobulk_de/de_hotspots.tsv, keyed tissue + cell type) with this arm's: if the
sets are identical its *_pooled.tsv networks are copied (or, with --link, symlinked -- how the 16-tissue
default tree was assembled) into this arm's grn_dir / grn_human_dir; otherwise steps 1-2 are run. Without
the option, steps 1-2 always build the networks (the clean path).

Folders and the model come from the deconvolution section of the config (grn_dir, grn_human_dir,
augur_input_dir, genecompass_input_dir, model_checkpoint -- else the latest checkpoint under
genecompass_model_dir with a warning, as in Stages 9 and 12), so a profile such as config/profiles/retrain_v2_downstream.yaml gives each model its own set.

Steps 1-2 loop per cell type and SKIP a cell type whose output already exists (idempotent;
--force re-runs). The heavy per-cell-type parallel path is slurm/analysis/build_grn_production.slurm
+ build_grn_human.slurm (array jobs); this orchestrator is the reproducible sequential entry point.

Usage:
  python pipeline/run_stage13.py                       # full: all abundant hotspots
  python pipeline/run_stage13.py --from 3              # re-run only conservation + enrichment
  python pipeline/run_stage13.py --only 4              # just enrichment
  python pipeline/run_stage13.py --cell-types "SKMVL:Skeletal muscle"   # debug subset
  python pipeline/run_stage13.py --dry-run
  python pipeline/run_stage13.py --force               # re-run cell types even if output exists
  python pipeline/run_stage13.py --reuse-grn-from config/profiles/refs4_14.yaml   # reuse GRNs if hotspots match

HPC: steps 1-2 need a GPU (the frozen model); run this orchestrator inside a GPU sbatch, or use the
array jobs for parallelism. Steps 3-4 are CPU (py / R). Full coverage; never subsamples.
"""
import argparse
import csv
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(os.environ.get('PIPELINE_ROOT') or Path(__file__).resolve().parents[1])
os.environ.setdefault('PIPELINE_ROOT', str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / 'lib'))
from gene_utils import load_config, resolve_path  # noqa: E402
sys.path.insert(0, str(_PROJECT_ROOT / 'deconvolution'))
from model_checkpoint import latest_checkpoint, resolve_model_dir as resolve_model_from_config  # noqa: E402

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

GC = _PROJECT_ROOT / 'finetune' / 'genecompass'
DECONV = _PROJECT_ROOT / 'deconvolution'
PY = os.environ.get('DECONV_PYTHON') or sys.executable
THETA_MIN = 0.01


def resolve_model_dir(base: Path) -> Path:
    """Accept a model dir (has config.json) or a parent holding checkpoint-*; pick the latest.
    (Kept for callers/tests; the config path goes through deconvolution/model_checkpoint.py.)"""
    return latest_checkpoint(base)


def _safe(tissue: str, ct: str) -> str:
    s = ct.lower().replace('+', 'pos').replace(' ', '_').replace('-', '_').replace('/', '_')
    while '__' in s:
        s = s.replace('__', '_')
    return f"{tissue}_{s}"


def abundant_hotspots(de_dir: Path, subset: list) -> list:
    """Read Stage-10 de_hotspots.tsv -> [(tissue_lower, cell_type, outname)] for hotspots with
    mean_fraction >= THETA_MIN. `subset` (list of 'TISSUE:cell type') restricts for debugging."""
    hs = de_dir / 'de_hotspots.tsv'
    if not hs.exists():
        return []
    want = {s.strip().upper() for s in subset}
    out = []
    for r in csv.DictReader(open(hs), delimiter='\t'):
        if r.get('is_hotspot', '').strip().upper() != 'TRUE':
            continue
        try:
            if float(r['mean_fraction']) < THETA_MIN:
                continue
        except (ValueError, KeyError):
            continue
        tissue, ct = r['tissue'].strip().lower(), r['cell_type'].strip()
        if want and f"{tissue.upper()}:{ct.upper()}" not in want and tissue.upper() not in want:
            continue
        out.append((tissue, ct, _safe(tissue, ct)))
    return out


def hotspot_set(de_dir: Path) -> set:
    """(TISSUE, cell type) of every is_hotspot == TRUE row of <de_dir>/de_hotspots.tsv (no abundance filter;
    the comparison the GRN-reuse decision is made on)."""
    hs = Path(de_dir) / 'de_hotspots.tsv'
    if not hs.exists():
        raise SystemExit(f"de_hotspots.tsv not found: {hs}")
    return {(r['tissue'].strip(), r['cell_type'].strip()) for r in csv.DictReader(open(hs), delimiter='\t')
            if r.get('is_hotspot', '').strip().upper() == 'TRUE'}


def profile_dirs(profile: str) -> dict:
    """de_dir / grn_dir / grn_human_dir of another config profile (PIPELINE_PROFILE semantics)."""
    old = os.environ.get('PIPELINE_PROFILE')
    os.environ['PIPELINE_PROFILE'] = str(profile)
    try:
        c = load_config()
    finally:
        if old is None:
            os.environ.pop('PIPELINE_PROFILE', None)
        else:
            os.environ['PIPELINE_PROFILE'] = old
    d = c['deconvolution']
    return {'de_dir': resolve_path(c, d['genecompass_input_dir']) / 'pseudobulk_de',
            'grn_dir': resolve_path(c, d['grn_dir']), 'grn_human_dir': resolve_path(c, d['grn_human_dir'])}


def reuse_grns(ctx: dict, profile: str, link: bool, dry_run: bool) -> bool:
    """Copy/link a previous arm's networks if its hotspot set equals this arm's. True = reused (skip 1-2)."""
    src = profile_dirs(profile)
    mine, theirs = hotspot_set(ctx['de_dir']), hotspot_set(src['de_dir'])
    if mine != theirs:
        logger.info(f"  hotspot set differs from {profile} (+{sorted(mine - theirs)} -{sorted(theirs - mine)}): "
                    f"building the networks (steps 1-2)")
        return False
    logger.info(f"  hotspot set identical to {profile} ({len(mine)} hotspots): "
                f"{'linking' if link else 'copying'} its networks")
    for a, b in ((src['grn_dir'], ctx['grn_dir']), (src['grn_human_dir'], ctx['grn_human_dir'])):
        files = sorted(Path(a).glob('*_pooled.tsv'))
        if not files:
            raise SystemExit(f"--reuse-grn-from: no *_pooled.tsv under {a}")
        logger.info(f"    {len(files)} networks {a} -> {b}")
        if dry_run:
            continue
        Path(b).mkdir(parents=True, exist_ok=True)
        for f in files:
            dst = Path(b) / f.name
            if dst.exists() or dst.is_symlink():
                continue
            if link:
                dst.symlink_to(f.resolve())
            else:
                shutil.copy2(f.resolve(), dst)
    return True


def build_steps(ctx: dict) -> list:
    hots = ctx['hotspots']
    model = ['--model-dir', str(ctx['model'])]

    def grn_items(script, dataset_root, out_root):
        items = []
        for tissue, ct, name in hots:
            out = out_root / f"{name}_pooled.tsv"
            items.append({
                'label': f"{tissue}/{ct}",
                'out': out,
                'cmd': [PY, str(GC / script)] + model + [
                    '--dataset', str(dataset_root / tissue / 'dataset'),
                    '--meta', str(ctx['augur_dir'] / tissue / 'meta.tsv'),
                    '--cell-type', ct,
                    '--out', str(out),
                    '--bootstrap', '30', '--seed', '0', '--device', 'cuda'],
            })
        return items

    return [
        {'num': 1, 'name': 'Differential GRN in rat space (GPU, per cell type)',
         'desc': f"build_grn_pooled.py over abundant hotspots -> {ctx['grn_dir']}/",
         'items': grn_items('build_grn_pooled.py', ctx['gc_input'], ctx['grn_dir']),
         'key': ctx['grn_dir']},
        {'num': 2, 'name': 'Differential GRN in human space (GPU, per cell type)',
         'desc': f"build_grn_human.py over abundant hotspots -> {ctx['grn_human_dir']}/",
         'items': grn_items('build_grn_human.py', ctx['gc_input_human'], ctx['grn_human_dir']),
         'key': ctx['grn_human_dir']},
        {'num': 3, 'name': 'Cross-species conservation (py)',
         'desc': f"compare_grn_conservation.py -> {ctx['grn_human_dir']}/conservation/",
         'cmd': [PY, str(GC / 'compare_grn_conservation.py'),
                 '--rat-dir', str(ctx['grn_dir']), '--human-dir', str(ctx['grn_human_dir']),
                 '--out-dir', str(ctx['grn_human_dir'] / 'conservation')],
         'key': ctx['grn_human_dir'] / 'conservation' / 'conservation_per_celltype.tsv'},
        {'num': 4, 'name': 'Pathway / gene-set enrichment (R)',
         'desc': 'run_gene_set_enrichment.sh (fgsea vs MSigDB Hallmark/Reactome) -> pseudobulk_de/enrichment/',
         'cmd': ['bash', str(DECONV / 'R' / 'run_gene_set_enrichment.sh')],
         'key': ctx['de_dir'] / 'enrichment' / 'enrichment_summary.tsv'},
        {'num': 5, 'name': 'Parenchyma data-anchored validation + dominant cell types (py)',
         'desc': 'validate_parenchyma_dataanchored.py -> pseudobulk_de/parenchyma_dataanchored_validation.tsv',
         'cmd': [PY, str(DECONV / 'validate_parenchyma_dataanchored.py')],
         'key': ctx['de_dir'] / 'parenchyma_dataanchored_validation.tsv'},
    ]


def run_step(step: dict, env: dict, dry_run: bool, force: bool) -> bool:
    logger.info("=" * 70)
    logger.info(f"STEP {step['num']}: {step['name']}")
    logger.info(f"  {step['desc']}")
    logger.info("=" * 70)
    items = step.get('items')
    if items is not None:
        todo = [it for it in items if force or not it['out'].exists()]
        logger.info(f"  {len(items)} cell types; {len(todo)} to run, {len(items) - len(todo)} already present"
                    + (" (--force)" if force else ""))
        for it in todo:
            logger.info(f"  -> {it['label']}")
            logger.info(f"     $ {' '.join(it['cmd'])}")
            if dry_run:
                continue
            t0 = time.time()
            if subprocess.run(it['cmd'], cwd=str(_PROJECT_ROOT), env=env).returncode != 0:
                logger.error(f"  cell type {it['label']} FAILED after {time.time() - t0:.1f}s")
                return False
            logger.info(f"     done in {time.time() - t0:.1f}s")
        return True
    logger.info(f"  $ {' '.join(step['cmd'])}")
    if dry_run:
        logger.info("  [dry-run] not executed")
        return True
    t0 = time.time()
    if subprocess.run(step['cmd'], cwd=str(_PROJECT_ROOT), env=env).returncode != 0:
        logger.error(f"Step {step['num']} FAILED after {time.time() - t0:.1f}s")
        return False
    key = step['key']
    logger.info(f"  output: {key} [{'OK' if key.exists() else 'MISSING'}]")
    logger.info(f"Step {step['num']} completed in {time.time() - t0:.1f}s")
    return True


def main() -> None:
    p = argparse.ArgumentParser(
        description='Stage 13: Mechanism (differential GRN, cross-species conservation, enrichment)',
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument('--model-dir', default=None,
                   help='fine-tuned rat GeneCompass dir or a parent with checkpoint-* (default: '
                        'deconvolution.model_checkpoint; else latest checkpoint, with a warning)')
    p.add_argument('--cell-types', nargs='*', default=[],
                   help="debug subset, e.g. 'SKMVL:Skeletal muscle' or a tissue name (GRN steps only)")
    p.add_argument('--from', dest='from_step', type=int, default=1, choices=[1, 2, 3, 4, 5])
    p.add_argument('--only', type=int, default=None, choices=[1, 2, 3, 4, 5], help='run just this step')
    p.add_argument('--reuse-grn-from', default=None, metavar='PROFILE',
                   help='reuse that profile\'s GRNs when its DE hotspot set equals this arm\'s (see above)')
    p.add_argument('--link', action='store_true',
                   help='with --reuse-grn-from: symlink instead of copy (reproduces the assembled default tree)')
    p.add_argument('--force', action='store_true', help='re-run cell types even if output exists')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('-v', '--verbose', action='store_true')
    args = p.parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    config = load_config()
    d = config['deconvolution']
    gc_input = resolve_path(config, d['genecompass_input_dir'])
    ctx = {
        'gc_input': gc_input,
        'gc_input_human': Path(str(gc_input) + '_human'),
        'augur_dir': resolve_path(config, d['augur_input_dir']),
        'grn_dir': resolve_path(config, d['grn_dir']),
        'grn_human_dir': resolve_path(config, d['grn_human_dir']),
        'de_dir': gc_input / 'pseudobulk_de',
        'model': resolve_model_from_config(config, args.model_dir, logger),
    }
    ctx['hotspots'] = abundant_hotspots(ctx['de_dir'], args.cell_types)

    logger.info("=" * 70)
    logger.info("STAGE 13: MECHANISM (differential GRN -> conservation -> enrichment)")
    logger.info("=" * 70)
    logger.info(f"  abundant hotspots (theta_bar>={THETA_MIN}): {len(ctx['hotspots'])} cell types"
                + (f" [subset {args.cell_types}]" if args.cell_types else ""))
    logger.info(f"  model = {ctx['model']}")
    logger.info(f"  grn = {ctx['grn_dir']}  grn_human = {ctx['grn_human_dir']}  meta = {ctx['augur_dir']}")

    errors = []
    if (args.only in (1, 2) or (args.only is None and args.from_step <= 2)):
        if not (ctx['model'] / 'config.json').exists():
            errors.append(f"fine-tuned model not found: {ctx['model']}\n"
                          f"  -> run Stage 7, or pass --model-dir / set deconvolution.model_checkpoint")
        if not ctx['hotspots']:
            errors.append(f"no abundant hotspots from {ctx['de_dir'] / 'de_hotspots.tsv'}\n  -> run Stage 10 first")
    if (args.only == 2 or (args.only is None and args.from_step <= 2)) and not ctx['gc_input_human'].exists():
        errors.append(f"human-space datasets not found: {ctx['gc_input_human']}\n  -> run Stage 12 transfer first")
    if errors:
        for e in errors:
            (logger.warning if args.dry_run else logger.error)(e)
        if not args.dry_run:
            sys.exit(1)

    # DE_DIR: the enrichment R script's DE folder, passed explicitly (see run_gene_set_enrichment.R).
    env = dict(os.environ, PIPELINE_ROOT=str(_PROJECT_ROOT), DE_DIR=str(ctx['de_dir']))
    env.setdefault('DECONV_PYTHON', PY)   # the R/bash wrappers read the config with this interpreter
    steps = build_steps(ctx)
    steps = [s for s in steps if (s['num'] == args.only if args.only else s['num'] >= args.from_step)]
    if args.reuse_grn_from and any(s['num'] in (1, 2) for s in steps):
        if reuse_grns(ctx, args.reuse_grn_from, args.link, args.dry_run):
            steps = [s for s in steps if s['num'] not in (1, 2)]
    t_total = time.time()
    for step in steps:
        if not run_step(step, env, args.dry_run, args.force):
            logger.error(f"Stage 13 aborted at step {step['num']}")
            sys.exit(1)
    logger.info("=" * 70)
    logger.info(f"STAGE 13 COMPLETE -- {len(steps)} step(s) in {time.time() - t_total:.1f}s")
    logger.info("=" * 70)


if __name__ == '__main__':
    main()
