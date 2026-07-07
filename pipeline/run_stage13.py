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
       -> data/deconvolution/grn/<tissue_celltype>_pooled.tsv
  2. Differential GRN in HUMAN space        (GPU; finetune/genecompass/build_grn_human.py per cell type)
       same perturbation on the transferred pseudo-cells (human ortholog token, species=0).
       -> data/deconvolution/grn_human/<tissue_celltype>_pooled.tsv
  3. Cross-species conservation             (py; finetune/genecompass/compare_grn_conservation.py)
       per-TF rat-vs-human target-list rank correlation + pre-registered anchor edges.
       -> data/deconvolution/grn_human/conservation/{conservation_per_tf,per_celltype,anchors}.tsv
  4. Pathway / gene-set enrichment          (R;  deconvolution/R/run_gene_set_enrichment.sh)
       fgsea (signed dose statistic) vs MSigDB Hallmark + Reactome over the DE blocks.
       -> data/deconvolution/genecompass_input/pseudobulk_de/enrichment/enrichment_summary.tsv

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

HPC: steps 1-2 need a GPU (the frozen model); run this orchestrator inside a GPU sbatch, or use the
array jobs for parallelism. Steps 3-4 are CPU (py / R). Full coverage; never subsamples.
"""
import argparse
import csv
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

GC = _PROJECT_ROOT / 'finetune' / 'genecompass'
DECONV = _PROJECT_ROOT / 'deconvolution'
PY = os.environ.get('DECONV_PYTHON') or sys.executable
THETA_MIN = 0.01


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
         'desc': 'build_grn_pooled.py over abundant hotspots -> data/deconvolution/grn/',
         'items': grn_items('build_grn_pooled.py', ctx['gc_input'], ctx['grn_dir']),
         'key': ctx['grn_dir']},
        {'num': 2, 'name': 'Differential GRN in human space (GPU, per cell type)',
         'desc': 'build_grn_human.py over abundant hotspots -> data/deconvolution/grn_human/',
         'items': grn_items('build_grn_human.py', ctx['gc_input_human'], ctx['grn_human_dir']),
         'key': ctx['grn_human_dir']},
        {'num': 3, 'name': 'Cross-species conservation (py)',
         'desc': 'compare_grn_conservation.py -> grn_human/conservation/',
         'cmd': [PY, str(GC / 'compare_grn_conservation.py'),
                 '--rat-dir', str(ctx['grn_dir']), '--human-dir', str(ctx['grn_human_dir']),
                 '--out-dir', str(ctx['grn_human_dir'] / 'conservation')],
         'key': ctx['grn_human_dir'] / 'conservation' / 'conservation_per_celltype.tsv'},
        {'num': 4, 'name': 'Pathway / gene-set enrichment (R)',
         'desc': 'run_gene_set_enrichment.sh (fgsea vs MSigDB Hallmark/Reactome) -> pseudobulk_de/enrichment/',
         'cmd': ['bash', str(DECONV / 'R' / 'run_gene_set_enrichment.sh')],
         'key': ctx['de_dir'] / 'enrichment' / 'enrichment_summary.tsv'},
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
    p.add_argument('--cell-types', nargs='*', default=[],
                   help="debug subset, e.g. 'SKMVL:Skeletal muscle' or a tissue name (GRN steps only)")
    p.add_argument('--from', dest='from_step', type=int, default=1, choices=[1, 2, 3, 4])
    p.add_argument('--only', type=int, default=None, choices=[1, 2, 3, 4], help='run just this step')
    p.add_argument('--force', action='store_true', help='re-run cell types even if output exists')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('-v', '--verbose', action='store_true')
    args = p.parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    config = load_config()
    d = config['deconvolution']
    gc_input = resolve_path(config, d['genecompass_input_dir'])
    models_dir = resolve_path(config, config['paths']['models_dir'])
    ctx = {
        'gc_input': gc_input,
        'gc_input_human': Path(str(gc_input) + '_human'),
        'augur_dir': gc_input.parent / 'augur_input',
        'grn_dir': gc_input.parent / 'grn',
        'grn_human_dir': gc_input.parent / 'grn_human',
        'de_dir': gc_input / 'pseudobulk_de',
        'model': models_dir / 'rat_genecompass_finetuned' / 'models' / 'rat_phase2_mixed_species' / 'models',
    }
    ctx['hotspots'] = abundant_hotspots(ctx['de_dir'], args.cell_types)

    logger.info("=" * 70)
    logger.info("STAGE 13: MECHANISM (differential GRN -> conservation -> enrichment)")
    logger.info("=" * 70)
    logger.info(f"  abundant hotspots (theta_bar>={THETA_MIN}): {len(ctx['hotspots'])} cell types"
                + (f" [subset {args.cell_types}]" if args.cell_types else ""))
    logger.info(f"  model = {ctx['model']}")

    errors = []
    if (args.only in (1, 2) or (args.only is None and args.from_step <= 2)):
        if not ctx['model'].exists():
            errors.append(f"fine-tuned model not found: {ctx['model']}\n  -> run Stage 7, or set paths.models_dir")
        if not ctx['hotspots']:
            errors.append(f"no abundant hotspots from {ctx['de_dir'] / 'de_hotspots.tsv'}\n  -> run Stage 10 first")
    if (args.only == 2 or (args.only is None and args.from_step <= 2)) and not ctx['gc_input_human'].exists():
        errors.append(f"human-space datasets not found: {ctx['gc_input_human']}\n  -> run Stage 12 transfer first")
    if errors:
        for e in errors:
            (logger.warning if args.dry_run else logger.error)(e)
        if not args.dry_run:
            sys.exit(1)

    env = dict(os.environ, PIPELINE_ROOT=str(_PROJECT_ROOT))
    steps = build_steps(ctx)
    steps = [s for s in steps if (s['num'] == args.only if args.only else s['num'] >= args.from_step)]
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
