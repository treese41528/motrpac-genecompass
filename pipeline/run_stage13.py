#!/usr/bin/env python3
"""
run_stage13.py -- Stage 13 Orchestrator: Mechanism (model-driven differential network,
its decoy null, rat-human self-consistency, pathway enrichment)

The Aim-2b / Aim-3b mechanistic layer, downstream of Stage 9 (embeddings) and Stage 12
(human-space transfer). A WHOLE-EXPERIMENT driver: it loops the in-silico perturbation
engine over every abundant hotspot cell type (theta_bar >= 0.01, read from the Stage-10
de_hotspots.tsv), in both rat and transferred-human embedding space, then scores the decoy
null, the rat-human self-consistency of the networks, and pathway enrichment.

How to read the outputs. The network and hub tables are model read-outs: the decoy null found no
TF-specific signal in them, and the rat-human score compares the same cells through the same model, so it is
self-consistency, not evidence about cross-species biology. The enrichment (step 4) does not use the model.

Chain:
  1. Differential GRN in RAT space         (GPU; finetune/genecompass/build_grn_pooled.py per cell type)
       dose-pooled bootstrap trained-minus-control TF->target network, per abundant hotspot. An edge exists
       only where the TF and the target are in both arms' profiles (STEP3_PREREG.md 8.1).
       -> <grn_dir>/<tissue_celltype>_pooled.tsv, + _pooled_arms.tsv, _pooled_arms_summary.tsv,
          provenance_<tissue_celltype>_pooled.json
  2. Differential GRN in HUMAN space        (GPU; finetune/genecompass/build_grn_human.py per cell type)
       same perturbation on the transferred pseudo-cells (human ortholog token, species=0).
       -> <grn_human_dir>/<tissue_celltype>_pooled.tsv
  3. Rat-human self-consistency             (py; finetune/genecompass/compare_grn_conservation.py)
       per-TF rank correlation of the rat-space and human-space edge differences; TFs scored on all targets
       for lack of |z|>=2 edges are flagged and counted.
       -> <grn_human_dir>/self_consistency/self_consistency_{per_tf,per_celltype,anchors}.tsv
  4. Pathway / gene-set enrichment          (R;  deconvolution/R/run_gene_set_enrichment.sh)
       fgsea (signed dose statistic on the headline's dose p) vs MSigDB Hallmark + Reactome over the DE
       blocks; rat genes carried to human through one-to-one ortholog pairs (STEP3_PREREG.md 8.2).
       -> <genecompass_input_dir>/pseudobulk_de/enrichment/{enrichment_summary,enrichment_blocks,
          gene_sets_used}.tsv  (or --enrich-out DIR)
  5. Parenchyma data-anchored validation    (py; deconvolution/validate_parenchyma_dataanchored.py)
       do the parenchyma DE blocks recover genes that move in the matched bulk? + dominant cell type per tissue.
       -> <genecompass_input_dir>/pseudobulk_de/{parenchyma_dataanchored_validation,dominant_celltype_flags}.tsv
  6. Decoy-null networks in rat space      (GPU; build_grn_pooled.py --decoy-seed 0 per cell type)
       every expressed TF replaced by a non-TF gene of the nearest expression, same build (GRN_DECOY_NULL_PREREG.md).
       -> <grn_dir>_decoy/<tissue_celltype>_pooled.tsv (+ sidecars, _pooled_decoy_match.tsv)
  7. Decoy-null scores                      (py; deconvolution/grn_decoy_fdr.py)
       edge-level FDR(t) and hub strength against decoys; TF x network pairs and distinct TFs.
       -> <grn_dir>_decoy/fdr/{fdr_per_network,fdr_curve_pooled,hub_vs_decoy}.tsv, summary.txt
     --skip-decoy leaves out steps 6-7 (for example a CPU-only rerun of steps 3-5).

Which cell types get a network. The hotspots of the Stage 11 gate (STEP3_PREREG.md 5.2) with a mean fraction of
at least 0.01. The set is read from de_hotspots.tsv (is_hotspot, which Stage 10 copies from the gate table)
and checked against the gate table itself (deconvolution.hotspots_table): its boolean column is_hotspot when
it has one, q_sup_trained < 0.05 for a gate table written before that column existed. If the two disagree the
network steps stop, because the DE folder was then made from a different gate table.

Edge rule. Every network built here carries provenance_<name>_pooled.json with params.edge_rule. A network
already in the output folder that was not built on shared genes (or has no record) is never skipped as
"already present" and never reused from another arm: the run stops and asks for --force or a new folder.

GRN reuse (--reuse-grn-from PROFILE). GRNs exist only for hotspots, and steps 1-2 cost GPU hours. Given a
previous arm's profile (e.g. config/profiles/refs4_14.yaml), steps 1-2 compare that arm's DE hotspot set
(is_hotspot == TRUE in its pseudobulk_de/de_hotspots.tsv, keyed tissue + cell type) with this arm's: if the
sets are identical AND that arm's networks were built on shared genes, its *_pooled.tsv networks and their
sidecar files are copied (or, with --link, symlinked -- how the 16-tissue default tree was assembled) into
this arm's grn_dir / grn_human_dir; otherwise steps 1-2 are run. Without the option, steps 1-2 always build
the networks (the clean path).

Folders and the model come from the deconvolution section of the config (grn_dir, grn_human_dir,
augur_input_dir, genecompass_input_dir, model_checkpoint -- else the latest checkpoint under
genecompass_model_dir with a warning, as in Stages 9 and 12), so a profile such as config/profiles/retrain_v2_downstream.yaml gives each model its own set.

Steps 1-2 loop per cell type and SKIP a cell type whose output already exists (idempotent;
--force re-runs). The heavy per-cell-type parallel path is slurm/analysis/build_grn_production.slurm
+ build_grn_human.slurm (array jobs); this orchestrator is the reproducible sequential entry point.

Usage:
  python pipeline/run_stage13.py                       # full: all abundant hotspots, with the decoy null
  python pipeline/run_stage13.py --from 3 --skip-decoy # re-run only self-consistency + enrichment (CPU)
  python pipeline/run_stage13.py --only 4              # just enrichment
  python pipeline/run_stage13.py --only 4 --enrich-out DIR   # enrichment into DIR, DE folder left untouched
  python pipeline/run_stage13.py --cell-types "SKMVL:Skeletal muscle"   # debug subset
  python pipeline/run_stage13.py --dry-run
  python pipeline/run_stage13.py --force               # re-run cell types even if output exists
  python pipeline/run_stage13.py --reuse-grn-from config/profiles/refs4_14.yaml   # reuse GRNs if hotspots match

HPC: steps 1-2 and 6 need a GPU (the frozen model); run this orchestrator inside a GPU sbatch, or use the
array jobs for parallelism. Steps 3-5 and 7 are CPU (py / R). Full coverage; never subsamples.
"""
import argparse
import csv
import json
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
# The edge rule the network builders write into each network's provenance record (EDGE_RULE in
# finetune/genecompass/build_grn_pooled.py; tests/test_stage13_step3.py checks the two stay equal).
# STEP3_PREREG.md 8.1: an edge exists only where the TF and the target are in both arms' profiles.
EDGE_RULE = 'shared_genes'
DECOY_SEED = 0          # GRN_DECOY_NULL_PREREG.md: one registered decoy seed


def resolve_model_dir(base: Path) -> Path:
    """Accept a model dir (has config.json) or a parent holding checkpoint-*; pick the latest.
    (Kept for callers/tests; the config path goes through deconvolution/model_checkpoint.py.)"""
    return latest_checkpoint(base)


def _safe(tissue: str, ct: str) -> str:
    s = ct.lower().replace('+', 'pos').replace(' ', '_').replace('-', '_').replace('/', '_')
    while '__' in s:
        s = s.replace('__', '_')
    return f"{tissue}_{s}"


def _flag(v) -> bool:
    """A boolean table cell as written by R (TRUE) or pandas (True)."""
    return str(v).strip().lower() in ('true', 't', '1', 'yes')


def abundant_hotspots(de_dir: Path, subset: list) -> list:
    """Read Stage-10 de_hotspots.tsv -> [(tissue_lower, cell_type, outname)] for hotspots with
    mean_fraction >= THETA_MIN. `subset` (list of 'TISSUE:cell type') restricts for debugging."""
    hs = de_dir / 'de_hotspots.tsv'
    if not hs.exists():
        return []
    want = {s.strip().upper() for s in subset}
    out = []
    for r in csv.DictReader(open(hs), delimiter='\t'):
        if not _flag(r.get('is_hotspot', '')):
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
    # Two cell types must never share a network file name: the second build would overwrite the first.
    seen = {}
    for tissue, ct, name in out:
        if name in seen and seen[name] != (tissue, ct):
            raise SystemExit(f"network file name collision: {seen[name]} and {(tissue, ct)} both map to "
                             f"'{name}_pooled.tsv'")
        seen[name] = (tissue, ct)
    return out


def hotspot_set(de_dir: Path) -> set:
    """(TISSUE, cell type) of every is_hotspot == TRUE row of <de_dir>/de_hotspots.tsv (no abundance filter;
    the comparison the GRN-reuse decision is made on)."""
    hs = Path(de_dir) / 'de_hotspots.tsv'
    if not hs.exists():
        raise SystemExit(f"de_hotspots.tsv not found: {hs}")
    return {(r['tissue'].strip(), r['cell_type'].strip()) for r in csv.DictReader(open(hs), delimiter='\t')
            if _flag(r.get('is_hotspot', ''))}


def gate_hotspots(table: Path):
    """The hotspot set of a Stage 11 gate table -> ({(TISSUE, cell type)}, rule used).

    STEP3_PREREG.md 5.2: a hotspot is a block whose call more permutations could not reverse, and the gate
    table of the rerun marks it in the boolean column is_hotspot. That column is used when the table has it.
    A gate table written before the registration has no such column; its hotspots are q_sup_trained < 0.05,
    as before.
    """
    with open(table) as f:
        reader = csv.DictReader(f, delimiter='\t')
        fields, rows = reader.fieldnames or [], list(reader)
    if 'is_hotspot' in fields:
        return ({(r['tissue'].strip().upper(), r['cell_type'].strip()) for r in rows if _flag(r['is_hotspot'])},
                'is_hotspot column')
    out = set()
    for r in rows:
        try:
            q = float(r.get('q_sup_trained', ''))
        except ValueError:
            continue
        if q < 0.05:                                    # a missing q parses to nan and is not a hotspot
            out.add((r['tissue'].strip().upper(), r['cell_type'].strip()))
    return out, 'q_sup_trained < 0.05'


def hotspot_mismatch(de_dir: Path, gate_table):
    """Compare the DE folder's hotspot set with the gate table's. Returns (message or None, description).

    The networks are built for the gate's hotspots, but the list comes through de_hotspots.tsv (it also
    holds the mean fraction). If that file was made from a different gate table, building from it would
    give networks for the wrong blocks, so a difference is reported as an error by the caller.
    """
    if not gate_table or not Path(gate_table).exists():
        return None, f"gate table not found ({gate_table}); hotspot set taken from de_hotspots.tsv unchecked"
    if not (Path(de_dir) / 'de_hotspots.tsv').exists():
        return None, "de_hotspots.tsv not found; nothing to check"
    gate, rule = gate_hotspots(Path(gate_table))
    de = {(t.upper(), ct) for t, ct in hotspot_set(de_dir)}
    if gate == de:
        return None, f"{len(gate)} hotspots, same in de_hotspots.tsv and the gate table ({rule})"
    return (f"de_hotspots.tsv and the gate table disagree on the hotspot set (gate rule: {rule}).\n"
            f"  only in the gate table {gate_table}: {sorted(gate - de)}\n"
            f"  only in {Path(de_dir) / 'de_hotspots.tsv'}: {sorted(de - gate)}\n"
            f"  -> rerun Stage 10 against this gate table, then Stage 13"), rule


def provenance_file(network: Path) -> Path:
    """provenance_<name>_pooled.json beside a network table (the name build_grn_pooled.py writes)."""
    network = Path(network)
    return network.parent / f"provenance_{network.stem}.json"


def network_edge_rule(network: Path):
    """The edge rule a network was built under (params.edge_rule of its provenance record), or None when
    the table has no record: every network built before 2026-11-14 is of that kind and was differenced
    against zero where a gene was missing from one arm (audit finding 9.1)."""
    try:
        return json.loads(provenance_file(network).read_text()).get('params', {}).get('edge_rule')
    except (OSError, ValueError):
        return None


def sidecar_files(network: Path) -> list:
    """The files that belong to a network table and travel with it when it is reused."""
    network = Path(network)
    stem = str(network)[:-4]
    return [Path(stem + '_arms.tsv'), Path(stem + '_arms_summary.tsv'), Path(stem + '_decoy_match.tsv'),
            provenance_file(network)]


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
    # STEP3_PREREG.md 8.1: only networks differenced on shared genes may be carried into this arm. A source
    # network with another rule, or with no provenance record (everything built before 2026-11-14), is not
    # reused, whatever the hotspot sets say.
    for a in (src['grn_dir'], src['grn_human_dir']):
        old = [f.name for f in sorted(Path(a).glob('*_pooled.tsv')) if network_edge_rule(f) != EDGE_RULE]
        if old:
            logger.info(f"  hotspot set identical to {profile}, but {len(old)} network(s) under {a} were not "
                        f"built on shared genes (edge rule '{EDGE_RULE}'; first: {old[0]}): "
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
        for net in files:
            # The sidecar tables and the provenance record travel with the network.
            for f in [net] + [s for s in sidecar_files(net) if s.exists()]:
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

    def grn_items(script, dataset_root, out_root, extra=()):
        items = []
        for tissue, ct, name in hots:
            out = out_root / f"{name}_pooled.tsv"
            dataset, meta = dataset_root / tissue / 'dataset', ctx['augur_dir'] / tissue / 'meta.tsv'
            items.append({
                'label': f"{tissue}/{ct}",
                'out': out,
                'script': script, 'dataset': dataset, 'meta': meta, 'model': ctx['model'],
                'cmd': [PY, str(GC / script)] + model + [
                    '--dataset', str(dataset),
                    '--meta', str(meta),
                    '--cell-type', ct,
                    '--out', str(out),
                    '--bootstrap', '30', '--seed', '0', '--device', 'cuda'] + list(extra),
            })
        return items

    cons = ctx['grn_human_dir'] / 'self_consistency'
    return [
        {'num': 1, 'name': 'Differential GRN in rat space (GPU, per cell type)',
         'desc': f"build_grn_pooled.py over abundant hotspots -> {ctx['grn_dir']}/",
         'items': grn_items('build_grn_pooled.py', ctx['gc_input'], ctx['grn_dir']),
         'key': ctx['grn_dir']},
        {'num': 2, 'name': 'Differential GRN in human space (GPU, per cell type)',
         'desc': f"build_grn_human.py over abundant hotspots -> {ctx['grn_human_dir']}/",
         'items': grn_items('build_grn_human.py', ctx['gc_input_human'], ctx['grn_human_dir']),
         'key': ctx['grn_human_dir']},
        {'num': 3, 'name': 'Rat-human self-consistency of the networks (py)',
         'desc': f"compare_grn_conservation.py -> {cons}/",
         'cmd': [PY, str(GC / 'compare_grn_conservation.py'),
                 '--rat-dir', str(ctx['grn_dir']), '--human-dir', str(ctx['grn_human_dir']),
                 '--out-dir', str(cons)],
         'key': cons / 'self_consistency_per_celltype.tsv'},
        {'num': 4, 'name': 'Pathway / gene-set enrichment (R)',
         'desc': f"run_gene_set_enrichment.sh (fgsea vs MSigDB Hallmark/Reactome) -> {ctx['enrich_out']}/",
         'cmd': ['bash', str(DECONV / 'R' / 'run_gene_set_enrichment.sh')],
         'key': ctx['enrich_out'] / 'enrichment_summary.tsv'},
        {'num': 5, 'name': 'Parenchyma data-anchored validation + dominant cell types (py)',
         'desc': 'validate_parenchyma_dataanchored.py -> pseudobulk_de/parenchyma_dataanchored_validation.tsv',
         'cmd': [PY, str(DECONV / 'validate_parenchyma_dataanchored.py')],
         'key': ctx['de_dir'] / 'parenchyma_dataanchored_validation.tsv'},
        {'num': 6, 'name': 'Decoy-null networks in rat space (GPU, per cell type)',
         'desc': f"build_grn_pooled.py --decoy-seed {DECOY_SEED} over abundant hotspots -> {ctx['grn_decoy_dir']}/",
         'items': grn_items('build_grn_pooled.py', ctx['gc_input'], ctx['grn_decoy_dir'],
                            extra=('--decoy-seed', str(DECOY_SEED))),
         'key': ctx['grn_decoy_dir']},
        {'num': 7, 'name': 'Decoy-null scores (py)',
         'desc': f"grn_decoy_fdr.py -> {ctx['grn_decoy_dir'] / 'fdr'}/",
         'cmd': [PY, str(DECONV / 'grn_decoy_fdr.py'), '--tf-dir', str(ctx['grn_dir']),
                 '--decoy-dir', str(ctx['grn_decoy_dir']), '--out', str(ctx['grn_decoy_dir'] / 'fdr')],
         'key': ctx['grn_decoy_dir'] / 'fdr' / 'summary.txt'},
    ]


def record_network_provenance(item: dict) -> None:
    """Write provenance_<name>_pooled.json for a network whose builder did not write one.

    build_grn_pooled.py writes its own record. build_grn_human.py does not yet (it takes the shared-gene
    difference by importing diff() from build_grn_pooled.py), so the orchestrator records, for the network
    it has just built: the command, the inputs, the md5 of the weights file and the edge rule in force at
    this commit (audit finding 9.6: the network tables had no provenance record)."""
    if provenance_file(item['out']).exists():
        return
    from provenance import write_provenance
    weights = Path(item['model']) / 'pytorch_model.bin'
    write_provenance(item['out'].parent, name=provenance_file(item['out']).name,
                     inputs=[item['dataset'], item['meta'], Path(item['model']) / 'config.json'],
                     md5_inputs=[weights, item['meta']],
                     params={'edge_rule': EDGE_RULE, 'builder': item['script'], 'builder_command': item['cmd'],
                             'weights_file': str(weights), 'recorded_by': 'pipeline/run_stage13.py',
                             'registration': 'docs/audit/remediation/STEP3_PREREG.md section 8.1'})


def run_step(step: dict, env: dict, dry_run: bool, force: bool) -> bool:
    logger.info("=" * 70)
    logger.info(f"STEP {step['num']}: {step['name']}")
    logger.info(f"  {step['desc']}")
    logger.info("=" * 70)
    items = step.get('items')
    if items is not None:
        # An existing table counts as done only if it was built on shared genes (STEP3_PREREG.md 8.1). One
        # built under the earlier rule, or with no record, is neither skipped nor silently overwritten.
        stale = [it for it in items if it['out'].exists() and network_edge_rule(it['out']) != EDGE_RULE]
        if stale and not force:
            msg = (f"  {len(stale)} network(s) already in the output folder were not built on shared genes "
                   f"(edge rule '{EDGE_RULE}') or have no provenance record, e.g. {stale[0]['out']}.\n"
                   f"  They are not skipped as present and not overwritten.\n"
                   f"  -> point grn_dir / grn_human_dir at a new folder, or pass --force to rebuild them here")
            if not dry_run:
                logger.error(msg)
                return False
            logger.warning(msg)
        todo = [it for it in items if force or not it['out'].exists()]
        logger.info(f"  {len(items)} cell types; {len(todo)} to run, {len(items) - len(todo)} already present"
                    + (" (--force)" if force else ""))
        for it in todo:
            logger.info(f"  -> {it['label']}")
            logger.info(f"     $ {' '.join(it['cmd'])}")
            if dry_run:
                continue
            t0 = time.time()
            # A build must not inherit an earlier build's record: the record is written after the table.
            provenance_file(it['out']).unlink(missing_ok=True)
            if subprocess.run(it['cmd'], cwd=str(_PROJECT_ROOT), env=env).returncode != 0:
                logger.error(f"  cell type {it['label']} FAILED after {time.time() - t0:.1f}s")
                return False
            record_network_provenance(it)
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
        description='Stage 13: Mechanism (differential GRN, decoy null, rat-human self-consistency, enrichment)',
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument('--model-dir', default=None,
                   help='fine-tuned rat GeneCompass dir or a parent with checkpoint-* (default: '
                        'deconvolution.model_checkpoint; else latest checkpoint, with a warning)')
    p.add_argument('--cell-types', nargs='*', default=[],
                   help="debug subset, e.g. 'SKMVL:Skeletal muscle' or a tissue name (GRN steps only)")
    p.add_argument('--from', dest='from_step', type=int, default=1, choices=[1, 2, 3, 4, 5, 6, 7])
    p.add_argument('--only', type=int, default=None, choices=[1, 2, 3, 4, 5, 6, 7], help='run just this step')
    p.add_argument('--skip-decoy', action='store_true',
                   help='leave out the decoy null (steps 6-7), e.g. for a CPU-only rerun of steps 3-5')
    p.add_argument('--enrich-out', default=None, metavar='DIR',
                   help='step 4: write the enrichment here instead of <pseudobulk_de>/enrichment '
                        '(sets ENRICH_OUT; the DE folder is then only read)')
    p.add_argument('--gene-sets', default=None, metavar='TSV',
                   help='step 4: a gene_sets_used.tsv saved by an earlier enrichment run, to test on the '
                        'identical gene sets (sets GSEA_GENE_SETS; its md5 is verified)')
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
    grn_dir = resolve_path(config, d['grn_dir'])
    de_dir = gc_input / 'pseudobulk_de'
    enrich_out = args.enrich_out or os.environ.get('ENRICH_OUT')
    ctx = {
        'gc_input': gc_input,
        'gc_input_human': Path(str(gc_input) + '_human'),
        'augur_dir': resolve_path(config, d['augur_input_dir']),
        'grn_dir': grn_dir,
        'grn_human_dir': resolve_path(config, d['grn_human_dir']),
        # The decoy networks sit beside the TF networks; grn_decoy_fdr.py uses the same default.
        'grn_decoy_dir': Path(str(grn_dir).rstrip('/') + '_decoy'),
        'de_dir': de_dir,
        'enrich_out': Path(enrich_out).resolve() if enrich_out else de_dir / 'enrichment',
        'gate_table': resolve_path(config, d['hotspots_table']) if d.get('hotspots_table') else None,
        'model': resolve_model_from_config(config, args.model_dir, logger),
    }
    ctx['hotspots'] = abundant_hotspots(ctx['de_dir'], args.cell_types)

    steps = build_steps(ctx)
    steps = [s for s in steps if (s['num'] == args.only if args.only else s['num'] >= args.from_step)]
    if args.skip_decoy:
        steps = [s for s in steps if s['num'] not in (6, 7)]
    nums = {s['num'] for s in steps}
    builds = nums & {1, 2, 6}                 # the steps that load the model and build networks

    logger.info("=" * 70)
    logger.info("STAGE 13: MECHANISM (differential GRN -> self-consistency -> enrichment -> decoy null)")
    logger.info("=" * 70)
    logger.info(f"  steps: {sorted(nums)}")
    logger.info(f"  abundant hotspots (theta_bar>={THETA_MIN}): {len(ctx['hotspots'])} cell types"
                + (f" [subset {args.cell_types}]" if args.cell_types else ""))
    logger.info(f"  model = {ctx['model']}")
    logger.info(f"  grn = {ctx['grn_dir']}  grn_human = {ctx['grn_human_dir']}  decoy = {ctx['grn_decoy_dir']}  "
                f"meta = {ctx['augur_dir']}")
    logger.info(f"  edge rule = {EDGE_RULE} (STEP3_PREREG.md 8.1)")

    errors = []
    if builds:
        if not (ctx['model'] / 'config.json').exists():
            errors.append(f"fine-tuned model not found: {ctx['model']}\n"
                          f"  -> run Stage 7, or pass --model-dir / set deconvolution.model_checkpoint")
        if not ctx['hotspots']:
            errors.append(f"no abundant hotspots from {ctx['de_dir'] / 'de_hotspots.tsv'}\n  -> run Stage 10 first")
        # Networks are built for the gate's hotspots (STEP3_PREREG.md 5.2 and 8.1): the DE folder's list
        # must be the gate table's list.
        mismatch, note = hotspot_mismatch(ctx['de_dir'], ctx['gate_table'])
        if mismatch:
            errors.append(mismatch)
        else:
            logger.info(f"  hotspot check: {note}")
    if 2 in nums and not ctx['gc_input_human'].exists():
        errors.append(f"human-space datasets not found: {ctx['gc_input_human']}\n  -> run Stage 12 transfer first")
    if errors:
        for e in errors:
            (logger.warning if args.dry_run else logger.error)(e)
        if not args.dry_run:
            sys.exit(1)

    # DE_DIR: the enrichment R script's DE folder, passed explicitly (see run_gene_set_enrichment.R).
    # ENRICH_OUT / GSEA_GENE_SETS: its output folder and a saved gene-set table, when given.
    env = dict(os.environ, PIPELINE_ROOT=str(_PROJECT_ROOT), DE_DIR=str(ctx['de_dir']))
    env.setdefault('DECONV_PYTHON', PY)   # the R/bash wrappers read the config with this interpreter
    if enrich_out:
        env['ENRICH_OUT'] = str(ctx['enrich_out'])
    if args.gene_sets:
        env['GSEA_GENE_SETS'] = str(Path(args.gene_sets).resolve())
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
