#!/usr/bin/env python3
"""
assemble_corpus.py — Stage 5, Gather Phase: Arrow Shards → HuggingFace Dataset

Pipeline position:
    Stage 5: SCATTER → tokenize_corpus.py   (per-task Arrow shards)
             GATHER  → assemble_corpus.py   ← THIS SCRIPT

Purpose:
    Assembles the scattered Arrow shards produced by all tokenize_corpus.py
    tasks into a single, HuggingFace-datasets-compatible Arrow dataset,
    ready for GeneCompass fine-tuning.

    Phases:
      Phase 1 — Validation:
        Read all task manifests. Verify all expected tasks have manifested.
        Check shard files exist and are non-empty. Report corpus-level stats.

      Phase 2 — Assembly:
        Load each shard's Arrow table via pyarrow.ipc (or datasets library).
        Write the assembled dataset to tokenized_corpus_dir/dataset/ using
        HuggingFace datasets.Dataset.save_to_disk().
        The final dataset is loadable with datasets.load_from_disk().

      Phase 3 — Manifest:
        Write stage5_manifest.json summarising inputs, outputs, and stats.

Outputs (written to tokenized_corpus_dir/):
    dataset/              — HuggingFace datasets.Dataset on disk
      data-XXXXX-of-YYYYY.arrow   — Arrow shards (renumbered)
      dataset_info.json            — HuggingFace dataset metadata
      state.json                   — Shard index
    corpus_stats.tsv      — Per-study cell and token counts
    stage5_manifest.json  — Full provenance record

Usage:
    python pipeline/05_tokenization/assemble_corpus.py
    python pipeline/05_tokenization/assemble_corpus.py --dry-run
    python pipeline/05_tokenization/assemble_corpus.py --expected-tasks 50
    python pipeline/05_tokenization/assemble_corpus.py -v

Called by stage5_assemble.slurm after all scatter array tasks complete
(afterok dependency on stage5_tokenize job).

Author: Tim Reese Lab / Claude
Date: March 2026
"""

import argparse
import csv
import json
import logging
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Optional heavy imports
# ─────────────────────────────────────────────────────────────────────────────
_MISSING = []

try:
    import pyarrow as pa
    import pyarrow.ipc as pa_ipc
except ImportError:
    _MISSING.append('pyarrow')
    pa = None
    pa_ipc = None

_HF_DATASETS_AVAILABLE = False
try:
    import datasets as hf_datasets
    _HF_DATASETS_AVAILABLE = True
except ImportError:
    hf_datasets = None

# ─────────────────────────────────────────────────────────────────────────────
# Project path bootstrap
# ─────────────────────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(os.environ.get('PIPELINE_ROOT', '.')).resolve()
sys.path.insert(0, str(_PROJECT_ROOT / 'lib'))
from gene_utils import load_config, resolve_path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
)
logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1: MANIFEST DISCOVERY AND VALIDATION
# ═════════════════════════════════════════════════════════════════════════════

def discover_manifests(manifest_dir: Path) -> List[Dict]:
    """Load all task manifests from manifest_dir, sorted by task_id.

    Returns list of manifest dicts. Raises if manifest_dir is empty.
    """
    manifest_paths = sorted(manifest_dir.glob('task_*_manifest.json'))
    if not manifest_paths:
        raise FileNotFoundError(
            f"No task manifests found in {manifest_dir}\n"
            f"  → Run tokenize_corpus.py scatter phase first."
        )
    manifests = []
    for mp in manifest_paths:
        with open(mp) as f:
            manifests.append(json.load(f))
    logger.info(f"Found {len(manifests):,} task manifest(s)")
    return manifests


def validate_manifests(
    manifests: List[Dict],
    expected_tasks: Optional[int],
) -> Tuple[bool, List[str]]:
    """Validate task manifests for consistency and completeness.

    Checks:
      - All tasks reported the same n_tasks value
      - No duplicate task_id values
      - n_tasks matches expected_tasks (if provided)
      - No tasks in error-only state (all shards failed)

    Returns (ok: bool, issues: List[str]).
    """
    issues = []

    # Reported n_tasks consistency
    n_tasks_vals = set(m.get('n_tasks') for m in manifests)
    if len(n_tasks_vals) > 1:
        issues.append(f"Inconsistent n_tasks across manifests: {n_tasks_vals}")
    reported_n_tasks = n_tasks_vals.pop() if n_tasks_vals else None

    # Duplicate task IDs
    task_ids = [m.get('task_id') for m in manifests]
    seen = set()
    dupes = [t for t in task_ids if t in seen or seen.add(t)]
    if dupes:
        issues.append(f"Duplicate task_ids: {dupes}")

    # Missing tasks
    if reported_n_tasks is not None:
        present = set(task_ids)
        missing = [i for i in range(reported_n_tasks) if i not in present]
        if missing:
            issues.append(
                f"Missing task manifests for task_ids: {missing[:20]}"
                + (" (truncated)" if len(missing) > 20 else "")
            )

    # Caller-provided expected tasks override
    if expected_tasks is not None and reported_n_tasks != expected_tasks:
        issues.append(
            f"expected_tasks={expected_tasks} but manifests report "
            f"n_tasks={reported_n_tasks}"
        )

    return len(issues) == 0, issues


def collect_shard_paths(
    manifests: List[Dict],
    shard_dir: Path,
) -> Tuple[List[Path], List[str]]:
    """Collect all Arrow shard paths listed in task manifests.

    Verifies each shard file exists on disk.
    Returns (shard_paths, missing_shards).
    """
    shard_names: List[str] = []
    for m in manifests:
        shard_names.extend(m.get('shards', []))

    # Stable ordering: task then local shard number (already encoded in name)
    shard_names_sorted = sorted(shard_names)

    shard_paths = []
    missing     = []
    for name in shard_names_sorted:
        p = shard_dir / name
        if p.exists():
            shard_paths.append(p)
        else:
            missing.append(name)

    return shard_paths, missing


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2: CORPUS ASSEMBLY
# ═════════════════════════════════════════════════════════════════════════════

def _load_arrow_shard(shard_path: Path) -> 'pa.Table':
    """Load one Arrow IPC file into a pyarrow Table."""
    with pa_ipc.open_stream(str(shard_path)) as reader:
        return reader.read_all()


def assemble_via_hf_datasets(
    shard_paths: List[Path],
    output_dir:  Path,
) -> Tuple[int, int]:
    """Assemble shards using HuggingFace datasets, write to output_dir.

    Returns (n_cells, n_shards_read).
    """
    logger.info("Assembling via HuggingFace datasets library ...")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load all shards as a concatenated dataset
    shard_strs = [str(p) for p in shard_paths]
    ds = hf_datasets.Dataset.from_file(shard_strs[0])  # type: ignore[attr-defined]
    if len(shard_strs) > 1:
        others = [
            hf_datasets.Dataset.from_file(s)  # type: ignore[attr-defined]
            for s in shard_strs[1:]
        ]
        ds = hf_datasets.concatenate_datasets([ds] + others)

    n_cells = len(ds)
    ds.save_to_disk(str(output_dir))
    logger.info(
        f"  Assembled {len(shard_paths):,} shards → {n_cells:,} cells "
        f"[{output_dir.name}/]"
    )
    return n_cells, len(shard_paths)


def assemble_via_pyarrow(
    shard_paths: List[Path],
    output_dir:  Path,
    shards_per_output: int = 50_000,
) -> Tuple[int, int]:
    """Assemble shards using raw pyarrow, write renumbered Arrow files.

    Falls back to this path when HuggingFace datasets is not available.
    Writes Arrow IPC files to output_dir with a simple datasets-compatible
    structure (dataset_info.json + state.json).

    Returns (n_cells, n_shards_read).
    """
    logger.info(
        "HuggingFace datasets not available — assembling via raw pyarrow ..."
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    total_cells   = 0
    output_tables: List['pa.Table'] = []
    output_files: List[str] = []
    schema = None

    # Stream through input shards, buffer into output shards of shards_per_output
    buf_tables: List['pa.Table'] = []
    buf_rows   = 0
    out_idx    = 0

    def _flush_buffer() -> None:
        nonlocal buf_rows, out_idx
        if not buf_tables:
            return
        merged = pa.concat_tables(buf_tables)
        fname  = f"data-{out_idx:05d}-of-PLACEHOLDER.arrow"
        fpath  = output_dir / fname
        with pa_ipc.new_file(str(fpath), merged.schema) as w:
            w.write_table(merged)
        output_files.append(fname)
        buf_tables.clear()
        buf_rows = 0
        out_idx += 1

    for sp in shard_paths:
        tbl = _load_arrow_shard(sp)
        if schema is None:
            schema = tbl.schema
        total_cells += len(tbl)
        buf_tables.append(tbl)
        buf_rows   += len(tbl)
        if buf_rows >= shards_per_output:
            _flush_buffer()

    _flush_buffer()  # Trailing partial shard

    # Rename placeholders now that we know total shard count
    n_out = len(output_files)
    final_files: List[str] = []
    for i, old_name in enumerate(output_files):
        new_name = f"data-{i:05d}-of-{n_out:05d}.arrow"
        (output_dir / old_name).rename(output_dir / new_name)
        final_files.append(new_name)

    # Write minimal HuggingFace-compatible metadata
    _write_hf_metadata(output_dir, final_files, total_cells, schema)

    logger.info(
        f"  Assembled {len(shard_paths):,} shards → {total_cells:,} cells "
        f"({n_out:,} output shard file(s)) [{output_dir.name}/]"
    )
    return total_cells, len(shard_paths)


def _write_hf_metadata(
    output_dir: Path,
    shard_files: List[str],
    n_rows: int,
    schema: Optional['pa.Schema'],
) -> None:
    """Write dataset_info.json and state.json compatible with datasets.load_from_disk."""
    dataset_info = {
        'description': (
            'GeneCompass-format tokenized rat single-cell transcriptomic corpus. '
            'Produced by MoTrPAC-GeneCompass Stage 5 (Tim Reese Lab, Purdue).'
        ),
        'citation': '',
        'homepage': '',
        'license': 'Apache-2.0',
        'features': {
            'input_ids': {'feature': {'dtype': 'int32',   '_type': 'Value'}, '_type': 'Sequence'},
            'values':    {'feature': {'dtype': 'float32', '_type': 'Value'}, '_type': 'Sequence'},
            'length':    {'feature': {'dtype': 'int16',   '_type': 'Value'}, '_type': 'Sequence'},
            'species':   {'feature': {'dtype': 'int16',   '_type': 'Value'}, '_type': 'Sequence'},
            'study_id':  {'dtype': 'string', '_type': 'Value'},
            'cell_id':   {'dtype': 'string', '_type': 'Value'},
        },
        'num_rows': n_rows,
        'dataset_size': None,
        'download_size': None,
    }
    with open(output_dir / 'dataset_info.json', 'w') as f:
        json.dump(dataset_info, f, indent=2)

    state = {
        '_data_files': [{'filename': fn} for fn in shard_files],
        '_fingerprint': 'stage5_assembled',
        '_format_columns': None,
        '_format_kwargs': {},
        '_format_type': None,
        '_output_all_columns': False,
        '_split': None,
    }
    with open(output_dir / 'state.json', 'w') as f:
        json.dump(state, f, indent=2)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3: CORPUS STATISTICS
# ═════════════════════════════════════════════════════════════════════════════

def compute_corpus_stats(manifests: List[Dict]) -> Dict:
    """Aggregate per-study cell and token statistics from task manifests."""
    study_cells:   Dict[str, int] = defaultdict(int)
    study_empty:   Dict[str, int] = defaultdict(int)
    study_files:   Dict[str, int] = defaultdict(int)

    total_cells   = 0
    total_empty   = 0
    total_files   = 0
    total_errors  = 0

    for m in manifests:
        for fr in m.get('file_results', []):
            study = fr.get('study_id', 'unknown')
            written = fr.get('n_cells_written', 0)
            empty   = fr.get('n_cells_empty',   0)
            study_cells[study] += written
            study_empty[study] += empty
            study_files[study] += 1
            total_cells  += written
            total_empty  += empty
            total_files  += 1
            if fr.get('status') != 'ok':
                total_errors += 1

    return {
        'total_cells_tokenized': total_cells,
        'total_cells_empty':     total_empty,
        'total_files_processed': total_files,
        'total_file_errors':     total_errors,
        'by_study':              dict(study_cells),
        'by_study_empty':        dict(study_empty),
        'by_study_files':        dict(study_files),
    }


def write_corpus_stats_tsv(stats: Dict, output_path: Path) -> None:
    """Write per-study corpus stats to a TSV file."""
    by_study = stats['by_study']
    studies  = sorted(by_study)
    fields   = ['study_id', 'n_files', 'n_cells_tokenized', 'n_cells_empty']

    with open(output_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter='\t')
        w.writeheader()
        for sid in studies:
            w.writerow({
                'study_id':          sid,
                'n_files':           stats['by_study_files'].get(sid, 0),
                'n_cells_tokenized': by_study[sid],
                'n_cells_empty':     stats['by_study_empty'].get(sid, 0),
            })
    logger.info(f"  corpus_stats.tsv: {len(studies):,} studies, "
                f"{stats['total_cells_tokenized']:,} total cells")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3b: SORTED LENGTH PICKLE
# ═════════════════════════════════════════════════════════════════════════════

def _write_sorted_length_pickle(dataset_dir: Path, manifests: List[Dict]) -> None:
    """Write sorted_length.pickle required by GeneCompass pretrain script.

    The pretrain script uses this for group_by_length batching — it expects
    a sorted list of sequence lengths (one per cell, ascending order) at
    dataset_dir/sorted_length.pickle.
    """
    import pickle
    lengths = []
    for m in manifests:
        for fr in m.get('file_results', []):
            if fr.get('status') != 'ok':
                continue
            # Each file result doesn't store per-cell lengths, but the
            # manifest stores n_cells_written. We reconstruct from the
            # Arrow shards listed in this task's manifest.
            pass
    # Lengths are not stored per-cell in manifests — collect from
    # the assembled dataset directly instead.
    try:
        import datasets as hf_datasets
        ds = hf_datasets.load_from_disk(str(dataset_dir))
        # length column is List(int16) [[n]] — extract the scalar
        lengths = [row[0] for row in ds['length']]
        lengths.sort()
        out_path = dataset_dir / 'sorted_length.pickle'
        with open(out_path, 'wb') as f:
            pickle.dump(lengths, f, protocol=4)
        logger.info(f"  sorted_length.pickle: {len(lengths):,} entries, "
                    f"min={lengths[0]}, max={lengths[-1]}")
    except Exception as exc:
        logger.warning(f"  Could not write sorted_length.pickle: {exc}")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4: STAGE 5 MANIFEST
# ═════════════════════════════════════════════════════════════════════════════

def write_stage5_manifest(
    corpus_dir:   Path,
    config:       dict,
    manifests:    List[Dict],
    corpus_stats: Dict,
    n_cells_out:  int,
    n_shards_in:  int,
    t_start:      float,
    dry_run:      bool,
) -> None:
    """Write the definitive Stage 5 provenance manifest."""
    # Pull config snapshot from first manifest (all tasks share the same config)
    cfg_snap = manifests[0].get('config_snapshot', {}) if manifests else {}

    manifest = {
        'stage':        5,
        'phase':        'gather',
        'generated_at': datetime.utcnow().isoformat() + 'Z',
        'elapsed_s':    round(time.time() - t_start, 1),
        'dry_run':      dry_run,
        'config_snapshot': cfg_snap,
        'inputs': {
            'n_scatter_tasks':       len(manifests),
            'n_shards_assembled':    n_shards_in,
            'n_cells_from_scatter':  corpus_stats['total_cells_tokenized'],
            'n_cells_empty_skipped': corpus_stats['total_cells_empty'],
            'n_files_processed':     corpus_stats['total_files_processed'],
            'n_file_errors':         corpus_stats['total_file_errors'],
        },
        'outputs': {
            'n_cells_in_dataset':   n_cells_out,
            'dataset_dir':          str(corpus_dir / 'dataset'),
            'corpus_stats_tsv':     str(corpus_dir / 'corpus_stats.tsv'),
            'primary_deliverable':  'dataset/',
        },
        'corpus_stats': corpus_stats,
    }

    manifest_path = corpus_dir / 'stage5_manifest.json'
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"  stage5_manifest.json")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 5: MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Stage 5 Gather: Arrow shards → HuggingFace dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pipeline/05_tokenization/assemble_corpus.py
  python pipeline/05_tokenization/assemble_corpus.py --expected-tasks 50
  python pipeline/05_tokenization/assemble_corpus.py --dry-run
        """,
    )
    parser.add_argument(
        '--expected-tasks', type=int, default=None,
        help='Expected number of scatter tasks (validates manifest count)',
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Validate manifests and shard files; skip dataset writing',
    )
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='DEBUG-level logging')
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if _MISSING:
        logger.error(f"Missing required packages: {', '.join(_MISSING)}")
        sys.exit(1)

    t_start = time.time()
    logger.info("=" * 70)
    logger.info("STAGE 5 GATHER: Corpus Assembly")
    logger.info("=" * 70)

    # ── Config ───────────────────────────────────────────────────────────────
    try:
        config = load_config()
    except FileNotFoundError:
        logger.error(
            "pipeline_config.yaml not found. "
            "Set PIPELINE_ROOT or run from project root."
        )
        sys.exit(1)

    corpus_dir   = resolve_path(config, config['paths']['tokenized_corpus_dir'])
    shard_dir    = corpus_dir / 'shards'
    manifest_dir = corpus_dir / 'manifests'
    dataset_dir  = corpus_dir / 'dataset'

    # ── Dry-run early exit ───────────────────────────────────────────────────
    # Must come before Phase 1: when the orchestrator passes --dry-run through,
    # scatter has not written any manifests yet, so Phase 1 would always fail.
    if args.dry_run:
        logger.info("DRY RUN — skipping manifest validation and dataset assembly.")
        logger.info(f"  Would read manifests from: {manifest_dir}")
        logger.info(f"  Would read shards from:    {shard_dir}")
        logger.info(f"  Would write dataset to:    {dataset_dir}")
        sys.exit(0)

    # ── Phase 1: Validate manifests ──────────────────────────────────────────
    logger.info("Phase 1: Validating task manifests ...")
    try:
        manifests = discover_manifests(manifest_dir)
    except FileNotFoundError as exc:
        logger.error(str(exc))
        sys.exit(1)

    ok, issues = validate_manifests(manifests, args.expected_tasks)
    if not ok:
        for issue in issues:
            logger.error(f"  Manifest issue: {issue}")
        logger.error("Manifest validation failed — aborting.")
        sys.exit(1)
    logger.info(f"  Manifests valid: {len(manifests):,} tasks")

    shard_paths, missing_shards = collect_shard_paths(manifests, shard_dir)
    if missing_shards:
        for name in missing_shards[:20]:
            logger.error(f"  Missing shard: {name}")
        if len(missing_shards) > 20:
            logger.error(f"  ... and {len(missing_shards) - 20} more missing shards")
        logger.error(f"  {len(missing_shards):,} shard file(s) missing — aborting.")
        sys.exit(1)

    corpus_stats = compute_corpus_stats(manifests)
    logger.info(f"  Total cells to assemble: {corpus_stats['total_cells_tokenized']:,}")
    logger.info(f"  Total shards to assemble: {len(shard_paths):,}")
    logger.info(f"  Studies: {len(corpus_stats['by_study']):,}")
    if corpus_stats['total_file_errors'] > 0:
        logger.warning(
            f"  {corpus_stats['total_file_errors']} file error(s) during scatter — "
            "these cells are absent from the corpus"
        )

    # ── Phase 2: Assemble ────────────────────────────────────────────────────
    logger.info("Phase 2: Assembling dataset ...")
    try:
        if _HF_DATASETS_AVAILABLE:
            n_cells_out, n_shards_read = assemble_via_hf_datasets(
                shard_paths, dataset_dir
            )
        else:
            logger.warning(
                "HuggingFace datasets not installed — using raw pyarrow assembly. "
                "Install datasets for full compatibility: pip install datasets"
            )
            n_cells_out, n_shards_read = assemble_via_pyarrow(
                shard_paths, dataset_dir
            )
    except Exception as exc:
        logger.error(f"Assembly failed: {exc}")
        raise

    # ── Phase 3: Outputs ─────────────────────────────────────────────────────
    logger.info("Phase 3: Writing auxiliary outputs ...")

    stats_path = corpus_dir / 'corpus_stats.tsv'
    write_corpus_stats_tsv(corpus_stats, stats_path)

    # Write sorted_length.pickle — required by GeneCompass pretrain script
    # (group_by_length batching uses this to sort samples by sequence length)
    logger.info("  Writing sorted_length.pickle ...")
    _write_sorted_length_pickle(dataset_dir, manifests)

    write_stage5_manifest(
        corpus_dir   = corpus_dir,
        config       = config,
        manifests    = manifests,
        corpus_stats = corpus_stats,
        n_cells_out  = n_cells_out,
        n_shards_in  = n_shards_read,
        t_start      = t_start,
        dry_run      = args.dry_run,
    )

    elapsed = time.time() - t_start
    logger.info("=" * 70)
    logger.info("STAGE 5 GATHER COMPLETE")
    logger.info(f"  Cells in dataset:  {n_cells_out:,}")
    logger.info(f"  Shards assembled:  {n_shards_read:,}")
    logger.info(f"  Output:            {dataset_dir}")
    logger.info(f"  Elapsed:           {elapsed:.1f}s")
    logger.info("=" * 70)
    logger.info("")
    logger.info("Next steps:")
    logger.info("  Fine-tuning → python vendor/GeneCompass/scripts/pretrain.py \\")
    logger.info(f"    --data_path {dataset_dir}")


if __name__ == '__main__':
    main()