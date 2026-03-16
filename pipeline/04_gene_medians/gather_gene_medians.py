#!/usr/bin/env python3
"""
gather_gene_medians.py — Stage 4, Gather Phase: Merge Scatter → Final Median Dictionary

Pipeline position:
    Stage 4: SCATTER → compute_gene_medians.py   (runs first, per SLURM array task)
             GATHER  → gather_gene_medians.py    ← THIS SCRIPT  (runs after all scatter)
    Stage 5: Reference Assembly & Tokenization   (reads rat_gene_medians.pickle)

Purpose:
    Aggregates all scatter task outputs (produced by compute_gene_medians.py)
    into the final per-gene statistics dictionary required by Stage 5.

    Phases:
      Phase 1 — Stats aggregation:
        Load all scatter_NNNN_stats.npz files. Sum running stats arrays to get
        global n_nonzero, n_cells_total, sum_norm, sum_sq_norm, sum_raw, sum_sq_raw.
        Compute: mean, variance, std, zero_fraction (all genes, all stats types).

      Phase 2 — Exact median computation:
        For each scatter task's _nonzero.npz, route gene values into per-gene
        accumulators. After all scatter files loaded, compute np.median and
        percentiles (25th, 75th, 95th) per gene.
        Memory strategy: gene-batch processing to cap RAM usage.

      Phase 3 — Filtering and output:
        Apply min_nonzero_cells threshold (from config: medians.min_nonzero_cells).
        Write all output files.

Primary deliverable — rat_gene_medians.pickle:
    A dict {ensembl_id (str): median_normalized_expression (float32)}.
    Format matches GeneCompass's human_gene_median_after_filter.pickle.
    Stage 5 reads this to convert raw counts → median-normalized → ranked tokens.

All other outputs:
    gene_median_stats.tsv      — Full per-gene stats table (all computed metrics)
    gene_median_stats.npz      — Numpy version of above (fast loading for Stage 5)
    biotype_summary.json       — Protein-coding vs lncRNA vs miRNA aggregate stats
    cell_stats_summary.json    — Corpus-level cell library size distribution
    stage4_manifest.json       — Provenance (config snapshot, input checksums, run info)

Usage:
    python gather_gene_medians.py
    python gather_gene_medians.py --dry-run
    python gather_gene_medians.py --gene-batch-size 2000  # reduce peak RAM
    python gather_gene_medians.py -v

Called by stage4_gather.slurm after all scatter tasks complete (afterok dependency).

Author: Tim Reese Lab / Claude
Date: March 2026
"""

import argparse
import csv
import gzip
import json
import logging
import os
import pickle
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

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
# SECTION 1: SCATTER FILE DISCOVERY AND VALIDATION
# ═════════════════════════════════════════════════════════════════════════════

def discover_scatter_files(scatter_dir: Path) -> List[Dict]:
    """Find all scatter task outputs and validate they are complete.

    Returns list of dicts, one per task: {task_id, stats_path, nonzero_path,
    cells_path, manifest_path, n_cells, status}.
    Raises if any task has missing stats or nonzero files (critical outputs).
    """
    manifests = sorted(scatter_dir.glob('scatter_*_manifest.json'))
    if not manifests:
        raise FileNotFoundError(
            f"No scatter manifests found in {scatter_dir}\n"
            f"  → Run scatter phase first (sbatch slurm/pipeline/stage4_scatter.slurm)"
        )

    tasks = []
    errors = []
    for mpath in manifests:
        with open(mpath) as f:
            mdata = json.load(f)

        task_id = mdata['task_id']
        pfx = scatter_dir / f"scatter_{task_id:04d}"

        stats_path = Path(str(pfx) + '_stats.npz')
        nonzero_path = Path(str(pfx) + '_nonzero.npz')
        cells_path = Path(str(pfx) + '_cells.tsv.gz')

        missing = []
        if not stats_path.exists():
            missing.append(stats_path.name)
        if not nonzero_path.exists():
            missing.append(nonzero_path.name)

        if missing:
            errors.append(f"Task {task_id}: missing {missing}")

        tasks.append({
            'task_id': task_id,
            'stats_path': stats_path,
            'nonzero_path': nonzero_path,
            'cells_path': cells_path if cells_path.exists() else None,
            'manifest': mdata,
            'n_cells': mdata.get('n_cells_processed', 0),
            'n_files': mdata.get('n_files_processed', 0),
            'complete': not missing,
        })

    if errors:
        for e in errors:
            logger.error(f"  {e}")
        raise RuntimeError(
            f"{len(errors)} scatter task(s) have missing output files.\n"
            f"  Re-run failed array tasks before gathering:\n"
            f"  sbatch --array=<failed_ids> slurm/pipeline/stage4_scatter.slurm"
        )

    tasks.sort(key=lambda t: t['task_id'])
    logger.info(f"Scatter tasks found: {len(tasks)} (all complete)")
    return tasks


def load_gene_universe(ortholog_dir: Path) -> Tuple[List[str], Dict[str, str]]:
    """Load the canonical sorted gene list from Stage 3. Same function as scatter."""
    mapping_path = ortholog_dir / 'rat_token_mapping.tsv'
    genes = {}
    with open(mapping_path, 'r') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            if row.get('tier', '').strip() == 'excluded':
                continue
            eid = row['rat_gene'].strip().split('.')[0].upper()
            if eid:
                genes[eid] = row.get('biotype', '').strip()
    gene_ids = sorted(genes.keys())
    gene_biotype = {eid: genes[eid] for eid in gene_ids}
    logger.info(f"Gene universe: {len(gene_ids):,} genes")
    return gene_ids, gene_biotype


def validate_scatter_gene_universe(stats_path: Path, n_expected: int) -> bool:
    """Sanity check: verify scatter file was built with same gene universe size."""
    data = np.load(stats_path, allow_pickle=True)
    n_got = len(data['n_cells_total'])
    if n_got != n_expected:
        logger.error(
            f"Gene universe mismatch: {stats_path.name} has {n_got} genes, "
            f"expected {n_expected}. Scatter/gather universe mismatch."
        )
        return False
    return True


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2: PHASE 1 — AGGREGATE RUNNING STATS
# ═════════════════════════════════════════════════════════════════════════════

def aggregate_stats(tasks: List[Dict], n_genes: int) -> Dict[str, np.ndarray]:
    """Sum running statistics across all scatter tasks.

    Produces global arrays (shape n_genes) for all running stats.
    This phase is fast and low-memory: each stats.npz is ~50 MB.
    """
    logger.info("Phase 1: Aggregating running stats across scatter tasks...")

    global_stats = {
        'sum_norm': np.zeros(n_genes, dtype=np.float64),
        'sum_sq_norm': np.zeros(n_genes, dtype=np.float64),
        'sum_raw': np.zeros(n_genes, dtype=np.float64),
        'sum_sq_raw': np.zeros(n_genes, dtype=np.float64),
        'n_nonzero_norm': np.zeros(n_genes, dtype=np.int64),
        'n_nonzero_raw': np.zeros(n_genes, dtype=np.int64),
        'n_cells_total': np.zeros(n_genes, dtype=np.int64),
    }

    for task in tasks:
        data = np.load(task['stats_path'], allow_pickle=True)
        for key in global_stats:
            global_stats[key] += data[key]
        logger.debug(f"  Loaded stats: task {task['task_id']}")

    n_total_cells = int(global_stats['n_cells_total'].max())
    n_expressed = int((global_stats['n_nonzero_norm'] > 0).sum())
    logger.info(f"  Total cells (max across genes): {n_total_cells:,}")
    logger.info(f"  Genes with ≥1 non-zero cell:    {n_expressed:,} / {n_genes:,}")

    return global_stats


def compute_derived_stats(global_stats: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Compute mean, variance, std, and zero_fraction from running sums.

    Uses the Welford-compatible formula:
        var = (sum_sq - sum^2 / n) / n   (population variance, n = n_nonzero)
    Non-zero statistics are computed only over non-zero cells.
    Including-zero statistics use n_cells_total as denominator.
    """
    logger.info("  Computing mean/variance/std/zero_fraction from running stats...")
    n_genes = len(global_stats['n_cells_total'])
    result = {}

    for kind in ('norm', 'raw'):
        n_nz = global_stats[f'n_nonzero_{kind}'].astype(np.float64)
        n_tot = global_stats['n_cells_total'].astype(np.float64)
        s = global_stats[f'sum_{kind}']
        sq = global_stats[f'sum_sq_{kind}']

        # Mean (non-zero cells only)
        mean_nz = np.where(n_nz > 0, s / n_nz, 0.0)
        # Variance (non-zero cells only, population)
        var_nz = np.where(
            n_nz > 1,
            np.maximum(0.0, (sq - s ** 2 / np.where(n_nz > 0, n_nz, 1)) / n_nz),
            0.0,
        )
        # Mean including zeros
        mean_all = np.where(n_tot > 0, s / n_tot, 0.0)
        # Zero fraction
        zero_frac = np.where(n_tot > 0, 1.0 - n_nz / n_tot, 1.0)

        result[f'mean_{kind}_nonzero'] = mean_nz.astype(np.float32)
        result[f'var_{kind}_nonzero'] = var_nz.astype(np.float32)
        result[f'std_{kind}_nonzero'] = np.sqrt(var_nz).astype(np.float32)
        result[f'mean_{kind}_all'] = mean_all.astype(np.float32)
        result[f'zero_fraction_{kind}'] = zero_frac.astype(np.float32)

    return result


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3: PHASE 2 — EXACT MEDIAN AND PERCENTILES
# ═════════════════════════════════════════════════════════════════════════════

def compute_exact_medians(
    tasks: List[Dict],
    n_genes: int,
    gene_batch_size: int = 2500,
) -> Dict[str, np.ndarray]:
    """Compute exact medians and percentiles by loading scatter nonzero files.

    Memory strategy:
        Process genes in batches of gene_batch_size. For each batch:
          - Allocate per-gene lists only for genes in this batch.
          - Scan ALL scatter nonzero.npz files, extracting only values for
            genes in the current batch.
          - Compute median and percentiles for the batch.
          - Free memory before next batch.

        Peak RAM: (values for gene_batch_size genes) × n_scatter_tasks.
        With batch_size=2500 and 10.2M cells × 10% non-zero per gene ≈ 1M
        values per gene: 2500 × 1M × 4 bytes ≈ 10 GB per batch.
        Adjust gene_batch_size down (e.g., 500) if RAM is tight.

    Returns dict of float32 arrays, shape (n_genes,):
        median_norm, p25_norm, p75_norm, p95_norm,
        median_raw,  p25_raw,  p75_raw,  p95_raw
    """
    logger.info(f"Phase 2: Computing exact medians "
                f"(gene batches of {gene_batch_size:,})...")

    median_norm = np.zeros(n_genes, dtype=np.float32)
    p25_norm = np.zeros(n_genes, dtype=np.float32)
    p75_norm = np.zeros(n_genes, dtype=np.float32)
    p95_norm = np.zeros(n_genes, dtype=np.float32)
    median_raw = np.zeros(n_genes, dtype=np.float32)
    p25_raw = np.zeros(n_genes, dtype=np.float32)
    p75_raw = np.zeros(n_genes, dtype=np.float32)
    p95_raw = np.zeros(n_genes, dtype=np.float32)

    n_batches = (n_genes + gene_batch_size - 1) // gene_batch_size
    percentiles = [25, 50, 75, 95]

    for batch_idx in range(n_batches):
        g_start = batch_idx * gene_batch_size
        g_end = min(g_start + gene_batch_size, n_genes)
        batch_size = g_end - g_start

        t_batch = time.time()
        logger.info(f"  Batch {batch_idx + 1}/{n_batches}: "
                    f"genes {g_start:,}–{g_end - 1:,}")

        # Per-gene value accumulators for this batch
        bufs_norm = [[] for _ in range(batch_size)]
        bufs_raw = [[] for _ in range(batch_size)]

        # Scan all scatter nonzero files
        for task in tasks:
            data = np.load(task['nonzero_path'], allow_pickle=False)
            offsets_norm = data['gene_offsets_norm']
            values_norm = data['nonzero_values_norm']
            offsets_raw = data['gene_offsets_raw']
            values_raw = data['nonzero_values_raw']

            for local_b, global_g in enumerate(range(g_start, g_end)):
                ns = offsets_norm[global_g]
                ne = offsets_norm[global_g + 1]
                if ne > ns:
                    bufs_norm[local_b].append(values_norm[ns:ne])

                rs = offsets_raw[global_g]
                re = offsets_raw[global_g + 1]
                if re > rs:
                    bufs_raw[local_b].append(values_raw[rs:re])

        # Compute statistics for this batch
        for local_b, global_g in enumerate(range(g_start, g_end)):
            if bufs_norm[local_b]:
                vals = np.concatenate(bufs_norm[local_b])
                p = np.percentile(vals, percentiles).astype(np.float32)
                p25_norm[global_g] = p[0]
                median_norm[global_g] = p[1]
                p75_norm[global_g] = p[2]
                p95_norm[global_g] = p[3]

            if bufs_raw[local_b]:
                vals = np.concatenate(bufs_raw[local_b])
                p = np.percentile(vals, percentiles).astype(np.float32)
                p25_raw[global_g] = p[0]
                median_raw[global_g] = p[1]
                p75_raw[global_g] = p[2]
                p95_raw[global_g] = p[3]

        elapsed_batch = time.time() - t_batch
        n_with_median = int((median_norm[g_start:g_end] > 0).sum())
        logger.info(f"    Genes with non-zero median: {n_with_median}/{batch_size}  "
                    f"({elapsed_batch:.1f}s)")

        # Explicit cleanup to release batch memory
        del bufs_norm, bufs_raw

    logger.info(f"  Median computation complete.")
    logger.info(f"  Genes with median > 0: {int((median_norm > 0).sum()):,}")

    return {
        'median_norm': median_norm,
        'p25_norm': p25_norm,
        'p75_norm': p75_norm,
        'p95_norm': p95_norm,
        'median_raw': median_raw,
        'p25_raw': p25_raw,
        'p75_raw': p75_raw,
        'p95_raw': p95_raw,
    }


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4: PHASE 3 — OUTPUT WRITING
# ═════════════════════════════════════════════════════════════════════════════

def build_cell_stats_summary(tasks: List[Dict]) -> Dict:
    """Aggregate per-file cell stats across all scatter tasks."""
    all_rows = []
    for task in tasks:
        if task['cells_path'] is None:
            continue
        try:
            with gzip.open(task['cells_path'], 'rt') as gz:
                reader = csv.DictReader(gz, delimiter='\t')
                all_rows.extend(list(reader))
        except Exception as e:
            logger.warning(f"  Could not read {task['cells_path'].name}: {e}")

    if not all_rows:
        return {}

    def _col_floats(col):
        vals = []
        for r in all_rows:
            try:
                vals.append(float(r[col]))
            except (KeyError, ValueError):
                pass
        return np.array(vals, dtype=np.float32) if vals else np.array([])

    n_cells_arr = _col_floats('n_cells')
    lib_mean_arr = _col_floats('lib_size_mean')
    lib_med_arr = _col_floats('lib_size_median')
    gene_mean_arr = _col_floats('n_genes_mean')

    # Weighted medians approximate corpus-level stats
    total_cells = int(n_cells_arr.sum()) if len(n_cells_arr) > 0 else 0
    return {
        'n_files': len(all_rows),
        'total_cells': total_cells,
        'lib_size_mean_of_file_means': float(np.mean(lib_mean_arr)) if len(lib_mean_arr) else 0,
        'lib_size_median_of_file_medians': float(np.median(lib_med_arr)) if len(lib_med_arr) else 0,
        'n_genes_mean_of_file_means': float(np.mean(gene_mean_arr)) if len(gene_mean_arr) else 0,
    }


def build_biotype_summary(
    gene_ids: List[str],
    gene_biotype: Dict[str, str],
    median_norm: np.ndarray,
    global_stats: Dict[str, np.ndarray],
    derived: Dict[str, np.ndarray],
    min_nonzero_cells: int,
) -> Dict:
    """Compute per-biotype aggregate statistics for the manifest."""
    from collections import defaultdict
    biotype_data = defaultdict(lambda: {
        'n_total': 0, 'n_with_median': 0, 'n_pass_filter': 0,
        'medians': [], 'zero_fracs': [],
    })
    n_nonzero_arr = global_stats['n_nonzero_norm']

    for i, eid in enumerate(gene_ids):
        bt = gene_biotype.get(eid, 'unknown')
        d = biotype_data[bt]
        d['n_total'] += 1
        if median_norm[i] > 0:
            d['n_with_median'] += 1
            d['medians'].append(float(median_norm[i]))
        if n_nonzero_arr[i] >= min_nonzero_cells:
            d['n_pass_filter'] += 1
        d['zero_fracs'].append(float(derived['zero_fraction_norm'][i]))

    summary = {}
    for bt, d in sorted(biotype_data.items()):
        summary[bt] = {
            'n_total': d['n_total'],
            'n_with_median': d['n_with_median'],
            'n_pass_filter': d['n_pass_filter'],
            'median_of_medians': float(np.median(d['medians'])) if d['medians'] else 0,
            'mean_median': float(np.mean(d['medians'])) if d['medians'] else 0,
            'mean_zero_fraction': float(np.mean(d['zero_fracs'])),
        }
    return summary


def write_outputs(
    output_dir: Path,
    scatter_dir: Path,
    gene_ids: List[str],
    gene_biotype: Dict[str, str],
    global_stats: Dict[str, np.ndarray],
    derived: Dict[str, np.ndarray],
    median_data: Dict[str, np.ndarray],
    tasks: List[Dict],
    min_nonzero_cells: int,
    config: dict,
    t_start: float,
) -> None:
    """Write all Stage 4 outputs to output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    n_genes = len(gene_ids)
    median_norm = median_data['median_norm']
    n_nonzero = global_stats['n_nonzero_norm']

    # ── Build pass-filter mask ────────────────────────────────────────────────
    pass_mask = n_nonzero >= min_nonzero_cells
    n_pass = int(pass_mask.sum())
    logger.info(f"Filter: ≥{min_nonzero_cells} non-zero cells → "
                f"{n_pass:,} / {n_genes:,} genes pass")

    # ── 1. Primary deliverable: rat_gene_medians.pickle ──────────────────────
    # Format: {ensembl_id: float32_median}  (matches GeneCompass human pickle)
    median_dict = {
        gene_ids[i]: float(median_norm[i])
        for i in range(n_genes)
        if pass_mask[i]
    }
    pickle_path = output_dir / 'rat_gene_medians.pickle'
    with open(pickle_path, 'wb') as f:
        pickle.dump(median_dict, f, protocol=4)
    logger.info(f"  rat_gene_medians.pickle  ({len(median_dict):,} genes, "
                f"{pickle_path.stat().st_size / 1e6:.1f} MB)")

    # ── 2. Full stats TSV ─────────────────────────────────────────────────────
    tsv_path = output_dir / 'gene_median_stats.tsv'
    fields = [
        'rat_gene', 'biotype', 'pass_filter',
        'median_norm', 'p25_norm', 'p75_norm', 'p95_norm',
        'mean_norm_nonzero', 'var_norm_nonzero', 'std_norm_nonzero',
        'mean_norm_all', 'zero_fraction_norm',
        'n_nonzero_norm', 'n_cells_total',
        'median_raw', 'p25_raw', 'p75_raw', 'p95_raw',
        'mean_raw_nonzero', 'std_raw_nonzero', 'zero_fraction_raw',
        'n_nonzero_raw',
    ]
    with open(tsv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter='\t')
        w.writeheader()
        for i, eid in enumerate(gene_ids):
            row = {
                'rat_gene': eid,
                'biotype': gene_biotype.get(eid, ''),
                'pass_filter': int(pass_mask[i]),
                'median_norm': round(float(median_norm[i]), 6),
                'p25_norm': round(float(median_data['p25_norm'][i]), 6),
                'p75_norm': round(float(median_data['p75_norm'][i]), 6),
                'p95_norm': round(float(median_data['p95_norm'][i]), 6),
                'mean_norm_nonzero': round(float(derived['mean_norm_nonzero'][i]), 6),
                'var_norm_nonzero': round(float(derived['var_norm_nonzero'][i]), 6),
                'std_norm_nonzero': round(float(derived['std_norm_nonzero'][i]), 6),
                'mean_norm_all': round(float(derived['mean_norm_all'][i]), 6),
                'zero_fraction_norm': round(float(derived['zero_fraction_norm'][i]), 4),
                'n_nonzero_norm': int(n_nonzero[i]),
                'n_cells_total': int(global_stats['n_cells_total'][i]),
                'median_raw': round(float(median_data['median_raw'][i]), 4),
                'p25_raw': round(float(median_data['p25_raw'][i]), 4),
                'p75_raw': round(float(median_data['p75_raw'][i]), 4),
                'p95_raw': round(float(median_data['p95_raw'][i]), 4),
                'mean_raw_nonzero': round(float(derived['mean_raw_nonzero'][i]), 4),
                'std_raw_nonzero': round(float(derived['std_raw_nonzero'][i]), 4),
                'zero_fraction_raw': round(float(derived['zero_fraction_raw'][i]), 4),
                'n_nonzero_raw': int(global_stats['n_nonzero_raw'][i]),
            }
            w.writerow(row)
    logger.info(f"  gene_median_stats.tsv    ({tsv_path.stat().st_size / 1e6:.1f} MB)")

    # ── 3. Numpy stats archive (fast loading) ─────────────────────────────────
    npz_path = output_dir / 'gene_median_stats.npz'
    gene_ids_bytes = np.array([g.encode('ascii') for g in gene_ids], dtype=object)
    np.savez_compressed(
        npz_path,
        gene_ids=gene_ids_bytes,
        pass_filter=pass_mask.astype(np.bool_),
        median_norm=median_data['median_norm'],
        p25_norm=median_data['p25_norm'],
        p75_norm=median_data['p75_norm'],
        p95_norm=median_data['p95_norm'],
        mean_norm_nonzero=derived['mean_norm_nonzero'],
        std_norm_nonzero=derived['std_norm_nonzero'],
        zero_fraction_norm=derived['zero_fraction_norm'],
        n_nonzero_norm=global_stats['n_nonzero_norm'],
        n_cells_total=global_stats['n_cells_total'],
        median_raw=median_data['median_raw'],
        mean_raw_nonzero=derived['mean_raw_nonzero'],
        zero_fraction_raw=derived['zero_fraction_raw'],
    )
    logger.info(f"  gene_median_stats.npz    ({npz_path.stat().st_size / 1e6:.1f} MB)")

    # ── 4. Biotype summary ─────────────────────────────────────────────────────
    biotype_summary = build_biotype_summary(
        gene_ids, gene_biotype, median_norm, global_stats, derived, min_nonzero_cells
    )
    bt_path = output_dir / 'biotype_summary.json'
    with open(bt_path, 'w') as f:
        json.dump(biotype_summary, f, indent=2)
    logger.info(f"  biotype_summary.json")

    # ── 5. Cell stats summary ──────────────────────────────────────────────────
    cell_summary = build_cell_stats_summary(tasks)
    cs_path = output_dir / 'cell_stats_summary.json'
    with open(cs_path, 'w') as f:
        json.dump(cell_summary, f, indent=2)
    logger.info(f"  cell_stats_summary.json  ({cell_summary.get('total_cells', 0):,} cells)")

    # ── 6. Stage 4 manifest ────────────────────────────────────────────────────
    med_cfg = config.get('medians', {})
    total_cells = sum(t['n_cells'] for t in tasks)
    total_files = sum(t['n_files'] for t in tasks)

    manifest = {
        'stage': 4,
        'phase': 'gather',
        'generated_at': datetime.utcnow().isoformat() + 'Z',
        'elapsed_s': round(time.time() - t_start, 1),
        'config_snapshot': {
            'target_sum': med_cfg.get('target_sum', 10000),
            'min_nonzero_cells': min_nonzero_cells,
            'ensembl_release': config.get('biomart', {}).get('ensembl_release', '?'),
        },
        'inputs': {
            'n_scatter_tasks': len(tasks),
            'total_files_processed': total_files,
            'total_cells_processed': total_cells,
        },
        'outputs': {
            'n_universe_genes': n_genes,
            'n_genes_with_median': int((median_norm > 0).sum()),
            'n_genes_pass_filter': n_pass,
            'n_genes_excluded_by_filter': n_genes - n_pass,
            'primary_deliverable': 'rat_gene_medians.pickle',
        },
        'biotype_summary': biotype_summary,
        'output_files': {
            'rat_gene_medians.pickle': str(pickle_path),
            'gene_median_stats.tsv': str(tsv_path),
            'gene_median_stats.npz': str(npz_path),
            'biotype_summary.json': str(bt_path),
            'cell_stats_summary.json': str(cs_path),
        },
        'scatter_dir': str(scatter_dir),
        'note': (
            "rat_gene_medians.pickle format: {ensembl_id: float32_median_normalized_expr}. "
            "Medians are computed on normalize_total(target_sum=10000) values (non-zero cells only). "
            "Matches GeneCompass human_gene_median_after_filter.pickle format. "
            "Stage 5 uses this to convert raw counts → median-normalized → ranked tokens."
        ),
    }
    mfst_path = output_dir / 'stage4_manifest.json'
    with open(mfst_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"  stage4_manifest.json")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 5: MAIN PIPELINE
# ═════════════════════════════════════════════════════════════════════════════

def gather_gene_medians(config: dict, gene_batch_size: int, dry_run: bool = False):
    """Main gather pipeline: discover → aggregate stats → exact medians → write."""
    t_start = time.time()

    paths = config['paths']
    med_cfg = config.get('medians', {})
    min_nonzero_cells = int(med_cfg.get('min_nonzero_cells', 10))

    ortholog_dir = resolve_path(config, paths['ortholog_dir'])
    median_dir = resolve_path(config, paths['median_dir'])
    scatter_dir = median_dir / 'scatter'

    logger.info("=" * 70)
    logger.info("STAGE 4 GATHER — Merge Scatter → Final Median Dictionary")
    logger.info("=" * 70)
    logger.info(f"  Scatter dir:          {scatter_dir}")
    logger.info(f"  Output dir:           {median_dir}")
    logger.info(f"  min_nonzero_cells:    {min_nonzero_cells}")
    logger.info(f"  gene_batch_size:      {gene_batch_size:,}")
    logger.info(f"  Dry run:              {dry_run}")
    logger.info("")

    # ── Load gene universe ────────────────────────────────────────────────────
    logger.info("Loading gene universe from Stage 3...")
    gene_ids, gene_biotype = load_gene_universe(ortholog_dir)
    n_genes = len(gene_ids)
    logger.info(f"  Gene universe: {n_genes:,} genes")

    if dry_run:
        logger.info("")
        logger.info("DRY RUN — config and gene universe validated")
        logger.info(f"  Would aggregate scatter outputs from: {scatter_dir}")
        logger.info(f"  Would process {n_genes:,} genes in batches of {gene_batch_size:,}")
        logger.info(f"  Primary output: {median_dir / 'rat_gene_medians.pickle'}")
        return

    # ── Discover scatter files ─────────────────────────────────────────────────
    logger.info("")
    logger.info("Discovering scatter task outputs...")
    tasks = discover_scatter_files(scatter_dir)

    # Validate first scatter file's gene universe size
    if tasks:
        if not validate_scatter_gene_universe(tasks[0]['stats_path'], n_genes):
            sys.exit(1)

    total_cells = sum(t['n_cells'] for t in tasks)
    total_files = sum(t['n_files'] for t in tasks)
    logger.info(f"  Total cells covered:  {total_cells:,}")
    logger.info(f"  Total files covered:  {total_files:,}")

    # ── Phase 1: Aggregate running stats ──────────────────────────────────────
    logger.info("")
    global_stats = aggregate_stats(tasks, n_genes)
    derived = compute_derived_stats(global_stats)

    # ── Phase 2: Exact medians and percentiles ────────────────────────────────
    logger.info("")
    median_data = compute_exact_medians(tasks, n_genes, gene_batch_size)

    # ── Phase 3: Write outputs ────────────────────────────────────────────────
    logger.info("")
    logger.info("Phase 3: Writing outputs...")
    write_outputs(
        output_dir=median_dir,
        scatter_dir=scatter_dir,
        gene_ids=gene_ids,
        gene_biotype=gene_biotype,
        global_stats=global_stats,
        derived=derived,
        median_data=median_data,
        tasks=tasks,
        min_nonzero_cells=min_nonzero_cells,
        config=config,
        t_start=t_start,
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    n_with_median = int((median_data['median_norm'] > 0).sum())
    n_nonzero = global_stats['n_nonzero_norm']
    n_pass = int((n_nonzero >= min_nonzero_cells).sum())

    logger.info("")
    logger.info("=" * 70)
    logger.info("STAGE 4 GATHER COMPLETE")
    logger.info("=" * 70)
    logger.info(f"  Gene universe:          {n_genes:,}")
    logger.info(f"  Genes with median > 0:  {n_with_median:,}  "
                f"({n_with_median / n_genes:.1%})")
    logger.info(f"  Genes pass filter       {n_pass:,}  "
                f"(≥{min_nonzero_cells} non-zero cells)")
    logger.info(f"  Total cells processed:  {total_cells:,}")
    logger.info(f"  Total files processed:  {total_files:,}")
    logger.info(f"  Elapsed:                {elapsed:.1f}s")
    logger.info(f"  Output:                 {median_dir}")
    logger.info("")
    logger.info("  Primary deliverable:    rat_gene_medians.pickle")
    logger.info("  Downstream:")
    logger.info("    Stage 5 → reads rat_gene_medians.pickle for tokenization")
    logger.info("    Aim 2  → gene_median_stats.tsv for expression QC")
    logger.info("    Aim 3  → zero_fraction_norm, biotype for cross-species flags")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Stage 4 Gather: Merge scatter outputs → rat_gene_medians.pickle",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Run after all scatter tasks complete:
  python gather_gene_medians.py
  python gather_gene_medians.py --dry-run
  python gather_gene_medians.py --gene-batch-size 500   # less RAM, slower

Memory guidance:
  Peak RAM ≈ gene_batch_size × (avg non-zero cells per gene) × 4 bytes
  With gene_batch_size=2500 and 1M non-zero cells per gene ≈ 10 GB.
  Default 128 GB SLURM request comfortably handles batch_size=2500.
  Reduce to 1000 if you see OOM errors.
        """,
    )
    parser.add_argument('--gene-batch-size', type=int, default=2500,
                        help='Genes per median-computation batch (default: 2500). '
                             'Reduce to lower peak RAM.')
    parser.add_argument('--dry-run', action='store_true',
                        help='Validate scatter outputs and exit')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='DEBUG-level logging')
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        config = load_config()
    except FileNotFoundError:
        logger.error("pipeline_config.yaml not found. "
                      "Set PIPELINE_ROOT or run from project root.")
        sys.exit(1)

    for section in ('medians', 'paths'):
        if section not in config:
            logger.error(f"Config missing '{section}' section")
            sys.exit(1)

    gather_gene_medians(
        config=config,
        gene_batch_size=args.gene_batch_size,
        dry_run=args.dry_run,
    )


if __name__ == '__main__':
    main()