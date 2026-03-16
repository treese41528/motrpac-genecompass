#!/usr/bin/env python3
"""
compute_gene_medians.py — Stage 4, Scatter Phase: Per-Gene Stats over h5ad Batch

Pipeline position:
    Stage 1: Data Harvesting & QC
    Stage 2: Gene Universe & Cell QC
      Step 1: build_gene_universe.py
      Step 2: preprocess_training_matrices.py  → QC'd h5ad (raw counts)
      Step 3: prune_gene_universe.py
    Stage 3: build_ortholog_mapping.py         → rat_token_mapping.tsv
    Stage 4: SCATTER → compute_gene_medians.py  ← THIS SCRIPT
             GATHER  → gather_gene_medians.py
    Stage 5: Reference Assembly & Tokenization

Purpose:
    SLURM array scatter worker. Each task processes a disjoint subset of the
    QC'd h5ad corpus. For every file in its batch the script:
      1. Loads the h5ad fully into memory (defensive against Gilbreth NFS).
      2. Maps var_names to the Stage 3 gene universe (versioned ID stripping,
         lookup into sorted index).
      3. Computes per-cell library sizes (raw counts → stored as cell stats).
      4. Applies normalize_total (scale each cell to target_sum=10,000) in
         memory only — raw counts are NEVER modified on disk.
      5. Accumulates, per-gene:
           - Non-zero normalized values (for exact median in gather phase)
           - Non-zero raw values (for raw-count baseline)
           - Running sums (sum, sum_sq, n_nonzero, n_cells) for mean/var/std
      6. Writes a compact CSR-style .npz (values + offsets) and a stats .npz.

Outputs (written to median_dir/scatter/scatter_{task_id:04d}*.{npz,json}):
    scatter_{id}_stats.npz     — Per-gene running stats arrays (shape: n_genes)
    scatter_{id}_nonzero.npz   — Non-zero values, CSR-layout (gene_offsets +
                                  nonzero_values_norm + nonzero_values_raw)
    scatter_{id}_cells.tsv.gz  — Per-file cell-level stats (lib size, n_genes)
    scatter_{id}_manifest.json — Provenance (files processed, cells seen, runtime)

Memory notes:
    Each task accumulates n_nonzero_total float32 pairs (norm + raw). With
    10.2M cells / N_tasks, ~10% non-zero, this is ≈1–2 GB per task at N=50.
    The gather script concatenates across tasks; 128 GB node headroom is fine.

Raw-count contract:
    This script NEVER writes normalized data back to disk.
    The QC'd h5ad files are opened read-only. Normalization is ephemeral.

Usage (direct):
    python compute_gene_medians.py --task-id 0 --n-tasks 50
    python compute_gene_medians.py --task-id 0 --n-tasks 50 --dry-run

Usage (SLURM — called by stage4_scatter.slurm):
    python compute_gene_medians.py --task-id $SLURM_ARRAY_TASK_ID \\
                                   --n-tasks $N_SCATTER_TASKS

Author: Tim Reese Lab / Claude
Date: March 2026
"""

import argparse
import csv
import gzip
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Optional heavy imports (checked at startup)
# ─────────────────────────────────────────────────────────────────────────────
_MISSING = []
try:
    import scipy.sparse as sp
except ImportError:
    _MISSING.append('scipy')
    sp = None

try:
    import anndata as ad
except ImportError:
    _MISSING.append('anndata')
    ad = None

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
# SECTION 1: GENE UNIVERSE LOADING
# ═════════════════════════════════════════════════════════════════════════════

def load_gene_universe(ortholog_dir: Path) -> Tuple[List[str], Dict[str, int], Dict[str, str]]:
    """Load Stage 3 gene universe: only non-excluded genes, sorted for stable indexing.

    Returns:
        gene_ids     — sorted list of ENSRNOG IDs (defines integer index 0..N-1)
        gene_to_idx  — {ensrnog: int_index}  (includes version-stripped lookup)
        gene_biotype — {ensrnog: biotype_string}
    """
    mapping_path = ortholog_dir / 'rat_token_mapping.tsv'
    if not mapping_path.exists():
        raise FileNotFoundError(
            f"rat_token_mapping.tsv not found: {mapping_path}\n"
            f"  → Run Stage 3 (python run_stage3.py) first."
        )

    genes = {}
    n_excluded = 0
    with open(mapping_path, 'r') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            tier = row.get('tier', '').strip()
            if tier == 'excluded':
                n_excluded += 1
                continue
            eid = row['rat_gene'].strip()
            if not eid:
                continue
            # Strip version suffix if present (e.g. ENSRNOG00000046319.4 → ENSRNOG00000046319)
            base_eid = eid.split('.')[0].upper()
            genes[base_eid] = row.get('biotype', '').strip()

    gene_ids = sorted(genes.keys())
    gene_to_idx = {eid: i for i, eid in enumerate(gene_ids)}
    gene_biotype = {eid: genes[eid] for eid in gene_ids}

    logger.info(f"Gene universe: {len(gene_ids):,} genes loaded from {mapping_path.name}")
    logger.info(f"  Excluded (no ortholog + wrong biotype): {n_excluded:,}")
    return gene_ids, gene_to_idx, gene_biotype


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2: H5AD FILE DISCOVERY AND TASK SPLITTING
# ═════════════════════════════════════════════════════════════════════════════

def discover_h5ad_files(qc_h5ad_dir: Path) -> List[Path]:
    """Collect all QC'd h5ad files, sorted for reproducible task splitting."""
    files = sorted(qc_h5ad_dir.glob('**/*.h5ad'))
    if not files:
        raise FileNotFoundError(
            f"No .h5ad files found in {qc_h5ad_dir}\n"
            f"  → Run Stage 2 (python run_stage2.py) first."
        )
    return files


def get_task_files(all_files: List[Path], task_id: int, n_tasks: int) -> List[Path]:
    """Return the slice of files assigned to this SLURM array task.

    Files are pre-sorted, so each task gets a deterministic, non-overlapping
    subset: task i handles indices [i, i+n_tasks, i+2*n_tasks, ...].
    Interleaving (stride) rather than contiguous blocks balances file-size
    variance across tasks when large datasets cluster by accession prefix.
    """
    return [f for j, f in enumerate(all_files) if j % n_tasks == task_id]


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3: PER-FILE PROCESSING
# ═════════════════════════════════════════════════════════════════════════════

def map_varnames_to_universe(
    var_names: List[str],
    gene_to_idx: Dict[str, int],
) -> Tuple[List[int], List[int]]:
    """Map h5ad var_names to gene universe integer indices.

    Strips version suffixes (e.g. ENSRNOG00000046319.4 → ENSRNOG00000046319)
    before lookup. This is the canonical fix for artificial gene uniqueness
    from versioned IDs documented in Stage 2.

    Returns:
        col_indices  — local column indices in adata (which columns are valid)
        gene_indices — corresponding gene universe integer indices
    """
    col_indices = []
    gene_indices = []
    for local_j, raw_name in enumerate(var_names):
        base = raw_name.strip().split('.')[0].upper()
        global_j = gene_to_idx.get(base)
        if global_j is not None:
            col_indices.append(local_j)
            gene_indices.append(global_j)
    return col_indices, gene_indices


def normalize_total_inplace_sparse(X_raw_csr, target_sum: float = 1e4) -> 'sp.csr_matrix':
    """Apply normalize_total (CPM-like scaling) to a raw count CSR matrix.

    Scales each cell (row) so that its total counts equal target_sum.
    Cells with zero library size are left as zero (not scaled).

    This normalization is EPHEMERAL — the result is returned, never written
    back to disk. Raw counts in X_raw_csr are untouched.

    Matches GeneCompass preprocessing: sc.pp.normalize_total(target_sum=1e4).
    """
    lib_sizes = np.asarray(X_raw_csr.sum(axis=1), dtype=np.float64).ravel()
    scale = np.where(lib_sizes > 0, target_sum / lib_sizes, 0.0).astype(np.float32)

    # Efficient row-wise scaling for sparse matrix:
    # diags(scale) @ X  =  scale[i] * X[i,:] for each row i
    D = sp.diags(scale, format='csr')
    X_norm = D.dot(X_raw_csr)
    return X_norm.astype(np.float32)


def process_file(
    h5ad_path: Path,
    gene_to_idx: Dict[str, int],
    n_universe_genes: int,
    target_sum: float,
    # Accumulators (modified in-place):
    sum_norm: np.ndarray,
    sum_sq_norm: np.ndarray,
    sum_raw: np.ndarray,
    sum_sq_raw: np.ndarray,
    n_nonzero_norm: np.ndarray,
    n_nonzero_raw: np.ndarray,
    n_cells_total: np.ndarray,
    # Per-gene value buffers (list of float32 arrays, extended in-place):
    nonzero_norm_bufs: List[List],
    nonzero_raw_bufs: List[List],
) -> Dict:
    """Process one h5ad file and update all accumulators.

    Reads the h5ad fully into memory before any processing (defensive against
    NFS stream corruption on Gilbreth). Returns per-file metadata dict.

    Returns: file_meta dict with n_cells, mapping_rate, lib_size stats, etc.
    """
    t0 = time.time()

    # ── Load fully into memory ──────────────────────────────────────────────
    logger.debug(f"  Loading: {h5ad_path.name}")
    try:
        adata = ad.read_h5ad(str(h5ad_path))
    except Exception as e:
        logger.error(f"  Failed to load {h5ad_path.name}: {e}")
        return {'file': h5ad_path.name, 'status': 'load_error', 'error': str(e)}

    n_cells_orig = adata.n_obs
    n_genes_orig = adata.n_vars

    # ── Map var_names to universe ────────────────────────────────────────────
    col_indices, gene_indices = map_varnames_to_universe(
        list(adata.var_names), gene_to_idx
    )
    n_mapped = len(col_indices)
    mapping_rate = n_mapped / n_genes_orig if n_genes_orig > 0 else 0.0

    if n_mapped == 0:
        logger.warning(f"  {h5ad_path.name}: 0 genes mapped to universe — skipping")
        return {
            'file': h5ad_path.name, 'status': 'no_genes_mapped',
            'n_cells': n_cells_orig, 'n_genes_orig': n_genes_orig,
            'n_mapped': 0, 'mapping_rate': 0.0,
        }

    # ── Subset to mapped genes ───────────────────────────────────────────────
    X_raw = adata.X[:, col_indices]
    if not sp.issparse(X_raw):
        X_raw = sp.csr_matrix(X_raw)
    else:
        X_raw = X_raw.tocsr()
    X_raw = X_raw.astype(np.float32)

    # ── Per-cell raw stats ───────────────────────────────────────────────────
    lib_sizes_raw = np.asarray(X_raw.sum(axis=1), dtype=np.float32).ravel()
    n_genes_detected = np.asarray((X_raw > 0).sum(axis=1), dtype=np.int32).ravel()

    # ── Normalize in memory ──────────────────────────────────────────────────
    X_norm = normalize_total_inplace_sparse(X_raw, target_sum=target_sum)

    # ── Convert to CSC for efficient column (gene) access ────────────────────
    X_norm_csc = X_norm.tocsc()
    X_raw_csc = X_raw.tocsc()

    n_cells = X_raw.shape[0]

    # ── Accumulate per-gene statistics ───────────────────────────────────────
    # We iterate over local columns (mapped genes), reading CSC indptr slices.
    # This avoids scipy getcol() overhead for sequential column access.
    norm_data = X_norm_csc.data
    norm_indptr = X_norm_csc.indptr
    raw_data = X_raw_csc.data
    raw_indptr = X_raw_csc.indptr

    for local_j, global_j in enumerate(gene_indices):
        # Normalized non-zero values for this gene in this file
        ns = norm_indptr[local_j]
        ne = norm_indptr[local_j + 1]
        nz_norm = norm_data[ns:ne]

        # Raw non-zero values for this gene in this file
        rs = raw_indptr[local_j]
        re = raw_indptr[local_j + 1]
        nz_raw = raw_data[rs:re]

        # Running stats (normalized)
        cnt_norm = len(nz_norm)
        if cnt_norm > 0:
            sum_norm[global_j] += nz_norm.sum()
            sum_sq_norm[global_j] += (nz_norm.astype(np.float64) ** 2).sum()
            n_nonzero_norm[global_j] += cnt_norm
            nonzero_norm_bufs[global_j].append(nz_norm.copy())

        # Running stats (raw)
        cnt_raw = len(nz_raw)
        if cnt_raw > 0:
            sum_raw[global_j] += nz_raw.sum()
            sum_sq_raw[global_j] += (nz_raw.astype(np.float64) ** 2).sum()
            n_nonzero_raw[global_j] += cnt_raw
            nonzero_raw_bufs[global_j].append(nz_raw.copy())

        n_cells_total[global_j] += n_cells

    elapsed = time.time() - t0

    # ── Collect file-level metadata ──────────────────────────────────────────
    file_meta = {
        'file': h5ad_path.name,
        'status': 'ok',
        'n_cells': int(n_cells),
        'n_genes_orig': int(n_genes_orig),
        'n_mapped': int(n_mapped),
        'mapping_rate': round(mapping_rate, 4),
        'lib_size_min': float(lib_sizes_raw.min()) if n_cells > 0 else 0.0,
        'lib_size_max': float(lib_sizes_raw.max()) if n_cells > 0 else 0.0,
        'lib_size_mean': float(lib_sizes_raw.mean()) if n_cells > 0 else 0.0,
        'lib_size_median': float(np.median(lib_sizes_raw)) if n_cells > 0 else 0.0,
        'n_genes_min': int(n_genes_detected.min()) if n_cells > 0 else 0,
        'n_genes_max': int(n_genes_detected.max()) if n_cells > 0 else 0,
        'n_genes_mean': float(n_genes_detected.mean()) if n_cells > 0 else 0.0,
        'n_genes_median': float(np.median(n_genes_detected)) if n_cells > 0 else 0.0,
        'elapsed_s': round(elapsed, 2),
    }

    logger.debug(
        f"  {h5ad_path.name}: {n_cells:,} cells, {n_mapped:,}/{n_genes_orig:,} genes "
        f"({mapping_rate:.1%}) in {elapsed:.1f}s"
    )
    return file_meta


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4: OUTPUT WRITING
# ═════════════════════════════════════════════════════════════════════════════

def write_scatter_outputs(
    task_id: int,
    scatter_dir: Path,
    gene_ids: List[str],
    n_universe_genes: int,
    sum_norm: np.ndarray,
    sum_sq_norm: np.ndarray,
    sum_raw: np.ndarray,
    sum_sq_raw: np.ndarray,
    n_nonzero_norm: np.ndarray,
    n_nonzero_raw: np.ndarray,
    n_cells_total: np.ndarray,
    nonzero_norm_bufs: List[List],
    nonzero_raw_bufs: List[List],
    file_metas: List[Dict],
    task_start_time: float,
    args_ns,
) -> None:
    """Write all scatter outputs for this task.

    Output format:
        scatter_{id}_stats.npz   — per-gene running stats (fixed-size arrays)
        scatter_{id}_nonzero.npz — non-zero values in CSR layout (variable-size)
        scatter_{id}_cells.tsv.gz — per-file cell-level stats
        scatter_{id}_manifest.json — provenance

    The nonzero .npz uses a CSR-like layout:
        gene_offsets_norm[g] : gene_offsets_norm[g+1] → slice into nonzero_values_norm
        gene_offsets_raw[g]  : gene_offsets_raw[g+1]  → slice into nonzero_values_raw
    This lets the gather script access any gene's values with two array reads.
    """
    scatter_dir.mkdir(parents=True, exist_ok=True)
    pfx = scatter_dir / f"scatter_{task_id:04d}"

    # ── Build CSR-style value arrays ─────────────────────────────────────────
    # Concatenate per-gene buffer lists into flat arrays with offset index.
    logger.info(f"  Building CSR value arrays for {n_universe_genes:,} genes...")

    offsets_norm = np.zeros(n_universe_genes + 1, dtype=np.int64)
    offsets_raw = np.zeros(n_universe_genes + 1, dtype=np.int64)

    for g in range(n_universe_genes):
        offsets_norm[g + 1] = offsets_norm[g] + sum(len(a) for a in nonzero_norm_bufs[g])
        offsets_raw[g + 1] = offsets_raw[g] + sum(len(a) for a in nonzero_raw_bufs[g])

    total_nz_norm = int(offsets_norm[-1])
    total_nz_raw = int(offsets_raw[-1])

    logger.info(f"  Total non-zero entries: norm={total_nz_norm:,}, raw={total_nz_raw:,}")

    values_norm = np.empty(total_nz_norm, dtype=np.float32)
    values_raw = np.empty(total_nz_raw, dtype=np.float32)

    for g in range(n_universe_genes):
        if nonzero_norm_bufs[g]:
            start = offsets_norm[g]
            end = offsets_norm[g + 1]
            values_norm[start:end] = np.concatenate(nonzero_norm_bufs[g])
        if nonzero_raw_bufs[g]:
            start = offsets_raw[g]
            end = offsets_raw[g + 1]
            values_raw[start:end] = np.concatenate(nonzero_raw_bufs[g])

    # ── Write stats .npz ─────────────────────────────────────────────────────
    stats_path = Path(str(pfx) + '_stats.npz')
    np.savez_compressed(
        stats_path,
        # Gene IDs for cross-validation (stored as bytes)
        gene_ids=np.array([g.encode('ascii') for g in gene_ids], dtype=object),
        # Running stats
        sum_norm=sum_norm,
        sum_sq_norm=sum_sq_norm,
        sum_raw=sum_raw,
        sum_sq_raw=sum_sq_raw,
        n_nonzero_norm=n_nonzero_norm,
        n_nonzero_raw=n_nonzero_raw,
        n_cells_total=n_cells_total,
    )
    logger.info(f"  Stats: {stats_path.name}  ({stats_path.stat().st_size / 1e6:.1f} MB)")

    # ── Write nonzero values .npz ─────────────────────────────────────────────
    nonzero_path = Path(str(pfx) + '_nonzero.npz')
    np.savez_compressed(
        nonzero_path,
        gene_offsets_norm=offsets_norm,
        nonzero_values_norm=values_norm,
        gene_offsets_raw=offsets_raw,
        nonzero_values_raw=values_raw,
    )
    logger.info(f"  Nonzero: {nonzero_path.name}  ({nonzero_path.stat().st_size / 1e6:.1f} MB)")

    # ── Write per-file cell stats (gzip TSV) ─────────────────────────────────
    cells_path = Path(str(pfx) + '_cells.tsv.gz')
    ok_metas = [m for m in file_metas if m.get('status') == 'ok']
    if ok_metas:
        fieldnames = [
            'file', 'n_cells', 'n_genes_orig', 'n_mapped', 'mapping_rate',
            'lib_size_min', 'lib_size_max', 'lib_size_mean', 'lib_size_median',
            'n_genes_min', 'n_genes_max', 'n_genes_mean', 'n_genes_median',
            'elapsed_s',
        ]
        with gzip.open(cells_path, 'wt', newline='') as gz:
            writer = csv.DictWriter(gz, fieldnames=fieldnames, delimiter='\t',
                                    extrasaction='ignore')
            writer.writeheader()
            writer.writerows(ok_metas)
        logger.info(f"  Cell stats: {cells_path.name}")

    # ── Write manifest ────────────────────────────────────────────────────────
    n_ok = sum(1 for m in file_metas if m.get('status') == 'ok')
    n_err = len(file_metas) - n_ok
    n_cells_processed = sum(m.get('n_cells', 0) for m in file_metas
                            if m.get('status') == 'ok')

    manifest = {
        'stage': 4,
        'phase': 'scatter',
        'task_id': task_id,
        'n_tasks': args_ns.n_tasks,
        'generated_at': datetime.utcnow().isoformat() + 'Z',
        'elapsed_s': round(time.time() - task_start_time, 1),
        'n_files_processed': n_ok,
        'n_files_error': n_err,
        'n_cells_processed': n_cells_processed,
        'n_universe_genes': n_universe_genes,
        'total_nz_norm': total_nz_norm,
        'total_nz_raw': total_nz_raw,
        'target_sum': args_ns.target_sum,
        'outputs': {
            'stats': str(stats_path.name),
            'nonzero': str(nonzero_path.name),
            'cells': str(cells_path.name) if ok_metas else None,
        },
        'file_statuses': [
            {'file': m['file'], 'status': m.get('status', 'unknown')}
            for m in file_metas
        ],
    }
    manifest_path = Path(str(pfx) + '_manifest.json')
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"  Manifest: {manifest_path.name}")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 5: MAIN PIPELINE
# ═════════════════════════════════════════════════════════════════════════════

def compute_gene_medians_scatter(config: dict, task_id: int, n_tasks: int,
                                  dry_run: bool = False):
    """Main scatter pipeline: load → split → process batch → write outputs."""
    t_start = time.time()

    paths = config['paths']
    med_cfg = config.get('medians', {})

    target_sum = float(med_cfg.get('target_sum', 1e4))
    qc_h5ad_dir = resolve_path(config, paths['qc_h5ad_dir'])
    ortholog_dir = resolve_path(config, paths['ortholog_dir'])
    median_dir = resolve_path(config, paths['median_dir'])
    scatter_dir = median_dir / 'scatter'

    logger.info("=" * 70)
    logger.info("STAGE 4 SCATTER — Gene Median Computation")
    logger.info("=" * 70)
    logger.info(f"  Task {task_id + 1} of {n_tasks}")
    logger.info(f"  QC h5ad dir:   {qc_h5ad_dir}")
    logger.info(f"  Ortholog dir:  {ortholog_dir}")
    logger.info(f"  Scatter dir:   {scatter_dir}")
    logger.info(f"  target_sum:    {target_sum:,.0f}  (normalize_total CPM)")
    logger.info(f"  Dry run:       {dry_run}")
    logger.info("")

    # ── Load gene universe from Stage 3 ──────────────────────────────────────
    logger.info("Loading gene universe from Stage 3...")
    gene_ids, gene_to_idx, gene_biotype = load_gene_universe(ortholog_dir)
    n_genes = len(gene_ids)
    logger.info(f"  Gene universe: {n_genes:,} genes (non-excluded)")

    # ── Discover and split h5ad files ────────────────────────────────────────
    logger.info("")
    logger.info("Discovering QC'd h5ad files...")
    all_files = discover_h5ad_files(qc_h5ad_dir)
    task_files = get_task_files(all_files, task_id, n_tasks)
    logger.info(f"  Total files in corpus: {len(all_files):,}")
    logger.info(f"  Files for this task:   {len(task_files):,}")

    if not task_files:
        logger.warning(f"No files assigned to task {task_id} — nothing to do.")
        return

    if dry_run:
        logger.info("")
        logger.info("DRY RUN — inputs validated. Sample of assigned files:")
        for f in task_files[:5]:
            logger.info(f"  {f.name}")
        if len(task_files) > 5:
            logger.info(f"  ... and {len(task_files) - 5} more")
        return

    # ── Check for missing dependencies ───────────────────────────────────────
    if _MISSING:
        logger.error(f"Missing required packages: {_MISSING}")
        logger.error("Activate foundational_models_env conda environment.")
        sys.exit(1)

    # ── Initialise accumulator arrays ─────────────────────────────────────────
    # float64 for running sums (precision matters for large cell counts)
    sum_norm = np.zeros(n_genes, dtype=np.float64)
    sum_sq_norm = np.zeros(n_genes, dtype=np.float64)
    sum_raw = np.zeros(n_genes, dtype=np.float64)
    sum_sq_raw = np.zeros(n_genes, dtype=np.float64)
    n_nonzero_norm = np.zeros(n_genes, dtype=np.int64)
    n_nonzero_raw = np.zeros(n_genes, dtype=np.int64)
    n_cells_total = np.zeros(n_genes, dtype=np.int64)

    # Per-gene buffers for exact median (list of np.float32 arrays)
    nonzero_norm_bufs: List[List] = [[] for _ in range(n_genes)]
    nonzero_raw_bufs: List[List] = [[] for _ in range(n_genes)]

    # ── Process each file ────────────────────────────────────────────────────
    logger.info("")
    logger.info(f"Processing {len(task_files):,} files...")
    file_metas = []

    for i, h5ad_path in enumerate(task_files):
        logger.info(f"  [{i + 1:3d}/{len(task_files):3d}] {h5ad_path.name}")
        meta = process_file(
            h5ad_path=h5ad_path,
            gene_to_idx=gene_to_idx,
            n_universe_genes=n_genes,
            target_sum=target_sum,
            sum_norm=sum_norm,
            sum_sq_norm=sum_sq_norm,
            sum_raw=sum_raw,
            sum_sq_raw=sum_sq_raw,
            n_nonzero_norm=n_nonzero_norm,
            n_nonzero_raw=n_nonzero_raw,
            n_cells_total=n_cells_total,
            nonzero_norm_bufs=nonzero_norm_bufs,
            nonzero_raw_bufs=nonzero_raw_bufs,
        )
        file_metas.append(meta)

        if meta.get('status') == 'ok':
            logger.info(
                f"    {meta['n_cells']:>7,} cells | "
                f"{meta['n_mapped']:,}/{meta['n_genes_orig']:,} genes "
                f"({meta['mapping_rate']:.1%}) | "
                f"lib_size median={meta['lib_size_median']:.0f} | "
                f"{meta['elapsed_s']:.1f}s"
            )
        else:
            logger.warning(f"    Status: {meta.get('status')} — {meta.get('error', '')}")

    # ── Write outputs ─────────────────────────────────────────────────────────
    logger.info("")
    logger.info("Writing scatter outputs...")
    write_scatter_outputs(
        task_id=task_id,
        scatter_dir=scatter_dir,
        gene_ids=gene_ids,
        n_universe_genes=n_genes,
        sum_norm=sum_norm,
        sum_sq_norm=sum_sq_norm,
        sum_raw=sum_raw,
        sum_sq_raw=sum_sq_raw,
        n_nonzero_norm=n_nonzero_norm,
        n_nonzero_raw=n_nonzero_raw,
        n_cells_total=n_cells_total,
        nonzero_norm_bufs=nonzero_norm_bufs,
        nonzero_raw_bufs=nonzero_raw_bufs,
        file_metas=file_metas,
        task_start_time=t_start,
        args_ns=type('A', (), {'n_tasks': n_tasks, 'target_sum': target_sum})(),
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    n_ok = sum(1 for m in file_metas if m.get('status') == 'ok')
    n_cells_total_seen = sum(m.get('n_cells', 0) for m in file_metas
                              if m.get('status') == 'ok')

    logger.info("")
    logger.info("=" * 70)
    logger.info(f"TASK {task_id} COMPLETE")
    logger.info("=" * 70)
    logger.info(f"  Files processed:   {n_ok:,} / {len(task_files):,}")
    logger.info(f"  Cells processed:   {n_cells_total_seen:,}")
    logger.info(f"  Elapsed:           {elapsed:.1f}s")
    logger.info(f"  Scatter dir:       {scatter_dir}")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Stage 4 Scatter: Compute per-gene stats for a batch of h5ad files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Called by the SLURM array job (stage4_scatter.slurm):
  --task-id    $SLURM_ARRAY_TASK_ID   (0-indexed)
  --n-tasks    $N_SCATTER_TASKS        (matches #SBATCH --array=0-(N-1))

For local testing (serial):
  python compute_gene_medians.py --task-id 0 --n-tasks 1
  python compute_gene_medians.py --task-id 0 --n-tasks 50 --dry-run

Memory budget:
  Each task holds ~1-2 GB of per-gene value buffers (float32).
  With N=50 tasks and 10.2M cells, each task processes ~204K cells.
  Request --mem=64G in SLURM to be safe.
        """,
    )
    parser.add_argument('--task-id', type=int, required=True,
                        help='0-indexed task ID (= $SLURM_ARRAY_TASK_ID)')
    parser.add_argument('--n-tasks', type=int, required=True,
                        help='Total number of scatter tasks (= SLURM array size)')
    parser.add_argument('--target-sum', type=float, default=None,
                        help='normalize_total target (overrides config; default: 10000)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Validate inputs and exit without processing')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='DEBUG-level logging')
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.task_id < 0 or args.task_id >= args.n_tasks:
        logger.error(f"--task-id {args.task_id} out of range [0, {args.n_tasks})")
        sys.exit(1)

    try:
        config = load_config()
    except FileNotFoundError:
        logger.error("pipeline_config.yaml not found. "
                      "Set PIPELINE_ROOT or run from project root.")
        sys.exit(1)

    # Allow command-line override of target_sum
    if args.target_sum is not None:
        if 'medians' not in config:
            config['medians'] = {}
        config['medians']['target_sum'] = args.target_sum

    for section in ('medians', 'paths'):
        if section not in config:
            logger.error(f"Config missing '{section}' section")
            sys.exit(1)

    compute_gene_medians_scatter(
        config=config,
        task_id=args.task_id,
        n_tasks=args.n_tasks,
        dry_run=args.dry_run,
    )


if __name__ == '__main__':
    main()