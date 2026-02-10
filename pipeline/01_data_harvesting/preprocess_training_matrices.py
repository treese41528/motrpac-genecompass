#!/usr/bin/env python3
"""
preprocess_training_matrices.py — Stage 3: Cell QC + Normalize + Rank → Pruned Gene List

Pipeline position:
    Stage 1: BioMart reference fetch
    Stage 2: Gene inventory (biotype-filtered)            ← done
    Stage 3: Cell QC + normalize + rank → pruned genes    ← THIS SCRIPT
    Stage 4: Ortholog mapping (pruned gene list)
    Stage 5: Corpus export + GeneCompass formatting

Purpose:
    Read raw count matrices from the training corpus, apply GeneCompass-style QC,
    normalize per cell, rank genes by expression, and determine which genes ever
    appear in a cell's top-2048. Produces:
      (a) rat_ensembl_ids_pruned.txt — genes that survive expression filtering
      (b) QC'd h5ad files per study — ready for downstream training export
      (c) preprocessing_report.json — statistics for every study

Design rationale:
    GeneCompass's embedding vocabulary maps 1:1 to its token dictionary. Every gene
    that gets a token costs 768 dimensions in the embedding matrix. Genes that never
    appear in any cell's top 2,048 expressions waste parameters. We cannot determine
    this without per-cell normalization (which changes rank order), so QC, normalization,
    and pruning MUST happen in the same pass.

Aligns with GeneCompass preprocessing (Cell Research 2024):
    1. Filter cells with < 200 expressed genes
    2. Filter samples with < 4 cells
    3. Filter cells with > 15% mitochondrial gene proportion
    4. Filter cells with gene count > 3 SD above sample mean
    5. Drop genes not in core gene list (protein_coding, lncRNA, miRNA)
    6. Normalize, rank, select top 2048 genes per cell

Matrix format handling:
    The corpus contains heterogeneous formats. This script detects and standardizes:
    - MTX triplets (10x filtered + unfiltered)
    - h5ad (AnnData, may be raw or normalized)
    - 10x HDF5 (.h5)
    - Loom
    - TSV/CSV (dense count matrices)

Usage:
    python preprocess_training_matrices.py \\
        --data-root /depot/reese18/data \\
        --gene-inventory /depot/reese18/data/training/gene_inventory/gene_inventory.tsv \\
        --rat-ensembl-ids /depot/reese18/data/training/gene_inventory/rat_ensembl_ids.txt \\
        --rat-reference /depot/reese18/data/references/biomart/rat_symbol_lookup.pickle \\
        -o /depot/reese18/data/training/preprocessed/ \\
        --workers 4 \\
        -v

    # SLURM wrapper (recommended):
    sbatch preprocess_training_matrices.slurm

Author: Tim Reese Lab / Claude  
Date: February 2026
"""

import os
import sys
import gzip
import json
import pickle
import argparse
import logging
import re
import time
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple, Any
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

import pandas as pd
    
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Optional imports — fail gracefully with clear messages
# ─────────────────────────────────────────────────────────────────────────────
MISSING_DEPS = []

try:
    import numpy as np
except ImportError:
    MISSING_DEPS.append('numpy')
    np = None

try:
    from scipy import io as scipy_io
    from scipy import sparse as sp
except ImportError:
    MISSING_DEPS.append('scipy')
    scipy_io = None
    sp = None

try:
    import h5py
except ImportError:
    MISSING_DEPS.append('h5py')
    h5py = None

try:
    import anndata as ad
except ImportError:
    MISSING_DEPS.append('anndata')
    ad = None

try:
    import scanpy as sc
except ImportError:
    MISSING_DEPS.append('scanpy')
    sc = None

try:
    import pandas as pd
except ImportError:
    MISSING_DEPS.append('pandas')
    pd = None


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# GeneCompass-style QC thresholds
MIN_GENES_PER_CELL = 200       # Drop cells with < 200 expressed genes
MIN_CELLS_PER_SAMPLE = 4       # Drop samples with < 4 cells after filtering
MAX_MITO_FRACTION = 0.15       # Drop cells with > 15% mitochondrial reads
OUTLIER_SD_THRESHOLD = 3.0     # Drop cells > 3 SD above mean gene count
TOP_N_GENES = 2048             # Number of top-expressed genes per cell

# Unfiltered 10x barcode space: these exact counts indicate raw/unfiltered data
UNFILTERED_BARCODE_COUNTS = {
    6794880,   # 10x v3 (standard)
    737280,    # 10x v2
    2097152,   # Some custom setups
}

# Maximum cells per barcode dimension to consider "filtered"
# If barcode dim > this and matches known unfiltered counts, treat as unfiltered
MAX_FILTERED_BARCODES = 500_000

# Rat mitochondrial gene patterns (for mito fraction calculation)
RAT_MITO_PATTERNS = [
    re.compile(r'^mt-', re.IGNORECASE),           # mt-Nd1, mt-Co1, etc.
    re.compile(r'^ENSRNOG\d+.*mt', re.IGNORECASE), # Ensembl mito genes
    re.compile(r'^Mt_', re.IGNORECASE),             # Mt_rRNA, Mt_tRNA (if not filtered by biotype)
]

# Patterns indicating a file is likely a count matrix (not metadata)
MATRIX_FILE_PATTERNS = [
    'matrix', 'counts', 'expression', 'umi', 'raw_count',
    'filtered_feature_bc_matrix', 'raw_feature_bc_matrix',
    'gene_expression', 'dgematrix', 'countmatrix',
]

# File patterns to skip entirely
SKIP_FILE_PATTERNS = [
    'barcodes', 'genes', 'features', 'metadata', 'clusters',
    'umap', 'tsne', 'pca', 'neighbors', 'annotation',
    'peaks', 'fragments', 'atac',  # scATAC files
    'samples', 'sra_runs', 'sample_files',  # GEO metadata tables
    'family.soft', 'miniml', 'series_matrix',  # GEO metadata formats
    'manifest', 'filelist', 'readme', 'changelog',
    'sdrf', 'idf', 'ena_runs',  # ArrayExpress metadata
]

# Supported matrix extensions (ordered by preference)
MATRIX_EXTENSIONS = ['.h5ad', '.h5', '.hdf5', '.loom', '.mtx.gz', '.mtx',
                     '.tsv.gz', '.csv.gz', '.tsv', '.csv', '.txt.gz', '.txt']


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1: FORMAT DETECTION AND MATRIX LOADING
# ═════════════════════════════════════════════════════════════════════════════

class MatrixLoadResult:
    """Container for a loaded expression matrix with metadata."""
    def __init__(self, adata: 'ad.AnnData', format_type: str, source_path: str,
                 is_raw_counts: bool, was_unfiltered: bool = False,
                 gene_id_format: str = 'unknown', warnings: List[str] = None):
        self.adata = adata
        self.format_type = format_type        # 'h5ad', 'mtx', 'h5', 'loom', 'tsv', 'csv'
        self.source_path = source_path
        self.is_raw_counts = is_raw_counts
        self.was_unfiltered = was_unfiltered
        self.gene_id_format = gene_id_format  # 'ensembl', 'symbol', 'loc', 'mixed'
        self.warnings = warnings or []


def detect_gene_id_format(gene_ids: List[str], sample_size: int = 100) -> str:
    """Classify the predominant gene ID format in a list.

    Returns one of: 'ensembl_rat', 'ensembl_mouse', 'ensembl_human',
                    'symbol', 'loc', 'ncbi', 'mixed', 'unknown'
    """
    if not gene_ids:
        return 'unknown'

    sample = gene_ids[:sample_size]
    counts = Counter()

    for gid in sample:
        gid_str = str(gid).strip()
        if gid_str.startswith('ENSRNOG'):
            counts['ensembl_rat'] += 1
        elif gid_str.startswith('ENSMUSG'):
            counts['ensembl_mouse'] += 1
        elif gid_str.startswith('ENSG'):
            counts['ensembl_human'] += 1
        elif gid_str.startswith('LOC') and gid_str[3:].isdigit():
            counts['loc'] += 1
        elif gid_str.isdigit():
            counts['ncbi'] += 1
        elif re.match(r'^[A-Z][a-z0-9]+$', gid_str) or re.match(r'^[A-Z][A-Za-z0-9]+$', gid_str):
            counts['symbol'] += 1
        else:
            counts['other'] += 1

    if not counts:
        return 'unknown'

    top_format, top_count = counts.most_common(1)[0]
    # If dominant format covers > 60% of sample, call it that
    if top_count / len(sample) > 0.6:
        return top_format
    return 'mixed'


def detect_is_raw_counts(X, sample_cells: int = 500) -> Tuple[bool, str]:
    """Heuristic: is this matrix likely raw integer counts?

    Strategy:
    - Sample up to `sample_cells` rows
    - Check if all nonzero values are integers (or very close)
    - If yes → raw counts
    - If no → normalized (could be log-transformed, TPM, FPKM, etc.)

    Returns (is_raw, evidence_string)
    """
    if sp is not None and sp.issparse(X):
        # For sparse matrices, check data array directly
        if hasattr(X, 'data') and len(X.data) > 0:
            sample_data = X.data[:min(100_000, len(X.data))]
        else:
            return True, "empty_sparse_matrix"
    else:
        # Dense matrix — sample rows
        n_cells = X.shape[0]
        idx = np.random.choice(n_cells, min(sample_cells, n_cells), replace=False)
        if sp is not None and sp.issparse(X):
            sample_data = np.asarray(X[idx].todense()).flatten()
        else:
            sample_data = np.asarray(X[idx]).flatten()
        sample_data = sample_data[sample_data != 0]  # only nonzero

    if len(sample_data) == 0:
        return True, "all_zeros"

    # Check if values are integers
    int_check = np.allclose(sample_data, np.round(sample_data), atol=1e-6)

    if int_check:
        max_val = float(np.max(sample_data))
        if max_val > 100:
            return True, f"integer_values_max={max_val:.0f}"
        else:
            # Small integers could be binned values (scGPT-style) — still treat as counts
            return True, f"small_integer_values_max={max_val:.0f}"
    else:
        # Check if it looks log-transformed (values typically 0-15 range)
        max_val = float(np.max(sample_data))
        min_nonzero = float(np.min(sample_data[sample_data > 0]))
        if max_val < 20 and min_nonzero < 1:
            return False, f"likely_log_transformed_range=[{min_nonzero:.3f}, {max_val:.3f}]"
        elif max_val > 1_000_000:
            return False, f"likely_TPM_or_FPKM_max={max_val:.1f}"
        else:
            return False, f"non_integer_values_range=[{min_nonzero:.4f}, {max_val:.2f}]"


def reverse_normalization(X) -> 'np.ndarray':
    """Attempt to recover approximate raw counts from normalized data.

    Following scFoundation's approach:
    "we treated the smallest nonzero value in the original matrix as a
     raw count value of 1, all remaining nonzero values were divided by
     this smallest value and the integer part was taken."

    Only call this when detect_is_raw_counts returns False.
    """
    if sp is not None and sp.issparse(X):
        X_dense = np.asarray(X.todense())
    else:
        X_dense = np.asarray(X)

    # Check if log-transformed (undo log1p first)
    max_val = np.max(X_dense)
    if max_val < 20:
        # Likely log1p transformed
        X_dense = np.expm1(X_dense)  # reverse of log1p

    # Find smallest nonzero per cell and normalize
    result = np.zeros_like(X_dense, dtype=np.int32)
    for i in range(X_dense.shape[0]):
        row = X_dense[i]
        nonzero = row[row > 0]
        if len(nonzero) == 0:
            continue
        min_val = np.min(nonzero)
        if min_val > 0:
            result[i] = np.floor(row / min_val).astype(np.int32)

    return sp.csr_matrix(result) if sp is not None else result


def detect_unfiltered_10x(n_barcodes: int) -> bool:
    """Check if a barcode count matches known unfiltered 10x barcode spaces."""
    if n_barcodes in UNFILTERED_BARCODE_COUNTS:
        return True
    if n_barcodes > MAX_FILTERED_BARCODES:
        return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Format-specific loaders
# ─────────────────────────────────────────────────────────────────────────────

def load_h5ad(filepath: str) -> Optional[MatrixLoadResult]:
    """Load an h5ad (AnnData) file.

    h5ad files may contain:
    - adata.X as raw counts (integer sparse/dense)
    - adata.X as normalized data with adata.raw.X as raw counts
    - adata.X as normalized data without raw layer
    - Sparse X stored as Group (CSR/CSC) rather than Dataset
    """
    try:
        adata = sc.read_h5ad(filepath, backed=None)
    except Exception as e:
        # Try with h5py directly for malformed h5ad files
        logger.warning(f"scanpy failed on {filepath}, trying h5py: {e}")
        try:
            adata = _load_h5ad_via_h5py(filepath)
            if adata is None:
                return None
        except Exception as e2:
            logger.error(f"h5py fallback also failed on {filepath}: {e2}")
            return None

    warnings = []

    # Determine gene IDs
    gene_ids = list(adata.var_names[:100])
    gene_id_format = detect_gene_id_format(gene_ids)

    # Check for raw counts
    is_raw, evidence = detect_is_raw_counts(adata.X)

    if not is_raw and adata.raw is not None:
        # Try the .raw layer
        is_raw_raw, evidence_raw = detect_is_raw_counts(adata.raw.X)
        if is_raw_raw:
            warnings.append(f"Using adata.raw.X (raw counts) instead of adata.X ({evidence})")
            adata = ad.AnnData(
                X=adata.raw.X,
                obs=adata.obs,
                var=adata.raw.var,
            )
            is_raw = True
            evidence = evidence_raw

    # Check for unfiltered data
    was_unfiltered = detect_unfiltered_10x(adata.n_obs)
    if was_unfiltered:
        warnings.append(f"Detected unfiltered barcode space: {adata.n_obs:,} barcodes")

    if not is_raw:
        warnings.append(f"Data appears normalized ({evidence}), will attempt reversal")

    return MatrixLoadResult(
        adata=adata,
        format_type='h5ad',
        source_path=filepath,
        is_raw_counts=is_raw,
        was_unfiltered=was_unfiltered,
        gene_id_format=gene_id_format,
        warnings=warnings,
    )


def _load_h5ad_via_h5py(filepath: str) -> Optional['ad.AnnData']:
    """Fallback loader for h5ad files with sparse X stored as HDF5 Groups.

    Some h5ad files store X as a Group with keys 'data', 'indices', 'indptr'
    rather than a Dataset, which older scanpy versions can't handle.
    """
    with h5py.File(filepath, 'r') as f:
        # Try to read X
        if 'X' not in f:
            logger.error(f"No 'X' key in {filepath}")
            return None

        x_obj = f['X']
        if isinstance(x_obj, h5py.Group):
            # Sparse matrix stored as group
            if all(k in x_obj for k in ('data', 'indices', 'indptr')):
                data = x_obj['data'][:]
                indices = x_obj['indices'][:]
                indptr = x_obj['indptr'][:]
                shape = tuple(x_obj.attrs.get('shape', None))
                if shape is None:
                    # Infer shape
                    n_obs = len(indptr) - 1
                    n_var = int(np.max(indices)) + 1 if len(indices) > 0 else 0
                    shape = (n_obs, n_var)
                X = sp.csr_matrix((data, indices, indptr), shape=shape)
            else:
                logger.error(f"Sparse group in {filepath} missing expected keys: {list(x_obj.keys())}")
                return None
        elif isinstance(x_obj, h5py.Dataset):
            X = x_obj[:]
        else:
            return None

        # Read var (gene names)
        var_index = None
        if 'var' in f and '_index' in f['var']:
            raw = f['var']['_index'][:]
            var_index = [x.decode() if isinstance(x, bytes) else str(x) for x in raw]

        # Read obs (cell barcodes)
        obs_index = None
        if 'obs' in f and '_index' in f['obs']:
            raw = f['obs']['_index'][:]
            obs_index = [x.decode() if isinstance(x, bytes) else str(x) for x in raw]

        if var_index is None:
            var_index = [f"gene_{i}" for i in range(X.shape[1])]
        if obs_index is None:
            obs_index = [f"cell_{i}" for i in range(X.shape[0])]

        var_df = pd.DataFrame(index=var_index)
        obs_df = pd.DataFrame(index=obs_index)

        return ad.AnnData(X=X, obs=obs_df, var=var_df)


def _load_10x_mtx_manual(mtx_path: str, barcodes_path: str, 
                          genes_path: str) -> 'ad.AnnData':
    """Manually load 10x MTX triplet, handling both v2 (2-col) and v3 (3-col) gene files."""
    import anndata as ad
    from scipy.io import mmread
    import gzip

    # Read matrix
    opener = gzip.open if mtx_path.endswith('.gz') else open
    with opener(mtx_path, 'rb') as f:
        X = mmread(f).T.tocsr()  # genes x cells -> cells x genes

    # Read barcodes
    opener = gzip.open if barcodes_path.endswith('.gz') else open
    with opener(barcodes_path, 'rt') as f:
        barcodes = [line.strip().split('\t')[0] for line in f]

    # Read genes (handles 2 or 3 columns)
    opener = gzip.open if genes_path.endswith('.gz') else open
    with opener(genes_path, 'rt') as f:
        gene_rows = [line.strip().split('\t') for line in f]

    gene_ids = [r[0] for r in gene_rows]
    gene_symbols = [r[1] if len(r) > 1 else r[0] for r in gene_rows]

    # Build AnnData
    obs = pd.DataFrame(index=barcodes)
    var = pd.DataFrame({'gene_ids': gene_ids, 'gene_symbols': gene_symbols}, 
                        index=gene_ids)
    var.index = var.index.astype(str)
    # Make unique
    if var.index.duplicated().any():
        var.index = ad.utils.make_index_unique(pd.Index(var.index))

    adata = ad.AnnData(X=X, obs=obs, var=var)
    logger.info(f"Manual MTX load: {adata.n_obs} cells x {adata.n_vars} genes")
    return adata


def load_10x_mtx(study_dir: str, mtx_path: str) -> Optional[MatrixLoadResult]:
    """Load 10x Genomics MTX triplet format (matrix.mtx + barcodes.tsv + features.tsv).
    Handles both filtered and unfiltered matrices. For unfiltered matrices,
    applies basic EmptyDrops-like filtering (cells with > MIN_GENES_PER_CELL genes).

    Supports:
    - Standard 10x directory layout (matrix.mtx.gz + barcodes.tsv.gz + features.tsv.gz)
    - GSM-prefixed files in flat directories (GSM123_sample.matrix.mtx.gz + ...)
    """
    mtx_path = Path(mtx_path)
    parent = mtx_path.parent

    # Detect GSM prefix from filename for prefix-based companion lookup
    # Pattern: GSM1234567_samplename.matrix.mtx.gz -> prefix = GSM1234567_samplename
    mtx_prefix = None
    mtx_stem = mtx_path.name
    for suffix in ['.matrix.mtx.gz', '.matrix.mtx',
                   '_matrix.mtx.gz', '_matrix.mtx']:
        if mtx_stem.lower().endswith(suffix):
            mtx_prefix = mtx_stem[:len(mtx_stem) - len(suffix)]
            break

    # Find companion files (try standard names first, then prefix-based)
    barcodes_file = _find_companion_file(parent, ['barcodes.tsv.gz', 'barcodes.tsv'],
                                         prefix=mtx_prefix)
    features_file = _find_companion_file(parent, [
        'features.tsv.gz', 'features.tsv',
        'genes.tsv.gz', 'genes.tsv'
    ], prefix=mtx_prefix)

    if barcodes_file is None or features_file is None:
        logger.warning(f"Missing companion files for {mtx_path} (prefix={mtx_prefix})")
        # Try loading as standalone MTX
        return _load_standalone_mtx(str(mtx_path))

    warnings = []

    # If files are GSM-prefixed, create a temp directory with standard names
    # so scanpy's read_10x_mtx can find them
    load_dir = str(parent)
    _temp_dir = None
    if mtx_prefix and not mtx_stem.lower().startswith('matrix'):
        import tempfile
        _temp_dir = tempfile.mkdtemp(prefix='10x_triplet_')
        _temp_path = Path(_temp_dir)
        os.symlink(str(mtx_path), str(_temp_path / 'matrix.mtx.gz'))
        os.symlink(str(barcodes_file), str(_temp_path / 'barcodes.tsv.gz'))
        os.symlink(str(features_file), str(_temp_path / 'genes.tsv.gz'))
        os.symlink(str(features_file), str(_temp_path / 'features.tsv.gz'))
        load_dir = _temp_dir
        warnings.append(f"GSM-prefixed triplet: {mtx_prefix}")

    try:
        try:
            adata = sc.read_10x_mtx(load_dir, var_names='gene_ids', make_unique=True)
        except Exception:
            try:
                adata = sc.read_10x_mtx(load_dir, var_names='gene_symbols', make_unique=True)
            except Exception:
                # Fallback: manual load for Cell Ranger v2 (2-column genes file)
                try:
                    adata = _load_10x_mtx_manual(
                        str(mtx_path), str(barcodes_file), str(features_file))
                except Exception as e:
                    logger.error(f"Failed to load 10x MTX from {load_dir}: {e}")
                    return None
    finally:
        if _temp_dir:
            import shutil as _shutil
            _shutil.rmtree(_temp_dir, ignore_errors=True)
            
    gene_ids = list(adata.var_names[:100])
    gene_id_format = detect_gene_id_format(gene_ids)
    is_raw, evidence = detect_is_raw_counts(adata.X)
    was_unfiltered = detect_unfiltered_10x(adata.n_obs)

    if was_unfiltered:
        warnings.append(f"Unfiltered 10x matrix: {adata.n_obs:,} barcodes, will filter empty droplets")

    return MatrixLoadResult(
        adata=adata,
        format_type='mtx',
        source_path=str(mtx_path),
        is_raw_counts=is_raw,
        was_unfiltered=was_unfiltered,
        gene_id_format=gene_id_format,
        warnings=warnings,
    )

def load_10x_h5(filepath: str) -> Optional[MatrixLoadResult]:
    """Load 10x Genomics HDF5 format."""
    warnings = []
    try:
        adata = sc.read_10x_h5(filepath)
    except Exception as e:
        logger.error(f"Failed to load 10x H5 {filepath}: {e}")
        return None

    gene_ids = list(adata.var_names[:100])
    gene_id_format = detect_gene_id_format(gene_ids)
    is_raw, evidence = detect_is_raw_counts(adata.X)
    was_unfiltered = detect_unfiltered_10x(adata.n_obs)

    if was_unfiltered:
        warnings.append(f"Unfiltered 10x H5: {adata.n_obs:,} barcodes")

    return MatrixLoadResult(
        adata=adata,
        format_type='h5',
        source_path=filepath,
        is_raw_counts=is_raw,
        was_unfiltered=was_unfiltered,
        gene_id_format=gene_id_format,
        warnings=warnings,
    )


def load_loom(filepath: str) -> Optional[MatrixLoadResult]:
    """Load Loom format."""
    warnings = []
    try:
        adata = sc.read_loom(filepath, sparse=True, cleanup=True)
    except Exception as e:
        logger.error(f"Failed to load loom {filepath}: {e}")
        return None

    gene_ids = list(adata.var_names[:100])
    gene_id_format = detect_gene_id_format(gene_ids)
    is_raw, evidence = detect_is_raw_counts(adata.X)

    return MatrixLoadResult(
        adata=adata,
        format_type='loom',
        source_path=filepath,
        is_raw_counts=is_raw,
        gene_id_format=gene_id_format,
        warnings=warnings,
    )


def load_tabular(filepath: str) -> Optional[MatrixLoadResult]:
    """Load TSV/CSV dense count matrix.

    Orientation detection: if n_rows >> n_cols, assume genes×cells (transpose needed).
    If n_cols >> n_rows, assume cells×genes (standard).
    """
    warnings = []
    fpath = Path(filepath)

    try:
        # Detect separator
        sep = '\t' if fpath.suffix in ('.tsv', '.gz') and '.tsv' in fpath.name else ','
        if '.txt' in fpath.name:
            sep = '\t'

        # Read with pandas
        if filepath.endswith('.gz'):
            df = pd.read_csv(filepath, sep=sep, index_col=0, compression='gzip',
                             nrows=5)
        else:
            df = pd.read_csv(filepath, sep=sep, index_col=0, nrows=5)

        # Check dimensions to detect orientation
        n_rows_sample = df.shape[0]
        n_cols_sample = df.shape[1]

        # Now read full file
        if filepath.endswith('.gz'):
            df = pd.read_csv(filepath, sep=sep, index_col=0, compression='gzip')
        else:
            df = pd.read_csv(filepath, sep=sep, index_col=0)

        # Orientation check: if rows look like gene IDs (Ensembl, symbols), transpose
        row_ids = list(df.index[:50])
        col_ids = list(df.columns[:50])
        row_format = detect_gene_id_format([str(x) for x in row_ids])
        col_format = detect_gene_id_format([str(x) for x in col_ids])

        needs_transpose = False
        if row_format in ('ensembl_rat', 'ensembl_mouse', 'ensembl_human', 'symbol', 'loc'):
            if col_format not in ('ensembl_rat', 'ensembl_mouse', 'ensembl_human', 'symbol', 'loc'):
                needs_transpose = True
                warnings.append(f"Transposed: rows were {row_format} (genes), cols were {col_format}")
        elif df.shape[0] > df.shape[1] * 5:
            # Many more rows than columns → probably genes × cells
            needs_transpose = True
            warnings.append(f"Transposed based on shape: {df.shape[0]}×{df.shape[1]}")

        if needs_transpose:
            df = df.T

        # Convert to AnnData
        adata = ad.AnnData(X=sp.csr_matrix(df.values.astype(np.float32)),
                           obs=pd.DataFrame(index=df.index.astype(str)),
                           var=pd.DataFrame(index=df.columns.astype(str)))

        gene_ids = list(adata.var_names[:100])
        gene_id_format = detect_gene_id_format(gene_ids)
        is_raw, evidence = detect_is_raw_counts(adata.X)

        return MatrixLoadResult(
            adata=adata,
            format_type='tsv' if '\t' in sep else 'csv',
            source_path=filepath,
            is_raw_counts=is_raw,
            gene_id_format=gene_id_format,
            warnings=warnings,
        )

    except Exception as e:
        logger.error(f"Failed to load tabular {filepath}: {e}")
        return None


def _load_standalone_mtx(filepath: str) -> Optional[MatrixLoadResult]:
    """Load a standalone MTX file without companion barcodes/features files."""
    try:
        mat = scipy_io.mmread(filepath)
        if sp.issparse(mat):
            mat = mat.tocsr()
        # Without gene names, we can't proceed meaningfully
        adata = ad.AnnData(
            X=mat,
            var=pd.DataFrame(index=[f"gene_{i}" for i in range(mat.shape[1])]),
            obs=pd.DataFrame(index=[f"cell_{i}" for i in range(mat.shape[0])]),
        )
        return MatrixLoadResult(
            adata=adata,
            format_type='mtx',
            source_path=filepath,
            is_raw_counts=True,
            gene_id_format='unknown',
            warnings=["Standalone MTX without gene names — limited utility"],
        )
    except Exception as e:
        logger.error(f"Failed to load standalone MTX {filepath}: {e}")
        return None


def _find_companion_file(directory: Path, candidates: List[str],
                          prefix: str = None) -> Optional[Path]:
    """Find a companion file from a list of candidates in a directory.

    If prefix is given, also searches for {prefix}.{candidate} patterns,
    which handles GSM-prefixed 10x files like GSM3633087_sample.barcodes.tsv.gz
    """
    # Standard lookup: exact names
    for name in candidates:
        path = directory / name
        if path.exists():
            return path

    # Prefix-based lookup
    if prefix:
        for name in candidates:
            for f in directory.iterdir():
                if not f.is_file():
                    continue
                fname = f.name.lower()
                if fname.startswith(prefix.lower()) and fname.endswith(name.lower()):
                    return f
    return None


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2: MATRIX DISCOVERY AND SELECTION
# ═════════════════════════════════════════════════════════════════════════════

def find_matrix_files(study_dir: str) -> List[Tuple[str, str]]:
    """Discover matrix files in a study directory.

    Returns list of (filepath, format_type) tuples, prioritized by format
    preference (h5ad > h5 > mtx > loom > tabular).

    Handles:
    - Top-level files
    - RAW/ subdirectories (common in GEO)
    - Nested sample directories
    """
    study_path = Path(study_dir)
    candidates = []

    # Directories to search
    search_dirs = [study_path]
    raw_dir = study_path / (study_path.name + '_RAW')
    if raw_dir.exists():
        search_dirs.append(raw_dir)
    # Also check for any directory named RAW
    for d in study_path.iterdir():
        if d.is_dir() and d.name.endswith('_RAW'):
            if d not in search_dirs:
                search_dirs.append(d)

    seen_mtx_dirs = set()  # Track dirs with standard MTX files
    seen_mtx_prefixes = set()  # Track GSM-prefixed MTX files

    for search_dir in search_dirs:
        for root, dirs, files in os.walk(search_dir):
            root_path = Path(root)
            for fname in files:
                fpath = root_path / fname
                fname_lower = fname.lower()

                # Skip metadata/annotation files
                if any(skip in fname_lower for skip in SKIP_FILE_PATTERNS):
                    continue

                # Detect format
                fmt = _detect_format_from_extension(fname)
                if fmt is None:
                    continue

                # For MTX files, handle standard vs GSM-prefixed
                if fmt == 'mtx':
                    # Check if GSM-prefixed (e.g., GSM123_sample.matrix.mtx.gz)
                    is_gsm_prefixed = bool(re.match(
                        r'^(GSM\d+[_.].*|E-\w+-\d+[_.].*)[._](matrix[._])?mtx(\.gz)?$',
                        fname, re.IGNORECASE
                    ))

                    if is_gsm_prefixed:
                        # Each GSM-prefixed MTX is a separate sample
                        prefix = fname
                        for sfx in ['.matrix.mtx.gz', '.matrix.mtx', '.mtx.gz', '.mtx']:
                            if fname_lower.endswith(sfx):
                                prefix = fname[:len(fname) - len(sfx)]
                                break
                        prefix_key = f"{root_path}::{prefix}"
                        if prefix_key not in seen_mtx_prefixes:
                            seen_mtx_prefixes.add(prefix_key)
                            candidates.append((str(fpath), 'mtx'))
                    else:
                        # Standard 10x: one per directory
                        if str(root_path) not in seen_mtx_dirs:
                            seen_mtx_dirs.add(str(root_path))
                            candidates.append((str(fpath), 'mtx'))
                else:
                    candidates.append((str(fpath), fmt))

    # Deduplicate and sort by priority
    format_priority = {'h5ad': 0, 'h5': 1, 'mtx': 2, 'loom': 3, 'tsv': 4, 'csv': 5}
    candidates.sort(key=lambda x: format_priority.get(x[1], 99))

    return candidates


def _detect_format_from_extension(filename: str) -> Optional[str]:
    """Detect matrix format from file extension."""
    fname_lower = filename.lower()

    # Check compound extensions first
    if fname_lower.endswith('.mtx.gz') or fname_lower.endswith('.mtx'):
        return 'mtx'
    if fname_lower.endswith('.h5ad'):
        return 'h5ad'
    if fname_lower.endswith('.h5') or fname_lower.endswith('.hdf5'):
        return 'h5'
    if fname_lower.endswith('.loom'):
        return 'loom'
    if fname_lower.endswith('.tsv.gz') or fname_lower.endswith('.tsv'):
        # Only if filename suggests a matrix (avoid metadata tables)
        if any(p in fname_lower for p in MATRIX_FILE_PATTERNS):
            return 'tsv'
        # Single-file TSVs with GSM prefix are often sample-level count matrices
        if fname_lower.startswith('gsm') and not any(s in fname_lower for s in ['_samples', '_sra']):
            return 'tsv'
        return None
    if fname_lower.endswith('.csv.gz') or fname_lower.endswith('.csv'):
        if any(p in fname_lower for p in MATRIX_FILE_PATTERNS):
            return 'csv'
        if fname_lower.startswith('gsm'):
            return 'csv'
        return None
    if fname_lower.endswith('.txt.gz') or fname_lower.endswith('.txt'):
        # Only if filename suggests a matrix
        if any(p in fname_lower for p in MATRIX_FILE_PATTERNS):
            return 'tsv'
    return None


def load_matrix(filepath: str, format_type: str, study_dir: str) -> Optional[MatrixLoadResult]:
    """Dispatch to the appropriate format-specific loader."""
    if format_type == 'h5ad':
        return load_h5ad(filepath)
    elif format_type == 'mtx':
        return load_10x_mtx(study_dir, filepath)
    elif format_type == 'h5':
        return load_10x_h5(filepath)
    elif format_type == 'loom':
        return load_loom(filepath)
    elif format_type in ('tsv', 'csv'):
        return load_tabular(filepath)
    else:
        logger.warning(f"Unknown format {format_type} for {filepath}")
        return None


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3: GENE ID MAPPING
# ═════════════════════════════════════════════════════════════════════════════

class GeneMapper:
    """Map diverse gene IDs to unified Ensembl rat IDs using the symbol lookup.

    Handles:
    - Ensembl IDs (with or without version suffixes)
    - Gene symbols (case-insensitive lookup)
    - LOC IDs
    - NCBI numeric IDs
    """

    def __init__(self, rat_reference_path: str, core_ensembl_ids: Set[str]):
        """
        Args:
            rat_reference_path: Path to rat_symbol_lookup.pickle
            core_ensembl_ids: Set of Ensembl IDs from the gene inventory (biotype-filtered)
        """
        self.core_ids = core_ensembl_ids
        self.symbol_to_ensembl = {}
        self.ensembl_versioned = {}  # ENSRNOG00000001234.5 → ENSRNOG00000001234

        # Load the symbol lookup
        with open(rat_reference_path, 'rb') as f:
            self.lookup = pickle.load(f)

        # Build reverse mappings
        self._build_mappings()

    def _build_mappings(self):
        """Build efficient lookup structures from the pickle reference."""
        # The lookup maps symbol → info dict
        # We need symbol → ensembl_id mapping
        for symbol, info in self.lookup.items():
            if isinstance(info, dict):
                eid = info.get('ensembl_id') or info.get('ensembl_gene_id')
                if eid and eid in self.core_ids:
                    self.symbol_to_ensembl[symbol.lower()] = eid
            elif isinstance(info, str):
                # Direct mapping
                if info in self.core_ids:
                    self.symbol_to_ensembl[symbol.lower()] = info

    def map_gene_ids(self, gene_names: List[str]) -> Tuple[Dict[str, str], Dict[str, str]]:
        """Map a list of gene IDs to core Ensembl IDs.

        Returns:
            mapped: {original_id: ensembl_id} for successfully mapped genes
            unmapped: {original_id: reason} for genes that couldn't be mapped
        """
        mapped = {}
        unmapped = {}

        for gid in gene_names:
            gid_str = str(gid).strip()

            # Direct Ensembl ID
            if gid_str.startswith('ENSRNOG'):
                # Strip version suffix
                base_id = gid_str.split('.')[0]
                if base_id in self.core_ids:
                    mapped[gid_str] = base_id
                else:
                    unmapped[gid_str] = 'ensembl_not_in_core'
                continue

            # Symbol lookup (case-insensitive)
            lower = gid_str.lower()
            if lower in self.symbol_to_ensembl:
                mapped[gid_str] = self.symbol_to_ensembl[lower]
                continue

            # LOC ID lookup
            if gid_str.startswith('LOC'):
                if lower in self.symbol_to_ensembl:
                    mapped[gid_str] = self.symbol_to_ensembl[lower]
                    continue

            # Not found
            unmapped[gid_str] = 'no_mapping_found'

        return mapped, unmapped

    def subset_adata_to_core(self, adata: 'ad.AnnData') -> Tuple['ad.AnnData', Dict]:
        """Subset an AnnData object to only core genes, renaming to Ensembl IDs.

        Returns:
            adata_subset: AnnData with only mapped genes, var_names = Ensembl IDs
            stats: mapping statistics
        """
        gene_names = list(adata.var_names)
        mapped, unmapped = self.map_gene_ids(gene_names)

        # Handle duplicate mappings (multiple symbols → same Ensembl ID)
        # Keep the one with higher total expression
        ensembl_to_originals = defaultdict(list)
        for orig, ens in mapped.items():
            ensembl_to_originals[ens].append(orig)

        # For duplicates, pick the gene with highest total expression
        final_mapping = {}
        duplicate_count = 0
        for ens, originals in ensembl_to_originals.items():
            if len(originals) == 1:
                final_mapping[originals[0]] = ens
            else:
                duplicate_count += 1
                # Sum expression across cells for each candidate
                best_orig = originals[0]
                best_sum = 0
                for orig in originals:
                    idx = list(adata.var_names).index(orig)
                    col = adata.X[:, idx]
                    total = col.sum() if sp.issparse(col) else np.sum(col)
                    if total > best_sum:
                        best_sum = total
                        best_orig = orig
                final_mapping[best_orig] = ens

        # Subset
        keep_genes = list(final_mapping.keys())
        if len(keep_genes) == 0:
            return None, {'mapped': 0, 'unmapped': len(gene_names), 'duplicates': 0}

        mask = np.isin(adata.var_names, keep_genes)
        adata_sub = adata[:, mask].copy()

        # Rename var_names to Ensembl IDs
        new_names = [final_mapping[g] for g in adata_sub.var_names]
        adata_sub.var_names = new_names
        adata_sub.var_names_make_unique()

        stats = {
            'total_input_genes': len(gene_names),
            'mapped': len(final_mapping),
            'unmapped': len(unmapped),
            'duplicates_resolved': duplicate_count,
            'mapping_rate': len(final_mapping) / len(gene_names) if gene_names else 0,
        }

        return adata_sub, stats


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4: CELL QUALITY CONTROL (GeneCompass-aligned)
# ═════════════════════════════════════════════════════════════════════════════

def identify_mito_genes(gene_names: List[str]) -> np.ndarray:
    """Identify mitochondrial genes from a list of gene names/IDs.

    Returns boolean mask: True for mitochondrial genes.
    """
    mito_mask = np.zeros(len(gene_names), dtype=bool)
    for i, gname in enumerate(gene_names):
        gname_str = str(gname)
        for pattern in RAT_MITO_PATTERNS:
            if pattern.search(gname_str):
                mito_mask[i] = True
                break
    return mito_mask


def apply_cell_qc(adata: 'ad.AnnData', sample_id: str = 'unknown') -> Tuple['ad.AnnData', Dict]:
    """Apply GeneCompass-style cell QC filters.

    Filter sequence:
    1. Cells with < MIN_GENES_PER_CELL expressed genes
    2. Cells with mitochondrial fraction > MAX_MITO_FRACTION
    3. Cells with gene count > OUTLIER_SD_THRESHOLD SD above sample mean
    4. (Post-filter) Drop sample if < MIN_CELLS_PER_SAMPLE cells remain

    Returns:
        adata_qc: QC'd AnnData (or None if sample dropped)
        stats: QC statistics
    """
    n_initial = adata.n_obs
    gene_names = list(adata.var_names)

    # Calculate per-cell metrics
    if sp.issparse(adata.X):
        genes_per_cell = np.asarray((adata.X > 0).sum(axis=1)).flatten()
        total_counts = np.asarray(adata.X.sum(axis=1)).flatten()
    else:
        X = np.asarray(adata.X)
        genes_per_cell = np.sum(X > 0, axis=1)
        total_counts = np.sum(X, axis=1)

    # Mitochondrial fraction
    mito_mask = identify_mito_genes(gene_names)
    n_mito_genes = int(np.sum(mito_mask))
    if n_mito_genes > 0 and sp.issparse(adata.X):
        mito_counts = np.asarray(adata.X[:, mito_mask].sum(axis=1)).flatten()
    elif n_mito_genes > 0:
        mito_counts = np.sum(np.asarray(adata.X)[:, mito_mask], axis=1)
    else:
        mito_counts = np.zeros(n_initial)

    mito_fraction = np.divide(mito_counts, total_counts,
                              out=np.zeros_like(mito_counts, dtype=float),
                              where=total_counts > 0)

    # ── Filter 1: minimum genes per cell ──
    pass_min_genes = genes_per_cell >= MIN_GENES_PER_CELL
    n_fail_min_genes = int(np.sum(~pass_min_genes))

    # ── Filter 2: mitochondrial fraction ──
    pass_mito = mito_fraction <= MAX_MITO_FRACTION
    n_fail_mito = int(np.sum(pass_min_genes & ~pass_mito))

    # ── Filter 3: outlier gene count (> 3 SD above mean) ──
    # Compute on cells that pass filters 1+2
    combined_pass = pass_min_genes & pass_mito
    if np.sum(combined_pass) > 10:
        passing_gene_counts = genes_per_cell[combined_pass]
        mean_genes = np.mean(passing_gene_counts)
        sd_genes = np.std(passing_gene_counts)
        threshold = mean_genes + OUTLIER_SD_THRESHOLD * sd_genes
        pass_outlier = genes_per_cell <= threshold
        n_fail_outlier = int(np.sum(combined_pass & ~pass_outlier))
    else:
        pass_outlier = np.ones(n_initial, dtype=bool)
        n_fail_outlier = 0
        threshold = float('inf')

    # Combined filter
    final_pass = pass_min_genes & pass_mito & pass_outlier
    n_final = int(np.sum(final_pass))

    stats = {
        'n_initial': n_initial,
        'n_fail_min_genes': n_fail_min_genes,
        'n_fail_mito': n_fail_mito,
        'n_fail_outlier': n_fail_outlier,
        'n_after_qc': n_final,
        'n_mito_genes_detected': n_mito_genes,
        'outlier_threshold': float(threshold) if threshold != float('inf') else None,
        'sample_id': sample_id,
    }

    # ── Filter 4: minimum cells per sample ──
    if n_final < MIN_CELLS_PER_SAMPLE:
        stats['dropped'] = True
        stats['drop_reason'] = f"too_few_cells_after_qc ({n_final} < {MIN_CELLS_PER_SAMPLE})"
        return None, stats

    stats['dropped'] = False
    adata_qc = adata[final_pass].copy()

    return adata_qc, stats


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 5: NORMALIZATION AND RANKING
# ═════════════════════════════════════════════════════════════════════════════

def normalize_and_rank(adata: 'ad.AnnData') -> Tuple['ad.AnnData', Set[str]]:
    """Normalize per cell and determine top-2048 genes per cell.

    Normalization: CPM (counts per 10,000) → log1p
    This is the standard approach used by GeneCompass, scGPT, scFoundation, UniCell.

    Returns:
        adata: with normalized X and raw counts in .raw
        top_genes: set of gene names that appeared in ANY cell's top 2048
    """
    # Store raw counts
    adata.raw = adata.copy()

    # Normalize: total counts → 10,000, then log1p
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # Per-cell top-2048 gene tracking
    top_genes = set()
    n_cells = adata.n_obs
    gene_names = np.array(adata.var_names)

    # Process in chunks to manage memory
    chunk_size = 5000
    for start in range(0, n_cells, chunk_size):
        end = min(start + chunk_size, n_cells)

        if sp.issparse(adata.X):
            chunk = np.asarray(adata.X[start:end].todense())
        else:
            chunk = np.asarray(adata.X[start:end])

        for i in range(chunk.shape[0]):
            row = chunk[i]
            # Get indices of top-2048 by expression value
            if np.count_nonzero(row) <= TOP_N_GENES:
                # If fewer than 2048 expressed genes, all of them qualify
                nonzero_idx = np.nonzero(row)[0]
            else:
                # Get top 2048 by value
                nonzero_idx = np.argpartition(row, -TOP_N_GENES)[-TOP_N_GENES:]

            top_genes.update(gene_names[nonzero_idx])

    return adata, top_genes


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 6: STUDY-LEVEL PROCESSING
# ═════════════════════════════════════════════════════════════════════════════

def process_study(accession: str, study_dir: str, gene_mapper: GeneMapper,
                  output_dir: str, save_qc_matrices: bool = True) -> Dict:
    """Process a single study end-to-end.

    Steps:
    1. Discover matrix files in study directory
    2. Load best available matrix format
    3. Detect raw counts vs normalized
    4. Map gene IDs to core Ensembl IDs
    5. Apply cell QC filters
    6. Normalize and rank
    7. Track top-2048 genes
    8. Optionally save QC'd h5ad

    Returns study-level report dict.
    """
    t_start = time.time()
    report = {
        'accession': accession,
        'study_dir': study_dir,
        'status': 'pending',
        'warnings': [],
        'errors': [],
        'top_genes': set(),
    }

    # ── Step 1: Discover matrix files ──
    matrix_files = find_matrix_files(study_dir)
    if not matrix_files:
        report['status'] = 'no_matrices_found'
        return report

    report['matrix_candidates'] = [(fp, fmt) for fp, fmt in matrix_files]

    # ── Step 2: Load matrices (try each candidate) ──
    loaded_results = []
    for filepath, fmt in matrix_files:
        try:
            result = load_matrix(filepath, fmt, study_dir)
            if result is not None and result.adata.n_obs > 0 and result.adata.n_vars > 0:
                loaded_results.append(result)
                report['warnings'].extend(result.warnings)
        except Exception as e:
            report['errors'].append(f"Failed to load {filepath}: {str(e)}")

    if not loaded_results:
        report['status'] = 'all_loads_failed'
        return report

    # Process each loaded matrix (a study may have multiple samples)
    all_top_genes = set()
    sample_reports = []
    total_cells_before_qc = 0
    total_cells_after_qc = 0

    for idx, load_result in enumerate(loaded_results):
        sample_id = f"{accession}_sample{idx}"
        adata = load_result.adata

        # ── Step 3: Handle normalized data ──
        if not load_result.is_raw_counts:
            try:
                adata.X = reverse_normalization(adata.X)
                report['warnings'].append(
                    f"{load_result.source_path}: reversed normalization (scFoundation method)")
            except Exception as e:
                report['errors'].append(
                    f"{load_result.source_path}: normalization reversal failed: {e}")
                continue

        # ── Step 3b: Handle unfiltered 10x matrices ──
        if load_result.was_unfiltered:
            # Basic empty droplet filtering: keep cells with >= MIN_GENES_PER_CELL genes
            if sp.issparse(adata.X):
                genes_per_cell = np.asarray((adata.X > 0).sum(axis=1)).flatten()
            else:
                genes_per_cell = np.sum(np.asarray(adata.X) > 0, axis=1)

            keep = genes_per_cell >= MIN_GENES_PER_CELL
            n_before = adata.n_obs
            adata = adata[keep].copy()
            report['warnings'].append(
                f"Filtered unfiltered 10x: {n_before:,} → {adata.n_obs:,} cells "
                f"(kept cells with >= {MIN_GENES_PER_CELL} genes)")

        # ── Step 4: Map gene IDs to core Ensembl IDs ──
        adata_mapped, mapping_stats = gene_mapper.subset_adata_to_core(adata)
        if adata_mapped is None or adata_mapped.n_vars == 0:
            report['errors'].append(
                f"{load_result.source_path}: no genes mapped to core set "
                f"(format: {load_result.gene_id_format})")
            continue

        sample_report = {
            'sample_id': sample_id,
            'source': load_result.source_path,
            'format': load_result.format_type,
            'gene_id_format': load_result.gene_id_format,
            'is_raw_counts': load_result.is_raw_counts,
            'mapping_stats': mapping_stats,
        }

        total_cells_before_qc += adata_mapped.n_obs

        # ── Step 5: Cell QC ──
        adata_qc, qc_stats = apply_cell_qc(adata_mapped, sample_id=sample_id)
        sample_report['qc_stats'] = qc_stats

        if adata_qc is None:
            sample_report['status'] = 'dropped_qc'
            sample_reports.append(sample_report)
            continue

        total_cells_after_qc += adata_qc.n_obs

        # ── Step 6: Normalize and rank ──
        try:
            adata_norm, top_genes = normalize_and_rank(adata_qc)
            all_top_genes.update(top_genes)
            sample_report['n_top_genes'] = len(top_genes)
            sample_report['status'] = 'success'
        except Exception as e:
            sample_report['status'] = 'normalize_failed'
            sample_report['error'] = str(e)
            sample_reports.append(sample_report)
            continue

        # ── Step 7: Save QC'd matrix ──
        if save_qc_matrices:
            out_path = Path(output_dir) / 'qc_matrices' / f"{sample_id}.h5ad"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                adata_norm.write_h5ad(str(out_path))
                sample_report['output_path'] = str(out_path)
            except Exception as e:
                sample_report['save_error'] = str(e)

        sample_reports.append(sample_report)

        # Free memory
        del adata, adata_mapped, adata_qc, adata_norm

    # ── Compile study report ──
    report['samples'] = sample_reports
    report['top_genes'] = all_top_genes
    report['n_samples_processed'] = len([s for s in sample_reports if s.get('status') == 'success'])
    report['n_samples_dropped'] = len([s for s in sample_reports if s.get('status') != 'success'])
    report['total_cells_before_qc'] = total_cells_before_qc
    report['total_cells_after_qc'] = total_cells_after_qc
    report['elapsed_seconds'] = time.time() - t_start
    report['status'] = 'success' if report['n_samples_processed'] > 0 else 'all_samples_failed'

    return report


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 7: CORPUS-LEVEL ORCHESTRATION
# ═════════════════════════════════════════════════════════════════════════════

def load_study_catalog(data_root: str) -> List[Dict]:
    """Load the unified study catalog and filter to usable studies.

    Catalog schema (unified_studies.json):
        {
          "studies": [
            {
              "accession": "GSE...",
              "source": "geo" | "arrayexpress",
              "catalog": {
                "data_type": "single_cell" | "bulk",
                "raw_organisms": ["Rattus norvegicus", ...],
                ...
              },
              "llm": {
                "motrpac_utility": {"is_rat": bool, "genecompass_useful": bool, ...},
                "study_type": {"is_single_cell": bool, ...},
                "validated_organism": {"species": "Rattus norvegicus", ...},
                ...
              },
              "matrix": {"n_cells": int|null, "n_genes": int|null, ...},
              "quality": {"has_matrix": bool, ...}
            }
          ]
        }

    Filtering criteria (all must pass):
        1. catalog.data_type == 'single_cell'
        2. Rat organism (LLM-validated preferred, raw_organisms fallback)
        3. LLM confirms is_single_cell (cross-check against catalog misclassification)
        4. Has matrix data on disk
        5. Not in exclusion list (data leakage prevention)

    Directory resolution:
        GEO studies:          {data_root}/geo/single_cell/geo_datasets/{accession}
        ArrayExpress studies:  {data_root}/arrayexpress/singlecell/{accession}
    """
    catalog_path = Path(data_root) / 'catalog' / 'unified_studies.json'
    if not catalog_path.exists():
        logger.error(f"Catalog not found: {catalog_path}")
        return []

    with open(catalog_path) as f:
        catalog = json.load(f)

    studies = []
    exclusion_list = {'GSE242354'}  # MoTrPAC — data leakage prevention

    # Counters for diagnostics
    n_excluded = 0
    n_not_sc_catalog = 0
    n_not_rat = 0
    n_not_sc_llm = 0
    n_no_dir = 0

    for study in catalog.get('studies', []):
        accession = study.get('accession', '')

        # ── Filter 1: Exclusion list ──
        if accession in exclusion_list:
            n_excluded += 1
            continue

        # ── Filter 2: Catalog data type must be single_cell ──
        cat_info = study.get('catalog', {})
        if cat_info.get('data_type') != 'single_cell':
            n_not_sc_catalog += 1
            continue

        # ── Filter 3: Must be rat ──
        # Primary: LLM-validated motrpac_utility
        llm_info = study.get('llm', {})
        utility = llm_info.get('motrpac_utility', {})
        is_rat = utility.get('is_rat')

        if is_rat is not True:
            if is_rat is False:
                n_not_rat += 1
                continue
            # Fallback: check LLM validated_organism
            validated_org = llm_info.get('validated_organism', {})
            species = str(validated_org.get('species', '')).lower()
            if 'rattus' in species or 'rat' in species:
                pass  # Accept
            else:
                # Last resort: check catalog raw_organisms
                raw_orgs = cat_info.get('raw_organisms', [])
                if not any('rattus' in str(o).lower() or 'rat' == str(o).strip().lower()
                           for o in raw_orgs):
                    n_not_rat += 1
                    continue

        # ── Filter 4: LLM confirms single-cell (cross-check) ──
        # The LLM analysis found catalog data_type was only ~83% accurate.
        # If LLM explicitly says NOT single-cell, skip.
        study_type = llm_info.get('study_type', {})
        llm_is_sc = study_type.get('is_single_cell')
        if llm_is_sc is False:
            n_not_sc_llm += 1
            continue

        # ── Filter 5: Resolve study directory ──
        source = study.get('source', 'geo')
        if source == 'arrayexpress':
            study_dir = str(Path(data_root) / 'arrayexpress' / 'singlecell' / 'datasets' / accession)
        else:
            # GEO (default)
            study_dir = str(Path(data_root) / 'geo' / 'single_cell' / 'geo_datasets' / accession)

        if not Path(study_dir).exists():
            n_no_dir += 1
            continue

        studies.append({
            'accession': accession,
            'study_dir': study_dir,
            'source': source,
            'metadata': study,
        })

    logger.info(f"Loaded {len(studies)} usable studies from catalog "
                f"(excluded={n_excluded}, not_sc_catalog={n_not_sc_catalog}, "
                f"not_rat={n_not_rat}, not_sc_llm={n_not_sc_llm}, no_dir={n_no_dir})")
    return studies


def run_corpus_preprocessing(args):
    """Main orchestration: process all studies and produce outputs."""
    t_total_start = time.time()

    # ── Load references ──
    logger.info("Loading gene inventory and references...")

    # Core Ensembl IDs (biotype-filtered from Stage 2)
    with open(args.rat_ensembl_ids) as f:
        core_ensembl_ids = set(line.strip() for line in f if line.strip())
    logger.info(f"Core gene list: {len(core_ensembl_ids):,} Ensembl IDs")

    # Gene mapper
    gene_mapper = GeneMapper(args.rat_reference, core_ensembl_ids)

    # Load study catalog
    studies = load_study_catalog(args.data_root)
    if not studies:
        logger.error("No usable studies found in catalog")
        return

    # ── Output directory ──
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'qc_matrices').mkdir(exist_ok=True)

    # ── Process studies ──
    global_top_genes = set()
    all_reports = []
    total_cells_corpus = 0
    n_success = 0
    n_failed = 0

    if args.workers > 1:
        logger.info(f"Processing {len(studies)} studies with {args.workers} workers...")
        # Note: ProcessPoolExecutor won't work well here because the GeneMapper
        # uses a large pickle. Use sequential with manual parallelism via SLURM
        # array jobs for real production runs.
        logger.warning("Multi-worker mode: gene_mapper is large, consider SLURM array instead")

    for i, study_info in enumerate(studies):
        accession = study_info['accession']
        study_dir = study_info['study_dir']

        logger.info(f"[{i+1}/{len(studies)}] Processing {accession}...")

        try:
            report = process_study(
                accession=accession,
                study_dir=study_dir,
                gene_mapper=gene_mapper,
                output_dir=str(output_dir),
                save_qc_matrices=args.save_matrices,
            )

            # Accumulate top genes
            if isinstance(report.get('top_genes'), set):
                global_top_genes.update(report['top_genes'])
                # Convert set to list for JSON serialization
                report['n_unique_top_genes'] = len(report['top_genes'])
                del report['top_genes']

            total_cells_corpus += report.get('total_cells_after_qc', 0)

            if report['status'] == 'success':
                n_success += 1
            else:
                n_failed += 1

            all_reports.append(report)

            # Periodic progress
            if (i + 1) % 10 == 0:
                logger.info(
                    f"  Progress: {i+1}/{len(studies)} studies, "
                    f"{n_success} success, {n_failed} failed, "
                    f"{len(global_top_genes):,} unique top genes, "
                    f"{total_cells_corpus:,} total QC'd cells"
                )

        except Exception as e:
            logger.error(f"Unhandled error in {accession}: {e}")
            logger.error(traceback.format_exc())
            all_reports.append({
                'accession': accession,
                'status': 'unhandled_error',
                'error': str(e),
            })
            n_failed += 1

    # ═════════════════════════════════════════════════════════════════════════
    # OUTPUTS
    # ═════════════════════════════════════════════════════════════════════════

    elapsed = time.time() - t_total_start

    # ── Output 1: Pruned gene list ──
    pruned_ids = sorted(global_top_genes & core_ensembl_ids)
    pruned_path = output_dir / 'rat_ensembl_ids_pruned.txt'
    with open(pruned_path, 'w') as f:
        for gid in pruned_ids:
            f.write(gid + '\n')
    logger.info(f"Pruned gene list: {len(pruned_ids):,} / {len(core_ensembl_ids):,} "
                f"Ensembl IDs survive expression filtering")

    # Genes that appeared in top-2048 but aren't Ensembl (e.g., symbols that got mapped)
    non_ensembl_top = global_top_genes - core_ensembl_ids
    if non_ensembl_top:
        logger.info(f"  {len(non_ensembl_top)} top genes are non-Ensembl "
                     f"(already mapped in matrix processing)")

    # ── Output 2: Preprocessing report ──
    corpus_report = {
        'timestamp': datetime.now().isoformat(),
        'parameters': {
            'min_genes_per_cell': MIN_GENES_PER_CELL,
            'min_cells_per_sample': MIN_CELLS_PER_SAMPLE,
            'max_mito_fraction': MAX_MITO_FRACTION,
            'outlier_sd_threshold': OUTLIER_SD_THRESHOLD,
            'top_n_genes': TOP_N_GENES,
        },
        'summary': {
            'total_studies': len(studies),
            'studies_success': n_success,
            'studies_failed': n_failed,
            'total_cells_after_qc': total_cells_corpus,
            'core_ensembl_ids': len(core_ensembl_ids),
            'pruned_ensembl_ids': len(pruned_ids),
            'genes_pruned': len(core_ensembl_ids) - len(pruned_ids),
            'pruning_rate': 1 - len(pruned_ids) / len(core_ensembl_ids) if core_ensembl_ids else 0,
            'elapsed_seconds': elapsed,
        },
        'studies': all_reports,
    }

    report_path = output_dir / 'preprocessing_report.json'
    with open(report_path, 'w') as f:
        json.dump(corpus_report, f, indent=2, default=str)
    logger.info(f"Report saved: {report_path}")

    # ── Output 3: Summary statistics ──
    summary_path = output_dir / 'preprocessing_summary.txt'
    with open(summary_path, 'w') as f:
        f.write("=" * 72 + "\n")
        f.write("Stage 3: Preprocessing Training Matrices — Summary\n")
        f.write("=" * 72 + "\n\n")
        f.write(f"Date: {datetime.now().isoformat()}\n")
        f.write(f"Elapsed: {elapsed/60:.1f} minutes\n\n")

        f.write("CORPUS OVERVIEW\n")
        f.write(f"  Studies processed:     {n_success}\n")
        f.write(f"  Studies failed:        {n_failed}\n")
        f.write(f"  Total QC'd cells:      {total_cells_corpus:,}\n\n")

        f.write("GENE PRUNING\n")
        f.write(f"  Core gene list (in):   {len(core_ensembl_ids):,} Ensembl IDs\n")
        f.write(f"  Pruned list (out):     {len(pruned_ids):,} Ensembl IDs\n")
        f.write(f"  Genes removed:         {len(core_ensembl_ids) - len(pruned_ids):,}\n")
        f.write(f"  Pruning rate:          {(1 - len(pruned_ids)/max(1,len(core_ensembl_ids)))*100:.1f}%\n\n")

        f.write("QC FILTER BREAKDOWN (across all studies)\n")
        total_initial = sum(r.get('total_cells_before_qc', 0) for r in all_reports)
        f.write(f"  Cells before QC:       {total_initial:,}\n")
        f.write(f"  Cells after QC:        {total_cells_corpus:,}\n")
        f.write(f"  Retention rate:        {total_cells_corpus/max(1,total_initial)*100:.1f}%\n\n")

        f.write("OUTPUT FILES\n")
        f.write(f"  {pruned_path}\n")
        f.write(f"  {report_path}\n")
        if args.save_matrices:
            f.write(f"  {output_dir / 'qc_matrices'}/ (per-sample h5ad files)\n")

    with open(summary_path) as f:
        print(f.read())


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 8: SLURM ARRAY SUPPORT
# ═════════════════════════════════════════════════════════════════════════════

def run_single_study_mode(args):
    """Process a single study (for SLURM array jobs).

    Usage: python preprocess_training_matrices.py --single-study GSE123456 ...

    For large corpora, this is more efficient than sequential processing:
    submit a SLURM array job where each task processes one study, then
    a final merge step combines the per-study outputs.
    """
    accession = args.single_study

    # Load references
    with open(args.rat_ensembl_ids) as f:
        core_ensembl_ids = set(line.strip() for line in f if line.strip())
    gene_mapper = GeneMapper(args.rat_reference, core_ensembl_ids)

    # Find study directory
    study_dir = str(Path(args.data_root) / 'geo' / 'single_cell' / 'geo_datasets' / accession)
    if not Path(study_dir).exists():
        logger.error(f"Study directory not found: {study_dir}")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report = process_study(
        accession=accession,
        study_dir=study_dir,
        gene_mapper=gene_mapper,
        output_dir=str(output_dir),
        save_qc_matrices=args.save_matrices,
    )

    # Save per-study report and top genes
    report_file = output_dir / f"report_{accession}.json"
    top_genes = report.pop('top_genes', set())

    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)

    # Save top genes for this study
    genes_file = output_dir / f"top_genes_{accession}.txt"
    with open(genes_file, 'w') as f:
        for g in sorted(top_genes):
            f.write(g + '\n')

    logger.info(f"Study {accession}: {report['status']}, "
                f"{report.get('total_cells_after_qc', 0):,} cells, "
                f"{len(top_genes):,} top genes")


def run_merge_mode(args):
    """Merge per-study outputs from SLURM array jobs.

    Usage: python preprocess_training_matrices.py --merge ...

    Reads all report_*.json and top_genes_*.txt files, combines them,
    and produces the final pruned gene list and corpus report.
    """
    output_dir = Path(args.output_dir)

    # Load core Ensembl IDs
    with open(args.rat_ensembl_ids) as f:
        core_ensembl_ids = set(line.strip() for line in f if line.strip())

    # Collect all per-study results
    global_top_genes = set()
    all_reports = []

    for genes_file in sorted(output_dir.glob('top_genes_*.txt')):
        with open(genes_file) as f:
            genes = set(line.strip() for line in f if line.strip())
        global_top_genes.update(genes)

    for report_file in sorted(output_dir.glob('report_*.json')):
        with open(report_file) as f:
            report = json.load(f)
        all_reports.append(report)

    logger.info(f"Merged {len(all_reports)} study reports, "
                f"{len(global_top_genes):,} unique top genes")

    # Write pruned gene list
    pruned_ids = sorted(global_top_genes & core_ensembl_ids)
    pruned_path = output_dir / 'rat_ensembl_ids_pruned.txt'
    with open(pruned_path, 'w') as f:
        for gid in pruned_ids:
            f.write(gid + '\n')

    n_success = sum(1 for r in all_reports if r.get('status') == 'success')
    total_cells = sum(r.get('total_cells_after_qc', 0) for r in all_reports)

    logger.info(f"Pruned gene list: {len(pruned_ids):,} / {len(core_ensembl_ids):,} IDs")
    logger.info(f"Studies: {n_success} success / {len(all_reports)} total")
    logger.info(f"Total QC'd cells: {total_cells:,}")

    # Write merged report
    merged_report = {
        'timestamp': datetime.now().isoformat(),
        'summary': {
            'studies_success': n_success,
            'studies_total': len(all_reports),
            'total_cells_after_qc': total_cells,
            'core_ensembl_ids': len(core_ensembl_ids),
            'pruned_ensembl_ids': len(pruned_ids),
            'pruning_rate': 1 - len(pruned_ids) / max(1, len(core_ensembl_ids)),
        },
        'studies': all_reports,
    }
    with open(output_dir / 'preprocessing_report_merged.json', 'w') as f:
        json.dump(merged_report, f, indent=2, default=str)


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Stage 3: Preprocess training matrices — Cell QC + Normalize + Rank → Pruned Gene List",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full corpus (sequential):
  python preprocess_training_matrices.py \\
      --data-root /depot/reese18/data \\
      --gene-inventory /depot/reese18/data/training/gene_inventory/gene_inventory.tsv \\
      --rat-ensembl-ids /depot/reese18/data/training/gene_inventory/rat_ensembl_ids.txt \\
      --rat-reference /depot/reese18/data/references/biomart/rat_symbol_lookup.pickle \\
      -o /depot/reese18/data/training/preprocessed/

  # Single study (for SLURM array):
  python preprocess_training_matrices.py --single-study GSE123456 \\
      --data-root /depot/reese18/data \\
      --rat-ensembl-ids /depot/reese18/data/training/gene_inventory/rat_ensembl_ids.txt \\
      --rat-reference /depot/reese18/data/references/biomart/rat_symbol_lookup.pickle \\
      -o /depot/reese18/data/training/preprocessed/

  # Merge SLURM array outputs:
  python preprocess_training_matrices.py --merge \\
      --rat-ensembl-ids /depot/reese18/data/training/gene_inventory/rat_ensembl_ids.txt \\
      -o /depot/reese18/data/training/preprocessed/
        """
    )

    parser.add_argument('--data-root', required=False, default='/depot/reese18/data',
                        help='Root data directory')
    parser.add_argument('--gene-inventory',
                        help='Path to gene_inventory.tsv from Stage 2')
    parser.add_argument('--rat-ensembl-ids', required=True,
                        help='Path to rat_ensembl_ids.txt (biotype-filtered)')
    parser.add_argument('--rat-reference', default=None,
                        help='Path to rat_symbol_lookup.pickle')
    parser.add_argument('-o', '--output-dir', required=True,
                        help='Output directory')
    parser.add_argument('--workers', type=int, default=1,
                        help='Number of parallel workers (1 = sequential)')
    parser.add_argument('--save-matrices', action='store_true', default=True,
                        help='Save QC\'d h5ad matrices (default: True)')
    parser.add_argument('--no-save-matrices', action='store_false', dest='save_matrices',
                        help='Skip saving QC\'d matrices')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Verbose logging')

    # SLURM array support
    parser.add_argument('--single-study', default=None,
                        help='Process a single study (for SLURM array)')
    parser.add_argument('--merge', action='store_true',
                        help='Merge per-study outputs from SLURM array')

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Check dependencies
    if MISSING_DEPS:
        logger.error(f"Missing required packages: {', '.join(MISSING_DEPS)}")
        logger.error("Install with: pip install numpy scipy h5py anndata scanpy pandas")
        sys.exit(1)

    # Dispatch to appropriate mode
    if args.merge:
        run_merge_mode(args)
    elif args.single_study:
        run_single_study_mode(args)
    else:
        if args.rat_reference is None:
            args.rat_reference = str(Path(args.data_root) / 'references' / 'biomart' / 'rat_symbol_lookup.pickle')
        run_corpus_preprocessing(args)


if __name__ == '__main__':
    main()