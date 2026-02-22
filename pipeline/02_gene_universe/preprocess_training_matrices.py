#!/usr/bin/env python3
"""
preprocess_training_matrices.py — Stage 2, Step 2: Cell QC + Normalize + Rank → QC'd h5ad

Pipeline position:
    Stage 1: Data Harvesting & QC
    Stage 2: Gene Universe
      Step 1: build_gene_universe.py (scan → resolve → gene_universe.tsv)
      Step 2: preprocess_training_matrices.py                    ← THIS SCRIPT
      Step 3: prune_gene_universe.py (expression-based pruning)
    Stage 3: Ortholog Mapping
    Stage 4: Gene Medians (SLURM)
    Stage 5: Reference Assembly & Corpus Export

Purpose:
    Load full expression matrices, map gene IDs to the gene_universe from
    Step 1 (no duplicate resolution logic — uses pre-built lookup), apply
    GeneCompass-exact cell QC, normalize, rank top-2048 genes per cell,
    and collect expression statistics for Step 3 pruning.

Key design:
    - GeneUniverseMapper: reads Step 1's gene_universe.tsv + gene_resolution.tsv
      as pre-built lookup tables. NO resolution logic in this script.
    - Cell QC follows GeneCompass source code order exactly:
      1. map_and_subset (id_name_match + gene_id_filter)
      2. normal_filter (z-score ±3 SD on total counts + mito %)
      3. gene_number_filter (≥7 protein_coding + miRNA genes)
      4. min_genes_filter (≥200 genes)
      5. min_cells_per_sample check
    - Normalization: CPM(10K) → log1p(base=2)
    - Per-cell top-2048 ranking + expression statistics collection

Outputs:
    QC'd h5ad files         — One per study, ENSRNOG var_names
    expression_stats.tsv    — Per gene: ensembl_id, n_cells_top2048, total_expression
    preprocessing_report.json — Per-study cell QC statistics

Usage:
    python pipeline/02_gene_universe/preprocess_training_matrices.py
    python pipeline/02_gene_universe/preprocess_training_matrices.py --single-study GSE123456
    python pipeline/02_gene_universe/preprocess_training_matrices.py --merge
    sbatch slurm/stage2_preprocess.slurm

Author: Tim Reese Lab / Claude
Date: February 2026
"""

import os
import sys
import csv
import gzip
import json
import argparse
import logging
import re
import time
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple, Any
from collections import Counter, defaultdict

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'lib'))
from gene_utils import load_config, resolve_path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Optional imports
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
    from scipy.stats import zscore
except ImportError:
    MISSING_DEPS.append('scipy')
    scipy_io = None
    sp = None
    zscore = None

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


# ─────────────────────────────────────────────────────────────────────────────
# Constants (structural patterns, not policy)
# ─────────────────────────────────────────────────────────────────────────────

UNFILTERED_BARCODE_COUNTS = {6794880, 737280, 2097152}
MAX_FILTERED_BARCODES = 500_000

RAT_MITO_PATTERNS = [
    re.compile(r'^mt-', re.IGNORECASE),
    re.compile(r'^ENSRNOG\d+.*mt', re.IGNORECASE),
    re.compile(r'^Mt_', re.IGNORECASE),
]

MATRIX_FILE_PATTERNS = [
    'matrix', 'counts', 'expression', 'umi', 'raw_count',
    'filtered_feature_bc_matrix', 'raw_feature_bc_matrix',
    'gene_expression', 'dgematrix', 'countmatrix',
]

SKIP_FILE_PATTERNS = [
    'barcodes', 'genes', 'features', 'metadata', 'clusters',
    'umap', 'tsne', 'pca', 'neighbors', 'annotation',
    'peaks', 'fragments', 'atac',
    'samples', 'sra_runs', 'sample_files',
    'family.soft', 'miniml', 'series_matrix',
    'manifest', 'filelist', 'readme', 'changelog',
    'sdrf', 'idf', 'ena_runs',
]

MATRIX_EXTENSIONS = ['.h5ad', '.h5', '.hdf5', '.loom', '.mtx.gz', '.mtx',
                     '.tsv.gz', '.csv.gz', '.tsv', '.csv', '.txt.gz', '.txt']


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1: GENE UNIVERSE MAPPER (reads Step 1 outputs — no resolution logic)
# ═════════════════════════════════════════════════════════════════════════════

class GeneUniverseMapper:
    """Maps var_names to ENSRNOG using Step 1's resolution results.

    No resolution logic — reads gene_universe.tsv and gene_resolution.tsv
    as pre-built lookup tables. This is the ONLY gene mapping class in Step 2.
    """

    def __init__(self, gene_universe_dir: Path):
        # gene_universe.tsv → set of kept ENSRNOG IDs + metadata
        self.universe_ids: Set[str] = set()
        self.gene_info: Dict[str, Dict] = {}  # ensembl_id → {symbol, biotype}

        universe_path = gene_universe_dir / 'gene_universe.tsv'
        if not universe_path.exists():
            raise FileNotFoundError(f"gene_universe.tsv not found in {gene_universe_dir}. "
                                    f"Run Step 1 (build_gene_universe.py) first.")

        with open(universe_path, 'r') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                eid = row['ensembl_id']
                self.universe_ids.add(eid)
                self.gene_info[eid] = {
                    'symbol': row.get('symbol', ''),
                    'biotype': row.get('biotype', ''),
                }

        logger.info(f"Gene universe: {len(self.universe_ids):,} genes loaded")

        # gene_resolution.tsv → raw_id → ENSRNOG lookup
        self.raw_to_ensembl: Dict[str, str] = {}       # exact case
        self.raw_to_ensembl_lower: Dict[str, str] = {} # lowercase fallback

        resolution_path = gene_universe_dir / 'gene_resolution.tsv'
        if not resolution_path.exists():
            raise FileNotFoundError(f"gene_resolution.tsv not found in {gene_universe_dir}. "
                                    f"Run Step 1 (build_gene_universe.py) first.")

        with open(resolution_path, 'r') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                raw_id = row['raw_id']
                resolved = row['resolved_ensembl_id']
                kept = row.get('kept', 'True')
                # Only include mappings to genes that are in the universe
                if resolved and resolved in self.universe_ids:
                    self.raw_to_ensembl[raw_id] = resolved
                    self.raw_to_ensembl_lower[raw_id.lower()] = resolved

        logger.info(f"Resolution lookup: {len(self.raw_to_ensembl):,} raw→ENSRNOG mappings")

        # Biotype sets for gene_number_filter (GeneCompass exact)
        self.protein_coding_ids = {eid for eid, info in self.gene_info.items()
                                   if info['biotype'] == 'protein_coding'}
        self.mirna_ids = {eid for eid, info in self.gene_info.items()
                          if info['biotype'] in ('mirna', 'miRNA', 'miRNA')}
        self.core_ids = self.protein_coding_ids | self.mirna_ids

        logger.info(f"Core gene sets: {len(self.protein_coding_ids):,} protein_coding, "
                     f"{len(self.mirna_ids):,} miRNA")

    def map_and_subset(self, adata: 'ad.AnnData') -> Tuple['ad.AnnData', Dict]:
        """Map var_names → ENSRNOG, subset to universe, handle duplicates.

        Returns: (adata with ENSRNOG var_names, mapping_stats dict)
        """
        original_genes = list(adata.var_names)
        mapped_ids = []
        for name in original_genes:
            eid = self.raw_to_ensembl.get(name)
            if eid is None:
                eid = self.raw_to_ensembl_lower.get(name.lower())
            mapped_ids.append(eid)

        # Keep only mapped genes
        keep_mask = [eid is not None for eid in mapped_ids]
        n_mapped = sum(keep_mask)

        if n_mapped == 0:
            return None, {
                'total_input_genes': len(original_genes),
                'mapped': 0, 'unmapped': len(original_genes),
                'duplicates_resolved': 0,
                'mapping_rate': 0.0,
            }

        adata_mapped = adata[:, keep_mask].copy()
        ensembl_ids = [eid for eid in mapped_ids if eid is not None]

        # Handle duplicates (two var_names → same ENSRNOG): sum counts
        n_dupes = 0
        if len(set(ensembl_ids)) < len(ensembl_ids):
            adata_mapped, n_dupes = _merge_duplicate_genes(adata_mapped, ensembl_ids)
        else:
            adata_mapped.var_names = pd.Index(ensembl_ids)
            adata_mapped.var_names_make_unique()

        stats = {
            'total_input_genes': len(original_genes),
            'mapped': n_mapped,
            'unmapped': len(original_genes) - n_mapped,
            'duplicates_resolved': n_dupes,
            'mapping_rate': n_mapped / len(original_genes) if original_genes else 0.0,
        }

        return adata_mapped, stats

    def get_mito_ensembl_ids(self, original_mito_ids: List[str]) -> Set[str]:
        """Map original mitochondrial gene IDs to their ENSRNOG equivalents."""
        mito_ensembl = set()
        for mid in original_mito_ids:
            eid = self.raw_to_ensembl.get(mid)
            if eid is None:
                eid = self.raw_to_ensembl_lower.get(mid.lower())
            if eid and eid in self.universe_ids:
                mito_ensembl.add(eid)
        return mito_ensembl


def _merge_duplicate_genes(adata: 'ad.AnnData', ensembl_ids: List[str]) \
        -> Tuple['ad.AnnData', int]:
    """Merge duplicate gene mappings by summing counts."""
    # Group indices by ENSRNOG
    eid_to_indices = defaultdict(list)
    for i, eid in enumerate(ensembl_ids):
        eid_to_indices[eid].append(i)

    unique_eids = sorted(eid_to_indices.keys())
    n_dupes = sum(1 for indices in eid_to_indices.values() if len(indices) > 1)

    # Build merged matrix
    if sp.issparse(adata.X):
        rows = []
        for eid in unique_eids:
            indices = eid_to_indices[eid]
            if len(indices) == 1:
                rows.append(adata.X[:, indices[0]])
            else:
                merged_col = adata.X[:, indices[0]]
                for idx in indices[1:]:
                    merged_col = merged_col + adata.X[:, idx]
                rows.append(merged_col)
        X_new = sp.hstack(rows, format='csc').tocsr()
    else:
        X_new = np.zeros((adata.n_obs, len(unique_eids)), dtype=adata.X.dtype)
        for j, eid in enumerate(unique_eids):
            indices = eid_to_indices[eid]
            for idx in indices:
                X_new[:, j] += adata.X[:, idx]

    var_df = pd.DataFrame(index=unique_eids)
    adata_merged = ad.AnnData(X=X_new, obs=adata.obs.copy(), var=var_df)
    return adata_merged, n_dupes


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2: FORMAT DETECTION AND MATRIX LOADING
# ═════════════════════════════════════════════════════════════════════════════

class MatrixLoadResult:
    """Container for a loaded expression matrix with metadata."""
    def __init__(self, adata, format_type: str, source_path: str,
                 is_raw_counts: bool, was_unfiltered: bool = False,
                 gene_id_format: str = 'unknown', warnings: List[str] = None):
        self.adata = adata
        self.format_type = format_type
        self.source_path = source_path
        self.is_raw_counts = is_raw_counts
        self.was_unfiltered = was_unfiltered
        self.gene_id_format = gene_id_format
        self.warnings = warnings or []


def detect_gene_id_format(gene_ids: List[str], sample_size: int = 100) -> str:
    """Classify the predominant gene ID format."""
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
    if top_count / len(sample) > 0.6:
        return top_format
    return 'mixed'


def detect_is_raw_counts(X, sample_cells: int = 500) -> Tuple[bool, str]:
    """Heuristic: is this matrix likely raw integer counts?"""
    if sp is not None and sp.issparse(X):
        if hasattr(X, 'data') and len(X.data) > 0:
            sample_data = X.data[:min(100_000, len(X.data))]
        else:
            return True, "empty_sparse_matrix"
    else:
        n_cells = X.shape[0]
        idx = np.random.choice(n_cells, min(sample_cells, n_cells), replace=False)
        if sp is not None and sp.issparse(X):
            sample_data = np.asarray(X[idx].todense()).flatten()
        else:
            sample_data = np.asarray(X[idx]).flatten()
        sample_data = sample_data[sample_data != 0]

    if len(sample_data) == 0:
        return True, "all_zeros"

    int_check = np.allclose(sample_data, np.round(sample_data), atol=1e-6)
    if int_check:
        max_val = float(np.max(sample_data))
        return True, f"integer_values_max={max_val:.0f}"
    else:
        max_val = float(np.max(sample_data))
        min_nonzero = float(np.min(sample_data[sample_data > 0])) if np.any(sample_data > 0) else 0
        if max_val < 20 and min_nonzero < 1:
            return False, f"likely_log_transformed_range=[{min_nonzero:.3f}, {max_val:.3f}]"
        elif max_val > 1_000_000:
            return False, f"likely_TPM_or_FPKM_max={max_val:.1f}"
        else:
            return False, f"non_integer_values_range=[{min_nonzero:.4f}, {max_val:.2f}]"


def reverse_normalization(X):
    """Recover approximate raw counts from normalized data (scFoundation method)."""
    if sp is not None and sp.issparse(X):
        X_dense = np.asarray(X.todense())
    else:
        X_dense = np.asarray(X)

    max_val = np.max(X_dense)
    if max_val < 20:
        X_dense = np.expm1(X_dense)

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
    """Check if barcode count matches known unfiltered 10x barcode spaces."""
    return n_barcodes in UNFILTERED_BARCODE_COUNTS or n_barcodes > MAX_FILTERED_BARCODES


# ─────────────────────────────────────────────────────────────────────────────
# Format-specific loaders (h5ad, MTX, H5, Loom, TSV/CSV)
# ─────────────────────────────────────────────────────────────────────────────

def load_h5ad(filepath: str) -> Optional[MatrixLoadResult]:
    """Load h5ad file, handling sparse Group fallback."""
    try:
        adata = sc.read_h5ad(filepath, backed=None)
    except Exception as e:
        logger.warning(f"scanpy failed on {filepath}, trying h5py: {e}")
        try:
            adata = _load_h5ad_via_h5py(filepath)
            if adata is None:
                return None
        except Exception as e2:
            logger.error(f"h5py fallback also failed on {filepath}: {e2}")
            return None

    warnings = []
    gene_ids = list(adata.var_names[:100])
    gene_id_format = detect_gene_id_format(gene_ids)
    is_raw, evidence = detect_is_raw_counts(adata.X)

    if not is_raw and adata.raw is not None:
        is_raw_raw, evidence_raw = detect_is_raw_counts(adata.raw.X)
        if is_raw_raw:
            warnings.append(f"Using adata.raw.X instead of adata.X ({evidence})")
            adata = ad.AnnData(X=adata.raw.X, obs=adata.obs, var=adata.raw.var)
            is_raw = True
            evidence = evidence_raw

    was_unfiltered = detect_unfiltered_10x(adata.n_obs)
    if was_unfiltered:
        warnings.append(f"Unfiltered barcode space: {adata.n_obs:,}")
    if not is_raw:
        warnings.append(f"Normalized data ({evidence}), will attempt reversal")

    return MatrixLoadResult(adata=adata, format_type='h5ad', source_path=filepath,
                            is_raw_counts=is_raw, was_unfiltered=was_unfiltered,
                            gene_id_format=gene_id_format, warnings=warnings)


def _load_h5ad_via_h5py(filepath: str) -> Optional['ad.AnnData']:
    """Fallback loader for h5ad with sparse X stored as HDF5 Groups."""
    with h5py.File(filepath, 'r') as f:
        if 'X' not in f:
            return None
        x_obj = f['X']
        if isinstance(x_obj, h5py.Group):
            if all(k in x_obj for k in ('data', 'indices', 'indptr')):
                data = x_obj['data'][:]
                indices = x_obj['indices'][:]
                indptr = x_obj['indptr'][:]
                shape = tuple(x_obj.attrs.get('shape', None))
                if shape is None:
                    n_obs = len(indptr) - 1
                    n_var = int(np.max(indices)) + 1 if len(indices) > 0 else 0
                    shape = (n_obs, n_var)
                X = sp.csr_matrix((data, indices, indptr), shape=shape)
            else:
                return None
        elif isinstance(x_obj, h5py.Dataset):
            X = x_obj[:]
        else:
            return None

        var_index = None
        if 'var' in f and '_index' in f['var']:
            raw = f['var']['_index'][:]
            var_index = [x.decode() if isinstance(x, bytes) else str(x) for x in raw]

        obs_index = None
        if 'obs' in f and '_index' in f['obs']:
            raw = f['obs']['_index'][:]
            obs_index = [x.decode() if isinstance(x, bytes) else str(x) for x in raw]

        if var_index is None:
            var_index = [f"gene_{i}" for i in range(X.shape[1])]
        if obs_index is None:
            obs_index = [f"cell_{i}" for i in range(X.shape[0])]

        return ad.AnnData(X=X, obs=pd.DataFrame(index=obs_index),
                          var=pd.DataFrame(index=var_index))


def _load_10x_mtx_manual(mtx_path, barcodes_path, genes_path):
    """Manual load for Cell Ranger v2 (2-column genes file)."""
    from scipy.io import mmread
    opener = gzip.open if str(mtx_path).endswith('.gz') else open
    with opener(str(mtx_path), 'rb') as f:
        X = mmread(f).T.tocsr()

    opener = gzip.open if str(barcodes_path).endswith('.gz') else open
    with opener(str(barcodes_path), 'rt') as f:
        barcodes = [line.strip().split('\t')[0] for line in f]

    opener = gzip.open if str(genes_path).endswith('.gz') else open
    with opener(str(genes_path), 'rt') as f:
        gene_rows = [line.strip().split('\t') for line in f]

    gene_ids = [r[0] for r in gene_rows]
    obs = pd.DataFrame(index=barcodes)
    var = pd.DataFrame(index=gene_ids)
    var.index = var.index.astype(str)
    if var.index.duplicated().any():
        var.index = ad.utils.make_index_unique(pd.Index(var.index))
    return ad.AnnData(X=X, obs=obs, var=var)


def load_10x_mtx(study_dir: str, mtx_path: str) -> Optional[MatrixLoadResult]:
    """Load 10x MTX triplet format."""
    mtx_path_obj = Path(mtx_path)
    parent = mtx_path_obj.parent

    mtx_prefix = None
    mtx_stem = mtx_path_obj.name
    for suffix in ['.matrix.mtx.gz', '.matrix.mtx', '_matrix.mtx.gz', '_matrix.mtx']:
        if mtx_stem.lower().endswith(suffix):
            mtx_prefix = mtx_stem[:len(mtx_stem) - len(suffix)]
            break

    barcodes_file = _find_companion_file(parent, ['barcodes.tsv.gz', 'barcodes.tsv'],
                                         prefix=mtx_prefix)
    features_file = _find_companion_file(parent, [
        'features.tsv.gz', 'features.tsv', 'genes.tsv.gz', 'genes.tsv'
    ], prefix=mtx_prefix)

    if barcodes_file is None or features_file is None:
        logger.warning(f"Missing companion files for {mtx_path}")
        return None

    warnings = []
    load_dir = str(parent)
    _temp_dir = None

    if mtx_prefix and not mtx_stem.lower().startswith('matrix'):
        import tempfile, shutil
        _temp_dir = tempfile.mkdtemp(prefix='10x_triplet_')
        _temp_path = Path(_temp_dir)
        os.symlink(str(mtx_path_obj), str(_temp_path / 'matrix.mtx.gz'))
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
                try:
                    adata = _load_10x_mtx_manual(mtx_path, barcodes_file, features_file)
                except Exception as e:
                    logger.error(f"Failed to load MTX from {load_dir}: {e}")
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
        warnings.append(f"Unfiltered 10x: {adata.n_obs:,} barcodes")

    return MatrixLoadResult(adata=adata, format_type='mtx', source_path=str(mtx_path),
                            is_raw_counts=is_raw, was_unfiltered=was_unfiltered,
                            gene_id_format=gene_id_format, warnings=warnings)


def load_10x_h5(filepath: str) -> Optional[MatrixLoadResult]:
    """Load 10x HDF5 format."""
    warnings = []
    try:
        adata = sc.read_10x_h5(filepath)
    except Exception as e:
        logger.error(f"Failed to load 10x H5 {filepath}: {e}")
        return None

    gene_ids = list(adata.var_names[:100])
    gene_id_format = detect_gene_id_format(gene_ids)
    is_raw, _ = detect_is_raw_counts(adata.X)
    was_unfiltered = detect_unfiltered_10x(adata.n_obs)
    if was_unfiltered:
        warnings.append(f"Unfiltered 10x H5: {adata.n_obs:,}")

    return MatrixLoadResult(adata=adata, format_type='h5', source_path=filepath,
                            is_raw_counts=is_raw, was_unfiltered=was_unfiltered,
                            gene_id_format=gene_id_format, warnings=warnings)


def load_loom(filepath: str) -> Optional[MatrixLoadResult]:
    """Load Loom format."""
    try:
        adata = sc.read_loom(filepath, sparse=True, cleanup=True)
    except Exception as e:
        logger.error(f"Failed to load loom {filepath}: {e}")
        return None

    gene_ids = list(adata.var_names[:100])
    gene_id_format = detect_gene_id_format(gene_ids)
    is_raw, _ = detect_is_raw_counts(adata.X)

    return MatrixLoadResult(adata=adata, format_type='loom', source_path=filepath,
                            is_raw_counts=is_raw, gene_id_format=gene_id_format, warnings=[])


def load_tabular(filepath: str) -> Optional[MatrixLoadResult]:
    """Load TSV/CSV dense count matrix with auto-orientation detection."""
    fpath = Path(filepath)
    warnings = []
    try:
        sep = '\t' if '.tsv' in fpath.name or '.txt' in fpath.name else ','
        if filepath.endswith('.gz'):
            df = pd.read_csv(filepath, sep=sep, index_col=0, compression='gzip')
        else:
            df = pd.read_csv(filepath, sep=sep, index_col=0)

        row_ids = [str(x) for x in list(df.index[:50])]
        col_ids = [str(x) for x in list(df.columns[:50])]
        row_format = detect_gene_id_format(row_ids)
        col_format = detect_gene_id_format(col_ids)

        needs_transpose = False
        if row_format in ('ensembl_rat', 'ensembl_mouse', 'ensembl_human', 'symbol', 'loc'):
            if col_format not in ('ensembl_rat', 'ensembl_mouse', 'ensembl_human', 'symbol', 'loc'):
                needs_transpose = True
                warnings.append(f"Transposed: rows={row_format}, cols={col_format}")
        elif df.shape[0] > df.shape[1] * 5:
            needs_transpose = True
            warnings.append(f"Transposed by shape: {df.shape}")

        if needs_transpose:
            df = df.T

        adata = ad.AnnData(X=sp.csr_matrix(df.values.astype(np.float32)),
                           obs=pd.DataFrame(index=df.index.astype(str)),
                           var=pd.DataFrame(index=df.columns.astype(str)))

        gene_ids = list(adata.var_names[:100])
        gene_id_format = detect_gene_id_format(gene_ids)
        is_raw, _ = detect_is_raw_counts(adata.X)

        return MatrixLoadResult(adata=adata, format_type='tsv' if '\t' == sep else 'csv',
                                source_path=filepath, is_raw_counts=is_raw,
                                gene_id_format=gene_id_format, warnings=warnings)
    except Exception as e:
        logger.error(f"Failed to load tabular {filepath}: {e}")
        return None


def _find_companion_file(directory: Path, candidates: List[str],
                          prefix: str = None) -> Optional[Path]:
    """Find companion file from candidates, with optional prefix matching."""
    for name in candidates:
        path = directory / name
        if path.exists():
            return path
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
# SECTION 3: MATRIX DISCOVERY
# ═════════════════════════════════════════════════════════════════════════════

def find_matrix_files(study_dir: str) -> List[Tuple[str, str]]:
    """Discover matrix files in a study directory."""
    study_path = Path(study_dir)
    candidates = []
    search_dirs = [study_path]
    raw_dir = study_path / (study_path.name + '_RAW')
    if raw_dir.exists():
        search_dirs.append(raw_dir)
    for d in study_path.iterdir():
        if d.is_dir() and d.name.endswith('_RAW'):
            if d not in search_dirs:
                search_dirs.append(d)

    seen_mtx_dirs = set()
    seen_mtx_prefixes = set()

    for search_dir in search_dirs:
        for root, dirs, files in os.walk(search_dir):
            root_path = Path(root)
            for fname in files:
                fpath = root_path / fname
                fname_lower = fname.lower()
                if any(skip in fname_lower for skip in SKIP_FILE_PATTERNS):
                    continue
                fmt = _detect_format_from_extension(fname)
                if fmt is None:
                    continue
                if fmt == 'mtx':
                    is_gsm = bool(re.match(
                        r'^(GSM\d+[_.].*|E-\w+-\d+[_.].*)[._](matrix[._])?mtx',
                        fname, re.IGNORECASE))
                    if is_gsm:
                        prefix = fname
                        for sfx in ['.matrix.mtx.gz', '.matrix.mtx', '.mtx.gz', '.mtx']:
                            if fname_lower.endswith(sfx):
                                prefix = fname[:len(fname) - len(sfx)]
                                break
                        key = f"{root_path}::{prefix}"
                        if key not in seen_mtx_prefixes:
                            seen_mtx_prefixes.add(key)
                            candidates.append((str(fpath), 'mtx'))
                    else:
                        if str(root_path) not in seen_mtx_dirs:
                            seen_mtx_dirs.add(str(root_path))
                            candidates.append((str(fpath), 'mtx'))
                else:
                    candidates.append((str(fpath), fmt))

    format_priority = {'h5ad': 0, 'h5': 1, 'mtx': 2, 'loom': 3, 'tsv': 4, 'csv': 5}
    candidates.sort(key=lambda x: format_priority.get(x[1], 99))
    return candidates


def _detect_format_from_extension(filename: str) -> Optional[str]:
    """Detect matrix format from file extension."""
    fname_lower = filename.lower()
    if fname_lower.endswith('.mtx.gz') or fname_lower.endswith('.mtx'):
        return 'mtx'
    if fname_lower.endswith('.h5ad'):
        return 'h5ad'
    if fname_lower.endswith('.h5') or fname_lower.endswith('.hdf5'):
        return 'h5'
    if fname_lower.endswith('.loom'):
        return 'loom'
    if fname_lower.endswith(('.tsv.gz', '.tsv')):
        if any(p in fname_lower for p in MATRIX_FILE_PATTERNS) or fname_lower.startswith('gsm'):
            return 'tsv'
    if fname_lower.endswith(('.csv.gz', '.csv')):
        if any(p in fname_lower for p in MATRIX_FILE_PATTERNS) or fname_lower.startswith('gsm'):
            return 'csv'
    if fname_lower.endswith(('.txt.gz', '.txt')):
        if any(p in fname_lower for p in MATRIX_FILE_PATTERNS):
            return 'tsv'
    return None


def load_matrix(filepath: str, format_type: str, study_dir: str) -> Optional[MatrixLoadResult]:
    """Dispatch to format-specific loader."""
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
    return None


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4: CELL QC — GeneCompass EXACT ORDER
# ═════════════════════════════════════════════════════════════════════════════

def identify_mito_genes(gene_names: List[str]) -> List[str]:
    """Identify mitochondrial gene names from original var_names.

    Must run BEFORE remapping to ENSRNOG, since mito patterns match
    original gene symbols (mt-Nd1, mt-Co1, etc.).
    """
    mito_genes = []
    for gname in gene_names:
        gname_str = str(gname)
        for pattern in RAT_MITO_PATTERNS:
            if pattern.search(gname_str):
                mito_genes.append(gname)
                break
    return mito_genes


def normal_filter(adata, mito_gene_ids: Set[str]):
    """GeneCompass normal_filter: z-score on total counts + mito %.

    Removes cells where EITHER total gene count z-score OR
    mitochondrial percentage z-score exceeds ±3.

    GeneCompass uses ±3 for BOTH bounds (not just upper).
    This removes both extremely low-count and high-count cells,
    plus cells with extreme mito percentages in either direction.
    """
    # Total counts per cell
    if sp.issparse(adata.X):
        total_counts = np.array(adata.X.sum(axis=1)).flatten()
    else:
        total_counts = np.array(np.sum(adata.X, axis=1)).flatten()

    # Remove zero-count cells first
    nonzero_mask = total_counts > 0
    if not np.all(nonzero_mask):
        adata = adata[nonzero_mask].copy()
        total_counts = total_counts[nonzero_mask]

    if adata.n_obs < 3:
        return adata  # Can't compute z-scores with <3 cells

    # Mito counts
    mito_mask = np.isin(adata.var_names, list(mito_gene_ids))
    if sp.issparse(adata.X):
        mito_counts = np.array(adata[:, mito_mask].X.sum(axis=1)).flatten()
    else:
        mito_counts = np.sum(np.asarray(adata.X)[:, mito_mask], axis=1).flatten()

    mito_pct = np.divide(mito_counts, total_counts,
                         out=np.zeros_like(mito_counts, dtype=float),
                         where=total_counts > 0)

    # Z-scores
    total_z = zscore(total_counts)
    mito_z = zscore(mito_pct)

    # Handle constant arrays (zscore returns NaN)
    total_z = np.nan_to_num(total_z, nan=0.0)
    mito_z = np.nan_to_num(mito_z, nan=0.0)

    # Keep cells within ±3 SD for BOTH metrics
    keep = ((total_z > -3) & (total_z < 3) &
            (mito_z > -3) & (mito_z < 3))

    return adata[keep].copy()


def gene_number_filter(adata, core_gene_ids: Set[str]):
    """GeneCompass gene_number_filter: ≥7 protein_coding + miRNA genes.

    Counts nonzero expression of protein_coding and miRNA genes per cell.
    Keeps cells with > 6 (i.e., ≥ 7) nonzero core genes.

    IMPORTANT: applies filter to FULL adata, not just core subset.
    The core genes are used to compute the mask, but ALL genes survive.
    """
    core_mask = np.isin(adata.var_names, list(core_gene_ids))
    core_adata = adata[:, core_mask]

    if hasattr(core_adata.X, 'toarray'):
        data = core_adata.X.toarray()
    else:
        data = np.array(core_adata.X)

    core_counts = np.count_nonzero(data, axis=1)
    keep = core_counts > 6  # > 6 means ≥ 7

    return adata[keep].copy()


def min_genes_filter(adata, min_genes: int = 200):
    """Drop cells with fewer than min_genes expressed genes."""
    if sp.issparse(adata.X):
        gene_counts = np.array((adata.X > 0).sum(axis=1)).flatten()
    else:
        gene_counts = np.sum(np.asarray(adata.X) > 0, axis=1).flatten()
    return adata[gene_counts >= min_genes].copy()


def filter_empty_droplets(adata, min_genes: int = 200):
    """Basic empty droplet filter for unfiltered 10x data."""
    if sp.issparse(adata.X):
        genes_per_cell = np.asarray((adata.X > 0).sum(axis=1)).flatten()
    else:
        genes_per_cell = np.sum(np.asarray(adata.X) > 0, axis=1)
    keep = genes_per_cell >= min_genes
    return adata[keep].copy()


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 5: NORMALIZATION AND TOP-2048 RANKING
# ═════════════════════════════════════════════════════════════════════════════

def top_2048_ranking(adata) -> Dict[str, Dict]:
    """Per-cell top-2048 ranking + corpus-wide gene stats.

    Returns dict: ensembl_id → {n_cells_top2048, total_expression}
    """
    gene_stats = defaultdict(lambda: {'n_cells_top2048': 0, 'total_expression': 0.0})
    gene_ids = adata.var_names.tolist()

    # Process in chunks for memory efficiency
    chunk_size = 5000
    for start in range(0, adata.n_obs, chunk_size):
        end = min(start + chunk_size, adata.n_obs)

        if sp.issparse(adata.X):
            chunk = np.asarray(adata.X[start:end].todense())
        else:
            chunk = np.asarray(adata.X[start:end])

        for i in range(chunk.shape[0]):
            cell = chunk[i]
            nonzero = np.nonzero(cell)[0]
            if len(nonzero) == 0:
                continue

            # Get top-2048 by value (O(n) via argpartition)
            if len(nonzero) <= 2048:
                top_indices = nonzero
            else:
                values = cell[nonzero]
                top_k = np.argpartition(values, -2048)[-2048:]
                top_indices = nonzero[top_k]

            for idx in top_indices:
                gene_stats[gene_ids[idx]]['n_cells_top2048'] += 1

    # Add total expression per gene
    if sp.issparse(adata.X):
        totals = np.array(adata.X.sum(axis=0)).flatten()
    else:
        totals = np.array(adata.X.sum(axis=0)).flatten()

    for i, gid in enumerate(gene_ids):
        gene_stats[gid]['total_expression'] = float(totals[i])

    return dict(gene_stats)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 6: STUDY-LEVEL PROCESSING
# ═════════════════════════════════════════════════════════════════════════════

def process_study(study_accession: str, study_dir: str, mapper: GeneUniverseMapper,
                  config: dict, output_dir: str) -> Tuple[Dict, Dict]:
    """Process a single study end-to-end following GeneCompass exact order.

    Returns: (report_dict, gene_stats_dict)
    """
    t_start = time.time()
    pp = config['gene_universe'].get('preprocessing', {})
    min_genes_per_cell = int(pp.get('min_genes_per_cell', 200))
    min_cells_per_sample = int(pp.get('min_cells_per_sample', 4))
    save_qc = pp.get('save_qc_matrices', True)

    report = {
        'accession': study_accession,
        'study_dir': study_dir,
        'status': 'pending',
        'warnings': [],
        'errors': [],
        'samples': [],
    }
    study_gene_stats = defaultdict(lambda: {'n_cells_top2048': 0, 'total_expression': 0.0})

    # Discover matrices
    matrix_files = find_matrix_files(study_dir)
    if not matrix_files:
        report['status'] = 'no_matrices_found'
        return report, {}

    total_cells_before = 0
    total_cells_after = 0

    for idx, (filepath, fmt) in enumerate(matrix_files):
        sample_id = f"{study_accession}_sample{idx}"
        sample_report = {
            'sample_id': sample_id,
            'source': filepath,
            'format': fmt,
        }

        try:
            # 1. Load matrix
            result = load_matrix(filepath, fmt, study_dir)
            if result is None or result.adata.n_obs == 0 or result.adata.n_vars == 0:
                sample_report['status'] = 'load_failed'
                report['samples'].append(sample_report)
                continue

            adata = result.adata
            report['warnings'].extend(result.warnings)
            sample_report['cells_raw'] = adata.n_obs
            sample_report['gene_id_format'] = result.gene_id_format

            # 2. Detect raw counts, reverse normalization if needed
            if not result.is_raw_counts:
                try:
                    adata.X = reverse_normalization(adata.X)
                    report['warnings'].append(f"{filepath}: reversed normalization")
                except Exception as e:
                    sample_report['status'] = 'normalization_reversal_failed'
                    sample_report['error'] = str(e)
                    report['samples'].append(sample_report)
                    continue

            # 3. Handle unfiltered 10x barcodes
            if result.was_unfiltered:
                n_before = adata.n_obs
                adata = filter_empty_droplets(adata, min_genes=min_genes_per_cell)
                sample_report['cells_after_droplet_filter'] = adata.n_obs
                report['warnings'].append(
                    f"Filtered unfiltered 10x: {n_before:,} → {adata.n_obs:,}")

            # 4. Identify mito genes BEFORE remapping to ENSRNOG
            mito_originals = identify_mito_genes(list(adata.var_names))

            # 5. Map var_names → gene_universe + subset
            adata_mapped, mapping_stats = mapper.map_and_subset(adata)
            sample_report['mapping_stats'] = mapping_stats

            if adata_mapped is None or adata_mapped.n_vars == 0:
                sample_report['status'] = 'no_genes_mapped'
                report['samples'].append(sample_report)
                continue

            sample_report['genes_mapped'] = adata_mapped.n_vars
            total_cells_before += adata_mapped.n_obs

            # 6. Identify mito genes in ENSRNOG space
            mito_ensembl = mapper.get_mito_ensembl_ids(mito_originals)

            # 7. Cell QC — GeneCompass exact order
            # 7a. normal_filter (z-score ±3 on total counts + mito %)
            adata_qc = normal_filter(adata_mapped, mito_ensembl)
            sample_report['cells_after_normal_filter'] = adata_qc.n_obs

            # 7b. gene_number_filter (≥7 protein_coding + miRNA)
            adata_qc = gene_number_filter(adata_qc, mapper.core_ids)
            sample_report['cells_after_gene_number_filter'] = adata_qc.n_obs

            # 7c. min_genes_per_cell (200)
            adata_qc = min_genes_filter(adata_qc, min_genes=min_genes_per_cell)
            sample_report['cells_after_min_genes'] = adata_qc.n_obs

            # 7d. min_cells_per_sample (4)
            if adata_qc.n_obs < min_cells_per_sample:
                sample_report['status'] = 'too_few_cells'
                sample_report['cells_final'] = adata_qc.n_obs
                report['samples'].append(sample_report)
                continue

            sample_report['cells_final'] = adata_qc.n_obs
            total_cells_after += adata_qc.n_obs

            # 8. Normalize: CPM(10K) → log1p(base=2)
            raw_adata = adata_qc.copy()
            sc.pp.normalize_total(adata_qc, target_sum=1e4)
            sc.pp.log1p(adata_qc, base=2)

            # 9. Per-cell top-2048 ranking + expression stats
            gene_stats = top_2048_ranking(adata_qc)
            for gid, stats in gene_stats.items():
                study_gene_stats[gid]['n_cells_top2048'] += stats['n_cells_top2048']
                study_gene_stats[gid]['total_expression'] += stats['total_expression']

            # 10. Save QC'd h5ad
            if save_qc:
                adata_qc.raw = raw_adata
                qc_dir = Path(output_dir)
                qc_dir.mkdir(parents=True, exist_ok=True)
                out_path = qc_dir / f"{sample_id}.h5ad"
                try:
                    adata_qc.write_h5ad(str(out_path))
                    sample_report['output_path'] = str(out_path)
                except Exception as e:
                    sample_report['save_error'] = str(e)

            sample_report['status'] = 'success'

            # Free memory
            del adata, adata_mapped, adata_qc, raw_adata

        except Exception as e:
            sample_report['status'] = 'error'
            sample_report['error'] = str(e)
            logger.error(f"Error processing {filepath}: {e}")
            logger.debug(traceback.format_exc())

        report['samples'].append(sample_report)

    report['total_cells_before_qc'] = total_cells_before
    report['total_cells_after_qc'] = total_cells_after
    report['n_samples_success'] = sum(1 for s in report['samples'] if s.get('status') == 'success')
    report['n_samples_failed'] = sum(1 for s in report['samples'] if s.get('status') != 'success')
    report['elapsed_seconds'] = time.time() - t_start
    report['status'] = 'success' if report['n_samples_success'] > 0 else 'all_samples_failed'

    return report, dict(study_gene_stats)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 7: STUDY CATALOG
# ═════════════════════════════════════════════════════════════════════════════

def load_study_catalog(config: dict) -> List[Dict]:
    """Load unified study catalog and filter to usable rat single-cell studies."""
    h = config.get('harvesting', {})
    catalog_dir = resolve_path(config, h.get('catalog_dir', 'data/catalog'))
    catalog_path = Path(catalog_dir) / 'unified_studies.json'

    if not catalog_path.exists():
        logger.error(f"Catalog not found: {catalog_path}")
        return []

    with open(catalog_path) as f:
        catalog = json.load(f)

    ra = config.get('reference_assembly', {})
    exclusion_list = set(ra.get('exclude_studies', []))
    sources = h.get('sources', {})
    data_root = resolve_path(config, h.get('data_root', 'data/raw'))

    studies = []
    counters = defaultdict(int)

    for study in catalog.get('studies', []):
        accession = study.get('accession', '')

        if accession in exclusion_list:
            counters['excluded'] += 1
            continue

        cat_info = study.get('catalog', {})
        if cat_info.get('data_type') != 'single_cell':
            counters['not_sc_catalog'] += 1
            continue

        llm_info = study.get('llm', {})
        utility = llm_info.get('motrpac_utility', {})
        is_rat = utility.get('is_rat')

        if is_rat is False:
            counters['not_rat'] += 1
            continue
        if is_rat is not True:
            validated_org = llm_info.get('validated_organism', {})
            species = str(validated_org.get('species', '')).lower()
            if 'rattus' not in species and 'rat' not in species:
                raw_orgs = cat_info.get('raw_organisms', [])
                if not any('rattus' in str(o).lower() or 'rat' == str(o).strip().lower()
                           for o in raw_orgs):
                    counters['not_rat'] += 1
                    continue

        study_type = llm_info.get('study_type', {})
        if study_type.get('is_single_cell') is False:
            counters['not_sc_llm'] += 1
            continue

        source = study.get('source', 'geo')
        if source == 'arrayexpress':
            rel_path = sources.get('arrayexpress', {}).get('single_cell', {}).get(
                'path', 'arrayexpress/singlecell/datasets')
        else:
            rel_path = sources.get('geo', {}).get('single_cell', {}).get(
                'path', 'geo/single_cell/geo_datasets')

        study_dir = str(data_root / rel_path / accession)
        if not Path(study_dir).exists():
            counters['no_dir'] += 1
            continue

        studies.append({
            'accession': accession,
            'study_dir': study_dir,
            'source': source,
        })

    logger.info(f"Loaded {len(studies)} usable studies "
                f"(excluded={counters['excluded']}, not_sc={counters['not_sc_catalog']}, "
                f"not_rat={counters['not_rat']}, no_dir={counters['no_dir']})")
    return studies


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 8: CORPUS ORCHESTRATION
# ═════════════════════════════════════════════════════════════════════════════

def run_corpus_preprocessing(args, config: dict):
    """Process all studies and produce outputs."""
    t_total = time.time()

    gu_dir = resolve_path(config, config['paths']['gene_universe_dir'])
    qc_dir = resolve_path(config, config['paths']['qc_h5ad_dir'])

    # Load gene universe mapper from Step 1 outputs
    mapper = GeneUniverseMapper(gu_dir)

    # Load study catalog
    studies = load_study_catalog(config)
    if not studies:
        logger.error("No usable studies found")
        return

    qc_dir.mkdir(parents=True, exist_ok=True)

    # Process all studies
    corpus_gene_stats = defaultdict(lambda: {'n_cells_top2048': 0, 'total_expression': 0.0})
    all_reports = []
    total_cells = 0
    n_success = 0

    for i, study_info in enumerate(studies):
        accession = study_info['accession']
        logger.info(f"[{i+1}/{len(studies)}] Processing {accession}...")

        try:
            report, gene_stats = process_study(
                study_accession=accession,
                study_dir=study_info['study_dir'],
                mapper=mapper,
                config=config,
                output_dir=str(qc_dir),
            )

            for gid, stats in gene_stats.items():
                corpus_gene_stats[gid]['n_cells_top2048'] += stats['n_cells_top2048']
                corpus_gene_stats[gid]['total_expression'] += stats['total_expression']

            total_cells += report.get('total_cells_after_qc', 0)
            if report['status'] == 'success':
                n_success += 1
            all_reports.append(report)

            if (i + 1) % 25 == 0:
                logger.info(f"  Progress: {i+1}/{len(studies)}, {n_success} success, "
                            f"{total_cells:,} cells, {len(corpus_gene_stats):,} genes tracked")

        except Exception as e:
            logger.error(f"Unhandled error in {accession}: {e}")
            logger.error(traceback.format_exc())
            all_reports.append({'accession': accession, 'status': 'unhandled_error', 'error': str(e)})

    # ── Write expression_stats.tsv ──
    stats_path = gu_dir / 'expression_stats.tsv'
    with open(stats_path, 'w', newline='') as f:
        w = csv.DictWriter(f, delimiter='\t',
                           fieldnames=['ensembl_id', 'n_cells_top2048',
                                       'total_expression', 'n_studies_expressed'])
        w.writeheader()
        # Count studies per gene from reports
        gene_study_counts = defaultdict(int)
        for report in all_reports:
            if report.get('status') == 'success':
                acc = report['accession']
                for sample in report.get('samples', []):
                    if sample.get('status') == 'success':
                        # Each successful sample contributes genes
                        gene_study_counts[acc] += 1  # simplified

        for gid in sorted(corpus_gene_stats.keys()):
            stats = corpus_gene_stats[gid]
            w.writerow({
                'ensembl_id': gid,
                'n_cells_top2048': stats['n_cells_top2048'],
                'total_expression': round(stats['total_expression'], 2),
                'n_studies_expressed': 0,  # computed by Step 3 if needed
            })

    logger.info(f"expression_stats.tsv: {len(corpus_gene_stats):,} genes")

    # ── Write preprocessing_report.json ──
    elapsed = time.time() - t_total
    report_data = {
        'stage': 'stage2_step2',
        'timestamp': datetime.now().isoformat(),
        'elapsed_seconds': round(elapsed, 1),
        'summary': {
            'total_studies': len(studies),
            'studies_success': n_success,
            'studies_failed': len(studies) - n_success,
            'total_cells_after_qc': total_cells,
            'genes_in_expression_stats': len(corpus_gene_stats),
        },
        'studies': all_reports,
    }
    report_path = gu_dir / 'preprocessing_report.json'
    with open(report_path, 'w') as f:
        json.dump(report_data, f, indent=2, default=str)

    logger.info("=" * 60)
    logger.info("PREPROCESSING SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  Studies processed:    {n_success}/{len(studies)}")
    logger.info(f"  Total QC'd cells:     {total_cells:,}")
    logger.info(f"  Genes tracked:        {len(corpus_gene_stats):,}")
    logger.info(f"  Elapsed:              {elapsed/60:.1f} min")


def run_single_study_mode(args, config: dict):
    """Process a single study (for SLURM array jobs)."""
    accession = args.single_study
    gu_dir = resolve_path(config, config['paths']['gene_universe_dir'])
    qc_dir = resolve_path(config, config['paths']['qc_h5ad_dir'])
    if args.output_dir:
        qc_dir = Path(args.output_dir)

    mapper = GeneUniverseMapper(gu_dir)

    # Find study directory
    h = config.get('harvesting', {})
    data_root = resolve_path(config, h.get('data_root', 'data/raw'))
    sources = h.get('sources', {})

    geo_path = sources.get('geo', {}).get('single_cell', {}).get('path', 'geo/single_cell/geo_datasets')
    ae_path = sources.get('arrayexpress', {}).get('single_cell', {}).get('path', 'arrayexpress/singlecell/datasets')

    study_dir = data_root / geo_path / accession
    if not study_dir.exists():
        study_dir = data_root / ae_path / accession
    if not study_dir.exists():
        logger.error(f"Study directory not found for {accession}")
        sys.exit(1)

    qc_dir.mkdir(parents=True, exist_ok=True)

    report, gene_stats = process_study(
        study_accession=accession,
        study_dir=str(study_dir),
        mapper=mapper,
        config=config,
        output_dir=str(qc_dir),
    )

    # Save per-study outputs for merge
    report_file = qc_dir / f"report_{accession}.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)

    stats_file = qc_dir / f"gene_stats_{accession}.json"
    with open(stats_file, 'w') as f:
        json.dump(gene_stats, f, indent=2)

    logger.info(f"Study {accession}: {report['status']}, "
                f"{report.get('total_cells_after_qc', 0):,} cells, "
                f"{len(gene_stats):,} genes")


def run_merge_mode(args, config: dict):
    """Merge per-study outputs from SLURM array jobs."""
    gu_dir = resolve_path(config, config['paths']['gene_universe_dir'])
    qc_dir = resolve_path(config, config['paths']['qc_h5ad_dir'])
    if args.output_dir:
        qc_dir = Path(args.output_dir)

    corpus_gene_stats = defaultdict(lambda: {'n_cells_top2048': 0, 'total_expression': 0.0})
    all_reports = []

    for stats_file in sorted(qc_dir.glob('gene_stats_*.json')):
        with open(stats_file) as f:
            gene_stats = json.load(f)
        for gid, stats in gene_stats.items():
            corpus_gene_stats[gid]['n_cells_top2048'] += stats['n_cells_top2048']
            corpus_gene_stats[gid]['total_expression'] += stats['total_expression']

    for report_file in sorted(qc_dir.glob('report_*.json')):
        with open(report_file) as f:
            all_reports.append(json.load(f))

    logger.info(f"Merged {len(all_reports)} studies, {len(corpus_gene_stats):,} genes")

    # Write expression_stats.tsv
    stats_path = gu_dir / 'expression_stats.tsv'
    with open(stats_path, 'w', newline='') as f:
        w = csv.DictWriter(f, delimiter='\t',
                           fieldnames=['ensembl_id', 'n_cells_top2048',
                                       'total_expression', 'n_studies_expressed'])
        w.writeheader()
        for gid in sorted(corpus_gene_stats.keys()):
            stats = corpus_gene_stats[gid]
            w.writerow({
                'ensembl_id': gid,
                'n_cells_top2048': stats['n_cells_top2048'],
                'total_expression': round(stats['total_expression'], 2),
                'n_studies_expressed': 0,
            })

    n_success = sum(1 for r in all_reports if r.get('status') == 'success')
    total_cells = sum(r.get('total_cells_after_qc', 0) for r in all_reports)

    logger.info(f"expression_stats.tsv: {len(corpus_gene_stats):,} genes")
    logger.info(f"Studies: {n_success}/{len(all_reports)} success, {total_cells:,} cells")

    report_data = {
        'stage': 'stage2_step2_merged',
        'timestamp': datetime.now().isoformat(),
        'summary': {
            'studies_success': n_success,
            'studies_total': len(all_reports),
            'total_cells_after_qc': total_cells,
            'genes_in_expression_stats': len(corpus_gene_stats),
        },
        'studies': all_reports,
    }
    with open(gu_dir / 'preprocessing_report.json', 'w') as f:
        json.dump(report_data, f, indent=2, default=str)


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Stage 2, Step 2: Preprocess training matrices — GeneCompass-exact Cell QC",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python preprocess_training_matrices.py                     # Full corpus
  python preprocess_training_matrices.py --single-study GSE123456  # SLURM array
  python preprocess_training_matrices.py --merge             # Merge SLURM outputs
  sbatch slurm/stage2_preprocess.slurm                       # SLURM wrapper
        """
    )
    parser.add_argument('-o', '--output-dir', default=None,
                        help='Output directory (default: from config)')
    parser.add_argument('--single-study', default=None,
                        help='Process a single study (for SLURM array)')
    parser.add_argument('--merge', action='store_true',
                        help='Merge per-study outputs from SLURM array')
    parser.add_argument('--dry-run', action='store_true',
                        help='Validate config and inputs, then exit')
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if MISSING_DEPS:
        logger.error(f"Missing required packages: {', '.join(MISSING_DEPS)}")
        logger.error("Install with: pip install numpy scipy h5py anndata scanpy pandas")
        sys.exit(1)

    try:
        config = load_config()
    except FileNotFoundError:
        logger.error("pipeline_config.yaml not found. Run from project root or set PIPELINE_ROOT.")
        sys.exit(1)

    # Validate required config
    for section in ('gene_universe', 'biomart', 'paths'):
        if section not in config:
            logger.error(f"Config missing '{section}' section")
            sys.exit(1)

    if args.dry_run:
        gu_dir = resolve_path(config, config['paths']['gene_universe_dir'])
        universe_path = gu_dir / 'gene_universe.tsv'
        resolution_path = gu_dir / 'gene_resolution.tsv'
        logger.info("DRY RUN — validating inputs:")
        logger.info(f"  gene_universe.tsv: {'EXISTS' if universe_path.exists() else 'MISSING (run Step 1)'}")
        logger.info(f"  gene_resolution.tsv: {'EXISTS' if resolution_path.exists() else 'MISSING (run Step 1)'}")
        logger.info("DRY RUN — exiting")
        return

    if args.merge:
        run_merge_mode(args, config)
    elif args.single_study:
        run_single_study_mode(args, config)
    else:
        run_corpus_preprocessing(args, config)


if __name__ == '__main__':
    main()