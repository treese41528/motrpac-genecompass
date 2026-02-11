#!/usr/bin/env python3
"""
analyze_matrices.py - FAST parallel matrix analysis

Key optimizations:
1. Multiprocessing for parallel study analysis
2. Header-only reads (don't load full matrices)
3. Early termination on first valid matrix per study
4. Streaming file counting (no full loads)

Usage:
    python analyze_matrices.py --config config.yaml --organism rattus --workers 8
    python analyze_matrices.py --config config.yaml --max-studies 100 --workers 4
"""

import os
import sys
import gzip
import json
import argparse
import logging
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import tarfile


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# --- Config integration ---
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'lib'))
from gene_utils import load_config, resolve_path

# Load config (used for defaults; CLI args still override)
try:
    _config = load_config()
except FileNotFoundError:
    _config = None




try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False
    h5py = None


def extract_tar_if_needed(study_path: Path, force: bool = False) -> bool:
    """
    Extract RAW.tar file if the RAW directory is empty.
    
    Returns True if extraction happened or RAW has files, False on error.
    """
    # Find tar file
    tar_file = None
    raw_dir = None
    
    for f in study_path.iterdir():
        if f.name.endswith('_RAW.tar'):
            tar_file = f
        elif f.name.endswith('_RAW') and f.is_dir():
            raw_dir = f
    
    if not tar_file:
        return True  # No tar file, nothing to extract
    
    if not raw_dir:
        # Create RAW directory
        raw_dir = study_path / tar_file.name.replace('.tar', '')
        raw_dir.mkdir(exist_ok=True)
    
    # Check if RAW directory has files
    try:
        existing_files = list(raw_dir.iterdir())
        if existing_files and not force:
            return True  # Already has files
    except Exception:
        pass
    
    # Extract tar
    try:
        with tarfile.open(tar_file, 'r') as tar:
            # Security: avoid path traversal
            for member in tar.getmembers():
                if member.name.startswith('/') or '..' in member.name:
                    continue
                tar.extract(member, raw_dir, filter='data')
        return True
    except Exception as e:
        return False


def deduplicate_studies(studies: List[Dict], 
                        source_paths: Dict[Tuple[str, str], Path]) -> List[Dict]:
    """
    Deduplicate studies that exist in both bulk/ and single_cell/ directories.
    
    Priority:
    1. Prefer single_cell if study has actual single-cell data (many cells)
    2. Prefer bulk if study appears to be bulk (few samples)
    3. If unclear, prefer single_cell (safer for downstream)
    """
    from collections import defaultdict
    
    # Group by accession
    by_accession = defaultdict(list)
    for study in studies:
        acc = study.get('accession')
        by_accession[acc].append(study)
    
    deduplicated = []
    duplicates_resolved = 0
    
    for acc, study_list in by_accession.items():
        if len(study_list) == 1:
            deduplicated.append(study_list[0])
            continue
        
        # Multiple entries - need to pick one
        duplicates_resolved += 1
        
        # Check which directories actually have data
        sc_study = None
        bulk_study = None
        
        for s in study_list:
            dtype = s.get('data_type')
            if dtype == 'single_cell':
                sc_study = s
            elif dtype == 'bulk':
                bulk_study = s
        
        # Decide which to keep
        if sc_study and bulk_study:
            # Check if single_cell path has more/better data
            sc_path = source_paths.get(('geo', 'single_cell'))
            bulk_path = source_paths.get(('geo', 'bulk'))
            
            sc_size = get_dir_size(sc_path / acc) if sc_path else 0
            bulk_size = get_dir_size(bulk_path / acc) if bulk_path else 0
            
            # Prefer single_cell if it has substantial data
            if sc_size > bulk_size * 0.5:  # SC has at least half the data
                chosen = sc_study
                chosen['duplicate_resolved'] = 'kept_single_cell'
            else:
                chosen = bulk_study
                chosen['duplicate_resolved'] = 'kept_bulk'
        else:
            chosen = study_list[0]
            chosen['duplicate_resolved'] = 'kept_first'
        
        deduplicated.append(chosen)
    
    if duplicates_resolved > 0:
        logger.info(f"Resolved {duplicates_resolved} duplicate studies")
    
    return deduplicated


def get_dir_size(path: Path) -> int:
    """Get total size of directory contents."""
    if not path or not path.exists():
        return 0
    try:
        total = 0
        for f in path.rglob('*'):
            if f.is_file():
                total += f.stat().st_size
        return total
    except Exception:
        return 0


def filter_non_rat_studies(results: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """
    Separate rat studies from contaminated (mouse/human) studies.
    
    Returns: (rat_studies, non_rat_studies)
    """
    rat_studies = []
    non_rat_studies = []
    
    for r in results:
        gene_id_type = r.get('gene_id_type', 'unknown')
        
        # Check for non-rat gene IDs
        if gene_id_type in ['ensembl_mouse', 'ensembl_human']:
            r['excluded_reason'] = f'Non-rat gene IDs: {gene_id_type}'
            non_rat_studies.append(r)
        else:
            rat_studies.append(r)
    
    return rat_studies, non_rat_studies

# =============================================================================
# IDENTIFIER PATTERN MATCHING
# =============================================================================

def classify_identifier(value: str) -> str:
    """Classify what type of identifier a string is."""
    if not value or not value.strip():
        return 'unknown'
    
    value = value.strip()
    
    # === GENE ID PATTERNS ===
    gene_patterns = [
        (r'^ENSRNO[GT]\d{11}$', 'ensembl_rat'),
        (r'^ENSMUS[GT]\d{11}$', 'ensembl_mouse'),
        (r'^ENS[A-Z]{0,4}[GT]\d{11}$', 'ensembl'),
        (r'^[NX][MR]_\d+', 'refseq'),
        (r'^LOC\d+$', 'ncbi_loc'),
        (r'^[A-Z][a-z0-9]{1,10}$', 'gene_symbol_mouse'),  # Gapdh, Actb
        (r'^[A-Z][A-Z0-9]{1,10}$', 'gene_symbol_human'),  # GAPDH, ACTB
    ]
    
    # === miRNA PATTERNS (NEW) ===
    mirna_patterns = [
        (r'^rno-(?:mir|miR|let)-', 'mirna_rat'),
        (r'^mmu-(?:mir|miR|let)-', 'mirna_mouse'),
        (r'^hsa-(?:mir|miR|let)-', 'mirna_human'),
        (r'^(?:mir|miR|let)-\d+', 'mirna_generic'),
    ]
    
    # === circRNA PATTERNS (NEW) ===
    circrna_patterns = [
        (r'^(?:circ|circRNA)[_-]?\w+', 'circrna'),
        (r'^chr\d+:\d+[\|\-]\d+', 'circrna_coords'),  # chr7:139461703|139469960
    ]
    
    # === lncRNA PATTERNS (NEW) ===
    lncrna_patterns = [
        (r'^(?:LINC|LOC|lncRNA)\d+', 'lncrna'),
        (r'^NONRAT\d+', 'lncrna_rat'),  # NONCODE database
    ]
    
    # === PROBE ID PATTERNS (NEW) ===
    probe_patterns = [
        (r'^\d+_(?:at|s_at|x_at|a_at)$', 'probe_affy'),
        (r'^ILMN_\d+$', 'probe_illumina'),
        (r'^A_\d+_P\d+$', 'probe_agilent'),
    ]
    
    # Cell barcode patterns
    barcode_patterns = [
        (r'^[ACGT]{16}-\d+$', 'barcode_10x'),
        (r'^[ACGT]{8,20}$', 'barcode_nucleotide'),
        (r'^[ACGT]{8,20}-\d+$', 'barcode_with_suffix'),
    ]
    
    # Sample ID patterns
    sample_patterns = [
        (r'^GSM\d{6,9}$', 'geo_sample'),
        (r'^SRR\d{6,10}$', 'sra_run'),
        (r'^SRX\d{6,10}$', 'sra_experiment'),
        (r'^SAMN\d+$', 'biosample'),
    ]
    
    # Check gene patterns first
    for pattern, id_type in gene_patterns:
        if re.match(pattern, value):
            return f'gene:{id_type}'
    
    # Check miRNA patterns (return as gene subtype)
    for pattern, id_type in mirna_patterns:
        if re.match(pattern, value):
            return f'gene:{id_type}'
    
    # Check circRNA patterns
    for pattern, id_type in circrna_patterns:
        if re.match(pattern, value):
            return f'gene:{id_type}'
    
    # Check lncRNA patterns
    for pattern, id_type in lncrna_patterns:
        if re.match(pattern, value):
            return f'gene:{id_type}'
    
    # Check probe patterns (return as gene - they map to genes)
    for pattern, id_type in probe_patterns:
        if re.match(pattern, value):
            return f'gene:{id_type}'
    
    # Check barcode patterns
    for pattern, id_type in barcode_patterns:
        if re.match(pattern, value):
            return f'cell:{id_type}'
    
    # Check sample patterns
    for pattern, id_type in sample_patterns:
        if re.match(pattern, value):
            return f'sample:{id_type}'
    
    return 'unknown'


def classify_identifier_list(ids: List[str], min_consensus: float = 0.6) -> Tuple[str, float]:
    """
    Classify a list of identifiers by majority vote.
    
    Returns:
        (type, confidence) where type is 'genes', 'cells', 'samples', or 'unknown'
    """
    if not ids:
        return 'unknown', 0.0
    
    # Classify each ID
    classifications = [classify_identifier(id_val) for id_val in ids if id_val]
    
    if not classifications:
        return 'unknown', 0.0
    
    # Count by category (gene/cell/sample)
    counts = {'gene': 0, 'cell': 0, 'sample': 0, 'unknown': 0}
    for c in classifications:
        if c.startswith('gene:'):
            counts['gene'] += 1
        elif c.startswith('cell:'):
            counts['cell'] += 1
        elif c.startswith('sample:'):
            counts['sample'] += 1
        else:
            counts['unknown'] += 1
    
    total = len(classifications)
    
    # Find majority
    for id_type in ['gene', 'cell', 'sample']:
        ratio = counts[id_type] / total
        if ratio >= min_consensus:
            return f'{id_type}s', ratio  # Return plural: genes, cells, samples
    
    return 'unknown', counts['unknown'] / total


def sample_file_ids(filepath: Path, max_ids: int = 50) -> List[str]:
    """Read first N identifiers from a file (first column)."""
    ids = []
    try:
        opener = gzip.open if str(filepath).endswith('.gz') else open
        with opener(filepath, 'rt') as f:
            for i, line in enumerate(f):
                if i >= max_ids:
                    break
                parts = line.strip().split('\t')
                if parts and parts[0]:
                    ids.append(parts[0])
    except Exception:
        pass
    return ids



# =============================================================================
# FAST MATRIX DIMENSION EXTRACTION (Header-only reads)
# =============================================================================
def detect_matrix_orientation(filepath: Path, n_rows: int, n_cols: int) -> Dict[str, Any]:
    """
    Determine matrix orientation by sampling identifiers.
    """
    parent = filepath.parent
    result = {
        'row_type': 'unknown',
        'col_type': 'unknown',
        'n_genes': None,
        'n_cells': None,
        'n_samples': None,
        'confidence': 'low',
        'reason': '',
    }
    
    # Find companion files
    row_ids_file = None  # genes/features file for MTX
    col_ids_file = None  # barcodes file for MTX
    
    for f in parent.iterdir():
        fname = f.name.lower()
        if any(x in fname for x in ['gene', 'feature']) and fname.endswith(('.tsv', '.tsv.gz', '.txt', '.txt.gz')):
            row_ids_file = f
        elif any(x in fname for x in ['barcode', 'cell']) and fname.endswith(('.tsv', '.tsv.gz', '.txt', '.txt.gz')):
            col_ids_file = f
    
    # Sample and classify row identifiers
    row_ids = []
    if row_ids_file:
        row_ids = sample_file_ids(row_ids_file, 50)
        n_row_ids = fast_count_lines(row_ids_file)
        # Verify it matches matrix dimension
        if n_row_ids and n_row_ids != n_rows and n_row_ids != n_cols:
            row_ids = []  # File doesn't match matrix
    
    # Sample and classify column identifiers  
    col_ids = []
    if col_ids_file:
        col_ids = sample_file_ids(col_ids_file, 50)
        n_col_ids = fast_count_lines(col_ids_file)
        if n_col_ids and n_col_ids != n_rows and n_col_ids != n_cols:
            col_ids = []
    
    # Classify what we found
    row_type, row_conf = classify_identifier_list(row_ids) if row_ids else ('unknown', 0)
    col_type, col_conf = classify_identifier_list(col_ids) if col_ids else ('unknown', 0)
    
    # Determine orientation based on classifications
    if row_type != 'unknown' and row_conf >= 0.6:
        result['row_type'] = row_type
        result['confidence'] = 'high' if row_conf >= 0.8 else 'medium'
        result['reason'] = f'Row IDs classified as {row_type} ({row_conf:.0%})'
        
    if col_type != 'unknown' and col_conf >= 0.6:
        result['col_type'] = col_type
        if result['confidence'] == 'low':
            result['confidence'] = 'high' if col_conf >= 0.8 else 'medium'
        result['reason'] += f', Col IDs classified as {col_type} ({col_conf:.0%})'
    
    # Map to counts
    _assign_counts(result, n_rows, n_cols)
    
    # If still unknown, try biological constraints as fallback
    if result['row_type'] == 'unknown' and result['col_type'] == 'unknown':
        result = _fallback_by_dimensions(n_rows, n_cols)
    
    return result


def _assign_counts(result: Dict, n_rows: int, n_cols: int):
    """Assign n_genes/n_cells/n_samples based on detected types."""
    if result['row_type'] == 'genes':
        result['n_genes'] = n_rows
    elif result['col_type'] == 'genes':
        result['n_genes'] = n_cols
    
    if result['row_type'] == 'cells':
        result['n_cells'] = n_rows
    elif result['col_type'] == 'cells':
        result['n_cells'] = n_cols
    
    if result['row_type'] == 'samples':
        result['n_samples'] = n_rows
    elif result['col_type'] == 'samples':
        result['n_samples'] = n_cols


def _fallback_by_dimensions(n_rows: int, n_cols: int) -> Dict[str, Any]:
    """Last resort: use biological constraints."""
    result = {
        'row_type': 'unknown',
        'col_type': 'unknown',
        'n_genes': None,
        'n_cells': None,
        'n_samples': None,
        'confidence': 'low',
        'reason': 'Fallback by dimension heuristics',
    }
    
    # Rat/mouse/human typically have 20k-35k genes
    GENE_RANGE = (15000, 40000)
    
    rows_in_gene_range = GENE_RANGE[0] <= n_rows <= GENE_RANGE[1]
    cols_in_gene_range = GENE_RANGE[0] <= n_cols <= GENE_RANGE[1]
    
    if rows_in_gene_range and not cols_in_gene_range:
        result['row_type'] = 'genes'
        result['n_genes'] = n_rows
        result['col_type'] = 'cells' if n_cols > 500 else 'samples'
        if result['col_type'] == 'cells':
            result['n_cells'] = n_cols
        else:
            result['n_samples'] = n_cols
        result['confidence'] = 'low'
        
    elif cols_in_gene_range and not rows_in_gene_range:
        result['col_type'] = 'genes'
        result['n_genes'] = n_cols
        result['row_type'] = 'cells' if n_rows > 500 else 'samples'
        if result['row_type'] == 'cells':
            result['n_cells'] = n_rows
        else:
            result['n_samples'] = n_rows
        result['confidence'] = 'low'
    
    return result


def fast_mtx_dimensions(filepath: Path) -> Optional[Dict[str, Any]]:
    """
    Read MTX header with IMPROVED orientation detection.
    
    Key fix: Use partial orientation results. If we know col_type='cells',
    infer n_genes from n_rows when rows are in plausible gene range.
    """
    try:
        opener = gzip.open if str(filepath).endswith('.gz') else open
        
        with opener(filepath, 'rt') as f:
            line = f.readline()
            while line.startswith('%'):
                line = f.readline()
            
            parts = line.strip().split()
            if len(parts) >= 3:
                n_rows, n_cols, nnz = int(parts[0]), int(parts[1]), int(parts[2])
                
                # Detect orientation
                orientation = detect_matrix_orientation(filepath, n_rows, n_cols)
                
                result = {
                    'format': 'mtx',
                    'n_rows': n_rows,
                    'n_cols': n_cols,
                    'n_nonzero': nnz,
                    'orientation': orientation,
                    'n_genes': None,
                    'n_cells': None,
                    'n_samples': None,
                }
                
                row_type = orientation.get('row_type', 'unknown')
                col_type = orientation.get('col_type', 'unknown')
                
                # Gene range for validation
                GENE_RANGE = (10000, 150000)  # Expanded for multimodal
                CELL_THRESHOLD = 500  # Below this, likely samples not cells
                
                # === IMPROVED LOGIC: Use partial orientation ===
                
                # Case 1: Both types known
                if row_type == 'genes':
                    result['n_genes'] = n_rows
                    if col_type == 'cells':
                        result['n_cells'] = n_cols
                    elif col_type == 'samples':
                        result['n_samples'] = n_cols
                    else:
                        # Infer: many columns = cells, few = samples
                        if n_cols > CELL_THRESHOLD:
                            result['n_cells'] = n_cols
                        else:
                            result['n_samples'] = n_cols
                            
                elif col_type == 'genes':
                    result['n_genes'] = n_cols
                    if row_type == 'cells':
                        result['n_cells'] = n_rows
                    elif row_type == 'samples':
                        result['n_samples'] = n_rows
                    else:
                        if n_rows > CELL_THRESHOLD:
                            result['n_cells'] = n_rows
                        else:
                            result['n_samples'] = n_rows
                
                # Case 2: Only column type known (KEY FIX)
                elif col_type == 'cells' and row_type == 'unknown':
                    result['n_cells'] = n_cols
                    # If rows in gene range, infer genes
                    if GENE_RANGE[0] <= n_rows <= GENE_RANGE[1]:
                        result['n_genes'] = n_rows
                        result['orientation']['row_type'] = 'genes'
                        result['orientation']['reason'] += ' (inferred genes from row count)'
                        
                elif col_type == 'samples' and row_type == 'unknown':
                    result['n_samples'] = n_cols
                    if GENE_RANGE[0] <= n_rows <= GENE_RANGE[1]:
                        result['n_genes'] = n_rows
                        result['orientation']['row_type'] = 'genes'
                
                # Case 3: Only row type known
                elif row_type == 'cells' and col_type == 'unknown':
                    result['n_cells'] = n_rows
                    if GENE_RANGE[0] <= n_cols <= GENE_RANGE[1]:
                        result['n_genes'] = n_cols
                        result['orientation']['col_type'] = 'genes'
                        
                elif row_type == 'samples' and col_type == 'unknown':
                    result['n_samples'] = n_rows
                    if GENE_RANGE[0] <= n_cols <= GENE_RANGE[1]:
                        result['n_genes'] = n_cols
                        result['orientation']['col_type'] = 'genes'
                
                # Case 4: Both unknown - use dimension heuristics
                else:
                    rows_in_gene_range = GENE_RANGE[0] <= n_rows <= GENE_RANGE[1]
                    cols_in_gene_range = GENE_RANGE[0] <= n_cols <= GENE_RANGE[1]
                    
                    if rows_in_gene_range and not cols_in_gene_range:
                        result['n_genes'] = n_rows
                        if n_cols > CELL_THRESHOLD:
                            result['n_cells'] = n_cols
                        else:
                            result['n_samples'] = n_cols
                    elif cols_in_gene_range and not rows_in_gene_range:
                        result['n_genes'] = n_cols
                        if n_rows > CELL_THRESHOLD:
                            result['n_cells'] = n_rows
                        else:
                            result['n_samples'] = n_rows
                    # Both in range or neither - leave as unknown
                
                total = n_rows * n_cols
                result['sparsity'] = 1 - (nnz / total) if total > 0 else 0
                
                return result
    except Exception:
        pass
    return None



# Known 10x unfiltered barcode counts
UNFILTERED_10X_BARCODES = {
    6794880,    # 10x v3 chemistry
    737280,     # 10x v2 chemistry  
    3222016,    # Some 10x versions
}

def is_unfiltered_10x(n_cells: int) -> bool:
    """Check if cell count matches known unfiltered 10x barcode spaces."""
    return n_cells in UNFILTERED_10X_BARCODES


def fast_mtx_dimensions_with_unfiltered_check(filepath: Path) -> Optional[Dict[str, Any]]:
    """
    MTX dimensions with unfiltered 10x detection.
    
    If cell count matches unfiltered barcode space, flag it and
    optionally look for filtered version.
    """
    result = fast_mtx_dimensions(filepath)
    
    if result is None:
        return None
    
    n_cells = result.get('n_cells')
    
    if n_cells and is_unfiltered_10x(n_cells):
        result['is_unfiltered_10x'] = True
        result['unfiltered_warning'] = f'Cell count {n_cells:,} matches 10x unfiltered barcode space'
        
        # Try to find filtered version in same directory
        parent = filepath.parent
        fname = filepath.name.lower()
        
        # Look for filtered alternative
        for f in parent.iterdir():
            f_lower = f.name.lower()
            if 'filtered' in f_lower and f_lower.endswith(('.mtx', '.mtx.gz')):
                filtered_result = fast_mtx_dimensions(f)
                if filtered_result and filtered_result.get('n_cells'):
                    # Return filtered version instead
                    filtered_result['replaced_unfiltered'] = True
                    filtered_result['original_unfiltered_path'] = str(filepath)
                    return filtered_result
        
        # No filtered version found - keep but flag
        result['confidence'] = 'low'
    else:
        result['is_unfiltered_10x'] = False
    
    return result


def fast_h5ad_dimensions(filepath: Path) -> Optional[Dict[str, Any]]:
    """Read H5AD dimensions with ID verification.
    
    FIXED:
    - Handle sparse X (stored as Group with data/indices/indptr)
    - Check multiple locations for gene names (feature_name, gene_name, etc.)
    """
    if not HAS_H5PY:
        return None
    try:
        with h5py.File(filepath, 'r') as f:
            # AnnData standard: obs (cells) x var (genes)
            if 'X' in f:
                # Handle sparse vs dense X
                if isinstance(f['X'], h5py.Group):
                    # Sparse matrix (CSR format): indptr has n_obs + 1 elements
                    if 'indptr' in f['X']:
                        n_obs = len(f['X']['indptr']) - 1
                    else:
                        n_obs = None
                    # Get n_var from var/_index or var/feature_name
                    n_var = None
                    if 'var' in f:
                        for key in ['_index', 'index', 'feature_name', 'gene_name', 'gene_ids']:
                            if key in f['var']:
                                n_var = len(f['var'][key])
                                break
                else:
                    # Dense matrix
                    shape = f['X'].shape
                    n_obs, n_var = shape[0], shape[1]
                
                if n_obs is None or n_var is None:
                    return None
                
                result = {
                    'format': 'h5ad',
                    'n_rows': n_obs,
                    'n_cols': n_var,
                    'n_genes': None,
                    'n_cells': None,
                    'confidence': 'low',
                }
                
                # Sample var names (should be genes)
                # Check multiple possible locations
                var_names = []
                if 'var' in f:
                    for key in ['_index', 'index', 'feature_name', 'gene_name', 'gene_ids']:
                        if key in f['var']:
                            raw = f['var'][key][:50]
                            var_names = [x.decode() if isinstance(x, bytes) else str(x) 
                                         for x in raw]
                            # If we got numeric indices, try next key
                            if var_names and not var_names[0].isdigit():
                                break
                            elif key != 'feature_name':  # Keep trying
                                var_names = []
                
                # Sample obs names (should be cells)
                obs_names = []
                if 'obs' in f:
                    for key in ['_index', 'index', 'cell_id', 'barcode']:
                        if key in f['obs']:
                            raw = f['obs'][key][:50]
                            obs_names = [x.decode() if isinstance(x, bytes) else str(x) 
                                         for x in raw]
                            if obs_names and not obs_names[0].isdigit():
                                break
                            elif key != 'cell_id':
                                obs_names = []
                
                # Classify
                var_type, var_conf = classify_identifier_list(var_names) if var_names else ('unknown', 0)
                obs_type, obs_conf = classify_identifier_list(obs_names) if obs_names else ('unknown', 0)
                
                # Assign based on classification
                if var_type == 'genes' and var_conf >= 0.6:
                    result['n_genes'] = n_var
                    result['n_cells'] = n_obs
                    result['confidence'] = 'high' if var_conf >= 0.8 else 'medium'
                elif obs_type == 'genes' and obs_conf >= 0.6:
                    # Transposed from standard
                    result['n_genes'] = n_obs
                    result['n_cells'] = n_var
                    result['confidence'] = 'medium'
                else:
                    # Trust AnnData standard as fallback
                    if 15000 <= n_var <= 40000:
                        result['n_genes'] = n_var
                        result['n_cells'] = n_obs
                        result['confidence'] = 'low'
                    elif 15000 <= n_obs <= 40000:
                        result['n_genes'] = n_obs
                        result['n_cells'] = n_var
                        result['confidence'] = 'low'
                
                return result
                
            # Alternative structure
            elif 'raw' in f and 'X' in f['raw']:
                if isinstance(f['raw']['X'], h5py.Group):
                    if 'indptr' in f['raw']['X']:
                        n_obs = len(f['raw']['X']['indptr']) - 1
                        n_var = None
                        if 'raw' in f and 'var' in f['raw']:
                            for key in ['_index', 'feature_name']:
                                if key in f['raw']['var']:
                                    n_var = len(f['raw']['var'][key])
                                    break
                        if n_var:
                            return {
                                'format': 'h5ad',
                                'n_genes': n_var,
                                'n_cells': n_obs,
                                'confidence': 'low',
                            }
                else:
                    shape = f['raw']['X'].shape
                    return {
                        'format': 'h5ad',
                        'n_genes': shape[1],
                        'n_cells': shape[0],
                        'confidence': 'low',
                    }
    except Exception:
        pass
    return None



def fast_h5_dimensions(filepath: Path) -> Optional[Dict[str, Any]]:
    """Read 10x HDF5 dimensions with ID verification."""
    if not HAS_H5PY:
        return None
    try:
        with h5py.File(filepath, 'r') as f:
            # 10x Genomics format
            if 'matrix' in f:
                result = {
                    'format': 'h5',
                    'n_genes': None,
                    'n_cells': None,
                    'confidence': 'low',
                }
                
                # Get shape - it's a dataset, not attribute
                n_genes, n_cells = None, None
                if 'shape' in f['matrix']:
                    shape = f['matrix']['shape'][:]  # Read as array
                    n_genes, n_cells = int(shape[0]), int(shape[1])
                
                # Sample feature IDs
                feature_ids = []
                if 'features' in f['matrix'] and 'id' in f['matrix']['features']:
                    raw_ids = f['matrix']['features']['id'][:50]
                    feature_ids = [x.decode() if isinstance(x, bytes) else x for x in raw_ids]
                    if n_genes is None:
                        n_genes = len(f['matrix']['features']['id'])
                
                # Sample barcodes
                barcode_ids = []
                if 'barcodes' in f['matrix']:
                    raw_barcodes = f['matrix']['barcodes'][:50]
                    barcode_ids = [x.decode() if isinstance(x, bytes) else x for x in raw_barcodes]
                    if n_cells is None:
                        n_cells = len(f['matrix']['barcodes'])
                
                if n_genes is None or n_cells is None:
                    return None
                
                result['n_rows'] = n_genes
                result['n_cols'] = n_cells
                
                # Classify
                feat_type, feat_conf = classify_identifier_list(feature_ids) if feature_ids else ('unknown', 0)
                barc_type, barc_conf = classify_identifier_list(barcode_ids) if barcode_ids else ('unknown', 0)
                
                # Assign based on classification
                if feat_type == 'genes' and feat_conf >= 0.6:
                    result['n_genes'] = n_genes
                    result['confidence'] = 'high' if feat_conf >= 0.8 else 'medium'
                elif 15000 <= n_genes <= 150000:  # Expanded for multi-modal
                    result['n_genes'] = n_genes
                    result['confidence'] = 'low'
                
                if barc_type == 'cells' and barc_conf >= 0.6:
                    result['n_cells'] = n_cells
                    if result['confidence'] == 'low':
                        result['confidence'] = 'high' if barc_conf >= 0.8 else 'medium'
                elif n_cells > 0 and result.get('n_genes'):
                    result['n_cells'] = n_cells
                
                return result
            
            # Try as AnnData
            return fast_h5ad_dimensions(filepath)
    except Exception as e:
        pass
    return None



def fast_loom_dimensions(filepath: Path) -> Optional[Dict[str, Any]]:
    """Read Loom dimensions with ID verification."""
    if not HAS_H5PY:
        return None
    try:
        with h5py.File(filepath, 'r') as f:
            if 'matrix' in f:
                shape = f['matrix'].shape
                n_rows, n_cols = shape[0], shape[1]
                
                result = {
                    'format': 'loom',
                    'n_rows': n_rows,
                    'n_cols': n_cols,
                    'n_genes': None,
                    'n_cells': None,
                    'confidence': 'low',
                }
                
                # Sample row attributes (typically genes)
                row_ids = []
                if 'row_attrs' in f:
                    for attr in ['Gene', 'Accession', 'gene_id', 'gene_name']:
                        if attr in f['row_attrs']:
                            row_ids = [x.decode() if isinstance(x, bytes) else x 
                                       for x in f['row_attrs'][attr][:50]]
                            break
                
                # Sample col attributes (typically cells)
                col_ids = []
                if 'col_attrs' in f:
                    for attr in ['CellID', 'cell_id', 'barcode', 'Barcode']:
                        if attr in f['col_attrs']:
                            col_ids = [x.decode() if isinstance(x, bytes) else x 
                                       for x in f['col_attrs'][attr][:50]]
                            break
                
                # Classify
                row_type, row_conf = classify_identifier_list(row_ids) if row_ids else ('unknown', 0)
                col_type, col_conf = classify_identifier_list(col_ids) if col_ids else ('unknown', 0)
                
                # Assign based on classification
                if row_type == 'genes' and row_conf >= 0.6:
                    result['n_genes'] = n_rows
                    result['confidence'] = 'high' if row_conf >= 0.8 else 'medium'
                elif 15000 <= n_rows <= 40000:
                    result['n_genes'] = n_rows
                    result['confidence'] = 'low'
                
                if col_type == 'cells' and col_conf >= 0.6:
                    result['n_cells'] = n_cols
                    if result['confidence'] == 'low':
                        result['confidence'] = 'high' if col_conf >= 0.8 else 'medium'
                elif n_cols > 0 and result['n_genes']:
                    result['n_cells'] = n_cols
                
                return result
    except Exception:
        pass
    return None


def fast_count_lines(filepath: Path, max_lines: int = None) -> Optional[int]:
    """Fast line counting without loading file into memory."""
    try:
        opener = gzip.open if str(filepath).endswith('.gz') else open
        count = 0
        with opener(filepath, 'rb') as f:
            for line in f:
                count += 1
                if max_lines and count >= max_lines:
                    break
        return count
    except Exception:
        return None


def fast_tsv_dimensions(filepath: Path) -> Optional[Dict[str, Any]]:
    """
    Estimate TSV/CSV/TXT dimensions with:
    - Orientation detection via ID sampling
    - Handle quoted fields
    - Broader file format support
    """
    try:
        opener = gzip.open if str(filepath).endswith('.gz') else open
        
        with opener(filepath, 'rt', errors='replace') as f:
            # Read first few lines to analyze
            lines = []
            for i, line in enumerate(f):
                lines.append(line)
                if i >= 100:
                    break
            
            if not lines:
                return None
            
            header = lines[0]
            
            # Detect separator
            if '\t' in header:
                sep = '\t'
            elif ',' in header:
                sep = ','
            elif ';' in header:
                sep = ';'
            else:
                sep = '\t'
            
            # Parse header - handle quoted fields
            header_parts = parse_delimited_line(header, sep)
            n_cols = len(header_parts)
            
            if n_cols < 2:
                return None
            
            # Sample first column values (potential gene/row IDs)
            first_col_ids = []
            for line in lines[1:]:
                parts = parse_delimited_line(line, sep)
                if parts and parts[0]:
                    first_col_ids.append(parts[0])
            
            # Count total rows
            n_rows = len(lines) - 1
            
            # Continue counting if partial file
            if len(lines) > 100:
                for line in f:
                    n_rows += 1
                    if n_rows > 10000000:
                        break
        
        if n_rows == 0:
            return None
        
        # Classify first column (row identifiers)
        row_type, row_conf = classify_identifier_list(first_col_ids)
        
        # Classify header (column identifiers) - skip first column
        col_ids = header_parts[1:] if len(header_parts) > 1 else []
        col_type, col_conf = classify_identifier_list(col_ids)
        
        result = {
            'format': 'tsv' if sep == '\t' else ('csv' if sep == ',' else 'txt'),
            'n_rows': n_rows,
            'n_cols': n_cols - 1,
            'n_genes': None,
            'n_cells': None,
            'n_samples': None,
            'row_type': row_type,
            'col_type': col_type,
            'confidence': 'low',
        }
        
        # Assign based on row classification
        if row_type == 'genes' and row_conf >= 0.6:
            result['n_genes'] = n_rows
            result['confidence'] = 'high' if row_conf >= 0.8 else 'medium'
            
            if col_type == 'cells' and col_conf >= 0.6:
                result['n_cells'] = n_cols - 1
            elif col_type == 'samples' and col_conf >= 0.6:
                result['n_samples'] = n_cols - 1
            else:
                # Heuristic: >500 columns likely cells, otherwise samples
                if n_cols - 1 > 500:
                    result['n_cells'] = n_cols - 1
                else:
                    result['n_samples'] = n_cols - 1
                
        elif row_type == 'cells' and row_conf >= 0.6:
            result['n_cells'] = n_rows
            result['confidence'] = 'medium'
            if col_type == 'genes' and col_conf >= 0.6:
                result['n_genes'] = n_cols - 1
            elif 15000 <= (n_cols - 1) <= 150000:
                result['n_genes'] = n_cols - 1
                
        elif row_type == 'samples' and row_conf >= 0.6:
            result['n_samples'] = n_rows
            result['confidence'] = 'medium'
            if col_type == 'genes' and col_conf >= 0.6:
                result['n_genes'] = n_cols - 1
            elif 15000 <= (n_cols - 1) <= 150000:
                result['n_genes'] = n_cols - 1
                
        else:
            # Fallback: use dimension heuristics
            if 15000 <= n_rows <= 150000:
                result['n_genes'] = n_rows
                if n_cols - 1 > 500:
                    result['n_cells'] = n_cols - 1
                else:
                    result['n_samples'] = n_cols - 1
                result['confidence'] = 'low'
            elif 15000 <= (n_cols - 1) <= 150000:
                result['n_genes'] = n_cols - 1
                if n_rows > 500:
                    result['n_cells'] = n_rows
                else:
                    result['n_samples'] = n_rows
                result['confidence'] = 'low'
        
        return result
        
    except Exception:
        pass
    return None


def parse_delimited_line(line: str, sep: str) -> List[str]:
    """Parse a delimited line, handling quoted fields."""
    line = line.strip()
    if not line:
        return []
    
    # Quick path: no quotes
    if '"' not in line and "'" not in line:
        return line.split(sep)
    
    # Handle quoted fields
    parts = []
    current = []
    in_quote = False
    quote_char = None
    
    i = 0
    while i < len(line):
        c = line[i]
        
        if not in_quote:
            if c == '"' or c == "'":
                in_quote = True
                quote_char = c
            elif c == sep:
                parts.append(''.join(current).strip())
                current = []
            else:
                current.append(c)
        else:
            if c == quote_char:
                # Check for escaped quote
                if i + 1 < len(line) and line[i + 1] == quote_char:
                    current.append(c)
                    i += 1
                else:
                    in_quote = False
                    quote_char = None
            else:
                current.append(c)
        i += 1
    
    # Add last field
    parts.append(''.join(current).strip())
    
    return parts

def extract_gene_ids(filepath: Path, max_ids: int = 20) -> List[str]:
    """Extract sample gene IDs from file."""
    try:
        opener = gzip.open if str(filepath).endswith('.gz') else open
        mode = 'rt' if str(filepath).endswith('.gz') else 'r'
        
        ids = []
        with opener(filepath, mode) as f:
            for i, line in enumerate(f):
                if i >= max_ids:
                    break
                parts = line.strip().split('\t')
                if parts:
                    ids.append(parts[0])
        return ids
    except Exception:
        return []


def detect_gene_id_type(gene_ids: List[str]) -> str:
    """Detect type of gene identifiers."""
    if not gene_ids:
        return 'unknown'
    
    patterns = {
        # Standard gene IDs
        'ensembl_rat': r'^ENSRNO[GT]\d{11}$',
        'ensembl_mouse': r'^ENSMUS[GT]\d{11}$',
        'ensembl_human': r'^ENSG\d{11}$',
        'refseq_mrna': r'^[NX]M_\d+',
        'refseq_ncrna': r'^[NX]R_\d+',
        'entrez': r'^\d{1,9}$',
        # miRNA patterns
        'mirna_rat': r'^rno-(?:mir|miR|let)-',
        'mirna_mouse': r'^mmu-(?:mir|miR|let)-',
        'mirna_human': r'^hsa-(?:mir|miR|let)-',
        # circRNA patterns
        'circrna': r'^(?:circ|circRNA)[_-]?\w+',
        'circrna_coords': r'^chr\d+:\d+[\|\-]\d+',
        # lncRNA patterns
        'lncrna': r'^(?:LINC|LOC|lncRNA)\d+',
        'lncrna_rat': r'^NONRAT\d+',
        # Probe IDs
        'probe_affy': r'^\d+_(?:at|s_at|x_at|a_at)$',
        'probe_illumina': r'^ILMN_\d+$',
    }
    
    for gene_id in gene_ids[:10]:
        for id_type, pattern in patterns.items():
            if re.match(pattern, gene_id):
                return id_type
    
    # Check for gene symbols last (broad pattern)
    if all(re.match(r'^[A-Za-z][A-Za-z0-9\-]{0,20}$', g) for g in gene_ids[:5] if g):
        return 'gene_symbol'
    
    return 'unknown'

# =============================================================================
# STUDY SCANNER
# =============================================================================

MATRIX_EXTENSIONS = {'.mtx', '.mtx.gz', '.h5ad', '.h5', '.hdf5', '.loom'}

# Patterns that suggest a file IS a count/expression matrix
MATRIX_PATTERNS = {
    'matrix', 'counts', 'count', 'expression', 'umi', 
    'filtered_feature_bc', 'raw_feature_bc',
    'fpkm', 'tpm', 'rpkm', 'cpm',
    'raw_count', 'gene_count', 'read_count',
    'dgematrix', 'countmatrix', 'expr',
    'gene_expression', 'transcript',
}

# Patterns to SKIP (metadata, not count matrices)
SKIP_PATTERNS = {
    # GEO/SRA metadata files (THE MAIN FIX)
    '_samples.tsv', '_sra_runs.tsv', '_gsm_files.tsv',
    'sample', 'samples', 'sample_info', 'sampleinfo',
    'metadata', 'meta_data', 'phenotype', 'phenodata',
    'clinical', 'annotation', 'annotations',
    'sra_run', 'sra_runs', 'srr_', 
    'manifest', 'filelist', 'readme', 'license',
    'family.soft', '.soft.gz',
    'barcode', 'barcodes',  # Companion files, not matrices
    'genes.tsv', 'features.tsv',  # Companion files (be specific)
    'clusters', 'cluster', 'umap', 'tsne', 'pca',
    'neighbors', 'obsm', 'varm', 'obsp',
    'coldata', 'rowdata', 'col_data', 'row_data',
    '.md5', '.md5sum',
    # ChIP-seq/ATAC-seq peaks (not expression)
    '_peaks.txt', '_peaks.bed', 'narrowpeak', 'broadpeak',
}


def find_matrix_files(study_path: Path, max_files: int = 50) -> List[Path]:
    """
    Find potential matrix files in a study directory.
    
    Returns files sorted by preference:
    1. Filtered matrices first
    2. Known matrix patterns
    3. Generic compressed files last
    """
    matrices = []
    
    try:
        for root, dirs, files in os.walk(study_path):
            # Skip hidden directories
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            
            for fname in files:
                fpath = Path(root) / fname
                fname_lower = fname.lower()
                
                # Skip known non-matrix files (but not h5ad/h5/mtx which are always matrices)
                is_matrix_format = any(fname_lower.endswith(ext) for ext in MATRIX_EXTENSIONS)
                if not is_matrix_format and any(skip in fname_lower for skip in SKIP_PATTERNS):
                    continue
                
                # Skip very small files (< 1KB) - likely not data
                try:
                    if fpath.stat().st_size < 1024:
                        continue
                except:
                    continue
                
                # Check by extension first
                is_matrix_ext = False
                for ext in MATRIX_EXTENSIONS:
                    if fname_lower.endswith(ext):
                        matrices.append(fpath)
                        is_matrix_ext = True
                        break
                
                if is_matrix_ext:
                    continue
                
                # Check TSV/CSV/TXT files
                if fname_lower.endswith(('.tsv', '.tsv.gz', '.csv', '.csv.gz', '.txt', '.txt.gz')):
                    # Include if has matrix pattern in name
                    if any(p in fname_lower for p in MATRIX_PATTERNS):
                        matrices.append(fpath)
                    # Include generic compressed files that might be matrices
                    elif fname_lower.endswith(('.txt.gz', '.tsv.gz', '.csv.gz')):
                        # Exclude obvious non-matrix files
                        if not any(x in fname_lower for x in ['readme', 'license', 'log', 'err']):
                            matrices.append(fpath)
                    # Include larger uncompressed files (>10KB) as candidates
                    elif fname_lower.endswith(('.txt', '.tsv', '.csv')):
                        try:
                            if fpath.stat().st_size > 10240:  # > 10KB
                                matrices.append(fpath)
                        except:
                            pass
                
                if len(matrices) >= max_files * 2:  # Collect more, then sort/filter
                    break
            
            if len(matrices) >= max_files * 2:
                break
                
    except Exception:
        pass
    
    # Sort by preference
    def sort_key(p: Path) -> tuple:
        name = p.name.lower()
        
        # HIGHEST PRIORITY: Skip metadata files entirely (shouldn't reach here, but safety)
        if any(skip in name for skip in ['_samples.tsv', '_sra_runs.tsv', '_gsm_files']):
            return (99, 99, 99, name)  # Sort to end
        
        # Prefer filtered over raw/unfiltered
        if 'filtered' in name and 'raw' not in name:
            filter_score = 0
        elif 'raw' in name or 'unfiltered' in name:
            filter_score = 2
        else:
            filter_score = 1
        
        # Prefer files with explicit matrix patterns
        high_priority = ['count', 'expression', 'fpkm', 'tpm', 'rpkm', 'umi', 'matrix']
        if any(pat in name for pat in high_priority):
            pattern_score = 0
        elif any(pat in name for pat in MATRIX_PATTERNS):
            pattern_score = 1
        else:
            pattern_score = 2
        
        # Prefer certain formats
        if name.endswith('.h5') or name.endswith('.h5ad'):
            format_score = 0
        elif name.endswith(('.mtx', '.mtx.gz')):
            format_score = 1
        elif name.endswith('.gz'):  # Compressed data files
            format_score = 2
        else:
            format_score = 3
        
        return (filter_score, pattern_score, format_score, name)
    
    matrices.sort(key=sort_key)
    
    return matrices[:max_files]

def get_data_modality(id_type: str) -> str:
    """Determine the data modality from gene ID type."""
    # Handle both raw type and prefixed type (e.g., 'gene:mirna_rat' or 'mirna_rat')
    id_type_clean = id_type.replace('gene:', '').replace('cell:', '').replace('sample:', '')
    
    mirna_types = {'mirna_rat', 'mirna_mouse', 'mirna_human', 'mirna_generic'}
    circrna_types = {'circrna', 'circrna_coords'}
    lncrna_types = {'lncrna', 'lncrna_rat'}
    probe_types = {'probe_affy', 'probe_illumina', 'probe_agilent'}
    mrna_types = {'ensembl_rat', 'ensembl_mouse', 'ensembl_human', 'ensembl',
                  'refseq', 'refseq_mrna', 'refseq_ncrna', 'ncbi_loc', 
                  'gene_symbol_mouse', 'gene_symbol_human', 'gene_symbol', 'entrez'}
    
    if id_type_clean in mirna_types:
        return 'mirna'
    elif id_type_clean in circrna_types:
        return 'circrna'
    elif id_type_clean in lncrna_types:
        return 'lncrna'
    elif id_type_clean in probe_types:
        return 'microarray'
    elif id_type_clean in mrna_types:
        return 'mrna'
    return 'unknown'

def analyze_study(args: Tuple[str, Path, bool]) -> Dict[str, Any]:
    """
    Analyze a single study with all fixes:
    - Partial orientation inference
    - Unfiltered 10x detection
    - Better confidence handling
    - Data modality detection
    """
    if len(args) == 3:
        accession, study_path, extract_tar = args
    else:
        accession, study_path = args
        extract_tar = False
    
    result = {
        'accession': accession,
        'n_genes': None,
        'n_cells': None,
        'n_samples': None,
        'formats': {},
        'gene_ids_sample': [],
        'gene_id_type': 'unknown',
        'data_modality': 'unknown',  # NEW
        'is_multimodal': False,
        'is_unfiltered_10x': False,
        'feature_types': {},
        'confidence': 'none',
        'data_type': None,
        'tar_extracted': False,
        'error': None,
    }
    
    if not study_path.exists():
        result['error'] = 'path_not_found'
        return result
    
    try:
        if extract_tar:
            result['tar_extracted'] = extract_tar_if_needed(study_path)
        
        matrices = find_matrix_files(study_path)
        
        confidence_order = {'high': 3, 'medium': 2, 'low': 1, 'none': 0}
        
        best_sc = {'n_genes': None, 'n_cells': None, 'confidence': 'none', 'unfiltered': False}
        best_bulk = {'n_genes': None, 'n_samples': None, 'confidence': 'none'}
        
        for mpath in matrices:
            fname = mpath.name.lower()
            stats = None
            
            if fname.endswith(('.mtx', '.mtx.gz')):
                stats = fast_mtx_dimensions_with_unfiltered_check(mpath)
                
                for f in mpath.parent.iterdir():
                    f_lower = f.name.lower()
                    if ('gene' in f_lower or 'feature' in f_lower):
                        if f_lower.endswith(('.tsv', '.tsv.gz', '.txt', '.txt.gz')):
                            if not result['gene_ids_sample']:
                                result['gene_ids_sample'] = extract_gene_ids(f, 20)
                            break
            
            elif fname.endswith('.h5ad'):
                stats = fast_h5ad_dimensions(mpath)
            
            elif fname.endswith(('.h5', '.hdf5')):
                stats = fast_h5_dimensions(mpath)
                if stats and stats.get('is_multimodal'):
                    result['is_multimodal'] = True
                    result['feature_types'] = stats.get('feature_types', {})
            
            elif fname.endswith('.loom'):
                stats = fast_loom_dimensions(mpath)
            
            elif fname.endswith(('.tsv', '.tsv.gz', '.csv', '.csv.gz', '.txt', '.txt.gz')):
                stats = fast_tsv_dimensions(mpath)
                if stats and stats.get('row_type') == 'genes' and not result['gene_ids_sample']:
                    result['gene_ids_sample'] = extract_gene_ids(mpath, 20)
            
            if stats:
                fmt = stats.get('format', 'unknown')
                result['formats'][fmt] = result['formats'].get(fmt, 0) + 1
                
                stat_conf = stats.get('confidence', 'none')
                stat_conf_val = confidence_order.get(stat_conf, 0)
                
                n_genes = stats.get('n_genes')
                n_cells = stats.get('n_cells')
                n_samples = stats.get('n_samples')
                is_unfiltered = stats.get('is_unfiltered_10x', False)
                
                if is_unfiltered:
                    result['is_unfiltered_10x'] = True
                
                if n_cells is not None and n_cells > 0:
                    sc_conf_val = confidence_order.get(best_sc['confidence'], 0)
                    
                    if best_sc['unfiltered'] and not is_unfiltered:
                        best_sc = {
                            'n_genes': n_genes,
                            'n_cells': n_cells,
                            'confidence': stat_conf,
                            'unfiltered': is_unfiltered
                        }
                    elif not best_sc['unfiltered'] and is_unfiltered:
                        if best_sc['n_cells'] is None:
                            best_sc = {
                                'n_genes': n_genes,
                                'n_cells': n_cells,
                                'confidence': stat_conf,
                                'unfiltered': is_unfiltered
                            }
                    elif stat_conf_val > sc_conf_val or (
                        stat_conf_val == sc_conf_val and 
                        n_cells > (best_sc['n_cells'] or 0)
                    ):
                        best_sc = {
                            'n_genes': n_genes,
                            'n_cells': n_cells,
                            'confidence': stat_conf,
                            'unfiltered': is_unfiltered
                        }
                
                elif n_samples is not None and n_samples > 0:
                    bulk_conf_val = confidence_order.get(best_bulk['confidence'], 0)
                    if stat_conf_val > bulk_conf_val or (
                        stat_conf_val == bulk_conf_val and 
                        n_samples > (best_bulk['n_samples'] or 0)
                    ):
                        best_bulk = {
                            'n_genes': n_genes,
                            'n_samples': n_samples,
                            'confidence': stat_conf
                        }
        
        sc_conf_val = confidence_order.get(best_sc['confidence'], 0)
        bulk_conf_val = confidence_order.get(best_bulk['confidence'], 0)
        
        if sc_conf_val >= bulk_conf_val and best_sc['n_cells'] and not best_sc['unfiltered']:
            result['n_genes'] = best_sc['n_genes']
            result['n_cells'] = best_sc['n_cells']
            result['confidence'] = best_sc['confidence']
            result['data_type'] = 'single_cell'
        elif best_bulk['n_samples']:
            result['n_genes'] = best_bulk['n_genes']
            result['n_samples'] = best_bulk['n_samples']
            result['confidence'] = best_bulk['confidence']
            result['data_type'] = 'bulk'
        elif best_sc['n_cells']:
            result['n_genes'] = best_sc['n_genes']
            result['n_cells'] = best_sc['n_cells']
            result['confidence'] = 'low'
            result['data_type'] = 'single_cell'
            result['unfiltered_warning'] = 'Only unfiltered matrix available'
        else:
            if best_sc.get('n_genes'):
                result['n_genes'] = best_sc['n_genes']
                result['confidence'] = best_sc['confidence']
            elif best_bulk.get('n_genes'):
                result['n_genes'] = best_bulk['n_genes']
                result['confidence'] = best_bulk['confidence']
        
        # Detect gene ID type AND modality (NEW)
        if result['gene_ids_sample']:
            result['gene_id_type'] = detect_gene_id_type(result['gene_ids_sample'])
            result['data_modality'] = get_data_modality(result['gene_id_type'])
        
    except Exception as e:
        result['error'] = str(e)
    
    return result


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def main():
    """Main function with deduplication and filtering."""
    parser = argparse.ArgumentParser(description='Fast parallel matrix analysis')
    parser.add_argument('--organism', '-o', help='Filter by organism')
    parser.add_argument('--data-type', '-t', help='Filter by data type')
    parser.add_argument('--max-studies', '-n', type=int, help='Max studies')
    parser.add_argument('--workers', '-w', type=int, default=8, help='Workers')
    parser.add_argument('--resume', action='store_true', help='Resume')
    parser.add_argument('--extract-tar', action='store_true', help='Extract tars')
    parser.add_argument('--deduplicate', action='store_true', default=True,
                        help='Deduplicate studies in multiple directories')
    parser.add_argument('--filter-rat', action='store_true', default=True,
                        help='Filter out non-rat studies')
    
    args = parser.parse_args()
    
    config = load_config()
    h = config.get('harvesting', {})
    catalog_dir = resolve_path(config, h.get('catalog_dir', 'data/catalog'))
    catalog_path = catalog_dir / 'master_catalog.json'
    output_path = catalog_dir / 'matrix_analysis.json'
    # Build source paths
    data_root = resolve_path(config, h.get('data_root', 'data/raw'))
    sources = h.get('sources', {})
    source_paths = {}
    for source_name, source_config in sources.items():
        for dtype in ['single_cell', 'bulk']:
            dtype_config = source_config.get(dtype, {})
            if dtype_config.get('enabled', True):
                rel_path = dtype_config.get('path', '')
                if rel_path:
                    full_path = data_root / rel_path
                    source_paths[(source_name, dtype)] = full_path
    
    logger.info(f"Catalog: {catalog_path}")
    logger.info(f"Output: {output_path}")
    logger.info(f"Workers: {args.workers}")
    logger.info(f"Extract tar: {args.extract_tar}")
    logger.info(f"Deduplicate: {args.deduplicate}")
    logger.info(f"Filter rat: {args.filter_rat}")
    
    # Load catalog
    logger.info("Loading catalog...")
    with open(catalog_path, 'r') as f:
        catalog = json.load(f)
    
    studies = catalog.get('studies', [])
    logger.info(f"Loaded {len(studies)} studies")
    
    if args.deduplicate:
        studies = deduplicate_studies(studies, source_paths)
        logger.info(f"After deduplication: {len(studies)} studies")
    
    # Filter
    if args.organism:
        org_lower = args.organism.lower()
        studies = [s for s in studies if any(org_lower in o.lower() for o in s.get('organism', []))]
    
    if args.data_type:
        studies = [s for s in studies if s.get('data_type') == args.data_type]
    
    logger.info(f"After filtering: {len(studies)} studies")
    
    if args.max_studies:
        studies = studies[:args.max_studies]
        logger.info(f"Limited to {len(studies)} studies")
    
    # Resume support
    processed_accessions = set()
    existing_results = []
    if args.resume and output_path.exists():
        with open(output_path, 'r') as f:
            existing = json.load(f)
            existing_results = existing.get('study_stats', [])
            processed_accessions = {r['accession'] for r in existing_results}
            logger.info(f"Resuming: {len(processed_accessions)} already processed")
    
    # Prepare tasks
    tasks = []
    for study in studies:
        accession = study.get('accession')
        if accession in processed_accessions:
            continue
        
        source = study.get('source', 'geo')
        dtype = study.get('data_type', 'bulk')
        base_path = source_paths.get((source, dtype))
        
        if base_path:
            study_path = base_path / accession
            tasks.append((accession, study_path, args.extract_tar))
    
    logger.info(f"Tasks to process: {len(tasks)}")
    
    if not tasks:
        logger.info("No new tasks. Exiting.")
        return
    
    # Parallel processing
    results = list(existing_results)
    seen_accessions = {r['accession'] for r in results}
    completed = 0
    start_time = datetime.now()
    
    logger.info(f"Starting parallel analysis with {args.workers} workers...")
    
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(analyze_study, task): task for task in tasks}
        
        for future in as_completed(futures):
            try:
                result = future.result(timeout=300)
                
                if result['accession'] not in seen_accessions:
                    results.append(result)
                    seen_accessions.add(result['accession'])
                
                completed += 1
                
                if completed % 50 == 0:
                    elapsed = (datetime.now() - start_time).total_seconds()
                    rate = completed / elapsed if elapsed > 0 else 0
                    eta = (len(tasks) - completed) / rate / 60 if rate > 0 else 0
                    logger.info(f"Progress: [{completed}/{len(tasks)}] ({rate:.1f}/sec, ETA: {eta:.1f} min)")
                    
                    save_results(results, output_path, studies)
                    
            except Exception as e:
                task = futures[future]
                logger.warning(f"Error processing {task[0]}: {e}")
    
    # Filter non-rat studies (NEW)
    excluded_results = None
    if args.filter_rat:
        results, excluded_results = filter_non_rat_studies(results)
        logger.info(f"Rat studies: {len(results)}, Non-rat excluded: {len(excluded_results)}")
    
    # Final save
    save_results(results, output_path, studies, excluded_results)
    
    elapsed = (datetime.now() - start_time).total_seconds()
    logger.info(f"Completed {completed} studies in {elapsed:.1f}s ({completed/elapsed:.1f}/sec)")


def save_results(results: List[Dict], output_path: Path, all_studies: List[Dict],
                 excluded_results: List[Dict] = None):
    """Save results with aggregated statistics."""
    
    aggregate = {
        'total_cells': 0,
        'total_samples': 0,
        'max_genes': 0,
        'studies_analyzed': len(results),
        'studies_with_matrices': 0,
        'formats': {},
        'gene_id_types': {},
        'by_organism': {},
        'by_data_type': {},
        'by_modality': {},  # NEW
    }
    
    study_meta = {s['accession']: s for s in all_studies}
    
    for r in results:
        n_genes = r.get('n_genes') or 0
        n_cells = r.get('n_cells') or 0
        n_samples = r.get('n_samples') or 0
        
        if n_genes > 0 or n_cells > 0 or n_samples > 0:
            aggregate['studies_with_matrices'] += 1
        
        aggregate['total_cells'] += n_cells
        aggregate['total_samples'] += n_samples
        aggregate['max_genes'] = max(aggregate['max_genes'], n_genes)
        
        for fmt, count in r.get('formats', {}).items():
            aggregate['formats'][fmt] = aggregate['formats'].get(fmt, 0) + count
        
        git = r.get('gene_id_type', 'unknown')
        if git and git != 'unknown':
            aggregate['gene_id_types'][git] = aggregate['gene_id_types'].get(git, 0) + 1
        
        # By organism
        meta = study_meta.get(r['accession'], {})
        for org in meta.get('organism', []):
            if org not in aggregate['by_organism']:
                aggregate['by_organism'][org] = {'studies': 0, 'cells': 0, 'samples': 0, 'max_genes': 0}
            aggregate['by_organism'][org]['studies'] += 1
            aggregate['by_organism'][org]['cells'] += n_cells
            aggregate['by_organism'][org]['samples'] += n_samples
            aggregate['by_organism'][org]['max_genes'] = max(
                aggregate['by_organism'][org]['max_genes'], n_genes
            )
        
        # By data type
        dtype = meta.get('data_type', 'unknown')
        if dtype not in aggregate['by_data_type']:
            aggregate['by_data_type'][dtype] = {'studies': 0, 'cells': 0, 'samples': 0, 'max_genes': 0}
        aggregate['by_data_type'][dtype]['studies'] += 1
        aggregate['by_data_type'][dtype]['cells'] += n_cells
        aggregate['by_data_type'][dtype]['samples'] += n_samples
        aggregate['by_data_type'][dtype]['max_genes'] = max(
            aggregate['by_data_type'][dtype]['max_genes'], n_genes
        )
        
        # By modality (NEW)
        modality = r.get('data_modality', 'unknown')
        if modality not in aggregate['by_modality']:
            aggregate['by_modality'][modality] = {'studies': 0, 'cells': 0, 'samples': 0}
        aggregate['by_modality'][modality]['studies'] += 1
        aggregate['by_modality'][modality]['cells'] += n_cells
        aggregate['by_modality'][modality]['samples'] += n_samples
    
    output = {
        'generated_at': datetime.now().isoformat(),
        'aggregate': aggregate,
        'studies_with_matrices': aggregate['studies_with_matrices'],
        'study_stats': results,
    }
    
    # Add excluded non-rat studies (NEW)
    if excluded_results:
        output['excluded_non_rat'] = excluded_results
        output['excluded_count'] = len(excluded_results)
    
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    logger.info(f"Saved {len(results)} results to {output_path}")


if __name__ == '__main__':
    main()