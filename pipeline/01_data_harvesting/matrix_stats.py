#!/usr/bin/env python3
"""
matrix_stats.py - Extract statistics from gene expression matrices

Supports multiple formats:
- MTX (Market Matrix) - scipy sparse format
- H5AD (AnnData) - scanpy format  
- H5/HDF5 - 10x Genomics and general HDF5
- TSV/CSV - tabular count matrices
- Loom - loompy format

Usage:
    # As module
    from matrix_stats import extract_matrix_stats
    stats = extract_matrix_stats(filepath)
    
    # Standalone - analyze a single file
    python matrix_stats.py /path/to/matrix.mtx.gz
    
    # Standalone - scan a study directory
    python matrix_stats.py /path/to/study/dir --scan
"""

import os
import sys
import gzip
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Optional imports - gracefully handle missing packages
AVAILABLE_PARSERS = set()

try:
    import numpy as np
    AVAILABLE_PARSERS.add('numpy')
except ImportError:
    np = None

try:
    from scipy import io as scipy_io
    from scipy import sparse
    AVAILABLE_PARSERS.add('scipy')
except ImportError:
    scipy_io = None
    sparse = None

try:
    import h5py
    AVAILABLE_PARSERS.add('h5py')
except ImportError:
    h5py = None

try:
    import anndata
    AVAILABLE_PARSERS.add('anndata')
except ImportError:
    anndata = None

try:
    import pandas as pd
    AVAILABLE_PARSERS.add('pandas')
except ImportError:
    pd = None

try:
    import loompy
    AVAILABLE_PARSERS.add('loompy')
except ImportError:
    loompy = None


# =============================================================================
# MATRIX FORMAT DETECTION
# =============================================================================

MATRIX_EXTENSIONS = {
    '.mtx': 'mtx',
    '.mtx.gz': 'mtx',
    '.h5ad': 'h5ad',
    '.h5': 'h5',
    '.hdf5': 'h5',
    '.loom': 'loom',
    '.tsv': 'tsv',
    '.csv': 'csv',
    '.txt': 'tsv',  # Often tab-separated
    '.tsv.gz': 'tsv',
    '.csv.gz': 'csv',
    '.txt.gz': 'tsv',
}

# Patterns that suggest a file is a count/expression matrix
MATRIX_PATTERNS = [
    'matrix', 'counts', 'expression', 'umi', 'raw_count',
    'filtered_feature_bc_matrix', 'raw_feature_bc_matrix',
    'gene_expression', 'dgematrix', 'countmatrix'
]

# Patterns to skip (not count matrices)
SKIP_PATTERNS = [
    'metadata', 'sample', 'clinical', 'phenotype', 'annotation',
    'barcodes', 'genes', 'features', 'cells', 'obs', 'var',
    'clusters', 'umap', 'tsne', 'pca', 'neighbors'
]


def detect_matrix_format(filepath: Path) -> Optional[str]:
    """Detect the format of a potential matrix file."""
    name = filepath.name.lower()
    
    # Check if it's likely a matrix file
    name_lower = name.lower()
    if any(skip in name_lower for skip in SKIP_PATTERNS):
        return None
    
    # Check extensions (handle double extensions like .mtx.gz)
    suffixes = filepath.suffixes
    if len(suffixes) >= 2:
        ext = ''.join(suffixes[-2:])
        if ext in MATRIX_EXTENSIONS:
            return MATRIX_EXTENSIONS[ext]
    
    if suffixes:
        ext = suffixes[-1]
        if ext in MATRIX_EXTENSIONS:
            return MATRIX_EXTENSIONS[ext]
    
    return None


def is_likely_matrix(filepath: Path) -> bool:
    """Check if a file is likely a count matrix based on name patterns."""
    name_lower = filepath.name.lower()
    
    # Skip metadata files
    if any(skip in name_lower for skip in SKIP_PATTERNS):
        return False
    
    # Check for matrix patterns
    if any(pat in name_lower for pat in MATRIX_PATTERNS):
        return True
    
    # Check extension
    return detect_matrix_format(filepath) is not None


# =============================================================================
# FORMAT-SPECIFIC PARSERS
# =============================================================================

def parse_mtx(filepath: Path) -> Dict[str, Any]:
    """Parse Market Matrix format (.mtx, .mtx.gz)."""
    if scipy_io is None:
        return {'error': 'scipy not installed'}
    
    stats = {
        'format': 'mtx',
        'n_genes': None,
        'n_cells': None,
        'n_nonzero': None,
        'sparsity': None,
    }
    
    try:
        # Read matrix
        mat = scipy_io.mmread(str(filepath))
        
        # MTX can be genes x cells or cells x genes
        # Convention: rows = genes, cols = cells (but check for associated files)
        n_rows, n_cols = mat.shape
        
        # Try to determine orientation from associated files
        parent = filepath.parent
        genes_file = None
        barcodes_file = None
        
        for f in parent.iterdir():
            fname = f.name.lower()
            if 'gene' in fname or 'feature' in fname:
                genes_file = f
            elif 'barcode' in fname or 'cell' in fname:
                barcodes_file = f
        
        # Count lines in associated files to determine orientation
        n_genes_from_file = None
        n_cells_from_file = None
        
        if genes_file and genes_file.exists():
            n_genes_from_file = count_lines(genes_file)
        if barcodes_file and barcodes_file.exists():
            n_cells_from_file = count_lines(barcodes_file)
        
        # Determine orientation
        if n_genes_from_file and n_cells_from_file:
            if n_genes_from_file == n_rows:
                stats['n_genes'] = n_rows
                stats['n_cells'] = n_cols
            else:
                stats['n_genes'] = n_cols
                stats['n_cells'] = n_rows
        else:
            # Heuristic: typically more genes than cells
            if n_rows > n_cols:
                stats['n_genes'] = n_rows
                stats['n_cells'] = n_cols
            else:
                stats['n_genes'] = n_cols
                stats['n_cells'] = n_rows
        
        # Sparsity stats
        if sparse.issparse(mat):
            stats['n_nonzero'] = mat.nnz
            total_elements = mat.shape[0] * mat.shape[1]
            stats['sparsity'] = 1 - (mat.nnz / total_elements) if total_elements > 0 else 0
        
        # Get gene names if available
        if genes_file:
            stats['gene_ids'] = read_id_file(genes_file, max_lines=100)
        
    except Exception as e:
        stats['error'] = str(e)
        logger.debug(f"MTX parse error: {e}")
    
    return stats


def parse_h5ad(filepath: Path) -> Dict[str, Any]:
    """Parse AnnData H5AD format."""
    if anndata is None:
        return {'error': 'anndata not installed'}
    
    stats = {
        'format': 'h5ad',
        'n_genes': None,
        'n_cells': None,
        'n_nonzero': None,
        'sparsity': None,
        'layers': [],
        'obs_columns': [],
        'var_columns': [],
    }
    
    try:
        # Read in backed mode for efficiency
        adata = anndata.read_h5ad(filepath, backed='r')
        
        stats['n_cells'] = adata.n_obs
        stats['n_genes'] = adata.n_vars
        
        # Get layer names
        stats['layers'] = list(adata.layers.keys()) if adata.layers else []
        
        # Get metadata columns
        stats['obs_columns'] = list(adata.obs.columns)[:20]  # Sample metadata
        stats['var_columns'] = list(adata.var.columns)[:20]  # Gene metadata
        
        # Get sample gene IDs
        if hasattr(adata, 'var_names'):
            stats['gene_ids'] = list(adata.var_names[:100])
        
        # Try to get sparsity from X matrix
        X = adata.X
        if sparse is not None and sparse.issparse(X):
            stats['n_nonzero'] = X.nnz
            total = X.shape[0] * X.shape[1]
            stats['sparsity'] = 1 - (X.nnz / total) if total > 0 else 0
        
        adata.file.close()
        
    except Exception as e:
        stats['error'] = str(e)
        logger.debug(f"H5AD parse error: {e}")
    
    return stats


def parse_h5(filepath: Path) -> Dict[str, Any]:
    """Parse HDF5 format (10x Genomics and general)."""
    if h5py is None:
        return {'error': 'h5py not installed'}
    
    stats = {
        'format': 'h5',
        'n_genes': None,
        'n_cells': None,
        'h5_structure': {},
    }
    
    try:
        with h5py.File(filepath, 'r') as f:
            # Map structure
            stats['h5_structure'] = map_h5_structure(f)
            
            # Try 10x Genomics format
            if 'matrix' in f:
                grp = f['matrix']
                if 'shape' in grp:
                    shape = grp['shape'][:]
                    stats['n_genes'] = int(shape[0])
                    stats['n_cells'] = int(shape[1])
                if 'features' in grp:
                    feat = grp['features']
                    if 'id' in feat:
                        stats['gene_ids'] = [x.decode() if isinstance(x, bytes) else x 
                                             for x in feat['id'][:100]]
                    if 'name' in feat:
                        stats['gene_names'] = [x.decode() if isinstance(x, bytes) else x 
                                               for x in feat['name'][:100]]
            
            # Try alternative structures
            elif 'X' in f:  # AnnData-like
                X = f['X']
                if hasattr(X, 'shape'):
                    stats['n_cells'] = X.shape[0]
                    stats['n_genes'] = X.shape[1]
            
            # Try expression/counts groups
            for key in ['expression', 'counts', 'data', 'raw']:
                if key in f:
                    grp = f[key]
                    if hasattr(grp, 'shape'):
                        stats['n_cells'] = grp.shape[0]
                        stats['n_genes'] = grp.shape[1]
                        break
                        
    except Exception as e:
        stats['error'] = str(e)
        logger.debug(f"H5 parse error: {e}")
    
    return stats


def parse_loom(filepath: Path) -> Dict[str, Any]:
    """Parse Loom format."""
    if loompy is None:
        return {'error': 'loompy not installed'}
    
    stats = {
        'format': 'loom',
        'n_genes': None,
        'n_cells': None,
        'row_attrs': [],
        'col_attrs': [],
    }
    
    try:
        with loompy.connect(filepath, mode='r') as ds:
            stats['n_genes'] = ds.shape[0]
            stats['n_cells'] = ds.shape[1]
            stats['row_attrs'] = list(ds.ra.keys())[:20]
            stats['col_attrs'] = list(ds.ca.keys())[:20]
            
            # Get gene IDs
            if 'Gene' in ds.ra:
                stats['gene_ids'] = list(ds.ra['Gene'][:100])
            elif 'Accession' in ds.ra:
                stats['gene_ids'] = list(ds.ra['Accession'][:100])
                
    except Exception as e:
        stats['error'] = str(e)
        logger.debug(f"Loom parse error: {e}")
    
    return stats


def parse_tabular(filepath: Path, sep: str = '\t') -> Dict[str, Any]:
    """Parse tabular format (TSV/CSV) - reads only header."""
    stats = {
        'format': 'tsv' if sep == '\t' else 'csv',
        'n_genes': None,
        'n_samples': None,
        'has_header': True,
    }
    
    try:
        opener = gzip.open if str(filepath).endswith('.gz') else open
        mode = 'rt' if str(filepath).endswith('.gz') else 'r'
        
        with opener(filepath, mode, errors='replace') as f:
            # Read first few lines to analyze structure
            lines = []
            for i, line in enumerate(f):
                lines.append(line.strip())
                if i >= 10:
                    break
            
            if not lines:
                stats['error'] = 'Empty file'
                return stats
            
            # Detect separator if not specified
            first_line = lines[0]
            if sep == '\t' and '\t' not in first_line and ',' in first_line:
                sep = ','
                stats['format'] = 'csv'
            
            # Parse header
            header = first_line.split(sep)
            n_cols = len(header)
            
            # Check if first column is gene IDs
            first_col_values = [line.split(sep)[0] if sep in line else line for line in lines[1:]]
            looks_like_genes = any(is_gene_id(v) for v in first_col_values[:5])
            
            if looks_like_genes:
                # Rows are genes, columns are samples
                stats['n_samples'] = n_cols - 1  # Subtract gene ID column
                stats['gene_ids'] = first_col_values[:100]
                
                # Count total rows (genes) by continuing to read
                n_rows = len(lines) - 1  # Subtract header
                for line in f:
                    n_rows += 1
                    if n_rows > 100000:  # Safety limit
                        break
                stats['n_genes'] = n_rows
            else:
                # Might be transposed or metadata
                stats['n_columns'] = n_cols
                stats['header'] = header[:20]
                
    except Exception as e:
        stats['error'] = str(e)
        logger.debug(f"Tabular parse error: {e}")
    
    return stats


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def count_lines(filepath: Path) -> int:
    """Count lines in a file efficiently."""
    opener = gzip.open if str(filepath).endswith('.gz') else open
    mode = 'rt' if str(filepath).endswith('.gz') else 'r'
    
    count = 0
    try:
        with opener(filepath, mode, errors='replace') as f:
            for _ in f:
                count += 1
    except:
        pass
    return count


def read_id_file(filepath: Path, max_lines: int = 100) -> List[str]:
    """Read gene/cell IDs from a file."""
    ids = []
    opener = gzip.open if str(filepath).endswith('.gz') else open
    mode = 'rt' if str(filepath).endswith('.gz') else 'r'
    
    try:
        with opener(filepath, mode, errors='replace') as f:
            for i, line in enumerate(f):
                if i >= max_lines:
                    break
                parts = line.strip().split('\t')
                ids.append(parts[0] if parts else '')
    except:
        pass
    return ids


def map_h5_structure(h5_obj, prefix: str = '', max_depth: int = 3) -> Dict:
    """Map HDF5 file structure."""
    structure = {}
    if max_depth <= 0:
        return structure
    
    for key in h5_obj.keys():
        item = h5_obj[key]
        full_key = f"{prefix}/{key}" if prefix else key
        
        if isinstance(item, h5py.Dataset):
            structure[key] = {
                'type': 'dataset',
                'shape': item.shape,
                'dtype': str(item.dtype)
            }
        elif isinstance(item, h5py.Group):
            structure[key] = {
                'type': 'group',
                'children': map_h5_structure(item, full_key, max_depth - 1)
            }
    
    return structure


def is_gene_id(value: str) -> bool:
    """Check if a value looks like a gene ID."""
    if not value:
        return False
    
    # Common gene ID patterns
    patterns = [
        value.startswith('ENSRNO'),  # Rat Ensembl
        value.startswith('ENSMUS'),  # Mouse Ensembl
        value.startswith('ENSG'),    # Human Ensembl
        value.startswith('ENS'),     # Any Ensembl
        value.startswith('LOC'),     # NCBI gene
        value.startswith('NM_'),     # RefSeq mRNA
        value.startswith('NR_'),     # RefSeq ncRNA
        value.startswith('XM_'),     # RefSeq predicted
        bool(value) and value[0].isupper() and len(value) <= 20,  # Gene symbol
    ]
    return any(patterns)


# =============================================================================
# MAIN EXTRACTION FUNCTION
# =============================================================================

def extract_matrix_stats(filepath: Path, timeout: int = 30) -> Optional[Dict[str, Any]]:
    """
    Extract statistics from a matrix file.
    
    Args:
        filepath: Path to matrix file
        timeout: Maximum seconds to spend parsing (not implemented yet)
    
    Returns:
        Dictionary with matrix statistics or None if not a matrix
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        return {'error': 'File not found'}
    
    fmt = detect_matrix_format(filepath)
    if fmt is None:
        return None
    
    # Get file size
    try:
        size_bytes = filepath.stat().st_size
    except:
        size_bytes = 0
    
    # Skip very large files for now (> 10GB)
    if size_bytes > 10 * 1024 * 1024 * 1024:
        return {
            'format': fmt,
            'skipped': True,
            'reason': 'File too large (>10GB)',
            'size_bytes': size_bytes
        }
    
    # Parse based on format
    if fmt == 'mtx':
        stats = parse_mtx(filepath)
    elif fmt == 'h5ad':
        stats = parse_h5ad(filepath)
    elif fmt == 'h5':
        stats = parse_h5(filepath)
    elif fmt == 'loom':
        stats = parse_loom(filepath)
    elif fmt in ('tsv', 'csv'):
        sep = ',' if fmt == 'csv' else '\t'
        stats = parse_tabular(filepath, sep)
    else:
        return None
    
    stats['filepath'] = str(filepath)
    stats['filename'] = filepath.name
    stats['size_bytes'] = size_bytes
    
    return stats


def scan_study_matrices(study_path: Path, max_matrices: int = 10) -> Dict[str, Any]:
    """
    Scan a study directory for matrix files and extract statistics.
    
    Args:
        study_path: Path to study directory
        max_matrices: Maximum number of matrices to parse
    
    Returns:
        Dictionary with aggregated matrix statistics
    """
    study_path = Path(study_path)
    
    result = {
        'matrices_found': 0,
        'matrices_parsed': 0,
        'total_genes': None,
        'total_cells': None,
        'total_samples': None,
        'gene_ids_sample': [],
        'formats': defaultdict(int),
        'matrices': [],
        'errors': []
    }
    
    # Find potential matrix files
    matrix_files = []
    for fp in study_path.rglob('*'):
        if fp.is_file() and is_likely_matrix(fp):
            matrix_files.append(fp)
    
    result['matrices_found'] = len(matrix_files)
    
    # Parse matrices (up to limit)
    genes_seen = set()
    total_cells = 0
    total_samples = 0
    
    for fp in matrix_files[:max_matrices]:
        stats = extract_matrix_stats(fp)
        if stats is None:
            continue
        
        result['matrices_parsed'] += 1
        
        if 'error' in stats:
            result['errors'].append({'file': fp.name, 'error': stats['error']})
            continue
        
        # Aggregate stats
        fmt = stats.get('format', 'unknown')
        result['formats'][fmt] += 1
        
        n_genes = stats.get('n_genes')
        n_cells = stats.get('n_cells')
        n_samples = stats.get('n_samples')
        
        if n_genes:
            result['total_genes'] = max(result['total_genes'] or 0, n_genes)
        if n_cells:
            total_cells += n_cells
        if n_samples:
            total_samples += n_samples
        
        # Collect gene IDs
        gene_ids = stats.get('gene_ids', [])
        for gid in gene_ids[:50]:
            if gid and gid not in genes_seen:
                genes_seen.add(gid)
                result['gene_ids_sample'].append(gid)
                if len(result['gene_ids_sample']) >= 100:
                    break
        
        # Store matrix info (without gene_ids to save space)
        matrix_info = {k: v for k, v in stats.items() 
                       if k not in ('gene_ids', 'gene_names', 'h5_structure')}
        result['matrices'].append(matrix_info)
    
    # Finalize
    if total_cells > 0:
        result['total_cells'] = total_cells
    if total_samples > 0:
        result['total_samples'] = total_samples
    
    result['formats'] = dict(result['formats'])
    
    return result


# =============================================================================
# CLI INTERFACE
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract matrix statistics')
    parser.add_argument('path', help='Path to matrix file or study directory')
    parser.add_argument('--scan', action='store_true', help='Scan directory for matrices')
    parser.add_argument('--max-matrices', type=int, default=10, 
                        help='Maximum matrices to parse per study')
    parser.add_argument('--json', action='store_true', help='Output as JSON')
    
    args = parser.parse_args()
    path = Path(args.path)
    
    print(f"Available parsers: {', '.join(AVAILABLE_PARSERS)}")
    print()
    
    if args.scan or path.is_dir():
        # Scan directory
        result = scan_study_matrices(path, args.max_matrices)
        
        if args.json:
            print(json.dumps(result, indent=2, default=str))
        else:
            print(f"Study: {path}")
            print(f"Matrices found: {result['matrices_found']}")
            print(f"Matrices parsed: {result['matrices_parsed']}")
            print(f"Total genes: {result['total_genes']}")
            print(f"Total cells: {result['total_cells']}")
            print(f"Total samples: {result['total_samples']}")
            print(f"Formats: {result['formats']}")
            print(f"Gene IDs (sample): {result['gene_ids_sample'][:10]}")
            
            if result['errors']:
                print(f"\nErrors:")
                for err in result['errors'][:5]:
                    print(f"  {err['file']}: {err['error']}")
    else:
        # Single file
        result = extract_matrix_stats(path)
        
        if result is None:
            print("Not a recognized matrix format")
            sys.exit(1)
        
        if args.json:
            print(json.dumps(result, indent=2, default=str))
        else:
            print(f"File: {path}")
            for key, value in result.items():
                if key == 'gene_ids':
                    print(f"  {key}: {value[:10]}... ({len(value)} total)")
                elif key == 'h5_structure':
                    print(f"  {key}: {list(value.keys())}")
                else:
                    print(f"  {key}: {value}")


if __name__ == '__main__':
    main()