#!/usr/bin/env python3
"""
compute_gene_medians.py (v2 — vectorized)

Compute per-gene non-zero median expression values from raw counts
across the entire rat scRNA-seq corpus (895 QC'd h5ad files, 10.2M cells).

GeneCompass normalizes raw counts by dividing each gene's expression by its
corpus-wide non-zero median before applying log2(1+x). This script computes
that median dictionary for our rat corpus.

Approach (v2):
  Uses scipy sparse CSC column access + numpy unique/bincount to build
  per-gene frequency histograms without any Python-level cell iteration.
  
  For each file:
    1. Load raw.X as scipy CSC sparse matrix
    2. For each valid gene column: extract non-zero data via CSC slicing
    3. Use np.unique(return_counts=True) to get frequency table
    4. Merge into global Counter per gene
  
  This is orders of magnitude faster than v1's cell-by-cell loop.

Inputs:
  - 895 QC'd h5ad files in /depot/reese18/data/training/preprocessed/qc_matrices/
  - Raw counts from .raw.X (verified integer counts)
  - Gene list: /depot/reese18/data/training/preprocessed/rat_ensembl_ids_pruned.txt

Outputs:
  - rat_gene_median.pickle        — {ENSRNOG_ID: float} median dict
  - rat_gene_median_stats.json    — computation statistics
  - rat_gene_median_summary.txt   — human-readable summary

Author: Tim Reese / MoTrPAC GeneCompass Project
Date: February 2026
"""

import numpy as np
import scipy.sparse as sp
import h5py
import glob
import os
import sys
import json
import pickle
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


# =============================================================================
# CONFIGURATION
# =============================================================================

QC_DIR = "/depot/reese18/data/training/preprocessed/qc_matrices"
GENE_LIST = "/depot/reese18/data/training/preprocessed/rat_ensembl_ids_pruned.txt"
OUT_DIR = "/depot/reese18/data/training/ortholog_mappings"

# Minimum non-zero cells for a gene to have a valid median
MIN_NONZERO_CELLS = 10


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def median_from_counter(freq: Counter, total_nonzero: int) -> float:
    """Compute median from a frequency distribution of integer counts."""
    if total_nonzero == 0:
        return 0.0
    
    sorted_values = sorted(freq.keys())
    
    if total_nonzero % 2 == 1:
        target = (total_nonzero + 1) // 2
        cumsum = 0
        for val in sorted_values:
            cumsum += freq[val]
            if cumsum >= target:
                return float(val)
    else:
        target_lo = total_nonzero // 2
        target_hi = target_lo + 1
        cumsum = 0
        val_lo = None
        for val in sorted_values:
            cumsum += freq[val]
            if val_lo is None and cumsum >= target_lo:
                val_lo = val
            if cumsum >= target_hi:
                return (val_lo + val) / 2.0
    
    return float(sorted_values[-1])


def load_sparse_raw(h5file):
    """Load raw/X from h5ad as scipy CSC sparse matrix.
    
    h5ad stores sparse matrices in CSR format (indptr over rows=cells).
    We convert to CSC so column (gene) access is O(nnz_col) not O(nnz_total).
    """
    raw_x = h5file['raw/X']
    
    if isinstance(raw_x, h5py.Group):
        data = raw_x['data'][:]
        indices = raw_x['indices'][:]
        indptr = raw_x['indptr'][:]
        
        # Determine shape
        n_rows = len(indptr) - 1  # cells
        n_cols = int(indices.max()) + 1 if len(indices) > 0 else 0
        
        # Also check var for n_cols
        if 'raw' in h5file and 'var' in h5file['raw'] and '_index' in h5file['raw/var']:
            n_cols_var = len(h5file['raw/var/_index'])
            n_cols = max(n_cols, n_cols_var)
        elif 'var' in h5file and '_index' in h5file['var']:
            n_cols_var = len(h5file['var/_index'])
            n_cols = max(n_cols, n_cols_var)
        
        # Build CSR then convert to CSC
        csr = sp.csr_matrix((data, indices, indptr), shape=(n_rows, n_cols))
        return csr.tocsc()
    else:
        # Dense — convert to CSC sparse
        dense = raw_x[:]
        return sp.csc_matrix(dense)


def get_gene_names(h5file):
    """Extract gene names from h5ad file."""
    if 'raw' in h5file and 'var' in h5file['raw'] and '_index' in h5file['raw/var']:
        names = h5file['raw/var/_index'][:]
    elif 'var' in h5file and '_index' in h5file['var']:
        names = h5file['var/_index'][:]
    else:
        return None
    
    return [g.decode() if isinstance(g, bytes) else g for g in names]


# =============================================================================
# MAIN COMPUTATION
# =============================================================================

def compute_gene_medians():
    start_time = datetime.now()
    print("=" * 70)
    print("COMPUTING RAT GENE NON-ZERO MEDIANS (v2 — vectorized)")
    print("=" * 70)
    print(f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"QC dir:  {QC_DIR}")
    print(f"Gene list: {GENE_LIST}")
    print(f"Output:  {OUT_DIR}")
    sys.stdout.flush()
    
    # Load gene list
    with open(GENE_LIST) as f:
        valid_genes = {line.strip() for line in f if line.strip()}
    print(f"Valid genes (pruned list): {len(valid_genes):,}")
    
    # Find h5ad files
    files = sorted(glob.glob(os.path.join(QC_DIR, "*.h5ad")))
    print(f"H5AD files: {len(files)}")
    print()
    sys.stdout.flush()
    
    if not files:
        print("ERROR: No h5ad files found!")
        return 1
    
    # Accumulate per-gene frequency histograms
    gene_freq = defaultdict(Counter)
    gene_nonzero_total = Counter()
    
    total_cells = 0
    total_nnz = 0
    files_processed = 0
    errors = []
    
    for i, fpath in enumerate(files):
        fname = os.path.basename(fpath)
        
        try:
            with h5py.File(fpath, 'r') as h:
                if 'raw' not in h or 'X' not in h['raw']:
                    errors.append(f"{fname}: missing raw/X")
                    continue
                
                # Get gene names
                gene_names = get_gene_names(h)
                if gene_names is None:
                    errors.append(f"{fname}: cannot find gene names")
                    continue
                
                # Build column index -> gene_id for valid genes
                valid_cols = {}
                for col_idx, gene_id in enumerate(gene_names):
                    if gene_id in valid_genes:
                        valid_cols[col_idx] = gene_id
                
                if not valid_cols:
                    errors.append(f"{fname}: no valid genes")
                    continue
                
                # Load as CSC sparse matrix
                csc = load_sparse_raw(h)
                n_cells = csc.shape[0]
                total_cells += n_cells
                
                # Process each valid gene column (vectorized)
                file_nnz = 0
                for col_idx, gene_id in valid_cols.items():
                    # CSC column slice: O(nnz in column)
                    col_start = csc.indptr[col_idx]
                    col_end = csc.indptr[col_idx + 1]
                    col_data = csc.data[col_start:col_end]
                    
                    if len(col_data) == 0:
                        continue
                    
                    # Round to int (handles float32 storage of integers)
                    int_data = np.rint(col_data).astype(np.int64)
                    
                    # Filter to positive (should be all, but safety)
                    int_data = int_data[int_data > 0]
                    
                    if len(int_data) == 0:
                        continue
                    
                    # Frequency table via numpy unique
                    values, counts = np.unique(int_data, return_counts=True)
                    
                    for v, c in zip(values, counts):
                        gene_freq[gene_id][int(v)] += int(c)
                    
                    gene_nonzero_total[gene_id] += len(int_data)
                    file_nnz += len(int_data)
                
                total_nnz += file_nnz
                files_processed += 1
                
        except Exception as e:
            errors.append(f"{fname}: {str(e)}")
            print(f"  ERROR: {fname}: {e}")
            sys.stdout.flush()
        
        # Progress reporting
        if (i + 1) % 25 == 0 or i == 0 or (i + 1) == len(files):
            elapsed = (datetime.now() - start_time).total_seconds()
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (len(files) - i - 1) / rate if rate > 0 else 0
            print(f"  [{i+1:>3}/{len(files)}] "
                  f"{rate:.1f} files/s | "
                  f"ETA {eta/60:.0f}m | "
                  f"cells: {total_cells:,} | "
                  f"nnz: {total_nnz:,} | "
                  f"genes: {len(gene_freq):,}")
            sys.stdout.flush()
    
    print()
    print(f"Files processed: {files_processed}/{len(files)}")
    print(f"Total cells: {total_cells:,}")
    print(f"Total non-zero entries (valid genes): {total_nnz:,}")
    print(f"Genes with data: {len(gene_freq):,}")
    if errors:
        print(f"Errors: {len(errors)}")
        for e in errors[:10]:
            print(f"  {e}")
    print()
    sys.stdout.flush()
    
    # =========================================================================
    # COMPUTE MEDIANS
    # =========================================================================
    
    print("Computing medians...")
    
    gene_medians = {}
    low_count_genes = 0
    zero_median_genes = 0
    
    for gene_id in sorted(gene_freq.keys()):
        freq = gene_freq[gene_id]
        n_nonzero = gene_nonzero_total[gene_id]
        
        if n_nonzero < MIN_NONZERO_CELLS:
            gene_medians[gene_id] = 1.0
            low_count_genes += 1
        else:
            med = median_from_counter(freq, n_nonzero)
            if med <= 0:
                gene_medians[gene_id] = 1.0
                zero_median_genes += 1
            else:
                gene_medians[gene_id] = med
    
    # Add genes not observed in any file
    missing_genes = valid_genes - set(gene_medians.keys())
    for gene_id in missing_genes:
        gene_medians[gene_id] = 1.0
    
    print(f"Gene medians computed: {len(gene_medians):,}")
    print(f"  With sufficient data: {len(gene_medians) - low_count_genes - len(missing_genes):,}")
    print(f"  Low count (default 1.0): {low_count_genes:,}")
    print(f"  Zero median (default 1.0): {zero_median_genes:,}")
    print(f"  Not observed (default 1.0): {len(missing_genes):,}")
    
    # Median distribution stats
    non_default = [v for g, v in gene_medians.items()
                   if gene_nonzero_total.get(g, 0) >= MIN_NONZERO_CELLS]
    
    if non_default:
        print(f"\nMedian value distribution (genes with sufficient data):")
        print(f"  Min:    {min(non_default):.2f}")
        print(f"  Q25:    {np.percentile(non_default, 25):.2f}")
        print(f"  Median: {np.percentile(non_default, 50):.2f}")
        print(f"  Q75:    {np.percentile(non_default, 75):.2f}")
        print(f"  Max:    {max(non_default):.2f}")
        print(f"  Mean:   {np.mean(non_default):.2f}")
    
    # Compare to GeneCompass human medians
    gc_median_path = "/depot/reese18/apps/GeneCompass/scdata/dict/human_gene_median_after_filter.pickle"
    if os.path.exists(gc_median_path):
        with open(gc_median_path, 'rb') as f:
            human_medians = pickle.load(f)
        human_vals = list(human_medians.values())
        print(f"\nComparison with GeneCompass human medians ({len(human_medians):,} genes):")
        print(f"  Human - min: {min(human_vals):.2f}, median: {np.median(human_vals):.2f}, "
              f"max: {max(human_vals):.2f}, mean: {np.mean(human_vals):.2f}")
        if non_default:
            print(f"  Rat   - min: {min(non_default):.2f}, median: {np.median(non_default):.2f}, "
                  f"max: {max(non_default):.2f}, mean: {np.mean(non_default):.2f}")
    
    sys.stdout.flush()
    
    # =========================================================================
    # SAVE OUTPUTS
    # =========================================================================
    
    os.makedirs(OUT_DIR, exist_ok=True)
    
    # Pickle
    median_path = os.path.join(OUT_DIR, "rat_gene_median.pickle")
    with open(median_path, 'wb') as f:
        pickle.dump(gene_medians, f)
    print(f"\nSaved: {median_path} ({len(gene_medians):,} genes)")
    
    # Statistics JSON
    end_time = datetime.now()
    elapsed = (end_time - start_time).total_seconds()
    
    stats = {
        "timestamp": end_time.isoformat(),
        "elapsed_seconds": elapsed,
        "elapsed_human": f"{elapsed/60:.1f} minutes",
        "input": {
            "qc_dir": QC_DIR,
            "gene_list": GENE_LIST,
            "n_files": len(files),
            "files_processed": files_processed,
            "n_valid_genes": len(valid_genes),
        },
        "corpus": {
            "total_cells": total_cells,
            "total_nnz_valid_genes": total_nnz,
            "genes_with_data": len(gene_freq),
        },
        "medians": {
            "total_genes": len(gene_medians),
            "with_sufficient_data": len(gene_medians) - low_count_genes - len(missing_genes),
            "low_count_default": low_count_genes,
            "zero_median_default": zero_median_genes,
            "not_observed_default": len(missing_genes),
            "min_nonzero_threshold": MIN_NONZERO_CELLS,
        },
        "median_distribution": {
            "min": float(min(non_default)) if non_default else 0,
            "q25": float(np.percentile(non_default, 25)) if non_default else 0,
            "median": float(np.percentile(non_default, 50)) if non_default else 0,
            "q75": float(np.percentile(non_default, 75)) if non_default else 0,
            "max": float(max(non_default)) if non_default else 0,
            "mean": float(np.mean(non_default)) if non_default else 0,
        },
        "nonzero_count_distribution": {
            "min": int(min(gene_nonzero_total.values())) if gene_nonzero_total else 0,
            "median": int(np.median(list(gene_nonzero_total.values()))) if gene_nonzero_total else 0,
            "max": int(max(gene_nonzero_total.values())) if gene_nonzero_total else 0,
        },
        "errors": errors,
    }
    
    stats_path = os.path.join(OUT_DIR, "rat_gene_median_stats.json")
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    # Human-readable summary
    summary_path = os.path.join(OUT_DIR, "rat_gene_median_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("Rat Gene Non-Zero Median Computation\n")
        f.write("=" * 50 + "\n")
        f.write(f"Date: {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Elapsed: {elapsed/60:.1f} minutes\n")
        f.write(f"\nCorpus:\n")
        f.write(f"  Files: {files_processed}/{len(files)}\n")
        f.write(f"  Cells: {total_cells:,}\n")
        f.write(f"  Non-zero entries (valid genes): {total_nnz:,}\n")
        f.write(f"\nMedians:\n")
        f.write(f"  Total genes: {len(gene_medians):,}\n")
        f.write(f"  With data (>={MIN_NONZERO_CELLS} cells): "
                f"{len(gene_medians) - low_count_genes - len(missing_genes):,}\n")
        f.write(f"  Low count (default 1.0): {low_count_genes:,}\n")
        f.write(f"  Not observed (default 1.0): {len(missing_genes):,}\n")
        if non_default:
            f.write(f"\nMedian value distribution:\n")
            f.write(f"  Min:    {min(non_default):.2f}\n")
            f.write(f"  Q25:    {np.percentile(non_default, 25):.2f}\n")
            f.write(f"  Median: {np.percentile(non_default, 50):.2f}\n")
            f.write(f"  Q75:    {np.percentile(non_default, 75):.2f}\n")
            f.write(f"  Max:    {max(non_default):.2f}\n")
        if errors:
            f.write(f"\nErrors ({len(errors)}):\n")
            for e in errors:
                f.write(f"  {e}\n")
    
    print(f"\nFinished: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Elapsed: {elapsed/60:.1f} minutes")
    sys.stdout.flush()
    
    return 0


if __name__ == "__main__":
    sys.exit(compute_gene_medians())