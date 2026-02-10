#!/usr/bin/env python3
"""
check_matrix_results.py - Comprehensive analysis of matrix_analysis.json

Usage:
    python check_matrix_results.py --config config.yaml
    python check_matrix_results.py --config config.yaml --spot-check GSE216247
"""

import json
import sys
import argparse
from collections import Counter
from pathlib import Path

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    if not HAS_YAML:
        raise ImportError("PyYAML required. Install: pip install pyyaml")
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_results(filepath: Path) -> dict:
    """Load matrix analysis results."""
    with open(filepath) as f:
        return json.load(f)


def print_summary(d: dict):
    """Print overall summary statistics."""
    a = d['aggregate']
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Studies analyzed:      {a['studies_analyzed']:,}")
    print(f"Studies with matrices: {a['studies_with_matrices']:,}")
    print(f"Total cells:           {a.get('total_cells', 0):,}")
    print(f"Total samples:         {a.get('total_samples', 0):,}")
    print(f"Max genes:             {a.get('max_genes', 0):,}")
    print(f"Formats:               {a.get('formats', {})}")
    print()
    
    print("BY DATA TYPE:")
    for dt, stats in a.get('by_data_type', {}).items():
        print(f"  {dt}:")
        print(f"    Studies: {stats['studies']:,}")
        print(f"    Cells:   {stats.get('cells', 0):,}")
        print(f"    Samples: {stats.get('samples', 0):,}")
        print(f"    Max genes: {stats.get('max_genes', 0):,}")
    print()


def check_duplicates(d: dict):
    """Check for duplicate accessions."""
    print("=" * 60)
    print("DUPLICATE CHECK")
    print("=" * 60)
    
    accessions = [s['accession'] for s in d['study_stats']]
    dupes = {k: v for k, v in Counter(accessions).items() if v > 1}
    
    print(f"Total entries: {len(accessions)}")
    print(f"Unique accessions: {len(set(accessions))}")
    print(f"Duplicates: {len(dupes)}")
    
    if dupes:
        print("\nDuplicate accessions (showing first 10):")
        for acc, count in list(dupes.items())[:10]:
            print(f"  {acc}: {count} times")
    print()


def check_suspicious_values(d: dict):
    """Check for suspicious duplicate cell/gene counts."""
    print("=" * 60)
    print("SUSPICIOUS VALUE CHECK")
    print("=" * 60)
    
    cell_counts = [s.get('n_cells') for s in d['study_stats'] if s.get('n_cells')]
    gene_counts = [s.get('n_genes') for s in d['study_stats'] if s.get('n_genes')]
    sample_counts = [s.get('n_samples') for s in d['study_stats'] if s.get('n_samples')]
    
    print("Top 10 most common CELL counts:")
    for val, count in Counter(cell_counts).most_common(10):
        flag = " ⚠️  SUSPICIOUS" if count > 3 else ""
        print(f"  {val:>12,} cells: {count} studies{flag}")
    
    print("\nTop 10 most common GENE counts:")
    for val, count in Counter(gene_counts).most_common(10):
        print(f"  {val:>12,} genes: {count} studies")
    
    print("\nTop 10 most common SAMPLE counts:")
    for val, count in Counter(sample_counts).most_common(10):
        print(f"  {val:>12,} samples: {count} studies")
    print()


def check_gene_distribution(d: dict):
    """Check gene count distribution."""
    print("=" * 60)
    print("GENE COUNT DISTRIBUTION")
    print("=" * 60)
    
    gene_counts = [s.get('n_genes') for s in d['study_stats'] if s.get('n_genes')]
    
    ranges = [
        (0, 5000, 'Suspicious low', '⚠️'),
        (5000, 15000, 'Low (subset?)', ''),
        (15000, 25000, 'Normal (protein-coding)', '✓'),
        (25000, 40000, 'Normal (with ncRNA)', '✓'),
        (40000, 100000, 'High (check orientation)', '⚠️'),
        (100000, float('inf'), 'Suspicious high', '❌'),
    ]
    
    for low, high, label, flag in ranges:
        count = sum(1 for g in gene_counts if low <= g < high)
        if count > 0:
            high_str = "∞" if high == float('inf') else f"{high:,}"
            print(f"  {flag} {label:30} ({low:,}-{high_str}): {count} studies")
    print()


def check_null_values(d: dict):
    """Check for null/None values."""
    print("=" * 60)
    print("NULL VALUE CHECK")
    print("=" * 60)
    
    stats = d['study_stats']
    n_total = len(stats)
    
    n_genes_none = sum(1 for s in stats if s.get('n_genes') is None)
    n_cells_none = sum(1 for s in stats if s.get('n_cells') is None)
    n_samples_none = sum(1 for s in stats if s.get('n_samples') is None)
    n_all_none = sum(1 for s in stats 
                     if s.get('n_genes') is None 
                     and s.get('n_cells') is None 
                     and s.get('n_samples') is None)
    
    n_has_formats_but_none = sum(1 for s in stats
                                  if s.get('formats') 
                                  and s.get('n_genes') is None)
    
    print(f"n_genes is None:   {n_genes_none:>5}/{n_total} ({100*n_genes_none/n_total:.1f}%)")
    print(f"n_cells is None:   {n_cells_none:>5}/{n_total} ({100*n_cells_none/n_total:.1f}%)")
    print(f"n_samples is None: {n_samples_none:>5}/{n_total} ({100*n_samples_none/n_total:.1f}%)")
    print(f"All None:          {n_all_none:>5}/{n_total} ({100*n_all_none/n_total:.1f}%)")
    print(f"Has formats but None values: {n_has_formats_but_none}")
    print()


def check_errors(d: dict):
    """Check error distribution."""
    print("=" * 60)
    print("ERROR CHECK")
    print("=" * 60)
    
    errors = {}
    for s in d['study_stats']:
        err = s.get('error')
        if err:
            errors[err] = errors.get(err, 0) + 1
    
    if errors:
        print("Errors encountered:")
        for err, count in sorted(errors.items(), key=lambda x: -x[1]):
            print(f"  {err}: {count}")
    else:
        print("No errors encountered ✓")
    print()


def show_top_studies(d: dict):
    """Show top studies by cell and sample count."""
    print("=" * 60)
    print("TOP STUDIES")
    print("=" * 60)
    
    # Deduplicate for display
    seen = set()
    unique_stats = []
    for s in d['study_stats']:
        if s['accession'] not in seen:
            unique_stats.append(s)
            seen.add(s['accession'])
    
    print("TOP 10 BY CELL COUNT:")
    studies = sorted(unique_stats, key=lambda x: x.get('n_cells') or 0, reverse=True)[:10]
    for s in studies:
        genes = s.get('n_genes') or 0
        cells = s.get('n_cells') or 0
        samples = s.get('n_samples') or 0
        print(f"  {s['accession']}: {genes:,} genes, {cells:,} cells, {samples:,} samples")
    
    print("\nTOP 10 BY SAMPLE COUNT:")
    studies = sorted(unique_stats, key=lambda x: x.get('n_samples') or 0, reverse=True)[:10]
    for s in studies:
        genes = s.get('n_genes') or 0
        samples = s.get('n_samples') or 0
        print(f"  {s['accession']}: {genes:,} genes, {samples:,} samples")
    print()


def show_suspicious_studies(d: dict):
    """Show studies that need investigation."""
    print("=" * 60)
    print("STUDIES NEEDING INVESTIGATION")
    print("=" * 60)
    
    # High gene counts
    high_genes = [s for s in d['study_stats'] if (s.get('n_genes') or 0) > 100000]
    if high_genes:
        print(f"\n>100k GENES ({len(high_genes)} studies):")
        for s in high_genes[:5]:
            print(f"  {s['accession']}: {s.get('n_genes'):,} genes")
            print(f"    formats: {s.get('formats')}")
            print(f"    gene_ids: {s.get('gene_ids_sample', [])[:3]}")
    
    # Studies with formats but no dimensions
    has_formats_no_dims = [s for s in d['study_stats'] 
                          if s.get('formats') 
                          and s.get('n_genes') is None 
                          and s.get('n_cells') is None]
    if has_formats_no_dims:
        print(f"\nHAS FORMATS BUT NO DIMENSIONS ({len(has_formats_no_dims)} studies):")
        for s in has_formats_no_dims[:5]:
            print(f"  {s['accession']}: formats={s.get('formats')}")
    print()


def show_gene_id_types(d: dict):
    """Show distribution of gene ID types."""
    print("=" * 60)
    print("GENE ID TYPES")
    print("=" * 60)
    
    id_types = d['aggregate'].get('gene_id_types', {})
    if id_types:
        for id_type, count in sorted(id_types.items(), key=lambda x: -x[1]):
            print(f"  {id_type}: {count}")
    else:
        print("  No gene ID types detected")
    print()


def show_format_distribution(d: dict):
    """Show distribution of file formats."""
    print("=" * 60)
    print("FORMAT DISTRIBUTION")
    print("=" * 60)
    
    formats = d['aggregate'].get('formats', {})
    total = sum(formats.values())
    for fmt, count in sorted(formats.items(), key=lambda x: -x[1]):
        pct = 100 * count / total if total > 0 else 0
        print(f"  {fmt}: {count} ({pct:.1f}%)")
    print()


def spot_check_study(d: dict, accession: str):
    """Show detailed info for a specific study."""
    print("=" * 60)
    print(f"SPOT CHECK: {accession}")
    print("=" * 60)
    
    for s in d['study_stats']:
        if s['accession'] == accession:
            print(json.dumps(s, indent=2, default=str))
            return
    
    print(f"Study {accession} not found")
    print()


def main():
    parser = argparse.ArgumentParser(description='Check matrix analysis results')
    parser.add_argument('--config', '-c', required=True, help='Path to config.yaml')
    parser.add_argument('--input', '-i', help='Override input path (default: from config)')
    parser.add_argument('--spot-check', '-s', help='Spot check a specific study accession')
    parser.add_argument('--summary-only', action='store_true', help='Show only summary')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    catalog_dir = Path(config.get('catalog_dir', './catalog'))
    
    # Determine input path
    if args.input:
        filepath = Path(args.input)
    else:
        filepath = catalog_dir / 'matrix_analysis.json'
    
    print(f"\nLoading: {filepath}\n")
    
    try:
        d = load_results(filepath)
    except FileNotFoundError:
        print(f"Error: File not found: {filepath}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON: {e}")
        sys.exit(1)
    
    # Run checks
    print_summary(d)
    
    if args.summary_only:
        return
    
    check_duplicates(d)
    check_null_values(d)
    check_errors(d)
    check_gene_distribution(d)
    check_suspicious_values(d)
    show_format_distribution(d)
    show_gene_id_types(d)
    show_top_studies(d)
    show_suspicious_studies(d)
    
    # Spot check if requested
    if args.spot_check:
        spot_check_study(d, args.spot_check)
    
    print("=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == '__main__':
    main()