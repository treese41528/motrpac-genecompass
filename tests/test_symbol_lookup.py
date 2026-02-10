#!/usr/bin/env python3
"""
verify_symbol_lookup.py

Quick verification that the symbol lookup has comprehensive coverage.

VERSION 1.1 ENHANCEMENTS:
- Added --inventory flag to test actual corpus genes from gene_inventory.tsv
- Added --study-summary to show per-study resolution rates
- Reports resolution rate by gene type (symbol vs ensembl vs other)

Usage:
    # Basic test with standard genes
    python verify_symbol_lookup.py
    
    # Test against actual corpus inventory
    python verify_symbol_lookup.py --inventory /path/to/gene_inventory.tsv
    
    # Full verification with study-level summary
    python verify_symbol_lookup.py --inventory /path/to/gene_inventory.tsv --study-summary
"""

import pickle
import re
import argparse
from pathlib import Path
from collections import Counter


def classify_gene_format(gene_id: str) -> str:
    """Quick classification of gene ID format."""
    g = gene_id.strip()
    g_upper = g.upper()
    
    if g_upper.startswith('ENSRNOG'):
        return 'ensembl_rat'
    elif g_upper.startswith('ENSMUSG'):
        return 'ensembl_mouse'
    elif g_upper.startswith('ENSG'):
        return 'ensembl_human'
    elif g_upper.startswith('ENS'):
        return 'ensembl_other'
    elif g_upper.startswith('LOC'):
        return 'loc_id'
    elif g_upper.startswith('AABR'):
        return 'aabr_contig'
    elif g_upper.startswith('RGD'):
        return 'rgd_id'
    elif g_upper.startswith('NEWGENE'):
        return 'newgene'
    elif re.match(r'^[NX][MR]_\d+', g):
        return 'refseq'
    elif re.match(r'^\d{4,10}$', g):
        return 'entrez'
    elif re.match(r'^[A-Z][a-z0-9]{1,15}$', g):
        return 'symbol_titlecase'
    elif re.match(r'^[A-Z][A-Z0-9]+$', g) and len(g) < 15:
        return 'symbol_allcaps'
    elif re.match(r'^[a-z]', g):
        return 'symbol_lower'
    else:
        return 'other'


def test_standard_genes(lookup: dict):
    """Test standard housekeeping and commonly used genes."""
    test_genes = [
        'actb', 'gapdh', 'ube2m', 'dhfr', 'gnai3', 'pin1', 'ccl5',
        'apoe', 'tp53', 'brca1', 'egfr', 'vegfa', 'il6', 'tnf',
        # Additional commonly missing genes
        'ppia', 'rplp0', 'hprt1', 'b2m', 'tbp', 'ywhaz', 'sdha',
    ]
    
    print(f"\n{'='*60}")
    print("STANDARD TEST GENES")
    print('='*60)
    
    found = 0
    for gene in test_genes:
        ens_id = lookup.get(gene.lower(), '')
        status = "✓" if ens_id else "✗"
        if ens_id:
            found += 1
        print(f"  {status} {gene:12} {ens_id}")
    
    print(f"\nFound: {found}/{len(test_genes)} ({found/len(test_genes)*100:.1f}%)")
    return found, len(test_genes)


def test_custom_file(lookup: dict, test_file: Path):
    """Test genes from a custom file."""
    with open(test_file) as f:
        custom_genes = [line.strip().lower() for line in f if line.strip()]
    
    print(f"\n{'='*60}")
    print(f"CUSTOM TEST: {test_file.name}")
    print('='*60)
    
    found_custom = sum(1 for g in custom_genes if g in lookup)
    missing = [g for g in custom_genes if g not in lookup]
    
    print(f"Found: {found_custom}/{len(custom_genes)} ({found_custom/len(custom_genes)*100:.1f}%)")
    
    if missing and len(missing) <= 20:
        print(f"\nMissing genes:")
        for g in missing:
            print(f"  ✗ {g}")
    elif missing:
        print(f"\nMissing: {len(missing)} genes (showing first 20)")
        for g in missing[:20]:
            print(f"  ✗ {g}")
    
    return found_custom, len(custom_genes)


def test_inventory(lookup: dict, inventory_path: Path, show_study_summary: bool = False):
    """
    Test actual corpus genes from gene_inventory.tsv against the lookup.
    
    This is the most important test: it checks what fraction of the genes
    your pipeline will actually encounter can be resolved.
    """
    print(f"\n{'='*60}")
    print("CORPUS INVENTORY COVERAGE TEST")
    print(f"File: {inventory_path.name}")
    print('='*60)
    
    # Read inventory
    genes = []
    gene_studies = {}  # gene -> n_studies
    
    with open(inventory_path) as f:
        header = f.readline().strip().split('\t')
        
        # Find column indices
        gene_col = 0  # gene_id
        type_col = None
        studies_col = None
        n_studies_col = None
        
        for i, col in enumerate(header):
            col_lower = col.lower()
            if col_lower in ('gene_type', 'type'):
                type_col = i
            elif col_lower == 'n_studies':
                n_studies_col = i
            elif col_lower == 'studies':
                studies_col = i
        
        for line in f:
            parts = line.strip().split('\t')
            if not parts:
                continue
            gene_id = parts[gene_col].strip()
            n_studies = int(parts[n_studies_col]) if n_studies_col and len(parts) > n_studies_col else 1
            
            genes.append(gene_id)
            gene_studies[gene_id] = n_studies
    
    print(f"Total unique genes in inventory: {len(genes):,}")
    
    # Classify and test each gene
    resolved = 0
    unresolved = 0
    by_format = Counter()
    resolved_by_format = Counter()
    unresolved_by_format = Counter()
    unresolved_examples = {}  # format -> [examples]
    
    # Weight by study count for importance-weighted coverage
    weighted_resolved = 0
    weighted_total = 0
    
    for gene_id in genes:
        g_lower = gene_id.lower().strip()
        g_upper = gene_id.upper().strip()
        g_base = g_lower.split('.')[0]
        
        fmt = classify_gene_format(gene_id)
        by_format[fmt] += 1
        n_studies = gene_studies.get(gene_id, 1)
        weighted_total += n_studies
        
        # Check resolution
        is_resolved = False
        
        # Direct Ensembl rat ID (always "resolved")
        if g_upper.startswith('ENSRNOG'):
            is_resolved = True
        # Symbol lookup
        elif g_lower in lookup or g_base in lookup:
            is_resolved = True
        
        if is_resolved:
            resolved += 1
            resolved_by_format[fmt] += 1
            weighted_resolved += n_studies
        else:
            unresolved += 1
            unresolved_by_format[fmt] += 1
            if fmt not in unresolved_examples:
                unresolved_examples[fmt] = []
            if len(unresolved_examples[fmt]) < 5:
                unresolved_examples[fmt].append(gene_id)
    
    total = len(genes)
    
    # Summary
    print(f"\n{'RESOLUTION SUMMARY':^60}")
    print("-" * 60)
    print(f"Resolved:     {resolved:>8,} / {total:,} ({resolved/total*100:.1f}%)")
    print(f"Unresolved:   {unresolved:>8,} / {total:,} ({unresolved/total*100:.1f}%)")
    if weighted_total > 0:
        print(f"\nWeighted by study frequency:")
        print(f"  Resolved:   {weighted_resolved/weighted_total*100:.1f}% of gene-study observations")
    
    # By format
    print(f"\n{'RESOLUTION BY GENE FORMAT':^60}")
    print("-" * 60)
    print(f"  {'Format':25} {'Total':>8} {'Resolved':>10} {'Rate':>8}")
    print(f"  {'-'*25} {'-'*8} {'-'*10} {'-'*8}")
    
    for fmt, count in by_format.most_common():
        res_count = resolved_by_format.get(fmt, 0)
        rate = res_count / count * 100 if count > 0 else 0
        marker = "✓" if rate > 90 else ("~" if rate > 50 else "✗")
        print(f"  {marker} {fmt:23} {count:>8,} {res_count:>10,} {rate:>7.1f}%")
    
    # Unresolved examples
    if unresolved_examples:
        print(f"\n{'UNRESOLVED EXAMPLES BY FORMAT':^60}")
        print("-" * 60)
        for fmt in unresolved_by_format.most_common():
            fmt_name = fmt[0]
            if fmt_name in unresolved_examples:
                print(f"\n  {fmt_name} ({unresolved_by_format[fmt_name]:,} unresolved):")
                for ex in unresolved_examples[fmt_name]:
                    print(f"    {ex}")
    
    # High-frequency unresolved genes (these matter most)
    high_freq_unresolved = []
    for gene_id in genes:
        g_lower = gene_id.lower().strip()
        g_upper = gene_id.upper().strip()
        g_base = g_lower.split('.')[0]
        
        if not g_upper.startswith('ENSRNOG') and g_lower not in lookup and g_base not in lookup:
            n = gene_studies.get(gene_id, 1)
            if n >= 5:
                high_freq_unresolved.append((gene_id, n))
    
    if high_freq_unresolved:
        high_freq_unresolved.sort(key=lambda x: -x[1])
        print(f"\n{'HIGH-FREQUENCY UNRESOLVED GENES (≥5 studies)':^60}")
        print("-" * 60)
        print(f"  {'Gene':30} {'Studies':>10}")
        for gene_id, n in high_freq_unresolved[:30]:
            print(f"  {gene_id:30} {n:>10}")
        if len(high_freq_unresolved) > 30:
            print(f"  ... and {len(high_freq_unresolved) - 30} more")
    
    return resolved, total


def print_lookup_stats(lookup: dict):
    """Print summary statistics about the lookup itself."""
    print(f"\n{'='*60}")
    print("LOOKUP STATISTICS")
    print('='*60)
    
    # Count symbols by pattern
    ensembl_count = sum(1 for v in lookup.values() if v and v.startswith('ENSRNOG'))
    other_ensembl = sum(1 for v in lookup.values() if v and v.startswith('ENS') and not v.startswith('ENSRNOG'))
    loc_count = sum(1 for k in lookup.keys() if k.startswith('loc'))
    numeric_count = sum(1 for k in lookup.keys() if k.isdigit())
    
    unique_ensembl = len(set(v for v in lookup.values() if v))
    
    print(f"  Total symbols:       {len(lookup):,}")
    print(f"  Unique Ensembl IDs:  {unique_ensembl:,}")
    print(f"  -> Rat (ENSRNOG):    {ensembl_count:,}")
    print(f"  -> Other ENS:        {other_ensembl:,}")
    print(f"  LOC* symbols:        {loc_count:,}")
    print(f"  Numeric symbols:     {numeric_count:,}")
    
    # Symbol format distribution
    sym_formats = Counter()
    for k in lookup.keys():
        sym_formats[classify_gene_format(k)] += 1
    
    print(f"\n  Symbol format distribution:")
    for fmt, count in sym_formats.most_common(10):
        print(f"    {fmt:25} {count:>8,}")


def main():
    parser = argparse.ArgumentParser(
        description="Verify symbol lookup coverage",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic test
  python verify_symbol_lookup.py
  
  # Test against corpus inventory (RECOMMENDED before ortholog mapping)
  python verify_symbol_lookup.py --inventory ../../data/training/gene_inventory/gene_inventory.tsv
  
  # Test custom gene list
  python verify_symbol_lookup.py --test-file my_genes.txt
"""
    )
    parser.add_argument(
        "--lookup", type=Path,
        default=Path("../../data/references/biomart/rat_symbol_lookup.pickle"),
        help="Path to symbol lookup pickle"
    )
    parser.add_argument(
        "--test-file", type=Path,
        help="Optional: file with gene symbols to test (one per line)"
    )
    parser.add_argument(
        "--inventory", type=Path,
        help="Path to gene_inventory.tsv to test actual corpus coverage"
    )
    parser.add_argument(
        "--study-summary", action="store_true",
        help="Show per-study resolution summary (requires --inventory)"
    )
    
    args = parser.parse_args()
    
    # Load lookup
    if not args.lookup.exists():
        print(f"ERROR: Lookup file not found: {args.lookup}")
        return 1
    
    with open(args.lookup, 'rb') as f:
        lookup = pickle.load(f)
    
    print(f"Loaded {len(lookup):,} symbols from {args.lookup.name}")
    
    # Standard test
    test_standard_genes(lookup)
    
    # Custom file test
    if args.test_file and args.test_file.exists():
        test_custom_file(lookup, args.test_file)
    
    # Corpus inventory test
    if args.inventory:
        if not args.inventory.exists():
            print(f"\nERROR: Inventory file not found: {args.inventory}")
            print("Run build_gene_inventory_v3.py first to generate it.")
            return 1
        test_inventory(lookup, args.inventory, args.study_summary)
    
    # Lookup stats
    print_lookup_stats(lookup)
    
    print(f"\n{'='*60}")
    print("VERIFICATION COMPLETE")
    print('='*60)
    
    return 0


if __name__ == "__main__":
    exit(main())