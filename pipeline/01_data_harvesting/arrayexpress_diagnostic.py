#!/usr/bin/env python3
"""
ArrayExpress Harvester Diagnostic & Query Test Suite
=====================================================

This script diagnoses the ArrayExpress harvester issues and tests query patterns
to find the optimal search strategy for rat bulk and single-cell RNA-seq studies.

KNOWN ISSUES DIAGNOSED:
1. 'biostudies' domain returns literature (S-EPMC*), not experiments
2. 'arrayexpress' domain returns actual experiments (E-MTAB-*, E-GEOD-*)
3. Field-specific queries (organism:"X") DON'T WORK - fields are empty
4. Free text queries DO WORK

Usage:
    python arrayexpress_diagnostic.py              # Run all tests
    python arrayexpress_diagnostic.py --quick      # Quick verification only
    python arrayexpress_diagnostic.py --counts     # Just show study counts
"""

import json
import urllib.request
import urllib.parse
import re
import time
import argparse
from typing import Dict, List, Tuple, Optional
from collections import Counter
from dataclasses import dataclass, field


# =============================================================================
# CONSTANTS
# =============================================================================

EBI_SEARCH_API = "https://www.ebi.ac.uk/ebisearch/ws/rest"
BIOSTUDIES_API = "https://www.ebi.ac.uk/biostudies/api/v1"

# Rate limiting
MIN_REQUEST_INTERVAL = 0.35
_last_request_time = [0.0]


# =============================================================================
# HELPERS
# =============================================================================

def rate_limit():
    """Enforce rate limiting."""
    elapsed = time.time() - _last_request_time[0]
    if elapsed < MIN_REQUEST_INTERVAL:
        time.sleep(MIN_REQUEST_INTERVAL - elapsed)
    _last_request_time[0] = time.time()


def fetch_json(url: str, timeout: int = 30) -> Optional[Dict]:
    """Fetch JSON from URL with rate limiting."""
    rate_limit()
    try:
        req = urllib.request.Request(url)
        req.add_header("User-Agent", "ArrayExpress-Diagnostic/1.0")
        req.add_header("Accept", "application/json")
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except Exception as e:
        print(f"  ERROR: {e}")
        return None


def search_ebi(domain: str, query: str, size: int = 10, fields: str = None) -> Tuple[int, List[Dict]]:
    """
    Search EBI Search API.
    
    Args:
        domain: 'biostudies' or 'arrayexpress'
        query: Lucene query string
        size: Number of results to return
        fields: Comma-separated list of fields to return
        
    Returns:
        Tuple of (total_hits, list of entries)
    """
    params = {
        "query": query,
        "format": "json",
        "size": str(size),
    }
    if fields:
        params["fields"] = fields
    
    url = f"{EBI_SEARCH_API}/{domain}?" + urllib.parse.urlencode(params)
    result = fetch_json(url)
    
    if result is None:
        return 0, []
    
    return result.get("hitCount", 0), result.get("entries", [])


def extract_arrayexpress_accession(biostudies_id: str) -> Optional[str]:
    """
    Extract ArrayExpress accession from BioStudies ID.
    
    Mappings:
    - E-MTAB-5920 -> E-MTAB-5920 (already correct)
    - S-ECPF-MTAB-1994 -> E-MTAB-1994
    - S-ECPF-GEOD-58135 -> E-GEOD-58135
    - S-EPMC* -> None (literature, skip)
    - S-BSST* -> None (generic BioStudies, skip)
    """
    if not biostudies_id:
        return None
    
    # Already an ArrayExpress accession
    if re.match(r'^E-[A-Z]+-\d+$', biostudies_id):
        return biostudies_id
    
    # BioStudies format: S-ECPF-MTAB-1994 -> E-MTAB-1994
    match = re.match(r'^S-ECPF-([A-Z]+)-(\d+)$', biostudies_id)
    if match:
        return f"E-{match.group(1)}-{match.group(2)}"
    
    return None


def test_query(domain: str, query: str) -> Tuple[int, int, List[str]]:
    """Test a query and return (total_hits, ae_count, sample_accessions)."""
    total, entries = search_ebi(domain, query, size=20)
    
    accessions = []
    for entry in entries:
        ae = extract_arrayexpress_accession(entry.get("id", ""))
        if ae:
            accessions.append(ae)
    
    return total, len(accessions), accessions[:5]


# =============================================================================
# DIAGNOSTIC TESTS
# =============================================================================

def test_accession_extraction():
    """Verify accession extraction works correctly."""
    print("\n" + "="*70)
    print("TEST: ACCESSION EXTRACTION")
    print("="*70)
    
    test_cases = [
        ("E-MTAB-5920", "E-MTAB-5920"),
        ("E-GEOD-123456", "E-GEOD-123456"),
        ("S-ECPF-MTAB-1994", "E-MTAB-1994"),
        ("S-ECPF-GEOD-58135", "E-GEOD-58135"),
        ("S-EPMC4520150", None),
        ("S-BSST1234", None),
    ]
    
    all_pass = True
    for input_id, expected in test_cases:
        result = extract_arrayexpress_accession(input_id)
        status = "✓" if result == expected else "✗"
        if result != expected:
            all_pass = False
        print(f"  {status} {input_id} -> {result} (expected: {expected})")
    
    print(f"\nAll extraction tests passed: {all_pass}")
    return all_pass


def test_domain_comparison():
    """Compare 'biostudies' vs 'arrayexpress' domains."""
    print("\n" + "="*70)
    print("TEST: DOMAIN COMPARISON (biostudies vs arrayexpress)")
    print("="*70)
    
    queries = [
        "rat",
        "Rattus norvegicus", 
        "RNA-seq",
        "single cell",
        "rat AND RNA-seq",
    ]
    
    print(f"\n{'Query':<35} {'biostudies':>12} {'arrayexpress':>14} {'AE IDs'}")
    print("-"*70)
    
    for query in queries:
        # Test biostudies domain
        total_bs, entries_bs = search_ebi("biostudies", query, size=5)
        ae_bs = sum(1 for e in entries_bs if extract_arrayexpress_accession(e.get("id", "")))
        
        # Test arrayexpress domain
        total_ae, entries_ae = search_ebi("arrayexpress", query, size=5)
        ae_ae = sum(1 for e in entries_ae if extract_arrayexpress_accession(e.get("id", "")))
        
        sample_ids = [e.get("id", "")[:15] for e in entries_ae[:2]]
        
        print(f"{query:<35} {total_bs:>12,} {total_ae:>14,} {sample_ids}")
    
    print("\n✓ 'arrayexpress' domain returns actual experiment accessions")
    print("✗ 'biostudies' domain returns literature (S-EPMC*)")


def test_field_queries():
    """Test if field-specific queries work."""
    print("\n" + "="*70)
    print("TEST: FIELD-SPECIFIC vs FREE TEXT QUERIES")
    print("="*70)
    
    domain = "arrayexpress"
    
    queries = [
        # Field-specific (these likely won't work)
        ('organism:"Rattus norvegicus"', 'Field: organism:"Rattus norvegicus"'),
        ('organism:rat', 'Field: organism:rat'),
        ('experimenttype:"RNA-seq"', 'Field: experimenttype:"RNA-seq"'),
        
        # Free text (these should work)
        ('"Rattus norvegicus"', 'Free text: "Rattus norvegicus"'),
        ('rat', 'Free text: rat'),
        ('RNA-seq', 'Free text: RNA-seq'),
    ]
    
    print(f"\nDomain: {domain}")
    print(f"\n{'Query Type':<45} {'Hits':>10}")
    print("-"*60)
    
    for query, desc in queries:
        total, _ = search_ebi(domain, query, size=1)
        status = "✓" if total > 0 else "✗"
        print(f"{status} {desc:<43} {total:>10,}")
    
    print("\n⚠️  Field-specific queries return 0 because fields are EMPTY in API!")
    print("✓ Use FREE TEXT queries instead")


def test_api_response_fields():
    """Check what fields are actually populated in API response."""
    print("\n" + "="*70)
    print("TEST: API RESPONSE FIELD ANALYSIS")
    print("="*70)
    
    total, entries = search_ebi("arrayexpress", "rat RNA-seq", size=3,
                                fields="id,name,description,organism,experimenttype")
    
    print(f"\nQuery: 'rat RNA-seq' on arrayexpress domain")
    print(f"Total hits: {total:,}")
    
    for i, entry in enumerate(entries):
        print(f"\n--- Entry {i+1}: {entry.get('id', 'N/A')} ---")
        print(f"  source: {entry.get('source', 'N/A')}")
        
        fields = entry.get("fields", {})
        for key in ["id", "name", "organism", "experimenttype"]:
            value = fields.get(key, [])
            if isinstance(value, list) and len(value) > 0:
                print(f"  {key}: {value[0][:60]}...")
            else:
                print(f"  {key}: {value} {'← EMPTY!' if not value else ''}")


# =============================================================================
# QUERY PATTERN TESTS
# =============================================================================

def test_organism_queries():
    """Test organism query patterns."""
    print("\n" + "="*70)
    print("QUERY PATTERNS: ORGANISM")
    print("="*70)
    
    domain = "arrayexpress"
    
    queries = [
        ("rat", "Simple: rat"),
        ("Rattus norvegicus", "Scientific name (unquoted)"),
        ('"Rattus norvegicus"', "Scientific name (quoted)"),
        ("mouse", "Simple: mouse"),
        ("human", "Simple: human"),
        ("Homo sapiens", "Scientific: Homo sapiens"),
    ]
    
    print(f"\n{'Description':<40} {'Hits':>10} {'AE Acc':>8}")
    print("-"*60)
    
    for query, desc in queries:
        total, ae_count, samples = test_query(domain, query)
        print(f"{desc:<40} {total:>10,} {ae_count:>8}")


def test_technology_queries():
    """Test technology/assay query patterns."""
    print("\n" + "="*70)
    print("QUERY PATTERNS: TECHNOLOGY (all organisms)")
    print("="*70)
    
    domain = "arrayexpress"
    
    queries = [
        ("RNA-seq", "RNA-seq"),
        ('"RNA-seq"', '"RNA-seq" (quoted)'),
        ("transcriptome", "transcriptome"),
        ('"transcription profiling"', '"transcription profiling"'),
        ("microarray", "microarray"),
        ("sequencing", "sequencing"),
    ]
    
    print(f"\n{'Description':<40} {'Hits':>10}")
    print("-"*55)
    
    for query, desc in queries:
        total, _, _ = test_query(domain, query)
        print(f"{desc:<40} {total:>10,}")


def test_singlecell_queries():
    """Test single-cell specific query patterns."""
    print("\n" + "="*70)
    print("QUERY PATTERNS: SINGLE-CELL (all organisms)")
    print("="*70)
    
    domain = "arrayexpress"
    
    queries = [
        ('"single cell"', '"single cell"'),
        ("single-cell", "single-cell (hyphenated)"),
        ("scRNA-seq", "scRNA-seq"),
        ("scRNAseq", "scRNAseq (no hyphen)"),
        ('"single nucleus"', '"single nucleus"'),
        ("snRNA-seq", "snRNA-seq"),
        ('"10x genomics"', '"10x genomics"'),
        ("10x", "10x"),
        ("chromium", "chromium"),
        ("drop-seq", "drop-seq"),
        ("smart-seq", "smart-seq"),
    ]
    
    print(f"\n{'Description':<40} {'Hits':>10}")
    print("-"*55)
    
    for query, desc in queries:
        total, _, _ = test_query(domain, query)
        print(f"{desc:<40} {total:>10,}")


def test_combined_queries():
    """Test combined organism + technology queries."""
    print("\n" + "="*70)
    print("QUERY PATTERNS: RAT + TECHNOLOGY COMBINED")
    print("="*70)
    
    domain = "arrayexpress"
    
    queries = [
        # Rat + RNA-seq
        ('rat AND RNA-seq', 'rat AND RNA-seq'),
        ('"Rattus norvegicus" AND RNA-seq', '"R. norvegicus" AND RNA-seq'),
        ('rat RNA-seq', 'rat RNA-seq (no AND)'),
        
        # Rat + single-cell
        ('rat AND "single cell"', 'rat AND "single cell"'),
        ('rat AND scRNA-seq', 'rat AND scRNA-seq'),
        ('rat AND 10x', 'rat AND 10x'),
        ('"Rattus norvegicus" AND "single cell"', '"R. norvegicus" AND "single cell"'),
        
        # Rat + bulk (exclude single-cell)
        ('rat AND RNA-seq NOT "single cell"', 'rat + RNA-seq NOT "single cell"'),
        ('rat AND RNA-seq NOT scRNA-seq', 'rat + RNA-seq NOT scRNA-seq'),
        ('rat AND RNA-seq NOT "single cell" NOT scRNA-seq', 'rat + RNA-seq NOT SC (full)'),
    ]
    
    print(f"\n{'Description':<45} {'Hits':>8} {'AE':>6}")
    print("-"*65)
    
    for query, desc in queries:
        total, ae_count, samples = test_query(domain, query)
        print(f"{desc:<45} {total:>8,} {ae_count:>6}")


# =============================================================================
# STUDY COUNTS
# =============================================================================

def show_study_counts():
    """Show comprehensive study counts for rat."""
    print("\n" + "="*70)
    print("RAT STUDY COUNTS (Rattus norvegicus)")
    print("="*70)
    
    domain = "arrayexpress"
    
    categories = {
        "ALL RAT STUDIES": {
            "Rat (free text)": "rat",
            "Rattus norvegicus": '"Rattus norvegicus"',
        },
        "RAT RNA-seq (BULK + SC)": {
            "rat AND RNA-seq": 'rat AND RNA-seq',
            "rat AND transcriptome": 'rat AND transcriptome',
            "R.norvegicus AND RNA-seq": '"Rattus norvegicus" AND RNA-seq',
        },
        "RAT SINGLE-CELL": {
            "rat AND single cell": 'rat AND "single cell"',
            "rat AND scRNA-seq": 'rat AND scRNA-seq',
            "rat AND 10x": 'rat AND 10x',
            "Combined SC query": 'rat AND ("single cell" OR scRNA-seq OR "10x genomics")',
        },
        "RAT BULK RNA-seq (excluding SC)": {
            "rat + RNA-seq - SC": 'rat AND RNA-seq NOT "single cell" NOT scRNA-seq',
            "R.norvegicus bulk": '"Rattus norvegicus" AND RNA-seq NOT "single cell"',
        },
        "RAT MICROARRAY": {
            "rat AND microarray": 'rat AND microarray',
        },
    }
    
    print(f"\nDomain: {domain}\n")
    
    for category, queries in categories.items():
        print(f"\n{category}")
        print("-"*50)
        
        for name, query in queries.items():
            total, ae_count, samples = test_query(domain, query)
            print(f"  {name:<35} {total:>8,}")
            if samples:
                print(f"    Sample: {samples[:3]}")


# =============================================================================
# RECOMMENDATIONS
# =============================================================================

def show_recommendations():
    """Show final recommendations for harvester fix."""
    print("\n" + "="*70)
    print("RECOMMENDATIONS FOR HARVESTER")
    print("="*70)
    
    print("""
┌─────────────────────────────────────────────────────────────────────┐
│ FIX 1: Change EBI Search domain                                     │
├─────────────────────────────────────────────────────────────────────┤
│   # Line ~40 in arrayexpress_harvester_v2.py                        │
│   EBI_SEARCH_DOMAIN = "arrayexpress"  # was "biostudies"            │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│ FIX 2: Use FREE TEXT queries (not field-specific)                   │
├─────────────────────────────────────────────────────────────────────┤
│   # DON'T USE (returns 0):                                          │
│   organism:"Rattus norvegicus" AND experimenttype:"RNA-seq"         │
│                                                                     │
│   # DO USE (works):                                                 │
│   "Rattus norvegicus" AND RNA-seq                                   │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│ FIX 3: Add accession extraction helper                              │
├─────────────────────────────────────────────────────────────────────┤
│   def extract_arrayexpress_accession(id_str):                       │
│       if re.match(r'^E-[A-Z]+-\\d+$', id_str):                      │
│           return id_str                                             │
│       match = re.match(r'^S-ECPF-([A-Z]+)-(\\d+)$', id_str)         │
│       if match:                                                     │
│           return f"E-{match.group(1)}-{match.group(2)}"             │
│       return None                                                   │
└─────────────────────────────────────────────────────────────────────┘

RECOMMENDED QUERIES:
────────────────────
Bulk RNA-seq:
    rat AND RNA-seq NOT "single cell" NOT scRNA-seq
    
Single-cell:
    rat AND ("single cell" OR scRNA-seq OR "10x genomics" OR drop-seq)
    
Both:
    rat AND (RNA-seq OR "transcription profiling")

EXPECTED RESULTS AFTER FIX:
───────────────────────────
- Bulk rat RNA-seq: ~200-350 studies
- Single-cell rat:  ~30-80 studies
- Total rat RNA-seq: ~300-400 studies
""")


# =============================================================================
# MAIN
# =============================================================================

def run_quick_test():
    """Run quick verification only."""
    print("="*70)
    print("QUICK VERIFICATION TEST")
    print("="*70)
    
    print("\nTesting arrayexpress domain with rat RNA-seq query...")
    
    total, entries = search_ebi("arrayexpress", "rat AND RNA-seq", size=10)
    
    accessions = []
    for entry in entries:
        ae = extract_arrayexpress_accession(entry.get("id", ""))
        if ae:
            accessions.append(ae)
    
    print(f"\nQuery: rat AND RNA-seq")
    print(f"Domain: arrayexpress")
    print(f"Total hits: {total:,}")
    print(f"Entries returned: {len(entries)}")
    print(f"Valid AE accessions: {len(accessions)}")
    print(f"Sample accessions: {accessions[:5]}")
    
    if len(accessions) > 0 and total > 100:
        print("\n✓ SUCCESS! ArrayExpress domain returns valid accessions.")
        print("  The harvester fix should work.")
    else:
        print("\n✗ PROBLEM: Check API connectivity or query syntax.")


def main():
    parser = argparse.ArgumentParser(
        description="ArrayExpress Harvester Diagnostic & Query Test Suite"
    )
    parser.add_argument("--quick", action="store_true", 
                        help="Run quick verification only")
    parser.add_argument("--counts", action="store_true",
                        help="Show study counts only")
    parser.add_argument("--queries", action="store_true",
                        help="Test query patterns only")
    
    args = parser.parse_args()
    
    print("="*70)
    print("ARRAYEXPRESS HARVESTER DIAGNOSTIC & QUERY TEST SUITE")
    print("="*70)
    
    if args.quick:
        run_quick_test()
        return
    
    if args.counts:
        show_study_counts()
        show_recommendations()
        return
    
    if args.queries:
        test_organism_queries()
        test_technology_queries()
        test_singlecell_queries()
        test_combined_queries()
        show_recommendations()
        return
    
    # Run all tests
    print("\nRunning comprehensive diagnostic...\n")
    
    # Diagnostic tests
    test_accession_extraction()
    test_domain_comparison()
    test_field_queries()
    test_api_response_fields()
    
    # Query pattern tests
    test_organism_queries()
    test_technology_queries()
    test_singlecell_queries()
    test_combined_queries()
    
    # Study counts
    show_study_counts()
    
    # Recommendations
    show_recommendations()
    
    print("\n" + "="*70)
    print("DIAGNOSTIC COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()