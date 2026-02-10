#!/usr/bin/env python3
"""
fetch_biomart_reference_data.py

Fetch gene symbol reference data from:
1. Ensembl BioMart (primary symbols, synonyms, UniProt, orthologs)
2. RGD (Rat Genome Database) - authoritative source for rat gene nomenclature

The RGD data provides comprehensive rat gene symbols and synonyms that 
BioMart often fails to deliver due to query timeouts.

Outputs:
- {species}_symbol_lookup.pickle  (symbol -> Ensembl ID)
- {species}_all_genes.tsv
- {species}_gene_synonyms.tsv (when available)
- GENES_RAT.txt (RGD source file)
- rat_human_orthologs.tsv
- rat_mouse_orthologs.tsv
- rat_gene_info.tsv
- fetch_metadata.json

Author: Tim Reese / MoTrPAC GeneCompass Project
Date: February 2026
"""

import requests
import time
import argparse
import logging
import json
import pickle
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

# =============================================================================
# CONFIGURATION
# =============================================================================

BIOMART_SERVER = "https://www.ensembl.org/biomart/martservice"
RGD_GENES_URL = "https://download.rgd.mcw.edu/data_release/GENES_RAT.txt"

DEFAULT_OUTPUT_DIR = Path("/depot/reese18/data/references/biomart")

# Test genes to verify coverage
TEST_GENES = [
    'actb', 'gapdh', 'ube2m', 'dhfr', 'gnai3', 'pin1', 'ccl5',
    'apoe', 'tp53', 'brca1', 'egfr', 'vegfa', 'il6', 'tnf'
]


def setup_logging(verbose: bool = False) -> logging.Logger:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)


# =============================================================================
# BIOMART QUERIES
# =============================================================================

def get_query(species: str, query_type: str) -> str:
    """Generate BioMart XML query."""
    
    datasets = {
        'rat': 'rnorvegicus_gene_ensembl',
        'mouse': 'mmusculus_gene_ensembl', 
        'human': 'hsapiens_gene_ensembl'
    }
    
    dataset = datasets[species]
    
    queries = {
        'all_genes': f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE Query>
<Query virtualSchemaName="default" formatter="TSV" header="1" uniqueRows="1" count="" datasetConfigVersion="0.6">
    <Dataset name="{dataset}" interface="default">
        <Attribute name="ensembl_gene_id"/>
        <Attribute name="external_gene_name"/>
        <Attribute name="gene_biotype"/>
    </Dataset>
</Query>""",
        
        'synonyms': f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE Query>
<Query virtualSchemaName="default" formatter="TSV" header="1" uniqueRows="1" count="" datasetConfigVersion="0.6">
    <Dataset name="{dataset}" interface="default">
        <Attribute name="ensembl_gene_id"/>
        <Attribute name="external_gene_name"/>
        <Attribute name="external_synonym"/>
    </Dataset>
</Query>""",
        
        'uniprot': f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE Query>
<Query virtualSchemaName="default" formatter="TSV" header="1" uniqueRows="1" count="" datasetConfigVersion="0.6">
    <Dataset name="{dataset}" interface="default">
        <Attribute name="ensembl_gene_id"/>
        <Attribute name="external_gene_name"/>
        <Attribute name="uniprot_gn_symbol"/>
    </Dataset>
</Query>""",

        'rat_human_orthologs': """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE Query>
<Query virtualSchemaName="default" formatter="TSV" header="1" uniqueRows="1" count="" datasetConfigVersion="0.6">
    <Dataset name="rnorvegicus_gene_ensembl" interface="default">
        <Attribute name="ensembl_gene_id"/>
        <Attribute name="hsapiens_homolog_ensembl_gene"/>
        <Attribute name="hsapiens_homolog_associated_gene_name"/>
        <Attribute name="hsapiens_homolog_orthology_type"/>
        <Attribute name="hsapiens_homolog_perc_id"/>
        <Attribute name="hsapiens_homolog_perc_id_r1"/>
    </Dataset>
</Query>""",

        'rat_mouse_orthologs': """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE Query>
<Query virtualSchemaName="default" formatter="TSV" header="1" uniqueRows="1" count="" datasetConfigVersion="0.6">
    <Dataset name="rnorvegicus_gene_ensembl" interface="default">
        <Attribute name="ensembl_gene_id"/>
        <Attribute name="mmusculus_homolog_ensembl_gene"/>
        <Attribute name="mmusculus_homolog_associated_gene_name"/>
        <Attribute name="mmusculus_homolog_orthology_type"/>
        <Attribute name="mmusculus_homolog_perc_id"/>
    </Dataset>
</Query>""",

        'rat_gene_info': """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE Query>
<Query virtualSchemaName="default" formatter="TSV" header="1" uniqueRows="1" count="" datasetConfigVersion="0.6">
    <Dataset name="rnorvegicus_gene_ensembl" interface="default">
        <Attribute name="ensembl_gene_id"/>
        <Attribute name="external_gene_name"/>
        <Attribute name="gene_biotype"/>
        <Attribute name="chromosome_name"/>
        <Attribute name="description"/>
    </Dataset>
</Query>"""
    }
    
    return queries.get(query_type, '')


def query_biomart(xml_query: str, description: str, logger: logging.Logger, 
                  timeout: int = 600) -> Optional[str]:
    """Execute BioMart query with error handling."""
    logger.info(f"Querying: {description}...")
    start = time.time()
    
    try:
        response = requests.get(
            BIOMART_SERVER,
            params={'query': xml_query},
            timeout=timeout
        )
        response.raise_for_status()
        
        if response.text.startswith('Query ERROR'):
            logger.error(f"BioMart error: {response.text[:500]}")
            return None
        
        elapsed = time.time() - start
        lines = response.text.count('\n')
        logger.info(f"  Got {lines:,} lines in {elapsed:.1f}s")
        return response.text
        
    except requests.exceptions.Timeout:
        logger.error(f"  Timeout after {timeout}s")
        return None
    except Exception as e:
        logger.error(f"Failed: {e}")
        return None


# =============================================================================
# RGD DATA FETCHING
# =============================================================================

def fetch_rgd_data(output_dir: Path, logger: logging.Logger, 
                   timeout: int = 300) -> Optional[Path]:
    """Download RGD GENES_RAT.txt file."""
    
    output_path = output_dir / "GENES_RAT.txt"
    
    logger.info(f"Downloading RGD data from {RGD_GENES_URL}...")
    start = time.time()
    
    try:
        response = requests.get(RGD_GENES_URL, timeout=timeout, stream=True)
        response.raise_for_status()
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        elapsed = time.time() - start
        lines = sum(1 for _ in open(output_path))
        logger.info(f"  Downloaded {lines:,} lines in {elapsed:.1f}s")
        logger.info(f"  Saved: {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Failed to download RGD data: {e}")
        return None


def parse_rgd_data(rgd_path: Path, logger: logging.Logger) -> Dict[str, str]:
    """
    Parse RGD GENES_RAT.txt to extract symbol -> Ensembl ID mapping.
    
    Key columns (0-indexed):
    - 1: SYMBOL (primary gene symbol)
    - 29: OLD_SYMBOL (semicolon-separated synonyms)
    - 37: ENSEMBL_ID (may have multiple, semicolon-separated)
    
    Returns:
        dict mapping lowercase symbol -> Ensembl ID
    """
    
    symbol_to_ensembl = {}
    primary_count = 0
    synonym_count = 0
    
    with open(rgd_path, 'r') as f:
        for line in f:
            # Skip comments and header
            if line.startswith('#') or line.startswith('GENE_RGD_ID\t'):
                continue
            
            parts = line.strip().split('\t')
            if len(parts) < 38:
                continue
            
            # Extract columns (1-indexed in docs, 0-indexed here)
            symbol = parts[1].strip() if len(parts) > 1 else ''
            old_symbols = parts[29].strip() if len(parts) > 29 else ''
            ensembl_ids = parts[37].strip() if len(parts) > 37 else ''
            
            # Skip if no Ensembl ID
            if not ensembl_ids:
                continue
            
            # Use first Ensembl ID if multiple
            ens_id = ensembl_ids.split(';')[0].strip()
            if not ens_id:
                continue
            
            # Add primary symbol
            if symbol:
                key = symbol.lower()
                if key not in symbol_to_ensembl:
                    symbol_to_ensembl[key] = ens_id
                    primary_count += 1
            
            # Add synonyms (OLD_SYMBOL column)
            if old_symbols:
                for syn in old_symbols.split(';'):
                    syn = syn.strip()
                    if syn:
                        key = syn.lower()
                        if key not in symbol_to_ensembl:
                            symbol_to_ensembl[key] = ens_id
                            synonym_count += 1
    
    logger.info(f"  RGD: {primary_count:,} primary symbols, {synonym_count:,} synonyms")
    return symbol_to_ensembl


# =============================================================================
# SYMBOL LOOKUP BUILDING
# =============================================================================

def build_symbol_lookup(output_dir: Path, species: str, logger: logging.Logger,
                        timeout: int = 600, include_rgd: bool = True) -> Dict[str, str]:
    """
    Build comprehensive symbol -> Ensembl ID lookup.
    
    For rat: merges BioMart + RGD (RGD provides synonyms that BioMart often misses)
    For mouse/human: uses BioMart only
    """
    
    lookup = {}
    metadata = {
        'species': species,
        'timestamp': datetime.now().isoformat(),
        'sources': []
    }
    
    # 1. All genes (primary symbols)
    data = query_biomart(get_query(species, 'all_genes'), 
                         f"{species} all genes", logger, timeout)
    if data:
        path = output_dir / f"{species}_all_genes.tsv"
        path.write_text(data)
        
        lines = data.strip().split('\n')[1:]  # skip header
        for line in lines:
            parts = line.split('\t')
            if len(parts) >= 2 and parts[1].strip():
                lookup[parts[1].strip().lower()] = parts[0]
        
        metadata['sources'].append({
            'type': 'biomart_all_genes',
            'records': len(lines),
            'status': 'success'
        })
        logger.info(f"  Added {len(lines):,} primary symbols from BioMart")
    
    # 2. Synonyms (may timeout for rat)
    data = query_biomart(get_query(species, 'synonyms'),
                         f"{species} synonyms", logger, timeout)
    if data:
        path = output_dir / f"{species}_gene_synonyms.tsv"
        path.write_text(data)
        
        lines = data.strip().split('\n')[1:]
        added = 0
        for line in lines:
            parts = line.split('\t')
            if len(parts) >= 3:
                eid = parts[0]
                if parts[1].strip():
                    lookup[parts[1].strip().lower()] = eid
                if parts[2].strip():
                    key = parts[2].strip().lower()
                    if key not in lookup:
                        lookup[key] = eid
                        added += 1
        
        metadata['sources'].append({
            'type': 'biomart_synonyms',
            'records': len(lines),
            'new_symbols': added,
            'status': 'success'
        })
        logger.info(f"  Added {added:,} new synonyms from BioMart")
    else:
        metadata['sources'].append({
            'type': 'biomart_synonyms',
            'status': 'failed_or_timeout'
        })
    
    # 3. UniProt symbols
    data = query_biomart(get_query(species, 'uniprot'),
                         f"{species} UniProt", logger, timeout)
    if data:
        path = output_dir / f"{species}_uniprot.tsv"
        path.write_text(data)
        
        lines = data.strip().split('\n')[1:]
        added = 0
        for line in lines:
            parts = line.split('\t')
            if len(parts) >= 3 and parts[2].strip():
                key = parts[2].strip().lower()
                if key not in lookup:
                    lookup[key] = parts[0]
                    added += 1
        
        metadata['sources'].append({
            'type': 'biomart_uniprot',
            'records': len(lines),
            'new_symbols': added,
            'status': 'success'
        })
        logger.info(f"  Added {added:,} UniProt symbols")
    
    # 4. RGD data (rat only)
    if species == 'rat' and include_rgd:
        logger.info("\n--- RGD Integration ---")
        rgd_path = output_dir / "GENES_RAT.txt"
        
        # Download if not exists or force refresh
        if not rgd_path.exists():
            rgd_path = fetch_rgd_data(output_dir, logger)
        
        if rgd_path and rgd_path.exists():
            rgd_lookup = parse_rgd_data(rgd_path, logger)
            
            # Merge: RGD fills gaps where BioMart is missing
            added = 0
            for symbol, ens_id in rgd_lookup.items():
                if symbol not in lookup:
                    lookup[symbol] = ens_id
                    added += 1
            
            metadata['sources'].append({
                'type': 'rgd',
                'total_symbols': len(rgd_lookup),
                'new_symbols': added,
                'status': 'success'
            })
            logger.info(f"  RGD added {added:,} new symbols to lookup")
        else:
            metadata['sources'].append({
                'type': 'rgd',
                'status': 'failed'
            })
    
    metadata['total_symbols'] = len(lookup)
    return lookup, metadata


def fetch_ortholog_data(output_dir: Path, logger: logging.Logger, 
                        timeout: int = 600) -> Dict:
    """Fetch rat ortholog data for mapping to human/mouse."""
    
    results = {}
    
    # Rat-Human orthologs
    data = query_biomart(get_query('rat', 'rat_human_orthologs'),
                         "rat-human orthologs", logger, timeout)
    if data:
        path = output_dir / "rat_human_orthologs.tsv"
        path.write_text(data)
        results['rat_human'] = {
            'path': str(path),
            'records': data.count('\n') - 1,
            'status': 'success'
        }
    else:
        results['rat_human'] = {'status': 'failed'}
    
    # Rat-Mouse orthologs
    data = query_biomart(get_query('rat', 'rat_mouse_orthologs'),
                         "rat-mouse orthologs", logger, timeout)
    if data:
        path = output_dir / "rat_mouse_orthologs.tsv"
        path.write_text(data)
        results['rat_mouse'] = {
            'path': str(path),
            'records': data.count('\n') - 1,
            'status': 'success'
        }
    else:
        results['rat_mouse'] = {'status': 'failed'}
    
    # Rat gene info
    data = query_biomart(get_query('rat', 'rat_gene_info'),
                         "rat gene info", logger, timeout)
    if data:
        path = output_dir / "rat_gene_info.tsv"
        path.write_text(data)
        results['rat_gene_info'] = {
            'path': str(path),
            'records': data.count('\n') - 1,
            'status': 'success'
        }
    else:
        results['rat_gene_info'] = {'status': 'failed'}
    
    return results


def test_coverage(lookups: Dict[str, Dict], logger: logging.Logger):
    """Test coverage with known genes."""
    
    logger.info("\n" + "=" * 60)
    logger.info("COVERAGE TEST")
    logger.info("=" * 60)
    
    for species, lookup in lookups.items():
        found = sum(1 for g in TEST_GENES if g in lookup)
        logger.info(f"\n{species.upper()}: {found}/{len(TEST_GENES)} test genes found")
        for g in TEST_GENES:
            status = "✓" if g in lookup else "✗"
            eid = lookup.get(g, "")[:20] if g in lookup else ""
            logger.info(f"  {status} {g:12} {eid}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Fetch gene symbol reference data from BioMart and RGD",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full fetch (all species + orthologs)
  python fetch_biomart_reference_data.py -v
  
  # Rat only with RGD
  python fetch_biomart_reference_data.py --species rat -v
  
  # Skip RGD (BioMart only)
  python fetch_biomart_reference_data.py --no-rgd -v
  
  # Refresh RGD data
  python fetch_biomart_reference_data.py --refresh-rgd -v
        """
    )
    parser.add_argument(
        "-o", "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help="Output directory"
    )
    parser.add_argument(
        "--species", nargs='+', default=['rat', 'mouse', 'human'],
        choices=['rat', 'mouse', 'human'],
        help="Species to fetch"
    )
    parser.add_argument(
        "--no-rgd", action="store_true",
        help="Skip RGD download (BioMart only for rat)"
    )
    parser.add_argument(
        "--refresh-rgd", action="store_true",
        help="Force re-download RGD data"
    )
    parser.add_argument(
        "--no-orthologs", action="store_true",
        help="Skip ortholog data fetch"
    )
    parser.add_argument(
        "--timeout", type=int, default=600,
        help="BioMart query timeout in seconds"
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    
    args = parser.parse_args()
    logger = setup_logging(args.verbose)
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("BIOMART + RGD REFERENCE DATA FETCH")
    logger.info("=" * 60)
    logger.info(f"Output: {args.output_dir}")
    logger.info(f"Species: {args.species}")
    logger.info(f"Include RGD: {not args.no_rgd}")
    
    # Track all metadata
    all_metadata = {
        'timestamp': datetime.now().isoformat(),
        'species': {}
    }
    lookups = {}
    
    # Optionally refresh RGD first
    if args.refresh_rgd and 'rat' in args.species:
        rgd_path = args.output_dir / "GENES_RAT.txt"
        if rgd_path.exists():
            rgd_path.unlink()
            logger.info("Removed old GENES_RAT.txt for refresh")
    
    # Build symbol lookups for each species
    for species in args.species:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"FETCHING {species.upper()} DATA")
        logger.info("=" * 60)
        
        include_rgd = (species == 'rat' and not args.no_rgd)
        lookup, metadata = build_symbol_lookup(
            args.output_dir, species, logger, args.timeout, include_rgd
        )
        
        lookups[species] = lookup
        all_metadata['species'][species] = metadata
        
        logger.info(f"\nTotal {species} symbols: {len(lookup):,}")
    
    # Fetch ortholog data
    if not args.no_orthologs and 'rat' in args.species:
        logger.info(f"\n{'=' * 60}")
        logger.info("FETCHING ORTHOLOG DATA")
        logger.info("=" * 60)
        
        ortholog_results = fetch_ortholog_data(args.output_dir, logger, args.timeout)
        all_metadata['orthologs'] = ortholog_results
    
    # Test coverage
    test_coverage(lookups, logger)
    
    # Save lookups as pickle
    for species, lookup in lookups.items():
        path = args.output_dir / f"{species}_symbol_lookup.pickle"
        with open(path, 'wb') as f:
            pickle.dump(lookup, f)
        logger.info(f"Saved: {path} ({len(lookup):,} symbols)")
    
    # Save metadata
    metadata_path = args.output_dir / "fetch_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(all_metadata, f, indent=2)
    logger.info(f"Saved: {metadata_path}")
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    for species, lookup in lookups.items():
        sources = all_metadata['species'][species].get('sources', [])
        source_info = ', '.join(
            f"{s['type']}: {s.get('new_symbols', s.get('records', '?'))}" 
            for s in sources if s.get('status') == 'success'
        )
        logger.info(f"  {species}: {len(lookup):,} symbols ({source_info})")
    
    return 0


if __name__ == "__main__":
    exit(main())