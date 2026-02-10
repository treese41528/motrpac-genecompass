#!/usr/bin/env python3
"""
build_ortholog_mapping.py

Build rat-to-human/mouse ortholog mappings for GeneCompass fine-tuning.

Requires: Run fetch_biomart_orthologs.py first to get reference data.

Inputs:
- Corpus rat Ensembl IDs (from build_gene_inventory.py)
- BioMart reference data (from fetch_biomart_orthologs.py)
- GeneCompass vocabulary

Outputs:
- rat_tokens.pickle (rat gene → token ID)
- rat_to_human_mapping.pickle
- rat_to_mouse_mapping.pickle
- rat_mapping_tiers.pickle
- mapping_statistics.json
- rat_token_mapping.tsv (with quality columns)

Author: Tim Reese / MoTrPAC GeneCompass Project
Date: February 2026
"""

import pandas as pd
import pickle
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Set
from collections import Counter

# =============================================================================
# CONFIGURATION
# =============================================================================

# Default identity thresholds by orthology type
DEFAULT_MIN_IDENTITY = {
    'ortholog_one2one': 50.0,
    'ortholog_one2many': 70.0,
    'ortholog_many2many': 80.0,
}

DEFAULT_CORPUS_GENES = Path("../../data/training/gene_inventory/rat_ensembl_ids.txt")
DEFAULT_BIOMART_DIR = Path("../../data/references/biomart")
DEFAULT_GENECOMPASS_VOCAB = Path("../GeneCompass/prior_knowledge/human_mouse_tokens.pickle")
DEFAULT_OUTPUT_DIR = Path("../../data/training/ortholog_mappings")


def setup_logging(verbose: bool = False) -> logging.Logger:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)


# =============================================================================
# DATA LOADING
# =============================================================================

def load_corpus_genes(corpus_path: Path, logger: logging.Logger) -> Set[str]:
    """Load rat Ensembl IDs from gene inventory."""
    if not corpus_path.exists():
        logger.error(f"Corpus file not found: {corpus_path}")
        logger.error("Run build_gene_inventory.py first")
        return set()
    
    with open(corpus_path, 'r') as f:
        genes = {line.strip().upper() for line in f if line.strip()}
    
    # Add base IDs (without version)
    base_genes = {g.split('.')[0] for g in genes}
    genes = genes | base_genes
    
    logger.info(f"Loaded {len(base_genes):,} corpus rat Ensembl IDs")
    return genes


def load_biomart_data(biomart_dir: Path, logger: logging.Logger) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load BioMart reference data."""
    
    human_path = biomart_dir / "rat_human_orthologs.tsv"
    mouse_path = biomart_dir / "rat_mouse_orthologs.tsv"
    geneinfo_path = biomart_dir / "rat_gene_info.tsv"
    
    for p in [human_path, mouse_path, geneinfo_path]:
        if not p.exists():
            logger.error(f"Missing BioMart file: {p}")
            logger.error("Run fetch_biomart_orthologs.py first")
            raise FileNotFoundError(p)
    
    # Load rat-human orthologs
    human_df = pd.read_csv(human_path, sep='\t')
    human_df.columns = ['rat_gene_id', 'human_gene_id', 'human_gene_name', 
                        'human_orthology_type', 'human_perc_id', 'human_perc_id_r1']
    human_df['rat_gene_id'] = human_df['rat_gene_id'].str.upper()
    human_df['human_gene_id'] = human_df['human_gene_id'].fillna('').str.upper()
    human_df['human_perc_id'] = pd.to_numeric(human_df['human_perc_id'], errors='coerce')
    logger.info(f"Loaded {len(human_df):,} rat-human ortholog records")
    
    # Load rat-mouse orthologs
    mouse_df = pd.read_csv(mouse_path, sep='\t')
    mouse_df.columns = ['rat_gene_id', 'mouse_gene_id', 'mouse_gene_name',
                        'mouse_orthology_type', 'mouse_perc_id']
    mouse_df['rat_gene_id'] = mouse_df['rat_gene_id'].str.upper()
    mouse_df['mouse_gene_id'] = mouse_df['mouse_gene_id'].fillna('').str.upper()
    mouse_df['mouse_perc_id'] = pd.to_numeric(mouse_df['mouse_perc_id'], errors='coerce')
    logger.info(f"Loaded {len(mouse_df):,} rat-mouse ortholog records")
    
    # Load gene info
    gene_df = pd.read_csv(geneinfo_path, sep='\t')
    gene_df.columns = ['gene_id', 'gene_name', 'biotype', 'chromosome', 'description']
    gene_df['gene_id'] = gene_df['gene_id'].str.upper()
    logger.info(f"Loaded {len(gene_df):,} rat gene info records")
    
    return human_df, mouse_df, gene_df


# In load_biomart_data() or create new function:
def load_rgd_biotypes(rgd_path: Path, logger: logging.Logger) -> Dict[str, str]:
    """Load gene types from RGD as fallback for BioMart."""
    ens_to_type = {}
    
    # Map RGD types to our categories
    type_map = {
        'protein-coding': 'protein_coding',
        'lincrna': 'lncRNA',
        'lncrna': 'lncRNA',
        'mirna': 'miRNA',
    }
    
    if not rgd_path.exists():
        logger.warning(f"RGD file not found: {rgd_path}")
        return ens_to_type
    
    with open(rgd_path) as f:
        for line in f:
            if line.startswith('#') or line.startswith('GENE_RGD_ID'):
                continue
            parts = line.strip().split('\t')
            if len(parts) > 37:
                gene_type = parts[36].strip().lower()
                ens_ids = parts[37].strip()
                if ens_ids and gene_type:
                    mapped_type = type_map.get(gene_type, gene_type)
                    for eid in ens_ids.split(';'):
                        eid = eid.strip().upper()
                        if eid:
                            ens_to_type[eid] = mapped_type
    
    logger.info(f"Loaded RGD biotypes: {len(ens_to_type):,} genes")
    return ens_to_type


def load_genecompass_data(vocab_path: Path, logger: logging.Logger) -> Tuple[Dict, Dict, Set[str], Set[str]]:
    """Load GeneCompass vocabulary and homolog map."""
    
    if not vocab_path.exists():
        logger.warning(f"GeneCompass vocab not found: {vocab_path}")
        return {}, {}, set(), set()
    
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    
    human_genes = {k for k in vocab.keys() if isinstance(k, str) and k.startswith('ENSG')}
    mouse_genes = {k for k in vocab.keys() if isinstance(k, str) and k.startswith('ENSMUSG')}
    
    logger.info(f"GeneCompass vocab: {len(vocab):,} tokens ({len(human_genes):,} human, {len(mouse_genes):,} mouse)")
    
    # Load homolog map
    homolog_path = vocab_path.parent / "homologous_hm_token.pickle"
    if homolog_path.exists():
        with open(homolog_path, 'rb') as f:
            homologs = pickle.load(f)
        logger.info(f"GeneCompass homologs: {len(homologs):,} mouse→human pairs")
    else:
        logger.warning("GeneCompass homolog map not found")
        homologs = {}
    
    return vocab, homologs, human_genes, mouse_genes


# =============================================================================
# ORTHOLOG PROCESSING
# =============================================================================

def merge_and_filter_orthologs(
    human_df: pd.DataFrame,
    mouse_df: pd.DataFrame,
    gene_df: pd.DataFrame,
    corpus_genes: Set[str],
    identity_thresholds: Dict[str, float],
    one2one_only: bool,
    logger: logging.Logger
) -> pd.DataFrame:
    """Merge ortholog data and filter to corpus genes with quality threshold."""
    
    corpus_base = {g.split('.')[0] for g in corpus_genes}
    
    # Merge human and mouse orthologs
    merged = human_df.merge(mouse_df, on='rat_gene_id', how='outer')
    merged['human_gene_id'] = merged['human_gene_id'].fillna('')
    merged['mouse_gene_id'] = merged['mouse_gene_id'].fillna('')
    
    # Add gene info
    gene_slim = gene_df[['gene_id', 'gene_name', 'biotype']].copy()
    gene_slim.columns = ['rat_gene_id', 'rat_gene_name', 'rat_biotype']
    merged = merged.merge(gene_slim, on='rat_gene_id', how='left')
    
    logger.info(f"Total ortholog records: {len(merged):,}")
    
    # Filter to corpus
    merged['rat_gene_base'] = merged['rat_gene_id'].str.split('.').str[0]
    corpus_mask = merged['rat_gene_base'].isin(corpus_base)
    filtered = merged[corpus_mask].copy()
    
    logger.info(f"After corpus filter: {filtered['rat_gene_id'].nunique():,} unique rat genes")
    
    # Log orthology type distribution before filtering
    logger.info("\nOrthology type distribution (before quality filter):")
    human_types = filtered[filtered['human_gene_id'] != '']['human_orthology_type'].value_counts()
    mouse_types = filtered[filtered['mouse_gene_id'] != '']['mouse_orthology_type'].value_counts()
    logger.info(f"  Human: {human_types.to_dict()}")
    logger.info(f"  Mouse: {mouse_types.to_dict()}")
    
    # Apply quality filters based on orthology type
    if one2one_only:
        logger.info("\nFiltering: one2one orthologs only")
        filtered['good_human'] = (
            (filtered['human_gene_id'] != '') & 
            (filtered['human_orthology_type'] == 'ortholog_one2one') &
            (filtered['human_perc_id'] >= identity_thresholds.get('ortholog_one2one', 50.0))
        )
        filtered['good_mouse'] = (
            (filtered['mouse_gene_id'] != '') & 
            (filtered['mouse_orthology_type'] == 'ortholog_one2one') &
            (filtered['mouse_perc_id'] >= identity_thresholds.get('ortholog_one2one', 50.0))
        )
    else:
        logger.info(f"\nFiltering with identity thresholds: {identity_thresholds}")
        
        # Apply type-specific thresholds
        def check_human_quality(row):
            if row['human_gene_id'] == '' or pd.isna(row['human_orthology_type']):
                return False
            otype = row['human_orthology_type']
            threshold = identity_thresholds.get(otype, 50.0)
            return row['human_perc_id'] >= threshold
        
        def check_mouse_quality(row):
            if row['mouse_gene_id'] == '' or pd.isna(row['mouse_orthology_type']):
                return False
            otype = row['mouse_orthology_type']
            threshold = identity_thresholds.get(otype, 50.0)
            return row['mouse_perc_id'] >= threshold
        
        filtered['good_human'] = filtered.apply(check_human_quality, axis=1)
        filtered['good_mouse'] = filtered.apply(check_mouse_quality, axis=1)
    
    # Log results
    good_human_count = filtered['good_human'].sum()
    good_mouse_count = filtered['good_mouse'].sum()
    logger.info(f"Good human orthologs: {good_human_count:,}")
    logger.info(f"Good mouse orthologs: {good_mouse_count:,}")
    
    # Log type breakdown after filtering
    good_human_df = filtered[filtered['good_human']]
    good_mouse_df = filtered[filtered['good_mouse']]
    
    if len(good_human_df) > 0:
        logger.info(f"  Human type breakdown: {good_human_df['human_orthology_type'].value_counts().to_dict()}")
        logger.info(f"  Human identity - mean: {good_human_df['human_perc_id'].mean():.1f}%, median: {good_human_df['human_perc_id'].median():.1f}%")
    
    if len(good_mouse_df) > 0:
        logger.info(f"  Mouse type breakdown: {good_mouse_df['mouse_orthology_type'].value_counts().to_dict()}")
        logger.info(f"  Mouse identity - mean: {good_mouse_df['mouse_perc_id'].mean():.1f}%, median: {good_mouse_df['mouse_perc_id'].median():.1f}%")
    
    return filtered


def resolve_best_per_gene(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """Select best ortholog for each rat gene.
    
    Priority:
    1. one2one with highest identity
    2. one2many with highest identity
    3. many2many with highest identity
    
    Prefer human over mouse.
    """
    
    df = df.copy()
    
    type_rank = {'ortholog_one2one': 0, 'ortholog_one2many': 1, 'ortholog_many2many': 2}
    df['human_rank'] = df['human_orthology_type'].map(type_rank).fillna(99)
    df['mouse_rank'] = df['mouse_orthology_type'].map(type_rank).fillna(99)
    
    # Priority: good human > good mouse, then by type rank, then by identity
    def priority(row):
        if row['good_human']:
            # Human: type_rank * 100 - identity (lower = better)
            return row['human_rank'] * 100 - (row['human_perc_id'] or 0)
        elif row['good_mouse']:
            # Mouse: 300 + type_rank * 100 - identity
            return 300 + row['mouse_rank'] * 100 - (row['mouse_perc_id'] or 0)
        return 999
    
    df['priority'] = df.apply(priority, axis=1)
    df = df.sort_values(['rat_gene_base', 'priority'])
    df = df.drop_duplicates('rat_gene_base', keep='first')
    
    logger.info(f"Resolved to {len(df):,} unique rat genes")
    return df


# =============================================================================
# TOKEN MAPPING
# =============================================================================

def create_token_mapping(
    orthologs: pd.DataFrame,
    corpus_genes: Set[str],
    gene_info: pd.DataFrame,
    gc_vocab: Dict,
    gc_homologs: Dict,
    human_genes_in_vocab: Set[str],
    mouse_genes_in_vocab: Set[str],
    logger: logging.Logger,
    rgd_biotypes: Dict[str, str] = None  # ADD THIS
) -> Tuple[Dict[str, int], Dict[str, str], Dict[str, str], List[str], Dict[str, str], Counter, List[Dict]]:
    """
    Create rat token mappings with tri-species priority.
    
    TIER 1 - TRIPLET: rat→human AND rat→mouse, human↔mouse linked in GeneCompass
    TIER 2 - HUMAN-RAT: rat→human in vocab
    TIER 3 - MOUSE-RAT: rat→mouse in vocab (no human)
    TIER 4 - NEW: create new rat token
    
    Returns:
        rat_vocab, rat_to_human, rat_to_mouse, new_tokens, mapping_tier, tier_counts, mapping_details
    """
    
    rat_vocab = {}
    rat_to_human = {}
    rat_to_mouse = {}
    mapping_tier = {}
    new_tokens = []
    tier_counts = Counter()
    mapping_details = []  # For TSV output with quality info
    
    # Build lookups
    human_to_token = {k: v for k, v in gc_vocab.items() if k in human_genes_in_vocab}
    mouse_to_token = {k: v for k, v in gc_vocab.items() if k in mouse_genes_in_vocab}
    
    # Identify linked pairs in GeneCompass
    mouse_token_to_gene = {v: k for k, v in mouse_to_token.items()}
    human_token_to_gene = {v: k for k, v in human_to_token.items()}
    linked_mouse = {mouse_token_to_gene.get(t) for t in gc_homologs.keys() if t in mouse_token_to_gene}
    linked_human = {human_token_to_gene.get(t) for t in gc_homologs.values() if t in human_token_to_gene}
    linked_mouse.discard(None)
    linked_human.discard(None)
    
    logger.info(f"GeneCompass linked: {len(linked_human):,} human, {len(linked_mouse):,} mouse")
    
    # Build biotype lookup (BioMart primary, RGD fallback)
    gene_biotypes = dict(zip(gene_info['gene_id'].str.upper(), gene_info['biotype']))
    if rgd_biotypes:
        # Add RGD entries for genes not in BioMart
        for eid, btype in rgd_biotypes.items():
            if eid not in gene_biotypes:
                gene_biotypes[eid] = btype
        logger.info(f"Combined biotypes: {len(gene_biotypes):,} genes (BioMart + RGD fallback)")
    
    # Process orthologs
    corpus_base = {g.split('.')[0] for g in corpus_genes}
    processed = set()
    
    for _, row in orthologs.iterrows():
        rat_base = row['rat_gene_base']
        if rat_base in processed:
            continue
        processed.add(rat_base)
        
        human_id = row['human_gene_id'] if row['good_human'] else None
        mouse_id = row['mouse_gene_id'] if row['good_mouse'] else None
        
        # Get quality info
        human_type = row.get('human_orthology_type', '') if human_id else ''
        human_pct = row.get('human_perc_id', 0) if human_id else 0
        mouse_type = row.get('mouse_orthology_type', '') if mouse_id else ''
        mouse_pct = row.get('mouse_perc_id', 0) if mouse_id else 0
        
        human_in_vocab = human_id in human_to_token if human_id else False
        mouse_in_vocab = mouse_id in mouse_to_token if mouse_id else False
        human_linked = human_id in linked_human if human_id else False
        mouse_linked = mouse_id in linked_mouse if mouse_id else False
        
        detail = {
            'rat_gene_id': rat_base,
            'human_ortholog': human_id or '',
            'human_orthology_type': human_type,
            'human_perc_id': human_pct if human_pct else '',
            'mouse_ortholog': mouse_id or '',
            'mouse_orthology_type': mouse_type,
            'mouse_perc_id': mouse_pct if mouse_pct else '',
        }
        
        # TIER 1: Triplet
        if human_in_vocab and mouse_in_vocab and human_linked and mouse_linked:
            h_tok = human_to_token[human_id]
            m_tok = mouse_to_token[mouse_id]
            if gc_homologs.get(m_tok) == h_tok:
                rat_vocab[rat_base] = h_tok
                rat_to_human[rat_base] = human_id
                rat_to_mouse[rat_base] = mouse_id
                mapping_tier[rat_base] = 'triplet'
                tier_counts['triplet'] += 1
                detail['tier'] = 'triplet'
                detail['token_id'] = h_tok
                mapping_details.append(detail)
                continue
        
        # TIER 2: Human-rat
        if human_in_vocab:
            rat_vocab[rat_base] = human_to_token[human_id]
            rat_to_human[rat_base] = human_id
            if mouse_id:
                rat_to_mouse[rat_base] = mouse_id
            mapping_tier[rat_base] = 'human-rat'
            tier_counts['human-rat'] += 1
            detail['tier'] = 'human-rat'
            detail['token_id'] = human_to_token[human_id]
            mapping_details.append(detail)
            continue
        
        # TIER 3: Mouse-rat
        if mouse_in_vocab:
            rat_vocab[rat_base] = mouse_to_token[mouse_id]
            rat_to_mouse[rat_base] = mouse_id
            mapping_tier[rat_base] = 'mouse-rat'
            tier_counts['mouse-rat'] += 1
            detail['tier'] = 'mouse-rat'
            detail['token_id'] = mouse_to_token[mouse_id]
            mapping_details.append(detail)
            continue
        
        # Mark for potential TIER 4 (will check biotype later)
        mapping_tier[rat_base] = 'new-candidate'
        detail['tier'] = 'new-candidate'
        detail['token_id'] = ''
        mapping_details.append(detail)
    
    # Add corpus genes not in orthologs as new-candidates
    for gene in corpus_base:
        if gene not in processed:
            mapping_tier[gene] = 'new-candidate'
            mapping_details.append({
                'rat_gene_id': gene,
                'tier': 'new-candidate',
                'token_id': '',
                'human_ortholog': '',
                'human_orthology_type': '',
                'human_perc_id': '',
                'mouse_ortholog': '',
                'mouse_orthology_type': '',
                'mouse_perc_id': '',
            })
    
    # Create new tokens only for protein-coding/miRNA/lncRNA
    next_token = max(gc_vocab.values()) + 1 if gc_vocab else 1000
    
    for detail in mapping_details:
        rat_id = detail['rat_gene_id']
        if detail['tier'] == 'new-candidate':
            biotype = gene_biotypes.get(rat_id, 'unknown')
            if biotype in ['protein_coding', 'miRNA', 'lncRNA']:
                rat_vocab[rat_id] = next_token
                new_tokens.append(rat_id)
                mapping_tier[rat_id] = 'new'
                tier_counts['new'] += 1
                detail['tier'] = 'new'
                detail['token_id'] = next_token
                next_token += 1
            else:
                # Not a usable biotype - mark as excluded
                mapping_tier[rat_id] = 'excluded'
                detail['tier'] = 'excluded'
    
    # Filter mapping_details to only include mapped genes
    mapping_details = [d for d in mapping_details if d['tier'] not in ['new-candidate', 'excluded']]
    
    # Report
    total = len(rat_vocab)
    logger.info(f"\nMapping results ({total:,} genes):")
    logger.info(f"  TIER 1 Triplet:    {tier_counts['triplet']:>6,} ({tier_counts['triplet']/total*100:5.1f}%)")
    logger.info(f"  TIER 2 Human-rat:  {tier_counts['human-rat']:>6,} ({tier_counts['human-rat']/total*100:5.1f}%)")
    logger.info(f"  TIER 3 Mouse-rat:  {tier_counts['mouse-rat']:>6,} ({tier_counts['mouse-rat']/total*100:5.1f}%)")
    logger.info(f"  TIER 4 New:        {tier_counts['new']:>6,} ({tier_counts['new']/total*100:5.1f}%)")
    
    return rat_vocab, rat_to_human, rat_to_mouse, new_tokens, mapping_tier, tier_counts, mapping_details


# =============================================================================
# OUTPUT
# =============================================================================

def save_outputs(
    output_dir: Path,
    rat_vocab: Dict,
    rat_to_human: Dict,
    rat_to_mouse: Dict,
    new_tokens: List,
    mapping_tier: Dict,
    tier_counts: Counter,
    mapping_details: List[Dict],
    corpus_genes: Set,
    identity_thresholds: Dict[str, float],
    one2one_only: bool,
    logger: logging.Logger
) -> Dict:
    """Save all output files."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Pickle files
    with open(output_dir / "rat_tokens.pickle", 'wb') as f:
        pickle.dump(rat_vocab, f)
    
    with open(output_dir / "rat_to_human_mapping.pickle", 'wb') as f:
        pickle.dump(rat_to_human, f)
    
    with open(output_dir / "rat_to_mouse_mapping.pickle", 'wb') as f:
        pickle.dump(rat_to_mouse, f)
    
    with open(output_dir / "rat_mapping_tiers.pickle", 'wb') as f:
        pickle.dump(mapping_tier, f)
    
    # Text files
    with open(output_dir / "new_rat_tokens.txt", 'w') as f:
        f.write('\n'.join(sorted(new_tokens)))
    
    # Mapping table TSV with quality info
    df = pd.DataFrame(mapping_details)
    # Reorder columns
    cols = ['rat_gene_id', 'token_id', 'tier', 
            'human_ortholog', 'human_orthology_type', 'human_perc_id',
            'mouse_ortholog', 'mouse_orthology_type', 'mouse_perc_id']
    df = df[[c for c in cols if c in df.columns]]
    df = df.sort_values('rat_gene_id')
    df.to_csv(output_dir / "rat_token_mapping.tsv", sep='\t', index=False)
    
    # Compute orthology type distribution for stats
    orthology_dist = {
        'human': {},
        'mouse': {}
    }
    for detail in mapping_details:
        if detail.get('human_orthology_type'):
            otype = detail['human_orthology_type']
            orthology_dist['human'][otype] = orthology_dist['human'].get(otype, 0) + 1
        if detail.get('mouse_orthology_type') and detail['tier'] == 'mouse-rat':
            otype = detail['mouse_orthology_type']
            orthology_dist['mouse'][otype] = orthology_dist['mouse'].get(otype, 0) + 1
    
    # Compute identity stats
    human_ids = [d['human_perc_id'] for d in mapping_details if d.get('human_perc_id') and d['tier'] in ['triplet', 'human-rat']]
    mouse_ids = [d['mouse_perc_id'] for d in mapping_details if d.get('mouse_perc_id') and d['tier'] in ['triplet', 'mouse-rat']]
    
    identity_stats = {
        'human': {
            'mean': sum(human_ids) / len(human_ids) if human_ids else 0,
            'min': min(human_ids) if human_ids else 0,
            'max': max(human_ids) if human_ids else 0,
        },
        'mouse': {
            'mean': sum(mouse_ids) / len(mouse_ids) if mouse_ids else 0,
            'min': min(mouse_ids) if mouse_ids else 0,
            'max': max(mouse_ids) if mouse_ids else 0,
        }
    }
    
    # Statistics
    stats = {
        'timestamp': datetime.now().isoformat(),
        'corpus_genes': len({g.split('.')[0] for g in corpus_genes}),
        'total_mapped': len(rat_vocab),
        'filter_settings': {
            'one2one_only': one2one_only,
            'identity_thresholds': identity_thresholds,
        },
        'tiers': {
            'triplet': tier_counts.get('triplet', 0),
            'human_rat': tier_counts.get('human-rat', 0),
            'mouse_rat': tier_counts.get('mouse-rat', 0),
            'new': tier_counts.get('new', 0)
        },
        'orthology_type_distribution': orthology_dist,
        'identity_stats': identity_stats,
        'new_tokens': len(new_tokens)
    }
    
    with open(output_dir / "mapping_statistics.json", 'w') as f:
        json.dump(stats, f, indent=2)
    
    logger.info(f"Saved outputs to {output_dir}")
    return stats


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Build rat ortholog mappings for GeneCompass",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default (type-specific thresholds)
  python build_ortholog_mapping.py -v
  
  # Strict: one2one orthologs only
  python build_ortholog_mapping.py --one2one-only -v
  
  # Custom thresholds
  python build_ortholog_mapping.py --min-identity-one2one 60 --min-identity-one2many 80 -v
        """
    )
    parser.add_argument(
        "-o", "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help="Output directory"
    )
    parser.add_argument(
        "--corpus-genes", type=Path, default=DEFAULT_CORPUS_GENES,
        help="Corpus rat Ensembl IDs"
    )
    parser.add_argument(
        "--biomart-dir", type=Path, default=DEFAULT_BIOMART_DIR,
        help="BioMart reference data directory"
    )
    parser.add_argument(
        "--genecompass-vocab", type=Path, default=DEFAULT_GENECOMPASS_VOCAB,
        help="GeneCompass vocabulary"
    )
    
    # Filtering options
    parser.add_argument(
        "--one2one-only", action="store_true",
        help="Only use one2one orthologs (strictest mode)"
    )
    parser.add_argument(
        "--min-identity-one2one", type=float, default=DEFAULT_MIN_IDENTITY['ortholog_one2one'],
        help=f"Min %% identity for one2one orthologs (default: {DEFAULT_MIN_IDENTITY['ortholog_one2one']})"
    )
    parser.add_argument(
        "--min-identity-one2many", type=float, default=DEFAULT_MIN_IDENTITY['ortholog_one2many'],
        help=f"Min %% identity for one2many orthologs (default: {DEFAULT_MIN_IDENTITY['ortholog_one2many']})"
    )
    parser.add_argument(
        "--min-identity-many2many", type=float, default=DEFAULT_MIN_IDENTITY['ortholog_many2many'],
        help=f"Min %% identity for many2many orthologs (default: {DEFAULT_MIN_IDENTITY['ortholog_many2many']})"
    )
    
    parser.add_argument(
        "-v", "--verbose", action="store_true"
    )
    
    args = parser.parse_args()
    logger = setup_logging(args.verbose)
    
    # Build identity thresholds dict
    identity_thresholds = {
        'ortholog_one2one': args.min_identity_one2one,
        'ortholog_one2many': args.min_identity_one2many,
        'ortholog_many2many': args.min_identity_many2many,
    }
    
    logger.info("=" * 60)
    logger.info("BUILDING RAT ORTHOLOG MAPPINGS")
    logger.info("=" * 60)
    logger.info(f"Filter mode: {'one2one only' if args.one2one_only else 'type-specific thresholds'}")
    logger.info(f"Identity thresholds: {identity_thresholds}")
    
    # Load data
    logger.info("\n[1] Loading corpus genes...")
    corpus = load_corpus_genes(args.corpus_genes, logger)
    if not corpus:
        return 1
    
    logger.info("\n[2] Loading BioMart reference data...")
    try:
        human_df, mouse_df, gene_df = load_biomart_data(args.biomart_dir, logger)
    except FileNotFoundError:
        return 1
    
    
    logger.info("\n[2b] Loading RGD biotypes (fallback)...")
    try:
        rgd_path = args.biomart_dir / "GENES_RAT.txt"
        rgd_biotypes = load_rgd_biotypes(rgd_path, logger)
    except FileNotFoundError:
        return 1
    
    
    logger.info("\n[3] Loading GeneCompass vocabulary...")
    gc_vocab, gc_homologs, human_genes, mouse_genes = load_genecompass_data(
        args.genecompass_vocab, logger
    )
    
    # Process
    logger.info(f"\n[4] Merging and filtering orthologs...")
    merged = merge_and_filter_orthologs(
        human_df, mouse_df, gene_df, corpus, identity_thresholds, args.one2one_only, logger
    )
    
    logger.info("\n[5] Resolving best ortholog per gene...")
    resolved = resolve_best_per_gene(merged, logger)
    
    logger.info("\n[6] Creating token mappings...")
    rat_vocab, rat_to_human, rat_to_mouse, new_tokens, mapping_tier, tier_counts, mapping_details = create_token_mapping(
        resolved, corpus, gene_df, gc_vocab, gc_homologs, human_genes, mouse_genes, logger,
        rgd_biotypes=rgd_biotypes
    )
    
    # Save
    logger.info(f"\n[7] Saving outputs...")
    stats = save_outputs(
        args.output_dir, rat_vocab, rat_to_human, rat_to_mouse, 
        new_tokens, mapping_tier, tier_counts, mapping_details, corpus,
        identity_thresholds, args.one2one_only, logger
    )
    
    # Summary
    total = stats['total_mapped']
    print("\n" + "=" * 60)
    print("MAPPING COMPLETE")
    print("=" * 60)
    print(f"Filter: {'one2one only' if args.one2one_only else 'type-specific'}")
    print(f"Thresholds: {identity_thresholds}")
    print(f"\nCorpus genes:  {stats['corpus_genes']:,}")
    print(f"Mapped:        {total:,} ({total/stats['corpus_genes']*100:.1f}%)")
    print(f"\nTier breakdown:")
    for tier, count in stats['tiers'].items():
        pct = count / total * 100 if total else 0
        print(f"  {tier:12} {count:>6,} ({pct:5.1f}%)")
    print(f"\nOrthology types used:")
    for species, types in stats['orthology_type_distribution'].items():
        if types:
            print(f"  {species}: {types}")
    print(f"\nIdentity stats:")
    for species, istats in stats['identity_stats'].items():
        if istats['mean'] > 0:
            print(f"  {species}: mean={istats['mean']:.1f}%, range=[{istats['min']:.1f}%, {istats['max']:.1f}%]")
    print(f"\nNew tokens:    {stats['new_tokens']:,}")
    print(f"Output:        {args.output_dir}")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    exit(main())