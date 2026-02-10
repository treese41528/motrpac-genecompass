#!/usr/bin/env python3
"""
combine_data_sources.py - Merge multiple data sources into unified studies catalog

OVERVIEW:
This script combines three complementary data sources:
1. master_catalog.json - Basic GEO/ArrayExpress metadata, file paths
2. matrix_analysis.json - Gene/cell counts, formats, data quality
3. llm_test.json - LLM-validated metadata, topics, MoTrPAC utility

OUTPUT:
- unified_studies.json - Combined view with all fields
- unified_studies_summary.csv - Flat summary for quick analysis

DATA SOURCE DETAILS:
===================

1. master_catalog.json (from extract_metadata.py)
   - accession, source, data_type
   - title, summary
   - organism[] (list of raw organism strings)
   - tissues[] (list of raw tissue strings)
   - sample_count, platforms, technologies
   - pubmed_ids[], submission_date
   - file_count, total_size_bytes, file_formats{}
   - has_processed_data, has_count_matrix, has_metadata
   - usability_score

2. matrix_analysis.json (from analyze_matrices_fast.py)
   - accession
   - n_genes, n_cells, n_samples
   - formats{}, gene_ids_sample[], gene_id_type
   - data_modality (mrna/mirna/circrna/lncrna/microarray/unknown)
   - is_multimodal, feature_types{}
   - is_unfiltered_10x, confidence, data_type
   - error (if any)

3. llm_test.json (from llm_study_analyzer_enhanced.py)
   - accession
   - metadata_validation{extracted_organism_correct, actual_organism, extracted_tissues_correct, actual_tissues[]}
   - study_overview{title, summary, primary_topic, topic_category, keywords[]}
   - study_type{data_type, is_single_cell, is_time_series, is_disease_study, is_treatment_study}
   - organism{species, strain, sex, age_value, age_unit, life_stage}
   - tissues[{name, category, motrpac_match}]
   - cell_types[]
   - disease_condition{has_disease_model, disease_name, disease_type, induction_method}
   - treatments{has_drug, drug_names[], has_diet, diet_type, has_exercise, exercise_type}
   - experimental_design{groups[], time_points[], total_samples, samples_per_group}
   - utility_for_motrpac{is_rat, genecompass_useful, deconvolution_useful, grn_useful, motrpac_tissues[]}
   - files{has_count_matrix, data_quality}
   - summary{what_is_this_study, how_we_can_use_it[]}
   - _meta{analyzed_at, model, input_tokens, output_tokens, success}

UNIFIED SCHEMA:
===============
{
  "accession": str,
  "source": "geo" | "arrayexpress",
  
  # === FROM CATALOG ===
  "catalog": {
    "data_type": str,
    "title": str,
    "summary": str,
    "raw_organisms": [str],
    "raw_tissues": [str],
    "sample_count": int,
    "platforms": [str],
    "technologies": [str],
    "pubmed_ids": [str],
    "submission_date": str,
    "file_count": int,
    "total_size_bytes": int,
    "file_formats": {},
    "has_processed_data": bool,
    "has_count_matrix": bool,
    "usability_score": int
  },
  
  # === FROM MATRIX ANALYSIS ===
  "matrix": {
    "n_genes": int,
    "n_cells": int,
    "n_samples": int,
    "formats": {},
    "gene_id_type": str,
    "data_modality": str,
    "confidence": str,
    "is_unfiltered_10x": bool,
    "gene_ids_sample": [str],
    "error": str or null
  },
  
  # === FROM LLM ANALYSIS ===
  "llm": {
    "success": bool,
    "model": str,
    "analyzed_at": str,
    
    "validated_organism": {
      "species": str,
      "strain": str,
      "sex": str,
      "age_value": number,
      "age_unit": str,
      "life_stage": str
    },
    
    "validated_tissues": [{
      "name": str,
      "category": str,
      "motrpac_match": str
    }],
    
    "topic": {
      "primary": str,
      "category": str,
      "keywords": [str]
    },
    
    "study_type": {
      "is_single_cell": bool,
      "is_time_series": bool,
      "is_disease_study": bool,
      "is_treatment_study": bool
    },
    
    "disease": {
      "has_model": bool,
      "name": str,
      "type": str,
      "induction_method": str
    },
    
    "treatments": {
      "has_drug": bool,
      "drug_names": [str],
      "has_diet": bool,
      "diet_type": str,
      "has_exercise": bool,
      "exercise_type": str
    },
    
    "design": {
      "groups": [str],
      "time_points": [str],
      "total_samples": int,
      "samples_per_group": int
    },
    
    "motrpac_utility": {
      "is_rat": bool,
      "genecompass_useful": bool,
      "deconvolution_useful": bool,
      "grn_useful": bool,
      "motrpac_tissues": [str]
    },
    
    "validation": {
      "organism_correct": bool,
      "tissues_correct": bool,
      "actual_organism": str,
      "actual_tissues": [str]
    },
    
    "summary": str,
    "use_cases": [str]
  },
  
  # === DERIVED QUALITY SCORES ===
  "quality": {
    "data_completeness": float,  # 0-1 based on available fields
    "has_matrix": bool,
    "gene_count_valid": bool,
    "llm_analyzed": bool,
    "validation_passed": bool
  }
}

Usage:
    python combine_data_sources.py --config config.yaml
    python combine_data_sources.py --config config.yaml --output unified.json
    python combine_data_sources.py --config config.yaml --csv-summary
"""

import json
import csv
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from collections import Counter

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_nested(d: dict, *keys, default=None):
    """Safely get nested dictionary value."""
    for key in keys:
        if isinstance(d, dict):
            d = d.get(key, default)
        else:
            return default
    return d


def format_size(size_bytes: int) -> str:
    """Format bytes to human readable."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} PB"


# =============================================================================
# DATA LOADING
# =============================================================================

def load_json(path: Path) -> Optional[dict]:
    """Load JSON file, return None if not found."""
    if not path.exists():
        logger.warning(f"File not found: {path}")
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading {path}: {e}")
        return None


def load_catalog(catalog_dir: Path, filename: str = 'master_catalog.json') -> Dict[str, dict]:
    """Load catalog JSON and index by accession."""
    path = catalog_dir / filename
    data = load_json(path)
    if not data:
        return {}
    
    studies = data.get('studies', [])
    logger.info(f"Loaded {len(studies)} studies from {filename}")
    return {s['accession']: s for s in studies}


def load_matrix_analysis(catalog_dir: Path, filename: str = 'matrix_analysis.json') -> Dict[str, dict]:
    """Load matrix analysis JSON and index by accession."""
    path = catalog_dir / filename
    data = load_json(path)
    if not data:
        return {}
    
    studies = data.get('study_stats', [])
    logger.info(f"Loaded {len(studies)} studies from {filename}")
    return {s['accession']: s for s in studies}


def load_llm_analysis(catalog_dir: Path, filename: str = 'llm_test.json') -> Dict[str, dict]:
    """Load LLM analysis JSON and index by accession."""
    path = catalog_dir / filename
    data = load_json(path)
    if not data:
        return {}
    
    analyses = data.get('analyses', [])
    logger.info(f"Loaded {len(analyses)} studies from {filename}")
    return {s['accession']: s for s in analyses}


# =============================================================================
# TRANSFORMATION FUNCTIONS
# =============================================================================

def transform_catalog(catalog_entry: dict) -> dict:
    """Transform catalog entry to unified format."""
    return {
        'data_type': catalog_entry.get('data_type'),
        'title': catalog_entry.get('title'),
        'summary': catalog_entry.get('summary'),
        'raw_organisms': catalog_entry.get('organism', []),
        'raw_tissues': catalog_entry.get('tissues', []),
        'sample_count': catalog_entry.get('sample_count'),
        'platforms': catalog_entry.get('platforms', []),
        'technologies': catalog_entry.get('technologies', []),
        'pubmed_ids': catalog_entry.get('pubmed_ids', []),
        'submission_date': catalog_entry.get('submission_date'),
        'file_count': catalog_entry.get('file_count'),
        'files': catalog_entry.get('files', []),
        'total_size_bytes': catalog_entry.get('total_size_bytes'),
        'total_size_human': format_size(catalog_entry.get('total_size_bytes', 0)),
        'file_formats': catalog_entry.get('file_formats', {}),
        'has_processed_data': catalog_entry.get('has_processed_data', False),
        'has_count_matrix': catalog_entry.get('has_count_matrix', False),
        'usability_score': catalog_entry.get('usability_score', 0),
    }


def transform_matrix(matrix_entry: dict) -> dict:
    """Transform matrix analysis entry to unified format."""
    return {
        'n_genes': matrix_entry.get('n_genes'),
        'n_cells': matrix_entry.get('n_cells'),
        'n_samples': matrix_entry.get('n_samples'),
        'formats': matrix_entry.get('formats', {}),
        'gene_id_type': matrix_entry.get('gene_id_type'),
        'data_modality': matrix_entry.get('data_modality'),
        'confidence': matrix_entry.get('confidence'),
        'is_unfiltered_10x': matrix_entry.get('is_unfiltered_10x', False),
        'gene_ids_sample': matrix_entry.get('gene_ids_sample', [])[:10],  # Limit sample
        'error': matrix_entry.get('error'),
    }


def safe_string(val, default='') -> str:
    """Convert value to string, joining lists if needed."""
    if val is None:
        return default
    if isinstance(val, list):
        return ', '.join(str(v) for v in val if v) if val else default
    return str(val)


# =============================================================================
# NORMALIZATION FUNCTIONS
# =============================================================================

def normalize_strain(strain: str) -> str:
    """Normalize strain names for consistent counting."""
    if not strain:
        return 'not_specified'
    
    s = strain.strip().lower()
    
    # Map common variations
    strain_map = {
        'sprague dawley': 'Sprague-Dawley',
        'sprague-dawley': 'Sprague-Dawley',
        'sd': 'Sprague-Dawley',
        'wistar': 'Wistar',
        'fischer 344': 'Fischer 344',
        'fischer344': 'Fischer 344',
        'f344': 'Fischer 344',
        'lewis': 'Lewis',
        'brown norway': 'Brown Norway',
        'long-evans': 'Long-Evans',
        'long evans': 'Long-Evans',
        'not_specified': 'not_specified',
        'not specified': 'not_specified',
        'unknown': 'not_specified',
        'not_applicable': 'not_applicable',
        'not applicable': 'not_applicable',
        'n/a': 'not_applicable',
    }
    
    # Check for exact match first
    if s in strain_map:
        return strain_map[s]
    
    # Check for partial matches
    for pattern, normalized in strain_map.items():
        if pattern in s:
            return normalized
    
    # Return original with proper casing if no match
    return strain


def normalize_sex(sex: str) -> str:
    """Normalize sex values for consistent counting."""
    if not sex:
        return 'not_specified'
    
    s = sex.strip().lower()
    
    sex_map = {
        'male': 'male',
        'm': 'male',
        'female': 'female',
        'f': 'female',
        'both': 'both',
        'mixed': 'mixed',
        'not_specified': 'not_specified',
        'not specified': 'not_specified',
        'unknown': 'not_specified',
        'not_applicable': 'not_applicable',
        'not applicable': 'not_applicable',
        'not_applicable_cell_line': 'not_applicable',
        'n/a': 'not_applicable',
    }
    
    if s in sex_map:
        return sex_map[s]
    
    # Handle complex multi-species descriptions
    if 'male' in s and 'female' in s:
        return 'mixed'
    if 'male' in s:
        return 'male'
    if 'female' in s:
        return 'female'
    
    return 'not_specified'


def normalize_motrpac_tissue(tissue: str) -> List[str]:
    """Normalize and split MoTrPAC tissue names."""
    if not tissue:
        return []
    
    # Tissue name normalization map
    tissue_map = {
        'small_intestine': 'small intestine',
        'wat-sc': 'WAT-SC',
        'wat_sc': 'WAT-SC',
        'white adipose': 'WAT-SC',
        'bat': 'BAT',
        'brown adipose': 'BAT',
        'blood rna': 'blood RNA',
        'blood_rna': 'blood RNA',
    }
    
    # Split compound entries (e.g., "hippocampus, hypothalamus, cortex")
    tissues = []
    for t in tissue.split(','):
        t = t.strip().lower()
        if t:
            # Apply normalization
            normalized = tissue_map.get(t, t)
            # Capitalize properly
            if normalized not in ['WAT-SC', 'BAT', 'blood RNA']:
                normalized = normalized.title() if len(normalized) > 3 else normalized
            tissues.append(normalized)
    
    return tissues


def is_mouse_strain(strain: str) -> bool:
    """Check if strain is a mouse strain (data quality flag)."""
    if not strain:
        return False
    s = strain.lower()
    mouse_indicators = ['c57bl', 'balb/c', 'balbc', 'dba', 'fvb', '129s', 'nod', 'icr', 'cd1', 'cd-1']
    return any(m in s for m in mouse_indicators)


def transform_llm(llm_entry: dict) -> dict:
    """Transform LLM analysis entry to unified format."""
    meta = llm_entry.get('_meta', {})
    
    # Check success
    success = meta.get('success', False)
    
    if not success:
        return {
            'success': False,
            'model': meta.get('model'),
            'analyzed_at': meta.get('analyzed_at'),
            'error': meta.get('error'),
        }
    
    # Handle organism fields that might be lists (multi-strain studies)
    organism = llm_entry.get('organism', {})
    if isinstance(organism, dict):
        species = safe_string(organism.get('species'))
        strain = safe_string(organism.get('strain'))
        sex = safe_string(organism.get('sex'))
        age_value = organism.get('age_value')
        age_unit = organism.get('age_unit')
        life_stage = organism.get('life_stage')
    else:
        species = strain = sex = ''
        age_value = age_unit = life_stage = None
    
    # Normalize strain and sex for consistent aggregation
    strain_normalized = normalize_strain(strain)
    sex_normalized = normalize_sex(sex)
    
    # Check for potential mouse contamination in rat dataset
    possible_mouse = is_mouse_strain(strain)
    
    # Get and normalize MoTrPAC tissue matches
    raw_motrpac_tissues = get_nested(llm_entry, 'utility_for_motrpac', 'motrpac_tissues', default=[])
    normalized_motrpac_tissues = []
    for t in raw_motrpac_tissues:
        if t:
            normalized_motrpac_tissues.extend(normalize_motrpac_tissue(t))
    
    return {
        'success': True,
        'model': meta.get('model'),
        'analyzed_at': meta.get('analyzed_at'),
        
        'validated_organism': {
            'species': species,
            'strain': strain,
            'strain_normalized': strain_normalized,
            'sex': sex,
            'sex_normalized': sex_normalized,
            'age_value': age_value,
            'age_unit': age_unit,
            'life_stage': life_stage,
            'possible_mouse_contamination': possible_mouse,
        },
        
        # Normalize tissue structures (some have extra keys like 'organism', 'note', 'sample_count')
        'validated_tissues': [
            {
                'name': t.get('name') if isinstance(t, dict) else str(t),
                'category': t.get('category') if isinstance(t, dict) else None,
                'motrpac_match': t.get('motrpac_match') if isinstance(t, dict) else None,
            }
            for t in llm_entry.get('tissues', [])
        ],
        
        'topic': {
            'primary': get_nested(llm_entry, 'study_overview', 'primary_topic'),
            'category': get_nested(llm_entry, 'study_overview', 'topic_category'),
            'keywords': get_nested(llm_entry, 'study_overview', 'keywords', default=[]),
        },
        
        'study_type': {
            'is_single_cell': get_nested(llm_entry, 'study_type', 'is_single_cell', default=False),
            'is_time_series': get_nested(llm_entry, 'study_type', 'is_time_series', default=False),
            'is_disease_study': get_nested(llm_entry, 'study_type', 'is_disease_study', default=False),
            'is_treatment_study': get_nested(llm_entry, 'study_type', 'is_treatment_study', default=False),
        },
        
        'disease': {
            'has_model': get_nested(llm_entry, 'disease_condition', 'has_disease_model', default=False),
            'name': get_nested(llm_entry, 'disease_condition', 'disease_name'),
            'type': get_nested(llm_entry, 'disease_condition', 'disease_type'),
            'induction_method': get_nested(llm_entry, 'disease_condition', 'induction_method'),
        },
        
        'treatments': {
            'has_drug': get_nested(llm_entry, 'treatments', 'has_drug', default=False),
            'drug_names': get_nested(llm_entry, 'treatments', 'drug_names', default=[]),
            'has_diet': get_nested(llm_entry, 'treatments', 'has_diet', default=False),
            'diet_type': get_nested(llm_entry, 'treatments', 'diet_type'),
            'has_exercise': get_nested(llm_entry, 'treatments', 'has_exercise', default=False),
            'exercise_type': get_nested(llm_entry, 'treatments', 'exercise_type'),
        },
        
        'design': {
            'groups': get_nested(llm_entry, 'experimental_design', 'groups', default=[]),
            'time_points': get_nested(llm_entry, 'experimental_design', 'time_points', default=[]),
            'total_samples': get_nested(llm_entry, 'experimental_design', 'total_samples'),
            'samples_per_group': get_nested(llm_entry, 'experimental_design', 'samples_per_group'),
        },
        
        'motrpac_utility': {
            'is_rat': get_nested(llm_entry, 'utility_for_motrpac', 'is_rat', default=False),
            'genecompass_useful': get_nested(llm_entry, 'utility_for_motrpac', 'genecompass_useful', default=False),
            'deconvolution_useful': get_nested(llm_entry, 'utility_for_motrpac', 'deconvolution_useful', default=False),
            'grn_useful': get_nested(llm_entry, 'utility_for_motrpac', 'grn_useful', default=False),
            'motrpac_tissues': normalized_motrpac_tissues,  # Normalized and split
            'motrpac_tissues_raw': raw_motrpac_tissues,  # Original for reference
        },
        
        'validation': {
            'organism_correct': get_nested(llm_entry, 'metadata_validation', 'extracted_organism_correct', default=True),
            'tissues_correct': get_nested(llm_entry, 'metadata_validation', 'extracted_tissues_correct', default=True),
            'actual_organism': get_nested(llm_entry, 'metadata_validation', 'actual_organism'),
            'actual_tissues': get_nested(llm_entry, 'metadata_validation', 'actual_tissues', default=[]),
        },
        
        'summary': get_nested(llm_entry, 'summary', 'what_is_this_study'),
        'use_cases': get_nested(llm_entry, 'summary', 'how_we_can_use_it', default=[]),
    }


def compute_quality_score(unified: dict) -> dict:
    """Compute quality metrics for a unified study entry."""
    catalog = unified.get('catalog', {})
    matrix = unified.get('matrix', {})
    llm = unified.get('llm', {})
    
    # Check data completeness
    has_catalog = bool(catalog.get('title'))
    has_matrix = matrix.get('n_genes') is not None
    has_llm = llm.get('success', False)
    
    # Gene count validity
    n_genes = matrix.get('n_genes')
    gene_count_valid = n_genes is not None and 10000 <= n_genes <= 100000
    
    # Validation passed
    validation_passed = True
    if has_llm:
        validation_passed = (
            llm.get('validation', {}).get('organism_correct', True) and
            llm.get('validation', {}).get('tissues_correct', True)
        )
    
    # Compute completeness score (0-1)
    completeness_factors = [
        has_catalog,
        has_matrix,
        has_llm,
        bool(catalog.get('title')),
        bool(catalog.get('summary')),
        bool(matrix.get('n_genes')),
        bool(llm.get('topic', {}).get('primary')),
    ]
    data_completeness = sum(completeness_factors) / len(completeness_factors)
    
    return {
        'data_completeness': round(data_completeness, 2),
        'has_catalog': has_catalog,
        'has_matrix': has_matrix,
        'has_llm': has_llm,
        'gene_count_valid': gene_count_valid,
        'validation_passed': validation_passed,
    }


# =============================================================================
# MAIN COMBINATION LOGIC
# =============================================================================

def combine_studies(
    catalog_by_acc: Dict[str, dict],
    matrix_by_acc: Dict[str, dict],
    llm_by_acc: Dict[str, dict]
) -> List[dict]:
    """Combine all data sources into unified study entries."""
    
    # Get all unique accessions
    all_accessions = set()
    all_accessions.update(catalog_by_acc.keys())
    all_accessions.update(matrix_by_acc.keys())
    all_accessions.update(llm_by_acc.keys())
    
    logger.info(f"Total unique accessions: {len(all_accessions)}")
    
    unified_studies = []
    
    for acc in sorted(all_accessions):
        catalog_entry = catalog_by_acc.get(acc, {})
        matrix_entry = matrix_by_acc.get(acc, {})
        llm_entry = llm_by_acc.get(acc, {})
        
        unified = {
            'accession': acc,
            'source': catalog_entry.get('source', 'geo'),
        }
        
        # Transform and add each source
        if catalog_entry:
            unified['catalog'] = transform_catalog(catalog_entry)
        else:
            unified['catalog'] = {}
        
        if matrix_entry:
            unified['matrix'] = transform_matrix(matrix_entry)
        else:
            unified['matrix'] = {}
        
        if llm_entry:
            unified['llm'] = transform_llm(llm_entry)
        else:
            unified['llm'] = {'success': False}
        
        # Compute quality scores
        unified['quality'] = compute_quality_score(unified)
        
        unified_studies.append(unified)
    
    return unified_studies


def compute_aggregate_stats(unified_studies: List[dict]) -> dict:
    """Compute aggregate statistics from unified studies."""
    
    stats = {
        'total_studies': len(unified_studies),
        'by_source': Counter(),
        'coverage': {
            'has_catalog': 0,
            'has_matrix': 0,
            'has_llm': 0,
            'has_all_three': 0,
            'catalog_llm_only': 0,  # Studies missing matrix analysis
            'multi_strain': 0,  # Studies with multiple strains
        },
        'matrix_stats': {
            'total_genes': 0,
            'total_cells': 0,
            'total_samples': 0,
            'gene_count_valid': 0,
        },
        'llm_stats': {
            'successful': 0,
            'genecompass_useful': 0,
            'deconvolution_useful': 0,
            'grn_useful': 0,
            'exercise_studies': 0,
            'disease_models': 0,
            'single_cell': 0,
            'time_series': 0,
            'validation_issues': 0,
            'possible_mouse_contamination': 0,  # Mouse strains in rat dataset
        },
        'by_topic_category': Counter(),
        'by_motrpac_tissue': Counter(),  # Uses normalized tissue names
        'by_disease_type': Counter(),
        'by_strain': Counter(),  # Uses normalized strain names
        'by_strain_raw': Counter(),  # Original values for comparison
        'by_sex': Counter(),  # Uses normalized sex values
    }
    
    for study in unified_studies:
        # Source counts
        stats['by_source'][study.get('source', 'unknown')] += 1
        
        quality = study.get('quality', {})
        catalog = study.get('catalog', {})
        matrix = study.get('matrix', {})
        llm = study.get('llm', {})
        
        # Coverage
        has_catalog = quality.get('has_catalog')
        has_matrix = quality.get('has_matrix')
        has_llm = quality.get('has_llm')
        
        if has_catalog:
            stats['coverage']['has_catalog'] += 1
        if has_matrix:
            stats['coverage']['has_matrix'] += 1
        if has_llm:
            stats['coverage']['has_llm'] += 1
        if has_catalog and has_matrix and has_llm:
            stats['coverage']['has_all_three'] += 1
        if has_catalog and has_llm and not has_matrix:
            stats['coverage']['catalog_llm_only'] += 1
        
        # Check for multi-strain studies (strain contains comma from list join)
        strain = llm.get('validated_organism', {}).get('strain', '')
        if strain and ',' in strain:
            stats['coverage']['multi_strain'] += 1
        
        # Matrix stats
        n_genes = matrix.get('n_genes') or 0
        n_cells = matrix.get('n_cells') or 0
        n_samples = matrix.get('n_samples') or 0
        
        stats['matrix_stats']['total_genes'] = max(stats['matrix_stats']['total_genes'], n_genes)
        stats['matrix_stats']['total_cells'] += n_cells
        stats['matrix_stats']['total_samples'] += n_samples
        
        if quality.get('gene_count_valid'):
            stats['matrix_stats']['gene_count_valid'] += 1
        
        # LLM stats
        if llm.get('success'):
            stats['llm_stats']['successful'] += 1
            
            utility = llm.get('motrpac_utility', {})
            if utility.get('genecompass_useful'):
                stats['llm_stats']['genecompass_useful'] += 1
            if utility.get('deconvolution_useful'):
                stats['llm_stats']['deconvolution_useful'] += 1
            if utility.get('grn_useful'):
                stats['llm_stats']['grn_useful'] += 1
            
            treatments = llm.get('treatments', {})
            if treatments.get('has_exercise'):
                stats['llm_stats']['exercise_studies'] += 1
            
            disease = llm.get('disease', {})
            if disease.get('has_model'):
                stats['llm_stats']['disease_models'] += 1
                dtype = disease.get('type')
                if dtype:
                    stats['by_disease_type'][dtype] += 1
            
            study_type = llm.get('study_type', {})
            if study_type.get('is_single_cell'):
                stats['llm_stats']['single_cell'] += 1
            if study_type.get('is_time_series'):
                stats['llm_stats']['time_series'] += 1
            
            validation = llm.get('validation', {})
            if not validation.get('organism_correct', True) or not validation.get('tissues_correct', True):
                stats['llm_stats']['validation_issues'] += 1
            
            # Check for possible mouse contamination
            organism = llm.get('validated_organism', {})
            if organism.get('possible_mouse_contamination'):
                stats['llm_stats']['possible_mouse_contamination'] += 1
            
            # Topic category
            topic_cat = llm.get('topic', {}).get('category')
            if topic_cat:
                stats['by_topic_category'][topic_cat] += 1
            
            # MoTrPAC tissues (already normalized in transform_llm)
            for tissue in utility.get('motrpac_tissues', []):
                if tissue:
                    stats['by_motrpac_tissue'][tissue] += 1
            
            # Strain - use normalized version for stats
            strain_normalized = organism.get('strain_normalized')
            strain_raw = organism.get('strain')
            if strain_normalized and strain_normalized not in ['not_specified', 'not_applicable']:
                stats['by_strain'][strain_normalized] += 1
            if strain_raw and strain_raw not in ['unknown', 'not_specified', 'not_applicable', None, '']:
                stats['by_strain_raw'][strain_raw] += 1
            
            # Sex - use normalized version
            sex_normalized = organism.get('sex_normalized')
            if sex_normalized:
                stats['by_sex'][sex_normalized] += 1
    
    # Convert counters to dicts for JSON serialization
    stats['by_source'] = dict(stats['by_source'])
    stats['by_topic_category'] = dict(sorted(stats['by_topic_category'].items(), key=lambda x: -x[1]))
    stats['by_motrpac_tissue'] = dict(sorted(stats['by_motrpac_tissue'].items(), key=lambda x: -x[1]))
    stats['by_disease_type'] = dict(sorted(stats['by_disease_type'].items(), key=lambda x: -x[1]))
    stats['by_strain'] = dict(sorted(stats['by_strain'].items(), key=lambda x: -x[1]))
    stats['by_strain_raw'] = dict(sorted(stats['by_strain_raw'].items(), key=lambda x: -x[1]))
    stats['by_sex'] = dict(sorted(stats['by_sex'].items(), key=lambda x: -x[1]))
    
    return stats


def generate_csv_summary(unified_studies: List[dict], output_path: Path):
    """Generate flat CSV summary for quick analysis."""
    
    fieldnames = [
        'accession', 'source',
        # Quality
        'has_catalog', 'has_matrix', 'has_llm', 'data_completeness',
        # Catalog
        'title', 'data_type', 'sample_count', 'file_count', 'total_size_human',
        # Matrix
        'n_genes', 'n_cells', 'n_samples', 'gene_id_type', 'data_modality',
        # LLM - Organism (raw and normalized)
        'primary_topic', 'topic_category', 'species', 
        'strain', 'strain_normalized', 
        'sex', 'sex_normalized',
        'possible_mouse_contamination',
        # Study type
        'is_single_cell', 'is_time_series', 'has_disease_model', 'disease_type',
        'has_exercise', 'exercise_type',
        # Utility
        'genecompass_useful', 'deconvolution_useful', 'grn_useful',
        'motrpac_tissues',
        'validation_passed', 'summary',
    ]
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for study in unified_studies:
            catalog = study.get('catalog', {})
            matrix = study.get('matrix', {})
            llm = study.get('llm', {})
            quality = study.get('quality', {})
            organism = llm.get('validated_organism', {})
            
            row = {
                'accession': study['accession'],
                'source': study.get('source'),
                # Quality
                'has_catalog': quality.get('has_catalog'),
                'has_matrix': quality.get('has_matrix'),
                'has_llm': quality.get('has_llm'),
                'data_completeness': quality.get('data_completeness'),
                # Catalog
                'title': (catalog.get('title') or '')[:100],
                'data_type': catalog.get('data_type'),
                'sample_count': catalog.get('sample_count'),
                'file_count': catalog.get('file_count'),
                'total_size_human': catalog.get('total_size_human'),
                # Matrix
                'n_genes': matrix.get('n_genes'),
                'n_cells': matrix.get('n_cells'),
                'n_samples': matrix.get('n_samples'),
                'gene_id_type': matrix.get('gene_id_type'),
                'data_modality': matrix.get('data_modality'),
                # LLM - Organism
                'primary_topic': llm.get('topic', {}).get('primary'),
                'topic_category': llm.get('topic', {}).get('category'),
                'species': organism.get('species'),
                'strain': organism.get('strain'),
                'strain_normalized': organism.get('strain_normalized'),
                'sex': organism.get('sex'),
                'sex_normalized': organism.get('sex_normalized'),
                'possible_mouse_contamination': organism.get('possible_mouse_contamination'),
                # Study type
                'is_single_cell': llm.get('study_type', {}).get('is_single_cell'),
                'is_time_series': llm.get('study_type', {}).get('is_time_series'),
                'has_disease_model': llm.get('disease', {}).get('has_model'),
                'disease_type': llm.get('disease', {}).get('type'),
                'has_exercise': llm.get('treatments', {}).get('has_exercise'),
                'exercise_type': llm.get('treatments', {}).get('exercise_type'),
                'genecompass_useful': llm.get('motrpac_utility', {}).get('genecompass_useful'),
                'deconvolution_useful': llm.get('motrpac_utility', {}).get('deconvolution_useful'),
                'grn_useful': llm.get('motrpac_utility', {}).get('grn_useful'),
                'motrpac_tissues': ', '.join(llm.get('motrpac_utility', {}).get('motrpac_tissues', [])),
                'validation_passed': quality.get('validation_passed'),
                'summary': (llm.get('summary') or '')[:200],
            }
            
            writer.writerow(row)
    
    logger.info(f"Saved CSV summary to {output_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Combine data sources into unified studies catalog',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Data sources merged:
  - master_catalog.json: Basic GEO/ArrayExpress metadata
  - matrix_analysis.json: Gene/cell counts, formats
  - llm_analysis.json: LLM-validated metadata, topics, utility

Output:
  - unified_studies.json: Combined view with all fields
  - unified_studies_summary.csv: Flat summary (with --csv-summary)
  
Examples:
  python combine_data_sources.py --config config.yaml
  python combine_data_sources.py --config config.yaml --csv-summary
  python combine_data_sources.py --config config.yaml --llm llm_analysis.json
  python combine_data_sources.py --config config.yaml --catalog my_catalog.json --matrix my_matrix.json
        """
    )
    
    parser.add_argument('--config', '-c', required=True, help='Path to config.yaml')
    parser.add_argument('--output', '-o', help='Output JSON path (default: unified_studies.json)')
    parser.add_argument('--csv-summary', action='store_true', help='Also generate CSV summary')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    # Input file names (relative to catalog_dir)
    parser.add_argument('--catalog', default='master_catalog.json',
                        help='Catalog JSON filename (default: master_catalog.json)')
    parser.add_argument('--matrix', default='matrix_analysis.json',
                        help='Matrix analysis JSON filename (default: matrix_analysis.json)')
    parser.add_argument('--llm', default='llm_test.json',
                        help='LLM analysis JSON filename (default: llm_test.json)')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if not HAS_YAML:
        print("ERROR: PyYAML required. Install: pip install pyyaml")
        return 1
    
    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    catalog_dir = Path(config.get('catalog_dir', './catalog'))
    
    output_path = Path(args.output) if args.output else catalog_dir / 'unified_studies.json'
    csv_path = output_path.with_suffix('.csv') if args.csv_summary else None
    
    logger.info(f"Catalog directory: {catalog_dir}")
    logger.info(f"Output: {output_path}")
    logger.info(f"Input files: {args.catalog}, {args.matrix}, {args.llm}")
    
    # Load all data sources (using user-specified filenames)
    catalog_by_acc = load_catalog(catalog_dir, args.catalog)
    matrix_by_acc = load_matrix_analysis(catalog_dir, args.matrix)
    llm_by_acc = load_llm_analysis(catalog_dir, args.llm)
    
    if not catalog_by_acc and not matrix_by_acc and not llm_by_acc:
        logger.error("No data sources found!")
        return 1
    
    # Combine
    unified_studies = combine_studies(catalog_by_acc, matrix_by_acc, llm_by_acc)
    
    # Compute aggregate stats
    stats = compute_aggregate_stats(unified_studies)
    
    # Build output
    output = {
        'generated_at': datetime.now().isoformat(),
        'sources': {
            'catalog': str(catalog_dir / args.catalog),
            'matrix': str(catalog_dir / args.matrix),
            'llm': str(catalog_dir / args.llm),
        },
        'statistics': stats,
        'studies': unified_studies,
    }
    
    # Save JSON
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    logger.info(f"Saved {len(unified_studies)} unified studies to {output_path}")
    
    # Save CSV summary
    if csv_path:
        generate_csv_summary(unified_studies, csv_path)
    
    # Print summary
    print("\n" + "=" * 60)
    print("UNIFIED CATALOG SUMMARY")
    print("=" * 60)
    print(f"Total studies: {stats['total_studies']}")
    print(f"\nCoverage:")
    print(f"  Has catalog: {stats['coverage']['has_catalog']}")
    print(f"  Has matrix: {stats['coverage']['has_matrix']}")
    print(f"  Has LLM analysis: {stats['coverage']['has_llm']}")
    print(f"  Has all three: {stats['coverage']['has_all_three']}")
    print(f"  Missing matrix only: {stats['coverage']['catalog_llm_only']}")
    print(f"  Multi-strain studies: {stats['coverage']['multi_strain']}")
    print(f"\nMatrix stats:")
    print(f"  Total cells: {stats['matrix_stats']['total_cells']:,}")
    print(f"  Total samples: {stats['matrix_stats']['total_samples']:,}")
    print(f"  Valid gene counts: {stats['matrix_stats']['gene_count_valid']}")
    print(f"\nLLM stats:")
    print(f"  Successfully analyzed: {stats['llm_stats']['successful']}")
    print(f"  GeneCompass useful: {stats['llm_stats']['genecompass_useful']}")
    print(f"  Deconvolution useful: {stats['llm_stats']['deconvolution_useful']}")
    print(f"  GRN useful: {stats['llm_stats']['grn_useful']}")
    print(f"  Exercise studies: {stats['llm_stats']['exercise_studies']}")
    print(f"  Disease models: {stats['llm_stats']['disease_models']}")
    print(f"  Single-cell: {stats['llm_stats']['single_cell']}")
    print(f"  Time series: {stats['llm_stats']['time_series']}")
    print(f"  Validation issues: {stats['llm_stats']['validation_issues']}")
    print(f"  ⚠ Possible mouse contamination: {stats['llm_stats']['possible_mouse_contamination']}")
    print(f"\nTop topic categories:")
    for cat, count in list(stats['by_topic_category'].items())[:10]:
        print(f"  {cat}: {count}")
    print(f"\nTop MoTrPAC tissues (normalized):")
    for tissue, count in list(stats['by_motrpac_tissue'].items())[:12]:
        print(f"  {tissue}: {count}")
    print(f"\nTop strains (normalized):")
    for strain, count in list(stats['by_strain'].items())[:8]:
        print(f"  {strain}: {count}")
    print(f"\nSex distribution (normalized):")
    for sex, count in list(stats['by_sex'].items())[:6]:
        print(f"  {sex}: {count}")
    
    return 0


if __name__ == '__main__':
    exit(main())