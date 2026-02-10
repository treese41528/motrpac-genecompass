#!/usr/bin/env python3
"""
export_training_corpus.py

Export filtered training corpus for GeneCompass fine-tuning from unified study data.

This script:
1. Loads unified_studies.json (merged catalog + matrix + LLM data)
2. Applies inclusion criteria for GeneCompass training
3. Applies exclusion list (MoTrPAC studies, contaminated data)
4. Exports training manifest with file paths and metadata

VERSION 1.1 FIXES:
- Safer is_rat handling: explicitly tracks studies excluded due to missing is_rat
  field vs. those with is_rat=False, with a warning report for manual review
- Added --warn-missing-is-rat flag to include studies with motrpac_utility but
  no explicit is_rat field (for cases where LLM omitted the field)

Author: Tim Reese / MoTrPAC GeneCompass Project
Date: January 2026 (v1.1: February 2026)
"""

import json
import csv
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict, field
from collections import Counter, defaultdict

# =============================================================================
# CONFIGURATION
# =============================================================================

# Studies to exclude (MoTrPAC consortium / data leakage prevention)
EXCLUSION_LIST = {
    "GSE242354": "MoTrPAC Endurance Exercise Training Study - primary evaluation target",
    # Add other MoTrPAC studies as identified
    # "GSE_XXXXX": "MoTrPAC related study - reason",
}

# MoTrPAC-related keywords to flag for manual review
MOTRPAC_KEYWORDS = [
    "motrpac",
    "molecular transducers of physical activity",
    "endurance exercise training",
    "fischer 344",  # MoTrPAC uses Fischer rats
    "f344",
]

# Gene count thresholds
MIN_GENES = 5000      # Minimum genes for usable matrix
MAX_GENES = 100000    # Maximum (above suggests contamination/wrong species)
NORMAL_MIN = 15000    # Normal range minimum
NORMAL_MAX = 40000    # Normal range maximum

# Preferred gene ID types for GeneCompass
PREFERRED_GENE_IDS = [
    "ensembl_rat",
    "ensembl_rattus",
    "ensembl",
    "gene_symbol",
    "symbol",
]

# MoTrPAC tissues of interest (from proposal)
MOTRPAC_TISSUES = [
    "heart", "liver", "kidney", "lung", "hippocampus", "cortex",
    "hypothalamus", "gastrocnemius", "vastus_lateralis", "white_adipose",
    "brown_adipose", "small_intestine", "colon", "spleen", "adrenal",
    "ovaries", "testes", "blood", "vena_cava", "aorta"
]


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class StudyQuality:
    """Quality assessment for a study."""
    score: float = 0.0
    flags: List[str] = field(default_factory=list)
    gene_count_status: str = "unknown"
    has_matrix: bool = False
    has_llm_validation: bool = False
    organism_validated: bool = False
    tissues_validated: bool = False


@dataclass 
class TrainingStudy:
    """A study approved for training corpus."""
    accession: str
    source: str  # geo or arrayexpress
    data_type: str  # single_cell or bulk
    
    # Matrix info
    n_genes: Optional[int] = None
    n_cells: Optional[int] = None
    n_samples: Optional[int] = None
    gene_id_type: Optional[str] = None
    matrix_formats: List[str] = field(default_factory=list)
    
    # Biological info
    organism: str = "Rattus norvegicus"
    strain: Optional[str] = None
    tissues: List[str] = field(default_factory=list)
    motrpac_tissues: List[str] = field(default_factory=list)
    
    # Study metadata
    title: str = ""
    topic: str = ""
    topic_category: str = ""
    has_exercise: bool = False
    has_disease_model: bool = False
    disease_name: Optional[str] = None
    
    # Utility flags
    genecompass_useful: bool = True
    deconvolution_useful: bool = False
    grn_useful: bool = False
    
    # Quality
    quality_score: float = 0.0
    quality_flags: List[str] = field(default_factory=list)
    
    # File paths
    matrix_files: List[str] = field(default_factory=list)
    base_path: str = ""


@dataclass
class ExcludedStudy:
    """A study excluded from training."""
    accession: str
    reason: str
    category: str  # "motrpac", "quality", "species", "missing_data", "manual"
    details: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def setup_logging(verbose: bool = False) -> logging.Logger:
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)


def safe_get(d: Dict, *keys, default=None):
    """Safely navigate nested dictionary."""
    for key in keys:
        if isinstance(d, dict):
            d = d.get(key, default)
        else:
            return default
    return d if d is not None else default


def check_motrpac_keywords(text: str) -> bool:
    """Check if text contains MoTrPAC-related keywords."""
    if not text:
        return False
    text_lower = text.lower()
    return any(kw in text_lower for kw in MOTRPAC_KEYWORDS)


def assess_gene_count(n_genes: Optional[int]) -> Tuple[str, List[str]]:
    """Assess gene count quality."""
    if n_genes is None:
        return "unknown", ["missing_gene_count"]
    
    flags = []
    if n_genes < MIN_GENES:
        return "too_low", ["gene_count_below_minimum"]
    elif n_genes > MAX_GENES:
        return "too_high", ["gene_count_above_maximum", "possible_contamination"]
    elif n_genes < NORMAL_MIN:
        flags.append("gene_count_below_normal")
        return "low", flags
    elif n_genes > NORMAL_MAX:
        flags.append("gene_count_above_normal")
        return "high", flags
    else:
        return "normal", []


def calculate_quality_score(study: Dict, has_llm: bool) -> StudyQuality:
    """Calculate quality score for a study."""
    quality = StudyQuality()
    score = 0.0
    
    # Matrix data (+30 points)
    matrix = study.get("matrix", {})
    if matrix:
        n_genes = matrix.get("n_genes")
        if n_genes:
            quality.has_matrix = True
            quality.gene_count_status, gene_flags = assess_gene_count(n_genes)
            quality.flags.extend(gene_flags)
            
            if quality.gene_count_status == "normal":
                score += 30
            elif quality.gene_count_status in ["low", "high"]:
                score += 15
            elif quality.gene_count_status == "too_low":
                score += 5
            # too_high gets 0 (likely contamination)
    
    # Gene ID type (+20 points)
    gene_id = matrix.get("gene_id_type", "").lower() if matrix else ""
    if any(pref in gene_id for pref in PREFERRED_GENE_IDS):
        score += 20
    elif gene_id:
        score += 10
    else:
        quality.flags.append("unknown_gene_id_type")
    
    # LLM validation (+30 points)
    llm = study.get("llm", {})
    if llm and has_llm:
        quality.has_llm_validation = True
        
        # Using correct schema: validation (was metadata_validation)
        validation = llm.get("validation", {})
        if validation.get("organism_correct", False):
            quality.organism_validated = True
            score += 15
        else:
            quality.flags.append("organism_validation_failed")
        
        if validation.get("tissues_correct", False):
            quality.tissues_validated = True
            score += 15
        else:
            quality.flags.append("tissue_validation_failed")
    else:
        quality.flags.append("no_llm_validation")
    
    # Utility flags (+20 points)
    utility = safe_get(llm, "motrpac_utility", default={})
    if utility.get("genecompass_useful"):
        score += 10
    if utility.get("grn_useful"):
        score += 5
    if utility.get("deconvolution_useful"):
        score += 5
    
    quality.score = min(score, 100.0)  # Cap at 100
    return quality


# =============================================================================
# MAIN PROCESSING
# =============================================================================

def load_unified_data(filepath: Path, logger: logging.Logger) -> Dict:
    """Load unified studies JSON."""
    logger.info(f"Loading unified data from {filepath}")
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # Handle different possible structures
    if isinstance(data, list):
        # Convert list to dict keyed by accession
        studies = {s.get("accession", f"unknown_{i}"): s for i, s in enumerate(data)}
    elif isinstance(data, dict):
        # Could be {"studies": [...]} or {accession: study, ...}
        if "studies" in data:
            studies = {s.get("accession", f"unknown_{i}"): s 
                      for i, s in enumerate(data["studies"])}
        elif "analyses" in data:
            # LLM output format
            studies = {s.get("accession", f"unknown_{i}"): s 
                      for i, s in enumerate(data["analyses"])}
        else:
            studies = data
    else:
        raise ValueError(f"Unexpected data format: {type(data)}")
    
    logger.info(f"Loaded {len(studies)} studies")
    return studies


def extract_study_info(accession: str, study: Dict, logger: logging.Logger) -> Optional[TrainingStudy]:
    """Extract relevant information from a study record."""
    
    # Determine source
    source = study.get("source", "")
    if not source:
        source = "arrayexpress" if accession.startswith("E-") else "geo"
    
    # Get matrix info
    matrix = study.get("matrix", {})
    n_genes = matrix.get("n_genes")
    n_cells = matrix.get("n_cells")
    n_samples = matrix.get("n_samples")
    gene_id_type = matrix.get("gene_id_type")
    formats = matrix.get("formats", [])
    if isinstance(formats, str):
        formats = [formats]
    
    # Get LLM info - using actual unified_studies.json schema
    llm = study.get("llm", {})
    
    # motrpac_utility
    utility = safe_get(llm, "motrpac_utility", default={})
    
    # topic
    topic_info = safe_get(llm, "topic", default={})
    
    # validated_organism
    organism_info = safe_get(llm, "validated_organism", default={})
    
    # treatments
    treatments = safe_get(llm, "treatments", default={})
    
    # disease
    disease = safe_get(llm, "disease", default={})
    
    # study_type
    study_type_info = safe_get(llm, "study_type", default={})
    
    # Determine data type
    data_type = study.get("data_type", "")
    if not data_type:
        if study_type_info.get("is_single_cell"):
            data_type = "single_cell"
        elif n_cells and n_cells > 1000:
            data_type = "single_cell"
        else:
            data_type = "bulk"
    
    # Get tissues from validated_tissues
    tissues = []
    motrpac_tissues = []
    llm_tissues = safe_get(llm, "validated_tissues", default=[])
    if isinstance(llm_tissues, list):
        for t in llm_tissues:
            if isinstance(t, dict):
                tname = t.get("name", "")
                if tname:
                    tissues.append(tname)
                mpt = t.get("motrpac_match", "")
                if mpt:
                    motrpac_tissues.append(mpt)
            elif isinstance(t, str):
                tissues.append(t)
    
    # Also check motrpac_utility.motrpac_tissues
    utility_tissues = utility.get("motrpac_tissues", [])
    if isinstance(utility_tissues, list):
        for t in utility_tissues:
            if t and t not in motrpac_tissues:
                motrpac_tissues.append(t)
    
    # Also check catalog tissues
    catalog = study.get("catalog", {})
    catalog_tissues = catalog.get("tissues", [])
    if isinstance(catalog_tissues, list):
        for t in catalog_tissues:
            if isinstance(t, str) and t not in tissues:
                tissues.append(t)
    
    # Get file paths from catalog
    matrix_files = []
    files_info = catalog.get("files", {})
    if isinstance(files_info, dict):
        matrix_files = files_info.get("matrix_files", [])
    elif isinstance(files_info, list):
        matrix_files = files_info
    
    # Base path
    base_path = catalog.get("base_path", catalog.get("path", ""))
    
    # Get title from catalog or topic
    title = catalog.get("title", "")
    if not title:
        title = topic_info.get("primary", "")
    
    # Build training study object
    ts = TrainingStudy(
        accession=accession,
        source=source,
        data_type=data_type,
        n_genes=n_genes,
        n_cells=n_cells,
        n_samples=n_samples,
        gene_id_type=gene_id_type,
        matrix_formats=formats,
        organism=organism_info.get("species", "Rattus norvegicus") if isinstance(organism_info, dict) else "Rattus norvegicus",
        strain=organism_info.get("strain_normalized", organism_info.get("strain")) if isinstance(organism_info, dict) else None,
        tissues=tissues,
        motrpac_tissues=motrpac_tissues,
        title=title,
        topic=topic_info.get("primary", "") if isinstance(topic_info, dict) else "",
        topic_category=topic_info.get("category", "") if isinstance(topic_info, dict) else "",
        has_exercise=treatments.get("has_exercise", False) if isinstance(treatments, dict) else False,
        has_disease_model=disease.get("has_model", False) if isinstance(disease, dict) else False,
        disease_name=disease.get("name") if isinstance(disease, dict) else None,
        genecompass_useful=utility.get("genecompass_useful", False) if isinstance(utility, dict) else False,
        deconvolution_useful=utility.get("deconvolution_useful", False) if isinstance(utility, dict) else False,
        grn_useful=utility.get("grn_useful", False) if isinstance(utility, dict) else False,
        matrix_files=matrix_files,
        base_path=base_path,
    )
    
    return ts


def should_exclude(accession: str, study: Dict, ts: TrainingStudy, 
                   logger: logging.Logger,
                   lenient_is_rat: bool = False) -> Optional[ExcludedStudy]:
    """Determine if a study should be excluded.
    
    Args:
        lenient_is_rat: If True, studies with motrpac_utility present but
            is_rat field missing are NOT excluded (assumes LLM omission).
            If False (default), missing is_rat is treated as False.
    """
    
    # Check explicit exclusion list
    if accession in EXCLUSION_LIST:
        return ExcludedStudy(
            accession=accession,
            reason=EXCLUSION_LIST[accession],
            category="motrpac",
            details={"source": "explicit_exclusion_list"}
        )
    
    # Check for MoTrPAC keywords in title/summary
    title = ts.title or ""
    catalog = study.get("catalog", {})
    summary = catalog.get("summary", "")
    if check_motrpac_keywords(title) or check_motrpac_keywords(summary):
        return ExcludedStudy(
            accession=accession,
            reason="Contains MoTrPAC-related keywords - potential data leakage",
            category="motrpac",
            details={"title": title[:100], "flagged_keywords": True}
        )
    
    # Check organism validation
    llm = study.get("llm", {})
    utility = safe_get(llm, "motrpac_utility", default={})
    
    # No LLM data at all -> exclude
    if not utility:
        return ExcludedStudy(
            accession=accession,
            reason="No LLM validation data - cannot confirm species",
            category="species",
            details={"organism": ts.organism, "has_llm": False}
        )
    
    # FIX: Distinguish between is_rat=False and is_rat missing
    is_rat_value = utility.get("is_rat")  # Returns None if key missing
    
    if is_rat_value is None:
        # Field is missing from LLM output
        if lenient_is_rat:
            # Lenient mode: assume LLM omission, let other checks decide
            logger.debug(f"{accession}: is_rat field missing from motrpac_utility "
                        f"(lenient mode - not excluding)")
        else:
            # Strict mode: missing = exclude, but with distinct reason
            return ExcludedStudy(
                accession=accession,
                reason="is_rat field missing from motrpac_utility (LLM may have omitted it)",
                category="species_missing_field",
                details={
                    "organism": ts.organism,
                    "is_rat": None,
                    "utility_keys": list(utility.keys()),
                    "note": "Re-run with --lenient-is-rat to include these studies"
                }
            )
    elif not is_rat_value:
        # Field is explicitly False
        return ExcludedStudy(
            accession=accession,
            reason="LLM validation explicitly indicates non-rat species (is_rat=False)",
            category="species",
            details={"organism": ts.organism, "is_rat": False}
        )
    
    # Check for extreme gene counts (likely wrong species)
    if ts.n_genes and ts.n_genes > MAX_GENES:
        return ExcludedStudy(
            accession=accession,
            reason=f"Gene count ({ts.n_genes}) exceeds maximum - likely contamination or wrong species",
            category="quality",
            details={"n_genes": ts.n_genes, "threshold": MAX_GENES}
        )
    
    # Check for GeneCompass usefulness (if LLM data available)
    if utility and not utility.get("genecompass_useful", True):
        return ExcludedStudy(
            accession=accession,
            reason="LLM flagged as not useful for GeneCompass",
            category="quality",
            details={"utility_flags": utility}
        )
    
    return None  # Include the study


def filter_studies(studies: Dict, logger: logging.Logger,
                   require_matrix: bool = True,
                   require_llm: bool = False,
                   min_quality: float = 0.0,
                   data_type_filter: str = "both",
                   min_cells: Optional[int] = None,
                   min_samples: Optional[int] = None,
                   exercise_only: bool = False,
                   tissue_filter: Optional[List[str]] = None,
                   motrpac_tissues_only: bool = False,
                   lenient_is_rat: bool = False) -> Tuple[List[TrainingStudy], List[ExcludedStudy]]:
    """Filter studies for training corpus.
    
    Args:
        studies: Dictionary of studies keyed by accession
        logger: Logger instance
        require_matrix: Only include studies with matrix data
        require_llm: Only include studies with LLM validation
        min_quality: Minimum quality score (0-100)
        data_type_filter: "single_cell", "bulk", or "both"
        min_cells: Minimum cell count for single-cell studies
        min_samples: Minimum sample count for bulk studies
        exercise_only: Only include exercise-related studies
        tissue_filter: List of tissues to filter by (any match)
        motrpac_tissues_only: Only include studies with MoTrPAC tissues
        lenient_is_rat: If True, don't exclude studies where is_rat field is missing
    """
    
    included = []
    excluded = []
    
    for accession, study in studies.items():
        # Extract study info
        ts = extract_study_info(accession, study, logger)
        if ts is None:
            excluded.append(ExcludedStudy(
                accession=accession,
                reason="Failed to extract study information",
                category="missing_data"
            ))
            continue
        
        # Check exclusion criteria
        exclusion = should_exclude(accession, study, ts, logger,
                                   lenient_is_rat=lenient_is_rat)
        if exclusion:
            excluded.append(exclusion)
            continue
        
        # Data type filter
        if data_type_filter != "both":
            if ts.data_type != data_type_filter:
                excluded.append(ExcludedStudy(
                    accession=accession,
                    reason=f"Data type '{ts.data_type}' does not match filter '{data_type_filter}'",
                    category="data_type_filter",
                    details={"data_type": ts.data_type, "filter": data_type_filter}
                ))
                continue
        
        # Check matrix requirement
        if require_matrix and not ts.n_genes:
            excluded.append(ExcludedStudy(
                accession=accession,
                reason="No matrix data available",
                category="missing_data",
                details={"has_matrix_formats": bool(ts.matrix_formats)}
            ))
            continue
        
        # Check minimum gene count
        if ts.n_genes and ts.n_genes < MIN_GENES:
            excluded.append(ExcludedStudy(
                accession=accession,
                reason=f"Gene count ({ts.n_genes}) below minimum threshold ({MIN_GENES})",
                category="quality",
                details={"n_genes": ts.n_genes}
            ))
            continue
        
        # Minimum cell count for single-cell
        if min_cells and ts.data_type == "single_cell":
            cell_count = ts.n_cells or 0
            if cell_count < min_cells:
                excluded.append(ExcludedStudy(
                    accession=accession,
                    reason=f"Cell count ({cell_count}) below minimum threshold ({min_cells})",
                    category="quality",
                    details={"n_cells": cell_count, "threshold": min_cells}
                ))
                continue
        
        # Minimum sample count for bulk
        if min_samples and ts.data_type == "bulk":
            sample_count = ts.n_samples or 0
            if sample_count < min_samples:
                excluded.append(ExcludedStudy(
                    accession=accession,
                    reason=f"Sample count ({sample_count}) below minimum threshold ({min_samples})",
                    category="quality",
                    details={"n_samples": sample_count, "threshold": min_samples}
                ))
                continue
        
        # Exercise-only filter
        if exercise_only and not ts.has_exercise:
            excluded.append(ExcludedStudy(
                accession=accession,
                reason="Not an exercise-related study",
                category="exercise_filter",
                details={"has_exercise": ts.has_exercise}
            ))
            continue
        
        # Tissue filter
        if tissue_filter:
            study_tissues = [t.lower() for t in (ts.tissues + ts.motrpac_tissues)]
            filter_tissues = [t.lower() for t in tissue_filter]
            if not any(ft in study_tissues or any(ft in st for st in study_tissues) 
                      for ft in filter_tissues):
                excluded.append(ExcludedStudy(
                    accession=accession,
                    reason=f"No matching tissues (filter: {tissue_filter})",
                    category="tissue_filter",
                    details={"study_tissues": ts.tissues, "filter": tissue_filter}
                ))
                continue
        
        # MoTrPAC tissues only filter
        if motrpac_tissues_only and not ts.motrpac_tissues:
            excluded.append(ExcludedStudy(
                accession=accession,
                reason="No MoTrPAC-relevant tissues identified",
                category="tissue_filter",
                details={"tissues": ts.tissues}
            ))
            continue
        
        # Check LLM requirement
        has_llm = bool(study.get("llm"))
        if require_llm and not has_llm:
            excluded.append(ExcludedStudy(
                accession=accession,
                reason="No LLM validation data available",
                category="missing_data"
            ))
            continue
        
        # Calculate quality score
        quality = calculate_quality_score(study, has_llm)
        ts.quality_score = quality.score
        ts.quality_flags = quality.flags
        
        # Check minimum quality
        if quality.score < min_quality:
            excluded.append(ExcludedStudy(
                accession=accession,
                reason=f"Quality score ({quality.score}) below threshold ({min_quality})",
                category="quality",
                details={"quality_score": quality.score, "flags": quality.flags}
            ))
            continue
        
        # Include the study
        included.append(ts)
    
    logger.info(f"Filtering complete: {len(included)} included, {len(excluded)} excluded")
    
    # Report on is_rat field coverage
    missing_is_rat = [e for e in excluded if e.category == "species_missing_field"]
    explicit_not_rat = [e for e in excluded if e.category == "species" and 
                        e.details.get("is_rat") is False]
    
    if missing_is_rat:
        logger.warning(f"  {len(missing_is_rat)} studies excluded because is_rat field "
                      f"is MISSING from motrpac_utility (possible LLM omission)")
        logger.warning(f"  Use --lenient-is-rat to include these studies")
        for e in missing_is_rat[:5]:
            logger.warning(f"    {e.accession}: keys={e.details.get('utility_keys', [])}")
        if len(missing_is_rat) > 5:
            logger.warning(f"    ... and {len(missing_is_rat) - 5} more")
    
    if explicit_not_rat:
        logger.info(f"  {len(explicit_not_rat)} studies excluded with explicit is_rat=False")
    
    return included, excluded


def generate_statistics(included: List[TrainingStudy], 
                       excluded: List[ExcludedStudy]) -> Dict:
    """Generate corpus statistics."""
    
    stats = {
        "generated_at": datetime.now().isoformat(),
        "total_input": len(included) + len(excluded),
        "total_included": len(included),
        "total_excluded": len(excluded),
        
        "included": {
            "by_source": Counter(s.source for s in included),
            "by_data_type": Counter(s.data_type for s in included),
            "by_topic_category": Counter(s.topic_category for s in included if s.topic_category),
            "by_gene_id_type": Counter(s.gene_id_type for s in included if s.gene_id_type),
            
            "total_cells": sum(s.n_cells or 0 for s in included),
            "total_samples": sum(s.n_samples or 0 for s in included),
            
            "with_exercise": sum(1 for s in included if s.has_exercise),
            "with_disease_model": sum(1 for s in included if s.has_disease_model),
            "with_motrpac_tissues": sum(1 for s in included if s.motrpac_tissues),
            
            "single_cell_count": sum(1 for s in included if s.data_type == "single_cell"),
            "bulk_count": sum(1 for s in included if s.data_type == "bulk"),
            
            "quality_score_distribution": {
                "high_90_100": sum(1 for s in included if s.quality_score >= 90),
                "good_70_89": sum(1 for s in included if 70 <= s.quality_score < 90),
                "medium_50_69": sum(1 for s in included if 50 <= s.quality_score < 70),
                "low_below_50": sum(1 for s in included if s.quality_score < 50),
            },
            
            "gene_count_distribution": {
                "normal_15k_40k": sum(1 for s in included if s.n_genes and NORMAL_MIN <= s.n_genes <= NORMAL_MAX),
                "low_5k_15k": sum(1 for s in included if s.n_genes and MIN_GENES <= s.n_genes < NORMAL_MIN),
                "high_40k_100k": sum(1 for s in included if s.n_genes and NORMAL_MAX < s.n_genes <= MAX_GENES),
                "unknown": sum(1 for s in included if not s.n_genes),
            },
        },
        
        "excluded": {
            "by_category": Counter(e.category for e in excluded),
            "motrpac_related": [e.accession for e in excluded if e.category == "motrpac"],
            "missing_is_rat_field": [e.accession for e in excluded if e.category == "species_missing_field"],
        },
        
        "tissue_coverage": Counter(
            t for s in included for t in s.motrpac_tissues
        ),
    }
    
    return stats


def export_corpus(included: List[TrainingStudy], 
                  excluded: List[ExcludedStudy],
                  output_dir: Path,
                  logger: logging.Logger,
                  data_type_filter: str = "both"):
    """Export the training corpus files."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Training manifest (complete JSON)
    manifest_path = output_dir / "training_manifest.json"
    manifest = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "version": "1.1",
            "total_studies": len(included),
            "data_type_filter": data_type_filter,
            "description": f"GeneCompass fine-tuning corpus for MoTrPAC rat transcriptomics ({data_type_filter})"
        },
        "studies": [asdict(s) for s in included]
    }
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"Wrote training manifest: {manifest_path}")
    
    # 2. Exclusion list (for documentation)
    exclusion_path = output_dir / "exclusion_list.json"
    exclusions = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "total_excluded": len(excluded)
        },
        "explicit_exclusions": EXCLUSION_LIST,
        "excluded_studies": [asdict(e) for e in excluded]
    }
    with open(exclusion_path, 'w') as f:
        json.dump(exclusions, f, indent=2)
    logger.info(f"Wrote exclusion list: {exclusion_path}")
    
    # 3. Summary CSV
    csv_path = output_dir / "training_studies.csv"
    fieldnames = [
        "accession", "source", "data_type", "n_genes", "n_cells", "n_samples",
        "gene_id_type", "strain", "topic_category", "has_exercise", 
        "has_disease_model", "quality_score", "motrpac_tissues"
    ]
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in included:
            writer.writerow({
                "accession": s.accession,
                "source": s.source,
                "data_type": s.data_type,
                "n_genes": s.n_genes,
                "n_cells": s.n_cells,
                "n_samples": s.n_samples,
                "gene_id_type": s.gene_id_type,
                "strain": s.strain,
                "topic_category": s.topic_category,
                "has_exercise": s.has_exercise,
                "has_disease_model": s.has_disease_model,
                "quality_score": round(s.quality_score, 1),
                "motrpac_tissues": ";".join(s.motrpac_tissues) if s.motrpac_tissues else ""
            })
    logger.info(f"Wrote summary CSV: {csv_path}")
    
    # 4. Statistics report
    stats = generate_statistics(included, excluded)
    stats_path = output_dir / "corpus_statistics.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2, default=str)
    logger.info(f"Wrote statistics: {stats_path}")
    
    # 5. Human-readable report
    report_path = output_dir / "corpus_report.txt"
    with open(report_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("GENECOMPASS TRAINING CORPUS REPORT\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Data Type Filter: {data_type_filter.upper()}\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("SUMMARY\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total studies included:  {len(included)}\n")
        f.write(f"Total studies excluded:  {len(excluded)}\n")
        f.write(f"Total cells:             {stats['included']['total_cells']:,}\n")
        f.write(f"Total samples:           {stats['included']['total_samples']:,}\n\n")
        
        f.write("BY DATA TYPE\n")
        f.write("-" * 40 + "\n")
        f.write(f"Single-cell:  {stats['included']['single_cell_count']}\n")
        f.write(f"Bulk RNA-seq: {stats['included']['bulk_count']}\n\n")
        
        f.write("BY SOURCE\n")
        f.write("-" * 40 + "\n")
        for source, count in stats['included']['by_source'].items():
            f.write(f"{source}: {count}\n")
        f.write("\n")
        
        f.write("SPECIAL CATEGORIES\n")
        f.write("-" * 40 + "\n")
        f.write(f"Exercise studies:        {stats['included']['with_exercise']}\n")
        f.write(f"Disease model studies:   {stats['included']['with_disease_model']}\n")
        f.write(f"With MoTrPAC tissues:    {stats['included']['with_motrpac_tissues']}\n\n")
        
        f.write("GENE COUNT DISTRIBUTION\n")
        f.write("-" * 40 + "\n")
        for category, count in stats['included']['gene_count_distribution'].items():
            f.write(f"{category}: {count}\n")
        f.write("\n")
        
        f.write("QUALITY SCORE DISTRIBUTION\n")
        f.write("-" * 40 + "\n")
        for category, count in stats['included']['quality_score_distribution'].items():
            f.write(f"{category}: {count}\n")
        f.write("\n")
        
        f.write("EXCLUSION REASONS\n")
        f.write("-" * 40 + "\n")
        for category, count in stats['excluded']['by_category'].items():
            f.write(f"{category}: {count}\n")
        f.write("\n")
        
        # Highlight missing is_rat field
        missing_is_rat = stats['excluded'].get('missing_is_rat_field', [])
        if missing_is_rat:
            f.write("WARNING: STUDIES WITH MISSING is_rat FIELD\n")
            f.write("-" * 40 + "\n")
            f.write(f"Count: {len(missing_is_rat)}\n")
            f.write("These studies have motrpac_utility data but the is_rat\n")
            f.write("field was not set by the LLM. Re-run with --lenient-is-rat\n")
            f.write("to include them, or manually verify species.\n")
            for acc in missing_is_rat[:20]:
                f.write(f"  - {acc}\n")
            if len(missing_is_rat) > 20:
                f.write(f"  ... and {len(missing_is_rat) - 20} more\n")
            f.write("\n")
        
        f.write("MOTRPAC-RELATED EXCLUSIONS (DATA LEAKAGE PREVENTION)\n")
        f.write("-" * 40 + "\n")
        for acc in stats['excluded']['motrpac_related']:
            f.write(f"  - {acc}\n")
        f.write("\n")
        
        f.write("TOP TOPIC CATEGORIES\n")
        f.write("-" * 40 + "\n")
        for topic, count in stats['included']['by_topic_category'].most_common(15):
            f.write(f"{topic}: {count}\n")
        f.write("\n")
        
        f.write("MOTRPAC TISSUE COVERAGE\n")
        f.write("-" * 40 + "\n")
        for tissue, count in stats['tissue_coverage'].most_common():
            marker = "✓" if tissue.lower() in [t.lower() for t in MOTRPAC_TISSUES] else " "
            f.write(f"[{marker}] {tissue}: {count}\n")
    
    logger.info(f"Wrote human-readable report: {report_path}")
    
    # 6. File inventory (paths to actual matrices)
    inventory_path = output_dir / "file_inventory.txt"
    with open(inventory_path, 'w') as f:
        f.write("# GeneCompass Training Corpus - File Inventory\n")
        f.write(f"# Generated: {datetime.now().isoformat()}\n")
        f.write("# Format: accession<TAB>data_type<TAB>file_path\n\n")
        
        for s in included:
            if s.matrix_files:
                for mf in s.matrix_files:
                    f.write(f"{s.accession}\t{s.data_type}\t{mf}\n")
            elif s.base_path:
                f.write(f"{s.accession}\t{s.data_type}\t{s.base_path}\n")
            else:
                f.write(f"{s.accession}\t{s.data_type}\t# NO PATH AVAILABLE\n")
    
    logger.info(f"Wrote file inventory: {inventory_path}")
    
    # 7. Quick accession lists
    for dtype in ["single_cell", "bulk"]:
        acc_path = output_dir / f"{dtype}_accessions.txt"
        accs = [s.accession for s in included if s.data_type == dtype]
        with open(acc_path, 'w') as f:
            f.write("\n".join(sorted(accs)))
        logger.info(f"Wrote {dtype} accession list: {acc_path} ({len(accs)} studies)")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Export GeneCompass training corpus from unified study data"
    )
    parser.add_argument(
        "-i", "--input",
        type=Path,
        default=Path("/depot/reese18/data/catalog/unified_studies.json"),
        help="Path to unified_studies.json"
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        default=Path("/depot/reese18/data/training/genecompass_corpus"),
        help="Output directory for corpus files"
    )
    parser.add_argument(
        "-t", "--data-type",
        choices=["single_cell", "bulk", "both"],
        default="both",
        help="Filter by data type: single_cell, bulk, or both (default: both)"
    )
    parser.add_argument(
        "--min-cells",
        type=int,
        default=None,
        help="Minimum cell count for single-cell studies"
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=None,
        help="Minimum sample count for bulk studies"
    )
    parser.add_argument(
        "--require-matrix",
        action="store_true",
        default=True,
        help="Only include studies with matrix data"
    )
    parser.add_argument(
        "--no-require-matrix",
        action="store_false",
        dest="require_matrix",
        help="Include studies without matrix data"
    )
    parser.add_argument(
        "--require-llm",
        action="store_true",
        default=False,
        help="Only include studies with LLM validation"
    )
    parser.add_argument(
        "--min-quality",
        type=float,
        default=0.0,
        help="Minimum quality score threshold (0-100)"
    )
    parser.add_argument(
        "--exercise-only",
        action="store_true",
        help="Only include exercise-related studies"
    )
    parser.add_argument(
        "--tissue",
        action="append",
        dest="tissues",
        help="Filter to specific tissues (can be repeated, e.g., --tissue heart --tissue liver)"
    )
    parser.add_argument(
        "--motrpac-tissues-only",
        action="store_true",
        help="Only include studies with MoTrPAC-relevant tissues"
    )
    parser.add_argument(
        "--lenient-is-rat",
        action="store_true",
        default=False,
        help="Don't exclude studies where is_rat field is missing from motrpac_utility "
             "(treats missing field as LLM omission rather than non-rat)"
    )
    parser.add_argument(
        "--add-exclusion",
        nargs=2,
        action="append",
        metavar=("ACCESSION", "REASON"),
        help="Add additional exclusion (can be repeated)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    logger = setup_logging(args.verbose)
    
    # Add any additional exclusions
    if args.add_exclusion:
        for acc, reason in args.add_exclusion:
            EXCLUSION_LIST[acc] = reason
            logger.info(f"Added exclusion: {acc} - {reason}")
    
    # Load data
    if not args.input.exists():
        logger.error(f"Input file not found: {args.input}")
        return 1
    
    studies = load_unified_data(args.input, logger)
    
    # Filter studies
    included, excluded = filter_studies(
        studies, 
        logger,
        require_matrix=args.require_matrix,
        require_llm=args.require_llm,
        min_quality=args.min_quality,
        data_type_filter=args.data_type,
        min_cells=args.min_cells,
        min_samples=args.min_samples,
        exercise_only=args.exercise_only,
        tissue_filter=args.tissues,
        motrpac_tissues_only=args.motrpac_tissues_only,
        lenient_is_rat=args.lenient_is_rat,
    )
    
    # Sort by quality score
    included.sort(key=lambda s: s.quality_score, reverse=True)
    
    # Adjust output directory based on data type filter
    output_dir = args.output_dir
    if args.data_type != "both":
        # Append data type to output directory if not already there
        if args.data_type not in str(output_dir):
            output_dir = output_dir / args.data_type
    
    # Export
    export_corpus(included, excluded, output_dir, logger, 
                  data_type_filter=args.data_type)
    
    # Print summary
    print("\n" + "=" * 60)
    print("EXPORT COMPLETE")
    print("=" * 60)
    print(f"Data type filter: {args.data_type}")
    print(f"Lenient is_rat:   {args.lenient_is_rat}")
    print(f"Included: {len(included)} studies")
    print(f"Excluded: {len(excluded)} studies")
    
    # Breakdown of exclusion categories
    excl_cats = Counter(e.category for e in excluded)
    if excl_cats:
        print(f"\nExclusion breakdown:")
        for cat, count in excl_cats.most_common():
            print(f"  {cat:30} {count:>5}")
    
    if included:
        sc_count = sum(1 for s in included if s.data_type == "single_cell")
        bulk_count = sum(1 for s in included if s.data_type == "bulk")
        print(f"\nIncluded breakdown:")
        print(f"  Single-cell: {sc_count}")
        print(f"  Bulk: {bulk_count}")
        total_cells = sum(s.n_cells or 0 for s in included)
        total_samples = sum(s.n_samples or 0 for s in included)
        print(f"  Total cells: {total_cells:,}")
        print(f"  Total samples: {total_samples:,}")
    print(f"\nOutput: {output_dir}")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    exit(main())