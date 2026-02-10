#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ArrayExpress/BioStudies Harvester — Diagnostic-Informed Edition
===================================================================

A comprehensive harvester for downloading gene expression data from ArrayExpress
(now part of BioStudies at EMBL-EBI). Supports single-cell RNA-seq, bulk RNA-seq,
and microarray data with extensive pattern matching and filtering.

NOTE: ArrayExpress was migrated to BioStudies on September 30, 2022.
All ArrayExpress data is now served via the BioStudies API.

CRITICAL FIXES (Based on Diagnostic Analysis):
-----------------------------------------------------
1. EBI Search domain changed from "biostudies" to "arrayexpress"
   - "biostudies" domain returns literature references (S-EPMC*)
   - "arrayexpress" domain returns actual experiments (E-MTAB-*, E-GEOD-*)

2. Query syntax changed from field-specific to FREE TEXT
   - Field queries like organism:"Rattus norvegicus" return 0 (fields are EMPTY)
   - Free text queries like "Rattus norvegicus" AND RNA-seq work correctly

3. Added accession extraction/conversion for BioStudies format
   - S-ECPF-MTAB-1994 → E-MTAB-1994
   - S-ECPF-GEOD-58135 → E-GEOD-58135

4. Optimized query patterns for organism + technology combinations
   - Bulk: "rat AND RNA-seq NOT single cell NOT scRNA-seq"
   - Single-cell: "rat AND (single cell OR scRNA-seq OR 10x)"

FEATURES:
- Search via EBI Search API (arrayexpress domain, Lucene syntax)
- Metadata retrieval via BioStudies REST API
- MAGE-TAB (IDF/SDRF) parsing for sample metadata
- FTP and HTTPS download support
- ENA integration for raw sequencing data (FASTQ)
- Comprehensive file pattern matching (RNA-seq, scRNA-seq, microarray)
- Discovery mode to log all files before filtering
- Parallel downloads with configurable workers
- Resume support with MD5 checksums
- Detailed run reports (JSON, TSV, TXT)
- FileFilterResult system for transparent filtering decisions
- Pattern validation against known good filenames
- Sample-level supplementary file support (from SDRF)
- Recursive FTP tree walking for nested directories

USAGE EXAMPLES:
---------------
# Basic harvest of rat single-cell data
python arrayexpress_harvester.py \\
    --mode single-cell \\
    --organism "Rattus norvegicus" \\
    --download \\
    --limit 10

# Discovery mode - see all available files
python arrayexpress_harvester.py \\
    --mode bulk \\
    --organism "Rattus norvegicus" \\
    --download \\
    --discovery \\
    --limit 5

# Full harvest with parallel downloads and sample-level files
python arrayexpress_harvester.py \\
    --mode both \\
    --organism "Rattus norvegicus" \\
    --download \\
    --workers 8 \\
    --include-sample-files \\
    --verify-checksums \\
    --resume

# Include microarray and ENA data
python arrayexpress_harvester.py \\
    --mode bulk \\
    --include-microarray \\
    --include-ena \\
    --ena-download \\
    --download \\
    --limit 10

# Dry run to preview what would be downloaded
python arrayexpress_harvester.py \\
    --mode single-cell \\
    --organism "Rattus norvegicus" \\
    --download \\
    --dry-run \\
    --limit 5

# Show pattern configuration
python arrayexpress_harvester.py --show-patterns --mode single-cell

# Test if a filename matches current patterns
python arrayexpress_harvester.py --test-pattern "matrix.mtx.gz" --mode single-cell

# Test search query before running
python arrayexpress_harvester.py --test-search --mode single-cell --organism "Rattus norvegicus"

AUTHOR: MoTrPAC GeneCompass Project
VERSION: 3.2.0
"""

import os
import re
import sys
import csv
import json
import time
import gzip
import ftplib
import tarfile
import zipfile
import hashlib
import argparse
import logging
import subprocess
import urllib.request
import urllib.parse
import urllib.error
from typing import List, Dict, Optional, Tuple, Set, Any
from dataclasses import dataclass, field, asdict
from functools import wraps
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    def tqdm(iterable, **kwargs):
        return iterable


# =============================================================================
# CONSTANTS AND CONFIGURATION
# =============================================================================

# API Endpoints
BIOSTUDIES_API_BASE = "https://www.ebi.ac.uk/biostudies/api/v1"
EBI_SEARCH_API = "https://www.ebi.ac.uk/ebisearch/ws/rest"

# CRITICAL FIX: Use "arrayexpress" domain, NOT "biostudies"
# - "biostudies" returns literature references (S-EPMC*)
# - "arrayexpress" returns actual experiments (E-MTAB-*, E-GEOD-*)
EBI_SEARCH_DOMAIN = "arrayexpress"

ENA_PORTAL_API = "https://www.ebi.ac.uk/ena/portal/api"

# FTP Servers
BIOSTUDIES_FTP_HOST = "ftp.ebi.ac.uk"
BIOSTUDIES_FTP_HTTPS = "https://ftp.ebi.ac.uk"
ENA_FTP_HOST = "ftp.sra.ebi.ac.uk"

# Rate Limiting
REQUESTS_PER_SECOND = 3
MIN_REQUEST_INTERVAL = 1.0 / REQUESTS_PER_SECOND
_last_request_time = [0.0]

# Version
__version__ = "3.1.1"  # Fixed ENA paired-end fastq_bytes parsing

# ArrayExpress accession pattern
AE_ACCESSION_PATTERN = re.compile(r'^E-[A-Z]+-\d+$')

# BioStudies to ArrayExpress conversion pattern
BIOSTUDIES_ECPF_PATTERN = re.compile(r'^S-ECPF-([A-Z]+)-(\d+)$')


# =============================================================================
# RATE LIMITING
# =============================================================================

def rate_limited(func):
    """Decorator to enforce rate limiting on API calls."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        elapsed = time.time() - _last_request_time[0]
        if elapsed < MIN_REQUEST_INTERVAL:
            time.sleep(MIN_REQUEST_INTERVAL - elapsed)
        _last_request_time[0] = time.time()
        return func(*args, **kwargs)
    return wrapper


# =============================================================================
# ACCESSION EXTRACTION 
# =============================================================================

def extract_arrayexpress_accession(biostudies_id: str) -> Optional[str]:
    """
    Extract ArrayExpress accession from BioStudies ID.
    
    Handles multiple formats:
    - E-MTAB-5920 -> E-MTAB-5920 (already correct)
    - S-ECPF-MTAB-1994 -> E-MTAB-1994
    - S-ECPF-GEOD-58135 -> E-GEOD-58135
    - S-EPMC* -> None (literature, skip)
    - S-BSST* -> None (generic BioStudies, skip)
    
    Args:
        biostudies_id: The ID string from EBI Search results
        
    Returns:
        ArrayExpress accession (E-*) or None if not an experiment
    """
    if not biostudies_id:
        return None
    
    # Already an ArrayExpress accession
    if AE_ACCESSION_PATTERN.match(biostudies_id):
        return biostudies_id
    
    # BioStudies format: S-ECPF-MTAB-1994 -> E-MTAB-1994
    match = BIOSTUDIES_ECPF_PATTERN.match(biostudies_id)
    if match:
        return f"E-{match.group(1)}-{match.group(2)}"
    
    # Other BioStudies formats (literature, etc.) - skip
    return None


def is_valid_arrayexpress_accession(accession: str) -> bool:
    """Check if string is a valid ArrayExpress accession."""
    return bool(AE_ACCESSION_PATTERN.match(accession)) if accession else False


# =============================================================================
# FILE PATTERN DEFINITIONS
# =============================================================================

# Single-cell RNA-seq patterns
SC_PATTERNS = [
    # 10x Genomics / CellRanger
    r'matrix\.mtx(\.gz)?$',
    r'barcodes\.tsv(\.gz)?$',
    r'(features|genes)\.tsv(\.gz)?$',
    r'filtered_feature_bc_matrix.*\.h5$',
    r'raw_feature_bc_matrix.*\.h5$',
    r'cellranger.*\.h5$',
    r'molecule_info\.h5$',
    
    # AnnData / Scanpy
    r'\.h5ad$',
    r'_adata\.h5ad$',
    r'_processed\.h5ad$',
    r'_raw\.h5ad$',
    
    # Loom format
    r'\.loom$',
    
    # Seurat / R objects
    r'\.rds$',
    r'_sce\.rds$',
    r'_seurat\.rds$',
    r'SingleCellExperiment\.rds$',
    
    # Generic HDF5
    r'\.h5$',
    
    # Count matrices (single-cell named)
    r'(sc|single[-_]?cell).*counts?\.(tsv|csv|txt)(\.gz)?$',
    r'.*[-_]counts[-_].*\.(tsv|csv|txt)(\.gz)?$',           # tuong_apc_counts_s1_wt1.tsv
    r'.*counts?\.(tsv|csv|txt)(\.gz)?$',                    # any counts file
    
    # UMI data - PERMISSIVE
    r'(umi|unique[-_]?molecular).*counts?\.(tsv|csv|txt)(\.gz)?$',
    r'.*umi.*\.(tsv|csv|txt)(\.gz)?$',                      # ProcessedDataCorrectedUMI.txt
    r'digital[-_]?expression\.(tsv|csv|txt)(\.gz)?$',
    
    # Normalized/processed expression - PERMISSIVE
    r'.*norm.*exp.*\.(tsv|csv|txt)(\.gz)?$',                # sample1_normExp.tsv
    r'.*processed.*\.(tsv|csv|txt)(\.gz)?$',                # ProcessedData*.txt
    r'.*zscore.*\.(tsv|csv|txt)(\.gz)?$',                   # Z-score data
    
    # Drop-seq
    r'dge\.(tsv|csv|txt)(\.gz)?$',
    r'digital[-_]?gene[-_]?expression\.(tsv|csv|txt)(\.gz)?$',
    
    # Raw archives (often contain 10x output)
    r'_RAW\.tar$',
    r'_raw\.tar(\.gz)?$',
    
    # Velocyto
    r'\.vlm$',
    r'spliced\.(mtx|tsv|csv)(\.gz)?$',
    r'unspliced\.(mtx|tsv|csv)(\.gz)?$',
    
    # Cell metadata
    r'cell[-_]?(meta)?data\.(tsv|csv|txt)(\.gz)?$',
    r'cluster.*\.(tsv|csv|txt)(\.gz)?$',
]

# Bulk RNA-seq patterns
BULK_PATTERNS = [
    # Count matrices - PERMISSIVE (catches rat_counts.txt, mouse_counts.txt, etc.)
    r'.*counts?\.(tsv|csv|txt)(\.gz)?$',                    # anything ending in count(s).tsv
    r'.*[-_]counts[-_].*\.(tsv|csv|txt)(\.gz)?$',           # counts anywhere in name
    r'(gene|transcript)[-_]?counts?\.(tsv|csv|txt)(\.gz)?$',
    r'(read|fragment)[-_]?counts?\.(tsv|csv|txt)(\.gz)?$',
    r'counts?[-_]?matrix\.(tsv|csv|txt)(\.gz)?$',
    r'counts?[-_]?table\.(tsv|csv|txt)(\.gz)?$',
    r'raw[-_]?counts?\.(tsv|csv|txt)(\.gz)?$',
    
    # Normalized expression - PERMISSIVE
    r'(tpm|fpkm|rpkm|cpm)\.(tsv|csv|txt)(\.gz)?$',
    r'(tpm|fpkm|rpkm|cpm)[-_]?matrix\.(tsv|csv|txt)(\.gz)?$',
    r'.*[-_](tpm|fpkm|rpkm|cpm)\.(tsv|csv|txt)(\.gz)?$',    # rat_tpm.txt, sample_fpkm.csv
    r'normalized[-_]?(counts?|expression)?\.(tsv|csv|txt)(\.gz)?$',
    r'.*norm.*exp.*\.(tsv|csv|txt)(\.gz)?$',                # normExp, normalized_expression
    r'vst[-_]?counts?\.(tsv|csv|txt)(\.gz)?$',
    r'log[-_]?cpm\.(tsv|csv|txt)(\.gz)?$',
    
    # Expression matrices
    r'expression[-_]?(matrix|data|table)?\.(tsv|csv|txt)(\.gz)?$',
    r'.*[-_]?exp(ression)?\.(tsv|csv|txt)(\.gz)?$',         # gene_exp.tsv, expression.txt
    r'gene[-_]?expression\.(tsv|csv|txt)(\.gz)?$',
    r'transcript[-_]?expression\.(tsv|csv|txt)(\.gz)?$',
    
    # Quantification tool outputs
    r'featureCounts.*\.(txt|tsv|csv)(\.gz)?$',
    r'ReadsPerGene\.out\.tab(\.gz)?$',
    r'quant\.sf(\.gz)?$',
    r'abundance\.(tsv|h5)(\.gz)?$',
    r'rsem.*\.(genes|isoforms)\.results(\.gz)?$',
    r'htseq[-_]?counts?\.(txt|tsv)(\.gz)?$',
    
    # DESeq2 / edgeR outputs
    r'deseq2?.*\.(tsv|csv|txt)(\.gz)?$',
    r'edger.*\.(tsv|csv|txt)(\.gz)?$',
    r'diff[-_]?exp.*\.(tsv|csv|txt)(\.gz)?$',
    
    # Processed data - PERMISSIVE
    r'data[-_]?matrix\.(tsv|csv|txt)(\.gz)?$',
    r'.*processed.*\.(tsv|csv|txt)(\.gz)?$',                # ProcessedData*.txt
    r'.*zscore.*\.(tsv|csv|txt)(\.gz)?$',                   # Z-score normalized
    
    # Raw archives
    r'_RAW\.tar$',
    r'supplementary.*\.tar(\.gz)?$',
]

# Microarray patterns
MICROARRAY_PATTERNS = [
    # Affymetrix
    r'\.CEL(\.gz)?$',
    r'\.cel(\.gz)?$',
    r'\.CHP(\.gz)?$',
    r'\.chp(\.gz)?$',
    r'\.CDF$',
    r'\.cdf$',
    
    # Illumina BeadArray
    r'\.idat(\.gz)?$',
    r'\.bgx(\.gz)?$',
    r'sample[-_]?probe[-_]?profile\.(txt|tsv)(\.gz)?$',
    r'control[-_]?probe[-_]?profile\.(txt|tsv)(\.gz)?$',
    
    # Agilent
    r'US\d+.*\.txt(\.gz)?$',
    
    # Processed microarray
    r'series[-_]?matrix\.txt(\.gz)?$',
    r'non[-_]?normalized\.(txt|tsv|csv)(\.gz)?$',
    r'normalized[-_]?signal\.(txt|tsv|csv)(\.gz)?$',
    r'rma[-_]?(normalized)?\.(txt|tsv|csv)(\.gz)?$',
    r'mas5[-_]?(normalized)?\.(txt|tsv|csv)(\.gz)?$',
    r'gcrma\.(txt|tsv|csv)(\.gz)?$',
    r'processed[-_]?data[-_]?matrix\.(txt|tsv|csv)(\.gz)?$',
    r'signal[-_]?intensities?\.(txt|tsv|csv)(\.gz)?$',
]

# Generic patterns
GENERIC_PATTERNS = [
    r'.*counts?\.(tsv|csv|txt)(\.gz)?$',
    r'.*expression\.(tsv|csv|txt)(\.gz)?$',
    r'.*matrix\.(tsv|csv|txt|mtx)(\.gz)?$',
    r'.*[-_]data\.(tsv|csv|txt)(\.gz)?$',
    r'supp(lementary)?[-_]?table.*\.(tsv|csv|txt|xlsx?)(\.gz)?$',
    r'\.xlsx?$',
    r'\.RData$',
    r'\.Rda$',
    r'\.rda$',
    r'\.pkl$',
    r'\.pickle$',
    r'\.parquet$',
    r'_RAW\.tar$',
    r'[-_]raw\.tar(\.gz)?$',
    r'supplementary.*\.(tar|zip)(\.gz)?$',
]

# Metadata patterns
METADATA_PATTERNS = [
    r'sample[-_]?(info|metadata|annotation)s?\.(tsv|csv|txt|xlsx?)(\.gz)?$',
    r'(pheno|phenotype)[-_]?data\.(tsv|csv|txt)(\.gz)?$',
    r'clinical[-_]?data\.(tsv|csv|txt|xlsx?)(\.gz)?$',
    r'annotation\.(tsv|csv|txt)(\.gz)?$',
    r'design\.(tsv|csv|txt)(\.gz)?$',
]

# Skip patterns
SKIP_PATTERNS = [
    r'^filelist\.txt$',
    r'\.html?$',
    r'\.pdf$',
    r'^README',
    r'\.md5$',
    r'\.sha\d+$',
]

ADDITIONAL_BULK_PATTERNS = [
    r'.*\.processed\.\d*\.?zip$',                         # E-GEOD-509.processed.1.zip
    r'.*processed.*\.zip$',                               # anything with processed.zip
    r'.*\.additional\.\d*\.?zip$',                        # E-GEOD-67787.additional.1.zip
    r'.*\.raw\.\d*\.?zip$',                               # *.raw.1.zip
    r'.*[-_]raw[-_]?data.*\.zip$',                        # raw_data.zip
]

ADDITIONAL_SC_PATTERNS = [
    r'.*\.processed\.\d*\.?zip$',
    r'.*\.additional\.\d*\.?zip$',
]


def compile_mode_patterns(
    mode: str,
    include_microarray: bool = False,
    include_metadata: bool = True,
    include_generic: bool = False,
) -> List[re.Pattern]:
    """Compile regex patterns based on mode and options."""
    patterns = []
    
    if mode == "single-cell":
        patterns.extend(SC_PATTERNS)
        patterns.extend(ADDITIONAL_SC_PATTERNS)      # ADD THIS
    elif mode == "bulk":
        patterns.extend(BULK_PATTERNS)
        patterns.extend(ADDITIONAL_BULK_PATTERNS)    # ADD THIS
    else:  # "both"
        patterns.extend(SC_PATTERNS)
        patterns.extend(BULK_PATTERNS)
        patterns.extend(ADDITIONAL_SC_PATTERNS)      # ADD THIS
        patterns.extend(ADDITIONAL_BULK_PATTERNS)    # ADD THIS
    
    if include_microarray:
        patterns.extend(MICROARRAY_PATTERNS)
    if include_metadata:
        patterns.extend(METADATA_PATTERNS)
    if include_generic:
        patterns.extend(GENERIC_PATTERNS)
    
    patterns = list(set(patterns))
    return [re.compile(p, re.IGNORECASE) for p in patterns]

def compile_skip_patterns() -> List[re.Pattern]:
    """Compile skip patterns."""
    return [re.compile(p, re.IGNORECASE) for p in SKIP_PATTERNS]


def is_target_file(name: str, patterns: List[re.Pattern]) -> bool:
    """Check if filename matches any target pattern."""
    if not patterns:
        return True
    return any(p.search(name) for p in patterns)


def should_skip_file(name: str, skip_patterns: List[re.Pattern]) -> bool:
    """Check if filename should be skipped."""
    return any(p.search(name) for p in skip_patterns)


def get_matching_pattern(name: str, patterns: List[re.Pattern]) -> Optional[str]:
    """Return the pattern that matched."""
    for p in patterns:
        if p.search(name):
            return p.pattern
    return None


# =============================================================================
# FILE FILTER RESULT SYSTEM
# =============================================================================

@dataclass
class FileFilterResult:
    """Result of file filtering decision - provides transparency into filtering."""
    filename: str
    action: str  # "download", "skip_pattern", "skip_no_match", "skip_other"
    reason: str
    matched_pattern: Optional[str] = None


def filter_file(
    name: str,
    compiled_patterns: List[re.Pattern],
    skip_patterns: List[re.Pattern],
    selective: bool,
    discovery_mode: bool,
    archives_from_manifest: Optional[Set[str]] = None,
) -> FileFilterResult:
    """
    Centralized file filtering logic with structured output.
    
    Returns a FileFilterResult with the decision and reason.
    """
    # Check skip patterns first (always applied)
    skip_match = get_matching_pattern(name, skip_patterns)
    if skip_match:
        return FileFilterResult(
            filename=name,
            action="skip_pattern",
            reason=f"Matches skip pattern: {skip_match}",
            matched_pattern=skip_match,
        )
    
    # Discovery mode: download everything not skipped
    if discovery_mode:
        return FileFilterResult(
            filename=name,
            action="download",
            reason="Discovery mode: downloading all files",
        )
    
    # Non-selective mode: download everything not skipped
    if not selective:
        return FileFilterResult(
            filename=name,
            action="download",
            reason="Non-selective mode: downloading all files",
        )
    
    # Check if file is in manifest archives
    if archives_from_manifest and name in archives_from_manifest:
        return FileFilterResult(
            filename=name,
            action="download",
            reason="Archive contains matching files (from manifest)",
        )
    
    # Check target patterns
    matched = get_matching_pattern(name, compiled_patterns)
    if matched:
        return FileFilterResult(
            filename=name,
            action="download",
            reason="Matches target pattern",
            matched_pattern=matched,
        )
    
    # No match
    return FileFilterResult(
        filename=name,
        action="skip_no_match",
        reason=f"No pattern match (checked {len(compiled_patterns)} patterns)",
    )


def log_filter_decision(result: FileFilterResult, accession: str, log_level: int = logging.DEBUG):
    """Log file filtering decision with consistent formatting."""
    if result.action == "download":
        if result.matched_pattern:
            logging.log(log_level, f"{accession}:   ✓ {result.filename} -> {result.matched_pattern}")
        else:
            logging.log(log_level, f"{accession}:   ✓ {result.filename} ({result.reason})")
    else:
        logging.log(log_level, f"{accession}:   ✗ {result.filename} ({result.reason})")


# =============================================================================
# PATTERN VALIDATION
# =============================================================================

def validate_patterns(patterns: List[re.Pattern], mode: str = "both") -> bool:
    """
    Validate that patterns are properly compiled against known good filenames.
    
    Tests patterns against a set of known filenames that should/shouldn't match.
    Returns True if all tests pass.
    """
    if not patterns:
        logging.warning("No patterns loaded - will download all files!")
        return False
    
    # Test patterns against known good filenames (mode-specific)
    sc_test_files = {
        "matrix.mtx.gz": True,
        "barcodes.tsv.gz": True,
        "features.tsv.gz": True,
        "genes.tsv.gz": True,
        "sample.h5ad": True,
        "data.loom": True,
        "seurat.rds": True,
        "filtered_feature_bc_matrix.h5": True,
    }
    
    bulk_test_files = {
        "gene_counts.tsv.gz": True,
        "tpm.csv": True,
        "fpkm.txt.gz": True,
        "expression_matrix.txt.gz": True,
        "normalized_counts.tsv": True,
    }
    
    common_skip_files = {
        "README.txt": False,
        "filelist.txt": False,
        "index.html": False,
    }
    
    # Select test files based on mode
    test_files = {}
    if mode in ("single-cell", "both"):
        test_files.update(sc_test_files)
    if mode in ("bulk", "both"):
        test_files.update(bulk_test_files)
    test_files.update(common_skip_files)
    
    all_pass = True
    for filename, should_match in test_files.items():
        matches = any(p.search(filename) for p in patterns)
        if matches != should_match:
            logging.warning(f"Pattern validation: '{filename}' match={matches}, expected={should_match}")
            all_pass = False
    
    if all_pass:
        logging.debug(f"Pattern validation: all {len(test_files)} tests passed")
    else:
        logging.warning("Pattern validation: some tests failed - check pattern configuration")
    
    return all_pass


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class DownloadEvent:
    """Record of a single download attempt."""
    timestamp: str
    accession: str
    filename: str
    source: str  # "biostudies", "ena_fastq", "mage_tab", "sample_suppl"
    status: str  # "success", "failed", "skipped", "exists"
    size_bytes: Optional[int] = None
    duration_sec: Optional[float] = None
    error_message: Optional[str] = None
    md5: Optional[str] = None


@dataclass
class StudyProcessingResult:
    """Summary of processing for a single study."""
    accession: str
    start_time: str
    end_time: str
    duration_sec: float
    status: str  # "success", "partial", "failed"
    
    title: str = ""
    organism: str = ""
    sample_count: int = 0
    
    files_downloaded: int = 0
    files_skipped: int = 0
    files_failed: int = 0
    total_bytes_downloaded: int = 0
    
    # Sample-level files
    sample_files_downloaded: int = 0
    sample_files_skipped: int = 0
    
    ena_runs_found: int = 0
    ena_fastqs_downloaded: int = 0
    
    errors: List[str] = field(default_factory=list)
    
    # Filter results tracking
    filter_results: List[FileFilterResult] = field(default_factory=list)


@dataclass
class HarvestRunReport:
    """Complete report for a harvesting run."""
    run_id: str
    start_time: str
    end_time: Optional[str] = None
    duration_sec: Optional[float] = None
    
    # Parameters
    mode: str = ""
    organism: str = ""
    search_term: str = ""
    output_dir: str = ""
    
    # Flags
    include_microarray: bool = False
    include_generic: bool = False
    include_ena: bool = False
    ena_download: bool = False
    include_sample_files: bool = False
    discovery_mode: bool = False
    
    # Results
    total_studies_found: int = 0
    total_studies_processed: int = 0
    total_studies_succeeded: int = 0
    total_studies_failed: int = 0
    
    total_files_downloaded: int = 0
    total_files_skipped: int = 0
    total_files_failed: int = 0
    total_bytes_downloaded: int = 0
    
    total_sample_files_downloaded: int = 0
    total_ena_runs: int = 0
    total_ena_fastqs: int = 0
    
    study_results: List[StudyProcessingResult] = field(default_factory=list)
    download_events: List[DownloadEvent] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


@dataclass
class ENARunInfo:
    """Information about a single ENA run."""
    run_accession: str
    experiment_accession: str
    sample_accession: str
    study_accession: str
    library_layout: str = ""
    library_source: str = ""
    library_strategy: str = ""
    fastq_ftp: str = ""
    fastq_bytes: int = 0


@dataclass
class MAGETABMetadata:
    """Parsed MAGE-TAB metadata."""
    accession: str
    title: str = ""
    description: str = ""
    pubmed_id: str = ""
    organism: str = ""
    experiment_type: str = ""
    release_date: str = ""
    
    # From IDF
    idf_fields: Dict[str, List[str]] = field(default_factory=dict)
    
    # From SDRF
    sample_count: int = 0
    sdrf_columns: List[str] = field(default_factory=list)
    samples_df_path: Optional[str] = None
    
    # Sample-level files extracted from SDRF
    sample_files: List['SampleSupplementaryFile'] = field(default_factory=list)


@dataclass
class SampleSupplementaryFile:
    """A supplementary file linked to a specific sample (from SDRF)."""
    sample_name: str
    filename: str
    ftp_path: str
    comment_field: str = ""  # The SDRF column name this came from
    size: Optional[int] = None


@dataclass 
class FTPTreeEntry:
    """Entry from FTP tree listing."""
    rel_path: str
    is_dir: bool
    size: Optional[int] = None


@dataclass
class FTPFileInfo:
    """Information about a file on FTP."""
    filename: str
    size: Optional[int] = None
    is_dir: bool = False

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def generate_run_id() -> str:
    """Generate a unique run ID."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def get_timestamp() -> str:
    """Get current ISO timestamp."""
    return datetime.now().isoformat()


def ensure_dir(path: str):
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def md5sum(path: str, chunk_size: int = 1 << 20) -> str:
    """Calculate MD5 checksum of a file."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def write_md5_file(path: str, digest: str):
    """Write MD5 checksum to companion file."""
    try:
        with open(f"{path}.md5", "w") as f:
            f.write(digest + "\n")
    except Exception as e:
        logging.debug(f"Could not write MD5 for {path}: {e}")


def read_md5_file(path: str) -> Optional[str]:
    """Read MD5 checksum from companion file."""
    try:
        with open(f"{path}.md5", "r") as f:
            return f.read().strip()
    except Exception:
        return None


def setup_logging(
    log_level: str = "INFO",
    log_dir: Optional[str] = None,
    log_file: Optional[str] = None,
):
    """Configure logging."""
    level = getattr(logging, log_level.upper(), logging.INFO)
    fmt = "%(asctime)s [%(levelname)s] %(message)s"
    
    handlers = [logging.StreamHandler()]
    
    if log_file:
        if os.path.dirname(log_file):
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file, mode='a', encoding='utf-8'))
    elif log_dir:
        os.makedirs(log_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(log_dir, f"arrayexpress_harvest_{timestamp}.log")
        handlers.append(logging.FileHandler(log_path, mode='a', encoding='utf-8'))
        print(f"Logging to: {log_path}")
    
    logging.basicConfig(level=level, format=fmt, handlers=handlers)


def setup_file_logging(output_dir: str, run_id: str, log_level: str = "DEBUG") -> str:
    """Set up file-based logging."""
    ensure_dir(output_dir)
    log_file = os.path.join(output_dir, f"harvest_{run_id}.log")
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(getattr(logging, log_level.upper(), logging.DEBUG))
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    
    logging.getLogger().addHandler(file_handler)
    logging.info(f"Logging to file: {log_file}")
    
    return log_file


# =============================================================================
# HTTP / API FUNCTIONS
# =============================================================================

@rate_limited
def fetch_json(url: str, timeout: int = 30, retries: int = 3) -> Optional[Dict]:
    """Fetch JSON from URL with rate limiting and retries."""
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url)
            req.add_header("User-Agent", f"ArrayExpress-Harvester/{__version__}")
            req.add_header("Accept", "application/json")
            
            with urllib.request.urlopen(req, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
                
        except urllib.error.HTTPError as e:
            logging.warning(f"HTTP {e.code} for {url[:80]}... (attempt {attempt}/{retries})")
            if attempt < retries:
                time.sleep(2 * attempt)
        except Exception as e:
            logging.warning(f"Error fetching {url[:80]}...: {e} (attempt {attempt}/{retries})")
            if attempt < retries:
                time.sleep(2 * attempt)
    
    return None


@rate_limited
def fetch_text(url: str, timeout: int = 30, retries: int = 3) -> Optional[str]:
    """Fetch text content from URL."""
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url)
            req.add_header("User-Agent", f"ArrayExpress-Harvester/{__version__}")
            
            with urllib.request.urlopen(req, timeout=timeout) as response:
                return response.read().decode("utf-8", errors="replace")
                
        except Exception as e:
            logging.warning(f"Error fetching {url[:80]}...: {e} (attempt {attempt}/{retries})")
            if attempt < retries:
                time.sleep(2 * attempt)
    
    return None


# =============================================================================
# EBI SEARCH API 
# =============================================================================

def search_arrayexpress(
    query: str,
    size: int = 100,
    start: int = 0,
    retries: int = 3,
) -> Tuple[List[Dict], int]:
    """
    Search ArrayExpress via EBI Search API.
    
    CRITICAL: Uses "arrayexpress" domain, NOT "biostudies"!
    - "biostudies" returns literature (S-EPMC*)
    - "arrayexpress" returns experiments (E-MTAB-*, E-GEOD-*)
    
    Args:
        query: Lucene query string (use FREE TEXT, not field-specific)
        size: Number of results per page (max 100)
        start: Offset for pagination
        retries: Number of retry attempts
        
    Returns:
        Tuple of (list of entries, total hit count)
    """
    params = {
        "query": query,
        "format": "json",
        "size": str(size),
        "start": str(start),
    }
    
    url = f"{EBI_SEARCH_API}/{EBI_SEARCH_DOMAIN}?" + urllib.parse.urlencode(params)
    
    result = fetch_json(url, retries=retries)
    
    if result is None:
        return [], 0
    
    total_hits = result.get("hitCount", 0)
    entries = result.get("entries", [])
    
    return entries, total_hits


def fetch_all_accessions(query: str, limit: int = 0) -> List[str]:
    """
    Fetch all ArrayExpress accessions matching the search query.
    
    Handles both direct ArrayExpress IDs and BioStudies format conversion.
    
    Args:
        query: Lucene query string
        limit: Maximum number of accessions to return (0 = no limit)
        
    Returns:
        List of unique ArrayExpress accessions (E-MTAB-*, E-GEOD-*, etc.)
    """
    all_accessions: List[str] = []
    page_size = 100
    start = 0
    
    # First search to get total count
    entries, total_count = search_arrayexpress(query, size=page_size, start=start)
    
    if total_count == 0:
        logging.warning(f"No results found for query: {query}")
        return []
    
    logging.info(f"Found {total_count} total results in ArrayExpress")
    
    # Process first page
    for entry in entries:
        entry_id = entry.get("id", "")
        
        # Try to extract ArrayExpress accession
        ae_acc = extract_arrayexpress_accession(entry_id)
        if ae_acc:
            all_accessions.append(ae_acc)
        else:
            # Check fields for accession (sometimes nested differently)
            fields = entry.get("fields", {})
            for key, values in fields.items():
                if isinstance(values, list):
                    for v in values:
                        if isinstance(v, str):
                            ae = extract_arrayexpress_accession(v)
                            if ae:
                                all_accessions.append(ae)
                                break
    
    start += page_size
    
    # Fetch remaining pages
    while start < total_count:
        if limit > 0 and len(all_accessions) >= limit:
            break
        
        entries, _ = search_arrayexpress(query, size=page_size, start=start)
        
        for entry in entries:
            entry_id = entry.get("id", "")
            ae_acc = extract_arrayexpress_accession(entry_id)
            if ae_acc:
                all_accessions.append(ae_acc)
            else:
                fields = entry.get("fields", {})
                for key, values in fields.items():
                    if isinstance(values, list):
                        for v in values:
                            if isinstance(v, str):
                                ae = extract_arrayexpress_accession(v)
                                if ae:
                                    all_accessions.append(ae)
                                    break
        
        start += page_size
        time.sleep(MIN_REQUEST_INTERVAL)
    
    # Deduplicate while preserving order
    all_accessions = list(dict.fromkeys(all_accessions))
    
    if limit > 0:
        all_accessions = all_accessions[:limit]
    
    logging.info(f"Retrieved {len(all_accessions)} unique ArrayExpress accessions")
    return all_accessions


def get_study_metadata(accession: str) -> Optional[Dict]:
    """Fetch study metadata from BioStudies API."""
    url = f"{BIOSTUDIES_API_BASE}/studies/{accession}"
    return fetch_json(url)


def get_study_info(accession: str) -> Optional[Dict]:
    """Fetch study info including FTP link."""
    url = f"{BIOSTUDIES_API_BASE}/studies/{accession}/info"
    return fetch_json(url)


# =============================================================================
# SEARCH QUERY BUILDING 
# =============================================================================

# CRITICAL FIX: Use FREE TEXT queries, NOT field-specific queries
# Field queries like organism:"Rattus norvegicus" return 0 because fields are EMPTY!

# Single-cell query terms (optimized based on diagnostic analysis)
SC_QUERY_TERMS = (
    '"single cell" OR scRNA-seq OR "single nucleus" OR snRNA-seq OR '
    '"10x genomics" OR drop-seq OR smart-seq'
)

# Bulk RNA-seq query terms
BULK_QUERY_TERMS = (
    'RNA-seq OR "bulk RNA" OR transcriptome OR "mRNA sequencing"'
)

# Organism-specific mappings (use simple names that work with free text)
ORGANISM_QUERY_MAP = {
    "Rattus norvegicus": "rat",
    "rattus norvegicus": "rat",
    "Mus musculus": "mouse",
    "mus musculus": "mouse", 
    "Homo sapiens": "human",
    "homo sapiens": "human",
}


def build_search_term(
    mode: str,
    organism: str,
    user_term: Optional[str] = None,
) -> str:
    """
    Build search query for EBI Search API (arrayexpress domain).
    
    CRITICAL: Uses FREE TEXT queries, NOT field-specific queries!
    - DON'T USE: organism:"Rattus norvegicus" (returns 0!)
    - DO USE: rat AND RNA-seq (works correctly)
    
    Args:
        mode: "single-cell", "bulk", or "both"
        organism: Organism name (will be simplified for query)
        user_term: Custom query (overrides mode/organism)
        
    Returns:
        Lucene query string optimized for EBI Search arrayexpress domain
    """
    if user_term:
        return user_term
    
    # Build technology part of query
    if mode == "single-cell":
        tech_query = f"({SC_QUERY_TERMS})"
    elif mode == "bulk":
        # Exclude single-cell from bulk queries
        tech_query = f"(RNA-seq OR transcriptome) NOT ({SC_QUERY_TERMS})"
    else:  # "both"
        tech_query = f"(RNA-seq OR transcriptome OR {SC_QUERY_TERMS})"
    
    # Add organism filter using simple name (free text)
    if organism:
        # Use simple organism name for better matching
        simple_organism = ORGANISM_QUERY_MAP.get(organism, organism)
        
        # Also include scientific name in quotes for exact match
        if organism != simple_organism:
            term = f'({simple_organism} OR "{organism}") AND {tech_query}'
        else:
            term = f'{simple_organism} AND {tech_query}'
    else:
        term = tech_query
    
    return term


def test_search_query(query: str, show_samples: bool = True) -> Tuple[int, List[str]]:
    """
    Test a search query and show results.
    
    Args:
        query: Lucene query string
        show_samples: Whether to show sample accessions
        
    Returns:
        Tuple of (total hits, sample accessions)
    """
    entries, total_count = search_arrayexpress(query, size=20)
    
    accessions = []
    for entry in entries:
        entry_id = entry.get("id", "")
        ae_acc = extract_arrayexpress_accession(entry_id)
        if ae_acc:
            accessions.append(ae_acc)
    
    return total_count, accessions


# =============================================================================
# FTP PATH CONSTRUCTION
# =============================================================================

def construct_ftp_path(accession: str) -> str:
    """
    Construct FTP path from ArrayExpress accession.
    
    Pattern: /biostudies/fire/{prefix}/{last3digits}/{accession}/Files/
    Example: E-MTAB-6798 -> /biostudies/fire/E-MTAB-/798/E-MTAB-6798/Files/
    """
    prefix = accession.rsplit('-', 1)[0] + '-'
    number = accession.rsplit('-', 1)[1]
    last_three = number[-3:].zfill(3)
    
    return f"/biostudies/fire/{prefix}/{last_three}/{accession}/Files"


def construct_https_ftp_url(accession: str, filename: str = "") -> str:
    """Construct HTTPS URL for FTP content."""
    prefix = accession.rsplit('-', 1)[0] + '-'
    number = accession.rsplit('-', 1)[1]
    last_three = number[-3:].zfill(3)
    
    base = f"{BIOSTUDIES_FTP_HTTPS}/biostudies/fire/{prefix}/{last_three}/{accession}/Files"
    
    if filename:
        return f"{base}/{filename}"
    return base


def construct_legacy_ftp_path(accession: str) -> str:
    """
    Construct legacy ArrayExpress FTP path.
    
    Legacy paths follow the pattern:
    /pub/databases/microarray/data/experiment/{TYPE}/{accession}/
    
    Where TYPE is extracted from accession:
    - E-GEOD-509 -> GEOD
    - E-MTAB-6081 -> MTAB
    """
    parts = accession.split('-')
    if len(parts) >= 2:
        acc_type = parts[1]  # GEOD, MTAB, MEXP, etc.
    else:
        acc_type = "MTAB"
    
    return f"/pub/databases/microarray/data/experiment/{acc_type}/{accession}"


def construct_legacy_ftp_url(accession: str, filename: str = "") -> str:
    """Construct full FTP URL for legacy ArrayExpress path."""
    base_path = construct_legacy_ftp_path(accession)
    if filename:
        return f"ftp://ftp.ebi.ac.uk{base_path}/{filename}"
    return f"ftp://ftp.ebi.ac.uk{base_path}/"


def list_legacy_ftp_files(accession: str, timeout: int = 60) -> List[FTPFileInfo]:
    """
    List files on legacy ArrayExpress FTP path.
    
    Returns:
        List of FTPFileInfo objects
    """
    remote_path = construct_legacy_ftp_path(accession)
    files: List[FTPFileInfo] = []
    
    try:
        ftp = ftplib.FTP("ftp.ebi.ac.uk", timeout=timeout)
        ftp.login()
        ftp.set_pasv(True)
        
        try:
            ftp.cwd(remote_path)
        except ftplib.error_perm as e:
            if "550" in str(e):
                logging.debug(f"{accession}: No legacy FTP directory at {remote_path}")
                return files
            raise
        
        lines: List[str] = []
        ftp.retrlines("LIST", lines.append)
        
        for line in lines:
            parts = line.split(None, 8)
            if len(parts) < 9:
                continue
            
            is_dir = line.startswith("d")
            try:
                size = int(parts[4]) if not is_dir else None
            except (ValueError, IndexError):
                size = None
            
            filename = parts[8]
            if filename in (".", ".."):
                continue
            
            files.append(FTPFileInfo(
                filename=filename,
                size=size,
                is_dir=is_dir,
            ))
        
        ftp.quit()
        
    except ftplib.error_perm as e:
        if "550" in str(e):
            logging.debug(f"{accession}: Legacy FTP path not accessible: {remote_path}")
        else:
            logging.warning(f"{accession}: FTP error listing legacy path: {e}")
    except Exception as e:
        logging.warning(f"{accession}: Error listing legacy FTP: {e}")
    
    return files



def download_via_ftp(ftp_url: str, local_path: str, timeout: int = 1800) -> bool:
    """
    Download file via FTP protocol (for legacy ArrayExpress URLs).
    
    Returns:
        True if download succeeded, False otherwise
    """
    parsed = urllib.parse.urlparse(ftp_url)
    host = parsed.hostname
    remote_path = parsed.path
    
    if not host or not remote_path:
        logging.warning(f"Invalid FTP URL: {ftp_url}")
        return False
    
    directory = os.path.dirname(remote_path)
    filename = os.path.basename(remote_path)
    
    if not filename:
        logging.warning(f"No filename in FTP URL: {ftp_url}")
        return False
    
    tmp_path = local_path + ".part"
    
    try:
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        
        ftp = ftplib.FTP(host, timeout=timeout)
        ftp.login()
        ftp.set_pasv(True)
        ftp.cwd(directory)
        
        with open(tmp_path, "wb") as f:
            ftp.retrbinary(f"RETR {filename}", f.write, blocksize=1024*1024)
        
        ftp.quit()
        
        if os.path.exists(tmp_path) and os.path.getsize(tmp_path) > 0:
            os.replace(tmp_path, local_path)
            return True
        return False
        
    except Exception as e:
        logging.warning(f"FTP download failed for {ftp_url}: {e}")
        return False
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except:
                pass


def download_file_with_fallback(
    url: str,
    local_path: str,
    timeout: int = 1800,
    verify_checksums: bool = False,
) -> bool:
    """
    Download file with HTTPS-first, FTP-fallback strategy.
    
    Returns:
        True if download succeeded
    """
    tmp_path = local_path + ".part"
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    
    success = False
    
    # Strategy 1: If FTP URL, try HTTPS conversion first
    if url.startswith("ftp://ftp.ebi.ac.uk/"):
        https_url = url.replace("ftp://ftp.ebi.ac.uk/", "https://ftp.ebi.ac.uk/")
        try:
            req = urllib.request.Request(https_url)
            req.add_header("User-Agent", f"ArrayExpress-Harvester/{__version__}")
            with urllib.request.urlopen(req, timeout=timeout) as response:
                with open(tmp_path, "wb") as f:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        f.write(chunk)
            if os.path.exists(tmp_path) and os.path.getsize(tmp_path) > 0:
                os.replace(tmp_path, local_path)
                success = True
        except urllib.error.HTTPError as e:
            logging.debug(f"HTTPS {e.code}, will try FTP: {os.path.basename(local_path)}")
        except Exception as e:
            logging.debug(f"HTTPS failed ({e}), will try FTP: {os.path.basename(local_path)}")
    
    # Strategy 2: Direct HTTPS
    if not success and url.startswith("https://"):
        try:
            req = urllib.request.Request(url)
            req.add_header("User-Agent", f"ArrayExpress-Harvester/{__version__}")
            with urllib.request.urlopen(req, timeout=timeout) as response:
                with open(tmp_path, "wb") as f:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        f.write(chunk)
            if os.path.exists(tmp_path) and os.path.getsize(tmp_path) > 0:
                os.replace(tmp_path, local_path)
                success = True
        except Exception as e:
            logging.debug(f"HTTPS download failed: {e}")
    
    # Strategy 3: True FTP download
    if not success and url.startswith("ftp://"):
        success = download_via_ftp(url, local_path, timeout)
        if success:
            logging.info(f"Downloaded via FTP: {os.path.basename(local_path)}")
    
    # Cleanup
    if os.path.exists(tmp_path):
        try:
            os.remove(tmp_path)
        except:
            pass
    
    # Checksum
    if success and verify_checksums:
        try:
            write_md5_file(local_path, md5sum(local_path))
        except:
            pass
    
    return success


# =============================================================================
# RECURSIVE FTP TREE WALKING
# =============================================================================

def list_ftp_tree(host: str, remote_path: str, timeout: int = 60) -> List[FTPTreeEntry]:
    """
    Recursively list all files and directories under remote_path.
    
    Returns list of FTPTreeEntry with relative paths.
    """
    results: List[FTPTreeEntry] = []
    
    try:
        ftp = ftplib.FTP(host, timeout=timeout)
        ftp.login()
        ftp.set_pasv(True)
        
        try:
            ftp.cwd(remote_path)
        except ftplib.error_perm as e:
            if "550" in str(e):
                logging.debug(f"No files at {remote_path}")
                return results
            raise
        
        def _walk(prefix: str = ""):
            items: List[str] = []
            ftp.retrlines("LIST", items.append)
            
            for item in items:
                parts = item.split(None, 8)
                if len(parts) < 9:
                    continue
                
                name = parts[8]
                is_dir = item.startswith("d")
                rel = f"{prefix}{name}" if prefix else name
                
                # Try to parse size
                try:
                    size = int(parts[4]) if not is_dir else None
                except (ValueError, IndexError):
                    size = None
                
                results.append(FTPTreeEntry(rel_path=rel, is_dir=is_dir, size=size))
                
                if is_dir:
                    cur = ftp.pwd()
                    try:
                        ftp.cwd(name)
                        _walk(prefix=f"{rel}/")
                    except ftplib.error_perm:
                        pass  # Skip directories we can't access
                    finally:
                        ftp.cwd(cur)
        
        _walk("")
        ftp.quit()
        
    except Exception as e:
        logging.warning(f"Error listing FTP tree at {remote_path}: {e}")
    
    return results


def list_ftp_files(accession: str, timeout: int = 60) -> List[str]:
    """List files available for a study via FTP (simple NLST)."""
    remote_path = construct_ftp_path(accession)
    files: List[str] = []
    
    try:
        ftp = ftplib.FTP(BIOSTUDIES_FTP_HOST, timeout=timeout)
        ftp.login()
        ftp.set_pasv(True)
        ftp.cwd(remote_path)
        ftp.retrlines("NLST", files.append)
        ftp.quit()
    except ftplib.error_perm as e:
        if "550" in str(e):
            logging.debug(f"{accession}: No files at {remote_path}")
        else:
            logging.warning(f"{accession}: FTP error: {e}")
    except Exception as e:
        logging.warning(f"{accession}: FTP error: {e}")
    
    return files


def list_ftp_files_recursive(accession: str, timeout: int = 60) -> List[FTPTreeEntry]:
    """
    List files recursively for a study via FTP.
    
    Returns list of FTPTreeEntry objects with relative paths.
    """
    remote_path = construct_ftp_path(accession)
    return list_ftp_tree(BIOSTUDIES_FTP_HOST, remote_path, timeout)


# =============================================================================
# MAGE-TAB PARSING
# =============================================================================

def parse_idf(content: str) -> Dict[str, List[str]]:
    """Parse IDF (Investigation Description Format) content."""
    idf_data = {}
    
    for line in content.split("\n"):
        if "\t" in line:
            parts = line.split("\t")
            key = parts[0].strip()
            values = [p.strip() for p in parts[1:] if p.strip()]
            if key and values:
                idf_data[key] = values
    
    return idf_data


def parse_sdrf(content: str) -> Tuple[List[str], List[Dict[str, str]]]:
    """
    Parse SDRF (Sample and Data Relationship Format) content.
    
    Returns:
        Tuple of (column headers, list of sample dicts)
    """
    lines = content.strip().split("\n")
    
    if len(lines) < 2:
        return [], []
    
    headers = lines[0].split("\t")
    samples = []
    
    for line in lines[1:]:
        if not line.strip():
            continue
        values = line.split("\t")
        sample = dict(zip(headers, values))
        samples.append(sample)
    
    return headers, samples


def extract_sample_files_from_sdrf(
    headers: List[str],
    samples: List[Dict[str, str]],
    accession: str,
) -> List[SampleSupplementaryFile]:
    """
    Extract sample-level supplementary files from SDRF.
    
    Looks for columns like:
    - Comment[ArrayExpress FTP file]
    - Comment[Derived ArrayExpress FTP file]
    - Comment[FASTQ_URI]
    - Array Data File
    - Derived Array Data File
    """
    sample_files: List[SampleSupplementaryFile] = []
    
    # Columns that may contain file references
    file_columns = [h for h in headers if any(keyword in h.lower() for keyword in [
        'ftp file', 'array data file', 'fastq', 'file', 'uri', 'url'
    ])]
    
    for sample in samples:
        sample_name = sample.get("Source Name", "") or sample.get("Sample Name", "")
        
        for col in file_columns:
            value = sample.get(col, "").strip()
            if not value:
                continue
            
            # Check if it's a URL/FTP path
            if value.startswith(("ftp://", "http://", "https://")):
                filename = os.path.basename(value)
                sample_files.append(SampleSupplementaryFile(
                    sample_name=sample_name,
                    filename=filename,
                    ftp_path=value,
                    comment_field=col,
                ))
            elif not value.startswith("/") and "." in value:
                # Assume it's a filename, construct FTP path
                ftp_path = construct_https_ftp_url(accession, value)
                sample_files.append(SampleSupplementaryFile(
                    sample_name=sample_name,
                    filename=value,
                    ftp_path=ftp_path,
                    comment_field=col,
                ))
    
    return sample_files


def fetch_mage_tab(accession: str, output_dir: str) -> MAGETABMetadata:
    """
    Fetch and parse MAGE-TAB files (IDF and SDRF) for a study.
    """
    metadata = MAGETABMetadata(accession=accession)
    
    # Download IDF
    idf_url = construct_https_ftp_url(accession, f"{accession}.idf.txt")
    idf_content = fetch_text(idf_url)
    
    if idf_content:
        idf_data = parse_idf(idf_content)
        metadata.idf_fields = idf_data
        
        # Extract key fields
        metadata.title = idf_data.get("Investigation Title", [""])[0]
        metadata.description = idf_data.get("Experiment Description", [""])[0]
        metadata.pubmed_id = ";".join(idf_data.get("PubMed ID", []))
        metadata.experiment_type = ";".join(idf_data.get("Experimental Design", []))
        metadata.release_date = idf_data.get("Public Release Date", [""])[0]
        
        # Save IDF
        idf_path = os.path.join(output_dir, f"{accession}.idf.txt")
        with open(idf_path, "w", encoding="utf-8") as f:
            f.write(idf_content)
        logging.debug(f"{accession}: Saved IDF to {idf_path}")
    else:
        logging.warning(f"{accession}: Could not fetch IDF file")
    
    # Download SDRF
    sdrf_url = construct_https_ftp_url(accession, f"{accession}.sdrf.txt")
    sdrf_content = fetch_text(sdrf_url)
    
    if sdrf_content:
        headers, samples = parse_sdrf(sdrf_content)
        metadata.sdrf_columns = headers
        metadata.sample_count = len(samples)
        
        # Extract organism from SDRF if available
        organism_cols = [h for h in headers if "organism" in h.lower()]
        if organism_cols and samples:
            orgs = set()
            for s in samples:
                for col in organism_cols:
                    if col in s and s[col]:
                        orgs.add(s[col])
            metadata.organism = ";".join(orgs)
        
        # Extract sample-level files
        metadata.sample_files = extract_sample_files_from_sdrf(headers, samples, accession)
        if metadata.sample_files:
            logging.info(f"{accession}: Found {len(metadata.sample_files)} sample-level files in SDRF")
        
        # Save SDRF
        sdrf_path = os.path.join(output_dir, f"{accession}.sdrf.txt")
        with open(sdrf_path, "w", encoding="utf-8") as f:
            f.write(sdrf_content)
        metadata.samples_df_path = sdrf_path
        logging.debug(f"{accession}: Saved SDRF ({metadata.sample_count} samples)")
    else:
        logging.warning(f"{accession}: Could not fetch SDRF file")
    
    return metadata


# =============================================================================
# SAMPLE-LEVEL FILE HANDLING
# =============================================================================


def download_sample_supplementary_file(
    sample_file: 'SampleSupplementaryFile',
    local_dir: str,
    compiled_patterns: List[re.Pattern],
    skip_patterns: List[re.Pattern],
    selective: bool = True,
    discovery_mode: bool = False,
    verify_checksums: bool = False,
    resume: bool = False,
    timeout: int = 1800,
) -> Tuple[Optional[str], 'FileFilterResult']:
    """
    Download a single sample supplementary file.
    
    FIXES:
    - Proper HTTP error handling (catches 404, 403, etc.)
    - URL validation before attempting download
    - FTP fallback for legacy URLs that can't be HTTPS-converted
    - Flat directory structure
    
    Returns tuple of (local_path or None, FileFilterResult).
    """
    from arrayexpress_harvester import filter_file, FileFilterResult, read_md5_file, md5sum, write_md5_file
    
    # Filter decision
    result = filter_file(
        sample_file.filename,
        compiled_patterns,
        skip_patterns,
        selective,
        discovery_mode,
    )
    
    if result.action != "download":
        return None, result
    
    # Validate URL
    if not sample_file.ftp_path:
        logging.debug(f"No URL for sample file {sample_file.filename}")
        return None, result
    
    url = sample_file.ftp_path
    
    # Skip invalid/old URLs
    if url.startswith("http://www.ebi.ac.uk/aerep"):
        logging.debug(f"Skipping deprecated aerep URL: {sample_file.filename}")
        return None, result
    
    if not url.startswith(("http://", "https://", "ftp://")):
        logging.debug(f"Invalid URL for sample file {sample_file.filename}: {url}")
        return None, result
    
    # Use flat directory structure
    local_path = os.path.join(local_dir, sample_file.filename)
    
    # Check if already exists
    if os.path.exists(local_path):
        if resume:
            logging.debug(f"Skipping {sample_file.filename} - already exists (resume mode)")
            return local_path, result
        if verify_checksums:
            existing_md5 = read_md5_file(local_path)
            if existing_md5:
                logging.debug(f"Skipping {sample_file.filename} - checksum exists")
                return local_path, result
    
    # Download with fallback
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    
    if download_file_with_fallback(url, local_path, timeout, verify_checksums):
        return local_path, result
    
    return None, result


def download_all_sample_files(
    sample_files: List['SampleSupplementaryFile'],
    local_dir: str,
    compiled_patterns: List[re.Pattern],
    skip_patterns: List[re.Pattern],
    selective: bool = True,
    discovery_mode: bool = False,
    verify_checksums: bool = False,
    resume: bool = False,
    workers: int = 0,
    dry_run: bool = False,
    study_files_downloaded: Optional[List[str]] = None,
) -> Tuple[List[str], List[str], List['FileFilterResult']]:
    """
    Download all sample-level supplementary files.
    
    FIXES:
    - Deduplicates files by URL (same file referenced by multiple samples)
    - Skips files already downloaded at study level
    - Adds rate limiting between downloads
    - FTP fallback for legacy URLs
    
    Args:
        sample_files: List of sample supplementary files from SDRF
        local_dir: Directory to save files
        compiled_patterns: Compiled regex patterns for file matching
        skip_patterns: Compiled skip patterns
        selective: Whether to apply pattern filtering
        discovery_mode: Whether to download all files
        verify_checksums: Whether to verify/save MD5 checksums
        resume: Whether to skip existing files
        workers: Number of parallel workers (0 = sequential)
        dry_run: Whether to simulate without downloading
        study_files_downloaded: List of files already downloaded at study level
    
    Returns:
        - List of downloaded file paths
        - List of skipped filenames
        - List of all FileFilterResults
    """
    from arrayexpress_harvester import filter_file, FileFilterResult, ensure_dir
    
    downloaded: List[str] = []
    skipped: List[str] = []
    filter_results: List[FileFilterResult] = []
    
    if not sample_files:
        return downloaded, skipped, filter_results
    
    # === DEDUPLICATION BY URL ===
    seen_urls: Dict[str, 'SampleSupplementaryFile'] = {}
    for sf in sample_files:
        # Normalize URL for comparison
        url_key = sf.ftp_path.lower().strip() if sf.ftp_path else sf.filename.lower()
        
        # Skip deprecated URLs
        if "aerep" in url_key:
            continue
            
        if url_key not in seen_urls:
            seen_urls[url_key] = sf
    
    unique_sample_files = list(seen_urls.values())
    
    if len(unique_sample_files) < len(sample_files):
        logging.info(f"Deduplicated sample files: {len(sample_files)} -> {len(unique_sample_files)} unique URLs")
    
    # === SKIP FILES ALREADY DOWNLOADED AT STUDY LEVEL ===
    if study_files_downloaded:
        study_filenames = {os.path.basename(f).lower() for f in study_files_downloaded}
        
        filtered_files = []
        for sf in unique_sample_files:
            if sf.filename.lower() in study_filenames:
                logging.debug(f"Skipping {sf.filename} - already downloaded at study level")
                skipped.append(sf.filename)
                filter_results.append(FileFilterResult(
                    filename=sf.filename,
                    action="skip_other",
                    reason="Already downloaded at study level",
                ))
            else:
                filtered_files.append(sf)
        
        if len(filtered_files) < len(unique_sample_files):
            logging.info(f"Skipped {len(unique_sample_files) - len(filtered_files)} files already at study level")
        
        unique_sample_files = filtered_files
    
    if not unique_sample_files:
        logging.info("No new sample files to download after deduplication")
        return downloaded, skipped, filter_results
    
    # === DRY RUN MODE ===
    if dry_run:
        for sf in unique_sample_files:
            result = filter_file(
                sf.filename,
                compiled_patterns,
                skip_patterns,
                selective,
                discovery_mode,
            )
            filter_results.append(result)
            if result.action == "download":
                downloaded.append(os.path.join(local_dir, sf.filename))
            else:
                skipped.append(sf.filename)
        return downloaded, skipped, filter_results
    
    # === CREATE OUTPUT DIRECTORY ===
    sample_dir = os.path.join(local_dir, "sample_files")
    ensure_dir(sample_dir)
    
    # === DOWNLOAD WITH RATE LIMITING ===
    sample_request_interval = 0.5  # 500ms between requests
    
    if workers and workers > 1:
        # Parallel download (capped at 4 workers)
        with ThreadPoolExecutor(max_workers=min(workers, 4)) as ex:
            futures = {}
            for sf in unique_sample_files:
                fut = ex.submit(
                    download_sample_supplementary_file,
                    sf, sample_dir, compiled_patterns, skip_patterns,
                    selective, discovery_mode, verify_checksums, resume
                )
                futures[fut] = sf
            
            for fut in tqdm(as_completed(futures), total=len(futures), 
                          desc="Downloading sample files", leave=False):
                sf = futures[fut]
                try:
                    path, result = fut.result()
                    filter_results.append(result)
                    if path:
                        downloaded.append(path)
                    else:
                        skipped.append(sf.filename)
                except Exception as e:
                    logging.warning(f"Failed to download {sf.filename}: {e}")
                    skipped.append(sf.filename)
    else:
        # Sequential download with rate limiting
        for sf in tqdm(unique_sample_files, desc="Downloading sample files", leave=False):
            path, result = download_sample_supplementary_file(
                sf, sample_dir, compiled_patterns, skip_patterns,
                selective, discovery_mode, verify_checksums, resume
            )
            filter_results.append(result)
            if path:
                downloaded.append(path)
            else:
                skipped.append(sf.filename)
            
            time.sleep(sample_request_interval)
    
    return downloaded, skipped, filter_results


def write_sample_file_table(sample_files: List[SampleSupplementaryFile], output_path: str):
    """Write sample supplementary file information to TSV."""
    if not sample_files:
        return
    
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, 
            fieldnames=["sample_name", "filename", "ftp_path", "comment_field"],
            delimiter="\t"
        )
        writer.writeheader()
        for sf in sample_files:
            writer.writerow({
                "sample_name": sf.sample_name,
                "filename": sf.filename,
                "ftp_path": sf.ftp_path,
                "comment_field": sf.comment_field,
            })


# =============================================================================
# ENA INTEGRATION
# =============================================================================

def extract_ena_study_accession(metadata: MAGETABMetadata) -> Optional[str]:
    """Extract ENA study accession from MAGE-TAB metadata."""
    # Look for ENA study in IDF fields
    for key, values in metadata.idf_fields.items():
        if "secondary" in key.lower() and "accession" in key.lower():
            for v in values:
                if v.startswith(("ERP", "SRP", "DRP", "PRJEB", "PRJNA", "PRJDA")):
                    return v
    
    # Also check Comment fields
    for key, values in metadata.idf_fields.items():
        if "ena" in key.lower() or "sra" in key.lower():
            for v in values:
                if re.match(r'^[ESDP]R[PXRS]\d+$', v) or v.startswith("PRJ"):
                    return v
    
    return None


def fetch_ena_runs(study_accession: str, limit: int = 1000) -> List[ENARunInfo]:
    """Fetch ENA run information for a study accession."""
    runs: List[ENARunInfo] = []
    
    params = {
        "result": "read_run",
        "query": f'study_accession="{study_accession}" OR secondary_study_accession="{study_accession}"',
        "limit": str(limit),
        "format": "json",
        "fields": "run_accession,experiment_accession,sample_accession,study_accession,"
                  "library_layout,library_source,library_strategy,fastq_ftp,fastq_bytes",
    }
    
    url = f"{ENA_PORTAL_API}/search?" + urllib.parse.urlencode(params)
    
    result = fetch_json(url, timeout=60)
    
    if result is None:
        return runs
    
    for r in result:
        # Handle semicolon-separated fastq_bytes for paired-end reads
        # e.g., "903654325;903743582" -> sum of both
        fastq_bytes_str = r.get("fastq_bytes", "0") or "0"
        try:
            if ";" in str(fastq_bytes_str):
                fastq_bytes = sum(int(x) for x in str(fastq_bytes_str).split(";") if x.strip())
            else:
                fastq_bytes = int(fastq_bytes_str)
        except (ValueError, TypeError):
            fastq_bytes = 0
        
        run_info = ENARunInfo(
            run_accession=r.get("run_accession", ""),
            experiment_accession=r.get("experiment_accession", ""),
            sample_accession=r.get("sample_accession", ""),
            study_accession=r.get("study_accession", ""),
            library_layout=r.get("library_layout", ""),
            library_source=r.get("library_source", ""),
            library_strategy=r.get("library_strategy", ""),
            fastq_ftp=r.get("fastq_ftp", ""),
            fastq_bytes=fastq_bytes,
        )
        runs.append(run_info)
    
    return runs


def construct_ena_fastq_url(run_accession: str) -> str:
    """Construct ENA FASTQ download URL from run accession."""
    prefix = run_accession[:6]
    
    if len(run_accession) == 9:
        path = f"vol1/fastq/{prefix}/{run_accession}"
    elif len(run_accession) == 10:
        suffix = "00" + run_accession[-1]
        path = f"vol1/fastq/{prefix}/{suffix}/{run_accession}"
    else:
        suffix = "0" + run_accession[-2:]
        path = f"vol1/fastq/{prefix}/{suffix}/{run_accession}"
    
    return f"ftp://{ENA_FTP_HOST}/{path}"


def write_ena_run_table(runs: List[ENARunInfo], output_path: str):
    """Write ENA run information to TSV."""
    if not runs:
        return
    
    fieldnames = [
        "run_accession", "experiment_accession", "sample_accession",
        "study_accession", "library_layout", "library_source",
        "library_strategy", "fastq_ftp", "fastq_bytes"
    ]
    
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for run in runs:
            writer.writerow(asdict(run))


def check_sra_tools_available() -> bool:
    """Check if sra-tools are available."""
    try:
        result = subprocess.run(["prefetch", "--version"], capture_output=True, text=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False


def download_ena_fastq(
    run: ENARunInfo,
    output_dir: str,
    timeout: int = 3600,
) -> List[str]:
    """Download FASTQ files for an ENA run."""
    downloaded: List[str] = []
    ensure_dir(output_dir)
    
    if not run.fastq_ftp:
        logging.warning(f"No FASTQ FTP path for {run.run_accession}")
        return downloaded
    
    # Parse FTP paths (may be semicolon-separated for paired-end)
    ftp_paths = run.fastq_ftp.split(";")
    
    for ftp_path in ftp_paths:
        if not ftp_path:
            continue
        
        # FTP path format: ftp.sra.ebi.ac.uk/vol1/fastq/...
        filename = os.path.basename(ftp_path)
        local_path = os.path.join(output_dir, filename)
        
        if os.path.exists(local_path):
            downloaded.append(local_path)
            continue
        
        # Download via HTTPS
        https_url = f"https://{ftp_path}"
        
        try:
            tmp_path = local_path + ".part"
            req = urllib.request.Request(https_url)
            req.add_header("User-Agent", f"ArrayExpress-Harvester/{__version__}")
            
            with urllib.request.urlopen(req, timeout=timeout) as response:
                with open(tmp_path, "wb") as f:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        f.write(chunk)
            
            os.replace(tmp_path, local_path)
            downloaded.append(local_path)
            logging.info(f"Downloaded: {filename}")
            
        except Exception as e:
            logging.warning(f"Failed to download {filename}: {e}")
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    
    return downloaded


# =============================================================================
# FTP DOWNLOAD FUNCTIONS
# =============================================================================

def download_file_https(
    accession: str,
    filename: str,
    local_dir: str,
    timeout: int = 1800,
    verify_checksums: bool = False,
    resume: bool = False,
) -> Optional[str]:
    """Download a single file via HTTPS."""
    url = construct_https_ftp_url(accession, filename)
    local_path = os.path.join(local_dir, filename)
    
    # Check if already exists
    if os.path.exists(local_path):
        if resume:
            return local_path
        if verify_checksums:
            existing_md5 = read_md5_file(local_path)
            if existing_md5:
                return local_path
    
    try:
        tmp_path = local_path + ".part"
        os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
        req = urllib.request.Request(url)
        req.add_header("User-Agent", f"ArrayExpress-Harvester/{__version__}")
        
        with urllib.request.urlopen(req, timeout=timeout) as response:
            with open(tmp_path, "wb") as f:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
        
        # Verify the temp file was actually created before renaming
        if not os.path.exists(tmp_path):
            logging.warning(f"Download incomplete for {filename} - temp file not created")
            return None
            
        os.replace(tmp_path, local_path)
        
        if verify_checksums:
            write_md5_file(local_path, md5sum(local_path))
        
        return local_path
        
    except Exception as e:
        logging.warning(f"Failed to download {filename}: {e}")
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except:
                pass
        return None
    


def download_file_ftp(
    host: str,
    remote_path: str,
    filename: str,
    local_path: str,
    timeout: int = 1800,
) -> bool:
    """Download a single file via FTP."""
    try:
        ftp = ftplib.FTP(host, timeout=timeout)
        ftp.login()
        ftp.set_pasv(True)
        ftp.cwd(remote_path)
        
        tmp_path = local_path + ".part"
        with open(tmp_path, "wb") as f:
            ftp.retrbinary(f"RETR {filename}", f.write)
        
        os.replace(tmp_path, local_path)
        ftp.quit()
        return True
        
    except Exception as e:
        logging.warning(f"FTP download failed for {filename}: {e}")
        return False

def download_study_files(
    accession: str,
    local_dir: str,
    compiled_patterns: List[re.Pattern],
    skip_patterns: List[re.Pattern],
    selective: bool = True,
    verify_checksums: bool = False,
    resume: bool = False,
    workers: int = 0,
    discovery_mode: bool = False,
    dry_run: bool = False,
    use_recursive: bool = False,
) -> Tuple[List[str], List[str], List[FileFilterResult]]:
    """
    Download supplementary files for a study.
    
    Checks BOTH BioStudies AND legacy FTP paths.
    """
    downloaded: List[str] = []
    skipped: List[str] = []
    filter_results: List[FileFilterResult] = []
    
    ensure_dir(local_dir)
    
    # === STEP 1: List files from BioStudies FTP ===
    if use_recursive:
        entries = list_ftp_files_recursive(accession)
        biostudies_files = [e.rel_path for e in entries if not e.is_dir]
    else:
        biostudies_files = list_ftp_files(accession)
    
    # Filter out metadata-only files
    data_files_on_biostudies = [
        f for f in biostudies_files 
        if not f.endswith(('.idf.txt', '.sdrf.txt'))
    ]
    
    logging.info(f"{accession}: BioStudies has {len(biostudies_files)} files ({len(data_files_on_biostudies)} data files)")
    
    # === STEP 2: Check legacy FTP if BioStudies has no data files ===
    legacy_files: List[FTPFileInfo] = []
    use_legacy = False
    
    if len(data_files_on_biostudies) == 0:
        logging.info(f"{accession}: No data files on BioStudies, checking legacy FTP...")
        legacy_files = list_legacy_ftp_files(accession)
        
        if legacy_files:
            legacy_data_files = [
                f for f in legacy_files 
                if not f.filename.endswith(('.idf.txt', '.sdrf.txt'))
                and not f.is_dir
            ]
            if legacy_data_files:
                logging.info(f"{accession}: Found {len(legacy_data_files)} data files on legacy FTP")
                use_legacy = True
    
    # === STEP 3: Combine file lists ===
    all_files: List[Tuple[str, str]] = []  # (filename, source)
    
    for f in biostudies_files:
        all_files.append((f, "biostudies"))
    
    if use_legacy:
        biostudies_filenames = {os.path.basename(f).lower() for f in biostudies_files}
        for finfo in legacy_files:
            if finfo.filename.lower() not in biostudies_filenames and not finfo.is_dir:
                all_files.append((finfo.filename, "legacy"))
    
    if not all_files:
        logging.info(f"{accession}: No supplementary files found on any FTP location")
        return downloaded, skipped, filter_results
    
    logging.info(f"{accession}: Total {len(all_files)} files to process")
    
    # === STEP 4: Filter files ===
    targets: List[Tuple[str, str]] = []
    
    for filename, source in all_files:
        basename = os.path.basename(filename)
        
        result = filter_file(
            basename,
            compiled_patterns,
            skip_patterns,
            selective,
            discovery_mode,
        )
        filter_results.append(result)
        log_filter_decision(result, accession)
        
        if result.action == "download":
            targets.append((filename, source))
        else:
            skipped.append(filename)
    
    # === STEP 5: Dry run ===
    if dry_run:
        logging.info(f"{accession}: [DRY-RUN] Would download {len(targets)} files:")
        for f, source in targets[:15]:
            logging.info(f"{accession}: [DRY-RUN]   ✓ {f} (from {source})")
        if len(targets) > 15:
            logging.info(f"{accession}: [DRY-RUN]   ... and {len(targets) - 15} more files")
        return [os.path.join(local_dir, os.path.basename(f)) for f, _ in targets], skipped, filter_results
    
    # === STEP 6: Download ===
    def download_single_file(filename: str, source: str) -> Optional[str]:
        local_path = os.path.join(local_dir, os.path.basename(filename))
        
        if os.path.exists(local_path):
            if resume:
                return local_path
            if verify_checksums and read_md5_file(local_path):
                return local_path
        
        if source == "biostudies":
            url = construct_https_ftp_url(accession, filename)
        else:
            url = construct_legacy_ftp_url(accession, filename)
        
        if download_file_with_fallback(url, local_path, verify_checksums=verify_checksums):
            return local_path
        return None
    
    if workers and workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {}
            for filename, source in targets:
                fut = executor.submit(download_single_file, filename, source)
                futures[fut] = (filename, source)
            
            for fut in as_completed(futures):
                filename, source = futures[fut]
                try:
                    result = fut.result()
                    if result:
                        downloaded.append(result)
                        logging.info(f"{accession}: Downloaded {os.path.basename(filename)} (from {source})")
                except Exception as e:
                    logging.warning(f"{accession}: Failed {filename}: {e}")
    else:
        for filename, source in tqdm(targets, desc=f"Downloading {accession}", leave=False):
            result = download_single_file(filename, source)
            if result:
                downloaded.append(result)
                logging.info(f"{accession}: Downloaded {os.path.basename(filename)} (from {source})")
            time.sleep(MIN_REQUEST_INTERVAL)
    
    return downloaded, skipped, filter_results


# =============================================================================
# ARCHIVE EXTRACTION
# =============================================================================

def extract_archives(
    local_dir: str,
    compiled_patterns: List[re.Pattern],
    skip_patterns: List[re.Pattern],
    discovery_mode: bool = False,
) -> Tuple[List[str], List[str], List[FileFilterResult]]:
    """Extract tar/zip/gz archives and return extracted files."""
    extracted: List[str] = []
    skipped: List[str] = []
    filter_results: List[FileFilterResult] = []
    
    for root, _, files in os.walk(local_dir):
        for fname in files:
            fpath = os.path.join(root, fname)
            
            # TAR archives
            if fname.endswith((".tar", ".tar.gz", ".tgz")):
                try:
                    with tarfile.open(fpath, "r:*") as tar:
                        extract_dir = fpath.replace(".tar.gz", "").replace(".tgz", "").replace(".tar", "")
                        ensure_dir(extract_dir)
                        
                        for member in tar.getmembers():
                            if not member.isfile():
                                continue
                            name = os.path.basename(member.name)
                            
                            # Use centralized filter
                            result = filter_file(
                                name,
                                compiled_patterns,
                                skip_patterns,
                                selective=not discovery_mode,
                                discovery_mode=discovery_mode,
                            )
                            filter_results.append(result)
                            
                            if result.action == "download":
                                member.name = os.path.normpath(member.name).lstrip(os.sep)
                                tar.extract(member, extract_dir)
                                extracted.append(os.path.join(extract_dir, member.name))
                            else:
                                skipped.append(name)
                                
                except Exception as e:
                    logging.warning(f"Error extracting {fname}: {e}")
            
            # ZIP archives
            elif fname.endswith(".zip"):
                try:
                    with zipfile.ZipFile(fpath, "r") as z:
                        extract_dir = fpath[:-4]
                        ensure_dir(extract_dir)
                        
                        for member in z.namelist():
                            name = os.path.basename(member)
                            if not name:
                                continue
                            
                            result = filter_file(
                                name,
                                compiled_patterns,
                                skip_patterns,
                                selective=not discovery_mode,
                                discovery_mode=discovery_mode,
                            )
                            filter_results.append(result)
                            
                            if result.action == "download":
                                dest = os.path.normpath(os.path.join(extract_dir, member))
                                if not dest.startswith(os.path.abspath(extract_dir)):
                                    continue
                                z.extract(member, extract_dir)
                                extracted.append(dest)
                            else:
                                skipped.append(name)
                                
                except Exception as e:
                    logging.warning(f"Error extracting {fname}: {e}")
            
            # Loose GZ files (not tar.gz)
            elif fname.endswith(".gz") and not fname.endswith(".tar.gz"):
                try:
                    out_path = fpath[:-3]
                    name = os.path.basename(out_path)
                    
                    result = filter_file(
                        name,
                        compiled_patterns,
                        skip_patterns,
                        selective=not discovery_mode,
                        discovery_mode=discovery_mode,
                    )
                    filter_results.append(result)
                    
                    if result.action == "download":
                        with gzip.open(fpath, "rb") as fin:
                            with open(out_path, "wb") as fout:
                                fout.write(fin.read())
                        extracted.append(out_path)
                    else:
                        skipped.append(fname)
                        
                except Exception as e:
                    logging.warning(f"Error decompressing {fname}: {e}")
    
    return extracted, skipped, filter_results


# =============================================================================
# MANIFEST BUILDING
# =============================================================================

def build_manifest(
    base_dir: str,
    accession: str,
    mode: str,
    verify_checksums: bool = False,
    skipped_files: Optional[List[str]] = None,
    filter_results: Optional[List[FileFilterResult]] = None,
) -> str:
    """Build manifest.json with file inventory."""
    study_dir = os.path.join(base_dir, accession)
    manifest_path = os.path.join(study_dir, "manifest.json")
    entries = []
    
    for root, _, files in os.walk(study_dir):
        for fname in files:
            if fname == "manifest.json":
                continue
            
            fpath = os.path.join(root, fname)
            rel = os.path.relpath(fpath, study_dir)
            
            try:
                size = os.path.getsize(fpath)
            except OSError:
                size = None
            
            digest = ""
            if verify_checksums and size and size < (1 << 31):
                try:
                    digest = md5sum(fpath)
                except Exception:
                    pass
            
            extracted_from = None
            for parent in root.split(os.sep):
                if parent.endswith((".tar", ".tar.gz", ".tgz", ".zip")):
                    extracted_from = parent
            
            entries.append({
                "name": rel,
                "size": size,
                "md5": digest,
                "mode": mode,
                "extracted_from": extracted_from,
            })
    
    manifest_data = {
        "accession": accession,
        "files": entries,
        "skipped_files": list(set(skipped_files)) if skipped_files else [],
    }
    
    # Include filter decisions summary if available
    if filter_results:
        action_counts = {}
        for r in filter_results:
            action_counts[r.action] = action_counts.get(r.action, 0) + 1
        manifest_data["filter_summary"] = action_counts
    
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)
    
    return manifest_path


# =============================================================================
# REPORT WRITING
# =============================================================================

def write_run_report(report: HarvestRunReport, output_dir: str):
    """Write run report to multiple formats."""
    ensure_dir(output_dir)
    
    # JSON report
    json_path = os.path.join(output_dir, f"harvest_report_{report.run_id}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "run_id": report.run_id,
            "start_time": report.start_time,
            "end_time": report.end_time,
            "duration_sec": report.duration_sec,
            "parameters": {
                "mode": report.mode,
                "organism": report.organism,
                "search_term": report.search_term,
                "output_dir": report.output_dir,
                "include_microarray": report.include_microarray,
                "include_generic": report.include_generic,
                "include_ena": report.include_ena,
                "ena_download": report.ena_download,
                "include_sample_files": report.include_sample_files,
                "discovery_mode": report.discovery_mode,
            },
            "summary": {
                "total_studies_found": report.total_studies_found,
                "total_studies_processed": report.total_studies_processed,
                "total_studies_succeeded": report.total_studies_succeeded,
                "total_studies_failed": report.total_studies_failed,
                "total_files_downloaded": report.total_files_downloaded,
                "total_files_skipped": report.total_files_skipped,
                "total_files_failed": report.total_files_failed,
                "total_bytes_downloaded": report.total_bytes_downloaded,
                "total_gb_downloaded": round(report.total_bytes_downloaded / (1024**3), 2),
                "total_sample_files_downloaded": report.total_sample_files_downloaded,
                "total_ena_runs": report.total_ena_runs,
                "total_ena_fastqs": report.total_ena_fastqs,
            },
            "errors": report.errors[:100],
            "study_results": [
                {
                    "accession": r.accession,
                    "status": r.status,
                    "title": r.title[:100] if r.title else "",
                    "sample_count": r.sample_count,
                    "files_downloaded": r.files_downloaded,
                    "sample_files_downloaded": r.sample_files_downloaded,
                    "bytes_downloaded": r.total_bytes_downloaded,
                    "errors": r.errors,
                }
                for r in report.study_results
            ],
        }, f, indent=2)
    
    # Download log TSV
    if report.download_events:
        log_path = os.path.join(output_dir, f"download_log_{report.run_id}.tsv")
        with open(log_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, delimiter="\t", fieldnames=[
                "timestamp", "accession", "filename", "source", "status",
                "size_bytes", "duration_sec", "md5", "error_message"
            ])
            writer.writeheader()
            for event in report.download_events:
                writer.writerow(asdict(event))
    
    # Summary text report
    summary_path = os.path.join(output_dir, f"harvest_summary_{report.run_id}.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("ARRAYEXPRESS/BIOSTUDIES HARVEST REPORT (Diagnostic-Informed)\n")
        f.write(f"Run ID: {report.run_id}\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("PARAMETERS\n")
        f.write("-" * 40 + "\n")
        f.write(f"Mode: {report.mode}\n")
        f.write(f"Organism: {report.organism}\n")
        f.write(f"Search term: {report.search_term}\n")
        f.write(f"Output directory: {report.output_dir}\n")
        f.write(f"Include microarray: {report.include_microarray}\n")
        f.write(f"Include generic: {report.include_generic}\n")
        f.write(f"Include ENA: {report.include_ena}\n")
        f.write(f"ENA download: {report.ena_download}\n")
        f.write(f"Include sample files: {report.include_sample_files}\n")
        f.write(f"Discovery mode: {report.discovery_mode}\n\n")
        
        f.write("TIMING\n")
        f.write("-" * 40 + "\n")
        f.write(f"Start: {report.start_time}\n")
        f.write(f"End: {report.end_time}\n")
        if report.duration_sec:
            hours = int(report.duration_sec // 3600)
            mins = int((report.duration_sec % 3600) // 60)
            secs = int(report.duration_sec % 60)
            f.write(f"Duration: {hours}h {mins}m {secs}s\n\n")
        
        f.write("RESULTS SUMMARY\n")
        f.write("-" * 40 + "\n")
        f.write(f"Studies found: {report.total_studies_found}\n")
        f.write(f"Studies processed: {report.total_studies_processed}\n")
        f.write(f"Studies succeeded: {report.total_studies_succeeded}\n")
        f.write(f"Studies failed: {report.total_studies_failed}\n\n")
        
        f.write(f"Files downloaded: {report.total_files_downloaded}\n")
        if report.include_sample_files:
            f.write(f"Sample files downloaded: {report.total_sample_files_downloaded}\n")
        f.write(f"Files skipped: {report.total_files_skipped}\n")
        f.write(f"Files failed: {report.total_files_failed}\n")
        gb = report.total_bytes_downloaded / (1024**3)
        f.write(f"Total data: {gb:.2f} GB\n\n")
        
        if report.include_ena:
            f.write(f"ENA runs found: {report.total_ena_runs}\n")
            f.write(f"ENA FASTQs downloaded: {report.total_ena_fastqs}\n\n")
        
        if report.errors:
            f.write("ERRORS (first 50)\n")
            f.write("-" * 40 + "\n")
            for err in report.errors[:50]:
                f.write(f"  - {err}\n")
            if len(report.errors) > 50:
                f.write(f"  ... and {len(report.errors) - 50} more errors\n")
            f.write("\n")
        
        f.write("PER-STUDY SUMMARY\n")
        f.write("-" * 40 + "\n")
        f.write(f"{'Accession':<18} {'Status':<10} {'Files':<8} {'Sample':<8} {'Size (MB)':<12} {'Samples':<8}\n")
        f.write("-" * 64 + "\n")
        for r in report.study_results:
            mb = r.total_bytes_downloaded / (1024**2)
            f.write(f"{r.accession:<18} {r.status:<10} {r.files_downloaded:<8} {r.sample_files_downloaded:<8} {mb:<12.1f} {r.sample_count:<8}\n")
    
    logging.info(f"Run report written to: {json_path}")
    logging.info(f"Summary written to: {summary_path}")


# =============================================================================
# STUDY PROCESSING
# =============================================================================


def process_study(
    accession: str,
    metadata_writer: csv.DictWriter,
    download: bool,
    base_dir: str,
    mode: str,
    compiled_patterns: List[re.Pattern],
    skip_patterns: List[re.Pattern],
    selective: bool = True,
    verify_checksums: bool = False,
    resume: bool = False,
    workers: int = 0,
    discovery_mode: bool = False,
    include_ena: bool = False,
    ena_download: bool = False,
    include_sample_files: bool = False,
    run_report: Optional['HarvestRunReport'] = None,
    dry_run: bool = False,
    use_recursive_ftp: bool = False,
) -> 'StudyProcessingResult':
    """
    Process a single ArrayExpress study.
    
    FIXES:
    - Uses updated download_study_files with legacy FTP support
    - Passes study_files_downloaded to prevent re-downloading
    - Tracks both BioStudies and legacy FTP sources
    """
    from arrayexpress_harvester import (
        StudyProcessingResult, DownloadEvent, ensure_dir, get_timestamp,
        fetch_mage_tab, extract_archives, build_manifest, 
        write_sample_file_table, extract_ena_study_accession,
        fetch_ena_runs, write_ena_run_table, download_ena_fastq,
    )
    
    start_time = get_timestamp()
    t0 = time.time()
    
    result = StudyProcessingResult(
        accession=accession,
        start_time=start_time,
        end_time="",
        duration_sec=0,
        status="success",
    )
    
    study_dir = os.path.join(base_dir, accession)
    ensure_dir(study_dir)
    
    # Fetch and parse MAGE-TAB metadata
    logging.info(f"{accession}: Fetching MAGE-TAB metadata...")
    mage_tab = fetch_mage_tab(accession, study_dir)
    
    result.title = mage_tab.title
    result.organism = mage_tab.organism
    result.sample_count = mage_tab.sample_count
    
    # Build metadata row
    metadata_row = {
        "accession": accession,
        "title": mage_tab.title,
        "description": mage_tab.description[:500] if mage_tab.description else "",
        "organism": mage_tab.organism,
        "experiment_type": mage_tab.experiment_type,
        "pubmed_id": mage_tab.pubmed_id,
        "release_date": mage_tab.release_date,
        "sample_count": mage_tab.sample_count,
        "sdrf_path": mage_tab.samples_df_path or "",
    }
    
    downloaded_files: List[str] = []
    skipped_files: List[str] = []
    all_filter_results: List['FileFilterResult'] = []
    
    # Track study-level files for deduplication
    study_level_files: List[str] = []
    
    # Download supplementary files (with legacy FTP support)
    if download:
        logging.info(f"{accession}: Downloading supplementary files...")
        dl_files, dl_skipped, dl_filter_results = download_study_files(
            accession=accession,
            local_dir=study_dir,
            compiled_patterns=compiled_patterns,
            skip_patterns=skip_patterns,
            selective=selective,
            verify_checksums=verify_checksums,
            resume=resume,
            workers=workers,
            discovery_mode=discovery_mode,
            dry_run=dry_run,
            use_recursive=use_recursive_ftp,
        )
        downloaded_files.extend(dl_files)
        skipped_files.extend(dl_skipped)
        all_filter_results.extend(dl_filter_results)
        
        # Track study-level files for deduplication
        study_level_files.extend(dl_files)
        
        # Extract archives
        if not dry_run and downloaded_files:
            logging.debug(f"{accession}: Extracting archives...")
            extracted, ext_skipped, ext_filter_results = extract_archives(
                study_dir,
                compiled_patterns if selective and not discovery_mode else [],
                skip_patterns,
                discovery_mode,
            )
            downloaded_files.extend(extracted)
            skipped_files.extend(ext_skipped)
            all_filter_results.extend(ext_filter_results)
            
            study_level_files.extend(extracted)
    
    # Sample-level files from SDRF
    if include_sample_files and mage_tab.sample_files:
        logging.info(f"{accession}: Processing {len(mage_tab.sample_files)} sample-level file references...")
        
        if not dry_run:
            sample_table_path = os.path.join(study_dir, f"{accession}_sample_files.tsv")
            write_sample_file_table(mage_tab.sample_files, sample_table_path)
            metadata_row["sample_file_table"] = sample_table_path
        
        metadata_row["sample_files_found"] = len(mage_tab.sample_files)
        
        if download:
            # Pass study_level_files to prevent re-downloading
            sample_dl, sample_skipped, sample_filter_results = download_all_sample_files(
                mage_tab.sample_files,
                study_dir,
                compiled_patterns,
                skip_patterns,
                selective=selective,
                discovery_mode=discovery_mode,
                verify_checksums=verify_checksums,
                resume=resume,
                workers=workers,
                dry_run=dry_run,
                study_files_downloaded=study_level_files,
            )
            
            result.sample_files_downloaded = len(sample_dl)
            result.sample_files_skipped = len(sample_skipped)
            downloaded_files.extend(sample_dl)
            skipped_files.extend(sample_skipped)
            all_filter_results.extend(sample_filter_results)
            
            metadata_row["sample_files_downloaded"] = len(sample_dl)
            
            if run_report and not dry_run:
                for fp in sample_dl:
                    try:
                        size = os.path.getsize(fp)
                        result.total_bytes_downloaded += size
                        run_report.download_events.append(DownloadEvent(
                            timestamp=get_timestamp(),
                            accession=accession,
                            filename=os.path.basename(fp),
                            source="sample_suppl",
                            status="success",
                            size_bytes=size,
                        ))
                    except:
                        pass
    
    # ENA integration
    if include_ena:
        logging.info(f"{accession}: Checking for ENA data...")
        ena_study = extract_ena_study_accession(mage_tab)
        
        if ena_study:
            logging.info(f"{accession}: Found ENA study {ena_study}")
            ena_runs = fetch_ena_runs(ena_study)
            
            if ena_runs:
                result.ena_runs_found = len(ena_runs)
                logging.info(f"{accession}: Found {len(ena_runs)} ENA runs")
                
                if not dry_run:
                    ena_table_path = os.path.join(study_dir, f"{accession}_ena_runs.tsv")
                    write_ena_run_table(ena_runs, ena_table_path)
                    metadata_row["ena_table"] = ena_table_path
                
                metadata_row["ena_runs"] = len(ena_runs)
                
                if ena_download:
                    if dry_run:
                        total_bytes = sum(r.fastq_bytes for r in ena_runs)
                        logging.info(f"{accession}: [DRY-RUN] Would download {len(ena_runs)} ENA runs (~{total_bytes/(1024**3):.2f} GB)")
                    else:
                        ena_dir = os.path.join(study_dir, "ena_fastq")
                        ensure_dir(ena_dir)
                        
                        for run in tqdm(ena_runs, desc=f"ENA {accession}", leave=False):
                            fastqs = download_ena_fastq(run, ena_dir)
                            downloaded_files.extend(fastqs)
                            result.ena_fastqs_downloaded += len(fastqs)
                            
                            if run_report:
                                for fq in fastqs:
                                    try:
                                        size = os.path.getsize(fq)
                                        result.total_bytes_downloaded += size
                                        run_report.download_events.append(DownloadEvent(
                                            timestamp=get_timestamp(),
                                            accession=accession,
                                            filename=os.path.basename(fq),
                                            source="ena_fastq",
                                            status="success",
                                            size_bytes=size,
                                        ))
                                    except:
                                        pass
        else:
            metadata_row["ena_runs"] = 0
    
    # Calculate statistics
    result.files_downloaded = len(downloaded_files)
    result.files_skipped = len(skipped_files)
    result.filter_results = all_filter_results
    
    for fp in downloaded_files:
        try:
            size = os.path.getsize(fp)
            result.total_bytes_downloaded += size
            
            if run_report:
                run_report.download_events.append(DownloadEvent(
                    timestamp=get_timestamp(),
                    accession=accession,
                    filename=os.path.basename(fp),
                    source="biostudies",
                    status="success",
                    size_bytes=size,
                ))
        except:
            pass
    
    metadata_row["files_downloaded"] = result.files_downloaded
    metadata_row["files_skipped"] = result.files_skipped
    metadata_row["bytes_downloaded"] = result.total_bytes_downloaded
    
    # Build manifest
    manifest_path = build_manifest(
        base_dir, accession, mode,
        verify_checksums, skipped_files, all_filter_results
    )
    metadata_row["manifest"] = manifest_path
    
    # Write metadata row
    metadata_writer.writerow(metadata_row)
    
    # Finalize result
    result.end_time = get_timestamp()
    result.duration_sec = time.time() - t0
    
    if result.errors:
        result.status = "partial" if result.files_downloaded > 0 else "failed"
    
    return result


# =============================================================================
# MAIN CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Download ArrayExpress/BioStudies data (Diagnostic-Informed Edition).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
CRITICAL FIXES (Based on Diagnostic Analysis):
-----------------------------------------------------
1. Uses "arrayexpress" domain (NOT "biostudies") - returns actual experiments
2. Uses FREE TEXT queries (field-specific queries return 0!)
3. Proper accession extraction (S-ECPF-* -> E-*)
4. Optimized organism + technology query patterns

Examples:
    # Basic single-cell harvest
    python arrayexpress_harvester.py --mode single-cell --organism "Rattus norvegicus" --download --limit 10
    
    # Test search query first (RECOMMENDED)
    python arrayexpress_harvester.py --test-search --mode single-cell --organism "Rattus norvegicus"
    
    # Discovery mode (see all files)
    python arrayexpress_harvester.py --mode bulk --download --discovery --limit 5
    
    # Full harvest with sample-level files
    python arrayexpress_harvester.py --mode both --download --include-sample-files --workers 8
    
    # Dry run to preview with pattern details
    python arrayexpress_harvester.py --mode single-cell --download --dry-run --limit 5
    
    # Show patterns and validate
    python arrayexpress_harvester.py --show-patterns --mode single-cell
    
    # Test a filename against patterns
    python arrayexpress_harvester.py --test-pattern "matrix.mtx.gz" --mode single-cell
        """
    )
    
    # Mode and organism
    parser.add_argument("--mode", choices=["single-cell", "bulk", "both"], default="single-cell",
                        help="Data mode to harvest (default: single-cell)")
    parser.add_argument("--organism", type=str, default="Rattus norvegicus",
                        help="Organism to filter by (default: Rattus norvegicus)")
    parser.add_argument("--search-term", type=str, default=None,
                        help="Custom search query (overrides mode/organism)")
    
    # Download options
    parser.add_argument("--download", action="store_true",
                        help="Download supplementary files")
    parser.add_argument("--all-files", action="store_true",
                        help="Download all files (ignore pattern filtering)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be downloaded without downloading")
    
    # Output options
    parser.add_argument("--output-dir", type=str, default="data/arrayexpress_harvest",
                        help="Output directory (default: data/arrayexpress_harvest)")
    parser.add_argument("--metadata-file", type=str, default="arrayexpress_metadata.csv",
                        help="Metadata CSV file (default: arrayexpress_metadata.csv)")
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit number of studies (0 = no limit)")
    
    # Performance options
    parser.add_argument("--workers", type=int, default=0,
                        help="Parallel download workers (0 = sequential)")
    parser.add_argument("--verify-checksums", action="store_true",
                        help="Verify and save MD5 checksums")
    parser.add_argument("--resume", action="store_true",
                        help="Skip already downloaded files")
    parser.add_argument("--recursive-ftp", action="store_true",
                        help="Use recursive FTP listing for nested directories")
    
    # Pattern options
    parser.add_argument("--include-microarray", action="store_true",
                        help="Include microarray file patterns")
    parser.add_argument("--include-generic", action="store_true",
                        help="Include generic catch-all patterns")
    parser.add_argument("--discovery", action="store_true",
                        help="Discovery mode: download all files")
    
    # Sample-level file options
    parser.add_argument("--include-sample-files", action="store_true",
                        help="Include sample-level supplementary files from SDRF")
    
    # ENA options
    parser.add_argument("--include-ena", action="store_true",
                        help="Extract and record ENA run information")
    parser.add_argument("--ena-download", action="store_true",
                        help="Download FASTQ files from ENA")
    
    # Pattern inspection
    parser.add_argument("--show-patterns", action="store_true",
                        help="Show all patterns, validate, and exit")
    parser.add_argument("--test-pattern", type=str, default=None, metavar="FILENAME",
                        help="Test if filename matches patterns and exit")
    
    # Query testing 
    parser.add_argument("--test-search", action="store_true",
                        help="Test search query and show results (recommended before harvest)")
    
    # Logging
    parser.add_argument("--log-level", type=str, default="INFO",
                        help="Log level (DEBUG, INFO, WARNING, ERROR)")
    parser.add_argument("--log-dir", type=str, default=None,
                        help="Directory for log files")
    
    args = parser.parse_args()
    
    # Handle search query testing 
    if args.test_search:
        print("\n" + "="*70)
        print("SEARCH QUERY TEST (Diagnostic-Informed)")
        print("="*70)
        print(f"\nUsing domain: {EBI_SEARCH_DOMAIN} (arrayexpress)")
        
        term = build_search_term(args.mode, args.organism, args.search_term)
        print(f"Query: {term}")
        print(f"Mode: {args.mode}")
        print(f"Organism: {args.organism}")
        
        print("\nSearching...")
        total_count, accessions = test_search_query(term)
        
        print(f"\n{'='*70}")
        print(f"Total hits: {total_count:,}")
        print(f"Valid ArrayExpress accessions: {len(accessions)}")
        
        if accessions:
            print(f"\nSample accessions (first 10):")
            for acc in accessions[:10]:
                print(f"  - {acc}")
            if len(accessions) > 10:
                print(f"  ... and {len(accessions) - 10} more")
        
        if total_count > 0 and len(accessions) > 0:
            print(f"\n✓ SUCCESS: Query returns valid ArrayExpress accessions")
            print(f"  Ready to harvest with --download flag")
        elif total_count > 0 and len(accessions) == 0:
            print(f"\n⚠ WARNING: {total_count} hits but no valid ArrayExpress accessions")
            print(f"  The query may be returning non-experiment entries")
        else:
            print(f"\n✗ No results found")
            print(f"  Try adjusting the query or organism")
        
        print("="*70)
        return
    
    # Handle pattern inspection
    if args.show_patterns or args.test_pattern:
        patterns = compile_mode_patterns(
            args.mode,
            include_microarray=args.include_microarray,
            include_generic=args.include_generic,
        )
        skip_patterns = compile_skip_patterns()
        
        if args.show_patterns:
            print(f"\n{'='*60}")
            print("PATTERN CONFIGURATION (Diagnostic-Informed)")
            print(f"{'='*60}")
            print(f"Mode: {args.mode}")
            print(f"Include microarray: {args.include_microarray}")
            print(f"Include generic: {args.include_generic}")
            print(f"\n--- TARGET PATTERNS ({len(patterns)}) ---")
            for p in patterns:
                print(f"  {p.pattern}")
            print(f"\n--- SKIP PATTERNS ({len(skip_patterns)}) ---")
            for p in skip_patterns:
                print(f"  {p.pattern}")
            
            # Validate patterns
            print(f"\n--- PATTERN VALIDATION ---")
            valid = validate_patterns(patterns, args.mode)
            print(f"Validation: {'PASSED' if valid else 'FAILED'}")
            return
        
        if args.test_pattern:
            filename = args.test_pattern
            print(f"\nTesting: '{filename}'")
            print(f"Mode: {args.mode}")
            
            # Use centralized filter logic
            result = filter_file(
                filename,
                patterns,
                skip_patterns,
                selective=True,
                discovery_mode=False,
            )
            
            if result.action == "download":
                print(f"  ✓ {result.action.upper()}")
                if result.matched_pattern:
                    print(f"  Pattern: {result.matched_pattern}")
                print(f"  Reason: {result.reason}")
            else:
                print(f"  ✗ {result.action.upper()}")
                print(f"  Reason: {result.reason}")
                if result.action == "skip_no_match":
                    print("\n  Try with --include-generic to enable catch-all patterns")
            return
    
    # Setup logging
    setup_logging(args.log_level, log_dir=args.log_dir)
    
    # Initialize run report
    run_id = generate_run_id()
    run_report = HarvestRunReport(
        run_id=run_id,
        start_time=get_timestamp(),
        mode=args.mode,
        organism=args.organism,
        output_dir=args.output_dir,
        include_microarray=args.include_microarray,
        include_generic=args.include_generic,
        include_ena=args.include_ena,
        ena_download=args.ena_download,
        include_sample_files=args.include_sample_files,
        discovery_mode=args.discovery,
    )
    
    # Setup file logging
    ensure_dir(args.output_dir)
    setup_file_logging(args.output_dir, run_id, args.log_level)
    logging.info(f"Starting ArrayExpress harvest run: {run_id}")
    
    # Log critical diagnostic-informed changes
    logging.info("="*60)
    logging.info("DIAGNOSTIC-INFORMED FIXES:")
    logging.info(f"  - Domain: {EBI_SEARCH_DOMAIN} (NOT biostudies)")
    logging.info(f"  - Query: FREE TEXT (NOT field-specific)")
    logging.info(f"  - Accession: S-ECPF-* -> E-* conversion")
    logging.info("="*60)
    
    # Compile patterns
    patterns = compile_mode_patterns(
        args.mode,
        include_microarray=args.include_microarray,
        include_generic=args.include_generic,
    )
    skip_patterns = compile_skip_patterns()
    
    logging.info(f"Mode: {args.mode}")
    logging.info(f"Organism: {args.organism}")
    logging.info(f"Patterns: {len(patterns)} target, {len(skip_patterns)} skip")
    
    # Validate patterns
    valid = validate_patterns(patterns, args.mode)
    if not valid:
        logging.warning("Pattern validation failed - some expected files may not be downloaded")
    
    if args.include_microarray:
        logging.info("Microarray patterns enabled")
    if args.include_generic:
        logging.info("Generic patterns enabled")
    if args.all_files:
        logging.info("All-files mode: pattern filtering DISABLED")
    if args.discovery:
        logging.info("Discovery mode: will download all files")
    if args.include_ena:
        logging.info("ENA integration enabled")
    if args.include_sample_files:
        logging.info("Sample-level file support enabled")
    if args.recursive_ftp:
        logging.info("Recursive FTP listing enabled")
    if args.dry_run:
        logging.info("DRY-RUN MODE: No files will be downloaded")
    
    # Build search term
    term = build_search_term(args.mode, args.organism, args.search_term)
    run_report.search_term = term
    logging.info(f"Search term: {term}")
    
    # Fetch accessions
    accessions = fetch_all_accessions(term, limit=args.limit)
    run_report.total_studies_found = len(accessions)
    
    if not accessions:
        logging.error("No ArrayExpress studies found")
        logging.error("Try running with --test-search to verify your query")
        run_report.end_time = get_timestamp()
        run_report.errors.append("No studies found")
        write_run_report(run_report, args.output_dir)
        return
    
    logging.info(f"Found {len(accessions)} studies: {', '.join(accessions[:5])}{'...' if len(accessions) > 5 else ''}")
    
    # Prepare output
    base_dir = os.path.join(args.output_dir, "datasets")
    ensure_dir(base_dir)
    
    metadata_path = os.path.join(args.output_dir, args.metadata_file)
    
    fieldnames = [
        "accession", "title", "description", "organism", "experiment_type",
        "pubmed_id", "release_date", "sample_count", "sdrf_path",
        "files_downloaded", "files_skipped", "bytes_downloaded", "manifest",
    ]
    if args.include_sample_files:
        fieldnames.extend(["sample_files_found", "sample_file_table", "sample_files_downloaded"])
    if args.include_ena:
        fieldnames.extend(["ena_runs", "ena_table"])
    
    # Process studies
    with open(metadata_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        
        for accession in tqdm(accessions, desc="Processing studies"):
            try:
                result = process_study(
                    accession=accession,
                    metadata_writer=writer,
                    download=args.download or args.dry_run,
                    base_dir=base_dir,
                    mode=args.mode,
                    compiled_patterns=patterns,
                    skip_patterns=skip_patterns,
                    selective=not args.all_files,
                    verify_checksums=args.verify_checksums,
                    resume=args.resume,
                    workers=args.workers,
                    discovery_mode=args.discovery,
                    include_ena=args.include_ena,
                    ena_download=args.ena_download,
                    include_sample_files=args.include_sample_files,
                    run_report=run_report,
                    dry_run=args.dry_run,
                    use_recursive_ftp=args.recursive_ftp,
                )
                
                run_report.study_results.append(result)
                run_report.total_studies_processed += 1
                
                if result.status in ("success", "partial"):
                    run_report.total_studies_succeeded += 1
                else:
                    run_report.total_studies_failed += 1
                
                run_report.total_files_downloaded += result.files_downloaded
                run_report.total_files_skipped += result.files_skipped
                run_report.total_bytes_downloaded += result.total_bytes_downloaded
                run_report.total_sample_files_downloaded += result.sample_files_downloaded
                run_report.total_ena_runs += result.ena_runs_found
                run_report.total_ena_fastqs += result.ena_fastqs_downloaded
                
                for err in result.errors:
                    run_report.errors.append(f"{accession}: {err}")
                
            except Exception as e:
                err_msg = f"Error processing {accession}: {e}"
                logging.error(err_msg)
                run_report.errors.append(err_msg)
                run_report.total_studies_failed += 1
                run_report.total_studies_processed += 1
            
            time.sleep(MIN_REQUEST_INTERVAL)
    
    # Finalize report
    run_report.end_time = get_timestamp()
    
    try:
        t_start = datetime.fromisoformat(run_report.start_time)
        t_end = datetime.fromisoformat(run_report.end_time)
        run_report.duration_sec = (t_end - t_start).total_seconds()
    except:
        pass
    
    write_run_report(run_report, args.output_dir)
    
    # Print summary
    logging.info("=" * 60)
    if args.dry_run:
        logging.info("DRY-RUN COMPLETE (no files downloaded)")
    else:
        logging.info("HARVEST COMPLETE (Diagnostic-Informed)")
    logging.info("=" * 60)
    logging.info(f"Run ID: {run_id}")
    logging.info(f"Studies processed: {run_report.total_studies_processed}")
    logging.info(f"Studies succeeded: {run_report.total_studies_succeeded}")
    logging.info(f"Studies failed: {run_report.total_studies_failed}")
    
    if args.dry_run:
        logging.info(f"Files would download: {run_report.total_files_downloaded}")
        if args.include_sample_files:
            logging.info(f"Sample files would download: {run_report.total_sample_files_downloaded}")
    else:
        logging.info(f"Files downloaded: {run_report.total_files_downloaded}")
        if args.include_sample_files:
            logging.info(f"Sample files downloaded: {run_report.total_sample_files_downloaded}")
    
    logging.info(f"Files skipped: {run_report.total_files_skipped}")
    gb = run_report.total_bytes_downloaded / (1024**3)
    logging.info(f"Total data: {gb:.2f} GB")
    
    if args.include_ena:
        logging.info(f"ENA runs found: {run_report.total_ena_runs}")
        logging.info(f"ENA FASTQs: {run_report.total_ena_fastqs}")
    
    logging.info("=" * 60)
    logging.info(f"Metadata: {metadata_path}")
    logging.info(f"Reports: {args.output_dir}")


if __name__ == "__main__":
    main()