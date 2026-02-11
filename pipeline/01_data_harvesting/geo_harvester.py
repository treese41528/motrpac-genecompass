#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GEO Series Harvester (Single-Cell / Bulk / Both) — Expanded Edition
--------------------------------------------------------------------
IMPROVEMENTS OVER ORIGINAL:
- Comprehensive file pattern matching (microarray, RNA-seq, single-cell, etc.)
- Parses filelist.txt manifests when FTP directory listing fails
- Better archive handling and nested extraction
- Discovery mode to log all files before filtering
- Microarray platform support (Affymetrix, Illumina, Agilent)
- SRA integration: fetches SRA run info and optionally downloads via sra-tools
- GSM-level supplementary file support (soft-linked sample files)

USAGE (examples)
---------------
# Maximum performance for rat single-cell data
export NCBI_API_KEY=your_key
python geo_harvester_expanded.py \
    --mode single-cell \
    --organism "Rattus norvegicus" \
    --download \
    --batch-esummary 400 \
    --workers 8 \
    --verify-checksums \
    --resume \
    --email your@email.com

# Discovery mode - see all files without filtering
python geo_harvester_expanded.py \
    --mode bulk \
    --organism "Rattus norvegicus" \
    --download \
    --discovery \
    --limit 5

# Include microarray data
python geo_harvester_expanded.py \
    --mode bulk \
    --include-microarray \
    --download \
    --limit 10

# Include SRA data (fetch run info, optionally download FASTQs)
python geo_harvester_expanded.py \
    --mode single-cell \
    --include-sra \
    --sra-download \
    --download \
    --limit 5

# Include GSM-level supplementary files
python geo_harvester_expanded.py \
    --mode bulk \
    --include-gsm-suppl \
    --download \
    --limit 10

# Full harvest with all options
python geo_harvester_expanded.py \
    --mode both \
    --include-microarray \
    --include-generic \
    --include-sra \
    --include-gsm-suppl \
    --download \
    --workers 8
"""

import os
import re
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
import shutil
import xml.etree.ElementTree as ET
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass, field
from urllib.request import urlopen
from urllib.error import URLError, HTTPError

import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from Bio import Entrez

try:
    import GEOparse
except ImportError:
    raise ImportError("GEOparse is required. Install with: pip install GEOparse")

# --- Config integration ---
import sys as _sys
import os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), '..', '..', 'lib'))
try:
    from gene_utils import load_config, resolve_path
    _config = load_config()
except (FileNotFoundError, ImportError):
    _config = None

def _cfg(key_path: str, fallback):
    """Get nested config value with dotted key path, or return fallback."""
    if _config is None:
        return fallback
    obj = _config
    for key in key_path.split('.'):
        if isinstance(obj, dict):
            obj = obj.get(key)
        else:
            return fallback
        if obj is None:
            return fallback
    return obj


# -----------------------
# SRA Data Structures
# -----------------------
# -----------------------
# Run Report / Statistics
# -----------------------
@dataclass
class DownloadEvent:
    """Record of a single download attempt"""
    timestamp: str
    gse: str
    filename: str
    source: str  # "series_suppl", "gsm_suppl", "sra_fastq"
    status: str  # "success", "failed", "skipped", "exists"
    size_bytes: Optional[int] = None
    duration_sec: Optional[float] = None
    error_message: Optional[str] = None
    md5: Optional[str] = None


@dataclass
class GSEProcessingResult:
    """Summary of processing for a single GSE"""
    gse: str
    start_time: str
    end_time: str
    duration_sec: float
    status: str  # "success", "partial", "failed"
    
    # Counts
    series_files_downloaded: int = 0
    series_files_skipped: int = 0
    series_files_failed: int = 0
    gsm_files_downloaded: int = 0
    gsm_files_skipped: int = 0
    sra_runs_found: int = 0
    sra_fastqs_downloaded: int = 0
    
    # Size
    total_bytes_downloaded: int = 0
    
    # Errors
    errors: List[str] = field(default_factory=list)


@dataclass 
class HarvestRunReport:
    """Complete report for a harvesting run"""
    # Run identification
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
    include_sra: bool = False
    sra_download: bool = False
    include_gsm_suppl: bool = False
    discovery_mode: bool = False
    
    # Results
    total_gse_found: int = 0
    total_gse_processed: int = 0
    total_gse_succeeded: int = 0
    total_gse_failed: int = 0
    
    # File counts
    total_files_downloaded: int = 0
    total_files_skipped: int = 0
    total_files_failed: int = 0
    total_bytes_downloaded: int = 0
    
    # SRA
    total_sra_runs: int = 0
    total_sra_fastqs: int = 0
    
    # Details
    gse_results: List[GSEProcessingResult] = field(default_factory=list)
    download_events: List[DownloadEvent] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


def generate_run_id() -> str:
    """Generate a unique run ID based on timestamp"""
    from datetime import datetime
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def get_timestamp() -> str:
    """Get current ISO timestamp"""
    from datetime import datetime
    return datetime.now().isoformat()


def write_run_report(report: HarvestRunReport, output_dir: str):
    """Write the run report to multiple formats"""
    ensure_dir(output_dir)
    
    # JSON report (complete)
    json_path = os.path.join(output_dir, f"harvest_report_{report.run_id}.json")
    with open(json_path, "w") as f:
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
                "include_sra": report.include_sra,
                "sra_download": report.sra_download,
                "include_gsm_suppl": report.include_gsm_suppl,
                "discovery_mode": report.discovery_mode,
            },
            "summary": {
                "total_gse_found": report.total_gse_found,
                "total_gse_processed": report.total_gse_processed,
                "total_gse_succeeded": report.total_gse_succeeded,
                "total_gse_failed": report.total_gse_failed,
                "total_files_downloaded": report.total_files_downloaded,
                "total_files_skipped": report.total_files_skipped,
                "total_files_failed": report.total_files_failed,
                "total_bytes_downloaded": report.total_bytes_downloaded,
                "total_gb_downloaded": round(report.total_bytes_downloaded / (1024**3), 2),
                "total_sra_runs": report.total_sra_runs,
                "total_sra_fastqs": report.total_sra_fastqs,
            },
            "errors": report.errors,
            "gse_results": [
                {
                    "gse": r.gse,
                    "status": r.status,
                    "duration_sec": r.duration_sec,
                    "files_downloaded": r.series_files_downloaded + r.gsm_files_downloaded,
                    "bytes_downloaded": r.total_bytes_downloaded,
                    "errors": r.errors,
                }
                for r in report.gse_results
            ],
        }, f, indent=2)
    
    # Download log (TSV)
    if report.download_events:
        log_path = os.path.join(output_dir, f"download_log_{report.run_id}.tsv")
        with open(log_path, "w", newline="") as f:
            writer = csv.DictWriter(f, delimiter="\t", fieldnames=[
                "timestamp", "gse", "filename", "source", "status", 
                "size_bytes", "duration_sec", "md5", "error_message"
            ])
            writer.writeheader()
            for event in report.download_events:
                writer.writerow({
                    "timestamp": event.timestamp,
                    "gse": event.gse,
                    "filename": event.filename,
                    "source": event.source,
                    "status": event.status,
                    "size_bytes": event.size_bytes or "",
                    "duration_sec": f"{event.duration_sec:.2f}" if event.duration_sec else "",
                    "md5": event.md5 or "",
                    "error_message": event.error_message or "",
                })
    
    # Summary text report
    summary_path = os.path.join(output_dir, f"harvest_summary_{report.run_id}.txt")
    with open(summary_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write(f"GEO HARVEST RUN REPORT\n")
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
        f.write(f"Include SRA: {report.include_sra}\n")
        f.write(f"SRA download: {report.sra_download}\n")
        f.write(f"Include GSM suppl: {report.include_gsm_suppl}\n")
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
        f.write(f"GSE found: {report.total_gse_found}\n")
        f.write(f"GSE processed: {report.total_gse_processed}\n")
        f.write(f"GSE succeeded: {report.total_gse_succeeded}\n")
        f.write(f"GSE failed: {report.total_gse_failed}\n\n")
        
        f.write(f"Files downloaded: {report.total_files_downloaded}\n")
        f.write(f"Files skipped: {report.total_files_skipped}\n")
        f.write(f"Files failed: {report.total_files_failed}\n")
        gb = report.total_bytes_downloaded / (1024**3)
        f.write(f"Total data: {gb:.2f} GB ({report.total_bytes_downloaded:,} bytes)\n\n")
        
        if report.include_sra:
            f.write(f"SRA runs found: {report.total_sra_runs}\n")
            f.write(f"SRA FASTQs downloaded: {report.total_sra_fastqs}\n\n")
        
        if report.errors:
            f.write("ERRORS\n")
            f.write("-" * 40 + "\n")
            for err in report.errors[:50]:  # Limit to first 50
                f.write(f"  - {err}\n")
            if len(report.errors) > 50:
                f.write(f"  ... and {len(report.errors) - 50} more errors\n")
            f.write("\n")
        
        # Per-GSE summary table
        f.write("PER-GSE SUMMARY\n")
        f.write("-" * 40 + "\n")
        f.write(f"{'GSE':<15} {'Status':<10} {'Files':<8} {'Size (MB)':<12} {'Time (s)':<10}\n")
        f.write("-" * 55 + "\n")
        for r in report.gse_results:
            mb = r.total_bytes_downloaded / (1024**2)
            total_files = r.series_files_downloaded + r.gsm_files_downloaded + r.sra_fastqs_downloaded
            f.write(f"{r.gse:<15} {r.status:<10} {total_files:<8} {mb:<12.1f} {r.duration_sec:<10.1f}\n")
    
    logging.info(f"Run report written to: {json_path}")
    logging.info(f"Download log written to: {os.path.join(output_dir, f'download_log_{report.run_id}.tsv')}")
    logging.info(f"Summary written to: {summary_path}")


def setup_file_logging(output_dir: str, run_id: str, log_level: str = "DEBUG"):
    """Set up file-based logging in addition to console"""
    ensure_dir(output_dir)
    log_file = os.path.join(output_dir, f"harvest_{run_id}.log")
    
    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(getattr(logging, log_level.upper(), logging.DEBUG))
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s"
    ))
    
    # Add to root logger
    logging.getLogger().addHandler(file_handler)
    logging.info(f"Logging to file: {log_file}")
    
    return log_file


@dataclass
class SRARunInfo:
    """Information about a single SRA run (SRR)"""
    run_accession: str  # SRR...
    experiment_accession: str  # SRX...
    sample_accession: str  # SRS...
    study_accession: str  # SRP...
    biosample: str  # SAMN...
    gsm: Optional[str] = None  # GSM... if linked
    platform: str = ""
    instrument: str = ""
    library_layout: str = ""  # SINGLE or PAIRED
    library_source: str = ""  # TRANSCRIPTOMIC, GENOMIC, etc.
    library_strategy: str = ""  # RNA-Seq, scRNA-Seq, etc.
    spots: int = 0
    bases: int = 0
    size_mb: float = 0.0


@dataclass
class GSMSupplementaryFile:
    """A supplementary file linked to a specific GSM sample"""
    gsm: str
    filename: str
    ftp_path: str
    size: Optional[int] = None
    file_type: Optional[str] = None

# -----------------------
# Logging Configuration
# -----------------------
def setup_logging(
    log_level: str = "INFO",
    log_dir: Optional[str] = None,
    log_file: Optional[str] = None,
):
    """
    Configure logging to console and optionally to file.
    
    Args:
        log_level: DEBUG, INFO, WARNING, ERROR
        log_dir: Directory for log files (auto-generates timestamped filename)
        log_file: Explicit log file path (overrides log_dir)
    """
    level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Base format
    fmt = "%(asctime)s [%(levelname)s] %(message)s"
    
    # Create handlers
    handlers = [logging.StreamHandler()]  # Console output
    
    # File output
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True) if os.path.dirname(log_file) else None
        handlers.append(logging.FileHandler(log_file, mode='a', encoding='utf-8'))
    elif log_dir:
        os.makedirs(log_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(log_dir, f"geo_harvest_{timestamp}.log")
        handlers.append(logging.FileHandler(log_path, mode='a', encoding='utf-8'))
        print(f"Logging to: {log_path}")
    
    # Configure root logger
    logging.basicConfig(
        level=level,
        format=fmt,
        handlers=handlers,
    )

# -----------------------
# Entrez / NCBI Settings
# -----------------------
def init_entrez(email: str, api_key: Optional[str]):
    Entrez.email = email
    
    # IMPORTANT: Use None instead of empty string to avoid HTTP 400 errors
    # Biopython sends api_key="" in the request which NCBI rejects
    env_key = os.environ.get("NCBI_API_KEY")
    Entrez.api_key = api_key or env_key or None  # Must be None, not ""
    
    if not Entrez.email:
        logging.warning("No Entrez email provided. Set --email to comply with NCBI E-utilities policy.")
    if not Entrez.api_key:
        logging.warning("No NCBI API key set. You will hit stricter rate limits (3 req/sec).")
        logging.warning("  Get a free key at: https://www.ncbi.nlm.nih.gov/account/settings/")
        logging.warning("  Then: export NCBI_API_KEY=your_key")
    else:
        logging.info(f"NCBI API key: {'from --api-key' if api_key else 'from NCBI_API_KEY env var'}")

def ncbi_sleep_time() -> float:
    return 0.12 if Entrez.api_key else 0.34

# -----------------------
# EXPANDED PATTERN SETS
# -----------------------

# === SINGLE-CELL PATTERNS ===
SC_PATTERNS = [
    # 10x Genomics / CellRanger outputs
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
    r'_obj\.h5ad$',
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
    r'(umi|unique[-_]?molecular).*counts?\.(tsv|csv|txt)(\.gz)?$',
    r'digital[-_]?expression\.(tsv|csv|txt)(\.gz)?$',
    
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

# === BULK RNA-SEQ PATTERNS ===
BULK_PATTERNS = [
    # Standard count matrices
    r'(gene|transcript)[-_]?counts?\.(tsv|csv|txt)(\.gz)?$',
    r'(read|fragment)[-_]?counts?\.(tsv|csv|txt)(\.gz)?$',
    r'counts?[-_]?matrix\.(tsv|csv|txt)(\.gz)?$',
    r'counts?[-_]?table\.(tsv|csv|txt)(\.gz)?$',
    r'raw[-_]?counts?\.(tsv|csv|txt)(\.gz)?$',
    
    # Normalized expression
    r'(tpm|fpkm|rpkm|cpm)\.(tsv|csv|txt)(\.gz)?$',
    r'(tpm|fpkm|rpkm|cpm)[-_]?matrix\.(tsv|csv|txt)(\.gz)?$',
    r'normalized[-_]?(counts?|expression)?\.(tsv|csv|txt)(\.gz)?$',
    r'vst[-_]?counts?\.(tsv|csv|txt)(\.gz)?$',
    r'log[-_]?cpm\.(tsv|csv|txt)(\.gz)?$',
    
    # Expression matrices (generic naming)
    r'expression[-_]?(matrix|data|table)?\.(tsv|csv|txt)(\.gz)?$',
    r'gene[-_]?expression\.(tsv|csv|txt)(\.gz)?$',
    r'transcript[-_]?expression\.(tsv|csv|txt)(\.gz)?$',
    
    # Quantification tool outputs
    r'featureCounts.*\.(txt|tsv|csv)(\.gz)?$',
    r'ReadsPerGene\.out\.tab(\.gz)?$',
    r'quant\.sf(\.gz)?$',  # Salmon
    r'abundance\.(tsv|h5)(\.gz)?$',  # Kallisto
    r'rsem.*\.(genes|isoforms)\.results(\.gz)?$',
    r'htseq[-_]?counts?\.(txt|tsv)(\.gz)?$',
    
    # DESeq2 / edgeR outputs
    r'deseq2?.*\.(tsv|csv|txt)(\.gz)?$',
    r'edger.*\.(tsv|csv|txt)(\.gz)?$',
    r'diff[-_]?exp.*\.(tsv|csv|txt)(\.gz)?$',
    
    # Data matrices
    r'data[-_]?matrix\.(tsv|csv|txt)(\.gz)?$',
    r'processed[-_]?data\.(tsv|csv|txt)(\.gz)?$',
    
    # Raw archives
    r'_RAW\.tar$',
    r'supplementary.*\.tar(\.gz)?$',
]

# === MICROARRAY PATTERNS ===
MICROARRAY_PATTERNS = [
    # Affymetrix
    r'\.CEL(\.gz)?$',
    r'\.cel(\.gz)?$',
    r'\.CHP(\.gz)?$',
    r'\.chp(\.gz)?$',
    r'\.EXP(\.gz)?$',
    r'\.exp(\.gz)?$',
    r'\.CDF$',
    r'\.cdf$',
    
    # Illumina BeadArray
    r'\.idat(\.gz)?$',
    r'\.bgx(\.gz)?$',
    r'sample[-_]?probe[-_]?profile\.(txt|tsv)(\.gz)?$',
    r'control[-_]?probe[-_]?profile\.(txt|tsv)(\.gz)?$',
    
    # Agilent
    r'\.txt\.gz$',  # Agilent raw files often just .txt.gz
    r'US\d+.*\.txt(\.gz)?$',  # Agilent array IDs
    
    # Processed microarray data
    r'series[-_]?matrix\.txt(\.gz)?$',
    r'non[-_]?normalized\.(txt|tsv|csv)(\.gz)?$',
    r'normalized[-_]?signal\.(txt|tsv|csv)(\.gz)?$',
    r'rma[-_]?(normalized)?\.(txt|tsv|csv)(\.gz)?$',
    r'mas5[-_]?(normalized)?\.(txt|tsv|csv)(\.gz)?$',
    r'gcrma\.(txt|tsv|csv)(\.gz)?$',
    
    # Generic processed
    r'processed[-_]?data[-_]?matrix\.(txt|tsv|csv)(\.gz)?$',
    r'signal[-_]?intensities?\.(txt|tsv|csv)(\.gz)?$',
]

# === GENERIC/CATCH-ALL PATTERNS ===
GENERIC_PATTERNS = [
    # Any counts file
    r'.*counts?\.(tsv|csv|txt)(\.gz)?$',
    
    # Any expression file
    r'.*expression\.(tsv|csv|txt)(\.gz)?$',
    
    # Any matrix file
    r'.*matrix\.(tsv|csv|txt|mtx)(\.gz)?$',
    
    # Data files
    r'.*[-_]data\.(tsv|csv|txt)(\.gz)?$',
    
    # Supplementary tables
    r'supp(lementary)?[-_]?table.*\.(tsv|csv|txt|xlsx?)(\.gz)?$',
    
    # Excel files (often contain expression data)
    r'\.xlsx?$',
    
    # R data files
    r'\.RData$',
    r'\.Rda$',
    r'\.rda$',
    
    # Pickle (Python)
    r'\.pkl$',
    r'\.pickle$',
    
    # Parquet
    r'\.parquet$',
    
    # Raw archives (always grab these)
    r'_RAW\.tar$',
    r'[-_]raw\.tar(\.gz)?$',
    r'supplementary.*\.(tar|zip)(\.gz)?$',
]

# === METADATA/ANNOTATION PATTERNS ===
METADATA_PATTERNS = [
    r'sample[-_]?(info|metadata|annotation)s?\.(tsv|csv|txt|xlsx?)(\.gz)?$',
    r'(pheno|phenotype)[-_]?data\.(tsv|csv|txt)(\.gz)?$',
    r'clinical[-_]?data\.(tsv|csv|txt|xlsx?)(\.gz)?$',
    r'annotation\.(tsv|csv|txt)(\.gz)?$',
    r'design\.(tsv|csv|txt)(\.gz)?$',
]

# Files to always skip
SKIP_PATTERNS = [
    r'^filelist\.txt$',  # GEO manifest file
    r'\.html?$',
    r'\.pdf$',
    r'README',
    r'\.md5$',
    r'\.sha\d+$',
]


def compile_mode_patterns(
    mode: str,
    include_microarray: bool = False,
    include_metadata: bool = True,
    include_generic: bool = False,
) -> List[re.Pattern]:
    """
    Compile regex patterns based on mode and options.
    """
    patterns = []
    
    if mode == "single-cell":
        patterns.extend(SC_PATTERNS)
    elif mode == "bulk":
        patterns.extend(BULK_PATTERNS)
    else:  # "both"
        patterns.extend(SC_PATTERNS)
        patterns.extend(BULK_PATTERNS)
    
    if include_microarray:
        patterns.extend(MICROARRAY_PATTERNS)
    
    if include_metadata:
        patterns.extend(METADATA_PATTERNS)
    
    if include_generic:
        patterns.extend(GENERIC_PATTERNS)
    
    # Deduplicate
    patterns = list(set(patterns))
    
    return [re.compile(p, re.IGNORECASE) for p in patterns]


def compile_skip_patterns() -> List[re.Pattern]:
    return [re.compile(p, re.IGNORECASE) for p in SKIP_PATTERNS]


def is_target_file(name: str, compiled_patterns: List[re.Pattern]) -> bool:
    """Check if filename matches any of the target patterns."""
    if not compiled_patterns:
        return True  # No patterns means accept all
    return any(p.search(name) for p in compiled_patterns)


def should_skip_file(name: str, skip_patterns: List[re.Pattern]) -> bool:
    """Check if filename matches any skip pattern."""
    return any(p.search(name) for p in skip_patterns)


def get_matching_pattern(name: str, compiled_patterns: List[re.Pattern]) -> Optional[str]:
    """Return the pattern that matched, for debugging."""
    for p in compiled_patterns:
        if p.search(name):
            return p.pattern
    return None


def get_skip_reason(name: str, skip_patterns: List[re.Pattern]) -> Optional[str]:
    """Return the skip pattern that matched, for debugging."""
    for p in skip_patterns:
        if p.search(name):
            return p.pattern
    return None


@dataclass
class FileFilterResult:
    """Result of file filtering decision."""
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
    archives_from_filelist: Optional[Set[str]] = None,
) -> FileFilterResult:
    """
    Centralized file filtering logic.
    
    Returns a FileFilterResult with the decision and reason.
    """
    # Check skip patterns first (always applied)
    skip_reason = get_skip_reason(name, skip_patterns)
    if skip_reason:
        return FileFilterResult(
            filename=name,
            action="skip_pattern",
            reason=f"Matches skip pattern: {skip_reason}",
            matched_pattern=skip_reason,
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
    
    # Check if file is in filelist archives
    if archives_from_filelist and name in archives_from_filelist:
        return FileFilterResult(
            filename=name,
            action="download",
            reason="Archive contains matching files (from filelist.txt)",
        )
    
    # Check target patterns
    matched = get_matching_pattern(name, compiled_patterns)
    if matched:
        return FileFilterResult(
            filename=name,
            action="download",
            reason=f"Matches target pattern",
            matched_pattern=matched,
        )
    
    # No match
    return FileFilterResult(
        filename=name,
        action="skip_no_match",
        reason=f"No pattern match (checked {len(compiled_patterns)} patterns)",
    )


def log_filter_decision(result: FileFilterResult, gse: str, log_level: int = logging.DEBUG):
    """Log file filtering decision."""
    if result.action == "download":
        if result.matched_pattern:
            logging.log(log_level, f"{gse}:   ✓ {result.filename} -> {result.matched_pattern}")
        else:
            logging.log(log_level, f"{gse}:   ✓ {result.filename} ({result.reason})")
    else:
        logging.log(log_level, f"{gse}:   ✗ {result.filename} ({result.reason})")


def validate_patterns(patterns: List[re.Pattern], mode: str = "both") -> bool:
    """Validate that patterns are properly compiled."""
    if not patterns:
        logging.warning("No patterns loaded - will download all files!")
        return False
    
    # Test patterns against known good filenames (mode-specific)
    sc_test_files = {
        "matrix.mtx.gz": True,
        "barcodes.tsv.gz": True,
        "features.tsv.gz": True,
        "sample.h5ad": True,
        "data.loom": True,
        "seurat.rds": True,
    }
    
    bulk_test_files = {
        "gene_counts.tsv.gz": True,
        "tpm.csv": True,
        "fpkm.txt.gz": True,
    }
    
    common_skip_files = {
        "README.txt": False,
        "filelist.txt": False,
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
        logging.debug("Pattern validation: all tests passed")
    
    return all_pass


# -----------------------
# Search Term Builder
# -----------------------
# Simple, robust queries for NCBI E-utilities
# Note: The GDS database can be finicky with complex boolean queries

SC_QUERY_GROUP = (
    'single cell RNA seq OR scRNA-seq OR single nucleus OR '
    'snRNA-seq OR 10x genomics OR drop-seq'
)

BULK_QUERY_GROUP = (
    'RNA-seq OR bulk RNA seq OR transcriptome sequencing OR mRNA sequencing'
)

def build_search_term(mode: str, organism: str, user_term: Optional[str]) -> str:
    """
    Build search term for GEO/GDS database.
    
    Note: NCBI E-utilities can be picky about syntax.
    If the default search fails, use --search_term to provide a custom query.
    
    Examples of working custom queries:
        --search_term "scRNA-seq AND Rattus norvegicus[Organism] AND gse[ETYP]"
        --search_term "single cell AND rat[Organism] AND gse[ETYP]"
    """
    if user_term:
        term = user_term
    else:
        if mode == "single-cell":
            base = f"({SC_QUERY_GROUP})"
        elif mode == "bulk":
            base = f"({BULK_QUERY_GROUP})"
        else:
            base = f"({SC_QUERY_GROUP}) OR ({BULK_QUERY_GROUP})"
        
        # Simple organism filter
        term = f'{base} AND {organism}[Organism]'
    
    # Ensure we're searching for GSE entries
    if "gse[" not in term.lower() and "gse[ETYP]" not in term.lower():
        term = f"{term} AND gse[ETYP]"
    
    return term


# -----------------------
# GEO / Entrez Helpers
# -----------------------
def search_geo_series(term: str, retmax: int = 200, retstart: int = 0, retries: int = 3) -> Tuple[List[str], int]:
    """
    Search for GEO Series (GSE) records with retry logic.
    
    Returns:
        Tuple of (list of UIDs, total count)
    """
    from urllib.parse import quote
    
    for attempt in range(1, retries + 1):
        try:
            handle = Entrez.esearch(db="gds", term=term, retmax=retmax, retstart=retstart)
            record = Entrez.read(handle)
            handle.close()
            return record.get("IdList", []), int(record.get("Count", 0))
        except Exception as e:
            error_str = str(e)
            logging.warning(f"GEO search attempt {attempt}/{retries} failed: {e}")
            
            # If it's a 400 error, the query syntax might be wrong
            if "400" in error_str:
                logging.error(f"HTTP 400 Bad Request - search term may have invalid syntax")
                logging.error(f"Term was: {term}")
                # Try a simplified version on next attempt
                if attempt < retries:
                    # Remove complex operators and try again
                    term = term.replace("[All Fields]", "")
                    logging.info(f"Retrying with simplified term: {term}")
            
            if attempt < retries:
                time.sleep(2 * attempt)  # Exponential backoff
            else:
                logging.error(f"All {retries} search attempts failed")
                return [], 0
    
    return [], 0


def fetch_all_geo_ids(term: str) -> List[str]:
    """Fetch all GEO UIDs matching the search term."""
    all_geo_ids: List[str] = []
    retmax = 500
    retstart = 0
    t0 = time.time()
    
    # First search to get total count
    ids, total_count = search_geo_series(term, retmax=retmax, retstart=retstart)
    
    if total_count == 0:
        logging.warning(f"No results found. Try simplifying your search or checking the term.")
        return []
    
    all_geo_ids.extend(ids)
    retstart += retmax
    
    while retstart < total_count:
        ids, _ = search_geo_series(term, retmax=retmax, retstart=retstart)
        all_geo_ids.extend(ids)
        retstart += retmax
        time.sleep(ncbi_sleep_time())
    
    logging.info(f"Fetched {len(all_geo_ids)} UIDs in {time.time() - t0:.1f}s (total={total_count}).")
    return all_geo_ids


def uid_to_gse(uid: str) -> Optional[str]:
    try:
        handle = Entrez.esummary(db="gds", id=uid)
        record = Entrez.read(handle)
        handle.close()
        if record and isinstance(record, list):
            return record[0].get("Accession")
    except Exception as e:
        logging.warning(f"Error converting UID {uid}: {e}")
    return None


def batch_uid_to_gse(uids: List[str], batch_size: int = 200, sleep: float = 0.0) -> List[str]:
    gses: List[str] = []
    for i in tqdm(range(0, len(uids), batch_size), desc="UID→GSE (batched)"):
        chunk = uids[i:i + batch_size]
        try:
            handle = Entrez.esummary(db="gds", id=",".join(chunk))
            record = Entrez.read(handle)
            handle.close()
            docs = []
            if isinstance(record, dict):
                docs = record.get("DocumentSummarySet", {}).get("DocumentSummary", [])
            elif isinstance(record, list):
                docs = record
            for doc in docs:
                acc = doc.get("Accession")
                if acc and acc.startswith("GSE"):
                    gses.append(acc)
        except Exception as e:
            logging.warning(f"esummary batch failed for {len(chunk)} UIDs: {e}")
        if sleep:
            time.sleep(sleep)
    return gses


def construct_suppl_path(gse: str) -> str:
    head = gse[:-3] + "nnn"
    return f"/geo/series/{head}/{gse}/suppl/"


# -----------------------
# Metadata (GEOparse)
# -----------------------
def fetch_geo_metadata(gse: str, output_dir: str) -> Dict:
    md: Dict = {}
    try:
        geo = GEOparse.get_GEO(geo=gse, destdir=output_dir, silent=True, annotate_gpl=False)
        md["title"] = geo.metadata.get("title", [""])[0]
        md["summary"] = geo.metadata.get("summary", [""])[0]
        md["overall_design"] = geo.metadata.get("overall_design", [""])[0]
        md["pubmed_id"] = ";".join(geo.metadata.get("pubmed_id", []))
        phenotype_data = geo.phenotype_data
        md["samples"] = int(len(phenotype_data))
        phenotype_file = os.path.join(output_dir, f"{gse}_samples.tsv")
        phenotype_data.to_csv(phenotype_file, sep="\t")
        md["sample_metadata_file"] = phenotype_file
    except Exception as e:
        logging.warning(f"Error parsing GEO SOFT for {gse}: {e}")
    return md


# -----------------------
# File Utilities
# -----------------------
def md5sum(path: str, chunk_size: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def md5_path(path: str) -> str:
    return f"{path}.md5"


def write_md5_file(path: str, digest: str):
    try:
        with open(md5_path(path), "w") as f:
            f.write(digest + "\n")
    except Exception as e:
        logging.debug(f"Could not write md5 for {path}: {e}")


def read_md5_file(path: str) -> Optional[str]:
    try:
        with open(md5_path(path), "r") as f:
            return f.read().strip()
    except Exception:
        return None


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


# -----------------------
# Filelist.txt Parser
# -----------------------
@dataclass
class FilelistEntry:
    """Represents an entry from GEO's filelist.txt"""
    entry_type: str  # "Archive" or "File"
    name: str
    timestamp: str
    size: int
    file_type: str
    parent_archive: Optional[str] = None


def parse_filelist_txt(content: str) -> Tuple[List[FilelistEntry], Dict[str, List[FilelistEntry]]]:
    """
    Parse GEO's filelist.txt format.
    
    Returns:
        - List of all entries
        - Dict mapping archive names to their contained files
    """
    entries: List[FilelistEntry] = []
    archive_contents: Dict[str, List[FilelistEntry]] = {}
    current_archive: Optional[str] = None
    
    lines = content.strip().split('\n')
    
    for line in lines:
        # Skip header
        if line.startswith('#') or line.startswith('Archive/File'):
            continue
        
        parts = line.split('\t')
        if len(parts) < 5:
            continue
        
        entry_type = parts[0].strip()
        name = parts[1].strip()
        timestamp = parts[2].strip()
        try:
            size = int(parts[3].strip())
        except ValueError:
            size = 0
        file_type = parts[4].strip() if len(parts) > 4 else ""
        
        entry = FilelistEntry(
            entry_type=entry_type,
            name=name,
            timestamp=timestamp,
            size=size,
            file_type=file_type,
            parent_archive=current_archive if entry_type == "File" else None
        )
        entries.append(entry)
        
        if entry_type == "Archive":
            current_archive = name
            archive_contents[name] = []
        elif entry_type == "File" and current_archive:
            archive_contents[current_archive].append(entry)
    
    return entries, archive_contents


def fetch_filelist_txt(gse: str, timeout: int = 60) -> Optional[str]:
    """
    Fetch filelist.txt from GEO FTP for a given GSE.
    Returns content as string, or None if not found.
    """
    base_url = "ftp.ncbi.nlm.nih.gov"
    remote_path = construct_suppl_path(gse)
    
    try:
        ftp = ftplib.FTP(base_url, timeout=timeout)
        ftp.login()
        ftp.set_pasv(True)
        ftp.cwd(remote_path)
        
        # Check if filelist.txt exists
        files = []
        ftp.retrlines("NLST", files.append)
        
        if "filelist.txt" in files:
            content_lines = []
            ftp.retrlines("RETR filelist.txt", content_lines.append)
            ftp.quit()
            return '\n'.join(content_lines)
        
        ftp.quit()
        return None
        
    except ftplib.error_perm as e:
        if "550" in str(e):
            logging.debug(f"{gse}: No filelist.txt (550)")
        return None
    except Exception as e:
        logging.warning(f"Error fetching filelist.txt for {gse}: {e}")
        return None


def get_download_targets_from_filelist(
    filelist_content: str,
    compiled_patterns: List[re.Pattern],
    skip_patterns: List[re.Pattern],
    selective: bool = True,
) -> Tuple[Set[str], Dict[str, List[str]]]:
    """
    Analyze filelist.txt to determine what to download.
    
    Returns:
        - Set of archive names to download (we'll extract from these)
        - Dict mapping archive names to specific files we want from them
    """
    entries, archive_contents = parse_filelist_txt(filelist_content)
    
    archives_to_download: Set[str] = set()
    wanted_files_per_archive: Dict[str, List[str]] = {}
    
    for archive_name, files in archive_contents.items():
        wanted_files = []
        for f in files:
            if should_skip_file(f.name, skip_patterns):
                continue
            if not selective or is_target_file(f.name, compiled_patterns):
                wanted_files.append(f.name)
        
        if wanted_files:
            archives_to_download.add(archive_name)
            wanted_files_per_archive[archive_name] = wanted_files
    
    # Also check for standalone files (not in archives)
    for entry in entries:
        if entry.entry_type == "Archive":
            # Check if we should download even if we didn't find matching files inside
            # (pattern might match the archive itself, like _RAW.tar)
            if not selective or is_target_file(entry.name, compiled_patterns):
                archives_to_download.add(entry.name)
    
    return archives_to_download, wanted_files_per_archive


# -----------------------
# SRA Integration
# -----------------------
def extract_sra_accessions_from_geo(geo_obj) -> Dict[str, List[str]]:
    """
    Extract SRA accessions from a GEOparse object.
    Returns dict mapping GSM -> list of SRX accessions.
    """
    gsm_to_srx: Dict[str, List[str]] = {}
    
    try:
        for gsm_name, gsm in geo_obj.gsms.items():
            srx_list = []
            
            # Check relations for SRA links
            relations = gsm.metadata.get("relation", [])
            for rel in relations:
                if "SRA:" in rel or "sra:" in rel.lower():
                    # Extract SRX/SRR from relation string
                    # Format: "SRA: https://www.ncbi.nlm.nih.gov/sra?term=SRX..."
                    match = re.search(r'(SRX\d+|SRR\d+|SRP\d+)', rel, re.IGNORECASE)
                    if match:
                        srx_list.append(match.group(1).upper())
            
            # Also check supplementary_file for SRA links
            suppl_files = gsm.metadata.get("supplementary_file", [])
            for sf in suppl_files:
                if "sra" in sf.lower():
                    match = re.search(r'(SRX\d+|SRR\d+)', sf, re.IGNORECASE)
                    if match:
                        srx_list.append(match.group(1).upper())
            
            if srx_list:
                gsm_to_srx[gsm_name] = list(set(srx_list))
                
    except Exception as e:
        logging.warning(f"Error extracting SRA accessions: {e}")
    
    return gsm_to_srx


def fetch_sra_run_info_from_accession(accession: str) -> List[SRARunInfo]:
    """
    Fetch SRA run information for an accession (SRX, SRP, or SRR).
    Uses NCBI's efetch/esearch to get run info.
    """
    runs: List[SRARunInfo] = []
    
    try:
        # Search for the accession in SRA
        handle = Entrez.esearch(db="sra", term=accession, retmax=100)
        search_results = Entrez.read(handle)
        handle.close()
        
        if not search_results.get("IdList"):
            logging.debug(f"No SRA records found for {accession}")
            return runs
        
        time.sleep(ncbi_sleep_time())
        
        # Fetch the run info
        ids = ",".join(search_results["IdList"])
        handle = Entrez.efetch(db="sra", id=ids, rettype="full", retmode="xml")
        xml_content = handle.read()
        handle.close()
        
        # Parse XML
        root = ET.fromstring(xml_content)
        
        for exp_pkg in root.findall(".//EXPERIMENT_PACKAGE"):
            exp = exp_pkg.find(".//EXPERIMENT")
            run_set = exp_pkg.find(".//RUN_SET")
            
            if exp is None or run_set is None:
                continue
            
            exp_acc = exp.get("accession", "")
            
            # Get library info
            lib = exp.find(".//LIBRARY_DESCRIPTOR")
            lib_layout = "SINGLE"
            lib_source = ""
            lib_strategy = ""
            
            if lib is not None:
                layout = lib.find(".//LIBRARY_LAYOUT")
                if layout is not None:
                    if layout.find("PAIRED") is not None:
                        lib_layout = "PAIRED"
                source = lib.find("LIBRARY_SOURCE")
                if source is not None:
                    lib_source = source.text or ""
                strategy = lib.find("LIBRARY_STRATEGY")
                if strategy is not None:
                    lib_strategy = strategy.text or ""
            
            # Get platform info
            platform_elem = exp.find(".//PLATFORM")
            platform = ""
            instrument = ""
            if platform_elem is not None:
                for child in platform_elem:
                    platform = child.tag
                    inst = child.find("INSTRUMENT_MODEL")
                    if inst is not None:
                        instrument = inst.text or ""
                    break
            
            # Get sample info
            sample = exp_pkg.find(".//SAMPLE")
            sample_acc = sample.get("accession", "") if sample is not None else ""
            
            # Get study info
            study = exp_pkg.find(".//STUDY")
            study_acc = study.get("accession", "") if study is not None else ""
            
            # Get biosample
            biosample = ""
            ext_ids = exp_pkg.findall(".//EXTERNAL_ID")
            for ext_id in ext_ids:
                if ext_id.get("namespace") == "BioSample":
                    biosample = ext_id.text or ""
                    break
            
            # Process each run
            for run in run_set.findall(".//RUN"):
                run_acc = run.get("accession", "")
                spots = int(run.get("total_spots", 0) or 0)
                bases = int(run.get("total_bases", 0) or 0)
                size_bytes = int(run.get("size", 0) or 0)
                
                run_info = SRARunInfo(
                    run_accession=run_acc,
                    experiment_accession=exp_acc,
                    sample_accession=sample_acc,
                    study_accession=study_acc,
                    biosample=biosample,
                    platform=platform,
                    instrument=instrument,
                    library_layout=lib_layout,
                    library_source=lib_source,
                    library_strategy=lib_strategy,
                    spots=spots,
                    bases=bases,
                    size_mb=size_bytes / (1024 * 1024),
                )
                runs.append(run_info)
        
    except Exception as e:
        logging.warning(f"Error fetching SRA info for {accession}: {e}")
    
    return runs


def fetch_all_sra_runs_for_gse(gsm_to_srx: Dict[str, List[str]]) -> Tuple[List[SRARunInfo], Dict[str, List[str]]]:
    """
    Fetch SRA run info for all SRX accessions found in a GSE.
    Returns:
        - List of all SRARunInfo objects
        - Dict mapping GSM -> list of SRR accessions
    """
    all_runs: List[SRARunInfo] = []
    gsm_to_srr: Dict[str, List[str]] = {}
    
    # Deduplicate accessions
    all_accessions = set()
    acc_to_gsm: Dict[str, str] = {}
    for gsm, accs in gsm_to_srx.items():
        for acc in accs:
            all_accessions.add(acc)
            acc_to_gsm[acc] = gsm
    
    for acc in tqdm(all_accessions, desc="Fetching SRA run info", leave=False):
        runs = fetch_sra_run_info_from_accession(acc)
        gsm = acc_to_gsm.get(acc)
        
        for run in runs:
            run.gsm = gsm
            all_runs.append(run)
            
            if gsm:
                if gsm not in gsm_to_srr:
                    gsm_to_srr[gsm] = []
                gsm_to_srr[gsm].append(run.run_accession)
        
        time.sleep(ncbi_sleep_time())
    
    return all_runs, gsm_to_srr


def write_sra_run_table(runs: List[SRARunInfo], output_path: str):
    """Write SRA run information to a TSV file."""
    if not runs:
        return
    
    fieldnames = [
        "run_accession", "experiment_accession", "sample_accession",
        "study_accession", "biosample", "gsm", "platform", "instrument",
        "library_layout", "library_source", "library_strategy",
        "spots", "bases", "size_mb"
    ]
    
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for run in runs:
            writer.writerow({
                "run_accession": run.run_accession,
                "experiment_accession": run.experiment_accession,
                "sample_accession": run.sample_accession,
                "study_accession": run.study_accession,
                "biosample": run.biosample,
                "gsm": run.gsm or "",
                "platform": run.platform,
                "instrument": run.instrument,
                "library_layout": run.library_layout,
                "library_source": run.library_source,
                "library_strategy": run.library_strategy,
                "spots": run.spots,
                "bases": run.bases,
                "size_mb": f"{run.size_mb:.2f}",
            })


def check_sra_tools_available() -> bool:
    """Check if sra-tools (prefetch, fasterq-dump) are available."""
    try:
        result = subprocess.run(["prefetch", "--version"], capture_output=True, text=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False


def download_sra_fastq(
    run_accession: str,
    output_dir: str,
    use_fasterq_dump: bool = True,
    threads: int = 4,
) -> List[str]:
    """
    Download FASTQ files for an SRA run using sra-tools.
    Returns list of downloaded file paths.
    """
    downloaded: List[str] = []
    ensure_dir(output_dir)
    
    try:
        # First prefetch the SRA file
        logging.info(f"Prefetching {run_accession}...")
        prefetch_cmd = ["prefetch", "-O", output_dir, run_accession]
        result = subprocess.run(prefetch_cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            logging.warning(f"prefetch failed for {run_accession}: {result.stderr}")
            return downloaded
        
        # Then convert to FASTQ
        if use_fasterq_dump:
            logging.info(f"Converting {run_accession} to FASTQ...")
            sra_file = os.path.join(output_dir, run_accession, f"{run_accession}.sra")
            if not os.path.exists(sra_file):
                sra_file = os.path.join(output_dir, f"{run_accession}.sra")
            
            fasterq_cmd = [
                "fasterq-dump",
                "--outdir", output_dir,
                "--threads", str(threads),
                "--split-files",
                sra_file if os.path.exists(sra_file) else run_accession
            ]
            result = subprocess.run(fasterq_cmd, capture_output=True, text=True, cwd=output_dir)
            
            if result.returncode != 0:
                logging.warning(f"fasterq-dump failed for {run_accession}: {result.stderr}")
            else:
                # Find the output files
                for fname in os.listdir(output_dir):
                    if fname.startswith(run_accession) and fname.endswith(".fastq"):
                        downloaded.append(os.path.join(output_dir, fname))
                
                # Optionally gzip the files
                for fq in downloaded:
                    if not fq.endswith(".gz"):
                        subprocess.run(["gzip", fq], capture_output=True)
                        if os.path.exists(f"{fq}.gz"):
                            downloaded[downloaded.index(fq)] = f"{fq}.gz"
        
    except Exception as e:
        logging.warning(f"Error downloading SRA {run_accession}: {e}")
    
    return downloaded


# -----------------------
# GSM Supplementary Files
# -----------------------
def extract_gsm_supplementary_files(geo_obj) -> List[GSMSupplementaryFile]:
    """
    Extract supplementary file information from individual GSM samples.
    These are files linked at the sample level, not the series level.
    """
    gsm_files: List[GSMSupplementaryFile] = []
    
    try:
        for gsm_name, gsm in geo_obj.gsms.items():
            suppl_files = gsm.metadata.get("supplementary_file", [])
            
            for sf in suppl_files:
                # Skip SRA links (handled separately)
                if "sra" in sf.lower() and "SRR" not in sf.upper():
                    continue
                
                # Parse FTP path
                # Format: ftp://ftp.ncbi.nlm.nih.gov/geo/samples/GSMnnn/GSM123456/suppl/filename.txt.gz
                if sf.startswith("ftp://"):
                    filename = os.path.basename(sf)
                    gsm_file = GSMSupplementaryFile(
                        gsm=gsm_name,
                        filename=filename,
                        ftp_path=sf,
                    )
                    gsm_files.append(gsm_file)
                elif sf.startswith("NONE"):
                    continue
                else:
                    # Some entries are just filenames, construct the FTP path
                    filename = sf
                    gsm_num = gsm_name[3:]  # Remove "GSM" prefix
                    head = gsm_name[:-3] + "nnn"  # GSMnnn
                    ftp_path = f"ftp://ftp.ncbi.nlm.nih.gov/geo/samples/{head}/{gsm_name}/suppl/{filename}"
                    gsm_file = GSMSupplementaryFile(
                        gsm=gsm_name,
                        filename=filename,
                        ftp_path=ftp_path,
                    )
                    gsm_files.append(gsm_file)
                    
    except Exception as e:
        logging.warning(f"Error extracting GSM supplementary files: {e}")
    
    return gsm_files


def download_gsm_supplementary_file(
    gsm_file: GSMSupplementaryFile,
    local_dir: str,
    compiled_patterns: List[re.Pattern],
    skip_patterns: List[re.Pattern],
    selective: bool = True,
    verify_checksums: bool = False,
    resume: bool = False,
) -> Optional[str]:
    """
    Download a single GSM supplementary file.
    Returns the local path if downloaded, None otherwise.
    """
    # Check if we should download this file
    if selective:
        if should_skip_file(gsm_file.filename, skip_patterns):
            return None
        if not is_target_file(gsm_file.filename, compiled_patterns):
            return None
    
    # Create GSM subdirectory
    gsm_dir = os.path.join(local_dir, gsm_file.gsm)
    ensure_dir(gsm_dir)
    
    local_path = os.path.join(gsm_dir, gsm_file.filename)
    
    # Check if already exists
    if os.path.exists(local_path):
        if resume:
            return local_path
        if verify_checksums:
            existing_md5 = read_md5_file(local_path)
            if existing_md5:
                return local_path
    
    # Parse FTP URL
    # ftp://ftp.ncbi.nlm.nih.gov/geo/samples/GSMnnn/GSM123456/suppl/filename.txt.gz
    try:
        url_parts = gsm_file.ftp_path.replace("ftp://", "").split("/", 1)
        host = url_parts[0]
        remote_path = "/" + url_parts[1]
        remote_dir = os.path.dirname(remote_path)
        remote_file = os.path.basename(remote_path)
        
        ftp = ftplib.FTP(host, timeout=300)
        ftp.login()
        ftp.set_pasv(True)
        ftp.cwd(remote_dir)
        
        tmp_path = local_path + ".part"
        with open(tmp_path, "wb") as f:
            ftp.retrbinary(f"RETR {remote_file}", f.write)
        
        os.replace(tmp_path, local_path)
        ftp.quit()
        
        if verify_checksums:
            write_md5_file(local_path, md5sum(local_path))
        
        return local_path
        
    except Exception as e:
        logging.warning(f"Error downloading GSM file {gsm_file.filename}: {e}")
        return None


def download_all_gsm_supplementary_files(
    gsm_files: List[GSMSupplementaryFile],
    local_dir: str,
    compiled_patterns: List[re.Pattern],
    skip_patterns: List[re.Pattern],
    selective: bool = True,
    verify_checksums: bool = False,
    resume: bool = False,
    workers: int = 0,
) -> Tuple[List[str], List[str]]:
    """
    Download all GSM supplementary files.
    Returns:
        - List of downloaded file paths
        - List of skipped filenames
    """
    downloaded: List[str] = []
    skipped: List[str] = []
    
    if workers and workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = []
            for gsm_file in gsm_files:
                fut = ex.submit(
                    download_gsm_supplementary_file,
                    gsm_file, local_dir, compiled_patterns, skip_patterns,
                    selective, verify_checksums, resume
                )
                futures.append((fut, gsm_file))
            
            for fut, gsm_file in tqdm(futures, desc="Downloading GSM files", leave=False):
                try:
                    result = fut.result()
                    if result:
                        downloaded.append(result)
                    else:
                        skipped.append(gsm_file.filename)
                except Exception as e:
                    logging.warning(f"Failed to download {gsm_file.filename}: {e}")
                    skipped.append(gsm_file.filename)
    else:
        for gsm_file in tqdm(gsm_files, desc="Downloading GSM files", leave=False):
            result = download_gsm_supplementary_file(
                gsm_file, local_dir, compiled_patterns, skip_patterns,
                selective, verify_checksums, resume
            )
            if result:
                downloaded.append(result)
            else:
                skipped.append(gsm_file.filename)
            time.sleep(0.1)  # Gentle rate limiting
    
    return downloaded, skipped


def write_gsm_file_table(gsm_files: List[GSMSupplementaryFile], output_path: str):
    """Write GSM supplementary file information to a TSV file."""
    if not gsm_files:
        return
    
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["gsm", "filename", "ftp_path"], delimiter="\t")
        writer.writeheader()
        for gf in gsm_files:
            writer.writerow({
                "gsm": gf.gsm,
                "filename": gf.filename,
                "ftp_path": gf.ftp_path,
            })


# -----------------------
# Archive Extraction
# -----------------------
def extract_and_scan_archives(
    local_dir: str,
    compiled_patterns: List[re.Pattern],
    skip_patterns: List[re.Pattern],
    discovery_mode: bool = False,
) -> Tuple[List[str], List[str]]:
    """
    Extract .tar/.tar.gz/.tgz, .zip, and loose .gz files.
    
    Returns:
        - List of extracted file paths
        - List of skipped file names (for discovery logging)
    """
    extracted: List[str] = []
    skipped: List[str] = []

    for root, _, files in os.walk(local_dir):
        for fname in files:
            fpath = os.path.join(root, fname)

            # tar archives
            if fname.endswith((".tar", ".tar.gz", ".tgz")):
                try:
                    with tarfile.open(fpath, "r:*") as tar:
                        extract_dir = os.path.join(root, fname.rsplit(".", 1)[0])
                        if fname.endswith(".tar.gz") or fname.endswith(".tgz"):
                            extract_dir = os.path.join(root, fname.replace(".tar.gz", "").replace(".tgz", ""))
                        ensure_dir(extract_dir)
                        
                        for member in tar.getmembers():
                            if not member.isfile():
                                continue
                            name = os.path.basename(member.name)
                            
                            if should_skip_file(name, skip_patterns):
                                skipped.append(name)
                                continue
                            
                            if discovery_mode or is_target_file(name, compiled_patterns) or not compiled_patterns:
                                member.name = os.path.normpath(member.name).lstrip(os.sep)
                                tar.extract(member, extract_dir)
                                extracted.append(os.path.join(extract_dir, member.name))
                            else:
                                skipped.append(name)
                                
                except Exception as e:
                    logging.warning(f"Error extracting tar {fname}: {e}")

            # zip archives
            elif fname.endswith(".zip"):
                try:
                    with zipfile.ZipFile(fpath, "r") as z:
                        extract_dir = os.path.join(root, fname[:-4])
                        ensure_dir(extract_dir)
                        
                        for member in z.namelist():
                            base = os.path.basename(member)
                            if not base:
                                continue
                            
                            if should_skip_file(base, skip_patterns):
                                skipped.append(base)
                                continue
                            
                            if discovery_mode or is_target_file(base, compiled_patterns) or not compiled_patterns:
                                dest = os.path.normpath(os.path.join(extract_dir, member))
                                if not dest.startswith(os.path.abspath(extract_dir)):
                                    continue
                                z.extract(member, extract_dir)
                                extracted.append(dest)
                            else:
                                skipped.append(base)
                                
                except Exception as e:
                    logging.warning(f"Error extracting zip {fname}: {e}")

            # loose gz (not tar.gz)
            elif fname.endswith(".gz") and not fname.endswith(".tar.gz"):
                try:
                    out_path = fpath[:-3]
                    base = os.path.basename(out_path)
                    
                    if should_skip_file(base, skip_patterns):
                        skipped.append(base)
                        continue
                    
                    if discovery_mode or is_target_file(base, compiled_patterns):
                        with gzip.open(fpath, "rb") as fin, open(out_path, "wb") as fout:
                            fout.write(fin.read())
                        extracted.append(out_path)
                    else:
                        skipped.append(fname)
                        
                except Exception as e:
                    logging.warning(f"Error decompressing {fname}: {e}")

    return extracted, skipped


# -----------------------
# FTP Download Functions
# -----------------------
def list_ftp_tree(base_url: str, remote_path: str) -> List[Tuple[str, bool]]:
    """Return list of (remote_rel_path, is_dir) under remote_path."""
    results: List[Tuple[str, bool]] = []
    ftp = ftplib.FTP(base_url, timeout=1800)
    ftp.login()
    ftp.set_pasv(True)
    try:
        ftp.cwd(remote_path)
        def _walk(prefix: str = ""):
            items: List[str] = []
            ftp.retrlines("LIST", items.append)
            for item in items:
                parts = item.split(None, 8)
                if len(parts) < 9:
                    continue
                name = parts[8]
                is_dir = item.startswith("d")
                rel = f"{prefix}{name}"
                results.append((rel, is_dir))
                if is_dir:
                    cur = ftp.pwd()
                    try:
                        ftp.cwd(name)
                        _walk(prefix=f"{rel}/")
                    finally:
                        ftp.cwd(cur)
        _walk("")
    except ftplib.error_perm as e:
        if "550" in str(e):
            logging.info(f"No suppl files at {remote_path}")
        else:
            raise
    finally:
        try:
            ftp.quit()
        except Exception:
            pass
    return results


def download_one(base_url: str, remote_path: str, rel_file: str, local_path: str) -> str:
    """Download a single file (fresh FTP session per worker)."""
    ensure_dir(os.path.dirname(local_path))
    if os.path.exists(local_path):
        return local_path
    tmp = local_path + ".part"
    ftp = ftplib.FTP(base_url, timeout=1800)
    ftp.login()
    ftp.set_pasv(True)
    try:
        ftp.cwd(remote_path)
        parts = [p for p in rel_file.split("/") if p]
        for p in parts[:-1]:
            ftp.cwd(p)
        with open(tmp, "wb") as f:
            ftp.retrbinary("RETR " + parts[-1], f.write)
        os.replace(tmp, local_path)
        return local_path
    finally:
        try:
            ftp.quit()
        except Exception:
            pass


def download_geo_supplementary_files(
    gse: str,
    local_dir: str,
    compiled_patterns: List[re.Pattern],
    skip_patterns: List[re.Pattern],
    selective: bool = True,
    retries: int = 3,
    timeout: int = 1800,
    verify_checksums: bool = False,
    resume: bool = False,
    workers: int = 0,
    discovery_mode: bool = False,
    dry_run: bool = False,
) -> Tuple[List[str], List[str]]:
    """
    Download supplementary files for a GSE from NCBI FTP.
    
    Args:
        dry_run: If True, list files that would be downloaded without actually downloading
    
    Returns:
        - List of downloaded/extracted file paths (or would-be-downloaded in dry_run)
        - List of skipped file names (for discovery logging)
    """
    base_url = "ftp.ncbi.nlm.nih.gov"
    remote_path = construct_suppl_path(gse)
    downloaded: List[str] = []
    all_skipped: List[str] = []
    filter_results: List[FileFilterResult] = []  # Track all filter decisions

    def _should_skip(local_path: str) -> bool:
        if dry_run:
            return False  # In dry run, we want to see all files
        if not os.path.exists(local_path):
            return False
        if verify_checksums:
            md5_on_disk = read_md5_file(local_path)
            if md5_on_disk:
                return md5_on_disk == md5sum(local_path)
            return resume
        return True

    def _filter_and_track(name: str, archives_from_filelist: Set[str]) -> bool:
        """Filter a file and track the result. Returns True if should download."""
        result = filter_file(
            name=name,
            compiled_patterns=compiled_patterns,
            skip_patterns=skip_patterns,
            selective=selective,
            discovery_mode=discovery_mode,
            archives_from_filelist=archives_from_filelist,
        )
        filter_results.append(result)
        
        if result.action == "download":
            return True
        else:
            all_skipped.append(name)
            return False

    # First, try to get filelist.txt for better planning
    filelist_content = fetch_filelist_txt(gse, timeout=60)
    archives_from_filelist: Set[str] = set()
    
    if filelist_content:
        logging.debug(f"{gse}: Found filelist.txt, parsing...")
        archives_from_filelist, wanted_files = get_download_targets_from_filelist(
            filelist_content, compiled_patterns, skip_patterns, selective and not discovery_mode
        )
        if dry_run:
            logging.info(f"{gse}: [DRY-RUN] filelist.txt indicates {len(archives_from_filelist)} archives would be downloaded")
            for arch in archives_from_filelist:
                logging.info(f"{gse}: [DRY-RUN]   Archive: {arch}")
                if arch in wanted_files:
                    for wf in wanted_files[arch][:5]:  # Show first 5 files
                        logging.info(f"{gse}: [DRY-RUN]     -> {wf}")
                    if len(wanted_files.get(arch, [])) > 5:
                        logging.info(f"{gse}: [DRY-RUN]     ... and {len(wanted_files[arch]) - 5} more files")
        else:
            logging.info(f"{gse}: filelist.txt indicates {len(archives_from_filelist)} archives to download")

    for attempt in range(1, retries + 1):
        try:
            # Parallel branch
            if workers and workers > 1:
                entries = list_ftp_tree(base_url, remote_path)
                files = [rel for (rel, is_dir) in entries if not is_dir]
                
                # Log all available files in discovery mode or dry run
                if discovery_mode or dry_run:
                    logging.info(f"{gse}: Available files on FTP: {len(files)} files")
                
                # Filter based on patterns
                filtered = []
                for f in files:
                    fname = os.path.basename(f)
                    if _filter_and_track(fname, archives_from_filelist):
                        filtered.append(f)
                files = filtered

                targets: List[Tuple[str, str]] = []
                for rel in files:
                    lp = os.path.join(local_dir, rel)
                    if _should_skip(lp):
                        downloaded.append(lp)
                        continue
                    targets.append((rel, lp))

                # DRY RUN: report what would be downloaded with pattern details
                if dry_run:
                    logging.info(f"{gse}: [DRY-RUN] Would download {len(targets)} files:")
                    for i, (rel, lp) in enumerate(targets[:15]):
                        fname = os.path.basename(rel)
                        # Find matching result
                        for r in filter_results:
                            if r.filename == fname and r.action == "download":
                                if r.matched_pattern:
                                    logging.info(f"{gse}: [DRY-RUN]   ✓ {rel} (pattern: {r.matched_pattern})")
                                else:
                                    logging.info(f"{gse}: [DRY-RUN]   ✓ {rel} ({r.reason})")
                                break
                        else:
                            logging.info(f"{gse}: [DRY-RUN]   ✓ {rel}")
                    if len(targets) > 15:
                        logging.info(f"{gse}: [DRY-RUN]   ... and {len(targets) - 15} more files")
                    
                    # Show some skipped files
                    skipped_results = [r for r in filter_results if r.action != "download"]
                    if skipped_results and len(skipped_results) <= 10:
                        logging.info(f"{gse}: [DRY-RUN] Skipped {len(skipped_results)} files:")
                        for r in skipped_results[:5]:
                            logging.info(f"{gse}: [DRY-RUN]   ✗ {r.filename} ({r.reason})")
                    elif skipped_results:
                        logging.info(f"{gse}: [DRY-RUN] Skipped {len(skipped_results)} files (not shown)")
                    
                    downloaded.extend([lp for _, lp in targets])
                    return downloaded, all_skipped

                ensure_dir(local_dir)
                with ThreadPoolExecutor(max_workers=workers) as ex:
                    futs = [ex.submit(download_one, base_url, remote_path, rel, lp) for (rel, lp) in targets]
                    for fu in as_completed(futs):
                        try:
                            path = fu.result()
                            downloaded.append(path)
                            if verify_checksums:
                                write_md5_file(path, md5sum(path))
                        except Exception as e:
                            logging.warning(f"Parallel download failed: {e}")

            else:
                # Single-connection recursive downloader
                ftp = ftplib.FTP(base_url, timeout=timeout)
                ftp.login()
                ftp.set_pasv(True)
                try:
                    ftp.cwd(remote_path)
                except ftplib.error_perm as e:
                    if "550" in str(e):
                        logging.info(f"{gse}: No supplementary files (550).")
                        ftp.quit()
                        return [], []
                    raise

                if not dry_run:
                    ensure_dir(local_dir)

                # Collect files to download (used for both dry_run and actual download)
                files_to_download: List[Tuple[str, str]] = []  # (name, local_path)

                def list_recursive(current_local: str, prefix: str = ""):
                    """List files recursively without downloading."""
                    items: List[str] = []
                    ftp.retrlines("LIST", items.append)
                    for item in items:
                        parts = item.split(None, 8)
                        if len(parts) < 9:
                            continue
                        name = parts[8]
                        is_dir = item.startswith("d")
                        
                        if is_dir:
                            next_local = os.path.join(current_local, name)
                            cur = ftp.pwd()
                            try:
                                ftp.cwd(name)
                                list_recursive(next_local, f"{prefix}{name}/")
                            finally:
                                ftp.cwd(cur)
                        else:
                            if _filter_and_track(name, archives_from_filelist):
                                local_path = os.path.join(current_local, name)
                                files_to_download.append((f"{prefix}{name}", local_path))

                # First pass: list all files
                list_recursive(local_dir)
                
                # DRY RUN: report with pattern details
                if dry_run:
                    ftp.quit()
                    logging.info(f"{gse}: [DRY-RUN] Would download {len(files_to_download)} files:")
                    for rel, lp in files_to_download[:15]:
                        fname = os.path.basename(rel)
                        for r in filter_results:
                            if r.filename == fname and r.action == "download":
                                if r.matched_pattern:
                                    logging.info(f"{gse}: [DRY-RUN]   ✓ {rel} (pattern: {r.matched_pattern})")
                                else:
                                    logging.info(f"{gse}: [DRY-RUN]   ✓ {rel} ({r.reason})")
                                break
                        else:
                            logging.info(f"{gse}: [DRY-RUN]   ✓ {rel}")
                    if len(files_to_download) > 15:
                        logging.info(f"{gse}: [DRY-RUN]   ... and {len(files_to_download) - 15} more files")
                    
                    # Show some skipped files
                    skipped_results = [r for r in filter_results if r.action != "download"]
                    if skipped_results and len(skipped_results) <= 10:
                        logging.info(f"{gse}: [DRY-RUN] Skipped {len(skipped_results)} files:")
                        for r in skipped_results[:5]:
                            logging.info(f"{gse}: [DRY-RUN]   ✗ {r.filename} ({r.reason})")
                    elif skipped_results:
                        logging.info(f"{gse}: [DRY-RUN] Skipped {len(skipped_results)} files (not shown)")
                    
                    downloaded.extend([lp for _, lp in files_to_download])
                    return downloaded, all_skipped

                # Actual download
                ftp.cwd(remote_path)  # Reset to base path
                
                def download_recursive(current_local: str):
                    items: List[str] = []
                    ftp.retrlines("LIST", items.append)
                    for item in items:
                        parts = item.split(None, 8)
                        if len(parts) < 9:
                            continue
                        name = parts[8]
                        is_dir = item.startswith("d")
                        
                        if is_dir:
                            next_local = os.path.join(current_local, name)
                            ensure_dir(next_local)
                            cur = ftp.pwd()
                            try:
                                ftp.cwd(name)
                                download_recursive(next_local)
                            finally:
                                ftp.cwd(cur)
                        else:
                            # Check if we already filtered this file as download
                            # (We already ran list_recursive, so we know what to download)
                            local_path = os.path.join(current_local, name)
                            should_dl = any(
                                lp == local_path for _, lp in files_to_download
                            )
                            if not should_dl:
                                continue
                            
                            if _should_skip(local_path):
                                downloaded.append(local_path)
                                continue
                            
                            tmp_path = local_path + ".part"
                            with open(tmp_path, "wb") as f:
                                ftp.retrbinary(f"RETR {name}", f.write)
                            os.replace(tmp_path, local_path)
                            downloaded.append(local_path)
                            if verify_checksums:
                                write_md5_file(local_path, md5sum(local_path))

                download_recursive(local_dir)
                ftp.quit()

            # Extract archives and rescan
            if downloaded:
                extracted, skipped_in_archives = extract_and_scan_archives(
                    local_dir,
                    compiled_patterns if selective and not discovery_mode else [],
                    skip_patterns,
                    discovery_mode=discovery_mode,
                )
                downloaded.extend(extracted)
                all_skipped.extend(skipped_in_archives)

            return downloaded, all_skipped

        except Exception as e:
            logging.warning(f"FTP error for {gse} (attempt {attempt}/{retries}): {e}")
            time.sleep(2)

    return downloaded, all_skipped


# -----------------------
# Manifest Builder
# -----------------------
def build_manifest(
    base_dir: str,
    gse: str,
    selective_patterns: List[re.Pattern],
    mode: str,
    verify_checksums: bool = False,
    skipped_files: Optional[List[str]] = None,
) -> str:
    """
    Build manifest.json with file inventory and what was skipped.
    """
    gse_dir = os.path.join(base_dir, gse)
    manifest_path = os.path.join(gse_dir, "manifest.json")
    entries = []

    for root, _, files in os.walk(gse_dir):
        for fname in files:
            if fname == "manifest.json":
                continue
            fpath = os.path.join(root, fname)
            rel = os.path.relpath(fpath, gse_dir)
            try:
                size = os.path.getsize(fpath)
            except OSError:
                size = None

            digest = ""
            if verify_checksums and size and size < (1 << 31):
                try:
                    digest = md5sum(fpath)
                except Exception:
                    digest = ""

            selected = is_target_file(fname, selective_patterns) if selective_patterns else True
            extracted_from = None
            for parent in root.split(os.sep):
                if parent.endswith((".tar", ".tar.gz", ".tgz", ".zip")):
                    extracted_from = parent

            entries.append({
                "name": rel,
                "size": size,
                "md5": digest,
                "selected": selected,
                "mode": mode,
                "extracted_from": extracted_from,
            })

    manifest_data = {
        "GSE": gse,
        "files": entries,
        "skipped_files": list(set(skipped_files)) if skipped_files else [],
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)
    return manifest_path


# -----------------------
# Per-series Processing
# -----------------------
def process_geo_series(
    gse: str,
    metadata_writer: csv.DictWriter,
    download: bool,
    base_download_dir: str,
    mode: str,
    selective_download: bool = True,
    verify_checksums: bool = False,
    resume: bool = False,
    workers: int = 0,
    include_microarray: bool = False,
    include_generic: bool = False,
    discovery_mode: bool = False,
    include_sra: bool = False,
    sra_download: bool = False,
    sra_threads: int = 4,
    include_gsm_suppl: bool = False,
    run_report: Optional[HarvestRunReport] = None,
    dry_run: bool = False,
) -> GSEProcessingResult:
    """Process a single GEO Series with optional SRA and GSM support.
    
    Returns GSEProcessingResult with detailed statistics.
    """
    start_time = get_timestamp()
    t0 = time.time()
    
    result = GSEProcessingResult(
        gse=gse,
        start_time=start_time,
        end_time="",
        duration_sec=0,
        status="success",
    )
    
    gse_dir = os.path.join(base_download_dir, gse)
    ensure_dir(gse_dir)

    # Fetch metadata using GEOparse - keep the object for SRA/GSM extraction
    geo_obj = None
    metadata: Dict = {}
    try:
        geo_obj = GEOparse.get_GEO(geo=gse, destdir=gse_dir, silent=True, annotate_gpl=False)
        metadata["title"] = geo_obj.metadata.get("title", [""])[0]
        metadata["summary"] = geo_obj.metadata.get("summary", [""])[0]
        metadata["overall_design"] = geo_obj.metadata.get("overall_design", [""])[0]
        metadata["pubmed_id"] = ";".join(geo_obj.metadata.get("pubmed_id", []))
        phenotype_data = geo_obj.phenotype_data
        metadata["samples"] = int(len(phenotype_data))
        phenotype_file = os.path.join(gse_dir, f"{gse}_samples.tsv")
        phenotype_data.to_csv(phenotype_file, sep="\t")
        metadata["sample_metadata_file"] = phenotype_file
    except Exception as e:
        err_msg = f"Error parsing GEO SOFT for {gse}: {e}"
        logging.warning(err_msg)
        result.errors.append(err_msg)
    
    metadata["GSE"] = gse
    metadata["suppl_path"] = f"ftp://ftp.ncbi.nlm.nih.gov{construct_suppl_path(gse)}"

    downloaded_files: List[str] = []
    skipped_files: List[str] = []
    
    compiled_patterns = compile_mode_patterns(
        mode,
        include_microarray=include_microarray,
        include_metadata=True,
        include_generic=include_generic,
    )
    skip_patterns = compile_skip_patterns()

    # === Standard supplementary file download ===
    if download:
        dl_start = time.time()
        downloaded_files, skipped_files = download_geo_supplementary_files(
            gse=gse,
            local_dir=gse_dir,
            compiled_patterns=compiled_patterns,
            skip_patterns=skip_patterns,
            selective=selective_download,
            verify_checksums=verify_checksums,
            resume=resume,
            workers=workers,
            discovery_mode=discovery_mode,
            dry_run=dry_run,
        )
        dl_duration = time.time() - dl_start
        
        # Track results
        result.series_files_downloaded = len(downloaded_files)
        result.series_files_skipped = len(skipped_files)
        
        # Calculate total bytes
        for fp in downloaded_files:
            try:
                size = os.path.getsize(fp)
                result.total_bytes_downloaded += size
                
                # Log download event
                if run_report is not None:
                    run_report.download_events.append(DownloadEvent(
                        timestamp=get_timestamp(),
                        gse=gse,
                        filename=os.path.basename(fp),
                        source="series_suppl",
                        status="success",
                        size_bytes=size,
                        md5=read_md5_file(fp) if verify_checksums else None,
                    ))
            except OSError:
                pass
        
        metadata["downloaded_files"] = ";".join([os.path.relpath(p, gse_dir) for p in downloaded_files])
        metadata["download_success"] = len(downloaded_files) > 0
        metadata["files_skipped"] = len(skipped_files)
        
        if discovery_mode and skipped_files:
            logging.info(f"{gse}: Skipped files: {skipped_files[:20]}{'...' if len(skipped_files) > 20 else ''}")

    # === SRA Integration ===
    sra_runs: List[SRARunInfo] = []
    if include_sra and geo_obj is not None:
        logging.info(f"{gse}: Extracting SRA accessions...")
        gsm_to_srx = extract_sra_accessions_from_geo(geo_obj)
        
        if gsm_to_srx:
            logging.info(f"{gse}: Found {sum(len(v) for v in gsm_to_srx.values())} SRA accessions across {len(gsm_to_srx)} samples")
            sra_runs, gsm_to_srr = fetch_all_sra_runs_for_gse(gsm_to_srx)
            
            if sra_runs:
                result.sra_runs_found = len(sra_runs)
                
                # Write SRA run table
                sra_table_path = os.path.join(gse_dir, f"{gse}_sra_runs.tsv")
                write_sra_run_table(sra_runs, sra_table_path)
                metadata["sra_runs"] = len(sra_runs)
                metadata["sra_run_table"] = sra_table_path
                logging.info(f"{gse}: Found {len(sra_runs)} SRA runs")
                
                # Optionally download FASTQ files
                if sra_download:
                    if dry_run:
                        logging.info(f"{gse}: [DRY-RUN] Would download {len(sra_runs)} SRA runs:")
                        for run in sra_runs[:5]:
                            logging.info(f"{gse}: [DRY-RUN]   {run.run_accession} ({run.library_layout}, {run.size_mb:.1f} MB)")
                        if len(sra_runs) > 5:
                            logging.info(f"{gse}: [DRY-RUN]   ... and {len(sra_runs) - 5} more runs")
                        total_mb = sum(r.size_mb for r in sra_runs)
                        logging.info(f"{gse}: [DRY-RUN]   Total estimated size: {total_mb:.1f} MB")
                        metadata["sra_fastq_downloaded"] = 0
                    elif check_sra_tools_available():
                        sra_dir = os.path.join(gse_dir, "sra_fastq")
                        ensure_dir(sra_dir)
                        sra_downloaded = []
                        
                        for run in tqdm(sra_runs, desc=f"Downloading SRA for {gse}", leave=False):
                            fastq_files = download_sra_fastq(
                                run.run_accession, sra_dir,
                                use_fasterq_dump=True, threads=sra_threads
                            )
                            sra_downloaded.extend(fastq_files)
                            
                            # Log events
                            if run_report is not None:
                                for fq in fastq_files:
                                    try:
                                        size = os.path.getsize(fq)
                                        result.total_bytes_downloaded += size
                                        run_report.download_events.append(DownloadEvent(
                                            timestamp=get_timestamp(),
                                            gse=gse,
                                            filename=os.path.basename(fq),
                                            source="sra_fastq",
                                            status="success",
                                            size_bytes=size,
                                        ))
                                    except OSError:
                                        pass
                        
                        result.sra_fastqs_downloaded = len(sra_downloaded)
                        metadata["sra_fastq_downloaded"] = len(sra_downloaded)
                        downloaded_files.extend(sra_downloaded)
                    else:
                        err_msg = f"{gse}: sra-tools not available, skipping FASTQ download"
                        logging.warning(err_msg)
                        result.errors.append(err_msg)
                        metadata["sra_fastq_downloaded"] = 0
        else:
            metadata["sra_runs"] = 0

    # === GSM Supplementary Files ===
    gsm_files: List[GSMSupplementaryFile] = []
    if include_gsm_suppl and geo_obj is not None:
        logging.info(f"{gse}: Extracting GSM-level supplementary files...")
        gsm_files = extract_gsm_supplementary_files(geo_obj)
        
        if gsm_files:
            # Write GSM file table (do this even in dry_run for planning)
            if not dry_run:
                gsm_table_path = os.path.join(gse_dir, f"{gse}_gsm_files.tsv")
                write_gsm_file_table(gsm_files, gsm_table_path)
                metadata["gsm_file_table"] = gsm_table_path
            metadata["gsm_suppl_files"] = len(gsm_files)
            logging.info(f"{gse}: Found {len(gsm_files)} GSM-level supplementary files")
            
            if download:
                if dry_run:
                    # Filter to show what would be downloaded
                    would_download = []
                    for gf in gsm_files:
                        if should_skip_file(gf.filename, skip_patterns):
                            continue
                        if selective_download and not discovery_mode:
                            if not is_target_file(gf.filename, compiled_patterns):
                                continue
                        would_download.append(gf)
                    
                    logging.info(f"{gse}: [DRY-RUN] Would download {len(would_download)} GSM supplementary files:")
                    for gf in would_download[:10]:
                        logging.info(f"{gse}: [DRY-RUN]   {gf.gsm}/{gf.filename}")
                    if len(would_download) > 10:
                        logging.info(f"{gse}: [DRY-RUN]   ... and {len(would_download) - 10} more files")
                    metadata["gsm_files_downloaded"] = 0
                else:
                    gsm_dir = os.path.join(gse_dir, "gsm_suppl")
                    gsm_downloaded, gsm_skipped = download_all_gsm_supplementary_files(
                        gsm_files, gsm_dir, compiled_patterns, skip_patterns,
                        selective=selective_download and not discovery_mode,
                        verify_checksums=verify_checksums,
                        resume=resume,
                        workers=workers,
                    )
                    
                    result.gsm_files_downloaded = len(gsm_downloaded)
                    result.series_files_skipped += len(gsm_skipped)
                    
                    # Calculate bytes and log events
                    for fp in gsm_downloaded:
                        try:
                            size = os.path.getsize(fp)
                            result.total_bytes_downloaded += size
                            
                            if run_report is not None:
                                run_report.download_events.append(DownloadEvent(
                                    timestamp=get_timestamp(),
                                    gse=gse,
                                    filename=os.path.basename(fp),
                                    source="gsm_suppl",
                                    status="success",
                                    size_bytes=size,
                                ))
                        except OSError:
                            pass
                
                downloaded_files.extend(gsm_downloaded)
                skipped_files.extend(gsm_skipped)
                metadata["gsm_files_downloaded"] = len(gsm_downloaded)
        else:
            metadata["gsm_suppl_files"] = 0

    # === Build Manifest ===
    manifest_path = build_manifest(
        base_download_dir,
        gse,
        compiled_patterns if selective_download else [],
        mode,
        verify_checksums=verify_checksums,
        skipped_files=skipped_files,
    )
    metadata["manifest"] = manifest_path

    # Update downloaded files count
    if download:
        metadata["total_files_downloaded"] = len(downloaded_files)

    metadata_writer.writerow(metadata)
    
    # Finalize result
    result.end_time = get_timestamp()
    result.duration_sec = time.time() - t0
    
    # Determine final status
    if result.errors:
        if result.series_files_downloaded > 0 or result.gsm_files_downloaded > 0:
            result.status = "partial"
        else:
            result.status = "failed"
    
    return result


# -----------------------
# Main CLI
# -----------------------
def main():
    parser = argparse.ArgumentParser(
        description="Download GEO Series data with expanded pattern matching, SRA integration, and GSM support."
    )
    parser.add_argument("--mode", choices=["single-cell", "bulk", "both"], default="single-cell")
    parser.add_argument("--organism", type=str, default="Rattus norvegicus")
    parser.add_argument("--search_term", type=str, default=None)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--all_files", action="store_true", help="Download all files (ignore filtering).")
    parser.add_argument("--output_dir", type=str,
                        default=str(resolve_path(_config, _cfg("harvesting.geo_output_dir", "data/geo_harvest"))) if _config else "data/geo_harvest")
    parser.add_argument("--metadata_file", type=str,
                        default=_cfg("harvesting.geo_metadata_file", "geo_metadata.csv"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--email", type=str,
                        default=_cfg("harvesting.email", "you@example.com"))
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--batch-esummary", type=int,
                        default=_cfg("harvesting.geo_batch_esummary", 0))
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--verify-checksums", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--log_level", type=str, default="INFO",
                        help="Logging level (DEBUG, INFO, WARNING, ERROR)")
    parser.add_argument("--log_dir", type=str, default=None,
                        help="Directory for log files (auto-generates timestamped filename)")
    parser.add_argument("--log_file", type=str, default=None,
                        help="Explicit log file path (overrides --log_dir)")
    
    # Pattern options
    parser.add_argument("--include-microarray", action="store_true",
                        help="Include microarray file patterns (CEL, idat, etc.)")
    parser.add_argument("--include-generic", action="store_true",
                        help="Include generic catch-all patterns (more permissive)")
    parser.add_argument("--discovery", action="store_true",
                        help="Discovery mode: log all files, download everything")
    parser.add_argument("--show-patterns", action="store_true",
                        help="Show all patterns that will be used and exit")
    parser.add_argument("--test-pattern", type=str, default=None, metavar="FILENAME",
                        help="Test if a filename matches current patterns and exit")
    
    # SRA options
    parser.add_argument("--include-sra", action="store_true",
                        help="Extract and record SRA run information from GEO records")
    parser.add_argument("--sra-download", action="store_true",
                        help="Download FASTQ files via sra-tools (requires prefetch/fasterq-dump)")
    parser.add_argument("--sra-threads", type=int, default=4,
                        help="Threads for fasterq-dump (default: 4)")
    
    # GSM options
    parser.add_argument("--include-gsm-suppl", action="store_true",
                        help="Include sample-level (GSM) supplementary files")
    
    # Dry-run option
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be downloaded without actually downloading")
    
    args = parser.parse_args()

    # Handle pattern inspection options early (before full setup)
    if args.show_patterns or args.test_pattern:
        patterns = compile_mode_patterns(
            args.mode,
            include_microarray=args.include_microarray,
            include_generic=args.include_generic,
        )
        skip_patterns = compile_skip_patterns()
        
        if args.show_patterns:
            print(f"\n{'='*60}")
            print(f"PATTERN CONFIGURATION")
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
            print()
            return
        
        if args.test_pattern:
            filename = args.test_pattern
            print(f"\nTesting filename: '{filename}'")
            print(f"Mode: {args.mode}")
            print(f"Include microarray: {args.include_microarray}")
            print(f"Include generic: {args.include_generic}")
            print()
            
            # Check skip patterns first
            skip_match = get_matching_pattern(filename, skip_patterns)
            if skip_match:
                print(f"  ✗ SKIPPED - matches skip pattern: {skip_match}")
                return
            
            # Check target patterns
            target_match = get_matching_pattern(filename, patterns)
            if target_match:
                print(f"  ✓ MATCH - matches target pattern: {target_match}")
            else:
                print(f"  ✗ NO MATCH - does not match any of {len(patterns)} patterns")
                print("\n  Try with --include-generic to enable catch-all patterns")
            return

    setup_logging(args.log_level, log_dir=args.log_dir, log_file=args.log_file)
    init_entrez(args.email, args.api_key)
    
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
        include_sra=args.include_sra,
        sra_download=args.sra_download,
        include_gsm_suppl=args.include_gsm_suppl,
        discovery_mode=args.discovery,
    )
    
    # Set up file logging
    ensure_dir(args.output_dir)
    log_file = setup_file_logging(args.output_dir, run_id, args.log_level)
    logging.info(f"Starting harvest run: {run_id}")
    if _config is not None:
        logging.info(f"Config: loaded (project_root={_config.get('_project_root', '?')})")
    else:
        logging.info("Config: not found (using built-in defaults)")

    # Print configuration summary and compile patterns
    patterns = compile_mode_patterns(
        args.mode,
        include_microarray=args.include_microarray,
        include_generic=args.include_generic,
    )
    skip_patterns = compile_skip_patterns()
    
    logging.info(f"Mode: {args.mode}")
    logging.info(f"Patterns loaded: {len(patterns)} target patterns, {len(skip_patterns)} skip patterns")
    
    # Validate patterns
    validate_patterns(patterns, args.mode)
    
    # Show example patterns at debug level
    if logging.getLogger().level <= logging.DEBUG:
        logging.debug("Sample target patterns:")
        for p in patterns[:5]:
            logging.debug(f"  - {p.pattern}")
        if len(patterns) > 5:
            logging.debug(f"  ... and {len(patterns) - 5} more")
    
    if args.include_microarray:
        logging.info("Microarray patterns enabled (CEL, idat, etc.)")
    if args.include_generic:
        logging.info("Generic catch-all patterns enabled")
    if args.all_files:
        logging.info("All-files mode: pattern filtering DISABLED")
    if args.discovery:
        logging.info("Discovery mode: will download all files and log skipped ones")
    if args.include_sra:
        logging.info("SRA integration enabled")
        if args.sra_download:
            if check_sra_tools_available():
                logging.info("SRA download enabled (sra-tools found)")
            else:
                logging.warning("SRA download requested but sra-tools not found! Install with: conda install -c bioconda sra-tools")
    if args.include_gsm_suppl:
        logging.info("GSM-level supplementary file download enabled")
    if args.dry_run:
        logging.info("DRY-RUN MODE: Will show what would be downloaded without downloading")

    term = build_search_term(mode=args.mode, organism=args.organism, user_term=args.search_term)
    run_report.search_term = term
    logging.info(f"Search term: {term}")

    uids = fetch_all_geo_ids(term)
    
    # If no results and using default term, try a simpler fallback
    if not uids and not args.search_term:
        logging.warning("Primary search returned no results. Trying simplified fallback query...")
        
        # Build a very simple fallback query
        if args.mode == "single-cell":
            fallback_term = f"scRNA-seq AND {args.organism}[Organism] AND gse[ETYP]"
        elif args.mode == "bulk":
            fallback_term = f"RNA-seq AND {args.organism}[Organism] AND gse[ETYP]"
        else:
            fallback_term = f"RNA-seq AND {args.organism}[Organism] AND gse[ETYP]"
        
        logging.info(f"Fallback search term: {fallback_term}")
        run_report.search_term = fallback_term
        uids = fetch_all_geo_ids(fallback_term)
    
    logging.info(f"Total GEO Series (UIDs) found: {len(uids)}")
    run_report.total_gse_found = len(uids)
    
    if not uids:
        logging.error("No GEO Series found. Exiting.")
        logging.error("Tip: Try a custom search term with --search_term, e.g.:")
        logging.error("  --search_term 'single cell AND rat[Organism] AND gse[ETYP]'")
        run_report.end_time = get_timestamp()
        run_report.errors.append("No GEO Series found for search term")
        write_run_report(run_report, args.output_dir)
        return

    if args.limit > 0:
        uids = uids[:args.limit]
        logging.info(f"Limiting to first {len(uids)} records.")

    logging.info("Converting UIDs to GSE accessions...")
    sleep_time = ncbi_sleep_time()
    if args.batch_esummary and args.batch_esummary > 0:
        gse_list = batch_uid_to_gse(uids, batch_size=args.batch_esummary, sleep=0.0)
    else:
        gse_list = []
        for uid in tqdm(uids, desc="UID→GSE"):
            gse = uid_to_gse(uid)
            if gse and gse.startswith("GSE"):
                gse_list.append(gse)
            time.sleep(sleep_time)

    if not gse_list:
        logging.error("No valid GSE accessions resolved from UIDs. Exiting.")
        run_report.end_time = get_timestamp()
        run_report.errors.append("No valid GSE accessions resolved")
        write_run_report(run_report, args.output_dir)
        return

    logging.info(f"Sample GSEs: {', '.join(gse_list[:5])}{'...' if len(gse_list) > 5 else ''}")

    base_download_dir = os.path.join(args.output_dir, "geo_datasets")
    ensure_dir(base_download_dir)

    # Extended fieldnames for new features
    fieldnames = [
        "GSE", "title", "summary", "overall_design", "samples", "pubmed_id",
        "suppl_path", "sample_metadata_file", "manifest"
    ]
    # dry-run shows what would be downloaded, so include download fields
    if args.download or args.dry_run:
        fieldnames.extend(["downloaded_files", "download_success", "files_skipped", "total_files_downloaded"])
    if args.include_sra:
        fieldnames.extend(["sra_runs", "sra_run_table"])
        if args.sra_download:
            fieldnames.append("sra_fastq_downloaded")
    if args.include_gsm_suppl:
        fieldnames.extend(["gsm_suppl_files", "gsm_file_table"])
        if args.download or args.dry_run:
            fieldnames.append("gsm_files_downloaded")

    success_count = 0
    with open(args.metadata_file, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()

        for gse in tqdm(gse_list, desc="Processing GSE"):
            try:
                # dry-run implies we want to see what would be downloaded
                should_download = args.download or args.dry_run
                
                result = process_geo_series(
                    gse=gse,
                    metadata_writer=writer,
                    download=should_download,
                    base_download_dir=base_download_dir,
                    mode=args.mode,
                    selective_download=not args.all_files,
                    verify_checksums=args.verify_checksums,
                    resume=args.resume,
                    workers=args.workers,
                    include_microarray=args.include_microarray,
                    include_generic=args.include_generic,
                    discovery_mode=args.discovery,
                    include_sra=args.include_sra,
                    sra_download=args.sra_download,
                    sra_threads=args.sra_threads,
                    include_gsm_suppl=args.include_gsm_suppl,
                    run_report=run_report,
                    dry_run=args.dry_run,
                )
                
                # Update run report
                run_report.gse_results.append(result)
                run_report.total_gse_processed += 1
                
                if result.status == "success":
                    run_report.total_gse_succeeded += 1
                    success_count += 1
                elif result.status == "partial":
                    run_report.total_gse_succeeded += 1  # Count partial as success
                    success_count += 1
                else:
                    run_report.total_gse_failed += 1
                
                # Aggregate stats
                run_report.total_files_downloaded += (
                    result.series_files_downloaded + 
                    result.gsm_files_downloaded + 
                    result.sra_fastqs_downloaded
                )
                run_report.total_files_skipped += result.series_files_skipped
                run_report.total_bytes_downloaded += result.total_bytes_downloaded
                run_report.total_sra_runs += result.sra_runs_found
                run_report.total_sra_fastqs += result.sra_fastqs_downloaded
                
                for err in result.errors:
                    run_report.errors.append(f"{gse}: {err}")
                    
            except Exception as e:
                err_msg = f"Error processing {gse}: {e}"
                logging.error(err_msg)
                run_report.errors.append(err_msg)
                run_report.total_gse_failed += 1
                run_report.total_gse_processed += 1
                
            time.sleep(sleep_time)

    # Finalize run report
    run_report.end_time = get_timestamp()
    t_start = time.mktime(time.strptime(run_report.start_time[:19], "%Y-%m-%dT%H:%M:%S"))
    t_end = time.mktime(time.strptime(run_report.end_time[:19], "%Y-%m-%dT%H:%M:%S"))
    run_report.duration_sec = t_end - t_start
    
    # Write all reports
    write_run_report(run_report, args.output_dir)

    logging.info(f"Done. Successfully processed {success_count}/{len(gse_list)} Series.")
    logging.info(f"Metadata saved to: {args.metadata_file}")
    
    # Print final summary
    logging.info("=" * 60)
    if args.dry_run:
        logging.info("DRY-RUN COMPLETE (no files actually downloaded)")
    else:
        logging.info("HARVEST COMPLETE")
    logging.info("=" * 60)
    logging.info(f"Run ID: {run_id}")
    logging.info(f"GSE processed: {run_report.total_gse_processed}")
    logging.info(f"GSE succeeded: {run_report.total_gse_succeeded}")
    logging.info(f"GSE failed: {run_report.total_gse_failed}")
    if args.dry_run:
        logging.info(f"Files would download: {run_report.total_files_downloaded}")
        logging.info(f"Files would skip: {run_report.total_files_skipped}")
    else:
        logging.info(f"Files downloaded: {run_report.total_files_downloaded}")
        logging.info(f"Files skipped: {run_report.total_files_skipped}")
    gb = run_report.total_bytes_downloaded / (1024**3)
    logging.info(f"Total data: {gb:.2f} GB")
    if args.include_sra:
        logging.info(f"SRA runs found: {run_report.total_sra_runs}")
        logging.info(f"SRA FASTQs: {run_report.total_sra_fastqs}")
    logging.info("=" * 60)
    logging.info(f"Reports saved to: {args.output_dir}")
    logging.info(f"  - harvest_report_{run_id}.json (full details)")
    logging.info(f"  - download_log_{run_id}.tsv (per-file log)")
    logging.info(f"  - harvest_summary_{run_id}.txt (human-readable)")
    logging.info(f"  - harvest_{run_id}.log (full console log)")


if __name__ == "__main__":
    main()