#!/usr/bin/env python3
"""
ArrayExpress/BioStudies Harvester Test Suite
=============================================
Systematically test different approaches for:
1. BioStudies REST API search queries (Lucene syntax)
2. Study metadata retrieval
3. FTP connections and file listing
4. MAGE-TAB (IDF/SDRF) parsing
5. ENA integration for raw sequencing data
6. File pattern matching for expression matrices
7. Bioservices library integration

Run this BEFORE modifying the main harvester to verify approaches work.

Note: ArrayExpress was migrated to BioStudies on September 30, 2022.
All ArrayExpress data is now served via the BioStudies API.

Usage:
    python test_arrayexpress_approaches.py --test-all
    python test_arrayexpress_approaches.py --test-diagnose
    python test_arrayexpress_approaches.py --test-search
    python test_arrayexpress_approaches.py --test-ftp
    python test_arrayexpress_approaches.py --test-ena
    python test_arrayexpress_approaches.py --accession E-MTAB-5920
"""

import os
import sys
import time
import json
import ftplib
import argparse
import logging
import urllib.request
import urllib.parse
import urllib.error
import re
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from functools import wraps

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# =============================================================================
# CONSTANTS
# =============================================================================

# BioStudies API endpoints (for retrieving study metadata)
BIOSTUDIES_API_BASE = "https://www.ebi.ac.uk/biostudies/api/v1"
BIOSTUDIES_ARRAYEXPRESS = "https://www.ebi.ac.uk/biostudies/arrayexpress"

# EBI Search API (for searching across BioStudies/ArrayExpress)
# Note: BioStudies does NOT have a public search endpoint - use EBI Search instead
EBI_SEARCH_API = "https://www.ebi.ac.uk/ebisearch/ws/rest"
EBI_SEARCH_BIOSTUDIES_DOMAIN = "biostudies"

# ENA API endpoints
ENA_PORTAL_API = "https://www.ebi.ac.uk/ena/portal/api"
ENA_BROWSER_API = "https://www.ebi.ac.uk/ena/browser/api"

# FTP servers
BIOSTUDIES_FTP = "ftp.ebi.ac.uk"
ENA_FTP = "ftp.sra.ebi.ac.uk"

# Rate limiting
REQUESTS_PER_SECOND = 3
MIN_REQUEST_INTERVAL = 1.0 / REQUESTS_PER_SECOND

# Try imports for optional libraries
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    logger.warning("requests not installed. Install with: pip install requests")

try:
    from bioservices import ArrayExpress as BioservicesAE
    BIOSERVICES_AVAILABLE = True
except ImportError:
    BIOSERVICES_AVAILABLE = False
    logger.warning("bioservices not installed. Install with: pip install bioservices")


# =============================================================================
# FILE PATTERN DEFINITIONS (FIXED)
# =============================================================================

# Patterns for single-cell data files
SC_PATTERNS = [
    r'matrix\.mtx(\.gz)?$',                    # 10x matrix files
    r'barcodes\.tsv(\.gz)?$',                  # 10x barcodes
    r'(features|genes)\.tsv(\.gz)?$',          # 10x features/genes
    r'\.h5ad$',                                # AnnData format
    r'\.loom$',                                # Loom format
    r'filtered.*\.h5$',                        # Filtered HDF5 (10x)
    r'raw.*\.h5$',                             # Raw HDF5 (10x)
    r'cellranger.*\.tar\.gz$',                 # Cell Ranger output
    r'10x.*\.tar\.gz$',                        # 10x archives
    r'_matrix\.tar\.gz$',                      # Matrix archives
]

# Patterns for bulk RNA-seq data files (FIXED - removed problematic anchor alternations)
BULK_PATTERNS = [
    # Count matrices - match anywhere in filename
    r'(gene|transcript)[_-]?counts?\.(tsv|csv|txt)(\.gz)?$',
    r'counts?\.(tsv|csv|txt)(\.gz)?$',
    
    # Normalized expression - simplified patterns that work with search()
    r'(^|.*[/_])tpm\.(tsv|csv|txt)(\.gz)?$',
    r'(^|.*[/_])fpkm\.(tsv|csv|txt)(\.gz)?$', 
    r'(^|.*[/_])rpkm\.(tsv|csv|txt)(\.gz)?$',
    r'(^|.*[/_])cpm\.(tsv|csv|txt)(\.gz)?$',
    
    # Tool-specific outputs
    r'featureCounts.*\.(txt|tsv)(\.gz)?$',
    r'rsem[._].*\.results(\.gz)?$',
    r'\.results(\.gz)?$',
    
    # Salmon/Kallisto outputs - simplified patterns
    r'(^|.*[/_])quant\.sf(\.gz)?$',
    r'\.sf(\.gz)?$',
    r'(^|.*[/_])abundance\.(tsv|h5)(\.gz)?$',
    r'abundance\.(tsv|h5)(\.gz)?$',
    
    # Expression matrices
    r'expression[_-]?matrix\.(tsv|csv|txt)(\.gz)?$',
    r'gene[_-]?expression\.(tsv|csv|txt)(\.gz)?$',
]

# Pattern for RDS files (could be either SC or bulk, often Seurat objects)
RDS_PATTERN = r'\.rds$'

# Compile patterns for efficiency
SC_COMPILED = [re.compile(p, re.IGNORECASE) for p in SC_PATTERNS]
BULK_COMPILED = [re.compile(p, re.IGNORECASE) for p in BULK_PATTERNS]
RDS_COMPILED = re.compile(RDS_PATTERN, re.IGNORECASE)


def classify_file(filename: str) -> str:
    """
    Classify a file as single-cell (SC), bulk (BULK), RDS, or NONE.
    
    Args:
        filename: The filename to classify
        
    Returns:
        Classification string: 'SC', 'BULK', 'RDS', or 'NONE'
    """
    # Check single-cell patterns first (more specific)
    for pattern in SC_COMPILED:
        if pattern.search(filename):
            return "SC"
    
    # Check bulk patterns
    for pattern in BULK_COMPILED:
        if pattern.search(filename):
            return "BULK"
    
    # Check RDS pattern (ambiguous - could be either)
    if RDS_COMPILED.search(filename):
        return "RDS"
    
    return "NONE"


# =============================================================================
# RATE LIMITING DECORATOR
# =============================================================================

_last_request_time = [0.0]

def rate_limited(func):
    """Decorator to enforce rate limiting"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        elapsed = time.time() - _last_request_time[0]
        if elapsed < MIN_REQUEST_INTERVAL:
            time.sleep(MIN_REQUEST_INTERVAL - elapsed)
        _last_request_time[0] = time.time()
        return func(*args, **kwargs)
    return wrapper


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class TestResult:
    """Result of a single test"""
    test_name: str
    success: bool
    message: str
    details: Dict = field(default_factory=dict)
    duration_sec: float = 0.0


@dataclass 
class TestSuite:
    """Collection of test results"""
    results: List[TestResult] = field(default_factory=list)
    
    def add(self, result: TestResult):
        self.results.append(result)
        status = "✓ PASS" if result.success else "✗ FAIL"
        logger.info(f"{status}: {result.test_name} - {result.message}")
        if result.details:
            for k, v in result.details.items():
                # Truncate long values for display
                v_str = str(v)
                if len(v_str) > 200:
                    v_str = v_str[:200] + "..."
                logger.info(f"    {k}: {v_str}")
    
    def summary(self):
        passed = sum(1 for r in self.results if r.success)
        failed = sum(1 for r in self.results if not r.success)
        logger.info(f"\n{'='*60}")
        logger.info(f"TEST SUMMARY: {passed} passed, {failed} failed")
        logger.info(f"{'='*60}")
        for r in self.results:
            status = "✓" if r.success else "✗"
            logger.info(f"  {status} {r.test_name}")
        return passed, failed


# =============================================================================
# SECTION 0: DIAGNOSTIC TESTS
# =============================================================================

def test_network_connectivity() -> TestResult:
    """Test basic network connectivity to EMBL-EBI services"""
    t0 = time.time()
    
    urls = [
        ("BioStudies Web", "https://www.ebi.ac.uk/biostudies"),
        ("BioStudies API", f"{BIOSTUDIES_API_BASE}/studies/E-MTAB-5920"),
        ("EBI Search API", f"{EBI_SEARCH_API}/{EBI_SEARCH_BIOSTUDIES_DOMAIN}?query=test&format=json&size=1"),
        ("ENA Portal", "https://www.ebi.ac.uk/ena/portal/api"),
        ("EBI FTP (HTTPS)", "https://ftp.ebi.ac.uk"),
    ]
    
    results = {}
    for name, url in urls:
        try:
            req = urllib.request.Request(url)
            req.add_header("User-Agent", "ArrayExpress-Harvester-Test/1.0")
            with urllib.request.urlopen(req, timeout=15) as response:
                results[name] = f"OK ({response.status})"
        except urllib.error.HTTPError as e:
            # Only 404 for API base URLs (not actual endpoints) might be OK
            results[name] = f"FAIL: HTTP {e.code} {e.reason}"
        except Exception as e:
            results[name] = f"FAIL: {str(e)[:50]}"
    
    all_ok = all(v.startswith("OK") for v in results.values())
    
    return TestResult(
        test_name="Network Connectivity",
        success=all_ok,
        message="All EMBL-EBI endpoints reachable" if all_ok else "Some endpoints failed",
        details=results,
        duration_sec=time.time() - t0
    )


def test_requests_library() -> TestResult:
    """Test if requests library is available and working"""
    t0 = time.time()
    
    if not REQUESTS_AVAILABLE:
        return TestResult(
            test_name="Requests Library",
            success=False,
            message="requests library not installed (pip install requests)",
            duration_sec=0
        )
    
    try:
        response = requests.get(
            f"{BIOSTUDIES_API_BASE}/studies/E-MTAB-5920",
            timeout=15,
            headers={"User-Agent": "ArrayExpress-Harvester-Test/1.0"}
        )
        
        return TestResult(
            test_name="Requests Library",
            success=response.status_code == 200,
            message=f"Working - status {response.status_code}",
            details={
                "status_code": response.status_code,
                "content_type": response.headers.get("content-type", "unknown"),
            },
            duration_sec=time.time() - t0
        )
    except Exception as e:
        return TestResult(
            test_name="Requests Library",
            success=False,
            message=f"Failed: {e}",
            duration_sec=time.time() - t0
        )


def test_bioservices_library() -> TestResult:
    """Test if bioservices library is available and working"""
    t0 = time.time()
    
    if not BIOSERVICES_AVAILABLE:
        return TestResult(
            test_name="Bioservices Library",
            success=False,
            message="bioservices not installed (pip install bioservices)",
            details={"note": "bioservices provides ArrayExpress query interface"},
            duration_sec=0
        )
    
    try:
        ae = BioservicesAE()
        # Try a simple query
        result = ae.queryExperiments(keywords="cancer", species="Homo sapiens")
        
        # Check if we got results - handle both Response objects and parsed data
        if result is not None:
            return TestResult(
                test_name="Bioservices Library",
                success=True,
                message="Working - bioservices ArrayExpress module functional",
                details={
                    "module": "bioservices.ArrayExpress",
                    "caching": getattr(ae, 'CACHING', 'unknown'),
                },
                duration_sec=time.time() - t0
            )
        else:
            return TestResult(
                test_name="Bioservices Library",
                success=False,
                message="No results returned",
                duration_sec=time.time() - t0
            )
    except Exception as e:
        return TestResult(
            test_name="Bioservices Library",
            success=False,
            message=f"Failed: {e}",
            duration_sec=time.time() - t0
        )


def test_direct_api_json() -> TestResult:
    """Test direct JSON API access (bypassing libraries)"""
    t0 = time.time()
    
    url = f"{BIOSTUDIES_API_BASE}/studies/E-MTAB-5920"
    
    try:
        req = urllib.request.Request(url)
        req.add_header("User-Agent", "ArrayExpress-Harvester-Test/1.0")
        req.add_header("Accept", "application/json")
        
        with urllib.request.urlopen(req, timeout=30) as response:
            data = response.read().decode("utf-8")
            result = json.loads(data)
            
            # Extract key fields
            accno = result.get("accno", "")
            title = result.get("attributes", [{}])[0].get("value", "")[:50] if result.get("attributes") else ""
            
            return TestResult(
                test_name="Direct API JSON Access",
                success=True,
                message=f"Retrieved {accno}",
                details={
                    "url": url,
                    "accession": accno,
                    "title_preview": title,
                    "response_size": len(data),
                },
                duration_sec=time.time() - t0
            )
    except urllib.error.HTTPError as e:
        return TestResult(
            test_name="Direct API JSON Access",
            success=False,
            message=f"HTTP Error {e.code}: {e.reason}",
            details={"url": url},
            duration_sec=time.time() - t0
        )
    except Exception as e:
        return TestResult(
            test_name="Direct API JSON Access",
            success=False,
            message=f"Failed: {e}",
            duration_sec=time.time() - t0
        )


def test_api_search_endpoint() -> TestResult:
    """Test the EBI Search API for BioStudies"""
    t0 = time.time()
    
    # Use EBI Search API with biostudies domain
    params = {
        "query": "rat",
        "format": "json",
        "size": "5",
    }
    
    url = f"{EBI_SEARCH_API}/{EBI_SEARCH_BIOSTUDIES_DOMAIN}?" + urllib.parse.urlencode(params)
    
    try:
        req = urllib.request.Request(url)
        req.add_header("User-Agent", "ArrayExpress-Harvester-Test/1.0")
        req.add_header("Accept", "application/json")
        
        with urllib.request.urlopen(req, timeout=30) as response:
            data = response.read().decode("utf-8")
            result = json.loads(data)
            
            # EBI Search response format
            total_hits = result.get("hitCount", 0)
            entries = result.get("entries", [])
            
            return TestResult(
                test_name="API Search Endpoint",
                success=total_hits > 0,
                message=f"Found {total_hits} total results, got {len(entries)} entries",
                details={
                    "api": "EBI Search",
                    "endpoint": f"{EBI_SEARCH_API}/{EBI_SEARCH_BIOSTUDIES_DOMAIN}",
                    "total_hits": total_hits,
                    "returned_entries": len(entries),
                    "sample_accessions": [e.get("id", "") for e in entries[:3]],
                },
                duration_sec=time.time() - t0
            )
    except Exception as e:
        return TestResult(
            test_name="API Search Endpoint",
            success=False,
            message=f"Failed: {e}",
            details={"url": url[:100]},
            duration_sec=time.time() - t0
        )


def run_diagnostic_tests(suite: TestSuite):
    """Run diagnostic tests to verify basic connectivity and libraries"""
    logger.info("\n" + "="*60)
    logger.info("DIAGNOSTIC TESTS - Verifying connectivity and libraries")
    logger.info("="*60)
    
    suite.add(test_network_connectivity())
    time.sleep(0.5)
    
    suite.add(test_requests_library())
    time.sleep(0.5)
    
    suite.add(test_bioservices_library())
    time.sleep(0.5)
    
    suite.add(test_direct_api_json())
    time.sleep(0.5)
    
    suite.add(test_api_search_endpoint())


# =============================================================================
# SECTION 1: BIOSTUDIES SEARCH TESTS
# =============================================================================

@rate_limited
def _fetch_json(url: str, timeout: int = 30) -> Dict:
    """Fetch JSON from URL with rate limiting"""
    req = urllib.request.Request(url)
    req.add_header("User-Agent", "ArrayExpress-Harvester-Test/1.0")
    req.add_header("Accept", "application/json")
    
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def test_search_query(query: str, expected_min: int = 0, description: str = None) -> TestResult:
    """Test a specific search query against EBI Search BioStudies API"""
    t0 = time.time()
    test_name = description or f"Search: {query[:40]}..."
    
    # Use EBI Search API with biostudies domain
    params = {
        "query": query,
        "format": "json",
        "size": "10",
    }
    
    url = f"{EBI_SEARCH_API}/{EBI_SEARCH_BIOSTUDIES_DOMAIN}?" + urllib.parse.urlencode(params)
    
    try:
        req = urllib.request.Request(url)
        req.add_header("User-Agent", "ArrayExpress-Harvester-Test/1.0")
        req.add_header("Accept", "application/json")
        
        with urllib.request.urlopen(req, timeout=30) as response:
            data = response.read().decode("utf-8")
            result = json.loads(data)
            
            # EBI Search response format
            total_hits = result.get("hitCount", 0)
            entries = result.get("entries", [])
            
            success = total_hits >= expected_min
            
            return TestResult(
                test_name=test_name,
                success=success,
                message=f"Found {total_hits} results" + (" (expected ≥{})".format(expected_min) if not success else ""),
                details={
                    "query": query,
                    "total_hits": total_hits,
                    "sample_accessions": [e.get("id", "") for e in entries[:5]],
                },
                duration_sec=time.time() - t0
            )
    except Exception as e:
        return TestResult(
            test_name=test_name,
            success=False,
            message=f"Query failed: {e}",
            details={"query": query, "url": url[:100]},
            duration_sec=time.time() - t0
        )


def test_lucene_query_syntax() -> List[TestResult]:
    """Test various query syntax patterns with EBI Search API"""
    results = []
    
    # Test queries - (query, expected_min_results, description)
    # EBI Search supports Lucene query syntax
    test_queries = [
        # Basic organism queries
        ('Rattus norvegicus', 10, "Organism: Rattus norvegicus (free text)"),
        ('rat', 10, "Organism: rat (simple)"),
        ('mouse', 100, "Organism: mouse (simple)"),
        
        # Technology queries
        ('RNA-seq', 100, "RNA-seq (term)"),
        ('single cell', 10, "Single cell (phrase)"),
        ('scRNA-seq', 1, "scRNA-seq (term)"),
        ('sequencing', 100, "Technology: sequencing"),
        
        # Combined queries with AND
        ('rat AND RNA-seq', 1, "Rat AND RNA-seq"),
        ('rat AND single cell', 1, "Rat AND single cell"),
        ('mouse AND RNA-seq', 10, "Mouse AND RNA-seq"),
        
        # Combined with OR
        ('(rat OR mouse) AND RNA-seq', 10, "Rat OR mouse, AND RNA-seq"),
        
        # Accession-based (free text matches accession fields)
        ('E-MTAB', 100, "E-MTAB accession pattern"),
        ('E-GEOD', 10, "E-GEOD accession pattern"),
        
        # Experiment type
        ('transcription profiling', 10, "Experiment type: transcription profiling"),
        ('gene expression', 100, "Experiment type: gene expression"),
        
        # Tissue/organ queries
        ('liver', 10, "Tissue: liver"),
        ('brain', 10, "Tissue: brain"),
        
        # NOT operator
        ('rat AND NOT microarray', 1, "Rat, NOT microarray"),
        
        # Broader technology terms
        ('10x Genomics', 1, "10x Genomics"),
        ('Smart-seq', 1, "Smart-seq"),
    ]
    
    for query, expected_min, description in test_queries:
        result = test_search_query(query, expected_min, description)
        results.append(result)
        time.sleep(0.4)  # Rate limiting
    
    return results


def test_pagination() -> TestResult:
    """Test pagination functionality with EBI Search API"""
    t0 = time.time()
    
    query = "human RNA-seq"
    page_size = 5
    
    try:
        # EBI Search API uses 'start' for offset-based pagination
        
        # Fetch first page (start=0)
        params_p1 = {"query": query, "format": "json", "size": str(page_size), "start": "0"}
        url_p1 = f"{EBI_SEARCH_API}/{EBI_SEARCH_BIOSTUDIES_DOMAIN}?" + urllib.parse.urlencode(params_p1)
        result_p1 = _fetch_json(url_p1)
        
        time.sleep(0.4)
        
        # Fetch second page (start=5)
        params_p2 = {"query": query, "format": "json", "size": str(page_size), "start": str(page_size)}
        url_p2 = f"{EBI_SEARCH_API}/{EBI_SEARCH_BIOSTUDIES_DOMAIN}?" + urllib.parse.urlencode(params_p2)
        result_p2 = _fetch_json(url_p2)
        
        total_hits = result_p1.get("hitCount", 0)
        entries_p1 = result_p1.get("entries", [])
        entries_p2 = result_p2.get("entries", [])
        
        # Check that pages are different
        accessions_p1 = set(e.get("id", "") for e in entries_p1)
        accessions_p2 = set(e.get("id", "") for e in entries_p2)
        
        pages_different = len(accessions_p1 & accessions_p2) == 0
        
        return TestResult(
            test_name="Pagination",
            success=pages_different and len(entries_p1) > 0,
            message=f"Total: {total_hits}, Page 1: {len(entries_p1)}, Page 2: {len(entries_p2)}",
            details={
                "total_hits": total_hits,
                "page_1_count": len(entries_p1),
                "page_2_count": len(entries_p2),
                "pages_different": pages_different,
                "page_1_accessions": list(accessions_p1)[:3],
                "page_2_accessions": list(accessions_p2)[:3],
            },
            duration_sec=time.time() - t0
        )
    except Exception as e:
        return TestResult(
            test_name="Pagination",
            success=False,
            message=f"Failed: {e}",
            duration_sec=time.time() - t0
        )


def run_search_tests(suite: TestSuite):
    """Run battery of search tests"""
    logger.info("\n" + "="*60)
    logger.info("BIOSTUDIES SEARCH TESTS")
    logger.info("="*60)
    
    # Run Lucene query syntax tests
    for result in test_lucene_query_syntax():
        suite.add(result)
    
    # Test pagination
    suite.add(test_pagination())


# =============================================================================
# SECTION 2: STUDY METADATA TESTS
# =============================================================================

def test_study_metadata(accession: str) -> TestResult:
    """Test fetching study metadata from BioStudies API"""
    t0 = time.time()
    
    url = f"{BIOSTUDIES_API_BASE}/studies/{accession}"
    
    try:
        result = _fetch_json(url)
        
        # Extract key information
        accno = result.get("accno", "")
        attributes = result.get("attributes", [])
        sections = result.get("section", {})
        
        # Parse attributes
        attr_dict = {}
        for attr in attributes:
            name = attr.get("name", "")
            value = attr.get("value", "")
            if name and value:
                attr_dict[name] = value[:100]
        
        # Count subsections
        subsections = sections.get("subsections", []) if isinstance(sections, dict) else []
        
        return TestResult(
            test_name=f"Study Metadata: {accession}",
            success=accno == accession,
            message=f"Retrieved metadata for {accno}",
            details={
                "accession": accno,
                "title": attr_dict.get("Title", "")[:50],
                "release_date": attr_dict.get("ReleaseDate", ""),
                "attribute_count": len(attributes),
                "subsection_count": len(subsections),
            },
            duration_sec=time.time() - t0
        )
    except Exception as e:
        return TestResult(
            test_name=f"Study Metadata: {accession}",
            success=False,
            message=f"Failed: {e}",
            duration_sec=time.time() - t0
        )


def test_study_info(accession: str) -> TestResult:
    """Test fetching study info (includes FTP link)"""
    t0 = time.time()
    
    url = f"{BIOSTUDIES_API_BASE}/studies/{accession}/info"
    
    try:
        result = _fetch_json(url)
        
        ftp_link = result.get("ftpLink", "")
        released = result.get("released", False)
        
        return TestResult(
            test_name=f"Study Info: {accession}",
            success=bool(ftp_link),
            message=f"FTP link: {'Found' if ftp_link else 'Not found'}",
            details={
                "accession": accession,
                "ftp_link": ftp_link,
                "released": released,
            },
            duration_sec=time.time() - t0
        )
    except Exception as e:
        return TestResult(
            test_name=f"Study Info: {accession}",
            success=False,
            message=f"Failed: {e}",
            duration_sec=time.time() - t0
        )


def test_study_files(accession: str) -> TestResult:
    """Test listing files for a study"""
    t0 = time.time()
    
    url = f"{BIOSTUDIES_API_BASE}/studies/{accession}"
    
    try:
        result = _fetch_json(url)
        
        # Navigate to files in the section structure
        files = []
        
        def extract_files(obj, path=""):
            """Recursively extract files from nested structure"""
            if isinstance(obj, dict):
                if "files" in obj:
                    for f in obj["files"]:
                        if isinstance(f, list):
                            for item in f:
                                if isinstance(item, dict):
                                    files.append({
                                        "path": item.get("path", ""),
                                        "size": item.get("size", 0),
                                        "type": item.get("type", ""),
                                    })
                        elif isinstance(f, dict):
                            files.append({
                                "path": f.get("path", ""),
                                "size": f.get("size", 0),
                                "type": f.get("type", ""),
                            })
                for key, value in obj.items():
                    extract_files(value, f"{path}/{key}")
            elif isinstance(obj, list):
                for item in obj:
                    extract_files(item, path)
        
        extract_files(result)
        
        # Categorize files
        idf_files = [f for f in files if f["path"].endswith(".idf.txt")]
        sdrf_files = [f for f in files if f["path"].endswith(".sdrf.txt")]
        data_files = [f for f in files if any(f["path"].endswith(ext) for ext in 
                     [".h5ad", ".loom", ".mtx.gz", ".mtx", ".rds", ".tsv.gz", ".csv.gz"])]
        
        return TestResult(
            test_name=f"Study Files: {accession}",
            success=len(files) > 0,
            message=f"Found {len(files)} files",
            details={
                "total_files": len(files),
                "idf_files": len(idf_files),
                "sdrf_files": len(sdrf_files),
                "data_files": len(data_files),
                "sample_paths": [f["path"] for f in files[:5]],
            },
            duration_sec=time.time() - t0
        )
    except Exception as e:
        return TestResult(
            test_name=f"Study Files: {accession}",
            success=False,
            message=f"Failed: {e}",
            duration_sec=time.time() - t0
        )


def run_metadata_tests(suite: TestSuite, accessions: List[str] = None):
    """Run metadata retrieval tests"""
    logger.info("\n" + "="*60)
    logger.info("STUDY METADATA TESTS")
    logger.info("="*60)
    
    if accessions is None:
        # Default test accessions (verified to exist)
        accessions = [
            "E-MTAB-5920",   # Well-known dataset (dermal fibroblasts)
            "E-MTAB-10868",  # Valid dataset from search results
        ]
    
    for accession in accessions:
        suite.add(test_study_metadata(accession))
        time.sleep(0.4)
        
        suite.add(test_study_info(accession))
        time.sleep(0.4)
        
        suite.add(test_study_files(accession))
        time.sleep(0.4)


# =============================================================================
# SECTION 3: MAGE-TAB (IDF/SDRF) PARSING TESTS
# =============================================================================

def test_idf_download(accession: str) -> TestResult:
    """Test downloading and parsing IDF (Investigation Description Format) file"""
    t0 = time.time()
    
    # Construct the correct FTP-based HTTPS URL
    prefix = accession.rsplit('-', 1)[0] + '-'
    number = accession.rsplit('-', 1)[1]
    last_three = number[-3:].zfill(3)
    
    # Use the HTTPS FTP URL which is known to work
    url = f"https://ftp.ebi.ac.uk/biostudies/fire/{prefix}/{last_three}/{accession}/Files/{accession}.idf.txt"
    
    try:
        req = urllib.request.Request(url)
        req.add_header("User-Agent", "ArrayExpress-Harvester-Test/1.0")
        
        with urllib.request.urlopen(req, timeout=30) as response:
            content = response.read().decode("utf-8", errors="replace")
        
        # Parse IDF format (tab-delimited key-value pairs)
        idf_data = {}
        for line in content.split("\n"):
            if "\t" in line:
                parts = line.split("\t")
                key = parts[0].strip()
                values = [p.strip() for p in parts[1:] if p.strip()]
                if key and values:
                    idf_data[key] = values
        
        return TestResult(
            test_name=f"IDF Download: {accession}",
            success=len(idf_data) > 0,
            message=f"Parsed {len(idf_data)} IDF fields",
            details={
                "title": idf_data.get("Investigation Title", [""])[0][:50],
                "description": idf_data.get("Experiment Description", [""])[0][:50],
                "pubmed_id": idf_data.get("PubMed ID", ["none"])[0],
                "field_count": len(idf_data),
                "sample_fields": list(idf_data.keys())[:10],
            },
            duration_sec=time.time() - t0
        )
    except Exception as e:
        return TestResult(
            test_name=f"IDF Download: {accession}",
            success=False,
            message=f"Failed: {e}",
            details={"url": url},
            duration_sec=time.time() - t0
        )


def test_sdrf_download(accession: str) -> TestResult:
    """Test downloading and parsing SDRF (Sample and Data Relationship Format) file"""
    t0 = time.time()
    
    # Construct the correct FTP-based HTTPS URL
    prefix = accession.rsplit('-', 1)[0] + '-'
    number = accession.rsplit('-', 1)[1]
    last_three = number[-3:].zfill(3)
    
    # Use the HTTPS FTP URL which is known to work
    url = f"https://ftp.ebi.ac.uk/biostudies/fire/{prefix}/{last_three}/{accession}/Files/{accession}.sdrf.txt"
    
    try:
        req = urllib.request.Request(url)
        req.add_header("User-Agent", "ArrayExpress-Harvester-Test/1.0")
        
        with urllib.request.urlopen(req, timeout=30) as response:
            content = response.read().decode("utf-8", errors="replace")
        
        # Parse SDRF (tab-delimited table with headers)
        lines = content.strip().split("\n")
        if len(lines) < 2:
            return TestResult(
                test_name=f"SDRF Download: {accession}",
                success=False,
                message="SDRF file too short",
                duration_sec=time.time() - t0
            )
        
        headers = lines[0].split("\t")
        sample_count = len(lines) - 1
        
        # Look for important columns
        organism_cols = [h for h in headers if "organism" in h.lower()]
        file_cols = [h for h in headers if "file" in h.lower() or "data" in h.lower()]
        ena_cols = [h for h in headers if "ena" in h.lower() or "sra" in h.lower() or "fastq" in h.lower()]
        
        # Parse first data row
        if sample_count > 0:
            first_row = lines[1].split("\t")
            first_sample = dict(zip(headers, first_row))
        else:
            first_sample = {}
        
        return TestResult(
            test_name=f"SDRF Download: {accession}",
            success=sample_count > 0,
            message=f"Found {sample_count} samples, {len(headers)} columns",
            details={
                "sample_count": sample_count,
                "column_count": len(headers),
                "organism_columns": organism_cols,
                "file_columns": file_cols[:5],
                "ena_columns": ena_cols,
                "sample_headers": headers[:10],
            },
            duration_sec=time.time() - t0
        )
    except Exception as e:
        return TestResult(
            test_name=f"SDRF Download: {accession}",
            success=False,
            message=f"Failed: {e}",
            details={"url": url},
            duration_sec=time.time() - t0
        )


def run_magetab_tests(suite: TestSuite, accessions: List[str] = None):
    """Run MAGE-TAB parsing tests"""
    logger.info("\n" + "="*60)
    logger.info("MAGE-TAB (IDF/SDRF) PARSING TESTS")
    logger.info("="*60)
    
    if accessions is None:
        accessions = ["E-MTAB-5920", "E-MTAB-10868"]
    
    for accession in accessions:
        suite.add(test_idf_download(accession))
        time.sleep(0.4)
        
        suite.add(test_sdrf_download(accession))
        time.sleep(0.4)


# =============================================================================
# SECTION 4: FTP ACCESS TESTS
# =============================================================================

def test_ftp_connection() -> TestResult:
    """Test basic FTP connection to EMBL-EBI"""
    t0 = time.time()
    
    try:
        ftp = ftplib.FTP(BIOSTUDIES_FTP, timeout=30)
        ftp.login()  # Anonymous login
        ftp.set_pasv(True)
        welcome = ftp.getwelcome()
        ftp.quit()
        
        return TestResult(
            test_name="FTP Connection (BioStudies)",
            success=True,
            message="Successfully connected to EMBL-EBI FTP",
            details={
                "server": BIOSTUDIES_FTP,
                "welcome": welcome[:50] if welcome else "None",
            },
            duration_sec=time.time() - t0
        )
    except Exception as e:
        return TestResult(
            test_name="FTP Connection (BioStudies)",
            success=False,
            message=f"Failed: {e}",
            duration_sec=time.time() - t0
        )


def construct_ftp_path(accession: str) -> str:
    """
    Construct FTP path from accession number.
    
    Pattern: /biostudies/fire/{prefix}/{last3digits}/{accession}/Files/
    Example: E-MTAB-6798 -> /biostudies/fire/E-MTAB-/798/E-MTAB-6798/Files/
    """
    # Extract prefix and number
    prefix = accession.rsplit('-', 1)[0] + '-'
    number = accession.rsplit('-', 1)[1]
    last_three = number[-3:].zfill(3)
    
    return f"/biostudies/fire/{prefix}/{last_three}/{accession}/Files"


def test_ftp_study_path(accession: str) -> TestResult:
    """Test FTP path construction and access for a study"""
    t0 = time.time()
    
    remote_path = construct_ftp_path(accession)
    
    try:
        ftp = ftplib.FTP(BIOSTUDIES_FTP, timeout=30)
        ftp.login()
        ftp.set_pasv(True)
        
        try:
            ftp.cwd(remote_path)
            
            # List files
            files = []
            ftp.retrlines("NLST", files.append)
            
            ftp.quit()
            
            # Categorize files
            idf_files = [f for f in files if f.endswith(".idf.txt")]
            sdrf_files = [f for f in files if f.endswith(".sdrf.txt")]
            data_files = [f for f in files if any(f.endswith(ext) for ext in 
                         [".h5ad", ".loom", ".mtx.gz", ".mtx", ".rds", ".tar.gz", ".zip"])]
            
            return TestResult(
                test_name=f"FTP Path: {accession}",
                success=len(files) > 0,
                message=f"Found {len(files)} files",
                details={
                    "path": remote_path,
                    "file_count": len(files),
                    "idf_files": idf_files,
                    "sdrf_files": sdrf_files,
                    "data_files": data_files[:5],
                    "all_files": files[:15],
                },
                duration_sec=time.time() - t0
            )
        except ftplib.error_perm as e:
            return TestResult(
                test_name=f"FTP Path: {accession}",
                success=False,
                message=f"Permission error: {e}",
                details={"path": remote_path},
                duration_sec=time.time() - t0
            )
    except Exception as e:
        return TestResult(
            test_name=f"FTP Path: {accession}",
            success=False,
            message=f"Failed: {e}",
            details={"path": remote_path},
            duration_sec=time.time() - t0
        )


def test_https_ftp_access(accession: str) -> TestResult:
    """Test HTTPS access to FTP content (browser-compatible)"""
    t0 = time.time()
    
    # Construct HTTPS URL (same structure, different protocol)
    prefix = accession.rsplit('-', 1)[0] + '-'
    number = accession.rsplit('-', 1)[1]
    last_three = number[-3:].zfill(3)
    
    base_url = f"https://ftp.ebi.ac.uk/biostudies/fire/{prefix}/{last_three}/{accession}/Files"
    
    # Try to access the IDF file via HTTPS
    idf_url = f"{base_url}/{accession}.idf.txt"
    
    try:
        req = urllib.request.Request(idf_url)
        req.add_header("User-Agent", "ArrayExpress-Harvester-Test/1.0")
        
        with urllib.request.urlopen(req, timeout=30) as response:
            content_length = response.headers.get("Content-Length", "unknown")
            content_type = response.headers.get("Content-Type", "unknown")
            
            # Read a bit to verify
            preview = response.read(500).decode("utf-8", errors="replace")
            
            return TestResult(
                test_name=f"HTTPS FTP Access: {accession}",
                success=True,
                message="HTTPS access works",
                details={
                    "url": idf_url,
                    "content_length": content_length,
                    "content_type": content_type,
                    "content_preview": preview[:100],
                },
                duration_sec=time.time() - t0
            )
    except Exception as e:
        return TestResult(
            test_name=f"HTTPS FTP Access: {accession}",
            success=False,
            message=f"Failed: {e}",
            details={"url": idf_url},
            duration_sec=time.time() - t0
        )


def run_ftp_tests(suite: TestSuite, accessions: List[str] = None):
    """Run FTP access tests"""
    logger.info("\n" + "="*60)
    logger.info("FTP ACCESS TESTS")
    logger.info("="*60)
    
    # Basic connection test
    suite.add(test_ftp_connection())
    time.sleep(0.5)
    
    if accessions is None:
        accessions = ["E-MTAB-5920", "E-MTAB-10868"]
    
    for accession in accessions:
        suite.add(test_ftp_study_path(accession))
        time.sleep(0.5)
        
        suite.add(test_https_ftp_access(accession))
        time.sleep(0.5)


# =============================================================================
# SECTION 5: ENA INTEGRATION TESTS
# =============================================================================

def test_ena_portal_api() -> TestResult:
    """Test ENA Portal API basic connectivity"""
    t0 = time.time()
    
    # Simple query for rat RNA-seq data
    params = {
        "result": "read_run",
        "query": 'tax_tree(10116) AND library_strategy="RNA-Seq"',  # 10116 = Rattus norvegicus
        "limit": "5",
        "format": "json",
        "fields": "run_accession,experiment_accession,study_accession,sample_accession,fastq_ftp",
    }
    
    url = f"{ENA_PORTAL_API}/search?" + urllib.parse.urlencode(params)
    
    try:
        req = urllib.request.Request(url)
        req.add_header("User-Agent", "ArrayExpress-Harvester-Test/1.0")
        
        with urllib.request.urlopen(req, timeout=30) as response:
            data = response.read().decode("utf-8")
            results = json.loads(data)
            
            return TestResult(
                test_name="ENA Portal API",
                success=len(results) > 0,
                message=f"Found {len(results)} runs",
                details={
                    "result_count": len(results),
                    "sample_runs": [r.get("run_accession", "") for r in results[:3]],
                    "has_fastq_links": any(r.get("fastq_ftp") for r in results),
                },
                duration_sec=time.time() - t0
            )
    except Exception as e:
        return TestResult(
            test_name="ENA Portal API",
            success=False,
            message=f"Failed: {e}",
            details={"url": url[:100]},
            duration_sec=time.time() - t0
        )


def test_ena_study_lookup(study_accession: str) -> TestResult:
    """Test looking up ENA study by accession"""
    t0 = time.time()
    
    params = {
        "result": "read_run",
        "query": f'study_accession="{study_accession}"',
        "limit": "10",
        "format": "json",
        "fields": "run_accession,experiment_accession,fastq_ftp,fastq_bytes,library_strategy",
    }
    
    url = f"{ENA_PORTAL_API}/search?" + urllib.parse.urlencode(params)
    
    try:
        req = urllib.request.Request(url)
        req.add_header("User-Agent", "ArrayExpress-Harvester-Test/1.0")
        
        with urllib.request.urlopen(req, timeout=30) as response:
            data = response.read().decode("utf-8")
            results = json.loads(data)
            
            # Extract FASTQ URLs
            fastq_urls = []
            for r in results:
                ftp = r.get("fastq_ftp", "")
                if ftp:
                    fastq_urls.extend(ftp.split(";"))
            
            return TestResult(
                test_name=f"ENA Study Lookup: {study_accession}",
                success=len(results) > 0,
                message=f"Found {len(results)} runs, {len(fastq_urls)} FASTQ files",
                details={
                    "run_count": len(results),
                    "fastq_count": len(fastq_urls),
                    "sample_runs": [r.get("run_accession", "") for r in results[:3]],
                    "sample_fastq": fastq_urls[:3],
                },
                duration_sec=time.time() - t0
            )
    except Exception as e:
        return TestResult(
            test_name=f"ENA Study Lookup: {study_accession}",
            success=False,
            message=f"Failed: {e}",
            duration_sec=time.time() - t0
        )


def construct_ena_fastq_url(run_accession: str) -> str:
    """
    Construct ENA FASTQ download URL from run accession.
    
    Pattern varies by accession length:
    - 9 chars (ERR000001): vol1/fastq/ERR000/ERR000001/
    - 10 chars (ERR0000001): vol1/fastq/ERR000/000/ERR0000001/
    - 11+ chars: vol1/fastq/ERR000/00X/ERRXXXXXXX/
    """
    prefix = run_accession[:6]
    
    if len(run_accession) == 9:
        path = f"vol1/fastq/{prefix}/{run_accession}"
    elif len(run_accession) == 10:
        suffix = "00" + run_accession[-1]
        path = f"vol1/fastq/{prefix}/{suffix}/{run_accession}"
    else:
        # 11+ characters
        suffix = "0" + run_accession[-2:]
        path = f"vol1/fastq/{prefix}/{suffix}/{run_accession}"
    
    return f"ftp://{ENA_FTP}/{path}"


def test_ena_fastq_url_construction() -> TestResult:
    """Test ENA FASTQ URL construction patterns"""
    t0 = time.time()
    
    # Test cases: (accession, expected_path_contains)
    # Based on actual ENA patterns observed:
    # - 9 chars: vol1/fastq/{prefix6}/{accession}
    # - 10 chars: vol1/fastq/{prefix6}/00{last_digit}/{accession}
    # - 11+ chars: vol1/fastq/{prefix6}/0{last_2_digits}/{accession}
    test_cases = [
        ("ERR000001", "vol1/fastq/ERR000/ERR000001"),           # 9 chars - no subfolder
        ("ERR1234567", "vol1/fastq/ERR123/007/ERR1234567"),     # 10 chars - 00 + last digit
        ("SRR0000001", "vol1/fastq/SRR000/001/SRR0000001"),     # 10 chars - 00 + last digit
        ("SRR12345678", "vol1/fastq/SRR123/078/SRR12345678"),   # 11 chars - 0 + last 2 digits
    ]
    
    results = []
    for accession, expected in test_cases:
        url = construct_ena_fastq_url(accession)
        matches = expected in url
        results.append({
            "accession": accession,
            "url": url,
            "expected": expected,
            "matches": matches,
        })
    
    all_correct = all(r["matches"] for r in results)
    
    return TestResult(
        test_name="ENA FASTQ URL Construction",
        success=all_correct,
        message=f"{sum(r['matches'] for r in results)}/{len(results)} patterns correct",
        details={
            "test_results": results,
        },
        duration_sec=time.time() - t0
    )


def run_ena_tests(suite: TestSuite, study_accessions: List[str] = None):
    """Run ENA integration tests"""
    logger.info("\n" + "="*60)
    logger.info("ENA INTEGRATION TESTS")
    logger.info("="*60)
    
    # Basic ENA API test
    suite.add(test_ena_portal_api())
    time.sleep(0.5)
    
    # URL construction test
    suite.add(test_ena_fastq_url_construction())
    
    if study_accessions is None:
        # Try some known ENA study accessions (BioProject format works better)
        study_accessions = ["PRJEB26011", "PRJEB4337"]
    
    for acc in study_accessions:
        suite.add(test_ena_study_lookup(acc))
        time.sleep(0.5)


# =============================================================================
# SECTION 6: FILE PATTERN MATCHING TESTS (FIXED)
# =============================================================================

def test_file_patterns() -> TestResult:
    """Test file pattern matching for single-cell vs bulk data"""
    
    # Test files with expected classifications
    test_files = {
        # Single-cell files
        "matrix.mtx.gz": ("SC", True),
        "barcodes.tsv.gz": ("SC", True),
        "features.tsv.gz": ("SC", True),
        "genes.tsv": ("SC", True),
        "sample.h5ad": ("SC", True),
        "data.loom": ("SC", True),
        "filtered_feature_bc_matrix.h5": ("SC", True),
        "cellranger_output.tar.gz": ("SC", True),
        "10x_matrix.tar.gz": ("SC", True),
        
        # Bulk files
        "gene_counts.tsv.gz": ("BULK", True),
        "transcript_counts.csv": ("BULK", True),
        "tpm.tsv": ("BULK", True),
        "fpkm.csv.gz": ("BULK", True),
        "featureCounts_output.txt": ("BULK", True),
        "rsem.genes.results": ("BULK", True),
        "quant.sf": ("BULK", True),
        "abundance.tsv": ("BULK", True),
        
        # RDS (ambiguous - mark as RDS)
        "seurat_object.rds": ("RDS", True),
        
        # Should NOT match expression patterns
        "README.txt": ("NONE", False),
        "sample_info.xlsx": ("NONE", False),
        "E-MTAB-5920.idf.txt": ("NONE", False),
        "E-MTAB-5920.sdrf.txt": ("NONE", False),
        "experiment_design.pdf": ("NONE", False),
    }
    
    results = []
    for filename, (expected_type, should_match) in test_files.items():
        actual_type = classify_file(filename)
        actual_match = actual_type != "NONE"
        
        # Check if classification is correct
        correct = (actual_match == should_match) and (actual_type == expected_type or not should_match)
        results.append((filename, expected_type, actual_type, "✓" if correct else "✗"))
    
    passed = sum(1 for r in results if r[3] == "✓")
    failed = sum(1 for r in results if r[3] == "✗")
    
    # Show failures in details
    failures = [(f, exp, act) for f, exp, act, status in results if status == "✗"]
    
    return TestResult(
        test_name="File Pattern Matching",
        success=failed == 0,
        message=f"{passed} passed, {failed} failed",
        details={
            "tests": results,
            "failures": failures if failures else "None",
            "sc_patterns": len(SC_PATTERNS),
            "bulk_patterns": len(BULK_PATTERNS),
        }
    )


def test_accession_patterns() -> TestResult:
    """Test accession number pattern recognition"""
    
    # ArrayExpress accession patterns
    ACCESSION_PATTERNS = {
        "E-MTAB": re.compile(r'^E-MTAB-\d+$'),      # Native submissions
        "E-GEOD": re.compile(r'^E-GEOD-\d+$'),      # GEO imports
        "E-MEXP": re.compile(r'^E-MEXP-\d+$'),      # Legacy MIAMExpress
        "E-TABM": re.compile(r'^E-TABM-\d+$'),      # Legacy Tab2MAGE
    }
    
    # ENA accession patterns
    ENA_PATTERNS = {
        "study": re.compile(r'^[ESD]RP\d+$'),       # Study (ERP, SRP, DRP)
        "experiment": re.compile(r'^[ESD]RX\d+$'),  # Experiment (ERX, SRX)
        "run": re.compile(r'^[ESD]RR\d+$'),         # Run (ERR, SRR)
        "sample": re.compile(r'^SAM[END][AG]?\d+$'), # BioSample
    }
    
    test_cases = {
        # ArrayExpress
        "E-MTAB-5920": ("E-MTAB", True),
        "E-MTAB-11155": ("E-MTAB", True),
        "E-GEOD-123456": ("E-GEOD", True),
        "E-MEXP-1000": ("E-MEXP", True),
        "E-TABM-500": ("E-TABM", True),
        
        # ENA
        "ERP108495": ("study", True),
        "SRP123456": ("study", True),
        "ERX1234567": ("experiment", True),
        "SRX9876543": ("experiment", True),
        "ERR1234567": ("run", True),
        "SRR9876543": ("run", True),
        "SAMEA123456": ("sample", True),
        "SAMN123456": ("sample", True),
        
        # Invalid
        "GSE123456": ("NONE", False),  # This is GEO format
        "RANDOM123": ("NONE", False),
        "": ("NONE", False),
    }
    
    all_patterns = {**ACCESSION_PATTERNS, **ENA_PATTERNS}
    
    results = []
    for accession, (expected_type, should_match) in test_cases.items():
        matched_type = "NONE"
        for ptype, pattern in all_patterns.items():
            if pattern.match(accession):
                matched_type = ptype
                break
        
        actual_match = matched_type != "NONE"
        correct = (actual_match == should_match) and (matched_type == expected_type or not should_match)
        results.append((accession, expected_type, matched_type, "✓" if correct else "✗"))
    
    passed = sum(1 for r in results if r[3] == "✓")
    failed = sum(1 for r in results if r[3] == "✗")
    
    return TestResult(
        test_name="Accession Pattern Matching",
        success=failed == 0,
        message=f"{passed} passed, {failed} failed",
        details={
            "tests": results,
        }
    )


def run_pattern_tests(suite: TestSuite):
    """Run pattern matching tests"""
    logger.info("\n" + "="*60)
    logger.info("PATTERN MATCHING TESTS")
    logger.info("="*60)
    
    suite.add(test_file_patterns())
    suite.add(test_accession_patterns())


# =============================================================================
# SECTION 7: BIOSERVICES TESTS (FIXED)
# =============================================================================

def _parse_bioservices_response(result):
    """
    Parse response from bioservices, handling different return types.
    
    Bioservices may return:
    - requests.Response object (newer versions)
    - BeautifulSoup/XML parsed object
    - dict/list (JSON parsed)
    - str (raw text)
    
    Returns: (data, data_type, error_message or None)
    """
    if result is None:
        return None, "none", "No data returned"
    
    # Check if it's a requests Response object
    if hasattr(result, 'status_code'):
        if result.status_code != 200:
            return None, "response", f"HTTP {result.status_code}"
        
        # Try to get content
        try:
            # Try JSON first
            data = result.json()
            return data, "json", None
        except:
            pass
        
        try:
            # Fall back to text
            data = result.text
            return data, "text", None
        except:
            pass
        
        try:
            # Fall back to content
            data = result.content
            return data, "bytes", None
        except:
            return None, "response", "Could not extract content from Response"
    
    # Check if it's BeautifulSoup or XML
    if hasattr(result, 'getchildren'):
        return result, "xml", None
    
    # Check if it's already parsed data
    if isinstance(result, (dict, list)):
        return result, type(result).__name__, None
    
    if isinstance(result, str):
        return result, "str", None
    
    # Unknown type - return as-is
    return result, type(result).__name__, None


def test_bioservices_query() -> TestResult:
    """Test bioservices ArrayExpress query functionality"""
    t0 = time.time()
    
    if not BIOSERVICES_AVAILABLE:
        return TestResult(
            test_name="Bioservices Query",
            success=False,
            message="bioservices not installed",
            duration_sec=0
        )
    
    try:
        ae = BioservicesAE()
        
        # Query for experiments - use simple keywords
        result = ae.queryExperiments(
            keywords="cancer",
            species="Homo sapiens",
        )
        
        data, data_type, error = _parse_bioservices_response(result)
        
        if error:
            return TestResult(
                test_name="Bioservices Query",
                success=False,
                message=f"Query failed: {error}",
                details={"data_type": data_type},
                duration_sec=time.time() - t0
            )
        
        # Determine result count based on data type
        if data_type == "xml" and hasattr(data, 'getchildren'):
            count = len(data.getchildren())
            preview = f"XML with {count} children"
        elif isinstance(data, dict):
            count = len(data)
            preview = f"Dict with {count} keys"
        elif isinstance(data, list):
            count = len(data)
            preview = f"List with {count} items"
        elif isinstance(data, str):
            count = len(data)
            preview = data[:100] if data else "Empty string"
        else:
            count = 1
            preview = str(data)[:100]
        
        return TestResult(
            test_name="Bioservices Query",
            success=count > 0,
            message=f"Query returned results (type: {data_type})",
            details={
                "result_type": type(result).__name__,
                "data_type": data_type,
                "count": count,
                "preview": preview,
            },
            duration_sec=time.time() - t0
        )
    except Exception as e:
        return TestResult(
            test_name="Bioservices Query",
            success=False,
            message=f"Failed: {e}",
            duration_sec=time.time() - t0
        )


def test_bioservices_get_experiment(accession: str) -> TestResult:
    """Test bioservices experiment retrieval using getAE method"""
    t0 = time.time()
    
    if not BIOSERVICES_AVAILABLE:
        return TestResult(
            test_name=f"Bioservices Get: {accession}",
            success=False,
            message="bioservices not installed",
            duration_sec=0
        )
    
    try:
        ae = BioservicesAE()
        
        # Use getAE which retrieves experiment data
        # According to docs: s.getAE('E-MEXP-31')
        result = ae.getAE(accession)
        
        data, data_type, error = _parse_bioservices_response(result)
        
        if error:
            return TestResult(
                test_name=f"Bioservices Get: {accession}",
                success=False,
                message=f"getAE failed: {error}",
                details={"data_type": data_type, "raw_type": type(result).__name__},
                duration_sec=time.time() - t0
            )
        
        # Successful retrieval
        if data_type == "xml" and hasattr(data, 'getchildren'):
            preview = f"XML with {len(data.getchildren())} children"
        elif isinstance(data, str):
            preview = data[:200] if data else "Empty"
        else:
            preview = str(data)[:200]
        
        return TestResult(
            test_name=f"Bioservices Get: {accession}",
            success=True,
            message=f"Retrieved experiment data (type: {data_type})",
            details={
                "accession": accession,
                "result_type": type(result).__name__,
                "data_type": data_type,
                "preview": preview,
            },
            duration_sec=time.time() - t0
        )
    except Exception as e:
        error_msg = str(e)
        # Provide helpful context for common errors
        if "not subscriptable" in error_msg:
            error_msg = "API response format changed - use direct HTTP instead"
        
        return TestResult(
            test_name=f"Bioservices Get: {accession}",
            success=False,
            message=f"Failed: {error_msg}",
            details={"exception_type": type(e).__name__},
            duration_sec=time.time() - t0
        )


def test_bioservices_files(accession: str) -> TestResult:
    """Test bioservices file listing using retrieveFilesFromExperiment"""
    t0 = time.time()
    
    if not BIOSERVICES_AVAILABLE:
        return TestResult(
            test_name=f"Bioservices Files: {accession}",
            success=False,
            message="bioservices not installed",
            duration_sec=0
        )
    
    try:
        ae = BioservicesAE()
        
        # According to docs: s.retrieveFilesFromExperiment("E-MEXP-31")
        # Returns: ['E-MEXP-31.raw.1.zip', 'E-MEXP-31.processed.1.zip', 
        #           'E-MEXP-31.idf.txt', 'E-MEXP-31.sdrf.txt']
        result = ae.retrieveFilesFromExperiment(accession)
        
        data, data_type, error = _parse_bioservices_response(result)
        
        if error:
            return TestResult(
                test_name=f"Bioservices Files: {accession}",
                success=False,
                message=f"retrieveFilesFromExperiment failed: {error}",
                details={"data_type": data_type, "raw_type": type(result).__name__},
                duration_sec=time.time() - t0
            )
        
        # Parse file list based on data type
        if isinstance(data, list):
            files = data
            file_count = len(files)
        elif isinstance(data, str):
            # Might be comma or newline separated
            files = [f.strip() for f in data.replace('\n', ',').split(',') if f.strip()]
            file_count = len(files)
        else:
            files = []
            file_count = 0
        
        # Categorize files
        idf_files = [f for f in files if '.idf.' in f.lower()]
        sdrf_files = [f for f in files if '.sdrf.' in f.lower()]
        data_files = [f for f in files if any(ext in f.lower() for ext in 
                     ['.raw.', '.processed.', '.zip', '.tar', '.h5ad', '.mtx'])]
        
        return TestResult(
            test_name=f"Bioservices Files: {accession}",
            success=file_count > 0,
            message=f"Retrieved {file_count} files",
            details={
                "accession": accession,
                "file_count": file_count,
                "files": files[:10],
                "idf_files": idf_files,
                "sdrf_files": sdrf_files,
                "data_files": data_files[:5],
                "data_type": data_type,
            },
            duration_sec=time.time() - t0
        )
    except Exception as e:
        error_msg = str(e)
        if "not subscriptable" in error_msg:
            error_msg = "API response format changed - use direct HTTP instead"
        
        return TestResult(
            test_name=f"Bioservices Files: {accession}",
            success=False,
            message=f"Failed: {error_msg}",
            details={"exception_type": type(e).__name__},
            duration_sec=time.time() - t0
        )


def test_bioservices_retrieve_file(accession: str) -> TestResult:
    """Test bioservices file download using retrieveFile"""
    t0 = time.time()
    
    if not BIOSERVICES_AVAILABLE:
        return TestResult(
            test_name=f"Bioservices File Download: {accession}",
            success=False,
            message="bioservices not installed",
            duration_sec=0
        )
    
    try:
        ae = BioservicesAE()
        
        # According to docs: s.retrieveFile("E-MEXP-31", "E-MEXP-31.idf.txt")
        idf_filename = f"{accession}.idf.txt"
        result = ae.retrieveFile(accession, idf_filename)
        
        data, data_type, error = _parse_bioservices_response(result)
        
        if error:
            return TestResult(
                test_name=f"Bioservices File Download: {accession}",
                success=False,
                message=f"retrieveFile failed: {error}",
                details={"data_type": data_type},
                duration_sec=time.time() - t0
            )
        
        # Check if we got content
        if isinstance(data, str) and len(data) > 0:
            content_preview = data[:200]
            content_size = len(data)
        elif isinstance(data, bytes) and len(data) > 0:
            content_preview = data[:200].decode('utf-8', errors='replace')
            content_size = len(data)
        else:
            content_preview = str(data)[:200] if data else "Empty"
            content_size = len(str(data)) if data else 0
        
        return TestResult(
            test_name=f"Bioservices File Download: {accession}",
            success=content_size > 0,
            message=f"Downloaded {idf_filename} ({content_size} bytes)",
            details={
                "filename": idf_filename,
                "content_size": content_size,
                "content_preview": content_preview,
                "data_type": data_type,
            },
            duration_sec=time.time() - t0
        )
    except Exception as e:
        return TestResult(
            test_name=f"Bioservices File Download: {accession}",
            success=False,
            message=f"Failed: {e}",
            duration_sec=time.time() - t0
        )


def run_bioservices_tests(suite: TestSuite, accessions: List[str] = None):
    """Run bioservices tests"""
    logger.info("\n" + "="*60)
    logger.info("BIOSERVICES LIBRARY TESTS")
    logger.info("="*60)
    
    suite.add(test_bioservices_query())
    time.sleep(0.5)
    
    if accessions is None:
        accessions = ["E-MTAB-5920"]
    
    for accession in accessions:
        suite.add(test_bioservices_get_experiment(accession))
        time.sleep(0.5)
        
        suite.add(test_bioservices_files(accession))
        time.sleep(0.5)
        
        suite.add(test_bioservices_retrieve_file(accession))
        time.sleep(0.5)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Test ArrayExpress/BioStudies Harvester approaches",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python test_arrayexpress_approaches.py --test-all
    python test_arrayexpress_approaches.py --test-diagnose
    python test_arrayexpress_approaches.py --test-search --test-ftp
    python test_arrayexpress_approaches.py --accession E-MTAB-5920 E-MTAB-10868
        """
    )
    
    parser.add_argument("--test-all", action="store_true", help="Run all tests")
    parser.add_argument("--test-diagnose", action="store_true", help="Run diagnostic tests only")
    parser.add_argument("--test-search", action="store_true", help="Test search queries")
    parser.add_argument("--test-metadata", action="store_true", help="Test metadata retrieval")
    parser.add_argument("--test-magetab", action="store_true", help="Test MAGE-TAB (IDF/SDRF) parsing")
    parser.add_argument("--test-ftp", action="store_true", help="Test FTP access")
    parser.add_argument("--test-ena", action="store_true", help="Test ENA integration")
    parser.add_argument("--test-patterns", action="store_true", help="Test file pattern matching")
    parser.add_argument("--test-bioservices", action="store_true", help="Test bioservices library")
    parser.add_argument("--accession", nargs="+", default=None, help="Specific accessions to test")
    parser.add_argument("--output", default=None, help="Output JSON file for results")
    
    args = parser.parse_args()
    
    # Initialize test suite
    suite = TestSuite()
    
    logger.info("="*60)
    logger.info("ARRAYEXPRESS/BIOSTUDIES HARVESTER TEST SUITE")
    logger.info("="*60)
    logger.info(f"requests library: {'Available' if REQUESTS_AVAILABLE else 'NOT INSTALLED'}")
    logger.info(f"bioservices library: {'Available' if BIOSERVICES_AVAILABLE else 'NOT INSTALLED'}")
    logger.info(f"BioStudies API: {BIOSTUDIES_API_BASE}")
    logger.info(f"EBI Search API: {EBI_SEARCH_API}/{EBI_SEARCH_BIOSTUDIES_DOMAIN}")
    logger.info(f"Rate limit: {REQUESTS_PER_SECOND} requests/second")
    
    # Determine what to test
    run_all = args.test_all or not any([
        args.test_diagnose, args.test_search, args.test_metadata,
        args.test_magetab, args.test_ftp, args.test_ena,
        args.test_patterns, args.test_bioservices
    ])
    
    # Always run diagnostics first if all tests or explicitly requested
    if run_all or args.test_diagnose:
        run_diagnostic_tests(suite)
    
    if run_all or args.test_search:
        run_search_tests(suite)
    
    if run_all or args.test_metadata:
        run_metadata_tests(suite, args.accession)
    
    if run_all or args.test_magetab:
        run_magetab_tests(suite, args.accession)
    
    if run_all or args.test_ftp:
        run_ftp_tests(suite, args.accession)
    
    if run_all or args.test_ena:
        run_ena_tests(suite)
    
    if run_all or args.test_patterns:
        run_pattern_tests(suite)
    
    if run_all or args.test_bioservices:
        run_bioservices_tests(suite, args.accession)
    
    # Summary
    passed, failed = suite.summary()
    
    # Output JSON if requested
    if args.output:
        output_data = {
            "summary": {"passed": passed, "failed": failed},
            "environment": {
                "requests_available": REQUESTS_AVAILABLE,
                "bioservices_available": BIOSERVICES_AVAILABLE,
                "biostudies_api": BIOSTUDIES_API_BASE,
                "ebi_search_api": f"{EBI_SEARCH_API}/{EBI_SEARCH_BIOSTUDIES_DOMAIN}",
            },
            "results": [asdict(r) for r in suite.results]
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        logger.info(f"\nResults written to: {args.output}")
    
    # Recommendations
    if failed > 0:
        logger.info("\n" + "="*60)
        logger.info("RECOMMENDATIONS")
        logger.info("="*60)
        
        # Check for specific failure patterns
        network_failures = [r for r in suite.results if "Network" in r.test_name and not r.success]
        api_failures = [r for r in suite.results if "API" in r.test_name and not r.success]
        ftp_failures = [r for r in suite.results if "FTP" in r.test_name and not r.success]
        
        if network_failures:
            logger.info("→ Network connectivity issues detected.")
            logger.info("  Check your internet connection and firewall settings.")
            logger.info("  Verify https://www.ebi.ac.uk is accessible in your browser.")
        
        if api_failures and not network_failures:
            logger.info("→ BioStudies API issues but network is OK.")
            logger.info("  The API might be experiencing issues or rate limiting.")
            logger.info("  Try again later or reduce request frequency.")
        
        if ftp_failures:
            logger.info("→ FTP access issues detected.")
            logger.info("  Some networks block FTP. Try using HTTPS URLs instead.")
            logger.info("  HTTPS alternative: https://ftp.ebi.ac.uk/biostudies/")
        
        if not REQUESTS_AVAILABLE:
            logger.info("→ Install requests library for better HTTP handling:")
            logger.info("  pip install requests")
        
        if not BIOSERVICES_AVAILABLE:
            logger.info("→ Install bioservices for ArrayExpress query interface:")
            logger.info("  pip install bioservices")
        
        # Check for bioservices-specific failures
        bioservices_failures = [r for r in suite.results if "Bioservices" in r.test_name and not r.success]
        if bioservices_failures and BIOSERVICES_AVAILABLE:
            logger.info("→ Bioservices tests failed - this may indicate API compatibility issues.")
            logger.info("  The direct HTTP approach (BioStudies API) is recommended as primary method.")
            logger.info(f"  Search: {EBI_SEARCH_API}/{EBI_SEARCH_BIOSTUDIES_DOMAIN}?query=...")
            logger.info(f"  Metadata: {BIOSTUDIES_API_BASE}/studies/{{accession}}")
        
        # Check for pattern matching failures
        pattern_failures = [r for r in suite.results if "Pattern" in r.test_name and not r.success]
        if pattern_failures:
            logger.info("→ File pattern matching has failures.")
            logger.info("  Review the SC_PATTERNS and BULK_PATTERNS definitions.")
            logger.info("  Check if new file naming conventions need to be added.")
    
    # Exit code
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()