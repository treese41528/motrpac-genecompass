#!/usr/bin/env python3
"""
GEO Harvester Test Suite
========================
Systematically test different approaches for:
1. NCBI E-utilities search queries (what syntax works?)
2. FTP connections and file listing
3. filelist.txt parsing
4. SRA metadata extraction
5. GSM supplementary file discovery

Run this BEFORE modifying the main harvester to verify approaches work.

Usage:
    python test_geo_approaches.py --email your@email.com [--api-key KEY]
    python test_geo_approaches.py --email your@email.com --test-all
    python test_geo_approaches.py --email your@email.com --test-search
    python test_geo_approaches.py --email your@email.com --test-ftp
    python test_geo_approaches.py --email your@email.com --test-diagnose
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
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field, asdict

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# Try imports
try:
    from Bio import Entrez
    import Bio
    BIOPYTHON_AVAILABLE = True
    BIOPYTHON_VERSION = Bio.__version__
except ImportError:
    BIOPYTHON_AVAILABLE = False
    BIOPYTHON_VERSION = None
    logger.warning("Biopython not installed. Install with: pip install biopython")

try:
    import GEOparse
    GEOPARSE_AVAILABLE = True
except ImportError:
    GEOPARSE_AVAILABLE = False
    logger.warning("GEOparse not installed. Install with: pip install GEOparse")


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
                logger.info(f"    {k}: {v}")
    
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
# SECTION 0: DIAGNOSTIC TESTS - Figure out why Entrez is failing
# =============================================================================

def test_direct_ncbi_http(email: str) -> TestResult:
    """Test direct HTTP request to NCBI (bypassing Biopython)"""
    t0 = time.time()
    
    # Try direct esearch via HTTP
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {
        "db": "gds",
        "term": "rat",
        "retmax": "5",
        "retmode": "json",
        "email": email,
    }
    
    url = f"{base_url}?{urllib.parse.urlencode(params)}"
    
    try:
        req = urllib.request.Request(url)
        req.add_header("User-Agent", "Python-GEO-Harvester/1.0")
        
        with urllib.request.urlopen(req, timeout=30) as response:
            data = response.read().decode("utf-8")
            result = json.loads(data)
            
            count = result.get("esearchresult", {}).get("count", "0")
            ids = result.get("esearchresult", {}).get("idlist", [])
            
            return TestResult(
                test_name="Direct HTTP to NCBI",
                success=True,
                message=f"Works! Found {count} results",
                details={
                    "url": url[:100] + "...",
                    "count": count,
                    "sample_ids": ids[:3],
                    "response_length": len(data)
                },
                duration_sec=time.time() - t0
            )
    except urllib.error.HTTPError as e:
        return TestResult(
            test_name="Direct HTTP to NCBI",
            success=False,
            message=f"HTTP Error {e.code}: {e.reason}",
            details={
                "url": url[:100] + "...",
                "error_code": e.code,
                "error_body": e.read().decode("utf-8")[:500] if hasattr(e, 'read') else ""
            },
            duration_sec=time.time() - t0
        )
    except Exception as e:
        return TestResult(
            test_name="Direct HTTP to NCBI",
            success=False,
            message=f"Failed: {e}",
            duration_sec=time.time() - t0
        )


def test_direct_ncbi_https_variants(email: str) -> List[TestResult]:
    """Test different NCBI URL variants"""
    results = []
    
    # Different base URLs to try
    urls_to_try = [
        ("eutils (HTTPS)", "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"),
        ("eutils (HTTP)", "http://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"),
    ]
    
    for name, base_url in urls_to_try:
        t0 = time.time()
        params = {
            "db": "pubmed",  # Try pubmed first (simpler)
            "term": "cancer",
            "retmax": "3",
            "retmode": "json",
            "email": email,
        }
        
        url = f"{base_url}?{urllib.parse.urlencode(params)}"
        
        try:
            req = urllib.request.Request(url)
            req.add_header("User-Agent", "Python-GEO-Harvester/1.0")
            
            with urllib.request.urlopen(req, timeout=30) as response:
                data = response.read().decode("utf-8")
                result = json.loads(data)
                count = result.get("esearchresult", {}).get("count", "0")
                
                results.append(TestResult(
                    test_name=f"Direct HTTP: {name}",
                    success=True,
                    message=f"Works! PubMed search returned {count} results",
                    details={"base_url": base_url},
                    duration_sec=time.time() - t0
                ))
        except Exception as e:
            results.append(TestResult(
                test_name=f"Direct HTTP: {name}",
                success=False,
                message=f"Failed: {e}",
                details={"base_url": base_url},
                duration_sec=time.time() - t0
            ))
        
        time.sleep(0.5)
    
    return results


def test_biopython_version() -> TestResult:
    """Check Biopython version"""
    if not BIOPYTHON_AVAILABLE:
        return TestResult(
            test_name="Biopython Version",
            success=False,
            message="Biopython not installed"
        )
    
    return TestResult(
        test_name="Biopython Version",
        success=True,
        message=f"Version {BIOPYTHON_VERSION}",
        details={
            "version": BIOPYTHON_VERSION,
            "recommended": "1.79 or higher"
        }
    )


def test_entrez_different_databases(email: str) -> List[TestResult]:
    """Test Entrez with different databases to isolate the problem"""
    results = []
    
    Entrez.email = email
    
    # Try different databases
    databases = [
        ("pubmed", "cancer", 1),
        ("nucleotide", "human", 1),
        ("gds", "rat", 1),
        ("sra", "rat", 1),
    ]
    
    for db, term, expected_min in databases:
        t0 = time.time()
        try:
            handle = Entrez.esearch(db=db, term=term, retmax=3)
            result = Entrez.read(handle)
            handle.close()
            
            count = int(result.get("Count", 0))
            
            results.append(TestResult(
                test_name=f"Entrez db={db}",
                success=count >= expected_min,
                message=f"Found {count} results",
                details={"database": db, "term": term, "count": count},
                duration_sec=time.time() - t0
            ))
        except Exception as e:
            results.append(TestResult(
                test_name=f"Entrez db={db}",
                success=False,
                message=f"Failed: {e}",
                details={"database": db, "term": term, "error": str(e)},
                duration_sec=time.time() - t0
            ))
        
        time.sleep(0.5)
    
    return results


def test_entrez_configuration() -> TestResult:
    """Check Entrez configuration and internal state"""
    details = {}
    
    try:
        details["email"] = Entrez.email
        details["tool"] = getattr(Entrez, "tool", "NOT SET")
        
        # Check api_key carefully - empty string is the bug!
        api_key = getattr(Entrez, "api_key", None)
        if api_key is None:
            details["api_key"] = "None (correctly unset)"
        elif api_key == "":
            details["api_key"] = "EMPTY STRING (BUG - will cause 400 errors!)"
        else:
            # Mask the key for security
            details["api_key"] = f"SET ({api_key[:4]}...{api_key[-4:]})" if len(str(api_key)) > 8 else "SET"
        
        # Check internal URL building
        # Biopython builds URLs internally - let's see what it looks like
        if hasattr(Entrez, "_construct_params"):
            details["has_construct_params"] = True
        
        # Check what base URL Biopython uses
        if hasattr(Entrez, "cgi"):
            details["base_cgi"] = Entrez.cgi
        
        # Check for any custom opener
        if hasattr(Entrez, "_urlopen"):
            details["custom_urlopen"] = True
            
        return TestResult(
            test_name="Entrez Configuration",
            success=True,
            message="Configuration checked",
            details=details
        )
    except Exception as e:
        return TestResult(
            test_name="Entrez Configuration", 
            success=False,
            message=f"Error: {e}",
            details=details
        )


def test_entrez_with_tool_set(email: str) -> TestResult:
    """Test if setting Entrez.tool helps"""
    t0 = time.time()
    
    try:
        Entrez.email = email
        Entrez.tool = "geo_harvester"  # Explicitly set tool name
        
        handle = Entrez.esearch(db="pubmed", term="cancer", retmax=3)
        result = Entrez.read(handle)
        handle.close()
        
        count = int(result.get("Count", 0))
        
        return TestResult(
            test_name="Entrez with tool set",
            success=count > 0,
            message=f"Found {count} results",
            details={"tool": Entrez.tool, "email": Entrez.email},
            duration_sec=time.time() - t0
        )
    except Exception as e:
        return TestResult(
            test_name="Entrez with tool set",
            success=False,
            message=f"Failed: {e}",
            duration_sec=time.time() - t0
        )


def test_entrez_url_construction(email: str) -> TestResult:
    """Inspect what URL Biopython constructs"""
    t0 = time.time()
    
    try:
        Entrez.email = email
        Entrez.tool = "geo_harvester"
        
        # Biopython constructs URLs in _open function
        # Let's manually construct what it should be and compare
        import urllib.parse
        
        # What we expect
        expected_params = {
            "db": "gds",
            "term": "rat",
            "retmax": "5",
            "email": email,
            "tool": "geo_harvester",
        }
        expected_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?" + \
                       urllib.parse.urlencode(expected_params)
        
        # Check Biopython's base URL
        base = getattr(Entrez, "cgi", "UNKNOWN")
        
        return TestResult(
            test_name="Entrez URL Construction",
            success=True,
            message=f"Base CGI: {base}",
            details={
                "biopython_base": base,
                "expected_format": expected_url[:80] + "..."
            },
            duration_sec=time.time() - t0
        )
    except Exception as e:
        return TestResult(
            test_name="Entrez URL Construction",
            success=False,
            message=f"Error: {e}",
            duration_sec=time.time() - t0
        )


def test_entrez_raw_request(email: str) -> TestResult:
    """Try to see exactly what request Biopython sends"""
    t0 = time.time()
    
    try:
        # Monkey-patch urllib to capture the request
        import urllib.request
        original_urlopen = urllib.request.urlopen
        captured_request = {}
        
        def capturing_urlopen(request, *args, **kwargs):
            if hasattr(request, 'full_url'):
                captured_request['url'] = request.full_url
                captured_request['headers'] = dict(request.headers)
                captured_request['method'] = request.get_method()
            elif isinstance(request, str):
                captured_request['url'] = request
            raise Exception("CAPTURED - not actually sending")
        
        urllib.request.urlopen = capturing_urlopen
        
        try:
            Entrez.email = email
            Entrez.tool = "geo_harvester"
            handle = Entrez.esearch(db="gds", term="rat", retmax=3)
        except Exception as e:
            if "CAPTURED" in str(e):
                pass  # Expected
            else:
                captured_request['error'] = str(e)
        finally:
            urllib.request.urlopen = original_urlopen
        
        return TestResult(
            test_name="Entrez Raw Request Capture",
            success='url' in captured_request,
            message="Captured request details" if captured_request else "Could not capture",
            details=captured_request,
            duration_sec=time.time() - t0
        )
    except Exception as e:
        return TestResult(
            test_name="Entrez Raw Request Capture",
            success=False,
            message=f"Error: {e}",
            duration_sec=time.time() - t0
        )


def test_compare_http_methods(email: str) -> TestResult:
    """
    Compare exactly what headers/params differ between working HTTP and Biopython.
    This is the key diagnostic test.
    """
    t0 = time.time()
    details = {}
    
    # What our working direct HTTP sends
    import urllib.request
    import urllib.parse
    
    working_params = {
        "db": "gds",
        "term": "rat",
        "retmax": "5",
        "retmode": "json",
        "email": email,
    }
    working_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?" + \
                  urllib.parse.urlencode(working_params)
    working_headers = {"User-Agent": "Python-GEO-Harvester/1.0"}
    
    details["working_url"] = working_url
    details["working_headers"] = working_headers
    
    # Try to capture what Biopython sends using a custom opener
    try:
        # Check Biopython's internal _open or _urlopen method
        import Bio.Entrez as EntrezModule
        
        # Look at the module to understand how it builds requests
        if hasattr(EntrezModule, '_urlopen'):
            details["biopython_uses_custom_urlopen"] = True
        
        # Check what URL function it uses
        for attr in dir(EntrezModule):
            if 'url' in attr.lower() or 'cgi' in attr.lower() or 'http' in attr.lower():
                val = getattr(EntrezModule, attr, None)
                if isinstance(val, str):
                    details[f"biopython_{attr}"] = val
        
        # The key insight: Biopython constructs requests in Bio.Entrez._open()
        # Let's look at what parameters it builds
        
        # Simulate what Biopython does internally
        Entrez.email = email
        Entrez.tool = "geo_harvester"
        
        # These are the params Biopython would add
        biopython_expected_params = {
            "db": "gds",
            "term": "rat", 
            "retmax": "3",
            "email": email,
            "tool": "geo_harvester",
        }
        
        details["biopython_expected_params"] = biopython_expected_params
        
    except Exception as e:
        details["biopython_inspection_error"] = str(e)
    
    # Now the key test: what User-Agent does Python's default urllib send?
    try:
        # Create a request without custom User-Agent (like Biopython does)
        test_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?" + \
                   urllib.parse.urlencode({"db": "gds", "term": "rat", "retmax": "1", 
                                          "retmode": "json", "email": email, "tool": "biopython"})
        
        # Method 1: No User-Agent (might be the problem!)
        req_no_ua = urllib.request.Request(test_url)
        details["default_user_agent"] = req_no_ua.get_header("User-agent") or "NONE SET"
        
        # Try without custom User-Agent
        try:
            with urllib.request.urlopen(req_no_ua, timeout=10) as response:
                details["no_useragent_result"] = f"OK ({response.status})"
        except urllib.error.HTTPError as e:
            details["no_useragent_result"] = f"FAIL ({e.code}: {e.reason})"
        
        time.sleep(0.5)
        
        # Method 2: With custom User-Agent
        req_with_ua = urllib.request.Request(test_url)
        req_with_ua.add_header("User-Agent", "Python-GEO-Harvester/1.0")
        
        try:
            with urllib.request.urlopen(req_with_ua, timeout=10) as response:
                details["with_useragent_result"] = f"OK ({response.status})"
        except urllib.error.HTTPError as e:
            details["with_useragent_result"] = f"FAIL ({e.code}: {e.reason})"
            
    except Exception as e:
        details["comparison_error"] = str(e)
    
    # Determine if User-Agent is the problem
    no_ua_works = "OK" in details.get("no_useragent_result", "")
    with_ua_works = "OK" in details.get("with_useragent_result", "")
    
    if with_ua_works and not no_ua_works:
        details["DIAGNOSIS"] = "User-Agent header is required! Biopython doesn't set one properly."
    elif with_ua_works and no_ua_works:
        details["DIAGNOSIS"] = "Both work - problem might be elsewhere in Biopython"
    else:
        details["DIAGNOSIS"] = "Unknown issue"
    
    return TestResult(
        test_name="Compare HTTP Methods",
        success=True,
        message=details.get("DIAGNOSIS", "See details"),
        details=details,
        duration_sec=time.time() - t0
    )


def test_biopython_source_inspection() -> TestResult:
    """Inspect Biopython source to understand how it makes requests"""
    t0 = time.time()
    details = {}
    
    try:
        import Bio.Entrez as EntrezModule
        import inspect
        
        # Get the file location
        details["entrez_file"] = inspect.getfile(EntrezModule)
        
        # Look for _open function which handles HTTP
        if hasattr(EntrezModule, '_open'):
            source = inspect.getsource(EntrezModule._open)
            # Look for User-Agent in the source
            if "User-Agent" in source or "user-agent" in source.lower():
                details["sets_user_agent"] = True
            else:
                details["sets_user_agent"] = False
                details["PROBLEM"] = "Biopython _open() does NOT set User-Agent header!"
            
            # Show relevant lines
            lines = source.split('\n')
            relevant_lines = [l for l in lines if 'request' in l.lower() or 'header' in l.lower() or 'urlopen' in l.lower()]
            details["relevant_source_lines"] = relevant_lines[:10]
        
        return TestResult(
            test_name="Biopython Source Inspection",
            success=True,
            message="Source inspected",
            details=details,
            duration_sec=time.time() - t0
        )
    except Exception as e:
        return TestResult(
            test_name="Biopython Source Inspection",
            success=False,
            message=f"Error: {e}",
            duration_sec=time.time() - t0
        )


def test_biopython_build_request(email: str) -> TestResult:
    """
    Inspect what Biopython's _build_request actually produces.
    This is the KEY test to see exactly what URL/params Biopython constructs.
    """
    t0 = time.time()
    details = {}
    
    try:
        import Bio.Entrez as EntrezModule
        
        Entrez.email = email
        Entrez.tool = "geo_harvester"
        
        # Check if _build_request exists and what it produces
        if hasattr(EntrezModule, '_build_request'):
            # Call _build_request directly
            params = {
                "db": "gds",
                "term": "rat",
                "retmax": "3",
            }
            
            try:
                request = EntrezModule._build_request("esearch.fcgi", params)
                
                # Inspect the request object
                if hasattr(request, 'full_url'):
                    details["request_url"] = request.full_url
                if hasattr(request, 'data'):
                    details["request_data"] = request.data
                if hasattr(request, 'headers'):
                    details["request_headers"] = dict(request.headers)
                if hasattr(request, 'get_method'):
                    details["request_method"] = request.get_method()
                    
                # Check if it's a POST request (might be the issue!)
                if request.data is not None:
                    details["IS_POST_REQUEST"] = True
                    details["POST_DATA"] = request.data.decode() if isinstance(request.data, bytes) else str(request.data)
                else:
                    details["IS_POST_REQUEST"] = False
                    
            except Exception as e:
                details["build_request_error"] = str(e)
        else:
            details["has_build_request"] = False
            
        # Also check _construct_params
        if hasattr(EntrezModule, '_construct_params'):
            try:
                constructed = EntrezModule._construct_params(params)
                details["constructed_params"] = constructed
            except Exception as e:
                details["construct_params_error"] = str(e)
        
        return TestResult(
            test_name="Biopython _build_request Inspection",
            success=True,
            message="Request inspected - check if POST vs GET",
            details=details,
            duration_sec=time.time() - t0
        )
    except Exception as e:
        return TestResult(
            test_name="Biopython _build_request Inspection",
            success=False,
            message=f"Error: {e}",
            duration_sec=time.time() - t0
        )


def test_post_vs_get_to_ncbi(email: str) -> TestResult:
    """
    Test if NCBI accepts POST vs GET requests.
    Biopython might be using POST while NCBI expects GET for some endpoints.
    """
    t0 = time.time()
    details = {}
    
    import urllib.request
    import urllib.parse
    
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {
        "db": "gds",
        "term": "rat",
        "retmax": "3",
        "retmode": "json",
        "email": email,
        "tool": "test",
    }
    
    # Test GET request
    try:
        get_url = base_url + "?" + urllib.parse.urlencode(params)
        req = urllib.request.Request(get_url, method="GET")
        with urllib.request.urlopen(req, timeout=10) as response:
            details["GET_result"] = f"OK ({response.status})"
    except urllib.error.HTTPError as e:
        details["GET_result"] = f"FAIL ({e.code}: {e.reason})"
    except Exception as e:
        details["GET_result"] = f"ERROR: {e}"
    
    time.sleep(0.5)
    
    # Test POST request
    try:
        data = urllib.parse.urlencode(params).encode('utf-8')
        req = urllib.request.Request(base_url, data=data, method="POST")
        with urllib.request.urlopen(req, timeout=10) as response:
            details["POST_result"] = f"OK ({response.status})"
    except urllib.error.HTTPError as e:
        details["POST_result"] = f"FAIL ({e.code}: {e.reason})"
        # Read error body for more info
        try:
            error_body = e.read().decode('utf-8')[:200]
            details["POST_error_body"] = error_body
        except:
            pass
    except Exception as e:
        details["POST_result"] = f"ERROR: {e}"
    
    # Diagnosis
    get_ok = "OK" in details.get("GET_result", "")
    post_ok = "OK" in details.get("POST_result", "")
    
    if get_ok and not post_ok:
        details["DIAGNOSIS"] = "GET works but POST fails! Biopython might be using POST."
    elif get_ok and post_ok:
        details["DIAGNOSIS"] = "Both GET and POST work"
    else:
        details["DIAGNOSIS"] = "Unexpected results"
    
    return TestResult(
        test_name="POST vs GET to NCBI",
        success=True,
        message=details.get("DIAGNOSIS", "See details"),
        details=details,
        duration_sec=time.time() - t0
    )


def test_biopython_internal_functions(email: str) -> TestResult:
    """
    Step through Biopython's internal functions to find exactly where it fails.
    """
    t0 = time.time()
    details = {}
    
    try:
        import Bio.Entrez as EntrezModule
        
        Entrez.email = email
        Entrez.tool = "geo_harvester"
        
        # Step 1: Check what functions exist
        internal_funcs = [attr for attr in dir(EntrezModule) if attr.startswith('_') and callable(getattr(EntrezModule, attr, None))]
        details["internal_functions"] = internal_funcs[:20]
        
        # Step 2: Try _construct_params
        if hasattr(EntrezModule, '_construct_params'):
            test_params = {"db": "gds", "term": "rat"}
            try:
                result = EntrezModule._construct_params(test_params)
                details["_construct_params_result"] = result[:200] if isinstance(result, str) else str(result)[:200]
            except Exception as e:
                details["_construct_params_error"] = str(e)
        
        # Step 3: Check the base URL Biopython uses
        for attr in ['_base_url', '_api_url', '_cgi_base', 'cgi']:
            if hasattr(EntrezModule, attr):
                details[f"biopython_{attr}"] = getattr(EntrezModule, attr)
        
        # Step 4: Look at esearch function source
        if hasattr(EntrezModule, 'esearch'):
            import inspect
            sig = inspect.signature(EntrezModule.esearch)
            details["esearch_params"] = str(sig)
        
        return TestResult(
            test_name="Biopython Internal Functions",
            success=True,
            message="Internal functions inspected",
            details=details,
            duration_sec=time.time() - t0
        )
    except Exception as e:
        return TestResult(
            test_name="Biopython Internal Functions",
            success=False,
            message=f"Error: {e}",
            duration_sec=time.time() - t0
        )


def test_empty_api_key_bug(email: str) -> TestResult:
    """
    Test if an empty api_key parameter causes the 400 error.
    This is likely the root cause!
    """
    t0 = time.time()
    details = {}
    
    import urllib.request
    import urllib.parse
    
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    
    # Test 1: With api_key="" (empty string) - what Biopython sends
    params_with_empty_key = {
        "db": "gds",
        "term": "rat",
        "retmax": "3",
        "retmode": "json",
        "email": email,
        "tool": "test",
        "api_key": "",  # EMPTY STRING - likely the problem!
    }
    
    try:
        url = base_url + "?" + urllib.parse.urlencode(params_with_empty_key)
        details["url_with_empty_apikey"] = url
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=10) as response:
            details["empty_apikey_result"] = f"OK ({response.status})"
    except urllib.error.HTTPError as e:
        details["empty_apikey_result"] = f"FAIL ({e.code}: {e.reason})"
    except Exception as e:
        details["empty_apikey_result"] = f"ERROR: {e}"
    
    time.sleep(0.5)
    
    # Test 2: Without api_key at all - what should be sent
    params_without_key = {
        "db": "gds",
        "term": "rat",
        "retmax": "3",
        "retmode": "json",
        "email": email,
        "tool": "test",
        # NO api_key parameter
    }
    
    try:
        url = base_url + "?" + urllib.parse.urlencode(params_without_key)
        details["url_without_apikey"] = url
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=10) as response:
            details["no_apikey_result"] = f"OK ({response.status})"
    except urllib.error.HTTPError as e:
        details["no_apikey_result"] = f"FAIL ({e.code}: {e.reason})"
    except Exception as e:
        details["no_apikey_result"] = f"ERROR: {e}"
    
    # Diagnosis
    empty_fails = "FAIL" in details.get("empty_apikey_result", "")
    no_key_works = "OK" in details.get("no_apikey_result", "")
    
    if empty_fails and no_key_works:
        details["DIAGNOSIS"] = "CONFIRMED: Empty api_key='' causes 400 error! Biopython bug found."
        details["FIX"] = "Set Entrez.api_key = None instead of empty string, or patch _construct_params"
    elif not empty_fails and no_key_works:
        details["DIAGNOSIS"] = "Both work - empty api_key is not the issue"
    else:
        details["DIAGNOSIS"] = "Unexpected results"
    
    return TestResult(
        test_name="Empty api_key Bug Test",
        success=True,
        message=details.get("DIAGNOSIS", "See details"),
        details=details,
        duration_sec=time.time() - t0
    )


def test_biopython_fix_api_key(email: str) -> TestResult:
    """
    Test if setting api_key to None fixes the issue.
    """
    t0 = time.time()
    details = {}
    
    try:
        # Save original value
        original_api_key = Entrez.api_key
        details["original_api_key"] = repr(original_api_key)
        
        # Try setting to None explicitly
        Entrez.email = email
        Entrez.tool = "geo_harvester"
        Entrez.api_key = None  # Explicitly None, not empty string
        
        details["set_api_key_to"] = repr(Entrez.api_key)
        
        # Now try a search
        try:
            handle = Entrez.esearch(db="gds", term="rat", retmax=3)
            result = Entrez.read(handle)
            handle.close()
            
            count = int(result.get("Count", 0))
            details["search_result"] = f"SUCCESS! Found {count} results"
            details["FIX_WORKS"] = True
            
        except Exception as e:
            details["search_error"] = str(e)
            details["FIX_WORKS"] = False
        
        # Restore original
        Entrez.api_key = original_api_key
        
        return TestResult(
            test_name="Fix: Set api_key=None",
            success=details.get("FIX_WORKS", False),
            message=details.get("search_result", details.get("search_error", "Unknown")),
            details=details,
            duration_sec=time.time() - t0
        )
    except Exception as e:
        return TestResult(
            test_name="Fix: Set api_key=None",
            success=False,
            message=f"Error: {e}",
            duration_sec=time.time() - t0
        )


def test_entrez_with_retmode(email: str) -> TestResult:
    """Test if specifying retmode helps"""
    t0 = time.time()
    
    try:
        Entrez.email = email
        Entrez.tool = "geo_harvester"
        
        # Try with explicit retmode
        handle = Entrez.esearch(db="gds", term="rat", retmax=3, retmode="xml")
        result = Entrez.read(handle)
        handle.close()
        
        count = int(result.get("Count", 0))
        
        return TestResult(
            test_name="Entrez with retmode=xml",
            success=count > 0,
            message=f"Found {count} results",
            duration_sec=time.time() - t0
        )
    except Exception as e:
        return TestResult(
            test_name="Entrez with retmode=xml",
            success=False,
            message=f"Failed: {e}",
            duration_sec=time.time() - t0
        )


def test_requests_library_to_ncbi(email: str) -> TestResult:
    """Test using requests library instead of urllib"""
    t0 = time.time()
    
    try:
        import requests
        REQUESTS_AVAILABLE = True
    except ImportError:
        return TestResult(
            test_name="Requests library to NCBI",
            success=False,
            message="requests library not installed (pip install requests)",
            duration_sec=0
        )
    
    try:
        url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        params = {
            "db": "gds",
            "term": "rat",
            "retmax": 5,
            "retmode": "json",
            "email": email,
            "tool": "geo_harvester",
        }
        
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        count = data.get("esearchresult", {}).get("count", "0")
        
        return TestResult(
            test_name="Requests library to NCBI",
            success=True,
            message=f"Works! Found {count} results",
            details={
                "status_code": response.status_code,
                "count": count,
                "url": response.url[:100] + "..."
            },
            duration_sec=time.time() - t0
        )
    except Exception as e:
        return TestResult(
            test_name="Requests library to NCBI",
            success=False,
            message=f"Failed: {e}",
            duration_sec=time.time() - t0
        )


def test_network_connectivity() -> TestResult:
    """Basic network connectivity test"""
    t0 = time.time()
    
    urls = [
        "https://www.ncbi.nlm.nih.gov",
        "https://eutils.ncbi.nlm.nih.gov",
        "https://ftp.ncbi.nlm.nih.gov",
    ]
    
    results = {}
    for url in urls:
        try:
            req = urllib.request.Request(url)
            req.add_header("User-Agent", "Python-Test/1.0")
            with urllib.request.urlopen(req, timeout=10) as response:
                results[url] = f"OK ({response.status})"
        except Exception as e:
            results[url] = f"FAIL ({e})"
    
    all_ok = all("OK" in v for v in results.values())
    
    return TestResult(
        test_name="Network Connectivity",
        success=all_ok,
        message="All NCBI endpoints reachable" if all_ok else "Some endpoints failed",
        details=results,
        duration_sec=time.time() - t0
    )


def run_diagnostic_tests(suite: TestSuite, email: str):
    """Run diagnostic tests to figure out why Entrez is failing"""
    logger.info("\n" + "="*60)
    logger.info("DIAGNOSTIC TESTS - Finding the root cause")
    logger.info("="*60)
    
    # 1. Check Biopython version
    suite.add(test_biopython_version())
    
    # 2. Test basic network connectivity
    suite.add(test_network_connectivity())
    time.sleep(0.5)
    
    # 3. Test direct HTTP (bypass Biopython)
    suite.add(test_direct_ncbi_http(email))
    time.sleep(0.5)
    
    # 4. Test HTTP/HTTPS variants
    for result in test_direct_ncbi_https_variants(email):
        suite.add(result)
    
    # 5. Check Entrez configuration
    logger.info("\n--- Biopython Entrez Diagnostics ---")
    suite.add(test_entrez_configuration())
    
    # 6. Test Entrez URL construction
    suite.add(test_entrez_url_construction(email))
    
    # 7. Capture raw request
    suite.add(test_entrez_raw_request(email))
    time.sleep(0.5)
    
    # 8. KEY TEST: Compare what works vs what doesn't
    logger.info("\n--- Key Comparison Test ---")
    suite.add(test_compare_http_methods(email))
    time.sleep(0.5)
    
    # 9. Inspect Biopython source code
    suite.add(test_biopython_source_inspection())
    
    # 10. Inspect _build_request output
    logger.info("\n--- Deep Biopython Inspection ---")
    suite.add(test_biopython_build_request(email))
    
    # 11. Test POST vs GET
    suite.add(test_post_vs_get_to_ncbi(email))
    time.sleep(0.5)
    
    # 12. Internal functions inspection
    suite.add(test_biopython_internal_functions(email))
    
    # 13. KEY BUG TEST: Empty api_key parameter
    logger.info("\n--- Testing Empty api_key Bug ---")
    suite.add(test_empty_api_key_bug(email))
    time.sleep(0.5)
    
    # 14. TEST THE FIX: Set api_key to None
    suite.add(test_biopython_fix_api_key(email))
    time.sleep(0.5)
    
    # 15. Test with tool set (will likely still fail without the fix)
    suite.add(test_entrez_with_tool_set(email))
    time.sleep(0.5)
    
    # 16. Test with explicit retmode
    suite.add(test_entrez_with_retmode(email))
    time.sleep(0.5)
    
    # 17. Test requests library as alternative
    suite.add(test_requests_library_to_ncbi(email))
    time.sleep(0.5)
    
    # 18. Test different Entrez databases
    for result in test_entrez_different_databases(email):
        suite.add(result)


# =============================================================================
# SECTION 1: NCBI E-UTILITIES SEARCH TESTS
# =============================================================================

def test_entrez_connection(email: str, api_key: Optional[str] = None) -> TestResult:
    """Test basic Entrez connection"""
    t0 = time.time()
    try:
        Entrez.email = email
        Entrez.api_key = api_key or os.environ.get("NCBI_API_KEY", "")
        
        # Simple test query
        handle = Entrez.einfo(db="gds")
        result = Entrez.read(handle)
        handle.close()
        
        return TestResult(
            test_name="Entrez Connection",
            success=True,
            message="Successfully connected to NCBI E-utilities",
            details={"db_info": "gds database accessible"},
            duration_sec=time.time() - t0
        )
    except Exception as e:
        return TestResult(
            test_name="Entrez Connection",
            success=False,
            message=f"Failed: {e}",
            duration_sec=time.time() - t0
        )


def test_search_query(query: str, expected_min: int = 0) -> TestResult:
    """Test a specific search query against GDS database"""
    t0 = time.time()
    try:
        handle = Entrez.esearch(db="gds", term=query, retmax=10)
        result = Entrez.read(handle)
        handle.close()
        
        count = int(result.get("Count", 0))
        ids = result.get("IdList", [])
        
        success = count >= expected_min
        return TestResult(
            test_name=f"Search: {query[:50]}...",
            success=success,
            message=f"Found {count} results, got {len(ids)} IDs",
            details={
                "query": query,
                "total_count": count,
                "sample_ids": ids[:5] if ids else []
            },
            duration_sec=time.time() - t0
        )
    except Exception as e:
        return TestResult(
            test_name=f"Search: {query[:50]}...",
            success=False,
            message=f"Query failed: {e}",
            details={"query": query, "error": str(e)},
            duration_sec=time.time() - t0
        )


def run_search_tests(suite: TestSuite):
    """Run battery of search query tests to find working syntax"""
    logger.info("\n" + "="*60)
    logger.info("SEARCH QUERY TESTS")
    logger.info("="*60)
    
    # Test different query syntaxes for single-cell rat data
    test_queries = [
        # Simple queries
        ("scRNA-seq AND rat[Organism] AND gse[ETYP]", 1),
        ("single cell AND Rattus norvegicus[Organism] AND gse[ETYP]", 1),
        ("scRNA-seq Rattus norvegicus gse[ETYP]", 1),
        
        # With quotes (might fail)
        ('"single-cell"[All Fields] AND Rattus norvegicus[Organism] AND gse[ETYP]', 0),
        ('"single cell RNA-seq" AND rat[Organism] AND gse[ETYP]', 0),
        
        # Different organism syntax
        ("scRNA-seq AND Rattus[Organism] AND gse[ETYP]", 1),
        ("scRNA-seq AND rat AND gse[ETYP]", 1),
        
        # Entry type variations
        ("scRNA-seq AND rat[Organism] AND gse[Entry Type]", 1),
        ("scRNA-seq rat[Organism] GSE[ETYP]", 1),
        
        # Broader queries
        ("RNA-seq AND Rattus norvegicus[Organism] AND gse[ETYP]", 1),
        ("transcriptome AND rat[Organism] AND gse[ETYP]", 1),
        
        # Technology-specific
        ("10x genomics AND rat[Organism] AND gse[ETYP]", 0),
        ("Drop-seq AND rat[Organism] AND gse[ETYP]", 0),
        
        # Combined with OR (might be problematic)
        ("(scRNA-seq OR single cell) AND rat[Organism] AND gse[ETYP]", 1),
        
        # Very simple
        ("rat single cell gse[ETYP]", 1),
        ("Rattus norvegicus scRNA gse[ETYP]", 1),
    ]
    
    for query, expected_min in test_queries:
        result = test_search_query(query, expected_min)
        suite.add(result)
        time.sleep(0.4)  # Rate limiting


def test_uid_to_gse_conversion(uid: str) -> TestResult:
    """Test converting a UID to GSE accession"""
    t0 = time.time()
    try:
        handle = Entrez.esummary(db="gds", id=uid)
        result = Entrez.read(handle)
        handle.close()
        
        if result and isinstance(result, list) and len(result) > 0:
            accession = result[0].get("Accession", "")
            title = result[0].get("title", "")[:50]
            
            return TestResult(
                test_name=f"UID→GSE: {uid}",
                success=accession.startswith("GSE"),
                message=f"Converted to {accession}",
                details={"uid": uid, "accession": accession, "title": title},
                duration_sec=time.time() - t0
            )
        else:
            return TestResult(
                test_name=f"UID→GSE: {uid}",
                success=False,
                message="No results returned",
                duration_sec=time.time() - t0
            )
    except Exception as e:
        return TestResult(
            test_name=f"UID→GSE: {uid}",
            success=False,
            message=f"Failed: {e}",
            duration_sec=time.time() - t0
        )


# =============================================================================
# SECTION 2: FTP CONNECTION TESTS
# =============================================================================

def test_ftp_connection() -> TestResult:
    """Test basic FTP connection to NCBI"""
    t0 = time.time()
    try:
        ftp = ftplib.FTP("ftp.ncbi.nlm.nih.gov", timeout=30)
        ftp.login()
        ftp.set_pasv(True)
        welcome = ftp.getwelcome()
        ftp.quit()
        
        return TestResult(
            test_name="FTP Connection",
            success=True,
            message="Successfully connected to NCBI FTP",
            details={"server": "ftp.ncbi.nlm.nih.gov", "welcome": welcome[:50]},
            duration_sec=time.time() - t0
        )
    except Exception as e:
        return TestResult(
            test_name="FTP Connection",
            success=False,
            message=f"Failed: {e}",
            duration_sec=time.time() - t0
        )


def test_ftp_gse_path(gse: str) -> TestResult:
    """Test FTP path construction and access for a GSE"""
    t0 = time.time()
    
    # Construct path: /geo/series/GSEnnn/GSE####/suppl/
    head = gse[:-3] + "nnn"
    remote_path = f"/geo/series/{head}/{gse}/suppl/"
    
    try:
        ftp = ftplib.FTP("ftp.ncbi.nlm.nih.gov", timeout=30)
        ftp.login()
        ftp.set_pasv(True)
        
        try:
            ftp.cwd(remote_path)
            
            # List files
            files = []
            ftp.retrlines("NLST", files.append)
            
            ftp.quit()
            
            return TestResult(
                test_name=f"FTP Path: {gse}",
                success=True,
                message=f"Found {len(files)} files",
                details={
                    "gse": gse,
                    "path": remote_path,
                    "file_count": len(files),
                    "sample_files": files[:10]
                },
                duration_sec=time.time() - t0
            )
        except ftplib.error_perm as e:
            if "550" in str(e):
                return TestResult(
                    test_name=f"FTP Path: {gse}",
                    success=True,  # Not an error, just no suppl files
                    message="Path exists but no supplementary files (550)",
                    details={"gse": gse, "path": remote_path},
                    duration_sec=time.time() - t0
                )
            raise
            
    except Exception as e:
        return TestResult(
            test_name=f"FTP Path: {gse}",
            success=False,
            message=f"Failed: {e}",
            details={"gse": gse, "path": remote_path},
            duration_sec=time.time() - t0
        )


def test_ftp_filelist_txt(gse: str) -> TestResult:
    """Test fetching and parsing filelist.txt"""
    t0 = time.time()
    
    head = gse[:-3] + "nnn"
    remote_path = f"/geo/series/{head}/{gse}/suppl/"
    
    try:
        ftp = ftplib.FTP("ftp.ncbi.nlm.nih.gov", timeout=30)
        ftp.login()
        ftp.set_pasv(True)
        ftp.cwd(remote_path)
        
        # Check if filelist.txt exists
        files = []
        ftp.retrlines("NLST", files.append)
        
        if "filelist.txt" not in files:
            ftp.quit()
            return TestResult(
                test_name=f"filelist.txt: {gse}",
                success=True,  # Not an error
                message="No filelist.txt present",
                details={"gse": gse, "available_files": files[:10]},
                duration_sec=time.time() - t0
            )
        
        # Fetch filelist.txt
        content_lines = []
        ftp.retrlines("RETR filelist.txt", content_lines.append)
        ftp.quit()
        
        content = "\n".join(content_lines)
        
        # Parse it
        archives = []
        files_in_archives = []
        for line in content_lines:
            if line.startswith("#") or "Archive/File" in line:
                continue
            parts = line.split("\t")
            if len(parts) >= 5:
                entry_type = parts[0]
                name = parts[1]
                if entry_type == "Archive":
                    archives.append(name)
                elif entry_type == "File":
                    files_in_archives.append(name)
        
        return TestResult(
            test_name=f"filelist.txt: {gse}",
            success=True,
            message=f"Found {len(archives)} archives, {len(files_in_archives)} files",
            details={
                "gse": gse,
                "archives": archives,
                "sample_files": files_in_archives[:10],
                "total_files_in_archives": len(files_in_archives)
            },
            duration_sec=time.time() - t0
        )
        
    except Exception as e:
        return TestResult(
            test_name=f"filelist.txt: {gse}",
            success=False,
            message=f"Failed: {e}",
            duration_sec=time.time() - t0
        )


def run_ftp_tests(suite: TestSuite, test_gses: List[str] = None):
    """Run FTP tests"""
    logger.info("\n" + "="*60)
    logger.info("FTP CONNECTION TESTS")
    logger.info("="*60)
    
    # Basic connection
    suite.add(test_ftp_connection())
    
    # Test specific GSEs
    if test_gses is None:
        # Default test GSEs (known to exist)
        test_gses = ["GSE123456", "GSE200000", "GSE150000"]
    
    for gse in test_gses:
        suite.add(test_ftp_gse_path(gse))
        time.sleep(0.5)
        suite.add(test_ftp_filelist_txt(gse))
        time.sleep(0.5)


# =============================================================================
# SECTION 3: GEOPARSE TESTS
# =============================================================================

def test_geoparse_fetch(gse: str, temp_dir: str = "/tmp/geo_test") -> TestResult:
    """Test fetching GSE metadata via GEOparse"""
    t0 = time.time()
    
    if not GEOPARSE_AVAILABLE:
        return TestResult(
            test_name=f"GEOparse: {gse}",
            success=False,
            message="GEOparse not installed",
            duration_sec=0
        )
    
    try:
        os.makedirs(temp_dir, exist_ok=True)
        
        geo = GEOparse.get_GEO(geo=gse, destdir=temp_dir, silent=True, annotate_gpl=False)
        
        title = geo.metadata.get("title", [""])[0]
        summary = geo.metadata.get("summary", [""])[0][:100]
        sample_count = len(geo.phenotype_data)
        gsm_list = list(geo.gsms.keys())[:5]
        
        # Check for SRA links
        sra_links = []
        for gsm_name, gsm in geo.gsms.items():
            relations = gsm.metadata.get("relation", [])
            for rel in relations:
                if "SRA:" in rel or "sra" in rel.lower():
                    sra_links.append((gsm_name, rel))
        
        # Check for supplementary files
        suppl_files = []
        for gsm_name, gsm in geo.gsms.items():
            files = gsm.metadata.get("supplementary_file", [])
            for f in files:
                if f and f != "NONE":
                    suppl_files.append((gsm_name, f))
        
        return TestResult(
            test_name=f"GEOparse: {gse}",
            success=True,
            message=f"Fetched {sample_count} samples",
            details={
                "gse": gse,
                "title": title[:50],
                "sample_count": sample_count,
                "gsm_samples": gsm_list,
                "sra_links_found": len(sra_links),
                "sample_sra": sra_links[:3],
                "suppl_files_found": len(suppl_files),
                "sample_suppl": suppl_files[:3]
            },
            duration_sec=time.time() - t0
        )
        
    except Exception as e:
        return TestResult(
            test_name=f"GEOparse: {gse}",
            success=False,
            message=f"Failed: {e}",
            duration_sec=time.time() - t0
        )


def run_geoparse_tests(suite: TestSuite, test_gses: List[str] = None):
    """Run GEOparse tests"""
    logger.info("\n" + "="*60)
    logger.info("GEOPARSE METADATA TESTS")
    logger.info("="*60)
    
    if test_gses is None:
        # Find some real GSEs first via search
        try:
            handle = Entrez.esearch(db="gds", term="scRNA-seq rat[Organism] gse[ETYP]", retmax=3)
            result = Entrez.read(handle)
            handle.close()
            
            uids = result.get("IdList", [])
            test_gses = []
            
            for uid in uids:
                time.sleep(0.4)
                handle = Entrez.esummary(db="gds", id=uid)
                summary = Entrez.read(handle)
                handle.close()
                if summary and isinstance(summary, list):
                    acc = summary[0].get("Accession", "")
                    if acc.startswith("GSE"):
                        test_gses.append(acc)
            
            logger.info(f"Found GSEs to test: {test_gses}")
            
        except Exception as e:
            logger.warning(f"Could not find test GSEs via search: {e}")
            test_gses = []
    
    for gse in test_gses:
        suite.add(test_geoparse_fetch(gse))
        time.sleep(1)


# =============================================================================
# SECTION 4: SRA METADATA TESTS
# =============================================================================

def test_sra_search(srx_or_srr: str) -> TestResult:
    """Test SRA metadata fetch"""
    t0 = time.time()
    
    try:
        handle = Entrez.esearch(db="sra", term=srx_or_srr, retmax=10)
        result = Entrez.read(handle)
        handle.close()
        
        count = int(result.get("Count", 0))
        ids = result.get("IdList", [])
        
        if not ids:
            return TestResult(
                test_name=f"SRA Search: {srx_or_srr}",
                success=False,
                message="No results found",
                duration_sec=time.time() - t0
            )
        
        time.sleep(0.4)
        
        # Fetch details
        handle = Entrez.efetch(db="sra", id=ids[0], rettype="full", retmode="xml")
        xml_content = handle.read()
        handle.close()
        
        # Quick parse
        import xml.etree.ElementTree as ET
        root = ET.fromstring(xml_content)
        
        runs = root.findall(".//RUN")
        experiments = root.findall(".//EXPERIMENT")
        
        return TestResult(
            test_name=f"SRA Search: {srx_or_srr}",
            success=True,
            message=f"Found {len(runs)} runs, {len(experiments)} experiments",
            details={
                "accession": srx_or_srr,
                "run_count": len(runs),
                "experiment_count": len(experiments)
            },
            duration_sec=time.time() - t0
        )
        
    except Exception as e:
        return TestResult(
            test_name=f"SRA Search: {srx_or_srr}",
            success=False,
            message=f"Failed: {e}",
            duration_sec=time.time() - t0
        )


# =============================================================================
# SECTION 5: PATTERN MATCHING TESTS
# =============================================================================

def test_file_patterns():
    """Test file pattern matching logic"""
    import re
    
    SC_PATTERNS = [
        r'matrix\.mtx(\.gz)?$',
        r'barcodes\.tsv(\.gz)?$',
        r'(features|genes)\.tsv(\.gz)?$',
        r'\.h5ad$',
        r'\.loom$',
        r'\.rds$',
        r'_RAW\.tar$',
    ]
    
    BULK_PATTERNS = [
        r'(gene|transcript)[-_]?counts?\.(tsv|csv|txt)(\.gz)?$',
        r'(tpm|fpkm|rpkm)\.(tsv|csv|txt)(\.gz)?$',
        r'featureCounts.*\.(txt|tsv)(\.gz)?$',
    ]
    
    # Test files
    test_files = {
        # Should match single-cell
        "matrix.mtx.gz": ("SC", True),
        "barcodes.tsv.gz": ("SC", True),
        "features.tsv.gz": ("SC", True),
        "genes.tsv": ("SC", True),
        "sample.h5ad": ("SC", True),
        "data.loom": ("SC", True),
        "seurat.rds": ("SC", True),
        "GSE123_RAW.tar": ("SC", True),
        
        # Should match bulk
        "gene_counts.tsv.gz": ("BULK", True),
        "transcript_counts.csv": ("BULK", True),
        "tpm.tsv": ("BULK", True),
        "fpkm.csv.gz": ("BULK", True),
        "featureCounts_output.txt": ("BULK", True),
        
        # Should NOT match
        "README.txt": ("NONE", False),
        "sample_info.xlsx": ("NONE", False),
        "filelist.txt": ("NONE", False),
    }
    
    sc_compiled = [re.compile(p, re.IGNORECASE) for p in SC_PATTERNS]
    bulk_compiled = [re.compile(p, re.IGNORECASE) for p in BULK_PATTERNS]
    
    results = []
    for filename, (expected_type, should_match) in test_files.items():
        sc_match = any(p.search(filename) for p in sc_compiled)
        bulk_match = any(p.search(filename) for p in bulk_compiled)
        
        actual_match = sc_match or bulk_match
        actual_type = "SC" if sc_match else ("BULK" if bulk_match else "NONE")
        
        correct = (actual_match == should_match) and (actual_type == expected_type or not should_match)
        results.append((filename, expected_type, actual_type, correct))
    
    passed = sum(1 for r in results if r[3])
    failed = sum(1 for r in results if not r[3])
    
    return TestResult(
        test_name="File Pattern Matching",
        success=failed == 0,
        message=f"{passed} passed, {failed} failed",
        details={
            "tests": [(f, exp, act, "✓" if ok else "✗") for f, exp, act, ok in results]
        }
    )


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Test GEO Harvester approaches")
    parser.add_argument("--email", required=True, help="Email for NCBI E-utilities")
    parser.add_argument("--api-key", default=None, help="NCBI API key")
    parser.add_argument("--test-all", action="store_true", help="Run all tests")
    parser.add_argument("--test-diagnose", action="store_true", help="Run diagnostic tests (start here if things fail)")
    parser.add_argument("--test-search", action="store_true", help="Test search queries only")
    parser.add_argument("--test-ftp", action="store_true", help="Test FTP only")
    parser.add_argument("--test-geoparse", action="store_true", help="Test GEOparse only")
    parser.add_argument("--test-sra", action="store_true", help="Test SRA only")
    parser.add_argument("--test-patterns", action="store_true", help="Test file patterns only")
    parser.add_argument("--gse", nargs="+", default=None, help="Specific GSEs to test")
    parser.add_argument("--output", default=None, help="Output JSON file for results")
    
    args = parser.parse_args()
    
    # Initialize
    suite = TestSuite()
    
    logger.info("="*60)
    logger.info("GEO HARVESTER TEST SUITE")
    logger.info("="*60)
    logger.info(f"Email: {args.email}")
    logger.info(f"Biopython: {'v' + BIOPYTHON_VERSION if BIOPYTHON_AVAILABLE else 'NOT INSTALLED'}")
    logger.info(f"GEOparse: {'Available' if GEOPARSE_AVAILABLE else 'NOT INSTALLED'}")
    
    # Get API key from argument or environment variable
    api_key = args.api_key or os.environ.get("NCBI_API_KEY")
    
    if BIOPYTHON_AVAILABLE:
        Entrez.email = args.email
        # Only set api_key if we actually have one (avoid empty string!)
        if api_key:
            Entrez.api_key = api_key
            logger.info(f"API Key: Set from {'--api-key argument' if args.api_key else 'NCBI_API_KEY environment variable'}")
        else:
            Entrez.api_key = None  # Explicitly None to avoid empty string bug
            logger.info("API Key: Not set (will be rate limited to 3 req/sec)")
            logger.info("  Tip: Set NCBI_API_KEY environment variable or use --api-key")
    
    # Determine what to test
    run_all = args.test_all or not any([
        args.test_diagnose, args.test_search, args.test_ftp, 
        args.test_geoparse, args.test_sra, args.test_patterns
    ])
    
    # ALWAYS run diagnostics first if everything else failed or if explicitly requested
    if args.test_diagnose or run_all:
        run_diagnostic_tests(suite, args.email)
    
    # Run other tests only if Biopython is available
    if BIOPYTHON_AVAILABLE:
        if run_all or args.test_search:
            suite.add(test_entrez_connection(args.email, args.api_key))
            time.sleep(0.5)
            run_search_tests(suite)
        
        if run_all or args.test_geoparse:
            run_geoparse_tests(suite, args.gse)
        
        if run_all or args.test_sra:
            logger.info("\n" + "="*60)
            logger.info("SRA METADATA TESTS")
            logger.info("="*60)
            for acc in ["SRX000001", "SRR000001"]:
                suite.add(test_sra_search(acc))
                time.sleep(0.5)
    
    # FTP tests don't need Biopython
    if run_all or args.test_ftp:
        run_ftp_tests(suite, args.gse)
    
    # Pattern tests don't need anything external
    if run_all or args.test_patterns:
        logger.info("\n" + "="*60)
        logger.info("FILE PATTERN TESTS")
        logger.info("="*60)
        suite.add(test_file_patterns())
    
    # Summary
    passed, failed = suite.summary()
    
    # Output JSON if requested
    if args.output:
        output_data = {
            "summary": {"passed": passed, "failed": failed},
            "environment": {
                "biopython_version": BIOPYTHON_VERSION,
                "geoparse_available": GEOPARSE_AVAILABLE,
            },
            "results": [asdict(r) for r in suite.results]
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        logger.info(f"\nResults written to: {args.output}")
    
    # Provide recommendations if there were failures
    if failed > 0:
        logger.info("\n" + "="*60)
        logger.info("RECOMMENDATIONS")
        logger.info("="*60)
        
        # Check specific failure patterns
        entrez_failures = [r for r in suite.results if "Entrez" in r.test_name and not r.success]
        direct_http_results = [r for r in suite.results if "Direct HTTP" in r.test_name]
        direct_http_success = any(r.success for r in direct_http_results)
        
        if entrez_failures and direct_http_success:
            logger.info("→ Biopython Entrez is failing but direct HTTP works.")
            logger.info("  This suggests a Biopython configuration issue.")
            logger.info("  Try: pip install --upgrade biopython")
            logger.info("  Or use the direct HTTP approach in the harvester.")
        
        if entrez_failures and not direct_http_success:
            logger.info("→ Both Biopython and direct HTTP are failing.")
            logger.info("  This might be a network/firewall issue or NCBI API issue.")
            logger.info("  Check if you can access https://eutils.ncbi.nlm.nih.gov in a browser.")
    
    # Exit code
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()