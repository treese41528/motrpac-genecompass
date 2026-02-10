#!/usr/bin/env python3
"""
llm_study_analyzer_enhanced.py - Enhanced study classification using Claude API

ENHANCEMENTS over original:
- Sends RAW metadata files (SOFT, SDRF, IDF) to Claude for deeper analysis
- Includes full directory/file listing for each study
- Validates already-extracted catalog fields
- MoTrPAC-focused analysis with exercise physiology emphasis
- Cross-references matrix analysis results

Usage:
    python llm_study_analyzer_enhanced.py --config config.yaml --max-studies 5  # Test
    python llm_study_analyzer_enhanced.py --config config.yaml                   # Full run
    python llm_study_analyzer_enhanced.py --config config.yaml --resume          # Resume

Requires:
    pip install anthropic pyyaml
    export ANTHROPIC_API_KEY=your_key
"""

import os
import sys
import json
import argparse
import logging
import time
import re
import gzip
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
import hashlib

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================================================
# ANALYSIS OUTPUT SCHEMA - Comprehensive information extraction
# =============================================================================

ANALYSIS_SCHEMA = """
{
  "accession": "GSE/E-MTAB ID",
  
  "metadata_validation": {
    "extracted_organism_correct": true/false,
    "actual_organism": "what raw metadata shows",
    "extracted_tissues_correct": true/false,
    "actual_tissues": ["tissue list from raw metadata"]
  },
  
  "study_overview": {
    "title": "study title",
    "summary": "1-2 sentence description (max 200 chars)",
    "primary_topic": "specific topic: hepatocellular_carcinoma, type2_diabetes, cardiac_hypertrophy, etc.",
    "topic_category": "cancer/metabolic/cardiovascular/neurological/liver/kidney/muscle/immune/developmental/aging/toxicology/normal_biology/other",
    "keywords": ["10-15 keywords max"]
  },
  
  "study_type": {
    "data_type": "single_cell/bulk/microarray",
    "is_single_cell": true/false,
    "is_time_series": true/false,
    "is_disease_study": true/false,
    "is_treatment_study": true/false,
    "is_genetic_perturbation": true/false
  },
  
  "organism": {
    "species": "Rattus norvegicus/Mus musculus/etc",
    "strain": "F344/Sprague-Dawley/Wistar/etc",
    "sex": "male/female/both/not_specified",
    "age_value": "numeric",
    "age_unit": "days/weeks/months/years",
    "life_stage": "embryonic/neonatal/juvenile/adult/aged"
  },
  
  "tissues": [
    {
      "name": "standardized name",
      "category": "brain/heart/liver/kidney/lung/skeletal_muscle/adipose/blood/other",
      "motrpac_match": "MoTrPAC tissue name or null"
    }
  ],
  
  "cell_types": ["cell types mentioned"],
  
  "disease_condition": {
    "has_disease_model": true/false,
    "disease_name": "disease name or null",
    "disease_type": "cancer/metabolic/cardiovascular/neurological/other/null",
    "induction_method": "genetic/diet/chemical/surgical/null"
  },
  
  "treatments": {
    "has_drug": true/false,
    "drug_names": ["drugs if any"],
    "has_diet": true/false,
    "diet_type": "high_fat/caloric_restriction/other/null",
    "has_exercise": true/false,
    "exercise_type": "treadmill/wheel/swimming/null",
    "has_genetic": true/false,
    "genetic_type": "knockout/overexpression/null",
    "target_genes": ["genes if any"]
  },
  
  "experimental_design": {
    "groups": ["list of experimental groups"],
    "time_points": ["time points if any"],
    "total_samples": "N",
    "samples_per_group": "N per group"
  },
  
  "technical": {
    "platform": "10x/Smart-seq2/Illumina/Affymetrix/etc",
    "platform_id": "GPL number"
  },
  
  "files": {
    "has_count_matrix": true/false,
    "data_quality": "complete/partial/minimal"
  },
  
  "utility_for_motrpac": {
    "is_rat": true/false,
    "genecompass_useful": true/false,
    "deconvolution_useful": true/false,
    "grn_useful": true/false,
    "motrpac_tissues": ["matching tissues"],
    "exercise_relevance": "none/indirect/direct"
  },
  
  "summary": {
    "what_is_this_study": "Brief description (max 150 chars)",
    "how_we_can_use_it": ["1-3 use cases"]
  }
}
"""

# =============================================================================
# ENHANCED SYSTEM PROMPT - More detailed MoTrPAC context
# =============================================================================

SYSTEM_PROMPT = """You are an expert bioinformatics curator. Extract information from gene expression study metadata and return structured JSON.

## Your Task
1. EXTRACT biological and technical information from raw metadata
2. VALIDATE what our automated pipeline extracted
3. DESCRIBE what the study is about
4. IDENTIFY experimental details: treatments, conditions, time points

## Key Information to Extract
- Organism and strain (F344, Sprague-Dawley, Wistar, etc.)
- Sex, age, life stage
- Tissues and cell types
- Disease models, drug treatments, genetic modifications, diet, exercise
- Time points (if time series)
- Sample sizes, platform details

## Study Topics (examples)
Cancer, diabetes, obesity, cardiovascular, neurological, liver/kidney disease, muscle biology, development, aging, immunology, drug/toxicology, cell atlases, exercise, and many others.

## MoTrPAC Context
We collect rat transcriptomics for MoTrPAC. Useful data includes:
- ANY quality rat transcriptomics (for GeneCompass training)
- Single-cell data from MoTrPAC tissues (deconvolution references)
- Perturbation or time-series data (GRN inference)

**18 MoTrPAC Tissues:** gastrocnemius, vastus lateralis, heart, liver, kidney, lung, WAT-SC, BAT, adrenal, blood RNA, colon, small intestine, hippocampus, hypothalamus, cortex, spleen, ovary, testis

## CRITICAL INSTRUCTIONS
1. Return ONLY valid JSON - no markdown, no explanations
2. BE CONCISE - use short phrases, not long sentences
3. For arrays, include only the most important items (max 10-15 keywords)
4. Keep summary fields under 200 characters
5. DO NOT truncate mid-field - complete every field you start

Return ONLY the JSON object following the schema provided."""

# =============================================================================
# RAW METADATA READING FUNCTIONS
# =============================================================================

def read_soft_file(filepath: Path, max_bytes: int = 100000) -> str:
    """Read GEO SOFT file content, truncating if too large."""
    try:
        opener = gzip.open if str(filepath).endswith('.gz') else open
        mode = 'rt' if str(filepath).endswith('.gz') else 'r'
        
        with opener(filepath, mode, errors='replace') as f:
            content = f.read(max_bytes)
            
        if len(content) >= max_bytes:
            content += f"\n... [TRUNCATED - file exceeds {max_bytes} bytes]"
            
        return content
    except Exception as e:
        return f"[Error reading SOFT file: {e}]"


def read_sdrf_file(filepath: Path, max_bytes: int = 50000) -> str:
    """Read ArrayExpress SDRF file content."""
    try:
        opener = gzip.open if str(filepath).endswith('.gz') else open
        mode = 'rt' if str(filepath).endswith('.gz') else 'r'
        
        with opener(filepath, mode, errors='replace') as f:
            content = f.read(max_bytes)
            
        if len(content) >= max_bytes:
            content += f"\n... [TRUNCATED - file exceeds {max_bytes} bytes]"
            
        return content
    except Exception as e:
        return f"[Error reading SDRF file: {e}]"


def read_idf_file(filepath: Path, max_bytes: int = 30000) -> str:
    """Read ArrayExpress IDF file content."""
    try:
        opener = gzip.open if str(filepath).endswith('.gz') else open
        mode = 'rt' if str(filepath).endswith('.gz') else 'r'
        
        with opener(filepath, mode, errors='replace') as f:
            content = f.read(max_bytes)
            
        return content
    except Exception as e:
        return f"[Error reading IDF file: {e}]"


def find_and_read_metadata_files(study_path: Path) -> Dict[str, str]:
    """Find and read all metadata files in a study directory."""
    metadata = {
        'soft_content': None,
        'sdrf_content': None,
        'idf_content': None,
        'other_metadata': []
    }
    
    if not study_path.exists():
        return metadata
    
    try:
        for fpath in study_path.rglob('*'):
            if not fpath.is_file():
                continue
                
            fname = fpath.name.lower()
            
            # SOFT files (GEO)
            if fname.endswith('.soft') or fname.endswith('.soft.gz'):
                if 'family' in fname:  # Prefer family.soft which has full metadata
                    metadata['soft_content'] = read_soft_file(fpath)
                elif metadata['soft_content'] is None:
                    metadata['soft_content'] = read_soft_file(fpath)
            
            # SDRF files (ArrayExpress)
            elif fname.endswith('.sdrf.txt') or fname.endswith('.sdrf.txt.gz'):
                metadata['sdrf_content'] = read_sdrf_file(fpath)
            
            # IDF files (ArrayExpress)
            elif fname.endswith('.idf.txt') or fname.endswith('.idf.txt.gz'):
                metadata['idf_content'] = read_idf_file(fpath)
            
            # Other potential metadata files
            elif any(x in fname for x in ['sample', 'metadata', 'phenotype', 'clinical']):
                if fname.endswith(('.txt', '.csv', '.tsv', '.txt.gz', '.csv.gz', '.tsv.gz')):
                    try:
                        opener = gzip.open if fname.endswith('.gz') else open
                        with opener(fpath, 'rt', errors='replace') as f:
                            content = f.read(20000)
                        metadata['other_metadata'].append({
                            'filename': fpath.name,
                            'content': content[:20000]
                        })
                    except:
                        pass
                        
    except Exception as e:
        logger.warning(f"Error reading metadata files: {e}")
    
    return metadata


def get_directory_tree(study_path: Path, max_depth: int = 3, max_files: int = 200) -> str:
    """Generate a directory tree listing for the study."""
    if not study_path.exists():
        return "[Directory not found]"
    
    lines = []
    file_count = 0
    
    def format_size(size_bytes: int) -> str:
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024:
                return f"{size_bytes:.1f}{unit}"
            size_bytes /= 1024
        return f"{size_bytes:.1f}TB"
    
    def walk_dir(path: Path, prefix: str = "", depth: int = 0):
        nonlocal file_count
        
        if depth > max_depth:
            return
        
        try:
            items = sorted(path.iterdir(), key=lambda x: (x.is_file(), x.name.lower()))
        except PermissionError:
            return
        
        for i, item in enumerate(items):
            if file_count >= max_files:
                lines.append(f"{prefix}... [truncated, {max_files}+ files]")
                return
            
            is_last = (i == len(items) - 1)
            connector = "└── " if is_last else "├── "
            
            if item.is_file():
                try:
                    size = format_size(item.stat().st_size)
                    lines.append(f"{prefix}{connector}{item.name} ({size})")
                except:
                    lines.append(f"{prefix}{connector}{item.name}")
                file_count += 1
            else:
                lines.append(f"{prefix}{connector}{item.name}/")
                extension = "    " if is_last else "│   "
                walk_dir(item, prefix + extension, depth + 1)
    
    lines.append(f"{study_path.name}/")
    walk_dir(study_path, "", 0)
    
    return "\n".join(lines)


# =============================================================================
# PROMPT BUILDING
# =============================================================================

USER_PROMPT_TEMPLATE = """Analyze this rat transcriptomics study comprehensively using ALL provided information.

## Study Identification
**Accession:** {accession}
**Source:** {source}
**Data Type (from catalog):** {data_type}

## Pre-extracted Metadata (from our automated pipeline - VALIDATE THESE)
**Title:** {title}
**Summary:** {summary}
**Overall Design:** {overall_design}
**Organisms (extracted):** {organism}
**Tissues (extracted):** {tissues}
**Technologies (detected):** {technologies}
**Platforms:** {platforms}
**Sample Count:** {sample_count}
**Submission Date:** {submission_date}
**PubMed IDs:** {pubmed_ids}

## Matrix Analysis Results
- Gene count: {n_genes}
- Cell count: {n_cells}
- Sample count (from matrix): {n_samples_matrix}
- Data modality: {data_modality}
- Gene ID type: {gene_id_type}
- Matrix formats found: {formats}
- Confidence: {confidence}
- Is unfiltered 10x: {is_unfiltered}

## Directory Structure
```
{directory_tree}
```

## RAW METADATA FILES - READ THESE CAREFULLY

{raw_metadata_section}

## Instructions

1. **VALIDATE** the pre-extracted metadata against the raw files
2. **EXTRACT** any additional details from raw metadata our pipeline missed
3. **ASSESS** relevance to MoTrPAC research and exercise biology
4. **EVALUATE** data quality and usability based on files available

Return a JSON object following this schema:

{schema}

Return ONLY valid JSON, no additional text or markdown formatting."""


class StudyAnalyzer:
    """Enhanced study analyzer with raw metadata support."""
    
    def __init__(self, api_key: str = None, model: str = "claude-haiku-4-5",
                 debug_dir: str = "debug_responses", save_all_responses: bool = False):
        if not HAS_ANTHROPIC:
            raise ImportError("anthropic package required. Install: pip install anthropic")
        
        self.api_key = api_key or os.environ.get('ANTHROPIC_API_KEY')
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not set")
        
        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.model = model
        self.rate_limiter = RateLimiter(calls_per_minute=40)  # Slightly lower for safety
        self.debug_dir = Path(debug_dir)
        self.save_all_responses = save_all_responses
    
    def build_prompt(self, study: Dict, matrix_info: Dict, study_path: Path) -> str:
        """Build enhanced prompt with raw metadata and directory listing."""
        
        # Basic study info
        accession = study.get('accession', 'Unknown')
        source = study.get('source', 'GEO')
        data_type = study.get('data_type', 'unknown')
        title = study.get('title') or 'Not provided'
        summary = study.get('summary') or study.get('abstract') or 'Not provided'
        overall_design = study.get('overall_design') or 'Not provided'
        organism = ', '.join(study.get('organism', [])) or 'Not specified'
        tissues = ', '.join(study.get('tissues', [])) or 'Not specified'
        technologies = ', '.join(study.get('technologies', [])) or 'Not detected'
        platforms = ', '.join(study.get('platforms', [])) or 'Not specified'
        sample_count = study.get('sample_count', 'Unknown')
        submission_date = study.get('submission_date', 'Unknown')
        pubmed_ids = ', '.join(study.get('pubmed_ids', [])) or 'None'
        
        # Matrix info
        n_genes = matrix_info.get('n_genes') or 'Unknown'
        n_cells = matrix_info.get('n_cells') or 'N/A'
        n_samples_matrix = matrix_info.get('n_samples') or 'N/A'
        data_modality = matrix_info.get('data_modality', 'unknown')
        gene_id_type = matrix_info.get('gene_id_type', 'unknown')
        formats = matrix_info.get('formats', {})
        confidence = matrix_info.get('confidence', 'unknown')
        is_unfiltered = matrix_info.get('is_unfiltered_10x', False)
        
        # Get directory tree
        directory_tree = get_directory_tree(study_path)
        
        # Get raw metadata
        raw_metadata = find_and_read_metadata_files(study_path)
        
        # Build raw metadata section
        raw_metadata_parts = []
        
        if raw_metadata.get('soft_content'):
            raw_metadata_parts.append("### GEO SOFT File Content:\n```\n" + 
                                      raw_metadata['soft_content'] + "\n```")
        
        if raw_metadata.get('idf_content'):
            raw_metadata_parts.append("### ArrayExpress IDF File Content:\n```\n" + 
                                      raw_metadata['idf_content'] + "\n```")
        
        if raw_metadata.get('sdrf_content'):
            raw_metadata_parts.append("### ArrayExpress SDRF File Content:\n```\n" + 
                                      raw_metadata['sdrf_content'] + "\n```")
        
        for other in raw_metadata.get('other_metadata', [])[:3]:  # Limit to 3 other files
            raw_metadata_parts.append(f"### Other Metadata ({other['filename']}):\n```\n" + 
                                      other['content'] + "\n```")
        
        if not raw_metadata_parts:
            raw_metadata_section = "[No raw metadata files found in study directory]"
        else:
            raw_metadata_section = "\n\n".join(raw_metadata_parts)
        
        prompt = USER_PROMPT_TEMPLATE.format(
            accession=accession,
            source=source,
            data_type=data_type,
            title=title,
            summary=summary,
            overall_design=overall_design,
            organism=organism,
            tissues=tissues,
            technologies=technologies,
            platforms=platforms,
            sample_count=sample_count,
            submission_date=submission_date,
            pubmed_ids=pubmed_ids,
            n_genes=n_genes,
            n_cells=n_cells,
            n_samples_matrix=n_samples_matrix,
            data_modality=data_modality,
            gene_id_type=gene_id_type,
            formats=formats,
            confidence=confidence,
            is_unfiltered=is_unfiltered,
            directory_tree=directory_tree,
            raw_metadata_section=raw_metadata_section,
            schema=ANALYSIS_SCHEMA
        )
        
        return prompt
    
    def analyze_study(self, study: Dict, matrix_info: Dict, study_path: Path) -> Dict:
        """Analyze a single study using Claude with raw metadata."""
        
        accession = study.get('accession', 'Unknown')
        
        try:
            self.rate_limiter.wait()
            
            prompt = self.build_prompt(study, matrix_info, study_path)
            
            # Log prompt size for debugging
            prompt_chars = len(prompt)
            logger.info(f"{accession}: Sending prompt ({prompt_chars:,} chars)")
            
            response = self.client.messages.create(
                model=self.model,
                max_tokens=8192,  # Increased significantly for detailed JSON
                temperature=0,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}]
            )
            
            content = response.content[0].text
            output_tokens = response.usage.output_tokens
            input_tokens = response.usage.input_tokens
            
            logger.info(f"{accession}: Received response ({len(content):,} chars, {output_tokens} tokens out, {input_tokens} tokens in)")
            
            # Check for truncation
            if response.stop_reason == "max_tokens":
                logger.warning(f"{accession}: OUTPUT TRUNCATED - hit max_tokens limit!")
                logger.warning(f"{accession}: Response ended with: ...{content[-100:]}")
                content = self._fix_truncated_json(content)
                logger.info(f"{accession}: Attempted JSON fix, now {len(content):,} chars")
            
            # Extract JSON - handle various formats
            json_str = self._extract_json(content)
            
            # Debug: show what we're trying to parse
            logger.debug(f"{accession}: JSON starts with: {json_str[:200]}...")
            logger.debug(f"{accession}: JSON ends with: ...{json_str[-200:]}")
            
            try:
                analysis = json.loads(json_str)
            except json.JSONDecodeError as e:
                # More detailed error logging
                logger.error(f"{accession}: JSON PARSE FAILED at position {e.pos}")
                logger.error(f"{accession}: Error: {e.msg}")
                
                # Show context around error
                start = max(0, e.pos - 100)
                end = min(len(json_str), e.pos + 100)
                context = json_str[start:end]
                error_marker = ' ' * min(100, e.pos - start) + '^'
                logger.error(f"{accession}: Context around error:\n{context}\n{error_marker}")
                
                # Save failed response for debugging
                self._save_debug_response(accession, content, json_str, e)
                
                raise  # Re-raise to be caught by outer handler
            
            analysis['_meta'] = {
                'analyzed_at': datetime.now().isoformat(),
                'model': self.model,
                'input_tokens': input_tokens,
                'output_tokens': output_tokens,
                'prompt_chars': prompt_chars,
                'response_chars': len(content),
                'truncated': response.stop_reason == "max_tokens",
                'stop_reason': response.stop_reason,
                'success': True
            }
            
            # Optionally save all responses for debugging
            if self.save_all_responses:
                self._save_debug_response(accession, content, json_str, success=True)
            
            logger.info(f"{accession}: ✓ Successfully parsed")
            return analysis
            
        except json.JSONDecodeError as e:
            logger.error(f"{accession}: JSON parse error - {e}")
            return {
                'accession': accession,
                '_meta': {
                    'analyzed_at': datetime.now().isoformat(),
                    'model': self.model,
                    'success': False,
                    'error': f'JSON parse error: {str(e)}',
                    'error_position': e.pos if hasattr(e, 'pos') else None,
                    'response_length': len(content) if 'content' in dir() else None,
                    'raw_response_end': content[-500:] if 'content' in dir() else None
                }
            }
        except Exception as e:
            logger.error(f"{accession}: API error - {e}")
            import traceback
            logger.debug(f"{accession}: Traceback:\n{traceback.format_exc()}")
            return {
                'accession': accession,
                '_meta': {
                    'analyzed_at': datetime.now().isoformat(),
                    'model': self.model,
                    'success': False,
                    'error': str(e),
                    'error_type': type(e).__name__
                }
            }
    
    def _save_debug_response(self, accession: str, raw_content: str, json_str: str, 
                               error: json.JSONDecodeError = None, success: bool = False):
        """Save response for debugging."""
        self.debug_dir.mkdir(exist_ok=True)
        
        suffix = "_success" if success else "_failed"
        debug_file = self.debug_dir / f"{accession}{suffix}.txt"
        
        with open(debug_file, 'w') as f:
            f.write(f"Accession: {accession}\n")
            f.write(f"Status: {'SUCCESS' if success else 'FAILED'}\n")
            if error:
                f.write(f"Error: {error}\n")
                f.write(f"Error position: {error.pos}\n")
            f.write(f"Raw content length: {len(raw_content)}\n")
            f.write(f"JSON string length: {len(json_str)}\n")
            f.write("\n" + "="*60 + "\n")
            f.write("RAW CONTENT:\n")
            f.write("="*60 + "\n")
            f.write(raw_content)
            f.write("\n\n" + "="*60 + "\n")
            f.write("EXTRACTED JSON:\n")
            f.write("="*60 + "\n")
            f.write(json_str)
        
        logger.info(f"{accession}: Saved debug info to {debug_file}")
    
    def _extract_json(self, content: str) -> str:
        """Extract JSON from response, handling various formats."""
        content = content.strip()
        
        # Try direct parse first
        if content.startswith('{'):
            return content
        
        # Try to extract from markdown code blocks
        if '```json' in content:
            json_str = content.split('```json')[1].split('```')[0].strip()
            return json_str
        
        if '```' in content:
            parts = content.split('```')
            for part in parts:
                part = part.strip()
                if part.startswith('{'):
                    return part
        
        # Find JSON object boundaries
        start = content.find('{')
        if start != -1:
            # Find matching closing brace
            depth = 0
            for i, char in enumerate(content[start:], start):
                if char == '{':
                    depth += 1
                elif char == '}':
                    depth -= 1
                    if depth == 0:
                        return content[start:i+1]
            # No matching brace found, return from start
            return content[start:]
        
        return content
    
    def _fix_truncated_json(self, content: str) -> str:
        """Attempt to fix truncated JSON by closing open structures."""
        content = content.rstrip()
        
        # Count open braces and brackets
        open_braces = content.count('{') - content.count('}')
        open_brackets = content.count('[') - content.count(']')
        
        # Check if we're in the middle of a string
        in_string = False
        escape = False
        for char in content:
            if escape:
                escape = False
                continue
            if char == '\\':
                escape = True
                continue
            if char == '"':
                in_string = not in_string
        
        # Close the string if needed
        if in_string:
            content += '"'
        
        # Remove trailing comma if present
        content = content.rstrip()
        if content.endswith(','):
            content = content[:-1]
        
        # Close brackets and braces
        content += ']' * open_brackets
        content += '}' * open_braces
        
        return content


# =============================================================================
# RATE LIMITER
# =============================================================================

class RateLimiter:
    """Simple rate limiter for API calls."""
    
    def __init__(self, calls_per_minute: int = 50):
        self.calls_per_minute = calls_per_minute
        self.min_interval = 60.0 / calls_per_minute
        self.last_call = 0
    
    def wait(self):
        """Wait if necessary to respect rate limit."""
        now = time.time()
        elapsed = now - self.last_call
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self.last_call = time.time()


# =============================================================================
# DATA LOADING
# =============================================================================

def load_data(config: Dict) -> Tuple[List[Dict], Dict[str, Dict], Dict[Tuple[str, str], Path]]:
    """Load catalog, matrix analysis data, and build source paths."""
    
    catalog_dir = Path(config.get('catalog_dir', './catalog'))
    data_root = Path(config.get('data_root', '.'))
    
    # Load catalog
    catalog_path = catalog_dir / 'master_catalog.json'
    logger.info(f"Loading catalog from {catalog_path}")
    with open(catalog_path) as f:
        catalog = json.load(f)
    studies = catalog.get('studies', [])
    
    # Load matrix analysis
    matrix_path = catalog_dir / 'matrix_analysis.json'
    logger.info(f"Loading matrix analysis from {matrix_path}")
    try:
        with open(matrix_path) as f:
            matrix_data = json.load(f)
        matrix_by_acc = {s['accession']: s for s in matrix_data.get('study_stats', [])}
    except FileNotFoundError:
        logger.warning("Matrix analysis not found, proceeding without it")
        matrix_by_acc = {}
    
    # Build source paths from config
    source_paths = {}
    sources = config.get('sources', {})
    for source_name, source_config in sources.items():
        for dtype in ['single_cell', 'bulk']:
            dtype_config = source_config.get(dtype, {})
            if dtype_config.get('enabled', True):
                rel_path = dtype_config.get('path', '')
                if rel_path:
                    full_path = data_root / rel_path
                    if full_path.exists():
                        source_paths[(source_name, dtype)] = full_path
    
    logger.info(f"Loaded {len(studies)} studies, {len(matrix_by_acc)} with matrix info")
    logger.info(f"Source paths: {list(source_paths.keys())}")
    
    return studies, matrix_by_acc, source_paths


def get_study_path(study: Dict, source_paths: Dict[Tuple[str, str], Path]) -> Optional[Path]:
    """Get the filesystem path for a study."""
    accession = study.get('accession')
    source = study.get('source', 'geo')
    dtype = study.get('data_type', 'bulk')
    
    base_path = source_paths.get((source, dtype))
    if base_path:
        study_path = base_path / accession
        if study_path.exists():
            return study_path
    
    # Try other combinations
    for (src, dt), path in source_paths.items():
        study_path = path / accession
        if study_path.exists():
            return study_path
    
    return None


# =============================================================================
# BATCH PROCESSING
# =============================================================================

def process_studies(
    studies: List[Dict],
    matrix_by_acc: Dict[str, Dict],
    source_paths: Dict[Tuple[str, str], Path],
    output_path: Path,
    max_studies: int = None,
    resume: bool = True,  # DEFAULT TO TRUE - safer behavior
    force_overwrite: bool = False,
    workers: int = 1,  # Default to 1 for sequential processing (safer with rate limits)
    filter_usable: bool = False,
    filter_organism: str = None,
    model: str = "claude-haiku-4-5",
    debug_dir: str = "debug_responses",
    save_all_responses: bool = False
) -> List[Dict]:
    """Process studies with Claude API.
    
    Args:
        resume: If True (default), skip already-processed studies and append results.
                If False, will warn if output exists unless force_overwrite=True.
        force_overwrite: If True, overwrite existing output without warning.
        model: Claude model to use for analysis.
        debug_dir: Directory to save debug files (default: debug_responses)
        save_all_responses: If True, save all API responses (not just failures)
    """
    
    analyzer = StudyAnalyzer(model=model, debug_dir=debug_dir, save_all_responses=save_all_responses)
    logger.info(f"Using model: {model}")
    if save_all_responses:
        logger.info(f"Saving all responses to: {debug_dir}/")
    
    # ALWAYS check for existing results
    processed = {}
    existing_count = 0
    if output_path.exists():
        try:
            with open(output_path) as f:
                existing = json.load(f)
                existing_count = len(existing.get('analyses', []))
                
                if resume:
                    # Load existing results to skip and append
                    processed = {r['accession']: r for r in existing.get('analyses', [])}
                    logger.info(f"Resume mode: {len(processed)} studies already processed, will skip and append")
                elif not force_overwrite:
                    # Warn user about potential data loss
                    logger.warning(f"Output file exists with {existing_count} analyses!")
                    logger.warning(f"Use --resume to append, or --force to overwrite")
                    response = input(f"Overwrite {output_path}? (y/N): ")
                    if response.lower() != 'y':
                        logger.info("Aborted. Use --resume to continue from previous run.")
                        return []
                    logger.info("User confirmed overwrite")
                else:
                    logger.warning(f"Force overwrite: discarding {existing_count} existing analyses")
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Could not parse existing output file: {e}")
            logger.info("Starting fresh")
    
    # Filter studies
    filtered_studies = studies
    
    if filter_organism:
        org_lower = filter_organism.lower()
        filtered_studies = [s for s in filtered_studies 
                          if any(org_lower in o.lower() for o in s.get('organism', []))]
        logger.info(f"Filtered to {len(filtered_studies)} {filter_organism} studies")
    
    if filter_usable:
        usable_acc = {
            acc for acc, m in matrix_by_acc.items()
            if m.get('n_genes') and 15000 <= m['n_genes'] <= 150000
            and m.get('data_modality') in ['mrna', 'unknown']
            and not m.get('is_unfiltered_10x')
        }
        filtered_studies = [s for s in filtered_studies if s['accession'] in usable_acc]
        logger.info(f"Filtered to {len(filtered_studies)} usable studies")
    
    # Filter to studies with valid paths
    studies_with_paths = []
    for study in filtered_studies:
        path = get_study_path(study, source_paths)
        if path:
            studies_with_paths.append((study, path))
        else:
            logger.debug(f"No path found for {study.get('accession')}")
    
    logger.info(f"{len(studies_with_paths)} studies have valid paths")
    
    # Filter to unprocessed
    to_process = [(s, p) for s, p in studies_with_paths if s['accession'] not in processed]
    
    if max_studies:
        to_process = to_process[:max_studies]
    
    logger.info(f"Processing {len(to_process)} studies")
    
    results = list(processed.values())
    completed = 0
    failed = 0
    truncated_count = 0
    error_types = {}  # Track error types for summary
    start_time = datetime.now()
    
    # Process sequentially (safer for rate limiting, easier to debug)
    for study, study_path in to_process:
        acc = study['accession']
        matrix_info = matrix_by_acc.get(acc, {})
        
        try:
            result = analyzer.analyze_study(study, matrix_info, study_path)
            results.append(result)
            
            meta = result.get('_meta', {})
            if meta.get('success'):
                completed += 1
                if meta.get('truncated'):
                    truncated_count += 1
                    logger.info(f"✓ {acc} - completed (truncated but recovered)")
                else:
                    logger.info(f"✓ {acc} - completed")
            else:
                failed += 1
                error = meta.get('error', 'unknown')
                # Categorize error
                if 'JSON parse error' in error:
                    error_type = 'json_parse'
                elif 'rate' in error.lower():
                    error_type = 'rate_limit'
                elif 'timeout' in error.lower():
                    error_type = 'timeout'
                else:
                    error_type = 'other'
                error_types[error_type] = error_types.get(error_type, 0) + 1
                logger.warning(f"✗ {acc} - failed: {error[:100]}")
            
            # Progress logging every 5 studies
            total_done = completed + failed
            if total_done % 5 == 0:
                elapsed = (datetime.now() - start_time).total_seconds()
                rate = total_done / elapsed * 60 if elapsed > 0 else 0
                remaining = len(to_process) - total_done
                eta = remaining / (rate / 60) if rate > 0 else 0
                success_rate = 100 * completed / total_done if total_done > 0 else 0
                logger.info(f"Progress: {total_done}/{len(to_process)} "
                           f"({completed} success [{success_rate:.0f}%], {failed} failed) "
                           f"[{rate:.1f}/min, ETA: {eta/60:.1f}min]")
            
            # Save intermediate results every 10 studies
            if total_done % 10 == 0:
                save_results(results, output_path)
                
        except Exception as e:
            logger.error(f"{acc}: Unexpected error - {e}")
            import traceback
            logger.debug(traceback.format_exc())
            failed += 1
            error_types['exception'] = error_types.get('exception', 0) + 1
    
    # Final save
    save_results(results, output_path)
    
    # Summary
    elapsed = (datetime.now() - start_time).total_seconds()
    logger.info("=" * 60)
    logger.info("PROCESSING SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total processed: {completed + failed}")
    logger.info(f"Successful: {completed} ({100*completed/(completed+failed):.1f}%)" if completed + failed > 0 else "Successful: 0")
    logger.info(f"Failed: {failed}")
    if truncated_count > 0:
        logger.info(f"Truncated but recovered: {truncated_count}")
    if error_types:
        logger.info(f"Error breakdown: {error_types}")
    logger.info(f"Time: {elapsed:.1f}s ({elapsed/60:.1f}min)")
    logger.info(f"Rate: {(completed+failed)/elapsed*60:.1f} studies/min" if elapsed > 0 else "")
    
    # Check for debug files
    debug_dir = Path(debug_dir) if isinstance(debug_dir, str) else debug_dir
    if debug_dir.exists():
        debug_files = list(debug_dir.glob('*_failed.txt'))
        if debug_files:
            logger.info(f"Debug files saved: {len(debug_files)} failures in {debug_dir}/")
        success_files = list(debug_dir.glob('*_success.txt'))
        if success_files:
            logger.info(f"Debug files saved: {len(success_files)} successes in {debug_dir}/")
    
    return results


def save_results(results: List[Dict], output_path: Path):
    """Save results with summary statistics."""
    
    successful = [r for r in results if r.get('_meta', {}).get('success')]
    
    # Compute statistics
    stats = {
        'total_analyzed': len(results),
        'successful': len(successful),
        'failed': len(results) - len(successful),
    }
    
    if successful:
        # Count by topic
        topics = {}
        topic_categories = {}
        
        # Study type flags
        single_cell_count = 0
        time_series_count = 0
        disease_count = 0
        treatment_count = 0
        exercise_count = 0
        genetic_count = 0
        
        # Validation stats
        organism_correct = 0
        tissues_correct = 0
        
        # MoTrPAC utility
        rat_count = 0
        genecompass_useful = 0
        deconv_useful = 0
        grn_useful = 0
        motrpac_tissues = {}
        
        # Disease types
        disease_types = {}
        
        for r in successful:
            # Study overview
            overview = r.get('study_overview', {})
            topic = overview.get('primary_topic', 'unknown')
            topics[topic] = topics.get(topic, 0) + 1
            
            cat = overview.get('topic_category', 'other')
            topic_categories[cat] = topic_categories.get(cat, 0) + 1
            
            # Study type
            stype = r.get('study_type', {})
            if stype.get('is_single_cell'):
                single_cell_count += 1
            if stype.get('is_time_series'):
                time_series_count += 1
            if stype.get('is_disease_study'):
                disease_count += 1
            if stype.get('is_treatment_study'):
                treatment_count += 1
            if stype.get('is_genetic_perturbation'):
                genetic_count += 1
            
            # Treatments - check for exercise
            treatments = r.get('treatments', {})
            if treatments.get('has_exercise'):
                exercise_count += 1
            
            # Disease details
            disease = r.get('disease_condition', {})
            if disease.get('has_disease_model'):
                dtype = disease.get('disease_type', 'other')
                if dtype:
                    disease_types[dtype] = disease_types.get(dtype, 0) + 1
            
            # MoTrPAC utility
            util = r.get('utility_for_motrpac', {})
            if util.get('is_rat'):
                rat_count += 1
            if util.get('genecompass_useful'):
                genecompass_useful += 1
            if util.get('deconvolution_useful'):
                deconv_useful += 1
            if util.get('grn_useful'):
                grn_useful += 1
            
            for tissue in util.get('motrpac_tissues', []):
                if tissue:
                    motrpac_tissues[tissue] = motrpac_tissues.get(tissue, 0) + 1
            
            # Validation
            val = r.get('metadata_validation', {})
            if val.get('extracted_organism_correct'):
                organism_correct += 1
            if val.get('extracted_tissues_correct'):
                tissues_correct += 1
        
        # Store stats
        stats['by_topic'] = dict(sorted(topics.items(), key=lambda x: x[1], reverse=True))
        stats['by_topic_category'] = topic_categories
        
        stats['study_types'] = {
            'single_cell': single_cell_count,
            'time_series': time_series_count,
            'disease_study': disease_count,
            'treatment_study': treatment_count,
            'exercise_study': exercise_count,
            'genetic_perturbation': genetic_count,
        }
        
        stats['disease_types'] = disease_types
        
        stats['motrpac_utility'] = {
            'rat_studies': rat_count,
            'genecompass_useful': genecompass_useful,
            'deconvolution_useful': deconv_useful,
            'grn_useful': grn_useful,
            'tissues_covered': motrpac_tissues
        }
        
        stats['validation'] = {
            'organism_correct': organism_correct,
            'organism_correct_pct': round(100 * organism_correct / len(successful), 1) if successful else 0,
            'tissues_correct': tissues_correct,
            'tissues_correct_pct': round(100 * tissues_correct / len(successful), 1) if successful else 0,
        }
        
        # Tissue distribution
        tissues = {}
        for r in successful:
            for t in r.get('tissues', []):
                if isinstance(t, dict):
                    tname = t.get('category', 'other')
                else:
                    tname = str(t)
                tissues[tname] = tissues.get(tname, 0) + 1
        stats['by_tissue_category'] = tissues
    
    output = {
        'generated_at': datetime.now().isoformat(),
        'statistics': stats,
        'analyses': results
    }
    
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    logger.info(f"Saved {len(results)} analyses to {output_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Enhanced LLM-powered study analyzer')
    parser.add_argument('--config', '-c', required=True, help='Path to config.yaml')
    parser.add_argument('--output', '-o', help='Output JSON path')
    parser.add_argument('--max-studies', '-n', type=int, help='Max studies to process')
    
    # Resume/overwrite handling
    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument('--resume', action='store_true', default=True,
                              help='Resume from previous run, skip processed studies (DEFAULT)')
    resume_group.add_argument('--fresh', action='store_true',
                              help='Start fresh (will prompt before overwriting)')
    parser.add_argument('--force', action='store_true',
                        help='Force overwrite without prompting (use with --fresh)')
    
    parser.add_argument('--filter-usable', action='store_true', 
                        help='Only analyze usable mRNA studies')
    parser.add_argument('--filter-organism', help='Filter by organism (e.g., "rattus")')
    parser.add_argument('--model', default=None,
                        help='Claude model to use (default: from config or claude-haiku-4-5)')
    
    # Debug options
    parser.add_argument('--debug', action='store_true', 
                        help='Enable debug logging (shows JSON parsing details)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='More verbose output (show response sizes)')
    parser.add_argument('--save-responses', action='store_true',
                        help='Save all API responses (not just failures) to debug_responses/')
    parser.add_argument('--debug-dir', default='debug_responses',
                        help='Directory for debug files (default: debug_responses)')
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.verbose:
        logging.getLogger().setLevel(logging.INFO)
    
    if not HAS_YAML:
        raise ImportError("PyYAML required. Install: pip install pyyaml")
    
    if not HAS_ANTHROPIC:
        raise ImportError("anthropic package required. Install: pip install anthropic")
    
    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    catalog_dir = Path(config.get('catalog_dir', './catalog'))
    output_path = Path(args.output) if args.output else catalog_dir / 'llm_study_analysis_enhanced.json'
    
    # Determine resume behavior
    resume = not args.fresh  # Default is resume=True unless --fresh specified
    
    logger.info(f"Config: {args.config}")
    logger.info(f"Output: {output_path}")
    logger.info(f"Mode: {'RESUME (append)' if resume else 'FRESH (overwrite)'}")
    
    # Load data
    studies, matrix_by_acc, source_paths = load_data(config)
    
    # Get model - can be specified in config or command line (command line takes precedence)
    model = args.model or config.get('llm_model', 'claude-haiku-4-5')
    
    # Process
    results = process_studies(
        studies=studies,
        matrix_by_acc=matrix_by_acc,
        source_paths=source_paths,
        output_path=output_path,
        max_studies=args.max_studies,
        resume=resume,
        force_overwrite=args.force,
        filter_usable=args.filter_usable,
        filter_organism=args.filter_organism,
        model=model,
        debug_dir=args.debug_dir,
        save_all_responses=args.save_responses
    )
    
    # Print summary
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    
    successful = [r for r in results if r.get('_meta', {}).get('success')]
    failed = [r for r in results if not r.get('_meta', {}).get('success')]
    
    print(f"Total analyzed: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    
    if failed:
        # Show failure reasons
        errors = {}
        for r in failed:
            err = r.get('_meta', {}).get('error', 'unknown')
            err_type = err.split(':')[0] if ':' in err else err[:50]
            errors[err_type] = errors.get(err_type, 0) + 1
        print(f"\nFailure types:")
        for err, count in sorted(errors.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  {err}: {count}")
    
    if successful:
        # Count key categories
        rat_count = sum(1 for r in successful if r.get('utility_for_motrpac', {}).get('is_rat'))
        single_cell = sum(1 for r in successful if r.get('study_type', {}).get('is_single_cell'))
        disease = sum(1 for r in successful if r.get('study_type', {}).get('is_disease_study'))
        exercise = sum(1 for r in successful if r.get('treatments', {}).get('has_exercise'))
        
        print(f"\nStudy characteristics:")
        print(f"  Rat studies: {rat_count}")
        print(f"  Single-cell: {single_cell}")
        print(f"  Disease models: {disease}")
        print(f"  Exercise studies: {exercise}")
        
        # MoTrPAC utility
        gc_useful = sum(1 for r in successful if r.get('utility_for_motrpac', {}).get('genecompass_useful'))
        deconv_useful = sum(1 for r in successful if r.get('utility_for_motrpac', {}).get('deconvolution_useful'))
        grn_useful = sum(1 for r in successful if r.get('utility_for_motrpac', {}).get('grn_useful'))
        
        print(f"\nMoTrPAC utility:")
        print(f"  GeneCompass training: {gc_useful}")
        print(f"  Deconvolution reference: {deconv_useful}")
        print(f"  GRN inference: {grn_useful}")
        
        # Top topics
        topics = {}
        for r in successful:
            topic = r.get('study_overview', {}).get('primary_topic', 'unknown')
            topics[topic] = topics.get(topic, 0) + 1
        top_topics = sorted(topics.items(), key=lambda x: x[1], reverse=True)[:5]
        print(f"\nTop study topics:")
        for topic, count in top_topics:
            print(f"  {topic}: {count}")
        
        # Validation summary
        val_org = sum(1 for r in successful if r.get('metadata_validation', {}).get('extracted_organism_correct'))
        val_tis = sum(1 for r in successful if r.get('metadata_validation', {}).get('extracted_tissues_correct'))
        print(f"\nValidation:")
        print(f"  Organism correct: {val_org}/{len(successful)} ({100*val_org/len(successful):.1f}%)")
        print(f"  Tissues correct: {val_tis}/{len(successful)} ({100*val_tis/len(successful):.1f}%)")
    
    print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()