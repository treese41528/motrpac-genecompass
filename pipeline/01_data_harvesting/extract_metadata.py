#!/usr/bin/env python3
"""
extract_metadata.py - Extract and catalog metadata from harvested datasets

PORTABLE VERSION - Works with any directory structure!

Usage:
    # With config file:
    python extract_metadata.py --config config.yaml
    
    # With command line args:
    python extract_metadata.py --data-root /path/to/data --output-dir ./catalog
    
    # Auto-detect structure:
    python extract_metadata.py --data-root /path/to/data --auto-detect
    
    # Specify custom paths:
    python extract_metadata.py \\
        --add-source geo_sc:/path/to/geo/singlecell:single_cell:geo \\
        --add-source geo_bulk:/path/to/geo/bulk:bulk:geo \\
        --output-dir ./catalog
"""

import os
import sys
import json
import csv
import gzip
import re
import logging
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Any, Optional, Tuple
import argparse

# Try to import yaml, but make it optional
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# ============================================================================
# FILE FORMAT DEFINITIONS
# ============================================================================

FILE_FORMAT_INFO = {
    '.h5ad': {'category': 'single-cell', 'usability': 'high'},
    '.h5': {'category': 'single-cell', 'usability': 'high'},
    '.mtx': {'category': 'single-cell', 'usability': 'high'},
    '.mtx.gz': {'category': 'single-cell', 'usability': 'high'},
    '.loom': {'category': 'single-cell', 'usability': 'high'},
    '.rds': {'category': 'r-object', 'usability': 'medium'},
    '.tsv': {'category': 'tabular', 'usability': 'high'},
    '.csv': {'category': 'tabular', 'usability': 'high'},
    '.txt': {'category': 'tabular', 'usability': 'medium'},
    '.gz': {'category': 'compressed', 'usability': 'high'},
    '.tar': {'category': 'archive', 'usability': 'medium'},
    '.zip': {'category': 'archive', 'usability': 'medium'},
    '.soft': {'category': 'metadata', 'usability': 'high'},
    '.fastq': {'category': 'sequence', 'usability': 'low'},
    '.fq': {'category': 'sequence', 'usability': 'low'},
    '.bam': {'category': 'alignment', 'usability': 'medium'},
}

CONTENT_PATTERNS = {
    'count_matrix': [r'counts?[_\.]', r'raw[_\.]?counts?', r'umi', r'expression', r'matrix'],
    'normalized': [r'norm', r'tpm', r'fpkm', r'rpkm', r'cpm'],
    'metadata': [r'meta', r'sample[_\.]?info', r'annotation', r'phenotype', r'clinical'],
    'processed': [r'processed', r'filtered', r'qc', r'clustered'],
}

TECH_PATTERNS = {
    '10x_genomics': [r'10x', r'chromium', r'cell.?ranger'],
    'smart_seq': [r'smart[\-_]?seq'],
    'drop_seq': [r'drop[\-_]?seq'],
    'illumina': [r'illumina', r'hiseq', r'novaseq', r'nextseq'],
}

# ============================================================================
# ORGANISM & TISSUE CLASSIFICATION
# ============================================================================

# Known organisms - maps various names to canonical form
ORGANISM_ALIASES = {
    # Rat
    'rattus norvegicus': 'Rattus norvegicus', 'rat': 'Rattus norvegicus',
    'rats': 'Rattus norvegicus', 'r. norvegicus': 'Rattus norvegicus',
    'norway rat': 'Rattus norvegicus', 'brown rat': 'Rattus norvegicus',
    'sprague-dawley': 'Rattus norvegicus', 'sprague dawley': 'Rattus norvegicus',
    'wistar': 'Rattus norvegicus', 'fischer 344': 'Rattus norvegicus',
    # Mouse
    'mus musculus': 'Mus musculus', 'mouse': 'Mus musculus', 'mice': 'Mus musculus',
    'm. musculus': 'Mus musculus', 'c57bl/6': 'Mus musculus', 'c57bl6': 'Mus musculus',
    'balb/c': 'Mus musculus', 'cd-1': 'Mus musculus', 'nude mouse': 'Mus musculus',
    # Human
    'homo sapiens': 'Homo sapiens', 'human': 'Homo sapiens', 'h. sapiens': 'Homo sapiens',
    # Pig
    'sus scrofa': 'Sus scrofa', 'pig': 'Sus scrofa', 'swine': 'Sus scrofa', 'porcine': 'Sus scrofa',
    # Cow
    'bos taurus': 'Bos taurus', 'cow': 'Bos taurus', 'cattle': 'Bos taurus', 'bovine': 'Bos taurus',
    # Other mammals
    'ovis aries': 'Ovis aries', 'sheep': 'Ovis aries',
    'capra hircus': 'Capra hircus', 'goat': 'Capra hircus',
    'equus caballus': 'Equus caballus', 'horse': 'Equus caballus',
    'canis familiaris': 'Canis familiaris', 'dog': 'Canis familiaris', 'canine': 'Canis familiaris',
    'felis catus': 'Felis catus', 'cat': 'Felis catus',
    'oryctolagus cuniculus': 'Oryctolagus cuniculus', 'rabbit': 'Oryctolagus cuniculus',
    'cavia porcellus': 'Cavia porcellus', 'guinea pig': 'Cavia porcellus',
    'mesocricetus auratus': 'Mesocricetus auratus', 'hamster': 'Mesocricetus auratus',
    'heterocephalus glaber': 'Heterocephalus glaber', 'naked mole rat': 'Heterocephalus glaber',
    # Primates
    'macaca mulatta': 'Macaca mulatta', 'rhesus': 'Macaca mulatta', 'macaque': 'Macaca mulatta',
    'macaca fascicularis': 'Macaca fascicularis', 'cynomolgus': 'Macaca fascicularis',
    'pan troglodytes': 'Pan troglodytes', 'chimpanzee': 'Pan troglodytes',
    'callithrix jacchus': 'Callithrix jacchus', 'marmoset': 'Callithrix jacchus',
    # Fish
    'danio rerio': 'Danio rerio', 'zebrafish': 'Danio rerio',
    'oryzias latipes': 'Oryzias latipes', 'medaka': 'Oryzias latipes',
    # Birds
    'gallus gallus': 'Gallus gallus', 'chicken': 'Gallus gallus',
    # Amphibians
    'xenopus laevis': 'Xenopus laevis', 'xenopus': 'Xenopus laevis', 'frog': 'Xenopus laevis',
    'xenopus tropicalis': 'Xenopus tropicalis',
    # Invertebrates
    'drosophila melanogaster': 'Drosophila melanogaster', 'drosophila': 'Drosophila melanogaster',
    'fruit fly': 'Drosophila melanogaster', 'd. melanogaster': 'Drosophila melanogaster',
    'caenorhabditis elegans': 'Caenorhabditis elegans', 'c. elegans': 'Caenorhabditis elegans',
    'apis mellifera': 'Apis mellifera', 'honey bee': 'Apis mellifera',
    # Microorganisms
    'saccharomyces cerevisiae': 'Saccharomyces cerevisiae', 'yeast': 'Saccharomyces cerevisiae',
    'escherichia coli': 'Escherichia coli', 'e. coli': 'Escherichia coli',
    'arabidopsis thaliana': 'Arabidopsis thaliana', 'arabidopsis': 'Arabidopsis thaliana',
}

# Tissue keywords - maps keywords to canonical tissue name
# More specific terms should come first to avoid incorrect matches
TISSUE_KEYWORDS = {
    # Brain regions (more specific first)
    'hippocampus': ['hippocampus', 'hippocampal', 'dentate gyrus', 'ca1 ', 'ca3 '],
    'cerebellum': ['cerebellum', 'cerebellar', 'purkinje'],
    'hypothalamus': ['hypothalamus', 'hypothalamic', 'arcuate nucleus', 'paraventricular nucleus', 'dorsomedial nucleus'],
    'striatum': ['striatum', 'striatal', 'caudate', 'putamen', 'nucleus accumbens'],
    'cortex': ['cortex', 'cortical', 'prefrontal', 'frontal cortex', 'motor cortex', 'cerebral cortex', 'neocortex'],
    'substantia nigra': ['substantia nigra', 'nigral'],
    'amygdala': ['amygdala'],
    'thalamus': ['thalamus', 'thalamic'],
    'brainstem': ['brainstem', 'brain stem', 'pons', 'medulla oblongata'],
    'spinal cord': ['spinal cord', 'spinal'],
    'brain': ['brain', 'cerebral', 'cerebrum', 'cns', 'central nervous system'],
    # Peripheral nervous
    'retina': ['retina', 'retinal', 'photoreceptor', 'rpe'],
    'cochlea': ['cochlea', 'cochlear', 'inner ear', 'organ of corti'],
    'peripheral nerve': ['sciatic nerve', 'peripheral nerve', 'drg', 'dorsal root ganglion', 
                         'vagus nerve', 'trigeminal', 'ganglion', 'ganglia', 'stellate'],
    # Cardiovascular (more specific first)
    'heart': ['heart', 'cardiac', 'myocardium', 'cardiomyocyte', 'ventricle', 'ventricular',
              'atrium', 'atrial', 'cardiac myocyte', 'infarct'],
    'aorta': ['aorta', 'aortic'],
    'blood vessel': ['artery', 'arterial', 'vein', 'venous', 'endothelium', 'endothelial',
                     'vascular', 'carotid', 'coronary artery'],
    'blood': ['blood', 'pbmc', 'peripheral blood', 'whole blood', 'serum', 'plasma',
              'leukocyte', 'white blood cell', 'lymphocyte', 'monocyte', 'erythrocyte'],
    # Immune
    'bone marrow': ['bone marrow', 'marrow', 'hematopoietic'],
    'spleen': ['spleen', 'splenic'],
    'thymus': ['thymus', 'thymic', 'thymocyte'],
    'lymph node': ['lymph node', 'lymphatic'],
    # Digestive
    'liver': ['liver', 'hepatic', 'hepatocyte', 'hepato', 'kupffer'],
    'stomach': ['stomach', 'gastric'],
    'small intestine': ['small intestine', 'duodenum', 'jejunum', 'ileum', 'enterocyte'],
    'colon': ['colon', 'colonic', 'large intestine', 'cecum', 'colorectal', 'colonocyte'],
    'intestine': ['intestine', 'intestinal', 'gut', 'enteric', 'bowel'],
    'pancreas': ['pancreas', 'pancreatic', 'islet', 'beta cell', 'alpha cell', 'langerhans'],
    'esophagus': ['esophagus', 'esophageal'],
    'salivary gland': ['salivary', 'parotid'],
    # Respiratory
    'lung': ['lung', 'pulmonary', 'alveolar', 'bronchial', 'pneumocyte', 'airway', 'respiratory'],
    # Urinary
    'kidney': ['kidney', 'renal', 'nephron', 'glomerulus', 'podocyte', 'tubule', 'tubular'],
    'bladder': ['bladder', 'urothelium'],
    # Reproductive
    'testis': ['testis', 'testes', 'testicular', 'spermatocyte', 'sertoli', 'leydig', 'seminiferous'],
    'ovary': ['ovary', 'ovarian', 'oocyte', 'follicle', 'granulosa'],
    'uterus': ['uterus', 'uterine', 'endometrium', 'myometrium', 'cervix'],
    'placenta': ['placenta', 'placental', 'trophoblast', 'decidua'],
    'mammary gland': ['mammary', 'breast', 'lactating'],
    'prostate': ['prostate', 'prostatic'],
    # Musculoskeletal
    'skeletal muscle': ['skeletal muscle', 'muscle', 'myocyte', 'myoblast', 'myotube', 'myofiber',
                        'gastrocnemius', 'soleus', 'tibialis', 'quadriceps', 'plantaris',
                        'satellite cell', 'c2c12'],
    'smooth muscle': ['smooth muscle', 'vsmc'],
    'bone': ['bone', 'osteoblast', 'osteocyte', 'osteoclast', 'osseous', 'femur', 'tibia', 'calvaria'],
    'cartilage': ['cartilage', 'chondrocyte', 'articular'],
    'tendon': ['tendon', 'achilles'],
    'synovium': ['synovium', 'synovial', 'joint'],
    # Adipose
    'adipose tissue': ['adipose', 'adipocyte', 'fat', 'white adipose', 'brown adipose',
                       'subcutaneous', 'visceral', 'epididymal fat', 'inguinal fat', '3t3-l1'],
    # Skin
    'skin': ['skin', 'dermis', 'dermal', 'epidermis', 'epidermal', 'keratinocyte', 'cutaneous'],
    # Endocrine
    'thyroid': ['thyroid'],
    'adrenal gland': ['adrenal', 'adrenal gland'],
    'pituitary': ['pituitary', 'hypophysis'],
    # Eye
    'eye': ['eye', 'ocular', 'cornea', 'lens', 'iris'],
    # Gonads (general)
    'gonad': ['gonad', 'gonads', 'gonadal'],
    # Embryo
    'embryo': ['embryo', 'embryonic', 'fetal', 'fetus'],
}

# Cell types that strongly indicate a tissue
CELL_TYPE_TISSUE_MAP = {
    'neuron': 'brain', 'neuronal': 'brain', 'neural': 'brain',
    'astrocyte': 'brain', 'microglia': 'brain', 'oligodendrocyte': 'brain',
    'cardiomyocyte': 'heart', 'cardiac myocyte': 'heart',
    'hepatocyte': 'liver', 'kupffer cell': 'liver',
    'adipocyte': 'adipose tissue', 'preadipocyte': 'adipose tissue',
    'myoblast': 'skeletal muscle', 'myocyte': 'skeletal muscle', 'myotube': 'skeletal muscle',
    'osteoblast': 'bone', 'osteocyte': 'bone', 'osteoclast': 'bone',
    'chondrocyte': 'cartilage',
    'keratinocyte': 'skin', 'melanocyte': 'skin',
    'pneumocyte': 'lung', 'alveolar cell': 'lung',
    'podocyte': 'kidney', 'tubular cell': 'kidney',
    'enterocyte': 'intestine', 'colonocyte': 'colon',
    'spermatocyte': 'testis', 'sertoli cell': 'testis',
    'oocyte': 'ovary', 'granulosa cell': 'ovary',
    'islet cell': 'pancreas', 'beta cell': 'pancreas',
    'thymocyte': 'thymus',
    'splenocyte': 'spleen',
    'photoreceptor': 'retina', 'retinal ganglion': 'retina',
}

# Terms that should NOT be classified as either organism or tissue
EXCLUDED_TERMS = {
    '', ' ', '  ', 'n/a', 'na', 'none', 'unknown', 'not applicable', 'other',
    'cell', 'cells', 'cell line', 'primary cell', 'culture', 'cultured',
    'sample', 'samples', 'control', 'treated', 'untreated', 'normal',
    'wild type', 'wildtype', 'wt', 'knockout', 'ko', 'transgenic', 'mutant',
    'male', 'female', 'adult', 'young', 'old', 'aged', 'mixed sample',
    'pooled', 'replicate', 'biological replicate', 'technical replicate',
    'in vitro', 'in vivo', 'normal tissue', 't cell', 'b cell', 'nk cell',
    'macrophage', 'dendritic cell', 'fibroblast', 'stem cell', 'progenitor',
}


def is_organism(text: str) -> Tuple[bool, Optional[str]]:
    """
    Check if text represents an organism.
    Returns (is_organism, canonical_name or None).
    """
    if not text:
        return False, None
    
    normalized = text.lower().strip()
    
    # Check excluded terms
    if normalized in EXCLUDED_TERMS:
        return False, None
    
    # Direct lookup
    if normalized in ORGANISM_ALIASES:
        return True, ORGANISM_ALIASES[normalized]
    
    # Check if it looks like a tissue (then not an organism)
    for tissue, keywords in TISSUE_KEYWORDS.items():
        for kw in keywords:
            if kw in normalized:
                return False, None
    
    # Check for Latin binomial pattern (Genus species)
    parts = text.strip().split()
    if len(parts) >= 2:
        # Capitalized genus + lowercase species
        if parts[0][0].isupper() and parts[1][0].islower():
            # Looks like binomial nomenclature - assume organism
            return True, text.strip()
    
    return False, None


def is_tissue(text: str) -> Tuple[bool, Optional[str]]:
    """
    Check if text represents a tissue.
    Returns (is_tissue, canonical_name or None).
    """
    if not text:
        return False, None
    
    normalized = text.lower().strip()
    
    # Check excluded terms
    if normalized in EXCLUDED_TERMS:
        return False, None
    
    # Check if it's an organism (then not a tissue)
    if normalized in ORGANISM_ALIASES:
        return False, None
    
    # Check for Latin binomial pattern
    parts = text.strip().split()
    if len(parts) >= 2 and parts[0][0].isupper() and parts[1][0].islower():
        # Looks like binomial nomenclature - not a tissue
        return False, None
    
    # Check tissue keywords (order matters - more specific first)
    for tissue, keywords in TISSUE_KEYWORDS.items():
        for kw in keywords:
            if kw in normalized:
                return True, tissue
    
    # Check cell type mapping
    for cell_type, tissue in CELL_TYPE_TISSUE_MAP.items():
        if cell_type in normalized:
            return True, tissue
    
    return False, None


def cleanup_organisms_and_tissues(organisms: List[str], tissues: List[str]) -> Tuple[List[str], List[str]]:
    """
    Clean up and correct misclassified organisms and tissues.
    Returns (cleaned_organisms, cleaned_tissues).
    """
    clean_organisms = set()
    clean_tissues = set()
    
    # Process organisms
    for item in organisms:
        is_org, org_name = is_organism(item)
        is_tis, tis_name = is_tissue(item)
        
        if is_org and org_name:
            clean_organisms.add(org_name)
        elif is_tis and tis_name:
            # Misclassified - move to tissues
            clean_tissues.add(tis_name)
    
    # Process tissues
    for item in tissues:
        is_org, org_name = is_organism(item)
        is_tis, tis_name = is_tissue(item)
        
        if is_tis and tis_name:
            clean_tissues.add(tis_name)
        elif is_org and org_name:
            # Misclassified - move to organisms
            clean_organisms.add(org_name)
    
    return sorted(clean_organisms), sorted(clean_tissues)


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Configuration for extraction."""
    
    def __init__(self):
        self.data_root = None
        self.catalog_dir = './catalog'
        self.sources = []  # List of (label, path, data_type, source_name)
        self.max_files_per_study = 500
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'Config':
        """Load from YAML config file."""
        if not HAS_YAML:
            raise ImportError("PyYAML required for config files. Install with: pip install pyyaml")
        
        config = cls()
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        
        config.data_root = data.get('data_root') or os.environ.get('MOTRPAC_DATA_ROOT')
        config.catalog_dir = data.get('catalog_dir', './catalog')
        
        root = Path(config.data_root) if config.data_root else Path('.')
        
        # Parse standard sources
        for source_name, source_cfg in data.get('sources', {}).items():
            if not source_cfg.get('enabled', True):
                continue
            for dtype in ['single_cell', 'bulk']:
                type_cfg = source_cfg.get(dtype, {})
                if type_cfg.get('enabled', True) and type_cfg.get('path'):
                    path = type_cfg['path']
                    full = root / path if not path.startswith('/') else Path(path)
                    if full.exists():
                        config.sources.append((f"{source_name}_{dtype}", str(full), dtype, source_name))
        
        # Custom sources
        for custom in data.get('custom_sources', []):
            if Path(custom['path']).exists():
                config.sources.append((custom['name'], custom['path'], custom['type'], custom.get('source', custom['name'])))
        
        config.max_files_per_study = data.get('extraction', {}).get('max_files_per_study', 500)
        return config
    
    @classmethod  
    def from_args(cls, args) -> 'Config':
        """Create from command line arguments."""
        config = cls()
        config.data_root = args.data_root
        config.catalog_dir = args.output_dir or './catalog'
        
        if args.auto_detect and config.data_root:
            config.sources = auto_detect_sources(config.data_root)
        elif hasattr(args, 'add_source') and args.add_source:
            for spec in args.add_source:
                parts = spec.split(':')
                if len(parts) >= 3:
                    label, path, dtype = parts[0], parts[1], parts[2]
                    source = parts[3] if len(parts) > 3 else label.split('_')[0]
                    if Path(path).exists():
                        config.sources.append((label, path, dtype, source))
                    else:
                        logger.warning(f"Path not found: {path}")
        elif config.data_root:
            config.sources = get_default_sources(config.data_root)
        
        return config


def auto_detect_sources(data_root: str) -> List[Tuple[str, str, str, str]]:
    """Auto-detect data sources by scanning directory."""
    sources = []
    root = Path(data_root)
    
    logger.info(f"Auto-detecting sources in {data_root}...")
    
    # Common patterns
    patterns = [
        ('geo/single_cell/geo_datasets', 'single_cell', 'geo'),
        ('geo/singlecell/geo_datasets', 'single_cell', 'geo'),
        ('geo/singlecell', 'single_cell', 'geo'),
        ('geo/bulk/geo_datasets', 'bulk', 'geo'),
        ('geo/bulk', 'bulk', 'geo'),
        ('arrayexpress/singlecell/datasets', 'single_cell', 'arrayexpress'),
        ('arrayexpress/single_cell/datasets', 'single_cell', 'arrayexpress'),
        ('arrayexpress/singlecell', 'single_cell', 'arrayexpress'),
        ('arrayexpress/bulk/datasets', 'bulk', 'arrayexpress'),
        ('arrayexpress/bulk', 'bulk', 'arrayexpress'),
    ]
    
    found_types = set()
    for pattern, dtype, source in patterns:
        key = (source, dtype)
        if key in found_types:
            continue
        path = root / pattern
        if path.exists() and any(path.iterdir()):
            sources.append((f"{source}_{dtype}", str(path), dtype, source))
            found_types.add(key)
            logger.info(f"  Found {source} {dtype}: {path}")
    
    # Also check for GSE*/E-* style directories
    for subdir in root.iterdir():
        if not subdir.is_dir():
            continue
        for sub2 in subdir.iterdir():
            if not sub2.is_dir():
                continue
            children = [c.name for c in list(sub2.iterdir())[:20] if c.is_dir()]
            
            if any(c.startswith('GSE') for c in children):
                dtype = 'single_cell' if 'single' in str(sub2).lower() or 'sc' in str(sub2).lower() else 'bulk'
                key = ('geo', dtype)
                if key not in found_types:
                    sources.append((f"geo_{dtype}", str(sub2), dtype, 'geo'))
                    found_types.add(key)
                    logger.info(f"  Found GEO {dtype}: {sub2}")
            
            if any(c.startswith('E-') for c in children):
                dtype = 'single_cell' if 'single' in str(sub2).lower() or 'sc' in str(sub2).lower() else 'bulk'
                key = ('arrayexpress', dtype)
                if key not in found_types:
                    sources.append((f"arrayexpress_{dtype}", str(sub2), dtype, 'arrayexpress'))
                    found_types.add(key)
                    logger.info(f"  Found ArrayExpress {dtype}: {sub2}")
    
    return sources


def get_default_sources(data_root: str) -> List[Tuple[str, str, str, str]]:
    """Get default source paths."""
    sources = []
    root = Path(data_root)
    
    defaults = [
        ('geo_sc', 'geo/single_cell/geo_datasets', 'single_cell', 'geo'),
        ('geo_bulk', 'geo/bulk/geo_datasets', 'bulk', 'geo'),
        ('ae_sc', 'arrayexpress/singlecell/datasets', 'single_cell', 'arrayexpress'),
        ('ae_bulk', 'arrayexpress/bulk/datasets', 'bulk', 'arrayexpress'),
    ]
    
    for label, rel, dtype, source in defaults:
        path = root / rel
        if path.exists():
            sources.append((label, str(path), dtype, source))
            logger.info(f"Found {source} {dtype}: {path}")
    
    return sources


# ============================================================================
# EXTRACTION FUNCTIONS
# ============================================================================

def format_size(size_bytes: int) -> str:
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} PB"


def get_file_info(filepath: Path) -> Optional[Dict]:
    try:
        stat = filepath.stat()
    except (OSError, PermissionError):
        return None
    
    name = filepath.name
    suffixes = filepath.suffixes
    ext = ''.join(suffixes[-2:]) if len(suffixes) >= 2 and suffixes[-1] == '.gz' else (suffixes[-1] if suffixes else '')
    
    fmt = FILE_FORMAT_INFO.get(ext, FILE_FORMAT_INFO.get(suffixes[-1] if suffixes else '', {}))
    
    content = []
    name_lower = name.lower()
    for ctype, patterns in CONTENT_PATTERNS.items():
        if any(re.search(p, name_lower) for p in patterns):
            content.append(ctype)
    
    return {
        'filename': name,
        'size_bytes': stat.st_size,
        'size_human': format_size(stat.st_size),
        'extension': ext,
        'format_category': fmt.get('category', 'unknown'),
        'usability': fmt.get('usability', 'unknown'),
        'detected_content': content or ['unknown'],
    }


def parse_geo_soft(soft_path: Path) -> Dict:
    meta = {'title': None, 'summary': None, 'organism': [], 'tissues': [],
            'platform': [], 'samples': [], 'pubmed_ids': [], 'submission_date': None}
    try:
        opener = gzip.open if str(soft_path).endswith('.gz') else open
        with opener(soft_path, 'rt', errors='replace') as f:
            current_sample = {}
            for line in f:
                line = line.strip()
                if line.startswith('^SAMPLE'):
                    if current_sample:
                        meta['samples'].append(current_sample)
                    current_sample = {'id': line.split('=')[-1].strip()}
                elif line.startswith('!Series_title'):
                    meta['title'] = line.split('=', 1)[-1].strip()
                elif line.startswith('!Series_summary'):
                    meta['summary'] = (meta['summary'] or '') + ' ' + line.split('=', 1)[-1].strip()
                elif line.startswith('!Series_organism'):
                    meta['organism'].append(line.split('=', 1)[-1].strip())
                elif line.startswith('!Series_platform_id'):
                    meta['platform'].append(line.split('=', 1)[-1].strip())
                elif line.startswith('!Series_pubmed_id'):
                    meta['pubmed_ids'].append(line.split('=', 1)[-1].strip())
                elif line.startswith('!Series_submission_date'):
                    meta['submission_date'] = line.split('=', 1)[-1].strip()
                # Sample-level organism (common in GEO)
                elif line.startswith('!Sample_organism_ch1') or line.startswith('!Sample_organism'):
                    org = line.split('=', 1)[-1].strip()
                    if org:
                        meta['organism'].append(org)
                # Sample-level tissue/source info
                elif line.startswith('!Sample_source_name'):
                    tissue = line.split('=', 1)[-1].strip()
                    if tissue:
                        meta['tissues'].append(tissue)
                elif line.startswith('!Sample_characteristics_ch1'):
                    # Often contains "tissue: liver" or "cell type: neuron"
                    val = line.split('=', 1)[-1].strip()
                    if ':' in val:
                        key, value = val.split(':', 1)
                        key = key.strip().lower()
                        value = value.strip()
                        if key in ('tissue', 'cell type', 'organ', 'tissue type'):
                            meta['tissues'].append(value)
                elif line.startswith('!Sample_title') and current_sample:
                    current_sample['title'] = line.split('=', 1)[-1].strip()
                elif line.startswith('!Sample_source_name') and current_sample:
                    current_sample['source'] = line.split('=', 1)[-1].strip()
            if current_sample:
                meta['samples'].append(current_sample)
    except Exception as e:
        logger.debug(f"SOFT parse error: {e}")
    
    meta['organism'] = list(set(meta['organism']))
    meta['tissues'] = list(set(meta['tissues']))
    meta['summary'] = (meta['summary'] or '').strip()
    return meta


def parse_idf(idf_path: Path) -> Dict:
    """Parse ArrayExpress IDF (Investigation Description Format) file.
    
    IDF files contain study-level metadata including title and description.
    Format: tab-separated with field name in first column, values in subsequent columns.
    """
    result = {
        'title': None,
        'summary': None,
        'pubmed_ids': [],
        'submission_date': None,
    }
    
    try:
        # Handle gzipped files
        opener = gzip.open if str(idf_path).endswith('.gz') else open
        mode = 'rt' if str(idf_path).endswith('.gz') else 'r'
        
        with opener(idf_path, mode, errors='replace') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                parts = line.split('\t')
                if len(parts) < 2:
                    continue
                
                field = parts[0].lower().strip()
                value = parts[1].strip() if len(parts) > 1 else ''
                
                if not value:
                    continue
                
                # Title - can be "Investigation Title" or "Experiment Name"
                if field in ('investigation title', 'experiment name', 'study title'):
                    result['title'] = value
                
                # Description/Summary
                elif field in ('experiment description', 'investigation description', 
                              'study description'):
                    result['summary'] = value
                
                # PubMed IDs
                elif 'pubmed' in field and value.isdigit():
                    result['pubmed_ids'].append(value)
                
                # Submission date
                elif field in ('public release date', 'date of experiment', 'submission date'):
                    result['submission_date'] = value
                    
    except Exception as e:
        logger.debug(f"IDF parse error: {e}")
    
    return result


def parse_sdrf(sdrf_path: Path) -> Dict:
    """Parse ArrayExpress SDRF (Sample and Data Relationship Format) file."""
    samples, organisms, tissues = [], set(), set()
    try:
        # Handle gzipped files
        opener = gzip.open if str(sdrf_path).endswith('.gz') else open
        mode = 'rt' if str(sdrf_path).endswith('.gz') else 'r'
        
        with opener(sdrf_path, mode, errors='replace') as f:
            reader = csv.DictReader(f, delimiter='\t')
            headers = reader.fieldnames or []
            
            # Find organism columns - must contain 'organism' but NOT be a tissue-related term
            org_cols = [h for h in headers if 'organism' in h.lower()]
            
            # Find tissue columns - be more specific to avoid matching 'organism'
            # Match: tissue, organ part, cell type, body part, anatomical site
            # Don't match: organism
            tissue_cols = []
            for h in headers:
                h_lower = h.lower()
                if 'organism' in h_lower:
                    continue  # Skip organism columns
                if any(x in h_lower for x in [
                    'tissue', 'organ part', 'organ site', 'cell type', 
                    'body part', 'anatomical', 'sample source', 'material type'
                ]):
                    tissue_cols.append(h)
            
            name_cols = [h for h in headers if 'source name' in h.lower() or 'sample name' in h.lower()]
            
            for row in reader:
                sample = {}
                for c in name_cols:
                    if row.get(c):
                        sample['name'] = row[c]
                        break
                for c in org_cols:
                    if row.get(c):
                        organisms.add(row[c].strip())
                for c in tissue_cols:
                    if row.get(c):
                        tissues.add(row[c].strip())
                if sample.get('name'):
                    samples.append(sample)
    except Exception as e:
        logger.debug(f"SDRF parse error: {e}")
    
    return {'samples': samples, 'organisms': list(organisms), 'tissues': list(tissues)}


def extract_study(study_path: Path, source: str, data_type: str, max_files: int) -> Dict:
    accession = study_path.name
    
    info = {
        'accession': accession, 'source': source, 'data_type': data_type,
        'title': None, 'summary': None, 'organism': [], 'tissues': [],
        'sample_count': 0, 'platforms': [], 'technologies': [],
        'pubmed_ids': [], 'submission_date': None, 'files': [],
        'total_size_bytes': 0, 'has_processed_data': False,
        'has_count_matrix': False, 'has_metadata': False, 'usability_score': 0,
    }
    
    file_formats = defaultdict(int)
    file_count = 0
    total_size = 0
    
    for fp in study_path.rglob('*'):
        if not fp.is_file():
            continue
        file_count += 1
        
        fi = get_file_info(fp)
        if not fi:
            continue
        
        total_size += fi['size_bytes']
        file_formats[fi['extension']] += 1
        
        if file_count <= max_files:
            info['files'].append(fi)
        
        if fi['usability'] == 'high':
            info['has_processed_data'] = True
        if 'count_matrix' in fi['detected_content']:
            info['has_count_matrix'] = True
        if 'metadata' in fi['detected_content']:
            info['has_metadata'] = True
        
        # Parse metadata files
        fname = fp.name.lower()
        if fname.endswith('.soft') or fname.endswith('.soft.gz'):
            m = parse_geo_soft(fp)
            info['title'] = m.get('title') or info['title']
            info['summary'] = m.get('summary') or info['summary']
            info['organism'] = m.get('organism') or info['organism']
            info['tissues'].extend(m.get('tissues', []))  # Add tissues from SOFT
            info['platforms'] = m.get('platform', [])
            info['pubmed_ids'] = m.get('pubmed_ids', [])
            info['submission_date'] = m.get('submission_date')
            info['sample_count'] = len(m.get('samples', []))
        elif fname.endswith('.idf.txt') or fname.endswith('.idf.txt.gz'):
            # ArrayExpress IDF file - contains title and description
            m = parse_idf(fp)
            info['title'] = m.get('title') or info['title']
            info['summary'] = m.get('summary') or info['summary']
            info['pubmed_ids'].extend(m.get('pubmed_ids', []))
            info['submission_date'] = m.get('submission_date') or info['submission_date']
        elif fname.endswith('.sdrf.txt') or fname.endswith('.sdrf.txt.gz'):
            m = parse_sdrf(fp)
            info['sample_count'] = len(m.get('samples', []))
            info['organism'].extend(m.get('organisms', []))
            info['tissues'].extend(m.get('tissues', []))
    
    info['total_size_bytes'] = total_size
    info['total_size_human'] = format_size(total_size)
    info['file_count'] = file_count
    info['file_formats'] = dict(file_formats)
    info['organism'] = list(set(info['organism']))
    info['tissues'] = list(set(info['tissues']))
    
    # Usability score
    score = 0
    if info['has_processed_data']: score += 30
    if info['has_count_matrix']: score += 30
    if info['has_metadata']: score += 10
    if info['title']: score += 10
    if info['summary']: score += 10
    if info['sample_count'] > 0: score += 10
    info['usability_score'] = score
    
    # Technologies
    text = ' '.join([info['title'] or '', info['summary'] or '', ' '.join(info['platforms'])]).lower()
    for tech, patterns in TECH_PATTERNS.items():
        if any(re.search(p, text) for p in patterns):
            info['technologies'].append(tech)
    
    return info


def run_extraction(config: Config):
    output = Path(config.catalog_dir)
    output.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Output: {output}")
    logger.info(f"Sources: {len(config.sources)}")
    
    studies, files = [], []
    
    for label, path, dtype, source in config.sources:
        base = Path(path)
        if not base.exists():
            logger.warning(f"Path not found: {path}")
            continue
        
        dirs = [d for d in base.iterdir() if d.is_dir()]
        logger.info(f"Scanning {source} {dtype}: {len(dirs)} studies in {path}")
        
        for i, d in enumerate(dirs):
            if (i + 1) % 100 == 0:
                logger.info(f"  [{i+1}/{len(dirs)}]")
            try:
                s = extract_study(d, source, dtype, config.max_files_per_study)
                studies.append(s)
                for f in s['files']:
                    f['study_accession'] = s['accession']
                    f['source'] = source
                    f['data_type'] = dtype
                    files.append(f)
            except Exception as e:
                logger.error(f"Error on {d.name}: {e}")
    
    # Save outputs
    catalog = {
        'generated_at': datetime.now().isoformat(),
        'data_root': config.data_root,
        'sources': [{'label': l, 'path': p, 'type': t, 'source': s} for l, p, t, s in config.sources],
        'study_count': len(studies),
        'file_count': len(files),
        'studies': studies,
    }
    
    with open(output / 'master_catalog.json', 'w') as f:
        json.dump(catalog, f, indent=2)
    logger.info(f"Saved master_catalog.json")
    
    # CSV summary
    with open(output / 'study_summary.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, ['accession', 'source', 'data_type', 'title', 'organism', 
                               'sample_count', 'file_count', 'total_size_human', 'usability_score'])
        w.writeheader()
        for s in studies:
            row = {k: s.get(k) for k in w.fieldnames}
            row['organism'] = '; '.join(s.get('organism', []))
            w.writerow(row)
    logger.info(f"Saved study_summary.csv")
    
    # File inventory
    with open(output / 'file_inventory.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, ['study_accession', 'source', 'data_type', 'filename', 
                               'extension', 'size_human', 'format_category'])
        w.writeheader()
        for fi in files:
            w.writerow({k: fi.get(k) for k in w.fieldnames})
    logger.info(f"Saved file_inventory.csv")
    
    print(f"\n{'='*60}")
    print(f"COMPLETE: {len(studies)} studies, {len(files)} files")
    print(f"Output: {output}")
    print(f"{'='*60}")
    
    return output


def main():
    parser = argparse.ArgumentParser(description='Extract metadata from harvested datasets')
    parser.add_argument('--config', '-c', help='Path to config.yaml')
    parser.add_argument('--data-root', '-d', help='Root data directory')
    parser.add_argument('--output-dir', '-o', default=None, help='Output directory (overrides config)')
    parser.add_argument('--auto-detect', action='store_true', help='Auto-detect sources')
    parser.add_argument('--add-source', action='append', 
                        help='Add source: label:path:type[:source] (repeatable)')
    
    args = parser.parse_args()
    
    if args.config and os.path.exists(args.config):
        config = Config.from_yaml(args.config)
    else:
        config = Config.from_args(args)
    
    # Only override config if explicitly provided on command line
    if args.output_dir is not None:
        config.catalog_dir = args.output_dir
    
    # Fall back to default if still not set
    if not config.catalog_dir:
        config.catalog_dir = './catalog'
    
    if not config.sources:
        print("ERROR: No data sources found.")
        print("Use --data-root with --auto-detect, or --add-source, or --config")
        sys.exit(1)
    
    run_extraction(config)


if __name__ == '__main__':
    main()