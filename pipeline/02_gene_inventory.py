#!/usr/bin/env python3
"""
build_gene_inventory_v3.py - Build complete gene inventory from training corpus matrices

VERSION 3 IMPROVEMENTS:
- Filter studies by rat_ref_match threshold (>= 0.3 to be usable)
- Skip scATAC-seq files (fragments, peaks.bed)
- Detect bulk RNA-seq (< 100 samples) vs single-cell
- Search RAW subfolders for rat-specific files in multi-species studies
- Track exclusion reasons for each study
- Only include genes from validated rat studies

VERSION 3.1 FIXES:
- Removed stale/dangling resolution comment block in build_gene_inventory()
- Added bulk detection for binary formats (h5ad, h5, loom)
- Cleaned up duplicated gene_types assignment

VERSION 3.2 FIXES:
- Filter genomic coordinates and probe IDs from gene inventory
  (chr1:123-456 entries from scATAC studies were inflating gene counts)

VERSION 3.3 FIXES:
- Enforce GeneCompass core gene list: only protein-coding, lncRNA, miRNA
- Exclude pseudogenes, tRNAs, rRNAs, scaffolds, GenBank accessions
- Biotype-based filtering for resolved genes (via rat_gene_info.tsv + RGD)
- Pattern-based filtering for unresolved non-gene entries

Usage:
    python build_gene_inventory_v3.py --data-type single_cell -o gene_inventory_sc/
    python build_gene_inventory_v3.py --data-type bulk -o gene_inventory_bulk/
"""

import os
import sys
import gzip
import json
import pickle
import argparse
import logging
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple, Any
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# Optional imports
try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False
    h5py = None

# =============================================================================
# CONSTANTS
# =============================================================================

# Minimum rat_ref_match to consider a study usable
MIN_RAT_REF_MATCH = 0.3

# Maximum samples for single-cell (bulk has fewer)
MAX_BULK_SAMPLES = 100

# =============================================================================
# GENECOMPASS CORE GENE LIST FILTER
# =============================================================================

# GeneCompass keeps ONLY these biotypes (from paper Methods):
# "Informative gene annotations from Ensembl were used to define a core
#  gene list including protein-coding genes, lncRNAs, and miRNAs."
ALLOWED_BIOTYPES = {
    'protein_coding',
    'lncRNA', 'lincRNA',       # lincRNA is Ensembl's older name for lncRNA
    'miRNA',
}

# Biotypes explicitly excluded by GeneCompass:
# "pseudogenes, tRNAs, rRNAs, and other non-capturable loci"
EXCLUDED_BIOTYPES = {
    'pseudogene', 'processed_pseudogene', 'unprocessed_pseudogene',
    'transcribed_processed_pseudogene', 'transcribed_unprocessed_pseudogene',
    'translated_processed_pseudogene', 'translated_unprocessed_pseudogene',
    'polymorphic_pseudogene', 'unitary_pseudogene',
    'IG_pseudogene', 'IG_C_pseudogene', 'IG_J_pseudogene',
    'IG_V_pseudogene', 'TR_V_pseudogene', 'TR_J_pseudogene',
    'rRNA', 'rRNA_pseudogene',
    'Mt_rRNA', 'Mt_tRNA',
    'tRNA',
    'snRNA', 'snoRNA', 'scRNA', 'scaRNA',
    'misc_RNA',
    'ribozyme', 'vault_RNA', 'sRNA',
    'TEC',  # "To be Experimentally Confirmed"
}

# Pattern-based exclusion for genes that can't be resolved to Ensembl
# but are clearly not protein-coding/lncRNA/miRNA
NON_GENE_PATTERNS = [
    # Rfam non-coding RNA families (rRNA, tRNA, snRNA, etc.)
    re.compile(r'^RF\d{5}$'),
    # GenBank/EMBL accessions (not gene IDs)
    re.compile(r'^[A-Z]{2}\d{6}$'),       # e.g., AY172581, AC119762
    re.compile(r'^[A-Z]{2}\d{8}$'),       # e.g., JACYVU01... (scaffolds handled below)
    # Genome scaffold contigs
    re.compile(r'^JACYVU\d+$'),           # mRatBN7.2 scaffolds
    re.compile(r'^MU\d{6}$'),             # GenBank scaffold accessions
    # BAC clone contigs (not annotated genes)
    re.compile(r'^AABR\d+$'),             # e.g., AABR07005779
    # Explicit rRNA/tRNA names
    re.compile(r'^5S-rRNA$', re.IGNORECASE),
    re.compile(r'^5_8S-rRNA$', re.IGNORECASE),
    re.compile(r'^(mt-)?[Rr]nr[12]$'),    # mitochondrial rRNA (mt-Rnr1, mt-Rnr2)
    re.compile(r'^mt-[Tt][a-z]{1,2}\d*$'),  # mitochondrial tRNA (mt-Ta, mt-Tc, mt-Tl1)
]


BIOTYPE_LOOKUP = None

def load_biotype_reference(gene_info_path: Path = None, rgd_biotype_path: Path = None) -> Optional[dict]:
    """Load gene biotype lookup from rat_gene_info.tsv and optionally RGD.
    
    Returns dict: ensembl_id (uppercase) -> biotype string
    """
    global BIOTYPE_LOOKUP
    
    if BIOTYPE_LOOKUP is not None:
        return BIOTYPE_LOOKUP
    
    biotypes = {}
    
    # Primary: BioMart rat_gene_info.tsv
    if gene_info_path and gene_info_path.exists():
        try:
            with open(gene_info_path, 'r') as f:
                header = f.readline().strip().split('\t')
                # Find column indices
                ens_col = None
                bio_col = None
                for i, col in enumerate(header):
                    col_lower = col.lower().replace(' ', '_')
                    if 'gene_stable_id' in col_lower or col_lower == 'ensembl_gene_id' or col_lower == 'gene_id':
                        ens_col = i
                    elif 'biotype' in col_lower or 'gene_type' in col_lower:
                        bio_col = i
                
                if ens_col is not None and bio_col is not None:
                    for line in f:
                        parts = line.strip().split('\t')
                        if len(parts) > max(ens_col, bio_col):
                            ens_id = parts[ens_col].strip().upper()
                            biotype = parts[bio_col].strip()
                            if ens_id and biotype:
                                biotypes[ens_id] = biotype
            
            logger.info(f"Loaded biotypes from gene_info: {len(biotypes):,} genes")
        except Exception as e:
            logger.warning(f"Failed to load gene_info biotypes: {e}")
    
    # Fallback: RGD biotypes (if separate file exists)
    if rgd_biotype_path and rgd_biotype_path.exists():
        rgd_count = 0
        try:
            with open(rgd_biotype_path, 'r') as f:
                header = f.readline()
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        ens_id = parts[0].strip().upper()
                        biotype = parts[1].strip()
                        if ens_id and biotype and ens_id not in biotypes:
                            biotypes[ens_id] = biotype
                            rgd_count += 1
            if rgd_count > 0:
                logger.info(f"Added {rgd_count:,} biotypes from RGD fallback")
        except Exception as e:
            logger.warning(f"Failed to load RGD biotypes: {e}")
    
    if biotypes:
        BIOTYPE_LOOKUP = biotypes
        return BIOTYPE_LOOKUP
    
    return None


def is_allowed_biotype(biotype: str) -> bool:
    """Check if biotype is in the GeneCompass core gene list."""
    return biotype in ALLOWED_BIOTYPES


def is_non_gene_pattern(gene_id: str) -> Optional[str]:
    """Check if gene ID matches a known non-gene pattern.
    
    Returns the pattern name if matched, None otherwise.
    """
    for pattern in NON_GENE_PATTERNS:
        if pattern.match(gene_id):
            return pattern.pattern
    return None

# =============================================================================
# REFERENCE DATA (loaded once)
# =============================================================================

RAT_REFERENCE = None

def load_rat_reference(ref_path: Path = None) -> Optional[dict]:
    """Load rat gene reference for validation and symbol resolution.
    
    Supports both pickle (symbol_lookup) and TSV (legacy) formats.
    """
    global RAT_REFERENCE
    
    if RAT_REFERENCE is not None:
        return RAT_REFERENCE
    
    if ref_path is None:
        ref_path = Path('../../data/references/biomart/rat_symbol_lookup.pickle')
    
    if not ref_path.exists():
        logger.warning(f"Rat reference not found: {ref_path}")
        return None
    
    try:
        # Handle pickle format (new)
        if str(ref_path).endswith('.pickle'):
            with open(ref_path, 'rb') as f:
                symbol_lookup = pickle.load(f)
            
            rat_symbols = set(symbol_lookup.keys())
            rat_ensembl = {v.upper() for v in symbol_lookup.values() if v.startswith('ENSRNOG')}
            
            RAT_REFERENCE = {
                'ensembl': rat_ensembl,
                'symbols': rat_symbols,
                'all_lower': rat_ensembl | {s.lower() for s in rat_ensembl} | rat_symbols,
                'symbol_lookup': symbol_lookup,  # For resolution
            }
            
            logger.info(f"Loaded rat reference (pickle): {len(rat_ensembl):,} Ensembl IDs, {len(rat_symbols):,} symbols")
            return RAT_REFERENCE
        
        # Handle TSV format (legacy)
        rat_symbols = set()
        rat_ensembl = set()
        
        with open(ref_path, 'r') as f:
            header = f.readline()
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    ensembl_id = parts[0].strip()
                    symbol = parts[1].strip() if len(parts) > 1 else ''
                    
                    if ensembl_id:
                        rat_ensembl.add(ensembl_id.upper())
                    if symbol:
                        rat_symbols.add(symbol.lower())
        
        RAT_REFERENCE = {
            'ensembl': rat_ensembl,
            'symbols': rat_symbols,
            'all_lower': rat_ensembl | {s.lower() for s in rat_ensembl} | rat_symbols,
            'symbol_lookup': {s: None for s in rat_symbols},  # No resolution for TSV
        }
        
        logger.info(f"Loaded rat reference (TSV): {len(rat_ensembl):,} Ensembl IDs, {len(rat_symbols):,} symbols")
        return RAT_REFERENCE
        
    except Exception as e:
        logger.warning(f"Failed to load rat reference: {e}")
        return None

def resolve_to_ensembl(gene_id: str, ref: dict) -> Tuple[Optional[str], str]:
    """
    Resolve gene ID to rat Ensembl ID.
    
    Returns: (ensembl_id or None, status)
    Status: 'native_ensembl', 'resolved', 'mouse_contamination', 'unresolved'
    """
    if not gene_id or not ref:
        return None, 'empty'
    
    gene_clean = gene_id.strip()
    gene_upper = gene_clean.upper()
    gene_lower = gene_clean.lower()
    
    # Already Ensembl rat ID
    if gene_upper.startswith('ENSRNOG'):
        return gene_upper.split('.')[0], 'native_ensembl'
    
    # Mouse contamination patterns
    if gene_clean.startswith('Gm') and len(gene_clean) > 2 and gene_clean[2:].isdigit():
        return None, 'mouse_contamination'
    if gene_upper.startswith('ENSMUSG'):
        return None, 'mouse_contamination'
    
    # Try symbol lookup
    symbol_lookup = ref.get('symbol_lookup', {})
    ens_id = symbol_lookup.get(gene_lower)
    
    if ens_id and ens_id.upper().startswith('ENSRNOG'):
        return ens_id.upper(), 'resolved'
    
    return None, 'unresolved'


# =============================================================================
# GENE ID PATTERNS
# =============================================================================

GENE_ID_PATTERNS = {
    # Ensembl IDs - order matters for specificity
    'ensembl_rat': re.compile(r'^ENSRNOG\d{11}$', re.IGNORECASE),
    'ensembl_rat_versioned': re.compile(r'^ENSRNOG\d{11}\.\d+$', re.IGNORECASE),
    'ensembl_mouse': re.compile(r'^ENSMUSG\d{11}', re.IGNORECASE),
    'ensembl_human': re.compile(r'^ENSG\d{11}', re.IGNORECASE),
    'ensembl_human_prefixed': re.compile(r'^GRCh38[_]{2,}ENSG', re.IGNORECASE),
    
    # Multi-species prefixed patterns (from comparative studies)
    'prefixed_human': re.compile(r'^hum_ENSG', re.IGNORECASE),
    'prefixed_mouse': re.compile(r'^mou_ENSMUSG', re.IGNORECASE),
    'prefixed_dog': re.compile(r'^dog_ENSCAFG', re.IGNORECASE),
    'prefixed_cat': re.compile(r'^ENSFCAG', re.IGNORECASE),
    'prefixed_hamster': re.compile(r'^ham_ENSCGRG', re.IGNORECASE),
    'prefixed_monkey': re.compile(r'^mky_ENSCSAG', re.IGNORECASE),
    'prefixed_fly': re.compile(r'^fly_FBgn', re.IGNORECASE),
    
    # RefSeq
    'refseq_mrna': re.compile(r'^[NX]M_\d+(\.\d+)?$'),
    'refseq_ncrna': re.compile(r'^[NX]R_\d+(\.\d+)?$'),
    
    # Entrez (numeric, 4-10 digits)
    'entrez': re.compile(r'^\d{4,10}$'),
    
    # Rat-specific patterns
    'aabr_clone': re.compile(r'^AABR\d+\.\d+$', re.IGNORECASE),
    'rat_rnor': re.compile(r'^Rnor', re.IGNORECASE),
    'rat_prefixed_ensembl': re.compile(r'^rat_ENSRNOG\d+', re.IGNORECASE),
    'rat_transcript': re.compile(r'^ENSRNOT\d{11}', re.IGNORECASE),
    'rgd_id': re.compile(r'^RGD\d+$', re.IGNORECASE),
    'rat_mhc': re.compile(r'^RT1-', re.IGNORECASE),
    'ncbi_loc': re.compile(r'^LOC\d+$'),
    
    # miRNA
    'mirna_rat': re.compile(r'^rno-(?:mir|miR|let)-', re.IGNORECASE),
    'mirna_mouse': re.compile(r'^mmu-(?:mir|miR|let)-', re.IGNORECASE),
    'mirna_human': re.compile(r'^hsa-(?:mir|miR|let)-', re.IGNORECASE),
    
    # Gene symbols (case-sensitive patterns)
    'gene_symbol_rat': re.compile(r'^[A-Z][a-z0-9]{1,15}$'),
    
    # Probes
    'probe_affy': re.compile(r'^\d+_(?:at|s_at|x_at|a_at)$'),
    'probe_illumina': re.compile(r'^ILMN_\d+$'),
    
    # Genomic coordinates (NOT genes - filter out)
    'genomic_coord': re.compile(r'^(\d+|[XYM]):(\d+)-(\d+)$'),
    'genomic_coord_chr': re.compile(r'^chr(\d+|[XYM]):(\d+)-(\d+)$', re.IGNORECASE),
}



# =============================================================================
# MATRIX FILE DISCOVERY
# =============================================================================

MATRIX_EXTENSIONS = {'.mtx', '.mtx.gz', '.h5ad', '.h5', '.hdf5', '.loom'}

SKIP_FILE_PATTERNS = {
    '_samples.tsv', '_sra_runs.tsv', '_gsm_files.tsv',
    '_samples.', 'metadata', 'phenotype', 'clinical', 'annotation',
    'barcode', 'barcodes',
    'clusters', 'umap', 'tsne', 'pca',
    '.md5', 'readme', 'license',
    # scATAC-seq patterns
    'fragments', 'peaks.bed', 'singlecell.csv',
}

MATRIX_FILE_PATTERNS = {
    'matrix', 'counts', 'count', 'expression', 'umi',
    'filtered_feature_bc', 'raw_feature_bc',
    'fpkm', 'tpm', 'rpkm', 'cpm',
}

# =============================================================================
# Technology detection patterns (in file content or filenames)
# =============================================================================
WRONG_TECHNOLOGY_PATTERNS = {
    'microarray_agilent': [
        r'Agilent Technologies',
        r'FEPARAMS.*Protocol_Name',
        r'FeatureExtractor',
        r'Grid_NumSubGridRows',
    ],
    'microarray_affymetrix': [
        r'\.CEL$',
        r'\.CDF$', 
        r'Affymetrix',
    ],
    'microarray_illumina': [
        r'HumanHT-12',
        r'MouseRef-8',
        r'RatRef-12',
        r'ILMN_\d+',
    ],
    'protein_array': [
        r'IgG-AutoAb',
        r'Plasma.*serum',
        r'mAb \d+',
    ],
    'chipseq': [
        r'\.bed\.gz$',
        r'\.wig\.gz$',
        r'\.bw$',
        r'\.bigwig$',
        r'peaks\.narrowPeak',
    ],
    'atacseq': [
        r'fragments\.tsv',
        r'peaks\.bed',
        r'cellranger-atac',
    ],
}

# =============================================================================
# SPECIES-AWARE FILE SELECTION
# =============================================================================


# File patterns indicating specific species (for multi-species studies)
HUMAN_FILE_PATTERNS = [
    re.compile(r'human', re.IGNORECASE),
    re.compile(r'hg38', re.IGNORECASE),
    re.compile(r'hg19', re.IGNORECASE),
    re.compile(r'GRCh38', re.IGNORECASE),
    re.compile(r'homo_sapiens', re.IGNORECASE),
    re.compile(r'_hum_', re.IGNORECASE),
]

MOUSE_FILE_PATTERNS = [
    re.compile(r'mouse', re.IGNORECASE),
    re.compile(r'mm10', re.IGNORECASE),
    re.compile(r'mm39', re.IGNORECASE),
    re.compile(r'GRCm38', re.IGNORECASE),
    re.compile(r'mus_musculus', re.IGNORECASE),
    re.compile(r'_mou_', re.IGNORECASE),
]

RAT_FILE_PATTERNS = [
    re.compile(r'[_\.]rat[_\.]', re.IGNORECASE),
    re.compile(r'rattus', re.IGNORECASE),
    re.compile(r'rn6', re.IGNORECASE),
    re.compile(r'rn7', re.IGNORECASE),
    re.compile(r'rnor', re.IGNORECASE),
    re.compile(r'_rat_', re.IGNORECASE),
    re.compile(r'WT_rat', re.IGNORECASE),
]

OTHER_SPECIES_PATTERNS = [
    re.compile(r'dog', re.IGNORECASE),
    re.compile(r'canine', re.IGNORECASE),
    re.compile(r'cat[_\.]', re.IGNORECASE),
    re.compile(r'feline', re.IGNORECASE),
    re.compile(r'pig[_\.]', re.IGNORECASE),
    re.compile(r'porcine', re.IGNORECASE),
    re.compile(r'horse', re.IGNORECASE),
    re.compile(r'equine', re.IGNORECASE),
    re.compile(r'zebrafish', re.IGNORECASE),
    re.compile(r'drosophila', re.IGNORECASE),
    re.compile(r'chicken', re.IGNORECASE),
    re.compile(r'monkey', re.IGNORECASE),
    re.compile(r'macaque', re.IGNORECASE),
]

SPECIES_PATTERNS = {
    'human': [
        r'^ENSG\d{11}',           # Ensembl human
        r'^GRCh38[_:]+',          # GRCh38 prefixed
        r'^hg19[_:]+|^hg38[_:]+', # UCSC human
        r'^hum_ENSG',             # Prefixed human
    ],
    'mouse': [
        r'^ENSMUSG\d{11}',        # Ensembl mouse
        r'^mm10[_:]+|^mm39[_:]+', # UCSC mouse
        r'^mou_ENSMUSG',          # Prefixed mouse
        r'^MGI:\d+',              # MGI IDs
    ],
    'dog': [
        r'^ENSCAFG\d{11}',        # Ensembl dog
        r'^dog_ENSCAFG',          # Prefixed dog
    ],
    'other_nonrat': [
        r'^ENSGALG',              # Chicken
        r'^ENSBTAG',              # Cow  
        r'^ENSSSCG',              # Pig
        r'^ENSECAG',              # Horse
        r'^FBgn\d+',              # Drosophila
        r'^WBGene\d+',            # C. elegans
        r'^ENSDARG',              # Zebrafish
    ],
}



# Patterns to SKIP (artifacts, headers)
SKIP_PATTERNS = [
    re.compile(r'^(Gene|GeneId|Ensembl|ID|Name|Symbol|tracking_id|RefSeq|Accession)', re.IGNORECASE),
    re.compile(r'^(TYPE|FEPARAMS|DATA|STATS|FEATURES|\*)$'),
    re.compile(r'^#'),
    re.compile(r'^\d+$'),  # Pure numeric (row indices)
    re.compile(r'^(\d+|[XYM]):\d+-\d+$'),  # Genomic coordinates without chr
]

# File patterns to SKIP entirely (wrong data modality)
SKIP_FILE_PATTERNS_ATAC = [
    re.compile(r'fragments\.tsv', re.IGNORECASE),
    re.compile(r'peaks\.bed', re.IGNORECASE),
    re.compile(r'_atac_', re.IGNORECASE),
    re.compile(r'atacseq', re.IGNORECASE),
]


def clean_gene_id(gene_id: str) -> str:
    """Clean gene ID: strip quotes, whitespace."""
    if not gene_id:
        return ''
    cleaned = gene_id.strip().strip('"').strip("'").strip()
    return cleaned


def is_artifact(gene_id: str) -> bool:
    """Check if gene ID is an artifact to skip."""
    if not gene_id:
        return True
    for pattern in SKIP_PATTERNS:
        if pattern.match(gene_id):
            return True
    return False


def is_atac_file(filepath: Path) -> bool:
    """Check if file is scATAC-seq (not gene expression)."""
    fname = filepath.name
    for pattern in SKIP_FILE_PATTERNS_ATAC:
        if pattern.search(fname):
            return True
    return False


def classify_gene_id(gene_id: str) -> str:
    """Classify a single gene ID."""
    if not gene_id or not gene_id.strip():
        return 'empty'
    
    gene_id = clean_gene_id(gene_id)
    
    if is_artifact(gene_id):
        return 'artifact'
    
    for id_type, pattern in GENE_ID_PATTERNS.items():
        if pattern.match(gene_id):
            return id_type
    
    return 'unknown'


def get_dominant_gene_type(gene_ids: List[str]) -> Tuple[str, float]:
    """Get dominant gene ID type from a list."""
    if not gene_ids:
        return 'empty', 0.0
    
    types = Counter()
    for g in gene_ids:
        g_clean = clean_gene_id(g)
        if g_clean:
            gtype = classify_gene_id(g_clean)
            if gtype not in ('empty', 'artifact'):
                types[gtype] += 1
    
    if not types:
        return 'unknown', 0.0
    
    dominant = max(types, key=types.get)
    confidence = types[dominant] / sum(types.values())
    
    return dominant, confidence


def is_valid_rat_gene_type(gene_type: str) -> bool:
    """Check if gene type is valid for rat training data."""
    valid_types = {
        'ensembl_rat', 'ensembl_rat_versioned',
        'aabr_clone', 'rat_rnor', 'rat_prefixed_ensembl', 'rat_transcript',
        'rgd_id', 'rat_mhc', 'ncbi_loc',
        'mirna_rat',
        'gene_symbol_rat',
        'refseq_mrna', 'refseq_ncrna',
        'entrez',
    }
    return gene_type in valid_types


def is_contaminated_species(gene_type: str) -> bool:
    """Check if gene type indicates species contamination."""
    contaminated = {
        'ensembl_mouse', 'ensembl_human', 'ensembl_human_prefixed',
        'mirna_mouse', 'mirna_human',
        'prefixed_human', 'prefixed_mouse', 'prefixed_dog', 
        'prefixed_cat', 'prefixed_hamster', 'prefixed_monkey', 'prefixed_fly',
    }
    return gene_type in contaminated


def validate_genes_against_reference(gene_ids: List[str], ref: dict, sample_size: int = 1000) -> float:
    """Validate gene IDs against rat reference. Returns fraction matching."""
    if not ref or not gene_ids:
        return 0.0
    
    sample = gene_ids[:sample_size]
    
    matches = 0
    checked = 0
    
    all_ref = ref.get('all_lower', set())
    
    for g in sample:
        g_clean = clean_gene_id(g).lower()
        if g_clean and not is_artifact(g_clean):
            checked += 1
            g_base = g_clean.split('.')[0]
            if g_base in all_ref or g_clean in all_ref:
                matches += 1
    
    return matches / checked if checked > 0 else 0.0



def detect_species_from_genes(gene_ids: List[str], sample_size: int = 500) -> Dict[str, Any]:
    """Detect species from gene IDs.
    
    Returns dict with:
        - dominant_species: most common species or 'rat' or 'unknown'
        - species_counts: {species: count}
        - is_contaminated: True if multiple species detected
        - rat_fraction: fraction of genes matching rat patterns
    """
    import re
    
    # Sample genes for efficiency
    sample = gene_ids[:sample_size] if len(gene_ids) > sample_size else gene_ids
    
    species_counts = {sp: 0 for sp in SPECIES_PATTERNS.keys()}
    species_counts['rat'] = 0
    species_counts['unknown'] = 0
    
    # Rat patterns (from existing code)
    rat_patterns = [
        r'^ENSRNOG\d{11}',
        r'^rat_ENSRNOG',
        r'^ENSRNOT\d{11}',
        r'^RGD\d+',
        r'^LOC\d+',
        r'^RT1-',
        r'^AABR\d+',
        r'^NEWGENE_\d+',
    ]
    
    for gene in sample:
        gene_upper = gene.upper()
        matched = False
        
        # Check rat first
        for pattern in rat_patterns:
            if re.match(pattern, gene, re.IGNORECASE):
                species_counts['rat'] += 1
                matched = True
                break
        
        if not matched:
            # Check other species
            for species, patterns in SPECIES_PATTERNS.items():
                for pattern in patterns:
                    if re.match(pattern, gene, re.IGNORECASE):
                        species_counts[species] += 1
                        matched = True
                        break
                if matched:
                    break
        
        if not matched:
            species_counts['unknown'] += 1
    
    total = len(sample)
    
    # Determine dominant species
    max_species = max(species_counts.items(), key=lambda x: x[1])
    dominant_species = max_species[0] if max_species[1] > total * 0.3 else 'unknown'
    
    # Check for contamination (multiple species with >10% each)
    significant_species = [sp for sp, count in species_counts.items() 
                          if count > total * 0.1 and sp != 'unknown']
    is_contaminated = len(significant_species) > 1
    
    return {
        'dominant_species': dominant_species,
        'species_counts': species_counts,
        'is_contaminated': is_contaminated,
        'rat_fraction': species_counts['rat'] / total if total > 0 else 0,
        'nonrat_fraction': sum(species_counts[sp] for sp in SPECIES_PATTERNS.keys()) / total if total > 0 else 0,
    }
    
    

def detect_wrong_technology(filepath: Path, content_sample: str = None) -> Dict[str, Any]:
    """Detect if file is from wrong technology (microarray, protein array, etc.).
    
    Returns dict with:
        - is_wrong_tech: True if wrong technology detected
        - detected_tech: name of detected technology or None
        - evidence: matching pattern
    """
    import re
    
    fname = filepath.name.lower()
    
    # Check filename patterns
    for tech, patterns in WRONG_TECHNOLOGY_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, fname, re.IGNORECASE):
                return {
                    'is_wrong_tech': True,
                    'detected_tech': tech,
                    'evidence': f'filename: {pattern}',
                }
    
    # Check content if provided
    if content_sample:
        for tech, patterns in WRONG_TECHNOLOGY_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, content_sample, re.IGNORECASE):
                    return {
                        'is_wrong_tech': True,
                        'detected_tech': tech,
                        'evidence': f'content: {pattern}',
                    }
    
    return {
        'is_wrong_tech': False,
        'detected_tech': None,
        'evidence': None,
    }


def score_file_for_rat(filepath: Path) -> int:
    """
    Score a file's likelihood of containing rat data.
    Higher = better for rat.
    Negative = definitely not rat (skip).
    """
    fname = filepath.name
    
    # Check for scATAC-seq files (skip entirely)
    if is_atac_file(filepath):
        return -200
    
    # Check for human patterns (strong negative)
    for pattern in HUMAN_FILE_PATTERNS:
        if pattern.search(fname):
            return -100
    
    # Check for mouse patterns (negative)
    for pattern in MOUSE_FILE_PATTERNS:
        if pattern.search(fname):
            return -50
    
    # Check for other species
    for pattern in OTHER_SPECIES_PATTERNS:
        if pattern.search(fname):
            return -75
    
    # Check for rat patterns (positive)
    score = 0
    for pattern in RAT_FILE_PATTERNS:
        if pattern.search(fname):
            score += 10
    
    fname_lower = fname.lower()
    if 'filtered' in fname_lower:
        score += 5
    if 'wt_rat' in fname_lower:
        score += 20
    if 'feature' in fname_lower or 'gene' in fname_lower:
        score += 2
    
    return score


# =============================================================================
# BULK vs SINGLE-CELL DETECTION
# =============================================================================

def detect_if_bulk(filepath: Path, gene_ids: List[str] = None) -> Tuple[bool, int]:
    """
    Detect if a file is bulk RNA-seq based on column count.
    
    Returns:
        (is_bulk, n_samples)
    """
    fname = filepath.name.lower()
    
    # Check file to count columns
    try:
        opener = gzip.open if fname.endswith('.gz') else open
        with opener(filepath, 'rt', errors='replace') as f:
            # Read header line
            header = f.readline().strip()
            
            # Detect delimiter
            if '\t' in header:
                n_cols = len(header.split('\t'))
            elif ',' in header:
                n_cols = len(header.split(','))
            else:
                return False, 0
            
            # First column is usually gene ID, rest are samples
            n_samples = n_cols - 1
            
            # Bulk typically has < 100 samples
            is_bulk = n_samples < MAX_BULK_SAMPLES and n_samples > 0
            
            return is_bulk, n_samples
    except:
        return False, 0


def detect_if_bulk_binary(filepath: Path) -> Tuple[bool, int]:
    """
    Detect if an h5ad/h5/loom file is bulk RNA-seq based on observation count.
    
    Returns:
        (is_bulk, n_samples)
    """
    if not HAS_H5PY:
        return False, 0
    
    fname = filepath.name.lower()
    
    try:
        with h5py.File(filepath, 'r') as f:
            n_obs = 0
            
            if fname.endswith('.h5ad'):
                # AnnData: obs group has the cell/sample dimension
                if 'obs' in f:
                    # Check for _index or index key for observation count
                    for key in ['_index', 'index']:
                        if key in f['obs']:
                            n_obs = len(f['obs'][key])
                            break
                    # Fallback: check X shape
                    if n_obs == 0 and 'X' in f:
                        shape = f['X'].shape if hasattr(f['X'], 'shape') else None
                        if shape:
                            n_obs = shape[0]
                        elif 'shape' in f['X'].attrs:
                            n_obs = f['X'].attrs['shape'][0]
            
            elif fname.endswith(('.h5', '.hdf5')):
                # 10x HDF5: matrix/shape or matrix/barcodes
                if 'matrix' in f:
                    if 'barcodes' in f['matrix']:
                        n_obs = len(f['matrix']['barcodes'])
                    elif 'shape' in f['matrix']:
                        n_obs = f['matrix']['shape'][1]  # CSC: (genes, cells)
            
            elif fname.endswith('.loom'):
                # Loom: columns are cells/samples
                if 'matrix' in f:
                    n_obs = f['matrix'].shape[1]
                elif 'col_attrs' in f:
                    # Count entries in any column attribute
                    for key in f['col_attrs']:
                        n_obs = len(f['col_attrs'][key])
                        break
            
            if n_obs > 0:
                is_bulk = n_obs < MAX_BULK_SAMPLES
                return is_bulk, n_obs
    
    except Exception:
        pass
    
    return False, 0


# =============================================================================
# GENE EXTRACTION FROM DIFFERENT FORMATS
# =============================================================================

def extract_genes_from_tsv(filepath: Path, max_genes: int = 200000) -> Tuple[List[str], Dict[str, Any]]:
    """Extract gene IDs from TSV/CSV format.
    
    Handles:
    - Standard: gene_id in column 1
    - Row-indexed: "1" "Gad1" (gene in column 2)
    - Transposed: barcodes in row 1, genes in column 1 from row 2
    
    Also detects wrong technology and species contamination.
    """
    metadata = {'format': 'tsv', 'n_genes': 0, 'source_file': str(filepath)}
    gene_ids = []
    
    try:
        fsize = filepath.stat().st_size
        estimated_size = fsize * 5 if str(filepath).endswith('.gz') else fsize
        is_large_file = estimated_size > 200 * 1024 * 1024
        if is_large_file:
            metadata['large_file'] = True
            metadata['file_size_mb'] = fsize / (1024*1024)
    except:
        is_large_file = False
    
    try:
        opener = gzip.open if str(filepath).endswith('.gz') else open
        
        with opener(filepath, 'rt', errors='replace') as f:
            # Read first few lines to detect format
            first_lines = []
            content_sample = ""
            for i, line in enumerate(f):
                first_lines.append(line.strip())
                content_sample += line
                if i >= 20:
                    break
            
            if not first_lines:
                metadata['error'] = 'empty_file'
                return gene_ids, metadata
            
            # Check for wrong technology
            tech_check = detect_wrong_technology(filepath, content_sample)
            if tech_check['is_wrong_tech']:
                metadata['error'] = f"wrong_technology:{tech_check['detected_tech']}"
                metadata['tech_evidence'] = tech_check['evidence']
                return gene_ids, metadata
            
            # Detect delimiter
            header = first_lines[0]
            if '\t' in header:
                delimiter = '\t'
            elif ',' in header:
                delimiter = ','
            else:
                delimiter = None
            
            # Check for transposed matrix (barcodes in first row)
            first_row_parts = header.replace('"', '').split(delimiter) if delimiter else header.split()
            is_transposed = False
            if len(first_row_parts) > 10:
                barcode_like = sum(1 for p in first_row_parts[:20] 
                                  if re.match(r'^[ACGT]{8,}-\d+', p) or 
                                     re.match(r'.*_r\d+$', p, re.IGNORECASE) or
                                     'barcode' in p.lower())
                if barcode_like >= 5:
                    is_transposed = True
                    metadata['transposed'] = True
            
            # For large transposed files, use fast first-column-only parsing
            if is_large_file and is_transposed:
                f.seek(0)
                f.readline()  # Skip header (barcodes)
                for i, line in enumerate(f):
                    if i >= max_genes:
                        metadata['truncated'] = True
                        break
                    # Only parse first column - much faster for wide files
                    if delimiter:
                        idx = line.find(delimiter)
                        if idx > 0:
                            val = line[:idx].strip().replace('"', '').replace("'", "")
                        else:
                            continue
                    else:
                        val = line.split()[0].strip() if line.strip() else ''
                    
                    gene_id = clean_gene_id(val)
                    if gene_id and not is_artifact(gene_id):
                        gene_ids.append(gene_id)
                
                metadata['n_genes'] = len(gene_ids)
                
                # Detect species from extracted genes
                if gene_ids:
                    species_info = detect_species_from_genes(gene_ids)
                    metadata['species_detection'] = species_info
                    if species_info['dominant_species'] not in ['rat', 'unknown']:
                        metadata['wrong_species'] = species_info['dominant_species']
                    if species_info['is_contaminated']:
                        metadata['species_contaminated'] = True
                
                return gene_ids, metadata
            
            # For large non-transposed files, skip entirely
            if is_large_file and not is_transposed:
                metadata['error'] = 'file_too_large'
                return gene_ids, metadata
            
            # Detect if column 1 is row indices (numeric or quoted numeric)
            col1_numeric = 0
            col2_has_genes = 0
            for line in first_lines[1:]:
                if not line:
                    continue
                clean_line = line.replace('"', '').replace("'", "")
                if delimiter:
                    parts = clean_line.split(delimiter)
                else:
                    parts = clean_line.split()
                if len(parts) >= 2:
                    col1, col2 = parts[0].strip(), parts[1].strip()
                    if col1.isdigit() or col1.lower() in ('x', 'rownames', '', 'v1'):
                        col1_numeric += 1
                    if col2 and not col2.isdigit() and any(c.isalpha() for c in col2):
                        col2_has_genes += 1
            
            # Decide which column to use
            use_col2 = (col1_numeric >= 3 and col2_has_genes >= 3)
            gene_col = 1 if use_col2 else 0
            metadata['gene_column'] = gene_col + 1
            
            # Reset and read all genes
            f.seek(0)
            first_line = True
            for i, line in enumerate(f):
                if i >= max_genes:
                    metadata['truncated'] = True
                    break
                
                line = line.strip()
                if not line:
                    continue
                
                # Handle quoted fields
                line = line.replace('"', '').replace("'", "")
                
                if delimiter:
                    parts = line.split(delimiter)
                else:
                    parts = line.split()
                
                # Skip first row if transposed (it's barcodes)
                if first_line:
                    first_line = False
                    if is_transposed:
                        continue
                    # Also skip if it looks like a header
                    if len(parts) > gene_col:
                        val = parts[gene_col].strip()
                        val_lower = val.lower()
                        if any(x in val_lower for x in ['gene', 'ensembl', 'symbol', 'id', 'name', 'feature', 'x', 'rownames']):
                            continue
                
                if len(parts) > gene_col:
                    val = parts[gene_col].strip()
                    gene_id = clean_gene_id(val)
                    if gene_id and not is_artifact(gene_id):
                        gene_ids.append(gene_id)
        
        metadata['n_genes'] = len(gene_ids)
        
        # Detect species from extracted genes
        if gene_ids:
            species_info = detect_species_from_genes(gene_ids)
            metadata['species_detection'] = species_info
            if species_info['dominant_species'] not in ['rat', 'unknown']:
                metadata['wrong_species'] = species_info['dominant_species']
            if species_info['is_contaminated']:
                metadata['species_contaminated'] = True
                
    except Exception as e:
        metadata['error'] = str(e)
    
    return gene_ids, metadata

def extract_genes_from_features_tsv(filepath: Path) -> Tuple[List[str], Dict[str, Any]]:
    """Extract gene IDs from features.tsv / genes.tsv file (companion to MTX).
    
    Handles:
    - Standard: gene_id <tab> symbol <tab> type
    - Row-indexed: "1" <tab> "Gad1" (gene in col 2)
    """
    metadata = {'format': 'features_tsv', 'n_genes': 0, 'source_file': str(filepath)}
    gene_ids = []
    
    try:
        opener = gzip.open if str(filepath).endswith('.gz') else open
        with opener(filepath, 'rt', errors='replace') as f:
            # Read first few lines to detect format
            first_lines = []
            for i, line in enumerate(f):
                first_lines.append(line.strip())
                if i >= 10:
                    break
            
            # Detect if column 1 is row indices
            col1_numeric = 0
            col2_has_genes = 0
            for line in first_lines:
                if not line:
                    continue
                clean_line = line.replace('"', '').replace("'", "")
                parts = clean_line.split('\t')
                if not parts or len(parts) < 2:
                    parts = clean_line.split()
                if len(parts) >= 2:
                    col1 = parts[0].strip()
                    col2 = parts[1].strip()
                    if col1.isdigit() or col1.lower() in ('x', 'rownames', '', 'v1'):
                        col1_numeric += 1
                    if col2 and not col2.isdigit() and any(c.isalpha() for c in col2):
                        col2_has_genes += 1
            
            # Decide which column to use
            use_col2 = (col1_numeric >= 3 and col2_has_genes >= 3)
            gene_col = 1 if use_col2 else 0
            metadata['gene_column'] = gene_col + 1
            
            # Reset and read all genes
            f.seek(0)
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                line = line.replace('"', '').replace("'", "")
                parts = line.split('\t')
                if not parts or len(parts) <= gene_col:
                    parts = line.split()
                
                if len(parts) > gene_col:
                    gene_id = clean_gene_id(parts[gene_col])
                    if gene_id and not is_artifact(gene_id):
                        gene_ids.append(gene_id)
        
        metadata['n_genes'] = len(gene_ids)
    except Exception as e:
        metadata['error'] = str(e)
    
    return gene_ids, metadata


def extract_genes_from_mtx(filepath: Path) -> Tuple[List[str], Dict[str, Any]]:
    """Extract all gene IDs from MTX format (via companion genes/features file)."""
    parent = filepath.parent
    metadata = {'format': 'mtx', 'n_genes': 0, 'source_file': None}
    
    candidate_files = []
    for f in parent.iterdir():
        fname = f.name.lower()
        if any(x in fname for x in ['gene', 'feature']) and \
           fname.endswith(('.tsv', '.tsv.gz', '.txt', '.txt.gz')):
            score = score_file_for_rat(f)
            candidate_files.append((score, f))
    
    if not candidate_files:
        metadata['error'] = 'no_features_file'
        return [], metadata
    
    candidate_files.sort(key=lambda x: -x[0])
    
    best_score = candidate_files[0][0]
    if best_score > 0:
        candidate_files = [(s, f) for s, f in candidate_files if s >= 0]
    elif best_score < 0:
        metadata['warning'] = 'all_files_appear_non_rat'
    
    for score, gene_file in candidate_files:
        gene_ids, file_meta = extract_genes_from_features_tsv(gene_file)
        
        if gene_ids:
            metadata['source_file'] = str(gene_file)
            metadata['n_genes'] = len(gene_ids)
            metadata['file_rat_score'] = score
            return gene_ids, metadata
    
    metadata['error'] = 'no_genes_extracted'
    return [], metadata


def extract_genes_from_h5ad(filepath: Path) -> Tuple[List[str], Dict[str, Any]]:
    """Extract all gene IDs from H5AD format."""
    metadata = {'format': 'h5ad', 'n_genes': 0, 'source_file': str(filepath)}
    gene_ids = []
    
    if not HAS_H5PY:
        metadata['error'] = 'h5py not available'
        return gene_ids, metadata
    
    try:
        with h5py.File(filepath, 'r') as f:
            if 'var' not in f:
                metadata['error'] = 'no_var_group'
                return gene_ids, metadata
            
            key_priority = ['gene_ids', 'gene_id', 'feature_name', 'gene_name', 'features', '_index', 'index']
            
            best_ids = None
            best_key = None
            
            for key in key_priority:
                if key in f['var']:
                    raw = f['var'][key][:]
                    ids = [x.decode() if isinstance(x, bytes) else str(x) for x in raw]
                    ids = [clean_gene_id(x) for x in ids]
                    
                    sample = ids[:100]
                    numeric_count = sum(1 for x in sample if x.isdigit())
                    
                    if numeric_count < len(sample) * 0.5:
                        best_ids = ids
                        best_key = key
                        break
                    elif best_ids is None:
                        best_ids = ids
                        best_key = key
            
            if best_ids:
                gene_ids = [g for g in best_ids if g and not is_artifact(g)]
                metadata['gene_key'] = best_key
                metadata['n_genes'] = len(gene_ids)
            else:
                metadata['error'] = 'no_valid_gene_key'
                
    except Exception as e:
        metadata['error'] = str(e)
    
    return gene_ids, metadata


def extract_genes_from_h5(filepath: Path) -> Tuple[List[str], Dict[str, Any]]:
    """Extract all gene IDs from 10x HDF5 format."""
    metadata = {'format': 'h5', 'n_genes': 0, 'source_file': str(filepath)}
    gene_ids = []
    
    if not HAS_H5PY:
        metadata['error'] = 'h5py not available'
        return gene_ids, metadata
    
    try:
        with h5py.File(filepath, 'r') as f:
            if 'matrix' in f and 'features' in f['matrix']:
                for key in ['id', 'name', 'gene_ids']:
                    if key in f['matrix']['features']:
                        raw = f['matrix']['features'][key][:]
                        gene_ids = [x.decode() if isinstance(x, bytes) else str(x) for x in raw]
                        gene_ids = [clean_gene_id(g) for g in gene_ids if clean_gene_id(g)]
                        metadata['gene_key'] = key
                        break
            
            if not gene_ids:
                return extract_genes_from_h5ad(filepath)
            
            metadata['n_genes'] = len(gene_ids)
    except Exception as e:
        metadata['error'] = str(e)
    
    return gene_ids, metadata


def extract_genes_from_loom(filepath: Path) -> Tuple[List[str], Dict[str, Any]]:
    """Extract all gene IDs from Loom format."""
    metadata = {'format': 'loom', 'n_genes': 0, 'source_file': str(filepath)}
    gene_ids = []
    
    if not HAS_H5PY:
        metadata['error'] = 'h5py not available'
        return gene_ids, metadata
    
    try:
        with h5py.File(filepath, 'r') as f:
            if 'row_attrs' in f:
                for attr in ['Gene', 'Accession', 'gene_id', 'gene_name', 'GeneID']:
                    if attr in f['row_attrs']:
                        raw = f['row_attrs'][attr][:]
                        gene_ids = [x.decode() if isinstance(x, bytes) else str(x) for x in raw]
                        gene_ids = [clean_gene_id(g) for g in gene_ids if clean_gene_id(g)]
                        metadata['gene_key'] = attr
                        break
            
            metadata['n_genes'] = len(gene_ids)
    except Exception as e:
        metadata['error'] = str(e)
    
    return gene_ids, metadata



def find_matrix_files(study_path: Path, max_files: int = 30) -> List[Path]:
    """Find potential matrix files in a study directory, including RAW subfolders."""
    matrices = []
    feature_files = []
    
    # Also search RAW subfolders
    search_paths = [study_path]
    for item in study_path.iterdir():
        if item.is_dir() and 'RAW' in item.name:
            search_paths.append(item)
    
    try:
        for search_path in search_paths:
            for root, dirs, files in os.walk(search_path):
                dirs[:] = [d for d in dirs if not d.startswith('.')]
                
                for fname in files:
                    fpath = Path(root) / fname
                    fname_lower = fname.lower()
                    
                    # Skip scATAC-seq and other non-matrix files
                    if any(skip in fname_lower for skip in SKIP_FILE_PATTERNS):
                        continue
                    
                    # Skip if it's an ATAC-seq file
                    if is_atac_file(fpath):
                        continue
                    
                    try:
                        fsize = fpath.stat().st_size
                        if fsize < 1024:
                            continue
                    except:
                        continue
                    
                    rat_score = score_file_for_rat(fpath)
                    
                    # Small feature/gene files - HIGHEST PRIORITY
                    if any(x in fname_lower for x in ['features.tsv', 'genes.tsv', 'feature.tsv', 'gene.tsv']):
                        if fsize < 10 * 1024 * 1024:
                            feature_files.append((rat_score, 0, fpath))
                            continue
                    
                    # Binary formats
                    is_binary = False
                    for ext in MATRIX_EXTENSIONS:
                        if fname_lower.endswith(ext):
                            is_binary = True
                            matrices.append((rat_score, 1, fpath))
                            break
                    
                    if is_binary:
                        continue
                    
                    # TSV/CSV files
                    if fname_lower.endswith(('.tsv', '.tsv.gz', '.csv', '.csv.gz', '.txt', '.txt.gz')):
                        if any(p in fname_lower for p in MATRIX_FILE_PATTERNS):
                            if fsize > 50 * 1024 * 1024:
                                matrices.append((rat_score, 3, fpath))
                            else:
                                matrices.append((rat_score, 2, fpath))
                        elif fsize < 5 * 1024 * 1024:
                            matrices.append((rat_score, 1, fpath))
                    
                    if len(matrices) + len(feature_files) >= max_files * 2:
                        break
                
                if len(matrices) + len(feature_files) >= max_files * 2:
                    break
    except Exception:
        pass
    
    # Combine and sort
    all_files = feature_files + matrices
    all_files.sort(key=lambda x: (x[1], -x[0], x[2].name.lower()))
    
    result = []
    has_positive = any(score > 0 for score, _, _ in all_files)
    
    for score, priority, path in all_files:
        if len(result) >= max_files:
            break
        if score >= 0 or not has_positive:
            result.append(path)
    
    return result


# =============================================================================
# STUDY PROCESSING
# =============================================================================

def process_study(args: Tuple[str, Path, Optional[dict], float]) -> Dict[str, Any]:
    """Process a single study and extract all gene IDs."""
    accession, study_path, rat_ref, min_rat_match = args
    
    result = {
        'accession': accession,
        'gene_ids': [],
        'gene_type': 'unknown',
        'n_genes': 0,
        'confidence': 0.0,
        'is_valid_rat': False,
        'is_contaminated': False,
        'is_bulk': False,
        'n_samples': 0,
        'is_usable': False,
        'rat_ref_match': 0.0,
        'exclusion_reason': None,
        'metadata': {},
        'error': None,
    }
    
    if not study_path.exists():
        result['error'] = 'path_not_found'
        result['exclusion_reason'] = 'path_not_found'
        return result
    
    try:
        matrices = find_matrix_files(study_path)
        
        if not matrices:
            result['error'] = 'no_matrices_found'
            result['exclusion_reason'] = 'no_matrices_found'
            return result
        
        for mpath in matrices:
            fname = mpath.name.lower()
            gene_ids = []
            metadata = {}
            
            file_score = score_file_for_rat(mpath)
            if file_score < -50:
                continue
            
            if fname.endswith(('.mtx', '.mtx.gz')):
                gene_ids, metadata = extract_genes_from_mtx(mpath)
            elif fname.endswith('.h5ad'):
                gene_ids, metadata = extract_genes_from_h5ad(mpath)
                # Bulk detection for binary format
                is_bulk, n_samples = detect_if_bulk_binary(mpath)
                if is_bulk:
                    result['is_bulk'] = True
                    result['n_samples'] = n_samples
                    metadata['n_samples'] = n_samples
                    metadata['bulk_detection'] = 'binary_obs_count'
            elif fname.endswith(('.h5', '.hdf5')):
                gene_ids, metadata = extract_genes_from_h5(mpath)
                # Bulk detection for binary format
                is_bulk, n_samples = detect_if_bulk_binary(mpath)
                if is_bulk:
                    result['is_bulk'] = True
                    result['n_samples'] = n_samples
                    metadata['n_samples'] = n_samples
                    metadata['bulk_detection'] = 'binary_obs_count'
            elif fname.endswith('.loom'):
                gene_ids, metadata = extract_genes_from_loom(mpath)
                # Bulk detection for binary format
                is_bulk, n_samples = detect_if_bulk_binary(mpath)
                if is_bulk:
                    result['is_bulk'] = True
                    result['n_samples'] = n_samples
                    metadata['n_samples'] = n_samples
                    metadata['bulk_detection'] = 'binary_obs_count'
            elif fname.endswith(('.tsv', '.tsv.gz', '.csv', '.csv.gz', '.txt', '.txt.gz')):
                gene_ids, metadata = extract_genes_from_tsv(mpath)
                
                # Check if it's bulk RNA-seq
                is_bulk, n_samples = detect_if_bulk(mpath)
                if is_bulk:
                    result['is_bulk'] = True
                    result['n_samples'] = n_samples
                    metadata['n_samples'] = n_samples
                    metadata['bulk_detection'] = 'column_count'
            
            if gene_ids and len(gene_ids) > 100:
                gene_type, confidence = get_dominant_gene_type(gene_ids[:1000])
                
                if is_contaminated_species(gene_type):
                    continue
                
                result['gene_ids'] = gene_ids
                result['n_genes'] = len(gene_ids)
                result['metadata'] = metadata
                result['gene_type'] = gene_type
                result['confidence'] = confidence
                result['is_valid_rat'] = is_valid_rat_gene_type(gene_type)
                result['is_contaminated'] = is_contaminated_species(gene_type)
                
                if rat_ref:
                    result['rat_ref_match'] = validate_genes_against_reference(gene_ids, rat_ref)
                
                break
        
        if not result['gene_ids']:
            result['error'] = 'no_genes_extracted'
            result['exclusion_reason'] = 'no_genes_extracted'
        else:
            # Determine if study is usable based on rat_ref_match
            if result['rat_ref_match'] >= min_rat_match:
                result['is_usable'] = True
            elif result['rat_ref_match'] == 0.0:
                result['exclusion_reason'] = 'no_rat_genes_detected'
            else:
                result['exclusion_reason'] = f'low_rat_match_{result["rat_ref_match"]:.2f}'
    
    except Exception as e:
        result['error'] = str(e)
        result['exclusion_reason'] = f'error: {str(e)}'
    
    return result


# =============================================================================
# MAIN INVENTORY BUILDER
# =============================================================================

def load_training_manifest(manifest_path: Path) -> List[Dict]:
    """Load training manifest."""
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    return manifest.get('studies', [])

def discover_studies(data_root: Path, data_type: str) -> List[Tuple[str, Path]]:
    """Discover studies by scanning directory."""
    studies = []
    seen_accessions = set()
    
    # Map data_type to possible folder names
    type_variants = [data_type]
    if data_type == 'single_cell':
        type_variants.extend(['singlecell', 'single-cell'])
    elif data_type == 'bulk':
        type_variants.extend(['bulk_rna', 'bulk-rna'])
    
    search_paths = []
    for dtype in type_variants:
        search_paths.extend([
            # GEO paths
            data_root / 'geo' / dtype / 'geo_datasets',
            data_root / dtype / 'geo_datasets',
            data_root / 'geo' / dtype,
            # ArrayExpress paths
            data_root / 'arrayexpress' / dtype / 'datasets',
            data_root / 'arrayexpress' / dtype,
            # Direct paths
            data_root / dtype / 'datasets',
            data_root / dtype,
        ])
    
    for search_path in search_paths:
        if search_path.exists():
            logger.info(f"Scanning: {search_path}")
            for item in search_path.iterdir():
                if item.is_dir() and item.name.startswith(('GSE', 'E-')):
                    # Avoid duplicates
                    if item.name not in seen_accessions:
                        studies.append((item.name, item))
                        seen_accessions.add(item.name)
    
    return studies


def build_gene_inventory(
    study_results: List[Dict],
    output_dir: Path,
    min_rat_match: float = 0.3,
    genecompass_vocab_path: Optional[Path] = None,
    rat_ref: Optional[dict] = None,
    biotype_lookup: Optional[dict] = None,
) -> Dict[str, Any]:
    """Build comprehensive gene inventory from USABLE study results only.
    
    If biotype_lookup is provided, enforces GeneCompass core gene list:
    only protein-coding, lncRNA, and miRNA genes are retained.
    """
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Separate usable and excluded studies
    usable_results = [r for r in study_results if r.get('is_usable', False)]
    excluded_results = [r for r in study_results if not r.get('is_usable', False)]
    
    logger.info(f"Usable studies: {len(usable_results)}, Excluded: {len(excluded_results)}")
    
    # Gene types to exclude from inventory (not actual genes)
    EXCLUDE_GENE_TYPES = {
        'genomic_coord', 'genomic_coord_chr',
        'probe_affy', 'probe_illumina',
        'artifact', 'empty',
        'ensembl_mouse',        # mouse genes leaking through from contaminated studies
        'mirna_mouse',          # same
    }
    
    # Build gene frequency counter FROM USABLE STUDIES ONLY
    gene_counter = Counter()
    gene_studies = defaultdict(set)
    gene_types = {}
    genes_excluded_by_type = Counter()    # format-based exclusion
    genes_excluded_by_biotype = Counter() # biotype-based exclusion
    genes_excluded_by_pattern = Counter() # pattern-based exclusion (unresolved)
    gene_ensembl = {}                     # gene_base -> resolved ensembl ID
    gene_biotype = {}                     # gene_base -> biotype (if known)
    
    # Resolution tracking
    resolution_stats = Counter()
    ensembl_from_resolution = set()
    
    # Log biotype filtering mode
    if biotype_lookup:
        logger.info(f"Biotype filter ENABLED: {len(biotype_lookup):,} genes with biotype info")
        logger.info(f"Allowed biotypes: {sorted(ALLOWED_BIOTYPES)}")
    else:
        logger.info("Biotype filter DISABLED (no gene_info reference provided)")
    
    study_summaries = []
    
    for result in study_results:
        accession = result['accession']
        gene_type = result.get('gene_type', 'unknown')
        
        study_summaries.append({
            'accession': accession,
            'n_genes': result.get('n_genes', 0),
            'gene_type': gene_type,
            'confidence': result.get('confidence', 0),
            'is_valid_rat': result.get('is_valid_rat', False),
            'is_contaminated': result.get('is_contaminated', False),
            'is_bulk': result.get('is_bulk', False),
            'n_samples': result.get('n_samples', 0),
            'is_usable': result.get('is_usable', False),
            'rat_ref_match': result.get('rat_ref_match', 0),
            'exclusion_reason': result.get('exclusion_reason'),
            'error': result.get('error'),
        })
        
        # ONLY add genes from USABLE studies
        if result.get('is_usable', False):
            for gene_id in result.get('gene_ids', []):
                gene_id_clean = clean_gene_id(gene_id)
                if not gene_id_clean or is_artifact(gene_id_clean):
                    continue
                
                # Use base ID (strip version) for counting
                gene_base = gene_id_clean.split('.')[0]
                
                # Classify gene type (only on first encounter)
                if gene_base not in gene_types:
                    gene_types[gene_base] = classify_gene_id(gene_id_clean)
                
                # FILTER 1: Skip non-gene entries (genomic coordinates, probes)
                if gene_types[gene_base] in EXCLUDE_GENE_TYPES:
                    genes_excluded_by_type[gene_types[gene_base]] += 1
                    continue
                
                # Resolve to Ensembl via rat reference (do this before biotype check)
                ens_id = gene_ensembl.get(gene_base)
                if ens_id is None and rat_ref:
                    resolved_id, status = resolve_to_ensembl(gene_id_clean, rat_ref)
                    resolution_stats[status] += 1
                    if resolved_id:
                        gene_ensembl[gene_base] = resolved_id
                    else:
                        gene_ensembl[gene_base] = ''  # mark as attempted
                
                # FILTER 2: Biotype check (for genes with Ensembl IDs)
                if biotype_lookup:
                    ens_id = gene_ensembl.get(gene_base, '')
                    
                    if ens_id:
                        # We have an Ensembl ID - check biotype
                        biotype = biotype_lookup.get(ens_id.upper(), '')
                        if biotype:
                            gene_biotype[gene_base] = biotype
                            if not is_allowed_biotype(biotype):
                                genes_excluded_by_biotype[biotype] += 1
                                continue
                        # else: no biotype info available, keep gene (conservative)
                    else:
                        # No Ensembl ID - use pattern heuristics
                        pattern_match = is_non_gene_pattern(gene_base)
                        if pattern_match:
                            genes_excluded_by_pattern[gene_types[gene_base]] += 1
                            continue
                
                gene_counter[gene_base] += 1
                gene_studies[gene_base].add(accession)
                
                # Only track Ensembl IDs for genes that passed ALL filters
                ens_id = gene_ensembl.get(gene_base, '')
                if ens_id:
                    ensembl_from_resolution.add(ens_id)
    
    # Load GeneCompass vocabulary if provided
    gc_human_genes = set()
    gc_mouse_genes = set()
    if genecompass_vocab_path and genecompass_vocab_path.exists():
        try:
            with open(genecompass_vocab_path, 'rb') as f:
                vocab = pickle.load(f)
            gc_human_genes = {k for k in vocab.keys() if isinstance(k, str) and k.startswith('ENSG')}
            gc_mouse_genes = {k for k in vocab.keys() if isinstance(k, str) and k.startswith('ENSMUSG')}
        except:
            pass
    
    # Build gene inventory records
    gene_records = []
    for gene_id, count in gene_counter.most_common():
        gene_type = gene_types.get(gene_id, 'unknown')
        studies_list = sorted(gene_studies[gene_id])
        
        gene_records.append({
            'gene_id': gene_id,
            'gene_id_base': gene_id.split('.')[0],
            'gene_type': gene_type,
            'n_studies': count,
            'studies': ','.join(studies_list[:10]) + ('...' if len(studies_list) > 10 else ''),
        })
    
    # Save gene inventory
    inventory_file = output_dir / 'gene_inventory.tsv'
    with open(inventory_file, 'w') as f:
        f.write('gene_id\tgene_id_base\tgene_type\tn_studies\tstudies\n')
        for rec in gene_records:
            f.write(f"{rec['gene_id']}\t{rec['gene_id_base']}\t{rec['gene_type']}\t{rec['n_studies']}\t{rec['studies']}\n")
    
    # Save study summary (ALL studies, with usability flag)
    study_file = output_dir / 'study_gene_summary.tsv'
    with open(study_file, 'w') as f:
        f.write('accession\tn_genes\tgene_type\tconfidence\tis_valid_rat\tis_contaminated\tis_bulk\tn_samples\tis_usable\trat_ref_match\texclusion_reason\terror\n')
        for rec in study_summaries:
            f.write(f"{rec['accession']}\t{rec['n_genes']}\t{rec['gene_type']}\t{rec['confidence']:.2f}\t"
                    f"{rec['is_valid_rat']}\t{rec['is_contaminated']}\t{rec['is_bulk']}\t{rec['n_samples']}\t{rec['is_usable']}\t"
                    f"{rec['rat_ref_match']:.2f}\t{rec['exclusion_reason'] or ''}\t{rec['error'] or ''}\n")
    
    # Save excluded studies list
    excluded_file = output_dir / 'excluded_studies.tsv'
    with open(excluded_file, 'w') as f:
        f.write('accession\tn_genes\tgene_type\trat_ref_match\texclusion_reason\n')
        for rec in study_summaries:
            if not rec['is_usable']:
                f.write(f"{rec['accession']}\t{rec['n_genes']}\t{rec['gene_type']}\t{rec['rat_ref_match']:.2f}\t{rec['exclusion_reason'] or 'unknown'}\n")
    
    # Compute summary statistics (only for genes in the inventory, not excluded)
    type_counts = Counter(gene_types[g] for g in gene_counter)
    
    freq_dist = {
        '1_study': sum(1 for c in gene_counter.values() if c == 1),
        '2-5_studies': sum(1 for c in gene_counter.values() if 2 <= c <= 5),
        '6-20_studies': sum(1 for c in gene_counter.values() if 6 <= c <= 20),
        '21-50_studies': sum(1 for c in gene_counter.values() if 21 <= c <= 50),
        '50+_studies': sum(1 for c in gene_counter.values() if c > 50),
    }
    
    # Extract rat Ensembl IDs (native + resolved)
    rat_ensembl_ids = {rec['gene_id_base'] for rec in gene_records 
                       if rec['gene_type'] in ('ensembl_rat', 'ensembl_rat_versioned', 'rat_prefixed_ensembl')}
    
    # Add resolved symbols
    rat_ensembl_ids.update(ensembl_from_resolution)
    
    logger.info(f"Resolution stats: {dict(resolution_stats)}")
    
    # Also get the prefixed versions (strip rat_ prefix)
    for rec in gene_records:
        if rec['gene_type'] == 'rat_prefixed_ensembl':
            base = rec['gene_id'].replace('rat_', '').split('.')[0]
            if base.upper().startswith('ENSRNOG'):
                rat_ensembl_ids.add(base.upper())
    
    # Exclusion reason breakdown
    exclusion_reasons = Counter(r.get('exclusion_reason') for r in study_summaries if r.get('exclusion_reason'))
    
    summary = {
        'timestamp': datetime.now().isoformat(),
        'min_rat_ref_match_threshold': min_rat_match,
        'total_genes': len(gene_counter),
        'genes_excluded_by_type': dict(genes_excluded_by_type),
        'genes_excluded_by_biotype': dict(genes_excluded_by_biotype),
        'genes_excluded_by_pattern': dict(genes_excluded_by_pattern),
        'biotype_filter_enabled': biotype_lookup is not None,
        'total_studies_processed': len(study_results),
        'studies_usable': len(usable_results),
        'studies_excluded': len(excluded_results),
        'studies_with_genes': sum(1 for s in study_summaries if s['n_genes'] > 0),
        'studies_valid_rat': sum(1 for s in study_summaries if s['is_valid_rat']),
        'studies_contaminated': sum(1 for s in study_summaries if s['is_contaminated']),
        'studies_bulk': sum(1 for s in study_summaries if s['is_bulk']),
        'exclusion_reasons': dict(exclusion_reasons),
        'gene_type_distribution': dict(type_counts),
        'frequency_distribution': freq_dist,
        'rat_ensembl_unique': len(rat_ensembl_ids),
        'resolution_stats': dict(resolution_stats),
        'genecompass_coverage': {
            'human_genes_in_vocab': len(gc_human_genes),
            'mouse_genes_in_vocab': len(gc_mouse_genes),
        },
    }
    
    # Save summary
    summary_file = output_dir / 'gene_inventory_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Save rat Ensembl IDs
    rat_ids_file = output_dir / 'rat_ensembl_ids.txt'
    with open(rat_ids_file, 'w') as f:
        for gene_id in sorted(rat_ensembl_ids):
            f.write(f"{gene_id}\n")
    
    logger.info(f"Saved gene inventory: {inventory_file}")
    logger.info(f"Saved study summary: {study_file}")
    logger.info(f"Saved excluded studies: {excluded_file}")
    logger.info(f"Saved statistics: {summary_file}")
    logger.info(f"Saved rat Ensembl IDs: {rat_ids_file}")
    
    return summary


def main():
    parser = argparse.ArgumentParser(description='Build complete gene inventory from training corpus (v3.3)')
    parser.add_argument('--manifest', '-m', help='Path to training manifest JSON')
    parser.add_argument('--data-root', type=Path, default=Path('../../data'),
                        help='Root path for study directories')
    parser.add_argument('--data-type', choices=['single_cell', 'bulk'], default='single_cell',
                        help='Data type to process')
    parser.add_argument('--rat-reference', type=Path,
                        default=Path('../../data/references/biomart/rat_symbol_lookup.pickle'),
                        help='Path to rat symbol lookup pickle for validation and resolution')
    parser.add_argument('--min-rat-match', type=float, default=MIN_RAT_REF_MATCH,
                        help=f'Minimum rat_ref_match for study to be usable (default: {MIN_RAT_REF_MATCH})')
    parser.add_argument('--genecompass-vocab', type=Path,
                        default=Path('../GeneCompass/prior_knowledge/human_mouse_tokens.pickle'),
                        help='Path to GeneCompass vocabulary')
    parser.add_argument('--rat-gene-info', type=Path,
                        default=None,
                        help='Path to rat_gene_info.tsv for biotype filtering (GeneCompass core gene list)')
    parser.add_argument('-o', '--output-dir', type=Path, required=True,
                        help='Output directory')
    parser.add_argument('--workers', '-w', type=int, default=8, help='Number of workers')
    parser.add_argument('--max-studies', '-n', type=int, help='Max studies to process')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    min_rat_match = args.min_rat_match
    
    logger.info("=" * 60)
    logger.info("GENE INVENTORY BUILDER v3.3")
    logger.info(f"Min rat_ref_match threshold: {min_rat_match}")
    logger.info("=" * 60)
    
    rat_ref = None
    if args.rat_reference and args.rat_reference.exists():
        rat_ref = load_rat_reference(args.rat_reference)
    
    # Load biotype reference for GeneCompass core gene list filtering
    biotype_lookup = None
    if args.rat_gene_info:
        biotype_lookup = load_biotype_reference(gene_info_path=args.rat_gene_info)
        if biotype_lookup:
            logger.info(f"GeneCompass core gene list filter: ENABLED")
        else:
            logger.warning(f"Could not load biotypes from {args.rat_gene_info}")
    else:
        # Try auto-discovery from biomart dir (same dir as rat_reference)
        if args.rat_reference and args.rat_reference.parent.exists():
            auto_path = args.rat_reference.parent / 'rat_gene_info.tsv'
            if auto_path.exists():
                biotype_lookup = load_biotype_reference(gene_info_path=auto_path)
                if biotype_lookup:
                    logger.info(f"Auto-discovered biotype reference: {auto_path}")
        if not biotype_lookup:
            logger.info("No biotype reference found; core gene list filter disabled")
    
    tasks = []
    
    if args.manifest:
        manifest_path = Path(args.manifest)
        if manifest_path.exists():
            logger.info(f"Loading manifest: {manifest_path}")
            studies = load_training_manifest(manifest_path)
            logger.info(f"Loaded {len(studies)} studies from manifest")
            
            for study in studies:
                accession = study.get('accession')
                
                possible_paths = [
                    args.data_root / args.data_type / 'geo_datasets' / accession,
                    args.data_root / 'geo' / args.data_type / 'geo_datasets' / accession,
                    args.data_root / 'geo' / args.data_type / accession,
                    args.data_root / args.data_type / accession,
                ]
                
                study_path = None
                for p in possible_paths:
                    if p.exists():
                        study_path = p
                        break
                
                if study_path:
                    tasks.append((accession, study_path, rat_ref, min_rat_match))
                else:
                    logger.debug(f"Path not found for {accession}")
        else:
            logger.error(f"Manifest not found: {manifest_path}")
            return 1
    else:
        logger.info(f"Discovering studies in {args.data_root} for {args.data_type}")
        discovered = discover_studies(args.data_root, args.data_type)
        logger.info(f"Discovered {len(discovered)} studies")
        
        for accession, study_path in discovered:
            tasks.append((accession, study_path, rat_ref, min_rat_match))
    
    if not tasks:
        logger.error("No studies found!")
        return 1
    
    if args.max_studies:
        tasks = tasks[:args.max_studies]
        logger.info(f"Limited to {len(tasks)} studies")
    
    logger.info(f"Processing {len(tasks)} studies with {args.workers} workers...")
    
    results = []
    completed = 0
    start_time = datetime.now()
    
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_study, task): task for task in tasks}
        
        for future in as_completed(futures):
            try:
                result = future.result(timeout=300)
                results.append(result)
                
                completed += 1
                
                if completed % 20 == 0 or completed == len(tasks):
                    elapsed = (datetime.now() - start_time).total_seconds()
                    rate = completed / elapsed if elapsed > 0 else 0
                    eta = (len(tasks) - completed) / rate / 60 if rate > 0 else 0
                    
                    usable = sum(1 for r in results if r.get('is_usable'))
                    with_genes = sum(1 for r in results if r.get('n_genes', 0) > 0)
                    
                    logger.info(f"Progress: [{completed}/{len(tasks)}] "
                               f"(usable={usable}, with_genes={with_genes}, "
                               f"{rate:.1f}/sec, ETA: {eta:.1f}m)")
                    
            except Exception as e:
                task = futures[future]
                logger.warning(f"Error processing {task[0]}: {e}")
                results.append({
                    'accession': task[0],
                    'gene_ids': [],
                    'gene_type': 'unknown',
                    'n_genes': 0,
                    'is_usable': False,
                    'exclusion_reason': f'error: {str(e)}',
                    'error': str(e),
                })
    
    elapsed = (datetime.now() - start_time).total_seconds()
    logger.info(f"Processed {completed} studies in {elapsed:.1f}s ({completed/elapsed:.1f}/sec)")
    
    logger.info("Building gene inventory...")
    summary = build_gene_inventory(
        results,
        args.output_dir,
        min_rat_match=min_rat_match,
        genecompass_vocab_path=args.genecompass_vocab if args.genecompass_vocab.exists() else None,
        rat_ref=rat_ref,
        biotype_lookup=biotype_lookup,
    )
    
    # Print summary
    print("\n" + "=" * 70)
    print("GENE INVENTORY SUMMARY (v3.3 - GeneCompass Core Gene List)")
    print("=" * 70)
    print(f"Min rat_ref_match threshold: {summary['min_rat_ref_match_threshold']}")
    print(f"\nTotal studies processed: {summary['total_studies_processed']:,}")
    print(f"Studies USABLE: {summary['studies_usable']:,}")
    print(f"Studies EXCLUDED: {summary['studies_excluded']:,}")
    print(f"  - Bulk RNA-seq detected: {summary['studies_bulk']:,}")
    
    print(f"\n{'EXCLUSION REASONS':-^70}")
    for reason, count in sorted(summary['exclusion_reasons'].items(), key=lambda x: -x[1]):
        print(f"  {reason:40} {count:>5}")
    
    print(f"\nTotal unique genes (from usable studies): {summary['total_genes']:,}")
    
    if summary['genes_excluded_by_type']:
        excluded_total = sum(summary['genes_excluded_by_type'].values())
        print(f"\nFormat-based exclusions: {excluded_total:,} observations")
        for gtype, count in sorted(summary['genes_excluded_by_type'].items(), key=lambda x: -x[1]):
            print(f"  {gtype:30} {count:>10,}")
    
    if summary.get('biotype_filter_enabled'):
        print(f"\n{'GENECOMPASS CORE GENE LIST FILTER':-^70}")
        if summary['genes_excluded_by_biotype']:
            biotype_total = sum(summary['genes_excluded_by_biotype'].values())
            print(f"Excluded by biotype: {biotype_total:,} observations")
            for biotype, count in sorted(summary['genes_excluded_by_biotype'].items(), key=lambda x: -x[1]):
                print(f"  {biotype:30} {count:>10,}")
        else:
            print("  No genes excluded by biotype")
        
        if summary['genes_excluded_by_pattern']:
            pattern_total = sum(summary['genes_excluded_by_pattern'].values())
            print(f"Excluded by pattern (unresolved non-genes): {pattern_total:,} observations")
            for ptype, count in sorted(summary['genes_excluded_by_pattern'].items(), key=lambda x: -x[1]):
                print(f"  {ptype:30} {count:>10,}")
        else:
            print("  No genes excluded by pattern")
    else:
        print("\nBiotype filter: DISABLED (no --rat-gene-info provided)")
    
    print(f"\nUnique rat Ensembl IDs: {summary['rat_ensembl_unique']:,}")
    
    print(f"\n{'GENE TYPE DISTRIBUTION':-^70}")
    for gtype, count in sorted(summary['gene_type_distribution'].items(), key=lambda x: -x[1]):
        pct = count / summary['total_genes'] * 100 if summary['total_genes'] > 0 else 0
        print(f"  {gtype:30} {count:>8,} ({pct:5.1f}%)")
    
    print(f"\n{'FREQUENCY DISTRIBUTION':-^70}")
    for freq_bin, count in summary['frequency_distribution'].items():
        print(f"  {freq_bin:20} {count:>8,} genes")
    
    print("\n" + "=" * 70)
    print(f"Output directory: {args.output_dir}")
    print("=" * 70)
    
    return 0


if __name__ == '__main__':
    exit(main())