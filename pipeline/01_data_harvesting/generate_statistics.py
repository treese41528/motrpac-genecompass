#!/usr/bin/env python3
"""
generate_statistics.py - Generate aggregate statistics from the harvested data catalog

This script reads the master_catalog.json and generates:
- Aggregate statistics by source, type, organism, etc.
- Data quality reports
- File format distributions
- Visual-ready data for the Study Explorer

Usage:
    python generate_statistics.py --config config.yaml [--report]
    python generate_statistics.py --catalog ./catalog/master_catalog.json --output ./catalog/statistics.json
"""

import os
import re
import json
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Any, Tuple, Optional
import sys

# --- Config integration ---
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'lib'))
from gene_utils import load_config, resolve_path

# Load config (used for defaults; CLI args still override)
try:
    _config = load_config()
except FileNotFoundError:
    _config = None


# =============================================================================
# ORGANISM DATABASE - Comprehensive aliases to canonical names
# =============================================================================

ORGANISM_ALIASES = {
    # Rat
    'rattus norvegicus': 'Rattus norvegicus', 'rat': 'Rattus norvegicus',
    'rats': 'Rattus norvegicus', 'r. norvegicus': 'Rattus norvegicus',
    'norway rat': 'Rattus norvegicus', 'brown rat': 'Rattus norvegicus',
    'sprague-dawley': 'Rattus norvegicus', 'sprague dawley': 'Rattus norvegicus',
    'wistar': 'Rattus norvegicus', 'fischer 344': 'Rattus norvegicus',
    'lewis rat': 'Rattus norvegicus',
    # Mouse
    'mus musculus': 'Mus musculus', 'mouse': 'Mus musculus', 'mice': 'Mus musculus',
    'm. musculus': 'Mus musculus', 'c57bl/6': 'Mus musculus', 'c57bl6': 'Mus musculus',
    'balb/c': 'Mus musculus', 'cd-1': 'Mus musculus', 'nude mouse': 'Mus musculus',
    'c57bl/6j': 'Mus musculus', 'c57bl/6n': 'Mus musculus',
    # Human
    'homo sapiens': 'Homo sapiens', 'human': 'Homo sapiens', 'h. sapiens': 'Homo sapiens',
    # Pig
    'sus scrofa': 'Sus scrofa', 'pig': 'Sus scrofa', 'swine': 'Sus scrofa', 'porcine': 'Sus scrofa',
    # Cow
    'bos taurus': 'Bos taurus', 'cow': 'Bos taurus', 'cattle': 'Bos taurus', 'bovine': 'Bos taurus',
    # Other mammals
    'ovis aries': 'Ovis aries', 'sheep': 'Ovis aries', 'ovine': 'Ovis aries',
    'capra hircus': 'Capra hircus', 'goat': 'Capra hircus',
    'equus caballus': 'Equus caballus', 'horse': 'Equus caballus', 'equine': 'Equus caballus',
    'canis familiaris': 'Canis familiaris', 'canis lupus familiaris': 'Canis familiaris',
    'dog': 'Canis familiaris', 'canine': 'Canis familiaris',
    'felis catus': 'Felis catus', 'cat': 'Felis catus', 'feline': 'Felis catus',
    'oryctolagus cuniculus': 'Oryctolagus cuniculus', 'rabbit': 'Oryctolagus cuniculus',
    'cavia porcellus': 'Cavia porcellus', 'guinea pig': 'Cavia porcellus',
    'mesocricetus auratus': 'Mesocricetus auratus', 'hamster': 'Mesocricetus auratus',
    'heterocephalus glaber': 'Heterocephalus glaber', 'naked mole rat': 'Heterocephalus glaber',
    'naked mole-rat': 'Heterocephalus glaber',
    # Primates
    'macaca mulatta': 'Macaca mulatta', 'rhesus': 'Macaca mulatta', 
    'rhesus macaque': 'Macaca mulatta', 'macaque': 'Macaca mulatta',
    'macaca fascicularis': 'Macaca fascicularis', 'cynomolgus': 'Macaca fascicularis',
    'pan troglodytes': 'Pan troglodytes', 'chimpanzee': 'Pan troglodytes',
    'pan paniscus': 'Pan paniscus', 'bonobo': 'Pan paniscus',
    'callithrix jacchus': 'Callithrix jacchus', 'marmoset': 'Callithrix jacchus',
    'pongo abelii': 'Pongo abelii', 'orangutan': 'Pongo abelii',
    # Fish
    'danio rerio': 'Danio rerio', 'zebrafish': 'Danio rerio', 'd. rerio': 'Danio rerio',
    'oryzias latipes': 'Oryzias latipes', 'medaka': 'Oryzias latipes',
    'takifugu rubripes': 'Takifugu rubripes', 'fugu': 'Takifugu rubripes',
    # Birds
    'gallus gallus': 'Gallus gallus', 'chicken': 'Gallus gallus', 'chick': 'Gallus gallus',
    # Amphibians
    'xenopus laevis': 'Xenopus laevis', 'xenopus': 'Xenopus laevis', 
    'african clawed frog': 'Xenopus laevis', 'x. laevis': 'Xenopus laevis',
    'xenopus tropicalis': 'Xenopus tropicalis', 'x. tropicalis': 'Xenopus tropicalis',
    # Invertebrates
    'drosophila melanogaster': 'Drosophila melanogaster', 'drosophila': 'Drosophila melanogaster',
    'fruit fly': 'Drosophila melanogaster', 'd. melanogaster': 'Drosophila melanogaster',
    'caenorhabditis elegans': 'Caenorhabditis elegans', 'c. elegans': 'Caenorhabditis elegans',
    'apis mellifera': 'Apis mellifera', 'honey bee': 'Apis mellifera',
    'bombyx mori': 'Bombyx mori', 'silkworm': 'Bombyx mori',
    # Microorganisms
    'saccharomyces cerevisiae': 'Saccharomyces cerevisiae', 'yeast': 'Saccharomyces cerevisiae',
    's. cerevisiae': 'Saccharomyces cerevisiae',
    'schizosaccharomyces pombe': 'Schizosaccharomyces pombe', 'fission yeast': 'Schizosaccharomyces pombe',
    'escherichia coli': 'Escherichia coli', 'e. coli': 'Escherichia coli',
    'arabidopsis thaliana': 'Arabidopsis thaliana', 'arabidopsis': 'Arabidopsis thaliana',
    # Additional species found in data
    'monodelphis domestica': 'Monodelphis domestica', 'opossum': 'Monodelphis domestica',
    'nannospalax galili': 'Nannospalax galili', 'blind mole rat': 'Nannospalax galili',
}

# =============================================================================
# TISSUE DATABASE - Keywords mapped to canonical tissue names
# =============================================================================

TISSUE_DATABASE = {
    # Brain regions (more specific first)
    'hippocampus': {
        'keywords': ['hippocampus', 'hippocampal', 'dentate gyrus', "ammon's horn"],
        'patterns': [r'\bca1\b', r'\bca3\b', r'\bca2\b'],
    },
    'cerebellum': {
        'keywords': ['cerebellum', 'cerebellar', 'purkinje'],
    },
    'hypothalamus': {
        'keywords': ['hypothalamus', 'hypothalamic', 'arcuate nucleus', 
                     'paraventricular nucleus of hypothalamus', 'dorsomedial nucleus of hypothalamus',
                     'ventromedial hypothalamus', 'paraventricular nucleus', 'dorsomedial nucleus',
                     'suprachiasmatic nucleus', 'supraoptic nucleus'],
    },
    'striatum': {
        'keywords': ['striatum', 'striatal', 'nucleus accumbens', 'accumbens'],
        'patterns': [r'\bcaudate\b', r'\bputamen\b'],
    },
    'prefrontal cortex': {
        'keywords': ['prefrontal cortex', 'prefrontal'],
    },
    'cerebral cortex': {
        'keywords': ['cerebral cortex', 'neocortex', 'frontal cortex', 'motor cortex',
                     'somatosensory cortex', 'visual cortex', 'auditory cortex',
                     'entorhinal cortex', 'piriform cortex', 'insular cortex', 'cortex'],
    },
    'corpus callosum': {
        'keywords': ['corpus callosum'],
    },
    'substantia nigra': {
        'keywords': ['substantia nigra'],
    },
    'amygdala': {
        'keywords': ['amygdala', 'amygdalar', 'basolateral amygdala'],
    },
    'thalamus': {
        'keywords': ['thalamus', 'thalamic'],
    },
    'brainstem': {
        'keywords': ['brainstem', 'brain stem', 'medulla oblongata', 'area postrema',
                     'nucleus of solitary tract', 'solitary tract', 'lateral septal nucleus',
                     'dorsal raphe', 'raphe nucleus', 'locus coeruleus', 'inferior olive',
                     'superior colliculus', 'inferior colliculus', 'periaqueductal gray',
                     'neurointermediate lobe'],
        'patterns': [r'\bpons\b'],
    },
    'olfactory bulb': {
        'keywords': ['olfactory bulb'],
    },
    'spinal cord': {
        'keywords': ['spinal cord'],
    },
    'brain': {
        'keywords': ['brain', 'cerebrum', 'whole brain'],
        'patterns': [r'\bcns\b', r'central nervous system'],
    },
    # Peripheral nervous
    'retina': {
        'keywords': ['retina', 'retinal', 'photoreceptor'],
        'patterns': [r'\brpe\b'],
    },
    'cochlea': {
        'keywords': ['cochlea', 'cochlear', 'inner ear', 'organ of corti'],
    },
    'ear': {
        'keywords': ['ear', 'outer ear', 'middle ear', 'auditory'],
    },
    'dorsal root ganglion': {
        'keywords': ['dorsal root ganglion', 'dorsal root ganglia'],
        'patterns': [r'\bdrg\b'],
    },
    'peripheral nerve': {
        'keywords': ['sciatic nerve', 'peripheral nerve', 'vagus nerve', 'trigeminal nerve',
                     'optic nerve', 'stellate ganglion', 'stellate ganglia'],
    },
    'ganglion': {
        'keywords': ['ganglion', 'ganglia'],
    },
    # Cardiovascular
    'heart': {
        'keywords': ['heart', 'cardiac', 'myocardium', 'myocardial', 'cardiomyocyte',
                     'cardiac myocyte', 'left ventricle', 'right ventricle', 
                     'heart left ventricle', 'heart right ventricle',
                     'left atrium', 'right atrium', 'epicardium', 'endocardium',
                     'infarct', 'ischemic heart', 'cardiac muscle cell'],
        'patterns': [r'\batrium\b', r'\batrial\b'],
    },
    'aorta': {
        'keywords': ['aorta', 'aortic'],
    },
    'blood vessel': {
        'keywords': ['artery', 'arterial', 'vein', 'venous', 'blood vessel',
                     'endothelium', 'vascular', 'carotid', 'femoral artery',
                     'pulmonary artery', 'coronary artery', 'mesenteric artery'],
        'patterns': [r'\bhuvec\b'],
    },
    'blood': {
        'keywords': ['whole blood', 'peripheral blood', 'blood sample', 'blood'],
        'patterns': [r'\bpbmc\b', r'\bpbmcs\b'],
    },
    # Immune/Lymphoid
    'bone marrow': {
        'keywords': ['bone marrow', 'marrow'],
    },
    'spleen': {
        'keywords': ['spleen', 'splenic', 'splenocyte'],
    },
    'thymus': {
        'keywords': ['thymus', 'thymic', 'thymocyte'],
    },
    'lymph node': {
        'keywords': ['lymph node', 'lymph nodes', 'lymphatic', 'lymphoid tissue'],
    },
    # Digestive
    'liver': {
        'keywords': ['liver', 'hepatic', 'hepatocyte', 'kupffer cell'],
        'patterns': [r'\bhepato'],
    },
    'stomach': {
        'keywords': ['stomach', 'gastric'],
    },
    'small intestine': {
        'keywords': ['small intestine', 'duodenum', 'jejunum', 'ileum'],
    },
    'colon': {
        'keywords': ['colon', 'colonic', 'large intestine', 'colorectal', 'colonocyte'],
        'patterns': [r'\bcecum\b', r'\brectum\b'],
    },
    'intestine': {
        'keywords': ['intestine', 'intestinal', 'gut', 'enteric', 'enterocyte', 'bowel'],
    },
    'pancreas': {
        'keywords': ['pancreas', 'pancreatic', 'islet', 'pancreatic islet', 'islet of langerhans',
                     'beta cell', 'alpha cell', 'acinar', 'exocrine pancreas'],
    },
    'esophagus': {
        'keywords': ['esophagus', 'esophageal', 'oesophagus'],
    },
    'salivary gland': {
        'keywords': ['salivary gland', 'salivary', 'parotid'],
    },
    'gallbladder': {
        'keywords': ['gallbladder', 'gall bladder', 'bile duct', 'biliary'],
    },
    # Respiratory
    'lung': {
        'keywords': ['lung', 'pulmonary', 'alveolar', 'alveoli', 'bronchial', 'bronchus',
                     'pneumocyte', 'airway', 'trachea', 'tracheal'],
    },
    # Urinary
    'kidney': {
        'keywords': ['kidney', 'renal', 'nephron', 'glomerulus', 'glomerular', 
                     'podocyte', 'proximal tubule', 'distal tubule', 'collecting duct',
                     'renal cortex', 'renal medulla', 'kidney cortex', 'kidney medulla',
                     'thin ascending limb', 'thick ascending limb', 'connecting tubule',
                     'cortical thick ascending limb', 'medullary thick ascending limb',
                     'loop of henle'],
        'patterns': [r'\btubule\b', r'\btubular\b'],
    },
    'bladder': {
        'keywords': ['urinary bladder', 'bladder', 'urothelium'],
    },
    # Reproductive - Male
    'testis': {
        'keywords': ['testis', 'testes', 'testicular', 'seminiferous', 'sertoli cell',
                     'leydig cell', 'spermatocyte', 'spermatid'],
    },
    'epididymis': {
        'keywords': ['epididymis', 'epididymal', 'gubernaculum'],
    },
    'prostate': {
        'keywords': ['prostate', 'prostatic'],
    },
    # Reproductive - Female
    'ovary': {
        'keywords': ['ovary', 'ovarian', 'oocyte', 'follicle', 'follicular',
                     'granulosa', 'corpus luteum'],
    },
    'uterus': {
        'keywords': ['uterus', 'uterine', 'endometrium', 'endometrial', 
                     'myometrium', 'cervix', 'cervical'],
    },
    'placenta': {
        'keywords': ['placenta', 'placental', 'trophoblast', 'decidua'],
    },
    'mammary gland': {
        'keywords': ['mammary gland', 'mammary', 'breast tissue', 'lactating', 'breast'],
    },
    # Gonads (general)
    'gonad': {
        'keywords': ['gonad', 'gonads', 'gonadal'],
    },
    # Musculoskeletal
    'skeletal muscle': {
        'keywords': ['skeletal muscle', 'striated muscle', 'myocyte', 'myoblast', 
                     'myotube', 'myofiber', 'satellite cell',
                     'gastrocnemius', 'soleus', 'tibialis', 'quadriceps', 'plantaris',
                     'extensor digitorum', 'flexor', 'diaphragm muscle'],
        'patterns': [r'\bc2c12\b'],
    },
    'muscle': {
        'keywords': ['muscle'],
    },
    'smooth muscle': {
        'keywords': ['smooth muscle', 'vascular smooth muscle'],
        'patterns': [r'\bvsmc\b'],
    },
    'bone': {
        'keywords': ['bone', 'osseous', 'osteoblast', 'osteocyte', 'osteoclast',
                     'trabecular bone', 'cortical bone', 'femur', 'tibia', 
                     'calvaria', 'vertebra', 'femoral mid-diaphysis', 'rib'],
    },
    'cartilage': {
        'keywords': ['cartilage', 'chondrocyte', 'articular cartilage', 'meniscus'],
    },
    'tendon': {
        'keywords': ['tendon', 'achilles tendon', 'patellar tendon'],
    },
    'synovium': {
        'keywords': ['synovium', 'synovial', 'joint fluid'],
    },
    # Adipose
    'white adipose tissue': {
        'keywords': ['white adipose', 'subcutaneous fat', 'visceral fat',
                     'epididymal fat', 'inguinal fat', 'gonadal fat', 
                     'mesenteric fat', 'omental fat', 'omentum'],
        'patterns': [r'\bwat\b', r'\bewat\b', r'\biwat\b'],
    },
    'brown adipose tissue': {
        'keywords': ['brown adipose', 'brown fat', 'interscapular fat'],
        'patterns': [r'\bbat\b'],
    },
    'adipose tissue': {
        'keywords': ['adipose', 'adipocyte', 'fat tissue', 'fat pad', 'preadipocyte'],
        'patterns': [r'\b3t3-l1\b', r'\b3t3l1\b'],
    },
    # Skin/Integument
    'skin': {
        'keywords': ['skin', 'dermis', 'dermal', 'epidermis', 'epidermal',
                     'keratinocyte', 'melanocyte', 'cutaneous', 'skin of body'],
    },
    # Endocrine
    'thyroid': {
        'keywords': ['thyroid', 'thyroid gland'],
    },
    'adrenal gland': {
        'keywords': ['adrenal gland', 'adrenal', 'adrenal cortex', 'adrenal medulla'],
    },
    'pituitary': {
        'keywords': ['pituitary', 'pituitary gland', 'hypophysis'],
    },
    # Eye
    'eye': {
        'keywords': ['eye', 'ocular', 'cornea', 'corneal', 'lens', 'iris'],
    },
    # Other
    'embryo': {
        'keywords': ['embryo', 'embryonic', 'fetal', 'fetus', 'inner cell mass', 'morula',
                     'amniotic fluid', 'amnion', 'blastocyst', 'epiblast', 
                     'yolk sac', 'whole larvae', 'larvae', 'larval'],
    },
    'tongue': {
        'keywords': ['tongue', 'lingual', 'taste bud'],
    },
    'nose': {
        'keywords': ['nasal', 'olfactory epithelium'],
    },
    'stroma': {
        'keywords': ['stroma', 'stromal', 'mesenchyme'],
    },
    # Insect tissues
    'antenna': {
        'keywords': ['antenna', 'antennae', 'antennal'],
    },
    'wing disc': {
        'keywords': ['wing disc', 'wing imaginal disc', 'imaginal disc'],
    },
    'fat body': {
        'keywords': ['fat body'],
    },
    # Ventricle - context usually cardiac in this dataset
    'ventricle': {
        'keywords': ['ventricle', 'ventricular'],
    },
}

# Build reverse lookup for fast keyword matching
TISSUE_KEYWORD_MAP = {}
TISSUE_PATTERN_MAP = {}
for tissue, data in TISSUE_DATABASE.items():
    for kw in data.get('keywords', []):
        TISSUE_KEYWORD_MAP[kw.lower()] = tissue
    for pat in data.get('patterns', []):
        TISSUE_PATTERN_MAP[pat] = tissue

# =============================================================================
# CELL TYPE DATABASE - Cell types mapped to tissue of origin
# =============================================================================

CELL_TYPE_TISSUE_MAP = {
    # Neural
    'neuron': 'brain', 'neuronal': 'brain', 'neural progenitor': 'brain',
    'astrocyte': 'brain', 'microglia': 'brain', 'oligodendrocyte': 'brain',
    'glia': 'brain', 'glial': 'brain', 'neural stem cell': 'brain',
    'purkinje cell': 'cerebellum',
    'pyramidal neuron': 'cerebral cortex', 'pyramidal cell': 'cerebral cortex',
    'hippocampal neuron': 'hippocampus',
    'dopaminergic neuron': 'substantia nigra',
    'schwann cell': 'peripheral nerve',
    # Heart
    'cardiomyocyte': 'heart', 'cardiac myocyte': 'heart', 'cardiac fibroblast': 'heart',
    'cardiac muscle cell': 'heart',
    # Liver
    'hepatocyte': 'liver', 'kupffer cell': 'liver', 'hepatic stellate': 'liver',
    # Pancreas
    'islet cell': 'pancreas', 'beta cell': 'pancreas', 'alpha cell': 'pancreas',
    'acinar cell': 'pancreas',
    # Adipose
    'adipocyte': 'adipose tissue', 'preadipocyte': 'adipose tissue',
    # Muscle
    'myoblast': 'skeletal muscle', 'myocyte': 'skeletal muscle', 
    'myotube': 'skeletal muscle', 'satellite cell': 'skeletal muscle',
    # Bone/Cartilage
    'osteoblast': 'bone', 'osteocyte': 'bone', 'osteoclast': 'bone',
    'chondrocyte': 'cartilage',
    # Skin
    'keratinocyte': 'skin', 'melanocyte': 'skin',
    # Lung
    'pneumocyte': 'lung', 'alveolar epithelial': 'lung', 'bronchial epithelial': 'lung',
    # Kidney
    'podocyte': 'kidney', 'tubular epithelial': 'kidney', 'mesangial cell': 'kidney',
    # Intestine
    'enterocyte': 'intestine', 'colonocyte': 'colon', 
    'goblet cell': 'intestine', 'paneth cell': 'small intestine',
    # Reproductive
    'spermatocyte': 'testis', 'sertoli cell': 'testis', 'leydig cell': 'testis',
    'oocyte': 'ovary', 'granulosa cell': 'ovary',
    # Immune
    'thymocyte': 'thymus', 'splenocyte': 'spleen',
    # Eye
    'photoreceptor': 'retina', 'retinal ganglion cell': 'retina',
    # Blood/Marrow
    'hematopoietic stem cell': 'bone marrow', 'mesenchymal stem cell': 'bone marrow',
}

# =============================================================================
# EXCLUDED TERMS - Terms that are neither organism nor tissue
# =============================================================================

EXCLUDED_TERMS = {
    '', ' ', '  ', '   ', 'n/a', 'na', 'none', 'unknown', 'not applicable', 
    'not available', 'other', 'unspecified', 'not specified',
    'cell', 'cells', 'cell line', 'cell lines', 'primary cell', 'primary cells',
    'culture', 'cultured', 'in vitro', 'in vivo', 'cell line (in vitro cultured)',
    'sample', 'samples', 'specimen', 'tissue sample',
    'control', 'treated', 'untreated', 'normal', 'healthy', 'diseased',
    'wild type', 'wildtype', 'wt', 'knockout', 'ko', 'transgenic', 'mutant',
    'male', 'female', 'adult', 'young', 'old', 'aged', 'juvenile', 'neonatal',
    'mixed sample', 'mixed', 'pooled', 'pool', 'mixed-tissue', 'mixed 7 tissues',
    'replicate', 'biological replicate', 'technical replicate',
    'normal tissue', 'tumor', 'tumour', 'cancer', 'carcinoma', 'neoplasm',
    'flow-sorted tumor cells', 'tumor cells', 'llc1 primary tumor',
    'disease model',  # Generic term
    # Cell types that are NOT tissues (immune cells, etc.)
    't cell', 't cells', 't-cell', 'cd4+ t cell', 'cd8+ t cell',
    'gamma-delta t-cell', 'gamma-delta t cell', 'cd4-positive, alpha-beta t cell',
    'double negative t-cell',
    'b cell', 'b cells', 'b-cell',
    'nk cell', 'nk cells', 'natural killer cell',
    'macrophage', 'macrophages', 'monocyte', 'monocytes',
    'dendritic cell', 'dendritic cells',
    'neutrophil', 'neutrophils', 'eosinophil', 'basophil',
    'lymphocyte', 'lymphocytes', 'leukocyte', 'leukocytes',
    'fibroblast', 'fibroblasts',
    'mononuclear cell', 'mononuclear cells',
    'langerhans cell', 'langerhans cells',
    # Stem/progenitor cells (too generic)
    'stem cell', 'stem cells', 'progenitor', 'progenitors',
    'induced pluripotent stem cell', 'ipsc', 'ips cell', 'ips cells',
    'hematopoietic stem and progenitor cell', 'hspc',
    'multipotent progenitor cell', 'multipotent progenitor', 'multipotent progenitors',
    'hematopoietic multipotent progenitor cell',
    'mesenchymal stromal cell', 'mesenchymal stromal cells',
    'common lymphoid progenitor',
    # Generic cell descriptors
    'endothelial cell', 'endothelial cells',
    'epithelial cell', 'epithelial cells', 'epithelial',
    'somatic', 'somatic cell',
}

# =============================================================================
# CLASSIFICATION FUNCTIONS
# =============================================================================

def normalize_text(text: str) -> str:
    """Normalize text for matching."""
    if not text:
        return ''
    return ' '.join(text.lower().split())


def classify_organism(text: str) -> Tuple[bool, Optional[str]]:
    """
    Check if text represents an organism.
    Returns (is_organism, canonical_name or None).
    """
    if not text:
        return False, None
    
    normalized = normalize_text(text)
    
    # Check excluded terms
    if normalized in EXCLUDED_TERMS:
        return False, None
    
    # Direct lookup in organism aliases
    if normalized in ORGANISM_ALIASES:
        return True, ORGANISM_ALIASES[normalized]
    
    # Check if it matches a tissue keyword (then not an organism)
    if normalized in TISSUE_KEYWORD_MAP:
        return False, None
    
    # Check tissue patterns
    for pattern, tissue in TISSUE_PATTERN_MAP.items():
        if re.search(pattern, normalized, re.IGNORECASE):
            return False, None
    
    # Check for partial organism matches
    for alias, canonical in ORGANISM_ALIASES.items():
        if len(alias) >= 6 and alias in normalized:
            return True, canonical
    
    # Check for Latin binomial pattern (Genus species)
    original_parts = text.strip().split()
    if len(original_parts) == 2:
        genus, species = original_parts
        if (len(genus) > 1 and genus[0].isupper() and genus[1:].islower() and 
            species[0].islower() and species.isalpha()):
            return True, text.strip()
    
    return False, None


def classify_tissue(text: str) -> Tuple[bool, Optional[str]]:
    """
    Check if text represents a tissue.
    Returns (is_tissue, canonical_name or None).
    """
    if not text:
        return False, None
    
    normalized = normalize_text(text)
    
    # Check excluded terms
    if normalized in EXCLUDED_TERMS:
        return False, None
    
    # Check if it's an organism (then not a tissue)
    if normalized in ORGANISM_ALIASES:
        return False, None
    
    # Check for Latin binomial pattern (organism, not tissue)
    original_parts = text.strip().split()
    if len(original_parts) == 2:
        genus, species = original_parts
        if (len(genus) > 1 and genus[0].isupper() and genus[1:].islower() and 
            species[0].islower() and species.isalpha()):
            return False, None
    
    # Direct lookup in tissue keywords
    if normalized in TISSUE_KEYWORD_MAP:
        return True, TISSUE_KEYWORD_MAP[normalized]
    
    # Check tissue patterns (regex)
    for pattern, tissue in TISSUE_PATTERN_MAP.items():
        if re.search(pattern, normalized, re.IGNORECASE):
            return True, tissue
    
    # Partial matching - check if any keyword is contained in text
    sorted_keywords = sorted(TISSUE_KEYWORD_MAP.keys(), key=len, reverse=True)
    for keyword in sorted_keywords:
        if len(keyword) >= 4 and keyword in normalized:
            return True, TISSUE_KEYWORD_MAP[keyword]
    
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
    
    # Process items from organism field
    for item in organisms:
        is_org, org_name = classify_organism(item)
        is_tis, tis_name = classify_tissue(item)
        
        if is_org and org_name:
            clean_organisms.add(org_name)
        elif is_tis and tis_name:
            clean_tissues.add(tis_name)
    
    # Process items from tissue field
    for item in tissues:
        is_org, org_name = classify_organism(item)
        is_tis, tis_name = classify_tissue(item)
        
        if is_tis and tis_name:
            clean_tissues.add(tis_name)
        elif is_org and org_name:
            clean_organisms.add(org_name)
    
    return sorted(clean_organisms), sorted(clean_tissues)




def load_catalog(catalog_path: str) -> Dict[str, Any]:
    """Load the master catalog JSON."""
    with open(catalog_path, 'r') as f:
        return json.load(f)


def compute_statistics(catalog: Dict[str, Any]) -> Dict[str, Any]:
    """Compute comprehensive statistics from the catalog."""
    studies = catalog.get('studies', [])
    
    stats = {
        'generated_at': datetime.now().isoformat(),
        'catalog_generated_at': catalog.get('generated_at'),
        'totals': {},
        'by_source': {},
        'by_data_type': {},
        'by_organism': {},
        'by_tissue': {},
        'by_technology': {},
        'by_file_format': {},
        'data_quality': {},
        'size_distribution': {},
        'sample_distribution': {},
        'usability_distribution': {},
        'top_studies': {},
        'rat_specific': {}
    }
    
    # Initialize counters
    source_counts = defaultdict(lambda: {'count': 0, 'samples': 0, 'files': 0, 'size_bytes': 0})
    type_counts = defaultdict(lambda: {'count': 0, 'samples': 0, 'files': 0, 'size_bytes': 0})
    organism_counts = defaultdict(lambda: {'count': 0, 'samples': 0})
    tissue_counts = defaultdict(int)
    tech_counts = defaultdict(int)
    format_counts = defaultdict(lambda: {'count': 0, 'size_bytes': 0})
    
    usability_bins = {'high': 0, 'medium': 0, 'low': 0}
    sample_bins = {'0': 0, '1-10': 0, '11-50': 0, '51-100': 0, '101-500': 0, '500+': 0}
    size_bins = {'<1MB': 0, '1-10MB': 0, '10-100MB': 0, '100MB-1GB': 0, '1-10GB': 0, '>10GB': 0}
    
    has_processed = 0
    has_counts = 0
    has_metadata = 0
    
    rat_studies = []
    
    for study in studies:
        source = study.get('source') or 'unknown'
        dtype = study.get('data_type') or 'unknown'
        raw_organisms = study.get('organism') or []
        raw_tissues = study.get('tissues') or []
        technologies = study.get('technologies') or []
        file_formats = study.get('file_formats') or {}
        
        # Clean up organisms and tissues using comprehensive classifier
        organisms, tissues = cleanup_organisms_and_tissues(raw_organisms, raw_tissues)
        
        sample_count = study.get('sample_count') or 0
        file_count = study.get('file_count') or 0
        size_bytes = study.get('total_size_bytes') or 0
        usability = study.get('usability_score') or 0
        
        # By source
        source_counts[source]['count'] += 1
        source_counts[source]['samples'] += sample_count
        source_counts[source]['files'] += file_count
        source_counts[source]['size_bytes'] += size_bytes
        
        # By data type
        type_counts[dtype]['count'] += 1
        type_counts[dtype]['samples'] += sample_count
        type_counts[dtype]['files'] += file_count
        type_counts[dtype]['size_bytes'] += size_bytes
        
        # By organism (cleaned)
        for org in organisms:
            organism_counts[org]['count'] += 1
            organism_counts[org]['samples'] += sample_count
            
            # Check for rat (canonical name)
            if org == 'Rattus norvegicus':
                rat_studies.append({
                    'accession': study.get('accession'),
                    'title': study.get('title'),
                    'source': source,
                    'data_type': dtype,
                    'sample_count': sample_count,
                    'usability_score': usability,
                    'tissues': tissues
                })
        
        # By tissue (cleaned)
        for tissue in tissues:
            tissue_counts[tissue] += 1
        
        # By technology
        for tech in technologies:
            tech_counts[tech] += 1
        
        # By file format
        for fmt, count in file_formats.items():
            format_counts[fmt]['count'] += count
        
        # Quality indicators
        if study.get('has_processed_data'):
            has_processed += 1
        if study.get('has_count_matrix'):
            has_counts += 1
        if study.get('has_metadata'):
            has_metadata += 1
        
        # Usability distribution (matches HTML thresholds)
        if usability >= 70:
            usability_bins['high'] += 1
        elif usability >= 40:
            usability_bins['medium'] += 1
        else:
            usability_bins['low'] += 1
        
        # Sample count distribution
        if sample_count == 0:
            sample_bins['0'] += 1
        elif sample_count <= 10:
            sample_bins['1-10'] += 1
        elif sample_count <= 50:
            sample_bins['11-50'] += 1
        elif sample_count <= 100:
            sample_bins['51-100'] += 1
        elif sample_count <= 500:
            sample_bins['101-500'] += 1
        else:
            sample_bins['500+'] += 1
        
        # Size distribution
        size_mb = size_bytes / (1024 * 1024)
        if size_mb < 1:
            size_bins['<1MB'] += 1
        elif size_mb < 10:
            size_bins['1-10MB'] += 1
        elif size_mb < 100:
            size_bins['10-100MB'] += 1
        elif size_mb < 1024:
            size_bins['100MB-1GB'] += 1
        elif size_mb < 10240:
            size_bins['1-10GB'] += 1
        else:
            size_bins['>10GB'] += 1
    
    # Compile totals
    total_size = sum(s.get('total_size_bytes', 0) for s in studies)
    total_samples = sum(s.get('sample_count', 0) for s in studies)
    total_files = sum(s.get('file_count', 0) for s in studies)
    
    stats['totals'] = {
        'studies': len(studies),
        'samples': total_samples,
        'files': total_files,
        'size_bytes': total_size,
        'size_human': format_size(total_size)
    }
    
    # Convert defaultdicts to regular dicts with human-readable sizes
    stats['by_source'] = {
        source: {
            **data,
            'size_human': format_size(data['size_bytes'])
        }
        for source, data in source_counts.items()
    }
    
    stats['by_data_type'] = {
        dtype: {
            **data,
            'size_human': format_size(data['size_bytes'])
        }
        for dtype, data in type_counts.items()
    }
    
    # Sort organisms by count
    stats['by_organism'] = dict(sorted(
        organism_counts.items(),
        key=lambda x: x[1]['count'],
        reverse=True
    ))
    
    # Top 30 tissues (expanded)
    stats['by_tissue'] = dict(sorted(
        tissue_counts.items(),
        key=lambda x: x[1],
        reverse=True
    )[:30])
    
    stats['by_technology'] = dict(tech_counts)
    
    # File format stats
    stats['by_file_format'] = {
        fmt: {
            **data,
            'size_human': format_size(data.get('size_bytes', 0))
        }
        for fmt, data in format_counts.items()
    }
    
    # Data quality summary
    stats['data_quality'] = {
        'has_processed_data': has_processed,
        'has_count_matrix': has_counts,
        'has_metadata': has_metadata,
        'pct_processed': round(100 * has_processed / len(studies), 1) if studies else 0,
        'pct_counts': round(100 * has_counts / len(studies), 1) if studies else 0,
        'pct_metadata': round(100 * has_metadata / len(studies), 1) if studies else 0
    }
    
    stats['usability_distribution'] = usability_bins
    stats['sample_distribution'] = sample_bins
    stats['size_distribution'] = size_bins
    
    # Top studies by various metrics
    sorted_by_samples = sorted(studies, key=lambda x: x.get('sample_count', 0), reverse=True)
    sorted_by_size = sorted(studies, key=lambda x: x.get('total_size_bytes', 0), reverse=True)
    sorted_by_usability = sorted(studies, key=lambda x: x.get('usability_score', 0), reverse=True)
    
    stats['top_studies'] = {
        'by_sample_count': [
            {
                'accession': s['accession'],
                'title': (s.get('title') or '')[:80],
                'sample_count': s.get('sample_count', 0),
                'source': s.get('source')
            }
            for s in sorted_by_samples[:10]
        ],
        'by_size': [
            {
                'accession': s['accession'],
                'title': (s.get('title') or '')[:80],
                'size': format_size(s.get('total_size_bytes', 0)),
                'source': s.get('source')
            }
            for s in sorted_by_size[:10]
        ],
        'by_usability': [
            {
                'accession': s['accession'],
                'title': (s.get('title') or '')[:80],
                'usability_score': s.get('usability_score', 0),
                'source': s.get('source')
            }
            for s in sorted_by_usability[:10]
        ]
    }
    
    # Rat-specific statistics
    stats['rat_specific'] = {
        'total_studies': len(rat_studies),
        'by_data_type': {
            'single_cell': len([s for s in rat_studies if s['data_type'] == 'single_cell']),
            'bulk': len([s for s in rat_studies if s['data_type'] == 'bulk'])
        },
        'by_source': {
            'geo': len([s for s in rat_studies if s['source'] == 'geo']),
            'arrayexpress': len([s for s in rat_studies if s['source'] == 'arrayexpress'])
        },
        'high_quality': len([s for s in rat_studies if s['usability_score'] >= 70]),
        'studies': rat_studies
    }
    
    # Cross-tabulation: source x type
    cross_tab = {}
    for study in studies:
        key = f"{study.get('source') or 'unknown'}_{study.get('data_type') or 'unknown'}"
        if key not in cross_tab:
            cross_tab[key] = {'count': 0, 'samples': 0}
        cross_tab[key]['count'] += 1
        cross_tab[key]['samples'] += study.get('sample_count') or 0
    
    stats['cross_tabulation'] = cross_tab
    
    return stats


def format_size(size_bytes: int) -> str:
    """Convert bytes to human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} PB"


def print_report(stats: Dict[str, Any]):
    """Print a human-readable statistics report."""
    print("\n" + "="*70)
    print("HARVESTED DATA STATISTICS REPORT")
    print("="*70)
    print(f"Generated: {stats['generated_at']}")
    print(f"Catalog from: {stats['catalog_generated_at']}")
    
    print("\n" + "-"*70)
    print("TOTALS")
    print("-"*70)
    totals = stats['totals']
    print(f"  Studies:     {totals['studies']:,}")
    print(f"  Samples:     {totals['samples']:,}")
    print(f"  Files:       {totals['files']:,}")
    print(f"  Total Size:  {totals['size_human']}")
    
    print("\n" + "-"*70)
    print("BY SOURCE")
    print("-"*70)
    print(f"{'Source':<15} {'Studies':>10} {'Samples':>10} {'Files':>10} {'Size':>12}")
    print("-" * 57)
    for source, data in stats['by_source'].items():
        print(f"{source:<15} {data['count']:>10,} {data['samples']:>10,} {data['files']:>10,} {data['size_human']:>12}")
    
    print("\n" + "-"*70)
    print("BY DATA TYPE")
    print("-"*70)
    print(f"{'Type':<15} {'Studies':>10} {'Samples':>10} {'Files':>10} {'Size':>12}")
    print("-" * 57)
    for dtype, data in stats['by_data_type'].items():
        print(f"{dtype:<15} {data['count']:>10,} {data['samples']:>10,} {data['files']:>10,} {data['size_human']:>12}")
    
    print("\n" + "-"*70)
    print("CROSS-TABULATION (Source × Type)")
    print("-"*70)
    for key, data in stats['cross_tabulation'].items():
        print(f"  {key}: {data['count']} studies, {data['samples']:,} samples")
    
    print("\n" + "-"*70)
    print("TOP ORGANISMS (normalized)")
    print("-"*70)
    for org, data in list(stats['by_organism'].items())[:15]:
        print(f"  {org}: {data['count']} studies, {data['samples']:,} samples")
    
    print("\n" + "-"*70)
    print("TOP TISSUES (normalized)")
    print("-"*70)
    for tissue, count in list(stats['by_tissue'].items())[:20]:
        print(f"  {tissue}: {count} studies")
    
    print("\n" + "-"*70)
    print("RAT-SPECIFIC DATA (Rattus norvegicus)")
    print("-"*70)
    rat = stats['rat_specific']
    print(f"  Total rat studies: {rat['total_studies']}")
    print(f"  Single-cell: {rat['by_data_type'].get('single_cell', 0)}")
    print(f"  Bulk: {rat['by_data_type'].get('bulk', 0)}")
    print(f"  High quality (usability >= 70): {rat['high_quality']}")
    print(f"  From GEO: {rat['by_source'].get('geo', 0)}")
    print(f"  From ArrayExpress: {rat['by_source'].get('arrayexpress', 0)}")
    
    print("\n" + "-"*70)
    print("DATA QUALITY")
    print("-"*70)
    quality = stats['data_quality']
    print(f"  Has processed data: {quality['has_processed_data']} ({quality['pct_processed']}%)")
    print(f"  Has count matrix:   {quality['has_count_matrix']} ({quality['pct_counts']}%)")
    print(f"  Has metadata:       {quality['has_metadata']} ({quality['pct_metadata']}%)")
    
    print("\n" + "-"*70)
    print("USABILITY SCORE DISTRIBUTION")
    print("-"*70)
    for bucket, count in stats['usability_distribution'].items():
        bar = "█" * (count // 5) if count > 0 else ""
        print(f"  {bucket:>8}: {count:>5} {bar}")
    
    print("\n" + "-"*70)
    print("TECHNOLOGIES DETECTED")
    print("-"*70)
    for tech, count in sorted(stats['by_technology'].items(), key=lambda x: x[1], reverse=True):
        print(f"  {tech}: {count}")
    
    print("\n" + "-"*70)
    print("TOP FILE FORMATS")
    print("-"*70)
    sorted_formats = sorted(stats['by_file_format'].items(), key=lambda x: x[1]['count'], reverse=True)
    for fmt, data in sorted_formats[:15]:
        print(f"  {fmt}: {data['count']:,} files")
    
    print("\n" + "="*70)


def main():
    parser = argparse.ArgumentParser(description='Generate statistics from harvested data catalog')
    parser.add_argument('--report', action='store_true', help='Print human-readable report')
    
    args = parser.parse_args()
    
    config = _config or {}
    h = config.get('harvesting', {})
    catalog_dir = resolve_path(config, h.get('catalog_dir', 'data/catalog'))
    catalog_path = catalog_dir / 'master_catalog.json'
    output_path = catalog_dir / 'statistics.json'
    
    # Load catalog
    print(f"Loading catalog from {catalog_path}...")
    catalog = load_catalog(str(catalog_path))
    
    # Compute statistics
    print("Computing statistics...")
    stats = compute_statistics(catalog)
    
    # Save statistics
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Saved statistics to {output_path}")
    
    # Print report
    if args.report:
        print_report(stats)
    else:
        # Always print a brief summary
        print(f"\nTotal studies: {stats['totals']['studies']}")
        print(f"Total size: {stats['totals']['size_human']}")
        print(f"Unique organisms: {len(stats['by_organism'])}")
        print(f"Unique tissues: {len(stats['by_tissue'])}")
        print(f"Rat studies: {stats['rat_specific']['total_studies']}")
        print(f"\nRun with --report for detailed breakdown")


if __name__ == '__main__':
    main()