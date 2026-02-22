"""
gene_utils.py — Shared gene normalization utilities for the rat GeneCompass pipeline.

MECHANISM ONLY. Policy (which biotypes to keep, which accession patterns
to accept, identity thresholds) lives in pipeline_config.yaml.

This module provides:
  - Config loading and path resolution
  - Ensembl ID normalization
  - Biotype canonicalization
  - Gene ID namespace detection
  - Non-gene pattern filtering
  - BioMart reference loading and resolution
  - Stage manifest creation
  - File checksum computation

Config section mapping (pipeline_config.yaml):
  biomart:              BioMart reference metadata and file paths
  rgd:                  RGD synonym resolution settings
  paths:                Data directory paths
  gene_universe:        Stage 2 settings (preprocessing + gene universe construction)
  orthologs:            Stage 3 ortholog mapping
  medians:              Stage 4 gene medians
  reference_assembly:   Stage 5 reference files + corpus export
  slurm:                SLURM job settings

Author: Tim Reese Lab / Claude
Date: February 2026
"""

import csv
import hashlib
import json
import os
import pickle
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Set, Tuple


# ============================================================
# CONFIG LOADING
# ============================================================

def load_config(config_path: Optional[str] = None) -> dict:
    """
    Load pipeline config from YAML.
    Resolves project_root from PIPELINE_ROOT env var or config default.

    Args:
        config_path: Path to YAML file. If None, searches for
                     config/pipeline_config.yaml relative to this file,
                     then cwd.
    """
    try:
        import yaml
    except ImportError:
        raise ImportError("PyYAML required: pip install pyyaml")

    if config_path is None:
        # Search relative to this module, then cwd
        candidates = [
            Path(__file__).parent.parent / "config" / "pipeline_config.yaml",
            Path.cwd() / "config" / "pipeline_config.yaml",
            Path.cwd() / "pipeline_config.yaml",
        ]
        for c in candidates:
            if c.is_file():
                config_path = str(c)
                break
        if config_path is None:
            raise FileNotFoundError(
                "Cannot find pipeline_config.yaml. Set config_path explicitly."
            )

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Resolve project root
    config["_project_root"] = Path(
        os.environ.get("PIPELINE_ROOT", config.get("project_root", "."))
    ).resolve()

    return config


def resolve_path(config: dict, relative_path: str) -> Path:
    """Resolve a config-relative path to absolute using project_root."""
    return config["_project_root"] / relative_path


# ============================================================
# BIOTYPE CANONICALIZATION
# ============================================================
# All keys are lowercase. Lookup always lowercases input first.
#
# GeneCompass core gene list (Cell Research 2024, Yang et al.):
#   protein_coding, lncrna, mirna
#
# Categories excluded by GeneCompass:
#   pseudogenes, tRNAs, rRNAs, and other non-capturable loci

_BIOTYPE_ALIASES = {
    # protein_coding
    "protein_coding":           "protein_coding",
    "protein-coding":           "protein_coding",
    "protein coding":           "protein_coding",
    "protein-coding gene":      "protein_coding",
    # mirna
    "mirna":                    "mirna",
    "micro_rna":                "mirna",
    # lncrna — GeneCompass paper includes lncRNA in core gene list
    "lncrna":                   "lncrna",
    "lincrna":                  "lncrna",
    "long_noncoding_rna":       "lncrna",
    "processed_transcript":     "lncrna",
    "antisense":                "lncrna",
    "sense_intronic":           "lncrna",
    "sense_overlapping":        "lncrna",
    "lincRNA":                  "lncrna",
    "3prime_overlapping_ncrna": "lncrna",
    "bidirectional_promoter_lncrna": "lncrna",
    "macro_lncrna":             "lncrna",
    # snorna
    "snorna":                   "snorna",
    # snrna
    "snrna":                    "snrna",
    # rrna (all rRNA subtypes map to rrna)
    "rrna":                     "rrna",
    "mt_rrna":                  "rrna",
    "mt_trna":                  "rrna",
    # misc_rna
    "misc_rna":                 "misc_rna",
    # pseudogene (all subtypes → pseudogene)
    "pseudo":                   "pseudogene",
    "pseudogene":               "pseudogene",
    "processed_pseudogene":     "pseudogene",
    "unprocessed_pseudogene":   "pseudogene",
    "transcribed_processed_pseudogene":   "pseudogene",
    "transcribed_unprocessed_pseudogene": "pseudogene",
    "translated_processed_pseudogene":    "pseudogene",
    "polymorphic_pseudogene":   "pseudogene",
    # ncrna (generic — NOT mapped to lncrna, these are short ncRNAs)
    "ncrna":                    "ncrna",
    # other
    "gene":                     "gene",
    "tec":                      "tec",
    "ig_c_gene":                "ig_gene",
    "ig_d_gene":                "ig_gene",
    "ig_j_gene":                "ig_gene",
    "ig_v_gene":                "ig_gene",
    "tr_c_gene":                "tr_gene",
    "tr_d_gene":                "tr_gene",
    "tr_j_gene":                "tr_gene",
    "tr_v_gene":                "tr_gene",
}


def normalize_biotype(raw: str) -> str:
    """
    Canonicalize a biotype string to lowercase, underscore-separated form.

    Uses exact lookup, then fuzzy matching for RGD/nonstandard variants.

    >>> normalize_biotype("protein-coding")
    'protein_coding'
    >>> normalize_biotype("protein-coding gene")
    'protein_coding'
    >>> normalize_biotype("miRNA")
    'mirna'
    >>> normalize_biotype("lncRNA")
    'lncrna'
    >>> normalize_biotype("lincRNA")
    'lncrna'
    """
    key = raw.strip().lower().replace("-", "_").replace(" ", "_")

    # Exact lookup
    canonical = _BIOTYPE_ALIASES.get(key)
    if canonical is not None:
        return canonical

    # Also try the raw lowercased form (before underscore normalization)
    key_raw = raw.strip().lower()
    canonical = _BIOTYPE_ALIASES.get(key_raw)
    if canonical is not None:
        return canonical

    # Fuzzy matching for RGD variants and nonstandard labels
    if "protein" in key and "coding" in key:
        return "protein_coding"
    if "mirna" in key or ("micro" in key and "rna" in key):
        return "mirna"
    if "lncrna" in key or "lincrna" in key or ("long" in key and "noncoding" in key):
        return "lncrna"
    if "snorna" in key:
        return "snorna"
    if "snrna" in key:
        return "snrna"
    if "rrna" in key:
        return "rrna"
    if "pseudogene" in key or key == "pseudo":
        return "pseudogene"

    return key


def is_kept_biotype(biotype: str, keep_set: FrozenSet[str]) -> bool:
    """
    Check if a (raw or canonical) biotype is in the keep set.

    Args:
        biotype: Raw or canonical biotype string.
        keep_set: Frozenset of canonical biotype strings (from config).
    """
    return normalize_biotype(biotype) in keep_set


def load_keep_biotypes(config: dict) -> FrozenSet[str]:
    """Load the set of biotypes to keep from config.

    Reads from gene_universe.keep_biotypes.
    GeneCompass paper specifies: protein_coding, lncrna, mirna.
    """
    return frozenset(config["gene_universe"]["keep_biotypes"])


# ============================================================
# ENSEMBL ID NORMALIZATION
# ============================================================

_ENSRNOG_PATTERN = re.compile(r'^ENSRNOG\d{11}$')
_ENSG_PATTERN = re.compile(r'^ENSG\d{11}$')
_ENSMUSG_PATTERN = re.compile(r'^ENSMUSG\d{11}$')


def normalize_ensembl_id(raw_id: str) -> str:
    """
    Strip whitespace, uppercase, remove version suffix.

    >>> normalize_ensembl_id("  ensrnog00000029043.3  ")
    'ENSRNOG00000029043'
    """
    return raw_id.strip().upper().split(".")[0]


def is_rat_ensembl_id(gene_id: str) -> bool:
    """Check if string is a valid rat Ensembl gene ID (unversioned)."""
    return bool(_ENSRNOG_PATTERN.match(normalize_ensembl_id(gene_id)))


def is_human_ensembl_id(gene_id: str) -> bool:
    return bool(_ENSG_PATTERN.match(normalize_ensembl_id(gene_id)))


def is_mouse_ensembl_id(gene_id: str) -> bool:
    return bool(_ENSMUSG_PATTERN.match(normalize_ensembl_id(gene_id)))


# ============================================================
# GENE ID NAMESPACE DETECTION
# ============================================================

class GeneNamespace:
    ENSEMBL_RAT = "ensembl_rat"
    ENSEMBL_HUMAN = "ensembl_human"
    ENSEMBL_MOUSE = "ensembl_mouse"
    RGD = "rgd"
    REFSEQ = "refseq"
    LOC = "loc"
    SYMBOL = "symbol"
    CONTIG = "contig"
    RFAM = "rfam"
    PROBE = "probe"
    GENOMIC_COORD = "genomic_coord"
    UNKNOWN = "unknown"


# All patterns compiled case-insensitive
_NAMESPACE_PATTERNS = [
    (re.compile(r'^ENSRNOG\d{11}$', re.I),         GeneNamespace.ENSEMBL_RAT),
    (re.compile(r'^ENSG\d{11}$', re.I),             GeneNamespace.ENSEMBL_HUMAN),
    (re.compile(r'^ENSMUSG\d{11}$', re.I),          GeneNamespace.ENSEMBL_MOUSE),
    (re.compile(r'^RGD\d+$', re.I),                 GeneNamespace.RGD),
    (re.compile(r'^(NM_|NR_|XM_|XR_)\d+', re.I),   GeneNamespace.REFSEQ),
    (re.compile(r'^LOC\d+$', re.I),                 GeneNamespace.LOC),
    (re.compile(r'^RF\d{5}$', re.I),                GeneNamespace.RFAM),
    (re.compile(r'^AABR\d+', re.I),                 GeneNamespace.CONTIG),
    (re.compile(r'^JACYVU\d+', re.I),               GeneNamespace.CONTIG),
    (re.compile(r'^MU\d{6}$', re.I),                GeneNamespace.CONTIG),
    (re.compile(r'^AC\d{6}', re.I),                 GeneNamespace.CONTIG),
    (re.compile(r'^CM\d{6}', re.I),                 GeneNamespace.CONTIG),
    (re.compile(r'^CHR\d+[_:]\d+', re.I),           GeneNamespace.GENOMIC_COORD),
    (re.compile(r'^\d{7,}_AT$', re.I),              GeneNamespace.PROBE),
]


def detect_namespace(gene_id: str) -> str:
    """
    Detect the namespace of a gene identifier.

    >>> detect_namespace("ENSRNOG00000029043")
    'ensembl_rat'
    >>> detect_namespace("Apoe")
    'symbol'
    """
    base = gene_id.strip().split(".")[0]
    for pattern, namespace in _NAMESPACE_PATTERNS:
        if pattern.match(base):
            return namespace
    if base and base[0].isalpha() and 2 <= len(base) <= 30:
        return GeneNamespace.SYMBOL
    return GeneNamespace.UNKNOWN


# Namespaces that should be excluded early (junk)
JUNK_NAMESPACES = frozenset({
    GeneNamespace.CONTIG,
    GeneNamespace.RFAM,
    GeneNamespace.PROBE,
    GeneNamespace.GENOMIC_COORD,
    GeneNamespace.UNKNOWN,
})


# ============================================================
# NON-GENE PATTERN FILTERING
# ============================================================

_NON_GENE_PATTERNS = [
    (re.compile(r'^AABR\d+', re.I),      "aabr_contig"),
    (re.compile(r'^JACYVU\d+', re.I),     "jacyvu_scaffold"),
    (re.compile(r'^MU\d{6}$', re.I),      "mu_contig"),
    (re.compile(r'^AC\d{6}', re.I),       "ac_contig"),
    (re.compile(r'^CM\d{6}', re.I),       "cm_contig"),
    (re.compile(r'^RF\d{5}$', re.I),      "rfam"),
    (re.compile(r'^\d{7,}_AT$', re.I),    "probe_id"),
    (re.compile(r'^CHR\d+[_:]\d+', re.I), "genomic_coord"),
    (re.compile(r'^U\d+$', re.I),         "small_rna"),
    (re.compile(r'^7SK$', re.I),          "small_rna"),
]


def is_non_gene_pattern(gene_id: str) -> Optional[str]:
    """
    Check if gene ID matches a known non-gene pattern.
    Returns category string or None.

    >>> is_non_gene_pattern("JACYVU010000738")
    'jacyvu_scaffold'
    >>> is_non_gene_pattern("Apoe") is None
    True
    """
    base = gene_id.strip().split(".")[0]
    for pattern, category in _NON_GENE_PATTERNS:
        if pattern.match(base):
            return category
    return None


# ============================================================
# ACCESSION EXTRACTION
# ============================================================

def compile_accession_patterns(config: dict) -> List[re.Pattern]:
    """Compile accession patterns from config (gene_universe section)."""
    return [re.compile(p) for p in config["gene_universe"]["accession_patterns"]]


def extract_accession(
    filename: str,
    patterns: Optional[List[re.Pattern]] = None,
) -> str:
    """
    Extract study accession from h5ad filename.

    Args:
        filename: e.g. "GSE127248_sample0.h5ad"
        patterns: Compiled regex patterns from config. If None,
                  uses default GSE + ArrayExpress patterns.

    Returns:
        Accession string.

    Raises:
        ValueError if no valid accession found.
    """
    if patterns is None:
        patterns = [
            re.compile(r'^GSE\d+$'),
            re.compile(r'^E-\w+-\d+$'),
        ]

    stem = os.path.splitext(os.path.basename(filename))[0]
    parts = stem.split("_")

    # Try each token against each pattern
    for part in parts:
        for pat in patterns:
            if pat.match(part):
                return part

    # Try hyphen-joined prefixes for ArrayExpress (E-MTAB-1234)
    for i in range(2, min(4, len(parts) + 1)):
        candidate = "-".join(parts[:i])
        for pat in patterns:
            if pat.match(candidate):
                return candidate

    raise ValueError(
        f"Cannot extract valid accession from '{filename}'. "
        f"Tokens: {parts}. Patterns: {[p.pattern for p in patterns]}"
    )


# ============================================================
# BIOMART REFERENCE
# ============================================================

class BioMartReference:
    """
    Loaded BioMart reference data for rat genes.

    Attributes:
        ensembl_ids: set of canonical ENSRNOG IDs (uppercase, unversioned)
        symbols_lower: dict {symbol.lower(): ENSRNOG}
        gene_info: dict {ENSRNOG: {symbol, biotype, chromosome, description}}
        metadata: dict with file path, checksum, etc.
    """

    def __init__(self, gene_info_path: str, expected_md5: Optional[str] = None):
        self.gene_info_path = gene_info_path
        self.ensembl_ids: Set[str] = set()
        self.symbols_lower: Dict[str, str] = {}
        self.gene_info: Dict[str, dict] = {}
        self.metadata: Dict[str, str] = {}
        self._load(gene_info_path, expected_md5)

    def _load(self, path: str, expected_md5: Optional[str]):
        actual_md5 = compute_file_checksum(path)
        self.metadata = {
            "file": os.path.basename(path),
            "path": os.path.abspath(path),
            "md5": actual_md5,
        }

        if expected_md5 is not None and actual_md5 != expected_md5:
            raise ValueError(
                f"BioMart checksum mismatch for {path}.\n"
                f"  Expected: {expected_md5}\n"
                f"  Actual:   {actual_md5}\n"
                f"This may indicate a different Ensembl release."
            )

        with open(path) as f:
            reader = csv.reader(f, delimiter="\t")
            header = next(reader)

            # Flexible column detection — handles different BioMart exports
            col_map = {}
            for i, col in enumerate(header):
                cl = col.strip().lower().replace(' ', '_')
                if 'gene_stable_id' in cl or cl in ('ensembl_gene_id', 'gene_id'):
                    col_map['ens'] = i
                elif 'gene_name' in cl or cl in ('symbol', 'external_gene_name'):
                    col_map['sym'] = i
                elif 'gene_type' in cl or 'biotype' in cl:
                    col_map['bio'] = i
                elif 'chromosome' in cl or cl == 'chr':
                    col_map['chr'] = i
                elif 'description' in cl:
                    col_map['desc'] = i

            # Fall back to positional if headers don't match
            if 'ens' not in col_map:
                col_map = {'ens': 0, 'sym': 1, 'bio': 2, 'chr': 3, 'desc': 4}

            for row in reader:
                if len(row) <= col_map['ens']:
                    continue
                gene_id = row[col_map['ens']].strip().upper().split('.')[0]
                if not gene_id.startswith("ENSRNOG"):
                    continue

                symbol = row[col_map.get('sym', 1)].strip() if len(row) > col_map.get('sym', 1) else ""
                if not symbol:
                    symbol = gene_id
                biotype = normalize_biotype(
                    row[col_map.get('bio', 2)].strip() if len(row) > col_map.get('bio', 2) else ""
                )
                chromosome = row[col_map.get('chr', 3)].strip() if len(row) > col_map.get('chr', 3) else ""
                description = row[col_map.get('desc', 4)].strip() if len(row) > col_map.get('desc', 4) else ""

                self.ensembl_ids.add(gene_id)
                self.gene_info[gene_id] = {
                    "symbol": symbol,
                    "biotype": biotype,
                    "chromosome": chromosome,
                    "description": description,
                }
                sym_lower = symbol.lower()
                if sym_lower not in self.symbols_lower:
                    self.symbols_lower[sym_lower] = gene_id

    def contains(self, gene_id: str) -> bool:
        """Check if a (possibly versioned) Ensembl ID is in BioMart."""
        return normalize_ensembl_id(gene_id) in self.ensembl_ids

    def resolve(self, raw_id: str) -> Tuple[Optional[str], str]:
        """
        Resolve a raw gene identifier to a canonical ENSRNOG ID.

        Resolution tiers (BioMart only — RGD is composed externally):
          1. Direct Ensembl match (direct_ensrnog)
          2. Symbol lookup, case-insensitive (biomart_symbol)
          3. Unresolved

        Returns:
            (resolved_id_or_None, method_string)
        """
        base = raw_id.strip().split(".")[0]
        upper = base.upper()

        # Tier 1: Direct ENSRNOG
        if upper.startswith("ENSRNOG") and upper in self.ensembl_ids:
            return upper, "direct_ensrnog"

        # Non-rat Ensembl — reject early
        if upper.startswith("ENS"):
            return None, "non_rat_ensembl"

        # Tier 2: BioMart symbol (case-insensitive)
        lower = base.lower()
        ens_id = self.symbols_lower.get(lower)
        if ens_id is not None:
            return ens_id, "biomart_symbol"

        return None, "unresolved"

    def get_biotype(self, ensembl_id: str) -> Optional[str]:
        info = self.gene_info.get(ensembl_id)
        return info["biotype"] if info else None

    def get_symbol(self, ensembl_id: str) -> Optional[str]:
        info = self.gene_info.get(ensembl_id)
        return info["symbol"] if info else None

    def get_chromosome(self, ensembl_id: str) -> Optional[str]:
        info = self.gene_info.get(ensembl_id)
        return info["chromosome"] if info else None

    def __len__(self):
        return len(self.ensembl_ids)

    def __contains__(self, item):
        return self.contains(item)

    @classmethod
    def from_config(cls, config: dict) -> "BioMartReference":
        """Load BioMartReference using paths and settings from config."""
        path = resolve_path(config, config["biomart"]["rat_gene_info"])
        # md5 key may be omitted entirely or set to empty string — both mean "skip"
        expected_md5 = config["biomart"].get("rat_gene_info_md5") or None
        return cls(str(path), expected_md5)


# ============================================================
# SYMBOL LOOKUP LOADING
# ============================================================

def load_symbol_lookup(pickle_path: str) -> Dict[str, str]:
    """
    Load a symbol lookup pickle, normalizing all keys to lowercase.
    Returns: {symbol_lower: ensembl_id}
    """
    with open(pickle_path, "rb") as f:
        raw = pickle.load(f)

    normalized = {}
    for k, v in raw.items():
        if k is None:
            continue
        key = str(k).strip().lower()
        if not key:
            continue
        normalized[key] = v
    return normalized


# ============================================================
# TIER LABELS (constants — consistent across all scripts)
# ============================================================

class Tier:
    TRIPLET = "triplet"
    HUMAN_RAT = "human_rat"
    MOUSE_RAT = "mouse_rat"
    NEW = "new"
    # Optional low-identity tier (enabled via config)
    LOW_IDENTITY = "low_identity_one2one"


# ============================================================
# GENECOMPASS FORMAT CONVERSION
# ============================================================

def to_genecompass_biotype(canonical: str, config: dict) -> str:
    """
    Convert pipeline canonical biotype to GeneCompass output format.

    Mapping comes from config (reference_assembly.biotype_output_map).
    Default mappings if config key missing:
        protein_coding → protein_coding
        lncrna → lncRNA
        mirna → miRNA
    """
    output_map = config.get("reference_assembly", {}).get("biotype_output_map", {})
    if not output_map:
        # Sensible defaults matching GeneCompass format
        output_map = {
            "protein_coding": "protein_coding",
            "lncrna": "lncRNA",
            "mirna": "miRNA",
        }
    return output_map.get(canonical, canonical)


# ============================================================
# FILE UTILITIES
# ============================================================

def compute_file_checksum(filepath: str, algorithm: str = "md5") -> str:
    """Compute hex digest of a file."""
    h = hashlib.new(algorithm)
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


# ============================================================
# STAGE MANIFEST
# ============================================================

def create_stage_manifest(
    stage_name: str,
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    config: dict,
    stats: dict,
    output_path: str,
) -> dict:
    """
    Write a JSON manifest for a pipeline stage.

    Records: inputs (with checksums), outputs, full config snapshot,
    git commit if available, timestamp, and runtime stats.
    """
    manifest = {
        "stage": stage_name,
        "timestamp": datetime.now().isoformat(),
        "inputs": {},
        "outputs": {},
        "config": {k: v for k, v in config.items() if k != "_project_root"},
        "stats": stats,
    }

    for label, fpath in inputs.items():
        entry = {"path": fpath}
        if os.path.isfile(fpath):
            entry["md5"] = compute_file_checksum(fpath)
            entry["size_bytes"] = os.path.getsize(fpath)
        manifest["inputs"][label] = entry

    for label, fpath in outputs.items():
        entry = {"path": fpath}
        if os.path.isfile(fpath):
            entry["md5"] = compute_file_checksum(fpath)
            entry["size_bytes"] = os.path.getsize(fpath)
        manifest["outputs"][label] = entry

    try:
        import subprocess
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            manifest["git_commit"] = result.stdout.strip()
    except Exception:
        pass

    with open(output_path, "w") as f:
        json.dump(manifest, f, indent=2)

    return manifest


# ============================================================
# CONFIG VALIDATION
# ============================================================

def validate_config(config: dict):
    """
    Validate that required config fields are present and correct.

    Checks all sections used by Stages 2-5:
      - biomart: reference metadata and file paths
      - gene_universe: preprocessing thresholds + gene universe construction
      - rgd: synonym resolution settings
      - orthologs: merge strategy
      - paths: required directory paths

    Raises ValueError with all failures collected.
    """
    errors = []

    # ── BioMart metadata ──
    bm = config.get("biomart", {})
    if not bm.get("ensembl_release"):
        errors.append("biomart.ensembl_release is required")
    if not bm.get("assembly"):
        errors.append("biomart.assembly is required")
    if not bm.get("download_date") or "XX" in str(bm.get("download_date", "")):
        errors.append("biomart.download_date must be filled (not placeholder)")
    if not bm.get("rat_gene_info"):
        errors.append("biomart.rat_gene_info path is required")

    # ── Gene Universe (Stage 2) ──
    gu = config.get("gene_universe", {})

    # BioMart gate is non-negotiable
    if not gu.get("biomart_gate", True):
        errors.append(
            "gene_universe.biomart_gate must be true — "
            "disabling the BioMart gate reintroduces ghost IDs"
        )

    # keep_biotypes — GeneCompass paper: protein_coding + lncrna + mirna
    kb = gu.get("keep_biotypes", [])
    if not kb:
        errors.append("gene_universe.keep_biotypes must list at least one biotype")
    else:
        kb_set = set(b.lower() for b in kb)
        if "lncrna" not in kb_set:
            errors.append(
                "gene_universe.keep_biotypes is missing 'lncrna' — "
                "GeneCompass paper includes lncRNA in core gene list"
            )

    # Step 2 thresholds
    for key in ("min_biomart_match", "min_studies", "accession_patterns"):
        if key not in gu:
            errors.append(f"gene_universe.{key} is required")

    # Step 1 preprocessing thresholds (cell QC)
    pp = gu.get("preprocessing", {})
    for key in ("min_genes_per_cell", "min_cells_per_sample",
                "max_mito_fraction", "outlier_sd_threshold"):
        if key not in pp:
            errors.append(f"gene_universe.preprocessing.{key} is required")

    # Sanity-check preprocessing values if present
    if "min_genes_per_cell" in pp:
        v = pp["min_genes_per_cell"]
        if not isinstance(v, (int, float)) or v < 1:
            errors.append(f"gene_universe.preprocessing.min_genes_per_cell must be >= 1, got {v}")
    if "max_mito_fraction" in pp:
        v = pp["max_mito_fraction"]
        if not isinstance(v, (int, float)) or not (0 < v <= 1):
            errors.append(f"gene_universe.preprocessing.max_mito_fraction must be in (0, 1], got {v}")
    if "outlier_sd_threshold" in pp:
        v = pp["outlier_sd_threshold"]
        if not isinstance(v, (int, float)) or v <= 0:
            errors.append(f"gene_universe.preprocessing.outlier_sd_threshold must be > 0, got {v}")

    # ── RGD ──
    rgd = config.get("rgd", {})
    if rgd.get("use_for_symbol_synonyms") and not rgd.get("require_biomart_resolution", True):
        errors.append(
            "rgd.require_biomart_resolution must be true when "
            "use_for_symbol_synonyms is true — prevents RGD from "
            "creating non-BioMart IDs"
        )

    # ── Orthologs (Stage 3) ──
    ms = config.get("orthologs", {}).get("merge_strategy")
    if ms and ms != "separate_best_then_merge":
        errors.append(
            f"orthologs.merge_strategy must be 'separate_best_then_merge', got '{ms}'"
        )

    # ── Paths ──
    paths = config.get("paths", {})
    for key in ("qc_h5ad_dir", "gene_universe_dir"):
        if key not in paths:
            errors.append(f"paths.{key} is required")

    # ── Reference Assembly (Stage 5) ──
    ra = config.get("reference_assembly", {})
    bom = ra.get("biotype_output_map", {})
    if bom:
        # If map is provided, verify lncrna is included
        if "lncrna" not in bom and kb and "lncrna" in set(b.lower() for b in kb):
            errors.append(
                "reference_assembly.biotype_output_map is missing 'lncrna' entry — "
                "lncrna is in keep_biotypes but has no output format mapping"
            )

    if errors:
        raise ValueError(
            "Config validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        )


def validate_qc_file_count(config: dict, actual_count: int):
    """
    Warn (not fail) if QC file count deviates significantly from expected.
    Call from Stage 2 after scanning the qc_h5ad directory.
    """
    import warnings
    expected = config.get("paths", {}).get("expected_qc_h5ad_files")
    if expected is None:
        return
    deviation = abs(actual_count - expected) / expected
    if deviation > 0.10:
        warnings.warn(
            f"QC file count ({actual_count}) differs from expected ({expected}) "
            f"by {deviation:.0%}. Check qc_h5ad_dir path or update "
            f"paths.expected_qc_h5ad_files in config.",
            stacklevel=2,
        )