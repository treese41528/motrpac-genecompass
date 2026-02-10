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

_BIOTYPE_ALIASES = {
    # protein_coding
    "protein_coding":           "protein_coding",
    "protein-coding":           "protein_coding",
    "protein coding":           "protein_coding",
    "protein-coding gene":      "protein_coding",
    # mirna
    "mirna":                    "mirna",
    "micro_rna":                "mirna",
    # lncrna
    "lncrna":                   "lncrna",
    "lincrna":                  "lncrna",
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
    # pseudogene
    "pseudo":                   "pseudogene",
    "pseudogene":               "pseudogene",
    "processed_pseudogene":     "pseudogene",
    "unprocessed_pseudogene":   "pseudogene",
    # ncrna
    "ncrna":                    "ncrna",
    # other
    "antisense":                "antisense",
    "gene":                     "gene",
}


def normalize_biotype(raw: str) -> str:
    """
    Canonicalize a biotype string to lowercase, underscore-separated form.

    Uses exact lookup, then fuzzy matching for RGD variants.

    >>> normalize_biotype("protein-coding")
    'protein_coding'
    >>> normalize_biotype("protein-coding gene")
    'protein_coding'
    >>> normalize_biotype("miRNA")
    'mirna'
    """
    key = raw.strip().lower()

    # Exact lookup
    canonical = _BIOTYPE_ALIASES.get(key)
    if canonical is not None:
        return canonical

    # Fuzzy matching for RGD variants
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

    return key.replace("-", "_").replace(" ", "_")


def is_kept_biotype(biotype: str, keep_set: FrozenSet[str]) -> bool:
    """
    Check if a (raw or canonical) biotype is in the keep set.

    Args:
        biotype: Raw or canonical biotype string.
        keep_set: Frozenset of canonical biotype strings (from config).
    """
    return normalize_biotype(biotype) in keep_set


def load_keep_biotypes(config: dict) -> FrozenSet[str]:
    """Load the set of biotypes to keep from config."""
    return frozenset(config["vocabulary"]["keep_biotypes"])


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
    """Compile accession patterns from config."""
    return [re.compile(p) for p in config["gene_inventory"]["accession_patterns"]]


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
            for row in reader:
                if len(row) < 4:
                    continue
                gene_id = row[0].strip()
                if not gene_id.startswith("ENSRNOG"):
                    continue

                symbol = row[1].strip() if row[1].strip() else gene_id
                biotype = normalize_biotype(row[2])
                chromosome = row[3].strip()
                description = row[4].strip() if len(row) > 4 else ""

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

        Resolution tiers:
          1. Direct Ensembl match (native_ensembl)
          2. Symbol lookup, case-insensitive (symbol_lookup)
          3. Unresolved

        Returns:
            (resolved_id_or_None, method_string)
        """
        base = raw_id.strip().split(".")[0]
        upper = base.upper()

        if upper.startswith("ENSRNOG") and upper in self.ensembl_ids:
            return upper, "native_ensembl"

        lower = base.lower()
        ens_id = self.symbols_lower.get(lower)
        if ens_id is not None:
            return ens_id, "symbol_lookup"

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
    Mapping comes from config, not hardcoded.
    """
    output_map = config.get("reference_files", {}).get("biotype_output_map", {})
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
    Validate that required config fields are present and non-placeholder.
    Raises ValueError on failures.
    """
    errors = []

    # BioMart metadata must be filled
    bm = config.get("biomart", {})
    if not bm.get("ensembl_release"):
        errors.append("biomart.ensembl_release is required")
    if not bm.get("assembly"):
        errors.append("biomart.assembly is required")
    if not bm.get("download_date") or "XX" in str(bm.get("download_date", "")):
        errors.append("biomart.download_date must be filled (not placeholder)")

    # BioMart gate must be on
    gi = config.get("gene_inventory", {})
    if not gi.get("biomart_gate", True):
        errors.append(
            "gene_inventory.biomart_gate must be true — "
            "disabling the BioMart gate reintroduces ghost IDs"
        )

    # RGD: if synonyms enabled, require_biomart_resolution must also be on
    rgd = config.get("rgd", {})
    if rgd.get("use_for_symbol_synonyms") and not rgd.get("require_biomart_resolution", True):
        errors.append(
            "rgd.require_biomart_resolution must be true when "
            "use_for_symbol_synonyms is true — prevents RGD from "
            "creating non-BioMart IDs"
        )

    # Vocabulary keep_biotypes must be present
    vk = config.get("vocabulary", {}).get("keep_biotypes", [])
    if not vk:
        errors.append("vocabulary.keep_biotypes must list at least one biotype")

    # Ortholog merge strategy
    ms = config.get("orthologs", {}).get("merge_strategy")
    if ms and ms != "separate_best_then_merge":
        errors.append(
            f"orthologs.merge_strategy must be 'separate_best_then_merge', got '{ms}'"
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