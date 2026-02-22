#!/usr/bin/env python3
"""
build_gene_universe.py — Stage 2, Step 1: Scan → Resolve → Prune → gene_universe.tsv

Pipeline position:
    Stage 1: Data Harvesting & QC
    Stage 2: Gene Universe
      Step 1: build_gene_universe.py                             ← THIS SCRIPT
      Step 2: preprocess_training_matrices.py (cell QC → h5ad)
      Step 3: prune_gene_universe.py (expression-based pruning)
    Stage 3: Ortholog Mapping
    Stage 4: Gene Medians (SLURM)
    Stage 5: Reference Assembly & Corpus Export

Purpose:
    Scan ALL raw matrices across the training corpus (reading only var_names —
    no cell data loaded), resolve every gene identifier to a canonical ENSRNOG
    via BioMart + RGD, prune by minimum study frequency and biotype.

    This produces the authoritative gene universe that Step 2 consumes.
    No chicken-and-egg: gene list is built from raw var_names, not from
    preprocessed data.

Key properties:
    - Deterministic: same inputs → same outputs regardless of pipeline state.
    - Fast: reads only var_names from raw matrices, never loads expression data.
    - Single resolution point: gene ID resolution happens HERE only.
      Step 2 uses gene_resolution.tsv as a pre-built lookup table.
    - BioMart-gated: every gene in the output exists in BioMart. Period.

Outputs:
    gene_universe.tsv      — One row per kept ENSRNOG (the canonical gene list)
    gene_resolution.tsv    — One row per unique raw gene ID (audit log)
    study_coverage.tsv     — Per-study resolution statistics

Usage:
    python pipeline/02_gene_universe/build_gene_universe.py
    python pipeline/02_gene_universe/build_gene_universe.py --dry-run
    python pipeline/02_gene_universe/build_gene_universe.py -v

Author: Tim Reese Lab / Claude
Date: February 2026
"""

import csv
import gzip
import hashlib
import json
import logging
import os
import random
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import h5py

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'lib'))
from gene_utils import load_config, resolve_path

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1: BIOMART GATE
# ═════════════════════════════════════════════════════════════════════════════

class BioMartGate:
    """The canonical gene universe is defined by BioMart. Period.

    Any gene ID not present in this gate does not exist for our pipeline.
    RGD is used ONLY to widen the symbol→ENSRNOG resolution funnel.
    """

    def __init__(self, gene_info_path: Path):
        self.ensembl_ids: Set[str] = set()
        self.symbol_to_id: Dict[str, str] = {}
        self.id_to_symbol: Dict[str, str] = {}
        self.id_to_biotype: Dict[str, str] = {}
        self._load(gene_info_path)

    def _load(self, path: Path):
        with open(path, 'r') as f:
            reader = csv.DictReader(f, delimiter='\t')
            col_map = {}
            for col in reader.fieldnames:
                cl = col.strip().lower().replace(' ', '_')
                if 'gene_stable_id' in cl or cl in ('ensembl_gene_id', 'gene_id'):
                    col_map['ens'] = col
                elif 'gene_name' in cl or cl in ('symbol', 'external_gene_name'):
                    col_map['sym'] = col
                elif 'gene_type' in cl or 'biotype' in cl:
                    col_map['bio'] = col

            if 'ens' not in col_map:
                raise ValueError(f"Cannot find Ensembl ID column in {path}. "
                                 f"Headers: {reader.fieldnames}")

            for row in reader:
                ens_raw = row.get(col_map['ens'], '').strip()
                if not ens_raw:
                    continue
                ens_id = ens_raw.split('.')[0].upper()
                if not ens_id.startswith('ENSRNOG'):
                    continue

                symbol = row.get(col_map.get('sym', ''), '').strip()
                biotype = row.get(col_map.get('bio', ''), '').strip().lower()

                self.ensembl_ids.add(ens_id)
                self.id_to_biotype[ens_id] = biotype
                if symbol:
                    self.id_to_symbol[ens_id] = symbol
                    sym_lower = symbol.lower()
                    if sym_lower not in self.symbol_to_id:
                        self.symbol_to_id[sym_lower] = ens_id

        logger.info(f"BioMart gate: {len(self.ensembl_ids):,} IDs, "
                     f"{len(self.symbol_to_id):,} symbols from {path.name}")

    def contains(self, eid: str) -> bool:
        return eid in self.ensembl_ids

    def biotype(self, eid: str) -> str:
        return self.id_to_biotype.get(eid, '')

    def symbol(self, eid: str) -> str:
        return self.id_to_symbol.get(eid, '')

    def resolve_symbol(self, sym: str) -> Optional[str]:
        return self.symbol_to_id.get(sym.lower())


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2: RGD SYNONYM RESOLVER
# ═════════════════════════════════════════════════════════════════════════════

class RGDSynonymResolver:
    """Resolve symbol synonyms → ENSRNOG via RGD GENES_RAT.txt.

    Every resolution MUST terminate at a BioMart ENSRNOG ID.
    RGD never creates new IDs, provides biotypes, or overrides BioMart.
    """

    def __init__(self, genes_file: Path, gate: BioMartGate):
        self.synonym_to_ensembl: Dict[str, str] = {}
        self._load(genes_file, gate)

    def _load(self, path: Path, gate: BioMartGate):
        if not path.exists():
            logger.warning(f"RGD genes file not found: {path}")
            return

        n_synonyms = 0
        with open(path, 'r', errors='replace') as f:
            header_line = None
            for line in f:
                if line.startswith('#'):
                    continue
                header_line = line.strip()
                break

            if not header_line:
                return

            headers = header_line.split('\t')
            col_idx = {}
            for i, h in enumerate(headers):
                hl = h.strip().upper()
                if hl == 'SYMBOL':
                    col_idx['symbol'] = i
                elif hl == 'OLD_SYMBOL':
                    col_idx['old_symbol'] = i
                elif hl == 'ENSEMBL_ID':
                    col_idx['ensembl'] = i

            if 'ensembl' not in col_idx:
                logger.warning(f"RGD file missing ENSEMBL_ID column")
                return

            ens_col = col_idx['ensembl']
            sym_col = col_idx.get('symbol')
            old_col = col_idx.get('old_symbol')
            max_col = max(c for c in [ens_col, sym_col, old_col] if c is not None)

            for line in f:
                if line.startswith('#'):
                    continue
                parts = line.strip().split('\t')
                if len(parts) <= max_col:
                    continue

                ens_raw = parts[ens_col].strip()
                if not ens_raw:
                    continue

                candidates = [e.strip().split('.')[0].upper()
                              for e in ens_raw.split(';') if e.strip()]
                valid = [e for e in candidates if gate.contains(e)]
                if len(valid) != 1:
                    continue
                target = valid[0]

                for col in [sym_col, old_col]:
                    if col is None:
                        continue
                    raw = parts[col].strip()
                    for alias in raw.split(';'):
                        alias = alias.strip()
                        if alias:
                            key = alias.lower()
                            if key not in self.synonym_to_ensembl:
                                self.synonym_to_ensembl[key] = target
                                n_synonyms += 1

        logger.info(f"RGD synonyms: {n_synonyms:,} mappings (all BioMart-validated)")

    def resolve(self, sym: str) -> Optional[str]:
        return self.synonym_to_ensembl.get(sym.lower())


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3: GENE ID RESOLUTION
# ═════════════════════════════════════════════════════════════════════════════

NON_GENE_PATTERNS = [
    re.compile(r'^\d+$'),
    re.compile(r'^(\d+|[XYM]):(\d+)-(\d+)$'),
    re.compile(r'^chr', re.IGNORECASE),
    re.compile(r'^RF\d{5}$'),
    re.compile(r'^[A-Z]{2}\d{6,8}$'),
    re.compile(r'^JACYVU\d+$'),
    re.compile(r'^AABR\d+$'),
    re.compile(r'^ENSMUSG', re.IGNORECASE),
    re.compile(r'^ENSG\d', re.IGNORECASE),
    re.compile(r'^\d+_(at|s_at|x_at)$'),
    re.compile(r'^ILMN_\d+$'),
]


def resolve_gene_id(raw_id: str, gate: BioMartGate,
                    rgd: Optional[RGDSynonymResolver]) -> Tuple[Optional[str], str]:
    """Resolve a single gene ID through the deterministic chain.

    Chain: direct ENSRNOG → BioMart symbol → RGD synonym → reject.
    Returns (ensembl_id_or_None, method_string).
    """
    raw = raw_id.strip()
    if not raw:
        return None, 'empty'

    for pat in NON_GENE_PATTERNS:
        if pat.match(raw):
            return None, 'non_gene'

    upper = raw.upper()

    # 1. Direct ENSRNOG
    if upper.startswith('ENSRNOG'):
        base = upper.split('.')[0]
        if gate.contains(base):
            return base, 'direct_ensrnog'
        return None, 'not_in_biomart'

    # Non-rat Ensembl
    if upper.startswith('ENS'):
        return None, 'non_rat_ensembl'

    # 2. BioMart symbol
    ens = gate.resolve_symbol(raw)
    if ens is not None:
        return ens, 'biomart_symbol'

    # 3. RGD synonym
    if rgd is not None:
        ens = rgd.resolve(raw)
        if ens is not None:
            return ens, 'rgd_synonym'

    return None, 'unresolved'


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4: FAST VAR_NAME EXTRACTION (multi-format, no cell data)
# ═════════════════════════════════════════════════════════════════════════════

# File patterns to skip
SKIP_FILE_PATTERNS = [
    'barcodes', 'genes', 'features', 'metadata', 'clusters',
    'umap', 'tsne', 'pca', 'neighbors', 'annotation',
    'peaks', 'fragments', 'atac',
    'samples', 'sra_runs', 'sample_files',
    'family.soft', 'miniml', 'series_matrix',
    'manifest', 'filelist', 'readme', 'changelog',
    'sdrf', 'idf', 'ena_runs',
]

MATRIX_FILE_PATTERNS = [
    'matrix', 'counts', 'expression', 'umi', 'raw_count',
    'filtered_feature_bc_matrix', 'raw_feature_bc_matrix',
    'gene_expression', 'dgematrix', 'countmatrix',
]


def read_var_names_h5ad(filepath: Path) -> List[str]:
    """Read var_names from h5ad via h5py (no matrix load)."""
    try:
        with h5py.File(filepath, 'r') as f:
            if 'var' not in f:
                return []
            for key in ['_index', 'index', 'gene_ids', 'gene_id', 'gene_name']:
                if key in f['var']:
                    raw = f['var'][key][:]
                    return [x.decode() if isinstance(x, bytes) else str(x) for x in raw]
    except Exception as e:
        logger.debug(f"Failed to read var_names from h5ad {filepath}: {e}")
    return []


def read_var_names_mtx(filepath: Path) -> List[str]:
    """Read gene IDs from 10x MTX companion features/genes file."""
    parent = filepath.parent

    # Find features/genes file (same directory or parent)
    candidates = []
    for d in [parent, parent.parent]:
        for name in ['features.tsv.gz', 'features.tsv', 'genes.tsv.gz', 'genes.tsv']:
            candidates.append(d / name)

    # Also look for GSM-prefixed companion files
    stem = filepath.name
    for suffix in ['.matrix.mtx.gz', '.matrix.mtx', '_matrix.mtx.gz', '_matrix.mtx',
                   '.mtx.gz', '.mtx']:
        if stem.lower().endswith(suffix):
            prefix = stem[:len(stem) - len(suffix)]
            for d in [parent]:
                for name in ['features.tsv.gz', 'genes.tsv.gz', 'features.tsv', 'genes.tsv']:
                    # e.g., GSM123_sample.features.tsv.gz
                    candidates.append(d / f"{prefix}.{name}")
                    candidates.append(d / f"{prefix}_{name}")
            break

    for feat_path in candidates:
        if feat_path.exists():
            try:
                opener = gzip.open if str(feat_path).endswith('.gz') else open
                with opener(feat_path, 'rt') as f:
                    gene_ids = []
                    for line in f:
                        parts = line.strip().split('\t')
                        if parts:
                            gene_ids.append(parts[0])
                    return gene_ids
            except Exception as e:
                logger.debug(f"Failed to read features from {feat_path}: {e}")

    return []


def read_var_names_h5(filepath: Path) -> List[str]:
    """Read gene IDs from 10x HDF5 (.h5) file."""
    try:
        with h5py.File(filepath, 'r') as f:
            # 10x HDF5 structure: matrix/features/id or matrix/features/name
            for path_chain in [
                ['matrix', 'features', 'id'],
                ['matrix', 'features', 'name'],
                ['matrix', 'genes'],
            ]:
                node = f
                found = True
                for key in path_chain:
                    if key in node:
                        node = node[key]
                    else:
                        found = False
                        break
                if found:
                    raw = node[:]
                    return [x.decode() if isinstance(x, bytes) else str(x) for x in raw]
    except Exception as e:
        logger.debug(f"Failed to read var_names from H5 {filepath}: {e}")
    return []


def read_var_names_loom(filepath: Path) -> List[str]:
    """Read gene IDs from Loom file via h5py."""
    try:
        with h5py.File(filepath, 'r') as f:
            for key in ['row_attrs/Gene', 'row_attrs/gene_name',
                        'row_attrs/Accession', 'row_attrs/gene_id']:
                parts = key.split('/')
                node = f
                found = True
                for p in parts:
                    if p in node:
                        node = node[p]
                    else:
                        found = False
                        break
                if found:
                    raw = node[:]
                    return [x.decode() if isinstance(x, bytes) else str(x) for x in raw]
    except Exception as e:
        logger.debug(f"Failed to read var_names from loom {filepath}: {e}")
    return []


def read_var_names_tabular(filepath: Path) -> List[str]:
    """Read gene IDs (first column / header) from TSV/CSV."""
    try:
        opener = gzip.open if str(filepath).endswith('.gz') else open
        mode = 'rt' if str(filepath).endswith('.gz') else 'r'
        with opener(filepath, mode) as f:
            # Read header to determine if genes are rows or columns
            header = f.readline().strip()
            sep = '\t' if '\t' in header else ','
            cols = header.split(sep)

            # Check if first row values look like gene IDs
            first_data = f.readline().strip()
            if not first_data:
                return cols[1:]  # Assume genes are columns, skip index

            first_vals = first_data.split(sep)

            # Heuristic: if first column of data row looks like a gene ID
            # then genes are rows (need to read all row indices)
            first_cell = first_vals[0].strip()
            if (first_cell.startswith('ENSRNOG') or
                first_cell.startswith('ENSMUSG') or
                first_cell.startswith('ENSG') or
                (first_cell and first_cell[0].isupper() and
                 not first_cell.replace('.', '').replace('-', '').isdigit())):
                # Genes are rows — collect first column
                gene_ids = [first_cell]
                for line in f:
                    parts = line.strip().split(sep)
                    if parts:
                        gene_ids.append(parts[0].strip())
                return gene_ids
            else:
                # Genes are columns
                return cols[1:]

    except Exception as e:
        logger.debug(f"Failed to read var_names from tabular {filepath}: {e}")
    return []


def detect_format(filepath: Path) -> Optional[str]:
    """Detect matrix format from file extension."""
    name = filepath.name.lower()
    if name.endswith('.h5ad'):
        return 'h5ad'
    if name.endswith('.mtx.gz') or name.endswith('.mtx'):
        return 'mtx'
    if name.endswith('.h5') or name.endswith('.hdf5'):
        return 'h5'
    if name.endswith('.loom'):
        return 'loom'
    if name.endswith(('.tsv.gz', '.tsv', '.csv.gz', '.csv', '.txt.gz', '.txt')):
        if any(p in name for p in MATRIX_FILE_PATTERNS):
            return 'tabular'
        if name.startswith('gsm'):
            return 'tabular'
    return None


def read_var_names_fast(filepath: Path) -> List[str]:
    """Dispatch to format-specific fast var_name reader."""
    fmt = detect_format(filepath)
    if fmt == 'h5ad':
        return read_var_names_h5ad(filepath)
    elif fmt == 'mtx':
        return read_var_names_mtx(filepath)
    elif fmt == 'h5':
        return read_var_names_h5(filepath)
    elif fmt == 'loom':
        return read_var_names_loom(filepath)
    elif fmt == 'tabular':
        return read_var_names_tabular(filepath)
    return []


def discover_matrix_files(source_dirs: List[Path]) -> List[Tuple[Path, str]]:
    """Discover all matrix files across source directories.

    Returns (filepath, format) tuples. Deduplicates MTX files per directory
    (one standard 10x triplet per directory).
    """
    results = []
    seen_mtx_dirs = set()
    seen_mtx_prefixes = set()

    for source_dir in source_dirs:
        if not source_dir.exists():
            logger.warning(f"Source directory not found: {source_dir}")
            continue

        for root, dirs, files in os.walk(source_dir):
            root_path = Path(root)
            for fname in files:
                fpath = root_path / fname
                fname_lower = fname.lower()

                # Skip metadata/annotation files
                if any(skip in fname_lower for skip in SKIP_FILE_PATTERNS):
                    continue

                fmt = detect_format(fpath)
                if fmt is None:
                    continue

                if fmt == 'mtx':
                    # Dedup: one per directory (standard) or per GSM prefix
                    is_gsm = bool(re.match(
                        r'^(GSM\d+[_.].*|E-\w+-\d+[_.].*)[._](matrix[._])?mtx',
                        fname, re.IGNORECASE))
                    if is_gsm:
                        prefix = fname
                        for sfx in ['.matrix.mtx.gz', '.matrix.mtx', '.mtx.gz', '.mtx']:
                            if fname_lower.endswith(sfx):
                                prefix = fname[:len(fname) - len(sfx)]
                                break
                        key = f"{root_path}::{prefix}"
                        if key not in seen_mtx_prefixes:
                            seen_mtx_prefixes.add(key)
                            results.append((fpath, fmt))
                    else:
                        if str(root_path) not in seen_mtx_dirs:
                            seen_mtx_dirs.add(str(root_path))
                            results.append((fpath, fmt))
                else:
                    results.append((fpath, fmt))

    return results


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 5: ACCESSION EXTRACTION
# ═════════════════════════════════════════════════════════════════════════════

def extract_accession_from_path(filepath: Path, patterns: List[re.Pattern],
                                fail_hard: bool = True) -> Optional[str]:
    """Extract study accession from file path.

    Walks up the directory tree looking for a directory name matching
    accession patterns (GSE..., E-MTAB-..., etc.).
    Also checks the filename stem.
    """
    # Check directory names (most reliable)
    for parent in filepath.parents:
        name = parent.name
        for pat in patterns:
            if pat.match(name):
                return name

    # Check filename stem
    stem = filepath.stem
    if stem.endswith('.mtx'):
        stem = stem[:-4]  # Remove .mtx from .mtx.gz stems

    # Try full stem
    for pat in patterns:
        if pat.match(stem):
            return stem

    # Try extracting GSE/E-MTAB prefix from GSM-prefixed files
    m = re.match(r'^(GSE\d+)', stem, re.IGNORECASE)
    if m:
        return m.group(1)

    # Look for accession in the path components
    for part in filepath.parts:
        for pat in patterns:
            if pat.match(part):
                return part

    if fail_hard:
        raise ValueError(f"Path '{filepath}' matches no accession pattern")
    return None


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 6: MAIN PIPELINE
# ═════════════════════════════════════════════════════════════════════════════

def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


def _git_hash() -> str:
    try:
        import subprocess
        r = subprocess.run(['git', 'rev-parse', 'HEAD'],
                           capture_output=True, text=True, timeout=5)
        return r.stdout.strip() if r.returncode == 0 else 'unknown'
    except Exception:
        return 'unknown'


def _resolve_source_dirs(config: dict) -> List[Path]:
    """Resolve all raw data source directories from config."""
    h = config.get('harvesting', {})
    data_root = resolve_path(config, h.get('data_root', 'data/raw'))
    sources = h.get('sources', {})
    dirs = []

    for source_name, source_cfg in sources.items():
        for data_type, type_cfg in source_cfg.items():
            if not isinstance(type_cfg, dict):
                continue
            if not type_cfg.get('enabled', True):
                continue
            rel_path = type_cfg.get('path', '')
            if rel_path:
                full = data_root / rel_path
                if full.exists():
                    dirs.append(full)
                else:
                    logger.warning(f"Source dir not found: {full}")

    if not dirs:
        logger.warning(f"No source dirs found, falling back to data_root: {data_root}")
        dirs = [data_root]

    return dirs


def build_gene_universe(config: dict, dry_run: bool = False):
    """Main pipeline: scan raw matrices → resolve → prune → output."""
    t_start = time.time()

    gu = config['gene_universe']
    bm = config['biomart']
    rgd_cfg = config.get('rgd', {})
    paths = config['paths']

    assert gu.get('biomart_gate', True), "BioMart gate is non-negotiable"

    min_biomart_match = float(gu['min_biomart_match'])
    match_sample_size = int(gu.get('match_sample_size', 1000))
    match_sample_seed = int(gu.get('match_sample_seed', 0))
    min_studies = int(gu['min_studies'])
    keep_biotypes = set(b.lower() for b in gu['keep_biotypes'])
    fail_on_bad_accession = gu.get('fail_on_unrecognized_accession', True)
    accession_patterns = [re.compile(p) for p in gu['accession_patterns']]

    output_dir = resolve_path(config, paths['gene_universe_dir'])
    gene_info_path = resolve_path(config, bm['rat_gene_info'])

    if not gene_info_path.exists():
        logger.error(f"BioMart gene info not found: {gene_info_path}")
        sys.exit(1)

    # Resolve raw source directories
    source_dirs = _resolve_source_dirs(config)
    logger.info(f"Source directories: {[str(d) for d in source_dirs]}")
    logger.info(f"Config: min_studies={min_studies}, keep_biotypes={keep_biotypes}, "
                f"min_biomart_match={min_biomart_match}")

    if dry_run:
        logger.info("DRY RUN — config validated, exiting")
        return

    # ── Load references ──
    gate = BioMartGate(gene_info_path)

    rgd_resolver = None
    if rgd_cfg.get('use_for_symbol_synonyms', False):
        rgd_path = resolve_path(config, rgd_cfg['genes_file'])
        if rgd_path.exists():
            rgd_resolver = RGDSynonymResolver(rgd_path, gate)

    # ── Phase A: Discover & Scan ──
    logger.info("=" * 60)
    logger.info("PHASE A: Discover & Scan raw matrices")
    logger.info("=" * 60)

    matrix_files = discover_matrix_files(source_dirs)
    logger.info(f"Discovered {len(matrix_files)} matrix files across {len(source_dirs)} source dirs")

    gene_to_studies: Dict[str, Set[str]] = defaultdict(set)
    gene_to_files: Dict[str, int] = defaultdict(int)
    resolution_log: Dict[str, Dict] = {}  # raw_id → {resolved_id, method, ...}
    study_stats: List[Dict] = []
    usable_studies: Set[str] = set()
    unusable_studies: Dict[str, str] = {}
    method_counts = defaultdict(int)
    rng = random.Random(match_sample_seed)

    for file_idx, (matrix_path, fmt) in enumerate(matrix_files):
        try:
            accession = extract_accession_from_path(
                matrix_path, accession_patterns, fail_hard=fail_on_bad_accession)
        except ValueError as e:
            logger.warning(str(e))
            if fail_on_bad_accession:
                continue
            continue

        if accession is None:
            continue

        # Fast var_name extraction (no cell data loaded)
        var_names = read_var_names_fast(matrix_path)
        n_raw = len(var_names)

        if n_raw == 0:
            study_stats.append({
                'accession': accession, 'file': matrix_path.name,
                'format': fmt, 'n_genes_raw': 0, 'n_genes_resolved': 0,
                'match_rate': 0.0, 'usable': False, 'reason': 'empty_var_names',
            })
            continue

        # Sample for match rate estimation
        sample_ids = (rng.sample(var_names, match_sample_size)
                      if n_raw > match_sample_size else var_names)

        n_resolved_sample = sum(
            1 for rid in sample_ids
            if resolve_gene_id(rid, gate, rgd_resolver)[0] is not None
        )
        match_rate = n_resolved_sample / len(sample_ids)

        if match_rate < min_biomart_match:
            unusable_studies[f"{accession}::{matrix_path.name}"] = f"match_rate_{match_rate:.3f}"
            study_stats.append({
                'accession': accession, 'file': matrix_path.name,
                'format': fmt, 'n_genes_raw': n_raw, 'n_genes_resolved': 0,
                'match_rate': round(match_rate, 4), 'usable': False,
                'reason': f'below_threshold_{min_biomart_match}',
            })
            continue

        usable_studies.add(accession)

        # Full resolution pass
        resolved_in_file: Set[str] = set()
        for raw_id in var_names:
            ens_id, method = resolve_gene_id(raw_id, gate, rgd_resolver)
            method_counts[method] += 1

            # Deduplicate resolution log by raw_id
            if raw_id not in resolution_log:
                resolution_log[raw_id] = {
                    'raw_id': raw_id,
                    'resolved_ensembl_id': ens_id or '',
                    'resolution_method': method,
                    'biotype': gate.biotype(ens_id) if ens_id else '',
                    'studies': set(),
                }
            if ens_id:
                resolution_log[raw_id]['studies'].add(accession)

            if ens_id is not None:
                resolved_in_file.add(ens_id)
                gene_to_files[ens_id] += 1

        for ens_id in resolved_in_file:
            gene_to_studies[ens_id].add(accession)

        study_stats.append({
            'accession': accession, 'file': matrix_path.name,
            'format': fmt, 'n_genes_raw': n_raw,
            'n_genes_resolved': len(resolved_in_file),
            'match_rate': round(match_rate, 4), 'usable': True, 'reason': '',
        })

        if (file_idx + 1) % 100 == 0 or file_idx + 1 == len(matrix_files):
            logger.info(f"  [{file_idx+1}/{len(matrix_files)}] "
                        f"{len(usable_studies)} usable studies, "
                        f"{len(gene_to_studies):,} unique genes")

    logger.info(f"Phase A: {len(usable_studies)} usable studies, "
                f"{len(unusable_studies)} unusable files, "
                f"{len(gene_to_studies):,} unique resolved genes")
    logger.info(f"Resolution: {dict(method_counts)}")

    # ── Phase B: Resolve — compute per-raw-ID study counts and kept status ──
    logger.info("=" * 60)
    logger.info("PHASE B: Resolve & annotate")
    logger.info("=" * 60)

    for raw_id, info in resolution_log.items():
        ens_id = info['resolved_ensembl_id']
        info['n_studies'] = len(info['studies'])
        info['study_list'] = ';'.join(sorted(info['studies']))
        del info['studies']  # Don't write set to TSV

        if not ens_id:
            info['kept'] = False
            info['exclusion_reason'] = info['resolution_method']
        elif ens_id not in gene_to_studies:
            info['kept'] = False
            info['exclusion_reason'] = 'no_study_assignment'
        else:
            n_studies = len(gene_to_studies[ens_id])
            bt = gate.biotype(ens_id)
            if n_studies < min_studies:
                info['kept'] = False
                info['exclusion_reason'] = f'below_min_studies_{n_studies}'
            elif bt not in keep_biotypes:
                info['kept'] = False
                info['exclusion_reason'] = f'excluded_biotype_{bt}'
            else:
                info['kept'] = True
                info['exclusion_reason'] = ''

    # ── Phase C: Prune ──
    logger.info("=" * 60)
    logger.info("PHASE C: Prune")
    logger.info("=" * 60)

    n_before = len(gene_to_studies)

    # Filter 1: min_studies
    n_rm_studies = 0
    after_studies = {}
    for eid, studies in gene_to_studies.items():
        if len(studies) >= min_studies:
            after_studies[eid] = studies
        else:
            n_rm_studies += 1
    logger.info(f"min_studies >= {min_studies}: {len(after_studies):,} kept, "
                f"{n_rm_studies:,} removed")

    # Filter 2: biotype
    n_rm_biotype = 0
    biotype_excluded = defaultdict(int)
    final_genes = {}
    for eid, studies in after_studies.items():
        bt = gate.biotype(eid)
        if bt in keep_biotypes:
            final_genes[eid] = studies
        else:
            n_rm_biotype += 1
            biotype_excluded[bt or 'unknown'] += 1
    logger.info(f"biotype in {keep_biotypes}: {len(final_genes):,} kept, "
                f"{n_rm_biotype:,} removed")

    for bt, cnt in sorted(biotype_excluded.items(), key=lambda x: -x[1])[:10]:
        logger.info(f"  excluded: {bt:30s} {cnt:>6,}")

    # ── Phase D: Output ──
    logger.info("=" * 60)
    logger.info("PHASE D: Output")
    logger.info("=" * 60)

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. gene_universe.tsv
    universe_path = output_dir / 'gene_universe.tsv'
    rows = []
    for eid in sorted(final_genes):
        rows.append({
            'ensembl_id': eid,
            'symbol': gate.symbol(eid),
            'biotype': gate.biotype(eid),
            'n_studies': len(final_genes[eid]),
            'study_list': ';'.join(sorted(final_genes[eid])),
        })
    with open(universe_path, 'w', newline='') as f:
        w = csv.DictWriter(f, delimiter='\t',
                           fieldnames=['ensembl_id', 'symbol', 'biotype',
                                       'n_studies', 'study_list'])
        w.writeheader()
        w.writerows(rows)
    logger.info(f"gene_universe.tsv: {len(rows):,} genes")

    # 2. gene_resolution.tsv — one row per unique raw gene ID
    resolution_path = output_dir / 'gene_resolution.tsv'
    res_rows = sorted(resolution_log.values(), key=lambda x: x['raw_id'])
    with open(resolution_path, 'w', newline='') as f:
        w = csv.DictWriter(f, delimiter='\t',
                           fieldnames=['raw_id', 'resolved_ensembl_id',
                                       'resolution_method', 'biotype',
                                       'n_studies', 'kept', 'exclusion_reason',
                                       'study_list'])
        w.writeheader()
        w.writerows(res_rows)
    logger.info(f"gene_resolution.tsv: {len(res_rows):,} unique raw IDs")

    # 3. study_coverage.tsv
    study_path = output_dir / 'study_coverage.tsv'
    with open(study_path, 'w', newline='') as f:
        w = csv.DictWriter(f, delimiter='\t',
                           fieldnames=['accession', 'file', 'format',
                                       'n_genes_raw', 'n_genes_resolved',
                                       'match_rate', 'usable', 'reason'])
        w.writeheader()
        w.writerows(study_stats)
    logger.info(f"study_coverage.tsv: {len(study_stats)} files")

    # 4. manifest.json
    elapsed = time.time() - t_start
    biotype_dist = defaultdict(int)
    for eid in final_genes:
        biotype_dist[gate.biotype(eid)] += 1

    study_counts = [len(s) for s in final_genes.values()]
    freq_dist = {
        '1_study': sum(1 for c in study_counts if c == 1),
        '2-5_studies': sum(1 for c in study_counts if 2 <= c <= 5),
        '6-20_studies': sum(1 for c in study_counts if 6 <= c <= 20),
        '21-50_studies': sum(1 for c in study_counts if 21 <= c <= 50),
        '50+_studies': sum(1 for c in study_counts if c > 50),
    }

    manifest = {
        'stage': 'stage2_step1',
        'script': 'build_gene_universe.py',
        'timestamp': datetime.now().isoformat(),
        'git_hash': _git_hash(),
        'elapsed_seconds': round(elapsed, 1),
        'config': {
            'biomart_release': bm.get('ensembl_release'),
            'biomart_assembly': bm.get('assembly'),
            'min_biomart_match': min_biomart_match,
            'min_studies': min_studies,
            'keep_biotypes': sorted(keep_biotypes),
            'rgd_synonyms_enabled': rgd_resolver is not None,
        },
        'inputs': {
            'source_dirs': [str(d) for d in source_dirs],
            'n_matrix_files': len(matrix_files),
            'gene_info': str(gene_info_path),
            'gene_info_md5': _md5(gene_info_path),
        },
        'pipeline': {
            'n_usable_studies': len(usable_studies),
            'n_unusable_files': len(unusable_studies),
            'n_unique_raw_ids': len(resolution_log),
            'n_unique_genes_resolved': n_before,
            'n_after_min_studies': len(after_studies),
            'n_after_biotype_filter': len(final_genes),
            'n_removed_singleton': n_rm_studies,
            'n_removed_biotype': n_rm_biotype,
            'resolution_methods': dict(method_counts),
            'biotype_distribution': dict(biotype_dist),
            'frequency_distribution': freq_dist,
            'biotypes_excluded': dict(biotype_excluded),
        },
        'outputs': {
            'gene_universe.tsv': {'n_genes': len(rows), 'md5': _md5(universe_path)},
            'gene_resolution.tsv': {'n_records': len(res_rows), 'md5': _md5(resolution_path)},
            'study_coverage.tsv': {'n_files': len(study_stats), 'md5': _md5(study_path)},
        },
    }

    manifest_path = output_dir / 'manifest_step1.json'
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"manifest_step1.json written")

    # ── Summary ──
    logger.info("=" * 60)
    logger.info("GENE UNIVERSE SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  Matrix files scanned:   {len(matrix_files)}")
    logger.info(f"  Usable studies:         {len(usable_studies)}")
    logger.info(f"  Unusable files:         {len(unusable_studies)}")
    logger.info(f"  Unique raw IDs:         {len(resolution_log):,}")
    logger.info(f"  Genes (BioMart-gated):  {n_before:,}")
    logger.info(f"  After min_studies:      {len(after_studies):,}")
    logger.info(f"  After biotype filter:   {len(final_genes):,}")
    for bt, cnt in sorted(biotype_dist.items(), key=lambda x: -x[1]):
        logger.info(f"    {bt:20s} {cnt:>6,}")
    logger.info(f"  Elapsed:                {elapsed:.1f}s")
    logger.info(f"  Output:                 {output_dir}")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Stage 2, Step 1: Build gene universe from raw matrices",
    )
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('--dry-run', action='store_true',
                        help='Validate config/inputs, then exit')
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        config = load_config()
    except FileNotFoundError:
        logger.error("pipeline_config.yaml not found. Set PIPELINE_ROOT or run from project root.")
        sys.exit(1)

    for section in ('gene_universe', 'biomart', 'paths'):
        if section not in config:
            logger.error(f"Config missing '{section}' section")
            sys.exit(1)

    gu = config['gene_universe']
    for key in ('min_biomart_match', 'min_studies', 'keep_biotypes', 'accession_patterns'):
        if key not in gu:
            logger.error(f"Config missing gene_universe.{key}")
            sys.exit(1)

    build_gene_universe(config, dry_run=args.dry_run)


if __name__ == '__main__':
    main()