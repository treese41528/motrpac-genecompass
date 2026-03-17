#!/usr/bin/env python3
"""
tokenize_corpus.py — Stage 5, Scatter Phase: Raw Counts → GeneCompass Token Sequences

Pipeline position:
    Stage 1: Data Harvesting & QC
    Stage 2: Gene Universe & Cell QC            → QC'd h5ad files (raw counts)
    Stage 3: Ortholog Mapping                   → rat_tokens.pickle, rat_token_mapping.tsv
    Stage 4: Gene Medians                       → rat_gene_medians.pickle
    Stage 5: SCATTER → tokenize_corpus.py       ← THIS SCRIPT
             GATHER  → assemble_corpus.py

Purpose:
    SLURM array scatter worker. Each task processes a disjoint subset of the
    QC'd h5ad corpus and converts every cell into a GeneCompass-format token
    sequence. The per-cell transformation pipeline exactly mirrors GeneCompass
    pre-training:

      1. Load raw counts (no normalization on disk — raw-count contract preserved).
      2. normalize_total(target_sum=10,000) — in memory only, ephemeral.
      3. Divide raw counts by per-gene hybrid median (hybrid_gene_medians.pickle).
         Hybrid median = human median for T1-T3 ortholog-mapped genes,
         rat median for T4 rat-specific genes. No normalize_total.
      4. log2(1+x) transform.
      5. Rank non-zero genes descending by transformed expression.
      6. Take top 2,048 (or all non-zero genes if fewer than 2,048 expressed).
      7. Map to token IDs via rat_tokens.pickle.

    Studies in reference_assembly.exclude_studies are silently skipped
    (MoTrPAC data-leakage prevention).

Outputs (written to tokenized_corpus_dir/shards/ and tokenized_corpus_dir/manifests/):
    shards/task_{task_id:04d}_shard_{local:04d}.arrow   — Arrow IPC record batches
    manifests/task_{task_id:04d}_manifest.json          — Provenance + shard index

Arrow schema per shard (GeneCompass-compatible):
    input_ids   — list<int32>   : token IDs, descending expression order, zero-padded to 2048
    values      — list<float32> : log2(1 + norm/median) expression, zero-padded to 2048
    length      — list<int16>   : [n_expressed_genes] (single-element list, ≤ 2048)
    species     — list<int16>   : [2] for rat (0=human, 1=mouse, 2=rat)
    study_id    — string        : source study accession (GEO/AE) — extra column for Aims 2-3
    cell_id     — string        : obs_name from adata — extra column for Aims 2-3

Design notes:
    - Reads each h5ad fully into memory before processing (Gilbreth NFS defence).
    - Token mapping: only genes with BOTH a valid token AND a passing median are
      eligible. Genes failing either criterion are excluded from ranking.
    - Cells with zero library size (all raw counts = 0) after gene subsetting are
      skipped with a warning — they carry no expression signal.
    - Cell batches of CELL_BATCH_SIZE are densified at a time to cap peak RAM.
      At 1,024 cells × 21,379 genes × float32 ≈ 87 MB per batch.

Usage (direct):
    python pipeline/05_tokenization/tokenize_corpus.py \\
        --task-id 0 --n-tasks 50
    python pipeline/05_tokenization/tokenize_corpus.py \\
        --task-id 0 --n-tasks 50 --dry-run
    python pipeline/05_tokenization/tokenize_corpus.py \\
        --task-id $SLURM_ARRAY_TASK_ID --n-tasks $N_TOKENIZE_TASKS

Author: Tim Reese Lab / Claude
Date: March 2026
"""

import argparse
import json
import logging
import os
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Optional heavy imports (checked at startup)
# ─────────────────────────────────────────────────────────────────────────────
_MISSING = []

try:
    import scipy.sparse as sp
except ImportError:
    _MISSING.append('scipy')
    sp = None

try:
    import anndata as ad
except ImportError:
    _MISSING.append('anndata')
    ad = None

try:
    import pyarrow as pa
    import pyarrow.ipc as pa_ipc
except ImportError:
    _MISSING.append('pyarrow')
    pa = None
    pa_ipc = None

# ─────────────────────────────────────────────────────────────────────────────
# Project path bootstrap
# ─────────────────────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(os.environ.get('PIPELINE_ROOT', '.')).resolve()
sys.path.insert(0, str(_PROJECT_ROOT / 'lib'))
from gene_utils import load_config, resolve_path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
)
logger = logging.getLogger(__name__)

# Per-cell tokenization constants
CELL_BATCH_SIZE = 1024       # Cells densified at a time  (87 MB at 21K genes)
TOP_N_DEFAULT   = 2048       # GeneCompass paper: top-2048 genes per cell

# Arrow schema for tokenized cells
_ARROW_SCHEMA = None          # Set after pa import check


def _build_arrow_schema() -> 'pa.Schema':
    # Matches GeneCompass reference schema (verified against randsel_5w_mouse dataset)
    # plus study_id / cell_id extra columns for Aims 2-3 downstream use.
    return pa.schema([
        pa.field('input_ids', pa.list_(pa.int32())),
        pa.field('values',    pa.list_(pa.float32())),
        pa.field('length',    pa.list_(pa.int16())),
        pa.field('species',   pa.list_(pa.int16())),
        pa.field('study_id',  pa.string()),
        pa.field('cell_id',   pa.string()),
    ])


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1: REFERENCE DATA LOADING
# ═════════════════════════════════════════════════════════════════════════════

def load_token_dict(ortholog_dir: Path) -> Dict[str, int]:
    """Load rat_tokens.pickle → {ENSRNOG_base: token_id (int)}.

    Strips version suffixes for consistent lookup (ENSRNOG00000046319.4 →
    ENSRNOG00000046319). Returns dict keyed by upper-case base IDs.
    """
    pickle_path = ortholog_dir / 'rat_tokens.pickle'
    if not pickle_path.exists():
        raise FileNotFoundError(
            f"rat_tokens.pickle not found: {pickle_path}\n"
            f"  → Run Stage 3 (python run_stage3.py) first."
        )
    with open(pickle_path, 'rb') as f:
        raw: Dict = pickle.load(f)

    # Normalise keys: strip version, upper-case
    result = {}
    for k, v in raw.items():
        base = str(k).strip().split('.')[0].upper()
        result[base] = int(v)
    logger.info(f"Loaded rat_tokens.pickle: {len(result):,} gene→token entries")
    return result


def load_median_dict(median_dir: Path) -> Dict[str, float]:
    """Load hybrid_gene_medians.pickle → {ENSRNOG_base: median_float}.

    Hybrid medians: human median for T1-T3 ortholog-mapped genes, rat
    median for T4 rat-specific genes. Built by build_hybrid_medians.py.
    Empirically validated: produces value distribution closest to the
    GeneCompass reference corpus (median offset -0.075, std offset -0.096).
    """
    pickle_path = median_dir / 'hybrid_gene_medians.pickle'
    if not pickle_path.exists():
        raise FileNotFoundError(
            f"hybrid_gene_medians.pickle not found: {pickle_path}\n"
            f"  → Run build_hybrid_medians.py first:\n"
            f"    python pipeline/05_tokenization/build_hybrid_medians.py"
        )
    with open(pickle_path, 'rb') as f:
        raw: Dict = pickle.load(f)

    result = {}
    for k, v in raw.items():
        base = str(k).strip().split('.')[0].upper()
        med  = float(v)
        if med > 0.0:
            result[base] = med
    logger.info(f"Loaded hybrid_gene_medians.pickle: {len(result):,} genes")
    return result


def build_eligible_gene_arrays(
    token_dict: Dict[str, int],
    median_dict: Dict[str, float],
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """Compute the intersection of genes with both a token and a median.

    Returns:
        eligible_ids    — sorted list of ENSRNOG base IDs (defines column ordering)
        eligible_tokens — np.int32 array of token IDs, aligned with eligible_ids
        eligible_medians— np.float32 array of medians, aligned with eligible_ids
    """
    common = sorted(set(token_dict) & set(median_dict))
    if not common:
        raise RuntimeError(
            "No genes appear in BOTH rat_tokens.pickle AND hybrid_gene_medians.pickle. "
            "Cannot proceed with Stage 5."
        )
    tokens  = np.array([token_dict[g]  for g in common], dtype=np.int32)
    medians = np.array([median_dict[g] for g in common], dtype=np.float32)
    logger.info(f"Eligible genes (token ∩ median): {len(common):,}")
    logger.info(f"  Token ID range: [{tokens.min()}, {tokens.max()}]")
    logger.info(f"  Median range:   [{medians.min():.4f}, {medians.max():.4f}]")
    return common, tokens, medians


def build_gene_index(eligible_ids: List[str]) -> Dict[str, int]:
    """Build {ensrnog_base: local_column_index} lookup for eligible genes."""
    return {eid: i for i, eid in enumerate(eligible_ids)}


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2: FILE DISCOVERY AND TASK SPLITTING
# ═════════════════════════════════════════════════════════════════════════════

def discover_h5ad_files(
    qc_h5ad_dir: Path,
    exclude_studies: Set[str],
) -> List[Path]:
    """Collect all QC'd h5ad files, sorted for reproducible task splitting.

    Files whose study accession appears in exclude_studies are silently
    skipped. Study accession is inferred from the filename prefix
    (e.g. GSE242354_sample01.h5ad → GSE242354).
    """
    all_files = sorted(qc_h5ad_dir.glob('**/*.h5ad'))
    if not all_files:
        raise FileNotFoundError(
            f"No .h5ad files found in {qc_h5ad_dir}\n"
            f"  → Run Stage 2 (python run_stage2.py) first."
        )

    kept = []
    skipped = []
    for p in all_files:
        study = _infer_study_id(p)
        if study in exclude_studies:
            skipped.append(p)
        else:
            kept.append(p)

    if skipped:
        logger.info(
            f"Excluded {len(skipped):,} file(s) matching exclude_studies "
            f"({', '.join(sorted(exclude_studies))})"
        )
    logger.info(f"h5ad files available for tokenization: {len(kept):,}")
    return kept


def _infer_study_id(h5ad_path: Path) -> str:
    """Extract study accession from filename stem.

    Convention: filenames start with the study accession followed by '_'
    or '-' (e.g. GSE242354_barcode.h5ad, E-MTAB-1234_sample.h5ad).
    Falls back to the full stem if no separator found.
    """
    stem = h5ad_path.stem
    for sep in ('_', '-'):
        idx = stem.find(sep)
        if idx > 0:
            candidate = stem[:idx]
            if candidate.startswith(('GSE', 'GSM', 'E-')):
                return candidate
    return stem


def get_task_files(all_files: List[Path], task_id: int, n_tasks: int) -> List[Path]:
    """Return the interleaved slice of files assigned to this SLURM array task.

    Interleaved assignment (stride = n_tasks) balances file-size variance
    better than contiguous blocks when large datasets cluster by accession.
    """
    return [f for j, f in enumerate(all_files) if j % n_tasks == task_id]


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3: TOKENIZATION ARITHMETIC
# ═════════════════════════════════════════════════════════════════════════════

def tokenize_cell_batch(
    X_raw_dense: np.ndarray,       # (batch, n_eligible) float32, raw counts
    eligible_tokens: np.ndarray,    # (n_eligible,) int32
    eligible_medians: np.ndarray,   # (n_eligible,) float32
    top_n: int,
) -> Tuple[List[List[int]], List[List[float]], List[int]]:
    """Convert a batch of raw-count vectors into GeneCompass token sequences.

    Transformation pipeline (verified against GeneCompass reference corpus):
      1. divide raw counts by per-gene median directly (no normalize_total)
      2. log2(1 + x)              — base 2, matching GeneCompass sc.pp.log1p(base=2)
      3. rank non-zero genes descending
      4. take top `top_n`
      5. map to token IDs and extract log2 values
      6. zero-pad both input_ids and values to exactly top_n

    Note: normalize_total(10,000) was confirmed as intent via GitHub issue but
    empirical comparison against the reference corpus (randsel_5w_mouse) shows
    the actual training data was built without it. Omitting it aligns our value
    distribution (median ~1.0) with the reference.

    Returns:
        token_seqs — list of zero-padded int lists, length exactly top_n
        value_seqs — list of zero-padded float lists, length exactly top_n
        lengths    — list of actual expressed gene counts (before padding)
    """
    n_cells, n_genes = X_raw_dense.shape
    token_seqs: List[List[int]]   = []
    value_seqs: List[List[float]] = []
    lengths:    List[int]         = []

    # Constant for log base 2 conversion: log2(1+x) = log(1+x) / log(2)
    LOG2 = float(np.log(2.0))

    # Zero-pad templates
    pad_ids  = [0] * top_n
    pad_vals = [0.0] * top_n

    for i in range(n_cells):
        # Steps 1–2: divide raw counts by hybrid median, log2(1+x).
        # No normalize_total — empirically validated: raw/hybrid_median produces
        # value distribution closest to GeneCompass reference corpus
        # (median offset -0.075 vs reference; normalize_total inflates to +0.573).
        x_log = np.log1p(X_raw_dense[i] / eligible_medians) / LOG2

        # Steps 3–4: find non-zero, rank descending, take top_n
        nz_mask = x_log > 0.0
        nz_idx  = np.where(nz_mask)[0]

        if nz_idx.size == 0:
            # All genes zero after median-divide (zero-count cell or
            # all genes below detection) — emit fully-padded sequence
            token_seqs.append(list(pad_ids))
            value_seqs.append(list(pad_vals))
            lengths.append(0)
            continue

        nz_vals    = x_log[nz_idx]
        sort_order = np.argsort(-nz_vals)          # descending
        top_local  = nz_idx[sort_order[:top_n]]    # indices into n_genes
        n_top      = len(top_local)

        # Step 6: map to token IDs and extract log2 expression values
        toks = eligible_tokens[top_local].tolist()
        vals = x_log[top_local].tolist()

        # Step 7: zero-pad to exactly top_n
        if n_top < top_n:
            toks = toks + [0] * (top_n - n_top)
            vals = vals + [0.0] * (top_n - n_top)

        token_seqs.append(toks)
        value_seqs.append(vals)
        lengths.append(n_top)

    return token_seqs, value_seqs, lengths


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4: PER-FILE PROCESSING
# ═════════════════════════════════════════════════════════════════════════════

def map_varnames_to_eligible(
    var_names: List[str],
    gene_index: Dict[str, int],
) -> Tuple[List[int], List[int]]:
    """Map h5ad var_names to local eligible-gene column indices.

    Strips version suffixes before lookup. Returns:
        col_indices  — h5ad column indices that are in the eligible set
        gene_indices — corresponding eligible-gene column indices
    """
    col_indices  = []
    gene_indices = []
    for local_j, raw_name in enumerate(var_names):
        base = raw_name.strip().split('.')[0].upper()
        gi = gene_index.get(base)
        if gi is not None:
            col_indices.append(local_j)
            gene_indices.append(gi)
    return col_indices, gene_indices


def process_file(
    h5ad_path: Path,
    gene_index:       Dict[str, int],
    eligible_tokens:  np.ndarray,
    eligible_medians: np.ndarray,
    top_n:            int,
    cells_per_shard:  int,
    output_shard_dir: Path,
    task_id:          int,
    shard_counter:    List[int],    # [current_shard_number] — mutable singleton
    dry_run:          bool = False,
) -> Dict:
    """Process one h5ad file: load → map → tokenize → write Arrow shards.

    Reads the h5ad fully into memory (defensive against Gilbreth NFS).
    Processes cells in CELL_BATCH_SIZE chunks to bound peak RAM.
    Writes one or more Arrow shards when cells_per_shard is reached.

    Returns a file-level metadata dict suitable for the task manifest.
    """
    t0 = time.time()
    study_id = _infer_study_id(h5ad_path)

    # ── Load fully into memory ───────────────────────────────────────────────
    logger.debug(f"  Loading: {h5ad_path.name}")
    try:
        adata = ad.read_h5ad(str(h5ad_path))
    except Exception as exc:
        logger.error(f"  Load failed: {h5ad_path.name}: {exc}")
        return {
            'file': h5ad_path.name,
            'study_id': study_id,
            'status': 'load_error',
            'error': str(exc),
        }

    n_cells_orig = adata.n_obs
    n_genes_orig = adata.n_vars

    # ── Map var_names → eligible gene indices ────────────────────────────────
    col_indices, gene_indices = map_varnames_to_eligible(
        list(adata.var_names), gene_index
    )
    n_mapped      = len(col_indices)
    mapping_rate  = n_mapped / n_genes_orig if n_genes_orig > 0 else 0.0

    if n_mapped == 0:
        logger.warning(f"  {h5ad_path.name}: 0 eligible genes mapped — skipping")
        return {
            'file': h5ad_path.name,
            'study_id': study_id,
            'status': 'no_genes_mapped',
            'n_cells': n_cells_orig,
            'n_genes_orig': n_genes_orig,
            'n_mapped': 0,
            'mapping_rate': 0.0,
        }

    # ── Subset expression to eligible genes → CSR ────────────────────────────
    X_full = adata.X[:, col_indices]
    if not sp.issparse(X_full):
        X_full = sp.csr_matrix(X_full)
    else:
        X_full = X_full.tocsr()
    X_full = X_full.astype(np.float32)

    # Build aligned gene arrays for THIS file's column order
    # (gene_indices maps local file cols → eligible gene index)
    gene_idx_arr   = np.array(gene_indices, dtype=np.int32)
    file_medians   = eligible_medians[gene_idx_arr]  # (n_mapped,)
    file_tokens    = eligible_tokens[gene_idx_arr]   # (n_mapped,)

    # ── Cell IDs from adata.obs_names ────────────────────────────────────────
    cell_ids = list(adata.obs_names)

    # ── Batch processing ─────────────────────────────────────────────────────
    n_cells_total   = adata.n_obs
    n_cells_written = 0
    n_cells_empty   = 0

    # Per-shard accumulators
    shard_input_ids: List[List[int]]   = []
    shard_values:    List[List[float]] = []
    shard_lengths:   List[List[int]]   = []   # [[n]] — single-element list per cell
    shard_species:   List[List[int]]   = []   # [[2]] for rat
    shard_study_ids: List[str]         = []
    shard_cell_ids:  List[str]         = []
    shards_written:  List[str]         = []

    def _flush_shard() -> None:
        """Write accumulated cells to an Arrow IPC shard file."""
        if not shard_input_ids:
            return
        shard_name = (
            f"task_{task_id:04d}_shard_{shard_counter[0]:04d}.arrow"
        )
        shard_path = output_shard_dir / shard_name
        if not dry_run:
            _write_arrow_shard(
                shard_path,
                shard_input_ids,
                shard_values,
                shard_lengths,
                shard_species,
                shard_study_ids,
                shard_cell_ids,
            )
        shards_written.append(shard_name)
        shard_counter[0] += 1
        shard_input_ids.clear()
        shard_values.clear()
        shard_lengths.clear()
        shard_species.clear()
        shard_study_ids.clear()
        shard_cell_ids.clear()

    for batch_start in range(0, n_cells_total, CELL_BATCH_SIZE):
        batch_end   = min(batch_start + CELL_BATCH_SIZE, n_cells_total)
        X_batch_csr = X_full[batch_start:batch_end]

        # Densify batch: (batch, n_mapped) float32
        X_batch_dense = np.asarray(X_batch_csr.todense(), dtype=np.float32)

        # Tokenize the batch
        batch_tokens, batch_values, batch_lengths = tokenize_cell_batch(
            X_batch_dense,
            file_tokens,
            file_medians,
            top_n,
        )

        batch_cell_ids = cell_ids[batch_start:batch_end]

        for i, (toks, vals, length, cid) in enumerate(
            zip(batch_tokens, batch_values, batch_lengths, batch_cell_ids)
        ):
            if length == 0:
                n_cells_empty += 1
                continue

            shard_input_ids.append(toks)
            shard_values.append(vals)
            shard_lengths.append([length])      # List(int16): [[n]]
            shard_species.append([2])           # List(int16): [[2]] = rat
            shard_study_ids.append(study_id)
            shard_cell_ids.append(cid)
            n_cells_written += 1

            if len(shard_input_ids) >= cells_per_shard:
                _flush_shard()

    # Final partial shard (may be < cells_per_shard)
    _flush_shard()

    elapsed = time.time() - t0
    logger.info(
        f"  {h5ad_path.name}: {n_cells_total:,} cells → "
        f"{n_cells_written:,} tokenized, {n_cells_empty:,} empty | "
        f"{n_mapped:,}/{n_genes_orig:,} genes mapped | "
        f"{len(shards_written)} shard(s) | {elapsed:.1f}s"
    )

    return {
        'file': h5ad_path.name,
        'study_id': study_id,
        'status': 'ok',
        'n_cells_orig': n_cells_orig,
        'n_cells_written': n_cells_written,
        'n_cells_empty': n_cells_empty,
        'n_genes_orig': n_genes_orig,
        'n_genes_mapped': n_mapped,
        'mapping_rate': round(mapping_rate, 4),
        'shards_written': shards_written,
        'elapsed_s': round(elapsed, 1),
    }


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 5: ARROW SHARD I/O
# ═════════════════════════════════════════════════════════════════════════════

def _write_arrow_shard(
    shard_path: Path,
    input_ids:  List[List[int]],
    values:     List[List[float]],
    lengths:    List[List[int]],    # [[n]] per cell
    species:    List[List[int]],    # [[2]] per cell (rat)
    study_ids:  List[str],
    cell_ids:   List[str],
) -> None:
    """Write one Arrow IPC streaming shard.

    Schema matches GeneCompass reference (randsel_5w_mouse) plus
    study_id / cell_id extra columns for Aims 2-3.
    """
    schema = _build_arrow_schema()

    table = pa.table(
        {
            'input_ids': pa.array(
                [pa.array(ids, type=pa.int32())   for ids in input_ids],
                type=pa.list_(pa.int32()),
            ),
            'values': pa.array(
                [pa.array(v,   type=pa.float32()) for v   in values],
                type=pa.list_(pa.float32()),
            ),
            'length': pa.array(
                [pa.array(l,   type=pa.int16())   for l   in lengths],
                type=pa.list_(pa.int16()),
            ),
            'species': pa.array(
                [pa.array(s,   type=pa.int16())   for s   in species],
                type=pa.list_(pa.int16()),
            ),
            'study_id': pa.array(study_ids, type=pa.string()),
            'cell_id':  pa.array(cell_ids,  type=pa.string()),
        },
        schema=schema,
    )

    with pa_ipc.new_stream(str(shard_path), schema) as writer:
        writer.write_table(table)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 6: TASK MANIFEST
# ═════════════════════════════════════════════════════════════════════════════

def write_task_manifest(
    manifest_dir: Path,
    task_id:      int,
    n_tasks:      int,
    file_results: List[Dict],
    config:       dict,
    top_n:        int,
    cells_per_shard: int,
    n_eligible_genes: int,
    t_start:      float,
    dry_run:      bool,
) -> None:
    """Write per-task JSON manifest for the gather phase."""
    n_cells_total   = sum(r.get('n_cells_written', 0)  for r in file_results)
    n_cells_empty   = sum(r.get('n_cells_empty', 0)    for r in file_results)
    n_files_ok      = sum(1 for r in file_results if r.get('status') == 'ok')
    n_files_error   = sum(1 for r in file_results if r.get('status') != 'ok')
    all_shards      = []
    for r in file_results:
        all_shards.extend(r.get('shards_written', []))

    manifest = {
        'stage':         5,
        'phase':         'scatter',
        'task_id':       task_id,
        'n_tasks':       n_tasks,
        'generated_at':  datetime.utcnow().isoformat() + 'Z',
        'elapsed_s':     round(time.time() - t_start, 1),
        'dry_run':       dry_run,
        'config_snapshot': {
            'top_n_genes':      top_n,
            'cells_per_shard':  cells_per_shard,
            'target_sum':       None,  # No normalize_total — raw counts divided by median directly
            'n_eligible_genes': n_eligible_genes,
            'ensembl_release':  config.get('biomart', {}).get('ensembl_release', '?'),
            'exclude_studies':  config.get('reference_assembly', {}).get(
                'exclude_studies', []
            ),
        },
        'summary': {
            'n_files_processed':  n_files_ok,
            'n_files_error':      n_files_error,
            'n_cells_tokenized':  n_cells_total,
            'n_cells_empty':      n_cells_empty,
            'n_shards_written':   len(all_shards),
        },
        'shards': all_shards,
        'file_results': file_results,
    }

    manifest_path = manifest_dir / f"task_{task_id:04d}_manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"Manifest written: {manifest_path.name}")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 7: MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Stage 5 Scatter: tokenize QC'd h5ad corpus → Arrow shards"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Local serial run (all files, single task):
  python pipeline/05_tokenization/tokenize_corpus.py --task-id 0 --n-tasks 1

  # SLURM array task (called by stage5_tokenize.slurm):
  python pipeline/05_tokenization/tokenize_corpus.py \\
      --task-id $SLURM_ARRAY_TASK_ID --n-tasks $N_TOKENIZE_TASKS

  # Dry-run (validate inputs, skip writing):
  python pipeline/05_tokenization/tokenize_corpus.py \\
      --task-id 0 --n-tasks 1 --dry-run
        """,
    )
    parser.add_argument('--task-id',  type=int, required=True,
                        help='SLURM_ARRAY_TASK_ID (0-indexed)')
    parser.add_argument('--n-tasks',  type=int, required=True,
                        help='Total number of scatter tasks')
    parser.add_argument('--dry-run',  action='store_true',
                        help='Validate inputs; skip h5ad loading and writing')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='DEBUG-level logging')
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # ── Dependency check ─────────────────────────────────────────────────────
    if _MISSING:
        logger.error(f"Missing required packages: {', '.join(_MISSING)}")
        logger.error("Activate foundational_models_env and retry.")
        sys.exit(1)

    t_start = time.time()
    logger.info("=" * 70)
    logger.info("STAGE 5 SCATTER: Corpus Tokenization")
    logger.info(f"  Task {args.task_id} of {args.n_tasks}")
    logger.info("=" * 70)

    # ── Config ───────────────────────────────────────────────────────────────
    try:
        config = load_config()
    except FileNotFoundError:
        logger.error(
            "pipeline_config.yaml not found. "
            "Set PIPELINE_ROOT or run from project root."
        )
        sys.exit(1)

    paths     = config['paths']
    ref_cfg   = config.get('reference_assembly', {})
    top_n     = int(ref_cfg.get('top_n_genes', TOP_N_DEFAULT))
    cells_per_shard = int(ref_cfg.get('cells_per_shard', 50_000))
    exclude_studies = set(ref_cfg.get('exclude_studies', []))

    orth_dir    = resolve_path(config, paths['ortholog_dir'])
    median_dir  = resolve_path(config, paths['median_dir'])
    qc_h5ad_dir = resolve_path(config, paths['qc_h5ad_dir'])
    corpus_dir  = resolve_path(config, paths['tokenized_corpus_dir'])

    shard_dir    = corpus_dir / 'shards'
    manifest_dir = corpus_dir / 'manifests'

    logger.info(f"Config: top_n={top_n}, cells_per_shard={cells_per_shard:,}, "
                f"exclude_studies={sorted(exclude_studies)}")

    if not args.dry_run:
        shard_dir.mkdir(parents=True, exist_ok=True)
        manifest_dir.mkdir(parents=True, exist_ok=True)

    # ── Load reference data ──────────────────────────────────────────────────
    logger.info("Loading reference data ...")
    try:
        token_dict   = load_token_dict(orth_dir)
        median_dict  = load_median_dict(median_dir)
        eligible_ids, eligible_tokens, eligible_medians = build_eligible_gene_arrays(
            token_dict, median_dict
        )
        gene_index = build_gene_index(eligible_ids)
    except (FileNotFoundError, RuntimeError) as exc:
        logger.error(str(exc))
        sys.exit(1)

    n_eligible = len(eligible_ids)

    # ── File discovery and task splitting ────────────────────────────────────
    logger.info("Discovering h5ad files ...")
    try:
        all_files = discover_h5ad_files(qc_h5ad_dir, exclude_studies)
    except FileNotFoundError as exc:
        logger.error(str(exc))
        sys.exit(1)

    task_files = get_task_files(all_files, args.task_id, args.n_tasks)
    logger.info(f"Task {args.task_id}: assigned {len(task_files):,} of "
                f"{len(all_files):,} files")

    if not task_files:
        logger.warning("No files assigned to this task — nothing to do.")
        # Write an empty manifest so the gather phase can count tasks correctly
        write_task_manifest(
            manifest_dir, args.task_id, args.n_tasks, [],
            config, top_n, cells_per_shard, n_eligible, t_start, args.dry_run,
        )
        sys.exit(0)

    if args.dry_run:
        logger.info("DRY RUN — skipping h5ad loading and Arrow writing.")
        logger.info(f"  Would process {len(task_files):,} file(s):")
        for p in task_files[:10]:
            logger.info(f"    {p.name}")
        if len(task_files) > 10:
            logger.info(f"    ... and {len(task_files) - 10} more")
        sys.exit(0)

    # ── Process files ────────────────────────────────────────────────────────
    shard_counter = [0]         # Mutable singleton passed into process_file
    file_results  = []

    for file_idx, h5ad_path in enumerate(task_files):
        logger.info(
            f"File {file_idx + 1}/{len(task_files)}: {h5ad_path.name}"
        )
        meta = process_file(
            h5ad_path     = h5ad_path,
            gene_index    = gene_index,
            eligible_tokens  = eligible_tokens,
            eligible_medians = eligible_medians,
            top_n         = top_n,
            cells_per_shard = cells_per_shard,
            output_shard_dir = shard_dir,
            task_id       = args.task_id,
            shard_counter = shard_counter,
            dry_run       = args.dry_run,
        )
        file_results.append(meta)

    # ── Write task manifest ──────────────────────────────────────────────────
    write_task_manifest(
        manifest_dir  = manifest_dir,
        task_id       = args.task_id,
        n_tasks       = args.n_tasks,
        file_results  = file_results,
        config        = config,
        top_n         = top_n,
        cells_per_shard = cells_per_shard,
        n_eligible_genes = n_eligible,
        t_start       = t_start,
        dry_run       = args.dry_run,
    )

    # ── Summary ──────────────────────────────────────────────────────────────
    n_cells_out = sum(r.get('n_cells_written', 0) for r in file_results)
    n_shards    = shard_counter[0]
    elapsed     = time.time() - t_start

    logger.info("=" * 70)
    logger.info(f"Task {args.task_id} COMPLETE")
    logger.info(f"  Files processed:  {len(task_files):,}")
    logger.info(f"  Cells tokenized:  {n_cells_out:,}")
    logger.info(f"  Shards written:   {n_shards:,}")
    logger.info(f"  Elapsed:          {elapsed:.1f}s")
    logger.info("=" * 70)


if __name__ == '__main__':
    main()