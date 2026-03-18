#!/usr/bin/env python3
"""
build_coexp_embedding.py — Stage 6, Step 1: Co-expression Embedding

Pipeline position:
    Stage 1: Data Harvesting & QC
    Stage 2: Gene Universe & Cell QC           → qc_h5ad/ (raw counts)
    Stage 3: build_ortholog_mapping.py         → rat_to_human_mapping.pickle
    Stage 4: Gene Medians
    Stage 5: Reference Assembly & Tokenization
    Stage 6, Step 1: build_coexp_embedding.py  ← THIS SCRIPT
    Stage 6, Step 2: build_family_embedding.py

Purpose:
    Compute 768-dimensional co-expression embeddings for rat gene tokens by
    following the GeneCompass prior knowledge construction protocol exactly:

      1. For each QC'd h5ad in the Stage 2 corpus, uniformly sample 3,000 cells.
      2. Compute nonzero-based Pearson correlation (PCC) between all gene pairs:
         only cells where BOTH genes have raw count ≥ 1 are used. This avoids
         inflating correlations through shared dropout structure in sparse data.
      3. Retain gene pairs with PCC ≥ 0.8 (GeneCompass exact threshold).
      4. Unify rat gene IDs to human Ensembl IDs via Stage 3's authoritative
         rat_to_human_mapping.pickle. Rat T4 genes (new tokens without human
         orthologs) keep their ENSRNOG ID — they form embeddings adjacent to
         their correlated partners in 768-dim space, which is exactly what
         fine-tuning requires for new token initialization.
      5. Deduplicate pairs across all 88 studies.
      6. Train gene2vec (Word2Vec Skip-Gram, 768 dimensions) on the pair list.
      7. Save embeddings keyed by the same unified ID used in the token lookup.

Key design decisions:
    - h5ad files are discovered by globbing paths.qc_h5ad_dir, NOT via a
      user-provided dataset TSV. The Stage 2 output directory IS the dataset list.
    - Rat gene ID mapping is consumed from Stage 3's rat_to_human_mapping.pickle,
      NOT re-derived from BioMart TSVs. Stage 3 is the single source of truth
      for rat gene token assignments.
    - var_names are version-stripped before lookup (ENSRNOG00000046319.4 →
      ENSRNOG00000046319), consistent with Stage 4's map_varnames_to_universe().
    - Raw count validation is performed on each loaded h5ad: if adata.X max
      value is suspiciously low (< 20), a warning is issued. The Stage 2
      preprocessing contract guarantees raw counts, but this guard catches
      any accidental normalization.
    - PCC extraction from the correlation matrix uses np.where() instead of
      nested Python loops for ~100x speedup over the naive approach.
    - Two methodological additions beyond the GeneCompass paper are noted in
      config and this docstring for methods transparency:
        min_shared_cells (default 10): minimum co-expressed cells required
          for a valid PCC. Not in the paper, but guards against spurious
          correlations from very sparse gene pairs.
        min_expr_ratio (default 0.05): genes expressed in fewer than 5% of
          sampled cells are excluded before PCC computation. Not in the paper,
          but reduces memory and compute substantially with minimal impact on
          biologically meaningful pairs.

Inputs (all paths from pipeline_config.yaml):
    Stage 2 → paths.qc_h5ad_dir                   — glob for all h5ad files
    Stage 3 → ortholog_dir/rat_to_human_mapping.pickle  — gene ID unification
    Config  → prior_knowledge.coexp.*              — tunable parameters

Outputs (all to paths.prior_knowledge_dir):
    coexp_gene_pairs.txt        — Gene pairs used for training
    coexp_gene2vec.model        — Gensim Word2Vec model
    coexp_embeddings.pkl        — {gene_id: np.ndarray(768,)} — primary output
    stage6_coexp_manifest.json  — Provenance record

Note on runtime:
    Sampling 3,000 cells × 88 studies = ~264,000 cells total. PCC computation
    on ~5,000–10,000 expressed genes per matrix typically takes 2–6 hours on a
    single CPU. On Gilbreth this script can be submitted as a single-node job:
      sbatch --time=8:00:00 --mem=32G --cpus-per-task=8 run_coexp_slurm.sh
    The gene2vec training step is fast (< 5 minutes).

Usage:
    python pipeline/06_prior_knowledge/build_coexp_embedding.py
    python pipeline/06_prior_knowledge/build_coexp_embedding.py --dry-run
    python pipeline/06_prior_knowledge/build_coexp_embedding.py -v

Author: Tim Reese Lab / Claude
Date: March 2026
"""

import argparse
import csv
import hashlib
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
# Optional heavy imports — checked at startup
# ─────────────────────────────────────────────────────────────────────────────
_MISSING = []
try:
    import anndata as ad
except ImportError:
    _MISSING.append('anndata')
    ad = None

try:
    import pandas as pd
except ImportError:
    _MISSING.append('pandas')
    pd = None

try:
    from gensim.models import Word2Vec
except ImportError:
    _MISSING.append('gensim')
    Word2Vec = None

# ─────────────────────────────────────────────────────────────────────────────
# Project path bootstrap — identical pattern to all other pipeline scripts
# ─────────────────────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(os.environ.get('PIPELINE_ROOT', '.')).resolve()
sys.path.insert(0, str(_PROJECT_ROOT / 'lib'))
from gene_utils import load_config, resolve_path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
)
logger = logging.getLogger(__name__)
_CHECKPOINT_FILE = 'coexp_checkpoint.pkl'
_CHECKPOINT_INTERVAL = 25   # save every N files

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1: DATA LOADING
# ═════════════════════════════════════════════════════════════════════════════

def load_rat_to_human_mapping(ortholog_dir: Path) -> Dict[str, str]:
    """Load Stage 3 rat_to_human_mapping.pickle.

    This is the authoritative rat→human Ensembl ID mapping, produced by the
    tiered ortholog resolution in Stage 3. It is the single source of truth —
    do NOT re-derive from raw BioMart TSVs.

    Returns: {rat_ensrnog: human_ensg}  (T4 genes absent — they keep ENSRNOG)
    """
    pkl_path = ortholog_dir / 'rat_to_human_mapping.pickle'
    if not pkl_path.exists():
        raise FileNotFoundError(
            f"rat_to_human_mapping.pickle not found: {pkl_path}\n"
            f"  → Run Stage 3 first:  python run_stage3.py"
        )
    with open(pkl_path, 'rb') as f:
        mapping = pickle.load(f)
    logger.info(f"Stage 3 rat_to_human_mapping: {len(mapping):,} rat→human pairs loaded")
    return mapping


def discover_h5ad_files(qc_h5ad_dir: Path) -> List[Path]:
    """Collect all QC'd h5ad files from the Stage 2 output directory.

    Sorted for reproducible ordering. Consistent with Stage 4's
    discover_h5ad_files() function.
    """
    files = sorted(qc_h5ad_dir.glob('**/*.h5ad'))
    if not files:
        raise FileNotFoundError(
            f"No .h5ad files found in {qc_h5ad_dir}\n"
            f"  → Run Stage 2 first:  python run_stage2.py"
        )
    return files


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2: CELL SAMPLING
# ═════════════════════════════════════════════════════════════════════════════

def uniform_sample(adata, n_cells: int, seed: int = 42):
    """Uniformly sample cells from an AnnData object without replacement.

    If the dataset has fewer cells than n_cells, all cells are used.
    This is the GeneCompass protocol: 3,000 cells per matrix sampled uniformly
    to cover the range of expression without biasing toward dominant cell types.
    """
    if adata.n_obs <= n_cells:
        logger.debug(f"  Dataset has {adata.n_obs} cells (≤{n_cells}), using all")
        return adata.copy()
    rng = np.random.default_rng(seed)
    indices = rng.choice(adata.n_obs, size=n_cells, replace=False)
    return adata[sorted(indices)].copy()


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3: GENE ID RESOLUTION
# ═════════════════════════════════════════════════════════════════════════════

def resolve_gene_ids(
    var_names: List[str],
    rat_to_human: Dict[str, str],
) -> Dict[str, str]:
    """Map h5ad var_names to unified gene IDs.

    Unification rules (consistent with Stage 3 and Stage 4):
      1. Strip version suffix: ENSRNOG00000046319.4 → ENSRNOG00000046319
      2. Look up in rat_to_human (Stage 3 authoritative mapping):
           - Found → use the human Ensembl ID  (T1–T3 ortholog-mapped tokens)
           - Not found → keep the stripped ENSRNOG  (T4 new tokens)

    Returns: {h5ad_var_name: unified_gene_id}
    """
    resolved: Dict[str, str] = {}
    for raw_name in var_names:
        stripped = raw_name.strip().split('.')[0].upper()
        if not stripped.startswith('ENSRNOG'):
            continue  # Skip any non-rat IDs that slipped through
        unified = rat_to_human.get(stripped, stripped)
        resolved[raw_name] = unified
    return resolved


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4: NONZERO-BASED PEARSON CORRELATION
# ═════════════════════════════════════════════════════════════════════════════

def compute_nonzero_pcc(
    expr: np.ndarray,
    gene_names: List[str],
    min_shared_cells: int,
    min_expr_ratio: float,
    pcc_threshold: float,
    chunk_size: int,
) -> List[Tuple[str, str]]:
    """Compute nonzero-based PCC for all gene pairs and return high-PCC pairs.

    GeneCompass protocol:
      - For gene pair (A, B), use only cells where BOTH A ≥ 1 AND B ≥ 1.
      - Compute Pearson correlation on those cells only.
      - This avoids inflating correlations via shared dropout structure.

    Implementation:
      - Zeros replaced with NaN before pandas .corr(), which excludes NaN
        values pairwise — exactly equivalent to the nonzero-only criterion.
      - PCC values extracted from the correlation matrix using np.where()
        rather than nested Python loops — ~100x faster for large gene sets.
      - For gene counts > chunk_size, computation proceeds in block pairs to
        avoid loading a full n_genes × n_genes matrix into memory.

    Methodological additions beyond GeneCompass paper (documented in config):
      min_shared_cells: minimum co-expressed cells for a valid PCC.
      min_expr_ratio:   pre-filter genes expressed in < this fraction of cells.

    Args:
        expr:             ndarray (n_cells × n_genes), raw counts
        gene_names:       list of gene identifiers (same length as n_genes axis)
        min_shared_cells: minimum co-expressed cells for valid PCC
        min_expr_ratio:   pre-filter genes expressed below this ratio
        pcc_threshold:    retain pairs with PCC strictly above this value
        chunk_size:       maximum genes per chunk for memory management

    Returns:
        List of (gene_id_1, gene_id_2) tuples where PCC > pcc_threshold.
    """
    n_cells, n_genes = expr.shape

    # ── Pre-filter: drop lowly-expressed genes ────────────────────────────────
    expr_counts = (expr >= 1).sum(axis=0)
    keep_mask = expr_counts > (n_cells * min_expr_ratio)
    expr_kept = expr[:, keep_mask]
    genes_kept = [gene_names[i] for i in range(n_genes) if keep_mask[i]]
    n_kept = int(keep_mask.sum())
    logger.debug(
        f"  Gene pre-filter: kept {n_kept}/{n_genes} "
        f"(expressed in >{min_expr_ratio*100:.0f}% of cells)"
    )

    if n_kept < 2:
        return []

    # ── Replace zeros with NaN for nonzero-based PCC ─────────────────────────
    expr_masked = expr_kept.astype(np.float64)
    expr_masked[expr_masked < 1] = np.nan

    pairs: List[Tuple[str, str]] = []

    if n_kept <= chunk_size:
        # ── Small: full correlation matrix in one shot ────────────────────────
        df = pd.DataFrame(expr_masked, columns=genes_kept)
        corr = df.corr(method='pearson', min_periods=min_shared_cells).values

        # Vectorized extraction of upper-triangle pairs above threshold
        mask = np.triu(~np.isnan(corr) & (corr > pcc_threshold), k=1)
        ii_arr, jj_arr = np.where(mask)
        for ii, jj in zip(ii_arr, jj_arr):
            pairs.append((genes_kept[ii], genes_kept[jj]))

    else:
        # ── Large: block-pair iteration ───────────────────────────────────────
        n_chunks = (n_kept + chunk_size - 1) // chunk_size

        for ci in range(n_chunks):
            i_start = ci * chunk_size
            i_end   = min(i_start + chunk_size, n_kept)
            genes_i = genes_kept[i_start:i_end]
            data_i  = expr_masked[:, i_start:i_end]

            for cj in range(ci, n_chunks):
                j_start = cj * chunk_size
                j_end   = min(j_start + chunk_size, n_kept)
                genes_j = genes_kept[j_start:j_end]
                data_j  = expr_masked[:, j_start:j_end]

                if ci == cj:
                    # Within-chunk: upper triangle only
                    df_block = pd.DataFrame(data_i, columns=genes_i)
                    corr = df_block.corr(
                        method='pearson', min_periods=min_shared_cells
                    ).values

                    mask = np.triu(
                        ~np.isnan(corr) & (corr > pcc_threshold), k=1
                    )
                    ii_arr, jj_arr = np.where(mask)
                    for ii, jj in zip(ii_arr, jj_arr):
                        pairs.append((genes_i[ii], genes_i[jj]))

                else:
                    # Cross-block: compute only the n_i × n_j off-diagonal block.
                    # Stack i then j, take full corr, extract the cross-block
                    # slice — avoids Python loops over individual pairs.
                    n_i = len(genes_i)
                    combined = pd.DataFrame(
                        np.hstack([data_i, data_j]),
                        columns=genes_i + genes_j,
                    )
                    corr_full = combined.corr(
                        method='pearson', min_periods=min_shared_cells
                    ).values
                    # Cross-block is corr_full[:n_i, n_i:]
                    cross = corr_full[:n_i, n_i:]
                    mask = ~np.isnan(cross) & (cross > pcc_threshold)
                    ii_arr, jj_arr = np.where(mask)
                    for ii, jj in zip(ii_arr, jj_arr):
                        pairs.append((genes_i[ii], genes_j[jj]))

    logger.debug(f"  Found {len(pairs):,} pairs (PCC > {pcc_threshold})")
    return pairs


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 5: PER-FILE PROCESSING
# ═════════════════════════════════════════════════════════════════════════════

def process_file(
    h5ad_path: Path,
    rat_to_human: Dict[str, str],
    n_sample_cells: int,
    min_shared_cells: int,
    min_expr_ratio: float,
    pcc_threshold: float,
    chunk_size: int,
) -> Tuple[List[Tuple[str, str]], dict]:
    """Process one QC'd h5ad file: sample → PCC → unify gene IDs → pairs.

    Args:
        h5ad_path:        path to a Stage 2 QC'd h5ad (raw counts)
        rat_to_human:     Stage 3 authoritative rat→human mapping
        n_sample_cells:   cells to sample per dataset (GeneCompass: 3,000)
        min_shared_cells: minimum co-expressed cells for valid PCC
        min_expr_ratio:   gene pre-filter fraction
        pcc_threshold:    PCC cutoff (GeneCompass: 0.8)
        chunk_size:       max genes per PCC chunk

    Returns:
        (unified_pairs, file_meta)
        unified_pairs: list of (unified_id_1, unified_id_2) tuples
        file_meta:     dict with per-file stats for the manifest
    """
    t0 = time.time()
    logger.info(f"  {h5ad_path.name}")

    try:
        adata = ad.read_h5ad(str(h5ad_path))
    except Exception as exc:
        logger.error(f"    Load failed: {exc}")
        return [], {'file': h5ad_path.name, 'status': 'load_error', 'error': str(exc)}

    n_cells_orig = adata.n_obs
    n_genes_orig = adata.n_vars

    # ── Validate raw counts ───────────────────────────────────────────────────
    # Stage 2 contract guarantees raw counts. If max < 20 the matrix looks
    # pre-normalized — issue a warning but continue.
    try:
        import scipy.sparse as sp
        if sp.issparse(adata.X):
            x_max = adata.X.max()
        else:
            x_max = float(np.array(adata.X).max())
        if x_max < 20:
            logger.warning(
                f"    adata.X max value is {x_max:.2f} (< 20). "
                "This matrix may be normalized. Stage 2 should produce raw counts. "
                "PCC results may be unreliable for this file."
            )
    except Exception:
        pass  # Non-critical check — never abort on this

    # ── Sample cells ──────────────────────────────────────────────────────────
    adata = uniform_sample(adata, n_sample_cells)
    logger.debug(f"    Sampled: {adata.n_obs}/{n_cells_orig} cells")

    # ── Resolve gene IDs ─────────────────────────────────────────────────────
    id_map = resolve_gene_ids(list(adata.var_names), rat_to_human)
    if not id_map:
        logger.warning(f"    No ENSRNOG var_names found — skipping")
        return [], {
            'file': h5ad_path.name, 'status': 'no_ensrnog_genes',
            'n_cells_orig': n_cells_orig, 'n_genes_orig': n_genes_orig,
        }

    # Reindex adata to only mapped genes
    mapped_vars = [v for v in adata.var_names if v in id_map]
    adata = adata[:, mapped_vars]

    # ── Dense expression matrix ───────────────────────────────────────────────
    try:
        import scipy.sparse as sp
        if sp.issparse(adata.X):
            expr = np.asarray(adata.X.todense(), dtype=np.float64)
        else:
            expr = np.asarray(adata.X, dtype=np.float64)
    except Exception as exc:
        logger.error(f"    Matrix extraction failed: {exc}")
        return [], {'file': h5ad_path.name, 'status': 'matrix_error', 'error': str(exc)}

    # ── Compute nonzero PCC ───────────────────────────────────────────────────
    raw_pairs = compute_nonzero_pcc(
        expr=expr,
        gene_names=list(adata.var_names),
        min_shared_cells=min_shared_cells,
        min_expr_ratio=min_expr_ratio,
        pcc_threshold=pcc_threshold,
        chunk_size=chunk_size,
    )

    # ── Unify gene IDs to human Ensembl (or ENSRNOG for T4 genes) ────────────
    unified_pairs: List[Tuple[str, str]] = []
    n_unmapped = 0
    for g1, g2 in raw_pairs:
        u1 = id_map.get(g1)
        u2 = id_map.get(g2)
        if u1 and u2:
            # Sort so (A,B) and (B,A) are treated as the same pair at dedup
            unified_pairs.append(tuple(sorted([u1, u2])))
        else:
            n_unmapped += 1

    elapsed = time.time() - t0
    logger.info(
        f"    {n_cells_orig} cells → {adata.n_obs} sampled | "
        f"{len(raw_pairs):,} raw pairs → {len(unified_pairs):,} unified | "
        f"{elapsed:.1f}s"
    )

    meta = {
        'file': h5ad_path.name,
        'status': 'ok',
        'n_cells_orig': n_cells_orig,
        'n_cells_sampled': adata.n_obs,
        'n_genes_orig': n_genes_orig,
        'n_genes_mapped': len(mapped_vars),
        'n_raw_pairs': len(raw_pairs),
        'n_unified_pairs': len(unified_pairs),
        'n_unmapped_pairs': n_unmapped,
        'elapsed_s': round(elapsed, 1),
    }
    return unified_pairs, meta


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 6: TRAIN gene2vec
# ═════════════════════════════════════════════════════════════════════════════

def train_gene2vec(
    gene_pairs: List[Tuple[str, str]],
    vector_size: int,
    window: int,
    sg: int,
    epochs: int,
    workers: int,
    seed: int = 42,
):
    """Train gene2vec (Word2Vec Skip-Gram) on co-expression pairs.

    Each co-expression pair is a 2-token "sentence". With window=1 and
    Skip-Gram, the model learns: given gene A, predict its co-expressed
    partner gene B. Genes that frequently co-express cluster together in
    the 768-dimensional embedding space.

    This matches the GeneCompass protocol (768 dimensions, Skip-Gram),
    upscaled from the original gene2vec paper (256 dimensions) to match
    the GeneCompass transformer hidden size.
    """
    if Word2Vec is None:
        raise ImportError("gensim is required:  pip install gensim")

    sentences = [list(p) for p in gene_pairs]

    logger.info("=" * 60)
    logger.info("Training gene2vec (co-expression)")
    logger.info("=" * 60)
    logger.info(f"  Training pairs:   {len(sentences):,}")
    logger.info(f"  Vector size:      {vector_size}")
    logger.info(f"  Algorithm:        {'Skip-gram' if sg == 1 else 'CBOW'}")
    logger.info(f"  Window:           {window}")
    logger.info(f"  Epochs:           {epochs}")
    logger.info(f"  Workers:          {workers}")

    t0 = time.time()
    model = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=1,
        sg=sg,
        workers=workers,
        epochs=epochs,
        seed=seed,
    )
    elapsed = time.time() - t0
    logger.info(f"  Training complete in {elapsed:.1f}s")
    logger.info(f"  Vocabulary size:  {len(model.wv):,} genes")
    return model


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 7: MANIFEST
# ═════════════════════════════════════════════════════════════════════════════

def file_md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(
    output_dir: Path,
    config: dict,
    rat_to_human_path: Path,
    qc_h5ad_dir: Path,
    n_files_processed: int,
    n_files_skipped: int,
    n_raw_pairs_total: int,
    n_unique_pairs: int,
    n_embeddings: int,
    vector_size: int,
    file_metas: List[dict],
    elapsed: float,
    dry_run: bool,
) -> None:
    pk   = config.get('prior_knowledge', {})
    coexp = pk.get('coexp', {})

    manifest = {
        'stage': '6_step1_coexp_embedding',
        'script': 'build_coexp_embedding.py',
        'generated': datetime.utcnow().isoformat() + 'Z',
        'dry_run': dry_run,
        'elapsed_seconds': round(elapsed, 1),
        'parameters': {
            'pcc_threshold':    coexp.get('pcc_threshold', 0.8),
            'n_sample_cells':   coexp.get('n_sample_cells', 3000),
            'min_shared_cells': coexp.get('min_shared_cells', 10),
            'min_expr_ratio':   coexp.get('min_expr_ratio', 0.05),
            'chunk_size':       coexp.get('chunk_size', 2000),
            'vector_size':      vector_size,
            'window':           coexp.get('window', 1),
            'sg':               coexp.get('sg', 1),
            'epochs':           coexp.get('epochs', 30),
            'workers':          coexp.get('workers', 8),
        },
        'methodological_notes': [
            'min_shared_cells is an addition beyond the GeneCompass paper — '
            'guards against spurious PCC from very sparse gene pairs.',
            'min_expr_ratio is an addition beyond the GeneCompass paper — '
            'pre-filters lowly-expressed genes to reduce compute.',
        ],
        'inputs': {
            'rat_to_human_mapping': {
                'path': str(rat_to_human_path),
                'md5':  file_md5(rat_to_human_path) if rat_to_human_path.exists() else None,
                'size_bytes': rat_to_human_path.stat().st_size
                              if rat_to_human_path.exists() else None,
            },
            'qc_h5ad_dir': str(qc_h5ad_dir),
        },
        'outputs': {
            'files_processed': n_files_processed,
            'files_skipped':   n_files_skipped,
            'raw_pairs_total': n_raw_pairs_total,
            'unique_pairs':    n_unique_pairs,
            'embeddings':      n_embeddings,
            'embedding_dimension': vector_size,
        },
        'per_file_stats': file_metas,
        'config_snapshot': {
            'biomart':          config.get('biomart', {}),
            'prior_knowledge':  pk,
        },
    }

    manifest_path = output_dir / 'stage6_coexp_manifest.json'
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"Manifest written: {manifest_path}")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 8: MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Stage 6 Step 1: Co-expression Embedding',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Validate all inputs and configuration without running PCC or training',
    )
    parser.add_argument(
        '-v', '--verbose', action='store_true',
        help='Enable DEBUG-level logging',
    )
    parser.add_argument(
        '--resume', action='store_true',
        help=(
            'Resume from checkpoint. Loads coexp_checkpoint.pkl from the '
            'output directory and skips already-processed files.'
        ),
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if _MISSING:
        logger.error(f"Missing required packages: {', '.join(_MISSING)}")
        logger.error("  Install:  pip install anndata pandas gensim")
        sys.exit(1)

    t_start = time.time()

    # ── Load configuration ────────────────────────────────────────────────────
    config     = load_config(_PROJECT_ROOT / 'config' / 'pipeline_config.yaml')
    paths_cfg  = config.get('paths', {})
    pk_cfg     = config.get('prior_knowledge', {})
    coexp_cfg  = pk_cfg.get('coexp', {})

    # ── Resolve all paths from config ─────────────────────────────────────────
    qc_h5ad_dir  = resolve_path(config, paths_cfg['qc_h5ad_dir'])
    ortholog_dir = resolve_path(config, paths_cfg['ortholog_dir'])
    output_dir   = resolve_path(config, paths_cfg['prior_knowledge_dir'])

    # ── Parameters — config with GeneCompass defaults ─────────────────────────
    pcc_threshold    = float(coexp_cfg.get('pcc_threshold',    0.8))
    n_sample_cells   = int(coexp_cfg.get('n_sample_cells',     3000))
    min_shared_cells = int(coexp_cfg.get('min_shared_cells',   10))
    min_expr_ratio   = float(coexp_cfg.get('min_expr_ratio',   0.05))
    chunk_size       = int(coexp_cfg.get('chunk_size',          2000))
    vector_size      = int(coexp_cfg.get('vector_size',         768))
    window           = int(coexp_cfg.get('window',              1))
    sg               = int(coexp_cfg.get('sg',                  1))
    epochs           = int(coexp_cfg.get('epochs',              30))
    workers          = int(coexp_cfg.get('workers',             8))

    logger.info("=" * 60)
    logger.info("Stage 6 Step 1: Co-expression Embedding")
    logger.info("=" * 60)
    logger.info(f"  Project root:      {_PROJECT_ROOT}")
    logger.info(f"  QC h5ad dir:       {qc_h5ad_dir}")
    logger.info(f"  Output dir:        {output_dir}")
    logger.info(f"  pcc_threshold:     {pcc_threshold}")
    logger.info(f"  n_sample_cells:    {n_sample_cells:,}")
    logger.info(f"  min_shared_cells:  {min_shared_cells}")
    logger.info(f"  min_expr_ratio:    {min_expr_ratio}")
    logger.info(f"  chunk_size:        {chunk_size:,}")
    logger.info(f"  vector_size:       {vector_size}")
    logger.info(f"  epochs:            {epochs}")
    logger.info(f"  dry_run:           {args.dry_run}")

    # ── Validate prerequisite files ───────────────────────────────────────────
    rat_to_human_path = ortholog_dir / 'rat_to_human_mapping.pickle'
    if not rat_to_human_path.exists():
        logger.error(f"MISSING: rat_to_human_mapping.pickle — {rat_to_human_path}")
        logger.error("  → Run Stage 3 first:  python run_stage3.py")
        sys.exit(1)

    try:
        h5ad_files = discover_h5ad_files(qc_h5ad_dir)
    except FileNotFoundError as exc:
        logger.error(str(exc))
        sys.exit(1)
    logger.info(f"  QC'd h5ad files: {len(h5ad_files):,} [OK]")

    output_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        logger.info("Dry run complete — all inputs validated.")
        return

    # ═══════════════════════════════════════════════════════════════════════════
    # Phase 1: Load Stage 3 mapping
    # ═══════════════════════════════════════════════════════════════════════════
    logger.info("\nPhase 1: Loading Stage 3 rat→human gene mapping")
    logger.info("-" * 60)
    rat_to_human = load_rat_to_human_mapping(ortholog_dir)

    # ═══════════════════════════════════════════════════════════════════════════
    # Phase 2: Process each h5ad file (with checkpoint/resume support)
    # ═══════════════════════════════════════════════════════════════════════════
    logger.info(f"\nPhase 2: Computing co-expression pairs across {len(h5ad_files):,} files")
    logger.info("-" * 60)

    checkpoint_path = output_dir / _CHECKPOINT_FILE
    all_pairs: Set[Tuple[str, str]] = set()
    file_metas: List[dict] = []
    n_raw_pairs_total = 0
    n_files_skipped = 0
    resume_from = 0   # index to start from (0 = beginning)

    # ── Load checkpoint if resuming ───────────────────────────────────────────
    if args.resume and checkpoint_path.exists():
        logger.info(f"  Loading checkpoint: {checkpoint_path}")
        with open(checkpoint_path, 'rb') as f:
            ckpt = pickle.load(f)
        all_pairs         = ckpt['all_pairs']
        file_metas        = ckpt['file_metas']
        n_raw_pairs_total = ckpt['n_raw_pairs_total']
        n_files_skipped   = ckpt['n_files_skipped']
        resume_from       = ckpt['last_completed_index'] + 1
        logger.info(
            f"  Resuming from file {resume_from}/{len(h5ad_files)} "
            f"({len(all_pairs):,} pairs accumulated so far)"
        )
    elif args.resume:
        logger.warning(
            f"  --resume specified but no checkpoint found at {checkpoint_path}. "
            f"Starting from the beginning."
        )

    # ── Main loop ─────────────────────────────────────────────────────────────
    for i, h5ad_path in enumerate(h5ad_files, 1):
        if i - 1 < resume_from:
            continue   # skip already-processed files

        logger.info(f"[{i:3d}/{len(h5ad_files)}]  {h5ad_path.name}")

        unified_pairs, meta = process_file(
            h5ad_path=h5ad_path,
            rat_to_human=rat_to_human,
            n_sample_cells=n_sample_cells,
            min_shared_cells=min_shared_cells,
            min_expr_ratio=min_expr_ratio,
            pcc_threshold=pcc_threshold,
            chunk_size=chunk_size,
        )

        file_metas.append(meta)

        if meta['status'] != 'ok':
            n_files_skipped += 1
        else:
            n_raw_pairs_total += meta.get('n_raw_pairs', 0)
            all_pairs.update(unified_pairs)
            logger.info(f"  Running unique pairs: {len(all_pairs):,}")

        # ── Checkpoint every N files ──────────────────────────────────────────
        if i % _CHECKPOINT_INTERVAL == 0:
            ckpt = {
                'all_pairs':          all_pairs,
                'file_metas':         file_metas,
                'n_raw_pairs_total':  n_raw_pairs_total,
                'n_files_skipped':    n_files_skipped,
                'last_completed_index': i - 1,
            }
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(ckpt, f, protocol=4)
            logger.info(
                f"  Checkpoint saved at file {i}/{len(h5ad_files)} "
                f"({checkpoint_path.name})"
            )

    n_files_processed = len(h5ad_files) - resume_from - n_files_skipped
    logger.info(f"\nPhase 2 complete: processed {len(h5ad_files) - n_files_skipped}/{len(h5ad_files)} files OK")

    # Clean up checkpoint on successful completion
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        logger.info(f"  Checkpoint deleted (run complete)")

    # ═══════════════════════════════════════════════════════════════════════════
    # Phase 3: Deduplication
    # ═══════════════════════════════════════════════════════════════════════════
    logger.info("\nPhase 3: Deduplication")
    logger.info("-" * 60)

    unique_pairs = sorted(all_pairs)
    logger.info(f"  Raw pair accumulation: {n_raw_pairs_total:,}")
    logger.info(f"  Unique pairs:          {len(unique_pairs):,}")

    if not unique_pairs:
        logger.error(
            "No co-expression pairs found. Check data and threshold settings."
        )
        sys.exit(1)

    # Save pairs for reproducibility and inspection
    pairs_path = output_dir / 'coexp_gene_pairs.txt'
    with open(pairs_path, 'w') as f:
        f.write("# Co-expression gene pairs (unified to human Ensembl IDs where possible)\n")
        f.write(f"# pcc_threshold={pcc_threshold}, n_sample_cells={n_sample_cells}\n")
        f.write(f"# Total: {len(unique_pairs):,} pairs\n")
        f.write("# gene_id_1\tgene_id_2\n")
        for g1, g2 in unique_pairs:
            f.write(f"{g1}\t{g2}\n")
    logger.info(f"  Pairs saved: {pairs_path}")

    # ═══════════════════════════════════════════════════════════════════════════
    # Phase 4: Train gene2vec
    # ═══════════════════════════════════════════════════════════════════════════
    logger.info("\nPhase 4: Training gene2vec")
    logger.info("-" * 60)

    model = train_gene2vec(
        gene_pairs=unique_pairs,
        vector_size=vector_size,
        window=window,
        sg=sg,
        epochs=epochs,
        workers=workers,
    )

    model_path = output_dir / 'coexp_gene2vec.model'
    model.save(str(model_path))
    logger.info(f"gene2vec model saved: {model_path}")

    # ═══════════════════════════════════════════════════════════════════════════
    # Phase 5: Save embeddings
    # ═══════════════════════════════════════════════════════════════════════════
    logger.info("\nPhase 5: Saving embeddings")
    logger.info("-" * 60)

    embeddings: Dict[str, np.ndarray] = {
        gene_id: model.wv[gene_id].astype(np.float32)
        for gene_id in model.wv.index_to_key
    }

    emb_path = output_dir / 'coexp_embeddings.pkl'
    with open(emb_path, 'wb') as f:
        pickle.dump(embeddings, f, protocol=4)
    logger.info(f"Embeddings saved: {emb_path}  ({len(embeddings):,} genes × {vector_size}d)")

    # ═══════════════════════════════════════════════════════════════════════════
    # Phase 6: Manifest
    # ═══════════════════════════════════════════════════════════════════════════
    write_manifest(
        output_dir=output_dir,
        config=config,
        rat_to_human_path=rat_to_human_path,
        qc_h5ad_dir=qc_h5ad_dir,
        n_files_processed=n_files_processed,
        n_files_skipped=n_files_skipped,
        n_raw_pairs_total=n_raw_pairs_total,
        n_unique_pairs=len(unique_pairs),
        n_embeddings=len(embeddings),
        vector_size=vector_size,
        file_metas=file_metas,
        elapsed=time.time() - t_start,
        dry_run=args.dry_run,
    )

    # ── Final summary ──────────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    logger.info("=" * 60)
    logger.info("STAGE 6 STEP 1 COMPLETE — Co-expression Embedding")
    logger.info("=" * 60)
    logger.info(f"  Files processed:  {n_files_processed}/{len(h5ad_files)}")
    logger.info(f"  Files skipped:    {n_files_skipped}")
    logger.info(f"  Unique pairs:     {len(unique_pairs):,}")
    logger.info(f"  Genes embedded:   {len(embeddings):,}")
    logger.info(f"  Elapsed:          {elapsed:.1f}s  ({elapsed/3600:.1f}h)")
    logger.info(f"\n  Output directory: {output_dir}")
    logger.info(f"  Primary output:   {emb_path}")
    logger.info("")
    logger.info("Next step:")
    logger.info(
        "  python pipeline/06_prior_knowledge/build_family_embedding.py"
    )


if __name__ == '__main__':
    main()