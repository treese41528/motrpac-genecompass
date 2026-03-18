#!/usr/bin/env python3
"""
build_grn_transfer.py — Stage 6, Step 4: GRN Embedding via PECA2vec Transfer

Pipeline position:
    Stage 6, Step 1: build_coexp_embedding.py
    Stage 6, Step 2: build_family_embedding.py
    Stage 6, Step 3: build_promoter_embedding.py
    Stage 6, Step 4: build_grn_transfer.py         <- THIS SCRIPT

Purpose:
    Construct rat GRN prior knowledge embeddings by direct transfer from
    GeneCompass's pre-trained PECA2vec embeddings (human and mouse), using
    Ensembl ortholog mapping to bridge to rat gene tokens.

    GeneCompass trained 768-dim GRN embeddings using PECA2 applied to ENCODE
    paired RNA-seq + ATAC-seq data across 76 human and 84 mouse cell/tissue
    contexts. These embeddings encode transcription factor regulatory
    relationships in 768-dimensional space. Since rat-mouse and rat-human
    evolutionary distances are short (~12-24 Mya and ~75 Mya respectively),
    and TF DNA-binding domains are among the most conserved protein families,
    direct transfer via ortholog mapping is scientifically well-justified.

    Data leakage note:
        MoTrPAC ATAC-seq data is intentionally NOT used here. The PECA2vec
        embeddings are derived from ENCODE (human/mouse), which is completely
        independent of MoTrPAC. This avoids circular validation when applying
        the fine-tuned model to MoTrPAC data in Aims 2 and 3.

Mapping chain:
    For each rat gene token (unified ID = ENSG for T1-T3, ENSRNOG for T4):

    Primary path (human PECA2vec):
        rat token -> rat_to_human_mapping.pickle -> human ENSG
        -> Gene_id_name_dict (inverted) -> human symbol
        -> human_PECA_vec.pickle[symbol] -> 768-dim vector

    Fallback path (mouse PECA2vec, for genes missing from human):
        rat ENSRNOG -> rat_mouse_orthologs.tsv -> mouse ENSMUSG
        -> mouse_ensembl_to_symbol (from BioMart) -> mouse symbol
        -> mouse_PECA_vec.pickle[symbol] -> 768-dim vector

    Zero vector:
        T4 genes with no human or mouse ortholog found in PECA2vec.

Key design decisions:
    - Gene ID unification follows Stage 3: ENSG for T1-T3, ENSRNOG for T4.
      This is identical to all other Stage 6 embeddings.
    - Human is tried first (priority) because GeneCompass's token dictionary
      uses human Ensembl IDs as the canonical unified space. 28,291 human
      genes are covered in human_PECA_vec vs 27,443 mouse genes in
      mouse_PECA_vec.
    - No gene2vec retraining needed — the 768-dim vectors are already trained
      by GeneCompass's PECA2 pipeline and used directly.
    - Mouse fallback is particularly useful for T4 rat-specific genes that
      have a mouse ortholog even though they lack a human ortholog.
    - The Gene_id_name_dict maps ENSG -> symbol (128K entries, human+mouse).
      It is inverted at runtime to symbol -> ENSG for lookup.

Inputs (all paths from pipeline_config.yaml):
    Stage 3 -> ortholog_dir/rat_token_mapping.tsv
    Stage 3 -> ortholog_dir/rat_to_human_mapping.pickle
    BioMart -> biomart.rat_mouse_orthologs (for mouse fallback)
    GeneCompass -> paths.genecompass_dir/prior_knowledge/PECA2vec/
                       human_PECA_vec.pickle
                       mouse_PECA_vec.pickle
    GeneCompass -> paths.genecompass_dir/prior_knowledge/gene_list/
                       Gene_id_name_dict_human_mouse.pickle

Outputs (all to paths.prior_knowledge_dir):
    grn_embeddings.pkl              <- primary output
    grn_transfer_coverage.tsv       <- per-gene coverage report
    stage6_grn_manifest.json

Config required in pipeline_config.yaml:
    prior_knowledge:
      grn:
        enabled: true
        method: "transfer"

Usage:
    python pipeline/06_prior_knowledge/build_grn_transfer.py
    python pipeline/06_prior_knowledge/build_grn_transfer.py --dry-run
    python pipeline/06_prior_knowledge/build_grn_transfer.py -v

Author: Tim Reese Lab / Claude
Date: March 2026
"""

import argparse
import csv
import json
import logging
import os
import pickle
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Set, Tuple

import numpy as np

# -----------------------------------------------------------------------------
# Project path bootstrap
# -----------------------------------------------------------------------------
_PROJECT_ROOT = Path(os.environ.get('PIPELINE_ROOT', '.')).resolve()
sys.path.insert(0, str(_PROJECT_ROOT / 'lib'))
from gene_utils import load_config, resolve_path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
)
logger = logging.getLogger(__name__)

VECTOR_SIZE = 768  # Fixed — matches GeneCompass PECA2vec


# =============================================================================
# SECTION 1: LOAD PECA2VEC EMBEDDINGS
# =============================================================================

def load_peca2vec(
    human_path: Path,
    mouse_path: Path,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Load GeneCompass PECA2vec pickle files.

    Returns:
        human_vecs: {gene_symbol: np.ndarray(768,)}
        mouse_vecs: {gene_symbol: np.ndarray(768,)}
    """
    logger.info(f"  Loading human PECA2vec: {human_path}")
    with open(human_path, 'rb') as f:
        human_vecs = pickle.load(f)
    logger.info(f"  Human PECA2vec: {len(human_vecs):,} gene symbols")

    logger.info(f"  Loading mouse PECA2vec: {mouse_path}")
    with open(mouse_path, 'rb') as f:
        mouse_vecs = pickle.load(f)
    logger.info(f"  Mouse PECA2vec: {len(mouse_vecs):,} gene symbols")

    # Normalise all vectors to float32
    human_vecs = {k: v.astype(np.float32) for k, v in human_vecs.items()}
    mouse_vecs = {k: v.astype(np.float32) for k, v in mouse_vecs.items()}

    return human_vecs, mouse_vecs


# =============================================================================
# SECTION 2: BUILD SYMBOL LOOKUP TABLES
# =============================================================================

def build_ensg_to_symbol(gene_id_dict_path: Path) -> Dict[str, str]:
    """Load GeneCompass Gene_id_name_dict: {ENSG: symbol}.

    The dict covers 128K entries combining human and mouse Ensembl IDs.
    Keys: Ensembl gene IDs (ENSG* for human, ENSMUSG* for mouse)
    Values: gene symbols
    """
    logger.info(f"  Loading Gene_id_name_dict: {gene_id_dict_path}")
    with open(gene_id_dict_path, 'rb') as f:
        ensg_to_symbol = pickle.load(f)
    logger.info(f"  ENSG->symbol entries: {len(ensg_to_symbol):,}")
    return ensg_to_symbol


def build_symbol_to_ensg(ensg_to_symbol: Dict[str, str]) -> Dict[str, str]:
    """Invert Gene_id_name_dict to {symbol: ENSG}.

    Where multiple ENSG IDs map to the same symbol, the first encountered
    is kept. In practice this is rare and does not affect coverage meaningfully.
    """
    symbol_to_ensg: Dict[str, str] = {}
    for ensg, symbol in ensg_to_symbol.items():
        if symbol not in symbol_to_ensg:
            symbol_to_ensg[symbol] = ensg
    logger.info(f"  Inverted symbol->ENSG: {len(symbol_to_ensg):,} unique symbols")
    return symbol_to_ensg


def load_rat_mouse_symbol_map(
    rat_mouse_orthologs_path: Path,
    ensg_to_symbol: Dict[str, str],
) -> Dict[str, str]:
    """Build {rat_ensrnog: mouse_symbol} for the mouse fallback path.

    rat_mouse_orthologs.tsv columns (BioMart):
        ensembl_gene_id          (rat ENSRNOG)
        mmusculus_homolog_ensembl_gene   (mouse ENSMUSG)
        mmusculus_homolog_associated_gene_name
        ...

    We prefer the gene name column directly from BioMart if present,
    otherwise fall back to Gene_id_name_dict lookup.
    """
    if not rat_mouse_orthologs_path.exists():
        logger.warning(
            f"  rat_mouse_orthologs.tsv not found: {rat_mouse_orthologs_path}\n"
            f"  Mouse fallback path disabled."
        )
        return {}

    logger.info(f"  Loading rat-mouse orthologs: {rat_mouse_orthologs_path}")
    rat_to_mouse_symbol: Dict[str, str] = {}
    n_rows = n_mapped = 0

    with open(rat_mouse_orthologs_path) as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            n_rows += 1
            rat_id   = row.get('Gene stable ID', '').strip()
            mouse_id = row.get('Mouse gene stable ID', '').strip()
            # Prefer the gene name column from BioMart
            mouse_sym = row.get('Mouse gene name', '').strip()

            if not rat_id or not mouse_id:
                continue

            # Fall back to Gene_id_name_dict if BioMart name is empty
            if not mouse_sym:
                mouse_sym = ensg_to_symbol.get(mouse_id, '')

            if mouse_sym:
                rat_to_mouse_symbol[rat_id] = mouse_sym
                n_mapped += 1

    logger.info(
        f"  Rat-mouse ortholog pairs: {n_rows:,} total | "
        f"{n_mapped:,} with resolvable mouse symbol"
    )
    return rat_to_mouse_symbol


# =============================================================================
# SECTION 3: LOAD RAT GENE LIST
# =============================================================================

def load_rat_genes(
    ortholog_dir: Path,
) -> Tuple[list, Dict[str, str]]:
    """Load rat token list and rat->human mapping from Stage 3.

    Returns:
        genes:        list of {rat_gene, rat_symbol, tier, biotype}
        rat_to_human: {rat_ensrnog: human_ensg}
    """
    mapping_path = ortholog_dir / 'rat_token_mapping.tsv'
    pkl_path     = ortholog_dir / 'rat_to_human_mapping.pickle'

    for p in (mapping_path, pkl_path):
        if not p.exists():
            raise FileNotFoundError(
                f"Stage 3 output not found: {p}\n"
                f"  -> Run Stage 3 first: python run_stage3.py"
            )

    with open(pkl_path, 'rb') as f:
        rat_to_human: Dict[str, str] = pickle.load(f)

    genes = []
    with open(mapping_path) as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            if row.get('tier', '') == 'excluded':
                continue
            genes.append({
                'rat_gene':   row['rat_gene'].strip(),
                'rat_symbol': row.get('rat_symbol', '').strip(),
                'tier':       row.get('tier', '').strip(),
                'biotype':    row.get('biotype', '').strip(),
            })

    logger.info(
        f"  Stage 3 genes loaded: {len(genes):,} | "
        f"rat->human pairs: {len(rat_to_human):,}"
    )
    return genes, rat_to_human


# =============================================================================
# SECTION 4: TRANSFER EMBEDDINGS
# =============================================================================

def transfer_embeddings(
    genes:               list,
    rat_to_human:        Dict[str, str],
    ensg_to_symbol:      Dict[str, str],
    rat_to_mouse_symbol: Dict[str, str],
    human_vecs:          Dict[str, np.ndarray],
    mouse_vecs:          Dict[str, np.ndarray],
) -> Tuple[Dict[str, np.ndarray], list]:
    """Transfer PECA2vec embeddings to rat gene tokens.

    For each rat gene:
      1. Determine unified ID: ENSG (T1-T3) or ENSRNOG (T4)
      2. Primary: look up ENSG -> symbol -> human_PECA_vec
      3. Fallback: look up ENSRNOG -> mouse symbol -> mouse_PECA_vec
      4. Miss: zero vector

    Returns:
        embeddings:     {unified_id: np.ndarray(768,)}
        coverage_rows:  list of per-gene coverage dicts for TSV report
    """
    embeddings:    Dict[str, np.ndarray] = {}
    coverage_rows: list = []

    # Invert ENSG->symbol for fast symbol lookup
    # (we need both directions)
    symbol_to_ensg = build_symbol_to_ensg(ensg_to_symbol)

    n_human_hit  = 0
    n_mouse_hit  = 0
    n_zero       = 0

    zero_vec = np.zeros(VECTOR_SIZE, dtype=np.float32)

    for gene in genes:
        rat_id    = gene['rat_gene']
        tier      = gene['tier']

        # Unified ID: T1-T3 use human ENSG, T4 keep ENSRNOG
        human_ensg  = rat_to_human.get(rat_id)
        unified_id  = human_ensg if human_ensg else rat_id

        source = 'zero'
        vec    = zero_vec.copy()

        # ── Primary path: human PECA2vec ─────────────────────────────────────
        if human_ensg:
            human_symbol = ensg_to_symbol.get(human_ensg, '')
            if human_symbol and human_symbol in human_vecs:
                vec    = human_vecs[human_symbol].copy()
                source = 'human_peca2vec'
                n_human_hit += 1

        # ── Fallback path: mouse PECA2vec ─────────────────────────────────────
        if source == 'zero' and rat_id in rat_to_mouse_symbol:
            mouse_symbol = rat_to_mouse_symbol[rat_id]
            if mouse_symbol in mouse_vecs:
                vec    = mouse_vecs[mouse_symbol].copy()
                source = 'mouse_peca2vec'
                n_mouse_hit += 1

        if source == 'zero':
            n_zero += 1

        embeddings[unified_id] = vec
        coverage_rows.append({
            'rat_gene':   rat_id,
            'unified_id': unified_id,
            'tier':       tier,
            'biotype':    gene['biotype'],
            'rat_symbol': gene['rat_symbol'],
            'source':     source,
        })

    total = len(genes)
    logger.info(
        f"  Transfer complete: {total:,} genes total\n"
        f"    Human PECA2vec:  {n_human_hit:,}  ({n_human_hit/total*100:.1f}%)\n"
        f"    Mouse PECA2vec:  {n_mouse_hit:,}  ({n_mouse_hit/total*100:.1f}%)\n"
        f"    Zero vector:     {n_zero:,}  ({n_zero/total*100:.1f}%)"
    )

    return embeddings, coverage_rows


# =============================================================================
# SECTION 5: MANIFEST
# =============================================================================

def write_manifest(
    output_dir: Path,
    config:     dict,
    n_genes:    int,
    n_human:    int,
    n_mouse:    int,
    n_zero:     int,
    elapsed:    float,
    dry_run:    bool,
    status:     str,
) -> None:
    manifest = {
        'stage':     '6_step4_grn_embedding',
        'script':    'build_grn_transfer.py',
        'method':    'transfer',
        'generated': datetime.now().isoformat() + 'Z',
        'status':    status,
        'dry_run':   dry_run,
        'elapsed_seconds': round(elapsed, 1),
        'description': (
            'Cross-species GRN embedding transfer from GeneCompass PECA2vec. '
            'Human PECA2vec used as primary source (76 human ENCODE GRNs), '
            'mouse PECA2vec as fallback (84 mouse ENCODE GRNs). '
            'MoTrPAC ATAC-seq intentionally excluded to prevent data leakage.'
        ),
        'sources': {
            'human_peca2vec': 'vendor/GeneCompass/prior_knowledge/PECA2vec/human_PECA_vec.pickle',
            'mouse_peca2vec': 'vendor/GeneCompass/prior_knowledge/PECA2vec/mouse_PECA_vec.pickle',
            'gene_id_dict':   'vendor/GeneCompass/prior_knowledge/gene_list/Gene_id_name_dict_human_mouse.pickle',
        },
        'outputs': {
            'genes_total':     n_genes,
            'human_peca2vec':  n_human,
            'mouse_peca2vec':  n_mouse,
            'zero_vector':     n_zero,
            'embedding_dimension': VECTOR_SIZE,
        },
    }

    manifest_path = output_dir / 'stage6_grn_manifest.json'
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"  Manifest written: {manifest_path}")


# =============================================================================
# SECTION 6: MAIN
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Stage 6 Step 4: GRN Embedding via PECA2vec Transfer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Transfers GeneCompass PECA2vec GRN embeddings (human + mouse) to rat tokens
via Ensembl ortholog mapping. No gene2vec retraining required.

The PECA2vec embeddings were trained by GeneCompass using PECA2 applied to
76 human and 84 mouse ENCODE paired RNA-seq + ATAC-seq datasets.

Usage:
  python pipeline/06_prior_knowledge/build_grn_transfer.py
  python pipeline/06_prior_knowledge/build_grn_transfer.py --dry-run
        """,
    )
    parser.add_argument('--dry-run', action='store_true',
                        help='Validate all inputs without running transfer')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    t_start = time.time()

    # ── Load config ───────────────────────────────────────────────────────────
    config    = load_config(_PROJECT_ROOT / 'config' / 'pipeline_config.yaml')
    paths_cfg = config.get('paths', {})
    bm_cfg    = config.get('biomart', {})

    # ── Resolve paths ─────────────────────────────────────────────────────────
    ortholog_dir   = resolve_path(config, paths_cfg['ortholog_dir'])
    output_dir     = resolve_path(config, paths_cfg['prior_knowledge_dir'])
    gc_prior_dir   = resolve_path(
        config, paths_cfg.get('genecompass_dir', 'vendor/GeneCompass')
    ) / 'prior_knowledge'

    human_peca_path  = gc_prior_dir / 'PECA2vec' / 'human_PECA_vec.pickle'
    mouse_peca_path  = gc_prior_dir / 'PECA2vec' / 'mouse_PECA_vec.pickle'
    gene_id_dict_path = gc_prior_dir / 'gene_list' / 'Gene_id_name_dict_human_mouse.pickle'
    rat_mouse_path   = resolve_path(
        config, bm_cfg.get('rat_mouse_orthologs', 'data/references/biomart/rat_mouse_orthologs.tsv')
    )

    logger.info("=" * 60)
    logger.info("Stage 6 Step 4: GRN Embedding (PECA2vec Transfer)")
    logger.info("=" * 60)
    logger.info(f"  Project root:    {_PROJECT_ROOT}")
    logger.info(f"  Output dir:      {output_dir}")
    logger.info(f"  Human PECA2vec:  {human_peca_path}")
    logger.info(f"  Mouse PECA2vec:  {mouse_peca_path}")
    logger.info(f"  Gene ID dict:    {gene_id_dict_path}")
    logger.info(f"  Rat-mouse orth:  {rat_mouse_path}")
    logger.info(f"  Dry run:         {args.dry_run}")

    # ── Validate inputs ───────────────────────────────────────────────────────
    errors = []
    for label, path in [
        ('human_PECA_vec', human_peca_path),
        ('mouse_PECA_vec', mouse_peca_path),
        ('Gene_id_name_dict', gene_id_dict_path),
        ('rat_token_mapping', ortholog_dir / 'rat_token_mapping.tsv'),
        ('rat_to_human_mapping', ortholog_dir / 'rat_to_human_mapping.pickle'),
    ]:
        if not path.exists():
            errors.append(f"  MISSING: {label} — {path}")
        elif path.stat().st_size < 1000:
            errors.append(f"  STUB/EMPTY: {label} — {path} ({path.stat().st_size} bytes)")
        else:
            logger.info(f"  {label}: {path.stat().st_size / 1e6:.1f} MB [OK]")

    # Rat-mouse orthologs optional (enables mouse fallback)
    if rat_mouse_path.exists():
        logger.info(f"  rat_mouse_orthologs: {rat_mouse_path.stat().st_size / 1e6:.1f} MB [OK]")
    else:
        logger.warning(
            f"  rat_mouse_orthologs not found: {rat_mouse_path}\n"
            f"  Mouse fallback path will be disabled."
        )

    if errors:
        for e in errors:
            logger.error(e)
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        logger.info("\nDry run complete — all required inputs validated.")
        write_manifest(output_dir, config, 0, 0, 0, 0,
                       time.time() - t_start, True, 'dry_run')
        return

    # =========================================================================
    # Phase 1: Load all reference data
    # =========================================================================
    logger.info("\nPhase 1: Loading reference data")
    logger.info("-" * 60)

    human_vecs, mouse_vecs = load_peca2vec(human_peca_path, mouse_peca_path)
    ensg_to_symbol         = build_ensg_to_symbol(gene_id_dict_path)
    rat_to_mouse_symbol    = load_rat_mouse_symbol_map(rat_mouse_path, ensg_to_symbol)

    # =========================================================================
    # Phase 2: Load rat gene list
    # =========================================================================
    logger.info("\nPhase 2: Loading rat gene list from Stage 3")
    logger.info("-" * 60)

    genes, rat_to_human = load_rat_genes(ortholog_dir)

    # =========================================================================
    # Phase 3: Transfer embeddings
    # =========================================================================
    logger.info("\nPhase 3: Transferring PECA2vec embeddings to rat tokens")
    logger.info("-" * 60)

    embeddings, coverage_rows = transfer_embeddings(
        genes, rat_to_human, ensg_to_symbol,
        rat_to_mouse_symbol, human_vecs, mouse_vecs,
    )

    # =========================================================================
    # Phase 4: Save outputs
    # =========================================================================
    logger.info("\nPhase 4: Saving outputs")
    logger.info("-" * 60)

    # Primary output
    emb_path = output_dir / 'grn_embeddings.pkl'
    with open(emb_path, 'wb') as f:
        pickle.dump(embeddings, f, protocol=4)
    logger.info(f"  Embeddings saved: {emb_path}  ({len(embeddings):,} genes × {VECTOR_SIZE}d)")

    # Coverage report
    cov_path = output_dir / 'grn_transfer_coverage.tsv'
    with open(cov_path, 'w', newline='') as f:
        if coverage_rows:
            writer = csv.DictWriter(
                f, fieldnames=list(coverage_rows[0].keys()), delimiter='\t'
            )
            writer.writeheader()
            writer.writerows(coverage_rows)
    logger.info(f"  Coverage report: {cov_path}")

    # Summary stats for manifest
    n_human = sum(1 for r in coverage_rows if r['source'] == 'human_peca2vec')
    n_mouse = sum(1 for r in coverage_rows if r['source'] == 'mouse_peca2vec')
    n_zero  = sum(1 for r in coverage_rows if r['source'] == 'zero')

    write_manifest(
        output_dir, config,
        n_genes=len(genes),
        n_human=n_human,
        n_mouse=n_mouse,
        n_zero=n_zero,
        elapsed=time.time() - t_start,
        dry_run=False,
        status='complete',
    )

    # Key breakdown
    n_ensg    = sum(1 for k in embeddings if k.startswith('ENSG'))
    n_ensrnog = sum(1 for k in embeddings if k.startswith('ENSRNOG'))

    elapsed = time.time() - t_start
    logger.info("=" * 60)
    logger.info("STAGE 6 STEP 4 COMPLETE — GRN Embedding (PECA2vec Transfer)")
    logger.info("=" * 60)
    logger.info(f"  Total genes:        {len(genes):,}")
    logger.info(f"  Human PECA2vec:     {n_human:,}  ({n_human/len(genes)*100:.1f}%)")
    logger.info(f"  Mouse PECA2vec:     {n_mouse:,}  ({n_mouse/len(genes)*100:.1f}%)")
    logger.info(f"  Zero vector (T4/no ortholog): {n_zero:,}  ({n_zero/len(genes)*100:.1f}%)")
    logger.info(f"  Key breakdown:      {n_ensg:,} ENSG | {n_ensrnog:,} ENSRNOG")
    logger.info(f"  Elapsed:            {elapsed:.1f}s")
    logger.info(f"\n  Primary output: {emb_path}")


if __name__ == '__main__':
    main()