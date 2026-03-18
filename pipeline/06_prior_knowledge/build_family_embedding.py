#!/usr/bin/env python3
"""
build_family_embedding.py — Stage 6, Step 2: Gene Family Embedding

Pipeline position:
    Stage 1: Data Harvesting & QC
    Stage 2: Gene Universe & Cell QC
    Stage 3: build_ortholog_mapping.py  → rat_token_mapping.tsv
                                          rat_to_human_mapping.pickle
    Stage 4: Gene Medians
    Stage 5: Reference Assembly & Tokenization
    Stage 6, Step 1: build_coexp_embedding.py
    Stage 6, Step 2: build_family_embedding.py          ← THIS SCRIPT

Purpose:
    Compute 768-dimensional gene family embeddings for all rat gene tokens,
    following the GeneCompass prior knowledge construction protocol.

    Method:
      1. Load human gene family assignments from HGNC.
      2. Derive mouse gene families via BioMart mouse↔human ortholog TSV.
      3. Derive rat gene families via Stage 3's authoritative rat→human mapping
         (inverted), which uses the tiered ortholog resolution logic — NOT the
         raw BioMart TSV. This ensures consistency with token assignments.
      4. For each family, generate all pairwise gene combinations across all
         three species, unified to human Ensembl IDs where possible.
         Rat T4 (new-token) genes are represented by their ENSRNOG ID,
         placing them in embedding space relative to their family members.
      5. Train gene2vec (Word2Vec Skip-Gram, 768 dimensions) on the pair list.
      6. Save embeddings keyed by the same unified ID used in the token lookup.

Key design decisions:
    - Rat ortholog mapping consumed from Stage 3 outputs (rat_to_human_mapping.pickle),
      NOT re-derived from raw BioMart TSVs. Stage 3 is the single source of truth
      for rat gene token assignments.
    - Mouse ortholog mapping read from mouse_human_orthologs.tsv (config: biomart
      .mouse_human_orthologs). Mouse is not in Stage 3 because Stage 3 is rat-centric.
    - human_to_species dicts are sets (defaultdict(set)), not scalars, so all rat
      paralogs of a given human gene are included in derived families. This is
      critical for expanded rat gene families (cytochrome P450s, olfactory
      receptors, carboxylesterases) where one human gene maps to many rat genes.
    - Only T4 genes strictly need new embeddings (T1–T3 tokens inherit pre-trained
      knowledge from GeneCompass). Both are computed for completeness; the fine-
      tuning script selects which embeddings to use based on tier.

Inputs (all paths from pipeline_config.yaml):
    Stage 3 → ortholog_dir/rat_token_mapping.tsv        — canonical gene list + tiers
    Stage 3 → ortholog_dir/rat_to_human_mapping.pickle  — {rat_ensrnog: human_ensg}
    BioMart → biomart.mouse_human_orthologs             — mouse↔human TSV (new config entry)
    HGNC    → paths.hgnc_file                           — gene family TSV
    Config  → prior_knowledge.gene_family.*             — tunable parameters

Outputs (all to paths.prior_knowledge_dir):
    family_gene_pairs.txt       — Gene pairs used for training (human Ensembl IDs)
    family_gene2vec.model       — Gensim Word2Vec model
    family_embeddings.pkl       — {gene_id: np.ndarray(768,)} — primary output
    stage6_family_manifest.json — Provenance record

Usage:
    python pipeline/06_prior_knowledge/build_family_embedding.py
    python pipeline/06_prior_knowledge/build_family_embedding.py --dry-run
    python pipeline/06_prior_knowledge/build_family_embedding.py -v

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
from collections import defaultdict
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Optional heavy imports — checked at startup so the error is actionable
# ─────────────────────────────────────────────────────────────────────────────
_MISSING = []
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

# HGNC primary download URL (Google Cloud Storage — stable)
_HGNC_GCS_URL = (
    'https://storage.googleapis.com/public-download-files/hgnc/tsv/tsv/'
    'hgnc_complete_set.txt'
)
# HGNC FTP fallback (EBI — more stable long-term than CGI endpoint)
_HGNC_FTP_URL = (
    'https://ftp.ebi.ac.uk/pub/databases/genenames/hgnc/tsv/'
    'hgnc_complete_set.txt'
)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1: DATA LOADING
# ═════════════════════════════════════════════════════════════════════════════

def load_rat_token_mapping(ortholog_dir: Path) -> Tuple[Dict[str, str], Set[str]]:
    """Load Stage 3 rat_token_mapping.tsv.

    Returns:
        rat_biotype: {rat_ensrnog: biotype}
        t4_genes:    set of rat ENSRNOG IDs assigned new tokens (Tier 4)
    """
    mapping_path = ortholog_dir / 'rat_token_mapping.tsv'
    if not mapping_path.exists():
        raise FileNotFoundError(
            f"rat_token_mapping.tsv not found: {mapping_path}\n"
            f"  → Run Stage 3 first:  python run_stage3.py"
        )

    rat_biotype: Dict[str, str] = {}
    t4_genes: Set[str] = set()

    with open(mapping_path, 'r') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            gene = row['rat_gene'].strip()
            tier = row.get('tier', '').strip()
            biotype = row.get('biotype', '').strip()
            if gene:
                rat_biotype[gene] = biotype
                if tier == 'T4_new_token':
                    t4_genes.add(gene)

    logger.info(f"Stage 3 rat_token_mapping: {len(rat_biotype):,} genes, "
                f"{len(t4_genes):,} T4 (new token) genes")
    return rat_biotype, t4_genes


def load_rat_to_human_mapping(ortholog_dir: Path) -> Dict[str, str]:
    """Load Stage 3 rat_to_human_mapping.pickle.

    This is the authoritative rat→human gene ID mapping produced by the
    tiered ortholog resolution in Stage 3. Do NOT re-derive this from BioMart.

    Returns: {rat_ensrnog: human_ensg}
    """
    pkl_path = ortholog_dir / 'rat_to_human_mapping.pickle'
    if not pkl_path.exists():
        raise FileNotFoundError(
            f"rat_to_human_mapping.pickle not found: {pkl_path}\n"
            f"  → Run Stage 3 first:  python run_stage3.py"
        )
    with open(pkl_path, 'rb') as f:
        mapping = pickle.load(f)
    logger.info(f"Stage 3 rat_to_human_mapping: {len(mapping):,} rat→human pairs")
    return mapping


def build_human_to_rat(rat_to_human: Dict[str, str]) -> Dict[str, Set[str]]:
    """Invert Stage 3 rat→human mapping to human→{rat genes}.

    A single human gene can have multiple rat orthologs (expanded gene families
    such as cytochrome P450s). This returns a set-valued dict so ALL rat
    paralogs are included in derived families, not just the first encountered.

    Args:
        rat_to_human: {rat_ensrnog: human_ensg}  (from Stage 3)

    Returns:
        {human_ensg: set of rat_ensrnog IDs}
    """
    human_to_rat: Dict[str, Set[str]] = defaultdict(set)
    for rat_id, human_id in rat_to_human.items():
        human_to_rat[human_id].add(rat_id)

    n_one2one = sum(1 for s in human_to_rat.values() if len(s) == 1)
    n_expanded = sum(1 for s in human_to_rat.values() if len(s) > 1)
    max_expansion = max((len(s) for s in human_to_rat.values()), default=0)
    logger.info(
        f"human_to_rat (inverted Stage 3): {len(human_to_rat):,} human genes "
        f"→ {n_one2one:,} one-to-one, {n_expanded:,} one-to-many "
        f"(max expansion: {max_expansion})"
    )
    return human_to_rat


def load_mouse_orthologs(tsv_path: Path) -> Dict[str, Set[str]]:
    """Load mouse↔human ortholog TSV from BioMart and return human→{mouse} mapping.

    BioMart TSV column layout (same format as rat_human_orthologs.tsv):
      col 0: Gene stable ID       — mouse Ensembl ID (ENSMUSG...)
      col 1: Human gene stable ID — human Ensembl ID (ENSG...)
      col 2: Human gene name      — human gene symbol
      col 3: Human homology type  — ortholog_one2one / one2many / many2many

    Uses a set-valued dict so all mouse paralogs of a given human gene are
    included in derived mouse families (same correctness fix as the rat case).

    Returns: {human_ensg: set of mouse_ensmusg IDs}
    """
    human_to_mouse: Dict[str, Set[str]] = defaultdict(set)

    with open(tsv_path, 'r') as f:
        reader = csv.DictReader(f, delimiter='\t')
        col_mouse = reader.fieldnames[0]  # "Gene stable ID"
        col_human = reader.fieldnames[1]  # "Human gene stable ID"

        for row in reader:
            mouse_id = row.get(col_mouse, '').strip()
            human_id = row.get(col_human, '').strip()
            if mouse_id and human_id:
                human_to_mouse[human_id].add(mouse_id)

    n_one2one = sum(1 for s in human_to_mouse.values() if len(s) == 1)
    n_expanded = sum(1 for s in human_to_mouse.values() if len(s) > 1)
    logger.info(
        f"Mouse orthologs from {tsv_path.name}: {len(human_to_mouse):,} human genes "
        f"→ {n_one2one:,} one-to-one, {n_expanded:,} one-to-many"
    )
    return human_to_mouse


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2: HGNC DOWNLOAD & PARSE
# ═════════════════════════════════════════════════════════════════════════════

def ensure_hgnc_data(hgnc_path: Path, skip_download: bool = False) -> Optional[Path]:
    """Verify HGNC file exists and is non-trivial. Download if missing."""
    if hgnc_path.exists() and hgnc_path.stat().st_size > 10_000:
        logger.info(f"HGNC data: {hgnc_path} ({hgnc_path.stat().st_size / 1024:.0f} KB)")
        return hgnc_path

    if hgnc_path.exists():
        logger.warning(f"HGNC file too small ({hgnc_path.stat().st_size} bytes) — may be corrupt")

    if skip_download:
        logger.error(f"HGNC data not found: {hgnc_path}")
        return None

    hgnc_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading HGNC gene family data...")

    for source_label, url in [
        ('Google Cloud Storage', _HGNC_GCS_URL),
        ('EBI FTP',              _HGNC_FTP_URL),
    ]:
        logger.info(f"  Trying: {source_label}  ({url[:80]}...)")
        try:
            import urllib.request
            urllib.request.urlretrieve(url, hgnc_path)
            size = hgnc_path.stat().st_size
            if size < 10_000:
                logger.warning(f"  File too small ({size} bytes), trying next source")
                continue
            logger.info(f"  Downloaded: {size / 1024:.0f} KB")
            return hgnc_path
        except Exception as exc:
            logger.warning(f"  Failed ({exc})")

    logger.error("All HGNC download sources failed.")
    return None


def parse_hgnc_families(
    hgnc_path: Path,
) -> Tuple[Dict[str, Set[str]], Dict[str, str], Dict[str, str]]:
    """Parse HGNC TSV to extract gene family assignments.

    HGNC columns of interest:
      Approved symbol    — gene symbol (e.g. TP53)
      Ensembl gene ID    — human Ensembl ID (e.g. ENSG00000141510)
      Gene family ID     — numeric family ID, pipe-separated for multi-family genes
      Gene family name   — human-readable name, pipe-separated

    Returns:
        families:    {family_id: set of human_ensembl_ids}
        fam_names:   {family_id: family_name_string}
        ens_to_sym:  {human_ensembl_id: gene_symbol}  (for sanity checks)
    """
    families: Dict[str, Set[str]] = defaultdict(set)
    fam_names: Dict[str, str] = {}
    ens_to_sym: Dict[str, str] = {}

    # Column name variants across HGNC file versions
    _SYM_COLS  = ['Approved symbol', 'Approved Symbol', 'symbol']
    _ENS_COLS  = ['Ensembl gene ID', 'Ensembl ID(supplied by Ensembl)', 'ensembl_gene_id']
    _FID_COLS  = ['Gene family ID', 'Gene Family ID', 'gene_family_id', 'gene_group_id']
    _FNAME_COLS = ['Gene family name', 'Gene Family Name', 'gene_family_name', 'gene_group']

    n_total = n_with_family = n_no_ensembl = 0

    with open(hgnc_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        fields = set(reader.fieldnames or [])

        def _pick(candidates):
            for c in candidates:
                if c in fields:
                    return c
            return None

        col_sym   = _pick(_SYM_COLS)
        col_ens   = _pick(_ENS_COLS)
        col_fid   = _pick(_FID_COLS)
        col_fname = _pick(_FNAME_COLS)

        for label, col in [
            ('symbol', col_sym), ('ensembl', col_ens),
            ('family_id', col_fid), ('family_name', col_fname),
        ]:
            if col:
                logger.info(f"  HGNC column [{label}] → '{col}'")
            else:
                logger.warning(f"  HGNC column [{label}] → NOT FOUND")

        if not col_ens or not col_fid:
            raise ValueError(
                "HGNC file is missing required Ensembl ID or Gene family ID column.\n"
                f"  Available columns: {sorted(fields)}"
            )

        for row in reader:
            n_total += 1

            ensembl_id = row.get(col_ens, '').strip()
            if not ensembl_id:
                n_no_ensembl += 1
                continue

            symbol = row.get(col_sym, '').strip() if col_sym else ''
            if symbol:
                ens_to_sym[ensembl_id] = symbol

            fam_id_str   = row.get(col_fid, '').strip()
            fam_name_str = row.get(col_fname, '').strip() if col_fname else ''

            if not fam_id_str:
                continue

            n_with_family += 1

            # Genes can belong to multiple families — pipe-separated
            fam_ids   = [x.strip() for x in fam_id_str.split('|') if x.strip()]
            fam_names_list = [x.strip() for x in fam_name_str.split('|')] \
                             if fam_name_str else []

            for i, fid in enumerate(fam_ids):
                families[fid].add(ensembl_id)
                if i < len(fam_names_list) and fid not in fam_names:
                    fam_names[fid] = fam_names_list[i]

    logger.info(
        f"HGNC parsed: {n_total:,} rows | "
        f"{n_total - n_no_ensembl:,} with Ensembl ID | "
        f"{n_with_family:,} with family assignment | "
        f"{len(families):,} families"
    )

    if families:
        top5 = sorted(families.items(), key=lambda x: len(x[1]), reverse=True)[:5]
        logger.info("  Largest families:")
        for fid, members in top5:
            logger.info(f"    {fam_names.get(fid, fid):40s}  ({len(members)} members)")

    return families, fam_names, ens_to_sym


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3: DERIVE SPECIES FAMILIES
# ═════════════════════════════════════════════════════════════════════════════

def derive_species_families(
    human_families: Dict[str, Set[str]],
    human_to_species: Dict[str, Set[str]],
    species_name: str,
    min_family_size: int = 2,
) -> Dict[str, Set[str]]:
    """Derive gene families for a target species via homolog mapping.

    For each human gene family, maps all human members to their species
    orthologs and collects the result as the derived family for that species.

    Critical requirement:
        human_to_species MUST be a set-valued dict {human_id: set(species_ids)}.
        A scalar-valued dict silently drops all paralogs beyond the first, which
        is catastrophic for expanded rat gene families (cytochrome P450s, etc.).

    Args:
        human_families:    {family_id: set of human_ensembl_ids}
        human_to_species:  {human_ensg: set of species_ensembl_ids}  ← must be set-valued
        species_name:      "mouse" or "rat" (logging only)
        min_family_size:   minimum members to keep a derived family

    Returns:
        {family_id: set of species_ensembl_ids}
    """
    # Sanity check: verify set-valued dict
    sample = next(iter(human_to_species.values()), None)
    if sample is not None and not isinstance(sample, (set, frozenset)):
        raise TypeError(
            f"human_to_species must be set-valued (got {type(sample).__name__}). "
            "Use build_human_to_rat() or load_mouse_orthologs() which return defaultdict(set)."
        )

    derived: Dict[str, Set[str]] = {}
    n_kept = n_dropped = 0
    total_genes: Set[str] = set()

    for fam_id, human_members in human_families.items():
        species_members: Set[str] = set()
        for human_gene in human_members:
            # Add ALL species orthologs of this human gene (not just the first)
            for sp_gene in human_to_species.get(human_gene, set()):
                species_members.add(sp_gene)

        if len(species_members) >= min_family_size:
            derived[fam_id] = species_members
            total_genes.update(species_members)
            n_kept += 1
        else:
            n_dropped += 1

    logger.info(
        f"Derived {species_name} families: {n_kept:,} kept "
        f"(≥{min_family_size} members), {n_dropped:,} dropped | "
        f"{len(total_genes):,} unique {species_name} genes in families"
    )
    return derived


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4: GENERATE GENE PAIRS
# ═════════════════════════════════════════════════════════════════════════════

def build_unified_id(
    gene_id: str,
    rat_to_human: Dict[str, str],
    mouse_to_human: Dict[str, str],
) -> str:
    """Map any species gene ID to its unified representation.

    Unification rules (same hierarchy as Stage 3 token logic):
      - Human Ensembl ID (ENSG...)   → itself (no lookup needed)
      - Mouse Ensembl ID (ENSMUSG...)→ human Ensembl ID via mouse_to_human
      - Rat Ensembl ID (ENSRNOG...)  → human Ensembl ID if in rat_to_human
                                       (ortholog-mapped, T1–T3 tiers)
                                     → its own ENSRNOG if NOT in rat_to_human
                                       (T4 new token — intentionally kept distinct)
    """
    if gene_id.startswith('ENSG'):
        return gene_id
    if gene_id.startswith('ENSMUSG'):
        return mouse_to_human.get(gene_id, gene_id)
    if gene_id.startswith('ENSRNOG'):
        return rat_to_human.get(gene_id, gene_id)  # T4 genes fall back to ENSRNOG
    return gene_id


def generate_family_gene_pairs(
    all_families: Dict[str, Dict[str, Set[str]]],
    rat_to_human: Dict[str, str],
    mouse_to_human: Dict[str, str],
) -> List[Tuple[str, str]]:
    """Generate all within-family pairwise gene combinations, unified to human IDs.

    For each family, every pair of members (across all three species) is a
    training example for gene2vec. Mouse and rat genes are first unified to
    their human Ensembl ID; rat T4 genes keep their ENSRNOG ID (they form
    their own new embedding, adjacent to their family members in space).

    After unification, duplicate pairs are removed (a human gene and its
    ortholog-mapped rat counterpart unify to the same ID, so self-pairs
    are naturally eliminated).

    Args:
        all_families:    {'human': {...}, 'mouse': {...}, 'rat': {...}}
        rat_to_human:    Stage 3 rat→human mapping
        mouse_to_human:  inverted mouse BioMart mapping {mouse_ensmusg: human_ensg}

    Returns:
        Deduplicated list of (unified_id_1, unified_id_2) tuples, sorted.
    """
    all_pairs: Set[Tuple[str, str]] = set()
    per_species: Dict[str, int] = {}

    for species, families in all_families.items():
        species_pairs = 0
        for fam_id, members in families.items():
            # Unify all gene IDs to human Ensembl space (or ENSRNOG for T4)
            unified_members: Set[str] = set()
            for gene_id in members:
                uid = build_unified_id(gene_id, rat_to_human, mouse_to_human)
                unified_members.add(uid)

            # Self-pairs eliminated by combinations() — unified dupes eliminated
            # by the outer set
            for g1, g2 in combinations(sorted(unified_members), 2):
                all_pairs.add((g1, g2))
                species_pairs += 1

        per_species[species] = species_pairs
        logger.info(
            f"  {species:5s}: {len(families):,} families → "
            f"{species_pairs:,} pairs (before dedup)"
        )

    unique_pairs = sorted(all_pairs)
    logger.info(
        f"  Total unique pairs after dedup: {len(unique_pairs):,} "
        f"(from {sum(per_species.values()):,} raw)"
    )
    return unique_pairs


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 5: TRAIN gene2vec
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
    """Train gene2vec (Word2Vec Skip-Gram) on gene family pairs.

    Each gene pair is a 2-token "sentence". With window=1 and Skip-Gram,
    the model learns: given gene A, predict its family partner gene B.
    Genes in the same families cluster together in the 768-dim embedding space.

    This is identical to the GeneCompass gene2vec protocol, scaled to 768
    dimensions (original gene2vec used 256; GeneCompass upsized to match
    the transformer hidden size).
    """
    if Word2Vec is None:
        raise ImportError("gensim is required: pip install gensim")

    sentences = [list(pair) for pair in gene_pairs]

    logger.info("=" * 60)
    logger.info("Training gene2vec (gene family)")
    logger.info("=" * 60)
    logger.info(f"  Training sentences (pairs): {len(sentences):,}")
    logger.info(f"  Vector size:                {vector_size}")
    logger.info(f"  Algorithm:                  {'Skip-gram' if sg == 1 else 'CBOW'}")
    logger.info(f"  Window:                     {window}")
    logger.info(f"  Epochs:                     {epochs}")
    logger.info(f"  Workers:                    {workers}")

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
    logger.info(f"  Vocabulary (genes with embeddings): {len(model.wv):,}")
    return model


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 6: SANITY CHECK
# ═════════════════════════════════════════════════════════════════════════════

def sanity_check(
    embeddings: Dict[str, np.ndarray],
    ens_to_sym: Dict[str, str],
    t4_genes: Set[str],
) -> None:
    """Log basic quality signals on the trained embeddings.

    Checks:
      1. Known same-family gene pairs should have high cosine similarity.
      2. Random gene pairs should have near-zero cosine similarity.
      3. T4 rat genes present in the embedding space.
    """
    def cosine(a, b):
        n = np.linalg.norm(a) * np.linalg.norm(b)
        return float(np.dot(a, b) / n) if n > 1e-8 else 0.0

    # Known same-family pairs (human Ensembl IDs, stable across releases)
    same_family_pairs = [
        ('ENSG00000141510', 'ENSG00000073282', 'TP53',   'TP63'),    # p53 family
        ('ENSG00000105991', 'ENSG00000105996', 'HOXA1',  'HOXA2'),   # HOX family
        ('ENSG00000108821', 'ENSG00000164692', 'COL1A1', 'COL1A2'),  # collagen family
    ]

    logger.info("Sanity check — same-family cosine similarities:")
    for g1, g2, n1, n2 in same_family_pairs:
        if g1 in embeddings and g2 in embeddings:
            sim = cosine(embeddings[g1], embeddings[g2])
            logger.info(f"  {n1:8s} ↔ {n2:8s} : {sim:.4f}  (same family, expect > 0)")
        else:
            logger.info(f"  {n1:8s} ↔ {n2:8s} : not in vocabulary")

    # Random baseline
    import random
    rng = random.Random(42)
    all_ids = list(embeddings.keys())
    if len(all_ids) >= 2:
        r1, r2 = rng.sample(all_ids, 2)
        sim = cosine(embeddings[r1], embeddings[r2])
        s1 = ens_to_sym.get(r1, r1[:15])
        s2 = ens_to_sym.get(r2, r2[:15])
        logger.info(f"  {s1:8s} ↔ {s2:8s} : {sim:.4f}  (random baseline, expect ≈ 0)")

    # T4 gene coverage
    t4_embedded = sum(1 for g in t4_genes if g in embeddings)
    logger.info(
        f"T4 gene coverage: {t4_embedded:,}/{len(t4_genes):,} new-token genes "
        f"have family embeddings ({t4_embedded / max(len(t4_genes), 1) * 100:.1f}%)"
    )
    if len(t4_genes) > 0 and t4_embedded == 0:
        logger.warning(
            "  No T4 genes have family embeddings. This means rat-specific genes "
            "were not assigned to any HGNC family. Check RGD integration or "
            "consider using the zero-vector fallback for T4 genes in fine-tuning."
        )


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
    input_paths: Dict[str, Path],
    n_human_families: int,
    n_mouse_families: int,
    n_rat_families: int,
    n_gene_pairs: int,
    n_embeddings: int,
    n_t4_genes: int,
    n_t4_embedded: int,
    vector_size: int,
    elapsed: float,
    dry_run: bool,
) -> None:
    pk = config.get('prior_knowledge', {})
    gf = pk.get('gene_family', {})

    manifest = {
        'stage': '6_step2_family_embedding',
        'script': 'build_family_embedding.py',
        'generated': datetime.utcnow().isoformat() + 'Z',
        'dry_run': dry_run,
        'elapsed_seconds': round(elapsed, 1),
        'parameters': {
            'min_family_size': gf.get('min_family_size', 2),
            'vector_size': vector_size,
            'window': gf.get('window', 1),
            'sg': gf.get('sg', 1),
            'epochs': gf.get('epochs', 30),
            'workers': gf.get('workers', 8),
        },
        'inputs': {
            name: {
                'path': str(path),
                'md5': file_md5(path) if path.exists() else None,
                'size_bytes': path.stat().st_size if path.exists() else None,
            }
            for name, path in input_paths.items()
        },
        'outputs': {
            'human_families': n_human_families,
            'mouse_families': n_mouse_families,
            'rat_families': n_rat_families,
            'gene_pairs': n_gene_pairs,
            'embeddings': n_embeddings,
            'embedding_dimension': vector_size,
            't4_genes_total': n_t4_genes,
            't4_genes_embedded': n_t4_embedded,
        },
        'config_snapshot': {
            'biomart': config.get('biomart', {}),
            'prior_knowledge': pk,
        },
    }

    manifest_path = output_dir / 'stage6_family_manifest.json'
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"Manifest written: {manifest_path}")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 8: MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Stage 6 Step 2: Gene Family Embedding',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Validate all inputs and configuration without training',
    )
    parser.add_argument(
        '-v', '--verbose', action='store_true',
        help='Enable DEBUG-level logging',
    )
    parser.add_argument(
        '--skip-download', action='store_true',
        help='Do not auto-download HGNC data if the file is missing',
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if _MISSING:
        logger.error(f"Missing required packages: {', '.join(_MISSING)}")
        logger.error("  Install:  pip install gensim")
        sys.exit(1)

    t_start = time.time()

    # ── Load configuration ──────────────────────────────────────────────────
    config = load_config(_PROJECT_ROOT / 'config' / 'pipeline_config.yaml')
    paths_cfg = config.get('paths', {})
    pk_cfg    = config.get('prior_knowledge', {})
    gf_cfg    = pk_cfg.get('gene_family', {})
    bm_cfg    = config.get('biomart', {})

    # ── Resolve all paths from config ───────────────────────────────────────
    ortholog_dir    = resolve_path(config, paths_cfg['ortholog_dir'])
    output_dir      = resolve_path(config, paths_cfg['prior_knowledge_dir'])
    hgnc_path       = resolve_path(config, paths_cfg['hgnc_file'])
    mouse_orth_path = resolve_path(config, bm_cfg['mouse_human_orthologs'])

    # Parameters — config with sensible defaults matching GeneCompass
    min_family_size = int(gf_cfg.get('min_family_size', 2))
    vector_size     = int(gf_cfg.get('vector_size', 768))
    window          = int(gf_cfg.get('window', 1))
    sg              = int(gf_cfg.get('sg', 1))
    epochs          = int(gf_cfg.get('epochs', 30))
    workers         = int(gf_cfg.get('workers', 8))

    logger.info("=" * 60)
    logger.info("Stage 6 Step 2: Gene Family Embedding")
    logger.info("=" * 60)
    logger.info(f"  Project root:  {_PROJECT_ROOT}")
    logger.info(f"  Output dir:    {output_dir}")
    logger.info(f"  HGNC file:     {hgnc_path}")
    logger.info(f"  Mouse orthologs: {mouse_orth_path}")
    logger.info(f"  min_family_size: {min_family_size}")
    logger.info(f"  vector_size:   {vector_size}")
    logger.info(f"  epochs:        {epochs}")
    logger.info(f"  dry_run:       {args.dry_run}")

    # ── Validate prerequisite files ─────────────────────────────────────────
    stage3_mapping_path = ortholog_dir / 'rat_token_mapping.tsv'
    stage3_pkl_path     = ortholog_dir / 'rat_to_human_mapping.pickle'

    missing = []
    for label, path in [
        ('Stage 3 rat_token_mapping.tsv',       stage3_mapping_path),
        ('Stage 3 rat_to_human_mapping.pickle', stage3_pkl_path),
        ('Mouse human orthologs TSV',           mouse_orth_path),
    ]:
        if not path.exists():
            missing.append(f"  MISSING: {label}\n           {path}")

    if missing:
        for m in missing:
            logger.error(m)
        sys.exit(1)

    logger.info("All prerequisite files found.")

    # ── HGNC data ────────────────────────────────────────────────────────────
    hgnc_resolved = ensure_hgnc_data(hgnc_path, skip_download=args.skip_download)
    if hgnc_resolved is None:
        logger.error("Cannot proceed without HGNC data.")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        logger.info("Dry run complete — all inputs validated.")
        return

    # ════════════════════════════════════════════════════════════════════════
    # Phase 1: Load Stage 3 authoritative rat gene data
    # ════════════════════════════════════════════════════════════════════════
    logger.info("\nPhase 1: Loading Stage 3 rat gene data")
    logger.info("-" * 60)

    rat_biotype, t4_genes = load_rat_token_mapping(ortholog_dir)
    rat_to_human = load_rat_to_human_mapping(ortholog_dir)

    # Build set-valued human→{rat genes} mapping from Stage 3 pickle
    human_to_rat = build_human_to_rat(rat_to_human)

    # ════════════════════════════════════════════════════════════════════════
    # Phase 2: Load mouse orthologs
    # ════════════════════════════════════════════════════════════════════════
    logger.info("\nPhase 2: Loading mouse orthologs")
    logger.info("-" * 60)

    human_to_mouse = load_mouse_orthologs(mouse_orth_path)

    # Build mouse→human for gene ID unification in pair generation
    mouse_to_human: Dict[str, str] = {}
    for human_id, mouse_set in human_to_mouse.items():
        for mouse_id in mouse_set:
            if mouse_id not in mouse_to_human:
                mouse_to_human[mouse_id] = human_id

    # ════════════════════════════════════════════════════════════════════════
    # Phase 3: Parse HGNC human gene families
    # ════════════════════════════════════════════════════════════════════════
    logger.info("\nPhase 3: Parsing HGNC gene families (human)")
    logger.info("-" * 60)

    human_families, fam_names, ens_to_sym = parse_hgnc_families(hgnc_resolved)

    if not human_families:
        logger.error("No families found in HGNC file. Check file format.")
        sys.exit(1)

    # ════════════════════════════════════════════════════════════════════════
    # Phase 4: Derive species families
    # ════════════════════════════════════════════════════════════════════════
    logger.info("\nPhase 4: Deriving mouse and rat gene families")
    logger.info("-" * 60)

    mouse_families = derive_species_families(
        human_families, human_to_mouse, 'mouse', min_family_size,
    )
    rat_families = derive_species_families(
        human_families, human_to_rat, 'rat', min_family_size,
    )

    all_families = {
        'human': human_families,
        'mouse': mouse_families,
        'rat':   rat_families,
    }

    # ════════════════════════════════════════════════════════════════════════
    # Phase 5: Generate gene pairs
    # ════════════════════════════════════════════════════════════════════════
    logger.info("\nPhase 5: Generating within-family gene pairs")
    logger.info("-" * 60)

    gene_pairs = generate_family_gene_pairs(all_families, rat_to_human, mouse_to_human)

    if not gene_pairs:
        logger.error("No gene pairs generated. Check HGNC file and ortholog data.")
        sys.exit(1)

    # Save pairs for inspection / reproducibility
    pairs_path = output_dir / 'family_gene_pairs.txt'
    with open(pairs_path, 'w') as f:
        f.write(f"# Gene family pairs (unified to human Ensembl IDs where possible)\n")
        f.write(f"# Total: {len(gene_pairs):,} pairs\n")
        f.write("# gene_id_1\tgene_id_2\n")
        for g1, g2 in gene_pairs:
            f.write(f"{g1}\t{g2}\n")
    logger.info(f"Gene pairs saved: {pairs_path}")

    # ════════════════════════════════════════════════════════════════════════
    # Phase 6: Train gene2vec
    # ════════════════════════════════════════════════════════════════════════
    logger.info("\nPhase 6: Training gene2vec")
    logger.info("-" * 60)

    model = train_gene2vec(
        gene_pairs=gene_pairs,
        vector_size=vector_size,
        window=window,
        sg=sg,
        epochs=epochs,
        workers=workers,
    )

    model_path = output_dir / 'family_gene2vec.model'
    model.save(str(model_path))
    logger.info(f"gene2vec model saved: {model_path}")

    # ════════════════════════════════════════════════════════════════════════
    # Phase 7: Save embeddings
    # ════════════════════════════════════════════════════════════════════════
    logger.info("\nPhase 7: Saving embeddings")
    logger.info("-" * 60)

    embeddings: Dict[str, np.ndarray] = {
        gene_id: model.wv[gene_id].astype(np.float32)
        for gene_id in model.wv.index_to_key
    }

    emb_path = output_dir / 'family_embeddings.pkl'
    with open(emb_path, 'wb') as f:
        pickle.dump(embeddings, f, protocol=4)
    logger.info(f"Embeddings saved: {emb_path}  ({len(embeddings):,} genes × {vector_size}d)")

    # ════════════════════════════════════════════════════════════════════════
    # Phase 8: Sanity check + manifest
    # ════════════════════════════════════════════════════════════════════════
    logger.info("\nPhase 8: Sanity check")
    logger.info("-" * 60)

    sanity_check(embeddings, ens_to_sym, t4_genes)

    n_t4_embedded = sum(1 for g in t4_genes if g in embeddings)

    write_manifest(
        output_dir=output_dir,
        config=config,
        input_paths={
            'rat_token_mapping':       stage3_mapping_path,
            'rat_to_human_mapping':    stage3_pkl_path,
            'mouse_human_orthologs':   mouse_orth_path,
            'hgnc_gene_families':      hgnc_resolved,
        },
        n_human_families=len(human_families),
        n_mouse_families=len(mouse_families),
        n_rat_families=len(rat_families),
        n_gene_pairs=len(gene_pairs),
        n_embeddings=len(embeddings),
        n_t4_genes=len(t4_genes),
        n_t4_embedded=n_t4_embedded,
        vector_size=vector_size,
        elapsed=time.time() - t_start,
        dry_run=args.dry_run,
    )

    # ── Final summary ────────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    logger.info("=" * 60)
    logger.info("STAGE 6 STEP 2 COMPLETE — Gene Family Embedding")
    logger.info("=" * 60)
    logger.info(f"  Human families:    {len(human_families):,}")
    logger.info(f"  Mouse families:    {len(mouse_families):,}")
    logger.info(f"  Rat families:      {len(rat_families):,}")
    logger.info(f"  Gene pairs:        {len(gene_pairs):,}")
    logger.info(f"  Genes embedded:    {len(embeddings):,}")
    logger.info(f"  T4 genes embedded: {n_t4_embedded:,} / {len(t4_genes):,}")
    logger.info(f"  Elapsed:           {elapsed:.1f}s")
    logger.info(f"\n  Output directory:  {output_dir}")
    logger.info(f"  Primary output:    {emb_path}")
    logger.info("")
    logger.info("Next step:")
    logger.info("  python pipeline/06_prior_knowledge/build_coexp_embedding.py")


if __name__ == '__main__':
    main()