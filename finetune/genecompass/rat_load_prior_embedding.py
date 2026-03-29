#!/usr/bin/env python
# coding: utf-8
"""
rat_load_prior_embedding.py — Build prior knowledge tensors for rat fine-tuning.

Replaces vendor/GeneCompass/genecompass/utils.py::load_prior_embedding, which
crashes on ENSRNOG tokens (T4 rat-specific genes). This version:

  1. Starts with ZERO tensors of shape [vocab_size, 768]
  2. Copies rows 0–50557 from the pre-trained GeneCompass checkpoint
     (preserving calibrated T1-T3 embeddings for all four knowledge types)
  3. Fills rows 50558–53031 (T4 rat tokens) from Stage 6 embedding pickles
  4. Applies co-expression norm rescaling (factor ~3.9×) to T4 rows only
  5. Returns the homologous_gene_human2mouse dict unchanged (rat uses identity)

The key insight: T1-T3 rows in our Stage 6 dicts are NOT used. The pre-trained
values at positions 0-50557 are already calibrated to the projection layers
(linear1) — overwriting them with our independently-computed embeddings would
break the learned projections.

Author: Tim Reese Lab
Date: March 2026
"""

import logging
import os
import pickle
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

ORIGINAL_VOCAB_SIZE = 50_558   # GeneCompass Base vocabulary
RAT_VOCAB_SIZE = 53_032        # 50558 + 2474 T4 rat tokens
T4_START = ORIGINAL_VOCAB_SIZE # First T4 token position
EMBEDDING_DIM = 768

EMBEDDING_TYPES = ('promoter', 'coexp', 'family', 'grn')


def load_rat_prior_embeddings(
    pretrained_state_dict: Dict[str, torch.Tensor],
    rat_token_dict: Dict[str, int],
    stage6_dir: str | Path,
    homologous_gene_path: str | Path,
    coexp_rescale_factor: float = 3.9,
    vocab_size: int = RAT_VOCAB_SIZE,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    """
    Build the five knowledge objects required by KnowledgeBertEmbeddings.

    Parameters
    ----------
    pretrained_state_dict : dict
        State dict from GeneCompass_Base checkpoint (torch.load output).
    rat_token_dict : dict
        Rat token dictionary {gene_id_str: token_int}. 53,032 entries.
    stage6_dir : path
        Directory containing Stage 6 embedding pickles:
          promoter_embeddings.pkl, coexp_embeddings.pkl,
          family_embeddings.pkl, grn_embeddings.pkl
    homologous_gene_path : path
        Path to GeneCompass homologous_hm_token.pickle.
    coexp_rescale_factor : float
        Scale factor for T4 co-expression embeddings (default 3.9).
        Our T4 co-exp norms are ~4.8 vs model expectation ~18.6.
    vocab_size : int
        Target vocabulary size (default 53032).

    Returns
    -------
    tuple of (promoter, coexp, family, grn, homologous_dict)
        Four [vocab_size, 768] float tensors + homolog mapping dict.
    """
    stage6_dir = Path(stage6_dir)

    # ── 1. Load Stage 6 embedding pickles ────────────────────────────────
    stage6_files = {
        'promoter': stage6_dir / 'promoter_embeddings.pkl',
        'coexp':    stage6_dir / 'coexp_embeddings.pkl',
        'family':   stage6_dir / 'family_embeddings.pkl',
        'grn':      stage6_dir / 'grn_embeddings.pkl',
    }

    stage6_dicts = {}
    for name, path in stage6_files.items():
        with open(path, 'rb') as f:
            stage6_dicts[name] = pickle.load(f)
        logger.info(f"Loaded {name} embeddings: {len(stage6_dicts[name])} entries from {path}")

    # ── 2. Map pre-trained state dict keys to knowledge types ────────────
    pretrained_keys = {
        'promoter': 'bert.embeddings.promoter_knowledge',
        'coexp':    'bert.embeddings.co_exp_knowledge',
        'family':   'bert.embeddings.gene_family_knowledge',
        'grn':      'bert.embeddings.peca_grn_knowledge',
    }

    # ── 3. Build [vocab_size, 768] tensors ───────────────────────────────
    tensors = {}
    for name in EMBEDDING_TYPES:
        # Start with zeros
        tensor = torch.zeros(vocab_size, EMBEDDING_DIM, dtype=torch.float32)

        # Copy pre-trained rows 0–50557
        pretrained = pretrained_state_dict[pretrained_keys[name]]
        assert pretrained.shape == (ORIGINAL_VOCAB_SIZE, EMBEDDING_DIM), \
            f"Expected {pretrained_keys[name]} shape [{ORIGINAL_VOCAB_SIZE}, {EMBEDDING_DIM}], " \
            f"got {pretrained.shape}"
        tensor[:ORIGINAL_VOCAB_SIZE] = pretrained

        # Fill T4 rows (50558–53031) from Stage 6
        stage6 = stage6_dicts[name]
        t4_filled = 0
        t4_zero = 0

        for gene_str, token_id in rat_token_dict.items():
            if token_id < T4_START:
                continue  # T1-T3: keep pre-trained values
            if gene_str in ('<pad>', '<mask>'):
                continue  # Special tokens: stay zero

            if gene_str in stage6:
                emb = stage6[gene_str]
                if isinstance(emb, np.ndarray):
                    emb = torch.from_numpy(emb).float()
                tensor[token_id] = emb
                t4_filled += 1
            else:
                t4_zero += 1  # Zero-vector fallback

        # Apply co-expression rescaling to T4 rows only
        if name == 'coexp' and coexp_rescale_factor != 1.0:
            tensor[T4_START:] *= coexp_rescale_factor
            logger.info(
                f"Co-expression T4 rows rescaled by {coexp_rescale_factor:.1f}× "
                f"(norm ratio correction)"
            )

        tensors[name] = tensor
        total_t4 = t4_filled + t4_zero
        coverage = (t4_filled / total_t4 * 100) if total_t4 > 0 else 0
        logger.info(
            f"{name}: T4 coverage {t4_filled}/{total_t4} ({coverage:.1f}%), "
            f"{t4_zero} zero-vector fallback"
        )

    # ── 4. Load homologous gene dict (unchanged from GeneCompass) ────────
    with open(homologous_gene_path, 'rb') as f:
        # Original file maps mouse→human. GeneCompass reverses it to human→mouse
        # for the homologous_index buffer. We preserve that convention.
        raw = pickle.load(f)
        homologous_gene_human2mouse = {v: k for k, v in raw.items()}

    logger.info(
        f"Homologous gene dict: {len(homologous_gene_human2mouse)} entries "
        f"(applied only to species==1 mouse cells)"
    )

    return (
        tensors['promoter'],
        tensors['coexp'],
        tensors['family'],
        tensors['grn'],
        homologous_gene_human2mouse,
    )


def build_knowledges_dict(
    pretrained_state_dict: Dict[str, torch.Tensor],
    rat_token_dict: Dict[str, int],
    stage6_dir: str | Path,
    homologous_gene_path: str | Path,
    coexp_rescale_factor: float = 3.9,
    vocab_size: int = RAT_VOCAB_SIZE,
) -> Dict[str, object]:
    """
    Convenience wrapper: returns the dict format expected by BertForMaskedLM.

    Returns
    -------
    dict with keys: 'promoter', 'co_exp', 'gene_family', 'peca_grn',
                    'homologous_gene_human2mouse'
    """
    promoter, coexp, family, grn, homolog = load_rat_prior_embeddings(
        pretrained_state_dict=pretrained_state_dict,
        rat_token_dict=rat_token_dict,
        stage6_dir=stage6_dir,
        homologous_gene_path=homologous_gene_path,
        coexp_rescale_factor=coexp_rescale_factor,
        vocab_size=vocab_size,
    )

    return {
        'promoter': promoter,
        'co_exp': coexp,
        'gene_family': family,
        'peca_grn': grn,
        'homologous_gene_human2mouse': homolog,
    }