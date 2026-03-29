#!/usr/bin/env python
# coding: utf-8
"""
rat_model_init.py — Extend GeneCompass_Base checkpoint for rat fine-tuning.

Loads the pre-trained GeneCompass_Base checkpoint (50,558 vocab, 2 species) and
produces an extended checkpoint (53,032 vocab, 3 species) suitable for rat
fine-tuning. The output is a complete state_dict + config.json that can be
loaded directly by BertForMaskedLM.

Operations performed:
  1. Resize all vocab-indexed tensors from [50558, ...] → [53032, ...]
  2. Fill T4 prior knowledge rows from Stage 6 embeddings
  3. Initialize T4 word embeddings (random or family-mean)
  4. Extend cls_embedding from [2, 768] → [3, 768] (rat = mouse + noise)
  5. Extend prediction head biases to [53032]
  6. Extend homologous_index to [53032] (identity for new positions)
  7. Maintain weight tying: decoder.weight == word_embeddings.weight
  8. Write extended config.json with vocab_size=53032

The original GeneCompass checkpoint is never modified. Output goes to a new
directory (default: data/models/rat_genecompass_init/).

Author: Tim Reese Lab
Date: March 2026
"""

import argparse
import json
import logging
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

# Resolve project root for imports
_PROJECT_ROOT = Path(os.environ.get('PIPELINE_ROOT', '.')).resolve()
sys.path.insert(0, str(_PROJECT_ROOT))

# Import from same directory (pipeline/07_finetuning/)
_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))

from rat_load_prior_embedding import (
    build_knowledges_dict,
    ORIGINAL_VOCAB_SIZE,
    RAT_VOCAB_SIZE,
    T4_START,
    EMBEDDING_DIM,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# T4 Word Embedding Initialization Strategies
# ─────────────────────────────────────────────────────────────────────────────

def init_t4_random(
    word_emb: torch.Tensor,
    rat_token_dict: dict,
    std: float = 0.02,
) -> None:
    """Initialize T4 word embeddings with N(0, std). Matches config.initializer_range."""
    n_t4 = RAT_VOCAB_SIZE - ORIGINAL_VOCAB_SIZE
    word_emb[T4_START:] = torch.randn(n_t4, EMBEDDING_DIM) * std
    logger.info(f"T4 word embeddings initialized: N(0, {std}) for {n_t4} tokens")


def init_t4_family_mean(
    word_emb: torch.Tensor,
    rat_token_dict: dict,
    family_dict: dict,
    std: float = 0.02,
) -> None:
    """
    Initialize T4 word embeddings from family-neighbor means.

    For each T4 gene that belongs to a known gene family, average the
    pre-trained word embeddings of its T1-T3 family members. Falls back
    to N(0, std) for T4 genes with no family members.

    Parameters
    ----------
    word_emb : Tensor [vocab_size, 768]
        Word embedding tensor (modified in-place).
    rat_token_dict : dict
        {gene_str: token_id} for all 53,032 tokens.
    family_dict : dict
        {family_name: [gene_str, ...]} mapping gene families to members.
    std : float
        Fallback noise std for genes without family members.
    """
    # Build reverse lookup: gene_str → family_name
    gene_to_family = {}
    for fam, members in family_dict.items():
        for gene in members:
            gene_to_family[gene] = fam

    n_family = 0
    n_fallback = 0

    for gene_str, token_id in rat_token_dict.items():
        if token_id < T4_START:
            continue
        if gene_str in ('<pad>', '<mask>'):
            continue

        fam = gene_to_family.get(gene_str)
        if fam is not None:
            # Collect T1-T3 family members with pre-trained embeddings
            member_ids = []
            for member in family_dict[fam]:
                mid = rat_token_dict.get(member)
                if mid is not None and mid < T4_START:
                    member_ids.append(mid)

            if member_ids:
                word_emb[token_id] = word_emb[member_ids].mean(dim=0)
                n_family += 1
                continue

        # Fallback: random
        word_emb[token_id] = torch.randn(EMBEDDING_DIM) * std
        n_fallback += 1

    logger.info(
        f"T4 word embeddings: {n_family} from family mean, "
        f"{n_fallback} random fallback"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main checkpoint extension
# ─────────────────────────────────────────────────────────────────────────────

def extend_checkpoint(
    checkpoint_path: str | Path,
    rat_token_dict_path: str | Path,
    stage6_dir: str | Path,
    homologous_gene_path: str | Path,
    output_dir: str | Path,
    t4_init_strategy: str = 'random',
    t4_init_std: float = 0.02,
    species_noise_std: float = 0.02,
    coexp_rescale_factor: float = 3.9,
    family_dict_path: str | Path | None = None,
) -> Path:
    """
    Extend GeneCompass_Base checkpoint for rat fine-tuning.

    Returns the output directory path.
    """
    checkpoint_path = Path(checkpoint_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load pre-trained state dict ───────────────────────────────────
    model_bin = checkpoint_path / 'pytorch_model.bin'
    logger.info(f"Loading pre-trained checkpoint: {model_bin}")
    state = torch.load(model_bin, map_location='cpu', weights_only=False)
    logger.info(f"Loaded {len(state)} tensors")

    # ── 2. Load rat token dictionary ─────────────────────────────────────
    with open(rat_token_dict_path, 'rb') as f:
        rat_token_dict = pickle.load(f)
    logger.info(f"Rat token dict: {len(rat_token_dict)} entries")
    assert len(rat_token_dict) == RAT_VOCAB_SIZE, \
        f"Expected {RAT_VOCAB_SIZE} tokens, got {len(rat_token_dict)}"

    # ── 3. Build prior knowledge tensors ─────────────────────────────────
    knowledges = build_knowledges_dict(
        pretrained_state_dict=state,
        rat_token_dict=rat_token_dict,
        stage6_dir=stage6_dir,
        homologous_gene_path=homologous_gene_path,
        coexp_rescale_factor=coexp_rescale_factor,
        vocab_size=RAT_VOCAB_SIZE,
    )

    # ── 4. Resize vocab-indexed tensors ──────────────────────────────────
    # Prior knowledge buffers → replace entirely with our tensors
    knowledge_mapping = {
        'bert.embeddings.promoter_knowledge':     knowledges['promoter'],
        'bert.embeddings.co_exp_knowledge':       knowledges['co_exp'],
        'bert.embeddings.gene_family_knowledge':  knowledges['gene_family'],
        'bert.embeddings.peca_grn_knowledge':     knowledges['peca_grn'],
    }
    for key, tensor in knowledge_mapping.items():
        state[key] = tensor
        logger.info(f"  {key}: {tensor.shape}")

    # Word embeddings: [50558, 768] → [53032, 768]
    old_word_emb = state['bert.embeddings.word_embeddings.weight']
    new_word_emb = torch.zeros(RAT_VOCAB_SIZE, EMBEDDING_DIM, dtype=old_word_emb.dtype)
    new_word_emb[:ORIGINAL_VOCAB_SIZE] = old_word_emb

    # Initialize T4 word embeddings
    if t4_init_strategy == 'random':
        init_t4_random(new_word_emb, rat_token_dict, std=t4_init_std)
    elif t4_init_strategy == 'family_mean':
        if family_dict_path is None:
            raise ValueError("family_dict_path required for family_mean init")
        with open(family_dict_path, 'rb') as f:
            family_dict = pickle.load(f)
        init_t4_family_mean(new_word_emb, rat_token_dict, family_dict, std=t4_init_std)
    else:
        raise ValueError(f"Unknown t4_init_strategy: {t4_init_strategy}")

    state['bert.embeddings.word_embeddings.weight'] = new_word_emb
    logger.info(f"  word_embeddings: {new_word_emb.shape}")

    # ── 5. Weight tying: decoder.weight == word_embeddings.weight ────────
    # CRITICAL: These must be the SAME tensor object at model load time.
    # When saving state_dict they become separate copies, but BertForMaskedLM
    # re-ties them via _tie_or_clone_weights after loading. We just need the
    # shapes to match.
    state['cls.predictions.decoder.weight'] = new_word_emb.clone()
    logger.info("  Weight tying: decoder.weight set to word_embeddings copy")

    # ── 6. Extend prediction head biases ─────────────────────────────────
    for bias_key in ('cls.predictions.bias', 'cls.predictions.decoder.bias'):
        old_bias = state[bias_key]
        new_bias = torch.zeros(RAT_VOCAB_SIZE, dtype=old_bias.dtype)
        new_bias[:ORIGINAL_VOCAB_SIZE] = old_bias
        state[bias_key] = new_bias
        logger.info(f"  {bias_key}: [{ORIGINAL_VOCAB_SIZE}] → [{RAT_VOCAB_SIZE}]")

    # ── 7. Extend homologous_index ───────────────────────────────────────
    old_index = state['bert.embeddings.homologous_index']
    new_index = torch.arange(RAT_VOCAB_SIZE, dtype=old_index.dtype)
    new_index[:ORIGINAL_VOCAB_SIZE] = old_index
    # T4 positions: identity (new tokens map to themselves)
    state['bert.embeddings.homologous_index'] = new_index
    logger.info(f"  homologous_index: [{ORIGINAL_VOCAB_SIZE}] → [{RAT_VOCAB_SIZE}] (T4 = identity)")

    # ── 8. Extend cls_embedding: [2, 768] → [3, 768] ────────────────────
    old_cls = state['bert.cls_embedding.weight']
    assert old_cls.shape == (2, EMBEDDING_DIM), f"Expected cls_embedding [2, 768], got {old_cls.shape}"
    new_cls = torch.zeros(3, EMBEDDING_DIM, dtype=old_cls.dtype)
    new_cls[:2] = old_cls
    # Rat (species=2) initialized from mouse (species=1) + noise
    new_cls[2] = old_cls[1] + torch.randn(EMBEDDING_DIM) * species_noise_std
    state['bert.cls_embedding.weight'] = new_cls
    logger.info(
        f"  cls_embedding: [2, 768] → [3, 768]  "
        f"(rat init: mouse + N(0, {species_noise_std}))"
    )

    # ── 9. Verify no tensors were missed ─────────────────────────────────
    for key, tensor in state.items():
        if isinstance(tensor, torch.Tensor) and tensor.dim() >= 1:
            if tensor.shape[0] == ORIGINAL_VOCAB_SIZE:
                logger.warning(f"UNREIZED tensor detected: {key} shape {tensor.shape}")

    # ── 10. Save extended checkpoint ─────────────────────────────────────
    output_model_bin = output_dir / 'pytorch_model.bin'
    torch.save(state, output_model_bin)
    logger.info(f"Extended checkpoint saved: {output_model_bin}")

    # ── 11. Write updated config.json ────────────────────────────────────
    config_src = checkpoint_path / 'config.json'
    if config_src.exists():
        with open(config_src) as f:
            config = json.load(f)
    else:
        # Reconstruct from known parameters
        config = {
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "initializer_range": 0.02,
            "layer_norm_eps": 1e-12,
            "attention_probs_dropout_prob": 0.02,
            "hidden_dropout_prob": 0.02,
            "intermediate_size": 3072,
            "hidden_act": "gelu",
            "max_position_embeddings": 2048,
            "model_type": "bert",
            "num_attention_heads": 12,
            "type_vocab_size": 2,
            "use_values": True,
            "use_promoter": True,
            "use_co_exp": True,
            "use_gene_family": True,
            "use_peca_grn": True,
            "use_cls_token": True,
        }

    # Update vocab size and pad token
    config['vocab_size'] = RAT_VOCAB_SIZE
    config['pad_token_id'] = rat_token_dict.get('<pad>', 0)

    config_out = output_dir / 'config.json'
    with open(config_out, 'w') as f:
        json.dump(config, f, indent=2)
    logger.info(f"Config saved: {config_out} (vocab_size={RAT_VOCAB_SIZE})")

    # ── 12. Summary ──────────────────────────────────────────────────────
    total_params = sum(t.numel() for t in state.values() if isinstance(t, torch.Tensor))
    logger.info(f"Extended model: {total_params:,} parameters")
    logger.info(f"Growth: {total_params - 326_847_102:+,} from base ({(total_params/326_847_102 - 1)*100:+.1f}%)")

    return output_dir


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Extend GeneCompass_Base checkpoint for rat fine-tuning"
    )
    parser.add_argument(
        '--checkpoint', required=True,
        help='Path to GeneCompass_Base checkpoint directory'
    )
    parser.add_argument(
        '--rat-token-dict', required=True,
        help='Path to rat_tokens.pickle (53,032 entries)'
    )
    parser.add_argument(
        '--stage6-dir', required=True,
        help='Directory containing Stage 6 embedding pickles'
    )
    parser.add_argument(
        '--homologous-gene-path', required=True,
        help='Path to homologous_hm_token.pickle'
    )
    parser.add_argument(
        '--output-dir', required=True,
        help='Output directory for extended checkpoint'
    )
    parser.add_argument(
        '--t4-init', default='random', choices=['random', 'family_mean'],
        help='T4 word embedding initialization strategy'
    )
    parser.add_argument(
        '--t4-std', type=float, default=0.02,
        help='Std for T4 word embedding random init'
    )
    parser.add_argument(
        '--species-noise-std', type=float, default=0.02,
        help='Noise std for rat cls_embedding init from mouse'
    )
    parser.add_argument(
        '--coexp-rescale', type=float, default=3.9,
        help='Rescale factor for T4 co-expression embeddings'
    )
    parser.add_argument(
        '--family-dict', default=None,
        help='Path to gene family dict pickle (required for family_mean init)'
    )

    args = parser.parse_args()

    extend_checkpoint(
        checkpoint_path=args.checkpoint,
        rat_token_dict_path=args.rat_token_dict,
        stage6_dir=args.stage6_dir,
        homologous_gene_path=args.homologous_gene_path,
        output_dir=args.output_dir,
        t4_init_strategy=args.t4_init,
        t4_init_std=args.t4_std,
        species_noise_std=args.species_noise_std,
        coexp_rescale_factor=args.coexp_rescale,
        family_dict_path=args.family_dict,
    )


if __name__ == '__main__':
    main()