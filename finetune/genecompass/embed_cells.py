#!/usr/bin/env python
# coding: utf-8
"""
embed_cells.py — Extract cell embeddings from fine-tuned rat GeneCompass.

Used for Stage 7 validation (before proceeding to Aim 2):
  1. Cell type clustering — held-out rat cells cluster by biology, not batch
  2. Homolog similarity — rat T1-T3 embeddings closer to orthologs than random
  3. T4 token quality — rat Cyp450s cluster near human CYP genes

The cell embedding is the CLS token output (position 0) from the final
transformer layer, which aggregates information across the entire gene
expression profile.

Usage:
  python embed_cells.py \
    --model-dir data/models/rat_genecompass_finetuned/models/rat_finetune/models \
    --dataset data/training/tokenized_corpus/dataset \
    --output data/models/rat_genecompass_finetuned/embeddings \
    --n-cells 10000 \
    --species 2

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
from torch.utils.data import DataLoader

_PROJECT_ROOT = Path(os.environ.get('PIPELINE_ROOT', '.')).resolve()
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / 'vendor' / 'GeneCompass'))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
)
logger = logging.getLogger(__name__)


def extract_embeddings(
    model_dir: str,
    dataset_path: str,
    output_dir: str,
    n_cells: int = 10000,
    batch_size: int = 32,
    species: int = 2,
    device: str = 'cuda',
):
    """
    Extract CLS embeddings from fine-tuned model.

    Parameters
    ----------
    model_dir : str
        Directory containing pytorch_model.bin and config.json
    dataset_path : str
        Path to HuggingFace dataset (Arrow files)
    output_dir : str
        Output directory for embeddings (.npy) and metadata
    n_cells : int
        Number of cells to embed (randomly sampled if < total)
    batch_size : int
        Inference batch size
    species : int
        Species ID for CLS token (0=human, 1=mouse, 2=rat)
    device : str
        Device for inference
    """
    from datasets import load_from_disk
    from transformers import BertConfig
    from genecompass import BertForMaskedLM
    from torch import nn

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load config and model ────────────────────────────────────────────
    config_path = Path(model_dir) / 'config.json'
    with open(config_path) as f:
        config_dict = json.load(f)

    # Load token dictionary for knowledges
    # (we need to reconstruct the model with knowledges to load state dict)
    token_dict_path = _PROJECT_ROOT / 'data' / 'training' / 'ortholog_mappings' / 'rat_tokens.pickle'
    with open(token_dict_path, 'rb') as f:
        token_dictionary = pickle.load(f)

    # Load state dict
    state_dict_path = Path(model_dir) / 'pytorch_model.bin'
    logger.info(f"Loading model from {state_dict_path}")
    state = torch.load(state_dict_path, map_location='cpu', weights_only=False)

    # Build knowledges from state dict
    knowledges = {
        'promoter': state['bert.embeddings.promoter_knowledge'],
        'co_exp': state['bert.embeddings.co_exp_knowledge'],
        'gene_family': state['bert.embeddings.gene_family_knowledge'],
        'peca_grn': state['bert.embeddings.peca_grn_knowledge'],
    }

    homolog_path = _PROJECT_ROOT / 'vendor' / 'GeneCompass' / 'prior_knowledge' / 'homologous_hm_token.pickle'
    with open(homolog_path, 'rb') as f:
        raw = pickle.load(f)
        knowledges['homologous_gene_human2mouse'] = {v: k for k, v in raw.items()}

    # Create model
    config_dict['warmup_steps'] = 0
    config_dict['emb_warmup_steps'] = 1
    model_config = BertConfig(**config_dict)
    model = BertForMaskedLM(model_config, knowledges=knowledges)

    # Patch cls_embedding for 3 species
    num_species = 3
    model.bert.cls_embedding = nn.Embedding(num_species, config_dict['hidden_size'])

    # Load weights
    model.load_state_dict(state, strict=False)
    model = model.eval().to(device)

    # ── Load dataset ─────────────────────────────────────────────────────
    dataset = load_from_disk(dataset_path)
    if n_cells < len(dataset):
        indices = np.random.choice(len(dataset), n_cells, replace=False)
        dataset = dataset.select(indices)
    logger.info(f"Embedding {len(dataset)} cells")

    # ── Extract embeddings ───────────────────────────────────────────────
    all_embeddings = []

    with torch.no_grad():
        for start_idx in range(0, len(dataset), batch_size):
            end_idx = min(start_idx + batch_size, len(dataset))
            batch = dataset[start_idx:end_idx]

            input_ids = torch.tensor(batch['input_ids']).to(device)
            values = torch.tensor(batch['values']).to(device)
            species_tensor = torch.full(
                (input_ids.shape[0], 1), species, dtype=torch.long
            ).to(device)

            # Build attention mask
            attention_mask = (input_ids != 0).long()

            # Forward pass — get hidden states
            outputs = model.bert(
                input_ids=input_ids,
                values=values,
                attention_mask=attention_mask,
                species=species_tensor,
                emb_warmup_alpha=1.0,
                output_hidden_states=False,
            )

            # CLS embedding is at position 0 (prepended by cls_embedding)
            cls_output = outputs[0][:, 0, :]  # [batch, 768]
            all_embeddings.append(cls_output.cpu().numpy())

            if (start_idx // batch_size) % 50 == 0:
                logger.info(f"  Processed {end_idx}/{len(dataset)} cells")

    embeddings = np.concatenate(all_embeddings, axis=0)
    logger.info(f"Embeddings shape: {embeddings.shape}")

    # ── Save ─────────────────────────────────────────────────────────────
    np.save(output_dir / 'cell_embeddings.npy', embeddings)
    logger.info(f"Saved: {output_dir / 'cell_embeddings.npy'}")

    return embeddings


def main():
    parser = argparse.ArgumentParser(description="Extract cell embeddings")
    parser.add_argument('--model-dir', required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--n-cells', type=int, default=10000)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--species', type=int, default=2)
    parser.add_argument('--device', default='cuda')

    args = parser.parse_args()
    extract_embeddings(
        model_dir=args.model_dir,
        dataset_path=args.dataset,
        output_dir=args.output,
        n_cells=args.n_cells,
        batch_size=args.batch_size,
        species=args.species,
        device=args.device,
    )


if __name__ == '__main__':
    main()