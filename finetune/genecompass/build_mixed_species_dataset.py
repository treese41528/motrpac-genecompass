#!/usr/bin/env python3
"""
build_mixed_species_dataset.py -- Interleave rat/human/mouse for Phase 2.

Produces a virtual interleaved dataset using datasets.interleave_datasets
with configurable species proportions (default: 40% rat, 45% human, 15% mouse).

The interleaving is at the sample level -- each batch naturally contains
a mix of all three species. This restores the multi-species gradient
equilibrium that GeneCompass was designed for.

Usage:
  python finetune/genecompass/build_mixed_species_dataset.py \
    --rat-dataset data/training/tokenized_corpus/dataset \
    --human-dataset vendor/GeneCompass/data/randsel_500w_human \
    --mouse-dataset vendor/GeneCompass/data/randsel_500w_mouse \
    --output data/training/mixed_species_dataset \
    --rat-prob 0.40 --human-prob 0.45 --mouse-prob 0.15

Author: Tim Reese Lab
Date: April 2026
"""

import argparse
import logging
import os
import pickle
import sys
from pathlib import Path

import numpy as np
from datasets import load_from_disk, interleave_datasets, disable_caching

disable_caching()

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(os.environ.get('PIPELINE_ROOT', '.')).resolve()


def build_mixed_dataset(
    rat_path: str,
    human_path: str,
    mouse_path: str,
    output_path: str,
    rat_prob: float = 0.40,
    human_prob: float = 0.45,
    mouse_prob: float = 0.15,
    seed: int = 42,
):
    """
    Build interleaved multi-species dataset for mixed-species Phase 2.

    Each dataset must have columns: input_ids, values, length
    Species ID is added as a column (0=human, 1=mouse, 2=rat).
    """
    assert abs(rat_prob + human_prob + mouse_prob - 1.0) < 1e-6, \
        f"Probabilities must sum to 1.0, got {rat_prob + human_prob + mouse_prob}"

    # Load datasets
    logger.info(f"Loading rat dataset: {rat_path}")
    rat_ds = load_from_disk(rat_path)
    logger.info(f"  Rat: {len(rat_ds):,} cells")

    logger.info(f"Loading human dataset: {human_path}")
    human_ds = load_from_disk(human_path)
    logger.info(f"  Human: {len(human_ds):,} cells")

    logger.info(f"Loading mouse dataset: {mouse_path}")
    mouse_ds = load_from_disk(mouse_path)
    logger.info(f"  Mouse: {len(mouse_ds):,} cells")

    # Align columns -- keep only the shared set
    shared_cols = ['input_ids', 'values', 'length', 'species']
    for col in rat_ds.column_names:
        if col not in shared_cols:
            rat_ds = rat_ds.remove_columns(col)
            logger.info(f"  Dropped rat column: {col}")



    # Add species column if not present
    def add_species(dataset, species_id):
        if 'species' not in dataset.column_names:
            dataset = dataset.map(
                lambda x: {'species': species_id},
                desc=f"Adding species={species_id}",
            )
        return dataset

    human_ds = add_species(human_ds, 0)
    mouse_ds = add_species(mouse_ds, 1)
    rat_ds = add_species(rat_ds, 2)

    # Interleave with probabilities
    # Order: human, mouse, rat (matches species IDs 0, 1, 2)
    logger.info(f"Interleaving: human={human_prob}, mouse={mouse_prob}, rat={rat_prob}")
    mixed = interleave_datasets(
        [human_ds, mouse_ds, rat_ds],
        probabilities=[human_prob, mouse_prob, rat_prob],
        seed=seed,
        stopping_strategy="all_exhausted",
    )

    logger.info(f"Mixed dataset: {len(mixed):,} total samples")

    # Save
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    mixed.save_to_disk(str(output_path))
    logger.info(f"Saved: {output_path}")

    # Build sorted_length.pickle for group_by_length
    logger.info("Building sorted_length.pickle...")
    lengths = mixed['length'] if 'length' in mixed.column_names else [2048] * len(mixed)
    sorted_indices = np.argsort(lengths).tolist()
    length_path = output_path / 'sorted_length.pickle'
    with open(length_path, 'wb') as f:
        pickle.dump(sorted_indices, f)
    logger.info(f"Saved: {length_path}")

    # Summary stats
    species_counts = {}
    sample = mixed.select(range(min(100000, len(mixed))))
    for sp in sample['species']:
        key = sp[0] if isinstance(sp, list) else sp
        species_counts[key] = species_counts.get(key, 0) + 1
    total = sum(species_counts.values())
    for sp, count in sorted(species_counts.items()):
        name = {0: 'human', 1: 'mouse', 2: 'rat'}[sp]
        logger.info(f"  {name}: {count:,} ({count/total*100:.1f}%) in first 100K")

    return mixed


def main():
    parser = argparse.ArgumentParser(
        description="Build interleaved multi-species dataset for Phase 2"
    )
    parser.add_argument('--rat-dataset', required=True)
    parser.add_argument('--human-dataset', required=True)
    parser.add_argument('--mouse-dataset', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--rat-prob', type=float, default=0.40)
    parser.add_argument('--human-prob', type=float, default=0.45)
    parser.add_argument('--mouse-prob', type=float, default=0.15)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()
    build_mixed_dataset(
        rat_path=args.rat_dataset,
        human_path=args.human_dataset,
        mouse_path=args.mouse_dataset,
        output_path=args.output,
        rat_prob=args.rat_prob,
        human_prob=args.human_prob,
        mouse_prob=args.mouse_prob,
        seed=args.seed,
    )


if __name__ == '__main__':
    main()