#!/usr/bin/env python
# coding: utf-8
"""
build_rat_token_dictionary.py — Merge human/mouse and T4 rat token dicts.

Produces a single 53,032-entry dictionary for Stage 7 fine-tuning:

  human_mouse_tokens.pickle (50,558 entries)
    <pad>      → 0
    <mask>     → 1
    ENSG...    → 2–23114     (23,113 human genes)
    ENSMUSG... → 23115–50557 (27,443 mouse genes)

  + T4 entries from rat_tokens.pickle (2,474 entries)
    ENSRNOG... → 50558–53031 (rat-specific genes, no human/mouse ortholog)

  = rat_human_mouse_tokens.pickle (53,032 entries)

T1-T3 rat genes are intentionally excluded — they reuse existing human/mouse
token IDs, so adding them would create duplicate values and inflate len().
The merged dict has exactly one entry per token position.

Usage:
  python finetune/genecompass/build_rat_token_dictionary.py

  # Or with explicit paths:
  python finetune/genecompass/build_rat_token_dictionary.py \
    --hm-tokens vendor/GeneCompass/prior_knowledge/human_mouse_tokens.pickle \
    --rat-tokens data/training/ortholog_mappings/rat_tokens.pickle \
    --output data/training/ortholog_mappings/rat_human_mouse_tokens.pickle

Author: Tim Reese Lab
Date: March 2026
"""

import argparse
import logging
import os
import pickle
import sys
from pathlib import Path

_PROJECT_ROOT = Path(os.environ.get('PIPELINE_ROOT', '.')).resolve()

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

ORIGINAL_VOCAB_SIZE = 50_558
T4_START = ORIGINAL_VOCAB_SIZE
EXPECTED_MERGED_SIZE = 53_032


def build_merged_dictionary(
    hm_tokens_path: str | Path,
    rat_tokens_path: str | Path,
    output_path: str | Path,
) -> dict:
    """
    Build merged token dictionary for rat fine-tuning.

    Returns the merged dict (also saved to output_path).
    """
    # Load human/mouse tokens
    with open(hm_tokens_path, 'rb') as f:
        hm_tokens = pickle.load(f)
    logger.info(f"human_mouse_tokens: {len(hm_tokens)} entries")

    assert len(hm_tokens) == ORIGINAL_VOCAB_SIZE, \
        f"Expected {ORIGINAL_VOCAB_SIZE} human/mouse tokens, got {len(hm_tokens)}"
    assert hm_tokens.get('<pad>') == 0, "<pad> must be token 0"
    assert hm_tokens.get('<mask>') == 1, "<mask> must be token 1"

    # Load rat tokens
    with open(rat_tokens_path, 'rb') as f:
        rat_tokens = pickle.load(f)
    logger.info(f"rat_tokens: {len(rat_tokens)} entries")

    # Count T4 tokens
    t4_entries = {k: v for k, v in rat_tokens.items() if v >= T4_START}
    t1t3_entries = {k: v for k, v in rat_tokens.items() if v < T4_START}
    logger.info(f"  T1-T3 (reuse existing IDs): {len(t1t3_entries)}")
    logger.info(f"  T4 (new rat-specific):      {len(t4_entries)}")

    # Verify no key collisions between T4 rat keys and human/mouse keys
    collisions = set(t4_entries.keys()) & set(hm_tokens.keys())
    assert len(collisions) == 0, \
        f"Key collision between T4 rat and human/mouse: {collisions}"

    # Verify T4 token IDs are contiguous and start at 50558
    t4_ids = sorted(t4_entries.values())
    assert t4_ids[0] == T4_START, \
        f"T4 IDs should start at {T4_START}, got {t4_ids[0]}"
    assert t4_ids[-1] == T4_START + len(t4_entries) - 1, \
        f"T4 IDs not contiguous: last={t4_ids[-1]}, expected={T4_START + len(t4_entries) - 1}"

    # Build merged dictionary
    merged = dict(hm_tokens)
    merged.update(t4_entries)

    expected = ORIGINAL_VOCAB_SIZE + len(t4_entries)
    assert len(merged) == expected, \
        f"Merged dict has {len(merged)} entries, expected {expected}"

    logger.info(f"Merged dictionary: {len(merged)} entries")
    logger.info(f"  <pad>={merged['<pad>']}, <mask>={merged['<mask>']}")
    logger.info(f"  ENSG:    {sum(1 for k in merged if isinstance(k,str) and k.startswith('ENSG') and not k.startswith('ENSMUSG'))}")
    logger.info(f"  ENSMUSG: {sum(1 for k in merged if isinstance(k,str) and k.startswith('ENSMUSG'))}")
    logger.info(f"  ENSRNOG: {sum(1 for k in merged if isinstance(k,str) and k.startswith('ENSRNOG'))}")
    logger.info(f"  max token ID: {max(merged.values())}")

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(merged, f)
    logger.info(f"Saved: {output_path}")

    return merged


def main():
    parser = argparse.ArgumentParser(
        description="Build merged token dictionary for rat fine-tuning"
    )
    parser.add_argument(
        '--hm-tokens',
        default=str(_PROJECT_ROOT / 'vendor' / 'GeneCompass' / 'prior_knowledge' / 'human_mouse_tokens.pickle'),
        help='Path to human_mouse_tokens.pickle (50,558 entries)',
    )
    parser.add_argument(
        '--rat-tokens',
        default=str(_PROJECT_ROOT / 'data' / 'training' / 'ortholog_mappings' / 'rat_tokens.pickle'),
        help='Path to rat_tokens.pickle (22,213 entries)',
    )
    parser.add_argument(
        '--output',
        default=str(_PROJECT_ROOT / 'data' / 'training' / 'ortholog_mappings' / 'rat_human_mouse_tokens.pickle'),
        help='Output path for merged dictionary (53,032 entries)',
    )

    args = parser.parse_args()
    build_merged_dictionary(args.hm_tokens, args.rat_tokens, args.output)


if __name__ == '__main__':
    main()