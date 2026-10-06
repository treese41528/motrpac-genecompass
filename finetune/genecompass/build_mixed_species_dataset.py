#!/usr/bin/env python3
"""
build_mixed_species_dataset.py -- Interleave rat/human/mouse for Phase 2.

Produces a virtual interleaved dataset using datasets.interleave_datasets
with configurable species proportions (default: 40% rat, 45% human, 15% mouse).

The interleaving is at the sample level -- each batch naturally contains
a mix of all three species. This restores the multi-species gradient
equilibrium that GeneCompass was designed for.

Every row keeps references back to its source (2026-10-06): `source` (which dataset) and
`source_row` (its row there); rat rows also keep study_id, sample_id, gsm, cell_id and
cell_index from Stage 5 (blank / -1 on human and mouse rows). These are not model inputs;
rat_pretrain.py drops them before training. sorted_length.pickle holds each row's length in
dataset order, which is what GeneCompass's LengthGroupedSampler expects (the April build
wrote [0] for every row, so length grouping never happened).

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
from datasets import load_from_disk, interleave_datasets, disable_caching, Value

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
    exclude_studies: tuple = (),
):
    """
    Build interleaved multi-species dataset for mixed-species Phase 2.

    Each dataset must have columns: input_ids, values, length
    Species ID is added as a column (0=human, 1=mouse, 2=rat).
    """
    assert abs(rat_prob + human_prob + mouse_prob - 1.0) < 1e-6, \
        f"Probabilities must sum to 1.0, got {rat_prob + human_prob + mouse_prob}"
    out = Path(output_path)
    if out.exists() and any((out / f).exists() for f in ('state.json', 'dataset_info.json')):
        raise FileExistsError(f"{out} already holds a dataset; write the new mix to a new directory "
                              f"(the current model was trained on the existing one)")

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

    # Held-out rat studies never enter the mix
    if exclude_studies:
        excl = set(exclude_studies)
        n0 = len(rat_ds)
        rat_ds = rat_ds.filter(lambda sid: [s not in excl for s in sid], batched=True,
                               input_columns=['study_id'])
        logger.info(f"  Excluded studies {sorted(excl)}: {n0 - len(rat_ds):,} rat cells removed")

    # Model inputs + source references. Rat keeps its Stage 5 source columns; human and mouse
    # get the same columns, blank, so the three schemas match for interleaving.
    model_cols = ['input_ids', 'values', 'length', 'species']
    ref_str = ['study_id', 'sample_id', 'gsm', 'cell_id']
    for col in rat_ds.column_names:
        if col not in model_cols + ref_str + ['cell_index']:
            rat_ds = rat_ds.remove_columns(col)
            logger.info(f"  Dropped rat column: {col}")
    missing = [c for c in ref_str + ['cell_index'] if c not in rat_ds.column_names]
    if missing:
        raise ValueError(f"Rat dataset lacks source columns {missing}: rebuild it with the current Stage 5")

    def add_refs(dataset, name, is_rat):
        n = len(dataset)
        dataset = dataset.add_column('source', [name] * n)
        dataset = dataset.add_column('source_row', np.arange(n, dtype=np.int64))
        if not is_rat:
            for c in ref_str:
                dataset = dataset.add_column(c, [''] * n)
            dataset = dataset.add_column('cell_index', np.full(n, -1, dtype=np.int32))
        return dataset.cast_column('cell_index', Value('int32'))



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

    human_ds = add_refs(human_ds, Path(human_path).name, is_rat=False)
    mouse_ds = add_refs(mouse_ds, Path(mouse_path).name, is_rat=False)
    rat_ds = add_refs(rat_ds, Path(rat_path).parent.name + '/' + Path(rat_path).name, is_rat=True)
    col_order = model_cols + ['source', 'source_row'] + ref_str + ['cell_index']
    human_ds, mouse_ds, rat_ds = (d.select_columns(col_order) for d in (human_ds, mouse_ds, rat_ds))

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

    # sorted_length.pickle: each row's length, in dataset order (GeneCompass's
    # LengthGroupedSampler reads it as lengths[i] for row i).
    logger.info("Building sorted_length.pickle (per-row lengths)...")
    lens = np.asarray(mixed.with_format('numpy', columns=['length'])['length']).reshape(-1)
    assert len(lens) == len(mixed), f"{len(lens)} lengths for {len(mixed)} rows"
    length_path = output_path / 'sorted_length.pickle'
    with open(length_path, 'wb') as f:
        pickle.dump(lens.astype(int).tolist(), f)
    logger.info(f"Saved: {length_path} (min {lens.min()}, median {int(np.median(lens))}, max {lens.max()})")

    # Summary over the whole mix: rows per species, distinct source cells per species
    num = mixed.with_format('numpy', columns=['species', 'source_row'])
    sp = np.asarray(num['species']).reshape(-1); src_row = np.asarray(num['source_row']).reshape(-1)
    counts = {}
    for sid, name in ((0, 'human'), (1, 'mouse'), (2, 'rat')):
        m = sp == sid
        counts[name] = {'rows': int(m.sum()), 'distinct_source_cells': int(np.unique(src_row[m]).size)}
        logger.info(f"  {name}: {counts[name]['rows']:,} rows ({m.mean() * 100:.1f}%), "
                    f"{counts[name]['distinct_source_cells']:,} distinct source cells")
    try:
        sys.path.insert(0, str(_PROJECT_ROOT))
        from lib.provenance import write_provenance
        write_provenance(output_path,
                         inputs=[Path(x) / 'state.json' for x in (rat_path, human_path, mouse_path)],
                         params=dict(rat_prob=rat_prob, human_prob=human_prob, mouse_prob=mouse_prob,
                                     seed=seed, stopping_strategy='all_exhausted',
                                     exclude_studies=sorted(exclude_studies), n_rows=len(mixed), counts=counts))
    except ImportError:
        logger.warning("lib.provenance not importable; no provenance record written")

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
    parser.add_argument('--exclude-studies', default='',
                        help='Comma-separated rat study accessions to keep out of the mix (held-out set)')

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
        exclude_studies=tuple(x.strip() for x in args.exclude_studies.split(',') if x.strip()),
    )


if __name__ == '__main__':
    main()