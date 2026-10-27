#!/usr/bin/env python3
"""
transfer_cosine.py -- Aim 3a / Module E: how far does re-expression as human move each cell?

For every tissue that has both a rat-space embedding (Stage 9) and a human-space embedding
(Stage 12) of its pseudo-cells, the cosine between the two CLS embeddings of the same pseudo-cell,
paired by cell_id. Both roots must be embedded by the same model. A cosine near 1 means the
transfer (species token 2 -> 0, rat genes without a human ortholog dropped) barely moves the cell;
compare_transfer.py asks whether the exercise axis survives that move.

Per tissue: pseudo-cells on each side, cells paired, mean / median / 5th percentile / min cosine.
Read-only; writes --out (TSV) if given, and prints the table plus the median over tissues.

Usage:
  python translation/transfer_cosine.py \
    --rat-root   data/deconvolution/genecompass_input \
    --human-root data/deconvolution/genecompass_input_human \
    --out        data/results/retrain_v2/c5/stage12_cosine_april.tsv
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_from_disk


def _load(d):
    ids = load_from_disk(str(d / "dataset"), keep_in_memory=True)["cell_id"]
    emb = np.load(d / "embeddings" / "cell_embeddings.npy").astype(np.float64)
    if len(ids) != len(emb):
        sys.exit(f"ERROR: {d}: {len(ids)} dataset rows vs {len(emb)} embeddings")
    if len(set(ids)) != len(ids):
        sys.exit(f"ERROR: {d}: duplicate cell_id")
    return ids, emb


def tissue_cosines(rat_dir, hum_dir):
    ids_r, er = _load(rat_dir)
    ids_h, eh = _load(hum_dir)
    pos = {c: i for i, c in enumerate(ids_r)}
    missing = [c for c in ids_h if c not in pos]
    if missing:
        sys.exit(f"ERROR: {hum_dir}: {len(missing)} human-space cells not in the rat dataset (e.g. {missing[0]})")
    a = er[[pos[c] for c in ids_h]]
    cos = (a * eh).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(eh, axis=1))
    return cos, len(ids_r), len(ids_h)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rat-root", required=True, type=Path)
    ap.add_argument("--human-root", required=True, type=Path)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    rows = []
    for hum_dir in sorted(p.parent.parent for p in args.human_root.glob("*/embeddings/cell_embeddings.npy")):
        rat_dir = args.rat_root / hum_dir.name
        if not (rat_dir / "embeddings" / "cell_embeddings.npy").exists():
            sys.exit(f"ERROR: {hum_dir.name} has a human-space embedding but no rat-space one under {args.rat_root}")
        cos, n_rat, n_hum = tissue_cosines(rat_dir, hum_dir)
        rows.append({"tissue": hum_dir.name, "n_rat": n_rat, "n_human": n_hum, "n_paired": len(cos),
                     "mean": cos.mean(), "median": np.median(cos), "p05": np.percentile(cos, 5), "min": cos.min()})
    if not rows:
        sys.exit(f"ERROR: no human-space embeddings under {args.human_root}")
    df = pd.DataFrame(rows)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"median over {len(df)} tissues: mean cosine {df['mean'].median():.4f}, median cosine {df['median'].median():.4f}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out, sep="\t", index=False, float_format="%.6f")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
