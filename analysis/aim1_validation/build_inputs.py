#!/usr/bin/env python3
"""
build_inputs.py -- Aim-1 validation, input builder (CPU).

Produces everything the GPU extractor + analysis need:

V1 (held-out clustering):
  - A STRATIFIED, LABELLED rat cell subset: pick a diverse set of studies, cap cells/study
    (so no single study dominates the batch-mixing test), join consensus cell-type labels by
    (study_id, barcode). Multi-sample studies reuse barcodes, so (study,barcode)->cell_type is
    built by unioning that study's per-sample celltypes files and KEEPING ONLY concordant
    barcodes (a barcode mapping to >1 cell_type across samples is dropped + counted). The corpus
    drops sample identity (cell_id=barcode only), so this is the only unambiguous join.
  - Saves out/subset_rat/ (HF dataset, order-preserving) + out/labels.tsv (row-aligned:
    row, cell_id, study_id, cell_type).

V2 (homolog similarity) target tokens + pairs:
  - From rat_token_mapping.tsv: sample T1 (shared human token), all T3a (new token + human
    ortholog), all T3b (new token + mouse ortholog), and a random null set. Map each gene's
    rat token + its ortholog's token (via rat_human_mouse_tokens.pickle: ENSG/ENSMUSG->id).
  - Saves rat_targets.npy / human_targets.npy / mouse_targets.npy (token ids to extract
    contextual means for) + pairs.tsv (rat_gene, tier, rat_token, human_token, mouse_token, role).

(V3 needs no forward pass -> handled entirely in analyze.py from the static word embeddings.)
"""
import argparse
import glob
import json
import os
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(os.environ.get("PIPELINE_ROOT", ".")).resolve()
ANN = _ROOT / "data/training/cell_annotations"
OMAP = _ROOT / "data/training/ortholog_mappings"


def base(x):
    return str(x).strip().split(".")[0].upper()


def build_celltype_map(study_id):
    """Concordant barcode->cell_type for a study (union its sample files; drop conflicts)."""
    files = glob.glob(str(ANN / f"{study_id}_sample*/{study_id}_sample*_celltypes.tsv"))
    bc = {}
    conflict = set()
    for f in files:
        df = pd.read_csv(f, sep="\t")
        for b, ct in zip(df["barcode"], df["cell_type"]):
            if b in conflict:
                continue
            if b in bc and bc[b] != ct:
                conflict.add(b); del bc[b]
            else:
                bc[b] = ct
    return bc, len(conflict)


def build_subset(args, out):
    """V1: stratified, label-joined rat cell subset (the expensive 9.48M-cell corpus join)."""
    from collections import defaultdict
    from datasets import load_from_disk
    print("[build] loading corpus study_id/cell_id columns ...", flush=True)
    ds = load_from_disk(args.corpus)
    studies = ds["study_id"]; barcodes = ds["cell_id"]
    print(f"[build] corpus n={len(studies)}", flush=True)
    by_study = defaultdict(list)
    for i, s in enumerate(studies):
        by_study[s].append(i)
    annotated = [s for s in by_study if glob.glob(str(ANN / f"{s}_sample*"))]
    print(f"[build] {len(by_study)} studies, {len(annotated)} annotated", flush=True)

    sel_rows, sel_cid, sel_study, sel_ct = [], [], [], []
    dropped_conflict = 0
    random.shuffle(annotated)
    for s in annotated:
        if len(sel_rows) >= args.target_cells:
            break
        ctmap, nconf = build_celltype_map(s)
        dropped_conflict += nconf
        rows = by_study[s]
        if len(rows) > args.per_study:
            rows = list(np.random.choice(rows, args.per_study, replace=False))
        for i in rows:
            ct = ctmap.get(barcodes[i])
            if ct is not None and str(ct).lower() not in ("nan", "unknown", ""):
                sel_rows.append(i); sel_cid.append(barcodes[i]); sel_study.append(s); sel_ct.append(ct)

    lab = pd.DataFrame({"row_pre": sel_rows, "cell_id": sel_cid, "study_id": sel_study, "cell_type": sel_ct})
    vc = lab["cell_type"].value_counts()
    lab = lab[lab["cell_type"].isin(set(vc[vc >= args.min_celltype_n].index))].reset_index(drop=True)
    lab = lab.sort_values("row_pre").reset_index(drop=True)
    print(f"[build] labelled subset: {len(lab)} cells, {lab['cell_type'].nunique()} cell types, "
          f"{lab['study_id'].nunique()} studies; dropped {dropped_conflict} conflicting barcodes", flush=True)
    ds.select(lab["row_pre"].tolist()).save_to_disk(str(out / "subset_rat"))
    lab = lab.drop(columns=["row_pre"]).reset_index(drop=True)
    lab.insert(0, "row", range(len(lab)))
    lab.to_csv(out / "labels.tsv", sep="\t", index=False)
    json.dump({"n_cells": int(len(lab)), "n_cell_types": int(lab["cell_type"].nunique()),
               "n_studies": int(lab["study_id"].nunique()), "dropped_conflict_barcodes": int(dropped_conflict),
               "cells_per_celltype": lab["cell_type"].value_counts().to_dict()},
              open(out / "subset_summary.json", "w"), indent=2)


def build_targets(args, out):
    """V2: homolog pairs + per-species target tokens (T1 shared-token + T3a/T3b new-token).
    No explicit null set -- analyze.py builds the null by permuting the ortholog pairing."""
    print("[build] building homolog target tokens ...", flush=True)
    tm = pd.read_csv(OMAP / "rat_token_mapping.tsv", sep="\t")
    with open(OMAP / "rat_human_mouse_tokens.pickle", "rb") as f:
        vocab = {base(k): int(v) for k, v in pickle.load(f).items()}   # ENSG/ENSMUSG/ENSRNOG -> token

    def tok(g):
        return vocab.get(base(g)) if pd.notna(g) else None

    rows = []
    t1 = tm[tm.tier == "T1_tri_species"].dropna(subset=["human_ortholog"])
    t1 = t1.sample(min(args.n_t1, len(t1)), random_state=args.seed)
    for _, r in t1.iterrows():
        # rat_token from the authoritative token_id column (T1 reuses the ortholog token id, NOT keyed
        # under ENSRNOG in the combined vocab); human_token resolves to the SAME id (shared by design) --
        # so the CONTEXTUAL test (rat-cell vs human-cell embedding of that token) is the meaningful one.
        rows.append(dict(rat_gene=r.rat_gene, tier="T1", rat_token=int(r.token_id),
                         human_token=tok(r.human_ortholog), mouse_token=tok(r.mouse_ortholog)))
    for _, r in tm[tm.tier == "T3a_human_multi"].dropna(subset=["human_ortholog"]).iterrows():
        rows.append(dict(rat_gene=r.rat_gene, tier="T3a", rat_token=int(r.token_id),
                         human_token=tok(r.human_ortholog), mouse_token=None))
    for _, r in tm[tm.tier == "T3b_mouse_multi"].dropna(subset=["mouse_ortholog"]).iterrows():
        rows.append(dict(rat_gene=r.rat_gene, tier="T3b", rat_token=int(r.token_id),
                         human_token=None, mouse_token=tok(r.mouse_ortholog)))

    pairs = pd.DataFrame(rows)
    pairs = pairs[pairs.rat_token.notna()].copy()
    pairs["rat_token"] = pairs.rat_token.astype(int)
    pairs.to_csv(out / "pairs.tsv", sep="\t", index=False)

    def col_tokens(c):
        return np.array(sorted({int(x) for x in pairs[c].dropna().tolist()}), dtype=np.int64)
    np.save(out / "rat_targets.npy", col_tokens("rat_token"))
    np.save(out / "human_targets.npy", col_tokens("human_token"))
    np.save(out / "mouse_targets.npy", col_tokens("mouse_token"))
    print(f"[build] pairs={len(pairs)} ({pairs.tier.value_counts().to_dict()}); "
          f"targets rat={len(col_tokens('rat_token'))} human={len(col_tokens('human_token'))} "
          f"mouse={len(col_tokens('mouse_token'))}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", default=str(_ROOT / "data/training/tokenized_corpus/dataset"))
    ap.add_argument("--out", default=str(_ROOT / "data/validation/aim1"))
    ap.add_argument("--per-study", type=int, default=2500, help="max cells/study (balance the batch-mixing test)")
    ap.add_argument("--target-cells", type=int, default=40000, help="approx labelled cells wanted for V1")
    ap.add_argument("--min-celltype-n", type=int, default=50, help="drop cell types rarer than this in the subset")
    ap.add_argument("--n-t1", type=int, default=1000, help="T1 genes sampled for the contextual homolog test")
    ap.add_argument("--targets-only", action="store_true",
                    help="only (re)build pairs+targets (cheap); reuse the existing V1 subset/labels")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    random.seed(args.seed); np.random.seed(args.seed)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    if not args.targets_only:
        build_subset(args, out)
    build_targets(args, out)
    print(f"[build] wrote -> {out}", flush=True)


if __name__ == "__main__":
    main()
