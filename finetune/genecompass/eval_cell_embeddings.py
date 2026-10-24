#!/usr/bin/env python3
"""eval_cell_embeddings.py — cell-embedding comparison for the retrain-v2 pre-registration (C3).

docs/RETRAIN_V2_PREREGISTRATION.md §3 C3 (implementation logged in §6, 2026-10-24). Three subcommands:

  join   reference cells (cells_meta.tsv: sample + barcode + cell_type) -> rows of both corpora, keeping
         the cells present in both; writes <out>/cells.tsv and the selected rows of each corpus, in the
         same order, as dataset chunks under --chunk-root/{v1,v2}/chunk_NN
           python finetune/genecompass/eval_cell_embeddings.py join --out data/results/retrain_v2/c3 \
               --chunk-root tmp/retrain_v2_c3 --n-chunks 8
  embed  one chunk with one model (embed_cells.extract_embeddings: final-layer CLS, species 2, fp32)
           python ... embed --model-dir <dir> --chunk <chunk dir> --out <npy>
  score  per-reference metrics for both models from <out>/cells.tsv + <out>/emb/{april,v2}_chunk_NN.npy

Joins: v2 corpus rows by (sample_id, cell_id); a reference sample that the clean corpus dropped as a
byte-identical copy (config/corpus_v2_plan.tsv) is looked up under the copy it kept. April's v1 corpus
has no sample column: each row's sample is rebuilt from the Stage 5 task manifests (tasks in order,
files in order, n_cells_written each; the assembler concatenates shards in that order) and checked
barcode-for-barcode against v2 for every sample both corpora tokenized.

Metrics per reference, on L2-normalised embeddings (cosine): cell type -- 5-fold stratified CV (seed
20261008), 10 nearest training cells, majority vote (ties to the first label in sorted order), macro-F1
and accuracy over all held-out predictions; silhouette by cell type, cosine distance, all cells; sample
mixing -- the same 5-fold kNN predicting sample_id within each cell type, over cell types with >= 2
samples of >= 5 cells (smaller samples left out), pooled accuracy (lower = less batch-driven).
"""
import argparse
import csv
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
REFS = ["BAT_GSE244451", "BLOOD_GSE285476", "CORTEX_GSE303115", "HEART_GSE280111_LV", "HIPPOC_GSE295314",
        "HYPOTH_GSE248413_Y", "KIDNEY_GSE240658", "LIVER_GSE220075", "LUNG_native_pooled",
        "MUSCLE_GSE137869_Y", "SMLINT_GSE272055", "TESTES_OMIX767_lifted", "WATSC_GSE137869_Y"]
V1 = ROOT / "data/training/tokenized_corpus"
V2 = ROOT / "data/training/tokenized_corpus_v2"
PLAN = ROOT / "config/corpus_v2_plan.tsv"
SEED, K, FOLDS, MIN_SAMPLE_CELLS = 20261008, 10, 5, 5


# ------------------------------------------------------------------------------------------- join
def corpus_keys(ds_dir, with_sample):
    from datasets import load_from_disk
    t = load_from_disk(str(ds_dir)).data
    cols = ["study_id", "cell_id"] + (["sample_id"] if with_sample else [])
    return pd.DataFrame({c: t.column(c).to_numpy(zero_copy_only=False) for c in cols})


def v1_samples(n_rows):
    names, counts, studies = [], [], []
    for p in sorted((V1 / "manifests").glob("task_*_manifest.json")):
        for f in json.loads(p.read_text())["file_results"]:
            if f.get("status") == "ok":
                names.append(f["file"].rsplit(".h5ad", 1)[0]); counts.append(f["n_cells_written"])
                studies.append(f["study_id"])
    if sum(counts) != n_rows:
        sys.exit(f"v1 manifests account for {sum(counts):,} rows, the dataset has {n_rows:,}")
    return np.repeat(names, counts), np.repeat(studies, counts)


def row_index(df, label):
    key = df["sample_id"] + "\t" + df["cell_id"]
    dup = key.duplicated(keep=False)
    print(f"{label}: {len(df):,} rows, {int(dup.sum()):,} in duplicated (sample, barcode) keys (left out)")
    return pd.Series(np.flatnonzero(~dup), index=key[~dup].to_numpy())


def join(a):
    out = Path(a.out)
    if out.exists():
        sys.exit(f"{out} exists")
    v2 = corpus_keys(V2 / "dataset", True)
    v1 = corpus_keys(V1 / "dataset", False)
    v1["sample_id"], study = v1_samples(len(v1))
    print(f"v1 rows whose manifest study equals the row's study_id: {(study == v1.study_id).mean():.6f}")
    # check the rebuilt v1 sample of every row against v2, sample by sample
    s1, s2 = v1.groupby("sample_id").cell_id.apply(list), v2.groupby("sample_id").cell_id.apply(list)
    both = s1.index.intersection(s2.index)
    same = sum(s1[s] == s2[s] for s in both)
    in2 = sum(len(set(s1[s]) & set(s2[s])) for s in both)
    print(f"samples in both corpora: {len(both)} (v1 {len(s1)}, v2 {len(s2)}); identical barcode sequence: "
          f"{same}/{len(both)}; v1 cells of shared samples found in v2's same sample: "
          f"{in2:,}/{sum(len(s1[s]) for s in both):,}")
    i1, i2 = row_index(v1, "v1"), row_index(v2, "v2")

    plan = list(csv.DictReader(open(PLAN), delimiter="\t"))
    act = {r["sample_id"]: r["action"] for r in plan}
    twin = {}
    for r in plan:                         # byte-identical copies only (not "same cells, other file format")
        t = re.match(r"byte-identical copy of (\S+)", r["reason"]) if r["action"] == "drop_duplicate" else None
        if t:
            twin[r["sample_id"]] = t.group(1)
    cells = []
    for ref in REFS:
        m = pd.read_csv(ROOT / f"data/deconvolution/references_v3/{ref}/cells_meta.tsv", sep="\t", dtype=str)
        s2name = m["sample"].map(lambda s: twin.get(s, s))
        if any(act.get(twin[s]) not in ("keep", "keep_raw_slot") for s in set(m["sample"]) if s in twin):
            sys.exit(f"{ref}: a sample's kept copy is not kept")
        r1 = (m["sample"] + "\t" + m["barcode"]).map(i1)
        r2 = (s2name + "\t" + m["barcode"]).map(i2)
        ok = r1.notna() & r2.notna()
        print(f"{ref:24s} {len(m):7,} cells | in v1 {int(r1.notna().sum()):7,} | in v2 {int(r2.notna().sum()):7,} | "
              f"both {int(ok.sum()):7,} | samples via kept copy: {int((s2name != m['sample']).groupby(m['sample']).any().sum())}")
        cells.append(pd.DataFrame({"reference": ref, "sample": m["sample"][ok], "v2_sample": s2name[ok],
                                   "barcode": m["barcode"][ok], "cell_type": m["cell_type"][ok],
                                   "v1_row": r1[ok].astype(int), "v2_row": r2[ok].astype(int)}))
    cells = pd.concat(cells, ignore_index=True)
    cells["chunk"] = np.arange(len(cells)) * a.n_chunks // len(cells)
    from datasets import load_from_disk
    for tag, d, col in (("v1", V1, "v1_row"), ("v2", V2, "v2_row")):
        ds = load_from_disk(str(d / "dataset"))
        for c in range(a.n_chunks):
            p = Path(a.chunk_root) / tag / f"chunk_{c:02d}"
            if p.exists():
                sys.exit(f"{p} exists")
            ds.select(cells.loc[cells.chunk == c, col].tolist()).save_to_disk(str(p))
        print(f"wrote {a.n_chunks} {tag} chunks under {a.chunk_root}/{tag}")
    out.mkdir(parents=True)
    cells.to_csv(out / "cells.tsv", sep="\t", index=False)
    print(f"wrote {out}/cells.tsv: {len(cells):,} cells, {cells.reference.nunique()} references")


# ------------------------------------------------------------------------------------------ embed
def embed(a):
    import shutil
    import tempfile
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from embed_cells import extract_embeddings
    with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as tmp:
        extract_embeddings(a.model_dir, a.chunk, tmp, species=2, device="cuda")
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(Path(tmp) / "cell_embeddings.npy"), a.out)


# ------------------------------------------------------------------------------------------ score
def knn_cv(X, y, folds, dev):
    """Out-of-fold kNN predictions (cosine on unit rows; k nearest training cells; majority vote,
    ties to the smallest label code)."""
    import torch
    from sklearn.model_selection import StratifiedKFold
    Xt = torch.as_tensor(X, device=dev)
    pred = np.empty(len(y), dtype=int)
    n_lab = int(y.max()) + 1
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
    for tr, te in skf.split(np.zeros(len(y)), y):
        k = min(K, len(tr))
        ytr = torch.as_tensor(y[tr], device=dev)
        Xtr = Xt[tr]
        for s in range(0, len(te), 2048):
            b = te[s:s + 2048]
            nn = (Xt[b] @ Xtr.T).topk(k, dim=1).indices
            votes = torch.zeros(len(b), n_lab, device=dev).scatter_add_(1, ytr[nn], torch.ones(nn.shape, device=dev))
            pred[b] = votes.argmax(1).cpu().numpy()          # first maximum = smallest label code
    return pred


def silhouette_cosine(X, y, dev):
    """sklearn's silhouette_score(metric='cosine') in closed form on unit rows."""
    import torch
    Xt = torch.as_tensor(X, dtype=torch.float64, device=dev)
    yt = torch.as_tensor(y, device=dev)
    n_lab = int(y.max()) + 1
    S = torch.zeros(n_lab, X.shape[1], dtype=torch.float64, device=dev).index_add_(0, yt, Xt)
    n = torch.bincount(yt, minlength=n_lab).double()
    M = Xt @ S.T                                             # x_i . sum of unit vectors per label
    own = M.gather(1, yt[:, None]).squeeze(1)
    a_ = 1 - (own - 1) / (n[yt] - 1).clamp(min=1)
    d = 1 - M / n.clamp(min=1)
    d[torch.arange(len(y), device=dev), yt] = float("inf")
    d[:, n == 0] = float("inf")
    b_ = d.min(1).values
    s = (b_ - a_) / torch.maximum(a_, b_)
    s[n[yt] == 1] = 0                                        # sklearn: singleton clusters score 0
    return float(s.mean())


def ref_metrics(X, cells, dev):
    from sklearn.metrics import f1_score
    ct = pd.Categorical(cells.cell_type)
    y = ct.codes.astype(int)
    pred = knn_cv(X, y, FOLDS, dev)
    out = {"n_cells": len(y), "n_cell_types": len(ct.categories),
           "knn_macro_f1": float(f1_score(y, pred, average="macro")), "knn_accuracy": float(np.mean(pred == y)),
           "silhouette": silhouette_cosine(X, y, dev)}
    hit = tot = n_ct = 0
    for c in ct.categories:
        sub = cells[cells.cell_type == c]
        cnt = sub["sample"].value_counts()
        keep = cnt[cnt >= MIN_SAMPLE_CELLS].index
        if len(keep) < 2:
            continue
        m = (cells.cell_type == c).to_numpy() & cells["sample"].isin(keep).to_numpy()
        ys = pd.Categorical(cells["sample"][m]).codes.astype(int)
        p = knn_cv(X[m], ys, FOLDS, dev)
        hit += int((p == ys).sum()); tot += len(ys); n_ct += 1
    out.update(sample_knn_accuracy=hit / tot if tot else float("nan"), sample_knn_cells=tot, sample_knn_cell_types=n_ct)
    return out


def score(a):
    import torch
    from scipy.stats import wilcoxon
    out = Path(a.out)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cells = pd.read_csv(out / "cells.tsv", sep="\t", dtype={"sample": str, "barcode": str})
    emb = {}
    for model in ("april", "v2"):
        parts = [np.load(out / "emb" / f"{model}_chunk_{c:02d}.npy") for c in sorted(cells.chunk.unique())]
        E = np.concatenate(parts).astype(np.float32)
        if len(E) != len(cells):
            sys.exit(f"{model}: {len(E)} embeddings for {len(cells)} cells")
        emb[model] = E / np.linalg.norm(E, axis=1, keepdims=True)
    rows = []
    for ref, g in cells.groupby("reference", sort=True):
        for model in ("april", "v2"):
            r = {"reference": ref, "model": model, **ref_metrics(emb[model][g.index.to_numpy()], g.reset_index(drop=True), dev)}
            rows.append(r)
            print(json.dumps(r))
    tab = pd.DataFrame(rows)
    tab.to_csv(out / "c3_per_reference.tsv", sep="\t", index=False)
    summary = {"n_references": int(tab.reference.nunique()), "metrics": {}}
    for met in ("knn_macro_f1", "knn_accuracy", "silhouette", "sample_knn_accuracy"):
        w = tab.pivot(index="reference", columns="model", values=met).dropna()
        d = w["v2"] - w["april"]
        p = float(wilcoxon(w["v2"], w["april"]).pvalue) if len(w) > 1 and (d != 0).any() else float("nan")
        summary["metrics"][met] = {"n": len(w), "median_april": float(w["april"].median()),
                                   "median_v2": float(w["v2"].median()), "median_v2_minus_april": float(d.median()),
                                   "v2_higher": int((d > 0).sum()), "wilcoxon_p": p}
        print(f"{met:20s} n={len(w):2d}  median April {w['april'].median():.4f}  v2 {w['v2'].median():.4f}  "
              f"median d {d.median():+.4f}  v2 higher in {int((d > 0).sum())}/{len(w)}  Wilcoxon p={p:.3g}")
    (out / "c3_summary.json").write_text(json.dumps(summary, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    j = sub.add_parser("join")
    j.add_argument("--out", required=True)
    j.add_argument("--chunk-root", required=True)
    j.add_argument("--n-chunks", type=int, default=8)
    e = sub.add_parser("embed")
    e.add_argument("--model-dir", required=True)
    e.add_argument("--chunk", required=True)
    e.add_argument("--out", required=True)
    s = sub.add_parser("score")
    s.add_argument("--out", required=True)
    a = ap.parse_args()
    {"join": join, "embed": embed, "score": score}[a.cmd](a)


if __name__ == "__main__":
    main()
