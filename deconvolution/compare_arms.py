#!/usr/bin/env python3
"""Compare every Stage 8-14 output of two config arms, file by file (e.g. the default tree and a clean rebuild).

Each output folder of arm A (deconvolution.results_dir, genecompass_input_dir and its _human twin,
augur_input_dir, grn_dir, grn_human_dir, validation_dir for the Stage 14 sweeps) is paired with arm B's, and every file is matched by relative path:
- IDENTICAL: same bytes.
- EQUAL: tables (.tsv/.csv/.txt) with the same shape, the same non-numeric cells, and numeric cells within --atol.
- CLOSE: embeddings (.npy) whose per-row cosine is at least --min-cos; h5ad files with X within --atol.
- DIFFERS: anything else; ONLY_A / ONLY_B: a file present in one arm only.
Provenance JSON (timestamps, paths), logs and .rds files are skipped, because their bytes change on every run;
each .rds has derived tables that are compared instead.

    python deconvolution/compare_arms.py --a config/profiles/refs_v4all.yaml --out <dir>   # assembled tree vs default
    (--a and --b default to the default config; since 2026-11-03 the default is the clean rebuild)
"""
import argparse
import hashlib
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
KEYS = ["results_dir", "genecompass_input_dir", "augur_input_dir", "grn_dir", "grn_human_dir", "validation_dir"]
SKIP_SUFFIX = (".rds", ".log", ".out", ".err", ".png", ".pdf")


def arm_dirs(profile):
    """{key: absolute folder} for one arm, read through the same config loader the stages use."""
    env = dict(os.environ, PIPELINE_ROOT=str(ROOT))
    env.pop("PIPELINE_PROFILE", None)
    if profile:
        env["PIPELINE_PROFILE"] = profile
    code = ("import sys, json; sys.path.insert(0, 'lib'); from gene_utils import load_config\n"
            "c = load_config('config/pipeline_config.yaml')['deconvolution']\n"
            f"print(json.dumps({{k: c[k] for k in {KEYS!r}}}))")
    import json
    d = json.loads(subprocess.check_output([sys.executable, "-c", code], cwd=ROOT, env=env, text=True))
    d["genecompass_human_dir"] = d["genecompass_input_dir"] + "_human"
    return {k: (ROOT / v).resolve() for k, v in d.items()}


def files_under(base: Path):
    out = {}
    for dirpath, _, names in os.walk(base, followlinks=True):
        for n in names:
            p = Path(dirpath) / n
            rel = p.relative_to(base).as_posix()
            if n.startswith("provenance") or n.endswith(SKIP_SUFFIX) or "/." in "/" + rel:
                continue
            out[rel] = p
    return out


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def cmp_table(a, b, atol):
    sep = "," if a.suffix == ".csv" else "\t"
    try:
        x = pd.read_csv(a, sep=sep, low_memory=False)
        y = pd.read_csv(b, sep=sep, low_memory=False)
    except Exception as e:  # not a table
        return "DIFFERS", f"unreadable as table: {e.__class__.__name__}"
    if x.shape != y.shape or list(x.columns) != list(y.columns):
        return "DIFFERS", f"shape {x.shape} vs {y.shape}"
    num = [c for c in x.columns if pd.api.types.is_numeric_dtype(x[c]) and pd.api.types.is_numeric_dtype(y[c])]
    oth = [c for c in x.columns if c not in num]
    if oth and not x[oth].astype(str).equals(y[oth].astype(str)):
        bad = [c for c in oth if not x[c].astype(str).equals(y[c].astype(str))]
        return "DIFFERS", f"non-numeric columns differ: {bad[:5]}"
    if not num:
        return "EQUAL", "0"
    xa, ya = x[num].to_numpy(float), y[num].to_numpy(float)
    nan_mismatch = int((np.isnan(xa) != np.isnan(ya)).sum())
    d = np.nanmax(np.abs(xa - ya)) if xa.size else 0.0
    d = 0.0 if np.isnan(d) else float(d)
    if nan_mismatch:
        return "DIFFERS", f"NaN pattern differs in {nan_mismatch} cells; max |diff| {d:.3g}"
    return ("EQUAL" if d <= atol else "DIFFERS"), f"{d:.3g}"


def cmp_npy(a, b, min_cos):
    x, y = np.load(a, mmap_mode="r"), np.load(b, mmap_mode="r")
    if x.shape != y.shape:
        return "DIFFERS", f"shape {x.shape} vs {y.shape}"
    if x.ndim != 2:
        return ("CLOSE" if np.allclose(x, y) else "DIFFERS"), ""
    x, y = np.asarray(x, float), np.asarray(y, float)
    cos = (x * y).sum(1) / (np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1) + 1e-12)
    return ("CLOSE" if cos.min() >= min_cos else "DIFFERS"), f"min cosine {cos.min():.6f}"


def cmp_h5ad(a, b, atol):
    import anndata as ad
    x, y = ad.read_h5ad(a), ad.read_h5ad(b)
    if x.shape != y.shape or list(x.obs_names) != list(y.obs_names) or list(x.var_names) != list(y.var_names):
        return "DIFFERS", f"shape/names {x.shape} vs {y.shape}"
    xa = x.X.toarray() if hasattr(x.X, "toarray") else np.asarray(x.X)
    ya = y.X.toarray() if hasattr(y.X, "toarray") else np.asarray(y.X)
    d = float(np.abs(xa - ya).max()) if xa.size else 0.0
    return ("CLOSE" if d <= atol else "DIFFERS"), f"max |dX| {d:.3g}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", default="", help="profile for arm A (default: the default config)")
    ap.add_argument("--b", default="", help="profile for arm B (default: the default config)")
    ap.add_argument("--atol", type=float, default=1e-9)
    ap.add_argument("--min-cos", type=float, default=0.9999)
    ap.add_argument("--out", required=True, help="folder for compare_files.tsv and compare_summary.tsv")
    ap.add_argument("--only", nargs="*", default=None, help="compare only these folders (keys as in the summary)")
    a = ap.parse_args()
    A, B = arm_dirs(a.a), arm_dirs(a.b)
    rows = []
    for key in (a.only or A):
        fa = files_under(A[key]) if A[key].is_dir() else {}
        fb = files_under(B[key]) if B[key].is_dir() else {}
        for rel in sorted(set(fa) | set(fb)):
            if rel not in fb:
                rows.append((key, rel, "ONLY_A", "")); continue
            if rel not in fa:
                rows.append((key, rel, "ONLY_B", "")); continue
            pa, pb = fa[rel], fb[rel]
            if pa.stat().st_size == pb.stat().st_size and md5(pa) == md5(pb):
                rows.append((key, rel, "IDENTICAL", "")); continue
            suf = pa.suffix
            if suf in (".tsv", ".csv", ".txt"):
                v, note = cmp_table(pa, pb, a.atol)
            elif suf == ".npy":
                v, note = cmp_npy(pa, pb, a.min_cos)
            elif suf == ".h5ad":
                v, note = cmp_h5ad(pa, pb, a.atol)
            else:
                v, note = "DIFFERS", "bytes differ"
            rows.append((key, rel, v, note))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=["folder", "file", "verdict", "note"])
    df.to_csv(out / "compare_files.tsv", sep="\t", index=False)
    summ = df.groupby(["folder", "verdict"]).size().unstack(fill_value=0)
    summ.to_csv(out / "compare_summary.tsv", sep="\t")
    print(f"A: {a.a or '(default config)'}\nB: {a.b or '(default config)'}\n")
    print(summ.to_string())
    bad = df[df.verdict.isin(["DIFFERS", "ONLY_A", "ONLY_B"])]
    if len(bad):
        print(f"\n{len(bad)} files differ or are unpaired (first 40):")
        print(bad.head(40).to_string(index=False))


if __name__ == "__main__":
    main()
