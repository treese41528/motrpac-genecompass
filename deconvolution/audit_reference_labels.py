#!/usr/bin/env python3
"""audit_reference_labels.py -- the label audit pre-registered in REFERENCE_RELABEL_PREREG.md §4.

References built from the corpus are labelled per sample (each sample clustered alone, one consensus
label per cluster), so two samples can name the same cells differently. For every label found in at most
half of a reference's samples, this compares its expression profile with the labels found in at least two
samples, against the yardstick of how closely each label matches itself across samples, and proposes
MERGE / REVIEW / DISTINCT. DISTINCT becomes KEEP or DROP on cell count and plausibility, recorded by hand
in the audit document; this script decides nothing else.

Profiles: per (sample, label), summed raw counts as log2(CPM + 1); genes with mean >= 1 over the
reference's groups, top 2,000 by variance across groups; each gene centred over groups; Pearson r.

Output (per reference, under --out/<REF>/): groups.tsv, self_similarity.tsv, correlations.tsv (every
(sample|label) group against every other), audit.tsv; and
--out/audit_all.tsv, --out/metric_check.tsv, --out/tau.tsv.

Usage (project venv; reads every reference matrix -> run via SLURM):
  python deconvolution/audit_reference_labels.py --out data/deconvolution/label_audit
"""
import argparse
import json
import os
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(os.environ.setdefault("PIPELINE_ROOT", str(Path(__file__).resolve().parents[1])))

CONSENSUS = ["KIDNEY", "LIVER", "CORTEX", "WATSC", "SKMGN", "SMLINT", "LUNG", "HIPPOC"]  # SKMGN = MUSCLE ref
MERGE_ONLY = ["BAT", "HEART", "TESTES"]
N_GENES, MIN_MEAN, MARGIN, TAU_Q, MIN_LABELS_FOR_TAU = 2000, 1.0, 0.05, 10, 5


def group_sums(ref_dir):
    """Stream reference_counts.mtx (cells x genes) into per-(sample, label) count sums."""
    meta = pd.read_csv(ref_dir / "cells_meta.tsv", sep="\t", dtype=str)
    genes = pd.read_csv(ref_dir / "genes.tsv", header=None)[0].astype(str).tolist()
    keys = meta["sample"] + "\t" + meta["cell_type"]
    groups = sorted(keys.unique())
    gidx = {g: i for i, g in enumerate(groups)}
    cell_group = keys.map(gidx).to_numpy()
    sums = np.zeros((len(groups), len(genes)), dtype=np.float64)
    path = ref_dir / "reference_counts.mtx"
    with open(path) as f:
        header = 0
        for line in f:
            header += 1
            if not line.startswith("%"):
                n_cells, n_genes, _ = map(int, line.split())
                break
    assert n_cells == len(meta) and n_genes == len(genes), f"{path}: shape mismatch with meta/genes"
    for chunk in pd.read_csv(path, sep=" ", header=None, skiprows=header, names=["c", "g", "v"],
                             dtype={"c": np.int64, "g": np.int64, "v": np.float64}, chunksize=20_000_000):
        flat = cell_group[chunk["c"].to_numpy() - 1] * len(genes) + (chunk["g"].to_numpy() - 1)
        sums += np.bincount(flat, weights=chunk["v"].to_numpy(), minlength=sums.size).reshape(sums.shape)
    gt = pd.DataFrame([g.split("\t") for g in groups], columns=["sample", "label"])
    gt["n_cells"] = keys.value_counts().reindex(groups).to_numpy()
    gt["libsize"] = sums.sum(1)
    return gt, sums, genes, meta["sample"].nunique()


def centred_profiles(sums):
    cpm = sums / sums.sum(1, keepdims=True) * 1e6
    lg = np.log2(cpm + 1)
    keep = np.where(lg.mean(0) >= MIN_MEAN)[0]
    top = keep[np.argsort(-lg[:, keep].var(0))[:N_GENES]]
    x = lg[:, top]
    return x - x.mean(0), cpm


def corr_matrix(x):
    z = (x - x.mean(1, keepdims=True)) / x.std(1, keepdims=True)
    return z @ z.T / x.shape[1]


def audit_reference(name, ref_dir, symbols):
    gt, sums, genes, n_samples = group_sums(ref_dir)
    x, cpm = centred_profiles(sums)
    R = corr_matrix(x)
    samples_of = gt.groupby("label")["sample"].apply(set).to_dict()
    rows_of = gt.groupby("label").groups

    def cross(a, b):
        rs = [R[i, j] for i, j in product(rows_of[a], rows_of[b]) if gt.at[i, "sample"] != gt.at[j, "sample"]]
        return float(np.median(rs)) if rs else np.nan

    def within(a, b):
        rs = [R[i, j] for i, j in product(rows_of[a], rows_of[b]) if gt.at[i, "sample"] == gt.at[j, "sample"]]
        return float(np.median(rs)) if rs else np.nan

    multi = sorted(l for l, s in samples_of.items() if len(s) >= 2)
    self_sim = pd.DataFrame({"label": multi, "n_samples": [len(samples_of[l]) for l in multi],
                             "r_self": [cross(l, l) for l in multi]})
    # metric check: same label across samples vs different labels across samples
    diff = [R[i, j] for i in range(len(gt)) for j in range(i + 1, len(gt))
            if gt.at[i, "sample"] != gt.at[j, "sample"] and gt.at[i, "label"] != gt.at[j, "label"]]
    check = {"reference": name, "n_samples": n_samples, "n_groups": len(gt), "n_multi_labels": len(multi),
             "median_r_self": float(self_sim.r_self.median()) if len(multi) else np.nan,
             "median_r_different": float(np.median(diff)) if diff else np.nan}
    check["passes"] = bool(check["median_r_self"] > check["median_r_different"])

    audit_set = sorted(l for l, s in samples_of.items() if len(s) <= n_samples / 2 or len(s) == 1)
    audit_set = [l for l in audit_set if n_samples >= 2]
    rows = []
    for L in audit_set:
        cands = sorted(((cross(L, T), T) for T in multi if T != L), reverse=True)
        cands = [(r, T) for r, T in cands if r == r]
        best_r, best = cands[0] if cands else (np.nan, None)
        second_r, second = cands[1] if len(cands) > 1 else (np.nan, None)
        # markers: L's samples, L vs the other groups of those samples (log2 fold change of CPM)
        li = list(rows_of[L])
        oi = [i for i in range(len(gt)) if gt.at[i, "sample"] in samples_of[L] and i not in li]
        a = sums[li].sum(0); a = a / a.sum() * 1e6
        b = sums[oi].sum(0) if oi else np.zeros_like(a); b = b / max(b.sum(), 1) * 1e6
        lfc = np.log2(a + 1) - np.log2(b + 1)
        top = np.argsort(-lfc)[:10]
        rows.append({
            "reference": name, "label": L, "samples": ",".join(sorted(samples_of[L])),
            "n_samples_label": len(samples_of[L]), "n_samples_ref": n_samples,
            "n_cells": int(gt.loc[li, "n_cells"].sum()), "libsize": float(gt.loc[li, "libsize"].sum()),
            "best": best, "r_best": best_r, "second": second, "r_second": second_r,
            "margin": best_r - second_r if second else np.nan,
            "r_within_best": within(L, best) if best else np.nan,
            "top_markers": ",".join(f"{symbols.get(genes[g], genes[g])}({lfc[g]:.1f})" for g in top),
        })
    lab = (gt["sample"] + "|" + gt["label"]).tolist()
    corr = pd.DataFrame(R, index=lab, columns=lab)
    return gt, self_sim, pd.DataFrame(rows), check, corr


def decide(a, tau, merge_only):
    if a.best is None or not (a.r_best == a.r_best):
        return "DISTINCT"
    if a.r_best >= tau:
        if not (a.margin == a.margin) or a.margin >= MARGIN:
            return "MERGE"
        return "REVIEW" if not merge_only else "KEEP (review: near-tie)"
    return "DISTINCT" if not merge_only else "KEEP"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/deconvolution/label_audit")
    ap.add_argument("--only", nargs="*", help="reference keys to run (default: all in scope)")
    args = ap.parse_args()
    out = ROOT / args.out
    refs = yaml.safe_load(open(ROOT / "deconvolution/tissue_references.yaml"))
    info = pd.read_csv(ROOT / "data/references/biomart/rat_gene_info.tsv", sep="\t", dtype=str)
    symbols = dict(zip(info["Gene stable ID"], info["Gene name"].fillna("")))
    symbols = {k: v for k, v in symbols.items() if v}

    keys = args.only or CONSENSUS + MERGE_ONLY
    results, self_all, checks = {}, [], []
    for k in keys:
        ref_dir = ROOT / refs[k]["reference_dir"]
        name = "MUSCLE" if k in ("SKMGN", "SKMVL") else k
        print(f"[{name}] {ref_dir.name} ...", flush=True)
        gt, ss, au, ck, corr = audit_reference(name, ref_dir, symbols)
        (out / name).mkdir(parents=True, exist_ok=True)
        gt.to_csv(out / name / "groups.tsv", sep="\t", index=False)
        ss.to_csv(out / name / "self_similarity.tsv", sep="\t", index=False)
        corr.round(4).to_csv(out / name / "correlations.tsv", sep="\t")
        ck["merge_only"] = k in MERGE_ONLY
        results[name] = au; checks.append(ck); self_all.append(ss.assign(reference=name, merge_only=k in MERGE_ONLY))
        print(f"  groups={len(gt)} multi-sample labels={len(ss)} audited={len(au)} "
              f"check: r_self {ck['median_r_self']:.3f} vs different {ck['median_r_different']:.3f}", flush=True)

    self_all = pd.concat(self_all)
    pooled = float(np.percentile(self_all.loc[~self_all.merge_only, "r_self"], TAU_Q))
    taus = []
    for name, au in results.items():
        ss = self_all[self_all.reference == name]
        tau, src = (float(np.percentile(ss.r_self, TAU_Q)), "reference") if len(ss) >= MIN_LABELS_FOR_TAU \
            else (pooled, "pooled")
        taus.append({"reference": name, "tau": tau, "source": src, "n_labels": len(ss)})
        mo = bool(ss.merge_only.iloc[0]) if len(ss) else name in MERGE_ONLY
        if len(au):
            au["tau"] = tau
            au["proposed"] = [decide(r, tau, mo) for r in au.itertuples()]
        au.to_csv(out / name / "audit.tsv", sep="\t", index=False)
    audited = [a for a in results.values() if len(a)]
    (pd.concat(audited) if audited else pd.DataFrame()).to_csv(out / "audit_all.tsv", sep="\t", index=False)
    pd.DataFrame(checks).to_csv(out / "metric_check.tsv", sep="\t", index=False)
    pd.DataFrame(taus).to_csv(out / "tau.tsv", sep="\t", index=False)
    json.dump({"n_genes": N_GENES, "min_mean_log2cpm": MIN_MEAN, "margin": MARGIN, "tau_percentile": TAU_Q,
               "min_labels_for_reference_tau": MIN_LABELS_FOR_TAU, "pooled_tau": pooled},
              open(out / "settings.json", "w"), indent=2)
    print(f"done -> {out}")


if __name__ == "__main__":
    main()
