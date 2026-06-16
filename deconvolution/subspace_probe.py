#!/usr/bin/env python3
"""subspace_probe.py -- less-conservative re-measurement of covariate effects in the
768-d GeneCompass pseudo-cell embeddings, per (tissue x cell-type).

WHY. The Aim-2 gate (pheno_merge_test.py) measures a GLOBAL trace-eta^2
    eta2 = trace(between-SS) / trace(total-SS)   over all 768 dims
which is a VARIANCE-WEIGHTED average of the per-dimension eta^2:
    eta2 = sum_j Var_j * eta2_j  /  sum_j Var_j .
That deliberately conservative measure (a) DILUTES a signal living in a low-dimensional
subspace -- the denominator still sums ~760 off-target dims -- and (b) UP-WEIGHTS broad,
high-variance axes (sex). So a trained median eta^2 ~ 0.023 may understate a concentrated
exercise effect. This script adds three complementary, less-conservative measures and a
1000-permutation null for each, to decide whether 0.023 is genuinely weak or merely diluted:

  (0) GLOBAL trace eta^2          -- recomputed identically to the gate, for direct comparison.
  (1) STANDARDIZED trace eta^2    -- z-score each of the 768 columns first, then trace-eta^2.
        Removes the variance weighting => equals the UNWEIGHTED mean of per-dim eta^2.
        If the signal sits in low-variance dims, this rises above the global eta^2.
  (2) PER-DIMENSION scan          -- per-dim univariate eta^2_j; report max_j and top-10 mean,
        with a MAX-statistic permutation null (correctly corrects for scanning 768 dims).
        Catches an effect concentrated in a few dims that the trace average washes out.
  (3) SUPERVISED-SUBSPACE probe   -- cross-validated single-component PLS: the max-covariance
        direction w ∝ X^T y (the standard p>>n supervised projection; no covariance inversion,
        so no LDA singularity). Fit on TRAIN folds, project HELD-OUT folds, measure
        out-of-fold separation (AUC for binary trained/sex; Spearman for ordinal dose) and
        out-of-fold-projection eta^2. The permutation null re-runs the WHOLE CV under permuted
        labels (Ojala & Garriga 2010), so the metric is immune to the p>>n in-sample
        overfitting trap that makes an unregularized in-sample LDA/logistic separate perfectly.

Dose is modelled ORDINALLY (control=0, 1w=1, 2w=2, 4w=4, 8w=8 weeks) -- per the Aim-2 pivot --
in addition to the binary trained-vs-control. Sex is reported as a positive control: it should
be large under every measure.

Join logic is identical to pheno_merge_test.py (mix{i} -> i-th viallabel in bulk_samples.tsv
-> pheno[viallabel]). Sex-chromosome genes were removed upstream, so SEX is autosomal.

Usage:
  python deconvolution/subspace_probe.py [--perms 1000] [--folds 5] [--jobs 8]
         [--gc-root ...] [--pheno ...] [--bulk-root ...] [--out ...]
"""
import argparse
import glob
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_from_disk
from joblib import Parallel, delayed
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(os.environ.setdefault(
    "PIPELINE_ROOT", str(Path(__file__).resolve().parents[1]))) / "lib"))
from gene_utils import load_config, resolve_path                          # noqa: E402

WEEK = {"control": 0, "1w": 1, "2w": 2, "4w": 4, "8w": 8}                  # ordinal dose (weeks)


# ----------------------------------------------------------------------------- eta^2 helpers
def trace_eta2(E, labels):
    """trace(between-SS)/trace(total-SS) over all columns of E (the gate's measure)."""
    labels = np.asarray(labels); mu = E.mean(0)
    sst = ((E - mu) ** 2).sum()
    if sst <= 0:
        return np.nan
    ssb = sum(((E[labels == g].mean(0) - mu) ** 2).sum() * (labels == g).sum()
              for g in np.unique(labels))
    return ssb / sst


def perdim_eta2(E, labels):
    """Vector of per-dimension eta^2_j (length = n_features)."""
    labels = np.asarray(labels); mu = E.mean(0)
    sst = ((E - mu) ** 2).sum(0)                                          # (p,)
    ssb = np.zeros(E.shape[1])
    for g in np.unique(labels):
        m = labels == g
        ssb += ((E[m].mean(0) - mu) ** 2) * m.sum()
    return np.where(sst > 0, ssb / sst, 0.0)


def cat_eta2_1d(scores, labels):
    """One-way eta^2 of a 1-D score vector by a categorical label."""
    labels = np.asarray(labels); mu = scores.mean()
    sst = ((scores - mu) ** 2).sum()
    if sst <= 0:
        return np.nan
    ssb = sum((scores[labels == g].mean() - mu) ** 2 * (labels == g).sum()
              for g in np.unique(labels))
    return ssb / sst


# ----------------------------------------------------------------------- supervised PLS-1 CV
def make_folds(strat_labels, k, rng):
    """Fixed stratified folds on the TRUE labels; reused across permutations."""
    cnt = np.unique(strat_labels, return_counts=True)[1]
    kk = int(min(k, cnt.min()))
    if kk < 2:
        return None
    skf = StratifiedKFold(n_splits=kk, shuffle=True, random_state=int(rng.integers(1 << 30)))
    return list(skf.split(np.zeros(len(strat_labels)), strat_labels))


def oof_scores(E, y, folds):
    """Out-of-fold PLS-1 (max-covariance direction) projection scores."""
    oof = np.full(E.shape[0], np.nan)
    for tr, te in folds:
        Xtr, mu = E[tr], E[tr].mean(0)
        w = (Xtr - mu).T @ (y[tr] - y[tr].mean())                         # PLS-1 weight ∝ X^T y
        nrm = np.linalg.norm(w)
        if nrm > 0:
            w = w / nrm
        oof[te] = (E[te] - mu) @ w
    return oof


def supervised(E, folds, y_num, metric, strat_for_eta2, perms, rng):
    """CV supervised probe. metric: 'auc' (binary y_num in {0,1}) or 'spearman' (ordinal y_num).
    Returns (obs_metric, oof_eta2, p_perm). Permutation re-runs the full CV under permuted y."""
    def score(y):
        s = oof_scores(E, y, folds)
        ok = ~np.isnan(s)
        if metric == "auc":
            if len(np.unique(y[ok])) < 2:
                return np.nan
            return roc_auc_score(y[ok], s[ok])
        r = spearmanr(s[ok], y[ok]).correlation
        return r if r == r else np.nan                                    # nan-safe

    obs = score(y_num)
    if obs != obs:
        return np.nan, np.nan, np.nan
    oof = oof_scores(E, y_num, folds)
    ok = ~np.isnan(oof)
    e2 = cat_eta2_1d(oof[ok], np.asarray(strat_for_eta2)[ok])
    ge = 0
    for _ in range(perms):
        p = score(y_num[rng.permutation(len(y_num))])
        ge += (p >= obs) if p == p else 0
    return float(obs), float(e2), (ge + 1) / (perms + 1)


# --------------------------------------------------------------------------- per work-item
def process(item, perms, folds_k):
    E = item["E"]; seed = item["seed"]
    rng = np.random.default_rng(seed)
    grp, trn, sex, wk = item["group"], item["trained"], item["sex"], item["week"]
    n = E.shape[0]
    r = {"tissue": item["tissue"], "cell_type": item["cell_type"], "n": int(n)}

    # standardized copy (z-score columns, drop zero-variance dims)
    sd = E.std(0); keep = sd > 0
    Ez = (E[:, keep] - E[:, keep].mean(0)) / sd[keep]

    # max-stat permutation null shared across factors needs per-factor labels; do per factor.
    def perm_null(stat_fn, obs, labels):
        ge = sum(stat_fn(rng.permutation(labels)) >= obs for _ in range(perms))
        return (ge + 1) / (perms + 1)

    for name, lab in [("group", grp), ("trained", trn), ("sex", sex)]:
        lab = np.asarray(lab)
        if len(np.unique(lab)) < 2:
            continue
        # (0) global trace eta^2
        g_obs = trace_eta2(E, lab)
        r[f"glob_{name}"] = round(float(g_obs), 4)
        r[f"p_glob_{name}"] = round(perm_null(lambda L: trace_eta2(E, L), g_obs, lab), 4)
        # (1) standardized trace eta^2
        s_obs = trace_eta2(Ez, lab)
        r[f"std_{name}"] = round(float(s_obs), 4)
        r[f"p_std_{name}"] = round(perm_null(lambda L: trace_eta2(Ez, L), s_obs, lab), 4)
        # (2) per-dimension scan (max + top-10 mean), max-stat permutation null
        pd_obs = perdim_eta2(E, lab)
        mx = float(pd_obs.max()); top10 = float(np.sort(pd_obs)[-10:].mean())
        r[f"pdmax_{name}"] = round(mx, 4)
        r[f"pdtop10_{name}"] = round(top10, 4)
        r[f"p_pdmax_{name}"] = round(perm_null(lambda L: perdim_eta2(E, L).max(), mx, lab), 4)

    # (3) supervised CV probes -- exercise (trained binary + dose ordinal) and sex control
    # trained binary
    if len(np.unique(trn)) == 2:
        f = make_folds(trn, folds_k, np.random.default_rng(seed + 1))
        if f:
            y = (np.asarray(trn) == "trained").astype(float)
            auc, e2, p = supervised(E, f, y, "auc", trn, perms, np.random.default_rng(seed + 2))
            r["sup_trained_auc"], r["sup_trained_eta2"], r["p_sup_trained"] = \
                _r(auc), _r(e2), _r(p, 4)
    # dose ordinal (only meaningful if >1 distinct week incl control)
    if len(set(wk)) > 2:
        f = make_folds(grp, folds_k, np.random.default_rng(seed + 3))     # stratify on 5-level group
        if f:
            y = np.asarray(wk, float)
            rho, e2, p = supervised(E, f, y, "spearman", grp, perms, np.random.default_rng(seed + 4))
            r["sup_dose_rho"], r["sup_dose_eta2"], r["p_sup_dose"] = _r(rho), _r(e2), _r(p, 4)
    # sex control
    if len(np.unique(sex)) == 2:
        f = make_folds(sex, folds_k, np.random.default_rng(seed + 5))
        if f:
            y = (np.asarray(sex) == "male").astype(float)
            auc, e2, p = supervised(E, f, y, "auc", sex, perms, np.random.default_rng(seed + 6))
            r["sup_sex_auc"], r["sup_sex_eta2"], r["p_sup_sex"] = _r(auc), _r(e2), _r(p, 4)
    return r


def _r(x, nd=3):
    return round(float(x), nd) if x == x else np.nan


# ------------------------------------------------------------------------------------ main
def main():
    cfg = load_config(); dc = cfg["deconvolution"]
    gc = str(resolve_path(cfg, dc["genecompass_input_dir"]))
    ap = argparse.ArgumentParser()
    ap.add_argument("--gc-root", default=gc)
    ap.add_argument("--pheno", default=str(resolve_path(cfg, dc["sample_pheno"])))
    ap.add_argument("--bulk-root", default=str(resolve_path(cfg, dc["motrpac_bulk_out"])))
    ap.add_argument("--perms", type=int, default=1000)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--out", default=os.path.join(gc, "subspace_probe.tsv"))
    args = ap.parse_args()

    ph = pd.read_csv(args.pheno, sep="\t", dtype=str)
    ph["viallabel"] = ph["viallabel"].astype(str)
    gcol = "group" if "group" in ph.columns else "key.anirandgroup"
    phm = ph.drop_duplicates("viallabel").set_index("viallabel")

    items = []
    for d in sorted(glob.glob(f"{args.gc_root}/*/")):
        tis = os.path.basename(d.rstrip("/"))
        emb_p, ds_p = os.path.join(d, "embeddings", "cell_embeddings.npy"), os.path.join(d, "dataset")
        bs_p = os.path.join(args.bulk_root, tis.upper(), "bulk_samples.tsv")
        if not (os.path.exists(emb_p) and os.path.isdir(ds_p) and os.path.exists(bs_p)):
            continue
        emb = np.load(emb_p).astype(np.float64); ds = load_from_disk(ds_p)
        ct = np.array(ds["cell_type"]); samp = np.array(ds["sample"])
        if emb.shape[0] != len(ct):
            print(f"[{tis}] MISALIGNED -- skip"); continue
        vls = [l.strip() for l in open(bs_p) if l.strip()]
        idx = np.array([int(re.sub(r"\D", "", s)) - 1 for s in samp])
        vl = np.array([vls[i] if 0 <= i < len(vls) else "" for i in idx])
        grp = np.array([phm.loc[v, gcol] if v in phm.index else None for v in vl], dtype=object)
        sex = np.array([phm.loc[v, "sex"] if v in phm.index else None for v in vl], dtype=object)
        trained = np.array([None if g is None else
                            ("control" if "control" in str(g).lower() else "trained")
                            for g in grp], dtype=object)
        for ci, c in enumerate(sorted(set(ct))):
            m = (ct == c) & (grp != None) & (sex != None)                 # noqa: E711
            if m.sum() < 6:
                continue
            g = grp[m]
            items.append(dict(
                tissue=tis, cell_type=c, E=emb[m], group=g, trained=trained[m], sex=sex[m],
                week=np.array([WEEK.get(str(x), np.nan) for x in g], float),
                seed=1000 * (hash(tis) % 9973) + ci))

    print(f"{len(items)} (tissue x cell-type) blocks; perms={args.perms} folds={args.folds} "
          f"jobs={args.jobs}")
    rows = Parallel(n_jobs=args.jobs, verbose=5)(
        delayed(process)(it, args.perms, args.folds) for it in items)
    res = pd.DataFrame(rows)
    res.to_csv(args.out, sep="\t", index=False)

    # ----------------------------------------------------------------- summary (the science)
    pd.set_option("display.width", 240, "display.max_rows", 300, "display.max_columns", 60)
    print(f"\nwrote {args.out}  ({len(res)} rows)\n")

    def sig(col):
        return int((res[col] < 0.05).sum()) if col in res else 0

    print("=== significant (p<0.05) cell-type blocks, by measure x factor (of %d) ===" % len(res))
    hdr = f"{'factor':9s} {'global':>8s} {'standzd':>8s} {'perdim':>8s} {'supervis':>9s}"
    print(hdr)
    for fac, sup in [("trained", "p_sup_trained"), ("group", "p_sup_dose"), ("sex", "p_sup_sex")]:
        lab = "dose(ord)" if fac == "group" else fac
        print(f"{lab:9s} {sig('p_glob_'+fac):>8d} {sig('p_std_'+fac):>8d} "
              f"{sig('p_pdmax_'+fac):>8d} {sig(sup):>9d}")

    print("\n=== EXERCISE effect size: median / max across all blocks ===")
    for fac in ["trained"]:
        for tag, col in [("global trace eta2", "glob_"+fac), ("standardized eta2", "std_"+fac),
                         ("per-dim MAX eta2", "pdmax_"+fac), ("supervised OOF eta2", "sup_trained_eta2")]:
            if col in res:
                v = res[col].dropna()
                print(f"  {tag:22s} median={v.median():.3f}  max={v.max():.3f}")
    if "sup_trained_auc" in res:
        v = res["sup_trained_auc"].dropna()
        print(f"  supervised trained AUC  median={v.median():.3f}  max={v.max():.3f}")
    if "sup_dose_rho" in res:
        v = res["sup_dose_rho"].dropna()
        print(f"  supervised dose Spearman median={v.median():.3f}  max={v.max():.3f}")

    print("\n=== RESCUE: blocks where a less-conservative measure flags exercise that GLOBAL missed ===")
    miss = (res.get("p_glob_trained", pd.Series(1, index=res.index)) >= 0.05)
    for tag, col in [("standardized", "p_std_trained"), ("per-dim max", "p_pdmax_trained"),
                     ("supervised trained", "p_sup_trained"), ("supervised dose", "p_sup_dose")]:
        if col in res:
            resc = int((miss & (res[col] < 0.05)).sum())
            print(f"  global-trained NS but {tag:18s} sig: {resc}")

    print("\n=== top exercise responders by SUPERVISED trained AUC ===")
    if "sup_trained_auc" in res:
        keep = [c for c in ["tissue", "cell_type", "n", "glob_trained", "std_trained",
                            "pdmax_trained", "sup_trained_auc", "sup_trained_eta2",
                            "p_sup_trained", "sup_dose_rho", "p_sup_dose"] if c in res]
        print(res.sort_values("sup_trained_auc", ascending=False)[keep].head(20).to_string(index=False))

    print("\n=== per-tissue exercise summary (supervised trained AUC median/max; dose rho median/max) ===")
    if "sup_trained_auc" in res:
        agg = res.groupby("tissue").agg(
            n_ct=("cell_type", "size"),
            sup_auc_med=("sup_trained_auc", "median"), sup_auc_max=("sup_trained_auc", "max"),
            sup_sig=("p_sup_trained", lambda s: int((s < 0.05).sum())),
            dose_rho_med=("sup_dose_rho", "median") if "sup_dose_rho" in res else ("n", "size"),
            dose_sig=("p_sup_dose", lambda s: int((s < 0.05).sum())) if "p_sup_dose" in res else ("n", "size"),
        ).round(3)
        print(agg.to_string())


if __name__ == "__main__":
    main()
