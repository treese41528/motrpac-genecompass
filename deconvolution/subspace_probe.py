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
        With --repeats R the metric and eta^2 are means over R independent fold splits and each
        permutation is scored on the same R splits; one split moves a block's AUC by ~0.04.

Dose is modelled ORDINALLY (control=0, 1w=1, 2w=2, 4w=4, 8w=8 weeks) -- per the Aim-2 pivot --
in addition to the binary trained-vs-control. Sex is reported as a positive control: it should
be large under every measure.

THE GATE (docs/audit/remediation/STEP3_PREREG.md section 5). The trained-against-control probe of (3) is the
test behind the hotspots. From the Step 3 rerun on it is run as follows.

  Scorer (5.2, "Implementation check"). The out-of-fold score of a held-out sample is linear in the centred
      training labels, so each fold's scores for any label vector come from two matrices computed once
      (fold_kernels). KernelProbe scores a batch of permuted label vectors with them. It draws the same
      sequence of rng.permutation(n) calls as the loop it replaces, so the first 10,000 permutations are the
      registered ones. A permutation whose statistic ties the observed one, or whose scores tie each other,
      is scored by the loop itself (roc_auc_score), so ties are handled exactly as before. --scorer loop keeps
      the loop for every permutation. --equivalence-check compares the scorer with a stored table.
  Sequential sampling (5.2), with --sequential. After the first --perms permutations, blocks of the family
      whose class is still open get further batches until they are a hotspot, not a hotspot, or at the cap
      (deconvolution/mc_gate.py). The family is the set of analysable blocks: by default the registered rule
      of corroborate_summary.py applied to the Stage 8 output, or the `analysable` column of --family-table
      (Stage 12 passes the rat gate's table, so the human side is classified over the same blocks). Blocks
      outside the family keep the first batch.
  Sensitivities (5.3), for every block, --perms permutations each, none changing the primary:
      S1  folds stratified on trained x sex, trained labels permuted within sex;
      S2  8-week trained against control only.

Trained-probe columns: sup_trained_auc, sup_trained_eta2; b_sup_trained (permuted statistics at or above the
observed one), m_sup_trained (permutations), p_sup_trained = (b + 1) / (m + 1) unrounded; p_sup_trained_10k (the
p after the first --perms permutations, to 4 decimals: the registered 10,000-permutation value when --perms is
10,000); in_family, p_lo_sup_trained, p_hi_sup_trained, gate_class (with --sequential); sup_trained_auc_s1,
p_sup_trained_s1; n_s2, sup_trained_auc_s2, p_sup_trained_s2.

The sample join (mix{i} -> i-th viallabel in bulk_samples.tsv -> pheno[viallabel]) goes through
deconvolution/sample_join.py, which checks it against the Stage 8 output (section 2.5). Sex-chromosome genes
were removed upstream, so SEX is autosomal.

Usage:
  python deconvolution/subspace_probe.py [--perms 1000] [--folds 5] [--repeats 1] [--jobs 8]
         [--gc-root ...] [--pheno ...] [--bulk-root ...] [--results-root ...] [--out ...]
         [--sequential [--family-table <rat gate>/corroboration_merged.tsv]] [--scorer kernel|loop]
  python deconvolution/subspace_probe.py --equivalence-check <stored subspace_probe.tsv> --out <new table>
         --perms 10000 --repeats 10 [--check-loop]
"""
import os
# Pin BLAS/OpenMP to one thread BEFORE numpy/sklearn load: the production gate ran with these set to 1 in its
# SLURM script, and multi-threaded BLAS can change floating-point summation order. Parallelism here is
# joblib processes (--jobs), so results no longer depend on the calling script's environment.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import argparse
import glob
import json
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_from_disk
from joblib import Parallel, delayed
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(os.environ.setdefault(
    "PIPELINE_ROOT", str(Path(__file__).resolve().parents[1]))) / "lib"))
from gene_utils import load_config, resolve_path                          # noqa: E402
from provenance import write_provenance                                   # noqa: E402
import mc_gate                                                            # noqa: E402
from corroborate_summary import MIN_MEDIAN_LIBSIZE, is_analysable, median_libsize   # noqa: E402
from sample_join import load_viallabels, mix_index                        # noqa: E402

WEEK = {"control": 0, "1w": 1, "2w": 2, "4w": 4, "8w": 8}                  # ordinal dose (weeks)

# Seed offsets added to a block's seed. Offsets 0 to 6 are the registered streams of this file (0 the eta^2
# tests; 1, 3, 5 the folds and 2, 4, 6 the permutations of the trained, dose and sex probes); augur_prep.py uses
# 7, 11 and 13. The two sensitivities of section 5.3 get their own, far enough from the others that no stream
# of a neighbouring cell type (whose block seed differs by its rank in the tissue) can coincide with them.
S1_FOLD_SEED, S1_PERM_SEED = 200, 300
S2_FOLD_SEED, S2_PERM_SEED = 400, 500
S2_GROUPS = ("8w", "control")                                             # S2: 8-week trained against control

# Two out-of-fold scores of one permutation closer than this fraction of the largest score count as tied for
# the kernel scorer, which then hands the permutation to the loop. The two scorers differ by rounding error
# (about 1e-13 of the score scale), so a gap above the tolerance has the same sign in both.
TIE_TOL = 1e-9


def block_seed(tissue, ci):
    """Per-(tissue, cell-type) RNG seed. zlib.crc32 is stable across runs; the builtin hash() of a
    str is salted per interpreter (PYTHONHASHSEED), so folds and permutation nulls changed run to run."""
    return 1000 * (zlib.crc32(tissue.encode()) % 9973) + ci


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


def make_fold_sets(strat_labels, k, seed, repeats):
    """`repeats` stratified k-fold splits; split r is seeded seed + 7919 r, so r = 0 is the single split.
    None if a label has fewer than 2 members (the same for every split)."""
    sets = [make_folds(strat_labels, k, np.random.default_rng(seed + 7919 * r)) for r in range(repeats)]
    return sets if sets[0] else None


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


def probe_stat(E, y, fold_sets, metric):
    """The probe's statistic for one label vector: the mean over the splits of the out-of-fold metric
    ('auc' for binary y in {0, 1}, 'spearman' for ordinal y); nan if any split is nan. This is the registered
    computation (roc_auc_score on the pooled out-of-fold scores of each split) and the reference every
    faster scorer is held to."""
    def score_one(folds):
        s = oof_scores(E, y, folds)
        ok = ~np.isnan(s)
        if metric == "auc":
            if len(np.unique(y[ok])) < 2:
                return np.nan
            return roc_auc_score(y[ok], s[ok])
        r = spearmanr(s[ok], y[ok]).correlation
        return r if r == r else np.nan                                    # nan-safe

    return float(np.mean([score_one(f) for f in fold_sets]))


def oof_eta2(E, y_num, fold_sets, strat_for_eta2):
    """Mean over the splits of the one-way eta^2 of the out-of-fold score by label."""
    e2 = []
    for folds in fold_sets:
        oof = oof_scores(E, y_num, folds)
        ok = ~np.isnan(oof)
        e2.append(cat_eta2_1d(oof[ok], np.asarray(strat_for_eta2)[ok]))
    return float(np.mean(e2))


# ------------------------------------------------------------------- label permutations
def draw_unrestricted(y):
    """draw(rng) -> y permuted over all samples by one rng.permutation(n) call. This is the registered null of
    the primary probe; successive calls consume the generator exactly as the original loop did."""
    n = len(y)
    return lambda rng: y[rng.permutation(n)]


def draw_within(y, groups):
    """draw(rng) -> y permuted within each index group (S1: within sex), one rng.permutation call per group in
    the order given. Every group keeps its own number of each label (restricted permutation within a nuisance
    factor; section 5.3)."""
    def draw(rng):
        out = y.copy()
        for idx in groups:
            out[idx] = y[idx[rng.permutation(len(idx))]]
        return out
    return draw


def loop_exceedances(E, y, fold_sets, metric, obs, draw, rng, perms):
    """Number of permuted statistics at or above the observed one, each permutation re-running the whole CV on
    the same splits. A permuted statistic that is not a number counts as not exceeding. This is the original
    loop."""
    ge = 0
    for _ in range(perms):
        p = probe_stat(E, draw(rng), fold_sets, metric)
        ge += (p >= obs) if p == p else 0
    return int(ge)


def supervised(E, fold_sets, y_num, metric, strat_for_eta2, perms, rng):
    """CV supervised probe over one or more splits (make_fold_sets). metric: 'auc' (binary y_num in
    {0,1}) or 'spearman' (ordinal y_num). The statistic and the OOF eta^2 are means over the splits.
    Returns (obs_metric, oof_eta2, p_perm). Permutation re-runs the full CV on the same splits under
    permuted y."""
    obs = probe_stat(E, y_num, fold_sets, metric)
    if obs != obs:
        return np.nan, np.nan, np.nan
    e2 = oof_eta2(E, y_num, fold_sets, strat_for_eta2)
    ge = loop_exceedances(E, y_num, fold_sets, metric, obs, draw_unrestricted(y_num), rng, perms)
    return obs, e2, (ge + 1) / (perms + 1)


# ------------------------------------------------------- kernel scorer of the AUC probe (section 5.2)
def fold_kernels(E, fold_sets):
    """Per split, per fold: (train index, test index, K, G) with K = (E[te] - mu)(E[tr] - mu)^T and
    G = (E[tr] - mu)(E[tr] - mu)^T, mu the training-fold mean.

    oof_scores gives a held-out sample the score (E[te] - mu) @ w with w = (E[tr] - mu)^T yc scaled to unit
    length, yc the centred training labels. So the fold's scores are K yc / sqrt(yc^T G yc): linear in yc up to
    one positive factor per fold. K and G do not depend on the labels and are computed once per block."""
    out = []
    for folds in fold_sets:
        split = []
        for tr, te in folds:
            mu = E[tr].mean(0)
            A, B = E[tr] - mu, E[te] - mu
            split.append((tr, te, B @ A.T, A @ A.T))
        out.append(split)
    return out


def kernel_split_scores(split, Y):
    """Pooled out-of-fold scores of one split for every row of Y (permutations x samples)."""
    S = np.full(Y.shape, np.nan)
    for tr, te, K, G in split:
        Yc = Y[:, tr] - Y[:, tr].mean(axis=1, keepdims=True)
        num = Yc @ K.T
        den = np.sqrt(np.maximum(((Yc @ G) * Yc).sum(axis=1), 0.0))
        ok = den > 0                                                      # den = 0: w = 0, scores stay 0, as in oof_scores
        num[ok] = num[ok] / den[ok, None]
        S[:, te] = num
    return S


def pair_counts(S, Yi):
    """Per row of S: U, the number of (label 1, label 0) pairs in which the label-1 sample has the higher
    score, and a flag for rows where U cannot be trusted (a tie or near-tie between two scores, or a score
    that is not finite). AUC = U / (n1 n0) when there is no tie."""
    order = np.argsort(S, axis=1, kind="stable")
    Ss = np.take_along_axis(S, order, axis=1)
    Ys = np.take_along_axis(Yi, order, axis=1)
    u = (Ys * np.cumsum(1 - Ys, axis=1)).sum(axis=1)                     # label-0 samples ranked below each label-1
    scale = np.maximum(np.abs(Ss[:, 0]), np.abs(Ss[:, -1]))
    with np.errstate(invalid="ignore"):
        unsafe = ~np.isfinite(Ss).all(axis=1) | (np.diff(Ss, axis=1) <= TIE_TOL * scale[:, None]).any(axis=1)
    return u, unsafe


def two_u(scores, y):
    """Twice the Mann-Whitney U of the label-1 scores over the label-0 scores, ties counting one half each:
    an integer, and roc_auc_score = two_u / (2 n1 n0)."""
    d = scores[y == 1][:, None] - scores[y == 0][None, :]
    return 2 * int((d > 0).sum()) + int((d == 0).sum())


class KernelProbe:
    """Exceedance counts of the AUC probe from precomputed fold kernels, equal to the loop's by construction.

    The statistic is the mean over R splits of an AUC, and AUC x n1 x n0 is a whole number of ordered pairs
    (half numbers with ties). So a permuted statistic is above or below the observed one by at least
    1 / (2 n1 n0 R), far more than rounding error, unless the two are exactly equal as fractions. The kernel
    path decides every permutation whose pair count differs from the observed one. The rest are scored by
    probe_stat and compared as floating-point numbers, exactly as the loop compares them: a permutation that
    ties the observed statistic, and a permutation in which two scores tie or nearly tie (where the pair count
    could differ between the two computations, and where roc_auc_score's own tie rule applies).
    """
    metric = "auc"

    def __init__(self, E, y, fold_sets):
        self.E, self.y, self.fold_sets = E, y, fold_sets
        self.obs = probe_stat(E, y, fold_sets, "auc")
        if self.obs != self.obs:
            return
        n1 = int((y == 1).sum()); n0 = int((y == 0).sum())
        if n1 + n0 != len(y):
            raise ValueError("the AUC probe needs labels in {0, 1}")
        self.obs_2u = sum(two_u(oof_scores(E, y, folds), y) for folds in fold_sets)
        if abs(self.obs_2u / (2.0 * n1 * n0 * len(fold_sets)) - self.obs) > 1e-9:
            raise RuntimeError("the pair count of the observed scores does not reproduce the observed AUC")
        self.kernels = fold_kernels(E, fold_sets)

    def count(self, draw, rng, perms, chunk=5000):
        """(number of the next `perms` permuted statistics at or above the observed one, number of them scored
        by the loop). Permutations are drawn one draw(rng) call at a time, as the loop draws them."""
        ge = n_loop = done = 0
        while done < perms:
            k = min(chunk, perms - done)
            Y = np.stack([draw(rng) for _ in range(k)])
            Yi = Y.astype(np.int64)
            u2 = np.zeros(k, dtype=np.int64)
            unsafe = np.zeros(k, dtype=bool)
            for split in self.kernels:
                u, bad = pair_counts(kernel_split_scores(split, Y), Yi)
                u2 += 2 * u
                unsafe |= bad
            to_loop = unsafe | (u2 == self.obs_2u)
            ge += int((u2[~to_loop] > self.obs_2u).sum())
            for i in np.flatnonzero(to_loop):
                p = probe_stat(self.E, Y[i], self.fold_sets, "auc")
                ge += (p >= self.obs) if p == p else 0
            n_loop += int(to_loop.sum())
            done += k
        return int(ge), n_loop


class LoopProbe:
    """The original permutation loop behind the same interface as KernelProbe (--scorer loop)."""
    metric = "auc"

    def __init__(self, E, y, fold_sets):
        self.E, self.y, self.fold_sets = E, y, fold_sets
        self.obs = probe_stat(E, y, fold_sets, "auc")

    def count(self, draw, rng, perms):
        return loop_exceedances(self.E, self.y, self.fold_sets, "auc", self.obs, draw, rng, perms), perms


SCORERS = {"kernel": KernelProbe, "loop": LoopProbe}


# --------------------------------------------------------------------------- per work-item
def trained_probe(item, folds_k, repeats, scorer):
    """The primary probe of a block (trained against control, section 5.1): (probe, draw, labels), or None when
    the block has one class or a class too small to split. Folds are seeded block seed + 1."""
    trn = item["trained"]
    if len(np.unique(trn)) != 2:
        return None
    f = make_fold_sets(trn, folds_k, item["seed"] + 1, repeats)
    if not f:
        return None
    y = (np.asarray(trn) == "trained").astype(float)
    return SCORERS[scorer](item["E"], y, f), draw_unrestricted(y), trn


def more_trained_perms(item, folds_k, repeats, rng_state, perms, scorer):
    """Further permutations of a block's primary probe, continuing its generator from rng_state. Returns
    (exceedances among them, the generator's new state)."""
    probe, draw, _ = trained_probe(item, folds_k, repeats, scorer)
    rng = np.random.default_rng()
    rng.bit_generator.state = rng_state
    ge, _ = probe.count(draw, rng, perms)
    return ge, rng.bit_generator.state


def s1_probe(item, folds_k, repeats, scorer):
    """S1 (section 5.3): folds stratified on trained x sex, and a draw that permutes the trained labels within
    sex. Returns (probe, draw) or None. In a single-sex tissue this is the primary design on its own seeds."""
    trn, sex = np.asarray(item["trained"]), np.asarray(item["sex"])
    if len(np.unique(trn)) != 2:
        return None
    strata = np.array([f"{t}|{s}" for t, s in zip(trn, sex)])
    f = make_fold_sets(strata, folds_k, item["seed"] + S1_FOLD_SEED, repeats)
    if not f:
        return None
    y = (trn == "trained").astype(float)
    groups = [np.flatnonzero(sex == s) for s in sorted(set(sex))]
    return SCORERS[scorer](item["E"], y, f), draw_within(y, groups)


def s2_probe(item, folds_k, repeats, scorer):
    """S2 (section 5.3): the block restricted to the 8-week trained and the control animals, 8w against
    control, folds stratified on that label. Returns (probe, draw, n) or None."""
    grp = np.asarray(item["group"]).astype(str)
    keep = np.isin(grp, S2_GROUPS)
    g = grp[keep]
    if len(np.unique(g)) != 2:
        return None
    f = make_fold_sets(g, folds_k, item["seed"] + S2_FOLD_SEED, repeats)
    if not f:
        return None
    y = (g == S2_GROUPS[0]).astype(float)
    return SCORERS[scorer](item["E"][keep], y, f), draw_unrestricted(y), int(keep.sum())


def process(item, perms, folds_k, repeats=1, scorer="kernel"):
    """All measures of one block. Returns (row, state): state is the bit-generator state of the primary
    probe's permutation stream after the first `perms` permutations (None when the block has no primary
    test), from which more_trained_perms continues."""
    E = item["E"]; seed = item["seed"]
    rng = np.random.default_rng(seed)
    grp, trn, sex, wk = item["group"], item["trained"], item["sex"], item["week"]
    n = E.shape[0]
    r = {"tissue": item["tissue"], "cell_type": item["cell_type"], "n": int(n)}
    state = None

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
    # trained binary: the gate's primary test (section 5.1). Permutations from generator seed + 2, as registered.
    tp = trained_probe(item, folds_k, repeats, scorer)
    if tp and tp[0].obs == tp[0].obs:
        probe, draw, strat = tp
        prng = np.random.default_rng(seed + 2)
        b, _ = probe.count(draw, prng, perms)
        p = (b + 1) / (perms + 1)
        r["sup_trained_auc"] = _r(probe.obs)
        r["sup_trained_eta2"] = _r(oof_eta2(E, probe.y, probe.fold_sets, strat))
        r["b_sup_trained"], r["m_sup_trained"] = int(b), int(perms)
        r["p_sup_trained"] = p                                            # unrounded; more permutations may refine it
        r["p_sup_trained_10k"] = _r(p, 4)                                 # the registered value, rounded as before
        state = prng.bit_generator.state
    # dose ordinal (only meaningful if >1 distinct week incl control)
    if len(set(wk)) > 2:
        f = make_fold_sets(grp, folds_k, seed + 3, repeats)               # stratify on 5-level group
        if f:
            y = np.asarray(wk, float)
            rho, e2, p = supervised(E, f, y, "spearman", grp, perms, np.random.default_rng(seed + 4))
            r["sup_dose_rho"], r["sup_dose_eta2"], r["p_sup_dose"] = _r(rho), _r(e2), _r(p, 4)
    # sex control
    if len(np.unique(sex)) == 2:
        f = make_fold_sets(sex, folds_k, seed + 5, repeats)
        if f:
            y = (np.asarray(sex) == "male").astype(float)
            auc, e2, p = supervised(E, f, y, "auc", sex, perms, np.random.default_rng(seed + 6))
            r["sup_sex_auc"], r["sup_sex_eta2"], r["p_sup_sex"] = _r(auc), _r(e2), _r(p, 4)

    # Sensitivities of section 5.3: extra columns, `perms` permutations each, their own seeds.
    s1 = s1_probe(item, folds_k, repeats, scorer)
    if s1 and s1[0].obs == s1[0].obs:
        b, _ = s1[0].count(s1[1], np.random.default_rng(seed + S1_PERM_SEED), perms)
        r["sup_trained_auc_s1"], r["p_sup_trained_s1"] = _r(s1[0].obs), (b + 1) / (perms + 1)
    s2 = s2_probe(item, folds_k, repeats, scorer)
    if s2:
        r["n_s2"] = s2[2]
        if s2[0].obs == s2[0].obs:
            b, _ = s2[0].count(s2[1], np.random.default_rng(seed + S2_PERM_SEED), perms)
            r["sup_trained_auc_s2"], r["p_sup_trained_s2"] = _r(s2[0].obs), (b + 1) / (perms + 1)
    return r, state


def _r(x, nd=3):
    return round(float(x), nd) if x == x else np.nan


# ------------------------------------------------------------------------- the blocks
def load_items(gc_root, pheno, bulk_root, results_motrpac):
    """One work item per (tissue, cell type) with at least 6 pseudo-cells of known group and sex."""
    ph = pd.read_csv(pheno, sep="\t", dtype=str)
    ph["viallabel"] = ph["viallabel"].astype(str)
    gcol = "group" if "group" in ph.columns else "key.anirandgroup"
    phm = ph.drop_duplicates("viallabel").set_index("viallabel")

    items = []
    for d in sorted(glob.glob(f"{gc_root}/*/")):
        tis = os.path.basename(d.rstrip("/"))
        emb_p, ds_p = os.path.join(d, "embeddings", "cell_embeddings.npy"), os.path.join(d, "dataset")
        bs_p = os.path.join(bulk_root, tis.upper(), "bulk_samples.tsv")
        if not (os.path.exists(emb_p) and os.path.isdir(ds_p) and os.path.exists(bs_p)):
            continue
        emb = np.load(emb_p).astype(np.float64); ds = load_from_disk(ds_p)
        ct = np.array(ds["cell_type"]); samp = np.array(ds["sample"])
        if emb.shape[0] != len(ct):
            print(f"[{tis}] MISALIGNED -- skip"); continue
        vls = load_viallabels(bulk_root, tis, results_motrpac)            # checked sample key (section 2.5)
        vl = np.array(vls)[mix_index(samp, len(vls), context=tis)]
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
                seed=block_seed(tis, ci)))
    return items


def family_of(items, family_table, results_motrpac):
    """The blocks the multiplicity control spans, as a set of (tissue, cell type), and a description of where
    it came from. With a table: the rows whose `analysable` column is true. Otherwise the registered rule
    (median per-sample library of at least MIN_MEDIAN_LIBSIZE), from the same function and threshold
    corroborate_summary.py applies."""
    if family_table:
        t = pd.read_csv(family_table, sep="\t")
        if "analysable" not in t.columns:
            sys.exit(f"ERROR: {family_table} has no `analysable` column to define the family")
        fam = t[t["analysable"].astype(str).str.lower() == "true"]
        return set(zip(fam["tissue"], fam["cell_type"])), f"analysable column of {family_table}"
    if not Path(results_motrpac).is_dir():
        sys.exit(f"ERROR: --sequential needs the Stage 8 output ({results_motrpac}) for the analysable rule, "
                 "or --family-table")
    keys = [(it["tissue"], it["cell_type"]) for it in items]
    lib = [median_libsize(results_motrpac, t, c) for t, c in keys]
    return ({k for k, v in zip(keys, lib) if is_analysable(v)},
            f"median library >= {MIN_MEDIAN_LIBSIZE:g} in {results_motrpac}")


def run_sequential(items, rows, states, in_family, args):
    """Section 5.2: more permutations for the family's undecided blocks, then intervals and classes into the
    rows. Returns the record that goes into provenance."""
    fam = [i for i in range(len(rows)) if in_family[i] and states[i] is not None]

    def sample_more(index, n):
        todo = [fam[j] for j in index]
        out = Parallel(n_jobs=args.jobs)(
            delayed(more_trained_perms)(items[i], args.folds, args.repeats, states[i], n, args.scorer)
            for i in todo)
        for i, (_, st) in zip(todo, out):
            states[i] = st
        return [ge for ge, _ in out]

    res = mc_gate.sequential_gate(
        [rows[i]["b_sup_trained"] for i in fam], [rows[i]["m_sup_trained"] for i in fam], sample_more,
        eps=args.mc_eps, alpha=args.mc_alpha, batch=args.mc_batch, cap=args.mc_cap, looks=args.mc_looks,
        log=lambda s: print(f"[sequential] {s}", flush=True))
    for j, i in enumerate(fam):
        b, m = int(res["b"][j]), int(res["m"][j])
        rows[i].update(b_sup_trained=b, m_sup_trained=m, p_sup_trained=(b + 1) / (m + 1),
                       p_lo_sup_trained=float(res["lo"][j]), p_hi_sup_trained=float(res["hi"][j]),
                       gate_class=res["cls"][j])
    return dict(n_family=len(fam), interval_error=res["delta"], looks_used=res["looks_used"],
                n_empty_intervals=res["n_empty"],
                classes={c: int((res["cls"] == c).sum()) for c in (mc_gate.HOTSPOT, mc_gate.BORDERLINE, mc_gate.NONE)},
                max_permutations=int(res["m"].max()) if len(fam) else 0)


# --------------------------------------------------- implementation check of the scorer (section 5.2)
def implied_count(p4, perms):
    """Exceedance count behind a stored p = round((b + 1) / (perms + 1), 4), or nan where the rounding does not
    determine it. For 10,000 permutations the stored value identifies b except at 0.5000, which b = 4999 and
    b = 5000 share."""
    if p4 != p4:
        return np.nan
    cand = [b for b in range(max(0, int(p4 * (perms + 1)) - 3), min(perms, int(p4 * (perms + 1)) + 3) + 1)
            if round((b + 1) / (perms + 1), 4) == round(float(p4), 4)]
    return cand[0] if len(cand) == 1 else np.nan


def equivalence_check(items, args):
    """Recompute the primary probe of every block with the kernel scorer and compare it with a stored table
    (and, with --check-loop, with the loop run on the same generator). Writes one row per block to --out and
    returns the summary. Registered acceptance: identical values expected; a difference of at most 2 in a
    block's exceedance count is tolerated as a floating-point tie; anything larger keeps the loop."""
    def one(item):
        out = {"tissue": item["tissue"], "cell_type": item["cell_type"], "n": int(item["E"].shape[0])}
        tp = trained_probe(item, args.folds, args.repeats, "kernel")
        if not tp or tp[0].obs != tp[0].obs:
            return out
        probe, draw, _ = tp
        b, n_loop = probe.count(draw, np.random.default_rng(item["seed"] + 2), args.perms)
        out.update(auc_new=_r(probe.obs), b_new=b, p_new=_r((b + 1) / (args.perms + 1), 4),
                   n_scored_by_loop=n_loop)
        if args.check_loop:
            out["b_loop"] = loop_exceedances(item["E"], probe.y, probe.fold_sets, "auc", probe.obs, draw,
                                             np.random.default_rng(item["seed"] + 2), args.perms)
        return out

    new = pd.DataFrame(Parallel(n_jobs=args.jobs, verbose=5)(delayed(one)(it) for it in items))
    for c in ("auc_new", "b_new", "p_new", "n_scored_by_loop"):
        if c not in new:
            new[c] = np.nan
    old = pd.read_csv(args.equivalence_check, sep="\t")[["tissue", "cell_type", "sup_trained_auc", "p_sup_trained"]]
    old = old.rename(columns={"sup_trained_auc": "auc_old", "p_sup_trained": "p_old"})
    t = old.merge(new, on=["tissue", "cell_type"], how="outer", indicator="in_tables")
    t["in_tables"] = t["in_tables"].map({"both": "both", "left_only": "stored_only", "right_only": "new_only"})
    t["b_old_implied"] = [implied_count(p, args.perms) for p in t["p_old"]]
    t["d_b"] = t["b_new"] - t["b_old_implied"]
    same = lambda a, b: (t[a] == t[b]) | (t[a].isna() & t[b].isna())      # noqa: E731
    t["auc_equal"], t["p_equal"] = same("auc_old", "auc_new"), same("p_old", "p_new")
    if args.check_loop:
        t["d_b_loop"] = t["b_new"] - t["b_loop"]
    cols = ["tissue", "cell_type", "n", "in_tables", "auc_old", "auc_new", "auc_equal", "p_old", "p_new", "p_equal",
            "b_old_implied", "b_new", "d_b", "n_scored_by_loop"] + (["b_loop", "d_b_loop"] if args.check_loop else [])
    t[cols].to_csv(args.out, sep="\t", index=False)

    def largest(col):                                                     # largest absolute difference, 0 if none
        return float(t[col].abs().max()) if t[col].notna().any() else 0.0

    max_db = largest("d_b")
    summary = dict(
        stored_table=str(args.equivalence_check), blocks_stored=int(len(old)), blocks_recomputed=int(len(new)),
        blocks_in_one_table_only=int((t["in_tables"] != "both").sum()),
        blocks_with_a_test=int(t["b_new"].notna().sum()),
        auc_differ=int((~t["auc_equal"]).sum()), p_differ=int((~t["p_equal"]).sum()),
        max_abs_count_difference_vs_stored=max_db,
        blocks_count_not_implied_by_stored_p=int((t["p_old"].notna() & t["b_old_implied"].isna()).sum()),
        permutations_scored_by_loop=int(t["n_scored_by_loop"].sum()), perms=args.perms, repeats=args.repeats)
    if args.check_loop:
        summary["count_differ_vs_loop"] = int((t["d_b_loop"].fillna(0) != 0).sum())
        summary["max_abs_count_difference_vs_loop"] = largest("d_b_loop")
        max_db = max(max_db, summary["max_abs_count_difference_vs_loop"])
    clean = summary["blocks_in_one_table_only"] == 0 and summary["auc_differ"] == 0
    summary["verdict"] = ("identical" if clean and summary["p_differ"] == 0 and max_db == 0 else
                          "tolerated: at most 2 per block" if clean and max_db <= 2 else "keep the loop")
    Path(args.out).with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return summary


# ------------------------------------------------------------------------------------ main
def main():
    cfg = load_config(); dc = cfg["deconvolution"]
    gc = str(resolve_path(cfg, dc["genecompass_input_dir"]))
    ap = argparse.ArgumentParser()
    ap.add_argument("--gc-root", default=gc)
    ap.add_argument("--pheno", default=str(resolve_path(cfg, dc["sample_pheno"])))
    ap.add_argument("--bulk-root", default=str(resolve_path(cfg, dc["motrpac_bulk_out"])))
    ap.add_argument("--results-root", default=str(resolve_path(cfg, dc["results_dir"])),
                    help="deconvolution results root (holds motrpac/<TISSUE>/): checks the sample key and "
                         "supplies the analysable rule for --sequential (default: config results_dir)")
    ap.add_argument("--perms", type=int, default=1000)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=1,
                    help="fold splits per supervised probe; statistic = mean over splits, and each "
                         "permutation is scored on the same splits (1 = a single split)")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--scorer", choices=sorted(SCORERS), default="kernel",
                    help="permutation scorer of the trained probe and its sensitivities: 'kernel' (fold "
                         "kernels, same counts as the loop) or 'loop' (the original loop)")
    ap.add_argument("--sequential", action="store_true",
                    help="section 5.2: sample the family's undecided blocks in further batches and write "
                         "intervals and classes")
    ap.add_argument("--family-table", default=None,
                    help="table whose `analysable` column defines the family (Stage 12: the rat gate's "
                         "corroboration_merged.tsv); default: the analysable rule on --results-root")
    ap.add_argument("--mc-eps", type=float, default=mc_gate.EPS)
    ap.add_argument("--mc-alpha", type=float, default=mc_gate.ALPHA)
    ap.add_argument("--mc-batch", type=int, default=mc_gate.BATCH)
    ap.add_argument("--mc-cap", type=int, default=mc_gate.CAP)
    ap.add_argument("--mc-looks", type=int, default=mc_gate.LOOKS)
    ap.add_argument("--equivalence-check", default=None, metavar="STORED_TSV",
                    help="only recompute the trained probe with the kernel scorer and compare it with this "
                         "stored subspace_probe.tsv; the comparison goes to --out")
    ap.add_argument("--check-loop", action="store_true",
                    help="with --equivalence-check: also run the loop on the same generator and compare counts")
    ap.add_argument("--out", default=None, help="output table (default: <gc-root>/subspace_probe.tsv)")
    args = ap.parse_args()
    if args.out is None:
        if args.equivalence_check:
            ap.error("--equivalence-check needs --out")
        args.out = os.path.join(args.gc_root, "subspace_probe.tsv")
    if args.sequential and mc_gate.looks_needed(args.perms, args.mc_batch, args.mc_cap) > args.mc_looks:
        ap.error(f"--mc-looks {args.mc_looks} cannot cover a first look at --perms {args.perms}, batches of "
                 f"{args.mc_batch} and a cap of {args.mc_cap}")
    results_motrpac = Path(args.results_root) / "motrpac"

    items = load_items(args.gc_root, args.pheno, args.bulk_root, results_motrpac)
    inputs = ([args.pheno] + sorted(glob.glob(f"{args.gc_root}/*/embeddings/cell_embeddings.npy"))
              + sorted(glob.glob(f"{args.bulk_root}/*/bulk_samples.tsv")))
    seeds = dict(S1_FOLD_SEED=S1_FOLD_SEED, S1_PERM_SEED=S1_PERM_SEED, S2_FOLD_SEED=S2_FOLD_SEED,
                 S2_PERM_SEED=S2_PERM_SEED, S2_GROUPS=list(S2_GROUPS), TIE_TOL=TIE_TOL)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    if args.equivalence_check:
        print(f"{len(items)} blocks; equivalence check against {args.equivalence_check}; perms={args.perms} "
              f"folds={args.folds} repeats={args.repeats} jobs={args.jobs} check_loop={args.check_loop}")
        summary = equivalence_check(items, args)
        write_provenance(Path(args.out).parent, params=dict(vars(args), summary=summary, **seeds),
                         inputs=inputs + [args.equivalence_check], name="provenance_gate_equivalence.json")
        sys.exit(0 if summary["verdict"] != "keep the loop" else 1)

    family, family_source = (family_of(items, args.family_table, results_motrpac)
                             if args.sequential else (None, None))
    print(f"{len(items)} (tissue x cell-type) blocks; perms={args.perms} folds={args.folds} "
          f"repeats={args.repeats} jobs={args.jobs} scorer={args.scorer}")
    done = Parallel(n_jobs=args.jobs, verbose=5)(
        delayed(process)(it, args.perms, args.folds, args.repeats, args.scorer) for it in items)
    rows, states = [d[0] for d in done], [d[1] for d in done]
    sequential = None
    if args.sequential:
        in_family = [(r["tissue"], r["cell_type"]) in family for r in rows]
        for r, f in zip(rows, in_family):
            r["in_family"] = bool(f)
        print(f"family: {sum(in_family)} of {len(rows)} blocks ({family_source})")
        sequential = dict(run_sequential(items, rows, states, in_family, args), family_source=family_source)
    res = pd.DataFrame(rows)
    for c in ("b_sup_trained", "m_sup_trained", "n_s2"):                  # counts stay whole numbers beside empty cells
        if c in res:
            res[c] = res[c].astype("Int64")
    res.to_csv(args.out, sep="\t", index=False)
    write_provenance(Path(args.out).parent, params=dict(vars(args), sequential=sequential, **seeds),
                     inputs=inputs + ([args.family_table] if args.family_table else []))

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

    if args.sequential and "gate_class" in res:
        print("\n=== gate classes over the family (section 5.2) ===")
        fam = res[res["in_family"] & res["gate_class"].notna()]
        print(fam["gate_class"].value_counts().to_string())
        keep = ["tissue", "cell_type", "sup_trained_auc", "b_sup_trained", "m_sup_trained", "p_sup_trained_10k",
                "p_sup_trained", "p_lo_sup_trained", "p_hi_sup_trained", "gate_class"]
        print(fam[fam["gate_class"] != mc_gate.NONE].sort_values("p_sup_trained")[keep].to_string(index=False))


if __name__ == "__main__":
    main()
