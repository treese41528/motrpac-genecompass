"""mc_gate.py -- a hotspot is a call that more permutations could not reverse.

Registered in docs/audit/remediation/STEP3_PREREG.md section 5.2 (repair of audit finding 6.1). The gate's
p-value is a Monte Carlo estimate: b of m random label permutations gave a statistic at or above the observed
one. With m = 10,000 a p near the Benjamini-Hochberg cut has a standard error of about 0.0006, and the hotspot
count depended on the random draw. This module implements the multiple-Monte-Carlo-test construction of Gandy
and Hahn (Scand J Stat 2014;41:1083-1101):

  1. every block's ideal p (the value an infinite number of permutations would give) gets a Clopper-Pearson
     interval [lo, hi] from its b and m, at a level that makes all intervals of the family hold together with
     probability at least 1 - eps;
  2. Benjamini-Hochberg is monotone, so a block rejected when every p is set to its upper limit is rejected
     for the ideal p-values too, and a block not rejected when every p is set to its lower limit is not
     rejected for the ideal p-values either;
  3. blocks in between get more permutations, in batches, up to a cap.

Nothing here reads a file or draws a random number. The permutations come from the caller
(deconvolution/subspace_probe.py); the family of blocks comes from the analysable rule in
deconvolution/corroborate_summary.py.
"""
import math

import numpy as np
from scipy.stats import beta

# The registered parameters (section 5.2, "Parameters, fixed now").
EPS = 0.01          # probability that any interval of the family fails at any look
ALPHA = 0.05        # Benjamini-Hochberg level; a block is rejected when its adjusted p is below it
BATCH = 10_000      # permutations per look
CAP = 1_000_000     # permutations per block at which sampling stops
LOOKS = 100         # looks the error is spent over: CAP / BATCH

HOTSPOT, BORDERLINE, NONE = "hotspot", "borderline", "none"


def look_error(eps, n_family, looks):
    """Two-sided error of one block's interval at one look: eps / (n_family x looks).

    The error is spent equally over a fixed number of looks. By the union bound all n_family x looks intervals
    hold together with probability at least 1 - eps.
    """
    if n_family < 1 or looks < 1:
        raise ValueError("the family and the number of looks must both be at least 1")
    return eps / (n_family * looks)


def clopper_pearson(b, m, delta):
    """Clopper-Pearson limits (lo, hi) for a binomial proportion from b successes in m trials.

    Here the proportion is a block's ideal p-value and b is the number of permuted statistics at or above the
    observed one. Each tail has error delta / 2. lo is 0 when b = 0 and hi is 1 when b = m.
    """
    b = np.asarray(b, dtype=float)
    m = np.asarray(m, dtype=float)
    if np.any(m < 1) or np.any(b < 0) or np.any(b > m):
        raise ValueError("need 0 <= b <= m and m >= 1")
    with np.errstate(invalid="ignore"):
        lo = np.where(b > 0, beta.ppf(delta / 2, b, m - b + 1), 0.0)
        hi = np.where(b < m, beta.isf(delta / 2, b + 1, m - b), 1.0)
    return lo, hi


def intersect(lo, hi, lo_new, hi_new):
    """A block's interval over several looks is the intersection of its intervals at each look."""
    return np.maximum(lo, lo_new), np.minimum(hi, hi_new)


def bh_adjust(p):
    """Benjamini-Hochberg adjusted p-values (step-up): sort, q_(i) = p_(i) n / i, running minimum from the
    largest rank down, clipped to [0, 1]. p must hold no missing value. This is the formula
    corroborate_summary.py has always applied."""
    p = np.asarray(p, dtype=float)
    if p.size == 0:
        return p.copy()
    if np.isnan(p).any():
        raise ValueError("bh_adjust needs p-values without missing entries")
    n = p.size
    order = np.argsort(p, kind="stable")
    q_sorted = np.minimum.accumulate((p[order] * n / np.arange(1, n + 1))[::-1])[::-1]
    q = np.empty(n)
    q[order] = np.clip(q_sorted, 0, 1)
    return q


def bh_reject(p, alpha=ALPHA):
    """Blocks rejected by Benjamini-Hochberg: adjusted p strictly below alpha. The strict cut is the one every
    reader of the gate table has applied (q_sup_trained < 0.05)."""
    return bh_adjust(p) < alpha


def classify(lo, hi, alpha=ALPHA):
    """Class of every block of the family from the limits of its ideal p (section 5.2, "Classes").

      hotspot     rejected with every p at its upper limit;
      borderline  rejected with every p at its lower limit, not at the upper;
      none        not rejected with every p at its lower limit.

    lo and hi cover the whole family, because the Benjamini-Hochberg cut depends on all of it.
    """
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    at_upper = bh_reject(hi, alpha)
    at_lower = bh_reject(lo, alpha)
    return np.where(at_upper, HOTSPOT, np.where(at_lower, BORDERLINE, NONE)).astype(object)


def looks_needed(first, batch, cap):
    """Number of looks when the first look is after `first` permutations and each later one adds `batch`,
    until `cap`."""
    return 1 + max(0, math.ceil((cap - first) / batch))


def sequential_gate(b, m, sample_more, eps=EPS, alpha=ALPHA, batch=BATCH, cap=CAP, looks=LOOKS, log=None):
    """Sample until every block of the family is a hotspot or not a hotspot, or the cap is reached.

    b, m         exceedance counts and numbers of permutations of the family after the first batch. Every block
                 starts with the same m.
    sample_more  sample_more(index, n) draws n further permutations for each family position in `index` and
                 returns their exceedance counts, in the order of `index`. It must continue each block's own
                 random stream.
    Blocks that are borderline at the current intervals get `batch` more permutations each round; the others
    stop. A block still borderline at `cap` stays borderline (section 5.2). A decided block cannot change
    class later: its interval no longer moves and the other blocks' intervals only shrink, which can only add
    rejections at the upper limits and remove them at the lower limits.

    Returns a dict: b, m (final), lo, hi (intersection over the block's looks), cls, delta (the error of one
    interval), looks_used, n_empty (blocks whose intersection is empty, which has probability below eps).
    """
    b = np.array(b, dtype=np.int64)
    m = np.array(m, dtype=np.int64)
    n_family = b.size
    if n_family == 0:
        return dict(b=b, m=m, lo=np.array([]), hi=np.array([]), cls=np.array([], dtype=object),
                    delta=float("nan"), looks_used=0, n_empty=0)
    if len(set(m.tolist())) != 1:
        raise ValueError("every block of the family must start with the same number of permutations")
    if looks_needed(int(m[0]), batch, cap) > looks:
        raise ValueError(f"{looks} looks cannot cover a first look at {int(m[0])} permutations, batches of "
                         f"{batch} and a cap of {cap}")
    delta = look_error(eps, n_family, looks)
    lo, hi = clopper_pearson(b, m, delta)
    cls = classify(lo, hi, alpha)
    look = 1
    while True:
        undecided = np.flatnonzero((cls == BORDERLINE) & (m < cap))
        if log:
            counts = {c: int((cls == c).sum()) for c in (HOTSPOT, BORDERLINE, NONE)}
            log(f"look {look}: {counts}; {len(undecided)} block(s) get more permutations")
        if undecided.size == 0 or look >= looks:
            break
        n_more = int(min(batch, cap - m[undecided].max()))
        added = np.asarray(sample_more(undecided, n_more), dtype=np.int64)
        if added.shape != undecided.shape or np.any(added < 0) or np.any(added > n_more):
            raise ValueError("sample_more must return one count between 0 and n for every requested block")
        b[undecided] += added
        m[undecided] += n_more
        look += 1
        lo_new, hi_new = clopper_pearson(b[undecided], m[undecided], delta)
        lo[undecided], hi[undecided] = intersect(lo[undecided], hi[undecided], lo_new, hi_new)
        cls = classify(lo, hi, alpha)
    return dict(b=b, m=m, lo=lo, hi=hi, cls=cls, delta=delta, looks_used=look, n_empty=int((lo > hi).sum()))
