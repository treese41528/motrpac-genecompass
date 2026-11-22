"""Hotspot calls that more permutations could not reverse (docs/audit/remediation/STEP3_PREREG.md section 5.2).

The pure parts: Clopper-Pearson limits of a block's ideal p, their intersection over looks, Benjamini-Hochberg,
the three classes, and the sequential driver with a sampler supplied by the test.
"""
import numpy as np
import pytest

import mc_gate as mg


# ------------------------------------------------------------------ interval limits
def test_look_error_spends_eps_over_family_and_looks():
    assert mg.look_error(0.01, 151, 100) == pytest.approx(0.01 / 15100)
    with pytest.raises(ValueError):
        mg.look_error(0.01, 0, 100)


def test_clopper_pearson_hand_examples():
    # no exceedance in m: lo = 0 and hi solves (1 - hi)^m = delta / 2; all m exceed: the mirror image
    lo, hi = mg.clopper_pearson(0, 10, 0.05)
    assert lo == 0 and hi == pytest.approx(1 - 0.025 ** 0.1)
    lo, hi = mg.clopper_pearson(10, 10, 0.05)
    assert lo == pytest.approx(0.025 ** 0.1) and hi == 1
    # the textbook 95% interval for 5 of 10
    lo, hi = mg.clopper_pearson(5, 10, 0.05)
    assert (lo, hi) == pytest.approx((0.187086, 0.812914), abs=1e-6)
    # the registered setting: no exceedance in 10,000 at delta = 0.01 / (151 x 100)
    delta = mg.look_error(mg.EPS, 151, mg.LOOKS)
    assert mg.clopper_pearson(0, 10_000, delta)[1] == pytest.approx(1 - (delta / 2) ** 1e-4)


def test_limits_are_monotone_in_the_count_and_tighten_with_more_permutations_and_larger_error():
    b = np.arange(0, 201)
    lo, hi = mg.clopper_pearson(b, 200, 1e-4)
    assert np.all(np.diff(lo) > 0) and np.all(np.diff(hi) > 0)
    assert np.all(lo <= b / 200) and np.all(b / 200 <= hi)
    lo1, hi1 = mg.clopper_pearson(30, 10_000, 1e-6)
    lo2, hi2 = mg.clopper_pearson(300, 100_000, 1e-6)                    # same proportion, 10 times the draws
    assert lo1 < lo2 < 0.003 < hi2 < hi1
    lo3, hi3 = mg.clopper_pearson(30, 10_000, 1e-2)                      # larger error: narrower
    assert lo1 < lo3 and hi3 < hi1


def test_half_width_at_the_cap_is_about_three_in_ten_thousand():
    # section 5.2: "At p near 0.0037 the interval's half-width at the cap is about 0.0003"
    lo, hi = mg.clopper_pearson(3700, 1_000_000, mg.look_error(mg.EPS, 151, mg.LOOKS))
    assert (hi - lo) / 2 == pytest.approx(0.0003, abs=0.0001)


def test_bad_counts_are_refused():
    with pytest.raises(ValueError):
        mg.clopper_pearson(11, 10, 0.05)


def test_intersection_keeps_the_tighter_limit_on_each_side():
    lo, hi = mg.intersect(np.array([0.1, 0.2]), np.array([0.5, 0.6]), np.array([0.15, 0.1]), np.array([0.7, 0.4]))
    assert lo.tolist() == [0.15, 0.2] and hi.tolist() == [0.5, 0.4]


# ------------------------------------------------------------------ Benjamini-Hochberg
def _bh_reference(p):
    """The formula corroborate_summary.py carried inline before Step 3."""
    p = np.asarray(p, float); o = np.argsort(p); n = len(p)
    q = (p[o] * n) / np.arange(1, n + 1); q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n); out[o] = np.clip(q, 0, 1)
    return out


def test_bh_hand_example_and_the_previous_formula():
    assert mg.bh_adjust([0.01, 0.04, 0.03, 0.005]) == pytest.approx([0.02, 0.04, 0.04, 0.02])
    rng = np.random.default_rng(0)
    p = np.round(rng.uniform(size=200) ** 3, 4)                           # rounded, so with ties
    assert np.array_equal(mg.bh_adjust(p), _bh_reference(p))
    assert mg.bh_adjust([]).size == 0
    with pytest.raises(ValueError):
        mg.bh_adjust([0.1, np.nan])


def test_rejection_is_strict_as_the_table_readers_apply_it():
    assert mg.bh_reject([0.05, 0.05], 0.05).tolist() == [False, False]
    assert mg.bh_reject([0.0499, 0.0499], 0.05).tolist() == [True, True]


def test_lowering_p_values_never_raises_an_adjusted_p():
    rng = np.random.default_rng(1)
    for _ in range(200):
        p = rng.uniform(size=30) ** 2
        lower = p * rng.uniform(size=30)
        assert np.all(mg.bh_adjust(lower) <= mg.bh_adjust(p) + 1e-15)


# ------------------------------------------------------------------ classes
def test_classes_hand_example():
    lo = np.array([0.0, 0.02, 0.005, 0.5])
    hi = np.array([0.001, 0.2, 0.06, 0.9])
    # at the upper limits BH gives q = 0.004, 0.267, 0.12, 0.9: only the first block is rejected;
    # at the lower limits q = 0, 0.0267, 0.01, 0.5: the first three are
    assert mg.classify(lo, hi, 0.05).tolist() == ["hotspot", "borderline", "borderline", "none"]


def test_any_p_values_inside_the_intervals_agree_with_the_decided_classes():
    rng = np.random.default_rng(2)
    for _ in range(300):
        lo = rng.uniform(size=25) ** 4 * 0.2
        hi = lo + rng.uniform(size=25) ** 4 * 0.2
        cls = mg.classify(lo, hi, 0.05)
        rejected = mg.bh_reject(lo + (hi - lo) * rng.uniform(size=25), 0.05)
        assert np.all(rejected[cls == "hotspot"])
        assert not np.any(rejected[cls == "none"])


# ------------------------------------------------------------------ sequential sampling
class Sampler:
    """Deterministic stand-in for the permutation sampler: block i gives round(rate_i x n) exceedances per call."""

    def __init__(self, rates):
        self.rates = np.asarray(rates, float)
        self.calls = []

    def first(self, n):
        return np.round(self.rates * n).astype(int), np.full(len(self.rates), n)

    def __call__(self, index, n):
        self.calls.append((list(index), n))
        return np.round(self.rates[index] * n).astype(int)


def test_a_block_at_the_cut_stays_borderline_at_the_cap_and_decided_blocks_stop():
    # two blocks at level 0.05: the larger p is rejected only if it is below 0.05, and block 1 sits exactly on it
    s = Sampler([0.0, 0.05])
    b, m = s.first(100)
    res = mg.sequential_gate(b, m, s, eps=0.01, alpha=0.05, batch=100, cap=1000, looks=10)
    assert res["cls"].tolist() == ["hotspot", "borderline"]
    assert res["m"].tolist() == [400, 1000] and res["b"].tolist() == [0, 50]
    assert res["looks_used"] == 10 and res["n_empty"] == 0
    assert res["lo"][1] < 0.05 < res["hi"][1]
    # block 0 is decided at the fourth look (its upper limit falls below 0.025) and is not sampled again
    assert [c[0] for c in s.calls] == [[0, 1]] * 3 + [[1]] * 6
    assert res["delta"] == pytest.approx(0.01 / (2 * 10))


def test_clear_blocks_are_decided_at_the_first_look():
    s = Sampler([0.0, 0.0, 0.6, 0.4])
    b, m = s.first(10_000)
    res = mg.sequential_gate(b, m, s)                                    # the registered parameters
    assert res["cls"].tolist() == ["hotspot", "hotspot", "none", "none"]
    assert res["looks_used"] == 1 and s.calls == [] and res["m"].tolist() == [10_000] * 4


def test_a_decided_block_never_changes_class_while_others_are_sampled():
    rng = np.random.default_rng(3)
    ideal = np.array([0.0005, 0.004, 0.0045, 0.006, 0.3, 0.7, 0.2, 0.9, 0.5, 0.05])
    history = []

    def more(index, n):
        return rng.binomial(n, ideal[index])

    res = mg.sequential_gate(rng.binomial(2000, ideal), np.full(10, 2000), more, batch=2000, cap=60_000, looks=30,
                             log=lambda line: history.append(line))
    assert len(history) == res["looks_used"]
    assert set(res["cls"]) <= {"hotspot", "borderline", "none"}
    assert np.all(res["m"][res["cls"] == "borderline"] == 60_000)        # undecided blocks were sampled to the cap
    assert np.all(res["lo"] <= res["hi"])


def test_the_looks_must_cover_the_cap():
    with pytest.raises(ValueError, match="looks"):
        mg.sequential_gate([1, 2], [100, 100], lambda i, n: [0] * len(i), batch=100, cap=5000, looks=10)
    with pytest.raises(ValueError, match="same number"):
        mg.sequential_gate([1, 2], [100, 200], lambda i, n: [0] * len(i))
    assert mg.looks_needed(10_000, 10_000, 1_000_000) == mg.LOOKS


def test_an_empty_family_is_not_an_error():
    res = mg.sequential_gate([], [], lambda i, n: [])
    assert res["looks_used"] == 0 and res["cls"].size == 0
