"""Stage 10 exercise gate: the eta^2 measures and the cross-validated supervised probe.

The canonical gate (pre-registration section 6) is 10 stratified 5-fold splits x 10,000
permutations; these tests use small sizes but the same code path.
"""
import zlib

import numpy as np
import pytest

import subspace_probe as sp_


def test_eta2_is_one_for_separated_groups_and_zero_for_equal_means():
    labels = np.array([0, 0, 1, 1])
    E = np.array([[0.0, 1.0], [0.0, 1.0], [5.0, 3.0], [5.0, 3.0]])
    assert sp_.trace_eta2(E, labels) == pytest.approx(1.0)
    E2 = np.array([[1.0, 0.0], [-1.0, 0.0], [1.0, 2.0], [-1.0, -2.0]])
    assert sp_.trace_eta2(E2, labels) == pytest.approx(0.0)
    assert np.isnan(sp_.trace_eta2(np.ones((4, 2)), labels))


def test_per_dimension_eta2_matches_the_one_dimensional_version():
    rng = np.random.default_rng(1)
    E = rng.normal(size=(30, 4)); labels = rng.integers(0, 3, 30)
    per = sp_.perdim_eta2(E, labels)
    assert per == pytest.approx([sp_.cat_eta2_1d(E[:, j], labels) for j in range(4)])


def test_block_seed_is_stable_across_interpreters():
    # builtin hash() of a str is salted per process; the seed must not be
    assert sp_.block_seed("liver", 3) == 1000 * (zlib.crc32(b"liver") % 9973) + 3


def test_fold_sets_are_stratified_partitions_and_reproducible():
    y = np.array([0] * 10 + [1] * 15)
    sets = sp_.make_fold_sets(y, 5, seed=42, repeats=3)
    assert len(sets) == 3
    for folds in sets:
        test_idx = np.concatenate([te for _, te in folds])
        assert sorted(test_idx) == list(range(25))                     # each sample tested once
        assert all(np.bincount(y[te], minlength=2).min() >= 2 for _, te in folds)
    again = sp_.make_fold_sets(y, 5, seed=42, repeats=3)
    assert all(np.array_equal(a[1], b[1]) for s1, s2 in zip(sets, again) for a, b in zip(s1, s2))
    first = sp_.make_folds(y, 5, np.random.default_rng(42))           # split 0 = the single split
    assert all(np.array_equal(a[1], b[1]) for a, b in zip(sets[0], first))


def test_fold_count_shrinks_to_the_smallest_class_and_singletons_give_none():
    y = np.array([0, 0, 0, 1, 1, 1, 1, 1])
    assert len(sp_.make_fold_sets(y, 5, seed=0, repeats=1)[0]) == 3
    assert sp_.make_fold_sets(np.array([0, 1, 1, 1]), 5, seed=0, repeats=2) is None


def _blocks(shift, n=40, p=6, seed=0):
    rng = np.random.default_rng(seed)
    y = np.array([0, 1] * (n // 2))
    E = rng.normal(size=(n, p)); E[:, 0] += shift * y
    return E, y


def test_probe_detects_a_real_axis_with_the_smallest_possible_p():
    E, y = _blocks(shift=4.0)
    folds = sp_.make_fold_sets(y, 5, seed=7, repeats=2)
    auc, eta2, p = sp_.supervised(E, folds, y.astype(float), "auc", y, perms=99,
                                  rng=np.random.default_rng(0))
    assert auc > 0.95 and eta2 > 0.5
    assert p == pytest.approx(1 / 100)


def test_probe_finds_nothing_in_noise():
    E, y = _blocks(shift=0.0, seed=3)
    folds = sp_.make_fold_sets(y, 5, seed=7, repeats=2)
    auc, _, p = sp_.supervised(E, folds, y.astype(float), "auc", y, perms=99,
                               rng=np.random.default_rng(0))
    assert p > 0.05 and auc < 0.8


def test_out_of_fold_scores_never_use_the_scored_sample():
    E, y = _blocks(shift=0.0, seed=5)
    folds = sp_.make_folds(y, 5, np.random.default_rng(1))
    base = sp_.oof_scores(E, y.astype(float), folds)
    # changing one sample's label changes only the OTHER folds' scores, never its own
    _, te0 = folds[0]
    y2 = y.astype(float).copy(); y2[te0[0]] = 1 - y2[te0[0]]
    moved = sp_.oof_scores(E, y2, folds)
    assert np.allclose(moved[te0], base[te0])
