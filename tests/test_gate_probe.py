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


# ------------------------------------------------------------------ kernel scorer (STEP3_PREREG 5.2)
def _random_block(n, p, n1, seed, shift=0.0, ties=False):
    rng = np.random.default_rng(seed)
    y = np.zeros(n); y[:n1] = 1; y = y[rng.permutation(n)]
    E = rng.normal(size=(n, p)); E[:, 0] += shift * y
    if ties:                                   # integer features and duplicated samples: tied scores
        E = np.round(E)
        E[1] = E[0]; E[3] = E[2]; E[7] = E[5]
    return E, y


@pytest.mark.parametrize("n, p, n1, shift, ties, repeats, perms", [
    (50, 768, 40, 0.0, False, 10, 150),        # a full block: 40 trained, 10 control, 768 dimensions
    (50, 20, 40, 1.0, False, 3, 300),          # a real axis: few exceedances
    (24, 8, 19, 0.5, False, 4, 300),           # the OVARY shape: 5 controls
    (20, 768, 10, 0.2, False, 10, 150),        # the S2 shape: 10 against 10
    (25, 5, 20, 0.0, True, 3, 200),            # tied scores in every permutation
    (30, 3, 15, 0.3, True, 2, 200),
])
def test_kernel_scorer_gives_the_loops_exceedance_count(n, p, n1, shift, ties, repeats, perms):
    E, y = _random_block(n, p, n1, seed=n + p, shift=shift, ties=ties)
    folds = sp_.make_fold_sets(y, 5, 11, repeats)
    probe = sp_.KernelProbe(E, y, folds)
    b_kernel, n_loop = probe.count(sp_.draw_unrestricted(y), np.random.default_rng(5), perms)
    b_loop = sp_.loop_exceedances(E, y, folds, "auc", probe.obs, sp_.draw_unrestricted(y),
                                  np.random.default_rng(5), perms)
    assert b_kernel == b_loop
    if ties:
        assert n_loop > 0                      # tied permutations went to roc_auc_score
    # the same generator gives the registered p of supervised()
    auc, _, pval = sp_.supervised(E, folds, y, "auc", y, perms, np.random.default_rng(5))
    assert auc == probe.obs and pval == (b_kernel + 1) / (perms + 1)


def test_kernel_scorer_handles_a_training_fold_with_one_class():
    # 2 controls and 2 folds: a permutation can put both controls in one test fold, so the other training fold
    # has one class, its direction is zero and its scores tie at zero (as in oof_scores)
    E, y = _random_block(12, 6, 10, seed=4)
    folds = sp_.make_fold_sets(y, 5, 3, repeats=3)
    assert len(folds[0]) == 2
    probe = sp_.KernelProbe(E, y, folds)
    b_kernel, n_loop = probe.count(sp_.draw_unrestricted(y), np.random.default_rng(9), 400)
    assert n_loop > 0
    assert b_kernel == sp_.loop_exceedances(E, y, folds, "auc", probe.obs, sp_.draw_unrestricted(y),
                                            np.random.default_rng(9), 400)


def test_kernel_scores_equal_the_out_of_fold_scores():
    E, y = _random_block(40, 30, 30, seed=8, shift=0.5)
    folds = sp_.make_fold_sets(y, 5, 2, repeats=2)
    rng = np.random.default_rng(0)
    Y = np.stack([y[rng.permutation(40)] for _ in range(6)])
    for split, f in zip(sp_.fold_kernels(E, folds), folds):
        S = sp_.kernel_split_scores(split, Y)
        for k in range(6):
            assert np.allclose(S[k], sp_.oof_scores(E, Y[k], f), rtol=1e-10, atol=1e-12)


def test_pair_count_is_the_auc_numerator_and_ties_are_flagged():
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(1)
    S = rng.normal(size=(5, 20)); Y = (rng.uniform(size=(5, 20)) < 0.4).astype(np.int64)
    Y[:, 0], Y[:, 1] = 1, 0
    u, unsafe = sp_.pair_counts(S, Y)
    assert not unsafe.any()
    for k in range(5):
        n1 = Y[k].sum(); n0 = 20 - n1
        assert u[k] / (n1 * n0) == pytest.approx(roc_auc_score(Y[k], S[k]))
        assert sp_.two_u(S[k], Y[k]) == 2 * u[k]
    S[2, 3] = S[2, 7]                                                     # an exact tie
    S[4, 5] = np.nan
    assert sp_.pair_counts(S, Y)[1].tolist() == [False, False, True, False, True]
    # two_u counts a tied pair as one half, as roc_auc_score does
    s = np.array([0.0, 0.0, 1.0, 2.0]); lab = np.array([1, 0, 1, 0])
    assert sp_.two_u(s, lab) / (2 * 2 * 2) == pytest.approx(roc_auc_score(lab, s))


def test_more_permutations_continue_the_same_stream():
    E, y = _random_block(30, 10, 22, seed=6)
    folds = sp_.make_fold_sets(y, 5, 1, repeats=2)
    probe, draw = sp_.KernelProbe(E, y, folds), sp_.draw_unrestricted(y)
    rng = np.random.default_rng(12)
    first, _ = probe.count(draw, rng, 70)
    state = rng.bit_generator.state                                       # what process() hands on
    again = np.random.default_rng(); again.bit_generator.state = state
    second, _ = probe.count(draw, again, 130)
    whole = sp_.loop_exceedances(E, y, folds, "auc", probe.obs, draw, np.random.default_rng(12), 200)
    assert first + second == whole
    assert probe.count(draw, np.random.default_rng(12), 200, chunk=37)[0] == whole    # batching does not matter


def _item(n_per=5, p=12, seed=0, tissue="liver", ci=2, sexes=("female", "male")):
    """A block with the MoTrPAC design: n_per animals per sex and group."""
    rng = np.random.default_rng(seed)
    group = np.array([g for g in ("control", "1w", "2w", "4w", "8w") for _ in sexes for _ in range(n_per)], object)
    sex = np.array([s for _ in range(5) for s in sexes for _ in range(n_per)], object)
    trained = np.array(["control" if g == "control" else "trained" for g in group], object)
    E = rng.normal(size=(len(group), p)) + 0.4 * (trained == "trained")[:, None]
    week = np.array([sp_.WEEK[g] for g in group], float)
    return dict(tissue=tissue, cell_type="Hepatocytes", E=E, group=group, trained=trained, sex=sex, week=week,
                seed=sp_.block_seed(tissue, ci))


def test_process_reports_counts_and_the_registered_p_and_hands_on_the_stream():
    item = _item()
    row, state = sp_.process(item, perms=60, folds_k=5, repeats=2)
    assert row["m_sup_trained"] == 60
    assert row["p_sup_trained"] == (row["b_sup_trained"] + 1) / 61
    assert row["p_sup_trained_10k"] == round(row["p_sup_trained"], 4)
    # the first batch is the registered stream: the loop, generator seed + 2, folds seed + 1
    y = (item["trained"] == "trained").astype(float)
    folds = sp_.make_fold_sets(item["trained"], 5, item["seed"] + 1, 2)
    auc, eta2, p = sp_.supervised(item["E"], folds, y, "auc", item["trained"], 60,
                                  np.random.default_rng(item["seed"] + 2))
    assert (row["sup_trained_auc"], row["sup_trained_eta2"]) == (round(auc, 3), round(eta2, 3))
    assert row["p_sup_trained"] == p
    # more permutations continue it; the loop scorer gives the same row
    more, _ = sp_.more_trained_perms(item, 5, 2, state, 40, "kernel")
    _, _, p100 = sp_.supervised(item["E"], folds, y, "auc", item["trained"], 100,
                                np.random.default_rng(item["seed"] + 2))
    assert (row["b_sup_trained"] + more + 1) / 101 == p100
    row_loop, _ = sp_.process(item, perms=60, folds_k=5, repeats=2, scorer="loop")
    assert row_loop == row


# ------------------------------------------------------------------ sensitivities S1, S2 (STEP3_PREREG 5.3)
def test_s1_permutations_keep_each_sexs_number_of_trained_labels():
    item = _item()
    y = (item["trained"] == "trained").astype(float)
    groups = [np.flatnonzero(item["sex"] == s) for s in ("female", "male")]
    draw, rng = sp_.draw_within(y, groups), np.random.default_rng(0)
    seen = set()
    for _ in range(200):
        yp = draw(rng)
        assert [yp[g].sum() for g in groups] == [y[g].sum() for g in groups] == [20, 20]
        seen.add(tuple(yp))
    assert len(seen) > 150                                                # and they do permute
    # an unrestricted draw does not keep the balance
    free = sp_.draw_unrestricted(y)
    assert any(free(rng)[groups[0]].sum() != 20 for _ in range(50))


def test_s1_folds_are_stratified_on_trained_by_sex_and_scored_like_the_loop():
    item = _item()
    probe, draw = sp_.s1_probe(item, 5, 3, "kernel")
    strata = np.array([f"{t}|{s}" for t, s in zip(item["trained"], item["sex"])])
    for folds in probe.fold_sets:
        for _, te in folds:                                               # 1 control and 4 trained of each sex
            assert sorted(np.unique(strata[te], return_counts=True)[1].tolist()) == [1, 1, 4, 4]
    assert probe.fold_sets[0][0][1].tolist() != \
        sp_.make_fold_sets(item["trained"], 5, item["seed"] + 1, 3)[0][0][1].tolist()   # its own splits
    b, _ = probe.count(draw, np.random.default_rng(item["seed"] + sp_.S1_PERM_SEED), 150)
    assert b == sp_.loop_exceedances(item["E"], probe.y, probe.fold_sets, "auc", probe.obs, draw,
                                     np.random.default_rng(item["seed"] + sp_.S1_PERM_SEED), 150)


def test_s2_is_eight_weeks_against_control_only():
    item = _item()
    probe, _, n = sp_.s2_probe(item, 5, 2, "kernel")
    assert n == 20 and probe.E.shape[0] == 20 and probe.y.sum() == 10
    keep = np.isin(item["group"].astype(str), ["8w", "control"])
    assert np.array_equal(probe.E, item["E"][keep])
    assert np.array_equal(probe.y, (item["group"][keep] == "8w").astype(float))
    row, _ = sp_.process(item, perms=40, folds_k=5, repeats=2)
    assert row["n_s2"] == 20 and 0 < row["p_sup_trained_s2"] <= 1 and 0 < row["p_sup_trained_s1"] <= 1
    assert row["sup_trained_auc_s2"] == round(probe.obs, 3)


def test_single_sex_block_has_s1_and_no_sex_probe():
    row, _ = sp_.process(_item(sexes=("female",)), perms=30, folds_k=5, repeats=1)
    assert "p_sup_trained_s1" in row and "sup_sex_auc" not in row


def test_sensitivity_seeds_do_not_reuse_a_registered_stream():
    offsets = {sp_.S1_FOLD_SEED, sp_.S1_PERM_SEED, sp_.S2_FOLD_SEED, sp_.S2_PERM_SEED}
    assert len(offsets) == 4 and min(offsets) > 13 + 100                 # clear of 0-6, 7, 11, 13 of any neighbour


# ------------------------------------------------------------------ equivalence check helpers
@pytest.mark.parametrize("p4, b", [(0.3013, 3012), (0.0002, 1), (0.0001, 0), (1.0, 10000), (0.4999, 4998),
                                   (0.5001, 5001)])
def test_stored_p_identifies_the_exceedance_count(p4, b):
    assert round((b + 1) / 10001, 4) == p4
    assert sp_.implied_count(p4, 10000) == b


def test_stored_p_of_one_half_is_shared_by_two_counts():
    assert round(5000 / 10001, 4) == round(5001 / 10001, 4) == 0.5
    assert np.isnan(sp_.implied_count(0.5, 10000)) and np.isnan(sp_.implied_count(np.nan, 10000))
