"""Stage 14 purity sweeps and the expression (Z) scoring: the pure logic of make_purity_sweep.py,
score_purity_sweep.py and score_z.py.

The audit (cross-stage "tests") found no unit test for these scripts. What is checked here:
  * a sweep mixture is exactly the sum of the cells drawn for it, and its true per-type expression and RNA
    fractions are computed from those same cells;
  * the focal type's cell count follows the purity grid, including the ends of the grid and a pool smaller
    than a mixture;
  * the summary bins scores the way the reports quote them (per target purity; the ">= 50%" regime);
  * the Z scorer aligns genes by name, filters mixtures by the type's true share, and resolves file names
    through the shared contract.
One test pins a defect found while writing these tests (the rounding-drift repair can leave a negative cell
count; docs/audit/remediation/STEP6_HYGIENE.md). It describes today's behaviour so that a repair has to change
it on purpose.
"""
import json
import sys

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import celltype_names as cn
import make_purity_sweep as mps
import score_purity_sweep
import score_z

TYPES = ["Hepatocytes", "Kupffer cells", "Stellate cells"]


def make_pool(n_per_type=(60, 40, 30), n_genes=12, seed=0):
    """A pool of cells: integer counts, one ref_type per cell, in the layout generate_purity_sweep reads."""
    rng = np.random.default_rng(seed)
    labels = np.repeat(TYPES, n_per_type)
    X = rng.poisson(lam=np.linspace(1, 6, n_genes), size=(len(labels), n_genes)).astype(np.int64)
    return ad.AnnData(X=sp.csr_matrix(X), obs=pd.DataFrame({"ref_type": labels},
                                                           index=[f"c{i}" for i in range(len(labels))]))


def sweep(pool=None, grid=(0.1, 0.5, 0.9), reps=3, cells=50, seed=1, alpha=1.0, focal="Hepatocytes"):
    pool = make_pool() if pool is None else pool
    return mps.generate_purity_sweep(pool, TYPES, focal, np.random.default_rng(seed), list(grid), reps, cells, alpha)


# ------------------------------------------------------------------ make_purity_sweep.generate_purity_sweep
def test_one_mixture_per_purity_point_and_replicate():
    counts, truth, design, z = sweep()
    assert counts.shape == (9, 12) and len(truth) == len(design) == 9
    assert list(design["mixture"]) == [f"mix{i}" for i in range(1, 10)] == list(truth["mixture"])
    assert list(design["target_purity"]) == [0.1] * 3 + [0.5] * 3 + [0.9] * 3
    assert list(design["rep"]) == [1, 2, 3] * 3
    assert set(design["focal_type"]) == {"Hepatocytes"}
    assert set(z) == set(TYPES) and all(v.shape == (9, 12) for v in z.values())


def test_a_mixture_is_the_sum_of_its_per_type_true_expression():
    counts, truth, _, z = sweep()
    total = sum(z[t] for t in TYPES)
    assert np.array_equal(counts.toarray(), total.astype(np.int64))


def test_rna_fractions_come_from_the_cells_drawn_and_sum_to_one():
    counts, truth, design, z = sweep()
    lib = counts.toarray().sum(1)
    for t in TYPES:
        assert np.allclose(truth[f"rnafrac__{t}"], z[t].sum(1) / lib)
    assert np.allclose(truth[[f"rnafrac__{t}" for t in TYPES]].sum(1), 1.0)
    assert np.allclose(design["focal_rnafrac"], truth["rnafrac__Hepatocytes"])


def test_focal_cell_fraction_follows_the_grid_and_cell_fractions_sum_to_one():
    _, truth, design, _ = sweep(grid=(0.1, 0.33, 0.9), cells=40)
    # whole cells: 0.1 * 40 = 4, 0.33 * 40 = 13.2 -> 13 (0.325 of the mixture), 0.9 * 40 = 36
    assert list(design["focal_cellfrac"].round(6)) == [0.1] * 3 + [0.325] * 3 + [0.9] * 3
    assert np.allclose(truth["cellfrac__Hepatocytes"], design["focal_cellfrac"])
    assert np.allclose(truth[[f"cellfrac__{t}" for t in TYPES]].sum(1), 1.0)


def test_ends_of_the_grid_give_pure_and_focal_free_mixtures():
    counts, truth, design, z = sweep(grid=(0.0, 1.0), reps=2)
    assert not z["Hepatocytes"][:2].any() and (design["focal_rnafrac"][:2] == 0).all()        # purity 0
    assert not z["Kupffer cells"][2:].any() and not z["Stellate cells"][2:].any()             # purity 1
    assert np.array_equal(counts.toarray()[2:], z["Hepatocytes"][2:].astype(np.int64))
    assert (truth["cellfrac__Hepatocytes"][2:] == 1.0).all()


def test_a_pool_smaller_than_the_mixture_is_sampled_with_replacement():
    pool = make_pool(n_per_type=(5, 40, 30))                      # 5 focal cells, 45 needed at purity 0.9
    counts, truth, design, z = sweep(pool=pool, grid=(0.9,), reps=2, cells=50)
    assert list(design["focal_cellfrac"]) == [0.9, 0.9]
    assert np.array_equal(counts.toarray(), sum(z[t] for t in TYPES).astype(np.int64))


def test_same_seed_same_sweep_and_another_seed_another_sweep():
    a, b, c = sweep(seed=3), sweep(seed=3), sweep(seed=4)
    assert np.array_equal(a[0].toarray(), b[0].toarray()) and a[1].equals(b[1]) and a[2].equals(b[2])
    assert not np.array_equal(a[0].toarray(), c[0].toarray())


def test_focal_type_must_be_usable_and_needs_a_background():
    with pytest.raises(SystemExit, match="focal type"):
        sweep(focal="Cholangiocytes")
    pool = make_pool()
    with pytest.raises(SystemExit, match="background"):
        mps.generate_purity_sweep(pool, ["Hepatocytes"], "Hepatocytes", np.random.default_rng(0), [0.5], 1, 10, 1.0)


class ScriptedRng:
    """A stand-in generator whose draws are fixed, to reach the rounding case deterministically."""

    def __init__(self, shares, drift_index):
        self.shares, self.drift_index = np.asarray(shares, float), drift_index

    def dirichlet(self, alpha):
        return self.shares

    def integers(self, n):
        return self.drift_index

    def choice(self, pool, size, replace):
        return np.resize(pool, size)


def test_known_defect_drift_repair_can_leave_a_negative_cell_count(capsys):
    """Today's behaviour, pinned on purpose. A mixture of 10 cells at purity 0.7 has 3 background cells. Split
    0.5 / 0.5 / 0 over three types they round to 2 + 2 + 0 = 4, one too many, and the drift of -1 is put on
    the type that has none. That type gets -1 cells and a negative cell fraction, is skipped when cells are
    drawn, and the mixture holds 11 cells. Counts, true expression and RNA fractions are still those of the
    cells actually drawn."""
    types = TYPES + ["Cholangiocytes"]
    labels = np.repeat(types, 20)
    X = np.ones((len(labels), 4), dtype=np.int64)                 # every cell: 4 counts
    pool = ad.AnnData(X=sp.csr_matrix(X), obs=pd.DataFrame({"ref_type": labels},
                                                           index=[f"c{i}" for i in range(len(labels))]))
    rng = ScriptedRng(shares=[0.5, 0.5, 0.0], drift_index=2)      # background order: Kupffer, Stellate, Cholangio
    counts, truth, design, z = mps.generate_purity_sweep(pool, types, "Hepatocytes", rng, [0.7], 1, 10, 1.0)
    r = truth.iloc[0]
    assert r["cellfrac__Cholangiocytes"] == pytest.approx(-0.1)               # -1 cell of a nominal 10
    assert r[[f"cellfrac__{t}" for t in types]].sum() == pytest.approx(1.0)   # the books still balance
    assert counts.toarray().sum() == 4 * 11                                   # 7 focal + 2 + 2 = 11 cells drawn
    assert not z["Cholangiocytes"].any()
    assert r[[f"rnafrac__{t}" for t in types]].sum() == pytest.approx(1.0)
    assert r["rnafrac__Hepatocytes"] == pytest.approx(7 / 11)                 # exact, from the cells drawn
    assert design["focal_cellfrac"][0] == pytest.approx(0.7)                  # nominal: 7 of 10, not 7 of 11
    assert "NOTE: 1 of 1 mixtures" in capsys.readouterr().out


def test_no_note_when_no_count_goes_negative(capsys):
    sweep()
    assert "NOTE:" not in capsys.readouterr().out


# ------------------------------------------------------------------ score_purity_sweep.py
@pytest.fixture
def no_git(monkeypatch):
    """The provenance record asks git for the commit; that is slow here and not what is tested."""
    import provenance
    monkeypatch.setattr(provenance, "_git", lambda *a: "")


def write_scores(stage, rows):
    (stage / "scores").mkdir(parents=True)
    pd.DataFrame(rows, columns=["mixture", "target_purity", "focal_rnafrac", "pearson_vst", "spearman_raw",
                                "n_genes"]).to_csv(stage / "scores" / "z_vst_focal.tsv", sep="\t", index=False)


SCORES = [("mix1", 0.3, 0.45, 0.90, 0.80, 100), ("mix2", 0.3, 0.55, 0.92, 0.82, 100),
          ("mix3", 0.3, 0.40, 0.94, 0.84, 100), ("mix4", 0.7, 0.75, 0.97, 0.90, 100),
          ("mix5", 0.7, 0.80, 0.99, 0.92, 100)]


def run_summary(stage, monkeypatch, *extra):
    monkeypatch.setattr(sys, "argv", ["score_purity_sweep.py", "--stage-dir", str(stage), *extra])
    score_purity_sweep.main()
    return pd.read_csv(stage / "scores" / "purity_sweep_summary.tsv", sep="\t")


def test_summary_has_one_row_per_target_purity_with_median_and_quartiles(tmp_path, monkeypatch, no_git):
    write_scores(tmp_path, SCORES)
    s = run_summary(tmp_path, monkeypatch)
    assert list(s["target_purity"]) == [0.3, 0.7] and list(s["n"]) == [3, 2]
    assert s["pearson_vst_median"].tolist() == pytest.approx([0.92, 0.98])
    assert s["pearson_vst_q1"].tolist() == pytest.approx([0.91, 0.975])
    assert s["pearson_vst_q3"].tolist() == pytest.approx([0.93, 0.985])
    assert s["spearman_raw_median"].tolist() == pytest.approx([0.82, 0.91])
    assert s["mean_focal_rnafrac"].tolist() == pytest.approx([(0.45 + 0.55 + 0.40) / 3, 0.775])


def test_the_fifty_percent_regime_is_binned_by_realised_rna_fraction_by_default(tmp_path, monkeypatch, capsys, no_git):
    write_scores(tmp_path, SCORES)
    run_summary(tmp_path, monkeypatch)
    out = capsys.readouterr().out
    # realised RNA fraction >= 0.5: mix2, mix4, mix5 -> median 0.97; below: mix1, mix3 -> median 0.92
    assert ">=50% purity (Fig 1h regime)     n=  3  median Pearson-VST=0.970" in out
    assert "<50% purity                      n=  2  median Pearson-VST=0.920" in out


def test_binning_by_nominal_purity_moves_the_borderline_mixture(tmp_path, monkeypatch, capsys, no_git):
    write_scores(tmp_path, SCORES)
    run_summary(tmp_path, monkeypatch, "--purity-col", "target_purity")
    out = capsys.readouterr().out
    assert ">=50% purity (Fig 1h regime)     n=  2  median Pearson-VST=0.980" in out      # mix4, mix5 only
    assert "<50% purity                      n=  3  median Pearson-VST=0.920" in out


def test_an_empty_bin_is_reported_not_scored(tmp_path, monkeypatch, capsys, no_git):
    write_scores(tmp_path, SCORES[:1] + SCORES[2:3])                    # both below 0.5
    run_summary(tmp_path, monkeypatch)
    assert "n=0  (sweep grid did not reach this bin)" in capsys.readouterr().out


def test_summary_writes_a_provenance_record_with_the_md5_of_its_input(tmp_path, monkeypatch, no_git):
    import hashlib
    write_scores(tmp_path, SCORES)
    run_summary(tmp_path, monkeypatch)
    rec = json.loads((tmp_path / "scores" / "provenance_score_purity_sweep.json").read_text())
    src = tmp_path / "scores" / "z_vst_focal.tsv"
    assert rec["inputs"][0]["md5"] == hashlib.md5(src.read_bytes()).hexdigest()
    assert rec["inputs"][0]["bytes"] == src.stat().st_size and rec["params"]["purity_col"] == "focal_rnafrac"


# ------------------------------------------------------------------ score_z.py
def test_corr_is_one_for_identical_vectors_and_undefined_for_a_constant_one():
    a = np.array([0.0, 1.0, 5.0, 20.0, 3.0])
    assert score_z.corr(a, a) == pytest.approx((1.0, 1.0))
    assert score_z.corr(a, a[::-1].copy())[0] == pytest.approx(-0.3)        # Spearman of the reversed ranks
    assert all(np.isnan(v) for v in score_z.corr(a, np.ones(5)))
    assert all(np.isnan(v) for v in score_z.corr(np.zeros(5), a))


def test_corr_uses_ranks_and_log_counts():
    a = np.array([1.0, 10.0, 100.0, 1000.0])
    sp_, lp = score_z.corr(a, a ** 2)                                       # monotone, far from linear
    assert sp_ == pytest.approx(1.0) and 0.99 < lp < 1.0


Z_TYPES = ["Hepatocytes", "α-cells"]                                        # one name needs the transliteration


def write_stage(stage, true, pred, true_genes, pred_genes):
    zt, zp = stage / "mixtures" / "true_z", stage / "results" / "pred_z"
    zt.mkdir(parents=True); zp.mkdir(parents=True)
    (zt / "genes.txt").write_text("\n".join(true_genes) + "\n")
    (zt / "types.txt").write_text("\n".join(Z_TYPES) + "\n")
    (zp / "genes.txt").write_text("\n".join(pred_genes) + "\n")
    for t in Z_TYPES:
        np.save(zt / f"truez__{cn.safe(t)}.npy", true[t])
        pd.DataFrame(pred[t], columns=pred_genes,
                     index=[f"mix{i + 1}" for i in range(len(pred[t]))]).to_csv(zp / f"predz__{cn.safe(t)}.csv")


def test_z_scoring_aligns_genes_by_name_and_filters_mixtures_by_true_share(tmp_path, monkeypatch, capsys):
    true_genes = ["g1", "g2", "g3", "g4", "g5"]
    pred_genes = ["g5", "g3", "g2", "g1", "gX"]                 # another order, g4 missing, gX not in truth
    hep = np.array([[10, 20, 30, 40, 50], [1, 2, 3, 4, 5], [100, 50, 20, 10, 5]], float)
    alpha = np.array([[5, 4, 3, 2, 1], [0, 0, 0, 0, 0], [50, 60, 70, 80, 90]], float)   # absent from mix2
    idx = [true_genes.index(g) if g in true_genes else None for g in pred_genes]
    as_pred = lambda m: np.column_stack([m[:, i] if i is not None else np.full(len(m), 7.0) for i in idx])  # noqa: E731
    pred = {"Hepatocytes": as_pred(hep), "α-cells": as_pred(alpha[:, ::-1].copy())}     # alpha: reversed profile
    write_stage(tmp_path, {"Hepatocytes": hep, "α-cells": alpha}, pred, true_genes, pred_genes)
    monkeypatch.setattr(sys, "argv", ["score_z.py", "--stage-dir", str(tmp_path)])
    score_z.main()
    m = pd.read_csv(tmp_path / "scores" / "metrics_z.tsv", sep="\t").set_index("cell_type")
    assert "4 common genes (true 5, pred 5)" in capsys.readouterr().out                # g1, g2, g3, g5
    assert set(m["n_genes"]) == {4}
    h = m.loc["Hepatocytes"]
    assert h["agg_spearman"] == pytest.approx(1.0) and h["agg_logpearson"] == pytest.approx(1.0)
    assert h["ps_spearman_median"] == pytest.approx(1.0) and h["n_samples_scored"] == 3
    a = m.loc["α-cells"]
    assert a["n_samples_scored"] == 2                           # mix2 holds none of this type: not scored
    assert a["agg_spearman"] < 0                                # the reversed profile is anti-correlated
    assert list(m.index) == ["Hepatocytes", "α-cells"]          # best aggregated log-Pearson first


def test_z_scoring_min_true_frac_drops_mixtures_where_the_type_is_rare(tmp_path, monkeypatch):
    genes = ["g1", "g2", "g3"]
    hep = np.array([[100, 200, 300], [100, 200, 300]], float)
    alpha = np.array([[1, 2, 3], [100, 200, 300]], float)       # 1% of mix1, 50% of mix2
    write_stage(tmp_path, {"Hepatocytes": hep, "α-cells": alpha}, {"Hepatocytes": hep, "α-cells": alpha}, genes, genes)
    monkeypatch.setattr(sys, "argv", ["score_z.py", "--stage-dir", str(tmp_path), "--min-true-frac", "0.02"])
    score_z.main()
    m = pd.read_csv(tmp_path / "scores" / "metrics_z.tsv", sep="\t").set_index("cell_type")
    assert m.loc["α-cells", "n_samples_scored"] == 1 and m.loc["Hepatocytes", "n_samples_scored"] == 2
    monkeypatch.setattr(sys, "argv", ["score_z.py", "--stage-dir", str(tmp_path), "--min-true-frac", "0"])
    score_z.main()
    m = pd.read_csv(tmp_path / "scores" / "metrics_z.tsv", sep="\t").set_index("cell_type")
    assert m.loc["α-cells", "n_samples_scored"] == 2
