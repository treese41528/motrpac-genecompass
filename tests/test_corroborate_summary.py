"""The gate table: family, classes, sensitivities and the reading rule
(docs/audit/remediation/STEP3_PREREG.md sections 5.2 to 5.4).
"""
import json

import numpy as np
import pandas as pd
import pytest

import corroborate_summary as cs
import mc_gate as mg
from celltype_names import safe


def _blocks(**cols):
    """A merged table of six blocks of one tissue; defaults give five analysable blocks and one that is not."""
    n = 6
    base = dict(
        tissue=["liver"] * n, cell_type=[f"type {i}" for i in range(n)], n=[50] * n,
        sup_trained_auc=[0.90, 0.85, 0.80, 0.60, 0.55, 0.88],
        # block 1 sits on the cut: with 5 blocks its p must be below 0.05 x 3 / 5 = 0.03, and its interval
        # after 1,000,000 permutations still straddles that value
        p_sup_trained=[0.0001, 0.0292, 0.0098, 0.30, 0.60, 0.0002],
        p_sup_trained_10k=[0.0001, 0.0285, 0.0099, 0.30, 0.60, 0.0002],
        b_sup_trained=[0, 29199, 9799, 2999, 5999, 1], m_sup_trained=[10000, 10 ** 6, 10 ** 6, 10000, 10000, 10000],
        in_family=[True, True, True, True, True, False],
        p_lo_sup_trained=[0.0, 0.0282, 0.0093, 0.27, 0.57, np.nan],
        p_hi_sup_trained=[0.0015, 0.0302, 0.0104, 0.33, 0.63, np.nan],
        sup_trained_auc_s1=[0.89, 0.84, 0.70, 0.60, 0.50, 0.80], p_sup_trained_s1=[0.0002, 0.004, 0.2, 0.3, 0.7, 0.001],
        n_s2=[20] * n, sup_trained_auc_s2=[0.8, 0.7, 0.6, 0.5, 0.5, 0.9], p_sup_trained_s2=[0.01, 0.1, 0.2, 0.5, 0.5, 0.01],
        auc_embed=[0.90, 0.85, 0.80, 0.60, 0.55, 0.88], auc_pca=[0.8, 0.7, 0.6, 0.5, 0.5, 0.6],
        auc_genes=[0.85, 0.50, 0.60, 0.5, 0.5, 0.9],
        augur_embed_trained=[0.88, 0.80, 0.75, 0.58, 0.56, 0.7], augur_pca_trained=[0.8, 0.7, 0.7, 0.5, 0.5, 0.6],
        median_libsize=[5e4, 2e3, 1e3, 4e5, 1e6, 12.0], mean_fraction=[0.30, 0.004, 0.02, 0.5, 0.17, 0.0001])
    base.update(cols)
    return pd.DataFrame(base)


def test_analysable_rule_and_its_threshold():
    assert cs.MIN_MEDIAN_LIBSIZE == 1000.0
    assert cs.is_analysable(1000.0) and not cs.is_analysable(999.9)
    assert cs.is_analysable(pd.Series([10.0, 1e4]), 100).tolist() == [False, True]


def test_bh_over_adjusts_inside_the_mask_only_and_matches_the_previous_formula():
    rng = np.random.default_rng(0)
    p = pd.Series(np.round(rng.uniform(size=40), 4)); p[3] = np.nan
    mask = pd.Series(rng.uniform(size=40) < 0.7)
    q = cs.bh_over(p, mask)
    sel = p[mask].dropna()
    o = np.argsort(sel.values); ml = len(sel)                             # the inline code before Step 3
    ref = (sel.values[o] * ml) / np.arange(1, ml + 1); ref = np.minimum.accumulate(ref[::-1])[::-1]
    want = np.empty(ml); want[o] = np.clip(ref, 0, 1)
    assert np.array_equal(q[sel.index].values, want)
    assert q[~mask].isna().all() and np.isnan(q[3])


def test_classes_come_from_the_interval_limits_over_the_family():
    m, method = cs.add_gate_columns(_blocks())
    assert method == "mmctest" and (m["gate_method"] == "mmctest").all()
    assert m["analysable"].tolist() == [True] * 5 + [False]
    fam = m["analysable"]
    want = mg.classify(m.loc[fam, "p_lo_sup_trained"], m.loc[fam, "p_hi_sup_trained"], 0.05)
    assert m.loc[fam, "gate_class"].tolist() == want.tolist() == ["hotspot", "borderline", "hotspot", "none", "none"]
    # the block under the library rule is never a hotspot, whatever its p
    assert m.loc[5, "gate_class"] == "none" and not m.loc[5, "is_hotspot"] and np.isnan(m.loc[5, "q_sup_trained"])
    assert m["is_hotspot"].tolist() == [True, False, True, False, False, False]
    # q from the final p, and beside it q from the registered 10,000-permutation p
    assert m.loc[fam, "q_sup_trained"].tolist() == pytest.approx(mg.bh_adjust(m.loc[fam, "p_sup_trained"]))
    assert m.loc[fam, "q_sup_trained_10k"].tolist() == pytest.approx(mg.bh_adjust(m.loc[fam, "p_sup_trained_10k"]))
    # a borderline block is not a hotspot even though its point estimate passes BH
    assert m.loc[1, "q_sup_trained"] < 0.05 and m.loc[1, "gate_class"] == "borderline"


def test_a_family_other_than_the_one_sampled_is_an_error():
    with pytest.raises(SystemExit, match="different family"):
        cs.add_gate_columns(_blocks(in_family=[True] * 6))
    with pytest.raises(SystemExit, match="without interval limits"):
        cs.add_gate_columns(_blocks(p_lo_sup_trained=[0.0, np.nan, 0.0093, 0.27, 0.57, np.nan]))
    with pytest.raises(SystemExit, match="differ from the classes the probe"):
        cs.add_gate_columns(_blocks(gate_class=["hotspot", "hotspot", "hotspot", "none", "none", np.nan]))


def test_library_rule_sensitivities_use_the_registered_p_and_plain_bh():
    for min_lib, n_family in ((100, 5), (0, 6)):
        m, method = cs.add_gate_columns(_blocks(), min_libsize=min_lib)
        assert method == "bh_10k" and int(m["analysable"].sum()) == n_family
        fam = m["analysable"]
        q = mg.bh_adjust(m.loc[fam, "p_sup_trained_10k"])
        assert m.loc[fam, "q_sup_trained"].tolist() == pytest.approx(q)
        assert m.loc[fam, "is_hotspot"].tolist() == (q < 0.05).tolist()
        assert set(m["gate_class"]) <= {"hotspot", "none"}                # no borderline class under plain BH
    assert m.loc[5, "is_hotspot"]                                         # no threshold: the low-library block counts


def test_a_probe_table_from_before_step3_gets_plain_bh_on_its_p():
    old = _blocks().drop(columns=["p_sup_trained_10k", "b_sup_trained", "m_sup_trained", "in_family",
                                  "p_lo_sup_trained", "p_hi_sup_trained", "sup_trained_auc_s1",
                                  "p_sup_trained_s1", "n_s2", "sup_trained_auc_s2", "p_sup_trained_s2"])
    m, method = cs.add_gate_columns(old)
    assert method == "bh_10k" and "q_sup_trained_s1" not in m
    fam = m["analysable"]
    assert m.loc[fam, "q_sup_trained"].tolist() == pytest.approx(mg.bh_adjust(old.loc[fam, "p_sup_trained"]))
    assert (m["support"][m["is_hotspot"]] != "supported").all()          # no S1, so nothing is "supported"


def test_sensitivity_q_values_have_their_own_families():
    m, _ = cs.add_gate_columns(_blocks())
    fam = m["analysable"]
    for s in ("s1", "s2"):
        assert m.loc[fam, f"q_sup_trained_{s}"].tolist() == pytest.approx(mg.bh_adjust(m.loc[fam, f"p_sup_trained_{s}"]))
        assert np.isnan(m.loc[5, f"q_sup_trained_{s}"])
    # S4: low-fraction blocks (mean fraction under 0.01) leave the family; BH on the registered p of the rest
    assert m["low_fraction"].tolist() == [False, True, False, False, False, True]
    s4 = fam & ~m["low_fraction"].astype(bool)
    assert m.loc[s4, "q_sup_trained_s4"].tolist() == pytest.approx(mg.bh_adjust(m.loc[s4, "p_sup_trained_10k"]))
    assert m.loc[~s4, "q_sup_trained_s4"].isna().all()


def test_a_block_without_a_fraction_is_in_neither_tier():
    m, _ = cs.add_gate_columns(_blocks(mean_fraction=[0.3, np.nan, 0.02, 0.5, 0.17, 0.0001]))
    assert pd.isna(m.loc[1, "low_fraction"]) and np.isnan(m.loc[1, "q_sup_trained_s4"])


@pytest.mark.parametrize("hot, q_s1, genes, label", [
    (True, 0.01, 0.70, "supported"), (True, 0.01, 0.65, "supported"),
    (True, 0.20, 0.70, "neither"),          # not significant in S1
    (True, 0.01, 0.60, "neither"),          # between the two gene-space thresholds
    (True, 0.01, 0.54, "model-only"), (True, 0.50, 0.40, "model-only"),
    (True, np.nan, 0.90, "neither"), (True, 0.01, np.nan, "neither"),
    (False, 0.01, 0.90, ""),                # the rule describes hotspots only
])
def test_reading_rule(hot, q_s1, genes, label):
    assert cs.support_label(hot, q_s1, genes) == label


def test_support_column_follows_the_rule():
    m, _ = cs.add_gate_columns(_blocks())
    # block 0: hotspot, S1 significant, all-genes AUC 0.85; block 2: hotspot, S1 not significant, 0.60
    assert m["support"].tolist() == ["supported", "", "neither", "", "", ""]
    m2, _ = cs.add_gate_columns(_blocks(auc_genes=[0.50, 0.50, 0.60, 0.5, 0.5, 0.9]))
    assert m2.loc[0, "support"] == "model-only"


def test_figures_cover_analysable_and_all_blocks_with_no_threshold():
    m, _ = cs.add_gate_columns(_blocks())
    figs = cs.corroboration_figures(m)
    assert set(figs["scope"]) == {"analysable", "all"}
    for scope, n in (("analysable", 5), ("all", 6)):
        f = figs[figs["scope"] == scope].set_index("figure")
        assert f.loc["PLS-1: embedding vs PCA-50", "n"] == n and f.loc["PLS-1: embedding vs all genes", "n"] == n
        assert 0 <= f.loc["PLS-1: embedding vs PCA-50", "wilcoxon_p"] <= 1
        assert -1 <= f.loc["Augur vs PLS-1 probe (subspace_probe)", "spearman_r"] <= 1
    assert not any("threshold" in c or "pass" in c for c in figs.columns)


# ------------------------------------------------------------------ the script, on a small tree
def _gate_tree(tmp_path):
    """A gate folder and a Stage 8 results root for one tissue with four cell types."""
    gate, res = tmp_path / "gate", tmp_path / "results"
    types = ["Hepatocytes", "Kupffer cells", "NK cells", "α cells"]
    lib = [5e5, 2e4, 30.0, 4e3]
    (res / "motrpac/LIVER/pred_z").mkdir(parents=True); gate.mkdir()
    rng = np.random.default_rng(0)
    for t, size in zip(types, lib):
        z = pd.DataFrame(rng.uniform(size=(6, 4)), index=[f"mix{i + 1}" for i in range(6)], columns=list("abcd"))
        (z / z.sum(axis=1).values[:, None] * size).to_csv(res / f"motrpac/LIVER/pred_z/predz__{safe(t)}.csv")
    pd.DataFrame([[0.9, 0.08, 0.001, 0.019]] * 6, index=[f"mix{i + 1}" for i in range(6)], columns=types
                 ).to_csv(res / "motrpac/LIVER/estimated_fractions.csv")
    key = dict(tissue=["liver"] * 4, cell_type=types)
    pd.DataFrame(dict(key, n=6, sup_trained_auc=[0.9, 0.6, 0.95, 0.7], p_sup_trained=[0.0001, 0.4, 0.0001, 0.2],
                      glob_trained=0.03, p_glob_trained=0.2, b_sup_trained=[0, 3999, 0, 1999], m_sup_trained=10000,
                      p_sup_trained_10k=[0.0001, 0.4, 0.0001, 0.2], in_family=[True, True, False, True],
                      p_lo_sup_trained=[0.0, 0.37, np.nan, 0.18], p_hi_sup_trained=[0.002, 0.43, np.nan, 0.22],
                      gate_class=["hotspot", "none", np.nan, "none"],
                      sup_trained_auc_s1=[0.9, 0.6, 0.9, 0.7], p_sup_trained_s1=[0.0002, 0.5, 0.0003, 0.3],
                      n_s2=4, sup_trained_auc_s2=[0.8, 0.5, 0.9, 0.6], p_sup_trained_s2=[0.03, 0.5, 0.02, 0.4])
                 ).to_csv(gate / "subspace_probe.tsv", sep="\t", index=False)
    pd.DataFrame(dict(key, n=6, auc_embed=[0.9, 0.6, 0.95, 0.7], p_embed=0.1, auc_pca=[0.8, 0.5, 0.6, 0.6], p_pca=0.2,
                      auc_genes=[0.7, 0.5, 0.6, 0.6], p_genes=0.3)).to_csv(gate / "pca_control.tsv", sep="\t", index=False)
    pd.DataFrame([dict(tissue="liver", cell_type=t, representation=r, condition=c,
                       augur_auc=0.5 + 0.1 * i + (0.02 * (i + 1) if r == "embed" else 0.0))
                  for i, t in enumerate(types) for r in ("embed", "pca") for c in ("trained", "sex")]
                 ).to_csv(gate / "augur_results.tsv", sep="\t", index=False)
    return gate, res


def test_each_merge_writes_its_own_table_figures_and_provenance(tmp_path, capsys):
    gate, res = _gate_tree(tmp_path)
    cs.main(["--gc-root", str(gate), "--results-root", str(res)])
    for lib in (100, 0):
        cs.main(["--gc-root", str(gate), "--results-root", str(res), "--min-libsize", str(lib),
                 "--out", str(gate / f"corroboration_merged_minlib{lib}.tsv")])
    capsys.readouterr()
    for stem, lib, method, n_family in (("corroboration_merged", 1000.0, "mmctest", 3),
                                        ("corroboration_merged_minlib100", 100.0, "bh_10k", 3),
                                        ("corroboration_merged_minlib0", 0.0, "bh_10k", 4)):
        t = pd.read_csv(gate / f"{stem}.tsv", sep="\t")
        prov = json.loads((gate / f"provenance_{stem}.json").read_text())
        assert prov["params"]["min_libsize"] == lib and prov["params"]["out"].endswith(f"{stem}.tsv")
        assert prov["params"]["gate_method"] == method == t["gate_method"].iloc[0]
        assert int(t["analysable"].sum()) == n_family == prov["params"]["n_analysable"]
        assert (gate / f"{stem}.figures.tsv").exists()
    assert not (gate / "provenance_corroborate_summary.json").exists()   # the one shared record is gone

    t = pd.read_csv(gate / "corroboration_merged.tsv", sep="\t").set_index("cell_type")
    assert list(t.columns[:6]) == ["tissue", "n", "sup_trained_auc", "p_sup_trained", "glob_trained", "p_glob_trained"]
    assert t["gate_class"].to_dict() == {"Hepatocytes": "hotspot", "Kupffer cells": "none", "NK cells": "none",
                                         "α cells": "none"}
    assert t["is_hotspot"].tolist() == [True, False, False, False]
    assert t["median_libsize"].tolist() == pytest.approx([5e5, 2e4, 30.0, 4e3])
    assert t["mean_fraction"].tolist() == pytest.approx([0.9, 0.08, 0.001, 0.019])
    assert t.loc["NK cells", "low_fraction"] and not t.loc["α cells", "low_fraction"]
    assert t.loc["Hepatocytes", "support"] == "supported"
    for c in ("q_sup_trained", "q_sup_trained_10k", "p_lo_sup_trained", "p_hi_sup_trained", "q_sup_trained_s1",
              "q_sup_trained_s2", "q_sup_trained_s4", "auc_genes", "auc_pca", "b_sup_trained", "m_sup_trained"):
        assert c in t.columns
    # with no library rule the low-library block is in the family and is called by plain BH
    t0 = pd.read_csv(gate / "corroboration_merged_minlib0.tsv", sep="\t").set_index("cell_type")
    assert t0.loc["NK cells", "is_hotspot"] and t0.loc["NK cells", "gate_class"] == "hotspot"


def test_a_mistyped_option_is_an_error(tmp_path):
    gate, res = _gate_tree(tmp_path)
    with pytest.raises(SystemExit):
        cs.main(["--gc-root", str(gate), "--results-root", str(res), "--minimum-library", "0"])
