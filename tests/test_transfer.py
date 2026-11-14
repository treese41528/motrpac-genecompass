"""Stage 12: re-expressing rat pseudo-cells as human cells, and the Stage 12 orchestration.

The transfer must change only three things relative to the rat path (token dict, medians,
species ID). These tests pin the parts that make that true: many-to-one orthologs are summed,
unmapped rat genes still count toward the library size, and the tokens match the rat path.
"""
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import compare_transfer as ct
import mc_gate as mg
import run_stage12 as s12
from compare_transfer import classify
from tokenize_corpus import tokenize_cell_batch
from transfer_to_human import HUMAN_SPECIES_ID, project_rat_to_human, transfer_one

RAT = ["ENSRNOG00000000001", "ENSRNOG00000000002", "ENSRNOG00000000003", "ENSRNOG00000000004"]
# rat genes 1 and 2 are both orthologs of ENSG...A; gene 3 -> B; gene 4 has no human ortholog
R2H = {RAT[0]: "ENSG00000000001", RAT[1]: "ENSG00000000001", RAT[2]: "ENSG00000000002"}


def _rat_cells(X):
    a = ad.AnnData(X=sp.csr_matrix(np.asarray(X, dtype=np.float32)),
                   obs=pd.DataFrame({"sample": ["s1", "s2"], "cell_type": ["Hep", "Hep"],
                                     "tissue": ["liver", "liver"]}, index=["c1", "c2"]))
    a.var_names = [g + ".1" for g in RAT]                 # versioned IDs must still map
    return a


def test_orthologs_sum_many_to_one_and_unmapped_genes_drop():
    Xh, genes, lib, stats = project_rat_to_human(_rat_cells([[1, 2, 3, 4], [0, 5, 0, 5]]), R2H)
    assert genes == ["ENSG00000000001", "ENSG00000000002"]
    assert Xh.tolist() == [[3, 3], [5, 0]]
    assert lib.ravel().tolist() == [10, 10]              # includes the dropped rat gene's counts
    assert stats["n_rat_mapped"] == 3 and stats["n_collisions_summed"] == 1
    assert stats["frac_count_mass_mapped"] == pytest.approx(11 / 20)


def test_no_orthologs_is_an_error():
    with pytest.raises(SystemExit):
        project_rat_to_human(_rat_cells([[1, 2, 3, 4], [1, 1, 1, 1]]), {})


def test_transfer_tokens_match_the_rat_path_on_shared_genes():
    X = np.array([[1, 2, 3, 4], [6, 0, 2, 2]], dtype=np.float32)
    token_dict = {"ENSG00000000001": 7, "ENSG00000000002": 9}
    median_dict = {"ENSG00000000001": 0.5, "ENSG00000000002": 2.0}
    rows, summary = transfer_one(_rat_cells(X), R2H, token_dict, median_dict,
                                 target_sum=100.0, top_n=4, min_genes=1)

    # rat path by hand: normalize over ALL rat genes, then tokenize the human-summed columns
    Xh = np.stack([X[:, 0] + X[:, 1], X[:, 2]], axis=1)
    expect = tokenize_cell_batch((Xh / X.sum(1, keepdims=True) * 100.0).astype(np.float32),
                                 np.array([7, 9], np.int32), np.array([0.5, 2.0], np.float32), 4)
    assert rows["input_ids"] == expect[0]
    assert np.allclose(rows["values"], expect[1])
    assert rows["species"] == [[HUMAN_SPECIES_ID]] * 2 and HUMAN_SPECIES_ID == 0
    assert rows["cell_id"] == ["c1", "c2"] and rows["tissue"] == ["liver", "liver"]
    assert summary["target_sum"] == 100.0 and summary["n_eligible_genes"] == 2


def test_cells_below_min_genes_are_dropped():
    X = np.array([[1, 2, 3, 4], [0, 0, 0, 9]], dtype=np.float32)   # c2 has only the unmapped gene
    rows, summary = transfer_one(_rat_cells(X), R2H, {"ENSG00000000001": 7, "ENSG00000000002": 9},
                                 {"ENSG00000000001": 1.0, "ENSG00000000002": 1.0},
                                 target_sum=100.0, top_n=4, min_genes=1)
    assert rows["cell_id"] == ["c1"] and summary["n_dropped_lt_min_genes"] == 1


# ------------------------------------------------------------------ verdicts
@pytest.mark.parametrize("auc, p, verdict", [
    (0.80, 0.001, "PRESERVED"), (0.80, 0.20, "WEAKENED"), (0.80, np.nan, "WEAKENED"),
    (0.60, 0.001, "LOST"), (np.nan, 0.001, "no_human_block"),
])
def test_transfer_verdicts(auc, p, verdict):
    row = pd.Series({"sup_trained_auc_human": auc, "p_sup_trained_human": p})
    assert classify(row, auc_thresh=0.70, p_thresh=0.05) == verdict


# ------------------------------------------------------------------ Stage 12 orchestration
def _cfg(tmp_path, hotspots):
    d = {"hotspots_table": hotspots} if hotspots else {}
    return {"_project_root": tmp_path, "deconvolution": d}


def test_gate_dirs_mirror_the_rat_gate_subfolder(tmp_path):
    rat = tmp_path / "data/genecompass_input_v2"
    human = Path(str(rat) + "_human")
    cfg = _cfg(tmp_path, "data/genecompass_input_v2/gate_rcv10/corroboration_merged.tsv")
    assert s12.resolve_gate_dirs(cfg, rat, human) == (rat / "gate_rcv10", human / "gate_rcv10")


def test_gate_dirs_default_to_the_input_root(tmp_path):
    rat = tmp_path / "gi"
    human = tmp_path / "gi_human"
    assert s12.resolve_gate_dirs(_cfg(tmp_path, None), rat, human) == (rat, human)


def test_gate_outside_the_input_root_keeps_its_folder_name(tmp_path):
    rat = tmp_path / "gi"
    cfg = _cfg(tmp_path, "elsewhere/gate_x/corroboration_merged.tsv")
    assert s12.resolve_gate_dirs(cfg, rat, tmp_path / "gi_human") == \
        (tmp_path / "elsewhere/gate_x", tmp_path / "gi_human/gate_x")


def test_probe_and_compare_commands_use_the_gate_method(tmp_path):
    ctx = {"human_root": tmp_path / "h", "human_gate": tmp_path / "h/g", "rat_gate": tmp_path / "r/g",
           "rat_root": tmp_path / "r", "perms": 10000, "repeats": 10, "jobs": 4}
    probe, compare = s12.global_steps(ctx)
    cmd = probe["cmd"]
    assert cmd[cmd.index("--perms") + 1] == "10000" and cmd[cmd.index("--repeats") + 1] == "10"
    assert cmd[cmd.index("--out") + 1] == str(tmp_path / "h/g/subspace_probe.tsv")
    # the human side is sampled and classified over the rat gate's family (STEP3_PREREG section 7)
    assert "--sequential" in cmd
    assert cmd[cmd.index("--family-table") + 1] == str(tmp_path / "r/g/corroboration_merged.tsv")
    assert cmd[cmd.index("--scorer") + 1] == "kernel"
    c = compare["cmd"]
    assert c[c.index("--rat-probe") + 1] == str(tmp_path / "r/g/subspace_probe.tsv")
    assert c[c.index("--human-probe") + 1] == str(tmp_path / "h/g/subspace_probe.tsv")
    assert c[c.index("--rat-root") + 1] == str(tmp_path / "r")
    assert c[c.index("--human-root") + 1] == str(tmp_path / "h")
    assert compare["key"] == tmp_path / "h/g/transfer_comparison.md"


def test_transfer_step_passes_the_target_sum(tmp_path):
    ctx = {"human_root": tmp_path / "h", "rat_root": tmp_path / "r", "target_sum": 6400.0,
           "model_dir": tmp_path / "m", "n_cells": 5, "device": "cpu"}
    transfer, embed = s12.per_tissue_steps("liver", ctx)
    assert transfer["cmd"][transfer["cmd"].index("--target-sum") + 1] == "6400.0"
    assert embed["cmd"][embed["cmd"].index("--species") + 1] == "0"


def test_model_dir_picks_the_highest_checkpoint_numerically(tmp_path):
    for n in (9, 10, 517793):
        (tmp_path / f"checkpoint-{n}").mkdir()
    assert s12.resolve_model_dir(tmp_path) == tmp_path / "checkpoint-517793"
    (tmp_path / "config.json").write_text("{}")
    assert s12.resolve_model_dir(tmp_path) == tmp_path


# ------------------------------------------------------------------ corrected transfer (STEP3_PREREG section 7)
@pytest.mark.parametrize("cls, auc, verdict", [
    ("hotspot", 0.80, "PASS"), ("borderline", 0.65, "PASS"), ("none", 0.80, "FAIL"),
    ("hotspot", 0.64, "FAIL"), ("hotspot", np.nan, "no_human_block"), ("none", np.nan, "no_human_block"),
])
def test_corrected_verdict(cls, auc, verdict):
    assert ct.corrected_verdict(cls, auc, auc_thresh=0.65) == verdict


def _human_side(**cols):
    base = dict(tissue=["liver"] * 5, cell_type=list("abcde"),
                family=[True, True, True, True, False],
                p_sup_trained_human=[0.0001, 0.0292, 0.0098, 0.30, 0.0001],
                p_registered_human=[0.0001, 0.0285, 0.0099, 0.30, 0.0001])
    base.update(cols)
    return pd.DataFrame(base)


def test_human_side_of_an_older_table_gets_plain_bh_over_the_rat_family():
    q, cls, method = ct.human_gate(_human_side())
    assert method == "bh_10k"
    assert q[:4].tolist() == pytest.approx(mg.bh_adjust([0.0001, 0.0285, 0.0099, 0.30]))
    assert cls.tolist() == ["hotspot", "hotspot", "hotspot", "none", "none"]
    assert np.isnan(q[4])                                 # outside the rat family: no q, never a hotspot


def test_human_side_with_interval_limits_gets_the_three_classes():
    m = _human_side(in_family_human=[True, True, True, True, False],
                    p_lo_sup_trained_human=[0.0, 0.0372, 0.0093, 0.27, np.nan],
                    p_hi_sup_trained_human=[0.0015, 0.0392, 0.0104, 0.33, np.nan])
    q, cls, method = ct.human_gate(m)
    assert method == "mmctest"
    assert q[:4].tolist() == pytest.approx(mg.bh_adjust([0.0001, 0.0292, 0.0098, 0.30]))   # the final p
    # with 4 blocks the third p must be below 0.05 x 3 / 4 = 0.0375: its interval straddles that value
    assert cls.tolist() == ["hotspot", "borderline", "hotspot", "none", "none"]
    with pytest.raises(SystemExit, match="different family"):
        ct.human_gate(m.assign(in_family_human=[True] * 5))
    # a family block with no human test is left out of the correction
    gone = m.assign(p_sup_trained_human=[0.0001, np.nan, 0.0098, 0.30, 0.0001])
    q, cls, _ = ct.human_gate(gone)
    assert np.isnan(q[1]) and cls[1] == "none"
    assert q[[0, 2, 3]].tolist() == pytest.approx(mg.bh_adjust([0.0001, 0.0098, 0.30]))


def _tokenized(path, ids, cell_type="Goblet cells"):
    from datasets import Dataset
    Dataset.from_dict({"cell_id": ids, "cell_type": [cell_type] * len(ids),
                       "sample": [f"mix{i + 1}" for i in range(len(ids))]}).save_to_disk(str(path))


def test_rat_only_pseudocells_are_listed_and_a_human_only_cell_is_an_error(tmp_path):
    rat, human = tmp_path / "gi", tmp_path / "gi_human"
    _tokenized(rat / "lung/dataset", ["c1", "c2", "c3", "c4"]); _tokenized(human / "lung/dataset", ["c1", "c3"])
    _tokenized(rat / "liver/dataset", ["h1", "h2"], "Hepatocytes"); _tokenized(human / "liver/dataset", ["h1", "h2"])
    _tokenized(rat / "bat/dataset", ["b1"])                               # no human dataset: not compared
    only, compared = ct.rat_only_pseudocells(rat, human)
    assert compared == ["liver", "lung"]
    assert only.to_dict("list") == {"tissue": ["lung", "lung"], "cell_type": ["Goblet cells"] * 2,
                                    "sample": ["mix2", "mix4"], "cell_id": ["c2", "c4"]}
    _tokenized(human / "bat/dataset", ["b1", "b9"])
    with pytest.raises(SystemExit, match="not in the rat dataset"):
        ct.rat_only_pseudocells(rat, human)


def _transfer_tables(tmp_path, new_format):
    """Rat gate table, rat probe and human probe for four blocks; the Step 3 columns only when new_format."""
    key = dict(tissue=["lung"] * 4, cell_type=["A", "B", "C", "D"])
    rat_c = dict(key, sup_trained_auc=[0.85, 0.80, 0.78, 0.60], p_sup_trained=[0.0002, 0.001, 0.02, 0.4],
                 q_sup_trained=[0.0006, 0.0015, 0.02, np.nan])
    rat_p = dict(key, sup_trained_auc=rat_c["sup_trained_auc"], sup_dose_rho=[0.4, 0.3, 0.2, 0.0],
                 sup_sex_auc=[0.8, 0.7, 0.75, 0.7])
    hum_p = dict(key, sup_trained_auc=[0.82, 0.70, 0.66, np.nan], p_sup_trained=[0.0003, 0.04996, 0.03, np.nan],
                 sup_dose_rho=[0.35, -0.1, 0.0, np.nan], sup_sex_auc=[0.8, 0.7, 0.75, np.nan])
    if new_format:
        rat_c.update(p_sup_trained_10k=[0.0002, 0.001, 0.02, 0.4], gate_class=["hotspot", "hotspot", "borderline", "none"],
                     is_hotspot=[True, True, False, False], analysable=[True, True, True, False])
        hum_p.update(p_sup_trained_10k=[0.0003, 0.05, 0.03, np.nan], b_sup_trained=[2, 499, 299, np.nan],
                     m_sup_trained=[10000, 10000, 10000, np.nan], in_family=[True, True, True, False],
                     p_lo_sup_trained=[0.0, 0.040, 0.022, np.nan], p_hi_sup_trained=[0.002, 0.060, 0.040, np.nan])
    paths = {}
    for name, d in (("rat_corr", rat_c), ("rat_probe", rat_p), ("human_probe", hum_p)):
        paths[name] = tmp_path / f"{name}.tsv"
        pd.DataFrame(d).to_csv(paths[name], sep="\t", index=False)
    return paths


def _run_compare(tmp_path, monkeypatch, capsys, new_format):
    paths = _transfer_tables(tmp_path, new_format)
    out = tmp_path / "out"
    monkeypatch.setattr("sys.argv", ["compare_transfer.py", "--rat-corr", str(paths["rat_corr"]),
                                     "--rat-probe", str(paths["rat_probe"]), "--human-probe", str(paths["human_probe"]),
                                     "--human-corr", str(tmp_path / "absent.tsv"), "--rat-root", str(tmp_path / "no_rat"),
                                     "--human-root", str(tmp_path / "no_human"), "--out", str(out)])
    ct.main()
    capsys.readouterr()
    return pd.read_csv(out / "transfer_comparison.tsv", sep="\t").set_index("cell_type"), \
        (out / "transfer_comparison.md").read_text()


def test_comparison_keeps_the_registered_status_and_adds_the_corrected_rule(tmp_path, monkeypatch, capsys):
    t, md = _run_compare(tmp_path, monkeypatch, capsys, new_format=True)
    # the registered rule reads the 10,000-permutation p as stored (0.05 is not below 0.05), not the final p
    assert t["status"].to_dict() == {"A": "PRESERVED", "B": "WEAKENED", "C": "PRESERVED", "D": "no_human_block"}
    assert t["is_hotspot"].to_dict() == {"A": True, "B": True, "C": False, "D": False}
    assert t["gate_class_rat"].to_dict() == {"A": "hotspot", "B": "hotspot", "C": "borderline", "D": "none"}
    # human side, 3 family blocks: upper limits 0.002, 0.060, 0.040 -> only A is rejected there;
    # lower limits 0, 0.040, 0.022 -> all three are
    assert t["gate_class_human"].to_dict() == {"A": "hotspot", "B": "borderline", "C": "borderline", "D": "none"}
    assert (t["gate_method_human"] == "mmctest").all()
    assert t["transfer_corrected"].to_dict() == {"A": "PASS", "B": "PASS", "C": "PASS", "D": "no_human_block"}
    assert t["dose_trend_lost"].to_dict() == {"A": False, "B": True, "C": True, "D": False}
    assert t.loc[["A", "B", "C"], "q_sup_trained_human"].tolist() == pytest.approx(mg.bh_adjust([0.0003, 0.04996, 0.03]))
    assert np.isnan(t.loc["D", "q_sup_trained_human"])
    for c in ("sup_dose_rho_rat", "sup_dose_rho_human", "p_sup_trained_10k_human", "p_lo_sup_trained_human"):
        assert c in t.columns
    # the markdown states the reuse where it states the counts, and names the borderline block
    lines = md.splitlines()
    i = next(k for k, line in enumerate(lines) if line.startswith("- Corrected rule: transfers 2/2"))
    assert lines[i - 1].startswith("- Registered rule: PRESERVED 1/2")
    assert lines[i + 1] == "- " + ct.REUSE_NOTE
    assert "reuses the rat test's cells, labels, folds and seeds" in ct.REUSE_NOTE and "not a replication" in ct.REUSE_NOTE
    assert "- Dose trend lost (human dose correlation zero or negative): 1/2: lung B (0.300 -> -0.100)" in md
    assert "borderline in rat space, not counted as hotspots (1): lung C" in md
    assert not (tmp_path / "out" / "rat_only_pseudocells.tsv").exists()   # the two trees are not on disk here


def test_comparison_still_reads_tables_from_before_step3(tmp_path, monkeypatch, capsys):
    t, md = _run_compare(tmp_path, monkeypatch, capsys, new_format=False)
    assert t["is_hotspot"].to_dict() == {"A": True, "B": True, "C": True, "D": False}      # q_sup_trained < 0.05
    assert t["status"].to_dict() == {"A": "PRESERVED", "B": "PRESERVED", "C": "PRESERVED", "D": "no_human_block"}
    assert (t["gate_method_human"] == "bh_10k").all() and "gate_class_rat" not in t.columns
    assert t["gate_class_human"].to_dict() == {"A": "hotspot", "B": "hotspot", "C": "hotspot", "D": "none"}
    assert ct.REUSE_NOTE in md
