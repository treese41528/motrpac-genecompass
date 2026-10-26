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
           "perms": 10000, "repeats": 10, "jobs": 4}
    probe, compare = s12.global_steps(ctx)
    cmd = probe["cmd"]
    assert cmd[cmd.index("--perms") + 1] == "10000" and cmd[cmd.index("--repeats") + 1] == "10"
    assert cmd[cmd.index("--out") + 1] == str(tmp_path / "h/g/subspace_probe.tsv")
    c = compare["cmd"]
    assert c[c.index("--rat-probe") + 1] == str(tmp_path / "r/g/subspace_probe.tsv")
    assert c[c.index("--human-probe") + 1] == str(tmp_path / "h/g/subspace_probe.tsv")
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
