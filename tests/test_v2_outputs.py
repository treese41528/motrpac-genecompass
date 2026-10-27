"""Consistency checks on the pipeline's outputs on this machine (pytest -m data).

They read whichever model the config selects (v2 by default, April with PIPELINE_PROFILE) and
check that the stages agree with each other: the target sum each tokenizer used, pseudo-cell
counts carried from rat to human, unique cell-type filenames, and the Stage 12 verdict table.
"""
import json

import anndata as ad
import pandas as pd
import pytest

import celltype_names as cn
from gene_utils import resolve_path

pytestmark = pytest.mark.data


@pytest.fixture(scope="module")
def roots(config, require):
    rat = require(resolve_path(config, config["deconvolution"]["genecompass_input_dir"]))
    return rat, rat.parent / (rat.name + "_human")


@pytest.fixture(scope="module")
def tissues(roots):
    labels = sorted(p.parent.name for p in roots[0].glob("*/pseudocells.h5ad"))
    assert labels, "no tissues with pseudocells.h5ad"
    return labels


def _summary(d):
    return json.loads((d / "tokenize_summary.json").read_text())


def test_rat_tokenization_used_the_configured_target_sum(config, roots, tissues):
    want = float(config["deconvolution"]["pseudocell_target_sum"])
    for t in tissues:
        s = _summary(roots[0] / t)
        assert (s["target_sum"], s["species"]) == (want, 2), t


def test_every_pseudocell_was_tokenized(roots, tissues):
    for t in tissues:
        n = ad.read_h5ad(roots[0] / t / "pseudocells.h5ad", backed="r").n_obs
        s = _summary(roots[0] / t)
        assert s["n_pseudocells"] + s["n_dropped_lt_min_genes"] == n, t


def test_cell_type_filenames_are_unique_in_every_tissue(roots, tissues):
    for t in tissues:
        obs = ad.read_h5ad(roots[0] / t / "pseudocells.h5ad", backed="r").obs
        assert cn.assert_injective(obs["cell_type"].astype(str).unique(), context=t)


def test_human_transfer_mirrors_the_rat_run(roots, tissues, require):
    require(roots[1])
    for t in tissues:
        rat, human = _summary(roots[0] / t), _summary(roots[1] / t)
        assert human["species"] == 0, t
        assert human["target_sum"] == rat["target_sum"], t
        assert human["n_pseudocells"] == rat["n_pseudocells"], t


def test_stage12_verdicts_cover_every_rat_hotspot(config, roots, require):
    from run_stage12 import resolve_gate_dirs
    rat_gate, human_gate = resolve_gate_dirs(config, *roots)
    corr = pd.read_csv(require(rat_gate / "corroboration_merged.tsv"), sep="\t")
    comp = pd.read_csv(require(human_gate / "transfer_comparison.tsv"), sep="\t")
    key = ["tissue", "cell_type"]
    assert set(map(tuple, comp[key].values)) == set(map(tuple, corr[key].values))
    assert comp["status"].isin({"PRESERVED", "WEAKENED", "LOST", "no_human_block"}).all()
    hot = comp[comp["is_hotspot"]]
    assert len(hot) > 0
    assert (comp.loc[comp["status"] == "PRESERVED", "sup_trained_auc_human"].notna()).all()
