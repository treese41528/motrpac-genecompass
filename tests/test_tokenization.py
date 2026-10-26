"""Stage 5 corpus tokenization and the Stage 8 pseudo-cell variant with PA-gene preference.

GeneCompass input = per gene log2(1 + count / median), genes ranked descending, top-N token IDs,
zero-padded. Every later stage (pseudo-cells, the human transfer) reuses this arithmetic.
"""
from pathlib import Path

import numpy as np
import pytest

from tokenize_corpus import (build_eligible_gene_arrays, build_gene_index, get_task_files,
                             map_varnames_to_eligible, tokenize_cell_batch, _infer_study_id)
from tokenize_pseudocells import tokenize_cell_batch_pa

TOKENS = np.array([11, 12, 13, 14, 15], dtype=np.int32)
MEDIANS = np.array([1.0, 2.0, 4.0, 1.0, 0.5], dtype=np.float32)


def test_values_are_log2_of_count_over_median_ranked_descending():
    X = np.array([[1, 2, 0, 3, 1]], dtype=np.float32)
    ids, vals, lens = tokenize_cell_batch(X, TOKENS, MEDIANS, top_n=8)
    # x/median = [1, 1, 0, 3, 2] -> log2(1+x) = [1, 1, 0, 2, log2 3]
    assert lens == [4]
    assert ids[0][:2] == [14, 15]                       # 2.0, then log2(3)
    assert sorted(ids[0][2:4]) == [11, 12]              # the two ties at 1.0
    assert vals[0][:4] == pytest.approx([2.0, np.log2(3), 1.0, 1.0])
    assert ids[0][4:] == [0] * 4 and vals[0][4:] == [0.0] * 4   # zero padding to top_n
    assert len(ids[0]) == len(vals[0]) == 8


def test_top_n_truncates_and_empty_cells_are_all_padding():
    X = np.array([[5, 5, 5, 5, 5], [0, 0, 0, 0, 0]], dtype=np.float32)
    ids, vals, lens = tokenize_cell_batch(X, TOKENS, MEDIANS, top_n=2)
    assert lens == [2, 0]
    assert ids[0][0] == 15 and ids[0][1] in (11, 14)    # largest count/median ratios (11, 14 tie)
    assert ids[1] == [0, 0] and vals[1] == [0.0, 0.0]


def test_ranking_ignores_library_size():
    """Scaling a cell changes the values but, without ties, never the token order."""
    X = np.array([[1, 7, 3, 2, 9]], dtype=np.float32)
    a = tokenize_cell_batch(X, TOKENS, MEDIANS, top_n=5)[0]
    b = tokenize_cell_batch(X * 37.0, TOKENS, MEDIANS, top_n=5)[0]
    assert a == b


def test_eligible_genes_need_both_a_token_and_a_median():
    ids, toks, meds = build_eligible_gene_arrays({"G2": 2, "G1": 1, "G3": 3}, {"G3": 0.5, "G1": 2.0, "G9": 1.0})
    assert ids == ["G1", "G3"]                          # sorted intersection
    assert toks.tolist() == [1, 3] and meds.tolist() == [2.0, 0.5]
    with pytest.raises(RuntimeError):
        build_eligible_gene_arrays({"A": 1}, {"B": 1.0})


def test_var_names_map_ignoring_version_and_case():
    index = build_gene_index(["ENSRNOG00000000001", "ENSRNOG00000000002"])
    cols, genes = map_varnames_to_eligible(
        ["ensrnog00000000002.3", "ENSRNOG00000000009", " ENSRNOG00000000001 "], index)
    assert cols == [0, 2] and genes == [1, 0]


@pytest.mark.parametrize("name, study", [
    ("GSE242354_sample01.h5ad", "GSE242354"), ("E-MTAB-1234_sample.h5ad", "E-MTAB-1234"),
    ("OMIX767_testis1.h5ad", "OMIX767"), ("mystery.h5ad", "mystery"),
])
def test_study_id_from_filename(name, study):
    assert _infer_study_id(Path(name)) == study


def test_slurm_task_split_covers_every_file_once():
    files = [Path(f"f{i}") for i in range(23)]
    parts = [get_task_files(files, t, 5) for t in range(5)]
    assert sorted(sum(parts, [])) == sorted(files)
    assert max(map(len, parts)) - min(map(len, parts)) <= 1


# ------------------------------------------------------------------ PA-gene preference (Stage 8)
def test_pa_preference_without_pa_genes_matches_corpus_tokenizer():
    rng = np.random.default_rng(0)
    X = rng.poisson(3, size=(6, 5)).astype(np.float32)
    plain = tokenize_cell_batch(X, TOKENS, MEDIANS, top_n=3)
    *pa, promoted = tokenize_cell_batch_pa(X, TOKENS, MEDIANS, 3, np.zeros(5, bool), pa_max_rank=5)
    assert promoted == 0
    assert [np.sort(r).tolist() for r in pa[0]] == [np.sort(r).tolist() for r in plain[0]]
    assert pa[2] == plain[2]


def test_pa_gene_just_below_the_cut_is_promoted_over_the_weakest_gene():
    medians = np.ones(5, np.float32)
    X = np.array([[50, 40, 30, 20, 10]], dtype=np.float32)
    pa_mask = np.array([False, False, False, True, False])     # gene 4th by value
    ids, vals, lens, promoted = tokenize_cell_batch_pa(X, TOKENS, medians, 3, pa_mask, pa_max_rank=4)
    assert promoted == 1
    assert ids[0] == [11, 12, 14]                       # 13 (lowest non-PA in the top 3) dropped
    assert vals[0] == sorted(vals[0], reverse=True)     # still expression-descending, true values
    assert vals[0][2] == pytest.approx(np.log2(21.0))


def test_pa_gene_outside_the_rank_window_is_not_promoted():
    medians = np.ones(5, np.float32)
    X = np.array([[50, 40, 30, 20, 10]], dtype=np.float32)
    pa_mask = np.array([False, False, False, False, True])     # 5th by value, window = 4
    ids, _, _, promoted = tokenize_cell_batch_pa(X, TOKENS, medians, 3, pa_mask, pa_max_rank=4)
    assert promoted == 0 and ids[0] == [11, 12, 13]
