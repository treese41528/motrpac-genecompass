"""Stage 8 connector: cell-type filenames and BayesPrism Z -> pseudo-cells.

The filename contract guards a real failure: the old sanitizer mapped KIDNEY's alpha- and
beta-intercalated cells to the same file, so one cell type's Z overwrote the other's.
"""
import os

import numpy as np
import pandas as pd
import pytest

import celltype_names as cn
from build_pseudocells import build_pseudocells, load_pred_z

KIDNEY = ["α-intercalated cells", "β-intercalated cells"]


# ------------------------------------------------------------------ filename contract
def test_greek_letters_are_spelled_out_so_names_stay_distinct():
    assert [cn.safe(t) for t in KIDNEY] == ["alpha_intercalated_cells", "beta_intercalated_cells"]
    assert cn.assert_injective(KIDNEY)


def test_legacy_sanitizer_collides_on_the_kidney_labels():
    assert cn.legacy_safe(KIDNEY[0]) == cn.legacy_safe(KIDNEY[1])


def test_ascii_labels_keep_their_legacy_names_apart_from_edge_underscores():
    for t in ["Hepatocytes", "T cells", "Endothelial (venous)", "NK/NKT"]:
        assert cn.safe(t) == cn.legacy_safe(t).strip("_")


def test_colliding_labels_are_refused():
    with pytest.raises(ValueError, match="collision"):
        cn.assert_injective(["T cell", "T-cell"], context="test")


def test_resolve_prefers_current_name_then_legacy(tmp_path):
    t = KIDNEY[0]
    assert cn.resolve(tmp_path, t, "predz__", ".csv").endswith("predz__alpha_intercalated_cells.csv")
    legacy = tmp_path / f"predz__{cn.legacy_safe(t)}.csv"
    legacy.write_text("x")
    assert cn.resolve(tmp_path, t, "predz__", ".csv") == str(legacy)
    current = tmp_path / "predz__alpha_intercalated_cells.csv"
    current.write_text("x")
    assert cn.resolve(tmp_path, t, "predz__", ".csv") == str(current)


def test_purge_stale_removes_only_orphans(tmp_path):
    for name in ["predz__Hepatocytes.csv", "predz__old_label.csv", "genes.txt"]:
        (tmp_path / name).write_text("x")
    dead = cn.purge_stale(tmp_path, ["Hepatocytes"], "predz__", ".csv")
    assert [os.path.basename(p) for p in dead] == ["predz__old_label.csv"]
    assert sorted(os.listdir(tmp_path)) == ["genes.txt", "predz__Hepatocytes.csv"]


# ------------------------------------------------------------------ pseudo-cells
GENES = ["ENSRNOG00000000001", "ENSRNOG00000000002", "ENSRNOG00000000003"]
SAMPLES = ["s1", "s2"]


def _pred_z(d, types, mats):
    d.mkdir(exist_ok=True)
    (d / "genes.txt").write_text("\n".join(GENES) + "\n")
    (d / "types.txt").write_text("\n".join(types) + "\n")
    for t, m in zip(types, mats):
        # columns deliberately out of order: the loader must re-align to genes.txt
        pd.DataFrame(m, index=SAMPLES, columns=GENES)[GENES[::-1]].to_csv(d / f"predz__{cn.safe(t)}.csv")
    return d


def test_one_pseudocell_per_sample_and_cell_type(tmp_path):
    z = [np.array([[1, 2, 3], [4, 5, 6]]), np.array([[7, 8, 9], [0, 0, 0]])]
    d = _pred_z(tmp_path / "pred_z", KIDNEY, z)
    adata, n_drop, n_samples, n_types = build_pseudocells(d, "kidney")
    assert (n_samples, n_types) == (2, 2)
    assert n_drop == 1                                   # the all-zero (s2, beta) block
    assert list(adata.var_names) == GENES
    assert list(adata.obs_names) == ["kidney|s1|alpha_intercalated_cells",
                                     "kidney|s2|alpha_intercalated_cells",
                                     "kidney|s1|beta_intercalated_cells"]
    assert adata.obs["cell_type"].tolist() == [KIDNEY[0], KIDNEY[0], KIDNEY[1]]
    assert np.array_equal(adata.X.toarray(), np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
    assert np.array_equal(adata.layers["z_raw"].toarray(), adata.X.toarray())


def test_metadata_joins_on_sample(tmp_path):
    d = _pred_z(tmp_path / "pred_z", ["Hepatocytes"], [np.ones((2, 3))])
    meta = pd.DataFrame({"viallabel": ["s1", "s2"], "sex": ["male", "female"]})
    adata, *_ = build_pseudocells(d, "liver", meta=meta, meta_key="viallabel")
    assert adata.obs["sex"].tolist() == ["male", "female"]


def test_missing_cell_type_file_stops_the_build(tmp_path):
    d = _pred_z(tmp_path / "pred_z", ["Hepatocytes"], [np.ones((2, 3))])
    (d / "types.txt").write_text("Hepatocytes\nKupffer cells\n")
    with pytest.raises(SystemExit, match="Kupffer"):
        load_pred_z(d)
