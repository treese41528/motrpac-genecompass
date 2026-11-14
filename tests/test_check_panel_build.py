"""The check that runs between building the mixture panel and deconvolving it
(docs/audit/remediation/STEP4_V5_V6_DRAFT.md note N10): a cross set is deconvolved only if its harmonization of
source labels to reference types is the one that was reviewed for the earlier panel.
"""
import sys

import pandas as pd
import pytest

import check_panel_build as cpb


def _set(root, name, mixture=b"m1", truth=b"t1", ref=b"r1", hmap=None):
    d = root / name
    (d / "mixtures").mkdir(parents=True); (d / "reference").mkdir()
    (d / "mixtures" / "pseudobulk_counts.mtx").write_bytes(mixture)
    (d / "mixtures" / "true_fractions.tsv").write_bytes(truth)
    (d / "mixtures" / "pseudobulk_genes.tsv").write_bytes(b"g")
    if not name.endswith("_cross"):
        (d / "reference" / "reference_counts.mtx").write_bytes(ref)
        (d / "reference" / "cells_meta.tsv").write_bytes(b"c")
    if hmap is not None:
        pd.DataFrame(hmap, columns=["cell_type", "n_cells", "ref_type"]).to_csv(
            d / "mixtures" / "harmonization_map.tsv", sep="\t", index=False)


HMAP = [("CD8+ T cells", 479, "Immune cells"), ("Podocytes", 80, "Podocytes"), ("Naive B cells", 12, "")]


def test_identical_build_is_ok_and_row_order_of_the_map_does_not_matter(tmp_path):
    now, prev = tmp_path / "now", tmp_path / "prev"
    for root, hm in ((now, HMAP), (prev, HMAP[::-1])):
        _set(root, "KIDNEY_holdout"); _set(root, "KIDNEY_cross", hmap=hm)
    (now / "logs").mkdir()                                     # not a set: ignored
    t, status = cpb.check(now, prev)
    assert status == "ok"
    assert list(t["set"]) == ["KIDNEY_holdout", "KIDNEY_cross"]          # holdout sets first
    t = t.set_index("set")
    assert t.loc["KIDNEY_cross", "harmonization_map"] == "identical"
    assert t.loc["KIDNEY_holdout", "harmonization_map"] == "none"
    assert (t[["mixtures_vs_previous", "truth_vs_previous"]] == "identical").all().all()
    assert t.loc["KIDNEY_holdout", "reference_vs_previous"] == "identical"
    assert t.loc["KIDNEY_cross", "reference_vs_previous"] == "absent"    # a cross set has no reference of its own


@pytest.mark.parametrize("changed", [
    [("CD8+ T cells", 479, "T cells"), HMAP[1], HMAP[2]],      # a label goes to another reference type
    [HMAP[0], HMAP[1], ("Naive B cells", 12, "B cells")],      # an unmapped label becomes mapped
    [HMAP[0], ("Podocytes", 81, "Podocytes"), HMAP[2]],        # another number of source cells
    HMAP[:2],                                                  # a source label is gone
])
def test_a_changed_cross_harmonization_blocks_the_cross_sets(tmp_path, changed):
    now, prev = tmp_path / "now", tmp_path / "prev"
    _set(now, "KIDNEY_cross", hmap=changed); _set(prev, "KIDNEY_cross", hmap=HMAP)
    t, status = cpb.check(now, prev)
    assert status != "ok" and "KIDNEY_cross (different)" in status
    assert t.set_index("set").loc["KIDNEY_cross", "harmonization_map"] == "different"


def test_a_cross_set_without_an_earlier_reviewed_map_is_not_ok(tmp_path):
    now, prev = tmp_path / "now", tmp_path / "prev"
    _set(now, "LUNG_cross", hmap=HMAP); prev.mkdir()
    _, status = cpb.check(now, prev)
    assert "LUNG_cross (no earlier map)" in status


def test_a_changed_holdout_set_is_reported_but_does_not_block(tmp_path):
    """HEART and HYPOTH changed by design in refs5: their mixtures differ from the earlier panel's."""
    now, prev = tmp_path / "now", tmp_path / "prev"
    _set(now, "HEART_holdout", mixture=b"new", truth=b"new", ref=b"new"); _set(prev, "HEART_holdout")
    t, status = cpb.check(now, prev)
    assert status == "ok"
    r = t.set_index("set").loc["HEART_holdout"]
    assert r["mixtures_vs_previous"] == r["truth_vs_previous"] == r["reference_vs_previous"] == "different"


def test_main_writes_the_table_and_the_status_and_exits_3_when_not_ok(tmp_path, monkeypatch):
    now, prev = tmp_path / "now", tmp_path / "prev"
    _set(now, "KIDNEY_cross", hmap=HMAP[:2]); _set(prev, "KIDNEY_cross", hmap=HMAP)
    monkeypatch.setattr(sys, "argv", ["check_panel_build.py", "--panel", str(now), "--previous", str(prev)])
    with pytest.raises(SystemExit) as e:
        cpb.main()
    assert e.value.code == 3
    assert (now / "build_check.status").read_text().startswith("cross harmonization not confirmed")
    assert len(pd.read_csv(now / "build_check.tsv", sep="\t")) == 1
    # and 0 with the status "ok" once the map is the reviewed one
    _set(tmp_path / "now2", "KIDNEY_cross", hmap=HMAP)
    monkeypatch.setattr(sys, "argv", ["check_panel_build.py", "--panel", str(tmp_path / "now2"), "--previous", str(prev)])
    with pytest.raises(SystemExit) as e:
        cpb.main()
    assert e.value.code == 0 and (tmp_path / "now2" / "build_check.status").read_text() == "ok\n"
