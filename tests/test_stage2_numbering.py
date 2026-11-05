"""Stage 2 sample numbering: pinned ids survive any discovery order; new files get sorted ids after them."""
import pytest

P = pytest.importorskip("preprocess_training_matrices")


def _cfg(tmp_path, rows):
    table = tmp_path / "order.tsv"
    table.write_text("# comment\nstudy\tsample_id\tsource\tformat\tstatus\n"
                     + "".join(f"GSE1\tGSE1_sample{i}\t{src}\tmtx\tsuccess\n" for i, src in rows))
    return {"_project_root": tmp_path, "paths": {"stage2_sample_order": str(table)},
            "harvesting": {"data_root": str(tmp_path / "raw")}}


def _files(tmp_path, names):
    return [(str(tmp_path / "raw" / n), "mtx") for n in names]


def test_pinned_ids_ignore_discovery_order(tmp_path):
    cfg = _cfg(tmp_path, [(0, "s/GSE1/b.mtx"), (1, "s/GSE1/a.mtx")])
    for order in (["s/GSE1/a.mtx", "s/GSE1/b.mtx"], ["s/GSE1/b.mtx", "s/GSE1/a.mtx"]):
        got = {sid: fp.rsplit("/", 1)[1] for sid, fp, _ in P.number_matrix_files("GSE1", _files(tmp_path, order), cfg)}
        assert got == {"GSE1_sample0": "b.mtx", "GSE1_sample1": "a.mtx"}


def test_duplicates_gz_twins_and_new_files(tmp_path):
    # the canonical run numbered a doubly-discovered file twice; x.mtx.gz stands in for a pinned x.mtx
    cfg = _cfg(tmp_path, [(0, "s/GSE1/a.mtx"), (1, "s/GSE1/a.mtx"), (2, "s/GSE1/c.mtx")])
    found = _files(tmp_path, ["s/GSE1/z_new.mtx", "s/GSE1/a.mtx", "s/GSE1/c.mtx.gz", "s/GSE1/a.mtx", "s/GSE1/m_new.mtx"])
    got = [(sid, fp.rsplit("/", 1)[1]) for sid, fp, _ in P.number_matrix_files("GSE1", found, cfg)]
    assert got == [("GSE1_sample0", "a.mtx"), ("GSE1_sample1", "a.mtx"), ("GSE1_sample2", "c.mtx.gz"),
                   ("GSE1_sample3", "m_new.mtx"), ("GSE1_sample4", "z_new.mtx")]


def test_unlisted_study_is_sorted(tmp_path):
    cfg = _cfg(tmp_path, [])
    got = [sid + ":" + fp.rsplit("/", 1)[1]
           for sid, fp, _ in P.number_matrix_files("GSE9", _files(tmp_path, ["s/GSE9/b.mtx", "s/GSE9/a.mtx"]), cfg)]
    assert got == ["GSE9_sample0:a.mtx", "GSE9_sample1:b.mtx"]
