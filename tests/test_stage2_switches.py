"""Stage 2 and catalog-merge switches for a future corpus (docs/audit/remediation/STEP5_PREREG.md).

Every switch is off by default, and off must mean the behaviour the v2 corpus was built with.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
P = pytest.importorskip("preprocess_training_matrices")
ad = pytest.importorskip("anndata")
sys.path.insert(0, str(ROOT / "pipeline" / "01_data_harvesting"))
cds = pytest.importorskip("combine_data_sources")

SWITCHES = ["mito_by_id", "apply_max_mito_fraction", "outlier_on_n_genes", "walk_each_file_once"]


def test_every_switch_is_off_in_the_committed_config():
    cfg = yaml.safe_load(open(ROOT / "config" / "pipeline_config.yaml"))
    pp = cfg["gene_universe"]["preprocessing"]
    assert {k: pp[k] for k in SWITCHES} == dict.fromkeys(SWITCHES, False)
    assert cfg["harvesting"]["prefer_single_cell_on_conflict"] is False


def _cells(total, mito, genes=("g1", "g2", "mt")):
    """Cells with `total` counts, of which `mito` on the gene 'mt' and the rest split over g1 and g2."""
    total, mito = np.asarray(total, float), np.asarray(mito, float)
    X = np.column_stack([(total - mito) / 2, (total - mito) / 2, mito]).astype(np.float32)
    a = ad.AnnData(X)
    a.var_names = list(genes)
    a.obs_names = [f"c{i}" for i in range(len(total))]
    return a


def test_max_mito_filter_drops_cells_above_the_cut_and_keeps_the_boundary():
    a = _cells([100, 100, 100, 100], [0, 15, 16, 60])
    kept = P.max_mito_filter(a, {"mt"}, 0.15)
    assert list(kept.obs_names) == ["c0", "c1"]
    # with no mitochondrial gene recognised nothing is dropped: the cut needs mito_by_id in ID-keyed files
    assert P.max_mito_filter(a, set(), 0.15).n_obs == 4


def test_normal_filter_default_is_the_v2_rule_and_the_switch_adds_gene_number():
    rng = np.random.default_rng(0)
    n = 60
    X = np.zeros((n, 40), dtype=np.float32)
    for i in range(n):                                   # 20 expressed genes per cell, ~100 counts each cell
        X[i, rng.choice(40, 20, replace=False)] = 5
    X[0, :] = 100.0 / 40                                 # same total counts, all 40 genes expressed
    a = ad.AnnData(X)
    a.var_names = [f"g{i}" for i in range(40)]
    a.obs_names = [f"c{i}" for i in range(n)]
    base = P.normal_filter(a, set())
    assert base.n_obs == n                               # total counts and mito are flat: nothing is an outlier
    with_genes = P.normal_filter(a, set(), outlier_on_n_genes=True)
    assert "c0" not in with_genes.obs_names and with_genes.n_obs == n - 1


def test_mito_ids_by_chromosome_reads_the_gene_table(tmp_path):
    t = tmp_path / "rat_gene_info.tsv"
    t.write_text("Gene stable ID\tGene name\tGene type\tChromosome/scaffold name\tGene description\n"
                 "ENSRNOG1\tMt-nd1\tprotein_coding\tMT\t\nENSRNOG2\tActb\tprotein_coding\t12\t\n"
                 "ENSRNOG3\t\tMt_tRNA\tMT\t\n")
    cfg = {"_project_root": tmp_path, "biomart": {"rat_gene_info": str(t)}}
    assert P.load_mito_ids_by_chromosome(cfg) == {"ENSRNOG1", "ENSRNOG3"}
    t.write_text("Gene stable ID\tGene name\tGene type\tChromosome/scaffold name\tGene description\n"
                 "ENSRNOG2\tActb\tprotein_coding\t12\t\n")
    with pytest.raises(ValueError, match="no chromosome-MT gene"):
        P.load_mito_ids_by_chromosome(cfg)


def test_symbol_pattern_finds_no_mitochondrial_gene_in_an_id_keyed_file():
    """Finding 1.3 as a test: why mito_by_id exists."""
    assert P.identify_mito_genes(["ENSRNOG00000030644", "ENSRNOG00000031033"]) == []
    assert P.identify_mito_genes(["Mt-nd1", "mt-Co1", "Actb"]) == ["Mt-nd1", "mt-Co1"]


def _study(tmp_path):
    s = tmp_path / "GSE1"
    (s / "GSE1_RAW").mkdir(parents=True)
    (s / "GSE1_RAW" / "GSM1_counts.h5").write_text("x")
    (s / "GSE1_RAW" / "GSM2_counts.h5").write_text("x")
    (s / "top_counts.h5").write_text("x")
    return s


def test_default_walk_reads_raw_twice_and_the_switch_reads_it_once(tmp_path):
    s = _study(tmp_path)
    twice = [Path(f).name for f, _ in P.find_matrix_files(str(s))]
    once = [Path(f).name for f, _ in P.find_matrix_files(str(s), once=True)]
    assert sorted(twice) == ["GSM1_counts.h5", "GSM1_counts.h5", "GSM2_counts.h5", "GSM2_counts.h5", "top_counts.h5"]
    assert sorted(once) == ["GSM1_counts.h5", "GSM2_counts.h5", "top_counts.h5"]
    # the files found once are the first occurrences of the default walk, in the same order
    seen, first = set(), []
    for f in twice:
        if f not in seen:
            seen.add(f); first.append(f)
    assert once == first


def test_pinned_ids_keep_their_files_when_each_file_is_read_once(tmp_path):
    """The pinned id of a second copy is not produced; every other pinned id is unchanged."""
    table = tmp_path / "order.tsv"
    table.write_text("study\tsample_id\tsource\tformat\tstatus\n"
                     "GSE1\tGSE1_sample0\ts/GSE1/a.h5\th5\tsuccess\nGSE1\tGSE1_sample1\ts/GSE1/b.h5\th5\tsuccess\n"
                     "GSE1\tGSE1_sample2\ts/GSE1/a.h5\th5\tsuccess\nGSE1\tGSE1_sample3\ts/GSE1/c.h5\th5\tsuccess\n")
    cfg = {"_project_root": tmp_path, "paths": {"stage2_sample_order": str(table)},
           "harvesting": {"data_root": str(tmp_path / "raw")}}
    f = lambda names: [(str(tmp_path / "raw" / n), "h5") for n in names]
    both = P.number_matrix_files("GSE1", f(["s/GSE1/a.h5", "s/GSE1/b.h5", "s/GSE1/c.h5", "s/GSE1/a.h5"]), cfg)
    once = P.number_matrix_files("GSE1", f(["s/GSE1/a.h5", "s/GSE1/b.h5", "s/GSE1/c.h5"]), cfg)
    ids = lambda rows: {sid: Path(fp).name for sid, fp, _ in rows}
    assert ids(both) == {"GSE1_sample0": "a.h5", "GSE1_sample1": "b.h5", "GSE1_sample2": "a.h5", "GSE1_sample3": "c.h5"}
    assert ids(once) == {"GSE1_sample0": "a.h5", "GSE1_sample1": "b.h5", "GSE1_sample3": "c.h5"}


ROWS = [{"accession": "GSE1", "data_type": "single_cell", "n": 1}, {"accession": "GSE2", "data_type": "bulk", "n": 2},
        {"accession": "GSE1", "data_type": "bulk", "n": 3}, {"accession": "GSE3", "data_type": "bulk", "n": 4},
        {"accession": "GSE3", "data_type": "single_cell", "n": 5}]


def test_catalog_merge_default_keeps_the_later_row():
    got = cds.index_catalog(ROWS)
    assert {a: r["n"] for a, r in got.items()} == {"GSE1": 3, "GSE2": 2, "GSE3": 5}
    assert got["GSE1"]["data_type"] == "bulk"            # finding 1.1: the single-cell row is lost


def test_catalog_merge_switch_keeps_the_single_cell_row():
    got = cds.index_catalog(ROWS, prefer_single_cell=True)
    assert {a: r["data_type"] for a, r in got.items()} == {"GSE1": "single_cell", "GSE2": "bulk", "GSE3": "single_cell"}
    assert got["GSE1"]["n"] == 1 and got["GSE3"]["n"] == 5
