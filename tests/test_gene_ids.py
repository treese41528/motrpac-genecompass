"""Stages 2-3: gene-ID normalization, biotypes, the symbol lookup, and finding raw counts.

The data tests (pytest -m data) check the real symbol lookup and every QC'd corpus file.
"""
import pickle

import anndata as ad
import h5py
import numpy as np
import pytest
import scipy.sparse as sp

import gene_utils as gu
from verify_raw_counts import check_file
from verify_symbol_lookup import classify_gene_format

STANDARD_GENES = ['actb', 'gapdh', 'ube2m', 'dhfr', 'gnai3', 'pin1', 'ccl5', 'apoe', 'tp53',
                  'brca1', 'egfr', 'vegfa', 'il6', 'tnf', 'ppia', 'rplp0', 'hprt1', 'b2m',
                  'tbp', 'ywhaz', 'sdha']


# ------------------------------------------------------------------ IDs
def test_ensembl_id_normalization():
    assert gu.normalize_ensembl_id("  ensrnog00000029043.3  ") == "ENSRNOG00000029043"
    assert gu.is_rat_ensembl_id("ENSRNOG00000029043.12")
    assert not gu.is_rat_ensembl_id("ENSRNOG0000002904")          # 10 digits
    assert gu.is_human_ensembl_id("ENSG00000139618")
    assert gu.is_mouse_ensembl_id("ENSMUSG00000017167")
    assert not gu.is_human_ensembl_id("ENSMUSG00000017167")


@pytest.mark.parametrize("gene, namespace", [
    ("ENSRNOG00000029043", "ensembl_rat"), ("ENSRNOG00000029043.2", "ensembl_rat"),
    ("ENSG00000139618", "ensembl_human"), ("ENSMUSG00000017167", "ensembl_mouse"),
    ("RGD1311892", "rgd"), ("NM_001007000", "refseq"), ("LOC100909441", "loc"),
    ("AABR07000046.1", "contig"), ("RF00001", "rfam"), ("Apoe", "symbol"),
    ("1234567_at", "probe"), ("chr1_12345", "genomic_coord"), ("9", "unknown"),
])
def test_namespace_detection(gene, namespace):
    assert gu.detect_namespace(gene) == namespace


def test_non_gene_patterns():
    assert gu.is_non_gene_pattern("JACYVU010000738") == "jacyvu_scaffold"
    assert gu.is_non_gene_pattern("U6") == "small_rna"
    assert gu.is_non_gene_pattern("Apoe") is None


@pytest.mark.parametrize("raw, canonical", [
    ("protein-coding", "protein_coding"), ("protein-coding gene", "protein_coding"),
    ("Protein Coding", "protein_coding"), ("miRNA", "mirna"), ("lincRNA", "lncrna"),
    ("processed_pseudogene", "pseudogene"), ("Mt_rRNA", "rrna"), ("weird_type", "weird_type"),
])
def test_biotype_canonicalization(raw, canonical):
    assert gu.normalize_biotype(raw) == canonical


def test_kept_biotypes_are_the_genecompass_core():
    keep = frozenset({"protein_coding", "lncrna", "mirna"})
    assert gu.is_kept_biotype("protein-coding gene", keep)
    assert gu.is_kept_biotype("lincRNA", keep)
    assert not gu.is_kept_biotype("processed_pseudogene", keep)


@pytest.mark.parametrize("gene, fmt", [
    ("ENSRNOG00000029043", "ensembl_rat"), ("LOC100909441", "loc_id"),
    ("NM_001007000", "refseq"), ("Apoe", "symbol_titlecase"), ("APOE", "symbol_allcaps"),
    ("12345", "entrez"), ("apoe", "symbol_lower"),
])
def test_verify_symbol_lookup_classifier(gene, fmt):
    assert classify_gene_format(gene) == fmt


def test_symbol_lookup_keys_are_lowercased(tmp_path):
    p = tmp_path / "lookup.pickle"
    p.write_bytes(pickle.dumps({"Apoe": "ENSRNOG00000018454", " GAPDH ": "ENSRNOG00000018630",
                                None: "x", "": "y"}))
    assert gu.load_symbol_lookup(str(p)) == {"apoe": "ENSRNOG00000018454",
                                             "gapdh": "ENSRNOG00000018630"}


# ------------------------------------------------------------------ raw counts
def _adata(X, raw_X=None):
    a = ad.AnnData(X=X)
    if raw_X is not None:
        a.raw = ad.AnnData(X=raw_X, var=a.var.copy())
    return a


def test_raw_counts_in_X_are_used_as_is():
    a = _adata(sp.csr_matrix(np.array([[0, 3], [5, 1]], dtype=np.float32)))
    out, used_raw = gu.raw_counts_adata(a)
    assert out is a and not used_raw


def test_log_normalized_X_falls_back_to_raw_slot():
    counts = sp.csr_matrix(np.array([[0, 3], [5, 1]], dtype=np.float32))
    logged = counts.copy(); logged.data = np.log1p(logged.data)
    out, used_raw = gu.raw_counts_adata(_adata(logged, counts))
    assert used_raw
    assert np.array_equal(out.X.toarray(), counts.toarray())


def test_non_integer_X_without_raw_is_refused():
    with pytest.raises(ValueError):
        gu.raw_counts_adata(_adata(np.array([[0.5, 1.2]], dtype=np.float32)))


def _write_h5ad(path, raw_X=None):
    a = _adata(sp.csr_matrix(np.ones((3, 2), np.float32)), raw_X)
    a.var_names = ["ENSRNOG00000000001", "ENSRNOG00000000002"]
    if raw_X is not None:
        a.raw = ad.AnnData(X=raw_X, var=a.var.copy())
    a.write_h5ad(path)


def test_verify_raw_counts_flags_each_case(tmp_path):
    ok = tmp_path / "GSE1_sample0.h5ad"
    _write_h5ad(ok, sp.csr_matrix(np.array([[1, 0], [2, 3], [0, 4]], np.float32)))
    rec, issues = check_file(str(ok))
    assert rec["status"] == "OK" and rec["n_cells"] == 3 and rec["gene_id_prefix"].startswith("ENSRNOG")
    assert rec["study"] == "GSE1" and not issues

    frac = tmp_path / "GSE2_sample0.h5ad"
    _write_h5ad(frac, sp.csr_matrix(np.array([[1.5, 0], [2, 3], [0, 4]], np.float32)))
    assert check_file(str(frac))[0]["status"] == "NOT_INTEGER"

    no_raw = tmp_path / "GSE3_sample0.h5ad"
    _write_h5ad(no_raw)
    assert check_file(str(no_raw))[0]["status"] == "NO_RAW"

    broken = tmp_path / "GSE4_sample0.h5ad"
    broken.write_text("not hdf5")
    rec, issues = check_file(str(broken))
    assert rec["status"] == "ERROR" and issues


# ------------------------------------------------------------------ real files (pytest -m data)
@pytest.mark.data
def test_real_symbol_lookup_covers_standard_genes(config, require):
    p = require(gu.resolve_path(config, "data/references/biomart/rat_symbol_lookup.pickle"))
    lookup = gu.load_symbol_lookup(str(p))
    missing = [g for g in STANDARD_GENES if g not in lookup]
    assert not missing
    assert all(gu.is_rat_ensembl_id(lookup[g]) for g in STANDARD_GENES)


def _integer_sample(ds):
    """True when a sample of an h5ad matrix (dense dataset or sparse group) is all integers."""
    vals = ds["data"][:20000] if isinstance(ds, h5py.Group) else ds[:20].ravel()
    vals = np.asarray(vals, dtype=np.float64)
    return bool(np.all(np.abs(vals - np.round(vals)) <= 1e-6))


@pytest.mark.data
def test_every_corpus_file_holds_raw_counts(config, require):
    require(gu.resolve_path(config, config["paths"]["qc_h5ad_dir"]))
    files = gu.qc_corpus_files(config)
    assert files
    bad = []
    for f in files:
        with h5py.File(f, "r") as h:
            if not (_integer_sample(h["X"]) or ("raw" in h and _integer_sample(h["raw/X"]))):
                bad.append(f.name)
    assert not bad, f"{len(bad)} files without integer counts, e.g. {bad[:5]}"
