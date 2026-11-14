"""The Stage 8 gene-exclusion list is built by rule from the gene table of the release the pipeline uses
(deconvolution/build_exclude_genes.py; docs/audit/remediation/STEP3_PREREG.md, Amendment 1)."""
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "deconvolution"))
import build_exclude_genes as b                     # noqa: E402


@pytest.mark.parametrize("description, chrom, expected", [
    ("ribosomal protein L10A [Source:RGD Symbol;Acc:1]", "20", "ribosomal"),
    ("mitochondrial ribosomal protein S17 [Source:RGD]", "12", "ribosomal"),
    ("large ribosomal subunit protein uL14-like [Source:RGD]", "5", "ribosomal"),        # newer naming
    ("60S ribosomal protein L35-like [Source:RGD]", "3", "ribosomal"),                   # LOC symbol, found by description
    ("FAU ubiquitin like and ribosomal protein S30 fusion [Source:RGD]", "1", "ribosomal"),
    ("ribosomal protein S19 binding protein 1 [Source:RGD]", "7", "ribosomal"),          # in BayesPrism's other_Rb
    ("ribosomal protein S6 kinase B1 [Source:RGD]", "10", None),                         # a kinase
    ("ribosomal protein S6 kinase related [Source:RGD]", "10", None),
    ("alpha-ketoglutarate dehydrogenase subunit 4 [Source:RGD]", "2", None),             # Kgd4, once Mrps36
    ("chorionic somatomammotropin hormone 1 [Source:NCBI gene]", "17", None),            # symbol Rpl-1
    ("hemoglobin subunit beta [Source:NCBI gene]", "1", "hemoglobin"),
    ("alpha hemoglobin stabilizing protein [Source:RGD]", "1", None),
    ("mitochondrially encoded cytochrome b [Source:RGD]", "MT", "mito"),
    ("", "MT", "mito"),                                                                  # MT tRNA with no description
    ("actin, beta [Source:RGD]", "12", None),
])
def test_category(description, chrom, expected):
    assert b.category(description, chrom) == expected


def test_build_is_keyed_by_gene_id_not_symbol(tmp_path):
    t = tmp_path / "genes.tsv"
    pd.DataFrame({
        "Gene stable ID": ["ENSRNOG1", "ENSRNOG2", "ENSRNOG3", "ENSRNOG3"],
        "Gene name": ["LOC1", "Rps6kb1", "Rpl7", "Rpl7"],
        "Gene type": ["protein_coding"] * 4,
        "Chromosome/scaffold name": ["1", "10", "5", "5"],
        "Gene description": ["40S ribosomal protein S3a-like", "ribosomal protein S6 kinase B1",
                             "ribosomal protein L7", "ribosomal protein L7"],
    }).to_csv(t, sep="\t", index=False)
    out = b.build(t)
    assert list(out.feature_ID) == ["ENSRNOG1", "ENSRNOG3"]          # the kinase is kept; duplicates collapse
    assert set(out.category) == {"ribosomal"}


@pytest.mark.data
def test_committed_v2_list_is_what_the_rule_builds():
    gene_info = ROOT / "data/references/biomart/rat_gene_info.tsv"
    if not gene_info.exists():
        pytest.skip(f"not on this machine: {gene_info}")
    built = b.build(gene_info)
    committed = pd.read_csv(ROOT / "deconvolution/reference/rat_exclude_genes_v2.tsv", sep="\t", dtype=str).fillna("")
    assert list(built.feature_ID) == list(committed.feature_ID)
    assert list(built.category) == list(committed.category)
    assert not committed.gene_symbol.str.match(r"(?i)^rps6k").any()
