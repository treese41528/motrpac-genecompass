"""Step 4, V4: the myeloid replication's matched null and reading (STEP4_PREREG.md, addendum to V4)."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("module_a_v4", ROOT / "analysis" / "audit_remediation" / "module_a_v4.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
SPI1, CEBPA = "ENSRNOG00000012172", "ENSRNOG00000010918"


def test_matched_null_is_within_fifteen_percent_and_excludes_the_factors():
    values = {SPI1: 1.0, CEBPA: 1.05, "a": 0.86, "b": 0.84, "c": 1.149, "d": 1.16, "e": 2.0}
    assert m.matched(values, values[SPI1], {SPI1, CEBPA}) == ["a", "c"]          # within 15%; the other factor left out


def _score(tmp_path, spi1_shift, spi1_cells=50, null=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)):
    genes = [f"g{i}" for i in range(len(null))]
    pd.DataFrame({"gene": [SPI1] + genes, "mean_value": [1.0] + [1.0 + 0.001 * i for i in range(len(null))],
                  "n_cells_present": [spi1_cells] + [50] * len(null), "role": ["factor:Spi1"] + ["null:Spi1"] * len(null),
                  "n_cells_of_type": 50}).to_csv(tmp_path / "panel.tsv", sep="\t", index=False)
    pd.DataFrame({"gene": [SPI1] + genes, "cell_type": "Classical monocytes", "n_cells": [spi1_cells] + [50] * len(null),
                  "mean_cell_shift": [spi1_shift] + list(null), "median_cell_shift": 0, "top_targets": ""}
                 ).to_csv(tmp_path / "results.tsv", sep="\t", index=False)
    m.score(SimpleNamespace(results=str(tmp_path / "results.tsv"), panel=str(tmp_path / "panel.tsv"), out_dir=str(tmp_path)))
    return pd.read_csv(tmp_path / "myeloid_replication.tsv", sep="\t").set_index("factor")


def test_reading_follows_the_ninetieth_percentile(tmp_path):
    t = _score(tmp_path, 10.5)
    assert t.loc["Spi1", "percentile"] == 100 and t.loc["Spi1", "reading"] == "replicates"
    t = _score(tmp_path, 9.5)                       # above 9 of 10: exactly the 90th percentile is not above it
    assert t.loc["Spi1", "percentile"] == 90 and t.loc["Spi1", "reading"] == "does not replicate"
    assert t.loc["Cebpa", "reading"].startswith("not testable (coverage")   # absent from the results


def test_fewer_than_twenty_cells_is_not_testable(tmp_path):
    t = _score(tmp_path, 99.0, spi1_cells=19)
    assert t.loc["Spi1", "reading"].startswith("not testable (power")
