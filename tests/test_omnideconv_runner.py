"""The cross-check runner (deconvolution/R/run_omnideconv.R) and the scoring message
(docs/audit/remediation/STEP4_PREREG.md V6; audit findings 5.5 and 5.6).

The R part runs tests/r/test_omnideconv_settings.R and is skipped where Rscript is not on PATH. On this cluster
R comes from a module: `source deconvolution/setup/site_env.sh` (with PROJECT_ROOT set) before pytest.
"""
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import score_validation as sv

ROOT = Path(__file__).resolve().parents[1]
RUNNER = (ROOT / "deconvolution/R/run_omnideconv.R").read_text()
RSCRIPT = shutil.which("Rscript")


# ---------------------------------------------------------------- the runner's source: what it passes and records
def test_runner_passes_both_dwls_cutoffs_and_the_method_to_the_signature_build():
    call = re.search(r"sig <- omnideconv::build_model\((.*?)\)\n", RUNNER, re.S).group(1)
    assert 'method = "dwls"' in call and "dwls_method = dwls_method" in call
    assert "pval_cutoff = dwls$pval_cutoff" in call and "diff_cutoff = dwls$diff_cutoff" in call
    assert "ncores = n.cores" in call
    assert "dwls <- parse_dwls_settings()" in RUNNER          # from the environment, defaults = the earlier runs


def test_runner_writes_a_provenance_record_with_the_registered_fields():
    call = RUNNER[RUNNER.index("write_provenance(out_dir"):]
    assert "md5 = TRUE" in call                               # inputs with md5
    for field in ("methods = I(methods)", "method_settings = settings", "method_result = method_record",
                  "packages = package_versions(", "reference_cap = list(max_cells_per_type = ref_max, seed = CAP_SEED",
                  "cells_dropped_per_type", "in_memory_patches", "guard_fired_for", "runner_md5", "gene_cleanup"):
        assert field in call, field
    for inp in ("reference_counts.mtx", "cells_meta.tsv", "genes.tsv", "excl_path"):
        assert inp in call, inp
    # one file per invocation (the methods of a set run in separate jobs that share the folder)
    assert 'sprintf("provenance_run_omnideconv_%s.json", paste(methods, collapse = "+"))' in call
    # written before the script stops on "all methods failed"
    assert RUNNER.index("write_provenance(out_dir") < RUNNER.index('stop("all omnideconv methods failed")')


def test_runner_never_writes_the_cibersortx_credentials_to_the_record():
    call = RUNNER[RUNNER.index("write_provenance(out_dir"):]
    assert "cx_email" not in call and "cx_token" not in call and "CIBERSORTX_TOKEN" not in call
    settings = RUNNER[RUNNER.index("settings <- list()"):RUNNER.index("runner_files <-")]
    assert "cx_email" not in settings and "cx_token" not in settings


def test_cap_seed_is_the_one_the_record_states():
    assert "CAP_SEED <- 1L" in RUNNER and "set.seed(CAP_SEED)" in RUNNER
    assert 'Sys.getenv("REF_MAX_CELLS_PER_TYPE", unset = "2000")' in RUNNER


# ---------------------------------------------------------------- the R functions
@pytest.mark.skipif(RSCRIPT is None, reason="Rscript not on PATH (source deconvolution/setup/site_env.sh first)")
def test_r_settings_guard_and_provenance(tmp_path):
    env = dict(os.environ, R_LIBS_USER=str(ROOT / "R_libs"), R_PROFILE="/dev/null", R_PROFILE_USER="/dev/null",
               R_ENVIRON="/dev/null", R_ENVIRON_USER="/dev/null", TMPDIR=str(tmp_path))
    r = subprocess.run([RSCRIPT, "--no-init-file", "--no-site-file", str(ROOT / "tests/r/test_omnideconv_settings.R"),
                        str(ROOT), str(tmp_path)], capture_output=True, text=True, env=env, timeout=600)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    assert "ALL OK" in r.stdout
    rec = json.loads((tmp_path / "provenance_test.json").read_text())
    assert re.fullmatch(r"[0-9a-f]{40}", rec["git_commit"])
    assert isinstance(rec["git_modified_tracked_files"], list)
    got = {Path(i["path"]).name: i for i in rec["inputs"]}
    assert got["input.txt"]["md5"] == hashlib.md5((tmp_path / "input.txt").read_bytes()).hexdigest()
    assert got["missing.txt"]["exists"] is False
    d = rec["params"]["method_settings"]["dwls"]
    assert d["dwls_method"] == "mast_optimized" and d["pval_cutoff"] == 0.05 and d["diff_cutoff"] == 0.5
    assert rec["params"]["methods"] == ["dwls"] and rec["params"]["reference_cap"] == {"max_cells_per_type": 2000, "seed": 1}
    assert rec["params"]["in_memory_patches"]["dwls"]["guard_fired_for"] == []
    if "dwls_installed FALSE" in r.stdout:
        pytest.skip("settings, guard and provenance pass; the DWLS package is not installed here, so the check "
                    "that its two builders each have one guard site did not run")


# ---------------------------------------------------------------- scoring: a type a method does not estimate
def _stage(tmp_path, est_cols, const=None):
    """A stage folder with 3 true types over 8 mixtures and an estimate file holding `est_cols`."""
    rng = np.random.default_rng(3)
    truth = pd.DataFrame(rng.dirichlet(np.ones(3), size=8), columns=["A cells", "B cells", "C cells"])
    (tmp_path / "mixtures").mkdir(); (tmp_path / "results").mkdir()
    t = truth.add_prefix("rnafrac__"); t.insert(0, "mixture", [f"mix{i + 1}" for i in range(8)])
    t.to_csv(tmp_path / "mixtures" / "true_fractions.tsv", sep="\t", index=False)
    est = (truth[est_cols] + rng.normal(0, 0.02, size=(8, len(est_cols)))).clip(lower=0)
    if const:
        est[const] = 0.0
    est.index = [f"s{i}" for i in range(8)]
    est.to_csv(tmp_path / "results" / "fractions_x.csv")
    return truth, est


def _score(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["score_validation.py", "--stage-dir", str(tmp_path),
                                      "--est-file", "results/fractions_x.csv", "--tag", "x"])
    sv.main()
    out = capsys.readouterr().out
    m = pd.read_csv(tmp_path / "scores" / "metrics_x.tsv", sep="\t")
    o = pd.read_csv(tmp_path / "scores" / "overall_metrics_x.tsv", sep="\t").set_index("metric")
    return out, m, o


def test_message_says_a_missing_type_is_left_out_not_set_to_zero():
    msg = sv.not_estimated_note(["C cells"])
    assert "left out" in msg and "not scored as 0" in msg and "C cells" in msg
    assert "set to 0" not in msg


def test_a_type_the_method_does_not_estimate_is_left_out_of_every_score(tmp_path, monkeypatch, capsys):
    truth, est = _stage(tmp_path, ["A cells", "B cells"])
    out, m, o = _score(tmp_path, monkeypatch, capsys)
    assert "did not estimate: ['C cells']" in out and "left out" in out and "set to 0" not in out
    assert sorted(m["cell_type"]) == ["A cells", "B cells"]                  # no row for the missing type
    # the pooled fit is over the two estimated types only: scoring the third as 0 would give another number
    pooled_true = np.concatenate([truth["A cells"], truth["B cells"]])
    pooled_est = np.concatenate([est["A cells"], est["B cells"]])
    assert o.loc["overall_pooled_rmse", "value"] == pytest.approx(np.sqrt(np.mean((pooled_true - pooled_est) ** 2)))
    as_zero = np.sqrt(np.mean((np.concatenate([pooled_true, truth["C cells"]]) -
                               np.concatenate([pooled_est, np.zeros(8)])) ** 2))
    assert abs(o.loc["overall_pooled_rmse", "value"] - as_zero) > 1e-3
    assert o.loc["median_pearson", "value"] == pytest.approx(m["pearson_r"].median())
    assert o.loc["n_types_truth", "value"] == 3 and o.loc["n_types_scored", "value"] == 2
    assert o.loc["n_types_not_estimated", "value"] == 1 and "C cells" in o.loc["n_types_not_estimated", "note"]


def test_a_constant_estimate_keeps_its_rmse_and_leaves_the_median(tmp_path, monkeypatch, capsys):
    truth, est = _stage(tmp_path, ["A cells", "B cells", "C cells"], const="C cells")
    out, m, o = _score(tmp_path, monkeypatch, capsys)
    row = m.set_index("cell_type").loc["C cells"]
    assert np.isnan(row["pearson_r"]) and row["rmse"] == pytest.approx(np.sqrt(np.mean(truth["C cells"] ** 2)))
    assert o.loc["median_pearson", "value"] == pytest.approx(m["pearson_r"].dropna().median())
    assert o.loc["n_types_constant_estimate", "value"] == 1 and o.loc["n_types_not_estimated", "value"] == 0
    assert "estimated as a constant" in out and "C cells" in out


def test_a_full_estimate_scores_as_before_and_reports_nothing_left_out(tmp_path, monkeypatch, capsys):
    truth, est = _stage(tmp_path, ["A cells", "B cells", "C cells"])
    out, m, o = _score(tmp_path, monkeypatch, capsys)
    assert "[note]" not in out and len(m) == 3
    assert o.loc["n_types_not_estimated", "value"] == 0 and o.loc["n_types_constant_estimate", "value"] == 0
    # the six metrics that existed before are still there, by name
    for k in ("overall_pooled_pearson", "overall_pooled_rmse", "macro_pearson", "median_pearson",
              "separable_pooled_pearson", "separable_pooled_rmse"):
        assert k in o.index
