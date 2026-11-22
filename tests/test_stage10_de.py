"""Stage 10 changes registered in docs/audit/remediation/STEP3_PREREG.md section 6.

Covers the Python side: the comparison of the headline DE with the technical-covariate reruns (6.4), the
composition-confound statuses (finding 7.6), the options of run_stage10.py that must reach the R step
(--alpha, finding 7.10; --tech-covariates and --covariate-variants, 6.4), and the sex-consistency call rule
that repfdr_call_sources.py re-derives for both table layouts (6.2). The R step itself is exercised by the
SLURM runs recorded under data/audit_remediation/step3/de_regression/.
"""
import re
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import build_composition_confound_table as confound
import compare_de_runs as cdr
import repfdr_call_sources as rcs
import run_stage10
from celltype_names import safe

ROOT = Path(__file__).resolve().parents[1]
GENES = [f"ENSRNOG{i:011d}" for i in range(50)]


# ---------------------------------------------------------------- compare_de_runs.py

def block(n_sig, order=None, drop=()):
    """A per-gene table of 50 genes: dose p rising with the gene index (or following `order`), the first
    n_sig genes of the index dose-significant. `drop` removes genes, as a rerun that did not test them."""
    p = (np.arange(50) + 1) / 1000.0
    if order is not None:
        p = p[np.asarray(order)]
    t = pd.DataFrame({"gene": GENES, "P_fisher": p, "FDR_IHW": np.where(np.arange(50) < n_sig, 0.01, 0.5)})
    return t[~t["gene"].isin(drop)]


def write_run(d, blocks, hotspots=(), skipped=(), alpha="0.05", covariates="none"):
    """A minimal pseudobulk_de folder: de_summary.tsv, de_methods.tsv, celltype_files.tsv and the tables.
    blocks: {(tissue, cell_type): table}; skipped: blocks the DE did not analyse (summary row only)."""
    rows, man = [], []
    for (tis, ct), tab in blocks.items():
        (d / tis).mkdir(parents=True, exist_ok=True)
        rel = f"{tis}/de__{safe(ct)}.tsv"
        tab.to_csv(d / rel, sep="\t", index=False)
        man.append({"tissue": tis, "cell_type": ct, "file": rel})
        rows.append({"tissue": tis, "cell_type": ct, "is_hotspot": (tis, ct) in hotspots, "mean_fraction": 0.2,
                     "low_fraction": False, "n_samples_dropped_covariate": 0, "status": "ok"})
    for tis, ct in skipped:
        rows.append({"tissue": tis, "cell_type": ct, "is_hotspot": (tis, ct) in hotspots, "mean_fraction": 0.001,
                     "low_fraction": np.nan, "n_samples_dropped_covariate": 0,
                     "status": "SKIP: degenerate (median libsize 10.0 < 1000)"})
    summ = pd.DataFrame(rows)
    summ["is_hotspot"] = summ["is_hotspot"].map({True: "TRUE", False: "FALSE"})          # as R writes it
    summ["low_fraction"] = summ["low_fraction"].map({True: "TRUE", False: "FALSE"})
    summ.to_csv(d / "de_summary.tsv", sep="\t", index=False, na_rep="NA")
    pd.DataFrame(man).to_csv(d / "celltype_files.tsv", sep="\t", index=False)
    (d / "de_methods.tsv").write_text(f"alpha\t{alpha}\ntechnical_covariates\t{covariates}\n")
    return d


H1, H2, H3, H4, H5 = (("BLOOD", "Classical monocytes"), ("BLOOD", "NK cells"), ("HYPOTH", "OPC"),
                      ("SKMVL", "α/β mix: cells"), ("LUNG", "Goblet cells"))
N1 = ("BAT", "ASPC")
HOT = {H1, H2, H3, H4, H5}


@pytest.fixture
def runs(tmp_path):
    rev = np.arange(50)[::-1]
    head = write_run(tmp_path / "pseudobulk_de",
                     {H1: block(10), H2: block(10), H3: block(0), H4: block(10), N1: block(10)},
                     hotspots=HOT, skipped=[H5])
    a = write_run(tmp_path / "pseudobulk_de_covA",
                  {H1: block(9), H2: block(10), H3: block(5), H4: block(10, order=rev), N1: block(12)},
                  hotspots=HOT, skipped=[H5], covariates="rin, pct_globin")
    # In B, H1 keeps exactly 8 of its 10 (two of them are not tested at all) and H2 keeps 7.
    b = write_run(tmp_path / "pseudobulk_de_covB",
                  {H1: block(10, drop=GENES[:2]), H2: block(7), H3: block(0), H4: block(10), N1: block(10)},
                  hotspots=HOT, skipped=[H5], covariates="rin, pct_globin, pct_umi_dup, median_5_3_bias")
    return head, a, b


def test_compare_counts_and_verdicts(runs):
    head, a, b = runs
    table, alpha = cdr.compare(head, {"A": a, "B": b})
    assert alpha == 0.05
    t = table.set_index(["tissue", "cell_type"])
    assert list(table["is_hotspot"]) == sorted(table["is_hotspot"], reverse=True)       # hotspots first

    # H1: 9 of 10 kept under A, 8 of 10 under B (the two missing genes count as lost), same p order
    r = t.loc[H1]
    assert (r["n_sig_headline"], r["n_retained_A"], r["n_lost_A"], r["n_gained_A"]) == (10, 9, 1, 0)
    assert (r["n_sig_B"], r["n_retained_B"], r["n_lost_B"], r["n_genes_compared_B"]) == (8, 8, 2, 48)
    assert r["retained_share_B"] == 0.8 and r["spearman_neglog10p_A"] == pytest.approx(1.0)
    assert r["verdict"] == "ROBUST"                           # 0.8 is "at least 0.8"

    assert t.loc[H2, "retained_share_B"] == 0.7 and t.loc[H2, "verdict"] == "NOT ROBUST"
    # H3 has no dose-significant gene in the headline: NO CALLS, never robust; its 5 gains are counted
    assert t.loc[H3, "verdict"] == "NO CALLS" and t.loc[H3, "n_gained_A"] == 5
    assert np.isnan(t.loc[H3, "retained_share_A"])
    # H4 keeps every call but its p-values are reordered under A
    assert t.loc[H4, "retained_share_A"] == 1.0 and t.loc[H4, "spearman_neglog10p_A"] == pytest.approx(-1.0)
    assert t.loc[H4, "verdict"] == "NOT ROBUST"
    # H5 was not analysed by the headline; a block that is not a hotspot has numbers and no verdict
    assert t.loc[H5, "verdict"] == "NOT ANALYSED" and t.loc[H5, "status_headline"].startswith("SKIP")
    assert t.loc[N1, "verdict"] == "" and t.loc[N1, "n_gained_A"] == 2

    text = "\n".join(cdr.summarise(table, {"A": a, "B": b}, alpha))
    assert "pairs at FDR_IHW < 0.05: 40" in text and "ROBUST 1" in text and "NO CALLS 1" in text


def test_borderline_blocks_are_carried_with_their_class(tmp_path):
    """STEP3 5.2: a borderline block is not a hotspot, is named in the table, and gets the same reading."""
    head = write_run(tmp_path / "h", {("T", "hot"): block(10), ("T", "edge"): block(10), ("T", "quiet"): block(10)},
                     hotspots={("T", "hot")})
    summ = pd.read_csv(head / "de_summary.tsv", sep="\t")
    summ["gate_class"] = ["hotspot", "borderline", "none"]
    summ.to_csv(head / "de_summary.tsv", sep="\t", index=False, na_rep="NA")
    same = {("T", "hot"): block(10), ("T", "edge"): block(10), ("T", "quiet"): block(10)}
    a = write_run(tmp_path / "a", same, covariates="rin, pct_globin")
    b = write_run(tmp_path / "b", same, covariates="rin, pct_globin, pct_umi_dup, median_5_3_bias")
    table, alpha = cdr.compare(head, {"A": a, "B": b})
    t = table.set_index("cell_type")
    assert t["gate_class"].to_dict() == {"hot": "hotspot", "edge": "borderline", "quiet": "none"}
    assert t.loc["hot", "verdict"] == "ROBUST" and t.loc["edge", "verdict"] == "ROBUST" and t.loc["quiet", "verdict"] == ""
    assert not t.loc["edge", "is_hotspot"]
    text = "\n".join(cdr.summarise(table, {"A": a, "B": b}, alpha))
    assert "hotspots: 1;" in text and "borderline blocks" in text and "T edge [borderline]" in text


def test_a_summary_without_gate_class_reads_as_before(runs):
    head, a, b = runs
    table, _ = cdr.compare(head, {"A": a, "B": b})
    assert set(table["gate_class"]) <= {"hotspot", ""}
    assert (table["gate_class"] == "hotspot").equals(table["is_hotspot"])


def test_confound_table_carries_the_gate_class():
    base = {"sup_trained_auc": np.nan, "mean_fraction": 0.1, "n_sig_interaction": 0, "frac_week_slope": 0.0,
            "frac_week_p": 0.5, "n_male": 5, "n_female": 5, "n_dose_tests": 100, "n_sig_dose_IHW": 3, "status": "ok"}
    df = pd.DataFrame([dict(base, tissue="T", cell_type="a", is_hotspot="FALSE", gate_class="borderline"),
                       dict(base, tissue="T", cell_type="b", is_hotspot="TRUE", gate_class="hotspot")])
    out = confound.build_table(df)
    assert list(out.columns[:4]) == ["tissue", "cell_type", "is_hotspot", "gate_class"]
    assert out.set_index("cell_type")["gate_class"].to_dict() == {"a": "borderline", "b": "hotspot"}
    assert "gate_class" not in confound.build_table(df.drop(columns="gate_class")).columns


def test_verdict_needs_both_registered_reruns(runs):
    head, a, _ = runs
    t = cdr.compare(head, {"A": a})[0].set_index(["tissue", "cell_type"])
    assert t.loc[H1, "verdict"] == "NOT ASSESSED"
    assert t.loc[H3, "verdict"] == "NO CALLS"


def test_block_missing_from_a_rerun_is_not_assessed(runs, tmp_path):
    head, a, _ = runs
    b = write_run(tmp_path / "b_short", {H2: block(10), H3: block(0), H4: block(10), N1: block(10)},
                  hotspots=HOT, skipped=[H1], covariates="x")
    t = cdr.compare(head, {"A": a, "B": b})[0].set_index(["tissue", "cell_type"])
    assert t.loc[H1, "status_B"].startswith("SKIP") and t.loc[H1, "verdict"] == "NOT ASSESSED"


def test_compare_refuses_different_alpha(runs, tmp_path):
    head, a, _ = runs
    b = write_run(tmp_path / "b_alpha", {H1: block(10)}, alpha="0.01")
    with pytest.raises(ValueError, match="alpha differs"):
        cdr.compare(head, {"A": a, "B": b})


def test_compare_main_writes_one_table(runs, tmp_path, capsys):
    head, a, b = runs
    out = tmp_path / "out" / "de_covariate_robustness.tsv"
    cdr.main(["--headline", str(head), "--variant", f"A={a}", "--variant", f"B={b}", "--out", str(out)])
    written = pd.read_csv(out, sep="\t")
    assert len(written) == 6 and (written["verdict"] == "ROBUST").sum() == 1
    assert "hotspots: 5" in capsys.readouterr().out
    # a headline that was itself run with covariates is not a headline
    with pytest.raises(SystemExit, match="technical covariates"):
        cdr.main(["--headline", str(a), "--variant", f"B={b}", "--out", str(out)])


def test_spearman_handles_ties_and_short_input():
    assert cdr.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert cdr.spearman([1, 2, 2, 4], [1, 3, 2, 4]) == pytest.approx(0.9486832980505138)   # average ranks
    assert cdr.spearman([1, 2, np.inf], [3, 2, 1]) == pytest.approx(-1.0)                   # p = 0 gives inf
    assert np.isnan(cdr.spearman([1, 2], [2, 1])) and np.isnan(cdr.spearman([1, 1, 1], [1, 2, 3]))


# ---------------------------------------------------------------- build_composition_confound_table.py

def summary_rows():
    base = {"is_hotspot": False, "sup_trained_auc": np.nan, "mean_fraction": 0.1, "n_sig_interaction": 0,
            "frac_week_slope": 0.001, "n_male": 25, "n_female": 25, "status": "ok", "n_dose_tests": 13000}
    rows = [
        {"cell_type": "flag", "n_sig_dose_IHW": 30, "frac_week_p": 0.01},
        {"cell_type": "pass", "n_sig_dose_IHW": 30, "frac_week_p": 0.5},
        {"cell_type": "quiet", "n_sig_dose_IHW": 3, "frac_week_p": 0.01},
        {"cell_type": "theta untested", "n_sig_dose_IHW": 30, "frac_week_p": np.nan},
        {"cell_type": "one sex, no dose p", "n_sig_dose_IHW": 0, "frac_week_p": np.nan,
         "n_female": 0, "n_dose_tests": 0},
        {"cell_type": "skipped", "n_sig_dose_IHW": np.nan, "frac_week_p": np.nan, "n_male": np.nan,
         "n_female": np.nan, "n_dose_tests": np.nan, "status": "SKIP: degenerate (median libsize 12.7 < 1000)"},
    ]
    return pd.DataFrame([{**base, "tissue": "KIDNEY", **r} for r in rows])


def test_confound_statuses_separate_untested_blocks_from_quiet():
    out = confound.build_table(summary_rows()).set_index("cell_type")["verdict"].to_dict()
    assert out == {"flag": "FLAG_COMPOSITION", "pass": "PASS_EXPRESSION", "quiet": "QUIET",
                   "theta untested": "THETA_NOT_TESTED", "one sex, no dose p": "NO_DOSE_TEST",
                   "skipped": "NOT_ANALYSED"}


def test_confound_statuses_on_a_summary_without_n_dose_tests():
    # A de_summary.tsv written before STEP3 has no n_dose_tests: the dose test ran in two-sex blocks only.
    table = confound.build_table(summary_rows().drop(columns="n_dose_tests"))
    out = table.set_index("cell_type")["verdict"].to_dict()
    assert out["one sex, no dose p"] == "NO_DOSE_TEST" and out["skipped"] == "NOT_ANALYSED"
    assert (table["verdict"] == "QUIET").sum() == 1
    assert pd.isna(table.set_index("cell_type").loc["skipped", "n_sig_dose_IHW"])       # not turned into 0
    assert confound.status_counts(table) == ("1 FLAG_COMPOSITION, 1 PASS_EXPRESSION, 1 THETA_NOT_TESTED, "
                                             "1 QUIET, 1 NO_DOSE_TEST, 1 NOT_ANALYSED")


@pytest.mark.data
def test_confound_statuses_on_the_refs4all_rebuild_default(require):
    """Audit finding 7.6: of the 112 blocks the old table called QUIET, 46 were never analysed and 21 are
    one-sex blocks whose dose test could not run; 45 were tested and quiet."""
    f = require(ROOT / "data/deconvolution/genecompass_input_v2_refs4all_rebuild/pseudobulk_de/de_summary.tsv")
    n = confound.build_table(pd.read_csv(f, sep="\t"))["verdict"].value_counts().to_dict()
    assert n == {"PASS_EXPRESSION": 62, "NOT_ANALYSED": 46, "QUIET": 45, "FLAG_COMPOSITION": 23, "NO_DOSE_TEST": 21}


# ---------------------------------------------------------------- run_stage10.py

@pytest.fixture
def stage10(monkeypatch):
    """run_stage10.main with the subprocess calls recorded instead of run, and the input checks skipped
    (they need Stage 8 outputs). Returns a function: argv -> list of (command, environment)."""
    calls = []

    def fake_run(cmd, cwd=None, env=None):
        calls.append(([str(c) for c in cmd], dict(env)))
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr(run_stage10.subprocess, "run", fake_run)
    monkeypatch.setattr(run_stage10, "validate_inputs", lambda ctx, from_step: [])
    monkeypatch.delenv("DE_ALPHA", raising=False)
    monkeypatch.delenv("DE_TECH_COVARIATES", raising=False)

    def run(argv):
        calls.clear()
        run_stage10.main(argv)
        return list(calls)
    return run


def test_alpha_reaches_the_r_step_and_the_comparison(stage10):
    de, posctrl = stage10(["--alpha", "0.01"])
    assert de[0][-1].endswith("run_pseudobulk_de.sh") and de[1]["DE_ALPHA"] == "0.01"
    assert float(de[1]["DE_ALPHA"]) == float(posctrl[0][posctrl[0].index("--alpha") + 1])
    assert de[1]["DE_TECH_COVARIATES"] == "" and de[1]["OUT_DIR"].endswith("pseudobulk_de")
    # the default is the registered level, in a form as.numeric() reads back exactly
    assert stage10([])[0][1]["DE_ALPHA"] == "0.05"


def test_tech_covariates_reach_the_r_step_in_a_fixed_order(stage10):
    calls = stage10(["--tech-covariates", "pct_globin", "rin", "--sensitivities", "100"])
    assert [c[1]["DE_TECH_COVARIATES"] for c in calls[:2]] == ["rin,pct_globin"] * 2      # DE and its sensitivity
    assert calls[1][1]["DE_MIN_MEDIAN_LIBSIZE"] == "100" and calls[1][1]["OUT_DIR"].endswith("pseudobulk_de_minlib100")


def test_covariate_variants_run_into_sibling_folders_then_compare(stage10):
    calls = stage10(["--covariate-variants", "B", "A"])
    de, posctrl, var_a, var_b, cmp = calls
    assert de[1]["DE_TECH_COVARIATES"] == ""                                              # the headline has none
    head = de[1]["OUT_DIR"]
    assert (var_a[1]["OUT_DIR"], var_a[1]["DE_TECH_COVARIATES"]) == (head + "_covA", "rin,pct_globin")
    assert (var_b[1]["OUT_DIR"], var_b[1]["DE_TECH_COVARIATES"]) == \
        (head + "_covB", "rin,pct_globin,pct_umi_dup,median_5_3_bias")
    assert var_a[0] == de[0] and var_b[1]["DE_ALPHA"] == de[1]["DE_ALPHA"] and var_b[1]["HOTSPOTS"] == de[1]["HOTSPOTS"]
    assert cmp[0][1].endswith("compare_de_runs.py")
    assert cmp[0][2:] == ["--headline", head, "--variant", f"A={head}_covA", "--variant", f"B={head}_covB",
                          "--out", head + "/de_covariate_robustness.tsv"]
    # --from 3 runs the reruns and the comparison only
    only = stage10(["--from", "3", "--covariate-variants", "A", "B"])
    assert [c[1]["OUT_DIR"][len(head):] for c in only[:2]] == ["_covA", "_covB"] and len(only) == 3


@pytest.mark.parametrize("argv", [["--alpha", "5"], ["--from", "3"], ["--tech-covariates", "bogus"],
                                  ["--covariate-variants", "C"],
                                  ["--tech-covariates", "rin", "--covariate-variants", "A"]])
def test_stage10_rejects_bad_options(argv):
    with pytest.raises(SystemExit):
        run_stage10.parse_args(argv)


def test_r_step_reads_what_the_orchestrator_sets():
    """The environment names and the covariate names are shared by three files; keep them in step."""
    r_src = (ROOT / "deconvolution/R/run_pseudobulk_de.R").read_text()
    sh_src = (ROOT / "deconvolution/R/run_pseudobulk_de.sh").read_text()
    for name in ("DE_ALPHA", "DE_TECH_COVARIATES", "DE_SEED"):
        assert f'Sys.getenv("{name}")' in r_src and name in sh_src
    m = re.search(r"TECH_COV_COLUMNS <- c\((.*?)\)", r_src, flags=re.S)
    assert tuple(re.findall(r"(\w+)\s*=", m.group(1))) == run_stage10.TECH_COVARIATES
    assert run_stage10.COVARIATE_VARIANTS == {"A": ("rin", "pct_globin"), "B": run_stage10.TECH_COVARIATES}
    assert 'covariate_type = IHW_COVARIATE_TYPE' in r_src and 'IHW_COVARIATE_TYPE <- "nominal"' in r_src


# ---------------------------------------------------------------- repfdr_call_sources.py

def test_sexcons_state_rule_for_both_table_layouts():
    """One gene per case; alpha 0.05. Since STEP3 6.2 concordance alone decides the state; before, repfdr OR
    concordance did. The repfdr column was renamed repfdr_localfdr -> repfdr_Fdr (it holds a tail-area Fdr)."""
    df = pd.DataFrame({
        #              both pass   both pass   repfdr only  male only  opposite  one-sex block
        "fdr_8w_M":   [0.01,       0.01,       0.20,        0.01,      0.01,     0.01],
        "fdr_8w_F":   [0.01,       0.01,       0.01,        0.50,      0.01,     np.nan],
        "z_8w_M":     [3.0,        -3.0,       1.0,         3.0,       3.0,      3.0],
        "z_8w_F":     [2.5,        -2.5,       3.0,         0.2,       -3.0,     np.nan],
        "repfdr_fdr": [0.001,      np.nan,     0.001,       0.90,      0.90,     np.nan],
        "single_sex": [False,      False,      False,       False,     False,    True],
    })
    st, repl, conc, same = rcs.states(df, 0.05, "primary_concordance")
    assert list(st) == ["up_both", "down_both", "F_only", "M_only", "opposite", "single_sex"]
    assert list(conc & same) == [True, True, False, False, False, False]
    assert list(repl & same) == [True, False, True, False, False, False]
    st_old, *_ = rcs.states(df, 0.05, "union")
    assert list(st_old) == ["up_both", "down_both", "up_both", "M_only", "opposite", "M_only"]


def call_source_folder(d, new_layout):
    """A DE folder with one two-sex block (six genes, one per case) and one one-sex block, written as
    run_pseudobulk_de.R writes them in the current layout (new_layout) or in the earlier one."""
    two = pd.DataFrame({
        "gene":     [f"g{i}" for i in range(6)],
        "z_8w_M":   [3.0, -3.0, 1.0, 3.0, 3.0, 1.0],
        "z_8w_F":   [2.5, -2.5, 3.0, 0.2, -3.0, 1.0],
        "fdr_8w_M": [0.01, 0.01, 0.20, 0.01, 0.01, 0.5],
        "fdr_8w_F": [0.01, 0.01, 0.01, 0.50, 0.01, 0.5],
    })
    fdr = [0.001, np.nan, 0.001, 0.9, 0.9, np.nan]
    one = pd.DataFrame({"gene": ["g0", "g1"], "z_8w_M": [3.0, 0.1], "z_8w_F": np.nan,
                        "fdr_8w_M": [0.01, 0.9], "fdr_8w_F": np.nan})
    if new_layout:
        two["repfdr_Fdr"] = fdr
        two["sexcons_concordance_8w"] = ["TRUE", "TRUE", "FALSE", "FALSE", "FALSE", "FALSE"]
        two["sexcons_repfdr_8w"] = ["TRUE", "FALSE", "TRUE", "FALSE", "FALSE", "FALSE"]
        two["sexcons_union_8w"] = ["TRUE", "TRUE", "TRUE", "FALSE", "FALSE", "FALSE"]
        two["sexcons_8w"] = ["up_both", "down_both", "F_only", "M_only", "opposite", "null"]
        one["repfdr_Fdr"] = np.nan
        for c in rcs.CALL_COLS:
            one[c] = np.nan
        one["sexcons_8w"] = "single_sex"
    else:
        two["repfdr_localfdr"] = fdr
        two["sexcons_8w"] = ["up_both", "down_both", "up_both", "M_only", "opposite", "null"]
        one[["z_8w_M", "fdr_8w_M"]] = np.nan          # the earlier code ran no per-sex fit in a one-sex block
        one["repfdr_localfdr"] = np.nan
        one["sexcons_8w"] = "null"
    (d / "HEART").mkdir(parents=True)
    (d / "TESTES").mkdir()
    two.to_csv(d / "HEART/de__T_cells.tsv", sep="\t", index=False, na_rep="NA")
    one.to_csv(d / "TESTES/de__Sertoli_cells.tsv", sep="\t", index=False, na_rep="NA")
    pd.DataFrame({"tissue": ["HEART", "TESTES"], "cell_type": ["T cells", "Sertoli cells"],
                  "file": ["HEART/de__T_cells.tsv", "TESTES/de__Sertoli_cells.tsv"]}
                 ).to_csv(d / "celltype_files.tsv", sep="\t", index=False)
    pd.DataFrame({"tissue": ["HEART", "TESTES"], "cell_type": ["T cells", "Sertoli cells"],
                  "n_male": [25, 25], "n_female": [25, 0]}).to_csv(d / "de_summary.tsv", sep="\t", index=False)
    (d / "de_methods.tsv").write_text("alpha\t0.05\nsex_consistency\ttest\n")
    return d


def test_call_sources_reads_both_layouts_and_checks_the_stored_calls(tmp_path, monkeypatch):
    new = call_source_folder(tmp_path / "new", new_layout=True)
    old = call_source_folder(tmp_path / "old", new_layout=False)
    monkeypatch.setattr("sys.argv", ["repfdr_call_sources.py", "--de-dir", f"new={new}", "--de-dir", f"old={old}",
                                     "--out", str(tmp_path / "out")])
    rcs.main()
    r = pd.read_csv(tmp_path / "out/repfdr_call_sources.tsv", sep="\t").set_index("run")
    assert list(r["layout"]) == ["primary_concordance", "union"]
    assert (r["n_state_mismatch"] == 0).all() and (r["n_call_mismatch"] == 0).all()
    n = r.loc["new"]
    assert (n["n_sexcons_concordance"], n["n_sexcons_repfdr"], n["n_sexcons_union"]) == (2, 2, 3)
    assert (n["n_up_both"], n["n_down_both"], n["n_opposite"], n["n_single_sex"]) == (1, 1, 1, 2)
    o = r.loc["old"]
    assert (o["n_repfdr_sig"], o["n_concordance_sig"], o["n_up_both"], o["n_sexcons_directional_repfdr_only"]) == (2, 3, 2, 1)

    # a stored call that the rule does not give stops the script
    t = pd.read_csv(new / "HEART/de__T_cells.tsv", sep="\t", dtype=str, keep_default_na=False)
    t.loc[2, "sexcons_concordance_8w"] = "TRUE"
    t.to_csv(new / "HEART/de__T_cells.tsv", sep="\t", index=False)
    with pytest.raises(SystemExit, match="differ from the tables"):
        rcs.main()


def test_repfdr_layout_is_read_from_the_column_name(tmp_path):
    new, old, neither = tmp_path / "new.tsv", tmp_path / "old.tsv", tmp_path / "x.tsv"
    new.write_text("gene\trepfdr_Fdr\tsexcons_8w\n")
    old.write_text("gene\trepfdr_localfdr\tsexcons_8w\n")
    neither.write_text("gene\tsexcons_8w\n")
    assert rcs.layout(new) == "primary_concordance" and rcs.layout(old) == "union"
    with pytest.raises(SystemExit):
        rcs.layout(neither)
