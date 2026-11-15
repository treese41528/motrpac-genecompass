"""The commands that build the V5 and V6 tables (docs/audit/remediation/STEP4_PREREG.md V5, V6), on small
made-up result folders: the registered rules, the common set, the floor, and what happens to missing runs.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis" / "audit_remediation"))
sys.path.insert(0, str(ROOT / "deconvolution" / "omnideconv_bench"))
import make_step4_v5_tables as v5  # noqa: E402
import make_step4_v6_tables as v6  # noqa: E402
import simbu_summary as ss  # noqa: E402

METHODS = ["bp_rna", "cibersortx", "dwls", "music", "scdc", "bisque"]


def _scores(stage, method, rmse, r=0.9, median=0.95, types=None):
    """overall_metrics_<m>.tsv and metrics_<m>.tsv as score_validation.py writes them."""
    sc = stage / "scores"; sc.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"metric": "overall_pooled_pearson", "value": r, "note": ""},
                  {"metric": "overall_pooled_rmse", "value": rmse, "note": ""},
                  {"metric": "median_pearson", "value": median, "note": ""}]).to_csv(
        sc / f"overall_metrics_{method}.tsv", sep="\t", index=False)
    types = types or {"T cells": (0.99, 0.01, 0.2, 0.2), "Hepatocytes": (0.9, rmse, 0.3, 0.3 + rmse)}
    pd.DataFrame([{"cell_type": t, "n": 50, "pearson_r": v[0], "spearman_rho": v[0], "rmse": v[1], "mean_bias": 0.0,
                   "true_mean": v[2], "est_mean": v[3]} for t, v in types.items()]).to_csv(
        sc / f"metrics_{method}.tsv", sep="\t", index=False)


# ============================================================================================ V5
ARM_META = {"A": "scaling=NONE\nremove_bias_in_counts=TRUE\nremove_bias_in_counts_method=read-number\n",
            "B": "scaling=NONE\nremove_bias_in_counts=FALSE\nremove_bias_in_counts_method=read-number\n",
            "C": "scaling=expressed_genes\nremove_bias_in_counts=FALSE\nremove_bias_in_counts_method=read-number\n"}


def _v5_reference(root, code, rmse, skip=(), truth_by_arm=None):
    """rmse: {method: (A, B, C)}; skip: (method, arm) pairs with no score file."""
    for arm in "ABC":
        st = root / code / arm
        (st / "reference").mkdir(parents=True); (st / "mixtures").mkdir()
        # 4 cells x 3 genes; hepatocytes have the larger libraries and more expressed genes
        (st / "reference" / "reference_counts.mtx").write_text(
            "%%MatrixMarket matrix coordinate integer general\n4 3 8\n1 1 1\n2 1 2\n3 1 5\n3 2 5\n3 3 5\n4 1 9\n4 2 9\n4 3 9\n")
        (st / "reference" / "cells_meta.tsv").write_text(
            "barcode\tsample\tcell_type\nc1\ts1\tT cells\nc2\ts1\tT cells\nc3\ts2\tHepatocytes\nc4\ts2\tHepatocytes\n")
        (st / "mixtures" / "true_fractions.tsv").write_text((truth_by_arm or {}).get(arm, "mixture\tcellfrac__T cells\nmix1\t0.5\n"))
        (st / "mixtures" / "sim_meta.txt").write_text(ARM_META[arm] + "n_ref_cells=4\nn_types=2\nseed=1\n")
        for m, vals in rmse.items():
            if (m, arm) not in skip:
                _scores(st, m, vals["ABC".index(arm)])


def _run_v5(root, out, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["x", "--root", str(root), "--previous", str(root / "none"), "--out", str(out)])
    v5.main()
    return capsys.readouterr().out


def test_corrector_rule_needs_both_correctors_below_both_others():
    base = {"bp_rna": 0.025, "cibersortx": 0.022}
    assert ss.corrector_rule({**base, "music": 0.018, "scdc": 0.016})
    assert not ss.corrector_rule({**base, "music": 0.0254, "scdc": 0.0215})     # the 2026-11-11 numbers
    assert not ss.corrector_rule({**base, "music": 0.021, "scdc": 0.023})       # SCDC above CIBERSORTx
    assert not ss.corrector_rule({**base, "music": 0.022, "scdc": 0.010})       # a tie is not "below"
    assert not ss.corrector_rule({**base, "scdc": 0.010})                       # a method without a mean
    assert not ss.corrector_rule({**base, "music": np.nan, "scdc": 0.010})


def test_v5_readings_on_a_made_up_battery(tmp_path, monkeypatch, capsys):
    root, out = tmp_path / "simbu", tmp_path / "rec"
    # R1, R2: all six methods. C - A: MuSiC and SCDC rise least -> reproduced. C - B: MuSiC rises most -> not.
    full = {"bp_rna": (0.010, 0.040, 0.070), "cibersortx": (0.020, 0.050, 0.075), "dwls": (0.005, 0.010, 0.040),
            "music": (0.010, 0.012, 0.050), "scdc": (0.030, 0.050, 0.060), "bisque": (0.060, 0.065, 0.066)}
    _v5_reference(root, "R1", full); _v5_reference(root, "R2", full)
    # R3: one-sample reference, so no MuSiC or Bisque; BayesPrism is BETTER under the native bias here
    r3 = {m: v for m, v in full.items() if m not in ("music", "bisque")}
    r3["bp_rna"] = (0.050, 0.045, 0.080)
    _v5_reference(root, "R3", r3)
    (root / "arm_check").mkdir()
    pd.DataFrame({"code": ["R1"] * 4, "cell_type": ["T cells", "Hepatocytes"] * 2, "arm": ["A", "A", "B", "B"],
                  "mean_library_size_as_summed": [1.0, 1.0, 1.5, 12.0]}).to_csv(root / "arm_check" / "R1.tsv", sep="\t", index=False)
    txt = _run_v5(root, out, monkeypatch, capsys)

    rmse = pd.read_csv(out / "v5_rmse.tsv", sep="\t")
    assert len(rmse) == 18 and (rmse["status"] == "ok").sum() == 16
    assert set(rmse[rmse["status"] != "ok"]["method"]) == {"music", "bisque"}
    bp = pd.read_csv(out / "v5_bayesprism.tsv", sep="\t").set_index("reference")
    assert bp.loc["R1", "d_rmse_B_minus_A"] == pytest.approx(0.030) and bp.loc["R3", "d_rmse_B_minus_A"] == pytest.approx(-0.005)
    assert bp.loc["R3", "d_rmse_C_minus_A"] == pytest.approx(0.030) and bp.loc["R1", "d_rmse_C_minus_B"] == pytest.approx(0.030)
    assert "RMSE(B) - RMSE(A) is positive on 2 of 3 references" in txt
    assert "RMSE(C) - RMSE(A) is positive on 3 of 3 references" in txt

    rule = pd.read_csv(out / "v5_corrector_rule.tsv", sep="\t").set_index("contrast")
    assert list(rule["common_set"].unique()) == ["all three arms"]              # one common set for every contrast
    assert rule.loc["C-A", "n_common"] == 2 and rule.loc["C-A", "references"] == "R1; R2"
    assert rule.loc["C-A", "verdict"] == "reproduced"                           # music +0.040, scdc +0.030 < cx +0.055, bp +0.060
    assert rule.loc["C-A", "music"] == pytest.approx(0.040) and rule.loc["C-A", "bp_rna"] == pytest.approx(0.060)
    assert rule.loc["C-B", "verdict"] == "not reproduced"                       # music +0.038 > bp +0.030 and cx +0.025
    assert rule.loc["C-A", "leave_one_out_same_verdict"] == "2 of 2"
    rank = pd.read_csv(out / "v5_ranking.tsv", sep="\t")
    r = rank[(rank["contrast"] == "C-A") & (rank["method"] == "scdc")].iloc[0]
    assert r["n_common"] == 2 and r["n_all"] == 3 and r["n_positive"] == 3      # SCDC also ran on R3
    assert r["mean_rmse_A_common"] == pytest.approx(0.030) and r["mean_rmse_C_common"] == pytest.approx(0.060)

    ms = pd.read_csv(out / "v5_method_status.tsv", sep="\t").set_index(["reference", "arm", "method"])["status"]
    assert len(ms) == 54 and (ms == "scored").sum() == 48
    assert ms[("R3", "A", "music")] == "not run"                 # no score and no record of a run
    ma = pd.read_csv(out / "v5_method_arm_rmse.tsv", sep="\t").set_index("method")
    assert ma.loc["music", "n_references"] == 2 and ma.loc["music", "mean_rmse_B"] == pytest.approx(0.012)
    assert ma.loc["bp_rna", "n_lowest_in_A"] == 2 and ma.loc["bp_rna", "n_lowest_in_B"] == 0   # R3 is not in the common set
    st = pd.read_csv(out / "v5_status.tsv", sep="\t").set_index("reference")
    assert st["same_truth"].all() and st["same_reference"].all()
    assert st["call_A_as_registered"].all() and st["call_B_as_registered"].all() and st["call_C_as_registered"].all()
    assert not st.loc["R3", "A_music"] and st.loc["R3", "A_scdc"]
    lib = pd.read_csv(out / "v5_arm_library_size_summary.tsv", sep="\t").set_index(["reference", "arm"])
    assert lib.loc[("R1", "A"), "max_over_min"] == 1.0 and lib.loc[("R1", "B"), "max_over_min"] == 8.0
    assert set(lib.xs("not checked", level="arm").index) == {"R2", "R3"}         # a missing check is listed
    drv = pd.read_csv(out / "v5_driver.tsv", sep="\t")
    d = drv[(drv["contrast"] == "C-A") & (drv["reference"] == "R1") & (drv["method"] == "bp_rna")].iloc[0]
    assert d["driver"] == "Hepatocytes" and d["driver_mrna_rank"] == 1 and d["direction"] == "inflated"
    mc = pd.read_csv(out / "v5_mrna_content.tsv", sep="\t").set_index(["reference", "cell_type"])
    assert mc.loc[("R1", "Hepatocytes"), "mean_expressed_genes"] == 3 and mc.loc[("R1", "T cells"), "mean_total_counts"] == 1.5


def test_v5_tells_not_applicable_from_failed_by_the_run_record(tmp_path, monkeypatch, capsys):
    root, out = tmp_path / "simbu", tmp_path / "rec"
    part = {m: (0.01, 0.02, 0.03) for m in METHODS if m not in ("music", "bisque", "scdc")}
    _v5_reference(root, "ONE", part)
    for arm in "ABC":
        st = root / "ONE" / arm
        (st / "reference" / "cells_meta.tsv").write_text(          # the same four cells, all from one sample
            "barcode\tsample\tcell_type\nc1\ts1\tT cells\nc2\ts1\tT cells\nc3\ts1\tHepatocytes\nc4\ts1\tHepatocytes\n")
        (st / "results").mkdir()
        (st / "results" / "provenance_run_omnideconv_music+scdc+bisque.json").write_text(json.dumps({"params": {"method_result": {
            "music": {"status": "failed", "error": "one subject"}, "scdc": {"status": "failed", "error": "singular matrix"},
            "bisque": {"status": "failed", "error": "Only one individual detected in single-cell data"}}}}))
    txt = _run_v5(root, out, monkeypatch, capsys)
    ms = pd.read_csv(out / "v5_method_status.tsv", sep="\t").set_index(["arm", "method"])
    assert ms.loc[("A", "music"), "status"] == "not applicable" and ms.loc[("C", "bisque"), "status"] == "not applicable"
    assert ms.loc[("B", "scdc"), "status"] == "failed" and ms.loc[("B", "scdc"), "detail"] == "singular matrix"
    assert "A not applicable" in txt and "B failed (singular matrix)" in txt


def test_v5_common_set_is_over_all_three_arms_and_a_differing_contrast_is_shown(tmp_path, monkeypatch, capsys):
    root, out = tmp_path / "simbu", tmp_path / "rec"
    full = {m: (0.01, 0.02, 0.03) for m in METHODS}
    _v5_reference(root, "R1", full); _v5_reference(root, "R2", full)
    _v5_reference(root, "R3", full, skip={("dwls", "A")})            # DWLS failed in arm A only
    _run_v5(root, out, monkeypatch, capsys)
    rule = pd.read_csv(out / "v5_corrector_rule.tsv", sep="\t")
    three = rule[rule["common_set"] == "all three arms"]
    assert (three["n_common"] == 2).all() and list(three["contrast"]) == ["C-A", "C-B"]   # no verdict for B - A
    rank = pd.read_csv(out / "v5_ranking.tsv", sep="\t")
    assert set(rank["contrast"]) == {"C-A", "B-A", "C-B"}                                 # but B - A is ranked
    own = rule[rule["common_set"] == "this contrast only"]
    assert list(own["contrast"]) == ["C-B"] and own.iloc[0]["n_common"] == 3    # C - B alone could use R3
    rmse = pd.read_csv(out / "v5_rmse.tsv", sep="\t")
    assert rmse[(rmse["reference"] == "R3") & (rmse["method"] == "dwls")]["status"].iloc[0] == "not scored in A"


def test_v5_compares_arms_b_and_c_with_the_earlier_battery_files(tmp_path, monkeypatch, capsys):
    root, prev, out = tmp_path / "simbu", tmp_path / "prev", tmp_path / "rec"
    _v5_reference(root, "LIVER", {m: (0.01, 0.02, 0.03) for m in METHODS})
    _v5_reference(root, "HEART", {m: (0.01, 0.02, 0.03) for m in METHODS})
    for code in ("LIVER", "HEART"):
        for arm, old, body in (("B", "nobias", "same"), ("C", "bias", "same" if code == "LIVER" else "other")):
            (root / code / arm / "mixtures" / "pseudobulk_counts.mtx").write_text("same")
            d = prev / code / old / "mixtures"; d.mkdir(parents=True)
            (d / "pseudobulk_counts.mtx").write_text(body)
    monkeypatch.setattr(sys, "argv", ["x", "--root", str(root), "--previous", str(prev), "--out", str(out)])
    v5.main()
    txt = capsys.readouterr().out
    st = pd.read_csv(out / "v5_status.tsv", sep="\t").set_index("reference")
    assert st.loc["LIVER", "B_mixtures_vs_20261111_nobias"] == "identical" and st.loc["LIVER", "C_mixtures_vs_20261111_bias"] == "identical"
    assert st.loc["HEART", "C_mixtures_vs_20261111_bias"] == "different" and st.loc["HEART", "truth_vs_20261111"] == "absent"
    assert st.loc["HEART", "reference_since_20261111"].startswith("labels changed")
    assert "on the 1 references that did not change: B identical 1/1, C identical 1/1" in txt     # HEART is not counted


def test_v5_flags_arms_that_do_not_share_a_truth_or_were_simulated_otherwise(tmp_path, monkeypatch, capsys):
    root, out = tmp_path / "simbu", tmp_path / "rec"
    _v5_reference(root, "R1", {m: (0.01, 0.02, 0.03) for m in METHODS},
                  truth_by_arm={"C": "mixture\tcellfrac__T cells\nmix1\t0.6\n"})
    (root / "R1" / "A" / "mixtures" / "sim_meta.txt").write_text(ARM_META["B"])   # arm A simulated without the removal
    txt = _run_v5(root, out, monkeypatch, capsys)
    st = pd.read_csv(out / "v5_status.tsv", sep="\t").iloc[0]
    assert not st["same_truth"] and st["same_reference"] and not st["call_A_as_registered"]
    assert "one cell-fraction truth: 0/1" in txt and "A 0/1" in txt


# ============================================================================================ V6
def _v6_set(panel, name, medians, samples=("s1", "s2"), bp_types=None, prov=None):
    """medians: {method: median per-type r}. bp_types: per-type rows for BayesPrism."""
    sd = panel / name
    (sd / "reference").mkdir(parents=True); (sd / "results").mkdir(); (sd / "mixtures").mkdir()
    (panel / "logs").mkdir(exist_ok=True)
    (sd / "reference" / "cells_meta.tsv").write_text("barcode\tsample\tcell_type\n" + "".join(
        f"c{i}\t{s}\tT cells\n" for i, s in enumerate(samples)))
    (panel / "logs" / f"{name}.log").write_text(
        "reference types: ['Hepatocytes', 'T cells']\ntypes usable in pool (>= 20 cells): ['Hepatocytes', 'T cells']\n")
    for m, med in medians.items():
        types = bp_types if (m == "bp_rna" and bp_types) else {"T cells": (med, 0.01, 0.2, 0.2), "Hepatocytes": (med, 0.02, 0.3, 0.3)}
        _scores(sd, m, 0.02, r=med - 0.05, median=med, types=types)
    for fname, params in (prov or {}).items():
        (sd / "results" / fname).write_text(json.dumps({
            "created": "2026-11-14T20:00:00-0500", "git_commit": "ab6f23b" + "0" * 33, "git_modified_tracked_files": [],
            "host": "node", "inputs": [{"path": "x/reference_counts.mtx", "md5": "aa"}, {"path": "x/pseudobulk_counts.mtx", "md5": "bb"}],
            "params": params}))


def _omni(methods, result, dwls=None, fired=()):
    p = {"methods": methods, "method_result": result, "pipeline_profile": "config/profiles/refs5.yaml",
         "method_settings": {m: {"call": "omnideconv::deconvolute, package defaults", "n_batches": 1} for m in methods if m != "dwls"},
         "reference_cap": {"max_cells_per_type": 2000, "seed": 1, "n_cells_before": 9000, "n_cells_after": 5000},
         "gene_cleanup": {"exclude_list_md5": "ee"}, "packages": {"omnideconv": "0.1.1", "DWLS": "0.5.2"},
         "in_memory_patches": {"dwls": {"applied": {"buildSignatureMatrixMAST": {"patched": True},
                                                   "buildSignatureMatrixMASTOptimized": {"patched": True}},
                                        "guard_fired_for": list(fired)},
                               "cibersortx_container_command": {"applied": False}},
         "runner_md5": {"run_omnideconv.R": "ff"}}
    if dwls:
        p["method_settings"]["dwls"] = {"dwls_method": dwls[0], "pval_cutoff": dwls[1], "pval_cutoff_set_by": "DWLS_PVAL_CUTOFF",
                                        "diff_cutoff": 0.5, "dwls_submethod": "DampenedWLS", "ncores": 4}
    return p


def _run_v6(panel, previous, out, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["x", "--panel", str(panel), "--previous", str(previous), "--out", str(out)])
    v6.main()
    return capsys.readouterr().out


def test_v6_floor_status_and_comparisons(tmp_path, monkeypatch, capsys):
    panel, prev, out = tmp_path / "now", tmp_path / "prev", tmp_path / "rec"
    all6 = {m: 0.95 for m in METHODS}
    # exactly at the floor: recovers. HEART has a type the earlier panel did not have, and lost one.
    _v6_set(panel, "HEART_holdout", {**all6, "bp_rna": 0.85, "dwls": 0.90},
            bp_types={"T cells": (0.80, 0.01, 0.2, 0.2), "Cardiomyocytes": (0.90, 0.02, 0.3, 0.3)},
            prov={"provenance_run_omnideconv_dwls.json": _omni(["dwls"], {"dwls": {"status": "ok", "minutes": 12.5}},
                                                              dwls=("mast_optimized", 0.05), fired=["T cells"]),
                  "provenance_run_deconvolution.json": {"n_cores": 16, "bayesprism": "2.2.3", "exclude_list_md5": "ee"}})
    _v6_set(prev, "HEART_holdout", {**all6, "bp_rna": 0.997, "dwls": 0.987},
            bp_types={"T cells": (0.99, 0.01, 0.2, 0.2), "CD8+ T cells": (0.98, 0.01, 0.1, 0.1), "Cardiomyocytes": (0.999, 0.02, 0.3, 0.3)})
    # below the floor
    _v6_set(panel, "LIVER_holdout", {**all6, "bp_rna": 0.849}); _v6_set(prev, "LIVER_holdout", {**all6, "bp_rna": 0.88})
    # a one-sample reference: MuSiC and Bisque not applicable, SCDC failed for another reason
    _v6_set(panel, "HYPOTH_holdout", {"bp_rna": 0.99, "dwls": 0.97, "cibersortx": 0.98}, samples=("s1",),
            prov={"provenance_run_omnideconv_music+scdc+bisque.json": _omni(["music", "scdc", "bisque"], {
                "music": {"status": "failed", "error": "need more than one subject"},
                "scdc": {"status": "failed", "error": "something else"},
                "bisque": {"status": "failed", "error": "Only one individual detected in single-cell data"}})})
    _v6_set(prev, "HYPOTH_holdout", {"bp_rna": 0.995})
    # built, nothing run yet
    _v6_set(panel, "KIDNEY_cross", {})
    pd.DataFrame({"set": ["KIDNEY_cross", "HEART_holdout"], "design": ["cross", "holdout"], "harmonization_map": ["identical", "none"],
                  "mixtures_vs_previous": ["identical", "different"]}).to_csv(panel / "build_check.tsv", sep="\t", index=False)
    txt = _run_v6(panel, prev, out, monkeypatch, capsys)

    fl = pd.read_csv(out / "v6_floor.tsv", sep="\t").set_index("set")
    assert fl.loc["HEART_holdout", "verdict"] == "recovers" and fl.loc["LIVER_holdout", "verdict"] == "weak"
    assert fl.loc["HYPOTH_holdout", "verdict"] == "recovers" and fl.loc["KIDNEY_cross", "verdict"] == "not scored"
    assert fl.loc["HEART_holdout", "n_types_below_floor"] == 1 and fl.loc["HEART_holdout", "worst_types"].startswith("T cells 0.800")
    assert (fl["floor"] == 0.85).all()
    assert "scored 3 of 4 sets; at or above the floor: 2; weak: 1; not scored: 1" in txt

    vs = pd.read_csv(out / "v6_vs_20261108.tsv", sep="\t")
    bp = vs[vs["method"] == "bp_rna"].set_index("set")
    assert bp.loc["HEART_holdout", "median_change"] == pytest.approx(-0.147) and bool(bp.loc["HEART_holdout", "median_drop_gt_0.05"])
    assert not bool(bp.loc["LIVER_holdout", "median_drop_gt_0.05"])              # -0.031: weak, but not a drop of > 0.05
    assert "dropped by more than 0.05: 1 (HEART_holdout -0.147)" in txt

    att = pd.read_csv(out / "v6_heart_hypoth_types.tsv", sep="\t")
    h = att[att["set"] == "HEART_holdout"].set_index("cell_type")
    assert h.loc["T cells", "r_20261108"] == 0.99 and h.loc["T cells", "r_now"] == 0.80
    assert np.isnan(h.loc["CD8+ T cells", "r_now"]) and h.loc["CD8+ T cells", "note"] == "benchmarked on 2026-11-08, not now"

    st = pd.read_csv(out / "v6_status.tsv", sep="\t").set_index(["set", "method"])["status"]
    assert st[("HYPOTH_holdout", "music")] == "not applicable" and st[("HYPOTH_holdout", "bisque")] == "not applicable"
    assert st[("HYPOTH_holdout", "scdc")] == "failed"                            # one sample does not excuse SCDC
    assert st[("HYPOTH_holdout", "bp_rna")] == "scored" and st[("KIDNEY_cross", "dwls")] == "not run"

    dw = pd.read_csv(out / "v6_dwls_setting.tsv", sep="\t").set_index("set")
    assert dw.loc["HEART_holdout", "setting_recorded"] == "mast_optimized, p 0.05, fold change 0.5"
    assert dw.loc["HEART_holdout", "guard_fired_for"] == "T cells" and dw.loc["HEART_holdout", "minutes"] == 12.5
    assert dw.loc["HEART_holdout", "median_change"] == pytest.approx(0.90 - 0.987)
    runs = pd.read_csv(out / "v6_run_settings.tsv", sep="\t")
    r = runs[runs["file"] == "provenance_run_omnideconv_dwls.json"].iloc[0]
    assert r["git_commit"] == "ab6f23b" and int(r["cell_cap"]) == 2000 and int(r["cap_seed"]) == 1
    assert "mast_optimized, p 0.05 (DWLS_PVAL_CUTOFF)" in r["settings"] and r["n_inputs_with_md5"] == 2
    assert "buildSignatureMatrixMASTOptimized: True" in r["dwls_patch_applied"]
    assert (out / "panel_median.tsv").exists() and (out / "v6_build_check.tsv").exists()   # the panel's own summary ran


def test_v6_tables_are_written_for_a_panel_with_no_bayesprism_score_yet(tmp_path, monkeypatch, capsys):
    panel, prev, out = tmp_path / "now", tmp_path / "prev", tmp_path / "rec"
    _v6_set(panel, "BAT_holdout", {"scdc": 0.9}); prev.mkdir()
    txt = _run_v6(panel, prev, out, monkeypatch, capsys)
    assert "not scored: 1" in txt and "was not written" in txt
    assert pd.read_csv(out / "v6_floor.tsv", sep="\t").iloc[0]["verdict"] == "not scored"


# ============================================================================================ the results text
def test_report_is_written_from_the_tables_and_says_what_is_missing(tmp_path, monkeypatch, capsys):
    import make_step4_v5v6_report as rep
    root, out5 = tmp_path / "simbu", tmp_path / "rec5"
    full = {"bp_rna": (0.010, 0.040, 0.070), "cibersortx": (0.020, 0.050, 0.075), "dwls": (0.005, 0.010, 0.040),
            "music": (0.010, 0.012, 0.050), "scdc": (0.030, 0.050, 0.060), "bisque": (0.060, 0.065, 0.066)}
    _v5_reference(root, "R1", full); _v5_reference(root, "R2", full)
    _run_v5(root, out5, monkeypatch, capsys)
    panel, prev, out6 = tmp_path / "now", tmp_path / "prev", tmp_path / "rec6"
    _v6_set(panel, "LIVER_holdout", {m: 0.95 for m in METHODS} | {"bp_rna": 0.849}); _v6_set(prev, "LIVER_holdout", {"bp_rna": 0.88})
    _v6_set(panel, "KIDNEY_cross", {})
    _run_v6(panel, prev, out6, monkeypatch, capsys)
    draft = tmp_path / "draft.md"
    draft.write_text(f"# title\n\nkept above\n\n{rep.BEGIN}\nold text\n{rep.END}\n\nkept below\n")
    monkeypatch.setattr(sys, "argv", ["x", "--v5", str(out5), "--v6", str(out6), "--draft", str(draft), "--date", "2026-11-14"])
    rep.main()
    t = draft.read_text()
    assert "kept above" in t and "kept below" in t and "old text" not in t
    assert t.count(rep.BEGIN) == 1 and t.count(rep.END) == 1
    # V5: 2 of 15 references, so the part is headed incomplete; the verdicts are the tables' own words
    assert "**INCOMPLETE.** 36 of 36 method runs are scored over 2 of 15 references" in t
    assert "Positive on **2 of 2** references" in t
    assert "rule on arm C minus arm A is **reproduced**" in t and "(arm C minus arm B) is **not reproduced**" in t
    assert "| R1 | 0.0100 | 0.0400 | 0.0700 | +0.0300 | +0.0600 | +0.0300 |" in t
    # V6: one weak set and one not scored; the weak set is named in the lead
    assert "The registered floor is NOT met on 1 of 1 sets scored: LIVER_holdout 0.849" in t
    assert "| KIDNEY_cross | n/a | not scored |" in t
    # run again: the part is replaced, not appended
    rep.main()
    assert draft.read_text() == t


def test_report_without_tables_says_so(tmp_path, monkeypatch, capsys):
    import make_step4_v5v6_report as rep
    monkeypatch.setattr(sys, "argv", ["x", "--v5", str(tmp_path / "none5"), "--v6", str(tmp_path / "none6"), "--print"])
    rep.main()
    out = capsys.readouterr().out
    assert out.count("No table yet") == 2


def test_cell_type_names_reported_by_the_dwls_guard_get_their_characters_back():
    tok = "7f8711ec9bf265fb9bf6ab9c"
    assert v6.deescape(f"Brown21b29fb0{tok}adipocytes") == "Brown adipocytes"
    assert v6.deescape(f"CD821b2c6e8{tok}21b29fb0{tok}T21b29fb0{tok}cells") == "CD8+ T cells"
    assert v6.deescape(f"Type21b29fb0{tok}A21b2c756{tok}like21b29fb0{tok}21b29fb8{tok}x21b29fb9{tok}") == "Type A-like (x)"
    assert v6.deescape("Hepatocytes") == "Hepatocytes"
