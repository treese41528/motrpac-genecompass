#!/usr/bin/env python
"""make_step4_v5_tables.py -- every table of V5, the SimBu mRNA-content-bias battery with a baseline that has no
bias (docs/audit/remediation/STEP4_PREREG.md V5; audit finding 5.2), from the finished score files.

  python analysis/audit_remediation/make_step4_v5_tables.py
         [--root data/deconvolution/simbu_bench_refs5] [--previous data/deconvolution/simbu_bench_refs4all]
         [--out docs/audit/remediation/record/step4/v5_simbu]

The three arms of each reference (folders A, B, C under <root>/<CODE>/):
  A  scaling NONE, remove_bias_in_counts TRUE     no mRNA bias
  B  scaling NONE, remove_bias_in_counts FALSE    native bias only (the "no bias" arm of 2026-11-11)
  C  scaling expressed_genes                      added bias (the "bias" arm of 2026-11-11)
All scores are against cell fractions (score_validation.py --truth cellfrac), pooled over types and mixtures.

Registered readings, in the order of the registration (summary.txt):
  1  BayesPrism under native bias: RMSE(B) - RMSE(A) per reference; on how many references it is positive
  2  BayesPrism under added bias:  RMSE(C) - RMSE(A) per reference
  3  Correctors: the rule of SIMBU_BIAS_REFS4ALL_PREREG.md (MuSiC and SCDC both below BayesPrism and CIBERSORTx)
     on the mean of RMSE(C) - RMSE(A) over the common set; reproduced or not
  4  The readings of 2026-11-11 (C against B) on these references, beside 1 to 3
The common set (STEP4_V5_V6_DRAFT.md note N5) is the references where all six methods are scored in all three
arms. If the set a single contrast would give differs from it, both rankings are written.

Tables (all TSV, under --out):
  v5_status.tsv            per reference: the three arms share one reference and one truth; the simulate call of
                           each arm; which method is scored in which arm
  v5_rmse.tsv              per reference and method: pooled RMSE and r in A, B, C and the three differences
  v5_bayesprism.tsv        the BayesPrism rows of v5_rmse.tsv (readings 1 and 2)
  v5_method_status.tsv     per reference, arm and method: scored, not applicable (MuSiC and Bisque on a one-sample
                           reference), failed (with the runner's message) or not run
  v5_ranking.tsv           per contrast and method: mean difference on the common set, over all scored, counts
  v5_corrector_rule.tsv    per contrast: the four means the rule uses, the verdict, the common set, and how many
                           leave-one-reference-out subsets give the same verdict (descriptive)
  v5_method_arm_rmse.tsv   per method: mean pooled RMSE in arms A, B and C over the common set, and the arm in
                           which the method's error is lowest on each reference (descriptive)
  v5_driver.tsv            per contrast, reference and method: the type with the largest per-type difference
  v5_mrna_content.tsv      per reference and type: mean expressed genes and mean counts per cell, with ranks
  v5_arm_library_size.tsv  per reference, type and arm: mean library size of the cells as SimBu sums them
                           (check_simbu_arms.R); v5_arm_library_size_summary.tsv: its spread per reference and arm
  v5_vs_20261111.tsv       C - B here beside bias - nobias of 2026-11-11, per reference and method
  v5_run_settings.tsv      per reference, arm and runner: commit, modified files, settings, from the provenance files
  summary.txt              the readings
Nothing is truncated: every reference, arm and method found on disk is reported and every missing one is listed.
"""
import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "deconvolution" / "omnideconv_bench"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import simbu_summary as ss  # noqa: E402
from make_step4_v6_tables import deescape, n_reference_samples, run_status  # noqa: E402

ARMS = ["A", "B", "C"]
ARM_CALL = {"A": {"scaling": "NONE", "remove_bias_in_counts": "TRUE", "remove_bias_in_counts_method": "read-number"},
            "B": {"scaling": "NONE", "remove_bias_in_counts": "FALSE"},
            "C": {"scaling": "expressed_genes", "remove_bias_in_counts": "FALSE"}}
# contrast name -> (arm with more bias, baseline arm, what it measures)
CONTRASTS = {"C-A": ("C", "A", "added + native bias against no bias (registered corrector reading)"),
             "B-A": ("B", "A", "native bias against no bias"),
             "C-B": ("C", "B", "added bias against native bias (the 2026-11-11 reading)")}


def dcol(name):
    """Column of a contrast's RMSE difference: 'C-A' -> 'd_rmse_C_minus_A'."""
    hi, lo, _ = CONTRASTS[name]
    return f"d_rmse_{hi}_minus_{lo}"


# the corrector rule is registered for C - A (STEP4_PREREG.md V5) and was registered for C - B on 2026-11-11;
# B - A gets a ranking and no verdict
RULE_CONTRASTS = ("C-A", "C-B")
CHANGED = {"HEART": "labels changed (CD8+ T cells merged into T cells)", "HYPOTH": "re-annotated"}
NAMES = {"bp_rna": "BayesPrism", "cibersortx": "CIBERSORTx", "dwls": "DWLS", "music": "MuSiC", "scdc": "SCDC",
         "bisque": "Bisque"}


def references(root):
    return ss.battery_sets(str(root), ARMS)


def same_as_previous(now, then):
    """'identical', 'different' or 'absent' (either file missing): md5 of two files."""
    if not (now.exists() and then.exists()):
        return "absent"
    return "identical" if ss.md5(str(now)) == ss.md5(str(then)) else "different"


def status_table(root, sets, previous=None):
    rows = []
    for s in sets:
        ref_md5, truth_md5, row = set(), set(), {"reference": s}
        present = [a for a in ARMS if (root / s / a / "mixtures" / "true_fractions.tsv").exists()]
        for a in ARMS:
            rp, tp = root / s / a / "reference" / "reference_counts.mtx", root / s / a / "mixtures" / "true_fractions.tsv"
            if rp.exists():
                ref_md5.add(ss.md5(str(rp)))
            if tp.exists():
                truth_md5.add(ss.md5(str(tp)))
            meta = ss.sim_meta(str(root / s / a))
            row[f"call_{a}"] = "; ".join(f"{k}={meta.get(k, '?')}" for k in
                                         ("scaling", "remove_bias_in_counts", "remove_bias_in_counts_method"))
            row[f"call_{a}_as_registered"] = bool(meta) and all(meta.get(k) == v for k, v in ARM_CALL[a].items())
            if a == "A":
                for k in ("n_ref_cells", "n_types", "n_genes_bulk", "nsamples", "ncells", "cells_per_type", "seed",
                          "total_read_counts", "simbu_version"):
                    row[k] = meta.get(k)
        row["arms_built"] = "".join(present)
        # arms B and C make the simulate call of 2026-11-11's nobias and bias arms: on a reference that did not
        # change, their mixtures should be the same files
        if previous is not None:
            for arm, old in (("B", "nobias"), ("C", "bias")):
                row[f"{arm}_mixtures_vs_20261111_{old}"] = same_as_previous(
                    root / s / arm / "mixtures" / "pseudobulk_counts.mtx",
                    Path(previous) / s / old / "mixtures" / "pseudobulk_counts.mtx")
            row["truth_vs_20261111"] = same_as_previous(root / s / "B" / "mixtures" / "true_fractions.tsv",
                                                        Path(previous) / s / "nobias" / "mixtures" / "true_fractions.tsv")
            row["reference_since_20261111"] = CHANGED.get(s, "byte-identical")
        row["same_reference"] = len(present) == 3 and len(ref_md5) == 1
        row["same_truth"] = len(present) == 3 and len(truth_md5) == 1
        for a in ARMS:
            for m in ss.METHODS:
                row[f"{a}_{m}"] = (root / s / a / "scores" / f"overall_metrics_{m}.tsv").exists()
        rows.append(row)
    return pd.DataFrame(rows)


def method_status(root, sets):
    rows = []
    for s in sets:
        for a in ARMS:
            sd = root / s / a
            ns = n_reference_samples(sd)
            for m in ss.METHODS:
                status, detail = run_status(sd, m, ns)
                rows.append({"reference": s, "arm": a, "method": m, "n_reference_samples": ns, "status": status,
                             "detail": detail})
    return pd.DataFrame(rows)


def rmse_table(root, sets):
    rows = []
    for s in sets:
        for m in ss.METHODS:
            row = {"reference": s, "method": m}
            for a in ARMS:
                row[f"rmse_{a}"] = ss.overall(str(root / s / a), m, "overall_pooled_rmse")
                row[f"r_{a}"] = ss.overall(str(root / s / a), m, "overall_pooled_pearson")
            for name, (hi, lo, _) in CONTRASTS.items():
                row[dcol(name)] = row[f"rmse_{hi}"] - row[f"rmse_{lo}"]
            missing = [a for a in ARMS if not np.isfinite(row[f"rmse_{a}"])]
            row["status"] = "ok" if not missing else "not scored in " + ",".join(missing)
            rows.append(row)
    return pd.DataFrame(rows)


def common_set(rmse):
    """References where all six methods are scored in all three arms (note N5)."""
    ok = rmse[rmse["status"] == "ok"]
    return sorted(s for s, g in ok.groupby("reference") if set(g["method"]) == set(ss.METHODS))


def contrast_delta(rmse, name):
    """The rows of one contrast in the shape simbu_summary.ranking expects (set, method, d_rmse, status)."""
    hi, lo, _ = CONTRASTS[name]
    d = pd.DataFrame({"set": rmse["reference"], "method": rmse["method"], "d_rmse": rmse[dcol(name)]})
    d["status"] = np.where(np.isfinite(rmse[f"rmse_{hi}"]) & np.isfinite(rmse[f"rmse_{lo}"]), "ok", "not scored")
    return d


def ranking_and_rule(rmse, common):
    rank_rows, rule_rows = [], []
    for name, (hi, lo, what) in CONTRASTS.items():
        d = contrast_delta(rmse, name)
        own = sorted(ss.ranking(d)[1])                      # the common set this contrast alone would give
        versions = [("all three arms", common)] + ([("this contrast only", own)] if own != list(common) else [])
        for label, cs in versions:
            rk, _ = ss.ranking(d, common=list(cs))
            mc = rk.set_index("method")["mean_d_rmse_common"]
            for r in rk.itertuples():
                sub = rmse[(rmse["method"] == r.method) & rmse["reference"].isin(cs)]
                rank_rows.append({"contrast": name, "common_set": label, "method": r.method,
                                  "mean_d_rmse_common": r.mean_d_rmse_common, "n_common": r.n_common,
                                  f"mean_rmse_{lo}_common": sub[f"rmse_{lo}"].mean(),
                                  f"mean_rmse_{hi}_common": sub[f"rmse_{hi}"].mean(),
                                  "mean_d_rmse_all": r.mean_d_rmse_all, "median_d_rmse_all": r.median_d_rmse_all,
                                  "n_all": r.n_all, "n_positive": r.n_positive, "references_missing": r.sets_missing})
            if name not in RULE_CONTRASTS:
                continue
            verdict = ss.corrector_rule(mc)
            # descriptive: does dropping any one reference change the verdict?
            same = 0
            for drop in cs:
                rk1, _ = ss.ranking(d, common=[x for x in cs if x != drop])
                same += ss.corrector_rule(rk1.set_index("method")["mean_d_rmse_common"]) == verdict
            rule_rows.append({"contrast": name, "measures": what, "common_set": label, "n_common": len(cs),
                              **{m: mc.get(m, np.nan) for m in ss.METHODS},
                              "music_and_scdc_below_bayesprism_and_cibersortx": verdict,
                              "verdict": "reproduced" if verdict else "not reproduced",
                              "leave_one_out_same_verdict": f"{same} of {len(cs)}", "references": "; ".join(cs)})
    R = pd.DataFrame(rank_rows)
    # one frame holds the mean-RMSE columns of every contrast; order them
    lead = ["contrast", "common_set", "method", "mean_d_rmse_common", "n_common"]
    arm_cols = [f"mean_rmse_{a}_common" for a in ARMS if f"mean_rmse_{a}_common" in R.columns]
    R = R[lead + arm_cols + [c for c in R.columns if c not in lead + arm_cols]] if len(R) else R
    return R, pd.DataFrame(rule_rows)


def method_arm_rmse(rmse, common):
    """Per method, over the common set: mean pooled RMSE in each arm, and on how many references each arm is the
    one with the method's lowest error. Descriptive: it shows which arm a method is closest to cell fractions in."""
    rows = []
    for m in ss.METHODS:
        d = rmse[(rmse["method"] == m) & rmse["reference"].isin(common) & (rmse["status"] == "ok")]
        row = {"method": m, "n_references": len(d)}
        for a in ARMS:
            row[f"mean_rmse_{a}"] = d[f"rmse_{a}"].mean()
        best = d[[f"rmse_{a}" for a in ARMS]].idxmin(axis=1).str[-1] if len(d) else pd.Series(dtype=str)
        for a in ARMS:
            row[f"n_lowest_in_{a}"] = int((best == a).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def driver_tables(root, sets, content):
    out = []
    for name, (hi, lo, _) in CONTRASTS.items():
        d = ss.driver_table(str(root), sets, content, hi, lo)
        if len(d):
            d = d.rename(columns={"set": "reference", "est_nobias": "est_baseline_arm", "est_bias": "est_bias_arm"})
            d.insert(0, "contrast", name)
            out.append(d)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def arm_library(root, sets):
    frames = [pd.read_csv(f, sep="\t") for f in sorted(glob.glob(str(root / "arm_check" / "*.tsv")))]
    if not frames:
        return pd.DataFrame(), pd.DataFrame()
    d = pd.concat(frames, ignore_index=True).rename(columns={"code": "reference"})
    g = d.groupby(["reference", "arm"])["mean_library_size_as_summed"]
    summ = g.agg(n_types="size", min="min", max="max").reset_index()
    summ["max_over_min"] = summ["max"] / summ["min"]
    missing = sorted(set(sets) - set(d["reference"]))
    if missing:
        summ = pd.concat([summ, pd.DataFrame({"reference": missing, "arm": "not checked"})], ignore_index=True)
    return d, summ


def vs_previous(rmse, previous):
    if not Path(previous).is_dir():
        return pd.DataFrame()
    sets = sorted(rmse["reference"].unique())
    old = ss.delta_table(str(previous), sets)                # bias - nobias, as run on 2026-11-11
    old = old.rename(columns={"set": "reference"}).set_index(["reference", "method"])
    rows = []
    for r in rmse.itertuples():
        o = old.loc[(r.reference, r.method)] if (r.reference, r.method) in old.index else None
        rows.append({"reference": r.reference, "method": r.method,
                     "reference_since_20261111": CHANGED.get(r.reference, "byte-identical"),
                     "rmse_B": r.rmse_B, "rmse_nobias_20261111": o["rmse_nobias"] if o is not None else np.nan,
                     "rmse_C": r.rmse_C, "rmse_bias_20261111": o["rmse_bias"] if o is not None else np.nan,
                     "d_rmse_C_minus_B": r.d_rmse_C_minus_B,
                     "d_rmse_20261111": o["d_rmse"] if o is not None else np.nan})
    return pd.DataFrame(rows)


def run_settings(root, sets):
    """One row per provenance file: what ran, from which commit, with which settings."""
    rows = []
    for s in sets:
        for a in ARMS:
            for f in sorted(glob.glob(str(root / s / a / "results" / "provenance_*.json"))):
                try:
                    rec = json.load(open(f))
                except Exception as e:                         # a file being written: report it, do not stop
                    rows.append({"reference": s, "arm": a, "file": os.path.basename(f), "error": str(e)})
                    continue
                p = rec.get("params", {})
                row = {"reference": s, "arm": a, "file": os.path.basename(f), "created": rec.get("created"),
                       "git_commit": rec.get("git_commit", "")[:7],
                       "n_modified_tracked_files": len(rec.get("git_modified_tracked_files") or []),
                       "host": rec.get("host"), "pipeline_profile": p.get("pipeline_profile"),
                       "exclude_list_md5": p.get("exclude_list_md5") or (p.get("gene_cleanup") or {}).get("exclude_list_md5"),
                       "n_inputs_with_md5": sum(1 for i in rec.get("inputs", []) if i.get("md5"))}
                if "method_settings" in p:
                    row["methods"] = ",".join(p.get("methods", []))
                    d = p["method_settings"].get("dwls")
                    if d:
                        row["dwls"] = f"{d['dwls_method']}, p {d['pval_cutoff']}, fold change {d['diff_cutoff']}, {d['ncores']} cores"
                        row["dwls_guard_fired_for"] = "; ".join(deescape(x) for x in
                                                                p["in_memory_patches"]["dwls"].get("guard_fired_for", []))
                    cap = p.get("reference_cap", {})
                    row["cell_cap"] = cap.get("max_cells_per_type"); row["cap_seed"] = cap.get("seed")
                    row["result"] = "; ".join(f"{m}: {v.get('status')}" for m, v in (p.get("method_result") or {}).items())
                    row["versions"] = "; ".join(f"{k} {v}" for k, v in (p.get("packages") or {}).items()
                                                if v and k in ("omnideconv", "DWLS", "MAST", "MuSiC", "SCDC", "BisqueRNA"))
                else:
                    row["methods"] = "bayesprism"
                    row["versions"] = f"BayesPrism {p.get('bayesprism')}"
                    row["cores"] = p.get("n_cores")
                rows.append(row)
    return pd.DataFrame(rows)


def r7(df):
    """Round the float columns to the seven decimals the tables are written with, so the summary, the figures and
    the results text all format the same stored number (a difference that sits on a rounding tie at four decimals
    otherwise shows as -0.0125 in one place and -0.0126 in another)."""
    f = df.select_dtypes("float").columns
    df[f] = df[f].round(7)
    return df


def fmt(x, n=4):
    return "n/a" if x is None or not np.isfinite(x) else f"{x:+.{n}f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default=str(ROOT / "data/deconvolution/simbu_bench_refs5"))
    ap.add_argument("--previous", default=str(ROOT / "data/deconvolution/simbu_bench_refs4all"))
    ap.add_argument("--out", default=str(ROOT / "docs/audit/remediation/record/step4/v5_simbu"))
    a = ap.parse_args()
    root, out = Path(a.root), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    sets = references(root)
    if not sets:
        sys.exit(f"no references with arms A, B or C under {root}")
    # seven decimals, so a number rounded again for the text or a figure label is the number itself rounded once
    w = lambda df, name: df.to_csv(out / name, sep="\t", index=False, float_format="%.7f")

    status = status_table(root, sets, a.previous if Path(a.previous).is_dir() else None); w(status, "v5_status.tsv")
    mstat = method_status(root, sets); w(mstat, "v5_method_status.tsv")
    rmse = r7(rmse_table(root, sets)); w(rmse, "v5_rmse.tsv")
    bp = rmse[rmse["method"] == "bp_rna"].drop(columns="method"); w(bp, "v5_bayesprism.tsv")
    common = common_set(rmse)
    rank, rule = ranking_and_rule(rmse, common); rank, rule = r7(rank), r7(rule)
    w(rank, "v5_ranking.tsv"); w(rule, "v5_corrector_rule.tsv")
    marm = r7(method_arm_rmse(rmse, common)); w(marm, "v5_method_arm_rmse.tsv")
    content = {s: ss.expressed_genes(str(root / s / "A" / "reference")) for s in sets
               if (root / s / "A" / "reference" / "reference_counts.mtx").exists()}
    if content:
        mc = pd.concat([c.assign(reference=s) for s, c in content.items()])[
            ["reference", "cell_type", "n_cells", "mean_expressed_genes", "mrna_rank", "mean_total_counts", "counts_rank"]]
        w(mc, "v5_mrna_content.tsv")
    drv = driver_tables(root, sets, content); drv = r7(drv) if len(drv) else drv; w(drv, "v5_driver.tsv")
    lib, libsum = arm_library(root, sets); w(lib, "v5_arm_library_size.tsv"); w(libsum, "v5_arm_library_size_summary.tsv")
    prev = vs_previous(rmse, a.previous); prev = r7(prev) if len(prev) else prev; w(prev, "v5_vs_20261111.tsv")
    runs = run_settings(root, sets); w(runs, "v5_run_settings.tsv")

    # ---------------------------------------------------------------- summary.txt
    L = [f"V5: SimBu mRNA-content bias with a baseline that has no bias. {len(sets)} references under {root}", ""]
    L.append(f"Arms built for all three: {int((status['arms_built'] == 'ABC').sum())}/{len(sets)}; "
             f"one reference across the arms: {int(status['same_reference'].sum())}/{len(sets)}; "
             f"one cell-fraction truth: {int(status['same_truth'].sum())}/{len(sets)}; "
             f"simulate call as registered: " + ", ".join(
                 f"{arm} {int(status[f'call_{arm}_as_registered'].sum())}/{len(sets)}" for arm in ARMS))
    if "B_mixtures_vs_20261111_nobias" in status:
        same = status[status["reference_since_20261111"] == "byte-identical"]
        L.append(f"Arms B and C against the nobias and bias mixtures of 2026-11-11, on the {len(same)} references that did "
                 f"not change: B identical {int((same['B_mixtures_vs_20261111_nobias'] == 'identical').sum())}/{len(same)}, "
                 f"C identical {int((same['C_mixtures_vs_20261111_bias'] == 'identical').sum())}/{len(same)}")
    nots = rmse[rmse["status"] != "ok"]
    L.append(f"Method runs scored: {int((rmse['status'] == 'ok').sum())} of {len(rmse)} reference x method (all three arms). "
             f"Not complete: {len(nots)}")
    for r in nots.itertuples():
        why = mstat[(mstat["reference"] == r.reference) & (mstat["method"] == r.method) & (mstat["status"] != "scored")]
        L.append(f"  {r.reference:8s} {NAMES[r.method]:10s} {r.status}: " + "; ".join(
            f"{q.arm} {q.status}" + (f" ({q.detail[:90]})" if q.detail else "") for q in why.itertuples()))
    L.append("Method runs by status (reference x arm x method): " + ", ".join(
        f"{k} {v}" for k, v in mstat["status"].value_counts().items()))
    if len(libsum):
        L.append("")
        L.append("Check of the arms (mean library size of a cell type's cells as SimBu sums them; largest over smallest "
                 "across the types of a reference):")
        for arm in ARMS:
            x = libsum[libsum["arm"] == arm]["max_over_min"]
            if len(x):
                L.append(f"  arm {arm}: {x.min():.4f} to {x.max():.4f} over {len(x)} references")
        nc = libsum[libsum["arm"] == "not checked"]
        if len(nc):
            L.append("  not checked: " + ", ".join(nc["reference"]))

    b = bp[np.isfinite(bp["d_rmse_B_minus_A"])]
    L += ["", f"== Reading 1. BayesPrism under native bias: RMSE(B) - RMSE(A) is positive on "
              f"{int((b['d_rmse_B_minus_A'] > 0).sum())} of {len(b)} references =="]
    for r in b.itertuples():
        L.append(f"  {r.reference:8s} A {r.rmse_A:.4f}  B {r.rmse_B:.4f}  B-A {fmt(r.d_rmse_B_minus_A)}")
    if len(b):
        x = b["d_rmse_B_minus_A"]
        L.append(f"  mean {fmt(x.mean())}, median {fmt(x.median())}, range {fmt(x.min())} to {fmt(x.max())}")
    c = bp[np.isfinite(bp["d_rmse_C_minus_A"])]
    L += ["", f"== Reading 2. BayesPrism under added bias: RMSE(C) - RMSE(A) is positive on "
              f"{int((c['d_rmse_C_minus_A'] > 0).sum())} of {len(c)} references =="]
    for r in c.itertuples():
        L.append(f"  {r.reference:8s} A {r.rmse_A:.4f}  C {r.rmse_C:.4f}  C-A {fmt(r.d_rmse_C_minus_A)}")
    if len(c):
        x = c["d_rmse_C_minus_A"]
        L.append(f"  mean {fmt(x.mean())}, median {fmt(x.median())}, range {fmt(x.min())} to {fmt(x.max())}")

    def rule_lines(name):
        out_lines = []
        for label in rank[rank["contrast"] == name]["common_set"].unique():
            rk = rank[(rank["contrast"] == name) & (rank["common_set"] == label)].sort_values("mean_d_rmse_common")
            cs = common if label == "all three arms" else sorted(ss.ranking(contrast_delta(rmse, name))[1])
            out_lines.append(f"  common set ({label}): {len(cs)} references: {'; '.join(cs)}")
            for q in rk.itertuples():
                out_lines.append(f"    {NAMES[q.method]:10s} common {fmt(q.mean_d_rmse_common)} (n={q.n_common})   "
                                 f"all scored {fmt(q.mean_d_rmse_all)}, median {fmt(q.median_d_rmse_all)} "
                                 f"(n={q.n_all}, {q.n_positive} > 0)")
            sel = rule[(rule["contrast"] == name) & (rule["common_set"] == label)] if len(rule) else rule
            if not len(sel):
                continue
            r = d = sel.iloc[0]
            out_lines.append(f"  rule (MuSiC and SCDC both below BayesPrism and CIBERSORTx): {r['verdict'].upper()} "
                             f"(MuSiC {fmt(d['music'])}, SCDC {fmt(d['scdc'])}; BayesPrism {fmt(d['bp_rna'])}, "
                             f"CIBERSORTx {fmt(d['cibersortx'])}; DWLS {fmt(d['dwls'])}, Bisque {fmt(d['bisque'])})")
            out_lines.append(f"  dropping one reference at a time gives the same verdict in {r['leave_one_out_same_verdict']} "
                             f"cases (descriptive)")
        return out_lines

    L += ["", "== Reading 3. Correctors: the registered rule on RMSE(C) - RMSE(A), mean over the common set =="]
    L += rule_lines("C-A")
    L += ["", f"   Mean pooled RMSE by arm over the common set ({len(common)} references), and the number of references "
              "where each arm is the method's lowest (descriptive):"]
    for r in marm.sort_values("mean_rmse_A").itertuples():
        L.append(f"    {NAMES[r.method]:10s} A {r.mean_rmse_A:.4f}  B {r.mean_rmse_B:.4f}  C {r.mean_rmse_C:.4f}   "
                 f"lowest in A {r.n_lowest_in_A}, B {r.n_lowest_in_B}, C {r.n_lowest_in_C}")
    L += ["", "   The native bias alone, RMSE(B) - RMSE(A), per method (a ranking; no rule is registered for it):"]
    L += rule_lines("B-A")
    L += ["", "== Reading 4. The 2026-11-11 readings, RMSE(C) - RMSE(B), on these references =="]
    L += rule_lines("C-B")
    cb = bp[np.isfinite(bp["d_rmse_C_minus_B"])]
    L.append(f"  BayesPrism RMSE(C) - RMSE(B) is positive on {int((cb['d_rmse_C_minus_B'] > 0).sum())} of {len(cb)} references")
    if len(drv):
        for name in ("C-B", "C-A", "B-A"):
            d = drv[(drv["contrast"] == name) & (drv["method"] == "bp_rna")]
            if not len(d) or "driver_mrna_rank" not in d:
                continue
            L.append(f"  BayesPrism's driver ({name}) is the highest-mRNA type (expressed genes) on "
                     f"{int((d['driver_mrna_rank'] == 1).sum())} of {len(d)} references; highest by total counts on "
                     f"{int((d['driver_counts_rank'] == 1).sum())} of {len(d)}")
        d = drv[(drv["contrast"] == "C-B") & (drv["method"] == "bp_rna")]
        rk = lambda v: "?" if v is None or not np.isfinite(v) else str(int(v))
        for r in d.itertuples():
            L.append(f"    {r.reference:8s} {r.driver[:36]:36s} rank {rk(r.driver_mrna_rank)}/{rk(r.n_types_reference)} "
                     f"(counts {rk(r.driver_counts_rank)}) {r.direction:8s} true {r.true_mean:.3f} "
                     f"est {r.est_baseline_arm:.3f} -> {r.est_bias_arm:.3f}")
    if len(prev):
        p = prev[prev["method"] == "bp_rna"].dropna(subset=["d_rmse_C_minus_B", "d_rmse_20261111"])
        if len(p):
            same = p[p["reference_since_20261111"] == "byte-identical"]
            L.append(f"  Against 2026-11-11 (BayesPrism, C - B here, bias - nobias then): largest absolute change "
                     f"{(p['d_rmse_C_minus_B'] - p['d_rmse_20261111']).abs().max():.4f} over {len(p)} references "
                     f"({(same['d_rmse_C_minus_B'] - same['d_rmse_20261111']).abs().max():.4f} over the {len(same)} whose "
                     f"reference did not change)")
    txt = "\n".join(L)
    (out / "summary.txt").write_text(txt + "\n")
    print(txt)
    print(f"\nwrote {out}/")


if __name__ == "__main__":
    main()
