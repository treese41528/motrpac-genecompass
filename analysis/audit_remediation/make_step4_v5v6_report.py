#!/usr/bin/env python
"""make_step4_v5v6_report.py -- write the results sections of docs/audit/remediation/STEP4_V5_V6_DRAFT.md from the
record tables, so every number in the text is read from the table named beside it.

  python analysis/audit_remediation/make_step4_v5v6_report.py
         [--v5 docs/audit/remediation/record/step4/v5_simbu] [--v6 docs/audit/remediation/record/step4/v6_mixture_panel]
         [--draft docs/audit/remediation/STEP4_V5_V6_DRAFT.md] [--print]

It replaces the text between the markers <!-- RESULTS:BEGIN --> and <!-- RESULTS:END --> of the draft (--print
writes it to the terminal instead). Run it after make_step4_v5_tables.py and make_step4_v6_tables.py. It states
what is finished and what is not: a battery whose runs are incomplete is headed INCOMPLETE and its readings are
given over the runs that exist, with the missing ones listed. The verdict words come from the tables
(v5_corrector_rule.tsv, v6_floor.tsv); this script applies no rule of its own.
"""
import argparse
import datetime
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BEGIN, END = "<!-- RESULTS:BEGIN -->", "<!-- RESULTS:END -->"
NAMES = {"bp_rna": "BayesPrism", "cibersortx": "CIBERSORTx", "dwls": "DWLS", "music": "MuSiC", "scdc": "SCDC",
         "bisque": "Bisque"}
ORDER = ["bp_rna", "cibersortx", "dwls", "music", "scdc", "bisque"]
REC5, REC6 = "record/step4/v5_simbu", "record/step4/v6_mixture_panel"


def read(path):
    if not path.exists():
        return None
    try:
        t = pd.read_csv(path, sep="\t")
    except pd.errors.EmptyDataError:
        return None
    return t


def sgn(x, n=4):
    return "n/a" if x is None or not np.isfinite(x) else f"{x:+.{n}f}"


def num(x, n=3):
    return "n/a" if x is None or not np.isfinite(x) else f"{x:.{n}f}"


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return out


# ------------------------------------------------------------------------------------------------ V5
def v5_section(rec):
    L = ["## V5 results: mRNA-content bias against a baseline that has no bias", ""]
    rmse, status = read(rec / "v5_rmse.tsv"), read(rec / "v5_status.tsv")
    if rmse is None or status is None:
        return L + ["No table yet: the battery has not been summarised (`make_step4_v5_tables.py`).", ""]
    ms, bp = read(rec / "v5_method_status.tsv"), read(rec / "v5_bayesprism.tsv")
    rule, rank, marm = read(rec / "v5_corrector_rule.tsv"), read(rec / "v5_ranking.tsv"), read(rec / "v5_method_arm_rmse.tsv")
    lib, drv, prev = read(rec / "v5_arm_library_size_summary.tsv"), read(rec / "v5_driver.tsv"), read(rec / "v5_vs_20261111.tsv")
    n_ref = len(status)
    counts = ms["status"].value_counts() if ms is not None else pd.Series(dtype=int)
    n_runs = int(counts.sum()); scored = int(counts.get("scored", 0)); na = int(counts.get("not applicable", 0))
    failed = int(counts.get("failed", 0)); notrun = int(counts.get("not run", 0))
    complete = n_ref == 15 and failed == 0 and notrun == 0
    b = bp.dropna(subset=["d_rmse_B_minus_A"]); c = bp.dropna(subset=["d_rmse_C_minus_A"]); cb = bp.dropna(subset=["d_rmse_C_minus_B"])
    r_ca = rule[(rule["contrast"] == "C-A") & (rule["common_set"] == "all three arms")] if rule is not None and len(rule) else None
    r_cb = rule[(rule["contrast"] == "C-B") & (rule["common_set"] == "all three arms")] if rule is not None and len(rule) else None

    if not complete:
        L += [f"**INCOMPLETE.** {scored} of {n_runs} method runs are scored over {n_ref} of 15 references "
              f"({na} not applicable, {failed} failed, {notrun} not run yet). The readings below cover the runs that "
              "exist and will change when the rest finish.", ""]
    # ---- lead
    lead = []
    if len(b):
        lead.append(f"Against a baseline with no mRNA bias, BayesPrism's error against cell fractions is higher under the "
                    f"references' native bias on {int((b['d_rmse_B_minus_A'] > 0).sum())} of {len(b)} references "
                    f"(mean RMSE {num(b['rmse_A'].mean(), 4)} without bias, {num(b['rmse_B'].mean(), 4)} with the native bias, "
                    f"{num(c['rmse_C'].mean(), 4)} with the added bias)")
    if r_ca is not None and len(r_ca):
        v = r_ca.iloc[0]
        lead.append(f"the registered corrector rule on arm C minus arm A is **{v['verdict']}** over the {int(v['n_common'])} "
                    f"references every method ran")
    if r_cb is not None and len(r_cb):
        lead.append(f"the 2026-11-11 reading (arm C minus arm B) is **{r_cb.iloc[0]['verdict']}** on these references")
    if lead:
        L += ["**Result.** " + "; ".join(lead) + ".", ""]

    # ---- what ran
    L += ["### What ran", ""]
    L.append(f"- {n_ref} references x 3 arms x 6 methods = {n_runs} method runs: {scored} scored, {na} not applicable, "
             f"{failed} failed, {notrun} not run (`{REC5}/v5_method_status.tsv`).")
    if ms is not None:
        for st, word in (("not applicable", "Not applicable"), ("failed", "Failed"), ("not run", "Not run")):
            d = ms[ms["status"] == st]
            if len(d):
                g = d.groupby(["reference", "method"])["arm"].apply(lambda x: "".join(sorted(x)))
                L.append(f"  - {word}: " + "; ".join(f"{NAMES[m]} on {r} (arm{'s' if len(a) > 1 else ''} {', '.join(a)})"
                                                     for (r, m), a in g.items()) + ".")
                if st == "not applicable":
                    L.append("    These references hold one sample; MuSiC and Bisque need at least two (the registered case).")
                if st == "failed":
                    for q in d.drop_duplicates(["reference", "method"]).itertuples():
                        L.append(f"    {NAMES[q.method]} on {q.reference}: {str(q.detail)[:200]}")
    L.append(f"- The three arms share one downsampled reference on {int(status['same_reference'].sum())} of {n_ref} references and "
             f"one cell-fraction truth on {int(status['same_truth'].sum())} of {n_ref}; the simulate call recorded in "
             f"`sim_meta.txt` is the registered one for arm A on {int(status['call_A_as_registered'].sum())}, arm B on "
             f"{int(status['call_B_as_registered'].sum())} and arm C on {int(status['call_C_as_registered'].sum())} of {n_ref} "
             f"(`{REC5}/v5_status.tsv`).")
    if "B_mixtures_vs_20261111_nobias" in status:
        same = status[status["reference_since_20261111"] == "byte-identical"]
        L.append(f"- On the {len(same)} references that did not change since 2026-11-11, arm B's mixtures are byte-identical to that "
                 f"battery's \"nobias\" mixtures on {int((same['B_mixtures_vs_20261111_nobias'] == 'identical').sum())} and arm C's "
                 f"to its \"bias\" mixtures on {int((same['C_mixtures_vs_20261111_bias'] == 'identical').sum())} (same table).")
    if lib is not None and len(lib):
        L += ["", "### Check: does arm A remove the library-size differences?", ""]
        chk = lib[lib["arm"].isin(list("ABC"))]
        row = []
        for arm in "ABC":
            x = chk[chk["arm"] == arm]["max_over_min"]
            row.append(f"arm {arm} {num(x.min(), 4)} to {num(x.max(), 4)}" if len(x) else f"arm {arm} not checked")
        a = chk[chk["arm"] == "A"]["max_over_min"]
        yes = len(a) and bool(np.allclose(a, 1.0, atol=1e-6))
        L.append(f"{'Yes' if yes else 'NOT ON EVERY REFERENCE'}. Through SimBu's own summation, the mean library size of 1,000 cells "
                 f"of one type, largest type over smallest type within a reference: {'; '.join(row)}, over "
                 f"{chk['reference'].nunique()} references (`{REC5}/v5_arm_library_size_summary.tsv`; per type in "
                 f"`v5_arm_library_size.tsv`; figure `figures/step4_v5_arm_check.png`). In arm A every cell contributes a total of 1.")
        nc = lib[lib["arm"] == "not checked"]
        if len(nc):
            L.append("Not checked: " + ", ".join(nc["reference"]) + ".")

    # ---- readings
    L += ["", "### Registered readings", ""]
    L.append(f"**1. BayesPrism under native bias (RMSE in arm B minus arm A).** Positive on "
             f"**{int((b['d_rmse_B_minus_A'] > 0).sum())} of {len(b)}** references"
             + (f"; mean {sgn(b['d_rmse_B_minus_A'].mean())}, range {sgn(b['d_rmse_B_minus_A'].min())} to "
                f"{sgn(b['d_rmse_B_minus_A'].max())}" if len(b) else "") + f" (`{REC5}/v5_bayesprism.tsv`).")
    L.append("")
    L.append(f"**2. BayesPrism under added bias (arm C minus arm A).** Positive on "
             f"**{int((c['d_rmse_C_minus_A'] > 0).sum())} of {len(c)}** references"
             + (f"; mean {sgn(c['d_rmse_C_minus_A'].mean())}, range {sgn(c['d_rmse_C_minus_A'].min())} to "
                f"{sgn(c['d_rmse_C_minus_A'].max())}" if len(c) else "") + " (same table; figure "
             "`figures/step4_v5_bayesprism_arms.png`).")
    L.append("")
    L += table(["Reference", "RMSE A", "RMSE B", "RMSE C", "B - A", "C - A", "C - B"],
               [[r.reference, num(r.rmse_A, 4), num(r.rmse_B, 4), num(r.rmse_C, 4), sgn(r.d_rmse_B_minus_A), sgn(r.d_rmse_C_minus_A),
                 sgn(r.d_rmse_C_minus_B)] for r in bp.itertuples()])
    L.append("")

    def rule_block(name, title):
        out = [title, ""]
        if rule is None or rank is None or not len(rule):
            return out + ["Not available yet.", ""]
        for _, v in rule[rule["contrast"] == name].iterrows():
            rk = rank[(rank["contrast"] == name) & (rank["common_set"] == v["common_set"])].set_index("method")
            out.append(f"Common set ({v['common_set']}): {int(v['n_common'])} references ({v['references']}).")
            out.append("")
            out += table(["Method", "Mean difference, common set", "Mean, all it scored", "References scored", "of which positive"],
                         [[NAMES[m], sgn(rk.loc[m, "mean_d_rmse_common"]), sgn(rk.loc[m, "mean_d_rmse_all"]), int(rk.loc[m, "n_all"]),
                           int(rk.loc[m, "n_positive"])] for m in sorted(rk.index, key=lambda m: rk.loc[m, "mean_d_rmse_common"])])
            out.append("")
            out.append(f"Rule: reproduced if MuSiC and SCDC are both below BayesPrism and CIBERSORTx. MuSiC {sgn(v['music'])}, SCDC "
                       f"{sgn(v['scdc'])}; BayesPrism {sgn(v['bp_rna'])}, CIBERSORTx {sgn(v['cibersortx'])}. **{v['verdict'].capitalize()}.** "
                       f"Dropping one reference at a time leaves the verdict unchanged in {v['leave_one_out_same_verdict']} cases "
                       f"(descriptive). Tables: `{REC5}/v5_corrector_rule.tsv`, `v5_ranking.tsv`.")
            out.append("")
        return out

    L += rule_block("C-A", "**3. Correctors: the registered rule on arm C minus arm A.**")
    L += rule_block("C-B", "**4. The 2026-11-11 readings (arm C minus arm B) on the refs5 references.**")
    if len(cb):
        L.append(f"- BayesPrism's RMSE rises from arm B to arm C on {int((cb['d_rmse_C_minus_B'] > 0).sum())} of {len(cb)} references "
                 f"(2026-11-11: 15 of 15), by {sgn(cb['d_rmse_C_minus_B'].min())} to {sgn(cb['d_rmse_C_minus_B'].max())}.")
    if drv is not None and len(drv) and "driver_mrna_rank" in drv:
        d = drv[(drv["contrast"] == "C-B") & (drv["method"] == "bp_rna")]
        if len(d):
            L.append(f"- BayesPrism's driver (the type with the largest per-type rise) is the type with the most expressed genes on "
                     f"{int((d['driver_mrna_rank'] == 1).sum())} of {len(d)} references (2026-11-11: 10 of 15); by total counts on "
                     f"{int((d['driver_counts_rank'] == 1).sum())} of {len(d)} (`{REC5}/v5_driver.tsv`).")
            L.append("")
            L += table(["Reference", "Driver (C - B)", "mRNA rank (genes; counts)", "True", "Estimate B -> C", "Direction"],
                       [[r.reference, r.driver, f"{int(r.driver_mrna_rank)} of {int(r.n_types_reference)}; {int(r.driver_counts_rank)}",
                         num(r.true_mean), f"{num(r.est_baseline_arm)} -> {num(r.est_bias_arm)}", r.direction] for r in d.itertuples()])
            L.append("")
    if prev is not None and len(prev):
        p = prev[prev["method"] == "bp_rna"].dropna(subset=["d_rmse_C_minus_B", "d_rmse_20261111"])
        if len(p):
            same = p[p["reference_since_20261111"] == "byte-identical"]
            L.append(f"- Against the battery of 2026-11-11 (BayesPrism, arm C minus arm B here, bias minus nobias then): the largest "
                     f"absolute change is {num((p['d_rmse_C_minus_B'] - p['d_rmse_20261111']).abs().max(), 4)} over {len(p)} references"
                     + (f", and {num((same['d_rmse_C_minus_B'] - same['d_rmse_20261111']).abs().max(), 4)} over the {len(same)} whose "
                        f"reference did not change (only the exclusion list differs there)" if len(same) else "")
                     + f" (`{REC5}/v5_vs_20261111.tsv`).")
    # ---- descriptive
    if marm is not None and len(marm):
        L += ["", "### Each method in the three arms (descriptive, not a registered reading)", ""]
        nref = int(marm["n_references"].max())
        L.append(f"Mean pooled RMSE against cell fractions over the {nref} references every method ran, and the number of those "
                 f"references on which each arm is the method's lowest (`{REC5}/v5_method_arm_rmse.tsv`; figure "
                 "`figures/step4_v5_methods_arms.png`).")
        L.append("")
        L += table(["Method", "Arm A", "Arm B", "Arm C", "Lowest in A", "Lowest in B", "Lowest in C"],
                   [[NAMES[r.method], num(r.mean_rmse_A, 4), num(r.mean_rmse_B, 4), num(r.mean_rmse_C, 4), r.n_lowest_in_A, r.n_lowest_in_B,
                     r.n_lowest_in_C] for r in marm.sort_values("mean_rmse_A").itertuples()])
        low_b = [NAMES[r.method] for r in marm.itertuples() if r.n_lowest_in_B > max(r.n_lowest_in_A, r.n_lowest_in_C)]
        low_a = [NAMES[r.method] for r in marm.itertuples() if r.n_lowest_in_A > max(r.n_lowest_in_B, r.n_lowest_in_C)]
        L.append("")
        if low_b:
            L.append(f"{', '.join(low_b)} {'is' if len(low_b) == 1 else 'are'} closest to cell fractions in arm B, not in arm A, on most "
                     "references: for such a method arm A is not the easy case, and its arm C minus arm A difference is small or negative "
                     "because its arm A error is high, not because it corrects the added bias. "
                     + (f"{', '.join(low_a)} {'is' if len(low_a) == 1 else 'are'} closest in arm A." if low_a else ""))
    L.append("")
    return L


# ------------------------------------------------------------------------------------------------ V6
def v6_section(rec):
    L = ["## V6 results: mixture panel on the refs5 references", ""]
    fl = read(rec / "v6_floor.tsv")
    if fl is None:
        return L + ["No table yet: the panel has not been summarised (`make_step4_v6_tables.py`).", ""]
    st, vs = read(rec / "v6_status.tsv"), read(rec / "v6_vs_20261108.tsv")
    att, dw, runs, bc = (read(rec / "v6_heart_hypoth_types.tsv"), read(rec / "v6_dwls_setting.tsv"),
                         read(rec / "v6_run_settings.tsv"), read(rec / "v6_build_check.tsv"))
    med, pool = read(rec / "panel_median.tsv"), read(rec / "panel_pooled.tsv")
    sc = fl[fl["verdict"] != "not scored"]
    n_ok, n_weak, n_no = int((sc["verdict"] == "recovers").sum()), int((sc["verdict"] == "weak").sum()), int((fl["verdict"] == "not scored").sum())
    counts = st["status"].value_counts() if st is not None else pd.Series(dtype=int)
    complete = len(fl) == 19 and n_no == 0 and int(counts.get("not run", 0)) == 0 and int(counts.get("failed", 0)) == 0
    if not complete:
        L += [f"**INCOMPLETE.** BayesPrism is scored on {len(sc)} of {len(fl)} sets built (19 registered); method runs: "
              + ", ".join(f"{k} {v}" for k, v in counts.items()) + ". The readings below cover the runs that exist.", ""]
    if len(sc):
        h, x = sc[sc["design"] == "holdout"]["bayesprism_median_r"], sc[sc["design"] == "cross"]["bayesprism_median_r"]
        rng = "; ".join(s for s in [f"{num(h.min())} to {num(h.max())} on the {len(h)} holdout sets" if len(h) else "",
                                    f"{num(x.min())} to {num(x.max())} on the {len(x)} cross sets" if len(x) else ""] if s)
        if n_weak == 0:
            L += [f"**Result.** The registered floor is met on every set scored: BayesPrism's median per-type r is {rng}, against "
                  f"0.85 ({n_ok} of {len(sc)}).", ""]
        else:
            w = sc[sc["verdict"] == "weak"]
            L += [f"**Result.** The registered floor is NOT met on {n_weak} of {len(sc)} sets scored: "
                  + "; ".join(f"{r.set} {num(r.bayesprism_median_r)} (worst types: {r.worst_types})" for r in w.itertuples())
                  + f". BayesPrism's median per-type r is {rng}.", ""]
    L += ["### Pass rule (registered, unchanged): BayesPrism's median per-type r of at least 0.85", ""]
    rows = []
    for r in fl.itertuples():
        m = med.set_index(med.columns[0]).loc[r.set] if med is not None and r.set in set(med.iloc[:, 0]) else None
        rows.append([r.set, num(r.bayesprism_median_r), r.verdict, num(r.bayesprism_pooled_r),
                     "" if r.verdict == "not scored" else f"{int(r.n_types_in_median)} of {int(r.n_types_scored)}",
                     "" if r.verdict == "not scored" else int(r.n_types_below_floor)]
                    + [("-" if m is None or k not in m or not np.isfinite(m[k]) else num(m[k])) for k in ORDER[1:]])
    L += table(["Set", "BayesPrism median r", "Verdict", "BayesPrism pooled r", "Types in the median", "Types below 0.85"]
               + [NAMES[k] for k in ORDER[1:]], rows)
    L += ["", f"Median per-type r of the other methods is descriptive; \"-\" is not scored (see the status below). Tables: "
              f"`{REC6}/v6_floor.tsv`, `panel_median.tsv`, `panel_pooled.tsv`, `panel_per_type.tsv`; figure "
              "`figures/step4_v6_median_r.png`.", ""]
    below = sc[sc["n_types_below_floor"] > 0]
    if len(below):
        L.append("Types below 0.85 for BayesPrism (a set can meet the floor on its median and still have such types): "
                 + "; ".join(f"{r.set}: {r.worst_types}" for r in below.itertuples()) + ". The list gives each set's three lowest types.")
        L.append("")

    if vs is not None and len(vs):
        b = vs[vs["method"] == "bp_rna"].dropna(subset=["median_change"])
        L += ["### Against the panel of 2026-11-08 (descriptive, as registered)", ""]
        if len(b):
            drops = b[b["median_change"] < -0.05]
            L.append(f"BayesPrism's median per-type r, {len(b)} sets compared: a drop of more than 0.05 on **{len(drops)}**"
                     + ("" if drops.empty else " (" + ", ".join(f"{r.set} {sgn(r.median_change, 3)}" for r in drops.itertuples()) + ")")
                     + f"; largest drop {sgn(b['median_change'].min())} ({b.loc[b['median_change'].idxmin(), 'set']}), largest rise "
                     f"{sgn(b['median_change'].max())} ({b.loc[b['median_change'].idxmax(), 'set']}) (`{REC6}/v6_vs_20261108.tsv`).")
        else:
            L.append("Nothing to compare yet.")
        if bc is not None and len(bc):
            diff = bc[bc["mixtures_vs_previous"] == "different"]["set"].tolist()
            cr = bc[bc["design"] == "cross"]
            L.append(f"The mixtures themselves are byte-identical to that panel's on {int((bc['mixtures_vs_previous'] == 'identical').sum())} "
                     f"of {len(bc)} sets" + (f"; they differ on {', '.join(diff)}" if diff else "") + ". The cross harmonization equals "
                     f"the reviewed maps on {int((cr['harmonization_map'] == 'identical').sum())} of {len(cr)} cross sets "
                     f"(`{REC6}/v6_build_check.tsv`). Where the mixtures are identical, a change in r comes from the exclusion list "
                     "alone.")
        L.append("")

    L += ["### HEART and HYPOTH, every type (their labels changed)", ""]
    for s in ("HEART_holdout", "HYPOTH_holdout"):
        x = att[att["set"] == s] if att is not None else None
        if x is None or not len(x):
            L.append(f"- **{s}**: not scored yet.")
            continue
        now = x.dropna(subset=["r_now"])
        new = now[~now["in_20261108_panel"].astype(bool)]["cell_type"].tolist()
        gone = x[x["r_now"].isna()]["cell_type"].tolist()
        low = now[now["r_now"] < 0.85]
        L.append(f"- **{s}**: {len(now)} types scored, r {num(now['r_now'].min())} to {num(now['r_now'].max())}; below 0.85: "
                 + (", ".join(f"{r.cell_type} {num(r.r_now)}" for r in low.itertuples()) if len(low) else "none")
                 + f". New since 2026-11-08: {', '.join(new) or 'none'}. Benchmarked then, not now: {', '.join(gone) or 'none'}.")
    L += [f"  Table: `{REC6}/v6_heart_hypoth_types.tsv`; figure `figures/step4_v6_heart_hypoth.png`.", ""]
    if att is not None and len(att):
        L += table(["Set", "Type", "r now", "r 2026-11-08", "True mean now", "Estimated mean now"],
                   [[r.set, r.cell_type, num(r.r_now), num(r.r_20261108), num(r.true_mean_now), num(r.est_mean_now)]
                    for r in att.sort_values(["set", "r_now"]).itertuples()])
        L.append("")

    if dw is not None and len(dw):
        L += ["### DWLS as the benchmark plan specified", ""]
        d = dw.dropna(subset=["median_r_mast_optimized_p0.05"])
        if len(d):
            both = d.dropna(subset=["median_change"])
            L.append(f"`mast_optimized` at p 0.05 is scored on {len(d)} of {len(dw)} sets: median per-type r "
                     f"{num(d['median_r_mast_optimized_p0.05'].min())} to {num(d['median_r_mast_optimized_p0.05'].max())}. Against `mast` at "
                     f"p 0.01 in the 2026-11-08 panel the change is {sgn(both['median_change'].min(), 3)} to "
                     f"{sgn(both['median_change'].max(), 3)} over {len(both)} sets (`{REC6}/v6_dwls_setting.tsv`). The comparison also "
                     "carries the change of exclusion list and, for HEART and HYPOTH, of labels.")
            if "setting_recorded" in d:
                L.append("Setting read back from the provenance files: "
                         + "; ".join(f"{k} ({v} sets)" for k, v in d["setting_recorded"].value_counts().items()) + ".")
            if "guard_fired_for" in d:
                g = d[d["guard_fired_for"].fillna("") != ""]
                L.append(f"The in-memory guard fired on {len(g)} sets" + ("." if g.empty else ": " + "; ".join(
                    f"{r.set} ({r.guard_fired_for})" for r in g.itertuples()) + ". Those types have no marker genes of their own in "
                    "the DWLS signature."))
        else:
            L.append("Not scored yet.")
        L.append("")

    if st is not None:
        L += ["### Status of every method run", ""]
        L.append(", ".join(f"{k} {v}" for k, v in counts.items()) + f" of {len(st)} (19 sets x 6 methods; `{REC6}/v6_status.tsv`).")
        for word in ("not applicable", "failed", "not run"):
            d = st[st["status"] == word]
            if len(d):
                L.append(f"- {word.capitalize()}: " + "; ".join(f"{NAMES[r.method]} on {r.set}" for r in d.itertuples()) + ".")
                if word == "failed":
                    for r in d.itertuples():
                        L.append(f"  - {NAMES[r.method]} on {r.set}: {str(r.detail)[:200]}")
        L.append("")
    if runs is not None and len(runs):
        oc = runs[runs["runner"] == "run_omnideconv.R"] if "runner" in runs else runs.iloc[0:0]
        L += ["### Provenance records (finding 5.6)", ""]
        mod = runs[runs["modified_tracked_files"].fillna("") != ""]
        L.append(f"{len(runs)} records: {len(oc)} from the cross-check runner and {len(runs) - len(oc)} from the Stage 8 runner. "
                 f"Commits: {', '.join(sorted(set(runs['git_commit'].dropna())))}. Records with a modified tracked file: {len(mod)}"
                 + ("" if mod.empty else " (" + "; ".join(sorted(set(mod["modified_tracked_files"]))) + ")") + ". Each record of the "
                 "cross-check runner holds the commit, the inputs with md5, the methods and their settings, package versions, the "
                 f"cell cap and its seed, and the in-memory patches (`{REC6}/v6_run_settings.tsv`).")
        L.append("")
    return L


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--v5", default="docs/audit/remediation/record/step4/v5_simbu")
    ap.add_argument("--v6", default="docs/audit/remediation/record/step4/v6_mixture_panel")
    ap.add_argument("--draft", default="docs/audit/remediation/STEP4_V5_V6_DRAFT.md")
    ap.add_argument("--print", action="store_true", help="print the section instead of writing it into the draft")
    ap.add_argument("--date", default=datetime.date.today().isoformat(), help="date printed in the heading")
    a = ap.parse_args()
    body = [BEGIN, "", f"*This part is written by `analysis/audit_remediation/make_step4_v5v6_report.py` from the record tables "
                       f"(last run {a.date}). Paths are relative to `docs/audit/remediation/`.*", ""]
    body += v5_section(Path(a.v5)) + v6_section(Path(a.v6)) + [END]
    text = "\n".join(body)
    if a.print:
        print(text)
        return
    draft = Path(a.draft)
    s = draft.read_text()
    if BEGIN in s and END in s:
        s = s[:s.index(BEGIN)] + text + s[s.index(END) + len(END):]
    else:
        sys.exit(f"{draft} has no {BEGIN} ... {END} markers")
    draft.write_text(s)
    print(f"wrote the results part of {draft}")


if __name__ == "__main__":
    main()
