#!/usr/bin/env python
"""make_step4_v6_tables.py -- every table of V6, the known-truth mixture panel on the refs5 references
(docs/audit/remediation/STEP4_PREREG.md V6; audit findings 5.5 and 5.6), from the finished score files.

  python analysis/audit_remediation/make_step4_v6_tables.py
         [--panel data/deconvolution/mixture_panel_refs5] [--previous data/deconvolution/mixture_panel_refs4all]
         [--out docs/audit/remediation/record/step4/v6_mixture_panel]

It first runs the panel's own summary (deconvolution/mixture_panel_summary.py, unchanged readings of
MIXTURE_PANEL_REFS4ALL_PREREG.md) into --out, which writes panel_median.tsv, panel_pooled.tsv,
panel_per_type.tsv, panel_scoring_notes.tsv, panel_coverage.tsv, panel_vs_earlier.tsv (against the June/July
panel, validation_v2) and summary.txt. Then it adds what V6 asks for:

  v6_floor.tsv             the pass rule, per set: BayesPrism's median per-type r against the registered floor
                           0.85; the verdict; the three worst types where it is weak; "not scored" where the
                           run has not finished
  v6_status.tsv            per set and method: scored, not applicable (MuSiC and Bisque on a one-sample
                           reference), failed (with the runner's message) or not run yet
  v6_vs_20261108.tsv       per set and method: median per-type r and pooled r now and in the panel of 2026-11-08
                           on the references then in use; for BayesPrism, whether the median dropped by more
                           than 0.05 (the registered descriptive comparison; note N9)
  v6_heart_hypoth_types.tsv  every type of HEART_holdout and HYPOTH_holdout: BayesPrism r, true and estimated
                           mean now and on 2026-11-08 (their labels changed)
  v6_dwls_setting.tsv      DWLS per set: mast_optimized at p 0.05 (now) beside mast at p 0.01 (2026-11-08)
  v6_run_settings.tsv      one row per provenance file: commit, modified tracked files, methods and settings, cap
                           and seed, where the DWLS guard fired, package versions (finding 5.6)
  v6_build_check.tsv       check_panel_build.py: cross harmonization against the reviewed maps; which sets'
                           mixtures are byte-identical to the 2026-11-08 panel's
  v6_summary.txt           the registered reading, then the descriptive ones
Nothing is truncated: every set and method found on disk is reported, and every missing one is listed.
"""
import argparse
import glob
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "deconvolution"))
import mixture_panel_summary as mps  # noqa: E402

FLOOR = 0.85            # MIXTURE_PANEL_REFS4ALL_PREREG.md, "Reading": median per-type r >= 0.85
DROP = 0.05             # same section: a drop of more than 0.05 against the earlier panel is named
NAMES = {"bp_rna": "BayesPrism", "cibersortx": "CIBERSORTx", "dwls": "DWLS", "music": "MuSiC", "scdc": "SCDC",
         "bisque": "Bisque"}
ATTENTION = ["HEART_holdout", "HYPOTH_holdout"]     # labels changed in refs5
# which runner invocation produces a method (file name of its provenance record under results/)
PROV = {"bp_rna": "provenance_run_deconvolution.json", "dwls": "provenance_run_omnideconv_dwls.json",
        "music": "provenance_run_omnideconv_music+scdc+bisque.json", "scdc": "provenance_run_omnideconv_music+scdc+bisque.json",
        "bisque": "provenance_run_omnideconv_music+scdc+bisque.json", "cibersortx": "provenance_run_omnideconv_cibersortx.json"}


# omnideconv replaces special characters in cell-type names by fixed tokens before it calls a method and puts them
# back in what it returns (omnideconv 0.1.1, escape_special_chars). The DWLS guard reports a type from inside
# DWLS, so its name arrives with the tokens; this puts the characters back for the tables.
_ESC_SUFFIX = "7f8711ec9bf265fb9bf6ab9c"
_ESC = {" ": "21b29fb0", "!": "21b29fb1", '"': "21b29fb2", "#": "21b29fb3", "$": "21b29fb4", "%": "21b29fb5",
        "&": "21b29fb6", "'": "21b29fb7", "(": "21b29fb8", ")": "21b29fb9", "*": "21b29fba", "+": "21b2c6e8",
        ",": "21b2c6e9", "-": "21b2c756", ".": "21b2c757", "/": "21b2c758", ":": "21b2c759", ";": "21b2c75a",
        "<": "21b2c75b", "=": "21b2c75c", ">": "21b2c75d", "?": "21b2c75e", "@": "21b2c75f", "[": "21b2c760",
        "\\": "21b2ee34", "]": "21b2ee35", "^": "21b2ee36", "_": "21b2ee37", "`": "21b2ee38", "\u00b4": "21b2ee39",
        "{": "21b2ee3a", "|": "21b2ee3b", "}": "21b2ee3c", "~": "21b2ee3d", "\u00a7": "21b2ee3e"}


def deescape(name):
    """'Brown21b29fb07f8711ec9bf265fb9bf6ab9cadipocytes' -> 'Brown adipocytes'."""
    name = str(name)
    for ch, tok in _ESC.items():
        name = name.replace(tok + _ESC_SUFFIX, ch)
    return name


def panel_sets(panel):
    return sorted([p.name for p in Path(panel).iterdir() if p.is_dir() and p.name.endswith(("_holdout", "_cross"))],
                  key=lambda s: (s.endswith("_cross"), s))


def per_type(sd, m):
    f = Path(sd) / "scores" / f"metrics_{m}.tsv"
    return pd.read_csv(f, sep="\t") if f.exists() else None


def n_reference_samples(sd, panel_set_ref=None):
    """Samples in the reference a set is deconvolved against (MuSiC and Bisque need more than one)."""
    f = Path(sd) / "reference" / "cells_meta.tsv"
    if not f.exists() and panel_set_ref:
        f = Path(panel_set_ref) / "cells_meta.tsv"
    if not f.exists():
        return np.nan
    return pd.read_csv(f, sep="\t", usecols=["sample"], dtype=str)["sample"].nunique()


def cross_reference_dir(sd):
    """The production reference a cross set was built against, from its build log line '[cross] ... vs <dir>' is
    not kept per set; read it from any omnideconv/BayesPrism provenance file instead (params.ref_dir)."""
    for f in sorted(glob.glob(str(Path(sd) / "results" / "provenance_*.json"))):
        try:
            return json.load(open(f)).get("params", {}).get("ref_dir")
        except Exception:
            continue
    return None


def run_status(sd, m, n_samples):
    """What happened to method m on the set in folder sd: (status, detail).
    scored          its score file exists
    not applicable  MuSiC or Bisque gave no output and the reference has one sample (the registered case)
    failed          the runner's provenance record says the method ran and gave no output (its message is the detail)
    not run         no score and no record of a run
    The distinction between 'not applicable' and 'failed' is made here from the record, not by hand."""
    sd = Path(sd)
    if (sd / "scores" / f"overall_metrics_{m}.tsv").exists():
        return "scored", ""
    pf = sd / "results" / PROV[m]
    if m == "bp_rna" or not pf.exists():
        return "not run", ""
    try:
        rec = json.load(open(pf)).get("params", {})
    except Exception as e:
        return "not run", f"unreadable provenance: {e}"
    res = (rec.get("method_result") or {}).get(m)
    if res is None:
        return "not run", ""
    msg = str(res.get("error") or "")
    if m in ("music", "bisque") and n_samples == 1:
        return "not applicable", f"one-sample reference; runner message: {msg[:160]}"
    return "failed", msg[:300]


def status_table(panel, sets):
    rows = []
    for s in sets:
        sd = Path(panel) / s
        ref = cross_reference_dir(sd) if s.endswith("_cross") else None
        ns = n_reference_samples(sd, (ROOT / ref) if ref and not os.path.isabs(ref) else ref)
        for m in mps.METHODS:
            status, detail = run_status(sd, m, ns)
            rows.append({"set": s, "method": m, "n_reference_samples": ns, "status": status, "detail": detail})
    return pd.DataFrame(rows)


def floor_table(panel, sets):
    rows = []
    for s in sets:
        o = mps.overall(Path(panel) / s, "bp_rna")
        t = per_type(Path(panel) / s, "bp_rna")
        row = {"set": s, "design": "cross" if s.endswith("_cross") else "holdout", "floor": FLOOR}
        if o is None:
            row.update(bayesprism_median_r=np.nan, bayesprism_pooled_r=np.nan, verdict="not scored")
        else:
            med, pooled = o
            row.update(bayesprism_median_r=med, bayesprism_pooled_r=pooled,
                       verdict="recovers" if med >= FLOOR else "weak",
                       n_types_scored=len(t), n_types_in_median=int(t["pearson_r"].notna().sum()),
                       min_type_r=t["pearson_r"].min(),
                       n_types_below_floor=int((t["pearson_r"] < FLOOR).sum()),
                       n_types_r_ge_095=int((t["pearson_r"] >= 0.95).sum()),
                       n_types_r_ge_090=int((t["pearson_r"] >= 0.90).sum()),
                       worst_types="; ".join(f"{x.cell_type} {x.pearson_r:.3f}" for x in
                                             t.nsmallest(3, "pearson_r").itertuples()))
        rows.append(row)
    return pd.DataFrame(rows)


def vs_previous(panel, previous, sets):
    rows = []
    for s in sets:
        for m in mps.METHODS:
            now, old = mps.overall(Path(panel) / s, m), mps.overall(Path(previous) / s, m)
            if now is None and old is None:
                continue
            row = {"set": s, "method": m,
                   "median_r_now": now[0] if now else np.nan, "median_r_20261108": old[0] if old else np.nan,
                   "pooled_r_now": now[1] if now else np.nan, "pooled_r_20261108": old[1] if old else np.nan}
            row["median_change"] = row["median_r_now"] - row["median_r_20261108"]
            row["pooled_change"] = row["pooled_r_now"] - row["pooled_r_20261108"]
            row["median_drop_gt_0.05"] = bool(row["median_change"] < -DROP) if np.isfinite(row["median_change"]) else ""
            rows.append(row)
    return pd.DataFrame(rows)


def attention_types(panel, previous, which=ATTENTION):
    rows = []
    for s in which:
        now, old = per_type(Path(panel) / s, "bp_rna"), per_type(Path(previous) / s, "bp_rna")
        if now is None:
            continue
        old = old.set_index("cell_type") if old is not None else pd.DataFrame()
        for r in now.itertuples():
            o = old.loc[r.cell_type] if r.cell_type in old.index else None
            rows.append({"set": s, "cell_type": r.cell_type, "in_20261108_panel": o is not None,
                         "r_now": r.pearson_r, "r_20261108": o["pearson_r"] if o is not None else np.nan,
                         "true_mean_now": r.true_mean, "est_mean_now": r.est_mean,
                         "true_mean_20261108": o["true_mean"] if o is not None else np.nan,
                         "est_mean_20261108": o["est_mean"] if o is not None else np.nan})
        for t in sorted(set(old.index) - set(now["cell_type"])):
            rows.append({"set": s, "cell_type": t, "in_20261108_panel": True, "r_now": np.nan,
                         "r_20261108": old.loc[t, "pearson_r"], "true_mean_now": np.nan, "est_mean_now": np.nan,
                         "true_mean_20261108": old.loc[t, "true_mean"], "est_mean_20261108": old.loc[t, "est_mean"],
                         "note": "benchmarked on 2026-11-08, not now"})
    return pd.DataFrame(rows, columns=["set", "cell_type", "in_20261108_panel", "r_now", "r_20261108", "true_mean_now",
                                       "est_mean_now", "true_mean_20261108", "est_mean_20261108", "note"])


def dwls_table(panel, previous, sets):
    rows = []
    for s in sets:
        now, old = mps.overall(Path(panel) / s, "dwls"), mps.overall(Path(previous) / s, "dwls")
        bp = mps.overall(Path(panel) / s, "bp_rna")
        row = {"set": s, "median_r_mast_optimized_p0.05": now[0] if now else np.nan,
               "median_r_mast_p0.01_20261108": old[0] if old else np.nan,
               "pooled_r_mast_optimized_p0.05": now[1] if now else np.nan,
               "pooled_r_mast_p0.01_20261108": old[1] if old else np.nan,
               "bayesprism_median_r_now": bp[0] if bp else np.nan}
        row["median_change"] = row["median_r_mast_optimized_p0.05"] - row["median_r_mast_p0.01_20261108"]
        pf = Path(panel) / s / "results" / PROV["dwls"]
        if pf.exists():
            try:
                p = json.load(open(pf))["params"]
                d = p["method_settings"]["dwls"]
                row["setting_recorded"] = f"{d['dwls_method']}, p {d['pval_cutoff']}, fold change {d['diff_cutoff']}"
                row["guard_fired_for"] = "; ".join(deescape(x) for x in p["in_memory_patches"]["dwls"].get("guard_fired_for", []))
                row["minutes"] = (p.get("method_result") or {}).get("dwls", {}).get("minutes")
            except Exception as e:
                row["setting_recorded"] = f"unreadable provenance: {e}"
        rows.append(row)
    return pd.DataFrame(rows)


def run_settings(panel, sets):
    rows = []
    for s in sets:
        for f in sorted(glob.glob(str(Path(panel) / s / "results" / "provenance_*.json"))):
            try:
                rec = json.load(open(f))
            except Exception as e:
                rows.append({"set": s, "file": os.path.basename(f), "error": str(e)})
                continue
            p = rec.get("params", {})
            row = {"set": s, "file": os.path.basename(f), "created": rec.get("created"),
                   "git_commit": rec.get("git_commit", "")[:7],
                   "modified_tracked_files": "; ".join(rec.get("git_modified_tracked_files") or []),
                   "host": rec.get("host"), "n_inputs": len(rec.get("inputs", [])),
                   "n_inputs_with_md5": sum(1 for i in rec.get("inputs", []) if i.get("md5")),
                   "reference_md5": next((i.get("md5", "") for i in rec.get("inputs", [])
                                          if i["path"].endswith("reference_counts.mtx")), ""),
                   "mixtures_md5": next((i.get("md5", "") for i in rec.get("inputs", [])
                                         if re.search(r"pseudobulk_counts\.mtx$", i["path"])), ""),
                   "pipeline_profile": p.get("pipeline_profile", "")}
            if "method_settings" in p:                           # the cross-check runner
                row["runner"] = "run_omnideconv.R"
                row["methods"] = ",".join(p.get("methods", []))
                ms = p["method_settings"]
                parts = []
                if "dwls" in ms:
                    d = ms["dwls"]
                    parts.append(f"dwls: {d['dwls_method']}, p {d['pval_cutoff']} ({d['pval_cutoff_set_by']}), "
                                 f"fold change {d['diff_cutoff']}, {d['dwls_submethod']}, {d['ncores']} cores")
                if "cibersortx" in ms:
                    c = ms["cibersortx"]
                    img = "; ".join(f"{os.path.basename(i['file'])} md5 {i['md5']}" for i in c.get("container_image", []))
                    parts.append(f"cibersortx: B-mode {c['rmbatch_B_mode']}, S-mode {c['rmbatch_S_mode']}, container {img}")
                for m in ("music", "scdc", "bisque"):
                    if m in ms:
                        parts.append(f"{m}: {ms[m]['call']}, batches {ms[m]['n_batches']}")
                row["settings"] = " | ".join(parts)
                cap = p.get("reference_cap", {})
                row["cell_cap"] = cap.get("max_cells_per_type"); row["cap_seed"] = cap.get("seed")
                row["cells_before_cap"] = cap.get("n_cells_before"); row["cells_after_cap"] = cap.get("n_cells_after")
                row["exclude_list_md5"] = (p.get("gene_cleanup") or {}).get("exclude_list_md5")
                patch = (p.get("in_memory_patches") or {}).get("dwls", {})
                row["dwls_patch_applied"] = "; ".join(f"{k}: {v.get('patched')}" for k, v in (patch.get("applied") or {}).items())
                row["dwls_guard_fired_for"] = "; ".join(deescape(x) for x in patch.get("guard_fired_for", []))
                row["cibersortx_command_patched"] = ((p.get("in_memory_patches") or {})
                                                     .get("cibersortx_container_command", {}).get("applied"))
                row["result"] = "; ".join(f"{m}: {v.get('status')}" + (f" ({str(v.get('error'))[:80]})" if v.get("error") else "")
                                          for m, v in (p.get("method_result") or {}).items())
                row["versions"] = "; ".join(f"{k} {v}" for k, v in (p.get("packages") or {}).items() if v)
                row["runner_md5"] = "; ".join(f"{k} {v}" for k, v in (p.get("runner_md5") or {}).items())
            else:                                                # the Stage 8 runner
                row["runner"] = "run_deconvolution.R"
                row["methods"] = "bayesprism"
                row["settings"] = f"{p.get('n_cores')} cores; outlier cut {p.get('outlier_cut')}, fraction {p.get('outlier_fraction')}"
                row["exclude_list_md5"] = p.get("exclude_list_md5")
                row["versions"] = f"BayesPrism {p.get('bayesprism')}"
                row["cell_cap"] = "none"
            rows.append(row)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--panel", default=str(ROOT / "data/deconvolution/mixture_panel_refs5"))
    ap.add_argument("--previous", default=str(ROOT / "data/deconvolution/mixture_panel_refs4all"))
    ap.add_argument("--out", default=str(ROOT / "docs/audit/remediation/record/step4/v6_mixture_panel"))
    a = ap.parse_args()
    panel, previous, out = Path(a.panel), Path(a.previous), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    sets = panel_sets(panel)
    if not sets:
        sys.exit(f"no sets under {panel}")

    # the panel's own summary, unchanged (registered readings of MIXTURE_PANEL_REFS4ALL_PREREG.md)
    # (it needs BayesPrism scored on at least one set; on an unfinished panel it may stop, which is reported and
    # does not stop the V6 tables below)
    argv = sys.argv
    sys.argv = ["mixture_panel_summary.py", "--panel", str(panel), "--out", str(out)]
    panel_summary_error = None
    try:
        mps.main()
    except Exception as e:
        panel_summary_error = f"{type(e).__name__}: {e}"
        print(f"WARNING: mixture_panel_summary.py did not finish on this panel ({panel_summary_error})")
    finally:
        sys.argv = argv

    # seven decimals, so a number rounded again for the text or a figure label is the number itself rounded once
    w = lambda df, name: df.to_csv(out / name, sep="\t", index=False, float_format="%.7f")
    def r7(df):
        """Round the float columns to the seven decimals the tables are written with, so the summary below, the
        figures and the results text all format the same stored number."""
        f = df.select_dtypes("float").columns
        df[f] = df[f].round(7)
        return df

    floor = r7(floor_table(panel, sets)); w(floor, "v6_floor.tsv")
    status = status_table(panel, sets); w(status, "v6_status.tsv")
    prev = vs_previous(panel, previous, sets); prev = r7(prev) if len(prev) else prev; w(prev, "v6_vs_20261108.tsv")
    att = attention_types(panel, previous); att = r7(att) if len(att) else att; w(att, "v6_heart_hypoth_types.tsv")
    dw = r7(dwls_table(panel, previous, sets)); w(dw, "v6_dwls_setting.tsv")
    runs = run_settings(panel, sets); w(runs, "v6_run_settings.tsv")
    bc = panel / "build_check.tsv"
    if bc.exists():
        b = pd.read_csv(bc, sep="\t"); w(b, "v6_build_check.tsv")
    else:
        b = pd.DataFrame()

    L = [f"V6: known-truth mixture panel on the refs5 references. {len(sets)} sets under {panel}", ""]
    if panel_summary_error:
        L += [f"NOTE: the panel's own summary (panel_*.tsv, summary.txt) was not written: {panel_summary_error}", ""]
    sc = floor[floor["verdict"] != "not scored"]
    L.append(f"== Pass rule (registered, unchanged): BayesPrism's median per-type r >= {FLOOR} ==")
    L.append(f"scored {len(sc)} of {len(floor)} sets; at or above the floor: {int((sc['verdict'] == 'recovers').sum())}; "
             f"weak: {int((sc['verdict'] == 'weak').sum())}; not scored: {int((floor['verdict'] == 'not scored').sum())}")
    for r in floor.itertuples():
        if r.verdict == "not scored":
            L.append(f"  {r.set:16s} not scored")
            continue
        L.append(f"  {r.set:16s} median {r.bayesprism_median_r:.3f}  {r.verdict:8s} pooled {r.bayesprism_pooled_r:.3f}  "
                 f"types {int(r.n_types_in_median)}/{int(r.n_types_scored)}  below {FLOOR}: {int(r.n_types_below_floor)}"
                 + (f"  worst: {r.worst_types}" if r.verdict == "weak" or r.n_types_below_floor else ""))
    for kind in ("holdout", "cross"):
        x = sc[sc["design"] == kind]["bayesprism_median_r"]
        if len(x):
            L.append(f"  {kind}: {x.min():.3f} to {x.max():.3f} over {len(x)} sets")
    if len(sc):
        L.append(f"  per-type r of BayesPrism over all sets: {int(sc['n_types_scored'].sum())} type tests; "
                 f"{int(sc['n_types_r_ge_095'].sum())} at 0.95 or above, {int(sc['n_types_r_ge_090'].sum())} at 0.90 or above, "
                 f"{int(sc['n_types_below_floor'].sum())} below {FLOOR}")

    bpv = prev[prev["method"] == "bp_rna"].dropna(subset=["median_change"]) if len(prev) else prev
    L += ["", "== Against the panel of 2026-11-08 (BayesPrism median per-type r; descriptive, as registered) =="]
    if len(bpv):
        drops = bpv[bpv["median_change"] < -DROP]
        L.append(f"compared {len(bpv)} sets; dropped by more than {DROP}: {len(drops)}"
                 + ("" if drops.empty else " (" + ", ".join(f"{r.set} {r.median_change:+.3f}" for r in drops.itertuples()) + ")"))
        L.append(f"largest drop {bpv['median_change'].min():+.4f} ({bpv.loc[bpv['median_change'].idxmin(), 'set']}); "
                 f"largest rise {bpv['median_change'].max():+.4f} ({bpv.loc[bpv['median_change'].idxmax(), 'set']})")
        for r in bpv.itertuples():
            L.append(f"  {r.set:16s} {r.median_r_20261108:.3f} -> {r.median_r_now:.3f} ({r.median_change:+.4f})")
    else:
        L.append("nothing to compare yet")

    L += ["", "== HEART_holdout and HYPOTH_holdout, every type (labels changed) =="]
    for s in ATTENTION:
        x = att[att["set"] == s] if len(att) else att
        if not len(x):
            L.append(f"  {s}: not scored")
            continue
        now = x.dropna(subset=["r_now"])
        L.append(f"  {s}: {len(now)} types scored; r {now['r_now'].min():.3f} to {now['r_now'].max():.3f}; "
                 f"below {FLOOR}: {int((now['r_now'] < FLOOR).sum())}; "
                 f"new since 2026-11-08: {', '.join(now[~now['in_20261108_panel']]['cell_type']) or 'none'}; "
                 f"benchmarked then, not now: {', '.join(x[x['r_now'].isna()]['cell_type']) or 'none'}")
        for r in now.sort_values("r_now").itertuples():
            L.append(f"    {r.cell_type[:38]:38s} r {r.r_now:.3f} (2026-11-08: "
                     f"{'n/a' if not np.isfinite(r.r_20261108) else format(r.r_20261108, '.3f')})  "
                     f"true {r.true_mean_now:.3f} est {r.est_mean_now:.3f}")

    L += ["", "== Status of every method run =="]
    for st, g in status.groupby("status"):
        L.append(f"  {st}: {len(g)}")
    for r in status[~status["status"].isin(["scored", "not run"])].itertuples():
        L.append(f"    {r.set:16s} {NAMES[r.method]:10s} {r.status}: {r.detail[:150]}")

    L += ["", "== DWLS as the benchmark plan specified (mast_optimized, p 0.05) beside the earlier run (mast, p 0.01) =="]
    d = dw.dropna(subset=["median_r_mast_optimized_p0.05"])
    if len(d):
        both = d.dropna(subset=["median_change"])
        L.append(f"scored {len(d)} of {len(dw)} sets; median per-type r {d['median_r_mast_optimized_p0.05'].min():.3f} to "
                 f"{d['median_r_mast_optimized_p0.05'].max():.3f}; change against 2026-11-08 "
                 f"{both['median_change'].min():+.3f} to {both['median_change'].max():+.3f} over {len(both)} sets")
        if "setting_recorded" in d:
            L.append("settings recorded: " + "; ".join(f"{k} ({v} sets)" for k, v in d["setting_recorded"].value_counts().items()))
        if "guard_fired_for" in d:
            g = d[d["guard_fired_for"].fillna("") != ""]
            L.append(f"the in-memory guard fired on {len(g)} sets" + ("" if g.empty else ": " + "; ".join(
                f"{r.set} ({r.guard_fired_for})" for r in g.itertuples())))
    else:
        L.append("not scored yet")

    L += ["", "== Provenance records (finding 5.6) =="]
    if len(runs):
        oc = runs[runs.get("runner", "") == "run_omnideconv.R"] if "runner" in runs else runs.iloc[0:0]
        L.append(f"{len(runs)} records; cross-check runner {len(oc)}, Stage 8 runner {len(runs) - len(oc)}; "
                 f"commits: {', '.join(sorted(set(runs['git_commit'].dropna())))}; "
                 f"records with modified tracked files: {int((runs['modified_tracked_files'].fillna('') != '').sum())}")
    if len(b):
        L += ["", "== Build check (before deconvolution) =="]
        cr = b[b["design"] == "cross"]
        L.append(f"cross harmonization identical to the reviewed maps: {int((cr['harmonization_map'] == 'identical').sum())} of {len(cr)}")
        L.append(f"mixtures byte-identical to the 2026-11-08 panel: {int((b['mixtures_vs_previous'] == 'identical').sum())} of {len(b)} sets; "
                 f"different: {', '.join(b[b['mixtures_vs_previous'] == 'different']['set']) or 'none'}")
    txt = "\n".join(L)
    (out / "v6_summary.txt").write_text(txt + "\n")
    print("\n" + txt)
    print(f"\nwrote {out}/")


if __name__ == "__main__":
    main()
