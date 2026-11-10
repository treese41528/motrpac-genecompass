#!/usr/bin/env python3
"""simbu_summary.py -- readouts of the SimBu mRNA-bias battery and dose-response on the current references
(SIMBU_BIAS_REFS4ALL_PREREG.md). Pure post-hoc over the score files and the simulated (downsampled) references.

  python deconvolution/omnideconv_bench/simbu_summary.py [--root data/deconvolution/simbu_bench_refs4all]
         [--out <root>/summary] [--july-root data/deconvolution/simbu_bench]

Writes simbu_status.tsv, simbu_delta.tsv, simbu_ranking.tsv, simbu_driver.tsv, simbu_mrna_content.tsv,
simbu_doseresp.tsv, simbu_doseresp_slope.tsv, simbu_vs_july.tsv and summary.txt. No truncation: every
reference, method and dose point found on disk is reported, and every missing one is listed.
"""
import argparse, hashlib, os, sys
import numpy as np
import pandas as pd

METHODS = ["bp_rna", "cibersortx", "dwls", "music", "scdc", "bisque"]
CORRECTORS = ["dwls", "music", "scdc"]
# July battery set -> current set (descriptive comparison only; most are different or relabelled references)
JULY_MAP = [("LIV_holdout", "LIVER"), ("LIV_novis", "LIVER"), ("PBMC_holdout", "BLOOD"), ("WAT_holdout", "WATSC"),
            ("HRT_holdout", "HEART"), ("GAS_cross", "MUSCLE"), ("SKMVL", "MUSCLE"), ("KID_cross", "KIDNEY"),
            ("HIP_merged", "HIPPOC"), ("LNG_cross", "LUNG"), ("LNG_native", "LUNG"), ("CTX_union", "CORTEX")]
JULY_PANEL = ["LIV_holdout", "PBMC_holdout", "WAT_holdout", "HRT_holdout", "GAS_cross", "KID_cross",
              "HIP_merged", "LNG_cross", "SKMVL", "CTX_union"]   # the 10 tissues July's ranking averaged
JULY_DOSE = {"LIVER": "LIV_holdout", "BLOOD": "PBMC_holdout"}
SKIP = {"doseresp", "summary", "logs"}


def overall(stage, m, key):
    p = os.path.join(stage, "scores", f"overall_metrics_{m}.tsv")
    if not os.path.exists(p):
        return np.nan
    d = pd.read_csv(p, sep="\t")
    v = d.loc[d["metric"] == key, "value"]
    return float(v.iloc[0]) if len(v) else np.nan


def pertype(stage, m):
    p = os.path.join(stage, "scores", f"metrics_{m}.tsv")
    if not os.path.exists(p):
        return None
    return pd.read_csv(p, sep="\t").set_index("cell_type")


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def sim_meta(stage):
    p = os.path.join(stage, "mixtures", "sim_meta.txt")
    if not os.path.exists(p):
        return {}
    return dict(line.rstrip("\n").split("=", 1) for line in open(p) if "=" in line)


def expressed_genes(refdir):
    """Per cell type: n_cells and mean number of expressed genes (count > 0) per cell, from the downsampled
    reference_counts.mtx (cells x genes, as simulate_simbu.R writes it) and cells_meta.tsv (same row order)."""
    mtx = os.path.join(refdir, "reference_counts.mtx")
    meta = pd.read_csv(os.path.join(refdir, "cells_meta.tsv"), sep="\t", usecols=["cell_type"], dtype=str)
    with open(mtx) as f:
        skip = 0
        for line in f:
            skip += 1
            if not line.startswith("%"):
                nrow, ncol, _ = (int(x) for x in line.split())
                break
    if nrow != len(meta):
        sys.exit(f"ERROR: {mtx}: {nrow} rows but {len(meta)} cells in cells_meta.tsv")
    trip = pd.read_csv(mtx, sep=r"\s+", header=None, skiprows=skip, names=["i", "j", "x"],
                       dtype={"i": np.int64, "j": np.int64, "x": np.float64}, engine="c")
    nz = trip[trip["x"] != 0]
    per_cell = np.bincount(nz["i"].to_numpy() - 1, minlength=nrow)
    counts = np.bincount(nz["i"].to_numpy() - 1, weights=nz["x"].to_numpy(), minlength=nrow)
    df = pd.DataFrame({"cell_type": meta["cell_type"].to_numpy(), "expressed_genes": per_cell, "counts": counts})
    g = df.groupby("cell_type").agg(n_cells=("expressed_genes", "size"),
                                    mean_expressed_genes=("expressed_genes", "mean"),
                                    mean_total_counts=("counts", "mean")).reset_index()
    # the registered rank is by expressed genes (SimBu's expressed_genes scaling); counts rank is descriptive
    g["mrna_rank"] = g["mean_expressed_genes"].rank(ascending=False, method="min").astype(int)
    g["counts_rank"] = g["mean_total_counts"].rank(ascending=False, method="min").astype(int)
    return g.sort_values("mrna_rank")


def battery_sets(root):
    if not os.path.isdir(root):
        return []
    return sorted(d for d in os.listdir(root) if d not in SKIP and os.path.isdir(os.path.join(root, d))
                  and any(os.path.isdir(os.path.join(root, d, c)) for c in ("bias", "nobias")))


def delta_table(root, sets):
    rows = []
    for s in sets:
        for m in METHODS:
            b, nb = os.path.join(root, s, "bias"), os.path.join(root, s, "nobias")
            rb, rn = overall(b, m, "overall_pooled_rmse"), overall(nb, m, "overall_pooled_rmse")
            pb, pn = overall(b, m, "overall_pooled_pearson"), overall(nb, m, "overall_pooled_pearson")
            status = "ok" if np.isfinite(rb) and np.isfinite(rn) else \
                ("not scored" if not (np.isfinite(rb) or np.isfinite(rn)) else
                 ("bias not scored" if not np.isfinite(rb) else "nobias not scored"))
            rows.append({"set": s, "method": m, "rmse_nobias": rn, "rmse_bias": rb, "d_rmse": rb - rn,
                         "r_nobias": pn, "r_bias": pb, "d_r": pb - pn, "status": status})
    return pd.DataFrame(rows)


def ranking(delta):
    ok = delta[delta["status"] == "ok"]
    common = [s for s, g in ok.groupby("set") if set(g["method"]) == set(METHODS)]
    rows = []
    for m in METHODS:
        d = ok[ok["method"] == m]
        dc = d[d["set"].isin(common)]
        rows.append({"method": m, "mean_d_rmse_common": dc["d_rmse"].mean(), "n_common": len(dc),
                     "mean_d_rmse_all": d["d_rmse"].mean(), "median_d_rmse_all": d["d_rmse"].median(),
                     "n_all": len(d), "n_positive": int((d["d_rmse"] > 0).sum()),
                     "sets_missing": "; ".join(sorted(set(delta["set"]) - set(d["set"])))})
    return pd.DataFrame(rows), common


def driver_table(root, sets, content):
    rows = []
    for s in sets:
        c = content.get(s)
        for m in METHODS:
            b, nb = pertype(os.path.join(root, s, "bias"), m), pertype(os.path.join(root, s, "nobias"), m)
            if b is None or nb is None:
                continue
            common = [t for t in b.index if t in nb.index]
            if not common:
                continue
            dd = (b.loc[common, "rmse"] - nb.loc[common, "rmse"]).astype(float)
            drv = dd.idxmax()
            rest = dd.drop(drv)
            row = {"set": s, "method": m, "driver": drv, "d_rmse_driver": dd[drv],
                   "rest_median_d_rmse": rest.median() if len(rest) else np.nan,
                   "true_mean": nb.loc[drv, "true_mean"], "est_nobias": nb.loc[drv, "est_mean"],
                   "est_bias": b.loc[drv, "est_mean"],
                   "direction": "inflated" if b.loc[drv, "est_mean"] > nb.loc[drv, "est_mean"] else "deflated",
                   "n_types_scored": len(common)}
            if c is not None:
                r = c.set_index("cell_type")
                row["driver_mrna_rank"] = int(r.loc[drv, "mrna_rank"]) if drv in r.index else np.nan
                row["driver_counts_rank"] = int(r.loc[drv, "counts_rank"]) if drv in r.index else np.nan
                row["n_types_reference"] = len(r)
                row["top_mrna_type"] = c.iloc[0]["cell_type"]
            rows.append(row)
    return pd.DataFrame(rows)


def dose_tables(root):
    base = os.path.join(root, "doseresp")
    rows, checks = [], []
    if not os.path.isdir(base):
        return pd.DataFrame(), pd.DataFrame(), checks
    for t in sorted(os.listdir(base)):
        tdir = os.path.join(base, t)
        alphas = []
        for name in os.listdir(tdir):
            try:
                alphas.append((float(name[1:]), name))
            except ValueError:
                pass
        alphas.sort()
        ref_md5, truth = set(), set()
        for a, name in alphas:
            st = os.path.join(tdir, name)
            rp = os.path.join(st, "reference", "reference_counts.mtx")
            tp = os.path.join(st, "mixtures", "true_fractions.tsv")
            if os.path.exists(rp):
                ref_md5.add(md5(rp))
            if os.path.exists(tp):
                truth.add(md5(tp))
            for m in METHODS:
                rows.append({"tissue": t, "alpha": a, "method": m,
                             "rmse": overall(st, m, "overall_pooled_rmse"),
                             "pearson_r": overall(st, m, "overall_pooled_pearson")})
        checks.append(f"dose {t}: {len(alphas)} alphas; references identical: {len(ref_md5) == 1}; "
                      f"cell-fraction truth identical: {len(truth) == 1}")
    d = pd.DataFrame(rows)
    srows = []
    for (t, m), g in d.groupby(["tissue", "method"]):
        g = g.dropna(subset=["rmse"]).sort_values("alpha")
        if g.empty:
            srows.append({"tissue": t, "method": m, "n_alpha": 0})
            continue
        r = g["rmse"].to_numpy(); a = g["alpha"].to_numpy()
        srows.append({"tissue": t, "method": m, "n_alpha": len(g), "rmse_alpha0": r[0] if a[0] == 0 else np.nan,
                      "rmse_alpha_max": r[-1], "alpha_max": a[-1],
                      "slope": (r[-1] - r[0]) / (a[-1] - a[0]) if len(g) > 1 and a[-1] != a[0] else np.nan,
                      "non_decreasing": bool(np.all(np.diff(r) >= 0)) if len(g) > 1 else np.nan})
    return d, pd.DataFrame(srows), checks


def july_tables(july_root, cur_delta):
    rows = []
    if not os.path.isdir(july_root):
        return pd.DataFrame(), None
    jd = delta_table(july_root, [s for s, _ in JULY_MAP if os.path.isdir(os.path.join(july_root, s))])
    for js, cs in JULY_MAP:
        for m in METHODS:
            j = jd[(jd["set"] == js) & (jd["method"] == m)]
            c = cur_delta[(cur_delta["set"] == cs) & (cur_delta["method"] == m)]
            rows.append({"july_set": js, "current_set": cs, "method": m,
                         "july_d_rmse": j["d_rmse"].iloc[0] if len(j) else np.nan,
                         "current_d_rmse": c["d_rmse"].iloc[0] if len(c) else np.nan})
    jr = jd[(jd["status"] == "ok") & jd["set"].isin(JULY_PANEL)].groupby("method")["d_rmse"].agg(["mean", "size"])
    return pd.DataFrame(rows), jr


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default="data/deconvolution/simbu_bench_refs4all")
    ap.add_argument("--out", default=None)
    ap.add_argument("--july-root", default="data/deconvolution/simbu_bench")
    a = ap.parse_args()
    out = a.out or os.path.join(a.root, "summary")
    os.makedirs(out, exist_ok=True)
    sets = battery_sets(a.root)
    if not sets:
        sys.exit(f"no battery sets under {a.root}")
    L = []

    # status + pairing sanity check + mRNA content (from the nobias reference = the bias reference)
    srows, content = [], {}
    for s in sets:
        refs = {c: os.path.join(a.root, s, c, "reference", "reference_counts.mtx") for c in ("bias", "nobias")}
        paired = all(os.path.exists(p) for p in refs.values()) and md5(refs["bias"]) == md5(refs["nobias"])
        tf = [os.path.join(a.root, s, c, "mixtures", "true_fractions.tsv") for c in ("bias", "nobias")]
        same_truth = all(os.path.exists(p) for p in tf) and md5(tf[0]) == md5(tf[1])
        meta = sim_meta(os.path.join(a.root, s, "nobias"))
        row = {"set": s, "paired_reference": paired, "same_truth": same_truth,
               "n_ref_cells": meta.get("n_ref_cells"), "n_types": meta.get("n_types"),
               "n_genes_bulk": meta.get("n_genes_bulk")}
        for c in ("bias", "nobias"):
            for m in METHODS:
                row[f"{c}_{m}"] = os.path.exists(os.path.join(a.root, s, c, "scores", f"overall_metrics_{m}.tsv"))
        srows.append(row)
        rd = os.path.join(a.root, s, "nobias", "reference")
        if os.path.exists(os.path.join(rd, "reference_counts.mtx")):
            content[s] = expressed_genes(rd)
    status = pd.DataFrame(srows)
    status.to_csv(os.path.join(out, "simbu_status.tsv"), sep="\t", index=False)
    pd.concat([c.assign(set=s) for s, c in content.items()])[
        ["set", "cell_type", "n_cells", "mean_expressed_genes", "mrna_rank", "mean_total_counts", "counts_rank"]].to_csv(
        os.path.join(out, "simbu_mrna_content.tsv"), sep="\t", index=False)

    delta = delta_table(a.root, sets)
    delta.to_csv(os.path.join(out, "simbu_delta.tsv"), sep="\t", index=False, float_format="%.5f")
    rank, common = ranking(delta)
    rank.to_csv(os.path.join(out, "simbu_ranking.tsv"), sep="\t", index=False, float_format="%.5f")
    drv = driver_table(a.root, sets, content)
    drv.to_csv(os.path.join(out, "simbu_driver.tsv"), sep="\t", index=False, float_format="%.5f")
    dose, slope, dchecks = dose_tables(a.root)
    dose.to_csv(os.path.join(out, "simbu_doseresp.tsv"), sep="\t", index=False, float_format="%.5f")
    slope.to_csv(os.path.join(out, "simbu_doseresp_slope.tsv"), sep="\t", index=False, float_format="%.5f")
    vj, july_rank = july_tables(a.july_root, delta)
    if len(vj):
        vj.to_csv(os.path.join(out, "simbu_vs_july.tsv"), sep="\t", index=False, float_format="%.5f")

    # ---------------- summary.txt ----------------
    L.append(f"SimBu mRNA-bias battery: {len(sets)} references under {a.root}")
    L.append(f"paired (bias and nobias references byte-identical): {int(status['paired_reference'].sum())}/{len(status)}; "
             f"same cell-fraction truth: {int(status['same_truth'].sum())}/{len(status)}")
    for c in dchecks:
        L.append(c)
    nots = delta[delta["status"] != "ok"]
    L.append(f"\nnot scored (set/method/status): {len(nots)}")
    for _, r in nots.iterrows():
        L.append(f"  {r['set']:8s} {r['method']:10s} {r['status']}")

    L.append("\n== Reading 1: corrector ranking, mean dRMSE (bias - nobias), lower = corrects ==")
    L.append(f"common set ({len(common)} references with all six methods scored): {', '.join(common)}")
    for _, r in rank.sort_values("mean_d_rmse_common").iterrows():
        L.append(f"  {r['method']:10s} common {r['mean_d_rmse_common']:+.4f} (n={r['n_common']})   "
                 f"all {r['mean_d_rmse_all']:+.4f} median {r['median_d_rmse_all']:+.4f} (n={r['n_all']}, "
                 f"{r['n_positive']} > 0)")
    mc = rank.set_index("method")["mean_d_rmse_common"]
    worst = min(mc["bp_rna"], mc["cibersortx"])
    rep = mc["music"] < worst and mc["scdc"] < worst
    L.append(f"paper's core finding (MuSiC and SCDC below BayesPrism and CIBERSORTx): "
             f"{'REPRODUCED' if rep else 'NOT REPRODUCED'} (music {mc['music']:+.4f}, scdc {mc['scdc']:+.4f}; "
             f"bp_rna {mc['bp_rna']:+.4f}, cibersortx {mc['cibersortx']:+.4f}; dwls {mc['dwls']:+.4f})")
    if july_rank is not None:
        L.append("July (10 panel tissues): " + ", ".join(
            f"{m} {july_rank.loc[m, 'mean']:+.4f} (n={int(july_rank.loc[m, 'size'])})"
            for m in METHODS if m in july_rank.index))

    bp = delta[(delta["method"] == "bp_rna") & (delta["status"] == "ok")]
    L.append(f"\n== Reading 2: BayesPrism dRMSE > 0 on {int((bp['d_rmse'] > 0).sum())}/{len(bp)} references ==")

    L.append("\n== Reading 3: bias driver (largest per-type dRMSE) vs mRNA content rank (1 = most expressed genes) ==")
    bpd = drv[drv["method"] == "bp_rna"] if len(drv) else drv
    if len(bpd):
        n1 = int((bpd["driver_mrna_rank"] == 1).sum())
        L.append(f"BayesPrism driver is the highest-mRNA type (expressed genes) on {n1}/{len(bpd)} references; "
                 f"highest by total counts on {int((bpd['driver_counts_rank'] == 1).sum())}/{len(bpd)}")
        for _, r in bpd.iterrows():
            rk = f"{int(r['driver_mrna_rank'])}" if np.isfinite(r.get("driver_mrna_rank", np.nan)) else "?"
            ck = f"{int(r['driver_counts_rank'])}" if np.isfinite(r.get("driver_counts_rank", np.nan)) else "?"
            L.append(f"  {r['set']:8s} {r['driver'][:34]:34s} rank {rk}/{int(r['n_types_reference'])} (counts rank {ck}) "
                     f"{r['direction']:8s} dRMSE {r['d_rmse_driver']:+.4f} (rest median {r['rest_median_d_rmse']:+.4f}); "
                     f"true {r['true_mean']:.3f} est {r['est_nobias']:.3f}->{r['est_bias']:.3f}; top-mRNA: {r['top_mrna_type']}")
        agree = drv.groupby("set")["driver"].agg(lambda x: x.value_counts().iloc[0])
        L.append("methods sharing the most common driver per set: " +
                 ", ".join(f"{s} {int(n)}/{int((drv['set'] == s).sum())}" for s, n in agree.items()))

    L.append("\n== Reading 4: dose-response (pooled RMSE vs alpha; composition pinned) ==")
    def mono(x):
        return "non-decreasing" if isinstance(x, (bool, np.bool_)) and bool(x) else "not monotone"
    for t, g in (slope.groupby("tissue") if len(slope) else []):
        g = g.set_index("method")
        L.append(f"{t}: " + "; ".join(
            f"{m} base {g.loc[m, 'rmse_alpha0']:.3f} slope {g.loc[m, 'slope']:+.4f} {mono(g.loc[m, 'non_decreasing'])}"
            for m in METHODS if m in g.index and g.loc[m, "n_alpha"] > 0))
        have = [m for m in METHODS if m in g.index and np.isfinite(g.loc[m].get("rmse_alpha0", np.nan))]
        if "dwls" in have:
            low = min(have, key=lambda m: g.loc[m, "rmse_alpha0"])
            corr = [m for m in CORRECTORS if m in have and np.isfinite(g.loc[m, "slope"])]
            steep = max(corr, key=lambda m: g.loc[m, "slope"]) if corr else None
            L.append(f"  DWLS lowest baseline of the six: {low == 'dwls'} (lowest: {low}); "
                     f"steepest slope of the correctors: {steep == 'dwls'} (steepest: {steep})")
        if t in JULY_DOSE:
            L.append(f"  July counterpart: {JULY_DOSE[t]} (earlier reference)")

    open(os.path.join(out, "summary.txt"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\nwrote {out}/")


if __name__ == "__main__":
    main()
