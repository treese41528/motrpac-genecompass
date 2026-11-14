#!/usr/bin/env python
"""make_step4_v5v6_figures.py -- figures for V5 and V6 of Step 4 (docs/audit/remediation/STEP4_V5_V6_DRAFT.md), from
the record TSVs that make_step4_v5_tables.py and make_step4_v6_tables.py write.

  MPLCONFIGDIR=tmp/mpl python analysis/audit_remediation/make_step4_v5v6_figures.py
         [--v5 docs/audit/remediation/record/step4/v5_simbu] [--v6 docs/audit/remediation/record/step4/v6_mixture_panel]
         [--fig docs/audit/remediation/figures]

Each figure is drawn only if its table exists and has rows, so the script can be rerun as the batteries finish.
  step4_v5_bayesprism_arms.png   BayesPrism's pooled RMSE in arms A, B, C, per reference      (v5_bayesprism.tsv)
  step4_v5_corrector_rule.png    mean RMSE difference per method on the common set, C - A and C - B  (v5_ranking.tsv)
  step4_v5_arm_check.png         spread of library size across cell types as each arm sums cells  (v5_arm_library_size_summary.tsv)
  step4_v6_median_r.png          median per-type r per set against the registered floor      (v6_floor.tsv, panel_median.tsv)
  step4_v6_heart_hypoth.png      per-type r in HEART and HYPOTH, 2026-11-08 and now          (v6_heart_hypoth_types.tsv)

Colours. The three arms are an ordered sequence (no bias, native bias, added bias), so they take one hue in three
steps, light to dark (#7fb0ea, #2a78d6, #174a8a: lightness monotone, adjacent steps apart, light end 2.2:1 on the
surface). Everything else uses figstyle's blue and orange (all-pairs colour-vision-deficiency separation 24.7) and its
neutral grey for context. Every value drawn is in the TSV named above.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as fs                      # noqa: E402
import matplotlib.pyplot as plt            # noqa: E402
from matplotlib.lines import Line2D        # noqa: E402

fs.use()
ARM_COL = {"A": "#7fb0ea", "B": fs.BLUE, "C": "#174a8a"}
ARM_LAB = {"A": "A: no mRNA bias", "B": "B: native bias only", "C": "C: added bias"}
NAMES = {"bp_rna": "BayesPrism", "cibersortx": "CIBERSORTx", "dwls": "DWLS", "music": "MuSiC", "scdc": "SCDC",
         "bisque": "Bisque"}
RING = dict(edgecolor=fs.SURFACE, linewidth=1.2)


def read(path):
    if not path.exists():
        return None
    t = pd.read_csv(path, sep="\t")
    return t if len(t) else None


def dot_legend(ax, items, **kw):
    """items: (label, colour, marker size)."""
    h = [Line2D([], [], marker="o", linestyle="", color=c, markeredgecolor=fs.SURFACE, markersize=ms, label=l)
         for l, c, ms in items]
    return ax.legend(handles=h, handletextpad=0.2, borderaxespad=0, **kw)


# ------------------------------------------------------------------------------------------------ V5
def fig_v5_arms(rec, fig):
    t = read(rec / "v5_bayesprism.tsv")
    if t is None:
        return
    t = t.dropna(subset=["rmse_A", "rmse_B", "rmse_C"], how="all").sort_values("rmse_C", na_position="first")
    if not len(t):
        return
    f, ax = plt.subplots(figsize=(7.4, 0.36 * len(t) + 1.7))
    y = np.arange(len(t))
    lo = t[["rmse_A", "rmse_B", "rmse_C"]].min(axis=1); hi = t[["rmse_A", "rmse_B", "rmse_C"]].max(axis=1)
    ax.hlines(y, lo, hi, color=fs.GRID, linewidth=2.0, zorder=1)
    for a in "ABC":
        ax.scatter(t[f"rmse_{a}"], y, s=62, color=ARM_COL[a], zorder=3, **RING)
    ax.set_yticks(y); ax.set_yticklabels(t["reference"])
    ax.set_xlim(0, float(np.nanmax(hi)) * 1.06); ax.set_ylim(-0.7, len(t) - 0.3)
    ax.set_xlabel("pooled RMSE of BayesPrism's fractions against cell fractions (50 mixtures)")
    ax.grid(axis="y", visible=False)
    ax.set_title("BayesPrism's error in the three arms, per reference", pad=30)
    dot_legend(ax, [(ARM_LAB[a], ARM_COL[a], 8) for a in "ABC"], loc="lower left", bbox_to_anchor=(0, 1.01), ncol=3,
               columnspacing=1.4)
    fs.save(f, fig / "step4_v5_bayesprism_arms.png")


def fig_v5_rule(rec, fig):
    t = read(rec / "v5_ranking.tsv"); rule = read(rec / "v5_corrector_rule.tsv")
    if t is None:
        return
    t = t[(t["common_set"] == "all three arms") & t["contrast"].isin(["C-A", "C-B"])].dropna(subset=["mean_d_rmse_common"])
    if not len(t):
        return
    role = {"music": fs.BLUE, "scdc": fs.BLUE, "bp_rna": fs.ORANGE, "cibersortx": fs.ORANGE,
            "dwls": fs.NEUTRAL, "bisque": fs.NEUTRAL}
    heads = {"C-A": "a. Added bias against no bias (C minus A): the registered reading",
             "C-B": "b. Added bias against native bias (C minus B): the 2026-11-11 reading"}
    xmax = float(t["mean_d_rmse_common"].max()) * 1.22
    xmin = min(0.0, float(t["mean_d_rmse_common"].min()) * 1.3)
    f, axes = plt.subplots(2, 1, figsize=(7.6, 5.6), sharex=True, gridspec_kw={"hspace": 0.5})
    for ax, name in zip(axes, ["C-A", "C-B"]):
        d = t[t["contrast"] == name].sort_values("mean_d_rmse_common", ascending=True)      # largest on top
        y = np.arange(len(d))
        ax.barh(y, d["mean_d_rmse_common"], height=0.46, color=[role[m] for m in d["method"]], zorder=2)
        for yi, v in zip(y, d["mean_d_rmse_common"]):
            ax.text(v + xmax * 0.012 if v >= 0 else 0 + xmax * 0.012, yi, f"{v:+.4f}", va="center", ha="left",
                    fontsize=8, color=fs.INK)
        ax.set_yticks(y); ax.set_yticklabels([NAMES[m] for m in d["method"]])
        ax.set_xlim(xmin, xmax); ax.set_ylim(-0.6, len(d) - 0.4); ax.grid(axis="y", visible=False)
        ax.axvline(0, color=fs.INK2, linewidth=0.8, zorder=1)
        verdict = ""
        if rule is not None:
            r = rule[(rule["contrast"] == name) & (rule["common_set"] == "all three arms")]
            if len(r):
                verdict = (f"Rule (both blue bars shorter than both orange bars): {r.iloc[0]['verdict']}, "
                           f"{int(r.iloc[0]['n_common'])} references")
        ax.set_title(heads[name] + "\n", fontsize=9.5)
        ax.text(0, 1.035, verdict.strip(), transform=ax.transAxes, fontsize=8.5, color=fs.INK2, va="bottom")
    axes[1].set_xlabel("mean difference in pooled RMSE over the references every method ran\n(higher = more affected by the bias)")
    axes[0].tick_params(labelbottom=True)
    h = [Line2D([], [], marker="s", linestyle="", color=c, markersize=8, label=l) for l, c in
         [("MuSiC and SCDC", fs.BLUE), ("BayesPrism and CIBERSORTx", fs.ORANGE), ("not part of the rule", fs.NEUTRAL)]]
    f.legend(handles=h, loc="lower left", bbox_to_anchor=(0.125, -0.035), ncol=3, handletextpad=0.3, columnspacing=1.4)
    fs.save(f, fig / "step4_v5_corrector_rule.png")


def fig_v5_check(rec, fig):
    t = read(rec / "v5_arm_library_size_summary.tsv")
    if t is None:
        return
    t = t[t["arm"].isin(list("ABC"))]
    if not len(t):
        return
    order = t[t["arm"] == "B"].sort_values("max_over_min")["reference"].tolist() or sorted(t["reference"].unique())
    f, ax = plt.subplots(figsize=(7.4, 0.36 * len(order) + 1.7))
    for a in "ABC":
        d = t[t["arm"] == a].set_index("reference").reindex(order)
        ax.scatter(d["max_over_min"], np.arange(len(order)), s=62, color=ARM_COL[a], zorder=3, **RING)
    ax.set_xscale("log")
    top = float(t["max_over_min"].max()) * 1.6
    ax.set_xlim(0.85, top)
    ticks = [v for v in (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000) if v <= top]
    ax.set_xticks(ticks); ax.set_xticklabels([f"{v:,}" for v in ticks]); ax.minorticks_off()
    ax.set_yticks(range(len(order))); ax.set_yticklabels(order); ax.set_ylim(-0.7, len(order) - 0.3)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("largest over smallest mean library size across a reference's cell types,\n"
                  "as SimBu sums the cells (1 = every type contributes equally per cell; log scale)")
    ax.set_title("What each arm sums: arm A removes the library-size differences", pad=30)
    dot_legend(ax, [(ARM_LAB[a], ARM_COL[a], 8) for a in "ABC"], loc="lower left", bbox_to_anchor=(0, 1.01), ncol=3,
               columnspacing=1.4)
    fs.save(f, fig / "step4_v5_arm_check.png")


# ------------------------------------------------------------------------------------------------ V6
def fig_v6_floor(rec, fig):
    fl = read(rec / "v6_floor.tsv"); med = read(rec / "panel_median.tsv")
    if fl is None:
        return
    fl = fl.dropna(subset=["bayesprism_median_r"])
    if not len(fl):
        return
    order = fl.sort_values(["design", "bayesprism_median_r"], ascending=[True, True])["set"].tolist()   # cross first = bottom
    pos = {s: i for i, s in enumerate(order)}
    f, ax = plt.subplots(figsize=(7.6, 0.33 * len(order) + 1.9))
    others = []
    if med is not None:
        med = med.rename(columns={med.columns[0]: "set"}).set_index("set")
        for m in [c for c in med.columns if c != "bp_rna"]:
            for s, v in med[m].dropna().items():
                if s in pos:
                    others.append((v, pos[s]))
    xmin = 0.55
    if others:
        ox, oy = zip(*others)
        ax.scatter(np.clip(ox, xmin, None), oy, s=20, color=fs.NEUTRAL, alpha=0.75, zorder=2, linewidth=0)
        n_clip = int((np.array(ox) < xmin).sum())
        if n_clip:
            ax.text(xmin, -0.62, f"{n_clip} value{'s' if n_clip > 1 else ''} below {xmin} drawn at the edge", fontsize=7,
                    color=fs.INK2, va="bottom", ha="left")
    ax.scatter(fl.set_index("set").loc[order, "bayesprism_median_r"].clip(lower=xmin), range(len(order)), s=70,
               color=fs.BLUE, zorder=4, **RING)
    ax.axvline(0.85, color=fs.ORANGE, linewidth=1.4, zorder=1)
    ax.text(0.85, len(order) - 0.25, "registered floor 0.85 ", ha="right", va="bottom", fontsize=8, color=fs.INK2)
    n_cross = int((fl["design"] == "cross").sum())
    if 0 < n_cross < len(order):
        ax.axhline(n_cross - 0.5, color=fs.GRID, linewidth=0.8)
    ax.set_yticks(range(len(order))); ax.set_yticklabels([s.replace("_", " ") for s in order])
    ax.set_xlim(xmin - 0.01, 1.006); ax.set_ylim(-0.7, len(order) + 0.35); ax.grid(axis="y", visible=False)
    ax.set_xlabel("median over cell types of the Pearson r between true and estimated fraction (50 mixtures)")
    ax.set_title("Mixture panel on the refs5 references: BayesPrism against the floor", pad=30)
    items = [("BayesPrism", fs.BLUE, 8.5)] + ([("the other methods, where they ran", fs.NEUTRAL, 5.5)] if others else [])
    dot_legend(ax, items, loc="lower left", bbox_to_anchor=(0, 1.01), ncol=2, columnspacing=1.4)
    fs.save(f, fig / "step4_v6_median_r.png")


def fig_v6_types(rec, fig):
    t = read(rec / "v6_heart_hypoth_types.tsv")
    if t is None:
        return
    sets = [s for s in ["HEART_holdout", "HYPOTH_holdout"] if (t["set"] == s).any()]
    if not sets:
        return
    n = [int((t["set"] == s).sum()) for s in sets]
    vals = pd.concat([t["r_now"], t["r_20261108"]]).dropna()
    xmin = min(0.9, float(np.floor(vals.min() * 20) / 20)) - 0.008        # room for a dot at the lowest value
    f, axes = plt.subplots(1, len(sets), figsize=(5.6 * len(sets), 0.33 * max(n) + 1.9), squeeze=False,
                           gridspec_kw={"wspace": 0.75})
    for ax, s in zip(axes[0], sets):
        d = t[t["set"] == s].copy()
        d["key"] = d["r_now"].fillna(d["r_20261108"])
        d = d.sort_values("key").reset_index(drop=True)
        y = np.arange(len(d))
        both = d.dropna(subset=["r_now", "r_20261108"])
        ax.hlines(both.index, both[["r_now", "r_20261108"]].min(axis=1), both[["r_now", "r_20261108"]].max(axis=1),
                  color=fs.GRID, linewidth=2.0, zorder=1)
        ax.scatter(d["r_20261108"], y, s=52, color="#7fb0ea", zorder=2, **RING)
        ax.scatter(d["r_now"], y, s=58, color="#174a8a", zorder=3, **RING)
        lab = []
        for r in d.itertuples():
            tag = "" if (np.isfinite(r.r_now) and np.isfinite(r.r_20261108)) else \
                ("  (new label)" if np.isfinite(r.r_now) else "  (2026-11-08 only)")
            lab.append(r.cell_type + tag)
        ax.set_yticks(y); ax.set_yticklabels(lab); ax.set_ylim(-0.7, len(d) - 0.3)
        ax.set_xlim(xmin, 1.004); ax.grid(axis="y", visible=False)
        if xmin < 0.85:
            ax.axvline(0.85, color=fs.ORANGE, linewidth=1.2, zorder=1)
        ax.set_xlabel("Pearson r, true against estimated fraction")
        ax.set_title(s.replace("_holdout", " holdout: every cell type"), pad=28)
    dot_legend(axes[0][0], [("panel of 2026-11-08", "#7fb0ea", 7.5), ("refs5", "#174a8a", 8)], loc="lower left",
               bbox_to_anchor=(0, 1.01), ncol=2, columnspacing=1.4)
    fs.save(f, fig / "step4_v6_heart_hypoth.png")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--v5", default="docs/audit/remediation/record/step4/v5_simbu")
    ap.add_argument("--v6", default="docs/audit/remediation/record/step4/v6_mixture_panel")
    ap.add_argument("--fig", default="docs/audit/remediation/figures")
    a = ap.parse_args()
    v5, v6, fig = Path(a.v5), Path(a.v6), Path(a.fig)
    fig.mkdir(parents=True, exist_ok=True)
    fig_v5_arms(v5, fig); fig_v5_rule(v5, fig); fig_v5_check(v5, fig)
    fig_v6_floor(v6, fig); fig_v6_types(v6, fig)


if __name__ == "__main__":
    main()
