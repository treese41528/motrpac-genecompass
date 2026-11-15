#!/usr/bin/env python
"""make_step3_figures.py -- figures for docs/audit/remediation/STEP3_RESULTS.md, from the committed record TSVs.

Each figure is drawn only if its record table exists, so the script can be rerun as the rerun's stages finish.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as fs                      # noqa: E402
import matplotlib.pyplot as plt            # noqa: E402
from matplotlib.colors import LinearSegmentedColormap   # noqa: E402

REC = Path("docs/audit/remediation/record/step3")
FIG = Path("docs/audit/remediation/figures")
fs.use()
SEQ = LinearSegmentedColormap.from_list("seq_blue", [fs.SURFACE, "#cfe0f6", "#7fb0ea", fs.BLUE, "#174a8a"])


def fig_hypoth():
    f = REC / "hypoth_old_vs_new_labels.tsv"
    if not f.exists():
        return
    t = pd.read_csv(f, sep="\t", index_col=0)
    order_new = [c for c in ["Glutamatergic neurons", "Neurons", "Astrocytes", "Oligodendrocytes",
                             "Oligodendrocyte precursor cells", "Tanycytes", "Ependymal cells", "Microglia",
                             "Endothelial cells", "Vascular leptomeningeal cells", "Pars tuberalis cells", "DROP"] if c in t.columns]
    order_old = [r for r in ["Glutamatergic neurons", "GABAergic neurons", "Neurons", "Dopaminergic neurons", "Astrocytes",
                             "Oligodendrocytes", "Oligodendrocyte precursor cells", "Tanycytes", "Ependymal cells",
                             "Microglia", "Endothelial cells", "Radial glial cells", "Neuroendocrine cells"] if r in t.index]
    t = t.loc[order_old, order_new]
    share = t.div(t.sum(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=(9.6, 5.6))
    ax.imshow(share.to_numpy(), cmap=SEQ, vmin=0, vmax=1, aspect="auto")
    for i in range(t.shape[0]):
        for j in range(t.shape[1]):
            v = int(t.iat[i, j])
            if v:
                ax.text(j, i, f"{v:,}", ha="center", va="center", fontsize=7.5,
                        color=fs.SURFACE if share.iat[i, j] > 0.55 else fs.INK)
    ax.set_xticks(range(t.shape[1])); ax.set_xticklabels([c if c != "DROP" else "dropped\n(doublets)" for c in t.columns], rotation=40, ha="right")
    ax.set_yticks(range(t.shape[0])); ax.set_yticklabels([f"{r}  ({int(n):,})" for r, n in zip(t.index, t.sum(axis=1))])
    ax.set_xlabel("label after the joint re-annotation (refs5)"); ax.set_ylabel("label in the reference used until now (cells)")
    ax.grid(False); ax.spines[:].set_visible(False)
    ax.set_title("HYPOTH reference: where each old label's cells went")
    fs.save(fig, FIG / "step3_hypoth_labels.png")


def fig_stage8():
    """Figure B1: estimated fractions under refs5 against the current default, per tissue x cell type."""
    f = REC / "stage8" / "fractions_refs5_vs_default_by_type.tsv"
    if not f.exists():
        return
    t = pd.read_csv(f, sep="\t").dropna(subset=["pearson_r"])
    kind = t["tissue"].map(lambda x: "HYPOTH" if x == "HYPOTH" else "HEART" if x == "HEART" else "other")
    col = {"other": fs.BLUE, "HYPOTH": fs.ORANGE, "HEART": fs.AQUA}
    lab = {"other": "14 tissues: only the exclusion list changed", "HYPOTH": "HYPOTH: reference re-annotated",
           "HEART": "HEART: CD8+ T cells merged into T cells"}
    fig, (a, b) = plt.subplots(1, 2, figsize=(11.2, 5.4), gridspec_kw={"width_ratios": [1, 1.15], "wspace": 0.42})

    lo = 2e-5
    for k in ["other", "HEART", "HYPOTH"]:
        d = t[kind == k]
        a.scatter(d["mean_old"].clip(lower=lo), d["mean_new"].clip(lower=lo), s=26, color=col[k], edgecolor=fs.SURFACE,
                  linewidth=0.6, label=f"{lab[k]} ({len(d)})", zorder=3 if k != "other" else 2)
    a.plot([lo, 1], [lo, 1], color=fs.NEUTRAL, linewidth=0.8, zorder=1)
    a.set_xscale("log"); a.set_yscale("log"); a.set_xlim(lo, 1); a.set_ylim(lo, 1); a.set_aspect("equal")
    a.set_xlabel("mean estimated fraction, current default"); a.set_ylabel("mean estimated fraction, refs5")
    a.set_title("a. Mean fraction of each cell type")
    a.legend(loc="upper left", bbox_to_anchor=(0.0, -0.16), ncol=1, handletextpad=0.3, borderaxespad=0)
    for name in ["Neurons", "Oligodendrocyte precursor cells"]:
        r = t[(t["tissue"] == "HYPOTH") & (t["cell_type"] == name)]
        if len(r):
            r = r.iloc[0]
            a.annotate("HYPOTH " + ("OPC" if name.startswith("Oligo") else name.lower()), (r["mean_old"], r["mean_new"]),
                       xytext=(-7, 5), textcoords="offset points", ha="right", fontsize=7.5, color=fs.INK2)

    order = t.groupby("tissue")["pearson_r"].median().sort_values().index.tolist()
    xmin = 0.80
    for i, tis in enumerate(order):
        d = t[t["tissue"] == tis]
        k = "HYPOTH" if tis == "HYPOTH" else "HEART" if tis == "HEART" else "other"
        b.scatter(d["pearson_r"].clip(lower=xmin), np.full(len(d), i), s=24, color=col[k], edgecolor=fs.SURFACE,
                  linewidth=0.6, zorder=3, alpha=0.9)
        off = d[d["pearson_r"] < xmin]
        for _, r in off.iterrows():
            b.annotate(f"{'OPC' if r['cell_type'].startswith('Oligodendrocyte p') else r['cell_type']}: r = {r['pearson_r']:.2f}",
                       (xmin, i), xytext=(8, 7), textcoords="offset points", fontsize=7.5, color=fs.INK2)
    b.set_yticks(range(len(order))); b.set_yticklabels(order)
    b.set_xlim(xmin - 0.006, 1.004); b.set_ylim(-0.7, len(order) - 0.3)
    b.set_xlabel("Pearson r across a tissue's samples, default against refs5\n(one dot per cell type)")
    b.set_title("b. Do samples keep their order?")
    b.grid(axis="y", visible=False)
    fs.save(fig, FIG / "step3_stage8_fractions.png")


def short(ct):
    return (ct.replace("Vascular smooth muscle cells", "VSMC").replace("Megakaryocytes and platelets", "megakaryocytes")
            .replace("Oligodendrocyte precursor cells", "OPC"))


def fig_gate():
    """Figure B2: (a) each block's gate AUC under the default and under refs5; (b) the permutation intervals of the calls."""
    f = REC / "downstream" / "gate_blocks_refs5_vs_default.tsv"
    if not f.exists():
        return
    m = pd.read_csv(f, sep="\t")
    tf = lambda c: m[c].astype(str).str.upper().isin(["TRUE", "1"])
    both = m[(m["in"] == "both") & tf("analysable") & tf("analysable_default")].copy()
    hot_new = both.gate_class.isin(["hotspot", "borderline"]); hot_old = both.gate_class_default == "hotspot"
    grp = np.where(hot_new & hot_old, "both", np.where(hot_new, "refs5", np.where(hot_old, "default", "none")))
    col = {"none": fs.NEUTRAL, "both": fs.BLUE, "refs5": fs.ORANGE, "default": fs.AQUA}
    lab = {"none": "called in neither", "both": "hotspot in both", "refs5": "called in refs5 only", "default": "hotspot in the default only"}
    fig, (a, b) = plt.subplots(1, 2, figsize=(12.6, 5.9), gridspec_kw={"width_ratios": [1, 1.0], "wspace": 0.72})
    for k in ["none", "both", "refs5", "default"]:
        d = both[grp == k]
        a.scatter(d.sup_trained_auc_default, d.sup_trained_auc, s=18 if k == "none" else 42, color=col[k], alpha=0.55 if k == "none" else 1,
                  edgecolor=fs.SURFACE, linewidth=0.6, zorder=2 if k == "none" else 4, label=f"{lab[k]} ({len(d)})")
    a.plot([0.2, 0.95], [0.2, 0.95], color=fs.INK2, lw=0.8, zorder=1)
    lo, hi = 0.22, 0.93
    a.set_xlim(lo, hi); a.set_ylim(lo, hi); a.set_aspect("equal")
    # label positions in data coordinates, chosen by eye so that no label sits on a point or leaves the panel
    pos = {("SKMGN", "Schwann cells"): (0.625, 0.838, "right"), ("SKMVL", "Tenocytes"): (0.735, 0.872, "right"),
           ("HEART", "Vascular smooth muscle cells"): (0.60, 0.905, "right"), ("SKMGN", "Vascular smooth muscle cells"): (0.60, 0.79, "right"),
           ("LUNG", "Endothelial cells"): (0.925, 0.615, "right"), ("SKMGN", "Tenocytes"): (0.925, 0.66, "right")}
    for r in both[(grp == "refs5") | (grp == "default")].itertuples():
        x, y, ha = pos.get((r.tissue, r.cell_type), (r.sup_trained_auc_default + 0.01, r.sup_trained_auc, "left"))
        a.annotate(f"{r.tissue} {short(r.cell_type)}", (r.sup_trained_auc_default, r.sup_trained_auc), xytext=(x, y),
                   textcoords="data", ha=ha, va="center", fontsize=7, color=fs.INK2,
                   arrowprops={"arrowstyle": "-", "color": fs.INK2, "lw": 0.6, "shrinkA": 1, "shrinkB": 4})
    a.set_xlabel("trained-against-control AUC, current default"); a.set_ylabel("trained-against-control AUC, refs5")
    a.set_title("a. The same blocks, scored twice")
    a.legend(loc="upper left", bbox_to_anchor=(0.0, -0.13), ncol=2, handletextpad=0.2, columnspacing=1.0, borderaxespad=0)

    fam = m[tf("analysable") & m.p_hi_sup_trained.notna()].sort_values("p_sup_trained").head(20).iloc[::-1].reset_index(drop=True)
    n_fam = int(tf("analysable").sum()); k_hi = int((m.gate_class == "hotspot").sum()); k_lo = int(m.gate_class.isin(["hotspot", "borderline"]).sum())
    cc = {"hotspot": fs.BLUE, "borderline": fs.ORANGE, "none": fs.NEUTRAL}
    yy = np.arange(len(fam))
    for y_, r in zip(yy, fam.itertuples()):
        b.plot([max(r.p_lo_sup_trained, 1e-7), r.p_hi_sup_trained], [y_, y_], color=cc[r.gate_class], lw=2.2, solid_capstyle="butt")
        b.scatter([r.p_sup_trained], [y_], s=26, color=cc[r.gate_class], edgecolor=fs.SURFACE, linewidth=0.6, zorder=3)
    c_hi, c_lo = 0.05 * k_hi / n_fam, 0.05 * k_lo / n_fam
    for cut in (c_hi, c_lo):
        b.axvline(cut, color=fs.INK2, lw=0.9, ls=(0, (4, 3)))
    b.text(c_hi * 0.8, -1.75, f"Benjamini-Hochberg cuts: {c_hi:.4f} at the upper limits,\n{c_lo:.4f} at the lower limits",
           ha="right", va="center", fontsize=7.5, color=fs.INK2)
    b.set_xscale("log"); b.set_xlim(1e-7, 0.2); b.set_ylim(-2.6, len(fam) - 0.4)
    b.set_yticks(yy); b.set_yticklabels([f"{r.tissue} {short(r.cell_type)}  ({int(r.m_sup_trained):,})" for r in fam.itertuples()], fontsize=7.5)
    b.grid(axis="y", visible=False)
    b.set_xlabel("permutation p: estimate and interval (log scale)\npermutations used are in brackets")
    b.set_title("b. A hotspot's whole interval is below the cut")
    from matplotlib.lines import Line2D
    b.legend(handles=[Line2D([], [], color=cc[k], lw=2.2, marker="o", ms=5, label=k if k != "none" else "not a hotspot") for k in cc],
             loc="center left", bbox_to_anchor=(0.02, 0.42), handlelength=1.6)
    fs.save(fig, FIG / "step3_gate.png")


def fig_called_blocks():
    """Figure B3: every hotspot and borderline block against every registered reading (pass / fail / not testable)."""
    f = REC / "downstream" / "called_blocks_all_readings.tsv"
    if not f.exists():
        return
    t = pd.read_csv(f, sep="\t")
    cols = [("s1_significant", "sex-stratified\npermutation\n(S1)"), ("s2_significant", "8-week against\ncontrol\n(S2)"),
            ("support", "expression\nalone separates\n(S3)"), ("not_low_fraction", "at least 1%\nof the tissue\n(S4)"),
            ("v1_validated_at_abundance", "expression\nrecovered at its\nabundance (V1)"), ("transfers_corrected_rule", "transfers\n(corrected\nrule)"),
            ("dose_trend_kept", "dose trend\nkept in\nhuman space"), ("composition_pass", "fraction does\nnot trend\nwith dose"),
            ("de_covariate_robust", "DE calls robust\nto technical\ncovariates")]

    def state(r, c):
        v = r[c]
        if c == "support":
            return "pass" if v == "supported" else "n/a" if pd.isna(v) else "fail"
        if pd.isna(v):
            return "n/t"
        return "pass" if str(v).upper() in ("TRUE", "1") else "fail"
    S = [[state(r, c) for c, _ in cols] for _, r in t.iterrows()]
    fill = {"pass": fs.BLUE, "fail": fs.ORANGE, "n/t": "#c9c8c2", "n/a": "#c9c8c2"}
    mark = {"pass": "pass", "fail": "fail", "n/t": "not\ntestable", "n/a": "no\nreading"}
    fig, ax = plt.subplots(figsize=(11.6, 6.6))
    for i, row in enumerate(S):
        for j, st in enumerate(row):
            ax.add_patch(plt.Rectangle((j + 0.04, i + 0.06), 0.92, 0.88, color=fill[st], lw=0))
            ax.text(j + 0.5, i + 0.5, mark[st], ha="center", va="center", fontsize=7.5,
                    color=fs.SURFACE if st in ("pass", "fail") else fs.INK)
    n_pass = [sum(st == "pass" for st in row) for row in S]
    for i, n in enumerate(n_pass):
        ax.text(len(cols) + 0.15, i + 0.5, f"{n} of {len(cols)}", va="center", fontsize=8, color=fs.INK2)
    ax.text(len(cols) + 0.15, -0.35, "passed", va="center", fontsize=8, color=fs.INK2)
    ax.set_xlim(0, len(cols) + 1.0); ax.set_ylim(len(t), -0.05)
    ax.set_xticks([j + 0.5 for j in range(len(cols))]); ax.set_xticklabels([c[1] for c in cols], fontsize=7.5)
    ax.xaxis.tick_top(); ax.tick_params(axis="x", length=0); ax.tick_params(axis="y", length=0)
    ax.set_yticks([i + 0.5 for i in range(len(t))])
    ax.set_yticklabels([f"{r.tissue} {short(r.cell_type)}" + ("  (borderline)" if r.gate_class == "borderline" else "") for r in t.itertuples()], fontsize=8)
    ax.grid(False); ax.spines[:].set_visible(False)
    fig.suptitle("Each called block of the refs5 gate against every registered reading", x=0.02, ha="left", fontsize=10, fontweight="bold", y=1.0)
    fs.save(fig, FIG / "step3_called_blocks.png")


if __name__ == "__main__":
    fig_hypoth()
    fig_stage8()
    fig_gate()
    fig_called_blocks()
