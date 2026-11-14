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


if __name__ == "__main__":
    fig_hypoth()
    fig_stage8()
