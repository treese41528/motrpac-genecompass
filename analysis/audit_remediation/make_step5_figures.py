#!/usr/bin/env python
"""make_step5_figures.py -- figures for docs/audit/remediation/STEP5_RESULTS.md, from the committed record TSVs."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as fs                      # noqa: E402
import matplotlib.pyplot as plt            # noqa: E402

REC = Path("docs/audit/remediation/record/step5")
FIG = Path("docs/audit/remediation/figures")
fs.use()

# ------------------------------------------------- Figure 5.1: mitochondrial fraction, corpus and references
f = pd.read_csv(REC / "mito_by_file.tsv", sep="\t")
s = pd.read_csv(REC / "mito_by_study.tsv", sep="\t")
r = pd.read_csv(REC / "mito_by_reference.tsv", sep="\t")
o = pd.read_csv(REC / "mito_overall.tsv", sep="\t", index_col=0)["value"]
fig, (a, b, c) = plt.subplots(1, 3, figsize=(13.5, 4.8), gridspec_kw={"width_ratios": [0.9, 1.15, 1.05], "wspace": 0.75})

cuts = [5, 10, 15, 20, 50]
inert = f.stored_mito_pct_max == 0
grp = {"all 6.8 M cells": f, "files where the\nmitochondrial rule\nsaw no genes (395)": f[inert],
       "files where it\nsaw them (162)": f[f.stored_mito_pct_max > 0]}
x = np.arange(len(cuts))
for (lab, d), col, mk in zip(grp.items(), (fs.INK2, fs.ORANGE, fs.BLUE), ("o", "s", "D")):
    y = [d[f"n_above_{k}"].sum() / d.n_cells.sum() for k in cuts]
    a.plot(x, y, color=col, marker=mk, ms=6, lw=2, label=lab.replace("\n", " "))
    a.annotate(f"{y[2]:.1%}", (x[2], y[2]), xytext=(6, 5), textcoords="offset points", fontsize=7.5, color=fs.INK2)
a.axvline(2, color=fs.GRID, lw=6, zorder=0)
a.set_xticks(x); a.set_xticklabels([f">{k}%" for k in cuts]); a.set_ylim(0, None)
a.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
a.set_xlabel("mitochondrial share of a cell's counts"); a.set_ylabel("share of training cells")
a.set_title("(a) Training cells above each cut")
a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1, handletextpad=0.4)

top = s.sort_values("n_above_15", ascending=False).head(12).iloc[::-1]
yy = np.arange(len(top))
b.barh(yy, top.n_above_15 / 1e3, color=fs.BLUE, height=0.62)
for y_, n_, sh in zip(yy, top.n_above_15, top.share_above_15):
    b.text(n_ / 1e3 + 1.2, y_, f"{n_ / 1e3:.0f}k  ({sh:.0%} of the study)", va="center", fontsize=7.5, color=fs.INK2)
b.set_yticks(yy); b.set_yticklabels(top.study); b.set_xlim(0, top.n_above_15.max() / 1e3 * 1.75)
b.set_xlabel("cells above 15% (thousands)"); b.grid(axis="y", visible=False)
b.set_title(f"(b) Twelve studies hold {top.n_above_15.sum() / o['cells_above_15']:.0%} of them")

g = r.groupby("reference").apply(lambda d: pd.Series({
    "matched": d.n_matched_to_a_qc_cell.sum(),
    "share": (d.share_above_15 * d.n_matched_to_a_qc_cell).sum() / max(d.n_matched_to_a_qc_cell.sum(), 1)}), include_groups=False)
g = g[g.matched > 0].sort_values("share")
yy = np.arange(len(g))
c.barh(yy, g.share, color=fs.BLUE, height=0.62)
for y_, v in zip(yy, g.share):
    c.text(v + 0.01, y_, f"{v:.1%}", va="center", fontsize=7.5, color=fs.INK2)
c.axvline(0.05, color=fs.NEUTRAL, lw=1, ls=(0, (4, 3)))
c.text(0.055, len(g) - 0.3, "registered flag: 5%", fontsize=7.5, color=fs.INK2, va="bottom")
c.set_yticks(yy); c.set_yticklabels([n.split("_")[0] for n in g.index]); c.set_xlim(0, 0.62)
c.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
c.set_xlabel("share of the reference's cells above 15%"); c.grid(axis="y", visible=False)
c.set_title("(c) Deconvolution references built from corpus cells")
fs.save(fig, FIG / "step5_mito.png")

# ------------------------------------------------- Figure 5.2: counts filed under another gene, by study and in the bulk
k = pd.read_csv(REC / "id_symbol_classes_by_study.tsv", sep="\t").sort_values("other gene", ascending=False).reset_index(drop=True)
bulk = pd.read_csv(REC / "bulk_direct_ids_other_gene_or_alignment_lost.tsv", sep="\t")
ref_studies = {"GSE240658": "KIDNEY", "GSE220075": "LIVER", "GSE303115": "CORTEX", "GSE137869": "WATSC, muscle",
               "GSE244451": "BAT", "GSE272055": "SMLINT", "GSE295314": "HIPPOC", "GSE248413": "HYPOTH", "GSE280111": "HEART",
               "GSE273062": "LUNG", "GSE252844": "LUNG"}
fig, (a, b) = plt.subplots(1, 2, figsize=(12.6, 5.2), gridspec_kw={"width_ratios": [1.25, 1], "wspace": 0.5})
x = np.arange(len(k))
is_ref = k.study.isin(ref_studies)
a.bar(x[~is_ref], k.loc[~is_ref, "other gene"], color=fs.BLUE, width=0.72, label=f"other corpus studies ({int((~is_ref).sum())})")
a.bar(x[is_ref], k.loc[is_ref, "other gene"], color=fs.ORANGE, width=0.72, label=f"studies that feed a deconvolution reference ({int(is_ref.sum())})")
a.axhline(0.005, color=fs.NEUTRAL, lw=1, ls=(0, (4, 3)))
a.text(len(k) - 0.5, 0.0062, "registered bound: 0.5%", ha="right", fontsize=7.5, color=fs.INK2)
for i in range(3):
    a.annotate(f"{k.study[i]}  {k['other gene'][i]:.1%}", (x[i], k["other gene"][i]), xytext=(8, -2), textcoords="offset points",
               fontsize=7.5, color=fs.INK2, va="top" if i == 0 else "center")
a.set_xticks([]); a.set_xlim(-1, len(k)); a.grid(axis="x", visible=False)
a.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
a.set_xlabel(f"{len(k)} studies with ID-keyed files, sorted"); a.set_ylabel("share of a study's counts under another gene's token")
a.set_title(f"(a) Corpus: {int((k['other gene'] > 0.005).sum())} of {len(k)} studies are above the bound")
a.legend(loc="upper right", bbox_to_anchor=(1.0, 0.9))

top = bulk[bulk["class"] == "other gene"].sort_values("bulk_counts", ascending=False).head(10).iloc[::-1]
yy = np.arange(len(top))
b.barh(yy, top.share_of_bulk_counts, color=fs.BLUE, height=0.62)
for y_, v in zip(yy, top.share_of_bulk_counts):
    b.text(v + 0.0002, y_, f"{v:.2%}", va="center", fontsize=7.5, color=fs.INK2)
b.set_yticks(yy); b.set_yticklabels([f"{s_.capitalize()} → {n_}" for s_, n_ in zip(top.symbol, top.release115_symbol)])
b.set_xlim(0, top.share_of_bulk_counts.max() * 1.25); b.grid(axis="y", visible=False)
b.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1%}"))
b.set_xlabel("share of all MoTrPAC bulk counts")
b.set_title("(b) Bulk: consortium gene → gene the ID names in release 115")
fs.save(fig, FIG / "step5_id_drift.png")
