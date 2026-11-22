#!/usr/bin/env python
"""make_step1_figures.py -- figures for docs/audit/remediation/STEP1_INTEGRITY.md, from the committed record TSVs."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as fs                      # noqa: E402
import matplotlib.pyplot as plt            # noqa: E402

REC = Path("docs/audit/remediation/record/step1")
FIG = Path("docs/audit/remediation/figures")
fs.use()

# ---------------------------------------------------------------- Figure 1.1: the sample join, two data-anchored checks
s = pd.read_csv(REC / "sample_join_summary.tsv", sep="\t")
p = pd.read_csv(REC / "sample_join_per_sample.tsv", sep="\t")
order = list(s.sort_values("A_share_equal_shifted").tissue)
fig, (a, b) = plt.subplots(1, 2, figsize=(10.5, 4.6), gridspec_kw={"width_ratios": [1, 1.25], "wspace": 0.28})
y = np.arange(len(order)); ss = s.set_index("tissue").loc[order]
a.hlines(y, ss.A_share_equal_shifted, ss.A_share_entries_equal, color=fs.GRID, lw=2, zorder=1)
a.scatter(ss.A_share_equal_shifted, y, s=42, color=fs.ORANGE, zorder=3, label="order shifted by one sample (control)",
          edgecolor=fs.SURFACE, linewidth=1)
a.scatter(ss.A_share_entries_equal, y, s=42, color=fs.BLUE, zorder=3, label="order in bulk_samples.tsv",
          edgecolor=fs.SURFACE, linewidth=1)
a.set_yticks(y); a.set_yticklabels(order); a.set_xlim(0, 1.06); a.set_ylim(-0.7, len(order) - 0.3)
a.set_xlabel("share of bulk.mtx entries equal to the matrix\nrebuilt from the data package by viallabel name")
a.set_title("(a) Every bulk row is the animal its line names")
a.legend(loc="lower center", bbox_to_anchor=(0.5, 1.07), ncol=1, handletextpad=0.3)
a.text(0.97, len(order) - 1 - 0.45, "1.000 in all 16 tissues", ha="right", va="center", color=fs.INK2, fontsize=8)
a.grid(axis="y", visible=False)

x = {t: i for i, t in enumerate(sorted(p.tissue.unique()))}
rng = np.random.default_rng(0)
for sex, col in (("female", fs.ORANGE), ("male", fs.BLUE)):
    d = p[p.sex == sex]
    b.scatter([x[t] + rng.uniform(-0.22, 0.22) for t in d.tissue], np.log10(d.y_cpm + 1), s=16, color=col, alpha=0.85,
              edgecolor=fs.SURFACE, linewidth=0.4, label=f"recorded {sex} (n = {len(d)})", zorder=3)
b.set_xticks(list(x.values())); b.set_xticklabels(list(x), rotation=60, ha="right")
b.set_ylabel("Y-linked expression, log10(1 + CPM)\n(Ddx3y, Eif2s3y, Kdm5d, Uty)")
lo, hi = p[p.sex == "male"].y_cpm.min(), p[p.sex == "female"].y_cpm.max()
b.axhspan(np.log10(hi + 1), np.log10(lo + 1), color=fs.GRID, alpha=0.6, zorder=0)
b.text(len(x) - 0.5, (np.log10(hi + 1) + np.log10(lo + 1)) / 2, f"gap: female max {hi:.1f}, male min {lo:.1f} CPM",
       ha="right", va="center", color=fs.INK2, fontsize=8)
b.set_title("(b) Expressed sex agrees with recorded sex in all 749 samples")
b.legend(loc="lower center", bbox_to_anchor=(0.5, 1.07), ncol=2, handletextpad=0.2)
b.grid(axis="x", visible=False)
fs.save(fig, FIG / "step1_sample_join.png")

# ---------------------------------------------------------------- Figure 1.2: the male/female asymmetry is in the bulk
g = pd.read_csv(REC / "sex_asymmetry_ours_vs_consortium.tsv", sep="\t").dropna()
lx = np.log2((g.bulk_male + 1) / (g.bulk_female + 1)); ly = np.log2((g.ours_male_per_block + 1) / (g.ours_female_per_block + 1))
fig, ax = plt.subplots(figsize=(6.2, 5.2))
xl = [lx.min() - 1, lx.max() + 1.6]; yl = [ly.min() - 1.2, ly.max() + 1.4]
ax.plot(xl, xl, color=fs.NEUTRAL, lw=1, ls=(0, (4, 3)), zorder=1)
ax.annotate("equal imbalance", (xl[1] - 0.2, xl[1] - 0.2), xytext=(-4, -12), textcoords="offset points", ha="right", fontsize=7.5, color=fs.NEUTRAL)
ax.axhline(0, color=fs.GRID, lw=1); ax.axvline(0, color=fs.GRID, lw=1)
ax.scatter(lx, ly, s=46, color=fs.BLUE, edgecolor=fs.SURFACE, linewidth=1, zorder=3)
off = {"BLOOD": (7, -3), "BAT": (7, -3), "SPLEEN": (7, -3), "SKMGN": (7, -3), "LUNG": (7, -3), "HIPPOC": (7, -3),
       "HEART": (-7, -3), "LIVER": (7, -3), "KIDNEY": (7, -3), "SKMVL": (7, -3), "WATSC": (7, -3), "CORTEX": (7, -12),
       "HYPOTH": (7, 3), "SMLINT": (-7, -3)}
for t, a_, b_ in zip(g.tissue, lx, ly):
    dx, dy = off.get(t, (6, 4))
    ax.annotate(t, (a_, b_), xytext=(dx, dy), textcoords="offset points", fontsize=7.5, color=fs.INK2,
                ha="left" if dx > 0 else "right")
ax.set_xlim(xl); ax.set_ylim(yl)
ax.set_xlabel("consortium bulk analysis: log2 (male calls + 1) / (female calls + 1), 8 weeks")
ax.set_ylabel("this pipeline, mean per cell-type block:\nlog2 (male calls + 1) / (female calls + 1), 8 weeks")
ax.set_title("The male/female imbalance in 8-week calls is inherited from the bulk data")
ax.text(xl[0] + 0.15, yl[1] - 0.3, "Spearman 0.90 over the 14 two-sex tissues;\nsame sign in 12", ha="left", va="top",
        color=fs.INK2, fontsize=8)
ax.text(xl[1] - 0.1, 0.25, "more male calls \u2192", ha="right", va="bottom", color=fs.INK2, fontsize=8)
ax.text(xl[0] + 0.15, 0.25, "\u2190 more female calls", ha="left", va="bottom", color=fs.INK2, fontsize=8)
fs.save(fig, FIG / "step1_sex_asymmetry.png")
