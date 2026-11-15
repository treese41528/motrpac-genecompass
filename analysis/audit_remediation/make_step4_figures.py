#!/usr/bin/env python
"""make_step4_figures.py -- figures for docs/audit/remediation/STEP4_RESULTS.md, from the committed record TSVs.

Each figure is drawn only if its record table exists, so the script can be rerun as validations finish.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as fs                      # noqa: E402
import matplotlib.pyplot as plt            # noqa: E402

REC = Path("docs/audit/remediation/record/step4")
FIG = Path("docs/audit/remediation/figures")
fs.use()


def ecdf_panel(ax, null, value, name, title, n_cells):
    x = np.sort(np.asarray(null, float)) * 1e6
    y = np.arange(1, len(x) + 1) / len(x)
    ax.step(x, y, where="post", color=fs.BLUE, lw=2, label=f"{len(x)} genes within 15% of {name}'s expression")
    pct = float((np.asarray(null) < value).mean())
    ax.axhline(0.9, color=fs.NEUTRAL, lw=1, ls=(0, (4, 3)))
    ax.text(0.02, 0.9, "registered bar: 90th percentile", transform=ax.get_yaxis_transform(), ha="left", va="bottom",
            fontsize=7.5, color=fs.INK2)
    ax.scatter([value * 1e6], [pct], s=70, color=fs.ORANGE, edgecolor=fs.SURFACE, linewidth=1.2, zorder=4)
    ax.annotate(f"{name}: {pct:.0%} of the matched genes\nhave a smaller shift", (value * 1e6, pct), xytext=(0.52, 0.42),
                textcoords="axes fraction", ha="left", va="center", fontsize=8, color=fs.INK,
                arrowprops={"arrowstyle": "-", "color": fs.INK2, "lw": 0.8, "shrinkB": 5})
    ax.set_xscale("log")
    ax.set_xlim(min(x.min(), value * 1e6) * 0.7, max(x.max(), value * 1e6) * 1.5); ax.set_ylim(0, 1.04)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}" if str(f"{v:g}")[0] in "25" else ""))
    ax.set_xlabel(f"cell-embedding shift when the gene is deleted\n(x 1e-6, log scale; {n_cells} pseudo-cells)")
    ax.set_ylabel("share of matched genes with a smaller shift")
    ax.set_title(title)
    ax.legend(loc="lower right", handlelength=1.5)


def fig_v4():
    fm, fb = REC / "v4_module_a/muscle/panel_shifts_with_roles.tsv", REC / "v4_module_a/monocytes/myeloid_panel_shifts.tsv"
    if not (fm.exists() and fb.exists()):
        return
    m = pd.read_csv(fm, sep="\t"); b = pd.read_csv(fb, sep="\t")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.4, 4.6), gridspec_kw={"wspace": 0.32})
    ecdf_panel(a1, m.loc[m.role == "matched null", "mean_cell_shift"], float(m.loc[m.role == "Mef2c", "mean_cell_shift"].iloc[0]),
               "Mef2c", "a. Mef2c in skeletal muscle cells (criterion 1)", int(m.n_cells.max()))
    spi = b[b.role.fillna("").str.contains("factor:Spi1")]
    ecdf_panel(a2, b.loc[b.role.fillna("").str.contains("null:Spi1"), "mean_cell_shift"], float(spi.mean_cell_shift.iloc[0]),
               "Spi1", "b. Spi1 in classical monocytes (replication)", int(b.n_cells.max()))
    fs.save(fig, FIG / "step4_v4_module_a.png")


def fig_v3():
    """Held-out model test: per-study and per-tier gene-identity loss of v2, relative to the in-sample probe."""
    R = Path("docs/audit/remediation/record/step4_heldout")
    if not (R / "losses_by_study.tsv").exists():
        return
    s = pd.read_csv(R / "losses_by_study.tsv", sep="\t"); t = pd.read_csv(R / "losses_by_tier.tsv", sep="\t")
    rr = pd.read_csv(R / "registered_reading.tsv", sep="\t").set_index("quantity")["value"]
    pooled = float(rr["ratio_v2_heldout_over_v2_probe_id_loss"])
    h = s[(s.model == "v2") & (s.set == "heldout")].sort_values("id_loss_over_probe").reset_index(drop=True)
    pr = s[(s.model == "v2") & (s.set == "probe") & (s.cells >= 20)]
    fig, (a, b) = plt.subplots(1, 2, figsize=(12.2, 6.4), gridspec_kw={"width_ratios": [1.5, 1], "wspace": 0.42})
    yy = np.arange(len(h))
    a.axvspan(pr.id_loss_over_probe.min(), pr.id_loss_over_probe.max(), color=fs.GRID, alpha=0.55, lw=0, zorder=0)
    a.text(pr.id_loss_over_probe.min() + 0.004, len(h) - 0.4, f"range of {len(pr)} training studies\n(20+ probe cells each)", fontsize=7.5,
           color=fs.INK2, va="top")
    a.scatter(h.id_loss_over_probe, yy, s=34, color=fs.BLUE, edgecolor=fs.SURFACE, linewidth=0.7, zorder=3,
              label=f"{len(h)} held-out studies (2,000 cells each)")
    for x, lab in ((1.10, "Generalises: at most 1.10"), (1.25, "Partly: at most 1.25")):
        a.axvline(x, color=fs.NEUTRAL, lw=1, ls=(0, (4, 3)))
        a.text(x + 0.004, 9.5, lab, fontsize=7.5, color=fs.INK2, rotation=90, va="bottom")
    a.axvline(pooled, color=fs.ORANGE, lw=2)
    a.text(pooled - 0.005, 19.5, f"all held-out cells:\n{pooled:.3f}", fontsize=8, color=fs.INK, ha="right")
    a.set_yticks(yy); a.set_yticklabels([f"{r.study}  {str(r.tissues).split(':')[0].split('(')[0].strip()}" for r in h.itertuples()], fontsize=6.8)
    a.set_xlim(0.86, 1.30); a.set_ylim(-0.8, len(h) - 0.2); a.grid(axis="y", visible=False)
    a.set_xlabel("gene-identity loss on a study's cells / loss on the in-sample probe (v2)")
    a.set_title("a. Every held-out study against the registered bars")
    a.legend(loc="upper left", bbox_to_anchor=(0.0, -0.1), handletextpad=0.3, borderaxespad=0)
    v = t[t.model == "v2"].pivot(index="tier", columns="set", values="id_loss")
    v["ratio"] = v["heldout"] / v["probe"]
    order = ["T1_tri_species", "T2a_human_one2one", "T2b_mouse_one2one", "T3a_human_multi", "T3b_mouse_multi", "T4_new_token"]
    names = ["T1 three-species\northolog", "T2a human\none-to-one", "T2b mouse\none-to-one", "T3a human\nmulti-mapping",
             "T3b mouse\nmulti-mapping", "T4 rat-only\ntoken"]
    v = v.loc[order]; yb = np.arange(len(v))[::-1]
    b.barh(yb, v.ratio - 1, left=1, color=fs.BLUE, height=0.62)
    for y_, r in zip(yb, v.ratio):
        b.text(r + 0.004, y_, f"{r:.3f}", va="center", fontsize=8, color=fs.INK2)
    b.axvline(1.10, color=fs.NEUTRAL, lw=1, ls=(0, (4, 3)))
    b.set_yticks(yb); b.set_yticklabels(names); b.set_xlim(1.0, 1.24); b.grid(axis="y", visible=False)
    b.set_xlabel("held-out loss / probe loss, by token tier (v2)")
    b.set_title("b. The gap is widest for rat-only tokens")
    fs.save(fig, FIG / "step4_v3_heldout.png")


def fig_v1():
    """Purity sweeps of the called cell types: median Pearson-VST against purity, by reference, with each type's abundance."""
    f = REC / "v1_purity_sweeps" / "sweep_points.tsv"
    cb = Path("docs/audit/remediation/record/step3/downstream/called_blocks_all_readings.tsv")
    if not (f.exists() and cb.exists()):
        return
    p = pd.read_csv(f, sep="\t"); p = p[p.kind == "hotspot"]; c = pd.read_csv(cb, sep="\t")
    groups = [("BLOOD (composite reference)", ["BLOOD"]), ("LUNG", ["LUNG"]), ("HEART", ["HEART"]), ("Muscle (one reference for SKMGN and SKMVL)", ["SKMGN", "SKMVL"])]
    cols = [fs.BLUE, fs.ORANGE, fs.AQUA, fs.INK2]
    short = lambda x: x.replace("Vascular smooth muscle cells", "VSMC").replace("Megakaryocytes and platelets", "Megakaryocytes")
    fig, axes = plt.subplots(1, 4, figsize=(15.2, 4.5), sharey=True, gridspec_kw={"wspace": 0.08})
    for ax, (title, tis) in zip(axes, groups):
        d = p[p.tissue.isin(tis)]
        for (focal, g), col, mk in zip(d.groupby("focal", sort=False), cols, ("o", "s", "D", "^")):
            g = g.sort_values("target_purity")
            ax.plot(g.target_purity, g.pearson_vst_median, color=col, marker=mk, ms=4.5, lw=1.8, label=short(focal))
            for r in c[(c.cell_type == focal) & c.tissue.isin(tis)].itertuples():       # the type's abundance in each tissue that calls it
                y = float(np.interp(np.log(max(r.mean_fraction, 0.005)), np.log(g.target_purity), g.pearson_vst_median))
                ax.scatter([max(r.mean_fraction, 0.005)], [y], s=120, facecolor="none", edgecolor=col, linewidth=1.6, zorder=5)
        ax.axhline(0.95, color=fs.NEUTRAL, lw=1, ls=(0, (4, 3)))
        ax.set_xscale("log"); ax.set_xlim(0.004, 1.0); ax.set_ylim(0.45, 1.01)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}" if v >= 0.01 else f"{v:.1%}"))
        ax.set_title(title, fontsize=9); ax.set_xlabel("share of the mixture that is the cell type")
        ax.legend(loc="lower right", handlelength=1.6, fontsize=7.5)
    axes[0].set_ylabel("median correlation of estimated\nwith true expression (Pearson, VST)")
    axes[2].text(0.0045, 0.955, "registered bar: 0.95", fontsize=7.5, color=fs.INK2, va="bottom")
    fig.text(0.125, -0.06, "Open circle: the cell type's mean estimated fraction in the MoTrPAC tissue (drawn at 0.5% when lower). "
             "SKMGN Schwann cells could not be swept: the reference has 30.", fontsize=8, color=fs.INK2)
    fs.save(fig, FIG / "step4_v1_purity_sweeps.png")


if __name__ == "__main__":
    fig_v4()
    fig_v3()
    fig_v1()
