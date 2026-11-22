#!/usr/bin/env python
"""make_step2_record.py -- re-derive, from the default run's tables, the figures that qualify the headline statements
(audit findings 7.3, 7.5, 8.1, 8.2), write them to docs/audit/remediation/record/step2/ and draw the Step 2 figure.

Read-only on the pipeline outputs. Run from the repository root on the default configuration.
"""
import glob
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as fs                      # noqa: E402
import matplotlib.pyplot as plt            # noqa: E402

G = "data/deconvolution/genecompass_input_v2_refs4all_rebuild"
D = f"{G}/pseudobulk_de"
REC = Path("docs/audit/remediation/record/step2"); REC.mkdir(parents=True, exist_ok=True)
FIG = Path("docs/audit/remediation/figures")

# ---- 7.5: what "53,084" counts
s = pd.read_csv(f"{D}/de_summary.tsv", sep="\t")
ok = s[s.status == "ok"]
per_t, allg = {}, set()
for f in glob.glob(f"{D}/*/de__*.tsv"):
    d = pd.read_csv(f, sep="\t", usecols=["gene", "FDR_IHW"])
    g = set(d.gene[d.FDR_IHW < 0.05]); allg |= g
    per_t.setdefault(f.split("/")[-2], set()).update(g)
t = ok.groupby("tissue").agg(blocks=("cell_type", "size"), pairs=("n_sig_dose_IHW", "sum"),
                             low_fraction_blocks=("mean_fraction", lambda x: int((x < 0.01).sum())))
t["pairs_in_low_fraction_blocks"] = ok[ok.mean_fraction < 0.01].groupby("tissue").n_sig_dose_IHW.sum().reindex(t.index).fillna(0)
t["distinct_genes"] = pd.Series({k: len(v) for k, v in per_t.items()}).reindex(t.index).fillna(0)
t = t.astype(int).sort_values("pairs", ascending=False)
t.loc["ALL"] = [len(ok), int(ok.n_sig_dose_IHW.sum()), int((ok.mean_fraction < 0.01).sum()),
                int(ok[ok.mean_fraction < 0.01].n_sig_dose_IHW.sum()), len(allg)]
t.to_csv(REC / "de_pairs_and_distinct_genes_by_tissue.tsv", sep="\t")
print(t.to_string())

# ---- 8.1 / 8.2: the nine hotspots under multiplicity control on the human side
tr = pd.read_csv(f"{G}_human/gate_rcv10/transfer_comparison.tsv", sep="\t")
a = tr[tr.q_sup_trained.notna()].copy()
a["q_human"] = multipletests(a.p_sup_trained_human, method="fdr_bh")[1]
cut = a.loc[a.q_human < 0.05, "p_sup_trained_human"].max()
h = a[a.is_hotspot.astype(str).str.lower() == "true"].sort_values("p_sup_trained_human")
cols = ["tissue", "cell_type", "sup_trained_auc_rat", "q_sup_trained", "sup_trained_auc_human", "p_sup_trained_human",
        "q_human", "sup_dose_rho_rat", "sup_dose_rho_human", "status"]
h[cols].to_csv(REC / "hotspot_transfer_with_human_bh.tsv", sep="\t", index=False)
print(h[cols].round(4).to_string(index=False)); print("human BH cut:", cut, "family:", len(a))

# ---- 7.3: the registered pipeline-sanity subset of the positive controls
pc = pd.read_csv(f"{D}/posctrl_results.tsv", sep="\t")
san = pc[(pc.tier == "A") & pc.tissue.isin(["SKMVL", "SKMGN", "HEART"])]
x = san.groupby(["tissue", "cell_type"]).outcome.value_counts().unstack(fill_value=0).reset_index()
tot = x.drop(columns=["tissue", "cell_type"]).sum(); tot["tissue"] = "ALL"; tot["cell_type"] = ""
x = pd.concat([x, tot.to_frame().T], ignore_index=True)
x["n"] = x.drop(columns=["tissue", "cell_type"]).sum(axis=1)
x.to_csv(REC / "posctrl_sanity_subset.tsv", sep="\t", index=False); print(x.to_string(index=False))

# ---- figure
fs.use()
fig, (ax, bx) = plt.subplots(1, 2, figsize=(12.5, 4.8), gridspec_kw={"width_ratios": [1.2, 1], "wspace": 0.95})
tt = t.drop("ALL"); tt = tt[tt.pairs > 0]
y = np.arange(len(tt))[::-1]; hgt = 0.36
ax.barh(y + hgt / 2 + 0.02, tt.pairs, height=hgt, color=fs.BLUE, label="gene x block pairs (the quoted 53,084)")
ax.barh(y - hgt / 2 - 0.02, tt.distinct_genes, height=hgt, color=fs.ORANGE, label="distinct genes (9,671 in all)")
for yy, p_, d_ in zip(y, tt.pairs, tt.distinct_genes):
    ax.text(p_ + 250, yy + hgt / 2 + 0.02, f"{p_:,}", va="center", fontsize=7.5, color=fs.INK2)
    ax.text(d_ + 250, yy - hgt / 2 - 0.02, f"{d_:,}", va="center", fontsize=7.5, color=fs.INK2)
ax.set_yticks(y); ax.set_yticklabels(tt.index); ax.set_xlim(0, tt.pairs.max() * 1.14)
ax.set_xlabel("dose-significant at FDR 0.05"); ax.grid(axis="y", visible=False)
ax.set_title("(a) One gene is counted once per cell type")
ax.legend(loc="center right")

hh = h.iloc[::-1]; yb = np.arange(len(hh))
passed = (hh.q_human < 0.05).to_numpy()
bx.axvline(cut, color=fs.NEUTRAL, lw=1, ls=(0, (4, 3)))
bx.text(cut * 0.9, len(hh) - 0.62, f"Benjamini-Hochberg cut\nover {len(a)} blocks:\np = {cut:.4f}", fontsize=7.5, color=fs.INK2, va="top", ha="right")
bx.axvline(0.05, color=fs.GRID, lw=1)
bx.text(0.05 * 0.93, len(hh) - 0.62, "registered rule:\np < 0.05", fontsize=7.5, color=fs.INK2, ha="right", va="top")
bx.scatter(hh.p_sup_trained_human[passed], yb[passed], s=46, color=fs.BLUE, edgecolor=fs.SURFACE, linewidth=1, zorder=3,
           label="passes with correction (7)")
bx.scatter(hh.p_sup_trained_human[~passed], yb[~passed], s=46, facecolor=fs.SURFACE, edgecolor=fs.ORANGE, linewidth=1.8,
           zorder=3, label="passes only uncorrected (2)")
bx.set_xscale("log"); bx.set_xlim(5e-5, 0.09); bx.set_ylim(-0.9, len(hh) - 0.2)
bx.set_yticks(yb); bx.set_yticklabels([f"{a_.upper()}  {b_}" for a_, b_ in zip(hh.tissue, hh.cell_type)], fontsize=7.5)
bx.set_xlabel("human-space permutation p (10,000 permutations)"); bx.grid(axis="y", visible=False)
bx.set_title("(b) Transfer of the nine hotspots")
bx.legend(loc="upper center", bbox_to_anchor=(0.4, -0.16), ncol=2, handletextpad=0.2)
fs.save(fig, FIG / "step2_headline_qualifiers.png")
