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


if __name__ == "__main__":
    fig_hypoth()
