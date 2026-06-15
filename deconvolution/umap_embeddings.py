#!/usr/bin/env python3
"""umap_embeddings.py -- UMAP of the deconv->GeneCompass pseudo-cell embeddings, across all
tissues, color-coded by cell type / tissue / sex / PA level (exercise group) / time (weeks).

Joins each pseudo-cell to PHENO via: sample 'mix{i}' -> i-th viallabel in <TISSUE>/bulk_samples.tsv
-> PHENO[viallabel] (sex, group). PA level = exercise group (control/1w/2w/4w/8w); time = weeks
(control=0). Outputs (under <out>):
  umap_coords.tsv        x,y + metadata per pseudo-cell (reusable)
  umap_panels.png        static faceted UMAP (cell_type/tissue/sex/PA/time)
  umap_interactive.html  plotly: legend toggles CELL TYPES on/off; buttons recolor by facet

Usage: python deconvolution/umap_embeddings.py [--gc-root ...] [--pheno ...] [--bulk-root ...] [--out ...]
"""
import argparse
import glob
import os
import re

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(os.environ.setdefault("PIPELINE_ROOT", str(Path(__file__).resolve().parents[1]))) / "lib"))
from gene_utils import load_config, resolve_path                       # noqa: E402


def weeks_of(group):
    g = str(group).lower()
    if "control" in g or "sedentary" in g:
        return 0
    m = re.search(r"(\d+)\s*w", g)
    return int(m.group(1)) if m else -1


def load_all(gc_root, bulk_root, pheno):
    ph = pd.read_csv(pheno, sep="\t", dtype=str).drop_duplicates("viallabel").set_index("viallabel")
    from datasets import load_from_disk
    embs, meta = [], []
    for d in sorted(glob.glob(f"{gc_root}/*/")):
        tis = os.path.basename(d.rstrip("/"))
        ep, dp = os.path.join(d, "embeddings", "cell_embeddings.npy"), os.path.join(d, "dataset")
        bs = os.path.join(bulk_root, tis.upper(), "bulk_samples.tsv")
        if not (os.path.exists(ep) and os.path.isdir(dp) and os.path.exists(bs)):
            continue
        e = np.load(ep); ds = load_from_disk(dp)
        ct = np.array(ds["cell_type"]); samp = np.array(ds["sample"])
        if e.shape[0] != len(ct):
            print(f"[{tis}] misaligned, skip"); continue
        vls = [l.strip() for l in open(bs) if l.strip()]
        for k in range(len(ct)):
            i = int("".join(filter(str.isdigit, str(samp[k])))) - 1
            v = vls[i] if 0 <= i < len(vls) else ""
            grp = ph.loc[v, "group"] if v in ph.index else "NA"
            sex = ph.loc[v, "sex"] if v in ph.index else "NA"
            meta.append({"tissue": tis, "cell_type": str(ct[k]), "sex": str(sex),
                         "pa_level": str(grp), "weeks": weeks_of(grp)})
        embs.append(e)
        print(f"  {tis}: {len(ct)} pseudo-cells")
    return np.vstack(embs), pd.DataFrame(meta)


def static_panels(M, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    facets = [("cell_type", "cat"), ("tissue", "cat"), ("sex", "cat"),
              ("pa_level", "cat"), ("weeks", "cont")]
    fig, axes = plt.subplots(2, 3, figsize=(22, 13))
    cmap = plt.colormaps["tab20"]
    for ax, (f, kind) in zip(axes.ravel(), facets):
        if kind == "cont":
            sc = ax.scatter(M.x, M.y, s=4, c=M[f], cmap="viridis", rasterized=True)
            fig.colorbar(sc, ax=ax, fraction=0.046)
        else:
            cats = sorted(M[f].astype(str).unique())
            for i, c in enumerate(cats):
                m = M[f].astype(str) == c
                ax.scatter(M.x[m], M.y[m], s=4, color=cmap(i % 20), label=c, rasterized=True)
            if len(cats) <= 22:                       # legend only when readable
                ax.legend(markerscale=3, fontsize=7, ncol=2, loc="best", framealpha=0.6)
            else:
                ax.set_xlabel(f"{len(cats)} categories (use interactive HTML)", fontsize=8)
        ax.set_title(f, fontsize=13); ax.set_xticks([]); ax.set_yticks([])
    axes.ravel()[5].axis("off")
    fig.suptitle("deconv->GeneCompass pseudo-cell UMAP", fontsize=15)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120, bbox_inches="tight")
    print(f"wrote {out_png}")


def interactive_html(M, out_html):
    import plotly.graph_objects as go
    import plotly.express as px
    import matplotlib.cm as cm

    def cat_map(series):
        cats = sorted(series.astype(str).unique())
        pal = px.colors.qualitative.Dark24 + px.colors.qualitative.Light24 + px.colors.qualitative.Alphabet
        return {c: pal[i % len(pal)] for i, c in enumerate(cats)}

    def cont_colors(series):
        v = series.astype(float).to_numpy(); rng = (v.max() - v.min()) or 1.0
        return ["#%02x%02x%02x" % tuple(int(255 * x) for x in cm.viridis((val - v.min()) / rng)[:3]) for val in v]

    facets = ["cell_type", "tissue", "sex", "pa_level"]
    cmaps = {f: cat_map(M[f]) for f in facets}
    weeks_col = cont_colors(M["weeks"])
    M = M.reset_index(drop=True)
    M["_wcol"] = weeks_col

    # one trace per cell type -> legend toggles cell types on/off
    fig = go.Figure()
    trace_idx = []                                    # row indices per trace, for recolor restyle
    for ct in sorted(M["cell_type"].unique()):
        sub = M[M["cell_type"] == ct]
        trace_idx.append(sub.index.to_numpy())
        hov = ("<b>%{customdata[0]}</b><br>tissue=%{customdata[1]}<br>"
               "sex=%{customdata[2]}<br>PA=%{customdata[3]}<br>weeks=%{customdata[4]}<extra></extra>")
        fig.add_trace(go.Scattergl(
            x=sub.x, y=sub.y, mode="markers", name=ct[:30],
            marker=dict(size=4, color=cmaps["cell_type"][ct]),
            customdata=sub[["cell_type", "tissue", "sex", "pa_level", "weeks"]].to_numpy(),
            hovertemplate=hov))

    def colors_for(facet):                            # per-trace list of per-point colors
        if facet == "weeks":
            return [M.loc[idx, "_wcol"].tolist() for idx in trace_idx]
        return [[cmaps[facet][str(v)] for v in M.loc[idx, facet]] for idx in trace_idx]

    buttons = [dict(label=f"color: {f}", method="restyle", args=[{"marker.color": colors_for(f)}])
               for f in facets + ["weeks"]]
    fig.update_layout(
        title="deconv->GeneCompass pseudo-cell UMAP  (legend: click a cell type to hide/show; "
              "buttons: recolor by facet)",
        updatemenus=[dict(type="dropdown", buttons=buttons, x=1.02, y=1.0, xanchor="left")],
        legend=dict(title="cell type (toggle)", font=dict(size=8), itemsizing="constant"),
        width=1300, height=850, template="plotly_white")
    fig.write_html(out_html, include_plotlyjs="cdn")
    print(f"wrote {out_html}")


def main():
    cfg = load_config(); dc = cfg["deconvolution"]
    gc = resolve_path(cfg, dc["genecompass_input_dir"])
    ap = argparse.ArgumentParser()
    ap.add_argument("--gc-root", default=gc)
    ap.add_argument("--pheno", default=resolve_path(cfg, dc["sample_pheno"]))
    ap.add_argument("--bulk-root", default=resolve_path(cfg, dc["motrpac_bulk_out"]))
    ap.add_argument("--out", default=os.path.join(gc, "umap"))
    ap.add_argument("--n-neighbors", type=int, default=30)
    ap.add_argument("--min-dist", type=float, default=0.3)
    args = ap.parse_args()

    E, M = load_all(args.gc_root, args.bulk_root, args.pheno)
    print(f"combined: {E.shape[0]} pseudo-cells x {E.shape[1]} dims, {M.tissue.nunique()} tissues, "
          f"{M.cell_type.nunique()} cell types")
    import umap
    XY = umap.UMAP(n_neighbors=args.n_neighbors, min_dist=args.min_dist,
                   metric="cosine", random_state=0).fit_transform(E)
    M["x"], M["y"] = XY[:, 0], XY[:, 1]
    os.makedirs(args.out, exist_ok=True)
    M.to_csv(os.path.join(args.out, "umap_coords.tsv"), sep="\t", index=False)
    static_panels(M, os.path.join(args.out, "umap_panels.png"))
    interactive_html(M, os.path.join(args.out, "umap_interactive.html"))


if __name__ == "__main__":
    main()
