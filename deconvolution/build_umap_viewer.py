#!/usr/bin/env python3
"""build_umap_viewer.py -- extract the pseudo-cell UMAP + Aim-2 gate stats into a
single self-contained, offline HTML viewer (Atlas / Signal / Focus tabs).

Reads (paths from config/pipeline_config.yaml, deconvolution section):
  <genecompass_input_dir>/umap/umap_coords.tsv  -- pseudo-cells: tissue,cell_type,sex,pa_level,weeks,x,y
  <genecompass_input_dir>/pheno_merge_test.tsv  -- gate: per (tissue,cell_type) eta^2/p for group/trained/sex
Writes (next to the coords):
  umap/viewer.html       self-contained, data inlined, no CDN
  umap/viewer_data.json  the extracted dataset, reusable
Front-end template: deconvolution/umap_viewer_template.html  (with a __DATA__ placeholder).
"""
import csv
import json
import os
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(os.environ.setdefault("PIPELINE_ROOT", str(Path(__file__).resolve().parents[1]))) / "lib"))
from gene_utils import load_config, resolve_path                       # noqa: E402


def fnum(s):
    s = (s or "").strip()
    return None if s == "" or s.lower() in ("nan", "na", "none") else float(s)


def embedding_projections(gc_dir, pts):
    """Load the 768-d GeneCompass CLS embeddings (row-aligned to `pts`) and compute two per-cell
    2-D projections for the Focus tab's swappable views:

      px,py  POOLED WITHIN-cell-type PCA (unsupervised). Each (tissue,cell type) centroid is
             removed, residuals are scaled per group, pooled, standardized, then reduced by ONE PCA
             -> a stable, comparable, drift-free common frame. Its axes capture the LARGEST
             within-type variance, which (gate medians: sex 0.04, exercise 0.02) is mostly
             design-ORTHOGONAL -- so it does NOT specifically show exercise/sex. A variance landscape.

      dx,dy  SUPERVISED design axes, LEAVE-ONE-OUT. dx = projection onto the within-group dose
             (ordinal-week regression) direction; dy = projection onto the sex (male-minus-female
             centroid) direction. Each cell is scored with the direction fit on the OTHER cells of
             its group, so separation is not self-manufactured and a null group collapses. This
             SHOWS the shape of the gate-tested effect -- significance stays the gate's eta^2/p.

    Returns {px,py,dx,dy} (lists), or None on missing / row-misaligned embeddings."""
    try:
        import glob
        import numpy as np
        from datasets import load_from_disk
    except Exception as ex:
        print(f"[warn] embedding projections skipped (imports: {ex})")
        return None
    embs, key_seq = [], []
    for d in sorted(glob.glob(os.path.join(gc_dir, "*", ""))):
        tis = os.path.basename(d.rstrip("/"))
        ep, dp = os.path.join(d, "embeddings", "cell_embeddings.npy"), os.path.join(d, "dataset")
        if not (os.path.exists(ep) and os.path.isdir(dp)):
            continue
        e = np.load(ep)
        ct = list(load_from_disk(dp)["cell_type"])
        if e.shape[0] != len(ct):
            print(f"[warn] embedding projections skipped ({tis} emb/dataset misaligned)")
            return None
        embs.append(e)
        key_seq.extend((tis, c) for c in ct)
    if not embs or len(key_seq) != len(pts):
        print(f"[warn] embedding projections skipped (rows {len(key_seq)} != coords {len(pts)})")
        return None
    for k, (r, (tis, c)) in enumerate(zip(pts, key_seq)):
        if r["tissue"] != tis or r["cell_type"] != c:
            print(f"[warn] embedding projections skipped (row {k} misaligned: {r['tissue']}/{r['cell_type']} vs {tis}/{c})")
            return None

    import numpy as np
    E = np.vstack(embs).astype(np.float64)
    E /= np.linalg.norm(E, axis=1, keepdims=True) + 1e-9          # match cosine geometry of the global UMAP
    n = len(pts)
    sex = np.array([r["sex"] for r in pts])
    week = np.array([float(r["weeks"]) for r in pts])
    groups = {}
    for i, key in enumerate(key_seq):
        groups.setdefault(key, []).append(i)

    # --- pooled WITHIN-cell-type PCA (px,py): remove centroid, per-group scale, pool, standardize, 1 PCA ---
    R = np.zeros_like(E)
    for idx in groups.values():
        idx = np.asarray(idx)
        r = E[idx] - E[idx].mean(0)
        s = np.sqrt((r ** 2).sum() / idx.size) or 1.0
        R[idx] = r / s
    sd = R.std(0); sd[sd == 0] = 1.0
    Z = R / sd; Z -= Z.mean(0)
    _, _, Vt = np.linalg.svd(Z, full_matrices=False)
    pca = Z @ Vt[:2].T
    pca /= pca[:, 0].std() or 1.0

    # --- supervised LEAVE-ONE-OUT design axes (dx=dose, dy=sex) ---
    def unit(v):
        nrm = np.linalg.norm(v)
        return v / nrm if nrm > 0 else v
    dx, dy = np.zeros(n), np.zeros(n)
    for idx in groups.values():
        idx = np.asarray(idx)
        Eg, sg, wg = E[idx], sex[idx], week[idx]
        for p in range(len(idx)):
            keep = np.ones(len(idx), bool); keep[p] = False
            Ek, sk, wk, xi = Eg[keep], sg[keep], wg[keep], Eg[p]
            wc = wk - wk.mean()                                   # dose axis: ordinal-week regression
            if (wc ** 2).sum() > 0:
                dx[idx[p]] = float(xi @ unit((Ek - Ek.mean(0)).T @ wc))
            m, f = Ek[sk == "male"], Ek[sk == "female"]           # sex axis: male - female centroid
            if len(m) and len(f):
                dy[idx[p]] = float(xi @ unit(m.mean(0) - f.mean(0)))
    for idx in groups.values():                                  # center each group, then global axis scale
        idx = np.asarray(idx)
        dx[idx] -= dx[idx].mean(); dy[idx] -= dy[idx].mean()
    dx /= dx.std() or 1.0
    dy /= dy.std() or 1.0

    print(f"embedding projections: pooled within-PCA + supervised LOO design axes for "
          f"{len(groups)} groups ({E.shape[1]}-d, {n} cells)")
    rnd = lambda a: [round(float(v), 4) for v in a]
    return {"px": rnd(pca[:, 0]), "py": rnd(pca[:, 1]), "dx": rnd(dx), "dy": rnd(dy)}


def main():
    cfg = load_config()
    dc = cfg["deconvolution"]
    gc = resolve_path(cfg, dc["genecompass_input_dir"])
    umap_dir = os.path.join(gc, "umap")
    coords_p = os.path.join(umap_dir, "umap_coords.tsv")
    gate_p = os.path.join(gc, "pheno_merge_test.tsv")
    tmpl_p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "umap_viewer_template.html")

    pts = list(csv.DictReader(open(coords_p), delimiter="\t"))
    tissues = sorted({r["tissue"] for r in pts})
    cell_types = sorted({r["cell_type"] for r in pts})
    sexes = sorted({r["sex"] for r in pts})                # female, male
    weeks = sorted({int(r["weeks"]) for r in pts})         # 0,1,2,4,8
    ti = {t: i for i, t in enumerate(tissues)}
    ci = {c: i for i, c in enumerate(cell_types)}
    si = {s: i for i, s in enumerate(sexes)}

    X, Y, T, C, S, W = [], [], [], [], [], []
    tct = {}
    for r in pts:
        X.append(round(float(r["x"]), 3))
        Y.append(round(float(r["y"]), 3))
        t, c = ti[r["tissue"]], ci[r["cell_type"]]
        T.append(t); C.append(c); S.append(si[r["sex"]]); W.append(int(r["weeks"]))
        tct.setdefault(t, set()).add(c)
    tissue_cell_types = {str(t): sorted(cs) for t, cs in tct.items()}

    gate = []
    if os.path.exists(gate_p):
        for r in csv.DictReader(open(gate_p), delimiter="\t"):
            if r["tissue"] not in ti or r["cell_type"] not in ci:
                continue
            gate.append({
                "t": ti[r["tissue"]], "c": ci[r["cell_type"]], "n": int(float(r["n"])),
                "eg": fnum(r["eta2_group"]), "pg": fnum(r["p_group"]),
                "et": fnum(r["eta2_trained"]), "pt": fnum(r["p_trained"]),
                "es": fnum(r["eta2_sex"]), "ps": fnum(r["p_sex"]),
            })

    proj = embedding_projections(gc, pts)
    points = {"x": X, "y": Y, "t": T, "c": C, "s": S, "w": W}
    if proj is not None:
        points.update(proj)

    data = {
        "tissues": tissues, "cellTypes": cell_types, "sexes": sexes, "weeks": weeks,
        "points": points,
        "gate": gate, "tissueCellTypes": tissue_cell_types,
        "generated": date.today().isoformat(),
    }

    js = json.dumps(data, separators=(",", ":"))
    if "</script" in js.lower():
        sys.exit("refusing to inline: data contains a </script> substring")

    json_out = os.path.join(umap_dir, "viewer_data.json")
    with open(json_out, "w") as f:
        f.write(js)

    tmpl = open(tmpl_p).read()
    if "__DATA__" not in tmpl:
        sys.exit(f"template {tmpl_p} missing __DATA__ placeholder")
    html_out = os.path.join(umap_dir, "viewer.html")
    with open(html_out, "w") as f:
        f.write(tmpl.replace("__DATA__", js))

    miss = sum(1 for g in gate if g["eg"] is None and g["et"] is None and g["es"] is None)
    print(f"points={len(X)}  tissues={len(tissues)}  cellTypes={len(cell_types)}  "
          f"weeks={weeks}  gate={len(gate)} ({miss} all-NaN)  projections={'px,py,dx,dy' if proj else 'none'}")
    print(f"wrote {html_out}  ({os.path.getsize(html_out) / 1e6:.2f} MB)")
    print(f"wrote {json_out}  ({os.path.getsize(json_out) / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
