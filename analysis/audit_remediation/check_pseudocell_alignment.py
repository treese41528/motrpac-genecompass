#!/usr/bin/env python
"""check_pseudocell_alignment.py -- audit findings 6.9 and 8.4, the Python half.

check_sample_join.R ties each bulk row to its animal and to BayesPrism's per-type expression (pred_z). This script
follows the same pseudo-cells through the remaining files, read-only:

  F. pred_z -> pseudocells.h5ad   every pseudo-cell row equals the pred_z row of its (mix<i>, cell type);
  G. h5ad -> tokenized dataset    every dataset row is a pseudo-cell of the h5ad, its sample and cell-type fields
                                  agree with its id, and the dataset keeps the h5ad's order;
  H. dataset -> embeddings        one embedding row per dataset row;
  I. rat run -> human run         which pseudo-cells exist in rat space and not in human space, by cell type
                                  (finding 8.4: 20 LUNG pseudo-cells).

Usage: check_pseudocell_alignment.py <results_root> <gc_root> <gc_human_root> <bulk_root> <out_dir>
"""
import os
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from datasets import load_from_disk

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "deconvolution"))
from celltype_names import safe          # noqa: E402  (the one cell-type -> file-name map; never re-implemented)

res_root, gc_root, gch_root, bulk_root, out_dir = [Path(p) for p in sys.argv[1:6]]
out_dir.mkdir(parents=True, exist_ok=True)

rows, only = [], []
for tdir in sorted(p for p in gc_root.iterdir() if (p / "pseudocells.h5ad").exists()):
    tis = tdir.name
    TIS = tis.upper()
    n_bulk = sum(1 for l in open(bulk_root / TIS / "bulk_samples.tsv") if l.strip())
    a = ad.read_h5ad(tdir / "pseudocells.h5ad")
    X = a.X.toarray() if hasattr(a.X, "toarray") else np.asarray(a.X)
    genes = a.var["ensrnog"].astype(str).to_numpy() if "ensrnog" in a.var else a.var_names.to_numpy()
    obs = a.obs.copy()
    obs["sample"] = obs["sample"].astype(str); obs["cell_type"] = obs["cell_type"].astype(str)

    # F. pseudo-cell rows against pred_z
    cf = pd.read_csv(res_root / TIS / "pred_z" / "celltype_files.tsv", sep="\t")
    f_cells = f_equal = 0
    f_maxdiff = 0.0
    for ct, fn in zip(cf["cell_type"], cf["file"]):
        z = pd.read_csv(res_root / TIS / "pred_z" / fn, index_col=0)
        m = (obs["cell_type"] == ct).to_numpy()
        if not m.any():
            continue
        zz = z.loc[obs.loc[m, "sample"].to_numpy(), genes].to_numpy(dtype=np.float32)
        d = np.abs(X[m].astype(np.float32) - zz)
        f_cells += int(m.sum()); f_equal += int((d.max(axis=1) == 0).sum()); f_maxdiff = max(f_maxdiff, float(d.max()))
    types_missing = sorted(set(obs["cell_type"]) - set(cf["cell_type"]))

    # G. the tokenized dataset against the h5ad
    ds = load_from_disk(str(tdir / "dataset"))
    cid = np.array(ds["cell_id"]); samp = np.array(ds["sample"]); ct = np.array(ds["cell_type"])
    hid = obs["pseudocell_id"].astype(str).to_numpy() if "pseudocell_id" in obs else obs.index.to_numpy()
    pos = pd.Series(np.arange(len(hid)), index=hid)
    g_subset = bool(np.isin(cid, hid).all())
    g_fields = int(sum(c == f"{tis}|{s}|{safe(t)}" for c, s, t in zip(cid, samp, ct)))   # ids carry the file-safe name
    g_order = bool(np.all(np.diff(pos.reindex(cid).to_numpy()) > 0)) if g_subset else False
    idx = np.array([int("".join(ch for ch in s if ch.isdigit())) for s in samp])
    g_range = bool(idx.min() >= 1 and idx.max() <= n_bulk)

    # H. embeddings
    emb = np.load(tdir / "embeddings" / "cell_embeddings.npy", mmap_mode="r")
    h_rows = emb.shape[0]

    # I. the human-space run
    hd = gch_root / tis
    i_n = i_emb = None; i_only = 0; i_extra = 0; i_order = None
    if (hd / "dataset").exists():
        dh = load_from_disk(str(hd / "dataset"))
        hc = np.array(dh["cell_id"]); i_n = len(hc)
        i_emb = np.load(hd / "embeddings" / "cell_embeddings.npy", mmap_mode="r").shape[0]
        lost = ~np.isin(cid, hc)
        i_only = int(lost.sum()); i_extra = int((~np.isin(hc, cid)).sum())
        i_order = bool(np.array_equal(cid[~lost], hc)) if i_extra == 0 else False
        for c, k in pd.Series(ct[lost]).value_counts().items():
            only.append(dict(tissue=tis, cell_type=c, n_rat_only=int(k),
                             n_rat=int((ct == c).sum()), n_human=int((ct == c).sum() - k)))

    rows.append(dict(
        tissue=tis, n_bulk_samples=n_bulk, n_pseudocells_h5ad=a.n_obs, n_genes=a.n_vars,
        F_cells_compared=f_cells, F_cells_identical=f_equal, F_max_abs_diff=f_maxdiff,
        F_types_without_predz=len(types_missing),
        G_dataset_rows=len(cid), G_ids_in_h5ad=g_subset, G_fields_match_id=g_fields, G_order_kept=g_order,
        G_sample_index_in_range=g_range, H_embedding_rows=h_rows,
        I_human_rows=i_n, I_human_embedding_rows=i_emb, I_rat_only=i_only, I_human_only=i_extra,
        I_shared_order_equal=i_order))
    print(f"[{tis}] h5ad={a.n_obs} F identical={f_equal}/{f_cells} (max diff {f_maxdiff:.3g}) | dataset={len(cid)} "
          f"ids-in-h5ad={g_subset} fields={g_fields} order={g_order} | emb={h_rows} | human={i_n} emb={i_emb} "
          f"rat-only={i_only} human-only={i_extra} order={i_order}", flush=True)

s = pd.DataFrame(rows)
s.to_csv(out_dir / "pseudocell_alignment_summary.tsv", sep="\t", index=False)
pd.DataFrame(only, columns=["tissue", "cell_type", "n_rat_only", "n_rat", "n_human"]).to_csv(
    out_dir / "rat_only_pseudocells.tsv", sep="\t", index=False)
ok = bool((s.F_cells_identical == s.F_cells_compared).all() and (s.F_types_without_predz == 0).all()
          and s.G_ids_in_h5ad.all() and (s.G_fields_match_id == s.G_dataset_rows).all() and s.G_order_kept.all()
          and s.G_sample_index_in_range.all() and (s.H_embedding_rows == s.G_dataset_rows).all()
          and (s.I_human_embedding_rows == s.I_human_rows).all() and (s.I_human_only == 0).all()
          and s.I_shared_order_equal.all())
print(f"\nALL TISSUES: links F-H and the shared part of I {'PASS' if ok else 'FAIL'}; "
      f"pseudo-cells in rat space only: {int(s.I_rat_only.sum())}")
