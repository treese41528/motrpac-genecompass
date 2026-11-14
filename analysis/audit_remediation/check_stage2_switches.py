#!/usr/bin/env python
"""check_stage2_switches.py -- do the Stage 2 switches of STEP5_PREREG.md leave the default untouched, and what do
they do when on?

Reads two Stage 2 single-study output folders made by the same code: <default_dir> (committed config) and
<switches_dir> (mito_by_id, apply_max_mito_fraction, outlier_on_n_genes, walk_each_file_once all true), and the
canonical QC folder the v2 corpus was built from.

  default against canonical   same cells in the same order, same genes, equal counts, and the file md5
  switches against default    files produced, cells kept, cells above 15% mitochondrial, stored mito_pct

Usage: check_stage2_switches.py <canonical_qc_dir> <default_dir> <switches_dir> <out_tsv>
"""
import hashlib
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analysis" / "audit_remediation"))
from measure_mito_reference_direct import mito_ids   # noqa: E402


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def frac_mito(a, mt):
    X = a.X if sp.issparse(a.X) else sp.csr_matrix(a.X)
    tot = np.asarray(X.sum(axis=1)).ravel().astype(float)
    m = np.asarray(X[:, np.isin(a.var_names, list(mt))].sum(axis=1)).ravel().astype(float)
    return np.divide(m, tot, out=np.zeros_like(m), where=tot > 0)


def main(canon, dflt, sw, out):
    canon, dflt, sw = Path(canon), Path(dflt), Path(sw)
    mt = set(mito_ids())
    rows = []
    for f in sorted(dflt.glob("*.h5ad")):
        a = ad.read_h5ad(f)
        row = {"sample_id": f.stem, "cells_default": a.n_obs, "default_md5": md5(f)}
        c = canon / f.name
        if c.exists():
            b = ad.read_h5ad(c)
            same_cells = list(a.obs_names) == list(b.obs_names)
            same_genes = list(a.var_names) == list(b.var_names)
            same_counts = same_cells and same_genes and (abs(a.X - b.X).sum() == 0)
            row.update(cells_canonical=b.n_obs, same_cells=same_cells, same_genes=same_genes, same_counts=bool(same_counts),
                       same_md5=row["default_md5"] == md5(c))
        fd = frac_mito(a, mt)
        row.update(default_above_15=int((fd > 0.15).sum()), default_stored_mito_pct_max=float(a.obs["mito_pct"].max()))
        s = sw / f.name
        if s.exists():
            z = ad.read_h5ad(s); fz = frac_mito(z, mt)
            row.update(produced_with_switches=True, cells_switches=z.n_obs, switches_above_15=int((fz > 0.15).sum()),
                       switches_max_mito=float(fz.max()), switches_stored_mito_pct_max=float(z.obs["mito_pct"].max()),
                       switches_cells_all_in_default=bool(set(z.obs_names) <= set(a.obs_names)))
        else:
            row.update(produced_with_switches=False)
        rows.append(row)
    t = pd.DataFrame(rows)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    t.to_csv(out, sep="\t", index=False)
    print(t.to_string())
    ok = t.dropna(subset=["same_counts"])
    print(f"\ndefault against canonical: {int(ok['same_counts'].sum())} of {len(ok)} files have the same cells, genes "
          f"and counts; {int(ok['same_md5'].sum())} have the same md5")


if __name__ == "__main__":
    main(*sys.argv[1:5])
