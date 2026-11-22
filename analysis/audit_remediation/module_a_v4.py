#!/usr/bin/env python
"""module_a_v4.py -- Step 4, V4: the myeloid replication of Module A, scored for the first time
(docs/audit/remediation/STEP4_PREREG.md, V4 and its addendum of 2026-11-14).

  panel   from a tokenized dataset and a cell type: the genes whose mean value over the cell type's pseudo-cells
          is within 15% of each factor's (Spi1, Cebpa). Writes the gene list for perturb_cells.py and a table of
          every gene's mean value and the cells it is in.
  score   from perturb_cells.py's results: each factor's cell shift as a percentile of its matched null (replicates
          above the 90th), whether it was testable, and the Spearman correlation of shift with mean value over all
          perturbed genes. Writes a TSV, a JSON and prints the lines.

The matched-null rule and the 90th-percentile threshold are those deconvolution/score_perturb_prereg.py applies to
Mef2c in muscle. Targets are not scored (addendum).

  module_a_v4.py panel --dataset D --cell-type "Classical monocytes" --out-dir O
  module_a_v4.py score --results O/results.tsv --panel O/panel.tsv --out-dir O
"""
import argparse
import collections
import json
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(os.environ.get("PIPELINE_ROOT", Path(__file__).resolve().parents[2])).resolve()
FACTORS = {"ENSRNOG00000012172": "Spi1", "ENSRNOG00000010918": "Cebpa"}     # deconvolution/reference/perturb_prereg_genes.tsv
TOL, PCT, MIN_CELLS = 0.15, 90.0, 20


def token_dict():
    return pickle.load(open(ROOT / "data" / "training" / "ortholog_mappings" / "rat_tokens.pickle", "rb"))


def mean_values(dataset, cell_type):
    """token -> (mean value over the cells it is in, number of those cells); number of cells of the type."""
    from datasets import load_from_disk
    ds = load_from_disk(dataset).filter(lambda e: e["cell_type"] == cell_type)
    vs, vn = collections.defaultdict(float), collections.defaultdict(int)
    for r in ds:
        for t, v in zip(r["input_ids"], r["values"]):
            if t:
                vs[t] += v; vn[t] += 1
    return {t: (vs[t] / vn[t], vn[t]) for t in vs}, len(ds)


def matched(values, factor_value, exclude):
    """Genes within TOL of the factor's mean value (the scorer's rule), without the excluded ones."""
    return sorted(g for g, v in values.items() if g not in exclude and abs(v - factor_value) < TOL * factor_value)


def panel(a):
    tok = token_dict(); inv = {v: k for k, v in tok.items()}
    mv, n = mean_values(a.dataset, a.cell_type)
    if n == 0:
        raise SystemExit(f"no cell of type {a.cell_type!r} in {a.dataset}")
    t = pd.DataFrame([{"gene": inv[t_], "mean_value": v, "n_cells_present": c} for t_, (v, c) in mv.items() if t_ in inv])
    values = dict(zip(t.gene, t.mean_value))
    t["role"] = ""
    genes = []
    for f, name in FACTORS.items():
        if f not in values:
            print(f"{name}: token not in the input of any {a.cell_type} pseudo-cell -> not testable (coverage)")
            continue
        m = matched(values, values[f], set(FACTORS))
        t.loc[t.gene == f, "role"] = "factor:" + name
        t.loc[t.gene.isin(m), "role"] = t.loc[t.gene.isin(m), "role"].map(lambda r: (r + ";" if r else "") + "null:" + name)
        genes += [f] + m
        print(f"{name}: mean value {values[f]:.4f} in {int(t.loc[t.gene == f, 'n_cells_present'].iloc[0])} of {n} cells; "
              f"{len(m)} genes within {TOL:.0%}")
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    genes = list(dict.fromkeys(genes))
    (out / "panel_genes.txt").write_text("\n".join(genes) + ("\n" if genes else ""))
    t["n_cells_of_type"] = n
    t[t.role != ""].sort_values("mean_value").to_csv(out / "panel.tsv", sep="\t", index=False)
    print(f"panel: {len(genes)} genes -> {out / 'panel_genes.txt'}")


def score(a):
    res = pd.read_csv(a.results, sep="\t")
    pan = pd.read_csv(a.panel, sep="\t")
    res = res.merge(pan[["gene", "mean_value", "n_cells_present", "role", "n_cells_of_type"]], on="gene", how="left")
    res["mean_cell_shift"] = pd.to_numeric(res["mean_cell_shift"], errors="coerce")
    rows = []
    for f, name in FACTORS.items():
        r = res[res.gene == f]
        row = {"factor": name, "gene": f}
        if r.empty or pd.isna(r.mean_cell_shift.iloc[0]):
            row.update(reading="not testable (coverage: token absent from the cells)")
        else:
            r = r.iloc[0]
            null = res[res.role.fillna("").str.contains("null:" + name) & res.mean_cell_shift.notna()]
            pct = float((null.mean_cell_shift < r.mean_cell_shift).mean() * 100) if len(null) else float("nan")
            row.update(n_cells=int(r.n_cells), n_cells_of_type=int(r.n_cells_of_type), mean_value=float(r.mean_value),
                       shift=float(r.mean_cell_shift), n_null=len(null), null_median_shift=float(null.mean_cell_shift.median()),
                       null_p90_shift=float(null.mean_cell_shift.quantile(0.9)), percentile=pct)
            row["reading"] = ("not testable (power: token in fewer than 20 cells)" if r.n_cells < MIN_CELLS else
                              "not testable (no matched gene)" if not len(null) else
                              "replicates" if pct > PCT else "does not replicate")
        rows.append(row)
    t = pd.DataFrame(rows)
    ok = res[res.mean_cell_shift.notna() & res.mean_value.notna()]
    rho, p = spearmanr(ok.mean_value, ok.mean_cell_shift) if len(ok) > 2 else (float("nan"), float("nan"))
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    t.to_csv(out / "myeloid_replication.tsv", sep="\t", index=False)
    res.to_csv(out / "myeloid_panel_shifts.tsv", sep="\t", index=False)
    (out / "myeloid_replication.json").write_text(json.dumps(
        {"factors": json.loads(t.to_json(orient="records")), "spearman_shift_value": {"rho": float(rho), "p": float(p), "n": len(ok)},
         "rule": {"tolerance": TOL, "percentile": PCT, "min_cells": MIN_CELLS}}, indent=2))
    print(t.to_string(index=False))
    print(f"\nSpearman(shift, mean value) over {len(ok)} perturbed genes: rho = {rho:.3f} (p = {p:.1e}); no verdict")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p1 = sub.add_parser("panel"); p1.add_argument("--dataset", required=True); p1.add_argument("--cell-type", required=True)
    p1.add_argument("--out-dir", required=True); p1.set_defaults(fn=panel)
    p2 = sub.add_parser("score"); p2.add_argument("--results", required=True); p2.add_argument("--panel", required=True)
    p2.add_argument("--out-dir", required=True); p2.set_defaults(fn=score)
    a = ap.parse_args(); a.fn(a)
