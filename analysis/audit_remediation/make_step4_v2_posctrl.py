#!/usr/bin/env python
"""make_step4_v2_posctrl.py -- Step 4, V2: the registered positive controls on refs5, reported in the registered order
(docs/audit/remediation/STEP4_PREREG.md V2; specification deconvolution/POSCTRL_PREREG.md, unchanged).

Reads the Stage 10 output of a run (posctrl_results.tsv, written by compare_posctrl.py) and writes
  sanity_subset.tsv        Tier A controls in SKMVL, SKMGN and HEART (the "expected to recover" set), by outcome
  by_tier.tsv              every tier, by outcome
  not_testable_trace.tsv   for each control that could not be tested: where its gene ID is lost (bulk, reference, the
                           deconvolved gene set, the DE table)
  against_default.tsv      each control's outcome in this run beside the earlier default's
  parenchyma_dataanchored.tsv  the later, unregistered check, copied as numbers
and prints the reading the registration fixed: fewer than half of the testable sanity controls recovered means the
registered red flag stands.

Usage: make_step4_v2_posctrl.py <de_dir> <results_root> <default_de_dir> <out_dir>
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "deconvolution"))
from celltype_names import safe          # noqa: E402

SANITY_TISSUES = ["SKMVL", "SKMGN", "HEART"]


def counts(d, by):
    x = d.groupby(by).outcome.value_counts().unstack(fill_value=0).reset_index()
    tot = x.drop(columns=by).sum(); x = pd.concat([x, tot.to_frame().T], ignore_index=True)
    x.loc[len(x) - 1, by] = ["ALL"] + [""] * (len(by) - 1)
    x["n"] = x.drop(columns=by).sum(axis=1)
    return x


def trace(pc, de_dir, res_root):
    refs = yaml.safe_load(open(ROOT / "deconvolution/tissue_references.yaml"))
    cache, rows, refsum = {}, [], {}
    gi = pd.read_csv(ROOT / "data/references/biomart/rat_gene_info.tsv", sep="\t", dtype=str).fillna("")
    gi.columns = [c.strip() for c in gi.columns]
    chrom = dict(zip(gi["Gene stable ID"], gi["Chromosome/scaffold name"]))
    gtype = dict(zip(gi["Gene stable ID"], gi["Gene type"]))
    excl = set(pd.read_csv(ROOT / "deconvolution/reference/rat_exclude_genes_v2.tsv", sep="\t", dtype=str)["feature_ID"])
    for r in pc[pc.outcome == "not_testable"].itertuples():
        t = r.tissue
        if t not in cache:
            bulk = set(pd.read_csv(ROOT / "data/deconvolution/motrpac_bulk" / t / "bulk_genes.tsv", sep="\t", header=None, dtype=str)[0])
            ref = set(l.split("\t")[0].strip() for l in open(ROOT / refs[t]["reference_dir"] / "genes.tsv"))
            z = set(l.strip() for l in open(Path(res_root) / t / "pred_z" / "genes.txt"))
            cache[t] = (bulk, ref, z)
        bulk, ref, z = cache[t]
        rd = refs[t]["reference_dir"]
        if rd not in refsum:                      # counts of every gene over the reference's cells, read once
            genes = [l.split("\t")[0].strip() for l in open(ROOT / rd / "genes.tsv") if l.strip()]
            X = scipy.io.mmread(ROOT / rd / "reference_counts.mtx").tocsr()
            n_cells = sum(1 for _ in open(ROOT / rd / "cells_meta.tsv")) - 1
            X = X.T.tocsr() if X.shape == (len(genes), n_cells) and X.shape != (n_cells, len(genes)) else X
            refsum[rd] = (dict(zip(genes, np.asarray(X.sum(axis=0)).ravel())),
                          dict(zip(genes, np.asarray((X > 0).sum(axis=0)).ravel())))
        e = r.ensembl if isinstance(r.ensembl, str) else ""
        f = Path(de_dir) / t / f"de__{safe(r.cell_type)}.tsv"
        in_de = f.exists() and e in set(pd.read_csv(f, sep="\t", usecols=["gene"], dtype=str)["gene"])
        where = ("no ID in the specification" if not e else "not in the bulk" if e not in bulk else
                 "not in the reference" if e not in ref else "in bulk and reference, not in the deconvolved gene set" if e not in z
                 else "no DE table for the block" if not f.exists() else "deconvolved, dropped by the DE (all-zero or filtered)" if not in_de
                 else "in the DE table, below the testing floor")
        rows.append({"tier": r.tier, "group": r.group, "symbol": r.symbol, "ensembl": e, "tissue": t, "cell_type": r.cell_type,
                     "in_bulk": e in bulk, "in_reference": e in ref, "in_deconvolved_genes": e in z, "in_de_table": in_de,
                     "chromosome": chrom.get(e, ""), "gene_type": gtype.get(e, ""), "on_exclusion_list_v2": e in excl,
                     "reference_counts": float(refsum[rd][0].get(e, np.nan)),
                     "reference_cells_expressing": float(refsum[rd][1].get(e, np.nan)),
                     "lost_at": where})
    return pd.DataFrame(rows)


def main(de_dir, res_root, default_de, out):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    pc = pd.read_csv(Path(de_dir) / "posctrl_results.tsv", sep="\t")
    san = pc[(pc.tier == "A") & pc.tissue.isin(SANITY_TISSUES)]
    s = counts(san, ["tissue", "cell_type"]); s.to_csv(out / "sanity_subset.tsv", sep="\t", index=False)
    counts(pc, ["tier"]).to_csv(out / "by_tier.tsv", sep="\t", index=False)
    tr = trace(pc, de_dir, res_root); tr.to_csv(out / "not_testable_trace.tsv", sep="\t", index=False)
    key = ["tier", "group", "symbol", "tissue", "sex", "cell_type"]
    old = pd.read_csv(Path(default_de) / "posctrl_results.tsv", sep="\t")
    m = pc[key + ["outcome", "FDR_IHW", "lfc_8w"]].merge(old[key + ["outcome", "FDR_IHW", "lfc_8w"]], on=key, how="outer",
                                                         suffixes=("_refs5", "_default"))
    m.to_csv(out / "against_default.tsv", sep="\t", index=False)
    pa = Path(de_dir) / "parenchyma_dataanchored_validation.tsv"
    if pa.exists():
        pd.read_csv(pa, sep="\t").to_csv(out / "parenchyma_dataanchored.tsv", sep="\t", index=False)

    tot = s.iloc[-1]
    rec = int(tot.get("recovered", 0)); nt = int(tot.get("not_testable", 0)); n = int(tot["n"]); testable = n - nt
    print(s.to_string(index=False))
    print(f"\nsanity subset: {rec} of {n} recovered; {testable} testable; {rec} of {testable} testable recovered "
          f"({rec / testable:.0%})" if testable else "\nsanity subset: none testable")
    print("registered reading: " + ("the red flag stands (fewer than half of the testable controls recovered)"
                                    if testable and rec < testable / 2 else "at least half of the testable controls recovered"))
    print("\nby tier:"); print(counts(pc, ["tier"]).to_string(index=False))
    print("\nnot testable, by where the gene is lost:"); print(tr.lost_at.value_counts().to_string() if len(tr) else "none")
    ch = m[(m.outcome_refs5 != m.outcome_default)]
    print(f"\ncontrols whose outcome differs from the default: {len(ch)} of {len(m)}")
    if len(ch):
        print(pd.crosstab(ch.outcome_default.fillna("absent"), ch.outcome_refs5.fillna("absent")).to_string())


if __name__ == "__main__":
    main(*sys.argv[1:5])
