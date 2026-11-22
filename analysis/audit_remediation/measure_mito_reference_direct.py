#!/usr/bin/env python
"""measure_mito_reference_direct.py -- Step 5, M1, second measurement: the mitochondrial fraction of every cell of
every production reference, read from the reference's own count matrix.

measure_mito_fraction.py measures reference cells by matching them to the corpus QC files, so it covers only
references built from corpus cells (12 of 15) and was run on the references in use before refs5. This script reads
each built reference directly (reference_counts.mtx, genes.tsv, cells_meta.tsv), so it covers all 15 refs5 references,
including BLOOD (composite), OVARY and SPLEEN, and the re-annotated HYPOTH. Read-only.

Per cell: counts on the 13 mitochondrial protein-coding genes (chromosome MT, by Ensembl ID) over the cell's total
counts in the reference matrix. Per reference and cell type: cells, median, share above 5/10/15/20/50%.
The reading rule is the registered one for references (STEP5_PREREG.md, M1): 5% or more of a cell type's cells
above 15%, in a type with at least 100 cells.

Usage: measure_mito_reference_direct.py <out_tsv>
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io
import yaml

ROOT = Path(os.environ.setdefault("PIPELINE_ROOT", str(Path(__file__).resolve().parents[2])))
CUTS = [0.05, 0.10, 0.15, 0.20, 0.50]


def mito_ids():
    gi = pd.read_csv(ROOT / "data/references/biomart/rat_gene_info.tsv", sep="\t", dtype=str).fillna("")
    gi.columns = [c.strip() for c in gi.columns]
    m = gi[(gi["Chromosome/scaffold name"] == "MT") & (gi["Gene type"] == "protein_coding")]
    return sorted(m["Gene stable ID"])


def main(out):
    mt = set(mito_ids())
    refs = yaml.safe_load(open(ROOT / "deconvolution/tissue_references.yaml"))
    by_ref = {}
    for code, c in refs.items():
        if isinstance(c, dict) and c.get("reference_dir") and c.get("status") == "ready":
            by_ref.setdefault(c["reference_dir"], []).append(code)
    rows = []
    for ref, codes in sorted(by_ref.items()):
        d = ROOT / ref
        genes = [l.split("\t")[0].strip() for l in open(d / "genes.tsv") if l.strip()]
        meta = pd.read_csv(d / "cells_meta.tsv", sep="\t", dtype=str)
        X = scipy.io.mmread(d / "reference_counts.mtx").tocsr()
        if X.shape == (len(genes), len(meta)) and X.shape != (len(meta), len(genes)):
            X = X.T.tocsr()
        if X.shape != (len(meta), len(genes)):
            raise SystemExit(f"{ref}: matrix {X.shape} does not match {len(meta)} cells x {len(genes)} genes")
        is_mt = np.array([g in mt for g in genes])
        total = np.asarray(X.sum(axis=1)).ravel()
        frac = np.asarray(X[:, is_mt].sum(axis=1)).ravel() / np.maximum(total, 1)
        print(f"{Path(ref).name}: {len(meta)} cells, {int(is_mt.sum())} of {len(mt)} MT genes in the matrix, "
              f"{(frac > 0.15).mean():.4f} above 15%", flush=True)
        for ct, idx in meta.groupby("cell_type").indices.items():
            f = frac[idx]
            row = {"reference": Path(ref).name, "tissues": ",".join(sorted(codes)), "cell_type": ct, "n_cells": len(f),
                   "n_mt_genes_in_matrix": int(is_mt.sum()), "median_mito": float(np.median(f))}
            row.update({f"share_above_{int(c * 100)}": float((f > c).mean()) for c in CUTS})
            rows.append(row)
    t = pd.DataFrame(rows)
    t["flag_registered_rule"] = (t["n_cells"] >= 100) & (t["share_above_15"] >= 0.05)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    t.to_csv(out, sep="\t", index=False)
    s = t.groupby("reference").apply(lambda g: pd.Series({
        "cells": int(g["n_cells"].sum()),
        "share_above_15": float((g["share_above_15"] * g["n_cells"]).sum() / g["n_cells"].sum()),
        "types_with_100_cells": int((g["n_cells"] >= 100).sum()), "types_flagged": int(g["flag_registered_rule"].sum())}),
        include_groups=False)
    s.to_csv(str(out).replace(".tsv", "_summary.tsv"), sep="\t")
    print(s.to_string())


if __name__ == "__main__":
    main(sys.argv[1])
