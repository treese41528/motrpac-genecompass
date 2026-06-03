#!/usr/bin/env python3
"""
build_sex_chrom_list.py -- Derive the rat sex-chromosome gene list for the Stage 8
deconvolution gene cleanup.

BayesPrism's standard cleanup.genes call removes the chrX and chrY gene groups
(gene.group=c("other_Rb","chrM","chrX","chrY","Rb","Mrp","act","hb","MALAT1")),
but its genelists are human/mouse only. We reproduce it for rat from the SAME
biomart annotation used elsewhere (data/references/biomart/rat_gene_info.tsv,
`Chromosome/scaffold name` column) -- all ENSRNOG genes on chromosome X or Y.

Removing sex-chromosome genes prevents sex composition from confounding cell-type
estimates in mixed-sex bulk (a male-only Y-gene or Xist signal can otherwise leak
into a cell-type fraction). For single-/balanced-sex validation it is a near-no-op.

Writes deconvolution/reference/rat_sex_chrom_genes.tsv
  feature_ID  gene_symbol  chromosome

Usage (project venv):  python deconvolution/build_sex_chrom_list.py
"""
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
os.environ.setdefault("PIPELINE_ROOT", str(Path(__file__).resolve().parents[1]))
from gene_utils import load_config, resolve_path, normalize_ensembl_id  # noqa: E402

ID_COL, SYM_COL, CHR_COL = "Gene stable ID", "Gene name", "Chromosome/scaffold name"
SEX_CHROMS = {"X", "Y"}


def main():
    cfg = load_config()
    src = resolve_path(cfg, cfg["biomart"]["rat_gene_info"])
    out = resolve_path(cfg, cfg["deconvolution"]["rat_sex_chrom_genes"])
    df = pd.read_csv(src, sep="\t", dtype=str)
    df.columns = [c.strip() for c in df.columns]
    df = df[df[ID_COL].notna()].copy()
    df["ens"] = df[ID_COL].map(normalize_ensembl_id)
    df = df[df["ens"].str.startswith("ENSRNOG")]
    sex = df[df[CHR_COL].isin(SEX_CHROMS)].drop_duplicates("ens").sort_values(["ens"])
    pd.DataFrame({
        "feature_ID": sex["ens"],
        "gene_symbol": sex[SYM_COL].fillna(""),
        "chromosome": sex[CHR_COL].fillna(""),
    }).to_csv(out, sep="\t", index=False)
    print(f"source: {src}")
    print(f"wrote {out}: {len(sex)} sex-chromosome ENSRNOG genes "
          f"(X={int((sex[CHR_COL]=='X').sum())}, Y={int((sex[CHR_COL]=='Y').sum())})")


if __name__ == "__main__":
    main()
