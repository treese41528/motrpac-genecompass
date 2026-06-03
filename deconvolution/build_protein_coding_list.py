#!/usr/bin/env python3
"""
build_protein_coding_list.py -- Derive the rat protein-coding gene list for the
Stage 8 deconvolution gene cleanup.

BayesPrism's select.gene.type="protein_coding" is human/mouse-only (process_input.R
stopifnot species %in% c("hs","mm")). We reproduce it for rat from the SAME biotype
source GeneCompass Stage 2 uses: biomart rat_gene_info.tsv (`Gene type` column),
via lib/gene_utils.normalize_biotype for biotype canonicalization -- so the
deconvolution protein-coding definition is identical to the corpus's.

Writes deconvolution/reference/rat_protein_coding_genes.tsv
  feature_ID  gene_symbol  chromosome    (chromosome enables a later sex-chrom filter)

Usage (project venv):  python deconvolution/build_protein_coding_list.py
"""
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
os.environ.setdefault("PIPELINE_ROOT", str(Path(__file__).resolve().parents[1]))
from gene_utils import load_config, resolve_path, normalize_biotype, normalize_ensembl_id  # noqa: E402

ID_COL, BIO_COL, SYM_COL, CHR_COL = (
    "Gene stable ID", "Gene type", "Gene name", "Chromosome/scaffold name")


def main():
    cfg = load_config()
    src = resolve_path(cfg, cfg["biomart"]["rat_gene_info"])
    out = resolve_path(cfg, cfg["deconvolution"]["rat_protein_coding_genes"])
    df = pd.read_csv(src, sep="\t", dtype=str)
    df.columns = [c.strip() for c in df.columns]
    df = df[df[ID_COL].notna()].copy()
    df["ens"] = df[ID_COL].map(normalize_ensembl_id)
    df = df[df["ens"].str.startswith("ENSRNOG")]
    df["bt"] = df[BIO_COL].map(normalize_biotype)
    pc = df[df["bt"] == "protein_coding"].drop_duplicates("ens").sort_values("ens")
    pd.DataFrame({
        "feature_ID": pc["ens"],
        "gene_symbol": pc[SYM_COL].fillna(""),
        "chromosome": pc.get(CHR_COL, pd.Series(dtype=str)).fillna(""),
    }).to_csv(out, sep="\t", index=False)
    print(f"source: {src}")
    print(f"wrote {out}: {len(pc)} protein_coding ENSRNOG genes")


if __name__ == "__main__":
    main()
