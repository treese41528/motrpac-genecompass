#!/usr/bin/env python3
"""
build_exclude_genes.py -- build the rat gene-exclusion list for the Stage 8 deconvolution cleanup.

BayesPrism's cleanup.genes removes ribosomal protein genes (its groups Rb, Mrp, other_Rb), mitochondrial genes
(chrM) and hemoglobin genes (hb), but its gene lists are human and mouse only (Chu et al. 2022). This script makes
the rat equivalent from the biomart table of the Ensembl release the pipeline uses (115;
data/references/biomart/rat_gene_info.tsv), so the list is in the same ID space as the references and the bulk.

  ribosomal    the gene description contains "ribosomal protein" or "ribosomal subunit protein" and does not
               contain "kinase". This takes the cytosolic and mitochondrial ribosomal protein genes, their
               paralogues and pseudogene-like copies (BayesPrism's other_Rb), Fau and Rps19bp1 (both in
               BayesPrism's lists), whatever symbol the release gives them: about 120 of the 415 carry a LOC,
               clone or empty symbol. It leaves out the ribosomal protein S6 kinases (Rps6ka1-6, Rps6kb1-2,
               Rps6kc1, Rps6kl1, Rskr), which are kinases; Kgd4, once named Mrps36 and now annotated as a subunit
               of alpha-ketoglutarate dehydrogenase; and Rpl-1, whose symbol looks ribosomal and which is a
               placental lactogen.
  mito         every gene on chromosome MT.
  hemoglobin   the gene description starts with "hemoglobin" (alpha, beta, embryonic and theta chains; not Ahsp).

History (audit finding 4.3; docs/audit/remediation/STEP3_PREREG.md section 2.1 and Amendment 1). Until 2026-11-14
the committed list had no builder. It had been made from gene symbols on an earlier annotation: 57 of its 247 IDs do
not exist in release 115, 39 more carry another symbol there, and it listed the ten S6-kinase genes as ribosomal. As
a result 47 to 116 ribosomal protein genes stayed in each tissue's deconvolution.

`--check` prints how the built list differs from the file at --out and exits non-zero if they differ.

Writes deconvolution/reference/rat_exclude_genes.tsv   (feature_ID, gene_symbol, category)

Usage (project venv):  python deconvolution/build_exclude_genes.py [--check] [--out PATH]
"""
import argparse
import os
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
os.environ.setdefault("PIPELINE_ROOT", str(Path(__file__).resolve().parents[1]))
from gene_utils import load_config, resolve_path  # noqa: E402

ID_COL, SYM_COL, TYPE_COL, CHR_COL, DESC_COL = ("Gene stable ID", "Gene name", "Gene type",
                                                  "Chromosome/scaffold name", "Gene description")


RIBOSOMAL = re.compile(r"ribosomal (?:subunit )?protein")     # also the newer "large ribosomal subunit protein uL14"


def category(description, chrom):
    """The exclusion category of one gene, or None if it is kept."""
    d = (description or "").lower()
    if chrom == "MT":
        return "mito"
    if RIBOSOMAL.search(d) and "kinase" not in d:
        return "ribosomal"
    if d.startswith("hemoglobin"):
        return "hemoglobin"
    return None


def build(gene_info):
    df = pd.read_csv(gene_info, sep="\t", dtype=str).fillna("")
    df.columns = [c.strip() for c in df.columns]
    df["category"] = [category(d, c) for d, c in zip(df[DESC_COL], df[CHR_COL])]
    out = df[df.category.notna()][[ID_COL, SYM_COL, "category"]].drop_duplicates(ID_COL)
    out.columns = ["feature_ID", "gene_symbol", "category"]
    return out.sort_values("feature_ID").reset_index(drop=True)


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(resolve_path(cfg, cfg["deconvolution"]["rat_exclude_genes"])))
    ap.add_argument("--check", action="store_true", help="compare with the file at --out instead of writing it")
    a = ap.parse_args()
    new = build(resolve_path(cfg, cfg["biomart"]["rat_gene_info"]))
    print(new.category.value_counts().to_string())
    if a.check:
        old = pd.read_csv(a.out, sep="\t", dtype=str).fillna("")
        o, n = dict(zip(old.feature_ID, old.category)), dict(zip(new.feature_ID, new.category))
        only_old = sorted(set(o) - set(n)); only_new = sorted(set(n) - set(o))
        changed = sorted(g for g in set(o) & set(n) if o[g] != n[g])
        sym = dict(zip(old.feature_ID, old.gene_symbol)) | dict(zip(new.feature_ID, new.gene_symbol))
        print(f"committed {len(o)} genes, built {len(n)}")
        print(f"only in the committed list ({len(only_old)}): {[sym[g] for g in only_old]}")
        print(f"only in the built list ({len(only_new)}): {[sym[g] for g in only_new]}")
        print(f"category differs ({len(changed)}): {[sym[g] for g in changed]}")
        sys.exit(0 if not (only_old or only_new or changed) else 1)
    new.to_csv(a.out, sep="\t", index=False)
    print(f"wrote {a.out} ({len(new)} genes)")


if __name__ == "__main__":
    main()
