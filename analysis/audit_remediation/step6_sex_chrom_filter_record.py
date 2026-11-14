#!/usr/bin/env python3
"""step6_sex_chrom_filter_record.py -- Step 6 (records, tests, hygiene), audit finding 5.8.

Stage 8 removes every gene on chromosome X or Y from each reference before BayesPrism runs
(deconvolution.exclude_sex_chromosomes, default true; deconvolution/R/run_deconvolution.R). That mirrors
BayesPrism's own cleanup for human and mouse. Its cost is that no X-linked gene can appear in an expression
table, a pseudo-cell or the differential expression, whether or not its expression differs by sex. This
script counts what the committed gene lists say is removed. It changes nothing.

  python analysis/audit_remediation/step6_sex_chrom_filter_record.py
Out: docs/audit/remediation/record/step6/sex_chrom_filter_genes.tsv
"""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REF = ROOT / "deconvolution" / "reference"
OUT = ROOT / "docs/audit/remediation/record/step6/sex_chrom_filter_genes.tsv"


def read(name):
    with open(REF / name, newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def main():
    sex = read("rat_sex_chrom_genes.tsv")
    coding = {r["feature_ID"] for r in read("rat_protein_coding_genes.tsv")}
    vocab = {r["feature_ID"] for r in read("rat_genecompass_genes.tsv")}
    rows = []
    for chrom in ("X", "Y"):
        ids = {r["feature_ID"] for r in sex if r["chromosome"] == chrom}
        rows.append({"chromosome": chrom, "genes_in_list": len(ids),
                     "protein_coding": len(ids & coding),
                     "in_genecompass_vocabulary": len(ids & vocab),
                     "protein_coding_and_in_vocabulary": len(ids & coding & vocab)})
    rows.append({"chromosome": "X + Y", **{k: sum(r[k] for r in rows) for k in rows[0] if k != "chromosome"}})
    rows.append({"chromosome": "all chromosomes (the two lists)", "genes_in_list": "",
                 "protein_coding": len(coding), "in_genecompass_vocabulary": len(vocab),
                 "protein_coding_and_in_vocabulary": len(coding & vocab)})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        w.writeheader(); w.writerows(rows)
    for r in rows:
        print("\t".join(str(v) for v in r.values()))
    other = sorted({r["chromosome"] for r in sex} - {"X", "Y"})
    print(f"chromosome labels in the list other than X and Y: {other or 'none'}")
    print(f"sources: {REF.relative_to(ROOT)}/rat_sex_chrom_genes.tsv, rat_protein_coding_genes.tsv, "
          f"rat_genecompass_genes.tsv\nwrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
