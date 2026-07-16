#!/usr/bin/env python3
"""adopt_authors_labels.py -- adopt the SOURCE PAPER's DEPOSITED per-cell cell-type labels for a
reference, by barcode-joining the authors' annotated object onto our corpus h5ads.

This is route (A) -- the gold-standard generalization of the TESTES fix. The generic PanglaoDB+ScType
consensus over-splits into collinear duplicates and mislabels tissue-specific types; and a hand-built
marker panel FAILS for dominant-parenchyma tissues (the parenchyma ambient inflates its score in every
cluster -- e.g. BAT collapses to all-adipocyte). The robust fix is to use the authors' OWN per-cell
annotations where they deposited them (GEO supplement / Single Cell Portal / OSF / GitHub), joined by
cell barcode. Cells the authors QC'd out (no label) -> 'Unknown' -> dropped by build_reference.clean_cells,
so we also inherit the authors' stricter QC. Writes celltypes.tsv + consensus.tsv (authors' label as a
per-cell pseudo-cluster) so build_reference.py consumes it directly; celltypes leiden is unused downstream.

Per tissue we record: the deposited object, its cell-type column, an optional tissue/sample filter, the
map from OUR corpus sample_id -> the authors' sample value, the barcode-core rule, and a MERGE map that
COLLAPSES collinear dominant-parenchyma subtypes into one basis label (never split them for BayesPrism).

Usage: python deconvolution/adopt_authors_labels.py --tissue BAT [--dry-run]
Then rebuild: python deconvolution/build_reference.py --study ... --tissue ... --sample-ids ... --out ...
"""
import argparse, os, sys
from pathlib import Path
import anndata as ad, scanpy as sc, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("PIPELINE_ROOT", str(Path(__file__).resolve().parents[1]))
import build_reference as br

# tissue -> deposited-label spec
DEPOSITED = {
  "BAT": {
    "authors_h5ad": "data/raw/geo/GSE244451/GSE244451_PVAT_BAT_SCPAnnotated.h5ad",
    "source": "GEO GSE244451 GSE244451_PVAT_BAT_SCPAnnotated.h5ad (Thompson et al. 2024, Single Cell Portal SCP2384)",
    "celltype_col": "celltype",          # authors' 8-type annotation
    "tissue_col": "tissue", "tissue_val": "BAT",   # keep BAT (drop the PVAT half of the object)
    "sample_col": "Sample",
    "sample_map": {"GSE244451_sample1": "1", "GSE244451_sample3": "3"},  # our sample_id -> authors Sample
    "bc_core_split": "-",                # authors bc = SEQ-1-<sampleidx>; core = SEQ (before first '-')
    # collapse the two collinear brown-adipocyte subtypes into ONE basis type (anti-over-split rule)
    "merge": {"Adipocytes": "Brown adipocytes", "Adipocytes_2": "Brown adipocytes"},
  },
}


def _code(lab, reg):
    if lab not in reg:
        reg[lab] = "".join(w[0] for w in lab.replace("&", "").split())[:6].upper() + str(len(reg))
    return reg[lab]


def adopt(tissue, dry):
    spec = DEPOSITED[tissue]
    au = ad.read_h5ad(br.PROJECT / spec["authors_h5ad"], backed="r")
    o = au.obs.copy()
    o["core"] = au.obs_names.astype(str).str.split(spec["bc_core_split"]).str[0]
    if spec.get("tissue_col"):
        o = o[o[spec["tissue_col"]] == spec["tissue_val"]]
    merge = spec.get("merge", {}); reg = {}
    for ours_s, au_s in spec["sample_map"].items():
        a = sc.read_h5ad(br.QC_DIR / f"{ours_s}.h5ad")
        bc = pd.Series(a.obs_names.astype(str)); core = bc.str.split(spec["bc_core_split"]).str[0]
        sub = o[o[spec["sample_col"]].astype(str) == str(au_s)]
        m = dict(zip(sub["core"], sub[spec["celltype_col"]].astype(str)))
        lab = [merge.get(m.get(c), m.get(c, "Unknown")) if m.get(c) is not None else "Unknown" for c in core]
        matched = sum(1 for x in lab if x != "Unknown")
        df = pd.DataFrame({"barcode": bc, "leiden": [_code(x, reg) for x in lab],
                           "cell_type": lab, "cell_type_confidence": 1.0})
        vc = df["cell_type"].value_counts()
        print(f"  {ours_s} <- authors Sample {au_s}: {matched}/{len(bc)} matched | "
              + ", ".join(f"{k}={v}" for k, v in vc.items() if k != "Unknown"))
        if dry:
            continue
        ctd = br.CT_DIR / ours_s; cod = br.CONS_DIR / ours_s
        ctd.mkdir(parents=True, exist_ok=True); cod.mkdir(parents=True, exist_ok=True)
        df.to_csv(ctd / f"{ours_s}_celltypes.tsv", sep="\t", index=False)
        g = df.groupby(["leiden", "cell_type"]).size().reset_index(name="n_cells")
        pd.DataFrame({"cluster": g["leiden"], "n_cells": g["n_cells"], "panglao_label": g["cell_type"],
                     "sctype_label": g["cell_type"], "consensus_label": g["cell_type"],
                     "consensus_source": "authors_deposited:" + spec["source"].split("(")[0].strip()}
                    ).to_csv(cod / f"{ours_s}_consensus.tsv", sep="\t", index=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tissue", required=True, choices=sorted(DEPOSITED))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    print(f"=== adopting authors' deposited labels for {args.tissue} ({DEPOSITED[args.tissue]['source']}) ===")
    adopt(args.tissue, args.dry_run)
    if not args.dry_run:
        print(f"\nwrote celltypes/consensus for {args.tissue}. Rebuild the reference, then run_deconv_all --submit --tissue {args.tissue}")


if __name__ == "__main__":
    main()
