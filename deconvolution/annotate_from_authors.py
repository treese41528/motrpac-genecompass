#!/usr/bin/env python3
"""annotate_from_authors.py -- re-annotate a reference's Leiden clusters using the SOURCE PAPER's own
cell-type scheme (the generalization of the TESTES paper-panel fix to the other tissues).

Most references were annotated by a generic PanglaoDB+ScType consensus, which (a) over-splits into
collinear near-duplicate labels and (b) mislabels tissue-specific types (e.g. BAT's brown-adipocyte
'adipocyte_2' subtype -- Ptprs/Kcnn3/Ca3 -- gets called Schwann/neurons). This scores each existing
Leiden cluster against the PAPER's rat-native marker sets and rewrites consensus.tsv (cluster ->
authors' label), which is what build_reference.py reads. celltypes.tsv (barcode->leiden) is untouched,
so this isolates the ANNOTATION change; rebuild the reference afterwards to apply it.

Design rules encoded in the panels (from the reference-paper review):
  - Keep the dominant parenchyma as ONE label (never split collinear adipocyte/myofiber/hepatocyte subtypes).
  - Keep immune subtypes resolved (omnideconv: coarse immune hurts BayesPrism).
  - A cluster whose best score is below MIN_SCORE, or which matches no panel type, -> 'Unknown' (dropped by clean_cells).

Usage: python deconvolution/annotate_from_authors.py --tissue BAT [--dry-run]
"""
import argparse, csv, os, sys
from pathlib import Path
import numpy as np, pandas as pd, scanpy as sc, yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("PIPELINE_ROOT", str(Path(__file__).resolve().parents[1]))
import build_reference as br

CONFIG = br.PROJECT / "deconvolution" / "tissue_references.yaml"
GENE_UNIVERSE = br.PROJECT / "data/training/gene_universe/gene_universe.tsv"
MIN_SCORE = 0.0   # cluster's top score must exceed this (score_genes is background-subtracted, so ~0 = no signal)

# ---- per-tissue author panels (paper marker sets). cell_type -> [rat gene symbols] ----
# Sources: the reference-paper review (authors' Fig/Table markers) + canonical rat markers.
PANELS = {
  "BAT": {  # Thompson 2024 (GSE244451): keep adipocytes ONE (incl adipocyte_2 neural-marker subtype); drop mesothelial
    "Brown adipocytes":        ["Ucp1","Cidea","Adipoq","Plin1","Plin5","Fabp4","Pparg","Cfd","Lep","Ptprs","Kcnn3","Ca3"],
    "Adipose stem/progenitor": ["Dcn","Fbn1","Cd34","Pdgfra","Pi16","Bmper","Gdf10","Col1a1","Col3a1"],
    "Endothelial cells":       ["Rasip1","Pecam1","Cdh5","Vwf","Cldn5","Flt1","Egfl7","Kdr"],
    "Smooth muscle/pericyte":  ["Myh11","Acta2","Tagln","Pdgfrb","Rgs5","Cnn1","Des","Notch3"],
    "Macrophages":             ["Cd68","Adgre1","Lyz2","C1qa","C1qb","Csf1r","Mrc1"],
    "T cells":                 ["Cd3e","Cd3g","Cd3d","Cd8a","Cd4","Themis"],
    "B cells":                 ["Ms4a1","Cd79a","Cd79b","Cd19","Ighm"],
    "NK cells":                ["Nkg7","Klrb1c","Ncr1","Gzmb","Prf1"],
  },
}


def sym_map(a):
    m = {}
    col = "gene_symbols" if "gene_symbols" in a.var else ("symbol" if "symbol" in a.var else None)
    if col:
        for e, s in zip(a.var_names, a.var[col]):
            m.setdefault(str(s), str(e))
    if GENE_UNIVERSE.exists():
        gu = pd.read_csv(GENE_UNIVERSE, sep="\t", dtype=str)
        for e, s in zip(gu["ensembl_id"], gu["symbol"]):
            m.setdefault(str(s), str(e))
    return m


def annotate_sample(sample, panel, dry):
    a = sc.read_h5ad(br.QC_DIR / f"{sample}.h5ad")
    ct = pd.read_csv(br.CT_DIR / sample / f"{sample}_celltypes.tsv", sep="\t", dtype={"leiden": str})
    lei = dict(zip(ct["barcode"].astype(str), ct["leiden"].astype(str)))
    a.obs["leiden"] = [lei.get(b, "NA") for b in a.obs_names.astype(str)]
    a = a[a.obs["leiden"] != "NA"].copy()
    s2e = sym_map(a)
    sc.pp.normalize_total(a, target_sum=1e4); sc.pp.log1p(a)
    classes = list(panel)
    for c in classes:
        ens = [s2e[g] for g in panel[c] if g in s2e and s2e[g] in a.var_names]
        sc.tl.score_genes(a, gene_list=ens, score_name="S_" + c, use_raw=False) if len(ens) >= 2 else a.obs.__setitem__("S_" + c, -9.0)
    scols = ["S_" + c for c in classes]
    cl_mean = a.obs.groupby("leiden")[scols].mean()
    rows = []
    for cl, row in cl_mean.iterrows():
        j = int(np.argmax(row.values)); best = float(row.values[j])
        lab = classes[j] if best > MIN_SCORE else "Unknown"
        n = int((a.obs["leiden"] == cl).sum())
        rows.append((cl, n, lab, best))
    df = pd.DataFrame(rows, columns=["cluster", "n_cells", "consensus_label", "score"]).sort_values("n_cells", ascending=False)
    print(f"\n  {sample}: {len(df)} clusters ->")
    for _, r in df.iterrows():
        print(f"     c{r['cluster']:>3} n={r['n_cells']:6d}  {r['consensus_label']:26s} (score {r['score']:.2f})")
    if not dry:
        out = br.CONS_DIR / sample; out.mkdir(parents=True, exist_ok=True)
        cons = pd.DataFrame({"cluster": df["cluster"], "n_cells": df["n_cells"],
                             "panglao_label": df["consensus_label"], "sctype_label": df["consensus_label"],
                             "consensus_label": df["consensus_label"], "consensus_source": "authors_panel"})
        cons.to_csv(out / f"{sample}_consensus.tsv", sep="\t", index=False)
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tissue", required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if args.tissue not in PANELS:
        sys.exit(f"ERROR: no author panel defined for {args.tissue}; have: {sorted(PANELS)}")
    cfg = {k: v for k, v in yaml.safe_load(open(CONFIG)).items() if isinstance(v, dict)}[args.tissue]
    samples = cfg.get("sample_ids") or []
    if not samples:  # take all in-corpus samples of the study/tissue
        samples = br.select_samples(cfg["study"], cfg["tissue"], organism=cfg.get("organism", "Rattus norvegicus"))
    print(f"=== re-annotating {args.tissue} ({len(samples)} samples) with the authors' panel ({len(PANELS[args.tissue])} types) ===")
    for s in samples:
        annotate_sample(s, PANELS[args.tissue], args.dry_run)
    if not args.dry_run:
        print(f"\nwrote consensus -> {br.CONS_DIR}/<sample>/ ; rebuild: "
              f"python deconvolution/build_reference.py --study {cfg['study']} --tissue \"{cfg['tissue']}\" "
              f"{'--sample-ids '+','.join(samples) if cfg.get('sample_ids') else ''} --out {cfg['reference_dir']}")


if __name__ == "__main__":
    main()
