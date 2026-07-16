#!/usr/bin/env python3
"""annotate_omix767_testes.py -- paper-panel, per-cell cell-type annotation for OMIX767 rat testis.

PHASE 2 of TESTES ingestion (after ingest_omix767_testes.py builds the qc h5ads). The generic
PanglaoDB / ScType panels LACK the spermatogenesis stages -- they have no Spermatogonia and no Spermatid
classes, only {Germ cells, Spermatocytes, Spermatozoa} -- so automated annotation collapses the whole
germ lineage into one "Germ cells" bucket. This uses the SOURCE PAPER's own scheme instead
(Guan et al., Sci Data 2022, PMID 35338159): the authors' ranked DEG lists per cell type
(OMIX767-20-02.xlsx, "DEG list for 7 cell types") plus their rat-VALIDATED text markers
(Dazl/Sohlh2/Elavl3 for spermatogonia -- NOT the mouse Stra8/Zbtb16/Gfra1 the paper explicitly reports
do NOT work in rat). Cells are scored PER CELL against each class, with a MARKER-CONSISTENCY GATE on the
sparse somatic types: a cell may only be called a somatic type if it actually expresses that type's
defining marker; otherwise it falls back to its best germ class. This fixes the immune false-positive
that plain per-cell argmax produces (immune is near-absent, so argmax over-assigns it).

Result: full germ trajectory SPG->SPC->RSPT->ESPT->CSPT resolves, and scattered Sertoli/Leydig are
recovered and validated (72-91% Sox9+ / 67-100% Cyp17a1+). OMIX767's somatic compartment is genuinely
THIN (Sertoli ~1%, Leydig ~0.2%, immune/endothelial near-absent) so absolute somatic fractions stay
untrustworthy and thin states (<min-state-cells) are dropped by build_reference.clean_cells -- germ-stage
resolution is the strength. (GSE268104 snRNA is the alternative if trustworthy somatic fractions matter.)

Writes, per sample, in the corpus annotation format so build_reference.py can consume it:
  CT_DIR/<sample>/<sample>_celltypes.tsv   barcode, leiden(=class code), cell_type, cell_type_confidence
  CONS_DIR/<sample>/<sample>_consensus.tsv cluster(=code), n_cells, panglao_label, sctype_label,
                                           consensus_label, consensus_source
Then: build_reference.py --study OMIX767 --tissue testis --sample-ids OMIX767_sampleC,OMIX767_sampleE7W

Usage (project venv): python deconvolution/annotate_omix767_testes.py [--top-n 50] [--samples ...]
"""
import argparse, os, sys
from pathlib import Path
import numpy as np, pandas as pd, scanpy as sc, scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("PIPELINE_ROOT", str(Path(__file__).resolve().parents[1]))
import build_reference as br  # QC_DIR / CT_DIR / CONS_DIR

DEG_XLSX = br.PROJECT / "data/raw/ngdc/datasets/OMIX767/OMIX767-20-02.xlsx"  # authors' 7-cell-type DEG list
SAMPLES = ["OMIX767_sampleC", "OMIX767_sampleE7W"]

# authors' DEG columns (Sheet1) -> full class name + short code (code = build_reference cell.state key)
CLASSES = {  # tag: (full_name, code, is_germ)
    "SPG":  ("Spermatogonia", "SPG", True),   "SPC":  ("Spermatocytes", "SPC", True),
    "RSPT": ("Round spermatids", "RSPT", True), "ESPT": ("Elongating spermatids", "ESPT", True),
    "CSPT": ("Condensed spermatids", "CSPT", True),
    "SC":   ("Sertoli cells", "Sertoli", False), "LC": ("Leydig cells", "Leydig", False),
}
GERM = [v[0] for v in CLASSES.values() if v[2]]
# rat-validated canonical markers (paper text) to reinforce each authors' DEG set
CANON = {
    "Spermatogonia": ["Dazl", "Sohlh2", "Elavl3"], "Spermatocytes": ["Phf2", "Id1", "Ngfr", "Sycp3"],
    "Round spermatids": ["Lrat", "Spag6"], "Elongating spermatids": ["Tnp1", "Tnp2", "Prm1", "Prm2"],
    "Condensed spermatids": ["Prm1", "Prm2", "Tnp2"],
    "Sertoli cells": ["Clu", "Gstm6", "Sox9", "Amh", "Wt1", "Rhox5", "Ctsl"],
    "Leydig cells": ["Cyp17a1", "Cyp11a1", "Insl3", "Star", "Hsd3b1", "Nr5a1"],
}
# extra somatic classes not in the paper's 7 (for per-cell recovery), canonical markers only
EXTRA = {
    "Immune cells":      ("Immune", ["Ptprc", "Cd68", "Adgre1", "Lyz2", "C1qa", "C1qb", "Cd14", "Aif1", "Csf1r"]),
    "Endothelial cells": ("Endo",   ["Pecam1", "Vwf", "Cdh5", "Flt1", "Egfl7", "Kdr", "Tek", "Cldn5"]),
    "Peritubular myoid": ("Myoid",  ["Acta2", "Myh11", "Des", "Tagln", "Cnn1", "Myl9", "Pdgfrb"]),
}
# GATE: a cell may be called this somatic type only if it expresses its DEFINING pan-marker.
# These markers are NOT part of the germ ambient (unlike Prm1/Tnp1), so >0 is discriminative -- but each
# must be the single specific pan-marker (a permissive set incl. Cd68/Csf1r lets ambient pass -> immune FP).
GATE = {
    "Sertoli cells": ["Sox9"], "Leydig cells": ["Cyp17a1", "Cyp11a1"],
    "Immune cells": ["Ptprc"], "Endothelial cells": ["Cdh5", "Vwf"], "Peritubular myoid": ["Myh11"],
}
CODE = {v[0]: v[1] for v in CLASSES.values()} | {k: v[0] for k, v in EXTRA.items()}


def build_panel(top_n):
    s1 = pd.read_excel(DEG_XLSX, sheet_name="Sheet1")
    panel = {}
    for tag, (name, _c, _g) in CLASSES.items():
        d = s1[[f"{tag} (DEGs)", f"{tag} (Log2 Fold Change)"]].dropna()
        d.columns = ["gene", "lfc"]
        d = d[d["lfc"] > 0].sort_values("lfc", ascending=False)
        panel[name] = [str(g).strip() for g in d["gene"].head(top_n)]
    for name, g in CANON.items():
        panel[name] = list(dict.fromkeys(panel.get(name, []) + g))
    for name, (_code, mk) in EXTRA.items():
        panel[name] = list(mk)
    return panel


def annotate(sample, panel, min_markers=3):
    a = sc.read_h5ad(br.QC_DIR / f"{sample}.h5ad")
    sym2ens = {}
    for ens, sym in zip(a.var_names, a.var["symbol"]):
        sym2ens.setdefault(str(sym), str(ens))
    X = sp.csr_matrix(a.X)                                   # raw counts (for the gate)
    def expressed(genes):
        cols = [a.var_names.get_loc(sym2ens[g]) for g in genes if g in sym2ens]
        if not cols:
            return np.zeros(a.n_obs, bool)
        return (np.asarray(X[:, cols].sum(1)).ravel() > 0)
    sc.pp.normalize_total(a, target_sum=1e4); sc.pp.log1p(a)
    pn = {n: [sym2ens[g] for g in gs if g in sym2ens] for n, gs in panel.items()}
    pn = {n: v for n, v in pn.items() if len(v) >= min_markers}
    for n, v in pn.items():
        sc.tl.score_genes(a, gene_list=v, score_name="S_" + n, use_raw=False)
    classes = list(pn); scols = ["S_" + n for n in classes]
    S = a.obs[scols].values
    top = S.argmax(1)
    lab = np.array([classes[i] for i in top], dtype=object)
    # --- marker-consistency gate: a somatic call must express the type's gate marker, else -> best germ ---
    germ_idx = [classes.index(g) for g in GERM if g in classes]
    n_gated = 0
    for name, mks in GATE.items():
        if name not in classes:
            continue
        is_name = lab == name
        if not is_name.any():
            continue
        expr = expressed(mks)
        bad = is_name & ~expr
        if bad.any():
            gs = S[:, germ_idx]
            lab[bad] = np.array([GERM[j] for j in gs[bad].argmax(1)], dtype=object)
            n_gated += int(bad.sum())
    conf = S[np.arange(len(top)), [classes.index(l) for l in lab]]
    a.obs["cell_type"] = lab
    a.obs["code"] = [CODE.get(l, l) for l in lab]
    a.obs["conf"] = conf
    return a, n_gated


def write_outputs(sample, a):
    ct_dir = br.CT_DIR / sample; cons_dir = br.CONS_DIR / sample
    ct_dir.mkdir(parents=True, exist_ok=True); cons_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"barcode": a.obs_names.astype(str), "leiden": a.obs["code"].values,
                       "cell_type": a.obs["cell_type"].values, "cell_type_confidence": a.obs["conf"].values})
    df.to_csv(ct_dir / f"{sample}_celltypes.tsv", sep="\t", index=False)
    # consensus: one row per class code present (build_reference maps cluster->consensus_label)
    g = df.groupby(["leiden", "cell_type"]).size().reset_index(name="n_cells")
    cons = pd.DataFrame({"cluster": g["leiden"], "n_cells": g["n_cells"],
                         "panglao_label": g["cell_type"], "sctype_label": g["cell_type"],
                         "consensus_label": g["cell_type"], "consensus_source": "paper_panel_percell_gated"})
    cons.to_csv(cons_dir / f"{sample}_consensus.tsv", sep="\t", index=False)
    return df["cell_type"].value_counts()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--top-n", type=int, default=50, help="top DEGs per class from the authors' list")
    ap.add_argument("--samples", nargs="*", default=SAMPLES)
    args = ap.parse_args()
    if not DEG_XLSX.exists():
        sys.exit(f"ERROR: authors' DEG list not found: {DEG_XLSX}")
    panel = build_panel(args.top_n)
    print(f"paper panel: {len(panel)} classes; sizes " + ", ".join(f"{n}={len(v)}" for n, v in panel.items()))
    for s in args.samples:
        a, n_gated = annotate(s, panel)
        vc = write_outputs(s, a)
        print(f"\n=== {s}: {len(vc)} cell types (gate reassigned {n_gated} cells to germ) ===")
        print(vc.to_string())
    print(f"\nwrote celltypes -> {br.CT_DIR}/<sample>/ ; consensus -> {br.CONS_DIR}/<sample>/")
    print("next: python deconvolution/build_reference.py --study OMIX767 --tissue testis "
          "--sample-ids OMIX767_sampleC,OMIX767_sampleE7W --out data/deconvolution/references_v3/TESTES_OMIX767")


if __name__ == "__main__":
    main()
