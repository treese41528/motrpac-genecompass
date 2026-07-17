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
  "HEART": {
    # SCP2828 ratmap_scp.h5ad (GSE280111, Nagai/Ellinor Cell Reports 2024) is 13 GB -> read obs via h5py
    # (never load X). Authors barcodes are '<sample>::<cellbc>-1'; our corpus barcodes are the bare '<cellbc>-1',
    # so we auto-match each of our GSE280111 samples to the authors' sample by core-barcode-set overlap
    # (verified 1:1: our 19 HEART samples <-> the 19 LV samples at 74-92% overlap). Keep only tissue==LV.
    "authors_h5ad": "data/raw/geo/GSE280111/ratmap_scp.h5ad",
    "source": "Single Cell Portal SCP2828 ratmap_scp.h5ad (GSE280111, Cell Reports 2024, Ellinor lab)",
    "read": "h5py_obs",
    "label_col": "cell_type__ontology_label",
    "tissue_col": "tissue", "tissue_val": "LV",
    "sample_col": "sample",
    "bc_core_split": "::", "bc_core_take": 1,   # authors core barcode is AFTER '::'
    "our_bc_split": None,                        # our barcodes used as-is
    "automatch": True, "min_overlap_frac": 0.30,
    # merge the 33 fine ontology labels -> a clean cardiac LV basis (immune kept resolved; structural collapsed;
    # trace mislabels/contaminants -> Unknown -> dropped). Labels not listed are kept as-is.
    "merge": {
      "cardiac ventricle fibroblast": "Cardiac fibroblasts",
      "cardiac atrium fibroblast": "Cardiac fibroblasts",
      "fibroblast": "Cardiac fibroblasts",
      "fibroblast of cardiac tissue": "Cardiac fibroblasts",
      "cardiac endothelial cell": "Endothelial cells",
      "endothelial cell": "Endothelial cells",
      "endothelial cell of lymphatic vessel": "Lymphatic endothelial cells",
      "regular ventricular cardiac myocyte": "Cardiomyocytes",
      "regular atrial cardiac myocyte": "Cardiomyocytes",
      "pericyte": "Pericytes",
      "vascular associated smooth muscle cell": "Vascular smooth muscle cells",
      "macrophage": "Macrophages",
      "monocyte": "Monocytes",
      "CD14-low, CD16-positive monocyte": "Monocytes",
      "dendritic cell": "Dendritic cells",
      "T cell": "T cells",
      "cycling T cell": "T cells",
      "CD8-positive, alpha-beta T cell": "CD8+ T cells",
      "natural killer cell": "NK cells",
      "B cell": "B cells",
      "mast cell": "Mast cells",
      "cardiac neuron": "Cardiac neurons",
      "mesothelial cell of epicardium": "Mesothelial cells",
      "white adipocyte": "Adipocytes", "brown adipocyte": "Adipocytes",
      "Schwann cell": "Unknown", "peripheral nervous system neuron": "Unknown", "club cell": "Unknown",
    },
  },
}


def _code(lab, reg):
    if lab not in reg:
        reg[lab] = "".join(w[0] for w in lab.replace("&", "").split())[:6].upper() + str(len(reg))
    return reg[lab]


def _write_sample(ours_s, bc, lab, reg, source, dry):
    matched = sum(1 for x in lab if x != "Unknown")
    df = pd.DataFrame({"barcode": pd.Series(bc).astype(str),
                       "leiden": [_code(x, reg) for x in lab],
                       "cell_type": lab, "cell_type_confidence": 1.0})
    vc = df["cell_type"].value_counts()
    print(f"  {ours_s}: {matched}/{len(bc)} matched | "
          + ", ".join(f"{k}={v}" for k, v in vc.items() if k != "Unknown"))
    if dry:
        return
    ctd = br.CT_DIR / ours_s; cod = br.CONS_DIR / ours_s
    ctd.mkdir(parents=True, exist_ok=True); cod.mkdir(parents=True, exist_ok=True)
    df.to_csv(ctd / f"{ours_s}_celltypes.tsv", sep="\t", index=False)
    g = df.groupby(["leiden", "cell_type"]).size().reset_index(name="n_cells")
    pd.DataFrame({"cluster": g["leiden"], "n_cells": g["n_cells"], "panglao_label": g["cell_type"],
                 "sctype_label": g["cell_type"], "consensus_label": g["cell_type"],
                 "consensus_source": "authors_deposited:" + source.split("(")[0].strip()}
                ).to_csv(cod / f"{ours_s}_consensus.tsv", sep="\t", index=False)


def _read_obs_h5py(spec):
    """Read ONLY the obs group of a (possibly huge) deposited h5ad via h5py -> DataFrame(core, authors_sample,
    label), filtered to the target tissue. Never touches X."""
    import h5py
    import numpy as np
    path = br.PROJECT / spec["authors_h5ad"]
    with h5py.File(path, "r") as f:
        obs = f["obs"]

        def cat(name):
            g = obs[name]
            if isinstance(g, h5py.Group) and "categories" in g:
                cats = [c.decode() if isinstance(c, bytes) else str(c) for c in g["categories"][:]]
                codes = g["codes"][:]
                return np.array([cats[i] if i >= 0 else "NA" for i in codes], dtype=object)
            arr = g[:]
            return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in arr], dtype=object)

        idx = obs["_index"][:] if "_index" in obs else obs["index"][:]
        bc = np.array([b.decode() if isinstance(b, bytes) else str(b) for b in idx], dtype=object)
        take = spec.get("bc_core_take", 0)
        core = np.array([x.split(spec["bc_core_split"])[take] if spec["bc_core_split"] in x else x
                         for x in bc], dtype=object)
        df = pd.DataFrame({"core": core,
                           "authors_sample": cat(spec["sample_col"]),
                           "tissue": cat(spec["tissue_col"]),
                           "label": cat(spec["label_col"])})
    if spec.get("tissue_col"):
        df = df[df["tissue"] == spec["tissue_val"]]
    return df


def adopt(tissue, dry):
    spec = DEPOSITED[tissue]
    merge = spec.get("merge", {}); reg = {}
    src = spec["source"]

    if spec.get("read") == "h5py_obs":                       # big deposited h5ad, auto-match by barcode overlap
        import yaml
        cfg = yaml.safe_load(open(br.PROJECT / "deconvolution" / "tissue_references.yaml"))
        our_samples = cfg[tissue]["expect"]["sample_ids"]
        obs = _read_obs_h5py(spec)
        au_sets = {s: set(g["core"]) for s, g in obs.groupby("authors_sample")}
        au_lab = {s: dict(zip(g["core"], g["label"])) for s, g in obs.groupby("authors_sample")}
        for ours_s in our_samples:
            a = sc.read_h5ad(br.QC_DIR / f"{ours_s}.h5ad", backed="r")
            bc = [str(x) for x in a.obs_names]
            core = [x.split(spec["our_bc_split"])[0] for x in bc] if spec.get("our_bc_split") else bc
            cset = set(core)
            best = max(au_sets, key=lambda s: len(cset & au_sets[s]))
            frac = len(cset & au_sets[best]) / max(1, len(cset))
            if frac < spec.get("min_overlap_frac", 0.3):
                print(f"  {ours_s}: WEAK best-overlap {frac:.0%} (authors {best}); labelling all Unknown");
            m = au_lab[best]
            lab = [merge.get(m.get(c), m.get(c, "Unknown")) if m.get(c) is not None else "Unknown" for c in core]
            print(f"    (authors {best} [{obs[obs.authors_sample==best].tissue.iloc[0]}], {frac:.0%} overlap)")
            _write_sample(ours_s, bc, lab, reg, src, dry)
        return

    au = ad.read_h5ad(br.PROJECT / spec["authors_h5ad"], backed="r")   # BAT-style: small h5ad + explicit map
    o = au.obs.copy()
    o["core"] = au.obs_names.astype(str).str.split(spec["bc_core_split"]).str[0]
    if spec.get("tissue_col"):
        o = o[o[spec["tissue_col"]] == spec["tissue_val"]]
    for ours_s, au_s in spec["sample_map"].items():
        a = sc.read_h5ad(br.QC_DIR / f"{ours_s}.h5ad")
        bc = pd.Series(a.obs_names.astype(str)); core = bc.str.split(spec["bc_core_split"]).str[0]
        sub = o[o[spec["sample_col"]].astype(str) == str(au_s)]
        m = dict(zip(sub["core"], sub[spec["celltype_col"]].astype(str)))
        lab = [merge.get(m.get(c), m.get(c, "Unknown")) if m.get(c) is not None else "Unknown" for c in core]
        _write_sample(ours_s, bc, lab, reg, src, dry)


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
