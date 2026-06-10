#!/usr/bin/env python3
"""
build_reference.py -- Assemble a BayesPrism single-cell reference from the
annotated rat SC corpus (Stage 8 deconvolution).

Paper-faithful: a single cohesive study, FULL cells with natural counts (no
balancing/subsampling), fine cell types + finer cell states. We do NOT pool
multiple studies into one reference.

Join: barcode -> leiden (per-cell) -> consensus_label (per-cluster)
      cell.type = consensus_label ; cell.state = "{sample}_c{leiden}"

Output (read by run_deconvolution.R; project-local, never /tmp):
  deconvolution/reference/{TISSUE}_{STUDY}/
    reference_counts.mtx   cells x genes integer (MatrixMarket)
    genes.tsv  genes (matrix columns) | cells_meta.tsv  per-cell labels | summary.txt

The loader/assembler/exporter are importable (make_pseudobulk.py reuses them).

Usage (project venv):
  python deconvolution/build_reference.py --study GSE220075 --tissue liver
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.sparse as sp
import anndata as ad
import scanpy as sc

# Paths are read from config/pipeline_config.yaml (deconvolution section), with an
# optional gitignored config/pipeline_config.local.yaml override. PIPELINE_ROOT is
# derived from this file's location so the module works from any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
os.environ.setdefault("PIPELINE_ROOT", str(Path(__file__).resolve().parents[1]))
from gene_utils import load_config, resolve_path  # noqa: E402

_CFG = load_config()
_DC = _CFG["deconvolution"]
PROJECT = _CFG["_project_root"]
QC_DIR = resolve_path(_CFG, _DC["qc_matrices_dir"])
CT_DIR = resolve_path(_CFG, _DC["cell_annotations_dir"])
CONS_DIR = resolve_path(_CFG, _DC["consensus_annotations_dir"])
INVENTORY = resolve_path(_CFG, _DC["annotation_inventory"])
UNKNOWN = {"Unknown", "unknown", "", "nan", "NA"}


# --- optional cell-type label canonicalization (collinear-fragment merging) ----
# BayesPrism cannot identify between collinear GEPs. Several rat brain atlases
# fragment ONE excitatory/pyramidal neuron population across many near-synonymous
# consensus labels ("CA3 pyramidal cells" vs "CA3 pyramidal neurons", "Pyramidal
# cells/neurons", "Principal neurons", "Cortical/Mature neurons", generic "Neurons").
# In cross-dataset deconvolution the source neuron mass then scatters across all of
# them and every bucket collapses to ~0 (reports/deconvolution/multitissue_validation.md
# V2). The "brain" scheme merges ONLY those collinear fragments into a few separable
# classes; biologically distinct, separable populations (GABAergic, granule,
# dopaminergic, cholinergic, Purkinje, astrocytes, ...) are kept as-is. It also folds
# two obvious glia synonym-duplicates. Applied IDENTICALLY to the reference and the
# cross source so harmonization stays an exact match.
_BRAIN_EXCIT_SUBSTR = ("pyramidal", "glutamat", "glutamin", "vglut", "principal")
_BRAIN_EXCIT_EXACT = {"neurons", "cortical neurons", "mature neurons", "hippocampal neurons"}


def _canon_brain(label):
    ll = str(label).strip().lower()
    if any(s in ll for s in _BRAIN_EXCIT_SUBSTR) or ll in _BRAIN_EXCIT_EXACT:
        return "Excitatory neurons"
    if ll in ("microglia", "microglial cells"):
        return "Microglia"
    if "oligodendrocyte precursor" in ll or "oligodendrocyte progenitor" in ll:
        return "Oligodendrocyte precursor cells"
    return label


LABEL_SCHEMES = {"brain": _canon_brain}


def canonicalize_labels(series, scheme):
    """Map fine consensus labels onto a coarser separable scheme (or identity)."""
    if not scheme or scheme == "none":
        return series
    fn = LABEL_SCHEMES.get(scheme)
    if fn is None:
        sys.exit(f"ERROR: unknown --label-scheme {scheme!r}; choices: {sorted(LABEL_SCHEMES)}")
    return series.map(fn)


def select_samples(study, tissue, conditions=None, sex=None, sample_ids=None):
    """In-corpus sample IDs for study+tissue, optionally filtered by
    condition_resolved and/or sex_resolved.

    sample_ids (explicit list) restricts to exactly those IDs -- needed when the
    healthy/disease (or genotype) split lives only in geo_title, not in
    condition_resolved (which is unreliable; e.g. GSE305314 WT-vs-Tau). The IDs
    are still validated against the study/tissue/in_corpus selection."""
    inv = pd.read_csv(INVENTORY, sep="\t", dtype=str)
    sel = inv[
        (inv["accession"] == study)
        & (inv["tissue_normalized"].str.lower() == tissue.lower())
        & (inv["in_corpus"].str.lower() == "true")
    ]
    if conditions:
        sel = sel[sel["condition_resolved"].isin(conditions)]
    if sex:
        sel = sel[sel["sex_resolved"].str.lower() == sex.lower()]
    if sample_ids:
        want = set(sample_ids)
        sel = sel[sel["sample_id"].isin(want)]
        got = set(sel["sample_id"])
        missing = want - got
        if missing:
            sys.exit(f"ERROR: --sample-ids not in corpus for {study}/{tissue}: "
                     f"{sorted(missing)}")
    samples = sorted(sel["sample_id"].tolist())
    if not samples:
        sys.exit(f"ERROR: no in-corpus {tissue} samples for {study} "
                 f"(conditions={conditions}, sex={sex}, sample_ids={sample_ids}) in {INVENTORY}")
    return samples


def load_sample(sample):
    """AnnData of raw counts over ENSRNOG genes, obs = barcode/sample/leiden/cell_type/cell_state."""
    h5 = QC_DIR / f"{sample}.h5ad"
    ct_tsv = CT_DIR / sample / f"{sample}_celltypes.tsv"
    cons_tsv = CONS_DIR / sample / f"{sample}_consensus.tsv"
    for p in (h5, ct_tsv, cons_tsv):
        if not p.exists():
            print(f"  [skip] {sample}: missing {p.name}")
            return None

    adata = sc.read_h5ad(h5)
    if adata.raw is not None:                       # raw integer counts
        counts, genes = adata.raw.X, np.asarray(adata.raw.var_names)
    else:
        counts, genes = adata.X, np.asarray(adata.var_names)
    counts = sp.csr_matrix(counts)
    if not np.all(counts.data == np.round(counts.data)):
        print(f"  [warn] {sample}: non-integer counts (not raw?)")

    cells = pd.read_csv(ct_tsv, sep="\t", dtype={"barcode": str, "leiden": str})
    cons = pd.read_csv(cons_tsv, sep="\t", dtype={"cluster": str})
    c2lab = dict(zip(cons["cluster"], cons["consensus_label"]))
    cells["cell_type"] = cells["leiden"].map(c2lab)
    cells["cell_state"] = sample + "_c" + cells["leiden"].astype(str)
    cells["sample"] = sample

    bc = pd.Index(adata.obs_names.astype(str))
    cells = cells.set_index("barcode").reindex(bc)
    miss = int(cells["cell_type"].isna().sum())
    if miss:
        print(f"  [warn] {sample}: {miss}/{len(bc)} cells lack annotation match")

    sub = ad.AnnData(X=counts, obs=cells.reset_index().rename(columns={"index": "barcode"}))
    sub.var_names = genes
    print(f"  [ok]  {sample}: {sub.n_obs} cells x {sub.n_vars} genes")
    return sub


def load_study(study, tissue, conditions=None, sex=None, sample_ids=None,
               gene_join="inner", min_gene_cells=0):
    """Concatenate all (loadable) samples of a study/tissue. gene_join='inner' keeps the
    shared (intersection) gene set -- the default, but it COLLAPSES studies with uneven
    per-sample gene depth (cortex GSE303115: per-sample 9.5k-21k -> 5.5k shared). Use
    gene_join='outer' to take the UNION (0-filled), optionally pruned by min_gene_cells
    (drop genes expressed in < that many pooled cells) to trim the union's long tail."""
    samples = select_samples(study, tissue, conditions, sex, sample_ids)
    print(f"{study} / {tissue}: {len(samples)} samples -> {samples}")
    parts = [s for s in (load_sample(x) for x in samples) if s is not None]
    if not parts:
        sys.exit("ERROR: no samples loaded.")
    adata = ad.concat(parts, join=gene_join, merge="same", fill_value=0)
    print(f"concatenated ({gene_join}): {adata.n_obs} cells x {adata.n_vars} genes")
    if min_gene_cells > 0:
        before = adata.n_vars
        sc.pp.filter_genes(adata, min_cells=min_gene_cells)
        print(f"min-gene-cells={min_gene_cells}: {before} -> {adata.n_vars} genes kept")
    return adata


def clean_cells(adata, min_state_cells=20, drop_unknown=True):
    """Drop unannotated/Unknown cells and cell states (clusters) with < min_state_cells."""
    keep = adata.obs["cell_type"].notna()
    if drop_unknown:
        keep &= ~adata.obs["cell_type"].isin(UNKNOWN)
    adata = adata[keep].copy()
    vc = adata.obs["cell_state"].value_counts()
    small = vc[vc < min_state_cells].index
    if len(small):
        print(f"dropping {len(small)} states <{min_state_cells} cells ({int(vc[small].sum())} cells)")
        adata = adata[~adata.obs["cell_state"].isin(small)].copy()
    return adata


def export_reference(adata, out_dir):
    """Write reference_counts.mtx (cells x genes int) + genes.tsv + cells_meta.tsv + summary.txt."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    X = sp.csr_matrix(adata.X)
    sio.mmwrite(str(out / "reference_counts.mtx"), X.astype(np.int32))
    pd.Series(adata.var_names).to_csv(out / "genes.tsv", index=False, header=False)
    adata.obs[["barcode", "sample", "leiden", "cell_type", "cell_state"]].to_csv(
        out / "cells_meta.tsv", sep="\t", index=False)
    ct = adata.obs["cell_type"].value_counts()
    with open(out / "summary.txt", "w") as fh:
        fh.write(f"cells={adata.n_obs} genes={adata.n_vars} "
                 f"cell_types={ct.size} cell_states={adata.obs['cell_state'].nunique()}\n\n")
        fh.write(ct.to_string())
    print(f"\n=== reference cell types ({ct.size}) ===")
    print(ct.to_string())
    print(f"\nWrote reference -> {out}/")
    return ct


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", required=True)
    ap.add_argument("--tissue", required=True)
    ap.add_argument("--conditions", help="comma-sep condition_resolved filter "
                    "(restrict the reference to healthy/control arms)")
    ap.add_argument("--sex", help="sex_resolved filter (e.g. male/female)")
    ap.add_argument("--sample-ids", help="comma-sep explicit sample_id list (use when the "
                    "healthy/genotype split is only in geo_title, not condition_resolved)")
    ap.add_argument("--min-state-cells", type=int, default=20)
    ap.add_argument("--label-scheme", default="none", choices=["none"] + sorted(LABEL_SCHEMES),
                    help="merge collinear consensus-label fragments before building the GEP "
                    "(e.g. 'brain' merges pyramidal/glutamatergic neuron synonyms into "
                    "'Excitatory neurons'); apply the SAME scheme to the cross source")
    ap.add_argument("--out", help="output dir (default deconvolution/reference/{tissue}_{study}); "
                    "set explicitly to avoid clobbering when tissue+study collide, e.g. a WT subset")
    ap.add_argument("--gene-join", choices=["inner", "outer"], default="inner",
                    help="combine per-sample gene sets by intersection ('inner', default) or "
                    "union ('outer', 0-filled) -- use 'outer' for uneven-depth studies (cortex)")
    ap.add_argument("--min-gene-cells", type=int, default=0,
                    help="after concat, drop genes expressed in < this many pooled cells "
                    "(prunes a 'outer'-join union's long tail; e.g. 10)")
    args = ap.parse_args()
    conds = [c.strip() for c in args.conditions.split(",")] if args.conditions else None
    sids = [s.strip() for s in args.sample_ids.split(",")] if args.sample_ids else None
    adata = load_study(args.study, args.tissue, conds, args.sex, sids,
                       gene_join=args.gene_join, min_gene_cells=args.min_gene_cells)
    adata.obs["cell_type"] = canonicalize_labels(adata.obs["cell_type"], args.label_scheme)
    adata = clean_cells(adata, args.min_state_cells)
    out = Path(args.out) if args.out else Path(resolve_path(_CFG, _DC["built_reference_dir"])) / f"{args.tissue}_{args.study}"
    export_reference(adata, out)


if __name__ == "__main__":
    main()
