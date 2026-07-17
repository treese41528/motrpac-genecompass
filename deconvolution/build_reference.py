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
import re
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


def _canon_muscle(label):
    # Skeletal-muscle parenchyma is split across two collinear labels in GSE254371
    # ("Skeletal muscle cells" + "Skeletal muscle fibers") -- snRNA recovers some
    # multinucleated fibers as a separate sparse bucket. Merge ONLY those two
    # parenchyma fragments into one "Skeletal muscle"; keep stroma/vascular separate
    # ("Muscle fibroblasts", "Smooth muscle cells" do NOT contain "skeletal muscle").
    ll = str(label).strip().lower()
    if "skeletal muscle" in ll or ll in ("myofibers", "muscle fibers"):
        return "Skeletal muscle"
    return label


def _canon_lung(label):
    # Pooling healthy control arms across independent rat lung studies (GSE273062 VeNx,
    # GSE252844 C3, GSE242310 NOX) to build a NATIVE reference (replacing the engineered
    # in-vitro GSE178405) leaves near-synonymous EPITHELIAL fragments the per-study consensus
    # annotators split differently: Clara == Club == airway-club (one secretory bronchiolar
    # cell), and Ciliated == ciliated-airway. Merge ONLY those collinear epithelial synonyms,
    # plus the "natural killer"/"NK" duplicate (keep NKT separate). Immune subtypes
    # (CD4/CD8/naive/memory/monocyte fragments) are KEPT resolved -- coarsening them hurts
    # BayesPrism (omnideconv benchmark). Endothelial/AT2/fibroblast/macrophage stay as-is.
    ll = str(label).strip().lower()
    if ll in ("clara cells", "club cells", "airway club cells"):
        return "Club cells"
    if "ciliated" in ll:
        return "Ciliated cells"
    if ll in ("natural killer cells", "nk cells"):
        return "NK cells"
    return label


LABEL_SCHEMES = {"brain": _canon_brain, "muscle": _canon_muscle, "lung": _canon_lung}


def canonicalize_labels(series, scheme):
    """Map fine consensus labels onto a coarser separable scheme (or identity)."""
    if not scheme or scheme == "none":
        return series
    fn = LABEL_SCHEMES.get(scheme)
    if fn is None:
        sys.exit(f"ERROR: unknown --label-scheme {scheme!r}; choices: {sorted(LABEL_SCHEMES)}")
    return series.map(fn)


def select_samples(study, tissue, conditions=None, sex=None, sample_ids=None,
                   organism="Rattus norvegicus", title_include=None, title_exclude=None,
                   dedup_gsm=True):
    """In-corpus sample IDs for study+tissue, filtered (in precedence order) by:
    organism gate -> condition_resolved -> sex_resolved -> geo_title include/exclude regex ->
    explicit sample_ids -> GSM de-duplication -> (caller-side) reference-QC gate.

    Two arm filters exist because the healthy/young/genotype arm lives in DIFFERENT columns
    across studies. Use `conditions` (condition_resolved) when it is populated (e.g. GSE240658
    "No treatment"); use `title_include`/`title_exclude` (regex over geo_title|geo_source_name|
    geo_cell_type) when the arm lives ONLY in the title -- condition_resolved is unreliable and
    is BLANK for the aging arms (GSE137869 Ma-2020 -Y/-O/-CR; GSE305314 WT-vs-Tau; GSE248413
    Y-vs-O, whose condition_resolved reads "no treatment" for BOTH young and old).

    `organism` defaults to rat-only -- the keystone guard against a multi-species study (cortex
    GSE303115 spans 6 species) silently seeding a cross-species reference. Pass organism="any"
    to disable.

    `sample_ids` restricts to exactly those IDs but is validated against what SURVIVED the
    organism/arm filters, so a stale ID list can never re-introduce a wrong-species/wrong-arm cell.

    `dedup_gsm` collapses duplicate GSMs (the same physical library ingested under >1 sample_id --
    the _RAW double-ingest: GSE280111 LV 38->19, GSE303115 cortex 4->2) to one sample_id per GSM."""
    inv = pd.read_csv(INVENTORY, sep="\t", dtype=str)
    sel = inv[
        (inv["accession"] == study)
        & (inv["tissue_normalized"].str.lower() == tissue.lower())
        & (inv["in_corpus"].str.lower() == "true")
    ]
    # --- organism gate (default rat-only): the keystone fix. select_samples used to filter on
    # accession+tissue only, which let cortex GSE303115 (6 species) seed an ~85%-non-rat reference. ---
    if organism and str(organism).lower() != "any":
        sel = sel[sel["geo_organism"].fillna("").str.strip().str.lower() == str(organism).lower()]
    if conditions:
        sel = sel[sel["condition_resolved"].isin(conditions)]
    if sex:
        sel = sel[sel["sex_resolved"].str.lower() == sex.lower()]
    # --- geo_title include/exclude: expresses arms that live only in the title (Ma-2020 young -Y
    # arm; GSE248413 young "Y"; explicit Visium drop) over geo_title|geo_source_name|geo_cell_type. ---
    def _title_blob(row):
        return " | ".join(str(row.get(c, "")) for c in ("geo_title", "geo_source_name", "geo_cell_type"))
    if title_include:
        rx = re.compile(title_include, re.I)
        sel = sel[sel.apply(lambda r: bool(rx.search(_title_blob(r))), axis=1)]
    if title_exclude:
        rx = re.compile(title_exclude, re.I)
        sel = sel[~sel.apply(lambda r: bool(rx.search(_title_blob(r))), axis=1)]
    if sample_ids:
        want = set(sample_ids)
        sel = sel[sel["sample_id"].isin(want)]
        got = set(sel["sample_id"])
        missing = want - got
        if missing:
            sys.exit(f"ERROR: --sample-ids not in the organism/arm-filtered {study}/{tissue} "
                     f"selection (organism={organism}, conditions={conditions}, sex={sex}, "
                     f"title_include={title_include}, title_exclude={title_exclude}): {sorted(missing)}")
    # --- GSM de-dup (default on): one sample_id per physical library, collapsing the _RAW
    # double-ingest (GSE280111 LV 38->19, GSE303115 cortex 4->2) without hand-listing dedup sets. ---
    if dedup_gsm and "gsm" in sel.columns:
        n_before = len(sel)
        sel = sel.sort_values("sample_id").drop_duplicates(subset=["gsm"], keep="first")
        if len(sel) < n_before:
            print(f"  [dedup-gsm] {n_before} -> {len(sel)} samples (one sample_id per unique GSM)")
    samples = sorted(sel["sample_id"].tolist())
    if not samples:
        sys.exit(f"ERROR: no in-corpus {tissue} samples for {study} "
                 f"(organism={organism}, conditions={conditions}, sex={sex}, "
                 f"title_include={title_include}, title_exclude={title_exclude}, "
                 f"sample_ids={sample_ids}) in {INVENTORY}")
    # --- reference-QC gate: drop categorically non-native samples (spatial/Visium, engineered/
    # cultured/sorted, bulk) so a contaminated study can't silently seed a reference. Structural guard
    # against the liver-Visium / engineered-lung class of bug (see reference_qc.py). Developmental
    # (embryonic/postnatal) samples are WARNed, not dropped. Override the drop with ALLOW_NONNATIVE_REF=1.
    import os
    from reference_qc import SPATIAL, ENGINEERED, BULK, DEVEL
    def _title(sid):
        row = sel[sel["sample_id"] == sid]
        return "" if row.empty else " | ".join(
            str(row.iloc[0].get(c, "")) for c in ("geo_title", "geo_source_name", "geo_cell_type"))
    titles = {s: _title(s) for s in samples}
    bad   = {s: t for s, t in titles.items()
             if SPATIAL.search(t) or ENGINEERED.search(t) or BULK.search(t)}
    devel = {s: t for s, t in titles.items() if s not in bad and DEVEL.search(t)}
    if devel:
        print("  [reference-QC] WARNING: developmental (non-adult) sample(s): "
              + ", ".join(f"{s}[{t[:30]}]" for s, t in devel.items()))
    if bad and os.environ.get("ALLOW_NONNATIVE_REF") != "1":
        print(f"  [reference-QC] DROPPING {len(bad)} non-native sample(s) (spatial/engineered/sorted/bulk); "
              "set ALLOW_NONNATIVE_REF=1 to keep them:")
        for s, t in bad.items():
            print(f"      {s}: {t[:60]}")
        samples = [s for s in samples if s not in bad]
        if not samples:
            sys.exit(f"ERROR: every {study}/{tissue} sample is non-native (spatial/engineered/sorted/"
                     f"bulk) -> pick a native single-cell/nucleus study for this tissue (see reference_qc.py).")
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
               gene_join="inner", min_gene_cells=0,
               organism="Rattus norvegicus", title_include=None, title_exclude=None,
               dedup_gsm=True):
    """Concatenate all (loadable) samples of a study/tissue. gene_join='inner' keeps the
    shared (intersection) gene set -- the default, but it COLLAPSES studies with uneven
    per-sample gene depth (cortex GSE303115: per-sample 9.5k-21k -> 5.5k shared). Use
    gene_join='outer' to take the UNION (0-filled), optionally pruned by min_gene_cells
    (drop genes expressed in < that many pooled cells) to trim the union's long tail.
    organism/title_include/title_exclude/dedup_gsm are forwarded to select_samples()."""
    samples = select_samples(study, tissue, conditions, sex, sample_ids,
                             organism=organism, title_include=title_include,
                             title_exclude=title_exclude, dedup_gsm=dedup_gsm)
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
    ap.add_argument("--organism", default="Rattus norvegicus",
                    help="geo_organism gate (default rat-only); pass 'any' to disable. Keystone "
                    "guard: prevents a multi-species study (e.g. cortex GSE303115) from seeding a "
                    "cross-species reference")
    ap.add_argument("--title-include", help="keep only samples whose geo_title|geo_source_name|"
                    "geo_cell_type matches this regex (case-insensitive) -- for arms that live only "
                    r"in the title, e.g. Ma-2020 young arm '-Y($|\b|_)', GSE248413 young '(^|[^A-Za-z])Y([^A-Za-z]|$)'")
    ap.add_argument("--title-exclude", help="drop samples whose geo_title blob matches this regex "
                    "(case-insensitive), e.g. 'visium|spatial|_vis'")
    ap.add_argument("--no-dedup-gsm", action="store_true",
                    help="disable GSM de-dup (default ON: collapse duplicate GSMs -- the _RAW "
                    "double-ingest, e.g. GSE280111 LV 38->19 -- to one sample_id per library)")
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
                       gene_join=args.gene_join, min_gene_cells=args.min_gene_cells,
                       organism=args.organism, title_include=args.title_include,
                       title_exclude=args.title_exclude, dedup_gsm=not args.no_dedup_gsm)
    adata.obs["cell_type"] = canonicalize_labels(adata.obs["cell_type"], args.label_scheme)
    adata = clean_cells(adata, args.min_state_cells)
    out = Path(args.out) if args.out else Path(resolve_path(_CFG, _DC["built_reference_dir"])) / f"{args.tissue}_{args.study}"
    export_reference(adata, out)


if __name__ == "__main__":
    main()
