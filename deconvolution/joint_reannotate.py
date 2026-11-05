#!/usr/bin/env python3
"""joint_reannotate.py -- joint re-annotation of a deconvolution reference (REFERENCE_RELABEL_PREREG.md §10).

The corpus labels were assigned one sample at a time: each sample was clustered alone, then PanglaoDB
(scored against every organ) and ScType (tissue lists that fell back to immune-only for brain, muscle,
adipose and jejunum) were reconciled per label pair by an LLM. Nothing kept two samples' names for the same
cells the same, and nothing checked a name against the cluster's markers. This script replaces that for the
references: all samples of a reference are clustered together, and each joint cluster is labelled from
marker panels fixed in advance (deconvolution/reference_marker_panels.yaml) by a deterministic rule.

  run       load every QC cell of the reference's samples, normalise, HVGs (per-sample batch), PCA,
            Harmony on sample, Leiden; score panels per cluster; write the automatic labels, the clusters
            the rule sends to REVIEW, per-cluster markers, and per-(sample, cluster) count sums.
  finalize  apply the committed review decisions (deconvolution/joint_reannotation_review.tsv), write the
            final per-cell labels, and run the checks of §10.4 on them.

Cluster-ID guard. The review decisions are keyed to Leiden cluster IDs (j0, j1, ...), which are only stable
if `run` reproduces the same partition AND numbering (same cells, package versions, seed). Next to the
decisions, deconvolution/joint_reannotation_clusters.tsv records a SHA-256 of each reference's cluster
assignment (sorted sample, barcode, joint_cluster rows of cells.tsv.gz). `run` reports whether it reproduced
the recorded assignment; `finalize` REFUSES to apply the decisions to a different assignment (renumbered or
re-partitioned clusters would silently get the wrong labels). After a deliberate re-review, record the new
assignment with `finalize --record-clusters`. For exact reproduction without re-running Leiden, the final
labels data/deconvolution/joint_labels/<REF>/cells_final.tsv.gz are deposited and the reference builds
read them directly (tissue_references.yaml `joint_labels:`).

Rule (§10.3). Per cluster, each panel gene's mean log-normalised expression is min-max scaled across
clusters; a panel's score is the mean over its genes found in the data (at least 2). Top-level classes:
the best class is accepted if it scores >= ACCEPT and beats the runner-up by >= MARGIN; a second class from
another compartment also scoring >= DOUBLET flags a possible doublet. Anything else goes to REVIEW.
Accepted classes descend into their sub-panels by the same ACCEPT / MARGIN test among siblings, and stop
at the parent when no child passes.

Output: data/deconvolution/joint_labels/<REF>/  (cells.tsv.gz, clusters.tsv, scores.tsv, panel_genes.tsv,
group_sums.npz, settings.json; after finalize: cells_final.tsv.gz, checks.json, label_checks.tsv,
old_vs_new.tsv)

Usage (project venv; loads every cell of the reference -> run via SLURM):
  python deconvolution/joint_reannotate.py run --ref KIDNEY
  python deconvolution/joint_reannotate.py finalize --ref KIDNEY
  python deconvolution/joint_reannotate.py finalize --ref KIDNEY --record-clusters   # after a re-review only
"""
import hashlib
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import yaml

ROOT = Path(os.environ.setdefault("PIPELINE_ROOT", str(Path(__file__).resolve().parents[1])))
sys.path.insert(0, str(ROOT / "deconvolution"))

PANELS = ROOT / "deconvolution/reference_marker_panels.yaml"
REVIEW = ROOT / "deconvolution/joint_reannotation_review.tsv"
SPLITS = ROOT / "deconvolution/joint_reannotation_splits.tsv"
CLUSTER_SUMS = ROOT / "deconvolution/joint_reannotation_clusters.tsv"
OUT = ROOT / "data/deconvolution/joint_labels"

# reference key -> tissue_references.yaml key whose spec selects the samples
REFS = {"KIDNEY": "KIDNEY", "LIVER": "LIVER", "CORTEX": "CORTEX", "WATSC": "WATSC", "MUSCLE": "SKMGN",
        "SMLINT": "SMLINT", "LUNG": "LUNG", "HIPPOC": "HIPPOC"}
# BLOOD option B (REFERENCE_RELABEL_PREREG.md §12): external GEO libraries, not corpus samples; all three
# are labelled with the BLOOD panel. Paths are under data/raw/geo/blood_option_b/.
BLOOD_RAW = ROOT / "data/raw/geo/blood_option_b"
BLOOD_SETS = {
    "BLOOD_PBMC": [(f"GSE335156_V{i}", BLOOD_RAW / "GSE335156", f"GSM98068{32 + i}_V{i}_", False) for i in range(1, 6)],
    "BLOOD_WB": [("GSE306764_BloodCtrl", BLOOD_RAW / "GSE306764/blood/Blood-Ctrl", "", True)],
    "BLOOD_BM": [("GSE306764_BoneCtrl", BLOOD_RAW / "GSE306764/bone/Bone-Ctrl", "", True)],
}
PANEL_KEY = {k: "BLOOD" for k in BLOOD_SETS}
# SPLEEN, OVARY (REFERENCE_RELABEL_PREREG.md §14) and the second heart reference (§15): external GEO libraries
# under data/raw/geo/. (sample, path, format, lift): format "mtx_dir" = a directory with genes.tsv /
# features.tsv, barcodes.tsv, matrix.mtx (optionally .gz), or a file prefix of them; "h5" = a 10x
# feature_bc_matrix.h5. lift: True = put Rnor_6.0 IDs through the bulk's liftover map;
# False = IDs already in the lifted (Ensembl 115) space; "symbol" = features are NCBI gene symbols (GSE307551,
# Cell Ranger on the RefSeq annotation): mapped to the bulk's IDs through symbol_index, a symbol kept only when
# it resolves to exactly one bulk ID, symbols landing on one ID summed (REFERENCE_RELABEL_PREREG.md §16).
GEO_RAW = ROOT / "data/raw/geo"
EXTERNAL_SETS = {
    "SPLEEN": [("GSE306764_SpleenCtrl", GEO_RAW / "GSE306764_spleen/extracted/Spl-Ctrl", "mtx_dir", True)],
    "OVARY": [("GSE343214_Normal", GEO_RAW / "GSE343214/GSM9947804_Normal_", "mtx_dir", False)],
    "HEART_GSE307551": [
        ("GSE307551_F3441", GEO_RAW / "GSE307551/GSM9226643_F3441_feature_bc_matrix.h5", "h5", "symbol"),
        ("GSE307551_F3442", GEO_RAW / "GSE307551/GSM9226644_F3442_feature_bc_matrix.h5", "h5", "symbol")],
}
BULK_TISSUE = {"SPLEEN": "SPLEEN", "OVARY": "OVARY", "HEART_GSE307551": "HEART"}
# RefSeq names of the 13 mitochondrial protein-coding genes (GSE307551 has no "mt-" prefix on them)
MITO_REFSEQ = {"ND1", "ND2", "ND3", "ND4", "ND4L", "ND5", "ND6", "COX1", "COX2", "COX3", "ATP6", "ATP8", "CYTB"}
PANEL_KEY.update({"SPLEEN": "SPLEEN", "OVARY": "OVARY", "HEART_GSE307551": "HEART_COARSE"})
ALL_REFS = sorted(REFS) + sorted(BLOOD_SETS) + sorted(EXTERNAL_SETS)

N_HVG, N_PCS, N_NEIGHBORS, RESOLUTION, SEED = 3000, 30, 15, 1.0, 0
ACCEPT, MARGIN, DOUBLET, MIN_PANEL_GENES = 0.5, 0.15, 0.5, 2
MIN_CLUSTER_CELLS = 20
MARKER_MIN_DETECT, N_MARKERS = 0.25, 25
SINGLE_SAMPLE_SHARE = 0.9


# ----------------------------------------------------------------------------- panels
def load_panels(ref, path=PANELS):
    """The reference's top-level classes, with `use:` pulled in from the shared block."""
    doc = yaml.safe_load(open(path))
    spec = dict(doc[ref])
    for name in spec.pop("use", []) or []:
        if name in spec:
            continue                       # a tissue-specific definition overrides the shared one
        spec[name] = doc["_shared"][name]
    return spec


def symbol_index(var_names):
    """Map rat symbols -> the reference's gene IDs: Ensembl 115 names, then the corpus gene universe,
    then RGD's symbol -> Ensembl ID lists (kept only where the ID is in the reference)."""
    in_ref = set(var_names)
    idx = {}

    def add(sym, gid):
        if isinstance(sym, str) and sym and gid in in_ref:
            idx.setdefault(sym.lower(), set()).add(gid)

    info = pd.read_csv(ROOT / "data/references/biomart/rat_gene_info.tsv", sep="\t", dtype=str)
    for s, g in zip(info["Gene name"], info["Gene stable ID"]):
        add(s, g)
    gu = pd.read_csv(ROOT / "data/training/gene_universe/gene_universe.tsv", sep="\t", dtype=str,
                     usecols=["ensembl_id", "symbol"])
    for g, s in zip(gu["ensembl_id"], gu["symbol"]):
        if not isinstance(s, str) or s.lower() not in idx:
            add(s, g)
    rgd = pd.read_csv(ROOT / "data/references/biomart/GENES_RAT.txt", sep="\t", comment="#", dtype=str,
                      usecols=["SYMBOL", "ENSEMBL_ID"])
    for s, ids in zip(rgd["SYMBOL"], rgd["ENSEMBL_ID"]):
        if isinstance(s, str) and s.lower() not in idx and isinstance(ids, str):
            for g in ids.split(";"):
                add(s, g)
    return idx


def resolve(spec, idx, parent=None, rows=None):
    """Attach gene IDs to every panel (recursively); record what was and wasn't found."""
    rows = [] if rows is None else rows
    out = {}
    for name, d in spec.items():
        ids, found, missing = [], [], []
        for s in d["genes"]:
            g = sorted(idx.get(s.lower(), ()))
            (found if g else missing).append(s)
            ids += g
        node = {"compartment": d.get("compartment"), "genes": d["genes"], "ids": sorted(set(ids)),
                "found": found, "missing": missing}
        rows.append({"panel": name, "parent": parent or "", "compartment": node["compartment"] or "",
                     "found": ",".join(found), "missing": ",".join(missing),
                     "scorable": len(found) >= MIN_PANEL_GENES})
        if d.get("sub"):
            node["sub"] = resolve(d["sub"], idx, name, rows)[0]
        out[name] = node
    return out, rows


# ----------------------------------------------------------------------------- data
def load_blood(ref):
    """One BLOOD_SETS entry as raw counts in the bulk's gene-ID space, with the corpus QC applied (>= 200
    genes, mitochondrial fraction <= 0.15, as config/pipeline_config.yaml sets for the corpus). GSE306764 ships the Rnor_6.0
    IDs the MoTrPAC bulk was released in, so it goes through the bulk's own liftover map (as TESTES);
    GSE335156 (Cell Ranger, mRatBN7.2) is already in the lifted ID space."""
    import anndata as ad
    import scipy.io as sio
    import build_reference as br
    min_genes, max_mito = 200, 0.15        # the corpus QC: min_genes_per_cell / max_mito_fraction in the config
    parts = []
    for sample, d, prefix, lift in BLOOD_SETS[ref]:
        def f(name):
            for cand in (d / f"{prefix}{name}.gz", d / f"{prefix}{name}"):
                if cand.exists():
                    return cand
            raise FileNotFoundError(d / f"{prefix}{name}")
        feat = pd.read_csv(f("features.tsv" if not lift else "genes.tsv"), sep="\t", header=None, dtype=str)
        bc = pd.read_csv(f("barcodes.tsv"), header=None, dtype=str)[0]
        X = sp.csr_matrix(sio.mmread(str(f("matrix.mtx"))).T, dtype=np.float32)      # cells x genes
        mito = feat[1].str.lower().str.startswith("mt-").values
        n_genes = np.asarray((X > 0).sum(1)).ravel()
        frac_mt = np.asarray(X[:, mito].sum(1)).ravel() / np.maximum(np.asarray(X.sum(1)).ravel(), 1)
        keep = (n_genes >= min_genes) & (frac_mt <= max_mito)
        print(f"  {sample}: {X.shape[0]} barcodes -> {int(keep.sum())} after QC (>= {min_genes} genes, "
              f"mito <= {max_mito}); {int(mito.sum())} mito genes")
        a = ad.AnnData(X=X[keep], obs=pd.DataFrame({"sample": sample, "barcode": bc[keep].values,
                                                     "leiden": "", "cell_type": ""}))
        a.var_names = feat[0].values
        a.var_names_make_unique()
        if lift:
            a = br.lift_genes(a, ROOT / "deconvolution/reference/motrpac_bulk_liftover.tsv")
        parts.append(a)
    adata = ad.concat(parts, join="inner", merge="same", fill_value=0) if len(parts) > 1 else parts[0]
    return adata


def _read_external(path, fmt):
    """Raw counts (cells x genes, csr), gene IDs, gene symbols, barcodes of one external library."""
    import scipy.io as sio
    if fmt == "h5":
        import scanpy as sc
        a = sc.read_10x_h5(str(path), gex_only=True)
        return sp.csr_matrix(a.X, dtype=np.float32), np.asarray(a.var["gene_ids"]), np.asarray(a.var_names), \
            np.asarray(a.obs_names)
    d, prefix = (Path(path), "") if Path(path).is_dir() else (Path(path).parent, Path(path).name)

    def f(*names):
        for name in names:
            for cand in (d / f"{prefix}{name}.gz", d / f"{prefix}{name}"):
                if cand.exists():
                    return cand
        raise FileNotFoundError(d / f"{prefix}{names[0]}")
    feat = pd.read_csv(f("features.tsv", "genes.tsv"), sep="\t", header=None, dtype=str)
    bc = pd.read_csv(f("barcodes.tsv"), header=None, dtype=str)[0].values
    X = sp.csr_matrix(sio.mmread(str(f("matrix.mtx"))).T, dtype=np.float32)
    sym = feat[1].values if feat.shape[1] > 1 else feat[0].values
    return X, feat[0].values, sym, bc


def symbols_to_bulk(a, tissue):
    """Rename symbol-keyed genes to the tissue bulk's Ensembl IDs (see EXTERNAL_SETS)."""
    import anndata as ad
    bulk = pd.read_csv(ROOT / f"data/deconvolution/motrpac_bulk/{tissue}/bulk_genes.tsv", header=None, dtype=str)[0]
    idx = symbol_index(bulk.values)
    genes = np.asarray(a.var_names)
    hit = [sorted(idx.get(g.lower(), ())) for g in genes]
    keep = np.array([i for i, h in enumerate(hit) if len(h) == 1])
    ids, col = np.unique([hit[i][0] for i in keep], return_inverse=True)
    agg = sp.csr_matrix((np.ones(keep.size), (keep, col)), shape=(genes.size, ids.size))
    stats = {"genes_in": int(genes.size), "mapped": int(keep.size), "genes_out": int(ids.size),
             "summed_into_shared_id": int(keep.size - ids.size),
             "dropped_ambiguous": int(sum(len(h) > 1 for h in hit)),
             "dropped_unmapped": int(sum(len(h) == 0 for h in hit)), "bulk_genes": int(bulk.size)}
    print(f"  symbols -> {tissue} bulk IDs: {stats}")
    out = ad.AnnData(X=sp.csr_matrix(a.X) @ agg, obs=a.obs.copy(), var=pd.DataFrame(index=pd.Index(ids)))
    out.uns["gene_map"] = {"path": f"symbol_index -> motrpac_bulk/{tissue}/bulk_genes.tsv", **stats}
    return out


def load_external(ref):
    """One EXTERNAL_SETS entry as raw counts in the bulk's gene-ID space, with the corpus QC (>= 200 genes,
    mitochondrial fraction <= 0.15), as load_blood."""
    import anndata as ad
    import build_reference as br
    min_genes, max_mito = 200, 0.15
    parts = []
    for sample, path, fmt, lift in EXTERNAL_SETS[ref]:
        X, ids, sym, bc = _read_external(path, fmt)
        mito = (pd.Series(sym).astype(str).str.lower().str.startswith("mt-")
                | pd.Series(sym).astype(str).isin(MITO_REFSEQ)).values
        n_genes = np.asarray((X > 0).sum(1)).ravel()
        frac_mt = np.asarray(X[:, mito].sum(1)).ravel() / np.maximum(np.asarray(X.sum(1)).ravel(), 1)
        keep = (n_genes >= min_genes) & (frac_mt <= max_mito)
        print(f"  {sample}: {X.shape[0]} barcodes -> {int(keep.sum())} after QC (>= {min_genes} genes, "
              f"mito <= {max_mito}); {int(mito.sum())} mito genes")
        a = ad.AnnData(X=X[keep], obs=pd.DataFrame({"sample": sample, "barcode": bc[keep],
                                                     "leiden": "", "cell_type": ""}))
        a.var_names = ids
        a.var_names_make_unique()
        if lift == "symbol":
            a = symbols_to_bulk(a, BULK_TISSUE[ref])
        elif lift:
            a = br.lift_genes(a, ROOT / "deconvolution/reference/motrpac_bulk_liftover.tsv")
        parts.append(a)
    return ad.concat(parts, join="inner", merge="same", fill_value=0) if len(parts) > 1 else parts[0]


def load_cells(ref):
    """Every QC cell of the reference's samples, with the old per-sample labels, in the gene space the
    production build uses (same loader, same gene join)."""
    import anndata as ad
    import scanpy as sc
    import build_reference as br
    if ref in BLOOD_SETS or ref in EXTERNAL_SETS:
        adata = load_blood(ref) if ref in BLOOD_SETS else load_external(ref)
        adata.obs = adata.obs.rename(columns={"cell_type": "old_label", "leiden": "old_cluster"})
        adata.obs_names = (adata.obs["sample"].astype(str) + ":" + adata.obs["barcode"].astype(str)).values
        adata.X = sp.csr_matrix(adata.X, dtype=np.float32)
        return adata
    spec = yaml.safe_load(open(ROOT / "deconvolution/tissue_references.yaml"))[REFS[ref]]
    if spec.get("pooled"):                 # LUNG: per-study outer join, then the union trimmed (build_lung_pooled)
        parts = []
        for st in spec["pooled_studies"]:
            sids = [s for s in spec["expect"]["sample_ids"] if s.startswith(st["study"] + "_")]
            parts.append(br.load_study(st["study"], st["tissue"], sample_ids=sids, gene_join="outer"))
        adata = ad.concat(parts, join="outer", merge="same", fill_value=0)
        sc.pp.filter_genes(adata, min_cells=10)
    else:
        adata = br.load_study(spec["study"], spec["tissue"], sample_ids=spec["expect"]["sample_ids"],
                              gene_join=spec.get("gene_join", "inner"),
                              min_gene_cells=spec.get("min_gene_cells", 0) or 0,
                              organism=spec.get("organism", "Rattus norvegicus"),
                              dedup_gsm=spec.get("dedup_gsm", True))
    got = sorted(adata.obs["sample"].astype(str).unique())
    assert got == sorted(spec["expect"]["sample_ids"]), f"{ref}: samples {got} != expect"
    adata.obs = adata.obs.rename(columns={"cell_type": "old_label", "leiden": "old_cluster"})
    adata.obs["old_label"] = adata.obs["old_label"].astype(str)
    adata.obs_names = (adata.obs["sample"].astype(str) + ":" + adata.obs["barcode"].astype(str)).values
    adata.X = sp.csr_matrix(adata.X, dtype=np.float32)
    return adata


def cluster(adata, ref=None):
    """Normalise, HVGs per sample, PCA, Harmony on sample, kNN, Leiden. Adds obs['joint_cluster']."""
    import scanpy as sc
    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    multi = adata.obs["sample"].nunique() > 1
    sc.pp.highly_variable_genes(adata, n_top_genes=N_HVG, flavor="seurat_v3", layer="counts",
                                batch_key="sample" if multi else None)
    h = adata[:, adata.var["highly_variable"]].copy()
    sc.pp.scale(h, max_value=10)
    sc.tl.pca(h, n_comps=50, random_state=SEED)
    adata.obsm["X_pca"] = h.obsm["X_pca"]
    rep = "X_pca"
    if multi:
        sc.external.pp.harmony_integrate(adata, "sample", basis="X_pca", adjusted_basis="X_pca_harmony",
                                         random_state=SEED)
        rep = "X_pca_harmony"
    sc.pp.neighbors(adata, n_neighbors=N_NEIGHBORS, n_pcs=N_PCS, use_rep=rep, random_state=SEED)
    sc.tl.leiden(adata, resolution=RESOLUTION, key_added="joint_cluster", flavor="igraph",
                 n_iterations=2, directed=False, random_state=SEED)
    adata.obs["joint_cluster"] = "j" + adata.obs["joint_cluster"].astype(str)
    split_clusters(adata, ref)
    return rep


def split_clusters(adata, ref, path=SPLITS):
    """§9 amendment: a cluster the review found to hold two distinct populations is re-clustered on its own
    (Leiden, same resolution and seed, on the same kNN graph restricted to its cells); its parts are named
    j<N>.<k> and go through the rule like any other cluster. The first clustering is unchanged (same seed),
    so every other cluster keeps its cells and name."""
    import scanpy as sc
    if not Path(path).exists():
        return
    sp_ = pd.read_csv(path, sep="\t", dtype=str, comment="#")
    for c in sp_.loc[sp_["reference"] == ref, "cluster"]:
        assert c in set(adata.obs["joint_cluster"]), f"{ref}: split names unknown cluster {c}"
        adata.obs["joint_cluster"] = adata.obs["joint_cluster"].astype("category")   # restrict_to needs it
        sc.tl.leiden(adata, resolution=RESOLUTION, key_added="_split", flavor="igraph", n_iterations=2,
                     directed=False, random_state=SEED, restrict_to=("joint_cluster", [c]))
        adata.obs["joint_cluster"] = adata.obs["_split"].astype(str).str.replace(",", ".", regex=False)
        del adata.obs["_split"]
        print(f"  split {c} -> {sorted(x for x in adata.obs['joint_cluster'].unique() if x.startswith(c + '.'))}")
    adata.obs["joint_cluster"] = adata.obs["joint_cluster"].astype(str)


def onehot(labels):
    cats = pd.Index(sorted(pd.unique(labels), key=lambda c: (len(c), c)))
    codes = cats.get_indexer(labels)
    return cats, sp.csr_matrix((np.ones(len(codes)), (codes, np.arange(len(codes)))), shape=(len(cats), len(codes)))


# ----------------------------------------------------------------------------- scoring
def scaled_means(means):
    """Min-max scale each gene's cluster means across clusters (0 = lowest cluster, 1 = highest)."""
    lo, hi = means.min(0), means.max(0)
    return (means - lo) / (hi - lo).replace(0, np.nan)


def panel_score(scaled, node):
    cols = [g for g in node["ids"] if g in scaled.columns]
    if len(node["found"]) < MIN_PANEL_GENES or not cols:
        return pd.Series(np.nan, index=scaled.index)
    return scaled[cols].mean(1)


def score_tree(scaled, tree, prefix="", out=None):
    out = {} if out is None else out
    for name, node in tree.items():
        out[prefix + name] = panel_score(scaled, node)
        if node.get("sub"):
            score_tree(scaled, node["sub"], prefix + name + " > ", out)
    return out


def best_two(s):
    s = s.dropna().sort_values(ascending=False)
    return (s.index[0] if len(s) else None, s.iloc[0] if len(s) else np.nan,
            s.index[1] if len(s) > 1 else None, s.iloc[1] if len(s) > 1 else np.nan)


def assign(scores, tree):
    """Labels per cluster by the §10.3 rule. scores: clusters x 'A > B > C' panel paths."""
    top = list(tree)
    rows = []
    for c in scores.index:
        s = scores.loc[c]
        b, bs, r, rs = best_two(s[top])
        row = {"cluster": c, "best": b, "best_score": bs, "runner_up": r, "runner_up_score": rs,
               "margin": bs - rs if r else np.nan, "review": "", "auto_label": "", "path": ""}
        if b is None or not bs >= ACCEPT:
            row["review"] = f"no panel >= {ACCEPT}"
        elif r is not None and bs - rs < MARGIN:
            row["review"] = f"near-tie with {r}"
        else:
            other = [t for t in top if t != b and tree[t]["compartment"] != tree[b]["compartment"]
                     and s[t] >= DOUBLET]
            if other:
                row["review"] = "possible doublet with " + ", ".join(other)
        # descend from the best class even when under review, so the reviewer sees the candidate
        path, node, label = [b] if b else [], tree.get(b, {}), b or ""
        while node.get("sub"):
            kids = list(node["sub"])
            k, ks, k2, ks2 = best_two(s[[" > ".join(path + [x]) for x in kids]])
            if k is None or not ks >= ACCEPT or (k2 is not None and ks - ks2 < MARGIN):
                break
            k = k.split(" > ")[-1]
            path.append(k)
            node, label = node["sub"][k], k
        row["path"] = " > ".join(path)
        row["auto_label"] = label if not row["review"] else ""
        row["candidate"] = label
        rows.append(row)
    return pd.DataFrame(rows).set_index("cluster")


# ----------------------------------------------------------------------------- cluster-ID guard
def cluster_assignment_sha256(cells):
    """SHA-256 of the (sample, barcode, joint_cluster) rows, sorted by sample then barcode."""
    c = cells[["sample", "barcode", "joint_cluster"]].astype(str).sort_values(["sample", "barcode"], kind="mergesort")
    h = hashlib.sha256()
    for row in c.itertuples(index=False):
        h.update(("\t".join(row) + "\n").encode())
    return h.hexdigest()


def recorded_clusters(path=CLUSTER_SUMS):
    if not Path(path).exists():
        return pd.DataFrame(columns=["reference", "sha256", "n_cells", "n_clusters"]).set_index("reference")
    return pd.read_csv(path, sep="\t", dtype=str, comment="#").set_index("reference")


def record_clusters(ref, cells, path=CLUSTER_SUMS):
    rec = recorded_clusters(path)
    rec.loc[ref] = [cluster_assignment_sha256(cells), str(len(cells)), str(cells["joint_cluster"].nunique())]
    with open(path, "w") as fh:
        fh.write("# SHA-256 of each reference's joint cluster assignment (joint_reannotate.py; sorted sample, barcode,\n"
                 "# joint_cluster rows of cells.tsv.gz). finalize refuses review decisions on any other assignment.\n")
        rec.sort_index().rename_axis("reference").to_csv(fh, sep="\t")
    print(f"[{ref}] recorded cluster assignment in {path}")


def check_clusters(ref, cells, path=CLUSTER_SUMS):
    """None if the assignment matches the recorded one; else a message saying how it differs."""
    rec = recorded_clusters(path)
    if ref not in rec.index:
        return (f"{ref}: no recorded cluster assignment in {path}; the review decisions cannot be verified "
                f"against these clusters (record one with `finalize --ref {ref} --record-clusters` only after "
                f"reviewing clusters.tsv)")
    r = rec.loc[ref]
    got = cluster_assignment_sha256(cells)
    if got == r["sha256"]:
        return None
    return (f"{ref}: cluster assignment differs from the one the review decisions were made on\n"
            f"  recorded: sha256 {r['sha256'][:16]}..  {r['n_cells']} cells, {r['n_clusters']} clusters\n"
            f"  this run: sha256 {got[:16]}..  {len(cells)} cells, {cells['joint_cluster'].nunique()} clusters\n"
            f"  Leiden renumbered or re-partitioned the clusters (package versions, inputs, seed), so cluster IDs "
            f"in {REVIEW.name} would label the wrong cells. Use the deposited cells_final.tsv.gz, or re-review.")


# ----------------------------------------------------------------------------- run
def run(ref):
    out = OUT / ref
    out.mkdir(parents=True, exist_ok=True)
    adata = load_cells(ref)
    print(f"[{ref}] {adata.n_obs} cells x {adata.n_vars} genes, samples "
          f"{adata.obs['sample'].value_counts().to_dict()}", flush=True)
    idx = symbol_index(adata.var_names)
    tree, prow = resolve(load_panels(PANEL_KEY.get(ref, ref)), idx)
    pd.DataFrame(prow).to_csv(out / "panel_genes.tsv", sep="\t", index=False)
    rep = cluster(adata, ref)

    cl = adata.obs["joint_cluster"].astype(str).values
    cats, M = onehot(cl)
    n = np.asarray(M.sum(1)).ravel()
    X = adata.X                                              # log-normalised
    means = pd.DataFrame((M @ X).toarray() / n[:, None], index=cats, columns=adata.var_names)
    detect = pd.DataFrame((M @ (X > 0).astype(np.float32)).toarray() / n[:, None], index=cats,
                          columns=adata.var_names)
    scores = pd.DataFrame(score_tree(scaled_means(means), tree))
    lab = assign(scores, tree)

    # markers: log2 fold change of pooled CPM, cluster vs the rest, among genes detected in >= 25% of it
    C = adata.layers["counts"]
    csum = np.asarray((M @ C).todense())
    cpm = csum / csum.sum(1, keepdims=True) * 1e6
    rest = (csum.sum(0, keepdims=True) - csum)
    rest = rest / rest.sum(1, keepdims=True) * 1e6
    lfc = np.log2(cpm + 1) - np.log2(rest + 1)
    info = pd.read_csv(ROOT / "data/references/biomart/rat_gene_info.tsv", sep="\t", dtype=str)
    names = dict(zip(info["Gene stable ID"], info["Gene name"]))
    def gname(g):
        n_ = names.get(g)
        return n_ if isinstance(n_, str) and n_ else g
    mk = []
    for i, c in enumerate(cats):
        ok = np.flatnonzero(detect.iloc[i].values >= MARKER_MIN_DETECT)
        top = ok[np.argsort(-lfc[i, ok])[:N_MARKERS]]
        mk.append(",".join(f"{gname(adata.var_names[j])}({lfc[i, j]:.1f})" for j in top))

    per_sample = pd.crosstab(adata.obs["joint_cluster"], adata.obs["sample"]).reindex(cats)
    share = per_sample.div(per_sample.sum(1), axis=0)
    ent = -(share * np.log(share.where(share > 0, 1))).sum(1) / max(np.log(per_sample.shape[1]), 1e-9)
    old = (adata.obs.groupby("joint_cluster", observed=True)["old_label"]
           .agg(lambda x: "; ".join(f"{k} {v}" for k, v in x.value_counts().head(4).items())).reindex(cats))
    qc = adata.obs.assign(n_counts=np.asarray(C.sum(1)).ravel(), n_genes=np.asarray((C > 0).sum(1)).ravel())
    qc = qc.groupby("joint_cluster", observed=True)[["n_counts", "n_genes"]].median().reindex(cats)

    clusters = pd.DataFrame({"n_cells": n.astype(int)}, index=cats).join(lab)
    clusters["too_small"] = clusters["n_cells"] < MIN_CLUSTER_CELLS
    clusters["max_sample_share"] = share.max(1).round(3)
    clusters["sample_entropy"] = ent.round(3)
    clusters = clusters.join(per_sample.add_prefix("n_")).join(qc.add_prefix("median_"))
    clusters["old_labels"] = old
    clusters["top_markers"] = mk
    clusters.index.name = "cluster"
    clusters.to_csv(out / "clusters.tsv", sep="\t")
    scores.round(4).rename_axis("cluster").to_csv(out / "scores.tsv", sep="\t")

    cells = adata.obs[["sample", "barcode", "old_cluster", "old_label", "joint_cluster"]].copy()
    cells["auto_label"] = cells["joint_cluster"].map(clusters["auto_label"])
    cells.to_csv(out / "cells.tsv.gz", sep="\t", index=False)
    msg = check_clusters(ref, cells.astype(str))
    print(f"[{ref}] cluster assignment {'matches the recorded one' if msg is None else 'CHECK FAILED'}"
          + ("" if msg is None else f"\n!!! {msg}\n!!! finalize will refuse these clusters"), flush=True)

    # per-(sample, cluster) raw count sums, so finalize can build label profiles without reloading
    key = (adata.obs["sample"].astype(str) + "\t" + adata.obs["joint_cluster"].astype(str)).values
    gcats, G = onehot(key)
    np.savez_compressed(out / "group_sums.npz", sums=np.asarray((G @ C).todense()),
                        groups=np.asarray(gcats, dtype=str), genes=np.asarray(adata.var_names, dtype=str),
                        n_cells=np.asarray(G.sum(1)).ravel())
    try:
        import scanpy as sc
        sc.tl.umap(adata, random_state=SEED)
        pd.DataFrame(adata.obsm["X_umap"], index=adata.obs_names, columns=["umap1", "umap2"]) \
            .assign(sample=adata.obs["sample"].values, joint_cluster=cl) \
            .to_csv(out / "umap.tsv.gz", sep="\t")
    except Exception as e:                                   # the UMAP is for figures only
        print(f"  umap skipped: {e}")
    json.dump({"ref": ref, "n_cells": int(adata.n_obs), "n_genes": int(adata.n_vars), "rep": rep,
               "n_hvg": N_HVG, "n_pcs": N_PCS, "n_neighbors": N_NEIGHBORS, "resolution": RESOLUTION,
               "seed": SEED, "accept": ACCEPT, "margin": MARGIN, "doublet": DOUBLET,
               "min_panel_genes": MIN_PANEL_GENES, "min_cluster_cells": MIN_CLUSTER_CELLS,
               "marker_min_detect": MARKER_MIN_DETECT, "n_clusters": int(len(cats)),
               "n_review": int((clusters["review"] != "").sum()),
               "panels": str(PANELS.relative_to(ROOT))}, open(out / "settings.json", "w"), indent=2)
    print(f"[{ref}] {len(cats)} clusters, {int((clusters['review'] != '').sum())} to review -> {out}")


# ----------------------------------------------------------------------------- finalize
DROP = {"DROP"}


def final_labels(ref, clusters, review=REVIEW):
    """Automatic label, or the committed review decision; DROP and too-small clusters leave the reference."""
    lab = clusters["auto_label"].fillna("").astype(str).copy()
    rv = pd.read_csv(review, sep="\t", dtype=str, comment="#") if Path(review).exists() else pd.DataFrame(
        columns=["reference", "cluster", "decision", "reason"])
    rv = rv[rv["reference"] == ref].set_index("cluster")
    for c, r in rv.iterrows():
        if c not in lab.index:
            raise SystemExit(f"{ref}: review names unknown cluster {c}")
        lab[c] = r["decision"]
    pending = lab.index[(lab == "") & ~clusters["too_small"]]
    if len(pending):
        raise SystemExit(f"{ref}: clusters without a label or a review decision: {list(pending)}")
    lab[clusters["too_small"]] = "DROP"
    return lab


def label_profile_checks(ref, sums, groups, cl2lab):
    """§10.4 K1: the §4 profile metric on the final labels (same genes, centring and Pearson r)."""
    from audit_reference_labels import centred_profiles, corr_matrix
    g = pd.DataFrame([x.split("\t") for x in groups], columns=["sample", "cluster"])
    g["label"] = g["cluster"].map(cl2lab)
    keep = ~g["label"].isin(DROP)
    g, sums = g[keep].reset_index(drop=True), sums[keep.values]
    key = g["sample"] + "\t" + g["label"]
    cats, M = onehot(key.values)
    agg = np.asarray(M @ sums)
    gt = pd.DataFrame([c.split("\t") for c in cats], columns=["sample", "label"])
    x, _ = centred_profiles(agg)
    R = corr_matrix(x)
    same_lab = gt["label"].values[:, None] == gt["label"].values[None, :]
    diff_smp = gt["sample"].values[:, None] != gt["sample"].values[None, :]
    iu = np.triu_indices(len(gt), 1)
    r_self = {}
    for l in gt["label"].unique():
        ii = np.flatnonzero(gt["label"].values == l)
        rs = [R[i, j] for a, i in enumerate(ii) for j in ii[a + 1:] if gt.at[i, "sample"] != gt.at[j, "sample"]]
        if rs:
            r_self[l] = float(np.median(rs))
    diff = R[iu][(~same_lab & diff_smp)[iu]]
    return gt, r_self, (float(np.median(list(r_self.values()))) if r_self else np.nan,
                        float(np.median(diff)) if diff.size else np.nan)


def finalize(ref, record=False):
    out = OUT / ref
    cells = pd.read_csv(out / "cells.tsv.gz", sep="\t", dtype=str, keep_default_na=False)
    if record:
        record_clusters(ref, cells)
    msg = check_clusters(ref, cells)
    if msg is not None:
        raise SystemExit("FATAL (cluster-ID guard): " + msg)
    clusters = pd.read_csv(out / "clusters.tsv", sep="\t", index_col="cluster", keep_default_na=False)
    clusters["too_small"] = clusters["too_small"].astype(str) == "True"
    lab = final_labels(ref, clusters)
    clusters["final_label"] = lab
    cells["cell_type"] = cells["joint_cluster"].map(lab)
    cells.to_csv(out / "cells_final.tsv.gz", sep="\t", index=False)

    z = np.load(out / "group_sums.npz", allow_pickle=True)    # files from before the dtype=str fix hold object arrays
    gt, r_self, (med_self, med_diff) = label_profile_checks(ref, z["sums"], z["groups"], lab.to_dict())
    n_samples = cells["sample"].nunique()
    kept = cells[cells["cell_type"] != "DROP"]
    per = pd.crosstab(kept["cell_type"], kept["sample"])
    lc = pd.DataFrame({"n_cells": per.sum(1), "n_samples": (per > 0).sum(1),
                       "min_sample_share": per.div(per.sum(0), axis=1).min(1).round(4),
                       "r_self": pd.Series(r_self)}).rename_axis("cell_type")
    lc = lc.join(per.add_prefix("n_"))
    lc.sort_values("n_cells", ascending=False).to_csv(out / "label_checks.tsv", sep="\t")
    pd.crosstab(cells["old_label"], cells["cell_type"]).to_csv(out / "old_vs_new.tsv", sep="\t")
    clusters.to_csv(out / "clusters_final.tsv", sep="\t")
    checks = {
        "ref": ref, "n_samples": int(n_samples), "n_cells": int(len(cells)),
        "n_dropped": int((cells["cell_type"] == "DROP").sum()),
        "dropped_share": round(float((cells["cell_type"] == "DROP").mean()), 4),
        "n_labels": int(lc.shape[0]),
        "K1_median_r_self": med_self, "K1_median_r_different": med_diff,
        "K1_passes": bool(med_self > med_diff) if n_samples > 1 else None,
        "K1_labels_r_self_below_0.5": sorted(l for l, r in r_self.items() if r < 0.5),
        "K2_labels_in_one_sample": sorted(lc.index[lc["n_samples"] == 1]) if n_samples > 1 else [],
        "K2_single_sample_clusters": sorted(clusters.index[(clusters["max_sample_share"].astype(float)
                                                            >= SINGLE_SAMPLE_SHARE) & (lab != "DROP")])
        if n_samples > 1 else [],
    }
    json.dump(checks, open(out / "checks.json", "w"), indent=2)
    print(json.dumps(checks, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["run", "finalize"])
    ap.add_argument("--ref", required=True, choices=ALL_REFS)
    ap.add_argument("--record-clusters", action="store_true",
                    help="finalize: record THIS cluster assignment as the one the review decisions refer to "
                         "(only after re-reviewing clusters.tsv)")
    a = ap.parse_args()
    if a.command == "run":
        run(a.ref)
    else:
        finalize(a.ref, record=a.record_clusters)


if __name__ == "__main__":
    main()
