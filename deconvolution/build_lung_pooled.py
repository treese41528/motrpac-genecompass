#!/usr/bin/env python3
"""build_lung_pooled.py -- build a POOLED NATIVE adult-rat-lung BayesPrism reference.

Replaces the engineered/in-vitro GSE178405 lung reference (cell isolates / engineered-d7 /
tri-culture / P7-developing -- the root cause of lung being the weakest deconv tissue) with a
native reference pooled from the HEALTHY control arms of adult rat lung scRNA studies
(all already in-corpus + consensus-annotated):
    GSE273062  VeNx (vehicle+normoxia) controls  -> sample0, sample2   (2)
    GSE252844  C3 control (blast-injury study)    -> sample0           (1)
= 3 healthy native adult lung samples.
DROPPED 2026-07-15 (StudyDescriptions review): GSE242310 -- BOTH its arms (NOX/HOX) are P1-P10 NEONATES
(a rule-6 developmental violation the reference_qc DEVEL check only WARNs on, so it slipped through). The
prior build wrongly included GSE242310_sample1 as a "normoxia control"; it is a P10 neonate. (GSE247625's
2 healthy adults are deferred -- mouse-mm10 aligned, needs re-alignment.) Reuses build_reference machinery:
per-study load_study (consensus labels via barcode->leiden->consensus_label), cross-study OUTER gene join
(studies differ in depth), the 'lung' label-scheme (merge Club/Clara/ciliated/NK synonyms; keep immune
resolved), clean_cells, export_reference. Run on a compute node (loads h5ads).

The samples are read from deconvolution/tissue_references.yaml (LUNG: the studies in the order of
`pooled_studies`, each with its samples from `expect.sample_ids`). Until 2026-11-14 the same three samples
were also typed into this file (audit finding 4.10); the list below is now derived, and
tests/test_reference_builders.py asserts it equals the list that was typed in. The output folder has no
default: --out is required, so a bare run cannot overwrite a reference in use.
"""
import argparse, os, sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
REFERENCES_YAML = ROOT / "deconvolution" / "tissue_references.yaml"


def healthy_samples(config_path=REFERENCES_YAML, tissue="LUNG"):
    """[(study, [sample_id, ...]), ...] of a pooled reference, from tissue_references.yaml: the studies in the
    order of `pooled_studies`, each with the `expect.sample_ids` that start with "<study>_", in the yaml's
    order. Every expected sample must belong to exactly one listed study and every study must have a sample."""
    cfg = yaml.safe_load(open(config_path))[tissue]
    studies = [c["study"] for c in cfg["pooled_studies"]]
    expected = list(cfg["expect"]["sample_ids"])
    out = [(s, [x for x in expected if x.startswith(s + "_")]) for s in studies]
    placed = [x for _, sids in out for x in sids]
    if sorted(placed) != sorted(expected) or len(set(studies)) != len(studies) or any(not sids for _, sids in out):
        sys.exit(f"ERROR: {config_path} [{tissue}]: expect.sample_ids {expected} cannot be assigned one-to-one to "
                 f"pooled_studies {studies}")
    return out


def _br():
    """build_reference, imported on first use: it loads the pipeline config and scanpy, which the sample list
    above does not need."""
    os.environ.setdefault("PIPELINE_ROOT", str(ROOT))
    sys.path.insert(0, str(ROOT / "deconvolution"))
    import build_reference
    return build_reference


def build_pooled_lung(joint_labels=None, healthy=None):
    """Assemble the pooled native-lung AnnData (the production recipe) and RETURN it
    (no export). Importable so make_purity_sweep.py --pooled-lung can run a within-pool purity
    sweep on the exact production reference."""
    import anndata as ad
    import scanpy as sc
    br = _br()
    healthy = healthy_samples() if healthy is None else healthy
    parts = []
    for study, sids in healthy:
        a = br.load_study(study, "lung", sample_ids=sids, gene_join="outer", min_gene_cells=0)
        a.obs["source_study"] = study
        print(f"  {study}: {a.n_obs} cells, {a.obs['cell_type'].nunique()} consensus labels")
        parts.append(a)

    adata = ad.concat(parts, join="outer", merge="same", fill_value=0)
    print(f"\npooled: {adata.n_obs} cells x {adata.n_vars} genes from {len(healthy)} studies")
    before = adata.n_vars
    sc.pp.filter_genes(adata, min_cells=10)           # trim the outer-join union's long tail
    print(f"min-gene-cells=10: {before} -> {adata.n_vars} genes")

    if joint_labels:                                  # joint re-annotation replaces the consensus labels
        adata = br.apply_joint_labels(adata, joint_labels)
    else:
        adata.obs["cell_type"] = br.canonicalize_labels(adata.obs["cell_type"], "lung")
    adata = br.clean_cells(adata, 20)                 # >=20 cells per (sample x state)
    print(f"\nafter clean_cells: {adata.n_obs} cells, {adata.obs['cell_type'].nunique()} cell types")
    print("cell-type counts:")
    print(adata.obs["cell_type"].value_counts().to_string())
    print("cells per source study:")
    print(adata.obs["source_study"].value_counts().to_string())
    return adata


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="output reference dir (tissue_references.yaml: LUNG.reference_dir)")
    ap.add_argument("--joint-labels", help="cells_final.tsv.gz from joint_reannotate.py finalize "
                    "(cell types + joint clusters as states) instead of the consensus labels")
    args = ap.parse_args()
    healthy = healthy_samples()
    adata = build_pooled_lung(args.joint_labels, healthy)
    br = _br()
    br.export_reference(adata, args.out)
    print(f"\nwrote pooled native lung reference -> {args.out}")
    # Provenance, written after the reference files and never read by them (audit finding 4.10 / cross-stage).
    obs = adata.obs
    br.write_provenance(args.out, inputs=[br.INVENTORY, REFERENCES_YAML] + ([args.joint_labels] if args.joint_labels else []),
                        md5_inputs=[REFERENCES_YAML] + ([args.joint_labels] if args.joint_labels else []),
                        params={**vars(args), "healthy_samples": healthy,
                                "n_cells": int(adata.n_obs), "n_genes": int(adata.n_vars),
                                "samples": sorted(obs["sample"].astype(str).unique()),
                                "cell_types": sorted(obs["cell_type"].astype(str).unique())})


if __name__ == "__main__":
    main()
