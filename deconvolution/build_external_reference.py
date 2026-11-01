#!/usr/bin/env python3
"""build_external_reference.py -- SPLEEN and OVARY references from external GEO libraries
(REFERENCE_RELABEL_PREREG.md §14) and the cross-study HEART check (§15).

Each library was labelled by joint_reannotate.py (run -> review -> finalize) with its panel in
reference_marker_panels.yaml. This script runs the registered checks on the final labels, builds a passing
reference, and makes/scores the simulated mixtures.

  checks       --ref SPLEEN|OVARY  S1 or O1/O2 + gene space (§12 check 5) -> data/deconvolution/s14_checks/<REF>.json
  build        --ref SPLEEN|OVARY --out DIR   the reference (DROP and the O2 guard classes left out)
  mixtures     --ref SPLEEN|OVARY --out DIR   R2: half-reference + 20 Dirichlet mixtures from the other half
  score        --stage DIR                    R2: r >= 0.9 for every type at >= 5% mean true fraction
  heart-mixtures --out DIR                    §15: 20 coarse-class Dirichlet mixtures from GSE307551
  heart-score    --stage DIR                  §15: production estimates summed to coarse classes; CM r, |err|

The R2 reference / §15 mixtures are deconvolved with deconvolution/R/run_deconvolution.sh (Stage 8) between
`mixtures` and `score` (SLURM: slurm/analysis/s14_checks.slurm, local).

Usage (project venv; loads every cell of a library -> SLURM):
  python deconvolution/build_external_reference.py checks --ref SPLEEN
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.sparse as sp
import yaml

ROOT = Path(os.environ.setdefault("PIPELINE_ROOT", str(Path(__file__).resolve().parents[1])))
sys.path.insert(0, str(ROOT / "deconvolution"))
import joint_reannotate as jr  # noqa: E402

LABELS = ROOT / "data/deconvolution/joint_labels"
CHECKS = ROOT / "data/deconvolution/s14_checks"
SEED, N_MIX, DEPTH = 0, 20, 5e6
GUARD = {"Lung epithelial cells (non-ovarian)", "Hepatocytes (non-ovarian)"}     # O2 guard classes
MYELOID_MONO_MAC = {"Monocytes", "Non-classical monocytes", "Macrophages", "Red-pulp macrophages", "Marginal zone macrophages"}
THECA_STROMA = {"Theca and stromal cells", "Theca cells", "Stromal cells"}


def labelled(ref):
    """QC'd raw counts (bulk ID space) of an external library joined to its final labels."""
    a = jr.load_external(ref)
    lab = pd.read_csv(LABELS / ref / "cells_final.tsv.gz", sep="\t", dtype=str, keep_default_na=False)
    lab.index = lab["sample"] + ":" + lab["barcode"]
    key = (a.obs["sample"].astype(str) + ":" + a.obs["barcode"].astype(str)).values
    assert set(key) == set(lab.index), f"{ref}: cells and labels differ"
    a.obs["cell_type"] = lab.loc[key, "cell_type"].values
    a.obs["joint_cluster"] = lab.loc[key, "joint_cluster"].values
    a.obs_names = key
    return a


def counts(ref):
    c = pd.read_csv(LABELS / ref / "cells_final.tsv.gz", sep="\t", dtype=str, keep_default_na=False)
    return c, c["cell_type"].value_counts()


def gene_space(a, tissue):
    bulk = set(pd.read_csv(ROOT / f"data/deconvolution/motrpac_bulk/{tissue}/bulk_genes.tsv", header=None)[0])
    return {"genes_after_mapping": int(a.n_vars), "in_bulk": int(a.var_names.isin(bulk).sum()),
            "bulk_genes": len(bulk), "gene_map": {k: (v if isinstance(v, (int, str)) else str(v))
                                                    for k, v in a.uns.get("gene_map", {}).items()}}


def checks(ref):
    CHECKS.mkdir(parents=True, exist_ok=True)
    cells, vc = counts(ref)
    res = {"ref": ref, "n_qc_cells": int(len(cells)), "n_dropped": int(vc.get("DROP", 0)),
           "cells_per_label": {k: int(v) for k, v in vc.items()}}
    if ref == "SPLEEN":
        n_b = int(vc.get("B cells", 0))
        n_t = int(sum(v for k, v in vc.items() if "T cells" in k))
        n_mm = int(sum(vc.get(k, 0) for k in MYELOID_MONO_MAC))
        n_neu = int(vc.get("Neutrophils", 0))
        res["S1"] = {"B_cells": n_b, "T_cells": n_t, "monocytes_plus_macrophages": n_mm, "neutrophils": n_neu,
                     "reported_red_pulp_macrophages": int(vc.get("Red-pulp macrophages", 0)),
                     "reported_erythroid": int(vc.get("Erythroid cells", 0)),
                     "pass": bool(n_b >= 100 and n_t >= 100 and (n_mm >= 100 or n_neu >= 100))}
        res["pass"] = res["S1"]["pass"]
    elif ref == "OVARY":
        n_g = int(vc.get("Granulosa cells", 0))
        n_ts = int(sum(vc.get(k, 0) for k in THECA_STROMA))
        res["O1"] = {"granulosa": n_g, "theca_stroma": n_ts, "pass": bool(n_g >= 100 and n_ts >= 100)}
        n_guard = int(sum(vc.get(k, 0) for k in GUARD))
        a = jr.load_external(ref)                       # raw counts, for the reported per-cell share
        idx = jr.symbol_index(a.var_names)
        cols = sorted({g for s in ("Sftpc", "Scgb1a1", "Alb", "Apoa1") for g in idx.get(s.lower(), ())})
        anyc = np.asarray((a.X[:, a.var_names.get_indexer(cols)] > 0).sum(1)).ravel() > 0
        res["O2"] = {"guard_cells": n_guard, "share": round(n_guard / len(cells), 4),
                     "reported_share_cells_with_any_Sftpc_Scgb1a1_Alb_Apoa1": round(float(anyc.mean()), 4),
                     "genes_found": len(cols), "pass": bool(n_guard / len(cells) <= 0.05)}
        res["pass"] = res["O1"]["pass"] and res["O2"]["pass"]
    res["gene_space"] = gene_space(jr.load_external(ref), jr.BULK_TISSUE[ref])
    json.dump(res, open(CHECKS / f"{ref}.json", "w"), indent=2)
    print(json.dumps(res, indent=2))


def reference_cells(a):
    """Cells that enter a reference: no DROP, no O2 guard class; cell_state = joint cluster."""
    a = a[~a.obs["cell_type"].isin(GUARD | {"DROP"}).values].copy()
    a.obs["cell_state"] = a.obs["joint_cluster"].astype(str)
    a.obs["leiden"] = a.obs["joint_cluster"].astype(str)
    return a


def build(ref, out):
    import build_reference as br
    a = br.clean_cells(reference_cells(labelled(ref)), 20)
    br.export_reference(a, out)


def half_split(types, seed=SEED):
    """Within each type, a seeded random half of the cells -> True (reference half)."""
    rng = np.random.default_rng(seed)
    m = np.zeros(len(types), bool)
    for t in sorted(pd.unique(types)):
        ii = np.flatnonzero(types == t)
        m[rng.choice(ii, size=len(ii) // 2, replace=False)] = True
    return m


def draw_mixtures(X, types, mix_types, out, rng, meta):
    """N_MIX mixtures, Dirichlet(1) mRNA fractions over mix_types, cells drawn with replacement up to each
    type's target reads (as build_blood_composite.mixtures); realised read shares are the truth."""
    libs = np.asarray(X.sum(1)).ravel()
    rows, truth = [], []
    for k in range(N_MIX):
        f = rng.dirichlet(np.ones(len(mix_types)))
        mix, got = np.zeros(X.shape[1]), {}
        for t, ft in zip(mix_types, f):
            ii = np.flatnonzero(types == t)
            need, s = ft * DEPTH, 0.0
            while s < need:
                draw = rng.choice(ii, size=max(1, min(200, int((need - s) / max(libs[ii].mean(), 1)) + 1)))
                mix += np.asarray(X[draw].sum(0)).ravel()
                s += libs[draw].sum()
            got[t] = s
        tot = sum(got.values())
        rows.append(mix)
        truth.append({"mixture": f"mix{k + 1}", **{f"rnafrac__{t}": got[t] / tot for t in mix_types}})
    (out / "mixtures").mkdir(parents=True, exist_ok=True)
    sio.mmwrite(str(out / "mixtures/pseudobulk_counts.mtx"),
                sp.csr_matrix(np.round(np.vstack(rows))).astype(np.int64))
    pd.DataFrame(truth).to_csv(out / "mixtures/true_fractions.tsv", sep="\t", index=False)
    json.dump({**meta, "n_mixtures": N_MIX, "depth": DEPTH, "seed": SEED, "alpha": 1.0,
               "mixture_types": list(mix_types)}, open(out / "design.json", "w"), indent=2)


def mixtures(ref, out):
    import build_reference as br
    out = Path(out)
    a = reference_cells(labelled(ref))
    types = a.obs["cell_type"].astype(str).values
    in_ref = half_split(types)
    r = br.clean_cells(a[in_ref].copy(), 20)
    br.export_reference(r, out / "reference")
    held = a[~in_ref].copy()
    n_ref = pd.Series(types[in_ref]).value_counts()
    n_held = pd.Series(types[~in_ref]).value_counts()
    mix_types = sorted(t for t in n_held.index if n_held[t] >= 20 and n_ref.get(t, 0) >= 20)
    X = sp.csr_matrix(held.X)
    (out / "mixtures").mkdir(parents=True, exist_ok=True)
    pd.Series(held.var_names).to_csv(out / "mixtures/pseudobulk_genes.tsv", index=False, header=False)
    draw_mixtures(X, held.obs["cell_type"].astype(str).values, mix_types, out, np.random.default_rng(SEED),
                  {"ref": ref, "split": "half of each cell type, seed 0",
                   "cells_reference_half": n_ref.to_dict(), "cells_mixture_half": n_held.to_dict()})
    print(f"wrote {out}: half-reference {r.n_obs} cells, {N_MIX} mixtures over {len(mix_types)} types")


def corr(t, e):
    return float(np.corrcoef(t, e)[0, 1]) if np.std(e) > 0 and np.std(t) > 0 else float("nan")


def score(stage):
    stage = Path(stage)
    truth = pd.read_csv(stage / "mixtures/true_fractions.tsv", sep="\t", index_col=0)
    est = pd.read_csv(stage / "results/estimated_fractions.csv", index_col=0).reindex(truth.index)
    rows = {}
    for c in truth.columns:
        t = c.split("__", 1)[1]
        e = est[t] if t in est.columns else pd.Series(0.0, index=truth.index)
        r = corr(truth[c], e)
        mean = float(truth[c].mean())
        rows[t] = {"mean_true": round(mean, 4), "mean_est": round(float(e.mean()), 4),
                   "pearson_r": round(r, 3), "mean_bias": round(float((e - truth[c]).mean()), 4),
                   "scored": mean >= 0.05, "pass": (r >= 0.9) if mean >= 0.05 else None}
    extra = [t for t in est.columns if f"rnafrac__{t}" not in truth.columns]
    res = {"per_type": rows, "estimated_mean_of_types_not_in_mixtures": {t: round(float(est[t].mean()), 4)
                                                                          for t in extra},
           "n_scored": sum(r["scored"] for r in rows.values()),
           "pass": all(r["pass"] for r in rows.values() if r["scored"])}
    json.dump(res, open(stage / "R2.json", "w"), indent=2)
    print(json.dumps(res, indent=2))


# ----------------------------------------------------------------------------- §15
def heart_map():
    return yaml.safe_load(open(jr.PANELS))["HEART_PRODUCTION_TO_COARSE"]


def heart_mixtures(out):
    out = Path(out)
    a = labelled("HEART_GSE307551")
    a = a[a.obs["cell_type"] != "DROP"].copy()
    coarse = set(heart_map().values()) | {"Adipocytes"}
    bad = set(a.obs["cell_type"]) - coarse
    assert not bad, f"labels outside the coarse classes: {bad}"
    vc = a.obs["cell_type"].value_counts()
    mix_types = sorted(vc.index[vc >= 50])
    (out / "mixtures").mkdir(parents=True, exist_ok=True)
    pd.Series(a.var_names).to_csv(out / "mixtures/pseudobulk_genes.tsv", index=False, header=False)
    draw_mixtures(sp.csr_matrix(a.X), a.obs["cell_type"].astype(str).values, mix_types, out,
                  np.random.default_rng(SEED),
                  {"source": "GSE307551 F344 libraries GSM9226643/4, all QC cells", "cells_per_class": vc.to_dict(),
                   "reference": "data/deconvolution/references_v3/HEART_GSE280111_LV",
                   "gene_map": {k: (v if isinstance(v, (int, str)) else str(v))
                                for k, v in a.uns.get("gene_map", {}).items()}})
    print(f"wrote {out}: {N_MIX} mixtures over {mix_types}")


def heart_score(stage):
    stage = Path(stage)
    truth = pd.read_csv(stage / "mixtures/true_fractions.tsv", sep="\t", index_col=0)
    est = pd.read_csv(stage / "results/estimated_fractions.csv", index_col=0).reindex(truth.index)
    m = heart_map()
    assert set(est.columns) <= set(m), f"unmapped production types: {set(est.columns) - set(m)}"
    est_c = est.T.groupby(est.columns.map(m)).sum().T
    rows = {}
    for c in sorted(set(est_c.columns) | {x.split("__", 1)[1] for x in truth.columns}):
        t = truth.get(f"rnafrac__{c}", pd.Series(0.0, index=truth.index))
        e = est_c[c] if c in est_c.columns else pd.Series(0.0, index=truth.index)
        rows[c] = {"in_mixtures": f"rnafrac__{c}" in truth.columns, "mean_true": round(float(t.mean()), 4),
                   "mean_est": round(float(e.mean()), 4), "pearson_r": round(corr(t, e), 3),
                   "mean_bias": round(float((e - t).mean()), 4),
                   "mean_abs_error": round(float((e - t).abs().mean()), 4)}
    cm = rows["Cardiomyocytes"]
    res = {"per_class": rows, "CM_r": cm["pearson_r"], "CM_mean_abs_error": cm["mean_abs_error"],
           "pass": bool(cm["pearson_r"] >= 0.9 and cm["mean_abs_error"] <= 0.10),
           "per_type_estimated_means": est.mean().round(4).to_dict()}
    est_c.to_csv(stage / "estimated_coarse_fractions.tsv", sep="\t")
    json.dump(res, open(stage / "s15_result.json", "w"), indent=2)
    print(json.dumps(res, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["checks", "build", "mixtures", "score", "heart-mixtures", "heart-score"])
    ap.add_argument("--ref", choices=["SPLEEN", "OVARY"])
    ap.add_argument("--out")
    ap.add_argument("--stage")
    a = ap.parse_args()
    {"checks": lambda: checks(a.ref), "build": lambda: build(a.ref, a.out),
     "mixtures": lambda: mixtures(a.ref, a.out), "score": lambda: score(a.stage),
     "heart-mixtures": lambda: heart_mixtures(a.out), "heart-score": lambda: heart_score(a.stage)}[a.command]()


if __name__ == "__main__":
    main()
