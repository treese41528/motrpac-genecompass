#!/usr/bin/env python3
"""build_blood_composite.py -- the BLOOD option-B reference (REFERENCE_RELABEL_PREREG.md §12).

PBMC types (lymphocytes, monocytes, dendritic cells, platelets) come from the five venous GSE335156 rats;
neutrophils and erythroid cells come from GSE306764 Blood-CFA (whole blood). Both were labelled with the
BLOOD panel by joint_reannotate.py (BLOOD_PBMC, BLOOD_WB, BLOOD_BM) and finalized.

  checks     §12 checks 1-5 on the finalized labels -> data/deconvolution/blood_option_b/checks.json
  build      the composite reference (all five rats, all whole-blood cells) -> --out
  mixtures   check 6: a reference from rats V1-V3 + half of the whole-blood cells, and 20 mixtures with
             known mRNA fractions summed from rats V4-V5 + the other half -> --out/{reference,mixtures}
  score      check 6 scoring of BayesPrism's estimated_fractions.csv against true_fractions.tsv

Usage (project venv; loads ~90k cells -> SLURM for build/mixtures):
  python deconvolution/build_blood_composite.py checks
  python deconvolution/build_blood_composite.py build --out data/deconvolution/references_v4/BLOOD_composite
  python deconvolution/build_blood_composite.py mixtures --out data/deconvolution/validation/BLOOD_composite_check6
  python deconvolution/build_blood_composite.py score --stage data/deconvolution/validation/BLOOD_composite_check6
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

ROOT = Path(os.environ.setdefault("PIPELINE_ROOT", str(Path(__file__).resolve().parents[1])))
sys.path.insert(0, str(ROOT / "deconvolution"))
import joint_reannotate as jr  # noqa: E402

LABELS = ROOT / "data/deconvolution/joint_labels"
OUT = ROOT / "data/deconvolution/blood_option_b"
FROM_WB = ["Neutrophils", "Erythroid cells", "Eosinophils", "Basophils"]      # only from whole blood
NOT_FROM_PBMC = set(FROM_WB) | {"DROP"}
SEED = 0
# check 6 groups: a scored type is the sum of the reference types listed
GROUPS = {"Erythroid cells": ["Erythroid cells"], "Neutrophils": ["Neutrophils"],
          "Monocytes": ["Classical monocytes", "Non-classical monocytes"],
          "T cells": ["CD4+ T cells", "CD8+ T cells", "Regulatory T cells", "T cells"],
          "B cells": ["B cells"], "NK cells": ["NK cells"]}
NON_GLOBIN_ERY = ["Alas2", "Slc4a1", "Epb42", "Car2", "Slc25a37", "Ahsp"]


def labelled(ref):
    """Raw counts (QC'd, bulk ID space) of a BLOOD set joined to its final labels."""
    a = jr.load_blood(ref)
    lab = pd.read_csv(LABELS / ref / "cells_final.tsv.gz", sep="\t", dtype=str, keep_default_na=False)
    lab.index = lab["sample"] + ":" + lab["barcode"]
    key = (a.obs["sample"].astype(str) + ":" + a.obs["barcode"].astype(str)).values
    assert set(key) == set(lab.index), f"{ref}: cells and labels differ"
    a.obs["cell_type"] = lab.loc[key, "cell_type"].values
    a.obs["joint_cluster"] = lab.loc[key, "joint_cluster"].values
    a.obs_names = key
    return a


def cluster_cpm(ref, genes):
    """Pooled CPM of named genes per joint cluster, from group_sums.npz."""
    z = np.load(LABELS / ref / "group_sums.npz", allow_pickle=True)
    g = pd.DataFrame([x.split("\t") for x in z["groups"]], columns=["s", "c"])
    cats, M = jr.onehot(g["c"].values)
    S = np.asarray(M @ z["sums"])
    cpm = S / S.sum(1, keepdims=True) * 1e6
    idx = jr.symbol_index(z["genes"])
    gi = {x: i for i, x in enumerate(z["genes"])}
    out = {}
    for s in genes:
        ids = [gi[i] for i in idx.get(s.lower(), ()) if i in gi]
        out[s] = cpm[:, ids].sum(1) if ids else np.full(len(cats), np.nan)
    return pd.DataFrame(out, index=cats)


def checks():
    OUT.mkdir(parents=True, exist_ok=True)
    res = {}
    cl = {r: pd.read_csv(LABELS / r / "clusters_final.tsv", sep="\t", index_col=0, keep_default_na=False)
          for r in ("BLOOD_PBMC", "BLOOD_WB", "BLOOD_BM")}
    # 1. composition of the GSE306764 libraries
    for r in ("BLOOD_WB", "BLOOD_BM"):
        res[f"check1_{r}_cells_per_label"] = cl[r].groupby("final_label")["n_cells"].sum().astype(int).to_dict()
    n_neu = res["check1_BLOOD_WB_cells_per_label"].get("Neutrophils", 0)
    res["check1_neutrophils_in_blood"] = int(n_neu)
    res["check1_pass"] = bool(n_neu >= 100)
    # 2. the erythroid cluster(s) of Blood-CFA are cells, not ambient RNA
    wb = cl["BLOOD_WB"]
    ery = wb.index[wb["final_label"] == "Erythroid cells"].tolist()
    cpm = cluster_cpm("BLOOD_WB", NON_GLOBIN_ERY)
    ng = cpm[NON_GLOBIN_ERY].sum(1)
    others = ng.drop(index=ery)
    res["check2"] = {"erythroid_clusters": ery, "n_cells": int(wb.loc[ery, "n_cells"].sum()),
                     "median_genes": [float(x) for x in wb.loc[ery, "median_n_genes"]],
                     "nonglobin_erythroid_cpm": {c: round(float(ng[c]), 1) for c in ery},
                     "max_other_cluster": [others.idxmax(), round(float(others.max()), 1)],
                     "ratio_to_max_other": round(float(ng[ery].min() / others.max()), 2),
                     "per_gene_cpm": cpm.loc[ery].round(1).to_dict(orient="index")}
    c2 = res["check2"]
    res["check2_pass"] = bool(c2["n_cells"] >= 50 and c2["ratio_to_max_other"] >= 4
                              and min(c2["median_genes"]) >= 100)
    # 3. neutrophil activation against the PBMC monocytes (reported, no threshold)
    ifn, act = ["Ifit1", "Ifit3", "Isg15", "Rsad2"], ["Il1b", "Cxcl2", "Nfkbia"]
    wbc, pbc = labelled("BLOOD_WB"), labelled("BLOOD_PBMC")

    def score(a, genes):
        idx = jr.symbol_index(a.var_names)
        cols = sorted({g for s in genes for g in idx.get(s.lower(), ())})
        x = a.X[:, a.var_names.get_indexer(cols)]
        lib = np.asarray(a.X.sum(1)).ravel()
        return np.asarray(np.log1p(x.multiply(1e4 / lib[:, None]).tocsr()).mean(1)).ravel(), cols
    out3 = {}
    for name, genes in (("interferon", ifn), ("activation", act)):
        s_wb, cols = score(wbc, genes)
        s_pb, _ = score(pbc, genes)
        mono = pbc.obs["cell_type"].isin(GROUPS["Monocytes"]).values
        neu = (wbc.obs["cell_type"] == "Neutrophils").values
        thr = float(np.percentile(s_pb[mono], 95))
        out3[name] = {"genes_found": len(cols), "monocyte_p95": round(thr, 3),
                      "share_of_neutrophils_above": round(float((s_wb[neu] > thr).mean()), 3),
                      "per_neutrophil_cluster": {c: round(float((s_wb[(wbc.obs['joint_cluster'] == c).values]
                                                                 > thr).mean()), 3)
                                                 for c in sorted(wbc.obs.loc[neu, "joint_cluster"].unique())}}
    res["check3_neutrophil_activation"] = out3
    # 4. every PBMC type in >= 4 of 5 rats
    lc = pd.read_csv(LABELS / "BLOOD_PBMC" / "label_checks.tsv", sep="\t", index_col=0)
    used = lc.drop(index=[l for l in lc.index if l in NOT_FROM_PBMC])
    res["check4_pbmc_types"] = {l: {"n_rats": int(r.n_samples), "n_cells": int(r.n_cells),
                                    "r_self": round(float(r.r_self), 3)} for l, r in used.iterrows()}
    res["check4_pass"] = bool((used["n_samples"] >= 4).all())
    # 5. gene space
    shared = pbc.var_names.intersection(wbc.var_names)
    bulk = pd.read_csv(ROOT / "data/deconvolution/motrpac_bulk/BLOOD/bulk_genes.tsv", sep="\t", header=None)[0]
    res["check5_gene_space"] = {"pbmc_genes": int(pbc.n_vars), "wholeblood_genes_after_liftover": int(wbc.n_vars),
                                "shared": int(len(shared)), "shared_and_in_bulk": int(shared.isin(set(bulk)).sum()),
                                "bulk_genes": int(len(bulk))}
    json.dump(res, open(OUT / "checks.json", "w"), indent=2)
    print(json.dumps({k: v for k, v in res.items() if "pass" in k or k.startswith("check5")
                      or k == "check3_neutrophil_activation"}, indent=2))


def composite(pbmc, wb):
    """PBMC types from `pbmc`, whole-blood-only types from `wb`; genes in both; cell states prefixed by source."""
    import anndata as ad
    p = pbmc[~pbmc.obs["cell_type"].isin(NOT_FROM_PBMC).values].copy()
    w = wb[wb.obs["cell_type"].isin(FROM_WB).values].copy()
    for a, src in ((p, "PBMC"), (w, "WB")):
        a.obs["source"] = src
        a.obs["cell_state"] = src + "_" + a.obs["joint_cluster"].astype(str)
        a.obs["leiden"] = a.obs["joint_cluster"].astype(str)
    out = ad.concat([p, w], join="inner", merge="same")
    out.obs_names_make_unique()
    return out


def build(out):
    import build_reference as br
    a = composite(labelled("BLOOD_PBMC"), labelled("BLOOD_WB"))
    a = br.clean_cells(a, 20)
    br.export_reference(a, out)
    a.obs[["barcode", "sample", "source", "cell_type", "cell_state"]].to_csv(Path(out) / "cells_source.tsv",
                                                                           sep="\t", index=False)


def half_split(a, seed=SEED):
    """Within each joint cluster, a random half of the cells -> True."""
    rng = np.random.default_rng(seed)
    m = np.zeros(a.n_obs, bool)
    for c in a.obs["joint_cluster"].unique():
        ii = np.flatnonzero((a.obs["joint_cluster"] == c).values)
        m[rng.choice(ii, size=len(ii) // 2, replace=False)] = True
    return m


def mixtures(out, n_mix=20, depth=5e6):
    import build_reference as br
    out = Path(out)
    pb, wb = labelled("BLOOD_PBMC"), labelled("BLOOD_WB")
    ref_rats = {"GSE335156_V1", "GSE335156_V2", "GSE335156_V3"}
    in_ref_pb = pb.obs["sample"].isin(ref_rats).values
    in_ref_wb = half_split(wb)
    ref = br.clean_cells(composite(pb[in_ref_pb], wb[in_ref_wb]), 20)
    br.export_reference(ref, out / "reference")
    held = composite(pb[~in_ref_pb], wb[~in_ref_wb])
    held = held[:, ref.var_names].copy()
    X = sp.csr_matrix(held.X)
    types = sorted(held.obs["cell_type"].unique())
    pbmc_types = [t for t in types if t not in FROM_WB]
    libs = np.asarray(X.sum(1)).ravel()
    tot = {t: libs[(held.obs["cell_type"] == t).values].sum() for t in types}
    pbmc_share = pd.Series({t: tot[t] for t in pbmc_types})
    pbmc_share /= pbmc_share.sum()                       # observed mRNA ratios among held-out PBMC types
    rng = np.random.default_rng(SEED)
    rows, truth = [], []
    for k in range(n_mix):
        e, n = rng.uniform(0, 0.6), rng.uniform(0, 0.4)
        target = {"Erythroid cells": e, "Neutrophils": n, **(pbmc_share * (1 - e - n)).to_dict()}
        mix = np.zeros(X.shape[1])
        got = {}
        for t, f in target.items():
            ii = np.flatnonzero((held.obs["cell_type"] == t).values)
            need, s = f * depth, 0.0
            vec = np.zeros(X.shape[1])
            while s < need and len(ii):                  # draw cells (with replacement) up to the target reads
                draw = rng.choice(ii, size=max(1, min(200, int((need - s) / max(libs[ii].mean(), 1)) + 1)))
                vec += np.asarray(X[draw].sum(0)).ravel()
                s += libs[draw].sum()
            mix += vec
            got[t] = s
        tot_reads = sum(got.values())
        rows.append(mix)
        truth.append({"mixture": f"mix{k + 1}", **{f"rnafrac__{t}": got[t] / tot_reads for t in got}})
    (out / "mixtures").mkdir(parents=True, exist_ok=True)
    sio.mmwrite(str(out / "mixtures/pseudobulk_counts.mtx"), sp.csr_matrix(np.round(np.vstack(rows))).astype(np.int64))
    pd.Series(ref.var_names).to_csv(out / "mixtures/pseudobulk_genes.tsv", index=False, header=False)
    pd.DataFrame(truth).to_csv(out / "mixtures/true_fractions.tsv", sep="\t", index=False)
    json.dump({"ref_rats": sorted(ref_rats), "held_rats": sorted(set(pb.obs["sample"]) - ref_rats),
               "wholeblood_split": "half of each joint cluster, seed 0", "n_mixtures": n_mix, "depth": depth,
               "erythroid_range": [0, 0.6], "neutrophil_range": [0, 0.4],
               "pbmc_mrna_ratios": pbmc_share.round(4).to_dict()}, open(out / "design.json", "w"), indent=2)
    print(f"wrote {out}: reference {ref.n_obs} cells, {n_mix} mixtures")


def score(stage):
    stage = Path(stage)
    truth = pd.read_csv(stage / "mixtures/true_fractions.tsv", sep="\t", index_col=0)
    res = {}
    for arm in sorted(p.name for p in stage.iterdir() if p.is_dir() and p.name.startswith("results")):
        est = pd.read_csv(stage / arm / "estimated_fractions.csv", index_col=0)
        rows = {}
        for g, members in GROUPS.items():
            t = truth[[f"rnafrac__{m}" for m in members if f"rnafrac__{m}" in truth]].sum(1)
            e = est[[m for m in members if m in est.columns]].sum(1).reindex(t.index)
            rows[g] = {"pearson_r": round(float(np.corrcoef(t, e)[0, 1]), 3),
                       "mean_bias": round(float((e - t).mean()), 4), "pass": bool(np.corrcoef(t, e)[0, 1] >= 0.9)}
        res[arm] = {"per_type": rows, "pass": all(r["pass"] for r in rows.values())}
    json.dump(res, open(stage / "check6.json", "w"), indent=2)
    print(json.dumps(res, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["checks", "build", "mixtures", "score"])
    ap.add_argument("--out")
    ap.add_argument("--stage")
    a = ap.parse_args()
    if a.command == "checks":
        checks()
    elif a.command == "build":
        build(a.out)
    elif a.command == "mixtures":
        mixtures(a.out)
    else:
        score(a.stage)


if __name__ == "__main__":
    main()
