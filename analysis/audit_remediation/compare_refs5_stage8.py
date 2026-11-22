#!/usr/bin/env python
"""compare_refs5_stage8.py -- refs5 against the current default at Stage 8 (STEP3_PREREG.md section 10, items 2-3).

Per tissue and cell type: the estimated fraction in every sample under both runs (Pearson r across samples, mean
absolute difference, means), for the types both runs have; the types only one run has; the number of genes in the
deconvolution and how many entered or left; and, from each run's provenance, the commit, whether the tree was
clean, the exclusion list and BayesPrism's settings. Read-only.

Usage: compare_refs5_stage8.py <old_results_root> <new_results_root> <out_dir>
  e.g. data/deconvolution/results_refs4all_rebuild/motrpac data/deconvolution/results_refs5/motrpac \
       docs/audit/remediation/record/step3
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

old_root, new_root, out = [Path(p) for p in sys.argv[1:4]]
out.mkdir(parents=True, exist_ok=True)
rows, trows, prov = [], [], []
for nd in sorted(p for p in new_root.iterdir() if (p / "estimated_fractions.csv").exists()):
    t = nd.name; od = old_root / t
    new = pd.read_csv(nd / "estimated_fractions.csv", index_col=0)
    old = pd.read_csv(od / "estimated_fractions.csv", index_col=0) if (od / "estimated_fractions.csv").exists() else None
    gn = [l.strip() for l in open(nd / "pred_z" / "genes.txt")] if (nd / "pred_z" / "genes.txt").exists() else []
    go = [l.strip() for l in open(od / "pred_z" / "genes.txt")] if (od / "pred_z" / "genes.txt").exists() else []
    p = json.load(open(nd / "provenance_run_deconvolution.json")) if (nd / "provenance_run_deconvolution.json").exists() else {}
    pp = p.get("params", {})
    prov.append({"tissue": t, "git_commit": p.get("git_commit", ""), "n_modified_tracked_files": len(p.get("git_modified_tracked_files", []) or []),
                 "exclude_list": Path(str(pp.get("exclude_list", ""))).name, "exclude_list_md5": pp.get("exclude_list_md5", ""),
                 "n_excluded": pp.get("n_excluded_ribo_mito_hb"), "n_ref_cells": pp.get("n_ref_cells"), "n_ref_genes": pp.get("n_ref_genes"),
                 "gibbs_control": json.dumps(pp.get("gibbs_control", {}), sort_keys=True), "opt_control": json.dumps(pp.get("opt_control", {}), sort_keys=True),
                 "samples_file_md5": pp.get("samples_file_md5", ""), "has_sample_key": (nd / "mixture_samples.tsv").exists(),
                 "run_prism_minutes": pp.get("run_prism_minutes")})
    trow = {"tissue": t, "n_samples": len(new), "types_new": new.shape[1], "types_old": old.shape[1] if old is not None else np.nan,
            "genes_new": len(gn), "genes_old": len(go), "genes_entered": len(set(gn) - set(go)), "genes_left": len(set(go) - set(gn))}
    if old is not None:
        both = [c for c in new.columns if c in old.columns]
        trow.update(types_both=len(both), types_only_new=";".join(c for c in new.columns if c not in old.columns),
                    types_only_old=";".join(c for c in old.columns if c not in new.columns))
        ad = []
        for c in both:
            a, b = old.loc[new.index, c].to_numpy(float), new[c].to_numpy(float)
            r = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else np.nan
            rows.append({"tissue": t, "cell_type": c, "mean_old": a.mean(), "mean_new": b.mean(), "mean_abs_diff": np.abs(a - b).mean(),
                         "max_abs_diff": np.abs(a - b).max(), "pearson_r": r})
            ad.append(np.abs(a - b).mean())
        trow.update(median_r=float(np.nanmedian([x["pearson_r"] for x in rows if x["tissue"] == t])) if both else np.nan,
                    max_mean_abs_diff=float(max(ad)) if ad else np.nan,
                    sum_mean_abs_diff=float(sum(ad)) if ad else np.nan)
        for c in new.columns:
            if c not in old.columns:
                rows.append({"tissue": t, "cell_type": c, "mean_old": np.nan, "mean_new": float(new[c].mean()), "mean_abs_diff": np.nan,
                             "max_abs_diff": np.nan, "pearson_r": np.nan})
    trows.append(trow)
pd.DataFrame(rows).to_csv(out / "fractions_refs5_vs_default_by_type.tsv", sep="\t", index=False)
T = pd.DataFrame(trows); T.to_csv(out / "fractions_refs5_vs_default_by_tissue.tsv", sep="\t", index=False)
P = pd.DataFrame(prov); P.to_csv(out / "stage8_provenance_refs5.tsv", sep="\t", index=False)
pd.set_option("display.width", 250)
print(T[["tissue", "types_old", "types_new", "genes_old", "genes_new", "genes_entered", "genes_left", "median_r", "max_mean_abs_diff"]].round(4).to_string(index=False))
print("\ncommits:", P.git_commit.value_counts().to_dict(), "| tissues with a dirty tree:", int((P.n_modified_tracked_files > 0).sum()),
      "| exclusion lists:", P.exclude_list.value_counts().to_dict(), "| sample key written:", int(P.has_sample_key.sum()), "of", len(P))
print("gibbs.control values:", P.gibbs_control.value_counts().to_dict())
