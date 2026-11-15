#!/usr/bin/env python
"""make_called_blocks_table.py -- every hotspot and borderline block of the refs5 gate against every registered reading.

One row per called block, from the record tables of Steps 3 and 4:
  gate          class, AUC, permutations used, interval of p
  S1, S2        sex-stratified q; 8-week-against-control AUC and q            (STEP3_PREREG.md 5.3)
  S3            all-genes AUC and the reading rule (supported / neither / model-only)
  S4            mean fraction and the low-fraction tier
  reference     cells of the type in its reference, libraries they come from
  V1            purity sweep: median correlation at the grid point nearest the type's abundance (STEP4_PREREG.md V1)
  transfer      registered status, corrected rule, dose trend                 (STEP3_PREREG.md 7)
  DE            dose-significant genes, covariate-rerun verdict, composition verdict (STEP3_PREREG.md 6)

Usage: make_called_blocks_table.py   (run from the repository root; reads and writes under docs/audit/remediation/record)
"""
from pathlib import Path

import numpy as np
import pandas as pd

R3 = Path("docs/audit/remediation/record/step3/downstream"); R4 = Path("docs/audit/remediation/record/step4")
flag = lambda s: s.astype(str).str.strip().str.upper().isin(["TRUE", "1", "PASS"])

g = pd.read_csv(R3 / "gate_called_blocks.tsv", sep="\t")
g = g[g.gate_class.isin(["hotspot", "borderline"])].copy()
ref = pd.read_csv(R3 / "gate_called_blocks_reference_cells.tsv", sep="\t")
t = pd.read_csv(R3 / "refs5_transfer_comparison.tsv", sep="\t"); t["tissue"] = t.tissue.str.upper()
rob = pd.read_csv(R3 / "refs5_de_covariate_robustness.tsv", sep="\t")
conf = pd.read_csv(R3 / "refs5_composition_confound_table.tsv", sep="\t")
pts = pd.read_csv(R4 / "v1_purity_sweeps/sweep_points.tsv", sep="\t"); plan = pd.read_csv(R4 / "v1_purity_sweeps/sweep_plan.tsv", sep="\t")
ver = pd.read_csv(R4 / "v1_purity_sweeps/sweep_verdicts.tsv", sep="\t")
hot = plan[plan.kind == "hotspot"]
K = ["tissue", "cell_type"]
out = g[K + ["gate_class", "sup_trained_auc", "sup_trained_auc_default", "gate_class_default", "m_sup_trained", "p_sup_trained",
             "p_lo_sup_trained", "p_hi_sup_trained", "q_sup_trained_s1", "sup_trained_auc_s2", "q_sup_trained_s2", "auc_genes", "auc_pca",
             "support", "mean_fraction", "low_fraction"]].copy()
out = out.merge(ref[K + ["reference", "reference_cells_of_type", "libraries_with_the_type", "reference_libraries", "median_libsize"]], on=K, how="left")
v1 = []
for r in out.itertuples():
    refdir = ref.loc[(ref.tissue == r.tissue) & (ref.cell_type == r.cell_type), "reference"].iloc[0]
    h = hot[hot.reference_dir.str.endswith(refdir) & (hot.focal == r.cell_type)]
    row = {"tissue": r.tissue, "cell_type": r.cell_type, "v1_sweep": h.tag.iloc[0] if len(h) else ""}
    p = pts[pts.tag == row["v1_sweep"]] if len(h) else pts.iloc[0:0]
    if len(p):
        near = p.iloc[(p.target_purity - r.mean_fraction).abs().argsort().iloc[0]]
        row.update(v1_purity_nearest=float(near.target_purity), v1_median_r_there=float(near.pearson_vst_median),
                   v1_validated_at_abundance=bool(near.pearson_vst_median >= 0.95),
                   v1_lowest_purity_with_r_095=float(p[p.pearson_vst_median >= 0.95].target_purity.min()) if (p.pearson_vst_median >= 0.95).any() else np.nan,
                   v1_pass_rule=ver.loc[ver.tag == row["v1_sweep"], "status"].iloc[0])
    else:
        row.update(v1_pass_rule="not testable", v1_validated_at_abundance=np.nan)
    v1.append(row)
out = out.merge(pd.DataFrame(v1), on=K, how="left")
out = out.merge(t[K + ["sup_trained_auc_human", "status", "q_sup_trained_human", "transfer_corrected", "sup_dose_rho_rat", "sup_dose_rho_human",
                       "dose_trend_lost"]].rename(columns={"status": "transfer_registered"}), on=K, how="left")
out = out.merge(rob[K + ["n_sig_headline", "retained_share_A", "spearman_neglog10p_A", "retained_share_B", "spearman_neglog10p_B", "verdict"]]
                .rename(columns={"verdict": "de_covariate_verdict"}), on=K, how="left")
out = out.merge(conf[K + ["frac_week_p", "verdict"]].rename(columns={"verdict": "composition_verdict"}), on=K, how="left")
out["s1_significant"] = out.q_sup_trained_s1 < 0.05
out["s2_significant"] = out.q_sup_trained_s2 < 0.05
out["not_low_fraction"] = ~flag(out.low_fraction)
out["transfers_corrected_rule"] = flag(out.transfer_corrected)
out["dose_trend_kept"] = ~flag(out.dose_trend_lost)
out["de_covariate_robust"] = out.de_covariate_verdict == "ROBUST"
out["composition_pass"] = out.composition_verdict == "PASS_EXPRESSION"
out = out.sort_values(["gate_class", "p_sup_trained"], ascending=[False, True]).reset_index(drop=True)
out.to_csv(R3 / "called_blocks_all_readings.tsv", sep="\t", index=False)
cols = ["tissue", "cell_type", "gate_class", "s1_significant", "s2_significant", "support", "not_low_fraction", "v1_median_r_there",
        "v1_validated_at_abundance", "transfers_corrected_rule", "dose_trend_kept", "de_covariate_robust", "composition_pass"]
pd.set_option("display.width", 250); print(out[cols].round(3).to_string(index=False))
checks = ["s1_significant", "s2_significant", "not_low_fraction", "v1_validated_at_abundance", "transfers_corrected_rule", "dose_trend_kept",
          "de_covariate_robust", "composition_pass"]
n = out[checks].apply(lambda c: c.fillna(False).astype(bool)).sum(axis=1) + (out.support == "supported")
print("\nreadings passed of 9, per block:", dict(zip(out.tissue + " " + out.cell_type, n)))
