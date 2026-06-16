#!/usr/bin/env python3
"""corroborate_summary.py -- merge the corroboration outputs for the Aim-2 supervised finding:
does CANONICAL Augur (RF) reproduce our PLS-1 probe, and does the GeneCompass embedding beat the
PCA baseline?
  subspace_probe.tsv -> sup_trained_auc  (our PLS-1 CV AUC on the 768-d embedding) + perm p
  pca_control.tsv    -> auc_embed/auc_pca/auc_genes (PLS-1 CV AUC by representation)
  augur_results.tsv  -> augur_auc (canonical neurorestore Augur, RF) by (representation, condition)
Writes corroboration_merged.tsv.
"""
import os, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr, wilcoxon

sys.path.insert(0, str(Path(os.environ.setdefault(
    "PIPELINE_ROOT", "/depot/reese18/apps/motrpac-genecompass")) / "lib"))
from gene_utils import load_config, resolve_path                          # noqa: E402

cfg = load_config(); gc = str(resolve_path(cfg, cfg["deconvolution"]["genecompass_input_dir"]))
sp = pd.read_csv(f"{gc}/subspace_probe.tsv", sep="\t")
pc = pd.read_csv(f"{gc}/pca_control.tsv", sep="\t")
au = pd.read_csv(f"{gc}/augur_results.tsv", sep="\t")

aug = au.pivot_table(index=["tissue", "cell_type"], columns=["representation", "condition"],
                     values="augur_auc")
aug.columns = [f"augur_{r}_{c}" for r, c in aug.columns]
aug = aug.reset_index()

m = (sp[["tissue", "cell_type", "n", "sup_trained_auc", "p_sup_trained", "glob_trained", "p_glob_trained"]]
     .merge(pc[["tissue", "cell_type", "auc_embed", "auc_pca", "auc_genes", "p_embed"]],
            on=["tissue", "cell_type"], how="outer")
     .merge(aug, on=["tissue", "cell_type"], how="outer"))

# BH-FDR on our PLS-1 perm p (the FDR-significant hotspot set)
p = m["p_sup_trained"].dropna(); idx = p.index; o = np.argsort(p.values); ml = len(p)
q = (p.values[o] * ml) / np.arange(1, ml + 1); q = np.minimum.accumulate(q[::-1])[::-1]
qq = np.empty(ml); qq[o] = np.clip(q, 0, 1); m.loc[idx, "q_sup_trained"] = qq

def corr(a, b, label):
    if a not in m or b not in m:
        print(f"  {label}: MISSING col"); return
    s = m[[a, b]].dropna()
    r = spearmanr(s[a], s[b]).correlation if len(s) > 2 else np.nan
    print(f"  {label:46s} Spearman r={r:.3f}  n={len(s)}  medians {s[a].median():.3f} / {s[b].median():.3f}")

print("=== METHOD ROBUSTNESS: canonical Augur (RF) vs our PLS-1 probe (trained, 768-d embedding) ===")
corr("augur_embed_trained", "sup_trained_auc", "Augur-RF vs PLS-1 (subspace_probe)")
corr("augur_embed_trained", "auc_embed", "Augur-RF vs PLS-1 (pca_control)")

print("\n=== REPRESENTATION: does the embedding beat PCA-50? (each method) ===")
for ce, cp, meth in [("auc_embed", "auc_pca", "PLS-1 "), ("augur_embed_trained", "augur_pca_trained", "Augur-RF")]:
    if ce in m and cp in m:
        s = m[[ce, cp]].dropna(); d = s[ce] - s[cp]
        w = wilcoxon(s[ce], s[cp]).pvalue if len(d) > 1 else np.nan
        print(f"  {meth}: median embed={s[ce].median():.3f} pca={s[cp].median():.3f} "
              f"Δ={d.median():+.3f}  embed>pca {int((d > 0).sum())}/{len(d)}  Wilcoxon p={w:.2g}")

print("\n=== SEX positive control (Augur should be high & near PLS-1) ===")
for r in ["embed", "pca"]:
    c = f"augur_{r}_sex"
    if c in m:
        print(f"  Augur {r} sex: median AUC {m[c].median():.3f}  max {m[c].max():.3f}")

print("\n=== concordance on the FDR-significant exercise hotspots (subspace_probe q<0.05) ===")
hot = m[m["q_sup_trained"] < 0.05]; nonhot = m[m["q_sup_trained"] >= 0.05]
if "augur_embed_trained" in m:
    print(f"  FDR-sig blocks: {len(hot)};  Augur embed AUC median on them = "
          f"{hot['augur_embed_trained'].median():.3f}  vs non-sig = {nonhot['augur_embed_trained'].median():.3f}")
    print(f"  of FDR-sig blocks, Augur embed AUC>=0.65: {int((hot['augur_embed_trained'] >= 0.65).sum())}/{len(hot)}"
          f", >=0.70: {int((hot['augur_embed_trained'] >= 0.70).sum())}/{len(hot)}")

print("\n=== top exercise responders (by PLS-1 embed AUC), all methods side by side ===")
cols = [c for c in ["tissue", "cell_type", "n", "glob_trained", "sup_trained_auc", "p_sup_trained",
        "augur_embed_trained", "auc_pca", "augur_pca_trained", "auc_genes"] if c in m]
print(m.dropna(subset=["sup_trained_auc"]).sort_values("sup_trained_auc", ascending=False)[cols]
      .head(22).to_string(index=False))

m.to_csv(f"{gc}/corroboration_merged.tsv", sep="\t", index=False)
print(f"\nwrote {gc}/corroboration_merged.tsv ({len(m)} rows)")
