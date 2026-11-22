#!/usr/bin/env python3
"""docs_refs5_preprint_numbers.py -- the docs pass after refs5 (docs/DOCS_REFS5_UPDATE.md, section 5).

Numbers the preprint quotes that the remediation results files do not print: the trace screen, the per-tissue
regime table, per-block DE counts, the positive-control detail, the transfer status counts, the network and
mixture-panel summaries, and the atlas accuracies of the cell types now called. Each is read from the tables
of the configured default (refs5 when this was written): the same tables the results files and notebooks
8-12 read. Read-only: it prints and writes nothing. Seconds; TSV files only.

  python analysis/audit_remediation/docs_refs5_preprint_numbers.py
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "lib"))
os.environ.setdefault("PIPELINE_ROOT", str(ROOT))
from gene_utils import load_config, resolve_path   # noqa: E402

_CFG = load_config(str(ROOT / "config/pipeline_config.yaml"))
M = str(ROOT)
G = str(resolve_path(_CFG, _CFG["deconvolution"]["genecompass_input_dir"]))
GH = G + "_human"
REC = f"{M}/docs/audit/remediation/record"
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 300); pd.set_option("display.max_columns", 40)


def bh(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p); q = np.empty(n)
    q[o] = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    return q


print("=== trace screen (pheno_merge_test.tsv) and gate")
g = pd.read_csv(f"{G}/pheno_merge_test.tsv", sep="\t")
m = pd.read_csv(f"{G}/gate_rcv10/corroboration_merged.tsv", sep="\t")
sp = pd.read_csv(f"{G}/gate_rcv10/subspace_probe.tsv", sep="\t")
print("blocks in pheno_merge_test:", len(g), "| corroboration:", len(m), "| probe:", len(sp))
for f in ("trained", "sex", "group"):
    x = g.dropna(subset=[f"p_{f}"])
    print(f"  {f}: sig p<0.05 {int((x[f'p_{f}'] < 0.05).sum())}/{len(x)}; median eta2 {x[f'eta2_{f}'].median():.3f}; "
          f"BH q<0.05 over all tested: {int((bh(x[f'p_{f}']) < 0.05).sum())}")
ga = g.merge(m[["tissue", "cell_type", "analysable"]], on=["tissue", "cell_type"])
xa = ga[ga.analysable]
print("  analysable only: trained BH q<0.05:", int((bh(xa.p_trained) < 0.05).sum()), "of", len(xa),
      "| glob_trained from probe, BH q<0.05 (analysable):",
      int((bh(sp.merge(m[['tissue','cell_type','analysable']]).query('analysable').p_glob_trained) < 0.05).sum()))
print("  std_trained median (all blocks):", round(sp.std_trained.median(), 3), "glob_trained median:", round(sp.glob_trained.median(), 3))
a = m[m.analysable]
hot = m[m.is_hotspot == True]
print("  max AUC analysable:", a.sort_values("sup_trained_auc").iloc[-1][["tissue", "cell_type", "sup_trained_auc"]].tolist(),
      "| hotspot AUC range:", hot.sup_trained_auc.min(), hot.sup_trained_auc.max(), "| median AUC analysable:", round(a.sup_trained_auc.median(), 3))
for t in ("cortex", "hippoc", "hypoth", "smlint", "testes"):
    print(f"  {t} median AUC (analysable): {a[a.tissue == t].sup_trained_auc.median():.3f}")
print("  Augur min in hotspots:", hot.augur_embed_trained.min(), "| Augur medians embed/pca (analysable):",
      round(a.augur_embed_trained.median(), 3), round(a.augur_pca_trained.median(), 3))
near = a[(a.gate_class == "none")].sort_values("p_sup_trained").head(8)
print("  not called, smallest p:\n", near[["tissue", "cell_type", "sup_trained_auc", "p_sup_trained", "q_sup_trained"]].to_string(index=False))
print("  AUC 0.70-0.80 among analysable, not hotspot:", int(((a.sup_trained_auc >= 0.70) & (a.sup_trained_auc < 0.80) & (a.is_hotspot != True)).sum()))

print("\n=== per-tissue regimes (Table buckets)")
rows = []
for t, x in g.groupby("tissue"):
    mm = m[m.tissue == t]
    r = {"tissue": t, "nCT": len(x), "nAn": int(mm.analysable.sum()),
         "trained": f"{int((x.p_trained < 0.05).sum())}/{len(x)}; {x.eta2_trained.median():.3f}/{x.eta2_trained.max():.3f}",
         "sex": ("n/a" if x.p_sex.isna().all() else f"{int((x.p_sex < 0.05).sum())}/{len(x)}; {x.eta2_sex.median():.3f}/{x.eta2_sex.max():.3f}"),
         "group": f"{int((x.p_group < 0.05).sum())}/{len(x)}",
         "hot": int((mm.is_hotspot == True).sum()), "border": int((mm.gate_class == "borderline").sum())}
    rows.append(r)
print(pd.DataFrame(rows).to_string(index=False))

print("\n=== mean fractions of named types (refs5)")
fr = pd.read_csv(f"{REC}/step3/stage8/fractions_refs5_vs_default_by_type.tsv", sep="\t")
print(list(fr.columns))
fc = "mean_new"
for t, pat in (("HIPPOC", "Endothelial|Excitatory"), ("KIDNEY", "onocyte"), ("BLOOD", "."), ("SPLEEN", "T cells|Endothelial"),
               ("HYPOTH", "."), ("TESTES", "Sertoli")):
    x = fr[(fr.tissue.str.upper() == t) & fr.cell_type.str.contains(pat)]
    print(t, "; ".join(f"{c} {v:.4f}" for c, v in zip(x.cell_type, x[fc])))

print("\n=== DE")
de = pd.read_csv(f"{G}/pseudobulk_de/de_summary.tsv", sep="\t")
ok = de[de.status == "ok"]; sk = de[de.status != "ok"]
print("status:", de.status.value_counts().to_dict(), "| skipped median library range:", round(sk.median_libsize.min(), 1), round(sk.median_libsize.max(), 1))
meth = dict(l.rstrip("\n").split("\t", 1) for l in open(f"{G}/pseudobulk_de/de_methods.tsv") if "\t" in l)
print("n_fisher_tests:", meth.get("n_fisher_tests"), "| single-sex dose tests:", meth.get("n_dose_tests_single_sex"))
for lib in (100, 0):
    f = f"{G}/pseudobulk_de_minlib{lib}/de_summary.tsv"
    if os.path.exists(f):
        d2 = pd.read_csv(f, sep="\t"); print(f"  minlib{lib}: pairs {int(d2[d2.status == 'ok'].n_sig_dose_IHW.sum()):,} over {int((d2.status == 'ok').sum())} blocks")
hl = pd.read_csv(f"{G}/pseudobulk_de/de_headline.tsv", sep="\t"); print(hl[hl.scope.isin(["ALL", "BLOOD", "SPLEEN"])].to_string(index=False))
cols = ["cell_type", "n_sig_dose_IHW", "n_sig_interaction", "gate_class", "mean_fraction", "sup_trained_auc", "frac_week_p"]
for t in ("BLOOD", "HEART", "SKMGN", "SKMVL", "LUNG", "BAT", "WATSC", "SPLEEN", "TESTES", "OVARY", "LIVER", "KIDNEY", "CORTEX", "HIPPOC", "HYPOTH", "SMLINT"):
    x = ok[ok.tissue == t].sort_values("n_sig_dose_IHW", ascending=False)
    print(f"-- {t}: blocks {len(x)}, pairs {int(x.n_sig_dose_IHW.sum())}, interaction total {int(x.n_sig_interaction.sum())}")
    print(x[cols].head(6 if t not in ("SKMGN", "SKMVL", "BLOOD", "LUNG") else 9).round(4).to_string(index=False))
print("interaction by tissue:", ok.groupby("tissue").n_sig_interaction.sum().sort_values(ascending=False).head(5).to_dict())
cc = pd.read_csv(f"{G}/pseudobulk_de/composition_confound_table.tsv", sep="\t")
print("composition verdicts:", cc.verdict.value_counts(dropna=False).to_dict())
print(cc[cc.verdict == "FLAG_COMPOSITION"][["tissue", "cell_type", "gate_class", "n_sig_dose_IHW", "frac_week_p"]].round(4).to_string(index=False))

print("\n=== positive controls: Tier B and C detail")
pr = pd.read_csv(f"{G}/pseudobulk_de/posctrl_results.tsv", sep="\t")
print(pr[pr.tier == "B"][["symbol", "tissue", "cell_type", "sex", "outcome", "FDR_IHW"]].to_string(index=False))
print(pr[(pr.tier == "A") & (pr.outcome.isin(["recovered", "wrong_direction"]))][["symbol", "tissue", "cell_type", "outcome", "FDR_IHW", "lfc_8w"]].to_string(index=False))
rs = pd.read_csv(f"{G}/pseudobulk_de/posctrl_responsiveness.tsv", sep="\t"); print(rs.to_string(index=False)[:3000])
pa = pd.read_csv(f"{G}/pseudobulk_de/parenchyma_dataanchored_validation.tsv", sep="\t"); print(pa.to_string(index=False)[:1500])

print("\n=== transfer")
tc = pd.read_csv(f"{GH}/gate_rcv10/transfer_comparison.tsv", sep="\t")
lost = tc[tc.status == "LOST"]
print("status:", tc.status.fillna("no_human_block").value_counts().to_dict(), "| lost with rat AUC < 0.65:", int((lost.sup_trained_auc_rat < 0.65).sum()))
h = tc[tc.is_hotspot == True]
print(h[["tissue", "cell_type", "sup_trained_auc_rat", "sup_trained_auc_human", "p_sup_trained_human", "q_sup_trained_human", "status", "transfer_corrected", "sup_dose_rho_rat", "sup_dose_rho_human"]].round(4).to_string(index=False))
print(tc[tc.gate_class_rat == "borderline"][["tissue", "cell_type", "sup_trained_auc_rat", "sup_trained_auc_human", "p_sup_trained_human", "q_sup_trained_human", "status", "transfer_corrected"]].round(4).to_string(index=False))
ro = f"{GH}/gate_rcv10/rat_only_pseudocells.tsv"
if os.path.exists(ro):
    r = pd.read_csv(ro, sep="\t"); print("rat-only pseudo-cells:", len(r), r.iloc[:, :4].drop_duplicates(subset=[c for c in r.columns if c in ("tissue", "cell_type")]).head(3).to_string(index=False))

print("\n=== networks")
hv = pd.read_csv(f"{REC}/step3/downstream/networks_decoy_hub_vs_decoy.tsv", sep="\t")
pn = pd.read_csv(f"{REC}/step3/downstream/networks_decoy_fdr_per_network.tsv", sep="\t")
print("TF x network pairs:", len(hv), "distinct TFs:", hv.tf.nunique(), "| min q:", round(hv.q.min(), 3), "| per-network FDR(2):", pn.fdr_z2.min(), pn.fdr_z2.max(),
      "| TF edges total:", int(pn.tf_edges.sum()))
sc = pd.read_csv(f"{REC}/step3/downstream/networks_self_consistency_refs5.tsv", sep="\t")
print(list(sc.columns)); print(sc.head(3).to_string(index=False))
rc = [c for c in sc.columns if c.lower() in ("rho", "spearman", "spearman_rho", "conservation_rho")]
if rc:
    r = sc[rc[0]].dropna(); print("scores:", len(r), "median", round(r.median(), 3), "share > 0:", round((r > 0).mean(), 3), "share >= 0.3:", round((r >= 0.3).mean(), 3))
    nc = [c for c in sc.columns if c in ("network", "cell_type", "tissue_celltype")]
    if nc: print(sc.groupby(nc[0])[rc[0]].median().round(3).to_dict())
    sym = [c for c in sc.columns if "symbol" in c.lower()]
    if sym: print(sc[sc[sym[0]].isin(["Cebpa", "Spi1", "Mef2c", "Myf6", "Nr4a1", "Epas1", "Tfeb"])].to_string(index=False)[:2500])

print("\n=== mixture panel on refs5 (V6)")
fl = pd.read_csv(f"{REC}/step4/v6_mixture_panel/v6_floor.tsv", sep="\t")
print("type tests:", int(fl.n_types_scored.sum()), "| >= 0.95:", int(fl.n_types_r_ge_095.sum()), "| >= 0.90:", int(fl.n_types_r_ge_090.sum()), "| below floor:", int(fl.n_types_below_floor.sum()))
pm = pd.read_csv(f"{REC}/step4/v6_mixture_panel/panel_median.tsv", sep="\t", index_col=0)
best = pm.idxmax(axis=1); print("BayesPrism best on", int((best == "bp_rna").sum()), "of", len(pm), "| max gap to best:", round((pm.max(axis=1) - pm.bp_rna).max(), 3),
                                "| sets where another method is best:", {k: (v, round(pm.loc[k, v], 3), round(pm.loc[k, 'bp_rna'], 3)) for k, v in best.items() if v != "bp_rna"})
print(pm.round(3).to_string())
pc = pd.read_csv(f"{REC}/step4/v6_mixture_panel/panel_coverage.tsv", sep="\t"); print(pc.to_string(index=False)[:2500])

print("\n=== atlas (earlier default's cells): per-type accuracy of the refs5 called types")
at = pd.read_csv(f"{M}/translation/atlas_record_16/atlas_projection_per_type.tsv", sep="\t")
print(list(at.columns))
called = [("skmvl", "Satellite cells"), ("skmvl", "Tenocytes"), ("skmvl", "Vascular smooth muscle cells"), ("skmgn", "Schwann cells"),
          ("skmgn", "Vascular smooth muscle cells"), ("skmgn", "Tenocytes"), ("heart", "Vascular smooth muscle cells"), ("lung", "Endothelial cells"),
          ("lung", "Vascular smooth muscle cells"), ("blood", "Neutrophils"), ("blood", "Erythroid cells"),
          ("blood", "Megakaryocytes and platelets"), ("blood", "Classical monocytes")]
tcol = [c for c in at.columns if "tissue" in c.lower()][0]; ccol = [c for c in at.columns if "cell_type" in c.lower() or "rat_type" in c.lower()][0]
for t, c in called:
    x = at[(at[tcol].str.lower() == t) & (at[ccol] == c)]
    print(t, c, "->", x.drop(columns=[tcol, ccol]).round(3).to_dict("records") if len(x) else "not scored (no atlas counterpart or not in the map)")
