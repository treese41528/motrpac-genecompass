#!/usr/bin/env python3
"""corroborate_summary.py -- merge the gate's outputs into the table that defines the exercise hotspots.

Inputs, all in the gate folder (--gc-root):
  subspace_probe.tsv -> the PLS-1 probe on the 768-d embedding: AUC, permutation counts and p, the interval
                        limits of the ideal p, and the sensitivities S1 and S2
  pca_control.tsv    -> the same probe by representation: auc_embed / auc_pca / auc_genes
  augur_results.tsv  -> augur_auc (canonical neurorestore Augur, RF) by (representation, condition)
and, from the Stage 8 output (--results-root): each block's Z matrix and the tissue's estimated fractions.
Writes corroboration_merged.tsv, its figures (<table>.figures.tsv) and provenance_<table>.json beside it.

What the table decides (docs/audit/remediation/STEP3_PREREG.md section 5):

  Analysable blocks (REFERENCE_RELABEL_PREREG.md section 6, unchanged). A block whose median per-sample
      library (the sum of BayesPrism's Z over genes, from <results>/motrpac/<TISSUE>/pred_z) is under
      --min-libsize (default MIN_MEDIAN_LIBSIZE = 1,000) is not analysed. The analysable blocks are the family
      of every multiplicity correction here. median_libsize(), is_analysable() and the threshold live in this
      file only; subspace_probe.py imports them to know the family before it samples.
  Hotspots (5.2). gate_class is hotspot / borderline / none from the limits of each block's ideal p
      (mc_gate.classify): a hotspot is rejected by Benjamini-Hochberg with every p at its upper limit, a
      borderline block at the lower limits only. is_hotspot is gate_class == hotspot; it is the column the
      later stages read. Blocks outside the family are 'none'. q_sup_trained is BH on the final p and
      q_sup_trained_10k is BH on the registered 10,000-permutation p (the previous method), for comparison.
  Sensitivities (5.3), beside every block, none replacing the primary:
      S1  q_sup_trained_s1: folds stratified on trained x sex, labels permuted within sex;
      S2  q_sup_trained_s2: 8-week trained against control only;
      S3  auc_pca, auc_genes: the probe in gene space (from pca_control.tsv);
      S4  mean_fraction, low_fraction (mean estimated fraction under 0.01), q_sup_trained_s4: BH within the
          analysable blocks that are not low-fraction.
      S1, S2 and S4 use the first-batch p and BH over their own family.
  Reading rule (5.3). support, for hotspots only: 'supported' when the block is significant in S1 and its
      all-genes AUC is at least 0.65; 'model-only' when its all-genes AUC is below 0.55; 'neither' otherwise.
  Library-rule sensitivities. With --min-libsize other than the registered value (Stage 11 runs 100 and 0) the
      family differs from the one the probe sampled for, so the table uses the 10,000-permutation p and plain
      BH (gate_method = bh_10k; no borderline class). The same applies to a probe table without interval
      limits. gate_method = mmctest marks the registered method.

It also prints the corroboration figures of 5.4 over the analysable blocks and over all blocks: embedding
against PCA-50 and against all genes (paired Wilcoxon signed-rank), and the Spearman correlation of Augur with
the probe. No threshold is applied to that correlation.
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(os.environ.setdefault(
    "PIPELINE_ROOT", str(Path(__file__).resolve().parents[1]))) / "lib"))
from gene_utils import load_config, resolve_path                          # noqa: E402
from provenance import write_provenance                                   # noqa: E402
from celltype_names import safe                                           # noqa: E402
import mc_gate                                                            # noqa: E402

KEY = ["tissue", "cell_type"]
MIN_MEDIAN_LIBSIZE = 1000.0   # the analysable rule: median per-sample library of at least this (registered)
LOW_FRACTION = 0.01           # S4 tier: mean estimated fraction below this is "low-fraction" (5.3, 6.3)
SUPPORTED_GENES_AUC = 0.65    # reading rule: all-genes AUC at or above this, with S1, is "supported"
MODEL_ONLY_GENES_AUC = 0.55   # reading rule: all-genes AUC below this is "model-only"

# Columns taken from subspace_probe.tsv: the ones every version has, then the ones the Step 3 probe adds.
PROBE_BASE = ["n", "sup_trained_auc", "p_sup_trained", "glob_trained", "p_glob_trained"]
PROBE_NEW = ["b_sup_trained", "m_sup_trained", "p_sup_trained_10k", "in_family", "p_lo_sup_trained",
             "p_hi_sup_trained", "gate_class", "sup_trained_auc_s1", "p_sup_trained_s1", "n_s2",
             "sup_trained_auc_s2", "p_sup_trained_s2"]


# ----------------------------------------------------------------------- the family and the fraction tier
def median_libsize(res_root, tissue, cell_type):
    """Median over samples of the block's summed Z (negatives and NaN as 0, as run_pseudobulk_de.R does).
    res_root is <results_dir>/motrpac."""
    z = pd.read_csv(Path(res_root) / tissue.upper() / "pred_z" / f"predz__{safe(cell_type)}.csv", index_col=0)
    return float(np.median(z.clip(lower=0).fillna(0).sum(axis=1)))


def is_analysable(median_lib, min_libsize=MIN_MEDIAN_LIBSIZE):
    """The registered analysable rule. Works on a number or a column."""
    return median_lib >= min_libsize


def mean_fractions(res_root, tissue):
    """Mean over samples of every cell type's estimated fraction in one tissue (Stage 8's
    estimated_fractions.csv), or None where the file is not on disk."""
    f = Path(res_root) / tissue.upper() / "estimated_fractions.csv"
    if not f.exists():
        return None
    return pd.read_csv(f, index_col=0).mean(axis=0)


# ----------------------------------------------------------------------------------------- the table
def bh_over(p, mask):
    """Benjamini-Hochberg adjusted p over the rows where mask is true and p is present; missing elsewhere."""
    q = pd.Series(np.nan, index=p.index, dtype=float)
    sel = p[mask.astype(bool)].dropna()
    q.loc[sel.index] = mc_gate.bh_adjust(sel.to_numpy(dtype=float))
    return q


def support_label(is_hotspot, q_s1, auc_genes, alpha=mc_gate.ALPHA):
    """The reading rule of section 5.3 for one block. Only a hotspot gets a label."""
    if not is_hotspot:
        return ""
    if pd.notna(auc_genes) and auc_genes < MODEL_ONLY_GENES_AUC:
        return "model-only"
    if pd.notna(q_s1) and q_s1 < alpha and pd.notna(auc_genes) and auc_genes >= SUPPORTED_GENES_AUC:
        return "supported"
    return "neither"


def merge_inputs(sp, pc, au):
    """One row per block from the probe table, the PCA control and the Augur results (outer join)."""
    aug = au.pivot_table(index=KEY, columns=["representation", "condition"], values="augur_auc")
    aug.columns = [f"augur_{r}_{c}" for r, c in aug.columns]
    aug = aug.reset_index()
    new = [c for c in PROBE_NEW if c in sp.columns]
    return (sp[KEY + PROBE_BASE + new]
            .merge(pc[KEY + ["auc_embed", "auc_pca", "auc_genes", "p_embed"]], on=KEY, how="outer")
            .merge(aug, on=KEY, how="outer"))


def add_gate_columns(m, min_libsize=MIN_MEDIAN_LIBSIZE, alpha=mc_gate.ALPHA):
    """Family, multiplicity control, classes, sensitivities and the reading rule. m needs median_libsize and
    mean_fraction. Returns (table, gate_method)."""
    m = m.copy()
    m["analysable"] = is_analysable(m["median_libsize"], min_libsize)
    if "p_sup_trained_10k" not in m:                                       # a probe table from before Step 3 holds
        m["p_sup_trained_10k"] = m["p_sup_trained"]                        # only the 10,000-permutation p
    family = m["analysable"] & m["p_sup_trained"].notna()
    m["q_sup_trained_10k"] = bh_over(m["p_sup_trained_10k"], m["analysable"])

    limits = ["p_lo_sup_trained", "p_hi_sup_trained"]
    sampled = "in_family" in m and all(c in m for c in limits)
    if min_libsize == MIN_MEDIAN_LIBSIZE and sampled:
        # The registered method. The intervals were computed for the family the probe sampled; it must be the
        # family of this table, or the classes would not have the registered error level.
        in_family = m["in_family"].astype(str).str.lower() == "true"
        differ = m[m["p_sup_trained"].notna() & (in_family != m["analysable"])]
        if len(differ):
            sys.exit("ERROR: the probe sampled a different family than the analysable rule gives here:\n"
                     + differ[KEY + ["median_libsize", "in_family", "analysable"]].to_string(index=False))
        if m.loc[family, limits].isna().any().any():
            sys.exit("ERROR: analysable blocks without interval limits in subspace_probe.tsv")
        method = "mmctest"
        m["q_sup_trained"] = bh_over(m["p_sup_trained"], m["analysable"])
        cls = pd.Series(mc_gate.NONE, index=m.index, dtype=object)
        cls[family] = mc_gate.classify(m.loc[family, limits[0]], m.loc[family, limits[1]], alpha)
        if "gate_class" in m:
            stored = m.loc[family, "gate_class"]
            if not (stored == cls[family]).all():
                sys.exit("ERROR: classes recomputed from the interval limits differ from the classes the probe "
                         f"stored (was the probe run with another level than {alpha}?)")
    else:
        method = "bh_10k"
        m["q_sup_trained"] = m["q_sup_trained_10k"]
        cls = pd.Series(np.where(m["q_sup_trained_10k"] < alpha, mc_gate.HOTSPOT, mc_gate.NONE),
                        index=m.index, dtype=object)
    m["gate_class"] = cls
    m["is_hotspot"] = m["gate_class"] == mc_gate.HOTSPOT
    m["gate_method"] = method

    # S1 and S2: Benjamini-Hochberg over the analysable blocks that have the sensitivity's p.
    for s in ("s1", "s2"):
        if f"p_sup_trained_{s}" in m:
            m[f"q_sup_trained_{s}"] = bh_over(m[f"p_sup_trained_{s}"], m["analysable"])
    # S4: the fraction tier. A block with no fraction on disk is in neither tier.
    known = m["mean_fraction"].notna()
    m["low_fraction"] = pd.array(np.where(known, m["mean_fraction"] < LOW_FRACTION, pd.NA), dtype="boolean")
    m["q_sup_trained_s4"] = bh_over(m["p_sup_trained_10k"],
                                    m["analysable"] & known & (m["mean_fraction"] >= LOW_FRACTION))
    q_s1 = m["q_sup_trained_s1"] if "q_sup_trained_s1" in m else pd.Series(np.nan, index=m.index)
    m["support"] = [support_label(h, q, g, alpha) for h, q, g in zip(m["is_hotspot"], q_s1, m["auc_genes"])]
    return m, method


def order_columns(m):
    """The columns of the table before Step 3 first, in their old order, then the new ones."""
    old = (KEY + PROBE_BASE + ["auc_embed", "auc_pca", "auc_genes", "p_embed"]
           + sorted(c for c in m.columns if c.startswith("augur_"))
           + ["median_libsize", "analysable", "q_sup_trained"])
    new = ["b_sup_trained", "m_sup_trained", "p_sup_trained_10k", "q_sup_trained_10k", "p_lo_sup_trained",
           "p_hi_sup_trained", "gate_class", "is_hotspot", "gate_method",
           "sup_trained_auc_s1", "p_sup_trained_s1", "q_sup_trained_s1",
           "n_s2", "sup_trained_auc_s2", "p_sup_trained_s2", "q_sup_trained_s2",
           "mean_fraction", "low_fraction", "q_sup_trained_s4", "support"]
    cols = [c for c in old + new if c in m.columns]
    return m[cols + [c for c in m.columns if c not in cols and c != "in_family"]]


# ------------------------------------------------------------------------------- figures of section 5.4
def corroboration_figures(m):
    """The quoted corroboration figures, each over the analysable blocks and over all blocks: paired
    comparisons of the probe's AUC by representation (two-sided Wilcoxon signed-rank) and Spearman
    correlations of Augur with the probe. One row per figure and scope."""
    rows = []
    for scope, sub in (("analysable", m[m["analysable"]]), ("all", m)):
        for a, b, label in (("auc_embed", "auc_pca", "PLS-1: embedding vs PCA-50"),
                            ("auc_embed", "auc_genes", "PLS-1: embedding vs all genes"),
                            ("augur_embed_trained", "augur_pca_trained", "Augur: embedding vs PCA-50")):
            if a not in sub or b not in sub:
                continue
            s = sub[[a, b]].dropna(); d = s[a] - s[b]
            try:
                w = wilcoxon(s[a], s[b]).pvalue if len(s) > 1 else np.nan
            except ValueError:                                             # every difference is zero
                w = np.nan
            rows.append(dict(scope=scope, figure=label, a=a, b=b, n=len(s), median_a=s[a].median(),
                             median_b=s[b].median(), median_difference=d.median(), n_a_above_b=int((d > 0).sum()),
                             wilcoxon_p=w, spearman_r=np.nan))
        for a, b, label in (("augur_embed_trained", "sup_trained_auc", "Augur vs PLS-1 probe (subspace_probe)"),
                            ("augur_embed_trained", "auc_embed", "Augur vs PLS-1 probe (pca_control)")):
            if a not in sub or b not in sub:
                continue
            s = sub[[a, b]].dropna()
            r = spearmanr(s[a], s[b]).correlation if len(s) > 2 else np.nan
            rows.append(dict(scope=scope, figure=label, a=a, b=b, n=len(s), median_a=s[a].median(),
                             median_b=s[b].median(), median_difference=(s[a] - s[b]).median(),
                             n_a_above_b=int((s[a] > s[b]).sum()), wilcoxon_p=np.nan, spearman_r=r))
    return pd.DataFrame(rows)


def print_report(m, figs, method, min_libsize, alpha):
    pd.set_option("display.width", 250, "display.max_rows", 300, "display.max_columns", 60)
    print(f"analysable blocks (median libsize >= {min_libsize:g}): {int(m['analysable'].sum())}/{len(m)}")

    print("\n=== CORROBORATION FIGURES (section 5.4): analysable blocks and all blocks ===")
    for _, r in figs.iterrows():
        if pd.notna(r["spearman_r"]) or r["figure"].startswith("Augur vs"):
            print(f"  [{r['scope']:10s}] {r['figure']:40s} Spearman r={r['spearman_r']:.3f}  n={r['n']}  "
                  f"medians {r['median_a']:.3f} / {r['median_b']:.3f}")
        else:
            print(f"  [{r['scope']:10s}] {r['figure']:40s} medians {r['median_a']:.3f} / {r['median_b']:.3f}  "
                  f"difference {r['median_difference']:+.3f}  above in {r['n_a_above_b']}/{r['n']}  "
                  f"Wilcoxon p={r['wilcoxon_p']:.2g}")
    print("  (the Augur correlation is reported as it is; no threshold is registered for it)")

    print("\n=== SEX positive control (Augur should be high & near PLS-1) ===")
    for r in ["embed", "pca"]:
        c = f"augur_{r}_sex"
        if c in m:
            print(f"  Augur {r} sex: median AUC {m[c].median():.3f}  max {m[c].max():.3f}")

    counts = m.loc[m["analysable"], "gate_class"].value_counts().to_dict()
    print(f"\n=== GATE (section 5.2; method {method}, BH level {alpha:g}) ===")
    print(f"  hotspot {counts.get(mc_gate.HOTSPOT, 0)}  |  borderline {counts.get(mc_gate.BORDERLINE, 0)}  |  "
          f"not a hotspot {counts.get(mc_gate.NONE, 0)}   of {int(m['analysable'].sum())} analysable blocks")
    print(f"  previous method (BH on the 10,000-permutation p): "
          f"{int((m['q_sup_trained_10k'] < alpha).sum())} blocks with q < {alpha:g}")
    if method != "mmctest":
        print("  NOTE: plain BH on the first-batch p; the interval method needs the probe's own family "
              "(--min-libsize at the registered value and a probe run with --sequential)")
    called = m[m["gate_class"] != mc_gate.NONE].sort_values(["gate_class", "p_sup_trained"],
                                                            ascending=[False, True])
    cols = [c for c in ["tissue", "cell_type", "gate_class", "sup_trained_auc", "m_sup_trained",
                        "p_sup_trained_10k", "p_sup_trained", "p_lo_sup_trained", "p_hi_sup_trained",
                        "q_sup_trained", "q_sup_trained_10k"] if c in m]
    if len(called):
        print(called[cols].to_string(index=False))
        print("\n=== SENSITIVITIES for the hotspots and borderline blocks (section 5.3) ===")
        cols = [c for c in ["tissue", "cell_type", "gate_class", "sup_trained_auc_s1", "q_sup_trained_s1",
                            "sup_trained_auc_s2", "p_sup_trained_s2", "q_sup_trained_s2", "auc_pca", "auc_genes",
                            "mean_fraction", "low_fraction", "q_sup_trained_s4", "support"] if c in m]
        print(called[cols].to_string(index=False))
    hot = m[m["is_hotspot"]]
    print(f"\n  reading rule: supported {int((hot['support'] == 'supported').sum())}, "
          f"model-only {int((hot['support'] == 'model-only').sum())}, "
          f"neither {int((hot['support'] == 'neither').sum())} of {len(hot)} hotspots")
    for s, label in (("s1", "S1 (sex-stratified)"), ("s2", "S2 (8-week against control)"),
                     ("s4", "S4 (not low-fraction)")):
        c = f"q_sup_trained_{s}"
        if c in m:
            print(f"  {label}: {int((m[c] < alpha).sum())} of {int(m[c].notna().sum())} blocks with q < {alpha:g}; "
                  f"{int((hot[c] < alpha).sum())} of the {len(hot)} hotspots")

    if "augur_embed_trained" in m:
        rest = m[m["analysable"] & ~m["is_hotspot"]]
        print("\n=== Augur on the hotspots ===")
        print(f"  Augur embed AUC median on the {len(hot)} hotspots = {hot['augur_embed_trained'].median():.3f}  "
              f"vs the other analysable blocks = {rest['augur_embed_trained'].median():.3f}")

    print("\n=== top exercise responders (by PLS-1 embed AUC), all methods side by side ===")
    cols = [c for c in ["tissue", "cell_type", "n", "glob_trained", "sup_trained_auc", "p_sup_trained",
                        "augur_embed_trained", "auc_pca", "augur_pca_trained", "auc_genes"] if c in m]
    print(m.dropna(subset=["sup_trained_auc"]).sort_values("sup_trained_auc", ascending=False)[cols]
          .head(22).to_string(index=False))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gc-root", default=None,
                    help="gate folder holding subspace_probe.tsv, pca_control.tsv, augur_results.tsv "
                         "(default: config genecompass_input_dir)")
    ap.add_argument("--results-root", default=None,
                    help="deconvolution results root holding motrpac/<TISSUE>/pred_z (default: config results_dir)")
    ap.add_argument("--min-libsize", type=float, default=MIN_MEDIAN_LIBSIZE,
                    help="median per-sample library below which a block is not analysed (registered: "
                         f"{MIN_MEDIAN_LIBSIZE:g}; 100 and 0 are the registered sensitivities)")
    ap.add_argument("--alpha", type=float, default=mc_gate.ALPHA, help="Benjamini-Hochberg level")
    ap.add_argument("--out", default=None, help="output table (default: <gc-root>/corroboration_merged.tsv)")
    args = ap.parse_args(argv)
    cfg = load_config()
    gc = args.gc_root or str(resolve_path(cfg, cfg["deconvolution"]["genecompass_input_dir"]))
    res_root = Path(args.results_root or resolve_path(cfg, cfg["deconvolution"]["results_dir"])) / "motrpac"
    out = Path(args.out or f"{gc}/corroboration_merged.tsv")
    inputs = [f"{gc}/subspace_probe.tsv", f"{gc}/pca_control.tsv", f"{gc}/augur_results.tsv"]
    sp, pc, au = (pd.read_csv(f, sep="\t") for f in inputs)

    m = merge_inputs(sp, pc, au)
    m["median_libsize"] = [median_libsize(res_root, t, c) for t, c in zip(m["tissue"], m["cell_type"])]
    fractions = {t: mean_fractions(res_root, t) for t in m["tissue"].unique()}
    missing = sorted(t for t, f in fractions.items() if f is None)
    if missing:
        print(f"NOTE: no estimated_fractions.csv under {res_root} for {missing}; their mean_fraction is empty")
    m["mean_fraction"] = [np.nan if fractions[t] is None else fractions[t].get(c, np.nan)
                          for t, c in zip(m["tissue"], m["cell_type"])]
    m, method = add_gate_columns(m, args.min_libsize, args.alpha)
    m = order_columns(m)
    figs = corroboration_figures(m)
    print_report(m, figs, method, args.min_libsize, args.alpha)

    # Each merge writes its own table, figures and provenance record, named after the table, so the default
    # merge and the library-rule sensitivities no longer overwrite one record (section 5.4).
    out.parent.mkdir(parents=True, exist_ok=True)
    m.to_csv(out, sep="\t", index=False)
    figs_out = out.with_suffix(".figures.tsv")
    figs.to_csv(figs_out, sep="\t", index=False)
    classes = m.loc[m["analysable"], "gate_class"].value_counts().to_dict()
    write_provenance(out.parent, name=f"provenance_{out.stem}.json",
                     params={"gc_root": gc, "results_root": str(res_root), "min_libsize": args.min_libsize,
                             "registered_min_libsize": MIN_MEDIAN_LIBSIZE, "alpha": args.alpha,
                             "gate_method": method, "low_fraction": LOW_FRACTION,
                             "supported_genes_auc": SUPPORTED_GENES_AUC,
                             "model_only_genes_auc": MODEL_ONLY_GENES_AUC, "out": str(out),
                             "figures": str(figs_out), "n_blocks": int(len(m)),
                             "n_analysable": int(m["analysable"].sum()), "classes": classes},
                     inputs=inputs + sorted(str(p) for p in res_root.glob("*/estimated_fractions.csv")))
    print(f"\nwrote {out} ({len(m)} rows), {figs_out.name} and provenance_{out.stem}.json")


if __name__ == "__main__":
    main()
