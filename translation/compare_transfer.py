#!/usr/bin/env python3
"""
compare_transfer.py -- Aim 3a / Module E step 4 (E.2 deliverable): does the rat
exercise axis SURVIVE the transfer into human GeneCompass embedding space?

For every (tissue, cell_type) block, compare the supervised exercise-axis detection on
the RAT-space embeddings (already computed: corroboration_merged.tsv + subspace_probe.tsv)
against the HUMAN-space embeddings (subspace_probe.tsv produced by Stage 12 step 3 on the
transferred cells). The transferred cells are the SAME cells re-expressed as human, so this
is a paired, within-block comparison.

THREE questions, in order:
  (1) POSITIVE CONTROL -- does a STRONG, transfer-agnostic biological axis survive? Sex is the
      pre-registered control (dominant, not exercise). If sup_sex_auc collapses in human space,
      the transfer corrupted the embedding and nothing downstream is trustworthy. This is checked
      and reported FIRST, before any exercise claim.
  (2) GLOBAL FIDELITY -- across all blocks, how well does the human-space trained AUC track the
      rat-space trained AUC (Spearman)? And the ordinal-dose rho?
  (3) HOTSPOT SURVIVAL -- among the rat exercise hotspots (q_sup_trained < --hotspot-q), classify
      each as PRESERVED / WEAKENED / LOST in human space. This is the headline E.2 result.

Read-only on the rat tables; writes transfer_comparison.tsv (full paired join) and
transfer_comparison.md (the verdict) under --out.

Usage:
  python translation/compare_transfer.py \
    --rat-corr   data/deconvolution/genecompass_input/corroboration_merged.tsv \
    --rat-probe  data/deconvolution/genecompass_input/subspace_probe.tsv \
    --human-probe data/deconvolution/genecompass_input_human/subspace_probe.tsv \
    --out        data/deconvolution/genecompass_input_human
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

_ROOT = Path(os.environ.setdefault("PIPELINE_ROOT", str(Path(__file__).resolve().parents[1])))
sys.path.insert(0, str(_ROOT / "lib"))
from gene_utils import load_config, resolve_path  # noqa: E402

KEY = ["tissue", "cell_type"]


def _read(path, cols, tag):
    df = pd.read_csv(path, sep="\t")
    miss = [c for c in cols if c not in df.columns]
    if miss:
        sys.exit(f"ERROR: {tag} ({path}) missing columns {miss}; has {list(df.columns)}")
    return df


def _spearman(a, b):
    m = a.notna() & b.notna()
    if m.sum() < 3:
        return float("nan"), int(m.sum())
    r, _ = spearmanr(a[m], b[m])
    return float(r), int(m.sum())


def classify(row, auc_thresh, p_thresh):
    """PRESERVED: human keeps a detectable trained axis (AUC>=thresh AND p<p_thresh).
       WEAKENED: still above the AUC floor but no longer significant (signal degraded).
       LOST: human trained AUC fell below the floor."""
    ha, hp = row["sup_trained_auc_human"], row["p_sup_trained_human"]
    if pd.isna(ha):
        return "no_human_block"
    if ha >= auc_thresh and (pd.notna(hp) and hp < p_thresh):
        return "PRESERVED"
    if ha >= auc_thresh:
        return "WEAKENED"
    return "LOST"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    cfg = load_config()
    gc = resolve_path(cfg, cfg["deconvolution"]["genecompass_input_dir"])
    ap.add_argument("--rat-corr", default=str(Path(gc) / "corroboration_merged.tsv"),
                    help="rat corroboration_merged.tsv (q_sup_trained defines hotspots)")
    ap.add_argument("--rat-probe", default=str(Path(gc) / "subspace_probe.tsv"),
                    help="rat subspace_probe.tsv (sup_dose_rho + sup_sex_auc positive control)")
    ap.add_argument("--human-probe", default=str(Path(str(gc) + "_human") / "subspace_probe.tsv"),
                    help="human-space subspace_probe.tsv (Stage 12 step 3)")
    ap.add_argument("--human-corr", default=str(Path(str(gc) + "_human") / "corroboration_merged.tsv"),
                    help="OPTIONAL human-space corroboration_merged.tsv (canonical Augur-RF); "
                         "if present, the verdict adds a method-robustness check vs the PLS-1 probe")
    ap.add_argument("--out", default=str(Path(str(gc) + "_human")))
    ap.add_argument("--auc-thresh", type=float, default=0.65,
                    help="trained-AUC floor for a 'detectable' axis (default 0.65)")
    ap.add_argument("--p-thresh", type=float, default=0.05)
    ap.add_argument("--hotspot-q", type=float, default=0.05,
                    help="rat q_sup_trained threshold defining exercise hotspots")
    args = ap.parse_args()

    rat_c = _read(args.rat_corr, KEY + ["sup_trained_auc", "p_sup_trained", "q_sup_trained"], "rat-corr")
    rat_p = _read(args.rat_probe, KEY + ["sup_trained_auc", "sup_dose_rho", "sup_sex_auc"], "rat-probe")
    hum_p = _read(args.human_probe, KEY + ["sup_trained_auc", "p_sup_trained", "sup_dose_rho", "sup_sex_auc"],
                  "human-probe")

    # rat side: AUC + q from corroboration; dose_rho + sex from the probe
    rat = rat_c[KEY + ["sup_trained_auc", "p_sup_trained", "q_sup_trained"]].merge(
        rat_p[KEY + ["sup_dose_rho", "sup_sex_auc"]], on=KEY, how="left")
    m = rat.merge(hum_p[KEY + ["sup_trained_auc", "p_sup_trained", "sup_dose_rho", "sup_sex_auc"]],
                  on=KEY, how="left", suffixes=("_rat", "_human"))

    # OPTIONAL canonical Augur-RF method robustness: corroboration_merged.tsv augur_embed_trained,
    # rat (always present) + human (only after the corroboration step has run -- graceful if absent)
    def _augur(path, tag):
        p = Path(path)
        if not p.exists():
            return None
        d = pd.read_csv(p, sep="\t")
        if "augur_embed_trained" not in d.columns:
            return None
        return d[KEY + ["augur_embed_trained"]].rename(columns={"augur_embed_trained": f"augur_{tag}"})
    ra, ha = _augur(args.rat_corr, "rat"), _augur(args.human_corr, "human")
    if ra is not None:
        m = m.merge(ra, on=KEY, how="left")
    if ha is not None:
        m = m.merge(ha, on=KEY, how="left")
    has_augur = (ra is not None) and (ha is not None)

    m["is_hotspot"] = m["q_sup_trained"] < args.hotspot_q
    m["delta_auc"] = m["sup_trained_auc_human"] - m["sup_trained_auc_rat"]
    m["status"] = m.apply(lambda r: classify(r, args.auc_thresh, args.p_thresh), axis=1)
    m = m.sort_values(["is_hotspot", "sup_trained_auc_rat"], ascending=[False, False])

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    m.to_csv(out / "transfer_comparison.tsv", sep="\t", index=False)

    # ---- (1) positive control: sex must survive ----
    sx_r, n_sx = _spearman(m["sup_sex_auc_rat"], m["sup_sex_auc_human"])
    sex_rat_med, sex_hum_med = m["sup_sex_auc_rat"].median(), m["sup_sex_auc_human"].median()
    # ---- (2) global fidelity ----
    auc_r, n_auc = _spearman(m["sup_trained_auc_rat"], m["sup_trained_auc_human"])
    dose_r, n_dose = _spearman(m["sup_dose_rho_rat"], m["sup_dose_rho_human"])
    # ---- (3) hotspot survival ----
    hot = m[m["is_hotspot"]]
    counts = hot["status"].value_counts().to_dict()
    n_pres = counts.get("PRESERVED", 0)
    n_hot = len(hot)

    md = []
    md.append("# Cross-species transfer of the rat exercise response (Aim 3a / Module E, E.2)\n")
    md.append("Does the rat trained-vs-control / ordinal-dose axis SURVIVE re-expression of the rat "
              "pseudo-cells in human GeneCompass embedding space? Paired per (tissue, cell_type) block: "
              "RAT-space vs HUMAN-space supervised PLS-1 probe on the SAME cells.\n")
    md.append(f"- blocks compared: {len(m)}   |   exercise hotspots (rat q_sup_trained < {args.hotspot_q}): {n_hot}\n")

    md.append("\n## (1) Positive control -- does the SEX axis survive? (transfer-validity gate)\n")
    sex_ok = (pd.notna(sex_hum_med) and sex_hum_med >= 0.70)
    md.append(f"- median sup_sex_auc: rat {sex_rat_med:.3f} -> human {sex_hum_med:.3f}  "
              f"(Spearman rat~human across {n_sx} blocks: r={sx_r:.3f})\n")
    md.append(f"- **transfer-validity: {'PASS' if sex_ok else 'FAIL'}** -- a strong, transfer-agnostic "
              f"biological axis {'remains' if sex_ok else 'does NOT remain'} detectable in human space "
              f"(floor 0.70). {'Embedding biology preserved; exercise claims below are interpretable.' if sex_ok else 'Transfer suspect -- do NOT trust the exercise comparison until resolved.'}\n")

    md.append("\n## (2) Global fidelity -- does human-space tracking follow rat-space?\n")
    md.append(f"- trained AUC  Spearman(rat, human) = {auc_r:.3f}  (n={n_auc})\n")
    md.append(f"- ordinal dose rho Spearman(rat, human) = {dose_r:.3f}  (n={n_dose})\n")

    md.append("\n## (3) Hotspot survival -- the headline result\n")
    md.append(f"- PRESERVED {n_pres}/{n_hot}  |  WEAKENED {counts.get('WEAKENED', 0)}/{n_hot}  "
              f"|  LOST {counts.get('LOST', 0)}/{n_hot}   "
              f"(PRESERVED = human AUC>={args.auc_thresh} and p<{args.p_thresh})\n")
    md.append("\n| tissue | cell_type | rat AUC | human AUC | dAUC | rat dose_rho | human dose_rho | status |\n")
    md.append("|---|---|---|---|---|---|---|---|\n")
    for _, r in hot.iterrows():
        md.append(f"| {r['tissue']} | {r['cell_type']} | {r['sup_trained_auc_rat']:.3f} | "
                  f"{_fmt(r['sup_trained_auc_human'])} | {_fmt(r['delta_auc'], signed=True)} | "
                  f"{_fmt(r['sup_dose_rho_rat'])} | {_fmt(r['sup_dose_rho_human'])} | {r['status']} |\n")

    md.append("\n## Per-tissue hotspot preservation\n")
    if n_hot:
        for tis, g in hot.groupby("tissue"):
            p = (g["status"] == "PRESERVED").sum()
            md.append(f"- {tis}: {p}/{len(g)} preserved\n")

    # ---- (4) method robustness: canonical Augur-RF (only if the corroboration step has run) ----
    if has_augur:
        ar, n_ar = _spearman(m["augur_rat"], m["augur_human"])
        ha_hot = hot.dropna(subset=["augur_human"])
        n_aug_surv = int((ha_hot["augur_human"] >= args.auc_thresh).sum())
        md.append("\n## (4) Method robustness -- does canonical Augur-RF agree the axis survives?\n")
        md.append(f"- Augur-RF embed AUC Spearman(rat, human) across blocks = {ar:.3f} (n={n_ar})\n")
        md.append(f"- of {len(ha_hot)} hotspots with an Augur AUC, human Augur-RF AUC>={args.auc_thresh}: "
                  f"{n_aug_surv}/{len(ha_hot)}  (independent RF corroboration of the PLS-1 survival above)\n")

    (out / "transfer_comparison.md").write_text("".join(md))
    print("".join(md))
    print(f"\nwrote {out/'transfer_comparison.tsv'} and {out/'transfer_comparison.md'}")


def _fmt(x, signed=False):
    if pd.isna(x):
        return "NA"
    return f"{x:+.3f}" if signed else f"{x:.3f}"


if __name__ == "__main__":
    main()
