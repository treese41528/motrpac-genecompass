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

Read-only on the rat tables; writes transfer_comparison.tsv (full paired join),
transfer_comparison.md (the verdict) and rat_only_pseudocells.tsv under --out.

Step 3 additions (docs/audit/remediation/STEP3_PREREG.md section 7; audit findings 8.1, 8.2, 8.4). The
registered rule and its `status` column are unchanged: PRESERVED = human AUC >= --auc-thresh and the raw
10,000-permutation p < --p-thresh. Beside it every row now carries
  q_sup_trained_human, gate_class_human   Benjamini-Hochberg on the human side over the rat gate's analysable
                         family, and the classes of section 5.2 (hotspot / borderline / none) from the human
                         probe's interval limits. A human probe table without limits (from before Step 3)
                         gets plain BH on its p (gate_method_human = bh_10k, no borderline class).
  transfer_corrected     PASS when the human class is hotspot or borderline and the human AUC >= --auc-thresh;
                         FAIL otherwise; no_human_block when the block has no human test.
  dose_trend_lost        true when the human dose correlation is zero or negative.
  gate_class_rat         the rat class, so borderline blocks are carried through this table by name.
rat_only_pseudocells.tsv lists the pseudo-cells that exist in rat space only, by cell_id in the two tokenized
datasets (the transfer drops a cell with too few mapped genes after orthology).

The human test reuses the rat test's cells, labels, folds and seeds. It shows that the model's human-species
reading keeps the structure. It is not a replication, and the markdown says so where it states the count.

Usage:
  python translation/compare_transfer.py \
    --rat-corr   data/deconvolution/genecompass_input_v2/gate_rcv10/corroboration_merged.tsv \
    --rat-probe  data/deconvolution/genecompass_input_v2/gate_rcv10/subspace_probe.tsv \
    --human-probe data/deconvolution/genecompass_input_v2_human/gate_rcv10/subspace_probe.tsv \
    --out        data/deconvolution/genecompass_input_v2_human/gate_rcv10
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
sys.path.insert(0, str(_ROOT / "deconvolution"))
from gene_utils import load_config, resolve_path  # noqa: E402
import mc_gate  # noqa: E402

KEY = ["tissue", "cell_type"]
LIMITS = ["p_lo_sup_trained", "p_hi_sup_trained"]
# Section 7: stated wherever the transfer count is quoted.
REUSE_NOTE = ("The human test reuses the rat test's cells, labels, folds and seeds. It shows that the model's "
              "human-species reading keeps the structure. It is not a replication.")


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


def _true(col):
    """A table column read as text or as booleans -> booleans (missing is false)."""
    return col.astype(str).str.lower() == "true"


def human_gate(m, alpha=mc_gate.ALPHA):
    """Multiplicity control on the human side (section 7), over the rat gate's analysable family.

    m holds one row per block with `family` (the rat family), the human p-values (`p_sup_trained_human` final,
    `p_registered_human` the 10,000-permutation value) and, where the human probe sampled sequentially,
    `in_family_human` and the interval limits `p_lo_sup_trained_human`, `p_hi_sup_trained_human`.
    Returns (q, class, method) as two columns and a word. With limits the classes are those of section 5.2
    and q is BH on the final p; without them q is BH on the registered p and a block is a hotspot when
    q < alpha. Blocks outside the family, or with no human test, are 'none' and have no q.
    """
    fam = m["family"] & m["p_sup_trained_human"].notna()
    q = pd.Series(np.nan, index=m.index, dtype=float)
    cls = pd.Series(mc_gate.NONE, index=m.index, dtype=object)
    lim = [c + "_human" for c in LIMITS]
    if "in_family_human" in m and all(c in m for c in lim):
        differ = m[m["p_sup_trained_human"].notna() & (_true(m["in_family_human"]) != m["family"])]
        if len(differ):
            sys.exit("ERROR: the human probe sampled a different family than the rat gate's analysable blocks:\n"
                     + differ[KEY].to_string(index=False))
        if m.loc[fam, lim].isna().any().any():
            sys.exit("ERROR: family blocks without interval limits in the human probe table")
        method = "mmctest"
        if fam.any():
            q[fam] = mc_gate.bh_adjust(m.loc[fam, "p_sup_trained_human"].to_numpy(dtype=float))
            cls[fam] = mc_gate.classify(m.loc[fam, lim[0]], m.loc[fam, lim[1]], alpha)
    else:
        method = "bh_10k"
        if fam.any():
            q[fam] = mc_gate.bh_adjust(m.loc[fam, "p_registered_human"].to_numpy(dtype=float))
            cls[fam] = np.where(q[fam] < alpha, mc_gate.HOTSPOT, mc_gate.NONE)
    return q, cls, method


def corrected_verdict(human_class, human_auc, auc_thresh):
    """Section 7: a block transfers when it is a hotspot or borderline in human space and its human AUC is at
    least auc_thresh."""
    if pd.isna(human_auc):
        return "no_human_block"
    return "PASS" if human_class in (mc_gate.HOTSPOT, mc_gate.BORDERLINE) and human_auc >= auc_thresh else "FAIL"


def rat_only_pseudocells(rat_root, human_root):
    """Pseudo-cells tokenized in rat space and absent from human space, by cell_id, for every tissue that has
    both datasets. Returns (table, tissues compared). A human cell_id that is not in the rat dataset is an
    error: the transfer can only drop cells."""
    from datasets import load_from_disk
    rows, tissues = [], []
    for d in sorted(Path(rat_root).glob("*/dataset")):
        tis = d.parent.name
        hd = Path(human_root) / tis / "dataset"
        if not hd.is_dir():
            print(f"NOTE: no human dataset for {tis}; not compared")
            continue
        rat = load_from_disk(str(d)).select_columns(["cell_id", "cell_type", "sample"]).to_pandas()
        human_ids = set(load_from_disk(str(hd))["cell_id"])
        extra = human_ids - set(rat["cell_id"])
        if extra:
            sys.exit(f"ERROR: [{tis}] {len(extra)} human-space cell ids are not in the rat dataset, e.g. "
                     f"{sorted(extra)[:3]}")
        only = rat[~rat["cell_id"].isin(human_ids)]
        rows.append(only.assign(tissue=tis)[["tissue", "cell_type", "sample", "cell_id"]])
        tissues.append(tis)
    cols = ["tissue", "cell_type", "sample", "cell_id"]
    return (pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=cols)), tissues


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
                    help="rat q_sup_trained threshold defining exercise hotspots, for a rat table from before "
                         "Step 3 (one with an is_hotspot column is read as it is)")
    ap.add_argument("--alpha", type=float, default=mc_gate.ALPHA,
                    help="Benjamini-Hochberg level on the human side (section 7)")
    ap.add_argument("--rat-root", default=str(gc), help="rat genecompass_input root (tokenized datasets)")
    ap.add_argument("--human-root", default=str(gc) + "_human",
                    help="human genecompass_input root; with --rat-root it gives the rat-only pseudo-cells")
    args = ap.parse_args()

    rat_c = _read(args.rat_corr, KEY + ["sup_trained_auc", "p_sup_trained", "q_sup_trained"], "rat-corr")
    rat_p = _read(args.rat_probe, KEY + ["sup_trained_auc", "sup_dose_rho", "sup_sex_auc"], "rat-probe")
    hum_p = _read(args.human_probe, KEY + ["sup_trained_auc", "p_sup_trained", "sup_dose_rho", "sup_sex_auc"],
                  "human-probe")

    # rat side: AUC + q (and, from Step 3 on, the class and the family) from corroboration; dose_rho + sex from
    # the probe
    rat_new = [c for c in ["p_sup_trained_10k", "gate_class", "is_hotspot", "analysable"] if c in rat_c.columns]
    rat = rat_c[KEY + ["sup_trained_auc", "p_sup_trained", "q_sup_trained"] + rat_new].rename(
        columns={"p_sup_trained_10k": "p_sup_trained_10k_rat", "gate_class": "gate_class_rat"}).merge(
        rat_p[KEY + ["sup_dose_rho", "sup_sex_auc"]], on=KEY, how="left")
    hum_new = [c for c in ["p_sup_trained_10k", "b_sup_trained", "m_sup_trained", "in_family"] + LIMITS
               if c in hum_p.columns]
    m = rat.merge(hum_p[KEY + ["sup_trained_auc", "p_sup_trained", "sup_dose_rho", "sup_sex_auc"] + hum_new]
                  .rename(columns={c: c + "_human" for c in hum_new}),
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

    # Rat hotspots: the gate's is_hotspot column (class hotspot, section 5.2); a table from before Step 3 has
    # only q, and the cut is applied here as it always was.
    new_rat_table = "is_hotspot" in m
    m["is_hotspot"] = _true(m["is_hotspot"]) if new_rat_table else m["q_sup_trained"] < args.hotspot_q
    m["delta_auc"] = m["sup_trained_auc_human"] - m["sup_trained_auc_rat"]
    # The registered rule, unchanged: it reads the 10,000-permutation human p (the only p of an older table).
    m["p_registered_human"] = m["p_sup_trained_10k_human"] if "p_sup_trained_10k_human" in m \
        else m["p_sup_trained_human"]
    m["status"] = m.assign(p_sup_trained_human=m["p_registered_human"]).apply(
        lambda r: classify(r, args.auc_thresh, args.p_thresh), axis=1)

    # Section 7 additions: BH and classes on the human side over the rat family, the corrected verdict, the
    # dose-trend flag.
    m["family"] = _true(m["analysable"]) if "analysable" in m else m["q_sup_trained"].notna()
    m["q_sup_trained_human"], m["gate_class_human"], human_method = human_gate(m, args.alpha)
    m["gate_method_human"] = human_method
    m["transfer_corrected"] = [corrected_verdict(c, a, args.auc_thresh)
                               for c, a in zip(m["gate_class_human"], m["sup_trained_auc_human"])]
    m["dose_trend_lost"] = m["sup_dose_rho_human"] <= 0
    m = m.drop(columns=["family", "p_registered_human"] + [c for c in ["in_family_human"] if c in m])
    m = m.sort_values(["is_hotspot", "sup_trained_auc_rat"], ascending=[False, False])
    first = [c for c in KEY + ["sup_trained_auc_rat", "p_sup_trained_rat", "q_sup_trained", "sup_dose_rho_rat",
                               "sup_sex_auc_rat", "sup_trained_auc_human", "p_sup_trained_human",
                               "sup_dose_rho_human", "sup_sex_auc_human", "augur_rat", "augur_human",
                               "is_hotspot", "delta_auc", "status"] if c in m]
    m = m[first + [c for c in m.columns if c not in first]]

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    m.to_csv(out / "transfer_comparison.tsv", sep="\t", index=False)

    # Pseudo-cells in rat space only (finding 8.4), listed beside the comparison.
    rat_only, compared = None, []
    if Path(args.rat_root).is_dir() and Path(args.human_root).is_dir():
        rat_only, compared = rat_only_pseudocells(args.rat_root, args.human_root)
        rat_only.to_csv(out / "rat_only_pseudocells.tsv", sep="\t", index=False)
    else:
        print(f"NOTE: {args.rat_root} or {args.human_root} is not on disk; rat-only pseudo-cells not listed")

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
    n_corr = int((hot["transfer_corrected"] == "PASS").sum())
    lost = hot[hot["dose_trend_lost"]]
    border = m[m["gate_class_rat"] == mc_gate.BORDERLINE] if "gate_class_rat" in m else m.iloc[0:0]
    hot_rule = "rat gate class 'hotspot'" if new_rat_table else f"rat q_sup_trained < {args.hotspot_q}"

    md = []
    md.append("# Cross-species transfer of the rat exercise response (Aim 3a / Module E, E.2)\n")
    md.append("Does the rat trained-vs-control / ordinal-dose axis SURVIVE re-expression of the rat "
              "pseudo-cells in human GeneCompass embedding space? Paired per (tissue, cell_type) block: "
              "RAT-space vs HUMAN-space supervised PLS-1 probe on the SAME cells.\n")
    md.append(f"\n{REUSE_NOTE}\n\n")
    md.append(f"- blocks compared: {len(m)}   |   exercise hotspots ({hot_rule}): {n_hot}\n")
    if len(border):
        md.append(f"- borderline in rat space, not counted as hotspots ({len(border)}): "
                  + "; ".join(f"{r['tissue']} {r['cell_type']}" for _, r in border.iterrows()) + "\n")

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
    md.append(f"- Registered rule: PRESERVED {n_pres}/{n_hot}  |  WEAKENED {counts.get('WEAKENED', 0)}/{n_hot}  "
              f"|  LOST {counts.get('LOST', 0)}/{n_hot}   "
              f"(PRESERVED = human AUC>={args.auc_thresh} and raw permutation p<{args.p_thresh}, no "
              f"multiplicity control)\n")
    md.append(f"- Corrected rule: transfers {n_corr}/{n_hot}   (hotspot or borderline in human space under "
              f"Benjamini-Hochberg at {args.alpha:g} over the rat gate's analysable family, and human "
              f"AUC>={args.auc_thresh}; human-side method: {human_method})\n")
    md.append(f"- {REUSE_NOTE}\n")
    md.append(f"- Dose trend lost (human dose correlation zero or negative): {len(lost)}/{n_hot}"
              + (": " + "; ".join(f"{r['tissue']} {r['cell_type']} ({_fmt(r['sup_dose_rho_rat'])} -> "
                                  f"{_fmt(r['sup_dose_rho_human'])})" for _, r in lost.iterrows())
                 if len(lost) else "") + "\n")

    def rows_md(frame):
        md.append("\n| tissue | cell_type | rat AUC | human AUC | dAUC | rat dose_rho | human dose_rho | status "
                  "| human q | human class | corrected | dose trend lost |\n")
        md.append("|---|---|---|---|---|---|---|---|---|---|---|---|\n")
        for _, r in frame.iterrows():
            md.append(f"| {r['tissue']} | {r['cell_type']} | {r['sup_trained_auc_rat']:.3f} | "
                      f"{_fmt(r['sup_trained_auc_human'])} | {_fmt(r['delta_auc'], signed=True)} | "
                      f"{_fmt(r['sup_dose_rho_rat'])} | {_fmt(r['sup_dose_rho_human'])} | {r['status']} | "
                      f"{_fmt(r['q_sup_trained_human'], nd=4)} | {r['gate_class_human']} | "
                      f"{r['transfer_corrected']} | {'yes' if r['dose_trend_lost'] else 'no'} |\n")

    rows_md(hot)
    if len(border):
        md.append("\n### Borderline blocks in rat space (carried through by name; not hotspots)\n")
        rows_md(border)

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

    # ---- (5) pseudo-cells that exist in rat space only ----
    if rat_only is not None:
        md.append("\n## (5) Pseudo-cells in rat space only\n")
        md.append(f"- {len(rat_only)} pseudo-cell(s) of {len(compared)} tissues were tokenized in rat space and "
                  "not in human space (too few mapped genes after orthology); listed in "
                  "rat_only_pseudocells.tsv. Every human-space cell is a rat-space cell.\n")
        for (tis, ct), g in rat_only.groupby(["tissue", "cell_type"]):
            md.append(f"- {tis} {ct}: {len(g)}\n")

    (out / "transfer_comparison.md").write_text("".join(md))
    print("".join(md))
    print(f"\nwrote {out/'transfer_comparison.tsv'} and {out/'transfer_comparison.md'}"
          + (f" and {out/'rat_only_pseudocells.tsv'}" if rat_only is not None else ""))


def _fmt(x, signed=False, nd=3):
    if pd.isna(x):
        return "NA"
    return f"{x:+.{nd}f}" if signed else f"{x:.{nd}f}"


if __name__ == "__main__":
    main()
