#!/usr/bin/env python
# coding: utf-8
r"""compare_grn_conservation.py -- Stage 13 step 3: rat-human SELF-CONSISTENCY of the differential networks.

Per TF, the Spearman correlation between its edge differences (delta) in the rat-space network and in the
human-space network of the same cell type, over the targets the two share.

What the score is, and is not. Both networks come from the SAME rat pseudo-cells through the SAME model: the
human-space network re-expresses those cells as human (Stage 12) and switches the species token, 91% of the
ortholog-mapped genes keep the same token, and the bootstrap resamples the same animals in both. No human
measurement enters. The score is therefore the model's self-consistency under that re-expression. It is not
evidence about cross-species biology, and a high value does not show that a TF's targets are shared between
rat and human (docs/audit/09_stage_13_perturbation_grn_enrichment.md, 13e). The script's file name and the
option names --rat-dir / --human-dir keep the older wording; the outputs do not.

Inputs (matched by file name):
  rat   : <rat-dir>/<name>_pooled.tsv     (tf, target[ENSRNOG], delta, z, ...)
  human : <human-dir>/<name>_pooled.tsv   (tf, target_human_ensg, delta, z, ...)
Rat targets are projected to human ENSG (Stage-3 ortholog map); rat targets without a human ortholog drop out.

Which rat edges enter (--select). The default takes a TF's rat edges with |z| >= 2. A TF with fewer than
--min-shared such edges is scored on ALL of its rat edges instead. That fallback used to be silent (audit
finding 9.7); it is now flagged per TF (fallback_all_targets) and counted per network and in the printed
summary, because a score on about 2,000 targets and a score on a few dozen selected ones are not the same
quantity. The human list is never pre-selected.

The p column is scipy's asymptotic two-sided p with targets treated as independent observations, uncorrected.
Targets of one TF are not independent, so it is not a significance of agreement.

Outputs in <out-dir> (default <human-dir>/self_consistency):
  self_consistency_per_tf.tsv        cell_type (the network name), tf, tf_symbol, n_rat_selected,
                                     fallback_all_targets, n_shared, spearman, p, anchor
  self_consistency_per_celltype.tsv  per network: n_tf_rat, n_tf_scored, n_tf_fallback, n_tf_no_human,
                                     n_tf_lt_min_shared, n_tf_nonfinite, median_spearman, frac_pos, frac_rho_ge_0.3
  self_consistency_anchors.tsv       the rows of four TFs named in advance (no pass rule is registered for them)
  provenance_compare_grn_conservation.json

Usage:
  python finetune/genecompass/compare_grn_conservation.py \
    --rat-dir data/deconvolution/grn --human-dir data/deconvolution/grn_human \
    --out-dir data/deconvolution/grn_human/self_consistency
"""
import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

_ROOT = Path(os.environ.get("PIPELINE_ROOT", ".")).resolve()
ANCHORS = {"Mef2c": "myogenic", "Myf6": "myogenic", "Spi1": "myeloid", "Cebpa": "myeloid"}
OUT_SUBDIR = "self_consistency"        # folder under <human-dir>, and the prefix of the output tables
PER_TF_COLUMNS = ["cell_type", "tf", "tf_symbol", "n_rat_selected", "fallback_all_targets", "n_shared",
                  "spearman", "p", "anchor"]
PER_CT_COLUMNS = ["cell_type", "n_tf_rat", "n_tf_scored", "n_tf_fallback", "n_tf_no_human",
                  "n_tf_lt_min_shared", "n_tf_nonfinite", "median_spearman", "frac_pos", "frac_rho_ge_0.3"]


def _base(x):
    return str(x).split(".")[0].upper()


def select_rat_edges(rg, select="z2", min_shared=8):
    """One TF's rat edges that enter the correlation -> (edges, fallback).

    select == "z2": the edges with |z| >= 2. If fewer than `min_shared` of them exist the TF is scored on all
    of its edges and fallback is True (audit finding 9.7: this used to happen without a trace).
    "top40" / "top200": the largest |delta|. "all": every edge. Only "z2" can fall back.
    """
    if select == "z2":
        sel = rg[rg["z"].abs() >= 2]
        return (sel, False) if len(sel) >= min_shared else (rg, True)
    if select in ("top40", "top200"):
        n = 40 if select == "top40" else 200
        return rg.reindex(rg["delta"].abs().sort_values(ascending=False).index).head(n), False
    return rg, False


def score_network(ct, rat, hum, select="z2", top_k=0, min_shared=8):
    """Score every TF of one network. rat and hum carry tf, tf_symbol (rat), delta, z and t_h, the target as a
    human gene id. Returns (per-TF rows, one per-network row with the counts of what was scored and skipped)."""
    rows = []
    n = {"n_tf_rat": 0, "n_tf_no_human": 0, "n_tf_lt_min_shared": 0, "n_tf_nonfinite": 0}
    for tf, rg in rat.groupby("tf"):
        n["n_tf_rat"] += 1
        hg = hum[hum["tf"] == tf]
        if hg.empty:
            n["n_tf_no_human"] += 1
            continue
        fallback = False
        if top_k and top_k > 0:
            # Older path, kept to reproduce the earlier figure: it pre-selects BOTH sides, so an edge enters
            # only if it is already large in rat AND in human, which selects for the agreement being measured.
            rg = rg.reindex(rg["delta"].abs().sort_values(ascending=False).index).head(top_k)
            hg = hg.reindex(hg["delta"].abs().sort_values(ascending=False).index).head(top_k)
        else:
            # Selection on the rat side only; the human list is left whole.
            rg, fallback = select_rat_edges(rg, select, min_shared)
        rmap = rg.groupby("t_h")["delta"].mean()
        hmap = hg.groupby("t_h")["delta"].mean()
        shared = rmap.index.intersection(hmap.index)
        if len(shared) < min_shared:
            n["n_tf_lt_min_shared"] += 1
            continue
        rho, p = spearmanr(rmap.loc[shared].values, hmap.loc[shared].values)
        if not np.isfinite(rho):
            n["n_tf_nonfinite"] += 1
            continue
        sym = rg["tf_symbol"].iloc[0]
        rows.append({"cell_type": ct, "tf": tf, "tf_symbol": sym, "n_rat_selected": int(len(rg)),
                     "fallback_all_targets": bool(fallback), "n_shared": int(len(shared)),
                     "spearman": round(float(rho), 4), "p": float(p), "anchor": ANCHORS.get(sym, "")})
    rhos = [r["spearman"] for r in rows]
    summary = {"cell_type": ct, **n, "n_tf_scored": len(rows),
               "n_tf_fallback": sum(r["fallback_all_targets"] for r in rows),
               "median_spearman": round(float(np.median(rhos)), 4) if rhos else np.nan,
               "frac_pos": round(float(np.mean([r > 0 for r in rhos])), 3) if rhos else np.nan,
               "frac_rho_ge_0.3": round(float(np.mean([r >= 0.3 for r in rhos])), 3) if rhos else np.nan}
    return rows, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rat-dir", default=None, help="default: deconvolution.grn_dir of the active config")
    ap.add_argument("--human-dir", default=None, help="default: deconvolution.grn_human_dir of the active config")
    ap.add_argument("--out-dir", default=None, help=f"default: <human-dir>/{OUT_SUBDIR}")
    ap.add_argument("--ortholog-map", default=str(_ROOT / "data/training/ortholog_mappings/rat_to_human_mapping.pickle"))
    ap.add_argument("--min-shared", type=int, default=8,
                    help="min shared targets to score a TF; also the min number of |z|>=2 rat edges below "
                         "which --select z2 falls back to all of the TF's edges")
    ap.add_argument("--top-k", type=int, default=0,
                    help="per TF, restrict to its top-k RAT targets by |delta| AND its top-k HUMAN targets "
                         "before intersecting. 0 = off (default).\n"
                         "WHY IT IS OFF: when the input networks were capped at 40 edges/TF, `shared` was "
                         "the intersection of the rat's top-40 and the human's top-40 -- an edge only "
                         "entered the correlation if it was already large in BOTH spaces. That is "
                         "selection for the very agreement the Spearman is meant to measure, and it biases "
                         "the score upward (n_shared was ~11 and 97%% of rhos were positive). "
                         "Set --top-k 40 only to reproduce that older number for comparison.")
    ap.add_argument("--select", default="z2", choices=["z2", "conf", "all", "top40", "top200"],
                    help="Which RAT edges enter the correlation. The HUMAN list is never pre-selected: "
                         "selecting on both sides (the old --top-k path) selects for the agreement being measured.\n"
                         "  z2     (DEFAULT) rat edges with |z|>=2; a TF with fewer than --min-shared of them is "
                         "scored on all of its edges and flagged fallback_all_targets.\n"
                         "  all              every shared target (~1,950), most of them with a near-zero delta.\n"
                         "  top40/top200     rat's top-N by |delta|.\n"
                         "  conf             old name of z2, accepted so earlier command lines still run.\n"
                         "The score depends on this choice (about 0.15 with all targets, higher with a selected "
                         "set), so quote the selection with the number. Ignored when --top-k > 0.")
    ap.add_argument("--out-prefix", default="", help="prefix for output filenames (e.g. 'topk40_')")
    args = ap.parse_args()
    if args.select == "conf":
        args.select = "z2"
    sys.path.insert(0, str(_ROOT / "lib"))
    if args.rat_dir is None or args.human_dir is None:   # derive from the config, never a hardcoded arm
        from gene_utils import load_config, resolve_path
        _c = load_config(); _d = _c["deconvolution"]
        args.rat_dir = args.rat_dir or str(resolve_path(_c, _d["grn_dir"]))
        args.human_dir = args.human_dir or str(resolve_path(_c, _d["grn_human_dir"]))
    if args.out_dir is None:
        args.out_dir = str(Path(args.human_dir) / OUT_SUBDIR)
    outd = Path(args.out_dir); outd.mkdir(parents=True, exist_ok=True)
    r2h = {_base(k): _base(v) for k, v in pickle.load(open(args.ortholog_map, "rb")).items()}

    rat_dir, hum_dir = Path(args.rat_dir), Path(args.human_dir)
    pairs = []
    for hf in sorted(hum_dir.glob("*_pooled.tsv")):
        rf = rat_dir / hf.name
        if rf.exists():
            pairs.append((hf.name.replace("_pooled.tsv", ""), rf, hf))
    print(f"matched {len(pairs)} networks with both a rat-space and a human-space table")

    per_tf, per_ct = [], []
    for ct, rf, hf in pairs:
        rat = pd.read_csv(rf, sep="\t"); hum = pd.read_csv(hf, sep="\t")
        rat["t_h"] = rat["target"].map(lambda x: r2h.get(_base(x)))          # rat target -> human ENSG
        rat = rat.dropna(subset=["t_h"])
        hum["t_h"] = hum["target_human_ensg"].map(_base)
        rows, summ = score_network(ct, rat, hum, args.select, args.top_k, args.min_shared)
        per_tf.extend(rows)
        if rows:
            per_ct.append(summ)
            print(f"  {ct:32} TFs={summ['n_tf_scored']:4}  median rho={summ['median_spearman']:.3f}  "
                  f"frac rho>=0.3={summ['frac_rho_ge_0.3']:.2f}  fallback to all targets={summ['n_tf_fallback']}")
        else:
            print(f"  {ct:32} no TF scored ({summ['n_tf_rat']} rat TFs: {summ['n_tf_no_human']} without a "
                  f"human-space table row, {summ['n_tf_lt_min_shared']} with fewer than {args.min_shared} "
                  f"shared targets)")

    px = args.out_prefix + OUT_SUBDIR
    T = pd.DataFrame(per_tf, columns=PER_TF_COLUMNS)
    C = pd.DataFrame(per_ct, columns=PER_CT_COLUMNS)
    T.to_csv(outd / f"{px}_per_tf.tsv", sep="\t", index=False)
    C.to_csv(outd / f"{px}_per_celltype.tsv", sep="\t", index=False)
    anchors = T[T["anchor"] != ""]
    if not anchors.empty:
        anchors.to_csv(outd / f"{px}_anchors.tsv", sep="\t", index=False)
        print("\n== anchor TFs (named in advance; no pass rule is registered for them) ==")
        print(anchors[["cell_type", "tf_symbol", "anchor", "n_shared", "spearman"]].to_string(index=False))

    n_fb = int(T["fallback_all_targets"].sum()) if len(T) else 0
    if len(T):
        sel = ("the top-k path" if args.top_k and args.top_k > 0 else f"--select {args.select}")
        print(f"\nself-consistency ({sel}): {len(T)} TF x network scores, {T['tf'].nunique()} distinct TFs, "
              f"{len(C)} networks; median rho over all scores {T['spearman'].median():.4f}, median of the "
              f"per-network medians {C['median_spearman'].median():.4f}")
        print(f"fallback: {n_fb} of {len(T)} scores ({T.loc[T['fallback_all_targets'], 'tf'].nunique()} distinct "
              f"TFs) had fewer than {args.min_shared} rat edges with |z|>=2 and were scored on all of the TF's "
              f"targets (flag fallback_all_targets)")
    print(f"wrote {outd}/{px}_per_tf.tsv, {px}_per_celltype.tsv")

    from provenance import write_provenance
    params = dict(vars(args))
    params.update({"n_networks": len(pairs), "n_scores": int(len(T)), "n_distinct_tfs": int(T["tf"].nunique()),
                   "n_scores_fallback_all_targets": n_fb,
                   "reading": "model self-consistency under re-expression as human; not cross-species biology"})
    write_provenance(outd, inputs=[rat_dir, hum_dir, args.ortholog_map], params=params,
                     name=f"provenance_{args.out_prefix}compare_grn_conservation.json")


if __name__ == "__main__":
    main()
