#!/usr/bin/env python3
"""Summarize the known-truth mixture panel on the current references (MIXTURE_PANEL_REFS4ALL_PREREG.md).

Reads <panel>/<SET>/scores/{overall_metrics,metrics}_<method>.tsv (score_validation.py, truth = rnafrac) and writes
  <out>/panel_median.tsv   median per-type Pearson r, set x method   (headline; the differential question)
  <out>/panel_pooled.tsv   pooled Pearson r, set x method
  <out>/panel_per_type.tsv every benchmarked type's r, per set and method
  <out>/panel_vs_earlier.tsv  BayesPrism median r vs the earlier panel's for the same production tissue,
                           flagged comparable when design (holdout/cross) and cross source match
  <out>/panel_scoring_notes.tsv  benchmarked types a method's median leaves out (not estimated / constant)
  <out>/panel_coverage.tsv benchmarked vs reference types per set, from <panel>/logs/<SET>.log
  <out>/summary.txt        the registered reading (BayesPrism >= 0.85; drops > 0.05; best/within-0.02 counts)
A method with no score file is 'n/a' (not applicable: single-sample reference for MuSiC/Bisque) or 'fail'.

  python deconvolution/mixture_panel_summary.py [--panel DIR] [--earlier DIR] [--out DIR]
"""
import argparse
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
METHODS = ["bp_rna", "dwls", "music", "scdc", "bisque", "cibersortx"]
# earlier panel (validation_v2) set for the same production tissue (OMNIDECONV_RESULTS.md, production-aligned table)
EARLIER = {"WATSC": ["WAT_holdout"], "BLOOD": ["PBMC_holdout"], "HEART": ["HRT_holdout"], "HIPPOC": ["HIP_merged"],
           "KIDNEY": ["KID_cross"], "LUNG": ["LNG_cross"], "MUSCLE": ["GAS_cross", "SKMVL"], "CORTEX": ["CTX_union"],
           "LIVER": ["LIV_holdout"]}


def overall(sd, m):
    f = sd / "scores" / f"overall_metrics_{m}.tsv"
    if not f.exists():
        return None
    d = pd.read_csv(f, sep="\t").set_index("metric")["value"]
    return float(d["median_pearson"]), float(d["overall_pooled_pearson"])


def design(sd):
    """'cross' (mixtures from another study; make_pseudobulk writes a harmonization map) or 'holdout'."""
    hm = sd / "mixtures" / "harmonization_map.tsv"
    return ("cross", frozenset(pd.read_csv(hm, sep="\t")["cell_type"])) if hm.exists() else ("holdout", None)


def coverage(log):
    """From a make_pseudobulk build log: the deconvolution reference's types, the benchmarked ones (>= 20 pool cells),
    and, for a holdout set, the built reference's types that the 70% split lost (too few cells left after the split)."""
    t = log.read_text()
    names = lambda pat: re.search(pat, t).group(1).strip("'").split("', '")
    ref = names(r"reference types: \[(.*?)\]"); bench = names(r"types usable in pool \(>= 20 cells\): \[(.*?)\]")
    m = re.search(r"built reference (\S+): ", t)
    built = sorted(pd.read_csv(ROOT / m.group(1) / "cells_meta.tsv", sep="\t", usecols=["cell_type"])
                   .cell_type.unique()) if m else ref
    return {"n_built": len(built), "n_in_reference": len(ref), "n_benchmarked": len(bench),
            "lost_at_split": "; ".join(x for x in built if x not in ref),
            "not_benchmarked": "; ".join(x for x in ref if x not in bench)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default=str(ROOT / "data/deconvolution/mixture_panel_refs4all"))
    ap.add_argument("--earlier", default=str(ROOT / "data/deconvolution/validation_v2"))
    ap.add_argument("--out", default=None, help="default: <panel>/summary")
    a = ap.parse_args()
    panel = Path(a.panel); out = Path(a.out or panel / "summary"); out.mkdir(parents=True, exist_ok=True)
    sets = sorted([p.name for p in panel.iterdir() if p.name.endswith(("_holdout", "_cross"))],
                  key=lambda s: (s.endswith("_cross"), s))
    med, pool, per = {}, {}, []
    for s in sets:
        sd = panel / s
        med[s], pool[s] = {}, {}
        for m in METHODS:
            o = overall(sd, m)
            if o is None:
                med[s][m] = pool[s][m] = float("nan")
                continue
            med[s][m], pool[s][m] = o
            t = pd.read_csv(sd / "scores" / f"metrics_{m}.tsv", sep="\t")
            for r in t.itertuples():
                per.append({"set": s, "method": m, "cell_type": r.cell_type, "pearson_r": r.pearson_r,
                            "true_mean": r.true_mean, "est_mean": r.est_mean})
    M = pd.DataFrame(med).T[METHODS]; P = pd.DataFrame(pool).T[METHODS]
    M = M.loc[:, M.notna().any()]; P = P[M.columns]
    M.round(3).to_csv(out / "panel_median.tsv", sep="\t"); P.round(3).to_csv(out / "panel_pooled.tsv", sep="\t")
    per = pd.DataFrame(per); per.to_csv(out / "panel_per_type.tsv", sep="\t", index=False)

    lines = ["Known-truth mixture panel on the current references (truth = rnafrac). Median per-type Pearson r:", "",
             M.round(3).fillna("-").to_string(), "", "Pooled Pearson r:", "", P.round(3).fillna("-").to_string(), ""]
    bp = M["bp_rna"]
    lines.append("BayesPrism reading (median per-type r >= 0.85):")
    for s in sets:
        r = bp[s]; verdict = "recovers" if r >= 0.85 else "WEAK"
        line = f"  {s:16s} {r:.3f}  {verdict}"
        if r < 0.85:
            w = per[(per.set == s) & (per.method == "bp_rna")].nsmallest(3, "pearson_r")
            line += "; worst: " + ", ".join(f"{x.cell_type} {x.pearson_r:.2f}" for x in w.itertuples())
        lines.append(line)
    comp = []
    for s in sets:
        code = s.split("_")[0]
        dn, src_now = design(panel / s)
        for e in EARLIER.get(code, []):
            o = overall(Path(a.earlier) / e, "bp_rna")
            if o:
                de, src_e = design(Path(a.earlier) / e)
                # comparable = same design, and for cross sets the same mixture source (identical source-label set)
                comp.append({"set": s, "design": dn, "earlier_set": e, "earlier_design": de,
                             "comparable": dn == de and src_now == src_e,
                             "earlier_median": round(o[0], 3), "now_median": round(bp[s], 3),
                             "change": round(bp[s] - o[0], 3), "drop_gt_0.05": bp[s] - o[0] < -0.05})
    C = pd.DataFrame(comp); C.to_csv(out / "panel_vs_earlier.tsv", sep="\t", index=False)
    lines += ["", "BayesPrism vs the earlier panel (same production tissue; descriptive). comparable = same design and,",
              "for cross sets, the same mixture source; the other rows set a holdout against an earlier cross set:", "",
              C.to_string(index=False), ""]
    best = within = pbest = 0; n = 0
    for s in sets:
        row = M.loc[s].dropna()
        if len(row) < 2:
            continue
        n += 1; best += row.idxmax() == "bp_rna"; within += row["bp_rna"] >= row.max() - 0.02
        pbest += P.loc[s].dropna().idxmax() == "bp_rna"
    lines.append(f"BayesPrism best on {best}/{n} sets; within 0.02 of the best on {within}/{n} (sets with >= 2 methods scored).")
    lines.append(f"On pooled r (descriptive), BayesPrism is best on {pbest}/{n}.")

    # what each median is taken over: benchmarked types a method did not estimate (MuSiC drops types seen in only
    # one subject), and types it estimated as a constant (r undefined; score_validation leaves them out of the median)
    notes = []
    for s in sets:
        bench = set(per[(per.set == s) & (per.method == "bp_rna")].cell_type)
        for m in M.columns:
            got = per[(per.set == s) & (per.method == m)]
            if got.empty:
                continue
            for t in sorted(bench - set(got.cell_type)):
                notes.append({"set": s, "method": m, "cell_type": t, "note": "not estimated (absent from output)"})
            for t in sorted(got[got.pearson_r.isna()].cell_type):
                notes.append({"set": s, "method": m, "cell_type": t, "note": "constant estimate (r undefined)"})
    N = pd.DataFrame(notes, columns=["set", "method", "cell_type", "note"])
    N.to_csv(out / "panel_scoring_notes.tsv", sep="\t", index=False)
    lines += ["", "Types left out of a method's median (scored on the rest):", "",
              N.to_string(index=False) if len(N) else "  none"]

    # what the panel certifies: benchmarked types per set (registered caveat: only those are tested)
    V = pd.DataFrame([{"set": s, **coverage(panel / "logs" / f"{s}.log")} for s in sets])
    V.to_csv(out / "panel_coverage.tsv", sep="\t", index=False)
    lines += ["", "Coverage (types in the deconvolution reference / benchmarked; holdout: types lost at the split):", "",
              V.to_string(index=False)]
    txt = "\n".join(lines); print(txt); (out / "summary.txt").write_text(txt + "\n")


if __name__ == "__main__":
    main()
