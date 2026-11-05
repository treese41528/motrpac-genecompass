#!/usr/bin/env python3
"""Summarize the known-truth mixture panel on the current references (MIXTURE_PANEL_REFS4ALL_PREREG.md).

Reads <panel>/<SET>/scores/{overall_metrics,metrics}_<method>.tsv (score_validation.py, truth = rnafrac) and writes
  <out>/panel_median.tsv   median per-type Pearson r, set x method   (headline; the differential question)
  <out>/panel_pooled.tsv   pooled Pearson r, set x method
  <out>/panel_per_type.tsv every benchmarked type's r, per set and method
  <out>/panel_vs_earlier.tsv  BayesPrism median r vs the earlier panel's for the same production tissue
  <out>/summary.txt        the registered reading (BayesPrism >= 0.85; drops > 0.05; best/within-0.02 counts)
A method with no score file is 'n/a' (not applicable: single-sample reference for MuSiC/Bisque) or 'fail'.

  python deconvolution/mixture_panel_summary.py [--panel DIR] [--earlier DIR] [--out DIR]
"""
import argparse
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
        for e in EARLIER.get(code, []):
            o = overall(Path(a.earlier) / e, "bp_rna")
            if o:
                comp.append({"set": s, "earlier_set": e, "earlier_median": round(o[0], 3), "now_median": round(bp[s], 3),
                             "change": round(bp[s] - o[0], 3), "drop_gt_0.05": bp[s] - o[0] < -0.05})
    C = pd.DataFrame(comp); C.to_csv(out / "panel_vs_earlier.tsv", sep="\t", index=False)
    lines += ["", "BayesPrism vs the earlier panel (same production tissue; descriptive):", "", C.to_string(index=False), ""]
    best = within = 0; n = 0
    for s in sets:
        row = M.loc[s].dropna()
        if len(row) < 2:
            continue
        n += 1; best += row.idxmax() == "bp_rna"; within += row["bp_rna"] >= row.max() - 0.02
    lines.append(f"BayesPrism best on {best}/{n} sets; within 0.02 of the best on {within}/{n} (sets with >= 2 methods scored).")
    txt = "\n".join(lines); print(txt); (out / "summary.txt").write_text(txt + "\n")


if __name__ == "__main__":
    main()
