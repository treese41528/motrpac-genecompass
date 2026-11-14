#!/usr/bin/env python
"""purity_sweeps_v1.py -- Step 4, V1: expression accuracy by purity sweep, for every tissue's dominant type and for
every hotspot and borderline cell type (docs/audit/remediation/STEP4_PREREG.md, V1).

  plan   writes the list of sweeps from the refs5 outputs:
           dominant  the cell type with the highest mean estimated fraction in each tissue's Stage 8 output,
                     on Stage 14's grid (0.1, 0.3, 0.5, 0.7, 0.85, 0.95)
           hotspot   every block the gate classes as hotspot or borderline, on the registered grid
                     (0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9)
         A tissue pair that shares one reference (SKMGN, SKMVL) is swept once per focal type.
  run    runs one sweep of the plan with the commands Stage 14 uses (make_purity_sweep -> run_deconvolution ->
         extract_z -> score_z_vst -> score_purity_sweep), into <out_root>/<tag>/
  score  applies the registered pass rule to every finished sweep and writes one table

Pass rule (registered): PASS if the median Pearson correlation on variance-stabilised expression is at least 0.95 at
every purity of 0.5 and above. For a hotspot type, "validated at its abundance" if it is at least 0.95 at the grid
point nearest the type's mean estimated fraction in the bulk.

  purity_sweeps_v1.py plan  --results-root R --gate-table G --out-root O
  purity_sweeps_v1.py run   --out-root O --index I
  purity_sweeps_v1.py score --out-root O
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(os.environ.setdefault("PIPELINE_ROOT", str(Path(__file__).resolve().parents[2])))
sys.path.insert(0, str(ROOT / "deconvolution"))
from celltype_names import safe          # noqa: E402

PY = os.environ.get("DECONV_PYTHON") or sys.executable
GRID_DOMINANT = "0.1,0.3,0.5,0.7,0.85,0.95"
GRID_HOTSPOT = "0.005,0.01,0.02,0.05,0.1,0.2,0.3,0.5,0.7,0.9"
PASS_R, PASS_FROM = 0.95, 0.5


def plan(a):
    refs = yaml.safe_load(open(ROOT / "deconvolution/tissue_references.yaml"))
    res = Path(a.results_root); out = Path(a.out_root); out.mkdir(parents=True, exist_ok=True)
    rows, seen = [], set()

    def add(kind, tis, focal, grid, mean_frac):
        ref = refs[tis]["reference_dir"]
        key = (ref, focal, kind)
        if key in seen:
            return
        seen.add(key)
        rows.append({"tag": f"{kind}__{tis.lower()}__{safe(focal)}", "kind": kind, "tissue": tis, "reference_dir": ref,
                     "focal": focal, "grid": grid, "mean_fraction_in_bulk": mean_frac})

    for d in sorted(p for p in res.iterdir() if (p / "estimated_fractions.csv").exists()):
        th = pd.read_csv(d / "estimated_fractions.csv", index_col=0).mean()
        add("dominant", d.name, th.idxmax(), GRID_DOMINANT, float(th.max()))
    g = pd.read_csv(a.gate_table, sep="\t")
    if "gate_class" not in g.columns:
        sys.exit("the gate table has no gate_class column: it is not a refs5 table")
    for _, r in g[g.gate_class.isin(["hotspot", "borderline"])].iterrows():
        tis = str(r.tissue).upper()
        th = pd.read_csv(res / tis / "estimated_fractions.csv", index_col=0).mean()
        add("hotspot", tis, r.cell_type, GRID_HOTSPOT, float(th.get(r.cell_type, np.nan)))
    p = pd.DataFrame(rows)
    p.insert(0, "index", range(len(p)))
    p.to_csv(out / "sweep_plan.tsv", sep="\t", index=False)
    print(p.to_string(index=False))


def run(a):
    out = Path(a.out_root)
    r = pd.read_csv(out / "sweep_plan.tsv", sep="\t").set_index("index").loc[a.index]
    S = out / r.tag
    if (S / "scores" / "purity_sweep_summary.tsv").exists() and not a.force:
        print(f"[skip] {r.tag}: already scored"); return
    R = ROOT / "deconvolution" / "R"
    cmds = [
        [PY, str(ROOT / "deconvolution/make_purity_sweep.py"), "--mode", "holdout", "--ref-source", str(ROOT / r.reference_dir),
         "--tissue", r.tissue.lower(), "--focal-type", r.focal, "--out", str(S), "--purity-grid", r.grid, "--reps", "10",
         "--cells-per-mixture", "1000"],
        ["bash", str(R / "run_deconvolution.sh"), str(S / "reference"), str(S / "mixtures"), str(S / "results")],
        ["bash", str(R / "extract_z.sh"), str(S / "results" / "bp_result.rds"), str(S / "results")],
        ["bash", str(R / "score_z_vst.sh"), str(S), r.focal],
        [PY, str(ROOT / "deconvolution/score_purity_sweep.py"), "--stage-dir", str(S)],
    ]
    for c in cmds:
        print("+ " + " ".join(c), flush=True)
        rc = subprocess.run(c, cwd=ROOT).returncode
        if rc != 0:
            (S / "FAILED.txt").write_text(f"step failed (rc {rc}): {' '.join(c)}\n")
            sys.exit(f"{r.tag}: step failed (rc {rc})")


def score(a):
    out = Path(a.out_root)
    p = pd.read_csv(out / "sweep_plan.tsv", sep="\t")
    rows, pts = [], []
    for _, r in p.iterrows():
        f = out / r.tag / "scores" / "purity_sweep_summary.tsv"
        base = {"tag": r.tag, "kind": r.kind, "tissue": r.tissue, "focal": r.focal, "mean_fraction_in_bulk": r.mean_fraction_in_bulk}
        if not f.exists():
            why = (out / r.tag / "FAILED.txt").read_text().strip() if (out / r.tag / "FAILED.txt").exists() else "not run"
            rows.append({**base, "status": "not testable or not run", "note": why[:160]}); continue
        s = pd.read_csv(f, sep="\t").sort_values("target_purity")
        for _, q in s.iterrows():
            pts.append({**base, "target_purity": q.target_purity, "n": q.n, "pearson_vst_median": q.pearson_vst_median,
                        "pearson_vst_q1": q.pearson_vst_q1, "pearson_vst_q3": q.pearson_vst_q3, "mean_focal_rnafrac": q.mean_focal_rnafrac})
        hi = s[s.target_purity >= PASS_FROM]
        ok = bool(len(hi) and (hi.pearson_vst_median >= PASS_R).all())
        row = {**base, "status": "PASS" if ok else "FAIL", "min_median_r_at_0.5_and_above": float(hi.pearson_vst_median.min()) if len(hi) else np.nan,
               "lowest_purity_with_r_at_least_0.95": float(s[s.pearson_vst_median >= PASS_R].target_purity.min()) if (s.pearson_vst_median >= PASS_R).any() else np.nan}
        if r.kind == "hotspot" and r.mean_fraction_in_bulk == r.mean_fraction_in_bulk:
            near = s.iloc[(s.target_purity - r.mean_fraction_in_bulk).abs().argsort().iloc[0]]
            row.update(purity_nearest_its_abundance=float(near.target_purity), median_r_there=float(near.pearson_vst_median),
                       validated_at_its_abundance=bool(near.pearson_vst_median >= PASS_R))
        rows.append(row)
    pd.DataFrame(rows).to_csv(out / "sweep_verdicts.tsv", sep="\t", index=False)
    pd.DataFrame(pts).to_csv(out / "sweep_points.tsv", sep="\t", index=False)
    pd.set_option("display.width", 250)
    print(pd.DataFrame(rows).round(3).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("plan"); s.add_argument("--results-root", required=True); s.add_argument("--gate-table", required=True); s.add_argument("--out-root", required=True)
    s = sub.add_parser("run"); s.add_argument("--out-root", required=True); s.add_argument("--index", type=int, required=True); s.add_argument("--force", action="store_true")
    s = sub.add_parser("score"); s.add_argument("--out-root", required=True)
    a = ap.parse_args()
    {"plan": plan, "run": run, "score": score}[a.cmd](a)
