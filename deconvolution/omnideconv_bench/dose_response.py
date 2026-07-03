#!/usr/bin/env python3
"""dose_response.py -- the controlled mRNA-bias magnitude test.

For each tissue under data/deconvolution/simbu_bench/doseresp/<T>/a<alpha>/, tabulate pooled RMSE
(and pooled Pearson r) vs the bias amplitude alpha, per method. Composition is pinned across alpha
(only the mRNA weighting changes), so a method whose RMSE RISES with alpha is failing to correct the
bias -- and the SLOPE is its bias sensitivity. Comparing liver (high mRNA-content spread) vs PBMC
(low spread): if the correctors' slopes are steep on liver but flat on PBMC, the paper's
"correctors correct" is magnitude-limited and driven by mRNA-content spread, not species.
"""
import os, csv, sys

ROOT = "data/deconvolution/simbu_bench/doseresp"
METHODS = ["bp_rna", "dwls", "music", "scdc", "bisque"]

def overall(path, key):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        for r in csv.reader(f, delimiter="\t"):
            if r and r[0] == key:
                try: return float(r[1])
                except ValueError: return None
    return None

def alphas_for(T):
    d = os.path.join(ROOT, T)
    if not os.path.isdir(d):
        return []
    out = []
    for name in os.listdir(d):
        if name.startswith("a") and os.path.isdir(os.path.join(d, name)):
            try: out.append((float(name[1:]), name))
            except ValueError: pass
    return sorted(out)

def main():
    if not os.path.isdir(ROOT):
        sys.exit(f"no dose-response output under {ROOT} (run simbu_doseresp.slurm first)")
    tissues = sorted(d for d in os.listdir(ROOT) if os.path.isdir(os.path.join(ROOT, d)))
    for key, label in [("overall_pooled_rmse", "pooled RMSE (rises = fails to correct)"),
                       ("overall_pooled_pearson", "pooled Pearson r (falls = degraded)")]:
        for T in tissues:
            al = alphas_for(T)
            if not al:
                continue
            print(f"\n=== {T}: {label} vs bias amplitude alpha ===")
            print(f"{'alpha':>6s} " + " ".join(f"{m:>9s}" for m in METHODS))
            series = {m: [] for m in METHODS}
            for a, name in al:
                vals = []
                for m in METHODS:
                    v = overall(f"{ROOT}/{T}/{name}/scores/overall_metrics_{m}.tsv", key)
                    vals.append(v)
                    series[m].append((a, v))
                print(f"{a:>6.1f} " + " ".join(f"{v:>9.3f}" if v is not None else f"{'-':>9s}" for v in vals))
            if key == "overall_pooled_rmse":
                # slope from alpha=0 baseline to max alpha (RMSE increase per unit alpha)
                print(f"{'slope':>6s} " + " ".join(
                    (lambda pts: f"{((pts[-1][1]-pts[0][1])/(pts[-1][0]-pts[0][0])):>+9.3f}"
                     if len(pts) >= 2 and pts[0][1] is not None and pts[-1][1] is not None
                     and pts[-1][0] != pts[0][0] else f"{'-':>9s}")(
                        [(a, v) for a, v in series[m] if v is not None])
                    for m in METHODS))

if __name__ == "__main__":
    main()
