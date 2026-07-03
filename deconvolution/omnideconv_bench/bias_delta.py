#!/usr/bin/env python3
"""bias_delta.py -- assemble the omnideconv mRNA-bias battery result (paper Fig 4 analog).

For each tissue in data/deconvolution/simbu_bench/<T>/{bias,nobias}, read each method's
scores and report ΔRMSE = RMSE(bias) - RMSE(no-bias). ΔRMSE ~ 0 => the method CORRECTS the
mRNA-content bias; ΔRMSE > 0 => it degrades (fails to correct, e.g. over-estimates high-mRNA
types). Also reports Δ(pooled Pearson) as a secondary view. Pure post-hoc over score files;
run after slurm/analysis/simbu_mrna_bias.slurm. No truncation: every tissue/method with both
conditions scored is included.
"""
import os, csv, sys, statistics as st

ROOT = "data/deconvolution/simbu_bench"
METHODS = ["bp_rna", "cibersortx", "dwls", "music", "scdc", "bisque"]
# expected from the paper: correctors (dwls/music/scdc) ~0; non-correctors (bp_rna/cibersortx) >0

def overall(path, key):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        for r in csv.reader(f, delimiter="\t"):
            if r and r[0] == key:
                try: return float(r[1])
                except ValueError: return None
    return None

def pertype(path):
    """{cell_type: (rmse, true_mean, est_mean)} from a per-type metrics_<m>.tsv"""
    d = {}
    if not os.path.exists(path):
        return d
    with open(path) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            try:
                d[r["cell_type"]] = (float(r["rmse"]), float(r["true_mean"]), float(r["est_mean"]))
            except (ValueError, KeyError):
                pass
    return d

def main():
    tissues = sorted(d for d in os.listdir(ROOT)
                     if os.path.isdir(os.path.join(ROOT, d)) and d != "doseresp") \
        if os.path.isdir(ROOT) else []
    if not tissues:
        sys.exit(f"no tissues under {ROOT} (run simbu_mrna_bias.slurm first)")

    for metric, key, tag in [
        ("ΔRMSE (bias - nobias): >0 = FAILS to correct mRNA bias", "overall_pooled_rmse", "d"),
        ("Δ pooled Pearson r (bias - nobias): <0 = degraded by bias", "overall_pooled_pearson", "r"),
    ]:
        print(f"\n=== {metric} ===")
        print(f"{'tissue':14s} " + " ".join(f"{m:>10s}" for m in METHODS))
        for T in tissues:
            row = f"{T:14s} "
            cells = []
            for m in METHODS:
                b  = overall(f"{ROOT}/{T}/bias/scores/overall_metrics_{m}.tsv", key)
                nb = overall(f"{ROOT}/{T}/nobias/scores/overall_metrics_{m}.tsv", key)
                cells.append(f"{(b-nb):>+10.3f}" if (b is not None and nb is not None) else f"{'-':>10s}")
            print(row + " ".join(cells))

    # per-method summary across tissues (mean ΔRMSE) — the headline corrector ranking
    print("\n=== mean ΔRMSE across tissues (corrector ranking; lower = better) ===")
    for m in METHODS:
        ds = []
        for T in tissues:
            b  = overall(f"{ROOT}/{T}/bias/scores/overall_metrics_{m}.tsv", "overall_pooled_rmse")
            nb = overall(f"{ROOT}/{T}/nobias/scores/overall_metrics_{m}.tsv", "overall_pooled_rmse")
            if b is not None and nb is not None:
                ds.append(b - nb)
        if ds:
            print(f"  {m:12s} mean ΔRMSE = {sum(ds)/len(ds):+.4f}  (n={len(ds)} tissues)")

    # --- mRNA-bias driver: WHICH cell type does the bias inflate? ---
    # The driver is the HIGH-mRNA type, NOT the most cell-abundant one, so key on the largest
    # per-type ΔRMSE (the type whose recovery degrades most under bias), and show its
    # estimate inflation (est no-bias -> est bias vs the fixed true cell fraction). If the same
    # (parenchymal) type is the driver across methods, the mRNA bias is landing there for all of
    # them -- the rat parenchyma-dominance modulation of the paper's finding. Also report the
    # driver's ΔRMSE vs the median ΔRMSE of the rest (concentration of the effect).
    print("\n=== mRNA-bias driver per tissue x method: max-ΔRMSE type (the high-mRNA type) ===")
    print(f"{'tissue':14s} {'method':10s} {'bias-driver type':26s} {'Δrmse':>7s} {'restMedΔ':>9s} "
          f"{'true':>6s} {'estNB->B':>15s}")
    for T in tissues:
        for m in METHODS:
            b  = pertype(f"{ROOT}/{T}/bias/scores/metrics_{m}.tsv")
            nb = pertype(f"{ROOT}/{T}/nobias/scores/metrics_{m}.tsv")
            common = [ct for ct in b if ct in nb]
            if not common:
                continue
            delta = {ct: b[ct][0] - nb[ct][0] for ct in common}   # per-type ΔRMSE
            drv = max(common, key=lambda ct: delta[ct])           # largest ΔRMSE = bias driver
            rest = [delta[ct] for ct in common if ct != drv]
            print(f"{T:14s} {m:10s} {drv[:26]:26s} {delta[drv]:>+7.3f} "
                  f"{(st.median(rest) if rest else float('nan')):>+9.3f} "
                  f"{nb[drv][1]:>6.3f} {nb[drv][2]:>6.3f}->{b[drv][2]:.3f}")
        print()

if __name__ == "__main__":
    main()
