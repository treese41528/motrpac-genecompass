#!/usr/bin/env python3
"""repfdr_call_sources.py -- where Stage 10's 8-week sex-consistency calls come from, per DE run.

R/run_pseudobulk_de.R calls a gene-block sex-consistent if repfdr's replication Fdr < alpha OR both per-sex 8-week
BH tests are significant (concordance), with matching signs (up_both / down_both; mixed signs = opposite). When
repfdr's fit is degenerate (Fdr = 1 everywhere) it records that in de_methods.tsv and the calls are concordance
only. This re-derives, from the run's own tables, the fit status and how many calls each source contributes.
Descriptive: no threshold or call is changed.

  python deconvolution/repfdr_call_sources.py --de-dir LABEL=DIR [--de-dir LABEL=DIR ...] --out DIR

Each DIR is a pseudobulk_de output (de_methods.tsv, celltype_files.tsv, <TISSUE>/de__*.tsv). Writes
repfdr_call_sources.tsv (one row per run), repfdr_call_sources_by_tissue.tsv and summary.txt.
"""
import argparse, os, sys
import numpy as np
import pandas as pd

COLS = ["gene", "z_8w_M", "z_8w_F", "fdr_8w_M", "fdr_8w_F", "repfdr_localfdr", "sexcons_8w"]


def methods(d):
    out = {}
    for line in open(os.path.join(d, "de_methods.tsv")):
        k, _, v = line.rstrip("\n").partition("\t")
        out[k] = v
    return out


def states(df, alpha):
    """Recompute sexcons_8w exactly as run_pseudobulk_de.R does, with the call source."""
    lf = df["repfdr_localfdr"].to_numpy(float)
    fm, ff = df["fdr_8w_M"].to_numpy(float), df["fdr_8w_F"].to_numpy(float)
    zm, zf = df["z_8w_M"].to_numpy(float), df["z_8w_F"].to_numpy(float)
    repl = np.isfinite(lf) & (lf < alpha)
    sig_m = np.isfinite(fm) & (fm < alpha)
    sig_f = np.isfinite(ff) & (ff < alpha)
    conc = sig_m & sig_f
    with np.errstate(invalid="ignore"):
        up, down = (zm > 0) & (zf > 0), (zm < 0) & (zf < 0)
    cons = repl | conc
    st = np.full(len(df), "null", dtype=object)
    st[cons & up] = "up_both"
    st[cons & down] = "down_both"
    st[cons & ~up & ~down] = "opposite"
    st[~cons & sig_m] = "M_only"
    st[~cons & sig_f] = "F_only"
    return st, repl, conc


def count(df, st, repl, conc):
    cons_dir = np.isin(st, ["up_both", "down_both"])
    lf = df["repfdr_localfdr"].to_numpy(float)
    fin = np.isfinite(lf)
    both_z = np.isfinite(df["z_8w_M"].to_numpy(float)) & np.isfinite(df["z_8w_F"].to_numpy(float))
    return {
        "n_gene_blocks": len(df),
        "n_both_sex_z": int(both_z.sum()),
        "n_repfdr_fdr_finite": int(fin.sum()),
        "frac_fdr_eq_1": float((lf[fin] >= 1 - 1e-9).mean()) if fin.any() else np.nan,
        "min_fdr": float(lf[fin].min()) if fin.any() else np.nan,
        "n_repfdr_sig": int(repl.sum()),
        "n_concordance_sig": int(conc.sum()),
        "n_both_sources": int((repl & conc).sum()),
        "n_repfdr_only": int((repl & ~conc).sum()),
        "n_concordance_only": int((conc & ~repl).sum()),
        "n_up_both": int((st == "up_both").sum()),
        "n_down_both": int((st == "down_both").sum()),
        "n_opposite": int((st == "opposite").sum()),
        "n_sexcons_directional": int(cons_dir.sum()),
        "n_sexcons_directional_repfdr_only": int((cons_dir & repl & ~conc).sum()),
        "n_sexcons_directional_concordance": int((cons_dir & conc).sum()),
        "n_state_mismatch": int((st != df["sexcons_8w"].astype(str).to_numpy()).sum()),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--de-dir", action="append", required=True, help="LABEL=DIR (repeatable)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    runs, tiss, L = [], [], []
    for spec in a.de_dir:
        label, _, d = spec.partition("=")
        meth = methods(d)
        alpha = float(meth.get("alpha", "0.05"))
        man = pd.read_csv(os.path.join(d, "celltype_files.tsv"), sep="\t")
        parts = []
        for _, r in man.iterrows():
            t = pd.read_csv(os.path.join(d, r["file"]), sep="\t", usecols=COLS, low_memory=False)
            t["tissue"] = r["tissue"]
            parts.append(t)
        df = pd.concat(parts, ignore_index=True)
        st, repl, conc = states(df, alpha)
        row = {"run": label, "de_dir": d, "sex_consistency": meth.get("sex_consistency", ""),
               "warnings": meth.get("sex_consistency_warnings", ""), "alpha": alpha,
               "n_blocks_ok": len(man), "min_median_libsize": meth.get("min_median_libsize", "")}
        row.update(count(df, st, repl, conc))
        runs.append(row)
        for tname, idx in df.groupby("tissue").indices.items():
            sub = df.iloc[idx]
            tr = {"run": label, "tissue": tname}
            tr.update(count(sub, st[idx], repl[idx], conc[idx]))
            tiss.append(tr)
        L.append(f"[{label}] {d}")
        L.append(f"  fit: {row['sex_consistency']}")
        L.append(f"  warnings: {row['warnings']}")
        L.append(f"  blocks {row['n_blocks_ok']} (min median libsize {row['min_median_libsize'] or 'none'}); "
                 f"gene-blocks {row['n_gene_blocks']:,}; with both-sex 8w z {row['n_both_sex_z']:,}; "
                 f"repfdr Fdr finite {row['n_repfdr_fdr_finite']:,}, = 1 for {row['frac_fdr_eq_1']:.4f}, "
                 f"min {row['min_fdr']:.3g}")
        L.append(f"  significant at {alpha}: repfdr {row['n_repfdr_sig']:,}; concordance {row['n_concordance_sig']:,}; "
                 f"both {row['n_both_sources']:,}; repfdr only {row['n_repfdr_only']:,}; "
                 f"concordance only {row['n_concordance_only']:,}")
        L.append(f"  calls: up_both {row['n_up_both']:,}, down_both {row['n_down_both']:,}, opposite {row['n_opposite']:,}; "
                 f"of the {row['n_sexcons_directional']:,} directional calls, {row['n_sexcons_directional_repfdr_only']:,} "
                 f"rest on repfdr alone; recomputed states that differ from the table: {row['n_state_mismatch']:,}")
    pd.DataFrame(runs).to_csv(os.path.join(a.out, "repfdr_call_sources.tsv"), sep="\t", index=False)
    pd.DataFrame(tiss).to_csv(os.path.join(a.out, "repfdr_call_sources_by_tissue.tsv"), sep="\t", index=False)
    open(os.path.join(a.out, "summary.txt"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    if any(r["n_state_mismatch"] for r in runs):
        sys.exit("ERROR: recomputed states differ from sexcons_8w; the call rule above does not match the R code")


if __name__ == "__main__":
    main()
