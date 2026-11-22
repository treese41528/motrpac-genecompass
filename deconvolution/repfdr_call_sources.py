#!/usr/bin/env python3
"""repfdr_call_sources.py -- where Stage 10's 8-week sex-consistency calls come from, per DE run.

R/run_pseudobulk_de.R makes two calls per gene-block of a two-sex block: concordance (both per-sex 8-week BH
tests significant) and repfdr (replication Fdr < alpha), each read with the signs of the two per-sex z. This
script re-derives the calls and the state sexcons_8w from the run's own tables, counts what each source
contributes, and stops with an error if its recomputation differs from what the run stored. It is
descriptive: no threshold or call is changed.

Two table layouts are read, told apart by the name of the repfdr column:

  repfdr_Fdr       (runs since STEP3_PREREG.md 6.2) Concordance is the primary call and alone decides
                   sexcons_8w. repfdr is fitted on dose-significant gene-blocks only and is reported beside
                   it. The tables carry the calls as sexcons_concordance_8w, sexcons_repfdr_8w and
                   sexcons_union_8w; a one-sex block has the state "single_sex".
  repfdr_localfdr  (earlier runs; the column held the same tail-area Fdr under a wrong name) A gene-block
                   was sex-consistent if repfdr OR concordance called it, and repfdr was fitted on every
                   gene-block. When that fit was degenerate (Fdr = 1 everywhere) the calls were concordance
                   only.

  python deconvolution/repfdr_call_sources.py --de-dir LABEL=DIR [--de-dir LABEL=DIR ...] --out DIR

Each DIR is a pseudobulk_de output (de_methods.tsv, de_summary.tsv, celltype_files.tsv, <TISSUE>/de__*.tsv).
Writes repfdr_call_sources.tsv (one row per run), repfdr_call_sources_by_tissue.tsv and summary.txt.
"""
import argparse, os, sys
import numpy as np
import pandas as pd

BASE_COLS = ["gene", "z_8w_M", "z_8w_F", "fdr_8w_M", "fdr_8w_F", "sexcons_8w"]
CALL_COLS = ["sexcons_concordance_8w", "sexcons_repfdr_8w", "sexcons_union_8w"]
FDR_NEW, FDR_OLD = "repfdr_Fdr", "repfdr_localfdr"


def methods(d):
    out = {}
    for line in open(os.path.join(d, "de_methods.tsv")):
        k, _, v = line.rstrip("\n").partition("\t")
        out[k] = v
    return out


def layout(path):
    """'primary_concordance' for a table with repfdr_Fdr, 'union' for one with repfdr_localfdr."""
    header = open(path).readline().rstrip("\n").split("\t")
    if FDR_NEW in header:
        return "primary_concordance"
    if FDR_OLD in header:
        return "union"
    sys.exit(f"ERROR: {path} has neither {FDR_NEW} nor {FDR_OLD}")


def flag(col):
    """A logical column written by R (TRUE / FALSE / NA) as floats 1 / 0 / nan."""
    return col.astype(str).str.strip().str.upper().map({"TRUE": 1.0, "FALSE": 0.0}).to_numpy(float)


def states(df, alpha, lay):
    """Recompute sexcons_8w exactly as run_pseudobulk_de.R does for this layout, with the two call sources.

    repl and conc are the calls before the sign is read: repfdr Fdr < alpha, and both per-sex BH < alpha.
    """
    lf = df["repfdr_fdr"].to_numpy(float)
    fm, ff = df["fdr_8w_M"].to_numpy(float), df["fdr_8w_F"].to_numpy(float)
    zm, zf = df["z_8w_M"].to_numpy(float), df["z_8w_F"].to_numpy(float)
    repl = np.isfinite(lf) & (lf < alpha)
    sig_m = np.isfinite(fm) & (fm < alpha)
    sig_f = np.isfinite(ff) & (ff < alpha)
    conc = sig_m & sig_f
    with np.errstate(invalid="ignore"):
        up, down = (zm > 0) & (zf > 0), (zm < 0) & (zf < 0)
    cons = conc if lay == "primary_concordance" else (repl | conc)
    st = np.full(len(df), "null", dtype=object)
    st[cons & up] = "up_both"
    st[cons & down] = "down_both"
    st[cons & ~up & ~down] = "opposite"
    st[~cons & sig_m] = "M_only"
    st[~cons & sig_f] = "F_only"
    if lay == "primary_concordance":
        st[df["single_sex"].to_numpy(bool)] = "single_sex"
    return st, repl, conc, up | down


def count(df, st, repl, conc, same_sign, lay):
    cons_dir = np.isin(st, ["up_both", "down_both"])
    lf = df["repfdr_fdr"].to_numpy(float)
    fin = np.isfinite(lf)
    both_z = np.isfinite(df["z_8w_M"].to_numpy(float)) & np.isfinite(df["z_8w_F"].to_numpy(float))
    out = {
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
        # the calls with the sign read: what the three logical columns of the current layout hold
        "n_sexcons_concordance": int((conc & same_sign).sum()),
        "n_sexcons_repfdr": int((repl & same_sign).sum()),
        "n_sexcons_union": int(((conc | repl) & same_sign).sum()),
        "n_up_both": int((st == "up_both").sum()),
        "n_down_both": int((st == "down_both").sum()),
        "n_opposite": int((st == "opposite").sum()),
        "n_single_sex": int((st == "single_sex").sum()),
        "n_sexcons_directional": int(cons_dir.sum()),
        "n_sexcons_directional_repfdr_only": int((cons_dir & repl & ~conc).sum()),
        "n_sexcons_directional_concordance": int((cons_dir & conc).sum()),
        "n_state_mismatch": int((st != df["sexcons_8w"].astype(str).to_numpy()).sum()),
        "n_call_mismatch": 0,
    }
    if lay == "primary_concordance":
        # The stored logical columns against the recomputation, in two-sex blocks. sexcons_repfdr_8w and
        # sexcons_union_8w are missing in the tables when the run has no usable repfdr fit; then there is
        # nothing to compare them with.
        two = ~df["single_sex"].to_numpy(bool)
        want = {"sexcons_concordance_8w": conc & same_sign, "sexcons_repfdr_8w": repl & same_sign,
                "sexcons_union_8w": (conc | repl) & same_sign}
        bad = 0
        for c, w in want.items():
            have = df[c].to_numpy(float)
            cmp = two & np.isfinite(have)
            bad += int((have[cmp] != w[cmp].astype(float)).sum())
            if c == "sexcons_concordance_8w":
                bad += int((two & ~np.isfinite(have)).sum())      # the primary call is never missing there
        out["n_call_mismatch"] = bad
    return out


def load(d):
    """Every per-gene table of the run, through its manifest, with the block's tissue and sex count."""
    man = pd.read_csv(os.path.join(d, "celltype_files.tsv"), sep="\t")
    summ = pd.read_csv(os.path.join(d, "de_summary.tsv"), sep="\t").set_index(["tissue", "cell_type"])
    lay, parts = None, []
    for _, r in man.iterrows():
        path = os.path.join(d, r["file"])
        this = layout(path)
        if lay not in (None, this):
            sys.exit(f"ERROR: {d} mixes table layouts")
        lay = this
        cols = BASE_COLS + ([FDR_NEW] + CALL_COLS if lay == "primary_concordance" else [FDR_OLD])
        t = pd.read_csv(path, sep="\t", usecols=cols, low_memory=False)
        t = t.rename(columns={FDR_NEW: "repfdr_fdr", FDR_OLD: "repfdr_fdr"})
        # R writes the state "null" (never NA); pandas reads the string "null" as missing by default
        t["sexcons_8w"] = t["sexcons_8w"].fillna("null")
        for c in CALL_COLS:
            if c in t:
                t[c] = flag(t[c])
        t["tissue"] = r["tissue"]
        s = summ.loc[(r["tissue"], r["cell_type"])]
        t["single_sex"] = bool(s["n_male"] == 0 or s["n_female"] == 0)
        parts.append(t)
    return pd.concat(parts, ignore_index=True), lay, len(man)


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
        df, lay, n_blocks = load(d)
        st, repl, conc, same = states(df, alpha, lay)
        row = {"run": label, "de_dir": d, "layout": lay, "sex_consistency": meth.get("sex_consistency", ""),
               "warnings": meth.get("sex_consistency_warnings", ""), "alpha": alpha,
               "n_blocks_ok": n_blocks, "min_median_libsize": meth.get("min_median_libsize", "")}
        row.update(count(df, st, repl, conc, same, lay))
        runs.append(row)
        for tname, idx in df.groupby("tissue").indices.items():
            sub = df.iloc[idx]
            tr = {"run": label, "tissue": tname}
            tr.update(count(sub, st[idx], repl[idx], conc[idx], same[idx], lay))
            tiss.append(tr)
        L.append(f"[{label}] {d}")
        L.append(f"  layout: {lay} ("
                 + ("concordance is the primary call; repfdr is fitted on dose-significant gene-blocks"
                    if lay == "primary_concordance" else "repfdr OR concordance; repfdr fitted on every gene-block") + ")")
        L.append(f"  fit: {row['sex_consistency']}")
        L.append(f"  warnings: {row['warnings']}")
        L.append(f"  blocks {row['n_blocks_ok']} (min median libsize {row['min_median_libsize'] or 'none'}); "
                 f"gene-blocks {row['n_gene_blocks']:,}; with both-sex 8w z {row['n_both_sex_z']:,}; "
                 f"repfdr Fdr finite {row['n_repfdr_fdr_finite']:,}, = 1 for {row['frac_fdr_eq_1']:.4f}, "
                 f"min {row['min_fdr']:.3g}")
        L.append(f"  significant at {alpha}: repfdr {row['n_repfdr_sig']:,}; concordance {row['n_concordance_sig']:,}; "
                 f"both {row['n_both_sources']:,}; repfdr only {row['n_repfdr_only']:,}; "
                 f"concordance only {row['n_concordance_only']:,}")
        L.append(f"  sex-consistent with equal sign: concordance {row['n_sexcons_concordance']:,}; "
                 f"repfdr {row['n_sexcons_repfdr']:,}; union {row['n_sexcons_union']:,}")
        L.append(f"  states: up_both {row['n_up_both']:,}, down_both {row['n_down_both']:,}, opposite {row['n_opposite']:,}; "
                 f"of the {row['n_sexcons_directional']:,} directional states, {row['n_sexcons_directional_repfdr_only']:,} "
                 f"rest on repfdr alone; recomputed states that differ from the table: {row['n_state_mismatch']:,}; "
                 f"recomputed calls that differ from the table: {row['n_call_mismatch']:,}")
    pd.DataFrame(runs).to_csv(os.path.join(a.out, "repfdr_call_sources.tsv"), sep="\t", index=False)
    pd.DataFrame(tiss).to_csv(os.path.join(a.out, "repfdr_call_sources_by_tissue.tsv"), sep="\t", index=False)
    open(os.path.join(a.out, "summary.txt"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    if any(r["n_state_mismatch"] or r["n_call_mismatch"] for r in runs):
        sys.exit("ERROR: recomputed states or calls differ from the tables; the call rule above does not match the R code")


if __name__ == "__main__":
    main()
