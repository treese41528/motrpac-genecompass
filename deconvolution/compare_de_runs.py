#!/usr/bin/env python3
"""compare_de_runs.py -- does the headline dose result survive technical covariates? (STEP3_PREREG.md 6.4)

The headline differential expression (R/run_pseudobulk_de.R) has no technical covariate. The registration
reruns the whole DE twice with covariates in every model, into sibling folders:

  A   RIN and percent globin (the covariates of the earlier check)
  B   RIN, percent globin, percent UMI duplication and median 5'-3' bias (the consortium's four)

This script compares the headline folder with each rerun on the headline statistic itself: the dose p
(P_fisher) and the call "dose-significant" (FDR_IHW < alpha, each folder with its own pooled IHW fit).
Per analysed block of the headline, and per rerun:

  n_sig_<V>               dose-significant genes in the rerun
  n_retained_<V>          dose-significant in the headline and in the rerun
  n_lost_<V>              dose-significant in the headline, not in the rerun (a gene the rerun did not test
                          counts as lost)
  n_gained_<V>            dose-significant in the rerun, not in the headline
  retained_share_<V>      n_retained / dose-significant genes in the headline (empty when there are none)
  spearman_neglog10p_<V>  Spearman correlation of -log10(P_fisher) over the genes with a dose p in both

Verdict per hotspot, as registered (the thresholds are those of the earlier check):
  ROBUST        correlation >= 0.9 and retained share >= 0.8 under both A and B
  NOT ROBUST    either condition fails under A or under B
  NO CALLS      the hotspot has no dose-significant gene in the headline; it is not counted as robust
  NOT ASSESSED  rerun A or B was not given, or did not analyse the block
  NOT ANALYSED  the headline did not analyse the block
Blocks that are not hotspots get the same numbers and no verdict.

Blocks are located through each folder's celltype_files.tsv, never by a file-name pattern.

  python deconvolution/compare_de_runs.py --headline DE_DIR --variant A=DE_DIR_covA --variant B=DE_DIR_covB
      [--out TABLE]        (default: <headline>/de_covariate_robustness.tsv)

R/de_technical_covariate_robustness.R is the earlier check on a different statistic (the ordinal slope with
per-block BH, RIN and globin only); this script is the check on the statistic the headline reports.
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(os.environ.get("PIPELINE_ROOT") or Path(__file__).resolve().parents[1]) / "lib"))
from provenance import write_provenance  # noqa: E402

REGISTERED_VARIANTS = ("A", "B")   # STEP3_PREREG.md 6.4: the verdict needs both
MIN_SPEARMAN = 0.9                 # registered thresholds (those of the earlier RIN/globin check)
MIN_RETAINED = 0.8
LOW_FRACTION = 0.01                # STEP3_PREREG.md 6.3; used only when de_summary.tsv has no low_fraction column
GENE_COLS = ["gene", "P_fisher", "FDR_IHW"]


def read_methods(de_dir):
    """de_methods.tsv as a dict (key <tab> value per line)."""
    out = {}
    with open(Path(de_dir) / "de_methods.tsv") as fh:
        for line in fh:
            k, _, v = line.rstrip("\n").partition("\t")
            out[k] = v
    return out


def is_true(series):
    """Boolean view of a column R or pandas wrote as TRUE/FALSE, True/False or 1/0."""
    return series.astype(str).str.strip().str.upper().isin(["TRUE", "T", "1"])


def read_run(de_dir):
    """One DE folder -> (summary keyed by (tissue, cell_type), {(tissue, cell_type): table path}, alpha)."""
    de_dir = Path(de_dir)
    summ = pd.read_csv(de_dir / "de_summary.tsv", sep="\t")
    summ["is_hotspot"] = is_true(summ["is_hotspot"])
    if "gate_class" not in summ.columns:              # a summary written before the gate had classes
        summ["gate_class"] = np.where(summ["is_hotspot"], "hotspot", "")
    summ["gate_class"] = summ["gate_class"].fillna("").astype(str)
    if "low_fraction" in summ.columns:
        low = summ["low_fraction"].astype(str).str.strip().str.upper()
        summ["low_fraction"] = low.map({"TRUE": True, "FALSE": False})      # missing stays missing
    else:
        frac = pd.to_numeric(summ["mean_fraction"], errors="coerce")
        summ["low_fraction"] = (frac < LOW_FRACTION).where(frac.notna())
    summ = summ.set_index(["tissue", "cell_type"], drop=False)
    man = pd.read_csv(de_dir / "celltype_files.tsv", sep="\t")
    files = {(r.tissue, r.cell_type): de_dir / r.file for r in man.itertuples()}
    alpha = float(read_methods(de_dir).get("alpha", "0.05"))
    return summ, files, alpha


def load_block(path):
    """A block's per-gene table: gene, dose p and IHW-adjusted p, indexed by gene."""
    t = pd.read_csv(path, sep="\t", usecols=GENE_COLS)
    if t["gene"].duplicated().any():
        raise ValueError(f"duplicate gene ids in {path}")
    return t.set_index("gene")


def spearman(x, y):
    """Spearman correlation with average ranks; nan under three pairs or when either side is constant."""
    x, y = pd.Series(np.asarray(x, float)), pd.Series(np.asarray(y, float))
    if len(x) < 3:
        return float("nan")
    rx, ry = x.rank().to_numpy(), y.rank().to_numpy()
    if np.ptp(rx) == 0 or np.ptp(ry) == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def compare_block(head, var, alpha):
    """Headline block table against the same block of a rerun."""
    sig_h = head["FDR_IHW"].notna() & (head["FDR_IHW"] < alpha)
    sig_v = var["FDR_IHW"].notna() & (var["FDR_IHW"] < alpha)
    genes_h, genes_v = set(head.index[sig_h]), set(var.index[sig_v])
    both = head.index.intersection(var.index)
    ph, pv = head.loc[both, "P_fisher"].to_numpy(float), var.loc[both, "P_fisher"].to_numpy(float)
    fin = np.isfinite(ph) & np.isfinite(pv)
    with np.errstate(divide="ignore"):
        rho = spearman(-np.log10(ph[fin]), -np.log10(pv[fin]))
    retained = len(genes_h & genes_v)
    return {
        "n_genes_compared": int(fin.sum()),
        "n_sig": len(genes_v),
        "n_retained": retained,
        "n_lost": len(genes_h - genes_v),
        "n_gained": len(genes_v - genes_h),
        "retained_share": retained / len(genes_h) if genes_h else float("nan"),
        "spearman_neglog10p": rho,
    }


def hotspot_verdict(row, labels):
    """The registered verdict of one hotspot row of the comparison table."""
    if row["status_headline"] != "ok":
        return "NOT ANALYSED"
    if row["n_sig_headline"] == 0:
        return "NO CALLS"
    if any(v not in labels or row[f"status_{v}"] != "ok" for v in REGISTERED_VARIANTS):
        return "NOT ASSESSED"
    ok = all(row[f"spearman_neglog10p_{v}"] >= MIN_SPEARMAN and row[f"retained_share_{v}"] >= MIN_RETAINED
             for v in REGISTERED_VARIANTS)      # a missing value compares False
    return "ROBUST" if ok else "NOT ROBUST"


def compare(headline_dir, variants):
    """variants: {label: folder}. Returns the per-block table (hotspots first)."""
    h_summ, h_files, alpha = read_run(headline_dir)
    runs = {}
    for label, d in variants.items():
        v_summ, v_files, v_alpha = read_run(d)
        if v_alpha != alpha:
            raise ValueError(f"alpha differs: headline {alpha}, rerun {label} {v_alpha}")
        runs[label] = (v_summ, v_files)
    rows = []
    for key, s in h_summ.iterrows():
        analysed = key in h_files
        called = bool(s["is_hotspot"]) or s["gate_class"] == "borderline"
        if not analysed and not called:
            continue                                  # an unanalysed block has nothing to compare
        row = {"tissue": s["tissue"], "cell_type": s["cell_type"], "is_hotspot": bool(s["is_hotspot"]),
               "gate_class": s["gate_class"],
               "low_fraction": s["low_fraction"], "mean_fraction": s["mean_fraction"],
               "status_headline": "ok" if analysed else str(s["status"]),
               "n_genes_headline": np.nan, "n_sig_headline": np.nan}
        head = load_block(h_files[key]) if analysed else None
        if analysed:
            row["n_genes_headline"] = len(head)
            row["n_sig_headline"] = int((head["FDR_IHW"].notna() & (head["FDR_IHW"] < alpha)).sum())
        for label, (v_summ, v_files) in runs.items():
            stats = dict.fromkeys(["n_genes_compared", "n_sig", "n_retained", "n_lost", "n_gained",
                                   "retained_share", "spearman_neglog10p"], np.nan)
            if analysed and key in v_files:
                status = "ok"
                stats = compare_block(head, load_block(v_files[key]), alpha)
            elif key in v_summ.index:
                status = str(v_summ.loc[key, "status"])
            else:
                status = "block absent from the rerun"
            row[f"status_{label}"] = status
            if "n_samples_dropped_covariate" in v_summ.columns and key in v_summ.index:
                row[f"n_samples_dropped_{label}"] = v_summ.loc[key, "n_samples_dropped_covariate"]
            for k, v in stats.items():
                row[f"{k}_{label}"] = v
        rows.append(row)
    out = pd.DataFrame(rows)
    # The registered verdict is per hotspot. A borderline block gets the same reading, so that it is carried
    # through this table by name (STEP3 §5.2); summaries count the two classes apart.
    out["verdict"] = [hotspot_verdict(r, set(variants)) if (r["is_hotspot"] or r["gate_class"] == "borderline") else ""
                      for _, r in out.iterrows()]
    out = out.sort_values(["is_hotspot", "n_sig_headline"], ascending=[False, False], kind="stable")
    count_cols = [c for c in out.columns if c.startswith(("n_genes", "n_sig", "n_retained", "n_lost", "n_gained"))]
    out[count_cols] = out[count_cols].astype("Int64")
    return out.reset_index(drop=True), alpha


def summarise(table, variants, alpha):
    """Text summary: totals over blocks per rerun, then the hotspots with their verdicts and gains."""
    L = [f"blocks analysed in the headline: {int((table['status_headline'] == 'ok').sum())}; "
         f"dose-significant gene x block pairs at FDR_IHW < {alpha:g}: {int(table['n_sig_headline'].sum())}"]
    for v in variants:
        ok = table[table[f"status_{v}"] == "ok"]
        L.append(f"rerun {v}: {len(ok)} blocks compared; pairs in the rerun {int(ok[f'n_sig_{v}'].sum())}; "
                 f"retained {int(ok[f'n_retained_{v}'].sum())}, lost {int(ok[f'n_lost_{v}'].sum())}, "
                 f"gained {int(ok[f'n_gained_{v}'].sum())}; median Spearman of -log10 p "
                 f"{ok[f'spearman_neglog10p_{v}'].median():.3f} (min {ok[f'spearman_neglog10p_{v}'].min():.3f})")
        not_ok = table[(table["status_headline"] == "ok") & (table[f"status_{v}"] != "ok")]
        if len(not_ok):
            L.append(f"  not analysed in rerun {v}: " + "; ".join(f"{r.tissue} {r.cell_type}" for r in not_ok.itertuples()))
    hot = table[table["is_hotspot"]]
    L.append(f"hotspots: {len(hot)}; " + ", ".join(f"{k} {n}" for k, n in hot["verdict"].value_counts().items()))
    if not all(v in variants for v in REGISTERED_VARIANTS):
        L.append("  the registered verdict needs reruns A and B; hotspots with calls are NOT ASSESSED")
    border = table[table["gate_class"] == "borderline"]
    if len(border):
        L.append(f"borderline blocks (not hotspots; same reading, counted apart): {len(border)}; "
                 + ", ".join(f"{k} {n}" for k, n in border["verdict"].value_counts().items()))
    for r in pd.concat([hot, border]).itertuples():
        tag = "" if r.is_hotspot else " [borderline]"
        parts = [f"  {r.tissue} {r.cell_type}{tag}: {r.verdict}; headline {r.n_sig_headline}"]
        for v in variants:
            if getattr(r, f"status_{v}") == "ok":
                parts.append(f"{v}: retained {getattr(r, f'n_retained_{v}')} "
                             f"(share {getattr(r, f'retained_share_{v}'):.3f}), lost {getattr(r, f'n_lost_{v}')}, "
                             f"gained {getattr(r, f'n_gained_{v}')}, Spearman {getattr(r, f'spearman_neglog10p_{v}'):.3f}")
            else:
                parts.append(f"{v}: {getattr(r, f'status_{v}')}")
        L.append("; ".join(parts))
    return L


def parse_variants(specs):
    variants = {}
    for spec in specs:
        label, sep, d = spec.partition("=")
        if not sep or not label or not d:
            raise SystemExit(f"--variant takes LABEL=DIR, got {spec!r}")
        if label in variants:
            raise SystemExit(f"--variant {label} given twice")
        if not label.replace("_", "").isalnum():
            raise SystemExit(f"--variant label {label!r} must be letters, digits or underscores")
        variants[label] = d
    return variants


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--headline", required=True, help="the headline pseudobulk_de folder (no covariates)")
    ap.add_argument("--variant", action="append", required=True, metavar="LABEL=DIR",
                    help="a covariate rerun (repeatable); the registered verdict needs A and B")
    ap.add_argument("--out", default=None, help="table to write (default: <headline>/de_covariate_robustness.tsv)")
    a = ap.parse_args(argv)
    variants = parse_variants(a.variant)
    out = Path(a.out) if a.out else Path(a.headline) / "de_covariate_robustness.tsv"
    head_cov = read_methods(a.headline).get("technical_covariates", "none")
    if not head_cov.startswith("none"):
        raise SystemExit(f"the headline folder was itself run with technical covariates ({head_cov})")
    table, alpha = compare(a.headline, variants)
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, sep="\t", index=False, na_rep="NA")
    lines = [f"headline: {a.headline}"]
    for label, d in variants.items():
        lines.append(f"rerun {label}: {d} (technical covariates: {read_methods(d).get('technical_covariates', 'not recorded')})")
    lines += summarise(table, variants, alpha)
    print("\n".join(lines))
    print(f"wrote {out} ({len(table)} blocks)")
    write_provenance(out.parent, name=f"provenance_{out.stem}.json",
                     inputs=[Path(d) / f for d in [a.headline, *variants.values()]
                             for f in ("de_summary.tsv", "de_methods.tsv", "celltype_files.tsv")],
                     params={"headline": a.headline, "variants": variants, "alpha": alpha,
                             "min_spearman": MIN_SPEARMAN, "min_retained": MIN_RETAINED, "out": str(out)})


if __name__ == "__main__":
    main()
