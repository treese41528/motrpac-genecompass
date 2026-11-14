#!/usr/bin/env python
"""compare_refs5_downstream.py -- refs5 against the current default after Stage 8 (STEP3_PREREG.md section 10, items
4-7): the gate, the differential expression, the transfer and the networks. Read-only; each part is written only if
its inputs exist, so the script can be rerun as stages finish.

Usage: compare_refs5_downstream.py <default_gc_root> <refs5_gc_root> <out_dir>
  gc_root is a genecompass_input folder (its gate_rcv10/, pseudobulk_de/ and the sibling <gc_root>_human/).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

KEY = ["tissue", "cell_type"]


def flag(s):
    return s.astype(str).str.strip().str.upper().isin(["TRUE", "T", "1"])


def read_gate(gc):
    f = Path(gc) / "gate_rcv10" / "corroboration_merged.tsv"
    if not f.exists():
        return None
    g = pd.read_csv(f, sep="\t"); g["tissue"] = g["tissue"].str.upper()
    g["analysable"] = flag(g["analysable"]) if "analysable" in g else True
    if "gate_class" not in g:                                   # a table from before the classes
        q = pd.to_numeric(g["q_sup_trained"], errors="coerce")
        g["gate_class"] = np.where(q < 0.05, "hotspot", "none")
    return g


def gate(old_gc, new_gc, out):
    o, n = read_gate(old_gc), read_gate(new_gc)
    if o is None or n is None:
        return
    oc = ["sup_trained_auc", "p_sup_trained", "q_sup_trained", "analysable", "gate_class", "median_libsize"]
    m = n.merge(o[KEY + [c for c in oc if c in o]], on=KEY, how="outer", suffixes=("", "_default"), indicator="in")
    m["in"] = m["in"].map({"both": "both", "left_only": "refs5 only", "right_only": "default only"})
    m.to_csv(out / "gate_blocks_refs5_vs_default.tsv", sep="\t", index=False)
    fam = n[n["analysable"]]
    rows = [("blocks", len(o), len(n)), ("analysable blocks", int(o["analysable"].sum()), len(fam)),
            ("hotspots", int(((o["gate_class"] == "hotspot") & o["analysable"]).sum()), int((fam["gate_class"] == "hotspot").sum())),
            ("borderline", "", int((fam["gate_class"] == "borderline").sum()))]
    if "q_sup_trained_10k" in n:
        q10 = pd.to_numeric(fam["q_sup_trained_10k"], errors="coerce") < 0.05
        rows.append(("refs5 blocks with q < 0.05 at the first 10,000 permutations (the earlier method)", "", int(q10.sum())))
        rows.append(("  of those: hotspot / borderline / none after sequential sampling", "",
                     " / ".join(str(int(((fam["gate_class"] == c) & q10).sum())) for c in ("hotspot", "borderline", "none"))))
        rows.append(("permutations used: median / maximum per analysable block", "",
                     f"{int(fam['m_sup_trained'].median()):,} / {int(fam['m_sup_trained'].max()):,}"))
    called = fam[fam["gate_class"].isin(["hotspot", "borderline"])]
    for c, lab in (("q_sup_trained_s1", "S1 sex-stratified: called blocks with q < 0.05"),
                   ("q_sup_trained_s2", "S2 8-week against control: called blocks with q < 0.05"),
                   ("q_sup_trained_s4", "S4 fraction tier: called blocks with q < 0.05 among blocks at mean fraction >= 0.01")):
        if c in called:
            rows.append((lab, "", f"{int((pd.to_numeric(called[c], errors='coerce') < 0.05).sum())} of {len(called)}"))
    if "low_fraction" in called:
        rows.append(("called blocks below mean fraction 0.01", "", f"{int(flag(called['low_fraction']).sum())} of {len(called)}"))
    if "support" in called:
        for k, v in called.groupby("gate_class")["support"].value_counts().items():
            rows.append((f"reading rule, {k[0]}: {k[1]}", "", int(v)))
    pd.DataFrame(rows, columns=["quantity", "default", "refs5"]).to_csv(out / "gate_summary.tsv", sep="\t", index=False)
    cols = [c for c in ["tissue", "cell_type", "gate_class", "gate_class_default", "sup_trained_auc", "sup_trained_auc_default",
                        "p_sup_trained_10k", "q_sup_trained_10k", "p_sup_trained", "m_sup_trained", "p_lo_sup_trained",
                        "p_hi_sup_trained", "q_sup_trained", "sup_trained_auc_s1", "q_sup_trained_s1", "n_s2", "sup_trained_auc_s2",
                        "q_sup_trained_s2", "auc_embed", "auc_pca", "auc_genes", "mean_fraction", "low_fraction", "q_sup_trained_s4",
                        "support", "median_libsize", "in"] if c in m]
    h = m[m["gate_class"].isin(["hotspot", "borderline"]) | (m["gate_class_default"] == "hotspot")][cols]
    h = h.sort_values(["gate_class", "p_sup_trained"], na_position="last")
    h.to_csv(out / "gate_called_blocks.tsv", sep="\t", index=False)
    print("== gate =="); print(pd.DataFrame(rows, columns=["quantity", "default", "refs5"]).to_string(index=False))
    print(h.to_string(index=False))


def de_pairs(de_dir):
    """Dose-significant gene x block pairs and distinct genes per tissue, from the per-block tables."""
    de_dir = Path(de_dir)
    man = pd.read_csv(de_dir / "celltype_files.tsv", sep="\t")
    summ = pd.read_csv(de_dir / "de_summary.tsv", sep="\t")
    mf = dict(zip(zip(summ.tissue, summ.cell_type), pd.to_numeric(summ.mean_fraction, errors="coerce")))
    rows, genes_all, genes_hi = [], {}, {}
    for r in man.itertuples():
        t = pd.read_csv(de_dir / r.file, sep="\t", usecols=["gene", "FDR_IHW"])
        sig = set(t.loc[t.FDR_IHW < 0.05, "gene"])
        low = mf.get((r.tissue, r.cell_type), np.nan) < 0.01
        rows.append({"tissue": r.tissue, "cell_type": r.cell_type, "n_sig": len(sig), "low_fraction": bool(low)})
        genes_all.setdefault(r.tissue, set()).update(sig)
        if not low:
            genes_hi.setdefault(r.tissue, set()).update(sig)
    b = pd.DataFrame(rows)
    t = b.groupby("tissue").agg(blocks=("n_sig", "size"), pairs=("n_sig", "sum")).reset_index()
    t["pairs_not_low_fraction"] = t.tissue.map(b[~b.low_fraction].groupby("tissue").n_sig.sum()).fillna(0).astype(int)
    t["genes_distinct"] = t.tissue.map({k: len(v) for k, v in genes_all.items()})
    tot = {"tissue": "ALL", "blocks": len(b), "pairs": int(b.n_sig.sum()), "pairs_not_low_fraction": int(b[~b.low_fraction].n_sig.sum()),
           "genes_distinct": len(set().union(*genes_all.values())) if genes_all else 0}
    return b, pd.concat([t, pd.DataFrame([tot])], ignore_index=True)


def de(old_gc, new_gc, out):
    od, nd = Path(old_gc) / "pseudobulk_de", Path(new_gc) / "pseudobulk_de"
    if not (nd / "de_summary.tsv").exists() or not (od / "de_summary.tsv").exists():
        return
    ob, ot = de_pairs(od); nb, nt = de_pairs(nd)
    t = nt.merge(ot, on="tissue", how="outer", suffixes=("_refs5", "_default"))
    t.to_csv(out / "de_by_tissue_refs5_vs_default.tsv", sep="\t", index=False)
    b = nb.merge(ob, on=KEY, how="outer", suffixes=("_refs5", "_default"))
    ns = pd.read_csv(nd / "de_summary.tsv", sep="\t")
    keep = [c for c in ["tissue", "cell_type", "gate_class", "status", "n_male", "n_female", "mean_fraction", "median_libsize",
                        "n_dose_tests", "n_sexcons_concordance_8w", "n_sexcons_repfdr_8w", "n_sexcons_union_8w"] if c in ns]
    b = b.merge(ns[keep], on=KEY, how="outer")
    b.to_csv(out / "de_blocks_refs5_vs_default.tsv", sep="\t", index=False)
    for f in ("de_headline.tsv", "de_covariate_robustness.tsv", "composition_confound_table.tsv", "de_methods.tsv"):
        if (nd / f).exists():
            (out / ("refs5_" + f)).write_bytes((nd / f).read_bytes())
    print("== DE =="); print(t.to_string(index=False))
    sx = ns[[c for c in ns if c.startswith("n_sexcons")]].sum(numeric_only=True)
    print("sex-consistent 8-week calls (refs5):", sx.astype(int).to_dict())
    one = ns[(pd.to_numeric(ns.n_male, errors="coerce") == 0) | (pd.to_numeric(ns.n_female, errors="coerce") == 0)]
    print("single-sex tissues (refs5):", nb[nb.tissue.isin(one.tissue.unique())].groupby("tissue").n_sig.sum().to_dict())
    if (nd / "de_covariate_robustness.tsv").exists():
        r = pd.read_csv(nd / "de_covariate_robustness.tsv", sep="\t")
        called = r[r.get("gate_class", pd.Series("", index=r.index)).isin(["hotspot", "borderline"]) | flag(r["is_hotspot"])]
        print(called[[c for c in ["tissue", "cell_type", "gate_class", "n_sig_headline", "retained_share_A", "spearman_neglog10p_A",
                                  "retained_share_B", "spearman_neglog10p_B", "verdict"] if c in called]].to_string(index=False))


def transfer(old_gc, new_gc, out):
    f = Path(str(new_gc) + "_human") / "gate_rcv10" / "transfer_comparison.tsv"
    cand = [f, Path(str(new_gc) + "_human") / "transfer_comparison.tsv"]
    f = next((c for c in cand if c.exists()), None)
    if f is None:
        return
    t = pd.read_csv(f, sep="\t")
    (out / "refs5_transfer_comparison.tsv").write_bytes(f.read_bytes())
    cls = t["gate_class_rat"] if "gate_class_rat" in t else np.where(flag(t["is_hotspot"]), "hotspot", "none")
    h = t[pd.Series(cls, index=t.index).isin(["hotspot", "borderline"])].copy(); h["gate_class_rat"] = pd.Series(cls, index=t.index)[h.index]
    rows = []
    for c in ("hotspot", "borderline"):
        d = h[h.gate_class_rat == c]
        rows.append({"rat_class": c, "blocks": len(d), "registered_PRESERVED": int((d.status == "PRESERVED").sum()),
                     "registered_WEAKENED": int((d.status == "WEAKENED").sum()), "registered_LOST": int((d.status == "LOST").sum()),
                     "corrected_transfers": int(d.transfer_corrected.astype(str).str.upper().isin(["PASS", "TRUE"]).sum()),
                     "dose_trend_lost": int(flag(d.dose_trend_lost).sum())})
    pd.DataFrame(rows).to_csv(out / "transfer_summary.tsv", sep="\t", index=False)
    print("== transfer =="); print(pd.DataFrame(rows).to_string(index=False))
    print(h[[c for c in ["tissue", "cell_type", "gate_class_rat", "sup_trained_auc_rat", "sup_trained_auc_human", "status",
                         "q_sup_trained_human", "gate_class_human", "transfer_corrected", "sup_dose_rho_rat", "sup_dose_rho_human",
                         "dose_trend_lost"] if c in h]].to_string(index=False))


if __name__ == "__main__":
    old_gc, new_gc, out = sys.argv[1:4]
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    gate(old_gc, new_gc, out); de(old_gc, new_gc, out); transfer(old_gc, new_gc, out)
