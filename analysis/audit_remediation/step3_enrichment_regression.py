#!/usr/bin/env python
"""step3_enrichment_regression.py -- what the registered enrichment changes (STEP3_PREREG.md 8.2) do to the
numbers on the CURRENT default DE tables. A code regression, not the refs5 rerun. Read-only on the default tree.

Three enrichment runs are compared, block by block:
  old           the default tree's enrichment (upper-cased rat symbol, P_dose_comb), as it is on disk;
  mapping_only  the changed script with GSEA_P_COLUMN=P_dose_comb: only the rat-to-human mapping differs
                from old (one-to-one ortholog pairs, matched on the human Ensembl gene ID);
  registered    the changed script as registered: the new mapping AND sign(slope) x -log10(P_fisher).

The old mapping is recomputed here from the DE tables by the old rule (finite slope_week and P_dose_comb, a
rat symbol, upper-cased, one statistic per symbol), so that "genes mapped" can be put side by side. Where the
old job log is given, the recomputed count is checked against the "ranks=" the old run printed.

  python analysis/audit_remediation/step3_enrichment_regression.py \
      --de-dir <default>/pseudobulk_de --registered DIR --mapping-only DIR --out DIR [--old-log LOG]
"""
import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# The repository root: PIPELINE_ROOT when set (a job may run a snapshot copy of this file from elsewhere).
ROOT = Path(os.environ.get("PIPELINE_ROOT") or Path(__file__).resolve().parents[2])
sys.path.insert(0, str(ROOT / "deconvolution"))
from celltype_names import safe            # noqa: E402  (the one cell-type -> file-name map)


def read_sig(d):
    s = pd.read_csv(Path(d) / "enrichment_summary.tsv", sep="\t")
    return s, s.groupby(["tissue", "cell_type"]).pathway.apply(set).to_dict()


def tests_table(d, blocks):
    """Per block: number of gene-set tests and of tests without a p-value, from the gsea__ tables."""
    rows = {}
    for t, ct in blocks:
        f = Path(d) / f"gsea__{t}__{safe(ct)}.tsv"
        if not f.exists():
            rows[(t, ct)] = (0, 0, None)
            continue
        g = pd.read_csv(f, sep="\t", usecols=["pathway", "pval", "NES"])
        rows[(t, ct)] = (len(g), int(g.pval.isna().sum()), g.set_index("pathway").NES)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--de-dir", required=True)
    ap.add_argument("--old", default=None, help="default: <de-dir>/enrichment")
    ap.add_argument("--registered", required=True)
    ap.add_argument("--mapping-only", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--old-log", default=None, help="job log of the old run, to check the recomputed old ranks")
    ap.add_argument("--token-mapping", default=str(ROOT / "data/training/ortholog_mappings/rat_token_mapping.tsv"))
    a = ap.parse_args()
    de, old = Path(a.de_dir), Path(a.old or Path(a.de_dir) / "enrichment")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    tm = pd.read_csv(a.token_mapping, sep="\t", dtype=str, keep_default_na=False)
    rat_sym = {g: s.upper() for g, s in zip(tm.rat_gene, tm.rat_symbol) if s}
    one2one = dict(zip(tm.rat_gene[tm.orth_type_human == "ortholog_one2one"],
                       tm.human_ortholog[tm.orth_type_human == "ortholog_one2one"]))
    gs = pd.read_csv(Path(a.registered) / "gene_sets_used.tsv", sep="\t", dtype=str, keep_default_na=False)
    set_syms, set_ens = set(gs.gene_symbol.str.upper()), set(gs.ensembl_gene)

    summ = pd.read_csv(de / "de_summary.tsv", sep="\t")
    blocks = [(t, c) for t, c in zip(summ.tissue[summ.status == "ok"], summ.cell_type[summ.status == "ok"])]
    bn = pd.read_csv(Path(a.registered) / "enrichment_blocks.tsv", sep="\t").set_index(["tissue", "cell_type"])
    bm = pd.read_csv(Path(a.mapping_only) / "enrichment_blocks.tsv", sep="\t").set_index(["tissue", "cell_type"])

    logged = {}
    if a.old_log and Path(a.old_log).exists():
        for line in open(a.old_log, errors="replace"):
            m = re.match(r"^\s+(\S+)\s+(.*?)\s+ranks=(\d+)\s+sig\(padj<\.05\)=(\d+)\s*$", line)
            if m:
                logged[(m.group(1), m.group(2).strip())] = int(m.group(3))

    # ---- genes mapped: old rule against the one-to-one ortholog pairs (both on P_dose_comb) ----
    rows = []
    for t, ct in blocks:
        d = pd.read_csv(de / t / f"de__{safe(ct)}.tsv", sep="\t", usecols=["gene", "slope_week", "P_dose_comb"])
        d = d[np.isfinite(d.slope_week) & np.isfinite(d.P_dose_comb)]
        sym = d.gene.map(rat_sym)
        old_ranked = sym.dropna().nunique()
        old_in = set(d.gene[sym.isin(set_syms)])                     # rat genes whose upper-cased symbol is in a set
        new_in = set(d.gene[d.gene.map(one2one).isin(set_ens)])      # rat genes whose ortholog is in a set
        rows.append({"tissue": t, "cell_type": ct, "n_genes_with_stat": len(d),
                     "old_n_ranked": old_ranked, "old_n_ranked_logged": logged.get((t, ct), np.nan),
                     "new_n_ranked": int(bm.loc[(t, ct), "n_ranked"]),
                     "new_n_no_pair": int(bm.loc[(t, ct), "n_no_pair"]),
                     "new_n_no_pair_not_one2one": int(bm.loc[(t, ct), "n_no_pair_not_one2one"]),
                     "old_n_in_sets": len(old_in), "new_n_in_sets": len(new_in),
                     "in_sets_both": len(old_in & new_in), "in_sets_old_only": len(old_in - new_in),
                     "in_sets_new_only": len(new_in - old_in),
                     "registered_n_missing_stat": bn.loc[(t, ct), "n_missing_stat"],
                     "registered_n_ranked": bn.loc[(t, ct), "n_ranked"],
                     "registered_status": bn.loc[(t, ct), "status"]})
    M = pd.DataFrame(rows)
    M.to_csv(out / "mapping_old_vs_new_per_block.tsv", sep="\t", index=False)

    # ---- significant sets ----
    so, po = read_sig(old); sm, pm = read_sig(a.mapping_only); sr, pr = read_sig(a.registered)
    to, tmo, tr = tests_table(old, blocks), tests_table(a.mapping_only, blocks), tests_table(a.registered, blocks)
    rows = []
    for b in blocks:
        o, m, r = po.get(b, set()), pm.get(b, set()), pr.get(b, set())
        nes_o, nes_m, nes_r = to[b][2], tmo[b][2], tr[b][2]

        def rho(x, y):
            if x is None or y is None:
                return np.nan
            j = pd.concat([x, y], axis=1, join="inner").dropna()
            return round(float(j.iloc[:, 0].corr(j.iloc[:, 1], method="spearman")), 4) if len(j) > 2 else np.nan
        rows.append({"tissue": b[0], "cell_type": b[1], "old_n_sig": len(o), "mapping_only_n_sig": len(m),
                     "registered_n_sig": len(r), "old_and_mapping_only": len(o & m), "old_and_registered": len(o & r),
                     "old_n_tests": to[b][0], "mapping_only_n_tests": tmo[b][0], "registered_n_tests": tr[b][0],
                     "old_n_no_p": to[b][1], "mapping_only_n_no_p": tmo[b][1], "registered_n_no_p": tr[b][1],
                     "nes_spearman_old_vs_mapping_only": rho(nes_o, nes_m),
                     "nes_spearman_old_vs_registered": rho(nes_o, nes_r)})
    S = pd.DataFrame(rows)
    S.to_csv(out / "significant_sets_old_vs_new_per_block.tsv", sep="\t", index=False)

    # ---- summary ----
    meta = pd.read_csv(Path(a.registered) / "gene_sets_used.meta.tsv", sep="\t", dtype=str).set_index("field").value
    show = ["tissue", "cell_type", "n_genes_with_stat", "old_n_ranked", "new_n_ranked", "new_n_no_pair",
            "old_n_in_sets", "new_n_in_sets", "in_sets_old_only", "in_sets_new_only"]
    pick = M[M.tissue.isin(["BLOOD", "SKMGN", "LUNG", "LIVER", "TESTES"])].groupby("tissue").head(1)
    chk = M.dropna(subset=["old_n_ranked_logged"])
    single = S[S.tissue.isin(["TESTES", "OVARY"])]
    two = S[~S.tissue.isin(["TESTES", "OVARY"])]
    L = [f"Enrichment regression on {de}",
         f"blocks (de_summary status ok): {len(blocks)}",
         "",
         "== gene sets ==",
         f"file: {Path(a.registered) / 'gene_sets_used.tsv'}",
         "\n".join(f"  {k}: {v}" for k, v in meta.items()),
         "",
         "== genes mapped, old rule (upper-cased rat symbol) against the one-to-one ortholog pairs; both on P_dose_comb ==",
         pick[show].to_string(index=False),
         "",
         "totals over the blocks (a gene counts once per block):",
         f"  genes with a statistic              {int(M.n_genes_with_stat.sum()):>10,}",
         f"  ranked, old rule                    {int(M.old_n_ranked.sum()):>10,}",
         f"  ranked, one-to-one pairs            {int(M.new_n_ranked.sum()):>10,}",
         f"  dropped, no one-to-one pair         {int(M.new_n_no_pair.sum()):>10,}  "
         f"(of which {int(M.new_n_no_pair_not_one2one.sum()):,} have a human ortholog that is not one-to-one)",
         f"  in at least one gene set, old rule  {int(M.old_n_in_sets.sum()):>10,}",
         f"  in at least one gene set, new rule  {int(M.new_n_in_sets.sum()):>10,}",
         f"    in both {int(M.in_sets_both.sum()):,}; old rule only {int(M.in_sets_old_only.sum()):,}; "
         f"new rule only {int(M.in_sets_new_only.sum()):,}",
         (f"  check: the recomputed old ranks equal the count the old run logged in "
          f"{int((chk.old_n_ranked == chk.old_n_ranked_logged).sum())} of {len(chk)} blocks"
          if len(chk) else "  check against the old job log: no log given"),
         "",
         "== significant gene sets (padj < 0.05 within block) ==",
         f"                         hits   blocks with a hit   tests   tests without p",
         f"  old                  {len(so):>7,} {int((S.old_n_sig > 0).sum()):>12} {int(S.old_n_tests.sum()):>14,} {int(S.old_n_no_p.sum()):>10,}",
         f"  mapping only         {len(sm):>7,} {int((S.mapping_only_n_sig > 0).sum()):>12} {int(S.mapping_only_n_tests.sum()):>14,} {int(S.mapping_only_n_no_p.sum()):>10,}",
         f"  registered           {len(sr):>7,} {int((S.registered_n_sig > 0).sum()):>12} {int(S.registered_n_tests.sum()):>14,} {int(S.registered_n_no_p.sum()):>10,}",
         f"  hits shared: old and mapping only {int(S.old_and_mapping_only.sum()):,}; old and registered {int(S.old_and_registered.sum()):,}",
         f"  two-sex tissues only: old {int(two.old_n_sig.sum()):,}, mapping only {int(two.mapping_only_n_sig.sum()):,}, "
         f"registered {int(two.registered_n_sig.sum()):,}",
         f"  single-sex tissues (TESTES, OVARY; {len(single)} blocks): old {int(single.old_n_sig.sum()):,}, mapping only "
         f"{int(single.mapping_only_n_sig.sum()):,}, registered {int(single.registered_n_sig.sum()):,}",
         f"  median over blocks of the Spearman correlation of NES: old vs mapping only "
         f"{S.nes_spearman_old_vs_mapping_only.median():.4f}; old vs registered {S.nes_spearman_old_vs_registered.median():.4f}",
         "",
         "== blocks the registered run did not test ==",
         (bn[bn.status != "ok"].reset_index()[["tissue", "cell_type", "n_genes", "n_missing_stat", "status"]]
          .to_string(index=False) if (bn.status != "ok").any() else "  none"),
         ""]
    txt = "\n".join(L)
    print(txt)
    (out / "regression_summary.txt").write_text(txt + "\n")


if __name__ == "__main__":
    main()
