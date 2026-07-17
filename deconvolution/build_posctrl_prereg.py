#!/usr/bin/env python3
"""Build the FROZEN positive-control pre-registration table for the exhaustive per-cell-type DE.

This is committed BEFORE looking at any per-gene DE result (see feedback: pre-register
positive-control checks). It freezes, per committed gene: tier, expected-direction (where
anchored), target tissue/cell-type/sex, and the source. The comparison script (compare_posctrl.py)
executes this spec; it does NOT add genes post hoc.

Tiers (by how cleanly the EXPRESSION direction is anchored):
  A  -- direction-anchored from the MoTrPAC PASS1B main papers (Nature 2024 #21 / Cell Genomics #22):
        expected expression direction WITH training (8w, sex-consistent unless noted). Scored on direction.
  Ai -- MoTrPAC identity-only (named in the papers but transcript direction not stated): scored on
        present+training-DE in the target; direction reported, not pass/fail.
  B  -- Vetr 2024 named genes (identity + tissue + sex; the paper's 'direction' is trait-level, not a
        clean expression sign): scored on present+training-DE in the matching tissue (+sex); direction reported.
  C  -- Yu 2023 named immune programs: cell-type/program RESPONSIVENESS only; direction NOT anchored
        (acute/human/n=3, up-only DE, protocol-dependent sign). Scored on program enrichment, never direction.

Output: deconvolution/reference/posctrl_prereg.tsv
"""
import csv, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GU   = os.path.join(ROOT, "data/training/gene_universe/pruned_gene_universe.tsv")
OUT  = os.path.join(ROOT, "deconvolution/reference/posctrl_prereg.tsv")

# symbol -> rel-113 ENSRNOG (case-insensitive), from the gene universe (same ID space as DE rownames)
sym2ens = {}
with open(GU) as fh:
    r = csv.reader(fh, delimiter="\t"); next(r)
    for row in r:
        if len(row) >= 2 and row[1] and row[1].lower() != "nan":
            sym2ens.setdefault(row[1].lower(), row[0])

# Each entry: (tier, group, [symbols], tissue, cell_type_target, sex, expected_dir, source, notes)
#   cell_type_target: "ANY" = any cell type in the tissue; else pipe-joined explicit cell types.
#   sex: any|male|female.  expected_dir: up|down|none (none = direction not anchored / reported only).
SPEC = [
 ("A","muscle_mito_biogenesis", ["Sod2","Slc2a4","Mef2c","Mef2a","Mef2d","Opa1","Mfn1","Prkab1","Tbc1d1","Plin2","Plin4","Plin5","Hspa1b","Hsp90aa1"],
   "SKMVL", "Skeletal myocytes", "any", "up", "MoTrPAC #21 Fig4 (8w sex-consistent mito/biogenesis/heat-shock UP)", "muscle parenchyma (label remapped split->merged when the SKMVL muscle merge was adopted; genes/direction/thresholds unchanged)"),
 ("A","muscle_mito_biogenesis", ["Sod2","Slc2a4","Mef2c","Mef2a","Mef2d","Opa1","Mfn1","Prkab1","Tbc1d1","Plin2","Plin4","Plin5","Hspa1b","Hsp90aa1"],
   "SKMGN", "Skeletal myocytes", "any", "up", "MoTrPAC #21 Fig4 + #22 (SKM-GN up-ERGs = mito energy)", "muscle parenchyma"),
 ("A","heart_mito_biogenesis", ["Sod2","Mef2c","Opa1","Mfn1","Hspa1b","Hsp90aa1"],
   "HEART", "Cardiomyocytes", "any", "up", "MoTrPAC #21 Fig4 (heart 8w sex-consistent mito UP) + #22 heart up-ERGs", "striated-muscle convergence"),
 ("A","liver_metabolic", ["Stat3","Pxn","Hsp90aa1"],
   "LIVER", "Hepatocytes", "any", "up", "MoTrPAC #21 p7-8 (liver metabolic/regeneration program UP, later in females)", "8w; female-delayed onset"),
 ("A","blood_hematopoietic_TF", ["Gabpa","Ets1","Klf3","Znf143"],
   "BLOOD", "ANY", "any", "up", "MoTrPAC #21 p3,p5 (blood = haematopoietic cellular mobilization)", "TF-program; broad across blood cells"),
 ("A","wat_adrenergic", ["Adra1b","Adra1d","Adra2b","Adrb1"],
   "WATSC", "ANY", "any", "down", "MoTrPAC #22 p12 (WAT adrenergic GPCR genes DOWN)", "WAT deconv has no adipocyte cell type -> coverage caveat"),
 # Ai -- MoTrPAC named, transcript direction not stated -> identity only
 ("Ai","heart_structural", ["Gja1","Cdh2"], "HEART", "Cardiomyocytes", "any", "none", "MoTrPAC #21 p3-4 (SRC/ECM remodelling; phospho-site, transcript dir unstated)", ""),
 ("Ai","kidney_signalling", ["Akt1","Mtor","Hnf4a","Hnf1b"], "KIDNEY", "Proximal tubule cells|Distal tubule cells", "any", "none", "MoTrPAC #21 p3 (AKT/mTOR) + #22 (Hnf4a/Hnf1b TF program)", ""),
 ("Ai","liver_tf", ["Hnf4a","Hnf1b"], "LIVER", "Hepatocytes", "any", "none", "MoTrPAC #22 (Hnf4a/Hnf1b)", ""),
 ("Ai","lung_prkaca_substrates", ["Dsp","Mylk","Stmn1","Syne1","Prkaca"], "LUNG", "ANY", "male", "none", "MoTrPAC #21 p7 (male-decreased PRKACA phosphosignalling; transcript dir unstated)", "phospho not transcript"),
 ("Ai","hippoc_neuro", ["Gpr37l1"], "HIPPOC", "ANY", "any", "none", "MoTrPAC #22 (hippocampus axonogenesis/neurogenesis up-ERGs)", "fewest ERGs of 8 tissues"),
 # B -- Vetr 2024 named genes (identity + tissue + sex); direction trait-level, reported not scored
 ("B","vetr_disease_gene", ["Ldlr"], "CORTEX", "ANY", "any", "none", "Vetr 2024 p3 (LDLR DE in cortex,hippoc,SKM-GN,SKM-VL)", ""),
 ("B","vetr_disease_gene", ["Ldlr"], "HIPPOC", "ANY", "any", "none", "Vetr 2024 p3", ""),
 ("B","vetr_disease_gene", ["Ldlr"], "SKMGN", "ANY", "any", "none", "Vetr 2024 p3", ""),
 ("B","vetr_disease_gene", ["Ldlr"], "SKMVL", "ANY", "any", "none", "Vetr 2024 p3", ""),
 ("B","vetr_disease_gene", ["Apob"], "LUNG", "ANY", "male", "none", "Vetr 2024 p3-4 (APOB male lung +9.7 SDpheno)", "strong-effect exception"),
 ("B","vetr_disease_gene", ["Slc6a8"], "HEART", "ANY", "any", "none", "Vetr 2024 p3 (SLC6A8 DE in heart,liver,lung)", ""),
 ("B","vetr_disease_gene", ["Slc6a8"], "LIVER", "ANY", "any", "none", "Vetr 2024 p3", ""),
 ("B","vetr_disease_gene", ["Slc6a8"], "LUNG", "ANY", "any", "none", "Vetr 2024 p3", ""),
 ("B","vetr_disease_gene", ["Foxp3"], "HEART", "ANY", "any", "none", "Vetr 2024 p3 (FOXP3 DE in heart,spleen)", "KNOWN NOT TESTABLE: Foxp3 absent from every pred_z (idmap) -> pre-declared coverage miss"),
 ("B","vetr_skmvl_male", ["Tmbim1"], "SKMVL", "ANY", "male", "none", "Vetr 2024 p4 (male training VL gene; cholesterol)", ""),
 ("B","vetr_skmvl_female", ["Atp6v1g2"], "SKMVL", "ANY", "female", "none", "Vetr 2024 p5 (female training VL gene; asthma, largest effect)", ""),
 ("B","vetr_blood_chol_male", ["Fads2","Pnkd","Aamp","Ogdh","Ndufa13"], "BLOOD", "ANY", "male", "none", "Vetr 2024 p4-5,p7 (male blood cholesterol-lowering program)", "Ndufa13/AAMP were novel (no prior lit)"),
 ("B","vetr_blood_asthma_male", ["Bag6","Ccnf","Crat","Fam89b"], "BLOOD", "ANY", "male", "none", "Vetr 2024 p5 (male blood asthma program)", "PTPA & 'CCNG' uncertain symbols -> omitted to avoid mis-map"),
 ("B","vetr_wat_female", ["Endou"], "WATSC", "ANY", "female", "none", "Vetr 2024 p5 (female WAT asthma gene)", ""),
 ("B","vetr_liver_female", ["Abcg8"], "LIVER", "ANY", "female", "none", "Vetr 2024 p4 (female liver cholesterol gene)", ""),
 # C -- Yu 2023 immune programs: responsiveness only, NO direction
 ("C","yu_monocyte_activation", ["S100a8","S100a9","S100a12","Vcan","Retn","Acsl1","Nampt"], "BLOOD", "Classical monocytes|Non-classical monocytes", "any", "none", "Yu 2023 Fig4 (S100A8/A9-hi activated monocyte state)", "direction NOT anchored (acute/human)"),
 ("C","yu_cytotoxicity", ["Prf1","Gnly","Nkg7","Gzma","Gzmb","Gzmk","Klrb1","Klrd1","Klrk1","Ctsw","Cst7"], "BLOOD", "Natural killer cells|Memory T cells|ISG-expressing T cells", "any", "none", "Yu 2023 Fig2C (cytotoxicity score genes)", "Gnly/Gzmb have no rat-universe ortholog -> coverage"),
 ("C","yu_naive", ["Ccr7","Tcf7","Lef1","Sell"], "BLOOD", "Naive B cells|Naive CD4+ T cells|Memory T cells", "any", "none", "Yu 2023 Fig2D (naive/resting score)", ""),
 ("C","yu_isg", ["Isg20"], "BLOOD", "ISG-expressing T cells", "any", "none", "Yu 2023 Fig3A (ISG/antiviral up-DEG)", ""),
 ("C","yu_chemokine", ["Cxcr4"], "BLOOD", "ANY", "any", "none", "Yu 2023 Fig3A (CXCR4 migration receptor, up across effector NKT)", ""),
]

rows = []
for tier, group, syms, tissue, ct, sex, edir, src, notes in SPEC:
    for s in syms:
        ens = sym2ens.get(s.lower(), "NA")
        testable_prior = "TRUE"
        n = notes
        if ens == "NA":
            testable_prior = "FALSE"; n = (n + "; " if n else "") + "absent from gene universe (coverage)"
        if s.lower() == "foxp3":
            testable_prior = "FALSE"
        rows.append(dict(tier=tier, group=group, symbol=s, ensembl=ens, tissue=tissue,
                         cell_type_target=ct, sex=sex, expected_dir=edir,
                         anchored_direction=("TRUE" if edir in ("up", "down") else "FALSE"),
                         testable_prior=testable_prior, source=src, notes=n))

os.makedirs(os.path.dirname(OUT), exist_ok=True)
cols = ["tier","group","symbol","ensembl","tissue","cell_type_target","sex",
        "expected_dir","anchored_direction","testable_prior","source","notes"]
with open(OUT, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=cols, delimiter="\t"); w.writeheader()
    for r in rows: w.writerow(r)

n_tot = len(rows); n_map = sum(1 for r in rows if r["ensembl"] != "NA")
n_anch = sum(1 for r in rows if r["anchored_direction"] == "TRUE")
print(f"wrote {OUT}: {n_tot} committed gene-target rows ({n_map} mapped to ENSRNOG, "
      f"{n_tot-n_map} not-in-universe), {n_anch} direction-anchored (Tier A up/down).")
for tier in ["A","Ai","B","C"]:
    print(f"  tier {tier}: {sum(1 for r in rows if r['tier']==tier)} rows")
