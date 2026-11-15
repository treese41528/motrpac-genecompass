# What an expert should review, and in what order

The audit and its remediation were done by an AI coding assistant. No domain expert has read the reference
labels, the reference cells or the results (audit finding 4.2). This plan lists what a person should read,
ordered by how much rides on each item. It is the PI's decision 5.

- **How to use it.** Each item names what to open, the question to answer, what a "no" would change, and
  roughly how long it takes. Work down the list; the first three items cover every called block.
- **Who.** "Tissue" means someone who knows that tissue's cell types and their markers. "Statistics" means a
  statistician. "Pipeline" means someone who can run a job.
- **What review cannot settle** is at the end: things that need new data or a retrain, not a reader.

## The order

| # | Item | Who | Time | Why it is here |
|---|---|---|---|---|
| 1 | The muscle reference and its five called blocks | Tissue (muscle) | half a day | 5 of 12 called blocks; two libraries; none validated at its abundance |
| 2 | The BLOOD composite reference and its four hotspots | Tissue (blood) | half a day | 4 hotspots; the erythroid profile is 90 cells and fails every purity |
| 3 | The LUNG and HEART references and their three hotspots | Tissue (lung, heart) | half a day | Both LUNG hotspots fail two readings; LUNG cell quality cannot be measured |
| 4 | The DE model against technical covariates | Statistics | 1 day | No block's DE calls are robust by the registered verdict |
| 5 | The gate's construction and what "hotspot" may claim | Statistics | half a day | Membership is unstable; one sensitivity is less conservative than the primary |
| 6 | HYPOTH: eight label decisions and one sample | Tissue (hypothalamus) | 2 hours | Re-annotated by the assistant; no GABAergic type could be made |
| 7 | SMLINT and KIDNEY reference cell quality | Tissue (gut, kidney) | 2 hours each | 51% and 17% of reference cells above 15% mitochondrial |
| 8 | The other 160 label decisions | Tissue, per reference | 1 to 3 hours each | Made by the assistant; carry no called block now |
| 9 | Gene identity: IDs that changed meaning | Pipeline, then tissue | half a day | Results under "Myh4" are the consortium's Myh1; controls lost to ID drift |
| 10 | TESTES: the pooled reference | The collaborator | 1 hour | Levels shift by up to 5.6 points with the choice |
| 11 | Corpus findings for a future corpus | The PI | 1 hour | No retrain was chosen; these decide what a next corpus must fix |

## 1. The muscle reference and its five called blocks

**Open:** `record/step3/downstream/called_blocks_all_readings.tsv` (rows SKMGN, SKMVL);
`figures/step4_v1_purity_sweeps.png`; the reference `data/deconvolution/references_v5/MUSCLE_GSE137869_Y/`
(`cells_meta.tsv`); the label decisions for MUSCLE in `deconvolution/joint_reannotation_review.tsv` (10 rows)
and `deconvolution/JOINT_REANNOTATION_RESULTS.md`.

**The facts.** One reference of 10,740 cells from two libraries serves both muscles. The called types and their
reference cells: satellite cells 261, tenocytes 212, vascular smooth muscle cells 167, Schwann cells 30. Each
is 0.08% to 0.7% of the estimated tissue. At 0.5% of a mixture their estimated expression correlates at 0.50
to 0.57 with the truth; Schwann cells cannot be tested. SKMGN tenocytes were a hotspot in the earlier default
and are not now; SKMGN Schwann cells and SKMVL tenocytes are now and were not.

**Questions.**
- Are the four cell types real and correctly named in this reference (markers per cluster in the results
  document)? Is a 30-cell Schwann-cell profile usable at all?
- Is it credible that training changes the expression of these types in a way a 0.2% component of bulk muscle
  can reveal, given the sweep result?
- Should blocks below a fraction floor be excluded from the gate? The registered rule keeps them and reports
  a tier; the PI declined a floor chosen with the result in view. An expert's floor, set on other grounds,
  would be a legitimate new rule.

**What a "no" changes.** The muscle hotspots, including the top one (SKMVL satellite cells), would be reported
as unsupported by the deconvolution. Seven called blocks would remain.

## 2. The BLOOD composite reference and its four hotspots

**Open:** `deconvolution/BLOOD_REFERENCE_PROPOSAL.md` and `deconvolution/build_blood_composite.py`; the BLOOD
rows of `called_blocks_all_readings.tsv` and `record/step4/v1_purity_sweeps/sweep_points.tsv`; the review rows
BLOOD_PBMC (17), BLOOD_WB (6), BLOOD_BM (12).

**The facts.** The reference is assembled from three sources. Erythroid cells are 90 cells of one
adjuvant-treated library and are estimated at 71% of blood; their expression is not recovered at any purity
(0.58 to 0.92). Neutrophils are 2,168 cells of one library. Classical monocytes and megakaryocytes pass eight
of nine readings; all four fail DE robustness (item 4).

**Questions.**
- Is a 90-cell erythroid profile from an immunised animal an acceptable stand-in for the red-cell lineage of
  whole blood? If not, the erythroid hotspot and the 71% estimate are artefacts of the profile.
- Do the other blood types' estimates depend on it? (The erythroid component absorbs most counts.)
- Are the PBMC cluster labels right (17 decisions, 11 clusters dropped)?

**What a "no" changes.** The erythroid hotspot is withdrawn, and the other three need a rerun on a reference
with a defensible erythroid profile before they are claimed.

## 3. The LUNG and HEART references and their three hotspots

**Open:** LUNG and HEART rows of `called_blocks_all_readings.tsv`; `deconvolution/JOINT_REANNOTATION_LITCHECK.md`
(the LUNG clusters checked and kept); review rows LUNG (20), HEART (4).

**The facts.** LUNG endothelial cells and vascular smooth muscle cells are hotspots that do not separate at 8
weeks against control (AUC 0.52 and 0.61) and that expression alone separates at 0.60 and 0.59. LUNG's source
studies deposited matrices without mitochondrial genes, so the quality of its reference cells cannot be
measured. The reference has no alveolar type I cluster. HEART vascular smooth muscle cells (810 cells of 17
libraries, 1.3% of the tissue) do not transfer by the corrected rule and are recovered at 0.68 at their
abundance.

**Questions.**
- Is a training effect in lung endothelium that is present across the pooled time points and absent at 8 weeks
  biologically plausible, or is it the signature of something else that differs between arms?
- Are "endothelial cells" at 53% of estimated lung, and no type I cells, an acceptable reference for lung?
- HEART: is the "CD8+ T cells into T cells" merge right (one of 19 samples carried the label)?

## 4. The DE model against technical covariates

**Open:** `STEP3_RESULTS.md` B5; `record/step3/downstream/refs5_de_covariate_robustness.tsv`,
`technical_covariates_by_group.tsv`; `deconvolution/R/run_pseudobulk_de.R` (the model is described in its header).

**The facts.** With RIN and globin percentage in the model, 44% of the dose-significant gene x block pairs
remain; with four covariates, 33%. No block meets the registered verdict. The covariates do not differ between
training groups beyond chance. An older check on a simpler model reads the hotspots as robust.

**Questions.**
- Which model is the right primary: with or without technical covariates? Rerun B uses the four covariates
  the consortium used for its own bulk analysis (`STEP3_PREREG.md` 6.4); this pipeline's headline uses none.
- Is the loss a loss of power (five animals per group and sex, two to four more parameters) or evidence that
  the unadjusted calls ride on technical variation? What analysis would tell them apart (for example
  residualising expression on the covariates within group, or a permutation of the covariates)?
- Is the registered verdict (80% retained, rank correlation 0.9) a sensible bar?

**What the answer changes.** If the adjusted model should be primary, the headline falls from 52,961 pairs to
about 27,000 or 22,000 and every per-tissue count changes.

## 5. The gate's construction and what "hotspot" may claim

**Open:** `STEP3_PREREG.md` section 5; `deconvolution/mc_gate.py`; `STEP3_RESULTS.md` B4;
`record/step3/downstream/gate_auc_default_vs_refs5.tsv`.

**Questions.**
- Is the sequential Monte Carlo rule (interval limits, Benjamini-Hochberg at both limits) implemented as the
  cited construction, and is the error level right for 150 blocks and repeated looks?
- A change of about 100 excluded genes and 5% in the target sum moves a block's AUC by 0.03 at the median and
  moves blocks across the line. Should a hotspot be required to hold under such perturbations (a stability
  rule), and how would one be registered?
- The sex-stratified sensitivity passes 55 of 150 blocks; the primary passes 12. Is the primary's permutation
  scheme (unrestricted over animals of both sexes) the right null when arms are balanced by sex?
- The trained arm pools 1, 2, 4 and 8 weeks against control (40 against 10). Is that the right primary
  contrast? Three of twelve called blocks do not hold at 8 weeks alone.

## 6. HYPOTH: eight label decisions and one sample

**Open:** `STEP3_RESULTS.md` A2 (the decision table) and B1; `deconvolution/joint_reannotation_review.tsv`
(HYPOTH rows); `figures/step3_hypoth_labels.png`.

**Questions.**
- Are the eight decisions right, in particular the three clusters dropped as doublets (479 cells) and the
  arcuate AgRP cluster kept as "Neurons"?
- No cluster passed the GABAergic test (best score 0.49 against 0.5). GABAergic clusters are therefore
  labelled "Neurons". Should the reference carry a GABAergic type on an expert's reading of the clusters?
- The deconvolution puts 58% of hypothalamus in glutamatergic neurons and 2% in "Neurons". Is that a usable
  estimate?
- Sample 90406015402 (female, 4-week) has a composition unlike the other 49 under both references. Is it a
  dissection that reached a neighbouring region? Should it be excluded, and by what rule?

HYPOTH has no called block in refs5, so nothing in the headline rests on this item.

## 7. SMLINT and KIDNEY reference cell quality

**Open:** `STEP5_RESULTS.md` M1; `record/step5/mito_by_reference_refs5_direct.tsv`.

**The facts.** 51% of SMLINT reference cells and 17% of KIDNEY reference cells are above 15% mitochondrial
counts; every cell type of both is flagged. Enterocytes have a median of 20%. SPLEEN and HIPPOC have two types
each at the 5% bound; 9 of 42 TESTES Sertoli cells are above 15%.

**Question.** For each tissue: is this biology (intestinal epithelium and kidney tubule are rich in
mitochondria) or damage? If damage, at what cut should the reference be rebuilt? A cut is a new analysis
choice and needs its own registration.

Neither tissue has a called block.

## 8. The other 160 label decisions

**Open:** `deconvolution/joint_reannotation_review.tsv` and, per reference, the cluster marker tables named in
`deconvolution/JOINT_REANNOTATION_RESULTS.md`; `deconvolution/JOINT_REANNOTATION_LITCHECK.md`, including its
list of refinements suggested and not applied.

| Reference | Decisions | Clusters dropped |
|---|---|---|
| HIPPOC | 22 | 12 |
| LUNG | 20 | 9 |
| BLOOD (three sources) | 35 | 11 |
| WATSC | 16 | 2 |
| SMLINT | 12 | 3 |
| SPLEEN | 11 | 3 |
| MUSCLE | 10 | 1 |
| OVARY | 10 | 0 |
| KIDNEY | 9 | 4 |
| LIVER | 6 | 0 |
| CORTEX | 5 | 0 |
| HEART | 4 | 2 |

MUSCLE, BLOOD, LUNG and HEART are covered by items 1 to 3. The citations in the literature check were not
verified against the papers. HIPPOC stays flagged unreliable (PI decision 3): it is an ambient-RNA sink that
two correction attempts did not fix, and a review of its labels will not change that.

## 9. Gene identity

**Open:** `STEP5_RESULTS.md` M3; `record/step5/bulk_direct_ids_other_gene_or_alignment_lost.tsv` (368 IDs);
`record/step4/v2_posctrl/not_testable_trace.tsv`.

**The facts.** The ID the consortium calls Myh1 is Myh4 in the annotation the pipeline uses, and carries 1.3%
of all bulk counts; alpha-globin copies are similarly shifted. Heat-shock controls (Hspa1b, Hsp90aa1) cannot
be tested in muscle and heart because the reference studies' counts sit under other IDs. Four reference studies
have more than 0.5% of counts under another gene's token (SMLINT 1.7%, KIDNEY and LIVER 0.8%, the WATSC and
muscle study 0.5%).

**To do (pipeline).** Decide whether to carry a symbol-based bridge for the 349 bulk IDs and the reference
studies, and rerun; this changes inputs and needs registration. Until then, any gene-level result should be
checked against the list. **Question (tissue):** which of the 349 matter for exercise biology besides the
myosin heavy chains?

## 10. TESTES

**Open:** `STEP3_RESULTS.md` B2; `record/step3/stage8/testes_controlonly_vs_primary_by_type.tsv`.

The pooled reference (control plus the animal seven weeks after EDS) and the control-only reference rank the
25 animals the same way (r of 0.988 or more) and differ in levels by up to 5.6 points; Sertoli cells are 37%
against 32%. **Question for the collaborator:** keep the pooled reference, with the levels read as dependent
on that choice, or switch?

## 11. Corpus findings, for a future corpus

**Open:** `STEP5_RESULTS.md`, in particular "What building the set showed about Stage 2" and the switches
table. Under the no-retrain decision these are disclosures. For any future corpus the PI should decide, before
harvesting: the mitochondrial cut and whether it is tissue-specific; the five switches; a rule for non-rat
files without a sample accession; a rule against second deposits; whether spatial spots and unfiltered
matrices belong in the corpus; and whether the 33 held-out series stay held out.

## What review cannot settle

| Limit | What it would take |
|---|---|
| No reference matches the bulk's strain; 9 of 15 references rest on one or two libraries | New single-cell data from the study's own animals or strain |
| BayesPrism's fractions are shares of mRNA, not of cells (Step 4, V5) | Cell counts or imaging on the study's tissue; or a correction registered and validated |
| The purity sweeps are a best case: reference and mixtures come from one study | Mixtures or sorted populations with known truth from independent data |
| The registered positive controls do not recover | Either a repaired pipeline that recovers them, or evidence that the controls were mis-specified that is not drawn from this pipeline's own output |
| The effect of the corpus findings on the trained model | A retrain |
| WATSC has no adipocytes; LUNG has no type I cells | References from protocols that capture them |
| Transfer and network "self-consistency" involve no human measurement | Human exercise data |
