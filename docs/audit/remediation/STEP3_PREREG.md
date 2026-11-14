# Step 3 registration: one rerun of Stages 8-14 that carries the audit's result-changing fixes

Registered 2026-11-14, before any output of the rerun exists. The rerun is called **refs5**. It writes to new
folders and leaves the current default (`_refs4all_rebuild`) untouched until the comparison in section 10 is
done.

- **What this is.** The rules for every change to Stages 8-14 that the audit (`docs/audit/FINDINGS.md`) calls
  for, fixed before the changed pipeline produces a number.
- **What it is not.** A blind registration. The current default's results are known, and section 0 lists what
  was known when each rule was written.
- **Who wrote it.** An AI coding assistant (Claude), under the PI's decisions of 2026-11-14 (section 1). No
  domain expert has reviewed it.

References in square brackets are listed in section 12. Each was looked up on Crossref on 2026-11-14
(`record/crossref_remediation.tsv`) or is already in `docs/audit/REFERENCES.md`.

## 0. What was already known, and the principle that follows

**Known when this was written (the current default, v2 + refs4, 16 tissues):**
- every block's gate AUC and permutation p, the 9 hotspots, and that two LUNG blocks miss the cut at p = 0.0037
  and 0.0039;
- every block's DE count, mean fraction and median library; SKMVL satellite cells, the top hotspot, have a mean
  fraction of 0.0023;
- that Benjamini-Hochberg on the human side passes 7 of the 9 hotspots (re-derived in Step 2);
- the networks' hub tables and the negative decoy null.

**Principle.** Rules written with results in view can be bent toward a preferred answer, even without intent
[simmons2011; nosek2018]. So the changes are split in two kinds.

| Kind | Rule | Examples here |
|---|---|---|
| **Repair** | A defect with one correct fix. Goes into the primary analysis. | A gene list that excludes kinases as "ribosomal"; a combination rule that returns nothing for single-sex tissues; a difference taken against zero; a covariate type that does not match the covariate |
| **Tier or sensitivity** | A choice with a range of defensible values, whose effect on known results was visible. The registered primary stays; the alternative is reported beside it for every block. | A minimum cell-type fraction; the 8-week-only contrast; sex-stratified permutations |

One change sits between the two and is stated as such: the hotspot definition (section 5.2). The statistic and
the multiplicity rule do not change. What changes is that a block is called a hotspot only when more
permutations could not reverse the call.

## 1. Decisions taken by the PI (2026-11-14)

| # | Decision | Effect on this registration |
|---|---|---|
| 1 | No retrain | The model is v2 throughout. Corpus and model findings are measured and disclosed in Step 5 |
| 2 | Keep the pooled trained-against-control contrast as the gate's primary | Section 5.1; the alternatives are sensitivities (5.3) |
| 3 | HIPPOC stays flagged unreliable | Its reference is not changed; it is excluded from headline claims |
| 4 | Module A criterion 2 is withdrawn for now | Not rescored as target validation; criteria 1, 3, 4 are rerun in Step 4 |
| 5 | A plan for what an expert should review, written after everything is done | `LABEL_REVIEW_PLAN.md`, last |

## 2. Stage 8 inputs

### 2.1 Gene exclusion list (finding 4.3) - repair

**Defect.** `deconvolution/reference/rat_exclude_genes.tsv` lists 226 genes as ribosomal. Ten of them are
ribosomal protein S6 kinases (Rps6ka1-6, Rps6kb1, Rps6kb2, Rps6kc1, Rps6kl1). They are kinases that
phosphorylate ribosomal protein S6, not ribosomal proteins [ban2014], and Rps6kb1 (p70 S6 kinase) is a
read-out of mTOR signalling that an exercise study would want to keep. No script builds the list.

**Evidence for the fix.** BayesPrism's own gene lists (`vendor/BayesPrism/BayesPrism/inst/extdata/
genelist.hs.new.txt`, `genelist.mm.new.txt`), which this file is meant to replace for rat, contain none of
the ten as ribosomal in either species. RPS6KA3 and RPS6KA6 appear there only as X-linked genes.

**A correction to the audit.** The audit counted eleven and included Rps19bp1. BayesPrism lists RPS19BP1 under
`other_Rb` in both human and mouse. The file's stated purpose is to reproduce BayesPrism's cleanup for rat
[chu2022], so Rps19bp1 stays excluded. The count is ten.

**Rule.**
- `deconvolution/build_exclude_genes.py` builds the list from `data/references/biomart/rat_gene_info.tsv`:
  ribosomal = symbol matches `^(Rp[sl]|Mrp[sl])` (case-insensitive) and does not match `^Rps6k`;
  mitochondrial and hemoglobin = the current members, by their rule in the script.
- The built list must equal the current list minus the ten kinases. Any other difference is printed, and the
  current membership is kept for those genes. A test asserts this.
- **Expected effect, stated now:** eight genes enter the deconvolution (Rps6ka3 and Rps6ka6 are X-linked and
  stay out through the sex-chromosome filter). Fractions should move by far less than a percentage point.

### 2.2 References (findings 4.1, 4.8, 4.9)

**All 15 are rebuilt from committed code into `data/deconvolution/references_v5/`** by
`build_references_from_config.py`, from the deposited per-cell labels, and compared file by file (md5 of
`reference_counts.mtx`, `genes.tsv`, `cells_meta.tsv`) with the builds in use.
- Identical: the provenance gap (7 of 15 with no record, 6 built from an uncommitted tree) is closed with no
  change to any result.
- Not identical: the difference is reported (cells, labels, genes) and the rebuilt reference is used, because
  it is the one that can be reproduced.

**Three references change by design.**

| Reference | Change | Kind | Basis |
|---|---|---|---|
| HYPOTH | Re-annotated by the registered joint method (`REFERENCE_RELABEL_PREREG.md` §10.3), with the panel in 2.3 | Repair | §10.2 deferred it "to go through the same code later with a panel"; the per-sample labels are the procedure §10.1 discredited |
| HEART | "CD8+ T cells" (1,624 cells, 1 of 19 samples) merged into "T cells" | Repair | The label audit's rule gave MERGE (`REFERENCE_LABEL_AUDIT.md:150`); it was never applied |
| TESTES | **Primary unchanged** (control + E7W). A control-only build is a sensitivity | Sensitivity | See below |

**TESTES departs from the plan of 2026-11-14.** The plan said "drop the EDS arm". On re-opening the
configuration, the pooling is a recorded decision of the collaborator: "pool C + E7W = normal-testosterone
arms (Geyu)" (`tissue_references.yaml`, TESTES). E7W is seven weeks after ethane dimethanesulfonate, when
Leydig cells and testosterone have recovered. Reversing a collaborator's recorded choice was not what the
audit finding established; it established that the choice is undocumented in the validator. So:
- the primary reference stays C + E7W (5,850 cells);
- a control-only reference (sample C, 2,155 cells) is built and deconvolved, and reported as: per-type
  Pearson r of the fractions over the 25 samples, mean absolute difference in fraction, and the number of
  types kept under the 20-cell state rule;
- the PI or the collaborator decides from that table.

**Not changed.** HIPPOC (decision 3). BLOOD's erythroid profile, 90 cells from one library (4.6), for which no
alternative data are on disk; it is listed in the review plan.

### 2.3 HYPOTH marker panel, written before any clustering of HYPOTH

Shared classes are the registered brain panel (`reference_marker_panels.yaml`, `CORTEX`), unchanged:
Astrocytes, Oligodendrocytes, Oligodendrocyte precursor cells, Microglia, Border-associated macrophages,
Endothelial cells, Mural cells, Vascular leptomeningeal cells, Choroid plexus epithelial cells, Ependymal
cells, T cells. The cortical "Excitatory neurons" and "Inhibitory neurons" classes are replaced, because
their markers (Slc17a7, Satb2, Neurod6, Tbr1) are cortical.

| Class | Compartment | Genes | Basis |
|---|---|---|---|
| Neurons | neural | Snap25, Syt1, Rbfox3, Stmn2, Syn1 | Pan-neuronal |
| - GABAergic neurons (sub) | | Slc32a1, Gad1, Gad2 | Vesicular GABA transporter and the two GABA-synthesis enzymes [chen2017; campbell2017; steuernagel2022] |
| - Glutamatergic neurons (sub) | | Slc17a6 | VGLUT2, the vesicular glutamate transporter of hypothalamic neurons; the three atlases split neurons on Slc17a6 against Slc32a1 [chen2017; campbell2017; steuernagel2022] |
| Tanycytes | glial | Rax, Dio2, Col23a1, Crym | [campbell2017; chen2017] |

**One departure from §10.3, stated now.** The registered rule asks for 3 to 6 genes per panel and does not
score a panel with fewer than 2. Hypothalamic glutamatergic neurons have one class-wide marker; the others
are specific to single nuclei. Padding the panel with genes that are absent from most of the class would
lower every glutamatergic cluster's score under the registered scoring (mean of min-max-scaled genes). So
the glutamatergic sub-class is scored on Slc17a6 alone, with the same 0.5 / 0.15 test against its sibling.
A neuron cluster that passes neither sub-class stays "Neurons", as §10.3 provides.

Everything else is §10.3 as registered: all QC cells of the one library, 3,000 highly variable genes, PCA 50,
no Harmony (one sample), Leiden at resolution 1.0 with seed 0, clusters under 20 cells dropped, review of
clusters that fail the rule recorded with the genes that decided each. Checks K3 (old against new labels) and
K4 (losses) are reported; K1 and K2 need more than one sample.

**The reviewer is again the AI assistant.** That is finding 4.2, and it applies to this reference too. The
review decisions go into the expert review plan.

### 2.4 BayesPrism settings recorded (5.3), runner committed (5.4)

- No setting changes. The run keeps the package defaults (chain 1,000, burn-in 500, thinning 2, seed 123,
  alpha 1; optimiser MAP with sigma 2) [chu2022].
- The values are read from the result object after the run and written to the provenance file, with the
  package version. Nothing is typed in by hand.
- Every tissue runs from one commit with a clean working tree. The provenance field
  `git_modified_tracked_files` must be empty for all 16.

### 2.5 The sample key is written down, not implied (6.9)

Step 1 showed the positional join is correct in all 16 tissues (`STEP1_INTEGRITY.md`). To keep it so:
- Stage 8 writes `mixture_samples.tsv` (mix, viallabel) beside its fractions, from the files BayesPrism read,
  and the md5 of the samples file goes into provenance;
- every later stage reads viallabels through one helper that checks the line count and, where the key file
  exists, its order; a sample index outside the list is an error where it used to be dropped silently;
- `analysis/audit_remediation/check_sample_join.R` and `check_pseudocell_alignment.py` run after the rebuild.

### 2.6 Pseudo-cells carry their own diagnostics (5.7, 5.8)

Each pseudo-cell records its library (sum of Z over genes) and its estimated fraction. The summary lists the
blocks under the analysable rule. Nothing is removed: the registered rule (median library of at least 1,000)
is applied where it always was. The gene-space label written into the files becomes "ensembl-115".

## 3. Stage 9: target sum by the registered formula (6.8)

The rule is already registered (`docs/RETRAIN_V2_PREREGISTRATION.md:155-164`): one target sum T for all
tissues, at which LIVER's pseudo-cells have the corpus's median token value. It was applied once, to the
July-reference LIVER, and kept at 6,400 for the later reference sets by a registered choice
(`REFERENCE_RELABEL_PREREG.md:339-340`).

**Rule for refs5.** Tokenize the refs5 LIVER pseudo-cells at 6,400, read their median non-zero value m, and set

    T = 6,400 x (2^0.9145 - 1) / (2^m - 1), rounded to the nearest 100

where 0.9145 is the v2 corpus median. All 16 tissues and the human transfer use T. On the current LIVER
(m = 0.878) the formula gives about 6,800; the value is fixed by the formula, not chosen.

The weights are loaded through the checked loader (0 missing and 0 unexpected tensors on v2, Step 1), and the
md5 of the weights file goes into every embedding's provenance.

## 4. What does not change

- The model, tokenization, PLS-1 probe, AUC statistic, 10 stratified 5-fold splits and block seeds.
- The analysable rule: median per-sample library of at least 1,000, applied before multiplicity control.
- limma-trend on log2-CPM of Z, the per-sex omnibus F over training duration, Fisher's combination for
  two-sex tissues, and independent hypothesis weighting at 0.05 [ritchie2015; ignatiadis2016].
- HIPPOC's reference; the BLOOD composite; the pooled contrast as the gate's primary.

## 5. Stage 11: the gate (findings 6.1-6.6)

### 5.1 Primary, unchanged

Per block: AUC of a one-component PLS probe for trained (1, 2, 4, 8 weeks) against control (8-week
sedentary), averaged over 10 stratified 5-fold splits; one-sided label-permutation p = (b + 1) / (m + 1)
[phipson2010; north2002]; Benjamini-Hochberg at 0.05 over the analysable blocks [benjamini1995].

### 5.2 A hotspot is a call that more permutations could not reverse - repair of 6.1

**Defect.** With 10,000 permutations a p near the Benjamini-Hochberg cut (about 0.003) has a Monte Carlo
standard error near 0.0006, and two blocks sat 0.0004 and 0.0006 from the cut. The count of hotspots depended
on the random draw.

**Method.** The multiple-Monte-Carlo-test construction of Gandy and Hahn [gandy2014; also sandve2011]:
1. For every block keep the count b of permuted statistics at or above the observed one, out of m.
2. Give every block's ideal p-value (the value an infinite number of permutations would give) a
   Clopper-Pearson interval [L, U] [clopper1934], at a level that makes all intervals hold together with
   probability at least 1 - epsilon.
3. Benjamini-Hochberg is monotone: lowering p-values can only add rejections. So the blocks rejected when
   every p is set to its upper limit are rejected for the ideal p-values too, and the blocks not rejected
   when every p is set to its lower limit are not rejected for the ideal p-values either.

**Classes.**

| Class | Definition | Use downstream |
|---|---|---|
| **Hotspot** | Rejected with every p at its upper limit | The hotspot set of Stages 10, 12, 13, 14 |
| **Borderline** | Rejected at the lower limits, not at the upper | Listed by name wherever the hotspot count is stated; carried through transfer and DE tables; no networks |
| Not a hotspot | Not rejected at the lower limits | |

**Parameters, fixed now.**
- epsilon = 0.01 for the whole family.
- Permutations are drawn in batches of 10,000, up to 1,000,000 per block (100 looks). Sampling stops for a
  block once it is a hotspot or not a hotspot at the current intervals; blocks still undecided at the cap are
  borderline.
- Each interval at each look has two-sided error epsilon / (n_blocks x 100), and a block's interval is the
  intersection of its intervals over looks. By the union bound all intervals hold at every look with
  probability at least 0.99. This spends the error equally over a fixed number of looks, where Gandy and Hahn
  use a spending sequence; the guarantee is the same and the intervals are slightly wider than theirs.
- The first 10,000 permutations use the registered seed stream, so the 10,000-permutation p of the previous
  method is a by-product and is reported beside the new class.

**What 1,000,000 permutations can and cannot settle.** At p near 0.0037 the interval's half-width at the cap
is about 0.0003. A block whose ideal p is within that of the cut stays borderline. That is the correct report
for such a block.

**Implementation check, before use.** The permutation loop is rewritten with precomputed fold kernels (the
probe's out-of-fold scores are linear in the centred labels). On the current default's embeddings it must
reproduce the stored exceedance counts of all 197 blocks at 10,000 permutations. Identical counts are
expected; a difference of at most 2 in a block is accepted as a floating-point tie and reported; anything
larger keeps the old loop.

### 5.3 Sensitivities (6.2, 6.3, 7.5), each reported for every block, none replacing 5.1

| | What changes | Why | Method reference |
|---|---|---|---|
| S1 | Folds stratified on trained x sex; trained labels permuted within sex | The primary ignores sex. A probe can score on sex where the arms' sex balance differs within a fold, and unrestricted permutation assumes the sexes are exchangeable | Restricted permutation within a nuisance factor [winkler2014; andersonterbraak2003] |
| S2 | 8-week trained (10 animals) against control (10) only | Thirty of the forty trained animals were killed earlier than any control, so the pooled contrast mixes training with time in the study. The design has no time-matched control for 1, 2 and 4 weeks [sanford2020; motrpac2024] | Same probe; 10 against 10 has low power, so a miss is not evidence against a block |
| S3 | The same probe on all genes and on 50 principal components of expression | A block whose embedding separates the arms but whose expression does not (two of the nine now) rests on the model alone | Already computed (`pca_control.tsv`); now printed beside every hotspot |
| S4 | Family restricted to blocks with mean estimated fraction of at least 0.01 | Expression of rare types is the least constrained part of a deconvolution [sturm2019; avilacobos2020; chu2022] and no purity sweep covers it | Tier, see 6.3 |

S1, S2 and S4 use 10,000 permutations and Benjamini-Hochberg over their own family.

**Reading rule, fixed now.** A hotspot is described as *supported* if it is also significant in S1 and has
AUC of at least 0.65 in gene space (S3), and as *model-only* if its all-genes AUC is below 0.55. S2 is
reported as a number, with no rule.

### 5.4 Corroboration figures and records (6.4, 6.5, 6.6)

- The gate code prints the figures that are quoted: embedding against PCA-50 and against all genes (paired
  Wilcoxon signed-rank), and the Spearman correlation of Augur with the probe, each over the analysable
  blocks and over all blocks.
- The Augur correlation is reported with no threshold. An earlier row called 0.80 "registered"; no
  registration states it.
- The default merge and the two library-rule sensitivities each write their own table and their own
  provenance file.

## 6. Stage 10: differential expression (findings 7.1-7.9)

### 6.1 Repairs

| Finding | Defect | Fix |
|---|---|---|
| 7.1 | The dose p is Fisher's combination of the two sexes' p-values and is missing when one sex is absent, so TESTES and OVARY can never have a call | The combination runs over the sexes present. With one sex, Fisher's statistic -2 ln p on 2 degrees of freedom returns p itself, so the dose p is that sex's omnibus p. The per-sex fit is run for single-sex tissues, which the code skipped |
| 7.8 | IHW is given tissue as a factor under its default `covariate_type = "ordinal"`, which smooths weights across alphabetically adjacent tissues | `covariate_type = "nominal"` [ignatiadis2016]. Affects power, not error control |
| 7.7 | The methods sidecar says `~sex*factor(week)` | It states the models the code fits |
| 7.10 | `run_stage10.py --alpha` does not reach the R step | It does |

Single-sex blocks enter the pooled IHW family for the first time (21 blocks now). That changes every
tissue's weights slightly; it is the registered consequence of the repair.

### 6.2 Sex-consistency (7.2, 7.9)

- **Primary:** concordance. A gene is sex-consistent at 8 weeks when both per-sex 8-week tests pass
  Benjamini-Hochberg at 0.05 within the block and the signs agree. This is what the manuscript already reads.
- **Secondary:** repfdr [heller2014aoas; repfdr], fitted as the consortium fits it, on training-regulated
  features only (dose FDR below 0.05) and not on every gene-block. The consortium's use is described in Vetr
  et al. [vetr2024], Methods: features differential "at 5% IHW" enter the repfdr step.
- The table reports three counts: concordance, repfdr, and their union. The stored column is renamed to what
  it holds, a tail-area Fdr.
- The male/female imbalance in 8-week calls is in the consortium's own bulk results (Step 1, Spearman 0.90
  across tissues). It is reported, not corrected.

### 6.3 Fraction tier (7.5) - tier, not a new primary rule

Every analysed block is labelled **low-fraction** when its mean estimated fraction is below 0.01. The headline
is reported three ways each time it is stated: gene x block pairs, distinct genes, and pairs outside
low-fraction blocks.

Why 0.01 and why a tier:
- deconvolution benchmarks find the estimates of rare types the least reliable [sturm2019; avilacobos2020],
  and the BayesPrism paper validates expression above 50% purity [chu2022];
- no value is derived from this project's data, and 0.01 is the round figure the audit used;
- the top hotspot (SKMVL satellite cells, 0.0023) is below it, and that was known when this was written.
  Making it the primary rule would be a choice made with the result in view. As a tier, every reader sees both.

### 6.4 Robustness on the headline statistic (7.6)

The whole DE is rerun twice with technical covariates added to every model, into sibling folders:
- **A:** RIN and percent globin (the covariates of the earlier check);
- **B:** RIN, percent globin, percent UMI duplication and median 5'-3' bias, the four the consortium used for
  its own differential analysis (`TRNSCRPT_*_DA`, column `covariates`) [motrpac2024].

RNA quality and library technical factors are known to move expression estimates [gallegoromero2014;
leek2010]. For each block, against the headline run: genes retained, genes lost, genes gained, and the
Spearman correlation of -log10 of the dose p.

**Verdict per hotspot, fixed now (thresholds are the earlier check's):** ROBUST if the Spearman correlation is
at least 0.9 and at least 0.8 of its dose-significant genes are retained, under both A and B. A hotspot with
no dose-significant gene is reported as NO CALLS and is not counted as robust. Gains are printed.

### 6.5 Positive controls

Rescored on refs5 by the registered specification, unchanged. The pipeline-sanity subset (34 muscle and heart
controls) is reported first, as recovered / not significant / wrong direction / not testable.

## 7. Stage 12: transfer (findings 8.1, 8.2, 8.4)

- The registered rule (human AUC of at least 0.65 and raw permutation p below 0.05) is kept and reported.
- **Added:** Benjamini-Hochberg over the analysable family on the human side, with the classes of 5.2. A
  hotspot **transfers** if it is a hotspot or borderline in human space and its human AUC is at least 0.65.
- **Added:** the dose trend. Each row carries the rat and human dose correlations and the flag "dose trend
  lost" when the human value is zero or negative.
- **Stated wherever the count is quoted:** the human test reuses the rat test's cells, labels, folds and
  seeds. It shows that the model's human-species reading keeps the structure. It is not a replication.
- Pseudo-cells that exist in rat space only (below 200 mapped genes after orthology; 20 LUNG goblet cells
  now) are listed, and the test that asserted equal counts asserts that the human cells are a subset.

## 8. Stage 13: networks and enrichment (findings 9.1, 9.3, 9.5, 9.7)

### 8.1 The difference is taken on shared genes - repair of 9.1

**Defect.** `diff()` fills a target missing from one arm with 0, so a gene in one arm's top 2,048 only is
differenced against zero. Those edges carried a median 37% and 53% of hub strength in two networks.

**Fix.** An edge exists only where the TF and the target are in both arms' profiles: difference = trained -
control on the intersection. Per network the table records how many TFs and targets were in one arm only.
Hub strength stays the sum of absolute differences over all of a TF's edges.

**Decoy null.** Rerun with the same rule, because it shared the defect. Reported with the number of distinct
TFs beside the TF x network count, and the match distance of each decoy is written out.

Networks are built for hotspots (class of 5.2) under the existing abundance rule. Hub tables remain model
read-outs: no table is called a list of regulators.

### 8.2 Enrichment (9.5) - repair

| | Now | refs5 |
|---|---|---|
| Rat to human | Upper-cased rat symbol | The one-to-one pairs of the Stage 12 ortholog map; a rat gene with no pair is dropped and counted |
| Ranking statistic | sign(slope) x -log10 of the sex-adjusted omnibus p | sign(slope) x -log10 of the dose p that defines the headline (Fisher combination, or the one sex's p) [reimand2019; gsea; fgsea] |
| Gene sets | Per-user cache | The table used is saved with its md5 and the msigdbr version beside the results |
| Tests with no p | Silent | Counted in the summary |

## 9. Stage 14

Purity sweeps on the refs5 references, as now. The registered pass rule and the sweeps for hotspot cell types
belong to Step 4 and have their own registration.

## 10. How refs5 is compared with the current default, and when it becomes the default

Reported in `STEP3_RESULTS.md`, with a figure for each:
1. references: identical or not, per reference; HYPOTH old against new labels; TESTES sensitivity;
2. fractions: per tissue, Pearson r and mean absolute difference of every type's fraction across samples;
3. pseudo-cells, analysable blocks, target sum;
4. gate: class of every block, the 10,000-permutation p beside the final p and interval, and the four
   sensitivities;
5. DE: pairs, distinct genes, tier split, single-sex tissues, sex-consistency by source, robustness;
6. transfer: registered rule, corrected rule, dose trend;
7. networks: edges, share of TFs and targets in one arm only, decoy null; enrichment.

refs5 becomes the default configuration when every stage has finished from one clean commit, the two join
checks pass, and the test suite passes. The current default stays available as
`config/profiles/refs4all_rebuild.yaml`.

**Whatever the numbers are, they are reported.** If the hotspot count falls, or a named hotspot becomes
borderline or model-only, the documents and the manuscript say so.

## 11. Not covered here

Step 4 (purity sweeps for hotspot types, held-out model test, Module A rerun, SimBu baseline, mixture panel
settings) and Step 5 (corpus measurements) are registered separately before they run.

## 12. References

Looked up on Crossref on 2026-11-14 unless marked as already in `docs/audit/REFERENCES.md` (REF).

| Key | Reference |
|---|---|
| andersonterbraak2003 | Anderson M, ter Braak C. Permutation tests for multi-factorial analysis of variance. J Stat Comput Simul 2003;73(2):85-113. doi:10.1080/00949650215733 |
| avilacobos2020 | Avila Cobos F, et al. Benchmarking of cell type deconvolution pipelines for transcriptomics data. Nat Commun 2020;11:5650. doi:10.1038/s41467-020-19015-1 |
| ban2014 | Ban N, et al. A new system for naming ribosomal proteins. Curr Opin Struct Biol 2014;24:165-169. doi:10.1016/j.sbi.2014.01.002 |
| benjamini1995 | Benjamini Y, Hochberg Y. Controlling the false discovery rate. J R Stat Soc B 1995 (REF) |
| campbell2017 | Campbell JN, et al. A molecular census of arcuate hypothalamus and median eminence cell types. Nat Neurosci 2017;20(3):484-496. doi:10.1038/nn.4495 |
| chen2017 | Chen R, Wu X, Jiang L, Zhang Y. Single-cell RNA-seq reveals hypothalamic cell diversity. Cell Rep 2017;18(13):3227-3241. doi:10.1016/j.celrep.2017.03.004 |
| chu2022 | Chu T, et al. Cell type and gene expression deconvolution with BayesPrism. Nat Cancer 2022 (REF) |
| clopper1934 | Clopper CJ, Pearson ES. The use of confidence or fiducial limits illustrated in the case of the binomial. Biometrika 1934;26(4):404-413. doi:10.1093/biomet/26.4.404 |
| fgsea, gsea | Korotkevich G, et al. Fast gene set enrichment analysis; Subramanian A, et al. Gene set enrichment analysis (REF) |
| gallegoromero2014 | Gallego Romero I, Pai AA, Tung J, Gilad Y. RNA-seq: impact of RNA degradation on transcript quantification. BMC Biol 2014;12:42. doi:10.1186/1741-7007-12-42 |
| gandy2014 | Gandy A, Hahn G. MMCTest - a safe algorithm for implementing multiple Monte Carlo tests. Scand J Stat 2014;41(4):1083-1101. doi:10.1111/sjos.12085 |
| heller2014aoas, repfdr | Heller R, Yekutieli D. Replicability analysis for genome-wide association studies; Heller R, Yaacoby S, Yekutieli D. repfdr (REF) |
| ignatiadis2016 | Ignatiadis N, et al. Data-driven hypothesis weighting increases detection power in genome-scale multiple testing. Nat Methods 2016 (REF) |
| leek2010 | Leek JT, et al. Tackling the widespread and critical impact of batch effects in high-throughput data. Nat Rev Genet 2010;11(10):733-739. doi:10.1038/nrg2825 |
| motrpac2024, sanford2020 | MoTrPAC Study Group. Temporal dynamics of the multi-omic response to endurance exercise training. Nature 2024; Sanford JA, et al. MoTrPAC: mapping the dynamic responses to exercise. Cell 2020 (REF) |
| north2002 | North BV, Curtis D, Sham PC. A note on the calculation of empirical P values from Monte Carlo procedures. Am J Hum Genet 2002;71(2):439-441. doi:10.1086/341527 |
| nosek2018 | Nosek BA, Ebersole CR, DeHaven AC, Mellor DT. The preregistration revolution. Proc Natl Acad Sci 2018;115(11):2600-2606. doi:10.1073/pnas.1708274114 |
| phipson2010 | Phipson B, Smyth GK. Permutation P-values should never be zero. Stat Appl Genet Mol Biol 2010 (REF) |
| reimand2019 | Reimand J, et al. Pathway enrichment analysis and visualization of omics data using g:Profiler, GSEA, Cytoscape and EnrichmentMap. Nat Protoc 2019;14(2):482-517. doi:10.1038/s41596-018-0103-9 |
| ritchie2015 | Ritchie ME, et al. limma powers differential expression analyses. Nucleic Acids Res 2015 (REF) |
| sandve2011 | Sandve GK, Ferkingstad E, Nygard S. Sequential Monte Carlo multiple testing. Bioinformatics 2011;27(23):3235-3241. doi:10.1093/bioinformatics/btr568 |
| simmons2011 | Simmons JP, Nelson LD, Simonsohn U. False-positive psychology. Psychol Sci 2011;22(11):1359-1366. doi:10.1177/0956797611417632 |
| steuernagel2022 | Steuernagel L, et al. HypoMap - a unified single-cell gene expression atlas of the murine hypothalamus. Nat Metab 2022;4(10):1402-1419. doi:10.1038/s42255-022-00657-y |
| sturm2019 | Sturm G, et al. Comprehensive evaluation of transcriptome-based cell-type quantification methods for immuno-oncology. Bioinformatics 2019;35(14):i436-i445. doi:10.1093/bioinformatics/btz363 |
| vetr2024 | Vetr NG, et al. The impact of exercise on gene regulation in association with complex trait genetics. Nat Commun 2024 (REF); the repfdr passage was read in `readings/Gene_Impact_PA.pdf`, page 8 |
| winkler2014 | Winkler AM, et al. Permutation inference for the general linear model. NeuroImage 2014;92:381-397. doi:10.1016/j.neuroimage.2014.01.060 |
