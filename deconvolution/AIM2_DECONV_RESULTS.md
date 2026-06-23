# Deconvolution → GeneCompass (Aim 2 bridge) — Results & Next Stages

_Status 2026-06-15 (supervised re-measurement + cross-method corroboration in §3b; external + cross-species validation in §3c; the §3 "exercise is small" reading is superseded — but see the §3c magnitude reconciliation). Companions: `MOTRPAC_BULK_LIFTOVER.md` (bulk gene-ID prep), `EMBEDDING_DE_STANDARDS.md` (methods in the literature), `README.md`._

---

## 1. The pipeline (grant Aim 2)

MoTrPAC bulk RNA-seq → **BayesPrism** deconvolution (per tissue, against rat single-cell references) → per-cell-type **pseudo-cells** (one per *bulk sample × cell type*, raw deconvolved expression `Z`) → corpus **tokenizer** → fine-tuned **rat GeneCompass** → 768-d cell embeddings.

The question Aim 2 asks of those embeddings: **does within-cell-type variation track the MoTrPAC exercise design (training dose, sex)?** If yes, cell-type-resolved downstream analyses (differential expression, GRN, in-silico perturbation) can recover exercise biology that bulk RNA-seq blurs across cell types.

---

## 2. Status — what is built and validated

| Stage | Status | Key artifact / result |
|---|---|---|
| Bulk gene-ID liftover & prep | ✅ | 32,432 / 32,883 ENSRNOG mapped; primary-gene vocab coverage 95.0%. See `MOTRPAC_BULK_LIFTOVER.md` |
| Per-tissue SC references (10 tissues) | ✅ | incl. **heart fix** (GSE280111 left ventricle — has cardiomyocytes) and **cortex** gene-rich union build |
| BayesPrism validation (per tissue) | ✅ | liver holdout **0.998** / cross **0.949** (meets the Chu 2022 paper ≥0.95); heart cardiomyocyte r=**0.995**; WAT corroborated across 5 methods (DWLS 0.98/0.99); **lung is the weakest cross (~0.73)** |
| Connector (pseudo-cells → tokenize → embed) | ✅ | liver smoke test: silhouette **0.775**, kNN purity **1.000**, 83% between-/17% within-cell-type variance |
| End-to-end run (all 10 tissues) | ✅ | **9,300 pseudo-cells** embedded (768-d) on disk |
| Aim-2 validation gate | ✅ | **186** (tissue × cell-type) rows across all 10 tissues (heart added 2026-06-11) |
| Interactive viewer | ✅ | self-contained, offline 3-tab `viewer.html` (Atlas / Signal / Focus) |
| **Supervised re-measurement** (§3b) | ✅ | exercise is concentrated, **not** absent: held-out AUC up to **0.91**; **22** FDR-significant hotspots (vs 5 under the trace gate) |
| **Cross-method corroboration** (§3b) | ✅ | canonical **Augur** reproduces it (Spearman **r=0.83**; 19–22/22 hotspots); embedding beats PCA-50, ties full genes |
| **External + cross-species validation** (§3c) | ✅ | same-data **Vetr 2024** corroborates hotspot geography + supplies the DE recipe & rat→human blueprint; human PBMC scRNA (**Yu 2023**) validates the immune hotspot; per-gene magnitude is modest → "reliably detectable," not "large" |

Tissues covered: blood, cortex, heart, hippoc, kidney, liver, lung, skmgn (gastrocnemius), skmvl (vastus lateralis), watsc (WAT). Design per tissue: 2 sex × 5 exercise group (control / 1w / 2w / 4w / 8w) × 5 reps = ~50 samples per cell type.

---

## 2a. How bulk is fed to GeneCompass — granularity & cross-tissue caveat

The data enters at **two nested levels**:
- **Tissue by tissue.** Each tissue is deconvolved against its *own* tissue-specific rat single-cell reference, and built → tokenized → embedded as a separate run against the same frozen fine-tuned rat GeneCompass checkpoint.
- **One pseudo-cell per (bulk sample × cell type)** is the unit fed to the model — its vector is the slice of *that one sample's* bulk that BayesPrism attributes to *that* cell type (raw `get.exp`), each tokenized and CLS-embedded **independently** (GeneCompass has no cross-cell attention). This deliberately preserves per-sample variation — the exercise/sex signal the gate and DE depend on. The point counts confirm it: e.g. blood 700 = 14 cell types × 50 samples; nothing is averaged across samples.

**⚠ Cross-tissue comparability caveat.** Because each tissue uses a different reference, the *same* cell-type label across tissues (e.g. "Macrophages" in WAT vs lung) is deconvolved against different references from different mixtures and is **not strictly the same population** in the embedding. The gate and the planned per-cell-type DE operate **strictly within a single (tissue × cell type)**, so they are insulated. What is **not** licensed: comparing the same cell type across tissues, or reading the Atlas's between-tissue layout as biology. To compare a shared cell type across tissues (e.g. a common immune exercise response), use a shared pan-tissue reference for those types, or scope claims within-tissue.

**Value-channel QC (per-tissue normalization drift).** Tokenization library-normalizes each pseudo-cell to a single global `target-sum = 6500`. The token *ranking* (`input_ids`, the primary signal) is normalization-invariant, but the `values` channel's median drifts by tissue vs the corpus reference (0.869): liver +0.3%, blood +1.4%, watsc +4.4%, skmgn +5.9%, hippoc +8.0%, heart +8.4%, cortex +9.1%, skmvl +12.4%, lung +17.0%, **kidney +23.3%**. The drift is ~constant within a tissue, so **within-tissue contrasts are unaffected**; it adds a small extra term to the cross-tissue confound above. If cross-tissue value comparisons become important, calibrate `target-sum` per tissue so each value-median lands near 0.869.

**Viewer note.** The Atlas UMAP is a cosine UMAP of the **768-d GeneCompass CLS embeddings** (not the raw input). The Focus tab projects those same embeddings onto a **pooled within-cell-type PCA**: each (tissue × cell-type) centroid is removed (stripping the dominant between-type variance), the residuals are scaled per group, then pooled, standardized, and reduced by a **single** PCA — one stable, drift-free, common frame across cell types, rather than 186 independent per-group PCAs on ~50 points. This surfaces the within-type (exercise/sex) structure the global UMAP compresses.

---

## 3. The gate result (the scientific finding)

**Method.** Per (tissue, cell type), multivariate η² — the fraction of the 768-d embedding's variance explained — for three factors, with a 1000-permutation p-value:
- **GROUP** — 5-level exercise dose (control / 1w / 2w / 4w / 8w)
- **TRAINED** — binary, any training vs control
- **SEX** — binary

Sex-chromosome genes were removed upstream, so the SEX signal is **autosomal** (real sexual dimorphism, not XIST/Y artifacts). **Reading caveat:** η² is *not* degrees-of-freedom-adjusted, so GROUP (4 df) is mechanically larger than the binary TRAINED/SEX (1 df). The fair effect-size comparison is **TRAINED vs SEX**.

**Overall (n = 186 tissue × cell-type):**

| factor | significant (p<0.05) | median η² |
|---|---|---|
| SEX | 99 / 186 (53%) | 0.043 |
| GROUP (5-lvl) | 52 / 186 (27%) | 0.092 *(df-inflated)* |
| TRAINED | 35 / 186 (18%) | **0.023** |

That trained median η² ≈ **0.023** is the honest size of the exercise effect at the embedding level: **real, but small and patchy.** Sex is the single most pervasive axis.

**Per tissue** (sig = p<0.05 count / #cell types; η² shown median/max):

| tissue | #ct | GROUP | TRAINED | SEX | read |
|---|---|---|---|---|---|
| blood | 14 | 11/14 · .219/.257 | **8/14 · .056/.132** | 10/14 · .062/.136 | **exercise hotspot** |
| skmvl | 15 | 9/15 · .135/.193 | **7/15 · .039/.099** | 6/15 · .032/.064 | **exercise hotspot** (exercise ≳ sex) |
| skmgn | 17 | 8/17 · .108/.189 | 7/17 · .036/.065 | 12/17 · .044/.076 | exercise moderate |
| lung | 27 | 9/27 · .102/.212 | 2/27 · .022/.040 | 11/27 · .035/.110 | **dose-only / weak** |
| hippoc | 15 | 8/15 · .114/.144 | 2/15 · .019/.052 | 1/15 · .018/.038 | **dose-only** |
| heart | 23 | 2/23 · .078/.165 | 1/23 · .022/.043 | 16/23 · .045/.212 | exercise-quiet, sex-present |
| kidney | 17 | 0/17 · .063/.094 | 1/17 · .015/.046 | **13/17 · .239/.493** | sex-dominated |
| liver | 6 | 0/6 · .064/.093 | 0/6 · .018/.037 | **6/6 · .304/.378** | sex-dominated, no exercise |
| watsc | 17 | 2/17 · .090/.165 | 4/17 · .038/.072 | **17/17 · .277/.780** | sex-dominated |
| cortex | 35 | 3/35 · .088/.138 | 3/35 · .019/.051 | 7/35 · .025/.063 | quiet |

**Four buckets:**
1. **Exercise hotspots — blood immune + skeletal muscle.** Blood: megakaryocytes (trained η² 0.132), basophils, classical/non-classical monocytes, ISG-T, NK. Vastus lateralis (skmvl): skeletal muscle fibers (0.099) and cells (0.092), endothelial, B cells, Schwann, myofibroblasts — here **trained ≳ sex**. Gastrocnemius echoes it. Biologically the right cells (immune trafficking + the muscle parenchyma itself).
2. **Dose-only — lung, hippocampus.** Many cell types significant for GROUP but almost none for TRAINED (lung 9 vs 2; hippoc 8 vs 2). Real structure *across the five dose levels* that the trained/untrained binary collapses away → **non-monotonic or late-emerging dose response.**
3. **Sex-dominated — kidney, liver, WAT (heart leans sex).** Sex swamps everything: WAT macrophages η² 0.78, luminal epithelial 0.65; kidney endothelial 0.49; liver 6/6 cell types. Exercise essentially absent. This is the known rat sexual dimorphism in these organs — it *validates* that the embeddings capture real physiology.
4. **Quiet — cortex** (and heart for exercise): little within-type signal of any kind.

**Top exercise responders (by trained η²):** blood Megakaryocytes 0.132, skmvl Skeletal muscle fibers 0.099, blood Basophils 0.098, skmvl Skeletal muscle cells 0.092, blood Classical monocytes 0.091, skmvl Schwann 0.091, skmvl B cells 0.079, skmvl Endothelial 0.078, blood ISG-T 0.075, blood Non-classical monocytes 0.075.

**Interpretation (revised — see §3b).** The embeddings separate cell types and tissues strongly and capture sex cleanly where it is real — so the substrate is sound. But the trained median η² ≈ 0.023 is **not** the biological size of the exercise effect — it is what a *global, variance-weighted* trace measure reports, and that measure structurally **dilutes** a low-dimensional signal (the exercise axis is low-variance, so it is swamped by the ~760 off-target dimensions in the denominator). A supervised re-measurement (§3b) shows exercise is in fact **concentrated and reliably detectable** (held-out AUC up to 0.91) in the same immune + muscle cells, a published method (Augur) independently confirms it, and external literature (§3c) corroborates the hotspot geography — though the per-gene *magnitude* stays modest (§3c). The GROUP-but-not-TRAINED pattern remains direct evidence we should **model dose**, not a binary contrast.

---

## 3b. Supervised re-measurement & cross-method corroboration (2026-06-15)

**Why re-measure.** The §3 gate's η² is `trace(between-group SS) / trace(total SS)` over all 768 dims — formally a PERMANOVA / `adonis` R² (Anderson 2001), and a *variance-weighted average* of the per-dimension η². That makes it a deliberately **conservative global screen**: a signal living in a low-dimensional (low-variance) subspace is diluted across the ~760 off-target dimensions in the denominator, and broad high-variance axes (sex) dominate. So the trained median η² ≈ 0.023 is a *lower bound*, not the effect size. Three less-conservative measures (`subspace_probe.py`, 1000-permutation null) test whether exercise is genuinely weak or merely diluted:

| measure (trained vs control) | median | max | reading |
|---|---|---|---|
| global trace η² (the gate) | 0.023 | 0.132 | conservative baseline |
| standardized trace η² (z-score dims first) | 0.024 | 0.090 | ≈ unchanged → **not** a variance-weighting artifact |
| per-dimension max η² (max-stat null) | 0.216 | 0.464 | individual dims carry signal (scan-inflated) |
| **supervised CV PLS-1, held-out AUC** | 0.593 | **0.910** | the real axis: cross-validated separability |
| supervised CV PLS-1, ordinal-dose Spearman | 0.069 | 0.654 | dose is recoverable in muscle |

**The supervised probe is decisive.** A cross-validated single-component PLS direction (the standard p≫n supervised projection) separates trained-vs-control out-of-fold at **AUC up to 0.91** (skmvl skeletal muscle fibers), 0.80–0.89 across blood immune + skeletal muscle. Multiple-testing makes the contrast with the gate stark: at **BH-FDR < 0.05** the global trace gate retains only **5/186** trained hits, the supervised probe retains **22** (AUC ≥ 0.70) — blood 7, skmvl 6, skmgn 5, lung 2, heart 1, kidney 1 (immune trafficking + muscle parenchyma/stroma, the biologically expected cells). The standardized η² ≈ global confirms the exercise axis is a *diagonal supervised direction*, invisible to any rotation-invariant trace ratio. Negative controls hold (cortex/kidney median AUC ≈ chance 0.50/0.45; sex positive control 93/186 FDR-significant), so this is not p≫n overfitting.

**Cross-method corroboration** (`run_augur.R`, `corroborate_summary.py`; see `EMBEDDING_DE_STANDARDS.md` for the methods context):
- **Method robustness.** Canonical **Augur** (Skinnider/Squair 2021 — the published cross-validated-RF standard for cell-type perturbation-responsiveness) reproduces our PLS-1 ranking at **Spearman r = 0.83** (n=186) and independently confirms **19/22 hotspots at AUC ≥ 0.70 (22/22 at ≥ 0.65)**; Augur's median AUC on the FDR-significant set is **0.82** vs 0.55 elsewhere; sex positive control max AUC **1.00** (liver, WAT). The finding is not an artifact of our specific linear method — a different, published, nonlinear method finds the same cells.
- **Representation control** (does GeneCompass beat a plain PCA? cf. "one PCA still rules them all"). The embedding beats a **PCA-50** baseline modestly and method-dependently — significant under the linear probe (median AUC 0.609 vs 0.562, p=0.001) but only a trend under Augur-RF (Δ+0.014, p=0.14) — and it **ties a full-gene probe** (0.609 vs 0.599, p=0.39). PCA-50 keeps the *top-variance* directions, exactly where exercise *isn't*, so it discards the signal the embedding and a full-gene probe both retain. Net: GeneCompass is a **faithful, compact** representation of the exercise axis, not a unique source of signal.

**Caveat.** A few skmgn immune types our linear PLS-1 rated high but Augur-RF rated only moderate (e.g. skmgn Monocytes 0.892 vs 0.682); where the two methods disagree, treat the call as softer. The muscle-parenchyma and blood hotspots, where all measures agree, are the solid ground.

**Bottom line.** Exercise is a **real, method-robust, concentrated** signal in blood-immune and skeletal-muscle cell types, and the embedding faithfully carries it. The §3 gate's "small/patchy" *detectability* was a property of the conservative trace measure — but per-gene *magnitude* genuinely is modest (same-data Vetr 2024, §3c), so the precise reading is **reliably detectable and concentrated, not large**. Proceed to per-cell-type DE (§4–5) with confidence on the hotspot set.

---

## 3c. External validation & cross-species benchmark (literature, 2026-06-15)

Two papers — read in full — anchor the §3b hotspots against published biology and the *same* MoTrPAC data. PDFs in `readings/` (`Gene_Impact_PA.pdf`, `sc_marathon.pdf`).

**A — Same-data disease-genetics map: Vetr, Gay, MoTrPAC Study Group & Montgomery 2024, _Nat Commun_ 15:3346** (doi:10.1038/s41467-024-45966-w). Uses the *identical* MoTrPAC PASS1B EET rat data we deconvolve (F344, both sexes, 1/2/4/8 wk, 48h post-bout, 47–50 rats/tissue, 8wk = adapted state; 15 tissues). Three things it gives us:
- **The DE recipe template** (matches the `EMBEDDING_DE_STANDARDS.md` pseudobulk prescription): per sex × tissue **DESeq2 LRT** (`nbinomLRT`) over the training time course with RNA-seq technical covariates (RIN, 5′-3′ bias, globin %, PCR-dup %); male/female combined by **Fisher**; multiple-testing by **IHW with tissue covariate**; per-timepoint contrasts vs sex-matched sedentary; sex-consistent states via `repfdr` (F1_M1 … F-1_M-1). Our per-cell-type DE (§5.1) should mirror it.
- **Independent corroboration of the hotspot geography.** Exercise DE is *extremely* tissue-specific — **78% of DE genes are DE in exactly one tissue, 95% in ≤2**; the only strongly-overlapping pair is gastrocnemius + vastus lateralis (Jaccard ≈ 0.21), echoing our skmgn+skmvl muscle hotspot and the "not a global axis" finding. The **strongest heritability enrichment is in blood, "especially traits corresponding to densities of immune cells"** (spleen carries 51% of significant immune/blood enrichments) — our #1 (blood-immune) hotspot from an orthogonal genetics angle. Per-tissue gene benchmarks for our DE: blood cholesterol program (NDUFA13, FADS2, PNKD, AAMP, OGDH), SKM-VL TMBIM1/ATP6V1G2, asthma-blood (BAG6, CCNF, CRAT, PTPA, FAM89B), LDLR (cortex/hippoc/both muscles), FOXP3 (heart/spleen).
- **The cross-species blueprint — and our novelty.** Their whole pipeline *is* grant Aim 2's cross-species step: rat exercise DE → human ortholog (94.5% map) → GTEx eQTL / 114 GWAS / S-PrediXcan / Open Targets → **5,523 trait-tissue-gene triplets**. It is all *bulk-tissue*; **our cell-type-resolved DE is the natural extension** that adds resolution their map lacks. Code + data: github.com/NikVetr/MoTrPAC_Complex_Traits (Zenodo 10211801).

**The magnitude reconciliation (refines §3b).** Vetr — *same data* — repeatedly finds the effect **small in magnitude**: trait enrichments within 0 ± 0.3 log-odds (max ≈ 7.5% on the probability scale), and per-gene only ~1 gene/tissue exceeds 2 SD of *genetic* and ~52/tissue exceed 2 SD of *phenotypic* expression variance — i.e. **most exercise DE sits within normal inter-individual variation.** This is *not* in tension with our "AUC up to 0.91": the two measure different things. Our AUC is cross-validated **separability** (the signal is *reliably present and concentrated*); Vetr's is per-gene **magnitude** and disease-genetics **enrichment** (both *modest but real*). The reviewer-proof joint claim: **exercise is a reliably detectable, cell-type-concentrated axis composed of many small per-gene changes, with subtle-but-real disease enrichment in blood/immune and muscle.** Read §3b's "strong" as "reliably detectable," not "large-magnitude."

**B — Human single-cell immune validation: Yu et al. 2023, _iScience_ 26:106532** (doi:10.1016/j.isci.2023.106532). Longitudinal human PBMC scRNA-seq (275k cells) after **acute** exercise (marathon n=3 / CPX n=3, male). It validates *which* blood cells are exercise-responsive and *what* programs:
- The most exercise-dynamic PBMC populations are **monocytes** (accumulate, peak 1h, with an S100A8/A9-hi ↔ HLA-DPA1/DPB1-hi state switch) and **effector/cytotoxic T cells** (selectively reduced), plus defined platelet (PPBP/PF4) clusters — mapping onto our top blood responders (classical/non-classical monocytes, megakaryocytes/platelets, ISG-T). Gene programs to check in our blood DE: **CXCR4, S100A8/A9**, cytotoxicity (PRF1, GNLY, NKG7, GZMA/B/H/K), naive (CCR7, TCF7, LEF1, SELL), ISG20 (→ ISG-T), monocyte cardiac-risk (VCAN, RETN, ACSL1, NAMPT).
- **Caveats before leaning on it.** Acute single-bout / human / male / n=3 vs our chronic 1–8 wk / rat / both-sexes / 48h-post (the *adapted* state). So it validates **cell types and gene programs, not effect direction/dynamics** — acute open-window immunodepression ≠ chronic adaptation. It also vividly underscores our **composition confound**: exercise moves cell *fractions* massively (T↓, mono↑), so per-cell-type DE on `Z` (expression) must be read against `θ` (fractions) — the Milo point in `EMBEDDING_DE_STANDARDS.md`.

**Three-way convergence.** Our deconv flags blood **Basophils** as a top responder (trained η² 0.098); Vetr finds basophil/eosinophil counts genetically correlate with asthma and exercise (Sastre 2013, "basophils, a new player" in exercise bronchoconstriction); Yu supplies the single-cell immune dynamics — deconv + disease-genetics + single-cell all landing on the same cells.

---

## 4. Implications for Aim 2 — the pivot

The §3b re-measurement and §3c external validation settle the gate's question: exercise is **recoverable and reliably detectable** in the hotspot cell types (held-out AUC up to 0.91, method-robust via Augur, hotspot geography corroborated by same-data Vetr 2024), but it does **not** fall out of the raw embedding as a *global* axis — it lives in a low-dimensional supervised direction, and its per-gene magnitude is modest (§3c). So: don't gate downstream work on a global embedding signal; instead proceed with confidence to **per-cell-type differential expression on the deconvolved `Z`** for the hotspot set, with three design rules the analysis hands us directly:

1. **Focus the hotspots** — blood immune subsets and skeletal-muscle parenchyma/stroma/endothelium. Expect little to nothing in kidney/liver/cortex/heart.
2. **Control for sex** — it is the dominant axis in 3–4 tissues and will otherwise masquerade as exercise.
3. **Model dose ordinally** (1/2/4/8 wk), not trained-vs-control — lung and hippocampus show the binary collapse discards real signal.

---

## 4a. Per-cell-type DE — RESULTS, positive-control verdict & reference reconsideration (2026-06-22)

The §4 pivot is executed. Implementation: `deconvolution/R/run_pseudobulk_de.R` (driven by Stage-10 orchestrator
`pipeline/run_stage10.py`: DE → positive-control comparison). Exhaustive Vetr-faithful design — per (tissue × cell
type), **limma-trend on log2-CPM of the continuous BayesPrism `Z`** (NOT DESeq2-NB: `Z` is posterior expected
count-mass, rare-in-tissue types mostly <1, verified on HEART/BLOOD/LUNG — integer rounding would zero rare-type and
~¼–½ of mid-abundance immune signal), with combined `~ sex * factor(week)` (omnibus dose F + per-timepoint contrasts
+ sex×dose interaction) + ordinal slope; per-sex `~ factor(week)` → signed-z at 8 wk → **Fisher** sex-combine;
**global IHW~tissue**; **repfdr** 8-wk sex-consistency; composition-confound check. Full gene coverage (only all-zero
genes dropped, logged). Run: 10 tissues, **186/186 blocks ok, 2,225,006 Fisher tests, IHW + repfdr converged**.

### Positive-control verdict (pre-registered)
The spec `POSCTRL_PREREG.md` / `reference/posctrl_prereg.tsv` (105 gene×target rows, frozen **before** any per-gene
result) was scored by `compare_posctrl.py` through the fixed miss-ladder (coverage → power → confound → biology):
- **Immune cell types recover known biology — the result.** Yu 2023 cytotoxicity program enriched in blood **NK
  (4.3×, binom p=0.0025), Memory-T (3.5×, p=0.017), ISG-T (3.3×, p=0.021)**; naive program in Memory-T (5.3×,
  p=0.01). Vetr 2024 blood genes (Fads2, Aamp, Bag6, Fam89b) recover in **ISG-expressing T cells**; Atp6v1g2 in
  SKMVL muscle fibroblasts (female); Ccnf in blood NK. This is the hotspot region the gate/Augur flagged (§3, §3b).
- **The direction-anchored Tier-A controls (canonical mito/heat-shock) did NOT recover in the dominant parenchyma**
  (4/45; Sod2/Mef2c/Slc2a4/Hspa1b/Hsp90aa1 flat in muscle cells/cardiomyocytes/hepatocytes). Per the pre-registration
  this was flagged as a possible pipeline problem and **investigated, not spun**.

### Resolution — the pipeline is sound; the Tier-A controls were mis-specified
`deconvolution/diagnose_parenchyma.py` + `validate_parenchyma_dataanchored.py` settle it three ways:
1. **Parenchyma `Z` faithfully tracks bulk dose genome-wide:** corr(bulk week-slope, parenchyma-`Z` week-slope) =
   0.68 (SKMGN) / 0.77 (SKMVL) / 0.71 (HEART) / 0.81 (LIVER), p≈0, shrinkage ≈1 (not compressed/prior-regressed).
2. **The controls are flat or down in the actual rat bulk transcript:** at 8 wk vs control Hspa1b = −0.96 (SKMGN)
   and down in all four tissues; Sod2/Mef2c ≈0; only Slc2a4/GLUT4 modestly up (+0.12–0.14). The MoTrPAC mito/HSP
   exercise signature is largely protein/PTM-level — the *transcripts* are weak. So the controls were mis-specified
   for a transcript-dose test; the DE faithfully reflected their bulk flatness (incl. the HSP down-direction).
3. **Data-anchored positive control (the correct one):** of genes that *do* move in the matched bulk (8 wk BH<.05,
   |log2FC|≥.25), the parenchyma DE recovers **cardiomyocytes 83%**, skeletal-muscle ~50% direction-concordant
   (per-block BH); LIVER has 0 bulk movers (liver's transcript dose signal is genuinely weak). **Pipeline validated.**

### Reference reconsideration (references are a secondary factor)
References recover the parenchyma fraction near-perfectly where the type is resolved (θ r≈0.99 holdout). Three
separable issues, each tested:
- **SKMVL was over-split** ("Skeletal muscle cells" + "Skeletal muscle fibers"). A new `muscle` label-scheme in
  `build_reference.py` merges them; re-deconvolving SKMVL **improves** recovery of bulk dose movers from 50%/41%
  (split) to **57%** (merged). **→ Adopt the muscle merge for SKMVL.** (Merged outputs in `*_merged/` dirs.)
- **Cortex was over-split AND un-merged** (12 collinear neuron labels) — asymmetric with hippocampus (brain-merged).
  Rebuilding gene-rich + brain-merged fixes the inconsistency, but cortex has **0 bulk exercise movers** (quiet
  tissue) → zero science impact. Adopt for correctness only.
- **Heart CM reference has holdout-only validation** (no cross-dataset CM check; v1 cross used a CM-less atlas).
  Documented limitation — the CM deconvolution is nonetheless externally validated against the MoTrPAC bulk (83%
  bulk-mover recovery, bulk-`Z` r=0.71); not pursued (needs a 2nd rat-heart-CM dataset).

**Now canonical in the pipeline (no ad-hoc rebuild).** The collinear-parenchyma merges — brain for
cortex + hippocampus, muscle for SKMVL — are built directly by `build_references_v2.sh` via
`--label-scheme` (the `muscle` scheme added to `build_reference.py`); the per-tissue canonical reference
(merged where applicable) is recorded in `reference/canonical_references.tsv`, so a standard pipeline run
produces the merged references without the manual rebuild step.

**Adopted into production (2026-06-23).** The merges were carried all the way through the embedding
stack — re-deconvolve → re-build pseudo-cells → **re-embed (Stage 9)** → re-run gate/probe/Augur →
re-derive hotspots → re-DE — so `pred_z`, the 768-d embeddings, the detection layer, the hotspots, and
the DE are all internally consistent on the merged references (split versions backed up). New production:
**178 blocks (was 186; cortex 35→28, SKMVL 15→14), 18 hotspots.** SKMVL's two collinear muscle blocks are
now one **"Skeletal muscle"** (262 dose-sig genes — the top SKMVL responder); cortex remains quiet (≤2).
Conclusions are unchanged: the positive-control verdict still has immune programs recovering (Yu/Vetr) and
the canonical mito/HSP controls `not_significant` (flat in rat bulk). The frozen pre-reg's SKMVL muscle
target was remapped split→"Skeletal muscle" (a deterministic label change from the merge; genes,
directions, thresholds untouched). NB: GeneCompass takes single cells, but these merges consolidate only
*collinear synonym-fragments of one population* (not distinct cell states), so each merged pseudo-cell is a
coherent single-population profile, not a chimera — a better embedding input than the noisy split estimates.

### Reporting policy (carry into all downstream use)
Down-weight absolute per-gene DE on the **dominant parenchyma** per tissue (`dominant_celltype_flags.tsv`); prefer
relative / 8-wk contrasts there. **Lean the exercise story on the immune + mid-abundance stromal/endothelial cell
types**, which validate against Yu/Vetr and the gate/Augur. The cell-type-resolved DE is the Aim-2 deliverable; the
parenchyma DE is faithful-to-bulk, but its canonical transcript controls are genuinely weak in rat.

**Artifacts:** `run_pseudobulk_de.{R,sh}`, `pipeline/run_stage10.py`, `POSCTRL_PREREG.md` + `build_posctrl_prereg.py`
+ `reference/posctrl_prereg.tsv`, `compare_posctrl.py`, `diagnose_parenchyma.py`, `validate_parenchyma_dataanchored.py`,
`build_reference.py` (muscle scheme); outputs under `genecompass_input/pseudobulk_de/` (+ `_merged/` for the
SKMVL/cortex re-deconvolutions). `DOWNSTREAM_BUILD_PLAN.md` = the GRN/perturbation/cross-species build map.

---

## 5. Next stages

1. **Per-cell-type DE on `Z` — ✅ DONE 2026-06-22 (see §4a for results, the pre-registered positive-control verdict, the parenchyma diagnostic, and the reference reconsideration).** (field-standard route — see `EMBEDDING_DE_STANDARDS.md`). Pseudobulk-style model per (tissue, cell type): `expression ~ dose(ordinal week) + sex (+ interaction)` on the deconvolved `Z`. Each pseudo-cell is already a sample-level profile, so this is sample-level / pseudobulk — the approach Squair 2021 shows controls false discoveries — **not** a single-cell-level test and **not** DE "on" the embedding. DESeq2 / edgeR / limma-voom or an equivalent empirical-Bayes / permutation test. Start with the **22 FDR-significant hotspots** (blood immune + skeletal muscle). **Mirror the same-data Vetr 2024 recipe** (§3c) for comparability — DESeq2 LRT over the ordinal time course, sexes combined by Fisher (or sex modeled explicitly), IHW for multiple testing — and **self-check against published positive controls**: Vetr's per-tissue gene lists (blood cholesterol/asthma programs, SKM-VL TMBIM1/ATP6V1G2) and Yu 2023's immune programs (CXCR4, S100A8/A9, cytotoxicity/naive sets), reading expression (`Z`) changes alongside fraction (`θ`) changes (the composition confound, §3c). Output: ranked, sex-adjusted, dose-shaped exercise-responsive genes per cell type. This is the deliverable that turns the corroborated AUC signal into biology.
2. **GRN / perturbation routes (after DE).** Caveats from the connector work: pseudo-cell GRN/perturbation is only valid for **abundant** cell types (rare ones sit at the reference prior, no exercise signal); n ≈ 50 samples/cell type is too thin for data-driven GRN (DeepSEM) → default to the **model-driven in-silico-perturbation** GRN route in GeneCompass.
3. **Dose modeling.** Evaluate **CPA** (compositional perturbation autoencoder) repurposed for exercise dose (1/2/4/8 wk) on the hotspot cell types.
4. **Cross-species (grant aim).** **Vetr 2024 (§3c) is the blueprint** — rat exercise DE → human ortholog → GTEx eQTL / GWAS / S-PrediXcan / Open Targets → trait-tissue-gene triplets; our contribution is the **cell-type-resolved** extension (push per-cell-type DE through the same pipeline). Still needs a human single-cell side for direct cross-species cell-type matching — not free; scope separately.
5. **Secondary / hardening.**
   - Multi-method θ cross-check (omnideconv: MuSiC/DWLS/SCDC/Bisque) on the production tissues, as done for WAT, to confirm the deconvolution fractions feeding the hotspots.
   - **Lung caveat:** lung is the weakest cross-dataset deconvolution (~0.73); treat any lung exercise claim cautiously.
   - Viewer v2: real per-cell-type **local PCA** projection in Focus (from the 768-d embeddings) so the within-type exercise/sex axis shows on its own canvas rather than the zoomed global coords.
6. **Risks to keep in view.**
   - **Activity confound** (the key threat): bulk fraction shifts can reflect expression drift, not true cell-fraction change → frame exercise results **relative / differential**, not as absolute fractions.
   - **Quiet tissues (cortex, heart-for-exercise):** is the absence biology, or reference/abundance limitation? Note before concluding "no effect." See `reference_dominant_parenchyma_cross_failure` (most-abundant parenchyma can collapse in cross-validation).

---

## 6. Artifacts & reproduce

**Outputs** (under `data/deconvolution/genecompass_input/`, gitignored):
- `<tissue>/embeddings/cell_embeddings.npy` + `<tissue>/dataset/` — per-tissue pseudo-cell embeddings (768-d) and tokenized datasets.
- `pheno_merge_test.tsv` — the gate (186 rows); `subspace_probe.tsv`, `pca_control.tsv`, `augur_results.tsv`, `corroboration_merged.tsv` — the §3b re-measurement + corroboration; `augur_input/<tissue>/` — Augur inputs.
- `umap/umap_coords.tsv`, `umap/viewer.html`, `umap/viewer_data.json` — UMAP + interactive viewer.

**Committed code** (`deconvolution/`):
- `pheno_merge_test.py` — the Aim-2 gate (global trace-η²).
- `subspace_probe.py` — the §3b re-measurement (standardized η², per-dim max-stat scan, supervised PLS-1 CV probe; 1000-perm null).
- `augur_prep.py` + `run_augur.R` + `corroborate_summary.py` — canonical Augur (R, neurorestore 1.0.3) + PCA-vs-embedding control + the merge/verdict.
- `build_umap_viewer.py` + `umap_viewer_template.html` — extract data → self-contained viewer.

**Regenerate:**
```bash
source /depot/reese18/apps/motrpac-env/bin/activate
python deconvolution/pheno_merge_test.py        # → pheno_merge_test.tsv  (auto-discovers all tissues with embeddings)
python deconvolution/build_umap_viewer.py       # → umap/viewer.html + viewer_data.json
```
