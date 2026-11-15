# Audit findings register (2026-11-11)

> **2026-11-15: a dated record.** The register describes the default of 2026-11-11. What was done about each
> finding, and the results of the rerun that followed (refs5, the default since 2026-11-15), are in
> [`remediation/README.md`](remediation/README.md). The register is kept as written.

The main findings of the nine audit parts in one place. Each row points to the part that holds the detail and the
evidence. The parts list more items than this register does, under "Deviations, known defects and caveats".

**How to read a row.**
- **Class A** changes how a reported result should be read. **Class B** is a method or record gap whose effect on
  results is unmeasured or limited. **Class C** is documentation, labelling or hygiene.
- **Checked** says how far the finding was verified.
  - "re-opened" means the cited lines or tables were opened again after the part's reader reported the finding,
    and they say what the row says.
  - "as read" means the finding is the reader's, with its `path:line` citations, and was not opened a second
    time.
- A count described in a part as "auditor's computation" or "computed here" was made once, read-only, from an
  on-disk table. Re-derive it before quoting it.
- Nothing listed here was fixed by the audit. No code, configuration or result was changed.

## The findings that matter most

These bear on headline statements in the README, the tracker or the manuscript.

1. **The gate's hotspot count is within sampling error of its own test, and two hotspots are thin** (Part 6).
   - LUNG endothelial and mesothelial cells miss the cut at p = 0.0037 and 0.0039 against about 0.0033, with a
     Monte Carlo standard error near 0.0005. The "9 against 11 hotspots" difference between the 16- and 14-tissue
     arms rests on that plus the larger test family.
   - HYPOTH oligodendrocyte precursor cells and LUNG vascular smooth muscle cells show no signal when the same
     probe is run in gene space (AUC 0.499 and 0.532). Both also lose their dose trend in human space (Part 8).
   - HYPOTH's reference is the one that was never re-annotated (Part 4).
2. **"9 of 9 hotspots preserved" in human space is an uncorrected per-block rule on non-independent tests**
   (Part 8). The rat and human tests share cells, labels, fold seeds and permutation seeds. By the reader's
   arithmetic, Benjamini-Hochberg over the 151 analysable blocks would pass 7 of 9.
3. **Differential expression: three statements need qualifying** (Part 7).
   - "53,084 dose-significant genes" counts gene x block pairs. 9,277 of them are in 45 blocks whose mean
     fraction is below 0.01.
   - TESTES and OVARY can never have a dose-significant gene, because the two sexes' p-values are combined and one
     is missing.
   - Of 5,126 "sex-consistent" calls, 144 are supported by both per-sex tests.
4. **The positive-control red flag fired** (Part 7). The pre-stated sanity subset recovered 5 of 34 controls, which
   the specification defines as a pipeline red flag. It was resolved by a later check outside the registered
   ladder.
5. **Cell-type-specific expression is validated more narrowly than the documents say** (Part 5).
   - LIVER has no purity sweep on its current reference. The quoted 0.998 / 0.949 is a May pilot on a reference
     that still held the Visium samples, and 0.949 is below the 0.95 bar.
   - The sweeps test only the dominant cell type, within one study. No hotspot cell type has an expression-accuracy
     test.
6. **SimBu mRNA-bias battery** (Part 5; `deconvolution/SIMBU_BIAS_REFS4ALL_RESULTS.md`). On the current references
   "MuSiC and SCDC correct the bias" is not reproduced, and the "no bias" arm keeps the references' native
   mRNA-content differences.
7. **Eleven genes are excluded from every deconvolution as "ribosomal" that are not** (Parts 4 and 5): the ten
   S6-kinase genes, including Rps6kb1, and Rps19bp1. They are absent from every expression table and pseudo-cell.
8. **Model validation uses no held-out data, but several documents say "held-out"** (Part 3). The V1 cells are
   drawn from the training corpus and their labels are automated marker labels.
9. **The April profile loads the v2 weights** (Parts 3 and 6). An "April" run of Stages 9, 12 or 13 made since
   2026-11-01 without `--model-dir` would use v2's weights with April's target sum. Whether one was made is not
   recorded.
10. **The GRN's trained-minus-control difference is not always computed on a shared gene set** (Part 9). Genes
    present in one arm only are differenced against zero and carry a large share of hub strength in two of the six
    default networks. The decoy null shares the defect, so its negative verdict stands.
11. **Module A's target criterion does not validate targets** (Part 9). Its curated set was written after the
    first run was seen, the retrained model fails it, and the results file still reads "ALL PASS".
12. **The manuscript overstates what the rat-specific tokens received** (Parts 2 and 3). It says their gene-family
    and regulatory-network prior rows "are filled"; the v2 initialisation log shows 0 of 2,474 family rows and
    36 of 2,474 network rows non-zero.
13. **Corpus construction differs from its description in two ways** (Part 1).
    - 79 rat single-cell GEO series that the LLM step confirmed never entered cell QC, through a catalog merge
      that keeps the bulk row.
    - No 15% mitochondrial cut is applied, although the configuration and the manuscript state one, and the
      within-file z-score that is applied cannot see mitochondrial genes in files keyed by Ensembl ID.

## Part 1. Stages 1-3: corpus, gene universe, orthologs

| # | Class | Finding | Evidence | Checked |
|---|---|---|---|---|
| 1.1 | A | 108 GEO series were returned by both the single-cell and the bulk query. The catalog merge keeps the later (bulk) row and Stage 2 skips non-single-cell types, so 79 LLM-confirmed rat single-cell series never entered cell QC; 35 have a usable matrix on disk | `pipeline/01_data_harvesting/combine_data_sources.py:257`; `pipeline/02_gene_universe/preprocess_training_matrices.py:1269-1272` | code re-opened; counts as read |
| 1.2 | A | `max_mito_fraction: 0.15` is read by no Stage 2 code. The only mitochondrial rule is a within-file z-score of +/-3. The GeneCompass paper, which the QC is said to copy exactly, applies the 15% cut | `config/pipeline_config.yaml:214`; `preprocess_training_matrices.py:839-845`; paper Methods, PDF page 12 | re-opened |
| 1.3 | A | That z-score is inert where genes are keyed by Ensembl ID, because mitochondrial genes are found by a symbol pattern: 368 of 555 kept files, 4.6 million cells, 68% of the corpus (inferred from code and the preprocessing report, not measured on the files) | `preprocess_training_matrices.py:130-134,575,1142` | pattern re-opened; counts as read |
| 1.4 | B | The LLM study step is misreported. Config and README say Claude Haiku; the stored output holds 1,336 records from `claude-sonnet-4-5-20250929`, 1,216 from `claude-haiku-4-5` and 10 from `claude-3-5-haiku-20241022`. Its flags vetoed 153 single-cell-typed studies; the exclusions were never reviewed and the run command is not recorded | `config/pipeline_config.yaml:158`; `data/catalog/llm_study_analysis.json` | model counts re-opened |
| 1.5 | B | Gene IDs from older Ensembl releases are accepted as the same gene whenever the ID exists in release 115, although the project's own OMIX767 script says such IDs can be reassigned. Exposure in the 80 GEO studies is unmeasured | `pipeline/02_gene_universe/build_gene_universe.py:270-275`; `pipeline/02_gene_universe/qc_external_mtx.py:10-15` | as read |
| 1.6 | B | 247 of 866 first-corpus QC files were md5 copies. The cause, a file walk that visits `_RAW` twice, is still in the code. v2 removes the copies, 48 non-rat files and 27 log-normalised files only through `config/corpus_v2_plan.tsv`, whose builder is not committed | `preprocess_training_matrices.py:711-724,752-753` | as read |
| 1.7 | B | Stage 2 does not reproduce from raw data; the vocabulary was fixed on the uncleaned corpus; the singleton rule counts bulk and array tables | `docs/repro_stages1_7/REPRODUCTION.md:14-17,94-100` | as read |
| 1.8 | B | The three-standard-deviation rule is applied to total counts and mitochondrial fraction. The paper applies it to the number of expressed genes | `preprocess_training_matrices.py:839-845`; paper PDF page 12 | re-opened |
| 1.9 | C | The canonical Stage 2 manifest and the manuscript still say Ensembl 113 / mRatBN7.2; the files are 115 / GRCr8 | `manuscript/motrpac_genecompass.tex:273,386,1252` | as read |
| 1.10 | C | Orthology calls are unfiltered (no confidence flag, no identity floor); the Stage 3 manifest's commit predates the token change it describes; a "flagged non-coding" list is always empty through a case mismatch; ArrayExpress contributes nothing | Part 1, Stage 3 deviations | as read |

## Part 2. Stages 4-6: medians, tokenization, priors

| # | Class | Finding | Evidence | Checked |
|---|---|---|---|---|
| 2.1 | A | The manuscript says the rat-specific (T4) gene-family and GRN prior rows "are filled". The v2 initialisation log: 0 of 2,474 family rows and 36 of 2,474 GRN rows non-zero. The family builder cannot produce a rat-token vector | `logs/init_v2_11910364.err:12-13`; `manuscript/motrpac_genecompass.tex:294-296`; `pipeline/06_prior_knowledge/build_family_embedding.py:486-487` | log and manuscript re-opened |
| 2.2 | B | The Stage 5 orchestrator describes `normalize_total(10,000)` then `log1p`. The code divides raw counts by the hybrid median and takes log2(1 + x), with no depth normalisation. Stage 4 says its CP10K step "matches GeneCompass", but upstream has that line commented out | `pipeline/run_stage5.py:22-30`; `pipeline/05_tokenization/tokenize_corpus.py:338-342`; `vendor/GeneCompass/preprocess/preprocess.py:157,160` | re-opened |
| 2.3 | B | The human and mouse training sets are called "0.5M" and linked to upstream's 0.5M row. They hold 5,000,000 and 4,969,697 cells. The mix is 33,138,703 rows, with each human cell repeated about 3 times and each rat cell about 2 times per phase. One of 165 mouse shards is missing and `state.json` was edited by hand | `data/training/mixed_species_dataset_v2/provenance_build_mixed_species_dataset.json:55-70`; `scripts/stage_genecompass_assets.sh:15-24` | counts re-opened |
| 2.4 | B | "Scheme B" medians (human median where one exists, otherwise rat median divided by 2.0527) are defined only in a commit message and an uncommitted scratch file | commit `ab5843a`; `tmp/data_audit/median_check/check_11893348.out:3-8` | as read |
| 2.5 | B | Promoter builder (used by April, not by v2): 10 of 45 BioMart batches returned nothing, so 5,000 genes lost their start site and the incomplete result is cached; the 2,500-base window is truncated to 512 tokens, about 515 bases; it loads stock `zhihan1996/DNA_bert_6` while its comment claims GeneCompass's fine-tuned model | `pipeline/06_prior_knowledge/build_promoter_embedding.py:274-277,322-327,545-553,606-612` | docstring re-opened; rest as read |
| 2.6 | B | 991 universe genes, 613 of them new tokens, have no median and are never tokenized. Co-expression vectors were built on the uncleaned corpus | Part 2, Stage 4 and 6 deviations | as read |
| 2.7 | C | The Stage 4 and 5 manifests carry no git commit and label Ensembl "113"; the legacy initialisation path still reproduces the old zero-prior defect; the committed April config no longer describes April's priors; Stage 6 outputs have no overwrite guard | `finetune/genecompass/rat_model_init.py:246-248,498`; `finetune/genecompass/configs/rat_finetune.yaml:71-73` | as read |
| 2.8 | fixed | The earlier defect that gave all 1,351 T3a tokens zero prior vectors is fixed for v2: 2,243 of 2,243 T3 rows are non-zero in all four priors, and the loader now raises on a zero T3 row | `logs/init_v2_11910364.err:10-13`; `finetune/genecompass/rat_load_prior_embedding.py:158-161,190-192` | log re-opened |

## Part 3. Stage 7: model, training, validation

| # | Class | Finding | Evidence | Checked |
|---|---|---|---|---|
| 3.1 | A | The April profile resolves to v2's weights. The default config pins `deconvolution.model_checkpoint`, that key has precedence, and the April profile overrides only `genecompass_model_dir` | `config/pipeline_config.yaml:364`; `deconvolution/model_checkpoint.py:33-38`; `config/profiles/april.yaml:21` | code re-opened |
| 3.2 | A | Nothing in Stage 7 is held out, by a recorded decision. Several documents nevertheless call the V1 cells "held-out". They are sampled from the training corpus, and their labels are automated PanglaoDB cluster labels | `docs/RETRAIN_V2_PREREGISTRATION.md:13-17`; `manuscript/motrpac_genecompass.tex:361,442`; `docs/OPEN_FRONTIER.md:122`; `analysis/aim1_validation/analyze.py:99,314,318` | pre-registration and manuscript re-opened |
| 3.3 | B | v2 was adopted on C1 and C2, both in-sample loss checks. Later comparisons are mixed: April is better on C4a, v2 has 13 hotspots to April's 16 on the July references, and v2 fails Module A criterion 2 | `docs/RETRAIN_V2_RESULTS.md:59`; `data/results/retrain_v2/c4/c4_summary.json:56,102`; `data/results/retrain_v2/c5/module_a/v2/report.md:9` | Module A re-opened |
| 3.4 | B | The pre-registration's first commit already contains smoke-test losses, so the repository cannot show that the 10% tolerance was set before any loss was seen. All C1-C5 result files are gitignored and absent from the deposit checksum table | `docs/RETRAIN_V2_PREREGISTRATION.md:98-104` | as read |
| 3.5 | B | The v2 initial checkpoint was built without a seed and cannot be regenerated. 53 and 35 of the logged steps in the two phases show an infinite gradient norm; only one step in 50 is logged. The v2 run folders hold no launch record | `finetune/genecompass/rat_model_init.py:511-519` | as read |
| 3.6 | B | C3 uses the July reference labels and C5 is frozen on the July references; neither was recomputed on the current default. No test of generalisation to an unseen rat study exists. No ablation isolates the priors, the mouse tokens or the two-phase schedule | Part 3, 7d and 7e deviations | as read |
| 3.7 | C | Config keys that do nothing (`t3_warm_start`, `t3_noise_std`, `species_init`); `--force` parsed and ignored; `forgetting_analysis.py` was run for April only and does not match training conditions; stale text in three places | `finetune/genecompass/configs/rat_finetune_v2.yaml:75-77`; `pipeline/run_stage7.py:298-299` | as read |
| 3.8 | confirmed | The vendored GeneCompass code is unmodified upstream at `66ab82f`. The model has 136,229,358 parameters. Phase 1 ran 517,793 steps (loss 0.713 to 0.667) and phase 2 591,763 steps (0.668 to 0.658). The earlier sampler defect, in which each phase saw a quarter of the data, is fixed | `finetune/genecompass/patches/transformers_compat.py:36-52` | as read |

## Part 4. Stage 8, part 1: references and bulk

| # | Class | Finding | Evidence | Checked |
|---|---|---|---|---|
| 4.1 | A | HYPOTH still uses the per-sample PanglaoDB + ScType + LLM labels that the project's own label audit discredited. It was never re-annotated. It carries a generic "Neurons" type beside the glutamatergic and GABAergic ones, and rests on one library. One of the nine hotspots is a HYPOTH cell type | `deconvolution/tissue_references.yaml:311-335`; `deconvolution/REFERENCE_RELABEL_PREREG.md:174-208` | configuration re-opened |
| 4.2 | A | The 160 manual label decisions of the joint re-annotation, and their literature check, were made by an AI assistant. The citations in the check are unverified and no expert has read the labels | `deconvolution/JOINT_REANNOTATION_RESULTS.md:173-176`; `deconvolution/JOINT_REANNOTATION_LITCHECK.md:26-29` | as read |
| 4.3 | A | Eleven non-ribosomal genes are excluded as "ribosomal" in every tissue: Rps6ka1-6, Rps6kb1, Rps6kb2, Rps6kc1, Rps6kl1 and Rps19bp1. No script builds the list | `deconvolution/reference/rat_exclude_genes.tsv:12,16,18,22,26,28,60,86,117,119,198`; `deconvolution/R/run_deconvolution.R:54-56` | re-opened |
| 4.4 | B | The reference QC gate does less than the manuscript says. Developmental and mixed-modality samples only warn; the depth check is not in the pre-deconvolution gate; samples absent from the corpus inventory are skipped, so the gate tests nothing for BLOOD, SPLEEN and OVARY | `deconvolution/reference_qc.py:63-77`; `pipeline/run_deconv_all.py:126-128`; `manuscript/motrpac_genecompass.tex:1295` | code re-opened |
| 4.5 | B | Bulk preparation excludes no sample; the consortium's outlier list is never read. The target annotation is called Ensembl 113 in the script, the liftover document and the manuscript. The data package reports version 2.0.0 while the submodule is tagged v2.1.0. 27% of bulk IDs are dropped by the liftover; the share of counts lost is unmeasured | `deconvolution/R/prepare_motrpac_bulk.R:2-3,164`; `vendor/MotrpacRatTraining6moData/DESCRIPTION:5` | script and version re-opened |
| 4.6 | B | BLOOD composite: the erythroid profile is 90 cells from one adjuvant-treated library. Check 6's six correlations above 0.9 are three independent quantities, because the PBMC types are mixed at fixed ratios. Erythroid cells are under-estimated by 0.24 on simulated mixtures | `deconvolution/build_blood_composite.py:192-198`; `data/deconvolution/blood_option_b/checks.json` | code re-opened |
| 4.7 | B | HIPPOC is still deconvolved with the reference diagnosed as an ambient-RNA sink, after both SoupX attempts failed | `deconvolution/REFERENCE_RELABEL_PREREG.md:484-518` | as read |
| 4.8 | B | Reference builds are single executions that were never reproduced: 7 of 15 have no build provenance and 6 were built from an uncommitted working tree | `deconvolution/REFS4ALL_REBUILD_PREREG.md:16` | as read |
| 4.9 | B | TESTES pools an EDS-treated arm with the control and the validator does not flag it. A registered HEART label merge was never applied. No reference matches the bulk's strain, and 9 of 15 references rest on one or two libraries | `deconvolution/tissue_references.yaml:433`; `deconvolution/REFERENCE_LABEL_AUDIT.md:150` | as read |
| 4.10 | C | Stale committed manifests; the LUNG builder hard-codes its samples; the script that scored the CDSeq components is not in the repository | `deconvolution/reference/canonical_references.tsv:2-11`; `deconvolution/build_lung_pooled.py:25-28` | as read |

## Part 5. Stage 8, part 2: BayesPrism, benchmarks, pseudo-cells

| # | Class | Finding | Evidence | Checked |
|---|---|---|---|---|
| 5.1 | A | Expression validation is narrower than claimed. LIVER has no sweep on its current reference; the quoted figures are a pilot on a reference that included the Visium samples, and 0.949 is below 0.95. Sweeps are holdout-only and test the dominant type; no hotspot cell type is tested; no pass rule is registered | `deconvolution/tissue_references.yaml:67-91`; `deconvolution/score_purity_sweep.py:52-65` | as read |
| 5.2 | A | SimBu battery on the current references: BayesPrism degrades on 15 of 15; the corrector ranking is not reproduced; the driver is the highest-mRNA type on 10 of 15. The "no bias" arm sums raw counts, so it keeps native mRNA differences | `deconvolution/SIMBU_BIAS_REFS4ALL_RESULTS.md`; `deconvolution/omnideconv_bench/simulate_simbu.R:109-116` | run and checked in this audit |
| 5.3 | B | BayesPrism's sampler and optimiser settings are never set or recorded. They are the package defaults: chain 1,000, burn-in 500, thinning 2, seed 123 for every sample, alpha 1, MAP with sigma 2. Provenance omits them | `deconvolution/R/run_deconvolution.R:111,124-129`; `vendor/BayesPrism/BayesPrism/R/run_prism.R:7-8,37` | re-opened |
| 5.4 | B | 13 of the 16 default-run tissues ran with an uncommitted edit of the runner, by their own provenance files | Part 5, 8g deviation 4 | as read |
| 5.5 | B | Mixture panel: compositions are flat Dirichlet, unlike tissue; pools as small as 20 cells are resampled; types a method does not estimate are left out of its scores while the message says "set to 0"; CIBERSORTx ran after other results were read, from an unpinned container | `deconvolution/make_pseudobulk.py:157-166,275`; `deconvolution/score_validation.py:54-57,64,86-88` | as read |
| 5.6 | B | Cross-check settings differ from the project's own plan: DWLS ran as `mast` at p 0.01 on raw counts against the planned `mast_optimized` at 0.05 with TPM. BayesPrism sees every reference cell and the other methods a capped subset. `run_omnideconv.R` writes no provenance and patches installed packages in memory | `deconvolution/R/run_omnideconv.R:146,163-217`; `deconvolution/OMNIDECONV_BENCHMARK_PLAN.md:58,293` | as read |
| 5.7 | B | No low-fraction screen at pseudo-cell construction: cell types estimated at essentially zero still yield pseudo-cells, and a near-empty one is rescaled to a full library at tokenization. The analysable-block rule removes them from the gate and the DE only | Part 5, 8h; Part 6, 9a | as read |
| 5.8 | C | A stale "rel-113" label is written into every pseudo-cell file; counts are truncated, not rounded, before the variance-stabilising transformation; the sex-chromosome filter removes every X-linked gene; SimBu is not in the R package manifest | `deconvolution/build_pseudocells.py:114,139`; `deconvolution/R/score_z_vst.R:61` | as read |
| 5.9 | fixed | The duplicate-column defect that made two purity sweeps exactly 1.0 is fixed, and no current value equals 1. The file-name defect that once destroyed a KIDNEY expression matrix is fixed | `deconvolution/R/score_z_vst.R:57-60` | as read |

## Part 6. Stages 9 and 11: embedding and the gate

| # | Class | Finding | Evidence | Checked |
|---|---|---|---|---|
| 6.1 | A | The hotspot boundary is within the permutation test's Monte Carlo error (see "matters most" 1). An independent replicate in `pca_control.tsv` gives 0.0035 and 0.0048 for the two LUNG blocks | `.../gate_rcv10/corroboration_merged.tsv:104,108` | as read |
| 6.2 | A | The gate compares all four training durations (40 animals) with the 8-week controls (10). Thirty of the trained animals were sacrificed earlier than any control, so the contrast mixes training with time in the study. Sex is not used in folds or permutations | `deconvolution/subspace_probe.py:280-282,169-172,215` | as read |
| 6.3 | A | Two hotspots have no signal in gene space: HYPOTH oligodendrocyte precursor cells (all-genes AUC 0.499) and LUNG vascular smooth muscle cells (0.532) | Part 6, 9f deviation 4 | as read |
| 6.4 | B | The headline corroboration figures are not produced by the gate code. It prints all-block values (embedding against PCA p = 0.47; Augur-probe Spearman 0.799 over 172 blocks). The reported analysable-only values (p = 0.15, 82 of 151; 0.785 over 130) are computed in a notebook | `deconvolution/corroborate_summary.py:76-86`; `notebooks/pipeline9_analysis.ipynb:677-683` | code re-opened |
| 6.5 | B | A "registered threshold 0.80" for the Augur correlation is cited in two places. No pre-registration file states it. The observed 0.785 is below it and the row carries a tick | `deconvolution/AIM2_DECONV_RESULTS.md:87`; `notebooks/pipeline11_analysis.ipynb:437` | searched the pre-registrations: not found |
| 6.6 | B | The hotspot table has no provenance of its own. All three merges write one file name, so the surviving record describes the no-threshold sensitivity | `deconvolution/corroborate_summary.py:109-111` | re-opened |
| 6.7 | B | `embed_cells.py` loads weights with `strict=False` and discards the list of missing keys, so an incomplete load would go unnoticed | `finetune/genecompass/embed_cells.py:129` | as read |
| 6.8 | B | The 6,400 target sum was calibrated on July-reference LIVER pseudo-cells and not repeated. The current LIVER value median is 0.878 against a corpus median of 0.9145 | `config/pipeline_config.yaml:411-413` | as read |
| 6.9 | B | Pseudo-cells are joined to animals by position (`mix<i>` is the i-th line of `bulk_samples.tsv`), with no check in Stages 8 to 13 | Parts 5, 6, 7 | as read |
| 6.10 | C | `pipeline/README.md` says there is no Stage 11, but `pipeline/run_stage11.py` is the gate that produced the default run. No UMAP viewer exists for the default, and the viewer's "hotspot" rule is not the gate's | `pipeline/README.md:25`; `deconvolution/umap_viewer_template.html:508` | Stage 11 re-opened |
| 6.11 | fixed | The earlier seed defect that made the hotspot set irreproducible is fixed | Part 6, 9d deviation 5 | as read |

## Part 7. Stages 10 and 14: differential expression and robustness

| # | Class | Finding | Evidence | Checked |
|---|---|---|---|---|
| 7.1 | A | Single-sex tissues can never be dose-significant: the Fisher combination returns a missing value unless both sexes have a p. All 21 TESTES and OVARY blocks therefore have 0 genes. Their composition test is missing too | `deconvolution/R/run_pseudobulk_de.R:265-267,273` | re-opened |
| 7.2 | A | 4,982 of 5,126 sex-consistency calls rest on repfdr alone, from a poorly fitted density. There is also an unexplained male/female asymmetry: BLOOD erythroid cells have 3,891 male-only and 0 female-only 8-week calls | `run_pseudobulk_de.R:358-360`; `deconvolution/REPFDR_STATUS_16_RESULTS.md` | repfdr part established 2026-11-10; asymmetry as read |
| 7.3 | A | Positive controls: the pre-stated sanity subset recovered 5 of 34 (Tier A 6 of 45), which the specification defines as a red flag. It was resolved by a later check that prints a fixed "pipeline is sound" verdict, on blocks that are 72-96% of their tissue, with SKMGN at 0.394 | `deconvolution/POSCTRL_PREREG.md:65-67`; `deconvolution/validate_parenchyma_dataanchored.py:84-86` | as read |
| 7.4 | B | The specification, the DE code and the first results share one commit, so freezing before results cannot be verified; the specification was edited three times afterwards. Tier C's one recovery bypasses the power floor. Sex-specific rows are judged on one statistic and printed with another | commit `6412cab`; `deconvolution/compare_posctrl.py:140-147,284-304` | as read |
| 7.5 | A | "53,084 dose-significant genes" counts gene x block pairs. 45 of the 151 analysed blocks have a mean fraction below 0.01 and hold 9,277 of the pairs. Blocks are analysed from a median library of 1,000 over about 13,700 genes | Part 7, 10a and 10b deviations | as read |
| 7.6 | B | "9 of 9 RIN/globin ROBUST" tests a different statistic from the headline (the ordinal slope with per-block BH, on hotspots only). The rule passes a hotspot with 0 genes and ignores calls gained. "QUIET 112" in the composition table includes 46 unanalysed and 21 single-sex blocks | `deconvolution/R/de_technical_covariate_robustness.R:79-90`; `deconvolution/build_composition_confound_table.py:56` | as read |
| 7.7 | B | Documents misdescribe the primary test. The code applies IHW to the Fisher combination of the per-sex omnibus F over `factor(week)`, and the combined model is `~ sex + factor(week)`. The manuscript says per-sex signed z at 8 weeks are combined by Fisher's method; one results document says "ordinal-dose test"; the methods sidecar says `~sex*factor(week)` | `run_pseudobulk_de.R:143-146,218,265-267,468`; `manuscript/motrpac_genecompass.tex:1350-1351`; `deconvolution/AIM2_DE_MOTRPAC_COMPARISON.md:24` | re-opened |
| 7.8 | B | IHW is given tissue as a factor with the default `covariate_type = "ordinal"`, so weights are smoothed across alphabetically adjacent tissues. This affects power, not FDR control | `run_pseudobulk_de.R:338`; IHW 1.34.0 source | found and checked in this audit |
| 7.9 | B | repfdr is run on all gene-blocks, where MoTrPAC runs it on training-regulated features only. The column `repfdr_localfdr` holds a tail-area Fdr | Part 7, 10c deviations 3-4 | as read |
| 7.10 | C | `fdr_direction.R` prints a hard-coded conclusion; both FDR scripts carry July constants; `run_stage10.py --alpha` does not reach the R step | `deconvolution/fdr_direction.R:39-40` | as read |

## Part 8. Stage 12 and E.3: transfer and atlas anchoring

| # | Class | Finding | Evidence | Checked |
|---|---|---|---|---|
| 8.1 | A | PRESERVED means human AUC of at least 0.65 and a raw permutation p below 0.05, with no multiplicity correction on the human side. The rat and human tests share cells, labels and seeds | `translation/compare_transfer.py:64-75`; `deconvolution/subspace_probe.py:72-75,215-218` | rule re-opened |
| 8.2 | A | The dose axis is not in the status rule. For LUNG vascular smooth muscle cells the dose correlation goes from 0.284 to -0.138 and for HYPOTH oligodendrocyte precursor cells from 0.112 to -0.240; both are PRESERVED | `.../genecompass_input_v2_refs4all_rebuild_human/gate_rcv10/transfer_comparison.md:25-26` | re-opened |
| 8.3 | B | "91.1%" is the share of ortholog-mapped vocabulary genes with the same token ID (13,883 of 15,234). No per-cell token overlap is recorded. The per-cell cosine (0.974) is lower than the cosine between different cells of a tissue (0.993), and the project has withdrawn it as identity evidence | `translation/transfer_to_human.py:34-36`; `docs/RETRAIN_V2_RESULTS.md:288-294` | first part re-opened |
| 8.4 | B | 20 LUNG pseudo-cells exist in rat space and not in human space (1,032 against 1,012), undocumented. A data test asserts the counts are equal and should fail on the default; it was not run | `tests/test_v2_outputs.py:62`; the two `lung/tokenize_summary.json` files | re-opened |
| 8.5 | B | E.3 passes its registered rule (0.430 against 0.086). It depends on the registered centring (uncentred p 0.059), covers 120 of 197 blocks, and 3 of 9 hotspots land on their counterpart in most cells | `translation/atlas_record_16/summary.txt:15` | uncentred p re-opened |
| 8.6 | B | Provenance gaps: `project_to_atlas.py` writes none; the atlas download exists only in gitignored scripts while the manuscript says retrieval scripts are released; the "pre-registered" sex control and its 0.70 floor have no pre-registration document | `manuscript/motrpac_genecompass.tex:1407-1410`; `translation/compare_transfer.py:13-14,156` | as read |
| 8.7 | C | The manuscript's Methods say "nine tissues" where the Results say eleven; the transfer script's "exact corpus tokenization" claim is wrong on normalisation | `manuscript/motrpac_genecompass.tex:1381-1382,956-957`; `translation/transfer_to_human.py:25-31` | as read |

## Part 9. Stage 13: perturbation, GRN, enrichment

| # | Class | Finding | Evidence | Checked |
|---|---|---|---|---|
| 9.1 | A | Each arm keeps its own top 2,048 genes and a missing side is filled with 0, so the difference is not always trained minus control. The docstring says both arms share one gene set. In the six default networks 34-102 genes per arm are arm-exclusive; those edges carry a median 37% and 53% of hub strength in two networks | `finetune/genecompass/build_grn_pooled.py:10-12,46-59,81-82` | code re-opened; shares are the reader's computation |
| 9.2 | A | Module A criterion 2: the curated set was written after the smoke run showed the targets; none of the four frozen targets is in Mef2c's top 25; top-25 lists of unrelated genes overlap Mef2c's by a median 17-19 of 25; v2 fails the criterion (0 of 25); the results file still reads "ALL PASS", which is April only; the registered myeloid replication was never scored | `deconvolution/score_perturb_prereg.py:31-37`; `data/results/retrain_v2/c5/module_a/v2/report.md:9`; `deconvolution/PERTURB_PREREG_RESULTS.md:14` | results files re-opened |
| 9.3 | confirmed, with limits | The decoy null is negative: FDR at abs(z) of 2 is 1.03, and no TF beats its decoys. But "981" and "723" count TF-network pairs (310 and 290 distinct TFs); the 16-tissue rerun is not independent, since its decoys are identical; one seed, one decoy per TF, and the match distance is never written | `deconvolution/grn_decoy_record_16/summary.txt:9-19`; `deconvolution/GRN_DECOY_NULL_16_RESULTS.md:30-32` | as read |
| 9.4 | B | Several quoted figures come from superseded arms. 3,478,564 edges is April's build; the default has 1,527,076. The hub-against-expression correlation of 0.83 exists only for April and v2 on the July references. "0 of 4 myogenic anchors" is the 2026-07-13 run only. "Median" self-consistency is used two ways (0.310 and 0.297) | `docs/OPEN_FRONTIER.md:141`; `deconvolution/REFERENCE_RELABEL_PREREG.md:531` | as read |
| 9.5 | B | Enrichment: rat genes are mapped to human by upper-casing the rat symbol, not through the ortholog table the comments name; the ranking takes its size from an omnibus F p and its sign from a linear slope, not the Fisher/IHW dose p, so OVARY has 141 hits with 0 dose-significant genes; the gene-set data are in a per-user cache, unpinned; 139 of 200,588 tests have no p-value | `deconvolution/R/run_gene_set_enrichment.R:6-8,38-42,86` | code re-opened |
| 9.6 | B | Deleting a gene shifts the absolute position of every lower-ranked gene, an effect not quantified anywhere. 48% of default edges have a stored difference of at most 1e-6. The GRN tables have no provenance record | `finetune/genecompass/perturb_cells.py:98-104`; `vendor/GeneCompass/genecompass/knowledge_embeddings.py:220-222` | as read |
| 9.7 | C | Wording left in code against the project's own reading: "Trustworthy edges", "conserved", "Cross-species conservation". The self-consistency script silently falls back to all targets for a TF with few edges. CPA is an environment only, with no driver or output | `build_grn_pooled.py:14-15`; `finetune/genecompass/compare_grn_conservation.py:9-12,58,104-106,130`; `pipeline/run_stage13.py:203` | first item re-opened |

## Findings that cut across stages

- **Pre-registration timing cannot be shown from the repository in three places:** the retrain's 10% tolerance
  (3.4), the positive-control specification (7.4) and Module A's target set (9.2). In each, the registered text and
  the first results share a commit, or the text already contains results.
- **Stale release labels.** "Ensembl 113 / mRatBN7.2" survives in manifests, scripts, pseudo-cell files, one
  pre-registration and the manuscript. The files are release 115 / GRCr8 (1.9, 2.7, 4.5, 5.8).
- **Provenance.** Several scripts write none (`run_omnideconv.R`, `make_pseudobulk.py`, `score_validation.py`,
  `pheno_merge_test.py`, `augur_prep.py`, `project_to_atlas.py`, the GRN builders, three reference builders). Some
  records store a file's size and time instead of a checksum. Most result evidence is gitignored (`data/`,
  `reports/`, `logs/`), so it is not protected by version control.
- **Stale documents.** `pipeline/README.md` names superseded output folders and omits Stage 11. The body of
  `deconvolution/DECONVOLUTION_PIPELINE_REPORT.md` below its current-figures table describes the July run and
  misdescribes the DE model in places. `docs/ARCHITECTURE.md` and `docs/BUGS.md` describe scripts that no longer
  exist.
- **Uncited methods.** Most methods the pipeline uses have no citation in the code, the docs or the manuscript.
  `REFERENCES.md` lists what a methods section would need.
- **Tests.** No unit test covers Stage 7, the mixture panel, the purity sweeps or the scoring scripts. The data
  tests are not known to be run, and one should currently fail (8.4).

## What the audit confirmed

- The vendored GeneCompass submodule is unmodified upstream (3.8).
- Training ran the stated number of steps, and the sampler defect is fixed (3.8).
- The T3a zero-prior defect is fixed for v2 (2.8).
- The duplicate-column purity-sweep defect, the file-name defect and the gate's seed defect are fixed (5.9, 6.11).
- The default tables hold the reported counts: 9,232 pseudo-cells built and 9,214 embedded; 197 blocks, 151
  analysable; 9 hotspots; 11 of 124 on the 14-tissue arm (Part 6).
- The project records that its clean rebuild reproduced every DE table byte for byte
  (`deconvolution/REFS4ALL_REBUILD_RESULTS.md:26`). The audit did not repeat the rebuild.
- The GeneCompass paper's QC rules, 2,048-gene sequence and prior constructions are quoted correctly in the
  configuration, apart from the mitochondrial cut that is stated and not applied (1.2).
- The BayesPrism paper's benchmark is quoted correctly (Part 5).
- The negative results are stated as negative: no TF-specific signal in the GRN, and the rat-human score is model
  self-consistency (Part 9).

## What could not be determined

Each part ends with its own list. The items that recur:
- package versions for the January-April runs that built the first corpus;
- how many cells the 79 excluded series would add, and how many genes carry counts from a reassigned ID;
- which upstream download the 5-million-cell human and mouse sets came from;
- whether the positional sample join is correct in all 16 tissues;
- whether every checkpoint tensor loads under `strict=False`;
- the generators of `rat_exclude_genes.tsv`, `motrpac_sample_pheno.tsv`, `rat_tf_ensrnog.tsv` and
  `config/corpus_v2_plan.tsv`;
- whether the test suite passes (it was not run).
