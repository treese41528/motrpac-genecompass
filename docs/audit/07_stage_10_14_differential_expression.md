# Part 7. Stage 10 (per-cell-type differential expression, positive controls) and Stage 14 steps 2-3 (composition confound, RIN/globin robustness, FDR sensitivity)

> One part of the methods audit of 2026-11-11; start at [`README.md`](README.md). The main findings of every
> part are collected in [`FINDINGS.md`](FINDINGS.md) and every reference in [`REFERENCES.md`](REFERENCES.md).
> Citations are `path:line` at commit `8249506` unless the text says otherwise.

Audit date 2026-11-11. Repository `/depot/reese18/apps/motrpac-genecompass`, branch `aim2b-perturbation-engine`.
Line numbers are those of HEAD `8249506` (a commit that landed during this audit; of the files cited here it
changed only `deconvolution/DECONVOLUTION_PIPELINE_REPORT.md` and `docs/OPEN_FRONTIER.md`). The manuscript is
gitignored and was also edited during the audit; its line numbers are those of the file as modified
2026-11-11 18:04 and may have moved since.

The default run audited here was produced at commit `aee68a3` on 2026-11-02 (SLURM job 12122211;
`DE_DIR/provenance_run_pseudobulk_de.json:3-4`). Between `aee68a3` and HEAD the Stage 10/14 code changed only in
comments, one docstring path and config folder names (`git diff --stat aee68a3 HEAD` on the files below: header
comment of `run_pseudobulk_de.R`, one usage line of `run_stage10.py`, one added key in `_config_sh.py`, folder
names in `pipeline_config.yaml`). The code described below is therefore the code that produced the outputs.

## Files and short names used in citations

Citations are `basename:line`. Full paths, relative to the repository root:

| Short name | Path |
|---|---|
| `run_stage10.py` | `pipeline/run_stage10.py` |
| `run_stage14.py` | `pipeline/run_stage14.py` |
| `run_pseudobulk_de.R` / `.sh` | `deconvolution/R/run_pseudobulk_de.R` / `.sh` |
| `compare_posctrl.py` | `deconvolution/compare_posctrl.py` |
| `build_posctrl_prereg.py` | `deconvolution/build_posctrl_prereg.py` |
| `posctrl_prereg.tsv`, `posctrl_prereg_refs4.tsv`, `posctrl_celltype_map_refs4.tsv` | `deconvolution/reference/` |
| `build_composition_confound_table.py` | `deconvolution/build_composition_confound_table.py` |
| `de_technical_covariate_robustness.R` / `.sh` | `deconvolution/R/` |
| `fdr_sensitivity.R`, `fdr_direction.R` | `deconvolution/` |
| `repfdr_call_sources.py`, `repfdr_record_16/` | `deconvolution/` |
| `validate_parenchyma_dataanchored.py` | `deconvolution/validate_parenchyma_dataanchored.py` |
| `extract_z.R`, `config.R`, `celltype_names.R` | `deconvolution/R/` |
| `_config_sh.py`, `load_config.sh` | `deconvolution/` |
| `pipeline_config.yaml` | `config/pipeline_config.yaml` |
| `r_packages.yaml`, `r_packages.lock.tsv` | `deconvolution/setup/` |
| `POSCTRL_PREREG.md`, `REPFDR_STATUS_16_RESULTS.md`, `ALL_TISSUE_RERUN_RESULTS.md`, `REFS4ALL_REBUILD_RESULTS.md`, `REFERENCE_RELABEL_PREREG.md`, `REFS4_RERUN_RESULTS.md`, `AIM2_DECONV_RESULTS.md`, `AIM2_DE_MOTRPAC_COMPARISON.md`, `EMBEDDING_DE_STANDARDS.md`, `DECONVOLUTION_PIPELINE_REPORT.md`, `TISSUE_REFERENCE_AND_PROPORTIONS.md` | `deconvolution/` |
| `OPEN_FRONTIER.md` | `docs/OPEN_FRONTIER.md` |
| `README.md` (pipeline) | `pipeline/README.md` |
| `data.R` (MoTrPAC) | `vendor/MotrpacRatTraining6moData/R/data.R` |
| manuscript | `manuscript/motrpac_genecompass.tex` (gitignored) |
| `DE_DIR` | `data/deconvolution/genecompass_input_v2_refs4all_rebuild/pseudobulk_de/` (default; `pipeline_config.yaml:407`, `run_stage10.py:156-157`) |
| run log | `logs/rebuild_xtissue_12122211.out` (gitignored) |

Abbreviations: DE, differential expression; Z, BayesPrism's per-sample, per-cell-type expression estimate;
θ (theta), BayesPrism's per-sample cell-type fraction; CPM, counts per million; IHW, independent hypothesis
weighting; BH, Benjamini-Hochberg; FDR, false discovery rate; RIN, RNA integrity number; "block", one
(tissue x cell type) pair; "hotspot", a block with gate q < 0.05 (Stage 11 gate, Part 6).

Numbers marked "computed here" were obtained in this audit by summing columns of the cited table with a one-line
read; nothing was re-fitted.

---

## Stage 10 - Per-cell-type differential expression and the pre-registered positive controls

### Research role

Question: for each cell type in each tissue, which genes change expression with endurance-training dose
(1, 2, 4, 8 weeks against sedentary controls), and does this cell-type-resolved result agree with what is already
known from the same animals in bulk (MoTrPAC) and from human single-cell exercise data?

Claim that rests on it: "53,084 dose-significant gene-block pairs" over 151 analysable blocks
(`README.md:39`; manuscript:727-728), the per-tissue ranking of responsiveness
(`AIM2_DE_MOTRPAC_COMPARISON.md:28-47`), and the statement that immune cell types recover known biology while the
dominant parenchyma does not (`AIM2_DECONV_RESULTS.md:280-289`). The DE does not use the GeneCompass model; only
the `is_hotspot` flag and the row order come from the gate (`AIM2_DECONV_RESULTS.md:248`,
`run_pseudobulk_de.R:108-117`).

How it is checked: a positive-control specification of 105 gene x target rows with expected directions and a
fixed ladder for reading misses (10e); a composition-confound test and a RIN/globin covariate test (Stage 14); the
registered library-size sensitivities (10d); a byte-for-byte clean rebuild (`REFS4ALL_REBUILD_RESULTS.md:26`).

Verdict so far (default, 16 tissues): 53,084 pairs at FDR_IHW < 0.05; TESTES and OVARY have none by construction
(10c); positive controls Tier A 6 of 45, Tier Ai 2 of 14, Tier B 7 of 22 recovered, and the pre-stated "pipeline
sanity" subset 5 of 34 (10e); of 5,126 "sex-consistent" 8-week calls only 144 are supported by both per-sex tests
(10c).

### Inputs and outputs

Inputs (default configuration):

| Input | Path | Source of the path |
|---|---|---|
| Z, one CSV per cell type (samples x genes), plus `types.txt`, `genes.txt` | `data/deconvolution/results_refs4all_rebuild/motrpac/<TISSUE>/pred_z/predz__<safe(cell type)>.csv` | `pipeline_config.yaml:406`; `run_pseudobulk_de.sh:30`; `run_pseudobulk_de.R:163-166,181` |
| θ per sample | `.../motrpac/<TISSUE>/estimated_fractions.csv` | `run_pseudobulk_de.R:170-171` |
| Sample order per tissue | `data/deconvolution/motrpac_bulk/<TISSUE>/bulk_samples.tsv` | `pipeline_config.yaml:441`; `run_pseudobulk_de.R:166` |
| Phenotype (viallabel, sex, group) | `deconvolution/reference/motrpac_sample_pheno.tsv` (6,156 rows; groups control/1w/2w/4w/8w) | `pipeline_config.yaml:387`; `run_pseudobulk_de.R:104-106` |
| Hotspot table | `data/deconvolution/genecompass_input_v2_refs4all_rebuild/gate_rcv10/corroboration_merged.tsv` | `pipeline_config.yaml:417`; `run_stage10.py:163-165` |
| Positive-control spec | `deconvolution/reference/posctrl_prereg_refs4.tsv` (105 rows) | `pipeline_config.yaml:424`; `run_stage10.py:161-162` |

Z is written by `extract_z.R:53-54` as `get.exp(bp, state.or.type = "type", cell.name = t)`, a sample x gene
matrix per cell type; the DE header describes it as "BayesPrism get.exp = posterior expected COUNT-MASS",
continuous, not integer (`run_pseudobulk_de.R:28-30`).

Sample sizes: 50 samples in each two-sex tissue (25 male, 25 female; 5 per sex per group, checked here for BLOOD,
HEART, LUNG by joining `bulk_samples.tsv` to the phenotype table), 25 in TESTES (male), 24 in OVARY (female; 4w
has 4) (`DE_DIR/de_summary.tsv`, columns `n_samples`, `n_male`, `n_female`).

Outputs under `DE_DIR` (`run_pseudobulk_de.R:414,442-454,457-476`; `compare_posctrl.py:307-309,362`):

| File | Content | Size on the default |
|---|---|---|
| `<TISSUE>/de__<safe(cell type)>.tsv` | one row per tested gene, 31 columns (statistics listed in 10a-10c) | 151 files; 2,045,228 gene rows (computed here: sum of `n_genes_tested`) |
| `de_summary.tsv` | one row per block, 23 columns | 197 rows: 151 `ok`, 46 `SKIP: degenerate` (computed here) |
| `de_hotspots.tsv` | the hotspot rows of the summary | 9 rows (`DE_DIR/de_hotspots.tsv:2-10`) |
| `celltype_files.tsv` | cell type -> file manifest for `ok` blocks | 151 rows |
| `de_methods.tsv` | methods sidecar | 11 lines (quoted below) |
| `provenance_run_pseudobulk_de.json` | commit, command, inputs, parameters | commit `aee68a3`, R 4.4.1, seed 1 (`:3-9,216-225`) |
| `posctrl_results.tsv`, `posctrl_responsiveness.tsv`, `posctrl_summary.md`, `provenance_compare_posctrl.json` | positive-control scoring | 81 gene rows, 11 program rows (run log:2795) |
| `../pseudobulk_de_minlib100/`, `../pseudobulk_de_minlib0/` | the two registered sensitivities (same layout) | 158 and 197 `ok` blocks |

`DE_DIR/de_methods.tsv` (default), verbatim: `multiple_testing IHW~tissue`; `sex_consistency repfdr(replication) OR
concordance(per-sex BH 8w), x sign`; `sex_consistency_warnings In column 1 male ,f(z) misfit = 677.7 . Rerun with
increased df | In column 2 female ,f(z) misfit = 614.1 . Rerun with increased df`; `alpha 0.05`; `seed 1`;
`n_blocks_ok 151`; `min_median_libsize 1000`; `n_blocks_skipped_degenerate 46`; `n_fisher_tests 1755187`
(`de_methods.tsv:1-9`).

### Procedure (orchestration)

1. `python pipeline/run_stage10.py --sensitivities 100 0` was the command of the default run
   (`slurm/analysis/rebuild_crosstissue.slurm:22`, under `PIPELINE_PROFILE=config/profiles/refs4all_rebuild.yaml`,
   `:18`; the same command is in `DECONV_REBUILD_RUNBOOK.md:134`).
2. Step 1: `bash deconvolution/R/run_pseudobulk_de.sh [TISSUE ...]` with `OUT_DIR`, `HOTSPOTS`, `DE_SEED` passed in
   the environment (`run_stage10.py:60-66,200-201`). The wrapper reads the config (fatal on failure), sets the
   project R library and `TMPDIR` to the project `tmp/`, and calls
   `Rscript --no-init-file --no-site-file run_pseudobulk_de.R <results_root> <bulk_root> <pheno> <hotspots> <out_dir>`
   (`run_pseudobulk_de.sh:17-21,29-36,42-44`).
3. After step 1, for each value of `--sensitivities`, the same wrapper is rerun with
   `DE_MIN_MEDIAN_LIBSIZE=<value>` into `<pseudobulk_de>_minlib<value>` (`run_stage10.py:209-217`).
4. Step 2: `compare_posctrl.py --de-dir DE_DIR --prereg <spec> --out DE_DIR --alpha 0.05 --min-fraction 0.01
   --min-nonzero 25` (`run_stage10.py:67-79`; exact command in `DE_DIR/provenance_compare_posctrl.json`).
5. The stage is whole-experiment: a `--tissues` subset changes the pooled IHW/repfdr fit and is for debugging only
   (`run_stage10.py:6-10,187-189`).

Run time of the default: step 1 934 s, the two sensitivities 959 s and 993 s, step 2 6.4 s (run log:1515, 2426,
2786, 2800).

### Models and software

| Component | Version | Where pinned | Call and non-default arguments |
|---|---|---|---|
| R | 4.4.1 | `r_packages.yaml:17`; `provenance_run_pseudobulk_de.json:9` | `Rscript --no-init-file --no-site-file` (`run_pseudobulk_de.sh:42`) |
| Bioconductor | 3.20 | `r_packages.yaml:18` | |
| limma | 3.62.2 | `r_packages.yaml:34`; `r_packages.lock.tsv:160` | `eBayes(lmFit(logcpm, design), trend = TRUE, robust = TRUE)` (`run_pseudobulk_de.R:133-134`); `topTable(..., number = Inf, sort.by = "none")` |
| edgeR | 4.4.2 | `r_packages.yaml:35`; `r_packages.lock.tsv:88` | `DGEList(counts = counts)`; `cpm(dge, log = TRUE, prior.count = 1)` (`run_pseudobulk_de.R:209-210`) |
| IHW | 1.34.0 | `r_packages.yaml:36`; `r_packages.lock.tsv:143` | `ihw(G_p, factor(G_tis), alpha = 0.05, seed = DE_SEED)`; `adj_pvalues(r)` (`run_pseudobulk_de.R:337-338`) |
| repfdr | 1.2.3 | `r_packages.yaml:29`; `r_packages.lock.tsv:257` | `ztobins(zmat)` (all defaults); `repfdr(bz$pdf.binned.z, bz$binned.z.mat, non.null = "replication")` (`run_pseudobulk_de.R:357-360`) |
| Python (positive controls, composition table) | pandas 2.3.3, numpy 2.4.3 | `requirements.lock.txt:155,134` | interpreter `/depot/reese18/apps/motrpac-env/bin/python` (`provenance_compare_posctrl.json`) |

IHW and repfdr were first installed by `deconvolution/R/install_ihw_repfdr.sh:32-41` (Bioconductor and
`https://cloud.r-project.org`, no version argument); they were added to the manifest on 2026-11-01 with the
versions read back from the installed library (`r_packages.yaml:28`; `r_packages.lock.tsv:3`).

Arguments not set in the code take the package defaults of these versions. The defaults were read from the
installed packages with `formals()` on 2026-11-11:

- `repfdr::ztobins(zmat, n.association.status = 3, n.bins = 120, type = 0, df = 7, central.prop = 0.5, pi0 = NULL,
  trim.z = FALSE, force.bin.number = FALSE, pi.using.plugin = FALSE)`.
- `repfdr::repfdr(..., control = em.control())`, with `em.control(max.iter = 10000, tol = 1e-12)`.
- `IHW::ihw.default(covariate_type = "ordinal", nbins = "auto", nfolds = 5, nfolds_internal = 5,
  nsplits_internal = 1, lambdas = "auto", seed = 1, distrib_estimator = "grenander", lp_solver = "lpsymphony",
  adjustment_type = "BH", null_proportion = FALSE)`.
  - The covariate passed is `factor(tissue)`. For a factor, IHW 1.34.0 uses one bin per level and, with
    `covariate_type = "ordinal"`, a total-variation penalty across the levels in their order (alphabetical here).
    `covariate_type = "nominal"` would use a uniform-deviation penalty with no order. Tissue is nominal, so the
    weights are smoothed across alphabetically adjacent tissues. This affects power, not FDR control.
  - `lambdas = "auto"` is the grid 0, 1, nbins/8, nbins/4, nbins/2, nbins, Inf, chosen by internal
    cross-validation.
- `limma::lmFit(method = "ls")`; `limma::eBayes(proportion = 0.01, stdev.coef.lim = c(0.1, 4),
  winsor.tail.p = c(0.05, 0.1))` beyond the `trend = TRUE, robust = TRUE` the code sets.

---

## Stage 10a - The DE model

### Research role
Produce, per block, gene-level statistics for a training effect (any time point), for each time point, for sex,
and for a sex x training interaction.

### Procedure
For each tissue with `pred_z/genes.txt` (hotspot tissues first; `run_pseudobulk_de.R:120-128`) and each cell type
in `types.txt` (`:164-165,180`):

1. Read the cell type's Z (samples x genes) and reorder rows to `mix1..mixN`, where sample i is the i-th line of
   `bulk_samples.tsv` (`:166-169,186-187`). The join of Z rows to animals is positional.
2. Keep samples whose `group` is one of control/1w/2w/4w/8w and whose sex is known; stop the block if fewer than
   12 remain (`:188-189`).
3. Transpose to genes x samples; set negative or missing Z to 0 (`:191`).
4. Gene filter: keep every gene with Z > 0 in at least one sample; record how many all-zero genes were dropped;
   stop the block if fewer than 50 genes remain (`:194-198`). No `filterByExpr`, no minimum-CPM rule.
5. Library size = column sum of Z over the kept genes (`:200`). Apply the analysable-block rule (10d, `:203-207`).
6. Normalise: `DGEList(counts)` then `edgeR::cpm(dge, log = TRUE, prior.count = 1)` (`:209-210`). There is no
   `calcNormFactors` call anywhere in `deconvolution/`, `pipeline/` or `lib/` (searched), so normalisation factors
   are 1: plain library-size log2-CPM, no TMM.
7. Fit four linear models on the same log2-CPM matrix, each with `eBayes(trend = TRUE, robust = TRUE)`
   (`:133-134`):

| Model | Design | Samples | What is taken from it | Lines |
|---|---|---|---|---|
| Combined factor | `~ sex + factor(week)` (`~ factor(week)` if one sex) | all | moderated F over the week coefficients (`F_dose_comb`, `P_dose_comb`); per-time-point log2 fold changes vs control (`lfc_1w`, `lfc_2w`, `lfc_4w`, `lfc_8w`); p of the 8w coefficient (`P_8w_comb`); sex coefficient (`logFC_sex`, `P_sex`) | `:218-230` |
| Ordinal slope | `~ sex + week`, week numeric 0, 1, 2, 4, 8 | all | `slope_week`, `P_week_lin` | `:85,233-235` |
| Interaction | `~ sex * factor(week)` | all, two-sex blocks only | moderated F over the interaction coefficients (`F_sexXweek`, `P_sexXweek`) | `:238-247` |
| Per sex | `~ factor(week)` within males and within females | each sex, needs >= 8 samples and >= 2 week levels | moderated F over the week coefficients (`P_train_M`, `P_train_F`); 8w log2 fold change and its p per sex | `:140-155,250-258` |

Control is the reference level of `factor(week)` (`:86,141,213`). The per-sex fits reuse the log2-CPM computed on
all samples of the block and then subset columns (`:250-251`); each has its own empirical-Bayes prior and trend.

Units of replication: one deconvolved profile per bulk sample (one animal x tissue). Within a block the samples
are different animals. The blocks of one tissue are the same animals deconvolved from the same bulk libraries.

### Parameters, thresholds and flags

| Name | Value | Where set | What it controls | Rationale given |
|---|---|---|---|---|
| `WEEK_NUM` | control 0, 1w 1, 2w 2, 4w 4, 8w 8 | `run_pseudobulk_de.R:85` | coding of the ordinal slope | "retained for continuity with the prior linear-only run" (`:15-16`) |
| `WEEK_LV` | control, 1w, 2w, 4w, 8w | `:86` | factor levels; control = reference | none stated |
| `TP` | 1w, 2w, 4w, 8w | `:87` | defined, not used again in the script | none stated |
| `MIN_SAMPLES_BLOCK` | 12 | `:89` | minimum usable samples per block | "combined fit needs sex + 5 week levels estimable + residual df" (`:89`) |
| `MIN_SAMPLES_PER_SEX` | 8 | `:90` | minimum samples for a per-sex fit | "needs residual df" (`:90`) |
| `MIN_GENES` | 50 | `:91` | minimum genes per block | "eBayes mean-variance trend needs a reasonable gene set" (`:91`) |
| Gene filter | Z > 0 in >= 1 sample | `:194-195` | genes tested | "FULL COVERAGE ... NOT the prior >=50%-of-samples filter" (`:8-9`) |
| Negative/NA Z | set to 0 | `:191` | input cleaning | none stated |
| Library size | column sum of Z | `:200,209` | CPM denominator | "Within a block per-sample libsizes are comparable" (`:34-35`) |
| `prior.count` | 1 | `:210` | offset before the log | none stated |
| Normalisation factors | none (no TMM) | absence of `calcNormFactors` | | none stated |
| `eBayes` | `trend = TRUE, robust = TRUE` | `:134` | limma-trend with robust hyperparameters | "logCPM + eBayes(trend, robust) is justified (Squair 2021 pseudobulk; EMBEDDING_DE_STANDARDS.md)" (`:35-36`) |
| Technical covariates | none | `:37-39` | | "absent from motrpac_sample_pheno.tsv, and Z is a deconvolved expectation" (`:37-39`) |

### Statistical procedures
- Each test is a limma moderated t (one coefficient) or moderated F (several coefficients) from
  `topTable(fit, coef = ...)`; two-sided (`:146,149,221,225,228,235,244`).
- Why limma-trend and not voom or a negative-binomial model, as stated: Z is continuous; "rare-in-tissue cell types
  (<~1% fraction; the MAJORITY of cell types by count) are ~100% <1 (median nonzero ~0.004) and mid-abundance
  immune types (~1-20%) are ~25-58% <1", so rounding to integers for DESeq2 "would zero out essentially all
  rare-type signal AND a quarter-to-half of the mid-abundance immune signal" (`:28-34`; verified by the authors on
  HEART/BLOOD/LUNG, `:29`). voom is not discussed anywhere in the code or docs (searched `deconvolution/`,
  `pipeline/`, `lib/` for "voom": only the vendored MoTrPAC documentation mentions it).
- Nothing in 10a is multiplicity-adjusted by itself; adjustments are in 10b and 10c. `P_dose_comb`, `P_week_lin`
  and `P_8w_comb` are written unadjusted and no FDR is ever computed for them.
- Not pre-registered. The model was introduced with its first results in one commit (`6412cab`, 2026-06-22).

### Checks, validation and their results
- Per-block diagnostics kept in `de_summary.tsv`: `median_libsize`, `frac_zero`, `n_genes_dropped_allzero`
  (`run_pseudobulk_de.R:301-305`). On the 151 analysed blocks (computed here): genes tested per block 9,698 to
  15,259 (median 13,701); all-zero genes dropped 8 to 3,553 (median 266); fraction of zero entries 0.012 to 0.42
  (median 0.089).
- Reproducibility: the clean rebuild reproduced every DE table byte for byte (`REFS4ALL_REBUILD_RESULTS.md:26`).

### Deviations, known defects and caveats
1. The combined model that the sidecar and the docs call "`~sex*factor(week)`" is `~ sex + factor(week)` in the
   code; the interaction model is separate and is used only for the interaction F (`run_pseudobulk_de.R:218,240`
   against the `design` line the script writes, `:468`, and `AIM2_DECONV_RESULTS.md:275`, manuscript:721,1350).
2. The primary dose statistic is not the ordinal slope. `AIM2_DE_MOTRPAC_COMPARISON.md:24` ("FDR_IHW < 0.05 on the
   ordinal-dose test") and `EMBEDDING_DE_STANDARDS.md:116-117` ("`expression ~ ordinal_dose + sex`") describe a
   model that is fitted (`:233-235`) but does not feed FDR_IHW (10b).
3. Low-count regime (arithmetic done here, not a statement in the repo): a block is analysed from a median
   library of 1,000; with about 13,700 genes tested that is below 0.1 count per gene on average, so with
   `prior.count = 1` the log2-CPM of most genes in such a block is set by the prior count. Nine analysed blocks have
   a median library between 1,000 and 10,000; the smallest is 2,253 (OVARY non-classical monocytes) (computed here
   from `de_summary.tsv`).
4. The Z-to-animal join is positional (`mix<i>` = i-th line of `bulk_samples.tsv`, `:166-169,187`); there is no
   check in this script that the Z row names equal the vial labels.
5. No sample is removed as an outlier and the script has no reference to MoTrPAC's outlier list (searched
   `run_pseudobulk_de.R` and `prepare_motrpac_bulk.R` for "outlier": none).
6. Stale comments: the wrapper says "Light ... so a login node is fine" and "full run, 10 tissues"
   (`run_pseudobulk_de.sh:5,9`) while the orchestrator says "this cluster needs a compute node"
   (`run_stage10.py:32-34`); the hotspot comment says "the canonical 22-block gate" (`run_pseudobulk_de.R:108`; the
   default has 9). `r_packages.yaml:35` annotates edgeR with "cpm/calcNormFactors"; `calcNormFactors` is not called.
7. `DECONVOLUTION_PIPELINE_REPORT.md` section 2 (lines 2574-2690) describes the July-reference run and is wrong
   about the code in places: "`log2-CPM ~ sex + factor(week) + offset(libsize)`" (`:2601`; there is no offset),
   "repfdr succeeded" (`:2632`; that run's fit was degenerate, `AIM2_DECONV_RESULTS.md:250-252`), "878,943
   gene-blocks" (`:2630`). Its Stage 10 summary also misdescribes the outputs (`:467,470,476`). The current figures
   are in its top table (`:48-53`).

---

## Stage 10b - Multiplicity control (IHW) and the dose and time-point tests

### Research role
Turn the per-gene training p-values of all blocks into one experiment-wide FDR, the quantity behind
"dose-significant".

### Procedure
1. Training (dose) test, per sex: moderated F-test that all `factor(week)` coefficients are zero in the per-sex
   model, i.e. any difference between the five groups (4 numerator df when all five groups are present)
   (`run_pseudobulk_de.R:143-146`). This is an omnibus across time points, not a per-time-point t-test and not a
   trend test.
2. The two sexes' p-values are combined by Fisher's method (10c) into `P_fisher` (`:265-267`).
3. Pass 2, after all blocks: every finite `P_fisher` from every analysed block is pooled into one vector, with
   the tissue name as covariate (`:294-296`), and `ihw(G_p, factor(G_tis), alpha = 0.05, seed = DE_SEED)` is run
   once; `adj_pvalues()` gives `FDR_IHW` (`:333-343`). If there are 50 or fewer p-values IHW is skipped; if IHW
   throws, the fallback is BH over the pooled vector and the sidecar says so (`:335,339-341`).
4. A gene-block is "dose-significant" when `FDR_IHW < 0.05` (`:423`; `POSCTRL_PREREG.md:35-36`).
5. Secondary, local: BH on `P_fisher` within each block, `FDR_BH_block` (`:287`).
6. Time-point tests: per-time-point log2 fold changes against control come from the combined model (`:222-223`);
   only the 8-week coefficient has a p-value in the output (`P_8w_comb`, `:224-225`). Per-sex per-time-point
   p-values are computed for all four time points (`:149-152`) but only the 8-week ones are kept
   (`:255-258`), turned into signed z and into a per-sex, per-block BH FDR (`fdr_8w_M`, `fdr_8w_F`, `:260-262`).
   No 1w/2w/4w p-value or z is written.

### Parameters, thresholds and flags

| Name | Value | Where set | What it controls | Rationale given |
|---|---|---|---|---|
| `ALPHA` | 0.05 | `run_pseudobulk_de.R:88` (hard-coded) | IHW nominal level; every "significant" count | none stated |
| IHW covariate | tissue (factor) | `:338` | strata for the weights | "Vetr used IHW with a tissue covariate" (`:24-25`) |
| IHW scope | pooled over all analysed blocks of all tissues | `:159,294-296,333-343` | one fit | "more features -> stable" is stated for repfdr (`:40-41`); for IHW none stated |
| `DE_SEED` | 1 | env `DE_SEED`, else `deconvolution.de_seed` (`pipeline_config.yaml:365`), else 1 (`run_pseudobulk_de.R:97-99`; `run_stage10.py:145-146,167`; `_config_sh.py:42`) | IHW's random fold assignment | "IHW's own default is seed = 1L, so DE_SEED = 1 reproduces the earlier (implicitly seeded) runs" (`run_pseudobulk_de.R:94-96`) |
| Minimum pooled tests for IHW | > 50 | `:335` | guard | none stated |
| Fallback | BH over the pooled p | `:340` | if IHW errors | none stated |
| Per-block BH | on `P_fisher` | `:287` | `FDR_BH_block` | "retained as a local secondary" (`:25`) |

### Statistical procedures
- Statistic: Fisher-combined p of two per-sex moderated F-tests. Null: no difference among the five groups in
  either sex. Null distribution: parametric (limma F, then chi-square for the combination). No permutation.
- Multiplicity: IHW (weighted BH with data-driven weights per covariate stratum) at alpha 0.05 across
  **1,755,187** tests (`de_methods.tsv:9`; run log:909), with 14 strata in practice: TESTES and OVARY contribute
  no finite `P_fisher` (10c). The 1,755,187 equals the genes tested in the 130 two-sex analysed blocks; the 21
  single-sex blocks add 290,041 tested genes with no dose p (computed here).
- Result: **53,084** gene-block pairs with `FDR_IHW < 0.05` (sum of `n_sig_dose_IHW` in `de_summary.tsv`, computed
  here; `README.md:39`). Per-block BH gives 53,138 (sum of `n_sig_dose_fisher_BHblock`, computed here).
- Per tissue (computed here, equal to `AIM2_DE_MOTRPAC_COMPARISON.md:30-45`): BLOOD 19,358; LUNG 9,583; BAT 7,303;
  SPLEEN 6,439; HEART 3,719; SKMVL 2,086; SKMGN 1,729; WATSC 1,456; LIVER 992; SMLINT 305; KIDNEY 104; CORTEX 9;
  HYPOTH 1; HIPPOC 0; OVARY 0; TESTES 0. The nine hotspot blocks hold 242, 216, 2,572, 2,438, 232, 2,500, 2,239,
  1,391 and 0 (`de_hotspots.tsv:2-10`).
- One-sided or two-sided: F-tests (direction-free). Direction is read separately from the sign of `lfc_8w`.
- Not pre-registered as a method; the significance definition is fixed in `POSCTRL_PREREG.md:35-36`.

### Checks, validation and their results
- IHW ran (no fallback): `de_methods.tsv:1`.
- Seed recorded: `de_methods.tsv:5`, `provenance_run_pseudobulk_de.json:221`.
- Sensitivity of the pooled fit to block membership: see 10d and 14d.

### Deviations, known defects and caveats
1. "53,084 dose-significant genes" (`README.md:39`, `ALL_TISSUE_RERUN_RESULTS.md:20`) counts gene x block pairs,
   not genes. The docs record that in SPLEEN, BLOOD and LUNG the same genes are significant across most cell types
   of the tissue, "the shared bulk signal of deconvolved profiles, not a cell-type-specific response"
   (`ALL_TISSUE_RERUN_RESULTS.md:48-51`). The blocks of a tissue are computed from the same bulk samples, so their
   tests are not independent hypotheses. The number of unique genes was not computed here.
2. 45 of the 151 analysed blocks have a mean fraction below 0.01 and hold 9,277 of the 53,084 pairs (computed
   here). The positive-control rule treats such blocks as underpowered (10e); the headline count does not exclude
   them.
3. `run_stage10.py --alpha` is passed only to `compare_posctrl.py` (`run_stage10.py:75`); the R step's 0.05 is
   hard-coded (`run_pseudobulk_de.R:88`), so a non-default `--alpha` would score controls at one level against an
   IHW fit made at another.
4. Individual calls near the threshold depend on the pooled fit. Adding SPLEEN and OVARY moved SKMGN Mfn1 from
   0.058 to 0.048 (`ALL_TISSUE_RERUN_RESULTS.md:61-62`); in the July-era sensitivity run, removing two near-empty
   KIDNEY blocks changed 617 calls while the total changed by 1 (14d). Only seed 1 has ever been used
   (`pipeline_config.yaml:365`); no seed-sensitivity run of IHW was found.

---

## Stage 10c - Sex handling: Fisher combination, repfdr 8-week sex-consistency, per-sex concordance

### Research role
Follow the MoTrPAC practice of testing each sex separately and combining, and label genes whose 8-week response
agrees between sexes.

### Procedure and formulas
1. Fisher combination (`run_pseudobulk_de.R:265-267`):
   `P_fisher = pchisq(-2 * (log(P_train_M) + log(P_train_F)), df = 4, lower.tail = FALSE)` if both per-sex p are
   finite, else `NA`.
2. Signed z at 8 weeks, per sex (`:260-261`): `z = qnorm(p/2, lower.tail = FALSE) * sign(log2FC)`, with p and
   log2FC from the per-sex 8w-vs-control coefficient ("exactly Vetr's z definition", `:20`).
3. Per-sex BH on the 8-week p within the block: `fdr_8w_M`, `fdr_8w_F` (`:262`).
4. repfdr, once over all analysed blocks: the pairs (z_M, z_F) of every gene-block with both finite are stacked
   into a two-column matrix (`:297-298,357`); `ztobins(zmat)` with defaults; `repfdr(pdf.binned.z, binned.z.mat,
   non.null = "replication")`; the column `Fdr` of `rr$mat` is kept (`:358-360`). Needs more than 200 rows
   (`:355`). Warnings from `ztobins` are captured and written to the sidecar (`:361-363,460-461`).
5. Degenerate-fit detection: if every finite Fdr is >= 1 - 1e-9, the Fdr vector is discarded and the sidecar is
   labelled "concordance ... repfdr degenerate" (`:369-372`); otherwise it is kept (`:373-376`).
6. State per gene-block (`:395-407`): `repl = Fdr < 0.05`; `conc = fdr_8w_M < 0.05 and fdr_8w_F < 0.05`;
   if `repl or conc`: `up_both` when both z > 0, `down_both` when both z < 0, else `opposite`; otherwise `M_only`
   or `F_only` when exactly one per-sex BH is < 0.05; otherwise `null`. Written as `sexcons_8w`, with the Fdr in a
   column named `repfdr_localfdr` (`:408-409`).

### Parameters, thresholds and flags

| Name | Value | Where set | What it controls | Rationale given |
|---|---|---|---|---|
| Fisher df | 4 (two p-values) | `run_pseudobulk_de.R:266` | combination | "FISHER's sum-of-logs" as in Vetr (`:21`) |
| Missing-sex rule | combined p = NA unless both finite | `:265,267` | single-sex tissues | none stated in code; reported as structural (`ALL_TISSUE_RERUN_RESULTS.md:52-56`) |
| z definition | `qnorm(p/2, lower.tail=FALSE) * sign(lfc)` | `:260-261` | repfdr input | "exactly Vetr's z definition" (`:20`) |
| `ztobins` | defaults | `:358` | binning and density fit | none stated |
| `non.null` | `"replication"` | `:359` | alternative: non-null in both columns | none stated |
| repfdr statistic used | tail-area `Fdr` | `:360` | | none stated |
| repfdr scope | global, all gene-blocks | `:160,297-298` | | "more features -> stable EM, and closer to Vetr's single cross-dataset analysis" (`:40-41`) |
| Minimum rows for repfdr | > 200 | `:355` | guard | none stated |
| Degenerate tolerance | all Fdr >= 1 - 1e-9 | `:369` | label | added after the 2026-10-05 finding (`:347-351`) |
| Call threshold | Fdr < 0.05; per-sex BH < 0.05 | `:395-396` (ALPHA, `:88`) | states | none stated |
| Union rule | repfdr OR concordance | `:401` | | "repfdr (global replicability) OR per-sex BH concordance" (`:22`) |

### Statistical procedures
- Fisher: under the null both per-sex p are uniform and independent (different animals), so -2 sum(log p) is
  chi-square with 4 df. Consequence, implemented at `:265-267`: a tissue sampled in one sex never has a finite
  `P_fisher`, so it never enters IHW and never has a dose-significant gene. On the default all 21 analysed blocks
  of TESTES (6) and OVARY (15) have `n_sig_dose_IHW = 0`, `n_sig_dose_fisher_BHblock = 0` and no sex-consistency
  state (`de_summary.tsv`, read here). Their combined-model statistics (`P_dose_comb`, `P_week_lin`, `lfc_*`) are
  still written but never adjusted.
- repfdr: two-column replicability analysis; the null for a call is "not non-null in both sexes". Units: one row
  per gene-block; 1,755,187 rows on the default (run log:1454). No multiplicity step beyond repfdr's own Fdr.
- Concordance: two per-block BH tests at 0.05 and equal sign.
- None of this is pre-registered. Sex-specific positive-control rows use `fdr_8w_M`/`fdr_8w_F`
  (`POSCTRL_PREREG.md:37`).

### Checks, validation and their results
`REPFDR_STATUS_16_RESULTS.md` (2026-11-10; script `repfdr_call_sources.py`, record `repfdr_record_16/`, job
12304673) re-derived the states from the tables of five runs with the R rule; the recomputed states equal the
stored ones in all five (`REPFDR_STATUS_16_RESULTS.md:10-11`; the script exits non-zero otherwise,
`repfdr_call_sources.py:125-126`).

| Run | Analysed blocks | Fit | f(z) misfit male / female | repfdr calls | Concordance calls | repfdr only |
|---|---|---|---|---|---|---|
| Default, rule 1,000 | 151 | not degenerate | 677.7 / 614.1 | 5,126 | 144 | 4,982 |
| Rule 100 | 158 | not degenerate | 809.5 / 725.4 | 5,106 | 144 | 4,962 |
| No rule | 197 | degenerate, female null fraction 1 | 3,728.3 / 2,831.3 | 0 | 144 | 0 |
| refs4, 14 tissues | 124 | not degenerate | 686.6 / 636.5 | 5,704 | 144 | 5,560 |
| v2, July references | 172 | degenerate | 3,775.4 / 3,398.5 | 0 | 136 | 0 |

(`repfdr_record_16/summary.txt:1-30`; `REPFDR_STATUS_16_RESULTS.md:16-22`.) On the default: 2,089 `up_both`, 3,037
`down_both`, 0 `opposite` (`summary.txt:6`); the 144 concordance calls are HEART 120, SKMVL 23, SKMGN 1, and every
one is also a repfdr call (`REPFDR_STATUS_16_RESULTS.md:33-36`; `repfdr_call_sources_by_tissue.tsv:2-17`); smallest
Fdr 1.9e-05 (`summary.txt:4`). The note's conclusion: "only the 144 concordance calls should be read as
sex-consistent" (`REPFDR_STATUS_16_RESULTS.md:49-50`); nothing downstream reads the states (`:44-46`).

### Deviations, known defects and caveats
1. **Single-sex tissues cannot be dose-significant** (above). MoTrPAC's own tables carry a separate statistic for
   that case ("`lrt` ... likelihood ratio test statistic when only one sex exists", `data.R:2668`). Recorded as open,
   not changed because it would alter the registered method and the pooled fit (`ALL_TISSUE_RERUN_RESULTS.md:80-82`;
   `OPEN_FRONTIER.md:200`). Several docs cite the code as "line ~255"; it is lines 265-267.
2. **97% of the sex-consistent calls rest on a poorly fitted repfdr density.** `ztobins` warns "f(z) misfit ...
   Rerun with increased df" in every non-degenerate run (`de_methods.tsv:3`); `ztobins` is called with defaults and
   never refitted. `de_summary.tsv`'s `n_up_both_8w`/`n_down_both_8w` are the union counts (5,126), not the 144.
3. The sidecar and column names can mislead: the column `repfdr_localfdr` holds the tail-area `Fdr`, not a local
   fdr (`run_pseudobulk_de.R:352,360,408`).
4. repfdr is run on all gene-blocks, whereas MoTrPAC runs it "on the z-score data matrix of all training-regulated
   features at 5% IWH FDR" (`data.R:2961-2962`); the script header names that as "the likely replacement"
   (`run_pseudobulk_de.R:46-48`).
5. **Unexplained male/female asymmetry in the per-sex 8-week calls.** The default has 52,271 sex-specific states
   (`M_only` + `F_only`; sum of `n_sexspecific_8w`, computed here), about ten times the union count. In the one
   block tallied here (BLOOD erythroid cells, `DE_DIR/BLOOD/de__Erythroid_cells.tsv`, column `sexcons_8w`):
   3,891 `M_only`, 0 `F_only`, 43 `up_both`, 62 `down_both`, 9,977 `null`. The four BLOOD hotspot blocks each
   have 3,159 to 4,022 sex-specific calls (`de_hotspots.tsv:4-5,7-8`). The split over all blocks was not
   recomputed (the per-block tables total 2.1 GB). The notes of the project's code review of 2026-10-04 (not in the repository) record the
   same pattern on the July references (42,751 male-only against 760 female-only) as unexplained; that figure was
   not re-derived here. No committed doc explains it; it bears on any male-specific positive control scored in
   BLOOD (10e).
6. `opposite` also catches a consistent call whose z is 0 or missing in one sex (`:405`); no run has any.
7. The manuscript's methods sentence says the "per-sex signed-z at 8 weeks were combined by Fisher's method"
   (manuscript:1350-1351, also 721-722). In the code the signed z go to repfdr; Fisher combines the per-sex
   training F-test p-values.

---

## Stage 10d - The analysable-block rule and degenerate blocks

### Research role
Keep blocks that hold almost no deconvolved expression out of the DE and out of the pooled IHW and repfdr fits.

### Procedure
Rule: a block is analysed when the median over its samples of the library size (sum of Z over genes) is at least
1,000 (`REFERENCE_RELABEL_PREREG.md:106-112`). In the DE it is applied after the gene filter and before
normalisation: a block under the threshold raises a `degenerate_block` condition, gets status
`SKIP: degenerate (median libsize X < 1000)`, keeps `n_samples`, `mean_fraction`, `median_libsize` and `frac_zero`
in `de_summary.tsv`, writes no per-gene table, and contributes nothing to IHW or repfdr
(`run_pseudobulk_de.R:92,203-207,310-321`). The threshold and the count go to the sidecar (`:465-466`). The same rule
is applied in the gate by `corroborate_summary.py --min-libsize` before its BH step
(`REFERENCE_RELABEL_PREREG.md:110-111,341-343`; that script is covered in Part 6).

### Parameters, thresholds and flags

| Name | Value | Where set | What it controls | Rationale given |
|---|---|---|---|---|
| `MIN_MEDIAN_LIBSIZE` | 1,000 | `run_pseudobulk_de.R:92`; env `DE_MIN_MEDIAN_LIBSIZE` overrides | analysable blocks | "the threshold sits in the gap between the two modes of the distribution (65 blocks under 1,000, 4 between 1,000 and 10,000, 103 above)" (`REFERENCE_RELABEL_PREREG.md:107-108`) |
| Registered sensitivities | 100 and 0 (no rule) | `run_stage10.py:142-144,209-217` | written to `_minlib100`, `_minlib0` | `REFERENCE_RELABEL_PREREG.md:112,343-345` |

### Statistical procedures
A fixed threshold on a block-level summary; no test. Pre-registered on 2026-10-27, "before the label audit ... is
computed and before any reference is rebuilt" (`REFERENCE_RELABEL_PREREG.md:3-6`), after the authors had seen the
`median_libsize`/`frac_zero` distribution of the 172 July-reference blocks and the counts of hotspots and dose
calls under 50, 100 and 1,000 (`:40-43`).

### Checks, validation and their results

| Rule | Analysed / all blocks | Fisher tests | Dose-significant pairs | repfdr fit |
|---|---|---|---|---|
| >= 1,000 (default) | 151 / 197 | 1,755,187 | 53,084 | not degenerate |
| >= 100 | 158 / 197 | 1,844,199 | 52,009 | not degenerate |
| none | 197 / 197 | 2,199,357 | 47,829 | degenerate |

(`de_methods.tsv:6-9` of `pseudobulk_de`, `pseudobulk_de_minlib100`, `pseudobulk_de_minlib0`; sums of
`n_sig_dose_IHW` computed here; manuscript:728-729 gives the same three totals.) The hotspot flags are the same 9
in all three (they come from the primary gate table).

The 46 skipped blocks have median libraries 1.28 to 767; by tissue: LUNG 14, BLOOD 5, HYPOTH 5, HEART 4,
HIPPOC 4, KIDNEY 4, OVARY 3, CORTEX 2, SPLEEN 2, SKMGN 1, SKMVL 1, WATSC 1 (computed here from `de_summary.tsv`).
All BLOOD T- and B-cell blocks are among them (`REFERENCE_RELABEL_PREREG.md:370-372`).

### Deviations, known defects and caveats
1. The earlier defect, two near-empty KIDNEY blocks: on the July references "Intercalated cells" and
   "Beta-intercalated cells" (median library 51 and 10) had `status = ok` and sat inside the pooled fits, because
   the script set `ok` on every block that did not throw (per the notes of the project's code review of 2026-10-04, which are not in the repository;
   the two labels are still hard-coded in `fdr_sensitivity.R:43`). In the current code a near-empty block is
   stopped before the fit (`run_pseudobulk_de.R:203-207`),
   and `OPEN_FRONTIER.md:167` records the item as closed by this rule. Those two labels do not exist in the current
   KIDNEY reference; the four KIDNEY blocks skipped now are injured proximal tubule cells (19.3), thin limb cells
   (12.7), principal cells (35.9) and cycling cells (22.4) (`de_summary.tsv`, read here).
2. The threshold was chosen after looking at the distribution it cuts (stated openly, `:40-43`, `:107-108`).
3. Removing blocks raises the count of significant pairs in the remaining ones (47,829 with no rule, 53,084 with
   it). The direction agrees with the July-era sensitivity scripts (14d).
4. A block that passes can still be very sparse (10a caveat 3).

---

## Stage 10e - Pre-registered positive controls

### Research role
Test whether the cell-type DE recovers gene-level findings already published for these animals in bulk
(MoTrPAC main papers, Vetr 2024) and for human blood cells after exercise (Yu 2023), with the genes, targets,
directions and the reading of a miss fixed in advance.

### Procedure
1. `build_posctrl_prereg.py` holds the specification as a literal table `SPEC`
   (`build_posctrl_prereg.py:68-109`) and writes one row per gene with tier, group, symbol, Ensembl rat ID,
   tissue, target cell type(s), sex, expected direction, `anchored_direction`, `testable_prior`, source and notes
   (`:111-140`). Symbols are mapped to rat Ensembl IDs through
   `data/training/gene_universe/pruned_gene_universe.tsv`, case-insensitively (`:30,57-63`). A gene absent from
   that file, and Foxp3, get `testable_prior = FALSE` (`:120-126`).
2. For references whose cell types were renamed, `--celltype-map` rewrites only the target names through
   `posctrl_celltype_map_refs4.tsv` (8 mappings, `posctrl_celltype_map_refs4.tsv:6-13`) and records each renaming
   in `notes`; an unused map entry is an error (`build_posctrl_prereg.py:36-55,132-134`). The default uses the
   resulting `posctrl_prereg_refs4.tsv` (`pipeline_config.yaml:422-424`).
3. `compare_posctrl.py` first checks every explicit target: a target that names no block in `de_summary.tsv`
   stops the run; a target whose block exists but was skipped is reported with a warning and scored
   `not_testable` (`compare_posctrl.py:160-188`).
4. Per gene row (tiers A, Ai, B) (`:250-281`): rows with `testable_prior = FALSE` are `not_testable_prior`
   (`:254-259`). Otherwise the gene is evaluated in each target block (`ANY` = every analysed block of the
   tissue, `:152-157`):
   - present = the gene has a row in the block's table (`:125`);
   - powered = block `mean_fraction >= 0.01` and gene `n_nonzero >= 25` (`:138-139`);
   - significant = `FDR_IHW < alpha` for `sex = any`; `fdr_8w_M < alpha` or `fdr_8w_F < alpha` for `sex = male` or
     `female` (`:140-147`);
   - direction = sign of `lfc_8w` (or `lfc_8w_M`/`lfc_8w_F`) (`:140-148`);
   - confounded = block `frac_week_p < alpha` (`:126-128`).
5. One outcome per gene, in this order (`:191-216`): no block has the gene -> `not_testable`; none powered ->
   `underpowered`; none significant -> `not_significant`; Tier A with an expected direction -> `recovered` (or
   `recovered_confounded`) if any significant powered block has the expected sign, else `wrong_direction`;
   Tiers Ai and B -> `recovered` (or `recovered_confounded`) if any powered block is significant.
6. Tier C, per program and target block (`:284-304`): k = program genes present in the block, m = those with
   `FDR_IHW < alpha`, block rate = fraction of the block's genes with `FDR_IHW < alpha` (`:105-116`); enrichment =
   (m/k)/rate; p = exact upper tail P(X >= m), X ~ Binomial(k, rate) (`:47-52,302-303`). Direction is not scored.
7. A summary line reports how many Tier A rows of the groups containing "mito" (muscle and heart) recovered
   (`:354-359`).

### The committed sets (`build_posctrl_prereg.py:68-109`; row counts from `posctrl_prereg.tsv`, read here)

| Tier | Rows | Content | Expected direction | Source as written |
|---|---|---|---|---|
| A | 45 | 14 genes (Sod2, Slc2a4, Mef2c, Mef2a, Mef2d, Opa1, Mfn1, Prkab1, Tbc1d1, Plin2, Plin4, Plin5, Hspa1b, Hsp90aa1) in SKMVL and in SKMGN skeletal muscle; 6 of them in HEART cardiomyocytes; Stat3, Pxn, Hsp90aa1 in LIVER hepatocytes; Gabpa, Ets1, Klf3, Znf143 in BLOOD (any cell type); Adra1b, Adra1d, Adra2b, Adrb1 in WATSC (any) | up, except WATSC down | "MoTrPAC #21 Fig4", "#21 p7-8", "#21 p3,p5", "#22 p12" (`:69-80`) |
| Ai | 14 | Gja1, Cdh2 (HEART cardiomyocytes); Akt1, Mtor, Hnf4a, Hnf1b (KIDNEY tubule types); Hnf4a, Hnf1b (LIVER hepatocytes); Dsp, Mylk, Stmn1, Syne1, Prkaca (LUNG any, male); Gpr37l1 (HIPPOC any) | none (identity only) | "MoTrPAC #21", "#22" (`:82-86`) |
| B | 22 | Ldlr (CORTEX, HIPPOC, SKMGN, SKMVL); Apob (LUNG, male); Slc6a8 (HEART, LIVER, LUNG); Foxp3 (HEART); Tmbim1 (SKMVL, male); Atp6v1g2 (SKMVL, female); Fads2, Pnkd, Aamp, Ogdh, Ndufa13 and Bag6, Ccnf, Crat, Fam89b (BLOOD, male); Endou (WATSC, female); Abcg8 (LIVER, female); all cell type ANY | none | "Vetr 2024 p3" to "p7" (`:88-102`) |
| C | 24 | five BLOOD programs: monocyte activation (7 genes), cytotoxicity (11), naive (4), ISG (Isg20), chemokine (Cxcr4) | not scored | "Yu 2023 Fig2C/2D/3A/4" (`:104-108`) |

"#21" and "#22" are the reading-list numbers of the two MoTrPAC main papers (`readings/21_Temporal dynamics...`,
`readings/22_Molecular adaptations...`; `POSCTRL_PREREG.md:21`). Pre-declared not testable: Foxp3, and Gnly, Gzmb,
Znf143, Ndufa13, S100a12 (absent from the gene universe) (`POSCTRL_PREREG.md:52-53`).

### Parameters, thresholds and flags

| Name | Value | Where set | What it controls | Rationale given |
|---|---|---|---|---|
| `--alpha` | 0.05 | `run_stage10.py:139`; `compare_posctrl.py:226` | significance and confound flag | `POSCTRL_PREREG.md:35-37` |
| `--min-fraction` | 0.01 | `run_stage10.py:140`; `compare_posctrl.py:227` | power floor, block abundance | `POSCTRL_PREREG.md:54` |
| `--min-nonzero` | 25 | `run_stage10.py:141`; `compare_posctrl.py:228` | power floor, gene | ">=50% of the 50 samples" (`POSCTRL_PREREG.md:55`) |
| Spec file | `posctrl_prereg_refs4.tsv` | `pipeline_config.yaml:424`; `run_stage10.py:161-162` | which spec is scored | targets renamed, "genes/directions unchanged" (`pipeline_config.yaml:422-423`) |
| Binomial p clip | rate clipped to [1e-12, 1 - 1e-12] | `compare_posctrl.py:51` | numerical guard | none stated |

### Statistical procedures
- Tier A is scored on direction: testable, powered, dose-significant, and the sign of the 8-week log2 fold
  change equals the expected one (`POSCTRL_PREREG.md:38-40`). Tiers Ai and B are scored on identity (significant
  in the matching tissue, and sex where given). Tier C is scored on enrichment only (`POSCTRL_PREREG.md:44-46`).
- Miss-reading ladder, fixed in advance (`POSCTRL_PREREG.md:48-61`): (1) coverage, (2) power, (3) confound,
  (4) biology; "No new post-hoc excuse for a specific miss is permitted outside this ladder" (`:60-61`).
- Pre-stated expectation: the Tier A muscle and heart mitochondrial and heat-shock genes are "expected to recover
  (pipeline sanity)"; if they do not, "that is a red flag for the deconvolution->DE pipeline ... not a biological
  null" (`POSCTRL_PREREG.md:65-67`).
- No multiplicity control across control rows or across the Tier C program x block tests.
- Tier C null: each program gene is significant independently with the block's overall rate.

### Checks, validation and their results (default; `DE_DIR/posctrl_summary.md`, run log:2796-2798)

| Tier | Recovered | Wrong direction | Not significant | Not testable | Not testable (prior) |
|---|---|---|---|---|---|
| A (45) | 6 | 3 | 25 | 10 | 1 |
| Ai (14) | 2 | 0 | 12 | 0 | 0 |
| B (22) | 7 | 0 | 8 | 5 | 2 |

- Tier A recovered: HEART cardiomyocytes Opa1 (FDR 0.0095, lfc_8w +0.01) and Mfn1 (0.028, +0.11); SKMGN Mfn1
  (0.048, +0.06) and Tbc1d1 (0.0057, +0.17); SKMVL Slc2a4 (0.028, +0.20); BLOOD erythroid Klf3 (0.023, +0.04)
  (`posctrl_summary.md:54-59`). Wrong direction: BLOOD megakaryocytes Gabpa (-0.30) and Ets1 (-0.56); SKMVL Plin4
  (-0.03) (`:60-62`). Not testable: Sod2, Hspa1b, Hsp90aa1 in both muscles; Hspa1b, Hsp90aa1 in HEART; Adrb1,
  Adra2b in WATSC (the gene has no row in the target block) (`:43-52`).
- Pipeline-sanity line: **5 of 34** muscle and heart Tier A controls recovered (`posctrl_summary.md:136`).
- Tier C: cytotoxicity in BLOOD NK cells 5 of 9 genes, enrichment 2.60, p 0.026; monocyte activation 1 of 6 in
  each monocyte block (p 0.68, 0.69); chemokine 1 of 1 in NK cells (p 0.21); the naive and ISG programs have no
  analysed target block (`posctrl_responsiveness.tsv:2-12`; run log:2792-2794).
- Reading recorded by the project: immune programs recover; the direction-anchored parenchyma controls do not
  (`AIM2_DECONV_RESULTS.md:283-289`; manuscript:739-749).
- Follow-up after the Tier A outcome (`AIM2_DECONV_RESULTS.md:291-301`): (i) genome-wide correlation of bulk and
  parenchyma-Z week slopes (`diagnose_parenchyma.py`; 0.72 to 0.83 on the default per manuscript:751-755);
  (ii) the control genes are flat or down in the rat bulk transcript; (iii) a "data-anchored" control
  (`validate_parenchyma_dataanchored.py`, run as Stage 13 step 5, `pipeline/run_stage13.py:214-215`): bulk
  "movers" = genes with Welch t-test 8w vs control on log2(CPM+1), BH < 0.05 and |log2FC| >= 0.25
  (`validate_parenchyma_dataanchored.py:30,57-64`); recovered = `FDR_IHW < 0.05` and same sign of `lfc_8w` in the
  parenchyma block, among movers with `n_nonzero >= 25` (`:70-73`). Default result: SKMGN 85 of 216 (0.394),
  SKMVL 93 of 170 (0.547), HEART 22 of 24 (0.917), LIVER 0 movers
  (`DE_DIR/parenchyma_dataanchored_validation.tsv:2-5`).

### Deviations, known defects and caveats
1. **Earlier defect (fixed): 28 muscle controls silently not testable.** The spec at first targeted
   "Skeletal muscle cells|Skeletal muscle fibers" (SKMVL) and "Skeletal muscle cells" (SKMGN) (`6412cab`), was
   changed to "Skeletal muscle" for SKMVL the next day (`414a4f4`), and was not regenerated when the muscle labels
   became "Skeletal myocytes"; `resolve_targets()` dropped unknown names, so all 28 rows scored `not_testable`
   (commit message of `92b067a`, 2026-10-05; `compare_posctrl.py:163-166`). Current state, verified: both TSVs
   carry valid targets ("Skeletal myocytes" in `posctrl_prereg.tsv`, "Skeletal muscle cells" in
   `posctrl_prereg_refs4.tsv:2-29`), and an unmatched target now stops the run (`compare_posctrl.py:175-177,186-188`).
2. **The spec has been edited three times since it was called frozen** (`414a4f4` 2026-06-23, `92b067a`
   2026-10-05, `8971469` 2026-10-28; `git log` of the two TSVs). Each is described as a label-only change with
   genes, directions and thresholds untouched (`REFERENCE_RELABEL_PREREG.md:358-365`). The refs4 map does more
   than rename for BLOOD: "Memory T cells" and "ISG-expressing T cells" become three broader types and
   "Naive B cells" becomes "B cells" (`posctrl_celltype_map_refs4.tsv:10-13`), "a loss of resolution"
   (`REFERENCE_RELABEL_PREREG.md:365-366`).
3. **Freezing before the results cannot be verified from the repository.** `POSCTRL_PREREG.md:3` states the spec
   was frozen "before inspecting any per-gene DE result for these genes". The spec, the DE script, the scoring
   script, the two follow-up diagnosis scripts and the first results text were all added in one commit (`6412cab`,
   2026-06-22 23:00), and the script header refers to an earlier "linear-only run" (`run_pseudobulk_de.R:15-16`).
4. **The pre-registered red flag fired and was resolved by an analysis outside the ladder.** 5 of 34 sanity
   controls recovered. The "data-anchored" control was written after the Tier A outcome, its docstring states the
   conclusion ("the pre-registered Tier-A controls were MIS-SPECIFIED", `validate_parenchyma_dataanchored.py:5`),
   and it prints a fixed verdict regardless of the numbers ("confirming the pipeline is sound",
   `:84-86`) with SKMGN at 0.394. The parenchyma blocks it tests are 96%, 95% and 72% of their tissue
   (`DE_DIR/dominant_celltype_flags.tsv:2-3,5`), so agreement of their DE with bulk DE is close to a comparison of
   the bulk with itself. Its BH step is `p * n / rank` clipped at 1 without the cumulative minimum (`:63`).
5. **Tier C has no power floor in the code** (`compare_posctrl.py:284-304` never tests `powered`). The one Tier C
   recovery is in BLOOD NK cells, mean fraction 0.0013 (`posctrl_responsiveness.tsv:10`), below the 0.01 floor
   that the ladder applies to every other tier (`POSCTRL_PREREG.md:54-55`). The 11 program x block p-values are
   uncorrected.
6. **Direction is the sign of a coefficient that is not itself tested.** Significance is the any-time-point
   Fisher/IHW test; direction is the sign of `lfc_8w`. Opa1 in cardiomyocytes counts as recovered with
   lfc_8w +0.008 and Klf3 with +0.036; Plin4 counts as wrong direction with -0.028
   (`DE_DIR/posctrl_results.tsv`, read here). `POSCTRL_PREREG.md:38-39` says the 8-week sign is "corroborated by
   sign of `slope_week`"; the code does not read `slope_week`.
7. **Sex-specific rows are judged on one statistic and reported with another.** For `sex = male/female` the
   decision uses the per-sex, per-block BH on the 8-week contrast, but the tables print `FDR_IHW`
   (`compare_posctrl.py:137,140-147,279`). So Bag6 (printed 0.12) and Crat (0.25) are "recovered" while Ogdh (0.048)
   and Abcg8 (0.026) are "not significant" (`posctrl_summary.md:97,101,114-115`). For Bag6 in BLOOD erythroid
   cells: `fdr_8w_M` 0.0106, `fdr_8w_F` 0.73 (`DE_DIR/BLOOD/de__Erythroid_cells.tsv`, gene ENSRNOG00000000851).
8. **Chance level of the male BLOOD controls is high.** In that block 3,891 of 13,973 genes (28%) are
   male-significant at 8 weeks (10c caveat 5), and an `ANY` target passes if any of the 8 analysed BLOOD blocks is
   significant. 4 of the 7 Tier B recoveries are male BLOOD genes (`posctrl_summary.md:111,114-116`).
9. The ladder's third rung is not a gate in the code: a confounded block only changes the label of a recovery
   (`recovered_confounded`); a miss in a confounded block is not marked (`compare_posctrl.py:126-128,211,216`).
   No default outcome is `recovered_confounded`. The composition p is missing for single-sex blocks (14b).
10. Six of the 28 muscle controls and two of the six heart controls are not testable because the gene is absent
    from the block's table (Sod2, Hspa1b, Hsp90aa1) (`posctrl_summary.md:43-52`); a coverage limit of the
    deconvolved gene set, not a DE outcome.
11. Doc drift: `POSCTRL_PREREG.md:5-6` and the docstrings (`run_stage10.py:20`, `compare_posctrl.py:6`) name
    `posctrl_prereg.tsv`, while the default scores `posctrl_prereg_refs4.tsv`; `POSCTRL_PREREG.md:27` still names
    the original muscle labels; `:30` lists three blood transcription factors where the spec has four (Znf143,
    `build_posctrl_prereg.py:77`). `POSCTRL_PREREG.md:45-46` promises "direction tallies" for Tier C; none are
    written.

---

## Stage 10f - Comparison with MoTrPAC's own differential analysis

### Research role
State how this DE differs from the consortium's published procedure, and whether the cell-type result follows
the bulk tissue ranking.

### Method differences

MoTrPAC's procedure is taken from the documentation of the vendored data package (`data.R`, submodule tag
v2.1.0, DESCRIPTION version 2.0.0) and from the project's own description of Vetr 2024
(`AIM2_DECONV_RESULTS.md:206`; `run_pseudobulk_de.R:4-5,27-28`). The publications themselves were not read in
this audit.

| Aspect | MoTrPAC, as documented in the repository | This pipeline |
|---|---|---|
| Unit | bulk tissue, RSEM gene counts | BayesPrism Z per (tissue x cell type) (`run_pseudobulk_de.R:186`) |
| Low-expression filter | genes with <= 0.5 CPM in all but one sample removed (stated for the normalised data, `data.R:1138-1139`) | any gene non-zero in >= 1 sample (`:194-195`) |
| Normalisation / model | TMM then log-CPM for the normalised data (`data.R:1140-1141`); DE by DESeq2 likelihood-ratio test `nbinomLRT` (`AIM2_DECONV_RESULTS.md:206`; `run_pseudobulk_de.R:28`) | library-size log2-CPM, prior count 1, no TMM; limma-trend, robust (`:209-210,134`) |
| Training test | per sex, full model (technical covariates + training group as a factor) against a reduced model (covariates only), F-test or LRT (`data.R:2744-2747`) | per sex, moderated F over `factor(week)`, no covariates (`:143-146`) |
| Technical covariates | RIN, 5'-3' bias, globin %, PCR-duplicate % (`AIM2_DECONV_RESULTS.md:206`) | none in the primary model (`:37-39`); RIN and globin % tested afterwards on hotspots (Stage 14c) |
| Combining sexes | "sum of logs" (`data.R:2675,2747-2748`) | same formula (`:265-267`) |
| One-sex tissues | a single-sex statistic exists ("`lrt` ... when only one sex exists", `data.R:2668`) | combined p set to missing; no dose test (`:265-267`) |
| Multiplicity | IHW on the combined training p; "training-regulated" = `training_q < 0.05` (`data.R:2749-2751`); IHW "with tissue as a covariate" (`data.R:2844`; Vetr per `run_pseudobulk_de.R:24-25`) | IHW, tissue covariate, pooled over tissue x cell type x gene (`:333-343`) |
| Time-point tests | per sex, each time point against sex-matched controls, with z-scores (`data.R:2062-2064,2091-2092`) | computed per sex for four time points, only 8w exported (`:149-152,255-262`) |
| repfdr | on the z-scores of training-regulated features only, eight columns (four time points x two sexes), configurations in {-1,0,1}^8, a state assigned when its summed posterior exceeds 0.5 (`data.R:2786-2797,2961-2963`) | two columns (8w male, 8w female), all gene-blocks, `non.null = "replication"`, Fdr < 0.05, in union with per-sex BH concordance (`:357-360,395-401`) |
| OVARY, TESTES in the state analysis | excluded "due to missing groups" (`data.R:2768-2770`) | no states (no second sex) |
| Outlier samples | an outlier table and a `removed_samples` field exist (`data.R:2052-2056,2083`) | no removal in the DE script |

The script header lists three "deliberate deviations": limma-trend on continuous Z instead of DESeq2 on counts;
no technical covariates; repfdr run globally (`run_pseudobulk_de.R:27-48`).

### Result comparison
`AIM2_DE_MOTRPAC_COMPARISON.md:28-47` tabulates the default per tissue against MoTrPAC's counts of unique
training-regulated transcript features (`MotrpacRatTraining6moData::TRAINING_REGULATED_FEATURES`, taken from
`REFS4_RERUN_RESULTS.md`): BLOOD 19,358 pairs against 1,582 features (rank 5 of 19); LUNG 9,583 against 961
(rank 6); SKMVL 2,086 against 766; SKMGN 1,729 against 566; CORTEX 9 against 105; HIPPOC 0 against 244; HYPOTH 1
against 16; TESTES 0 against 107. The doc's reading: the tissue ranking holds (blood, lung and muscle high, brain
quiet), TESTES "cannot be compared" because its zero is structural, and HYPOTH is "the one tension" (`:60-65`).

### Deviations, known defects and caveats
1. The comparison is of gene x cell-type pairs against unique bulk features; no rank correlation or test is
   computed; eight tissues have no MoTrPAC figure in the table (`:32-34,37-40,44`).
2. HIPPOC has 0 pairs here against 244 MoTrPAC features (`:43`); the doc's "brain is quiet in both" (`:61-62`)
   holds for CORTEX and HYPOTH more than for HIPPOC. HIPPOC composition is recorded as unreliable
   (`DECONVOLUTION_PIPELINE_REPORT.md:125-126`).
3. Sections 0-3 of the doc (`:89-144`) are the June analysis on earlier references and name cell types that no
   longer exist (`:16-19`); several of its claims (ISG-expressing T cells, memory T cells, the naive program) are
   withdrawn in the current section (`:73-76`).
4. `AIM2_DE_MOTRPAC_COMPARISON.md:24` misstates the test behind FDR_IHW (10a caveat 2).

### Methods references (Stage 10)

| Method | Citation as found | Where |
|---|---|---|
| limma | Ritchie ME, Phipson B, Wu D, et al. limma powers differential expression analyses for RNA-sequencing and microarray studies. Nucleic Acids Res 2015;43(7):e47 | manuscript:1505-1507 |
| limma-trend (mean-variance trend in `eBayes`) | not cited in the repo. Law CW, Chen Y, Shi W, Smyth GK. voom: precision weights unlock linear model analysis tools for RNA-seq read counts. Genome Biol 2014;15:R29 [verified against Crossref; see REFERENCES.md] | |
| Robust empirical Bayes (`robust = TRUE`) | not cited in the repo. Phipson B, Lee S, Majewski IJ, Alexander WS, Smyth GK. Ann Appl Stat 2016;10(2):946-963 [verified against Crossref; see REFERENCES.md] | |
| edgeR (`cpm`) | not cited in the repo. Robinson MD, McCarthy DJ, Smyth GK. Bioinformatics 2010;26(1):139-140 [verified against Crossref; see REFERENCES.md] | |
| Pseudobulk DE | Squair et al. 2021, Nat Commun 12:5692, "Confronting false discoveries in single-cell differential expression" | `EMBEDDING_DE_STANDARDS.md:28-29`; `run_pseudobulk_de.R:36` |
| IHW | Ignatiadis N, Klaus B, Zaugg JB, Huber W. Data-driven hypothesis weighting increases detection power in genome-scale multiple testing. Nat Methods 2016;13:577-580 | manuscript:1509-1511 |
| BH | Benjamini Y, Hochberg Y. Controlling the false discovery rate: a practical and powerful approach to multiple testing. J R Stat Soc Series B 1995;57:289-300 | manuscript:1545-1547 |
| Fisher's method | "FISHER's sum-of-logs" (no reference given). Fisher RA, Statistical Methods for Research Workers, 1925 [book; cited from memory, no registry record checked] | `run_pseudobulk_de.R:21` |
| repfdr | "an algorithm suggested by Yekutieli and Heller in 2014 (Bioinformatics)". Heller R, Yekutieli D. Replicability analysis for genome-wide association studies. Ann Appl Stat 2014;8(1):481-498 [verified against Crossref; see REFERENCES.md]; Heller R, Yaacoby S, Yekutieli D. repfdr: a tool for replicability analysis for genome-wide association studies. Bioinformatics 2014;30(20):2971-2972 [verified against Crossref; see REFERENCES.md] | `data.R:2951-2952` |
| DE recipe followed | Vetr NG, Gay NR, MoTrPAC Study Group, Montgomery SB. The impact of exercise on gene regulation in association with complex trait genetics. Nat Commun 2024;15:3346 | manuscript:1513-1515; `run_pseudobulk_de.R:4-5` |
| Tier A/Ai sources | MoTrPAC Study Group. Temporal dynamics of the multi-omic response to endurance exercise training. Nature 2024;629(8010):174-183 ("#21"); Nair VD, Pincas H, Smith GR, et al. Molecular adaptations in response to exercise training are associated with tissue-specific transcriptomic and epigenomic signatures. Cell Genomics 2024;4(6):100421 ("#22") | manuscript:1458-1464; `POSCTRL_PREREG.md:21` |
| Tier C source | Yu Y, Zhang X, Chen Y, et al. Single-cell sequencing of immune cells after marathon and symptom-limited cardiopulmonary exercise. iScience 2023;26(4):106532 | manuscript:1517-1519 |
| BayesPrism (source of Z) | Chu T, Wang Z, Pe'er D, Danko CG. Cell type and gene expression deconvolution with BayesPrism enables Bayesian integrative analysis across bulk and single-cell RNA sequencing in oncology. Nat Cancer 2022;3:505-517 | manuscript:1491-1493 |
| MoTrPAC data and procedure | R package MotrpacRatTraining6moData, submodule tag v2.1.0 | `vendor/MotrpacRatTraining6moData/DESCRIPTION:5` (Version 2.0.0) |

---

## Stage 14 (steps 2-3) - Composition confound and technical-covariate robustness of the DE

### Research role
Question: could the expression changes called in Stage 10 be explained (a) by a change in the cell type's
fraction with training rather than a change within the cell type, or (b) by RNA quality (RIN) and globin
contamination, which the primary model leaves out?

Claim: "9 of 9 RIN/globin ROBUST; 1 hotspot composition-flagged (SKMVL VSMC)" (`README.md:42`).

Verdict so far: composition verdicts over 197 blocks are 112 QUIET, 62 PASS_EXPRESSION, 23 FLAG_COMPOSITION, one
flag on a hotspot; all nine hotspots are ROBUST under the script's rule
(`DECONVOLUTION_PIPELINE_REPORT.md:52-53`). The scope of each check is narrower than the claim suggests (caveats
below).

### Inputs and outputs

| Step | Input | Output |
|---|---|---|
| 2 | `DE_DIR/de_summary.tsv` | `DE_DIR/composition_confound_table.tsv` (197 rows, 11 columns) |
| 3 | `DE_DIR/de_hotspots.tsv`; Z of each hotspot from `results_refs4all_rebuild/motrpac`; `bulk_samples.tsv`; phenotype table; `vendor/MotrpacRatTraining6moData/data/TRNSCRPT_META.rda` (899 samples; RIN present for 898, `pct_globin` for 899; run log:6525) | `DE_DIR/rin_globin_robustness.tsv` (9 rows), `DE_DIR/provenance_de_technical_covariate_robustness.json` |

(`build_composition_confound_table.py:48-50,66-67`; `de_technical_covariate_robustness.R:22-35,57,61,99-100`.)

### Procedure (orchestration)
`python pipeline/run_stage14.py` (default run: `slurm/analysis/rebuild_crosstissue.slurm:25`). Step 1 (purity
sweeps) is in Part 5. Step 2 runs `build_composition_confound_table.py --de-dir DE_DIR`; step 3 runs
`bash deconvolution/R/de_technical_covariate_robustness.sh` with `DE_DIR` in the environment
(`run_stage14.py:135-142,236`). `--from 2` or `--only 2|3` runs them alone and requires `de_summary.tsv`
(`run_stage14.py:204-205,226-228`). The DE folder comes from the config only; there is no command-line override
(`run_stage14.py:215-217`).

### Models and software
limma 3.62.2, edgeR 4.4.2, data.table 1.18.4 (`r_packages.lock.tsv:160,88,66`), R 4.4.1
(`provenance_de_technical_covariate_robustness.json`); pandas 2.3.3 for step 2. `fdr_sensitivity.R` and
`fdr_direction.R` use IHW 1.34.0.

---

## Stage 14b - Composition-confound table

### Research role
Flag blocks where the cell type's estimated fraction itself trends with training, so that an expression change
there is read as relative rather than within-cell.

### Procedure
1. The test is computed inside the Stage 10 DE script, per analysed block: ordinary least squares
   `lm(fraction ~ week + sex)`, with `fraction` the block's θ per sample from `estimated_fractions.csv`, `week`
   numeric (0, 1, 2, 4, 8) and `sex` a factor; the slope estimate and its two-sided t-test p are stored as
   `frac_week_slope` and `frac_week_p` (`run_pseudobulk_de.R:270-277,309`).
2. Stage 14 step 2 reads `de_summary.tsv` and assigns one verdict per block
   (`build_composition_confound_table.py:37-42,50-58`):
   - `QUIET` if `n_sig_dose_IHW < 25` (missing counts are set to 0 first, `:56`);
   - otherwise `FLAG_COMPOSITION` if `frac_week_p` is present and < 0.05;
   - otherwise `PASS_EXPRESSION`.
3. Rows are sorted hotspots first and written (`:60-67`).

### Parameters, thresholds and flags

| Name | Value | Where set | What it controls | Rationale given |
|---|---|---|---|---|
| Fraction model | `lm(fraction ~ week_numeric + sex)` on raw θ | `run_pseudobulk_de.R:273` | the trend test | "A gene flagged in a block whose fraction also shifts with dose may be a composition artifact" (`:52-54`) |
| `ALPHA` | 0.05 | `build_composition_confound_table.py:33` | θ-trend significance | none stated |
| `MIN_EXPR_SIG` | 25 | `build_composition_confound_table.py:34` | minimum dose-significant genes for a block to be judged | ">= this many dose-IHW-significant genes counts as a real response" (`:34`) |

### Statistical procedures
One linear-regression t-test per block, two-sided, on untransformed fractions; no multiplicity correction across
blocks; not pre-registered. A flag does not exclude a block or a gene; it changes how the block is read
(`compare_posctrl.py:23`; `POSCTRL_PREREG.md:56-57`).

### Checks, validation and their results (default; `DE_DIR/composition_confound_table.tsv`)
- 197 rows: 112 QUIET, 62 PASS_EXPRESSION, 23 FLAG_COMPOSITION (computed here; `DECONVOLUTION_PIPELINE_REPORT.md:52`).
- Hotspots: 1 FLAG_COMPOSITION, 7 PASS_EXPRESSION, 1 QUIET (run log:6493). The flagged hotspot is SKMVL vascular
  smooth muscle cells (mean fraction 0.0022, 216 dose-significant genes, slope +0.000197 per week, p 0.0062;
  `composition_confound_table.tsv:2`). The QUIET hotspot is HYPOTH oligodendrocyte precursor cells (0 genes;
  `:10`). LUNG vascular smooth muscle cells pass at p 0.065 (`:7`).
- The other 22 flags, none a hotspot (read here): BLOOD non-classical monocytes; BAT ASPC, brown adipocytes,
  neuronal-like cells; LUNG mesothelial cells; SPLEEN B cells, non-classical monocytes, NK cells, dendritic
  cells, cycling cells, endothelial cells; HEART T cells, endothelial cells, macrophages, NK cells; SKMVL
  endothelial cells; SKMGN fibro-adipogenic progenitors, endothelial cells, Schwann cells; WATSC mammary luminal
  and glandular epithelial cells; LIVER Kupffer cells.

### Deviations, known defects and caveats
1. **QUIET is not "no confound".** Of the 112 QUIET rows, 46 are blocks that were never analysed (skipped by 10d;
   their missing count becomes 0), 21 are single-sex blocks that cannot have a dose-significant gene, and 45 are
   analysed two-sex blocks with fewer than 25 genes (cross-tabulation computed here). Two of the last group have
   a significant θ trend (WATSC non-classical monocytes p 0.0014, 21 genes; KIDNEY monocytes p 0.025, 9 genes).
2. **The θ-trend p is missing for every single-sex block** (all 21 TESTES and OVARY rows have `frac_week_p`
   empty; read here). The model formula always includes `sex` (`run_pseudobulk_de.R:273`), and the fit is wrapped
   in `tryCatch(..., error = function(e) NULL)`, so a one-level factor leaves the p missing without a message
   (inference from the code; the log was not searched for the error).
3. 25 of the 130 two-sex analysed blocks have p < 0.05 with no correction; about 6.5 would be expected by chance
   at this level if no fraction moved (arithmetic done here).
4. Fractions are regressed on a linear week scale on the raw 0-1 scale; fractions within a sample sum to one, so
   the 130 tests are not independent, and a monotone linear trend is the only alternative tested.
5. The check asks whether θ moves, not whether the expression calls depend on it: θ is not a covariate in any DE
   model.
6. `PASS_EXPRESSION` is defined as "the expression response is not attributable to a fraction shift"
   (`build_composition_confound_table.py:15-16`); what is shown is a non-significant linear trend at 0.05.

---

## Stage 14c - RIN and globin robustness

### Research role
The primary DE has no technical covariates. This step asks, for the hotspot blocks, whether the dose statistic
changes when RIN and percent globin reads are added to the model.

### Procedure (`de_technical_covariate_robustness.R`)
1. Load `TRNSCRPT_META.rda` from the vendored MoTrPAC data package; take `RIN` and `pct_globin` per vial label
   (`:30-35,50-53`).
2. For each row of `de_hotspots.tsv` with `is_hotspot` TRUE (`:57`): read the block's Z and sample list, attach
   sex, group, RIN and globin (`:61-68`); keep samples with all of them present, require at least 12 (`:69-71`).
3. Same preparation as Stage 10: negative or missing Z to 0, drop all-zero genes, require at least 50 genes,
   `edgeR::cpm(DGEList(counts), log = TRUE, prior.count = 1)` (`:73-76`).
4. Fit two limma-trend models with `eBayes(trend = TRUE, robust = TRUE)` (`:79-82`):
   base `~ sex + week` (week numeric 0, 1, 2, 4, 8) and covariate `~ sex + week + RIN + pct_globin` (sex dropped
   when one sex).
5. From the `week` coefficient of each (`:83-89`): Spearman correlation of the moderated t across genes
   (`spearman_dose_t`); Pearson correlation of the log2 fold changes (`pearson_dose_logfc`); number of genes with
   BH `adj.P.Val < 0.05` in the base model (`n_dose_sig_base`), in the covariate model (`n_dose_sig_cov`) and in
   both; `retain_frac` = both / base (missing when base is 0).
6. Verdict (`:90`): `ROBUST` if `spearman_dose_t >= 0.9` and (`retain_frac` is missing or `>= 0.8`); otherwise
   `CHECK`.

### Parameters, thresholds and flags

| Name | Value | Where set | What it controls | Rationale given |
|---|---|---|---|---|
| Covariates | `RIN`, `pct_globin` | `de_technical_covariate_robustness.R:52-53,80` | the adjusted model | "they ARE in the local MotrpacRatTraining6moData::TRNSCRPT_META" (`:3-5`) |
| Metadata file | `vendor/MotrpacRatTraining6moData/data/TRNSCRPT_META.rda`; env `MOTRPAC_TRNSCRPT_META` overrides | `:32-33` | source of covariates | submodule "v2.1.0" (`:30`) |
| Blocks tested | hotspots only | `:57` | scope | "For every HOTSPOT block" (`:5-6`) |
| Dose term | numeric week (ordinal slope) | `:44,79-80` | the statistic compared | none stated |
| `MIN_SAMPLES_BLOCK`, `MIN_GENES` | 12, 50 | `:45` | guards | none stated |
| `prior.count` | 1 | `:76` | log2-CPM | none stated |
| Significance | BH within block, `adj.P.Val < 0.05` | `:87` | counts | none stated |
| ROBUST rule | Spearman >= 0.9 and retention >= 0.8 (or undefined) | `:90` | verdict | none stated |

### Statistical procedures
A descriptive comparison of two fits on the same data; no test of the difference. Per-block BH. Not
pre-registered; `REFERENCE_RELABEL_PREREG.md:127` registers only that Stage 14 steps 2-3 are rerun and reported.

### Checks, validation and their results (default; `DE_DIR/rin_globin_robustness.tsv:2-10`; run log:6526-6537)

| Hotspot | n | Mean RIN | Mean globin % | Significant base -> with covariates (both) | Retained | Spearman t | Verdict |
|---|---|---|---|---|---|---|---|
| SKMVL satellite cells | 50 | 9.75 | 0.031 | 252 -> 339 (217) | 0.861 | 0.957 | ROBUST |
| SKMVL vascular smooth muscle cells | 50 | 9.75 | 0.031 | 210 -> 298 (181) | 0.862 | 0.954 | ROBUST |
| BLOOD megakaryocytes and platelets | 50 | 8.62 | 12.461 | 935 -> 1,343 (934) | 0.999 | 0.998 | ROBUST |
| BLOOD erythroid cells | 50 | 8.62 | 12.461 | 841 -> 1,281 (841) | 1.000 | 0.997 | ROBUST |
| SKMGN tenocytes | 50 | 9.59 | 0.020 | 326 -> 336 (315) | 0.966 | 0.999 | ROBUST |
| BLOOD classical monocytes | 50 | 8.62 | 12.461 | 612 -> 1,152 (612) | 1.000 | 0.997 | ROBUST |
| BLOOD neutrophils | 50 | 8.62 | 12.461 | 104 -> 369 (104) | 1.000 | 0.997 | ROBUST |
| LUNG vascular smooth muscle cells | 50 | 9.98 | 0.042 | 114 -> 143 (95) | 0.833 | 0.950 | ROBUST |
| HYPOTH oligodendrocyte precursor cells | 50 | 9.75 | 0.483 | 0 -> 0 (0) | undefined | 0.938 | ROBUST |

9 of 9 ROBUST; median Spearman 0.997, median retention 0.98 (run log:6537). refs4 (14 tissues): 11 of 11; July
references: 11 of 13 (`DECONVOLUTION_PIPELINE_REPORT.md:53,102`).

### Deviations, known defects and caveats
1. **The check is on a different statistic from the headline.** It refits the ordinal slope with per-block BH;
   "dose-significant" in Stage 10 is the Fisher-combined per-sex omnibus F with IHW. The counts differ
   accordingly: BLOOD neutrophils 104 here against 2,239 in `de_hotspots.tsv:8`; megakaryocytes 935 against
   2,572 (`:4`). An earlier project note says the same: "the two tables use different models"
   (`TISSUE_REFERENCE_AND_PROPORTIONS.md:971-974`).
2. **Scope is the nine hotspot blocks**, not the 151 analysed blocks or the 53,084 pairs.
3. **The rule can pass with nothing to test.** HYPOTH oligodendrocyte precursor cells have no significant gene in
   either model; retention is undefined and the verdict rests on the correlation alone. The same vacuous pass was
   noted on the July references (`TISSUE_REFERENCE_AND_PROPORTIONS.md:976-978`).
4. **The rule ignores calls gained.** In the four BLOOD hotspots the covariates raise the number of significant
   genes by 44% to 255% (935 to 1,343; 104 to 369), which says the covariates explain residual variance in blood
   (mean globin 12.5%); retention only measures calls lost. The primary model still omits them, and the sidecar
   still gives the reason as "absent" (`run_pseudobulk_de.R:469`; `de_methods.tsv:11`), although this script shows
   they are available (`de_technical_covariate_robustness.R:3-5`).
5. Thresholds 0.9 and 0.8 have no stated rationale and were not registered.
6. The other two technical covariates of the MoTrPAC recipe as the project describes it (5'-3' bias,
   PCR-duplicate %; `AIM2_DECONV_RESULTS.md:206`) are not tested.
7. Earlier defect (fixed 2026-10-29): the script hard-coded the July Z folder, and the wrappers did not export
   the config values, so two passes scored the wrong Z; it now reads `CFG_RESULTS_DIR` and prints `Z from:`
   (`REFERENCE_RELABEL_PREREG.md:379-390`; verified at `de_technical_covariate_robustness.R:22,29` and run
   log:6524).
8. The header says the metadata come from "MotrpacRatTraining6moData::TRNSCRPT_META" (`:5`); the file is read
   from the vendored submodule path, not from the installed package (`:32-33`). The submodule is at tag v2.1.0
   with DESCRIPTION version 2.0.0; the installed library copy is 2.0.0 (`r_packages.lock.tsv:185`).

---

## Stage 14d - FDR sensitivity and direction scripts

### Research role
Ask whether near-empty blocks distort the pooled IHW fit for the blocks that are reported, and in which
direction.

### Procedure
Two standalone R scripts, not called by any orchestrator (searched `pipeline/`, `deconvolution/`, `slurm/`), with
no wrapper, no output file and no provenance record; they print to standard output.

- `fdr_sensitivity.R`: read `de_summary.tsv` and, through `celltype_files.tsv`, every block's `P_fisher`
  (`:17-33`); refit IHW (tissue covariate, alpha 0.05, seed from the config, BH fallback; `:35-39`) on four
  block subsets (`:41-46`): `A_all185` (all), `B_drop_kidney` (without KIDNEY "Intercalated cells" and
  "Beta-intercalated cells"), `C_libsize_50` (median library >= 50), `D_abundant_P2` (mean fraction >= 0.01);
  print blocks, tests, significant count, "flips" against scenario A among retained tests, and hotspot blocks
  (`:48-64`). The per-block limma fits are not redone (`:1-5`).
- `fdr_direction.R`: the same assembly; fit IHW on all blocks and on blocks with median library >= 50; among the
  retained tests count calls gained and lost (`:21-38`).

### Parameters, thresholds and flags

| Name | Value | Where set | What it controls | Rationale given |
|---|---|---|---|---|
| `ALPHA` | 0.05 | `fdr_sensitivity.R:12`; `fdr_direction.R:7` | IHW level, call threshold | none stated |
| Seed | `deconvolution.de_seed`, default 1; second CLI argument overrides | `fdr_sensitivity.R:14`; `fdr_direction.R:9` | IHW folds | "IHW's own default" (`fdr_sensitivity.R:9`) |
| Library threshold | 50 | `fdr_sensitivity.R:44`; `fdr_direction.R:28` | "real-mass" blocks | none stated |
| Abundance threshold | mean fraction >= 0.01 | `fdr_sensitivity.R:45` | scenario D | none stated |
| KIDNEY blocks dropped | "Intercalated cells", "Beta-intercalated cells" | `fdr_sensitivity.R:43` | scenario B | the two near-empty July blocks (10d) |

### Checks, validation and their results
The only recorded outputs found are two gitignored logs of 2026-07-13, on the then 185-block DE:

| Scenario | Blocks | Tests | Significant | Flips vs all |
|---|---|---|---|---|
| A, all | 185 | 2,158,329 | 24,915 | 0 |
| B, without the two KIDNEY blocks | 183 | 2,140,525 | 24,916 | 617 |
| C, median library >= 50 | 118 | 1,586,284 | 34,294 | 9,652 |
| D, mean fraction >= 0.01 | 59 | 834,269 | 29,879 | 10,574 |

(`logs/fdr_sens_11281005.out:4-11`.) Within the 118 blocks: 24,642 significant under the all-block fit, 34,294
under the 118-block fit; 9,652 gained, 0 lost (`logs/fdr_sens_11281016.out:5-9`). No committed document reports
these numbers (searched `deconvolution/`, `docs/`, the manuscript). The scripts have not been run on the refs4 or
16-tissue runs; the registered equivalents there are the full reruns of 10d (53,084 / 52,009 / 47,829).

### Deviations, known defects and caveats
1. Both scripts carry July-era constants that do not fit the current default: labels "all185" and "118"
   (`fdr_sensitivity.R:42,66`; `fdr_direction.R:27-36`), a library threshold of 50 where the registered rule is
   1,000, and two KIDNEY cell-type names that no longer exist (`fdr_sensitivity.R:43`). On the current default
   they would read the 151 analysed blocks (the manifest lists only those), so scenarios A, B and C would fit
   the same set of tests.
2. `fdr_direction.R:39-40` prints a fixed conclusion ("the low-mass blocks were making the shared calibration
   MORE CONSERVATIVE, suppressing real discoveries") whatever the counts are.
3. Scenario B is informative about call stability: removing 17,804 tests from two blocks changed 617 calls
   while the total moved by 1 (table above). No analysis of IHW seed or fold sensitivity exists.
4. The header comment says the folder and seed "are recorded in the output" (`fdr_sensitivity.R:9`); the output
   is standard output only.

### Methods references (Stage 14)
limma, edgeR, IHW, BH as for Stage 10. The composition and RIN/globin checks cite no method paper. RIN and
`pct_globin` definitions: `data.R:1694,1734`.

---

## Could not be determined in this audit
- (Resolved afterwards.) The default argument values of `ztobins`, `repfdr`, `ihw`, `lmFit` and `eBayes` in the
  installed versions were read with `formals()` and are listed under "Models and software" of Stage 10.
- Whether the positive-control spec was fixed before any per-gene result was seen (same-commit history; 10e
  caveat 3).
- The split of `M_only` against `F_only` over all 151 blocks (one block tallied; 10c caveat 5), and its cause.
- The number of unique dose-significant genes behind the 53,084 pairs.
- The MoTrPAC and Vetr procedures were taken from the vendored package documentation and the project's own
  description, not from the publications.
