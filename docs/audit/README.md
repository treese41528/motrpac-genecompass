# Methods audit of the MoTrPAC x GeneCompass pipeline (2026-11-11)

> **2026-11-15: a dated record.** The audit describes the default of 2026-11-11 (refs4 references). Its
> findings were acted on afterwards: [`remediation/README.md`](remediation/README.md) says what was done about
> each, and the default configuration is now refs5, the rerun registered there. The audit's parts are kept as
> written.

A stage-by-stage record of what the pipeline does: the models and software used, every threshold and flag with
its value, the statistical procedures in full, and the references behind them. Each statement is tied to the code
line, configuration key or result file that supports it.

- **State audited:** branch `aim2b-perturbation-engine` at commit `8249506`, the default configuration (retrain v2
  model, refs4 references, all 16 tissues, clean-rebuild output tree).
- **Start with** [`FINDINGS.md`](FINDINGS.md) for what a reviewer would question, then the part for the stage you
  need.
- **Nothing was changed.** The audit describes and checks. It fixed no code, configuration or result.

## The research question

Can a single-cell foundation model, extended to rat, turn the MoTrPAC rat endurance-training study (bulk RNA-seq)
into cell-type-resolved exercise biology that carries over to human?

| Aim | Claim the pipeline has to support | Stages |
|---|---|---|
| 1 | A GeneCompass model extended to rat represents rat cells well, without losing human and mouse | 1-7 |
| 2 | Bulk tissue can be split into per-cell-type profiles in which a training response is detectable, at the level of cell types (the gate) and genes (differential expression) | 8-11, 14 |
| 2b / 3b | The model can say which genes drive the response (in-silico perturbation, regulatory networks) | 13 |
| 3a | The response survives re-expression in human terms and lands on the matching human cell types | 12 |

## The pipeline, with each stage's role and the audit's reading

| Stage | What is done | Method and key settings | Reading | Part |
|---|---|---|---|---|
| 1 | Find rat single-cell studies on GEO; type them with an LLM step; build the study catalog | Entrez queries; Claude models classify each study | 79 confirmed single-cell series were lost to a catalog merge; the LLM model is misreported | [1](01_stages_1_3_corpus.md) |
| 2 | Fix the gene universe and filter cells | Ensembl 115 gene set; cells with at least 200 genes, samples with at least 4 cells, at least 7 core genes, +/-3 SD within file. v2 corpus: 6,803,206 cells, 584 files, 81 studies | The stated 15% mitochondrial cut is not applied; Stage 2 does not reproduce from raw | [1](01_stages_1_3_corpus.md) |
| 3 | Map rat genes to tokens | BioMart orthology, tiers T1-T4; vocabulary 55,275 (50,558 inherited, 4,717 new) | Orthology calls are unfiltered; the manifest's commit predates the code that made the outputs | [1](01_stages_1_3_corpus.md) |
| 4 | Per-gene medians for normalisation | Median of each gene's non-zero normalised values over all cells; "scheme B" hybrid of human and rat medians | Scheme B is defined only outside committed records | [2](02_stages_4_6_medians_tokens_priors.md) |
| 5 | Tokenize the corpus and build the training mix | Raw count / hybrid median, log2(1 + x), rank, top 2,048. Mix of 33,138,703 rows from 5.0 M human, 4.97 M mouse and 6.8 M rat cells | Code and its own description disagree on normalisation; the human and mouse sets are ten times their label | [2](02_stages_4_6_medians_tokens_priors.md) |
| 6 | Prior-knowledge embeddings for rat tokens | Promoter, co-expression, gene family, GRN. v2 copies the ortholog's rows for multi-mapping tokens | Rat-specific tokens got almost nothing, contrary to the manuscript | [2](02_stages_4_6_medians_tokens_priors.md) |
| 7 | Continued pretraining of GeneCompass on the mix; acceptance checks C1-C5; Aim-1 validation | BERT-base-size encoder (12 layers, width 768, 12 heads, 2,048 positions; 136,229,358 parameters). Phase 1: 517,793 steps at 1e-4; phase 2: 591,763 steps at 1e-5 | Trained as stated. No held-out data; adoption rested on in-sample loss checks | [3](03_stage_7_model_training_validation.md) |
| 8 (references) | One rat single-cell reference per tissue: 15 references for 16 tissues; bulk prepared from the MoTrPAC data package | Joint re-annotation with marker panels; composite whole-blood reference; gene-ID liftover of the bulk | HYPOTH was never re-annotated; the manual labels are an AI assistant's; 11 genes wrongly excluded | [4](04_stage_8_references_and_bulk.md) |
| 8 (deconvolution) | Split each bulk sample into cell-type fractions and expression; build pseudo-cells; benchmark | BayesPrism 2.2.3 at package-default sampler settings. 9,232 pseudo-cells in 197 tissue x cell-type blocks. Mixture panel, SimBu battery, purity sweeps | Fractions track well on simulated mixtures; expression accuracy is tested only for dominant types; SimBu ranking not reproduced | [5](05_stage_8_bayesprism_benchmarks_pseudocells.md) |
| 9 | Tokenize and embed the pseudo-cells | Target sum 6,400; 768-dimensional CLS vector; 9,214 embedded | The target sum was calibrated on earlier references | [6](06_stage_9_11_embedding_and_gate.md) |
| 11 (gate) | Find the blocks with a detectable training signal | One-component PLS probe, trained against control; AUC over 10 stratified 5-fold splits; 10,000 label permutations; BH q < 0.05 over the 151 blocks with median library of at least 1,000. Augur and a PCA baseline alongside | 9 hotspots, with the cut inside the test's sampling error and two hotspots thin | [6](06_stage_9_11_embedding_and_gate.md) |
| 10 | Genes that respond to training, per block | limma-trend on log2-CPM of the BayesPrism expression; per-sex F over week, Fisher-combined; IHW with tissue as covariate; alpha 0.05. 105 pre-specified positive controls | 53,084 gene x block pairs; single-sex tissues cannot be called; the positive-control red flag fired | [7](07_stage_10_14_differential_expression.md) |
| 14 | Robustness of the DE | Composition-confound table; RIN and globin covariates; purity sweeps (in Part 5) | The robustness checks test a different statistic from the headline | [7](07_stage_10_14_differential_expression.md) |
| 12 | Re-express the pseudo-cells as human and re-test; project onto Tabula Sapiens | 15,234 ortholog pairs; same model with species set to human; nearest-centroid accuracy against a name-shuffle null | "Preserved" is an uncorrected rule; the atlas test passes its registered rule (0.430 against 0.086) | [8](08_stage_12_transfer_and_atlas.md) |
| 13 | In-silico gene deletion; trained-minus-control networks; decoy null; enrichment | Token deletion and re-embedding; bootstrap z per edge; expression-matched decoys; fgsea on Hallmark and Reactome | Negative: no TF-specific signal. The difference is not always on a shared gene set | [9](09_stage_13_perturbation_grn_enrichment.md) |

## Contents

| File | Covers | Lines |
|---|---|---|
| [`FINDINGS.md`](FINDINGS.md) | The main findings of all parts, classed and with how each was checked | 241 |
| [`REFERENCES.md`](REFERENCES.md) | Every reference, checked against Crossref, arXiv or the paper itself | 157 |
| [`01_stages_1_3_corpus.md`](01_stages_1_3_corpus.md) | Harvest, gene universe and cell QC, ortholog mapping | 1,020 |
| [`02_stages_4_6_medians_tokens_priors.md`](02_stages_4_6_medians_tokens_priors.md) | Gene medians, tokenization, training mix, prior embeddings | 952 |
| [`03_stage_7_model_training_validation.md`](03_stage_7_model_training_validation.md) | Architecture, initialisation, training, C1-C5, Aim-1 validation | 695 |
| [`04_stage_8_references_and_bulk.md`](04_stage_8_references_and_bulk.md) | Reference selection, labels, QC gate, special builds, bulk preparation | 1,485 |
| [`05_stage_8_bayesprism_benchmarks_pseudocells.md`](05_stage_8_bayesprism_benchmarks_pseudocells.md) | BayesPrism run, pseudo-cells, mixture panel, cross-check methods, SimBu, purity sweeps, CDSeq | 1,359 |
| [`06_stage_9_11_embedding_and_gate.md`](06_stage_9_11_embedding_and_gate.md) | Pseudo-cell tokenization, embedding, QC, the gate, Augur, PCA baseline, viewer | 1,173 |
| [`07_stage_10_14_differential_expression.md`](07_stage_10_14_differential_expression.md) | DE model, IHW, sex handling, positive controls, composition and RIN/globin checks | 1,007 |
| [`08_stage_12_transfer_and_atlas.md`](08_stage_12_transfer_and_atlas.md) | Re-expression as human, transfer comparison, Tabula Sapiens projection | 1,089 |
| [`09_stage_13_perturbation_grn_enrichment.md`](09_stage_13_perturbation_grn_enrichment.md) | Perturbation engine, Module A validation, GRN, decoy null, self-consistency, enrichment, CPA | 1,230 |
| `crossref_record.tsv` | Raw output of the reference lookups | 86 |

Every part uses one layout per stage or sub-stage: research role; inputs and outputs; procedure; models and
software; a table of parameters, thresholds and flags with where each is set; statistical procedures; checks and
their results; deviations, known defects and caveats; methods references. Each part ends with what it could not
determine.

## How the audit was done

- **Who.** An AI coding assistant (Claude) working in the repository, at the PI's request. Nine reader agents
  each took one group of stages. No domain expert has reviewed the result.
- **Sources, in order of authority.**
  1. The code and `config/pipeline_config.yaml` with its profiles.
  2. On-disk result tables, manifests and provenance files, and the committed record folders.
  3. Pre-registration and results documents. Where a document and the code disagree, the part reports both and
     says which the default run used.
- **Rule for every value.** Each parameter, threshold, flag, seed and count carries a `path:line` citation to a
  file that was opened. Where something could not be found, the part says "NOT FOUND" and where it looked.
- **Central checks, after the parts were written.**
  - The main findings of each part were re-opened at their cited lines or tables. `FINDINGS.md` marks which.
  - Default arguments the readers could not see were read from the installed packages (repfdr, IHW, limma,
    fgsea, SimBu, scikit-learn, SciPy, scanpy).
  - Every reference was looked up; see `REFERENCES.md`.
  - Two papers were opened to check what the code attributes to them (GeneCompass and BayesPrism).

**Limits.**
- No pipeline step was re-run for the audit, and no result table was re-derived. The one exception is the SimBu
  battery, which was finishing as the audit began and is reported in
  `deconvolution/SIMBU_BIAS_REFS4ALL_RESULTS.md`.
- Counts marked "auditor's computation" or "computed here" were made once from an on-disk table. Re-derive them
  before quoting them.
- "As read" findings were not opened a second time.
- Line numbers are those of commit `8249506`. The manuscript is gitignored and is cited as it stood on
  2026-11-11; its line numbers will move when it is edited.
- Evidence under `data/`, `logs/`, `reports/` and `slurm/` is not in version control. A citation to it is to this
  machine's copy.
- The test suite was not run.
