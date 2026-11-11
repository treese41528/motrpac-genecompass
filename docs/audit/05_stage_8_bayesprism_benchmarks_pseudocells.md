# Part 5. Stage 8, part 2: BayesPrism deconvolution, its validation benchmarks, pseudo-cells; Stage 14 step 1 (purity sweeps)

> One part of the methods audit of 2026-11-11; start at [`README.md`](README.md). The main findings of every
> part are collected in [`FINDINGS.md`](FINDINGS.md) and every reference in [`REFERENCES.md`](REFERENCES.md).
> Citations are `path:line` at commit `8249506` unless the text says otherwise.

Audit date 2026-11-11, branch `aim2b-perturbation-engine`. HEAD was `e690c22` when the audit began and
`8249506` when it ended: that commit (18:05, the SimBu results on the current references) landed during the
audit and also edited `deconvolution/OMNIDECONV_RESULTS.md`, `deconvolution/DECONVOLUTION_PIPELINE_REPORT.md`
and, uncommitted because it is ignored, `manuscript/motrpac_genecompass.tex`. All paths are relative to
`/depot/reese18/apps/motrpac-genecompass`. A citation `path:line` refers to the file at `8249506`; the line
numbers of those three files were re-checked after the commit.

**Scope.** This section covers what happens after the references are built and the bulk is prepared: the
BayesPrism run (8g), the extraction of cell-type-specific expression and the construction of pseudo-cells (8h),
the known-truth mixture panel (8i), the five cross-check methods run through omnideconv (8j), the SimBu
mRNA-content-bias battery (8k), the purity sweeps of Stage 14 step 1 (8l) and the reference-free CDSeq run on
BLOOD (8m). Reference selection, labelling and building, and the bulk gene-ID liftover, are in Part 4;
they appear here only as named inputs.

**Abbreviations, defined once.**
- θ (theta): the fraction of a bulk sample's reads that BayesPrism assigns to each cell type. "First" θ comes
  from the initial Gibbs sampler; "final" θ comes from a second sampler run on an updated reference.
- Z: the cell-type-specific expression BayesPrism infers, a (sample × gene × cell type) array of expected
  read counts.
- φ (phi): the reference expression profile per cell state or cell type, each normalised to sum to one.
  ψ (psi): the reference after BayesPrism's update step.
- Cell type and cell state: BayesPrism's two label levels. A cell type is made of one or more cell states.
- Pseudo-cell: one row of Z for one (bulk sample, cell type) pair, treated downstream as if it were a single cell.
- Block: one (tissue, cell type) pair, i.e. all pseudo-cells of one cell type in one tissue.
- VST: DESeq2's variance-stabilising transformation.
- RMSE: root-mean-square error. BH: Benjamini-Hochberg. PBMC: peripheral blood mononuclear cells.
- Holdout set: mixtures and reference come from disjoint cells of the same study. Cross set: mixtures come
  from a different study than the reference.

**Which run is "the default".** `config/pipeline_config.yaml` with no profile. Stage 8 results are in
`data/deconvolution/results_refs4all_rebuild/` (`config/pipeline_config.yaml:406`), pseudo-cells in
`data/deconvolution/genecompass_input_v2_refs4all_rebuild/` (`:407`), Stage 14 sweeps in
`data/deconvolution/validation_refs4all_rebuild/` (`:395`). These folders are the clean rebuild of
2026-11-01/02 (`deconvolution/REFS4ALL_REBUILD_RESULTS.md:1-9,114-122`).

**A note on sources.** Three kinds of evidence used below are not in the committed repository: job logs
(`logs/`, `.gitignore:5`), the SLURM driver scripts (`slurm/`, `.gitignore:37`) and scratch tables (`tmp/`,
`.gitignore:58`). They are on disk and were read, and each use is marked. The defaults of functions inside
installed R packages (omnideconv, DWLS, SimBu) were read from the package databases under `R_libs/` with
`lazyLoad()`, without loading the packages; those are cited as `R_libs/<pkg>/R/<pkg>.rdb::<function>`
because a binary database has no line numbers.

---

## Stage 8g - BayesPrism production run on MoTrPAC bulk

### Research role
- **Question.** Can each tissue's bulk RNA-seq sample be split into per-cell-type fractions (θ) and
  per-cell-type expression (Z), so that the exercise response can be studied one cell type at a time?
- **Claim that rests on it.** Every downstream result (embeddings, the exercise gate, per-cell-type differential
  expression, human transfer, the in-silico-perturbation networks) takes Z as its input.
- **How it is checked.** Known-truth mixtures (8i), a six-method comparison (8j), an mRNA-bias battery (8k),
  purity sweeps (8l). There is no ground truth for the real MoTrPAC samples.
- **Verdict so far.** The project reads θ and Z differentially (a cell type across samples), not as absolute
  composition. Several tissues have implausible absolute fractions that are reported, not corrected
  (`manuscript/motrpac_genecompass.tex:489-497`).

### Inputs and outputs
Inputs, per tissue:
- A built single-cell reference folder: `reference_counts.mtx` (cells × genes, integer), `genes.tsv`,
  `cells_meta.tsv` with columns barcode, sample, leiden, cell_type, cell_state
  (`deconvolution/R/run_deconvolution.R:42-48`; written by `deconvolution/build_reference.py:359-367`).
  The tissue-to-reference map is `deconvolution/tissue_references.yaml`.
- The prepared bulk: `data/deconvolution/motrpac_bulk/<TISSUE>/bulk.mtx` (samples × genes),
  `bulk_genes.tsv`, `bulk_samples.tsv` (`pipeline/run_stage8.py:78-81`; `config/pipeline_config.yaml:441`).
- Three committed gene lists (below).

Outputs, per tissue, under `data/deconvolution/results_refs4all_rebuild/motrpac/<TISSUE>/`
(`pipeline/run_stage8.py:233`):
- `estimated_fractions.csv`: final θ, samples × cell types (`deconvolution/R/run_deconvolution.R:115-116`).
- `bp_result.rds`: the whole BayesPrism object (`:117`).
- `provenance_run_deconvolution.json` (`:119-129`).

Default run, read from each tissue's `provenance_run_deconvolution.json` (cells, samples, minutes), from the
job logs `logs/deconv_<label>_<jobid>.out` (types, states, filter counts; jobs 12121714-12121729) and from
`pred_z/genes.txt` (genes in Z). All 16 ran BayesPrism 2.2.3 with 16 cores.

| Tissue | Reference folder (under `data/deconvolution/`) | Cells | Types | States | Genes in | Excl. list | chrX/Y | Not protein-coding | In < 3 cells | Genes kept | Bulk outlier genes | Genes in Z | Samples | run.prism min |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| BAT | `references_v3/BAT_GSE244451` | 28,246 | 6 | 12 | 17,895 | 190 | 734 | 296 | 2,512 | 14,163 | 10 | 14,162 | 50 | 8.80 |
| BLOOD | `references_v4/BLOOD_composite` | 67,147 | 13 | 21 | 18,260 | 188 | 750 | 158 | 2,233 | 14,931 | 18 | 14,920 | 50 | 14.13 |
| CORTEX | `references_v4/CORTEX_GSE303115` | 12,936 | 12 | 29 | 21,003 | 179 | 854 | 880 | 3,060 | 16,030 | 8 | 15,385 | 50 | 14.71 |
| HEART | `references_v3/HEART_GSE280111_LV` | 135,288 | 16 | 249 | 19,256 | 166 | 733 | 227 | 2,044 | 16,086 | 12 | 15,864 | 50 | 35.68 |
| HIPPOC | `references_v4/HIPPOC_GSE295314` | 23,892 | 11 | 20 | 17,895 | 190 | 734 | 296 | 1,362 | 15,313 | 5 | 15,313 | 50 | 13.02 |
| HYPOTH | `references_v3/HYPOTH_GSE248413_Y` | 8,471 | 13 | 21 | 17,351 | 188 | 667 | 76 | 2,359 | 14,061 | 5 | 14,061 | 50 | 10.04 |
| KIDNEY | `references_v4/KIDNEY_GSE240658` | 24,479 | 15 | 20 | 17,867 | 190 | 728 | 294 | 1,762 | 14,893 | 9 | 14,892 | 50 | 11.01 |
| LIVER | `references_v4/LIVER_GSE220075` | 27,059 | 8 | 18 | 17,895 | 190 | 734 | 296 | 3,904 | 12,771 | 6 | 12,766 | 50 | 12.01 |
| LUNG | `references_v4/LUNG_native_pooled` | 44,158 | 21 | 29 | 16,956 | 151 | 588 | 401 | 0 | 15,816 | 2 | 15,233 | 50 | 16.21 |
| OVARY | `references_v4/OVARY_GSE343214` | 9,104 | 18 | 28 | 30,562 | 205 | 1,024 | 11,180 | 3,579 | 14,574 | 1 | 14,176 | 24 | 6.60 |
| SKMGN | `references_v4/MUSCLE_GSE137869_Y` | 10,740 | 9 | 17 | 17,895 | 190 | 734 | 296 | 3,524 | 13,151 | 18 | 13,142 | 50 | 9.02 |
| SKMVL | `references_v4/MUSCLE_GSE137869_Y` | 10,740 | 9 | 17 | 17,895 | 190 | 734 | 296 | 3,524 | 13,151 | 19 | 13,141 | 50 | 9.24 |
| SMLINT | `references_v4/SMLINT_GSE272055` | 16,193 | 13 | 23 | 17,895 | 190 | 734 | 296 | 3,799 | 12,876 | 6 | 12,874 | 50 | 10.40 |
| SPLEEN | `references_v4/SPLEEN_GSE306764` | 13,013 | 14 | 23 | 22,050 | 190 | 911 | 2,537 | 5,230 | 13,182 | 2 | 13,181 | 50 | 10.27 |
| TESTES | `references_v3/TESTES_OMIX767_lifted` | 5,836 | 6 | 12 | 14,482 | 177 | 511 | 83 | 0 | 13,711 | 0 | 13,711 | 25 | 4.92 |
| WATSC | `references_v4/WATSC_GSE137869_Y` | 11,984 | 13 | 20 | 17,895 | 190 | 734 | 296 | 3,805 | 12,870 | 6 | 12,865 | 50 | 11.62 |

Reading the columns: "Genes in" is the reference's gene count before filtering; the four filter columns are the
genes each filter removed, in the order the code applies them; "Genes kept" is what enters `new.prism`; "Bulk
outlier genes" is BayesPrism's own count of genes it dropped from the bulk; "Genes in Z" is the reference-bulk
intersection that the model actually uses. The sums are 197 cell types and 749 bulk samples (14 tissues × 50, OVARY 24, TESTES 25).

### Procedure
Orchestration.
1. `pipeline/run_deconv_all.py` reads `deconvolution/tissue_references.yaml`, runs two reference gates
   (`:110-130`, described in the reference section) and, per runnable tissue, submits one SLURM job that runs
   `pipeline/run_stage8.py --tissue <T> --ref-dir <ref> --label <t> --from 1 --n-cores 16` followed by Stage 9
   (`pipeline/run_deconv_all.py:131-133`). Job resources: 16 CPUs, 128 GB, 8 h (`:63-65,169-173`).
2. `pipeline/run_stage8.py` runs four steps per tissue (`:74-110`): prepare bulk; deconvolve
   (`deconvolution/R/run_deconvolution.sh`); extract Z; build pseudo-cells. `--n-cores` is passed on as the
   environment variable `N_CORES` (`:268-269`).
3. `deconvolution/R/run_deconvolution.sh` reads the config (`:14`), exports the three gene-list paths and
   two on/off flags (`:15-19`), sets `R_LIBS_USER` to the project library (`:20`), `TMPDIR` to the project
   `tmp/` (`:21`), `N_CORES` (`:22`), and calls the R script (`:33-34`).

`deconvolution/R/run_deconvolution.R`, in execution order.
1. Read the reference as a dense cells × genes matrix; name columns by gene and rows by barcode
   (`:42-46`). Cell-type labels are `cells_meta$cell_type`; cell-state labels are `cells_meta$cell_state`
   (`:47-48`). No cells are dropped or subsampled here: the full reference is used.
2. Remove genes on the exclusion list (ribosomal, mitochondrial, hemoglobin) (`:54-56`).
3. Remove chromosome X and Y genes, if `EXCLUDE_SEX_CHROMOSOMES` is on (`:62-73`).
4. Keep only protein-coding genes, if `PROTEIN_CODING_ONLY` is on (`:79-91`).
5. Keep genes detected (count > 0) in at least 3 reference cells (`:93-96`).
6. Read the bulk as samples × genes; rename the rows `mix1 ... mixN` (`:99-103`). The bulk's own sample
   IDs are discarded here; later stages recover them by row order from `bulk_samples.tsv`
   (`deconvolution/pheno_merge_test.py:8-9,84`; `deconvolution/R/run_pseudobulk_de.R:166-168`).
7. `new.prism(...)` (`:106-109`), then `run.prism(prism, n.cores)` (`:111`).
8. `get.fraction(bp, which.theta = "final", state.or.type = "type")` and write θ (`:115-116`); save the
   object (`:117`); write provenance (`:119-129`).

All samples of a tissue are deconvolved in one call (`:99-111`). Both sexes and all five exercise groups are
therefore in the same call, against the same reference.

What BayesPrism does inside, from the vendored source.
1. `new.prism` drops bulk genes whose share of a sample's reads exceeds `outlier.cut` in more than
   `outlier.fraction` of the samples (`vendor/BayesPrism/BayesPrism/R/new_prism.R:89-100,177-179`), drops
   reference genes with zero total (`:182`), intersects reference and bulk genes (`:185`), sums the reference
   counts within each cell state and within each cell type (`:192-193`), and normalises each profile to sum
   to one with zeros replaced by `pseudo.min` (`:203-204`, function at `:6-11`).
2. `run.prism` runs the initial Gibbs sampler at the **cell-state** level
   (`vendor/BayesPrism/BayesPrism/R/run_prism.R:98-104`), then sums over the states of each type to get
   cell-type θ and Z (`:107-108`; `JointPost_functions.R:90-124`).
3. Per bulk sample, the sampler starts θ at 1/K (`run_gibbs.R:42`), then alternates: allocate each gene's
   reads to the K states by a multinomial with probabilities proportional to φ × θ (`:55-59`); draw θ from
   a Dirichlet with parameter (reads allocated to each state + α) (`:61-62`). Z and θ are the means over the
   retained draws (`:86-87`).
4. `updateReference` sums Z over **all samples of the tissue** (`update_reference.R:96`) and fits an updated
   reference ψ by the "MAP" optimiser (`:98-102`).
5. A second Gibbs sampler, using ψ, yields the final θ (`run_prism.R:127-133`). It returns θ only; Z is not
   re-estimated.

Two consequences follow directly from this code and matter for interpretation.
- Z comes from the initial sampler and the original reference φ. The reported θ comes from the second sampler
  and ψ. They are not estimates under the same reference.
- Z for one sample depends only on that sample's reads and φ (plus the shared outlier-gene filter). Final θ for
  one sample depends on every sample of the tissue, through ψ.

### Models and software
| Item | Value | Where pinned |
|---|---|---|
| BayesPrism | 2.2.3, Danko-Lab R package (not pyBayesPrism) | `R_libs/BayesPrism/DESCRIPTION` (Version, Date 2026-04-02, Built R 4.4.1 2026-05-27); `deconvolution/setup/r_packages.yaml:63-66`; recorded per run in provenance `params.bayesprism` |
| BayesPrism source commit | git submodule `vendor/BayesPrism` at `19052052a6f30833b27e2148294459b0ba2f923e` | `git submodule status`; `.gitmodules:5-7`; `deconvolution/setup/r_packages.yaml:64` |
| Install route | `R CMD INSTALL` of `vendor/BayesPrism/BayesPrism` into `R_libs/` | `deconvolution/R/install_bayesprism.sh:117-119` |
| R | 4.4.1 (module `r/4.4.1`) | `config/pipeline_config.yaml:429`; `deconvolution/setup/r_packages.yaml:17`; provenance `r_version` |
| Bioconductor | 3.20 | `deconvolution/setup/r_packages.yaml:18` |
| BayesPrism dependencies | snowfall 1.84-6.3, NMF 0.30.4.900 (GitHub renozao/NMF `5d062d6f`), scran 1.34.0, BiocParallel 1.40.2 | `deconvolution/setup/r_packages.lock.tsv:301,190,286,28` |
| Container definitions | `rocker/r-ver:4.4.1`, CRAN snapshot `packagemanager.posit.co/.../2026-06-02` | `deconvolution/setup/Dockerfile:20-21,41`; `deconvolution/setup/Apptainer.def:19-20,39` |

The installed copy carries no commit hash of its own (it was installed from a local path), so the link between
`R_libs/BayesPrism` and the submodule commit rests on the matching version and date
(`deconvolution/setup/r_packages.yaml:61-62`).

### Parameters, thresholds and flags
Arguments to `new.prism` and `run.prism`. "Default" means the script does not pass the argument and the value
is the one in the vendored source.

| Name | Value | Where set (path:line) | What it controls | Rationale given |
|---|---|---|---|---|
| `input.type` | `"count.matrix"` | `deconvolution/R/run_deconvolution.R:106` | Reference is a raw cell × gene count matrix, collapsed by BayesPrism | none stated |
| `cell.type.labels` | `cells_meta$cell_type` | `deconvolution/R/run_deconvolution.R:47,107` | The level at which θ and Z are reported | none stated |
| `cell.state.labels` | `cells_meta$cell_state` | `deconvolution/R/run_deconvolution.R:48,108` | The level at which the initial sampler runs | none stated |
| `key` | `NULL` | `deconvolution/R/run_deconvolution.R:109` | No malignant cell type; all types treated equally | "normal tissue, no malignant reference" (`:105`) |
| `outlier.cut` | 0.01 | `deconvolution/R/run_deconvolution.R:109` (equals default, `new_prism.R:133`) | A bulk gene is an outlier in a sample if it takes more than 1% of that sample's reads | none stated |
| `outlier.fraction` | 0.1 | `deconvolution/R/run_deconvolution.R:109` (equals default, `new_prism.R:134`) | Drop the gene if it is an outlier in more than 10% of the tissue's samples | none stated |
| `pseudo.min` | 1e-8 (default) | `vendor/BayesPrism/BayesPrism/R/new_prism.R:135` | Floor replacing zeros in the normalised reference | none stated |
| `n.cores` | 16 in the default run; 4 if `N_CORES` is not given | `pipeline/run_deconv_all.py:64,132`; `deconvolution/R/run_deconvolution.sh:22`; `config/pipeline_config.yaml:428`; `deconvolution/R/run_deconvolution.R:32,111` | Parallel workers; passed to both the sampler and the optimiser (`run_prism.R:78-79`) | none stated |
| `update.gibbs` | `TRUE` (default) | `vendor/BayesPrism/BayesPrism/R/run_prism.R:73` | Whether the reference is updated and θ re-sampled | none stated |
| `gibbs.control$chain.length` | 1000 (default) | `vendor/BayesPrism/BayesPrism/R/run_prism.R:37` | Gibbs iterations per sample | none stated |
| `gibbs.control$burn.in` | 500 (default) | `run_prism.R:37` | Iterations discarded | none stated |
| `gibbs.control$thinning` | 2 (default) | `run_prism.R:37` | Keep every 2nd draw after burn-in: draws 501, 503, ..., 999, i.e. 250 draws (`run_gibbs.R:4-12`) | none stated |
| `gibbs.control$seed` | 123 (default) | `run_prism.R:37` | `set.seed(123)` at the start of **each** sample's chain (`run_gibbs.R:260,281,328,348`) | none in code; named in `deconvolution/REFS4ALL_REBUILD_PREREG.md:14-15` |
| `gibbs.control$alpha` | 1 (default) | `run_prism.R:37` | Symmetric Dirichlet prior on θ | none stated |
| `opt.control$optimizer` | `"MAP"` (default) | `run_prism.R:7-8` | Reference update with a cell-type-specific scaling under a log-normal prior (`run_prism.R:67`) | none stated |
| `opt.control$sigma` | 2 (default) | `run_prism.R:7-8` | Prior scale of the MAP update (`update_reference.R:101`) | none stated |
| `opt.control$maxit` | 100000 (default) | `run_prism.R:7-8` | Optimiser iteration cap | none stated |
| other `opt.control` | `maximize=FALSE, trace=0, eps=1e-07, dowarn=TRUE, tol=0, maxNA=500` (defaults) | `run_prism.R:7-8` | Optimiser controls | none stated |
| `which.theta` | `"final"` | `deconvolution/R/run_deconvolution.R:115` | θ written to `estimated_fractions.csv` is from the second sampler | none stated |

Rat gene filters applied to the reference before `new.prism`. BayesPrism's own `cleanup.genes` and
`select.gene.type` accept only human and mouse, so the project substitutes its own lists
(`deconvolution/R/run_deconvolution.R:7-9,53,58,75`; `deconvolution/build_protein_coding_list.py:6-7`).

| Name | Value | Where set (path:line) | What it controls | Rationale given |
|---|---|---|---|---|
| Exclusion list | `deconvolution/reference/rat_exclude_genes.tsv`: 247 genes labelled 226 ribosomal, 13 mito, 8 hemoglobin (counted from column 3 of the file) | `config/pipeline_config.yaml:372`; `deconvolution/R/run_deconvolution.sh:15`; applied `run_deconvolution.R:54-56` | Removes these genes from the reference, and so from Z | "replaces BayesPrism's hs/mm-only cleanup" (`config/pipeline_config.yaml:371`) |
| Sex-chromosome list | `deconvolution/reference/rat_sex_chrom_genes.tsv`: 1,734 genes (1,574 X, 160 Y) | `config/pipeline_config.yaml:383`; built by `deconvolution/build_sex_chrom_list.py:31-49` from `data/references/biomart/rat_gene_info.tsv`, column "Chromosome/scaffold name" in {X, Y} | Removes all chrX and chrY genes | "stops sex composition from confounding cell-type fractions in mixed-sex bulk" (`config/pipeline_config.yaml:379-381`) |
| `exclude_sex_chromosomes` | `true` | `config/pipeline_config.yaml:384`; `deconvolution/_config_sh.py:38`; `run_deconvolution.sh:19`; `run_deconvolution.R:62` | Turns the filter on | same |
| Protein-coding list | `deconvolution/reference/rat_protein_coding_genes.tsv`: 22,016 genes | `config/pipeline_config.yaml:377`; built by `deconvolution/build_protein_coding_list.py:27-46` from the same BioMart file, "Gene type" normalised to `protein_coding` | Keeps only protein-coding genes | "the same biotype source GeneCompass Stage 2 uses" (`config/pipeline_config.yaml:373-376`) |
| `protein_coding_only` | `true` | `config/pipeline_config.yaml:378`; `deconvolution/_config_sh.py:35`; `run_deconvolution.sh:17`; `run_deconvolution.R:79` | Turns the filter on | same |
| Low-expression filter | gene detected in ≥ 3 reference cells | `deconvolution/R/run_deconvolution.R:93` (hard-coded) | Drops rarely detected genes | "species-agnostic low-expression filter" (`:93`) |
| BioMart source | Ensembl 115 archive (sep2025), assembly GRCr8, md5 `aa206a03...` | `config/pipeline_config.yaml:36-44` | Annotation both generated lists come from | pinned checksum |

Provenance of the three lists.
- Sex-chromosome and protein-coding lists: generated by committed scripts from the pinned BioMart file (above).
- Exclusion list: **NOT FOUND**. No script in the repository builds `rat_exclude_genes.tsv`. Searched every
  `*.py`, `*.R`, `*.sh` for the filename and for ribosomal/hemoglobin patterns; the only similar patterns are
  in `deconvolution/R/prepare_motrpac_bulk.R:292-294`, which classify genes for a liftover report and do not
  write this list. The file was committed once, on 2026-05-29 (commit `e4d5c93`), and has not changed.

Cell caps and subsampling in this step: none. `run_deconvolution.R` uses every reference cell (`:42-46`).
(Caps exist only in the cross-check methods, 8j, and in the SimBu battery, 8k.)

Cell-type versus cell-state labels.
- Cell types are the reference's `cell_type` column, whatever labelling the reference build produced.
- Cell states differ by how the reference was built. Corpus-label builds use one state per (sample, Leiden
  cluster): `cell_state = "{sample}_c{leiden}"` (`deconvolution/build_reference.py:249`). Jointly re-annotated
  builds use the joint cluster, shared across samples (`deconvolution/build_reference.py:338`). This is why
  HEART has 249 states for 16 types while LIVER has 18 for 8 (table above).
- States with fewer than 20 cells are dropped at reference build (`deconvolution/build_reference.py:345-356`);
  in the default run no tissue triggered BayesPrism's "recommend to have sufficient number of cells in each
  cell state" message (`vendor/BayesPrism/BayesPrism/R/new_prism.R:145-146`; absent from all 16 job logs).
- **Why immune subtypes are not merged before deconvolution.** The stated reason is the omnideconv benchmark's
  finding that BayesPrism degrades with coarser types, so the project resolves subtypes and aggregates after
  deconvolution if needed (`deconvolution/OMNIDECONV_RESULTS.md:135`;
  `deconvolution/DECONVOLUTION_PIPELINE_REPORT.md:1347,2005`;
  `deconvolution/OMNIDECONV_BENCHMARK_PLAN.md:217-218`). See the caveats below: this was never tested on rat
  data, and it is not true of every current reference.

### Statistical procedures
- **Estimator.** Posterior means from a Gibbs sampler: 1,000 iterations, the first 500 discarded, every second
  one kept, 250 draws averaged (`run_prism.R:37`; `run_gibbs.R:4-12,86-87`). One chain per bulk sample.
- **Seeds.** Every sample's chain begins with `set.seed(123)` (`run_gibbs.R:260,281`), in both samplers
  (`:328,348`). The clean rebuild reproduced the earlier θ and Z byte for byte
  (`deconvolution/REFS4ALL_REBUILD_RESULTS.md:22-23,108-109`).
- **Uncertainty.** None is carried forward. BayesPrism computes a per-sample coefficient of variation of θ
  across draws (`run_gibbs.R:88,140`), and it is stored in `bp_result.rds`, but no script reads it (searched
  `deconvolution/` and `pipeline/` for `theta.cv`: no use).
- **Convergence.** No diagnostic is computed. One chain per sample, fixed length.
- **Multiplicity.** Not applicable; no test is made in this step.
- **Pre-registration.** The BayesPrism settings themselves are not pre-registered. The rebuild
  pre-registration fixed only "16 cores ... and its default Gibbs seed of 123"
  (`deconvolution/REFS4ALL_REBUILD_PREREG.md:14-15`).

### Checks, validation and their results
| Check | Result | Pass rule | Where |
|---|---|---|---|
| Reproducibility of θ across two independent runs | max abs difference 0.0 in all 16 tissues; files byte-identical | pre-registered ≤ 1e-9 | `deconvolution/REFS4ALL_REBUILD_PREREG.md:22`; `deconvolution/REFS4ALL_REBUILD_RESULTS.md:22` |
| Reproducibility of Z | all 261 compared files byte-identical | pre-registered relative difference ≤ 1e-9 | `deconvolution/REFS4ALL_REBUILD_PREREG.md:23`; `deconvolution/REFS4ALL_REBUILD_RESULTS.md:23` |
| BayesPrism install on the package's own tutorial | script exists; tumor expected as dominant fraction | "tumor" has the largest mean θ | `deconvolution/R/run_tutorial_gbm.R:96-100`; output path `:26,103-105`. Result file not checked in this audit. |
| Plausibility of absolute fractions | hippocampus, kidney, blood, spleen reported implausible | none (reported, not adjusted) | `manuscript/motrpac_genecompass.tex:489-497` |

### Deviations, known defects and caveats
1. **The exclusion list removes a signalling-kinase family as "ribosomal".** `rat_exclude_genes.tsv` lists
   `Rps6ka1`-`Rps6ka6`, `Rps6kb1`, `Rps6kb2`, `Rps6kc1`, `Rps6kl1` and `Rps19bp1` with category "ribosomal"
   (`deconvolution/reference/rat_exclude_genes.tsv:12,16,18,22,26,28,60,86,117,119,198`). These are S6 kinases
   and a binding protein, not ribosomal proteins. The defect was written up in July
   (`deconvolution/TISSUE_REFERENCE_AND_PROPORTIONS.md:287-296,1341`) and is still present: `Rps6kb1`
   (ENSRNOG00000003919) and `Rps6ka1` (ENSRNOG00000042411) are on the bulk gene axis
   (`data/deconvolution/motrpac_bulk/SKMGN/bulk_genes.tsv`) and in none of the 16 `pred_z/genes.txt` files.
   No downstream analysis can therefore report on these genes. Whether removing the error changes any result
   is unknown; it needs a rerun.
2. **Sampler and optimiser settings are inherited, not set.** `run.prism` is called with `n.cores` only
   (`deconvolution/R/run_deconvolution.R:111`). Chain length, burn-in, thinning, seed, α, optimiser and σ are
   the package defaults, and none is written to the provenance record (`:124-129` lists only
   `outlier_cut`, `outlier_fraction`, the two filter flags and sizes). A change of defaults in a future
   BayesPrism version would change results silently. The values are recoverable from
   `bp_result.rds@control_param` (`run_prism.R:141-143`).
3. **Same seed for every sample.** Each sample's chain starts from the same random-number state
   (`run_gibbs.R:260,281`). This makes the run reproducible and independent of the core count as coded, but it
   also means the Monte Carlo noise is not independent across samples. No analysis of its effect was found.
4. **Thirteen of 16 tissues ran with an uncommitted version of the runner.** Their provenance records commit
   `bf0f417` and list `deconvolution/R/run_deconvolution.R` under `git_modified_tracked_files`; LUNG, OVARY
   and SPLEEN ran at `aee68a3` with a clean tree
   (`data/deconvolution/results_refs4all_rebuild/motrpac/*/provenance_run_deconvolution.json`). The edit in
   flight was the one committed as `aee68a3` ("run.prism time in provenance"), judging by the
   `run_prism_minutes` field present in all 16 records, but the exact file that ran is not recoverable from
   the record.
5. **θ and Z are from different stages of the model** (see Procedure). Pseudo-cell library sizes follow the
   initial sampler; `estimated_fractions.csv` is the final sampler. The first θ is not written to a table.
6. **Doc-versus-code disagreements in `deconvolution/DECONVOLUTION_PIPELINE_REPORT.md`.**
   - `:1259` calls `outlier.cut`/`outlier.fraction` "outlier-cell removal". The code filters **bulk genes**
     (`vendor/BayesPrism/BayesPrism/R/new_prism.R:85-100`). The code is what ran.
   - `:1283` calls θ an "Integer-valued categorical distribution". θ is a vector of fractions.
   - `:1265` gives "15-60 min per tissue on 4 cores". The default run used 16 cores and took 4.9-35.7 min
     (provenance).
   - `:1194` gives 1,953 lines for the exclusion list; the file has 248 lines (247 genes).
7. **The decision not to merge immune subtypes rests on the published benchmark only.** The planned
   granularity test on rat data (`deconvolution/OMNIDECONV_BENCHMARK_PLAN.md:211-220`) was not run
   (`deconvolution/OMNIDECONV_RESULTS.md:257`: "Not run: pure/weighted/mirror_db SimBu scenarios").
   Two current references do carry a coarse "Immune cells" type next to resolved types: BAT and KIDNEY
   (`data/deconvolution/results_refs4all_rebuild/motrpac/BAT/pred_z/types.txt:2`,
   `.../KIDNEY/pred_z/types.txt:12`).
8. **The low-expression threshold differs from the BayesPrism tutorial.** The rat runner uses 3 cells
   (`run_deconvolution.R:93`); the tutorial replication uses `exp.cells = 5`
   (`deconvolution/R/run_tutorial_gbm.R:61-66`). No reason is stated.
9. **The sex-chromosome filter removes every X-linked gene**, 511-1,024 genes per reference (table), not only
   sex-specific ones. This mirrors BayesPrism's own chrX/chrY groups (`run_tutorial_gbm.R:64-65`) and is
   stated as intended (`deconvolution/build_sex_chrom_list.py:6-14`), but X-linked genes are then absent from Z.
10. **Stale examples.** `pipeline/run_stage8.py:27-28` shows a reference path that is no longer a production
    reference; `deconvolution/README.md:64` says "all 14 tissues".

### Methods references
- BayesPrism: Chu T, Wang Z, Pe'er D, Danko CG. Nat Cancer 2022;3:505-517
  (`manuscript/motrpac_genecompass.tex:1491-1493`).
- BayesPrism software: Danko-Lab/BayesPrism, version 2.2.3, commit `19052052`
  (`deconvolution/setup/r_packages.yaml:63-66`).

---

## Stage 8h - Extraction of cell-type-specific expression Z and construction of pseudo-cells

### Research role
- **Question.** What exactly is handed to the foundation model as a "cell"?
- **Claim.** One pseudo-cell per (bulk sample, cell type) preserves per-animal variation, which the sex,
  group and time-point contrasts need (`deconvolution/build_pseudocells.py:10-13`).
- **How it is checked.** Unit tests of the connector (`tests/test_pseudocells.py`); file-level reproducibility
  (8g); the count identities below.
- **Verdict.** The connector is a faithful, unfiltered copy of BayesPrism's Z. All selection (minimum genes,
  minimum library) happens in later stages.

### Inputs and outputs
- Input: `bp_result.rds` (8g).
- `pred_z/` under each tissue's results folder: `genes.txt`, `types.txt`, one `predz__<type>.csv`
  (samples × genes) per cell type, `celltype_files.tsv`, `provenance_extract_z.json`
  (`deconvolution/R/extract_z.R:43-44,52-60`).
- `data/deconvolution/genecompass_input_v2_refs4all_rebuild/<label>/pseudocells.h5ad`, `summary.txt`,
  `provenance_build_pseudocells.json` (`deconvolution/build_pseudocells.py:132-147`).

Counts on the default run (each tissue's `summary.txt`, line 1: "pseudocells=N ... (from S samples x T cell
types; 0 all-zero dropped)"):

| Tissue | Samples | Cell types | Pseudo-cells | Analysable blocks |
|---|---|---|---|---|
| BAT | 50 | 6 | 300 | 6 |
| BLOOD | 50 | 13 | 650 | 8 |
| CORTEX | 50 | 12 | 600 | 10 |
| HEART | 50 | 16 | 800 | 12 |
| HIPPOC | 50 | 11 | 550 | 7 |
| HYPOTH | 50 | 13 | 650 | 8 |
| KIDNEY | 50 | 15 | 750 | 11 |
| LIVER | 50 | 8 | 400 | 8 |
| LUNG | 50 | 21 | 1,050 | 7 |
| OVARY | 24 | 18 | 432 | 15 |
| SKMGN | 50 | 9 | 450 | 8 |
| SKMVL | 50 | 9 | 450 | 8 |
| SMLINT | 50 | 13 | 650 | 13 |
| SPLEEN | 50 | 14 | 700 | 12 |
| TESTES | 25 | 6 | 150 | 6 |
| WATSC | 50 | 13 | 650 | 12 |
| **Total** | 749 | **197** | **9,232** | **151** |

- 9,232 pseudo-cells, 197 blocks: sums of the `summary.txt` files; also `deconvolution/README.md:16`,
  `pipeline/README.md:37`.
- 151 analysable blocks: column `analysable` of
  `data/deconvolution/genecompass_input_v2_refs4all_rebuild/gate_rcv10/corroboration_merged.tsv` (197 rows,
  151 true, 9 with `q_sup_trained < 0.05`); per-tissue counts in the last column are from the same file.
- No pseudo-cell was all-zero in any tissue ("0 all-zero dropped" in all 16 `summary.txt`).
- 9,214 of the 9,232 are embedded; the 18 lost are LUNG pseudo-cells under the tokenizer's 200-gene minimum
  (`data/deconvolution/genecompass_input_v2_refs4all_rebuild/lung/tokenize_summary.json`:
  `n_pseudocells` 1032, `n_dropped_lt_min_genes` 18).

### Procedure
`deconvolution/R/extract_z.R`.
1. Load the object; stop if it has no per-sample Z (`:34-38`).
2. `Z <- bp@posterior.initial.cellType@Z`, a sample × gene × cell-type array (`:40`).
3. Write gene and type order (`:43-44`).
4. Check that no two cell types map to the same file name, and delete per-type files left by an earlier
   labelling (`:48-50`; `deconvolution/R/celltype_names.R`).
5. For each type, `get.exp(bp, state.or.type = "type", cell.name = t)` and write the sample × gene matrix
   (`:52-56`). `get.exp` returns `bp@posterior.initial.cellType@Z[,,t]`
   (`vendor/BayesPrism/BayesPrism/R/extract_result.R:44-45`).
6. Write the type-to-file manifest and provenance (`:57-60`).

`deconvolution/build_pseudocells.py`.
1. Read `genes.txt`, `types.txt` and each `predz__<type>.csv`; stop if a file or a gene is missing; reorder
   every matrix to the `genes.txt` order (`:53-70`).
2. Check that all types have the same sample axis (`:76-79`).
3. Stack the per-type matrices; one row per (sample, type); row ID `"<tissue>|<sample>|<safe(type)>"`
   (`:81-90`). Values are stored as 32-bit floats (`:83`).
4. Drop rows whose sum over genes is zero (`:92-94`).
5. Optionally join sample metadata by key (`:96-105`). Off by default (`pipeline/run_stage8.py:107,209-211`);
   the default run did not use it (`provenance_build_pseudocells.json`, `params.meta_tsv` null).
6. Write an AnnData with the same matrix in `.X` and `layers["z_raw"]` (`:107-116`).

### What a pseudo-cell is, precisely
- **Unit.** One (bulk sample, cell type) pair (`deconvolution/build_pseudocells.py:10`). A tissue with S
  samples and T types yields S × T pseudo-cells.
- **Value.** For gene g, the posterior mean number of that sample's reads of g that the initial Gibbs sampler
  allocated to the cell type (8g). Values are continuous, not integers, and are on the scale of the bulk
  library: within a sample, the cell types' values for a gene sum to the bulk count of that gene (each read is
  allocated to exactly one state, `run_gibbs.R:57-59`).
- **Rounding.** None (`deconvolution/build_pseudocells.py:83`).
- **Minimum θ rule.** None in Stage 8. A cell type with θ near zero still yields a pseudo-cell for every
  sample.
- **Minimum counts rule in Stage 8.** Only "not all zero" (`:92-94`).
- **Normalisation.** None in Stage 8 (`:16-19`). In Stage 9 each pseudo-cell is scaled to a fixed total of
  6,400 (`config/pipeline_config.yaml:411-413`; `deconvolution/tokenize_pseudocells.py:115-116,135`), and
  pseudo-cells with fewer than 200 expressed genes are dropped (`deconvolution/tokenize_pseudocells.py:100,176`).
- **Library-size rule downstream.** A block enters the exercise gate and the differential expression only if
  its median per-sample library (sum of Z over genes) is at least 1,000
  (`deconvolution/REFERENCE_RELABEL_PREREG.md:104-112`; `deconvolution/corroborate_summary.py:31-32,61`;
  `deconvolution/R/run_pseudobulk_de.R:92`). This is applied per block, not per pseudo-cell, and belongs to
  the gate and DE sections.
- **Sample identity.** The pseudo-cell's `sample` field is `mix<i>`, the row index of the bulk matrix
  (`deconvolution/R/run_deconvolution.R:102`). The animal ID is recovered later by position.

### Models and software
- BayesPrism 2.2.3 for `get.exp` (as 8g).
- Python `anndata`, `numpy`, `pandas`, `scipy` from the project virtual environment
  (`deconvolution/build_pseudocells.py:37-40`). Versions: the Python environment section.

### Parameters, thresholds and flags
| Name | Value | Where set (path:line) | What it controls | Rationale given |
|---|---|---|---|---|
| Z source slot | `posterior.initial.cellType@Z` | `deconvolution/R/extract_z.R:40`; `extract_result.R:45` | Z from the initial sampler, summed over a type's states | "No Gibbs re-run: get.exp reads the stored posterior" (`extract_z.R:4`) |
| Granularity | sample × cell type | `deconvolution/build_pseudocells.py:10-13,113` | One pseudo-cell per pair | preserves per-sample variation (`:12-13`) |
| Zero-row drop | row sum > 0 | `deconvolution/build_pseudocells.py:92` | Removes empty pseudo-cells | "drop pseudo-cells with no signal" (`:92`) |
| dtype | float32 | `deconvolution/build_pseudocells.py:83` | Storage precision | none stated |
| File-name map | transliterate Greek and accented letters, then replace non-alphanumerics by `_` | `deconvolution/celltype_names.py:19-37`; R twin `deconvolution/R/celltype_names.R` | One file per cell type | earlier non-injective map overwrote one KIDNEY type's Z (`deconvolution/celltype_names.py:3-11`) |
| `--with-pheno` | off | `pipeline/run_stage8.py:209-211` | Metadata join at build time | "a plain key merge here can mis-join" (`:210-211`) |
| `pseudocell_target_sum` | 6400 (applied in Stage 9, not here) | `config/pipeline_config.yaml:413` | Per-pseudo-cell total before tokenization | calibrated to the model's corpus (`:411-412`) |

### Statistical procedures
None. This step copies and reshapes.

### Checks, validation and their results
| Check | Result | Where |
|---|---|---|
| One pseudo-cell per (sample, type); all-zero block dropped; gene order re-aligned; values copied unchanged | unit tests | `tests/test_pseudocells.py:72-84` |
| Metadata join on sample key | unit test | `tests/test_pseudocells.py:87-91` |
| A missing per-type file stops the build | unit test | `tests/test_pseudocells.py:94-98` |
| Greek-letter labels stay distinct; colliding labels are refused; stale files purged | unit tests | `tests/test_pseudocells.py:19-54` |
| Every pseudo-cell is either tokenized or counted as dropped | data test | `tests/test_v2_outputs.py:43-47` |
| Pseudo-cell totals | 9,232 = sum of 16 `summary.txt`; 197 = sum of cell types | files above |

Whether these tests pass today was not checked (running the suite is outside the audit's limits).

### Deviations, known defects and caveats
1. **Stale gene-space label written into every output.** `build_pseudocells.py` records
   `"gene_space": "rel-113 ENSRNOG"` in the AnnData and `gene_space=rel-113_ENSRNOG` in `summary.txt`
   (`deconvolution/build_pseudocells.py:15,114,139`). The pinned annotation is Ensembl 115 / GRCr8
   (`config/pipeline_config.yaml:36-38`). The label is text only and does not affect values, but it is wrong
   in the artefact.
2. **Sample identity is positional.** The link from pseudo-cell to animal depends on `bulk_samples.tsv`
   keeping the row order of `bulk.mtx` (`deconvolution/R/run_deconvolution.R:102`;
   `deconvolution/pheno_merge_test.py:84`). No checksum ties the two files together in the pseudo-cell record.
3. **No low-fraction screen.** Cell types that BayesPrism estimates at essentially zero still produce 50
   pseudo-cells each. The library rule removes 46 of the 197 blocks later (197 - 151). A reader should not
   read "9,232 pseudo-cells" as 9,232 informative profiles.
4. **Earlier defect, fixed.** The non-injective file-name map that destroyed one KIDNEY cell type's Z is
   replaced by the shared map with an explicit collision check (`deconvolution/R/extract_z.R:18-19,48`;
   `deconvolution/build_pseudocells.py:43-49`; commit `f4cbf12`, 2026-07-13). Verified in the current code and
   in the current `pred_z` folder, which has 15 distinct KIDNEY files for 15 types.
5. **The provenance of the gate merge that records "151" is overwritten.**
   `gate_rcv10/provenance_corroborate_summary.json` holds the command of the last sensitivity run
   (`--min-libsize 0`), not the default run, because all three runs write the same file name. The 151 figure
   itself is in the table. (Gate section; noted here because this section quotes the count.)

### Methods references
- BayesPrism `get.exp`: Chu et al. 2022, as above.

---

## Stage 8i - Known-truth mixture panel (holdout and cross-study sets, six methods)

### Research role
- **Question.** Does each of the 15 production references recover the cell-type fractions of mixtures whose
  composition is known?
- **Claim that rests on it.** "Every current reference was validated against pseudobulk mixtures of known
  composition" (`manuscript/motrpac_genecompass.tex:506`), and the reading that θ can be used differentially.
- **How it is checked.** Simulated mixtures of single cells; per-type correlation of estimated against true
  RNA fraction across mixtures; a pre-registered floor.
- **Verdict.** All 19 sets pass the registered floor for BayesPrism
  (`deconvolution/MIXTURE_PANEL_REFS4ALL_RESULTS.md:13-16`). Absolute fractions are not secure in two places
  (kidney cross-study, liver hepatocytes) (`:188-193`).

### Inputs and outputs
- Inputs: the `reference_dir` of each runnable tissue (15 references for 16 tissues;
  `deconvolution/build_mixture_panel.sh:20-36`); four source studies for the cross sets (`:39-44`); the
  override table `deconvolution/reference/mixture_panel_harmonization_overrides.tsv` (12 rows).
- Outputs: `data/deconvolution/mixture_panel_refs4all/<SET>/{reference,mixtures,results,scores}/`
  (`deconvolution/build_mixture_panel.sh:14`); summary tables in
  `data/deconvolution/mixture_panel_refs4all/summary/`, copied to the committed
  `deconvolution/mixture_panel_record/` (7 files).
- Sets: 15 holdout, 4 cross (`deconvolution/build_mixture_panel.sh:20-44`).
- Per-type tests: 193 for BayesPrism (170 holdout + 23 cross), counted from
  `deconvolution/mixture_panel_record/panel_per_type.tsv`.

### Procedure
Build (`deconvolution/build_mixture_panel.sh`, calling `deconvolution/make_pseudobulk.py`).
1. **Holdout sets** (`build_mixture_panel.sh:50-56`): `--mode holdout --ref-source <reference_dir>
   --holdout-frac 0.30`.
   - Load the built reference with its labels (`make_pseudobulk.py:120-134,225-226`).
   - Within each cell type, shuffle the cells and send `round(0.30 × n)` to the mixture pool
     (`:233-237`).
   - Re-apply the minimum-cells-per-state rule to the remaining 70% and write it as the set's reference
     (`:238-239`). A type can be lost here.
2. **Cross sets** (`build_mixture_panel.sh:57-65`): `--mode cross --source-study <S> --ref-dir <reference_dir>`.
   - Load the source study, apply its label scheme, clean (`make_pseudobulk.py:249-251`).
   - Map source labels to reference types automatically: exact match; else the reference type sharing the
     most whole word tokens; else, if the reference has an "immune" type, immune-like labels go to it; else
     unmapped and dropped (`:85-117`).
   - Apply the reviewed overrides, which may only un-map a label (`:253-262`; the table has an empty
     `ref_type` in all 12 rows).
3. Keep reference types with at least 20 pool cells (`make_pseudobulk.py:273-275`).
4. **Simulate** (`make_pseudobulk.py:137-185`). For each of 50 mixtures:
   - draw type proportions from a Dirichlet with all parameters equal to α = 1 (`:157`);
   - cells per type = `round(proportion × 1000)` (`:158`);
   - draw that many cells of the type from the pool, without replacement if the pool is large enough and
     with replacement otherwise (`:166`);
   - sum the raw counts (`:167-168`).
   - Record two truths per type: the fraction of cells and the fraction of total counts ("rnafrac")
     (`:171-177`).

Deconvolve and score (driver scripts are local, not committed: `slurm/analysis/mixpanel_{bp,omni,dwls,cx}.slurm`,
`mixpanel_sets.sh`).
5. BayesPrism: `deconvolution/R/run_deconvolution.sh <ref> <set>/mixtures <set>/results` with `N_CORES=16`.
   A holdout set is deconvolved against its own 70% split; a cross set against the full production reference
   (`slurm/analysis/mixpanel_sets.sh`). The gene filters of 8g apply.
6. Other methods: 8j.
7. Score each method with `deconvolution/score_validation.py --est-file ... --tag <method>` against
   `rnafrac`.
8. Summarise with `deconvolution/mixture_panel_summary.py`.

### Models and software
BayesPrism as in 8g (16 cores; provenance in each set's `results/provenance_run_deconvolution.json`: commits
`37b9099`, `3b5a65d`, `bddcdae`, clean trees, 50 mixtures each). The five other methods: 8j.

### Parameters, thresholds and flags
| Name | Value | Where set (path:line) | What it controls | Rationale given |
|---|---|---|---|---|
| Mixtures per set | 50 | `deconvolution/build_mixture_panel.sh:16`; default `make_pseudobulk.py:211` | Sample size of each correlation | as the earlier panel (`MIXTURE_PANEL_REFS4ALL_PREREG.md:39-40`) |
| Cells per mixture | 1,000 | `build_mixture_panel.sh:16`; `make_pseudobulk.py:212` | Mixture size | same |
| Dirichlet α | 1.0 | `build_mixture_panel.sh:16`; `make_pseudobulk.py:213` | Composition spread (uniform over the simplex) | same |
| Seed | 1 | `build_mixture_panel.sh:16`; `make_pseudobulk.py:215,217` | One NumPy generator drives the split and the mixtures | same |
| Holdout fraction | 0.30 | `build_mixture_panel.sh:55`; `make_pseudobulk.py:210` | Share of each type's cells sent to the pool | same |
| Minimum pool cells | 20 | `build_mixture_panel.sh:16`; `make_pseudobulk.py:214,275` | A type with fewer pool cells is not benchmarked | registered (`PREREG:39-40`) |
| Minimum cells per state | 20 | `deconvolution/build_reference.py:345`; applied `make_pseudobulk.py:238` | Re-cleaning of the 70% reference | registered (`PREREG:19`) |
| Cross sources | KIDNEY: GSE137869, no condition filter; CORTEX: GSE213978, "sham"; HIPPOC: GSE307917, "naïve,sham"; MUSCLE: GSE255196, no filter | `build_mixture_panel.sh:40-43` | Where cross mixtures come from | "the ones the earlier panel used, wherever that source is still a different study" (`PREREG:22-28`) |
| Label scheme per cross set | none, brain, brain, muscle | `build_mixture_panel.sh:40-43` | Canonicalises source labels | `PREREG:30` |
| Truth basis | `rnafrac` | `score_validation.py:29` (default) | Score against the RNA fraction, which θ estimates | `score_validation.py:6-8` |
| Pass floor | BayesPrism median per-type r ≥ 0.85 | `MIXTURE_PANEL_REFS4ALL_PREREG.md:67`; `mixture_panel_summary.py:92` | Verdict per set | pre-registered |
| Drop flag | > 0.05 below the earlier panel | `PREREG:70-71`; `mixture_panel_summary.py:110` | Named if exceeded | pre-registered |
| "Within" margin | 0.02 of the best method | `PREREG:73`; `mixture_panel_summary.py:120` | Descriptive count | pre-registered |

### Statistical procedures
Exact definitions in `deconvolution/score_validation.py`. Let t index cell types present in both truth and
estimate, and m index the 50 mixtures.
- **Alignment.** Estimate rows are relabelled `mix1 ... mixN` by position (`:50`). If the lengths differ, both
  are cut to the shorter (`:63`).
- **Per-type Pearson r.** `pearsonr(y_t, ŷ_t)` over the 50 mixtures; set to missing if the estimate is
  constant (`:64`). Spearman's ρ likewise (`:65`).
- **Per-type RMSE.** `sqrt(mean((y_t - ŷ_t)^2))` (`:66`). **Per-type bias.** `mean(ŷ_t - y_t)` (`:67`).
- **Pooled r** (`overall_pooled_pearson`). One Pearson correlation over all (type, mixture) pairs stacked
  (`:76-77`). **Pooled RMSE** over the same pairs (`:78`).
- **Median per-type r** (`median_pearson`). The median of the non-missing per-type r (`:86-88`). This is the
  headline. **Macro r** is their mean (`:87`).
- **"Separable" pooled r.** The pooled r after removing the single type with the highest true mean
  (`:31-34,89-94`). Reported in the score files, not used in the panel reading.
- **Units of replication.** 50 simulated mixtures per set, all drawn with one seed from one pool. There are
  no confidence intervals, no tests and no multiplicity control; the reading is a threshold on a point
  estimate.
- **Pre-registration.** `deconvolution/MIXTURE_PANEL_REFS4ALL_PREREG.md` (commit `780b149`, 2026-11-04,
  "before any mixture was built"). It fixes the sets, sources, simulation settings, methods, scoring and
  reading, and states that nothing will change after results are seen (`:73-74`). The harmonization overrides
  were committed before any deconvolution (`37b9099`; `MIXTURE_PANEL_REFS4ALL_RESULTS.md:7-8`).

### Checks, validation and their results
BayesPrism, median per-type r and pooled r, from `deconvolution/mixture_panel_record/panel_median.tsv` and
`panel_pooled.tsv` (same numbers in `summary.txt` and `MIXTURE_PANEL_REFS4ALL_RESULTS.md:23-69`). "-" means
the method could not run on a single-sample reference.

| Set | BayesPrism median / pooled | CIBERSORTx | DWLS | MuSiC | SCDC | Bisque | Types benchmarked / in built reference |
|---|---|---|---|---|---|---|---|
| BAT_holdout | 0.998 / 0.957 | 0.990 / 0.895 | 0.996 / 0.958 | 0.997 / 0.970 | 0.991 / 0.624 | 0.959 / 0.061 | 6 / 6 |
| BLOOD_holdout | 0.989 / 0.981 | 0.987 / 0.941 | 0.979 / 0.832 | 0.968 / 0.807 | 0.925 / 0.369 | 0.771 / 0.465 | 13 / 13 |
| CORTEX_holdout | 0.999 / 0.999 | 0.997 / 0.997 | 0.938 / 0.631 | 0.932 / 0.466 | 0.954 / 0.786 | 0.982 / 0.604 | 8 / 12 |
| HEART_holdout | 0.997 / 0.973 | 0.993 / 0.932 | 0.987 / 0.853 | 0.986 / 0.789 | 0.927 / 0.516 | 0.969 / 0.578 | 16 / 16 |
| HIPPOC_holdout | 0.998 / 0.997 | 0.993 / 0.981 | 0.980 / 0.864 | 0.972 / 0.794 | 0.970 / 0.843 | 0.965 / 0.545 | 10 / 11 |
| HYPOTH_holdout | 0.995 / 0.964 | 0.993 / 0.896 | 0.993 / 0.970 | - | 0.993 / 0.944 | - | 10 / 13 |
| KIDNEY_holdout | 0.995 / 0.974 | 0.993 / 0.893 | 0.990 / 0.916 | 0.982 / 0.685 | 0.979 / 0.781 | 0.951 / 0.332 | 15 / 15 |
| LIVER_holdout | 0.992 / 0.841 | 0.990 / 0.808 | 0.985 / 0.896 | 0.988 / 0.899 | 0.987 / 0.859 | 0.929 / 0.262 | 7 / 8 |
| LUNG_holdout | 0.991 / 0.948 | 0.990 / 0.959 | 0.985 / 0.822 | 0.979 / 0.701 | 0.962 / 0.771 | 0.671 / 0.421 | 21 / 21 |
| MUSCLE_holdout | 0.997 / 0.958 | 0.996 / 0.874 | 0.995 / 0.979 | 0.995 / 0.977 | 0.994 / 0.962 | 0.969 / 0.212 | 8 / 9 |
| OVARY_holdout | 0.995 / 0.995 | 0.993 / 0.986 | 0.966 / 0.564 | - | 0.946 / 0.191 | - | 16 / 18 |
| SMLINT_holdout | 0.992 / 0.979 | 0.984 / 0.938 | 0.966 / 0.722 | 0.965 / 0.739 | 0.950 / 0.709 | 0.874 / 0.243 | 13 / 13 |
| SPLEEN_holdout | 0.990 / 0.987 | 0.991 / 0.950 | 0.964 / 0.781 | - | 0.960 / 0.794 | - | 13 / 14 |
| TESTES_holdout | 0.999 / 0.997 | 0.996 / 0.983 | 0.983 / 0.899 | 0.982 / 0.902 | 0.977 / 0.944 | 0.983 / 0.253 | 5 / 6 |
| WATSC_holdout | 0.994 / 0.953 | 0.984 / 0.915 | 0.976 / 0.845 | 0.984 / 0.855 | 0.975 / 0.846 | 0.936 / 0.199 | 9 / 13 |
| CORTEX_cross | 0.983 / 0.896 | 0.973 / 0.866 | 0.913 / 0.435 | 0.926 / 0.222 | 0.965 / 0.242 | 0.961 / 0.404 | 6 / 12 |
| HIPPOC_cross | 0.996 / 0.828 | 0.997 / 0.783 | 0.951 / 0.642 | 0.975 / 0.737 | 0.985 / 0.819 | 0.961 / 0.374 | 6 / 11 |
| KIDNEY_cross | 0.973 / 0.163 | 0.944 / 0.659 | 0.708 / 0.158 | 0.588 / 0.034 | 0.737 / 0.046 | 0.768 / 0.318 | 7 / 15 |
| MUSCLE_cross | 0.986 / 0.830 | 0.944 / 0.756 | 0.934 / 0.715 | 0.962 / 0.708 | 0.965 / 0.698 | 0.901 / 0.069 | 4 / 9 |

Other recorded results.
- **Pass.** All 19 BayesPrism medians are ≥ 0.85; range 0.989-0.999 holdout, 0.973-0.996 cross
  (`MIXTURE_PANEL_REFS4ALL_RESULTS.md:13-16`).
- **Per-type.** Of BayesPrism's 193 per-type r, 180 are ≥ 0.95, 186 are ≥ 0.90 and 6 are < 0.85 (recounted
  from `panel_per_type.tsv`): LIVER_holdout hepatocytes 0.786 (true mean 0.134, estimated 0.231);
  OVARY_holdout erythroid cells 0.410; HIPPOC_cross excitatory neurons 0.737 (true 0.172, estimated 0.008);
  KIDNEY_cross distal convoluted tubule cells 0.694 (true 0.175, estimated 0.003), immune cells 0.696,
  monocytes 0.583 (true 0.118, estimated 0.572).
- **Kidney cross-study.** BayesPrism, DWLS, MuSiC and SCDC each put more than half the mass into monocytes;
  BayesPrism's pooled r is 0.163 (`RESULTS:118-121`).
- **Coverage.** 170 of 188 reference types are benchmarked in holdout (`RESULTS:140-152`;
  `panel_coverage.tsv`). Three types are lost when the reference is split (CORTEX vascular leptomeningeal
  cells, TESTES Sertoli cells, WATSC mural cells); 15 have fewer than 20 pool cells.
- **Hotspot types.** All nine exercise-hotspot types are benchmarked in holdout, at BayesPrism r 0.983-0.999;
  none is benchmarked in a cross set (`RESULTS:129-136`; values recounted from `panel_per_type.tsv`).
- **Against the earlier panel.** Eight comparable pairs, changes from -0.010 to +0.122; no drop over 0.05
  (`RESULTS:154-171`; `panel_vs_earlier.tsv`).
- **Leakage check.** In all 15 holdout sets, built-reference cells = 70% split + pool + cells dropped by
  re-cleaning (`RESULTS:85-86`). The script that made this check was **NOT FOUND** in the committed code
  (searched `deconvolution/` for a cell-accounting check; `mixture_panel_summary.py` does not do it).

### Deviations, known defects and caveats
1. **CIBERSORTx was run after the other methods' results had been read** (2026-11-08), because its
   credentials were wrongly reported missing on 11-04 (`RESULTS:73-77`). The registered settings were used.
   The pre-registration allowed "run only if its credentials are present at run time" (`PREREG:55`); they
   were present.
2. **The summary script was edited after results** to add descriptive outputs (`RESULTS:87-89`). No
   threshold changed, by the document's account; the edit history was not audited line by line.
3. **Types a method does not estimate, or estimates as a constant, leave its median and pooled r.**
   `score_validation.py` scores only types present in the estimate (`:54`), although its message says the
   missing ones are "set to 0" (`:57`); a constant estimate gets a missing r (`:64`) and is dropped from the
   median (`:86-88`). MuSiC's medians therefore omit 4 types and the HIPPOC_cross medians of MuSiC, SCDC and
   CIBERSORTx omit excitatory neurons (`panel_scoring_notes.tsv`; `RESULTS:91-101`). This favours those
   methods, not BayesPrism. The printed message is wrong; the documented behaviour is what ran.
4. **Compositions are not tissue-like.** With α = 1 over K benchmarked types, every type averages 1/K of the
   cells. In LIVER_holdout, hepatocytes have a true mean RNA fraction of 0.134
   (`panel_per_type.tsv`), far from a real liver. The panel tests whether a type's fraction tracks across a
   wide artificial range. It does not test recovery at realistic compositions, nor sensitivity to the small
   between-animal differences an exercise effect would produce. (Auditor's observation; the documents state
   only that holdout is "the easier test", `PREREG:76-78`.)
5. **Small pools are resampled heavily.** A type needs only 20 pool cells (`make_pseudobulk.py:275`) and is
   drawn with replacement when a mixture asks for more (`:166`). In LIVER_holdout the pool has 30 vascular
   endothelial cells, 25 cholangiocytes and 54 NK cells
   (`data/deconvolution/mixture_panel_refs4all/LIVER_holdout/mixtures/pool_composition.tsv`), while single
   mixtures ask for several hundred of one type (e.g. `mix1`, cell fraction 0.436 for vascular endothelial
   cells in `true_fractions.tsv`). Such a type's contribution is then close to a fixed profile times a
   scalar, which makes it easier to recover. Mixtures also share cells with each other, so the 50 points of a
   correlation are not independent draws of distinct cells.
6. **Cross-set conditions rely on the corpus's `condition_resolved` field** (`make_pseudobulk.py:196,249`;
   `build_mixture_panel.sh:41-42`), and KIDNEY_cross and MUSCLE_cross apply no condition filter at all
   (`build_mixture_panel.sh:40,43`), so all arms of those source studies are pooled into the mixtures.
7. **Driver scripts are not in the repository.** The SLURM scripts that set `N_CORES`, the 2,000-cell cap
   and the method lists are under the ignored `slurm/` folder. The committed record of those settings is
   the pre-registration text (`PREREG:50-58`).
8. **No provenance files.** `make_pseudobulk.py`, `score_validation.py` and `mixture_panel_summary.py` write
   no provenance record (no `write_provenance` call in any of them). The build commit is known only from the
   job log header (`slurm/analysis/mixpanel_build.slurm`).
9. **What the certificate does not cover** (stated in `RESULTS:194-197`): 18 of 188 types; every hotspot
   type across studies; TESTES Sertoli cells, whose overestimate in production is therefore untested.
10. **No unit tests** exist for `make_pseudobulk.py`, `score_validation.py` or `mixture_panel_summary.py`
    (`tests/` has none that import them).

Related reference-specific checks, registered and reported elsewhere (reference section): BLOOD composite
recovery on simulated mixtures, per-type r 0.91-0.96 against a floor of 0.9
(`deconvolution/REFERENCE_RELABEL_PREREG.md:311-315,328-330`); SPLEEN and OVARY holdout recovery ("R2") and
the HEART cross-study check against GSE307551, cardiomyocyte r 0.944 (`:583-600,649-655`).

Earlier, superseded validation: the June/July panel on the earlier references
(`deconvolution/OMNIDECONV_RESULTS.md:52-142`;
`data/deconvolution/validation_v2/`) and the pilot liver stages V0/V1/V2
(`data/deconvolution/validation/V0_liver` etc.; `config/pipeline_config.yaml:396-398`). Per-type Z scoring
with `deconvolution/compute_true_z.py` and `deconvolution/score_z.py` exists only for four pilot liver stages
(`data/deconvolution/validation/V2_*/scores/metrics_z.tsv`).

### Methods references
- Benchmark design and method panel: Dietrich A, Merotto L, Pelz K, et al. omnideconv. Genome Biology
  2026;27:6 (`manuscript/motrpac_genecompass.tex:1495-1497`).
- BayesPrism: Chu et al. 2022, as above.

---

## Stage 8j - omnideconv cross-check methods and their settings

### Research role
- **Question.** Do independent reference-based methods agree with BayesPrism on the same mixtures?
- **Claim.** BayesPrism is best or within 0.02 of the best on the per-type axis
  (`deconvolution/MIXTURE_PANEL_REFS4ALL_RESULTS.md:175-179`). The methods are a cross-check on fractions
  only; none of them supplies the per-cell-type expression the pipeline needs
  (`deconvolution/OMNIDECONV_RESULTS.md:215-216`).
- **Verdict.** Agreement on the per-type axis; on pooled r BayesPrism is best on 13 of 19 sets
  (`RESULTS:180-183`).

### Inputs and outputs
Same reference and mixture folders as BayesPrism (`deconvolution/R/run_omnideconv.R:5-10`). Output: one
`results/fractions_<method>.csv` per method (`:264-265`), scored as in 8i.

### Procedure (`deconvolution/R/run_omnideconv.R`)
1. Read the reference; cell types from `cell_type`; batch IDs from `cells_meta$sample` (`:50-58`).
2. Apply the same four gene filters as BayesPrism (`:60-90`), driven by the same environment variables
   (`deconvolution/R/run_omnideconv.sh:18-22`).
3. **Cap the reference** at `REF_MAX_CELLS_PER_TYPE` cells per type, by seeded random sampling without
   replacement, and print the cells dropped per type (`:99-116`).
4. Transpose to genes × cells and genes × samples (`:119-125`). Raw counts are passed to every method; the
   script applies no CPM or TPM normalisation.
5. Run each requested method inside `tryCatch`, so that one failure does not stop the others (`:219-270`).
6. Two run-time patches to installed packages (`assignInNamespace`), leaving `R_libs/` untouched:
   - CIBERSORTx container command: `apptainer exec --cleanenv --home <writable dir>` in place of
     omnideconv's `--no-home -c`, which broke the container's internal R (`:163-186`).
   - DWLS `buildSignatureMatrixMAST`: a guard so that a cell type with fewer than 2 marker genes counts as 0
     markers instead of aborting the run (`:189-217`).

### Models and software
| Method | Package and version | Pin | Call made |
|---|---|---|---|
| omnideconv (wrapper) | 0.1.1, patched at install (doc-only imports demoted) | `deconvolution/setup/r_packages.yaml:56` (GitHub `omnideconv/omnideconv` `2dcd4d83`); lock file `r_packages.lock.tsv:194` records it as "local" with no hash | `deconvolution/R/run_omnideconv.R:226-249` |
| DWLS | 0.5.2, `omnideconv/DWLS` `9a767ced` | `r_packages.yaml:43`; `r_packages.lock.tsv:86` | `build_model(method="dwls", dwls_method, ncores, verbose=TRUE)` then `deconvolute(model=sig, method="dwls")` (`run_omnideconv.R:226-230`) |
| MAST (used by DWLS) | 1.32.0 | `r_packages.lock.tsv:171` | inside DWLS |
| MuSiC | 0.3.0, `omnideconv/MuSiC` `7b18b034` | `r_packages.yaml:42`; `r_packages.lock.tsv:188` | `deconvolute(method="music", single_cell_object, cell_type_annotations, batch_ids)` (`run_omnideconv.R:246-249`) |
| SCDC | 0.0.0.9000, `omnideconv/SCDC` `dc2b0ef4` | `r_packages.yaml:44`; `r_packages.lock.tsv:284` | same generic call |
| Bisque | BisqueRNA 1.0.5, `cozygene/bisque` `8640250d` | `r_packages.yaml:46`; `r_packages.lock.tsv:33` | same generic call |
| CIBERSORTx | container `fractions_latest.sif`, pulled from `docker://cibersortx/fractions` | file `data/deconvolution/cibersortx/fractions_latest.sif` (358,105,088 bytes, dated 2026-06-28); pull command in `R_libs/omnideconv/R/omnideconv.rdb::setup_apptainer_container`. **No version tag or digest is recorded anywhere** (NOT FOUND in `r_packages.yaml`, `SETUP.md`, the R script). | `build_model(method="cibersortx", container="apptainer", container_path)` then `deconvolute(..., rmbatch_B_mode=FALSE, rmbatch_S_mode=FALSE)` (`run_omnideconv.R:237-244`) |

### Parameters, thresholds and flags
Settings the project chooses.

| Name | Value | Where set (path:line) | What it controls | Rationale given |
|---|---|---|---|---|
| `OMNIDECONV_METHODS` | default `music,dwls,scdc,bisque`; panel ran `music,scdc,bisque`, `dwls` and `cibersortx` in three job arrays | `run_omnideconv.R:41`; `run_omnideconv.sh:23`; local `slurm/analysis/mixpanel_{omni,dwls,cx}.slurm` | Which methods run | DWLS split out for memory/time |
| `REF_MAX_CELLS_PER_TYPE` | default 2000; 2000 in the mixture panel; 0 (no cap) in the SimBu battery | `run_omnideconv.R:99`; `MIXTURE_PANEL_REFS4ALL_PREREG.md:52-54`; `SIMBU_BIAS_REFS4ALL_PREREG.md:22` | Cells per type in the signature reference | "signature-building methods (esp. DWLS/MAST) blow memory ... a signature is a stable per-type estimate" (`run_omnideconv.R:93-98`) |
| Cap seed | `set.seed(1)` | `run_omnideconv.R:101` | Which cells are kept | reproducible, logged (`:96`) |
| `DWLS_METHOD` | `"mast"` | `run_omnideconv.R:146` | Signature-building route | "identical to the unforwarded default" (`:143-145`) |
| `N_CORES` | 32 (MuSiC/SCDC/Bisque job), 4 (DWLS job), 8 (CIBERSORTx job; unused by it) | local `slurm/analysis/mixpanel_*.slurm`; used at `run_omnideconv.R:228` only | DWLS parallelism | DWLS out-of-memory at high core counts (`deconvolution/OMNIDECONV_RESULTS.md:100-103`) |
| CIBERSORTx batch correction | B-mode off, S-mode off | `run_omnideconv.R:244` | No cross-platform batch correction | "apples-to-apples panel" (`:151`) |
| CIBERSORTx credentials | from `CIBERSORTX_EMAIL` and `CIBERSORTX_TOKEN` | `run_omnideconv.R:157-160` | Licence check at run time | token never in code or logs (`:150-151,235-236`) |
| Gene filters | as BayesPrism (8g) | `run_omnideconv.R:60-90` | Same gene universe | `:15-18` |

Settings inherited from the wrappers (not passed by the project; read from the installed packages).

| Method | Inherited settings | Source |
|---|---|---|
| DWLS signature | `diff_cutoff = 0.5`, `pval_cutoff = 0.01` | `R_libs/omnideconv/R/omnideconv.rdb::build_model_dwls`; `R_libs/DWLS/R/DWLS.rdb::buildSignatureMatrixMAST` (`diff.cutoff = 0.5`, `pval.cutoff = 0.01`) |
| DWLS solver | `dwls_submethod = "DampenedWLS"` (first of `DampenedWLS, OLS, SVR`), i.e. `DWLS::solveDampenedWLS` per sample | `R_libs/omnideconv/R/omnideconv.rdb::deconvolute_dwls` |
| MuSiC | `MuSiC::music_prop` with `markers = NULL`, `select.ct = NULL`, `cell_size = NULL`, `ct.cov = FALSE`, `iter.max = 1000`, `nu = 1e-04`, `eps = 0.01`, `centered = FALSE`, `normalize = FALSE`; the result taken is `Est.prop.weighted` | `R_libs/omnideconv/R/omnideconv.rdb::deconvolute_music`, `::deconvolute` |
| SCDC | single reference: `SCDC_prop` if the reference has more than one sample, `SCDC_prop_ONE` if it has one; `iter_max = 1000`, `nu = 1e-04`, `epsilon = 0.01`, `weight_basis = TRUE`, `quality_control = FALSE`, `Transform_bisque = FALSE` | `R_libs/omnideconv/R/omnideconv.rdb::deconvolute_scdc` |
| Bisque | `BisqueRNA::ReferenceBasedDecomposition` with `markers = NULL`, `use.overlap = FALSE`, `old.cpm = TRUE` | `R_libs/omnideconv/R/omnideconv.rdb::deconvolute_bisque` |
| CIBERSORTx signature matrix | `--single_cell TRUE`, `--G.min 300`, `--G.max 500`, `--q.value 0.01`, `--filter FALSE`, `--k.max 999`, `--remake FALSE`, `--replicates 5`, `--sampling 0.5`, `--fraction 0.75` | `run_omnideconv.R:175-177`; `R_libs/omnideconv/R/omnideconv.rdb::build_model_cibersortx` (`k_max = 999`), `::get_signature_matrix_options` |
| CIBERSORTx fractions | `--perm 0`, `--QN FALSE`, `--absolute FALSE`, `--abs_method sig.score`, `--rmbatchBmode FALSE`, `--rmbatchSmode FALSE` | `R_libs/omnideconv/R/omnideconv.rdb::get_cell_fractions_options` |
| CIBERSORTx inputs | the single-cell and bulk matrices are written as tab-separated count tables with no normalisation | `R_libs/omnideconv/R/omnideconv.rdb::transform_and_save_single_cell`, `::transform_and_save_bulk` |

Sample requirements.
- MuSiC and Bisque need a reference with more than one sample (batch). On HYPOTH, OVARY and SPLEEN, whose
  references have one, they are reported "not applicable" (`MIXTURE_PANEL_REFS4ALL_PREREG.md:57-58`;
  `MIXTURE_PANEL_REFS4ALL_RESULTS.md:80-81`). The requirement is enforced inside the method packages; the
  omnideconv wrappers check only for at least two **bulk** samples
  (`R_libs/omnideconv/R/omnideconv.rdb::deconvolute_music`, `::deconvolute_bisque`).
- MuSiC drops cell types that occur in only one reference sample (`RESULTS:95-96`).

### Statistical procedures
None beyond 8i's scoring. Each method returns point estimates of fractions.

### Checks, validation and their results
Results per method: table in 8i. Run notes (`MIXTURE_PANEL_REFS4ALL_RESULTS.md:73-89`): no DWLS task ran out
of memory (peak 269 GiB of 480 GiB), so the registered retry was not used; HYPOTH, OVARY and TESTES needed no
downsampling under the 2,000-cell cap.

### Deviations, known defects and caveats
1. **The method settings differ from the benchmark paper's, as the project's own plan lists them.** The plan
   called for DWLS `mast_optimized` with `pval_cutoff = 0.05`, TPM bulk for MuSiC, SCDC and CIBERSORTx, CPM
   single-cell input for CIBERSORTx, and BayesPrism on cell types only without the final update for
   matched-assay runs (`deconvolution/OMNIDECONV_BENCHMARK_PLAN.md:53-66,286-300`). What ran: DWLS `mast`
   with `pval_cutoff = 0.01`; raw counts for every method; BayesPrism with cell states and the final update.
   The missing rat gene-length table for TPM is acknowledged as open item D1 (`:306-309`) and listed as "Not
   run: ... bulk-TPM (D1)" (`deconvolution/OMNIDECONV_RESULTS.md:257-258`). For mixtures summed from
   single-cell counts the length normalisation matters less than for real bulk, but the comparison is not a
   like-for-like replication of the paper's settings, and no document states the DWLS deviation.
2. **BayesPrism and the other methods do not see the same reference.** BayesPrism uses every cell; the others
   use at most 2,000 per type (`MIXTURE_PANEL_REFS4ALL_PREREG.md:52-54`). This is registered and logged.
3. **The CIBERSORTx container is unpinned and its internal sampling is unseeded.** The image is whatever
   `docker://cibersortx/fractions` served on 2026-06-28. Its signature build uses 5 replicates at a sampling
   rate of 0.5 (table above); no seed is passed (none exists in the wrapper's options). Run-to-run
   reproducibility of the CIBERSORTx column was not tested.
4. **No provenance record.** `run_omnideconv.R` does not call `write_provenance`; the commit, the cap applied
   and the package versions of a given `fractions_<method>.csv` are known only from job logs.
5. **Installed packages are patched in memory at run time** (`run_omnideconv.R:163-217`). The patches are in
   the committed script, but a reader comparing against upstream DWLS or omnideconv behaviour should know.
6. **Failures are swallowed by design** (`run_omnideconv.R:223-261`); the script stops only if every method
   fails (`:274`). A missing `fractions_<method>.csv` is the only sign. The summary distinguishes "n/a" from
   "fail" by document, not by code (`deconvolution/mixture_panel_summary.py:13`).

### Methods references
- omnideconv and the benchmark: Dietrich et al. 2026, as above.
- CIBERSORTx: Newman AM et al., Nat Biotechnol 37, 773-782 (2019); CIBERSORT: Newman AM et al., Nat Methods
  12, 453-457 (2015) (as cited in `deconvolution/README.md:133-135`; not in the manuscript bibliography).
- DWLS: Tsoucas D et al., Nat Commun 2019 [verified against Crossref; see REFERENCES.md]. MuSiC: Wang X et al., Nat Commun 2019 [verified against Crossref; see REFERENCES.md]. SCDC:
  Dong M et al., Brief Bioinform 2021 [verified against Crossref; see REFERENCES.md]. Bisque: Jew B et al., Nat Commun 2020 [verified against Crossref; see REFERENCES.md]. MAST: Finak G
  et al., Genome Biol 2015 [verified against Crossref; see REFERENCES.md]. None of these is cited in the code, the docs read, or the manuscript
  bibliography.

---

## Stage 8k - SimBu mRNA-content-bias battery (protocol and registered readings)

Results on the 15 current references are written up separately in
`deconvolution/SIMBU_BIAS_REFS4ALL_RESULTS.md`, committed during this audit (`8249506`) with its record folder
`deconvolution/simbu_record_refs4all/`. This block describes the protocol, the registered readings and July's
results on the earlier references. Read that file for the current results; they are not audited here.

### Research role
- **Question.** Cell types differ in how much mRNA a cell contains. BayesPrism estimates the fraction of
  reads, not of cells. How much does each method's error grow when mixtures are weighted by mRNA content, and
  which cell type carries the distortion in each tissue?
- **Claim that rests on it.** The project's reporting policy: differential claims and embedding-based results
  are stated directly; absolute abundances and expression changes of the mRNA-rich parenchyma are gated
  (`manuscript/motrpac_genecompass.tex:534-540`; `deconvolution/OMNIDECONV_RESULTS.md:213-215`).
- **Verdict so far.** July (earlier references): MuSiC and SCDC degrade least, BayesPrism and CIBERSORTx most.
  The header note in `deconvolution/OMNIDECONV_RESULTS.md:7-15` says the corrector ranking is **not
  reproduced** on the current references, while BayesPrism still degrades on all 15; see the results file.

### Inputs and outputs
- Inputs: the same 15 references (`deconvolution/omnideconv_bench/build_simbu_battery.sh:21-37`); the builder
  stops if its table differs from the ready references in `tissue_references.yaml` (`:40-52`).
- Outputs: `data/deconvolution/simbu_bench_refs4all/<CODE>/{bias,nobias}/{reference,mixtures,results,scores}/`
  and `.../doseresp/<CODE>/a<alpha>/...` (`:6-7,13`); summaries from `simbu_summary.py` in `.../summary/`.
- July outputs: `data/deconvolution/simbu_bench/` (12 set folders and `doseresp/`).
- Counts: 30 battery sets and 12 dose-response sets (`slurm/analysis/simbu16_sets.sh`, local).

### Procedure
Simulation (`deconvolution/omnideconv_bench/simulate_simbu.R`, via `simulate_simbu.sh`).
1. Read the reference (`:63-69`).
2. `set.seed(seed)`; downsample each cell type to at most 500 cells; types with fewer keep all (`:72-79`).
3. Build a SimBu dataset from the count matrix with `filter_genes = FALSE` (`:83-86`). The reference's full
   gene set is used; the rat gene filters of 8g are applied only later, at deconvolution.
4. **Battery mode** (`:108-117`): `SimBu::simulate_bulk(scenario = "random", scaling_factor =
   "expressed_genes" | "NONE", nsamples, ncells, total_read_counts, seed, run_parallel = FALSE)`.
5. **Dose-response mode**, when `--bias-alpha` is given (`:89-107`):
   - per-type scale s_k = (mean number of expressed genes per cell of type k ÷ the mean of those means)^α,
     then divided by its own mean (`:95-97`);
   - compositions drawn once from normalised Gamma(1) variables, i.e. a flat Dirichlet, with
     `set.seed(seed + 1000)`, so they are the same for every α (`:98-101`);
   - `simulate_bulk(scenario = "custom", custom_scenario_data = comp, scaling_factor = "custom",
     custom_scaling_vector = s, ...)` (`:104-107`).
6. Write the downsampled reference, the rounded bulk counts and the **cell-fraction** truth (`:125-137`), and
   a text file of the settings (`:138-147`).

Deconvolution and scoring.
7. Each set is deconvolved against its own downsampled reference by BayesPrism (`run_deconvolution.sh`) and
   the five other methods (`run_omnideconv.sh`, `REF_MAX_CELLS_PER_TYPE=0`)
   (`deconvolution/SIMBU_BIAS_REFS4ALL_PREREG.md:22-23,29-33`).
8. Scored with `score_validation.py --truth cellfrac` (`:34`).
9. Summarised by `deconvolution/omnideconv_bench/simbu_summary.py` (current) and, in July, by
   `bias_delta.py` and `dose_response.py`.

### Models and software
| Item | Value | Where pinned |
|---|---|---|
| SimBu | 1.8.0 (Bioconductor 3.20) | `deconvolution/setup/r_packages.lock.tsv:295`; `R_libs/SimBu/DESCRIPTION` (Version 1.8.0); printed at run time (`simulate_simbu.R:58-60`) |
| BayesPrism | 2.2.3, 8 cores in the battery | provenance in `data/deconvolution/simbu_bench_refs4all/*/*/results/provenance_run_deconvolution.json` (42 records, commit `e690c22`, `n_cores` 8) |
| Other methods | as 8j, no cap | `SIMBU_BIAS_REFS4ALL_PREREG.md:22` |

SimBu is **not** in the pinned manifest `deconvolution/setup/r_packages.yaml` (no entry), nor in
`Dockerfile`, `Apptainer.def` or any committed install script (searched `deconvolution/setup/` and
`deconvolution/R/install_*.sh`); it appears only in the lock file. A container built from the manifest would
lack it.

### Parameters, thresholds and flags
| Name | Value | Where set (path:line) | What it controls | Rationale given |
|---|---|---|---|---|
| Scenario | `random` | `build_simbu_battery.sh:69`; default `simulate_simbu.R:45` | SimBu draws random compositions | July protocol (`PREREG:15-16`) |
| Scaling, "bias" arm | `expressed_genes` | `build_simbu_battery.sh:67-69`; `simulate_simbu.R:46` | Each cell's contribution scaled by its number of expressed genes | the benchmark paper's mRNA-bias model (`OMNIDECONV_BENCHMARK_PLAN.md:74-76`) |
| Scaling, "nobias" arm | `NONE` | `build_simbu_battery.sh:67` | No added scaling | same |
| Samples per set | 50 | `build_simbu_battery.sh:15`; `simulate_simbu.R:47` | Pseudobulks per arm | July protocol |
| Cells per sample | 1,000 | `build_simbu_battery.sh:15`; `simulate_simbu.R:48` | Mixture size | July protocol |
| Cells per type in the reference | ≤ 500 | `build_simbu_battery.sh:15`; `simulate_simbu.R:49,72-79` | Downsample; also the deconvolution reference | "paper uses 500" (`simulate_simbu.R:11-12`) |
| Total reads | 1,000,000 | default `simulate_simbu.R:51`; not passed by the builder | Sequencing depth of each pseudobulk | none stated |
| Seed | 1 | `build_simbu_battery.sh:15`; `simulate_simbu.R:50,72,107,111` | Downsample and SimBu's sampling; same for both arms | paired arms (`simulate_simbu.R:24-25`) |
| `--bias-alpha` | 0, 0.5, 1, 2, 3, 4 | `build_simbu_battery.sh:16` | Exponent of the per-type mRNA scale | July protocol (`PREREG:25-27`) |
| Dose-response tissues | LIVER, BLOOD | `build_simbu_battery.sh:17` | Two tissues | July's two (`PREREG:25-26`) |
| Composition seed (dose mode) | seed + 1000 = 1001 | `simulate_simbu.R:98` | Pins the compositions across α | "independent of alpha" (`:98`) |
| `filter_genes` | `FALSE` | `simulate_simbu.R:85-86` | Keep all reference genes | "keeps the ref gene space" (`:82`) |
| `run_parallel` | `FALSE` | `simulate_simbu.R:107,111` | Serial simulation | none stated |
| SimBu defaults not set by the project | `remove_bias_in_counts = FALSE`, `norm_counts = FALSE`, `scaling_factor_single_cell = TRUE`, `balance_even_mirror_scenario = 0.01` | `R_libs/SimBu/R/SimBu.rdb::simulate_bulk` (formals) | SimBu behaviour | none stated |
| Truth basis for scoring | `cellfrac` | `PREREG:34`; `simulate_simbu.R:19-20,134-136` | Score against cell fractions | the bias shows as error against cell fractions (`simulate_simbu.R:19-22`) |
| Reference cap at deconvolution | none (`REF_MAX_CELLS_PER_TYPE=0`) | `PREREG:22` | Use the downsampled reference as is | no reference mismatch (`PREREG:22-23`) |

### Statistical procedures
Definitions in `deconvolution/omnideconv_bench/simbu_summary.py`.
- **ΔRMSE** per reference and method: pooled RMSE under bias minus pooled RMSE under nobias, each from
  `overall_metrics_<method>.tsv` (`:95-107`). Pooled RMSE is as defined in 8i, against cell fractions.
- **Δr**: the same difference for pooled Pearson r (`:101,106`).
- **Corrector ranking**: mean ΔRMSE per method over the "common set", the references where all six methods
  were scored in both arms; also the mean and median over all references a method scored, and the number
  with ΔRMSE > 0 (`:110-121`).
- **Bias driver** per reference and method: the cell type with the largest per-type ΔRMSE; its true mean and
  its mean estimate in both arms; "inflated" if the estimate rises under bias; the median ΔRMSE of the other
  types (`:124-151`).
- **mRNA-content rank**: mean number of expressed genes per cell (count > 0) in the downsampled reference,
  ranked within the reference, 1 = highest; total counts per cell ranked alongside (`:59-85`).
- **Dose-response slope**: (RMSE at the largest α - RMSE at α = 0) ÷ (largest α - 0); "non-decreasing" if
  every successive difference is ≥ 0 (`:184-194`).
- **Sanity checks**: md5 equality of the two arms' reference matrices and of their truth files; one
  reference and one truth across the α values (`:227-241,168-182`).
- **Replication.** 50 pseudobulks per arm, one seed. No intervals or tests.
- **Pre-registration.** `deconvolution/SIMBU_BIAS_REFS4ALL_PREREG.md` (commit `25b682d`, 2026-11-09, "before
  any simulation"). Registered readings (`:70-87`):
  1. The paper's core finding is "reproduced" if MuSiC's and SCDC's mean ΔRMSE on the common set are both
     below BayesPrism's and CIBERSORTx's (`:73-74`; coded at `simbu_summary.py:281-286`).
  2. On how many of the 15 references BayesPrism's ΔRMSE is > 0 (`:76`).
  3. On how many references BayesPrism's driver is the type with the most expressed genes (`:77-80`).
  4. Per dose-response tissue: which methods' RMSE is non-decreasing in α; whether DWLS has the lowest
     baseline and the steepest slope of DWLS, MuSiC and SCDC (`:81-85`).
  5. The current drivers replace the per-tissue driver table in `OMNIDECONV_RESULTS.md` (`:86-87`).
  One retry is allowed for a task killed for memory or time (`:37-39`). Nothing else may change (`:98-99`).
- July's battery was not pre-registered (no pre-registration file for it was found; the plan
  `OMNIDECONV_BENCHMARK_PLAN.md` is an execution plan with expected outcomes at `:201-204`).

### Checks, validation and their results (July, earlier references)
July ran on 10 tissues' earlier references, jobs 11167195 (battery), 11167410 (dose-response), 11176839
(corrected liver and lung) (`deconvolution/OMNIDECONV_RESULTS.md:149-151`; drivers
`slurm/analysis/simbu_mrna_bias.slurm`, `simbu_doseresp.slurm`, local). Settings on disk match the protocol
(`data/deconvolution/simbu_bench/LIV_holdout/bias/mixtures/sim_meta.txt`: random, expressed_genes, 50, 1000,
500, seed 1).

| July statement | Value in the document | Recomputed from `data/deconvolution/simbu_bench/*/{bias,nobias}/scores/overall_metrics_*.tsv` (10 panel sets) |
|---|---|---|
| Mean ΔRMSE, Bisque | +0.003 | +0.0033 (9 sets) |
| SCDC | +0.016 | +0.0159 (9 sets) |
| MuSiC | +0.018 | +0.0179 (9 sets) |
| DWLS | +0.024 | +0.0245 (9 sets) |
| CIBERSORTx | +0.024 | +0.0239 (10 sets) |
| BayesPrism | +0.026 | +0.0262 (10 sets); > 0 on 10 of 10 |
| Liver, BayesPrism ΔRMSE before and after removing two spatial samples from the reference | 0.081 → 0.040 | 0.0810 (`LIV_holdout`) and 0.0401 (`LIV_novis`) |

Document lines: ranking `deconvolution/OMNIDECONV_RESULTS.md:153-156`; liver `:197-202`.

July dose-response, pooled RMSE at α = 0, 0.5, 1, 2, 3, 4 (recomputed from
`data/deconvolution/simbu_bench/doseresp/<set>/a<α>/scores/overall_metrics_<method>.tsv`):

| Set | Method | RMSE at the six α | Slope |
|---|---|---|---|
| LIV_holdout | BayesPrism | 0.083, 0.098, 0.113, 0.138, 0.158, 0.176 | +0.0234 |
| LIV_holdout | DWLS | 0.016, 0.030, 0.047, 0.083, 0.111, 0.135 | +0.0296 |
| LIV_holdout | MuSiC | 0.112, 0.111, 0.111, 0.115, 0.125, 0.139 | +0.0067 |
| LIV_holdout | SCDC | 0.123, 0.117, 0.115, 0.123, 0.131, 0.134 | +0.0027 |
| LIV_holdout | Bisque | 0.164, 0.169, 0.174, 0.183, 0.191, 0.196 | +0.0080 |
| PBMC_holdout | BayesPrism | 0.057, 0.070, 0.083, 0.108, 0.131, 0.152 | +0.0239 |
| PBMC_holdout | DWLS | 0.026, 0.030, 0.040, 0.062, 0.083, 0.107 | +0.0205 |
| PBMC_holdout | SCDC | 0.042, 0.046, 0.052, 0.067, 0.085, 0.106 | +0.0158 |
| PBMC_holdout | MuSiC, Bisque | not scored (single-donor reference) | - |

July's other statements: per-tissue bias drivers, "always the highest-mRNA type"
(`deconvolution/OMNIDECONV_RESULTS.md:168-171`); DWLS has the best baseline and the steepest slope
(`:160-163`); the pure, weighted and mirror_db scenarios were not run (`:257`).

### Deviations, known defects and caveats
1. **July's "RMSE rises monotonically with amplitude within every tissue"**
   (`deconvolution/OMNIDECONV_RESULTS.md:164`) is not exact. On liver, SCDC falls from 0.123 to 0.115
   between α = 0 and 1, and MuSiC from 0.112 to 0.111 (table above). It holds for BayesPrism, DWLS and
   Bisque.
2. **July's ranking averaged different tissue sets per method.** MuSiC, SCDC, DWLS and Bisque were scored on
   9 of the 10 panel sets, BayesPrism and CIBERSORTx on 10 (recount above). The current protocol fixes this
   with the common set (`SIMBU_BIAS_REFS4ALL_PREREG.md:47-48`). July's script `bias_delta.py` also averages
   over every folder present, which includes two extra sets (`LIV_novis`, `LNG_native`)
   (`deconvolution/omnideconv_bench/bias_delta.py:41-43,62-72`); the document's numbers match the 10-set
   means, not the 12-set means (MuSiC would be +0.0124 over 11 sets).
3. **The "nobias" arm is not free of mRNA-content differences.** `simulate_simbu.R` does not set
   `remove_bias_in_counts`, whose SimBu default is `FALSE` (formals above), and the input is raw counts. July's
   liver baseline at α = 0 is an RMSE of 0.083 for BayesPrism against cell fractions. ΔRMSE therefore
   measures the effect of the **added** expressed-genes weighting on top of whatever depth differences the
   reference's cells already have. This reading of the SimBu argument should be confirmed against the SimBu
   documentation [confirmed in the installed SimBu 1.8.0 source: `simulate_bulk` leaves `remove_bias_in_counts` at FALSE and `simulate_sample` sums raw counts times the scaling vector]; the new results file words it as "with and without the added bias"
   (`deconvolution/SIMBU_BIAS_REFS4ALL_RESULTS.md:18`).
4. **The mRNA weighting is computed on all reference genes**, including the ribosomal, mitochondrial,
   sex-chromosome and non-coding genes that the deconvolution then removes (`simulate_simbu.R:83-86,95`).
   The plan had asked for it to be computed after the gene cleanup (`OMNIDECONV_BENCHMARK_PLAN.md:315-317`).
5. **Simulation sizes differ from the plan.** The plan specified 10,000 cells and a depth of 1e7
   (`OMNIDECONV_BENCHMARK_PLAN.md:198`); the battery uses 1,000 cells and 1e6 reads. July and the current
   run agree with each other.
6. **The battery has no reference mismatch by construction** (`SIMBU_BIAS_REFS4ALL_PREREG.md:22-23,94-96`).
   It isolates the weighting effect and says nothing about accuracy against independent data.
7. **The manuscript changed during this audit.** At the start it said the battery, "run on the earlier
   references, reproduces the benchmark's central finding". The file as modified at 18:04 now describes the
   rerun on the 15 current references, says the benchmark's ordering of the methods is not reproduced, and
   changes the cross-check advice to "DWLS or MuSiC" (`manuscript/motrpac_genecompass.tex:527-534,540`;
   comment `:501-503`; figure caption `:597`). The July claims in this block (ranking, drivers, "always the
   highest-mRNA type") are therefore historical. The July section of `deconvolution/OMNIDECONV_RESULTS.md`
   still carries them, under a note pointing to the new file (`:144-171`).
8. **Hard-coded roots in July's scripts.** `bias_delta.py:13` and `dose_response.py:13` point at
   `data/deconvolution/simbu_bench`; they do not read the config and cannot summarise the current run.
   `simbu_summary.py` takes `--root` (`:216`).
9. **No provenance record** from `simulate_simbu.R` beyond `sim_meta.txt` (no commit hash).

### Methods references
- SimBu: Dietrich A et al., Bioinformatics 2022 [verified against Crossref; see REFERENCES.md] (not cited in code, docs or the manuscript
  bibliography; the docs cite only the omnideconv paper for the battery design).
- mRNA-bias benchmark design: Dietrich et al. 2026, Genome Biology 27:6 (`manuscript/motrpac_genecompass.tex:1495-1497`;
  `deconvolution/OMNIDECONV_BENCHMARK_PLAN.md:16-18`).

---

## Stage 14 step 1 (8l) - Purity sweeps: validation of the inferred expression Z

### Research role
- **Question.** How accurately does BayesPrism recover one cell type's expression profile, as a function of
  how much of the mixture that cell type makes up?
- **Claim that rests on it.** "The dominant parenchyma's inferred expression is faithfully recovered", meeting
  the BayesPrism paper's benchmark (`manuscript/motrpac_genecompass.tex:544-553`).
- **How it is checked.** Mixtures in which one focal cell type is set to 10-95% of the cells; correlation of
  inferred against true focal expression per mixture.
- **Verdict.** On 11 sweeps of 10 current references, the median correlation is 0.949-0.998 at 10% and
  ≥ 0.992 at 50% target purity (table below). These are within-study (holdout) tests of the dominant cell
  type only.

### Inputs and outputs
- Inputs: each tissue's production `reference_dir` and its `purity_sweep:` entries in
  `deconvolution/tissue_references.yaml` (`pipeline/run_stage14.py:62-83`).
- Outputs under `data/deconvolution/validation_refs4all_rebuild/SWEEP_<tag>_holdout/`
  (`config/pipeline_config.yaml:395`; `pipeline/run_stage14.py:112`): `reference/`, `mixtures/`
  (`pseudobulk_counts.mtx`, `true_fractions.tsv`, `sweep_design.tsv`, `true_z/`, `true_z_focal.csv`),
  `results/` (BayesPrism output and `pred_z/`), `scores/z_vst_focal.tsv`, `scores/purity_sweep_summary.tsv`,
  `sweep_spec.json`.
- 11 sweeps (`pipeline/run_stage14.py:31`): kidney, cortex, heart, skmgn (the shared muscle reference), bat,
  hypoth, smlint, lung_club, lung_at2, hippoc, testes
  (`deconvolution/tissue_references.yaml:42-43,118-119,146-147,232-233,286-287,314-315,340-341,368-370,394-395,427-428`).
- Not swept: BLOOD and WATSC ("no capturable dominant parenchyma", `pipeline/run_stage14.py:26-28`); SPLEEN
  and OVARY (no entry); LIVER (no entry in `deconvolution/tissue_references.yaml:67-91`); SKMVL (same
  reference as SKMGN, swept once, `pipeline/run_stage14.py:76-79`).

### Procedure
`pipeline/run_stage14.py` step 1, per sweep (`:113-129`), run in the cross-tissue job 12122211
(`deconvolution/REFS4ALL_REBUILD_RESULTS.md:7`).
1. **Build** (`deconvolution/make_purity_sweep.py --mode holdout --ref-source <reference_dir>
   --focal-type <focal> --purity-grid 0.1,0.3,0.5,0.7,0.85,0.95 --reps 10 --cells-per-mixture <n>`).
   - Split the built reference 70/30 within each cell type; re-clean and write the 70% as the sweep's
     reference; the 30% is the pool (`make_purity_sweep.py:184-185,198-207`).
   - Keep types with ≥ 20 pool cells; stop if the focal type is not among them (`:222-229`).
   - For each target purity p and replicate: focal cells = `round(p × cells_per_mixture)` (`:86-88`); the
     rest is split among the other types by a Dirichlet with parameter 1, rounded, with the rounding
     remainder added to one randomly chosen type (`:90-98`); cells drawn without replacement when the pool
     allows, else with (`:109`); counts summed (`:111-112`).
   - The true per-type expression (summed counts of each type's cells) is saved at generation
     (`:119-123,252-259`), as are the realised cell and RNA fractions of the focal type (`:126-129`).
2. **Deconvolve** against the sweep's own 70% reference with `run_deconvolution.sh` (`run_stage14.py:125`).
   `N_CORES` is not set by the orchestrator, so the config value 4 applies
   (`config/pipeline_config.yaml:428`; provenance of all 11 sweeps: `n_cores` 4, commit `aee68a3`).
3. **Extract Z** with `extract_z.sh` (`run_stage14.py:126`).
4. **Score** with `deconvolution/R/score_z_vst.R <sweep> <focal>` (`run_stage14.py:127`).
5. **Summarise** with `deconvolution/score_purity_sweep.py` (`run_stage14.py:128`).
6. Write `sweep_spec.json` (`run_stage14.py:183`). On a later run, an existing sweep whose spec is missing or
   differs is refused unless `--force` is given (`:160-168`).

`deconvolution/R/score_z_vst.R`, in order.
1. Read the true and the inferred focal matrices (samples × genes) (`:41-44`); keep the common genes
   (`:49-50`); require identical sample order (`:51`).
2. Stack them as genes × (2 × samples) and give the columns unique names `true_<mix>` and `pred_<mix>`
   (`:56-60`).
3. Convert to integers (`:61-63`), drop genes that are zero in all columns (`:64-65`).
4. Fit DESeq2's `varianceStabilizingTransformation(mat, blind = TRUE)` once on the stacked matrix
   (`:66-71`).
5. Per mixture: Pearson r between the VST values of the true and inferred columns over genes; Spearman's ρ
   on the untransformed values (`:76-82`).

### Models and software
| Item | Value | Where pinned |
|---|---|---|
| BayesPrism | 2.2.3, 4 cores | sweep provenance `results/provenance_run_deconvolution.json` |
| DESeq2 | 1.46.0 | `deconvolution/setup/r_packages.yaml:33`; `r_packages.lock.tsv:73` |
| VST call | `varianceStabilizingTransformation(mat, blind = TRUE)`, default `fitType` | `deconvolution/R/score_z_vst.R:67` |

### Parameters, thresholds and flags
| Name | Value | Where set (path:line) | What it controls | Rationale given |
|---|---|---|---|---|
| What is swept | the focal type's share of **cells** | `make_purity_sweep.py:86` | x-axis of the design | the paper's design (`make_purity_sweep.py:2-10`) |
| Purity grid | 0.1, 0.3, 0.5, 0.7, 0.85, 0.95 | `pipeline/run_stage14.py:86` (hard-coded; the script default is a different 10-point grid, `make_purity_sweep.py:147`) | Target focal cell fractions | "denser at the high end" because RNA fraction lags cell fraction in some sources (`deconvolution/run_purity_sweep.sh:25-28`) |
| Replicates | 10 per purity point (60 mixtures per sweep) | `pipeline/run_stage14.py:116`; default `make_purity_sweep.py:149` | Mixtures per point | none stated |
| Cells per mixture | 1,000; 300 for lung alveolar type II cells | `deconvolution/tissue_references.yaml` `purity_sweep` entries (e.g. `:43`, `:370`); `pipeline/run_stage14.py:82` | Mixture size | none stated for 300 |
| Background split | Dirichlet, parameter 1 | `make_purity_sweep.py:151-152` | How the non-focal cells are divided | none stated |
| Holdout fraction | 0.30 | `make_purity_sweep.py:153` | Pool share | as 8i |
| Minimum pool cells | 20 | `make_purity_sweep.py:154` | Usable types | as 8i |
| Seed | 1 | `make_purity_sweep.py:155,176` (not passed by Stage 14) | Split and mixtures | none stated |
| Focal types | proximal tubule cells; excitatory neurons (cortex, hippocampus); cardiomyocytes; skeletal muscle cells; brown adipocytes; glutamatergic neurons; enterocytes; club cells; alveolar type II cells; round spermatids | `deconvolution/tissue_references.yaml` lines above | The cell type scored | "the parenchyma label in that reference" (`pipeline/run_stage14.py:66`) |
| Metric | Pearson r on VST values, per mixture; Spearman on raw values | `deconvolution/R/score_z_vst.R:2-7,76-82` | Accuracy of inferred focal expression | the paper's metric (`:2-7`) |
| Summary | median, first and third quartile of r over the 10 replicates, per target purity | `deconvolution/score_purity_sweep.py:37-45` | Reported numbers | none stated |
| "Paper-comparable" bin | mixtures with realised focal RNA fraction ≥ 0.5 | `deconvolution/score_purity_sweep.py:29-32,52` | The number to compare with the paper | "lines up 1-1 with the paper's >50% claim" (`:9-10`) |
| Comparison bar | ">0.95 for tumors with >50% purity" | `make_purity_sweep.py:7-9`; `score_purity_sweep.py:66` | Descriptive benchmark | Chu 2022, Fig. 1h, as the code comments state |

### Statistical procedures
- **Statistic.** For mixture i, r_i = Pearson correlation over genes between VST(true focal counts_i) and
  VST(inferred focal Z_i), with the VST fitted once on all true and inferred columns together
  (`score_z_vst.R:56-82`). Also ρ_i, Spearman over genes on untransformed values (`:78-79`).
- **Summary.** Median and quartiles of r_i over the 10 replicates at each target purity
  (`score_purity_sweep.py:37-45`).
- **Units of replication.** 10 mixtures per point, drawn from one pool with one seed.
- **Null, tests, multiplicity.** None. No baseline correlation is computed (for instance between the true
  focal expression and the unseparated mixture, or the reference's mean profile).
- **Pre-registration.** No pass rule for the sweeps is pre-registered (searched every `*PREREG*.md` for
  "purity": only the reproduction criterion "equal / within 0.01" in
  `deconvolution/REFS4ALL_REBUILD_PREREG.md:29`). The 0.95 bar is applied descriptively.

### Checks, validation and their results
Median Pearson-VST per target purity, from each sweep's `scores/purity_sweep_summary.tsv`; mean realised
focal RNA fraction in brackets at 10% and 50%; pool size and sampling from the build messages in
`logs/rebuild_xtissue_12122211.out`; reference cells from each sweep's provenance.

| Sweep | Focal type | Reference cells (70%) | Focal pool cells; sampling at 95% | 10% | 30% | 50% | 70% | 85% | 95% |
|---|---|---|---|---|---|---|---|---|---|
| bat | Brown adipocytes | 19,771 | 7,450; without replacement | 0.9878 [0.094] | 0.9954 | 0.9979 [0.488] | 0.9989 | 0.9994 | 0.9997 |
| cortex | Excitatory neurons | 9,026 | 1,600; without | 0.9968 [0.347] | 0.9992 | 0.9996 [0.796] | 0.9998 | 0.9999 | 0.9999 |
| heart | Cardiomyocytes | 94,370 | 6,722; without | 0.9817 [0.256] | 0.9956 | 0.9981 [0.758] | 0.9991 | 0.9995 | 0.9997 |
| hippoc | Excitatory neurons | 16,724 | 915; **with** | 0.9914 [0.166] | 0.9978 | 0.9990 [0.653] | 0.9995 | 0.9997 | 0.9998 |
| hypoth | Glutamatergic neurons | 5,912 | 586; **with** | 0.9552 [0.107] | 0.9840 | 0.9924 [0.522] | 0.9956 | 0.9977 | 0.9988 |
| kidney | Proximal tubule cells | 17,137 | 3,645; without | 0.9857 [0.136] | 0.9956 | 0.9980 [0.598] | 0.9990 | 0.9994 | 0.9996 |
| lung_at2 | Alveolar type II cells (300 cells per mixture) | 30,910 | 422; without | 0.9778 [0.173] | 0.9926 | 0.9960 [0.622] | 0.9976 | 0.9986 | 0.9992 |
| lung_club | Club cells | 30,910 | 765; **with** | 0.9836 [0.169] | 0.9939 | 0.9970 [0.652] | 0.9981 | 0.9985 | 0.9991 |
| skmgn | Skeletal muscle cells | 7,518 | 2,175; without | 0.9487 [0.144] | 0.9846 | 0.9927 [0.603] | 0.9971 | 0.9987 | 0.9994 |
| smlint | Enterocytes | 11,334 | 2,208; without | 0.9920 [0.191] | 0.9976 | 0.9986 [0.663] | 0.9992 | 0.9996 | 0.9997 |
| testes | Round spermatids | 4,056 | 632; **with** | 0.9983 [0.093] | 0.9993 | 0.9996 [0.495] | 0.9998 | 0.9998 | 0.9999 |

- **Against the bar.** Every median, at every purity, is ≥ 0.9487. The minimum at 50% is 0.9924 (HYPOTH); at
  85% it is 0.9977 (HYPOTH). The documents round these to 0.992 and 0.998
  (`deconvolution/REFS4ALL_REBUILD_RESULTS.md:87-88`; `manuscript/motrpac_genecompass.tex:547-551`).
- **Reproduction.** The three sweeps whose reference is unchanged since their earlier run (BAT, HEART, HYPOTH)
  agree with it to 8 decimals; the other eight are first runs on the current references
  (`deconvolution/REFS4ALL_REBUILD_RESULTS.md:59-86`).
- **Earlier defect, fixed.** Before 2026-10-05, the stacked matrix had each mixture name twice. DESeq2's
  local-fit path rebuilds its output by column name, so the inferred half came back as a copy of the true
  half and every correlation was exactly 1 (seen for HEART and HYPOTH). The current code assigns unique names
  before the VST (`deconvolution/R/score_z_vst.R:57-60`; commit `e1b9ba0`). Verified in the current results:
  all 11 sweeps were scored on 2026-11-02 at commit `aee68a3` with a clean tree
  (`scores/provenance_score_z_vst.json`), and no summary value equals 1; HEART is 0.9817-0.9997 and HYPOTH
  0.9552-0.9988.
- **Second earlier defect, fixed.** Stage 14 used to rebuild superseded references from a hard-coded table.
  The sweep list is now derived from `tissue_references.yaml` and each sweep splits the production reference
  (`pipeline/run_stage14.py:62-83,113-117`).

Comparison with the BayesPrism paper.
- The benchmark as the code states it: Pearson on DESeq2-VST values between inferred and true malignant-cell
  expression, "> 0.95 for tumors with > 50% purity" (Chu 2022, Fig. 1h; `make_purity_sweep.py:7-9`;
  `score_z_vst.R:2-7`). This audit did not check the statement against the paper [checked afterwards against `readings/s43018-022-00356-3.pdf`: page 2 reads "The correlation between BayesPrism gene expression estimates and known ground truth was >0.95 for tumors, with >50% purity (Fig. 1h)", and the Methods say Pearson correlations were computed on variance-stabilizing transformed values].
- Current references: all 11 holdout sweeps exceed 0.95 at every purity point from 30% upward, and 10 of 11 at
  10% (SKMGN 0.9487).
- Pilot liver sweep (the only liver sweep, and the only cross-study sweep), built 2026-05-30 with
  `deconvolution/run_purity_sweep.sh`: holdout, median r 0.9976 over the 46 mixtures with focal RNA fraction
  ≥ 0.5; cross-study (mixtures from GSE245240), 0.9490 over the 20 such mixtures, minimum 0.9051; at lower
  purity the cross-study medians are 0.62-0.80 (recomputed from
  `data/deconvolution/validation/SWEEP_hepato_{holdout,cross}/scores/z_vst_focal.tsv` and
  `purity_sweep_summary.tsv`). The documents quote these as "holdout 0.998 / cross 0.949"
  (`deconvolution/DECONVOLUTION_PIPELINE_REPORT.md:182,1164,1310-1314,1329`).

### Deviations, known defects and caveats
1. **LIVER has no sweep on its current reference.** `tissue_references.yaml:67-91` has no `purity_sweep`
   entry for LIVER. The liver numbers that the documents cite come from the pilot sweep of 2026-05-30, whose
   reference and pool include the two spatial (Visium) samples that were removed from the liver reference on
   2026-07-01: `GSE220075_sample11` (1,701 cells) and `GSE220075_sample2` (1,667 cells) are in
   `data/deconvolution/validation/SWEEP_hepato_holdout/reference/cells_meta.tsv`; the reference QC drops
   exactly these two (`deconvolution/tissue_references.yaml:86`). The pilot also used the earlier 6-type
   labels (its `summary.txt`), not the current 8 types. Liver is the tissue where the mixture panel finds
   hepatocytes overestimated (8i).
2. **The cross-study number is below the bar it is said to meet.** 0.949 < 0.95. The documents call it
   "meets" or "PASS (marginal)" (`deconvolution/DECONVOLUTION_PIPELINE_REPORT.md:1981,2014`). No current
   reference has a cross-study sweep.
3. **The sweeps test the dominant cell type within one study.** None of the nine exercise-hotspot cell types
   is a focal type (compare the focal list above with the hotspot list in
   `deconvolution/DECONVOLUTION_PIPELINE_REPORT.md:70-72`). The accuracy of Z for the low-fraction cell types
   that carry the project's headline results is not measured on the current references: the mixture panel
   scores fractions only, and per-type Z scoring exists only for four pilot liver stages (8i, last
   paragraph).
4. **No baseline.** A correlation across thousands of genes between two expression profiles of the same cell
   type is high for reasons unrelated to deconvolution quality: both share the type's overall expression
   ranking, and the reference comes from the same study. At 95% purity the mixture itself is nearly the
   focal profile. Without a comparator (the raw mixture, or the reference mean profile, against the truth),
   the sweeps do not show how much BayesPrism adds. (Auditor's observation.)
5. **"Purity" is cell fraction in the tables but RNA fraction in the paper comparison.** The grid sets the
   focal share of cells (`make_purity_sweep.py:86`). The realised RNA fraction differs: at a 10% cell
   fraction it ranges from 0.093 (testes) to 0.347 (cortex); at 50% from 0.488 (bat) to 0.796 (cortex)
   (table). The documents' "at 50% purity" figures are by cell fraction. The pooled figure for RNA fraction
   ≥ 0.5, which the scorer calls the paper-comparable one, is **printed only and not saved**
   (`score_purity_sweep.py:52-65`; the saved table is by target purity, `:47-48`).
6. **Counts are truncated, not rounded, before the VST.** The comment says "round to integer counts"
   (`score_z_vst.R:55`); the code sets `mode(mat) <- "integer"` (`:61`), which truncates toward zero. True
   counts are integers already; inferred values below 1 become 0. The effect is confined to low-count genes
   and was not quantified.
7. **Which VST fit ran is not recorded.** DESeq2 may substitute a local dispersion fit for the parametric
   one; the earlier defect depended on which was used (`score_z_vst.R:57-59`). The script's fallback branch
   (`:68-71`) calls the same function with the same defaults, and its comment ("per-column geomean fitType")
   does not describe what the code does. Neither the fit type nor whether the fallback fired is written to
   the output.
8. **Four sweeps resample the focal pool.** HIPPOC, HYPOTH, lung club cells and TESTES have fewer focal pool
   cells than the top grid point needs, so focal cells are drawn with replacement there (table). The
   high-purity points of those sweeps contain repeated cells.
9. **Core count differs from production** (4 against 16). As coded, each sample's chain is seeded
   identically whatever the core count (`run_gibbs.R:260,281`), so results should not depend on it; this was
   not tested (the rebuild pre-registration names the core count as the first suspect if results differ,
   `deconvolution/REFS4ALL_REBUILD_PREREG.md:33`).
10. **Older sweeps remain on disk** under `data/deconvolution/validation/SWEEP_*` (built from earlier
    references, including `SWEEP_skmvl_holdout`). Other config profiles still point there
    (`deconvolution/REFS4ALL_REBUILD_RESULTS.md:120`).
11. **An earlier document claim was wrong and has been corrected**: HYPOTH was said to be "1.000 throughout";
    it is 0.992 at 50% (`deconvolution/REFS4ALL_REBUILD_RESULTS.md:89-91`).
12. **No unit tests** for `make_purity_sweep.py`, `score_z_vst.R` or `score_purity_sweep.py`.
13. **Design difference from the paper's setting.** In the paper the focal type is the malignant cell, which
    BayesPrism models separately through `key`; here `key = NULL` and the focal parenchyma is an ordinary
    type (`deconvolution/R/run_deconvolution.R:109`; contrast `deconvolution/R/run_tutorial_gbm.R:80`). The
    code calls the holdout design "paper-faithful" (`make_purity_sweep.py:13-14`). How close the analogy is
    was not assessed in this audit.

### Methods references
- Benchmark and metric: Chu et al. 2022, Nat Cancer 3:505-517, Fig. 1h (as cited in
  `deconvolution/R/score_z_vst.R:2-7`; bibliography `manuscript/motrpac_genecompass.tex:1491-1493`).
- DESeq2 variance-stabilising transformation: Love MI, Huber W, Anders S. Genome Biol 2014 [verified against Crossref; see REFERENCES.md] (not
  cited in the code or the manuscript bibliography).

---

## Stage 8m - Reference-free check of BLOOD with CDSeq

### Research role
- **Question (July).** BayesPrism's BLOOD fractions against the then-current PBMC reference were implausible.
  Was the reference missing cell populations, or was the method misfitting an adequate reference?
- **Role.** A one-off diagnostic, not a validation of the current default. Its conclusion (the PBMC reference
  lacks erythroid cells and granulocytes) led to the whole-blood composite reference
  (`deconvolution/REFERENCE_RELABEL_PREREG.md:114-120,275-282`).
- **Verdict.** Reported as: about 80% of the bulk mRNA mass sits in CDSeq components that match no PBMC
  reference type (`deconvolution/TISSUE_REFERENCE_AND_PROPORTIONS.md:1041-1051`).
- CDSeq was not rerun after the composite reference was adopted, and it is not cited in the manuscript
  (no occurrence of "CDSeq" in `manuscript/motrpac_genecompass.tex`, `REFS4_RERUN_RESULTS.md`,
  `ALL_TISSUE_RERUN_RESULTS.md` or `MIXTURE_PANEL_REFS4ALL_RESULTS.md`).

### Inputs and outputs
- Input: the BLOOD bulk (`data/deconvolution/motrpac_bulk/BLOOD/`), restricted to the gene axis of the
  BayesPrism run of that time (`deconvolution/R/run_cdseq_rr.R:43-56`): 11,319 genes × 50 samples.
- Output kept on disk: `data/deconvolution/results/_deployed_backup_20260715/BLOOD/cdseq_rr/`
  (`cdseq_proportions.csv`, `cdseq_gep.csv`, `cdseq_result.rds`, `RUN_PARAMS.txt`). `RUN_PARAMS.txt` reads
  `mode=reduce-recover blocks=16 gene_subset=1500 dilution=25 iters=700 K=14`, `genes_in=11319
  genes_recovered=11319 genes_dropped=0`.

### Procedure
1. A first run with `block_number = 1`, `dilution_factor = 1` (`deconvolution/R/run_cdseq_realbulk.R:48-55`)
   did not finish within its time limit and wrote nothing (`deconvolution/R/run_cdseq_rr.R:4-11`).
2. The run that produced the result used CDSeq's Reduce-Recover mode (`deconvolution/R/run_cdseq_rr.R:66-73`;
   driver `slurm/analysis/cdseq_blood_rr.slurm`, local): counts divided by 25; 16 blocks of 1,500 randomly
   chosen genes; 14 cell types; 700 iterations; 16 cores.
3. The components were then compared with the PBMC reference's cell-type profiles by correlation and named by
   their top genes (`deconvolution/TISSUE_REFERENCE_AND_PROPORTIONS.md:1044-1051`).

### Models and software
| Item | Value | Where pinned |
|---|---|---|
| CDSeq | 1.0.9 | `deconvolution/setup/r_packages.lock.tsv:49`; `R_libs/CDSeq/DESCRIPTION` (Version 1.0.9). Not in `r_packages.yaml` ("optional ... not pinned here", `:72`) |
| harmony (CDSeq dependency) | 2.0.5 | `deconvolution/setup/r_packages.lock.tsv:133` |

### Parameters, thresholds and flags
| Name | Value | Where set (path:line) | What it controls | Rationale given |
|---|---|---|---|---|
| `cell_type_number` (K) | 14 | `deconvolution/R/run_cdseq_rr.R:33`; `slurm/analysis/cdseq_blood_rr.slurm` | Number of components | none stated (the PBMC reference had 14 types, `deconvolution/DECONVOLUTION_PIPELINE_REPORT.md:995`) |
| `mcmc_iterations` | 700 | `run_cdseq_rr.R:34` | Sampler iterations | none stated |
| `dilution_factor` | 25 | `run_cdseq_rr.R:35,70` | Counts divided by 25 before sampling | runtime; "unbiased in expectation ... widens the posterior" (`:14-16`) |
| `block_number` | 16 | `run_cdseq_rr.R:36,72` | Gene blocks, run in parallel | runtime (`:6-11`) |
| `gene_subset_size` | 1,500 | `run_cdseq_rr.R:37,71` | Genes per block, drawn at random | about 2× expected coverage of 11,319 genes (`:17-21`) |
| `cpu_number` | 16 | `run_cdseq_rr.R:38,69` | Cores | - |
| Seed | `set.seed(1)` | `run_cdseq_rr.R:30` | R-level random state | none stated |
| Gene axis | BayesPrism's `pred_z/genes.txt` of that run | `run_cdseq_rr.R:50-56` | Same genes as BayesPrism | "directly comparable" (`:50`) |

### Statistical procedures
- CDSeq estimates proportions and component expression profiles from the bulk alone.
- Annotation, as recorded in two scratch tables (`tmp/cdseq_component_annotation.tsv`,
  `tmp/cdseq_annot_centered.tsv`, dated 2026-07-14): each component's best and second-best matching reference
  type by correlation, uncentred and gene-centred. In the centred table, three components match a reference
  type at r 0.18-0.41 (megakaryocytes 0.41, classical monocytes 0.259 and 0.18; together 20.2% of the mass)
  and the other eleven at r ≤ 0.103 (together 79.8%). That sum is the "about 80%".
- No threshold separating "matches" from "matches nothing" was fixed in advance, and no null distribution of
  the correlation was computed. The documents give a yardstick instead: components are mutually distinct at a
  median off-diagonal r of -0.094, against -0.081 among the reference types
  (`deconvolution/TISSUE_REFERENCE_AND_PROPORTIONS.md:1044-1045`).

### Checks, validation and their results
- All 11,319 genes were recovered by the random blocking (`RUN_PARAMS.txt`).
- The one component with a reference counterpart (platelet/megakaryocyte, r 0.41) was recovered
  (`deconvolution/TISSUE_REFERENCE_AND_PROPORTIONS.md:1050-1051`).
- The full-fidelity run never completed; the result is from the approximate mode, and the script says the
  exact percentages are not robust to it (`deconvolution/R/run_cdseq_rr.R:13-27`).

### Deviations, known defects and caveats
1. **The annotation step is not reproducible from the repository.** The code that correlated components with
   reference profiles, centred them, and named components by marker genes was **NOT FOUND**: searched all
   `*.py`, `*.R`, `*.sh`, `*.md`, `*.slurm` for the scratch file names and for `cdseq_gep`; only the two run
   scripts mention CDSeq outputs. Its results exist only as tables in the ignored `tmp/` folder.
2. **Hard-coded paths.** Both scripts read and write `data/deconvolution/results/motrpac/<TISSUE>/...`
   (`run_cdseq_rr.R:40,51`; `run_cdseq_realbulk.R:27,37`) and ignore the config. Run today, they would look
   for a gene axis in a folder the default configuration does not use; if the file is absent the restriction
   is skipped silently (`run_cdseq_rr.R:52-56`).
3. **The result predates the current BLOOD reference and gene axis** (11,319 genes then; 14,920 now, table in
   8g). A registered use of CDSeq as an "absorption" check of the new neutrophil and erythroid profiles was
   proposed (`deconvolution/REFERENCE_LABEL_AUDIT.md:252-254`) but is not among the six checks that were
   registered and run for the composite (`deconvolution/REFERENCE_RELABEL_PREREG.md:294-315`).
4. **No provenance record** (no `write_provenance` call); the settings are in `RUN_PARAMS.txt` only.
5. **The seed's reach is unverified.** `set.seed(1)` sets R's generator; whether CDSeq's compiled sampler and
   its parallel workers use it was not determined.

### Methods references
- CDSeq: Kang et al., PLoS Comput Biol 2019 (as cited in `deconvolution/R/run_cdseq_realbulk.R:12-13`;
  `deconvolution/BLOOD_REFERENCE_SEARCH.md:72` marks the citation "not re-checked") [verified against Crossref; see REFERENCES.md].

---

## Summary of audit findings for this section

Defects and discrepancies, most consequential first.
1. **Eleven non-ribosomal genes are excluded as ribosomal**, among them the S6-kinase family; they are absent
   from Z in all 16 tissues, and the list has no builder script
   (`deconvolution/reference/rat_exclude_genes.tsv:12,16,18,22,26,28,60,86,117,119,198`;
   `deconvolution/R/run_deconvolution.R:54-56`).
2. **BayesPrism's sampler and optimiser settings are package defaults that the code never states and the
   provenance never records** (`deconvolution/R/run_deconvolution.R:111,124-129`;
   `vendor/BayesPrism/BayesPrism/R/run_prism.R:7-8,37`); 13 of 16 default-run tissues ran with an
   uncommitted edit of the runner (their `provenance_run_deconvolution.json`).
3. **The expression-accuracy evidence is narrower than the claim.** No current sweep for LIVER; the liver
   numbers come from a pilot whose reference included two spatial samples later excluded; the only
   cross-study value (0.949) is under the 0.95 bar; sweeps cover the dominant cell type within one study, and
   no hotspot cell type has an expression test (`deconvolution/tissue_references.yaml:67-91`;
   `data/deconvolution/validation/SWEEP_hepato_holdout/reference/cells_meta.tsv`;
   `deconvolution/score_purity_sweep.py:52-65`).
4. **Mixture-panel design and scoring caveats**: uniform-Dirichlet compositions unlike real tissue, heavy
   resampling of pools as small as 20 cells, scoring that silently drops types a method does not estimate
   while printing "set to 0", CIBERSORTx run after other results were read, unpinned CIBERSORTx container,
   local-only driver scripts, no provenance (`deconvolution/make_pseudobulk.py:157-166,275`;
   `deconvolution/score_validation.py:54-57,64,86-88`; `deconvolution/MIXTURE_PANEL_REFS4ALL_RESULTS.md:73-77`).
5. **SimBu battery**: July's monotonicity statement is not exact, July's ranking averaged unequal tissue
   sets, the "nobias" arm keeps the reference's native depth differences, the mRNA weighting is computed on
   genes the deconvolution later removes, and SimBu is missing from the pinned package manifest
   (`deconvolution/OMNIDECONV_RESULTS.md:164`; `deconvolution/omnideconv_bench/simulate_simbu.R:83-86,104-111`;
   `deconvolution/setup/r_packages.yaml`, no SimBu entry). July's conclusions are superseded by the rerun
   committed during this audit (`deconvolution/OMNIDECONV_RESULTS.md:7-15`).

Smaller items: documentation errors on the outlier filter, θ's type and run times
(`deconvolution/DECONVOLUTION_PIPELINE_REPORT.md:1259,1265,1283`); stale "rel-113" gene-space label in every
pseudo-cell file (`deconvolution/build_pseudocells.py:114,139`); counts truncated before the VST
(`deconvolution/R/score_z_vst.R:55,61`); cross-check method settings differ from the plan without a stated
reason for DWLS (`deconvolution/OMNIDECONV_BENCHMARK_PLAN.md:58,293`; `deconvolution/R/run_omnideconv.R:146`);
the immune-subtype decision untested on rat data and not uniform across references; CDSeq annotation code
absent.

Could not be determined.
- Which dispersion fit (parametric or local) DESeq2 used in each sweep.
- The version of the CIBERSORTx container image, and whether its output is reproducible run to run.
- Whether the installed `R_libs/BayesPrism` is byte-for-byte the submodule commit (no hash is stored with
  the installed copy).
- The exact content of `run_deconvolution.R` that ran for the 13 tissues with a modified tree.
- Whether the BayesPrism paper's benchmark is quoted correctly (figure, threshold, design); not checked
  against the paper.
- Whether the test suite passes today (not run).
