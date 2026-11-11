# Part 2. Stages 4-6: gene medians, corpus assembly and tokenization, prior-knowledge embeddings

> One part of the methods audit of 2026-11-11; start at [`README.md`](README.md). The main findings of every
> part are collected in [`FINDINGS.md`](FINDINGS.md) and every reference in [`REFERENCES.md`](REFERENCES.md).
> Citations are `path:line` at commit `8249506` unless the text says otherwise.

Audit date 2026-11-11. Code state: branch `aim2b-perturbation-engine`, HEAD `8249506` (no Stage 4-6 file differs
from `e690c22`). Default configuration = `config/pipeline_config.yaml` (v2 model); `config/profiles/retrain_v2.yaml`
is now a no-op overlay kept because the v2 build jobs set it (`config/profiles/retrain_v2.yaml:1-3`).

Conventions used here.
- Citations are `path:line`, relative to the repository root. `vendor/GeneCompass/` is the pristine upstream
  submodule at commit `66ab82f90fa94f88613df7d89a2fb38165e2a82f` (`git submodule status`).
- `logs/` and `tmp/` are gitignored local folders. Where a number exists only there, the citation says
  "(local, uncommitted)". Such evidence would be lost by a scratch clean-up.
- "Tabulated for this audit" marks a count I obtained by reading a small on-disk TSV/JSON/pickle (no
  matrices or checkpoints were loaded); the files read are cited.
- Abbreviations: QC = quality control; T1-T4 = ortholog tiers (below); CP10K = counts scaled to 10,000 per
  cell; PCC = Pearson correlation coefficient; TSS = transcription start site; GRN = gene regulatory network;
  HGNC = HUGO Gene Nomenclature Committee; CLS = the per-cell summary position prepended by the model;
  MLM = masked language modelling; HF = HuggingFace `datasets`.

### Background shared by the three stages: the vocabulary and the ortholog tiers

GeneCompass's vocabulary has 50,558 tokens: `<pad>` = 0, `<mask>` = 1, 23,113 human genes (IDs 2-23,114) and
27,443 mouse genes (IDs 23,115-50,557) (`finetune/genecompass/build_rat_token_dictionary.py:8-12,48,68-71`;
confirmed by reading `vendor/GeneCompass/prior_knowledge/human_mouse_tokens.pickle`). Stage 3 assigns each of
the 22,213 rat genes to one tier (`data/training/ortholog_mappings/mapping_statistics.json:3-15`):

| Tier | Rat genes | Token the gene uses | Token-ID range (tabulated for this audit from `data/training/ortholog_mappings/rat_token_mapping.tsv`) |
|---|---|---|---|
| T1 `T1_tri_species` (one-to-one in human and mouse) | 13,178 | the human ortholog's existing token | 3-23,114 |
| T2a `T2a_human_one2one` | 705 | the human ortholog's existing token | 14-23,104 |
| T2b `T2b_mouse_one2one` | 3,613 | the mouse ortholog's existing token | 23,116-50,551 |
| T3a `T3a_human_multi` (one-to-many / many-to-many with human) | 1,351 | a new rat token | 50,560-55,274 |
| T3b `T3b_mouse_multi` | 892 | a new rat token | 50,559-55,263 |
| T4 `T4_new_token` | 2,474 | a new rat token | 50,558-55,273 |

17,496 genes reuse a pre-trained token and 4,717 get a new one (`mapping_statistics.json:34-35`), so the
vocabulary grows from 50,558 to 55,275 (`finetune/genecompass/configs/rat_finetune_v2.yaml:11`;
`data/models/rat_genecompass_init_v2/config.json:32`). The merged dictionary
`rat_human_mouse_tokens.pickle` is built by `build_rat_token_dictionary.py`: GeneCompass's dictionary plus
every rat entry whose ID is >= 50,558, with assertions that `<pad>`/`<mask>` are 0/1, that there is no key
collision, and that the new IDs are contiguous from 50,558
(`finetune/genecompass/build_rat_token_dictionary.py:68-71,79-102`). Reading the pickle: 55,275 entries =
23,113 ENSG + 27,443 ENSMUSG + 4,717 ENSRNOG + 2 special. Its docstring still calls all new tokens "T4" and
says "T1-T3 rat genes are intentionally excluded" (`build_rat_token_dictionary.py:14-21`), which predates the
April 2026 change that gave T3 genes their own tokens; the code selects by ID (`:79`), so T3 tokens are included.

Doc-vs-data disagreement on "before": `build_rat_token_dictionary.py:6` and `finetune/genecompass/README.md:10`
say April's model had 53,032 tokens. April's checkpoints record `vocab_size: 55275`
(`data/models/rat_genecompass_init/config.json:32`;
`data/models/rat_genecompass_finetuned/models/rat_phase2_mixed_species/checkpoint-147941/config.json:30`).
53,032 (= 50,558 + 2,474) is the design from before the T3 reclassification
(`finetune/genecompass/rat_pretrain.py:10`). April and v2 share the same 55,275-token vocabulary.

"T4" means "no ortholog that has a GeneCompass token", not "no ortholog". Tabulated for this audit from
`rat_token_mapping.tsv`, `data/references/biomart/rat_human_orthologs.tsv`, `rat_mouse_orthologs.tsv` and
the GeneCompass token pickle: 237 of the 2,474 T4 genes have a human and/or mouse ortholog row in the Ensembl
tables (165 human, 156 mouse), and none of those partner IDs is in GeneCompass's vocabulary. Example: rat
Ndufs3 (ENSRNOG00000009155) is `ortholog_one2one` with human ENSG00000213619 and mouse ENSMUSG00000005510
(`data/references/biomart/rat_human_orthologs.tsv:39194`, `rat_mouse_orthologs.tsv:54327`) and is T4, token
50,758 (`rat_token_mapping.tsv:4359`). This matters for the Stage 6 GRN result below.

---

## Stage 4 - Gene medians

### Research role
GeneCompass does not see counts. Each gene's count in a cell is divided by a fixed per-gene constant (the
gene's "median"), log-transformed, and genes are ranked by the result; the ranked genes and their values are
the model input. The medians therefore decide which genes a rat cell presents to the model and on what scale.
Question for the project: on what per-gene scale should rat expression enter a model whose weights were
trained on human and mouse cells scaled by GeneCompass's own medians?

- Claim resting on it: rat token values are on the scale of GeneCompass's own corpora, and a rat cell and its
  re-expression as human (Stage 12) share per-gene scaling
  (`pipeline/05_tokenization/build_hybrid_medians.py:24-27`).
- How checked: (i) a descriptive comparison of four median schemes against GeneCompass's human and mouse
  corpora (local, uncommitted: `tmp/data_audit/median_check/check_11893348.out:3-29`); (ii) the corpus value
  median after tokenization (`tmp/rebuild_v2/v2_assemble_11911301_4294967294.out:37`, local, uncommitted;
  `docs/RETRAIN_V2_PREREGISTRATION.md:159`); (iii) a byte-for-byte rebuild
  (`docs/repro_stages1_7/REPRODUCTION.md:53`).
- Verdict so far: the scheme adopted for v2 ("scheme B") moves the rat value distribution toward
  GeneCompass's but does not reach it: median stored value 0.9145 for the v2 corpus
  (`docs/RETRAIN_V2_PREREGISTRATION.md:159`) against 1.015 (human) and 1.007 (mouse) in samples of
  GeneCompass's own `randsel_500w` sets (`tmp/data_audit/median_check/check_11893348.out:9-10,21-22`). No
  pass/fail rule was registered for this stage.

### Inputs and outputs
Inputs
- QC file list `config/corpus_v2_plan.tsv` (`config/pipeline_config.yaml:86`): 868 rows; 557 `keep` + 27
  `keep_raw_slot` = 584 files kept; 234 `drop_duplicate`, 48 `drop_nonrat`, 2 `drop_superseded` (tabulated for
  this audit; same counts in `data/training/gene_medians_v2/scatter/scatter_0000_manifest.json:16-24`). The 584
  kept files hold 6,803,206 cells from 81 studies (80 GEO series + OMIX767).
- QC h5ad files `data/training/qc_h5ad/<sample_id>.h5ad` (`config/pipeline_config.yaml:72`), raw counts in `X`
  except the 27 `keep_raw_slot` files, whose `X` is log-normalised and whose counts are in `.raw`
  (`lib/gene_utils.py:157-177`).
- Gene universe: the non-excluded rows of `data/training/ortholog_mappings/rat_token_mapping.tsv`
  (22,213 genes, 0 excluded: `mapping_statistics.json:3-5`).
- For the hybrid step: `data/training/ortholog_mappings/rat_to_human_mapping.pickle` (15,234 rat genes = T1 +
  T2a + T3a, mapping to 14,476 distinct human genes; `mapping_statistics.json:36`, distinct count tabulated for
  this audit) and GeneCompass's `vendor/GeneCompass/scdata/dict/human_gene_median_after_filter.pickle`
  (`config/pipeline_config.yaml:97`; md5 `33a03030...` pinned in `config/genecompass_assets_md5.tsv:32`; 19,560
  human genes, values 0.11-257.75, median 1.15, tabulated for this audit).

Outputs, all in `data/training/gene_medians_v2/` (`config/pipeline_config.yaml:87`)
- `scatter/scatter_NNNN_{stats.npz,nonzero.npz,cells.tsv.gz,manifest.json}` for 16 tasks.
- `rat_gene_medians.pickle`: {ENSRNOG: median}, 21,222 genes
  (`data/training/gene_medians_v2/stage4_manifest.json:19`), md5 `df45b472...`
  (`docs/repro_stages1_7/stage4_gene_medians.tsv:8`).
- `gene_median_stats.tsv` / `.npz` (22,213 rows; median, 25th/75th/95th percentiles, mean, variance, zero
  fraction, counts, for normalised and raw values), `biotype_summary.json`, `cell_stats_summary.json`,
  `stage4_manifest.json`.
- From the hybrid step: `hybrid_gene_medians.pickle` (21,222 genes; md5 `3bcd501e...`,
  `docs/repro_stages1_7/stage4_gene_medians.tsv:6`) and `hybrid_median_provenance.json` (per-gene source).

Recorded counts (`data/training/gene_medians_v2/stage4_manifest.json:11-21`): 16 scatter tasks, 584 files,
6,803,206 cells, 22,213 universe genes, 22,199 with a median > 0, 21,222 passing the filter, 991 excluded.
Of the 991, 14 genes have no non-zero cell and 977 have 1-9 (tabulated for this audit from
`gene_median_stats.tsv`). All 27 raw-slot files were read from `.raw` (sum of `n_files_raw_slot` over the 16
scatter manifests, tabulated for this audit). Corpus-level cell statistics: median of per-file median library
size 2,604.5; mean of per-file mean library size 4,619.2; mean genes detected 1,435.4
(`data/training/gene_medians_v2/cell_stats_summary.json:4-6`).

### Procedure
1. **Orchestrator.** `pipeline/run_stage4.py` runs two steps as subprocesses (`pipeline/run_stage4.py:83-100`):
   `compute_gene_medians.py` once per task for tasks 0..N-1 with N = `medians.n_scatter_tasks` (`:196-198`),
   then `gather_gene_medians.py`. On the cluster the same two scripts were run as a SLURM array plus a
   dependent gather job (`pipeline/run_stage4.py:50-54`). `build_hybrid_medians.py` is **not** called by
   `run_stage4.py` or `run_stage5.py`; it is a separate manual command (`README.md:127-128`), and Stage 5 only
   checks that its output exists (`pipeline/run_stage5.py:190-198`).
2. **Scatter, file selection.** The kept files are the plan rows whose `action` starts with `keep`, sorted by
   path (`lib/gene_utils.py:143-146`); a kept file missing from disk is an error (`:147-150`). Task *t* takes
   files whose index *j* satisfies *j* mod N = *t* (`pipeline/04_gene_medians/compute_gene_medians.py:162-170`).
3. **Scatter, per file** (`compute_gene_medians.py:223-368`):
   a. Load the h5ad fully (`:252`). `raw_counts_adata` decides whether `X` is counts: it looks at the first
      20,000 stored values of a sparse `X` (or the first 20 rows of a dense one) and treats `X` as counts if all
      are within 1e-6 of an integer; otherwise it uses `.raw`, and raises if `.raw` is missing or non-integer
      (`lib/gene_utils.py:167-177`; call at `compute_gene_medians.py:260-262`).
   b. Map `var_names` to the universe after stripping the version suffix and upper-casing (`:193-198`); keep
      only mapped columns (`:285-290`).
   c. Library size L_c = sum of the cell's counts **over the mapped universe genes only** (`:285`, `:213`).
   d. Normalise in memory: x_cg = count_cg * 10,000 / L_c (float32; cells with L_c = 0 stay zero)
      (`:202-220`, called at `:297`). Nothing is written back to the QC file.
   e. For every gene, append the stored normalised values and the stored raw values of this file to per-gene
      buffers, and update running sums, sums of squares, non-zero counts and the number of cells seen
      (`:313-340`). `n_cells_total` for a gene counts only cells of files that contain the gene (`:340`).
   f. Write the per-gene value buffers in CSR layout, the running statistics, per-file cell statistics and a
      manifest (`:375-519`).
4. **Gather** (`pipeline/04_gene_medians/gather_gene_medians.py:629-704`):
   a. Sum the running statistics over the 16 tasks (`:184-213`); derive mean, population variance, standard
      deviation over non-zero cells, mean over all cells and zero fraction (`:216-253`).
   b. For genes in batches of 2,500, concatenate each gene's non-zero normalised values from all 16 tasks and
      take the 25th, 50th, 75th and 95th percentiles with `np.percentile` (no `method` argument, so numpy's
      default interpolation) (`:296-347`). The 50th percentile is the gene's rat median. The same is done for
      raw counts.
   c. Keep a gene in `rat_gene_medians.pickle` if it has at least `min_nonzero_cells` non-zero normalised
      values in the whole corpus (`:477-491`).
5. **Hybrid medians ("scheme B")** (`pipeline/05_tokenization/build_hybrid_medians.py:278-386`):
   a. Load the rat medians (> 0 only), the rat-to-human map and GeneCompass's human medians (> 0 only)
      (`:85-127`).
   b. F = median over the genes that have both of (rat median / human median of the gene's human ortholog)
      (`:134-146`). For v2: F = 2.052656 over 14,766 genes
      (`data/training/gene_medians_v2/hybrid_median_provenance.json:12-14`).
   c. For each rat gene with a rat median: if its human ortholog has a GeneCompass median, the divisor is that
      human median; otherwise the divisor is (rat median / F); a divisor of 1.0 is the last resort and was not
      used (`build_hybrid_medians.py:173-188`; coverage 14,766 human / 6,456 rat / 0 fallback,
      `hybrid_median_provenance.json:17-20`).
   d. Refuses to replace an existing `hybrid_gene_medians.pickle` unless `--overwrite` (`:329-333`).

**Exactly how a gene's median is computed.** For gene *g*, pool every cell of every kept QC file (all 584
files, all 6,803,206 cells, no subsampling, no per-study step or weight); scale each cell's counts to a total
of 10,000 over the universe genes present in its file; take the values of *g* that are stored as non-zero;
the median of those values is the rat median. It is corpus-wide, not "per study then pooled", so studies
contribute in proportion to their cell counts. The six largest studies supply 2,527,800 of the 6,803,206
cells (37.2%; `data/training/tokenized_corpus_v2/corpus_stats.tsv:7,8,63,68,69,71`).

**What "hybrid medians" and "scheme B" are.** "Hybrid" = GeneCompass's human median where the rat gene's
human ortholog has one, the rat median elsewhere. The label "scheme B" appears in the pre-registration and
results documents without a definition (`docs/RETRAIN_V2_PREREGISTRATION.md:22,46`;
`docs/RETRAIN_V2_RESULTS.md:55-56`). The only places that define it are the commit message of `ab5843a`
("Hybrid medians: rescale the rat-median genes onto the human-median scale (scheme B)", `git log`) and a
local, uncommitted job output that lists the four schemes compared
(`tmp/data_audit/median_check/check_11893348.out:3-8`):

| Scheme | Definition (as listed there) | Used by |
|---|---|---|
| A | human median where available, **unscaled** rat median elsewhere (`--fallback-rescale none`, `build_hybrid_medians.py:294-297`) | the April model |
| B | human median where available, rat median / F elsewhere (`--fallback-rescale auto`, the default) | v2 |
| C | rat median / F for every gene | not used |
| D | unscaled rat median for every gene | context only |

Reasons given in the code for B: (1) GeneCompass's own corpora hold raw counts divided by a fixed per-gene
median with no per-cell normalisation, so the divisor must be on that scale
(`build_hybrid_medians.py:19-23`); (2) rat medians alone compress the values, and unscaled rat medians put
the genes without a human median at about half the value of their neighbours (0.50 against 0.94 in the same
cells) (`:24-31`); (3) keeping the human median for ortholog genes keeps a rat cell and its Stage 12 human
re-expression on the same per-gene scaling (`:25-27`).

**How rat medians relate to the medians the base model was trained with.**
- Human: GeneCompass ships its human medians and its released preprocessing divides raw counts by them
  (`vendor/GeneCompass/preprocess/preprocess.py:132-163,223`). v2 uses these same numbers for the 14,766 rat
  genes whose human ortholog has one.
- Mouse: GeneCompass ships no mouse median file (the only `*median*` file in the submodule is
  `scdata/dict/human_gene_median_after_filter.pickle`). The statement that GeneCompass's mouse cells used
  mouse-specific medians on the same scale is made in a docstring (`build_hybrid_medians.py:21-23`); I found no
  committed evidence file for it. Rat genes that reuse a **mouse** token (T2b, 3,321 with a median) are
  therefore scaled by rescaled rat medians, not by whatever divisor the base model saw for that mouse token.
- How GeneCompass computed its human medians (which cells, after what normalisation) is NOT FOUND: the
  submodule has the pickle but no script that builds it (searched `vendor/GeneCompass/preprocess/`,
  `genecompass/`, `pretraining/`, `README.md`).
- The rat medians are computed on CP10K-normalised values, a different unit from the raw counts they later
  divide. One global factor F converts between the two for the 6,456 rat-median genes.

Source of the divisor by tier (tabulated for this audit from `hybrid_median_provenance.json` `per_gene_source`
and `rat_token_mapping.tsv`):

| Tier | Human median | Rat median / F | No median (gene not tokenized) |
|---|---|---|---|
| T1 | 13,116 | 21 | 41 |
| T2a | 560 | 100 | 45 |
| T2b | 0 | 3,321 | 292 |
| T3a | 1,090 | 40 | 221 |
| T3b | 0 | 681 | 211 |
| T4 | 0 | 2,293 | 181 |
| Total | 14,766 | 6,456 | 991 |

### Models and software
No statistical model. Python with numpy, scipy.sparse and anndata. The current lock pins `anndata==0.12.10`,
`numpy==2.4.3`, `scipy==1.17.1` (`requirements.lock.txt:31,134,198`). The v2 build of 2026-10-09 predates the
provenance writer for these scripts (added in commit `c6be8aa`, 2026-11-01, `git log`), so
`data/training/gene_medians_v2/` holds manifests but no `provenance_*.json`: the git commit and package
versions of the v2 run are not recorded there.

### Parameters, thresholds and flags
| Name | Value | Where set (path:line) | What it controls | Rationale given |
|---|---|---|---|---|
| `paths.qc_file_list` | `config/corpus_v2_plan.tsv` | `config/pipeline_config.yaml:86` | which QC files are read (rows with `action` starting `keep`) | drop non-rat samples, duplicate files, superseded files (`config/profiles/retrain_v2.yaml:9-13`) |
| `paths.median_dir` | `data/training/gene_medians_v2` | `config/pipeline_config.yaml:87` | output folder | keep April's inputs untouched (`config/profiles/retrain_v2.yaml:8-9`) |
| `medians.n_scatter_tasks` | 16 | `config/pipeline_config.yaml:277` | file-to-task split; fixes the scatter files and the float sums | "keep 16 to reproduce v2's Stage 4 outputs byte for byte" (`config/pipeline_config.yaml:274-276`) |
| `target_sum` | 10,000 | code default, `compute_gene_medians.py:534` (CLI `--target-sum`, `:706`); not in the config | per-cell scaling before the median | "Matches GeneCompass preprocessing" (`compute_gene_medians.py:211`); see Deviations |
| `medians.min_nonzero_cells` | 10 | `config/pipeline_config.yaml:273` (default 10, `gather_gene_medians.py:635`) | minimum non-zero cells for a gene to get a median | none stated |
| `medians.method` | `sparse_vectorized` | `config/pipeline_config.yaml:272` | nothing: no code reads it | none stated |
| percentiles | 25, 50, 75, 95 | hard-coded, `gather_gene_medians.py:297` | reported quantiles | none stated |
| `--gene-batch-size` | 2,500 | CLI default, `gather_gene_medians.py:754` | memory only | `gather_gene_medians.py:267-278` |
| raw-count test | first 20,000 stored values (sparse) or first 20 rows (dense); tolerance 1e-6 | hard-coded, `lib/gene_utils.py:168-170` | whether `X` or `.raw` is read | Feb-18 files hold log-normalised `X` (`lib/gene_utils.py:160-162`) |
| version-suffix stripping | split at ".", upper-case | `compute_gene_medians.py:135,194` | gene-ID matching | `compute_gene_medians.py:183-185` |
| `--fallback-rescale` | `auto` | CLI default, `build_hybrid_medians.py:294` | divide rat-median genes by F | `build_hybrid_medians.py:28-34`, "decided 2026-10-06" |
| F (measured, not set) | 2.052656 over 14,766 genes | `data/training/gene_medians_v2/hybrid_median_provenance.json:12-14` | rescaling of the 6,456 rat-median genes | - |
| `paths.genecompass_medians` | `vendor/GeneCompass/scdata/dict/human_gene_median_after_filter.pickle` | `config/pipeline_config.yaml:97` | human medians | - |
| `--overwrite` | off | `build_hybrid_medians.py:300-301` | protects an existing hybrid pickle | `build_hybrid_medians.py:44-45` |
| `paths.hybrid_medians` | `data/training/gene_medians_v2/hybrid_gene_medians.pickle` | `config/pipeline_config.yaml:88` | not read by Stage 4/5 code (Stage 5 reads `median_dir/hybrid_gene_medians.pickle`, `tokenize_corpus.py:180`); a search of `pipeline/`, `lib/`, `deconvolution/`, `finetune/`, `translation/` found no reader | none stated |

Seeds: none; the stage is deterministic given the file list and task count.

### Statistical procedures
- **Per-gene median.** Estimator: sample median (50th percentile, numpy default interpolation) of the
  non-zero CP10K values of the gene, pooled over all cells. Unit of replication: the cell. No weighting by
  study, no confidence interval, no test. Exact (all values are held, `gather_gene_medians.py:260-283`), not a
  sketch.
- **Inclusion rule.** n_nonzero >= 10 (`gather_gene_medians.py:477`).
- **Rescaling factor.** F = median_g (rat_median_g / human_median_h(g)) over genes with both
  (`build_hybrid_medians.py:141-146`). The spread of the per-gene ratios around F is not written to any
  committed file.
- Nothing in this stage is pre-registered. `docs/RETRAIN_V2_PREREGISTRATION.md` fixes the medians only by
  naming v2's inputs ("scheme-B", `:46`) and by the 2026-10-24 entry on the Stage 9 target sum (`:155-164`).

### Checks, validation and their results
| Check | Result | Rule | File |
|---|---|---|---|
| Kept files read, cells counted | 584 files, 6,803,206 cells, 0 errors, 27 raw-slot | equals the plan | `stage4_manifest.json:11-15`; scatter manifests (tabulated) |
| Scheme comparison on 20,370 rat cells (pre-clean-up March medians) | median stored value: GeneCompass human 1.015, mouse 1.007; A 0.886; B 0.946; C 0.950; D 0.501. Under A, rat-median genes sit at 0.499 against 0.936 for human-median genes; under B, 0.963 against 0.946 | descriptive, no pass mark | `tmp/data_audit/median_check/check_11893348.out:21-28` (local, uncommitted) |
| v2 corpus value median (every 211th row) | 0.9145 (April's corpus 0.8690) | none | `docs/RETRAIN_V2_PREREGISTRATION.md:159` |
| Rebuild from the canonical QC files, 16 tasks | `rat_gene_medians.pickle`, `hybrid_gene_medians.pickle`, stats TSV/NPZ byte-identical; 48/48 scatter npz identical | byte identity | `docs/repro_stages1_7/REPRODUCTION.md:53`; `docs/repro_stages1_7/stage4_gene_medians.tsv:2-9` |
| md5 of the on-disk pickles today | `df45b472...` (rat), `3bcd501e...` (hybrid): equal to the reproduction table | - | computed for this audit; `docs/repro_stages1_7/stage4_gene_medians.tsv:6,8` |

### Deviations, known defects and caveats
1. **The Stage 4 normalisation does not match upstream, although the code says it does.**
   `compute_gene_medians.py:211` says CP10K "Matches GeneCompass preprocessing". In the upstream script the
   per-cell normalisation line is commented out and the active line divides by the gene median only
   (`vendor/GeneCompass/preprocess/preprocess.py:157,160`); the project's own tokenizer states that
   GeneCompass's corpora were built without it (`pipeline/05_tokenization/tokenize_corpus.py:315-318`). The
   mismatch is absorbed by F, a single global constant, for 6,456 genes.
2. **Stale labels in the on-disk manifests.** `stage4_manifest.json:9` records `ensembl_release: "113"`; the
   reference files are Ensembl 115 / GRCr8 (`config/pipeline_config.yaml:31-36`). The manifest note says Stage 5
   uses `rat_gene_medians.pickle` to tokenize (`stage4_manifest.json:57`); Stage 5 uses the hybrid pickle.
3. **`build_hybrid_medians.py` header quotes the March build** (14,812 human / 6,567 rat of 21,379; "~2.24x",
   `build_hybrid_medians.py:28,36-39`). v2's numbers are 14,766 / 6,456 of 21,222 and F = 2.0527.
4. **Scheme definition and decision evidence are not in a committed record** (see above). The scheme
   comparison was run on the pre-clean-up March medians (`check_11893348.out:11-12`), not on v2's.
5. **Paralogs share a divisor.** 15,234 rat genes map to 14,476 human genes, so some T3a paralogs are divided
   by the same human median. With equal counts they tie exactly in the Stage 5 ranking.
6. **991 universe genes get no median and are never tokenized**, including 613 of the 4,717 new tokens (table
   above). Upstream instead uses a divisor of 1 for a gene missing from the median dictionary
   (`vendor/GeneCompass/preprocess/preprocess.py:141-145`).
7. **Library size uses universe genes only** (`compute_gene_medians.py:285,213`), so the CP10K scale depends on
   the gene universe, not on the cell's full count total.
8. **Stored zeros.** Normalised and raw non-zero totals differ in 4 of 16 scatter tasks (for example 512,353,052
   against 517,659,229, `scatter/scatter_0008_manifest.json:12-13`); over all genes the raw side counts
   10,843,683 more stored entries (tabulated from `gene_median_stats.tsv`; first data row, `:3`, shows 375,516
   against 377,235). The likely cause is explicitly stored zeros in some QC matrices that the sparse product at
   `compute_gene_medians.py:218-219` drops; I did not verify this. The medians used downstream come from the
   normalised side, and the inclusion filter gives 21,222 genes either way, so tokenization is unaffected;
   `median_raw`, `n_nonzero_raw` and `zero_fraction_raw` in the stats table are slightly off for those files.
9. **No study weighting.** A reviewer may ask why the median is cell-pooled when six studies hold 37% of cells.
   No rationale is stated in the code or docs.
10. `docs/ARCHITECTURE.md:32-33,136-143` describes "Stage 5: Gene Medians" with integer medians from raw counts
    over 895 files: a superseded design and stage numbering, still linked from `README.md:203`.
11. `docs/BUGS.md` (last updated 2026-02-10, `docs/BUGS.md:2`) lists no Stage 4-6 item; it predates these stages.

### Methods references
- Median scaling and rank-value input: GeneCompass, Yang X, Liu G, Feng G, et al. Cell Research 2024;34:830-845
  (`manuscript/motrpac_genecompass.tex:1479-1481`); upstream implementation
  `vendor/GeneCompass/preprocess/preprocess.py:132-193`.
- "Rank value encoding" is the term in upstream's own docstring (`preprocess.py:121-123`); the upstream
  trainer points to Geneformer's corpus (`vendor/GeneCompass/genecompass/pretrainer.py:632`): Theodoris CV,
  Xiao L, Chopra A, et al. Nature 2023;618(7965):616-624 (`manuscript/motrpac_genecompass.tex:1469-1470`).

---

## Stage 5 - Corpus assembly and tokenization

### Research role
Turns the cleaned rat single-cell corpus into the model's input format so that continued pre-training can
teach the model rat expression. Question: does the rat input follow the recipe the base model was trained
with, and can every training row be traced to its source cell?

- Claims resting on it: (a) rat cells are tokenized as GeneCompass tokenized its human and mouse cells; (b) the
  training corpus contains only rat cells, each kept file once, and no MoTrPAC data; (c) every row of the
  training mix traces to a study, sample, barcode and row of a QC file.
- How checked: a corpus verification and a 21-point mix verification at build time (local, uncommitted logs);
  a byte-for-byte rebuild; offline unit tests of the arithmetic.
- Verdict so far: the checks pass (below). The recipe matches upstream's released script step for step, with
  one difference (a gene without a median is dropped here and divided by 1 upstream); the remaining difference
  from the base corpora is the value scale (Stage 4).

### Inputs and outputs
Inputs
- The 584 kept QC h5ads (`config/corpus_v2_plan.tsv`), `data/training/ortholog_mappings/rat_tokens.pickle`
  (22,213 genes, token IDs 3-55,274, 4,717 of them >= 50,558; tabulated for this audit),
  `data/training/gene_medians_v2/hybrid_gene_medians.pickle` (21,222 genes).

Outputs in `data/training/tokenized_corpus_v2/` (`config/pipeline_config.yaml:90`)
- `shards/task_NNNN_shard_MMMM.arrow`: 607 Arrow stream files; `manifests/task_NNNN_manifest.json`: 50.
- `dataset/`: the HF dataset, 225 Arrow files plus `dataset_info.json`, `state.json`, `sorted_length.pickle`.
- `corpus_stats.tsv` (per study), `stage5_manifest.json`.
- Size: 209 GB for `dataset/` (`README.md:131`); the folder with shards is 418 GB (`du`, this audit;
  `docs/repro_stages1_7/REPRODUCTION.md:190`).

Recorded counts (`data/training/tokenized_corpus_v2/stage5_manifest.json:8-11,28-37`): top 2,048 genes, 50,000
cells per shard, `target_sum: null`, 21,222 eligible genes, 50 scatter tasks, 607 shards, 584 files, 0 file
errors, 0 empty cells, **6,803,206 cells** in the dataset. Studies: 81 (`corpus_stats.tsv:2-82`); the largest is
GSE280111 with 675,260 cells in 78 files (`corpus_stats.tsv:63`), the smallest GSE260849 with 3,174
(`corpus_stats.tsv:49`). Sequence length: minimum 198, maximum 2,048
(`tmp/rebuild_v2/v2_assemble_11911301_4294967294.out:17`, local, uncommitted). Gene-to-file mapping rate per
file: 0.9465-1.0 (task manifests, tabulated for this audit).

Row schema (`pipeline/05_tokenization/tokenize_corpus.py:128-141`;
`data/training/tokenized_corpus_v2/dataset/dataset_info.json:5-50`):

| Column | Type | Content |
|---|---|---|
| `input_ids` | list<int32>, length 2,048 | token IDs in descending value order, padded with 0 |
| `values` | list<float32>, length 2,048 | log2(1 + count/divisor) for the same positions, padded with 0.0 |
| `length` | list<int16>, one element | number of real (non-pad) tokens |
| `species` | list<int16>, one element | 2 for rat (0 human, 1 mouse) |
| `study_id`, `cell_id`, `sample_id`, `gsm` | string | study accession, barcode (`obs_names`), QC file stem, GEO sample from the plan ('' if unknown) |
| `cell_index` | int32 | the cell's row in its QC file |

### Procedure
1. **Prerequisites (manual).** `build_rat_token_dictionary.py` after Stage 3 (`README.md:122-123`) and
   `build_hybrid_medians.py` after Stage 4 (`README.md:128`).
2. **Orchestrator.** `pipeline/run_stage5.py` validates config and inputs (`:118-246`), then runs
   `tokenize_corpus.py` for tasks 0..49 serially (`:265-277`) and `assemble_corpus.py` (`:88-111`).
3. **Scatter set-up** (`tokenize_corpus.py:788-844`): eligible genes = genes with both a token and a positive
   hybrid median, sorted by ID (`:200-222`), 21,222 for v2; file list = the plan's kept files minus any study in
   `exclude_studies` (`:828-834`; 0 removed, because GSE242354 is not in the plan or the QC folder); stride
   split by task (`:286-292`, `:844`).
4. **Per file** (`:402-597`): load fully (`:431`); take counts from `X` or `.raw` (`:444-446`); map
   `var_names` to eligible genes after version stripping (`:381-399`, `:454-456`); subset to those columns as
   float32 CSR (`:473-478`); process cells in batches of 1,024 (`:535-548`); skip a cell with no expressed
   eligible gene (`:555-557`; none in v2); write a shard when 50,000 cells have accumulated and at the end of
   each file (`:568-572`), so a shard never mixes files.
5. **Per cell: the tokenization recipe** (`tokenize_cell_batch`, `:299-374`):

| Step | What the code does | Line | Upstream GeneCompass (released script) |
|---|---|---|---|
| 1. Count normalisation | none. Raw counts are used as stored; no per-cell scaling, no target sum | `tokenize_corpus.py:342`; `stage5_manifest.json:10` | none in the active line; the per-cell `* Parmerters` variant is commented out (`vendor/GeneCompass/preprocess/preprocess.py:157,160`); `Parmerters=1e4` is passed but unused (`:310`) |
| 2. Divide by the gene's divisor | count / hybrid median, per gene, float32 | `:342` | count / human median; 1 if the gene has none (`preprocess.py:141-146,160`) |
| 3. Log transform | log2(1 + x), computed as `log1p(x)/ln 2` | `:331,342` | `sc.pp.log1p(adata, base=2)` (`preprocess.py:165-167`) |
| 4. Rank | genes with value > 0, sorted by descending value with `np.argsort(-values)` (default sort kind, not stable) | `:345-357` | same call (`preprocess.py:124-125`) |
| 5. Top N | first 2,048 genes | `:358`; N from `config/pipeline_config.yaml:294` | 2,048 (`preprocess.py:182-185`) |
| 6. Map to tokens | rat gene -> token ID from `rat_tokens.pickle` | `:362` | `token_dict[gene]` (`preprocess.py:128`) |
| 7. Pad | token 0 (`<pad>`) and value 0.0 up to 2,048 | `:366-368` | zeros to 2,048 (`preprocess.py:171-172,187-190`) |
| 8. Length, species | `length` = number of real tokens; `species` = [2] | `:561-562` | `length`; species 0 human, 1 mouse (`preprocess.py:196-204`) |

   No special token is written into `input_ids`: there is no CLS or species token in the sequence. The model
   prepends one embedding looked up by `species` after the embedding layer
   (`vendor/GeneCompass/genecompass/modeling_bert.py:405-410`), and positions 0-2,047 carry the genes
   (`max_position_embeddings` 2,048, `vendor/GeneCompass/pretrained_models/GeneCompass_Base/config.json:15`;
   `pad_token_id` 0, `:19`).
   **How expression values reach the model.** The `values` column is passed alongside the IDs: the model
   appends each token's value as one extra input feature to its word embedding (`use_values: true`,
   `GeneCompass_Base/config.json:31`; `vendor/GeneCompass/genecompass/knowledge_embeddings.py:188-190,114-120`)
   and has a second head that predicts masked values (loss = 0.2 x token cross-entropy + 0.8 x value mean
   squared error, `modeling_bert.py:788-804`).
6. **Gather** (`pipeline/05_tokenization/assemble_corpus.py:506-663`): check that all 50 task manifests exist,
   agree on the task count and have no duplicates (`:120-166`, expected count from the config, `:552-554`);
   list the shards named in the manifests **sorted by file name** (`:183`), i.e. task, then shard number;
   concatenate with `datasets.concatenate_datasets` and `save_to_disk` (`:219-229`); write `corpus_stats.tsv`
   (`:628-629`), `sorted_length.pickle` (all lengths, sorted ascending, `:443-447`) and the stage manifest.

**Corpus assembly rules.**
- Which cells enter: every cell of every kept QC file that has at least one expressed eligible gene. Cell QC
  itself is Stage 2's. The plan removes 234 duplicate files, 48 non-rat files and 2 superseded files before
  this stage.
- Per-study or per-file caps: none. There is no cap, down-sampling or weighting in `tokenize_corpus.py` or
  `assemble_corpus.py`.
- Shuffling and seeds: none in Stage 5. Row order is fixed by task count (50), sorted file order and cell
  order within a file (`config/pipeline_config.yaml:296-300`).
- Train/validation split: none. The whole corpus is one dataset.
- Held-out set: none, as pre-registered (`docs/RETRAIN_V2_PREREGISTRATION.md:13-17`). Confirmed in the
  artefacts: `exclude_studies: []` in the mix settings (`finetune/genecompass/configs/rat_finetune_v2.yaml:95`)
  and in the mix's provenance (`data/training/mixed_species_dataset_v2/provenance_build_mixed_species_dataset.json:58`).
  The evaluation probe is drawn from the training mix and is in-sample by design
  (`docs/RETRAIN_V2_PREREGISTRATION.md:31-38`).
- MoTrPAC exclusion: `exclude_studies: [GSE242354]` (`config/pipeline_config.yaml:303-304`) is applied at
  tokenization (`tokenize_corpus.py:831`). It is a guard only: no GSE242354 file exists in the plan or in
  `data/training/qc_h5ad/` (checked for this audit).

**How rat genes map to token IDs.** See the tier table in the Background. In the corpus only genes with a
median occur: 13,137 T1 + 660 T2a (human tokens), 3,321 T2b (mouse tokens), and 1,130 T3a + 681 T3b + 2,293
T4 = 4,104 new tokens (tabulated for this audit). A verification on every 211th row found token IDs between
3 and 55,266 (`tmp/rebuild_v2/v2_assemble_11911301_4294967294.out:37`, local, uncommitted).

#### The mixed human + mouse + rat training set
Built by Stage 7 step 2 (`pipeline/run_stage7.py:213-230`) with
`finetune/genecompass/build_mixed_species_dataset.py`; summarised here because it is the dataset v2 trained
on in both phases (`finetune/genecompass/configs/rat_finetune_v2.yaml:34,57`).

- Sources: the rat corpus above; GeneCompass's `randsel_500w_human` (5,000,000 cells) and
  `randsel_500w_mouse` (`rat_finetune_v2.yaml:25-26`). The mouse set as downloaded lacks one of 165 shards
  (`data-00107-of-00165.arrow`); its `state.json` was edited to drop that entry, leaving 4,969,697 cells
  (`scripts/stage_genecompass_assets.sh:20-24`; `config/genecompass_assets_md5.tsv:11-12`).
- Method: add `species` where missing (0 human, 1 mouse, 2 rat) (`build_mixed_species_dataset.py:132-142`);
  add `source` and `source_row` to every row and blank rat-only columns to human and mouse rows
  (`:119-127,144-148`); `datasets.interleave_datasets` over [human, mouse, rat] with probabilities
  [0.45, 0.15, 0.40], `seed=42`, `stopping_strategy="all_exhausted"` (`:153-158`); save; write
  `sorted_length.pickle` holding each row's length **in dataset order** (`:171-181`). An existing mix is never
  overwritten (`:82-84`).
- Under `all_exhausted`, sampling continues until every source has been used up at least once, so the smaller
  pools repeat. Result (`data/training/mixed_species_dataset_v2/provenance_build_mixed_species_dataset.json:52-74`):

| Species | Rows | Share | Distinct source cells | Rows per cell |
|---|---|---|---|---|
| Human | 14,913,398 | 45.0% | 5,000,000 | 2.98 |
| Mouse | 4,969,697 | 15.0% | 4,969,697 | 1.00 |
| Rat | 13,255,608 | 40.0% | 6,803,206 | 1.95 |
| Total | **33,138,703** | | 16,772,903 | |

  (Shares and rows per cell: `tmp/rebuild_v2/v2_mix_11911302_4294967294.out:16-18,25,27,29`, local,
  uncommitted.) Sequence length in the mix: 22 to 2,048 (`:30`). The mouse pool sets the total. One training
  "epoch" therefore shows each human cell about 3 times and each rat cell about twice. Cross-check: 33,138,703
  rows / (16 per GPU x 4 GPUs) = 517,793 steps and / (14 x 4) = 591,763 steps, the recorded phase lengths
  (`rat_finetune_v2.yaml:32,49,115`; `docs/RETRAIN_V2_RESULTS.md:21-22`).
- **Per-row source references.** Every row carries `source` (dataset name) and `source_row` (row in that
  dataset); rat rows also carry `study_id`, `sample_id`, `gsm`, `cell_id` and `cell_index`; human and mouse
  rows have '' and -1 there (`build_mixed_species_dataset.py:12-15,109-127,147`). These columns are dropped
  before training (`finetune/genecompass/rat_pretrain.py:507-511`). 40 of the 584 kept samples have no GSM in
  the plan (tabulated; same figure in `tmp/rebuild_v2/v2_mix_11911302_4294967294.out:36`).

### Models and software
No model is fitted. Software: anndata, scipy, numpy, pyarrow, HF `datasets`; lock pins `pyarrow==23.0.1`,
`datasets==4.8.4` (`requirements.lock.txt:56,170`). The target format is GeneCompass's
(`vendor/GeneCompass/preprocess/preprocess.py:195-213`); the model it feeds is a 12-layer, 12-head, 768-wide
BERT with 2,048 positions (`rat_finetune_v2.yaml:13-17`). As for Stage 4, the v2 run wrote manifests but no
`provenance_*.json` for tokenization or assembly; the mix has one (git `8ef7f86`, 2026-10-09,
`provenance_build_mixed_species_dataset.json:3-4`).

### Parameters, thresholds and flags
| Name | Value | Where set (path:line) | What it controls | Rationale given |
|---|---|---|---|---|
| `reference_assembly.top_n_genes` | 2,048 | `config/pipeline_config.yaml:294` (fallback `TOP_N_DEFAULT`, `tokenize_corpus.py:122`) | genes kept per cell = sequence length | GeneCompass paper quote in `config/pipeline_config.yaml:289-293` [confirmed: the quoted sentence is in the GeneCompass paper's Methods (Yang et al. 2024; `readings/` PDF page 12)]; model positions 2,048 |
| `reference_assembly.cells_per_shard` | 50,000 | `config/pipeline_config.yaml:283` | shard size (I/O only) | none stated |
| `reference_assembly.n_tokenize_tasks` | 50 | `config/pipeline_config.yaml:300` | file-to-task split, hence the row order of the corpus and of the mix | `config/pipeline_config.yaml:296-299` |
| `reference_assembly.exclude_studies` | `[GSE242354]` | `config/pipeline_config.yaml:303-304` | studies never tokenized | "data leakage prevention" (`:302`) |
| `paths.tokenized_corpus_dir` | `data/training/tokenized_corpus_v2` | `config/pipeline_config.yaml:90` | output | - |
| `CELL_BATCH_SIZE` | 1,024 | hard-coded, `tokenize_corpus.py:121` | memory only | `tokenize_corpus.py:54-55` |
| log base | 2 | hard-coded, `tokenize_corpus.py:331,342` | value transform | "matching GeneCompass sc.pp.log1p(base=2)" (`:309`) |
| per-cell normalisation | none (`target_sum: None`) | hard-coded, `tokenize_corpus.py:342,690` | - | `tokenize_corpus.py:315-318,338-341` |
| pad token / pad value | 0 / 0.0 | hard-coded, `tokenize_corpus.py:334-335,366-368` | padding | upstream convention (`preprocess.py:171-172`) |
| species value | 2 | hard-coded, `tokenize_corpus.py:562` | species index for rat | 0 human, 1 mouse upstream (`preprocess.py:196-199`) |
| sort kind | numpy default (not stable) | `tokenize_corpus.py:357` | order of exactly tied values | none stated |
| `biotype_output_map` | protein_coding, lncRNA, miRNA | `config/pipeline_config.yaml:284-287` | not used by Stage 5 scripts (read by `lib/gene_utils.py:705`) | - |
| `--expected-tasks` | 50 (from config) | `assemble_corpus.py:552-554` | completeness check | - |
| mix `rat_prob / human_prob / mouse_prob` | 0.40 / 0.45 / 0.15 | `rat_finetune_v2.yaml:90-93`; script defaults `build_mixed_species_dataset.py:67-69` | species shares of the mix | "restores the multi-species gradient equilibrium" (`build_mixed_species_dataset.py:8-10`) |
| mix `seed` | 42 | `rat_finetune_v2.yaml:94`; default `build_mixed_species_dataset.py:70` | interleave order | none stated |
| mix `stopping_strategy` | `all_exhausted` | hard-coded, `build_mixed_species_dataset.py:157` | oversampling until every pool is used once | none stated in the script |
| mix `exclude_studies` | `[]` | `rat_finetune_v2.yaml:95` | held-out rat studies | no held-out set (`docs/RETRAIN_V2_PREREGISTRATION.md:13`) |

### Statistical procedures
None. Tokenization and assembly are deterministic transforms. The only random element in this part of the
pipeline is the interleaving of the mix (HF `interleave_datasets`, seed 42), which reproduces byte for byte
(`docs/repro_stages1_7/REPRODUCTION.md:55`).

### Checks, validation and their results
| Check | Result | Rule | File |
|---|---|---|---|
| Corpus against the plan | 6,803,206 rows; 584 samples = the kept set; 0 dropped samples present; 0 samples with more rows than cells; 0 duplicate (sample, cell_index) pairs; 0 GSM mismatches in a 1/997 sample | all zero / equal | `tmp/rebuild_v2/v2_assemble_11911301_4294967294.out:32-37` (local, uncommitted); script `tmp/data_audit/rebuild_v2/verify_corpus.py:1-25` |
| Corpus value median and token range (every 211th row) | 0.915; IDs 3-55,266 | "scheme-B check: ~0.95" printed as a reference, not a gate | same file, `:37` |
| Mix verification | 21 of 21 PASS: species IDs, shares 0.45/0.15/0.40, every source cell present, `sorted_length.pickle` equals per-row lengths, source columns blank or filled as designed, rat samples = kept set, 500 sampled rows per species identical to their source row | all PASS | `tmp/rebuild_v2/v2_mix_11911302_4294967294.out:21-43` (local, uncommitted); cited in `docs/RETRAIN_V2_PREREGISTRATION.md:105-106` |
| Rebuild, Stage 5 | 607/607 shards and 225/225 dataset files + `sorted_length.pickle` byte-identical | byte identity | `docs/repro_stages1_7/REPRODUCTION.md:54` |
| Rebuild, mix | 1,093/1,093 Arrow files byte-identical | byte identity | `docs/repro_stages1_7/REPRODUCTION.md:55` |
| Unit tests | value formula, descending order, ties, top-N truncation, zero padding, empty cells, library-size invariance of the order, eligible-gene rule, version/case-insensitive ID mapping, study-ID parsing, task split | pytest asserts | `tests/test_tokenization.py:19-74` |

### Deviations, known defects and caveats
1. **The orchestrator's description contradicts the code it runs.** `pipeline/run_stage5.py:22-30` lists
   "normalize_total(10,000)" then "log1p"; the worker divides raw counts by the hybrid median with no per-cell
   normalisation and uses log base 2 (`tokenize_corpus.py:342`), and the manifest records `target_sum: null`
   (`stage5_manifest.json:10`). The same file's step description is correct (`run_stage5.py:93-96`). Also stale
   in `tokenize_corpus.py`: the schema comment "log2(1 + norm/median)" (`:39`) and the `load_median_dict`
   docstring ("human median for T1-T3 ... rat median for T4", `:175-178`; the actual sources by tier are in the
   Stage 4 table). `run_stage5.py:18` says "864 QC'd files" (584 are read). The current default run used the
   code, not the diagram.
2. **The claim that GeneCompass intended CP10K but trained without it** rests on a code comment citing a GitHub
   issue and a comparison with `randsel_5w_mouse` (`tokenize_corpus.py:315-318,338-341`). The upstream script
   supports "no normalisation in the released code" (`preprocess.py:157,160`). The issue and the comparison
   output are not in the repository: unverifiable here.
3. **Stage 9 uses a different recipe from the corpus**: pseudo-cells are first scaled to a target sum (6,400)
   and then divided by the hybrid median (`config/pipeline_config.yaml:411-413`). That is Stage 9's design and
   is covered in its section; it is flagged here because "the same tokenization" is not literally true.
4. **Exact ties are ordered by an unstable sort** (`tokenize_corpus.py:357`). The output is reproducible on
   the same software (byte-identical rebuild), but the order among exactly tied genes, including which tied
   gene falls at the 2,048 cut, is arbitrary. Ties are expected among paralogs that share a human median.
5. **`sorted_length.pickle` of the rat corpus is sorted ascending** (`assemble_corpus.py:443-447`), following
   upstream's `save_disk` (`vendor/GeneCompass/preprocess/preprocess.py:215-220`), while the sampler reads
   `lengths[i]` as the length of row *i* (`vendor/GeneCompass/genecompass/pretrainer.py:897`). The mix builder
   writes row-order lengths for that reason and notes that April's mix held `[0]` for every row
   (`build_mixed_species_dataset.py:15-17,176-180`). v2 trains only on the mix, so v2 is not affected; training
   directly on `tokenized_corpus_v2/dataset` would group by wrong lengths.
6. **No per-study cap or weighting.** GSE280111 alone is 9.9% of the corpus (`corpus_stats.tsv:63`).
7. **Oversampling in the mix.** Human rows are 45% of the mix but each of the 5,000,000 human cells appears
   about three times and each rat cell about twice; the mouse pool is one shard short of upstream's.
8. **The size and download source of the human and mouse sets are mislabelled.** The repository calls them
   "0.5M" cells (`README.md:192`; `finetune/genecompass/README.md:8`;
   `scripts/stage_genecompass_assets.sh:15-16`) and gives the two download links of upstream's "0.5M |
   Pretrained data of 500k single cells" row (`vendor/GeneCompass/README.md:52`; upstream's "5M" row has
   different links, `:53`). The sets on disk hold 5,000,000 and 4,969,697 rows
   (`provenance_build_mixed_species_dataset.json:62-67`), ten times the label. Either upstream's table
   mislabels that row or the recorded link is not the one the on-disk files came from; I could not tell which
   without downloading. A rebuild must check the row count, not the label.
9. **Stale labels in the manifest.** `stage5_manifest.json:12` records `ensembl_release: "113"` (the files are
   Ensembl 115, `config/pipeline_config.yaml:31-36`).
10. **Git commit of the v2 tokenization run is not recorded** (no provenance file; see Models and software).
    The rebuild at a later commit is byte-identical, which bounds the risk.
11. **Study IDs are inferred from file names** (`tokenize_corpus.py:269-283`); a file whose prefix is not
    GSE/GSM/E-/OMIX would get its whole stem as study ID. All 81 v2 study IDs parse as accessions
    (`corpus_stats.tsv:2-82`).
12. `assemble_corpus.py:420-438` contains a loop that does nothing before the real implementation (dead code).

### Methods references
- Tokenization recipe: GeneCompass (Yang et al., Cell Research 2024;34:830-845,
  `manuscript/motrpac_genecompass.tex:1479-1481`); implementation followed:
  `vendor/GeneCompass/preprocess/preprocess.py:120-213`.
- Rank value encoding: Geneformer (Theodoris et al., Nature 2023;618(7965):616-624,
  `manuscript/motrpac_genecompass.tex:1469-1470`), as referenced by upstream (`preprocess.py:121-123`;
  `pretrainer.py:632`).
- Dataset interleaving: HuggingFace `datasets.interleave_datasets` (software; no paper cited in the repo).

---

## Stage 6 - Prior-knowledge embeddings

### Research role
GeneCompass attaches four fixed 768-dimensional "prior knowledge" vectors to every gene token: promoter
sequence, co-expression, gene family and GRN. They are stored as non-trainable buffers, each passed through
its own linear layer and concatenated with the token's word embedding and value
(`vendor/GeneCompass/genecompass/knowledge_embeddings.py:74-127,196-214`). Rat genes that reuse a human or
mouse token inherit that token's rows. The 4,717 new rat tokens have none. Question: can rat-specific prior
vectors be built for the new tokens in a form the pre-trained linear layers can use?

- Claim that could rest on it: new rat tokens receive biological prior knowledge.
- How checked: (i) for the genes that exist in both, comparison of each Stage 6 vector with GeneCompass's own
  row (direct cosine, and held-out cosine after a fitted rotation against a shuffled null); (ii) a
  pre-specified held-out test of a rebuilt co-expression prior; (iii) the non-zero row counts logged when the
  model's starting checkpoint is built.
- Verdict so far: **negative for three of the four priors.** Only the GRN transfer is in GeneCompass's
  coordinates, because it copies GeneCompass's own vectors. v2's starting checkpoint takes from Stage 6
  exactly **36 GRN rows** for T4 tokens and nothing else; T3 tokens copy their ortholog's pre-trained rows
  without using Stage 6 (`logs/init_v2_11910364.err:10-13`, local, uncommitted; `README.md:141`;
  `finetune/genecompass/README.md:56-58`).

### Inputs and outputs
Inputs
- Stage 3: `rat_token_mapping.tsv`, `rat_to_human_mapping.pickle` (15,234 pairs covering T1, T2a, T3a; T2b,
  T3b and T4 genes are absent from it and keep their ENSRNOG ID in every Stage 6 output; tabulated for this
  audit from `data/training/prior_knowledge/grn_transfer_coverage.tsv`).
- Co-expression: QC h5ads under `data/training/qc_h5ad/`.
- Family: `data/references/hgnc/hgnc_complete_set.txt` (md5 `5e1e18fe...`,
  `data/training/prior_knowledge/stage6_family_manifest.json:31-35`) and
  `data/references/biomart/mouse_human_orthologs.tsv`.
- Promoter: TSS coordinates from Ensembl BioMart, a rat genome FASTA, DNABERT-6 weights.
- GRN: GeneCompass's `prior_knowledge/PECA2vec/{human,mouse}_PECA_vec.pickle`,
  `prior_knowledge/gene_list/Gene_id_name_dict_human_mouse.pickle`
  (`config/genecompass_assets_md5.tsv:21,27-28`) and `data/references/biomart/rat_mouse_orthologs.tsv`.

Outputs in `data/training/prior_knowledge/` (`config/pipeline_config.yaml:101`). The folder is shared by
April and v2 (it is not versioned); every file is dated 2026-03-17 to 03-20 and a copy made before the October
changes is in `data/training/prior_knowledge_backup_2026-10-05/` (directory listings, this audit).

| File | Keys | Built | Recorded counts |
|---|---|---|---|
| `coexp_embeddings.pkl` | human ENSG for T1/T2a/T3a genes, ENSRNOG otherwise | 2026-03-20 | 864 files processed, 189,992,325 raw pairs, 36,396,244 unique pairs, 16,939 vectors, 176,402 s (`stage6_coexp_manifest.json:6,31-38`) |
| `family_embeddings.pkl` | human ENSG | 2026-03-18 | 1,927 human / 1,731 mouse / 1,583 rat families, 8,864,659 pairs, 26,019 vectors, 0 of 2,474 T4 genes embedded (`stage6_family_manifest.json:37-46`) |
| `promoter_embeddings.pkl` | as co-expression | 2026-03-18 | 22,213 genes in, 16,120 sequences, 6,093 skipped, 15,610 vectors (`stage6_promoter_manifest.json:19-25`) |
| `grn_embeddings.pkl` | as co-expression | 2026-03-18 | 22,213 genes: 15,234 human PECA2vec, 3,516 mouse PECA2vec, 3,463 zero (`stage6_grn_manifest.json:15-21`); 21,455 distinct keys (`logs/init_v2_11910364.err:9`) |

Plus the gene-pair lists, gensim models, `promoter_sequences.fa`, `promoter_skipped.tsv`,
`tss_coordinates_cache.pkl`, `grn_transfer_coverage.tsv` and one manifest per step.

### Procedure
**Orchestrator.** `pipeline/run_stage6.py` runs four steps as subprocesses (`:102-151`): co-expression and
family always, promoter and GRN if `prior_knowledge.<step>.enabled` (`:322-334`). Under the default config Step
1 exits immediately (see co-expression below), so the documented call is `run_stage6.py --from 2`
(`README.md:136-140`).

**How the vectors reach the model (Stage 7 step 1).** `finetune/genecompass/rat_load_prior_embedding.py`
builds four [55,275 x 768] tensors: zeros; rows 0-50,557 copied from the GeneCompass_Base checkpoint
(`:148-155`); each T3 row copied from its ortholog's pre-trained row (`:158-161`); every other new row looked
up in the Stage 6 pickle **by rat gene ID** unless that prior is listed in `t4_zero_priors` (`:164-173`);
co-expression rows that came from Stage 6 multiplied by 3.9 (`:176-179`). It raises if any T3 row is still
zero (`:190-192`). The upstream loader cannot be used because it raises on any key that is not ENSG or
ENSMUSG (`vendor/GeneCompass/genecompass/utils.py:86-97`).

#### 6.1 Co-expression (`pipeline/06_prior_knowledge/build_coexp_embedding.py`)
- Upstream prior: `prior_knowledge/gene_co_express_emb/{Human_dim_768_gene_28291,Mouse_dim_768_gene_27444}_random.pickle`
  (`vendor/GeneCompass/genecompass/utils.py:17-19`), keyed by gene symbol. The submodule ships the vectors
  but no code that builds them. The repository describes the upstream protocol as: 3,000 cells sampled per
  matrix, Pearson correlation on cells where both genes are expressed, pairs with PCC above 0.8, gene2vec at
  768 dimensions (`build_coexp_embedding.py:15-29,183-185,236-239`; "GeneCompass exact",
  `config/pipeline_config.yaml:314-315`) [confirmed in the GeneCompass paper's Methods (Yang et al. 2024; `readings/` PDF page 13): 3,000 cells sampled uniformly from each matrix; Pearson correlation between two expressed genes (count >= 1); pairs with correlation above 0.8 embedded with gene2vec].
- Rat build, step by step:
  1. Every `*.h5ad` under the QC folder, sorted (`:160-172`). The builder refuses to run when
     `paths.qc_file_list` is set, which the default config does (`:703-708`); the on-disk result was built in
     March on all 864 files then present (`stage6_coexp_manifest.json:32`).
  2. Per file: sample 3,000 cells uniformly without replacement, or all cells if fewer, with
     `np.random.default_rng(42)` (same seed for every file) (`:179-191`); warn if the maximum of `X` is below
     20 (`:394-405`).
  3. Keep genes with count >= 1 in more than 5% of the sampled cells (`:267-270`); set values below 1 to
     missing (`:281-282`); pairwise Pearson correlation of **raw counts** with pandas, requiring at least 10
     cells where both genes are present (`:289`); keep pairs with correlation **strictly greater than** 0.8
     (`:292`; the docstring at `:22` says ">= 0.8"). Genes are processed in blocks of 2,000 (`:297-344`).
  4. Rename each gene to its human ortholog ID if Stage 3 has one, else keep the rat ID (`:198-219`); store the
     pair sorted so (A,B) = (B,A) (`:455`); take the union over files (`:784`, `:817`).
  5. Train gensim `Word2Vec` on the pairs as two-word sentences: `vector_size=768`, `window=1`,
     `min_count=1`, `sg=1` (skip-gram), `workers=8`, `epochs=30`, `seed=42` (`:521-530`). All other arguments
     are gensim defaults and are not recorded.
  6. Save one float32 vector per vocabulary entry (`:863-870`).
- April: new rows were looked up by rat ID, so the T3b and T4 rows present in the pickle were used (338 of 892
  and 812 of 2,474 non-zero, from the zero counts in `tmp/data_audit/prior_zero_rows_by_tier.tsv:12-13`, local,
  uncommitted). The 3.9 rescale was written for these rows (`rat_load_prior_embedding.py:91-94`); whether
  April's init applied it is not recorded (no provenance file for that checkpoint). v2: not used (zero by
  policy).

#### 6.2 Gene family (`pipeline/06_prior_knowledge/build_family_embedding.py`)
- Upstream prior: `prior_knowledge/gene_family/{Human_dim_768_gene_28291,Mouse_dim_768_gene_27934}_random.pickle`
  (`utils.py:21-23`). Protocol as described in the repository: HGNC families, gene2vec at 768 dimensions
  (`build_family_embedding.py:16-30,566-568`) [confirmed in the GeneCompass paper's Methods (Yang et al. 2024; `readings/` PDF page 13): HGNC families (1,645 human, 1,539 mouse, the mouse ones by homology); any two genes of a family form a pair; pairs encoded with gene2vec at 768 dimensions].
- Rat build:
  1. Parse HGNC: for each row with an Ensembl ID, add the gene to each pipe-separated family ID
     (`:296-397`; column name candidates `:317-320`).
  2. Mouse families: map each human member to all its mouse orthologs from the BioMart table
     (`:214-247`, `:404-460`). Rat families: the same through the inverse of Stage 3's rat-to-human map
     (`:186-211`). Keep derived families with at least 2 members (`:448`).
  3. For each species' families, rename members to the unified ID (human ID where a mapping exists) and emit
     all within-family pairs (`:467-488`, `:491-544`).
  4. gensim `Word2Vec`: `vector_size=768`, `window=1`, `min_count=1`, `sg=1`, `epochs=30`, `workers` and
     `seed` from the config (`:589-598`, `:803-813`). Since 2026-11-01 the script re-executes itself under a
     fixed `PYTHONHASHSEED` (`:746-760`).
- By construction this step cannot produce a vector for a rat-specific token. A rat gene enters a family only
  through `rat_to_human`, and such a gene is then renamed to its human ID (`:486-487`); T2b, T3b and T4 genes
  are never in a family. The on-disk manifest and log confirm 0 of 2,474 T4 genes
  (`stage6_family_manifest.json:44-45`; `logs/stage6_family_10421541.err:1356`, local), and
  `data/training/prior_knowledge/family_gene_pairs.txt` contains no ENSRNOG and no ENSMUSG identifier (grep for
  this audit): the family vocabulary is human IDs only. The config key
  `rat_only_tokens: true` ("Only embed rat-specific tokens (T4)", `config/pipeline_config.yaml:343`) is not
  read by any code.
- April and v2: no new-token row comes from this file.

#### 6.3 Promoter (`pipeline/06_prior_knowledge/build_promoter_embedding.py`)
- Upstream prior: `prior_knowledge/promoter_emb/{human,mouse}_emb_768.pickle` (`utils.py:13-15`). Protocol as
  described in the repository: a window of 500 bp upstream and 2,000 bp downstream of the TSS, embedded by a
  DNABERT model that GeneCompass fine-tuned for 40 epochs on human and mouse promoters
  (`build_promoter_embedding.py:17-33,550-553`) [confirmed in the GeneCompass paper's Methods (Yang et al. 2024; `readings/` PDF page 13): 2,500 bases, 500 upstream of the TSS and 2,000 downstream; fine-tuned on DNABert for 40 epochs; 768 dimensions].
- Rat build:
  1. TSS: BioMart query for `transcript_start`, `transcript_end`, `strand` per gene, in batches of 500
     (`:287-304`); TSS = transcript start on the + strand, transcript end on the - strand; the **first row
     returned** per gene is kept (`:331-338`). The result is cached and a cache, if present, is reused without
     any check (`:274-277`, `:354-357`).
  2. Sequence: [TSS-500, TSS+2000] on the + strand, [TSS-2000, TSS+500] reverse-complemented on the - strand
     (`:494-508`); padded with N to 2,500 bp (`:511-512`); dropped if more than 10% N (`:514-520`).
  3. Embedding: split into overlapping 6-mers at stride 1 (`:589-591`); tokenize with `truncation=True,
     max_length=512` (`:606-612`); run the model without gradient; take position 0 of the last hidden state
     (`:615-617`). Model: `zhihan1996/DNA_bert_6` from the HuggingFace Hub (`config/pipeline_config.yaml:349`),
     loaded with `AutoModel` (`:584-585`).
  4. Store under the human ID where Stage 3 has one, else the rat ID (`:621-622`).
- April: the T3b and T4 rows present in the pickle were used (719 of 892 and 2,010 of 2,474 non-zero, from
  `tmp/data_audit/prior_zero_rows_by_tier.tsv:6-7`, local, uncommitted). v2: not used (zero by policy).

#### 6.4 GRN (`pipeline/06_prior_knowledge/build_grn_transfer.py`)
- Upstream prior: `prior_knowledge/PECA2vec/{human,mouse}_PECA_vec.pickle` (`utils.py:25-27`), keyed by gene
  symbol. The repository describes them as 768-dimensional vectors that GeneCompass trained from PECA2 applied
  to paired RNA-seq and ATAC-seq of 76 human and 84 mouse ENCODE contexts (`build_grn_transfer.py:16-22`)
  [confirmed in the GeneCompass paper's Methods (Yang et al. 2024; `readings/` PDF page 13): PECA2 networks from ENCODE paired expression and accessibility data, 84 mouse and 76 human, embedded with gene2vec].
- Rat build: no training. For each rat gene (`:320-358`): unified ID = human ortholog ID if Stage 3 has one,
  else the rat ID (`:325-326`); if there is a human ortholog, look up its symbol
  (`Gene_id_name_dict_human_mouse.pickle`) in the human PECA2vec dictionary (`:332-337`); otherwise, or on a
  miss, look up the gene's mouse ortholog symbol from `rat_mouse_orthologs.tsv` in the mouse dictionary
  (`:340-345`, map built at `:210-228`); otherwise a zero vector (`:318,328-329`). MoTrPAC ATAC-seq is not
  used, by design (`:24-28`).
- Coverage by tier (tabulated for this audit from `grn_transfer_coverage.tsv`): T1 13,178 human; T2a 705
  human; T2b 2,917 mouse + 696 zero; T3a 1,351 human (stored under the human ID); T3b 563 mouse + 329 zero;
  **T4 36 mouse + 2,438 zero**.
- April: T3b rows (563 non-zero) and T4 rows (36 non-zero) came from this file. v2: only the 36 T4 rows.

**Why exactly 36.** A T4 gene has no entry in `rat_to_human`, so the human path is never tried for it
(`build_grn_transfer.py:332`). The mouse path reads the raw BioMart ortholog table rather than Stage 3's
tiers. 156 T4 genes have a mouse ortholog row there whose mouse gene has no GeneCompass token (Background);
for 36 of them the mouse gene's symbol is a key of `mouse_PECA_vec.pickle`. All 36 are protein-coding
(tabulated for this audit); examples are Ifi47, Ccnb3, Ndufs3, Kbtbd4, Ptpmt1.

#### What v2 actually trained with (prior rows of the 4,717 new tokens)
From the build log of `data/models/rat_genecompass_init_v2` (`logs/init_v2_11910364.err:5,10-13`, local,
uncommitted) and its provenance
(`data/models/rat_genecompass_init_v2/provenance_rat_model_init.json:4,37-42,103-113`: git `9620e61`,
`--t3-priors ortholog`, `--t4-zero-priors promoter,coexp`, `--coexp-rescale 3.9`, 2,243 T3 orthologs found, 0
missing):

| Prior | T3a (1,351) and T3b (892) | T4 (2,474) |
|---|---|---|
| Promoter | 2,243/2,243 non-zero: the pre-trained row of the human (T3a) or mouse (T3b) ortholog token | 0 non-zero: zero by policy |
| Co-expression | 2,243/2,243 non-zero: ortholog's pre-trained row (no 3.9 rescale; that applies only to Stage 6 rows, `rat_load_prior_embedding.py:176-177`) | 0 non-zero: zero by policy |
| Gene family | 2,243/2,243 non-zero: ortholog's pre-trained row | 0 non-zero: no Stage 6 vector exists (0 found, 2,474 absent) |
| GRN | 2,243/2,243 non-zero: ortholog's pre-trained row | **36 non-zero** from Stage 6 (2,474 keys found, 2,438 are zero vectors) |

Settings: `finetune/genecompass/configs/rat_finetune_v2.yaml:79-81`. The rows of tokens 0-50,557 are
GeneCompass's own. Because the priors are buffers, not parameters
(`knowledge_embeddings.py:76,86,96,106`), training does not change them. The 2,243 T3 tokens share 1,094
ortholog tokens (`docs/RETRAIN_V2_RESULTS.md:91`), so many T3 paralogs start with identical prior rows.

#### The T3a zero-prior defect: history, current code, v2
- Defect (April model): the April 2026 reclassification gave T3 genes new tokens keyed by rat ID. Stage 6
  stores the vectors of T3a genes under the **human** ID (`build_coexp_embedding.py:217`,
  `build_promoter_embedding.py:621`, `build_grn_transfer.py:325-326`, `build_family_embedding.py:486-487`),
  while the loader looked new rows up by **rat** ID. Every T3a row was therefore zero in all four priors
  (`finetune/genecompass/rat_load_prior_embedding.py:33-36`). Recorded zero rows per tier in April's init and
  final checkpoints (`tmp/data_audit/prior_zero_rows_by_tier.tsv:5,11,17,23`, local, uncommitted): T3a
  1,351/1,351 zero in promoter, co-expression, family and GRN. The same table shows the other new tiers were
  partly empty too: T3b 173 (promoter), 554 (co-expression), 892 (family), 329 (GRN) zero of 892; T4 464,
  1,662, 2,474, 2,438 zero of 2,474 (`:6-7,12-13,18-19,24-25`).
- What the code does now (commits `916680e`, 2026-10-05, and `45daf2d`, 2026-10-06, `git log`): T3 rows copy
  the ortholog's pre-trained row for all four priors, T3a from the human ortholog and T3b from the mouse
  ortholog, using the same ortholog map as the word-embedding warm start
  (`rat_load_prior_embedding.py:158-161`; `finetune/genecompass/rat_model_init.py:74-112,292-304`); the
  loader raises if any T3 row is zero afterwards (`rat_load_prior_embedding.py:190-192`). Stage 6 itself was
  not changed: it still keys T3a vectors by human ID, and they are simply no longer read.
- The old behaviour remains selectable: `--t3-priors stage6` reproduces the rat-ID lookup that leaves T3a
  rows zero (`rat_model_init.py:246-248,498`). The default is `ortholog` (`:239`).
- v2: T3 2,243/2,243 non-zero in all four priors (table above).

### Models and software
| Component | Model / software | Architecture and settings | Where pinned |
|---|---|---|---|
| Co-expression, family | gensim `Word2Vec` (called "gene2vec" in the code) | skip-gram, 768 dimensions, window 1, `min_count` 1, 30 epochs; co-expression `workers=8`, `seed=42`; family `workers` and `seed` from config; other arguments at gensim defaults | call sites `build_coexp_embedding.py:521-530`, `build_family_embedding.py:589-598`; `gensim==4.4.0` in the current lock (`requirements.lock.txt:76`). The gensim version of the March build is not recorded in the manifests |
| Promoter | DNABERT-6, `zhihan1996/DNA_bert_6` (stock pre-trained weights from the Hub, not a GeneCompass fine-tune) | k-mer 6, stride 1, at most 512 tokens per sequence, `[CLS]` of the last layer, 768 dimensions, batch 32, inference only | model ID `config/pipeline_config.yaml:349`; revision `c56e67ea5827e0ddc67ef059addcf71569b1216e` (`:351`; taken from the local HF cache, `:350`; the March manifest records no revision, `stage6_promoter_manifest.json:9-18`); `transformers==4.57.6`, `torch==2.11.0` in the current lock (`requirements.lock.txt:221,226`); `pyfaidx` is required and not installed (`requirements.lock.txt:16-18`) |
| GRN | none (dictionary lookup of GeneCompass's PECA2vec vectors) | 768 dimensions (`build_grn_transfer.py:118`) | asset md5s `config/genecompass_assets_md5.tsv:27-28` |
| Correlation | pandas `DataFrame.corr(method='pearson', min_periods=10)` | pairwise-complete observations | `build_coexp_embedding.py:289`; `pandas==2.3.3` (`requirements.lock.txt:155`) |
| Reference genome | config now: Ensembl 115 GRCr8 top-level FASTA (`config/pipeline_config.yaml:107`; URL `build_promoter_embedding.py:149-152`). On-disk result: release-113 mRatBN7.2 FASTA (`logs/stage6_promoter_10422924.err:71`, local; `build_promoter_embedding.py:82-84`). Only the mRatBN7.2 FASTA is present under `data/references/genome/` (directory listing) | - | - |
| BioMart for TSS | config now: Ensembl 115 archive (`config/pipeline_config.yaml:120`; `build_promoter_embedding.py:156,776`). On-disk result: the then-live BioMart (`build_promoter_embedding.py:82-84`) | - | - |
| HGNC | local file fetched 2026-03-17, not in the HGNC archive, deposited; download fallback pinned to the 2026-03-06 archive | - | `config/pipeline_config.yaml:339-342`; `build_family_embedding.py:108-116` |

### Parameters, thresholds and flags
| Name | Value | Where set (path:line) | What it controls | Rationale given |
|---|---|---|---|---|
| `coexp.pcc_threshold` | 0.8 (strict >) | `config/pipeline_config.yaml:314`; comparison `build_coexp_embedding.py:292` | pair retained | "GeneCompass exact" (`config:314`) [confirmed in the paper, PDF page 13: "Gene pairs with PCCs larger than 0.8"] |
| `coexp.n_sample_cells` | 3,000 per file | `config/pipeline_config.yaml:315` | cells per correlation matrix | "GeneCompass exact" (`config:315`) [confirmed in the paper, PDF page 13: "selecting 3000 cells from each gene expression matrix"] |
| cell-sampling seed | 42, same for every file | hard-coded default, `build_coexp_embedding.py:179` | which cells | none stated |
| `coexp.min_shared_cells` | 10 | `config/pipeline_config.yaml:316` | minimum co-expressing cells for a correlation | "Methodological addition (guard sparse data)" (`config:316`; `build_coexp_embedding.py:46-50`) |
| `coexp.min_expr_ratio` | 0.05 (strict >) | `config/pipeline_config.yaml:317`; `build_coexp_embedding.py:268` | gene pre-filter | "Methodological addition (pre-filter)" (`config:317`; `build_coexp_embedding.py:51-54`) |
| expression threshold | count >= 1 | hard-coded, `build_coexp_embedding.py:267,282` | "expressed" | none stated |
| `coexp.chunk_size` | 2,000 genes | `config/pipeline_config.yaml:318` | memory only | - |
| `coexp.vector_size / window / sg / epochs / workers` | 768 / 1 / 1 / 30 / 8 | `config/pipeline_config.yaml:319-323` | Word2Vec | 768 = model hidden size (`build_coexp_embedding.py:501-503`) |
| co-expression Word2Vec `seed`, `min_count` | 42, 1 | hard-coded, `build_coexp_embedding.py:492,525` | - | none stated |
| raw-count warning | max(X) < 20 | hard-coded, `build_coexp_embedding.py:400` | warning only, never stops | `build_coexp_embedding.py:392-393` |
| checkpoint interval | 25 files | hard-coded, `build_coexp_embedding.py:133` | resume | - |
| `gene_family.min_family_size` | 2 | `config/pipeline_config.yaml:331` | derived families kept | "GeneCompass exact" (`config:331`) [the paper says only that any two genes of a family form a pair (PDF page 13); it states no minimum family size] |
| `gene_family.vector_size / window / sg / epochs` | 768 / 1 / 1 / 30 | `config/pipeline_config.yaml:332-335` | Word2Vec | as above |
| `gene_family.workers` | 1 now; **8 for the on-disk file** | `config/pipeline_config.yaml:336`; `stage6_family_manifest.json:13` | determinism | `config/pipeline_config.yaml:326-329` |
| `gene_family.seed`, `hash_seed` | 42, 0 | `config/pipeline_config.yaml:337-338` | Word2Vec seed, `PYTHONHASHSEED` | `build_family_embedding.py:746-752` |
| `gene_family.hgnc_url` | 2026-03-06 monthly archive | `config/pipeline_config.yaml:342` | download if the local file is missing | `config/pipeline_config.yaml:339-341` |
| `gene_family.rat_only_tokens` | true | `config/pipeline_config.yaml:343` | nothing: not read | - |
| `promoter.enabled`, `grn.enabled` | true, true | `config/pipeline_config.yaml:348,353` | steps 3 and 4 run | - |
| `promoter.dnabert_model`, `dnabert_revision` | `zhihan1996/DNA_bert_6`, `c56e67ea...` | `config/pipeline_config.yaml:349,351` | model | `config/pipeline_config.yaml:350` |
| window upstream / downstream | 500 / 2,000 bp | code defaults, `build_promoter_embedding.py:779-780`; not in the config | sequence window | "GeneCompass prior knowledge construction protocol" (`:17-24`) [confirmed in the paper, PDF page 13] |
| `max_n_fraction` | 0.10 | code default, `build_promoter_embedding.py:781` | drop sequences with more N | none stated |
| `batch_size` | 32 | code default, `build_promoter_embedding.py:782` | inference batch | - |
| k-mer, tokenizer `max_length` | 6, 512 | hard-coded, `build_promoter_embedding.py:540,611` | how much sequence the model sees | none stated |
| TSS batch, timeout | 500 genes, 120 s | hard-coded, `build_promoter_embedding.py:287,311,317` | BioMart queries | "avoid URL length limits" (`:286`) |
| `n_test_genes` | 200 | code default, `build_promoter_embedding.py:785` | `--test` mode only | - |
| GRN `VECTOR_SIZE` | 768 | hard-coded, `build_grn_transfer.py:118` | - | matches PECA2vec |
| `coexp_rescale_factor` | 3.9 | `rat_finetune_v2.yaml:79` (default `rat_load_prior_embedding.py:70`) | multiplies co-expression rows taken from Stage 6; no row qualifies in v2 | "new token co-exp norms are ~4.8 vs model expectation ~18.6" (`rat_load_prior_embedding.py:93`) |
| `t3_prior_source` | `ortholog` | `rat_finetune_v2.yaml:80` | T3 rows = ortholog's pre-trained rows | `rat_load_prior_embedding.py:23-31` |
| `t4_zero_priors` | `[promoter, coexp]` | `rat_finetune_v2.yaml:81` | T4 rows left zero | `rat_model_init.py:249-252` |

Seeds: cell sampling 42; Word2Vec 42 (co-expression: 8 threads, not deterministic; family: deterministic only
since 2026-11-01); no seed in promoter or GRN (inference and lookup).

### Statistical procedures
- **Co-expression pairs.** Statistic: Pearson correlation of raw counts over the cells (at most 3,000 per
  file) in which both genes have count >= 1. Decision: r > 0.8 with at least 10 such cells. No null
  distribution, no p-value, no multiplicity control over the gene pairs of a file, no adjustment for library
  size. A pair is kept if it passes in any one file; the number of files supporting it is discarded (set
  union, `build_coexp_embedding.py:784`).
- **Embeddings.** Word2Vec skip-gram on unordered pairs; no held-out objective is evaluated in the builders.
- **Promoter, GRN.** No statistical procedure.
- **Pre-specified test of a co-expression rebuild** (2026-10-05/06; local, uncommitted:
  `tmp/data_audit/coexp_t4_attempt/build_11891393.out:1-13`). T4 vectors were predicted as a weighted average
  of GeneCompass's human co-expression rows of their PCC partners on the cleaned corpus. 10% of the anchor
  genes (seed 0) were held out and rebuilt the same way. Accept only if (a) median cosine to the true row
  minus median cosine to a shuffled row >= 0.10 and (b) the true row is in the top 10 for >= 10% of held-out
  genes. This rule is written in the job output before the results; it is not in a committed
  pre-registration document.
- Nothing in Stage 6 is covered by `docs/RETRAIN_V2_PREREGISTRATION.md` beyond naming the v2 init as "fixed
  priors" (`:23`).

### Checks, validation and their results
| Check | Result | Rule | File |
|---|---|---|---|
| Are Stage 6 vectors in GeneCompass's coordinates? Genes present in both; direct cosine; held-out cosine after a fitted rotation vs shuffled null; top-10 retrieval | promoter: 10,212 genes, direct 0.305, rotated 0.650 vs null 0.651, top-10 0.4%. co-expression: 13,246 genes, direct 0.00, rotated 0.042 vs 0.038, top-10 1.7%, norm ratio 4.47. family: 16,913 genes, direct 0.00, rotated 0.777 vs 0.343, top-10 24.5%. GRN: 14,476 genes, direct 1.0, top-10 100% | descriptive | `tmp/data_audit/prior_space_alignment.tsv:2-5` (local, uncommitted; the script that wrote it was not identified); summary in `rat_load_prior_embedding.py:27-31` |
| Co-expression rebuild for T4, held-out test | 1,264 held-out genes with >= 3 partners: median cosine 0.034 vs 0.035 shuffled; top-10 0.1% (chance 0.07%); verdict REJECT | (a) >= 0.10 and (b) >= 10% | `tmp/data_audit/coexp_t4_attempt/build_11891393.out:30-38` (local, uncommitted); quoted in `rat_model_init.py:250-252` |
| Non-zero prior rows in the v2 init | T3 2,243/2,243 in all four; T4 0, 0, 0, 36 | loader raises if a T3 row is zero | `logs/init_v2_11910364.err:10-13` (local, uncommitted) |
| Rebuild (2026-10-31) | GRN byte-identical; family: same 8.86 M pairs and 26,019 keys, vectors differ (mean per-gene correlation 0.77, 8 gensim threads); co-expression and promoter not rerun | byte identity | `docs/repro_stages1_7/REPRODUCTION.md:51,60`; `docs/repro_stages1_7/stage6_prior_knowledge.tsv:2-21` |
| Init rebuilt twice from the rerun Stage 6 | all four prior buffers identical | tensor identity | `docs/repro_stages1_7/REPRODUCTION.md:52` |
| Downstream reading for the tokens left without priors | T4 tokens are predicted in context (gene-ID loss 19.20 at init to 2.20; 43.0% top-1), but their input embeddings show no family structure (V3 rank-AUC 0.471, verdict WEAK) | descriptive | `docs/RETRAIN_V2_RESULTS.md:624,633,656,668-669` |

### Deviations, known defects and caveats
1. **Documents overstate what T4 tokens received.** The manuscript says the T4 gene-family and GRN rows "are
   filled" (`manuscript/motrpac_genecompass.tex:294-296`, repeated at `:1261-1262`). In the v2 init 0 of 2,474
   T4 family rows and 36 of 2,474 T4 GRN rows are non-zero (`logs/init_v2_11910364.err:12-13`). `README.md:141`
   and `finetune/genecompass/README.md:56-58` state the 36 correctly.
2. **The family step cannot serve the tokens it is configured for** (6.2), and `rat_only_tokens` is a dead
   config key (`config/pipeline_config.yaml:343`).
3. **Promoter builder: 5,000 genes silently lost at the TSS fetch.** In the run that produced the on-disk
   file, 10 of the 45 BioMart batches added no gene (22,213 queried, 17,213 with a TSS;
   `logs/stage6_promoter_10422924.err:24-69`, local; 5,000 `no_tss_coordinate` rows in
   `data/training/prior_knowledge/promoter_skipped.tsv`, tabulated). No "BioMart query failed" line was
   logged: a response without tab-separated rows is skipped without an error
   (`build_promoter_embedding.py:322-327`), and a request that raises is only logged (`:348-349`). The
   incomplete result is cached and reused (`:274-277`); the cache file is still in the output folder.
4. **Promoter builder: the model sees about one fifth of the window.** A 2,500 bp sequence is 2,495 6-mers; the
   tokenizer truncates to 512 tokens (`build_promoter_embedding.py:589-591,606-612`), i.e. roughly the first
   515 bp, which after strand handling is the 500 bp upstream plus about 15 bp downstream of the TSS. The
   manifest and manuscript describe a 2.5 kb window (`stage6_promoter_manifest.json:10-12`;
   `manuscript/motrpac_genecompass.tex:1263-1264`).
5. **Promoter builder: not the upstream model.** The code comment says the model is "fine-tuned by GeneCompass
   on human+mouse promoters" (`build_promoter_embedding.py:545-553`); the model loaded is the stock Hub
   checkpoint (`config/pipeline_config.yaml:349`). This is why the vectors are not in GeneCompass's space
   (check above) and why v2 zeroes them.
6. **Promoter builder: mixed genome releases in the on-disk file.** It was built from the mRatBN7.2 FASTA with
   TSS coordinates from the then-live BioMart (`build_promoter_embedding.py:82-84`); the BioMart files the
   project fetched from the live host three weeks earlier were release 115 / GRCr8
   (`config/pipeline_config.yaml:31-33`). If the TSS query also returned GRCr8 coordinates, the windows were
   cut from the wrong assembly. 1,090 genes were dropped for N content, some with a fraction of 1.000
   (`promoter_skipped.tsv:2-3`, tabulated), which is consistent with that but not proof. Not verified.
7. **Promoter builder: "canonical TSS" is the first BioMart row** (`build_promoter_embedding.py:332-338`), and
   vectors of rat genes that share a human ortholog overwrite each other (`:621-622`): 16,120 sequences gave
   15,610 vectors (`stage6_promoter_manifest.json:21,23`).
8. **Co-expression: built on the uncleaned corpus and on some non-count matrices.** The March run read 864
   files (`stage6_coexp_manifest.json:32`), which include the files v2 later dropped as duplicates or non-rat
   and the files whose `X` is log-normalised; the builder never calls `raw_counts_adata`. The run log has 55
   "may be normalized" warnings (`logs/stage6_coexp_10423009.err`, count for this audit, local). It still
   cannot run on the v2 file list (`build_coexp_embedding.py:703-708`). Not used by v2.
9. **Co-expression: correlation of raw counts without depth adjustment or a null.** The rebuild test found
   these pairs carry no recoverable signal relative to GeneCompass's rows (check above).
10. **The 3.9 rescale** matched vector norms only (`rat_load_prior_embedding.py:91-94`); the alignment check
    shows norm was not the problem. It is inert in v2.
11. **Legacy path still reproduces the T3a defect** (`rat_model_init.py:246-248,498`).
12. **No overwrite protection on Stage 6 outputs.** Each builder writes straight into
    `prior_knowledge_dir` (`build_family_embedding.py:972-974`, `build_promoter_embedding.py:890-892`,
    `build_grn_transfer.py:553-555`). The documented `run_stage6.py --from 2` (`README.md:136`) would replace the
    March `family_embeddings.pkl`, which the config says cannot be regenerated
    (`config/pipeline_config.yaml:326-329`), and (once `pyfaidx` is installed) would rebuild the promoter file
    from the cached, incomplete March TSS table combined with a newly downloaded GRCr8 genome
    (`build_promoter_embedding.py:274-277,376-403`). A backup exists
    (`data/training/prior_knowledge_backup_2026-10-05/`). v2's init would not change (it reads 36 GRN rows,
    which reproduce).
13. **Stale text.** `run_stage6.py:5-7` says "Runs two steps" (there are four, `:102-151`);
    `config/pipeline_config.yaml:346` calls promoter and GRN "stubs, not yet implemented" directly above
    `enabled: true`; `build_grn_transfer.py:75-79` asks for a `method: "transfer"` key that the script never
    reads and the config does not have; `build_coexp_embedding.py:28` and `run_stage6.py:62` speak of "88
    studies"; the on-disk manifests label the reference "113 / mRatBN7.2"
    (`stage6_family_manifest.json:49-50`; `stage6_promoter_manifest.json:28-29`);
    `docs/repro_stages1_7/REPRODUCTION.md:58-59` reports that `run_stage6.py --from 2` fails without
    `coexp_embeddings.pkl`, which has since been changed to a warning (`run_stage6.py:291-308`).
14. **Family provenance.** The on-disk family file was built with an earlier Stage 3 token table (md5
    `26a810ed...`, `stage6_family_manifest.json:18`; today `258ffccb...`, computed for this audit;
    `docs/repro_stages1_7/REPRODUCTION.md:32-34`). `rat_to_human_mapping.pickle` is unchanged (`31790282...` in
    the manifest, `:23`, and today).
15. **GRN transfer details.** For a rat gene with several mouse orthologs the last table row wins
    (`build_grn_transfer.py:226-228`); T4 genes are never tried against the human dictionary (`:332`);
    `symbol_to_ensg` is built and never used (`:312`).
16. **Hard-coded values that bypass the config** (the README states there are none, `README.md:209`): the
    promoter window, N fraction, batch size, k-mer, token limit and TSS batch; the co-expression sampling seed,
    Word2Vec seed and `min_count`; Stage 4's target sum (see the parameter tables).
17. **The committed April config no longer describes April's priors.** `finetune/genecompass/configs/rat_finetune.yaml:71-73`
    carries `t3_prior_source: ortholog` and `t4_zero_priors: [promoter, coexp]`, added on 2026-10-05/06
    (`git log` for that file: `916680e`, `45daf2d`). April's init (`data/models/rat_genecompass_init/`, files
    dated 2026-04-02, no provenance file) was built before those options existed, with the rat-ID lookup. The
    only record of April's actual prior rows is the local, uncommitted table
    `tmp/data_audit/prior_zero_rows_by_tier.tsv`.

### Methods references
- Four prior-knowledge embeddings and their upstream construction: GeneCompass, Yang X, Liu G, Feng G, et al.
  Cell Research 2024;34:830-845 (`manuscript/motrpac_genecompass.tex:1479-1481`). Every protocol detail quoted
  above as "upstream" comes from this repository's code comments, not from the paper or from upstream code
  (the submodule ships the vectors only): none was checked against the paper in this audit.
- DNABERT: Ji Y, Zhou Z, Liu H, Davuluri RV. Bioinformatics 2021;37(15):2112-2120
  (`manuscript/motrpac_genecompass.tex:1517-1519`; named at `build_promoter_embedding.py:26`).
- gene2vec: named in the code (`build_coexp_embedding.py:494-503`) without a citation; not in the manuscript
  bibliography. Du J, et al. Gene2vec: distributed representation of genes based on co-expression. BMC
  Genomics 2019 [verified against Crossref; see REFERENCES.md].
- Word2Vec skip-gram (gensim): no citation in the repo. Mikolov T, et al. 2013 [verified against arXiv 1301.3781; see REFERENCES.md]; gensim: Rehurek R,
  Sojka P. 2010 [no registry record found; cited from memory].
- PECA2 / PECA2vec: named in the code (`build_grn_transfer.py:16-18`) without a citation. Duren Z, et al.
  [PECA: Duren Z, Chen X, Jiang R, Wang Y, Wong WH. PNAS 2017;114(25):E4914-E4923, verified, see REFERENCES.md; whether that paper describes the PECA2 version GeneCompass used was not checked].
- HGNC gene families: named in the code (`build_family_embedding.py:20,296-311`) without a citation [suggested: Seal et al. 2023, verified; see REFERENCES.md].
- Ensembl BioMart, release 115 archive (`config/pipeline_config.yaml:35-38`): no citation in the repo [suggested: Dyer et al. 2025 and Kinsella et al. 2011, verified; see REFERENCES.md].

---

## Items not determined
- How GeneCompass computed `human_gene_median_after_filter.pickle`, and the divisor it used for mouse cells
  (no upstream script; the mouse statement is a docstring, `build_hybrid_medians.py:21-23`).
- The git commit and package versions of the v2 Stage 4 and Stage 5 runs (no provenance file was written then)
  and of the March 2026 Stage 6 runs (the manifests record parameters only).
- gensim's default values for the Word2Vec arguments the code does not set (I did not open gensim's source).
- Why 10 BioMart batches returned no rows on 2026-03-18, and whether the 1,090 all-N or high-N promoter
  windows come from an assembly mismatch.
- Which QC files hold the 10,843,683 stored zero entries (Stage 4, caveat 8).
- The script that produced `tmp/data_audit/prior_space_alignment.tsv` and
  `tmp/data_audit/prior_zero_rows_by_tier.tsv`.
