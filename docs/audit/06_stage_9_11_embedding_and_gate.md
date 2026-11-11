# Part 6. Stage 9 (tokenize and embed the pseudo-cells) and Stage 11 (the exercise-signal gate and its corroboration)

> One part of the methods audit of 2026-11-11; start at [`README.md`](README.md). The main findings of every
> part are collected in [`FINDINGS.md`](FINDINGS.md) and every reference in [`REFERENCES.md`](REFERENCES.md).
> Citations are `path:line` at commit `8249506` unless the text says otherwise.

Audit date 2026-11-11. Repository root for every path below: `/depot/reese18/apps/motrpac-genecompass`
(branch `aim2b-perturbation-engine`). A citation `path:line` refers to the file as it stands on that date.

**Scope.** Sub-blocks 9a to 9h. 9a and 9b are Stage 9 proper (`pipeline/run_stage9.py`). 9c to 9g are the
"gate", which the code runs as its own orchestrator, `pipeline/run_stage11.py` ("Stage 11"). 9h is the UMAP
viewer, which no orchestrator runs.

**The run audited.** The default configuration: retrain-v2 model, refs4 references, all 16 tissues, clean
rebuild tree.

| Item | Value | Source |
|---|---|---|
| Stage 9 root (`genecompass_input_dir`) | `data/deconvolution/genecompass_input_v2_refs4all_rebuild` | `config/pipeline_config.yaml:407` |
| Gate folder (folder of `hotspots_table`) | `data/deconvolution/genecompass_input_v2_refs4all_rebuild/gate_rcv10/` | `config/pipeline_config.yaml:417`; `pipeline/run_stage11.py:89` |
| Augur exports (`augur_input_dir`) | `data/deconvolution/augur_input_v2_refs4all_rebuild` | `config/pipeline_config.yaml:419` |
| BayesPrism results (`results_dir`) | `data/deconvolution/results_refs4all_rebuild` | `config/pipeline_config.yaml:406` |
| Stage 9 jobs | 16 per-tissue jobs 12121714 to 12121729 | `deconvolution/REFS4ALL_REBUILD_RESULTS.md:6` |
| Gate job | 12122211, `python pipeline/run_stage11.py --jobs 48`, commit aee68a3 | `logs/rebuild_xtissue_12122211.out:1-2` |
| Local override file | sets only `deconvolution.n_cores = 4` | `config/pipeline_config.local.yaml` (read 2026-11-11) |

**Abbreviations.** CLS: the classification token that GeneCompass prepends to each cell's gene sequence;
its final-layer vector is the cell embedding. AUC: area under the receiver operating characteristic curve.
PLS-1: single-component partial least squares. CV: cross-validation. OOF: out-of-fold. BH:
Benjamini-Hochberg false discovery rate. RF: random forest. PCA: principal component analysis. UMAP:
uniform manifold approximation and projection. eta^2: fraction of variance explained by a factor. Z:
BayesPrism's per-sample, per-cell-type expression estimate. MC: Monte Carlo. Block: one tissue x cell type.
Pseudo-cell: the Z of one bulk sample for one cell type.

**Naming discrepancy to know before reading.** The gate is `pipeline/run_stage11.py` ("Stage 11",
`pipeline/run_stage11.py:3`), yet `pipeline/README.md:25` states "There is no Stage 11; the numbering skips
it", `pipeline/run_stage9.py:7-8` and `:236-237` call the gate "NOT part of the pipeline", and
`tests/test_gate_probe.py:1` calls it the "Stage 10 exercise gate". `deconvolution/REFS4ALL_REBUILD_RESULTS.md:25`
and the runbook (`deconvolution/DECONV_REBUILD_RUNBOOK.md:133`) use "Stage 11". The code is authoritative:
Stage 11 exists and produced the default run's gate.

---

## Stage 9a - Tokenization of pseudo-cells

### Research role
Turns each BayesPrism pseudo-cell into the input GeneCompass was trained on: a rank-ordered list of gene
tokens plus one value per token. Every later claim about embeddings assumes the pseudo-cells reach the
model in the corpus's input distribution. It is checked by unit tests of the arithmetic, by a data test that
every pseudo-cell is accounted for, and by a calibration of the value scale against the training corpus.
Verdict: the gene ranking uses the corpus recipe unchanged; the value channel differs from the corpus by
design (one extra library normalisation), and on the current default its per-tissue median ranges from 0.837
to 1.121 against a corpus median of 0.9145 (table below).

### Inputs and outputs
- Input: `<genecompass_input_dir>/<label>/pseudocells.h5ad` from Stage 8 (`pipeline/run_stage9.py:187`, `:198`).
- Token dictionary: `rat_tokens.pickle` under `paths.ortholog_dir` = `data/training/ortholog_mappings`
  (`deconvolution/tokenize_pseudocells.py:117`; `pipeline/05_tokenization/tokenize_corpus.py:154`;
  `config/pipeline_config.yaml:82`).
- Gene medians: `hybrid_gene_medians.pickle` under `paths.median_dir` = `data/training/gene_medians_v2`
  (`deconvolution/tokenize_pseudocells.py:118`; `tokenize_corpus.py:180`; `config/pipeline_config.yaml:87`).
  "Hybrid" means: the GeneCompass human median for a rat gene whose human ortholog has one; otherwise the
  Stage 4 rat median divided by a fallback scale (`pipeline/05_tokenization/build_hybrid_medians.py:12-15`,
  `:158-186`). The v2 file holds 21,222 genes, 14,766 with a human median and 6,456 with a rescaled rat
  median (`data/training/gene_medians_v2/hybrid_median_provenance.json`, key `coverage`).
- Outputs per tissue: `dataset/` (a HuggingFace dataset with columns `input_ids`, `values`, `length`,
  `species`, `cell_id`, and `sample`, `cell_type`, `tissue` when present; `tokenize_pseudocells.py:187-200`),
  `tokenize_summary.json` (`:201-205`), `provenance_tokenize_pseudocells.json` (`:206-209`).

Counts on the default run (pseudo-cells from line 1 of each `<label>/summary.txt`; the rest from each
`<label>/tokenize_summary.json`; all 16 used target sum 6,400, top-N 2,048, species 2):

| Tissue | Pseudo-cells built | Tokenized | Dropped (< 200 genes) | Eligible genes | Value median (non-zero) | Mean expressed length |
|---|---|---|---|---|---|---|
| bat | 300 | 300 | 0 | 14,162 | 0.837 | 2,044.9 |
| blood | 650 | 650 | 0 | 14,920 | 0.838 | 1,659.0 |
| cortex | 600 | 600 | 0 | 15,382 | 0.991 | 2,032.0 |
| heart | 800 | 800 | 0 | 15,864 | 0.907 | 2,015.1 |
| hippoc | 550 | 550 | 0 | 15,313 | 1.025 | 2,035.2 |
| hypoth | 650 | 650 | 0 | 14,061 | 0.983 | 2,039.7 |
| kidney | 750 | 750 | 0 | 14,892 | 0.988 | 2,032.9 |
| liver | 400 | 400 | 0 | 12,766 | 0.878 | 2,048.0 |
| lung | 1,050 | 1,032 | 18 | 15,233 | 1.121 | 1,680.8 |
| ovary | 432 | 432 | 0 | 14,175 | 0.958 | 1,934.0 |
| skmgn | 450 | 450 | 0 | 13,142 | 0.867 | 2,015.7 |
| skmvl | 450 | 450 | 0 | 13,141 | 0.883 | 2,034.7 |
| smlint | 650 | 650 | 0 | 12,874 | 0.850 | 2,048.0 |
| spleen | 700 | 700 | 0 | 13,179 | 0.941 | 1,999.9 |
| testes | 150 | 150 | 0 | 13,701 | 1.093 | 2,048.0 |
| watsc | 650 | 650 | 0 | 12,865 | 0.901 | 2,047.6 |
| **Total** | **9,232** | **9,214** | **18** | | | |

**9,232 versus 9,214.** No pseudo-cell fails in the embedding step. The 18 missing rows are LUNG Goblet-cell
pseudo-cells that the tokenizer drops because fewer than 200 eligible genes are expressed
(`deconvolution/tokenize_pseudocells.py:100-101`, `:176-177`;
`data/deconvolution/genecompass_input_v2_refs4all_rebuild/lung/tokenize_summary.json:2-3`). LUNG Goblet cells
therefore enter the gate with n = 32 instead of 50
(`data/deconvolution/genecompass_input_v2_refs4all_rebuild/pheno_merge_test.tsv:106`); the block's median
library is 1.3 reads, so it is not analysable anyway (9f). Docs agree: `pipeline/README.md:37`,
`deconvolution/AIM2_DECONV_RESULTS.md:18`, `deconvolution/REFS4_RERUN_RESULTS.md:169-170`.

### Procedure
1. `pipeline/run_stage9.py` resolves the tissue folder (`:180-187`) and the target sum (`:203-204`), then runs
   `deconvolution/tokenize_pseudocells.py --h5ad ... --out ... --target-sum <T> --species 2` (`:88-92`). The
   optional preference for training-regulated genes (`--pa-genes`) is off unless the flag is given (`:169-170`,
   `:202`); the default run did not use it (`.../liver/provenance_tokenize_pseudocells.json`, `pa_genes: null`).
2. Load the token dictionary and medians with the corpus tokenizer's own loaders; the eligible genes are
   those with both a token and a positive median (`tokenize_pseudocells.py:117-120`;
   `tokenize_corpus.py:194-195`, `:211`). Map the h5ad gene IDs to them after stripping the version suffix
   and upper-casing (`tokenize_corpus.py:393-398`). Abort if none map (`tokenize_pseudocells.py:124-125`).
3. **Library normalisation (the one step the corpus tokenizer does not have).** Each pseudo-cell is divided
   by its total over all genes in the h5ad and multiplied by the target sum; only then is it restricted to the
   eligible genes (`tokenize_pseudocells.py:132-136`). A zero total is replaced by 1 (`:134`).
4. Per pseudo-cell, the corpus function `tokenize_cell_batch` (`tokenize_pseudocells.py:169`): value =
   log2(1 + x / median) (`tokenize_corpus.py:342`); keep genes with value > 0 (`:345-346`); sort descending
   (`:357`); take the first `top_n` = 2,048 (`:358`; constant `TOP_N_DEFAULT`, `:122`); emit token IDs and
   values; pad both with zeros to length 2,048 (`:366-368`).
5. Drop pseudo-cells whose expressed length is below `--min-genes` = 200 (`tokenize_pseudocells.py:100-101`,
   `:176`).
6. Write the dataset with `species` = [[2]] for every row (`:191`), the summary and the provenance record.

**Does it match the training tokenization exactly?** No. Same and different:

| Aspect | Corpus (training) tokenization | Pseudo-cell tokenization | Same? |
|---|---|---|---|
| Per-cell function | `tokenize_cell_batch` (`tokenize_corpus.py:299-374`) | the same function, imported (`tokenize_pseudocells.py:43-46`, `:169`) | yes |
| Token dictionary and medians | `rat_tokens.pickle`, `hybrid_gene_medians.pickle` from the config paths | the same loaders and config paths (`tokenize_pseudocells.py:117-118`) | yes |
| Transform | raw count / median, log2(1 + x), no `normalize_total` (`tokenize_corpus.py:308`, `:338-342`) | library-normalise to 6,400 first, then the same (`tokenize_pseudocells.py:132-136`) | **no** |
| Input values | raw counts (`tokenize_corpus.py:441-446`) | BayesPrism Z, real-valued, bulk scale | **no** |
| Sequence length | top 2,048, zero-padded | the same constant | yes |
| Cell filter | only empty cells dropped (`tokenize_corpus.py:555-557`); the 200-gene filter is in Stage 2 (`config/pipeline_config.yaml:212`) | expressed length >= 200 (`tokenize_pseudocells.py:176`) | equivalent threshold, applied later |
| Species | [[2]] (`tokenize_corpus.py:562`) | [[2]] (`tokenize_pseudocells.py:191`) | yes |

A per-cell scale factor cannot change the ranking, so the token sequence (`input_ids`) is unaffected by the
target sum; only `values` moves (unit test `tests/test_tokenization.py:39-44`).

### Models and software
No model. Python 3.12.11 virtual environment frozen in `requirements.lock.txt:4`: numpy 2.4.3 (`:134`),
anndata 0.12.10 (`:31`), datasets 4.8.4 (`:56`), scipy 1.17.1 (`:198`).

### Parameters, thresholds and flags
| Name | Value | Where set (path:line) | What it controls | Rationale given |
|---|---|---|---|---|
| `pseudocell_target_sum` | 6,400 | `config/pipeline_config.yaml:413`; read at `pipeline/run_stage9.py:203-204` and `tokenize_pseudocells.py:115-116` | per-pseudo-cell library total before the median division | calibration rule, `docs/RETRAIN_V2_PREREGISTRATION.md:155-164` (see below) |
| fallback target sum | 6,500 | `pipeline/run_stage9.py:204`; `tokenize_pseudocells.py:116`; `pipeline/run_stage12.py:216` | used only if the config key is absent | April's value (`config/profiles/april.yaml:22`) |
| `--top-n` | 2,048 | `tokenize_pseudocells.py:98`; constant `tokenize_corpus.py:122` | sequence length | "GeneCompass paper: top-2048 genes per cell" (`tokenize_corpus.py:122`). Hard-coded: not read from `reference_assembly.top_n_genes` (`config/pipeline_config.yaml:294`) |
| `--min-genes` | 200 | `tokenize_pseudocells.py:100-101` | drops sparse pseudo-cells | "GeneCompass QC parity" (`:101`). Hard-coded default |
| `--species` | 2 (rat) | `pipeline/run_stage9.py:92`; default `tokenize_pseudocells.py:99` | species column | none stated |
| `--pa-genes` | not used | `pipeline/run_stage9.py:169-170`, `:202` | optional promotion of training-regulated genes into the top-N | off by default |
| `--pa-max-rank` | 4,096 | `tokenize_pseudocells.py:104-107` | only with `--pa-genes` | "2x top-N" (`:106`) |
| `CELL_BATCH_SIZE` | 1,024 | `tokenize_corpus.py:121` | batch size, no effect on output | memory |
| Printed corpus reference | median 0.869, mean 0.960 | `tokenize_pseudocells.py:181-182` | a printed comparison only | hard-coded; these are April-corpus values (`data/results/retrain_v2/c5/stage9_calibration/calibration.json:3`) |

### Statistical procedures
None; the step is deterministic. The only fitted quantity is the target sum. Rule
(`docs/RETRAIN_V2_PREREGISTRATION.md:155-164`, logged 2026-10-24 before any v2 downstream output;
implemented in `data/results/retrain_v2/c5/stage9_calibration/calibrate.py:34-37`): one target sum T for all
tissues, chosen so that LIVER's pseudo-cells have the corpus's median non-zero token value. Because a value
is log2(1 + T r), T* = 6,500 x (2^m_corpus - 1) / (2^m_liver(6,500) - 1), rounded to the nearest 100. Corpus
medians are taken over every 211th row (`calibrate.py:17`): April 0.8690, v2 0.9145 (`calibration.json:3-4`).
LIVER at 6,500 gave 0.9302 with v2 medians, so T* = 6,352, rounded to 6,400 (`calibration.json:45`, `:80-81`).

### Checks, validation and their results
- Unit tests of the arithmetic, ties, padding, top-N truncation, scale invariance of the ranking and the
  gene-ID mapping: `tests/test_tokenization.py:19-59`. Tests of the optional gene-preference variant:
  `:78-104`.
- Data tests on the default tree: every tissue used the configured target sum and species 2
  (`tests/test_v2_outputs.py:36-40`); tokenized + dropped = pseudo-cells in the h5ad (`:43-47`).
- Value scale: per-tissue medians in the table above (0.837 to 1.121), against the v2 corpus median 0.9145.
- The rebuild's embeddings are byte-identical to the earlier assembled tree's
  (`deconvolution/REFS4ALL_REBUILD_RESULTS.md:24`); of the files compared in the Stage 9 root, 788 are
  identical and the 4 that differ are Stage 10 tables (`:36`, `:44-46`).

### Deviations, known defects and caveats
1. **The calibration was made on the July-reference LIVER pseudo-cells and has not been repeated for the
   current references.** The config comment says 6,400 is "calibrated so the pseudo-cells' token-value median
   matches the model's corpus" (`config/pipeline_config.yaml:411-412`). On the default run LIVER's median at
   6,400 is 0.878 (`.../liver/tokenize_summary.json`), not 0.9145; applying the registered formula to that
   figure gives about 6,800 (auditor's arithmetic, not a pipeline output). Keeping 6,400 for the refs4 runs
   was itself registered (`deconvolution/REFERENCE_RELABEL_PREREG.md:339-340`), so this is a stated choice,
   but the comment's claim no longer holds for the tissues as they are now. Token ranks are unaffected.
2. **Stale docstring.** `tokenize_pseudocells.py:13-18` and its usage example (`:26`) present 10,000 as "the
   PRINCIPLED target" because the Stage 4 medians were built on `normalize_total(10,000)` values. The default
   is 6,400, and 70% of v2's medians are GeneCompass human medians, not Stage 4 rat medians (see Inputs).
3. **Stale printed reference.** The script prints "corpus reference: median 0.869, mean 0.960"
   (`tokenize_pseudocells.py:181-182`), the April corpus. The v2 corpus median is 0.9145.
4. **Stale figures in a doc.** The "Value-channel QC" paragraph of `deconvolution/AIM2_DECONV_RESULTS.md:103`
   lists per-tissue drifts for April at 6,500 (for example kidney +23.3%); they do not describe the default run.
5. **Near-empty pseudo-cells are rescaled to a full library.** Step 3 scales any non-zero pseudo-cell to
   6,400, whatever its original total (`tokenize_pseudocells.py:133-135`). LUNG Goblet cells have a median
   library of 1.3 reads (`.../gate_rcv10/corroboration_merged.tsv`, row "lung / Goblet cells",
   `median_libsize`), yet 32 of 50 pass the 200-gene filter and are embedded. Nothing in Stage 9 removes such
   blocks; the library rule of 9f removes them from the hotspot test only. The same point is made in
   `deconvolution/REFERENCE_RELABEL_PREREG.md:30-31`.
6. **Stale ID-space label.** Stage 8's `summary.txt` says `gene_space=rel-113_ENSRNOG`
   (`.../lung/summary.txt:2`) and the tokenizer's help text says "rel-113 ENSRNOG"
   (`tokenize_pseudocells.py:102-103`); the canonical ID space is Ensembl 115 (`config/pipeline_config.yaml:31-36`).
   The mapping itself works (eligible genes equal or nearly equal the pseudo-cell genes in every tissue).
7. **Provenance without checksums.** The record stores size and modification time of the two reference
   *directories*, not an md5 of the token or median files (`tokenize_pseudocells.py:206-207`;
   `lib/provenance.py:82-84`).
8. Thirteen of the 16 tissues' Stage 9 steps ran at commit bf0f417 with uncommitted edits in the working tree
   (among them the Stage 8 script `deconvolution/R/run_deconvolution.R`), three at aee68a3
   (`.../<tissue>/embeddings/provenance_embed_cells.json:4-12` and the tokenize records).
   `deconvolution/REFS4ALL_REBUILD_RESULTS.md:4` says "at commit aee68a3". The Stage 9 scripts do not differ
   between the two commits (`git diff --stat bf0f417 aee68a3` over `tokenize_pseudocells.py`, `embed_cells.py`,
   `run_stage9.py`, `model_checkpoint.py`, `tokenize_corpus.py` is empty), so Stage 9 is unaffected; the
   Stage 8 edit is for that section.

### Methods references
- GeneCompass tokenization (rank genes by count/median, top 2,048): Yang X, Liu G, Feng G, et al. Cell
  Research 2024;34:830-845 (`manuscript/motrpac_genecompass.tex:1487-1489`).

---

## Stage 9b - Embedding

### Research role
Produces one 768-dimensional vector per pseudo-cell from the rat GeneCompass model. Everything in 9c to 9h
and Stages 12 and 13 reads these vectors. Verdict: 9,214 embeddings, reproducible byte for byte.

### Inputs and outputs
- Input: `<label>/dataset` (9a) and the model folder.
- Output: `<label>/embeddings/cell_embeddings.npy`, N x 768, 32-bit floats (BAT: 921,728 bytes = 300 x 768 x 4
  + 128 header; `.../gate_rcv10/provenance_subspace_probe.json:32-34`), and
  `<label>/embeddings/provenance_embed_cells.json`.
- Row k of the array is row k of the dataset (`finetune/genecompass/embed_cells.py:70-73`, `:134-139`).

### Procedure
1. `pipeline/run_stage9.py` resolves the model folder (`:188`) and the number of cells: the pseudo-cell count
   read from `summary.txt`, or 1,000,000 if absent (`:71-78`, `:190-193`, `:58`), then runs `embed_cells.py
   --model-dir ... --dataset ... --output ... --n-cells N --species 2 --device cuda` (`:99-105`).
2. `embed_cells.py` reads `config.json` and `pytorch_model.bin` (`:90-103`), rebuilds the four prior-knowledge
   tensors from the state dict (`:106-111`) and the human-mouse homolog map from the vendored file (`:113-116`),
   builds `BertForMaskedLM` (`:121-122`), replaces the species embedding by a 3-row table (`:124-126`), loads
   the weights with `strict=False` (`:129`) and switches to evaluation mode (`:130`).
3. The dataset is capped only if `--n-cells` is smaller than its length, and then to an order-preserving
   prefix (`:134-139`). On the default run N was never smaller (for example LUNG: N = 1,050, dataset 1,032).
4. Batches of 32 (`:55`, `:193`): `input_ids`, `values`, a species tensor filled with the command-line value
   (`:150-154`), attention mask = token ID not 0 (`:157`); forward pass through `model.bert` with
   `emb_warmup_alpha=1.0` (`:160-167`).
5. The embedding is `outputs[0][:, 0, :]` (`:170`), saved without normalisation (`:176-180`).

**Which representation.** `outputs[0]` is the last hidden state of the encoder
(`vendor/GeneCompass/genecompass/modeling_bert.py:424-431`). Position 0 is the species-specific CLS vector,
which the model prepends to the gene embeddings before the encoder (`modeling_bert.py:405-410`). So the cell
embedding is the **final-layer (layer 12) hidden state at the CLS position**: not a mean over gene tokens,
not the pooler output, not an intermediate layer. The encoder sees 2,049 positions (CLS + 2,048 genes).

**Which checkpoint.** Precedence (`deconvolution/model_checkpoint.py:3-9`, `:27-43`): (1) `--model-dir`;
(2) `deconvolution.model_checkpoint` in the config, which must contain `config.json`; (3) the
highest-numbered `checkpoint-*` under `deconvolution.genecompass_model_dir`, with a warning. The default
config sets (2) to `data/models/rat_genecompass_finetuned_v2/models/rat_phase2_mixed_species/models`
(`config/pipeline_config.yaml:364`), the final save of phase 2. Every one of the 16 provenance records shows
that folder (`.../<tissue>/embeddings/provenance_embed_cells.json`, `model_dir_resolved`). The config comment
says its `pytorch_model.bin` has the same md5 as `checkpoint-591763` (`config/pipeline_config.yaml:362-363`);
this audit did not recompute the md5 (no checkpoint reads on the login node); the two files have the same
size, 1,224,674,943 bytes (directory listing; `.../liver/embeddings/provenance_embed_cells.json:43`).

### Models and software
| Item | Value | Source |
|---|---|---|
| Model class | `BertForMaskedLM` from the vendored GeneCompass package | `embed_cells.py:83`, `:122` |
| Vendored code | `vendor/GeneCompass` at commit 66ab82f | `git submodule status` |
| Layers / heads / hidden / feed-forward | 12 / 12 / 768 / 3,072 | model `config.json:18`, `:17`, `:11`, `:13` |
| Vocabulary | 55,275 tokens | `config.json:30` |
| Maximum gene positions | 2,048 | `config.json:15` |
| Inputs used | token IDs, values, CLS token, four priors (promoter, co-expression, gene family, GRN) | `config.json:24-29` |
| Species table | 3 rows: 0 human, 1 mouse, 2 rat | `embed_cells.py:77`, `:124-126` |
| Precision | 32-bit; no autocast in the script | `embed_cells.py:145-171` |
| torch / transformers / datasets | 2.11.0 / 4.57.6 / 4.8.4 | `requirements.lock.txt:221`, `:226`, `:56`; `config.json:21` |
| Hardware | A100 GPUs | `deconvolution/REFS4ALL_REBUILD_RESULTS.md:110` |

(`config.json` = `data/models/rat_genecompass_finetuned_v2/models/rat_phase2_mixed_species/models/config.json`.)

### Parameters, thresholds and flags
| Name | Value | Where set (path:line) | What it controls | Rationale given |
|---|---|---|---|---|
| `model_checkpoint` | `.../rat_phase2_mixed_species/models` | `config/pipeline_config.yaml:364` | weights | "explicit, so a new checkpoint cannot change results" (`:362`) |
| `--species` | 2 | `pipeline/run_stage9.py:104` | which CLS vector is prepended | rat |
| `--n-cells` | pseudo-cell count per tissue | `pipeline/run_stage9.py:190-193` | cap (never binding) | "never subsample" (`:164-165`) |
| `_FALLBACK_N_CELLS` | 1,000,000 | `pipeline/run_stage9.py:58` | cap when `summary.txt` is missing | hard-coded |
| `--batch-size` | 32 | `embed_cells.py:193` | inference batch | none stated; hard-coded default |
| `num_species` | 3 | `embed_cells.py:125` | size of the species table | hard-coded |
| `emb_warmup_alpha` | 1.0 | `embed_cells.py:165` | full weight on the prior-knowledge embeddings | hard-coded |
| `warmup_steps`, `emb_warmup_steps` | 0, 1 | `embed_cells.py:119-120` | overwritten config entries for inference | hard-coded |
| Token dictionary path | `data/training/ortholog_mappings/rat_tokens.pickle` | `embed_cells.py:96` | loaded, then never used | hard-coded; bypasses `paths.ortholog_dir` |
| Homolog map path | `vendor/GeneCompass/prior_knowledge/homologous_hm_token.pickle` | `embed_cells.py:113` | model construction | hard-coded; bypasses `paths.genecompass_homologs` (`config/pipeline_config.yaml:99`) |

### Statistical procedures
None.

### Checks, validation and their results
- Determinism: every `cell_embeddings.npy` of the rebuild is byte-identical to the earlier tree
  (`deconvolution/REFS4ALL_REBUILD_RESULTS.md:24`, `:110`).
- Row alignment is checked by every consumer, which skips a tissue when the array and dataset lengths differ
  (`deconvolution/embed_qc.py:35-36`; `pheno_merge_test.py:81-82`; `subspace_probe.py:273-274`).
- No unit test covers `embed_cells.py` or `model_checkpoint.py` (`tests/` holds none; searched by name).

### Deviations, known defects and caveats
1. **The April profile loads the v2 weights.** `config/profiles/april.yaml:21` sets `genecompass_model_dir`
   but not `model_checkpoint`, and the base config's `model_checkpoint` (`config/pipeline_config.yaml:364`)
   takes precedence (`deconvolution/model_checkpoint.py:33-38`). Under `PIPELINE_PROFILE=config/profiles/april.yaml`,
   Stages 9, 12 and 13 would therefore embed with the v2 checkpoint on April's medians and target sum unless
   `--model-dir` is passed. The config key and the resolver were added on 2026-11-01 (commits c6be8aa and
   d38b83a); the profile was not updated, and `tests/test_config.py:64-69` does not test the checkpoint. The
   default run is not affected. This reading is from the code; the April arm was not rerun to confirm it.
2. **Silent partial loading.** `load_state_dict(state, strict=False)` (`embed_cells.py:129`) discards the list
   of missing and unexpected keys. A checkpoint whose keys did not match would leave those weights at their
   random initial values without any message. Nothing in the script or tests asserts a clean load.
3. **Hard-coded paths and an unused load** (table above): `embed_cells.py:96-98` loads the rat token
   dictionary and never uses it; `:96` and `:113` ignore the config.
4. `_PROJECT_ROOT` in `embed_cells.py:39` is the `PIPELINE_ROOT` environment variable or the current
   directory. The orchestrator sets the variable (`pipeline/run_stage9.py:224`); a direct call from another
   directory would resolve the two hard-coded paths wrongly.
5. The dataset's `species` column is not read; the species comes from the command line (`embed_cells.py:152-154`).
   Both are 2 in the default run.
6. Provenance stores size and time of the weights, not an md5 (`embed_cells.py:211-214`).
7. Stale comments: `pipeline/run_stage9.py:27-29` says `embed_cells` "random-subsamples and breaks row
   alignment"; the code has used an order-preserving prefix since that was fixed (`embed_cells.py:134-139`).
   `config/pipeline_config.yaml:408-409` says Stage 9 "picks the latest checkpoint-*"; with `model_checkpoint`
   set it does not.

### Methods references
- GeneCompass: Yang X, Liu G, Feng G, et al. Cell Research 2024;34:830-845
  (`manuscript/motrpac_genecompass.tex:1487-1489`).

---

## Stage 9c - Embedding QC (cell type / sex / batch separation)

### Research role
Asks whether the pseudo-cell embeddings of a tissue separate by cell type while keeping within-type
variation. It is a sanity check and, in the v2 pre-registration, a comparison metric between models
(`docs/RETRAIN_V2_PREREGISTRATION.md:80`, `:205-207`). No pass threshold is defined anywhere in the code.
Verdict on the default: median cosine silhouette 0.810 over 16 tissues, lowest BAT 0.382 and LUNG 0.520.

### Inputs and outputs
Input: each `<label>/embeddings/cell_embeddings.npy` and `<label>/dataset`. Output: a table on standard
output only (`deconvolution/embed_qc.py:27-28`, `:46-47`); it is not written to a file. For the default run
it survives in `logs/rebuild_xtissue_12122211.out:11-29`.

### Procedure
`pipeline/run_stage11.py` step 1 runs `embed_qc.py <genecompass_input_dir>` (`:54`). Per tissue
(`deconvolution/embed_qc.py:29-47`):
1. L2-normalise the embeddings (`:37`).
2. **Silhouette** by cell type with cosine distance over all pseudo-cells of the tissue (`:38`).
3. **kNN purity**: for each pseudo-cell, the share of its 10 nearest neighbours (cosine; 11 requested, self
   removed) with the same cell type, averaged (`:39-40`).
4. **Variance split** on the un-normalised embeddings: between-cell-type and within-cell-type sums of
   squares as percentages of their sum (`:41-45`).
5. **Value median**: median of the non-zero token values of the tissue's dataset (`:44`).

Sex and batch: `embed_qc.py` computes neither. Sex separation is measured in the gate (9d: eta^2 for sex and
the sex probe; 9e: Augur on sex). A batch-separation metric for the pseudo-cell embeddings was **NOT FOUND**
(searched `deconvolution/embed_qc.py`, `pheno_merge_test.py`, `subspace_probe.py`, `pipeline/run_stage11.py`).
Technical covariates (RNA integrity, globin) are examined at the differential-expression level in Stage 14,
outside this section.

### Models and software
scikit-learn 1.8.0 (`requirements.lock.txt:196`): `silhouette_score`, `NearestNeighbors`, `normalize`
(`embed_qc.py:18-20`). scikit-learn is in the lock file but not in `requirements.txt`.

### Parameters, thresholds and flags
| Name | Value | Where set | What it controls | Rationale |
|---|---|---|---|---|
| Distance | cosine | `embed_qc.py:38-39` | silhouette and kNN | none stated |
| k | 10 neighbours | `embed_qc.py:39-40` | kNN purity | none stated; hard-coded |
| Root | `genecompass_input_dir` or argv[1] | `embed_qc.py:26` | which run | config |

### Statistical procedures
Descriptive only; no test, no null, no seed (the metrics are deterministic).

### Checks, validation and their results
Default run (`logs/rebuild_xtissue_12122211.out:13-29`):

| Tissue | Cells | Types | Silhouette | kNN purity | Between % | Within % | Value median |
|---|---|---|---|---|---|---|---|
| bat | 300 | 6 | 0.382 | 0.97 | 43.6 | 56.4 | 0.837 |
| blood | 650 | 13 | 0.553 | 0.87 | 81.9 | 18.1 | 0.838 |
| cortex | 600 | 12 | 0.912 | 1.00 | 96.5 | 3.5 | 0.991 |
| heart | 800 | 16 | 0.876 | 1.00 | 92.9 | 7.1 | 0.907 |
| hippoc | 550 | 11 | 0.738 | 0.99 | 93.0 | 7.0 | 1.025 |
| hypoth | 650 | 13 | 0.765 | 0.99 | 88.8 | 11.2 | 0.983 |
| kidney | 750 | 15 | 0.768 | 1.00 | 88.2 | 11.8 | 0.988 |
| liver | 400 | 8 | 0.852 | 1.00 | 91.5 | 8.5 | 0.878 |
| lung | 1,032 | 21 | 0.520 | 0.97 | 79.1 | 20.9 | 1.121 |
| ovary | 432 | 18 | 0.747 | 1.00 | 88.3 | 11.7 | 0.958 |
| skmgn | 450 | 9 | 0.923 | 1.00 | 94.2 | 5.8 | 0.867 |
| skmvl | 450 | 9 | 0.870 | 1.00 | 90.9 | 9.1 | 0.883 |
| smlint | 650 | 13 | 0.852 | 1.00 | 95.0 | 5.0 | 0.850 |
| spleen | 700 | 14 | 0.921 | 1.00 | 95.3 | 4.7 | 0.941 |
| testes | 150 | 6 | 0.988 | 1.00 | 99.5 | 0.5 | 1.093 |
| watsc | 650 | 13 | 0.692 | 1.00 | 83.0 | 17.0 | 0.901 |

Median silhouette 0.810 (mean of the 8th and 9th values, 0.768 and 0.852); notebook 9 prints the same
(`notebooks/pipeline9_analysis.ipynb:635-637`). For the model comparison on the July references the medians
were April 0.712 and v2 0.731 (`docs/RETRAIN_V2_RESULTS.md:150-166`;
`data/results/retrain_v2/c5/stage9_embed_qc.txt`).

### Deviations, known defects and caveats
1. `deconvolution/AIM2_DECONV_RESULTS.md:19` says the 16-tissue silhouette was "not recomputed". It was: the
   gate log and notebook 9 both hold it (above). The doc line is stale.
2. The table is printed, not saved, and the script writes no provenance. A rerun is needed to recover it
   if the log is lost.
3. The script lists every subfolder of the root, so non-tissue folders appear as "(no embeddings yet)"
   (`logs/rebuild_xtissue_12122211.out:16`, the `gate_rcv10` line). Harmless.
4. What the metric can show (auditor's note): the pseudo-cells of one cell type are BayesPrism estimates
   built from the same reference profile, so high cell-type separation shows that the model keeps
   distinctions already present in Z. It is not a test on independent cells. The reference-cell embedding
   check (criterion C3, `docs/RETRAIN_V2_PREREGISTRATION.md:53-62`) is the independent one and belongs to the
   model section.

### Methods references
- Silhouette and kNN purity: standard clustering diagnostics; no citation in the code or manuscript
  bibliography. Rousseeuw PJ. J Comput Appl Math 1987;20:53-65 [verified against Crossref; see REFERENCES.md].

---

## Stage 9d - The gate (`pheno_merge_test.py`, `subspace_probe.py`, and the 10 splits x 10,000 permutations version in `gate_rcv10/`)

### Research role
Asks, for each tissue x cell type, whether the pseudo-cell embeddings carry a detectable difference between
trained and sedentary animals. The blocks that pass (the "hotspots", defined in 9f) are the ones Stages 10,
12 and 13 treat as exercise-responsive. Two measures are computed: a global variance screen
(`pheno_merge_test.py`), kept as a conservative description, and a cross-validated supervised probe
(`subspace_probe.py`), which is the detector the hotspots are based on (`pipeline/run_stage11.py:15-16`).
Verdict on the default: 9 hotspots among 151 analysable blocks (9f); the global screen retains none after
BH correction.

### Inputs and outputs
- Inputs: every `<label>/embeddings/cell_embeddings.npy` and `<label>/dataset`;
  `deconvolution/reference/motrpac_sample_pheno.tsv` (`config/pipeline_config.yaml:387`; 6,156 vial labels,
  columns `viallabel`, `sex`, `group`, `key.anirandgroup`, `intervention`, `sacrificetime`,
  `deconvolution/reference/motrpac_sample_pheno.tsv:1`); `data/deconvolution/motrpac_bulk/<TISSUE>/bulk_samples.tsv`
  (`config/pipeline_config.yaml:441`).
- Outputs: `<root>/pheno_merge_test.tsv` (197 rows; columns `tissue, cell_type, n, eta2_group, p_group,
  eta2_trained, p_trained, eta2_sex, p_sex`); `<root>/gate_rcv10/subspace_probe.tsv` (197 rows, 33 columns)
  and `provenance_subspace_probe.json`.
- 197 blocks in 16 tissues. Block sizes: n = 50 in 172 blocks, 24 in the 18 OVARY blocks, 25 in the 6 TESTES
  blocks, 32 in LUNG Goblet cells (`.../gate_rcv10/corroboration_merged.tsv`, column `n`).

### Procedure
**Unit of analysis.** One block = the pseudo-cells of one cell type in one tissue; each row is one bulk
sample (one animal's tissue). A full block is 2 sexes x 5 groups x 5 animals = 50 rows.

**Phenotype join** (identical in `pheno_merge_test.py:83-89`, `subspace_probe.py:275-282`,
`augur_prep.py:99-107`, `umap_embeddings.py:51-58`). Stage 8 names samples `mix1 ... mixN`; the scripts take
the digits of that name as a position, read the vial label on that line of `bulk_samples.tsv`, and look it
up in the phenotype table. The join is therefore by row order, not by an identifier
(`pheno_merge_test.py:8-10`).

**Groups compared.** `trained` = any group whose name does not contain "control"; `control` otherwise
(`subspace_probe.py:280-282`). The groups are `control`, `1w`, `2w`, `4w`, `8w`; so the contrast is **all
four training durations pooled (40 animals in a full block) against the sedentary controls (10 animals)**,
not 8 weeks against control. The controls are all from the eight-week arm (`sacrificetime` = 8w for every
control row of the phenotype table).

**Blocks kept.** A block needs at least 6 rows with known group and sex (`pheno_merge_test.py:91-93`;
`subspace_probe.py:284-286`).

**Step 2 of Stage 11: the global screen** (`pipeline/run_stage11.py:55-57`; `deconvolution/pheno_merge_test.py`).
For GROUP (5 levels), TRAINED (2) and SEX (2): eta^2 = between-group sum of squares / total sum of squares,
summed over all 768 dimensions (`:37-45`), with a label-permutation p-value (`:48-51`). GROUP is skipped
when any group has fewer than 3 rows (`:62`, `:95`, `:98-99`).

**Step 3: the supervised probe** (`pipeline/run_stage11.py:58-60`; `deconvolution/subspace_probe.py`), run as
`--perms 10000 --folds 5 --repeats 10 --jobs 48` (`logs/rebuild_xtissue_12122211.out:238-240`;
`.../gate_rcv10/provenance_subspace_probe.json:246-249`). Per block (`process`, `:177-235`):
1. Features: the raw 768-dimensional embedding in 64-bit floats (`:271`). No standardisation, no PCA, no
   feature selection.
2. Splits: `repeats` = 10 stratified k-fold splits, k = min(5, size of the smaller class), shuffled, split r
   seeded with block seed + 1 + 7919 r (`make_folds`, `:113-120`; `make_fold_sets`, `:123-127`; call `:215`).
   Stratification is on the trained/control label only. In a full block each test fold holds 2 controls and
   8 trained; in OVARY and TESTES (5 controls), 1 control and 3 or 4 trained.
3. Model, per fold (`oof_scores`, `:130-140`): centre on the training-fold mean mu; weight vector
   w = (X_train - mu)^T (y_train - mean(y_train)), scaled to unit length; score of a held-out row =
   (x - mu) . w. This is the first PLS component (the maximum-covariance direction). It has no tuning
   parameter and no regularisation term.
4. Statistic (`supervised`, `:143-173`): for each split, the AUC of the pooled out-of-fold scores against
   the label (`:148-154`); the block's `sup_trained_auc` is the mean over the 10 splits (`:158-161`).
   `sup_trained_eta2` is the mean over splits of the one-way eta^2 of the out-of-fold score by label
   (`:164-168`).
5. Null and p-value: 9d "Statistical procedures".
6. The same machinery gives two more probes: dose (ordinal weeks 0, 1, 2, 4, 8, `:69`; folds stratified on
   the 5-level group, `:223`; statistic = Spearman correlation of the out-of-fold score with weeks, `:155`)
   and sex (AUC, `:229-234`), reported as a positive control.
7. Three eta^2 variants for each of GROUP, TRAINED, SEX, each with its own permutation p (`:193-210`):
   global trace eta^2 (the screen's statistic), the same after z-scoring every dimension, and the maximum
   per-dimension eta^2 with a maximum-statistic null (plus the mean of the top 10 dimensions, no p).

**The 10 splits x 10,000 permutations version versus the original.**

| | Original ("registered") gate | Current gate (`gate_rcv10/`) |
|---|---|---|
| Splits | 1 (`--repeats` default 1, `subspace_probe.py:252`) | 10 (`pipeline/run_stage11.py:77`) |
| Permutations | 1,000 (`subspace_probe.py:250`); PCA control 500 (`augur_prep.py:71`) | 10,000 for both (`pipeline/run_stage11.py:76`, `:50`, `:62`) |
| Statistic | AUC of the one split | mean AUC over the 10 splits; each permutation scored on the same 10 splits |
| Seeds | until 2026-10-05, Python's salted `hash()`; since then crc32 (`subspace_probe.py:72-75`) | crc32; split 0 is the single split (`:123-125`; `tests/test_gate_probe.py:45-46`) |
| Output | `<root>/subspace_probe.tsv` | `<root>/gate_rcv10/subspace_probe.tsv` |

Why it was adopted (`docs/RETRAIN_V2_PREREGISTRATION.md:218-228`, a user decision logged 2026-10-24 "before
any v2 gate number exists"): with one split and 1,000 permutations the seed decided the hotspot set. On
April's embeddings, 10 seed sets gave 7 to 16 hotspots with only 3 common to all and 20 in any; one fold
split moved a block's AUC by about 0.04; and with 1,000 permutations a hotspot could have at most 3 permuted
AUCs above its own. With 10 splits and 10,000 permutations, 5 seed sets gave 16 to 17 hotspots, 15 common
and 18 in any (mean pairwise Jaccard 0.91 against 0.56). Evidence:
`data/results/retrain_v2/gate_stability/hot_stab_12013892.out:2-18` and `hot_rcv_12013975.out:2-13`. Control:
with `--repeats 1` the new code reproduced April's tables byte for byte
(`data/results/retrain_v2/c5/gate_control.txt:1-4`). On the default tree only the current version exists
(there is no `<root>/subspace_probe.tsv`).

**Stage 12's use of the same method.** `pipeline/run_stage12.py` step 3 runs `subspace_probe.py` on the
human-space embeddings with the same defaults (10,000 permutations, 5 folds, 10 splits; `:117-123`,
`:179-182`) into `<human root>/gate_rcv10/` (`:72-80`); the default run's record shows those arguments
(`data/deconvolution/genecompass_input_v2_refs4all_rebuild_human/gate_rcv10/provenance_subspace_probe.json`).
The transfer result itself is for the Stage 12 section.

### Models and software
PLS-1 written directly in numpy (`subspace_probe.py:130-140`); scikit-learn 1.8.0 `StratifiedKFold` and
`roc_auc_score` (`:61-62`); scipy 1.17.1 `spearmanr` (`:60`); joblib 1.5.3 process parallelism (`:59`, `:295-296`).
BLAS and OpenMP threads are pinned to 1 before numpy loads (`:47-48`).

### Parameters, thresholds and flags
| Name | Value | Where set (path:line) | What it controls | Rationale given |
|---|---|---|---|---|
| `--perms` (probe) | 10,000 | `pipeline/run_stage11.py:76` (script default 1,000, `subspace_probe.py:250`) | permutations for every probe and eta^2 test in `subspace_probe.py` | `docs/RETRAIN_V2_PREREGISTRATION.md:218-234` |
| `--repeats` | 10 | `pipeline/run_stage11.py:77` (script default 1, `subspace_probe.py:252`) | number of fold splits | same |
| `--folds` | 5 | literal `'5'` at `pipeline/run_stage11.py:50` (script default `subspace_probe.py:251`) | k; reduced to the smaller class size if that is below 5 (`:115-118`) | none stated |
| `--jobs` | 48 in the run | default 8 at `pipeline/run_stage11.py:78`; 48 from the runbook command (`deconvolution/DECONV_REBUILD_RUNBOOK.md:133`) | worker processes | runtime only |
| Minimum rows per block | 6 | `subspace_probe.py:285`; `pheno_merge_test.py:92` | block inclusion | none stated; hard-coded |
| `WEEK` | control 0, 1w 1, 2w 2, 4w 4, 8w 8 | `subspace_probe.py:69` | ordinal dose | "per the Aim-2 pivot" (`:32`) |
| Block seed | 1000 x (crc32(tissue) mod 9973) + index of the cell type in the tissue's sorted list | `subspace_probe.py:72-75`, `:283`, `:291` | all random draws of the block | stability across interpreters (`:73-74`) |
| Seed offsets | folds +1 (trained), +3 (dose), +5 (sex); permutations +2, +4, +6; eta^2 tests +0; split r adds 7919 r | `subspace_probe.py:179`, `:215-233`, `:126` | independent streams per probe | none stated |
| Top-k dimensions | 10 | `subspace_probe.py:207` | `pdtop10_*` columns | hard-coded |
| Rounding | AUC, rho, eta^2 to 3 decimals; p to 4 | `subspace_probe.py:219-220`, `:238-239` | stored precision | none stated |
| `--perms` (screen) | 1,000 | `pheno_merge_test.py:61` (not passed by `run_stage11.py:56-57`) | permutations of the global screen | none stated |
| `--min-per-group` | 3 | `pheno_merge_test.py:62` | GROUP test needs every group this large | none stated |
| Screen seed | 0, one generator for all blocks in sequence | `pheno_merge_test.py:70` | permutations | none stated |
| Print thresholds | p < 0.05; eta^2 >= 0.10 | `pheno_merge_test.py:109-113`; `subspace_probe.py:307-308` | console summaries only | hard-coded |

None of these is read from `config/pipeline_config.yaml`. The config only names the folders and describes
the method in a comment (`:414-417`).

### Statistical procedures
**Supervised probe, trained versus control (the test behind the hotspots).**
- Statistic: mean over 10 splits of the pooled out-of-fold AUC of the PLS-1 score (above).
- Null: the label vector is permuted across all rows of the block, without restriction (not within sex or
  group), and the whole cross-validation is repeated on the same 10 splits (`subspace_probe.py:169-172`). The
  folds are not redrawn; they stay stratified on the true labels (`:113-114`).
- Number of permutations: 10,000, generator seeded with block seed + 2 (`:218`).
- p-value: (number of permuted statistics >= observed, + 1) / (10,000 + 1) (`:173`). One-sided, upper tail.
  Smallest attainable value 1/10,001; stored to 4 decimals (`:219-220`). A permuted statistic that is not a
  number counts as not exceeding (`:172`).
- Units of replication: animals within a block (one pseudo-cell each).
- Sex: not used in the trained probe: no stratification, no covariate, no within-sex permutation.
- Multiplicity: none inside `subspace_probe.py`; BH is applied in `corroborate_summary.py` (9f).
- Pre-registration: the 10 x 10,000 method and its seeding are in
  `docs/RETRAIN_V2_PREREGISTRATION.md:218-241`; its use for the reference reruns in
  `deconvolution/REFERENCE_RELABEL_PREREG.md:124-126` and `:675-676`.

**Dose probe.** Statistic: mean over splits of the Spearman correlation between the out-of-fold score and
weeks; same permutation scheme, generator seed + 4; one-sided upper tail (`:222-227`, `:155`, `:169-173`). It is
reported but does not enter the hotspot definition.

**Sex probe.** As the trained probe with sex as the label, generator seed + 6 (`:229-234`). Not computed for
single-sex tissues (24 blocks).

**eta^2 tests in `subspace_probe.py`.** Statistic as named; null by unrestricted label permutation, 10,000
times, generator seeded with the block seed and shared by the nine tests of the block in sequence (`:179`,
`:189-191`, `:193-210`); p = (count + 1) / (perms + 1). The per-dimension test compares the observed
maximum with the permutation distribution of the maximum, which accounts for scanning 768 dimensions (`:210`).

**Global screen (`pheno_merge_test.py`).** Statistic: trace eta^2 (`:37-45`). Null: unrestricted label
permutation, 1,000 times (`:48-51`, `:61`); p = (count + 1) / 1,001, stored to 4 decimals (`:101`). One
generator, seed 0, runs through all tissues and blocks in order (`:70`), so a block's permutations depend on
which tissues precede it. No multiplicity correction; the script counts p < 0.05 (`:109-113`). Described as a
permutational multivariate analysis of variance in `manuscript/motrpac_genecompass.tex:611-612`.

### Checks, validation and their results
- Unit tests (`tests/test_gate_probe.py`): eta^2 limits (`:14-20`); per-dimension eta^2 (`:23-27`); stable
  seed (`:30-32`); folds are stratified partitions, reproducible, and split 0 equals the single split
  (`:35-46`); fold count shrinks to the smaller class and singletons give none (`:49-52`); a planted axis is
  detected with the smallest possible p (`:62-68`); noise is not (`:71-76`); a held-out row's score never
  depends on its own label (`:79-87`). No test covers `pheno_merge_test.py`.
- Global screen on the default (`.../pheno_merge_test.tsv`; console `logs/rebuild_xtissue_12122211.out:232-234`):
  p < 0.05 in 61 of 197 blocks for GROUP (median eta^2 0.110), 31 of 197 for TRAINED (0.026), 101 of 173 for
  SEX (0.048). The console line prints the sex count over 197 (`:234`; `pheno_merge_test.py:113` divides by
  the number of rows), but only 173 blocks have a sex test (the 24 OVARY and TESTES blocks have none). The
  manuscript carries the same denominator, "101/197" (`manuscript/motrpac_genecompass.tex:614`).
- Probe on the default (`logs/rebuild_xtissue_12122211.out:248-260`): blocks with p < 0.05, of 197: trained
  34 (global) / 39 (standardised) / 60 (per-dimension) / 41 (supervised); sex 102 / 121 / 133 / 123. Trained
  AUC median 0.597, maximum 0.887; dose Spearman median 0.047, maximum 0.519. Over the 151 analysable blocks
  the median trained AUC is 0.605 (auditor's recomputation from `.../gate_rcv10/subspace_probe.tsv` joined to
  `corroboration_merged.tsv`; matches `deconvolution/ALL_TISSUE_RERUN_RESULTS.md:23`).
- Positive control: the sex probe passes BH q < 0.05 in 106 of 130 analysable two-sex blocks; the global
  trained eta^2 passes in 0 of 151 (auditor's recomputation with the BH formula of 9f on the same table;
  the manuscript states the latter at `manuscript/motrpac_genecompass.tex:628`).
- Reproducibility: every numeric column of the rebuild's gate table equals the earlier tree's
  (`deconvolution/REFS4ALL_REBUILD_RESULTS.md:25`).
- Monte Carlo replicate (auditor's check, no new computation beyond reading two tables): `pca_control.tsv`
  holds a second, independently seeded 10,000-permutation p for the same statistic (`p_embed`; 9g). The
  observed AUCs are equal in all 197 blocks; the two p-values differ by at most 0.017. BH on `p_embed` over
  the analysable blocks gives the same 9 hotspots.

### Deviations, known defects and caveats
1. **Join by position.** The sample-to-animal link relies on the order of `bulk_samples.tsv` matching the
   order of the deconvolved mixture (`pheno_merge_test.py:8-10`, "validated 300/300 on liver"). No current
   script or test re-checks it for the other 15 tissues; this audit confirmed only that every listed vial
   label of LIVER, BLOOD, OVARY and TESTES is in the phenotype table with the expected design (5 per sex per
   group; OVARY has 4 in the 4-week group).
2. **Silent skips.** A tissue without `bulk_samples.tsv`, or whose array and dataset lengths differ, is
   skipped with at most a printed line (`subspace_probe.py:269-274`; `pheno_merge_test.py:77-82`). Because the
   BH family spans all tissues, a skipped tissue would change every q-value without an error.
   `run_stage11.py:95-101` stops only for a tissue with no embeddings.
3. **Sex and time are not modelled.** Folds and permutations ignore sex, and all controls were sacrificed at
   8 weeks while 30 of the 40 trained animals were sacrificed at 1, 2 or 4 weeks. The contrast therefore
   mixes training with time-in-study for those groups. This follows from the study design, but the gate does
   not test 8-week trained against control separately.
4. **Permutation streams are shared between neighbouring blocks.** Adjacent cell types of a tissue have
   block seeds that differ by 1 (`subspace_probe.py:75`, `:283`), and the per-probe offsets are +1 to +6
   (`:215-233`). So, for example, the trained-probe permutation generator of cell type i (seed + 2) is the
   generator of cell type i + 2's eta^2 tests (seed + 0) and of cell type i - 2's dose probe (seed + 4). Each
   p-value is still valid on its own; the Monte Carlo errors of neighbouring blocks are correlated.
5. **Earlier defect, fixed.** Seeds from Python's salted `hash()` made the hotspot set irreproducible; the
   code now uses crc32 (`subspace_probe.py:72-75`) with a test (`tests/test_gate_probe.py:30-32`). On April's
   embeddings the seed change alone moved the count from 15 to 7 (`docs/RETRAIN_V2_PREREGISTRATION.md:146-152`).
6. **Two versions of the same test.** `p_trained` in `pheno_merge_test.tsv` (1,000 permutations, shared
   generator) and `p_glob_trained` in `subspace_probe.tsv` (10,000, block seed) estimate the same quantity
   and give 31 and 34 blocks at p < 0.05. The manuscript quotes the first (`manuscript/motrpac_genecompass.tex:612-615`).
7. **Default output path ignores `--gc-root`.** `--out` defaults to the config's root, not to the root given
   on the command line (`subspace_probe.py:245`, `:256`; `pheno_merge_test.py:56`, `:63`). Calling either
   script with another `--gc-root` and no `--out` would overwrite the default run's table. `run_stage11.py`
   always passes `--out`.
8. `pheno_merge_test.py` writes no provenance record.
9. The stability evidence for choosing 10 x 10,000 was gathered on April's embeddings and the July
   references (172 blocks). It has not been repeated on the default run; item "Monte Carlo replicate" above
   covers permutation noise only, not the fold splits.

### Methods references
- Permutational multivariate analysis of variance (the trace eta^2): Anderson MJ. Austral Ecol 2001;26:32-46
  (`manuscript/motrpac_genecompass.tex:1553-1555`).
- Permutation test of a cross-validated classifier, re-running the whole CV under permuted labels: "Ojala &
  Garriga 2010" as cited in `deconvolution/subspace_probe.py:27`. Not in the manuscript bibliography. Full
  reference from memory: Ojala M, Garriga GC. J Mach Learn Res 2010;11:1833-1863 [verified against the JMLR record; see REFERENCES.md].
- PLS discriminant analysis and its cross-validation pitfalls: "Westerhuis et al. 2008, Metabolomics 4:81"
  as given, marked unverified, in `deconvolution/EMBEDDING_DE_STANDARDS.md:76-77` [verified against Crossref; see REFERENCES.md].
- Maximum-statistic permutation correction: "Westfall-Young max-T family" named without a reference in
  `deconvolution/EMBEDDING_DE_STANDARDS.md:79-80` [book: Westfall PH, Young SS. Resampling-Based Multiple Testing. Wiley, 1993; cited from memory, no registry record checked].

---

## Stage 9e - Augur

### Research role
An independent published method scores the same blocks, to show that the probe's ranking is not an artefact
of the project's own linear method. Claim: Augur's AUC per block follows the probe's. Verdict on the
default: Spearman 0.785 over the 130 analysable blocks that Augur scores (0.799 over all 172 it scores); all
9 hotspots have an Augur AUC of at least 0.73. Documents call 0.80 a registered threshold for this
correlation; no registration of it was found (caveat 1).

### Inputs and outputs
- Inputs, per tissue under `augur_input_dir` (written by `augur_prep.py:116-129`): `embed.tsv` (the 768
  embedding dimensions, one row per pseudo-cell with a known phenotype), `pca.tsv` (the 50 principal
  components of 9g), `meta.tsv` (`rowid, cell_id, cell_type, sample, label, sex, week`).
- Output: `<gate>/augur_results.tsv`, 690 rows: `tissue, representation (embed/pca), condition
  (trained/sex), cell_type, augur_auc`; and `provenance_run_augur.json`.
- Coverage: 172 blocks for `trained` and 173 for `sex` in each representation. OVARY (18 blocks) and TESTES
  (6) have no score for either condition, and LUNG Goblet cells none for `trained` (auditor's tabulation of
  `augur_results.tsv` and `corroboration_merged.tsv`).

### Procedure
`pipeline/run_stage11.py` step 5 runs `deconvolution/run_augur.sh <augur_input_dir> <gate>/augur_results.tsv`
(`:64-66`), a wrapper that loads the R environment and calls `run_augur.R` (`deconvolution/run_augur.sh:6-15`).
For each tissue, each representation (`embed`, `pca`) and each condition (`trained`, `sex`)
(`deconvolution/run_augur.R:61-84`): build a metadata frame of cell type and label (`:70-73`), drop rows with a
missing label, skip if fewer than two label values (`:73-74`), and call

`calculate_auc(X, meta, label_col = "label", cell_type_col = "cell_type", augur_mode = "velocity",
classifier = "rf", n_threads = 8, subsample_size = 10, min_cells = 10, show_progress = FALSE)`
(`run_augur.R:44-48`, values from `:33-37`).

If that call fails, it is retried with `augur_mode = "default", select_var = FALSE` (`:49-55`). X is the
features x cells matrix (`:39-42`).

What Augur 1.0.3 does with these arguments (read from the installed package,
`R_libs/Augur/R/Augur.rdb`, function `calculate_auc`, de-parsed without loading the package; this file is a
binary database, so the line numbers below are lines of the de-parsed function body, not file lines):
1. A cell type is skipped when its smaller class has fewer than `min_cells` = 10 cells (body lines 181-186).
   A full block has exactly 10 controls; OVARY and TESTES have 5, hence no score.
2. "velocity" mode sets `feature_perc = 1` and `var_quantile = 1`: no feature selection (lines 136-140).
   Variance-based selection is skipped in any case below 1,000 features (lines 196-199).
3. For each of `n_subsamples` = 50 subsamples (default, line 2; loop at 246), with `set.seed(subsample_idx)`
   (247): draw `subsample_size` = 10 cells per class without replacement (268-270), drop zero-variance
   features (281), split into `folds` = 3 folds stratified by label (308), fit a random forest on each
   training part and predict the held-out part.
4. Random forest: `randomForest(..., ntree = 100, mtry = 2)`, after `set.seed(1)` (lines 146-162; defaults
   `rf_params = list(trees = 100, mtry = 2, ...)`, lines 5-6).
5. AUC per cell type = mean over subsamples of the mean over folds of the fold's ROC AUC (376-381).

So in a full block each subsample uses all 10 controls and 10 of the 40 trained animals; each forest is
trained on about 13 pseudo-cells and tested on about 7, with 2 of 768 (or 50) features tried at each split.

### Models and software
| Item | Value | Source |
|---|---|---|
| Augur | 1.0.3, GitHub `neurorestore/Augur` at b252b84e | `deconvolution/setup/r_packages.yaml:48-49`; `R_libs/Augur/DESCRIPTION:3`, `:34` |
| R | 4.4.1 | `config/pipeline_config.yaml:429`; `.../gate_rcv10/provenance_run_augur.json:9` |
| randomForest / rsample / yardstick | 4.7-1.2 / 1.3.2 / 1.4.0 | `deconvolution/setup/r_packages.lock.tsv:236`, `:271`, `:351` |

### Parameters, thresholds and flags
| Name | Value | Where set (path:line) | What it controls | Rationale given |
|---|---|---|---|---|
| `classifier` | "rf" | `run_augur.R:47` | random forest | "CANONICAL Augur" (`:2`) |
| `augur_mode` | "velocity" | `run_augur.R:44` | features treated as continuous; no feature selection | `:8-9` |
| `subsample_size` | 10 | `run_augur.R:36` (env `AUGUR_SUBSAMPLE`) | cells per class per subsample (Augur default 20) | controls have only 10 pseudo-cells per cell type (`:34-35`) |
| `min_cells` | 10 | `run_augur.R:37` (env `AUGUR_MINCELLS`) | minimum cells in the smaller class | same |
| `n_subsamples` | 50 | Augur default, not passed | subsamples averaged | not stated |
| `folds` | 3 | Augur default, not passed | CV folds within a subsample | not stated |
| RF trees / mtry | 100 / 2 | Augur defaults, not passed | forest size; features tried per split | not stated |
| `n_threads` | 8 | `run_augur.R:33` (env `AUGUR_THREADS`) | parallelism | runtime |
| Seed | 1 | `run_augur.R:30-31`, from `deconvolution.de_seed` (`config/pipeline_config.yaml:365`) or env `AUGUR_SEED` | session seed; Augur reseeds each subsample and forest itself | `run_augur.R:20-21` |

The run's record confirms seed 1, subsample 10, minimum 10, 8 threads
(`.../gate_rcv10/provenance_run_augur.json:111-114`). Only the seed comes from the config; the rest are
environment-variable defaults in the script.

### Statistical procedures
Augur yields a cross-validated AUC per cell type and no p-value in this mode. Sex is not used when the label
is `trained`. The comparison with the gate is a Spearman rank correlation, across blocks, between
`augur_embed_trained` and `sup_trained_auc` (`deconvolution/corroborate_summary.py:69-78`), plus counts of
hotspots with Augur AUC >= 0.65 and >= 0.70 (`:94-100`). No test is attached to the correlation.

### Checks, validation and their results
| Quantity | Value | Source |
|---|---|---|
| Spearman, Augur vs probe, all blocks with both | 0.799, n = 172 | `logs/rebuild_xtissue_12122211.out:515` |
| Spearman, analysable blocks with both | 0.785, n = 130 | `notebooks/pipeline9_analysis.ipynb:641`; `deconvolution/ALL_TISSUE_RERUN_RESULTS.md:21`; auditor's recomputation agrees |
| Hotspots with Augur AUC >= 0.65 / >= 0.70 | 9 of 9 / 9 of 9; lowest 0.730 (HYPOTH oligodendrocyte precursor cells) | `logs/rebuild_xtissue_12122211.out:528`; `.../gate_rcv10/corroboration_merged.tsv:69` |
| Median Augur AUC, hotspots vs other analysable blocks | 0.817 vs 0.608 | `logs/rebuild_xtissue_12122211.out:527` |
| Sex control, median Augur AUC (embedding / PCA-50) | 0.740 / 0.847 | `logs/rebuild_xtissue_12122211.out:523-524` |
| Augur trained AUC, embedding vs PCA-50, analysable blocks | medians 0.613 vs 0.646 | `notebooks/pipeline9_analysis.ipynb:644-645` |

The manuscript reports "Spearman r = 0.79 (0.785, n = 130 analysable blocks)"
(`manuscript/motrpac_genecompass.tex:660-664`).

### Deviations, known defects and caveats
1. **"Registered threshold 0.80": registration NOT FOUND.** `notebooks/pipeline11_analysis.ipynb:437` ("r >=
   0.80 (pre-registered threshold)"), `notebooks/pipeline9_analysis.ipynb:642` and
   `deconvolution/AIM2_DECONV_RESULTS.md:87` ("just under the registered 0.80") refer to a registered
   threshold. No pre-registration file mentions a threshold for this correlation (searched every
   `*PREREG*.md` and `docs/RETRAIN_V2_PREREGISTRATION.md` for "Augur" and "0.80"; the all-tissue registration
   lists the Augur-PLS Spearman only as something to report, `deconvolution/REFERENCE_RELABEL_PREREG.md:680-681`).
   If the threshold exists, the observed 0.785 is below it, and `AIM2_DECONV_RESULTS.md:87` still marks the row
   with a tick.
2. **Two values of the same correlation.** The committed gate script prints 0.799 over all 172 scored blocks,
   including non-analysable ones (`corroborate_summary.py:76-78`). The reported 0.785 over analysable blocks is
   computed only in notebooks (`notebooks/pipeline9_analysis.ipynb:677-681`; `pipeline11_analysis.ipynb:732-736`),
   whose comment says "as deconvolution/corroborate_summary.py computes them" (`pipeline9_analysis.ipynb:677`);
   the script does not restrict to analysable blocks.
3. **Stale figure.** The summary table of `deconvolution/EMBEDDING_DE_STANDARDS.md:18` still says Augur
   "reproduces our probe at r=0.83" (June); its closing section gives the current values (`:109-115`).
4. **Not a like-for-like classifier.** Augur trains on 13 or so pseudo-cells with `mtry = 2` of 768 features;
   the probe trains on 40. Agreement in ranking is the claim; the AUC levels are not comparable.
5. **Coverage.** 21 analysable blocks (15 OVARY, 6 TESTES) cannot be corroborated by Augur at all.
6. Human-space Augur is not run (`docs/RETRAIN_V2_PREREGISTRATION.md:243-245`).
7. The fallback call (`run_augur.R:49-55`) switches to Augur's default mode on any error without recording
   which mode produced a score. In the default run the only errors are the four expected ones, "no cell type
   had at least 10 cells in all conditions", for OVARY and TESTES on `trained` in both representations
   (`logs/rebuild_xtissue_12122211.out:460`, `:462`, `:496`, `:498`); the fallback failed the same way, so no
   score in the table comes from the fallback mode.

### Methods references
- Augur: Skinnider MA, Squair JW, Kathe C, et al. Cell type prioritization in single-cell data. Nat
  Biotechnol 2021;39:30-34; protocol: Squair JW, Skinnider MA, Gautier M, Foster LJ, Courtine G. Nat Protoc
  2021;16:3836-3873 (`manuscript/motrpac_genecompass.tex:1499-1503`).
- Random forest: no citation in the code or manuscript. Breiman L. Mach Learn 2001;45:5-32 [verified against Crossref; see REFERENCES.md].

---

## Stage 9f - Corroboration table and hotspot definition

### Research role
Merges the probe, the PCA control and Augur into one table per block, applies the analysable-block rule and
the multiplicity correction, and so defines the hotspots that Stages 10, 12 and 13 read
(`config/pipeline_config.yaml:414-417`). Verdict: 9 hotspots among 151 analysable blocks of 197 (16 tissues);
11 among 124 of 165 on the 14-tissue arm.

### Inputs and outputs
- Inputs, all in the gate folder: `subspace_probe.tsv`, `pca_control.tsv`, `augur_results.tsv`
  (`deconvolution/corroborate_summary.py:39-41`); and each block's Z matrix
  `<results_dir>/motrpac/<TISSUE>/pred_z/predz__<cell type file name>.csv` (`:37`, `:54-57`), the file name
  built by the shared `safe()` function (`deconvolution/celltype_names.py:35-37`; imported at
  `corroborate_summary.py:24`).
- Outputs: `<gate>/corroboration_merged.tsv` (197 rows) and, as registered sensitivities,
  `corroboration_merged_minlib100.tsv` and `corroboration_merged_minlib0.tsv`
  (`pipeline/run_stage11.py:51-52`, `:67-69`); `provenance_corroborate_summary.json`.

Columns of `corroboration_merged.tsv` (header at line 1 of the file):

| Column | Meaning | Source in code |
|---|---|---|
| `tissue`, `cell_type`, `n` | block key and number of pseudo-cells | `corroborate_summary.py:48` |
| `sup_trained_auc`, `p_sup_trained` | probe AUC and permutation p (9d) | `:48` |
| `glob_trained`, `p_glob_trained` | global trace eta^2 and its p (9d) | `:48` |
| `auc_embed`, `auc_pca`, `auc_genes`, `p_embed` | PLS-1 AUC on the embedding, PCA-50 and all genes; a second permutation p for the embedding (9g) | `:49` |
| `augur_embed_sex`, `augur_embed_trained`, `augur_pca_sex`, `augur_pca_trained` | Augur AUCs (9e) | `:43-46` |
| `median_libsize` | median over samples of the block's summed Z (negatives and missing as 0) | `:54-60` |
| `analysable` | `median_libsize` >= threshold | `:61` |
| `q_sup_trained` | BH-adjusted `p_sup_trained`, analysable blocks only; empty otherwise | `:64-67` |

### Procedure
`pipeline/run_stage11.py` step 6 runs `corroborate_summary.py --gc-root <gate>` and then the same with
`--min-libsize 100` and `--min-libsize 0` (`:51-52`, `:67-69`). The script:
1. Pivots Augur to one row per block (`corroborate_summary.py:43-46`) and outer-joins the three tables on
   tissue and cell type (`:48-51`).
2. Computes `median_libsize` from the Z file (`:54-60`) and the `analysable` flag (`:61`).
3. Applies BH to `p_sup_trained` over the analysable blocks (`:64-67`).
4. Prints the summaries used in 9e and 9g (`:76-106`) and writes the table and a provenance record (`:108-111`).

**Definition of a hotspot.** A block with `q_sup_trained < 0.05` in `corroboration_merged.tsv`. The table
stores q; each reader applies the cut itself: `deconvolution/R/run_pseudobulk_de.R:112-113` (q < 0.05,
literal), `translation/compare_transfer.py:96-97` and `:128` (`--hotspot-q`, default 0.05),
`corroborate_summary.py:95` (literal). There is no config key for the threshold.

### Models and software
pandas 2.3.3, numpy 2.4.3, scipy 1.17.1 (`requirements.lock.txt:155`, `:134`, `:198`).

### Parameters, thresholds and flags
| Name | Value | Where set (path:line) | What it controls | Rationale given |
|---|---|---|---|---|
| `--min-libsize` | 1,000 | `corroborate_summary.py:31-32` (argparse default) | analysable-block rule | `deconvolution/REFERENCE_RELABEL_PREREG.md:104-112`: the threshold "sits in the gap between the two modes of the distribution" (65 blocks under 1,000, 4 between 1,000 and 10,000, 103 above, on the July references) |
| Sensitivities | 100 and 0 | `pipeline/run_stage11.py:79-80` | extra merges | registered, `REFERENCE_RELABEL_PREREG.md:112` |
| Hotspot cut | q < 0.05 | literals listed above | hotspot set | conventional; none stated |
| Augur agreement marks | AUC >= 0.65 and >= 0.70 | `corroborate_summary.py:99-100` | printed counts only | none stated |
| Rows printed | top 22 | `corroborate_summary.py:105-106` | console | hard-coded |

The same 1,000 threshold is set a second time, independently, for the differential-expression stage
(`deconvolution/R/run_pseudobulk_de.R:92`, env `DE_MIN_MEDIAN_LIBSIZE`). The two defaults must be kept equal
by hand.

### Statistical procedures
- **Analysable-block rule** (`REFERENCE_RELABEL_PREREG.md:104-112`, registered 2026-10-27): a block is analysed
  when the median over its samples of the summed Z is at least 1,000. A block under it gets no q and cannot
  be a hotspot (`corroborate_summary.py:10-13`, `:61-67`). The median is taken over all samples in the Z file,
  including any whose pseudo-cell was dropped by the tokenizer.
- **Multiplicity.** BH step-up adjustment computed in the script: sort the p-values, q_(i) = p_(i) m / i,
  take the running minimum from the largest rank down, clip to [0, 1] (`corroborate_summary.py:65-67`). The
  family is every analysable block of every tissue in the run: m = 151 on the default. Hotspot: q < 0.05.
  The p-values entering are the 4-decimal values of 9d.
- **Resolution.** With 10,000 permutations the smallest p is 1/10,001, so the smallest possible q is about
  151 / 10,001 = 0.015. The two top blocks tie at p = 0.0002 and both have q = 0.0151.
- **Pre-registration.** BH over all blocks: `docs/RETRAIN_V2_PREREGISTRATION.md:234`. Restriction to
  analysable blocks: `REFERENCE_RELABEL_PREREG.md:111-112`. One family over all 16 tissues, with the
  consequence for the 14-tissue hotspots to be reported: `REFERENCE_RELABEL_PREREG.md:577-581`.

### Checks, validation and their results
Default run, 16 tissues (`data/deconvolution/genecompass_input_v2_refs4all_rebuild/gate_rcv10/corroboration_merged.tsv`;
file line of each row in the first column; counts confirmed in `logs/rebuild_xtissue_12122211.out:513`, `:527`):

| Line | Tissue | Cell type | n | AUC | p | q | Augur AUC | PCA-50 AUC | All-genes AUC | Median library |
|---|---|---|---|---|---|---|---|---|---|---|
| 148 | SKMVL | Satellite cells | 50 | 0.887 | 0.0002 | 0.0151 | 0.885 | 0.769 | 0.874 | 20,433 |
| 152 | SKMVL | Vascular smooth muscle cells | 50 | 0.874 | 0.0002 | 0.0151 | 0.816 | 0.759 | 0.761 | 19,089 |
| 14 | BLOOD | Megakaryocytes and platelets | 50 | 0.837 | 0.0009 | 0.0302 | 0.858 | 0.810 | 0.790 | 1,280,744 |
| 13 | BLOOD | Erythroid cells | 50 | 0.828 | 0.0012 | 0.0302 | 0.836 | 0.799 | 0.813 | 6,085,146 |
| 142 | SKMGN | Tenocytes | 50 | 0.825 | 0.0006 | 0.0302 | 0.763 | 0.679 | 0.876 | 117,372 |
| 11 | BLOOD | Classical monocytes | 50 | 0.821 | 0.0010 | 0.0302 | 0.841 | 0.767 | 0.787 | 323,858 |
| 16 | BLOOD | Neutrophils | 50 | 0.819 | 0.0014 | 0.0302 | 0.817 | 0.797 | 0.763 | 601,420 |
| 116 | LUNG | Vascular smooth muscle cells | 50 | 0.817 | 0.0018 | 0.0340 | 0.777 | 0.686 | 0.532 | 1,419,096 |
| 69 | HYPOTH | Oligodendrocyte precursor cells | 50 | 0.798 | 0.0022 | 0.0369 | 0.730 | 0.443 | 0.499 | 2,266 |

- **9 hotspots of 151 analysable blocks (197 total).** Analysable per tissue: BAT 6 of 6, BLOOD 8 of 13,
  CORTEX 10 of 12, HEART 12 of 16, HIPPOC 7 of 11, HYPOTH 8 of 13, KIDNEY 11 of 15, LIVER 8 of 8, LUNG 7 of
  21, OVARY 15 of 18, SKMGN 8 of 9, SKMVL 8 of 9, SMLINT 13 of 13, SPLEEN 12 of 14, TESTES 6 of 6, WATSC 12 of
  13 (auditor's tabulation of the `analysable` column).
- **Just outside.** LUNG endothelial cells (AUC 0.786, p 0.0037, q 0.0535; file line 104) and LUNG
  mesothelial cells (0.785, 0.0039, 0.0535; line 108).
- **Sensitivities.** At a threshold of 100: 158 analysable, the same 9. With no threshold: 197, the same 9
  (`logs/rebuild_xtissue_12122211.out:557`, `:571`, `:601`, `:615`; set equality from the two files).
- **14-tissue arm** (`data/deconvolution/genecompass_input_v2_refs4/gate_rcv10/corroboration_merged.tsv`;
  profile `config/profiles/refs4_14.yaml`): **11 hotspots of 124 analysable blocks (165 total)**: the 9 above
  plus LUNG endothelial and LUNG mesothelial cells (q 0.0440 each; `deconvolution/ALL_TISSUE_RERUN_RESULTS.md:33-39`).
  Their AUC and p are unchanged on 16 tissues; only the family grew.
- **July references, same model and gate:** 13 hotspots of 172 blocks, no library rule
  (`data/deconvolution/genecompass_input_v2/gate_rcv10/corroboration_merged.tsv`; `docs/RETRAIN_V2_RESULTS.md:196`).
- Reproduction: 197 / 151 / the same 9 with zero difference in every numeric column
  (`deconvolution/REFS4ALL_REBUILD_RESULTS.md:25`).
- No unit test covers `corroborate_summary.py` (the BH code or the rule).

### Deviations, known defects and caveats
1. **The provenance record of the hotspot table is overwritten.** All three merges write the same file name,
   `provenance_corroborate_summary.json` (`corroborate_summary.py:109-111`; `lib/provenance.py:109`). What is
   on disk describes the last one, the no-threshold sensitivity
   (`.../gate_rcv10/provenance_corroborate_summary.json:41-42`: `min_libsize 0.0`, output
   `corroboration_merged_minlib0.tsv`). The primary table has no record of its own. The same was noted for
   the 14-tissue run (`deconvolution/REFS4_RERUN_RESULTS.md:174-176`) and the code was not changed.
2. **The hotspot boundary is within Monte Carlo error of the permutation test.** With 151 tests, the tenth
   block needs p <= 0.05 x 10 / 151 = 0.0033 to pass. LUNG endothelial and mesothelial cells have p = 0.0037
   and 0.0039. The standard error of a p near 0.003 estimated from 10,000 permutations is about
   sqrt(0.003 x 0.997 / 10,000) = 0.0005 (auditor's arithmetic). The independent replicate in `pca_control.tsv`
   gives 0.0035 and 0.0048 for the same two blocks. Their exclusion, and the "9 versus 11" difference between
   arms, therefore rests on differences of about one standard error plus the growth of the family.
3. **Blocks are not independent tests.** All blocks of a tissue come from the same animals and the same bulk
   samples; four of the nine hotspots are BLOOD blocks. BH is applied as if the family were independent or
   positively dependent; this is not examined anywhere. The shared signal across blocks of a tissue is
   acknowledged for the differential-expression results (`deconvolution/ALL_TISSUE_RERUN_RESULTS.md:48-51`).
4. **Two hotspots have no signal in gene space.** For HYPOTH oligodendrocyte precursor cells the same probe
   gives AUC 0.443 on PCA-50 and 0.499 on all genes (embedding 0.798), with a median library of 2,266, close
   to the 1,000 cut; for LUNG vascular smooth muscle cells the all-genes AUC is 0.532 (table above). A signal
   that the model's embedding shows and the expression it was computed from does not deserves a comment in
   any text that names these two blocks.
5. **Family definition changed between registrations.** `docs/RETRAIN_V2_PREREGISTRATION.md:234` and
   `REFERENCE_RELABEL_PREREG.md:125-126` say "over all blocks"; `REFERENCE_RELABEL_PREREG.md:111-112` says the
   rule is applied "before the BH correction". The code follows the latter (`corroborate_summary.py:64-65`).
   The no-threshold sensitivity is the "all blocks" version and gives the same 9.
6. **Other uses of the word "hotspot" in the code** differ from the definition above: `augur_prep.py:175-176`
   prints "hotspots (embed held-out AUC >= 0.70)" (49 blocks, `logs/rebuild_xtissue_12122211.out:343`); the
   viewer labels a block "Exercise hotspot" when the screen's p < 0.05 and eta^2 >= 0.05 (9h); a stale comment
   in `deconvolution/R/run_pseudobulk_de.R:108` speaks of "the canonical 22-block gate".
7. `corroborate_summary.py:34` uses `parse_known_args`, so a mistyped option is ignored without error.
8. Non-analysable blocks are still embedded, probed and written with their p; only q is withheld. Five of the
   46 have p < 0.05 (for example LUNG non-classical monocytes, p 0.0032, median library 6.6).

### Methods references
- BH: Benjamini Y, Hochberg Y. J R Stat Soc Series B 1995;57:289-300
  (`manuscript/motrpac_genecompass.tex:1545-1547`).

---

## Stage 9g - The embedding-versus-PCA comparison

### Research role
Asks whether the foundation-model embedding separates trained from control better than a plain
dimensionality reduction of the same pseudo-cells, or than the genes themselves. Verdict on the default: no.
Over the 151 analysable blocks the embedding's AUC is higher than PCA-50's in 82, median difference +0.009,
Wilcoxon p = 0.15; over all 197 blocks, 103, +0.007, p = 0.47.

### Inputs and outputs
- Inputs: per tissue, `pseudocells.h5ad`, the embeddings, the dataset, the phenotype join of 9d
  (`deconvolution/augur_prep.py:84-108`).
- Outputs: `<gate>/pca_control.tsv` (197 rows: `tissue, cell_type, n, auc_embed, p_embed, auc_pca, p_pca,
  auc_genes, p_genes`); the Augur exports of 9e. No provenance record is written by `augur_prep.py`; the
  arguments of the default run are known from the orchestrator and its log
  (`logs/rebuild_xtissue_12122211.out:313`: `--pca 50 --perms 10000 --folds 5 --repeats 10 --jobs 48`).

### Procedure
`pipeline/run_stage11.py` step 4 (`:61-63`) runs `augur_prep.py`. Per tissue:
1. Reorder the h5ad rows to the embedding's row order by `cell_id` (`augur_prep.py:92-98`), so the 18 dropped
   LUNG pseudo-cells are absent from every representation.
2. Log-normalise the raw Z: each pseudo-cell scaled to a total of 10,000 over all its genes, then
   log(1 + x) (`lognorm`, `:51-54`, `:110`).
3. Standardise each gene across **all pseudo-cells of the tissue** (all cell types together) and clip to
   +/-10 (`zscore_clip`, `:57-59`, `:111`). This matrix is the "genes" representation.
4. PCA with min(50, cells - 1, genes) components, `random_state = 0`, fitted on the same tissue-wide matrix
   (`:113-114`). This is "PCA-50". The components are therefore those of the whole tissue, dominated by
   differences between cell types, and are fitted on all samples including those later held out.
5. For each block with at least 6 rows and both labels (`:133-139`), run the PLS-1 probe of 9d on three
   representations: `embed`, `pca`, `genes`, with the same folds (block seed + 1, the same seed as
   `subspace_probe.py`) and a separate permutation generator per representation (block seed + 7, 11, 13;
   `:48`, `:144-156`).

### Models and software
scikit-learn 1.8.0 `PCA` (`augur_prep.py:39`); the probe functions imported from `subspace_probe.py` (`:42`);
scipy 1.17.1 `wilcoxon` (`:35`; also `corroborate_summary.py:18`).

### Parameters, thresholds and flags
| Name | Value | Where set (path:line) | What it controls | Rationale given |
|---|---|---|---|---|
| `--pca` | 50 | literal `'50'` at `pipeline/run_stage11.py:62`; script default `augur_prep.py:70` | number of components | none stated |
| PCA `random_state` | 0 | `augur_prep.py:114` | solver seed | none stated; hard-coded |
| Normalisation target | 10,000 | `augur_prep.py:51` | library size before log | none stated; hard-coded (differs from the tokenizer's 6,400) |
| Clip | +/-10 | `augur_prep.py:57` | bound on z-scores | "standard scaled (Seurat/scanpy)" (`:111`); hard-coded |
| `--perms`, `--folds`, `--repeats` | 10,000, 5, 10 | `pipeline/run_stage11.py:50`, `:62` (script defaults 500, 5, 1: `augur_prep.py:71-74`) | as in 9d | as in 9d |
| `SPACE_SEED` | embed 7, pca 11, genes 13 | `augur_prep.py:48` | permutation generator offsets | none stated |
| Minimum rows | 6 | `augur_prep.py:135` | block inclusion | hard-coded |

### Statistical procedures
- Per block and representation: the trained-versus-control test of 9d (same statistic, null, one-sided
  p = (count + 1) / 10,001).
- Across blocks: a Wilcoxon signed-rank test on the paired AUCs (`auc_embed` vs `auc_pca`), called with scipy's
  default arguments, so two-sided (`augur_prep.py:164-173`; `corroborate_summary.py:81-86`). The unit is the
  block. No correction for the number of such comparisons (embedding vs PCA, embedding vs genes, PCA vs
  genes, and the Augur version).
- The comparison is described as "context, not registered" for the model comparison
  (`docs/RETRAIN_V2_RESULTS.md:241-245`); the all-tissue registration lists "embedding vs PCA" among the things
  to report, without a criterion (`deconvolution/REFERENCE_RELABEL_PREREG.md:680-681`).

### Checks, validation and their results
| Comparison | Blocks | Medians | Median difference | Higher in | Wilcoxon p | Source |
|---|---|---|---|---|---|---|
| Embedding vs PCA-50, all blocks | 197 | 0.597 vs 0.610 | +0.007 | 103 | 0.47 | `logs/rebuild_xtissue_12122211.out:339`, `:519` |
| Embedding vs PCA-50, analysable | 151 | 0.605 vs 0.622 | +0.009 | 82 | **0.15** (0.147) | `notebooks/pipeline9_analysis.ipynb:643-644`; `notebooks/pipeline11_analysis.ipynb:715-716`; auditor's recomputation from `corroboration_merged.tsv` agrees |
| Embedding vs all genes, all blocks | 197 | 0.597 vs 0.589 | +0.007 | 104 | 0.79 | `logs/rebuild_xtissue_12122211.out:340` |
| Embedding vs all genes, analysable | 151 | 0.605 vs 0.599 | +0.008 | 84 | 0.33 | auditor's recomputation; quoted in `manuscript/motrpac_genecompass.tex:667-668` |
| Augur, embedding vs PCA-50, all scored blocks | 172 | 0.593 vs 0.619 | +0.002 | 87 | 0.45 | `logs/rebuild_xtissue_12122211.out:520` |
| 14-tissue arm, embedding vs PCA-50, analysable | 124 | | +0.019 | 71 | 0.013 | `deconvolution/ALL_TISSUE_RERUN_RESULTS.md:22`; auditor's recomputation from the refs4 table agrees |

The PCA-50 and all-genes probes do not use the model; for the model comparison they gave identical values
for April and v2 in all 172 blocks (`docs/RETRAIN_V2_RESULTS.md:191-192`).

### Deviations, known defects and caveats
1. **The headline p = 0.15 is not produced by the gate code.** `augur_prep.py:164-173` and
   `corroborate_summary.py:81-86` test all 197 blocks (p = 0.47). The analysable-only figure (82 of 151,
   p = 0.15) is computed in notebooks 9 and 11 (`notebooks/pipeline9_analysis.ipynb:677-683`;
   `pipeline11_analysis.ipynb:732-737`). Both lead to the same reading (no significant advantage), and the
   manuscript gives both (`manuscript/motrpac_genecompass.tex:664-668`).
2. **What "PCA-50" is.** A tissue-wide PCA across all cell types (step 4), not a PCA within the block. Within
   a block of 50 animals its 50 components are mostly directions that separate cell types. A within-block PCA
   could behave differently; none is computed. `deconvolution/AIM2_DECONV_RESULTS.md:193` describes PCA-50 as
   keeping "the top-variance directions, exactly where exercise isn't".
3. **Paired blocks are dependent** (same animals within a tissue), so the Wilcoxon p is approximate.
4. **No provenance** for `pca_control.tsv` or the Augur exports (`augur_prep.py` has no `write_provenance`
   call).
5. The significant result on 14 tissues (p = 0.013) and its loss on 16 (p = 0.15) are both reported
   (`deconvolution/ALL_TISSUE_RERUN_RESULTS.md:71-73`). Statements that the embedding "beats PCA-50 modestly"
   (`deconvolution/EMBEDDING_DE_STANDARDS.md:21`, `:98-99`; `deconvolution/AIM2_DECONV_RESULTS.md:193`) date
   from June and do not hold on the default; the same files state the current result further down
   (`EMBEDDING_DE_STANDARDS.md:100-103`).

### Methods references
- "One PCA still rules them all": arXiv 2410.13956, as cited in `deconvolution/augur_prep.py:18-19` and
  `deconvolution/EMBEDDING_DE_STANDARDS.md:96-97`. Not in the manuscript bibliography.
- Wilcoxon signed-rank test: no citation in code or manuscript. Wilcoxon F. Biometrics Bulletin 1945;1:80-83
  [verified against Crossref; see REFERENCES.md].

---

## Stage 9h - UMAP viewer

### Research role
A visual aid: a two-dimensional map of all pseudo-cell embeddings, coloured by cell type, tissue, sex,
group and weeks, with the global screen's statistics beside it. No claim or test rests on it.
Verdict: **it does not exist for the default configuration.**

### Inputs and outputs
- Inputs: all embeddings and datasets under the root, the phenotype join of 9d
  (`deconvolution/umap_embeddings.py:37-61`); for the viewer, `umap/umap_coords.tsv` and
  `pheno_merge_test.tsv` (`deconvolution/build_umap_viewer.py:130-132`) and the template
  `deconvolution/umap_viewer_template.html` (`:133`).
- Outputs under `<root>/umap/`: `umap_coords.tsv`, `umap_panels.png`, `umap_interactive.html`
  (`umap_embeddings.py:163-166`); `viewer.html`, `viewer_data.json` (`build_umap_viewer.py:182-191`).
- On disk these exist only for April's tree (`data/deconvolution/genecompass_input/umap/`, 2026-07-17) and
  for v2 on the July references (`data/deconvolution/genecompass_input_v2/umap/`, 2026-10-24: 8,450
  pseudo-cells, 14 tissues, 110 cell types; `docs/RETRAIN_V2_RESULTS.md:597-601`). There is no `umap/` folder
  in `genecompass_input_v2_refs4`, `_refs4all` or `_refs4all_rebuild` (directory listings, 2026-11-11).

### Procedure
Not part of any orchestrator (`pipeline/run_stage11.py:13-20` lists six steps, none of them UMAP); run by a
local job script (`slurm/analysis/run_umap_viz.slurm:17-25`).
1. `umap_embeddings.py`: stack all tissues' embeddings (`:37-61`); UMAP with 30 neighbours, minimum distance
   0.3, cosine metric, `random_state = 0` (`:152-153`, `:159-161`); write coordinates, a static figure and a
   plotly page.
2. `build_umap_viewer.py`: read the coordinates and the screen table (`:135-164`); compute two extra
   projections of the L2-normalised embeddings (`:75-76`): (a) a pooled within-cell-type PCA (remove each
   block's centroid, scale each block, standardise, one singular value decomposition; `:84-95`) and (b)
   leave-one-out design axes, x = projection on the ordinal-week regression direction and y = projection on
   the male-minus-female centroid direction, each fitted without the cell being scored (`:97-118`); inline
   everything into the template (`:178-191`).

### Models and software
umap-learn 0.5.11, pynndescent 0.6.0, numba 0.64.0, plotly 6.8.0, matplotlib 3.10.8
(`requirements.lock.txt:231`, `:177`, `:132`, `:161`, `:115`). umap-learn is in the lock file but not in
`requirements.txt` (only plotly is listed for this script, `requirements.txt:77`).

### Parameters, thresholds and flags
| Name | Value | Where set (path:line) | What it controls | Rationale given |
|---|---|---|---|---|
| `--n-neighbors` | 30 | `umap_embeddings.py:152` | UMAP neighbourhood | none stated |
| `--min-dist` | 0.3 | `umap_embeddings.py:153` | UMAP spread | none stated |
| Metric | cosine | `umap_embeddings.py:161` | distance | hard-coded |
| `random_state` | 0 | `umap_embeddings.py:161` | layout seed | hard-coded |
| Norm epsilon | 1e-9 | `build_umap_viewer.py:76` | numerical guard | hard-coded |
| Viewer: significance | p < 0.05 | `deconvolution/umap_viewer_template.html:410`, `:415-417`, `:490` | bold type and counts | hard-coded |
| Viewer: "Exercise hotspot" | screen p_trained < 0.05 and eta^2_trained >= 0.05 | `umap_viewer_template.html:508` | verdict text | hard-coded |
| Viewer: "Sex-dominated" | p_sex < 0.05 and eta^2_sex >= 0.15 | `umap_viewer_template.html:509` | verdict text | hard-coded |
| Viewer: bar scale | eta^2 / 0.45 | `umap_viewer_template.html:410`, `:490` | display | hard-coded |

### Statistical procedures
None of its own. It displays the eta^2 and p of the global screen (9d), described in the template as "a
1000-permutation p" (`umap_viewer_template.html:215`).

### Checks, validation and their results
The viewer checks that its embedding rows align with the coordinate rows and skips the projections otherwise
(`build_umap_viewer.py:61-72`). No test covers either script.

### Deviations, known defects and caveats
1. **No viewer for the default run**, or for either refs4 arm (above). The newest one shows v2 on the July
   references, whose cell types and hotspots differ from the current ones.
   `deconvolution/AIM2_DECONV_RESULTS.md:85` lists the viewer as done without saying which run it shows.
2. **The viewer's "Exercise hotspot" is not the gate's hotspot.** It uses the uncorrected screen p and an
   eta^2 floor (`umap_viewer_template.html:508`); the gate uses BH q < 0.05 on the probe (9f). Applied to the
   default run's `pheno_merge_test.tsv`, the viewer's rule would label 19 blocks, of which 7 are gate
   hotspots, 12 are not (4 of those are not even analysable), and it would miss 2 gate hotspots (LUNG
   vascular smooth muscle cells, SKMGN tenocytes) (auditor's tabulation of the two tables).
3. The template's fallback message still mentions "heart awaiting the refreshed gate"
   (`umap_viewer_template.html:504`).
4. `build_umap_viewer.py` has no command-line options; it always reads and writes the config's root
   (`:127-133`). `umap_embeddings.py --out` defaults to the config's root even when `--gc-root` is given
   (`:146`, `:151`).
5. Neither script writes provenance. UMAP with a fixed `random_state` is repeatable on one machine; the
   layout is not guaranteed across library versions (no statement in the repo; auditor's note).
6. Between-tissue positions in the map are not comparable biology, since each tissue has its own reference
   (`deconvolution/AIM2_DECONV_RESULTS.md:101`).

### Methods references
- UMAP: no citation in the code or the manuscript bibliography. McInnes L, Healy J, Melville J. arXiv
  1802.03426 [verified against arXiv 1802.03426; see REFERENCES.md].

---

## Consolidated list: values that are hard-coded or script defaults rather than read from the config

`config/pipeline_config.yaml` supplies only folders, the checkpoint, the target sum and the seed used by
Augur (`:364-365`, `:406-407`, `:413`, `:417`, `:419`). Everything statistical is a literal or an argparse
default:

| Value | Where | Used for |
|---|---|---|
| top-N 2,048 | `pipeline/05_tokenization/tokenize_corpus.py:122` | sequence length (the config's `top_n_genes`, `:294`, is not read here) |
| minimum genes 200 | `deconvolution/tokenize_pseudocells.py:100` | pseudo-cell filter |
| fallback target 6,500 | `pipeline/run_stage9.py:204`; `tokenize_pseudocells.py:116`; `pipeline/run_stage12.py:216` | if the config key is missing |
| corpus reference 0.869 / 0.960 | `tokenize_pseudocells.py:182` | printed comparison (April values) |
| batch 32; 3 species; alpha 1.0; two file paths | `finetune/genecompass/embed_cells.py:193`, `:125`, `:165`, `:96`, `:113` | embedding |
| cap 1,000,000 | `pipeline/run_stage9.py:58`; `pipeline/run_stage12.py:63` | cell cap fallback |
| k = 10, cosine | `deconvolution/embed_qc.py:38-40` | embedding QC |
| 1,000 permutations; seed 0; minimum 3 per group; minimum 6 rows | `deconvolution/pheno_merge_test.py:61`, `:70`, `:62`, `:92` | global screen |
| 10,000 permutations; 10 splits; 5 folds; sensitivities 100 and 0; PCA 50 | `pipeline/run_stage11.py:76`, `:77`, `:50`, `:79`, `:62` | gate |
| 10,000 permutations; 10 splits; 5 folds | `pipeline/run_stage12.py:179`, `:181`, `:121` | human-space probe |
| seed constants 1000, 9973, 7919; offsets +1 to +6; minimum 6 rows; weeks map; top 10 | `deconvolution/subspace_probe.py:75`, `:126`, `:215-233`, `:285`, `:69`, `:207` | probe |
| library target 10,000; clip 10; PCA seed 0; offsets 7 / 11 / 13; AUC >= 0.70 print | `deconvolution/augur_prep.py:51`, `:57`, `:114`, `:48`, `:175` | PCA control |
| subsample 10; minimum cells 10; threads 8; RF; velocity mode (and Augur's own 50 subsamples, 3 folds, 100 trees, mtry 2) | `deconvolution/run_augur.R:36`, `:37`, `:33`, `:47`, `:44` | Augur |
| library threshold 1,000; q < 0.05; AUC 0.65 / 0.70 | `deconvolution/corroborate_summary.py:31`, `:95`, `:99-100` | hotspots |
| q < 0.05 in readers | `deconvolution/R/run_pseudobulk_de.R:113`; `translation/compare_transfer.py:96` | hotspot use downstream |
| library threshold 1,000 (second copy) | `deconvolution/R/run_pseudobulk_de.R:92` | Stage 10 |
| UMAP 30 / 0.3 / cosine / seed 0 | `deconvolution/umap_embeddings.py:152-153`, `:161` | UMAP |
| viewer cuts 0.05, 0.05, 0.15, 0.45 | `deconvolution/umap_viewer_template.html:508-509`, `:410` | viewer text |

## Summary of audit findings for this section

1. **Hotspot provenance overwritten** (`deconvolution/corroborate_summary.py:109-111`): the on-disk record
   describes the no-threshold sensitivity, not the table that defines the hotspots (9f, caveat 1).
2. **April profile loads the v2 checkpoint** (`config/profiles/april.yaml:21` with
   `deconvolution/model_checkpoint.py:33-38` and `config/pipeline_config.yaml:364`) (9b, caveat 1). Default run
   unaffected.
3. **Reported corroboration numbers come from notebooks, not the gate script**: embedding vs PCA p = 0.15 and
   Augur-probe Spearman 0.785 are analysable-block figures computed in `notebooks/pipeline9_analysis.ipynb:677-683`;
   `deconvolution/corroborate_summary.py:76-86` prints 0.47 and 0.799 over all blocks (9e caveat 2, 9g caveat 1).
4. **"Registered threshold 0.80" for the Augur correlation has no registration** that could be found, and the
   observed value is below it (`notebooks/pipeline11_analysis.ipynb:437`; `deconvolution/AIM2_DECONV_RESULTS.md:87`)
   (9e, caveat 1).
5. **The hotspot cut sits within Monte Carlo error**: LUNG endothelial and mesothelial cells miss by p 0.0037
   and 0.0039 against a cut of 0.0033 (`.../gate_rcv10/corroboration_merged.tsv:104`, `:108`) (9f, caveat 2).
6. **Silent partial weight loading** (`finetune/genecompass/embed_cells.py:129`) (9b, caveat 2).
7. **Target sum not recalibrated for the current references**: LIVER value median 0.878 against a corpus
   median of 0.9145, while the config comment says they match (`config/pipeline_config.yaml:411-413`) (9a,
   caveat 1).
8. **No UMAP or viewer for the default run**, and the viewer's "hotspot" uses a different rule
   (`deconvolution/umap_viewer_template.html:508`) (9h).
9. **Stage numbering and stale statements**: `pipeline/README.md:25` denies Stage 11; `pipeline/run_stage9.py:27-29`,
   `deconvolution/tokenize_pseudocells.py:13-18`, `:181-182`, `config/pipeline_config.yaml:408-409`,
   `deconvolution/AIM2_DECONV_RESULTS.md:19`, `:103`, `deconvolution/EMBEDDING_DE_STANDARDS.md:18` are out of date.
10. **Two hotspots (HYPOTH oligodendrocyte precursor cells, LUNG vascular smooth muscle cells) show no signal
    in the gene-space probe** of the same pseudo-cells (`.../gate_rcv10/corroboration_merged.tsv:69`, `:116`)
    (9f, caveat 4).

**Not determined.** (a) The md5 of the checkpoint against `checkpoint-591763` (not recomputed; sizes match).
(b) Whether the row-order phenotype join is correct for all 16 tissues (no current check exists; only vial
labels of four tissues were looked up). (c) Whether `load_state_dict` reported missing or unexpected keys in
the default run (the script discards them). (d) The exact call used for `pca_control.tsv` has no provenance
record; it is taken from the orchestrator log. (e) A batch-separation metric for the embeddings: none found.
(f) Augur's internal line references are to the de-parsed function, since the installed package is a binary
database.
