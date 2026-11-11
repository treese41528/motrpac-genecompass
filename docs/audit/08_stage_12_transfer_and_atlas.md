# Part 8. Stage 12 (cross-species transfer) and E.3 (Tabula Sapiens anchoring)

> One part of the methods audit of 2026-11-11; start at [`README.md`](README.md). The main findings of every
> part are collected in [`FINDINGS.md`](FINDINGS.md) and every reference in [`REFERENCES.md`](REFERENCES.md).
> Citations are `path:line` at commit `8249506` unless the text says otherwise.

Audit date 2026-11-11, branch `aim2b-perturbation-engine`. All paths are relative to
`/depot/reese18/apps/motrpac-genecompass`. Citations are `path:line` to files opened for this audit.

Two files moved under the audit. HEAD went from `e690c22` to `8249506` (committed 2026-11-11 18:05), which
added lines to `docs/OPEN_FRONTIER.md` and `deconvolution/DECONVOLUTION_PIPELINE_REPORT.md`; line numbers for those
two files are as of `8249506`. The gitignored manuscript `manuscript/motrpac_genecompass.tex` was also being
edited (modification time 2026-11-11 18:04:55); its line numbers are as of that state and will shift if it is
edited again. No other cited file changed.

Shorthand used in citations:

- `RR` = `data/deconvolution/genecompass_input_v2_refs4all_rebuild` (the default rat root,
  `config/pipeline_config.yaml:407`).
- `HR` = `data/deconvolution/genecompass_input_v2_refs4all_rebuild_human` (the default human root; it is not a
  config key, it is the rat root plus the suffix `_human`, `pipeline/run_stage12.py:195`).
- `AR` = `data/deconvolution/atlas_tabula_sapiens` (atlas outputs, `translation/project_to_atlas.py:173`).

Abbreviations: AUC, area under the ROC curve. BH, Benjamini-Hochberg false-discovery-rate adjustment. CLS, the
summary position the model prepends to each cell's token sequence. PLS-1, single-component partial least squares.
OOF, out of fold. ENSRNOG / ENSG, Ensembl rat / human gene identifiers. T1-T4, the ortholog tiers of Stage 3
(T1 tri-species one-to-one, T2a human one-to-one, T2b mouse one-to-one, T3a human one-to-many or many-to-many,
T3b mouse one-to-many or many-to-many, T4 rat-only new token;
`pipeline/03_ortholog_mapping/build_ortholog_mapping.py:21-26`). TS, Tabula Sapiens. E.1, E.2, E.3, the project's
names for the transfer, the human-space re-test and the atlas check
(`deconvolution/DOWNSTREAM_BUILD_PLAN.md:229-254`). "Block" = one (tissue, cell type). "Hotspot" = a block whose
rat-space gate q is below 0.05. "Pseudo-cell" = BayesPrism's expression profile of one cell type in one bulk
sample.

Values marked **[auditor count]** were obtained by me from a small on-disk table or pickle with a seconds-long
read. They are not project outputs and no project file states them unless a citation is also given.

## Overview of what Stage 12 is and is not

Stage 12 takes every rat pseudo-cell (the Stage 8 output), rewrites its gene list in human gene identifiers,
tokenizes it with human tokens and human medians, embeds it with the same fine-tuned model under the human species
token, and re-runs the supervised exercise probe on those embeddings (`pipeline/run_stage12.py:12-25`). The
project frames this as a one-directional transfer, not an alignment of two measured datasets, and as "a
counterfactual human representation of the rat response (an inference)" (`translation/transfer_to_human.py:11-14`;
`deconvolution/DOWNSTREAM_BUILD_PLAN.md:204-210`). No human expression measurement enters steps 1-4. The only
human data in this section is the Tabula Sapiens atlas used by E.3, which tests cell identity, not exercise
(`manuscript/motrpac_genecompass.tex:978-979`).

Default-run provenance: commit `aee68a3`, 2026-11-02, host `gilbreth-g008`
(`HR/liver/provenance_transfer_to_human.json:3-4`; `HR/gate_rcv10/provenance_subspace_probe.json:3-4`), run as part
of cross-tissue job 12122211 (`deconvolution/REFS4ALL_REBUILD_RESULTS.md:7`) by the runbook command
`python pipeline/run_stage12.py --jobs 48` (`deconvolution/DECONV_REBUILD_RUNBOOK.md:135`).

---

## Stage 12a - Re-expression of rat pseudo-cells as human (ortholog projection and tokenization)

### Research role

Question: can a rat pseudo-cell be written in the human gene vocabulary of the model without changing anything
except the species? The claim that rests on it is that the human-space embeddings in 12b-12d are the same cells
under a different species encoding, so that differences between rat-space and human-space results are due to the
encoding and not to tokenizer drift. It is checked by a unit test on the arithmetic, by summaries written per
tissue (genes mapped, count mass mapped, value scale), and by one cross-model input control. Verdict: the
arithmetic is consistent with the rat pseudo-cell tokenizer (Stage 9) on every shared step; 81-83% of each tissue's
pseudo-cell genes and 79-87% of its count mass survive the projection; 20 LUNG pseudo-cells that exist in rat
space do not exist in human space (below).

### Inputs and outputs

Inputs (default run, from `HR/liver/provenance_transfer_to_human.json`, `inputs` block, and
`translation/transfer_to_human.py:261-266`):

| Input | Path | Notes |
|---|---|---|
| Rat pseudo-cells | `RR/<tissue>/pseudocells.h5ad` | Stage 8 output; one file per tissue; 16 tissues discovered by glob (`pipeline/run_stage12.py:83-85`) |
| Ortholog map | `data/training/ortholog_mappings/rat_to_human_mapping.pickle` | Stage 3 output, 15,234 rat genes (`data/training/ortholog_mappings/mapping_statistics.json:36`), written 2026-04-01 at commit `b9e1d76` (`data/training/ortholog_mappings/stage3_manifest.json:5-6`) |
| Human token dictionary | `vendor/GeneCompass/prior_knowledge/human_mouse_tokens.pickle` | `config/pipeline_config.yaml:98`; only ENSG keys are kept (`translation/transfer_to_human.py:89-98`) |
| Human gene medians | `vendor/GeneCompass/scdata/dict/human_gene_median_after_filter.pickle` | `config/pipeline_config.yaml:97`; only positive medians kept (`translation/transfer_to_human.py:101-109`) |

Outputs per tissue under `HR/<tissue>/` (`translation/transfer_to_human.py:43-50, 291-302`): `dataset/` (HuggingFace
dataset with columns `input_ids`, `values`, `length`, `species`, `cell_id`, `sample`, `cell_type`, `tissue`),
`tokenize_summary.json`, `transfer_summary.json`, `provenance_transfer_to_human.json`, and relative symlinks
`pseudocells.h5ad` and `summary.txt` back to the rat folder.

Per-tissue record of the default run (each row from `HR/<tissue>/transfer_summary.json:2-7` and
`HR/<tissue>/tokenize_summary.json:2-13`; I read all 16 pairs):

| Tissue | Rat genes in pseudo-cells | Rat genes with a human ortholog (fraction) | Distinct human genes after summing | Rat genes merged by many-to-one | Human genes with token and median | Count mass mapped | Pseudo-cells kept | Dropped (< 200 genes) | Median non-zero token value | Mean expressed length |
|---|---|---|---|---|---|---|---|---|---|---|
| bat | 14,162 | 11,695 (0.8258) | 11,578 | 117 | 11,565 | 0.8378 | 300 | 0 | 0.763 | 2043.8 |
| blood | 14,920 | 12,147 (0.8141) | 11,976 | 171 | 11,959 | 0.7885 | 650 | 0 | 0.709 | 1614.5 |
| cortex | 15,385 | 12,608 (0.8195) | 12,479 | 129 | 12,460 | 0.8655 | 600 | 0 | 0.909 | 2007.8 |
| heart | 15,864 | 12,902 (0.8133) | 12,752 | 150 | 12,734 | 0.8468 | 800 | 0 | 0.824 | 1999.7 |
| hippoc | 15,313 | 12,475 (0.8147) | 12,333 | 142 | 12,316 | 0.8666 | 550 | 0 | 0.962 | 2019.7 |
| hypoth | 14,061 | 11,682 (0.8308) | 11,613 | 69 | 11,597 | 0.8585 | 650 | 0 | 0.913 | 2027.6 |
| kidney | 14,892 | 12,186 (0.8183) | 12,049 | 137 | 12,034 | 0.8437 | 750 | 0 | 0.912 | 2017.4 |
| liver | 12,766 | 10,599 (0.8303) | 10,486 | 113 | 10,474 | 0.8152 | 400 | 0 | 0.799 | 2048.0 |
| lung | 15,233 | 12,556 (0.8243) | 12,433 | 123 | 12,416 | 0.8453 | 1,012 | 38 | 1.027 | 1603.2 |
| ovary | 14,176 | 11,722 (0.8269) | 11,598 | 124 | 11,583 | 0.8364 | 432 | 0 | 0.873 | 1906.6 |
| skmgn | 13,142 | 10,946 (0.8329) | 10,854 | 92 | 10,842 | 0.8623 | 450 | 0 | 0.787 | 1991.4 |
| skmvl | 13,141 | 10,945 (0.8329) | 10,853 | 92 | 10,841 | 0.8543 | 450 | 0 | 0.803 | 2022.7 |
| smlint | 12,874 | 10,631 (0.8258) | 10,488 | 143 | 10,475 | 0.8049 | 650 | 0 | 0.767 | 2048.0 |
| spleen | 13,181 | 10,829 (0.8216) | 10,683 | 146 | 10,668 | 0.8149 | 700 | 0 | 0.852 | 1983.7 |
| testes | 13,711 | 11,226 (0.8188) | 11,147 | 79 | 11,129 | 0.8019 | 150 | 0 | 0.907 | 2048.0 |
| watsc | 12,865 | 10,687 (0.8307) | 10,575 | 112 | 10,564 | 0.8464 | 650 | 0 | 0.823 | 2046.6 |

Totals **[auditor count]**: 9,194 human-space pseudo-cells. The rat side embeds 9,214
(`pipeline/README.md:37`); the difference is 20 LUNG pseudo-cells (rat LUNG keeps 1,032 and drops 18,
`RR/lung/tokenize_summary.json:2-3`; human LUNG keeps 1,012 and drops 38, `HR/lung/tokenize_summary.json:2-3`).
Gene fraction mapped ranges 0.8133-0.8329 (mean 0.824) and count mass mapped 0.7885 (blood) to 0.8666 (hippoc)
(mean 0.837), so 13-21% of expression mass is lost; the manuscript states the same ranges
(`manuscript/motrpac_genecompass.tex:919-923`).

Two properties of these summary fields that a reader should know:

- `frac_count_mass_mapped` is one number per tissue, the sum of mapped counts over all pseudo-cells divided by the
  sum of full rat libraries (`translation/transfer_to_human.py:151`). It is weighted by pseudo-cell library size
  and is not a per-cell mean. It is computed before the token-and-median filter, so mass lost to human genes
  without a median (the difference between the "distinct human genes" and "with token and median" columns, 11-19
  genes per tissue) is not included and is not recorded anywhere.
- `mean_expressed_length` is taken over all pseudo-cells including the ones later dropped
  (`translation/transfer_to_human.py:216`).

### Procedure

Orchestrator step 1 (`pipeline/run_stage12.py:91-99`) runs, per tissue,
`translation/transfer_to_human.py --label <t> --in-root RR --out-root HR --target-sum 6400.0` (command recorded in
`HR/liver/provenance_transfer_to_human.json`, `command` block). Inside `main()` and `transfer_one()`:

1. **Load the three dictionaries** (`translation/transfer_to_human.py:281-283`). Identifiers are normalised by
   stripping the version suffix and upper-casing (`:79-81`).
2. **Project rat genes onto human genes** (`project_rat_to_human`, `:117-155`). Every pseudo-cell gene present in
   the ortholog map contributes a (rat column, human gene) pair (`:134`). A sparse 0/1 matrix P (rat genes by
   distinct human genes) is built (`:137-142`) and the human matrix is `X @ P` (`:145`), so when several rat genes
   map to one human gene their values are **summed**. Rat genes absent from the map are dropped from the gene list.
3. **Library size** is taken over **all** rat genes of the pseudo-cell, including the dropped ones (`:144`), so
   the surviving genes keep the value they have on the rat path. A zero library is replaced by 1 (`:178-179`).
4. **Eligible human genes** are those with both a token and a positive median
   (`:163-170`, using `build_eligible_gene_arrays`, `pipeline/05_tokenization/tokenize_corpus.py:200-222`).
5. **Normalise**: `Xn = Xh / full_rat_library * target_sum`, then restrict to eligible genes (`:180-181`).
6. **Tokenize** in batches of 1,024 cells (`:185-188`; `pipeline/05_tokenization/tokenize_corpus.py:121`) with the
   corpus function `tokenize_cell_batch` (`pipeline/05_tokenization/tokenize_corpus.py:299-374`): value =
   `log2(1 + x / median)` (`:342`); keep genes with value > 0 (`:345-346`); sort descending and keep the first
   `top_n` (`:357-358`); emit token IDs and values (`:362-363`); pad both with zeros to exactly `top_n` (`:366-368`).
   The sort is `np.argsort` on negated values with the default (unstable) algorithm, so the order of exactly tied
   genes is not fixed (`:357`; consequence documented in `docs/RETRAIN_V2_RESULTS.md:262-264`).
7. **Quality filter**: drop pseudo-cells with fewer than `min_genes` expressed eligible genes
   (`translation/transfer_to_human.py:191-192`).
8. **Write rows** with `species = [0]` for every cell and the rat `cell_id`, `sample`, `cell_type`, `tissue`
   carried over unchanged (`:201-211`); save the dataset and the two JSON summaries (`:294-296`); symlink the rat
   `pseudocells.h5ad` and `summary.txt` (`:298-299`); write provenance (`:300-302`).

**The ortholog table.** Source, release, cardinality rule and coverage:

- Source file: `data/references/biomart/rat_human_orthologs.tsv` (`config/pipeline_config.yaml:56`), an Ensembl
  BioMart export of release 115 / assembly GRCr8 fetched 2026-02-24 and stated to be byte-identical to the
  September 2025 archive (`config/pipeline_config.yaml:31-39`). Its md5 `88d44d06...` is recorded by Stage 3
  (`data/training/ortholog_mappings/stage3_manifest.json:28`) and pinned (`config/reference_md5sums.tsv:25`); the
  file on disk has that md5 **[auditor count]**.
- Which rat genes enter the map: a rat gene is written to `rat_to_human_mapping.pickle` if Stage 3 recorded a
  `human_ortholog` for it (`pipeline/03_ortholog_mapping/build_ortholog_mapping.py:787-794`). That happens in tier
  T1 (`:334-360`), T2a (`:362-380`) and T3a (`:396-410`), and in each case the human gene must already be in the
  GeneCompass vocabulary (`:335`, `:397`, `in_gc_vocab` set at `:470`). Counts: T1 13,178 + T2a 705 + T3a 1,351 =
  15,234 (`data/training/ortholog_mappings/mapping_statistics.json:8-11, 36`).
- Rat genes not in the map and therefore dropped at transfer: T2b 3,613 (mouse one-to-one only), T3b 892, T4 2,474,
  together 6,979 of the 22,213 vocabulary genes, 31.4% (`mapping_statistics.json:3, 10, 12, 13`). The technical
  report states the same thing: 15,234 pairs are about 69% of the 22,213 rat genes and about 82% of the genes in a
  given tissue's pseudo-cells, "not 94.5%" (`deconvolution/DECONVOLUTION_PIPELINE_REPORT.md:207-208`).
- Cardinality: every mapped rat gene has exactly one human target. For one-to-many and many-to-many orthology
  (T3a) the candidate with the highest percent identity is taken (`build_ortholog_mapping.py:397-399, 477-479`;
  `config/pipeline_config.yaml:259-260`), ties going to the first row in file order (Python `max`). The identity
  column used is the fifth column of the BioMart file, "%id. target Human gene identical to query gene"
  (`data/references/biomart/rat_human_orthologs.tsv:1`), selected by the loader's positional fallback
  (`build_ortholog_mapping.py:197-206`) because the header patterns at `:181-195` do not match this file's column
  names. No identity threshold is applied (`config/pipeline_config.yaml:255-257`).
- The reverse direction is many-to-one: the 15,234 rat genes map onto 14,476 distinct human genes; 413 human genes
  receive two or more rat genes (1,171 rat genes in all), the largest group being 29 rat genes **[auditor count
  from the pickle]**; the script's docstring states "up to 29-to-1" (`translation/transfer_to_human.py:120`). These
  are summed (step 2). Per tissue 69-171 rat genes are merged this way (table above).
- Of the 15,234 mapped rat genes, 14,987 have a human target with both a token and a median **[auditor count]**;
  the other 247 are mapped and then dropped at the eligibility step.

**The human vocabulary.** The token pickle has 50,558 keys, of which 23,113 are human ENSG with token IDs 2 to
23,114, plus `<pad>` = 0 and `<mask>` = 1; the human median pickle has 19,560 genes, all positive and all with a
token **[auditor count]**. The docstring gives the same ID range and notes they lie below the checkpoint's
vocabulary size (`translation/transfer_to_human.py:36-37`; `vocab_size` 55,275 at
`data/models/rat_genecompass_finetuned_v2/models/rat_phase2_mixed_species/models/config.json:30`).

### Models and software

No model is used in 12a. Python packages as pinned in `requirements.lock.txt`: anndata 0.12.10 (`:31`), datasets
4.8.4 (`:56`), numpy 2.4.3 (`:134`), scipy 1.17.1 (`:198`). The tokenizer function is imported from Stage 5
(`translation/transfer_to_human.py:72-74`).

### Parameters, thresholds and flags

| Name | Value | Where set (path:line) | What it controls | Rationale given |
|---|---|---|---|---|
| `--target-sum` | 6400 | `config/pipeline_config.yaml:413`; passed by `pipeline/run_stage12.py:97, 215-216`; recorded `HR/liver/provenance_transfer_to_human.json:53`. Fallback 6500 if the config key is absent (`translation/transfer_to_human.py:259-260`; `pipeline/run_stage12.py:216`) | Per-pseudo-cell library normalisation target | Calibrated on the rat side so liver pseudo-cells have the corpus's median token value, rounded to the nearest 100; Stage 12 "must equal the rat side" (`docs/RETRAIN_V2_PREREGISTRATION.md:155-162`; `config/pipeline_config.yaml:411-412`) |
| `--top-n` | 2048 | default `TOP_N_DEFAULT` (`translation/transfer_to_human.py:252`; `pipeline/05_tokenization/tokenize_corpus.py:122`); recorded `provenance_transfer_to_human.json:54`. Not read from `reference_assembly.top_n_genes` (`config/pipeline_config.yaml:294`), which holds the same value | Sequence length (genes per cell) | "GeneCompass paper: top-2048 genes per cell" (`tokenize_corpus.py:122`) |
| `--min-genes` | 200 | `translation/transfer_to_human.py:253-254`; recorded `provenance_transfer_to_human.json:55` | Pseudo-cells with fewer expressed eligible genes are dropped | "GeneCompass QC parity" (`:254`) |
| `HUMAN_SPECIES_ID` | 0 | `translation/transfer_to_human.py:76` | Value written to the `species` column | "0=human, 1=mouse, 2=rat" (`:76`) |
| `CELL_BATCH_SIZE` | 1024 | `pipeline/05_tokenization/tokenize_corpus.py:121` | Cells tokenized per batch | memory (`:121`) |
| Many-to-one rule | sum | `translation/transfer_to_human.py:141-145` | How several rat orthologs of one human gene combine | "total ortholog count-mass" (`:120-121`) |
| Library denominator | all rat genes, mapped or not | `translation/transfer_to_human.py:144, 180` | Value scale of surviving genes | Mapped-only library "would inflate every value ~1.2x" (`:127-131`) |
| Ortholog map | `<paths.ortholog_dir>/rat_to_human_mapping.pickle` | `translation/transfer_to_human.py:263-264`; `config/pipeline_config.yaml:82` | Rat-to-human gene map | none stated beyond "done" in Stage 3 |
| Token dictionary | `paths.genecompass_tokens` | `translation/transfer_to_human.py:265`; `config/pipeline_config.yaml:98` | Human token IDs | upstream GeneCompass |
| Median dictionary | `paths.genecompass_medians` | `translation/transfer_to_human.py:266`; `config/pipeline_config.yaml:97` | Per-gene divisor | upstream GeneCompass human medians |
| Output root | `<in-root>_human` | `translation/transfer_to_human.py:262`; `pipeline/run_stage12.py:195` | Where human-space files go | mirrors the rat layout so detection scripts run unchanged (`translation/transfer_to_human.py:43-44`) |
| Preference for training-regulated genes | not applied | `translation/transfer_to_human.py:183` | Whether listed genes are promoted into the top 2,048 | The rat default also does not apply it (`pa_genes: None` in `RR/liver/provenance_tokenize_pseudocells.json`, `params` block) |

### Statistical procedures

None. The step is deterministic apart from the order of exactly tied values (step 6).

### Checks, validation and their results

1. **Unit tests** (`tests/test_transfer.py`): many-to-one summing and dropping of unmapped genes with the full
   library kept (`:33-39`); an empty map is an error (`:42-44`); tokens and values equal the rat-path arithmetic
   computed by hand on a toy example, species 0, IDs carried over (`:47-62`); the 200-gene filter (`:65-70`). These
   test the arithmetic on four synthetic genes, not the production data.
2. **What "parity" is parity with.** The script states that it "reuses the EXACT corpus tokenization math:
   normalize_total(target_sum) over the full gene set -> divide by per-gene median -> log2(1+x) -> rank descending
   -> top-2048 -> zero-pad" and that "ONLY THREE things change vs the rat path: the token dict (human ENSG), the
   median dict (human), and the species token (0, not 2)" (`translation/transfer_to_human.py:25-31`). Compared with
   the code:
   - The **training-corpus** tokenizer does not library-normalise at all ("no normalize_total",
     `pipeline/05_tokenization/tokenize_corpus.py:308, 338-342`). Library normalisation exists only on the two
     pseudo-cell paths: rat (`deconvolution/tokenize_pseudocells.py:10-18, 132-136`) and human
     (`translation/transfer_to_human.py:176-181`). So the parity that holds is with the Stage 9 rat pseudo-cell
     tokenizer, not with the corpus tokenizer; the first quoted sentence is inaccurate on the normalisation step.
   - Against the rat pseudo-cell path the shared steps are identical in code: same full-library denominator
     (`deconvolution/tokenize_pseudocells.py:133-135` vs `translation/transfer_to_human.py:144, 180`), same
     `tokenize_cell_batch` (`tokenize_pseudocells.py:169` vs `transfer_to_human.py:186`), same 200-gene filter
     (`tokenize_pseudocells.py:100, 176` vs `transfer_to_human.py:253, 191`), same target sum and top-n (rat
     `RR/liver/tokenize_summary.json`: 6400, 2048, species 2; human `HR/liver/tokenize_summary.json:5-7`).
   - A fourth difference, implied but not listed as one of the three: the **gene set**. Rat genes without a human
     ortholog are removed, rat genes sharing a human ortholog are summed, and lower-ranked genes move up into the
     vacated top-2,048 positions.
   - The medians: v2's rat ("hybrid") median equals GeneCompass's human median for every rat gene whose human
     ortholog has one, by construction, "to keep rat cells and their human re-expression (Stage 12) on the same
     per-gene scaling" (`pipeline/05_tokenization/build_hybrid_medians.py:11-27`). So for a gene mapped one-to-one
     the value in the human encoding equals the value in the rat encoding.
   - A numerical nuance: the human path divides by a float64 library (`translation/transfer_to_human.py:144`)
     while the rat path works in float32 (`deconvolution/tokenize_pseudocells.py:132-135`), so values agree to
     float32 precision rather than bit for bit.
3. **Value scale.** Median non-zero token value per tissue is 0.709-1.027 in human space (table above) against
   0.878 (liver), 0.838 (blood) and 1.121 (lung) on the rat side (`RR/<tissue>/tokenize_summary.json`, key
   `value_stats.median`; I read these three). The human value is lower in each of the three (0.799, 0.709, 1.027).
   Removing the non-orthologous genes changes which genes make up the sequence; no file breaks the shift down
   further. The reference the script prints, "corpus reference: median 0.869"
   (`translation/transfer_to_human.py:190, 196-197`), is the April corpus's value; v2's corpus median is 0.9145
   (`docs/RETRAIN_V2_PREREGISTRATION.md:159`).
4. **Cross-model input control** (2026-10-24, July-reference arm, 8,450 cells): v2's human-space tokens are the
   same cells and tokens as April's, values equal after rescaling 6,400 to 6,500 (|difference| <= 2e-15), no order
   inversions, 930 cells with exactly tied tokens in another order, one cell with a different tied gene at the
   2,048 cut (`docs/RETRAIN_V2_RESULTS.md:259-264`; `data/results/retrain_v2/c5/stage12_input_control.txt`). This
   compares two models' human inputs with each other. It is not a rat-versus-human comparison and it was not
   repeated on the current default references.
5. **Reproduction.** The clean rebuild's human root is byte-identical to the previously assembled tree in all 127
   compared files (`deconvolution/REFS4ALL_REBUILD_RESULTS.md:27, 37`).
6. **A design review** ("5-agent map + 4-lens adversarial review", 2026-06-24) is cited as having verified parity
   and as having found the mapped-only-library bug (`deconvolution/DOWNSTREAM_BUILD_PLAN.md:214-223`). No artifact
   of that review is in the repository; the bug fix itself is in the code (step 3).

### Deviations, known defects and caveats

- **20 LUNG pseudo-cells are lost only in human space.** The 200-gene filter is applied after the projection, so
  a pseudo-cell that passes on the rat side can fail on the human side. All 20 are LUNG goblet cells: the block has
  32 cells in the rat probe table and 12 in the human one, and every other block has equal n **[auditor count from
  `RR/gate_rcv10/subspace_probe.tsv` and `HR/gate_rcv10/subspace_probe.tsv`, column `n`]**. The docs record the 18
  rat-side drops (`docs/OPEN_FRONTIER.md:79`) and that "LUNG goblet has no human block"
  (`notebooks/RERUN_EDITS.md:97`), but I found no statement of the human-side count (9,194) or of the extra 20.
- **A data test asserts the opposite.** `tests/test_v2_outputs.py:62` asserts
  `human["n_pseudocells"] == rat["n_pseudocells"]` for every tissue. By the on-disk summaries this is false for
  LUNG on the default (1,012 vs 1,032). I did not run the test (it opens h5ad files); either it fails under
  `pytest -m data` or it is not being run.
- **Stale statements in the script's docstring.** "Rat-specific (T4) genes have NO human ortholog and are DROPPED"
  (`translation/transfer_to_human.py:38`) understates the drop: T2b and T3b genes are dropped too (they have a
  mouse ortholog but no entry in the rat-to-human map). "per tissue ~10.7k-11.6k human-eligible genes remain"
  (`:39`) does not match the current default (10,474-12,734, table above). The "corpus reference" print is April's
  (check 3).
- **Stale statements in the build plan.** "`rat_to_human_mapping.pickle`, 15,234 / 94.5%" and "~5% non-mappable rat
  genes are dropped" (`deconvolution/DOWNSTREAM_BUILD_PLAN.md:232, 239`) are superseded by 69% of the vocabulary /
  about 82% of pseudo-cell genes (`deconvolution/DECONVOLUTION_PIPELINE_REPORT.md:207-208`).
- **No per-cell-type drop log.** `n_dropped_lt_min_genes` is recorded per tissue only
  (`translation/transfer_to_human.py:213`).
- **The human ortholog must be in GeneCompass's vocabulary** to be in the map (Procedure). A rat gene whose Ensembl
  human ortholog exists but is absent from that vocabulary is treated as having no human ortholog.
- The `species` column written here is not what the embedding step reads (12b, caveats).

### Methods references

- Ortholog source: Ensembl BioMart release 115 / GRCr8 (`config/pipeline_config.yaml:31-39`); no publication is
  cited in the code.
- GeneCompass tokens, medians and tokenization: Yang X, Liu G, Feng G, et al. "GeneCompass: deciphering universal
  gene regulatory mechanisms with a knowledge-informed cross-species foundation model." Cell Research
  2024;34:830-845 (as in `manuscript/motrpac_genecompass.tex:1487-1489`).

---

## Stage 12b - Embedding in human space and the transfer comparison (cosine, hotspot preservation)

### Research role

Question: when the same pseudo-cells are embedded under the human encoding, (i) how far does each cell move, and
(ii) does each rat-space exercise hotspot still show a detectable trained-versus-control separation? The claim
resting on it is the manuscript's "of the 9 exercise hotspots, all 9 are preserved"
(`manuscript/motrpac_genecompass.tex:938-941`). It is checked by a per-block comparison table and a sex control.
Verdict so far: 9 of 9 on the default, 11 of 11 on the 14-tissue arm; the per-cell cosine has been explicitly
withdrawn as evidence that a cell "stays near" its rat embedding (below).

**What the result can and cannot mean.** The rat and human encodings of one pseudo-cell are produced from the same
counts by the same model. From the code, they differ only in the following ways:

1. the species row used for the CLS position, 2 for rat and 0 for human (`finetune/genecompass/embed_cells.py:152-154`;
   `vendor/GeneCompass/genecompass/modeling_bert.py:405-410`);
2. genes with no entry in the rat-to-human map (T2b, T3b, T4) and the 247 mapped genes without a human median are
   absent (12a);
3. the 1,351 T3a genes carry a different token: on the rat side each has its own new token
   (`pipeline/03_ortholog_mapping/build_ortholog_mapping.py:400-407`), on the human side its count is added to its
   human ortholog's pretrained token (12a step 2);
4. rat genes sharing a human ortholog are summed;
5. lower-ranked genes fill the freed positions.

For T1 and T2a genes (13,178 + 705 = 13,883) the rat token ID **is** the human ortholog's token ID
(`build_ortholog_mapping.py:342-345, 365-368`), and by 12a check 2 their values are equal too whenever the human
gene has a median and receives no other rat gene. 13,883 of the 15,234
ortholog-mapped genes is 91.1% (`translation/transfer_to_human.py:34-36`;
`deconvolution/DECONVOLUTION_PIPELINE_REPORT.md:2953-2956`; `docs/OPEN_FRONTIER.md:143`; confirmed
**[auditor count]** by comparing `rat_tokens.pickle` with `human_mouse_tokens.pickle`). The prior-knowledge lookup
is by token ID with no remapping for species 0 or 2; only species 1 (mouse) is remapped
(`vendor/GeneCompass/genecompass/knowledge_embeddings.py:192-194`).

Two consequences for interpretation:

- **The 91.1% figure is a property of the vocabulary, not a measured per-cell overlap.** It is the share of
  ortholog-mapped rat genes whose token ID equals their human ortholog's. The share of positions in a given cell's
  2,048-token sequence that are identical in the two encodings is **NOT FOUND** in any file (looked in
  `HR/*/transfer_summary.json`, `HR/*/tokenize_summary.json`,
  `data/results/retrain_v2/c5/stage12_input_control.txt`, `docs/RETRAIN_V2_RESULTS.md`, `docs/OPEN_FRONTIER.md`).
  What is recorded is that 81-83% of a tissue's pseudo-cell genes and 79-87% of count mass survive (12a).
- **"Transfer" here is model self-consistency under a change of species encoding, not independent human evidence.**
  The project says so in several places: "transferability in the embedding space ... not validation against
  measured human exercise cells. The sex gate is a within-block fidelity control (rat labels), not independent
  human data" (`deconvolution/AIM2_DECONV_RESULTS.md:421-425`); "Its checks so far reuse rat labels, so they show
  the transfer keeps the rat structure" (`translation/ATLAS_PROJECTION_PREREG.md:7-9`); "no human measurement enters
  at any point" (`manuscript/motrpac_genecompass.tex:1043-1046`, written about the Stage 13 network score, which
  the project calls model self-consistency and which this audit does too). Hotspot preservation should be read in
  that sense: the same labels, the same cells, the same model, a near-identical input.

### Inputs and outputs

Inputs: `HR/<tissue>/dataset` (12a); model weights (below); for the comparison, the rat gate tables
`RR/gate_rcv10/corroboration_merged.tsv` and `RR/gate_rcv10/subspace_probe.tsv` and the human probe table
`HR/gate_rcv10/subspace_probe.tsv` (`pipeline/run_stage12.py:129-133`). The gate folder is taken from
`deconvolution.hotspots_table` (`config/pipeline_config.yaml:417`; `pipeline/run_stage12.py:72-80`).

Outputs: `HR/<tissue>/embeddings/cell_embeddings.npy` (float32, cells by 768; e.g. liver 1,228,928 bytes = 400 x
768 x 4 + header) with `provenance_embed_cells.json`; `HR/gate_rcv10/transfer_comparison.tsv` (197 rows, 15
columns **[auditor count]**) and `HR/gate_rcv10/transfer_comparison.md`.

### Procedure

**Step 2 of the orchestrator: embed** (`pipeline/run_stage12.py:100-108`). Per tissue:
`finetune/genecompass/embed_cells.py --model-dir <checkpoint> --dataset HR/<t>/dataset --output HR/<t>/embeddings
--n-cells 1000000 --species 0 --device cuda` (recorded in
`HR/liver/embeddings/provenance_embed_cells.json:10-17, 45-52`). Inside `extract_embeddings`:

1. Read `config.json`, load `pytorch_model.bin` on CPU (`finetune/genecompass/embed_cells.py:90-92, 101-103`).
2. Take the four prior-knowledge matrices (promoter, co-expression, gene family, PECA GRN) from the checkpoint's own
   state dict (`:106-111`) and the human-mouse homolog map from the vendored file (`:113-116`).
3. Build `BertForMaskedLM` with `warmup_steps = 0`, `emb_warmup_steps = 1` (`:119-122`), replace the CLS embedding
   by a 3-row table (`:125-126`), load weights with `strict=False` (`:129`), set eval mode (`:130`).
4. Load the dataset; if `--n-cells` is smaller than the dataset take an order-preserving prefix (`:133-139`). The
   orchestrator passes 1,000,000 so every cell is embedded (`pipeline/run_stage12.py:63, 217`).
5. For each batch of 32 cells (`:146-148`, default `:193`): `input_ids`, `values`, a species tensor filled with the
   command-line species (`:150-154`), attention mask = 1 where `input_ids != 0` (`:157`), forward through
   `model.bert` with `emb_warmup_alpha = 1.0` (`:160-167`). The model prepends the CLS position and extends the
   mask for it (`vendor/GeneCompass/genecompass/modeling_bert.py:361-362, 405-410`).
6. **Representation taken**: the final-layer hidden state at position 0, `outputs[0][:, 0, :]`, 768 numbers per
   cell (`finetune/genecompass/embed_cells.py:169-171`). It is the last hidden state, not the pooler output.
7. Save in dataset row order (`:176-181`) and write provenance (`:207-214`).

**Per-cell cosine** (`translation/transfer_cosine.py`; not called by the orchestrator). For each tissue with both
embeddings, pair cells by `cell_id` (`:39-46`), cosine between the rat-space and human-space vectors (`:47`),
report mean, median, 5th percentile and minimum (`:64-65`) and the median over tissues (`:70`). It exits with an
error if a human-space cell is missing from the rat set or if IDs are duplicated (`:32-35, 43-45`); rat cells with
no human partner (the 20 LUNG cells) are simply not paired.

**Step 4 of the orchestrator: compare** (`pipeline/run_stage12.py:125-135`; `translation/compare_transfer.py`).

1. Rat side: trained AUC, permutation p and BH q from `corroboration_merged.tsv`; ordinal-dose rho and sex AUC from
   the rat `subspace_probe.tsv` (`translation/compare_transfer.py:100-107`).
2. Left-join the human probe table on (tissue, cell type) (`:108-109`); add the rat Augur column if present, and
   the human one only if a human `corroboration_merged.tsv` exists (`:113-126`).
3. `is_hotspot = q_sup_trained < 0.05` on the **rat** q (`:128`); `delta_auc = human - rat` (`:129`); status by
   `classify` (`:64-75, 130`).
4. Write the TSV (`:134`) and a Markdown verdict with three parts: sex control (`:136-138, 155-161`), global
   fidelity (`:140-141, 163-165`), hotspot survival (`:143-146, 167-182`).

### Models and software

| Item | Value | Where pinned |
|---|---|---|
| Model | Rat GeneCompass retrain v2, phase-2 final save, directory `data/models/rat_genecompass_finetuned_v2/models/rat_phase2_mixed_species/models` | `config/pipeline_config.yaml:362-364` (explicit `model_checkpoint`; comment: md5 of the weights `ae6100ff...`, identical to `checkpoint-591763`); resolved by `deconvolution/model_checkpoint.py:27-38`; used in the run: `HR/liver/embeddings/provenance_embed_cells.json:45, 52` |
| Same model as the rat side | yes: Stages 9, 12 and 13 read the same key | `deconvolution/model_checkpoint.py:1-10`; `pipeline/run_stage12.py:15-16, 196` |
| Architecture | BERT encoder (`BertForMaskedLM`), 12 layers, hidden size 768, 12 attention heads, intermediate size 3072, GELU, dropout 0.02, max 2,048 positions, vocabulary 55,275, value channel on, four prior-knowledge inputs on, CLS position on | `data/models/rat_genecompass_finetuned_v2/models/rat_phase2_mixed_species/models/config.json:2-30` |
| Model code | `vendor/GeneCompass` submodule at commit `66ab82f` (upstream `xCompass-AI/GeneCompass`) | `git submodule status` **[auditor read]**; `.gitmodules:2-3` |
| Species encoding | CLS embedding table with 3 rows, 0 human, 1 mouse, 2 rat; row 0 selected by `--species 0` | `finetune/genecompass/embed_cells.py:76-77, 124-126, 152-154`; `pipeline/run_stage12.py:106` |
| Libraries | torch 2.11.0, transformers 4.57.6, datasets 4.8.4, numpy 2.4.3, pandas 2.3.3, scipy 1.17.1 | `requirements.lock.txt:221, 226, 56, 134, 155, 198`; transformers also in `config.json:21` |
| Weights file recorded in provenance | size 1,224,674,943 bytes and mtime only (no md5 in the record) | `HR/liver/embeddings/provenance_embed_cells.json:32` and following lines |

### Parameters, thresholds and flags

| Name | Value | Where set (path:line) | What it controls | Rationale given |
|---|---|---|---|---|
| `--species` | 0 | `pipeline/run_stage12.py:106`; default of the script is 2 (`finetune/genecompass/embed_cells.py:194`) | Row of the CLS species table | human = 0 |
| `--n-cells` | 1,000,000 | `pipeline/run_stage12.py:63, 217` | Cap on cells (not binding) | "never subsample" (`:63`) |
| `--batch-size` | 32 | `finetune/genecompass/embed_cells.py:193` | Inference batch | none stated |
| `--device` | cuda | `pipeline/run_stage12.py:184` | Device | embed step needs a GPU (`:33-34`) |
| `emb_warmup_alpha` | 1.0 | `finetune/genecompass/embed_cells.py:165` | Weight of the prior-knowledge inputs | none stated |
| Config overrides at load | `warmup_steps = 0`, `emb_warmup_steps = 1` | `finetune/genecompass/embed_cells.py:119-120` | Model construction only | none stated |
| `--hotspot-q` | 0.05 | `translation/compare_transfer.py:96-97` | Which rat blocks are hotspots | rat gate rule (`config/pipeline_config.yaml:414-416`) |
| `--auc-thresh` | 0.65 | `translation/compare_transfer.py:93-94` | Human AUC floor for PRESERVED and WEAKENED | "floor for a 'detectable' axis" (`:94`); no derivation stated |
| `--p-thresh` | 0.05 | `translation/compare_transfer.py:95` | Human permutation p for PRESERVED | none stated |
| Sex floor | 0.70 on the median human sex AUC | hard-coded, `translation/compare_transfer.py:156` | PASS / FAIL of the sex control | called "pre-registered" (`:13-14`); see caveats |
| Minimum pairs for a Spearman | 3 | `translation/compare_transfer.py:58` | Guard | none stated |

### Statistical procedures

**Per-cell cosine.** A descriptive statistic, no test. Registered as a reported metric before it was computed for
either model: "per tissue the mean, median and 5th percentile over all paired cells; summary = the median over
tissues of the per-tissue mean" (`docs/RETRAIN_V2_PREREGISTRATION.md:211-214`).

**Hotspot status.** For each block, with human trained AUC `a` and human permutation p `p` (both from 12c):

- PRESERVED if `a >= 0.65` and `p < 0.05`;
- WEAKENED if `a >= 0.65` and not (`p < 0.05`);
- LOST if `a < 0.65`;
- `no_human_block` if the human AUC is missing

(`translation/compare_transfer.py:64-75`). Notes on what this rule is:

- The p is the **raw, one-sided permutation p** of the human-space probe. **No multiplicity control is applied on
  the human side.** The BH adjustment exists only on the rat side, over the 151 analysable blocks, and defines the
  hotspot set (`deconvolution/corroborate_summary.py:60-67`). No human-space `corroboration_merged.tsv` exists in
  the default tree (listing of `HR/gate_rcv10/`: `provenance_subspace_probe.json`, `subspace_probe.tsv`,
  `transfer_comparison.md`, `transfer_comparison.tsv`).
- The rule uses the human block alone. It does not test the change from the rat AUC, and no interval is given for
  `delta_auc`.
- Only the binary trained-versus-control probe enters. The ordinal-dose rho is tabulated but plays no part in the
  status.
- AUC and p are compared after rounding to 3 and 4 decimals (`deconvolution/subspace_probe.py:219-220, 238-239`).
- Units: one pseudo-cell per bulk sample per cell type; hotspot blocks have n = 50 **[auditor count from
  `HR/gate_rcv10/subspace_probe.tsv`]**.

**Sex control.** PASS if the median over blocks of the human-space sex AUC is at least 0.70
(`translation/compare_transfer.py:137-138, 156`); the Spearman correlation between rat and human sex AUC over
blocks is reported alongside (`:137`).

**Global fidelity.** Spearman correlation over blocks between rat and human trained AUC, and between rat and human
dose rho (`translation/compare_transfer.py:140-141`); descriptive, no test.

**Pre-registration status.** The retrain pre-registration lists, for Stage 12, the metrics to report ("status
counts among each model's own hotspots and over all blocks, the sex control's human-space median, and the rat ~
human Spearman", `docs/RETRAIN_V2_PREREGISTRATION.md:249-251`) and the gate method (10 splits, 10,000
permutations, in rat and human space, `:218-220`). I found no `*_PREREG.md` that registers the 0.65 / 0.05 status
thresholds or the 0.70 sex floor as pass criteria. They have been constants of `compare_transfer.py` since its
first commit, `fc4a497` of 2026-06-25 **[auditor read of `git show fc4a497:translation/compare_transfer.py`]**,
which predates every v2 and refs4 result. The results document says of the sex check: "The check is not a
registered C5 criterion and is reported as printed" (`docs/RETRAIN_V2_RESULTS.md:332-333`).

### Checks, validation and their results

**Per-cell cosine.**

| Arm | Cells paired | Median over tissues of the per-tissue mean | Source |
|---|---|---|---|
| June smoke test, liver only, April model | not on disk | 0.979 | `deconvolution/AIM2_DECONV_RESULTS.md:393`; "July's 0.979 (liver, smoke job 11103493) is not on disk" (`docs/RETRAIN_V2_PREREGISTRATION.md:214-215`) |
| April model, July references, 14 tissues | 8,450 | 0.973 | `docs/RETRAIN_V2_RESULTS.md:285`; `data/results/retrain_v2/c5/stage12_cosine_april.tsv` |
| v2, July references, 14 tissues | 8,450 | 0.969 | `docs/RETRAIN_V2_RESULTS.md:285`; `data/results/retrain_v2/c5/stage12_cosine_v2.tsv` |
| v2, current default, 16 tissues | 9,194 | 0.974 (median of medians 0.976) | `notebooks/pipeline12_analysis.ipynb:291-310` (executed output); `notebooks/RERUN_EDITS.md:11-13`; `manuscript/motrpac_genecompass.tex:929-933` |

On the default the per-tissue mean ranges 0.960 (bat) to 0.986 (blood), while two **different** pseudo-cells of the
same tissue have a median rat-space cosine of 0.989-0.998 (median over tissues 0.993)
(`notebooks/pipeline12_analysis.ipynb:291-310`). A cell is therefore farther from its own human-space embedding
than from other cells of its tissue. The project's reading: "the cosine mostly measures how far apart the species'
regions of the embedding sit, not whether a cell keeps its identity. The July reading of 0.979 as 'the same cell
lands near its rat embedding' ... does not hold" (`docs/RETRAIN_V2_RESULTS.md:288-294`). For the default arm the
cosine exists only as notebook output; `transfer_cosine.py` wrote TSVs only for the two July-reference arms.

**Transfer comparison, default (16 tissues)**, all from `HR/gate_rcv10/transfer_comparison.md`:

| Check | Result | Line |
|---|---|---|
| Blocks compared; hotspots | 197; 9 | `:3` |
| Sex control | median sex AUC rat 0.743, human 0.745; Spearman 0.930 over 173 blocks; PASS (floor 0.70) | `:6-7` |
| Trained AUC, Spearman rat vs human | 0.766 (n = 196) | `:10` |
| Dose rho, Spearman rat vs human | 0.725 (n = 196) | `:11` |
| Hotspots | PRESERVED 9 of 9, WEAKENED 0, LOST 0 | `:14` |

Per hotspot (`HR/gate_rcv10/transfer_comparison.md:18-26`; human p and both dose p values **[auditor read]** from
`HR/gate_rcv10/transfer_comparison.tsv` and the two `subspace_probe.tsv` files):

| Hotspot | Rat AUC | Human AUC | Human p | Rat dose rho (p) | Human dose rho (p) |
|---|---|---|---|---|---|
| SKMVL Satellite cells | 0.887 | 0.863 | 0.0003 | 0.519 (0.0002) | 0.380 (0.0053) |
| SKMVL Vascular smooth muscle cells | 0.874 | 0.811 | 0.0014 | 0.424 (0.0024) | 0.414 (0.0027) |
| BLOOD Megakaryocytes and platelets | 0.837 | 0.855 | 0.0003 | 0.376 (0.0092) | 0.372 (0.0095) |
| BLOOD Erythroid cells | 0.828 | 0.799 | 0.0024 | 0.356 (0.0114) | 0.340 (0.0167) |
| SKMGN Tenocytes | 0.825 | 0.925 | 0.0001 | 0.298 (0.0211) | 0.462 (0.0019) |
| BLOOD Classical monocytes | 0.821 | 0.827 | 0.0015 | 0.316 (0.0249) | 0.310 (0.0271) |
| BLOOD Neutrophils | 0.819 | 0.830 | 0.0006 | 0.383 (0.0064) | 0.268 (0.0489) |
| LUNG Vascular smooth muscle cells | 0.817 | 0.774 | 0.0079 | 0.284 (0.0334) | -0.138 (0.7458) |
| HYPOTH Oligodendrocyte precursor cells | 0.798 | 0.692 | 0.0384 | 0.112 (0.2075) | -0.240 (0.9233) |

Over all 197 blocks: 44 PRESERVED, 29 WEAKENED, 123 LOST, 1 without a human block (LUNG goblet cells)
(`deconvolution/ALL_TISSUE_RERUN_RESULTS.md:25`; `deconvolution/AIM2_DECONV_RESULTS.md:367-368`; confirmed
**[auditor count]** from the TSV). 108 of the 123 LOST blocks had a rat AUC below 0.65 already
(`manuscript/motrpac_genecompass.tex:944-945`; confirmed **[auditor count]**).

**Transfer comparison, 14-tissue arm** (`data/deconvolution/genecompass_input_v2_refs4_human/gate_rcv10/transfer_comparison.md`):
165 blocks, 11 hotspots (`:3`); sex 0.717 to 0.730, Spearman 0.918 over 159 blocks, PASS (`:6-7`); trained AUC
Spearman 0.751 (n = 164), dose 0.670 (`:10-11`); PRESERVED 11 of 11 (`:14`), the nine above with identical values
plus LUNG Endothelial cells (0.786 to 0.841) and LUNG Mesothelial cells (0.785 to 0.734) (`:27-28`). All blocks:
42 / 15 / 107 (`deconvolution/ALL_TISSUE_RERUN_RESULTS.md:25`).

**Earlier arms**, for context: v2 on the July references 13 of 13, April 14 of 16
(`docs/RETRAIN_V2_RESULTS.md:310`); the sex check printed FAIL for both models on those references (human median
0.671 and 0.6875, `:312, 328-333`).

**Reproduction.** `run_stage12.py` reproduced the v2 probe table and both comparison files byte for byte when
rerun (`docs/RETRAIN_V2_RESULTS.md:339-341`); the clean rebuild reproduced the whole human root
(`deconvolution/REFS4ALL_REBUILD_RESULTS.md:27`).

**Unit tests.** Status rule on five synthetic rows, with `auc_thresh = 0.70` passed explicitly
(`tests/test_transfer.py:74-80`); gate-folder resolution (`:89-106`); commands carry the gate method and the
target sum and species (`:109-127`).

**Auditor's illustrations (not project results; simple arithmetic on `HR/gate_rcv10/transfer_comparison.tsv`).**
They show how much the headline depends on the choice not to correct on the human side:

- Applying BH to the human p values over the same 151 analysable blocks gives human q < 0.05 for 7 of the 9
  hotspots; LUNG vascular smooth muscle cells (q 0.070) and HYPOTH oligodendrocyte precursor cells (q 0.182) would
  not pass. Under that calculation 16 analysable blocks have human q < 0.05, nine of which are not rat hotspots
  (for example SKMGN and SKMVL skeletal muscle cells, LUNG endothelial cells, SKMVL tenocytes).
- Applying the PRESERVED rule itself (AUC >= 0.65 and p < 0.05) in rat space selects 41 blocks; 31 of them are
  PRESERVED in human space, and 13 blocks are PRESERVED in human space without meeting the rule in rat space.

### Deviations, known defects and caveats

- **"Preserved" is an uncorrected, single-block criterion** (Statistical procedures and the illustration above).
  The manuscript states the rule correctly ("human AUC >= 0.65 and permutation p < 0.05",
  `manuscript/motrpac_genecompass.tex:939`) but does not say that the p is uncorrected while the hotspot set was
  defined with BH.
- **The dose axis is not part of the status, and for two hotspots it is not detected in human space.** The
  orchestrator describes the test as whether "the trained-vs-control / ordinal-dose axis SURVIVES"
  (`pipeline/run_stage12.py:8-9`). LUNG vascular smooth muscle cells go from dose rho 0.284 (p 0.033) to -0.138
  (p 0.75) and HYPOTH oligodendrocyte precursor cells from 0.112 (p 0.21) to -0.240 (p 0.92), yet both are
  PRESERVED (`HR/gate_rcv10/transfer_comparison.md:25-26`).
- **The rat and human tests are not independent replicates.** They use the same cells, the same labels, and, by
  the seeding scheme, the same fold splits and the same permutation sequences (12c). Agreement is expected to a
  large degree by construction.
- **"Pre-registered" sex control.** The script, an older results section and the manuscript call sex the
  "pre-registered" control with a "pre-set 0.70 validity floor" (`translation/compare_transfer.py:13-14`;
  `deconvolution/AIM2_DECONV_RESULTS.md:400-403`; `manuscript/motrpac_genecompass.tex:935-938, 1365-1366`). I could
  not find the pre-registration; see Pre-registration status. The floor is applied to a pooled median that is close
  to 0.70 in rat space as well and failed for both models on the July references
  (`docs/RETRAIN_V2_RESULTS.md:328-333`).
- **No human-space corroboration in the default.** The comparison's optional fourth section (canonical Augur in
  human space) is absent because nothing in Stage 12 produces a human `corroboration_merged.tsv`
  (`translation/compare_transfer.py:111-126, 184-192`); "Human-space Augur is run for neither model"
  (`docs/RETRAIN_V2_PREREGISTRATION.md:243-245`). The only tree with human-space Augur, PCA control and
  `pheno_merge_test` is April's `data/deconvolution/genecompass_input_human/`, produced by
  `slurm/analysis/run_stage12_corroborate.slurm:25-49`.
- **`embed_cells.py` loads weights with `strict=False` and does not inspect missing or unexpected keys**
  (`finetune/genecompass/embed_cells.py:129`), unlike the evaluation script, which raises on unexpected keys
  (`finetune/genecompass/eval_mlm.py:86-88`). Whether every tensor, including the 3-row species table, was loaded
  from the checkpoint cannot be confirmed without loading it, which this audit did not do.
- **The species used at inference is the command-line flag, not the dataset column.** The `species` column written
  in 12a is ignored; the tensor is filled from `--species` (`finetune/genecompass/embed_cells.py:152-154`), whose
  default is 2 (`:194`). The orchestrator passes 0 and the provenance confirms it
  (`HR/liver/embeddings/provenance_embed_cells.json:50`), so the default run is correct; a manual call without the
  flag on a human dataset would embed it as rat without any warning.
- **Hard-coded paths in `embed_cells.py`**: the rat token dictionary and the homolog map are read from fixed
  relative paths (`finetune/genecompass/embed_cells.py:96, 113`) rather than from config; the token dictionary is
  loaded and not used.
- **`compare_transfer.py` defaults do not match the default layout.** Its default table paths are in the root of
  the input folders (`translation/compare_transfer.py:83-92`), whereas the default gate tables are in
  `gate_rcv10/`. It works only because the orchestrator passes explicit paths (`pipeline/run_stage12.py:129-133`).
- **`deconvolution/pheno_merge_test.py` is not called by Stage 12.** The orchestrator runs four scripts
  (`pipeline/run_stage12.py:57-60`). The trace-eta-squared test in that file is run by Stage 11 on the rat root only
  (`pipeline/run_stage11.py:14-16`). `subspace_probe.py` does recompute the same global eta-squared in human space
  as its measure (0) (`deconvolution/subspace_probe.py:15, 197-200`), but those columns are not read by the
  comparison.
- **Stale documentation.** `pipeline/README.md:29-33` names the default output folders `..._refs4all/` and says
  `refs_v4all.yaml` is a no-op; the config's default has been the `_refs4all_rebuild` folders since 2026-11-03
  (`config/pipeline_config.yaml:399-407`), and the no-op profile is `refs4all_rebuild.yaml`
  (`config/profiles/refs4all_rebuild.yaml:6`). `pipeline/README.md:25` says there is no Stage 11, while
  `pipeline/run_stage11.py` exists and produces the gate tables Stage 12 requires
  (`pipeline/run_stage12.py:199-204`). `pipeline/README.md:40` still quotes the 14-tissue atlas figure.
- **Provenance records size and mtime of the weights, not a hash** (table above).

### Methods references

- GeneCompass (model, CLS species token): Yang et al., Cell Research 2024;34:830-845
  (`manuscript/motrpac_genecompass.tex:1487-1489`).
- Benjamini-Hochberg (rat-side hotspot definition): Benjamini Y, Hochberg Y. J R Stat Soc Series B
  1995;57:289-300 (`manuscript/motrpac_genecompass.tex:1545-1547`).
- Augur (absent human-space corroboration): Skinnider MA, Squair JW, Kathe C, et al. Nat Biotechnol
  2021;39:30-34 (`manuscript/motrpac_genecompass.tex:1499-1503`).

---

## Stage 12c - The gate re-run in human space

### Research role

Question: is there a detectable trained-versus-control, ordinal-dose and sex separation in the human-space
embedding of each block? It supplies the human columns of the comparison in 12b. What is re-run is **only the
supervised probe**; the other parts of the rat gate (Stage 11: embedding quality control, the phenotype-merge
eta-squared test, the PCA control, canonical Augur, and the BH merge that defines hotspots,
`pipeline/run_stage11.py:13-20`) are not run on the human root.

### Inputs and outputs

Inputs: `HR/<tissue>/embeddings/cell_embeddings.npy` and `HR/<tissue>/dataset` (cell type and sample labels);
`deconvolution/reference/motrpac_sample_pheno.tsv`; `data/deconvolution/motrpac_bulk/<TISSUE>/bulk_samples.tsv`
(`HR/gate_rcv10/provenance_subspace_probe.json`, `params` block; 36 input files recorded). Output:
`HR/gate_rcv10/subspace_probe.tsv`, 197 rows and 33 columns **[auditor count]**, with
`provenance_subspace_probe.json`.

### Procedure

Orchestrator step 3 (`pipeline/run_stage12.py:115-124`):
`deconvolution/subspace_probe.py --gc-root HR --out HR/gate_rcv10/subspace_probe.tsv --perms 10000 --folds 5
--repeats 10 --jobs 48` (as run: `HR/gate_rcv10/provenance_subspace_probe.json:246-249`).

1. For every tissue folder with embeddings, a dataset and a bulk sample list, join each pseudo-cell to its animal:
   sample name `mix{i}` is the i-th vial label in `bulk_samples.tsv`, which is looked up in the phenotype table for
   group and sex (`deconvolution/subspace_probe.py:265-282`). "Trained" is every non-control group (`:280-282`).
2. One work item per cell type with at least 6 labelled pseudo-cells (`:283-291`).
3. Per item, compute the eta-squared measures and their permutation p values (`:193-210`) and the three supervised
   probes (`:212-234`), in parallel processes (`:295-296`).
4. Write the table and provenance (`:297-301`).

### Models and software

No model. scikit-learn 1.8.0 (`StratifiedKFold`, `roc_auc_score`), scipy 1.17.1 (`spearmanr`), joblib 1.5.3, numpy
2.4.3 (`requirements.lock.txt:196, 198, 100, 134`). BLAS and OpenMP threads are pinned to 1 inside the script
(`deconvolution/subspace_probe.py:44-48`).

### Parameters, thresholds and flags

| Name | Value | Where set (path:line) | What it controls | Rationale given |
|---|---|---|---|---|
| `--perms` | 10,000 | `pipeline/run_stage12.py:179-180`; as run `provenance_subspace_probe.json:246` | Label permutations per test | Seed stability study: with 1 split and 1,000 permutations "the seed decides April's hotspot set" (`docs/RETRAIN_V2_PREREGISTRATION.md:218-228`) |
| `--repeats` | 10 | `pipeline/run_stage12.py:181-182`; `provenance_subspace_probe.json:248` | Independent fold splits averaged | same; "One fold split moves a block's AUC by about 0.04" (`docs/RETRAIN_V2_PREREGISTRATION.md:223`) |
| `--folds` | 5 | hard-coded in the orchestrator, `pipeline/run_stage12.py:121`; `provenance_subspace_probe.json:247` | Folds per split; reduced to the smallest class count if smaller (`deconvolution/subspace_probe.py:115-118`) | none stated |
| `--jobs` | 48 as run; default 8 | `pipeline/run_stage12.py:183`; `deconvolution/DECONV_REBUILD_RUNBOOK.md:135` | Parallel processes | runtime only |
| Minimum pseudo-cells per block | 6 | `deconvolution/subspace_probe.py:285` | Which blocks are scored | none stated |
| Dose coding | control 0, 1w 1, 2w 2, 4w 4, 8w 8 | `deconvolution/subspace_probe.py:69` | Ordinal dose | weeks of training (`:32`) |
| Block seed | `1000 * (crc32(tissue) mod 9973) + index of the cell type in the tissue's sorted list` | `deconvolution/subspace_probe.py:72-75, 283, 291` | All randomness of a block | crc32 is stable across runs, the built-in string hash is not (`:73-74`) |
| Fold-split seeds | split r of the trained probe: `seed + 1 + 7919 r`; dose `seed + 3 + 7919 r`; sex `seed + 5 + 7919 r` | `deconvolution/subspace_probe.py:123-127, 215, 223, 230` | Fold membership | r = 0 is the earlier single split (`:124`) |
| Permutation seeds | eta-squared tests `seed`; trained `seed + 2`; dose `seed + 4`; sex `seed + 6` | `deconvolution/subspace_probe.py:179, 218, 226, 233` | Null draws | none stated |
| Rounding | statistics 3 decimals, p 4 decimals | `deconvolution/subspace_probe.py:219-220, 238-239` | Stored precision | none stated |

### Statistical procedures

**Supervised trained-versus-control probe** (the statistic 12b uses).

- Statistic: for each of 10 stratified 5-fold splits, in each fold compute the direction
  `w = (X_train - mean)^T (y_train - mean(y_train))`, normalise it, and score the held-out cells as
  `(X_test - mean_train) . w` (`deconvolution/subspace_probe.py:130-140`); the AUC of the pooled out-of-fold scores
  against y (1 trained, 0 control) is one split's value; the statistic is the mean over the 10 splits
  (`:148-161`). Folds are stratified on the trained label only (`:215`).
- Null: y is permuted across all pseudo-cells of the block (`rng.permutation`, unrestricted, so not within sex) and
  the whole cross-validation is repeated on the same 10 splits, 10,000 times (`:169-172`). The docstring attributes
  the design to Ojala and Garriga 2010 (`:26-28`).
- p value: `(1 + number of permuted statistics >= observed) / (1 + 10,000)`, one-sided, so the smallest possible
  value is 0.0001 (`:172-173`). A permutation whose statistic is undefined counts as not exceeding (`:172`).
- Units of replication: pseudo-cells, one per bulk sample per cell type.
- Multiplicity: none in human space (12b).

**Ordinal-dose probe.** Same construction with y = weeks, folds stratified on the five-level group, statistic the
Spearman correlation between out-of-fold scores and weeks, same permutation scheme
(`deconvolution/subspace_probe.py:221-227, 155-156`).

**Sex probe.** Same with y = 1 for male, folds stratified on sex (`:228-234`). Sex-chromosome genes were removed
before deconvolution (`config/pipeline_config.yaml:379-384`), so this is an autosomal signal.

**Eta-squared measures**, also written for group, trained and sex: global trace eta-squared, standardised
eta-squared, and the per-dimension maximum, each with a 10,000-permutation p (`deconvolution/subspace_probe.py:189-210`).
Not used downstream in Stage 12.

**Pairing with the rat gate.** Because the block seed depends only on the tissue name and the cell type's rank in
the sorted list, and the human root has the same 197 blocks in the same order, a block has the same seed in both
spaces. For every block with equal n (all but LUNG goblet cells) the fold splits and the permutation sequences are
therefore identical in the rat and human runs. This follows from the code cited in the parameter table; it is not
stated in the documents.

**Pre-registered:** the method (10 splits, 10,000 permutations, "for both models in rat and human space") was
logged on 2026-10-24 before any v2 gate number existed (`docs/RETRAIN_V2_PREREGISTRATION.md:218-220`).

### Checks, validation and their results

- The human table has the same 197 blocks as the rat table; n is equal in 196 and differs in one (LUNG goblet
  cells, 32 vs 12), which has no human trained AUC **[auditor count]**; that is the `no_human_block` row.
- 173 blocks have a sex AUC in both spaces (`HR/gate_rcv10/transfer_comparison.md:6`); the single-sex tissues
  (OVARY, TESTES) cannot have one.
- With `--repeats 1` the code reproduced April's on-disk rat and human probe tables in every column
  (`docs/RETRAIN_V2_PREREGISTRATION.md:235-237`).
- Command line carries the gate method: `tests/test_transfer.py:109-119`.

### Deviations, known defects and caveats

- "Gate re-run" overstates what is repeated: only the probe is. In particular there is no human-space
  embedding-versus-PCA control and no human-space BH.
- The permutation is not blocked by sex. Sex is a strong axis in these embeddings (median sex AUC 0.743 rat,
  0.745 human, `HR/gate_rcv10/transfer_comparison.md:6`), and the design crosses 2 sexes with 5 groups
  (`deconvolution/pheno_merge_test.py:10`), but the null does not preserve that balance in each draw. This is
  inherited from the rat gate and applies equally there.
- `--folds` is fixed at 5 in the orchestrator and cannot be changed from the command line
  (`pipeline/run_stage12.py:121`).
- The probe reads the bulk sample lists from the default `motrpac_bulk_out` (`deconvolution/subspace_probe.py:249`),
  not from anything under the human root; the join is by row order of the mixture (`:36-37`).

### Methods references

- Permutation test that re-runs the whole cross-validation: "Ojala & Garriga 2010" as cited in
  `deconvolution/subspace_probe.py:27`; not in the manuscript bibliography. Full reference from memory: Ojala M,
  Garriga GC. Permutation tests for studying classifier performance. J Mach Learn Res 2010;11:1833-1863 [verified against the JMLR record; see REFERENCES.md].
- PLS-1: no citation in the code or the manuscript.

---

## Stage 12d (E.3) - Projection onto Tabula Sapiens

### Research role

Question: does a transferred rat pseudo-cell lie nearest the **real human** cell type it should, when real human
cells are put through the same tokenizer and model? This is the only part of Stage 12 in which human measurements
enter. The claim resting on it: "the human-space coordinates are biological, not only a re-encoding of rat
structure" (`translation/ATLAS_PROJECTION_RESULTS.md:49-51`). It is a pre-registered, interpretive test, "not a
gate: a FAIL weakens cell-type claims made in human space; it does not undo the rat-space results"
(`translation/ATLAS_PROJECTION_PREREG.md:40-41`). Verdict: PASS on both runs, with named failures.

**What it replaces.** Before the test, the manuscript's limitation (v) read: "The cross-species transfer is
demonstrated in embedding space and is not yet anchored to a real human atlas (the planned E.3 check)"
(`manuscript/_july_backup_20261031/motrpac_genecompass.tex:947-948`). The pre-registration tied the outcome to
that sentence: on FAIL "the manuscript's limitation (v) stands as written"
(`translation/ATLAS_PROJECTION_PREREG.md:35-36`); on PASS the results document says it "can be replaced by this
result" (`translation/ATLAS_PROJECTION_RESULTS.md:51`; `docs/OPEN_FRONTIER.md:163`). The current limitation (v)
states the anchoring holds only at the level of cell
identity, lists the failures, and ends "no human exercise measurement enters the transfer"
(`manuscript/motrpac_genecompass.tex:1213-1220`).

### Inputs and outputs

**Atlas.** Tabula Sapiens from the CELLxGENE collection `e5f58829-1a66-40b5-a624-9046778e74f5`, "revision of
2026-06-11" (`translation/ATLAS_PROJECTION_PREREG.md:11-12`; `translation/ATLAS_PROJECTION_16_PREREG.md:15-17`).
Eleven tissue files under `data/external/tabula_sapiens/`, each listed with dataset ID, byte size and md5 in
`data/external/tabula_sapiens/MANIFEST.txt:1-11`. They were fetched with
`curl https://datasets.cellxgene.cziscience.com/<dataset id>.h5ad`, the size and md5 being appended to the manifest
at download (`slurm/analysis/ts_download.slurm:17-31` for the first nine; `slurm/analysis/ts_download16.slurm:18-25`
for Spleen and Ovary). A named Tabula Sapiens release or version label is **NOT FOUND** (looked in both
pre-registrations, both results documents, the manifest, the two download scripts and
`translation/project_to_atlas.py`).

Human side as prepared (each row from `AR/<Tissue>/prep_summary.json:3-9` and its `cell_types` block; dataset IDs
from the manifest):

| TS tissue file | CELLxGENE dataset ID | 10x cells in file | Pseudobulks | TS cell types | Median non-zero token value | Maps to MoTrPAC tissue(s) |
|---|---|---|---|---|---|---|
| Blood | 1bf55482-... | 82,785 | 90 | 18 | 0.532 | blood |
| Lung | fddf43a3-... | 62,954 | 102 | 28 | 0.512 | lung |
| Muscle | 4919a450-... | 41,876 | 68 | 22 | 0.509 | skmgn, skmvl |
| Heart | 82197577-... | 25,419 | 42 | 15 | 0.495 | heart |
| Liver | a4b3a49e-... | 21,398 | 57 | 20 | 0.447 | liver |
| Kidney | 76b6cf23-... | 11,024 | 10 | 10 | 0.478 | kidney |
| Fat | 8115812f-... | 93,195 | 72 | 20 | 0.503 | watsc, bat |
| Small_Intestine | 6db4dfe6-... | 40,350 | 89 | 31 | 0.484 | smlint |
| Testis | 09e3ae14-... | 7,513 | 8 | 8 | 0.612 | testes |
| Spleen | 5498457f-... | 67,601 | 97 | 21 | 0.527 | spleen |
| Ovary | 6ad78e84-... | 45,384 | 20 | 8 | 0.461 | ovary |

Totals **[auditor count]**: 499,499 10x cells in the eleven files, 655 pseudobulks (538 in the nine first-run
tissues, matching `translation/ATLAS_PROJECTION_RESULTS.md:15-17`). Every pseudobulk has 17,810 eligible genes and
none fell below 200 genes (`prep_summary.json:5, 8` in each). The tissue mapping is the constant `TISSUES`
(`translation/project_to_atlas.py:37-39`). CORTEX, HIPPOC and HYPOTH are out of scope: "There is no brain in
Tabula Sapiens" (`translation/ATLAS_PROJECTION_PREREG.md:13-14`).

**Rat side.** The transferred pseudo-cells of 12a/12b: first run `data/deconvolution/genecompass_input_v2_refs4_human/`
(`translation/ATLAS_PROJECTION_PREREG.md:20-21`), second run `HR`
(`translation/ATLAS_PROJECTION_16_PREREG.md:7-8`; scorer default `translation/project_to_atlas.py:185-186`).

**Correspondence table.** `translation/atlas_correspondence.tsv`, three columns (`tissue_label`, `rat_cell_type`,
`ts_cell_types`, the last a "; "-separated list, empty meaning no counterpart; `translation/project_to_atlas.py:94-101`).
161 rows: 129 from the first run, 14 SPLEEN and 18 OVARY added for the second (`translation/atlas_correspondence.tsv:2-162`;
`translation/ATLAS_PROJECTION_16_PREREG.md:22-24`). 120 rows have a counterpart and 41 are empty **[auditor
count]**.

**Outputs.** First run: `AR/score/` ; second run: `AR/score_refs4all_rebuild/`; each holds
`atlas_projection_per_type.tsv` and `summary.txt`. Committed copies: `translation/atlas_record/` and
`translation/atlas_record_16/`, byte-identical to the scored outputs **[auditor check with `cmp`]**; the second also
holds the Tabula Sapiens name lists the new map rows were written from (`translation/atlas_record_16/Spleen_celltypes.tsv`,
`Ovary_celltypes.tsv`).

### Procedure

Driver scripts (gitignored, local): `slurm/analysis/atlas_e3.slurm:22-27` (first run, job 12103831,
`translation/ATLAS_PROJECTION_RESULTS.md:4`) and `slurm/analysis/atlas_e3_16.slurm:22-28` (second run, job
12192423, `translation/ATLAS_PROJECTION_16_RESULTS.md:4`). Three steps.

**1. Prepare** (`translation/project_to_atlas.py prep --tissue <T>`, function `prep`, `:42-91`).

1. Read `<atlas-dir>/<Tissue>.h5ad`; use `.raw.X` if a raw slot exists, else `.X` (`:48-51`); gene IDs normalised as
   in 12a (`:51`).
2. Keep cells whose `assay` string contains "10x" (`:53`). Reason given: "Smart-seq2 counts are on another scale"
   (`translation/ATLAS_PROJECTION_PREREG.md:14`).
3. Group by `donor_id` and `cell_type` (`:55-58`); keep groups with at least 20 cells (`:35, 59`).
4. Sum raw counts within each group; abort if the sums are not integers (`:60-63`).
5. Normalise each pseudobulk to the target sum over **all** genes of the file, then restrict to genes with a human
   token and median (`:64-70`); tokenize with the same `tokenize_cell_batch`, top 2,048 (`:71-74`); keep
   pseudobulks with at least 200 expressed genes (`:75`).
6. Save a dataset with `species = [0]`, `cell_id = donor|cell_type`, `donor`, `cell_type`, `n_cells` (`:77-84`) and
   `prep_summary.json` (`:85-90`).

**2. Embed.** `finetune/genecompass/embed_cells.py --model-dir <v2 final save> --dataset AR/<T>/dataset --output
AR/<T>/embeddings --species 0 --device cuda` (`slurm/analysis/atlas_e3.slurm:19, 24-25`; for Spleen and Ovary also
recorded in `AR/Spleen/embeddings/provenance_embed_cells.json`). Same script and model as 12b.

**3. Score** (`translation/project_to_atlas.py score`, function `score`, `:104-164`). For each MoTrPAC tissue that
has atlas embeddings (`:110-114`):

1. Load the transferred rat embeddings R with their cell types and the atlas embeddings T with theirs (`:115-117`).
2. Two variants: "centered", `R - mean(R)` and `T - mean(T)`, each set centred on its own mean over **all** its
   rows in that tissue; and "raw" (`:119`).
3. One centroid per Tabula Sapiens cell type = the unweighted mean of that type's (centred) pseudobulk embeddings
   (`:118, 121`). Every type with at least one pseudobulk in the tissue is a candidate, whether or not any rat
   type maps to it.
4. Normalise pseudo-cells and centroids to unit length; cosine matrix; each rat pseudo-cell is assigned the
   centroid with the highest cosine (`:122-124`).
5. For each rat cell type with a non-empty map entry (`:125-128`): `top1_acc` = fraction of its pseudo-cells
   assigned to any of its mapped Tabula Sapiens types; `chance` = fraction of that tissue's Tabula Sapiens types that
   are in its mapped set; `top1_mode` = the most frequent assigned type (`:129-137`). A warning is printed if none
   of the mapped names exists in the tissue (`:131-132`); no row triggered it (`n_match` is at least 1 in all 120
   rows **[auditor count]**).
6. Write the per-type table (`:139-141`), then the summary and verdict (`:142-164`).

**Correspondence rules and ambiguity.** As registered: the map "was written from the cell-type names alone (rat
reference labels and the Tabula Sapiens `cell_type` values with at least one donor pseudobulk), with no embedding
computed" (`translation/ATLAS_PROJECTION_PREREG.md:3-5`); "a rat type maps to every Tabula Sapiens type it names or
is a subtype of; empty when there is no counterpart" (`translation/ATLAS_PROJECTION_16_PREREG.md:25-26`). In the
table this gives four kinds of row:

- one rat type to one atlas type (e.g. `blood  Neutrophils -> neutrophil`, `translation/atlas_correspondence.tsv:10`);
- one rat type to several atlas types, any of which counts as correct (e.g. `lung  Endothelial cells -> capillary
  endothelial cell; vein endothelial cell; endothelial cell of artery`, `:23`; `kidney  Immune cells ->` eight
  types, `:65`, giving chance 0.8 for that row);
- several rat types to the same atlas type (nine kidney tubule and duct types to `kidney epithelial cell`,
  `:60-74`; three spermatid stages to `spermatid`, `:107-109`);
- no counterpart, left empty and not scored (41 rows; e.g. `lung  Mesothelial cells`, `:27`; `testes  Sertoli
  cells`, `:110`; twelve of eighteen OVARY rows, `:131-148`).

The table has no column recording the reason for a row. Every one of the 161 blocks in the 13 covered tissues has
a row, so no block is skipped silently **[auditor count against `HR/gate_rcv10/subspace_probe.tsv`]**. The 36
brain blocks have none by design. Coverage of the default: 120 of 197 blocks and 5,681 of 9,194 human-space
pseudo-cells are scored **[auditor count from `translation/atlas_record_16/atlas_projection_per_type.tsv`, column
`n`]**.

### Models and software

Same model, checkpoint, species token and representation as 12b (`translation/ATLAS_PROJECTION_PREREG.md:16-18`;
`slurm/analysis/atlas_e3.slurm:19`; `AR/Spleen/embeddings/provenance_embed_cells.json`, `params.model_dir`).
Human tokens and medians as in 12a (`translation/project_to_atlas.py:64-65`). anndata 0.12.10, scipy 1.17.1, numpy
2.4.3, datasets 4.8.4 (`requirements.lock.txt:31, 198, 134, 56`).

### Parameters, thresholds and flags

| Name | Value | Where set (path:line) | What it controls | Rationale given |
|---|---|---|---|---|
| `MIN_CELLS` | 20 | hard-coded, `translation/project_to_atlas.py:35` | Minimum cells per (donor, cell type) pseudobulk | registered (`translation/ATLAS_PROJECTION_PREREG.md:14-15`); "so each human point is a pseudobulk like the rat pseudo-cells" |
| Assay filter | `assay` contains "10x" | `translation/project_to_atlas.py:53` | Which cells are used | Smart-seq2 on another scale (`ATLAS_PROJECTION_PREREG.md:14`) |
| Grouping | `donor_id` x `cell_type` | `translation/project_to_atlas.py:55` | Pseudobulk unit | registered (`ATLAS_PROJECTION_PREREG.md:14-15`) |
| Target sum | 6400 (from config; `--target-sum` overrides) | `translation/project_to_atlas.py:47, 174`; `AR/<T>/prep_summary.json:7` | Library normalisation | "the rat run's target sum" (`ATLAS_PROJECTION_PREREG.md:16`) |
| Top-n | 2048 | `translation/project_to_atlas.py:73` (`TOP_N_DEFAULT`) | Sequence length | as the transfer |
| Minimum genes | 200 | hard-coded, `translation/project_to_atlas.py:75` | Drop pseudobulks | as the transfer (`ATLAS_PROJECTION_PREREG.md:17`) |
| Species | 0 | `translation/project_to_atlas.py:78`; `slurm/analysis/atlas_e3.slurm:25` | Species token | human |
| Tissue pairs | 13 MoTrPAC labels to 11 atlas files | `translation/project_to_atlas.py:37-39` | What is scored | registered (`ATLAS_PROJECTION_PREREG.md:11-14`; `ATLAS_PROJECTION_16_PREREG.md:20`) |
| Centring | each set on its own within-tissue mean | `translation/project_to_atlas.py:119` | Removes a global offset | "the two sources differ in technology and in how a pseudobulk is formed, so a global offset is expected and is not the question" (`ATLAS_PROJECTION_PREREG.md:23-25`) |
| Distance | cosine to centroid, highest wins | `translation/project_to_atlas.py:122-124` | Assignment | registered (`ATLAS_PROJECTION_PREREG.md:25-26`) |
| `--n-perm` | 10,000 | `translation/project_to_atlas.py:180` | Null draws | registered (`ATLAS_PROJECTION_PREREG.md:31-32`) |
| Random seed | `np.random.default_rng(0)`, one generator for both variants, centred first | `translation/project_to_atlas.py:107, 143, 154` | Null draws | none stated |
| PASS rule | `p < 0.05` and accuracy `>= 2 x` null mean; WEAK if only `p < 0.05`; FAIL otherwise; centred variant only | `translation/project_to_atlas.py:158` | Verdict | registered (`ATLAS_PROJECTION_PREREG.md:33-36`) |
| `--map` | `translation/atlas_correspondence.tsv` | `translation/project_to_atlas.py:178` | Correspondence | registered |
| `--out` | first run `AR/score` (default, `:179, 187`); second run `AR/score_refs4all_rebuild` (`slurm/analysis/atlas_e3_16.slurm:28`) | as cited | Output folder | keep the first run's folder (`ATLAS_PROJECTION_16_PREREG.md:30-31`) |

### Statistical procedures

- **Statistic (primary):** macro mean of `top1_acc` over all scored rat cell types of all tissues, each type
  weighted equally (`translation/project_to_atlas.py:145`). It is nearest-centroid top-1 agreement, not a
  k-nearest-neighbour vote.
- **Null:** within each MoTrPAC tissue, permute which centroid carries which Tabula Sapiens name; one permutation per
  tissue per draw, applied to all rat types of that tissue; recompute the macro accuracy; 10,000 draws
  (`:146-156`). The embeddings, the centroids and each pseudo-cell's assigned centroid are held fixed; only the
  name-to-centroid correspondence is shuffled. SKMGN and SKMVL (both scored against Muscle) and WATSC and BAT (both
  against Fat) are separate tissues for this purpose and get independent draws (`:147-155`, keyed by MoTrPAC label).
  The expected null accuracy of a type is its `chance`, so types mapped to many atlas names raise the observed
  statistic and the null together.
- **p value:** `(1 + number of null draws >= observed) / (1 + 10,000)`, one-sided (`:157`). The reported 0.0001 is
  the smallest attainable value: no null draw reached the observed accuracy.
- **Units:** the null treats atlas cell-type names as exchangeable within tissue. Rat pseudo-cells are not
  resampled; a rat type's 50 pseudo-cells (24 in OVARY, 25 in TESTES, 12 for LUNG goblet cells; column `n`) are
  the same cell type in different animals and are highly dependent. Accuracy is averaged within a type before the
  macro mean, so each type counts once.
- **Multiplicity:** one primary test per run. Per-tissue and per-type values, the hotspot rows and the uncentred
  variant are "reported without a verdict" (`translation/ATLAS_PROJECTION_PREREG.md:37-38`).
- **Seeds:** generator seed 0 (table above). Preparation and embedding have no random component.
- **Subsampling of the atlas:** none. All 10x cells of each file enter the grouping. Two exclusions apply: non-10x
  cells and groups with fewer than 20 cells. The number of cells or groups removed by either is **not logged** by
  `prep` (it prints the total and the 10x count and writes `n_cells_10x` and `n_pseudobulks`,
  `translation/project_to_atlas.py:54, 86-90`). For Spleen and Ovary the committed name lists give the cells inside
  kept pseudobulks: 67,364 of 67,601 and 45,272 of 45,384 10x cells **[auditor sum of the `n_cells` column of
  `translation/atlas_record_16/Spleen_celltypes.tsv` and `Ovary_celltypes.tsv`]**. For the nine first-run tissues
  no such record exists.
- **Pre-registration.** First run: `translation/ATLAS_PROJECTION_PREREG.md`, commit `4895834` (2026-10-31 17:35),
  scoring script `a18ce30` (17:36), map `de65969` (17:40); the first atlas dataset and embedding folders are
  time-stamped 17:52 **[auditor read of `git log` and directory mtimes]**, consistent with "Registered before any
  Tabula Sapiens cell was embedded" (`ATLAS_PROJECTION_PREREG.md:3`). Second run:
  `translation/ATLAS_PROJECTION_16_PREREG.md`, commit `780b149` (2026-11-04 23:07), map rows `0595e80` (23:10),
  Spleen embedding provenance created 23:14:36. The 129 first-run map rows are unchanged since `de65969`, and the
  only change to the scoring script between the runs is the addition of the two tissue pairs **[auditor `git diff`
  and `diff`]**, as registered (`ATLAS_PROJECTION_16_PREREG.md:20, 22`). Both documents end: "No other centring,
  distance, pseudobulk size or map revision will be tried after the results are seen."

### Checks, validation and their results

| | First run (14-tissue arm, 11 MoTrPAC tissues scored) | Second run (16-tissue default, 13 scored) |
|---|---|---|
| Macro top-1 accuracy, centred | 0.404 over 103 rat types | 0.430 over 120 rat types |
| Null mean (95th percentile) | 0.084 (0.124) | 0.086 (0.125) |
| p (10,000 permutations) | 0.0001 | 0.0001 |
| Ratio to null mean | 4.8 | 5.0 |
| Verdict | PASS | PASS |
| Uncentred (secondary, no verdict) | 0.144 vs 0.085, p 0.0662 | 0.140 vs 0.087, p 0.0591 |
| Types with accuracy >= 0.5 / exactly 0 | 39 / 29 of 103 | 49 / 31 of 120 |
| Source | `translation/atlas_record/summary.txt:1, 13`; `translation/ATLAS_PROJECTION_RESULTS.md:12-13, 52, 61` | `translation/atlas_record_16/summary.txt:1, 15`; `translation/ATLAS_PROJECTION_16_RESULTS.md:8-11, 50-51` |

Per tissue, second run, accuracy (mean chance, scored types) (`translation/atlas_record_16/summary.txt:2-14`):
ovary 0.653 (0.125, 6); spleen 0.553 (0.091, 11); skmvl 0.550 (0.062, 8); liver 0.529 (0.050, 7); heart 0.512
(0.078, 12); lung 0.432 (0.050, 17); skmgn 0.415 (0.062, 8); testes 0.400 (0.125, 5); bat 0.395 (0.200, 4); smlint
0.369 (0.044, 11); blood 0.320 (0.086, 11); watsc 0.290 (0.087, 8); kidney 0.267 (0.167, 12).

**Reproduction.** All 103 first-run per-type accuracies are equal in the second run (maximum difference 0.0), and
over those 11 tissues alone the accuracy is again 0.404 (`translation/ATLAS_PROJECTION_16_RESULTS.md:13-16`;
0.4037 **[auditor count]**).

**Hotspots** (`translation/ATLAS_PROJECTION_16_RESULTS.md:38-48`; values identical in the first run,
`translation/ATLAS_PROJECTION_RESULTS.md:35-46`; `n_match` and number of candidate atlas types from the per-type
table):

| Default hotspot | Accuracy | Most frequent assigned atlas type | Mapped names / candidates |
|---|---|---|---|
| SKMVL Vascular smooth muscle cells | 1.00 | smooth muscle cell | 1 / 22 |
| BLOOD Megakaryocytes and platelets | 0.74 | platelet | 1 / 18 |
| BLOOD Neutrophils | 0.66 | neutrophil | 1 / 18 |
| BLOOD Classical monocytes | 0.48 | monocyte | 2 / 18 |
| SKMVL Satellite cells | 0.46 | slow muscle cell (not its counterpart) | 1 / 22 |
| SKMGN Tenocytes | 0.02 | erythrocyte | 1 / 22 |
| LUNG Vascular smooth muscle cells | 0.00 | alveolar adventitial fibroblast | 1 / 28 |
| BLOOD Erythroid cells | 0.00 | classical monocyte | 1 / 18 |
| HYPOTH Oligodendrocyte precursor cells | not scored | no brain in the atlas | - |

So three of the nine default hotspots land on their counterpart in most pseudo-cells, two in just under half, three
do not, and one cannot be tested. The two additional 14-tissue hotspots: LUNG Endothelial cells 1.00; LUNG
Mesothelial cells not scored, no counterpart (`translation/ATLAS_PROJECTION_RESULTS.md:37, 46`).

**Failures the project reports.**

- Three hotspots: "SKMGN tenocytes, LUNG vascular smooth muscle and BLOOD erythroid cells. Human-space claims about
  those three should not be made from this transfer." A cause is proposed for erythroid cells only: atlas
  erythrocytes are enucleated with little RNA beyond globins, while the rat erythroid profile is deconvolved from
  whole blood against a reference that includes precursors (`translation/ATLAS_PROJECTION_RESULTS.md:56-60`). No
  cause is given for tenocytes or LUNG vascular smooth muscle.
- KIDNEY (one donor; every epithelial segment is one atlas type and several segments land on immune centroids),
  TESTES (one donor), and lymphocyte subtypes (`translation/ATLAS_PROJECTION_RESULTS.md:52-55`). In the per-type
  table seven of twelve KIDNEY rows score exactly 0 **[auditor count]**.
- SPLEEN: CD4+ and CD8+ T cells 0.04 and NK cells 0.00, all landing on regulatory T cells; endothelial cells 0.20
  and monocytes 0.28, mostly on macrophages. OVARY: theca cells 0.04 and smooth muscle cells 0.00, on the ovarian
  stromal centroid; twelve OVARY rat types have no counterpart (`translation/ATLAS_PROJECTION_16_RESULTS.md:27-34`).

**Why centring matters.** Without it the test does not reach p < 0.05 in either run. The project attributes this
to a global offset: the atlas pseudobulks' non-zero token values have medians 0.45-0.61 against 0.71 for the
transferred BLOOD pseudo-cells (`translation/ATLAS_PROJECTION_RESULTS.md:61-63`; table of `value_median` above).

### Deviations, known defects and caveats

- **What PASS establishes.** The macro statistic exceeds a name-shuffle null. It does not establish that each
  type, or each hotspot, is anchored: 31 of 120 types score 0, and only three of nine default hotspots land on
  their counterpart in a majority of pseudo-cells. The manuscript reports these failures
  (`manuscript/motrpac_genecompass.tex:964-975, 1213-1220`). E.3 tests cell identity; it says nothing about the
  exercise axis in human cells (`manuscript/motrpac_genecompass.tex:978-979`).
- **The result depends on the registered centring** (above). Centring each set on its own mean makes the origin
  depend on each side's composition: all rat cell types of the tissue with 50 pseudo-cells each, and atlas
  pseudobulks in proportion to how many donors have each type.
- **The human and rat inputs are not filtered alike.** Rat pseudo-cells come from a deconvolution restricted to
  protein-coding genes with ribosomal, mitochondrial, haemoglobin and sex-chromosome genes removed
  (`config/pipeline_config.yaml:371-384`) and lose their non-orthologous genes (12a), leaving 10,474-12,734
  eligible genes; atlas pseudobulks are normalised over every gene in the file and keep all 17,810 genes with a
  token and median (`translation/project_to_atlas.py:64-70`; `AR/<T>/prep_summary.json:8`). This is one concrete
  source of the offset the centring removes; no document quantifies it.
- **Thin centroids.** Kidney and Testis have one donor, so every centroid there is a single pseudobulk
  (`AR/Kidney/prep_summary.json:10-21`; `AR/Testis/prep_summary.json:10-19`). In Spleen the `platelet` centroid is
  one pseudobulk of 22 cells and in Ovary `theca cell` is one pseudobulk
  (`translation/atlas_record_16/Spleen_celltypes.tsv:21`; `translation/atlas_record_16/Ovary_celltypes.tsv:9`).
- **Coarse correspondences inflate chance for some rows** (KIDNEY Immune cells 8 of 10 names; BAT mean chance
  0.200). The null accounts for this, but per-row accuracies are not comparable across rows without the `chance`
  column.
- **Provenance gaps.**
  - `project_to_atlas.py` writes no provenance file in `prep` or `score` (no call to `write_provenance` in the
    file).
  - The nine first-run atlas embedding folders contain only `cell_embeddings.npy`; they have no
    `provenance_embed_cells.json`, because they were embedded on 2026-10-31, before provenance was added to
    `embed_cells.py` (commit `d38b83a`, 2026-11-01) **[auditor read of folder listings and `git log`]**. Spleen
    and Ovary have one. That the first nine were embedded with the v2 final save rests on the job script
    (`slurm/analysis/atlas_e3.slurm:19`), which is gitignored, and on the documents.
  - The atlas download exists only as two local job scripts under the gitignored `slurm/` tree
    (`slurm/analysis/ts_download.slurm`, `ts_download16.slurm`; `.gitignore:37`), and `MANIFEST.txt` lives under
    the gitignored `data/` tree (`.gitignore:2`). No committed file contains the dataset IDs of the first nine
    tissues; the two pre-registrations give the collection ID, and the second gives the Spleen and Ovary dataset
    IDs. The manuscript's statement that "Retrieval and preprocessing scripts ... are released" at the public
    repository (`manuscript/motrpac_genecompass.tex:1407-1410`) therefore does not hold for the atlas retrieval as
    the repository stands.
- **Re-running the first run is not possible with the current script as is.** The scorer skips a tissue only when
  the atlas embeddings are missing (`translation/project_to_atlas.py:113-114`) and then loads the rat-side tissue
  unconditionally (`:115`). With Spleen and Ovary now present on the atlas side, scoring against the 14-tissue
  human root (which has no `spleen` or `ovary` folder) would stop with a file-not-found error. The first run's
  numbers are nonetheless reproduced inside the second (Reproduction).
- **Second-run pre-registration item not in the summary file.** "The verdict over the 11 first-run tissues alone"
  was to be reported (`translation/ATLAS_PROJECTION_16_PREREG.md:36-39`). The results document gives the accuracy
  (0.404) and says "that verdict is the first run's PASS" (`translation/ATLAS_PROJECTION_16_RESULTS.md:16`); a
  separate null and p for those 11 tissues under the second run's draws is not in `summary.txt`.
- **Manuscript inconsistency.** Results say "the eleven atlas tissues with a MoTrPAC counterpart"
  (`manuscript/motrpac_genecompass.tex:956-957`); Methods still say "the nine tissues"
  (`manuscript/motrpac_genecompass.tex:1381-1382`).
- **Stale summary.** `pipeline/README.md:40` gives only the first-run figure (0.404 vs 0.084).
- **Citation and release.** The manuscript cites the 2022 Tabula Sapiens paper
  (`manuscript/motrpac_genecompass.tex:1549-1551`) for a CELLxGENE collection revision dated 2026-06-11. Whether
  that revision corresponds to the 2022 release or a later one is not recorded [not verified in this audit].

### Methods references

- Tabula Sapiens: The Tabula Sapiens Consortium. "The Tabula Sapiens: a multiple-organ, single-cell transcriptomic
  atlas of humans." Science 2022;376(6594):eabl4896 (as in `manuscript/motrpac_genecompass.tex:1549-1551`). Data
  accessed through CELLxGENE collection `e5f58829-1a66-40b5-a624-9046778e74f5`
  (`translation/ATLAS_PROJECTION_PREREG.md:11-12`; the manuscript names the collection ID at
  `manuscript/motrpac_genecompass.tex:1407`); no bibliographic citation for CELLxGENE appears in the code or the
  manuscript.
- GeneCompass: Yang et al., Cell Research 2024;34:830-845 (`manuscript/motrpac_genecompass.tex:1487-1489`).
- Permutation p value with the "+1" correction: implemented at `translation/project_to_atlas.py:157`; no citation
  in the code.

---

## Cross-cutting summary for the reader

1. Stage 12 changes the species encoding of the same rat cells. 91.1% is the share of ortholog-mapped vocabulary
   genes with an identical token in both encodings, not a per-cell measurement; no per-cell overlap is recorded.
2. "9 of 9 preserved" (11 of 11 on 14 tissues) means: human-space AUC at least 0.65 and an uncorrected permutation
   p below 0.05, for blocks selected by a BH-corrected rat-space test, using the same cells, labels, folds and
   permutations. The dose axis is not part of the rule and is not detected in human space for two of the nine.
3. The per-cell cosine (0.974 on the default; 0.979 in the June liver smoke test) is smaller than the cosine
   between different cells of a tissue and has been withdrawn by the project as evidence of identity.
4. E.3 is the only external human check. It passes its pre-registered rule on both runs (0.404 vs 0.084; 0.430 vs
   0.086), depends on the registered centring, covers 120 of 197 blocks, and fails for three hotspots.

## Items I could not determine

- The per-cell fraction of identical tokens between the rat and human encodings (not recorded; computing it needs
  the datasets, which this audit did not load).
- Whether all checkpoint tensors, including the species table, load under `strict=False` (needs the checkpoint).
- The Tabula Sapiens release label behind the CELLxGENE revision of 2026-06-11 (the files themselves are pinned by
  dataset ID and md5).
- The number of atlas cells excluded by the 20-cell rule in the nine first-run tissues.
- Any pre-registration document for the 0.65 / 0.05 status rule and the 0.70 sex floor.
- Whether `pytest -m data` is currently run; `tests/test_v2_outputs.py:62` contradicts the on-disk LUNG counts.
