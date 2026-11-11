# Part 9. Stage 13 (mechanism): in-silico perturbation, differential GRNs, decoy null, rat-human self-consistency, enrichment, CPA

> One part of the methods audit of 2026-11-11; start at [`README.md`](README.md). The main findings of every
> part are collected in [`FINDINGS.md`](FINDINGS.md) and every reference in [`REFERENCES.md`](REFERENCES.md).
> Citations are `path:line` at commit `8249506` unless the text says otherwise.

Audit date 2026-11-11. Repository `/depot/reese18/apps/motrpac-genecompass`, branch `aim2b-perturbation-engine`,
HEAD `8249506` at the time of writing. All paths are relative to the repository root unless they start with `/`.

Conventions used in this section.

- `path:line` cites a file that was opened for this audit. Line numbers are those at HEAD `8249506`.
  `manuscript/motrpac_genecompass.tex` is gitignored and was being edited during the audit; its line numbers are as
  read on 2026-11-11 after 18:05.
- "Auditor's computation" marks a number that is not written in any repository document. Each was obtained
  read-only from an on-disk table that is named with the computation, in a single pass of a few seconds. These
  numbers have not been reviewed by the project and should be re-derived before they are quoted.
- `logs/` and `slurm/` are local-only (not committed); they are cited because they are the only record of the exact
  commands and counts of the production runs.
- Abbreviations: TF, transcription factor; GRN, gene regulatory network; CLS, the model's cell-level token;
  FDR, false-discovery rate; BH, Benjamini-Hochberg; SD, standard deviation; DE, differential expression;
  NES, normalized enrichment score; CPA, compositional perturbation autoencoder; VSMC, vascular smooth muscle
  cells; OPC, oligodendrocyte precursor cells; theta, the BayesPrism cell-type fraction.
- Wording. The code, file names and some docstrings still say "conservation", "conserved" and `conf`. In this
  section the rat-human score is called the self-consistency score, hub genes are called hub TFs, and edges with
  abs(z) >= 2 are called just that. Where an identifier with the older wording is quoted, it is quoted as an
  identifier.

---

## Stage 13 - Mechanism (orchestrator and overview)

### Research role

Stage 13 asks whether the fine-tuned rat GeneCompass model can name the TFs whose influence on other genes changes
with endurance training, in the cell types where the Stage 11 gate found a detectable exercise signal ("hotspots"),
and whether that readout survives re-expressing the same cells as human (Aim 2b and Aim 3b). It also summarises the
Stage 10 per-cell-type dose response as pathways (Aim 2).

The claims that would rest on it are (i) a list of exercise-remodelled TF-to-target edges and hub TFs per cell type,
and (ii) a rat-human agreement score for those lists. They are checked by a pre-registered validation of the
perturbation engine (13b), a pre-registered decoy null for TF specificity (13d), and anchor TFs (13d, 13e).

Verdict so far, in one paragraph. The decoy null is negative: edges with abs(z) >= 2 are as frequent among deletions
of expression-matched non-TF genes as among TF deletions (estimated FDR 1.03), and no TF's hub strength exceeds its
decoys' (0 of 723 TF-network pairs on the default, 0 of 981 on the 14-tissue arm)
(`deconvolution/grn_decoy_record_16/summary.txt:9-19`, `deconvolution/grn_decoy_record/summary.txt:11-23`). The
current model (v2) fails one of the four pre-registered engine checks (`data/results/retrain_v2/c5/module_a/v2/report.md:9`).
The rat-human score is model self-consistency, not cross-species biology (13e). The project's own documents now
report the GRN as a descriptive model readout (`deconvolution/GRN_DECOY_NULL_RESULTS.md:56-58`,
`pipeline/README.md:41`). The pathway enrichment (13f) does not use the model and stands apart from these verdicts.
The CPA dose model (13g) has an installed environment and nothing else.

### Inputs and outputs (default configuration: v2 model, refs4 references, 16 tissues)

| Item | Path | Source of the path |
|---|---|---|
| Model weights | `data/models/rat_genecompass_finetuned_v2/models/rat_phase2_mixed_species/models` | `config/pipeline_config.yaml:364`; resolved by `deconvolution/model_checkpoint.py:28-39` |
| Tokenized rat pseudo-cells | `data/deconvolution/genecompass_input_v2_refs4all_rebuild/<tissue>/dataset` | `config/pipeline_config.yaml:407`; `pipeline/run_stage13.py:186` |
| Tokenized human-space pseudo-cells (Stage 12) | `data/deconvolution/genecompass_input_v2_refs4all_rebuild_human/<tissue>/dataset` | suffix `_human` is appended in code, not read from the config: `pipeline/run_stage13.py:282` |
| Arm labels (control/trained), sex, week | `data/deconvolution/augur_input_v2_refs4all_rebuild/<tissue>/meta.tsv` | `config/pipeline_config.yaml:419`; `pipeline/run_stage13.py:187` |
| Hotspot list with mean fraction | `data/deconvolution/genecompass_input_v2_refs4all_rebuild/pseudobulk_de/de_hotspots.tsv` | `pipeline/run_stage13.py:96-116, 286` |
| TF list | `deconvolution/reference/rat_tf_ensrnog.tsv` (1,049 TFs + header) | `finetune/genecompass/build_grn_pooled.py:142` |
| Token dictionaries, ortholog map | `data/training/ortholog_mappings/{rat_tokens.pickle, rat_human_mouse_tokens.pickle, rat_to_human_mapping.pickle, rat_token_mapping.tsv}` | `finetune/genecompass/perturb_cells.py:51`; `finetune/genecompass/build_grn_human.py:68, 88, 96` |
| Rat-space GRNs (output) | `data/deconvolution/grn_v2_refs4all_rebuild/<tissue>_<celltype>_pooled.tsv` (6 files) | `config/pipeline_config.yaml:420` |
| Human-space GRNs (output) | `data/deconvolution/grn_v2_refs4all_rebuild_human/<...>_pooled.tsv` (6 files) | `config/pipeline_config.yaml:421` |
| Self-consistency tables (output) | `data/deconvolution/grn_v2_refs4all_rebuild_human/conservation/conservation_{per_tf,per_celltype,anchors}.tsv` | `pipeline/run_stage13.py:203-208` |
| Enrichment (output) | `data/deconvolution/genecompass_input_v2_refs4all_rebuild/pseudobulk_de/enrichment/` (151 `gsea__*.tsv`, `enrichment_summary.tsv`, `provenance_run_gene_set_enrichment.json`) | `pipeline/run_stage13.py:209-212` |
| Decoy networks and scores (not an orchestrator step) | `data/deconvolution/grn_v2_refs4all_rebuild_decoy/` (+ `fdr/`); committed copies `deconvolution/grn_decoy_record_16/` and `deconvolution/grn_decoy_record/` | `slurm/analysis/grn_decoy16.slurm:21`; `deconvolution/grn_decoy_fdr.py:26-28` |

Comparison arms, each with its own `grn_dir` / `grn_human_dir`: 14-tissue refs4 (`config/profiles/refs4_14.yaml:10-11`,
8 networks), v2 on the July references (`config/profiles/july_refs.yaml:12-13`, 9 networks), April model
(`config/profiles/april.yaml:25-26`, 11 networks). Superseded folders kept on disk:
`data/deconvolution/grn_uncapped_backup_predeploy_20260715/` (the 2026-07-13 uncapped build, 14 networks),
`grn_backup_predeploy_20260715/` (capped build), `grn_null/` (13 permutation-null tables), `grn_human_null/` (empty).

Counts for the default run (from the job log of the clean rebuild, job 12122211, commit `aee68a3`,
`logs/rebuild_xtissue_12122211.out:1`):

| Network | Hotspot mean theta (`de_hotspots.tsv` line) | Cells control / trained | Rat: TFs present, edges, abs(z)>=2 (log lines) | Human: TFs present, edges, abs(z)>=2 (log lines) |
|---|---|---|---|---|
| BLOOD megakaryocytes and platelets | 0.1454 (4) | 10 / 40 | 110; 235,472; 33,030 (3721, 3725) | 128; 272,224; 56,901 (3835, 3839) |
| BLOOD erythroid cells | 0.6994 (5) | 10 / 40 | 123; 261,457; 31,023 (3739, 3743) | 127; 268,875; 32,656 (3854, 3858) |
| SKMGN tenocytes | 0.0111 (6) | 10 / 40 | 119; 249,356; 23,422 (3757, 3761) | 123; 258,625; 28,533 (3873, 3877) |
| BLOOD classical monocytes | 0.0359 (7) | 10 / 40 | 105; 223,105; 32,349 (3775, 3779) | 121; 257,233; 30,771 (3892, 3896) |
| BLOOD neutrophils | 0.0736 (8) | 10 / 40 | 117; 247,787; 26,374 (3793, 3797) | 124; 261,020; 26,658 (3911, 3915) |
| LUNG VSMC | 0.0703 (9) | 10 / 40 | 149; 309,899; 23,021 (3811, 3815) | 151; 314,645; 22,703 (3930, 3934) |
| Total | | | 723 TF-network pairs; 1,527,076 edges; 169,219 (11.1%) | 774 pairs; 1,632,622 edges; 198,222 (12.1%) |

The rat totals match the manuscript (`manuscript/motrpac_genecompass.tex:1003-1004`). The human totals are sums of
the log lines (auditor's addition). 723 TF-network pairs are 290 distinct TFs (auditor's computation from
`deconvolution/grn_decoy_record_16/hub_vs_decoy.tsv`; also `manuscript/motrpac_genecompass.tex:1003`).

Three of the default's nine hotspots get no network because their mean fraction is under 0.01: SKMVL satellite
cells (0.0023), SKMVL VSMC (0.0022), HYPOTH OPC (0.0005)
(`data/deconvolution/genecompass_input_v2_refs4all_rebuild/pseudobulk_de/de_hotspots.tsv:2-3, 10`).

### Procedure (execution order)

`pipeline/run_stage13.py` runs five steps in sequence (`pipeline/run_stage13.py:174-217`):

1. Rat-space differential GRN, one process per abundant hotspot: `finetune/genecompass/build_grn_pooled.py`
   with `--bootstrap 30 --seed 0 --device cuda` (`pipeline/run_stage13.py:178-198`). Details in 13c.
2. Human-space differential GRN, same loop with `finetune/genecompass/build_grn_human.py`
   (`pipeline/run_stage13.py:199-202`). Details in 13c.
3. Self-consistency score: `finetune/genecompass/compare_grn_conservation.py` with only the three folder arguments,
   so every other option is at its default (`pipeline/run_stage13.py:203-208`). Details in 13e.
4. Gene-set enrichment: `deconvolution/R/run_gene_set_enrichment.sh` with `DE_DIR` exported
   (`pipeline/run_stage13.py:209-212, 315`). Details in 13f.
5. Parenchyma data-anchored validation and dominant-cell-type flags:
   `deconvolution/validate_parenchyma_dataanchored.py` (`pipeline/run_stage13.py:213-216`). This is a check on the
   Stage 10 DE (bulk movers at 8 weeks vs control with BH < 0.05 and abs(log2FC) >= 0.25, minimum 25 non-zero:
   `deconvolution/validate_parenchyma_dataanchored.py:30`), does not use the model, and is not audited in this section.

Which cell types get a network: rows of `de_hotspots.tsv` with `is_hotspot == TRUE` and `mean_fraction >= 0.01`
(`pipeline/run_stage13.py:80, 96-116`). `is_hotspot` is the gate's `q_sup_trained < 0.05` taken from
`deconvolution.hotspots_table` (`config/pipeline_config.yaml:414-417`; `deconvolution/R/run_pseudobulk_de.R:108-117`);
`mean_fraction` is the mean BayesPrism fraction of the cell type over the tissue's samples
(`deconvolution/R/run_pseudobulk_de.R:201`). That is why 6 networks exist on the default (9 hotspots), 8 on the
14-tissue arm (11 hotspots), 9 for v2 on the July references (13 hotspots) and 11 for the April model
(`deconvolution/ALL_TISSUE_RERUN_RESULTS.md:19, 28`; `docs/RETRAIN_V2_RESULTS.md:444-450`).

A step-1/2 item is skipped when its output file already exists unless `--force` is given
(`pipeline/run_stage13.py:227`). `--reuse-grn-from PROFILE` copies or links another arm's networks when the two
arms' hotspot sets are identical (`pipeline/run_stage13.py:145-171`); the default tree did not use it (all 12 tables
are regular files written 2026-11-02 by job 12122211).

The decoy null (13d), the Module A validation (13b), the hub-versus-expression analysis and the abandoned
permutation null are separate jobs and are not steps of the orchestrator.

### Models and software

| Component | Value | Where pinned |
|---|---|---|
| Model | rat GeneCompass v2, `BertForMaskedLM`; 12 layers, 12 heads, hidden size 768, intermediate 3,072, vocabulary 55,275, max 2,048 positions, absolute position embeddings, value channel and four prior-knowledge embeddings on, CLS token on | `data/models/rat_genecompass_finetuned_v2/models/rat_phase2_mixed_species/models/config.json:3, 11, 13, 15, 17-18, 20, 24-30` |
| Model class code | vendored `vendor/GeneCompass` at `66ab82f` (unmodified upstream) | `git submodule status`; `.gitmodules` |
| Loader | `load_model`: prior-knowledge tensors read from the checkpoint state-dict, species embedding resized to 3 (0 human, 1 mouse, 2 rat), `load_state_dict(strict=False)`, `eval()` | `finetune/genecompass/perturb_cells.py:55-72` |
| Forward pass | `model.bert(...)[0]` (last hidden state), `emb_warmup_alpha=1.0`, attention mask = token id != 0, no gradient, float32 | `finetune/genecompass/perturb_cells.py:82-91` |
| Python stack | torch 2.11.0, transformers 4.57.6, datasets 4.8.4, numpy 2.4.3, scipy 1.17.1, pandas 2.3.3 | `requirements.lock.txt:221, 226, 56, 134, 198, 155` |
| R stack (13f) | R 4.4.1, fgsea 1.32.4 (Bioconductor 3.20), msigdbr 26.1.0 | `deconvolution/setup/r_packages.yaml:17-18, 30, 37`; provenance JSON (13f) |
| Hardware | one A100 40 GB GPU; Stage 13 took 8,108.6 s end to end | `slurm/analysis/rebuild_crosstissue.slurm:4-10`; `logs/rebuild_xtissue_12122211.out:4181` |

### Parameters, thresholds and flags (orchestrator level)

| Name | Value | Where set | What it controls | Rationale given |
|---|---|---|---|---|
| `THETA_MIN` | 0.01 | `pipeline/run_stage13.py:80` (constant, not in the config) | minimum mean fraction for a hotspot to get a network | "abundant hotspot cell type" (`pipeline/run_stage13.py:8-9`); no numerical justification stated |
| `--bootstrap` | 30 | `pipeline/run_stage13.py:190` (hard-coded in the command) | bootstrap resamples per network | none stated |
| `--seed` | 0 | `pipeline/run_stage13.py:190` | bootstrap RNG seed | none stated |
| `--device` | cuda | `pipeline/run_stage13.py:190` | GPU | - |
| `--top-edges` | 0 (not passed; script default) | `finetune/genecompass/build_grn_pooled.py:146-151` | no per-TF edge cap | `finetune/genecompass/build_grn_pooled.py:147-151` |
| model | `deconvolution.model_checkpoint` | `config/pipeline_config.yaml:362-364` | explicit weights, "so a new checkpoint cannot change results" | `config/pipeline_config.yaml:362-363` |
| output-name rule | lower-case, `+` to `pos`, space/`-`/`/` to `_`, collapse `__` | `pipeline/run_stage13.py:89-93` | GRN file names | none stated; this is a local re-implementation, separate from `deconvolution/celltype_names.py` |

### Statistical procedures, checks and results

Given per sub-stage below.

### Deviations, known defects and caveats (orchestrator level)

- `pipeline/README.md:27-30` names the default folders as `grn_v2_refs4all/` and so on; the config has pointed at
  the `_refs4all_rebuild` folders since 2026-11-03 (`config/pipeline_config.yaml:399-407, 420-421`). The same README
  row gives "6 GRNs" with "0/981 TFs", which is the 8-network result; the default's own figure is 0/723
  (`pipeline/README.md:41`; `deconvolution/grn_decoy_record_16/summary.txt:14-19`).
- The orchestrator's step names and docstring still read "Cross-species conservation"
  (`pipeline/run_stage13.py:4, 19-21, 203`).
- `pipeline/README.md:26` says there is no Stage 11, but the gate that defines the hotspots is run by
  `pipeline/run_stage11.py` (`slurm/analysis/rebuild_crosstissue.slurm:21`). Flagged only; the gate is in Part 6.
- The GRN tables have no provenance record (no `provenance_*.json` in `grn_v2_refs4all_rebuild*/`); the model path,
  commit and arguments of the production run are recoverable only from the local job log.
- The on-disk default tables were written at commit `aee68a3`, before the tie-order fix; contents are identical
  under the current code, row order of tied abs(z) is not (`deconvolution/REFS4ALL_REBUILD_RESULTS.md:49-57`).

---

## Stage 13a - In-silico perturbation (Module A engine)

### Research role

Provides the primitive on which 13c-13e rest: remove one gene from a cell's input and measure how the model's
representation changes. The question is whether that change carries gene-specific information. 13b and 13d are the
checks; see their verdicts.

### Inputs and outputs

Input: a tokenized dataset with fields `input_ids`, `values`, `cell_type`, `sample` (and others)
(`data/deconvolution/genecompass_input_v2_refs4all_rebuild/skmgn/dataset/dataset_info.json`). Each pseudo-cell is a
sequence of at most 2,048 gene tokens in descending order of value, padded with 0, where value =
log2(1 + normalized count / gene median) after normalizing the pseudo-cell to a target sum of 6,400
(`deconvolution/tokenize_pseudocells.py:8, 57, 61, 82-83`;
`data/deconvolution/genecompass_input_v2_refs4all_rebuild/skmgn/tokenize_summary.json`: `target_sum` 6400, `top_n` 2048).
Stage 9 owns those details.

Output of the stand-alone engine: a TSV with `gene, cell_type, n_cells, mean_cell_shift, median_cell_shift,
top_targets` (`finetune/genecompass/perturb_cells.py:243-247`).

### Procedure

What a perturbation is.

- Deletion (the only mode used in any recorded run). Every position holding the gene's token is removed together
  with its value; the surviving tokens and values are shifted left by one position and a pad (0) is appended, so the
  sequence length stays 2,048. The other genes' values are not renormalized
  (`finetune/genecompass/perturb_cells.py:95-106`; docstring lines 19-21).
- Over-expression (implemented, never used in a recorded run). The gene's value is overwritten with a constant
  (default 8.0); the token order is left unchanged (`finetune/genecompass/perturb_cells.py:109-113, 204`). The
  gene is therefore not moved to the front of the ranked sequence, although the docstring says "a high value -> top
  rank" (`finetune/genecompass/perturb_cells.py:110`).

What is measured (`perturb_gene`, `finetune/genecompass/perturb_cells.py:122-180`).

1. Only cells that contain the gene's token are perturbed (lines 127-131). A gene present in no cell returns
   `n_cells = 0` (lines 129-130).
2. The original and perturbed sequences are embedded in batches (lines 134-149). The species token is 2 (rat) by
   default (line 83).
3. Cell shift = 1 - cosine(CLS original, CLS perturbed), per cell (lines 116-118, 150). It is aggregated over the
   expressing cells as mean and median (lines 171-172).
4. Per-gene target shift (with `--targets`) = 1 - cosine between each surviving gene's last-layer contextual
   embedding before and after, after aligning the two sequences by dropping the deleted position from the original
   and the trailing pad from the perturbed (lines 153-167). Across cells it is the mean over the cells in which the
   target survives; a target is kept only if it is present in at least max(3, 0.5 x n_cells) cells (line 175). The
   25 largest are written as `top_targets` (line 176); the full dictionary is kept in memory for the pilot GRN
   builder (line 179).

The engine is strictly per-cell: the model has no attention across cells
(`finetune/genecompass/perturb_cells.py:21`).

### Models and software

As in the overview table. The engine is a rewrite, not the vendored perturbation class; the docstring lists the
defects of `vendor/GeneCompass/genecompass/perturb_delete_chipseq.py` that motivated it
(`finetune/genecompass/perturb_cells.py:12-21`). The approach is described there as "Geneformer-style".

### Parameters, thresholds and flags

| Name | Value | Where set | What it controls | Rationale given |
|---|---|---|---|---|
| `--mode` | `delete` (default) | `finetune/genecompass/perturb_cells.py:203` | deletion vs over-expression | "prefer `delete` first (unambiguous)" `deconvolution/DOWNSTREAM_BUILD_PLAN.md:97-100` |
| `--overexpress-level` | 8.0 | `finetune/genecompass/perturb_cells.py:204` | value written in over-expression mode | "values ~O(1), max ~7.6" (same line) |
| `--batch-size` | 32 | `finetune/genecompass/perturb_cells.py:209` | cells per forward pass | none stated |
| species token | 2 (rat) | `finetune/genecompass/perturb_cells.py:83` | species embedding | `finetune/genecompass/perturb_cells.py:19` |
| distance | 1 - cosine | `finetune/genecompass/perturb_cells.py:116-118` | effect size for cell and target shifts | none stated |
| target survival filter | present in >= max(3, 0.5 x n_cells) cells | `finetune/genecompass/perturb_cells.py:175` | which targets get a mean shift | none stated |
| top targets reported | 25 | `finetune/genecompass/perturb_cells.py:176` | size of the `top_targets` list | matches the pre-registration (`deconvolution/PERTURB_PREREG.md:20-22`) |
| `--max-cells` | none | `finetune/genecompass/perturb_cells.py:210` | optional truncation (first N cells), not used | - |
| token dictionary | `rat_tokens.pickle`, 22,213 rat genes | `finetune/genecompass/perturb_cells.py:50-51` | gene ID to token | - |
| `strict=False` on weight load | on | `finetune/genecompass/perturb_cells.py:71` | missing/unexpected keys do not raise | mirrors the Stage 9 loader (`finetune/genecompass/perturb_cells.py:56`) |

### Statistical procedures

None in the engine itself; it returns effect sizes. No test, null or seed is involved. The model is in eval mode
(`finetune/genecompass/perturb_cells.py:72`), and the clean rebuild reproduced every GRN edge value exactly
(`deconvolution/REFS4ALL_REBUILD_RESULTS.md:28`).

### Checks, validation and their results

See 13b.

### Deviations, known defects and caveats

- Position effect (auditor's observation from the code; not quantified anywhere in the repository). The model adds
  an absolute position embedding to every gene (`vendor/GeneCompass/genecompass/knowledge_embeddings.py:132, 220-222`;
  `config.json:20`). Because deletion shifts every lower-ranked gene up by one position
  (`finetune/genecompass/perturb_cells.py:98-104`), the input of each such gene changes whichever gene is deleted.
  The target shift therefore contains a component that depends only on where the deleted gene sits in the ranking,
  that is, on its expression. This is consistent with three results reported below: hub strength follows the TF's
  own expression (13c), TFs are indistinguishable from expression-matched non-TF genes (13d), and the top-25 target
  lists of unrelated genes largely coincide (13b).
- The single-deletion cell shift is of order 1e-6 to 1e-5 (Mef2c: 8.15e-06, `deconvolution/PERTURB_PREREG_RESULTS.md:7`),
  which the project describes as near noise in absolute terms (`deconvolution/PERTURB_PREREG_RESULTS.md:14`). The
  model is float32 (`config.json:7`).
- Over-expression: docstring and code disagree (above). No output produced with `--mode overexpress` was found
  under `data/` (looked in `data/deconvolution/perturb_*`, `data/results/retrain_v2/c5/module_a/`).

### Methods references

- GeneCompass: Yang X, Liu G, Feng G, et al. Cell Research 2024;34:830-845 (`manuscript/motrpac_genecompass.tex:1487`).
- In-silico deletion in the Geneformer style (named in `finetune/genecompass/perturb_cells.py:13`): Theodoris CV, et al.
  Nature 2023;618:616-624 (`manuscript/motrpac_genecompass.tex:1477`).
- Network inference plus in-silico perturbation paradigm, cited by the manuscript: Kamimoto K, et al. (CellOracle)
  Nature 2023;614:742-751 (`manuscript/motrpac_genecompass.tex:1541`).

---

## Stage 13b - Module A pre-registered validation (four checks)

### Research role

Asks whether the engine behaves sensibly before it is used to build networks: does deleting a known myogenic TF
(Mef2c) in skeletal-muscle pseudo-cells move the cell more than deleting equally expressed genes, do its top
predicted targets look myogenic, do absent genes give no effect, and does the effect grow with expression.
The pre-registration is `deconvolution/PERTURB_PREREG.md`, "frozen 2026-07-03, before the first real perturbation
run" (line 3; commit `bca3221`, 2026-07-03 12:42).

Verdict. With the April model, 4 of 4 PASS as scored (2026-07-03 and again 2026-10-24). With the current v2
model, 3 of 4: criterion 2 (target enrichment) FAILS, 0 of 25
(`data/results/retrain_v2/c5/module_a/v2/report.md:7-10`). The check has not been run on the current references
(`manuscript/motrpac_genecompass.tex:873-874`). The audit adds that criterion 2 was re-specified after the first
results were visible and that, as scored, it does not isolate a Mef2c-specific signal (see caveats).

### Inputs and outputs

| Run | Model | Cells | Gene panel | Outputs |
|---|---|---|---|---|
| Smoke, 2026-07-03 13:37 (job 11186618) | April, `data/models/rat_genecompass_finetuned/.../models` | SKMVL "Skeletal muscle" (50 cells); BLOOD "Classical monocytes" | Mef2c, Alb, Snap25; Spi1, Cebpa | `data/deconvolution/perturb_smoke/{muscle_mef2c,myeloid_spi1}.tsv`; `slurm/analysis/perturb_smoke.slurm:22-39` |
| Pre-registration run, 2026-07-03 15:18 (job 11187034) | April (same) | SKMVL "Skeletal muscle", 50 cells, pseudo-cells from before the July reference rebuild | 80 genes, `data/deconvolution/perturb_prereg/prereg_genes.txt` | `data/deconvolution/perturb_prereg/results.tsv`; `deconvolution/PERTURB_PREREG_RESULTS.md`; `slurm/analysis/perturb_prereg.slurm:22-27` |
| Rerun, 2026-10-24 (jobs 12013970, 12014307) | April `checkpoint-147941` and v2 | SKMVL "Skeletal myocytes", 50 cells, July-reference pseudo-cells, each model's own tokenization | the 80 genes plus Alb and Snap25 | `data/results/retrain_v2/c5/module_a/{april,v2}/{results.tsv,report.md,report.json}`; `slurm/analysis/v2_c5_module_a.slurm:22-42`; registered in `docs/RETRAIN_V2_PREREGISTRATION.md:165-176` |

The frozen gene list is `deconvolution/reference/perturb_prereg_genes.tsv:2-10`: positive controls Mef2c
(skeletal muscle), Spi1 and Cebpa (classical monocytes); expected Mef2 targets Myog, Ckm, Des, Actn2, each with the
expectation "appears in Mef2c top-25 targets"; negatives Alb and Snap25.

### Procedure

1. `perturb_cells.py --mode delete --targets` over the panel in one cell type (13a).
2. `deconvolution/score_perturb_prereg.py` scores the four criteria from the result table and the dataset:
   - Mean value per gene = mean of the token's value over the cells that contain it; `present` = the set of tokens
     seen in the cell type (lines 43-52).
   - Criterion 1: the matched null is every other panel gene whose mean value is within +-15% of Mef2c's
     (lines 79-80); the statistic is the percentage of those genes whose mean cell shift is below Mef2c's (line 82);
     PASS if > 90 (line 83).
   - Criterion 4: Spearman correlation across panel genes between mean value and mean cell shift; PASS if rho > 0
     (lines 86-88).
   - Criterion 2: hypergeometric upper tail for the overlap between Mef2c's top-25 targets and a curated set of 73
     muscle-contractile, oxidative-metabolism and exercise gene symbols written in the script (lines 33-37), with
     N = number of tokens present in the cell type, K = curated-set tokens present, n = number of top targets,
     k = overlap, p = `hypergeom.sf(k-1, N, K, n)`; PASS if p < 0.05 (lines 91-101).
   - Criterion 3: each negative is in 0 cells, or its shift is below the median of the matched null
     (lines 104-116). It is scored only when both negatives are in the result table; otherwise NOT RUN
     (lines 115-119).

### Models and software

As 13a. scipy `spearmanr` and `hypergeom` (`deconvolution/score_perturb_prereg.py:27`).

### Parameters, thresholds and flags

| Name | Value | Where set | What it controls | Rationale given |
|---|---|---|---|---|
| Criterion 1 threshold | > 90th percentile, one-sided | `deconvolution/PERTURB_PREREG.md:38`; `deconvolution/score_perturb_prereg.py:83` | Mef2c vs matched null | "top decile" `deconvolution/PERTURB_PREREG.md:18-20` |
| Matched-null window | within +-15% of Mef2c's mean value | `deconvolution/score_perturb_prereg.py:79-80` | which panel genes form the null | not in the pre-registration, which says "~50 random genes matched on mean expression value/rank" (`deconvolution/PERTURB_PREREG.md:18-19`) |
| Criterion 2 threshold | hypergeometric p < 0.05 | `deconvolution/PERTURB_PREREG.md:39`; `deconvolution/score_perturb_prereg.py:101` | target enrichment | pre-registered |
| Curated set for criterion 2 | 73 symbols (70 map to rat gene IDs; auditor's count) | `deconvolution/score_perturb_prereg.py:33-37` | the "frozen Mef2/myogenic set" as scored | "fixes the original mis-specification where Myog/Ckm weren't even in the input" (`deconvolution/score_perturb_prereg.py:10-12`) |
| Top targets | 25 | `finetune/genecompass/perturb_cells.py:176` | n in the hypergeometric test | pre-registered |
| Criterion 3 rule | n_cells = 0, or shift < matched-null median | `deconvolution/PERTURB_PREREG.md:40`; `deconvolution/score_perturb_prereg.py:114` | negatives | pre-registered |
| Criterion 4 threshold | Spearman rho > 0 | `deconvolution/PERTURB_PREREG.md:41`; `deconvolution/score_perturb_prereg.py:88` | self-consistency | pre-registered |
| Panel | 80 genes: Mef2c + expression-matched + value-spanning | `data/deconvolution/perturb_prereg/prereg_genes.txt` (written 2026-07-03 15:11) | genes deleted | how the 80 were chosen: NOT FOUND (no script or seed; only the comment in `slurm/analysis/perturb_prereg.slurm:15-17`) |
| Scorer defaults | `--dataset data/deconvolution/genecompass_input/skmvl/dataset`, `--cell-type "Skeletal muscle"`, `--out deconvolution/PERTURB_PREREG_RESULTS.md` | `deconvolution/score_perturb_prereg.py:57-60` | April-era defaults; running with defaults would overwrite the committed results file | none |

### Statistical procedures

- Criterion 1: empirical percentile of one gene among 26 to 52 comparison genes. No p-value, no multiplicity
  control (single pre-specified gene). One-sided.
- Criterion 2: one hypergeometric test; the null is 25 genes drawn at random from the genes present in the cell
  type. Units are genes. This null does not account for the fact that every deletion yields a similar top-25 list
  (see caveats).
- Criterion 4: Spearman rho with its asymptotic two-sided p-value over 77 to 80 panel genes.
- Replication unit: one tissue, one cell type, 50 pseudo-cells; cell-level variation enters only through the means.
- Pre-registered: the four criteria, their thresholds and a miss-reading ladder (`deconvolution/PERTURB_PREREG.md:35-53`).

### Checks, validation and their results

| Criterion (pass rule) | April, 2026-07-03, pre-rebuild cells | April, 2026-10-24, July-reference cells | v2, 2026-10-24, July-reference cells |
|---|---|---|---|
| 1. Mef2c cell-shift percentile among matched genes (> 90) | 98th of 52, shift 8.15e-06 - PASS | 96.3 of 27, shift 7.15e-06 - PASS | 100 of 26, shift 5.94e-06 - PASS |
| 2. Mef2c top-25 targets in the curated set (p < 0.05) | 7/25, K = 48 of N = 2,549, p = 2.0e-07 - PASS | 4/25, K = 46 of 2,612, p = 8.1e-04 - PASS | 0/25, K = 48 of 2,597, p = 1.0 - FAIL |
| 3. Alb, Snap25 absent or below the null median | "n_cells=0 (from the smoke run)" - PASS | both 0 cells - PASS | both 0 cells - PASS |
| 4. Spearman(shift, mean value) (> 0) | rho = 0.70, p = 8.3e-13 - PASS | rho = 0.741, p = 1.3e-14 - PASS | rho = 0.759, p = 1.3e-15 - PASS |
| Source | `deconvolution/PERTURB_PREREG_RESULTS.md:3, 7-10` | `data/results/retrain_v2/c5/module_a/april/report.json` | `data/results/retrain_v2/c5/module_a/v2/report.md:3, 7-10`; `report.json` |

The same three columns are tabulated in `docs/RETRAIN_V2_RESULTS.md:350-357, 364`. In the October runs 77 of the 80
panel genes are present in the cells (`docs/RETRAIN_V2_RESULTS.md:347-348`).

Not scored: the pre-registered myeloid replication (Spi1 and Cebpa in classical monocytes: cell shift in the top
decile of a matched null, targets enriched for myeloid genes; "report it" - `deconvolution/PERTURB_PREREG.md:23-26, 42`).
Only the smoke run exists (Spi1 mean cell shift 4.73e-06 in 50 cells; Cebpa 4.64e-07 in 38 cells,
`data/deconvolution/perturb_smoke/myeloid_spi1.tsv:2-3`); no matched null was run and no result is reported in
`deconvolution/PERTURB_PREREG_RESULTS.md`.

### Deviations, known defects and caveats

1. Criterion 2 was re-specified after the first results were visible. Timeline on 2026-07-03: pre-registration
   committed 12:42 (`bca3221`); smoke run output, which already contains Mef2c's top-25 targets, written 13:37
   (`data/deconvolution/perturb_smoke/muscle_mef2c.tsv:2`); curated 73-gene set committed in the scorer 15:16
   (`ffbebff`); pre-registration run written 15:18; results committed 15:23 (`b162409`). The scorer says the set
   includes "the muscle/mito genes the smoke surfaced" (`deconvolution/score_perturb_prereg.py:31-32`). The project
   records this (`docs/RETRAIN_V2_RESULTS.md:359-363`; `manuscript/motrpac_genecompass.tex:881-884`), and
   notes that three of April's four October hits are genes that run surfaced.
2. By the frozen gene file, the expectation was that Myog, Ckm, Des and Actn2 appear in Mef2c's top-25
   (`deconvolution/reference/perturb_prereg_genes.tsv:5-8`). None of the four appears in the top-25 in any of the
   four result tables (auditor's computation: search of the Mef2c row of
   `data/deconvolution/perturb_smoke/muscle_mef2c.tsv`, `data/deconvolution/perturb_prereg/results.tsv`,
   `data/results/retrain_v2/c5/module_a/{april,v2}/results.tsv` for the four gene IDs; 0 matches each).
3. The top-25 target lists are largely the same whichever gene is deleted (auditor's computation from the
   `top_targets` column of the three result tables). Overlap between Mef2c's top-25 and each other panel gene's
   top-25: median 17 of 25 (range 7-22) in the July run, 18 (5-23) for April in October, 19 (6-25) for v2. Applying
   the criterion-2 count to every panel gene: in the July run Mef2c has 7 hits and the other 79 genes have 2 to 5
   (median 4), so nearly every deletion would pass the hypergeometric test against a random-gene null (expected
   hits under that null: 25 x 48 / 2,549, about 0.5); for April in October Mef2c has 4 and three other genes also
   have 4 (median 3); for v2 all 77 genes have 0 or 1. The hypergeometric p-value therefore mostly reflects a
   gene-independent set of strongly shifting targets, and a null built from the other panel genes would be the
   relevant comparison. No such comparison is in the repository.
4. The 2026-07-03 numbers cannot be regenerated from the on-disk inputs: the dataset path now holds the
   July-reference pseudo-cells with a different cell-type label (`docs/RETRAIN_V2_PREREGISTRATION.md:166-168`).
5. `deconvolution/PERTURB_PREREG_RESULTS.md` is the April result only ("ALL PASS", line 14) and carries no note that
   the current model fails criterion 2. Its criterion-3 row reads "n_cells=0 (from the smoke run)" (line 10): at
   that time the scorer printed PASS without scoring; this was fixed on 2026-10-24 (`af96ebb`;
   `deconvolution/score_perturb_prereg.py:13-15, 104-119`). `docs/OPEN_FRONTIER.md:139` does state the v2 failure.
   No machine-readable result (`PERTURB_PREREG_RESULTS.json`) exists for the July run.
6. The check was not repeated on the refs4 or 16-tissue references (`manuscript/motrpac_genecompass.tex:873-874`).
   The project's stated reason is that it tests the model, not the reference
   (`manuscript/motrpac_genecompass.tex:871`).
7. Criterion 1 compares Mef2c with 26-52 genes in one cell type; whether TFs in general move the embedding more
   than matched genes is the decoy test of 13d, which is negative.
8. The pre-registration text labels the gene IDs "rel-113" (`deconvolution/PERTURB_PREREG.md:6`); the ID-release
   question belongs to the Stage 1-3 audit and is only flagged here.

### Methods references

Hypergeometric test and Spearman correlation as implemented in scipy 1.17.1 (`requirements.lock.txt:198`). No
method paper is cited in the code or docs for these.

---

## Stage 13c - Differential GRN construction (Module B), rat and human space

### Research role

For each abundant hotspot, estimate for every TF and every other gene how much more (or less) deleting the TF shifts
that gene's representation in trained animals than in controls. The intended reading was "exercise-remodelled
TF-to-target edges" and a ranking of hub TFs. The check is the decoy null (13d). Verdict: the tables exist and are
reproducible, but they carry no TF-specific signal, and the audit finds that a large part of the hub statistic is
not a trained-minus-control difference at all (caveat 1 below).

### Inputs and outputs

Inputs as in the overview. Output columns: rat space `tf, tf_symbol, target, target_symbol, delta, boot_sd, z`
(`finetune/genecompass/build_grn_pooled.py:283`); human space `tf, tf_symbol, tf_human_ensg, target_human_ensg,
delta, boot_sd, z` (`finetune/genecompass/build_grn_human.py:194`). `delta` and `boot_sd` are rounded to 6 decimals
and `z` to 2 (`finetune/genecompass/build_grn_pooled.py:276`). Rows are sorted by descending abs(z), ties by
(tf, target) (line 278). Counts are in the overview table.

### Procedure

TF list. `deconvolution/reference/rat_tf_ensrnog.tsv` has 1,049 TFs (1,050 lines with header). It is produced by
`finetune/genecompass/build_rat_tf_list.py`: a list of TF gene symbols is matched, case-insensitively, against the
rat symbol column of `rat_token_mapping.tsv`, keeping genes that have a token
(`finetune/genecompass/build_rat_tf_list.py:52, 59-70`). The only symbol file on disk is
`data/references/tf/TF_names_v_1.01.txt` (1,639 lines), which the script's docstring identifies as the human TF list
of Lambert et al. 2018 (`finetune/genecompass/build_rat_tf_list.py:12-13, 20`). By tier, 909 of the 1,049 are
tri-species ortholog tokens, 23 human one-to-one, 91 mouse one-to-one, 14 multi-mapping and 12 new rat-only tokens;
941 have a human ortholog with a token (auditor's count from the `tier` and `human_ortholog` columns;
the 941 is also printed by the run, `logs/rebuild_xtissue_12122211.out:3824`). All 1,049 are in the rat vocabulary
(`logs/rebuild_xtissue_12122211.out:3712`: "TFs=1049").

The figure "981 TFs" that appears in the documents is not the size of this list. It is the number of TF-network
pairs in the eight 14-tissue networks (105 + 123 + 110 + 117 + 140 + 118 + 149 + 119,
`deconvolution/grn_decoy_record/fdr_per_network.tsv:2-9`), which are 310 distinct TFs (auditor's computation from
`deconvolution/grn_decoy_record/hub_vs_decoy.tsv`). Likewise 723 pairs are 290 distinct TFs on the default.

Arms. `load_arms` reads the tissue's dataset in order and keeps the rows of the cell type whose `(sample,
cell_type)` has label `control` or `trained` in `meta.tsv` (`finetune/genecompass/build_grn.py:39-61`). Control is
week 0 and trained is weeks 1, 2, 4 and 8 pooled (`deconvolution/augur_prep.py:47, 104-105`;
`finetune/genecompass/build_grn_pooled.py:17-18`). In every default network that is 10 control and 40 trained
pseudo-cells, 5 per sex per week (auditor's tabulation of
`data/deconvolution/augur_input_v2_refs4all_rebuild/{blood,lung,skmgn}/meta.tsv`; log lines in the overview table).
One pseudo-cell is one animal's deconvolved profile for that cell type. Sex and week are not used by the builder.
This pooling of the four training durations is what the code calls "dose-pooling".

Steps for one network (`finetune/genecompass/build_grn_pooled.py:170-287`):

1. Pseudobulk each arm. For every token, sum its value over the arm's cells (absent = 0), divide by the number of
   cells, and keep the 2,048 tokens with the largest mean, ordered by descending mean (lines 46-59). Each arm is
   thus one synthetic profile, and the two arms' token sets are selected independently.
2. For every TF whose token is in at least one arm's profile, delete it from each profile that contains it and
   compute every surviving gene's target shift, 1 - cosine of its contextual embedding before and after (two forward
   passes of batch size 1 per profile and TF, original and deleted; lines 62-78, 225-229). A profile that lacks the
   TF returns an empty result (lines 66-67).
3. Edge value: delta(TF, target) = shift in the trained profile - shift in the control profile, where a target
   missing from one side contributes 0 for that side; the edge set is the union of the two sides
   (`diff`, lines 81-82). All edges are kept (`--top-edges 0`, lines 232-235).
4. Bootstrap, B = 30: draw 10 control cells and 40 trained cells with replacement (one index vector per arm, whole
   cells), rebuild both pseudobulk profiles, repeat steps 2-3 for every TF, and accumulate the sum and the sum of
   squares of each edge's delta; an edge absent from a replicate contributes 0 (lines 245-262).
5. boot_sd = population SD of the 30 replicate deltas, z = point delta / boot_sd, and z = 0 when boot_sd = 0
   (lines 266-275).

Human space (`finetune/genecompass/build_grn_human.py`) repeats this with three changes stated in its docstring
(lines 9-13): the dataset is the Stage 12 human-space re-expression of the same pseudo-cells; each rat TF is
replaced by its human ortholog's token through `rat_to_human_mapping.pickle` and the merged token dictionary, TFs
without one being skipped (lines 88-106; 941 of 1,049 remain, log line 3824); the species token is 0
(lines 40, 49-50). Targets are reported as human gene IDs and tokens without a human ID are dropped (lines 58, 91).
The seed and the order of RNG calls are the same as in rat space (lines 136, 161-162 against
`build_grn_pooled.py:214, 248-249`), so the 30 resamples pick the same animals in both spaces.

Hub statistic. A TF's hub strength is the sum of abs(delta) over all of its edges, with no filter on z. It is not
computed by the builders; it is computed where needed: `deconvolution/grn_decoy_fdr.py:53`,
`data/results/retrain_v2/downstream/grn_compare/hub_expr.py:38-39`. No hub table is written for the default arm
other than `hub_vs_decoy.tsv`.

The removed edge cap. Until 2026-07-13 the builders kept only the 40 largest abs(delta) edges per TF. The cap was
removed (commit `0239789`; default 0, `finetune/genecompass/build_grn_pooled.py:146-151`) because it selected edges
on the statistic that z then tests and fixed every TF at the same number of candidates. The same commit fixed the
bootstrap, which had drawn token IDs and values with independent index vectors
(`finetune/genecompass/build_grn_pooled.py:242-244`). The uncapped build of that date had 3,478,564 rat-space edges
over 14 networks and 389 TFs, 398,586 of them with abs(z) >= 2 (`docs/OPEN_FRONTIER.md:141`; confirmed by the
auditor from `data/deconvolution/grn_uncapped_backup_predeploy_20260715/hub_strength.tsv`). That "3.48 million"
figure belongs to the April model on the pre-July references and is superseded: the default has 1,527,076
rat-space edges in 6 networks. The pilot per-cell builder `finetune/genecompass/build_grn.py` (cap 25, minimum 5
expressing cells per arm, lines 74-75) is kept only for `load_arms`.

### Models and software

As in the overview. One forward pass per (profile, TF, original-or-deleted); about 10 minutes per network on an
A100 (`logs/rebuild_xtissue_12122211.out:3726, 3744, 3762, 3780, 3798, 3816`).

### Parameters, thresholds and flags

| Name | Value | Where set | What it controls | Rationale given |
|---|---|---|---|---|
| `--tf-file` | `deconvolution/reference/rat_tf_ensrnog.tsv` | `finetune/genecompass/build_grn_pooled.py:142` | genes to delete | `deconvolution/DOWNSTREAM_BUILD_PLAN.md:111-116` |
| TF "expressed" rule | token in the top-2,048 pseudobulk profile of at least one arm | `finetune/genecompass/build_grn_pooled.py:66-67, 228-229` | which TFs are perturbed (105-149 of 1,049 per network) | none stated |
| `L` | 2,048 | `finetune/genecompass/build_grn_pooled.py:43` (constant) | profile length | model input length |
| `VOCAB` | 55,275 | `finetune/genecompass/build_grn_pooled.py:42` (constant) | accumulator size; equals the model's vocabulary (`config.json:30`) | none; not read from the model |
| `--bootstrap` | 30 | `finetune/genecompass/build_grn_pooled.py:144`; `pipeline/run_stage13.py:190` | resamples | none stated |
| `--seed` | 0 | `finetune/genecompass/build_grn_pooled.py:145, 214` | `numpy.random.default_rng` for the resamples | none stated |
| `--top-edges` | 0 (no cap) | `finetune/genecompass/build_grn_pooled.py:146-151`; `build_grn_human.py:72-74` | edges kept per TF | lines 147-151 |
| SD estimator | population SD (divide by B) over replicates, missing = 0 | `finetune/genecompass/build_grn_pooled.py:256-262, 269-271` | denominator of z | "matches np.std(ddof=0)" (line 270) |
| z when SD = 0 | 0 | `finetune/genecompass/build_grn_pooled.py:275` | undefined ratio | none stated |
| abs(z) >= 2 | reporting threshold only (count printed) | `finetune/genecompass/build_grn_pooled.py:279, 286` | no filtering of the output | see 13d |
| rounding | delta, boot_sd 6 decimals; z 2 decimals | `finetune/genecompass/build_grn_pooled.py:276` | stored precision | none stated |
| `--max-tfs` | none | `finetune/genecompass/build_grn_pooled.py:153` | debug truncation, unused | - |
| `--permute-seed` | -1 (off) | `finetune/genecompass/build_grn_pooled.py:154-160` | abandoned permutation null (13d) | - |
| `--decoy-seed` | -1 (off) | `finetune/genecompass/build_grn_pooled.py:161-167` | decoy null (13d) | - |
| `--ortholog-map` (human) | `data/training/ortholog_mappings/rat_to_human_mapping.pickle` | `finetune/genecompass/build_grn_human.py:68` | rat TF to human gene | - |
| species token | 2 rat, 0 human | `finetune/genecompass/perturb_cells.py:83`; `build_grn_human.py:40` | species embedding | - |

### Statistical procedures

For TF g, target k, arm a: s_a(g, k) = 1 - cos(e_k in the arm-a profile, e_k in the arm-a profile with g deleted),
with s_a(g, k) = 0 if k or g is not in that profile.

- delta(g, k) = s_trained(g, k) - s_control(g, k).
- boot_sd(g, k) = sqrt( mean_b delta_b^2 - (mean_b delta_b)^2 ), b = 1..30, replicate edges missing = 0.
- z(g, k) = delta(g, k) / boot_sd(g, k); 0 if boot_sd = 0.
- hub(g) = sum over k of abs(delta(g, k)).

Unit of replication: the pseudo-cell (animal), resampled within arm, not stratified by sex or by training week.
Seeds: 0 for the resamples. No test is applied in the builder, and none is calibrated: z is the point estimate on
the full arms divided by the spread of 30 replicate estimates, each of which re-selects its own top-2,048 genes;
it is not centred on a null, has no reference distribution, and no multiplicity control is applied to the roughly
250,000 edges per network. Nothing in 13c is pre-registered.

Why no permutation null exists for z. The arm label is a deterministic function of training week (control is week
0; trained is weeks 1/2/4/8), so a permutation within week is the identity and any permutation that changes the
label mixes training weeks into the sham control arm; the permuted and observed z are then not identically
distributed under the null (`deconvolution/GRN_NULL_WHY_ABANDONED.md:10-28`). See 13d.

Why hub strength is the sum of abs(delta) and not a count of abs(z) >= 2 edges. The project's stated reason:
z is a ratio that grows without bound where both delta and its bootstrap SD are near zero, so counting abs(z) >= 2
edges rewards TFs whose deletion changes almost nothing; on the 2026-07-13 uncapped April networks the TFs the
project expected to rank high sat at the 33rd percentile by count and Nr4a1 at the 17th
(`docs/OPEN_FRONTIER.md:141`; `manuscript/motrpac_genecompass.tex:1007-1011`). Auditor's recomputation from
`data/deconvolution/grn_uncapped_backup_predeploy_20260715/hub_strength.tsv` (1,655 rows; within-cell-type
percentile ranks): for Mef2c, Myf6, Myod1, Spi1 and Cebpa (19 rows) the median percentile is 33.2 by count and 78.2
by sum; Nr4a1 (8 rows) median 17.4 vs 78.9; Mef2c (6 rows) 38.1 vs 91.4; the correlation between the count and the
largest abs(delta) is negative in 9 of 14 cell types. These reproduce the documented figures. On the default networks 42.3% of the abs(z) >= 2 edges have a stored boot_sd <= 1e-5 and 22.1% have a
stored abs(delta) <= 1e-6 (auditor's computation over the six `grn_v2_refs4all_rebuild/*_pooled.tsv`). The
mechanism is therefore confirmed in the current tables. Three qualifications: (i) the comparison was made on the
April model's July-13 networks and has not been repeated for v2 or the default; (ii) no script that produces
`hub_strength.tsv` or the percentile comparison is in the repository (NOT FOUND; `hub_strength` appears in code only
in `deconvolution/grn_decoy_fdr.py`); (iii) the statistic was chosen after seeing which one ranked the expected TFs
high, and the later decoy null shows that the chosen statistic does not separate TFs from matched non-TF genes
either. "Inverts the biology" should be read as "ranks the expected TFs low", not as evidence that the sum recovers
biology.

Hub strength follows the TF's own expression. Spearman correlation between hub strength and the TF token's
pseudobulk value (mean of the two arms): median 0.831 (rat space) and 0.832 (human space) over the April model's
11 networks; 0.790 and 0.836 over v2's 9 July-reference networks; range 0.626-0.953
(`data/results/retrain_v2/downstream/grn_compare/grn_hub_value.txt:2-48`; script `hub_expr.py:30-43`;
`docs/RETRAIN_V2_RESULTS.md:477-479`). On identical human-space tokens the two models rank hubs at Spearman
0.991-0.997 while their edge deltas agree at 0.414-0.565 (`docs/RETRAIN_V2_RESULTS.md:472-474`). This analysis has
not been rerun on the default's six networks (NOT FOUND for refs4 or the 16-tissue arm).

### Checks, validation and their results

| Check | Result | Pass rule | File |
|---|---|---|---|
| Clean rebuild reproduces the networks | 6 rat + 6 human tables: same edge sets, numeric difference 0 on every edge; only the order of tied rows differed | pre-registered "edges equal" | `deconvolution/REFS4ALL_REBUILD_RESULTS.md:28, 49-57` |
| Six default networks equal six of the eight 14-tissue networks | identical after sorting | reported | `deconvolution/ALL_TISSUE_RERUN_RESULTS.md:63-65` |
| Uncapped tables contain the capped ones | delta identical for all 66,200 capped edges | reported | `docs/OPEN_FRONTIER.md:141` (capped tables: `data/deconvolution/grn_backup_predeploy_20260715/`, 66,200 rows, auditor's count) |
| April vs v2 on identical human-space inputs | same TFs and same edge sets | control | `docs/RETRAIN_V2_RESULTS.md:439-440` |
| TF specificity | negative | pre-registered | 13d |

### Deviations, known defects and caveats

1. One-arm-only genes: the docstring says pseudobulking makes "both arms carry the SAME gene set"
   (`finetune/genecompass/build_grn_pooled.py:10-12`). The code does not ensure this: each arm keeps its own top
   2,048 tokens (lines 46-59) and `diff` fills a missing side with 0 (lines 81-82). The project's code review of 2026-10-04 flagged it (the review's notes are not in the repository), and it is
   still present in the code and in the default output.
   Auditor's computation from the six default rat-space tables:
   - Every TF present in both arms has the same number of edges, and it exceeds the 2,047 a common gene set would
     give: 2,133 (classical monocytes), 2,129 (erythroid), 2,149 (megakaryocytes), 2,121 (neutrophils), 2,081 (LUNG
     VSMC), 2,098 (tenocytes). The excess is the number of genes that are in only one arm's profile: 86, 82, 102, 74,
     34 and 51 genes per arm. For these targets delta is one arm's raw shift (with the other side set to 0), not a
     trained-minus-control difference.
   - 5 to 10 TFs per network are themselves in only one arm's profile (exactly 2,047 edges, all of one sign):
     10, 5, 9, 5, 5 and 6. For these TFs every edge is a raw one-arm shift. `grn_hub_value.txt` records the same
     thing for the earlier arms (column `tfs_in_both_arms`, for example 108 of 114, line 2).
   - In SKMGN tenocytes, among the 113 TFs present in both arms, one-arm-only targets are 4.9% of the edges
     (11,526 of 237,074) but carry a median 36.5% of each TF's hub strength (range 20.4-44.1%), 19.1% of the
     abs(z) >= 2 edges (3,840 of 20,091) and 64.5% of each TF's 20 largest abs(delta) edges. Median abs(delta) is
     1.25e-4 for one-arm-only edges against 1e-6 for the others.
   - In BLOOD classical monocytes (95 TFs in both arms): 8.1% of edges, median 53.1% of hub strength (38.3-63.4%),
     26.5% of the abs(z) >= 2 edges (6,723 of 25,391), 77.6% of the top-20 edges.
   Method: genes exclusive to an arm were identified from the target lists of the TFs that are in one arm only;
   two networks were examined. The decoy networks are built by the same code, so the negative result of 13d is not
   an artefact of this defect; but hub strength, the share of abs(z) >= 2 edges and the 13e correlation all include
   this non-differential component, and the documented definition of delta
   (`manuscript/motrpac_genecompass.tex:1370-1371`; `docs/RETRAIN_V2_RESULTS.md:429-430`) does not describe it.
2. Numerical resolution. 48.4% of all default rat-space edges have a stored abs(delta) <= 1e-6 and 32.0% are stored
   as exactly 0; 19.8% have a stored boot_sd of 0; 53,000 edges have z = 0 (auditor's computation, same six files).
   The stored delta is quantized at 1e-6 while z is computed before rounding, so the table contains edges with
   delta 0.0 and a non-zero z.
3. z is not a calibrated statistic (above), and with 10 control animals the bootstrap SD is itself coarse. The
   bootstrap is not stratified by sex or week, so trained-arm replicates vary in their sex and week mix.
4. The position effect of 13a applies to every edge.
5. Residual wording in code: "Trustworthy edges have |z| >= 2" (`finetune/genecompass/build_grn_pooled.py:14-15`)
   and the variable `conf` (line 279), after the registered reading of the decoy null required that such edges no
   longer be described that way (`deconvolution/GRN_DECOY_NULL_PREREG.md:31-32`). The `--permute-seed` help text
   still presents the permutation null as the calibration of the threshold (lines 154-160).
6. TF list provenance. The command that produced `rat_tf_ensrnog.tsv` is not recorded and the file has no
   provenance; whether `--expressed-dir` was used cannot be determined (NOT FOUND). `docs/OPEN_FRONTIER.md:140` and
   `deconvolution/DOWNSTREAM_BUILD_PLAN.md:113` name AnimalTFDB as the source and "intersect with expressed genes",
   but the only symbol file on disk is the human list `TF_names_v_1.01.txt`, matched by upper-cased rat symbol
   (`finetune/genecompass/build_rat_tf_list.py:56-57`). A rat TF whose symbol differs from the human symbol is
   missed, and a non-TF rat gene sharing a symbol with a human TF would be included. The script's docstring labels
   the IDs "rel-113" (line 6).
7. Hard-coded values that bypass the config: `THETA_MIN`, the bootstrap count and seed (orchestrator table), `L` and
   `VOCAB`, and the `_human` folder suffix.
8. The project documents call the hub lists "expression-driven model readouts" and name no hub on GRN evidence
   (`deconvolution/GRN_DECOY_NULL_RESULTS.md:56-58`). The heading of `docs/RETRAIN_V2_RESULTS.md:423` and the
   Module B row of `docs/OPEN_FRONTIER.md:141` ("Top hubs = Tfeb/Mef2c/Epas1/Myf6/Nr4a1") predate that reading and
   are qualified only by the notes placed beside them (`docs/RETRAIN_V2_RESULTS.md:6-7`).

### Methods references

- Bootstrap resampling: no citation in code or docs. Efron B. Bootstrap methods: another look at the jackknife.
  Ann Statist 1979;7:1-26 [verified against Crossref; see REFERENCES.md].
- TF list: Lambert SA, et al. The Human Transcription Factors. Cell 2018 (title and year as in
  `finetune/genecompass/build_rat_tf_list.py:12`; volume and pages 172:650-665 [verified against Crossref; see REFERENCES.md]); not in the manuscript
  bibliography.

---

## Stage 13d - Decoy null for TF specificity, and the abandoned permutation null

### Research role

Asks whether abs(z) >= 2 is an error rate or only a ranking, and whether any TF's hub strength exceeds that of a
random gene of the same expression. Pre-registered twice: `deconvolution/GRN_DECOY_NULL_PREREG.md` (commit
`6ac6676`, 2026-10-31 17:26; the decoy job started 17:41, `logs/grn_decoy_12103791.out:1`) for the eight 14-tissue
networks, and `deconvolution/GRN_DECOY_NULL_16_PREREG.md` (commit `780b149`, 2026-11-04 23:07; job started 23:08 at
that commit, `logs/grn_decoy16_12192357.out:1`) for the default's six.

Verdict: negative at both levels, on both runs. This is the principal negative result of Stage 13.

### Inputs and outputs

| Run | TF networks | Decoy networks | Scores | Committed record |
|---|---|---|---|---|
| 14-tissue refs4, 8 networks, job 12103791 | `data/deconvolution/grn_v2_refs4/` | `data/deconvolution/grn_v2_refs4_decoy/` | `<decoy dir>/fdr/` | `deconvolution/grn_decoy_record/` |
| 16-tissue default, 6 networks, job 12192357 | `data/deconvolution/grn_v2_refs4all_rebuild/` | `data/deconvolution/grn_v2_refs4all_rebuild_decoy/` | `<decoy dir>/fdr/` | `deconvolution/grn_decoy_record_16/` |

The committed records equal the on-disk scorer output (byte comparison of `hub_vs_decoy.tsv` for the 16-tissue run
and of `summary.txt` for the 8-network run; auditor's check). Decoy tables have one more column, `decoy_for`, the TF
each decoy replaces (`finetune/genecompass/build_grn_pooled.py:283`).

### Procedure

Decoy construction (`match_decoys`, `finetune/genecompass/build_grn_pooled.py:107-133`, called at lines 219-222):

1. Expression of a token = the mean of its value in the control and trained pseudobulk profiles, absent = 0
   (lines 112-116).
2. The TFs to replace are those present in either profile (line 119), the same set the TF network uses.
3. The candidate pool is every token present in either profile that is not in the TF list (all 1,049 TFs excluded)
   and has a rat gene ID (lines 117-120).
4. TFs are visited in a random order drawn from `default_rng(decoy_seed)`; each takes the unused pool gene nearest
   in log1p(expression) (lines 121-131). Matching is without replacement, one decoy per TF.
5. The decoy list replaces the TF list and the build of 13c runs unchanged: same labels, cells, model, bootstrap
   seed and resamples (the bootstrap generator is created from `--seed` before and independently of the matching,
   lines 214, 123).

Commands: `--bootstrap 30 --seed 0 --device cuda --decoy-seed 0` for each network
(`slurm/analysis/grn_decoy.slurm:26-27`; `slurm/analysis/grn_decoy16.slurm:26-27`), then
`deconvolution/grn_decoy_fdr.py` with defaults.

Scoring (`deconvolution/grn_decoy_fdr.py`):

- Edge level. For each threshold t on the grid 2.0, 2.5, ..., 50 (line 30): FDR(t) = (decoy edges with abs(z) >= t /
  all decoy edges) / (TF edges with abs(z) >= t / all TF edges), per network at t = 2 (lines 46-52) and pooled over
  networks at every t (lines 61-63). t* = the smallest grid t with pooled FDR <= 0.10 and at least 100 TF edges
  (lines 64-65).
- Hub level. Per network, each TF's sum of abs(delta) is compared with the decoys' sums in the same network:
  p = (1 + number of decoys >= TF) / (1 + number of decoys) (lines 53-55); BH within network (lines 33-36, 56);
  count of TFs with q < 0.10 and the q of the anchors Mef2c, Nr4a1, Myf6, Tfeb, Epas1 where present
  (lines 29, 79-82).

Registered reading, fixed before the decoys were built (`deconvolution/GRN_DECOY_NULL_PREREG.md:26-36`):
FDR(2) <= 0.10, report the edges at that error rate; 0.10 < FDR(2) <= 0.50, abs(z) stays a ranking and t* is
reported; FDR(2) > 0.50, the edges are not distinguishable from random-gene deletions and the confidence label the
documents had attached to them is to be removed everywhere. The hub reading does not depend on the edge reading.
"No other threshold, network subset or matching rule will be tried after the results are seen" (lines 38-39). The
reading is implemented in the scorer (`deconvolution/grn_decoy_fdr.py:71-73`).

### Models and software

As 13c. Run times 1 h 24 min and 57 min on an A100 (`deconvolution/GRN_DECOY_NULL_RESULTS.md:3-4`;
`deconvolution/GRN_DECOY_NULL_16_RESULTS.md:4`).

### Parameters, thresholds and flags

| Name | Value | Where set | What it controls | Rationale given |
|---|---|---|---|---|
| `--decoy-seed` | 0 | `slurm/analysis/grn_decoy.slurm:27`; `grn_decoy16.slurm:27`; registered `GRN_DECOY_NULL_PREREG.md:11` | order in which TFs pick decoys | single seed; a second seed "may be run only to report seed variability" (`GRN_DECOY_NULL_PREREG.md:38-39`); not run (`GRN_DECOY_NULL_RESULTS.md:60`) |
| matching variable | log1p of the mean of the two arm-profile values | `finetune/genecompass/build_grn_pooled.py:112-116, 121, 127` | what "same expression" means | hub strength follows the TF's own expression (`GRN_DECOY_NULL_PREREG.md:11-12`) |
| decoys per TF | 1, without replacement | `finetune/genecompass/build_grn_pooled.py:122-131` | size of the null (105-149 per network) | none stated |
| decoy pool | non-TF tokens present in either arm profile | `finetune/genecompass/build_grn_pooled.py:117-120` | eligible decoys | conservative: some non-TF genes have real effects (`GRN_DECOY_NULL_PREREG.md:17-18`) |
| threshold grid | 2 to 50 in steps of 0.5 | `deconvolution/grn_decoy_fdr.py:30` | FDR curve | registered (`GRN_DECOY_NULL_PREREG.md:29-30`) |
| edge-level reading cuts | 0.10 and 0.50 | `deconvolution/grn_decoy_fdr.py:71-73` | verdict | registered (`GRN_DECOY_NULL_PREREG.md:28-32`) |
| t* rule | pooled FDR <= 0.10 and >= 100 TF edges | `deconvolution/grn_decoy_fdr.py:64-65` | calibrated threshold, if any | registered |
| hub-level alpha | BH q < 0.10 within network | `deconvolution/grn_decoy_fdr.py:56, 81` | TFs beating decoys | registered (`GRN_DECOY_NULL_PREREG.md:33-35`) |
| anchors | Mef2c, Nr4a1, Myf6, Tfeb, Epas1 | `deconvolution/grn_decoy_fdr.py:29` | named TFs reported | registered (`GRN_DECOY_NULL_PREREG.md:35`) |

### Statistical procedures

- Null hypothesis: deleting this TF acts like deleting a random non-TF gene of the same expression in the same
  profile; under it, decoy and TF z are identically distributed (`deconvolution/GRN_DECOY_NULL_PREREG.md:12-15`).
  It tests TF specificity, not whether exercise has any effect (lines 17-18).
- Null generation: one decoy network per TF network, same cells and same 30 resamples (paired design).
- Edge-level estimator: ratio of tail fractions; it can exceed 1. No confidence interval is computed; TF and
  decoy edges share targets and resamples and are not independent.
- Hub-level test: one-sided empirical p against 105-149 decoy values, so the smallest attainable p is
  1/(1 + number of decoys), 0.0067-0.0095; BH within network over the network's TFs; no correction across networks.
- Units: edges (edge level), TFs within a network (hub level). Seeds: bootstrap 0, decoy 0.
- Pre-registered: design, statistics, cut-offs and reading, in both documents.

### Checks, validation and their results

Edge level (all values from `deconvolution/grn_decoy_record/fdr_per_network.tsv:2-9` and
`deconvolution/grn_decoy_record_16/fdr_per_network.tsv:2-7`; the six shared rows are identical in the two files):

| Network | TFs = decoys | TF edges | Decoy edges | Fraction abs(z)>=2, TF | Fraction abs(z)>=2, decoy | FDR(2) |
|---|---|---|---|---|---|---|
| BLOOD classical monocytes | 105 | 223,105 | 223,105 | 0.1450 | 0.1487 | 1.0253 |
| BLOOD erythroid cells | 123 | 261,457 | 261,457 | 0.1187 | 0.1237 | 1.0425 |
| BLOOD megakaryocytes and platelets | 110 | 235,472 | 235,574 | 0.1403 | 0.1420 | 1.0127 |
| BLOOD neutrophils | 117 | 247,787 | 247,861 | 0.1064 | 0.1030 | 0.9676 |
| LUNG endothelial cells (14-tissue arm only) | 140 | 291,612 | 291,612 | 0.0886 | 0.0890 | 1.0041 |
| LUNG mesothelial cells (14-tissue arm only) | 118 | 245,571 | 245,571 | 0.0829 | 0.0896 | 1.0812 |
| LUNG VSMC | 149 | 309,899 | 309,899 | 0.0743 | 0.0776 | 1.0444 |
| SKMGN tenocytes | 119 | 249,356 | 249,356 | 0.0939 | 0.1014 | 1.0796 |

| Summary | 8 networks (2026-10-31) | 6 default networks (2026-11-05) |
|---|---|---|
| Pooled FDR(2) | 1.030; 215,429 TF and 221,796 decoy edges at abs(z) >= 2 (`grn_decoy_record/summary.txt:11`) | 1.027; 169,219 and 173,828 (`grn_decoy_record_16/summary.txt:9`) |
| Pooled FDR over the whole grid | 0.977-1.066 (auditor's min/max of `fdr_curve_pooled.tsv`) | 0.972-1.052 (same) |
| t* | none on the grid (`summary.txt:12`) | none (`summary.txt:10`) |
| Registered reading | FDR(2) > 0.50: not distinguishable from random-gene deletion (`summary.txt:13`) | same (`summary.txt:11`) |
| Hub level, q < 0.10 | 0 of 981 TF-network pairs (`summary.txt:16-23`) | 0 of 723 (`summary.txt:14-19`) |
| Smallest q in any network | 0.372, LUNG mesothelial (auditor, from `hub_vs_decoy.tsv`; doc: 0.37) | 0.841, SKMGN tenocytes (doc: 0.84) |
| Anchors present | Nr4a1 q 0.960; Tfeb 0.983, 0.974; Epas1 0.923, 0.969, 0.917, 0.895 (`summary.txt:16-23`) | Nr4a1 0.960; Tfeb 0.983, 0.974; Epas1 0.917, 0.895 (`summary.txt:14-19`) |
| Anchors absent | Mef2c and Myf6 are not present in any network (`GRN_DECOY_NULL_RESULTS.md:41`) | same |

In the summary table, `summary.txt` is `deconvolution/grn_decoy_record/summary.txt` in the middle column and
`deconvolution/grn_decoy_record_16/summary.txt` in the right column; likewise `fdr_curve_pooled.tsv` and
`hub_vs_decoy.tsv`.

Auditor's additional reading of the hub-level p-values (`hub_vs_decoy.tsv` in each record): their median is 0.513
(8 networks) and 0.527 (6 networks), and 3.7% and 3.3% of them are below 0.05. A coarse p-value floor alone cannot
produce that: the p-values are distributed as they would be if TFs were further draws from the decoy distribution.

Reproduction check of the second run: all six decoy tables are identical to the first run's tables of the same
name, with maximum difference 0 in delta, boot_sd and z (`deconvolution/GRN_DECOY_NULL_16_RESULTS.md:30-32`).

Project's reading, which the audit finds supported by the tables: deleting a TF's token moves the embedding no more,
and no more differently between arms, than deleting a random gene of the same expression; the GRN carries no
TF-specific signal in its edges or its hub ranking (`deconvolution/GRN_DECOY_NULL_RESULTS.md:43-58`;
`manuscript/motrpac_genecompass.tex:1013-1035`).

### The abandoned permutation null

- Design as launched (2026-07-13): shuffle control/trained labels among a cell type's pseudo-cells within sex,
  keeping per-sex control counts, then run the identical build; FDR(t) = P_null(abs(z) >= t) / P_obs(abs(z) >= t)
  (`finetune/genecompass/build_grn_pooled.py:154-160, 184-210`; `load_arm_sex`, lines 85-104; the human builder has
  the same option, `build_grn_human.py:76-81, 116-132`).
- Why it was abandoned: label is determined by week, so every label shuffle makes the sham control arm a mixture of
  training weeks while the observed control arm is all week 0; the 10-cell control arm also carries four times the
  weight of the trained arm in the difference; the bootstrap SD then differs systematically between null and
  observed (measured ratio 0.93-1.08, sign varying by tissue), and the resulting FDR is biased by a median of about
  16.5% with no consistent sign (`deconvolution/GRN_NULL_WHY_ABANDONED.md:10-28`).
- Two further defects of that run: the same permutation seed gave every cell type of a tissue the same sham-control
  animals (12 distinct label draws instead of 42), and three draws cannot estimate a tail when edges are strongly
  clustered by TF (design effect about 105) (`deconvolution/GRN_NULL_WHY_ABANDONED.md:33-39`).
- An early partial estimate of 70-88% FDR at abs(z) >= 2 is declared untrustworthy by the project and is not to be
  quoted (`deconvolution/GRN_NULL_WHY_ABANDONED.md:30-31`).
- What remains on disk: 13 permuted rat-space tables in `data/deconvolution/grn_null/` (April model, pre-July
  references) and an empty `data/deconvolution/grn_human_null/`. The measured ratios, the 16.5% bias and the design
  effect have no script or table in the repository (NOT FOUND; looked in `deconvolution/`, `finetune/genecompass/`,
  `data/deconvolution/grn_null/`).

### Deviations, known defects and caveats

1. The second run is not an independent replication. Its six TF networks are edge-for-edge six of the first run's
   eight, and its decoys are identical (above). It confirms that the null reproduces on the default's own inputs;
   it adds no new evidence. The pooled figures differ only because two networks drop out.
2. One decoy seed, one decoy per TF. Seed variability of the matching was never measured
   (`deconvolution/GRN_DECOY_NULL_RESULTS.md:60`).
3. Matching quality is not recorded: the distance between each TF and its decoy is not written to any output, so
   how close "expression-matched" is cannot be verified from the tables (NOT FOUND).
4. The hub test's resolution is limited: with a p floor of 1/(1 + n decoys) and BH over the same number of TFs,
   q < 0.10 requires roughly ten or more TFs to exceed every decoy in a network (auditor's derivation from
   `deconvolution/grn_decoy_fdr.py:55-56`). The uniform-looking p distribution above shows the negative is not due
   to this alone.
5. `deconvolution/GRN_DECOY_NULL_RESULTS.md:37` says the top TF sits at the p floor "in five networks"; in the
   record it does so in six of the eight (all except classical monocytes and megakaryocytes; auditor's check
   against `hub_vs_decoy.tsv` and `fdr_per_network.tsv`).
6. "981 TFs", "723 TFs": TF-network pairs, not distinct TFs (310 and 290).
7. Both the TF and decoy networks include the one-arm-only component of 13c caveat 1 and the position effect of
   13a, so the test compares TFs with non-TFs under the same artefacts; it does not test whether a cleaner
   differential statistic would show TF specificity.
8. The decoy null says nothing about human space: only rat-space networks were given decoys
   (`deconvolution/GRN_DECOY_NULL_PREREG.md:20-21`).

### Methods references

- Benjamini Y, Hochberg Y. J R Stat Soc Series B 1995;57:289-300 (`manuscript/motrpac_genecompass.tex:1545`).
- The decoy (matched-gene) null is the project's own design; no external citation is given in the code or docs.

---

## Stage 13e - Rat-human self-consistency score (code and file names: "conservation")

### Research role

Asks whether a TF's differential target profile is the same when the same pseudo-cells are presented to the same
model as human cells. It was introduced as an Aim 3b "regulatory conservation" test
(`finetune/genecompass/build_grn_human.py:6-9`). It cannot be read as cross-species biology, for reasons that are
properties of the computation:

- No human measurement enters. The human-space dataset is the Stage 12 re-expression of the same rat pseudo-cells,
  in the same order, with the same arm split (`finetune/genecompass/build_grn_human.py:64-65, 78-80`;
  `docs/RETRAIN_V2_RESULTS.md:432-435`).
- The same model is used for both (`pipeline/run_stage13.py:176, 185`).
- 91.1% of rat genes with a human ortholog (13,883 of 15,234) carry the same token as that ortholog
  (`manuscript/motrpac_genecompass.tex:1044-1046`; confirmed by the auditor from `rat_tokens.pickle`,
  `rat_human_mouse_tokens.pickle` and `rat_to_human_mapping.pickle`). For the TF list specifically, 932 of the 941
  TFs with an in-vocabulary human ortholog are deleted as the identical token in both spaces (auditor's computation,
  same three files).
- The bootstrap resamples are the same animals in both spaces (13c).

The score therefore measures how stable the model's output is when the species token is switched from 2 to 0 and the
input passes through the ortholog projection and human tokenization. After 13d it is the self-consistency of a
quantity that is not TF-specific (`deconvolution/GRN_DECOY_NULL_RESULTS.md:54-55`).

Verdict: median per-TF Spearman 0.310 on the default, positive in 96% of scores, described by the project as
"predominantly positive but moderate" and explicitly not as conservation
(`manuscript/motrpac_genecompass.tex:1037-1042`).

### Inputs and outputs

Inputs: the rat and human `_pooled.tsv` tables with the same file name, and `rat_to_human_mapping.pickle`
(`finetune/genecompass/compare_grn_conservation.py:41, 75-83`). Outputs in
`<grn_human_dir>/conservation/`: `conservation_per_tf.tsv` (network, TF, n_shared, Spearman, p, anchor label),
`conservation_per_celltype.tsv`, `conservation_anchors.tsv` (lines 121-139).

### Procedure (`finetune/genecompass/compare_grn_conservation.py:86-132`)

1. Rat targets are mapped to human gene IDs through the ortholog map; rat targets without a human ortholog are
   dropped (lines 88-89).
2. For each TF present in both tables: select its rat edges with abs(z) >= 2 (the default `--select conf`); if
   fewer than 8 such edges exist, the selection is silently replaced by all of the TF's rat edges (lines 103-106).
   The human list is never pre-selected (lines 54-57, 102).
3. Average delta per human target ID on each side (several rat genes can map to one human gene) (lines 112-113).
4. Intersect the target sets; skip the TF if fewer than 8 targets are shared (lines 114-116).
5. Spearman correlation of rat delta against human delta over the shared targets, with scipy's two-sided p-value
   (line 117).
6. Per network: number of TFs scored, median rho, fraction positive, fraction with rho >= 0.3 (lines 126-130).
   Anchors Mef2c, Myf6 (myogenic) and Spi1, Cebpa (myeloid) are extracted into their own table (lines 29, 137-139).

### Models and software

pandas and scipy `spearmanr` (`finetune/genecompass/compare_grn_conservation.py:24-26`); CPU, 18.9 s
(`logs/rebuild_xtissue_12122211.out:3957`).

### Parameters, thresholds and flags

| Name | Value | Where set | What it controls | Rationale given |
|---|---|---|---|---|
| `--select` | `conf`: rat edges with abs(z) >= 2 | `finetune/genecompass/compare_grn_conservation.py:53-62` | which rat edges enter | selecting on the rat side only is unbiased; using all targets dilutes the score with near-zero edges (lines 54-62) |
| `--top-k` | 0 (off) | `finetune/genecompass/compare_grn_conservation.py:43-52` | old path that selects the top-k on both sides | kept only to reproduce the earlier, selection-inflated score (lines 46-52) |
| `--min-shared` | 8 | `finetune/genecompass/compare_grn_conservation.py:42` | minimum selected rat edges (else fallback to all) and minimum shared targets | none stated |
| rho >= 0.3 | reporting cut | `finetune/genecompass/compare_grn_conservation.py:130` | column `frac_conserved_rho>=0.3` | none stated |
| anchors | Mef2c, Myf6, Spi1, Cebpa | `finetune/genecompass/compare_grn_conservation.py:29` | rows copied to the anchor table | Mef2c, Spi1, Cebpa are the engine's pre-registered positive controls (`deconvolution/PERTURB_PREREG.md:17-26`); Myf6 is not in that pre-registration |

### Statistical procedures

- Statistic: Spearman rho between rat-space and human-space delta over a TF's selected, shared targets. The stored
  deltas are rounded to 1e-6, so ties are frequent among small edges.
- The p-value is scipy's asymptotic two-sided p treating targets as independent observations. No null distribution
  is generated in the code, no permutation, and no multiplicity correction across TFs or networks.
- Summary: median over TF-network scores, pooled or per network.
- Nothing about this score is pre-registered: no document registers a threshold, a null or a pass rule for it or for
  its anchors (searched all `*PREREG*.md` and `docs/RETRAIN_V2_PREREGISTRATION.md`; the latter only registers that the
  step is rerun, lines 263-264).
- Comparison values quoted in `deconvolution/GRN_NULL_WHY_ABANDONED.md:41-43` (0.410 when targets are chosen by
  abs(delta) at matched set size, 0.026 for the smallest abs(delta)) have no script in the repository (NOT FOUND).

### Checks, validation and their results

Summary by arm (medians recomputed by the auditor from each arm's `conservation_per_tf.tsv`; documents agree):

| Arm | Networks | TF-network scores (distinct TFs) | Median rho | Share rho >= 0.3 | Share rho > 0 | Document |
|---|---|---|---|---|---|---|
| Default: v2, refs4, 16 tissues | 6 | 633 (259) | 0.3099 | 0.532 | 0.956 | `deconvolution/ALL_TISSUE_RERUN_RESULTS.md:28` |
| v2, refs4, 14 tissues | 8 | 859 (276) | 0.3096 | 0.531 | 0.964 | `deconvolution/REFS4_RERUN_RESULTS.md:149` |
| v2, July references | 9 | 944 (291) | 0.2924 | 0.487 | 0.940 | `docs/RETRAIN_V2_RESULTS.md:499` |
| April model, July references (2026-07-17) | 11 | 1,275 (318) | 0.4509 | 0.764 | 0.988 | `docs/RETRAIN_V2_RESULTS.md:499` |
| April model, pre-July references, uncapped (2026-07-13) | 14 | 1,591 | 0.4506 | 0.771 | 0.988 | `data/deconvolution/grn_human_uncapped/conservation/conservation_per_tf.tsv` |
| Same, old two-sided top-40 selection | 14 | 1,227 | 0.5385 | 0.855 | 0.974 | `.../grn_human_uncapped/conservation/topk40_conservation_per_tf.tsv` |

The documents write "633 TFs", "859 TFs", "944 TFs"; these are TF-network scores.

Default, per network (`data/deconvolution/grn_v2_refs4all_rebuild_human/conservation/conservation_per_celltype.tsv:2-7`):

| Network | TFs scored | Median rho | Fraction positive | Fraction rho >= 0.3 |
|---|---|---|---|---|
| BLOOD classical monocytes | 95 | 0.2738 | 0.926 | 0.432 |
| BLOOD erythroid cells | 104 | 0.2611 | 0.894 | 0.413 |
| BLOOD megakaryocytes and platelets | 96 | 0.2854 | 0.938 | 0.458 |
| BLOOD neutrophils | 102 | 0.3807 | 0.990 | 0.775 |
| LUNG VSMC | 131 | 0.2894 | 0.977 | 0.450 |
| SKMGN tenocytes | 105 | 0.3869 | 1.000 | 0.676 |

633 of the 723 rat TF-network pairs receive a score; the median number of shared targets is 171 (range 15-1,331;
auditor's computation).

Anchors:

- Default (`.../grn_v2_refs4all_rebuild_human/conservation/conservation_anchors.tsv:2-4`): Spi1 in classical
  monocytes rho 0.109 (p 0.146, 178 targets); Cebpa in neutrophils 0.174 (p 0.024, 169); Spi1 in neutrophils 0.171
  (p 0.022, 180). No myogenic anchor row exists because neither muscle network is a myocyte block
  (`deconvolution/REFS4_RERUN_RESULTS.md:153-154`).
- v2 on the July references: 5 of 10 anchor rows at p < 0.05; SKMGN skeletal myocytes Mef2c 0.022 (p 0.80), Myf6
  0.335 (p 3.5e-6) (`docs/RETRAIN_V2_RESULTS.md:505-521`).
- April model, July references: 10 of 12 at p < 0.05; in the two myocyte networks Mef2c 0.074 (p 0.37, SKMGN) and
  0.230 (p 0.0027, SKMVL), Myf6 0.368 and 0.277 (`data/deconvolution/grn_human/conservation/conservation_anchors.tsv`;
  `docs/RETRAIN_V2_RESULTS.md:514-519`).
- April model, pre-July references, uncapped run of 2026-07-13: 12 of 17 at p < 0.05, and the myogenic anchors fail
  in the muscle parenchyma itself, 0 of 4: SKMGN muscle cells Myf6 0.125 (p 0.12) and Mef2c 0.028 (p 0.73); SKMVL
  muscle Myf6 0.091 (p 0.25) and Mef2c 0.097 (p 0.21); while Mef2c reaches 0.41 and 0.39 in SKMGN smooth muscle and
  SKMVL myofibroblasts (`data/deconvolution/grn_human_uncapped/conservation/conservation_anchors.tsv`, rows for
  `skmgn_skeletal_muscle_cells`, `skmvl_skeletal_muscle`, `skmgn_smooth_muscle_cells`, `skmvl_myofibroblasts`).
  The "0 of 4" belongs to that superseded run; on the July-reference rebuild of the same model three of the four
  myocyte rows are at p < 0.05.

Controls recorded by the project: April's score recomputed with the current script reproduces its 2026-07-17 tables
apart from the last digit of one p-value (`docs/RETRAIN_V2_RESULTS.md:441-442`). v2's score is lower than April's in
566 of the 729 TF scores both models make, and the drop cannot be attributed to weights or to tokenization
separately; v2's rat profiles share a median 0.810 of their 2,048 tokens with the human profile of the same cells
against 0.893 under April's tokenization (`docs/RETRAIN_V2_RESULTS.md:499, 539-548`;
`data/results/retrain_v2/downstream/grn_compare/grn_inputs.txt:2-54`). That input comparison was not repeated for
the refs4 arms.

### Deviations, known defects and caveats

1. Wording in code and outputs contradicts the registered reading: docstring "A TF is 'conserved' if the paired lists
   agree beyond chance" (`finetune/genecompass/compare_grn_conservation.py:9-12`), option value `conf` and its help
   text "Unbiased AND undiluted. median rho ~0.45" (line 58), column `frac_conserved_rho>=0.3` (line 130), "the
   regulatory logic ... is conserved" (`finetune/genecompass/build_grn_human.py:6-9`), output folder `conservation/`.
   The anchor table is printed as "pre-registered anchor edges" (line 140) although no criterion for it was
   registered.
2. Silent fallback: a TF with fewer than 8 rat edges at abs(z) >= 2 is scored on all of its targets
   (`finetune/genecompass/compare_grn_conservation.py:104-106`). Two of the 633 default scores have more than 1,000
   shared targets and are of this kind (auditor's count). Scores are therefore not all computed on the same kind of
   edge set.
3. The selected rat edges include one-arm-only targets (13c caveat 1; 19-27% of the abs(z) >= 2 edges in the two
   networks examined). Because both spaces use the same cells, a gene that is in only one arm's profile in rat
   space is likely to be so in human space, and its large same-signed delta would raise the rank correlation
   without reflecting any differential effect. The share of the score that comes from such edges has not been
   measured (not computed in this audit either).
4. The p-values treat targets as independent and are not corrected; they should not be read as significance of
   agreement. The anchor "significant" counts above are reported as the documents report them.
5. The score is selection-dependent by construction: 0.538 with two-sided top-40 selection, 0.451 with rat-side
   abs(z) >= 2, and about 0.15 with all targets on the same networks
   (`finetune/genecompass/compare_grn_conservation.py:46-62`; first two confirmed above). The earlier published
   figure of 0.507 was the two-sided selection (`docs/OPEN_FRONTIER.md:143`).
6. Two different summaries circulate under the name "median". 0.310 is the median over all TF-network scores. The
   same eight 14-tissue networks are also reported as "median 0.297" (`deconvolution/REFERENCE_RELABEL_PREREG.md:531`),
   which is the median of the eight per-network medians (auditor's computation from
   `data/deconvolution/grn_v2_refs4_human/conservation/conservation_per_celltype.tsv`). On the default the median of
   the six per-network medians is 0.287 (same computation on the default table).
7. The capstone consequence stated by the project: the network contributes no TF-specific evidence to the planned
   synthesis with human genetics (`manuscript/motrpac_genecompass.tex:1052-1054`).

### Methods references

Spearman rank correlation as in scipy 1.17.1; no method paper cited in code or docs.

---

## Stage 13f - Gene-set enrichment of the per-cell-type dose response

### Research role

Summarises each analysable (tissue x cell type) DE block of Stage 10 as pathways: which gene sets are concentrated
at the top or bottom of the block's signed dose statistic. It does not use the GeneCompass model
(`docs/RETRAIN_V2_RESULTS.md:559-560`) and is not affected by 13b-13e. It is descriptive: no claim is tested and
nothing is pre-registered. Result on the default: 7,502 gene-set hits at adjusted p < 0.05 in 132 of 151 blocks.

### Inputs and outputs

Inputs: `<genecompass_input_dir>/pseudobulk_de/de_summary.tsv` (blocks with `status == "ok"`, 151 on the default) and
each block's `de__<cell type>.tsv` (`deconvolution/R/run_gene_set_enrichment.R:64-72`); `rat_token_mapping.tsv` for
gene symbols (line 34). Outputs in `pseudobulk_de/enrichment/`: one `gsea__<TISSUE>__<cell type>.tsv` per block
(151 files) with columns `pathway, pval, padj, log2err, ES, NES, size, leadingEdge, tissue, cell_type`; the
combined `enrichment_summary.tsv` (hits with padj < 0.05); `provenance_run_gene_set_enrichment.json`
(lines 96-112).

### Procedure (`deconvolution/R/run_gene_set_enrichment.R`)

1. Gene sets: `msigdbr(species = "Homo sapiens")`, keeping collection H (Hallmark) and sub-collection CP:REACTOME,
   as lists of upper-cased human gene symbols (lines 46-55). 1,889 sets were loaded
   (`.../enrichment/provenance_run_gene_set_enrichment.json:28`).
2. Rat gene to human symbol: the rat gene's own symbol, upper-cased; genes without a rat symbol are dropped
   (lines 39-42, 84-85).
3. Per block: keep genes with finite `slope_week` and `P_dose_comb`; statistic =
   sign(`slope_week`) x -log10(max(`P_dose_comb`, 1e-300)); where several genes share a symbol keep the one with the
   largest absolute statistic; skip the block if fewer than 50 genes remain (lines 82-90).
   - `P_dose_comb` is the p-value of the limma-trend moderated F test on all training-week coefficients of the model
     `~ sex + week (factor)` (or `~ week` in a single-sex tissue) (`deconvolution/R/run_pseudobulk_de.R:218-221, 281`).
   - `slope_week` is the coefficient of numeric week in `~ sex + week (numeric)`
     (`deconvolution/R/run_pseudobulk_de.R:233-235, 282`).
4. `set.seed(SEED)` before each block, then `fgsea(pathways, stats, minSize = 10, maxSize = 500)` with every other
   argument at the package default (lines 91-94). A block whose call errors is skipped without a message
   (lines 92-95).
5. Write the block table sorted by padj, with the leading edge truncated to 20 genes; collect rows with padj < 0.05
   into the summary (lines 96-107).

The wrapper sets the R library, tries to install missing packages, and runs the script
(`deconvolution/R/run_gene_set_enrichment.sh:11-36`).

### Models and software

| Item | Value | Where recorded |
|---|---|---|
| R | 4.4.1 | `.../enrichment/provenance_run_gene_set_enrichment.json` (`r_version`); `deconvolution/setup/r_packages.yaml:17` |
| fgsea | 1.32.4, Bioconductor 3.20 | provenance JSON line 29; `deconvolution/setup/r_packages.yaml:37`; `R_libs/fgsea/DESCRIPTION` |
| Algorithm | the output has a `log2err` column, which is the multilevel variant's output; the script does not name the algorithm | header of `gsea__BLOOD__Classical_monocytes.tsv`; fgsea's default being `fgseaMultilevel` is from the package documentation [confirmed in the installed fgsea 1.32.4: `fgsea()` calls `fgseaMultilevel` unless `nperm` is given] |
| msigdbr | 26.1.0 (CRAN) | provenance JSON line 30; `deconvolution/setup/r_packages.yaml:30` |
| Gene-set database release | MSigDB v2026.1 (human) | `R_libs/msigdbr/NEWS.md:1-3`; the data files are `~/.cache/R/msigdbr/msigdb.2026.1.Hs.{H,C2,...}.rds`, downloaded 2026-07-06 into the user's home cache (the installed package refers to a Zenodo archive `msigdb.2026.1.zip`; string in `R_libs/msigdbr/R/msigdbr.rdb`) |
| Collections | Hallmark (H) and Reactome (C2 CP:REACTOME); 1,889 sets loaded; in one block 50 Hallmark + 1,297 Reactome sets passed the size filter | script lines 51-52; provenance line 28; auditor's count in `gsea__BLOOD__Classical_monocytes.tsv` |
| `msigdbdf` | not installed; the wrapper tries to install it at every run and the attempt fails with "not available for this version of R" | `deconvolution/setup/r_packages.yaml:30`; `deconvolution/R/run_gene_set_enrichment.sh:28-30`; `logs/rebuild_xtissue_12122211.out` (Step 4 header, lines 3959-3984) |

### Parameters, thresholds and flags

| Name | Value | Where set | What it controls | Rationale given |
|---|---|---|---|---|
| `MIN_SET` / `MAX_SET` | 10 / 500 | `deconvolution/R/run_gene_set_enrichment.R:35` | gene-set size after intersecting with the ranked genes | none stated |
| `SEED` | 1 (env `GSEA_SEED`) | `deconvolution/R/run_gene_set_enrichment.R:36, 91` | R RNG, reset per block | "per block, so results don't depend on block order" (line 91) |
| Ranking statistic | sign(slope_week) x -log10(P_dose_comb) | line 86 | gene ranking | "signed dose statistic" (lines 5-9) |
| p floor | 1e-300 | line 86 | avoids infinite statistics | none stated |
| Duplicate symbols | keep largest absolute statistic | line 87 | one statistic per symbol | none stated |
| Minimum ranked genes | 50 | line 90 | blocks with fewer are skipped | none stated |
| Significance | padj < 0.05 | lines 100-101 | rows written to the summary | none stated |
| Leading edge stored | first 20 genes | line 98 | truncation of the reported leading edge | none stated |
| Blocks | `status == "ok"` in `de_summary.tsv` | lines 64-66 | which blocks are tested | avoids orphan files and wrong labels (lines 60-63) |
| `DE_DIR` | set by the orchestrator | `pipeline/run_stage13.py:315`; script lines 28-31 | DE folder | - |

### Statistical procedures

- Test: pre-ranked gene-set enrichment (fgsea) of each set against the block's full ranked list. The universe is the
  ranked list itself: every gene tested in the block's DE with a finite statistic and a rat symbol, de-duplicated by
  symbol; 11,631 to 13,937 genes in the first five blocks logged (`logs/rebuild_xtissue_12122211.out`, Step 4 block
  lines after 3985). There is no separate background set.
- Two-sided in the sense that positive and negative enrichment scores are both reported (NES of both signs in the
  output); the `scoreType` argument is left at its default [confirmed in the installed fgsea 1.32.4: `scoreType = c("std", "pos", "neg")`, so "std"].
- p-values are estimated by fgsea's sampling, so they depend on the seed; enrichment scores do not
  (`docs/RETRAIN_V2_RESULTS.md:559-565`).
- Multiplicity: `padj` is BH over the gene sets tested within one block. Auditor's check on one block: BH applied to
  the block's non-missing p-values reproduces `padj` to 4e-15. There is no correction across the 151 blocks, and no
  pre-registered hypothesis.
- Seed: 1, reset before every block (since commit `3a0165f`, 2026-10-05; `docs/RETRAIN_V2_RESULTS.md:561-562`).

### Checks, validation and their results

| Item | Value | File |
|---|---|---|
| Blocks tested | 151 | `logs/rebuild_xtissue_12122211.out:3985`; 151 `gsea__*.tsv` files |
| Hits with padj < 0.05 | 7,502 across 132 blocks | `logs/rebuild_xtissue_12122211.out:4139`; auditor's count of `enrichment_summary.tsv`; `manuscript/motrpac_genecompass.tex:833` |
| Gene-set tests in total | 200,588, of which 139 have no p-value | auditor's count over the 151 block files |
| Earlier arm (v2, July references) | 4,724 hits in 98 blocks; April 4,687 in 97 | `docs/RETRAIN_V2_RESULTS.md:558` |
| Sensitivity to the seed | outside TESTES, 4,381 hits in both runs, 306 only in April's and 342 only in v2's, although the ranks are identical; flipped calls have a median padj of 0.058 in the other run (max 0.15) | `docs/RETRAIN_V2_RESULTS.md:563-565` |
| Missing p-values can be very strong hits | Hallmark heme metabolism has no p-value in one run where the other gives about 1e-22 | `docs/RETRAIN_V2_RESULTS.md:566-569` |

### Deviations, known defects and caveats

1. The ranking statistic combines two different models: its magnitude is an omnibus F test over the four training
   weeks (not directional) and its sign is that of a linear week slope. A gene with a strong non-monotone response
   is placed at an extreme of the ranking with the sign of whatever linear trend it has.
2. The ranking p-value is not the one Stage 10 uses to call dose-significant genes (the Fisher combination of the
   per-sex training p-values, weighted by IHW; `deconvolution/R/run_pseudobulk_de.R:265-266, 293-296, 338`). In
   single-sex tissues the Fisher p is missing by construction, so OVARY and TESTES have no dose-significant genes
   (`deconvolution/ALL_TISSUE_RERUN_RESULTS.md:52-56`), yet they are enriched here because `P_dose_comb` exists for
   them: OVARY has 141 hits and TESTES 1 in the default summary (auditor's count of `enrichment_summary.tsv`; 21
   OVARY/TESTES block files). The 7,502 total includes them.
3. Rat-to-human mapping is by upper-casing the rat symbol. The script header says the mapping is "via the Stage-3
   ortholog table -- 1:1 for orthologous genes" and a comment says "keep genes w/ human ortholog" (lines 6-8, 38),
   but the code neither consults the ortholog table nor requires an ortholog (lines 39-42). A rat gene whose symbol
   differs from its human ortholog's is silently outside every gene set.
4. The gene-set data are not under project control: msigdbr 26.1.0 keeps them in a per-user cache in the home
   directory, outside the repository and `R_libs/`; the provenance file records the msigdbr version and the number
   of sets but not the MSigDB release or a checksum. The wrapper attempts a network install on the compute node at
   every run.
5. 139 tests return no p-value and are silently excluded from the adjusted p-values and the summary; fgsea errors
   and blocks with fewer than 50 genes are skipped without a message (lines 90, 92-95). On the default all 151 blocks
   produced a file.
6. Results are seed-dependent near the cut-off (above); the manuscript states this
   (`manuscript/motrpac_genecompass.tex:839-841`).
7. `deconvolution/setup/r_packages.yaml:11-12` says versions were captured on 2026-06-02, whereas fgsea and msigdbr
   were installed on 2026-07-06 (`R_libs/fgsea/DESCRIPTION`, `R_libs/msigdbr/DESCRIPTION`, "Built" fields) and added
   to the manifest on 2026-11-01 (line 28).

### Methods references

- fgsea: Korotkevich G, Sukhov V, Budin N, Shpak B, Artyomov MN, Sergushichev A. Fast gene set enrichment analysis.
  bioRxiv 2021, doi:10.1101/060012 [verified against Crossref; see REFERENCES.md]; not in the manuscript bibliography.
- GSEA: Subramanian A, et al. PNAS 2005;102:15545-15550 [verified against Crossref; see REFERENCES.md].
- MSigDB Hallmark: Liberzon A, et al. Cell Systems 2015;1:417-425 [verified against Crossref; see REFERENCES.md]. Reactome: citation not given anywhere in
  the repository [suggested: Milacic et al. 2024, verified; see REFERENCES.md].
- msigdbr: Dolgalev I, R package msigdbr [software; no registry record checked].
- limma-trend statistics that feed the ranking are documented in the Stage 10 section.

---

## Stage 13g - CPA dose model (Module C): status

### Research role

Planned Aim 2c analysis: model the training-week response of hotspot pseudo-cells as a learned dose-response curve
with a compositional perturbation autoencoder (week as dose, sex as covariate) and compare it with the Stage 10
linear week slope (`deconvolution/DOWNSTREAM_BUILD_PLAN.md:159-171`;
`manuscript/wip/wip_aim2c_cpa_dose_response.md`). Verdict: not built and not run. No result exists.

### What exists

| Item | State | Evidence |
|---|---|---|
| Isolated Python environment `/depot/reese18/apps/cpa-env` | built 2026-07-27, Python 3.9.25 | `deconvolution/cpa/README.md:24-25`; directory present; `python --version` |
| Pinned packages | `cpa-tools` from git commit `fbd7c0250edc23eff003a10c99655579c53afd63`, scvi-tools 0.20.3, torch 2.0.0, pytorch-lightning 1.9.5, numpy 1.23.5, anndata 0.9.2, scanpy 1.10.3, pyarrow 15.0.2, ray 2.9.3 | `deconvolution/cpa/requirements-cpa.lock.txt:34, 206, 225, 190, 129, 8, 203, 169, 194` |
| Setup script | creates the venv, installs, checks that `cpa.CPA` and `cpa.CPA.setup_anndata` import, freezes the lock | `deconvolution/cpa/setup_cpa_env.sh:46-65` |
| Requirements | `pyarrow<16`, `cpa-tools` from `github.com/theislab/cpa` | `deconvolution/cpa/requirements-cpa.txt:19, 22` |
| Driver `deconvolution/run_cpa_dose.py` | does not exist ("the driver, to be built") | `deconvolution/cpa/README.md:35`; no such file in `deconvolution/` |
| Outputs | none: no `cpa` folder under `data/deconvolution/` to depth 3 | directory search |
| Tracker | open item | `docs/OPEN_FRONTIER.md:142, 199` |
| Manuscript | work-in-progress box with the placeholder "[Latent dose trajectories and DE-vs-CPA concordance to be inserted.]" | `manuscript/motrpac_genecompass.tex:1068-1077` |

The only thing ever executed is the import check in the setup script. No model was trained, no hyperparameters are
fixed, no pre-registration exists.

### Deviations and caveats

- The plan changed engines: the build plan specifies wrapping the vendored
  `vendor/GeneCompass/downstream_tasks/drug_dose_response/cpa/` (`deconvolution/DOWNSTREAM_BUILD_PLAN.md:161-162`); the
  environment installs the maintained `cpa-tools` instead (`deconvolution/cpa/README.md:3-4`). The build plan was not
  updated.
- `deconvolution/cpa/requirements-cpa.txt:10-12` says git main "resolves modern scvi-tools compatible with Python
  3.12"; the README and setup script state the opposite (Python >= 3.9, < 3.11; scvi-tools 0.20.3)
  (`deconvolution/cpa/README.md:24-28, 52-56`; `deconvolution/cpa/setup_cpa_env.sh:22-24`).
- `cpa-tools` is pinned in the lock file by commit, but `requirements-cpa.txt` installs the moving git head, so a
  rebuild from the requirements file need not reproduce the lock.
- The README's input path is the v2 July-reference arm (`deconvolution/cpa/README.md:45`), not the current default.
- Stated risk: about 50 pseudo-cells per cell type is thin for an autoencoder
  (`deconvolution/DOWNSTREAM_BUILD_PLAN.md:169-171`; `docs/OPEN_FRONTIER.md:142`).
- This environment is a second Python environment beside `motrpac-env`; the README gives the reasons it cannot be
  merged (`deconvolution/cpa/README.md:49-66`).

### Methods references

CPA: Lotfollahi M, Klimovskaia Susmelj A, De Donno C, et al. Predicting cellular responses to complex perturbations
in high-throughput screens. Mol Syst Biol 2023;19(6):e11517 (`manuscript/motrpac_genecompass.tex:1521`).

---

## Negative results of Stage 13, stated plainly

1. The differential GRN has no TF-specific signal. Edges with abs(z) >= 2 occur at the same rate for deletions of
   expression-matched non-TF genes (FDR(2) = 1.03 on both runs; no threshold up to 50 reaches 0.10), and no TF's
   hub strength exceeds its decoys' (0 of 723 TF-network pairs on the default; 0 of 981 on the 14-tissue arm). Every
   named anchor that is present fails (q 0.90-0.98). (13d)
2. abs(z) has no calibrated error rate and cannot be given one by label permutation in this design; the permutation
   null was abandoned as invalid. (13d)
3. Hub strength largely reflects the TF's own expression (Spearman 0.79-0.84 with the TF's value in the profile),
   and by the auditor's computation 20-63% of it in two default networks comes from genes that are in only one
   arm's profile, where the "difference" is one arm's raw shift. (13c)
4. The current model fails criterion 2 of the engine's pre-registration (0 of 25 targets). The earlier PASS on that
   criterion was scored against a gene set written after the first results were seen; none of the four targets named
   in the frozen gene file ever appears in Mef2c's top 25; and the top-25 lists of unrelated genes largely coincide.
   (13b)
5. The rat-human score is the model's self-consistency under a species-token switch on the same cells, not
   cross-species conservation. It is lower for the current model (0.310 default, 0.292 on the July references) than
   for the April model (0.451). The myogenic anchors failed in muscle parenchyma (0 of 4) in the 2026-07-13 run,
   and no myogenic anchor can be evaluated on the default. (13e)
6. The pre-registered myeloid replication of the engine check was never scored. (13b)
7. The CPA dose model has not been built or run. (13g)

## Items that could not be determined

- How the 80-gene Module A panel was drawn (no script or seed).
- The exact command and inputs that produced `rat_tf_ensrnog.tsv` (whether an expression filter was applied).
- The scripts behind the hub-statistic comparison (count vs sum) and behind the numbers quoted in
  `GRN_NULL_WHY_ABANDONED.md` (SD ratio 0.93-1.08, bias 16.5%, design effect 105, 0.410, 0.026).
- How closely each decoy matches its TF in expression (not written out).
- The hub-versus-expression correlation and the rat/human profile overlap for the refs4 and 16-tissue arms (only
  computed for April and for v2 on the July references).
- How much of the self-consistency correlation is carried by one-arm-only targets, and how much of each target
  shift is the position effect of deletion.
- (Resolved afterwards.) fgsea's algorithm and `scoreType` defaults were first inferred from the output columns and
  then confirmed in the installed fgsea 1.32.4: `fgseaMultilevel`, `scoreType = "std"`, `sampleSize = 101`,
  `eps = 1e-50`.
