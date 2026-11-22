# Pipeline

**The question.** MoTrPAC measured the endurance-trained rat by bulk RNA-seq, which blends every cell type of a
tissue. Can a rat-adapted GeneCompass model, given per-cell-type expression deconvolved from that bulk, show
which cell types respond to training, which genes respond in them, and whether the response carries over to the
model's human representation?

**Where it stands.** The table gives each stage's question, its result on the default configuration, and what
qualifies that result. The default is **refs5**, the registered rerun of Stages 8-14 that followed the
2026-11-11 audit. Every number comes from `docs/audit/remediation/STEP3_RESULTS.md` (the rerun),
`STEP4_RESULTS.md` (the validations) and `STEP5_RESULTS.md` (the corpus), where each has a table on disk.

| Stage | Question | Result (refs5: 16 tissues, 749 samples, v2 model) | What qualifies it |
|---|---|---|---|
| 1-7 | Does the model learn rat without forgetting human and mouse, and does it generalise to a new rat study? | Rat gene-identity loss 7.13 to 3.06; human +0.05% and mouse +1.32% against the base model (limit +10%). On 66,000 cells of 33 studies the model never saw, the loss is **1.046** times the in-sample loss (registered bar 1.10) | The gap is largest for rat-only tokens (1.18). The corpus departs from its description and was not repaired (no retrain): 10.4% of the 6,803,206 training cells are above 15% mitochondrial reads, 1.04% of counts in ID-keyed files sit under another gene's token, and 79 confirmed single-cell series never entered it (33 of them are now the held-out set) |
| 8 | Can the bulk be split into per-cell-type expression? | 194 cell-type blocks; a repeat of the run is byte-identical. On known-truth mixtures BayesPrism meets the registered floor on **19 of 19** sets (median per-type r 0.976 to 0.999) | Fractions are shares of mRNA, not of cells: BayesPrism's error against cell fractions rises under the references' own mRNA differences on 13 of 15 references. Expression is recovered at a cell type's own abundance for **4 of the 12 called blocks** and for none of the five muscle blocks. Of the dominant types, 12 of 15 pass their sweep; BLOOD erythroid cells and SPLEEN red-pulp macrophages fail |
| 9 | Do the pseudo-cells enter the model as its training cells did? | 9,060 pseudo-cells embedded; target sum 6,700 by the registered formula; the weights load completely (234 of 234 tensors) | One target sum for 16 tissues is a compromise: LIVER sits on the corpus median (0.91), the others between 0.87 (BAT) and 1.13 (LUNG). 22 LUNG goblet-cell pseudo-cells are dropped (under 200 genes) |
| 11 | Which cell types carry a detectable training signal? | **11 hotspots and 1 borderline block** of 150 analysable blocks | **The list is not stable to small changes of the inputs**: 7 of the 9 earlier hotspots remain, and a block's AUC moves by up to 0.31 between runs in tissues whose reference did not change. At 8 weeks against control 9 of the 12 called blocks are significant. The two LUNG hotspots do not separate on expression alone (AUC 0.59, 0.60). All five muscle blocks are below 1% of their tissue. The embedding's AUC is above PCA-50's in 88 of 150 blocks (p 0.013) and above the all-genes probe's in 80 (p 0.38) |
| 10 | Which genes respond to training dose, in which cell type? | **52,961** dose-significant gene x block pairs over 9,617 distinct genes; 41,401 pairs outside low-fraction blocks. Single-sex tissues now have a dose test (TESTES 219 pairs, OVARY 8) | **No block's calls are robust to technical covariates** by the registered verdict (0 of 150): 44% and 33% of the pairs survive the two reruns. **The positive-control red flag stands**: 5 of the 26 testable pipeline-sanity controls recover. 139 pairs are sex-consistent at 8 weeks by concordance |
| 12 | Does the signal survive re-expression of the rat cells as human? | **10 of 11** hotspots by the registered rule; **9 of 11** by the corrected rule (not SKMGN Schwann cells or HEART vascular smooth muscle cells). LUNG vascular smooth muscle cells lose their dose trend | The human test reuses the rat test's cells, labels, folds and seeds. It is not a replication, and no human measurement enters. The projection onto the Tabula Sapiens atlas (0.430 against 0.086 by chance) was run on the default before refs5 and has not been repeated |
| 13 | Do the model's in-silico deletions carry TF-specific signal? | **No.** 7 networks, 1,540,997 edges. The decoy null is negative: the pooled false-discovery estimate at \|z\| of 2 is 1.02, and 0 of 777 TF x network pairs beat expression-matched decoys | The networks are descriptive model readouts; nothing in them is read as regulation. The rat-human score (median 0.48 over 685 scores) compares the same cells through the same model: it is self-consistency of the model, not conservation. Module A: criteria 1, 3 and 4 pass on refs5; the registered myeloid replication fails (Spi1 at the 81st percentile, bar 90th). Enrichment: 4,956 significant sets in 116 of 150 blocks |
| 14 | Do the calls hold against composition, and is the dominant type's expression recovered? | The 11 configured purity sweeps pass (median correlation 0.993 or more at every purity of 0.5 and above). Composition: 25 blocks flagged, 66 pass, 59 quiet, 44 not analysed | In 2 of the 11 hotspots the estimated fraction itself trends with dose (SKMVL vascular smooth muscle cells, SKMGN Schwann cells). Stage 14's own RIN/globin check refits a pooled-sex dose slope, a different model from the one that makes the headline calls; the registered check is the one in the Stage 10 row |

**No called block passes all nine registered readings** (`STEP3_RESULTS.md`, B8 and Figure B3). BLOOD
megakaryocytes and platelets and BLOOD classical monocytes pass eight; the one they fail is the robustness of
their DE calls to technical covariates, which every block fails.

The 11 hotspots: BLOOD classical monocytes, erythroid cells, megakaryocytes and platelets, neutrophils; LUNG
endothelial cells and vascular smooth muscle cells; HEART vascular smooth muscle cells; SKMVL satellite cells,
tenocytes and vascular smooth muscle cells; SKMGN Schwann cells. Borderline: SKMGN vascular smooth muscle cells.

## Orchestrators

Each stage has a `run_stageN.py` orchestrator that validates config + inputs and
runs its steps as subprocesses (`--dry-run` / `--from N` / `-v` supported). Stage
workers live in `NN_*/` here, except the fine-tuning (7), deconvolution and
downstream-analysis stages (8–14), which drive the scripts in `finetune/genecompass/`,
`deconvolution/` and `translation/` in place.

| Stage | Orchestrator | What it does |
|---|---|---|
| 1 | `run_stage1.py` | Data harvesting & QC |
| 2 | `run_stage2.py` | Gene universe & cell QC |
| 3 | `run_stage3.py` | Ortholog mapping |
| 4 | `run_stage4.py` | Gene medians (scatter / gather). The divisor Stage 5 and Stage 9 use is the hybrid of these and GeneCompass's human medians: [`docs/SCHEME_B_MEDIANS.md`](../docs/SCHEME_B_MEDIANS.md) |
| 5 | `run_stage5.py` | Reference assembly & tokenization |
| 6 | `run_stage6.py` | Prior-knowledge embeddings |
| 7 | `run_stage7.py` | Fine-tuning, in four steps: initial checkpoint for the rat vocabulary, the rat/human/mouse training mix, the fixed evaluation probe, then training (phase 1 with the encoder frozen, phase 2 unfrozen). It never regenerates an existing checkpoint, mix or probe |
| **8** | `run_stage8.py` | **Deconvolution** (MoTrPAC bulk → pseudo-cells; drives `deconvolution/`) |
| **9** | `run_stage9.py` | **Tokenize + embed** (pseudo-cells → GeneCompass embeddings) |
| **10** | `run_stage10.py` | **Aim-2 analysis** — per-cell-type DE, the pre-registered positive-control comparison, and the registered rerun of the whole DE with technical covariates (whole-experiment) |
| **11** | `run_stage11.py` | **Aim-2 gate** — which cell types carry a detectable training signal: the supervised probe with its permutation test over every tissue's embeddings (10,000 permutations, then batches up to 1,000,000 for blocks still open), the Augur and PCA corroboration, and the merge that gives every block its class: hotspot, borderline or none (run once, after every tissue is embedded) |
| **12** | `run_stage12.py` | **Cross-species transfer** — re-express rat pseudo-cells in human embedding space |
| **13** | `run_stage13.py` | **Mechanism** — differential network of in-silico TF deletions (rat + human space, the difference taken on genes present in both arms) → rat–human self-consistency score → pathway enrichment → decoy-null networks and their scores. The networks carry no TF-specific signal (decoy null) |
| **14** | `run_stage14.py` | **Hardening** — per-tissue expression-Z purity sweeps of the dominant type, the composition-confound table, and the earlier RIN/globin check on the pooled-sex dose slope |

The order in which the stages are run is 8, 9, 11, 10, 12, 13, 14: Stage 10 reads the hotspot table that
Stage 11 writes (`deconvolution/DECONV_REBUILD_RUNBOOK.md` gives the commands). The table omitted Stage 11
until 2026-11-14.

**Where the outputs are.** The folders change with each arm of the analysis, so they are not listed here:
read them from `config/pipeline_config.yaml`.

| Config key | Holds |
|---|---|
| `deconvolution/tissue_references.yaml`, `reference_dir` of each tissue | The single-cell reference Stage 8 deconvolves that tissue against |
| `deconvolution.rat_exclude_genes` | Stage 8: the ribosomal, mitochondrial and hemoglobin genes left out of the deconvolution |
| `deconvolution.results_dir` | Stage 8: fractions, expression (Z) and the sample key of each tissue, under `motrpac/<TISSUE>/` |
| `deconvolution.genecompass_input_dir` | Stages 8–9: pseudo-cells, tokenized datasets and embeddings per tissue; Stage 10's `pseudobulk_de/` |
| `deconvolution.pseudocell_target_sum` | Stages 9 and 12: the tokenizer's target sum |
| `deconvolution.hotspots_table` | Stage 11: the gate's merged table; its folder holds every gate output |
| `deconvolution.augur_input_dir` | Stage 11: the exports for Augur |
| `deconvolution.genecompass_input_dir` + `_human` | Stage 12: the same pseudo-cells re-expressed as human, and their gate |
| `deconvolution.grn_dir`, `deconvolution.grn_human_dir` | Stage 13: the network tables in rat and human space |
| `deconvolution.validation_dir` | Stage 14: the purity sweeps |

A hotspot is a block whose `gate_class` is `hotspot` in the table at `deconvolution.hotspots_table` (column
`is_hotspot`). A `borderline` block is listed beside the hotspots and is not one. Tables written before refs5
have no class column; for them a hotspot is `q_sup_trained < 0.05`.

The default arm is the one `config/pipeline_config.yaml` names: refs5 since 2026-11-15
(`docs/audit/remediation/STEP3_PREREG.md` and `STEP3_RESULTS.md`). Another arm is selected with
`PIPELINE_PROFILE=config/profiles/<name>.yaml`, a file of overrides whose header says what it is:
`refs4all_rebuild.yaml` (the default before refs5, with its own exclusion list and target sum), `refs4_14.yaml`
(14 tissues), `july_refs.yaml` (the July references), `april.yaml` (April's model). `refs5.yaml` now repeats
the default and changes nothing; its header says so.

Stages 8–9 are the Aim-2 deconvolution bridge; Stage 11 the gate; Stage 10 the cell-type-resolved DE; Stage 12
the Aim-3a transfer; and Stages 13–14 the Aim-2b/3b mechanism and the hardening checks. Stages 13–14 loop
per cell type / per tissue and skip work whose output already exists (`--force` to re-run); the fast
parallel path is the array jobs in `slurm/analysis/` (gitignored, local), while the orchestrators are
the reproducible sequential entry points. Stage 13 steps 1–2 need a GPU; Stage 14 step 1 is CPU-heavy
(BayesPrism). See `deconvolution/README.md` for per-tissue usage and `deconvolution/setup/SETUP.md` for
first-time setup (environment, data, config).
