# Pipeline

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
| **10** | `run_stage10.py` | **Aim-2 analysis** — per-cell-type DE + pre-registered positive-control comparison (whole-experiment) |
| **11** | `run_stage11.py` | **Aim-2 gate** — which cell types carry a detectable training signal: the supervised probe with its permutation test over every tissue's embeddings, the Augur and PCA corroboration, and the merge that defines the hotspots (run once, after every tissue is embedded) |
| **12** | `run_stage12.py` | **Cross-species transfer** — re-express rat pseudo-cells in human embedding space |
| **13** | `run_stage13.py` | **Mechanism** — differential GRN (rat + human space) → rat–human self-consistency score → pathway enrichment |
| **14** | `run_stage14.py` | **Hardening** — per-tissue expression-Z purity sweeps, composition-confound + RIN/globin DE robustness |

The order in which the stages are run is 8, 9, 11, 10, 12, 13, 14: Stage 10 reads the hotspot table that
Stage 11 writes (`deconvolution/DECONV_REBUILD_RUNBOOK.md` gives the commands). The table omitted Stage 11
until 2026-11-14.

**Where the outputs are.** The folders change with each arm of the analysis, so they are not listed here:
read them from `config/pipeline_config.yaml`. This paragraph named folders of a superseded arm until
2026-11-14.

| Config key | Holds |
|---|---|
| `deconvolution/tissue_references.yaml`, `reference_dir` of each tissue | The single-cell reference Stage 8 deconvolves that tissue against |
| `deconvolution.results_dir` | Stage 8: fractions, expression (Z) and the sample key of each tissue, under `motrpac/<TISSUE>/` |
| `deconvolution.genecompass_input_dir` | Stages 8–9: pseudo-cells, tokenized datasets and embeddings per tissue; Stage 10's `pseudobulk_de/` |
| `deconvolution.hotspots_table` | Stage 11: the gate's merged table; its folder holds every gate output |
| `deconvolution.augur_input_dir` | Stage 11: the exports for Augur |
| `deconvolution.genecompass_input_dir` + `_human` | Stage 12: the same pseudo-cells re-expressed as human, and their gate |
| `deconvolution.grn_dir`, `deconvolution.grn_human_dir` | Stage 13: the network tables in rat and human space |
| `deconvolution.validation_dir` | Stage 14: the purity sweeps |

The default arm is the one `config/pipeline_config.yaml` names. Another arm is selected with
`PIPELINE_PROFILE=config/profiles/<name>.yaml`, a file of overrides whose header says what it is:
`refs5.yaml` (the audit-remediation rerun, `docs/audit/remediation/STEP3_PREREG.md`), `refs4_14.yaml` (14
tissues), `july_refs.yaml` (the July references), `april.yaml` (April's model). Several profiles now repeat
the default and change nothing; their headers say so.

**The table below quotes the default arm as it stood before the refs5 rerun.** It is rewritten when that
rerun is adopted.

| Stage | Question | Current result (16 tissues; `deconvolution/ALL_TISSUE_RERUN_RESULTS.md`; refs4 14-tissue figures in brackets) |
|---|---|---|
| 8–9 | Can bulk be split into per-cell-type pseudo-cells the model separates? | 16 tissues, 9,232 pseudo-cells (9,214 embedded), 197 cell-type blocks [14 tissues, 8,100 pseudo-cells (8,082 embedded), 165 blocks] |
| gate | Which cell types carry a detectable exercise signal? (10 splits × 10,000 permutations, BH q < 0.05, blocks with median library ≥ 1,000) | 9 hotspots of 151 analysable blocks [11 of 124] |
| 10 | Which genes respond to training dose, per cell type? | 53,084 dose-significant gene × block pairs (FDR_IHW < 0.05; 9,671 distinct genes) [48,522]; single-sex tissues (TESTES, OVARY) cannot have any (Fisher sex combine) |
| 12 | Does the exercise signal survive re-expression as human? | 9 of 9 hotspots meet the registered rule (uncorrected, same cells and seeds as the rat test); 7 of 9 under BH over 151 blocks [11 of 11 by the registered rule]; E.3 atlas anchoring PASS on the 14-tissue hotspots (0.404 vs null 0.084; `translation/ATLAS_PROJECTION_RESULTS.md`) |
| 13 | Does the model name exercise regulators? | 6 GRNs [8]; no TF-specific signal (\|z\| ≥ 2 FDR 1.03, 0/981 TFs beat expression-matched decoys; `deconvolution/GRN_DECOY_NULL_RESULTS.md`); rat–human self-consistency median ρ 0.310 (model self-consistency, not cross-species biology) |
| 14 | Are the hotspots robust to composition and technical covariates? | 9 of 9 RIN/globin ROBUST; 1 hotspot composition-flagged (SKMVL VSMC) [11 of 11; 2 flagged] |

Stages 8–9 are the Aim-2 deconvolution bridge; Stage 10 the cell-type-resolved DE; Stage 12 the
Aim-3a transfer; and Stages 13–14 the Aim-2b/3b mechanism and the hardening checks. Stages 13–14 loop
per cell type / per tissue and skip work whose output already exists (`--force` to re-run); the fast
parallel path is the array jobs in `slurm/analysis/` (gitignored, local), while the orchestrators are
the reproducible sequential entry points. Stage 13 steps 1–2 need a GPU; Stage 14 step 1 is CPU-heavy
(BayesPrism). See `deconvolution/README.md` for per-tissue usage and `deconvolution/setup/SETUP.md` for
first-time setup (environment, data, config).
