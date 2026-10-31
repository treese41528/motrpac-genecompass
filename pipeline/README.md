# Pipeline

Each stage has a `run_stageN.py` orchestrator that validates config + inputs and
runs its steps as subprocesses (`--dry-run` / `--from N` / `-v` supported). Stage
workers live in `NN_*/` here, except the deconvolution and downstream-analysis
stages (8, 9, 10, 12–14), which drive the scripts in `deconvolution/` and
`finetune/genecompass/` in place.

| Stage | Orchestrator | What it does |
|---|---|---|
| 1 | `run_stage1.py` | Data harvesting & QC |
| 2 | `run_stage2.py` | Gene universe & cell QC |
| 3 | `run_stage3.py` | Ortholog mapping |
| 4 | `run_stage4.py` | Gene medians (scatter / gather) |
| 5 | `run_stage5.py` | Reference assembly & tokenization |
| 6 | `run_stage6.py` | Prior-knowledge embeddings |
| 7 | `run_stage7.py` | Fine-tuning |
| **8** | `run_stage8.py` | **Deconvolution** (MoTrPAC bulk → pseudo-cells; drives `deconvolution/`) |
| **9** | `run_stage9.py` | **Tokenize + embed** (pseudo-cells → GeneCompass embeddings) |
| **10** | `run_stage10.py` | **Aim-2 analysis** — per-cell-type DE + pre-registered positive-control comparison (whole-experiment) |
| **12** | `run_stage12.py` | **Cross-species transfer** — re-express rat pseudo-cells in human embedding space |
| **13** | `run_stage13.py` | **Mechanism** — differential GRN (rat + human space) → rat–human self-consistency score → pathway enrichment |
| **14** | `run_stage14.py` | **Hardening** — per-tissue expression-Z purity sweeps, composition-confound + RIN/globin DE robustness |

*(There is no Stage 11; the numbering skips it.)*

**Default configuration (2026-10-31): the v2 model on the refs4 references.** `config/pipeline_config.yaml`
selects the jointly re-annotated references (`data/deconvolution/references_v4/`, BLOOD as a whole-blood
composite) and writes to `data/deconvolution/results_refs4/`, `genecompass_input_v2_refs4/` (+ `_human/`),
`augur_input_v2_refs4/` and `grn_v2_refs4/` (+ `_human/`). Other arms are profiles:
`PIPELINE_PROFILE=config/profiles/july_refs.yaml` (v2 on the July references) and
`config/profiles/april.yaml` (April's model); `config/profiles/refs_v4.yaml` is now a no-op.

| Stage | Question | Current result (refs4; `deconvolution/REFS4_RERUN_RESULTS.md`) |
|---|---|---|
| 8–9 | Can bulk be split into per-cell-type pseudo-cells the model separates? | 14 tissues, 8,100 pseudo-cells (8,082 embedded), 165 cell-type blocks |
| gate | Which cell types carry a detectable exercise signal? (10 splits × 10,000 permutations, BH q < 0.05, blocks with median library ≥ 1,000) | 11 hotspots of 124 analysable blocks |
| 10 | Which genes respond to training dose, per cell type? | 48,522 dose-significant genes (FDR_IHW < 0.05) |
| 12 | Does the exercise signal survive re-expression as human? | 11 of 11 hotspots preserved; E.3 atlas anchoring PASS (0.404 vs null 0.084; `translation/ATLAS_PROJECTION_RESULTS.md`) |
| 13 | Does the model name exercise regulators? | 8 GRNs; no TF-specific signal (\|z\| ≥ 2 FDR 1.03, 0/981 TFs beat expression-matched decoys; `deconvolution/GRN_DECOY_NULL_RESULTS.md`); rat–human self-consistency median ρ 0.310 (model self-consistency, not cross-species biology) |
| 14 | Are the hotspots robust to composition and technical covariates? | 11 of 11 RIN/globin ROBUST; 2 hotspots composition-flagged |

Stages 8–9 are the Aim-2 deconvolution bridge; Stage 10 the cell-type-resolved DE; Stage 12 the
Aim-3a transfer; and Stages 13–14 the Aim-2b/3b mechanism and the hardening checks. Stages 13–14 loop
per cell type / per tissue and skip work whose output already exists (`--force` to re-run); the fast
parallel path is the array jobs in `slurm/analysis/` (gitignored, local), while the orchestrators are
the reproducible sequential entry points. Stage 13 steps 1–2 need a GPU; Stage 14 step 1 is CPU-heavy
(BayesPrism). See `deconvolution/README.md` for per-tissue usage and `deconvolution/setup/SETUP.md` for
first-time setup (environment, data, config).
