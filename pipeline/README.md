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
| **13** | `run_stage13.py` | **Mechanism** — differential GRN (rat + human space) → cross-species conservation → pathway enrichment |
| **14** | `run_stage14.py` | **Hardening** — per-tissue expression-Z purity sweeps, composition-confound + RIN/globin DE robustness |

*(There is no Stage 11; the numbering skips it.)*

Stages 8–9 are the Aim-2 deconvolution bridge; Stage 10 the cell-type-resolved DE; Stage 12 the
Aim-3a transfer; and Stages 13–14 the Aim-2b/3b mechanism and the hardening checks. Stages 13–14 loop
per cell type / per tissue and skip work whose output already exists (`--force` to re-run); the fast
parallel path is the array jobs in `slurm/analysis/` (gitignored, local), while the orchestrators are
the reproducible sequential entry points. Stage 13 steps 1–2 need a GPU; Stage 14 step 1 is CPU-heavy
(BayesPrism). See `deconvolution/README.md` for per-tissue usage and `deconvolution/setup/SETUP.md` for
first-time setup (environment, data, config).
