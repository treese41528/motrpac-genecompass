# Pipeline

Each stage has a `run_stageN.py` orchestrator that validates config + inputs and
runs its steps as subprocesses (`--dry-run` / `--from N` / `-v` supported). Stage
workers live in `NN_*/` here, except the deconvolution stages (8–9), which drive
the scripts in `deconvolution/` in place.

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

Stages 8–9 are the Aim-2 deconvolution bridge; see `deconvolution/README.md` for the
per-tissue usage and the analysis scripts deliberately *not* in the pipeline, and
`deconvolution/setup/SETUP.md` for first-time setup (environment, data, config).
