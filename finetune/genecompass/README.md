# Stage 7: Fine-Tuning GeneCompass for Rat Transcriptomics

## Overview

Stage 7 adapts GeneCompass (12-layer BERT, GeneCompass_Base) to rat transcriptomics by continuing
masked-language-model pre-training on a rat/human/mouse mix. The working model is **v2**
(`docs/RETRAIN_V2_RESULTS.md`): 6,803,206 rat cells (584 QC files kept by `config/corpus_v2_plan.tsv`)
interleaved 40/45/15 with GeneCompass's human and mouse sets (5,000,000 and 4,969,697 cells; the folders
are named `randsel_500w_*`, and upstream's table labels that download "0.5M"). The vocabulary grows from
50,558 to **55,275** tokens: 2,243 T3 rat tokens (warm-started from their human/mouse ortholog) and 2,474
T4 rat-specific tokens (random rows). April's model (`configs/rat_finetune.yaml`) has the same 55,275-token
vocabulary (`vocab_size` in its checkpoints' `config.json`); this page said 53,032 until 2026-11-14, the
size of an earlier design that no model was trained on.

## What has been tested, and what is known about the corpus (2026-11-15)

The question for this stage: does the model learn rat without losing human and mouse, and does what it learned
hold on rat data it has not seen?

| Check | Result | Source |
|---|---|---|
| Rat learned, human and mouse kept (pre-registered, in sample) | Rat gene-identity loss 7.13 to 3.06; human +0.05% and mouse +1.32% against the base model (limit +10%) | `docs/RETRAIN_V2_RESULTS.md`, C1–C2 |
| Generalisation to rat studies outside the corpus (registered) | On 66,000 cells of 33 studies, loss 1.046 times the in-sample probe's (bar 1.10). Per study 0.93 to 1.20; rat-only tokens 1.18 | `docs/audit/remediation/STEP4_RESULTS.md`, V3 |
| Mitochondrial fraction of the training cells | 10.4% of the 6,803,206 cells are above 15% (GeneCompass's cut, which was configured and never applied); the outlier rule was inert in 395 of 584 files | `docs/audit/remediation/STEP5_RESULTS.md`, M1 |
| Gene IDs carried over from earlier annotations | 1.04% of counts in ID-keyed files sit under another gene's token; a few highly expressed genes (Tpt1, Rps3, Rps20, Rpl23a, Rpl9, Rps27a) were seen under rat-only tokens | `STEP5_RESULTS.md`, M3 |
| Series that never entered the corpus | 79 confirmed rat single-cell series; 33 of them are the held-out set above | `STEP5_RESULTS.md`, M2 |

None of the corpus findings was repaired: the decision was not to retrain. The repairs exist as switches in
`config/pipeline_config.yaml` that are off, for a future corpus. The held-out test is on single cells of new
studies and says nothing about the pseudo-cells of Stages 9 to 13. The v2 initial checkpoint was built without
a seed and cannot be regenerated.

## Quick start (v2 config: `configs/rat_finetune_v2.yaml`)

```bash
python pipeline/run_stage7.py --dry-run             # validate prerequisites, print every step
python pipeline/run_stage7.py --from 1 --to 3       # init (seeded), training mix, evaluation probe
python pipeline/run_stage7.py --from 4 --phase 1    # train phase 1 (frozen encoder); rerun to resume
python pipeline/run_stage7.py --from 4 --phase 2    # train phase 2 (all layers)
```

Step 4 runs `scripts/train_stage7.sh` (torchrun, `distributed.nproc_per_node` GPUs, fp16, gradient
checkpointing, auto-resume from the newest `checkpoint-*`). On a cluster, wrap that script in your
scheduler's batch file, or pass `--sbatch <your wrapper>`. The individual commands are in the root
`README.md` ("Rebuilding the v2 model").

For exact v2 reproduction the init (`data/models/rat_genecompass_init_v2`, built before
`initialization.seed` existed, so not regenerable byte for byte), the mix and the checkpoints are
deposited; see `docs/repro_stages1_7/REPRODUCTION.md`.

## Files

| File | Purpose |
|---|---|
| `build_rat_token_dictionary.py` | Merges GeneCompass's human/mouse tokens with the Stage 3 rat tokens (55,275 entries for v2) |
| `rat_load_prior_embedding.py` | Builds the [vocab, 768] prior-knowledge tensors (Stage 6 rows only where the init policy allows) |
| `rat_model_init.py` | Extends GeneCompass_Base to the rat vocabulary (`--seed` for the random rows) |
| `build_mixed_species_dataset.py` | The rat/human/mouse training mix with per-row source references |
| `eval_mlm.py` | `build-probe`: the fixed evaluation probe; `eval`: masked-LM scores on it |
| `rat_pretrain.py` | Training script (torchrun); copies the training YAML and a provenance record into each run folder |
| `monitor.py` | W&B setup and custom metric helpers |

The orchestrator is `pipeline/run_stage7.py`; the launch is `scripts/train_stage7.sh`.

## Architecture changes (v2)

| Component | Original | Extended |
|---|---|---|
| `word_embeddings` / tied decoder | [50558, 768] | [55275, 768] |
| `cls_embedding` | [2, 768] | [3, 768] (rat = 2, from mouse + noise) |
| `prior_knowledge` buffers | [50558, 768] ×4 | [55275, 768] ×4 |
| `homologous_index` | [50558] | [55275] (T4 = identity) |
| prediction biases | [50558] | [55275] |

## Key design decisions

1. **T1–T3 prior knowledge comes from the pre-trained model** (T3 rows copy their ortholog's rows); only
   T4 rows may come from Stage 6, and v2 leaves the T4 promoter and co-expression rows zero
   (`initialization.t4_zero_priors`), so v2's init takes only 36 T4 GRN rows from Stage 6.
2. **Vendor submodule unmodified** — all rat code lives here; `cls_embedding` is extended after model
   construction.
3. **Weight tying maintained** — `cls.predictions.decoder.weight` and `word_embeddings.weight` are the
   same tensor.

## Configuration

All hyperparameters are in `configs/rat_finetune_v2.yaml` (sections `model`, `data`, `phase1`, `phase2`,
`training`, `initialization`, `mixing`, `probe`, `logging`, `output`, `distributed`). CLI arguments to
`rat_pretrain.py` override them.

## Runtime (v2, one node with 4 × A100-40GB)

Phase 1: 517,793 steps (job 11917121, ended 2026-10-15; started after the 2026-10-08 init). Phase 2:
591,763 steps (ended 2026-10-23; resumed once from `checkpoint-470000` after a node failure). The mix build
takes about 4.7 h and the probe about 1.6 h (timings from the 2026-10-31 reproduction).

W&B project: `reese18-purdue-university/motrpac-genecompass-rat`.
