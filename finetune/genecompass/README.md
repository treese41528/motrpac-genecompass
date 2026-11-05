# Stage 7: Fine-Tuning GeneCompass for Rat Transcriptomics

## Overview

Stage 7 adapts GeneCompass (12-layer BERT, GeneCompass_Base) to rat transcriptomics by continuing
masked-language-model pre-training on a rat/human/mouse mix. The working model is **v2**
(`docs/RETRAIN_V2_RESULTS.md`): 6,803,206 rat cells (584 QC files kept by `config/corpus_v2_plan.tsv`)
interleaved 40/45/15 with GeneCompass's 0.5M human and mouse cells. The vocabulary grows from 50,558 to
**55,275** tokens: 2,243 T3 rat tokens (warm-started from their human/mouse ortholog) and 2,474 T4
rat-specific tokens (random rows). April's model (`configs/rat_finetune.yaml`) had 53,032.

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
