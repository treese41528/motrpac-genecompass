# Stage 7: Fine-Tuning GeneCompass for Rat Transcriptomics

## Overview

Stage 7 adapts GeneCompass (326.8M parameters, 12-layer BERT) for rat
transcriptomics by continuing masked language model pre-training on our
9,483,420-cell rat corpus. The model grows by 2.9% to accommodate 2,474
rat-specific T4 gene tokens.

## Quick Start

```bash
# 1. Validate prerequisites and extend checkpoint
python pipeline/07_finetuning/run_stage7.py --dry-run

# 2. Run (Step 1: init checkpoint, Step 2: submit SLURM job)
python pipeline/07_finetuning/run_stage7.py

# Or run steps individually:
python pipeline/07_finetuning/rat_model_init.py \
    --checkpoint vendor/GeneCompass/pretrained_models/GeneCompass_Base \
    --rat-token-dict data/training/ortholog_mappings/rat_tokens.pickle \
    --stage6-dir data/training/prior_knowledge \
    --homologous-gene-path vendor/GeneCompass/prior_knowledge/homologous_hm_token.pickle \
    --output-dir data/models/rat_genecompass_init

sbatch slurm/stage7.slurm
```

## Files

| File | Purpose |
|---|---|
| `rat_load_prior_embedding.py` | Builds [53032, 768] prior knowledge tensors from Stage 6 |
| `rat_model_init.py` | Extends GeneCompass_Base checkpoint to rat vocabulary |
| `rat_pretrain.py` | Main training script (torchrun-compatible) |
| `run_stage7.py` | Orchestrator: validates prerequisites → init → submit |
| `monitor.py` | W&B setup and custom metric helpers |

## Architecture Changes

| Component | Original | Extended |
|---|---|---|
| `word_embeddings` | [50558, 768] | [53032, 768] |
| `cls_embedding` | [2, 768] | [3, 768] (rat=2) |
| `prior_knowledge` buffers | [50558, 768] ×4 | [53032, 768] ×4 |
| `homologous_index` | [50558] | [53032] (T4=identity) |
| `prediction biases` | [50558] | [53032] |
| **Total parameters** | 326.8M | 336.4M (+2.9%) |

## Key Design Decisions

1. **T1-T3 prior knowledge preserved from pre-trained model** — only T4 rows
   come from Stage 6. The pre-trained embedding values are calibrated to the
   learned projection layers; overwriting them would degrade performance.

2. **Vendor submodule unmodified** — all changes in `pipeline/07_finetuning/`.
   `cls_embedding` extended via monkey-patch after model construction.

3. **Co-expression rescaling** — T4 co-exp norms ~4.8 vs model expectation
   ~18.6. Rescaled by 3.9× to match the projection layer's input scale.

4. **Weight tying maintained** — `cls.predictions.decoder.weight` and
   `word_embeddings.weight` are the same tensor.

## Monitoring

W&B project: `reese18-purdue-university/motrpac-genecompass-rat`

Metrics per step: `id_loss`, `value_loss`, `loss`, `emb_warmup_alpha`,
`grad_norm`, `learning_rate`

## Configuration

All hyperparameters in `config/pipeline_config.yaml` under `finetuning:`.
CLI arguments override config values.

## Estimated Runtime

~18 hours/epoch on 4× A100-40GB → ~54 hours for 3 epochs.