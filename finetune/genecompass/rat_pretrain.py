#!/usr/bin/env python
# coding: utf-8
"""
rat_pretrain.py — Fine-tune GeneCompass on rat transcriptomic corpus.

Adapted from vendor/GeneCompass/pretraining/pretrain_genecompass_w_human_mouse_base.py.

Key changes from the original:
  1. Imports rat_load_prior_embedding instead of load_prior_embedding
  2. vocab_size = 53,032 (50,558 + 2,474 T4 rat tokens)
  3. cls_embedding = nn.Embedding(3, 768) — monkey-patched after model creation
  4. Loads from extended checkpoint (rat_model_init.py output)
  5. Reads hyperparameters from config/pipeline_config.yaml finetuning: section
  6. Integrates W&B for training monitoring
  7. All paths resolved through config — no hardcoded paths

IMPORTANT: The vendor GeneCompass submodule is NOT modified. We add
  vendor/GeneCompass to sys.path to import GenecompassPretrainer and
  BertForMaskedLM, then use our own prior embedding loader.

Usage (via torchrun):
  torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0 --master_port=12348 \
    pipeline/07_finetuning/rat_pretrain.py \
    --do_train --save_model --fp16 --gradient_checkpointing

Author: Tim Reese Lab
Date: March 2026
"""

import os
import sys
import pickle
import random
import argparse
import logging

import numpy as np
import torch
import yaml

# ─────────────────────────────────────────────────────────────────────────────
# Project setup: resolve paths before any GeneCompass imports
# ─────────────────────────────────────────────────────────────────────────────

_PROJECT_ROOT = os.environ.get('PIPELINE_ROOT', '.')
_PROJECT_ROOT = os.path.abspath(_PROJECT_ROOT)

# Add vendor GeneCompass to path for GenecompassPretrainer, BertForMaskedLM
_VENDOR_GC = os.path.join(_PROJECT_ROOT, 'vendor', 'GeneCompass')
sys.path.insert(0, _VENDOR_GC)

# Add project root for our pipeline modules
sys.path.insert(0, _PROJECT_ROOT)

# ─────────────────────────────────────────────────────────────────────────────
# Now safe to import GeneCompass modules
# NOTE: pretrainer.py loads h&m_token1000W.pickle at import time (line 61).
# This is a harmless side effect — that module-level token_dictionary is NOT
# used in our training path. GenecompassPretrainer.__init__ accepts our
# rat_token_dict via the token_dictionary kwarg and creates a fresh
# GenecompassPreCollator from it.
# ─────────────────────────────────────────────────────────────────────────────

os.environ["NCCL_DEBUG"] = "INFO"
os.environ["OMPI_MCA_opal_cuda_support"] = "true"
os.environ["CONDA_OVERRIDE_GLIBC"] = "2.56"

from transformers import BertConfig, TrainingArguments, TrainerCallback
from datasets import load_from_disk, disable_caching
disable_caching()

# Import from same directory (pipeline/07_finetuning/)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)

from genecompass import GenecompassPretrainer, BertForMaskedLM
from torch import nn

from rat_load_prior_embedding import build_knowledges_dict

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# W&B callback for custom metrics
# ─────────────────────────────────────────────────────────────────────────────

class WandbMetricsCallback(TrainerCallback):
    """Log emb_warmup_alpha and phase info to W&B on each logging step."""

    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        if logs is None or model is None:
            return
        try:
            import wandb
            if wandb.run is None:
                return
            extra = {}
            if hasattr(model, 'emb_warmup'):
                extra["emb_warmup_alpha"] = model.emb_warmup.alpha.item()
            if extra:
                wandb.log(extra, step=state.global_step)
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# Config loader
# ─────────────────────────────────────────────────────────────────────────────

def load_pipeline_config(config_path: str = None) -> dict:
    """
    Load finetuning config. Supports two formats:

    1. Standalone config (finetune/genecompass/configs/rat_4gpu.yaml):
       Has top-level sections: model, data, training, initialization, logging, output

    2. Legacy pipeline_config.yaml with a flat finetuning: section

    Returns a flat dict with all keys merged for uniform access.
    """
    if config_path is None:
        config_path = os.path.join(_PROJECT_ROOT, 'config', 'pipeline_config.yaml')
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Standalone config: merge nested sections into flat dict,
    # but preserve phase1/phase2 as nested dicts for get_phase_config()
    if 'model' in config or 'training' in config:
        flat = {}
        for section in ('model', 'data', 'training', 'initialization',
                        'logging', 'output', 'distributed'):
            flat.update(config.get(section, {}))
        # Keep phase sections intact (not flattened)
        for phase in ('phase1', 'phase2'):
            if phase in config:
                flat[phase] = config[phase]
        return flat

    # Legacy: flat finetuning: section inside pipeline_config.yaml
    ft = config.get('finetuning', {})
    if not ft:
        logger.warning("No recognized config sections found — using defaults")
    return ft


def resolve_path(path: str) -> str:
    """Resolve a path relative to project root if not absolute."""
    if os.path.isabs(path):
        return path
    return os.path.join(_PROJECT_ROOT, path)


# ─────────────────────────────────────────────────────────────────────────────
# Freeze / unfreeze
# ─────────────────────────────────────────────────────────────────────────────

def freeze_encoder(model, world_rank: int = 0) -> int:
    """
    Freeze all pre-trained components. Only new/extended parameters train.

    Frozen:
      - bert.encoder (12 transformer layers)
      - bert.embeddings: prior knowledge projections (linear1), concat layer,
        LayerNorm, position embeddings, token_type embeddings
      - cls4value head (expression value prediction)

    Trainable:
      - bert.embeddings.word_embeddings (all rows — T1-T3 adapt, T4 learn)
      - bert.cls_embedding (species embeddings — row 2 learns)
      - cls head (gene ID prediction — biases for new tokens)

    Returns number of trainable parameters.
    """
    # First freeze everything
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze word embeddings
    model.bert.embeddings.word_embeddings.weight.requires_grad = True

    # Unfreeze cls_embedding (species tokens)
    model.bert.cls_embedding.weight.requires_grad = True

    # Unfreeze MLM prediction head (cls) — decoder.weight is tied to
    # word_embeddings, but bias and transform need to be trainable
    for param in model.cls.parameters():
        param.requires_grad = True

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())

    if world_rank == 0:
        logger.info(
            f"Phase 1 FREEZE: {n_trainable:,} trainable / {n_total:,} total "
            f"({n_trainable / n_total * 100:.1f}%)"
        )
        # Log which parameter groups are trainable
        for name, param in model.named_parameters():
            if param.requires_grad:
                logger.info(f"  trainable: {name} {list(param.shape)}")

    return n_trainable


def unfreeze_all(model, world_rank: int = 0) -> int:
    """Unfreeze all parameters for end-to-end fine-tuning (phase 2)."""
    for param in model.parameters():
        if param.is_floating_point() or param.is_complex():
            param.requires_grad = True

    n_total = sum(p.numel() for p in model.parameters() if p.is_floating_point())
    if world_rank == 0:
        logger.info(f"Phase 2 UNFREEZE: all {n_total:,} parameters trainable")

    return n_total


def get_phase_config(ft_config: dict, phase: int) -> dict:
    """
    Merge phase-specific settings over shared training settings.

    Phase 1 reads from 'phase1:' section, phase 2 from 'phase2:'.
    Phase-specific keys override shared 'training:' keys.
    """
    # Start with shared training settings
    merged = dict(ft_config)

    # Overlay phase-specific settings
    phase_key = f'phase{phase}'
    phase_cfg = {}

    # In standalone config, phase settings are already flattened if the
    # load_pipeline_config flattener ran. But if the raw YAML was loaded
    # with nested structure preserved, we need to dig:
    if phase_key in ft_config:
        phase_cfg = ft_config[phase_key]
    # Also check for flattened keys like 'freeze_encoder', 'resume_from'
    # which would come from the CLI --phase override

    merged.update(phase_cfg)
    return merged


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    # ── Load pipeline config ─────────────────────────────────────────────
    ft_config = load_pipeline_config(args.config)

    # ── Merge phase-specific settings ────────────────────────────────────
    phase = args.phase
    if phase in (1, 2):
        ft_config = get_phase_config(ft_config, phase)

    # ── Set seeds ────────────────────────────────────────────────────────
    seed_num = args.seed_num
    seed_val = args.seed_val
    random.seed(seed_num)
    np.random.seed(seed_num)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    # ── Distributed setup ────────────────────────────────────────────────
    n_gpu = torch.cuda.device_count()
    args.local_rank = int(os.environ.get("LOCAL_RANK", 0))
    args.world_rank = int(os.environ.get("RANK", 0))
    args.world_size = int(os.environ.get("WORLD_SIZE", 1))

    if args.world_rank == 0:
        logger.info(f"GPUs: {n_gpu}, World size: {args.world_size}")

    # ── Resolve paths from config ────────────────────────────────────────
    checkpoint_dir = resolve_path(
        ft_config.get('base_checkpoint',
                      'vendor/GeneCompass/pretrained_models/GeneCompass_Base')
    )
    init_checkpoint_dir = resolve_path(
        ft_config.get('init_checkpoint', 'data/models/rat_genecompass_init')
    )
    token_dict_path = resolve_path(
        args.token_dict_path or ft_config.get('token_dict_path',
            'data/training/ortholog_mappings/rat_human_mouse_tokens.pickle')
    )
    dataset_dir = resolve_path(
        args.dataset_directory or ft_config.get('dataset_directory',
            'data/training/tokenized_corpus/dataset')
    )
    output_dir = resolve_path(
        args.output_directory or ft_config.get('output_directory',
            'data/models/rat_genecompass_finetuned')
    )
    stage6_dir = resolve_path(
        ft_config.get('stage6_dir', 'data/training/prior_knowledge')
    )
    homologous_gene_path = resolve_path(
        ft_config.get('homologous_gene_path',
            'vendor/GeneCompass/prior_knowledge/homologous_hm_token.pickle')
    )

    # ── Hyperparameters ──────────────────────────────────────────────────
    with open(token_dict_path, 'rb') as f:
        token_dictionary = pickle.load(f)
    vocab_size = max(token_dictionary.values()) + 1
    num_species = ft_config.get('num_species', 3)
    max_lr = ft_config.get('max_learning_rate', 1e-5)
    epochs = args.num_train_epochs or ft_config.get('num_train_epochs', 2)
    warmup_steps = ft_config.get('warmup_steps', 5000)
    emb_warmup_steps = ft_config.get('emb_warmup_steps', 10000)
    batch_size = ft_config.get('train_batch_size_per_gpu', 10)
    weight_decay = ft_config.get('weight_decay', 0.01)
    lr_scheduler = ft_config.get('lr_scheduler_type', 'linear')
    logging_steps = ft_config.get('logging_steps', 50)
    save_steps = ft_config.get('save_steps', 50000)
    coexp_rescale = ft_config.get('coexp_rescale_factor', 3.9)

    # ── Output directories ───────────────────────────────────────────────
    run_name = args.run_name or ft_config.get('run_name', 'rat_finetune')
    training_output_dir = os.path.join(output_dir, 'models', run_name)
    logging_dir = os.path.join(output_dir, 'runs', run_name)
    model_output_dir = os.path.join(training_output_dir, 'models')

    # Check for existing model
    model_output_file = os.path.join(model_output_dir, 'pytorch_model.bin')
    if os.path.isfile(model_output_file):
        raise RuntimeError(f"Model already saved: {model_output_file}")

    if args.world_rank == 0:
        os.makedirs(training_output_dir, exist_ok=True)
        os.makedirs(model_output_dir, exist_ok=True)

    # ── Load token dictionary ────────────────────────────────────────────
    with open(token_dict_path, 'rb') as f:
        token_dictionary = pickle.load(f)
    assert len(token_dictionary) == vocab_size, \
        f"Token dict has {len(token_dictionary)} entries, expected {vocab_size}"

    if args.world_rank == 0:
        logger.info(f"Token dictionary: {len(token_dictionary)} entries")

    # ── Load extended checkpoint state dict ──────────────────────────────
    init_model_bin = os.path.join(init_checkpoint_dir, 'pytorch_model.bin')
    if args.world_rank == 0:
        logger.info(f"Loading extended checkpoint: {init_model_bin}")
    extended_state = torch.load(init_model_bin, map_location='cpu', weights_only=False)

    # ── Build prior knowledge dict from extended state ───────────────────
    # Extract the prior knowledge tensors that rat_model_init.py already built
    knowledges = {
        'promoter': extended_state['bert.embeddings.promoter_knowledge'],
        'co_exp': extended_state['bert.embeddings.co_exp_knowledge'],
        'gene_family': extended_state['bert.embeddings.gene_family_knowledge'],
        'peca_grn': extended_state['bert.embeddings.peca_grn_knowledge'],
    }

    # Build homologous_gene_human2mouse from homologous_hm_token.pickle
    with open(homologous_gene_path, 'rb') as f:
        raw_homolog = pickle.load(f)
        knowledges['homologous_gene_human2mouse'] = {v: k for k, v in raw_homolog.items()}

    # ── Model configuration ──────────────────────────────────────────────
    config = {
        "hidden_size": 768,
        "num_hidden_layers": 12,
        "initializer_range": 0.02,
        "layer_norm_eps": 1e-12,
        "attention_probs_dropout_prob": 0.02,
        "hidden_dropout_prob": 0.02,
        "intermediate_size": 3072,
        "hidden_act": "gelu",
        "max_position_embeddings": 2048,
        "model_type": "bert",
        "num_attention_heads": 12,
        "pad_token_id": token_dictionary.get("<pad>"),
        "vocab_size": vocab_size,
        "use_values": True,
        "use_promoter": True,
        "use_co_exp": True,
        "use_gene_family": True,
        "use_peca_grn": True,
        "warmup_steps": warmup_steps,
        "emb_warmup_steps": emb_warmup_steps,
        "use_cls_token": True,
    }

    model_config = BertConfig(**config)
    model = BertForMaskedLM(model_config, knowledges=knowledges)

    # ── Monkey-patch cls_embedding for 3 species ─────────────────────────
    # BertModel.__init__ hardcodes nn.Embedding(2, hidden_size).
    # We replace it with nn.Embedding(3, hidden_size) before loading weights.
    if num_species != 2:
        model.bert.cls_embedding = nn.Embedding(num_species, 768)
        if args.world_rank == 0:
            logger.info(f"cls_embedding patched: nn.Embedding({num_species}, 768)")

    # ── Load extended weights ────────────────────────────────────────────
    # strict=False because the model was just constructed with random weights
    # for the knowledge buffers and embeddings — we're replacing them all
    missing, unexpected = model.load_state_dict(extended_state, strict=False)
    if args.world_rank == 0:
        if missing:
            logger.warning(f"Missing keys on load: {missing}")
        if unexpected:
            logger.warning(f"Unexpected keys on load: {unexpected}")
        logger.info("Extended state dict loaded successfully")

    # ── Declare tied weights for transformers ≥4.57 save compatibility ───
    from patches.transformers_compat import declare_tied_weights
    declare_tied_weights(model)

    # ── Preserve emb_warmup alpha from pre-trained model ─────────────────
    # Pre-trained model has alpha=1.0 (fully converged). Don't reset it.
    if hasattr(model, 'emb_warmup'):
        if args.world_rank == 0:
            logger.info(
                f"emb_warmup: alpha={model.emb_warmup.alpha.item():.4f}, "
                f"steps={model.emb_warmup.steps.item()}"
            )

    model = model.train()

    # ── Apply freeze strategy ────────────────────────────────────────────
    do_freeze = ft_config.get('freeze_encoder', False)
    if do_freeze:
        freeze_encoder(model, world_rank=args.world_rank)
    else:
        if args.world_rank == 0:
            n_total = sum(p.numel() for p in model.parameters())
            logger.info(f"No freeze: all {n_total:,} parameters trainable")

    # ── W&B setup ────────────────────────────────────────────────────────
    wandb_project = ft_config.get('wandb_project', 'motrpac-genecompass-rat')
    os.environ["WANDB_PROJECT"] = wandb_project

    # ── Training arguments ───────────────────────────────────────────────
    # When encoder is frozen, DDP must find unused parameters (frozen
    # params produce None gradients). Performance cost is small.
    save_total_limit = ft_config.get('save_total_limit', 3)

    training_args = TrainingArguments(
        run_name=run_name,
        fp16=args.fp16,
        fp16_opt_level="O1",
        ddp_find_unused_parameters=do_freeze,
        gradient_checkpointing=args.gradient_checkpointing,
        report_to="wandb",
        dataloader_num_workers=args.dataloader_num_workers,
        learning_rate=max_lr,
        do_train=args.do_train,
        do_eval=args.do_eval,
        group_by_length=True,
        length_column_name="length",
        disable_tqdm=False,
        lr_scheduler_type=lr_scheduler,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        per_device_train_batch_size=batch_size,
        num_train_epochs=epochs,
        save_strategy="steps" if args.save_model else "no",
        save_steps=save_steps if args.save_model else None,
        save_total_limit=save_total_limit,
        max_grad_norm=1.0,
        save_safetensors=False,
        logging_steps=logging_steps,
        output_dir=training_output_dir,
        logging_dir=logging_dir,
    )

    # ── Load training dataset ────────────────────────────────────────────
    train_dataset = load_from_disk(dataset_dir)
    example_lengths_file = os.path.join(dataset_dir, 'sorted_length.pickle')

    if args.world_rank == 0:
        logger.info(f"Dataset: {len(train_dataset)} cells from {dataset_dir}")
        logger.info(f"Starting training: {epochs} epochs, LR={max_lr}, batch={batch_size}/GPU")

    # ── W&B init metadata (rank 0 only) ──────────────────────────────────
    if args.world_rank == 0:
        try:
            import wandb
            wandb.init(
                project=wandb_project,
                name=run_name,
                config={
                    "n_rat_cells": len(train_dataset),
                    "n_rat_genes": vocab_size - 2,  # minus pad and mask
                    "n_t4_new_tokens": vocab_size - 50558,
                    "base_model": "GeneCompass-Base",
                    "total_params": sum(
                        p.numel() for p in model.parameters()
                    ),
                    "species": ["human", "mouse", "rat"],
                    "phase": phase,
                    "freeze_encoder": do_freeze,
                    "max_learning_rate": max_lr,
                    "num_epochs": epochs,
                    "batch_size_per_gpu": batch_size,
                    "warmup_steps": warmup_steps,
                    "coexp_rescale_factor": coexp_rescale,
                },
            )
        except Exception as e:
            logger.warning(f"W&B init failed (non-fatal): {e}")

    # ── Apply transformers 4.57 compatibility patches ──────────────────
    from patches.transformers_compat import apply_patches
    apply_patches()

    # ── Create trainer ───────────────────────────────────────────────────
    trainer = GenecompassPretrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        example_lengths_file=example_lengths_file,
        token_dictionary=token_dictionary,
        callbacks=[WandbMetricsCallback()],
    )

    # ── Train ────────────────────────────────────────────────────────────
    resume_from = args.resume_from or ft_config.get('resume_from', None)
    if resume_from:
        resume_from = resolve_path(resume_from)
        if args.world_rank == 0:
            logger.info(f"Resuming from checkpoint: {resume_from}")
        trainer.train(resume_from_checkpoint=resume_from)
    else:
        trainer.train()

    # ── Save final model ─────────────────────────────────────────────────
    if args.save_model:
        trainer.save_model(model_output_dir)
        if args.world_rank == 0:
            logger.info(f"Model saved: {model_output_dir}")

    if args.world_rank == 0:
        logger.info("Stage 7 fine-tuning complete")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tune GeneCompass on rat transcriptomic corpus"
    )
    # Config
    parser.add_argument("--config", default=None, type=str,
                        help="Path to config YAML")
    parser.add_argument("--phase", type=int, default=1, choices=[1, 2],
                        help="Training phase: 1=freeze encoder, 2=unfreeze all")
    parser.add_argument("--run_name", default=None, type=str)
    parser.add_argument("--resume_from", default=None, type=str,
                        help="Resume from HF Trainer checkpoint directory")

    # Seeds
    parser.add_argument("--seed_num", type=int, default=0)
    parser.add_argument("--seed_val", type=int, default=42)

    # Paths (override config)
    parser.add_argument("--token_dict_path", default=None, type=str)
    parser.add_argument("--dataset_directory", default=None, type=str)
    parser.add_argument("--output_directory", default=None, type=str)

    # Training (override config)
    parser.add_argument("--num_train_epochs", default=None, type=int)
    parser.add_argument("--dataloader_num_workers", default=0, type=int)

    # Flags
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_eval", action="store_true")
    parser.add_argument("--save_model", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")

    args = parser.parse_args()
    main(args)