#!/usr/bin/env python
# coding: utf-8
"""
monitor.py — Weights & Biases monitoring setup for Stage 7.

Provides:
  - W&B initialization with run metadata
  - Custom metric logging helpers for GeneCompass dual-objective training
  - Gradient norm tracking
  - Checkpoint artifact logging

Metrics logged per step:
  - id_loss:           Gene ID (token) prediction loss (CrossEntropy)
  - value_loss:        Expression value prediction loss (MSE)
  - loss:              Combined (0.2 × id_loss + 0.8 × value_loss)
  - emb_warmup_alpha:  Prior knowledge warmup factor (should stay 1.0)
  - grad_norm:         Gradient L2 norm (catch instability)
  - learning_rate:     Current LR from scheduler

W&B project: motrpac-genecompass-rat
Credentials: ~/.netrc (pre-configured)
Mode: online (confirmed working from Gilbreth compute nodes)

Author: Tim Reese Lab
Date: March 2026
"""

import logging
import os
from typing import Dict, Optional

logger = logging.getLogger(__name__)


def init_wandb(
    project: str = "motrpac-genecompass-rat",
    run_name: str = "rat_finetune",
    config: Optional[Dict] = None,
) -> bool:
    """
    Initialize W&B run. Returns True on success, False on failure.

    Non-fatal: training continues even if W&B fails to initialize.
    """
    try:
        import wandb

        os.environ["WANDB_PROJECT"] = project

        wandb.init(
            project=project,
            name=run_name,
            config=config or {},
            tags=["stage7", "rat", "fine-tuning"],
        )
        logger.info(f"W&B initialized: project={project}, run={run_name}")
        return True

    except Exception as e:
        logger.warning(f"W&B initialization failed (non-fatal): {e}")
        return False


def log_training_step(
    step: int,
    loss: float,
    id_loss: float,
    value_loss: float,
    learning_rate: float,
    emb_warmup_alpha: float = 1.0,
    grad_norm: Optional[float] = None,
    extra: Optional[Dict] = None,
) -> None:
    """Log a single training step to W&B."""
    try:
        import wandb

        metrics = {
            "train/loss": loss,
            "train/id_loss": id_loss,
            "train/value_loss": value_loss,
            "train/learning_rate": learning_rate,
            "train/emb_warmup_alpha": emb_warmup_alpha,
        }
        if grad_norm is not None:
            metrics["train/grad_norm"] = grad_norm

        if extra:
            metrics.update(extra)

        wandb.log(metrics, step=step)

    except Exception:
        pass  # Silent fail — don't interrupt training


def compute_grad_norm(model) -> float:
    """Compute total gradient L2 norm across all parameters."""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    return total_norm ** 0.5


def log_model_artifact(model_dir: str, name: str = "rat-genecompass") -> None:
    """Log a saved model directory as a W&B artifact."""
    try:
        import wandb

        artifact = wandb.Artifact(name, type="model")
        artifact.add_dir(model_dir)
        wandb.log_artifact(artifact)
        logger.info(f"W&B artifact logged: {name} from {model_dir}")

    except Exception as e:
        logger.warning(f"W&B artifact logging failed: {e}")