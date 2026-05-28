#!/usr/bin/env python3
"""
Forgetting Progression Analysis: GC Base vs Rat Checkpoints on Human / Mouse / Rat

Evaluates the original GeneCompass Base and every rat fine-tuning checkpoint
on all three species corpora to quantify catastrophic forgetting progression
across the training trajectory.

Checkpoints are AUTO-DISCOVERED from the models directory tree — no hardcoded
paths. The script scans:
  1. GC Base        → vendor/GeneCompass/pretrained_models/GeneCompass_Base
  2. Rat Init       → data/models/rat_genecompass_init
  3. All run dirs   → data/models/rat_genecompass_finetuned/models/*/
     Within each:   checkpoint-NNNNN/ and models/ (final snapshot)

Phase is inferred from directory name ("phase1" → phase1, "phase2" → phase2).
Step is parsed from checkpoint-NNNNN. Final step for models/ is read from the
highest sibling checkpoint's trainer_state.json.
Vocab size is read from each checkpoint's config.json when available.

Evaluation protocol (matches eval_baseline.py):
  - Sample N cells from each corpus
  - Apply 15% random masking: replace token IDs with mask token (ID=1),
    zero expression values at masked positions
  - Forward pass with species token
  - Record id_loss (CrossEntropy on gene identity) and value_loss (MSE on expression)

Output:
  - forgetting_results.json:  structured results for all model x corpus combinations
  - forgetting_summary.tsv:   tabular summary for quick inspection
  - encoder_drift.json:       per-layer L2 drift from GC Base for each checkpoint
  - forgetting_report.md:     markdown narrative report

Usage:
  python forgetting_analysis.py [--n-cells 5000] [--mask-prob 0.15] [--seed 42]
                                [--output-dir reports/forgetting]

Cluster:  Gilbreth, 1x A100-40GB (eval only, no training)
Account:  reese18
Runtime:  ~30-60 min depending on number of checkpoints
"""

import argparse
import json
import logging
import os
import pickle
import re
import sys
import time
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from datasets import load_from_disk

# ---------------------------------------------------------------------------
# Project paths -- all relative to PROJECT_ROOT
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path("/depot/reese18/apps/motrpac-genecompass")

# Add vendor to path for GeneCompass imports
sys.path.insert(0, str(PROJECT_ROOT / "vendor" / "GeneCompass"))
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MASK_TOKEN_ID = 1
PAD_TOKEN_ID = 0

# Species codes used in GeneCompass
SPECIES_HUMAN = 0
SPECIES_MOUSE = 1
SPECIES_RAT = 2

# Corpora locations
CORPORA = {
    "human": {
        "path": PROJECT_ROOT / "vendor/GeneCompass/data/randsel_500w_human",
        "species": SPECIES_HUMAN,
        "label": "Human 5M",
    },
    "mouse": {
        "path": PROJECT_ROOT / "vendor/GeneCompass/data/randsel_500w_mouse",
        "species": SPECIES_MOUSE,
        "label": "Mouse 5M",
    },
    "rat": {
        "path": PROJECT_ROOT / "data/training/tokenized_corpus/dataset",
        "species": SPECIES_RAT,
        "label": "Rat 9.48M",
    },
}

# Anchors: fixed-location models that are always included
GC_BASE_PATH = PROJECT_ROOT / "vendor/GeneCompass/pretrained_models/GeneCompass_Base"
RAT_INIT_PATH = PROJECT_ROOT / "data/models/rat_genecompass_init"
FINETUNED_ROOT = PROJECT_ROOT / "data/models/rat_genecompass_finetuned/models"

# Default vocab sizes (used as fallback when config.json is absent)
GC_BASE_VOCAB = 50558
RAT_EXTENDED_VOCAB = 55275

# Phase detection patterns (applied to run directory name)
PHASE_PATTERNS = [
    (re.compile(r"phase1", re.IGNORECASE), "phase1"),
    (re.compile(r"phase2", re.IGNORECASE), "phase2"),
]

CHECKPOINT_RE = re.compile(r"^checkpoint-(\d+)$")


# ---------------------------------------------------------------------------
# Checkpoint auto-discovery
# ---------------------------------------------------------------------------
def _read_vocab_size(ckpt_path: Path) -> int | None:
    """Read vocab_size from a checkpoint's config.json, if present."""
    config_file = ckpt_path / "config.json"
    if config_file.exists():
        try:
            with open(config_file) as f:
                return json.load(f).get("vocab_size")
        except (json.JSONDecodeError, OSError):
            pass
    return None


def _read_final_step(run_dir: Path) -> int | None:
    """
    Infer the final training step for a run's models/ directory.

    Strategy:
      1. Read trainer_state.json from the highest-numbered sibling checkpoint.
      2. Fall back to highest checkpoint step + 1.
    """
    # Find all sibling checkpoint dirs
    ckpt_steps = []
    for child in sorted(run_dir.iterdir()):
        m = CHECKPOINT_RE.match(child.name)
        if m and child.is_dir():
            ckpt_steps.append((int(m.group(1)), child))

    if not ckpt_steps:
        return None

    # Try trainer_state.json in the highest checkpoint
    highest_step, highest_dir = max(ckpt_steps, key=lambda x: x[0])
    trainer_state = highest_dir / "trainer_state.json"
    if trainer_state.exists():
        try:
            with open(trainer_state) as f:
                ts = json.load(f)
                global_step = ts.get("global_step")
                if global_step is not None:
                    return int(global_step)
        except (json.JSONDecodeError, OSError):
            pass

    # Fallback: highest checkpoint step (the final models/ is at least this)
    return highest_step


def _infer_phase(run_dir_name: str) -> str:
    """Infer training phase from run directory name."""
    for pattern, phase in PHASE_PATTERNS:
        if pattern.search(run_dir_name):
            return phase
    return "unknown"


def discover_checkpoints(
    gc_base_path: Path = GC_BASE_PATH,
    rat_init_path: Path = RAT_INIT_PATH,
    finetuned_root: Path = FINETUNED_ROOT,
) -> OrderedDict:
    """
    Auto-discover all available checkpoints from the model directory tree.

    Returns an OrderedDict sorted by (phase_order, step) where phase_order is:
      pretrained=0, init=1, phase1=2, phase2=3, unknown=4

    Each entry has the same schema the rest of the script expects:
      path, label, vocab_size, n_species, phase, step, eval_species
    """
    PHASE_ORDER = {"pretrained": 0, "init": 1, "phase1": 2, "phase2": 3, "unknown": 4}
    discovered = {}

    # ---- Anchor 1: GC Base ----
    if gc_base_path.exists():
        vocab = _read_vocab_size(gc_base_path) or GC_BASE_VOCAB
        discovered["gc_base"] = {
            "path": gc_base_path,
            "label": "GC Base (original)",
            "vocab_size": vocab,
            "n_species": 2,
            "phase": "pretrained",
            "step": 0,
            "eval_species": ["human", "mouse"],
            "_sort": (PHASE_ORDER["pretrained"], 0),
        }
        logger.info(f"  Anchor: GC Base ({gc_base_path})")
    else:
        logger.warning(f"  GC Base not found: {gc_base_path}")

    # ---- Anchor 2: Rat Init ----
    if rat_init_path.exists():
        vocab = _read_vocab_size(rat_init_path) or RAT_EXTENDED_VOCAB
        discovered["rat_init"] = {
            "path": rat_init_path,
            "label": "Rat Init (extended, untrained)",
            "vocab_size": vocab,
            "n_species": 3,
            "phase": "init",
            "step": 0,
            "eval_species": ["human", "mouse", "rat"],
            "_sort": (PHASE_ORDER["init"], 0),
        }
        logger.info(f"  Anchor: Rat Init ({rat_init_path})")
    else:
        logger.warning(f"  Rat Init not found: {rat_init_path}")

    # ---- Scan finetuned_root for run directories ----
    if not finetuned_root.exists():
        logger.warning(f"  Finetuned root not found: {finetuned_root}")
        return OrderedDict(sorted(discovered.items(), key=lambda kv: kv[1]["_sort"]))

    for run_dir in sorted(finetuned_root.iterdir()):
        if not run_dir.is_dir():
            continue

        phase = _infer_phase(run_dir.name)
        phase_ord = PHASE_ORDER.get(phase, 4)
        run_label = run_dir.name

        logger.info(f"  Run dir: {run_dir.name}  (phase={phase})")

        # Discover checkpoint-NNNNN subdirectories
        for child in sorted(run_dir.iterdir()):
            if not child.is_dir():
                continue

            m = CHECKPOINT_RE.match(child.name)
            if m:
                step = int(m.group(1))
                vocab = _read_vocab_size(child) or RAT_EXTENDED_VOCAB
                key = f"{run_label}_ckpt{step}"
                discovered[key] = {
                    "path": child,
                    "label": f"{run_label} -- checkpoint {step:,}",
                    "vocab_size": vocab,
                    "n_species": 3,
                    "phase": phase,
                    "step": step,
                    "eval_species": ["human", "mouse", "rat"],
                    "_sort": (phase_ord, step),
                }
                logger.info(f"    {child.name}  step={step:,}  vocab={vocab}")

            elif child.name == "models" and (child / "pytorch_model.bin").exists():
                # Final snapshot
                final_step = _read_final_step(run_dir)
                vocab = _read_vocab_size(child) or RAT_EXTENDED_VOCAB
                key = f"{run_label}_final"
                discovered[key] = {
                    "path": child,
                    "label": f"{run_label} -- final" + (f" ({final_step:,} steps)" if final_step else ""),
                    "vocab_size": vocab,
                    "n_species": 3,
                    "phase": phase,
                    "step": final_step or 999_999_999,  # sort last if unknown
                    "eval_species": ["human", "mouse", "rat"],
                    "_sort": (phase_ord, final_step or 999_999_999),
                }
                logger.info(f"    models/ (final)  step={final_step or '?'}  vocab={vocab}")

    # Sort by phase order then step
    ordered = OrderedDict(sorted(discovered.items(), key=lambda kv: kv[1]["_sort"]))

    # Strip internal sort key
    for v in ordered.values():
        v.pop("_sort", None)

    logger.info(f"  Total checkpoints discovered: {len(ordered)}")
    return ordered


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model(ckpt_info, device):
    """Load a GeneCompass checkpoint the same way rat_pretrain.py and embed_cells.py do."""
    from genecompass.modeling_bert import BertForMaskedLM
    from transformers import BertConfig
    from torch import nn

    ckpt_path = Path(ckpt_info["path"])
    vocab_size = ckpt_info.get("vocab_size", GC_BASE_VOCAB)
    num_species = ckpt_info.get("n_species", 2)

    logger.info(f"Loading {ckpt_info['label']} from {ckpt_path}")

    # Find pytorch_model.bin
    if (ckpt_path / "pytorch_model.bin").exists():
        model_bin = ckpt_path / "pytorch_model.bin"
    elif (ckpt_path / "models" / "pytorch_model.bin").exists():
        model_bin = ckpt_path / "models" / "pytorch_model.bin"
    else:
        raise FileNotFoundError(f"No pytorch_model.bin in {ckpt_path}")

    state = torch.load(model_bin, map_location="cpu", weights_only=False)

    # Build knowledges from state dict buffers
    knowledges = {
        "promoter": state["bert.embeddings.promoter_knowledge"],
        "co_exp": state["bert.embeddings.co_exp_knowledge"],
        "gene_family": state["bert.embeddings.gene_family_knowledge"],
        "peca_grn": state["bert.embeddings.peca_grn_knowledge"],
    }

    homolog_path = PROJECT_ROOT / "vendor/GeneCompass/prior_knowledge/homologous_hm_token.pickle"
    with open(homolog_path, "rb") as f:
        raw = pickle.load(f)
        knowledges["homologous_gene_human2mouse"] = {v: k for k, v in raw.items()}

    # Build config -- derive vocab_size from state dict if not in config.json
    actual_vocab = state.get("bert.embeddings.word_embeddings.weight", torch.empty(0)).shape[0]
    if actual_vocab > 0:
        vocab_size = actual_vocab

    config_dict = {
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
        "pad_token_id": 0,
        "vocab_size": vocab_size,
        "use_values": True,
        "use_promoter": True,
        "use_co_exp": True,
        "use_gene_family": True,
        "use_peca_grn": True,
        "warmup_steps": 0,
        "emb_warmup_steps": 1,
        "use_cls_token": True,
    }

    model_config = BertConfig(**config_dict)
    model = BertForMaskedLM(model_config, knowledges=knowledges)

    # Patch cls_embedding if needed
    if num_species != 2:
        model.bert.cls_embedding = nn.Embedding(num_species, 768)

    model.load_state_dict(state, strict=False)
    model = model.eval().to(device)

    return model


# ---------------------------------------------------------------------------
# Data sampling
# ---------------------------------------------------------------------------
def sample_cells(corpus_key: str, n_cells: int, seed: int) -> dict:
    """
    Sample n_cells from a corpus and return as batched tensors.

    Returns dict with keys: input_ids, values, species, lengths
    All tensors have shape [n_cells, 2048] (or [n_cells, 1] for species).
    """
    corpus = CORPORA[corpus_key]
    logger.info(f"Sampling {n_cells:,} cells from {corpus['label']} ({corpus['path']})")

    ds = load_from_disk(str(corpus["path"]))
    n_total = len(ds)
    logger.info(f"  Corpus size: {n_total:,} cells")

    rng = np.random.RandomState(seed)
    indices = rng.choice(n_total, size=min(n_cells, n_total), replace=False)
    sample = ds.select(indices.tolist())

    # Extract tensors
    input_ids = torch.tensor(sample["input_ids"], dtype=torch.long)
    values = torch.tensor(sample["values"], dtype=torch.float32)

    # Species: corpus stores as [[species_code]], extract scalar
    species_val = corpus["species"]
    species = torch.full((len(sample), 1), species_val, dtype=torch.long)

    # Lengths: stored as [[n]], extract scalar
    lengths_raw = sample["length"]
    lengths = torch.tensor([l[0] if isinstance(l, list) else l for l in lengths_raw], dtype=torch.long)

    logger.info(f"  Sampled: {len(sample):,} cells, input_ids shape {input_ids.shape}")

    return {
        "input_ids": input_ids,
        "values": values,
        "species": species,
        "lengths": lengths,
    }


# ---------------------------------------------------------------------------
# Masking
# ---------------------------------------------------------------------------
def apply_masking(data: dict, mask_prob: float, seed: int):
    """
    Apply random masking to input_ids and values.

    Critical protocol (from eval_baseline.py):
    1. Replace masked input_ids with MASK_TOKEN_ID (=1)
    2. Zero out values at masked positions
    3. Set labels = original token IDs at masked positions, -100 elsewhere
    4. Set labels_values = original expression values (all positions)

    Only mask non-pad positions (input_ids != PAD_TOKEN_ID).
    """
    rng = np.random.RandomState(seed)

    input_ids = data["input_ids"].clone()
    values = data["values"].clone()

    # Original values for labels
    labels = torch.full_like(input_ids, -100)
    labels_values = values.clone()

    # Create mask: only at non-pad positions
    non_pad = input_ids != PAD_TOKEN_ID
    mask_probs = torch.zeros_like(input_ids, dtype=torch.float32)
    mask_probs[non_pad] = mask_prob

    # Sample mask
    rand = torch.tensor(rng.random(input_ids.shape), dtype=torch.float32)
    mask = rand < mask_probs

    # Set labels at masked positions to original token IDs
    labels[mask] = input_ids[mask]

    # Apply masking: replace with mask token and zero values
    input_ids_masked = input_ids.clone()
    input_ids_masked[mask] = MASK_TOKEN_ID
    values_masked = values.clone()
    values_masked[mask] = 0.0

    n_masked = mask.sum().item()
    n_non_pad = non_pad.sum().item()
    effective_rate = n_masked / n_non_pad if n_non_pad > 0 else 0
    logger.info(f"  Masking: {n_masked:,} tokens masked ({effective_rate:.1%} of non-pad)")

    return {
        "input_ids": input_ids_masked,
        "values": values_masked,
        "species": data["species"],
        "labels": labels,
        "labels_values": labels_values,
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate_model(
    model,
    masked_data: dict,
    device: torch.device,
    batch_size: int = 64,
) -> dict:
    """
    Run masked evaluation and return id_loss, value_loss, combined_loss.

    Uses the GeneCompass BertForMaskedLM.forward() API:
        model(input_ids, values, species, labels, labels_values)
    Returns MaskedLMOutputBoth with: .loss, .id_loss, .value_loss

    Loss formula: combined = 0.2 * id_loss + 0.8 * value_loss
    """
    model.eval()

    input_ids = masked_data["input_ids"]
    values = masked_data["values"]
    species = masked_data["species"]
    labels = masked_data["labels"]
    labels_values = masked_data["labels_values"]

    n_cells = input_ids.shape[0]
    n_batches = (n_cells + batch_size - 1) // batch_size

    total_id_loss = 0.0
    total_value_loss = 0.0
    total_combined_loss = 0.0
    n_valid_batches = 0

    for i in range(n_batches):
        start = i * batch_size
        end = min(start + batch_size, n_cells)

        batch_input_ids = input_ids[start:end].to(device)
        batch_values = values[start:end].to(device)
        batch_species = species[start:end].to(device)
        batch_labels = labels[start:end].to(device)
        batch_labels_values = labels_values[start:end].to(device)

        try:
            outputs = model(
                input_ids=batch_input_ids,
                values=batch_values,
                species=batch_species,
                labels=batch_labels,
                labels_values=batch_labels_values,
            )

            if hasattr(outputs, "id_loss") and outputs.id_loss is not None:
                total_id_loss += outputs.id_loss.item()
                total_value_loss += outputs.value_loss.item()
                total_combined_loss += outputs.loss.item()
                n_valid_batches += 1
        except Exception as e:
            logger.error(f"  Batch {i}/{n_batches} failed: {e}")
            continue

    if n_valid_batches == 0:
        return {"id_loss": float("nan"), "value_loss": float("nan"), "combined_loss": float("nan")}

    return {
        "id_loss": total_id_loss / n_valid_batches,
        "value_loss": total_value_loss / n_valid_batches,
        "combined_loss": total_combined_loss / n_valid_batches,
    }


# ---------------------------------------------------------------------------
# Encoder drift analysis
# ---------------------------------------------------------------------------
def compute_encoder_drift(model, base_state_dict: dict) -> dict:
    """
    Compute per-layer L2 drift between a model and the GC Base state dict.

    Returns a dict with:
      - per-layer L2 distances for all 12 encoder layers
      - total encoder drift (sum of all layer drifts)
      - embedding layer drifts (word_embeddings, cls_embedding, etc.)
      - prediction head drifts
    """
    drift = {}
    model_sd = {k: v.cpu() for k, v in model.state_dict().items()}

    # Encoder layers
    total_encoder_drift = 0.0
    for layer_idx in range(12):
        prefix = f"bert.encoder.layer.{layer_idx}."
        layer_drift = 0.0
        for key in base_state_dict:
            if key.startswith(prefix) and key in model_sd:
                base_val = base_state_dict[key].float()
                model_val = model_sd[key].float()
                if base_val.shape == model_val.shape:
                    d = torch.norm(model_val - base_val, p=2).item()
                    layer_drift += d ** 2
        layer_l2 = layer_drift ** 0.5
        drift[f"encoder_layer_{layer_idx:02d}_l2"] = round(layer_l2, 6)
        total_encoder_drift += layer_drift

    drift["encoder_total_l2"] = round(total_encoder_drift ** 0.5, 6)

    # Key non-encoder components
    components = {
        "word_embeddings": "bert.embeddings.word_embeddings.weight",
        "cls_predictions_transform_dense_weight": "cls.predictions.transform.dense.weight",
        "cls_predictions_transform_dense_bias": "cls.predictions.transform.dense.bias",
        "cls_predictions_bias": "cls.predictions.bias",
    }

    for comp_name, key in components.items():
        if key in base_state_dict and key in model_sd:
            base_val = base_state_dict[key].float()
            model_val = model_sd[key].float()
            min_rows = min(base_val.shape[0], model_val.shape[0])
            b = base_val[:min_rows]
            m = model_val[:min_rows]
            if b.shape == m.shape:
                d = torch.norm(m - b, p=2).item()
                drift[f"{comp_name}_l2"] = round(d, 6)
                base_norm = torch.norm(b, p=2).item()
                drift[f"{comp_name}_relative"] = round(d / base_norm if base_norm > 0 else 0, 6)

    # Species embedding
    base_cls_key = "bert.embeddings.cls_embedding.weight"
    if base_cls_key in base_state_dict and base_cls_key in model_sd:
        base_cls = base_state_dict[base_cls_key].float()
        model_cls = model_sd[base_cls_key].float()
        for sp_idx, sp_name in enumerate(["human", "mouse"]):
            if sp_idx < base_cls.shape[0] and sp_idx < model_cls.shape[0]:
                d = torch.norm(model_cls[sp_idx] - base_cls[sp_idx], p=2).item()
                drift[f"species_emb_{sp_name}_l2"] = round(d, 6)
        if model_cls.shape[0] > 2:
            drift["species_emb_rat_norm"] = round(torch.norm(model_cls[2], p=2).item(), 4)

    # Word embedding norms by range
    we_key = "bert.embeddings.word_embeddings.weight"
    if we_key in model_sd:
        we = model_sd[we_key].float()
        norms = torch.norm(we, p=2, dim=1)
        orig_range = min(GC_BASE_VOCAB, we.shape[0])
        drift["word_emb_original_mean_norm"] = round(norms[:orig_range].mean().item(), 4)
        if we.shape[0] > GC_BASE_VOCAB:
            drift["word_emb_t4_mean_norm"] = round(norms[GC_BASE_VOCAB:].mean().item(), 4)

    return drift


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------
def generate_report(results: dict, drift_results: dict, checkpoints: OrderedDict, args) -> str:
    """Generate a markdown narrative report from evaluation results."""
    lines = []
    lines.append("# Catastrophic Forgetting Progression Analysis")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Cells per evaluation:** {args.n_cells:,}")
    lines.append(f"**Mask probability:** {args.mask_prob}")
    lines.append(f"**Random seed:** {args.seed}")
    lines.append(f"**Checkpoints evaluated:** {len(results)}")
    lines.append(f"**Random guess id_loss:** ln({GC_BASE_VOCAB}) = {np.log(GC_BASE_VOCAB):.2f}")
    lines.append("")

    # Summary table
    lines.append("## 1. Loss Progression Across Checkpoints")
    lines.append("")
    lines.append("| Checkpoint | Phase | Step | Human id_loss | Mouse id_loss | Rat id_loss | Human value_loss | Mouse value_loss | Rat value_loss |")
    lines.append("|---|---|---|---|---|---|---|---|---|")

    for ckpt_key, ckpt_info in checkpoints.items():
        if ckpt_key not in results:
            continue
        r = results[ckpt_key]
        row = [ckpt_info["label"], ckpt_info["phase"], f"{ckpt_info['step']:,}"]
        for species in ["human", "mouse", "rat"]:
            if species in r:
                row.append(f"{r[species]['id_loss']:.4f}")
            else:
                row.append("N/A")
        for species in ["human", "mouse", "rat"]:
            if species in r:
                row.append(f"{r[species]['value_loss']:.6f}")
            else:
                row.append("N/A")
        lines.append("| " + " | ".join(row) + " |")

    lines.append("")

    # Forgetting quantification
    lines.append("## 2. Forgetting Quantification")
    lines.append("")
    gc_human = results.get("gc_base", {}).get("human", {}).get("id_loss")
    gc_mouse = results.get("gc_base", {}).get("mouse", {}).get("id_loss")

    if gc_human is not None:
        lines.append("**Human id_loss degradation from GC Base:**")
        lines.append("")
        lines.append("| Checkpoint | id_loss | Delta | % Degradation |")
        lines.append("|---|---|---|---|")
        for ckpt_key, ckpt_info in checkpoints.items():
            if ckpt_key == "gc_base":
                continue
            r = results.get(ckpt_key, {}).get("human", {})
            if r and "id_loss" in r and not np.isnan(r["id_loss"]):
                delta = r["id_loss"] - gc_human
                pct = (delta / gc_human) * 100
                sign = "+" if delta > 0 else ""
                lines.append(f"| {ckpt_info['label']} | {r['id_loss']:.4f} | {sign}{delta:.4f} | {sign}{pct:.1f}% |")
        lines.append("")

    # Encoder drift
    if drift_results:
        lines.append("## 3. Per-Layer Encoder Drift from GC Base")
        lines.append("")
        lines.append("| Checkpoint | Total Encoder L2 | Layer 0 | Layer 5 | Layer 11 | Word Emb Rel | Species Emb Human L2 | T4 Mean Norm |")
        lines.append("|---|---|---|---|---|---|---|---|")

        for ckpt_key, ckpt_info in checkpoints.items():
            if ckpt_key == "gc_base" or ckpt_key not in drift_results:
                continue
            d = drift_results[ckpt_key]
            row = [
                ckpt_info["label"],
                f"{d.get('encoder_total_l2', 'N/A')}",
                f"{d.get('encoder_layer_00_l2', 'N/A')}",
                f"{d.get('encoder_layer_05_l2', 'N/A')}",
                f"{d.get('encoder_layer_11_l2', 'N/A')}",
                f"{d.get('word_embeddings_relative', 'N/A')}",
                f"{d.get('species_emb_human_l2', 'N/A')}",
                f"{d.get('word_emb_t4_mean_norm', 'N/A')}",
            ]
            lines.append("| " + " | ".join(row) + " |")

        lines.append("")

    # Auto-generated key findings
    lines.append("## 4. Key Findings")
    lines.append("")
    lines.append("*(Auto-generated observations -- verify manually)*")
    lines.append("")

    # Phase 1 vs Phase 2 forgetting comparison
    phase1_ckpts = [k for k, v in checkpoints.items() if v["phase"] == "phase1" and k in results]
    phase2_ckpts = [k for k, v in checkpoints.items() if v["phase"] == "phase2" and k in results]

    if phase1_ckpts and gc_human is not None:
        last_p1 = phase1_ckpts[-1]
        p1_human = results[last_p1].get("human", {}).get("id_loss")
        if p1_human is not None and not np.isnan(p1_human):
            delta_pct = ((p1_human - gc_human) / gc_human) * 100
            lines.append(f"- **Phase 1 end forgetting (human):** id_loss = {p1_human:.4f} ({delta_pct:+.1f}% vs GC Base)")
            if abs(delta_pct) < 5:
                lines.append("  - Minimal forgetting, consistent with frozen encoder strategy")
            lines.append("")

    if phase2_ckpts and gc_human is not None:
        p2_human_trajectory = []
        for ck in phase2_ckpts:
            h = results[ck].get("human", {}).get("id_loss")
            if h is not None and not np.isnan(h):
                p2_human_trajectory.append((checkpoints[ck]["step"], h))
        if len(p2_human_trajectory) >= 2:
            first_step, first_loss = p2_human_trajectory[0]
            last_step, last_loss = p2_human_trajectory[-1]
            direction = "accelerates" if last_loss > first_loss else "stabilizes"
            lines.append(f"- **Phase 2 forgetting {direction}:** human id_loss {first_loss:.4f} (step {first_step:,}) -> {last_loss:.4f} (step {last_step:,})")
            lines.append("")

    # Rat performance at last checkpoint
    all_rat_results = []
    for ck, ck_info in checkpoints.items():
        rat_r = results.get(ck, {}).get("rat", {})
        if rat_r and "id_loss" in rat_r and not np.isnan(rat_r["id_loss"]):
            all_rat_results.append((ck_info["step"], rat_r["id_loss"], ck_info["label"]))
    if all_rat_results:
        best = min(all_rat_results, key=lambda x: x[1])
        lines.append(f"- **Best rat id_loss:** {best[1]:.4f} at {best[2]}")
        lines.append("")

    lines.append("## 5. Recommended Next Steps")
    lines.append("")
    lines.append("1. Review Phase 1 vs GC Base comparison (Section 2) to isolate embedding vs encoder contributions to forgetting")
    lines.append("2. Compare encoder drift progression (Section 3) between Phase 1 and Phase 2 checkpoints")
    lines.append("3. Use these results to calibrate remediation strategy (mixed-species ratios, early stopping)")
    lines.append("4. If forgetting stabilizes after a certain Phase 2 step, consider early stopping for future runs")
    lines.append("")

    return "\n".join(lines)


def generate_tsv(results: dict, checkpoints: OrderedDict) -> str:
    """Generate a tab-separated summary table."""
    header = ["checkpoint", "phase", "step", "corpus", "species", "id_loss", "value_loss", "combined_loss"]
    rows = ["\t".join(header)]

    for ckpt_key, ckpt_info in checkpoints.items():
        if ckpt_key not in results:
            continue
        for species_key in ["human", "mouse", "rat"]:
            r = results[ckpt_key].get(species_key)
            if r is None:
                continue
            row = [
                ckpt_key,
                ckpt_info["phase"],
                str(ckpt_info["step"]),
                CORPORA[species_key]["label"],
                str(CORPORA[species_key]["species"]),
                f"{r['id_loss']:.6f}",
                f"{r['value_loss']:.6f}",
                f"{r['combined_loss']:.6f}",
            ]
            rows.append("\t".join(row))

    return "\n".join(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Forgetting progression analysis across checkpoints")
    parser.add_argument("--n-cells", type=int, default=5000, help="Cells to sample per corpus (default: 5000)")
    parser.add_argument("--mask-prob", type=float, default=0.15, help="Masking probability (default: 0.15)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--batch-size", type=int, default=64, help="Eval batch size (default: 64)")
    parser.add_argument("--output-dir", type=str, default="reports/forgetting", help="Output directory")
    parser.add_argument("--skip-missing", action="store_true", help="Skip checkpoints that don't exist on disk")
    parser.add_argument("--phases", type=str, nargs="*", default=None,
                        help="Filter to specific phases (e.g. --phases phase1 phase2). Default: all.")
    parser.add_argument("--max-checkpoints-per-phase", type=int, default=None,
                        help="Limit checkpoints per phase (keeps first, last, evenly spaced middle)")
    args = parser.parse_args()

    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    if torch.cuda.is_available():
        logger.info(f"  GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ---- Discover checkpoints ----
    logger.info("=" * 70)
    logger.info("STEP 0: Discovering checkpoints")
    logger.info("=" * 70)

    checkpoints = discover_checkpoints()

    # Apply phase filter if specified
    if args.phases:
        allowed = set(args.phases) | {"pretrained", "init"}  # always keep anchors
        checkpoints = OrderedDict(
            (k, v) for k, v in checkpoints.items() if v["phase"] in allowed
        )
        logger.info(f"  After phase filter ({args.phases}): {len(checkpoints)} checkpoints")

    # Apply per-phase cap if specified
    if args.max_checkpoints_per_phase:
        cap = args.max_checkpoints_per_phase
        by_phase: dict[str, list] = {}
        for k, v in checkpoints.items():
            by_phase.setdefault(v["phase"], []).append(k)

        keep_keys = set()
        for phase, keys in by_phase.items():
            if len(keys) <= cap:
                keep_keys.update(keys)
            else:
                # Always keep first and last; evenly space the rest
                keep_keys.add(keys[0])
                keep_keys.add(keys[-1])
                remaining = cap - 2
                if remaining > 0:
                    step = (len(keys) - 2) / (remaining + 1)
                    for i in range(1, remaining + 1):
                        idx = int(round(i * step))
                        keep_keys.add(keys[idx])

        checkpoints = OrderedDict(
            (k, v) for k, v in checkpoints.items() if k in keep_keys
        )
        logger.info(f"  After per-phase cap ({cap}): {len(checkpoints)} checkpoints")

    if not checkpoints:
        logger.error("No checkpoints found. Exiting.")
        sys.exit(1)

    logger.info(f"  Evaluation order:")
    for i, (k, v) in enumerate(checkpoints.items()):
        logger.info(f"    {i+1}. [{v['phase']}] {v['label']}  (vocab={v['vocab_size']})")

    # ---- Pre-sample all corpora ----
    logger.info("=" * 70)
    logger.info("STEP 1: Sampling evaluation data")
    logger.info("=" * 70)

    corpus_data = {}
    for corpus_key in CORPORA:
        if not CORPORA[corpus_key]["path"].exists():
            logger.warning(f"Corpus not found: {corpus_key} ({CORPORA[corpus_key]['path']})")
            continue
        raw = sample_cells(corpus_key, args.n_cells, args.seed)
        masked = apply_masking(raw, args.mask_prob, args.seed)
        corpus_data[corpus_key] = masked

    # ---- Load GC Base state dict for drift comparison ----
    logger.info("=" * 70)
    logger.info("STEP 2: Loading GC Base state dict for drift reference")
    logger.info("=" * 70)

    gc_base_sd = None
    gc_base_info = checkpoints.get("gc_base")
    if gc_base_info:
        gc_path = Path(gc_base_info["path"])
        bin_path = gc_path / "pytorch_model.bin"
        if bin_path.exists():
            gc_base_sd = torch.load(str(bin_path), map_location="cpu", weights_only=False)
            logger.info(f"  GC Base state dict: {len(gc_base_sd)} tensors")
        else:
            # Try safetensors
            try:
                from safetensors.torch import load_file
                st_path = gc_path / "model.safetensors"
                if st_path.exists():
                    gc_base_sd = load_file(str(st_path))
                    logger.info(f"  GC Base state dict (safetensors): {len(gc_base_sd)} tensors")
            except ImportError:
                pass

    if gc_base_sd is None:
        logger.warning("  GC Base state dict not available -- drift analysis will be skipped")

    # ---- Evaluate each checkpoint ----
    logger.info("=" * 70)
    logger.info("STEP 3: Evaluating checkpoints")
    logger.info("=" * 70)

    all_results = {}
    drift_results = {}

    for ckpt_key, ckpt_info in checkpoints.items():
        logger.info("-" * 60)
        logger.info(f"Checkpoint: {ckpt_info['label']}")
        logger.info("-" * 60)

        ckpt_path = Path(ckpt_info["path"])
        if not ckpt_path.exists():
            if args.skip_missing:
                logger.warning(f"  SKIPPED (not found): {ckpt_path}")
                continue
            else:
                logger.error(f"  NOT FOUND: {ckpt_path}")
                continue

        # Load model
        t0 = time.time()
        try:
            model = load_model(ckpt_info, device)
        except Exception as e:
            logger.error(f"  Failed to load: {e}")
            continue

        load_time = time.time() - t0
        logger.info(f"  Model loaded in {load_time:.1f}s")

        # Evaluate on each applicable corpus
        ckpt_results = {}
        for corpus_key in ckpt_info["eval_species"]:
            if corpus_key not in corpus_data:
                logger.warning(f"  Corpus '{corpus_key}' not available, skipping")
                continue

            logger.info(f"  Evaluating on {CORPORA[corpus_key]['label']}...")
            t1 = time.time()
            eval_result = evaluate_model(model, corpus_data[corpus_key], device, args.batch_size)
            eval_time = time.time() - t1

            ckpt_results[corpus_key] = eval_result
            logger.info(
                f"    id_loss={eval_result['id_loss']:.4f}  "
                f"value_loss={eval_result['value_loss']:.6f}  "
                f"combined={eval_result['combined_loss']:.4f}  "
                f"({eval_time:.1f}s)"
            )

        all_results[ckpt_key] = ckpt_results

        # Compute encoder drift from GC Base
        if gc_base_sd is not None and ckpt_key != "gc_base":
            logger.info("  Computing encoder drift from GC Base...")
            drift = compute_encoder_drift(model, gc_base_sd)
            drift_results[ckpt_key] = drift
            logger.info(f"    Total encoder L2 drift: {drift.get('encoder_total_l2', 'N/A')}")

        # Free GPU memory
        del model
        torch.cuda.empty_cache()

    # ---- Save results ----
    logger.info("=" * 70)
    logger.info("STEP 4: Saving results")
    logger.info("=" * 70)

    # JSON results
    results_path = output_dir / "forgetting_results.json"
    with open(results_path, "w") as f:
        json.dump({
            "metadata": {
                "generated": datetime.now().isoformat(),
                "n_cells": args.n_cells,
                "mask_prob": args.mask_prob,
                "seed": args.seed,
                "device": str(device),
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
                "checkpoints_discovered": len(checkpoints),
                "checkpoints_evaluated": len(all_results),
            },
            "checkpoints": {k: {kk: str(vv) if isinstance(vv, Path) else vv
                                 for kk, vv in v.items()}
                            for k, v in checkpoints.items()},
            "results": all_results,
        }, f, indent=2)
    logger.info(f"  Results: {results_path}")

    # Drift JSON
    drift_path = output_dir / "encoder_drift.json"
    with open(drift_path, "w") as f:
        json.dump(drift_results, f, indent=2)
    logger.info(f"  Drift: {drift_path}")

    # TSV summary
    tsv_path = output_dir / "forgetting_summary.tsv"
    with open(tsv_path, "w") as f:
        f.write(generate_tsv(all_results, checkpoints))
    logger.info(f"  TSV: {tsv_path}")

    # Markdown report
    report_path = output_dir / "forgetting_report.md"
    with open(report_path, "w") as f:
        f.write(generate_report(all_results, drift_results, checkpoints, args))
    logger.info(f"  Report: {report_path}")

    logger.info("=" * 70)
    logger.info("DONE")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()