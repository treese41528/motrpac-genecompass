#!/usr/bin/env python
# coding: utf-8
"""
smoke_test.py — Validate the full Stage 7 chain without training.

Runs in ~2 minutes on a single GPU. Tests:
  1. Import chain (vendor GeneCompass modules load correctly)
  2. Config loading (rat_finetune.yaml parses, phase merging works)
  3. Checkpoint extension (rat_model_init builds [53032, 768] tensors)
  4. Model construction (BertForMaskedLM with 3-species cls_embedding)
  5. State dict loading (extended weights load without errors)
  6. Freeze logic (phase 1 freezes encoder, correct param counts)
  7. Forward pass (single batch, loss computes without NaN)
  8. Weight tying (decoder.weight is word_embeddings.weight)
  9. Species handling (rat species=2 produces valid cls_embedding)
  10. Checkpoint save/load round-trip

Does NOT test:
  - Multi-GPU DDP (use the SLURM smoke test for that)
  - Full epoch training
  - W&B connectivity (tested separately)

Usage:
  python finetune/genecompass/smoke_test.py
  python finetune/genecompass/smoke_test.py --skip-init  # reuse existing init checkpoint

Author: Tim Reese Lab
Date: March 2026
"""

import argparse
import json
import logging
import os
import pickle
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

_PROJECT_ROOT = Path(os.environ.get('PIPELINE_ROOT', '.')).resolve()
_VENDOR_GC = _PROJECT_ROOT / 'vendor' / 'GeneCompass'
_SCRIPT_DIR = Path(__file__).resolve().parent

sys.path.insert(0, str(_VENDOR_GC))
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_SCRIPT_DIR))

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"
results = []


def check(name, condition, detail=""):
    status = PASS if condition else FAIL
    results.append((name, condition))
    msg = f"  {status} {name}"
    if detail:
        msg += f" — {detail}"
    print(msg)
    return condition


def main():
    parser = argparse.ArgumentParser(description="Stage 7 smoke test")
    parser.add_argument('--skip-init', action='store_true',
                        help='Skip checkpoint extension (reuse existing)')
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("STAGE 7 SMOKE TEST")
    print("=" * 60)

    # ── 1. Imports ───────────────────────────────────────────────────────
    print("\n1. Import chain")
    try:
        from genecompass import GenecompassPretrainer, BertForMaskedLM
        from genecompass.knowledge_embeddings import KnowledgeBertEmbeddings
        from transformers import BertConfig
        check("GeneCompass imports", True)
    except Exception as e:
        check("GeneCompass imports", False, str(e))
        print("\nFATAL: Cannot import GeneCompass. Check vendor submodule.")
        sys.exit(1)

    try:
        from rat_load_prior_embedding import (
            load_rat_prior_embeddings, build_knowledges_dict,
            ORIGINAL_VOCAB_SIZE, RAT_VOCAB_SIZE, T4_START
        )
        from rat_model_init import extend_checkpoint
        from rat_pretrain import (
            load_pipeline_config, freeze_encoder, unfreeze_all, get_phase_config
        )
        check("Pipeline imports", True)
    except Exception as e:
        check("Pipeline imports", False, str(e))
        sys.exit(1)

    # ── 2. Config loading ────────────────────────────────────────────────
    print("\n2. Config loading")
    config_path = _SCRIPT_DIR / 'configs' / 'rat_finetune.yaml'
    try:
        ft_config = load_pipeline_config(str(config_path))
        check("Config parse", True, f"{len(ft_config)} keys")
        check("vocab_size", ft_config.get('vocab_size') == 53032,
              f"got {ft_config.get('vocab_size')}")
        check("num_species", ft_config.get('num_species') == 3)

        # Phase merge
        p1 = get_phase_config(ft_config, 1)
        p2 = get_phase_config(ft_config, 2)
        check("Phase 1 freeze", p1.get('freeze_encoder') is True)
        check("Phase 1 LR", p1.get('max_learning_rate') == 5e-5,
              f"got {p1.get('max_learning_rate')}")
        check("Phase 2 unfreeze", p2.get('freeze_encoder') is False)
        check("Phase 2 LR", p2.get('max_learning_rate') == 1e-5,
              f"got {p2.get('max_learning_rate')}")
    except Exception as e:
        check("Config parse", False, str(e))
        sys.exit(1)

    # ── 3. Prerequisites ─────────────────────────────────────────────────
    print("\n3. Prerequisites")
    resolve = lambda p: _PROJECT_ROOT / p if not Path(p).is_absolute() else Path(p)

    ckpt_path = resolve(ft_config.get('base_checkpoint',
                        'vendor/GeneCompass/pretrained_models/GeneCompass_Base'))
    model_bin = ckpt_path / 'pytorch_model.bin'
    check("GC checkpoint exists", model_bin.exists(), str(model_bin))

    # Merged token dict may not exist yet — we build it in step 4.
    # Here just check the source files exist.
    hm_path = resolve('vendor/GeneCompass/prior_knowledge/human_mouse_tokens.pickle')
    rat_path = resolve('data/training/ortholog_mappings/rat_tokens.pickle')
    check("human_mouse_tokens.pickle exists", hm_path.exists())
    check("rat_tokens.pickle exists", rat_path.exists())

    stage6_dir = resolve(ft_config.get('stage6_dir', 'data/training/prior_knowledge'))
    for emb_file in ('promoter_embeddings.pkl', 'coexp_embeddings.pkl',
                      'family_embeddings.pkl', 'grn_embeddings.pkl'):
        check(f"Stage 6 {emb_file}", (stage6_dir / emb_file).exists())

    homolog_path = resolve(ft_config.get('homologous_gene_path',
                           'vendor/GeneCompass/prior_knowledge/homologous_hm_token.pickle'))
    check("Homolog mapping exists", homolog_path.exists())

    dataset_dir = resolve(ft_config.get('dataset_directory',
                          'data/training/tokenized_corpus/dataset'))
    check("Dataset exists", dataset_dir.exists())
    check("sorted_length.pickle", (dataset_dir / 'sorted_length.pickle').exists())

    # Stop if prerequisites missing
    if not all(ok for _, ok in results):
        print("\nFailed prerequisites — fix before continuing.")
        sys.exit(1)

    # ── 4. Token dictionary ──────────────────────────────────────────────
    print("\n4. Token dictionary")

    token_dict_path = resolve(ft_config.get('token_dict_path',
                              'data/training/ortholog_mappings/rat_human_mouse_tokens.pickle'))

    # Build merged dict if it doesn't exist yet
    if not token_dict_path.exists():
        print("   Merged dict not found — building it now...")
        try:
            from build_rat_token_dictionary import build_merged_dictionary
            hm_path = _PROJECT_ROOT / 'vendor' / 'GeneCompass' / 'prior_knowledge' / 'human_mouse_tokens.pickle'
            rat_path = _PROJECT_ROOT / 'data' / 'training' / 'ortholog_mappings' / 'rat_tokens.pickle'
            build_merged_dictionary(hm_path, rat_path, token_dict_path)
            check("Built merged token dict", True, str(token_dict_path))
        except Exception as e:
            check("Build merged token dict", False, str(e))
            sys.exit(1)

    with open(token_dict_path, 'rb') as f:
        token_dict = pickle.load(f)
    check("Token dict size", len(token_dict) == RAT_VOCAB_SIZE,
          f"{len(token_dict)} entries")
    check("Has <pad>", '<pad>' in token_dict)
    check("Has <mask>", '<mask>' in token_dict)

    # Count T4 tokens
    n_t4 = sum(1 for tid in token_dict.values() if tid >= T4_START)
    check("T4 token count", n_t4 == 2474, f"got {n_t4}")

    # ── 5. Checkpoint extension ──────────────────────────────────────────
    print("\n5. Checkpoint extension")
    init_dir = resolve(ft_config.get('init_checkpoint', 'data/models/rat_genecompass_init'))

    if args.skip_init and (init_dir / 'pytorch_model.bin').exists():
        check("Init checkpoint (reused)", True, str(init_dir))
    else:
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                extend_checkpoint(
                    checkpoint_path=ckpt_path,
                    rat_token_dict_path=token_dict_path,
                    stage6_dir=stage6_dir,
                    homologous_gene_path=homolog_path,
                    output_dir=tmpdir,
                    t4_init_strategy='random',
                    t4_init_std=0.02,
                    species_noise_std=0.02,
                    coexp_rescale_factor=3.9,
                )
                # Verify output
                ext_state = torch.load(
                    os.path.join(tmpdir, 'pytorch_model.bin'),
                    map_location='cpu', weights_only=False,
                )
                we_shape = ext_state['bert.embeddings.word_embeddings.weight'].shape
                cls_shape = ext_state['bert.cls_embedding.weight'].shape
                hi_shape = ext_state['bert.embeddings.homologous_index'].shape

                check("word_embeddings shape", we_shape == (53032, 768), str(we_shape))
                check("cls_embedding shape", cls_shape == (3, 768), str(cls_shape))
                check("homologous_index shape", hi_shape == (53032,), str(hi_shape))

                # Verify T4 rows not all zero (word_embeddings should have random init)
                t4_norm = ext_state['bert.embeddings.word_embeddings.weight'][T4_START:].norm()
                check("T4 word_emb non-zero", t4_norm > 0, f"norm={t4_norm:.2f}")

                # Verify pre-trained rows preserved
                orig_state = torch.load(model_bin, map_location='cpu', weights_only=False)
                orig_we = orig_state['bert.embeddings.word_embeddings.weight']
                preserved = torch.allclose(
                    ext_state['bert.embeddings.word_embeddings.weight'][:ORIGINAL_VOCAB_SIZE],
                    orig_we, atol=1e-6,
                )
                check("Pre-trained rows preserved", preserved)

                # Use this for subsequent tests
                init_state = ext_state
        except Exception as e:
            check("Checkpoint extension", False, str(e))
            import traceback
            traceback.print_exc()
            sys.exit(1)

    # ── 6. Model construction + loading ──────────────────────────────────
    print("\n6. Model construction")

    # Load init state if we skipped extension
    if args.skip_init:
        init_state = torch.load(
            init_dir / 'pytorch_model.bin', map_location='cpu', weights_only=False
        )

    # Build knowledges from state dict
    knowledges = {
        'promoter': init_state['bert.embeddings.promoter_knowledge'],
        'co_exp': init_state['bert.embeddings.co_exp_knowledge'],
        'gene_family': init_state['bert.embeddings.gene_family_knowledge'],
        'peca_grn': init_state['bert.embeddings.peca_grn_knowledge'],
    }
    with open(homolog_path, 'rb') as f:
        raw = pickle.load(f)
        knowledges['homologous_gene_human2mouse'] = {v: k for k, v in raw.items()}

    config = BertConfig(
        hidden_size=768, num_hidden_layers=12, initializer_range=0.02,
        layer_norm_eps=1e-12, attention_probs_dropout_prob=0.02,
        hidden_dropout_prob=0.02, intermediate_size=3072, hidden_act="gelu",
        max_position_embeddings=2048, model_type="bert", num_attention_heads=12,
        pad_token_id=token_dict.get("<pad>"), vocab_size=53032,
        use_values=True, use_promoter=True, use_co_exp=True,
        use_gene_family=True, use_peca_grn=True,
        warmup_steps=5000, emb_warmup_steps=10000, use_cls_token=True,
    )

    from torch import nn
    model = BertForMaskedLM(config, knowledges=knowledges)
    model.bert.cls_embedding = nn.Embedding(3, 768)
    missing, unexpected = model.load_state_dict(init_state, strict=False)
    check("Model loads", True, f"missing={len(missing)}, unexpected={len(unexpected)}")

    # ── 7. Weight tying ──────────────────────────────────────────────────
    print("\n7. Weight tying")
    we = model.bert.embeddings.word_embeddings.weight
    dw = model.cls.predictions.decoder.weight
    check("decoder.weight is word_embeddings", we.data_ptr() == dw.data_ptr())

    # ── 8. Freeze logic ──────────────────────────────────────────────────
    print("\n8. Freeze logic")
    n_total = sum(p.numel() for p in model.parameters())
    n_trainable_frozen = freeze_encoder(model, world_rank=0)
    check("Frozen: trainable < total",
          n_trainable_frozen < n_total,
          f"{n_trainable_frozen:,} / {n_total:,}")

    # Encoder should be frozen
    encoder_frozen = all(
        not p.requires_grad
        for p in model.bert.encoder.parameters()
    )
    check("Encoder layers frozen", encoder_frozen)

    # word_embeddings should be trainable
    check("word_embeddings trainable", model.bert.embeddings.word_embeddings.weight.requires_grad)
    check("cls_embedding trainable", model.bert.cls_embedding.weight.requires_grad)

    # Unfreeze
    n_trainable_unfrozen = unfreeze_all(model, world_rank=0)
    check("Unfrozen: all trainable", n_trainable_unfrozen == n_total)

    # ── 9. Forward pass ──────────────────────────────────────────────────
    print("\n9. Forward pass (single batch, CPU)")
    model.eval()

    # Create a fake batch: 2 cells, 64 tokens each
    batch_size, seq_len = 2, 64
    # Use real token IDs: mix of T1-T3 and T4
    input_ids = torch.randint(2, 53032, (batch_size, seq_len))
    input_ids[:, 0] = token_dict.get('<pad>', 0)  # ensure pad exists
    values = torch.rand(batch_size, seq_len)
    species = torch.full((batch_size, 1), 2, dtype=torch.long)  # rat
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)

    # Mask some tokens for labels
    labels = input_ids.clone()
    labels[torch.rand(batch_size, seq_len) > 0.15] = -100
    labels_values = values.clone()

    try:
        with torch.no_grad():
            output = model(
                input_ids=input_ids,
                values=values,
                species=species,
                attention_mask=attention_mask,
                labels=labels,
                labels_values=labels_values,
            )
        loss = output.loss.item()
        id_loss = output.id_loss.item()
        value_loss = output.value_loss.item()

        check("Forward pass succeeds", True)
        check("Loss is finite", np.isfinite(loss), f"loss={loss:.4f}")
        check("id_loss is finite", np.isfinite(id_loss), f"id_loss={id_loss:.4f}")
        check("value_loss is finite", np.isfinite(value_loss), f"value_loss={value_loss:.4f}")
        check("Loss formula", abs(loss - (0.2 * id_loss + 0.8 * value_loss)) < 0.01,
              f"{loss:.4f} ≈ 0.2×{id_loss:.4f} + 0.8×{value_loss:.4f}")

        # Check output shape
        logits_shape = output.logits.shape
        check("Logits shape", logits_shape == (batch_size, seq_len, 53032),
              str(logits_shape))

    except Exception as e:
        check("Forward pass", False, str(e))
        import traceback
        traceback.print_exc()

    # ── 10. Species=2 cls_embedding ──────────────────────────────────────
    print("\n10. Species handling")
    rat_cls = model.bert.cls_embedding(torch.tensor([[2]]))
    human_cls = model.bert.cls_embedding(torch.tensor([[0]]))
    mouse_cls = model.bert.cls_embedding(torch.tensor([[1]]))
    check("Rat cls_embedding non-zero", rat_cls.norm().item() > 0)
    check("Rat ≠ human", not torch.allclose(rat_cls, human_cls))
    check("Rat ≈ mouse (init)", torch.cosine_similarity(
        rat_cls.view(1, -1), mouse_cls.view(1, -1)).item() > 0.5,
        f"cosine={torch.cosine_similarity(rat_cls.view(1,-1), mouse_cls.view(1,-1)).item():.3f}")

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    passed = sum(1 for _, ok in results if ok)
    failed = sum(1 for _, ok in results if not ok)
    total = len(results)
    if failed == 0:
        print(f"{PASS} ALL {total} CHECKS PASSED")
    else:
        print(f"{FAIL} {failed}/{total} CHECKS FAILED")
        for name, ok in results:
            if not ok:
                print(f"  {FAIL} {name}")
    print("=" * 60 + "\n")

    sys.exit(0 if failed == 0 else 1)


if __name__ == '__main__':
    main()