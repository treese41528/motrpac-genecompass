#!/usr/bin/env python
# coding: utf-8
"""
smoke_test.py — Validate the full Stage 7 chain without training.

Runs in ~2 minutes on a single GPU (or CPU). Tests:
  1. Import chain (vendor GeneCompass modules load correctly)
  2. Config loading (rat_finetune.yaml parses, phase merging works)
  3. Prerequisites (all input files exist)
  4. Token dictionary (correct size, structure, T3+T4 counts)
  5. Checkpoint extension (vocab-indexed tensors resized correctly)
  6. Model construction + state dict loading
  7. Weight tying (decoder.weight is word_embeddings.weight)
  8. Freeze logic (phase 1 freezes encoder, correct param counts)
  9. Forward pass (single batch, loss computes without NaN)
  10. Species handling (rat species=2 produces valid cls_embedding)

Does NOT test:
  - Multi-GPU DDP (use the SLURM smoke test for that)
  - Full epoch training
  - W&B connectivity (tested separately)

Vocab size is derived at runtime from the token dictionary. No hardcoded
sizes -- works with any vocabulary (original 50,558 + N new tokens).

Usage:
  python finetune/genecompass/smoke_test.py
  python finetune/genecompass/smoke_test.py --skip-init  # reuse existing init checkpoint

Author: Tim Reese Lab
Date: March 2026 (updated April 2026 for T3 reclassification)
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
            ORIGINAL_VOCAB_SIZE, EMBEDDING_DIM,
        )
        from rat_model_init import extend_checkpoint, init_t3_warm_start
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

    if not token_dict_path.exists():
        print("   Merged dict not found — building it now...")
        try:
            from build_rat_token_dictionary import build_merged_dictionary
            build_merged_dictionary(hm_path, rat_path, token_dict_path)
            check("Built merged token dict", True, str(token_dict_path))
        except Exception as e:
            check("Build merged token dict", False, str(e))
            sys.exit(1)

    with open(token_dict_path, 'rb') as f:
        token_dict = pickle.load(f)

    vocab_size = max(token_dict.values()) + 1
    check("Token dict consistent", len(token_dict) == vocab_size,
          f"{len(token_dict)} entries, vocab_size={vocab_size}")
    check("Has <pad>", '<pad>' in token_dict)
    check("Has <mask>", '<mask>' in token_dict)

    # Count new tokens (T3 + T4, all above ORIGINAL_VOCAB_SIZE)
    n_new = sum(1 for tid in token_dict.values() if tid >= ORIGINAL_VOCAB_SIZE)
    n_ensrnog = sum(1 for k in token_dict if isinstance(k, str) and k.startswith('ENSRNOG'))
    check("New tokens (T3+T4)", n_new == n_ensrnog,
          f"{n_new} new IDs, {n_ensrnog} ENSRNOG keys")
    check("Pre-trained tokens preserved", ORIGINAL_VOCAB_SIZE + n_new == vocab_size,
          f"{ORIGINAL_VOCAB_SIZE} + {n_new} = {vocab_size}")

    # ── 5. Checkpoint extension ──────────────────────────────────────────
    print("\n5. Checkpoint extension")
    init_dir = resolve(ft_config.get('init_checkpoint', 'data/models/rat_genecompass_init'))

    if args.skip_init and (init_dir / 'pytorch_model.bin').exists():
        check("Init checkpoint (reused)", True, str(init_dir))
        init_state = torch.load(
            init_dir / 'pytorch_model.bin', map_location='cpu', weights_only=False
        )
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
                init_state = torch.load(
                    os.path.join(tmpdir, 'pytorch_model.bin'),
                    map_location='cpu', weights_only=False,
                )

                we_shape = init_state['bert.embeddings.word_embeddings.weight'].shape
                cls_shape = init_state['bert.cls_embedding.weight'].shape
                hi_shape = init_state['bert.embeddings.homologous_index'].shape

                check("word_embeddings shape",
                      we_shape == (vocab_size, EMBEDDING_DIM), str(we_shape))
                check("cls_embedding shape",
                      cls_shape == (3, EMBEDDING_DIM), str(cls_shape))
                check("homologous_index shape",
                      hi_shape == (vocab_size,), str(hi_shape))

                # Verify new rows not all zero
                new_norm = init_state[
                    'bert.embeddings.word_embeddings.weight'
                ][ORIGINAL_VOCAB_SIZE:].norm()
                check("New token word_emb non-zero", new_norm > 0,
                      f"norm={new_norm:.2f}")

                # Verify pre-trained rows preserved
                orig_state = torch.load(
                    model_bin, map_location='cpu', weights_only=False
                )
                orig_we = orig_state['bert.embeddings.word_embeddings.weight']
                preserved = torch.allclose(
                    init_state[
                        'bert.embeddings.word_embeddings.weight'
                    ][:ORIGINAL_VOCAB_SIZE],
                    orig_we, atol=1e-6,
                )
                check("Pre-trained rows preserved", preserved)

                # Verify T3 warm-start: new T3 rows should have norms
                # similar to pre-trained rows (cloned from orthologs)
                pretrained_mean_norm = orig_we[2:100].norm(dim=1).mean().item()
                t3_norms = init_state[
                    'bert.embeddings.word_embeddings.weight'
                ][ORIGINAL_VOCAB_SIZE:ORIGINAL_VOCAB_SIZE + 100].norm(dim=1)
                t3_mean_norm = t3_norms.mean().item()
                check("T3 warm-start norms reasonable",
                      t3_mean_norm > pretrained_mean_norm * 0.5,
                      f"T3 mean={t3_mean_norm:.3f}, pretrained mean={pretrained_mean_norm:.3f}")

        except Exception as e:
            check("Checkpoint extension", False, str(e))
            import traceback
            traceback.print_exc()
            sys.exit(1)

    # ── 6. Model construction + loading ──────────────────────────────────
    print("\n6. Model construction")

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
        pad_token_id=token_dict.get("<pad>"), vocab_size=vocab_size,
        use_values=True, use_promoter=True, use_co_exp=True,
        use_gene_family=True, use_peca_grn=True,
        warmup_steps=5000, emb_warmup_steps=10000, use_cls_token=True,
    )

    from torch import nn
    model = BertForMaskedLM(config, knowledges=knowledges)
    model.bert.cls_embedding = nn.Embedding(3, EMBEDDING_DIM)

    missing, unexpected = model.load_state_dict(init_state, strict=False)
    check("Model loads", True,
          f"missing={len(missing)}, unexpected={len(unexpected)}")

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

    encoder_frozen = all(
        not p.requires_grad for p in model.bert.encoder.parameters()
    )
    check("Encoder layers frozen", encoder_frozen)
    check("word_embeddings trainable",
          model.bert.embeddings.word_embeddings.weight.requires_grad)
    check("cls_embedding trainable",
          model.bert.cls_embedding.weight.requires_grad)

    n_trainable_unfrozen = unfreeze_all(model, world_rank=0)
    check("Unfrozen: all trainable", n_trainable_unfrozen == n_total)

    # ── 9. Forward pass ──────────────────────────────────────────────────
    print("\n9. Forward pass (single batch, CPU)")
    model.eval()

    batch_size, seq_len = 2, 64
    input_ids = torch.randint(2, vocab_size, (batch_size, seq_len))
    input_ids[:, 0] = token_dict.get('<pad>', 0)
    values = torch.rand(batch_size, seq_len)
    species = torch.full((batch_size, 1), 2, dtype=torch.long)  # rat
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)

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
        check("value_loss is finite", np.isfinite(value_loss),
              f"value_loss={value_loss:.4f}")
        check("Loss formula",
              abs(loss - (0.2 * id_loss + 0.8 * value_loss)) < 0.01,
              f"{loss:.4f} ≈ 0.2*{id_loss:.4f} + 0.8*{value_loss:.4f}")
        check("Logits shape",
              output.logits.shape == (batch_size, seq_len, vocab_size),
              str(output.logits.shape))
    except Exception as e:
        check("Forward pass", False, str(e))
        import traceback
        traceback.print_exc()

    # ── 10. Species handling ─────────────────────────────────────────────
    print("\n10. Species handling")
    rat_cls = model.bert.cls_embedding(torch.tensor([[2]]))
    human_cls = model.bert.cls_embedding(torch.tensor([[0]]))
    mouse_cls = model.bert.cls_embedding(torch.tensor([[1]]))

    check("Rat cls_embedding non-zero", rat_cls.norm().item() > 0)
    check("Rat != human", not torch.allclose(rat_cls, human_cls))

    cos_sim = torch.cosine_similarity(
        rat_cls.view(1, -1), mouse_cls.view(1, -1)
    ).item()
    check("Rat ~ mouse (init)", cos_sim > 0.5, f"cosine={cos_sim:.3f}")

    # ── 11. Multi-species forward pass ───────────────────────────────────
    print("\n11. Multi-species forward pass")
    try:
        # Human cell
        human_species = torch.full((batch_size, 1), 0, dtype=torch.long)
        with torch.no_grad():
            h_out = model(
                input_ids=input_ids, values=values, species=human_species,
                attention_mask=attention_mask, labels=labels,
                labels_values=labels_values,
            )
        check("Human forward pass", np.isfinite(h_out.loss.item()),
              f"loss={h_out.loss.item():.4f}")

        # Mouse cell
        mouse_species = torch.full((batch_size, 1), 1, dtype=torch.long)
        with torch.no_grad():
            m_out = model(
                input_ids=input_ids, values=values, species=mouse_species,
                attention_mask=attention_mask, labels=labels,
                labels_values=labels_values,
            )
        check("Mouse forward pass", np.isfinite(m_out.loss.item()),
              f"loss={m_out.loss.item():.4f}")

        # Losses should differ by species (different cls_embedding)
        check("Species produce different losses",
              abs(h_out.loss.item() - m_out.loss.item()) > 1e-6)
    except Exception as e:
        check("Multi-species forward", False, str(e))
        import traceback
        traceback.print_exc()

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