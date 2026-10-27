"""
transformers_compat.py — Compatibility patches for GeneCompass + transformers ≥4.57.

GeneCompass's GenecompassPretrainer was written against transformers ~4.30.
Transformers 4.57 changed several Trainer method signatures and save behavior.
These patches bridge the gap without modifying the vendor submodule.

Four patches:
  1. _get_train_sampler: added `dataset` positional arg; multi-GPU returns one
     full-dataset sampler (accelerate shards it — no double sharding)
  2. training_step: added `num_items_in_batch` arg + removed do_grad_scaling/use_apex
  3. compute_loss: added `num_items_in_batch` arg
  4. GenecompassPreCollator.save_pretrained: Trainer._save() now calls it

Usage:
    from patches.transformers_compat import apply_patches
    apply_patches()  # call once before creating GenecompassPretrainer

Author: Tim Reese Lab
Date: March 2026
"""

import logging

import torch

logger = logging.getLogger(__name__)


def apply_patches():
    """Apply all transformers 4.57 compatibility patches to GeneCompass classes."""
    from genecompass import GenecompassPretrainer
    from genecompass.pretrainer import GenecompassPreCollator

    # ── 1. _get_train_sampler(self) → _get_train_sampler(self, dataset=None)
    # Multi-GPU: upstream returns a rank-sharded CustomDistributedLengthGroupedSampler,
    # but in 4.57 Trainer passes the DataLoader through accelerator.prepare(), which
    # shards again — each rank then saw 1/world_size of its own shard, and an epoch
    # covered only 1/world_size of the dataset. Return one full-dataset sampler,
    # identically ordered on every rank (seed + epoch), and let accelerate shard it.
    _orig_get_train_sampler = GenecompassPretrainer._get_train_sampler

    def _patched_get_train_sampler(self, dataset=None):
        if self.args.world_size <= 1:
            return _orig_get_train_sampler(self)
        return SeededLengthGroupedSampler(
            batch_size=self.args.train_batch_size,
            lengths=self.example_lengths,
            seed=self.args.seed,
        )

    GenecompassPretrainer._get_train_sampler = _patched_get_train_sampler

    # ── 2. training_step: full replacement
    # Vendor uses self.do_grad_scaling and self.use_apex (removed in 4.57).
    # Reimplemented with self.accelerator.backward(). Same logic as vendor
    # line 636-676: forward, log id_loss/value_loss, backward.
    def _patched_training_step(self, model, inputs, num_items_in_batch=None):
        model.train()
        inputs = self._prepare_inputs(inputs)
        with self.compute_loss_context_manager():
            loss, id_loss, value_loss = self.compute_loss(model, inputs)
        if self.args.n_gpu > 1:
            loss = loss.mean()
            id_loss = id_loss.mean()
            value_loss = value_loss.mean()

        self.log({"id_loss": id_loss.item(), "value_loss": value_loss.item()})
        self.accelerator.backward(loss)

        return loss.detach() / self.args.gradient_accumulation_steps

    GenecompassPretrainer.training_step = _patched_training_step

    # ── 3. compute_loss: absorb num_items_in_batch
    _orig_compute_loss = GenecompassPretrainer.compute_loss

    def _patched_compute_loss(self, model, inputs, return_outputs=False,
                              num_items_in_batch=None):
        return _orig_compute_loss(self, model, inputs,
                                  return_outputs=return_outputs)

    GenecompassPretrainer.compute_loss = _patched_compute_loss

    # ── 4. GenecompassPreCollator.save_pretrained
    # Trainer._save() calls self.data_collator.tokenizer.save_pretrained()
    # which hits the PreCollator (used as a tokenizer stand-in). No-op.
    if not hasattr(GenecompassPreCollator, 'save_pretrained'):
        GenecompassPreCollator.save_pretrained = lambda self, *a, **kw: None

    logger.info("Applied 4 transformers 4.57 compatibility patches")


class SeededLengthGroupedSampler(torch.utils.data.Sampler):
    """Length-grouped order over the WHOLE dataset, the same on every rank.

    Same grouping as transformers' LengthGroupedSampler; the generator is seeded with
    seed + epoch, so every rank yields the same order and accelerate's BatchSamplerShard
    hands each rank distinct batches. Resume (skip_first_batches) replays the same order.
    """

    def __init__(self, batch_size, lengths, seed=0):
        self.batch_size = batch_size
        self.lengths = list(lengths)
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        from transformers.trainer_pt_utils import get_length_grouped_indices
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        return iter(get_length_grouped_indices(self.lengths, self.batch_size, generator=g))


def declare_tied_weights(model):
    """
    Declare intentionally shared tensors for transformers ≥4.57 save.

    transformers 4.57 checks for shared tensors before saving and raises
    RuntimeError if they're not declared in _tied_weights_keys. These are
    the intentionally shared tensors in GeneCompass:
      - decoder.weight == word_embeddings.weight (MLM output tying)
      - decoder.bias == predictions.bias (both cls and cls4value heads)
    """
    model._tied_weights_keys = [
        "cls.predictions.decoder.weight",
        "cls.predictions.decoder.bias",
        "cls4value.predictions.decoder.bias",
    ]
    logger.info("Declared 3 tied weight keys for safe checkpoint saving")