#!/usr/bin/env python
# coding: utf-8
r"""
perturb_cells.py -- rat in-silico perturbation engine (Module A; Aim-2b GRN + Aim-3b keystone).

Given a fine-tuned rat GeneCompass model and tokenized pseudo-cells, delete (or overexpress) a gene's
token in each cell, re-embed, and quantify (a) the shift in the cell's CLS embedding [the perturbation's
overall effect] and (b) the per-gene contextual-embedding shifts [the predicted downstream targets ->
the model-driven GRN edge for Module B]. Reused once by Aim 2b (GRN) and Aim 3b (perturb transferred
human cells) -- do not fork per aim.

DESIGN (why this is a clean rewrite, not the vendored class):
  vendor/GeneCompass/genecompass/perturb_delete_chipseq.py is Geneformer-style but rough -- it hardcodes
  species=0 and 'cuda:1', has a token-dict bug (self.ensembl_id = last-loop-var, l.424), a trailing-space
  key bug ("input_ids ", l.96), and absolute /home/ict paths. We keep its *concepts* (delete a token ->
  re-embed -> cosine shift; align original-minus-deleted vs perturbed for per-gene target scores) but
  build on embed_cells.py's VALIDATED loader (the same one the embeddings + transfer use), with the
  rat-readiness fixes baked in: (a) load knowledges from the state-dict + 3-species cls_embedding; (b)
  species=2; (c) --device; (d) rat_tokens.pickle (ENSRNOG->token idx, all 22,213 rat genes); (e) values
  kept on the pseudo-cell's own log-normalized scale (no renorm on delete; overexpress sets the value
  channel). GeneCompass has NO cross-cell attention, so perturbation is strictly per-cell.

Usage:
  python finetune/genecompass/perturb_cells.py \
      --model-dir data/models/rat_genecompass_finetuned/models/rat_phase2_mixed_species/models \
      --dataset   data/deconvolution/genecompass_input/skmvl/dataset \
      --genes ENSRNOG00000010829 \            # or --gene-file <tsv of ENSRNOG, 1/line>
      --cell-type "Skeletal muscle" \          # optional; default = all cell types
      --mode delete --targets --out out.tsv --device cuda
"""
import argparse
import json
import logging
import os
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(os.environ.get("PIPELINE_ROOT", ".")).resolve()
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "vendor" / "GeneCompass"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("perturb_cells")

# ENSRNOG -> token index (all 22,213 rat genes; ortholog token where one exists, else T4 token).
DEFAULT_TOKEN_DICT = _ROOT / "data" / "training" / "ortholog_mappings" / "rat_tokens.pickle"


# ── model + token dict ────────────────────────────────────────────────────────────────────────────
def load_model(model_dir, device="cuda"):
    """Mirror of embed_cells.py:89-130 (the validated loader). NOT the vendor human/mouse loader."""
    from transformers import BertConfig
    from genecompass import BertForMaskedLM
    from torch import nn

    cfg = json.load(open(Path(model_dir) / "config.json"))
    state = torch.load(Path(model_dir) / "pytorch_model.bin", map_location="cpu", weights_only=False)
    knowledges = {k: state[f"bert.embeddings.{k}_knowledge"]
                  for k in ("promoter", "co_exp", "gene_family", "peca_grn")}
    raw = pickle.load(open(_ROOT / "vendor" / "GeneCompass" / "prior_knowledge" / "homologous_hm_token.pickle", "rb"))
    knowledges["homologous_gene_human2mouse"] = {v: k for k, v in raw.items()}
    cfg["warmup_steps"] = 0
    cfg["emb_warmup_steps"] = 1
    model = BertForMaskedLM(BertConfig(**cfg), knowledges=knowledges)
    model.bert.cls_embedding = nn.Embedding(3, cfg["hidden_size"])   # (c) 3 species: 0=human 1=mouse 2=rat
    model.load_state_dict(state, strict=False)
    return model.eval().to(device)


def load_token_dict(path=DEFAULT_TOKEN_DICT):
    d = pickle.load(open(path, "rb"))
    assert all(isinstance(k, str) for k in list(d)[:5]), "token dict must be ENSRNOG(str) -> index(int)"
    return d


# ── forward pass ──────────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def embed(model, input_ids, values, device, species=2):
    """input_ids,values: [B,L] tensors -> (cls [B,H], gene_emb [B,L,H]). CLS is prepended by the model."""
    input_ids = input_ids.to(device).long()
    values = values.to(device).float()
    attn = (input_ids != 0).long()
    sp = torch.full((input_ids.shape[0], 1), species, dtype=torch.long, device=device)
    out = model.bert(input_ids=input_ids, values=values, attention_mask=attn, species=sp,
                     emb_warmup_alpha=1.0, output_hidden_states=False)[0]
    return out[:, 0, :], out[:, 1:, :]


# ── perturbations (per-cell, on the padded 2048-length profile) ─────────────────────────────────────
def delete_token(ids_row, vals_row, token):
    """Remove every position == token, left-shift the survivors, pad with 0 to keep length L.
    Returns (new_ids, new_vals, removed_positions)."""
    keep = ids_row != token
    L = ids_row.shape[0]
    new_ids = torch.zeros(L, dtype=ids_row.dtype)
    new_vals = torch.zeros(L, dtype=vals_row.dtype)
    ki, kv = ids_row[keep], vals_row[keep]
    new_ids[: ki.shape[0]] = ki
    new_vals[: kv.shape[0]] = kv
    removed = (~keep).nonzero(as_tuple=True)[0].tolist()
    return new_ids, new_vals, removed


def overexpress_token(ids_row, vals_row, token, level):
    """Set the value channel of `token` to `level` (a high value -> top rank). Structure unchanged."""
    v = vals_row.clone()
    v[ids_row == token] = level
    return ids_row.clone(), v


def _cos_shift(a, b):
    """1 - cosine, row-wise. 0 = no change, larger = bigger perturbation effect."""
    return (1.0 - torch.nn.functional.cosine_similarity(a, b, dim=-1)).cpu().numpy()


# ── the engine ──────────────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def perturb_gene(model, ids, vals, token, device, mode="delete", overexpress_level=None,
                 batch_size=32, want_targets=False, invert=None):
    """Perturb `token` in every cell of (ids[N,L], vals[N,L]) that expresses it. Returns a dict with
    n_cells, mean/median cell-shift, and (if want_targets) per-target mean gene-embedding shift."""
    has = (ids == token).any(dim=1)
    idx = has.nonzero(as_tuple=True)[0]
    if idx.numel() == 0:
        return {"n_cells": 0}
    ids, vals = ids[idx], vals[idx]

    cell_shifts, target_scores = [], defaultdict(list)
    for s in range(0, ids.shape[0], batch_size):
        bi, bv = ids[s:s + batch_size], vals[s:s + batch_size]
        # build the perturbed batch row-by-row (delete shifts positions per cell)
        pi = torch.zeros_like(bi)
        pv = torch.zeros_like(bv)
        removed_pos = []
        for r in range(bi.shape[0]):
            if mode == "delete":
                ni, nv, rem = delete_token(bi[r], bv[r], token)
                removed_pos.append(rem[0] if rem else None)
            else:
                ni, nv = overexpress_token(bi[r], bv[r], token, overexpress_level)
                removed_pos.append(None)
            pi[r], pv[r] = ni, nv
        cls0, g0 = embed(model, bi, bv, device)
        cls1, g1 = embed(model, pi, pv, device)
        cell_shifts.append(_cos_shift(cls0, cls1))

        if want_targets:
            for r in range(bi.shape[0]):
                if mode == "delete" and removed_pos[r] is not None:
                    p = removed_pos[r]
                    # align original-minus-deleted (drop position p) vs perturbed (drop trailing pad)
                    keep = torch.ones(g0.shape[1], dtype=torch.bool); keep[p] = False
                    orig = g0[r][keep]                       # [L-1, H], gene order = original minus deleted
                    pert = g1[r][: orig.shape[0]]            # [L-1, H]
                    surviving_tokens = bi[r][keep].tolist()
                else:
                    orig, pert = g0[r], g1[r]
                    surviving_tokens = bi[r].tolist()
                shift = _cos_shift(orig, pert)
                for tok, sh in zip(surviving_tokens, shift):
                    if tok != 0:
                        target_scores[int(tok)].append(float(sh))

    out = {
        "n_cells": int(ids.shape[0]),
        "mean_cell_shift": float(np.mean(np.concatenate(cell_shifts))),
        "median_cell_shift": float(np.median(np.concatenate(cell_shifts))),
    }
    if want_targets:
        agg = {t: float(np.mean(v)) for t, v in target_scores.items() if len(v) >= max(3, 0.5 * out["n_cells"])}
        top = sorted(agg.items(), key=lambda x: -x[1])[:25]
        out["top_targets"] = [(invert.get(t, str(t)) if invert else str(t), round(sh, 5)) for t, sh in top]
    return out


def load_cells(dataset_path, cell_type=None, max_cells=None):
    """Return (ids[N,L], vals[N,L]) tensors for the (optionally cell-type-filtered) pseudo-cells."""
    from datasets import load_from_disk
    ds = load_from_disk(dataset_path)
    if cell_type is not None:
        ds = ds.filter(lambda e: e["cell_type"] == cell_type)
    if max_cells is not None and max_cells < len(ds):
        ds = ds.select(range(max_cells))
    ids = torch.tensor(np.array(ds["input_ids"]), dtype=torch.long)
    vals = torch.tensor(np.array(ds["values"]), dtype=torch.float)
    return ids, vals, len(ds)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--dataset", required=True, help="a tissue's genecompass_input/<tissue>/dataset")
    ap.add_argument("--genes", nargs="*", default=[], help="ENSRNOG ids to perturb")
    ap.add_argument("--gene-file", help="TSV/txt with one ENSRNOG per line (col 1)")
    ap.add_argument("--cell-type", default=None, help="restrict to this cell_type (default: all)")
    ap.add_argument("--mode", choices=["delete", "overexpress"], default="delete")
    ap.add_argument("--overexpress-level", type=float, default=8.0, help="value for overexpress (values ~O(1), max ~7.6)")
    ap.add_argument("--targets", action="store_true", help="also score predicted downstream targets (GRN edges)")
    ap.add_argument("--token-dict", default=str(DEFAULT_TOKEN_DICT))
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-cells", type=int, default=None)
    args = ap.parse_args()

    genes = list(args.genes)
    if args.gene_file:
        genes += [l.split("\t")[0].strip() for l in open(args.gene_file) if l.strip() and not l.startswith("#")]
    genes = [g for g in dict.fromkeys(genes) if g]
    if not genes:
        ap.error("give --genes or --gene-file")

    tok = load_token_dict(Path(args.token_dict))
    invert = {v: k for k, v in tok.items()}
    model = load_model(args.model_dir, args.device)
    ids, vals, n = load_cells(args.dataset, args.cell_type, args.max_cells)
    log.info(f"loaded {n} cells (cell_type={args.cell_type}); perturbing {len(genes)} gene(s), mode={args.mode}")

    rows = []
    for g in genes:
        if g not in tok:
            log.warning(f"{g}: not in token dict -> skip"); continue
        r = perturb_gene(model, ids, vals, tok[g], args.device, mode=args.mode,
                         overexpress_level=args.overexpress_level, batch_size=args.batch_size,
                         want_targets=args.targets, invert=invert)
        if r["n_cells"] == 0:
            log.warning(f"{g}: expressed in 0 cells -> ~zero shift by construction");
        r["gene"] = g
        r["cell_type"] = args.cell_type or "ALL"
        rows.append(r)
        log.info(f"{g} [{args.cell_type or 'ALL'}]: n={r['n_cells']} "
                 f"mean_shift={r.get('mean_cell_shift', 0):.4f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        f.write("gene\tcell_type\tn_cells\tmean_cell_shift\tmedian_cell_shift\ttop_targets\n")
        for r in rows:
            tgt = ";".join(f"{n}:{s}" for n, s in r.get("top_targets", [])) if "top_targets" in r else ""
            f.write(f"{r['gene']}\t{r['cell_type']}\t{r['n_cells']}\t{r.get('mean_cell_shift','')}\t"
                    f"{r.get('median_cell_shift','')}\t{tgt}\n")
    log.info(f"wrote {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
