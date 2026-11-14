#!/usr/bin/env python
"""check_weight_load.py -- audit finding 6.7: does every tensor of the fine-tuned checkpoint load?

Rebuilds the model exactly as embed_cells.py and perturb_cells.py do (BertForMaskedLM with the four prior-knowledge
tensors taken from the checkpoint, a 3-species CLS embedding), loads the weights with strict=False through
finetune/genecompass/checked_load.py in report mode, and writes which model tensors the checkpoint did not supply,
which checkpoint tensors the model did not use, and whether every loaded tensor equals the checkpoint's.

Usage: check_weight_load.py <out_json> <name>=<model_dir> [<name>=<model_dir> ...]
"""
import json
import os
import pickle
import sys
from pathlib import Path

import torch

ROOT = Path(os.environ.get("PIPELINE_ROOT", ".")).resolve()
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor" / "GeneCompass"))
sys.path.insert(0, str(ROOT / "finetune" / "genecompass"))

from torch import nn                                    # noqa: E402
from transformers import BertConfig                      # noqa: E402
from genecompass import BertForMaskedLM                  # noqa: E402
from checked_load import load_state_checked              # noqa: E402

out = {}
for spec in sys.argv[2:]:
    name, model_dir = spec.split("=", 1)
    cfg = json.load(open(Path(model_dir) / "config.json"))
    state = torch.load(Path(model_dir) / "pytorch_model.bin", map_location="cpu", weights_only=False)
    knowledges = {k: state[f"bert.embeddings.{k}_knowledge"] for k in ("promoter", "co_exp", "gene_family", "peca_grn")}
    raw = pickle.load(open(ROOT / "vendor" / "GeneCompass" / "prior_knowledge" / "homologous_hm_token.pickle", "rb"))
    knowledges["homologous_gene_human2mouse"] = {v: k for k, v in raw.items()}
    cfg["warmup_steps"] = 0
    cfg["emb_warmup_steps"] = 1
    model = BertForMaskedLM(BertConfig(**cfg), knowledges=knowledges)
    model.bert.cls_embedding = nn.Embedding(3, cfg["hidden_size"])
    rep = load_state_checked(model, state, raise_on_error=False, allow_missing=(), allow_unexpected=())
    rep["model_dir"] = str(model_dir)
    rep["n_parameters"] = int(sum(p.numel() for p in model.parameters()))
    # which missing tensors does the encoder (model.bert), the part that produces the CLS embedding, own?
    rep["missing_in_encoder"] = [k for k in rep["missing"] if k.startswith("bert.")]
    out[name] = rep
    print(f"[{name}] parameters={rep['n_parameters']:,} loaded={rep['n_loaded']}/{rep['n_model_tensors']} "
          f"missing={len(rep['missing'])} (in encoder: {len(rep['missing_in_encoder'])}) "
          f"unexpected={len(rep['unexpected'])} loaded-but-different={rep['n_loaded_but_different']}")
    for k in rep["missing"]:
        print(f"    missing     {k}")
    for k in rep["unexpected"]:
        print(f"    unexpected  {k}  {tuple(state[k].shape)}")
Path(sys.argv[1]).parent.mkdir(parents=True, exist_ok=True)
json.dump(out, open(sys.argv[1], "w"), indent=1)
