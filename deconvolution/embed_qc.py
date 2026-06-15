#!/usr/bin/env python3
"""embed_qc.py -- per-tissue QC of the deconv->GeneCompass pseudo-cell embeddings.

For each tissue under <root> (default data/deconvolution/genecompass_input), reports:
  - cell-type separation: silhouette (cosine) + kNN purity (k=10)
  - sample-level signal retained: between-cell-type vs within-type variance split
  - value(nonzero) median (calibration vs corpus ~0.869)
A healthy result: high silhouette/purity AND a non-trivial within-type % (Aim-2 signal).

Usage: python deconvolution/embed_qc.py [genecompass_input_dir]
"""
import glob
import os
import sys

import numpy as np
from datasets import load_from_disk
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

from pathlib import Path                                  # config-driven root (no hardcoded paths in code)
sys.path.insert(0, str(Path(os.environ.setdefault("PIPELINE_ROOT", str(Path(__file__).resolve().parents[1]))) / "lib"))
from gene_utils import load_config, resolve_path
_cfg = load_config()
root = sys.argv[1] if len(sys.argv) > 1 else resolve_path(_cfg, _cfg["deconvolution"]["genecompass_input_dir"])
hdr = f"{'tissue':12s} {'cells':>6s} {'types':>5s} {'silhou':>7s} {'kNNpur':>6s} {'btwVar%':>7s} {'withinVar%':>10s} {'valMed':>7s}"
print(hdr); print("-" * len(hdr))
for d in sorted(glob.glob(f"{root}/*/")):
    tis = os.path.basename(d.rstrip("/"))
    emb_p, ds_p = os.path.join(d, "embeddings", "cell_embeddings.npy"), os.path.join(d, "dataset")
    if not (os.path.exists(emb_p) and os.path.isdir(ds_p)):
        print(f"{tis:12s}  (no embeddings yet)"); continue
    emb = np.load(emb_p); ds = load_from_disk(ds_p); ct = np.array(ds["cell_type"])
    if emb.shape[0] != len(ct):
        print(f"{tis:12s}  MISALIGNED {emb.shape[0]} vs {len(ct)}"); continue
    nt = len(np.unique(ct)); embn = normalize(emb)
    sil = silhouette_score(embn, ct, metric="cosine") if nt > 1 else float("nan")
    _, idx = NearestNeighbors(n_neighbors=min(11, len(ct)), metric="cosine").fit(embn).kneighbors(embn)
    pur = float(np.mean([(ct[idx[i, 1:]] == ct[i]).mean() for i in range(len(ct))]))
    mu = emb.mean(0)
    btw = sum(((emb[ct == t].mean(0) - mu) ** 2).sum() * (ct == t).sum() for t in np.unique(ct))
    wth = sum(((emb[ct == t] - emb[ct == t].mean(0)) ** 2).sum() for t in np.unique(ct))
    vals = np.asarray(ds["values"], dtype=np.float32); vm = float(np.median(vals[vals > 0]))
    tot = btw + wth
    print(f"{tis:12s} {len(ct):6d} {nt:5d} {sil:7.3f} {pur:6.2f} "
          f"{100*btw/tot:7.1f} {100*wth/tot:10.1f} {vm:7.3f}")
