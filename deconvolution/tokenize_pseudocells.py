#!/usr/bin/env python3
"""
tokenize_pseudocells.py -- Stage 8 connector step 2: tokenize deconvolution
pseudo-cells (build_pseudocells.py) into a GeneCompass HuggingFace dataset that
finetune/genecompass/embed_cells.py consumes.

Reuses the EXACT corpus tokenization math (pipeline/05_tokenization/tokenize_corpus.py):
  divide by hybrid median -> log2(1+x) -> rank descending -> top-2048 -> rat token ids.

The ONE deviation from corpus tokenization, required because BayesPrism get.exp values
are bulk-scale and vary ~10^4x across (sample x cell type): each pseudo-cell is
library-normalized to --target-sum BEFORE the log2/median transform. Rationale:
  - the corpus tokenizer feeds RAW single-cell counts (already at the right scale), and
    Stage 4 built the gene medians on normalize_total(10,000) values -- so 10,000 is the
    PRINCIPLED target (the scale the median denominators live on), not an arbitrary pick.
  - the token RANKING (input_ids) is normalization-invariant, so the primary gene-sequence
    signal is unaffected; normalization only brings the `values` channel into distribution
    (corpus value(nonzero) median ~= 0.869 -- printed here for calibration).

Output: <out>/dataset  (datasets.load_from_disk-able; cols input_ids, values, length,
        species, cell_id, sample, cell_type, tissue) + <out>/tokenize_summary.json

Usage (project venv):
  python deconvolution/tokenize_pseudocells.py \
      --h5ad deconvolution/genecompass_input/liver/pseudocells.h5ad \
      --out  deconvolution/genecompass_input/liver --target-sum 10000
"""
import argparse
import json
import os
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import scipy.sparse as sp

_ROOT = Path(os.environ.setdefault("PIPELINE_ROOT", str(Path(__file__).resolve().parents[1])))
sys.path.insert(0, str(_ROOT / "lib"))
sys.path.insert(0, str(_ROOT / "pipeline" / "05_tokenization"))
from gene_utils import load_config, resolve_path                       # noqa: E402
from tokenize_corpus import (load_token_dict, load_median_dict,         # noqa: E402
                             build_eligible_gene_arrays, build_gene_index,
                             map_varnames_to_eligible, tokenize_cell_batch,
                             TOP_N_DEFAULT, CELL_BATCH_SIZE)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--h5ad", required=True, help="pseudocells.h5ad from build_pseudocells.py")
    ap.add_argument("--out", required=True, help="output dir (writes dataset/ + tokenize_summary.json)")
    ap.add_argument("--target-sum", type=float, default=6500.0,
                    help="per-pseudo-cell library normalization target; 6500 calibrates the value "
                    "distribution to the corpus (value median ~0.87). The script prints the achieved "
                    "median -- re-check/adjust per tissue if it drifts from the corpus reference.")
    ap.add_argument("--top-n", type=int, default=TOP_N_DEFAULT)
    ap.add_argument("--species", type=int, default=2, help="GeneCompass species id (rat=2)")
    ap.add_argument("--min-genes", type=int, default=200,
                    help="drop pseudo-cells expressing < this many eligible genes (GeneCompass QC parity)")
    args = ap.parse_args()

    cfg = load_config()
    paths = cfg["paths"]
    token_dict = load_token_dict(Path(resolve_path(cfg, paths["ortholog_dir"])))
    median_dict = load_median_dict(Path(resolve_path(cfg, paths["median_dir"])))
    eligible_ids, eligible_tokens, eligible_medians = build_eligible_gene_arrays(token_dict, median_dict)
    gene_index = build_gene_index(eligible_ids)

    adata = ad.read_h5ad(args.h5ad)
    col_idx, gene_idx = map_varnames_to_eligible(list(adata.var_names), gene_index)
    if not col_idx:
        sys.exit("ERROR: 0 pseudo-cell genes map to the GeneCompass eligible (token∩median) set")
    gene_idx_arr = np.asarray(gene_idx, dtype=np.int32)
    file_tokens = eligible_tokens[gene_idx_arr]
    file_medians = eligible_medians[gene_idx_arr]
    print(f"genes: {adata.n_vars} pseudo-cell genes -> {len(col_idx)} eligible (token∩median) mapped")

    # --- per-pseudo-cell library normalization over the FULL gene set, to the median scale ---
    X = adata.X.toarray().astype(np.float32) if sp.issparse(adata.X) else np.asarray(adata.X, np.float32)
    full_libsize = X.sum(axis=1, keepdims=True)
    full_libsize[full_libsize == 0] = 1.0
    Xn = (X / full_libsize) * args.target_sum            # normalize_total(target_sum) over all genes
    Xn_elig = Xn[:, col_idx]                              # then restrict to the tokenizable genes

    # --- tokenize in batches (reusing the corpus per-cell transform exactly) ---
    all_ids, all_vals, all_lens = [], [], []
    for s in range(0, Xn_elig.shape[0], CELL_BATCH_SIZE):
        ids, vals, lens = tokenize_cell_batch(Xn_elig[s:s + CELL_BATCH_SIZE],
                                              file_tokens, file_medians, args.top_n)
        all_ids.extend(ids); all_vals.extend(vals); all_lens.extend(lens)

    # --- QC parity + value-scale calibration check ---
    keep = np.asarray(all_lens) >= args.min_genes
    n_drop = int((~keep).sum())
    flat = np.array([v for row, k in zip(all_vals, keep) if k for v in row if v > 0], dtype=np.float32)
    vstats = {"median": float(np.median(flat)), "mean": float(flat.mean()),
              "p90": float(np.percentile(flat, 90))}
    print(f"value(nonzero) median={vstats['median']:.3f} mean={vstats['mean']:.3f} "
          f"p90={vstats['p90']:.3f}  (corpus reference: median 0.869, mean 0.960)")
    print(f"expressed length: median={int(np.median(all_lens))} "
          f"min={min(all_lens)} max={max(all_lens)}; dropped {n_drop} cells < {args.min_genes} genes")

    obs = adata.obs.reset_index(drop=True)
    rows = {
        "input_ids": [r for r, k in zip(all_ids, keep) if k],
        "values":    [r for r, k in zip(all_vals, keep) if k],
        "length":    [[l] for l, k in zip(all_lens, keep) if k],
        "species":   [[args.species]] * int(keep.sum()),
        "cell_id":   list(np.asarray(adata.obs_names)[keep]),
    }
    for c in ("sample", "cell_type", "tissue"):
        if c in obs.columns:
            rows[c] = obs[c].astype(str).to_numpy()[keep].tolist()

    from datasets import Dataset
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    Dataset.from_dict(rows).save_to_disk(str(out / "dataset"))
    json.dump({"n_pseudocells": int(keep.sum()), "n_dropped_lt_min_genes": n_drop,
               "n_eligible_genes": len(col_idx), "target_sum": args.target_sum,
               "top_n": args.top_n, "species": args.species,
               "value_stats": vstats, "mean_expressed_length": float(np.mean(all_lens))},
              open(out / "tokenize_summary.json", "w"), indent=2)
    print(f"\nwrote {int(keep.sum())} tokenized pseudo-cells -> {out}/dataset")


if __name__ == "__main__":
    main()
