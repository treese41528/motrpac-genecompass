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
      --h5ad data/deconvolution/genecompass_input/liver/pseudocells.h5ad \
      --out  data/deconvolution/genecompass_input/liver --target-sum 10000
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


def tokenize_cell_batch_pa(X, tokens, medians, top_n, pa_mask, pa_max_rank):
    """Like tokenize_corpus.tokenize_cell_batch, but among EXPRESSED genes give PA genes (pa_mask)
    priority into the top_n IF sufficiently expressed (value-rank <= pa_max_rank), displacing the
    LOWEST-value non-PA genes. The emitted sequence stays expression-descending and the `values`
    are the true log2 values -- only membership at the margin changes. Returns (..., n_promoted)."""
    LOG2 = float(np.log(2.0))
    ids_out, vals_out, lens, promoted = [], [], [], 0
    for i in range(X.shape[0]):
        x = np.log1p(X[i] / medians) / LOG2
        nz = np.where(x > 0.0)[0]
        if nz.size == 0:
            ids_out.append([0] * top_n); vals_out.append([0.0] * top_n); lens.append(0); continue
        order = nz[np.argsort(-x[nz])]                       # expressed, value-descending
        if order.size <= top_n:
            sel = order
        else:
            base = order[:top_n]
            window = order[:min(pa_max_rank, order.size)]
            pa_in_window = window[pa_mask[window]]            # PA genes that are sufficiently expressed
            base_set = set(base.tolist())
            missing = np.array([g for g in pa_in_window.tolist() if g not in base_set], dtype=int)
            if missing.size:
                base_nonpa = base[~pa_mask[base]]             # value-descending; tail = lowest value
                ndrop = int(min(missing.size, base_nonpa.size))
                drop_set = set(base_nonpa[base_nonpa.size - ndrop:].tolist())
                keep = np.array([g for g in base.tolist() if g not in drop_set], dtype=int)
                sel = np.concatenate([keep, missing[:ndrop]])  # promote the highest-value missing PA
                promoted += ndrop
            else:
                sel = base
            sel = sel[np.argsort(-x[sel])][:top_n]            # keep the sequence expression-descending
        n = int(sel.size)
        toks = tokens[sel].tolist(); vv = x[sel].tolist()
        if n < top_n:
            toks += [0] * (top_n - n); vv += [0.0] * (top_n - n)
        ids_out.append(toks); vals_out.append(vv); lens.append(n)
    return ids_out, vals_out, lens, promoted


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
    ap.add_argument("--pa-genes", help="optional TSV with a 'feature_ID' column of PA/training-regulated "
                    "rel-113 ENSRNOG; these are PREFERRED into the top-N when sufficiently expressed")
    ap.add_argument("--pa-max-rank", type=int, default=4096,
                    help="'sufficiently expressed' bar for PA-preference: a PA gene is promoted only if "
                    "its expression ranks within the top this-many in the pseudo-cell (default 4096 = 2x top-N). "
                    "Bounds how far below the top-N cut a PA gene can be promoted from, limiting OOD drift.")
    ap.add_argument("--pa-tissue", help="if --pa-genes has a 'tissue' column, restrict the PA set to genes "
                    "training-regulated in THIS tissue (MoTrPAC label, e.g. LIVER, SKM-GN) -- the per-tissue "
                    "set is far more selective than the cross-tissue union (~43%% of genes)")
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

    # --- PA-gene mask over the eligible-mapped columns (preferred into top-N if sufficiently expressed) ---
    pa_mask = None
    if args.pa_genes:
        with open(args.pa_genes) as fh:
            hdr = fh.readline().rstrip("\n").split("\t")
            fi = hdr.index("feature_ID"); ti = hdr.index("tissue") if "tissue" in hdr else None
            pa = set()
            for ln in fh:
                if not ln.strip():
                    continue
                p = ln.rstrip("\n").split("\t")
                if args.pa_tissue and ti is not None and args.pa_tissue not in p[ti].split(","):
                    continue
                pa.add(p[fi].strip().split(".")[0].upper())
        mapped_base = [str(adata.var_names[c]).strip().split(".")[0].upper() for c in col_idx]
        pa_mask = np.array([g in pa for g in mapped_base], dtype=bool)
        print(f"PA-preference: {len(pa)} PA genes"
              f"{f' (tissue={args.pa_tissue})' if args.pa_tissue else ' (cross-tissue union)'}; "
              f"{int(pa_mask.sum())}/{len(pa_mask)} eligible pseudo-cell genes are PA; "
              f"sufficiently-expressed bar = expression-rank <= {args.pa_max_rank}")

    # --- tokenize in batches (corpus per-cell transform; PA-preferred selection if --pa-genes) ---
    all_ids, all_vals, all_lens = [], [], []
    n_promoted = 0
    for s in range(0, Xn_elig.shape[0], CELL_BATCH_SIZE):
        Xb = Xn_elig[s:s + CELL_BATCH_SIZE]
        if pa_mask is not None:
            ids, vals, lens, prom = tokenize_cell_batch_pa(Xb, file_tokens, file_medians,
                                                           args.top_n, pa_mask, args.pa_max_rank)
            n_promoted += prom
        else:
            ids, vals, lens = tokenize_cell_batch(Xb, file_tokens, file_medians, args.top_n)
        all_ids.extend(ids); all_vals.extend(vals); all_lens.extend(lens)
    if pa_mask is not None:
        print(f"PA-preference: promoted {n_promoted} PA slots "
              f"({n_promoted/max(len(all_ids),1):.1f}/pseudo-cell avg into the top-{args.top_n})")

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
