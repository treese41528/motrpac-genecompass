#!/usr/bin/env python3
"""
transfer_to_human.py -- Aim 3a / Module E step 1 (E.1): cross-species TRANSFER of
the rat exercise pseudo-cells INTO the human GeneCompass embedding space.

WHY
  MoTrPAC's invasive, multi-tissue exercise time-course cannot be run in humans
  (ethics + feasibility). The rat is the experimental proxy and GeneCompass is the
  transfer vehicle: we re-express every measured rat pseudo-cell AS a human cell,
  then (E.2) test whether the trained-vs-control / ordinal-dose axis SURVIVES in
  human space, per cell type. This is one-directional TRANSFER (rat data -> human
  representation -> analyze), NOT alignment of two measured datasets. The output is
  a COUNTERFACTUAL human representation of the rat response (an inference), and a
  human atlas is OPTIONAL (validation backdrop), never a binding input.

WHAT  (this script = step 1, CPU)
  rat ENSRNOG pseudo-cell  (BayesPrism get.exp, raw count-mass; build_pseudocells.py)
    --ortholog map (rat_to_human_mapping.pickle; many rat genes -> one human ENSG are SUMMED)-->
  human ENSG expression profile
    --tokenize (HUMAN ENSG tokens [genecompass_tokens] + HUMAN gene medians
       [genecompass_medians], SAME target-sum 6500 + top-2048 as the rat path)-->
  HuggingFace dataset/ with species=0 (human)
  Step 2 = embed_cells.py --species 0 (GPU); both driven by pipeline/run_stage12.py.

PARITY  (make-or-break -- verified 2026-06-24)
  Reuses the EXACT corpus tokenization math
  (pipeline/05_tokenization/tokenize_corpus.tokenize_cell_batch):
      normalize_total(6500) over the full gene set
      -> divide by per-gene median -> log2(1+x) -> rank descending -> top-2048 -> zero-pad.
  ONLY THREE things change vs the rat path: the token dict (human ENSG), the median
  dict (human), and the species token (0, not 2). Empirical grounding:
    * target_sum = 6500.0 in ALL 10 rat tissues (tokenize_summary.json), value median ~0.87.
    * 13,883 / 15,234 ortholog-mapped rat genes already carry the SAME GeneCompass token
      ID as their human ortholog (the rat tokens reuse the GC ID space) -> for those the
      transfer reuses the identical token. Human ENSG token IDs are 2..23114, all < the
      checkpoint vocab_size (55,275), so they index real pretrained embedding rows.
    * Rat-specific (T4) genes have NO human ortholog and are DROPPED (logged). ~69% of rat
      genes map; per tissue ~10.7k-11.6k human-eligible genes remain.
    * Only species==1 (mouse) triggers homolog token-remapping in the model; human (0) and
      rat (2) use identity lookup -- so feeding human ENSG tokens with species=0 is valid.

Output (mirrors the rat genecompass_input/<label>/ layout so the existing detection
scripts run unchanged with --gc-root <human-root>):
  <out-root>/<label>/dataset/                 HF dataset (input_ids, values, length,
                                              species=[0], cell_id, sample, cell_type, tissue)
  <out-root>/<label>/tokenize_summary.json    value-scale QC (compare to corpus median 0.869)
  <out-root>/<label>/transfer_summary.json    ortholog-projection coverage
  <out-root>/<label>/pseudocells.h5ad         symlink to the rat h5ad (augur_prep PCA control)
  <out-root>/<label>/summary.txt              symlink to the rat summary (pseudocell count)

Usage (project venv):
  python translation/transfer_to_human.py --label liver
  python translation/transfer_to_human.py --label skmvl --out-root data/deconvolution/genecompass_input_human
"""
import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import scipy.sparse as sp

_ROOT = Path(os.environ.setdefault("PIPELINE_ROOT", str(Path(__file__).resolve().parents[1])))
sys.path.insert(0, str(_ROOT / "lib"))
sys.path.insert(0, str(_ROOT / "pipeline" / "05_tokenization"))
from gene_utils import load_config, resolve_path                              # noqa: E402
from tokenize_corpus import (build_eligible_gene_arrays, build_gene_index,    # noqa: E402
                             map_varnames_to_eligible, tokenize_cell_batch,
                             TOP_N_DEFAULT, CELL_BATCH_SIZE)

HUMAN_SPECIES_ID = 0   # GeneCompass CLS species row: 0=human, 1=mouse, 2=rat


def _base(x: str) -> str:
    """rel-aware base id: strip version suffix, upper-case (ENSG00000123.4 -> ENSG00000123)."""
    return str(x).strip().split(".")[0].upper()


def _load_pickle(p: Path):
    with open(p, "rb") as f:
        return pickle.load(f)


def load_human_token_dict(path: Path) -> dict:
    """human_mouse_tokens.pickle -> {ENSG_base: token_id}. Keep ONLY human ENSG tokens
    (drop ENSMUSG + the <pad>/<mask> specials); a human profile has no mouse/rat ids."""
    raw = _load_pickle(path)
    d = {}
    for k, v in raw.items():
        b = _base(k)
        if b.startswith("ENSG"):
            d[b] = int(v)
    return d


def load_human_median_dict(path: Path) -> dict:
    """human_gene_median_after_filter.pickle -> {ENSG_base: median>0}."""
    raw = _load_pickle(path)
    d = {}
    for k, v in raw.items():
        m = float(v)
        if m > 0.0:
            d[_base(k)] = m
    return d


def load_ortholog_map(path: Path) -> dict:
    """rat_to_human_mapping.pickle -> {ENSRNOG_base: ENSG_base} (15,234 mapped genes)."""
    return {_base(k): _base(v) for k, v in _load_pickle(path).items()}


def project_rat_to_human(adata: ad.AnnData, r2h: dict):
    """Project rat ENSRNOG expression onto human ENSG genes via the ortholog map.

    Many rat genes can be orthologs of one human gene (up to 29-to-1 in this corpus);
    their expression is SUMMED into the shared human column (total ortholog count-mass).
    Rat genes with no human ortholog (T4 rat-specific) are dropped.

    Returns (X_human [cells x H_uniq], human_genes list[ENSG_base], full_rat_lib [cells x 1], stats).
    X_human spans only the ortholog-MAPPED human genes; full_rat_lib is the per-cell sum over ALL
    rat genes (incl. unmapped/T4) -- the SAME normalize_total denominator the rat path uses
    (tokenize_pseudocells.py: full_libsize = X.sum over all n_vars, THEN restrict). Dividing the
    surviving mapped count-mass by the full rat library keeps each surviving gene's value scale
    IDENTICAL to the rat path; using the mapped-only sum instead would inflate every value ~1.2x
    (~17% of rat count-mass is in dropped T4 genes) and re-introduce the value-scale drift that
    target_sum=6500 was calibrated to remove.
    """
    rat_base = [_base(v) for v in adata.var_names]
    pairs = [(j, r2h[g]) for j, g in enumerate(rat_base) if g in r2h]   # (rat_col, ENSG)
    if not pairs:
        raise SystemExit("ERROR: 0 rat pseudo-cell genes have a human ortholog")
    human_genes = sorted({h for _, h in pairs})
    hidx = {h: i for i, h in enumerate(human_genes)}
    rows = np.fromiter((j for j, _ in pairs), dtype=np.int64, count=len(pairs))      # rat cols
    cols = np.fromiter((hidx[h] for _, h in pairs), dtype=np.int64, count=len(pairs))  # human cols
    P = sp.csr_matrix((np.ones(len(pairs), np.float32), (rows, cols)),
                      shape=(adata.n_vars, len(human_genes)))           # rat x human (sums many->one)
    X = adata.X if sp.issparse(adata.X) else sp.csr_matrix(adata.X)
    full_rat_lib = np.asarray(X.sum(axis=1), dtype=np.float64).reshape(-1, 1)  # rat-path normalize_total denom
    Xh = X @ P                                                          # cells x human (mapped genes only)
    Xh = Xh.toarray().astype(np.float32) if sp.issparse(Xh) else np.asarray(Xh, np.float32)
    stats = {
        "n_rat_genes": int(adata.n_vars),
        "n_rat_mapped": int(len(pairs)),
        "frac_rat_mapped": round(len(pairs) / max(adata.n_vars, 1), 4),
        "frac_count_mass_mapped": round(float(Xh.sum() / max(full_rat_lib.sum(), 1.0)), 4),
        "n_human_unique": int(len(human_genes)),
        "n_collisions_summed": int(len(pairs) - len(human_genes)),
    }
    return Xh, human_genes, full_rat_lib, stats


def transfer_one(adata, r2h, token_dict, median_dict, target_sum, top_n, min_genes):
    """Full E.1 transform for one tissue's pseudo-cells. Returns (rows dict, summary dict)."""
    Xh, human_genes, full_libsize, proj = project_rat_to_human(adata, r2h)

    # eligible human genes = those with BOTH a token AND a positive median (mirrors rat path)
    eligible_ids, eligible_tokens, eligible_medians = build_eligible_gene_arrays(token_dict, median_dict)
    gene_index = build_gene_index(eligible_ids)
    col_idx, gene_idx = map_varnames_to_eligible(human_genes, gene_index)
    if not col_idx:
        raise SystemExit("ERROR: 0 projected human genes map to the GeneCompass eligible (token∩median) set")
    gene_idx_arr = np.asarray(gene_idx, dtype=np.int32)
    file_tokens = eligible_tokens[gene_idx_arr]
    file_medians = eligible_medians[gene_idx_arr]
    print(f"  genes: {adata.n_vars} rat -> {proj['n_rat_mapped']} orthologs "
          f"({proj['frac_rat_mapped']*100:.1f}%) -> {proj['n_human_unique']} human "
          f"({proj['n_collisions_summed']} many-to-one summed) -> {len(col_idx)} eligible (token∩median); "
          f"mapped count-mass {proj['frac_count_mass_mapped']*100:.1f}%")

    # per-cell library normalization to the rat-path denominator (sum over ALL rat genes, incl. dropped
    # T4 mass), then restrict to tokenizable -- keeps the surviving genes' value scale identical to rat
    full_libsize = full_libsize.copy()
    full_libsize[full_libsize == 0] = 1.0
    Xn = (Xh / full_libsize) * target_sum
    Xn_elig = Xn[:, col_idx]

    # tokenize in batches with the EXACT corpus per-cell transform (no PA preference)
    all_ids, all_vals, all_lens = [], [], []
    for s in range(0, Xn_elig.shape[0], CELL_BATCH_SIZE):
        ids, vals, lens = tokenize_cell_batch(Xn_elig[s:s + CELL_BATCH_SIZE],
                                              file_tokens, file_medians, top_n)
        all_ids.extend(ids); all_vals.extend(vals); all_lens.extend(lens)

    # QC parity + value-scale calibration (corpus reference: median 0.869, mean 0.960)
    keep = np.asarray(all_lens) >= min_genes
    n_drop = int((~keep).sum())
    flat = np.array([v for row, k in zip(all_vals, keep) if k for v in row if v > 0], dtype=np.float32)
    vstats = {"median": float(np.median(flat)), "mean": float(flat.mean()),
              "p90": float(np.percentile(flat, 90))}
    print(f"  value(nonzero) median={vstats['median']:.3f} mean={vstats['mean']:.3f} "
          f"p90={vstats['p90']:.3f}  (corpus reference: median 0.869, mean 0.960)")
    print(f"  expressed length: median={int(np.median(all_lens))} "
          f"min={min(all_lens)} max={max(all_lens)}; dropped {n_drop} cells < {min_genes} genes")

    obs = adata.obs.reset_index(drop=True)
    rows = {
        "input_ids": [r for r, k in zip(all_ids, keep) if k],
        "values":    [r for r, k in zip(all_vals, keep) if k],
        "length":    [[l] for l, k in zip(all_lens, keep) if k],
        "species":   [[HUMAN_SPECIES_ID]] * int(keep.sum()),
        "cell_id":   list(np.asarray(adata.obs_names)[keep]),
    }
    for c in ("sample", "cell_type", "tissue"):
        if c in obs.columns:
            rows[c] = obs[c].astype(str).to_numpy()[keep].tolist()

    summary = {"n_pseudocells": int(keep.sum()), "n_dropped_lt_min_genes": n_drop,
               "n_eligible_genes": len(col_idx), "target_sum": target_sum,
               "top_n": top_n, "species": HUMAN_SPECIES_ID, "value_stats": vstats,
               "mean_expressed_length": float(np.mean(all_lens)),
               "ortholog_projection": proj}
    return rows, summary


def _symlink(src: Path, dst: Path):
    """Best-effort relative symlink (rat pseudocells.h5ad/summary.txt into the human dir)."""
    if not src.exists():
        return False
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        dst.symlink_to(os.path.relpath(src, dst.parent))
        return True
    except OSError:
        return False


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", required=True,
                    help="genecompass_input/<label> dir holding the rat pseudocells.h5ad (Stage 8/9 output)")
    ap.add_argument("--in-root", default=None,
                    help="rat genecompass_input root (default: deconvolution.genecompass_input_dir from config)")
    ap.add_argument("--out-root", default=None,
                    help="human output root (default: <in-root> + '_human')")
    ap.add_argument("--ortholog-map", default=None,
                    help="rat_to_human_mapping.pickle (default: paths.ortholog_dir/rat_to_human_mapping.pickle)")
    ap.add_argument("--token-dict", default=None,
                    help="human ENSG token pickle (default: paths.genecompass_tokens)")
    ap.add_argument("--median-dict", default=None,
                    help="human ENSG median pickle (default: paths.genecompass_medians)")
    ap.add_argument("--target-sum", type=float, default=6500.0,
                    help="per-pseudo-cell normalize_total target; MUST match the rat run (6500)")
    ap.add_argument("--top-n", type=int, default=TOP_N_DEFAULT)
    ap.add_argument("--min-genes", type=int, default=200,
                    help="drop pseudo-cells expressing < this many eligible genes (GeneCompass QC parity)")
    args = ap.parse_args()

    cfg = load_config()
    paths, dc = cfg["paths"], cfg["deconvolution"]
    in_root = Path(args.in_root) if args.in_root else resolve_path(cfg, dc["genecompass_input_dir"])
    out_root = Path(args.out_root) if args.out_root else Path(str(in_root) + "_human")
    omap = Path(args.ortholog_map) if args.ortholog_map \
        else Path(resolve_path(cfg, paths["ortholog_dir"])) / "rat_to_human_mapping.pickle"
    tok = Path(args.token_dict) if args.token_dict else Path(resolve_path(cfg, paths["genecompass_tokens"]))
    med = Path(args.median_dict) if args.median_dict else Path(resolve_path(cfg, paths["genecompass_medians"]))

    in_dir = in_root / args.label
    h5ad = in_dir / "pseudocells.h5ad"
    if not h5ad.exists():
        sys.exit(f"ERROR: rat pseudocells.h5ad not found: {h5ad}\n  -> run Stage 8 first for label '{args.label}'")

    print("=" * 70)
    print(f"TRANSFER rat -> human  label={args.label}  target_sum={args.target_sum}  species={HUMAN_SPECIES_ID}")
    print(f"  in    = {h5ad}")
    print(f"  omap  = {omap}")
    print(f"  tokens= {tok}")
    print(f"  median= {med}")
    print("=" * 70)

    r2h = load_ortholog_map(omap)
    token_dict = load_human_token_dict(tok)
    median_dict = load_human_median_dict(med)
    print(f"  ortholog map: {len(r2h):,} rat->human  |  human tokens: {len(token_dict):,} ENSG  |  "
          f"human medians: {len(median_dict):,} ENSG")

    adata = ad.read_h5ad(h5ad)
    rows, summary = transfer_one(adata, r2h, token_dict, median_dict,
                                 args.target_sum, args.top_n, args.min_genes)

    from datasets import Dataset
    out_dir = out_root / args.label
    out_dir.mkdir(parents=True, exist_ok=True)
    Dataset.from_dict(rows).save_to_disk(str(out_dir / "dataset"))
    json.dump(summary, open(out_dir / "tokenize_summary.json", "w"), indent=2)
    json.dump(summary["ortholog_projection"], open(out_dir / "transfer_summary.json", "w"), indent=2)
    # make the rat pseudocells/summary reachable from the human dir (augur_prep PCA control, pseudocell count)
    _symlink(h5ad, out_dir / "pseudocells.h5ad")
    _symlink(in_dir / "summary.txt", out_dir / "summary.txt")

    print(f"\nwrote {summary['n_pseudocells']} human-transferred pseudo-cells -> {out_dir}/dataset")
    print(f"  next: embed with species=0 -> python finetune/genecompass/embed_cells.py "
          f"--dataset {out_dir}/dataset --output {out_dir}/embeddings --species 0 "
          f"(or: python pipeline/run_stage12.py --label {args.label})")


if __name__ == "__main__":
    main()
