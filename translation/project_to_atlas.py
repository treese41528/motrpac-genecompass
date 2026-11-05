#!/usr/bin/env python3
"""
project_to_atlas.py -- E.3: anchor the rat->human transfer to a real human atlas (Tabula Sapiens).
Pre-registered in translation/ATLAS_PROJECTION_PREREG.md.

The transfer (transfer_to_human.py) re-expresses each rat pseudo-cell as a human profile and embeds it with
species=0. Its only checks so far reuse rat labels. Here real human cells are put through the same path:
Tabula Sapiens 10x cells -> one pseudobulk per (donor, cell type) -> the transfer's exact tokenization
(normalize_total to the rat run's target sum over all genes, human tokens + human medians, top-2048) ->
embed_cells.py --species 0 with the same checkpoint. Then: does each transferred rat pseudo-cell sit
nearest to the human cell type it should?

  prep   python translation/project_to_atlas.py prep --tissue Blood --out <atlas-root>
         (then: embed_cells.py --species 0 --dataset <atlas-root>/Blood/dataset --output <atlas-root>/Blood/embeddings)
  score  python translation/project_to_atlas.py score --atlas-root <atlas-root> --out <dir>
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lib"))
sys.path.insert(0, str(ROOT / "pipeline" / "05_tokenization"))
sys.path.insert(0, str(ROOT / "translation"))
from gene_utils import load_config, resolve_path                                      # noqa: E402
from transfer_to_human import (HUMAN_SPECIES_ID, _base, load_human_median_dict,      # noqa: E402
                               load_human_token_dict)
from tokenize_corpus import (CELL_BATCH_SIZE, TOP_N_DEFAULT, build_eligible_gene_arrays,  # noqa: E402
                             build_gene_index, map_varnames_to_eligible, tokenize_cell_batch)

MIN_CELLS = 20          # cells per (donor, cell type) pseudobulk
# MoTrPAC tissue label (genecompass_input/<label>) -> Tabula Sapiens tissue file
TISSUES = {"blood": "Blood", "lung": "Lung", "skmgn": "Muscle", "skmvl": "Muscle", "heart": "Heart",
           "liver": "Liver", "kidney": "Kidney", "watsc": "Fat", "bat": "Fat", "smlint": "Small_Intestine",
           "testes": "Testis"}


def prep(args):
    import anndata as ad
    import scipy.sparse as sp
    from datasets import Dataset
    cfg = load_config(); paths, dc = cfg["paths"], cfg["deconvolution"]
    target_sum = float(args.target_sum or dc.get("pseudocell_target_sum", 6500.0))
    src = Path(args.atlas_dir) / f"{args.tissue}.h5ad"
    a = ad.read_h5ad(src)
    X = a.raw.X if a.raw is not None else a.X                        # CELLxGENE: raw counts in .raw when X is normalized
    genes = np.array([_base(g) for g in (a.raw.var_names if a.raw is not None else a.var_names)])
    obs = a.obs
    tenx = obs["assay"].astype(str).str.contains("10x").values
    print(f"{args.tissue}: {a.n_obs:,} cells, {tenx.sum():,} 10x; assays {sorted(obs['assay'].astype(str).unique())}")
    key = obs["donor_id"].astype(str) + "|" + obs["cell_type"].astype(str)
    groups = {}
    for i in np.flatnonzero(tenx):
        groups.setdefault(key.iloc[i], []).append(i)
    groups = {k: v for k, v in sorted(groups.items()) if len(v) >= MIN_CELLS}
    X = sp.csr_matrix(X)
    sums = np.vstack([np.asarray(X[v].sum(axis=0)).ravel() for v in groups.values()])
    if not np.allclose(sums, np.round(sums)):
        raise SystemExit("ERROR: matrix is not integer counts")
    tok = load_human_token_dict(Path(resolve_path(cfg, paths["genecompass_tokens"])))
    med = load_human_median_dict(Path(resolve_path(cfg, paths["genecompass_medians"])))
    eligible_ids, eligible_tokens, eligible_medians = build_eligible_gene_arrays(tok, med)
    col_idx, gene_idx = map_varnames_to_eligible(list(genes), build_gene_index(eligible_ids))
    gi = np.asarray(gene_idx, dtype=np.int32)
    lib = sums.sum(1, keepdims=True); lib[lib == 0] = 1.0
    Xn = (sums / lib * target_sum)[:, col_idx]                         # normalize over ALL genes, then restrict
    ids, vals, lens = [], [], []
    for s in range(0, Xn.shape[0], CELL_BATCH_SIZE):
        i_, v_, l_ = tokenize_cell_batch(Xn[s:s + CELL_BATCH_SIZE], eligible_tokens[gi], eligible_medians[gi], TOP_N_DEFAULT)
        ids += i_; vals += v_; lens += l_
    keep = np.asarray(lens) >= 200
    ks = list(groups)
    rows = {"input_ids": [r for r, k in zip(ids, keep) if k], "values": [r for r, k in zip(vals, keep) if k],
            "length": [[l] for l, k in zip(lens, keep) if k], "species": [[HUMAN_SPECIES_ID]] * int(keep.sum()),
            "cell_id": [k for k, kk in zip(ks, keep) if kk],
            "donor": [k.split("|")[0] for k, kk in zip(ks, keep) if kk],
            "cell_type": [k.split("|", 1)[1] for k, kk in zip(ks, keep) if kk],
            "n_cells": [len(groups[k]) for k, kk in zip(ks, keep) if kk]}
    out = Path(args.out) / args.tissue; out.mkdir(parents=True, exist_ok=True)
    Dataset.from_dict(rows).save_to_disk(str(out / "dataset"))
    flat = np.array([v for r in rows["values"] for v in r if v > 0], dtype=np.float32)
    summ = {"source": str(src), "n_cells_10x": int(tenx.sum()), "n_pseudobulks": int(keep.sum()),
            "n_dropped_lt_200_genes": int((~keep).sum()), "min_cells": MIN_CELLS, "target_sum": target_sum,
            "n_eligible_genes": len(col_idx), "value_median": float(np.median(flat)),
            "cell_types": {c: rows["cell_type"].count(c) for c in sorted(set(rows["cell_type"]))}}
    json.dump(summ, open(out / "prep_summary.json", "w"), indent=2)
    print(json.dumps({k: v for k, v in summ.items() if k != "cell_types"}))


def load_map(path):
    """correspondence TSV: tissue_label, rat_cell_type, ts_cell_types ('; '-separated; empty = no counterpart)."""
    m = {}
    for r in csv.DictReader(open(path), delimiter="\t"):
        ts = {s.strip() for s in r["ts_cell_types"].split(";") if s.strip()}
        if ts:
            m[(r["tissue_label"], r["rat_cell_type"])] = ts
    return m


def score(args):
    from datasets import load_from_disk
    cmap = load_map(args.map)
    rng = np.random.default_rng(0)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    per_type, data = [], []
    for lab, tsname in TISSUES.items():
        rd = Path(args.human_root) / lab
        ad_ = Path(args.atlas_root) / tsname
        if not (ad_ / "embeddings" / "cell_embeddings.npy").exists():
            print(f"skip {lab}: no atlas embeddings for {tsname}"); continue
        R = np.load(rd / "embeddings" / "cell_embeddings.npy"); rct = np.array(load_from_disk(str(rd / "dataset"))["cell_type"])
        T = np.load(ad_ / "embeddings" / "cell_embeddings.npy"); tct = np.array(load_from_disk(str(ad_ / "dataset"))["cell_type"])
        assert len(R) == len(rct) and len(T) == len(tct)
        names = sorted(set(tct))
        variants = {"centered": (R - R.mean(0), T - T.mean(0)), "raw": (R, T)}
        for var, (Rv, Tv) in variants.items():
            C = np.vstack([Tv[tct == n].mean(0) for n in names])
            Rn = Rv / np.linalg.norm(Rv, axis=1, keepdims=True); Cn = C / np.linalg.norm(C, axis=1, keepdims=True)
            S = Rn @ Cn.T                                              # pseudo-cells x TS cell types
            top = S.argmax(1)
            for ct in sorted(set(rct)):
                match = cmap.get((lab, ct))
                if not match:
                    continue
                idx = np.flatnonzero(rct == ct)
                ok = np.array([n in match for n in names])
                if not ok.any():
                    print(f"WARNING {lab}/{ct}: none of {sorted(match)} is a {tsname} cell type with >= 1 pseudobulk")
                per_type.append({"variant": var, "tissue": lab, "ts_tissue": tsname, "rat_cell_type": ct,
                                 "n": len(idx), "n_ts_types": len(names), "n_match": int(ok.sum()),
                                 "top1_acc": float(ok[top[idx]].mean()),
                                 "chance": float(ok.mean()),
                                 "top1_mode": names[np.bincount(top[idx], minlength=len(names)).argmax()]})
                data.append((var, lab, ct, top[idx], ok))
    P = per_type
    with open(out / "atlas_projection_per_type.tsv", "w") as f:
        w = csv.DictWriter(f, fieldnames=list(P[0]), delimiter="\t"); w.writeheader(); w.writerows(P)
    lines = []
    for var in ("centered", "raw"):
        D = [d for d in data if d[0] == var]
        A = float(np.mean([d[4][d[3]].mean() for d in D]))
        # null: shuffle which TS centroid carries which name, within each tissue (same draw for all its rat types)
        by_t = {}
        for d in D:
            by_t.setdefault(d[1], []).append(d)
        null = np.empty(args.n_perm)
        for b in range(args.n_perm):
            acc = []
            for lab, ds in by_t.items():
                perm = rng.permutation(len(ds[0][4]))
                acc += [d[4][perm][d[3]].mean() for d in ds]
            null[b] = np.mean(acc)
        p = (1 + (null >= A).sum()) / (1 + args.n_perm)
        verdict = ("PASS" if p < 0.05 and A >= 2 * null.mean() else "WEAK" if p < 0.05 else "FAIL") if var == "centered" else "(secondary, no verdict)"
        lines.append(f"{var}: macro top-1 accuracy {A:.3f} over {len(D)} rat cell types; null mean {null.mean():.3f} "
                     f"(95th pct {np.percentile(null, 95):.3f}); p = {p:.4f} ({args.n_perm} perms) -> {verdict}")
        for lab in by_t:
            r = [x for x in P if x["variant"] == var and x["tissue"] == lab]
            lines.append(f"    {lab}: {np.mean([x['top1_acc'] for x in r]):.3f} (chance {np.mean([x['chance'] for x in r]):.3f}, {len(r)} types)")
    txt = "\n".join(lines); print(txt); (out / "summary.txt").write_text(txt + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prep")
    p.add_argument("--tissue", required=True, choices=sorted(set(TISSUES.values())))
    p.add_argument("--atlas-dir", default=str(ROOT / "data/external/tabula_sapiens"))
    p.add_argument("--out", default=str(ROOT / "data/deconvolution/atlas_tabula_sapiens"))
    p.add_argument("--target-sum", type=float, default=None)
    s = sub.add_parser("score")
    s.add_argument("--atlas-root", default=str(ROOT / "data/deconvolution/atlas_tabula_sapiens"))
    s.add_argument("--human-root", default=None, help="default: <deconvolution.genecompass_input_dir>_human")
    s.add_argument("--map", default=str(ROOT / "translation/atlas_correspondence.tsv"))
    s.add_argument("--out", default=None, help="default: <atlas-root>/score")
    s.add_argument("--n-perm", type=int, default=10000)
    a = ap.parse_args()
    if a.cmd == "prep":
        prep(a)
    else:
        if a.human_root is None:
            cfg = load_config(); a.human_root = str(resolve_path(cfg, cfg["deconvolution"]["genecompass_input_dir"])) + "_human"
        a.out = a.out or str(Path(a.atlas_root) / "score")
        score(a)


if __name__ == "__main__":
    main()
