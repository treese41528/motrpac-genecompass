#!/usr/bin/env python
# coding: utf-8
r"""
build_grn_pooled.py -- Module B refinement: pseudobulk + bootstrap differential GRN.

The per-cell pilot (build_grn.py) averaged the target shift over each arm's cells; with only ~10 control
pseudo-cells the differential (trained - control) was noise-dominated and produced one-arm-only artifact
edges (a target passing the survival filter in one arm but not the other). This refinement fixes both:

  1. PSEUDOBULK each arm -- collapse the arm's pseudo-cells into ONE representative profile (per-token mean
     value over the arm, top-2048). Perturbing that single profile is deterministic (no per-cell averaging
     noise) and both arms carry the SAME gene set (no survival-filter asymmetry).
  2. BOOTSTRAP -- resample the arm's cells with replacement B times, re-pseudobulk, re-perturb; the spread
     of the differential across resamples is the per-edge uncertainty. Trustworthy edges have |z| >= 2
     (point differential large relative to its bootstrap SD).

"Dose-pooling": the trained arm pools weeks 1/2/4/8 vs the week-0 control (the MoTrPAC design gives 10
control / 40 trained muscle pseudo-cells). Edge weight is the per-gene TARGET shift (not the near-noise
CLS shift). Reuses the arm split (augur meta) and the engine primitives from build_grn / perturb_cells.

Usage:
  python finetune/genecompass/build_grn_pooled.py --model-dir <...> \
      --dataset data/deconvolution/genecompass_input/skmvl/dataset \
      --meta data/deconvolution/augur_input/skmvl/meta.tsv --cell-type "Skeletal muscle" \
      --out data/deconvolution/grn/skmvl_skeletal_muscle_pooled.tsv --bootstrap 30 --device cuda
"""
import argparse
import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(os.environ.get("PIPELINE_ROOT", ".")).resolve()
sys.path.insert(0, str(_ROOT / "finetune" / "genecompass"))
import perturb_cells as pc          # noqa: E402
from build_grn import load_arms     # noqa: E402

VOCAB = 55275
L = 2048


def pseudobulk(ids, vals):
    """arm cells (ids[N,L], vals[N,L]) -> one representative profile (pid[L], pvl[L]): per-token mean
    value over the arm (absent = 0), top-2048 tokens."""
    acc = torch.zeros(VOCAB)
    acc.index_add_(0, ids.reshape(-1).long(), vals.reshape(-1).float())
    acc[0] = 0.0                                    # ignore pad token
    acc /= float(ids.shape[0])
    k = int(min(L, (acc > 0).sum().item()))
    top = torch.topk(acc, k)
    pid = torch.zeros(L, dtype=torch.long)
    pvl = torch.zeros(L, dtype=torch.float)
    pid[:k] = top.indices.long()
    pvl[:k] = top.values.float()
    return pid, pvl


@torch.no_grad()
def perturb_profile(model, pid, pvl, token, device, invert):
    """Delete `token` from one profile; return {target_ensrnog: shift} for all surviving genes (no
    survival filter -- it's a single profile)."""
    if int((pid == token).sum()) == 0:
        return {}
    ni, nv, rem = pc.delete_token(pid, pvl, token)
    _, g0 = pc.embed(model, pid[None, :], pvl[None, :], device)
    _, g1 = pc.embed(model, ni[None, :], nv[None, :], device)
    p = rem[0]
    keep = torch.ones(g0.shape[1], dtype=torch.bool)
    keep[p] = False
    orig = g0[0][keep]
    pert = g1[0][: orig.shape[0]]
    surv = pid[keep].tolist()
    sh = pc._cos_shift(orig, pert)
    return {invert.get(t, str(t)): float(s) for t, s in zip(surv, sh) if t != 0}


def diff(tc, tt):
    return {k: tt.get(k, 0.0) - tc.get(k, 0.0) for k in set(tc) | set(tt)}


def load_arm_sex(dataset_path, meta_path, cell_type):
    """Per-cell sex for the control and trained arms, in the SAME order load_arms() emits them.

    Mirrors load_arms exactly (iterate the dataset in order, keep rows of this cell type whose meta label is
    'control'/'trained'), so index i of sex_ctrl corresponds to index i of ids_ctrl. Used only by the
    sex-stratified permutation null.
    """
    from datasets import load_from_disk
    ds = load_from_disk(dataset_path)
    meta = {}
    for r in csv.DictReader(open(meta_path), delimiter="\t"):
        meta[(r["sample"], r["cell_type"])] = (r.get("label", "").strip().lower(), r.get("sex", "").strip().lower())
    arms = {"control": [], "trained": []}
    for r in ds:
        if r["cell_type"] != cell_type:
            continue
        lab, sx = meta.get((r["sample"], r["cell_type"]), ("", ""))
        if lab in arms:
            arms[lab].append(sx)
    return np.array(arms["control"]), np.array(arms["trained"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--cell-type", required=True)
    ap.add_argument("--tf-file", default=str(_ROOT / "deconvolution" / "reference" / "rat_tf_ensrnog.tsv"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--bootstrap", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--top-edges", type=int, default=0,
                    help="candidate edges per TF, by |point delta|. 0 = NO CAP (keep every surviving "
                         "target; the default). A cap selects edges on the same statistic that z then "
                         "tests (winner's curse) and makes 'hub strength = #confident edges' meaningless, "
                         "since every TF would carry exactly `cap` candidates. Costs nothing to disable: "
                         "perturb_profile already computes every target's shift on every bootstrap pass.")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-tfs", type=int, default=None)
    ap.add_argument("--permute-seed", type=int, default=-1,
                    help="PERMUTATION NULL. >=0 shuffles the trained/control labels among the cell type's "
                         "pseudo-cells (preserving the arm SIZES) before anything else, so there is no real "
                         "exercise contrast left. Everything downstream is identical. The resulting |z| "
                         "distribution is the null, and is what calibrates the |z|>=2 'confident' threshold: "
                         "empirical FDR(t) = P_null(|z|>=t) / P_observed(|z|>=t). Without this, |z|>=2 is a "
                         "relative ranking dressed up as a significance claim.")
    args = ap.parse_args()

    tok = pc.load_token_dict()
    invert = {v: k for k, v in tok.items()}
    sym = {}
    for r in csv.DictReader(open(_ROOT / "data" / "training" / "ortholog_mappings" / "rat_token_mapping.tsv"),
                            delimiter="\t"):
        sym[r["rat_gene"]] = r.get("rat_symbol", "")
    tfs = [l.split("\t")[0].strip() for i, l in enumerate(open(args.tf_file)) if i > 0 and l.strip()]
    tfs = [g for g in tfs if g in tok]
    if args.max_tfs:
        tfs = tfs[:args.max_tfs]

    (idc, vlc), (idt, vlt) = load_arms(args.dataset, args.meta, args.cell_type)
    nC, nT = idc.shape[0], idt.shape[0]

    if args.permute_seed >= 0:
        # NULL: shuffle arm membership, keeping arm sizes fixed -> destroys the exercise contrast and nothing
        # else, so the resulting |z| distribution is the null.
        #
        # STRATIFIED BY SEX, and this is NOT optional. The real arms are exactly sex-balanced (e.g. 5M/5F
        # control vs 20M/20F trained), and SEX is the DOMINANT axis in this data. A *free* permutation does
        # not preserve that balance -- one draw can hand the control arm 8 males -- which manufactures a
        # spurious arm difference driven by sex rather than exercise, inflating the null tail and making the
        # FDR estimate meaningless (conservative, and measuring the wrong contrast). Permuting WITHIN each
        # sex stratum holds sex fixed and isolates the exercise contrast, which is what we want to null out.
        sexc, sext = load_arm_sex(args.dataset, args.meta, args.cell_type)
        prng = np.random.default_rng(args.permute_seed)
        all_id = torch.cat([idc, idt], dim=0)
        all_vl = torch.cat([vlc, vlt], dim=0)
        all_sx = np.concatenate([sexc, sext])
        n_ctrl_by_sex = {s: int((sexc == s).sum()) for s in np.unique(all_sx)}
        ctrl_idx, trn_idx = [], []
        for s in np.unique(all_sx):
            pool = np.where(all_sx == s)[0]
            prng.shuffle(pool)
            k = n_ctrl_by_sex.get(s, 0)
            ctrl_idx.extend(pool[:k]); trn_idx.extend(pool[k:])
        ctrl_idx, trn_idx = np.array(ctrl_idx), np.array(trn_idx)
        idc, vlc = all_id[ctrl_idx], all_vl[ctrl_idx]
        idt, vlt = all_id[trn_idx], all_vl[trn_idx]
        print(f"[{args.cell_type}] *** SEX-STRATIFIED PERMUTATION NULL (seed={args.permute_seed}) *** "
              f"per-sex control counts preserved: {n_ctrl_by_sex}", flush=True)

    print(f"[{args.cell_type}] control={nC} trained={nT}; TFs={len(tfs)}; bootstrap={args.bootstrap}", flush=True)
    model = pc.load_model(args.model_dir, args.device)
    rng = np.random.default_rng(args.seed)

    # point estimate on the full-arm pseudobulks
    pbc, pbt = pseudobulk(idc, vlc), pseudobulk(idt, vlt)
    point, cand, cidx = {}, {}, {}
    expressed = []
    for g in tfs:
        d = diff(perturb_profile(model, *pbc, tok[g], args.device, invert),
                 perturb_profile(model, *pbt, tok[g], args.device, invert))
        if not d:
            continue
        expressed.append(g)
        point[g] = d
        ks = sorted(d, key=lambda k: -abs(d[k]))
        if args.top_edges and args.top_edges > 0:      # 0 = no cap (default)
            ks = ks[:args.top_edges]
        cand[g] = ks
        cidx[g] = {k: i for i, k in enumerate(ks)}
    ncand = sum(len(v) for v in cand.values())
    print(f"expressed TFs: {len(expressed)}; candidate edges: {ncand:,} "
          f"({'UNCAPPED' if not args.top_edges else f'capped at {args.top_edges}/TF'}); bootstrapping...",
          flush=True)

    # bootstrap: resample each arm (WHOLE CELLS -- one index vector per arm, used for BOTH ids and values;
    # drawing ids and values independently would pair cell i's genes with cell j's expression), re-pseudobulk,
    # perturb every expressed TF, accumulate running sum/sum-of-squares so memory does not scale with B.
    bsum = {g: np.zeros(len(cand[g])) for g in expressed}
    bsq = {g: np.zeros(len(cand[g])) for g in expressed}
    for b in range(args.bootstrap):
        bi = rng.integers(0, nC, nC)                   # one draw per arm -> resamples CELLS, not id/val pairs
        bt = rng.integers(0, nT, nT)
        pbc_b = pseudobulk(idc[bi], vlc[bi])
        pbt_b = pseudobulk(idt[bt], vlt[bt])
        for g in expressed:
            d = diff(perturb_profile(model, *pbc_b, tok[g], args.device, invert),
                     perturb_profile(model, *pbt_b, tok[g], args.device, invert))
            ci = cidx[g]
            vec = np.zeros(len(cand[g]))
            for k, v in d.items():
                p = ci.get(k)
                if p is not None:
                    vec[p] = v
            bsum[g] += vec
            bsq[g] += vec * vec
        if (b + 1) % 10 == 0:
            print(f"  bootstrap {b + 1}/{args.bootstrap}", flush=True)

    B = float(args.bootstrap)
    edges = []
    for g in expressed:
        mean = bsum[g] / B
        var = np.maximum(bsq[g] / B - mean * mean, 0.0)   # population SD, matches np.std(ddof=0)
        sds = np.sqrt(var)
        for i, k in enumerate(cand[g]):
            d = point[g][k]
            sd = float(sds[i])
            z = d / sd if sd > 0 else 0.0
            edges.append((g, sym.get(g, ""), k, sym.get(k, ""), round(d, 6), round(sd, 6), round(z, 2)))
    edges.sort(key=lambda e: -abs(e[6]))
    conf = [e for e in edges if abs(e[6]) >= 2]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        f.write("tf\ttf_symbol\ttarget\ttarget_symbol\tdelta\tboot_sd\tz\n")
        for e in edges:
            f.write("\t".join(str(x) for x in e) + "\n")
    print(f"wrote {len(edges)} candidate edges ({len(conf)} confident |z|>=2) from {len(expressed)} TFs "
          f"-> {args.out}")


if __name__ == "__main__":
    main()
