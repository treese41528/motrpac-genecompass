#!/usr/bin/env python
# coding: utf-8
r"""
build_grn_pooled.py -- Module B: pseudobulk + bootstrap differential network (a model read-out).

For one tissue x cell type, each arm's pseudo-cells are collapsed into one profile, every TF is deleted from
both profiles, and an edge is the trained-minus-control difference of a target's embedding shift.

  1. PSEUDOBULK each arm: per-token mean value over the arm's pseudo-cells (absent = 0), top 2,048 tokens.
     The two profiles are selected independently, so they do NOT hold the same genes: in the six default
     networks of 2026-11, 34 to 102 genes per arm were in one arm's profile only (audit finding 9.1).
  2. EDGE = trained shift - control shift, on SHARED genes only (docs/audit/remediation/STEP3_PREREG.md 8.1).
     An edge exists only where the TF and the target are in both arms' profiles. Before 2026-11-14 a gene
     missing from one arm was differenced against 0, so those "edges" were one arm's raw shift.
  3. BOOTSTRAP: resample each arm's cells with replacement B times, rebuild both profiles, repeat 1-2.
     boot_sd is the spread of the difference over the resamples (an edge that does not exist in a resample
     counts as 0 there, as before) and z = point difference / boot_sd.

How to read the output. z is a ranking, not an error rate: the decoy null found edges with |z| >= 2 as frequent
for expression-matched non-TF genes as for TFs (deconvolution/GRN_DECOY_NULL_RESULTS.md). Hub strength, the sum
of |delta| over ALL of a TF's edges, follows the TF's own expression. The tables are model read-outs.

"Dose-pooling": the trained arm pools weeks 1/2/4/8 against the week-0 control (10 control / 40 trained
pseudo-cells in the MoTrPAC design). The edge weight is the per-gene TARGET shift (not the near-noise CLS
shift). Reuses the arm split (augur meta) and the engine primitives from build_grn / perturb_cells.

Outputs, for --out <dir>/<name>_pooled.tsv:
  <name>_pooled.tsv                 edges: tf, tf_symbol, target, target_symbol, delta, boot_sd, z
                                    (+ decoy_for, match_dist with --decoy-seed)
  <name>_pooled_arms.tsv            one row per TF found in at least one arm: in_control, in_trained,
                                    in_both_arms, n_targets_both, n_targets_control_only,
                                    n_targets_trained_only, n_edges, n_boot_in_both, mean_frac_edges_in_boot
  <name>_pooled_arms_summary.tsv    one row for the network: genes and TFs in both arms / one arm only, edges
  <name>_pooled_decoy_match.tsv     (--decoy-seed only) one row per decoy: the TF it replaces and match_dist
  provenance_<name>_pooled.json     commit, command, inputs, md5 of the weights file, edge rule
With --decoy-seed the column tf of every table holds the decoy gene, the sidecar rows describe the decoys, and
n_tf_listed is the number of decoys (one per expressed TF).

Usage:
  python finetune/genecompass/build_grn_pooled.py --model-dir <...> \
      --dataset data/deconvolution/genecompass_input_v2/skmvl/dataset \
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
# Name of the edge rule, written into every network's summary and provenance record. pipeline/run_stage13.py
# and deconvolution/grn_decoy_fdr.py read it to tell a network built on shared genes from an earlier one.
EDGE_RULE = "shared_genes"


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
    """Trained-minus-control difference of one TF deletion, on the targets present in BOTH arms.

    tc, tt: {target: shift} from the control and the trained profile; {} when the TF is not in that profile.
    Registration 8.1 (docs/audit/remediation/STEP3_PREREG.md): an edge exists only where the TF and the target
    are in both arms' profiles. A target in one arm only gets no edge, and a TF in one arm only gets no edges
    at all (the intersection with an empty dict is empty). The earlier rule took the union and filled the
    missing side with 0, which made those edges one arm's raw shift rather than a difference.
    The point estimate and every bootstrap resample go through this function, in rat and in human space
    (build_grn_human.py imports it).
    """
    return {k: tt[k] - tc[k] for k in tc.keys() & tt.keys()}


def arm_counts(tc, tt, in_control=None, in_trained=None):
    """Sidecar counts for one TF (registration 8.1): which arms hold the TF and how its targets split.

    tc, tt are the two perturbation results that diff() receives. n_targets_both is the number of edges the
    shared-gene rule makes; n_targets_control_only and n_targets_trained_only are the targets the earlier
    union rule differenced against 0. For a TF in one arm only, every target of that arm is one-arm-only.
    in_control / in_trained default to "the perturbation returned targets"; pass the profile's own answer
    when it is at hand.
    """
    kc, kt = set(tc), set(tt)
    in_c = bool(kc) if in_control is None else bool(in_control)
    in_t = bool(kt) if in_trained is None else bool(in_trained)
    return {"in_control": in_c, "in_trained": in_t, "in_both_arms": in_c and in_t,
            "n_targets_both": len(kc & kt),
            "n_targets_control_only": len(kc - kt),
            "n_targets_trained_only": len(kt - kc)}


def profile_overlap(pbc, pbt):
    """Genes (non-pad tokens) of the control and trained profiles: how many are in both and in one only."""
    c = set(pbc[0].tolist()) - {0}
    t = set(pbt[0].tolist()) - {0}
    return {"n_genes_control": len(c), "n_genes_trained": len(t), "n_genes_both": len(c & t),
            "n_genes_control_only": len(c - t), "n_genes_trained_only": len(t - c)}


ARM_COLUMNS = ["tf", "tf_symbol", "in_control", "in_trained", "in_both_arms", "n_targets_both",
               "n_targets_control_only", "n_targets_trained_only", "n_edges", "n_boot_in_both",
               "mean_frac_edges_in_boot"]
SUMMARY_COLUMNS = ["network", "cell_type", "n_control_cells", "n_trained_cells",
                   "n_genes_control", "n_genes_trained", "n_genes_both", "n_genes_control_only",
                   "n_genes_trained_only",
                   "n_tf_listed", "n_tf_in_both", "n_tf_control_only", "n_tf_trained_only", "n_tf_neither",
                   "n_edges", "n_pairs_union", "n_pairs_one_arm_only", "share_pairs_one_arm_only",
                   "bootstrap", "seed", "edge_rule"]


def sidecar_paths(out):
    """Sidecar file names for a network table. None of them ends in _pooled.tsv, so the globs that collect
    networks (self-consistency score, decoy scorer, GRN reuse) do not pick them up."""
    out = str(out)
    stem = out[:-4] if out.endswith(".tsv") else out
    return {"arms": stem + "_arms.tsv", "summary": stem + "_arms_summary.tsv",
            "match": stem + "_decoy_match.tsv",
            "provenance": str(Path(out).parent / f"provenance_{Path(stem).name}.json")}


def network_summary(rows, overlap, n_tf_listed, n_edges, **extra):
    """One-row summary of a network from its per-TF sidecar rows (registration 8.1: "per network the table
    records how many TFs and targets were in one arm only").

    n_pairs_union is the number of TF x target pairs the earlier union rule differenced (shared and
    one-arm-only, over every TF found in at least one arm); n_pairs_one_arm_only is the part of it that had
    one side filled with 0 and now gets no edge.
    """
    both = sum(r["in_both_arms"] for r in rows)
    c_only = sum(r["in_control"] and not r["in_trained"] for r in rows)
    t_only = sum(r["in_trained"] and not r["in_control"] for r in rows)
    union = sum(r["n_targets_both"] + r["n_targets_control_only"] + r["n_targets_trained_only"] for r in rows)
    shared = sum(r["n_targets_both"] for r in rows)
    s = dict(extra)
    s.update(overlap)
    s.update({"n_tf_listed": n_tf_listed, "n_tf_in_both": both, "n_tf_control_only": c_only,
              "n_tf_trained_only": t_only, "n_tf_neither": n_tf_listed - len(rows),
              "n_edges": n_edges, "n_pairs_union": union, "n_pairs_one_arm_only": union - shared,
              "share_pairs_one_arm_only": round((union - shared) / union, 4) if union else 0.0,
              "edge_rule": EDGE_RULE})
    return s


def _cell(v):
    """Table cell: booleans as TRUE/FALSE (the convention of the project's R-written tables), None as NA."""
    if v is None:
        return "NA"
    if isinstance(v, (bool, np.bool_)):
        return "TRUE" if v else "FALSE"
    return str(v)


def write_arm_sidecars(out, rows, summary, extra_columns=()):
    """Write <out stem>_arms.tsv (one row per TF) and <out stem>_arms_summary.tsv (one row). Returns the paths."""
    p = sidecar_paths(out)
    cols = ARM_COLUMNS + list(extra_columns)
    with open(p["arms"], "w") as f:
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join(_cell(r.get(c)) for c in cols) + "\n")
    with open(p["summary"], "w") as f:
        f.write("\t".join(SUMMARY_COLUMNS) + "\n")
        f.write("\t".join(_cell(summary.get(c)) for c in SUMMARY_COLUMNS) + "\n")
    return p["arms"], p["summary"]


def summary_line(s):
    """The one-line summary printed for a network."""
    return (f"[{s['network']}] edge rule {s['edge_rule']}: genes in both arms {s['n_genes_both']}, control only "
            f"{s['n_genes_control_only']}, trained only {s['n_genes_trained_only']}; TFs in both arms "
            f"{s['n_tf_in_both']}, control only {s['n_tf_control_only']}, trained only {s['n_tf_trained_only']} "
            f"(of {s['n_tf_listed']} listed); edges {s['n_edges']:,}; TF x target pairs with one arm missing, "
            f"not made into edges: {s['n_pairs_one_arm_only']:,} of {s['n_pairs_union']:,} "
            f"({100 * s['share_pairs_one_arm_only']:.1f}%)")


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


def match_decoys(tfs, pbc, pbt, tok, invert, tf_file, seed):
    """Expressed TFs -> expression-matched non-TF decoys. Expression = mean of the token's value in the two arm
    profiles (absent = 0); a TF is matched if it is present in either profile, and the pool is every non-TF
    token present in either profile. Each TF, in a seeded random order, takes the unused non-TF token nearest
    in log1p(expression).

    The matching is the registered one (GRN_DECOY_NULL_PREREG.md) and is unchanged by registration 8.1: the
    same TFs get the same decoys as before. What changed is diff(): a TF or a decoy that is in one arm only
    now gets no edges, so it is matched here and then drops out of its network; the sidecar tables count them.

    Returns (decoy ENSRNOG list, {decoy: TF}, {decoy: {tf_expr, decoy_expr, match_dist}}), where match_dist is
    |log1p(decoy_expr) - log1p(tf_expr)|, the distance the match minimised (audit finding 9.3: it was never
    written out).
    """
    val = defaultdict(float)
    for pid, pvl in (pbc, pbt):
        for t, v in zip(pid.tolist(), pvl.tolist()):
            if t != 0:
                val[t] += v / 2.0
    all_tf = {tok[l.split("\t")[0].strip()] for i, l in enumerate(open(tf_file))
              if i > 0 and l.strip() and l.split("\t")[0].strip() in tok}
    expressed = [g for g in tfs if tok[g] in val]
    pool = sorted((t for t in val if t not in all_tf and t in invert), key=lambda t: val[t])
    x = np.log1p(np.array([val[t] for t in pool]))
    used = np.zeros(len(pool), dtype=bool)
    order = np.random.default_rng(seed).permutation(len(expressed))
    out, info = {}, {}
    for i in order:
        g = expressed[i]
        d = np.abs(x - np.log1p(val[tok[g]]))
        d[used] = np.inf
        j = int(np.argmin(d))
        used[j] = True
        out[invert[pool[j]]] = g
        info[invert[pool[j]]] = {"tf_expr": float(val[tok[g]]), "decoy_expr": float(val[pool[j]]),
                                 "match_dist": float(d[j])}
    decoys = [k for k in sorted(out, key=lambda k: expressed.index(out[k]))]
    return decoys, out, info


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
                         "tests (winner's curse) and makes 'hub strength = #edges with |z|>=2' meaningless, "
                         "since every TF would carry exactly `cap` candidates. Costs nothing to disable: "
                         "perturb_profile already computes every target's shift on every bootstrap pass.")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-tfs", type=int, default=None)
    ap.add_argument("--permute-seed", type=int, default=-1,
                    help="ABANDONED PERMUTATION NULL, kept only so its 2026-07 tables can be reproduced. >=0 "
                         "shuffles the trained/control labels among the cell type's pseudo-cells within sex "
                         "(arm sizes kept) and runs the identical build. It does NOT calibrate |z|: control is "
                         "week 0 by design, so any shuffle mixes training weeks into the control arm and the "
                         "permuted z is not distributed like the observed z under the null "
                         "(deconvolution/GRN_NULL_WHY_ABANDONED.md). The null in use is --decoy-seed.")
    ap.add_argument("--decoy-seed", type=int, default=-1,
                    help="DECOY NULL (GRN_DECOY_NULL_PREREG.md). >=0 replaces every expressed TF with a non-TF gene "
                         "of the nearest pseudobulk expression (mean of the two arm profiles; drawn without "
                         "replacement, TFs matched in a seeded random order), then runs the identical pipeline. "
                         "Labels, arms, bootstrap and z are untouched, so under H0 'this TF acts like a random "
                         "gene of the same expression' decoy and TF z are identically distributed: "
                         "FDR(t) = P_decoy(|z|>=t) / P_TF(|z|>=t). Writes tf = the decoy gene, decoy_for = the TF "
                         "and match_dist = |log1p(decoy expression) - log1p(TF expression)|; every match, with "
                         "or without edges, is also listed in <out stem>_decoy_match.tsv.")
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

    n_tf_listed = len(tfs)

    (idc, vlc), (idt, vlt) = load_arms(args.dataset, args.meta, args.cell_type)
    if idc is None or idt is None:
        sys.exit(f"ERROR: empty arm(s) for cell type {args.cell_type!r} in {args.dataset}")
    nC, nT = idc.shape[0], idt.shape[0]

    if args.permute_seed >= 0:
        # Abandoned permutation null (see --permute-seed): shuffle arm membership, keeping arm sizes fixed.
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
    ctok, ttok = set(pbc[0].tolist()) - {0}, set(pbt[0].tolist()) - {0}
    decoy_for, match = {}, {}
    if args.decoy_seed >= 0:
        tf_list = tfs
        tfs, decoy_for, match = match_decoys(tfs, pbc, pbt, tok, invert, args.tf_file, args.decoy_seed)
        n_tf_listed = len(tfs)
        print(f"[{args.cell_type}] *** DECOY NULL (seed={args.decoy_seed}) *** {len(tfs)} expressed TFs replaced "
              f"by expression-matched non-TF genes", flush=True)
    # Registration 8.1: a TF gets edges only if it is in both arms' profiles, and then only to the targets
    # that are in both. Every TF found in at least one arm gets a sidecar row saying where it and its targets
    # were, so the pairs that the earlier union rule differenced against 0 are counted, not silently lost.
    point, cand, cidx = {}, {}, {}
    expressed = []
    arm_rows = {}
    for g in tfs:
        in_c, in_t = tok[g] in ctok, tok[g] in ttok
        if not (in_c or in_t):
            continue                                   # in neither profile: nothing to delete
        tc = perturb_profile(model, *pbc, tok[g], args.device, invert)
        tt = perturb_profile(model, *pbt, tok[g], args.device, invert)
        row = arm_counts(tc, tt, in_c, in_t)
        row.update({"tf": g, "tf_symbol": sym.get(g, ""), "n_edges": 0, "n_boot_in_both": None,
                    "mean_frac_edges_in_boot": None})
        if decoy_for:
            row["decoy_for"] = decoy_for[g]
        arm_rows[g] = row
        d = diff(tc, tt)
        if not d:
            continue
        expressed.append(g)
        point[g] = d
        ks = sorted(d, key=lambda k: (-abs(d[k]), k))   # ties by id, so a --top-edges cut does not follow the hash seed
        if args.top_edges and args.top_edges > 0:      # 0 = no cap (default)
            ks = ks[:args.top_edges]
        cand[g] = ks
        cidx[g] = {k: i for i, k in enumerate(ks)}
        row["n_edges"] = len(ks)
    ncand = sum(len(v) for v in cand.values())
    print(f"TFs in both arms: {len(expressed)} (in one arm only, no edges: {len(arm_rows) - len(expressed)}); "
          f"candidate edges: {ncand:,} "
          f"({'UNCAPPED' if not args.top_edges else f'capped at {args.top_edges}/TF'}); bootstrapping...",
          flush=True)

    # bootstrap: resample each arm (WHOLE CELLS -- one index vector per arm, used for BOTH ids and values;
    # drawing ids and values independently would pair cell i's genes with cell j's expression), re-pseudobulk,
    # perturb every expressed TF, accumulate running sum/sum-of-squares so memory does not scale with B.
    # The shared-gene rule holds in every resample too (diff). An edge of the point estimate that does not
    # exist in a resample (its TF or target fell out of one resampled profile) contributes 0 to that
    # resample, as before; n_boot_in_both and mean_frac_edges_in_boot record how often that happened.
    bsum = {g: np.zeros(len(cand[g])) for g in expressed}
    bsq = {g: np.zeros(len(cand[g])) for g in expressed}
    boot_both = {g: 0 for g in expressed}
    boot_hits = {g: 0 for g in expressed}
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
                    boot_hits[g] += 1
            boot_both[g] += bool(d)
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
        arm_rows[g]["n_boot_in_both"] = boot_both[g]
        arm_rows[g]["mean_frac_edges_in_boot"] = (round(boot_hits[g] / (B * len(cand[g])), 4)
                                                  if B and cand[g] else None)
        for i, k in enumerate(cand[g]):
            d = point[g][k]
            sd = float(sds[i])
            z = d / sd if sd > 0 else 0.0
            edges.append((g, sym.get(g, ""), k, sym.get(k, ""), round(d, 6), round(sd, 6), round(z, 2))
                         + ((decoy_for[g], round(match[g]["match_dist"], 6)) if decoy_for else ()))
    edges.sort(key=lambda e: (-abs(e[6]), e[0], e[2]))   # ties by (tf, target): set order follows the hash seed
    n_z2 = sum(abs(e[6]) >= 2 for e in edges)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        f.write("tf\ttf_symbol\ttarget\ttarget_symbol\tdelta\tboot_sd\tz"
                + ("\tdecoy_for\tmatch_dist" if decoy_for else "") + "\n")
        for e in edges:
            f.write("\t".join(str(x) for x in e) + "\n")
    print(f"wrote {len(edges)} edges ({n_z2} with |z|>=2) from {len(expressed)} TFs -> {args.out}")

    # Sidecar tables (registration 8.1): one row per TF, and one row for the network.
    paths = sidecar_paths(args.out)
    name = Path(args.out).name
    name = name[:-len("_pooled.tsv")] if name.endswith("_pooled.tsv") else Path(args.out).stem
    rows = list(arm_rows.values())
    summary = network_summary(rows, profile_overlap(pbc, pbt), n_tf_listed, len(edges),
                              network=name, cell_type=args.cell_type, n_control_cells=nC, n_trained_cells=nT,
                              bootstrap=args.bootstrap, seed=args.seed)
    write_arm_sidecars(args.out, rows, summary, extra_columns=("decoy_for",) if decoy_for else ())
    print(summary_line(summary), flush=True)
    print(f"wrote {paths['arms']} and {paths['summary']}")

    if decoy_for:
        # Every match is listed, including a decoy that is in one arm only and therefore has no edges: the
        # edge table cannot show those. Audit finding 9.3: the match distance was never written out.
        with open(paths["match"], "w") as f:
            f.write("decoy\tdecoy_symbol\tdecoy_for\tdecoy_for_symbol\ttf_expr\tdecoy_expr\tmatch_dist\t"
                    "tf_in_control\ttf_in_trained\tdecoy_in_control\tdecoy_in_trained\tn_edges\n")
            for g in tfs:
                t, m = decoy_for[g], match[g]
                f.write("\t".join(_cell(x) for x in (
                    g, sym.get(g, ""), t, sym.get(t, ""), round(m["tf_expr"], 6), round(m["decoy_expr"], 6),
                    round(m["match_dist"], 6), tok[t] in ctok, tok[t] in ttok, tok[g] in ctok, tok[g] in ttok,
                    arm_rows.get(g, {}).get("n_edges", 0))) + "\n")
        dist = np.array([match[g]["match_dist"] for g in tfs])
        print(f"decoy match distance |log1p(decoy) - log1p(TF)|: median {np.median(dist):.4g}, "
              f"max {dist.max():.4g} over {len(dist)} matches; {len(tf_list)} TFs listed -> {paths['match']}")

    # Provenance (audit finding 9.6: the network tables had none). md5_inputs hashes the weights file.
    sys.path.insert(0, str(_ROOT / "lib"))
    from provenance import write_provenance
    weights = Path(args.model_dir) / "pytorch_model.bin"
    params = dict(vars(args))
    params.update({"space": "rat", "edge_rule": EDGE_RULE,
                   "registration": "docs/audit/remediation/STEP3_PREREG.md section 8.1",
                   "weights_file": str(weights), "n_edges": len(edges), "n_edges_abs_z_ge_2": n_z2,
                   "summary": summary})
    write_provenance(Path(args.out).parent, name=Path(paths["provenance"]).name,
                     inputs=[args.dataset, args.meta, args.tf_file, pc.DEFAULT_TOKEN_DICT,
                             _ROOT / "data" / "training" / "ortholog_mappings" / "rat_token_mapping.tsv",
                             Path(args.model_dir) / "config.json"],
                     md5_inputs=[weights, args.tf_file, args.meta], params=params)


if __name__ == "__main__":
    main()
