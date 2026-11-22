#!/usr/bin/env python
# coding: utf-8
r"""build_grn_human.py -- Aim 3b step 1: the SAME dose-pooled bootstrap differential network as
build_grn_pooled.py, but run in HUMAN embedding space on the transferred pseudo-cells.

Self-consistency input: each rat TF's in-silico perturbation is re-run on the human-space
re-expression of the SAME rat pseudo-cells (genecompass_input_human/<tissue>/dataset, species=0).
Comparing the rat-space and human-space target lists per TF (compare_grn_conservation.py) measures
how stable the model's read-out is under that re-expression. No human measurement enters, so the
comparison says nothing about cross-species biology. Only THREE things change vs build_grn_pooled.py:
  1. dataset  = the transferred human-space dataset (human ENSG tokens, species token 0);
  2. TF token = the rat TF's HUMAN ORTHOLOG token (rat ENSRNOG -> human ENSG via the Stage-3
     ortholog map -> GeneCompass token); rat-specific TFs with no human ortholog are skipped;
  3. species  = 0 (human) in the embed call.
Targets are resolved to human ENSG so the output aligns with the rat network after ortholog
projection (compare_grn_conservation.py).

The edge rule is build_grn_pooled.diff: an edge exists only where the TF and the target are in both
arms' profiles (docs/audit/remediation/STEP3_PREREG.md 8.1). The same sidecar tables and provenance
record as the rat-space builder are written: <name>_pooled_arms.tsv, <name>_pooled_arms_summary.tsv,
provenance_<name>_pooled.json (with the md5 of the weights file).

Usage (species-aware; mirrors build_grn_pooled args):
  python finetune/genecompass/build_grn_human.py --model-dir <...> \
    --dataset data/deconvolution/genecompass_input_v2_human/skmvl/dataset \
    --meta data/deconvolution/augur_input/skmvl/meta.tsv --cell-type "Skeletal muscle" \
    --out data/deconvolution/grn_human/skmvl_skeletal_muscle_pooled.tsv --bootstrap 30 --device cuda
"""
import argparse
import csv
import os
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(os.environ.get("PIPELINE_ROOT", ".")).resolve()
sys.path.insert(0, str(_ROOT / "finetune" / "genecompass"))
import perturb_cells as pc              # noqa: E402
from build_grn import load_arms         # noqa: E402
from build_grn_pooled import (EDGE_RULE, arm_counts, diff, network_summary, profile_overlap,  # noqa: E402
                              pseudobulk, sidecar_paths, summary_line, write_arm_sidecars)

HUMAN_SPECIES = 0


@torch.no_grad()
def perturb_profile_human(model, pid, pvl, token, device, tok2ensg):
    """Delete `token` from one HUMAN-space profile (species=0); return {target_ensg: shift}."""
    if int((pid == token).sum()) == 0:
        return {}
    ni, nv, rem = pc.delete_token(pid, pvl, token)
    _, g0 = pc.embed(model, pid[None, :], pvl[None, :], device, species=HUMAN_SPECIES)
    _, g1 = pc.embed(model, ni[None, :], nv[None, :], device, species=HUMAN_SPECIES)
    p = rem[0]
    keep = torch.ones(g0.shape[1], dtype=torch.bool)
    keep[p] = False
    orig = g0[0][keep]
    pert = g1[0][: orig.shape[0]]
    surv = pid[keep].tolist()
    sh = pc._cos_shift(orig, pert)
    return {tok2ensg[t]: float(s) for t, s in zip(surv, sh) if t != 0 and t in tok2ensg}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--dataset", required=True, help="genecompass_input_human/<tissue>/dataset")
    ap.add_argument("--meta", required=True, help="rat augur_input/<tissue>/meta.tsv (same sample/cell_type keys)")
    ap.add_argument("--cell-type", required=True)
    ap.add_argument("--tf-file", default=str(_ROOT / "deconvolution" / "reference" / "rat_tf_ensrnog.tsv"))
    ap.add_argument("--ortholog-map", default=str(_ROOT / "data/training/ortholog_mappings/rat_to_human_mapping.pickle"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--bootstrap", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--top-edges", type=int, default=0,
                    help="candidate edges per TF, by |point delta|. 0 = NO CAP (default); must match "
                         "build_grn_pooled.py or the rat/human target lists are not comparable.")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--permute-seed", type=int, default=-1,
                    help="ABANDONED PERMUTATION NULL (deconvolution/GRN_NULL_WHY_ABANDONED.md), identical logic "
                         "and seed semantics to build_grn_pooled.py; kept only to reproduce its 2026-07 tables. "
                         "Because the human pseudo-cells are the SAME cells in the same order "
                         "(transfer_to_human.py re-expresses them), the same seed yields the SAME arm split "
                         "in both spaces.")
    ap.add_argument("--max-tfs", type=int, default=None)
    args = ap.parse_args()

    # MERGED token dict (ENSG/ENSMUSG/ENSRNOG -> GC token id). NOT pc.load_token_dict(), which is
    # rat_tokens.pickle (ENSRNOG-only) and has no human ENSG keys -> would find 0 human orthologs.
    # ENSG->id here == the vendor human_mouse_tokens the transfer tokenized with (verified identical).
    _raw = pickle.load(open(_ROOT / "data/training/ortholog_mappings/rat_human_mouse_tokens.pickle", "rb"))
    tok = {(str(k).split(".")[0].upper() if isinstance(k, str) else k): v for k, v in _raw.items()}
    # token -> human ENSG (prefer ENSG keys so shared T1/T2 tokens resolve to the human id, not ENSRNOG)
    tok2ensg = {v: k for k, v in tok.items() if isinstance(k, str) and k.startswith("ENSG") and not k.startswith("ENSMUSG")}
    r2h = {str(k).split(".")[0].upper(): str(v).split(".")[0].upper()
           for k, v in pickle.load(open(args.ortholog_map, "rb")).items()}
    # rat symbols (for reporting the TF) + human ENSG->symbol for targets
    sym = {}
    for r in csv.DictReader(open(_ROOT / "data/training/ortholog_mappings/rat_token_mapping.tsv"), delimiter="\t"):
        sym[r["rat_gene"]] = r.get("rat_symbol", "")

    # each rat TF -> its human ortholog token (skip rat-specific TFs / orthologs out of vocab)
    tfs_raw = [l.split("\t")[0].strip() for i, l in enumerate(open(args.tf_file)) if i > 0 and l.strip()]
    tf_htok = {}
    for g in tfs_raw:
        h = r2h.get(g)
        if h and h in tok:
            tf_htok[g] = tok[h]
    tfs = list(tf_htok)
    if args.max_tfs:
        tfs = tfs[:args.max_tfs]
    print(f"[{args.cell_type}] human-space; {len(tfs)}/{len(tfs_raw)} rat TFs have an in-vocab human ortholog", flush=True)

    (idc, vlc), (idt, vlt) = load_arms(args.dataset, args.meta, args.cell_type)
    if idc is None or idt is None:
        sys.exit(f"ERROR: empty arm(s) for cell type {args.cell_type!r} in {args.dataset}")
    nC, nT = idc.shape[0], idt.shape[0]

    if args.permute_seed >= 0:
        import sys as _sys
        _sys.path.insert(0, str(_ROOT / "finetune" / "genecompass"))
        from build_grn_pooled import load_arm_sex
        sexc, sext = load_arm_sex(args.dataset, args.meta, args.cell_type)
        prng = np.random.default_rng(args.permute_seed)
        all_id = torch.cat([idc, idt], dim=0); all_vl = torch.cat([vlc, vlt], dim=0)
        all_sx = np.concatenate([sexc, sext])
        n_ctrl_by_sex = {s: int((sexc == s).sum()) for s in np.unique(all_sx)}
        ci, ti = [], []
        for s in np.unique(all_sx):
            pool = np.where(all_sx == s)[0]; prng.shuffle(pool)
            k = n_ctrl_by_sex.get(s, 0); ci.extend(pool[:k]); ti.extend(pool[k:])
        ci, ti = np.array(ci), np.array(ti)
        idc, vlc = all_id[ci], all_vl[ci]
        idt, vlt = all_id[ti], all_vl[ti]
        print(f"  *** SEX-STRATIFIED PERMUTATION NULL (seed={args.permute_seed}) *** {n_ctrl_by_sex}", flush=True)

    print(f"  control={nC} trained={nT}; bootstrap={args.bootstrap}", flush=True)
    model = pc.load_model(args.model_dir, args.device)
    rng = np.random.default_rng(args.seed)

    pbc, pbt = pseudobulk(idc, vlc), pseudobulk(idt, vlt)
    ctok, ttok = set(pbc[0].tolist()) - {0}, set(pbt[0].tolist()) - {0}
    # Registration 8.1, as in build_grn_pooled.py: edges only for a TF in both arms' profiles and only to
    # targets in both; every TF found in at least one arm gets a sidecar row with the counts.
    point, cand, cidx, expressed = {}, {}, {}, []
    arm_rows = {}
    for g in tfs:
        in_c, in_t = tf_htok[g] in ctok, tf_htok[g] in ttok
        if not (in_c or in_t):
            continue                                   # ortholog in neither profile: nothing to delete
        tc = perturb_profile_human(model, *pbc, tf_htok[g], args.device, tok2ensg)
        tt = perturb_profile_human(model, *pbt, tf_htok[g], args.device, tok2ensg)
        row = arm_counts(tc, tt, in_c, in_t)
        row.update({"tf": g, "tf_symbol": sym.get(g, ""), "n_edges": 0, "n_boot_in_both": None,
                    "mean_frac_edges_in_boot": None})
        arm_rows[g] = row
        d = diff(tc, tt)
        if not d:
            continue
        expressed.append(g)
        point[g] = d
        ks = sorted(d, key=lambda k: (-abs(d[k]), k))   # ties by id, so a --top-edges cut does not follow the hash seed
        if args.top_edges and args.top_edges > 0:      # 0 = no cap (default); see build_grn_pooled.py
            ks = ks[:args.top_edges]
        cand[g] = ks
        cidx[g] = {k: i for i, k in enumerate(ks)}
        row["n_edges"] = len(ks)
    ncand = sum(len(v) for v in cand.values())
    print(f"TFs whose human ortholog is in both arms: {len(expressed)} (in one arm only, no edges: "
          f"{len(arm_rows) - len(expressed)}); candidate edges: {ncand:,} "
          f"({'UNCAPPED' if not args.top_edges else f'capped at {args.top_edges}/TF'}); bootstrapping...",
          flush=True)

    # resample WHOLE CELLS: one index vector per arm, used for both ids and values.
    bsum = {g: np.zeros(len(cand[g])) for g in expressed}
    bsq = {g: np.zeros(len(cand[g])) for g in expressed}
    boot_both = {g: 0 for g in expressed}
    boot_hits = {g: 0 for g in expressed}
    for b in range(args.bootstrap):
        bi = rng.integers(0, nC, nC)
        bt = rng.integers(0, nT, nT)
        pbc_b = pseudobulk(idc[bi], vlc[bi])
        pbt_b = pseudobulk(idt[bt], vlt[bt])
        for g in expressed:
            d = diff(perturb_profile_human(model, *pbc_b, tf_htok[g], args.device, tok2ensg),
                     perturb_profile_human(model, *pbt_b, tf_htok[g], args.device, tok2ensg))
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
        sds = np.sqrt(np.maximum(bsq[g] / B - mean * mean, 0.0))
        arm_rows[g]["n_boot_in_both"] = boot_both[g]
        arm_rows[g]["mean_frac_edges_in_boot"] = (round(boot_hits[g] / (B * len(cand[g])), 4)
                                                  if B and cand[g] else None)
        for i, k in enumerate(cand[g]):
            d = point[g][k]
            sd = float(sds[i])
            z = d / sd if sd > 0 else 0.0
            edges.append((g, sym.get(g, ""), r2h.get(g, ""), k, round(d, 6), round(sd, 6), round(z, 2)))
    edges.sort(key=lambda e: (-abs(e[6]), e[0], e[3]))   # ties by (tf, target): set order follows the hash seed
    n_z2 = sum(abs(e[6]) >= 2 for e in edges)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        f.write("tf\ttf_symbol\ttf_human_ensg\ttarget_human_ensg\tdelta\tboot_sd\tz\n")
        for e in edges:
            f.write("\t".join(str(x) for x in e) + "\n")
    print(f"wrote {len(edges)} human-space edges ({n_z2} with |z|>=2) from {len(expressed)} TFs -> {args.out}")

    # Sidecar tables (registration 8.1) and provenance (audit finding 9.6), as in build_grn_pooled.py.
    paths = sidecar_paths(args.out)
    name = Path(args.out).name
    name = name[:-len("_pooled.tsv")] if name.endswith("_pooled.tsv") else Path(args.out).stem
    rows = list(arm_rows.values())
    summary = network_summary(rows, profile_overlap(pbc, pbt), len(tfs), len(edges),
                              network=name, cell_type=args.cell_type, n_control_cells=nC, n_trained_cells=nT,
                              bootstrap=args.bootstrap, seed=args.seed)
    write_arm_sidecars(args.out, rows, summary)
    print(summary_line(summary), flush=True)
    sys.path.insert(0, str(_ROOT / "lib"))
    from provenance import write_provenance
    weights = Path(args.model_dir) / "pytorch_model.bin"
    params = dict(vars(args))
    params.update({"space": "human", "edge_rule": EDGE_RULE,
                   "registration": "docs/audit/remediation/STEP3_PREREG.md section 8.1",
                   "weights_file": str(weights), "n_edges": len(edges), "n_edges_abs_z_ge_2": n_z2,
                   "summary": summary})
    write_provenance(Path(args.out).parent, name=Path(paths["provenance"]).name,
                     inputs=[args.dataset, args.meta, args.tf_file, args.ortholog_map,
                             Path(args.model_dir) / "config.json"],
                     md5_inputs=[weights, args.tf_file, args.meta], params=params)


if __name__ == "__main__":
    main()
