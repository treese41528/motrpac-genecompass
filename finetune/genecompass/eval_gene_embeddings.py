#!/usr/bin/env python3
"""eval_gene_embeddings.py — gene-embedding comparison for the retrain-v2 pre-registration (C4).

docs/RETRAIN_V2_PREREGISTRATION.md §3 C4, on the input word embeddings
(bert.embeddings.word_embeddings.weight), cosine similarity throughout:

  C4a  T4 symbol-matched genes: the 255 T4 (rat-only) tokens whose rat symbol, upper-cased, is the
       symbol of a human token in GeneCompass's vocabulary (Gene_id_name_dict_human_mouse.pickle; every
       match is to exactly one human token). For each, the rank (1 = nearest) of that human counterpart
       among all 23,113 human tokens. Statistic: median rank, top-10 and top-100 hit rates. Null: the same
       statistics over 1,000 random re-pairings, i.e. permutations of the 255 counterparts across the
       255 rat tokens (seed 20261008); p = (1 + #null at least as good) / 1,001.
  C4b  T3 ortholog retention: each of the 2,243 T3 tokens and the ortholog token its init was cloned
       from (rat_model_init.t3_ortholog_token_ids: human ortholog for T3a, mouse ortholog for T3b).
       Cosine to that ortholog, and the ortholog's rank among the tokens of its own species (human tokens
       for T3a; mouse tokens for T3b). Reported as drift from init, not as success.

  python finetune/genecompass/eval_gene_embeddings.py --out data/results/retrain_v2/c4 \
      --model init_v2=data/models/rat_genecompass_init_v2 --model april=<dir> --model v2=<dir>

Writes <out>/c4_summary.json, <out>/c4a_genes.tsv and <out>/c4b_genes.tsv (one column per model).
"""
import argparse
import csv
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from rat_model_init import t3_ortholog_token_ids          # noqa: E402

PK = ROOT / "vendor/GeneCompass/prior_knowledge"
OM = ROOT / "data/training/ortholog_mappings"


def token_sets():
    gc = pickle.load(open(PK / "human_mouse_tokens.pickle", "rb"))
    names = pickle.load(open(PK / "gene_list/Gene_id_name_dict_human_mouse.pickle", "rb"))
    rat = pickle.load(open(OM / "rat_human_mouse_tokens.pickle", "rb"))
    human = np.array(sorted(t for g, t in gc.items() if str(g).startswith("ENSG")))
    mouse = np.array(sorted(t for g, t in gc.items() if str(g).startswith("ENSMUSG")))
    sym2ensg = {}
    for g, s in names.items():
        if str(g).startswith("ENSG") and g in gc:
            sym2ensg.setdefault(str(s).upper(), []).append(g)
    rows = list(csv.DictReader(open(OM / "rat_token_mapping.tsv"), delimiter="\t"))
    t4 = [r for r in rows if r["tier"] == "T4_new_token" and r["rat_symbol"]
          and r["rat_symbol"].upper() in sym2ensg]
    if any(len(sym2ensg[r["rat_symbol"].upper()]) != 1 for r in t4):
        sys.exit("a T4 symbol matches more than one human token; C4a assumes one counterpart")
    c4a = pd.DataFrame({"rat_gene": [r["rat_gene"] for r in t4], "symbol": [r["rat_symbol"] for r in t4],
                        "biotype": [r["biotype"] for r in t4], "token": [rat[r["rat_gene"]] for r in t4],
                        "human_gene": [sym2ensg[r["rat_symbol"].upper()][0] for r in t4]})
    c4a["human_token"] = [gc[g] for g in c4a.human_gene]
    found, missing = t3_ortholog_token_ids(rat, OM / "rat_token_mapping.tsv", gc)
    if missing:
        sys.exit(f"{len(missing)} T3 tokens have no usable ortholog")
    tier = {rat[r["rat_gene"]]: r["tier"] for r in rows if r["tier"].startswith("T3")}
    sym = {rat[r["rat_gene"]]: r["rat_symbol"] for r in rows}
    c4b = pd.DataFrame({"token": sorted(found)})
    c4b["tier"] = [tier[t] for t in c4b.token]
    c4b["symbol"] = [sym[t] for t in c4b.token]
    c4b["ortholog_token"] = [found[t] for t in c4b.token]
    return human, mouse, c4a, c4b


def ranks_against(W, rows, pool):
    """Rank matrix: R[i, j] = rank (1 = nearest) of pool[j] by cosine to rows[i]."""
    S = W[rows] @ W[pool].T
    order = np.argsort(-S, axis=1, kind="stable")
    R = np.empty_like(order)
    R[np.arange(len(rows))[:, None], order] = np.arange(1, len(pool) + 1)
    return S, R


def c4a_stats(r):
    return {"median_rank": float(np.median(r)), "top10": float(np.mean(r <= 10)), "top100": float(np.mean(r <= 100))}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", action="append", required=True, help="name=model_dir (pytorch_model.bin inside)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-null", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20261008)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        sys.exit(f"{out} exists")
    human, mouse, c4a, c4b = token_sets()
    hpos = {t: j for j, t in enumerate(human)}
    mpos = {t: j for j, t in enumerate(mouse)}
    print(f"C4a: {len(c4a)} T4 symbol-matched tokens ({(c4a.biotype == 'protein_coding').sum()} protein-coding) "
          f"vs {len(human):,} human tokens | C4b: {len(c4b)} T3 tokens "
          f"({c4b.tier.value_counts().to_dict()}), {c4b.ortholog_token.nunique()} distinct ortholog tokens")
    summary = {"n_c4a": len(c4a), "n_c4b": len(c4b), "n_null": a.n_null, "seed": a.seed, "models": {}}
    rng = np.random.default_rng(a.seed)
    perms = [rng.permutation(len(c4a)) for _ in range(a.n_null)]
    cidx = np.array([hpos[t] for t in c4a.human_token])
    for spec in a.model:
        name, mdir = spec.split("=", 1)
        state = torch.load(Path(mdir) / "pytorch_model.bin", map_location="cpu", weights_only=False)
        W = state["bert.embeddings.word_embeddings.weight"].float().numpy()
        del state
        W = W / np.linalg.norm(W, axis=1, keepdims=True)
        # C4a
        S, R = ranks_against(W, c4a.token.to_numpy(), human)
        obs = R[np.arange(len(c4a)), cidx]
        c4a[f"rank_{name}"] = obs
        c4a[f"cos_{name}"] = S[np.arange(len(c4a)), cidx].round(4)
        st = c4a_stats(obs)
        null = [c4a_stats(R[np.arange(len(c4a)), cidx[p]]) for p in perms]
        res = {"c4a": st, "c4a_null": {}}
        for k in st:
            v = np.array([n[k] for n in null])
            better = (v <= st[k]) if k == "median_rank" else (v >= st[k])
            res["c4a_null"][k] = {"mean": float(v.mean()), "q025": float(np.quantile(v, 0.025)),
                                  "q975": float(np.quantile(v, 0.975)), "p": float((1 + better.sum()) / (1 + len(v)))}
        # C4b
        res["c4b"] = {}
        for tier, pool, pos in (("T3a_human_multi", human, hpos), ("T3b_mouse_multi", mouse, mpos)):
            sub = c4b[c4b.tier == tier]
            S, R = ranks_against(W, sub.token.to_numpy(), pool)
            j = np.array([pos[t] for t in sub.ortholog_token])
            cos, rk = S[np.arange(len(sub)), j], R[np.arange(len(sub)), j]
            c4b.loc[sub.index, f"cos_{name}"] = cos.round(4)
            c4b.loc[sub.index, f"rank_{name}"] = rk
            res["c4b"][tier] = {"n": len(sub), "median_cos": float(np.median(cos)), "mean_cos": float(cos.mean()),
                                "median_rank": float(np.median(rk)), "top1": float(np.mean(rk == 1)),
                                "top10": float(np.mean(rk <= 10))}
        summary["models"][name] = {"model": mdir, **res}
        n = res["c4a_null"]
        print(f"\n[{name}] C4a median rank {st['median_rank']:.0f} (null {n['median_rank']['mean']:.0f}, "
              f"p {n['median_rank']['p']:.3g}); top-10 {st['top10']:.3f} (null {n['top10']['mean']:.3f}, "
              f"p {n['top10']['p']:.3g}); top-100 {st['top100']:.3f} (null {n['top100']['mean']:.3f}, "
              f"p {n['top100']['p']:.3g})")
        for tier, s in res["c4b"].items():
            print(f"[{name}] C4b {tier}: median cos {s['median_cos']:.3f}, median ortholog rank "
                  f"{s['median_rank']:.0f}, top-1 {s['top1']:.3f}, top-10 {s['top10']:.3f}")
    out.mkdir(parents=True)
    (out / "c4_summary.json").write_text(json.dumps(summary, indent=2))
    c4a.to_csv(out / "c4a_genes.tsv", sep="\t", index=False)
    c4b.to_csv(out / "c4b_genes.tsv", sep="\t", index=False)
    sys.path.insert(0, str(ROOT / "lib"))
    from provenance import write_provenance
    write_provenance(out, inputs=[spec.split("=", 1)[1] for spec in a.model],
                     params={"n_null": a.n_null, "seed": a.seed})
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
