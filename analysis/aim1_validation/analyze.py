#!/usr/bin/env python3
"""
analyze.py -- Aim-1 validation analysis + report (CPU).

V1 held-out clustering: silhouette(cell_type) vs silhouette(study) on the rat CLS embeddings,
   ARI(kmeans, cell_type), study-mixing -> do cells cluster by BIOLOGY not BATCH? (+ UMAP figure)
V2 homolog similarity:
   - input embeddings: T1 share tokens (=1.0 by design); T3a/T3b were WARM-STARTED from the ortholog,
     so we report cosine-to-ortholog vs random null AND vs the INIT checkpoint (preservation, not naive).
   - contextual embeddings (init-free, the clean test): cosine(rat-gene-in-rat-context, ortholog-in-its-
     species-context) vs null; ROC-AUC homolog>null. (+ distribution figure)
V3 T4 token quality: T4 genes are RANDOM-init, so cosine of T4 family genes (OR, CYP) to their human/mouse
   family peers vs random null is a clean learned-similarity test; rank-AUC + projection figure.

Writes deconvolution/.. no -> writes reports/aim1_validation/AIM1_VALIDATION_REPORT.md + figures.
"""
import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, adjusted_rand_score, roc_auc_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

_ROOT = Path(os.environ.get("PIPELINE_ROOT", ".")).resolve()
OMAP = _ROOT / "data/training/ortholog_mappings"


def base(x):
    return str(x).strip().split(".")[0].upper()


def load_word_emb(model_dir):
    import torch
    sd = torch.load(Path(model_dir) / "pytorch_model.bin", map_location="cpu", weights_only=False)
    w = sd["bert.embeddings.word_embeddings.weight"].float().numpy()
    del sd
    return w


def cos_rows(W, a, b):
    """row-wise cosine between W[a] and W[b] (a,b arrays of token ids)."""
    A = normalize(W[a]); B = normalize(W[b])
    return (A * B).sum(1)


# ============================== V1 ==============================
def v1(inp, fig_dir, k=15):
    """Local-neighbourhood metrics (NOT global silhouette, which is confounded because study~tissue):
    kNN cell-type purity (biology cohesion) vs the chance baseline, and -- the real batch test --
    cross-study mixing WITHIN each cell type (do same-cell-type neighbours come from other studies?)."""
    from sklearn.neighbors import NearestNeighbors
    cls = np.load(inp / "out_rat" / "cls_embeddings.npy")
    lab = pd.read_csv(inp / "labels.tsv", sep="\t")
    assert len(cls) == len(lab), f"cls {len(cls)} != labels {len(lab)}"
    ct = lab["cell_type"].to_numpy(); st = lab["study_id"].to_numpy()
    N = len(cls)

    nn = NearestNeighbors(n_neighbors=k + 1).fit(cls)
    _, idx = nn.kneighbors(cls); idx = idx[:, 1:]                     # drop self
    same_ct = (ct[idx] == ct[:, None])                                # [N,k]
    same_st = (st[idx] == st[:, None])
    p_ct = float(same_ct.mean())                                     # biology: kNN share cell type
    p_st = float(same_st.mean())                                     # batch: kNN share study
    # chance baselines (sum of squared label frequencies = P(two random cells share label))
    chance_ct = float(((lab["cell_type"].value_counts() / N) ** 2).sum())
    chance_st = float(((lab["study_id"].value_counts() / N) ** 2).sum())

    # disentangled batch test: among SAME-cell-type neighbours, fraction from a DIFFERENT study
    multi_ct = set(lab.groupby("cell_type")["study_id"].nunique().loc[lambda s: s >= 2].index)
    in_multi = np.array([c in multi_ct for c in ct])
    cross = []
    for i in np.where(in_multi)[0]:
        m = same_ct[i]
        if m.sum() > 0:
            cross.append(float((st[idx[i]][m] != st[i]).mean()))
    cross_study_mix = float(np.mean(cross)) if cross else float("nan")
    # KMeans ARI for reference
    nk = lab["cell_type"].nunique()
    km = KMeans(n_clusters=nk, n_init=4, random_state=0).fit_predict(cls)
    ari_ct = float(adjusted_rand_score(ct, km))

    try:
        import umap
        emb = umap.UMAP(n_neighbors=30, min_dist=0.3, random_state=0).fit_transform(cls)
        fig, ax = plt.subplots(1, 2, figsize=(16, 7))
        for a, key, title in [(ax[0], ct, "cell type (biology)"), (ax[1], st, "study (batch)")]:
            a.scatter(emb[:, 0], emb[:, 1], c=pd.Categorical(key).codes, cmap="tab20", s=2, alpha=0.5)
            a.set_title(f"UMAP coloured by {title}"); a.set_xticks([]); a.set_yticks([])
        fig.suptitle("V1: held-out rat cells — biology vs batch")
        fig.savefig(fig_dir / "v1_umap.png", dpi=110, bbox_inches="tight"); plt.close(fig)
        fig_ok = True
    except Exception as e:
        print(f"[v1] UMAP skipped: {e}", flush=True); fig_ok = False

    # PASS = local neighbourhoods are cell-type-enriched well above chance AND same-cell-type cells
    # mix substantially across studies (batch not dominant within biology)
    verdict = "PASS" if (p_ct > 3 * chance_ct and cross_study_mix > 0.30) else \
              ("PARTIAL" if p_ct > 1.5 * chance_ct else "WEAK")
    return dict(n_cells=N, n_cell_types=nk, n_studies=int(lab["study_id"].nunique()),
                n_celltypes_multistudy=int(in_multi.sum() and len(multi_ct)), k=k,
                knn_purity_celltype=p_ct, chance_celltype=chance_ct,
                knn_purity_study=p_st, chance_study=chance_st,
                cross_study_mix_within_celltype=cross_study_mix,
                ari_celltype=ari_ct, fig=fig_ok, verdict=verdict)


# ============================== V2 ==============================
def v2(inp, final_dir, init_dir, fig_dir):
    """Homolog similarity with a PERMUTED-ortholog null (rat gene vs a random OTHER ortholog of the
    same species), which needs only the extracted T3a/T3b tokens -- no separate null extraction.
    T1/T2 share the ortholog token by design (input cosine 1.0); T3a/T3b got NEW warm-started tokens,
    so input cosine is reported as init->final PRESERVATION and the clean test is the CONTEXTUAL AUC."""
    pairs = pd.read_csv(inp / "pairs.tsv", sep="\t")
    Wf = load_word_emb(final_dir)
    rng = np.random.default_rng(0)
    res = {"input": {}, "contextual": {}}
    res["input"]["T1"] = dict(note="T1/T2 rat genes reuse their ortholog's token id -> input embedding "
                              "identical by design (cosine=1.0); the cross-species sharing mechanism, not learned.")

    def perm_auc(ch, cn_emb_a, cn_emb_b):
        """ch = homolog cosines; build null by deranged permutation of the b side."""
        n = len(cn_emb_b)
        perm = rng.permutation(n)
        bad = perm == np.arange(n)
        if bad.any():
            perm[bad] = (perm[bad] + 1) % n
        cn = (normalize(cn_emb_a) * normalize(cn_emb_b[perm])).sum(1)
        y = np.r_[np.ones(len(ch)), np.zeros(len(cn))]; s = np.r_[ch, cn]
        return cn, float(roc_auc_score(y, s))

    # ---- input embeddings: T3a/T3b homolog cosine + permuted null + init preservation ----
    Wi = load_word_emb(init_dir) if (init_dir and Path(init_dir, "pytorch_model.bin").exists()) else None
    for tier, side in [("T3a", "human_token"), ("T3b", "mouse_token")]:
        sub = pairs[(pairs.tier == tier) & pairs[side].notna()]
        if len(sub) < 10:
            continue
        a = sub.rat_token.to_numpy().astype(int); b = sub[side].to_numpy().astype(int)
        ch = cos_rows(Wf, a, b)
        _, auc = perm_auc(ch, Wf[a], Wf[b])
        d = dict(n=len(sub), median_homolog=float(np.median(ch)), auc=float(auc))
        if Wi is not None:
            d["cos_init_median"] = float(np.median(cos_rows(Wi, a, b)))
            d["cos_final_median"] = float(np.median(ch))
        res["input"][tier] = d
    if Wi is not None:
        del Wi

    # ---- contextual embeddings (init-free clean test) ----
    def load_ctx(name):
        p = inp / name / "contextual.npz"
        if not p.exists():
            return None
        z = np.load(p)
        return ({int(t): z["mean"][i] for i, t in enumerate(z["tokens"])},
                {int(t): int(z["count"][i]) for i, t in enumerate(z["tokens"])})
    rat = load_ctx("out_rat"); hum = load_ctx("out_human"); mou = load_ctx("out_mouse")

    def ctx_arrays(df, side, ref, min_cells=20):
        rm, rc = rat; sm, sc = ref
        A, B = [], []
        for _, r in df.iterrows():
            rt = int(r.rat_token); ot = r[side]
            if pd.isna(ot):
                continue
            ot = int(ot)
            if rt in rm and ot in sm and rc.get(rt, 0) >= min_cells and sc.get(ot, 0) >= min_cells:
                av, bv = rm[rt], sm[ot]
                if np.isfinite(av).all() and np.isfinite(bv).all():
                    A.append(av); B.append(bv)
        return np.array(A), np.array(B)

    plotted = []
    if rat is not None:
        # T1 shares the token id, but rat-cell vs human-cell CONTEXTUAL embeddings still differ
        # (species token + context) -> a real cross-species consistency test for the shared-token genes.
        for tier, side, ref in [("T1", "human_token", hum), ("T3a", "human_token", hum), ("T3b", "mouse_token", mou)]:
            if ref is None:
                continue
            A, B = ctx_arrays(pairs[pairs.tier == tier], side, ref)
            if len(A) >= 10:
                ch = (normalize(A) * normalize(B)).sum(1)
                cn, auc = perm_auc(ch, A, B)
                res["contextual"][tier] = dict(n=len(A), median_homolog=float(np.median(ch)),
                                               median_null=float(np.median(cn)), auc=auc)
                plotted.append((tier, ch, cn, auc, len(A)))

    if plotted:
        fig, axes = plt.subplots(1, len(plotted), figsize=(6 * len(plotted), 5), squeeze=False)
        for ax, (t, ch, cn, auc, n) in zip(axes[0], plotted):
            ax.hist(cn, bins=30, alpha=0.5, label="permuted null", color="gray", density=True)
            ax.hist(ch, bins=30, alpha=0.6, label="true ortholog", color="C0", density=True)
            ax.set_title(f"{t} contextual cosine (n={n})\nAUC={auc:.3f}")
            ax.set_xlabel("cosine similarity"); ax.legend()
        fig.suptitle("V2: contextual gene embeddings — rat genes near their true ortholog vs a permuted ortholog")
        fig.savefig(fig_dir / "v2_contextual.png", dpi=110, bbox_inches="tight"); plt.close(fig)
    del Wf
    return res


# ============================== V3 ==============================
import re

def _root(sym):
    """family root = leading alphabetic run of a gene symbol (>=3 chars), upper-cased.
    'Cyp2j3'->'CYP', 'Olr1720'->'OLR', 'Slc7a11'->'SLC'. None if no >=3-char alpha root."""
    m = re.match(r"^([A-Za-z]{3,})", str(sym))
    return m.group(1).upper() if m else None


def v3(final_dir, fig_dir):
    """T4 tokens were RANDOM-initialised, so cosine to same-family human/mouse peers vs random is a
    clean learned-similarity test. The named-family set is thin (only ~4 CYP, 1 OLR by symbol; most T4
    are LOC/unnamed), so we (a) report CYP explicitly (the promise's example) and (b) POOL per-gene
    rank-AUC across ALL T4 genes whose symbol root matches >=3 human/mouse peers, for power."""
    tm = pd.read_csv(OMAP / "rat_token_mapping.tsv", sep="\t")
    t4 = tm[tm.tier == "T4_new_token"].copy()
    t4["sym"] = t4["rat_symbol"].astype(str)
    t4["root"] = t4["sym"].apply(_root)
    with open(OMAP / "rat_human_mouse_tokens.pickle", "rb") as f:
        vocab = {base(k): int(v) for k, v in pickle.load(f).items()}
    with open(_ROOT / "vendor/GeneCompass/prior_knowledge/gene_list/Gene_id_name_dict_human_mouse.pickle", "rb") as f:
        id2sym = pickle.load(f)
    root2tok = {}   # human/mouse symbol root -> [tokens]
    for gid, sym in id2sym.items():
        t = vocab.get(base(gid)); r = _root(sym)
        if t is not None and r is not None:
            root2tok.setdefault(r, []).append(t)

    Wf = load_word_emb(final_dir); Wn = normalize(Wf)
    rng = np.random.default_rng(0)
    null_tok = rng.choice(Wf.shape[0], size=2000, replace=False)
    nullm = Wn[null_tok]

    def gene_auc(rt, peer_tokens):
        peer = Wn[[p for p in peer_tokens if p != rt]]
        if len(peer) < 3:
            return None
        v = Wn[rt]; ch = peer @ v; cn = nullm @ v
        y = np.r_[np.ones(len(ch)), np.zeros(len(cn))]; s = np.r_[ch, cn]
        return float(roc_auc_score(y, s)), float(ch.mean()), float(cn.mean())

    pooled = []     # (root, rat_token, auc)
    cyp = []
    for _, r in t4.iterrows():
        root = r["root"]; rt = int(r["token_id"])
        if root is None or root not in root2tok or not (0 <= rt < Wf.shape[0]):
            continue
        peers = sorted(set(root2tok[root]))
        if len(peers) < 3 or len(peers) > 600:   # skip ultra-generic roots (e.g. LOC, ZNF) as non-specific
            continue
        g = gene_auc(rt, peers)
        if g is None:
            continue
        pooled.append((root, rt, g[0], len(peers)))
        if root == "CYP":
            cyp.append((r["sym"], g[0], g[1], g[2], len(peers)))

    out = {}
    if pooled:
        aucs = np.array([p[2] for p in pooled])
        out["pooled"] = dict(n_t4_genes=len(pooled), n_distinct_families=len({p[0] for p in pooled}),
                             mean_auc=float(aucs.mean()), median_auc=float(np.median(aucs)),
                             frac_auc_gt_0p65=float((aucs > 0.65).mean()),
                             example_roots=sorted({p[0] for p in pooled})[:25],
                             verdict="PASS" if aucs.mean() > 0.65 else ("PARTIAL" if aucs.mean() > 0.55 else "WEAK"))
    if cyp:
        ca = np.array([c[1] for c in cyp])
        out["CYP"] = dict(n_t4=len(cyp), genes=[c[0] for c in cyp], mean_auc=float(ca.mean()),
                          mean_cos_family=float(np.mean([c[2] for c in cyp])),
                          mean_cos_null=float(np.mean([c[3] for c in cyp])), n_peers=int(cyp[0][4]),
                          verdict="PASS" if ca.mean() > 0.65 else "WEAK")
        # CYP projection figure
        try:
            from sklearn.decomposition import PCA
            cyp_t4 = [int(t4[t4.sym == c[0]].token_id.iloc[0]) for c in cyp]
            cyp_peers = sorted(set(root2tok["CYP"]))
            pts = np.vstack([Wf[cyp_t4], Wf[cyp_peers], Wf[null_tok[:300]]])
            p2 = PCA(2, random_state=0).fit_transform(normalize(pts))
            nA, nB = len(cyp_t4), len(cyp_peers)
            fig, ax = plt.subplots(figsize=(7, 6))
            ax.scatter(p2[nA+nB:, 0], p2[nA+nB:, 1], s=6, c="lightgray", label="random genes", alpha=0.5)
            ax.scatter(p2[nA:nA+nB, 0], p2[nA:nA+nB, 1], s=20, c="C0", label=f"human/mouse CYP (n={nB})", alpha=0.6)
            ax.scatter(p2[:nA, 0], p2[:nA, 1], s=120, c="crimson", marker="*", label=f"rat T4 Cyp (n={nA})")
            ax.set_title(f"V3: rat T4 Cyp450 new tokens vs CYP family (mean AUC {ca.mean():.3f})")
            ax.legend(); ax.set_xticks([]); ax.set_yticks([])
            fig.savefig(fig_dir / "v3_cyp.png", dpi=110, bbox_inches="tight"); plt.close(fig)
        except Exception as e:
            print(f"[v3] CYP projection skipped: {e}", flush=True)
    del Wf, Wn
    return out


def md(v1r, v2r, v3r, inp):
    L = []
    L.append("# Aim-1 GeneCompass representation-quality validation\n")
    L.append("The three checks promised in `reports/pipeline_report.md §3.6`, run on the fine-tuned rat "
             "checkpoint `rat_phase2_mixed_species/checkpoint-147941`. Built/ran 2026-06-25.\n")

    L.append("\n## V1 — held-out cell-type clustering (biology vs batch)\n")
    L.append("Global silhouette is confounded here (each study ≈ one tissue, so 'study' and coarse biology "
             "co-vary), so we use LOCAL kNN metrics: cell-type purity vs its chance baseline (biology), and — "
             "the real batch test — cross-study mixing *within* each cell type.\n")
    L.append(f"- **{v1r['verdict']}** — {v1r['n_cells']} held-out rat cells, {v1r['n_cell_types']} cell types, "
             f"{v1r['n_studies']} studies ({v1r['n_celltypes_multistudy']} cell types span ≥2 studies); k={v1r['k']}.\n")
    L.append(f"- **kNN cell-type purity = {v1r['knn_purity_celltype']:.3f}** vs chance {v1r['chance_celltype']:.3f} "
             f"(**{v1r['knn_purity_celltype']/max(v1r['chance_celltype'],1e-9):.1f}×** enrichment ⇒ neighbourhoods are biology-coherent).\n")
    L.append(f"- **Cross-study mixing within cell type = {v1r['cross_study_mix_within_celltype']:.3f}** "
             f"(fraction of same-cell-type neighbours from a *different* study; higher ⇒ biology not batch drives local structure).\n")
    L.append(f"- kNN study purity = {v1r['knn_purity_study']:.3f} (chance {v1r['chance_study']:.3f}); ARI(KMeans, cell type) = {v1r['ari_celltype']:.3f}.\n")
    if v1r.get("fig"):
        L.append("- Figure: `figures/v1_umap.png` (UMAP coloured by cell type vs by study).\n")

    L.append("\n## V2 — homolog embedding similarity (rat vs human/mouse orthologs vs random)\n")
    L.append("Rat T1/T2 genes **reuse their ortholog's GeneCompass token id**, so their *input* embedding is "
             "identical by design (the cross-species sharing mechanism). T3a/T3b genes got NEW tokens "
             "**warm-started from the ortholog**, so input cosine is reported as init→final *preservation* and the "
             "clean test is the **contextual** embedding (from the forward pass, init-free). Null = each rat gene "
             "vs a *permuted* (random other) ortholog of the same species; AUC = true-ortholog separability.\n")
    L.append("\n**Input embeddings (warm-start aware):**\n")
    for t, d in v2r["input"].items():
        if t == "T1":
            L.append(f"- T1/T2: {d['note']}\n")
        else:
            line = f"- {t}: cosine-to-ortholog median {d.get('median_homolog', float('nan')):.3f}, AUC vs permuted null {d.get('auc', float('nan')):.3f} (n={d.get('n')})"
            if "cos_init_median" in d:
                line += f"; warm-start preservation: init {d['cos_init_median']:.3f} → final {d['cos_final_median']:.3f}"
            L.append(line + "\n")
    L.append("\n**Contextual embeddings (clean, init-free — the GeneCompass Fig-2a analogue):**\n")
    if v2r["contextual"]:
        for t, d in v2r["contextual"].items():
            L.append(f"- {t}: cosine median true-ortholog {d['median_homolog']:.3f} vs permuted-null {d['median_null']:.3f}, "
                     f"**AUC {d['auc']:.3f}** (n={d['n']} fully-covered pairs).\n")
        L.append("- Figure: `figures/v2_contextual.png`.\n")
    else:
        L.append("- (contextual extraction not available — run extract_embeddings on rat/human/mouse first)\n")

    L.append("\n## V3 — T4 new-token quality (random-init → clean learned-similarity test)\n")
    L.append("T4 rat-specific tokens were **random-initialised** (`N(0,0.02)`), so a T4 gene sitting nearer its "
             "human/mouse family peers than random measures what fine-tuning learned. The named-family set is thin "
             "(only ~4 CYP / 1 OLR by symbol; most T4 are LOC/unnamed), so we report CYP (the promise's example) "
             "directly and pool per-gene rank-AUC across all T4 genes whose symbol root matches ≥3 cross-species peers.\n")
    if "CYP" in v3r:
        d = v3r["CYP"]
        L.append(f"- **CYP (Cyp450) — {d['verdict']}**: {d['n_t4']} rat T4 Cyp genes ({', '.join(d['genes'])}) vs "
                 f"{d['n_peers']} human/mouse CYP peers — mean rank-AUC **{d['mean_auc']:.3f}** "
                 f"(cos family {d['mean_cos_family']:.3f} vs null {d['mean_cos_null']:.3f}); figure `figures/v3_cyp.png`.\n")
    if "pooled" in v3r:
        d = v3r["pooled"]
        L.append(f"- **Pooled across families — {d['verdict']}**: {d['n_t4_genes']} T4 genes spanning "
                 f"{d['n_distinct_families']} symbol-root families — mean per-gene rank-AUC **{d['mean_auc']:.3f}** "
                 f"(median {d['median_auc']:.3f}; {d['frac_auc_gt_0p65']*100:.0f}% of genes AUC>0.65). "
                 f"Example families: {', '.join(d['example_roots'][:12])}.\n")
    if not v3r:
        L.append("- (no T4 genes with a resolvable cross-species family — check symbol availability)\n")
    return "".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", default=str(_ROOT / "data/validation/aim1"))
    ap.add_argument("--final", default=str(_ROOT / "data/models/rat_genecompass_finetuned/models/rat_phase2_mixed_species/checkpoint-147941"))
    ap.add_argument("--init", default=str(_ROOT / "data/models/rat_genecompass_init"))
    ap.add_argument("--out", default=str(_ROOT / "reports/aim1_validation"))
    args = ap.parse_args()
    inp = Path(args.inp); out = Path(args.out); fig = out / "figures"
    fig.mkdir(parents=True, exist_ok=True)

    print("[analyze] V1 ...", flush=True); v1r = v1(inp, fig)
    print("[analyze] V2 ...", flush=True); v2r = v2(inp, args.final, args.init, fig)
    print("[analyze] V3 ...", flush=True); v3r = v3(args.final, fig)
    json.dump({"v1": v1r, "v2": v2r, "v3": v3r}, open(out / "metrics.json", "w"), indent=2)
    (out / "AIM1_VALIDATION_REPORT.md").write_text(md(v1r, v2r, v3r, inp))
    print(f"[analyze] wrote {out/'AIM1_VALIDATION_REPORT.md'} + metrics.json + figures/", flush=True)
    print("\n=== VERDICTS ===")
    print(f"  V1 clustering: {v1r['verdict']} (kNN cell-type purity {v1r['knn_purity_celltype']:.3f} vs chance "
          f"{v1r['chance_celltype']:.3f}; cross-study mix {v1r['cross_study_mix_within_celltype']:.3f})")
    print("  V2 contextual homolog AUC: " + (", ".join(f"{t}={d['auc']:.3f}" for t, d in v2r['contextual'].items()) or "n/a"))
    print("  V2 input AUC: " + (", ".join(f"{t}={d['auc']:.3f}" for t, d in v2r['input'].items() if 'auc' in d) or "n/a"))
    print("  V3 T4: " + ", ".join(f"{f}={d.get('mean_auc', float('nan')):.3f}" for f, d in v3r.items()))


if __name__ == "__main__":
    main()
