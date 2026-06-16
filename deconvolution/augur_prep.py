#!/usr/bin/env python3
"""augur_prep.py -- corroboration step (1) for the Aim-2 supervised-subspace finding.

Two outputs:
  (A) Per-tissue inputs for CANONICAL Augur (R / neurorestore) under
      data/deconvolution/augur_input/<tissue>/:
        embed.tsv   768-d GeneCompass CLS embedding, cells x dims
        pca.tsv     PCA-50 of the log-normalized deconvolved Z, cells x PCs
        meta.tsv    cell_id, cell_type, sample, label(trained/control), sex, week
      run_augur.R runs Augur (RF, velocity mode = continuous features) on BOTH representations,
      so the published method weighs in on embedding-vs-PCA too. (The full gene space is covered
      by the Python PLS-1 probe in part B, avoiding a multi-GB dense genes x cells export.)

  (B) PCA-vs-embedding CONTROL (Python, reusing subspace_probe's PLS-1 CV probe): per
      (tissue x cell-type), held-out trained-vs-control AUC from three representations of the
      SAME pseudo-cells -- {GeneCompass 768-d embedding, PCA-50 of the deconvolved Z, full
      scaled genes} -- each with a permutation p. Answers: does the foundation-model embedding
      separate exercise better than a plain PCA baseline? (cf. "one PCA still rules them all",
      arXiv 2410.13956). Writes pca_control.tsv.

Join + alignment identical to pheno_merge_test.py / subspace_probe.py (mix{i} -> viallabel -> pheno;
h5ad reindexed to the embedding row order by pseudocell_id).
"""
import argparse, glob, os, re, sys
from pathlib import Path

import numpy as np, pandas as pd
import scipy.sparse as sp
from scipy.stats import wilcoxon
import anndata as ad
from datasets import load_from_disk
from joblib import Parallel, delayed
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parent))
from subspace_probe import make_folds, supervised                         # reuse the CV probe
sys.path.insert(0, str(Path(os.environ.setdefault(
    "PIPELINE_ROOT", str(Path(__file__).resolve().parents[1]))) / "lib"))
from gene_utils import load_config, resolve_path                          # noqa: E402

WEEK = {"control": 0, "1w": 1, "2w": 2, "4w": 4, "8w": 8}
SPACE_SEED = {"embed": 7, "pca": 11, "genes": 13}


def lognorm(X, target=1e4):
    X = X.toarray() if sp.issparse(X) else np.asarray(X, float)
    s = X.sum(1, keepdims=True); s[s == 0] = 1
    return np.log1p(X / s * target)


def zscore_clip(X, clip=10.0):
    mu = X.mean(0); sd = X.std(0); sd[sd == 0] = 1
    return np.clip((X - mu) / sd, -clip, clip)


def main():
    cfg = load_config(); dc = cfg["deconvolution"]
    gc = str(resolve_path(cfg, dc["genecompass_input_dir"]))
    pheno = str(resolve_path(cfg, dc["sample_pheno"]))
    bulk = str(resolve_path(cfg, dc["motrpac_bulk_out"]))
    ap = argparse.ArgumentParser()
    ap.add_argument("--gc-root", default=gc)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(gc), "augur_input"))
    ap.add_argument("--pca", type=int, default=50)
    ap.add_argument("--perms", type=int, default=500)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--ctl-out", default=os.path.join(gc, "pca_control.tsv"))
    args = ap.parse_args()

    ph = pd.read_csv(pheno, sep="\t", dtype=str); ph["viallabel"] = ph["viallabel"].astype(str)
    gcol = "group" if "group" in ph.columns else "key.anirandgroup"
    phm = ph.drop_duplicates("viallabel").set_index("viallabel")

    items = []
    for d in sorted(glob.glob(f"{args.gc_root}/*/")):
        tis = os.path.basename(d.rstrip("/"))
        emb_p, ds_p = os.path.join(d, "embeddings", "cell_embeddings.npy"), os.path.join(d, "dataset")
        h5_p = os.path.join(d, "pseudocells.h5ad")
        bs_p = os.path.join(bulk, tis.upper(), "bulk_samples.tsv")
        if not (os.path.exists(emb_p) and os.path.isdir(ds_p) and os.path.exists(h5_p)
                and os.path.exists(bs_p)):
            continue
        ds = load_from_disk(ds_p)
        order = list(ds["cell_id"]); emb = np.load(emb_p).astype(np.float64)
        samp = np.array(ds["sample"]); ct = np.array(ds["cell_type"])
        A = ad.read_h5ad(h5_p)
        pos = {str(n): i for i, n in enumerate(A.obs_names)}
        ridx = np.array([pos[o] for o in order])                          # align h5ad -> embedding order
        Xraw = A.X[ridx]; genes = list(map(str, A.var_names))
        vls = [l.strip() for l in open(bs_p) if l.strip()]
        idx = np.array([int(re.sub(r"\D", "", s)) - 1 for s in samp])
        vl = np.array([vls[i] if 0 <= i < len(vls) else "" for i in idx])
        grp = np.array([phm.loc[v, gcol] if v in phm.index else None for v in vl], dtype=object)
        sex = np.array([phm.loc[v, "sex"] if v in phm.index else None for v in vl], dtype=object)
        trained = np.array([None if g is None else
                            ("control" if "control" in str(g).lower() else "trained")
                            for g in grp], dtype=object)
        week = np.array([WEEK.get(str(g), np.nan) for g in grp], float)
        keep = (grp != None) & (sex != None)                              # noqa: E711

        Xln = lognorm(Xraw)[keep]                                         # cells x genes, log-norm
        Xz = zscore_clip(Xln)                                            # standard scaled (Seurat/scanpy)
        cells = [order[i] for i in np.where(keep)[0]]
        ncomp = min(args.pca, Xz.shape[0] - 1, Xz.shape[1])
        pcs = PCA(n_components=ncomp, random_state=0).fit_transform(Xz)

        # ---- (A) export canonical-Augur inputs (embedding + PCA-50 + meta; all small) ----
        # cell_id (= sanitized pseudocell_id) can COLLIDE across cell-type subtypes whose names
        # sanitized to the same string (e.g. kidney alpha/beta intercalated cells -> "_intercalated_cells"),
        # and R rejects duplicate row names -> index by a guaranteed-unique positional rowid instead;
        # keep cell_id as an informational column.
        od = os.path.join(args.out, tis); os.makedirs(od, exist_ok=True)
        rowid = pd.Index([f"c{i}" for i in range(len(cells))], name="rowid")
        pd.DataFrame(emb[keep], index=rowid).to_csv(os.path.join(od, "embed.tsv"), sep="\t")
        pd.DataFrame(pcs, index=rowid, columns=[f"PC{i+1}" for i in range(ncomp)]
                     ).to_csv(os.path.join(od, "pca.tsv"), sep="\t")
        pd.DataFrame({"rowid": rowid, "cell_id": cells, "cell_type": ct[keep], "sample": samp[keep],
                      "label": trained[keep], "sex": sex[keep], "week": week[keep]}
                     ).to_csv(os.path.join(od, "meta.tsv"), sep="\t", index=False)

        # ---- (B) per-cell-type probe items ----
        ek, ck, tk, wk = emb[keep], ct[keep], trained[keep], week[keep]
        for ci, c in enumerate(sorted(set(ck))):
            m = ck == c
            if m.sum() < 6 or len(set(tk[m])) < 2:
                continue
            items.append(dict(tissue=tis, cell_type=c, n=int(m.sum()),
                              embed=ek[m], pca=pcs[m], genes=Xz[m],
                              trained=tk[m], week=wk[m], seed=1000 * (hash(tis) % 9973) + ci))
        print(f"[{tis}] exported {int(keep.sum())} cells x {len(genes)} genes; "
              f"{len(set(ck))} cell types; PCA={ncomp}")

    # ---- (B) run the PCA-vs-embedding control (parallel) ----
    def run(it):
        rng_seed = it["seed"]
        out = {"tissue": it["tissue"], "cell_type": it["cell_type"], "n": it["n"]}
        y = (np.asarray(it["trained"]) == "trained").astype(float)
        f = make_folds(it["trained"], args.folds, np.random.default_rng(rng_seed + 1))
        if not f:
            return out
        for space in ["embed", "pca", "genes"]:
            auc, _, p = supervised(it[space], f, y, "auc", it["trained"], args.perms,
                                   np.random.default_rng(rng_seed + SPACE_SEED[space]))
            out[f"auc_{space}"] = round(auc, 3) if auc == auc else np.nan
            out[f"p_{space}"] = round(p, 4) if p == p else np.nan
        return out

    rows = Parallel(n_jobs=args.jobs, verbose=5)(delayed(run)(it) for it in items)
    res = pd.DataFrame(rows); res.to_csv(args.ctl_out, sep="\t", index=False)

    pd.set_option("display.width", 200, "display.max_rows", 300)
    print(f"\nwrote {args.ctl_out} ({len(res)} blocks); Augur inputs under {args.out}\n")
    print("=== does the GeneCompass embedding beat the baselines? (paired across blocks) ===")
    for a, b in [("embed", "pca"), ("embed", "genes"), ("pca", "genes")]:
        sub = res[[f"auc_{a}", f"auc_{b}"]].dropna()
        d = sub[f"auc_{a}"] - sub[f"auc_{b}"]
        try:
            w = wilcoxon(sub[f"auc_{a}"], sub[f"auc_{b}"]).pvalue
        except ValueError:
            w = np.nan
        print(f"  {a:5s} vs {b:5s}: median {a}={sub[f'auc_{a}'].median():.3f} "
              f"{b}={sub[f'auc_{b}'].median():.3f}  Δmedian={d.median():+.3f}  "
              f"{a}>{b} in {(d > 0).sum()}/{len(d)} blocks  Wilcoxon p={w:.2g}")

    hs = res[res.auc_embed >= 0.70].sort_values("auc_embed", ascending=False)
    print(f"\n=== hotspots (embed held-out AUC >= 0.70, n={len(hs)}): embedding vs PCA vs genes ===")
    print(hs[["tissue", "cell_type", "n", "auc_embed", "auc_pca", "auc_genes"]].head(25).to_string(index=False))


if __name__ == "__main__":
    main()
