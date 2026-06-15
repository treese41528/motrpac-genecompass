#!/usr/bin/env python3
"""pheno_merge_test.py -- the Aim-2 validation GATE.

Question: does within-cell-type pseudo-cell EMBEDDING variation track the MoTrPAC exercise
design (training group / sex)? If not, cell-type-resolved downstream analyses (GRN, perturbation)
on these pseudo-cells cannot recover exercise biology, regardless of how clean the wiring is.

Join: pseudo-cell sample 'mix{i}' -> i-th viallabel in <TISSUE>/bulk_samples.tsv -> PHENO[viallabel].
(run_deconvolution.R renames mixture rows to mix{i}, discarding viallabels, so they are recovered
by ROW ORDER -- validated 300/300 on liver, design 2 sex x 5 group.)

Per (tissue, cell type) reports multivariate variance-explained (eta^2) of:
  GROUP (control/1w/2w/4w/8w), TRAINED (any training vs control), SEX -- on the 768-d embedding,
with a label-permutation p-value. Healthy Aim-2 substrate = group/trained eta^2 above the null in
abundant cell types. (Sex-chromosome genes were removed upstream, so SEX signal must be autosomal.)

Usage: python deconvolution/pheno_merge_test.py [--gc-root data/deconvolution/genecompass_input]
       [--pheno deconvolution/reference/motrpac_sample_pheno.tsv]
       [--bulk-root data/deconvolution/motrpac_bulk] [--perms 1000] [--min-per-group 3]
"""
import argparse
import glob
import os
import re

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_from_disk

sys.path.insert(0, str(Path(os.environ.setdefault("PIPELINE_ROOT", str(Path(__file__).resolve().parents[1]))) / "lib"))
from gene_utils import load_config, resolve_path                       # noqa: E402


def eta2(E, labels):
    """Multivariate variance-explained of a categorical factor over rows of E."""
    labels = np.asarray(labels); mu = E.mean(0)
    sst = ((E - mu) ** 2).sum()
    if sst <= 0:
        return np.nan
    ssb = sum(((E[labels == g].mean(0) - mu) ** 2).sum() * (labels == g).sum()
              for g in np.unique(labels))
    return ssb / sst


def perm_p(E, labels, obs, perms, rng):
    labels = np.asarray(labels)
    ge = sum(eta2(E, rng.permutation(labels)) >= obs for _ in range(perms))
    return (ge + 1) / (perms + 1)


def main():
    cfg = load_config(); dc = cfg["deconvolution"]
    gc = resolve_path(cfg, dc["genecompass_input_dir"])
    ap = argparse.ArgumentParser()
    ap.add_argument("--gc-root", default=gc)
    ap.add_argument("--pheno", default=resolve_path(cfg, dc["sample_pheno"]))
    ap.add_argument("--bulk-root", default=resolve_path(cfg, dc["motrpac_bulk_out"]))
    ap.add_argument("--perms", type=int, default=1000)
    ap.add_argument("--min-per-group", type=int, default=3, help="skip cell types with a group cell < this")
    ap.add_argument("--out", default=os.path.join(gc, "pheno_merge_test.tsv"))
    args = ap.parse_args()

    ph = pd.read_csv(args.pheno, sep="\t", dtype=str)
    ph["viallabel"] = ph["viallabel"].astype(str)
    gcol = "group" if "group" in ph.columns else "key.anirandgroup"
    phm = ph.drop_duplicates("viallabel").set_index("viallabel")
    rng = np.random.default_rng(0)

    rows = []
    for d in sorted(glob.glob(f"{args.gc_root}/*/")):
        tis = os.path.basename(d.rstrip("/"))
        emb_p, ds_p = os.path.join(d, "embeddings", "cell_embeddings.npy"), os.path.join(d, "dataset")
        bs_p = os.path.join(args.bulk_root, tis.upper(), "bulk_samples.tsv")
        if not (os.path.exists(emb_p) and os.path.isdir(ds_p) and os.path.exists(bs_p)):
            continue
        emb = np.load(emb_p); ds = load_from_disk(ds_p)
        ct = np.array(ds["cell_type"]); samp = np.array(ds["sample"])
        if emb.shape[0] != len(ct):
            print(f"[{tis}] MISALIGNED — skip"); continue
        vls = [l.strip() for l in open(bs_p) if l.strip()]
        idx = np.array([int(re.sub(r"\D", "", s)) - 1 for s in samp])          # mix{i} -> i-1
        vl = np.array([vls[i] if 0 <= i < len(vls) else "" for i in idx])
        grp = np.array([phm.loc[v, gcol] if v in phm.index else None for v in vl], dtype=object)
        sex = np.array([phm.loc[v, "sex"] if v in phm.index else None for v in vl], dtype=object)
        trained = np.array([None if g is None else ("control" if "control" in str(g).lower() else "trained")
                            for g in grp], dtype=object)
        for c in sorted(set(ct)):
            m = (ct == c) & (grp != None) & (sex != None)                       # noqa: E711
            if m.sum() < 6:
                continue
            E = emb[m]; g, s, t = grp[m], sex[m], trained[m]
            gc_ok = min(np.bincount(pd.factorize(g)[0])) >= args.min_per_group
            r = {"tissue": tis, "cell_type": c, "n": int(m.sum())}
            for name, lab, ok in [("group", g, gc_ok), ("trained", t, True), ("sex", s, True)]:
                if len(np.unique(lab)) < 2 or not ok:
                    r[f"eta2_{name}"], r[f"p_{name}"] = np.nan, np.nan; continue
                o = eta2(E, lab); r[f"eta2_{name}"] = round(float(o), 3)
                r[f"p_{name}"] = round(perm_p(E, lab, o, args.perms, rng), 4)
            rows.append(r)
    res = pd.DataFrame(rows)
    if res.empty:
        print("no tissues with embeddings+bulk_samples found"); return
    res.to_csv(args.out, sep="\t", index=False)
    pd.set_option("display.width", 200, "display.max_rows", 300)
    print(res.to_string(index=False))
    sig = res[(res.p_trained < 0.05) | (res.p_group < 0.05)]
    print(f"\n(tissue,cell_type) with exercise signal (p_group or p_trained < 0.05): "
          f"{len(sig)}/{len(res)}")
    print(f"  ... of which eta2_trained >= 0.10: {int((sig.eta2_trained >= 0.10).sum())}")
    print(f"sex signal (p_sex<0.05): {int((res.p_sex < 0.05).sum())}/{len(res)}  -> wrote {args.out}")


if __name__ == "__main__":
    main()
