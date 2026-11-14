"""Score the GRN decoy null (GRN_DECOY_NULL_PREREG.md): edge-level FDR(t) = P_decoy(|z|>=t) / P_TF(|z|>=t),
per network and pooled, plus t* (smallest grid t with pooled FDR <= 0.10 and >= 100 TF edges); hub level =
each TF's sum|delta| against the same network's decoys (empirical p, BH within network).

Reading. The null asks whether deleting a TF moves the model's read-out differently from deleting a non-TF
gene of the same expression. It does not make the hub tables anything more than model read-outs.

Added for the Step 3 rerun (docs/audit/remediation/STEP3_PREREG.md 8.1, audit finding 9.3):
  * the summary states the number of DISTINCT TFs beside the number of TF x network pairs. Earlier documents
    quoted the pair count ("0 of 723 TFs") as if it were a count of TFs;
  * the match distance of the decoys (written by build_grn_pooled.py --decoy-seed) is summarised per network;
  * the edge rule of each TF network and of its decoy network is read from their provenance records, and a
    pair built under different rules is not scored: the null is only valid when both went through the same
    difference.

Outputs in --out: fdr_per_network.tsv, fdr_curve_pooled.tsv, hub_vs_decoy.tsv, summary.txt,
provenance_grn_decoy_fdr.json.

  python deconvolution/grn_decoy_fdr.py [--tf-dir DIR] [--decoy-dir DIR] [--out DIR]
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("PIPELINE_ROOT") or Path(__file__).resolve().parents[1])
ANCHORS = {"Mef2c", "Nr4a1", "Myf6", "Tfeb", "Epas1"}
GRID = np.arange(2.0, 50.01, 0.5)
NOT_RECORDED = "not recorded"          # a network built before provenance records were written


def bh(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p)
    q = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    r = np.empty(n); r[o] = np.minimum(q, 1); return r


def edge_rule(network):
    """The edge rule a network table was built under, from provenance_<stem>.json beside it
    (build_grn_pooled.py writes params.edge_rule). A table with no record predates the shared-gene rule."""
    f = Path(network).parent / f"provenance_{Path(network).stem}.json"
    try:
        return json.loads(f.read_text()).get("params", {}).get("edge_rule") or NOT_RECORDED
    except (OSError, ValueError):
        return NOT_RECORDED


def match_distances(decoy_network):
    """Match distance of every decoy of one network, from <stem>_decoy_match.tsv (all matches) or, failing
    that, the match_dist column of the decoy table (decoys with edges only). None if neither exists."""
    f = Path(str(decoy_network)[:-4] + "_decoy_match.tsv")
    if f.exists():
        return pd.read_csv(f, sep="\t", usecols=["match_dist"]).match_dist.values
    if "match_dist" in pd.read_csv(decoy_network, sep="\t", nrows=0).columns:
        d = pd.read_csv(decoy_network, sep="\t", usecols=["tf", "match_dist"])
        return d.drop_duplicates("tf").match_dist.values
    return None


def score(tf_dir, decoy_dir):
    """Score every TF network that has a decoy network of the same file name.
    Returns (per-network table, hub table, pooled FDR curve, t*, list of skipped networks)."""
    rows, hubs, skipped = [], [], []
    pooled = {"tf_n": 0, "dc_n": 0, "tf": np.zeros(len(GRID)), "dc": np.zeros(len(GRID))}
    for f in sorted(Path(tf_dir).glob("*_pooled.tsv")):
        g = Path(decoy_dir) / f.name
        name = f.name.replace("_pooled.tsv", "")
        if not g.exists():
            skipped.append(f"missing decoy network: {g.name}"); continue
        rt, rd = edge_rule(f), edge_rule(g)
        if rt != rd:
            # A TF network on shared genes against decoys differenced against zero (or the reverse) compares
            # two different statistics. Registration 8.1: the decoy null is rerun with the same rule.
            skipped.append(f"{name}: TF network edge rule '{rt}' but decoy network '{rd}'; not scored"); continue
        tf = pd.read_csv(f, sep="\t", usecols=["tf", "tf_symbol", "delta", "z"])
        dc = pd.read_csv(g, sep="\t", usecols=["tf", "delta", "z", "decoy_for"])
        zt, zd = tf.z.abs().values, dc.z.abs().values
        ct = np.array([(zt >= t).sum() for t in GRID]); cd = np.array([(zd >= t).sum() for t in GRID])
        pooled["tf_n"] += len(zt); pooled["dc_n"] += len(zd); pooled["tf"] += ct; pooled["dc"] += cd
        fdr2 = (cd[0] / len(zd)) / (ct[0] / len(zt)) if ct[0] else np.nan
        md = match_distances(g)
        rows.append({"network": name, "n_tf": tf.tf.nunique(), "n_decoy": dc.tf.nunique(),
                     "tf_edges": len(zt), "decoy_edges": len(zd), "tf_frac_z2": ct[0] / len(zt),
                     "decoy_frac_z2": cd[0] / len(zd), "fdr_z2": round(fdr2, 4),
                     "edge_rule": rt,
                     "n_matches": len(md) if md is not None else np.nan,
                     "match_dist_median": round(float(np.median(md)), 6) if md is not None else np.nan,
                     "match_dist_p95": round(float(np.percentile(md, 95)), 6) if md is not None else np.nan,
                     "match_dist_max": round(float(np.max(md)), 6) if md is not None else np.nan})
        # Hub strength = sum of |delta| over ALL of a TF's edges (never a count of |z|>=2 edges).
        hs = tf.assign(a=tf.delta.abs()).groupby(["tf", "tf_symbol"]).a.sum().reset_index()
        ds = dc.assign(a=dc.delta.abs()).groupby("tf").a.sum().values
        hs["p"] = [(1 + (ds >= v).sum()) / (1 + len(ds)) for v in hs.a]
        hs["q"] = bh(hs.p.values); hs["network"] = name
        hubs.append(hs.rename(columns={"a": "hub_strength"}))
    if not rows:
        return None, None, None, None, skipped
    R = pd.DataFrame(rows)
    H = pd.concat(hubs, ignore_index=True)
    pf = (pooled["dc"] / pooled["dc_n"]) / np.where(pooled["tf"] > 0, pooled["tf"] / pooled["tf_n"], np.nan)
    curve = pd.DataFrame({"t": GRID, "tf_edges": pooled["tf"].astype(int), "decoy_edges": pooled["dc"].astype(int),
                          "pooled_fdr": np.round(pf, 4)})
    ok = curve[(curve.pooled_fdr <= 0.10) & (curve.tf_edges >= 100)]
    tstar = float(ok.t.iloc[0]) if len(ok) else None
    return R, H, curve, tstar, skipped


def summary_text(R, H, curve, tstar, skipped=()):
    """The printed summary. Counts are given as TF x network pairs AND as distinct TFs (finding 9.3)."""
    f2 = curve.pooled_fdr.iloc[0]
    verdict = ("FDR(2) <= 0.10: |z|>=2 edges carry FDR <= 10% vs random-gene deletion" if f2 <= 0.10 else
               "0.10 < FDR(2) <= 0.50: |z| stays a ranking" if f2 <= 0.50 else
               "FDR(2) > 0.50: |z|>=2 edges not distinguishable from random-gene deletion")
    n_pairs, n_tfs = len(H), H.tf.nunique()
    n_sig = int((H.q < 0.10).sum())
    rules = sorted(set(R.edge_rule))
    lines = [R.to_string(index=False), "",
             f"edge rule of the networks scored: {', '.join(rules)}",
             f"TF x network pairs: {n_pairs} in {len(R)} networks; distinct TFs: {n_tfs}; "
             f"decoy x network pairs: {int(R.n_decoy.sum())}",
             f"pooled FDR(2) = {f2:.3f}  ({int(curve.tf_edges.iloc[0]):,} TF / {int(curve.decoy_edges.iloc[0]):,} decoy edges at |z|>=2)",
             f"t* (pooled FDR <= 0.10, >= 100 TF edges): {tstar if tstar is not None else 'none on the grid 2..50'}",
             f"registered reading: {verdict}", ""]
    if R.match_dist_median.notna().any():
        lines += ["decoy match distance |log1p(decoy expression) - log1p(TF expression)| per network: median "
                  f"{R.match_dist_median.min():.4g} to {R.match_dist_median.max():.4g}, largest single "
                  f"distance {R.match_dist_max.max():.4g} (columns match_dist_* above)", ""]
    else:
        lines += ["decoy match distance: not recorded in these decoy networks (built before it was written out)", ""]
    lines += [f"hub level (TF sum|delta| vs same-network decoys, BH within network): {n_sig} of {n_pairs} "
              f"TF x network pairs at q<0.10 ({H[H.q < 0.10].tf.nunique()} of {n_tfs} distinct TFs)"]
    for n, h in H.groupby("network"):
        an = h[h.tf_symbol.isin(ANCHORS)]
        lines.append(f"  {n}: {int((h.q < 0.10).sum())}/{len(h)} TFs q<0.10; anchors: " +
                     (", ".join(f"{r.tf_symbol} q={r.q:.3f}" for r in an.itertuples()) or "none present"))
    if skipped:
        lines += ["", "not scored:"] + [f"  {s}" for s in skipped]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tf-dir", default=None, help="default: deconvolution.grn_dir of the active config")
    ap.add_argument("--decoy-dir", default=None, help="default: <tf-dir>_decoy")
    ap.add_argument("--out", default=None, help="default: <decoy-dir>/fdr")
    a = ap.parse_args()
    sys.path.insert(0, str(ROOT / "lib"))
    if a.tf_dir is None:                                   # derive from the config, never a hardcoded arm
        from gene_utils import load_config, resolve_path
        _c = load_config()
        a.tf_dir = str(resolve_path(_c, _c["deconvolution"]["grn_dir"]))
    if a.decoy_dir is None:
        a.decoy_dir = str(a.tf_dir).rstrip("/") + "_decoy"
    out = Path(a.out or Path(a.decoy_dir) / "fdr"); out.mkdir(parents=True, exist_ok=True)

    R, H, curve, tstar, skipped = score(a.tf_dir, a.decoy_dir)
    for s in skipped:
        print(s)
    if R is None:
        sys.exit(f"no TF network under {a.tf_dir} could be scored against {a.decoy_dir}")
    R.to_csv(out / "fdr_per_network.tsv", sep="\t", index=False)
    curve.to_csv(out / "fdr_curve_pooled.tsv", sep="\t", index=False)
    H.sort_values(["network", "p"]).to_csv(out / "hub_vs_decoy.tsv", sep="\t", index=False)
    txt = summary_text(R, H, curve, tstar, skipped)
    print(txt); (out / "summary.txt").write_text(txt + "\n")

    from provenance import write_provenance
    write_provenance(out, inputs=[a.tf_dir, a.decoy_dir],
                     params={"tf_dir": a.tf_dir, "decoy_dir": a.decoy_dir, "n_networks": int(len(R)),
                             "n_tf_network_pairs": int(len(H)), "n_distinct_tfs": int(H.tf.nunique()),
                             "edge_rules": sorted(set(R.edge_rule)), "pooled_fdr_z2": float(curve.pooled_fdr.iloc[0]),
                             "t_star": tstar, "not_scored": list(skipped)})


if __name__ == "__main__":
    main()
