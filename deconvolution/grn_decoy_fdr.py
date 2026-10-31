"""Score the GRN decoy null (GRN_DECOY_NULL_PREREG.md): edge-level FDR(t) = P_decoy(|z|>=t) / P_TF(|z|>=t),
per network and pooled, plus t* (smallest grid t with pooled FDR <= 0.10 and >= 100 TF edges); hub level =
each TF's sum|delta| against the same network's decoys (empirical p, BH within network).

  python deconvolution/grn_decoy_fdr.py [--tf-dir DIR] [--decoy-dir DIR] [--out DIR]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--tf-dir", default=str(ROOT / "data/deconvolution/grn_v2_refs4"))
ap.add_argument("--decoy-dir", default=str(ROOT / "data/deconvolution/grn_v2_refs4_decoy"))
ap.add_argument("--out", default=None, help="default: <decoy-dir>/fdr")
a = ap.parse_args()
out = Path(a.out or Path(a.decoy_dir) / "fdr"); out.mkdir(parents=True, exist_ok=True)
ANCHORS = {"Mef2c", "Nr4a1", "Myf6", "Tfeb", "Epas1"}
GRID = np.arange(2.0, 50.01, 0.5)


def bh(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p)
    q = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    r = np.empty(n); r[o] = np.minimum(q, 1); return r


rows, hubs, pooled = [], [], {"tf_n": 0, "dc_n": 0, "tf": np.zeros(len(GRID)), "dc": np.zeros(len(GRID))}
for f in sorted(Path(a.tf_dir).glob("*_pooled.tsv")):
    g = Path(a.decoy_dir) / f.name
    if not g.exists():
        print(f"missing decoy network: {g.name}"); continue
    tf = pd.read_csv(f, sep="\t", usecols=["tf", "tf_symbol", "delta", "z"])
    dc = pd.read_csv(g, sep="\t", usecols=["tf", "delta", "z", "decoy_for"])
    zt, zd = tf.z.abs().values, dc.z.abs().values
    ct = np.array([(zt >= t).sum() for t in GRID]); cd = np.array([(zd >= t).sum() for t in GRID])
    pooled["tf_n"] += len(zt); pooled["dc_n"] += len(zd); pooled["tf"] += ct; pooled["dc"] += cd
    fdr2 = (cd[0] / len(zd)) / (ct[0] / len(zt)) if ct[0] else np.nan
    rows.append({"network": f.name.replace("_pooled.tsv", ""), "n_tf": tf.tf.nunique(), "n_decoy": dc.tf.nunique(),
                 "tf_edges": len(zt), "decoy_edges": len(zd), "tf_frac_z2": ct[0] / len(zt),
                 "decoy_frac_z2": cd[0] / len(zd), "fdr_z2": round(fdr2, 4)})
    hs = tf.assign(a=tf.delta.abs()).groupby(["tf", "tf_symbol"]).a.sum().reset_index()
    ds = dc.assign(a=dc.delta.abs()).groupby("tf").a.sum().values
    hs["p"] = [(1 + (ds >= v).sum()) / (1 + len(ds)) for v in hs.a]
    hs["q"] = bh(hs.p.values); hs["network"] = rows[-1]["network"]
    hubs.append(hs.rename(columns={"a": "hub_strength"}))

R = pd.DataFrame(rows)
H = pd.concat(hubs, ignore_index=True)
pf = (pooled["dc"] / pooled["dc_n"]) / np.where(pooled["tf"] > 0, pooled["tf"] / pooled["tf_n"], np.nan)
curve = pd.DataFrame({"t": GRID, "tf_edges": pooled["tf"].astype(int), "decoy_edges": pooled["dc"].astype(int),
                      "pooled_fdr": np.round(pf, 4)})
ok = curve[(curve.pooled_fdr <= 0.10) & (curve.tf_edges >= 100)]
tstar = float(ok.t.iloc[0]) if len(ok) else None
R.to_csv(out / "fdr_per_network.tsv", sep="\t", index=False)
curve.to_csv(out / "fdr_curve_pooled.tsv", sep="\t", index=False)
H.sort_values(["network", "p"]).to_csv(out / "hub_vs_decoy.tsv", sep="\t", index=False)

f2 = curve.pooled_fdr.iloc[0]
verdict = ("FDR(2) <= 0.10: |z|>=2 edges carry FDR <= 10% vs random-gene deletion" if f2 <= 0.10 else
           "0.10 < FDR(2) <= 0.50: |z| stays a ranking" if f2 <= 0.50 else
           "FDR(2) > 0.50: |z|>=2 edges not distinguishable from random-gene deletion")
lines = [R.to_string(index=False), "",
         f"pooled FDR(2) = {f2:.3f}  ({int(curve.tf_edges.iloc[0]):,} TF / {int(curve.decoy_edges.iloc[0]):,} decoy edges at |z|>=2)",
         f"t* (pooled FDR <= 0.10, >= 100 TF edges): {tstar if tstar is not None else 'none on the grid 2..50'}",
         f"registered reading: {verdict}", "",
         "hub level (TF sum|delta| vs same-network decoys, BH within network):"]
for n, h in H.groupby("network"):
    an = h[h.tf_symbol.isin(ANCHORS)]
    lines.append(f"  {n}: {int((h.q < 0.10).sum())}/{len(h)} TFs q<0.10; anchors: " +
                 (", ".join(f"{r.tf_symbol} q={r.q:.3f}" for r in an.itertuples()) or "none present"))
txt = "\n".join(lines); print(txt); (out / "summary.txt").write_text(txt + "\n")
