#!/usr/bin/env python
"""05b_check_barcodes.py -- is any kept held-out file the same cells as a training file, under other accessions?

A supplementary check, not in the registrations. The registered exclusions (STEP5_PREREG.md M2, STEP4_PREREG.md
V3) compare GEO sample accessions and declared SuperSeries / SubSeries relations. The same library can be
deposited twice under new sample accessions and processed with another reference (inside the held-out set,
GSE263307 is such a second deposit of half of GSE171165, found by 03_file_plan.py), and then neither the
accessions nor the token sequences (05_check_unseen.py) agree. What survives a second deposit is the cell
barcode. This script applies the corpus audit's barcode test (tmp/data_audit/audit.py `barcodes` and `pairs`,
the test behind the corpus plan's "same cells as ..." rows) between every kept held-out QC file and every QC
file of the corpus plan (config/corpus_v2_plan.tsv, all 868 rows that exist under data/training/qc_h5ad, kept
or dropped; read only), with no restriction on cell counts:

  a pair is examined when the two files share at least half of the smaller file's barcode cores (a run of
  12 or more of A, C, G, T in the cell name); it is then "same_cells" when at least 90% of the shared cells
  have the same total count, "same_cells_other_processing" when their log totals correlate at 0.98 or more,
  otherwise "different_cells" (barcodes reused by design: plate, combinatorial and spatial protocols).

Writes
  record/step4_heldout/unseen_barcode_pairs.tsv    every examined held-out x training pair
  record/step4_heldout/unseen_barcode_check.tsv    per held-out series: files compared, pairs examined, verdicts
Usage: 05b_check_barcodes.py
"""
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C   # noqa: E402

CORE = re.compile(r"[ACGT]{12,}")
TRAIN_QC = C.ROOT / "data/training/qc_h5ad"


def obs_names(h):
    obs = h["obs"]
    key = obs.attrs.get("_index", "_index")
    key = key.decode() if isinstance(key, bytes) else key
    return [x.decode() if isinstance(x, bytes) else str(x) for x in obs[key][:]]


def cores_of(path):
    """Sorted unique 64-bit hashes of the file's barcode cores (hash() is stable within this process)."""
    with h5py.File(path, "r") as h:
        names = obs_names(h)
    return np.unique(np.fromiter((hash(m.group(0) if (m := CORE.search(n)) else n) for n in names),
                                 dtype=np.int64, count=len(names))), len(names)


def totals(path):
    """{barcode core: total count} from the stored library_size, or from the counts when it is absent
    (the `raw` slot when X is not integer), as tmp/data_audit/audit.py did."""
    with h5py.File(path, "r") as h:
        names = obs_names(h)
        if "library_size" in h["obs"]:
            tot = np.asarray(h["obs"]["library_size"][:], dtype=float)
        else:
            X = h["X"]
            probe = np.asarray(X["data"][:5000] if isinstance(X, h5py.Group) else X[:5].ravel(), dtype=float)
            if probe.size and np.any(np.abs(probe - np.round(probe)) > 1e-6) and "raw" in h:
                X = h["raw"]["X"]
            if isinstance(X, h5py.Group):
                data = X["data"][:]; indptr = X["indptr"][:]
                cs = np.concatenate([[0.0], np.cumsum(data, dtype=np.float64)])
                tot = cs[indptr[1:]] - cs[indptr[:-1]]
            else:
                tot = np.asarray(X[:]).sum(1)
    return dict(zip([(m.group(0) if (m := CORE.search(n)) else n) for n in names], np.asarray(tot, dtype=float)))


def verdict(a, b):
    shared = sorted(set(a) & set(b))
    if not shared:
        return 0, float("nan"), float("nan"), "different_cells"
    x = np.array([a[k] for k in shared]); y = np.array([b[k] for k in shared])
    eq = float(np.mean(np.abs(x - y) < 1e-6))
    r = float(np.corrcoef(np.log1p(x), np.log1p(y))[0, 1]) if len(shared) > 2 and x.std() > 0 and y.std() > 0 \
        else float("nan")
    v = "same_cells" if eq >= 0.9 else "same_cells_other_processing" if r >= 0.98 else \
        "partly_same" if eq >= 0.1 else "different_cells"
    return len(shared), eq, r, v


def main():
    plan = C.read_tsv(C.PLAN)
    train = [(r["sample_id"], r["study"], r["action"], TRAIN_QC / f"{r['sample_id']}.h5ad") for r in plan]
    missing = [t[0] for t in train if not t[3].is_file()]
    train = [t for t in train if t[3].is_file()]
    held = [(r["sample_id"], r["series"], C.QC_DIR / f"{r['sample_id']}.h5ad")
            for r in C.read_tsv(C.REC / "file_plan.tsv") if r["action"] == "keep"]
    print(f"training QC files: {len(train)} (missing on disk: {len(missing)}); held-out kept files: {len(held)}", flush=True)

    hashes, owner, n_train = [], [], []
    for i, (sid, study, action, path) in enumerate(train):
        u, n = cores_of(path)
        hashes.append(u); owner.append(np.full(len(u), i, dtype=np.int32)); n_train.append(len(u))
        if i % 200 == 0:
            print(f"  read {i + 1}/{len(train)} training files", flush=True)
    th = np.concatenate(hashes); to = np.concatenate(owner)
    order = np.argsort(th, kind="stable"); th, to = th[order], to[order]
    n_train = np.asarray(n_train)

    pairs, per = [], defaultdict(lambda: {"files": 0, "cells": 0, "pairs_examined": 0, "max_overlap": 0.0,
                                          "verdicts": Counter(), "partners": Counter()})
    for j, (sid, series, path) in enumerate(held):
        u, n = cores_of(path)
        lo, hi = np.searchsorted(th, u, "left"), np.searchsorted(th, u, "right")
        cnt = hi - lo
        idx = np.repeat(lo, cnt) + (np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt))
        shared = np.bincount(to[idx], minlength=len(train)) if len(idx) else np.zeros(len(train), dtype=int)
        ov = shared / np.minimum(len(u), np.maximum(n_train, 1))
        p = per[series]; p["files"] += 1; p["cells"] += n; p["max_overlap"] = max(p["max_overlap"], float(ov.max()))
        mine = None
        for i in np.flatnonzero(ov >= 0.5):
            mine = mine if mine is not None else totals(path)
            k, eq, r, v = verdict(totals(train[i][3]), mine)
            pairs.append({"heldout_file": sid, "heldout_series": series, "training_file": train[i][0],
                          "training_series": train[i][1], "training_plan_action": train[i][2],
                          "cells_heldout": n, "cells_training": int(n_train[i]), "overlap_of_smaller": round(float(ov[i]), 3),
                          "shared": k, "identical_total_share": round(eq, 3), "log_total_correlation": round(r, 4),
                          "verdict": v})
            p["pairs_examined"] += 1; p["verdicts"][v] += 1
            if v != "different_cells":
                p["partners"][train[i][1]] += 1
        if j % 25 == 0:
            print(f"  compared {j + 1}/{len(held)} held-out files; pairs examined so far {len(pairs)}", flush=True)

    C.write_tsv(C.REC / "unseen_barcode_pairs.tsv", pairs,
                ["heldout_file", "heldout_series", "training_file", "training_series", "training_plan_action",
                 "cells_heldout", "cells_training", "overlap_of_smaller", "shared", "identical_total_share",
                 "log_total_correlation", "verdict"])
    rows = [{"series": s, "kept_files_compared": p["files"], "cells": p["cells"],
             "training_files_compared": len(train), "max_barcode_overlap_with_a_training_file": round(p["max_overlap"], 4),
             "pairs_examined_overlap_0.5": p["pairs_examined"], "verdicts": C.counter_str(list(p["verdicts"].elements())),
             "files_same_cells_as_training": sum(v for k, v in p["verdicts"].items() if k != "different_cells"),
             "training_series_matched": C.counter_str(list(p["partners"].elements()))} for s, p in sorted(per.items())]
    C.write_tsv(C.REC / "unseen_barcode_check.tsv", rows)
    bad = [r for r in pairs if r["verdict"] != "different_cells"]
    print(f"held-out files {len(held)} x training files {len(train)}: pairs sharing >= half the barcodes {len(pairs)}; "
          f"same cells (any processing) {len(bad)}")
    for r in bad:
        print("  SAME CELLS:", r)


if __name__ == "__main__":
    main()
