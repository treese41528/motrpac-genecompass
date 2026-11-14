#!/usr/bin/env python
"""compare_references.py -- is each rebuilt reference (references_v5/) the same as the build it replaces?

For every tissue of deconvolution/tissue_references.yaml that has both `reference_dir` and `refs4_reference_dir`,
compare the three files BayesPrism reads (reference_counts.mtx, genes.tsv, cells_meta.tsv) by md5. Where they
differ, say how: cells, genes, and the cells per label in each build. Audit finding 4.8: 7 of 15 references had no
build record and 6 were built from an uncommitted tree, so none had been reproduced
(docs/audit/remediation/STEP3_PREREG.md 2.2).

Usage: compare_references.py <out_dir>
"""
import hashlib
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
FILES = ["reference_counts.mtx", "genes.tsv", "cells_meta.tsv"]


def md5(path, chunk=1 << 24):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def main():
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(open(ROOT / "deconvolution/tissue_references.yaml"))
    rows, labels, seen = [], [], {}
    for tis, c in cfg.items():
        if not isinstance(c, dict) or not c.get("reference_dir") or not c.get("refs4_reference_dir"):
            continue
        new, old = ROOT / c["reference_dir"], ROOT / c["refs4_reference_dir"]
        if str(new) in seen:                                   # SKMGN and SKMVL share one build
            rows.append({**seen[str(new)], "tissue": tis, "note": f"shared with {seen[str(new)]['tissue']}"})
            continue
        r = {"tissue": tis, "reference": new.name, "built": (new / "cells_meta.tsv").exists(), "note": ""}
        if r["built"] and (old / "cells_meta.tsv").exists():
            for f in FILES:
                r[f"same_{f.split('.')[0]}"] = md5(new / f) == md5(old / f)
            r["identical"] = all(r[f"same_{f.split('.')[0]}"] for f in FILES)
            mn = pd.read_csv(new / "cells_meta.tsv", sep="\t", dtype=str)
            mo = pd.read_csv(old / "cells_meta.tsv", sep="\t", dtype=str)
            gn = [l.strip() for l in open(new / "genes.tsv")]; go = [l.strip() for l in open(old / "genes.tsv")]
            r.update(cells_new=len(mn), cells_old=len(mo), genes_new=len(gn), genes_old=len(go),
                     genes_shared=len(set(gn) & set(go)), types_new=mn.cell_type.nunique(), types_old=mo.cell_type.nunique(),
                     states_new=mn.cell_state.nunique(), states_old=mo.cell_state.nunique())
            if not r["identical"]:
                cn, co = mn.cell_type.value_counts(), mo.cell_type.value_counts()
                for lab in sorted(set(cn.index) | set(co.index)):
                    if int(cn.get(lab, 0)) != int(co.get(lab, 0)):
                        labels.append({"tissue": tis, "cell_type": lab, "cells_old": int(co.get(lab, 0)),
                                       "cells_new": int(cn.get(lab, 0))})
        seen[str(new)] = r
        rows.append(r)
        print({k: r.get(k) for k in ("tissue", "built", "identical", "cells_old", "cells_new", "genes_old", "genes_new",
                                     "types_old", "types_new")}, flush=True)
    pd.DataFrame(rows).to_csv(out / "references_v5_vs_refs4.tsv", sep="\t", index=False)
    pd.DataFrame(labels, columns=["tissue", "cell_type", "cells_old", "cells_new"]).to_csv(
        out / "references_v5_label_changes.tsv", sep="\t", index=False)
    d = pd.DataFrame(rows)
    print(f"\n{int(d.identical.fillna(False).sum())} of {len(d)} tissues' references are byte-identical to the build they replace")


if __name__ == "__main__":
    main()
