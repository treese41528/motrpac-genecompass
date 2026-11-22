#!/usr/bin/env python3
"""step6_release_labels_on_disk.py -- Step 6 (records, tests, hygiene).

Which manifests and summaries on disk still carry the wrong annotation label ("113", "mRatBN7.2",
"rel-113")? The BioMart files every stage read are Ensembl release 115 / GRCr8 (config/pipeline_config.yaml,
biomart block; byte-identical to the September 2025 archive). The label was wrong in the config until
2026-10-05, and each writer copied it. The files listed here are historical outputs: they are not edited.
This table is the correction note that goes with them.

Read-only. Looks at the JSON manifests and text summaries of the stage output folders named in the config
(the default arm) and at the Stage 2 manifests.

  python analysis/audit_remediation/step6_release_labels_on_disk.py
Out: docs/audit/remediation/record/step6/stale_release_labels_on_disk.tsv
"""
import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "lib"))
from gene_utils import load_config, resolve_path   # noqa: E402

OUT = ROOT / "docs/audit/remediation/record/step6/stale_release_labels_on_disk.tsv"
STALE = re.compile(r"\b113\b|mRatBN7|rel-113", re.I)
KEYS = re.compile(r"release|assembly|gene_space|biomart", re.I)


def walk(x, path=""):
    if isinstance(x, dict):
        for k, v in x.items():
            yield from walk(v, f"{path}/{k}")
    elif isinstance(x, list):
        for i, v in enumerate(x[:50]):
            yield from walk(v, f"{path}[{i}]")
    else:
        yield path, x


def main():
    cfg = load_config()
    p, d = cfg["paths"], cfg["deconvolution"]
    right = f"{cfg['biomart']['ensembl_release']} / {cfg['biomart']['assembly']}"
    folders = {
        "Stage 2 gene universe": resolve_path(cfg, p["gene_universe_dir"]),
        "Stage 3 ortholog mapping": resolve_path(cfg, p["ortholog_dir"]),
        "Stage 4 medians": resolve_path(cfg, p["median_dir"]),
        "Stage 5 tokenized corpus": resolve_path(cfg, p["tokenized_corpus_dir"]),
        "Stage 6 prior knowledge": resolve_path(cfg, p["prior_knowledge_dir"]),
        "Stage 8-9 pseudo-cells (default arm)": resolve_path(cfg, d["genecompass_input_dir"]),
    }
    rows = []
    for stage, folder in folders.items():
        files = sorted(folder.glob("*manifest*.json")) + sorted(folder.glob("*/summary.txt"))
        for f in files:
            rel = f.relative_to(ROOT / "data") if str(f).startswith(str(ROOT / "data")) else f
            if f.suffix == ".json":
                try:
                    doc = json.load(open(f))
                except (OSError, ValueError) as e:
                    rows.append({"stage": stage, "file": f"data/{rel}", "key": "-", "value_on_disk": f"unreadable: {e}",
                                 "stale": "", "correct_value": right})
                    continue
                for key, val in walk(doc):
                    if KEYS.search(key.rsplit("/", 1)[-1]) and isinstance(val, (str, int)):
                        rows.append({"stage": stage, "file": f"data/{rel}", "key": key, "value_on_disk": val,
                                     "stale": bool(STALE.search(str(val))), "correct_value": right})
            else:
                for line in open(f, errors="replace"):
                    m = re.search(r"gene_space=(\S+)", line)
                    if m:
                        rows.append({"stage": stage, "file": f"data/{rel}", "key": "gene_space",
                                     "value_on_disk": m.group(1), "stale": bool(STALE.search(m.group(1))),
                                     "correct_value": right})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["stage", "file", "key", "value_on_disk", "stale", "correct_value"],
                           delimiter="\t", lineterminator="\n")
        w.writeheader(); w.writerows(rows)
    stale = [r for r in rows if r["stale"] is True]
    print(f"{len(rows)} release/assembly labels read in {len({r['file'] for r in rows})} files; {len(stale)} stale")
    by = {}
    for r in stale:
        by.setdefault(r["stage"], []).append(r)
    for stage, rs in by.items():
        print(f"  {stage}: {len(rs)} stale labels in {len({r['file'] for r in rs})} files, e.g. "
              f"{rs[0]['file']} {rs[0]['key']} = {rs[0]['value_on_disk']!r}")
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
