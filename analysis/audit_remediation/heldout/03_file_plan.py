#!/usr/bin/env python
"""03_file_plan.py -- which QC'd held-out files are kept, by the corpus's own cleaning rules; QC counts.

Implements STEP5_PREREG.md M2 ("Reported: studies, tissues, cells passing QC") and the organism exclusion of
the held-out test (STEP4_PREREG.md V3; the task's rule "remove non-rat samples by the same organism rule the
corpus used"). The v2 corpus is not every file Stage 2 wrote: config/corpus_v2_plan.tsv keeps a file only if
it is rat by its GEO sample organism, is not a copy of another kept file, and holds raw counts. This script
writes the same kind of plan for the held-out files, one row per file Stage 2 discovered, first rule that
applies:

  no_cells              Stage 2 wrote no QC file (load failed, no gene mapped, too few cells).
  drop_nonrat           (corpus rule) the file's GEO sample (GSM in the file name) has an organism other than
                        exactly "Rattus norvegicus"; reason "GEO organism: ...".
  drop_nonrat           (held-out safeguard, stricter than the corpus plan, which had no rule for a file without
                        a GSM) the file's own feature names are annotated with a mouse, human or mixed genome; or
                        the file has no GSM, its series holds non-rat samples and its features do not show rat.
  drop_not_raw_counts   Stage 2 "reverse-normalised" a non-integer source matrix. The v2 corpus holds no such
                        file (audit Part 1: the path fired in 4 studies and none of their files passed QC), so a
                        held-out file of that kind is not data "of the same kind as the corpus".
  drop_duplicate        (corpus rule) same QC'd content as an earlier kept file (cells, genes and counts equal:
                        the corpus compared the QC file's md5; here the count matrix and its labels are hashed,
                        because a QC file now also records its source path); or (corpus rule, tmp/data_audit/
                        audit.py `barcodes` and `pairs`) shares at least half of the smaller file's cell barcodes
                        with an earlier kept file of the same series (any series when cell counts are within 5%)
                        and at least 90% of the shared cells have the same total count.
  drop_duplicate        (held-out safeguard) as the barcode rule, when totals are not identical but their log
                        values correlate at 0.98 or more: the same cells in another processing.
  keep

Not a rule, recorded only: spatial_assay marks a kept file whose GEO sample record names a spatial assay (Visium
and the like: spots, not single cells). Stage 2 has no rule for them and the corpus holds such files too, so they
stay in the held-out set; 07_tables.py gives the ratio without them as context.

Tissue, sex and strain per file are resolved by the corpus inventory's own code (analysis/annotation_recovery.py:
GEO sample characteristics, then file name, then the LLM's study-level tissue). GEO's free-text tissue strings
are then grouped by keyword (common.tissue_group: the sample's GEO tissue first, the LLM's study-level tissues
as fallback) for the per-tissue tables.

Writes
  data/audit_remediation/heldout/qc_reports/preprocessing_report_heldout.json   the per-series reports, joined
  record/step4_heldout/file_plan.tsv        one row per discovered file
  record/step4_heldout/exclusions.tsv       every exclusion and note, series level (from 01) and file level
  record/step4_heldout/qc_by_series.tsv     files and cells at each QC step, per series
  record/step4_heldout/qc_by_tissue.tsv     kept series, files and cells per tissue
  record/step4_heldout/duplicate_pairs.tsv  every file pair the barcode rule examined
Usage: 03_file_plan.py
"""
import hashlib
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C   # noqa: E402

sys.path.insert(0, str(C.ROOT / "analysis"))
CORE = re.compile(r"[ACGT]{12,}")


def natural(sample_id):
    study, _, i = sample_id.rpartition("_sample")
    return (study, int(i))


def qc_fingerprint(path):
    """(content hash, barcode cores, {core: total count}) of one QC h5ad, read with h5py."""
    with h5py.File(path, "r") as h:
        X = h["X"]
        md = hashlib.md5()
        for k in ("data", "indices", "indptr"):
            md.update(np.ascontiguousarray(X[k][:]).tobytes())
        obs, var = h["obs"], h["var"]

        def index(g):
            key = g.attrs.get("_index", "_index")
            key = key.decode() if isinstance(key, bytes) else key
            return [x.decode() if isinstance(x, bytes) else str(x) for x in g[key][:]]
        names, genes = index(obs), index(var)
        md.update("\n".join(names).encode()); md.update("\n".join(genes).encode())
        lib = np.asarray(obs["library_size"][:], dtype=float)
    cores = [(m.group(0) if (m := CORE.search(n)) else n) for n in names]
    return md.hexdigest(), dict(zip(cores, lib)), len(names)


def pair_verdict(a, b):
    """The corpus audit's test for 'same cells': a, b = {barcode core: total count}."""
    shared = sorted(set(a) & set(b))
    if not shared:
        return 0, float("nan"), float("nan"), "different_cells"
    x = np.array([a[k] for k in shared]); y = np.array([b[k] for k in shared])
    eq = float(np.mean(np.abs(x - y) < 1e-6))
    r = float(np.corrcoef(np.log1p(x), np.log1p(y))[0, 1]) if len(shared) > 2 and x.std() > 0 and y.std() > 0 \
        else float("nan")
    if eq >= 0.9:
        v = "same_cells"
    elif r >= 0.98:
        v = "same_cells_other_processing"
    elif eq >= 0.1:
        v = "partly_same"
    else:
        v = "different_cells"
    return len(shared), eq, r, v


def main():
    series = sorted(r["accession"] for r in C.read_tsv(C.REC / "series_list.tsv") if r["heldout_series"] == "True")
    titles = {r["accession"]: r["title"] for r in C.read_tsv(C.REC / "series_list.tsv")}
    llm_tissues = {r["accession"]: r["llm_tissues"] for r in C.read_tsv(C.REC / "series_list.tsv")}
    reports = []
    for a in series:
        p = C.QC_REPORTS / f"report_{a}.json"
        if not p.is_file():
            sys.exit(f"missing QC report for {a}: run 02_cell_qc.py first")
        reports.append(C.load_json(p))
    joined = C.out_path(C.QC_REPORTS / "preprocessing_report_heldout.json")
    with open(joined, "w") as f:
        json.dump({"stage": "stage2_step2_heldout", "studies": reports}, f, indent=1, default=str)

    # --- tissue / sex / strain: the corpus inventory's code, on the held-out reports ----------------------
    import annotation_recovery as AR
    samples = AR.load_preprocessing_report(joined)
    geo_meta = AR.load_geo_sample_metadata([C.GEO], set(series))
    llm = AR.load_llm_analysis(C.CATALOG / "llm_study_analysis.json")
    inv = {r["sample_id"]: r for r in AR.build_annotation_inventory(samples, geo_meta, llm, {},
                                                                    AR.load_tissue_map(AR.TISSUE_MAP_CACHE))}
    soft = {a: C.parse_soft(a) for a in series}

    rows = []
    for rep in reports:
        a = rep["accession"]
        s = soft[a] or {"organisms": {}, "sample_title": {}}
        series_nonrat = any(not C.sample_is_rat(v) for v in s["organisms"].values())
        series_tissues = Counter((geo_meta.get(a, {}).get(g, {}) or {}).get("tissue") for g in s.get("gsm", []))
        for sm in sorted(rep.get("samples", []), key=lambda x: natural(x["sample_id"])):
            sid, src = sm["sample_id"], sm.get("source", "")
            m = C.GSM_RE.search(os.path.basename(src))
            gsm = m.group(1) if m else ""
            org = s["organisms"].get(gsm, []) if gsm else []
            try:
                flabel, fcounts = C.feature_species(*C.source_feature_names(src, sm.get("format")))
            except Exception as e:                                    # unreadable features: say so, do not guess
                flabel, fcounts = f"unreadable ({type(e).__name__})", {}
            iv = inv.get(sid, {})
            tissue = iv.get("tissue_resolved") or ""
            if not gsm and not iv.get("filename_tissue") and len([t for t in series_tissues if t]) == 1:
                tissue = next(t for t in series_tissues if t)         # a series-level file of a one-tissue series
            tnorm = (iv.get("tissue_normalized") or "") if tissue == (iv.get("tissue_resolved") or "") else tissue.lower()
            geo_tissue = (geo_meta.get(a, {}).get(gsm, {}) or {}).get("tissue") or "" if gsm else \
                (next(t for t in series_tissues if t) if len([t for t in series_tissues if t]) == 1 else "")
            rows.append({
                "sample_id": sid, "series": a, "source_file": os.path.basename(src), "format": sm.get("format", ""),
                "qc_status": sm.get("status", ""), "gsm": gsm, "geo_organism": "; ".join(org),
                "geo_title": s["sample_title"].get(gsm, ""), "series_has_nonrat_samples": series_nonrat,
                "geo_library_strategy": s.get("library_strategy", {}).get(gsm, ""),
                "spatial_assay": bool(s.get("spatial", {}).get(gsm, False)),
                "features_annotation": flabel,
                "features_counts": "; ".join(f"{k}:{v}" for k, v in sorted(fcounts.items())),
                "gene_id_format": sm.get("gene_id_format", ""),
                "genes_in": sm.get("mapping_stats", {}).get("total_input_genes", ""),
                "genes_mapped": sm.get("genes_mapped", ""),
                "mapping_rate": round(sm.get("mapping_stats", {}).get("mapping_rate", 0.0), 4) if sm.get("mapping_stats") else "",
                "cells_raw": sm.get("cells_raw", ""),
                "cells_after_droplet_filter": sm.get("cells_after_droplet_filter", ""),
                "cells_after_normal_filter": sm.get("cells_after_normal_filter", ""),
                "cells_after_gene_number_filter": sm.get("cells_after_gene_number_filter", ""),
                "cells_after_min_genes": sm.get("cells_after_min_genes", ""),
                "cells_qc": sm.get("cells_final", 0) if sm.get("status") == "success" else 0,
                "reverse_normalized": f"{src}: reversed normalization" in rep.get("warnings", []),
                "source_md5": (sm.get("source_file") or {}).get("md5", ""),
                "tissue_resolved": tissue, "tissue_normalized": tnorm or "unknown",
                "tissue_group": C.tissue_group(geo_tissue, llm_tissues.get(a, "")),
                "motrpac_tissue_match": iv.get("motrpac_tissue_match") or "",
                "sex_resolved": iv.get("sex_resolved") or "", "strain_resolved": iv.get("strain_resolved") or "",
                "error": sm.get("error", ""),
            })

    # --- rules, in order -----------------------------------------------------------------------------------
    fp, pairs, kept = {}, [], []
    for r in sorted(rows, key=lambda x: natural(x["sample_id"])):
        r["action"], r["reason"] = "keep", ""
        if r["qc_status"] != "success":
            r["action"], r["reason"] = "no_cells", f"Stage 2: {r['qc_status']}" + (f" ({r['error']})" if r["error"] else "")
        elif r["gsm"] and r["geo_organism"] != C.RAT:
            r["action"], r["reason"] = "drop_nonrat", f"GEO organism: {r['geo_organism'] or 'not in the SOFT file'}"
        elif r["features_annotation"] in ("mouse", "human", "mixed"):
            r["action"], r["reason"] = "drop_nonrat", (f"feature names are annotated with a "
                                                       f"{r['features_annotation']} genome (held-out safeguard)")
        elif not r["gsm"] and r["series_has_nonrat_samples"] and r["features_annotation"] != "rat":
            r["action"], r["reason"] = "drop_nonrat", ("no GSM in the file name, the series holds non-rat samples "
                                                       "and the feature names do not show a rat genome "
                                                       "(held-out safeguard)")
        elif r["reverse_normalized"]:
            r["action"], r["reason"] = "drop_not_raw_counts", ("source matrix is not integer counts; Stage 2 "
                                                               "reverse-normalised it (no such file is in v2)")
        if r["action"] != "keep":
            continue
        fp[r["sample_id"]] = qc_fingerprint(C.QC_DIR / f"{r['sample_id']}.h5ad")
        h, bc, n = fp[r["sample_id"]]
        r["qc_content_md5"] = h
        for k in kept:                                     # earlier kept files, in order
            hk, bk, nk = fp[k["sample_id"]]
            if hk == h:
                r["action"], r["reason"] = "drop_duplicate", f"identical QC content to {k['sample_id']}"
                break
            same_series = k["series"] == r["series"]
            if not same_series and abs(n - nk) > 0.05 * max(n, nk):
                continue
            ov = len(set(bc) & set(bk)) / max(1, min(len(bc), len(bk)))
            if ov < 0.5:
                continue
            shared, eq, corr, verdict = pair_verdict(bk, bc)
            pairs.append({"file_a": k["sample_id"], "file_b": r["sample_id"], "same_series": same_series,
                          "cells_a": nk, "cells_b": n, "overlap_of_smaller": round(ov, 3), "shared": shared,
                          "identical_total_share": round(eq, 3), "log_total_correlation": round(corr, 4),
                          "verdict": verdict})
            if verdict == "same_cells":
                r["action"], r["reason"] = "drop_duplicate", f"same cells as {k['sample_id']}, other file"
                break
            if verdict in ("same_cells_other_processing", "partly_same"):
                r["action"], r["reason"] = "drop_duplicate", (f"same cells as {k['sample_id']} in another processing "
                                                              f"(identical totals {eq:.2f}, log-total r {corr:.3f}; "
                                                              f"held-out safeguard)")
                break
        if r["action"] == "keep":
            kept.append(r)

    fields = ["sample_id", "series", "action", "reason", "cells_qc", "gsm", "geo_organism", "geo_title",
              "geo_library_strategy", "spatial_assay", "source_file",
              "format", "qc_status", "features_annotation", "series_has_nonrat_samples", "reverse_normalized",
              "tissue_group", "tissue_resolved", "tissue_normalized", "motrpac_tissue_match", "sex_resolved",
              "strain_resolved", "gene_id_format", "genes_in", "genes_mapped", "mapping_rate", "cells_raw", "cells_after_droplet_filter",
              "cells_after_normal_filter", "cells_after_gene_number_filter", "cells_after_min_genes", "source_md5",
              "qc_content_md5", "features_counts", "error"]
    C.write_tsv(C.REC / "file_plan.tsv", rows, fields)
    C.write_tsv(C.REC / "duplicate_pairs.tsv", pairs,
                ["file_a", "file_b", "same_series", "cells_a", "cells_b", "overlap_of_smaller", "shared",
                 "identical_total_share", "log_total_correlation", "verdict"])

    ex = [dict(r) for r in C.read_tsv(C.REC / "series_exclusions.tsv")]
    for r in rows:
        if r["action"] not in ("keep",):
            ex.append({"level": "file", "accession": r["series"], "file": f"{r['sample_id']} ({r['source_file']})",
                       "kind": "exclusion" if r["action"].startswith("drop") else "no QC output",
                       "rule": r["action"], "detail": f"{r['reason']}; cells passing QC: {r['cells_qc']}"})
    C.write_tsv(C.REC / "exclusions.tsv", ex, ["level", "accession", "file", "kind", "rule", "detail"])

    def num(x):
        return int(x) if str(x).strip() not in ("", "None") else 0
    by = defaultdict(list)
    for r in rows:
        by[r["series"]].append(r)
    srows = []
    for a in series:
        rs = by.get(a, [])
        act = Counter(r["action"] for r in rs)
        k = [r for r in rs if r["action"] == "keep"]
        srows.append({
            "series": a, "title": titles.get(a, ""), "stage2_status": next(x["status"] for x in reports if x["accession"] == a),
            "files_discovered": len(rs), "files_qc_success": sum(r["qc_status"] == "success" for r in rs),
            "files_failed": C.counter_str(r["qc_status"] for r in rs if r["qc_status"] != "success"),
            "barcodes_read": sum(num(r["cells_raw"]) for r in rs),
            "cells_after_normal_filter": sum(num(r["cells_after_normal_filter"]) for r in rs),
            "cells_after_gene_number_filter": sum(num(r["cells_after_gene_number_filter"]) for r in rs),
            "cells_passing_qc": sum(num(r["cells_qc"]) for r in rs),
            "files_drop_nonrat": act["drop_nonrat"], "cells_drop_nonrat": sum(num(r["cells_qc"]) for r in rs if r["action"] == "drop_nonrat"),
            "files_drop_not_raw_counts": act["drop_not_raw_counts"],
            "cells_drop_not_raw_counts": sum(num(r["cells_qc"]) for r in rs if r["action"] == "drop_not_raw_counts"),
            "files_drop_duplicate": act["drop_duplicate"], "cells_drop_duplicate": sum(num(r["cells_qc"]) for r in rs if r["action"] == "drop_duplicate"),
            "files_kept": len(k), "samples_kept_gsm": len({r["gsm"] for r in k if r["gsm"]}),
            "files_kept_spatial_assay": sum(r["spatial_assay"] for r in k),
            "cells_kept_spatial_assay": sum(num(r["cells_qc"]) for r in k if r["spatial_assay"]),
            "cells_kept": sum(num(r["cells_qc"]) for r in k),
            "tissue_groups_kept": C.counter_str(r["tissue_group"] for r in k),
            "tissues_kept": C.counter_str(r["tissue_normalized"] for r in k),
            "in_heldout_set": bool(k),
        })
    C.write_tsv(C.REC / "qc_by_series.tsv", srows)
    trows = []
    for t in sorted({r["tissue_group"] for r in kept}):
        k = [r for r in kept if r["tissue_group"] == t]
        trows.append({"tissue_group": t, "series": len({r["series"] for r in k}), "files": len(k),
                      "cells_kept": sum(num(r["cells_qc"]) for r in k),
                      "series_list": ",".join(sorted({r["series"] for r in k})),
                      "geo_tissue_strings": C.counter_str(r["tissue_normalized"] for r in k),
                      "motrpac_tissue_match": C.counter_str(r["motrpac_tissue_match"] or "none" for r in k)})
    C.write_tsv(C.REC / "qc_by_tissue.tsv", trows)

    tot = {"series_entering_qc": len(series), "series_with_qc_cells": sum(s["cells_passing_qc"] > 0 for s in srows),
           "series_kept": sum(s["in_heldout_set"] for s in srows), "files_discovered": len(rows),
           "files_qc_success": sum(r["qc_status"] == "success" for r in rows),
           "cells_passing_qc": sum(num(r["cells_qc"]) for r in rows), "files_kept": len(kept),
           "cells_kept": sum(num(r["cells_qc"]) for r in kept), "actions": dict(Counter(r["action"] for r in rows)),
           "cells_by_action": {a: sum(num(r["cells_qc"]) for r in rows if r["action"] == a)
                               for a in sorted({r["action"] for r in rows})},
           "tissues_kept": len(trows),
           "files_kept_spatial_assay": sum(r["spatial_assay"] for r in kept),
           "cells_kept_spatial_assay": sum(num(r["cells_qc"]) for r in kept if r["spatial_assay"])}
    with open(C.out_path(C.REC / "qc_totals.json"), "w") as f:
        json.dump(tot, f, indent=1)
    print(json.dumps(tot, indent=1))


if __name__ == "__main__":
    main()
