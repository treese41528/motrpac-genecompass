#!/usr/bin/env python3
"""reference_qc.py -- gate a deconvolution SC reference against the failure modes that cost us a
multi-day backtrack (liver Visium contamination + engineered/developmental lung reference).

A valid single-cell/nucleus deconvolution reference for adult native MoTrPAC tissue must be:
NATIVE whole-tissue dissociation (not spatial/engineered/cultured/sorted), ADULT (not embryonic/
postnatal), single-CELL or single-NUCLEUS (not spatial spots / bulk), and a single coherent modality
(not a bimodal mix). This gate scans a built reference's samples against the study inventory
(geo_title / geo_source_name / age) AND, with --deep, the per-sample expression depth (a hidden
modality mix shows as a bimodal genes/cell distribution -- how we caught the liver Visium spots).

Usage:
  reference_qc.py --ref-dir data/deconvolution/references/lung_native_pooled [--deep] [--fail]
  reference_qc.py --all [--deep]            # scan references/ + references_v2/
Exit code: 0 = clean/warn; 2 = a FAIL-class violation (spatial/engineered/sorted/bulk) when --fail set.
"""
import argparse, csv, os, re, sys, glob

# --- FAIL classes: categorically wrong for a native single-cell reference ---
SPATIAL   = re.compile(r"(visium|_vis\b|\bvis\b|spatial|slide-?seq|geomx|cosmx)", re.I)
ENGINEERED= re.compile(r"(engineered|cell isolate|tri-?culture|co-?culture|organoid|cultured|"
                       r"\bin[- ]?vitro|ipsc|cell line|passage|\bday ?\d|\bd\d+\b|sorted|"
                       r"cd\d+\+|facs|flow-?sorted|magnetic)", re.I)
BULK      = re.compile(r"(bulk rna|whole[- ]tissue rna(?!-seq single))", re.I)
# --- WARN class: developmental (usable in principle, wrong for ADULT deconvolution) ---
DEVEL     = re.compile(r"(\bE\d+\.?\d*\b|\bP\d+\b|embryo|embryonic|fetal|foetal|neonat|postnatal)", re.I)

def load_inventory(path):
    inv = {}
    with open(path) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            inv[r["sample_id"]] = (r.get("geo_title", "") + " | " + r.get("geo_source_name", "")
                                   + " | " + r.get("geo_cell_type", "")).strip()
    return inv

def per_sample_depth(ref_dir):
    """median genes/cell per sample (for the modality-mix check). Needs scipy; returns {} on failure."""
    try:
        import scipy.io, numpy as np
        M = scipy.io.mmread(os.path.join(ref_dir, "reference_counts.mtx")).tocsr()   # cells x genes
        gpc = np.asarray((M > 0).sum(1)).ravel()
        samp = [r["sample"] for r in csv.DictReader(open(os.path.join(ref_dir, "cells_meta.tsv")),
                                                    delimiter="\t")]
        import collections
        d = collections.defaultdict(list)
        for s, g in zip(samp, gpc):
            d[s].append(g)
        return {s: float(np.median(v)) for s, v in d.items()}
    except Exception as e:
        print(f"    [depth check skipped: {e}]")
        return {}

def qc_reference(ref_dir, inv, deep=False):
    cm = os.path.join(ref_dir, "cells_meta.tsv")
    if not os.path.exists(cm):
        return {"ref": ref_dir, "fail": [f"no cells_meta.tsv"], "warn": [], "ok": False}
    samples = sorted(set(r["sample"] for r in csv.DictReader(open(cm), delimiter="\t")))
    fails, warns = [], []
    for s in samples:
        title = inv.get(s, "")
        if not title:
            warns.append(f"{s}: not in inventory (provenance unknown)"); continue
        if SPATIAL.search(title):    fails.append(f"{s}: SPATIAL/Visium -> not single cells [{title[:50]}]")
        elif ENGINEERED.search(title):fails.append(f"{s}: ENGINEERED/cultured/sorted [{title[:50]}]")
        elif BULK.search(title):     fails.append(f"{s}: BULK, not single-cell [{title[:50]}]")
        if DEVEL.search(title):      warns.append(f"{s}: DEVELOPMENTAL age (not adult) [{title[:50]}]")
    if deep:
        meds = per_sample_depth(ref_dir)
        if len(meds) >= 2:
            mn, mx = min(meds.values()), max(meds.values())
            if mn > 0 and mx / mn > 2.5:
                hi = [s for s, v in meds.items() if v > 2 * mn]
                warns.append(f"MODALITY-MIX? per-sample genes/cell {mn:.0f}..{mx:.0f} (ratio {mx/mn:.1f}); "
                             f"high-depth outliers: {hi}")
    return {"ref": ref_dir, "fail": fails, "warn": warns, "ok": not fails, "n_samples": len(samples)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-dir")
    ap.add_argument("--all", action="store_true", help="scan references/ + references_v2/")
    ap.add_argument("--inventory", default="reports/annotations/annotation_inventory.tsv")
    ap.add_argument("--deep", action="store_true", help="also run the per-sample depth modality-mix check")
    ap.add_argument("--fail", action="store_true", help="exit 2 if any reference has a FAIL-class violation")
    args = ap.parse_args()
    inv = load_inventory(args.inventory)

    refs = []
    if args.all:
        for root in ("data/deconvolution/references", "data/deconvolution/references_v2"):
            refs += sorted(d for d in glob.glob(f"{root}/*") if os.path.isdir(d))
    elif args.ref_dir:
        refs = [args.ref_dir]
    else:
        ap.error("give --ref-dir or --all")

    any_fail = False
    for d in refs:
        r = qc_reference(d, inv, deep=args.deep)
        status = "FAIL" if r["fail"] else ("WARN" if r["warn"] else "PASS")
        any_fail |= bool(r["fail"])
        print(f"[{status}] {os.path.basename(d)}  ({r.get('n_samples','?')} samples)")
        for m in r["fail"]: print(f"    FAIL {m}")
        for m in r["warn"]: print(f"    warn {m}")
    if any_fail and args.fail:
        sys.exit(2)

if __name__ == "__main__":
    main()
