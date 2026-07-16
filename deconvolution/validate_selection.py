#!/usr/bin/env python3
"""validate_selection.py -- the pre-deconvolution GUARD.

For every tissue in `deconvolution/tissue_references.yaml`, this re-resolves the sample selection from the
config's filters and ASSERTS that it picks EXACTLY what we specify -- before any reference is built or any
bulk is deconvolved. It exists because `select_samples()` historically filtered on accession+tissue only,
which let a ~85%-non-rat cortex reference and a rat-mouse-chimera muscle reference reach production.

Three checks per (buildable) tissue:
  1. EXACT SELECTION   set(resolved) == set(expect.sample_ids)  (and n_samples)
  2. QC DROPS ONLY THE INTENDED  the reference_qc gate (SPATIAL/ENGINEERED/BULK) drops exactly
     expect.qc_dropped -- catches BOTH a wrong-thing drop (a good sample killed) and a missed drop.
  3. ORGANISM          every resolved sample's geo_organism == expect.organism (rat)
Plus a WARN if any selected sample's condition/title looks like a treatment/aged arm, and -- if the
reference is already built -- a POST-BUILD check that cells_meta.tsv used exactly expect.sample_ids.

Usage (project venv):
  python deconvolution/validate_selection.py                 # validate all; exit 2 on any FAIL
  python deconvolution/validate_selection.py --tissue CORTEX HEART
  python deconvolution/validate_selection.py --built-only     # only check already-built references
Exit code: 0 = all buildable tissues PASS; 2 = at least one FAIL.
"""
import argparse, csv, os, re, sys, io, contextlib
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "deconvolution"))
os.environ.setdefault("PIPELINE_ROOT", str(ROOT))
from build_reference import select_samples, INVENTORY, QC_DIR, CT_DIR, CONS_DIR  # noqa: E402

CONFIG = ROOT / "deconvolution" / "tissue_references.yaml"

# a selected sample whose condition/title matches this (and is NOT a control term) => WARN to eyeball
_TREAT = re.compile(r"(tumou?r|cancer|puromycin|nephros|reject|allograft|transplant|disease|injur|"
                    r"explos|blast|nitrofen|hypox|sugen|infarct|stroke|ischem|\bLPS\b|poly.?i.?c|atroph|"
                    r"disuse|reload|senescen|hyperox|\bIBS\b|colitis|\bDSS\b|estradiol|tau\b|transgenic|"
                    r"\bold\b|aged|\b27.?month|\b30.?month|caloric|\bCR\b|-CR\b|-O\b)", re.I)


def load_config():
    d = yaml.safe_load(open(CONFIG))
    return {k: v for k, v in d.items() if isinstance(v, dict)}


def _inventory():
    inv = {}
    with open(INVENTORY) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            inv[r["sample_id"]] = r
    return inv


def _resolve(study, tissue, cfg, keep_nonnative):
    """sorted sample_ids from select_samples with this cfg's filters; keep_nonnative bypasses the QC drop."""
    prev = os.environ.get("ALLOW_NONNATIVE_REF")
    os.environ["ALLOW_NONNATIVE_REF"] = "1" if keep_nonnative else "0"
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            ids = select_samples(study, tissue,
                                 conditions=cfg.get("conditions"), sample_ids=cfg.get("sample_ids"),
                                 organism=cfg.get("organism", "Rattus norvegicus"),
                                 title_include=cfg.get("title_include"), title_exclude=cfg.get("title_exclude"),
                                 dedup_gsm=cfg.get("dedup_gsm", True))
        return sorted(ids)
    finally:
        if prev is None:
            os.environ.pop("ALLOW_NONNATIVE_REF", None)
        else:
            os.environ["ALLOW_NONNATIVE_REF"] = prev


def _resolve_pooled(cfg, keep_nonnative):
    ids = []
    for comp in cfg.get("pooled_studies", []):
        ids += _resolve(comp["study"], comp["tissue"], comp, keep_nonnative)
    return sorted(ids)


def _missing_inputs(sample_ids):
    """samples whose h5ad / celltypes / consensus annotation is absent (build would silently drop them)."""
    miss = []
    for s in sample_ids:
        if not ((QC_DIR / f"{s}.h5ad").exists()
                and (CT_DIR / s / f"{s}_celltypes.tsv").exists()
                and (CONS_DIR / s / f"{s}_consensus.tsv").exists()):
            miss.append(s)
    return miss


def _built_samples(ref_dir):
    cm = ROOT / ref_dir / "cells_meta.tsv"
    if not cm.exists():
        return None
    with open(cm) as f:
        return sorted(set(r["sample"] for r in csv.DictReader(f, delimiter="\t")))


def check_tissue(code, cfg, inv, built_only=False):
    """Return (verdict, lines) where verdict in {PASS, FAIL, WARN, SKIP, NEEDS-DATA, NOT-BUILT}."""
    L, status = [], cfg.get("status", "")
    exp = cfg.get("expect", {}) or {}
    exp_ids = sorted(exp.get("sample_ids") or []) if exp.get("sample_ids") else None
    ref_dir = cfg.get("reference_dir")

    if status == "blocked":
        return "SKIP", [f"BLOCKED -- no rat reference exists ({cfg.get('notes','').strip()[:70]})"]
    if status == "needs-online-data" or cfg.get("data_present") is False:
        present = ref_dir and (ROOT / ref_dir / "cells_meta.tsv").exists()
        return ("NOT-BUILT" if not present else "PASS",
                [f"NEEDS-DATA -- {cfg.get('study')} not ingested; intended selection = {exp_ids}"])

    fails, warns = [], []

    # --- resolution-based checks (skipped with --built-only) ---
    if not built_only:
        try:
            if cfg.get("pooled"):
                final = _resolve_pooled(cfg, keep_nonnative=bool(cfg.get("allow_nonnative")))
                raw = _resolve_pooled(cfg, keep_nonnative=True)
            else:
                final = _resolve(cfg["study"], cfg["tissue"], cfg, keep_nonnative=bool(cfg.get("allow_nonnative")))
                raw = _resolve(cfg["study"], cfg["tissue"], cfg, keep_nonnative=True)
        except SystemExit as e:
            return "FAIL", [f"resolution ERROR: {e}"]

        qc_dropped = sorted(set(raw) - set(final))
        exp_drop = sorted(exp.get("qc_dropped") or [])

        # 1. EXACT SELECTION
        if exp_ids is not None and set(final) != set(exp_ids):
            fails.append(f"selection != expect: +{sorted(set(final)-set(exp_ids))} -{sorted(set(exp_ids)-set(final))}")
        if exp.get("n_samples") is not None and len(final) != exp["n_samples"]:
            fails.append(f"n_samples {len(final)} != expect {exp['n_samples']}")
        # 2. reference_qc DROPS ONLY THE INTENDED
        if set(qc_dropped) != set(exp_drop):
            fails.append(f"reference_qc dropped the WRONG set: got {qc_dropped}, expected {exp_drop}")
        elif qc_dropped:
            L.append(f"reference_qc dropped (intended): {qc_dropped}")
        # 3. ORGANISM
        want_org = (exp.get("organism") or "Rattus norvegicus").lower()
        bad_org = [s for s in final if inv.get(s, {}).get("geo_organism", "").strip().lower() != want_org]
        if bad_org:
            fails.append(f"non-{want_org} samples selected: {bad_org}")
        # 4. INPUTS PRESENT (h5ad + annotations) -- else the build silently drops the sample
        miss_in = _missing_inputs(final)
        if miss_in:
            fails.append(f"missing build inputs (h5ad/annotation) for: {miss_in}")
        # WARN: treatment/aged hint on a selected sample
        for s in final:
            row = inv.get(s, {})
            blob = f"{row.get('condition_resolved','')} | {row.get('geo_title','')}"
            if _TREAT.search(blob):
                warns.append(f"{s} looks like treatment/aged arm: '{blob.strip()[:60]}'")
        L.append(f"resolved {len(final)} sample(s): {final}")

    # --- post-build check (if the reference is on disk) ---
    if ref_dir:
        got = _built_samples(ref_dir)
        if got is None:
            L.append("reference NOT built yet")
            if built_only:
                return "NOT-BUILT", L
        elif exp_ids is not None and set(got) != set(exp_ids):
            fails.append(f"BUILT reference used the wrong samples: got {got}, expect {exp_ids}")
        else:
            L.append(f"built reference OK: {len(got)} sample(s) match expect")

    if fails:
        return "FAIL", [("FAIL: " + f) for f in fails] + L
    if warns:
        return "WARN", [("warn: " + w) for w in warns] + L
    return "PASS", L


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tissue", nargs="*", help="subset of BULK codes (default: all)")
    ap.add_argument("--built-only", action="store_true", help="only check already-built references (skip resolution)")
    args = ap.parse_args()

    cfg_all = load_config()
    inv = _inventory()
    tissues = args.tissue or list(cfg_all)
    order = {"FAIL": 0, "WARN": 1, "NOT-BUILT": 2, "NEEDS-DATA": 3, "PASS": 4, "SKIP": 5}
    results = {}
    for code in tissues:
        if code not in cfg_all:
            results[code] = ("FAIL", [f"not in {CONFIG.name}"]); continue
        results[code] = check_tissue(code, cfg_all[code], inv, args.built_only)

    print(f"\n=== validate_selection: {len(tissues)} tissue(s) ===")
    for code in sorted(tissues, key=lambda c: (order.get(results[c][0], 9), c)):
        v, lines = results[code]
        print(f"\n[{v:9s}] {code}")
        for ln in lines:
            print(f"    {ln}")

    nfail = sum(1 for v, _ in results.values() if v == "FAIL")
    nwarn = sum(1 for v, _ in results.values() if v == "WARN")
    npass = sum(1 for v, _ in results.values() if v == "PASS")
    print(f"\n=== {npass} PASS · {nwarn} WARN · {nfail} FAIL · "
          f"{sum(1 for v,_ in results.values() if v in ('SKIP','NEEDS-DATA','NOT-BUILT'))} skipped/pending ===")
    sys.exit(2 if nfail else 0)


if __name__ == "__main__":
    main()
