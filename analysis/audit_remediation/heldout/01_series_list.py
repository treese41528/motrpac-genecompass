#!/usr/bin/env python
"""01_series_list.py -- the series lost by the catalog merge, and the registered series-level exclusions.

Implements STEP5_PREREG.md M2 ("the 79 single-cell series that never entered the corpus", "Check before use")
and the "Exclusions before any loss is computed" paragraph of STEP4_PREREG.md V3. Read-only on the catalog and
the training tree; reads no cell and computes no loss.

Step 1 (audit finding 1.1). data/catalog/master_catalog.json lists a GEO series once per query that returned
it. A series returned by both the single-cell and the bulk query appears twice; the merge kept the bulk row, so
data/catalog/unified_studies.json types it "bulk" and Stage 2 step 2 never read it
(preprocess_training_matrices.py, load_study_catalog). For each such series this script records the LLM's
verdict (unified_studies.json, the fields Stage 2 reads; llm_study_analysis.json for cross-check) and whether
the Stage 2 step 1 scan found a usable MTX, H5 or h5ad matrix (data/training/gene_universe/study_coverage.tsv,
the audit's count), plus what the Stage 2 file discovery finds in the series folder today.

Step 2. A candidate series is removed if any of its GEO samples is a sample of a series in the corpus plan
(config/corpus_v2_plan.tsv; all 89 series, kept or dropped), or if it is a SuperSeries or SubSeries of one
(relations from the harvested SOFT files, read from both sides).

Writes (docs/audit/remediation/record/step4_heldout/):
  series_list.tsv         one row per series returned by both queries, with its status at every step
  series_exclusions.tsv   one row per exclusion reason or note
  series_counts.tsv       the counts at each step
Usage: 01_series_list.py
"""
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C   # noqa: E402

sys.path.insert(0, str(C.ROOT / "pipeline/02_gene_universe"))
sys.path.insert(0, str(C.ROOT / "lib"))


def main():
    master = C.load_json(C.CATALOG / "master_catalog.json")["studies"]
    unified = {s["accession"]: s for s in C.load_json(C.CATALOG / "unified_studies.json")["studies"]}
    llm_raw = {a["accession"]: a for a in C.load_json(C.CATALOG / "llm_study_analysis.json")["analyses"]}
    plan = C.read_tsv(C.PLAN)
    train_series = {r["study"] for r in plan}

    # --- returned by both queries -------------------------------------------------------------------------
    types = {}
    for i, s in enumerate(master):
        types.setdefault(s["accession"], []).append((i, s.get("source"), s.get("data_type")))
    both = sorted(a for a, t in types.items() if {x[2] for x in t} == {"single_cell", "bulk"})
    n_sc_hits = sum(1 for s in master if s.get("source") == "geo" and s.get("data_type") == "single_cell")

    import pandas as pd
    cov = pd.read_csv(C.COVERAGE, sep="\t")
    cov["usable"] = cov["usable"].astype(str) == "True"

    import preprocess_training_matrices as P
    from gene_utils import load_config
    cfg = load_config()

    soft = {a: C.parse_soft(a) for a in sorted(set(both) | train_series)}
    train_gsm = C.training_gsms(plan)

    rows = []
    for a in both:
        u = unified[a]
        cat, llm = u.get("catalog", {}), u.get("llm", {})
        raw = llm_raw.get(a, {})
        is_rat = llm.get("motrpac_utility", {}).get("is_rat")
        is_sc = llm.get("study_type", {}).get("is_single_cell")
        c = cov[cov.accession == a]
        cu = c[c.usable]
        cu3 = cu[cu.format.isin(["mtx", "h5", "h5ad"])]
        s = soft.get(a)
        sdir = C.GEO / a
        found = []
        if sdir.is_dir():
            found = P.number_matrix_files(a, P.find_matrix_files(str(sdir)), cfg)
        orgs = Counter("+".join(v) for v in (s or {}).get("organisms", {}).values())
        if is_sc is not True or is_rat is not True:
            status = "llm_not_rat_single_cell"
        elif c.empty:
            status = "no_matrix_file_in_step1_scan"
        elif cu.empty:
            status = "no_file_passed_step1_match_gate"
        elif cu3.empty:
            status = "usable_tabular_only"
        else:
            status = "candidate"
        rows.append({
            "accession": a,
            "title": cat.get("title", ""),
            "catalog_data_type_kept": cat.get("data_type", ""),
            "master_catalog_rows": ",".join(f"{t[2]}@{t[0]}" for t in types[a]),
            "organism_catalog": "; ".join(o for o in cat.get("raw_organisms", []) if "rattus" in str(o).lower()
                                          or "musculus" in str(o).lower() or "sapiens" in str(o).lower()),
            "geo_sample_organisms": "; ".join(f"{k}:{v}" for k, v in sorted(orgs.items())),
            "n_samples_catalog": cat.get("sample_count", ""),
            "n_gsm_soft": len(s["gsm"]) if s else "",
            "llm_model": llm.get("model", ""),
            "llm_is_rat": is_rat,
            "llm_is_single_cell": is_sc,
            "llm_species": llm.get("validated_organism", {}).get("species", ""),
            "llm_genecompass_useful": llm.get("motrpac_utility", {}).get("genecompass_useful"),
            "llm_data_type": raw.get("study_type", {}).get("data_type", ""),
            "llm_is_single_cell_raw_file": raw.get("study_type", {}).get("is_single_cell"),
            "llm_is_rat_raw_file": raw.get("utility_for_motrpac", {}).get("is_rat"),
            "llm_tissues": "; ".join(t.get("name", "") for t in llm.get("validated_tissues", []) if isinstance(t, dict)),
            "llm_motrpac_tissues": "; ".join(map(str, llm.get("motrpac_utility", {}).get("motrpac_tissues", []))),
            "llm_summary": llm.get("summary", ""),
            "step1_scan_file_rows": len(c),
            "step1_usable_files_by_format": C.counter_str(cu.format),
            "step1_usable_mtx_h5_h5ad": len(cu3),
            "stage2_discovery_files_by_format": C.counter_str(f[2] for f in found),
            "stage2_discovery_n_files": len(found),
            "status": status,
        })

    cand = [r["accession"] for r in rows if r["status"] == "candidate"]
    excluded, notes = C.series_exclusions(cand, soft, train_series, train_gsm)
    exrows = []
    for r in rows:
        a = r["accession"]
        r["in_corpus_plan"] = a in train_series
        r["series_excluded"] = a in excluded
        r["series_exclusion_reason"] = " | ".join(f"{x['rule']}: {x['detail']}" for x in excluded.get(a, []))
        r["series_note"] = " | ".join(f"{x['rule']}: {x['detail']}" for x in notes.get(a, []))
        r["heldout_series"] = r["status"] == "candidate" and a not in excluded
        for x in excluded.get(a, []):
            exrows.append({"level": "series", "accession": a, "file": "", "kind": "exclusion", **x})
        for x in notes.get(a, []):
            exrows.append({"level": "series", "accession": a, "file": "", "kind": "note", **x})
    # relations among the remaining candidates (the same cells could enter the held-out set twice)
    keep = {r["accession"] for r in rows if r["heldout_series"]}
    for a in sorted(keep):
        for kind, other in (soft.get(a) or {}).get("relations", []):
            if other in keep:
                exrows.append({"level": "series", "accession": a, "file": "", "kind": "note",
                               "rule": "related_to_another_heldout_series", "detail": f"{kind} {other}"})
    gsm_owner = {}
    for a in sorted(keep):
        for g in (soft.get(a) or {}).get("gsm", []):
            gsm_owner.setdefault(g, []).append(a)
    shared = {g: v for g, v in gsm_owner.items() if len(v) > 1}
    for g, v in sorted(shared.items()):
        exrows.append({"level": "series", "accession": ",".join(v), "file": "", "kind": "note",
                       "rule": "gsm_in_two_heldout_series", "detail": g})

    C.write_tsv(C.REC / "series_list.tsv", rows)
    C.write_tsv(C.REC / "series_exclusions.tsv", exrows, ["level", "accession", "file", "kind", "rule", "detail"])

    st = Counter(r["status"] for r in rows)
    n_llm = sum(1 for r in rows if r["llm_is_single_cell"] is True and r["llm_is_rat"] is True)
    counts = [
        ("GEO series in the single-cell query (master_catalog.json)", n_sc_hits),
        ("returned by both the single-cell and the bulk query", len(both)),
        ("  of those, typed 'bulk' in unified_studies.json", sum(r["catalog_data_type_kept"] == "bulk" for r in rows)),
        ("  of those, in config/corpus_v2_plan.tsv", sum(r["in_corpus_plan"] for r in rows)),
        ("LLM verdict rat and single-cell", n_llm),
        ("  with a matrix file row in the Stage 2 step 1 scan", n_llm - st["no_matrix_file_in_step1_scan"]),
        ("  with a file that passed the step 1 match gate", st["candidate"] + st["usable_tabular_only"]),
        ("  with a usable MTX, H5 or h5ad matrix (the candidates)", st["candidate"]),
        ("  usable tabular file only (not in the registered set)", st["usable_tabular_only"]),
        ("candidates excluded: shares a GEO sample with a training series",
         sum(any(x["rule"] == "shares_gsm_with_training" for x in excluded[a]) for a in excluded)),
        ("candidates excluded: SuperSeries or SubSeries of a training series",
         sum(any(x["rule"].endswith("series_of_training") for x in excluded[a]) for a in excluded)),
        ("candidates excluded, total", len(excluded)),
        ("held-out series entering cell QC", len(keep)),
        ("training-side sample accessions compared", len(train_gsm)),
        ("series in the corpus plan compared", len(train_series)),
        ("corpus-plan series with no SOFT file on disk", sum(soft.get(t) is None for t in train_series)),
    ]
    C.write_tsv(C.REC / "series_counts.tsv", [{"step": k, "n": v} for k, v in counts])
    for k, v in counts:
        print(f"{v:>6}  {k}")
    print("excluded:", json.dumps({a: [x["rule"] for x in v] for a, v in excluded.items()}))
    print("notes:", json.dumps({a: [x["rule"] for x in v] for a, v in notes.items()}))
    print("held-out series:", " ".join(sorted(keep)))


if __name__ == "__main__":
    main()
