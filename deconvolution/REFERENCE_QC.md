# Deconvolution Reference QC — how to not repeat our backtrack

**TL;DR.** A Stage-8 deconvolution reference must be built from **native, adult, healthy, whole-tissue,
single-cell/single-nucleus** samples. Two production references silently violated this and cost a
multi-day backtrack. `deconvolution/reference_qc.py` now codifies the checks and is wired into
`build_reference.py`, so the whole class of bug fails loudly at build time instead of being discovered
downstream. **Before trusting any reference, run the gate.**

## The two bugs we hit (2026-07-01)

1. **Liver `liver_GSE220075` mixed ~15–24% Visium spatial spots** (`GSE220075_sample2` = Rat_B1_VIS,
   `sample11` = Rat_A1_VIS) into an snRNA reference — multi-cell 55 µm spots treated as single nuclei.
   Both `in_corpus=True`. *Impact:* real data-hygiene bug, but NEGLIGIBLE on results (liver is ~91%
   hepatocyte; the Visium spots resembled hepatocyte nuclei; cleaned hepatocyte Z was 0.998-correlated
   to the old). The parenchyma-Z red flag is therefore **mRNA-content bias, not contamination**.
2. **Lung `lung_GSE178405` was built from an in-vitro tissue-ENGINEERING study** — "Cell isolate",
   "Engineered lung d7", "Tri-culture", "Native control - P7" (postnatal-day-7 developing). NOT native
   adult lung. *Impact:* SUBSTANTIAL — the root cause of lung being the weakest deconv tissue
   (median r ~0.5, 0 exercise hotspots). Replaced with a **native pooled** reference → lung 0→3 hotspots.

Both passed the corpus `llm_deconvolution_useful` screen. Nothing enforced the "native/adult/healthy/
single-cell/whole-tissue" criteria — that's the gap this QC closes.

## Valid-reference criteria (what the gate enforces)

| requirement | why | FAIL patterns caught |
|---|---|---|
| **single-cell / single-nucleus** (not spatial, not bulk) | spots/bulk are multi-cell → wrong per-type GEP | `visium`, `_vis`, `spatial`, `slide-seq`, `bulk` |
| **native whole-tissue** (not engineered/cultured/sorted) | in-vitro/sorted expression ≠ in-vivo | `engineered`, `cell isolate`, `tri-culture`, `organoid`, `cultured`, `in-vitro`, `sorted`, `CD\d+\+`, `FACS`, `day N` |
| **adult** (not embryonic/postnatal) | developmental expression ≠ adult MoTrPAC bulk | `E\d+`, `P\d+`, `embryo`, `fetal`, `neonat` (→ WARN) |
| **single coherent modality** (no hidden mix) | a bimodal depth distribution = a mixed modality | per-sample median genes/cell ratio > 2.5 (`--deep`) |

## The gate — `deconvolution/reference_qc.py`

```bash
# audit every built reference (metadata scan)
python deconvolution/reference_qc.py --all
# one reference, incl. the per-sample depth modality-mix check
python deconvolution/reference_qc.py --ref-dir data/deconvolution/references/lung_native_pooled --deep
# CI/build gating: non-zero exit on any FAIL-class violation
python deconvolution/reference_qc.py --all --fail
```
It reads each reference's `cells_meta.tsv` (sample column) → the study inventory
(`reports/annotations/annotation_inventory.tsv`) `geo_title`/`geo_source_name`, plus (with `--deep`)
the per-sample expression depth. Validated: it FAILs `lung_GSE178405` (all engineered) and the old
Visium liver (both title + depth), PASSes all corrected + native refs, no false positives.

**Wired into `build_reference.py`:** `select_samples` now auto-DROPS FAIL-class samples (spatial/
engineered/sorted/bulk) with a loud log line, and WARNs on developmental — so a future
`build_reference.py --study GSE220075 --tissue liver` excludes the Visium spots by default (no
`--sample-ids` needed), and an all-engineered study errors out ("pick a native study"). Override with
`ALLOW_NONNATIVE_REF=1`.

## Known-bad rat lung studies (do NOT use as an adult native reference)
- **GSE178405** — engineered/in-vitro (the old lung ref; superseded).
- **GSE196313** — embryonic E21.5 (also our old lung *validation source* — that cross test was
  developmental-mismatched; TODO: replace with a native cross source).
- **GSE312833** — CD31+ SORTED endothelial only (not whole-lung).

## The native lung solution — `references/lung_native_pooled`
No single clean adult rat lung atlas exists in the corpus, so `deconvolution/build_lung_pooled.py`
pools the healthy control arms of 3 studies: **GSE273062** (VeNx ×2) + **GSE252844** (C3) +
**GSE242310** (NOX), via `build_reference.load_study` + cross-study outer gene join + the `lung`
label-scheme (merges Club/Clara/ciliated/NK synonyms; immune kept resolved). 50,643 cells, 34 native
cell types. `build_references_v2.sh` now builds this instead of GSE178405. Caveats: endothelial from
GSE273062 only (591 cells); Naive-B ~14% (disease-control-arm immune infiltration) — watch on validation.
GSE247625 (2 more healthy) deferred — quantified to mouse mm10, needs re-alignment.

## Adding a new tissue reference — checklist
1. Pick a **native adult healthy whole-tissue single-cell/nucleus** study (verify on GEO, not just the
   inventory `llm_*` flags).
2. `build_reference.py --study … --tissue …` (the QC gate auto-drops contamination + warns).
3. `reference_qc.py --ref-dir … --deep` → must be PASS (or understood WARN).
4. Validate accuracy on a known-truth holdout/cross (the omnideconv panel) before adopting to production.
