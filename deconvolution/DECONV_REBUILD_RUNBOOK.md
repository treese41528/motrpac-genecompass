# Deconvolution rebuild runbook — config-driven, validated, from start (2026-07-15)

The Stage-8 deconvolution reference layer was **rebuilt from scratch** to the reference decisions in
`REFERENCE_SELECTION_PLAN.md` (Geyu's note), with a hard guard that every reference is built from — and validated
against — an exact, locked sample selection *before* any bulk is deconvolved. This document is the reproducible
runbook + the verification record. It exists because the old build path let a ~85%-non-rat cortex reference and
a rat-mouse-chimera muscle reference reach production; that class of bug is now structurally impossible.

## 1. Data flow

```
tissue_references.yaml (v3)  ──►  validate_selection.py  ──►  build_references_from_config.py  ──►  references_v3/
   (spec + locked expect)         (GATE 1: exact select)       (build_reference.py / build_lung_pooled.py)
                                                                          │
   run_deconv_all.py  ◄──────────────────────────────────────────────────┘
   GATE 1 (validate_selection) + GATE 2 (reference_qc) ──►  run_stage8 (BayesPrism deconv) ──► run_stage9 (GC embed)
                                                        ──►  Stage 10 (pseudobulk DE) / 12 (transfer) / 13-14
```

## 2. Files (all in `deconvolution/` unless noted)

| File | Role |
|---|---|
| `tissue_references.yaml` | **Single source of truth** (schema v3). Per-tissue selection spec + `expect:` contract. |
| `build_reference.py` | Builds one reference. Filters: `--organism` (default rat), `--title-include/--exclude`, `--conditions`, `--sample-ids`, `--no-dedup-gsm`, `--label-scheme`, `--gene-join`. |
| `build_lung_pooled.py` | Pooled native-lung build (GSE273062+GSE252844 controls; GSE242310 P10-neonate dropped). |
| `build_references_from_config.py` | Builds **every** reference from the config; shared muscle built once; post-build validates. |
| `validate_selection.py` | **The guard.** Asserts resolved selection == `expect` exactly; reference_qc drops only intended; all rat; inputs present; built `cells_meta` matches. |
| `reference_qc.py` | Native/adult/healthy/single-cell gate (SPATIAL/ENGINEERED/BULK drop; DEVEL warn). |
| `pipeline/run_deconv_all.py` | Orchestrator: GATE 1 (validate) + GATE 2 (reference_qc) → run_stage8 + run_stage9 per tissue. |

## 3. `tissue_references.yaml` v3 — per-tissue keys

`status` (ready | needs-build | needs-online-data | proposed | blocked) · `reference_dir` · `study` · `tissue`
(tissue_normalized, exact) · `organism` (default `Rattus norvegicus`) · `conditions` · `title_include` /
`title_exclude` (geo_title regex) · `sample_ids` · `dedup_gsm` · `label_scheme` · `gene_join` · `allow_nonnative`
(bypass reference_qc when a sort is intrinsic, e.g. SPLEEN) · `expect:` {`n_samples`, `sample_ids` (post-QC,
exact), `qc_dropped` (intended drops), `organism`} · `build` (reproducible command).

Note: a geo_title regex that starts with `-` (e.g. Ma's `-Y` arm) must be written `[-]Y($|\b|_)` — a leading dash
is otherwise mistaken for a CLI flag by argparse. The build driver also passes title filters as `--opt=value`.

## 4. What the guard guarantees (run BEFORE deconvolving)

For every buildable tissue, `validate_selection.py` asserts (FAIL = refuse to deconvolve):
1. **Exact selection** — resolved `select_samples()` == `expect.sample_ids` (and `n_samples`).
2. **QC drops only the intended** — the reference_qc gate removes exactly `expect.qc_dropped` (catches both a
   wrong-thing drop and a missed drop). E.g. LIVER drops exactly its 2 Visium spots.
3. **Organism** — every selected sample is `Rattus norvegicus`.
4. **Inputs present** — h5ad + celltypes + consensus annotation exist (else the build silently drops the sample).
5. **Built artifact** — if built, `cells_meta.tsv` used exactly `expect.sample_ids`.
Negative-tested: it FAILs on wrong sample_ids, a disabled organism gate, and a wrong qc_dropped expectation.

## 5. Commands

```bash
# validate the whole plan (fast; login-safe)
python deconvolution/validate_selection.py
# build all references (compute node)
python deconvolution/build_references_from_config.py --submit          # or --run on a node
# validate the pipeline end-to-end (both gates), no run
python pipeline/run_deconv_all.py --validate-only
# deconvolve everything runnable (submits one SLURM job per tissue)
python pipeline/run_deconv_all.py --submit                             # add --include-proposed for HIPPOC
```

## 6. Verification record (2026-07-15)

Selections verified at **three levels**: mechanical (config→resolution→built cells_meta), inventory raw
(geo_title/organism/arm per sample), and an **adversarial GEO-record check** (13 verifiers WebFetched each study
+ GSMs). **Result: all 13 references CONFIRMED — zero wrong-organism, wrong-region, or disease/treatment-arm
contamination; every disease/aged/treated/other-species sibling correctly excluded.**

Real catches (the check was genuine): the GSE137869 study *page* once mislabeled the muscle GSM as "Brain-M-Y"
(a page-index artifact; the per-GSM record is correctly "Muscle-M-Y"); HIPPOC's "Mmul10 macaque" line is a benign
metadata copy-paste (organism is rat; ventral-HPC confirmed via Nat Commun 2025, PMID 40500290); BAT GSE244451 has
no hidden salt/hypertension arm (4 naive samples; the 2 taPVAT siblings correctly excluded); LUNG uses only the
VeNx/C3 disease-model controls.

| Tissue | Study | GEO verdict | n | Control/young arm confirmed |
|---|---|---|---|---|
| KIDNEY | GSE240658 | CONFIRMED | 4 | "healthy control"; 12 puromycin-nephrosis siblings excluded |
| LIVER | GSE220075 | CONFIRMED | 4 | snRNA whole-liver healthy; scRNA/immune-enriched/Visium excluded |
| BLOOD | GSE285476 | CONFIRMED | 1 | "healthy control 0d"; transplant rejection/syngeneic excluded |
| CORTEX | GSE303115 | CONFIRMED | 2 | the 2 rat snRNA; 5 other species excluded |
| HEART | GSE280111 | CONFIRMED | 19 | healthy Wistar left ventricle, 17wk |
| VENACV | GSE280111 | CONFIRMED | 8 | healthy pulmonary veins (venous proxy) |
| WATSC | GSE137869 | CONFIRMED | 2 | young ad-lib `-Y`; old/CR excluded |
| SKMGN+SKMVL | GSE137869 | CONFIRMED | 2 | young ad-lib `-Y` (one shared muscle ref) |
| BAT | GSE244451 | CONFIRMED | 2 | 2 naive BAT; taPVAT siblings excluded |
| HYPOTH | GSE248413 | CONFIRMED | 1 | young "Y" (4mo); old + 17α-estradiol excluded |
| SMLINT | GSE272055 | CONFIRMED | 2 | healthy jejunum epithelium |
| LUNG | GSE273062+GSE252844 | CONFIRMED | 3 | disease-model CONTROLS; GSE242310 neonate dropped |
| HIPPOC (proposed) | GSE295314 | CONFIRMED | 6 | ventral-HPC "Fed" arm (refed-after-fast) |

**Caveats surfaced = pre-documented limitations, NOT selection errors** (all handled by differential-only framing):
strain ≠ F344 (most refs); young-adult ages (LIVER 8–10wk, SMLINT 6–8wk, HEART 17wk, HYPOTH 4mo, Ma ~5mo);
male-only (LIVER/HEART/VENACV/HYPOTH/HIPPOC); composition-incomplete (BLOOD PBMC-Ficoll, WATSC SVF-no-adipocytes,
SMLINT epithelium-only, VENACV pulmonary-vein proxy). HIPPOC "Fed" = refed-after-24h-fast, not ad-lib baseline.

## 7. Execution log

- Reference builds: SLURM `11310416` (11/13 OK) + `11310443` (WATSC+muscle, after the `[-]Y` argparse fix) → all
  13 built in `data/deconvolution/references_v3/` and post-validated (deployed `references/`+`references_v2/` untouched).
- Deconvolution: `run_deconv_all.py --submit` → see `logs/deconv_*.out`. (Job IDs appended at run time.)

## 8. Status & what remains

- **RUNNING/DONE:** deconvolution of the 13 validated tissues (KIDNEY, LIVER, BLOOD, CORTEX, HEART, VENACV, WATSC,
  SKMGN, SKMVL, BAT, HYPOTH, SMLINT, LUNG).
- **HELD — HIPPOC** (proposed): built + verified, but deconvolving it commits to the GSE305314→GSE295314 switch
  (Geyu's call). Add with `run_deconv_all.py --submit --include-proposed --tissue HIPPOC`.
- **NEEDS INGESTION:** TESTES (OMIX767 C+E7W staged), SPLEEN (GSE186158, 7-species → organism-gate), COLON
  (GSE326856, verify Epcam+). Not deconvolvable until ingested.
- **BLOCKED:** ADRNL, OVARY — no rat reference exists; reference-free CDSeq if a composition is required.
- **Downstream:** any reference change perturbs the globally-pooled IHW/repfdr fit across all 185 pseudobulk-DE
  blocks → re-run Stage 10 → 12 (and re-execute the per-stage notebooks) **together** after the deconv wave.
