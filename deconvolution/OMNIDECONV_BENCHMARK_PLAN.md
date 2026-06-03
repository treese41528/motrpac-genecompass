# omnideconv Benchmark Replication on MoTrPAC Rat Data — Execution Plan

**Status:** PLAN (not yet executed). Authored 2026-06-03.
**Trigger to execute:** Phases 0–4 can start immediately (no license needed). Phase 5
(CIBERSORTx + Bseq-SC) is gated on the Stanford CIBERSORT/CIBERSORTx license, expected
"in a few days."
**Source paper:** Dietrich, Merotto et al. (2025/2026), *"omnideconv: a unifying framework
for using and benchmarking single-cell-informed deconvolution of bulk RNA-seq data,"*
Genome Biology 27:6 / bioRxiv 2024.06.10.598226 (v. posted 2025-10-28).
Local copy: `readings/omnideconv.pdf`. See also memory `reference_omnideconv_benchmark`.

---

## 0. Goal & scope (decided)

Reproduce the **omnideconv benchmark *methodology*** — not its human/mouse datasets — on
**our rat data** (MoTrPAC bulk target + the rat single-cell corpus references +
`validation_v2` known-truth pseudobulk mixtures). Deliver, per method, both the **estimated
cell-type fractions** and the paper's **systematic-bias / confounder analyses**.

Three scoping decisions were made by the user (2026-06-03):

| Decision | Choice |
|---|---|
| **Method panel (now)** | Add **AutoGeneS + Scaden** to the already-installed Bisque/DWLS/MuSiC/SCDC/BayesPrism → the paper's exact 8-method panel **minus** the 2 license-gated methods. |
| **Bias analyses** | **All four**: mRNA-content bias (Fig 4), spillover (Fig 5A/B), granularity/aggregation (Fig 3), unknown content (Fig 5C / S7). |
| **Tissue scope** | **All 11** `validation_v2` holdout/cross sets. |

License-gated, deferred to **Phase 5**: **CIBERSORTx** (Docker `fractions` image + token)
and **Bseq-SC** (needs the registration-gated `CIBERSORT.R` source). Both come from the
same Stanford registration.

Out of committed scope (noted as optional stretch in §11): Fig 6 reference-variability
sweep (we have ~1 reference per tissue), and the non-benchmark methods CPM/MOMF/CDSeq
(heavy `terra`/`rgl`/Gibbs deps, never in the paper's panel).

---

## 1. What "the omnideconv run" actually is (paper distillation)

The paper supports **12** methods but **benchmarks 8** — those that (1) use annotated
scRNA-seq directly, (2) need no marker genes, (3) return relative fractions. Our panel:

| Method | Lang | sc input | bulk input | Key non-default params (paper) | Install status |
|---|---|---|---|---|---|
| **AutoGeneS** 1.0.4 | Py | CPM | TPM | `ngen=5000`, `max_iter=1e6`, HVG off (use all genes) | **INSTALL (Phase 0)** |
| **BayesPrism** 2.x | R | counts | counts | cell *types* only (not states); **no** final Gibbs update when sc/bulk are matched assays; `species` set | ✅ (Danko-Lab 2.2.3 in `R_libs/`) |
| **Bisque** 1.0.5 | R | counts | counts | defaults (Bisque internally CPMs) | ✅ `BisqueRNA` |
| **DWLS** 0.1 (opt.) | R | counts | counts¹ | `dwls_method="mast_optimized"` (paper default), `pval_cutoff=0.05` | ✅ `DWLS` |
| **MuSiC** 0.3.0 | R | counts | TPM | `batch_ids` = subject/sample | ✅ `MuSiC` |
| **Scaden** 1.1.2 | Py | counts | TPM | 5000 steps, batch 128, lr 1e-4, 1000 sims × 100 cells | **INSTALL (Phase 0)** |
| **SCDC** 0.0.0.9 | R | counts | TPM | single reference (defaults) | ✅ `SCDC` |
| CIBERSORTx 1.0 | Docker | CPM | TPM | no batch correction (S/B-mode off) | **Phase 5 (license)** |
| Bseq-SC | R | — | — | needs `CIBERSORT.R` | **Phase 5 (license)** |

¹ DWLS bulk normalization is omnideconv-internal; we pass counts and let the signature
build (MAST) handle it, matching our validated WAT run.

**Metrics (paper Methods "Performance metrics"):** for ground-truth `x` vs estimate `y`
over C cell types and S samples — **Pearson R** and **RMSE**, computed both *globally*
(all type×sample points pooled) and *per cell type*; plus **MAE** and **MAPE** per type.
RMSE+R together expose **estimation bias** (systematic over/under-estimation). Visuals:
Fig 2A pred-vs-true scatter; Fig 2B radar of `log(1/RMSE)` and R per cell type.

**Simulation engine:** **SimBu v1.8**. mRNA-content bias is modeled by scaling each cell's
expression by a factor derived from its **number of expressed genes** (the SimBu "silver
standard"); turning this off = `scaling_factor = NONE`.

---

## 2. Mapping the paper onto our infrastructure

| Paper component | Our analog | Reuse / build |
|---|---|---|
| Real bulk + FACS truth | `validation_v2/*_cross` (cross-study, harder) | reuse mixtures |
| Matched pseudobulk | `validation_v2/*_holdout` (same study) | reuse mixtures |
| SimBu simulator | `make_pseudobulk.py` (Dirichlet, **no mRNA bias**, **collapses immune**) | **add SimBu**; keep our simulator for core Fig 2 |
| scRNA references | `reference_v2/<tissue>/` (+ `reference/` for WAT/PBMC/HRT/SKM) | reuse; build a tissue→ref **manifest** |
| `omnideconv::deconvolute` | `R/run_omnideconv.R` (music/dwls/scdc/bisque) | **extend** to 7 methods + per-method normalization |
| Performance metrics | `score_validation.py` (per-type R/ρ/RMSE/bias already) | **extend** (MAE/MAPE/global; multi-method aggregator) |
| Fig 2/3/4/5 plots | — | **build** viz module (R `omnideconv::make_benchmarking_scatterplot` + custom) |

**Reference-location caveat:** `reference_v2/` holds cortex, gastroc, hippocampus_WT,
kidney, lung (+merged). Heart, WAT, PBMC, skeletal-muscle references live under
`reference/`. Phase 1 builds a single manifest mapping each of the 11 `validation_v2` sets →
(reference dir, mixture dir, mtx basename) so every downstream step is config-driven.

---

## 3. Phase 0 — Environment & installs (pre-license, login node)

> **STATUS (2026-06-03): DONE — unified into the single `motrpac-env`, not a separate venv.**
> Per the clone-and-go reproducibility goal, AutoGeneS + Scaden were installed into
> `motrpac-env` itself (Python 3.12) on **modern CPU TensorFlow 2.21 + tf-keras** (Keras-2 shim,
> `TF_USE_LEGACY_KERAS=1`), NOT the Python-3.9 venv first prototyped. A pip dry-run proved this
> adds 26 packages and changes only `h5py` (3.16→3.14), leaving the torch / numpy 2.4.3 /
> transformers GeneCompass stack untouched (validated: torch+transformers import, numpy intact,
> Scaden trains+predicts via tf-keras). **SimBu 1.8.0** installed into `R_libs/`. reticulate wired
> to `motrpac-env` in `run_omnideconv.sh` (+ `CUDA_VISIBLE_DEVICES=-1` so TF stays CPU, never
> contending with torch). Pins added to root `requirements.txt` and both container defs
> (`scaden` via `--no-deps`); exact lockfile in `setup/requirements-omnideconv.txt`; installer
> `R/install_omnideconv_python.sh`. Leftover conda envs deleted. Bullets below are the original
> plan (superseded only on the venv-vs-unified-env choice).

> All installs run on a **login node** (compute nodes are offline). Reuse the conda-strip /
> module pattern in `R/install_omnideconv.sh`. Pin versions to the paper where it matters.
> **Scratch → project `tmp/` only** (never `/tmp`); clean up after.

- **0.1 Python methods (AutoGeneS, Scaden) via reticulate.**
  - Stand up a dedicated reticulate venv (recommended: `R_libs/../py_omnideconv` or reuse
    `motrpac-env`) and point `RETICULATE_PYTHON` at it. AutoGeneS needs `autogenes==1.0.4`
    (+ `pymoo`); Scaden needs `scaden==1.1.2` (+ TensorFlow).
  - Prefer `omnideconv::install_all_python()` (handles the wheels); fall back to explicit
    `pip install` if it pulls the wrong versions. Validate `build_model_autogenes` /
    `build_model_scaden` import cleanly on a toy dataset.
  - **TensorFlow note:** Scaden trains a small DNN. CPU works; on Gilbreth the scheduler is
    GPU-mandatory anyway (a100-40gb) so a GPU is present — pin a CUDA-compatible TF or force
    CPU to avoid version churn. Decide at install time (see §9).
- **0.2 SimBu (Bioconductor, v1.8)** into `R_libs/` — the simulator for **all four** bias
  scenarios. Install via `pak::pkg_install("bioc::SimBu")`; verify `SimBu::simulate_bulk`
  with `scaling_factor` ∈ {`expressed_genes`, `NONE`} on a toy dataset.
- **0.3 Update setup manifests** so the env is reproducible for any researcher: add SimBu +
  the Python methods to `deconvolution/setup/r_packages.yaml` (+ a `python_packages` block),
  extend `install_omnideconv.sh` (method list + reticulate bootstrap), and note them in
  `setup/SETUP.md`. (Keep the Danko-Lab BayesPrism untouched — never install
  `omnideconv/BayesPrism`.)
- **0.4 Probe** the full panel loads: `omnideconv`, MuSiC, DWLS, SCDC, BisqueRNA, SimBu,
  reticulate→{autogenes, scaden}. Record versions to a `setup/versions_omnideconv.txt`
  snapshot.

**Exit criteria:** all 7 methods + SimBu run on a toy 200-cell reference end-to-end.

---

## 4. Phase 1 — Shared scaffolding

- **4.1 Tissue manifest** (`deconvolution/omnideconv_bench/tissues.yaml`): per validation
  set → reference dir, mixture dir, mtx basename, focal/parenchymal type, immune labels.
  Drives every later step (and matches the existing config convention,
  `pipeline_config.yaml` + local override).
- **4.2 Bulk TPM normalization (fidelity gap).** The paper feeds **TPM bulk** for AutoGeneS,
  MuSiC, SCDC, Scaden, CIBERSORTx (counts for BayesPrism/Bisque/DWLS). We currently feed
  counts for everything. Build a **rat gene-length table** from the Ensembl
  **mRatBN7.2** GTF (exon-union length per gene), align to our gene IDs, and add a
  counts→TPM step. SimBu can also emit TPM directly when feature lengths are supplied —
  use that for simulated bulks. **Decision point (D1, §11):** acquire rat gene lengths
  (faithful) vs. CPM-approximate vs. counts-only.
- **4.3 Extend `run_omnideconv.R`** to the 7-method panel:
  - Add `autogenes`, `scaden` branches (reticulate; `build_model`→`deconvolute`, mirroring
    the DWLS two-step fix already in place).
  - Per-method input matrix selection (counts vs TPM) from a small lookup, honoring the
    table in §1. Keep the validated rat gene-cleanup path identical.
  - Set paper defaults: DWLS `dwls_method=mast_optimized`, `pval_cutoff=0.05`; AutoGeneS
    `ngen=5000`, `max_iter=1e6`; Scaden default steps. BayesPrism (our runner) — confirm
    cell-types-only + no terminal Gibbs for matched-assay runs.
- **4.4 Extend scoring** (`score_validation.py` → add MAE, MAPE, global pooled R/RMSE; new
  `aggregate_bench.py` to stack metrics across method × tissue × cell type into one tidy
  table). Keep existing per-type R/ρ/RMSE/mean_bias and the separable-compartment logic.
- **4.5 Visualization module** (`omnideconv_bench/plots.R` + `.py`): Fig 2A scatter (pred vs
  true, colored by type), Fig 2B radar (`log(1/RMSE)`, R), Fig 4 ΔRMSE heatmap, Fig 5A
  %-correct bars, Fig 5B chord diagrams (`circlize`), Fig 5C unknown-content line plots.
  Use `omnideconv::make_benchmarking_scatterplot` / `make_barplot` where they fit.

---

## 5. Phase 2 — Core fraction benchmark (Fig 2), all 11 tissues

Run **7 methods × 11 tissues** on the existing `validation_v2` mixtures (reuse the 4 WAT
results already computed).

- **5.1** Submit the 77-cell matrix as SLURM jobs (§9). Outputs:
  `validation_v2/<TISSUE>/results/fractions_<method>.csv`.
- **5.2** Score each vs `cellfrac` and `rnafrac` truth → per-type + global metrics; assemble
  the master table and per-tissue Fig 2A/2B figures.
- **5.3** Report per method: global R/RMSE, per-type winners/losers, false-negatives
  (types driven to ~0), and the holdout (matched, easy) vs cross (cross-study, hard) gap —
  the rat analog of the paper's pseudobulk-vs-real-bulk contrast. Cross-reference our
  existing BayesPrism numbers (e.g., WAT: BayesPrism 0.95/0.95, DWLS 0.98/0.99).

---

## 6. Phase 3 — Bias battery (the four confounder analyses)

All scenarios **self-simulate from each tissue's own reference and deconvolve with that same
reference** (downsampled to **500 cells/type**, paper-faithful), so the *only* moving part
is the confounder. SimBu seed fixed per sample for paired comparisons.

- **6.1 mRNA-content bias (Fig 4) — headline.**
  Per tissue: 50 pseudobulks, 10,000 cells in random proportions, depth fixed 1e7, simulated
  **twice with identical seeds** — once `scaling_factor=expressed_genes` (bias on), once
  `NONE` (bias off). Deconvolve both with the same reference. Compute
  **ΔRMSE = RMSE(bias) − RMSE(no-bias)** per method × cell type → heatmap (purple = worse
  with bias / fails to correct; green = better). Expected: DWLS/MuSiC/Scaden/SCDC correct
  the bias; BayesPrism/AutoGeneS (and CIBERSORTx in Phase 5) degrade — **this is the
  mechanism behind our hepatocyte over-estimation.**
- **6.2 Spillover (Fig 5A/B).**
  Per tissue: SimBu **"pure"** scenario — 50 reps per cell type, 1000 cells, single type
  each. Deconvolve with the tissue reference. Quantify **% correctly assigned** (diagonal,
  ideal 100%) and **% spillover** to other types (off-diagonal, ideal 0%). Chord diagrams
  per tissue + a method × tissue %-correct summary. Watch transcriptionally similar pairs
  (immune subtypes; nephron segments; neuron subclasses).
- **6.3 Granularity / aggregation (Fig 3).**
  Build a curated **coarse ↔ fine** label hierarchy per tissue **from `cell_type`** (NB:
  `cell_state` is per-sample leiden IDs, *not* biological subtypes — unusable as a level).
  E.g. kidney {Effector CD8⁺ T, Immune, Myeloid} → "Leukocyte"; {Principal, Intercalated,
  Beta-intercalated} → "Collecting-duct". Simulate fine-resolution pseudobulks (SimBu
  `mirror_db`); deconvolve at each level **and** run the **aggregation strategy**
  (deconvolve fine → sum to coarse) vs **direct-coarse**. Tests our standing decision to
  keep immune subtypes resolved and aggregate *after* deconvolution (memory
  `reference_omnideconv_benchmark` item 3). Restrict to tissues with a genuine multi-level
  hierarchy (immune-rich tissues; brain neuron subclasses).
- **6.4 Unknown content (Fig 5C + S7).**
  - **(a) Leave-one-cell-type-out (S7):** retrain the reference with one type removed,
    deconvolve the full-composition mixtures, report ΔRMSE per removed type — quantifies the
    incomplete-reference penalty and spillover redistribution.
  - **(b) Dominant-parenchyma sweep (5C):** SimBu **"weighted"** scenario sweeping the
    most-abundant parenchymal type (rat analog of "tumor": hepatocyte / cardiomyocyte /
    proximal-tubule …) at 0/5/10/20/30/50/70/80/90%, reference trained **without** the focal
    type (500 cells/type), 5 reps × 10 samples per level. Mean Pearson per cell type vs
    focal %. Extends `make_purity_sweep.py` / `SWEEP_hepato_*` and our dominant-parenchyma
    findings (memory `dominant_parenchyma_cross_failure`).

---

## 7. Phase 4 — Synthesis & report

- **7.1** One tidy master table: method × tissue × cell type × analysis × metric.
- **7.2** Rat replicas of Fig 2 (scatter+radar), Fig 3 (granularity), Fig 4 (mRNA-bias
  heatmap), Fig 5 (spillover chord + unknown-content curves).
- **7.3** Narrative `omnideconv_bench/REPORT.md`: rat method ranking; which methods correct
  mRNA bias; spillover leaders; the granularity (resolve-then-aggregate) recommendation;
  unknown-content robustness — each cross-referenced to our BayesPrism results and the
  paper's human/mouse findings. Flag rat-specific caveats (adipocyte/parenchyma single-cell
  loss; activity confound for the real MoTrPAC run, memory
  `motrpac_deconv_production_decisions`).
- **7.4** Update memory (`omnideconv-crosscheck-status`) with the full-panel outcome.

---

## 8. Phase 5 — Post-license addendum (CIBERSORTx + Bseq-SC)

Slots into the *same* scaffolding once the license lands — only the method set grows.

- **8.1 CIBERSORTx:** pull the official `fractions` Docker image (v1.0, 12/21/2019); on the
  cluster, run via **Apptainer** (Docker is unavailable) — verify omnideconv's
  `check_container` / `set_cibersortx_credentials` path works through Apptainer, else shell
  out to the container directly. Token/email via env. **No batch correction** (S/B-mode
  off), CPM sc / TPM bulk.
- **8.2 Bseq-SC (optional, 12-method extra):** `bseqsc_config()` pointing at the
  registration-gated `CIBERSORT.R`. Lower priority — not one of the benchmarked 8.
- **8.3** Re-run Phases 2 & 3 adding CIBERSORTx (+ Bseq-SC), regenerate every table/figure
  to the full **8(+1)-method** panel. Re-state the headline comparisons.

---

## 9. Compute & SLURM plan (Gilbreth, reese18 assoc)

- **GPU-mandatory scheduler:** every job must request `--gres=gpu:1` (CPU-only jobs are
  rejected), capped to **a100-40gb only**. `standby` QOS wall = **4h**; long jobs (DWLS on
  full refs, Scaden training) need **`--qos=normal`** (14d). (Memory
  `omnideconv-crosscheck-status`, `slurm_local_absolute_paths`.)
- **Long pole = DWLS** signature build (scales with reference cell count: ~10.5h single-core
  MAST on the 22k-cell WAT ref). Mitigations: `dwls_method=mast_optimized` (paper default,
  ~2–3× faster) + forward `ncores` (already fixed in `run_omnideconv.R`). **Bias scenarios
  use 500-cell/type downsampled refs → DWLS drops to minutes**, so Phase 3 is cheap; Phase 2
  full-ref runs are the heavy part.
- **Job matrix:** Phase 2 = 7 × 11 = 77 runs (detached `--qos=normal` for DWLS; standby OK
  for the fast methods). Phase 3 = 4 analyses × 11 tissues × 7 methods on small refs.
  Stagger to respect the 7-GPU GRES cap (`AssocGrpGRES` pends when saturated).
- **Reticulate/TF:** Scaden imports TensorFlow inside R via reticulate on a compute node —
  ensure the venv + `RETICULATE_PYTHON` are exported in the SLURM wrapper (compute nodes are
  offline, so the env must be fully pre-installed in Phase 0). SLURM scripts stay local-only
  with absolute paths (not portability-refactored).

---

## 10. Per-method fidelity checklist (from paper Methods "Details on method settings")

- **AutoGeneS:** sc **CPM**, bulk **TPM**; `ngen=5000`, `max_iter=1e6`; HVG selection
  **off** (use the shared gene universe); one-step impl.
- **BayesPrism:** sc+bulk **counts**; predict **cell types** (not states); **skip** the
  terminal Gibbs fraction update for matched-assay (holdout) runs; set `species`.
- **Bisque:** sc+bulk **counts** (internal CPM); defaults.
- **DWLS:** `mast_optimized` + `pval_cutoff=0.05`; `ncores` forwarded.
- **MuSiC:** sc **counts**, bulk **TPM**; `batch_ids` = subject/sample.
- **Scaden:** sc **counts**, bulk **TPM**; 5000 steps, batch 128, lr 1e-4, 1000×100-cell
  sims.
- **SCDC:** sc **counts**, bulk **TPM**; single reference.
- **CIBERSORTx (Phase 5):** sc **CPM**, bulk **TPM**; **no** batch correction.
- **SimBu:** v1.8; mRNA bias = expressed-genes scaling factor; depth + composition matched
  to scenario; fixed seeds for paired ±bias.

---

## 11. Risks, gaps & open decisions

- **D1 — Rat TPM (bulk normalization).** No rat gene-length table exists in-repo. Faithful
  replication needs TPM bulk for 4–5 methods. Plan: build exon-union lengths from Ensembl
  mRatBN7.2 GTF. *Fallback:* CPM-approximate, or counts-only (note the deviation). **Needs a
  call before Phase 2** if strict fidelity is required.
- **D2 — Granularity hierarchy.** `cell_state` is per-sample leiden clusters, not biological
  subtypes. The coarse↔fine map must be curated by hand from `cell_type` per tissue; some
  tissues may lack a meaningful multi-level hierarchy (→ run granularity only where it
  exists). Use `condition_resolved`-independent label logic (memory
  `condition_resolved_unreliable`).
- **D3 — SimBu rat support.** SimBu is organism-agnostic (operates on the count matrix +
  cell labels) so rat is fine, but the expressed-genes mRNA-bias factor must be computed on
  the rat reference *after* our gene-cleanup; verify the scaling behaves on sparse rat data.
- **D4 — Reference downsampling.** Paper uses 500 cells/type (bias scenarios) / 10% (HaoSub).
  Some rat types are rare (<500); SimBu/our sampler must allow resampling, and very-rare
  types may be unstable — report n per type.
- **D5 — Adipocyte / parenchyma single-cell loss.** snRNA loses adipocytes (WAT) and
  under-captures some parenchyma; "missing from reference" overlaps with the unknown-content
  analysis — interpret §6.4 results with this in mind (memory
  `multitissue_deconv_survey`, `dominant_parenchyma_cross_failure`).
- **Stretch (out of committed scope):** Fig 6 reference-variability (needs ≥2 refs/tissue —
  partially available via the `*_cross` + `*_merged` sets); CPM/MOMF/CDSeq; CIBERSORTx
  web-vs-Docker discrepancy.

---

## 12. Deliverables / artifact inventory

```
deconvolution/
  OMNIDECONV_BENCHMARK_PLAN.md            (this file)
  omnideconv_bench/
    tissues.yaml                          tissue→ref/mixture manifest
    simulate_simbu.R                       ±mRNA-bias / pure / weighted / mirror_db scenarios
    run_panel.sh / .R                      7-method runner (extends run_omnideconv.R)
    aggregate_bench.py                     master tidy metrics (R/RMSE/MAE/MAPE, global+type)
    plots.R / plots.py                     Fig 2/3/4/5 rat replicas
    REPORT.md                             narrative synthesis
  setup/  (updated)                        SimBu + python methods in r_packages.yaml, install_omnideconv.sh
results live under: validation_v2/<TISSUE>/results/ and …/bias/<analysis>/
```

**Net new code:** SimBu scenario simulator, 2 method branches in the runner, TPM step,
metrics extension + aggregator, plotting module, tissue manifest, setup updates.
**Reused:** references, validation mixtures, gene-cleanup, BayesPrism results, scoring core,
SLURM patterns.
