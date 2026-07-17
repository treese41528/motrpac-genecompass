# MoTrPAC Deconvolution: Tissues, References, Proportions, and How Much to Trust Them

**Status:** working reference document. Compiled 2026-07-14 from the files on disk (see §10 for the full
provenance list). Supersedes the tissue/reference/proportion sections of
`deconvolution/DECONVOLUTION_PIPELINE_REPORT.md` and `deconvolution/AIM2_DECONV_RESULTS.md`, both of which
are stale and contain errors corrected here (§6.11).

> ⚠️ **SUPERSEDED IN PART — 2026-07-16 rebuild.** This document audits the **pre-rebuild deployed
> references**. Four tissues were rebuilt on 2026-07-16; the per-tissue verdicts, rosters, cell counts and θ
> tables **below** for these four now describe **retired references** and should be read against the new
> facts (authority: `REFERENCE_SELECTION_PLAN.md` §2, `DECONV_REBUILD_RUNBOOK.md`, and the canonical results
> in `data/deconvolution/references_v3/` + `results/motrpac/`):
> - **HEART** — now uses the authors' **SCP2828 (GSE280111) deposited per-cell labels → 16 clean cardiac
>   types**, **135,288 cells**, mean θ ≈ **72.0% cardiomyocytes** (Endothelial 10.3%, Cardiac fibroblasts
>   6.3%, Monocytes 6.3%). The "99.58% saturation" narrative in §1/§3.2/§4.5/§6.7 no longer holds.
> - **SKMGN + SKMVL** — now **one shared GSE137869 (Ma young `-Y`) snRNA reference**, myofiber over-split
>   **merged → 5 clean types** (Skeletal myocytes / Fibroblasts / Endothelial cells / Vascular smooth muscle
>   cells / Macrophages), **10,763 cells** each; θ ≈ 96% skeletal myocytes. The GSE184413 (§3.4) and
>   GSE254371 rat-mouse-chimera (§3.5) analyses no longer describe production.
> - **BAT** — now **deconvolved** from **GSE244451** (authors' deposited labels → **6 types**, 28,246 cells);
>   the "buildable — never built" note (§2.3) is stale.
> - **VENACV** — the GSE280111 pulmonary-vein proxy was built then **DROPPED (BLOCKED)** — irreducibly
>   lung-contaminated — so it is **not deconvolved**.
>
> The broader rebuild wave also replaced several references that this 07-14 audit still describes as deployed
> — **CORTEX** (now rat-only GSE303115, 12,933 cells / 11 types, no longer 85% non-rat), **HIPPOC** (now
> GSE295314, not the GSE305314 tauopathy-WT with the debris cluster), **LUNG** (P10-neonate GSE242310 dropped
> → 46,653 cells / 28 types) and **WATSC** (young-`-Y`-only → 12,223 cells / 13 types, not the 3-arm-pooled
> 31,870). Their §3/§4/§6 analyses below therefore also describe retired references. Only **KIDNEY, LIVER and
> BLOOD** are byte-for-byte the production references this audit was written against. Downstream also moved:
> the rebuilt DE run yields **15 hotspots** (was 21), so the hotspot tables in §5.2, §5.4 and §8 are
> superseded. New-state authority: `REFERENCE_SELECTION_PLAN.md`, `DECONV_REBUILD_RUNBOOK.md`, and the
> canonical on-disk results.

---

## 0. Purpose and how to read this document

This is the single place to answer four questions: **what tissues do we have**, **what single-cell
reference did each one use**, **what came out of the deconvolution**, and **how much of it can be
believed**.

Two things must be kept apart while reading:

**θ (theta) is a fraction of mRNA mass, not a fraction of cells.** BayesPrism estimates what share of the
bulk RNA in a sample came from each cell type. A hepatocyte carries far more mRNA than a lymphocyte, so a
liver that is ~60% hepatocytes *by cell count* can legitimately be >90% hepatocyte *by mRNA*. This is not
an error — it is the definition of the quantity. It is also the reason **differential claims (does this
cell type's θ or expression change with training/sex?) are far safer than absolute claims (what fraction
of this tissue is cell type X?)**.

**A "validated" tissue is not the same as a "correct" tissue.** Almost all of our validation is
*self-consistency*: mixtures are simulated from the same reference that is then used to deconvolve them.
That certifies the solver, not the biology. Where a reference is missing a cell type that exists in the
bulk, or contains cells from the wrong species, self-consistency validation passes anyway and tells you
nothing. Two of our ten tissues fail precisely in that blind spot (§6.1, §6.2).

Every number below is traceable to a file path, given inline in `code formatting`. Anything not verifiable
from a file is labelled **UNKNOWN** or **UNVERIFIED** and is never silently filled in.

### Which trees this document uses

The repository contains several parallel result and reference trees. Only one of each is canonical, and
this was established from code and file contents, not assumed:

| Artifact | Canonical location | Why |
|---|---|---|
| Deconvolution results | `data/deconvolution/results/motrpac/` | `config/pipeline_config.yaml:340` sets `results_dir: data/deconvolution/results`; `pipeline/run_stage10.py:143-144` reads `<results_dir>/motrpac`. `results_merged/motrpac/` and `results_novis/motrpac/` are **literally empty directories** — staging areas whose contents were adopted into `results/`, originals archived under `_merge_adoption_backup_20260623/` and `_liverlung_adoption_backup_20260701/`. |
| References | The `reference_dir` values in `deconvolution/tissue_references.yaml` (v2026-07-02) | These deliberately straddle `references/` (BLOOD, HEART, LIVER, LUNG, WATSC) and `references_v2/` (CORTEX, HIPPOC, KIDNEY, SKMGN, SKMVL). The YAML is the single source of truth; no code reads it, so it must be honoured by hand (§6.10). |
| Per-cell-type DE | `data/deconvolution/genecompass_input/pseudobulk_de/` | 185 blocks / 21 hotspots, matching the known production totals. `pseudobulk_de_novis` (LUNG only, 34 rows), `_merged` (CORTEX only, 28 rows) and `_smoke` (LIVER only, 6 rows) are orphaned single-tissue side-runs. |

**The LIVER results in `results/motrpac/LIVER` ARE the Visium-excluded ("novis") rebuild** — confirmed by
content, not mtime alone: `estimated_fractions.csv` md5 `ba6da424…` differs from the archived pre-fix copy
(`645ccf72…`), and `references/liver_GSE220075/cells_meta.tsv` now holds 27,041 cells / 4 samples with the
two Visium GSMs removed (backup: 31,820 / 6).

**No trees are mixed in §2–§4.** They *are* mixed inside the historical validation artefacts, and that is
itself a finding — see §5.4 and §6.9.

---

## 1. One-page verdict

Verdict scale: **CLEAN** = absolute composition and differential claims both usable. **CAUTION** =
differential claims usable, absolute composition unreliable. **BROKEN** = the basis or the output is
demonstrably wrong; do not report composition, and treat even differential results from this tissue as
provisional.

| Tissue | Reference study | Ref cells (true) | Dominant cell type (mean θ) | Verdict | One-line reason |
|---|---|---:|---|---|---|
| **LIVER** | GSE220075 rat liver atlas | 27,041 | Hepatocytes **91.95%** | **CLEAN** | Healthy atlas, Visium contamination removed, cross-method agreement on real bulk (DWLS ρ=0.943). |
| **KIDNEY** | GSE240658 (No-treatment arm) | 28,626 | Proximal tubule **79.89%** | CAUTION | Plausible roster, but reference is the control arm of a diet/nephrosis study and there is no real-bulk cross-check. |
| **SKMGN** (gastrocnemius) | GSE184413 (Normal ambulation) | 19,936 | Skeletal muscle **74.74%** | CAUTION | Only **2 donor animals** (one aged); scRNA of mononucleated cells, so myonuclei are under-sampled. |
| **HEART** | GSE280111 LV atlas | 166,344 | Cardiomyocytes **99.58%** | CAUTION | Best reference in the set, but θ is **saturated** — all 22 other types sit at ≤0.08%, so absolute composition is meaningless. |
| **SKMVL** (vastus lateralis) | GSE254371 | 20,490 | Skeletal muscle **64.88%** | CAUTION | **52% of the reference is a rat–mouse chimera** from an iPSC engineering study; tissue is absent from the omnideconv validation panel. |
| **LUNG** | GSE273062 + GSE252844 + GSE242310 (pooled) | 50,643 | Endothelial **39.62%** | CAUTION (borderline) | Three pooled disease/injury models, one donor is a **P10 neonate**; **alveolar type I cells are absent** from the basis; 22 of 34 DE blocks are near-empty. |
| **CORTEX** | GSE303115 nine-mammal multiome | 86,844 | Excitatory neurons **56.08%** | **BROKEN** | **85% of the reference cells are not rat** — cat, rabbit, zebrafish, cow, horse. The θ *looks* sane; the basis is wrong-species. |
| **HIPPOC** | GSE305314 (WT arm of a tauopathy study) | 45,038 | "Intermediate monocytes" **63.56%** | **BROKEN** | An ambient/debris cluster mislabelled as monocytes absorbs 64% of the mRNA mass; excitatory neurons collapse to 0.02%. A brain is not 64% monocytes. |
| **BLOOD** | GSE285476 PBMC (transplant-rejection study) | 12,315 | "ISG-expressing T cells" **55.26%** | **BROKEN** | Reference is **PBMC**; bulk is **whole blood (PAXgene)**. Erythroid and granulocyte mRNA — ~80% of the mass — has no basis vector and is force-fit onto lymphocytes. |
| **WATSC** | GSE137869 (aging + calorie-restriction) | 31,870 | "Luminal epithelial" **37.89%** | **BROKEN** | The reference contains **no adipocytes**. A 155-cell mammary cluster (0.5% of reference cells) absorbs 37.9% of bulk mRNA, and *which* surrogate wins flips with sex. |

**One CLEAN tissue. Five CAUTION. Four BROKEN.** The failures are not random: they are two specific,
nameable mechanisms (§7), and both are properties of the *reference*, never of the bulk data or the
solver.

---

## 2. The MoTrPAC tissue inventory

### 2.1 All 19 bulk tissues

Bulk matrices live at `data/deconvolution/motrpac_bulk/<CODE>/{bulk.mtx, bulk_genes.tsv, bulk_samples.tsv}`.
`bulk.mtx` is stored **samples × genes**. The gene axis is **byte-identical across all 19 tissues** (single
md5 `76f400d4…`, 22,050 genes).

| Code | Full anatomical name | Genes | Samples | Deconvolved? |
|---|---|---:|---:|:---:|
| ADRNL | Adrenal gland | 22,050 | 50 | ✗ |
| BAT | Brown adipose tissue (interscapular) | 22,050 | 50 | ✗ |
| BLOOD | Whole blood (PAXgene) | 22,050 | 50 | ✓ |
| COLON | Colon (large intestine) | 22,050 | 50 | ✗ |
| CORTEX | Cerebral cortex | 22,050 | 50 | ✓ |
| HEART | Heart (left ventricle) | 22,050 | 50 | ✓ |
| HIPPOC | Hippocampus | 22,050 | 50 | ✓ |
| HYPOTH | Hypothalamus | 22,050 | 50 | ✗ |
| KIDNEY | Kidney | 22,050 | 50 | ✓ |
| LIVER | Liver | 22,050 | 50 | ✓ |
| LUNG | Lung | 22,050 | 50 | ✓ |
| OVARY | Ovary | 22,050 | **24** | ✗ |
| SKMGN | Skeletal muscle — **gastrocnemius** | 22,050 | 50 | ✓ |
| SKMVL | Skeletal muscle — **vastus lateralis** | 22,050 | 50 | ✓ |
| SMLINT | Small intestine | 22,050 | 50 | ✗ |
| SPLEEN | Spleen | 22,050 | 50 | ✗ |
| TESTES | Testes | 22,050 | **25** | ✗ |
| VENACV | Vena cava | 22,050 | 50 | ✗ |
| WATSC | White adipose tissue — **subcutaneous** | 22,050 | 50 | ✓ |

**899 bulk samples total.** Codes-to-names cross-checked against
`deconvolution/reference/canonical_references.tsv` and `deconvolution/tissue_references.yaml`.

### 2.2 Experimental design

From `deconvolution/reference/motrpac_sample_pheno.tsv` (6,156 vials; joined to each tissue's
`bulk_samples.tsv`). **All 899 bulk samples join to the phenotype table — 0 unmatched.**

- **Sex:** male / female. **Group:** `control, 1w, 2w, 4w, 8w`. **n = 5 per cell.**
- 17 tissues: 2 sex × 5 group × 5 reps = **50** samples (25M / 25F; 10 control, 40 trained).
- **OVARY:** female only, **24** samples — the `4w` cell has n=4, not 5.
- **TESTES:** male only, **25** samples (5 × 5, complete).

**A design warning that matters downstream.** `group` and `sacrificetime` are *perfectly confounded*: the
cross-tabulation is strictly block-diagonal, and **all 1,223 control vials were sacrificed at 8 weeks**
("Eight-week program Control Group"). There is no 1w, 2w or 4w control. So `group` is **one five-level
factor, not a crossed 2 (intervention) × 4 (timepoint) factorial**. Training duration cannot be separated
from age-at-sacrifice by any model fitted to this data. This is also why the perturbation/GRN work has no
permutation null available: dose is deterministically nested inside the training label.

As actually fitted (`deconvolution/R/run_pseudobulk_de.R`): `~ sex + week_numeric` (an ordinal dose slope),
plus a per-sex `~ factor(week)` omnibus F-test.

### 2.3 Why nine tissues were not deconvolved — and why the recorded reason is wrong

`deconvolution/DECONVOLUTION_PIPELINE_REPORT.md:3234` states that ADRNL, BAT, COLON, OVARY, HYPOTH, SMLINT,
SPLEEN, TESTES and VENACV were "bulk lifted but not deconvolved (**no exercise metadata**)."

**That is false.** Every one of the nine has complete exercise metadata: a 100% join to
`motrpac_sample_pheno.tsv` (0 of 899 unmatched) with the identical 2 × 5 × 5 design as the ten that *were*
deconvolved. Missing metadata is not the reason.

**The real reason is reference availability**, and it splits three ways. Evidence: `motrpac_tissue_match`
and `geo_title` in `reports/annotations/annotation_inventory.tsv` (810 rows), plus the absence of any
reference directory for these tissues in either `references/` or `references_v2/`, plus
`deconvolution/build_all_references.sh`, which only ever attempts the ten.

| Tissue | Rat SC samples in corpus | Verdict |
|---|---:|---|
| ADRNL | 0 | **True zero** — no rat adrenal single-cell data anywhere in the corpus. |
| OVARY | 0 | **True zero.** |
| TESTES | 0 | **True zero.** |
| VENACV | 0 | **True zero** (aorta and pulmonary vein exist — wrong vessel). |
| SPLEEN | 2 → effectively 0 | **Mis-mapped.** Both rows have `geo_title = Rnorvegicus_thymus_MAIT` — thymus, MAIT-**sorted**, and the two are duplicates. A free-text search for `spleen` returns 0 rows. The `motrpac_tissue_match=spleen` label is an inventory bug. *(Flagged by one agent, not independently re-verified — treat as **UNVERIFIED**.)* |
| BAT | 8 | **Buildable — never built.** GSE137869 (the same Ma 2020 aging/CR study that supplies WATSC) contributes 6 BAT samples; GSE244451 adds 2 (hypertension model). |
| COLON | 14 | **Thin / compromised.** 12 of 14 come from GSE223564, an ulcerative-colitis drug trial whose samples are **cell-type-sorted** (`Enterocyte_*`, `Tcells_*`) and duplicated — sorted fractions cannot yield a proportion-bearing roster. Only GSE143920 (2 samples) is plausibly healthy whole colon. |
| SMLINT | 5 | **Thin.** GSE272055 gives 2 healthy proximal-jejunum samples; the other 3 are colorectal-cancer and diabetes models. |
| HYPOTH | 3 | **Thin.** GSE248413 only — an aging study with exactly one young healthy sample. |

Five tissues (ADRNL, OVARY, TESTES, VENACV, and effectively SPLEEN) have **literally no rat single-cell
data** in the corpus; for these, "no adequate reference exists" is exact. Four (BAT, COLON, SMLINT, HYPOTH)
**do** have candidate data that was simply never built. **BAT is the cheapest gap to close** — it is the
same study and the same build recipe as WATSC. It would, however, inherit WATSC's exact defects: pooled
young/old/calorie-restricted arms, and droplet scRNA that almost certainly captures no adipocytes (§6.2).

### 2.4 The gene axis, and where genes are lost

**Bulk liftover** (`deconvolution/reference/motrpac_bulk_liftover_report.txt`): 32,883 raw MoTrPAC gene IDs
→ **22,050**. The arithmetic closes exactly:

```
32,883 raw
 −  8,880 unmapped
 = 24,003 lifted   (20,203 direct + 2,702 by symbol + 1,098 via Entrez/RGD id_history)
 −  1,953 many-to-one collapses (summed)
 = 22,050 final
```

**Into the deconvolution, the *reference* — not the bulk — is the binding constraint.**
`deconvolution/R/run_deconvolution.R` filters the reference in four steps and then intersects with the
22,050-gene bulk axis:

```
reference genes
  − ribosomal / mitochondrial / hemoglobin   (deconvolution/reference/rat_exclude_genes.tsv)
  − chrX / chrY
  − non-protein-coding
  − genes expressed in < 3 cells
  ∩ bulk axis
  = pred_z gene set
```

Worked example (SKMVL, from `logs/rebuild_dx_11091415.out`): 15,394 − 185 − 546 − 33 − 0 = 14,630 reference
genes → intersected with bulk → **14,212** genes in `pred_z`.

| Tissue | pred_z genes | | Tissue | pred_z genes |
|---|---:|---|---|---:|
| BLOOD | **11,319** | | LIVER | 12,759 |
| WATSC | 13,874 | | LUNG | 15,348 |
| SKMGN | 13,907 | | HEART | 15,802 |
| SKMVL | 14,212 | | HIPPOC | 16,085 |
| KIDNEY | 15,028 | | CORTEX | **16,132** |

Verified: `pred_z` is a strict subset of the bulk axis in all ten tissues, and no excluded gene survives.

**Defect in the exclusion list (new, not previously recorded).**
`deconvolution/reference/rat_exclude_genes.tsv` holds 247 genes labelled 226 ribosomal / 13 mitochondrial /
8 hemoglobin. The "ribosomal" set was built with a `^RP[SL]…` regex and therefore **also deletes the entire
S6-kinase family**: `Rps6ka1`–`Rps6ka6`, `Rps6kb1`, `Rps6kb2`, `Rps6kc1`, `Rps6kl1` (plus `Rps19bp1`).
These are **signalling kinases, not ribosomal proteins** — `Rps6kb1` is p70-S6K, the canonical mTORC1
hypertrophy readout; `Rps6ka1-3` are RSK1-3, core ERK exercise-signalling nodes. All ten are present on the
MoTrPAC bulk axis and **absent from `pred_z/genes.txt` in every one of the ten tissues**. The per-cell-type
DE is therefore structurally incapable of detecting the most canonical exercise-signalling family in the
project. (True ribosomal count is ~215, not 226.) **Whether removing this filter would change any DE
conclusion is UNKNOWN — it requires a rerun.**

---

## 3. The reference studies, tissue by tissue

Provenance is from `reports/annotations/annotation_inventory.tsv` (`sample_id` → `gsm` / `geo_title` /
`geo_organism`) and study-level facts from `/depot/reese18/data/catalog/master_catalog.json` (`studies[]`).
Cell/gene/type counts are from each reference's `summary.txt` and `cells_meta.tsv`.

**Read this first: only two of the ten references come from a healthy tissue atlas — HEART and LIVER.**
Every other reference is a control arm carved out of a disease, injury, aging or engineering study. The
cell-type clusters in those references were defined in the context of that disease, and the "control"
animals were chosen to control for *that* study's intervention, not for exercise in a 6-month-old F344 rat.

### 3.0 Cross-cutting defect: three references double-count every cell

The sample inventory maps **two `sample_id`s onto one `gsm`** in three studies, and
`build_reference.py::select_samples()` de-duplicates on `sample_id`, not on `gsm`. The paired h5ad files
have identical shapes and identical non-zero counts.

| Tissue | sample_ids ingested | unique GSMs | cells as built | **true cells** |
|---|---:|---:|---:|---:|
| CORTEX | 22 | 11 | 173,688 | **86,844** |
| HEART | 38 | 19 | 332,688 | **166,344** |
| SKMGN | 4 | **2** | 39,872 | **19,936** |

Exact duplication leaves the cell-type *mean* profile ψ unchanged, so this is **not science-invalidating**
for BayesPrism — but every reported cell count for these three tissues is 2× inflated wherever it appears,
and it doubles memory and compute. The sting is SKMGN: **the gastrocnemius reference rests on two animals,
not four.** **UNVERIFIED:** we confirmed identical GSMs and identical per-sample cell counts but did not
compare the count matrices byte-for-byte. If the duplicate pairs are *not* identical, ψ *is* affected.

### 3.1 LIVER — `data/deconvolution/references/liver_GSE220075`

**GSE220075** — "A Rat Liver Cell Atlas Reveals Intrahepatic Myeloid Heterogeneity" (PMID 38026201,
submitted 2022-12-05, 12 study samples). snRNA-seq (plus scRNA and Visium arms we do not use). This is a
genuine healthy atlas; Dark Agouti and Lewis strains.

Built from the 4 sham/control snRNA samples (`LEW1`, `LEW2`, `DA1`, `DA2`) = **27,041 cells / 17,895 genes /
6 cell types / 44 cell states**.

**Exclusion (the 2026-07-01 fix):** the two Visium spatial samples — `GSE220075_sample2` (`Rat_B1_VIS`) and
`GSE220075_sample11` (`Rat_A1_VIS`) — were previously being ingested as if they were nuclei, contaminating
the reference with ~15% spatial spots. They are now dropped automatically by the `reference_qc` gate inside
`build_reference.py`. The pre-fix reference had 31,820 cells / 6 samples.

Roster (cells): Hepatocytes 23,655 · Endothelial 2,453 · Kupffer 381 · Hepatic stellate 378 · Hepatic
immune 145 · Cholangiocytes 29.

### 3.2 HEART — `data/deconvolution/references/heart_GSE280111_LV`

**GSE280111** — "Transcriptional profile of the rat cardiovascular system at single-cell resolution"
(PMID 39709602, 2024-10-23, 78 study samples, 505,835 nuclei, Wistar). A healthy atlas. We take the **left
ventricle** samples only: 19 GSMs from 10 animals.

Built: 332,688 cells as recorded, **166,344 true** (§3.0) / 16,802 genes / 23 types / 682 states.
Supersedes the earlier `heart_GSE155699` reference (spontaneously-hypertensive rat, and it had **no
cardiomyocytes**), which still sits on disk unused.

Roster (as-built counts; halve for true): Cardiac fibroblasts 122,460 · Cardiomyocytes 66,880 · Endothelial
66,258 · Naive T 20,584 · CD8+ T 18,998 · Pericytes 13,896 · VSMC 5,254 · Naive B 4,210 · CD4+ T 2,708 ·
Memory T 2,430 · Schwann 1,432 · DC 1,398 · Monocytes 1,336 · CD8+ NKT 1,066 · Intermediate mono 1,034 ·
Macrophages 738 · Non-classical mono 618 · Neutrophils 430 · Mesothelial 324 · Erythroid precursor 274 ·
Neurons 146 · γδ T 142 · Memory CD4+ T 72.

### 3.3 KIDNEY — `data/deconvolution/references_v2/kidney_GSE240658`

**GSE240658** — "A kidney specific fasting-mimicking diet induces podocyte reprogramming … in
glomerulopathy" (2023-08-11, 16 study samples). snRNA-seq. A **diet-intervention / puromycin-nephrosis
model**; we take the four `No treatment` samples (`Kidney, healhty control, rep1–rep4` — the typo is in
GEO).

Built: **28,626 cells / 17,867 genes / 17 types / 74 states.**

Roster: Proximal tubule 15,910 · Loop of Henle 3,441 · Distal tubule 1,961 · Endothelial 1,783 · Principal
1,497 · Renal fibroblasts 1,077 · α-intercalated 811 · Podocytes 433 · Immune 419 · Myeloid 307 · Renal
vesicle 269 · β-intercalated 250 · Mesangial 180 · Renal interstitial fibroblasts 150 · Effector CD8+ T 58 ·
Intercalated 43 · Beta-intercalated 37.

Note the three near-duplicate intercalated labels (`α-intercalated`, `β-intercalated`, `Intercalated`,
`Beta-intercalated`) — a collinearity hazard (§7.2), and the source of two of the near-empty DE blocks
(§5.3).

### 3.4 SKMGN (gastrocnemius) — `data/deconvolution/references_v2/gastrocnemius_GSE184413`

**GSE184413** — "Mechanotherapy promotes ECM remodeling in aged rat muscle recovering from disuse"
(PMIDs 35434632, 36895061; 2021-09-19; 6 study samples). scRNA-seq. A **disuse-atrophy + mechanotherapy
model**; we take the "Normal ambulation" arm.

Built: 39,872 cells as recorded, **19,936 true** — **two GSMs**: `Adult WB` and `Old WB`. So the reference is
**one adult animal and one aged animal**. 18,937 genes / 17 types / 92 states.

It is also **scRNA of mononucleated cells**, which by construction under-samples myonuclei — the very
compartment that dominates bulk muscle mRNA.

Roster (as-built; halve for true): Fibroblasts 12,208 · Skeletal muscle cells 11,686 · Muscle fibroblasts
7,418 · Endothelial 1,600 · Memory CD8+ T 1,388 · CD8+ T 1,374 · Macrophages 1,040 · Myeloid DC 1,002 ·
Satellite 690 · Smooth muscle 360 · Non-classical mono 356 · Monocytes 184 · NK 168 · MDSC 162 ·
Neutrophils 104 · Naive B 80 · Mast 52.

`Fibroblasts` vs `Muscle fibroblasts` is an unmitigated collinear label split.

### 3.5 SKMVL (vastus lateralis) — `data/deconvolution/references_v2/skeletal_muscle_GSE254371_muscle_merged`

**GSE254371** — "Generation of allogenic and **xenogeneic** functional muscle stem cells [scRNA-seq]"
(PMID 38713532, 2024-01-28, 2 study samples). This is an **iPSC / blastocyst-chimera engineering study**,
not a muscle atlas.

Built: **20,490 cells / 15,394 genes / 14 types / 48 states**, from two samples:

| geo_title | cells | share |
|---|---:|---:|
| `SD rat_muscle` | 9,797 | 48% |
| **`Rat-mouse chimera_muscle`** | **10,693** | **52%** |

**The majority of the vastus-lateralis reference comes from a rat–mouse chimeric animal** whose
satellite/myogenic compartment is mouse-iPSC-derived, mapped into rat gene space (`geo_source_name = "Mus
musculus, Rattus norvegicus chimera"`). Note that the inventory records `geo_organism = Rattus norvegicus`
for this sample, so an organism filter alone would **not** catch it.

Compounding this: **SKMVL is not in the omnideconv validation panel at all** (the panel's only muscle is
gastrocnemius). It is the least defensible reference in the set — and it is also the tissue producing our
highest-AUC exercise hotspots (§5.2).

Roster: Fibroblasts 9,293 · Skeletal muscle 2,864 · Endothelial 2,415 · Macrophages 1,979 · Muscle
fibroblasts 1,332 · Naive CD8+ T 757 · Myofibroblasts 735 · B 506 · Mast 163 · CD8+ NKT-like 162 ·
Progenitor 116 · Neurons 82 · Schwann 53 · Neutrophils 33.

### 3.6 LUNG — `data/deconvolution/references/lung_native_pooled` (THREE pooled studies)

Built by `deconvolution/build_lung_pooled.py`, not by a single `build_reference.py` call. It replaced the
previous `lung_GSE178405` reference, an **in-vitro tissue-engineering** study (cell isolates, engineered
day-7 constructs, tri-culture, P7-developing tissue) that `reference_qc.py` FAILs and that made lung the
weakest tissue in the panel.

The replacement is *native*, but it is still three control arms borrowed from three different injury models:

| Accession | Study | PMID | Sample used | What its "control" is |
|---|---|---|---|---|
| GSE273062 | Lung in VeNx and **SuHx** rats | 40357547 | `VeNx rep1`, `VeNx rep2` | control arm of a **pulmonary-hypertension** model |
| GSE252844 | scRNA-seq of rat lung, **blast-exposure** heterogeneity | 38237742 | `C3 rep1` | control arm of a **gas-explosion lung-injury** model |
| GSE242310 | Cellular senescence in **hyperoxic neonatal** rat lung | 37874230 | `NOX lung` | room-air control of a **bronchopulmonary-dysplasia** model — the animal is a **P10 neonate** |

Built: **50,643 cells / 17,110 genes / 34 types / 106 states** from 4 donors.

**Alveolar type I cells are absent from the roster.** AT1 cells line the alveolar surface and are a major
component of bulk lung; with no AT1 basis vector, their mRNA must be projected onto something else — the
same omitted-component mechanism that breaks BLOOD and WATSC (§7.1). We have not quantified how much mass
that is, so the size of the distortion is **UNKNOWN**.

Roster: Alveolar macrophages 19,309 · Club 4,345 · Naive B 4,116 · CD8+ T 3,464 · Monocytes 2,875 · Ciliated
2,700 · Airway epithelial 2,477 · **Alveolar type II 1,441** · Neutrophils 1,067 · Pulmonary fibroblasts 778 ·
Myeloid DC 726 · CD4+ T 692 · CD8+ NKT-like 679 · NK 652 · Endothelial 591 · Naive CD8+ T 569 · Ionocytes
521 · Naive CD4+ T 520 · Memory CD4+ T 437 · Erythroid precursor 427 · Classical mono 356 · Tuft 304 ·
Macrophages 300 · Intermediate mono 290 · Pro-B 216 · Secretory 180 · Mesothelial 156 · Mast 121 · pDC 76 ·
Pre-B 72 · T 69 · Effector CD8+ T 54 · Basal 40 · γδ T 23.

### 3.7 CORTEX — `data/deconvolution/references_v2/cortex_GSE303115_union_merged`

**GSE303115** — "Single Cell Multiomics Across **Nine Mammals** Reveals Cell Type Specific Regulatory
Conservation in the Brain [Multiome]" (2025-07-21, 22 study samples, 6 platforms). snRNA Multiome.

Built: 173,688 cells as recorded, **86,844 true** / 18,162 genes / 28 types / 512 states.

**85% of the reference is not rat.** `build_reference.py --study GSE303115 --tissue cortex` has **no
organism filter** — `select_samples()` matches on accession + `tissue_normalized` + `in_corpus` only, and
*every* species' cortex in this study carries `tissue_normalized = cortex`. All 11 GSMs were ingested. From
`cells_meta.tsv` joined to the inventory's `geo_organism`:

| Organism | cells (as built) | share |
|---|---:|---:|
| *Felis catus* (cat) | 47,318 | 27.2% |
| *Oryctolagus cuniculus* (rabbit) | 39,894 | 23.0% |
| **_Danio rerio_ (zebrafish)** | **30,596** | **17.6%** |
| ***Rattus norvegicus*** | **25,866** | **14.9%** |
| *Bos taurus* (cow) | 15,078 | 8.7% |
| *Equus caballus* (horse) | 14,936 | 8.6% |

True unique rat cells: **12,933** (rat rep1 6,596 + rep2 6,337). The corpus had already force-mapped every
species into rat `ENSRNOG` gene space, so the non-rat nuclei are pooled *silently* into the rat cell-type
profiles ψ. **14 of the 28 CORTEX cell types contain ZERO rat cells** — including Endothelial (1,458),
Macrophages (2,860), Purkinje neurons (2,718), Plasma cells (2,314), Crypt cells (724) and Meningeal
fibroblasts. Excitatory neurons — the type carrying 56% of the θ mass — are only **15.1% rat**.

The roster itself gives the game away: **Purkinje neurons** (cerebellum, not cortex), **Crypt cells**
(intestine) and **Radial glial cells** are what you get when a rat marker panel is used to annotate fish,
cow and horse nuclei.

**This overturns the recorded "Fix 2" diagnosis.** The project record says the old 5,536-gene cortex
reference was "a BUILD bug (intersection join), NOT shallow data," repaired by `--gene-join outer` →
18,162 genes. Recomputed from the h5ad `var` indices:

```
RAT-ONLY inner join  (rat rep1 ∩ rep2)      = 21,003 genes
ALL-SPECIES inner join (the v1 behaviour)   =  5,536 genes   ← the "shallow cortex" number, exactly
```

5,536 was the **six-species gene intersection**. `--gene-join outer` treated the symptom and *preserved the
cause*: it kept all six species and padded the non-rat cells with structural zeros. The correct fix —
rat-only, 2 GSMs — yields **21,003 genes from clean data, more than the 18,162 of the "fixed" reference.**

Roster (as built): Excitatory neurons 69,068 · GABAergic 24,224 · Oligodendrocytes 23,868 · Astrocytes
13,498 · OPC 12,882 · Microglia 11,660 · Macrophages 2,860 · Purkinje 2,718 · Plasma cells 2,314 · Radial
glial 2,296 · CD8+ T 2,224 · Endothelial 1,458 · Meningeal fibroblasts 1,056 · Crypt cells 724 · Naive B
522 · Dopaminergic 414 · Immature neurons 358 · Immune 350 · Glial 318 · Neuroblasts 204 · Brain endothelial
190 · CD4+ T 92 · Neuroepithelial 82 · Myofibroblasts 78 · Myelinating Schwann 76 · γδ T 64 · Fibroblasts 48
· Cholinergic 42.

### 3.8 HIPPOC — `data/deconvolution/references_v2/hippocampus_GSE305314_WT_merged`

**GSE305314** — "Intercellular signaling and synaptic deconstruction … in an **AD tauopathy** model"
(PMID 41249845, 2025-08-13, 28 study samples). snRNA-seq (the study also has scATAC and Visium arms).

The build itself is **clean and verified**: 6 samples, all wild-type, all snRNA (`WT 10mo` ×3, `WT 20mo`
×3). The Tau arm and the non-snRNA arms are correctly excluded. **45,038 cells / 19,774 genes / 15 types /
122 states.**

Two provenance caveats remain. (i) The clusters were defined in the context of a tauopathy study.
(ii) **The WT rats are 10 and 20 months old**, against MoTrPAC's ~6-month adults — half the reference is an
aged animal.

**Footgun:** the stale v1 directory `data/deconvolution/references/hippocampus_GSE305314` still exists and
**does contain the Tau arm** (12 samples, 86,223 cells, 27 types). Production does not use it, but anything
globbing `references/` would pick it up.

Roster: Oligodendrocytes 15,520 · Excitatory neurons 8,124 · Immature granule 6,464 · Astrocytes 4,086 ·
GABAergic 3,356 · OPC 2,472 · Microglia 2,258 · Fibroblasts 582 · Choroid plexus 524 · **Intermediate
monocytes 482** · Endothelial 391 · Ependymal 327 · Pericytes 233 · Cajal-Retzius 116 · Müller cells 103.

That 482-cell "Intermediate monocytes" cluster is the one that destroys this tissue — see §6.3.

### 3.9 BLOOD — `data/deconvolution/references/peripheral blood mononuclear cells_GSE285476`

**GSE285476** — "Decoding the Retn-Cap1 Pathway in Intermediate Monocytes Mediating **Liver Allograft
Rejection**" (2024-12-28, 24 study samples). scRNA-seq of PBMCs from a BN→Lewis liver-transplant model. We
take **one** sample: `PBMC, healthy control, 0d, rep1`.

Built: **12,315 cells / 17,338 genes / 14 types / 18 states. One donor.**

Roster: Naive B 4,495 · Naive CD4+ T 3,109 · Non-classical mono 1,422 · Memory CD4+ T 925 · Memory T 564 ·
Memory CD8+ T 502 · CD4+ T 354 · NK 240 · Platelets 173 · Classical mono 158 · Megakaryocytes 155 · B 122 ·
ISG-T 74 · Basophils 22.

**No erythroid cells. No granulocytes.** That is not an oversight in the annotation — it is what "PBMC"
*means*. Ficoll density-gradient isolation removes red cells and granulocytes by design. The MoTrPAC bulk
is **whole blood in PAXgene tubes**. See §6.1.

### 3.10 WATSC — `data/deconvolution/references/white adipose tissue_GSE137869`

**GSE137869** — "Caloric restriction reprograms the single-cell transcriptional landscape of *Rattus
norvegicus* aging" (Ma 2020, PMID 32109414; 2019-09-23; 60 study samples). scRNA/snRNA.

> 📄 **Full dossier: [`REFERENCE_STUDY_GSE137869_MA2020.md`](REFERENCE_STUDY_GSE137869_MA2020.md)** — the
> paper reviewed in full (41 pp), the exact arm ages (young **5 mo**, old **27 mo**, CR = 70% for 9 months),
> the design-by-design comparison to MoTrPAC, the **paper's own confirmation that adipocytes were never
> captured** (collagenase/dispase → stromal-vascular fraction; "adipocyte" appears as no cluster anywhere),
> the **unresolvable WAT depot**, and the tissues Ma covers that we are *not* using (notably **BAT**, which
> MoTrPAC has in bulk and we never deconvolved).

Built: **31,870 cells / 17,895 genes / 17 types / 96 states**, from **all six** WAT samples:
`WAT-M-Y`, `WAT-F-Y`, `WAT-M-O`, `WAT-F-O`, `WAT-M-CR`, `WAT-F-CR` — i.e. **all three arms pooled**:
38.4% young ad-lib, 31.2% **old** ad-lib, 30.4% **old + calorie-restricted**. `build_reference.py` has no
age or diet filter. MoTrPAC rats are ~6-month adults. The per-sample Leiden clusters are arm-pure, but the
type-level profile ψ is an average over young, aged and calorie-restricted adipose.

Roster: Macrophages 13,960 · Fibroblasts 10,586 · CD8+ T 1,615 · Endothelial 1,280 · Neutrophils 1,004 ·
Myeloid DC 671 · γδ T 567 · Non-classical mono 563 · NK 411 · Naive CD8+ T 384 · Naive B 336 · **Luminal
epithelial 155** · Memory CD4+ T 129 · Myofibroblasts 80 · Pre-B 68 · Monocytes 31 · pDC 30.

**No adipocytes.** `Adipoq`, `Plin1`, `Lep` and `Cidec` are at floor in every cluster — mature adipocytes
float and lyse in droplet scRNA. For bulk white adipose, adipocytes are *the* dominant mRNA source. See
§6.2.

**UNKNOWN:** Ma 2020 never states the WAT **depot**. GEO is silent and the Cell STAR Methods is paywalled.
If the depot is visceral, WATSC is matched to the wrong depot entirely.

---

## 4. The deconvolved proportions

All θ from `data/deconvolution/results/motrpac/<TIS>/estimated_fractions.csv`. n = 50 bulk samples per
tissue; rows sum to exactly 1.0. **All values below are percent of bulk mRNA mass, not percent of cells.**

The sample mapping (`mixN` → viallabel → phenotype) is **code-guaranteed, not a row-order assumption**:
`deconvolution/R/prepare_motrpac_bulk.R:168-170` writes the matrix and the sample list from the same
`scols` variable, and `deconvolution/R/run_deconvolution.R:97` assigns `mix{i}` in that row order. All ten
tissues: 50/50 viallabels joined to `motrpac_sample_pheno.tsv`.

Full by-sex, by-intervention and by-dose breakdowns for every cell type in every tissue are in the working
dump (§10). Only the marginal tables and the differences that matter are reproduced here.

### 4.1 LIVER (6 types) — CLEAN

| Cell type | mean % | sd | min | max |
|---|---:|---:|---:|---:|
| Hepatocytes | 91.95 | 2.63 | 87.70 | 96.02 |
| Endothelial cells | 2.97 | 1.20 | 1.24 | 5.32 |
| Kupffer cells | 2.67 | 1.13 | 1.09 | 4.86 |
| Hepatic stellate cells | 2.27 | 0.57 | 1.23 | 3.27 |
| Hepatic immune cells | 0.14 | 0.19 | 0.00 | 0.97 |
| Cholangiocytes | 0.00 | 0.03 | 0.00 | 0.20 |

**Cross-method on the real bulk** (`results/motrpac/LIVER/omnideconv/fractions_{dwls,music,scdc}.csv`) —
methods agree on the dominant type:

| Method | shared types | Spearman ρ (per-type mean θ) | hepatocyte mean % | per-sample hepatocyte r vs BayesPrism |
|---|---:|---:|---:|---:|
| BayesPrism | 6 | — | 91.95 | — |
| DWLS | 6 | 0.943 | 91.84 | 0.846 |
| MuSiC | 4 (2 types not recovered) | 0.400 | 98.28 | 0.914 |
| SCDC | 6 | 0.406 | 90.78 | 0.721 |

Hepatocytes land at 91–98% under every method. The **minor** types do not agree on rank (MuSiC/SCDC
ρ≈0.40), so minor liver types are not cross-method robust. Note that Pearson across per-type means is
inflated to ~1.0 by the single dominant point; Spearman is the honest statistic.

### 4.2 KIDNEY (17 types) — CAUTION

| Cell type | mean % | sd | min | max |
|---|---:|---:|---:|---:|
| Proximal tubule cells | 79.89 | 2.82 | 74.08 | 84.96 |
| Renal fibroblasts | 6.65 | 0.69 | 5.38 | 8.65 |
| Loop of Henle cells | 5.07 | 1.44 | 2.71 | 8.54 |
| Immune cells | 2.41 | 0.21 | 1.83 | 2.73 |
| Endothelial cells | 2.33 | 0.41 | 1.57 | 3.25 |
| Distal tubule cells | 1.78 | 0.58 | 0.67 | 3.45 |
| Principal cells | 0.96 | 0.83 | 0.00 | 3.66 |
| β-intercalated cells | 0.29 | 0.09 | 0.10 | 0.52 |
| Podocytes | 0.27 | 0.08 | 0.10 | 0.46 |
| Renal vesicle cells | 0.09 | 0.18 | 0.00 | 0.80 |
| α-intercalated cells | 0.08 | 0.05 | 0.01 | 0.22 |
| Intercalated cells | 0.07 | 0.17 | 0.00 | 0.98 |
| Myeloid cells | 0.05 | 0.08 | 0.00 | 0.29 |
| Mesangial cells | 0.03 | 0.04 | 0.00 | 0.16 |
| Beta-intercalated cells | 0.02 | 0.03 | 0.00 | 0.17 |
| Renal interstitial fibroblasts | 0.01 | 0.03 | 0.00 | 0.14 |
| Effector CD8+ T cells | 0.01 | 0.01 | 0.00 | 0.05 |

Biologically plausible (proximal tubule dominates kidney mRNA). **No cross-method check exists on the real
bulk for this tissue.**

### 4.3 SKMGN — gastrocnemius (17 types) — CAUTION

| Cell type | mean % | sd | min | max |
|---|---:|---:|---:|---:|
| Skeletal muscle cells | 74.74 | 0.94 | 72.34 | 77.23 |
| Smooth muscle cells | 9.20 | 0.51 | 7.68 | 10.00 |
| Endothelial cells | 6.01 | 0.38 | 5.17 | 6.80 |
| Muscle fibroblasts | 4.07 | 0.60 | 3.09 | 5.68 |
| Fibroblasts | 2.70 | 1.02 | 1.18 | 7.43 |
| Satellite cells | 2.02 | 0.28 | 1.11 | 2.67 |
| Naive B cells | 0.55 | 0.05 | 0.45 | 0.68 |
| Macrophages | 0.32 | 0.05 | 0.26 | 0.49 |
| Monocytes | 0.25 | 0.03 | 0.15 | 0.35 |
| CD8+ T cells | 0.09 | 0.03 | 0.02 | 0.17 |
| Memory CD8+ T cells | 0.02 | 0.03 | 0.00 | 0.14 |
| Mast cells | 0.02 | 0.01 | 0.00 | 0.03 |
| NK cells | 0.01 | 0.01 | 0.00 | 0.04 |
| Non-classical monocytes | 0.01 | 0.02 | 0.00 | 0.11 |
| Myeloid dendritic cells | 0.00 | 0.01 | 0.00 | 0.02 |
| Neutrophils | 0.00 | 0.00 | 0.00 | 0.00 |
| Myeloid-derived suppressor cells | 0.00 | 0.00 | 0.00 | 0.00 |

Plausible. `Fibroblasts` (2.70%) and `Muscle fibroblasts` (4.07%) are collinear labels — their *sum* is
identified, the split between them is not.

### 4.4 SKMVL — vastus lateralis (14 types) — CAUTION

| Cell type | mean % | sd | min | max |
|---|---:|---:|---:|---:|
| Skeletal muscle | 64.88 | 4.31 | 42.48 | 72.43 |
| Muscle fibroblasts | 14.11 | 3.22 | 7.61 | 26.76 |
| Myofibroblasts | 8.48 | 1.13 | 3.59 | 11.11 |
| Endothelial cells | 6.83 | 0.74 | 3.99 | 8.12 |
| Fibroblasts | 2.73 | 1.85 | 0.41 | 12.45 |
| B cells | 2.66 | 0.32 | 1.62 | 3.26 |
| Macrophages | 0.15 | 0.74 | 0.00 | 5.24 |
| Progenitor cells | 0.08 | 0.51 | 0.00 | 3.65 |
| Schwann cells | 0.05 | 0.05 | 0.00 | 0.21 |
| Neurons | 0.01 | 0.01 | 0.00 | 0.05 |
| Mast cells | 0.01 | 0.01 | 0.00 | 0.05 |
| Naive CD8+ T cells | 0.01 | 0.02 | 0.00 | 0.13 |
| CD8+ NKT-like cells | 0.00 | 0.00 | 0.00 | 0.01 |
| Neutrophils | 0.00 | 0.00 | 0.00 | 0.00 |

The roster is plausible, which is exactly what makes it dangerous: the reference underneath it is 52%
rat–mouse chimera (§3.5) and has no validation of any kind. **B cells at 2.66% of muscle mRNA is
implausibly high** and should be treated as a fitting artefact.

### 4.5 HEART (23 types) — CAUTION (saturated)

| Cell type | mean % | sd | min | max |
|---|---:|---:|---:|---:|
| Cardiomyocytes | **99.58** | 0.25 | 98.89 | 99.87 |
| Endothelial cells | 0.08 | 0.01 | 0.04 | 0.11 |
| Cardiac fibroblasts | 0.07 | 0.11 | 0.00 | 0.42 |
| Schwann cells | 0.06 | 0.01 | 0.04 | 0.08 |
| Vascular smooth muscle cells | 0.03 | 0.06 | 0.00 | 0.32 |
| Macrophages | 0.02 | 0.04 | 0.00 | 0.18 |
| Mesothelial cells | 0.02 | 0.01 | 0.01 | 0.04 |
| Intermediate monocytes | 0.02 | 0.01 | 0.01 | 0.03 |
| Pericytes | 0.02 | 0.03 | 0.00 | 0.15 |
| *(13 further types)* | ≤0.01 | | 0.00 | ≤0.13 |

Cardiomyocytes are a genuinely enormous mRNA source, but **99.58% is saturation, not measurement**. Every
other cell type is at the numerical floor and carries no usable compositional signal. Differential
cardiomyocyte claims may still hold (the CM profile validates at r=0.995 against pseudobulk), but **no
absolute heart composition claim should be made from these numbers.**

### 4.6 LUNG (34 types) — CAUTION (borderline)

| Cell type | mean % | sd | min | max |
|---|---:|---:|---:|---:|
| Endothelial cells | 39.62 | 1.88 | 34.42 | 42.88 |
| Pulmonary fibroblasts | 21.66 | 1.00 | 19.87 | 24.53 |
| Ciliated cells | 14.75 | 1.19 | 12.32 | 18.42 |
| Naive B cells | 13.66 | 0.69 | 12.66 | 15.64 |
| Alveolar type II cells | 9.44 | 0.65 | 7.84 | 10.91 |
| Airway epithelial cells | 0.23 | 0.30 | 0.00 | 1.61 |
| Club cells | 0.12 | 0.26 | 0.00 | 1.30 |
| Erythroid precursor cells | 0.09 | 0.03 | 0.02 | 0.16 |
| Myeloid dendritic cells | 0.08 | 0.19 | 0.00 | 0.77 |
| Macrophages | 0.06 | 0.12 | 0.00 | 0.51 |
| Alveolar macrophages | 0.03 | 0.05 | 0.00 | 0.18 |
| *(23 further types)* | ≤0.05 | | 0.00 | ≤0.91 |

Five types carry 99.1% of the mass and the remaining 29 are at floor. **Naive B cells at 13.66% of lung
mRNA is not credible.** Alveolar macrophages — 38% of the reference cells — get 0.03% of the mRNA. With
alveolar type I absent from the basis (§3.6), the alveolar compartment's mRNA has nowhere correct to go.
Treat the lung roster as a fit, not as an anatomy.

### 4.7 CORTEX (28 types) — BROKEN basis, plausible output

| Cell type | mean % | sd | min | max |
|---|---:|---:|---:|---:|
| Excitatory neurons | 56.08 | 3.49 | 48.14 | 62.82 |
| Astrocytes | 16.31 | 0.84 | 14.32 | 18.62 |
| GABAergic neurons | 9.31 | 2.15 | 4.50 | 14.41 |
| Oligodendrocytes | 8.94 | 0.97 | 7.25 | 11.91 |
| Endothelial cells | 4.50 | 0.28 | 3.77 | 5.29 |
| Cholinergic neurons | 1.35 | 0.48 | 0.60 | 3.11 |
| Microglia | 1.09 | 0.11 | 0.87 | 1.44 |
| CD8+ T cells | 0.78 | 0.25 | 0.25 | 1.52 |
| Crypt cells | 0.43 | 0.19 | 0.06 | 1.07 |
| Myofibroblasts | 0.37 | 0.10 | 0.13 | 0.56 |
| Myelinating Schwann cells | 0.37 | 0.09 | 0.16 | 0.56 |
| *(17 further types)* | ≤0.09 | | 0.00 | ≤0.54 |

This is the most *biologically believable* roster in the whole set — and it was produced by a reference
whose cells are 85% cat, rabbit, zebrafish, cow and horse (§3.7), in which the dominant type (Excitatory
neurons) is only 15% rat and 14 of the 28 types contain **no rat cells at all** (Endothelial, Macrophages,
Purkinje, Crypt cells, …). A plausible-looking answer from an invalid basis is worse than an obviously
broken one. **Do not cite CORTEX absolute composition** until the reference is rebuilt rat-only.

### 4.8 HIPPOC (15 types) — BROKEN

| Cell type | mean % | sd | min | max |
|---|---:|---:|---:|---:|
| **Intermediate monocytes** | **63.56** | 5.61 | 34.18 | 69.99 |
| Oligodendrocytes | 15.00 | 2.59 | 11.27 | 23.29 |
| Endothelial cells | 6.55 | 0.61 | 5.53 | 8.16 |
| Pericytes | 4.12 | 0.42 | 3.49 | 5.38 |
| Müller cells | 3.68 | 0.93 | 2.42 | 6.36 |
| Microglia | 3.06 | 1.09 | 2.30 | 10.35 |
| Ependymal cells | 1.65 | 0.82 | 1.19 | 7.11 |
| Astrocytes | 1.06 | 0.67 | 0.05 | 4.22 |
| Choroid plexus cells | 0.79 | 0.80 | 0.00 | 3.46 |
| Fibroblasts | 0.23 | 0.44 | 0.00 | 2.15 |
| Immature granule cells | 0.14 | 0.20 | 0.00 | 0.78 |
| Oligodendrocyte precursor cells | 0.08 | 0.27 | 0.00 | 1.61 |
| GABAergic neurons | 0.04 | 0.09 | 0.00 | 0.45 |
| Cajal-Retzius cells | 0.03 | 0.18 | 0.00 | 1.27 |
| **Excitatory neurons** | **0.02** | 0.03 | 0.00 | 0.11 |

A hippocampus is not 64% monocytes and it is not 0.02% excitatory neurons. Diagnosis in §6.3.

### 4.9 BLOOD (14 types) — BROKEN

| Cell type | mean % | sd | min | max |
|---|---:|---:|---:|---:|
| ISG-expressing T cells | 55.26 | 4.54 | 43.76 | 64.88 |
| Megakaryocytes | 19.51 | 2.03 | 15.02 | 24.93 |
| Basophils | 9.31 | 0.88 | 7.43 | 12.01 |
| Non-classical monocytes | 5.47 | 1.32 | 3.08 | 8.39 |
| Classical monocytes | 5.40 | 1.88 | 2.14 | 11.53 |
| Memory T cells | 1.68 | 1.02 | 0.14 | 5.03 |
| Natural killer cells | 1.26 | 0.32 | 0.48 | 2.01 |
| Naive CD4+ T cells | 0.78 | 1.27 | 0.00 | 5.31 |
| Naive B cells | 0.63 | 0.21 | 0.06 | 1.18 |
| CD4+ T cells | 0.48 | 1.13 | 0.00 | 5.23 |
| Platelets | 0.11 | 0.59 | 0.00 | 3.98 |
| Memory CD8+ T cells | 0.06 | 0.35 | 0.00 | 2.39 |
| Memory CD4+ T cells | 0.03 | 0.07 | 0.00 | 0.26 |
| B cells | 0.02 | 0.08 | 0.00 | 0.52 |

A 74-cell "ISG-expressing T" cluster (0.6% of the reference) takes 55% of whole-blood mRNA. **Cross-method
on the real bulk — the methods do not agree**:

| Method | dominant-type θ (ISG-T) | notes |
|---|---:|---|
| BayesPrism | 55.26% | |
| DWLS | 68.12% | Spearman ρ vs BP = 0.793 |
| SCDC | 20.24% | its own top type is Megakaryocytes at 40.6%; ρ = 0.876 |
| MuSiC | — | **failed outright, no output** |

A three-way split on the dominant type plus one outright method failure. **Nothing about blood composition
is cross-method robust.** See §6.1 for why.

### 4.10 WATSC (17 types) — BROKEN

| Cell type | mean % | sd | min | max |
|---|---:|---:|---:|---:|
| Luminal epithelial cells | 37.89 | **22.21** | 8.76 | 75.13 |
| Fibroblasts | 29.37 | 12.71 | 3.48 | 51.56 |
| Endothelial cells | 14.64 | 5.92 | 4.13 | 24.08 |
| Myofibroblasts | 5.70 | 2.47 | 1.03 | 11.92 |
| Plasmacytoid dendritic cells | 2.92 | 1.00 | 0.63 | 4.83 |
| NK cells | 1.81 | 3.30 | 0.00 | 17.27 |
| Macrophages | 1.30 | 1.42 | 0.00 | 3.70 |
| Myeloid dendritic cells | 1.26 | 3.13 | 0.00 | 16.22 |
| Naive CD8+ T cells | 1.25 | 3.13 | 0.00 | 17.45 |
| Naive B cells | 0.91 | 4.02 | 0.00 | 27.97 |
| CD8+ T cells | 0.82 | 1.74 | 0.00 | 10.59 |
| Pre-B cells | 0.61 | 1.33 | 0.00 | 5.30 |
| Non-classical monocytes | 0.55 | 0.77 | 0.00 | 2.98 |
| Memory CD4+ T cells | 0.35 | 1.10 | 0.00 | 7.71 |
| Gamma delta T cells | 0.32 | 0.40 | 0.00 | 1.25 |
| Monocytes | 0.30 | 0.64 | 0.00 | 3.46 |
| Neutrophils | 0.00 | 0.01 | 0.00 | 0.06 |

Note the standard deviations: ±22 points on the top type, ranging 8.8% to 75.1% across 50 samples of the
same tissue. Macrophages are **43.8% of the reference cells** but **1.30% of θ**; "Luminal epithelial cells"
are **0.49% of the reference cells** (155 of 31,870) but **37.89% of θ**. The θ is inverted with respect to
its own reference. See §6.2.

### 4.11 Cross-method coverage — read this before quoting any cross-check

**omnideconv was run on the real MoTrPAC bulk for only two of ten tissues: BLOOD and LIVER.** The other
eight have no `omnideconv/` directory under `results/motrpac/<TIS>/`. The much-cited "11-tissue omnideconv
panel" lives under `data/deconvolution/validation_v2/` and consists of **synthetic pseudobulk mixtures
simulated from the same references** — it cannot cross-check these θ, and it must not be presented as if it
could. Of the two tissues that *do* have a real-bulk cross-check, one agrees (LIVER) and one splits three
ways (BLOOD).

---

## 5. Validation and QC

### 5.1 Purity sweeps (Z-matrix recovery)

The purity sweep is the project's pre-registered expression-recovery test, following Chu 2022: simulate
mixtures, score the median Pearson-VST correlation between recovered and true cell-type expression across
mixtures where the focal cell type holds ≥50% of the RNA. Threshold: **≥0.95**. Numbers below were
**recomputed from `data/deconvolution/validation/SWEEP_*/scores/z_vst_focal.tsv`**, not copied from the
pre-summarised files.

| Sweep | Score | n mixtures | Result |
|---|---:|---:|---|
| SWEEP_cortex_holdout | 0.9996 | 50 | PASS |
| SWEEP_hippoc_holdout | 0.9997 | 50 | PASS |
| SWEEP_kidney_holdout | 0.9991 | 40 | PASS |
| SWEEP_heart_holdout | 0.9976 | 48 | PASS |
| SWEEP_hepato_holdout | 0.9976 | 46 | PASS |
| SWEEP_lung_club_holdout | 0.9968 | 42 | PASS |
| SWEEP_skmvl_holdout | 0.9968 | 32 | PASS |
| SWEEP_lung_at2_holdout | 0.9948 | 50 | PASS |
| SWEEP_skmgn_holdout | 0.9913 | 40 | PASS |
| **SWEEP_hepato_cross** | **0.9490** | 20 | **FAIL** (< 0.95) |

Three things must be said plainly about this table.

**(a) BLOOD and WATSC were never swept.** `data/deconvolution/validation/PBMC_holdout/scores/` and
`WAT_holdout/scores/` contain only `metrics.tsv` and `merged_long.tsv` — no `purity_sweep_summary.tsv`
exists. They did not fail the sweep; they were never given it. The claim "8 of 10 tissues meet the bar"
that appears elsewhere in the repo is a **coverage** number being read as a **pass-rate**. The two unswept
tissues are exactly the two proven omitted-component failures — the QC gap and the science gap coincide.

**(b) Nine of the ten sweeps are *holdout* sweeps** — the mixtures and the reference come from the same
study. Only liver has a **cross-study** sweep, and **it scores 0.9490, which misses the ≥0.95 bar.**
`deconvolution/DECONVOLUTION_PIPELINE_REPORT.md:1023` (and lines 1173, 1188, 1847, 3253, plus
`AIM2_DECONV_RESULTS.md:21`) describe this as "Exceeds Chu 2022 threshold (≥0.95)". It does not.
`reports/deconvolution/liver_expression_Z_purity_sweep.md:100` softens it to "lands right at the
threshold." **The project's single cross-study expression validation does not clear its own pre-registered
bar**, and cross-study Z-recovery is entirely **unmeasured for the other nine tissues**.

**(c) Two of the sweeps validate a reference we no longer use.** `SWEEP_hepato_holdout/reference/` and
`SWEEP_hepato_cross/reference/` (both mtime 2026-05-30) still contain `GSE220075_sample11` (Rat_A1_VIS,
1,701 cells) and `GSE220075_sample2` (Rat_B1_VIS, 1,667 cells) — **3,368 of 22,257 cells = 15.1% Visium
spatial spots.** The clean liver reference was built 2026-07-01. The mixtures were drawn from the same
contaminated pool, so the Visium spots sit on both sides of the score. **The one tissue whose reference we
fixed is the one tissue whose expression validation was never re-run.** LIVER currently has **no valid
Z-recovery number**, and the frequently-quoted "we meet Chu 2022 (holdout 0.998 / cross 0.949)" inherits
this. And the CORTEX sweep — `SWEEP_cortex_holdout/reference/cells_meta.tsv`, 121,392 cells, **14.9% rat** —
is a self-consistency test on a multi-species cell pool. Its 0.9996 certifies nothing about rat.

### 5.2 Per-cell-type differential expression

`data/deconvolution/genecompass_input/pseudobulk_de/de_summary.tsv`: **185 blocks** (tissue × cell type),
one per cell type per tissue, exactly matching the reference rosters. Model: `~ sex + week` on log2-CPM of
the BayesPrism Z (per-cell-type expression) matrix, limma-trend, IHW.

**21 hotspots** (`de_hotspots.tsv`; a hotspot is a block whose held-out supervised trained-vs-control AUC
clears the pre-registered bar), sorted by AUC:

| Tissue | Cell type | AUC | mean θ % | median libsize | n sig dose (IHW) |
|---|---|---:|---:|---:|---:|
| SKMVL | Skeletal muscle | 0.935 | 64.88 | 7,247,132 | 328 |
| SKMVL | Myofibroblasts | 0.925 | 8.48 | 935,799 | 306 |
| SKMVL | Endothelial cells | 0.890 | 6.83 | 767,247 | 301 |
| BLOOD | Megakaryocytes | 0.885 | 19.51 | 1,644,174 | 1,345 |
| SKMVL | B cells | 0.883 | 2.66 | 298,700 | 254 |
| SKMGN | Skeletal muscle cells | 0.878 | 74.74 | 9,036,340 | 266 |
| HEART | CD8+ T cells | 0.857 | **0.01** | 2,325 | 8 |
| BLOOD | ISG-expressing T cells | 0.845 | 55.26 | 4,698,429 | 1,815 |
| SKMGN | Naive B cells | 0.845 | 0.55 | 66,949 | 107 |
| BLOOD | Basophils | 0.843 | 9.31 | 787,260 | 1,202 |
| BLOOD | Naive B cells | 0.838 | 0.63 | 55,573 | 1,365 |
| LUNG | Pulmonary fibroblasts | 0.838 | 21.66 | 4,311,341 | 1,075 |
| SKMGN | Monocytes | 0.838 | 0.25 | 32,026 | 139 |
| SKMGN | CD8+ T cells | 0.835 | 0.09 | 9,953 | 98 |
| **LUNG** | **Myeloid dendritic cells** | 0.833 | **0.08** | **21.6** | **17** |
| LUNG | Alveolar macrophages | 0.823 | **0.03** | 439.8 | 157 |
| BLOOD | Natural killer cells | 0.820 | 1.26 | 102,755 | 1,766 |
| KIDNEY | Proximal tubule cells | 0.820 | 79.89 | 14,215,820 | 5 |
| SKMGN | Smooth muscle cells | 0.818 | 9.20 | 1,152,845 | 220 |
| BLOOD | Classical monocytes | 0.815 | 5.40 | 425,307 | 1,767 |
| BLOOD | Non-classical monocytes | 0.802 | 5.47 | 440,580 | 1,743 |

Hotspots by tissue: BLOOD 7 · SKMGN 5 · SKMVL 4 · LUNG 3 · HEART 1 · KIDNEY 1 · CORTEX/HIPPOC/LIVER/WATSC 0.

### 5.3 Degenerate DE blocks — no gating exists

**All 185 blocks carry `status=ok`.** `run_pseudobulk_de.R` computes per-block library-size diagnostics and
then never gates on them. The distribution is wildly bimodal:

| Statistic | Value |
|---|---|
| Minimum median library size | **3.03** (LUNG Intermediate monocytes) |
| Blocks with median libsize < 50 | **67 of 185** |
| Blocks with median libsize < 10 | **25 of 185** |
| Median of block median libsizes | 439.83 |
| 90th percentile | 1,641,418 |

**LUNG is the epicentre: 22 of its 34 blocks fall below 50, and 13 below 10.** The two KIDNEY blocks
previously flagged as degenerate (`Beta-intercalated` at 10.10, `Intercalated` at 51.30) are not outliers —
they are typical. All 67 sit inside the pooled global IHW / repfdr fit, so the multiple-testing calibration
is being influenced by blocks with essentially no data. The size of that effect is **UNKNOWN** (it needs a
rerun with gating to measure).

### 5.4 Composition confound, and the covariate check

`composition_confound_table.tsv`: **145 QUIET / 30 PASS_EXPRESSION / 10 FLAG_COMPOSITION**.

A `FLAG_COMPOSITION` verdict means the block's apparent expression change is not separable from a change in
that cell type's θ. **Six of the ten flags are themselves headline hotspots**: BLOOD NK cells, BLOOD Naive
B cells, SKMVL Endothelial cells, SKMVL B cells, SKMGN Monocytes, SKMGN Naive B cells. The other four
(LUNG Erythroid precursor, SKMGN Muscle fibroblasts, SKMGN Endothelial, SKMGN Satellite cells) are not
hotspots.

`rin_globin_robustness.tsv` re-fits each hotspot with RIN and globin-% as technical covariates. Three
hotspots come back as `CHECK` rather than `ROBUST`:

| Hotspot | retain_frac | ρ | Note |
|---|---:|---:|---|
| **LUNG Myeloid dendritic cells** | **0.00** | 0.955 | every dose-significant gene disappears under covariates |
| KIDNEY Proximal tubule cells | 0.643 | 0.979 | only 5 dose-significant genes to begin with |
| HEART CD8+ T cells | 0.750 | 0.966 | only 8 dose-significant genes |

**Do not chain the two tables.** `de_summary.n_sig_dose_IHW` for LUNG Myeloid DC is 17, but
`rin_globin_robustness.tsv` has `n_dose_sig_base = 1` and `n_dose_sig_cov = 0` — the two tables use
different models. `retain_frac = 0` there means **0 of 1**, not 0 of 17. That block had almost no power to
begin with.

**And the RIN gate can pass vacuously.** LUNG Alveolar macrophages has `n_dose_sig_base = 0`,
`n_dose_sig_cov = 0`, `retain_frac = NaN`, `verdict = ROBUST` — it "passed" because it had nothing to
retain, while `de_summary` calls the same block a hotspot with 157 dose-significant genes. **Both LUNG
hotspots on this check are therefore worthless**: one retains 0 of 1, the other has nothing to retain.

### 5.5 Positive controls — a coverage failure, not (only) a biology failure

`posctrl_results.tsv`: **10 of 81 pre-registered control genes have `present = False`** (`outcome =
not_testable`) — including three of the headline Tier-A parenchyma anchors: **SKMVL `Sod2`**, **HEART
`Hspa1b`**, **HEART `Hsp90aa1`**. `Sod2` (ENSRNOG00000086727) *is* on the MoTrPAC bulk axis but is **not in
the SKMVL reference gene set**, so it never reaches the Z matrix at all.

This matters for how the recorded "dominant parenchyma fails its positive controls" verdict should be read.
Those three anchors were **never testable**, so they cannot be evidence that the parenchyma failed — that
is rung 1 of the miss-reading ladder (coverage), not rung 4 (biology). `Mef2c` and `Slc2a4` (GLUT4) *were*
testable and came back `not_significant` **with the correct direction**. The parenchyma positive-control
result is weaker than recorded in both directions: less evidence of failure, and still no evidence of
success.

### 5.6 What `reference_qc.py` does and does not check

`deconvolution/reference_qc.py` gates on **sample-level string matching** over `geo_title`,
`geo_source_name` and `geo_cell_type`:

- **FAIL** classes: `SPATIAL` (visium / `_vis` / spatial / slide-seq / geomx / cosmx); `ENGINEERED`
  (engineered / cell isolate / tri-culture / organoid / cultured / in-vitro / ipsc / cell line / passage /
  `day N` / sorted / `CD\d+\+` / FACS); `BULK` (bulk rna / whole-tissue rna).
- **WARN** class: `DEVEL` (`E\d+`, `P\d+`, embryo / fetal / neonatal / postnatal).
- With `--deep`: per-sample median genes/cell from the `.mtx`, warning if max/min > 2.5 — the bimodality
  trick that caught the liver Visium spots.

**It contains zero occurrences of the strings `organism` or `gsm`.** What it cannot see:

1. **Organism.** It never reads `geo_organism`. This is why the cortex reference **passes with zebrafish
   and cow nuclei in it**.
2. **Duplicate GSMs.** It counts `sample`s, never `gsm`s. It reports "cortex_GSE303115_union_merged (22
   samples)" — those are 11 GSMs, twice.
3. **What the study was.** It never opens `master_catalog.json`. A control arm of a tauopathy,
   transplant-rejection, blast-injury or chimera study is invisible to it.
4. **Donor count.** BLOOD (n=1) and SKMGN (n=2) pass silently.
5. **Age**, unless the age happens to appear in the sample title. Proven: the P10 neonatal lung sample
   (`GSE242310`) has `geo_title = "NOX lung, scRNA-seq"` and `geo_source_name = "lung"` — no `P\d+` token,
   so no DEVEL warning, and `lung_native_pooled` passes.
6. **Missing cell types.** No adipocytes in WAT, no erythroid/granulocytes in blood, no AT1 in lung — all
   pass.
7. **Arm pooling.** WATSC pooling young + old + calorie-restricted passes.
8. **Assay↔bulk compatibility.** A PBMC reference against whole-blood PAXgene bulk passes.

Live output of `reference_qc.py --all`: **every production reference PASSES.** The only FAIL in either tree
is the already-retired `lung_GSE178405`. **The gate is currently a regression test for the two bugs we
already found, not a forward-looking QC.**

---

## 6. Known defects, ranked

### 6.1 BLOOD — omitted component (SOLVED diagnosis; the tissue is not usable as composition)

**The MoTrPAC bulk is whole blood (PAXgene).** MoTrPAC's own DE recipe uses globin % as a technical
covariate, which only makes sense for whole blood. **Our reference is PBMC** — peripheral blood
*mononuclear* cells — which **by construction excludes erythrocytes and granulocytes** (Ficoll removes
them). The reference cannot represent most of the mRNA in the sample.

This was confirmed directly with a **reference-free** method. CDSeq (Reduce-Recover, K=14, 700 iterations,
11,319 genes × 50 samples — exactly the BLOOD `pred_z` dimensions) was run on the real blood bulk:

- Its 14 recovered components are **mutually distinct** (median off-diagonal r = −0.094, against −0.081 for
  the real reference cell types among themselves) — CDSeq did not collapse or fail.
- **~80% of the bulk mRNA mass sits in components that match nothing in the PBMC reference** (r ≈ 0.06–0.10).
- Marker genes name those components: **erythroid** (`Slc25a37`, `Epb41`, `Epb42`, `Car2` — two components,
  ~21% combined) and **granulocyte** (`S100a9`, `Mcemp1`, `Chi3l1`, `Lrg1`, `Siglec8` — force-fit onto
  "classical monocytes").
- The one component that matches cleanly is platelet/megakaryocyte (`Itgb3`, `Alox12`, `Treml1`, `Cd9`;
  r = 0.41) — **and the reference has megakaryocytes.** When the basis vector exists, the method works.

Corroborating evidence: every reference-based method inverts on blood (Spearman of θ against reference
abundance ≈ −0.26 to −0.40; MuSiC failed outright), and BayesPrism's `theta.first` (before the reference
update) is 56.4% versus `theta.final` 55.3% — **a reference update refines the profiles you supplied; it
cannot invent one you did not.**

**Correction to the written record.** An earlier claim in this project held that "the globin sink is
refuted because hemoglobin genes are stripped from the gene axis." **That claim was wrong and is
retracted.** Stripping globin *genes* does not remove erythroid *cells*, which contribute mRNA across
thousands of non-globin transcripts. Independent support: only **2** hemoglobin genes were ever on the
22,050-gene bulk axis to strip (of the 8 in `rat_exclude_genes.tsv`) — removing two genes cannot remove a
cell population. The correct diagnosis is **omitted-component misspecification**: θ is identified, and the
model converges confidently to a wrong answer because the truth lies outside the span of the basis.

**Consequence:** seven of the 21 hotspots are BLOOD hotspots, and they rest on this basis. See §8.

### 6.2 WATSC — omitted component (adipocytes) plus arm pooling

Two independent defects.

**(a) No adipocytes.** The roster is 43.8% macrophages and 33.2% fibroblasts; `Adipoq`, `Plin1`, `Lep` and
`Cidec` are at floor in **every** cluster. Mature adipocytes float and lyse in droplet scRNA — this is a
known, expected artefact of the assay. For bulk white adipose, adipocytes are the dominant mRNA source. So
the single largest mRNA component of the tissue has **no basis vector**, and its mass is projected onto
whichever surrogate profile fits best.

The fingerprint is unmistakable, and it is what produces the fake sex effect:

| WATSC cell type | female θ % | male θ % | M − F |
|---|---:|---:|---:|
| Luminal epithelial cells | 19.25 | 56.54 | **+37.30** |
| Fibroblasts | 39.49 | 19.26 | **−20.23** |
| Endothelial cells | 19.51 | 9.78 | −9.73 |

"Luminal epithelial" is *mammary* epithelium — and it is **higher in males than females**, the exact
opposite of any real mammary signal. The two "dominant" types simply **trade places by sex**, because
male and female adipocyte transcriptomes differ and therefore project onto different surrogates. **Any
downstream claim of a sex difference in WAT composition from these θ is spurious.** This also explains the
±22-point standard deviations in §4.10.

**(b) All three study arms pooled.** The reference mixes 38.4% young ad-lib / 31.2% old ad-lib / 30.4% old
+ calorie-restricted (`build_reference.py` has no age or diet filter). MoTrPAC rats are ~6-month adults.
The per-sample clusters are arm-pure, but the type-level ψ is an average over young, aged and
calorie-restricted adipose.

**No arm of GSE137869 can fix (a).** WATSC needs an **snRNA** adipose reference, which the corpus does not
currently contain.

### 6.3 HIPPOC — an ambient/debris cluster acting as a mass sink

The reference is clean (§3.8); the *annotation* is not. The 482-cell cluster labelled "Intermediate
monocytes" takes 63.6% of hippocampal mRNA while excitatory neurons collapse to 0.02%. Checking that
cluster against the reference matrix:

| HIPPOC reference cluster | cells | median UMIs | monocyte-marker % of UMIs | neuron-marker % |
|---|---:|---:|---:|---:|
| Microglia (the real myeloid cells) | 2,258 | 1,672 | **0.309** | 0.076 |
| **"Intermediate monocytes"** | **482** | **949** | **0.017** | 0.263 |
| Excitatory neurons | 8,124 | 10,470 | 0.003 | 0.539 |

Monocyte markers (`Cd14`, `Lyz2`, `Csf1r`, `Aif1`, `Ptprc`, `Cd68`, `Mrc1`, `Cx3cr1`) are **18× lower** in
the "monocyte" cluster than in actual microglia. It is not monocytes. It is the **lowest-depth cluster in
the reference** (median 949 UMIs) and its expression is a blend of oligodendrocyte and neuron transcripts:
an **ambient-RNA / debris cluster** to which an automated marker-based annotator assigned a nonsense immune
label.

**Why it is a near-perfect sink:** an ambient profile is approximately the *tissue-average* transcriptome,
i.e. approximately a linear combination of the real cell types. In a non-negative deconvolution, such a
basis vector is collinear with the mixture itself and can absorb almost unlimited mass — annihilating the
true parenchyma. This is **collinearity / approximate non-identifiability**, not an omitted component. (The
*mechanism* is an inference; the marker and depth evidence above is measured.)

**CORTEX is the control that proves the pipeline can do brain:** same label scheme, same code, sane roster.
Cortex also carries nonsense labels ("Crypt cells", "Myelinating Schwann cells") — but they stay at ≤0.4%.
Junk labels are only catastrophic when the junk cluster is *ambient-like*.

**The validation blind spot, stated explicitly:** HIPPOC scored r ≈ 0.82 on pseudobulk validation and
**0.9997 on its purity sweep — the highest score of any tissue.** Pseudobulk is generated from the same
reference, so the ambient cluster is perfectly self-consistent there and the pathology is invisible. It
only appears on real bulk. **Self-consistency validation does not certify real-bulk performance.** This is
the single most important methodological lesson in this document.

### 6.4 CORTEX — wrong-species basis

85% of the reference cells are cat, rabbit, zebrafish, cow or horse (§3.7). Fourteen of the 28 cell types
contain no rat cells at all. The θ is plausible; the basis is not rat. The recorded "Fix 2 = build bug"
diagnosis is wrong: 5,536 was the six-species gene intersection, and `--gene-join outer` hid the cause
rather than removing it. A rat-only rebuild (2 GSMs, 12,933 cells) would give 21,003 genes — *more* than
the current "fixed" reference — from clean data.

**Corollary:** CORTEX must be removed from the "our cleanest references come from atlases" claim. That list
is **HEART and LIVER**, full stop.

### 6.5 SKMVL — chimeric basis, no validation

52% of the vastus-lateralis reference is a rat–mouse blastocyst chimera from an iPSC engineering study
(§3.5), and SKMVL is absent from the omnideconv validation panel entirely. It nonetheless produces our
**four highest-AUC hotspots** (0.935, 0.925, 0.890, 0.883). Those hotspots are on the weakest basis in the
set. An organism filter would not catch this sample — the inventory records it as *Rattus norvegicus*.

### 6.6 LUNG — three pooled disease models, a neonate, and a missing basis vector

The native pooled reference (§3.6) is a large improvement on the engineered `lung_GSE178405` it replaced,
but: all three source studies are injury/disease models, one of the four donors is a **P10 neonate**, and
**alveolar type I cells are absent from the basis**. Downstream, LUNG owns **22 of the 67 near-empty DE
blocks** and **13 of the 25 blocks with median library size below 10**, and **both** of its RIN/globin
checks are uninformative (§5.4).

### 6.7 HEART — saturation

99.58% cardiomyocytes. The reference is the best in the set; the output is at the ceiling. Absolute
composition is unusable. The one HEART hotspot (CD8+ T cells, AUC 0.857) sits on a cell type at **θ = 0.01%**
with a median library size of 2,325 and 8 significant dose genes — it is a hotspot on a floor-level cell
type and should be treated as provisional.

### 6.8 KIDNEY / SKMGN — no demonstrated failure, but thin support

KIDNEY's roster is plausible and its reference is a genuine control arm; there is simply no independent
corroboration on the real bulk. SKMGN rests on **two animals**, one of them aged, sampled by an assay that
under-collects the very myonuclei that dominate the bulk. Neither is broken; neither is confirmed.

### 6.9 Process defects (not tissue-specific)

- **The validation scorecard mixes trees column-wise.** The purity-sweep column reflects current references
  (July builds), but the omnideconv column reflects **June 1–30 runs on two superseded references**:
  `OMNIDECONV_RESULTS.md:52` reads literally `| lung | lung_GSE178405 | LNG_cross | already matched |` —
  i.e. the widely-quoted "LUNG omnideconv 0.156 / 0.705, the panel's worst" describes the **retired
  engineered** reference, not production `lung_native_pooled` (built 2026-07-01). Likewise
  `validation_v2/LIV_holdout` "reused V0 holdout", so its 0.960 / 0.992 are **pre-novis**. Only the SimBu
  mRNA-bias battery was re-run on the corrected references (`simbu_bench/LIV_novis`, `simbu_bench/LNG_native`,
  both 2026-07-02). **Neither the LUNG nor the LIVER omnideconv number can be quoted as production QC.**
- **`SWEEP_heart_holdout/reference/summary.txt` reports 19,256 genes** while production
  `heart_GSE280111_LV` has 16,802 — the sweep's reference was built with different gene-filter parameters,
  so the HEART sweep is not validating the exact production basis. **UNEXPLAINED.**
- **Config default writes rebuilds into the wrong tree.** `config/pipeline_config.yaml:337` sets
  `built_reference_dir: data/deconvolution/references`, and `deconvolution/build_reference.py:304` defaults
  `--out` to it — but **five of the ten production references live in `references_v2/`**. A rebuild of
  cortex/hippoc/skmvl without an explicit `--out` lands in `references/`, on top of the stale Tau-included
  hippocampus and the 5,536-gene cortex, while production silently keeps reading `references_v2/`. **No
  code reads `tissue_references.yaml`.** This is a live footgun.
- **Stale directories a glob would silently ingest:** `references/hippocampus_GSE305314` (12 samples,
  **includes the Tau arm**), `references/cortex_GSE303115{,_union}`, `references_v2/cortex_GSE303115{,_merged}`,
  `references/heart_GSE155699` (SHR, no cardiomyocytes), `references/skeletal muscle_GSE254371`,
  `references{,_v2}/lung_GSE178405`, `references_v2/hippocampus_GSE305314_WT`. (Harmless duplicates:
  `gastrocnemius_GSE184413` and `kidney_GSE240658` are byte-identical across the two trees.)

### 6.10 Corrections to the written record

Where a previously-recorded claim is wrong, it is listed here rather than quietly dropped.

| # | Claim in the record | Status |
|---|---|---|
| 1 | "Liver cross-study purity sweep 0.949 **exceeds** the Chu 2022 threshold (≥0.95)" — `DECONVOLUTION_PIPELINE_REPORT.md:1023, 1173, 1188, 1847, 3253`; `AIM2_DECONV_RESULTS.md:21` | **WRONG.** 0.9490 < 0.95 is a **FAIL**. And the sweep ran on the Visium-contaminated reference and was never re-run. |
| 2 | "The 9 undeconvolved tissues have **no exercise metadata**" — `DECONVOLUTION_PIPELINE_REPORT.md:3234` | **WRONG.** All 9 have 100% phenotype coverage and the identical design. The blocker is reference availability (§2.3). |
| 3 | "Fix 2: cortex's 5,536 genes was a **build bug** (intersection join), not shallow data" | **WRONG DIAGNOSIS.** 5,536 is exactly the six-species gene intersection. The reference was, and still is, 85% non-rat (§3.7, §6.4). |
| 4 | "The globin sink is refuted because hemoglobin genes are stripped" | **WRONG, retracted.** Stripping globin genes does not remove erythroid cells (§6.1). |
| 5 | "Our three cleanest references come from healthy atlases: CORTEX, HEART, LIVER" | **WRONG on cortex.** The list is **HEART and LIVER**. |
| 6 | Reported cell counts for CORTEX / HEART / SKMGN | **2× inflated everywhere they appear** (§3.0). |
| 7 | `DECONVOLUTION_PIPELINE_REPORT.md` (~3225-3234) reference table | Stale on **three** references: `lung_GSE178405` (retired), `hippocampus_GSE305314_WT` (production is `_WT_merged`), `cortex_GSE303115_union` with 35 cell types (production is `_union_merged` with 28). |
| 8 | "8 of 10 tissues meet the purity bar" | **Misreading of coverage as pass-rate.** 8 of 10 were *swept at all*; BLOOD and WATSC were **never swept** (§5.1). |
| 9 | "The 2 degenerate KIDNEY DE blocks (median libsize 51 and 10)" | **Drastic undercount.** 67 of 185 blocks have median libsize < 50; 25 < 10; minimum is **3.03**. The KIDNEY pair is typical, not exceptional (§5.3). |
| 10 | "The dominant parenchyma fails its positive controls" | **Partly a coverage failure.** 3 of the Tier-A anchors (SKMVL `Sod2`, HEART `Hspa1b`, `Hsp90aa1`) were never testable — absent from the reference gene set (§5.5). |
| 11 | "LUNG Myeloid DC: 17 dose-significant genes, retain_frac 0" | **Two tables conflated.** The RIN test had `n_dose_sig_base = 1` — retain_frac is 0 of **1** (§5.4). |
| 12 | LUNG omnideconv "0.156 / 0.705, the panel's worst" | Describes the **retired** `lung_GSE178405`, not production (§6.9). |

---

## 7. The two failure mechanisms

Section 6 keeps returning to two distinct pathologies. They are routinely conflated, and the fixes are
different, so they must be kept apart.

### 7.1 Omitted component (model misspecification)

**A cell type that is present in the bulk is absent from the reference basis.** Its mRNA cannot be
assigned to nothing — the model is constrained to a simplex — so it is projected onto whichever available
profile is nearest. The result is a θ that is *confidently* wrong: it is a well-identified answer to the
wrong question.

**Confirmed instances:** BLOOD (erythroid + granulocytes, ~80% of the bulk mRNA mass — proven with
reference-free CDSeq), WATSC (adipocytes), LUNG (alveolar type I).

**Diagnostic signature:** the surrogate that absorbs the mass is often a *tiny* cluster (a 74-cell ISG-T
cluster taking 55% of blood; a 155-cell mammary cluster taking 38% of adipose), and *which* surrogate wins
is unstable across covariates — the WATSC surrogate flips with sex, which is exactly why its "sex effect"
is fake.

**Fix:** add the missing cell type to the reference. Nothing else works. In particular, **the BayesPrism
reference-update step cannot help**: it refines profiles you supplied, it cannot invent a basis vector you
never gave it (BLOOD `theta.first` = 56.4% vs `theta.final` = 55.3% — the update moved the answer by one
point).

### 7.2 Collinearity (approximate non-identifiability)

**The reference contains profiles that are near-linear-combinations of each other.** The *sum* of the
collinear group is identified; the split between them is not. Small perturbations move mass arbitrarily
within the group.

**Confirmed instances:** HIPPOC's ambient/debris cluster (§6.3) — the extreme case, because an ambient
profile is approximately the tissue average and therefore approximately collinear with *the entire
mixture*, letting it absorb 64% of the mass. Milder instances: SKMGN/SKMVL `Fibroblasts` vs `Muscle
fibroblasts` (unmitigated); KIDNEY's four near-duplicate intercalated labels; brain neuron fragments (this
one *was* mitigated, via `--label-scheme brain`).

**Fix:** merge the collinear labels, or report the pooled group and never the split.

### 7.3 Neither of these is mRNA-content bias

mRNA-content "bias" — the fact that hepatocytes and cardiomyocytes take a much larger θ than their cell
fraction — is **not an error**. θ is defined as a share of mRNA, not of cells. It is a definitional feature
of the quantity being estimated. It is precisely *why* differential and across-condition claims remain safe
where absolute fractions do not: the bias is a roughly fixed multiplier per cell type, so it largely cancels
when comparing the same cell type across conditions, and does not cancel at all when reading an absolute
percentage.

### 7.4 A third category, newly named: wrong-species basis

CORTEX (85% non-rat) and SKMVL (52% rat–mouse chimera) belong to neither of the above. Their profiles are
neither missing nor collinear — they are **measured in the wrong animal**. This category has no established
diagnostic and, notably, **no QC in this project can currently detect it**: `reference_qc.py` never reads
`geo_organism`, and for SKMVL even an organism filter would fail, because GEO labels the chimera as
*Rattus norvegicus*.

---

## 8. What is safe to claim, and what is not

### Safe

- **Differential / across-condition claims**: does a cell type's expression change with training dose, or
  differ by sex? The mRNA-content multiplier largely cancels; the reference's cell-type *profile* is used
  identically in every sample.
- **Immune and stromal cell types**, in tissues whose reference is not omitted-component-broken. These are
  usually mid-abundance and transfer reliably (per-type r ≈ 0.9–0.99 in the validation panel).
- **LIVER absolute composition.** The one tissue with a healthy atlas reference, a corrected build, and
  real-bulk cross-method agreement (§4.1).
- **Embedding-based downstream work** (GeneCompass hotspots, cross-species transfer): the tokenizer's
  target-sum normalisation insulates these from absolute-fraction errors.

### Not safe

- **Any absolute composition figure outside LIVER.** In particular: HEART's 99.58% cardiomyocytes (ceiling),
  BLOOD's 55% ISG-T (misspecified), HIPPOC's 64% monocytes (ambient sink), WATSC's 38% luminal epithelial
  (adipocyte mass in disguise), CORTEX's roster (wrong-species basis), LUNG's roster (missing AT1).
- **Any WAT sex-composition claim.** It is an artefact of the missing adipocyte basis (§6.2).
- **Any HIPPOC or CORTEX result presented as rat brain biology** until those references are rebuilt.
- **Dominant-parenchyma DE in general.** The known dominant-parenchyma failure mode applies, and three of
  the Tier-A parenchyma anchors were never even testable (§5.5).

### The 21 hotspots, triaged

| Risk | Hotspots | Why |
|---|---|---|
| **Highest risk — do not headline** | LUNG Myeloid dendritic cells | Triple-flagged: hotspot AUC 0.833 on a block with **median library size 21.6**, and a RIN/globin `retain_frac` of exactly 0 (of 1). It should not survive as a headline. |
| | LUNG Alveolar macrophages | Its RIN/globin `verdict = ROBUST` is **vacuous** (`n_dose_sig_base = 0`). θ = 0.03%. |
| | HEART CD8+ T cells | θ = 0.01%, 8 dose genes, RIN verdict `CHECK`. |
| | KIDNEY Proximal tubule cells | Only 5 dose-significant genes; RIN verdict `CHECK`; dominant parenchyma. |
| **Composition-confounded** | BLOOD NK, BLOOD Naive B, SKMVL Endothelial, SKMVL B, SKMGN Monocytes, SKMGN Naive B | **6 of the 21 hotspots carry `FLAG_COMPOSITION`** — the expression change is not separable from a θ change (§5.4). |
| **Basis-compromised** | all 7 BLOOD hotspots | Rest on a PBMC basis against whole-blood bulk (§6.1). Blood is our single largest hotspot tissue. |
| | all 4 SKMVL hotspots | Rest on a 52%-chimera basis with no validation (§6.5). |
| **Most defensible** | SKMGN Skeletal muscle cells (0.878), SKMGN Smooth muscle (0.818), LUNG Pulmonary fibroblasts (0.838) | Large blocks (median libsize 10⁶–10⁷), quiet composition verdicts, plausible references. Note SKMGN still rests on 2 donors. |

The honest summary is uncomfortable but simple: **the exercise signal is concentrated in exactly the two
tissues (BLOOD, SKMVL) whose references are weakest.** That does not mean the signal is false — the
supervised AUCs are computed on held-out samples and the blocks are large — but it does mean the *cell-type
attribution* of those signals is the thing at risk, not their existence.

---

## 9. Open questions and what would fix each broken tissue

| # | Question / defect | What would resolve it | Cost |
|---|---|---|---|
| 1 | **BLOOD**: no erythroid or granulocyte basis | A **whole-blood** (not PBMC) rat scRNA reference. The corpus does not currently have one — this needs a data search, and if none exists, blood cannot be deconvolved reference-based. CDSeq (reference-free) already recovers the components and is a viable fallback. | Data search; possible new source |
| 2 | **WATSC**: no adipocytes | An **snRNA** rat adipose reference. No arm of GSE137869 can fix this — droplet scRNA never captures adipocytes. | Data search |
| 3 | **HIPPOC**: ambient/debris cluster | Drop or merge the "Intermediate monocytes" cluster and rebuild; re-run stage 8–10 for HIPPOC. This is the cheapest high-value fix in the list. | 1 rebuild + rerun |
| 4 | **CORTEX**: 85% non-rat | Rebuild rat-only (2 GSMs, 12,933 cells, 21,003 genes, inner join). Requires adding an organism filter to `select_samples()`. | 1 rebuild + rerun |
| 5 | **SKMVL**: 52% chimera | Rebuild from `SD rat_muscle` alone (9,797 cells) and accept a single-donor reference, **or** find a native rat vastus-lateralis scRNA study. Either way, SKMVL needs to enter the validation panel. | Rebuild ± data search |
| 6 | **LUNG**: no alveolar type I | Find a native adult rat lung study with AT1 in the roster. | Data search |
| 7 | **LIVER**: no valid Z-recovery number | Re-run `SWEEP_hepato_{holdout,cross}` on the clean (novis) reference. Until then "we meet Chu 2022" is unsupported for the shipped reference, and the cross number we do have (0.9490) is a fail. | 1 compute job |
| 8 | **LUNG / LIVER**: omnideconv panel is stale | Re-run the correlation panel on `lung_native_pooled` and the novis liver. The current numbers describe retired references. | 1 compute job |
| 9 | **BLOOD / WATSC**: never purity-swept | Run the sweeps. (They will likely pass — self-consistency is blind to the actual defect — but the coverage gap should be closed and the result reported honestly.) | 1 compute job |
| 10 | **Reference QC has three structural holes** | Extend `reference_qc.py` to read `geo_organism`, to de-duplicate on `gsm`, and to consult `master_catalog.json` `studies[]` for study type and donor count. | Small code change |
| 11 | **`build_reference.py` double-counts and ignores organism** | Add `--organism` (default `Rattus norvegicus`) and GSM-level dedup to `select_samples()`. **Note this would not catch SKMVL** — GEO labels the chimera as rat. | Small code change |
| 12 | **DE has no block gating** | Gate blocks on median library size before the pooled IHW/repfdr fit, or fit FDR per tissue. 67 of 185 blocks currently below libsize 50. | Small code change + rerun |
| 13 | **The `Rps6k*` family is silently excluded** | Fix the `^RP[SL]` regex in `rat_exclude_genes.tsv` and re-run. Whether any DE conclusion changes is **UNKNOWN**. | 1 rerun |
| 14 | Config default writes rebuilds into `references/` while 5 of 10 production refs live in `references_v2/` | Make `tissue_references.yaml` machine-read, or unify the trees. | Small code change |
| 15 | **BAT is deconvolvable and never was** | GSE137869 has 6 BAT samples; same build recipe as WATSC. It would inherit WATSC's defects (pooled arms; no adipocytes). | 1 build |
| 16 | **UNKNOWN — the WAT depot.** Ma 2020 never states whether its WAT is subcutaneous or visceral. If visceral, WATSC is the wrong depot. | Read the Cell STAR Methods (paywalled) or contact the authors. | — |
| 17 | **UNVERIFIED — are the duplicated GSM pairs byte-identical?** | If yes, ψ is unaffected and only cell counts are inflated. If no, ψ *is* affected for CORTEX, HEART, SKMGN. | 1 compute check |
| 18 | **UNEXPLAINED — `SWEEP_heart_holdout` used a 19,256-gene reference** vs production's 16,802. | Chase the build args. | Small check |

---

## 10. Provenance — every file this document was built from

Paths are relative to the repository root, `/depot/reese18/apps/motrpac-genecompass`, unless absolute.

**Manifest and configuration**
- `deconvolution/tissue_references.yaml` (v2026-07-02) — the authoritative tissue → reference map
- `config/pipeline_config.yaml` (`:337` `built_reference_dir`, `:340` `results_dir`)
- `pipeline/run_stage10.py:143-144`

**Bulk data and design**
- `data/deconvolution/motrpac_bulk/<TIS>/{bulk.mtx, bulk_genes.tsv, bulk_samples.tsv}` — all 19 tissues
- `deconvolution/reference/motrpac_sample_pheno.tsv` — 6,156 vials
- `deconvolution/reference/motrpac_bulk_liftover_report.txt`
- `deconvolution/reference/canonical_references.tsv`
- `deconvolution/reference/rat_exclude_genes.tsv` — 247 genes

**Reference provenance**
- `reports/annotations/annotation_inventory.tsv` — **the authoritative sample_id → gsm / geo_title / geo_organism map**
- `/depot/reese18/data/catalog/master_catalog.json` — `studies[]`
- `data/deconvolution/references/{peripheral blood mononuclear cells_GSE285476, heart_GSE280111_LV, liver_GSE220075, lung_native_pooled, white adipose tissue_GSE137869}/{summary.txt, cells_meta.tsv}`
- `data/deconvolution/references_v2/{cortex_GSE303115_union_merged, hippocampus_GSE305314_WT_merged, kidney_GSE240658, gastrocnemius_GSE184413, skeletal_muscle_GSE254371_muscle_merged}/{summary.txt, cells_meta.tsv}`

**Proportions**
- `data/deconvolution/results/motrpac/<TIS>/estimated_fractions.csv` — all 10 tissues
- `data/deconvolution/results/motrpac/<TIS>/pred_z/` — per-cell-type expression, gene lists
- `data/deconvolution/results/motrpac/{LIVER,BLOOD}/omnideconv/fractions_{dwls,music,scdc}.csv` — the only real-bulk cross-method runs
- `tmp/proportions_dump.md` — the working dump with full by-sex / by-intervention / by-dose tables for every cell type in every tissue *(scratch; not committed)*

**Validation and QC**
- `data/deconvolution/validation/SWEEP_*/scores/{purity_sweep_summary.tsv, z_vst_focal.tsv}`
- `data/deconvolution/validation/SWEEP_*/reference/{cells_meta.tsv, summary.txt}`
- `data/deconvolution/validation/{PBMC_holdout, WAT_holdout}/scores/` — **note: no purity_sweep_summary.tsv**
- `data/deconvolution/validation/purity_sweep_manifest{,_ext}.tsv`
- `reports/deconvolution/liver_expression_Z_purity_sweep.md`
- `deconvolution/OMNIDECONV_RESULTS.md` (markdown only — **no TSV exists**; `:52` names the retired lung reference)
- `data/deconvolution/simbu_bench/{LIV_novis, LNG_native}/` — the only battery re-run on corrected refs

**Differential expression**
- `data/deconvolution/genecompass_input/pseudobulk_de/{de_summary.tsv, de_hotspots.tsv, composition_confound_table.tsv, rin_globin_robustness.tsv, posctrl_results.tsv, dominant_celltype_flags.tsv}`

**Code**
- `deconvolution/build_reference.py` (`select_samples()`, `:304`), `deconvolution/build_lung_pooled.py`, `deconvolution/build_all_references.sh`
- `deconvolution/reference_qc.py`
- `deconvolution/R/{prepare_motrpac_bulk.R:168-170, run_deconvolution.R:97, run_pseudobulk_de.R}`

**Documents corrected by this one**
- `deconvolution/DECONVOLUTION_PIPELINE_REPORT.md` (`:1023, :1173, :1188, :1847, :3225-3234, :3253`)
- `deconvolution/AIM2_DECONV_RESULTS.md` (`:21`)

> **CORRECTION 2026-07-17:** the 13-hotspot figure above was a stale-join ARTIFACT (Stage 10 was run before the detection layer `redetect_redE`, so newly-merged labels had no AUC row). The authoritative correct-order re-run gives **15 hotspots / 172 blocks**, muscle myofiber RECOVERED as #1 (SKM-GN Skeletal myocytes AUC 0.893). See `project_deposited_label_adoption_2026-07-16` memory.
