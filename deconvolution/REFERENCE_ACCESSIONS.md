# Reference accessions — source records

Every single-cell study behind a MoTrPAC deconvolution reference, linked to its source record.
Companion to [`TISSUE_REFERENCE_AND_PROPORTIONS.md`](TISSUE_REFERENCE_AND_PROPORTIONS.md) (full audit)
and [`REFERENCE_STUDY_GSE137869_MA2020.md`](REFERENCE_STUDY_GSE137869_MA2020.md) (WATSC dossier).

Manifest of record: `deconvolution/tissue_references.yaml`.

---

## 1. Production references (10 tissues, 12 accessions)

| Tissue | Accession | Link | What the study actually is | Status |
|---|---|---|---|---|
| **LIVER** | GSE220075 | [GEO](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE220075) | Rat liver cell **atlas** (myeloid heterogeneity) | ✅ CLEAN — Visium samples excluded |
| **KIDNEY** | GSE240658 | [GEO](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE240658) | Fasting-mimicking-diet study; we take the "No treatment" arm | CAUTION |
| **HEART** | GSE280111 | [GEO](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE280111) | Rat cardiovascular **atlas** (left ventricle) | CAUTION — θ saturated (CM 99.6%) |
| **SKMGN** | GSE184413 | [GEO](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE184413) | Mechanotherapy in **aged** rats recovering from **disuse**; "Normal ambulation" arm | CAUTION — 2 donors |
| **SKMVL** | GSE254371 | [GEO](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE254371) | Muscle stem-cell **transplantation** engineering study | ⚠️ **52% of the reference is a `Rat-mouse chimera_muscle` sample.** GEO labels it *Rattus norvegicus* — an organism gate will NOT catch it. |
| **LUNG** (1/3) | GSE273062 | [GEO](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE273062) | **Pulmonary hypertension** model (SuHx / VeNx control) | CAUTION |
| **LUNG** (2/3) | GSE252844 | [GEO](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE252844) | **Blast-exposure lung injury** | CAUTION |
| **LUNG** (3/3) | GSE242310 | [GEO](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE242310) | **Hyperoxic NEONATAL** lung — even the control arm is developmentally immature | CAUTION |
| **CORTEX** | GSE303115 | [GEO](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE303115) | **Six-species** comparative brain multiome (cat, cow, horse, rabbit, zebrafish, rat) | 🔴 **BROKEN — 85% of the reference is not rat.** Only 2 of 11 GSMs are rat. |
| **HIPPOC** | GSE305314 | [GEO](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE305314) | WT arm of an **Alzheimer's / tauopathy** model | 🔴 BROKEN — a debris cluster absorbs 64% of θ. Ref itself verified WT-only, ages 10 & 20 mo. |
| **BLOOD** | GSE285476 | [GEO](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE285476) | Liver-allograft-**rejection** study; 1 healthy-control donor | 🔴 BROKEN — **PBMC** reference vs **whole-blood** bulk |
| **WATSC** | GSE137869 | [GEO](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE137869) · [Cell 2020](https://pubmed.ncbi.nlm.nih.gov/32109414/) | Ma 2020 — **aging + caloric restriction** 9-tissue atlas | 🔴 BROKEN — **no adipocytes**; all 3 arms pooled (61% geriatric) |

⚠️ **The organism column is not a sufficient gate.** `geo_organism` says *Rattus norvegicus* for the
GSE254371 chimera sample. Any species check must also read the free-text `geo_title`.

---

## 2. Candidate replacements / additions (identified, not adopted)

| For | Accession | Link | Why it matters | Blocker |
|---|---|---|---|---|
| **TESTES** (never deconvolved) | **OMIX767** | [NGDC/CNCB](https://ngdc.cncb.ac.cn/omix/release/OMIX767) · [Sci Data 2022](https://www.nature.com/articles/s41597-022-01225-5) · [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC8956705/) | First rat testis scRNA-seq. **Has a true control arm** (normal testosterone) **and the full germ-cell series, spermatogonia → spermatozoa.** Open access, processed MTX. | Only **10,983 cells across 4 conditions** → control arm may be ~2–3k. EDS Leydig-ablation study. |
| **LUNG** | GSE133747 | [GEO](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE133747) | *"Single-cell connectomic analysis of **adult mammalian lungs**"* — a genuine **healthy adult** lung atlas (mouse/rat/pig/human), n=20. The reference lung actually needs. | **No processed count matrix** — raw only. This is why it was skipped. ⚠️ Multi-species — must filter to rat. |
| **LUNG** | GSE312833 | [GEO](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE312833) | Glenn vs **sham** surgery — the sham arm is adult healthy lung. Has a count matrix. | Surgical model |
| **HIPPOC** | GSE295314 | [GEO](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE295314) | snRNA of ventral hippocampus from **healthy rats** (feeding/memory paradigm), n=12, count matrix. Far better provenance than a tauopathy model. | *Ventral* hippocampus specifically; fasted/fed state |
| **BAT** (never deconvolved) | GSE137869 | [GEO](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE137869) | MoTrPAC **has BAT bulk**; Ma 2020 has an exact-match BAT scRNA factorial (Y/O/CR × M/F). | ⚠️ Same collagenase protocol → likely the **same missing-adipocyte hole** as WAT. Check before building. |
| **BLOOD** | — | — | **No adequate replacement exists.** No healthy whole-PBMC rat atlas in the 2,670-study catalog. What blood actually needs is a reference with **erythroid + granulocyte** compartments — that search has not been run. | — |
| **WATSC** | — | — | **No adequate replacement exists.** GSE137869 is the only rat scRNA study containing subcutaneous WAT. A real fix needs an **snRNA** adipose reference (nuclei survive what whole adipocytes do not). | — |

---

## 3. Not used (checked, ruled out)

| Accession | Link | Why ruled out |
|---|---|---|
| GSE134705 | [GEO](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE134705) | *"Cross-species analysis across 450 million years"* — never entered the corpus (0 rows in the inventory). Would have been a second species-contamination vector. |
| GSE178405 | [GEO](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE178405) | **Superseded.** In-vitro tissue-**engineering** study (engineered d7 / tri-culture / P7-developing) — not native adult lung. Replaced by the 3-study pooled lung reference. |
| GSE155699 | [GEO](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE155699) | **Superseded** by GSE280111 (LV atlas) for HEART. |

---

## 4. The MoTrPAC bulk (19 tissues, 10 deconvolved)

MoTrPAC PASS1B bulk RNA-seq: `data/deconvolution/motrpac_bulk/`. Design is 2 sex × 5 group × 5 replicates.

**Deconvolved (10):** BLOOD · CORTEX · HEART · HIPPOC · KIDNEY · LIVER · LUNG · SKMGN · SKMVL · WATSC

**Not deconvolved (9):** ADRNL (adrenal) · **BAT** (brown adipose) · COLON · HYPOTH (hypothalamus) ·
OVARY · SMLINT (small intestine) · SPLEEN · **TESTES** · VENACV (vena cava)

The two with live reference candidates are **BAT** (Ma 2020, exact tissue match — but check the adipocyte
hole first) and **TESTES** (OMIX767, §2).
