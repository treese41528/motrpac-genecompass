# Reference study dossier — GSE137869 (Ma et al., *Cell* 2020)

**Caloric Restriction Reprograms the Single-Cell Transcriptional Landscape of *Rattus Norvegicus* Aging**
Ma S, Sun S, Geng L, Song M, *et al.* **Cell** 2020 Mar 5;180(5):984–1001.e22. **PMID 32109414.**

This is the source of our **white adipose (WATSC)** deconvolution reference — and, since the 2026-07-16
rebuild, also of the shared **skeletal-muscle (SKMGN + SKMVL)** reference (young `-Y` arm; myofiber
over-split merged → 5 clean types; see §6). It is the only reference in the panel drawn from a
*multi-tissue* atlas. It deserves its own dossier for three reasons: it is the reference behind one of our
confirmed BROKEN tissues (WATSC); almost every design axis is mismatched to MoTrPAC; and it covers tissues
we are **not** using it for but should consider.

Companion: [`TISSUE_REFERENCE_AND_PROPORTIONS.md`](TISSUE_REFERENCE_AND_PROPORTIONS.md) §3 (WATSC), §6.

**Sources for everything below.** The paper PDF (`readings/PIIS0092867420301525.pdf`, 41 pages, 133,255
characters — searched in full, STAR Methods included); the GEO deposit
(`/depot/reese18/data/geo/geo_datasets/GSE137869/`, incl. `GSE137869_family.soft.gz` and
`manifest.json`); and the built reference
(`data/deconvolution/references/white adipose tissue_GSE137869/`).

---

## 1. The study design, from the paper's own words

> *"Tissues were isolated from randomly selected **Y-AL (5-month-old**, n = 6, 3 male and 3 female rats),
> **O-AL (27-month-old**, n = 6, 2 male and 4 female rats), and **O-CR (27-month-old**, n = 6, 3 male and
> 3 female rats) animals **after perfusion with normal saline**."*

| arm | code | age | diet | n rats (M/F) |
|---|---|---|---|---|
| young, ad libitum | `Y-AL` | **5 months** | ad libitum | 6 (3/3) |
| old, ad libitum | `O-AL` | **27 months** | ad libitum | 6 (2/4) ⚠️ sex-imbalanced |
| old, calorie-restricted | `O-CR` | **27 months** | **70% CR**, 9 months, begun at 18 months | 6 (3/3) |

The authors describe the 27-month animals as *"analogous to 70 years"* in humans. **There is no young-CR
arm**, so age and diet are partially confounded by design — you cannot cleanly separate an "old" effect
from a "CR" effect in this dataset even if you wanted to.

**Strain: Sprague-Dawley** ("SD rats", GEO `extract_protocol`; Beijing Vital River Laboratory).

**Critical structural limitation — no biological replication at the library level.** GEO's
`extract_protocol` states *"Pooled samples of 2 to 3 rats"* per library, and the sample tokens in
`manifest.json` resolve to exactly **9 tissues × 2 sexes × 3 arms = 54 rat libraries** — i.e. **one library
per (tissue × sex × arm)**. Any reference derived from this study rests on a **single pooled library** per
cell of the design.

Tissues were collected after **systemic saline perfusion through the heart**, which flushes circulating
blood — so blood contamination in Ma-derived references should be low.

---

## 2. The nine tissues, and how each was assayed

From the paper: *"seven scRNA-seq datasets (BAT, WAT, aorta, kidney, liver, skin, and BM) and two snRNA-seq
datasets on two additional tissues (brain and skeletal muscle)."* 210,000 cells/nuclei total.

| tissue | assay |
|---|---|
| Brown adipose (BAT) · White adipose (WAT) · Aorta · Kidney · Liver · Skin · Bone marrow (BM) | **scRNA-seq** |
| Brain · Skeletal muscle | **snRNA-seq** |

The assay split is the single most consequential fact in this document — see §4.

**The "human" samples are a mirage.** The catalog lists the study's organisms as *Homo sapiens; Rattus
norvegicus*, and the GEO file tokens include `GL2` and `YBX`. These are **bulk RNA-seq of human
adipose-derived stem cells** (`shYBX1` vs `shGL2` knockdown — the paper's follow-up experiment). They are
**not single-cell, not a tissue, and not usable as a human reference.** Six samples; ignore them.

---

## 3. How Ma 2020 differs from MoTrPAC — nearly every axis

| axis | Ma 2020 (GSE137869) | MoTrPAC PASS1B |
|---|---|---|
| **Strain** | **Sprague-Dawley** | **Fischer 344** |
| **Intervention** | **aging + caloric restriction** | **endurance exercise training** |
| **Arms** | `Y-AL`, `O-AL`, `O-CR` (no young-CR) | sedentary control vs trained |
| **Timepoints** | none — cross-sectional | 1w / 2w / 4w / 8w |
| **Age** | 5 months (young) / **27 months** (old) | **~6 months** (adult) |
| **Replication** | **1 library per tissue × sex × arm**; 2–3 rats *pooled* per library | 5 replicates per sex × group |
| **Assay** | scRNA (7 tissues) + snRNA (2) | bulk RNA-seq |
| **Tissues** | 9 | 19 |

The one axis that *matches*, and matters: **Ma's young arm is 5 months old; MoTrPAC's rats are ~6 months.**
That is an excellent age match, and it is why the young-ad-libitum-only rebuild (§5) is the obviously
correct fix rather than a judgment call.

### Tissue overlap with MoTrPAC's 19

| Ma tissue | MoTrPAC counterpart | verdict |
|---|---|---|
| Kidney | KIDNEY | **exact** |
| Liver | LIVER | **exact** |
| BAT | **BAT** | **exact — and MoTrPAC's BAT was never deconvolved** (see §6) |
| WAT | WATSC (subcutaneous) | **depot unstated in Ma** — see §4.3 |
| Skeletal muscle | SKMGN / SKMVL | **muscle group unstated** |
| Brain | CORTEX / HIPPOC / HYPOTH | **region unstated** — not a usable substitute |
| Aorta | VENACV | **no** — artery ≠ vein; and aorta has no cardiomyocytes, so it is not HEART either. (VENACV was **dropped 2026-07-16** — no genuine rat vena-cava reference exists anywhere.) |
| Bone marrow | BLOOD / SPLEEN | **no** — marrow carries progenitors (pro-B, erythroid precursors, MDSC) absent from circulation |
| Skin | — | no MoTrPAC counterpart |

**Ten of MoTrPAC's nineteen tissues have no Ma counterpart at all**: adrenal, blood, colon, heart, lung,
ovary, small intestine, spleen, testes, vena cava.

---

## 4. What is wrong with our WATSC reference

Built at `data/deconvolution/references/white adipose tissue_GSE137869/` —
**31,870 cells · 17,895 genes · 17 cell types · 96 cell states** (`summary.txt`).

### 4.1 It pools all three arms — 61% of it is geriatric tissue

`deconvolution/tissue_references.yaml:84` builds WATSC with
`build_reference.py --study GSE137869 --tissue "white adipose tissue"` — **no `--sample-ids`, no
`--conditions`, no age or diet argument.** `build_reference.py::select_samples()` filters on `accession`,
`tissue_normalized` and `in_corpus` only; **there is no age/diet axis anywhere in the file.** All six WAT
libraries are `in_corpus=True`, so all six go in.

Verified from `cells_meta.tsv` (sums to 31,870 exactly):

| sample | arm | age | cells |
|---|---|---|---:|
| `GSE137869_sample10` / `sample35` | `Y-AL` (M/F) | 5 mo | 6,130 / 6,093 |
| `GSE137869_sample15` / `sample11` | `O-AL` (M/F) | **27 mo** | 4,618 / 5,334 |
| `GSE137869_sample4` / `sample43` | `O-CR` (M/F) | **27 mo, 70% CR** | 4,174 / 5,521 |

**38.4% young ad-lib · 31.2% aged ad-lib · 30.4% aged + calorie-restricted.** So **~61% of the reference is
27-month-old (geriatric) tissue**, and half of *that* is on a severe caloric restriction — against MoTrPAC
rats that are ~6-month adults.

**Be precise about the mechanism.** The per-sample leiden **clusters are arm-pure** (`build_reference.py`
reads per-sample labels), so the *cluster definitions* are not contaminated. What is contaminated is **ψ,
the type-level reference expression profile** that θ and Z are estimated against: each of the 17 cell-type
labels averages young, aged and calorie-restricted cells. That is the expression-drift confound, one level
down from the clusters.

**This is structural, not measured.** Nothing on disk shows the pooled ψ actually moves WATSC's θ relative
to a young-only ψ. Rebuilding and diffing would settle it. **UNMEASURED.**

### 4.2 It contains no adipocytes — and the paper confirms this by design

This is the larger defect, and it is not our bug — it is intrinsic to the assay.

The roster is **43.8% macrophages (13,960) and 33.2% fibroblasts (10,586)**, with the rest immune and
endothelial. Per-leiden mean CP10K in both young WAT samples: **Adipoq ≤ 0.22, Plin1 ≤ 0.04, Lep ≤ 0.02,
Cidec ≤ 0.05 in *every* cluster** (against Col1a1 = 201 in fibroblasts). No adipocytes anywhere.

**The paper corroborates this independently, three ways:**

1. **The protocol excludes them by construction.** WAT was dissociated with *"collagenase I (2 mg/mL),
   collagenase IV (2 mg/mL) and dispase (2 mg/mL) at 37 °C for 1 h"* and run on droplet 10x. That is a
   **stromal-vascular-fraction** protocol — mature adipocytes are buoyant and are removed at the
   centrifugation step.
2. **The paper's WAT cell types are stromal**: *"`Pdgfra`+ **adipose-derived stem cells (ASCs)**"*,
   macrophages, T/B cells, neutrophils, endothelium.
3. **The word "adipocyte" never appears as a cluster** anywhere in the 41-page paper. Adipocyte *size* is
   discussed — from **histology**, not transcriptomics.

**Consequence.** For bulk white adipose, **adipocytes are the dominant mRNA source.** Our reference is
100% stromal-vascular fraction, so BayesPrism has no basis vector for the dominant compartment and must
project its mRNA onto the nearest available profile. This is a textbook **omitted-component
misspecification** (see `TISSUE_REFERENCE_AND_PROPORTIONS.md` §7) — the same class of failure as BLOOD.

It shows up exactly as theory predicts: a **155-cell "Luminal epithelial" cluster — 0.5% of the reference —
absorbs 37.9% of the bulk mRNA**, and *which* surrogate wins **flips with sex**, manufacturing a spurious
sex effect.

**No arm of GSE137869 can fix this.** A young-only rebuild fixes §4.1 and leaves §4.2 completely intact.
WATSC needs an **snRNA** adipose reference (nuclei survive what whole adipocytes do not) — and the catalog
does not contain one.

### 4.3 The depot is unknowable from the paper — ⚠️ UNRESOLVED

MoTrPAC's WATSC is **subcutaneous** white adipose. **Ma 2020 never states which depot it sampled.**

Searched across all 133,255 characters of the PDF, STAR Methods included:

| term | hits |
|---|---|
| subcutaneous · inguinal · epididymal · gonadal · perirenal · visceral · periovarian · omental | **0** |
| stromal vascular · SVF | **0** |

The tissue-collection paragraph says only *"WAT, BAT, and kidney tissues were incubated with dissociation
buffer…"* — no anatomical depot. GEO is equally silent. **This is a hard limit, not a gap in the search.**

**The one available inference, flagged as such:** every arm contains **both male and female** rats.
Epididymal fat is male-only and periovarian is female-only, so a gonadal depot would have forced
sex-specific dissections — which the authors never mention. That points toward a sex-neutral depot
(inguinal/subcutaneous), which is what we would want. **But this is an inference from silence, not a
statement, and WATSC's depot should not be considered resolved on it.** If Ma's WAT is visceral, WATSC is
the wrong depot entirely.

The same silence applies to the **muscle group** and the **brain region** — neither is ever specified.

### 4.4 A label-identity issue

Our reference calls **33.2% of its cells "Fibroblasts."** The paper's corresponding population is
**`Pdgfra`+ adipose-derived stem cells (ASCs)** — adipocyte progenitors. Our consensus annotator appears to
have renamed them. "Fibroblast" and "adipocyte progenitor" are not the same thing biologically, and any
WATSC result phrased in terms of fibroblasts should be read with that in mind. **Not independently
verified** — flagged for checking.

---

## 5. The fix, and what it does and does not buy

### 5.0 ⚠️ The arm was never missing — it was in `geo_title` all along

This is the important part, and it reframes §4.1 from a data limitation into a **code** one.

Every one of the **54/54** GSE137869 rows in `reports/annotations/annotation_inventory.tsv` carries the arm
in `geo_title`, in a trivially parseable three-token form:

| `sample_id` | `geo_title` | `condition_resolved` |
|---|---|---|
| `GSE137869_sample10` | **`WAT-M-Y`** | `NaN` |
| `GSE137869_sample35` | **`WAT-F-Y`** | `NaN` |
| `GSE137869_sample15` | **`WAT-M-O`** | `NaN` |
| `GSE137869_sample11` | **`WAT-F-O`** | `NaN` |
| `GSE137869_sample4` | **`WAT-M-CR`** | `NaN` |
| `GSE137869_sample43` | **`WAT-F-CR`** | `NaN` |

(GEO's own `characteristics_ch1.0.Stage` field says the same thing: `Y-AL` / `O-AL` / `O-CR`, 18 samples
each. `strain_resolved` is likewise correctly populated — `Sprague-Dawley`, 54/54.)

**`condition_resolved` — the one column `build_reference.py --conditions` filters on — is `NaN` for all
54 rows.** The arm was sitting one column over from the field the code reads.

**This is the third instance of a single systemic bug**, and the other two are documented elsewhere in this
audit:

| metadata field | state | consumed by `select_samples()`? | consequence |
|---|---|---|---|
| `geo_organism` | populated, correct | **never read** | CORTEX: six species admitted (see `TISSUE_REFERENCE_AND_PROPORTIONS.md`) |
| `condition_resolved` | **empty** | read (`--conditions`) | WATSC: `--conditions` unusable |
| `geo_title` (carries the arm) | populated, correct | **display only** | WATSC: all three arms pooled |

**And the lesson had already been learned.** The project's own standing note reads: *"Inventory's
`condition_resolved` misses healthy arms; pick deconvolution references by **regex on `geo_title`**, not
`condition_resolved` — this caused the hippocampus WT+Tau pooling."* That is exactly why **HIPPOC** carries
an explicit `--sample-ids` list and is clean today. **The fix was applied to hippocampus and never
propagated to WAT**, which was still being built by the naive `--study` + `--tissue` path.

**Systemic fix (do this, or it recurs on the next multi-arm study):** teach
`build_reference.py::select_samples()` to filter on `geo_title` (regex) and to gate on `geo_organism`, and
either populate `condition_resolved` from `geo_title` upstream or stop filtering on it. All three columns
already exist and are already correct.

### 5.1 The immediate fix

Given the *current* builder, `--conditions` cannot be used (the column it reads is empty), so
**`--sample-ids` is the only available lever.** Change `deconvolution/tissue_references.yaml:84` to:

```
build: build_reference.py --study GSE137869 --tissue "white adipose tissue" \
       --sample-ids "GSE137869_sample10 GSE137869_sample35"
```

→ young ad-lib only, both sexes, **12,223 cells** after `clean_cells`.

Cost: one `build_reference.py` run (minutes) plus one `run_deconvolution.R` on the WAT bulk (**sbatch — not
the login node**). Everything downstream of WATSC's θ/Z must be re-run: the WATSC pseudobulk DE blocks and
any WATSC hotspot or GRN claim. The other nine tissues are untouched — **except** that the pseudobulk DE's
IHW/repfdr fit is **pooled globally across all 185 blocks**, so changing WATSC's blocks perturbs adjusted
p-values in *every* tissue.

**Be clear about what this buys.** It makes ψ **honest** (young-adult, age-matched to MoTrPAC). It does
**not** make WATSC's absolute fractions trustworthy, because §4.2 is untouched and unfixable with this
study. **Differential/across-condition WAT claims remain defensible; absolute WAT composition does not.**

Recommended sequencing:
1. Rebuild young-AL-only and **diff θ against production** — this *measures* the aging/CR confound rather
   than assuming it (§4.1 is currently UNMEASURED).
2. Treat *"find an snRNA rat adipose reference"* as the actual WATSC fix.
3. Resolve the depot (§4.3) — it may invalidate the tissue mapping outright.

---

## 6. What we are **not** using this study for — and should consider

The mismatch cuts both ways, and one direction is a genuine missed opportunity.

As of the 2026-07-16 rebuild we use Ma 2020 for **WATSC and both skeletal muscles (SKMGN + SKMVL)** — WATSC
being the tissue it fits *worst* (unknown depot, no adipocytes, pooled arms), muscle the tissue it fits
*best* (young `-Y` snRNA, myonuclei captured; §6 muscle bullet). Meanwhile:

- **BAT — since deconvolved, but NOT via Ma.** MoTrPAC has **brown adipose bulk**
  (`data/deconvolution/motrpac_bulk/BAT/`) and Ma has **BAT scRNA** (exact tissue match, full Y/O/CR × M/F
  factorial), but Ma's BAT was dissociated with the *same* collagenase protocol → the *same* missing-adipocyte
  hole. **BAT is now deconvolved from GSE244451** (subscapular-BAT snRNA, the authors' deposited SCP-annotated
  labels → **6 clean types**), which recovers brown adipocytes — chosen over Ma precisely to avoid that hole.
- **Skeletal muscle. — ADOPTED as the production muscle reference (2026-07-16).** Ma's young-AL `-Y` muscle
  is **snRNA** (so it *does* contain myonuclei), sex-balanced and young-adult. It **replaced** the former
  incumbents — SKMGN's GSE184413 (aged disuse, 2 donors) and SKMVL's GSE254371 (52% rat-mouse chimera) — and
  now serves **both** muscles as one shared reference. The myofiber over-split was **merged → 5 clean cell
  types: Skeletal myocytes · Fibroblasts · Endothelial cells · Vascular smooth muscle cells · Macrophages**
  (`references_v3/MUSCLE_GSE137869_Y`, 10,763 cells). Caveat unchanged: its muscle group is unstated (§4.3),
  and dropping GSE184413 loses the only F344-strain / omnideconv-validated gastroc ref → head-to-head pending.
- **Kidney / Liver.** Exact tissue matches, but Ma is **scRNA** where our incumbents are **snRNA**, and
  snRNA is strictly better for fragile proximal tubule and hepatocyte parenchyma. **Cross-check only — do
  not replace.**

**Do not use Ma for**: BLOOD (bone marrow ≠ blood), HEART (aorta has no cardiomyocytes), LUNG (not
sampled), CORTEX/HIPPOC (brain region unspecified, and our labels for it are 60% oligodendrocyte with
Bergmann glia — a cerebellar type — present).

⚠️ **Live footgun.** `reports/annotations/annotation_inventory.tsv` already tags all six Ma **aorta** rows
as `motrpac_tissue_match = heart` and all six **bone marrow** rows as `blood`. Anyone who builds a
reference off that column instead of `tissue_normalized` gets a **cardiomyocyte-free heart** and a
**progenitor-laden blood**.

---

## 7. Open questions

| # | question | how to settle it | blocks |
|---|---|---|---|
| 1 | **Which WAT depot?** (§4.3) | Not in the paper or GEO. Email the corresponding author, or find a follow-up citing the deposit. | Whether WATSC's tissue mapping is valid *at all* |
| 2 | **Magnitude of the aging/CR confound** (§4.1) | Rebuild young-AL-only; re-run `run_deconvolution.R`; correlate per-cell-type θ | Whether the §5 fix matters in practice |
| 3 | **Is "Fibroblasts" actually `Pdgfra`+ ASCs?** (§4.4) | Check `Pdgfra` in the fibroblast clusters of `cells_meta.tsv` | How to phrase any WATSC stromal result |
| 4 | **Would Ma's BAT have the same adipocyte hole?** (§6) | Check `Ucp1`/`Adipoq`/`Plin1` in Ma's BAT samples before building | Whether BAT deconvolution is worth attempting |
| 5 | **Muscle group / brain region** | Not in the paper. Same as #1. | Any SKMGN or brain cross-check |

---

## 8. Provenance

| what | path |
|---|---|
| Paper (41 pp, searched in full) | `readings/PIIS0092867420301525.pdf` |
| GEO deposit + SOFT | `/depot/reese18/data/geo/geo_datasets/GSE137869/` |
| File manifest (172 files, 56 sample tokens) | `data/raw/geo/single_cell/geo_datasets/GSE137869/manifest.json` |
| Catalog record | `/depot/reese18/data/catalog/master_catalog.json` |
| Built WATSC reference | `data/deconvolution/references/white adipose tissue_GSE137869/` |
| Build manifest (line 80–84) | `deconvolution/tissue_references.yaml` |
| Sample → arm map (**authoritative** — do *not* use GEO row order) | `reports/annotations/annotation_inventory.tsv` |
| Deconvolved θ | `data/deconvolution/results/motrpac/WATSC/estimated_fractions.csv` |
| Companion audit | `deconvolution/TISSUE_REFERENCE_AND_PROPORTIONS.md` |
