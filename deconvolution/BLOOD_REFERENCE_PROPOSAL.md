# BLOOD deconvolution reference: options for review

> **2026-10-31 note.** Option B (the composite) was adopted and is the default BLOOD reference in refs4
> (`references_v4/BLOOD_composite`; `REFERENCE_RELABEL_PREREG.md` §12). On it BLOOD has 4 hotspots (classical
> monocytes, erythroid cells, megakaryocytes and platelets, neutrophils); the July v2 hotspots named below do not
> carry over. Results: `REFS4_RERUN_RESULTS.md`. The proposal is kept as written.

For Geyu. Written 2026-10-27. Nothing here has been built or run. The current BLOOD reference (GSE285476
PBMC) was Geyu's choice, so this note sets out the problem, what single-cell data exist, the options, and
what would have to be checked before any of them is used. The decision is Geyu's.

## Outcome (added 2026-10-27, after the user chose option B)

The user went ahead with option B before this note reached Geyu; it can still be reversed (the PBMC
reference is kept as `previous_reference_dir`). Registered in `REFERENCE_RELABEL_PREREG.md` §12 before
any data were downloaded; built in `references_v4/BLOOD_composite`. Nothing has been deconvolved on the
MoTrPAC blood samples yet.

- **Draw site.** MoTrPAC bled rats by cardiac puncture (Schenk et al. 2024, *Function*), so the GSE335156
  venous libraries (V1–V5) were used. That cardiac puncture samples venous blood is an inference.
- **Labels.** All three libraries were labelled with one blood panel written before download. The PBMC
  libraries held many doublets and platelet–lymphocyte aggregates (dropped, 13%), plus reticulocytes
  (1,775, mostly from one rat; not used, since erythroid cells come from whole blood by design).
- **Checks (§4 above, as registered):**
  - 1, composition: 2,168 neutrophils in Blood-CFA. Pass.
  - 2, erythroid cells are real: one cluster, 90 cells, median 358 genes; non-globin erythroid genes
    (Alas2, Car2, Slc4a1, Ahsp) far above every other cluster. Pass, but it is a small cluster.
  - 3, neutrophil activation: 53% of the CFA-control neutrophils score above the 95th percentile of the
    PBMC monocytes on interferon genes (Ifit1/3, Isg15, Rsad2), and 19% on Il1b/Cxcl2/Nfkbia. The control
    neutrophils are interferon-primed, as feared. This is a limitation; there was no threshold.
  - 4, PBMC types: all 12 present in all five rats (median cross-rat r 0.74). Pass.
  - 5, gene space: 18,260 genes shared, all in the bulk's space. Pass.
  - 6, recovery on simulated mixtures (reference from 3 rats + half the whole-blood cells; mixtures from
    the other 2 rats and half): per-type Pearson r between true and estimated fractions: erythroid 0.95,
    neutrophils 0.95, monocytes 0.96, T 0.91, B 0.95, NK 0.96. Pass.
  - **Absolute levels are biased.** Erythroid fractions are underestimated (mean −0.24, or −0.12 against
    a globin-free truth) and neutrophils overestimated (+0.10).
- **Globins.** BayesPrism's own outlier filter drops genes above 1% of reads in more than 10% of samples,
  so globins are removed whether or not the exclusion list keeps them. The "with globin" arm was
  therefore identical to the primary one, and the question in §4.6 is moot for this pipeline.
- **Not in the reference:** eosinophils and basophils. The marrow library has both, but the
  pre-registration allowed only the whole-blood library.

For Geyu: decisions 2 (strain) and 3 (adjuvant controls) in §5 were taken by going ahead. Check 3
suggests the CFA neutrophils are interferon-primed, which matters if BLOOD neutrophil Z is ever
interpreted.

## 1. The question

MoTrPAC's rat blood samples are **PAXgene whole blood**: RNA is stabilised from the whole draw, with no
red-cell lysis, density gradient or globin depletion. Every cell type in blood contributes RNA in
proportion to its number and its RNA content.

The reference, **GSE285476**, is **PBMC from one healthy rat** (the control arm of a liver-allograft study).
A PBMC preparation removes red cells, reticulocytes and granulocytes by density gradient, so the reference
has profiles for lymphocytes, monocytes, dendritic cells and platelets, but none for:

- **Reticulocytes and red cells.** Mature red cells have almost no RNA, but reticulocytes do, and in
  unlysed whole blood their transcripts (globins, plus Alas2, Slc4a1, Epb42, Car2, Slc25a37 and others) are
  a large share of the total. This is why whole-blood RNA studies report globin fractions and often deplete
  globin before sequencing.
- **Neutrophils.** In rats they are roughly 15–30% of leukocytes (lymphocytes about 65–80%). Neutrophils
  hold little RNA per cell, but their number makes them a substantial compartment.
- **Eosinophils and basophils.** Each is a few percent of leukocytes or less.

BayesPrism has no "unknown" cell type. Every compartment the bulk contains must have a reference profile,
or its RNA is assigned to whichever profiles the reference does have. A deconvolution reference for whole
blood therefore needs, at minimum, erythroid and neutrophil profiles alongside the PBMC types. The standard
human whole-blood signature LM22 (Newman et al. 2015) includes neutrophils and eosinophils for this reason.

A second, smaller issue: GSE285476 is **one animal**. Per-donor variation in the lymphocyte profiles cannot
be represented, and the reference cannot be checked for donor effects.

## 2. What single-cell data exist for rat blood

Searched on 2026-10-27 (GEO, ArrayExpress and the literature; full search report in
`BLOOD_REFERENCE_SEARCH.md`). The facts below were checked against the GEO series and sample records.

- No rat **whole-blood** single-cell dataset exists other than GSE306764.
- No rat reticulocyte or erythroid single-cell dataset exists.
- No F344 blood single-cell dataset exists. Every option below is another strain.

| Accession | What it is | What it could supply | Limits |
|---|---|---|---|
| **GSE306764** (Que et al., J Neuroinflammation 2026; PMID 41709234) | Singleron GEXSCOPE 3′ scRNA from Lewis rats, aligned to GRCr8, which is our gene-ID space. **Whole blood** (EDTA, orbital draw) after RBC lysis, plus **bone marrow** (tibia), spleen, thymus and lymph node. One library per tissue per group. Control libraries: Blood-CFA (GSM9207340) and Bone-CFA (GSM9207344). | **Neutrophils** and, if present, **eosinophils/basophils** from the blood library. A **reticulocyte/erythroid** profile: the paper's own annotation names an Hbb-high "erythrocyte" cluster in blood, and marrow holds erythroblasts. | The controls were immunised with adjuvant (CFA + PBS, no antigen), so their neutrophils may be activated. Each library pools several rats (the GEO protocol says 3 per group, the paper 5). Lewis females, not F344 males. Singleron, not 10x. Per-cell labels are not public, and cluster sizes are not reported. RBC lysis removes most red cells, so the erythroid cluster is whatever resists lysis; its size and quality are unknown. Droplet and microwell platforms capture neutrophils poorly. |
| **GSE335156** (Lai et al., FASEB J 2026; PMID 42141902) | 10x 3′ v3 **PBMC from 5 healthy rats**, arterial and venous (10 libraries, 119,481 cells). | A five-donor replacement for the lymphocyte, monocyte, dendritic-cell and platelet profiles. | PBMC, so no neutrophils or erythroid cells. Strain not stated on GEO. Per-cell labels not public. Arterial and venous draws differ (that is the study's question), so one site should be used. |
| **GSE343407** | Bone-marrow scRNA (Sprague-Dawley, SeekOne), 2 healthy baseline libraries (femur, spine). | Erythroblasts, as a fallback erythroid profile. | Erythroblasts are nucleated and cycling, unlike circulating reticulocytes. Marrow neutrophils are mostly immature. The tissue was digested with trypsin and collagenase. No paper yet. |
| **GSE285476** (current) | 10x PBMC, one healthy rat. | Kept as a sensitivity arm in any option. | One animal; PBMC only. |
| Mouse: **GSE137540** (Xie et al., Nat Immunol 2020) | Mouse neutrophils from bone marrow and peripheral blood, with labelled mature circulating subsets. | A cross-species check on a rat neutrophil profile, not a basis. | Mouse. Orthology shrinks the gene set, and rat globin paralogs map ambiguously. |

Ruled out on inspection: GSE220747 (hypothalamus microarray despite its title), GSE185988 (fracture
callus), GSE301731 (skin), E-MTAB-6754 (cultured macrophages), GSE293074 and GSE345203 (PBMC from a
hypertensive strain), GSE186158 (spleen, live-sorted, one rat).

## 3. Options

**A. Keep the PBMC reference.** No work. BLOOD results stay flagged as from a reference that cannot span
whole blood, and are reported as provisional or left out of claims. Absolute fractions are not
interpretable. Whether across-condition differences survive cannot be tested without a better reference.

**B. Composite whole-blood reference (recommended, subject to the checks in §4).**
- Lymphocytes, monocytes, dendritic cells and platelets from **GSE335156**, using one draw site,
  jointly annotated across the five rats with a marker panel written in advance (the procedure now being
  used for the other references; `REFERENCE_RELABEL_PREREG.md` §10).
- Neutrophils, and eosinophils/basophils if a cluster exists, from **GSE306764 Blood-CFA**.
- Erythroid cells from the **GSE306764 Blood-CFA** Hbb-high cluster, if it passes §4.2. Otherwise late,
  non-cycling erythroblasts from GSE306764 Bone-CFA (and GSE343407 baselines), accepting that erythroblasts
  are not reticulocytes.
- GSE285476 kept as a sensitivity arm for the PBMC compartment.

  This covers every major compartment and gives five donors for the lymphocytes. The cost is that the
  reference mixes two platforms and three strains. BayesPrism re-estimates each sample's cell-type
  profiles, which gives it some tolerance to platform shift between reference and bulk. Its tolerance to
  platform differences *between* reference cell types has to be tested (§4.4).

**C. GSE306764 Blood-CFA alone.** The only whole-blood single-cell library, so every compartment comes
from one platform and one preparation. The weaknesses are all in one place: one pooled library,
adjuvant-treated animals, a Lewis strain, and lymphocyte profiles from about three rats. Best as a
sensitivity arm to B, not on its own.

**D. Leave BLOOD out of the deconvolution-based analyses.** This is the fallback if no option passes the
checks.

## 4. What must be checked before any option is used

All checks use only single-cell data and are fixed before the reference is built.

### 4.1 Download and inspect GSE306764 (two GSMs, about 100 MB)

- Cluster Blood-CFA and Bone-CFA and annotate them from a marker panel written in advance.
- Count cells per compartment: neutrophils (S100a8, S100a9, Csf3r, Cxcr2), the Hbb-high cluster,
  eosinophils (Prg2, Epx), basophils (Cpa3, Fcer1a, Hdc), megakaryocytes and
  platelets (Itga2b, Pf4, Ppbp).
- Check doublets. Estimate the ambient-RNA level from empty droplets, or from globin reads in clearly
  non-erythroid clusters.
- Check neutrophil activation: compare interferon and acute-phase genes against the mouse resting
  circulating subsets (GSE137540). The question is whether the CFA control is usable as a resting profile.
- Ask the authors, or check SRA, how many rats each control library pools.

### 4.2 Accept the erythroid cluster only if it is real cells

- At least 50 cells.
- Erythroid genes other than globins (Alas2, Slc4a1, Epb42, Car2, Slc25a37) clearly above every other
  cluster, not only Hba/Hbb.
- Few genes per cell, as expected for reticulocytes, but not an ambient-only profile.

If it fails, use late erythroblasts (Alas2/Slc4a1 high, Mki67/Top2a low) from marrow and record the
mismatch.

### 4.3 GSE335156 set-up

- Confirm the strain and the reference genome. Choose the draw site from MoTrPAC's blood-collection
  protocol; this is still to be checked (the MoTrPAC protocol documents, not the data package, record it).
- Jointly annotate the five rats; check that each type appears in every rat.

### 4.4 Recovery on simulated mixtures of held-out cells

- Split cells by donor (GSE335156) and by half (GSE306764).
- Build the reference from one part. Sum cells from the other part at known proportions spanning the
  plausible range, with erythroid and neutrophil shares varied widely.
- Deconvolve. Require a per-type Pearson r ≥ 0.9 between true and estimated fractions for every
  compartment (the bar used for the other tissues' purity sweeps). Report the bias for each compartment.
- Repeat with mixtures made from GSE306764 cells only and the reference from B. This tests whether a
  composite reference handles a whole-blood library from one platform.

### 4.5 Cross-dataset agreement of the new profiles

- The GSE306764 blood neutrophil profile should correlate with mature (not immature) marrow neutrophils in
  Bone-CFA, and its top genes should match mouse circulating neutrophils (GSE137540).
- The GSE335156 lymphocyte and monocyte profiles should match GSE285476's (centred Pearson r, compared to
  the donor-to-donor yardstick within GSE335156).

### 4.6 Globins

Build the reference both with and without the Hba/Hbb paralogs, and report both.

## 5. Decisions for Geyu

1. **Option A, B, C or D.** Recommendation: B, with C and the current PBMC reference as sensitivity arms,
   provided §4 passes. If §4.4 fails for B and C, use D.
2. **Strain.** No F344 blood data exist, so B and C use Lewis and other-strain profiles. Is that
   acceptable for a reference? (BayesPrism uses the reference for profiles, not proportions, which limits
   the damage, but strain-specific genes will be off.)
3. **Adjuvant controls.** Is the GSE306764 CFA control acceptable as a source of resting neutrophil and
   erythroid profiles, conditional on the activation check in §4.1?
4. **Draw site** for GSE335156, once MoTrPAC's protocol is checked.

Until a decision is made and carried out, every BLOOD result, including its two v2 exercise hotspots
(naive B cells, NK cells), stays flagged.
