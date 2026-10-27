# Reference label audit and the BLOOD reference — results

2026-10-27. The audit pre-registered in `REFERENCE_RELABEL_PREREG.md` (8f7acf3), run before any reference
is rebuilt (SLURM jobs 12033639 and 12033678, the second adding the full correlation matrix; decisions
identical). Code: `deconvolution/audit_reference_labels.py`. Outputs: `data/deconvolution/label_audit/`
(`audit_all.tsv`, `metric_check.tsv`, `tau.tsv`, and per reference `groups.tsv`, `self_similarity.tsv`,
`correlations.tsv`, `audit.tsv`). Nothing has been rebuilt or deconvolved again.

## 1. Summary

1. **The rule finds the duplicate labels it was built for.** Outside HIPPOC, 12 labels MERGE and each pairs
   related cells: KIDNEY "Beta-intercalated" → "β-intercalated" and "Myeloid" → "Immune cells", LIVER
   "Hepatic immune cells" → "Kupffer cells", and T-cell subsets in SMLINT, WATSC, LUNG and HEART. The merged
   group keeps the target's name even when it is narrower than the group: SMLINT's T-cell subsets become
   "Gamma delta T cells", LUNG's CD4+ T cells "CD8+ T cells", LUNG's classical monocytes "Myeloid dendritic
   cells", WATSC's NK cells "CD8+ T cells".
2. **HIPPOC's labels do not hold across samples, so the rule can't be used there.** The same label matches
   itself across HIPPOC's 6 samples at a median r of only 0.12 (other references 0.51–0.97), so its τ is
   −0.07 and almost any pairing passes; the rule would merge astrocytes into excitatory neurons and
   oligodendrocytes into adrenergic neurons. HIPPOC passes the registered metric check (0.12 > −0.08)
   only on the letter of it.
3. **The rule cannot merge two single-sample labels with each other.** Candidates must occur in at least
   two samples. Two such pairs clearly name the same cells in different samples: KIDNEY "Renal interstitial
   fibroblasts" (sample 4) and "Renal vesicle cells" (sample 8), r = 0.61 (τ 0.45); WATSC "Pre-B" and
   "Naive B", r = 0.53 (τ 0.26).
4. **Many labels are wrong, not only inconsistent.** Of the 44 labels audited in the 6 references where the
   full rule applies and anything was audited, 8 that the rule would keep carry a name their markers
   contradict, 3 more are approximate or doubtful, and 4 of the dropped ones were misnamed too. Examples: KIDNEY's hotspot "Renal interstitial fibroblasts" expresses
   Havcr1 (Kim-1), Spp1, Cd44 and Ccl7, the signature of injured tubular epithelium, not fibroblasts; LUNG
   "Intermediate monocytes" are neutrophils (S100a8/9, Retnlg, Mmp8), "Ionocytes" are endothelial (Vwf,
   Cldn5, Ccl21) and "Secretory cells" are blood (Hba/Hbb, Alas2, platelet Tubb1/Clec1b); in SMLINT the true
   enteroendocrine cells (Sst, Gip, Pyy, Gcg) are labelled "Merkel cells" while "Enteroendocrine cells" are
   BEST4+ enterocytes. The rule never renames, and the labels that occur in most samples were not audited
   for this at all.
5. **The registered fallback fits better than the rule.** §5 provides joint re-annotation (all samples
   clustered together, one label per cluster from markers) for references the rule cannot handle. Findings
   2–4 argue for using it for every consensus-labelled reference, not only HIPPOC. That is a scope decision
   and a §9 entry; it is not made here.

## 2. Metric check and τ

Every reference passes the registered check (median same-label r across samples above the median
different-label r). HIPPOC passes with 0.120 against −0.084; every other reference has a margin of at
least 0.6.

| Reference | Samples | Groups | Same label, r | Different labels, r | τ | Audited | MERGE | REVIEW | DISTINCT |
|---|---|---|---|---|---|---|---|---|---|
| KIDNEY | 4 | 46 | 0.568 | −0.129 | 0.454 | 7 | 2 | 0 | 5 |
| LIVER | 4 | 15 | 0.570 | −0.221 | 0.231 (pooled) | 3 | 1 | 0 | 2 |
| CORTEX | 2 | 17 | 0.747 | −0.176 | 0.733 | 5 | 0 | 0 | 5 |
| WATSC | 2 | 20 | 0.667 | −0.160 | 0.255 | 6 | 2 | 1 | 3 |
| MUSCLE | 2 | 10 | 0.514 | −0.206 | 0.334 | 0 | — | — | — |
| SMLINT | 2 | 19 | 0.684 | −0.282 | 0.440 | 9 | 3 | 0 | 6 |
| LUNG | 3 | 48 | 0.779 | −0.117 | 0.668 | 14 | 3 | 0 | 11 |
| HIPPOC | 6 | 35 | 0.120 | −0.084 | −0.069 | 16 | 9 | 6 | 1 |
| BAT (author labels) | 2 | 12 | 0.651 | −0.242 | 0.490 | 0 | — | — | — |
| HEART (author labels) | 19 | 249 | 0.509 | −0.091 | 0.386 | 3 | 1 | — | 2 KEEP |
| TESTES (paper panel) | 2 | 12 | 0.965 | −0.174 | 0.889 | 0 | — | — | — |

τ varies a lot (0.23–0.89) because it is each reference's own self-consistency. A high τ (CORTEX 0.73,
LUNG 0.67) leaves some obvious matches unmerged: LUNG "Intermediate monocytes" match "Neutrophils" at 0.61.

## 3. Decisions under the rule as registered

"Rule" is the script's proposal. "Outcome" applies §4's remaining steps by hand: DISTINCT becomes KEEP with
at least 50 cells and a cell type that plausibly occurs in the tissue (judged from the markers, since the
names are often wrong), otherwise DROP; REVIEW is decided from the markers. "DE reads" is the block's median
library in the current v2 DE (★ = a v2 hotspot). §4 does not say whether plausibility means the name or the
cells; I judged the cells and flag misnamed groups. KIDNEY "Renal vesicle cells" is left open: the name is
implausible in adult kidney and the markers don't identify the cells.

#### KIDNEY — τ = 0.454 (reference, 10 labels)

| Label | Samples | Cells | Best match (r) | Next (r) | Rule | Outcome | Markers / note | DE reads |
|---|---|---|---|---|---|---|---|---|
| Beta-intercalated cells | 1/4 | 37 | β-intercalated cells (0.47) | α-intercalated cells (0.21) | MERGE | MERGE → β-intercalated cells |  | 10 |
| Effector CD8+ T cells | 1/4 | 58 | Immune cells (0.39) | Loop of Henle cells (-0.05) | DISTINCT | KEEP | T/NK (Themis, Cd247, Tbx21, Zap70) | 14 |
| Intercalated cells | 1/4 | 43 | Loop of Henle cells (0.11) | Principal cells (0.01) | DISTINCT | DROP | <50 cells; no intercalated-cell markers (Nrip3, Pcsk9, Mamdc2) | 59 |
| Mesangial cells | 1/4 | 180 | Renal fibroblasts (0.20) | Proximal tubule cells (0.17) | DISTINCT | KEEP | mural / mesangial (Myh11, Acta2, Lmod1) | 19 |
| Myeloid cells | 1/4 | 307 | Immune cells (0.53) | Endothelial cells (0.02) | MERGE | MERGE → Immune cells |  | 18 |
| Renal interstitial fibroblasts | 1/4 | 150 | Proximal tubule cells (0.15) | Podocytes (0.06) | DISTINCT | KEEP *misnamed* | injured tubular epithelium, not fibroblasts (Havcr1/Kim-1, Spp1, Cd44, Ccl7); r = 0.61 with Renal vesicle cells | 15 ★ |
| Renal vesicle cells | 1/4 | 269 | Proximal tubule cells (0.15) | Principal cells (0.05) | DISTINCT | ? (see text) | name implausible in adult kidney; cells unresolved (Corin, Htr4, Upk3bl1); r = 0.61 with Renal interstitial fibroblasts | 11 |

#### LIVER — τ = 0.231 (pooled, 4 labels)

| Label | Samples | Cells | Best match (r) | Next (r) | Rule | Outcome | Markers / note | DE reads |
|---|---|---|---|---|---|---|---|---|
| Cholangiocytes | 1/4 | 29 | Hepatocytes (-0.03) | Hepatic stellate cells (-0.06) | DISTINCT | DROP | <50 cells; genuine cholangiocytes (Spp1, Ggt1, Muc1, Itgb6) — a real type lost | 32 |
| Hepatic immune cells | 1/4 | 145 | Kupffer cells (0.41) | Hepatocytes (-0.01) | MERGE | MERGE → Kupffer cells |  | 37 |
| Kupffer cells | 2/4 | 381 | Hepatocytes (0.14) | Endothelial cells (-0.29) | DISTINCT | KEEP | Kupffer (Cd163, Slc11a1) | 411,579 |

#### CORTEX — τ = 0.733 (reference, 6 labels)

| Label | Samples | Cells | Best match (r) | Next (r) | Rule | Outcome | Markers / note | DE reads |
|---|---|---|---|---|---|---|---|---|
| CD4+ T cells | 1/2 | 46 | Microglia (0.31) | Astrocytes (-0.13) | DISTINCT | DROP | <50 cells; myeloid / MHC-II, not T cells (Cd74, RT1-Da, Mrc1, S100a9) | 23 |
| CD8+ T cells | 1/2 | 94 | Microglia (0.18) | Excitatory neurons (0.02) | DISTINCT | KEEP *doubtful* | mixed: neutrophil (S100a8/9), NK (Gzmk, Nkg7), endothelial (Flt1) — likely blood/doublets | 510,677 |
| Cholinergic neurons | 1/2 | 21 | GABAergic neurons (0.43) | Excitatory neurons (0.29) | DISTINCT | DROP | <50 cells; genuine cholinergic (Chat, Slc5a7) | 321,729 |
| Neuroblasts | 1/2 | 102 | GABAergic neurons (0.24) | Oligodendrocyte precursor cells (0.23) | DISTINCT | KEEP *approx.* | cycling cells (Top2a, Cdk1, Cenpf) | 11 |
| Radial glial cells | 1/2 | 152 | Astrocytes (0.13) | Oligodendrocyte precursor cells (-0.13) | DISTINCT | KEEP *misnamed* | vascular mural + endothelial (Tagln, Acta2, Myl9, Flt1, Slco1a5) | 573,883 |

#### WATSC — τ = 0.255 (reference, 7 labels)

| Label | Samples | Cells | Best match (r) | Next (r) | Rule | Outcome | Markers / note | DE reads |
|---|---|---|---|---|---|---|---|---|
| Monocytes | 1/2 | 31 | Myeloid dendritic cells (0.24) | Gamma delta T cells (0.07) | DISTINCT | DROP | <50 cells; plasmacytoid DC (Siglech, Spib, Irf8) | 555,183 |
| Myofibroblasts | 1/2 | 20 | Endothelial cells (0.37) | Fibroblasts (0.34) | REVIEW | MERGE → Fibroblasts | REVIEW near-tie; contractile mural/myofibroblast program (Acta2, Tagln, Des), no endothelial markers | 575,921 |
| NK cells | 1/2 | 67 | CD8+ T cells (0.37) | Myeloid dendritic cells (-0.02) | MERGE | MERGE → CD8+ T cells |  | 164,001 |
| Naive B cells | 1/2 | 41 | CD8+ T cells (0.12) | Gamma delta T cells (0.04) | DISTINCT | DROP | <50 cells; B cells | 17 |
| Naive CD8+ T cells | 1/2 | 384 | CD8+ T cells (0.66) | Myeloid dendritic cells (-0.09) | MERGE | MERGE → CD8+ T cells |  | 370,533 |
| Pre-B cells | 1/2 | 68 | Myeloid dendritic cells (0.11) | CD8+ T cells (0.09) | DISTINCT | KEEP *misnamed* | mature B (Cd19, Cd79b, Fcer2, Cr2); r = 0.53 with Naive B cells | 101,406 |

#### SMLINT — τ = 0.440 (reference, 5 labels)

| Label | Samples | Cells | Best match (r) | Next (r) | Rule | Outcome | Markers / note | DE reads |
|---|---|---|---|---|---|---|---|---|
| CD8+ NKT-like cells | 1/2 | 1,033 | Gamma delta T cells (0.75) | Erythroid cells (-0.02) | MERGE | MERGE → Gamma delta T cells |  | 21 |
| CD8+ T cells | 1/2 | 911 | Gamma delta T cells (0.72) | Erythroid cells (-0.56) | MERGE | MERGE → Gamma delta T cells |  | 44 |
| Cholangiocytes | 1/2 | 732 | Goblet cells (0.20) | Enterocytes (-0.02) | DISTINCT | KEEP *misnamed* | crypt S-phase cells (Olfm4, Mcm2, Mcm6, Cdc6) | 27 |
| Enteroendocrine cells | 1/2 | 125 | Enterocytes (0.20) | Tuft cells (0.19) | DISTINCT | KEEP *misnamed* | BEST4+ enterocytes (Best4, Otop2) | 7 |
| Memory CD8+ T cells | 1/2 | 223 | Gamma delta T cells (0.55) | Erythroid cells (-0.52) | MERGE | MERGE → Gamma delta T cells |  | 122,155 |
| Merkel cells | 1/2 | 41 | Goblet cells (0.40) | Tuft cells (0.09) | DISTINCT | DROP | <50 cells; these are the enteroendocrine cells (Sst, Gip, Pyy, Gcg, Chga) | 31,153 |
| Naive B cells | 1/2 | 262 | Gamma delta T cells (0.19) | Tuft cells (-0.36) | DISTINCT | KEEP | B cells (Cd79a/b, Cd19) | 1,037,186 |
| Non-classical monocytes | 1/2 | 134 | Gamma delta T cells (0.13) | Tuft cells (-0.24) | DISTINCT | KEEP *approx.* | macrophages (C1qa-c, Csf1r) | 801,143 |
| Proliferating cells | 1/2 | 376 | Goblet cells (0.15) | Enterocytes (-0.04) | DISTINCT | KEEP | G2/M cycling cells (Top2a, Ube2c) | 42 |

#### LUNG — τ = 0.668 (reference, 14 labels)

| Label | Samples | Cells | Best match (r) | Next (r) | Rule | Outcome | Markers / note | DE reads |
|---|---|---|---|---|---|---|---|---|
| Basal cells | 1/3 | 40 | Club cells (0.50) | Tuft cells (0.34) | DISTINCT | DROP | <50 cells; genuine basal (Krt5, Krt14, Krt17) | 64 |
| CD4+ T cells | 1/3 | 692 | CD8+ T cells (0.75) | Naive B cells (0.31) | MERGE | MERGE → CD8+ T cells |  | 17 |
| CD8+ NKT-like cells | 1/3 | 679 | CD8+ T cells (0.46) | Alveolar macrophages (0.30) | DISTINCT | KEEP | cytotoxic (Gzma, Prf1, Nkg7) | 17 |
| Classical monocytes | 1/3 | 356 | Myeloid dendritic cells (0.73) | Monocytes (0.56) | MERGE | MERGE → Myeloid dendritic cells |  | 22 |
| Gamma delta T cells | 1/3 | 23 | CD8+ T cells (0.59) | Naive B cells (0.16) | DISTINCT | DROP | <50 cells | 17 |
| Intermediate monocytes | 1/3 | 290 | Neutrophils (0.61) | Monocytes (0.20) | DISTINCT | KEEP *misnamed* | neutrophils (S100a8/9, Retnlg, Mmp8); r = 0.61 to Neutrophils, under tau 0.67 | 9 |
| Ionocytes | 1/3 | 521 | Endothelial cells (0.44) | Myeloid dendritic cells (0.15) | DISTINCT | KEEP *misnamed* | endothelial incl. lymphatic (Vwf, Cldn5, Ptprb, Ccl21) | 39 |
| Macrophages | 1/3 | 300 | Myeloid dendritic cells (0.64) | Alveolar macrophages (0.29) | DISTINCT | KEEP | interstitial macrophages (C1q, Folr2, Cd163) — a v2 hotspot | 42,424 ★ |
| Mast cells | 1/3 | 121 | CD8+ T cells (0.26) | Naive B cells (0.04) | DISTINCT | KEEP | mast (Tpsb2, Cpa3) | 32 |
| Memory CD4+ T cells | 1/3 | 437 | CD8+ T cells (0.56) | Myeloid dendritic cells (0.30) | DISTINCT | KEEP | Treg/Th17 (Foxp3, Il17a); r = 0.86 with CD4+ T cells | 15 |
| NK cells | 1/3 | 565 | CD8+ T cells (0.64) | Naive B cells (0.19) | DISTINCT | KEEP | NK (Gzma, Klri1/2) | 10 |
| Naive CD8+ T cells | 1/3 | 569 | CD8+ T cells (0.74) | Naive B cells (0.25) | MERGE | MERGE → CD8+ T cells |  | 15 |
| Plasmacytoid dendritic cells | 1/3 | 76 | Myeloid dendritic cells (0.54) | Naive B cells (0.36) | DISTINCT | KEEP | pDC (Siglech) | 21 |
| Secretory cells | 1/3 | 180 | Alveolar macrophages (0.40) | Myeloid dendritic cells (0.15) | DISTINCT | KEEP *misnamed* | blood: erythrocytes + platelets (Hba/Hbb, Alas2, Tubb1, Clec1b) | 8 |

#### HEART — τ = 0.386 (reference, 15 labels)

| Label | Samples | Cells | Best match (r) | Next (r) | Rule | Outcome | Markers / note | DE reads |
|---|---|---|---|---|---|---|---|---|
| CD8+ T cells | 1/19 | 1,624 | T cells (0.49) | NK cells (0.32) | MERGE | MERGE → T cells |  | 9 |
| Mast cells | 4/19 | 115 | Dendritic cells (0.30) | Macrophages (0.26) | KEEP | KEEP | author labels: only MERGE applies | 38 |
| Mesothelial cells | 8/19 | 292 | Cardiac fibroblasts (0.22) | Endothelial cells (0.13) | KEEP | KEEP | author labels: only MERGE applies | 56,972 |

#### HIPPOC — not decided by the rule

The rule's output is in `label_audit/HIPPOC/audit.tsv` (9 MERGE, 6 REVIEW, 1 DISTINCT). It is not used.
With τ = −0.07 the merges follow noise: "Astrocytes" (11,285 cells, one sample) → "Excitatory neurons",
"Endothelial cells" → "Oligodendrocyte precursor cells", "Oligodendrocytes" → "Adrenergic neurons". The
label set itself is implausible for hippocampus: "Adrenergic neurons" (15,681 cells in 3 samples),
"Bergmann glia" (cerebellar), "Schwann cells" (peripheral). The July reference review already listed
HIPPOC as the reference with a real labelling defect (27% microglia, wrong-region labels). HIPPOC needs
joint re-annotation whatever is decided for the others.

## 4. What the audit means for the rebuild

The registered rule would fix the duplicates (finding 1) and leave everything else: two single-sample
pairs unmerged, about a dozen kept groups under names their markers contradict (one of them a hotspot),
merged groups under narrower names than they hold, and the majority labels unexamined. Cell-type names are
not cosmetic here: the positive controls, the enrichment, the hotspot interpretation and the GRN anchors are
all read by name.

Two ways forward, both needing a §9 entry before anything is rebuilt:

| | A. Rule as registered, minimal amendments | B. Joint re-annotation (the registered fallback, extended) |
|---|---|---|
| What | apply §3; HIPPOC to joint re-annotation; amend §4 to merge mutually-matching single-sample pairs and to rename merged groups to their common type | for each consensus-labelled reference (KIDNEY, LIVER, CORTEX, WATSC, MUSCLE, SMLINT, LUNG, HIPPOC): cluster all samples together, label clusters from a marker panel fixed per tissue before clustering, merge clusters that share a label |
| Fixes | label drift | label drift and wrong names, in majority labels too |
| Leaves | wrong names (KIDNEY hotspot, LUNG neutrophils/endothelium/blood, SMLINT enteroendocrine, CORTEX vascular) | judgement in the marker panels (mitigated by fixing them first and recording every call) |
| Cost | hours | about a day per reference for the panels and review; then the same rebuild and reruns as A |

The audit outputs stay useful under B: the correlations and markers are the evidence each joint label is
checked against, and BAT, HEART and TESTES (consistent labels, no merges needed except HEART "CD8+ T cells"
→ "T cells") need no re-annotation.

**Recommendation: B**, with the marker panels pre-registered per tissue in §9 before any clustering.

## 5. BLOOD

### 5.1 What is wrong (established, with this session's numbers)

- MoTrPAC BLOOD is **PAXgene whole blood**: `TRNSCRPT_META` "PaxGene RNA", 50 samples, globin reads
  9.2–16.9% (median 12.3%; every other tissue ≤ 0.86%). `MotrpacRatTraining6moData` has **no blood counts**
  (no CBC or differential in `PHENO`; only collection fields), so there is no ground truth for composition.
- The reference, GSE285476, is **PBMC from one healthy rat** (Ficoll prep): no erythroid cells and no
  granulocytes by construction. Reference-free CDSeq on the real bulk (July; `results/_deployed_backup_20260715/BLOOD/cdseq_rr/`)
  put ~80% of the centred bulk mRNA mass in components matching nothing in the reference, named by their
  markers as erythroid (Slc25a37, Epb41, Epb42, Car2) and granulocyte (S100a9, Mcemp1, Chi3l1, Lrg1,
  Siglec8). The one component the reference has a profile for (platelet/megakaryocyte) was recovered.
- The current BayesPrism fractions show where that mass went (mean of 50 samples): "ISG-expressing T
  cells" 55.3%, megakaryocytes 19.5%, basophils 9.3%, non-classical and classical monocytes 5.5% and 5.4%;
  everything else ≤ 1.7%. Rat whole blood has well under 1% basophils, and an interferon-stimulated T-cell
  state is not half of anything. Five lymphocyte blocks and the platelet block are near-empty (median 5–15 reads per sample); the two
  BLOOD hotspots (naive B cells, NK cells) are not among them.
- This is **omitted-component misspecification**: the basis cannot span the bulk, so the fit is confidently
  wrong. A better label scheme cannot fix it; the reference needs erythroid and granulocyte profiles.

### 5.2 What exists (searched 2026-10-27; key facts re-checked against the GEO records)

Full search report: `deconvolution/BLOOD_REFERENCE_SEARCH.md`. No rat reticulocyte or erythroid RNA-seq
exists in GEO; there is one rat whole-blood single-cell dataset. Checked against the GEO series and sample
records:

| Accession | What it is (verified) | Use | Risk |
|---|---|---|---|
| **GSE239996** (PMID 37941435) | Bulk RNA-seq, Sprague-Dawley males, WT, untreated. **Neutrophils ×6 and monocytes ×6 FACS-sorted from blood** (neutrophils CD45+/CD11b+/CD3−/CD45RA−/CD161−/CD43int/SSC-high, 7-AAD−); B, CD4, CD8, NK ×6 each sorted from **spleen**; whole blood ×5, **ACK RBC-lysed** (not PAXgene). NEBNext Ultra II Directional, Salmon on Rnor_6.0 cDNA; supplementary files have gene raw counts and TPM. The authors built a rat deconvolution reference from these. | the **Neutrophil** profile | bulk, so it enters as pseudo-cells; SD not F344; Rnor_6.0 transcript IDs need mapping to Ensembl 115 genes; no eosinophils, basophils or erythroid |
| **GSE306764** (PMID 41709234) | Singleron GEXSCOPE scRNA, Lewis rats, aligned to **GRCr8**. Whole blood (EDTA, orbital) with **RBC lysis**, plus bone marrow, spleen, thymus, lymph node; one library per tissue per group. Control libraries ("Blood-CFA", GSM9207340; "Bone-CFA", GSM9207344) are from adjuvant-immunised controls, 3 rats selected per group. | the **Erythroid** profile, if its Hbb+ cluster holds real cells; a neutrophil cross-check | adjuvant-primed controls; pooled single library; Singleron, not 10x; cluster sizes unknown; labels not public |
| **GSE335156** (PMID 42141902) | 10x 3′ v3 PBMC, **5 healthy rats**, arterial and venous (10 libraries, 119,481 cells). | replace the one-donor PBMC basis for lymphocytes, monocytes, platelets | no granulocytes or erythroid by design; strain not stated; labels not public |
| GSE343407 | Bone marrow scRNA (SD, SeekOne), 2 healthy baseline libraries | erythroid fallback (late erythroblasts) | erythroblasts are nucleated and cycling, not reticulocytes; marrow neutrophils immature |

Ruled out on inspection: GSE220747 (hypothalamus microarray despite its title), GSE185988 (fracture
callus), GSE301731 (skin), E-MTAB-6754 (cultured macrophages), GSE293074/GSE345203 (PBMC, hypertensive
strain). No F344 blood reference of any kind exists.

### 5.3 Proposed BLOOD reference

1. **Lymphocytes, monocytes, platelets:** GSE335156, the libraries of one site (the study's own question is
   the arterial–venous difference, so mixing sites adds variance; the site is chosen to match MoTrPAC's draw,
   to be checked in its protocol), jointly annotated from a marker panel as in §4 B. GSE285476 is kept as a sensitivity arm.
2. **Neutrophils:** GSE239996's six sorted blood neutrophil samples, each entered as one pseudo-cell of type
   "Neutrophils" (one cell state). Their top genes (S100a9, S100a8, Lrg1, Prok2, Il1r2) match the CDSeq
   granulocyte component.
3. **Erythroid:** the Hbb-high cluster of GSE306764 Blood-CFA, accepted only if it holds real cells rather
   than ambient RNA: at least 50 cells that carry erythroid genes other than globins (Alas2, Slc4a1, Car2,
   Epb42) at a level the other clusters don't. Otherwise late, non-cycling erythroblasts (Alas2/Slc4a1 high,
   Mki67/Top2a low) from GSE306764 Bone-CFA and GSE343407 baselines, accepting the erythroblast-versus-
   reticulocyte mismatch.
4. **Eosinophils and basophils:** no rat source; a Siglec8/Prg2 cluster in GSE306764 Blood-CFA is used if
   one exists, otherwise they stay inside "Neutrophils".
5. **Gene space:** GRCr8 (GSE306764) is the bulk's Ensembl 115 space; GSE335156's reference must be checked;
   GSE239996's Rnor_6.0 transcripts go to genes through the Ensembl 104 transcript table, then the bulk's own
   liftover map (`deconvolution/reference/motrpac_bulk_liftover.tsv`). The share of rows that survive is
   reported.
6. **Globins:** run with and without the Hba/Hbb paralogs; report both.

Not used as basis: CDSeq's own components (fitted to the same 50 samples, so exercise-related drift would
enter the basis and undercut the trained-versus-sedentary contrast). They serve as a check instead (5.4).

### 5.4 How it is validated before any BLOOD result is used (to pre-register in §9)

There is no blood-count ground truth, so validation rests on four checks, all fixed before the rerun:

1. **Absorption.** The new Neutrophil and Erythroid profiles must each match a CDSeq component (centred
   Pearson r above BLOOD's own self-similarity yardstick), and after the rerun the "ISG-expressing T cells",
   megakaryocyte and basophil fractions must fall out of the sink range (each below 10%).
2. **Plausible composition.** Among leukocyte types, lymphocytes > neutrophils > monocytes, with basophils
   below 2%. (Rats are lymphocyte-dominant: roughly 65–80% lymphocytes, 15–30% neutrophils. BayesPrism's
   fractions track mRNA share more than cell number, so only the order and broad ranges are checked.)
3. **Recovery on mixtures with known answers.** Pseudo-bulk mixtures built from held-out GSE306764 Blood-CFA
   cells and held-out sorted samples at known proportions; per-type Pearson r ≥ 0.9 between true and
   estimated fractions, the bar used for the other tissues' purity sweeps.
4. **An external real mixture.** GSE239996's five lysed whole-blood samples, deconvolved with the new
   reference: neutrophils and lymphocytes must both be substantial and erythroid low (they were RBC-lysed).

### 5.5 Decisions needed

- The PBMC reference was Geyu's choice ("field-standard whole-blood basis"). Replacing and augmenting it is
  a change to put to Geyu.
- Whether the strain mismatch (SD and Lewis profiles for F344 bulk) is acceptable; no F344 alternative
  exists.
- Until BLOOD is rebuilt, its two v2 hotspots (naive B cells, NK cells) and every BLOOD result stay flagged.
