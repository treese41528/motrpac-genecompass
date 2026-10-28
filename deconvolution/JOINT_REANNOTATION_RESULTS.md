# Joint re-annotation of the deconvolution references: results

Run 2026-10-27 under `REFERENCE_RELABEL_PREREG.md` §10 (registered in 4e0a40f before any joint clustering)
and the §9 entries of the same day. Code: `deconvolution/joint_reannotate.py`; panels
`reference_marker_panels.yaml`; decisions `joint_reannotation_review.tsv` and `joint_reannotation_splits.tsv`;
outputs in `data/deconvolution/joint_labels/<REF>/` (gitignored). Jobs: 12035670 (seven references),
12035671 and 12035779 (HIPPOC, the second with its split), 12035758 (CORTEX with its split). **No reference has been rebuilt and
nothing has been deconvolved.**

## 1. Summary

- **Why the old names were wrong.** It was the annotation procedure (§10.1 of the pre-registration), not
  chance:
  - PanglaoDB was scored against every organ.
  - ScType fell back to **immune-only** marker lists for HIPPOC, MUSCLE, WATSC, SMLINT and BLOOD, because
    their tissue names matched nothing in its lookup.
  - An LLM then named each (PanglaoDB, ScType) label pair rather than each cluster, and invented names
    neither tool gave for 68% of HIPPOC's cells, 55% of MUSCLE's and 43% of WATSC's.
  - Nothing checked a name against the cluster's own markers.
- **The fix replaces that procedure for the references.** All samples of a reference are clustered together
  (Harmony on sample, Leiden), and each joint cluster is labelled by a fixed rule from marker panels written
  in advance. Clusters the rule cannot decide are reviewed by their markers, with the genes recorded.
  Labels are now consistent across samples by construction:
  - In every multi-sample reference, every label is present in every sample (K2: no single-sample labels).
  - Every reference passes the consistency check (K1).
- **The single-sample pairs are fixed** (K3, the registered expectation held):
  - KIDNEY "Renal interstitial fibroblasts" (the v2 hotspot) and "Renal vesicle cells" land mostly in one
    joint label, **Injured proximal tubule cells** (Havcr1/Kim-1, Cdh6, Spp1, Cd44). This label is present
    in all four samples.
  - WATSC "Pre-B cells" and "Naive B cells" land in **B cells**.
- **HIPPOC: 91% of its barcodes are not usable cells.**
  - 255,303 of 279,195 barcodes are low-quality nuclei: about 250 genes each, with housekeeping,
    mitochondrial and ambient markers.
  - These carried the old labels "Neurons", "Glutamatergic neurons", "Adrenergic neurons" and the 19%
    "Microglia", which is why the old labels barely matched across samples (r_self 0.12).
  - The 23,892 real nuclei re-annotate cleanly into 10 labels, all present in all six samples, median
    r_self **0.89**. This passes the HIPPOC criterion (≥ 0.5).
- **Deviations from the registered rule, logged in §9 before the affected checks were read:**
  - Accepted labels are also checked against their markers. **18 accepted labels were overridden**: 11
    doublet, ambient or low-quality clusters dropped, and 7 relabelled where the markers contradicted the
    name.
  - Clusters accepted only at a parent label ("Immune cells") were made specific when their markers named
    them (12).
  - CD4/CD8 calls are made only where the markers are lopsided. Cd4 is missing from every reference's gene
    space, and rat NK cells carry CD8α.
  - Two mixed clusters were re-clustered on their own (CORTEX j24, HIPPOC j13).

  The rule as registered would have shipped, among others, mammary epithelium as "Mesothelial cells", B–
  macrophage doublets as "Alveolar macrophages", and lung tuft cells as "Pulmonary neuroendocrine cells".
  Read the review file, not just the rule.

## 2. Per reference

`r_self` = median correlation of a label's profile across samples (§4 metric). The old references' median
r_self, from the audit, is in brackets.

| Reference | Cells | Joint clusters | Sent to review | Overrides | Dropped | Labels | Median r_self vs different labels | K1 |
|---|---|---|---|---|---|---|---|---|
| KIDNEY | 28,626 | 24 | 8 | 1 | 4,147 (14.5%) | 15 | 0.60 vs −0.10 [0.57] | pass |
| LIVER | 27,059 | 18 | 5 | 1 | 0 | 8 | 0.57 vs −0.11 [0.57] | pass |
| CORTEX | 12,969 | 31 (j24 split in 7) | 5 | 0 | 33 (0.3%) | 12 | 0.78 vs −0.12 [0.75] | pass |
| WATSC | 12,240 | 22 | 6 | 5 | 256 (2.1%) | 13 | 0.64 vs −0.16 [0.67] | pass |
| MUSCLE | 10,780 | 18 | 8 | 1 | 40 (0.4%) | 9 | 0.44 vs −0.14 [0.51] | pass |
| SMLINT | 18,950 | 27 | 9 | 1 | 2,757 (14.6%) | 13 | 0.65 vs −0.18 [0.68] | pass |
| LUNG | 46,669 | 38 | 8 | 9 | 2,067 (4.4%) | 21 | 0.73 vs −0.15 [0.78] | pass |
| HIPPOC | 279,195 | 35 (j13 split in 10) | 24 | 0 | 255,303 (91.4%) | 10 | 0.89 vs −0.10 [0.12] | pass, and ≥ 0.5 |

Medians are not comparable one to one: the old labels split one population across several names, so each
name held only part of it. The comparison that matters is K2: no new label is confined to one sample, while
the old references had 54 single-sample labels.

**Drops** are of four kinds:
- low-quality barcodes: few genes, mitochondrial or housekeeping markers only (HIPPOC 255,303; KIDNEY 3,431; SMLINT 2,740);
- doublets: two lineages' markers and more genes than either parent (KIDNEY j3, j12; WATSC j15, j16; LUNG
  j9, j13, j26, j29, j35, j36);
- an ambient cluster (LUNG j5);
- clusters under 20 cells.

The low-quality clusters were each spread across every sample.

### New labels, with cells (r_self)

- **KIDNEY:** Proximal tubule 12,149 (0.83); Thick ascending limb 4,015 (0.62); Endothelial 1,714 (0.70);
  Principal 1,178 (0.61); Fibroblasts 1,077 (0.66); Distal convoluted tubule 883 (0.60); Type A intercalated
  801 (0.46); Immune 639 (0.54); **Injured proximal tubule 546 (0.42)**; Connecting tubule 461 (0.52); Thin
  limb 318 (0.23); Type B intercalated 301 (0.48); Podocytes 189 (0.73); Cycling 107 (0.32); Monocytes 101
  (0.64).
- **LIVER:** Hepatocytes 24,353 (0.91); LSEC 1,168 (0.78); Hepatic stellate 570 (0.63); Kupffer 547 (0.67);
  NK 181 (0.51); Vascular endothelial 100 (0.29); Cholangiocytes 82 (0.27); Mesothelial 58 (0.18).
  - About 1,240 cells the old labels called "Endothelial cells" are hepatocytes (periportal, pericentral
    and stress-state hepatocyte clusters).
- **CORTEX:** Excitatory neurons 5,334 (0.97); Oligodendrocytes 2,267 (0.94); Inhibitory neurons 1,737
  (0.96); Astrocytes 921; Microglia 921; Medium spiny neurons 685 (one sample's striatal piece); OPC 679;
  Neuroblasts 250; VLMC 42; BAM 41; T cells 30; Monocytes 29.
  - Endothelial and mural cells are lost: after the split their parts had 16 and 17 cells, under the
    20-cell floor. The reference has no endothelial profile.
- **WATSC:** Adipose stromal progenitors 5,097 (0.92); Macrophages 3,741 (0.83); Dendritic cells 876; T
  cells 714; Non-classical monocytes 491; Endothelial 411; NK and cytotoxic T 250; B cells 120; Cycling 105;
  Mammary luminal epithelial 62; pDC 49; Glandular epithelial 45; Mural 23.
- **MUSCLE:** Skeletal muscle (myonuclei of all fibre types, MTJ and NMJ) 7,250 (0.75); FAPs 1,145;
  Endothelial 818; Pericytes 600; Satellite 261; Macrophages 257; Tenocytes 212; VSMC 167; Schwann 30.
- **SMLINT:** Enterocytes 7,361 (0.79); NK 2,136; CD8+ T 2,110; Transit-amplifying 886 (0.07); Epithelial
  (unresolved secretory/absorptive) 868; Stem 726 (0.19); Goblet 662; Erythroid 442; Tuft 402; B cells 261
  (98% one sample); BEST4+ enterocytes 142; Macrophages 128; Enteroendocrine 69.
- **LUNG:** Alveolar macrophages 15,784; Non-classical monocytes 4,267; B 3,364; Mesothelial 3,038
  (Msln/Upk3b/Lrrn4; the old label was "Airway epithelial cells"); T 2,576; Club 2,549; Ciliated 2,413; NK
  2,134; AT2 1,405; Neutrophils 1,149; CD4+ T 1,060; Goblet 1,038; Endothelial 783; Monocytes 687;
  Interstitial macrophages 501; Fibroblasts 412; DC 405; Tuft 389; VSMC 327; pDC 246; Basal 75.
  - Still no AT1 cells.

Labels with r_self under 0.5 are mostly small types (under ~600 cells) or cycling/stem states whose profile
moves with the sample's proliferative state (SMLINT TA 0.07, stem 0.19). They are listed in each
`checks.json` and are the first candidates for §6's analysable-block rule to remove downstream.

### K2: clusters dominated by one sample

Every one is real biology or dissection, not batch:
- **CORTEX:** medium spiny neurons (striatum in sample 18's dissection); j20 inhibitory neurons.
- **WATSC:** a sample-35 macrophage state (Cxcl10/Gbp5); sample-10 non-classical monocytes; sample-35
  stromal j0; sample-35 mammary epithelium.
- **SMLINT:** B cells (lymphoid tissue in sample 0); BEST4+ cells; epithelial j23.
- **LUNG:** club j21 and goblet j34, airway epithelium more abundant in one study's dissection.

## 3. Old names against new (K3 and the audit's misnamed labels)

| Old label (audit finding) | Where its cells went |
|---|---|
| KIDNEY "Renal interstitial fibroblasts" (v2 hotspot) | Injured proximal tubule 105 of 150, Thin limb 33 |
| KIDNEY "Renal vesicle cells" | Injured proximal tubule 179 of 269, Proximal tubule 53 |
| WATSC "Pre-B cells" / "Naive B cells" | B cells 68 of 68 / 41 of 58 (17 to pDC) |
| LUNG "Ionocytes" | Endothelial cells (j4) |
| LUNG "Intermediate monocytes" | Neutrophils (j15) |
| SMLINT "Merkel cells" | Enteroendocrine cells (j24) |
| SMLINT "Enteroendocrine cells" | BEST4+ enterocytes (j17) |
| SMLINT "Cholangiocytes" | Stem cells (j20), transit-amplifying (j3) |
| CORTEX "Radial glial cells" | vascular cells (split j24: VLMC, BAM; endothelial/mural lost to the size floor) |
| LIVER "Endothelial cells" (part) | Hepatocytes (j6, j7, j9) |
| HIPPOC "Microglia" (old 19% of HIPPOC) | 68,004 of 72,724 dropped as low quality; Microglia 1,555, Oligodendrocytes 1,338 |

Full cross-tabulations: `old_vs_new.tsv` per reference.

## 4. HIPPOC

- **Cells.** 279,195 barcodes over 6 samples. The low-quality share per sample is 74–96% (sample 2: 74%;
  samples 1, 7, 10: 95–96%).
- **Low-quality clusters (j0–j7, j9, j14; 255,069 barcodes).** Median 243–446 genes per nucleus (real
  HIPPOC nuclei here: 1,156–3,202). Markers are oxidative-phosphorylation, housekeeping and ambient genes:
  Ptgds and Plp1 from oligodendrocytes, Nrgn/Cck/Sst from neurons, Atp1a2/S100b from astrocytes. No panel
  scored above 0.27. A 200-gene QC floor lets these through.
  - The old labels of these barcodes: "Neurons" 111,763, "Microglia" 68,004, "Glutamatergic neurons"
    36,205, "Adrenergic neurons" 15,484.
  - The old reference's 19% microglia and its "Adrenergic neurons" were labels on ambient RNA. The
    per-sample LLM step had turned a "Glutaminergic neurons" + ScType "Non-classical monocytes" (immune-only
    list) cluster into "Microglial cells".
- **Real nuclei (23,892).**
  - Oligodendrocytes 4,815 (r_self 0.90); Dentate gyrus granule cells 4,803 (0.96); Inhibitory neurons
    4,743 (0.95: CGE Adarb2/Reln, MGE Lhx6, Sst); Excitatory neurons 3,049 (0.91); OPC 2,146 (0.89);
    Microglia 2,139 (0.91); Astrocytes 1,631 (0.85); Endothelial 341 (0.54); VLMC 185 (0.39); Mural 40
    (0.24).
  - Every label is in all six samples, except Mural (5).
  - Excitatory neurons are uneven: sample 1 has 10, sample 2 has 2,043. CA pyramidal nuclei are scarce
    overall, probably among the low-quality barcodes.
- **K1:** median r_self 0.89 against −0.10 for different labels. HIPPOC passes and its labels can be used.
- **What it means for deconvolution.** Profiles come from 23,892 good nuclei instead of 278,549 mostly
  empty barcodes. The memory problem (~500 GB to deconvolve) goes away. The July A/B result (73.9%
  excitatory neurons on the HIPPOC bulk with the old reference) has to be redone with the new labels.

## 5. What this does not fix

- **Panel genes missing from the gene space.** Cd4, Il7r, Pecam1, Rgs5, Hbb, Ccr2, Cxcr2 and (CORTEX,
  HIPPOC) Cldn5 are absent under every ID we know in the corpus matrices. Panels were scored on the genes
  present (all had ≥ 2), but the CD4/CD8 split is unreliable for this reason.
- **The review is judgement.** 100 review decisions (70 reviews, 18 overrides, 12 under-calls) were made by
  Claude from markers, each with its genes in the review file. They should be read by someone who knows the
  tissue before the references are rebuilt. KIDNEY's "Injured proximal tubule cells", HIPPOC's low-quality
  drop and the LUNG overrides matter most.
- **Missing compartments stay missing:** WATSC adipocytes, LUNG AT1, CORTEX endothelium, SMLINT muscularis.
- **HYPOTH and BLOOD** carry the same old procedure and are not re-annotated (§10.2). BLOOD's reference is
  under review separately (`BLOOD_REFERENCE_PROPOSAL.md`).
- The **training corpus** keeps its consensus labels. Only the deconvolution references, reports and
  `make_pseudobulk.py` read them; GeneCompass training and tokenization do not.

## 6. Next steps (each needs the user's go-ahead)

1. Someone with tissue knowledge reads `joint_reannotation_review.tsv`.
2. `build_reference.py` takes `cells_final.tsv.gz` (cell type plus the joint cluster as cell state) in
   place of the consensus join. The eight references are rebuilt into new directories and must pass
   `reference_qc.py`.
3. Then §6 (analysable-block rule) and §8 (rerun list) of the pre-registration, as registered.
