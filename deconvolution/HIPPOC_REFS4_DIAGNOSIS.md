# HIPPOC deconvolution: why refs4 gives 50% endothelial and 0.3% excitatory neurons (2026-10-29)

**Question.** With the jointly re-annotated reference (`references_v4/HIPPOC_GSE295314`, 23,892 nuclei),
BayesPrism assigns the MoTrPAC hippocampus bulk the following mean fractions: endothelial 0.498, inhibitory
neurons 0.358, astrocytes 0.085, DG granule cells 0.028, excitatory neurons 0.003. July (microglia 0.19,
excitatory 0.74) was not plausible either. What drives this, and can it be fixed?

**Answer.**
- **The cause is ambient RNA in the nuclei reference.** The ambient RNA ("soup") in this snRNA library is
  mostly cytoplasmic, so it resembles whole-tissue bulk more than any nucleus profile does. Whichever
  reference type carries the most soup absorbs the bulk.
  - In refs4 that is the endothelial cluster, which is about 55% ambient.
  - In July it was "Excitatory neurons" and "Microglia", 91% and 94% of whose barcodes were empty droplets
    or ambient nuclei. So the July figure of 74% excitatory was the same artefact, not a good estimate.
- **Correcting the reference for ambient RNA helps.** With ambient correction and the tiny vascular
  clusters merged, neurons reach about two thirds of the mass. The split between excitatory and inhibitory
  neurons is still not identifiable from this reference.
- **Nothing in the reported results depends on HIPPOC composition.** HIPPOC has no hotspots and no
  dose-significant genes under either reference.

Scripts and full outputs: `diagnostics/hippoc_refs4/` (1–5). An NNLS fit (square-root scale, the
BayesPrism gene set) stands in for BayesPrism in the what-if tests. On CORTEX, a nuclei reference that
works, NNLS reproduces BayesPrism (excitatory 0.608 vs 0.592; script 4), so it is a fair proxy.

## Evidence

1. **The endothelial profile is a soup** (script 1).
   - It is the lowest-quality cluster: 341 nuclei, median 967 UMI and 709 genes, against 4,000–7,000 UMI
     for the neurons.
   - Its profile carries every other type's markers: Apoe 7,828 CPM, Mbp 2,467, Snap25 664, Gfap 402,
     Mt-co1 9,601. Only 30% of its mass is on genes where it is the top type.
   - Its endothelial markers are real (Flt1 2,957, Slco1a4 958). These are endothelial nuclei swamped by
     ambient RNA.
2. **BayesPrism's endothelial Z is cytoplasmic and dendritic RNA from other cells** (script 1).
   - Top genes by endothelial Z: Mbp, Scd2, Apoe, Map1a, Map2, Ddn, Agap2, Atp1a2, Shank1, Camk2a, Nrgn,
     Hpca, Glul, S100b. These are oligodendrocyte, astrocyte and neuronal-dendrite transcripts.
   - Nuclei under-represent such transcripts. The bulk is rich in them: Slc17a7 1,975 CPM in bulk against
     at most 480 in any reference type; Camk2a 1,144 vs ≤ 804; Mt-co1 35,441 vs ≤ 9,601.
3. **An independent method fails the same way** (scripts 1–2).
   - NNLS on the same genes gives endothelial 0.59–0.69 and neurons about 0.09. On 550 marker genes,
     endothelial falls to 0.13.
   - Removing the endothelial type does not restore neurons (0.16). The mass moves to astrocytes (0.40)
     and to the 40-nucleus mural cluster (0.21).
   - Dropping the 20–40% of genes most depleted in nuclei relative to the bulk helps only a little
     (endothelial 0.43–0.36).
4. **The soup matches the bulk better than any type** (script 5).
   - The soup is the summed profile of the 254,663 barcodes the July reference kept and refs4 dropped.
   - Correlation with the bulk mean (log CPM): soup 0.87, against 0.66–0.82 for every refs4 type.
   - Given the soup as an extra component, NNLS assigns it 76% of the bulk.
   - Estimated ambient fraction per type (SoupX-style upper bound): endothelial 0.55, astrocytes 0.39,
     oligodendrocytes 0.27, OPC 0.23, microglia 0.22, … excitatory neurons 0.07. The ranking of types
     by ambient content is the ranking of types by how much bulk they absorb.
5. **Ambient correction changes the answer** (script 5). Each type's profile is corrected with the soup
   (SoupX-style subtraction) and the fit is rerun:

   | NNLS variant | Neurons (Exc / Inh / Thal / DG) | Vascular | Astro | Microglia | Oligo |
   |---|---|---|---|---|---|
   | refs4 as built | 0.09 | 0.60 (endo) | 0.21 | 0.00 | 0.11 |
   | ambient-corrected | 0.54 (0.24 / 0.14 / 0.16 / 0.01) | 0.42 (**mural 0.41**) | 0.00 | 0.03 | 0.01 |
   | corrected, vascular types merged | **0.66** (0.24 / 0.42 / – / –) | 0.13 | 0.07 | 0.11 | 0.04 |
   | corrected, mural dropped | 0.68 (0.24 / 0.19 / 0.25 / 0.00) | 0.19 (endo) | 0.03 | 0.08 | 0.02 |

   - After correction the neurons carry most of the mass. Hippocampal bulk RNA is plausibly neuron-dominated.
   - The tiny vascular clusters (mural 40 nuclei, VLMC 185) still act as sinks unless they are merged.
   - The split among excitatory, inhibitory, thalamic and DG neurons moves between variants (inhibitory
     0.14–0.42, thalamic 0–0.25). Their profiles are near-collinear (log-CPM r: Exc–DG 0.99, Exc–Inh 0.96,
     Exc–Thal 0.96). With inhibitory neurons above excitatory in the best variant, the subtype split is
     still wrong for hippocampus.
6. **Smaller contributors** (script 3).
   - **Choroid plexus is in the MoTrPAC dissection but not in the reference:** Ttr 1,799 CPM, Kl 105,
     Clic6 103, F5 62, Sostdc1 51, Aqp1 17, all ≤ 1 CPM in the reference. There are also traces of
     hypothalamic neuropeptides (Avp 16, Oxt 19, Pmch 18). This is a missing component of a few percent.
   - **19 well-expressed bulk genes (0.4% of bulk mass) are near zero in the reference.** Eno2 (1,066 vs
     0.2) is the striking one. None is the same symbol under another ID, so they are too small to matter
     here, but they are worth a look in the reference build.

## Why CORTEX works and HIPPOC does not

Both are snRNA references, so nuclei alone do not explain it.
- **Depth:** CORTEX neurons have a median of 15,290 UMI, against 5,910 for HIPPOC excitatory neurons.
- **Ambient content:** in HIPPOC, 91% of barcodes were ambient or empty.
- **Sink clusters:** CORTEX has no endothelial cluster, and its smallest clusters (VLMC 42, BAM 41) take
  only 1–2%.

A shallow, ambient-heavy library gives every HIPPOC profile a share of soup. The lowest-quality cluster
then carries the most soup and wins.

## What this means

- **For the reported results: no change.** HIPPOC has no hotspots and 0 dose-significant genes, in both
  arms. HIPPOC composition, and any per-cell-type HIPPOC result, should be marked unreliable. That holds
  for the July run as well as refs4.
- **Possible fix (a reference change, so it needs its own pre-registration entry before a rerun):**
  1. Ambient-correct the HIPPOC reference. The soup can be estimated from the dropped barcodes (SoupX) or
     from the raw matrices (CellBender).
  2. Merge Endothelial + Mural + VLMC into one vascular type.
  3. Rebuild, re-deconvolve HIPPOC, and rerun its downstream blocks.

  The NNLS proxy predicts neurons at about two thirds, with vascular cells at 0.13 and microglia at 0.11.
  It does not predict a correct excitatory/inhibitory split. Reporting merged "Neurons" for HIPPOC, or
  accepting the subtype split as unidentifiable, would be part of that decision.
- **What MoTrPAC itself found for hippocampus (checked 2026-10-29).** The MoTrPAC papers are bulk-only, so
  there are no cell-type hotspots to compare with. At tissue level, HIPPOC has 244 training-regulated genes,
  15th of 19 tissues (`MotrpacRatTraining6moData`). The Cell Genomics 2024 companion paper gives
  hippocampus the fewest exercise-responsive genes of 8 tissues (124 at week 8) but a strong female
  DNA-methylation response, which it links to neurogenesis. A correct HIPPOC composition is therefore not
  expected to produce hotspots. The fix was attempted anyway (user, 2026-10-29); see "The fix: refs4h" below.
- **Wider lesson:** the ambient fraction of a nuclei reference should be part of reference QC. A
  low-quality cluster with a high ambient share is a deconvolution sink. LIVER, HEART and KIDNEY are also
  snRNA references and have not been checked this way. KIDNEY's "Monocytes" at 0.16 (101 nuclei) fits the
  same pattern and is the first one to check.

## The fix: refs4h (REFERENCE_RELABEL_PREREG.md §13)

Pre-registered (6fae0bf) before the build. `deconvolution/R/soupx_correct_reference.R` runs SoupX 1.6.2 per
sample, with the soup from that sample's dropped barcodes, `autoEstCont` on the refs4 cell types, and
`adjustCounts`. It merges Endothelial + Mural + VLMC into Vascular cells and the four neuron types into
Neurons, keeping the joint clusters as cell states. The output is
`references_v4/HIPPOC_GSE295314_ambcorr`. The whole downstream chain reruns in `_refs4h` folders,
with the other 13 tissues linked from refs4. Jobs 12058463 (build), 12058464 (Stage 8/9 and
`6_refs4h_check.py`), 12058465 (gate), 12058466 (Stage 10), 12058467 (Stage 12) and 12058468 (follow-up).

Results: pending.
