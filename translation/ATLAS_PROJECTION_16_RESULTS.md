# E.3 on the 16-tissue default: result (2026-11-04)

Pre-registration: `ATLAS_PROJECTION_16_PREREG.md` (780b149). The SPLEEN/OVARY map rows (0595e80) were committed
before any Spleen or Ovary atlas cell was embedded. Job 12192423. Record: `atlas_record_16/` (`summary.txt`,
`atlas_projection_per_type.tsv`, and the Tabula Sapiens name lists the new rows were written from). Scores:
`data/deconvolution/atlas_tabula_sapiens/score_refs4all_rebuild/`.

**Verdict: PASS.**
- Macro top-1 accuracy is **0.430** over 120 rat cell types in 13 tissues.
- The label-shuffle null is 0.086 (95th percentile 0.125); p = 0.0001 over 10,000 permutations.
- That is 5.0 times the null mean.

**Reproduction.**
- All 103 per-type accuracies of the 11 first-run tissues equal the first run's (max difference 0.0); their
  transferred cells are byte-identical.
- Over those 11 tissues alone the accuracy is 0.404, as before, so that verdict is the first run's PASS.

**New tissues.**
- **Tabula Sapiens pseudobulks:** Spleen gives 97 (21 types, 6 donors) and Ovary gives 20 (8 types, 3 donors).
  None fell below 200 genes.

| Tissue | Accuracy (chance) | Scored rat types |
|---|---|---|
| SPLEEN | 0.553 (0.091) | 11 |
| OVARY | 0.653 (0.125) | 6 |

- **SPLEEN types that land:** B cells 1.00, red-pulp macrophages 1.00, plasma cells 0.94, erythroid progenitors
  0.92 (on hematopoietic precursor cells), non-classical monocytes 0.88, neutrophils 0.78.
- **SPLEEN types that fail:** CD4+ and CD8+ T cells 0.04 and NK cells 0.00, all three landing on regulatory T
  cells. Lymphocyte subtypes failed in the first run too. Endothelial cells score 0.20 and monocytes 0.28, both
  landing mostly on macrophages.
- **OVARY:** endothelial, lymphatic endothelial and macrophages 1.00; stromal cells 0.88. Theca cells (0.04) and
  smooth muscle cells (0.00) land on the ovarian stromal centroid. Twelve rat ovary types have no Tabula
  Sapiens counterpart: granulosa, luteal, pericytes and most immune types.

**Hotspots.** The default's nine; values are unchanged from the first run.

| Hotspot | Accuracy | Lands on |
|---|---|---|
| SKMVL Vascular smooth muscle cells | 1.00 | smooth muscle cell |
| BLOOD Megakaryocytes and platelets | 0.74 | platelet |
| BLOOD Neutrophils | 0.66 | neutrophil |
| BLOOD Classical monocytes | 0.48 | monocyte |
| SKMVL Satellite cells | 0.46 | slow muscle cell |
| SKMGN Tenocytes | 0.02 | erythrocyte |
| LUNG Vascular smooth muscle cells | 0.00 | alveolar adventitial fibroblast |
| BLOOD Erythroid cells | 0.00 | classical monocyte |
| HYPOTH OPC | not scored | no brain in Tabula Sapiens |

**Overall.** 49 of 120 types reach accuracy ≥ 0.5 and 31 score 0. Uncentred (secondary, no verdict): 0.140
against 0.087, p = 0.059.

**Reading.** The anchoring holds on the default and extends to SPLEEN and OVARY. The three hotspots that did not
land in the first run (SKMGN tenocytes, LUNG vascular smooth muscle, BLOOD erythroid) still do not, so human-space
claims about them remain unsupported. No other centring, distance, pseudobulk size or map revision was tried.
