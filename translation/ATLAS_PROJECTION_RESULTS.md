# E.3 result: transferred rat cells land nearest their Tabula Sapiens counterparts (2026-10-31)

Pre-registration: `ATLAS_PROJECTION_PREREG.md` (4895834) and the correspondence map
`atlas_correspondence.tsv` (de65969), both committed before any atlas cell was embedded. Job 12103831.
Record tables: `atlas_record/` (per-type table, summary).

> **2026-11-01 note.** E.3 was run on the refs4 (14-tissue) cells and its 11 hotspots. The current all-tissue (16)
> default has 9 hotspots, all among those 11 (LUNG endothelial and mesothelial cells drop out;
> `deconvolution/ALL_TISSUE_RERUN_RESULTS.md`); SPLEEN and OVARY were not projected. Not rerun.

**Verdict: PASS.** Macro top-1 accuracy **0.404** over 103 rat cell types in 11 tissues, against a label-shuffle
null of 0.084 (95th percentile 0.124); p = 0.0001 (10,000 permutations); 4.8 times the null mean.

**Human side.** Tabula Sapiens 10x cells, one pseudobulk per (donor, cell type) with ≥ 20 cells: Blood 90
pseudobulks (18 types), Lung 102 (28), Muscle 68 (22), Heart 42 (15), Liver 57 (20), Kidney 10 (10, one donor),
Fat 72 (20), Small intestine 89 (31), Testis 8 (8, one donor). No pseudobulk fell below 200 genes.

| Tissue | Accuracy (chance) | Scored rat types |
|---|---|---|
| SKMVL | 0.550 (0.062) | 8 |
| LIVER | 0.529 (0.050) | 7 |
| HEART | 0.512 (0.078) | 12 |
| LUNG | 0.432 (0.050) | 17 |
| SKMGN | 0.415 (0.062) | 8 |
| TESTES | 0.400 (0.125) | 5 |
| BAT | 0.395 (0.200) | 4 |
| SMLINT | 0.369 (0.044) | 11 |
| BLOOD | 0.320 (0.086) | 11 |
| WATSC | 0.290 (0.087) | 8 |
| KIDNEY | 0.267 (0.167) | 12 |

**Hotspots** (accuracy, most common human type):

| Hotspot | Accuracy | Lands on |
|---|---|---|
| LUNG Endothelial cells | 1.00 | endothelial cell of artery |
| SKMVL Vascular smooth muscle cells | 1.00 | smooth muscle cell |
| BLOOD Megakaryocytes and platelets | 0.74 | platelet |
| BLOOD Neutrophils | 0.66 | neutrophil |
| BLOOD Classical monocytes | 0.48 | monocyte |
| SKMVL Satellite cells | 0.46 | slow muscle cell |
| SKMGN Tenocytes | 0.02 | erythrocyte |
| LUNG Vascular smooth muscle cells | 0.00 | alveolar adventitial fibroblast |
| BLOOD Erythroid cells | 0.00 | classical monocyte |
| LUNG Mesothelial cells, HYPOTH OPC | not scored | no Tabula Sapiens counterpart (no mesothelium in TS lung; no brain) |

**Reading.**
- The transfer is anchored. Transferred rat cells sit nearest the right real-human cell type four to five times
  as often as chance, so the human-space coordinates are biological, not only a re-encoding of rat structure.
  The manuscript's limitation (v) can be replaced by this result.
- It holds per cell type for 39 of 103 types (accuracy ≥ 0.5) and fails outright for 29 (accuracy 0). Failures
  concentrate where the atlas is thin or coarse: KIDNEY (one donor; every epithelial segment is one TS type,
  and its segments land on immune centroids), TESTES (one donor), and lymphocyte subtypes (CD8 vs CD4, Treg,
  B vs plasma).
- Three hotspots do not land on their counterpart: SKMGN tenocytes, LUNG vascular smooth muscle and BLOOD
  erythroid cells. Human-space claims about those three should not be made from this transfer. BLOOD erythroid
  is the one with a likely cause: TS erythrocytes are enucleated cells with almost no RNA beyond globins, while
  the rat erythroid profile is deconvolved from whole-blood bulk and its reference cells include erythroid
  precursors (§12 check 2).
- Without centring (the secondary test, no verdict), accuracy is 0.144 against 0.085 (p = 0.066). The two sources
  differ by a global offset (pseudobulk construction, technology; the TS pseudobulks' nonzero token values have
  median 0.45–0.61 against 0.71 for the transferred BLOOD pseudo-cells), which the registered centring removes.
