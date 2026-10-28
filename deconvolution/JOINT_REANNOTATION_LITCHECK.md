# Joint re-annotation: literature check of the review decisions (2026-10-28)

Nobody with tissue knowledge had read the decisions in `joint_reannotation_review.tsv`, so each one was
checked against the published marker literature. Rows covered: the 100 decisions for the eight tissue
references and the 35 for the three blood libraries. The auto-accepted clusters were also scanned for
top markers that contradict their label. This check is not a substitute for an expert read. It found the
decisions that are wrong by the literature's own markers, and those were then tested in the data.

Per-row verdicts, rationale and citations: `joint_reannotation_litcheck.tsv` (156 rows; `kind` = review row
or flagged auto-accepted cluster; `action` = what was done).

## How

Four agents, one per tissue group (kidney/liver/muscle; cortex/hippocampus; lung/intestine;
adipose/blood). Each got the review rows and `clusters_final.tsv` and searched the literature (rodent
atlases and canonical marker papers). Most also read pooled per-cluster CPM from `group_sums.npz` to test
markers the review reasons did not name. Verdicts:
- SUPPORTED: the markers are established for the label.
- PLAUSIBLE: consistent, but weak or broader or narrower than ideal.
- QUESTIONABLE: the markers point elsewhere.
- SOUND: a DROP whose doublet or low-quality reasoning holds.

Every QUESTIONABLE call and every flagged LUNG cluster was then re-checked here against the pooled CPM
before anything changed.

**Citation caveat.** The citations are papers the agents saw in search results or on fetched pages. Most
were read at abstract level, and none has been checked against a local PDF; this project has had
fabricated citations before. Rows where no source was found say so. Use the citations as pointers, not
as verified references.

## Result

| | Review rows | SUPPORTED | PLAUSIBLE | QUESTIONABLE | SOUND (DROP) |
|---|---|---|---|---|---|
| Eight tissue references | 100 | 53 | 19 | 3 | 25 |
| Blood libraries | 35 | 19 | 5 | 0 | 11 |

Some DROP rows were scored SUPPORTED or PLAUSIBLE instead of SOUND (e.g. KIDNEY j3, PLAUSIBLE: the
doublet call holds, but its "1.5x genes" argument is weak because the S3 PT clusters are gene-rich too).
No DROP was judged unsound.

Flagged auto-accepted clusters: 21, of which 1 is QUESTIONABLE (BLOOD_BM j19) and 20 PLAUSIBLE.

### Changed (three clusters; logged in REFERENCE_RELABEL_PREREG.md §11)

| Cluster | Was | Now | Evidence (pooled CPM) |
|---|---|---|---|
| LUNG j37 (444 cells) | Mesothelial cells | DROP | Mesothelial genes at a third of the real mesothelial cluster j31 (Msln 801 vs 2,237; Upk3b 525 vs 1,555). Macrophage and AT2 genes present: Lyz2 32,240 (macrophage clusters about 47,000; j31 1,959), Cd68 510, Ptprc 755, Sftpc 11,309. Median 3,463 genes vs 2,210. Old labels "Alveolar macrophages". This is the doublet pattern already used to drop j36. |
| CORTEX j24.5 (29 cells) | Monocytes | Neutrophils | S100a9 10,667, S100a8 3,076, Ngp 2,341, Camp 602, Retnlg 234. No classical-monocyte signal (Cd14 0, Cx3cr1 0). Intravascular blood leukocytes. |
| HIPPOC j24 (598 cells) | Inhibitory neurons | Thalamic neurons | Lef1 521 and Tcf7l2 183, against 35 or less in every hippocampal neuron cluster; Zic1 and Shox2 present. Lef1 and Tcf7l2 mark thalamus and are absent from the hippocampal formation (Nagalski 2012; Lipiec 2020). Gad1 281, a third to a sixth of the interneuron clusters. A dissection carry-over, kept as its own type so it stays out of the interneuron profile. The MoTrPAC dissection may carry the same tissue. |

### Questionable but not changed

- **BLOOD_BM j19 "B cells"** is pro-B/pre-B cells (Dntt, Vpreb1a, Rag1). The marrow library is not in the BLOOD
  composite, so nothing downstream reads this label.

### Flagged LUNG clusters checked in the data and kept

- **j18 "Alveolar macrophages":** cell-cycle markers on a macrophage background (Lyz2 29,560, Cd68 668,
  Mki67 607). T-cell genes are low (Cd3e 48). These are proliferating alveolar macrophages.
- **j12 "Alveolar macrophages":** Mrc1 691 like the other alveolar macrophage clusters, plus Fn1 1,089 and
  Mafb 128. A monocyte-derived or transitional state; the label holds.
- **j24 "AT2":** Sftpc 171,794; AT1 genes low (Ager 337, Rtkn2 43, Pdpn 55). The reference has no AT1
  cluster, a known gap.
- **j28 "Vascular smooth muscle":** Myh11 1,769 and Acta2 952 dominate. Some pericyte genes (Abcc9 479).
- **j32 "Mesothelial":** Msln 2,565 and Upk3b 1,247 confirm it. Dcn 1,116 shows some fibroblast admixture.

### Label refinements suggested, not applied

These sharpen a label without moving cells between types BayesPrism would separate. They are left for
an expert:
- **Lung and intestine:**
  - LUNG j34 Goblet → airway mucous/secretory (Muc5b+).
  - LUNG j8 is Treg-rich (Foxp3, Il2ra, Ikzf2).
  - SMLINT j25 "Epithelial cells" → secretory, goblet/Paneth-like.
  - SMLINT j6 and j13 "NK cells": check for γδ T cells (Cd3e, Trdc).
- **Kidney:**
  - KIDNEY j2 → "PT, injured/dedifferentiated". Vcam1 and Krt20 are absent; Havcr1 is only 31 CPM.
  - KIDNEY j13 → "Stromal (fibroblast-dominant)": some pericyte/mesangial genes.
- **Adipose:**
  - WATSC j18 → mammary luminal hormone-sensing epithelium (Prlr, Gata3).
  - WATSC j14 is the CD142+ (F3) preadipocyte subtype.
  - WATSC j13 includes mast cells (Cpa3).
- **Blood:**
  - BLOOD_PBMC j19 and BLOOD_WB j18 "T cells" are NKT-like (Klra receptors, Cd8a ≫ Cd8b).
  - BLOOD_WB j6 is Treg-enriched.
- **Brain:**
  - HIPPOC j21 Microglia hides a border-associated macrophage subset (Mrc1 377, Cd163 174).
  - HIPPOC j13.1 and j13.7 endothelium are low-quality and could merge with j13.0/j13.8.
  - CORTEX j10 oligodendrocytes are newly formed (Gpr17, Bcas1).

### Rules confirmed

- **CD4/CD8.** Rat NK cells carry CD8α only (Torres-Nagel 1992), so Cd8b is the T-cell discriminator.
  Cd40lg as a CD4 surrogate is supported, but it is activation-induced and will miss some resting CD4 cells.
- **HIPPOC mass drop.** The 91% dropped were low-quality barcodes, and the call is sound. Medians are
  243–330 genes against 1,200–3,200 kept. Mitochondrial reads are high (Mt-co1 about 10,000–13,000 vs
  800–1,400). Markers of several cell types co-occur, spread evenly over samples. That is the profile of
  ambient RNA or empty droplets (Lun 2019 EmptyDrops; Fleming 2023 CellBender; Caglayan 2022).
