> Search run by a delegated agent on 2026-10-27. The facts used in `REFERENCE_LABEL_AUDIT.md` §5 for GSE239996, GSE306764 and GSE335156 were re-checked against the GEO series and sample records the same day; other entries are as the agent reported, with its own [V]/[I]/[U] marks.

# Rat whole-blood deconvolution reference: search for erythroid and granulocyte profiles

Date: 2026-10-27. Research only. Nothing was downloaded into the repo, built or rerun.

**Problem:** MoTrPAC rat BLOOD bulk is PAXgene whole blood (F344). Our reference, GSE285476, is PBMC from one rat and has no reticulocytes and no granulocytes. CDSeq puts about 80% of the bulk mRNA mass into erythroid and granulocyte components.

**Goal:** find rat (or, failing that, mouse) profiles for these two missing compartments.

**Evidence labels used below:**
- **[V]** verified from the source in this session: a GEO record, the paper text, or the data file itself.
- **[I]** inferred.
- **[U]** could not be verified.

---

## 1. Ranked candidates (best first)

| # | Accession | Species / strain | Tissue and prep | Platform | Healthy or control arm (n) | Neutrophils / erythroid present? | Per-cell labels public? | Raw counts public? | Main risk |
|---|---|---|---|---|---|---|---|---|---|
| **1** | **GSE239996** (Mizuno lab; PMID 37941435, "Rat deconvolution as knowledge miner…") | Rat, Sprague-Dawley, male, WT, untreated **[V]** | **FACS-sorted circulating blood neutrophils**, gated CD45+/CD11b+/CD3−/CD45RA−/CD161−/CD43int/SSC-high, 7-AAD−. Same series also has sorted blood monocytes and spleen B, CD4 T, CD8 T and NK cells, ×6 each. Plus 5 **whole-blood (cardiac puncture, ACK RBC-lysed) bulk** samples and 5 spleen bulk samples **[V]** | Bulk RNA-seq (NEBNext Ultra II Directional, NovaSeq). Salmon against Rnor_6.0 cDNA **[V]** | All untreated. **Neu ×6, Mon ×6, B/CD4/CD8/NK ×6, wbc ×5** **[V]** | **Neutrophils: yes, sorted and pure-looking.** I checked the processed file: the transcripts most enriched in Neu over the lymphoid and monocyte samples are **S100a9 (~33k CPM), S100a8 (~42k CPM), Lrg1, Prok2, Il1r2**. These are mature circulating-neutrophil markers and overlap the CDSeq granulocyte signature (S100a9, Lrg1) **[V]**. **Erythroid: no** sorted population. The ACK-lysed wbc bulk is still dominated by Hba-a1/Hba-a3, so globin survives lysis **[V]** | Not applicable (bulk) | **Yes.** `GSE239996_processed_data.txt.gz` (9.3 MB) has 61 columns, 31,196 rows, roughly 10M reads per column, so it looks like **raw counts**. Rows are **ENSRNOT transcript IDs (Rnor_6.0)**, one per row **[V]** | Bulk, not single cells: must enter BayesPrism as pseudo-cells (section 2a). Strain is SD, not F344. Sorting stress could induce ex vivo activation genes **[I]**. IDs are Rnor_6.0 transcripts, so they need ENSRNOT→ENSRNOG (Ensembl ≤104) then a symbol bridge to Ensembl 115. Our biomart folder has no transcript map **[V]**. Neu_5 has only 3.4M reads **[V]**. No eosinophil or basophil profile (CDSeq flagged Siglec8). |
| **2** | **GSE306764** (Que et al., J Neuroinflammation 2026, PMID 41709234, PMC13020373) | Rat, **Lewis, female**, about 8 weeks at immunisation **[V, paper]** | **Whole blood (orbital, EDTA) after RBC lysis** (Singleron GEXSCOPE buffer). Same rats also give **bone marrow (tibia)**, spleen, thymus and lymph node, all RBC-lysed **[V]** | Singleron GEXSCOPE microwell 3′ scRNA. **Aligned to GRCr8** (CeleScope/STAR) **[V]** | Control = **CFA + PBS immunised, no AChR** (an adjuvant-inflamed control, not naive), 28 days after the 2nd immunisation **[V, paper]**. GEO has **one library per tissue per group** (Blood-Ctrl GSM9207340, Bone-Ctrl GSM9207344). Animal count conflicts: GEO protocol says 3 rats, paper says n = 5 per group, so the libraries are probably pooled **[U]** | Paper's own annotation of the rat blood names **neutrophils (S100a8/S100a9)** and **"erythrocytes" (Hbb)**, plus megakaryocytes (Itga2b/Pf4) in the marker list **[V]**. **Cell numbers and proportions are not reported** in the text I could read **[U]**. Bone marrow should contain the immature neutrophil lineage and erythroblasts **[I]** | **No.** Only barcodes/genes/matrix per sample. Labels must be re-derived from the paper's marker list **[V]** | **Yes.** Per-GSM tar.gz with mtx: Blood-Ctrl 47.9 MB, Bone-Ctrl 53.6 MB **[V]** | This is the **only rat whole-blood single-cell dataset found**. Risks: the control arm is CFA-primed (possible neutrophilia or activation), there is one (pooled?) library, the strain is Lewis, and the platform is Singleron rather than 10x. RBC lysis removes most mature RBCs, so the "erythrocyte" cluster is the lysis-resistant reticulocyte/RBC residue: its **size and quality are unknown** **[U]**. Neutrophil capture on microwell platforms is usually poor **[I]**. The upside: GRCr8 IDs match our Ensembl 115 bulk space **[I, per our rat-ID notes]**. |
| 3 | **GSE343407** (osteoporosis BM; no PMID yet) | Rat, SD; sex not stated, but OVX model so female **[I]** | **Bone marrow** (proximal femur, lumbar vertebrae), **enzymatic digestion** (trypsin then collagenase II), RBC lysis **[V]** | SeekOne droplet 3′ (Seekgene), NovaSeq X; mRatBN7.2 **[V]** | **Pre-OVX baseline H0 (femur) and S0 (spine): 2 healthy libraries**. H1/H4/S1/S4 are OVX **[V]** | Marrow should hold the neutrophil lineage and nucleated erythroblasts **[I]**. **Counts not verified, no paper** **[U]** | **No** **[V]** | **Yes**, per-GSM mtx (H0 matrix 52 MB) **[V]** | Marrow neutrophils are mostly **immature** (pro/myelocytes, band cells). Erythroblasts are **nucleated, cycling, not reticulocytes**. Trypsin/collagenase digestion is harsh on granulocytes **[I]**. Best used as a supplementary erythroid source only. |
| 4 | **GSE335156** (Lai et al., FASEB J 2026, PMID 42141902) | Rat, healthy; strain not given on GEO **[U]** | **PBMC** from paired arterial and venous blood **[V]** | 10x-type scRNA (GPL31008), 119,481 cells **[V]** | **5 healthy rats × 2 sites = 10 libraries** **[V]** | PBMC prep: **no granulocytes or reticulocytes by design** **[I]** | Not on GEO (RAW.tar only) **[V]** | Yes (RAW.tar 1.19 GB) **[V]** | Does **not** fill the gap. It is the obvious **replacement for the one-donor GSE285476 PBMC basis** for the lymphocyte, monocyte and platelet compartment (5 animals, healthy). |
| 5 | GSE186158 (multi-species spleen atlas) | Rat; strain not given **[V]** | **Spleen**, ACK RBC-lysed, then **FACS-sorted for live (7-AAD−) cells** **[V]** | 10x (Cell Ranger 3.0.1, Rnor_6.0) **[V]** | Healthy, **1 rat library** **[V]**. Local file `qc_h5ad/GSE186158_sample5.h5ad` has **14,432 cells × 17,338 genes; obs has only `_index`, so no labels** **[V]** | Rat red pulp has some erythropoiesis and few neutrophils **[I]**. Not checked: I did not load the matrix **[U]** | No **[V]** | Yes **[V]** | Spleen, not blood. ACK lysis plus live sorting depletes erythroid cells. Low value. |
| 6 | GSE293074 / GSE345203 (Dahl SS, salt hypertension) | Rat, Dahl SS **[V]** | **PBMC** (+ intestinal LP or kidney) **[V]** | 10x 3′ v3.1 (GSE293074, mRatBN7.2); 10x 5′ v2 (GSE345203) **[V]** | GSE293074: **PBMC NS ×3** (normal salt) **[V]**. GSE345203: low-salt ×2 **[V]** | PBMC: no granulocytes or erythroid by design **[I]**. The Ficoll/density prep is **not stated** in the GEO protocol **[U]** | No | Yes (local raw for GSE293074) **[V]** | Hypertensive strain and PBMC. At most an extra lymphoid/monocyte source. |
| 7 | GSE122792 (c-kit+ BM under phenylhydrazine stress) | Rat, ACI **[V]** | BM **c-kit+ sorted**; PHZ vs PBS, 3 vs 3 **[V]** | scRNA (details not checked) | PBS ×3 **[V]** | Erythroid **progenitors** only (c-kit+), no reticulocytes or mature neutrophils **[I]** | Not checked | Not checked | Wrong maturation stage. Not useful for a whole-blood basis. |
| ✗ | **GSE220747** | Rat, Wistar, male, 9 wk **[V]** | **Hypothalamus**, TRIzol **[V]** | **Agilent microarray** (GPL22145) **[V]** | 6 ctrl / 6 stress / 6 stress+exercise **[V]** | **No bone marrow or blood data in this series** **[V]** | — | — | **Irrelevant.** The "brain–bone marrow" title is not reflected in the deposited data. Not in local qc_h5ad **[V]**. |
| ✗ | GSE185988 | Rat, SD **[V]** | **Fracture callus** cells, collagenase 4 h **[V]** | scRNA, HiSeq 3000, Rnor_6.0 **[V]** | wk4 / wk12 controls ×2 **[V]** | Callus marrow may contain some myeloid/erythroid cells **[I]** | No | Yes (local raw) | Injury tissue, not marrow. **Not useful.** |
| ✗ | GSE301731 | Rat, SD / SRG **[V]** | **Skin** (calciphylaxis). Mostly bulk; 2 skin scRNA h5 files **[V]** | — | vehicle + H2O skin **[V]** | Tissue-infiltrating neutrophils only | — | — | **Irrelevant** (skin). |
| ✗ | E-MTAB-6754 | Multi-species incl. rat **[V, catalog]** | **Cultured BM-derived mononuclear phagocytes** + LPS/poly I:C **[V, catalog]** | scRNA | Time-0 arm | No neutrophils or erythroid (cultured macrophages) **[I]** | — | — | **Irrelevant.** |

None of GSE220747, GSE185988, GSE293074, GSE301731 or E-MTAB-6754 has a file in `data/training/qc_h5ad/`. Of the local candidates, only `GSE186158_sample5.h5ad` (the rat spleen) is present **[V]**. Raw GEO downloads for GSE220747, GSE185988, GSE293074, GSE301731 and GSE186158 are in `data/raw/geo/geo_datasets/` **[V]**.

**Search coverage:**
- GEO, via NCBI E-utilities: rat × {single cell} × {whole blood / peripheral blood / bone marrow / neutrophil / leukocyte / spleen / reticulocyte / erythroid}.
- ArrayExpress/BioStudies.
- Web and literature search.

No rat whole-blood scRNA-seq other than GSE306764 was found. **No rat reticulocyte or erythroid RNA-seq (bulk or single-cell) exists in GEO**; rat erythroid hits were all miRNA arrays or unrelated **[V]**. The Ma 2020 rat atlas (GSE137869) has no blood or marrow **[V]**. CELLxGENE and Single Cell Portal are human/mouse-centric and were not searched exhaustively **[U]**. China's GSA/NGDC was not searched systematically **[U]**. The Li 2026 rat MRSA bone-infection marrow scRNA (PMID 41355542, 7 neutrophil subsets) has no accession in its abstract, and the arm is infected **[U]**.

### Mouse cross-species fallbacks (best 2–3; via rat–mouse orthologs)
1. **Xie et al. 2020 Nat Immunol, GSE137540.** Mouse neutrophils from BM and peripheral blood across homeostasis and infection, more than 25k cells, 8 populations, including the **3 mature PB subsets (G5a–c)** **[V, abstract]**. Excellent mature-circulating-neutrophil labels. **No erythroid.** Sorting details per sample not checked **[U]**.
2. **Mouse Cell Atlas (Han et al. 2018 Cell), GSE108097.** Has **PeripheralBlood1–6** (female, 6–10 wk) and BoneMarrow1–3 **[V]**, on Microwell-seq. Whether the PB annotation contains neutrophil and erythroblast/reticulocyte clusters was **not confirmed** **[U]**. Microwell-seq depth is low.
3. **An et al. 2014 Blood, GSE53983 (mouse arm).** Sorted bulk of mouse terminal erythroid stages **[V, accession]**. Whether a reticulocyte stage is included was **not confirmed** **[U]**. Bulk, so it would enter as pseudo-cells.

Cross-species risk: neutrophil and erythroid core genes (S100a8/9, Hba/Hbb, Alas2, Slc4a1) are well conserved, but the rat-specific globin paralogs (Hba-a1/a3, Hbb-b1/bs) and mouse-only genes map ambiguously. The ortholog-restricted gene set shrinks the basis.

---

## 2. Non-reference options

**(a) Sorted bulk as pseudo-cells in BayesPrism.** BayesPrism sums the reference counts per cell type (or cell state) and normalises each to a profile. It then re-estimates a sample-specific profile per type in its Gibbs step, which is what gives it some robustness to platform shift (Chu et al. 2022 Nat Cancer). Entering each sorted replicate as one "cell" with label `Neutrophil` is therefore mechanically valid: the type-level profile is what matters, and its absolute count scale drops out after normalisation **[I, from the model structure; I found no BayesPrism-documented example in this session]**.

Caveats:
- **3′-tag single-cell vs full-length directional bulk** differ in gene-length and coverage bias, so neutrophil marker weights will be on a different footing than the other types **[I]**. Mitigations:
  - Restrict to the marker / protein-coding gene set already used.
  - Optionally re-derive the other types from GSE239996's own sorted bulk, so that every type is bulk-based. This makes the reference internally consistent and all-rat. GSE239996 has B, CD4, CD8, NK, Mon and Neu; it lacks platelets and erythroid.
- With 6 pseudo-cells, `cell.state` variance is meaningless. Use a single state.
- **Precedent:** the GSE239996 authors built a rat deconvolution reference from exactly these sorted profiles and validated it on whole-blood and spleen bulk (PMID 37941435) **[V, abstract]**. Sorted-bulk signature matrices are also the classic CIBERSORT/LM22 setting **[I, well known]**.

**(b) CDSeq-estimated erythroid and granulocyte profiles as extra basis vectors.** This is technically easy: add two "types" whose profiles are the CDSeq component vectors.

**Circularity:** the profiles are fitted to the same 50 samples. BayesPrism would then partly reproduce CDSeq's own fractions. Worse, any exercise-associated expression drift inside those compartments gets baked into the basis and can reappear as spurious or suppressed composition change. That is double dipping, and it directly undermines the differential (trained vs sedentary) claim.

If used at all:
1. Learn the profiles on a split (e.g. one sex, or sedentary-only) and apply them to the rest.
2. Treat the two components as **nuisance sinks** and report only the lymphoid/monocyte fractions and Z.
3. Validate each CDSeq component against an external profile (GSE239996 Neu; the GSE306764 erythrocyte cluster) by correlation before trusting it.

CDSeq reference: Kang et al. 2019 PLoS Comput Biol **[I, citation not re-checked this session]**. My recommendation is to use (b) only as a **diagnostic**, i.e. to check that the new external basis absorbs the CDSeq components, not as the basis itself.

**(c) Explicit "unknown/missing component".**
- **BayesPrism has no free "unknown" type.** Every type needs reference cells. Its tumour handling (per-sample re-estimated malignant profile and the embedding-learning module) still starts from reference malignant cells, so it cannot recover a compartment that is entirely absent. This omitted-component misspecification is exactly what we observed.
- Methods that do estimate an uncharacterised fraction:
  - **EPIC** (Racle et al. 2017 eLife): "other cells" fraction, via markers assumed not expressed by the unknown cells plus mRNA-content scaling.
  - **quanTIseq** (Finotello et al. 2019 Genome Med): "other" fraction.
  - **ISOpure** (Quon et al. 2013) and **DeMixT** (Wang et al. 2018): learn one unknown (tumour) profile per sample.
  - Fully unsupervised: **CDSeq**, **debCAM**.

  **[I, citations from knowledge, not re-verified this session.]**
- All of these give at best a lumped "other" fraction. With about 80% of the mass unknown, the marker assumptions (lymphocyte markers absent in the unknown mass) are plausible for erythroid but **not** for granulocyte vs monocyte (S100a8/9, Lyz2 are shared). So an unknown-component method would at best report "~X% other" and still bias the monocyte estimate.

---

## 3. Recommendation

1. **Granulocytes: add GSE239996 sorted blood neutrophils (rat SD, ×6) as a `Neutrophil` type** (pseudo-cells, single state).
   - This is the only rat mature circulating-neutrophil profile found, and its top genes match the CDSeq granulocyte component.
   - Consider rebuilding the lymphoid and monocyte types from the same series (all-bulk, all-rat reference) as a sensitivity arm against the scRNA-based basis.
   - Cross-check the profile against the GSE306764 Blood-Ctrl neutrophil cluster.
2. **Erythroid: derive a reticulocyte-like `Erythroid` type from GSE306764 Blood-Ctrl** (the Hbb-high "erythrocyte" cluster of lysed whole blood).
   - If that cluster is too small or too ambient-like, fall back to **late, non-cycling erythroblasts** (Hbb/Alas2/Slc4a1-high, Mki67/Top2a-low) from GSE306764 Bone-Ctrl and GSE343407 H0/S0, accepting the erythroblast-vs-reticulocyte mismatch.
   - Mouse GSE53983 is the last resort.
   - With 9–17% globin reads, consider whether to exclude Hba/Hbb paralogs from the gene set entirely. This stops one erythroid type from soaking up globin noise, at the cost of erythroid identifiability; test both.
3. **Lymphocyte, monocyte and platelet compartment:** replace or augment the one-donor GSE285476 PBMC with **GSE335156** (5 healthy rats).
4. **Eosinophils and basophils:** no rat source was found. Accept them as a residual inside Neutrophil, or derive them from GSE306764 if a Siglec8/Prg2 cluster exists **[U]**.
5. Use CDSeq only as a check: after adding the new types, the unmatched-mass fraction should collapse and lymphocyte blocks should stop being near-empty.

### What needs verifying before use
- **GSE306764 (download ~100 MB, two GSMs):**
  - Cell counts for neutrophil, "erythrocyte", megakaryocyte and any eosinophil clusters in Blood-Ctrl and Bone-Ctrl.
  - Whether the erythrocyte cluster is real cells or ambient Hbb.
  - Doublet rate.
  - Confirm the GRCr8 gene-ID version matches our Ensembl 115 space.
  - Ask the authors, or look in SRA, whether the Blood-Ctrl library pools 3 or 5 rats.
  - Assess CFA-induced activation: compare neutrophil IFN/acute-phase genes to GSE239996.
- **GSE239996:**
  - Map ENSRNOT (Rnor_6.0) to gene (Ensembl 104 archive), then symbol-bridge to Ensembl 115. Check how many of the 31,196 rows survive.
  - Confirm sorting purity (low Cd3e/Cd79a/Cd68 in Neu).
  - Decide whether to drop Neu_5 (3.4M reads).
  - Confirm the column values are counts, not salmon effective-length-scaled values (they look like counts).
- **Platform mixing (bulk pseudo-cells + scRNA):** test on simulated mixtures before the real run, e.g. pseudo-bulk from GSE306764 Blood-Ctrl plus GSE239996 wbc bulk as a semi-real check. The wbc bulk is lysed whole blood with Hba-dominated reads, so it is the closest available external mixture.
- **Strain mismatch** (SD / Lewis vs F344): no F344 blood reference exists. Note it as a limitation.
- **Pre-register** the expected rat leukocyte split (lymphocytes ~65–80%, neutrophils ~15–30% of leukocytes) and the erythroid/granulocyte absorption criterion before rerunning, per project practice.
- For the record: GSE220747, GSE185988, GSE301731 and E-MTAB-6754 are **dropped** (verified: wrong tissue or no marrow data).

---

## Sources
- [GSE239996](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE239996), PMID [37941435](https://pubmed.ncbi.nlm.nih.gov/37941435/)
- [GSE306764](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE306764), [PMC13020373](https://pmc.ncbi.nlm.nih.gov/articles/PMC13020373/)
- [GSE343407](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE343407)
- [GSE335156](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE335156), PMID [42141902](https://pubmed.ncbi.nlm.nih.gov/42141902/)
- [GSE345203](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE345203), [GSE293074](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE293074), [GSE186158](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE186158), [GSE122792](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE122792), [GSE220747](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE220747)
- Li 2026 rat MRSA marrow scRNA: [PMID 41355542](https://pubmed.ncbi.nlm.nih.gov/41355542/)
- Mouse: [Xie 2020 (GSE137540)](https://pubmed.ncbi.nlm.nih.gov/32719519/), [MCA GSE108097](https://pubmed.ncbi.nlm.nih.gov/29474909/), [An 2014 GSE53983](https://pmc.ncbi.nlm.nih.gov/articles/PMC4041167)
- Ma 2020 rat atlas (no blood/marrow): [bis.zju.edu.cn](https://bis.zju.edu.cn/cellatlas/rattus-norvegicus-cell-atlas.html)
