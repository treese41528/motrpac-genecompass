# Cell-type-resolved DE vs the MoTrPAC references — detailed comparison

_2026-06-22. Companion to `AIM2_DECONV_RESULTS.md` §4a. Compares our per-(tissue × cell-type)
pseudobulk DE on the deconvolved BayesPrism `Z` against the published MoTrPAC findings, tissue by
tissue. Sources: **MoTrPAC PASS1B** — #21 Nature 2024 (temporal multi-omic atlas), #22 Cell Genomics
2024 (tissue-specific transcriptomic/epigenomic); **Vetr 2024** (Nat Commun, same-data disease
genetics); **Yu 2023** (iScience, human PBMC scRNA, acute). Our numbers are from
`genecompass_input/pseudobulk_de/{de_summary,de_hotspots,posctrl_*}`._

Read directionally: the MoTrPAC papers report **bulk-tissue** transcriptomics; **our contribution is
cell-type resolution** — so the goal is (a) do we *confirm* the bulk tissue-level signal, and (b) what
do we *add* (which cell types carry it) that bulk could not see.

## 0. Headline concordances (cross-tissue)

| MoTrPAC bulk finding | Our cell-type DE |
|---|---|
| **Tissue responsiveness ranking** (most: blood, WAT, …; least: cortex, hypothalamus, testes) | **Confirmed:** strongest cell-type DE in **blood** (immune subsets 1600+ dose-sig genes) and **muscle/liver/lung**; **cortex & hippocampus ≈ 0** dose-sig genes. |
| **Effect sizes are modest** (56% of fold-changes within 0.67–1.5×) | **Confirmed & quantified:** the canonical bulk controls are near-flat at the transcript level; signal is *separability/consistency*, not large logFC (see §4a diagnostic). |
| **Sex divergence is pervasive** (58% of 8-wk features sex-differentiated; lung/WAT/liver opposite) | **Strongly confirmed & localized:** huge sex×dose **interaction** in lung (Pulmonary fibroblasts 251 interaction genes), WATSC (Myeloid dendritic cells 148), SKMVL endothelial; few *sex-consistent* genes — exactly the "mostly sex-differentiated" pattern. |
| **Immune infiltration is a top, sex-split axis** (UP in WAT/BAT males, DOWN in lung) | **Confirmed at cell-type resolution** (see BLOOD, LUNG, WAT below) and refined into *expression* (blood) vs *composition/recruitment* (WAT). |

## 1. Per-tissue comparison

### BLOOD — strongest agreement + the clearest cell-type extension
- **MoTrPAC:** one of the *most* transcriptomically responsive tissues; theme = haematopoietic mobilization (TFs GABPA/ETS1/KLF3). Vetr: strongest disease-heritability enrichment is in blood, "especially immune-cell-density traits." Yu: monocytes / cytotoxic-effector T / NK are the responsive PBMC populations.
- **Ours:** the top dose-responsive cell types are **ISG-expressing T cells (1,805), NK (1,696), classical (1,753) & non-classical (1,739) monocytes** — i.e. the bulk-blood signal is carried by exactly these immune subsets. **Positive controls recover:** Yu cytotoxicity program enriched in **NK (4.3×, p=0.0025), Memory-T (3.5×, p=0.017), ISG-T (3.3×, p=0.021)**; naive program in Memory-T (5.3×, p=0.01); Vetr blood genes (Fads2/Aamp/Bag6/Fam89b) recover in ISG-T. NK also shows a composition shift (`frac_week_p=0.006`) — consistent with Yu's exercise-driven NK redistribution. **Extension:** bulk blood → resolved to monocyte/NK/ISG-T programs, matching three independent references.

### HEART — confirms the sex-consistent striated-muscle program, in cardiomyocytes
- **MoTrPAC:** strongly responsive; converges with the two muscles to a **sex-consistent week-8 UP** mito/biogenesis/heat-shock program (Opa1, Mfn1, Mef2c, Sod2, HSPs).
- **Ours:** **Cardiomyocytes 364 dose-sig genes, with 13 sex-consistent-UP (`up_both`) at 8 wk** + sex interaction — the only tissue where we see a clear sex-consistent-up parenchyma signal, matching the MoTrPAC 8w_F1_M1 node. The data-anchored validation recovers **83%** of real heart bulk movers in cardiomyocytes (the best of any parenchyma). **Caveat:** the heart CM reference has holdout-only cross-validation (documented limitation).

### SKMGN & SKMVL (gastrocnemius, vastus lateralis) — confirms muscle response, adds stroma vs parenchyma
- **MoTrPAC:** core robust responders; sex-consistent week-8 mito/heat-shock UP program.
- **Ours:** muscle is responsive, but the cell-type breakdown is informative — the **parenchyma leads and the stroma responds alongside it**: SKMGN **Skeletal myocytes 308** / endothelial 168 / macrophages 121 / fibroblasts 109; SKMVL **Skeletal myocytes 355** / endothelial 291 / macrophages 165 / fibroblasts 38 (both muscles now deconvolved against the merged 5-type GSE137869 reference). Composition confounds fire on SKMVL endothelial (`frac_week_p=0.005`) — angiogenic/remodelling fraction shifts. **The canonical mito/HSP transcript controls are weak even in bulk muscle** (§4a) — MoTrPAC's muscle mito story is largely protein/PTM-level. **Reference fix adopted (production):** the over-split muscle parenchyma ("Skeletal muscle cells" + "Skeletal muscle fibers") was merged into one **"Skeletal myocytes"** label (SKMGN and SKMVL now share the merged GSE137869 5-type reference); the merge was carried through re-deconvolution + re-embedding + re-detection so the whole stack is consistent.

### LIVER — confirms the metabolic + sex-divergent responder
- **MoTrPAC:** strong, distinctive metabolic responder; the shared metabolic program rises in both sexes but **later in females** (liver acetyl-sites among sex-opposite features); Stat3/Pxn/Hnf4a/Hnf1b.
- **Ours:** **Hepatocytes 152, Kupffer cells 102, Hepatic stellate 95, Endothelial 55** dose-sig — broad hepatic response with **sex interaction in hepatic stellate (4), hepatocytes (2) and Kupffer (1)**, matching the documented liver sex-divergence/delayed-female onset. **Extension:** the response is not hepatocyte-only — Kupffer (immune) and stellate (stroma) carry comparable signal. (LIVER had 0 strong 8-wk bulk *movers* at the data-anchored threshold — liver's transcript dose signal is genuinely weak/late, consistent with MoTrPAC's metabolite/proteome-led liver story.)

### LUNG — striking confirmation of the sex-divergence headline
- **MoTrPAC:** moderately-strong responder with a **distinctive immune-DOWN + strong sex-OPPOSITE** signature (lung phospho/chromatin opposite M/F; immune transcripts down by wk8; lung the lone female-biased-ERG tissue in #22).
- **Ours:** **huge sex×dose interaction** — Pulmonary fibroblasts **251** interaction genes, endothelial 228, alveolar type II 190, ciliated 66 — by far the largest interaction signal of any tissue, directly mirroring MoTrPAC's "lung = top sex-opposite tissue." Strong dose signal in pulmonary fibroblasts (553), endothelial (504), alveolar type II (419), CD8+ T (245). **Extension:** the sex divergence localizes to lung stromal/endothelial + epithelial cells. **Caveat:** lung is our weakest cross-dataset deconvolution (~0.73) — treat lung claims cautiously.

### WAT-SC (white adipose) — an informative *divergence* that confirms the mechanism
- **MoTrPAC:** one of the *most* responsive tissues; the headline is **male-specific immune-cell RECRUITMENT at week 8** (B/T/NK-marker-correlated) + adrenergic-receptor down-regulation.
- **Ours:** the dominant luminal epithelial carries 214 dose-sig genes, but **the composition-confound check fires on it** (`frac_week_p=0.003`, `FLAG_COMPOSITION`, with sex interaction). This is the expected and *informative* divergence: MoTrPAC's WAT immune signal is **cell-fraction recruitment**, which the confound check attributes to our `θ` (fraction) channel rather than clean within-cell `Z` expression. So we *confirm* the WAT signal but correctly attribute it to composition/infiltration rather than within-cell expression — exactly the Milo/activity-confound point. (WAT also lacks an adipocyte reference label, so the adrenergic parenchyma program isn't directly testable.)

### KIDNEY — partial: moderate in bulk, quiet in our cell-type DE
- **MoTrPAC:** moderately responsive (AKT1/mTOR signalling; down-regulated disease terms; sex-specific mito changes).
- **Ours:** weak — proximal tubule 4, others ≤2 dose-sig. A mild **under-recovery**: kidney's bulk response is partly phospho/signalling-level and concentrated in the dominant tubule (a parenchyma down-weighted per our reporting policy), plus KID_cross showed tubule sink behaviour. Not a contradiction, but kidney exercise claims are weakly supported at cell-type resolution.

### CORTEX & HIPPOCAMPUS — concordant quiet
- **MoTrPAC:** among the **least** transcriptomically responsive tissues; brain immune enrichment limited; hippocampus shows strong female *methylation* divergence but few DE genes.
- **Ours:** **≈ 0 dose-sig genes** across all cortex/hippocampus cell types (cortex 0 bulk movers; repfdr "fraction of nulls = 1"). Fully concordant — the quiet tissues are quiet at cell-type resolution too. (Cortex reference was rebuilt gene-rich + brain-merged for consistency with hippocampus; no signal to recover either way.)

## 2. What the cell-type resolution adds over the MoTrPAC bulk map
1. **Localizes the blood signal** to monocyte/NK/ISG-T programs (3-reference agreement: gate/Augur + Yu + Vetr).
2. **Separates expression from recruitment** — WAT's "immune response" is fraction/recruitment (`θ`), not within-cell expression; blood's is genuine within-cell expression. Bulk conflates the two.
3. **Resolves sex divergence to cell types** — lung's top-sex-opposite signature is carried by stromal/endothelial cells (Pulmonary fibroblasts interaction 251).
4. **Shows stroma responds alongside parenchyma in muscle** — endothelial/macrophage stroma responds alongside the Skeletal myocytes (with angiogenic composition shifts on SKMVL endothelial), invisible in bulk muscle.
5. **Confirms the magnitude reconciliation** — the canonical mito/HSP genes are weak at the transcript level (protein/PTM-level in MoTrPAC); our DE faithfully reflects this rather than over-calling it.

## 3. Caveats carried from the analysis
- **Small magnitude** (MoTrPAC: 56% FC within 0.67–1.5×) → judge on direction-concordant significance, not logFC.
- **Composition/activity confound** — read `Z` (expression) against `θ` (fraction); flagged blocks (`frac_week_p<0.05`) are relative/differential (NK, SKM stroma, WAT luminal).
- **Dominant parenchyma** down-weighted for absolute DE (`dominant_celltype_flags.tsv`); the pipeline is sound (parenchyma `Z` tracks bulk r=0.68–0.81) but the story leans on immune + mid-abundance stromal/endothelial cells.
- **Lung** is the weakest cross-dataset deconvolution (~0.73); **heart CM** has holdout-only validation — both noted limitations.
- Yu (acute/human/male) validates **which** immune cells/programs respond, **not** effect direction.
