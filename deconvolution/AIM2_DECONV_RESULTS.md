# Deconvolution → GeneCompass (Aim 2 bridge) — Results & Next Stages

_Status 2026-06-11. Branch `stage8-omnideconv-setup`. Companion to `MOTRPAC_BULK_LIFTOVER.md` (bulk gene-ID prep) and `README.md`._

---

## 1. The pipeline (grant Aim 2)

MoTrPAC bulk RNA-seq → **BayesPrism** deconvolution (per tissue, against rat single-cell references) → per-cell-type **pseudo-cells** (one per *bulk sample × cell type*, raw deconvolved expression `Z`) → corpus **tokenizer** → fine-tuned **rat GeneCompass** → 768-d cell embeddings.

The question Aim 2 asks of those embeddings: **does within-cell-type variation track the MoTrPAC exercise design (training dose, sex)?** If yes, cell-type-resolved downstream analyses (differential expression, GRN, in-silico perturbation) can recover exercise biology that bulk RNA-seq blurs across cell types.

---

## 2. Status — what is built and validated

| Stage | Status | Key artifact / result |
|---|---|---|
| Bulk gene-ID liftover & prep | ✅ | 32,432 / 32,883 ENSRNOG mapped; primary-gene vocab coverage 95.0%. See `MOTRPAC_BULK_LIFTOVER.md` |
| Per-tissue SC references (10 tissues) | ✅ | incl. **heart fix** (GSE280111 left ventricle — has cardiomyocytes) and **cortex** gene-rich union build |
| BayesPrism validation (per tissue) | ✅ | liver holdout **0.998** / cross **0.949** (meets the Chu 2022 paper ≥0.95); heart cardiomyocyte r=**0.995**; WAT corroborated across 5 methods (DWLS 0.98/0.99); **lung is the weakest cross (~0.73)** |
| Connector (pseudo-cells → tokenize → embed) | ✅ | liver smoke test: silhouette **0.775**, kNN purity **1.000**, 83% between-/17% within-cell-type variance |
| End-to-end run (all 10 tissues) | ✅ | **9,300 pseudo-cells** embedded (768-d) on disk |
| Aim-2 validation gate | ✅ | **186** (tissue × cell-type) rows across all 10 tissues (heart added 2026-06-11) |
| Interactive viewer | ✅ | self-contained, offline 3-tab `viewer.html` (Atlas / Signal / Focus) |

Tissues covered: blood, cortex, heart, hippoc, kidney, liver, lung, skmgn (gastrocnemius), skmvl (vastus lateralis), watsc (WAT). Design per tissue: 2 sex × 5 exercise group (control / 1w / 2w / 4w / 8w) × 5 reps = ~50 samples per cell type.

---

## 2a. How bulk is fed to GeneCompass — granularity & cross-tissue caveat

The data enters at **two nested levels**:
- **Tissue by tissue.** Each tissue is deconvolved against its *own* tissue-specific rat single-cell reference, and built → tokenized → embedded as a separate run against the same frozen fine-tuned rat GeneCompass checkpoint.
- **One pseudo-cell per (bulk sample × cell type)** is the unit fed to the model — its vector is the slice of *that one sample's* bulk that BayesPrism attributes to *that* cell type (raw `get.exp`), each tokenized and CLS-embedded **independently** (GeneCompass has no cross-cell attention). This deliberately preserves per-sample variation — the exercise/sex signal the gate and DE depend on. The point counts confirm it: e.g. blood 700 = 14 cell types × 50 samples; nothing is averaged across samples.

**⚠ Cross-tissue comparability caveat.** Because each tissue uses a different reference, the *same* cell-type label across tissues (e.g. "Macrophages" in WAT vs lung) is deconvolved against different references from different mixtures and is **not strictly the same population** in the embedding. The gate and the planned per-cell-type DE operate **strictly within a single (tissue × cell type)**, so they are insulated. What is **not** licensed: comparing the same cell type across tissues, or reading the Atlas's between-tissue layout as biology. To compare a shared cell type across tissues (e.g. a common immune exercise response), use a shared pan-tissue reference for those types, or scope claims within-tissue.

**Value-channel QC (per-tissue normalization drift).** Tokenization library-normalizes each pseudo-cell to a single global `target-sum = 6500`. The token *ranking* (`input_ids`, the primary signal) is normalization-invariant, but the `values` channel's median drifts by tissue vs the corpus reference (0.869): liver +0.3%, blood +1.4%, watsc +4.4%, skmgn +5.9%, hippoc +8.0%, heart +8.4%, cortex +9.1%, skmvl +12.4%, lung +17.0%, **kidney +23.3%**. The drift is ~constant within a tissue, so **within-tissue contrasts are unaffected**; it adds a small extra term to the cross-tissue confound above. If cross-tissue value comparisons become important, calibrate `target-sum` per tissue so each value-median lands near 0.869.

**Viewer note.** The Atlas UMAP is a cosine UMAP of the **768-d GeneCompass CLS embeddings** (not the raw input). The Focus tab projects those same embeddings onto a **pooled within-cell-type PCA**: each (tissue × cell-type) centroid is removed (stripping the dominant between-type variance), the residuals are scaled per group, then pooled, standardized, and reduced by a **single** PCA — one stable, drift-free, common frame across cell types, rather than 186 independent per-group PCAs on ~50 points. This surfaces the within-type (exercise/sex) structure the global UMAP compresses.

---

## 3. The gate result (the scientific finding)

**Method.** Per (tissue, cell type), multivariate η² — the fraction of the 768-d embedding's variance explained — for three factors, with a 1000-permutation p-value:
- **GROUP** — 5-level exercise dose (control / 1w / 2w / 4w / 8w)
- **TRAINED** — binary, any training vs control
- **SEX** — binary

Sex-chromosome genes were removed upstream, so the SEX signal is **autosomal** (real sexual dimorphism, not XIST/Y artifacts). **Reading caveat:** η² is *not* degrees-of-freedom-adjusted, so GROUP (4 df) is mechanically larger than the binary TRAINED/SEX (1 df). The fair effect-size comparison is **TRAINED vs SEX**.

**Overall (n = 186 tissue × cell-type):**

| factor | significant (p<0.05) | median η² |
|---|---|---|
| SEX | 99 / 186 (53%) | 0.043 |
| GROUP (5-lvl) | 52 / 186 (27%) | 0.092 *(df-inflated)* |
| TRAINED | 35 / 186 (18%) | **0.023** |

That trained median η² ≈ **0.023** is the honest size of the exercise effect at the embedding level: **real, but small and patchy.** Sex is the single most pervasive axis.

**Per tissue** (sig = p<0.05 count / #cell types; η² shown median/max):

| tissue | #ct | GROUP | TRAINED | SEX | read |
|---|---|---|---|---|---|
| blood | 14 | 11/14 · .219/.257 | **8/14 · .056/.132** | 10/14 · .062/.136 | **exercise hotspot** |
| skmvl | 15 | 9/15 · .135/.193 | **7/15 · .039/.099** | 6/15 · .032/.064 | **exercise hotspot** (exercise ≳ sex) |
| skmgn | 17 | 8/17 · .108/.189 | 7/17 · .036/.065 | 12/17 · .044/.076 | exercise moderate |
| lung | 27 | 9/27 · .102/.212 | 2/27 · .022/.040 | 11/27 · .035/.110 | **dose-only / weak** |
| hippoc | 15 | 8/15 · .114/.144 | 2/15 · .019/.052 | 1/15 · .018/.038 | **dose-only** |
| heart | 23 | 2/23 · .078/.165 | 1/23 · .022/.043 | 16/23 · .045/.212 | exercise-quiet, sex-present |
| kidney | 17 | 0/17 · .063/.094 | 1/17 · .015/.046 | **13/17 · .239/.493** | sex-dominated |
| liver | 6 | 0/6 · .064/.093 | 0/6 · .018/.037 | **6/6 · .304/.378** | sex-dominated, no exercise |
| watsc | 17 | 2/17 · .090/.165 | 4/17 · .038/.072 | **17/17 · .277/.780** | sex-dominated |
| cortex | 35 | 3/35 · .088/.138 | 3/35 · .019/.051 | 7/35 · .025/.063 | quiet |

**Four buckets:**
1. **Exercise hotspots — blood immune + skeletal muscle.** Blood: megakaryocytes (trained η² 0.132), basophils, classical/non-classical monocytes, ISG-T, NK. Vastus lateralis (skmvl): skeletal muscle fibers (0.099) and cells (0.092), endothelial, B cells, Schwann, myofibroblasts — here **trained ≳ sex**. Gastrocnemius echoes it. Biologically the right cells (immune trafficking + the muscle parenchyma itself).
2. **Dose-only — lung, hippocampus.** Many cell types significant for GROUP but almost none for TRAINED (lung 9 vs 2; hippoc 8 vs 2). Real structure *across the five dose levels* that the trained/untrained binary collapses away → **non-monotonic or late-emerging dose response.**
3. **Sex-dominated — kidney, liver, WAT (heart leans sex).** Sex swamps everything: WAT macrophages η² 0.78, luminal epithelial 0.65; kidney endothelial 0.49; liver 6/6 cell types. Exercise essentially absent. This is the known rat sexual dimorphism in these organs — it *validates* that the embeddings capture real physiology.
4. **Quiet — cortex** (and heart for exercise): little within-type signal of any kind.

**Top exercise responders (by trained η²):** blood Megakaryocytes 0.132, skmvl Skeletal muscle fibers 0.099, blood Basophils 0.098, skmvl Skeletal muscle cells 0.092, blood Classical monocytes 0.091, skmvl Schwann 0.091, skmvl B cells 0.079, skmvl Endothelial 0.078, blood ISG-T 0.075, blood Non-classical monocytes 0.075.

**Interpretation.** The embeddings separate cell types and tissues strongly and capture sex cleanly where it is real — so the substrate is sound. Exercise is a **small within-cell-type axis concentrated in immune and muscle cells**, not a global axis visible in the raw embedding. The GROUP-but-not-TRAINED pattern is direct evidence we should **model dose**, not a binary contrast.

---

## 4. Implications for Aim 2 — the pivot

Exercise will **not** fall out of the raw embeddings globally, so do not gate downstream work on a global embedding signal. Pivot to **per-cell-type differential expression on the deconvolved `Z`**, with three design rules the gate hands us directly:

1. **Focus the hotspots** — blood immune subsets and skeletal-muscle parenchyma/stroma/endothelium. Expect little to nothing in kidney/liver/cortex/heart.
2. **Control for sex** — it is the dominant axis in 3–4 tissues and will otherwise masquerade as exercise.
3. **Model dose ordinally** (1/2/4/8 wk), not trained-vs-control — lung and hippocampus show the binary collapse discards real signal.

---

## 5. Next stages

1. **Per-cell-type DE on `Z` (immediate).** Linear model per (tissue, cell type): expression ~ dose(ordinal week) + sex (+ interaction), empirical-Bayes / permutation for significance. Start with blood + vastus lateralis. Output: ranked exercise-responsive genes per cell type, sex-adjusted, with a dose shape. This is the deliverable that turns the gate signal into biology.
2. **GRN / perturbation routes (after DE).** Caveats from the connector work: pseudo-cell GRN/perturbation is only valid for **abundant** cell types (rare ones sit at the reference prior, no exercise signal); n ≈ 50 samples/cell type is too thin for data-driven GRN (DeepSEM) → default to the **model-driven in-silico-perturbation** GRN route in GeneCompass.
3. **Dose modeling.** Evaluate **CPA** (compositional perturbation autoencoder) repurposed for exercise dose (1/2/4/8 wk) on the hotspot cell types.
4. **Cross-species (grant aim).** Needs a human side — not free; scope separately.
5. **Secondary / hardening.**
   - Multi-method θ cross-check (omnideconv: MuSiC/DWLS/SCDC/Bisque) on the production tissues, as done for WAT, to confirm the deconvolution fractions feeding the hotspots.
   - **Lung caveat:** lung is the weakest cross-dataset deconvolution (~0.73); treat any lung exercise claim cautiously.
   - Viewer v2: real per-cell-type **local PCA** projection in Focus (from the 768-d embeddings) so the within-type exercise/sex axis shows on its own canvas rather than the zoomed global coords.
6. **Risks to keep in view.**
   - **Activity confound** (the key threat): bulk fraction shifts can reflect expression drift, not true cell-fraction change → frame exercise results **relative / differential**, not as absolute fractions.
   - **Quiet tissues (cortex, heart-for-exercise):** is the absence biology, or reference/abundance limitation? Note before concluding "no effect." See `reference_dominant_parenchyma_cross_failure` (most-abundant parenchyma can collapse in cross-validation).

---

## 6. Artifacts & reproduce

**Outputs** (under `data/deconvolution/genecompass_input/`, gitignored):
- `<tissue>/embeddings/cell_embeddings.npy` + `<tissue>/dataset/` — per-tissue pseudo-cell embeddings (768-d) and tokenized datasets.
- `pheno_merge_test.tsv` — the gate (186 rows).
- `umap/umap_coords.tsv`, `umap/viewer.html`, `umap/viewer_data.json` — UMAP + interactive viewer.

**Committed code** (`deconvolution/`):
- `pheno_merge_test.py` — the Aim-2 gate.
- `build_umap_viewer.py` + `umap_viewer_template.html` — extract data → self-contained viewer.

**Regenerate:**
```bash
source /depot/reese18/apps/motrpac-env/bin/activate
python deconvolution/pheno_merge_test.py        # → pheno_merge_test.tsv  (auto-discovers all tissues with embeddings)
python deconvolution/build_umap_viewer.py       # → umap/viewer.html + viewer_data.json
```
