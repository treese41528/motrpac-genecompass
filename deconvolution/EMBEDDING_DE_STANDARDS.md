# Embedding-based condition analysis & differential expression — methods in the literature

_2026-06-15. Companion to `AIM2_DECONV_RESULTS.md` (§3b). A web-verified review of how the field
(a) quantifies how strongly a covariate is encoded in single-cell / embedding data, and (b) does the
downstream differential expression — and where our Aim-2 pipeline sits relative to that precedent._

> Citation provenance: items marked **[✓web]** were verified against primary sources in this session;
> **[mem]** are from prior knowledge (well-established methods, not re-verified here) and should be
> confirmed before formal citation.

---

## TL;DR — where our methods sit (and that they are standard)

| Our step | Established equivalent | Verdict |
|---|---|---|
| Gate η² = trace(B)/trace(T) on the 768-d embedding | **PERMANOVA / `adonis`** R² (distance-based variance partition) | **STANDARD** |
| Supervised CV PLS-1 / classifier-AUC probe, per cell type | **Augur** (CV classifier AUC for perturbation-responsiveness) | **STANDARD** — we ran canonical Augur; it reproduces our probe at r=0.83 |
| Per-cell-type DE on the deconvolved `Z` | **pseudobulk + DESeq2/edgeR/limma**, per cell type | **STANDARD** |
| (not done) DE *on* embedding dimensions | — | not accepted practice; decode latent→genes (scVI) or test in gene space |
| Embedding as the representation | foundation-model embeddings ≈ **PCA** for perturbation | **caution** — confirmed: embedding > PCA-50 only modestly, ties full genes |

---

## 1. Differential expression "with" embeddings — and the real standard

**The accepted norm for per-cell-type DE is pseudobulk, in gene space — not "DE on the embedding."**
Squair et al. 2021, *Nat Commun* 12:5692, "Confronting false discoveries in single-cell differential
expression" **[✓web]** showed that the top-performing DE methods all **aggregate cells to sample-level
pseudobulk** before testing (DESeq2 / edgeR / limma); cell-level tests inflate false discoveries and
over-credit highly expressed genes, and pseudobulk also better predicts matched proteomics. Our
pseudo-cells are already one-per-sample, so a per-(tissue × cell-type) linear model on `Z` *is* the
pseudobulk approach.

Where embeddings *do* enter DE, it is via **generative models that decode the latent back to gene
space**, not by testing embedding coordinates:
- **scVI / scANVI** model-based DE — Lopez et al. 2018, *Nat Methods* **[mem]**; the empirical-Bayes
  log-fold-change test, Boyeau et al. 2023, *PNAS* **[✓web]** — Bayesian decision theory on the latent
  generative model, with effect-size (LFC) control.
- **CPA** (compositional perturbation autoencoder) — Lotfollahi et al. 2023 **[mem]** — models
  perturbation/dose in latent space; relevant to our dose-modeling next stage.

**Implication for us:** do the DE in gene space, per cell type, pseudobulk-style; the embedding's job
was detection/prioritization (done, §3b), and CPA is the latent-space option for dose later.

---

## 2. Quantifying how strongly a covariate structures an embedding — the standards

Ranked by how established:
- **PERMANOVA / `adonis`** — Anderson 2001, *Austral Ecology* 26:32; McArdle & Anderson 2001;
  `vegan::adonis2` **[✓web]**. The permutational, distance-based partition of multivariate variance by a
  factor; the reported R² is exactly our trace-η². The de-facto standard for "how much does factor X
  explain this (embedding) distance structure." **[STANDARD]**
- **Augur** — Skinnider et al. 2021, *Nat Biotechnology* (protocol: *Nat Protocols* 16:3836–3873)
  **[✓web]**. Per cell type, a cross-validated random forest reports the AUC of separating the condition,
  with abundance-balanced subsampling; the recommended method (single-cell best practices) for
  prioritizing perturbation-responsive cell types; reported to "outperform DE-based methods." This *is*
  our supervised probe. **[STANDARD]**
- **Linear probing** of representations — the generic representation-learning standard for "is X linearly
  decodable from the embedding." **[STANDARD]**
- **Milo** — Dann et al. 2022, *Nat Biotechnology* 40:245 **[✓web]**. Differential *abundance* on a kNN
  graph (negative-binomial GLM per neighbourhood). A different question (fraction shift, not expression)
  but directly relevant to our activity-confound: a cell-fraction change can masquerade as an expression
  change. **[STANDARD for DA]**
- Related: MELD, DA-seq (differential abundance); LISI / kBET (batch-mixing, repurposable as "how much
  does factor X separate"); distance correlation / energy distance / MMD (effect size on distributions).
  **[emerging / niche]**

---

## 3. Our three added measures — precedent & pitfalls

- **Supervised-subspace probe (PLS-DA / CV classifier).** PLS-DA is the standard p≫n supervised
  projection in omics/chemometrics; the documented pitfall is an over-optimistic in-sample AUC without a
  permutation null (Westerhuis et al. 2008, *Metabolomics* 4:81, "Assessment of PLSDA cross validation"
  **[mem]**) — which is exactly why our probe is **cross-validated and permutation-calibrated**, and why
  Augur (its single-cell incarnation) subsamples + cross-validates. **[STANDARD with the CV+perm guard]**
- **Per-dimension max-η² with a max-statistic permutation null** — the Westfall–Young max-T family of
  multiple-testing corrections **[mem]**; the max-stat null correctly accounts for scanning 768 dims.
  **[STANDARD]**
- **Standardized (column-z-scored) trace η²** = unweighted mean per-dimension η². A reasonable diagnostic
  lens (removes the variance weighting), though it can amplify noise dimensions; here it ≈ the global η²,
  which was itself the informative result — the exercise signal is a *diagonal* direction, not a
  low-variance axis. **[niche / diagnostic]**

---

## 4. Foundation-model-specific evaluation

- Foundation-model papers (scGPT, Geneformer, scFoundation, UCE, GeneCompass) evaluate representation
  quality mainly via **batch-integration / label-conservation metrics** (scIB; Luecken et al. 2022,
  *Nat Methods* **[mem]**) and label probes, and condition effects via **in-silico perturbation**
  (Geneformer; GEARS, Roohani et al. 2023, *Nat Biotech* **[mem]**; scGPT perturbation). **[STANDARD for
  those tasks]**
- **Caution — "Benchmarking Transcriptomics Foundation Models for Perturbation Analysis: one PCA still
  rules them all,"** arXiv 2410.13956 **[✓web]**: across perturbation benchmarks, simple PCA matches
  foundation-model embeddings. Our PCA control reproduces the spirit — GeneCompass beats PCA-50 only
  modestly and ties a full-gene probe, so the *supervised signal*, not the representation, carries the
  exercise effect.

---

## Bottom line for Aim 2

1. The gate (PERMANOVA R²) and the supervised probe (Augur) are both standard; we ran **canonical Augur**
   and it corroborates the probe (Spearman r = 0.83, 19–22/22 hotspots).
2. Do the DE in **gene space, per cell type, pseudobulk-style** (Squair) on `Z` —
   `expression ~ ordinal_dose + sex` — **not** on the embedding. Optionally CPA for latent dose modeling.
3. Value-add over PCA was sanity-checked (modest), per the "one PCA" caution.

So our plan (PERMANOVA-style gate → Augur-style prioritization → pseudobulk gene-space DE) **aligns with
precedent at every step**; the one course-correction the literature enforces is to keep the DE itself in
gene space rather than on the embedding.

---

## Sources (web-verified this session)

- Augur — https://www.nature.com/articles/s41596-021-00561-x · https://pubmed.ncbi.nlm.nih.gov/34172974/
- Squair 2021 (pseudobulk DE) — https://www.nature.com/articles/s41467-021-25960-2
- scVI empirical-Bayes DE (Boyeau/Lopez) — https://www.pnas.org/doi/abs/10.1073/pnas.2209124120
- PERMANOVA / adonis (vegan) — https://vegandevs.github.io/vegan/reference/adonis.html
- Milo — https://www.nature.com/articles/s41587-021-01033-z
- Single-cell best practices (perturbation modeling) — https://www.sc-best-practices.org/conditions/perturbation_modeling.html
- "One PCA still rules them all" — https://arxiv.org/pdf/2410.13956
