# Open Frontier — what's left across the full proposal

Status tracker for the whole grant (NIH RFA-RM-24-011: rat GeneCompass foundation model applied to the
MoTrPAC rat endurance-exercise study, translated to human). It maps every Aim/module to its current state,
effort, and blockers, and is meant to be updated as work lands. Per-module build specs (file names,
dependency graph, recommended order) live in [`deconvolution/DOWNSTREAM_BUILD_PLAN.md`](../deconvolution/DOWNSTREAM_BUILD_PLAN.md).

**Last updated:** 2026-07-13.
**Legend:** ✅ done · ◐ partial · ☐ unbuilt · ⛔ blocked (external) · ⊘ out-of-scope.

**Recently closed (2026-07-06/13, branch `aim2b-perturbation-engine`):** **the keystone landed.** Module A
(rat in-silico perturbation engine) is built and its pre-registration is **CLOSED — all 4 criteria PASS**
(`b162409`); Module A.5 (rat TF list) shipped with it (`bca3221`). Module B (model-driven differential GRN)
is built, including the dose-pooling refinement (`ffbebff`, `fb07c3c`). **Aim 3b** (perturbation on the
already-transferred human cells) + the human-space GRN self-consistency score, pathway enrichment, DE robustness, and the
per-tissue purity-sweep extension all deployed (`bbf33de`). These are now orchestrated as **Stage 13**
(mechanism) and **Stage 14** (hardening) (`bb47beb`). Finally, a non-injective cell-type filename sanitizer
was root-caused and fixed (`f4cbf12`) — it had destroyed a BayesPrism Z matrix; DE, enrichment, and the
kidney/hippocampus purity sweeps were rebuilt. Headline biology unchanged (185 blocks / 21 hotspots).

**Earlier (2026-07, PR `deconv-reference-integrity`):** reference-integrity hardening (QC gate +
canonical `tissue_references.yaml` manifest + from-scratch driver `run_deconv_all.py`; liver-Visium and
engineered-lung references fixed), the full omnideconv cross-method panel + mRNA-bias analysis + downstream-
claims guidance, and the lung fix (0→3 exercise hotspots). These moved the headline figures.

**Verified canonical figures** (so numbers below are consistent): ~9,250 pseudo-cells (lung native ref:
1,350→1,700); **185 per-cell-type DE blocks / 21 hotspots** (was 178/18 pre-lung-fix); ~2.2M DE meta-tests;
fine-tuned checkpoint `rat_phase2_mixed_species/checkpoint-147941`; ortholog map 15,234 rat→human pairs
(~69% of 22,213 rat genes; 100% get a token, incl. random-init T4); cross-species transfer commit `fc4a497`
— **19/21 hotspots survive** (was 16/18). NB: committed `AIM2_DECONV_RESULTS.md` + `notebooks/pipeline8-12`
still show the pre-lung-fix 178/18, pending re-execution (`notebooks/RERUN_EDITS.md`).

---

## Done ✅ (do not re-litigate)

| Aim | What | Evidence |
|---|---|---|
| **1** | Rat GeneCompass model (pipeline Stages 1–7): corpus, ortholog tokens, medians, tokenization, two-phase mixed-species fine-tuning | checkpoint-147941; `reports/pipeline_report.md` |
| **1** | Catastrophic-forgetting validation (human/mouse preserved, rat learned 7.82→3.90 id-loss) | `reports/forgetting/forgetting_report.md` |
| **1** | Representation-quality validation trio (§3.6) — see [Aim-1 validation results](#aim-1-validation-results-2026-06-2526) | `reports/aim1_validation/` |
| **2a** | Per-cell-type DE on deconvolved Z (limma-trend, IHW, repfdr); **185 blocks, 21 exercise hotspots** (post-lung-fix; was 178/18); pre-registered positive-control verdict | `deconvolution/AIM2_DECONV_RESULTS.md §4a` |
| **3a** | Cross-species **transfer** of the rat exercise response into human embedding space; **19/21 hotspots survive** (was 16/18; PLS-1 + Augur agree) | commit `fc4a497`; `AIM2_DECONV_RESULTS.md §4b` |
| **2b** | **Rat in-silico perturbation engine (the keystone)** — pre-registration CLOSED, all 4 criteria PASS; + the model-driven **differential GRN** (dose-pooled, bootstrapped). Independently re-surfaced **Nr4a1**, the regulator the proposal itself predicted. | commits `bca3221`, `b162409`, `ffbebff`, `fb07c3c`; Stage 13 |
| **3b** | **Perturbation on the transferred human cells** — human-space GRN + a rat↔human **self-consistency** score. ⚠️ NOT cross-species biological conservation (same cells, 91.1% identical tokens) — see the Aim-3b row below. | commit `bbf33de`; Stage 13 |
| **hardening** | Purity sweeps (8/10 meet the Chu-2022 bar), composition-confound on all 185 blocks, RIN/globin robustness (18/21 hotspots ROBUST) | commit `bbf33de`; Stage 14 |
| **hardening** | Reference integrity (QC gate + manifest + from-scratch driver; liver-Visium + engineered-lung fixed); full omnideconv panel + mRNA-bias analysis | PR `deconv-reference-integrity`; `REFERENCE_QC.md`, `OMNIDECONV_RESULTS.md` |

---

## Aim 1 validation results (2026-06-25/26)

The three checks promised in `reports/pipeline_report.md §3.6`, run on `checkpoint-147941`
(code `analysis/aim1_validation/`, report `reports/aim1_validation/AIM1_VALIDATION_REPORT.md`):

- ✅ **Homolog embedding similarity — PASS (the key result).** Contextual gene embeddings place rat genes far
  closer to their true human/mouse ortholog than to a permuted one across all tiers: ROC-AUC **1.000** (T1, median
  cosine 0.988 — shared-token orthologs stay consistent even when embedded from rat vs human cells), **0.973**
  (T3a vs human), **0.908** (T3b vs mouse). Validates the cross-species representation that Aims 2–3 rely on.
- ◐ **Held-out cell-type clustering — PARTIAL.** Biology is captured (kNN cell-type purity 0.813 = 17× chance),
  but the cell embeddings are **not batch-corrected** (kNN study purity 0.975 = 16× chance; cross-study mixing
  within a cell type ~2%). Largely insulated from Aim 2/3 (which use per-sample×cell-type pseudo-cells *within*
  tissue), but raw single-cell embeddings shouldn't be pooled across studies without batch handling.
- ◐ **T4 new-token quality — WEAK.** Random-init rat-specific tokens did not acquire strong cross-species family
  structure at the input layer (Cyp450 rank-AUC 0.60; pooled 0.53 ≈ chance). Caveat: only ~4 clean CYP genes; tests
  input embeddings only.

---

## Open frontier

### Aim 2b/2c + 3b — model-driven science (largest unbuilt block; pivots on the keystone)

| Status | Item | Aim | Effort | Blocker |
|---|---|---|---|---|
| ✅ | **Module A — rat in-silico perturbation engine (KEYSTONE)** — DONE. Pre-registration **CLOSED, all 4 criteria PASS** (Mef2c 98th pctile; self-consistency ρ=0.70; target enrichment p=2e-7). Use the per-gene **target** shift, not the near-noise cell shift. | 2b/3b | — | — |
| ✅ | Module A.5 — rat TF list (AnimalTFDB → ENSRNOG ∩ expressed) — DONE (shipped with Module A). | 2b/3b | — | — |
| ✅ | Module B — model-driven differential GRN — DONE, and **REBUILT UNCAPPED 2026-07-13** (`grn_uncapped/`: 3,478,564 edges / 398,586 reproducible / 389 TFs; `delta` bit-identical to the old capped tables → strict superset). Top hubs = Tfeb/**Mef2c**/Epas1/Myf6/**Nr4a1** — but **only under the corrected statistic `sum\|Δ\|`**; ranking by *count* of `\|z\|≥2` edges inverts the biology (anchors 33rd pctile, Nr4a1 17th) because `z` diverges where the model is numerically inert. See [[project_grn_hub_statistic]]. | 2b | — | — |
| ☐ | Module C — CPA dose model (ordinal week; compare to DE slope) | 2c | L (GPU) | none (n≈50/cell-type thin) |
| ✅ | Aim 3b — perturbation on the transferred human cells — DONE, rebuilt uncapped. **⚠️ RE-SCOPED 2026-07-13: this is NOT cross-species biological conservation.** The transfer re-expresses the *same* pseudo-cells (identical 700 cells, same order) and **91.1% of orthologous rat genes carry the identical GeneCompass token as their human ortholog** — so the human-space perturbation deletes the same token from a near-identical sequence in the same model. No human measurement enters. It is a **model self-consistency** score (median ρ 0.451, unbiased; the published 0.507 was selection-inflated). **The capstone therefore has TWO independent evidence legs, not three.** | 3b | — | — |
| ✅ | Stage orchestrators — realized as **Stage 13** (mechanism: perturb→GRN→conservation→enrichment) and **Stage 14** (hardening). *There is no `run_stage11.py`; the build plan's "Stage 11" chain became Stage 13.* | — | — | — |

### Aim 3c — human genetics / disease translation (highest-value *startable now*)

| Status | Item | Effort | Blocker |
|---|---|---|---|
| ☐ | **Module D.0 — stage genetics data** (GTEx v8 eQTL, ~114-GWAS set, S-PrediXcan MASHR/JTI, Open Targets). No GPU, no DE dependency → can start immediately. | M | none |
| ☐ | Module D.1 — `validate_human.py`: replicate Vetr 2024 bulk pipeline (control), then swap in our per-cell-type DE → **cell-type-resolved trait–tissue–gene triplets** (the novel deliverable). | L | D.0 |
| ☐ | Module F — conserved regulators (join perturbation/GRN + genetics). Aim-3 capstone. **2 of its 3 input streams are now on disk** (differential GRN + human-space conservation); it awaits only the genetics. | S | **D** (A and B are done) |

### Validation / hardening (completeness, not new aims)

| Status | Item | Effort |
|---|---|---|
| ✅ | **omnideconv** multi-method θ cross-check — DONE: 11-tissue panel × 6 methods + SimBu mRNA-bias battery + dose-response; agreement with the omnideconv paper + downstream-claims bias guidance (`OMNIDECONV_RESULTS.md`) | — |
| ✅ | **Reference integrity** — DONE (2026-07): `reference_qc.py` gate (wired into `build_reference`), `tissue_references.yaml`, `run_deconv_all.py`; liver Visium contamination + engineered→native lung fixed (`REFERENCE_QC.md`) | — |
| ✅ | **Lung** — DONE: engineered GSE178405 → native pooled reference; **0→3 exercise hotspots, 2 conserved rat→human** (was ◐ weak) | — |
| ✅ | Per-tissue expression **purity sweeps** — DONE (Stage 14; 10 sweeps on disk). We **meet the Chu-2022 bar** (>0.95 above ~50% focal purity) on **8 of 10** tissues: ≥0.986 at 50% purity, ≥0.99 at 85%. BLOOD and subcutaneous fat are excluded honestly — neither has a capturable dominant parenchyma to sweep. | — |
| ✅ | Technical covariates (**RIN / %globin**) robustness on every hotspot — DONE (`rin_globin_robustness.tsv`): **18 of 21 hotspots ROBUST**; the 3 that need a caveat are reported, not hidden. Composition (θ) confound run on all 185 blocks: 145 QUIET / 30 PASS_EXPRESSION / 10 FLAG_COMPOSITION. | — |
| ◐ | **E.3** Tabula Sapiens human-atlas backdrop (external identity check the sex-gate doesn't provide; needs the atlas) | S |
| ☐ | **Second heart reference** — heart CM is the one cardinal cell type with *holdout-only* validation (`GSE280111` LV, CM r≈0.995). Needs an independent native adult healthy rat-LV snRNA study. Data availability is the whole blocker; the machinery is tissue-agnostic and built. | S |
| ☐ | **(new, 2026-07-13) A valid null for the GRN's `|z|` statistic.** `|z|≥2` is a *ranking* device, **not** FDR-calibrated, and a permutation null is **impossible** for this design — `label` is a deterministic function of `week` (week 0 ⟺ control), so no shuffle can break the label while preserving dose (`deconvolution/GRN_NULL_WHY_ABANDONED.md`). Any FDR statement about GRN edges is currently unsupported. A valid null needs a non-dose-confounded contrast. | M |
| ☐ | **(new, 2026-07-13) Degenerate-block gate in the DE.** `run_pseudobulk_de.R` sets `status="ok"` unconditionally on the success path — it records `median_libsize` / `frac_zero` and never gates on them. KIDNEY's over-split intercalated labels leave two near-empty blocks (`Intercalated cells`, median libsize **51**; `Beta-intercalated cells`, median libsize **10**, 80% zeros) marked `ok` and pooled into the **global IHW/repfdr fit** — the same failure class as the phantom duplicate that forced the `f4cbf12` re-run. Neither is a hotspot, so no headline result rests on them; the exposure is to the shared calibration and is unmeasured. Fix = degeneracy gate **or** merge the over-split reference labels; either changes the 185 count and needs a DE re-run → a deliberate science call. | S–M |
| ☐ | Module G — viewer v2 per-cell-type local PCA (lowest value, defer) | M |

### Blocked / out of scope

- ⛔ **CIBERSORTx** — license-gated, Stanford registration pending (the only hard external blocker; off the critical path since BayesPrism is primary).
- ⊘ Cross-tissue cell-type comparison (different reference per tissue) · ⊘ GEARS / drug-CPA / DeepCE (no rat perturb-seq training data).

---

## Critical path / next moves

**The keystone is no longer the constraint.** Module A (the perturbation engine) and Module B (the GRN) are
done, and Aim 3b with them — so the old critical path `M0 → A → B/F` is cleared up to the capstone. What
remains is a single chain:

1. **Module D.0 → D.1** (human genetics) — now the *only* thing on the critical path, and still the
   **lowest-friction high-value start**: public data (GTEx v8 / PredictDB / Open Targets / Vetr's ~114-GWAS
   set), no GPU, no dependency on the fine-tuned model or the perturbation engine, with Vetr 2024 as a
   working blueprint. Led by the human-genetics co-investigator (Dr. Boran Gao).
2. **Module F** (the Aim-3 capstone) — once D lands. Two of its three evidence streams (differential GRN +
   human-space conservation) are already on disk, so the capstone itself is a join and a ranking, roughly a
   day's work. Its central risk is **circularity** (GeneCompass carries human/mouse regulatory priors), which
   the independence guard + leakage log exist to control — recruit an outside skeptic to attack it.

Off the critical path: **Module C** (CPA dose model — confirmatory, thin-N) and the **hardening** items
(E.3 Tabula Sapiens atlas backdrop; a second heart reference; the degenerate-block DE gate above).

The Aim-1 validation trio closed a literally-promised gap; the remaining Aim-1 items (fine-tuning
ablations, per-tier loss, end-to-end Stages 1–7 reproduction) are lower-priority methods depth.

**Detailed handoff specs** for the four remaining items — including what to actually *say* to each
collaborator — live in `manuscript/wip/` (local-only; see its `README.md`).
