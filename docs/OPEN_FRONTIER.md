# Open Frontier — what's left across the full proposal

Status tracker for the whole grant (NIH RFA-RM-24-011: rat GeneCompass foundation model applied to the
MoTrPAC rat endurance-exercise study, translated to human). It maps every Aim/module to its current state,
effort, and blockers, and is meant to be updated as work lands. Per-module build specs (file names,
dependency graph, recommended order) live in [`deconvolution/DOWNSTREAM_BUILD_PLAN.md`](../deconvolution/DOWNSTREAM_BUILD_PLAN.md).

**Last updated:** 2026-07-03.
**Legend:** ✅ done · ◐ partial · ☐ unbuilt · ⛔ blocked (external) · ⊘ out-of-scope.

**Recently closed (2026-07, PR `deconv-reference-integrity`):** reference-integrity hardening (QC gate +
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
| ☐ | **Module A — rat in-silico perturbation engine (KEYSTONE)**: adapt vendored `perturb_delete_chipseq.py` (~5 rat edits). Unblocks 2b, 3b, and Module F. | 2b/3b | M–L (GPU) | none |
| ☐ | Module A.5 — rat TF list (AnimalTFDB → ENSRNOG ∩ expressed). Startable now. | 2b/3b | S | none |
| ☐ | Module B — model-driven GRN: TF-deletion → differential trained-vs-control network on abundant hotspots | 2b | L | Module A |
| ☐ | Module C — CPA dose model (ordinal week; compare to DE slope) | 2c | L (GPU) | none (n≈50/cell-type thin) |
| ⛔ | Aim 3b — perturbation on the already-transferred human cells | 3b | S | Module A |
| ☐ | Stage 11 orchestrator — chain A→B→C | — | S | A/B/C |

### Aim 3c — human genetics / disease translation (highest-value *startable now*)

| Status | Item | Effort | Blocker |
|---|---|---|---|
| ☐ | **Module D.0 — stage genetics data** (GTEx v8 eQTL, ~114-GWAS set, S-PrediXcan MASHR/JTI, Open Targets). No GPU, no DE dependency → can start immediately. | M | none |
| ☐ | Module D.1 — `validate_human.py`: replicate Vetr 2024 bulk pipeline (control), then swap in our per-cell-type DE → **cell-type-resolved trait–tissue–gene triplets** (the novel deliverable). | L | D.0 |
| ☐ | Module F — conserved regulators (join perturbation/GRN + genetics). Aim-3 capstone. | S | A, B, D |

### Validation / hardening (completeness, not new aims)

| Status | Item | Effort |
|---|---|---|
| ✅ | **omnideconv** multi-method θ cross-check — DONE: 11-tissue panel × 6 methods + SimBu mRNA-bias battery + dose-response; agreement with the omnideconv paper + downstream-claims bias guidance (`OMNIDECONV_RESULTS.md`) | — |
| ✅ | **Reference integrity** — DONE (2026-07): `reference_qc.py` gate (wired into `build_reference`), `tissue_references.yaml`, `run_deconv_all.py`; liver Visium contamination + engineered→native lung fixed (`REFERENCE_QC.md`) | — |
| ✅ | **Lung** — DONE: engineered GSE178405 → native pooled reference; **0→3 exercise hotspots, 2 conserved rat→human** (was ◐ weak) | — |
| ☐ | Per-tissue expression **purity sweeps** (Chu-2022 Fig-1h paper-faithful done only for liver; other 9 untested at that rigor) | M |
| ◐ | **E.3** Tabula Sapiens human-atlas backdrop (external identity check the sex-gate doesn't provide; needs the atlas) | S |
| ☐ | Minor: technical covariates (RIN/globin) absent from DE; activity/composition (θ) confound sweep on the 18 hotspots; heart-CM cross-dataset (holdout-only today) | S |
| ☐ | Module G — viewer v2 per-cell-type local PCA (lowest value, defer) | M |

### Blocked / out of scope

- ⛔ **CIBERSORTx** — license-gated, Stanford registration pending (the only hard external blocker; off the critical path since BayesPrism is primary).
- ⊘ Cross-tissue cell-type comparison (different reference per tissue) · ⊘ GEARS / drug-CPA / DeepCE (no rat perturb-seq training data).

---

## Critical path / next moves

Two moves unlock most of what's left:

1. **Module A** (the perturbation engine) — keystone for Aim 2b GRN, Aim 3b, and Module F.
2. **Module D.0 → D.1** (human genetics) — the clinical translation, and the **lowest-friction high-value start** (public data, zero blockers, Vetr 2024 as a working blueprint).

The Aim-1 validation trio (just completed) closed a literally-promised gap; the remaining Aim-1 items (fine-tuning
ablations, per-tier loss, end-to-end Stages 1–7 reproduction) are lower-priority methods depth.
