# Open Frontier — what's left across the full proposal

Status tracker for the whole grant (NIH RFA-RM-24-011: rat GeneCompass foundation model applied to the
MoTrPAC rat endurance-exercise study, translated to human). It maps every Aim/module to its current state,
effort, and blockers, and is meant to be updated as work lands. Per-module build specs (file names,
dependency graph, recommended order) live in [`deconvolution/DOWNSTREAM_BUILD_PLAN.md`](../deconvolution/DOWNSTREAM_BUILD_PLAN.md).

**Last updated:** 2026-10-25. The working model is now **retrain v2** (the default in
`config/pipeline_config.yaml`; April's model is `config/profiles/april.yaml`). Every model-dependent result
below is v2's; the April-vs-v2 comparison and the reruns are in [`RETRAIN_V2_RESULTS.md`](RETRAIN_V2_RESULTS.md).
**Legend:** ✅ done · ◐ partial · ☐ unbuilt · ⛔ blocked (external) · ⊘ out-of-scope.

**Recently closed (2026-07-06/13, branch `aim2b-perturbation-engine`):** **the keystone landed.** Module A
(rat in-silico perturbation engine) is built and its pre-registration is **CLOSED — all 4 criteria PASS**
(`b162409`); Module A.5 (rat TF list) shipped with it (`bca3221`). Module B (model-driven differential GRN)
is built, including the dose-pooling refinement (`ffbebff`, `fb07c3c`). **Aim 3b** (perturbation on the
already-transferred human cells) + the human-space GRN self-consistency score, pathway enrichment, DE robustness, and the
per-tissue purity-sweep extension all deployed (`bbf33de`). These are now orchestrated as **Stage 13**
(mechanism) and **Stage 14** (hardening) (`bb47beb`). Finally, a non-injective cell-type filename sanitizer
was root-caused and fixed (`f4cbf12`) — it had destroyed a BayesPrism Z matrix; DE, enrichment, and the
kidney/hippocampus purity sweeps were rebuilt. Headline biology unchanged (then 185 blocks / 21 hotspots).

**Retrain v2 (2026-10-06 → 10-25, pre-registered in [`RETRAIN_V2_PREREGISTRATION.md`](RETRAIN_V2_PREREGISTRATION.md)):**
the model was retrained on a cleaned corpus (584 QC files kept of 868), scheme-B medians and fixed priors.
C1–C2 pass (rat learned, human/mouse kept), so v2 replaced April and everything downstream was rerun with it.
What changed in the science: 13 exercise hotspots instead of April's 16 by the same gate method, all 13
keep their signal in human space, but the rat↔human GRN self-consistency drops (per-TF median ρ 0.292 vs
0.451) and Module A's target-enrichment check fails under v2 (see the rows below).

**Earlier (2026-07, PR `deconv-reference-integrity`):** reference-integrity hardening (QC gate +
canonical `tissue_references.yaml` manifest + from-scratch driver `run_deconv_all.py`; liver-Visium and
engineered-lung references fixed), the full omnideconv cross-method panel + mRNA-bias analysis + downstream-
claims guidance, and the lung fix (0→3 exercise hotspots). These moved the headline figures.

**Verified canonical figures** (v2; so numbers below are consistent): **8,450 pseudo-cells** (14 tissues);
**172 per-cell-type DE blocks / 13 exercise hotspots** (Aim-2 gate, 10 splits × 10,000 permutations;
April by the same method: 16); 2,017,173 gene-block DE tests; model
`rat_genecompass_finetuned_v2/models/rat_phase2_mixed_species` (= `checkpoint-591763`); ortholog map 15,234
rat→human pairs (~69% of 22,213 rat genes; 100% get a token, incl. random-init T4); cross-species transfer —
**13 of 13 hotspots preserved** (April 14 of 16). The script's pooled sex check (human median ≥ 0.70) prints
FAIL for both models; block by block sex carries over closely (Spearman 0.90). April's July figures
(15 hotspots, 13 of 15 transferred) are in git history (ad72cfa).

---

## Done ✅ (do not re-litigate)

| Aim | What | Evidence |
|---|---|---|
| **1** | Rat GeneCompass model (pipeline Stages 1–7): corpus, ortholog tokens, medians, tokenization, two-phase mixed-species fine-tuning. **v2** (retrained 2026-10; April's checkpoint-147941 kept as `config/profiles/april.yaml`) | v2 `checkpoint-591763`; `docs/RETRAIN_V2_RESULTS.md`, `reports/pipeline_report.md` |
| **1** | Catastrophic-forgetting check (pre-registered C1–C2, PASS): rat gene-ID loss 7.13 → 3.06 (v2 init → v2); human +0.05% and mouse +1.32% vs base (limit +10%) | `docs/RETRAIN_V2_RESULTS.md` C1–C2; April's: `reports/forgetting/forgetting_report.md` |
| **1** | Representation-quality validation trio (§3.6) — see [Aim-1 validation results](#aim-1-validation-results) | `reports/aim1_validation_v2/` |
| **2a** | Per-cell-type DE on deconvolved Z (limma-trend, IHW; repfdr's fit is degenerate, so sex-consistency calls are concordance-only); **172 blocks, 13 exercise hotspots**; pre-registered positive controls: all 81 outcomes as April's (Tier A 4/45, Ai 2/14, B 7/22) | `docs/RETRAIN_V2_RESULTS.md` (Stage 10); `deconvolution/AIM2_DECONV_RESULTS.md §4a` |
| **3a** | Cross-species **transfer** of the rat exercise response into human embedding space; **13 of 13 hotspots preserved** (April 14 of 16); v2's block order carries over less closely than April's (trained-AUC Spearman rat~human 0.616 vs 0.762) | `docs/RETRAIN_V2_RESULTS.md` (Stage 12); `AIM2_DECONV_RESULTS.md §4b` |
| **2b** | **Rat in-silico perturbation engine (the keystone)** — April's pre-registration CLOSED, all 4 criteria PASS. **Under v2: 3 of 4 pass; #2 (Mef2c targets in the curated muscle/mito set) FAILS** (0/25; the set was written after April's smoke run and is not neutral between models). + the model-driven **differential GRN** (dose-pooled, bootstrapped; v2: 9 networks). | commits `bca3221`, `b162409`, `ffbebff`, `fb07c3c`; Stage 13; `RETRAIN_V2_RESULTS.md` (Module A) |
| **3b** | **Perturbation on the transferred human cells** — human-space GRN + a rat↔human **self-consistency** score (v2 per-TF median ρ **0.292**; April 0.451). ⚠️ NOT cross-species biological conservation (same cells through the same model) — see the Aim-3b row below. | commit `bbf33de`; Stage 13; `RETRAIN_V2_RESULTS.md` (Stage 13) |
| **hardening** | Purity sweeps (12 of 14 tissues swept, all meet the Chu-2022 bar; model-independent), composition-confound on all 172 blocks, RIN/globin robustness (v2: 11 of 13 hotspots ROBUST) | commit `bbf33de`; Stage 14 |
| **hardening** | Reference integrity (QC gate + manifest + from-scratch driver; liver-Visium + engineered-lung fixed); full omnideconv panel + mRNA-bias analysis | PR `deconv-reference-integrity`; `REFERENCE_QC.md`, `OMNIDECONV_RESULTS.md` |

---

## Aim 1 validation results

The three checks promised in `reports/pipeline_report.md §3.6`, run on v2 (2026-10-24; code
`analysis/aim1_validation/`, report `reports/aim1_validation_v2/AIM1_VALIDATION_REPORT.md`). April's run
(2026-06-25/26, `reports/aim1_validation/`) in brackets.

- ✅ **Homolog embedding similarity — PASS.** Contextual gene embeddings place rat genes far closer to their
  true human/mouse ortholog than to a permuted one: ROC-AUC **1.000** (T1) [1.000], **0.948** (T3a vs human)
  [0.973], **0.851** (T3b vs mouse) [0.908]. v2's T3 genes sit less close to their orthologs than April's,
  as their drift from init in C4b shows.
- ◐ **Held-out cell-type clustering — PARTIAL.** kNN cell-type purity **0.849** (chance 0.065) [0.813], but
  the cell embeddings are **not batch-corrected**: kNN study purity 0.966 [0.975], cross-study mixing within a
  cell type 3.6% [2.0%]. Each model draws its own labelled subset (they share 1,935 cells), so this is a rerun,
  not a paired comparison; C3 compares the two on the same reference cells (v2 better in 12 of 12). Largely
  insulated from Aims 2–3, which use pseudo-cells *within* tissue.
- ◐ **T4 new-token quality — WEAK.** Random-init rat-specific tokens did not acquire cross-species family
  structure at the input layer (CYP rank-AUC 0.569 [0.601]; pooled 0.471 [0.527], ≈ chance). Only ~4 clean
  CYP genes; tests input embeddings only.

---

## Open frontier

### Aim 2b/2c + 3b — model-driven science (largest unbuilt block; pivots on the keystone)

| Status | Item | Aim | Effort | Blocker |
|---|---|---|---|---|
| ✅ | **Module A — rat in-silico perturbation engine (KEYSTONE)** — DONE. April's pre-registration **CLOSED, all 4 criteria PASS** (Mef2c 98th pctile; self-consistency ρ=0.70; target enrichment p=2e-7). Rerun on the current SKMVL cells: April 4/4; **v2 3/4 — #2 target enrichment FAILS** (0/25 vs April 4/25), with the curated set's provenance caveat (`RETRAIN_V2_RESULTS.md`). Use the per-gene **target** shift, not the near-noise cell shift. | 2b/3b | — | — |
| ✅ | Module A.5 — rat TF list (AnimalTFDB → ENSRNOG ∩ expressed) — DONE (shipped with Module A). | 2b/3b | — | — |
| ✅ | Module B — model-driven differential GRN — DONE, and **REBUILT UNCAPPED 2026-07-13** (`grn_uncapped/`: 3,478,564 edges / 398,586 reproducible / 389 TFs; `delta` bit-identical to the old capped tables → strict superset). Top hubs = Tfeb/**Mef2c**/Epas1/Myf6/**Nr4a1** — but **only under the corrected statistic `sum\|Δ\|`**; ranking by *count* of `\|z\|≥2` edges inverts the biology (anchors 33rd pctile, Nr4a1 17th) because `z` diverges where the model is numerically inert. See [[project_grn_hub_statistic]]. **v2 (2026-10-25):** 9 hotspot networks (`grn_v2/`); hub strength mostly follows each TF's own expression in the profile (Spearman ~0.8), so hub rankings are largely input-driven; the two models agree on hub order but not on individual edges. | 2b | — | — |
| ☐ | Module C — CPA dose model (ordinal week; compare to DE slope) | 2c | L (GPU) | none (n≈50/cell-type thin) |
| ✅ | Aim 3b — perturbation on the transferred human cells — DONE, rebuilt uncapped. **⚠️ RE-SCOPED 2026-07-13: this is NOT cross-species biological conservation.** The transfer re-expresses the *same* pseudo-cells (identical 700 cells, same order) and **91.1% of orthologous rat genes carry the identical GeneCompass token as their human ortholog** — so the human-space perturbation deletes the same token from a near-identical sequence in the same model. No human measurement enters. It is a **model self-consistency** score (April median ρ 0.451, unbiased; the published 0.507 was selection-inflated; **v2 0.292** — lower in 566 of 729 TF scores both models make; the drop cannot be split between v2's weights and its tokenization). **The capstone therefore has TWO independent evidence legs, not three.** | 3b | — | — |
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
| ✅ | **Lung** — DONE: engineered GSE178405 → native pooled reference; lung went from 0 exercise hotspots to real immune/stromal ones (v2: endothelial cells + macrophages, both preserved rat→human) (was ◐ weak) | — |
| ✅ | Per-tissue expression **purity sweeps** — DONE (Stage 14; model-independent; 13 sweeps on disk across 12 tissues). We **meet the Chu-2022 bar** (>0.95 above ~50% focal purity) on all **12 tissues swept** (of the 14-tissue panel): ≥0.985 at 50% purity, ≥0.99 at 85%. The 4 tissues added in the 2026-07-16 rebuild (BAT/HYPOTH/SMLINT/TESTES) were **swept 2026-07-27 and all pass** (VST ≥0.998 at 50% purity; HYPOTH glutamatergic-neuron VST = 1.000 throughout). Only BLOOD and subcutaneous fat remain unswept — neither has a capturable dominant parenchyma. | — |
| ✅ | Technical covariates (**RIN / %globin**) robustness on every hotspot — DONE (`rin_globin_robustness.tsv`): v2 **11 of 13 hotspots ROBUST**; the 2 CHECKs (LUNG endothelial cells, keeps 0.77 of its dose genes; LUNG macrophages, 0.78) are reported, not hidden. Composition (θ) confound on all 172 blocks: 114 QUIET / 46 PASS_EXPRESSION / 12 FLAG_COMPOSITION (model-independent); v2's hotspots 9 PASS_EXPRESSION / 2 FLAG_COMPOSITION (BLOOD NK, naive B) / 2 QUIET. | — |
| ◐ | **E.3** Tabula Sapiens human-atlas backdrop (external identity check the sex-gate doesn't provide; needs the atlas) | S |
| ☐ | **Second heart reference** — heart CM is the one cardinal cell type with *holdout-only* validation (`GSE280111` LV, CM r≈0.995). Needs an independent native adult healthy rat-LV snRNA study. Data availability is the whole blocker; the machinery is tissue-agnostic and built. | S |
| ☐ | **(new, 2026-07-13) A valid null for the GRN's `|z|` statistic.** `|z|≥2` is a *ranking* device, **not** FDR-calibrated, and a permutation null is **impossible** for this design — `label` is a deterministic function of `week` (week 0 ⟺ control), so no shuffle can break the label while preserving dose (`deconvolution/GRN_NULL_WHY_ABANDONED.md`). Any FDR statement about GRN edges is currently unsupported. A valid null needs a non-dose-confounded contrast. | M |
| ☐ | **(new, 2026-07-13) Degenerate-block gate in the DE.** `run_pseudobulk_de.R` sets `status="ok"` unconditionally on the success path — it records `median_libsize` / `frac_zero` and never gates on them. KIDNEY's over-split intercalated labels leave two near-empty blocks (`Intercalated cells`, median libsize **51**; `Beta-intercalated cells`, median libsize **10**, 80% zeros) marked `ok` and pooled into the **global IHW/repfdr fit** — the same failure class as the phantom duplicate that forced the `f4cbf12` re-run. Neither is a hotspot, so no headline result rests on them; the exposure is to the shared calibration and is unmeasured. Fix = degeneracy gate **or** merge the over-split reference labels; either changes the 172 count and needs a DE re-run → a deliberate science call. | S–M |
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
