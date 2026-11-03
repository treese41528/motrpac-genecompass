# Open Frontier — what's left across the full proposal

Status tracker for the whole grant (NIH RFA-RM-24-011: rat GeneCompass foundation model applied to the
MoTrPAC rat endurance-exercise study, translated to human). It maps every Aim/module to its current state,
effort, and blockers, and is meant to be updated as work lands. Per-module build specs (file names,
dependency graph, recommended order) live in [`deconvolution/DOWNSTREAM_BUILD_PLAN.md`](../deconvolution/DOWNSTREAM_BUILD_PLAN.md).

**Last updated:** 2026-11-01. The working model is **retrain v2** and the working references are **refs4**
(the jointly re-annotated references, `references_v4/`) over **all 16 tissues** (refs4's 14 plus SPLEEN and
OVARY), the default in `config/pipeline_config.yaml`. The 14-tissue refs4 arm is
`config/profiles/refs4_14.yaml`; the July references are `config/profiles/july_refs.yaml`; April's model is
`config/profiles/april.yaml`. Every result below is v2 on refs4 unless marked; headline figures are the
16-tissue ones (`deconvolution/ALL_TISSUE_RERUN_RESULTS.md`), with the 14-tissue refs4 figures as comparison. **Legend:** ✅ done · ◐ partial · ☐ unbuilt · ⛔ blocked
(external) · ⊘ out-of-scope.

**Recently closed (2026-11-01).**
- **All-tissue (16) rerun: done, now the default** (c3b27c7, `deconvolution/ALL_TISSUE_RERUN_RESULTS.md`).
  SPLEEN and OVARY joined: 197 blocks, 151 analysable, **9 exercise hotspots** (refs4: 11). LUNG endothelial
  and mesothelial cells drop out (q 0.0440 → 0.0535, AUC unchanged) because the BH family is larger. No
  SPLEEN or OVARY hotspot. Embedding beats PCA is no longer significant (p 0.15; refs4 p 0.013). Single-sex
  tissues (TESTES, OVARY) can never have a dose-significant gene under the Fisher sex combine: TESTES's 0
  genes is structural, not biology.

**Closed 2026-10-31.**
- **refs4 adopted as the default** (`904dfb2`). The references were relabelled by one joint procedure, BLOOD got
  a whole-blood composite, and everything from Stage 8 on was rerun (`deconvolution/REFS4_RERUN_RESULTS.md`).
- **The DE near-empty-block gate is closed.** It is the §6 rule (a block is analysed only if its median
  per-sample library is at least 1,000), applied to every refs4 table, with sensitivities at 100 and 0.
- **GRN `|z|` null: done, and negative** (`deconvolution/GRN_DECOY_NULL_RESULTS.md`). Against expression-matched
  random-gene deletions, `|z| ≥ 2` edges have FDR 1.03, and 0 of 981 TFs beat their decoys on hub strength.
  The GRN carries no TF-specific signal; edges are no longer called "confident".
- **E.3 Tabula Sapiens: PASS** (`translation/ATLAS_PROJECTION_RESULTS.md`). Transferred rat cells land on
  their real-human counterpart cell type at 0.404 macro accuracy against 0.084 by chance (p = 0.0001).
- **Aim-1 per-tier loss** (`docs/RETRAIN_V2_RESULTS.md`, last section). v2's gain is in the mouse-ortholog and
  new rat-only tokens, which April never learned.
- **Second heart reference: PASS** (§15). An independent F344 heart study recovers cardiomyocyte fractions
  through the production HEART reference at r 0.944 (mean error 0.043).
- **More MoTrPAC tissues** (`REFERENCE_RELABEL_PREREG.md` §14): SPLEEN and OVARY references pass their checks
  and are deconvolved; the all-tissue (16) rerun is done (above). ADRNL, COLON and VENACV still have none.

**Retrain v2 (2026-10-06 → 10-25, pre-registered in [`RETRAIN_V2_PREREGISTRATION.md`](RETRAIN_V2_PREREGISTRATION.md)):**
the model was retrained on a cleaned corpus (584 QC files kept of 868), scheme-B medians and fixed priors.
C1–C2 pass (rat learned, human/mouse kept), so v2 replaced April and everything downstream was rerun with it.

**Verified canonical figures** (v2 + refs4, all 16 tissues, 2026-11-01): **9,232 pseudo-cells** (9,214
embedded); **197 cell-type blocks, 151
analysable under §6, 9 exercise hotspots** (Aim-2 gate, 10 splits × 10,000 permutations, BH q < 0.05); 53,084
dose-significant genes (FDR_IHW < 0.05; SPLEEN 6,439, OVARY 0); **9 of 9 hotspots preserved** in human
transfer; all 9 RIN/globin ROBUST; 6 GRNs, rat–human self-consistency median ρ 0.310 (633 TFs). The 14-tissue
refs4 figures were 8,100 pseudo-cells (8,082 embedded; 18 LUNG goblet pseudo-cells fall below the tokenizer's
200-gene minimum), 165 blocks, 124 analysable, 11 hotspots, 48,522 dose-significant genes, 11 of 11
preserved, 8 GRNs (`deconvolution/REFS4_RERUN_RESULTS.md`). Model
`rat_genecompass_finetuned_v2/models/rat_phase2_mixed_species` (= `checkpoint-591763`); ortholog map 15,234
rat→human pairs. The July-reference v2 figures (172 blocks, 13 hotspots) and April's are in
`docs/RETRAIN_V2_RESULTS.md` and git history.

The 9 hotspots: BLOOD classical monocytes, erythroid cells, megakaryocytes and platelets, neutrophils; LUNG
vascular smooth muscle cells; SKMGN tenocytes; SKMVL satellite cells and vascular smooth muscle cells; HYPOTH
oligodendrocyte precursor cells. refs4 (14 tissues) also had LUNG endothelial and mesothelial cells.

---

## Done ✅ (do not re-litigate)

| Aim | What | Evidence |
|---|---|---|
| **1** | Rat GeneCompass model (pipeline Stages 1–7): corpus, ortholog tokens, medians, tokenization, two-phase mixed-species fine-tuning. **v2** (retrained 2026-10; April's checkpoint-147941 kept as `config/profiles/april.yaml`) | v2 `checkpoint-591763`; `docs/RETRAIN_V2_RESULTS.md`, `reports/pipeline_report.md` |
| **1** | Catastrophic-forgetting check (pre-registered C1–C2, PASS): rat gene-ID loss 7.13 → 3.06 (v2 init → v2); human +0.05% and mouse +1.32% vs base (limit +10%) | `docs/RETRAIN_V2_RESULTS.md` C1–C2 |
| **1** | Per-tier masked-prediction loss (2026-10-31): T2b/T3b/T4 tokens 19–20 → 2.2–2.4 nats (v2 init → v2); April stays at 11–14 on them | `docs/RETRAIN_V2_RESULTS.md` (per-tier section); `eval_mlm.py --by-tier` |
| **1** | Representation-quality validation trio (§3.6) — see [Aim-1 validation results](#aim-1-validation-results) | `reports/aim1_validation_v2/` |
| **2a** | Per-cell-type DE on deconvolved Z (limma-trend, IHW), refs4 + §6 rule, 16 tissues: **197 blocks, 151 analysable, 9 exercise hotspots**, 53,084 dose-significant genes; pre-registered positive controls Tier A 6/45, Ai 2/14, B 7/22 recovered (refs4, 14 tissues: 165 / 124 / 11 hotspots / 48,522 genes; Tier A 5/45). Single-sex tissues (TESTES, OVARY) cannot have a dose-significant gene (Fisher sex combine) | `deconvolution/ALL_TISSUE_RERUN_RESULTS.md`; `REFS4_RERUN_RESULTS.md`; `AIM2_DECONV_RESULTS.md` |
| **3a** | Cross-species **transfer** of the rat exercise response into human embedding space; **9 of 9 hotspots preserved**; rat-to-human Spearman of the exercise AUC 0.766 (refs4 0.751 with 11 of 11; July 0.616) | `ALL_TISSUE_RERUN_RESULTS.md`; `REFS4_RERUN_RESULTS.md` (Stage 12) |
| **3a** | **E.3 atlas anchoring — PASS** (pre-registered): transferred rat cells sit nearest their Tabula Sapiens counterparts (macro accuracy 0.404 vs null 0.084, p = 0.0001). Fails for 3 hotspots (SKMGN tenocytes, LUNG vascular smooth muscle, BLOOD erythroid); LUNG mesothelial and HYPOTH OPC have no atlas counterpart. Run on the 14-tissue refs4 hotspots (11) |  `translation/ATLAS_PROJECTION_RESULTS.md` |
| **2b** | **Rat in-silico perturbation engine** — April's pre-registration CLOSED, all 4 criteria PASS. **Under v2: 3 of 4 pass; #2 (Mef2c targets in the curated muscle/mito set) FAILS** (0/25; the set was written after April's smoke run and is not neutral between models). | commits `bca3221`, `b162409`; `RETRAIN_V2_RESULTS.md` (Module A) |
| **2b** | Model-driven **differential GRN** (dose-pooled, bootstrapped; 16 tissues: 6 networks, identical to six of refs4's 8). **Decoy null (2026-10-31): no TF-specific signal** — `\|z\| ≥ 2` FDR 1.03 against expression-matched random genes; 0 of 981 TFs beat their decoys on hub strength. Hub tables are descriptive model readouts, not regulator calls. | `ffbebff`, `fb07c3c`; Stage 13; `GRN_DECOY_NULL_RESULTS.md` |
| **3b** | **Perturbation on the transferred human cells** — human-space GRN + a rat↔human **self-consistency** score (median ρ **0.310** over 633 TFs, 16 tissues; refs4 0.310 over 859; July 0.292; April 0.451). NOT cross-species biological conservation (same cells through the same model), and after the decoy null, self-consistency of an expression-driven quantity. | `bbf33de`; Stage 13; `REFS4_RERUN_RESULTS.md` |
| **hardening** | Purity sweeps (12 of 14 tissues, all meet the Chu-2022 bar; model-independent; SPLEEN and OVARY not swept), composition confound on all blocks (16 tissues: QUIET 112 / PASS_EXPRESSION 62 / FLAG 23; hotspot SKMVL VSMC flagged; refs4: 91 / 56 / 18, LUNG mesothelial also flagged), RIN/globin robustness (**9 of 9 hotspots ROBUST**; refs4 11 of 11) | Stage 14; `ALL_TISSUE_RERUN_RESULTS.md`; `REFS4_RERUN_RESULTS.md` |
| **hardening** | DE near-empty-block gate: the §6 rule (median per-sample library ≥ 1,000), with sensitivities at 100 and 0; the hotspot set is the same at 1,000 and 100 | `REFERENCE_RELABEL_PREREG.md` §6; `REFS4_RERUN_RESULTS.md` |
| **hardening** | Joint re-annotation of the references (refs4) + BLOOD whole-blood composite; reference integrity (QC gate + manifest + from-scratch driver); full omnideconv panel + mRNA-bias analysis | `REFERENCE_RELABEL_PREREG.md`; `REFERENCE_QC.md`, `OMNIDECONV_RESULTS.md` |

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
| ✅ | Module B — model-driven differential GRN — DONE, and **REBUILT UNCAPPED 2026-07-13** (`grn_uncapped/`: 3,478,564 edges / 398,586 reproducible / 389 TFs; `delta` bit-identical to the old capped tables → strict superset). Top hubs = Tfeb/**Mef2c**/Epas1/Myf6/**Nr4a1** — but **only under the corrected statistic `sum\|Δ\|`**; ranking by *count* of `\|z\|≥2` edges inverts the biology (anchors 33rd pctile, Nr4a1 17th) because `z` diverges where the model is numerically inert. See [[project_grn_hub_statistic]]. **v2 (2026-10-25):** 9 hotspot networks (`grn_v2/`); hub strength mostly follows each TF's own expression in the profile (Spearman ~0.8), so hub rankings are largely input-driven; the two models agree on hub order but not on individual edges. **refs4 decoy null (2026-10-31): FDR(2) = 1.03 and 0/981 TFs beat expression-matched decoys, so the hubs (Mef2c, Nr4a1, Epas1, Tfeb included) are not TF-specific; the GRN is reported as a descriptive model readout only** (`GRN_DECOY_NULL_RESULTS.md`). | 2b | — | — |
| ☐ | Module C — CPA dose model (ordinal week; compare to DE slope) | 2c | L (GPU) | none (n≈50/cell-type thin) |
| ✅ | Aim 3b — perturbation on the transferred human cells — DONE, rebuilt uncapped. **⚠️ RE-SCOPED 2026-07-13: this is NOT cross-species biological conservation.** The transfer re-expresses the *same* pseudo-cells (identical 700 cells, same order) and **91.1% of orthologous rat genes carry the identical GeneCompass token as their human ortholog** — so the human-space perturbation deletes the same token from a near-identical sequence in the same model. No human measurement enters. It is a **model self-consistency** score (April median ρ 0.451, unbiased; the published 0.507 was selection-inflated; **v2 0.292 on the July references; 0.310 on refs4** — lower in 566 of 729 TF scores both models make; the drop cannot be split between v2's weights and its tokenization). **The capstone therefore has TWO independent evidence legs, not three.** | 3b | — | — |
| ✅ | Stage orchestrators — realized as **Stage 13** (mechanism: perturb→GRN→conservation→enrichment) and **Stage 14** (hardening). *There is no `run_stage11.py`; the build plan's "Stage 11" chain became Stage 13.* | — | — | — |

### Aim 3c — human genetics / disease translation (highest-value *startable now*)

| Status | Item | Effort | Blocker |
|---|---|---|---|
| ☐ | **Module D.0 — stage genetics data** (GTEx v8 eQTL, ~114-GWAS set, S-PrediXcan MASHR/JTI, Open Targets). No GPU, no DE dependency → can start immediately. | M | none |
| ☐ | Module D.1 — `validate_human.py`: replicate Vetr 2024 bulk pipeline (control), then swap in our per-cell-type DE → **cell-type-resolved trait–tissue–gene triplets** (the novel deliverable). | L | D.0 |
| ☐ | Module F — conserved regulators (join perturbation/GRN + genetics). Aim-3 capstone. **2 of its 3 input streams are now on disk** (differential GRN + human-space conservation); it awaits only the genetics. After the decoy null, the GRN stream contributes no TF-specific evidence, so the capstone rests on genetics + DE/transfer. | S | **D** |

### Validation / hardening (completeness, not new aims)

| Status | Item | Effort |
|---|---|---|
| ✅ | **omnideconv** multi-method θ cross-check — DONE: 11-tissue panel × 6 methods + SimBu mRNA-bias battery + dose-response; agreement with the omnideconv paper + downstream-claims bias guidance (`OMNIDECONV_RESULTS.md`) | — |
| ✅ | **Reference integrity** — DONE (2026-07): `reference_qc.py` gate (wired into `build_reference`), `tissue_references.yaml`, `run_deconv_all.py`; liver Visium contamination + engineered→native lung fixed (`REFERENCE_QC.md`) | — |
| ✅ | **Lung** — DONE: engineered GSE178405 → native pooled reference; lung went from 0 exercise hotspots to real immune/stromal ones (16 tissues: vascular smooth muscle cells, preserved rat→human; refs4 (14 tissues): endothelial, mesothelial and vascular smooth muscle cells, all preserved; July: endothelial cells + macrophages) (was ◐ weak) | — |
| ✅ | Per-tissue expression **purity sweeps** — DONE (Stage 14; model-independent; 13 sweeps on disk across 12 tissues). We **meet the Chu-2022 bar** (>0.95 above ~50% focal purity) on all **12 tissues swept** (of the 14-tissue panel): ≥0.985 at 50% purity, ≥0.99 at 85%. The 4 tissues added in the 2026-07-16 rebuild (BAT/HYPOTH/SMLINT/TESTES) were **swept 2026-07-27 and all pass** (focal VST ≥0.992 at 50% purity, HYPOTH lowest; an earlier "≥0.998" / "HYPOTH 1.000 throughout" here was wrong). **Current references (2026-11-02, clean rebuild):** these July/October sweeps predate the refs4 relabel and the TESTES lift for 8 of 11 sweeps; the rebuild re-swept all 11 on the production references and all meet the bar (≥0.992 at 50%, ≥0.998 at 85%; `deconvolution/REFS4ALL_REBUILD_RESULTS.md`, data in `validation_refs4all_rebuild/`). Only BLOOD and subcutaneous fat remain unswept — neither has a capturable dominant parenchyma. | — |
| ✅ | Technical covariates (**RIN / %globin**) robustness on every hotspot — DONE (`rin_globin_robustness.tsv`): 16 tissues **9 of 9 hotspots ROBUST** (median Spearman of the dose statistic with vs without the covariates 0.997, median retention 0.98); refs4 (14 tissues) 11 of 11 (0.957, 0.91); July 11 of 13. Composition (θ) confound on all 197 blocks: 112 QUIET / 62 PASS_EXPRESSION / 23 FLAG_COMPOSITION; one hotspot flagged (SKMVL vascular smooth muscle cells). refs4: 91 / 56 / 18, with LUNG mesothelial cells also flagged. | — |
| ✅ | **E.3** Tabula Sapiens atlas anchoring — **PASS** 2026-10-31 (0.404 vs null 0.084; `translation/ATLAS_PROJECTION_RESULTS.md`). Replaces the manuscript's limitation (v). | — |
| ✅ | **Second heart reference** (`REFERENCE_RELABEL_PREREG.md` §15): GSE307551 F344 LV (independent study), 20 coarse mixtures deconvolved with the production HEART reference — **PASS** 2026-11-01: CM r 0.944, mean absolute CM error 0.043 (bar: r ≥ 0.9, error ≤ 0.10). Every class correlates (r 0.943–0.999) but absolute levels are off (endothelial +0.151, lymphocytes −0.103), so HEART stays read differentially (`data/deconvolution/validation/HEART_GSE307551_cross/`). | — |
| ✅ | **More MoTrPAC tissues** (§14). Of the five tissues without a reference: **SPLEEN** (GSE306764, whole Lewis spleen, 14 types) and **OVARY** (GSE343214, one rat, 18 types) pass every registered check (S1; O1/O2; R1; R2 within-library r ≥ 0.94) and are built, deconvolved and embedded (2026-11-01). The SPLEEN fractions are implausible (T cells about 1%, endothelium 15% from a 34-cell type), so SPLEEN is read differentially only. COLON rejected (GSE326856 is sorted cells, no epithelium); ADRNL and VENACV have no usable public rat single-cell reference. The registered all-tissue (16) rerun is **done** (2026-11-01, c3b27c7; `ALL_TISSUE_RERUN_RESULTS.md`) and is now the default config: 197 blocks / 151 analysable, 9 hotspots (refs4's LUNG endothelial and mesothelial cells cross q 0.05 as the BH family grows), no SPLEEN or OVARY hotspot, SPLEEN 6,439 dose-significant genes, OVARY 0 (single sex: structural). OVARY and TESTES have no Augur score (5 controls per arm). **Open:** a single-sex DE fix would change the registered method and needs its own registration. | — |
| ✅ | **A null for the GRN's `|z|`** — done 2026-10-31 by expression-matched decoy genes, since a label permutation is impossible here (`GRN_NULL_WHY_ABANDONED.md`). **Negative:** FDR(2) = 1.03, no t reaches FDR 0.10, 0/981 TFs beat their decoys (`GRN_DECOY_NULL_RESULTS.md`). | — |
| ✅ | **Degenerate-block gate in the DE** — closed by the §6 rule (median per-sample library ≥ 1,000), applied to every refs4 and all-tissue table, with sensitivities at 100 and 0. The near-empty KIDNEY intercalated blocks are skipped. | — |
| ☐ | Module G — viewer v2 per-cell-type local PCA (lowest value, defer) | M |

### Blocked / out of scope

- ⛔ **CIBERSORTx** — license-gated, Stanford registration pending (the only hard external blocker; off the critical path since BayesPrism is primary).
- ⊘ Cross-tissue cell-type comparison (different reference per tissue) · ⊘ GEARS / drug-CPA / DeepCE (no rat perturb-seq training data).

---

## Aim 1 — methods depth still open

| Status | Item | Effort |
|---|---|---|
| ✅ | Per-tier masked-prediction loss (2026-10-31; see Done) | — |
| ✅ | **End-to-end Stages 1–7 reproduction** (2026-11-01; `docs/repro_stages1_7/REPRODUCTION.md`). Stages 1, 3, 4, 5, the training mix and the eval probe rebuild byte for byte; the Stage 6 GRN is identical; the init differs only in its 3 unseeded random tensors (now `--seed`); a 2,000-step training restart tracks v2's loss within fp16 nondeterminism (39/40 logged points equal to 4 decimals). **Stage 2 does not rebuild byte for byte from raw GEO**: the RGD synonym files were rewritten on 2026-02-24, after the canonical run, so 187 of the kept QC files lose 1–27 genes. The February code gives the same result as HEAD, so the code is ruled out; reference md5s are now pinned and checked. The 27 `keep_raw_slot` files **do** regenerate, with identical counts, through `scripts/restore_rawslot_studies.sh`, and Stage 2 numbering is pinned (`config/stage2_sample_order.tsv`). No effect on v2, whose canonical inputs are on disk and verified. Reproducibility fixes for Stages 0–14: c6be8aa, d38b83a, ec5895b. Deposit list: REPRODUCTION.md, follow-up section. **Stages 8–14 (2026-11-02):** a clean rebuild of the 16-tissue default into fresh folders reproduces it byte for byte through Stage 13 (θ, Z, embeddings, gate, DE, transfer; GRN edges and values identical, tie order now deterministic); `deconvolution/REFS4ALL_REBUILD_RESULTS.md`. | — |
| ☐ | **Fine-tuning ablations** (priors off, no mouse tokens, single-phase). Each is a full two-phase retrain (v2 took about three weeks on 4 A100s), so none is run; they need a dedicated allocation. | L (GPU-weeks) |

---

## Critical path / next moves

Modules A and B are built, and Aim 3b with them, but the GRN decoy null (2026-10-31) shows the GRN carries no
TF-specific signal. The Aim-3 capstone therefore rests on the DE/transfer leg and the genetics leg.

1. **Module D.0 → D.1** (human genetics): the critical path. Public data (GTEx v8 / PredictDB / Open
   Targets / Vetr's ~114-GWAS set), no GPU, Vetr 2024 as the blueprint. Led by the human-genetics
   co-investigator (Dr. Boran Gao), with Yang He on the TWAS side.
2. **Module F** (the Aim-3 capstone) once D lands: join the per-cell-type DE + transfer hotspots with the
   genetics. Its central risk is **circularity** (GeneCompass carries human/mouse regulatory priors), which
   the independence guard + leakage log exist to control.

Off the critical path: **Module C** (CPA dose model, confirmatory, thin-N); the Aim-1 ablations (GPU-weeks);
a single-sex DE method for TESTES and OVARY (needs its own registration).

**Detailed handoff specs** live in `manuscript/wip/` (local-only; see its `README.md`).
