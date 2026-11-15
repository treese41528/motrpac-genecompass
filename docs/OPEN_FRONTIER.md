# Open Frontier — what's left across the full proposal

Status tracker for the whole grant (NIH RFA-RM-24-011: rat GeneCompass foundation model applied to the
MoTrPAC rat endurance-exercise study, translated to human). It maps every Aim/module to its current state,
effort, and blockers, and is meant to be updated as work lands. Per-module build specs (file names,
dependency graph, recommended order) live in [`deconvolution/DOWNSTREAM_BUILD_PLAN.md`](../deconvolution/DOWNSTREAM_BUILD_PLAN.md).

**Last updated:** 2026-11-15. The working model is **retrain v2** and the working run is **refs5**: the
registered rerun of Stages 8-14 that followed the 2026-11-11 audit, over **all 16 tissues**, the default in
`config/pipeline_config.yaml` (references in `references_v5/`, exclusion list `rat_exclude_genes_v2.tsv`,
target sum 6,700). The default before it is `config/profiles/refs4all_rebuild.yaml`; the 14-tissue refs4 arm
is `config/profiles/refs4_14.yaml`; the July references are `config/profiles/july_refs.yaml`; April's model is
`config/profiles/april.yaml`. Every current figure below is v2 on refs5 and comes from
`docs/audit/remediation/STEP3_RESULTS.md`, `STEP4_RESULTS.md` or `STEP5_RESULTS.md`; figures of earlier arms
are marked as such. **Legend:** ✅ done · ◐ partial · ☐ unbuilt · ⛔ blocked (external) · ⊘ out-of-scope.

**The research question, and the answer so far.** Can a rat-adapted GeneCompass model, given per-cell-type
expression deconvolved from MoTrPAC's bulk RNA-seq, show which cell types respond to endurance training, which
genes respond in them, and whether the response carries over to the model's human representation?

| Aim | Claim tested | Verdict on refs5 | What qualifies it |
|---|---|---|---|
| 1 | The model learns rat, keeps human and mouse, and generalises to a rat study it never saw | Pass: C1-C2 in sample; held-out loss ratio 1.046 against a bar of 1.10 | Rat-only tokens 1.18. Corpus departures measured and not repaired: 10.4% of training cells above 15% mitochondrial; 1.04% of counts in ID-keyed files under another gene's token |
| 2 bridge | Deconvolution recovers per-cell-type signal | Fractions across samples: floor met on 19 of 19 mixture sets. Expression at a type's own abundance: 4 of 12 called blocks | BayesPrism estimates mRNA share: error against cell fractions rises under native bias on 13 of 15 references. None of the five muscle blocks is recovered |
| 2a gate | Some cell types carry a detectable training signal | 11 hotspots, 1 borderline, of 150 analysable blocks | Membership is not stable to small input changes (7 of the earlier 9 remain). No called block passes all nine registered readings |
| 2a DE | Genes respond to training dose within cell types | 52,961 gene x block pairs, 9,617 genes | No block's calls are robust to technical covariates (44% and 33% of pairs survive). Positive-control red flag stands (5 of 26 testable sanity controls) |
| 2b | In-silico deletions identify regulation | **Negative.** 7 networks, 1,540,997 edges, no TF-specific signal (decoy null) | Module A criteria 1, 3, 4 pass; the myeloid replication fails |
| 3a | The signal carries over to human embedding space | 10 of 11 by the registered rule, 9 of 11 by the corrected rule | Same cells, labels, folds and seeds as the rat test: not a replication. Atlas anchoring not repeated on refs5 |
| 3b | Networks agree between rat and human space | Median 0.48 over 685 scores | Self-consistency of the model, not cross-species biology |

**Closed 2026-11-15: the audit remediation** (`docs/audit/remediation/README.md`; what the audit found:
`docs/audit/FINDINGS.md`).
- **refs5 rerun of Stages 8-14: done, now the default** (`STEP3_RESULTS.md`). Every stage ran from one clean
  commit; Stage 8 repeated byte for byte. What changed against the earlier default:
  - References: all 15 rebuilt from committed code (13 byte-identical); HYPOTH re-annotated; the HEART merge
    applied. Exclusion list rebuilt in Ensembl 115 (464 genes; the old list left 47 to 116 ribosomal protein
    genes in each tissue). Target sum 6,700 by the registered formula.
  - Gate: **11 hotspots and 1 borderline block** of 150 (9 before). Seven of the nine earlier hotspots remain.
    SKMGN tenocytes and HYPOTH oligodendrocyte precursor cells are no longer called; SKMVL tenocytes, SKMGN
    Schwann cells, HEART vascular smooth muscle cells and LUNG endothelial cells are new; SKMGN vascular smooth
    muscle cells are borderline. In the 14 tissues whose reference did not change, a block's AUC moves by 0.03
    at the median and by up to 0.31. Which of the two changes (exclusion list, target sum) moved the blocks was
    not separated.
  - DE: 52,961 pairs. Single-sex tissues have a dose test (TESTES 219 pairs, OVARY 8). Sex-consistent 8-week
    calls: 139 by concordance, 3,623 by repfdr on dose-significant gene-blocks. **The registered check against
    technical covariates fails for every block** (0 of 150 ROBUST).
  - Transfer: 10 of 11 registered, 9 of 11 corrected; LUNG vascular smooth muscle cells lose their dose trend.
  - Networks: difference taken on genes present in both arms; decoy null negative again (pooled estimate 1.02;
    0 of 777 TF x network pairs).
- **Validations** (`STEP4_RESULTS.md`).
  - V1 purity sweeps: dominant types 12 of 15 pass (BLOOD erythroid cells and SPLEEN red-pulp macrophages
    fail; TESTES Sertoli cells not testable). Called blocks: 4 of 12 validated at their abundance.
  - V2 positive controls: the red flag stands (5 of 26 testable sanity controls; 5 of 34 in all).
  - V3 held-out model test: generalises (1.046; per study 0.93 to 1.20; rat-only tokens 1.18).
  - V4 Module A on v2 and refs5: criteria 1, 3 and 4 pass; the myeloid replication fails (Spi1 at the 81st
    percentile; Cebpa not testable).
  - V5 SimBu with a bias-free baseline: BayesPrism's error against cell fractions rises under native mRNA
    differences on 13 of 15 references and under added bias on 14 of 15. The corrector rule is reproduced on
    its registered contrast (C minus A) and not on C minus B; MuSiC and DWLS lose as much as BayesPrism from B
    to C.
  - V6 mixture panel: floor met on 19 of 19 sets; 6 of 192 type tests below 0.85.
- **Corpus and model findings, measured and disclosed; no retrain** (`STEP5_RESULTS.md`): 10.4% of training
  cells above 15% mitochondrial (a floor: twelve files cannot be measured); SMLINT and KIDNEY references 50.6%
  and 16.7%; 1.04% of counts in ID-keyed corpus files and 2.24% of bulk counts under another gene's token (the
  consortium's Myh1 is Myh4 in release 115); 33 never-harvested series became the held-out set; untokenizable
  genes carry at most 0.01% of a tissue's pseudo-cell counts.
- **Statements corrected and records repaired** (`STEP2_STATEMENTS.md`, `STEP6_HYGIENE.md`).

**Open, for the PI** (each is stated in the results files as undecided or untested):
- The expert review of the reference labels: the 160 label decisions and the HYPOTH review decisions were made
  by an AI assistant; SMLINT, KIDNEY, SPLEEN and HIPPOC cell types flagged for mitochondrial share; four
  reference studies above the gene-ID drift bound.
- The TESTES reference choice (pooled or control-only) and the one HYPOTH sample that stands apart.
- A rerun with one change at a time, to say whether the exclusion list or the target sum moved the hotspots.
- Atlas anchoring (E.3), the second heart reference check and the BLOOD composite's checks were run on the
  earlier references and not repeated on refs5.
- The purity-sweep generator's bookkeeping defect (`STEP6_HYGIENE.md`, section 4), left unrepaired.
- The Stage 2 switches for a future corpus (off by default) and the six further Stage 2 behaviours seen on the
  held-out series, none measured on the training corpus.

The blocks below this line, down to "Retrain v2", are the tracker's dated entries for earlier defaults. They
are kept as written; their figures are superseded by the table and the 2026-11-15 entry above.

**Recently closed (2026-11-04 to 11-11): the last checks still on older data, rerun on the 16-tissue default.**
- **E.3 Tabula Sapiens: PASS** (`translation/ATLAS_PROJECTION_16_RESULTS.md`). The macro accuracy is 0.430 over
  120 types in 13 tissues, against 0.086 by chance (p = 0.0001). The first run's 103 types reproduce exactly. SPLEEN
  scores 0.553 and OVARY 0.653.
- **GRN decoy null: still negative** (`deconvolution/GRN_DECOY_NULL_16_RESULTS.md`). FDR(2) is 1.03, and 0 of
  723 TFs beat their decoys.
- **Known-truth mixture panel on the current references** (`deconvolution/MIXTURE_PANEL_REFS4ALL_RESULTS.md`).
  - All 15 references recover their benchmarked types: BayesPrism's median per-type r is 0.989–0.999 holdout and
    0.973–0.996 on 4 cross sets, with six methods including CIBERSORTx.
  - Absolute fractions are weaker: a monocyte sink in KIDNEY_cross (pooled 0.163) and overestimated liver
    hepatocytes.
- **repfdr's sex-consistency fit (2026-11-10)** (`deconvolution/REPFDR_STATUS_16_RESULTS.md`).
  - The fit is not degenerate on the default. It is degenerate again with no library rule, as in July.
  - It fits poorly (f(z) misfit 678 / 614) and supplies 4,982 of the 5,126 calls.
  - Only the 144 per-sex concordance calls (HEART 120, SKMVL 23, SKMGN 1) are read as sex-consistent.
  - A usable repfdr needs a registered refit, for example MoTrPAC's practice of running it only on IHW-significant
    features. Open, for the user to decide.
- **SimBu mRNA-bias battery and dose-response (2026-11-11)** (`deconvolution/SIMBU_BIAS_REFS4ALL_RESULTS.md`;
  pre-registered, July's protocol unchanged, 15 references, six methods).
  - BayesPrism still degrades under the added bias on 15 of 15 references (mean ΔRMSE +0.025).
  - July's corrector ranking is **not reproduced**: MuSiC (+0.025) and SCDC (+0.022) are no longer below
    BayesPrism (+0.025) and CIBERSORTx (+0.022).
  - The driver is the highest-mRNA type on 10 of 15 references. LIVER's is the reference's NK cells, not
    hepatocytes.
  - The driver table and the cross-check advice in `OMNIDECONV_RESULTS.md` are updated (cross-check against DWLS
    or MuSiC).
  - Open, for the user to decide: the NK label on the liver reference's most RNA-rich cells, and a registered
    rerun whose baseline removes the references' native mRNA differences.
- **Full methods audit (2026-11-11)** (`docs/audit/README.md`).
  - A stage-by-stage record of what is done: models, thresholds, flags, statistical procedures and references,
    each tied to a code line or result file.
  - `docs/audit/FINDINGS.md` collects what a reviewer would question, classed by whether it changes how a reported
    result should be read. `docs/audit/REFERENCES.md` holds the reference list, checked against Crossref.
  - The audit changed no code, configuration or result. Its findings are open, for the user to decide.

**Closed 2026-11-01.**
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

**Current figures** (v2 + refs5, all 16 tissues, 749 samples, 2026-11-15; each from a table named in
`docs/audit/remediation/STEP3_RESULTS.md`): **9,060 pseudo-cells embedded** (22 LUNG goblet-cell pseudo-cells
fall below the tokenizer's 200-gene minimum); **194 cell-type blocks, 150 analysable under §6, 11 exercise
hotspots and 1 borderline block** (Aim-2 gate: 10 splits; 10,000 permutations, then batches up to 1,000,000; a
hotspot is a Benjamini-Hochberg call that more permutations could not reverse); **52,961 dose-significant gene
× block pairs** (FDR_IHW < 0.05; 9,617 distinct genes; 41,401 pairs outside the 44 low-fraction blocks; BLOOD
18,745, LUNG 9,802, SPLEEN 7,252, BAT 7,026, TESTES 219, OVARY 8); **10 of 11 hotspots meet the registered
transfer rule and 9 of 11 the corrected rule** (the human test reuses the rat test's cells, labels, folds and
seeds); **no block's DE calls are robust to technical covariates** by the registered verdict; 7 networks with
1,540,997 edges and no TF-specific signal; rat–human self-consistency median 0.48 over 685 TF × network scores
(0.55 over the per-network medians). Model `rat_genecompass_finetuned_v2/models/rat_phase2_mixed_species`
(= `checkpoint-591763`); ortholog map 15,234 rat→human pairs. Figures of earlier arms: the default before
refs5 (197 blocks, 151 analysable, 9 hotspots, 53,084 pairs, 6 networks) in
`deconvolution/ALL_TISSUE_RERUN_RESULTS.md`; refs4 on 14 tissues (165 blocks, 11 hotspots) in
`deconvolution/REFS4_RERUN_RESULTS.md`; the July references (172 blocks, 13 hotspots) and April's in
`docs/RETRAIN_V2_RESULTS.md` and git history.

The 11 hotspots: BLOOD classical monocytes, erythroid cells, megakaryocytes and platelets, neutrophils; LUNG
endothelial cells and vascular smooth muscle cells; HEART vascular smooth muscle cells; SKMVL satellite cells,
tenocytes and vascular smooth muscle cells; SKMGN Schwann cells. Borderline: SKMGN vascular smooth muscle
cells. The list is not stable to small changes of the inputs: of the nine hotspots of the earlier default,
SKMGN tenocytes and HYPOTH oligodendrocyte precursor cells are no longer called. The five muscle blocks are
each below 1% of their tissue, and none has its expression recovered at that abundance.

---

## Done ✅ (do not re-litigate)

| Aim | What | Evidence |
|---|---|---|
| **1** | Rat GeneCompass model (pipeline Stages 1–7): corpus, ortholog tokens, medians, tokenization, two-phase mixed-species fine-tuning. **v2** (retrained 2026-10; April's checkpoint-147941 kept as `config/profiles/april.yaml`) | v2 `checkpoint-591763`; `docs/RETRAIN_V2_RESULTS.md`, `reports/pipeline_report.md` |
| **1** | Catastrophic-forgetting check (pre-registered C1–C2, PASS): rat gene-ID loss 7.13 → 3.06 (v2 init → v2); human +0.05% and mouse +1.32% vs base (limit +10%) | `docs/RETRAIN_V2_RESULTS.md` C1–C2 |
| **1** | Per-tier masked-prediction loss (2026-10-31): T2b/T3b/T4 tokens 19–20 → 2.2–2.4 nats (v2 init → v2); April stays at 11–14 on them | `docs/RETRAIN_V2_RESULTS.md` (per-tier section); `eval_mlm.py --by-tier` |
| **1** | Representation-quality validation trio (§3.6) — see [Aim-1 validation results](#aim-1-validation-results) | `reports/aim1_validation_v2/` |
| **1** | **Held-out test of the model** (registered, 2026-11-15): on 66,000 cells of 33 rat studies that never entered the corpus, v2's gene-identity loss is **1.046** times its in-sample loss (bar 1.10): generalises. Per study 0.93 to 1.20; rat-only tokens 1.18. It tests the masked-token task on single cells of a new study, not the pseudo-cells of Stages 9–13 | `docs/audit/remediation/STEP4_RESULTS.md` V3 |
| **1** | **Corpus findings measured, not repaired** (no retrain, PI decision 1): 10.4% of the 6,803,206 training cells are above 15% mitochondrial reads; 1.04% of counts in ID-keyed files sit under another gene's token (43 of 53 studies above the 0.5% bound); 79 confirmed single-cell series never entered the corpus | `docs/audit/remediation/STEP5_RESULTS.md` |
| **2a** | **Exercise gate** (refs5): **194 blocks, 150 analysable, 11 hotspots and 1 borderline block.** Membership is not stable to small changes of the inputs (7 of the earlier 9 remain). S2 (8-week animals against controls): 9 of 12 called blocks significant. S3: 9 hotspots supported, 2 neither (both LUNG). S4: all five muscle blocks are below 1% of their tissue. No called block passes all nine registered readings | `STEP3_RESULTS.md` B4, B8 |
| **2a** | **Per-cell-type DE** on deconvolved Z (limma-trend; per-sex omnibus F combined over the sexes present; IHW with a nominal tissue covariate), refs5: **52,961 dose-significant gene × block pairs** (9,617 distinct genes; 11,560 pairs in the 44 blocks with mean fraction < 0.01). Single-sex tissues now have a dose test (TESTES 219, OVARY 8). **No block's calls are robust to technical covariates** by the registered verdict (0 of 150; 44% and 33% of pairs retained). **Positive controls: the registered red flag stands** (5 of the 26 testable pipeline-sanity controls; Tier A 6/45, Ai 2/14, B 6/22); the later data-anchored check is outside the registered ladder and does not lift it | `STEP3_RESULTS.md` B5; `STEP4_RESULTS.md` V2 |
| **3a** | Cross-species **transfer** of the rat exercise response into human embedding space (refs5): **10 of 11 hotspots meet the registered rule** (human AUC ≥ 0.65 and raw permutation p < 0.05; HEART vascular smooth muscle cells weakened); **9 of 11 the corrected rule** (also called in human space under Benjamini-Hochberg; not SKMGN Schwann cells or HEART vascular smooth muscle cells); LUNG vascular smooth muscle cells lose their dose trend. The human test reuses the rat test's cells, labels, folds and seeds: it is not a replication, and no human measurement enters. Rat-to-human Spearman of the exercise AUC 0.73 | `STEP3_RESULTS.md` B6 |
| **3a** | **E.3 atlas anchoring — PASS on the default before refs5; not repeated on refs5** (pre-registered): transferred rat cells sit nearest their Tabula Sapiens counterparts. 16 tissues: macro accuracy 0.430 over 120 types in 13 tissues, against null 0.086, p = 0.0001. refs4, 14 tissues: 0.404 against 0.084. It failed for 3 of that run's hotspots (SKMGN tenocytes, LUNG vascular smooth muscle cells, BLOOD erythroid cells) | `translation/ATLAS_PROJECTION_16_RESULTS.md`; `ATLAS_PROJECTION_RESULTS.md` |
| **2b** | **Rat in-silico perturbation engine (Module A)**, v2 on refs5 (2026-11-15): criteria 1, 3 and 4 **pass** (Mef2c at the 100th percentile of 26 matched genes; Alb and Snap25 in no cell; shift against expression 0.77). **The registered myeloid replication fails**: Spi1 at the 81st percentile of 232 matched genes (bar 90th); Cebpa not testable. Criterion 2 is withdrawn as a test of predicted targets (0 of 25, printed for the record). Module A does not establish that the model's perturbation responses identify a factor's targets. April's "all 4 PASS" is April only | `STEP4_RESULTS.md` V4; `RETRAIN_V2_RESULTS.md` (Module A) |
| **2b** | Model-driven **differential network** of TF deletions (dose-pooled, bootstrapped; difference on genes present in both arms), refs5: 7 networks, 1,540,997 edges. **Decoy null: no TF-specific signal** — pooled false-discovery estimate at `\|z\| ≥ 2` of 1.02 against expression-matched random genes; 0 of 777 TF × network pairs (281 distinct TFs) beat their decoys on hub strength. Hub tables are descriptive model readouts; nothing in them is read as regulation. Earlier arms gave the same verdict (0 of 723; 0 of 981) | `STEP3_RESULTS.md` B7; `GRN_DECOY_NULL_16_RESULTS.md`; `GRN_DECOY_NULL_RESULTS.md` |
| **3b** | **Perturbation on the transferred human cells** — human-space network + a rat↔human **self-consistency** score (refs5: median **0.48** over 685 TF × network scores, 0.55 over the per-network medians; per network 0.29 to 0.64). NOT cross-species biological conservation (same cells through the same model), and after the decoy null, self-consistency of an expression-driven quantity. Earlier arms: 0.310 (the default before refs5, under the old edge rule), 0.292 (July), 0.451 (April) | `STEP3_RESULTS.md` B7; `REFS4_RERUN_RESULTS.md` |
| **hardening** | **Purity sweeps** on refs5: the 11 configured dominant-type sweeps pass (0.993 or more at every purity of 0.5 and above); by the registered rule 12 of 15 dominant types pass, BLOOD erythroid cells and SPLEEN red-pulp macrophages fail, TESTES Sertoli cells are not testable. **Called types: 4 of 12 validated at their abundance.** **Composition** on all blocks: 25 flagged, 66 pass, 59 quiet, 44 not analysed; hotspots SKMVL vascular smooth muscle cells and SKMGN Schwann cells flagged. **Technical covariates:** the registered check refits the headline model and finds no block robust; Stage 14's older check (a pooled-sex dose slope refitted with RIN and globin) reads 10 of 11 hotspots as robust and tests a different model | `STEP3_RESULTS.md` B5, B8; `STEP4_RESULTS.md` V1 |
| **hardening** | DE near-empty-block gate: the §6 rule (median per-sample library ≥ 1,000), with sensitivities at 100 and 0; the hotspot set is the same at 1,000 and 100 | `REFERENCE_RELABEL_PREREG.md` §6; `REFS4_RERUN_RESULTS.md` |
| **hardening** | **References.** Joint re-annotation (refs4) + BLOOD whole-blood composite; for refs5 all 15 references rebuilt from committed code into `references_v5/` (13 byte-identical; HYPOTH re-annotated, with no GABAergic type passing the registered bar; HEART merge applied). Reference integrity (QC gate + manifest + from-scratch driver). The label decisions were made by an AI assistant; no specialist has read them. **Mixture panel on refs5:** floor met on 19 of 19 sets. **mRNA-content bias on refs5:** BayesPrism's error against cell fractions rises under native bias on 13 of 15 references | `STEP3_RESULTS.md` A1, A2; `STEP4_RESULTS.md` V5, V6; `REFERENCE_RELABEL_PREREG.md`; `REFERENCE_QC.md`; `OMNIDECONV_RESULTS.md` |

---

## Aim 1 validation results

The three checks promised in `reports/pipeline_report.md §3.6`, run on v2 (2026-10-24; code
`analysis/aim1_validation/`, report `reports/aim1_validation_v2/AIM1_VALIDATION_REPORT.md`). April's run
(2026-06-25/26, `reports/aim1_validation/`) in brackets.

- ✅ **Homolog embedding similarity — PASS.** Contextual gene embeddings place rat genes far closer to their
  true human/mouse ortholog than to a permuted one: ROC-AUC **1.000** (T1) [1.000], **0.948** (T3a vs human)
  [0.973], **0.851** (T3b vs mouse) [0.908]. v2's T3 genes sit less close to their orthologs than April's,
  as their drift from init in C4b shows.
- ◐ **Cell-type clustering — PARTIAL.** The cells are sampled from the training corpus, not held out (no cell was
  held out of training, by a recorded decision), and their labels are automated marker labels. kNN cell-type purity **0.849** (chance 0.065) [0.813], but
  the cell embeddings are **not batch-corrected**: kNN study purity 0.966 [0.975], cross-study mixing within a
  cell type 3.6% [2.0%]. Each model draws its own labelled subset (they share 1,935 cells), so this is a rerun,
  not a paired comparison; C3 compares the two on the same reference cells (v2 better in 12 of 12). Largely
  insulated from Aims 2–3, which use pseudo-cells *within* tissue.
- ◐ **T4 new-token quality — WEAK.** Random-init rat-specific tokens did not acquire cross-species family
  structure at the input layer (CYP rank-AUC 0.569 [0.601]; pooled 0.471 [0.527], ≈ chance). Only ~4 clean
  CYP genes; tests input embeddings only.

A fourth check was added on 2026-11-15 and is the only one on data the model never saw
(`docs/audit/remediation/STEP4_RESULTS.md`, V3):

- ✅ **Held-out studies — GENERALISES.** On 66,000 cells of 33 rat studies outside the corpus, v2's
  gene-identity loss is 1.046 times its in-sample probe loss (registered bar 1.10; April 1.017 on the same
  cells). Five studies are between 1.10 and 1.25 (testis 1.20). The gap grows as tokens lose their
  cross-species anchor: 1.04 for three-species orthologs, 1.18 for rat-only tokens. The held-out cells have
  no labels, so the clustering check above was not repeated on them.

---

## Open frontier

### Aim 2b/2c + 3b — model-driven science (largest unbuilt block; pivots on the keystone)

| Status | Item | Aim | Effort | Blocker |
|---|---|---|---|---|
| ✅ | **Module A — rat in-silico perturbation engine (KEYSTONE)** — built. **Current reading (v2 on refs5, 2026-11-15; `docs/audit/remediation/STEP4_RESULTS.md` V4): criteria 1, 3 and 4 pass (Mef2c 100th percentile of 26 matched genes; negatives in 0 cells; ρ 0.77); the registered myeloid replication fails (Spi1 81st percentile, bar 90th; Cebpa not testable); criterion 2 is withdrawn (0 of 25).** The engine runs, behaves sensibly on not-expressed genes, and ranks one muscle factor above its matched genes; it does not establish that the model's perturbation responses identify a factor's targets. History: April's pre-registration closed with all 4 criteria passing for the April model only (Mef2c 98th pctile; ρ=0.70; target enrichment p=2e-7); on the July-reference SKMVL cells April 4/4 and v2 3/4, criterion 2 failing (0/25 vs April 4/25) with the curated set's provenance caveat (`RETRAIN_V2_RESULTS.md`). Use the per-gene **target** shift, not the near-noise cell shift. | 2b/3b | — | — |
| ✅ | Module A.5 — rat TF list (AnimalTFDB → ENSRNOG ∩ expressed) — DONE (shipped with Module A). | 2b/3b | — | — |
| ✅ | Module B — model-driven differential GRN — built. **Current reading (refs5, 2026-11-15; `STEP3_RESULTS.md` B7): 7 networks, 1,540,997 edges, the trained-minus-control difference taken on genes present in both arms; decoy null negative (pooled estimate 1.02 at \|z\| ≥ 2; 0 of 777 TF × network pairs beat their decoys). The networks carry no TF-specific signal and no hub is read as a regulator.** History follows. **REBUILT UNCAPPED 2026-07-13** (`grn_uncapped/`: 3,478,564 edges, 398,586 of them with \|z\| ≥ 2, 389 TFs; `delta` bit-identical to the old capped tables → strict superset). **Those counts are the April model's on the pre-July references and describe no later arm** (audit finding 9.4; a later arm's count is in its own `hub_strength.tsv`). Until 2026-11-14 this row called the \|z\| ≥ 2 edges "reproducible"; the decoy null at the end of this row does not support that word. Top hubs = Tfeb/**Mef2c**/Epas1/Myf6/**Nr4a1** — but **only under the corrected statistic `sum\|Δ\|`**; ranking by *count* of `\|z\|≥2` edges inverts the biology (anchors 33rd pctile, Nr4a1 17th) because `z` diverges where the model is numerically inert. See [[project_grn_hub_statistic]]. **v2 (2026-10-25):** 9 hotspot networks (`grn_v2/`); hub strength mostly follows each TF's own expression in the profile (Spearman ~0.8, measured for April and for v2 on the July references and not recomputed on a later arm), so hub rankings are largely input-driven; the two models agree on hub order but not on individual edges. **refs4 decoy null (2026-10-31): FDR(2) = 1.03 and 0/981 TFs beat expression-matched decoys, so the hubs (Mef2c, Nr4a1, Epas1, Tfeb included) are not TF-specific; the GRN is reported as a descriptive model readout only** (`GRN_DECOY_NULL_RESULTS.md`). | 2b | — | — |
| ☐ | Module C — CPA dose model (ordinal week; compare to DE slope) | 2c | L (GPU) | none (n≈50/cell-type thin) |
| ✅ | Aim 3b — perturbation on the transferred human cells — DONE, rebuilt uncapped. **⚠️ RE-SCOPED 2026-07-13: this is NOT cross-species biological conservation.** The transfer re-expresses the *same* pseudo-cells (identical 700 cells, same order) and **91.1% of orthologous rat genes carry the identical GeneCompass token as their human ortholog** — so the human-space perturbation deletes the same token from a near-identical sequence in the same model. No human measurement enters. It is a **model self-consistency** score (refs5: median 0.48 over 685 TF × network scores, under the repaired edge rule; April median ρ 0.451, unbiased; the published 0.507 was selection-inflated; **v2 0.292 on the July references; 0.310 on refs4** — lower in 566 of 729 TF scores both models make; the drop cannot be split between v2's weights and its tokenization). **The capstone therefore has TWO independent evidence legs, not three.** | 3b | — | — |
| ✅ | Stage orchestrators — realized as **Stage 13** (mechanism: perturb→GRN→conservation→enrichment) and **Stage 14** (hardening). *Correction 2026-11-14 (audit finding 6.10): `pipeline/run_stage11.py` exists and is the Aim-2 gate; it runs before Stage 10. This row said there was no `run_stage11.py`. The build plan's "Stage 11" chain (perturb → GRN) is what became Stage 13.* | — | — | — |

### Aim 3c — human genetics / disease translation (highest-value *startable now*)

| Status | Item | Effort | Blocker |
|---|---|---|---|
| ☐ | **Module D.0 — stage genetics data** (GTEx v8 eQTL, ~114-GWAS set, S-PrediXcan MASHR/JTI, Open Targets). No GPU, no DE dependency → can start immediately. | M | none |
| ☐ | Module D.1 — `validate_human.py`: replicate Vetr 2024 bulk pipeline (control), then swap in our per-cell-type DE → **cell-type-resolved trait–tissue–gene triplets** (the novel deliverable). | L | D.0 |
| ☐ | Module F — conserved regulators (join perturbation/GRN + genetics). Aim-3 capstone. **2 of its 3 input streams are now on disk** (differential GRN + human-space conservation); it awaits only the genetics. After the decoy null, the GRN stream contributes no TF-specific evidence, so the capstone rests on genetics + DE/transfer. | S | **D** |

### Validation / hardening (completeness, not new aims)

| Status | Item | Effort |
|---|---|---|
| ✅ | **omnideconv** multi-method θ cross-check. **On refs5 (2026-11-15; `docs/audit/remediation/STEP4_RESULTS.md` V5, V6):** mixture panel, 19 sets × 6 methods, BayesPrism's median per-type r 0.976–0.999, floor met on 19 of 19, 6 of 192 type tests below 0.85, KIDNEY cross-study pooled r 0.17; SimBu with a bias-free baseline, BayesPrism's error against cell fractions rises under native mRNA differences on 13 of 15 references (mean +0.027) and under added bias on 14 of 15 (+0.053); the corrector rule is reproduced on C minus A and not on C minus B. Earlier runs: 11-tissue panel × 6 methods + SimBu mRNA-bias battery + dose-response; agreement with the omnideconv paper + downstream-claims bias guidance (`OMNIDECONV_RESULTS.md`); **the battery was rerun on the current references 2026-11-11** (`SIMBU_BIAS_REFS4ALL_RESULTS.md`: the corrector ranking is not reproduced, BayesPrism degrades on 15 of 15). **Panel rerun on the current references 2026-11-08** (`MIXTURE_PANEL_REFS4ALL_RESULTS.md`): 15 holdout + 4 cross sets × 6 methods (CIBERSORTx included); BayesPrism median per-type r 0.973–0.999 on all 19, best on 17; KIDNEY_cross pooled 0.163 (monocyte sink); 170 of 188 reference types benchmarked (TESTES Sertoli cells not among them) | — |
| ✅ | **Reference integrity** — DONE (2026-07): `reference_qc.py` gate (wired into `build_reference`), `tissue_references.yaml`, `run_deconv_all.py`; liver Visium contamination + engineered→native lung fixed (`REFERENCE_QC.md`) | — |
| ✅ | **Lung** — DONE: engineered GSE178405 → native pooled reference. refs5 calls two LUNG hotspots (endothelial cells, vascular smooth muscle cells). Both rest on the early time points and on the model: at 8 weeks against control they do not separate (AUC 0.52, 0.61), expression alone separates them at 0.60 and 0.59, and the vascular smooth muscle cells lose their dose trend in human space. The LUNG list has changed with every arm (the default before refs5: vascular smooth muscle cells; refs4 on 14 tissues: endothelial, mesothelial and vascular smooth muscle cells; July: endothelial cells + macrophages). The reference's cell quality cannot be measured: its source matrices hold no mitochondrial counts | — |
| ◐ | Per-tissue expression **purity sweeps**. **On refs5 (2026-11-15; `STEP4_RESULTS.md` V1), by the registered pass rule (median correlation at least 0.95 at every purity of 0.5 and above): dominant types, 12 of 15 pass, BLOOD erythroid cells (0.83 at 0.5; never above 0.93) and SPLEEN red-pulp macrophages fail, TESTES Sertoli cells are not testable. Called cell types, also read at their own abundance: 4 of 12 validated (BLOOD classical monocytes, BLOOD megakaryocytes and platelets, LUNG endothelial cells, LUNG vascular smooth muscle cells); none of the five muscle blocks (0.50 to 0.57 at 0.5% of a mixture), including the top hotspot, SKMVL satellite cells. A sweep mixes cells of the reference's own study, so it is the method's best case.** Earlier record (dominant types only, no registered pass rule): Stage 14; model-independent; 13 sweeps on disk across 12 tissues. We **meet the Chu-2022 bar** (>0.95 above ~50% focal purity) on all **12 tissues swept** (of the 14-tissue panel): ≥0.985 at 50% purity, ≥0.99 at 85%. The 4 tissues added in the 2026-07-16 rebuild (BAT/HYPOTH/SMLINT/TESTES) were **swept 2026-07-27 and all pass** (focal VST ≥0.992 at 50% purity, HYPOTH lowest; an earlier "≥0.998" / "HYPOTH 1.000 throughout" here was wrong). **Current references (2026-11-02, clean rebuild):** these July/October sweeps predate the refs4 relabel and the TESTES lift for 8 of 11 sweeps; the rebuild re-swept all 11 on the production references and all meet the bar (≥0.992 at 50%, ≥0.998 at 85%; `deconvolution/REFS4ALL_REBUILD_RESULTS.md`, data in `validation_refs4all_rebuild/`). Only BLOOD and subcutaneous fat remain unswept — neither has a capturable dominant parenchyma. | — |
| ◐ | Technical covariates. **Registered check on refs5 (`STEP3_RESULTS.md` B5): the whole DE refitted with RIN and globin percentage (A) and with those plus duplication rate and 5'-3' bias (B). 23,408 (44%) and 17,321 (33%) of the 52,961 headline pairs are retained; 0 of 150 blocks and 0 of the 12 called blocks are ROBUST. The gene ranking is steadier than the calls (median correlation 0.92 under A, 0.80 under B). The four covariates do not differ between training groups beyond chance (a descriptive check, not registered), so the loss is not explained by the covariates standing in for the groups; which reading is right was not established.** The older check below refits a pooled-sex dose slope, a different model from the one that makes the headline calls, and again reads 10 of 11 refs5 hotspots as robust. Composition on refs5: 25 flagged, 66 pass, 59 quiet, 44 not analysed; two hotspots flagged (SKMVL vascular smooth muscle cells, p 0.006; SKMGN Schwann cells, p 0.033). Earlier record (`rin_globin_robustness.tsv`): 16 tissues **9 of 9 hotspots ROBUST** (median Spearman of the dose statistic with vs without the covariates 0.997, median retention 0.98); refs4 (14 tissues) 11 of 11 (0.957, 0.91); July 11 of 13. Composition (θ) confound on all 197 blocks: 112 QUIET / 62 PASS_EXPRESSION / 23 FLAG_COMPOSITION; one hotspot flagged (SKMVL vascular smooth muscle cells). refs4: 91 / 56 / 18, with LUNG mesothelial cells also flagged. | — |
| ✅ | **E.3** Tabula Sapiens atlas anchoring — **PASS** 2026-10-31 (0.404 vs null 0.084; `translation/ATLAS_PROJECTION_RESULTS.md`); on the 16-tissue default of that date 2026-11-04, 0.430 vs 0.086 (`translation/ATLAS_PROJECTION_16_RESULTS.md`). **Not repeated on refs5**: the per-type results are for the earlier default's cells, and three of refs5's hotspots (SKMVL tenocytes, SKMGN Schwann cells, HEART vascular smooth muscle cells) were not hotspots then. | — |
| ✅ | **Second heart reference** (`REFERENCE_RELABEL_PREREG.md` §15): GSE307551 F344 LV (independent study), 20 coarse mixtures deconvolved with the production HEART reference — **PASS** 2026-11-01: CM r 0.944, mean absolute CM error 0.043 (bar: r ≥ 0.9, error ≤ 0.10). Every class correlates (r 0.943–0.999) but absolute levels are off (endothelial +0.151, lymphocytes −0.103), so HEART stays read differentially (`data/deconvolution/validation/HEART_GSE307551_cross/`). Run on the HEART reference before its refs5 label merge ("CD8+ T cells" into "T cells"); not repeated. | — |
| ✅ | **More MoTrPAC tissues** (§14). Of the five tissues without a reference: **SPLEEN** (GSE306764, whole Lewis spleen, 14 types) and **OVARY** (GSE343214, one rat, 18 types) pass every registered check (S1; O1/O2; R1; R2 within-library r ≥ 0.94) and are built, deconvolved and embedded (2026-11-01). The SPLEEN fractions are implausible (T cells about 1%, endothelium 15% from a 34-cell type), so SPLEEN is read differentially only. COLON rejected (GSE326856 is sorted cells, no epithelium); ADRNL and VENACV have no usable public rat single-cell reference. The registered all-tissue (16) rerun was done on 2026-11-01 (c3b27c7; `ALL_TISSUE_RERUN_RESULTS.md`): 197 blocks / 151 analysable, 9 hotspots, no SPLEEN or OVARY hotspot. On refs5 there is still no SPLEEN or OVARY hotspot; SPLEEN has 7,252 dose-significant pairs. **The single-sex DE defect is repaired in refs5** under its registration (`STEP3_PREREG.md` 6.1): a single-sex block's dose p is its one sex's p, giving TESTES 219 pairs and OVARY 8. OVARY and TESTES have no Augur score (5 controls per arm). | — |
| ✅ | **A null for the GRN's `|z|`** — done 2026-10-31 by expression-matched decoy genes, since a label permutation is impossible here (`GRN_NULL_WHY_ABANDONED.md`). **Negative:** FDR(2) = 1.03, no t reaches FDR 0.10, 0/981 TFs beat their decoys (`GRN_DECOY_NULL_RESULTS.md`). Same on the 16-tissue default of 2026-11-04: FDR(2) = 1.03, 0/723 (`GRN_DECOY_NULL_16_RESULTS.md`). Same on refs5 under the repaired edge rule (2026-11-15): 1.02, 0 of 777 TF × network pairs (`docs/audit/remediation/STEP3_RESULTS.md` B7). | — |
| ✅ | **Degenerate-block gate in the DE** — closed by the §6 rule (median per-sample library ≥ 1,000), applied to every refs4 and all-tissue table, with sensitivities at 100 and 0. The near-empty KIDNEY intercalated blocks are skipped. | — |
| ☐ | Module G — viewer v2 per-cell-type local PCA (lowest value, defer) | M |

### Blocked / out of scope

- **CIBERSORTx** — no longer blocked: it ran as one of the six methods in the mixture panels and the SimBu batteries (2026-11-08, 2026-11-11 and the refs5 runs, `docs/audit/remediation/STEP4_RESULTS.md` V5 and V6). It stays license-gated per user; setup is in `deconvolution/README.md`.
- ⊘ Cross-tissue cell-type comparison (different reference per tissue) · ⊘ GEARS / drug-CPA / DeepCE (no rat perturb-seq training data).

---

## Aim 1 — methods depth still open

| Status | Item | Effort |
|---|---|---|
| ✅ | Per-tier masked-prediction loss (2026-10-31; see Done) | — |
| ✅ | **End-to-end Stages 1–7 reproduction** (2026-11-01; `docs/repro_stages1_7/REPRODUCTION.md`). Stages 1, 3, 4, 5, the training mix and the eval probe rebuild byte for byte; the Stage 6 GRN is identical; the init differs only in its 3 unseeded random tensors (now `--seed`); a 2,000-step training restart tracks v2's loss within fp16 nondeterminism (39/40 logged points equal to 4 decimals). **Stage 2 does not rebuild byte for byte from raw GEO**: the RGD synonym files were rewritten on 2026-02-24, after the canonical run, so 187 of the kept QC files lose 1–27 genes. The February code gives the same result as HEAD, so the code is ruled out; reference md5s are now pinned and checked. The 27 `keep_raw_slot` files **do** regenerate, with identical counts, through `scripts/restore_rawslot_studies.sh`, and Stage 2 numbering is pinned (`config/stage2_sample_order.tsv`). No effect on v2, whose canonical inputs are on disk and verified. Reproducibility fixes for Stages 0–14: c6be8aa, d38b83a, ec5895b. Deposit list: REPRODUCTION.md, follow-up section. **Stages 8–14 (2026-11-02):** a clean rebuild of the 16-tissue default into fresh folders reproduces it byte for byte through Stage 13 (θ, Z, embeddings, gate, DE, transfer; GRN edges and values identical, tie order now deterministic); `deconvolution/REFS4ALL_REBUILD_RESULTS.md`. From 2026-11-03 to 2026-11-15 that clean rebuild was the default tree (`_refs4all_rebuild`, now `config/profiles/refs4all_rebuild.yaml`; the assembled tree = `config/profiles/refs_v4all.yaml`). **refs5 (2026-11-15):** every stage ran from one clean commit (`44332e6`); Stage 8 was run twice and repeated byte for byte (fractions, Z, the fitted object, pseudo-cells); all 15 references rebuilt from committed code, 13 byte-identical to the builds in use. Stages 9–14 of refs5 were run once (`docs/audit/remediation/STEP3_RESULTS.md` A1, B1). | — |
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

Both moves take the per-cell-type DE as their input. On refs5 no block's DE calls are robust to technical
covariates by the registered verdict, and the positive-control red flag stands
(`docs/audit/remediation/STEP3_RESULTS.md` B5; `STEP4_RESULTS.md` V2). Neither is resolved;
`docs/audit/remediation/LABEL_REVIEW_PLAN.md` puts the DE model before a statistician.

Off the critical path: **Module C** (CPA dose model, confirmatory, thin-N); the Aim-1 ablations (GPU-weeks).

**Detailed handoff specs** live in `manuscript/wip/` (local-only; see its `README.md`).
