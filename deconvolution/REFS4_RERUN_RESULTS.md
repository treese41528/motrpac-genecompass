# Rerun on the jointly re-annotated references (refs4): results (2026-10-29)

**Question.** Do the Aim-2 and Aim-3 results hold once the deconvolution references carry labels assigned by
one joint procedure (and BLOOD has a whole-blood reference) instead of the July labels? This is the rerun
registered in `REFERENCE_RELABEL_PREREG.md` §8, reported against the July-reference v2 results.

**What changed between the arms.**
- Ten tissues were re-deconvolved: the eight relabelled references, BLOOD with the option-B composite, and
  LUNG/CORTEX/HIPPOC after the literature check (`JOINT_REANNOTATION_LITCHECK.md`).
- BAT, HEART, HYPOTH and TESTES are unchanged and carried over. Their Stage 9 embedding QC reproduces v2
  exactly.
- The §6 rule applies to every refs4 table: a block is analysed only if its median per-sample library is
  at least 1,000. Sensitivities at 100 and at 0 (no rule) are reported. To separate the effect of the rule
  from the effect of the references, a third arm, "July + §6", re-scores the July gate under the rule.

Profile: `config/profiles/refs_v4.yaml`. Full comparison printout: `data/results/refs4/compare_refs4.txt`
(from `compare_refs4.py`, gitignored).

## Headline

| | July v2 | July + §6 | refs4 (§6) | refs4, ≥100 | refs4, no rule |
|---|---|---|---|---|---|
| Blocks | 172 | 172 | 165 | 165 | 165 |
| Analysable | 172 | 107 | 124 | 131 | 165 |
| Hotspots (q < 0.05) | 13 | 19 | **11** | 11 | 9 |
| Hotspots preserved in human transfer | 13/13 | — | **11/11** | — | — |
| Hotspots with Augur AUC ≥ 0.70 | — | — | 11/11 | 11/11 | 9/9 |

- **The rule's effect.** On the July references the §6 rule raises the hotspot count from 13 to 19. Removing
  near-empty blocks shrinks the BH family, so marginal blocks pass.
- **The references' effect.** With the same rule, refs4 gives 11 hotspots against 19 for July. The count
  falls because of the references, not the rule.
- **Stability.** The refs4 hotspot set does not change between the §6 threshold and 100. With no rule it
  loses LUNG endothelial and LUNG mesothelial cells, both of which pass the rule easily.

## Hotspots

| Tissue | Cell type | July v2 | refs4 | refs4 AUC / q |
|---|---|---|---|---|
| BLOOD | Classical monocytes | ✓ | ✓ | 0.821 / 0.025 |
| HYPOTH | Oligodendrocyte precursor cells | ✓ | ✓ | 0.798 / 0.030 |
| LUNG | Endothelial cells | ✓ | ✓ | 0.786 / 0.044 |
| BLOOD | Erythroid cells | — (no type) | ✓ | 0.828 / 0.025 |
| BLOOD | Megakaryocytes and platelets | Megakaryocytes ✓ | ✓ | 0.837 / 0.025 |
| BLOOD | Neutrophils | — (no type) | ✓ | 0.819 / 0.025 |
| LUNG | Mesothelial cells | | ✓ | 0.785 / 0.044 |
| LUNG | Vascular smooth muscle cells | | ✓ | 0.817 / 0.028 |
| SKMGN | Tenocytes | — (no type) | ✓ | 0.825 / 0.025 |
| SKMVL | Satellite cells | — (no type) | ✓ | 0.887 / 0.012 |
| SKMVL | Vascular smooth muscle cells | | ✓ | 0.874 / 0.012 |

July hotspots not found with refs4:
- **BLOOD lymphocyte and basophil hotspots** (Naive B, NK, ISG-expressing T, Basophils). Their types no
  longer exist, or their blocks are now near-empty (see BLOOD below).
- **BLOOD Non-classical monocytes:** q 0.060.
- **KIDNEY Renal interstitial fibroblasts:** median library 15 in July, so it fails §6 in both arms.
- **LUNG Macrophages:** the refs4 alveolar and interstitial macrophage blocks are near-empty (median
  library 20 and 14), so they fail §6.
- **SKMGN Endothelial cells:** q 0.089.
- **SKMGN Skeletal myocytes:** the muscle block, renamed Skeletal muscle cells, narrowly misses (AUC 0.772,
  q 0.061; SKMVL 0.742, q 0.089).

Of the July hotspots, those that were July-reference artefacts (BLOOD ISG-expressing T cells at 55% of whole
blood) or near-empty blocks (KIDNEY fibroblasts) were expected to go. The losses in muscle (a near-miss) and
LUNG macrophages (the refs4 LUNG reference puts almost no mass on macrophages) are real changes in the result.

## Transfer to human (Stage 12)

Every refs4 hotspot survives transfer (11/11 PRESERVED; ΔAUC −0.106 to +0.100). Across all blocks the
rat-to-human Spearman of the exercise AUC rises from 0.616 to 0.751.

## Positive controls (Stage 10, frozen spec)

The refs4 run uses `posctrl_prereg_refs4.tsv`, which renames targets only.

| Tier | July recovered | refs4 recovered | Changed outcomes |
|---|---|---|---|
| A | 4 | 5 | BLOOD Klf3 not sig → recovered; SKMVL Plin4 not sig → **wrong direction** |
| Ai | 2 | 2 | — |
| B | 6 + 1 confounded | 7 | BLOOD Crat → recovered; BLOOD Aamp recovered → not sig; WATSC Endou confounded → recovered |

5 of 81 gene-level outcomes changed. Net: one more recovery in Tier A and in Tier B, and one more
wrong-direction control.

**Tier C, BLOOD (program enrichment).**

| Program | July | refs4 |
|---|---|---|
| Cytotoxicity | recovered in NK (p 0.003), Memory T (p 0.0003), ISG-T (p 0.03) | recovered in **NK cells** (5/9, 2.5×, p 0.028) |
| Naive (Ccr7, Tcf7, Lef1, Sell) | recovered in Memory T (p 0.013) | **not testable** |
| ISG (Isg20) | not recovered | not testable |
| Monocyte activation | not recovered (p 0.64) | not recovered (p 0.71) |

Every BLOOD T- and B-cell block (CD4+, CD8+, T, Treg, B) has a median library of 3 to 55 and is skipped under
§6, so those programs cannot be tested. The July reference had blocks of the same kind (5 to 15), which ran
unflagged and gave 3 to 23 dose-significant genes. What disappears are July's mid-sized lymphocyte blocks
(Memory T 126,000, Naive B 55,000, NK 102,000). Those blocks came from a PBMC-only reference that is known to
be misspecified for whole blood. The refs4 lymphocyte total (under 1%, against 7% neutrophils) is also
implausible for rat blood, where lymphocytes outnumber neutrophils. Neither reference measures rat blood
lymphocytes reliably.

## DE, enrichment and Stage 14

| | July v2 | refs4 §6 | refs4 ≥100 | refs4, no rule |
|---|---|---|---|---|
| DE blocks run / skipped | 172 / 0 | 124 / 41 | 131 / 34 | 165 / 0 |
| Dose-significant genes (FDR_IHW < 0.05) | 28,999 | 48,522 | 47,526 | 44,112 |
| Enrichment: significant pathways (blocks) | 4,724 (98) | 5,906 (106) | | |

- **Where the dose-significant genes are.** BLOOD accounts for 20,029 of the refs4 genes, now spread over
  erythroid, platelet and neutrophil blocks. HIPPOC and TESTES give 0, HYPOTH 1 and CORTEX 10.
- **Composition confound (all blocks).** July: QUIET 114, PASS_EXPRESSION 46, FLAG_COMPOSITION 12. refs4:
  QUIET 91, PASS_EXPRESSION 56, FLAG 18. Two refs4 hotspots are flagged: LUNG Mesothelial cells and SKMVL
  Vascular smooth muscle cells. Their exercise signal may partly reflect composition shifts rather than
  expression. July flagged two BLOOD hotspots (NK, Naive B).
- **RIN/globin robustness (hotspots).** July: ROBUST 11, CHECK 2. refs4: **ROBUST 11/11** (median Spearman of
  the dose statistic with vs without RIN + % globin 0.957, median retention 0.91). The first two refs4 passes
  read the July Z matrices and were discarded (see `REFERENCE_RELABEL_PREREG.md`, 2026-10-29).
- **Parenchyma bulk-mover recovery.**

  | Tissue | July | refs4 |
  |---|---|---|
  | SKMGN | 0.380 | 0.394 |
  | SKMVL | 0.529 | 0.582 |
  | HEART | 0.917 | 0.917 (unchanged reference) |

  The dominant-parenchyma DE problem is unchanged by relabelling.

## GRNs and conservation (Stage 13)

Networks are built for hotspots with mean fraction ≥ 0.01.
- **July:** 9 networks, median rat–human Spearman 0.292 over 944 TFs (48.7% ≥ 0.3).
- **refs4:** 8 networks (4 BLOOD, 3 LUNG, SKMGN tenocytes), median 0.310 over 859 TFs (53.1% ≥ 0.3).

As before, this measures how consistent the model is with itself across species, not cross-species biology.
The rat and human networks are built from the same cells. Anchors: Cebpa and Spi1 in BLOOD neutrophils are
positive (ρ ≈ 0.17, p ≈ 0.02); Spi1 in classical monocytes is 0.11 (n.s.). No myogenic anchor applies:
neither muscle hotspot is a myocyte block.

## Composition checks: implausible fractions (reported, not adjusted)

| Tissue | Observation | Reading |
|---|---|---|
| **HIPPOC** | Endothelial cells 0.498, Inhibitory neurons 0.358, Excitatory neurons **0.003**, DG granule 0.028 | Diagnosed in `HIPPOC_REFS4_DIAGNOSIS.md`. The cause is ambient RNA in the nuclei reference: the soup matches whole-tissue bulk better than any nucleus profile, and the endothelial cluster (about 55% ambient) absorbs it. July's 74% excitatory was the same artefact (91% of those barcodes were ambient). Ambient correction plus merged vascular types brings neurons to about two thirds in an NNLS proxy, but the neuron subtype split stays unidentifiable. No hotspots or dose-significant genes; mark HIPPOC composition unreliable. |
| **KIDNEY** | "Monocytes" 0.161 (j22, non-classical: Adgre4, Treml4, Spn; 101 reference cells) | Far too high for kidney. Either a sink profile or blood carried with the tissue. Not a hotspot. |
| **BLOOD** | Erythroid 0.70, platelets 0.15, neutrophils 0.07, lymphocytes < 0.01 | Erythroid and platelet dominance is expected for whole-blood RNA. The lymphocyte total is too low (above). |
| **WATSC** | Plasmacytoid dendritic cells 0.048 | High for adipose. Not a hotspot. |

## Embedding QC (Stage 9, cell-type silhouette)

Silhouette rises in HIPPOC (0.573 → 0.738), KIDNEY (0.661 → 0.768), LIVER (0.768 → 0.852), SMLINT
(0.660 → 0.852), BLOOD (0.490 → 0.553) and CORTEX (0.901 → 0.912). LUNG falls (0.600 → 0.520), and muscle
and WATSC are within 0.01. LUNG has 1,032 pseudocells, not 1,050. 18 of the 50 Goblet-cell pseudocells
(median 207 expressed genes; a near-empty block) fall below the tokenizer's 200-gene minimum.

## Fixes made during the rerun (none changes a frozen threshold)

Logged in `REFERENCE_RELABEL_PREREG.md` (2026-10-28, 2026-10-29).
- `compare_posctrl.py`: a target whose block exists but was skipped under §6 now scores `not_testable` with
  a warning. Before, it was treated as a stale label.
- `run_stage12.py` creates the human gate folder.
- `de_technical_covariate_robustness.R` reads the Z matrices from the configured results folder. It
  hardcoded the July folder. The R wrappers also never exported the config values they `eval`, so no R
  script could see a profile's folders. `de_technical_covariate_robustness.sh` and
  `run_gene_set_enrichment.sh` now export them. Enrichment was unaffected, because Stage 13 passes `DE_DIR`.
- `validate_selection.py` skips the corpus checks for references built outside the corpus (BLOOD composite).

## Reading

- **What holds.** The exercise signal survives the change of references, but on a different set of blocks:
  - 3 of 13 July hotspots persist exactly, and a fourth (Megakaryocytes) as Megakaryocytes and platelets.
  - The new hotspots are mostly cell types the July references did not have (BLOOD erythroid, neutrophils,
    platelets; muscle satellite cells and tenocytes) or could not separate.
  - Transfer to human holds for every hotspot.
  - Every hotspot is robust to RIN and % globin.
  - Positive controls are essentially unchanged.
- **What the July results owed to their references.** The BLOOD lymphocyte hotspots and the muscle-myocyte
  hotspot. Claims about those cell types should not be carried forward from the July run.
- **What remains unreliable under either reference.** Rat blood lymphocytes, hippocampal composition, and
  DE in the dominant parenchyma.
