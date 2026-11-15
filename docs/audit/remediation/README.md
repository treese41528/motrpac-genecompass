# Remediation of the 2026-11-11 audit

What was done about each finding of `docs/audit/FINDINGS.md`, in the order agreed on 2026-11-14, with the
reason for every choice and the evidence behind it.

- **Start with** the status table below, then the step you need.
- **Who.** An AI coding assistant (Claude), under the PI's decisions. No domain expert has reviewed this work;
  `LABEL_REVIEW_PLAN.md` (written last) lists what one should read.
- **Rule for every number.** It comes from a table on disk, named beside it, and the script that made the table
  is in `analysis/audit_remediation/`. Every reference was looked up on Crossref
  (`record/crossref_remediation.tsv`) or is already in `docs/audit/REFERENCES.md`.

## The PI's decisions (2026-11-14)

| # | Question | Decision |
|---|---|---|
| 1 | Retrain the model for the corpus findings? | No. Measure and disclose |
| 2 | The gate's contrast | Keep trained (all durations) against control as primary; add the sensitivities |
| 3 | HIPPOC | Keep it flagged unreliable |
| 4 | Module A criterion 2 | Withdrawn for now |
| 5 | Expert review of labels | A plan for what to review, after everything else is done |

## What the remediation found, in one table

| Question | Answer | Where |
|---|---|---|
| Was anything silently wrong in the sample or weight handling? | No. Every sample is tied to its animal in 16 tissues; the model weights load completely | Step 1 |
| How many hotspots are there, by a rule that more permutations cannot reverse? | **11, and 1 borderline block**, of 150 analysable blocks. There were 9 | Step 3, B4 |
| Is the hotspot list stable? | **No.** 7 of the earlier 9 remain. In tissues whose reference did not change, a block's AUC moves by 0.03 at the median and by 0.10 or more in one block in ten | Step 3, B4 |
| Do the hotspots survive every registered reading? | **None passes all nine.** Two BLOOD types pass eight | Step 3, B8 |
| Are the DE calls robust to technical covariates? | **No, in every block.** 44% and 33% of the 52,961 gene x block pairs survive the two reruns | Step 3, B5 |
| Is per-cell-type expression recovered at the abundance of the called types? | In **4 of 12**. In none of the five muscle blocks, including the top hotspot | Step 4, V1 |
| Do the registered positive controls recover? | **No. The red flag stands**: 5 of 26 testable sanity controls | Step 4, V2 |
| Do the networks carry TF-specific signal? | **No.** The decoy null is negative under the repaired edge rule | Step 3, B7 |
| Does the model generalise to rat studies it has not seen? | **Yes** by the registered bar: loss ratio 1.046 against 1.10; weakest for rat-only tokens (1.18) | Step 4, V3 |
| Does Module A hold on the current model and references? | Criteria 1, 3 and 4 pass. The pre-named myeloid replication fails | Step 4, V4 |
| Are BayesPrism's fractions cell proportions? | Not where cell types differ in mRNA content: its error against cell fractions rises on 13 of 15 references under their own mRNA differences | Step 4, V5 |
| Does deconvolution recover composition on mixtures of the references? | Yes: the registered floor is met on 19 of 19 sets | Step 4, V6 |
| How clean is the training corpus? | 10.4% of training cells are above GeneCompass's 15% mitochondrial cut; about 1% of counts sit under another gene's token; six further Stage 2 behaviours were found | Step 5 |

## Order and status

| Step | What | Findings | Status | Document |
|---|---|---|---|---|
| 1 | Integrity checks: could anything be silently wrong? | 3.1, 6.7, 6.9, 7.2 (imbalance), 8.4 | **Done. All clean or explained; guards added** | [`STEP1_INTEGRITY.md`](STEP1_INTEGRITY.md) |
| 2 | Statements that are wrong whatever a rerun shows | 1.2, 1.4, 1.8, 1.9, 2.1, 2.2, 2.3, 3.2, 4.2, 4.4, 5.1, 6.5, 6.10, 7.3, 7.5, 7.7, 8.1, 8.2, 8.7, 9.2 | **Done** | [`STEP2_STATEMENTS.md`](STEP2_STATEMENTS.md) |
| 3 | One registered rerun of Stages 8-14 with every result-changing fix ("refs5") | 4.1, 4.3, 4.8, 4.9, 5.3, 5.4, 5.7, 5.8, 6.1-6.6, 6.8, 6.9, 7.1, 7.2, 7.5-7.10, 8.1, 8.2, 8.4, 9.1, 9.3, 9.5, 9.7 | **Done. refs5 is the default configuration since 2026-11-15** | [`STEP3_PREREG.md`](STEP3_PREREG.md), [`STEP3_RESULTS.md`](STEP3_RESULTS.md) |
| 4 | Missing validations | 3.2, 3.6, 5.1, 5.2, 5.5, 5.6, 7.3, 7.4, 9.2 | **Done** | [`STEP4_PREREG.md`](STEP4_PREREG.md), [`STEP4_RESULTS.md`](STEP4_RESULTS.md), [`STEP4_V5_V6_RESULTS.md`](STEP4_V5_V6_RESULTS.md) |
| 5 | Corpus and model findings: measure and disclose | 1.1, 1.2, 1.3, 1.5, 1.6, 1.7, 2.5, 2.6, 3.4, 3.5 | **Done** | [`STEP5_PREREG.md`](STEP5_PREREG.md), [`STEP5_RESULTS.md`](STEP5_RESULTS.md) |
| 6 | Records, tests, hygiene | 1.10, 2.4, 2.7, 3.7, 4.10, 5.8, 6.7, 8.6, 9.4, 9.6, cross-stage | **Done; every change checked output-neutral** | [`STEP6_HYGIENE.md`](STEP6_HYGIENE.md) |
| - | What an expert should review | 4.2, 4.6 and what the steps turned up | **Written** | [`LABEL_REVIEW_PLAN.md`](LABEL_REVIEW_PLAN.md) |
| - | Living documents, notebooks 8-12 and the preprint brought to refs5 | - | **Done** | [`docs/DOCS_REFS5_UPDATE.md`](../../DOCS_REFS5_UPDATE.md) |

## Where the work departs from the plan agreed on 2026-11-14

Each departure was found on re-opening the evidence. Each is conservative and is flagged for the PI.

| Plan or registration said | Done instead | Why |
|---|---|---|
| Remove the 11 non-ribosomal genes from the exclusion list | Rebuild the whole list by rule in the current annotation: 10 kinases and Kgd4 leave, 57 dead IDs leave, 285 genes enter | Rps19bp1 stays (BayesPrism lists it). Building the list showed its IDs came from an earlier annotation: 47 to 116 ribosomal protein genes per tissue were never excluded (`STEP3_PREREG.md`, Amendment 1) |
| Drop the EDS arm from TESTES | Keep the pooled reference as primary; build a control-only reference as a sensitivity | The pooling is a recorded decision of the collaborator ("normal-testosterone arms"). The audit showed the validator does not document it, not that it is wrong |
| A minimum-fraction rule for analysable blocks | A reporting tier at mean fraction 0.01, beside the registered rule | The top hotspot (SKMVL satellite cells, 0.0023) is below it, and that was known when the rule was written. A primary rule chosen with the result in view is the practice the audit criticised |
| Stage 8 runs once | Stage 8 ran at `32bb97b`, then again with Stage 9 at `44332e6` | The target sum can only be set after Stage 8, and setting it is a commit. The repeat makes "every stage from one clean commit" literally true, and it reproduced the first run byte for byte |
| The file-walk repair (finding 1.6) needs no switch | It is a switch, off by default | The Stage 2 reproduction check counts all 868 files of the v2 file list, copies included |
| Step 5 M3: "one name became a different name" bounds the counts under a wrong gene | That class was split by the token the model receives (same gene, other gene, alignment lost) | The class mixed reassignments with renames of the same gene |
| Step 4 V3: series-level exclusions for the held-out set | File-level rules as well (not rat, not raw counts, duplicate), fixed before any loss was computed | Reading the files showed series-level rules let non-rat matrices, non-count tables and a second deposit through |
| Step 4 V4: the myeloid replication "is scored" | Its scoring rule was registered in an addendum before any perturbation ran | The 2026-07-03 registration froze no null for it |
| - | A descriptive check of whether the technical covariates differ between training groups, made after the DE robustness result | To help read that result. It is labelled as not registered |

## What was not repeated on refs5

These analyses sit outside the registered rerun (Stages 8-14) and the six validations. Each was last run on
an earlier default, and the documents that report them now say so. None was rerun without a registration.

| Analysis | Last run on | Where it is reported |
|---|---|---|
| Projection onto the Tabula Sapiens atlas (E.3) | The default of 2026-11-04 (9 hotspots) | `translation/ATLAS_PROJECTION_16_RESULTS.md` |
| Comparison of the DE with the published MoTrPAC results | The June references, with a section on the default of 2026-11-01 | `deconvolution/AIM2_DE_MOTRPAC_COMPARISON.md` |
| Second heart reference check (`REFERENCE_RELABEL_PREREG.md` §15) | references_v4, the HEART reference before its merge | `docs/OPEN_FRONTIER.md` |
| The BLOOD composite's checks | references_v4; the composite is byte-identical in references_v5, the exclusion list is not | `docs/OPEN_FRONTIER.md` |
| Two post-hoc parenchyma analyses of the positive-control miss (bulk-to-parenchyma slope correlation; bulk direction of the canonical controls) | The default before refs5 | Removed from the preprint; they need recomputing to return |
| Hub strength against the TF's own expression (finding 9.4) | Earlier network arms | Removed from the preprint |

## How the work was run

- **One clean commit for the rerun.** Stages 8 to 14 ran from `44332e6` with no modified tracked file. While
  they ran, all other work was committed on side branches in separate worktrees and merged afterwards.
- **Background agents** did four pieces under written briefs: the held-out model test, the SimBu and
  mixture-panel batteries, the Step 6 hygiene work, and the rewrite of the living documents. Their reported
  numbers were checked against the record tables before being quoted here; the order of steps of the held-out
  test was checked against job logs and file times.
- **Nothing was pushed.**

## Layout

| Path | Holds |
|---|---|
| `STEP<n>_*.md` | One report per step (`_PREREG` written before results, `_RESULTS` after) |
| `LABEL_REVIEW_PLAN.md` | What a domain expert and a statistician should read, in order |
| `figures/` | Every figure, as PNG. Colours are checked for colour-vision-deficiency separation (`analysis/audit_remediation/figstyle.py`) |
| `record/step<n>/` | The table behind every figure and every quoted number |
| `record/crossref_remediation.tsv` | The Crossref record of each reference added here |
| `analysis/audit_remediation/` (repository root) | The scripts that made the tables and figures |
