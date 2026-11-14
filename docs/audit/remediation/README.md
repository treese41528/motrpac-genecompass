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

## Order and status

| Step | What | Findings | Status | Document |
|---|---|---|---|---|
| 1 | Integrity checks: could anything be silently wrong? | 3.1, 6.7, 6.9, 7.2 (imbalance), 8.4 | **Done. All clean or explained; guards added** | [`STEP1_INTEGRITY.md`](STEP1_INTEGRITY.md) |
| 2 | Statements that are wrong whatever a rerun shows | 1.2, 1.4, 1.8, 1.9, 2.1, 2.2, 2.3, 3.2, 4.2, 4.4, 5.1, 6.5, 6.10, 7.3, 7.5, 7.7, 8.1, 8.2, 8.7, 9.2 | **Done** | [`STEP2_STATEMENTS.md`](STEP2_STATEMENTS.md) |
| 3 | One registered rerun of Stages 8-14 with every result-changing fix | 4.1, 4.3, 4.8, 4.9, 5.3, 5.4, 5.7, 5.8, 6.1-6.6, 6.8, 6.9, 7.1, 7.2, 7.5-7.10, 8.1, 8.2, 8.4, 9.1, 9.3, 9.5, 9.7 | **Registered; in progress** | [`STEP3_PREREG.md`](STEP3_PREREG.md) |
| 4 | Missing validations | 3.2, 3.6, 5.1, 5.2, 5.5, 5.6, 7.3, 7.4, 9.2 | Registered; waits for refs5 | [`STEP4_PREREG.md`](STEP4_PREREG.md) |
| 5 | Corpus and model findings: measure and disclose | 1.1, 1.2, 1.3, 1.5, 1.6, 1.7, 2.5, 2.6, 3.4, 3.5 | Registered; measurements running | [`STEP5_PREREG.md`](STEP5_PREREG.md) |
| 6 | Records, tests, hygiene | 1.10, 2.4, 2.7, 3.7, 4.10, 5.8, 6.7, 8.6, 9.4, 9.6, cross-stage | Not started | - |
| - | Expert review plan | 4.2, 4.6 | Last | - |

## Where the work departs from the plan agreed on 2026-11-14

Each departure was found on re-opening the evidence. Each is conservative and is flagged for the PI.

| Plan said | Done instead | Why |
|---|---|---|
| Remove the 11 non-ribosomal genes from the exclusion list | Rebuild the whole list by rule in the current annotation: 10 kinases and Kgd4 leave, 57 dead IDs leave, 285 genes enter | Rps19bp1 stays (BayesPrism lists it). Building the list showed its IDs came from an earlier annotation: 47 to 116 ribosomal protein genes per tissue were never excluded (`STEP3_PREREG.md`, Amendment 1) |
| Drop the EDS arm from TESTES | Keep the pooled reference as primary; build a control-only reference as a sensitivity | The pooling is a recorded decision of the collaborator ("normal-testosterone arms"). The audit showed the validator does not document it, not that it is wrong |
| A minimum-fraction rule for analysable blocks | A reporting tier at mean fraction 0.01, beside the registered rule | The top hotspot (SKMVL satellite cells, 0.0023) is below it, and that was known when the rule was written. A primary rule chosen with the result in view is the practice the audit criticised |

## Layout

| Path | Holds |
|---|---|
| `STEP<n>_*.md` | One report per step |
| `figures/` | Every figure, as PNG. Colours are checked for colour-vision-deficiency separation (`analysis/audit_remediation/figstyle.py`) |
| `record/step<n>/` | The table behind every figure and every quoted number |
| `record/crossref_remediation.tsv` | The Crossref record of each reference added here |
| `analysis/audit_remediation/` (repository root) | The scripts that made the tables and figures |
