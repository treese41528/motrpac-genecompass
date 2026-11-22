# repfdr's 8-week sex-consistency fit on the 16-tissue default (2026-11-10)

> **2026-11-15: a dated record.** "The 16-tissue default" here is the run on the refs4 references, where
> repfdr was fitted on every gene-block. In **refs5** the primary reading is per-sex concordance and repfdr is
> fitted on dose-significant gene-blocks only, as the consortium does: 139 calls by concordance, 3,623 by
> repfdr, 3,629 in their union (`docs/audit/remediation/STEP3_RESULTS.md`, A4 and B5).

**Question.** `DECONVOLUTION_PIPELINE_REPORT.md` left one item unchecked on the current references: whether
repfdr's fit is still degenerate. It was degenerate in the July-reference v2 run, so every 8-week sex-consistency
call there rested on per-sex concordance alone. Stage 10 has recorded the fit's status in `de_methods.tsv` since
fb27b46 (2026-10-05). This note reads that status back for each run on disk and counts where the calls come from.

- Script: `repfdr_call_sources.py` (85b9b95, e104d6a); job 12304673.
- Record: `repfdr_record_16/`.
- Nothing in Stage 10 was rerun or changed. The script recomputes every state with `run_pseudobulk_de.R`'s rule,
  and the recomputed states match the tables' `sexcons_8w` exactly in all five runs.

**Answer: the fit is not degenerate on the default, but it supplies almost every call from a poorly fitted
density.**

| Run | Analysable blocks (library rule) | repfdr fit | f(z) misfit, male / female | repfdr calls (Fdr < 0.05) | Concordance calls | Calls on repfdr alone |
|---|---|---|---|---|---|---|
| **Default, 16 tissues** | **151** (median library ≥ 1,000) | **not degenerate** | 678 / 614 | **5,126** | **144** | **4,982 (97%)** |
| Default, rule at 100 | 158 | not degenerate | 810 / 725 | 5,106 | 144 | 4,962 |
| Default, no rule | 197 | degenerate (female null fraction 1) | 3,728 / 2,831 | 0 | 144 | 0 |
| refs4, 14 tissues | 124 (≥ 1,000) | not degenerate | 687 / 637 | 5,704 | 144 | 5,560 |
| v2, July references | 172 (no rule) | degenerate (female null fraction 1) | 3,775 / 3,398 | 0 | 136 | 0 |

A concordance call needs both per-sex 8-week BH tests at < 0.05, with the same sign. All calls are directional
(up_both or down_both); no run has an "opposite" call.

- **What changed it is the analysable-block rule** (`REFERENCE_RELABEL_PREREG.md` §6).
  - With the near-empty blocks left in the pooled z matrix, `ztobins` puts the female column at 100% null and
    every Fdr is 1. That happens on the same current references with no rule, as it did in July.
  - Without those blocks, the fit is not degenerate: no Fdr equals 1, and the smallest is 1.9e-05.
- **The density fit is still poor.** Every non-degenerate run warns "f(z) misfit ... Rerun with increased df",
  at 614–810 per sex.
- **Every concordance call is also a repfdr call.** On the default:
  - The 144 concordance calls are HEART 120, SKMVL 23 and SKMGN 1.
  - repfdr adds 4,982 more: HEART 1,351, SKMVL 1,054, BLOOD 827, SKMGN 657, LUNG 267, LIVER 259, SPLEEN 224,
    KIDNEY 105, WATSC 96, BAT 95, SMLINT 26, HIPPOC 11, HYPOTH 5, CORTEX 5.
  - OVARY and TESTES are single-sex and have no calls.

**Consequences.**
- **`de_summary.tsv`'s counts are mostly repfdr calls.** The default's sex-consistent counts (2,089 up_both,
  3,037 down_both) are 97% repfdr calls from a fit with a poor density estimate.
- **Concordance alone gives 144.** Read as concordance calls, which is how the preprint describes them, the
  default has 144 sex-consistent gene-blocks, not 5,126.
- **Nothing downstream reads the sex-consistency states.** The gate, hotspots, positive controls, transfer,
  GRNs and the dose significance (`FDR_IHW`) do not use them. Only the `de_summary.tsv` counts, and any statement
  about sex-consistent genes, depend on them.
- **Making the repfdr calls usable needs a registered change.** Options include a refit with more `ztobins` df,
  or MoTrPAC's practice of running repfdr only on features already significant at 5% IHW FDR (see the header of
  `run_pseudobulk_de.R`). That is a methods change, so it is not done here. Until then, only the 144 concordance
  calls should be read as sex-consistent.
