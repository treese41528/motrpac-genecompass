# Module A pre-registration -- RESULTS

> **Read this first (2026-11-14).** This file is the April model's scoring only. The retrained model (v2, the
> default) passes criteria 1, 3 and 4 and **fails criterion 2** (0 of 25; `docs/RETRAIN_V2_RESULTS.md`, Module A).
> Criterion 2's curated set was written after the smoke run had shown the targets, so it cannot validate
> them, and it is **withdrawn as a test of the predicted targets**. "ALL PASS" below describes April only.
> The registered myeloid replication was never scored. See `docs/audit/remediation/STEP2_STATEMENTS.md`.

Scored on Skeletal muscle pseudo-cells (50 cells; background = 2549 expressed genes). Panel: 80 genes (Mef2c + 52 expression-matched null + value-spanning). See `PERTURB_PREREG.md` for the frozen criteria.

| Criterion | Result | Verdict |
|---|---|---|
| #1 Mef2c CLS-shift vs matched null (>90th pctile) | Mef2c at **98th** pctile of 52 matched genes (shift 8.15e-06) | PASS |
| #4 self-consistency Spearman(shift, value) (>0) | rho=**0.70** (p=8.3e-13) | PASS |
| #2 Mef2c top-25 targets in curated muscle/mito set | **7/25** hits (background K=48/2549); hypergeom p=**2.0e-07** | PASS |
| #3 not-expressed negatives (Alb/Snap25) | n_cells=0 (from the smoke run) | PASS |

**Mef2c target hits in the muscle/mito set:** ENSRNOG00000005934, ENSRNOG00000008569, ENSRNOG00000051534, ENSRNOG00000017752, ENSRNOG00000020244, ENSRNOG00000015962, ENSRNOG00000002218.

**Overall:** ALL PASS. Note (from the smoke findings): the single-deletion CLS shift is near-noise in absolute terms, so #1 tests the *ranking* (is Mef2c above matched genes?), and the GRN (Module B) uses the per-gene TARGET shift, not the cell shift.
