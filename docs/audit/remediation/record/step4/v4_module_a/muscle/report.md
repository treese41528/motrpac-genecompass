# Module A pre-registration -- RESULTS

Scored on Skeletal muscle cells pseudo-cells (50 cells; background = 2604 expressed genes). Panel: 77 genes (Mef2c + 26 expression-matched null + value-spanning). See `PERTURB_PREREG.md` for the frozen criteria.

| Criterion | Result | Verdict |
|---|---|---|
| #1 Mef2c CLS-shift vs matched null (>90th pctile) | Mef2c at **100th** pctile of 26 matched genes (shift 6.03e-06) | PASS |
| #4 self-consistency Spearman(shift, value) (>0) | rho=**0.77** (p=2.1e-16) | PASS |
| #2 Mef2c top-25 targets in curated muscle/mito set | **0/25** hits (background K=48/2604); hypergeom p=**1.0e+00** | WITHDRAWN (printed for the record; no verdict) |
| #3 not-expressed negatives (Alb/Snap25) | Alb n_cells=0; Snap25 n_cells=0 (matched-null median 1.47e-06) | PASS |

**Mef2c target hits in the muscle/mito set:** (none).

**Overall (criteria 1, 3 and 4; criterion 2 withdrawn):** 1, 3 and 4 pass. Note (from the smoke findings): the single-deletion CLS shift is near-noise in absolute terms, so #1 tests the *ranking* (is Mef2c above matched genes?), and the GRN (Module B) uses the per-gene TARGET shift, not the cell shift.
