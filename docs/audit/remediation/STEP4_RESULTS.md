# Step 4 results: the validations the audit found missing

Each check below was registered in `STEP4_PREREG.md` before the refs5 rerun produced a number. Sections are
filled in as their jobs finish.

| | Check | Registered reading | Result |
|---|---|---|---|
| V1 | Expression accuracy by purity sweep, dominant and hotspot cell types | Median correlation of at least 0.95 at every purity of 0.5 and above | *(running)* |
| V2 | Positive controls on refs5 | Red flag stands if fewer than half of the testable sanity controls recover | *(after Stage 10)* |
| V3 | Held-out test of the model | Held-out loss at most 1.10 times the in-sample loss | **Generalises: 1.046** |
| V4 | Module A on the current model and references | Criteria 1, 3 and 4; myeloid replication reported, not a gate | **1, 3 and 4 pass. Replication fails:** Spi1 at the 81st percentile; Cebpa not testable |
| V5 | mRNA-content bias with a bias-free baseline | Per-reference RMSE differences | *(running)* |
| V6 | Mixture panel on refs5 | The registered floor | *(running)* |

## V3. A held-out test of the model (findings 3.2, 3.6)

*(written below when the order of its steps is confirmed)*

## V4. Module A on the current model and references (finding 9.2)

**What ran.** The v2 model, the refs5 pseudo-cells, the perturbation engine unchanged (job 12378166).
- **Muscle:** the frozen 80-gene panel plus Alb and Snap25, deleted one gene at a time in the 50 SKMVL
  "Skeletal muscle cells" pseudo-cells.
- **Monocytes:** the replication the 2026-07-03 registration named and nobody had scored. Its rules were
  written on 2026-11-14 before any perturbation ran (addendum to V4): every gene within 15% of the factor's
  mean value in the 50 BLOOD "Classical monocytes" pseudo-cells is its null.

![Module A](figures/step4_v4_module_a.png)

*Figure V4. Share of expression-matched genes whose deletion moves the cell embedding less than the factor's
does. (a) Mef2c in skeletal muscle cells. (b) Spi1 in classical monocytes. The dashed line is the registered
90th percentile. Tables: `record/step4/v4_module_a/muscle/panel_shifts_with_roles.tsv`,
`record/step4/v4_module_a/monocytes/myeloid_panel_shifts.tsv`.*

### The registered criteria, in muscle

| # | Criterion | refs5, v2 | v2 on the July references (2026-10-24) | April on the July references |
|---|---|---|---|---|
| 1 | Mef2c's shift above the 90th percentile of its matched null | **100th of 26 genes: pass** | 100th of 26 | 96th of 27 |
| 3 | Alb and Snap25 in no cell, or below the null median | **in 0 cells each: pass** | pass | pass |
| 4 | Shift rises with expression (Spearman above 0) | **0.77: pass** | 0.76 | 0.74 |
| 2 | Mef2c's top 25 targets in the muscle set | withdrawn; 0 of 25, printed for the record | 0 of 25 | 4 of 25 |

- **Criteria 1, 3 and 4 pass on the current model and references**, with numbers close to the earlier v2 run.
  The audit's point was that they had never been scored on them.
- **Criterion 2 carries no verdict** (PI decision 4). The overlap is 0 of 25, as it was for v2 before.
- **How well the frozen panel still matches.** The panel was matched to Mef2c on the July pseudo-cells. On
  refs5, Mef2c's mean value is 1.65, the 16th highest of the 77 panel genes that are in the cells, and 26
  panel genes fall within 15% of it (1.40 to 1.86). Three panel genes are in no muscle pseudo-cell.

### The myeloid replication

| Factor | In the cells | Matched null | Shift | Percentile | Reading |
|---|---|---|---|---|---|
| Spi1 | 50 of 50 pseudo-cells, mean value 1.43 | 232 genes | 2.3e-6 (null median 1.3e-6, 90th percentile 4.4e-6) | 81st | **does not replicate** |
| Cebpa | token in no classical-monocyte pseudo-cell | - | - | - | **not testable** (coverage, step 1 of the registered ladder) |

- **The replication fails.** The one testable factor, Spi1, moves the monocyte embedding more than 81% of
  equally expressed genes. The registered bar is 90%.
- **It was registered as a replication, not a gate,** so Module A's verdict on criteria 1, 3 and 4 stands as
  scored. What the failure does say: the muscle result for Mef2c is one factor in one cell type, and the
  second lineage chosen in advance to guard against "a muscle-only artifact" does not show it.
- **Why Cebpa is absent.** Its token is not among the top 2,048 genes of any classical-monocyte pseudo-cell.
  That is a limit of the input (the deconvolved expression of a low-expressed gene), not a test of the model.
- **Printed without a verdict:** over the 233 perturbed genes the shift rises with expression, Spearman 0.22
  (0.77 in muscle).

### What this does and does not show

- The shifts are of order 1e-6 in cosine distance. Criterion 1 is a ranking among about 26 genes, on 50
  pseudo-cells that are 50 animals' estimates of one cell type, not 50 cells.
- A null of 26 genes makes the 90th percentile coarse: Mef2c had to beat 24 of them.
- Deleting a gene moves every lower-ranked gene up one position, and the model uses position embeddings, so a
  shift holds both effects. No control separates them (finding 9.6).
- With the network results (the decoy null finds no factor-specific signal), Module A supports this much:
  the engine runs, behaves sensibly on not-expressed genes, and ranks one muscle factor above its matched
  genes. It does not establish that the model's perturbation responses identify a factor's targets.

Record: `record/step4/v4_module_a/` (`muscle/report.md`, `report.json`, `results.tsv`;
`monocytes/myeloid_replication.tsv`, `panel.tsv`).

## V1, V2, V5, V6

*(filled in as they finish)*
