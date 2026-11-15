# Step 4 results: the validations the audit found missing

Each check below was registered in `STEP4_PREREG.md` before the refs5 rerun produced a number. Sections are
filled in as their jobs finish.

| | Check | Registered reading | Result |
|---|---|---|---|
| V1 | Expression accuracy by purity sweep, dominant and hotspot cell types | Median correlation of at least 0.95 at every purity of 0.5 and above; for called types, also at their own abundance | Dominant types: 12 of 15 pass, 2 fail, 1 not testable. **Called blocks: 4 of 12 validated at their abundance**, 7 not, 1 not testable |
| V2 | Positive controls on refs5 | Red flag stands if fewer than half of the testable sanity controls recover | **The red flag stands: 5 of 26 testable sanity controls recovered** (5 of 34 in all) |
| V3 | Held-out test of the model | Held-out loss at most 1.10 times the in-sample loss | **Generalises: 1.046** |
| V4 | Module A on the current model and references | Criteria 1, 3 and 4; myeloid replication reported, not a gate | **1, 3 and 4 pass. Replication fails:** Spi1 at the 81st percentile; Cebpa not testable |
| V5 | mRNA-content bias with a bias-free baseline | Per-reference RMSE differences | *(running)* |
| V6 | Mixture panel on refs5 | The registered floor | *(running)* |

## V3. A held-out test of the model (findings 3.2, 3.6)

**Result.** On rat cells from 33 studies the model never saw, v2's gene-identity loss is 1.046 times its loss
on the in-sample probe. The registered bar for "Generalises" is 1.10.

| Model | Cells | Studies | Gene-identity loss | Value loss | Top-1 accuracy |
|---|---|---|---|---|---|
| v2, in-sample probe | 2,000 | 77 | 3.0590 | 2.48e-4 | 25.5% |
| **v2, held-out** | **66,000** | **33** | **3.1991** | 3.75e-4 | 23.5% |
| April, in-sample probe | 2,000 | 77 | 5.3102 | 3.92e-4 | 18.8% |
| April, held-out | 66,000 | 33 | 5.3999 | 5.65e-4 | 17.7% |

| Registered quantity | Value | Reading |
|---|---|---|
| v2 held-out loss / v2 probe loss | **1.0458** | **Generalises** (at most 1.10) |
| The same for April (context; April saw none of these cells either) | 1.0169 | - |

Both models scored the same held-out cells under the same masks (13,805,095 masked tokens). The evaluator's
recorded probe losses were reproduced digit for digit before any held-out cell was scored
(`record/step4_heldout/probe_reproduction.tsv`).

![Held-out test](figures/step4_v3_heldout.png)

*Figure V3. (a) Each held-out study's loss over the pooled in-sample probe loss; the shaded band is the range
of the 28 training studies with at least 20 probe cells. (b) The same ratio by token tier. Tables:
`record/step4_heldout/losses_by_study.tsv`, `losses_by_tier.tsv`, `registered_reading.tsv`.*

### Reported with it, as registered

- **Per study.** The ratio runs from 0.93 to 1.20, median 1.05. 28 of 33 studies are at or under 1.10; five
  are between 1.10 and 1.25 (testis 1.20, spinal cord 1.14, embryonic heart 1.11, dorsal root ganglion 1.11,
  cortex and hippocampus 1.11); none is above 1.25. Training studies themselves span 0.90 to 1.10.
- **Per tier.** The gap grows as tokens lose their cross-species anchor: 1.04 for three-species orthologs,
  1.07 and 1.05 for one-to-one human and mouse orthologs, 1.10 and 1.12 for multi-mapping ones, and **1.18
  for rat-only tokens**. Rat-only tokens are learned from rat data alone, and that is where the model
  transfers least well to a new study.
- **Value loss.** 1.51 times the probe's. No bar was registered for it. It is largest in cells with fewer
  than 500 expressed genes (`losses_by_length.tsv`).
- **Per tissue.** Lowest loss in joint and cartilage (2.86), highest in testis (3.67)
  (`losses_by_tissue.tsv`).

### How the held-out set was built

| Step | Series |
|---|---|
| Returned by both GEO queries and so lost to the corpus (finding 1.1) | 108 |
| Confirmed rat single-cell by the LLM step | 79 |
| With a usable MTX, H5 or h5ad matrix (the registered candidates) | 35 |
| Removed by the registered exclusions | 1 |
| With cells after QC and the file rules | **33** |

- **The registered exclusion removed one series.** GSE304998 is the SuperSeries of training series GSE284698
  and shares 41 of its 71 samples with it. 1,120 training-side sample accessions were compared.
- **Cell QC** is Stage 2's, unchanged, into a folder of its own: 257 files, 4,303,867 cells passing.
- **File rules.** The registration fixed series-level exclusions only. Reading the files showed that was not
  enough, so the corpus plan's own file rules were applied, with two safeguards, before any cell was
  tokenized:

| Action | Files | Cells | Why |
|---|---|---|---|
| Kept | 220 | 3,299,571 | 33 series, 21 tissue groups |
| Dropped: not rat | 10 | 53,857 | 4 human-donor files (by GEO organism); 6 mouse-annotated files with no sample accession (by feature names) |
| Dropped: not raw counts | 13 | 163,916 | Matrices Stage 2 had "reverse-normalised", including pseudo-bulk group tables |
| Dropped: duplicate | 14 | 786,523 | 11 are GSE263307, a second deposit of 11 of GSE171165's libraries under new accessions (99.8% shared barcodes) |

- **The draw.** 2,000 cells from each of the 33 studies, fixed seed: 66,000 cells.
- **Unseen at the level of the cell, not only the accession.** None of the 66,000 token sequences equals any
  of the 6,803,206 training cells, and the largest barcode overlap of a kept file with a training file is
  7.8% (`unseen_check.tsv`, `unseen_barcode_check.tsv`).

### The order of steps

The evaluated cells were fixed before any loss on them existed. The evaluation set was written at 13:21 and
the first evaluation job started at 13:47; both evaluation jobs print the set's checksums at start, and they
equal the files on disk. One preparation job ran again at 14:07, while the v2 evaluation was running, to add
two descriptive columns to the file table. Its log prints the same counts per action as the first run, and
the set itself was not rewritten. No file rule was added or changed after a held-out loss was seen.

### Departures from the registration, and limits

- **The evaluator could not score a rat-only set** (it divides by the number of masked human tokens), and it
  has no per-study output. A wrapper calls its own masking and model-loading functions and reproduces its
  recorded numbers exactly; a test shows both facts.
- **File rules were not registered** (above). They were fixed from file metadata, not from any loss, and they
  removed 1,004,296 of the 4,303,867 QC-passing cells.
- **One sibling series is kept.** GSE304911 shares a SuperSeries with a training series but no sample; the
  registered rule does not remove it. Without it the ratio is 1.048.
- **The two sets are weighted differently.** The probe is weighted by cell over 77 studies; the held-out set
  takes 2,000 cells from each study. Held-out cells are also longer (mean 1,395 expressed genes against
  1,059). Within bins of length the held-out excess is 0.3% to 4.7%.
- **Not registered, for context:** resampling the 33 studies gives a 95% interval of 1.026 to 1.065 for the
  ratio; leaving any one study out gives 1.042 to 1.049.
- **What the test is.** Generalisation to a new rat study that came through the same harvest and the same QC,
  on the masked-token task the model was trained on. It is not a test on a new kind of data, and it says
  nothing about the pseudo-cells of Stages 9 to 13, which are not single cells. Held-out cells have no labels,
  so the clustering checks were not repeated.
- **243 of the 66,000 cells are Visium spots** (GSE299772), kept because the corpus's own rules keep such
  files. Without them the ratio is 1.046.

Scripts: `analysis/audit_remediation/heldout/`. Tests: `tests/test_heldout.py`. Record:
`record/step4_heldout/`. Jobs: 12375158-60 (QC), 12376266 (probe), 12377246 and 12378063 (file plan, tokenize,
draw), 12377965 (v2), 12377966 (April).

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

## V1. Expression accuracy by purity sweep (finding 5.1)

**What ran.** 26 holdout sweeps on the refs5 references (job array 12382906): the cell type with the largest
mean estimated fraction in each tissue (15; SKMGN and SKMVL share a reference), on Stage 14's grid, and every
called cell type (11 sweeps for the 12 called blocks), on the registered grid from 0.5% to 90%.

### Dominant types

| Reading | Tissues |
|---|---|
| **Pass** (median correlation at least 0.95 at every purity of 0.5 and above) | 12: BAT, CORTEX, HEART, HIPPOC, HYPOTH, KIDNEY, LIVER, LUNG, OVARY, SKMGN and SKMVL, SMLINT, WATSC |
| **Fail** | 2: BLOOD erythroid cells (0.83 at 0.5; 0.93 at 0.95) and SPLEEN red-pulp macrophages (0.85 at 0.5; 0.94 at 0.95) |
| Not testable | 1: TESTES Sertoli cells (42 reference cells; too few to hold out) |

- "Dominant" is the type with the largest estimated fraction, as registered. In BAT, HIPPOC and LUNG that is
  endothelial cells, not the tissue's parenchyma.
- HIPPOC passes at the edge (0.954 at 0.5).
- **BLOOD erythroid cells fail at every purity.** They are 71% of the estimated blood, their reference profile
  is 90 cells of one library (finding 4.6), and their expression is never recovered above 0.93.

### Called cell types

![Purity sweeps](figures/step4_v1_purity_sweeps.png)

*Figure V1. Median correlation of estimated with true expression against the share of the mixture that is the
cell type. Open circles mark each type's own abundance in the MoTrPAC tissue. Tables:
`record/step4/v1_purity_sweeps/sweep_points.tsv`, `sweep_verdicts.tsv`;
`record/step3/downstream/called_blocks_all_readings.tsv`.*

| Called block | Mean fraction in the tissue | Correlation at the nearest grid point | Reaches 0.95 at | Validated at its abundance |
|---|---|---|---|---|
| BLOOD classical monocytes | 3.5% | 0.966 | 2% | **yes** |
| BLOOD megakaryocytes and platelets | 14.5% | 0.977 | 5% | **yes** |
| LUNG endothelial cells | 53% | 0.992 | 5% | **yes** |
| LUNG vascular smooth muscle cells | 7.2% | 0.959 | 5% | **yes** |
| BLOOD neutrophils | 7.0% | 0.946 | 10% | no, by 0.004 |
| BLOOD erythroid cells | 71% | 0.872 | never | no |
| HEART vascular smooth muscle cells | 1.3% | 0.68 | 20% | no |
| SKMVL tenocytes | 0.7% | 0.51 | 70% | no |
| SKMVL vascular smooth muscle cells | 0.2% | 0.57 | 70% | no |
| SKMGN vascular smooth muscle cells (borderline) | 0.1% | 0.57 | 70% | no |
| SKMVL satellite cells | 0.2% | 0.50 | 70% | no |
| SKMGN Schwann cells | 0.08% | - | - | not testable (30 reference cells) |

- **Four of the 12 called blocks have their expression recovered at the abundance they are estimated at.**
- **None of the five muscle blocks does.** At 0.5% of a mixture, the lowest point of the grid and still above
  their abundance, the estimated expression of these types correlates at 0.50 to 0.57 with the truth. The
  pseudo-cells the gate scores for them are estimates of that quality at best.
- **The top hotspot, SKMVL satellite cells, is among them** (0.50).

**What a sweep can and cannot show.** Reference and mixtures come from one study, so this is the method's best
case. A failure here is a failure of the method at that abundance. A pass does not show that the reference
matches MoTrPAC tissue.

A side note from Step 6: the sweep generator can leave one or two cells too many in a mixture when it repairs
rounding. Counts and true expression are exact, so scores are unaffected.

## V2. Positive controls (findings 7.3, 7.4)

Rescored on refs5 by the unchanged specification (`deconvolution/POSCTRL_PREREG.md`), in the registered order.

**The pipeline-sanity subset** (Tier A controls in muscle and heart parenchyma, the set the specification
expected to recover):

| Tissue | Recovered | Not significant | Not testable | Controls |
|---|---|---|---|---|
| HEART cardiomyocytes | 2 | 2 | 2 | 6 |
| SKMGN skeletal muscle cells | 2 | 9 | 3 | 14 |
| SKMVL skeletal muscle cells | 1 | 10 | 3 | 14 |
| **All** | **5** | 21 | 8 | **34** |

**The registered red flag stands.** 5 of the 26 testable controls recover (19%). The specification reads a
miss here as "a red flag for the deconvolution to DE pipeline, not a biological null". The current default
gave the same 5 of 34.

**By tier:**

| Tier | Recovered | Not significant | Wrong direction | Not testable | Controls |
|---|---|---|---|---|---|
| A | 6 | 26 | 2 | 11 | 45 |
| Ai | 2 | 12 | 0 | 0 | 14 |
| B | 6 | 9 | 0 | 7 | 22 |

**Why controls are not testable.** Each was traced to where its gene is lost
(`record/step4/v2_posctrl/not_testable_trace.tsv`).

| Control | Where it is lost |
|---|---|
| Sod2 (muscle), Apob (LUNG) | The reference has no gene under the specification's ID |
| Hspa1b, Hsp90aa1 (muscle, heart), Adrb1, Adra2b (WATSC), Atp6v1g2 (SKMVL) | The ID is in the reference's gene list with 0 to 2 counts over all its cells, so BayesPrism drops it. For heat-shock genes in muscle that is not biology; it is the gene-ID drift measured in Step 5: the study's counts sit under another ID |
| Slc6a8 (HEART, LIVER, LUNG) | X-linked; the sex-chromosome filter removes every X-linked gene |

**The data-anchored parenchyma check**, reported as what it is: a later, unregistered description. Of the genes
that move with training in the matched bulk, the dominant cell type's DE calls the same direction for 88 of
215 in SKMGN (41%), 92 of 170 in SKMVL (54%) and 22 of 24 in HEART (92%); LIVER has no bulk mover at the
threshold. These types are 72% to 96% of their tissue, so agreement with the bulk is expected by construction.
It does not lift the red flag.

Record: `record/step4/v2_posctrl/` (`sanity_subset.tsv`, `by_tier.tsv`, `against_default.tsv`,
`parenchyma_dataanchored.tsv`).

## V5, V6

*(the SimBu battery and the mixture panel are running; filled in when they finish)*
