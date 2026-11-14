# Step 4 registration: the validations the audit found missing

Registered 2026-11-14, before the refs5 rerun has produced a fraction, a pseudo-cell or a hotspot list. Each
item below is a check the pipeline's claims lean on and that either was never run on the current references,
had no pass rule, or had its rule written after its result.

They run on refs5 output (`STEP3_PREREG.md`). Thresholds are taken from the method papers or from tolerances
this project registered earlier; none is derived from the data it will judge.

## V1. Expression accuracy by purity sweep (finding 5.1)

**The gap.** The sweeps had no registered pass rule, tested only each tissue's dominant type, and did not
cover LIVER on its current reference. No hotspot cell type has an expression-accuracy test, and the hotspots
are where the per-cell-type claims are made.

**Method, unchanged.** Holdout sweeps as in Stage 14: 30% of a reference's cells are held out and mixed at set
purities of the focal type; BayesPrism deconvolves the mixtures against the other 70%; the estimated
expression of the focal type is compared with its true expression by Pearson correlation on
variance-stabilised values, the measure of the BayesPrism paper [chu2022].

**What is added.**

| Sweep | Focal type | Purity grid |
|---|---|---|
| Every tissue with a reference (16), dominant type | As configured; LIVER hepatocytes, BLOOD erythroid cells, SPLEEN, OVARY, WATSC and HIPPOC are added | 0.05 to 0.9, as now |
| Every refs5 hotspot and borderline block | That block's cell type | 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9 |

**Correction (2026-11-14, before any refs5 sweep was built).** The first row's grid is Stage 14's own, 0.1, 0.3,
0.5, 0.7, 0.85, 0.95. "0.05 to 0.9" is the default of the sweep builder, which Stage 14 does not use; "as now"
is what was meant. The dominant type of a tissue is the type with the highest mean estimated fraction in
the refs5 Stage 8 output.

**Pass rule, fixed now.** From the BayesPrism benchmark, which reports correlation above 0.95 above about 50%
purity [chu2022]:
- **PASS:** median correlation of at least 0.95 at every purity of 0.5 and above.
- For hotspot types a second reading is given at the grid point nearest the type's mean estimated fraction in
  the bulk: **validated at its abundance** if the median correlation there is at least 0.95, otherwise **not
  validated at its abundance**, with the lowest purity at which it reaches 0.95.

**What a sweep cannot show.** Reference and mixtures come from one study, so a sweep tests the method, not the
match between the reference and MoTrPAC tissue. A type with too few cells to hold out 20 is reported as not
testable.

## V2. Positive controls (findings 7.3, 7.4)

Rescored on refs5 by the specification of `deconvolution/POSCTRL_PREREG.md`, unchanged. Reported in this
order: the pipeline-sanity subset (34 muscle and heart controls), then the tiers. If the sanity subset again
recovers fewer than half of its testable controls, the documents say the registered red flag stands.

The data-anchored parenchyma check is reported as a later, unregistered analysis, with its fixed "pipeline is
sound" sentence replaced by the numbers.

## V3. A held-out test of the model (findings 3.2, 3.6)

**The gap.** No cell was held out of training, by a recorded decision, and adoption of v2 rested on in-sample
loss. Nothing shows how the model does on a rat study it has not seen.

**The held-out set.** The single-cell series that the LLM step confirmed and a catalog merge then dropped
(finding 1.1): rat single-cell data of the same kind as the corpus that never reached cell QC, so never
reached training. Step 5's M2 puts those with a usable matrix through the same Stage 2 QC into a folder of
their own.

**Exclusions before any loss is computed.** A series is removed if any of its samples shares a GEO sample
accession with a training file, or if it is a sub-series or super-series of a training series.

**Measurement.** The masked-token evaluator used for the retrain's criteria C1 and C2
(`finetune/genecompass/eval_mlm.py`), same masking and seed, on:
- the held-out cells, capped at 2,000 per study, drawn with a fixed seed;
- the in-sample rat probe of C1.

for v2 and, for comparison, the earlier (April) model. Both models have seen none of the held-out cells.

**Criterion, fixed now.** The 10% tolerance is the one the retrain registered for "kept" species
(`docs/RETRAIN_V2_PREREGISTRATION.md`, C2).

| Reading | Rule |
|---|---|
| **Generalises** | v2's gene-identity loss on held-out cells is at most 1.10 times its loss on the in-sample probe |
| Partly | Between 1.10 and 1.25 times |
| Does not | Above 1.25 times |

Reported with it: the value loss, the per-study and per-tissue losses, the number of studies and cells, and
the per-tier loss (inherited, multi-mapping and rat-specific tokens).

**Limits, stated now.** The held-out studies went through the same harvest as the training studies, so this
tests generalisation to a new study, not to a new kind of data. Held-out cells have no cell-type labels, so
the clustering checks are not repeated on them.

## V4. Module A on the current model and references (finding 9.2)

- Criteria 1 (the deleted TF ranks above expression-matched genes), 3 (not-expressed negative controls give no
  cells) and 4 (shift is monotone in expression) are rescored with v2 on the refs5 SKMVL skeletal-muscle
  pseudo-cells, thresholds as in `deconvolution/PERTURB_PREREG.md`.
- The myeloid replication the registration named (Spi1 and Cebpa in classical monocytes) is scored for the
  first time. It is a replication, not a gate, as registered.
- **Criterion 2 is withdrawn** as a test of predicted targets (PI decision 4). Its overlap is printed for the
  record and carries no verdict.

**Addendum to V4 (2026-11-14, written before any perturbation on refs5 and before the refs5 BLOOD dataset was
read for these genes).** V4 as written does not say how the myeloid replication is scored, and the 2026-07-03
registration froze no null for it. The rules below are fixed now.

| Item | Rule |
|---|---|
| Muscle cells | refs5 SKMVL, cell type "Skeletal muscle cells" (the name of the July "Skeletal muscle" type since the joint re-annotation), all pseudo-cells |
| Muscle panel | The frozen 80-gene panel (`data/deconvolution/perturb_prereg/prereg_genes.txt`) plus Alb and Snap25, unchanged. Its matched null is, as the scorer has always computed it, the panel genes whose mean value in the scored cells is within 15% of Mef2c's |
| Monocyte cells | refs5 BLOOD, cell type "Classical monocytes", all pseudo-cells |
| Monocyte null | For each of Spi1 and Cebpa: every gene whose token is in the input of the monocyte pseudo-cells and whose mean value over them is within 15% of the factor's (the scorer's rule for Mef2c). All of them are perturbed; none is drawn at random. The two factors are left out of each other's null |
| Reading | A factor **replicates** if its cell shift is above the 90th percentile of its null (criterion 1's threshold, one-sided). It is **not testable** if its token is absent from the cells or present in fewer than 20 pseudo-cells (steps 1 and 2 of the registered ladder) |
| Targets | Not scored. The 2026-07-03 text expects the factors' targets to "enrich for myeloid genes", but froze no myeloid set, and the construct is the one withdrawn as criterion 2 |
| Also printed, no verdict | Spearman correlation of shift with mean value over every perturbed gene in the monocytes (criterion 4's quantity), and where Mef2c's mean value ranks among the muscle panel on refs5, since the panel was matched on the July pseudo-cells |

The replication is reported and is not a gate, as registered. Code: `analysis/audit_remediation/module_a_v4.py`;
`deconvolution/score_perturb_prereg.py --withdraw-criterion-2`.

## V5. mRNA-content bias with a baseline that has no bias (finding 5.2)

**The gap.** In the battery of 2026-11-11 the "no bias" arm summed raw counts, so it kept each reference's
native differences in mRNA content. The registered question (does BayesPrism degrade under bias, and does any
method correct it) was asked against a baseline that already contained some of the bias.

**Run.** The same battery (`SIMBU_BIAS_REFS4ALL_PREREG.md`: 50 mixtures of 1,000 cells, six methods, 15
references, seed 1) on the refs5 references, with three arms:

| Arm | SimBu setting | Is |
|---|---|---|
| A | `remove_bias_in_counts = TRUE`, scaling NONE | No mRNA bias |
| B | `remove_bias_in_counts = FALSE`, scaling NONE | Native bias only (the earlier "no bias" arm) |
| C | scaling `expressed_genes` | Added bias (the earlier "bias" arm) |

**Readings, fixed now.** Scored against cell fractions, as before.
- **BayesPrism under native bias:** pooled RMSE(B) minus RMSE(A), per reference. Reported as the number of
  references where it is positive.
- **BayesPrism under added bias:** RMSE(C) minus RMSE(A).
- **Correctors:** the earlier registered rule (MuSiC and SCDC degrade less than the others) is scored on
  RMSE(C) minus RMSE(A) over the references every method can run. Reported as reproduced or not.
- The 2026-11-11 result (B against C) is reported beside it on the refs5 references.

## V6. Mixture panel on the refs5 references (findings 5.5, 5.6)

The registered panel (`MIXTURE_PANEL_REFS4ALL_PREREG.md`: 19 sets, floor on BayesPrism's median per-type
correlation) is rerun on refs5, because the references and the gene exclusions changed.

Two corrections, stated now:
- **DWLS** runs as the benchmark plan specified (`OMNIDECONV_BENCHMARK_PLAN.md`): `mast_optimized`, p 0.05. The
  earlier panel ran `mast` at 0.01.
- The cross-check runner writes a provenance record, and the scoring message says what the code does with a
  type a method does not estimate (it is left out of that method's scores).

Not changed, and disclosed: mixture compositions are flat Dirichlet, unlike tissue; a pool as small as 20
cells is resampled; BayesPrism sees every reference cell and the other methods a capped subset.

**Pass rule:** the registered floor, unchanged. HYPOTH and HEART are read with attention, because their
labels changed.

## Order

V1 and V2 come with the refs5 chain (Stages 14 and 10). V4 needs the refs5 SKMVL pseudo-cells. V3 needs
Step 5's M2. V5 and V6 are the longest and run last.

## References

- [chu2022] Chu T, Wang Z, Pe'er D, Danko CG. Cell type and gene expression deconvolution with BayesPrism
  enables Bayesian integrative analysis across bulk and single-cell RNA sequencing in oncology. Nat Cancer
  2022;3:505-517 (in `docs/audit/REFERENCES.md`; the 0.95 benchmark was read in `readings/s43018-022-00356-3.pdf`,
  page 2).
