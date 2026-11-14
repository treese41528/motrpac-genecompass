# Step 4, V5 and V6: draft

Draft for `STEP4_PREREG.md` sections V5 (SimBu mRNA-content bias, finding 5.2) and V6 (mixture panel, findings
5.5 and 5.6). Results are filled in below as the jobs finish. The notes in the next section were written and
committed before any simulation, mixture or deconvolution of either battery existed.

## Notes written before any result (2026-11-14)

Each note settles a point the registration leaves open, by the conservative reading, before the output it
affects exists.

**N1. V5, how arm A removes the bias.** The registration gives `remove_bias_in_counts = TRUE` and names no
method. SimBu 1.8.0 has a companion argument, `remove_bias_in_counts_method`, with two values:
- `"read-number"` (SimBu's default): each sampled cell's counts are divided by that cell's own total count
  before the cells are summed, so every cell contributes the same number of reads;
- `"gene-number"`: the counts are divided by the cell's number of expressed genes, which leaves library-size
  differences in place.

Arm A uses `"read-number"`. It is the default and the only value that makes the arm what the registration calls
it, "no mRNA bias". The exact call is recorded in each set's `mixtures/sim_meta.txt`. Checked in the installed
package (`SimBu:::simulate_sample`): with `remove_bias_in_counts = TRUE` the division happens before the scaling
factor is applied, so arm A with scaling NONE sums cells that each total 1.

**N2. V5, a difference between the arms that is not the bias.** SimBu brings every mixture to 1,000,000 reads.
A mixture whose summed counts exceed that is subsampled to it (`phyloseq::rarefy_even_depth`); a mixture below
it is multiplied up without sampling. Arm A's mixtures always total 1,000 before this step (1,000 cells of total
1), so they are multiplied up and then rounded by `simulate_simbu.R`; arms B and C are subsampled wherever
their sum exceeds 1,000,000. Arm A therefore carries rounding only, and B and C carry sampling noise at
1,000,000 reads. This is how SimBu works and is not changed. It is disclosed with the result.

**N3. V5, DWLS setting.** `SIMBU_BIAS_REFS4ALL_PREREG.md` names DWLS as `mast` and says no method setting
changes; it does not state the p cut-off. The battery of 2026-11-11 ran with omnideconv's defaults, which the
runner did not set: p cut-off 0.01 and fold-change cut-off 0.5. V5 runs DWLS as `mast` with p 0.01 and
fold-change 0.5, now passed explicitly and recorded. V6's corrected setting (`mast_optimized`, p 0.05) is not
used in V5, so the comparison with 2026-11-11 stays like for like.

**N4. Both batteries, the gene exclusion list.** The profile `config/profiles/refs5.yaml` is set for every
method, not only BayesPrism. The cross-check runner applies the same gene clean-up as the Stage 8 runner from
the same configuration, so all six methods see the exclusion list of the refs5 rerun
(`deconvolution/reference/rat_exclude_genes_v2.tsv`).

**N5. V5, the common set.** "The references every method can run" is read as the references where all six
methods are scored in all three arms. One set of references then serves the C minus A ranking and the C minus B
ranking. If a method fails in only one arm, so that the set would differ by contrast, both versions are
reported.

**N6. V5, what "the 2026-11-11 result (B against C)" covers.** Readings 1, 2, 3 and 5 of
`SIMBU_BIAS_REFS4ALL_PREREG.md`, recomputed as arm C minus arm B on the refs5 references: the corrector
ranking and its rule, the number of references where BayesPrism's RMSE rises, the mRNA rank of the driver, and
the driver table. Reading 4 (dose-response) is not part of V5 and is not rerun.

**N7. V6, the DWLS patch.** The runner patches DWLS in memory so that a cell type with fewer than two genes
above the fold-change cut-off does not stop the signature build; such a type is counted as having no marker
genes of its own and stays a column of the signature. The patch was written for
`DWLS::buildSignatureMatrixMAST`. `mast_optimized` goes through another function,
`DWLS::buildSignatureMatrixMASTOptimized`, which stops at the same point for the same reason (read in the
installed DWLS 0.5.2). The same guard, with the same meaning, is added there. Without it the registered
setting could not run on a reference with such a type. Every run logs and records the types for which the
guard fired, so the sets where it mattered can be listed.

**N8. V6, inputs.** The registration lists two corrections. Nothing else about the cross-check changes: every
method still gets raw counts for the reference and for the mixtures. Finding 5.6 also notes that the benchmark
plan gave TPM mixtures to MuSiC, SCDC and CIBERSORTx; that is not implemented (the project has no rat
gene-length table) and is disclosed with the three items the registration lists.

**N9. V6, the earlier panel.** `MIXTURE_PANEL_REFS4ALL_PREREG.md` compares each set with "the earlier panel".
For V6 that is the panel of 2026-11-08 on the references then in use (`data/deconvolution/mixture_panel_refs4all`),
which has the same 19 set names. As registered, a drop of more than 0.05 in BayesPrism's median per-type r is
named. For HEART_holdout and HYPOTH_holdout every type's r is tabled beside its earlier value, because their
labels changed.

**N10. V6, cross sets.** The four cross sets use the sources and the committed harmonization overrides of the
earlier panel (`deconvolution/reference/mixture_panel_harmonization_overrides.tsv`). The four references they
are deconvolved against are byte-identical to the earlier builds, so each `harmonization_map.tsv` must equal
the one reviewed on 2026-11-04. This is checked before any cross set is deconvolved. If a map differs, the
cross sets wait for a review on names alone.

**N11. V6, CIBERSORTx.** Its credentials are present, so it runs, with the same cap of 2,000 cells per type as
the other cross-check methods, as in the earlier panel.

**N12. Both batteries, scheduling.** Sets are run several to a job, and some jobs run sets side by side. Core
counts per run are the earlier ones: BayesPrism 8 (V5) and 16 (V6), DWLS 4. Retries are the registered ones
only: a task killed for memory or time is rerun once with more (DWLS additionally at 2 cores).

**N13. Smoke runs.** Each battery is first run on TESTES (the smallest reference with more than one sample)
into folders of its own (`simbu_bench_refs5_smoke`, `mixture_panel_refs5_smoke`). The smoke outputs are used to
test the runner, the provenance record and the table-building commands. They are not part of the result: the
full batteries rebuild and rerun TESTES in the registered output roots.

**N14. V5, what arm A changes besides the bias between types.** Dividing each cell by its own total also
weights the cells of one type equally. Arms B and C weight them by library size, and so does every method's
reference profile of a type, which is built from summed counts. A part of any difference between arm A and
arm B is therefore this change of weighting within a type, not the difference in mRNA content between types.
SimBu offers no setting that removes one without the other. Not changed; disclosed with the result.

**N15. V5, how the check of the arms is made.** For each reference and cell type, SimBu's own `simulate_bulk`
draws one sample of 1,000 cells of that type alone, in each arm, with no rescaling to a read total
(`deconvolution/omnideconv_bench/check_simbu_arms.R`). The sample's total divided by 1,000 is the mean library
size of that type's cells as the arm sums them. Arm A removes the library-size differences if this number is
the same for every cell type of a reference. The check runs on all 15 references, not only on the smoke
reference.

## Code changes (worktree branch `audit-remediation-v56`)

| Change | File | Earlier behaviour still reachable |
|---|---|---|
| Arm without mRNA bias: `--remove-bias-in-counts`, `--remove-bias-method`; the call is written to `sim_meta.txt` | `deconvolution/omnideconv_bench/simulate_simbu.R` | Yes. Without the arguments the call to SimBu is the one made before. On a test reference, arm B's mixtures equal the old "nobias" call's byte for byte and arm C's equal the old "bias" call's |
| Battery builder: arms A, B, C on the `references_v5` builds; arm check | `deconvolution/omnideconv_bench/build_simbu_battery.sh`, `check_simbu_arms.R` | Yes. `REFSET=refs4all` builds the 2026-11-11 battery (bias, nobias, dose-response) |
| DWLS method and both cut-offs passed explicitly and recorded | `deconvolution/R/run_omnideconv.R`, `omnideconv_settings.R` | Yes. Unset, they are `mast`, 0.01 and 0.5, the values every earlier run used |
| In-memory DWLS guard extended to the `mast_optimized` builder; each firing printed and recorded | same | The guard for `mast` has the same effect as before |
| Provenance record per run of the cross-check runner | same, with `deconvolution/R/provenance.R` | Adds a file; changes no output |
| Scoring message says a type a method does not estimate is left out of its scores; the counts go into `overall_metrics` | `deconvolution/score_validation.py` | Scores are unchanged; four rows are added to `overall_metrics_<method>.tsv` |
| Panel builder on the `references_v5` builds; check of a built panel before deconvolution | `deconvolution/build_mixture_panel.sh`, `check_panel_build.py` | Yes. `REFSET=refs4all` builds the 2026-11-04 panel |
| Summary of two named arms; the corrector rule as a function | `deconvolution/omnideconv_bench/simbu_summary.py` | Yes. The defaults are the arms of 2026-11-11 |

Tests added (`tests/`): `test_omnideconv_runner.py` (with `tests/r/test_omnideconv_settings.R`),
`test_check_panel_build.py`, `test_step4_v5_v6_tables.py`.

## Commands that build the tables and figures

Run from the repository root, on a compute node (`slurm/analysis/v56_tables.slurm final` runs the three).

```
python analysis/audit_remediation/make_step4_v5_tables.py     # -> docs/audit/remediation/record/step4/v5_simbu/
python analysis/audit_remediation/make_step4_v6_tables.py     # -> docs/audit/remediation/record/step4/v6_mixture_panel/
MPLCONFIGDIR=tmp/mpl python analysis/audit_remediation/make_step4_v5v6_figures.py   # -> docs/audit/remediation/figures/step4_v5_*.png, step4_v6_*.png
```

## Settings of the two batteries

Fixed in `slurm/analysis/v56_task.sh` (local job script; the settings also appear in each run's provenance
file) before any run. Every job runs the committed code of this branch with
`PIPELINE_PROFILE=config/profiles/refs5.yaml`.

| | V5, SimBu battery | V6, mixture panel |
|---|---|---|
| References | the 15 `references_v5` builds | the same |
| Sets | 45: 15 references x arms A, B, C | 19: 15 holdout, 4 cross |
| Mixtures | 50 of 1,000 cells, SimBu `random` composition, 1,000,000 reads, seed 1, from a seeded downsample to at most 500 cells per type | 50 of 1,000 cells, Dirichlet alpha 1, seed 1, 30% holdout, at least 20 pool cells per benchmarked type |
| Deconvolved against | the set's own downsampled reference, no further cap | holdout: the set's 70% split; cross: the production reference. Cap of 2,000 cells per type (seed 1) for every method but BayesPrism |
| Scored against | cell fractions (`--truth cellfrac`) | RNA fractions (`rnafrac`) |
| BayesPrism | Stage 8 runner, 8 cores | Stage 8 runner, 16 cores |
| DWLS | `mast`, p 0.01, fold change 0.5, 4 cores (note N3) | `mast_optimized`, p 0.05, fold change 0.5, 4 cores (the registered correction) |
| MuSiC, SCDC, Bisque | omnideconv defaults, sample of origin as batch | the same |
| CIBERSORTx | no batch correction; existing container and credentials | the same |
| Gene exclusion | `rat_exclude_genes_v2.tsv`, every method (note N4) | the same |
