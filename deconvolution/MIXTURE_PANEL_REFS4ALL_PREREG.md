# Known-truth mixture panel on the current production references (pre-registered 2026-11-04, before any mixture was built)

**Why.** The Dirichlet-mixture validation and the omnideconv cross-method panel (`OMNIDECONV_RESULTS.md`,
`data/deconvolution/validation_v2/`) certified the June/July references. The default now deconvolves against
other references: the eight jointly re-annotated ones (`references_v4/`, `JOINT_REANNOTATION_RESULTS.md`), the
BLOOD composite, the TESTES gene-ID lift, and the new SPLEEN and OVARY. The panel has not been run on any of
these. Stage 14's purity sweeps already cover 11 of them, but they ask a different question (one focal type's
expression across purity); this panel asks whether every type's *fraction* is recovered.

**References.** The `reference_dir` of every runnable tissue in `deconvolution/tissue_references.yaml`, which is
what Stage 8 of the default deconvolves against: 15 distinct references for the 16 tissues (SKMGN and SKMVL share
`references_v4/MUSCLE_GSE137869_Y`).

**Mixture sets.** Output under `data/deconvolution/mixture_panel_refs4all/<SET>/`.

1. **Holdout, all 15 references (primary).** `make_pseudobulk.py --mode holdout --ref-source <reference_dir>`.
   - The built reference's cells, with their labels as built, are split by cell type: 30% go to the mixture
     pool, 70% form the reference these mixtures are deconvolved against. The two sets share no cells.
   - The 70% reference is re-cleaned by `clean_cells`.
   - Sets: `<CODE>_holdout` for BAT, BLOOD, CORTEX, HEART, HIPPOC, HYPOTH, KIDNEY, LIVER, LUNG, MUSCLE, OVARY,
     SMLINT, SPLEEN, TESTES, WATSC.
2. **Cross-study (secondary).** `make_pseudobulk.py --mode cross --ref-dir <reference_dir>`: mixtures are
   drawn from a different study and deconvolved against the full production reference. The sources are the ones
   the earlier panel used, wherever that source is still a different study from the current reference:
   - `KIDNEY_cross`: GSE137869, kidney.
   - `CORTEX_cross`: GSE213978, cerebral cortex, sham.
   - `HIPPOC_cross`: GSE307917, hippocampal dentate gyrus, naïve and sham.
   - `MUSCLE_cross`: GSE255196, skeletal muscle.

   Each source is canonicalized with its reference's `label_scheme` (brain, muscle, none).

   Not included:
   - **GAS_cross:** its source, GSE137869, is now the muscle reference's own study.
   - **LNG_cross:** its source, GSE196313, is embryonic, and the pooled lung reference already uses the three
     adult native-lung studies.

   No new cross source is chosen.

**Fixed settings.** 50 mixtures per set, 1,000 cells each, Dirichlet α = 1, `--seed 1`, `--holdout-frac 0.30`,
`--min-pool-cells 20`; a reference type with fewer than 20 pool cells is not benchmarked and is listed.

**Cross harmonization.**
- `make_pseudobulk.py` maps source labels to reference types automatically. Each `harmonization_map.tsv` is
  reviewed on names alone before any cross set is deconvolved.
- A source label is set to unmapped if its automatic target is a different cell class. Labels are never moved to
  a new target by hand.
- Such changes go in `deconvolution/reference/mixture_panel_harmonization_overrides.tsv`, are committed, and the
  mixtures are rebuilt, all before deconvolution. No change after any result is seen.

**Methods.**
- **BayesPrism:** `R/run_deconvolution.sh` as in Stage 8, 16 cores, on the full reference.
- **omnideconv:** `R/run_omnideconv.sh` with DWLS (`mast`), MuSiC, SCDC and Bisque. Its signature reference is
  capped at 2,000 cells per type (`REF_MAX_CELLS_PER_TYPE`, seeded, dropped cells logged), as in the earlier
  panel; BayesPrism gets no cap.
- **CIBERSORTx:** run only if its credentials are present at run time; otherwise reported as not run.
- **Failures:** a method that fails on a set is reported as failed. The one allowed retry is DWLS out of memory,
  rerun at 2 cores, the earlier panel's fix. Bisque and MuSiC need a reference with more than one sample; where
  a reference has one, they are reported as not applicable.

**Scoring.** `score_validation.py` against `rnafrac` (the RNA fraction, what θ estimates) for every method, as in
the earlier panel. Per set and method:
- the median per-type Pearson r across the 50 mixtures (headline; the differential question);
- the pooled r;
- the per-type r of every benchmarked type.

**Reading (BayesPrism, per set).**
- Median per-type r ≥ 0.85: the reference recovers its benchmarked types' fractions. Below 0.85: named as weak,
  with its worst types.
- Pooled r is reported without a verdict.
- Each set is compared, descriptively, with the earlier panel's number for the same production tissue. A drop of
  more than 0.05 in median per-type r is named.

The other methods are descriptive: how many sets BayesPrism is best or within 0.02 of best, and where the methods
disagree. No threshold, set, source, harmonization or method setting will be changed after results are seen.

**Caveat carried over.** The panel certifies only the benchmarked types (those in the mixture pool). For a
holdout set that is all the reference's types with at least 20 pool cells. Holdout mixtures share a study with
their reference, so they are the easier test; cross sets are the harder one.
