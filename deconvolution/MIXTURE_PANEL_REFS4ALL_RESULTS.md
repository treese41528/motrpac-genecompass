# Known-truth mixture panel on the current references: result (2026-11-08)

**Question.** The 16-tissue default deconvolves against 15 references. Does each one recover the cell-type
fractions of mixtures whose composition is known? The earlier panel (`OMNIDECONV_RESULTS.md`,
`data/deconvolution/validation_v2/`) certified the June/July references; this reruns it on the current ones.

- Pre-registration: `MIXTURE_PANEL_REFS4ALL_PREREG.md` (780b149).
- Cross harmonization overrides: 37b9099, committed before any deconvolution.
- Record (summary tables): `mixture_panel_record/`. Data: `data/deconvolution/mixture_panel_refs4all/`.
- Jobs: build 12192378, 12192456; BayesPrism 12192436, 12192733; MuSiC, SCDC and Bisque 12192437, 12192734;
  DWLS 12192438, 12192735; CIBERSORTx 12291190.

**Verdict: every reference recovers its benchmarked types.**
- BayesPrism's median per-type r is 0.989–0.999 on the 15 holdout sets and 0.973–0.996 on the 4 cross sets,
  against the registered 0.85.
- No set is more than 0.05 below the earlier panel.

## Median per-type Pearson r (headline)

Truth is `rnafrac`; 50 mixtures per set. "–" means not applicable: MuSiC and Bisque need a reference with more
than one sample.

| Set | BayesPrism | CIBERSORTx | DWLS | MuSiC | SCDC | Bisque |
|---|---|---|---|---|---|---|
| BAT_holdout | **0.998** | 0.990 | 0.996 | 0.997 | 0.991 | 0.959 |
| BLOOD_holdout | **0.989** | 0.987 | 0.979 | 0.968 | 0.925 | 0.771 |
| CORTEX_holdout | **0.999** | 0.997 | 0.938 | 0.932 | 0.954 | 0.982 |
| HEART_holdout | **0.997** | 0.993 | 0.987 | 0.986 | 0.927 | 0.969 |
| HIPPOC_holdout | **0.998** | 0.993 | 0.980 | 0.972 | 0.970 | 0.965 |
| HYPOTH_holdout | **0.995** | 0.993 | 0.993 | – | 0.993 | – |
| KIDNEY_holdout | **0.995** | 0.993 | 0.990 | 0.982 | 0.979 | 0.951 |
| LIVER_holdout | **0.992** | 0.990 | 0.985 | 0.988 | 0.987 | 0.929 |
| LUNG_holdout | **0.991** | 0.990 | 0.985 | 0.979 | 0.962 | 0.671 |
| MUSCLE_holdout | **0.997** | 0.996 | 0.995 | 0.995 | 0.994 | 0.969 |
| OVARY_holdout | **0.995** | 0.993 | 0.966 | – | 0.946 | – |
| SMLINT_holdout | **0.992** | 0.984 | 0.966 | 0.965 | 0.950 | 0.874 |
| SPLEEN_holdout | 0.990 | **0.991** | 0.964 | – | 0.960 | – |
| TESTES_holdout | **0.999** | 0.996 | 0.983 | 0.982 | 0.977 | 0.983 |
| WATSC_holdout | **0.994** | 0.984 | 0.976 | 0.984 | 0.975 | 0.936 |
| CORTEX_cross | **0.983** | 0.973 | 0.913 | 0.926 | 0.965 | 0.961 |
| HIPPOC_cross | 0.996 | **0.997** ¹ | 0.951 | 0.975 ¹ | 0.985 ¹ | 0.961 |
| KIDNEY_cross | **0.973** | 0.944 | 0.708 | 0.588 | 0.737 | 0.768 |
| MUSCLE_cross | **0.986** | 0.944 | 0.934 | 0.962 | 0.965 | 0.901 |

¹ Over 5 of 6 types: the method estimates excitatory neurons as a constant 0 (see "What the medians leave out").

## Pooled Pearson r (no verdict)

| Set | BayesPrism | CIBERSORTx | DWLS | MuSiC | SCDC | Bisque |
|---|---|---|---|---|---|---|
| BAT_holdout | 0.957 | 0.895 | 0.958 | **0.970** | 0.624 | 0.061 |
| BLOOD_holdout | **0.981** | 0.941 | 0.832 | 0.807 | 0.369 | 0.465 |
| CORTEX_holdout | **0.999** | 0.997 | 0.631 | 0.466 | 0.786 | 0.604 |
| HEART_holdout | **0.973** | 0.932 | 0.853 | 0.789 | 0.516 | 0.578 |
| HIPPOC_holdout | **0.997** | 0.981 | 0.864 | 0.794 | 0.843 | 0.545 |
| HYPOTH_holdout | 0.964 | 0.896 | **0.970** | – | 0.944 | – |
| KIDNEY_holdout | **0.974** | 0.893 | 0.916 | 0.685 | 0.781 | 0.332 |
| LIVER_holdout | 0.841 | 0.808 | 0.896 | **0.899** | 0.859 | 0.262 |
| LUNG_holdout | 0.948 | **0.959** | 0.822 | 0.701 | 0.771 | 0.421 |
| MUSCLE_holdout | 0.958 | 0.874 | **0.979** | 0.977 | 0.962 | 0.212 |
| OVARY_holdout | **0.995** | 0.986 | 0.564 | – | 0.191 | – |
| SMLINT_holdout | **0.979** | 0.938 | 0.722 | 0.739 | 0.709 | 0.243 |
| SPLEEN_holdout | **0.987** | 0.950 | 0.781 | – | 0.794 | – |
| TESTES_holdout | **0.997** | 0.983 | 0.899 | 0.902 | 0.944 | 0.253 |
| WATSC_holdout | **0.953** | 0.915 | 0.845 | 0.855 | 0.846 | 0.199 |
| CORTEX_cross | **0.896** | 0.866 | 0.435 | 0.222 | 0.242 | 0.404 |
| HIPPOC_cross | **0.828** | 0.783 | 0.642 | 0.737 | 0.819 | 0.374 |
| KIDNEY_cross | 0.163 | **0.659** | 0.158 | 0.034 | 0.046 | 0.318 |
| MUSCLE_cross | **0.830** | 0.756 | 0.715 | 0.708 | 0.698 | 0.069 |

## Run notes and deviations

- **CIBERSORTx ran late, on 2026-11-08, after the other methods' results had been read.**
  - Its credentials were present at run time all along: the token the earlier panel used, on disk since
    2026-06-28. They were wrongly reported missing on 11-04 and the method was skipped.
  - It ran with the registered settings, including the same 2,000-cell signature cap. Only the job's CPU and
    memory request differ, and the method does not use them.
- **DWLS:** no task ran out of memory (peak 269 GiB of the 480 GiB requested), so the registered retry was not
  used.
- **MuSiC and Bisque:** not applicable on HYPOTH, OVARY and SPLEEN, whose references come from one sample. Bisque
  stops with "Only one individual detected"; MuSiC fails on the same three.
- **Signature cap.** The omnideconv methods' signature reference was capped at 2,000 cells per type (seeded), and
  every run logs the cells dropped per type. BayesPrism used the full references. HYPOTH, OVARY and TESTES needed
  no downsampling.
- **No leakage.** Every holdout set was deconvolved against its own 70% split. In all 15, the built reference's
  cells equal that split plus the mixture pool plus the cells `clean_cells` dropped.
- **Summary script edited after results.** `mixture_panel_summary.py` gained descriptive outputs only: the
  `comparable` flag, the scoring notes, coverage, and the pooled-best count. No threshold, set, source,
  harmonization or method setting changed.

## What the medians leave out

`score_validation.py` (as in the earlier panel) takes the median over types with a defined r
(`panel_scoring_notes.tsv`):
- MuSiC drops types that occur in only one sample: BLOOD erythroid cells and neutrophils, CORTEX medium spiny
  neurons, HEART CD8+ T cells. Its medians there are over the remaining types.
- MuSiC, SCDC and CIBERSORTx estimate HIPPOC_cross excitatory neurons as a constant 0 (true mean 0.17), so their
  HIPPOC_cross medians are over 5 of 6 types. BayesPrism keeps the type (r 0.74), though it estimates 0.008 for
  a true 0.172.

Both inflate the other methods' medians, so these exclusions understate BayesPrism's lead rather than overstate it.

## Per-type results

180 of BayesPrism's 193 per-type r are ≥ 0.95 and 186 are ≥ 0.90. Six are below 0.85:

| Set | Type | r | True mean | Estimated mean | Other methods |
|---|---|---|---|---|---|
| KIDNEY_cross | Monocytes | 0.58 | 0.118 | 0.572 | 0.49–0.65 |
| KIDNEY_cross | Distal convoluted tubule cells | 0.69 | 0.175 | 0.003 | 0.51–0.77 |
| KIDNEY_cross | Immune cells | 0.70 | 0.130 | 0.069 | −0.15–0.71 |
| HIPPOC_cross | Excitatory neurons | 0.74 | 0.172 | 0.008 | zero ²; Bisque 0.04 |
| LIVER_holdout | Hepatocytes | 0.79 | 0.134 | 0.231 | 0.80–0.93 |
| OVARY_holdout | Erythroid cells | 0.41 | 0.011 | 0.010 | 0.16–0.55 |

² MuSiC, SCDC and CIBERSORTx estimate exactly 0. DWLS estimates about 1e-20, so its r of 0.17 is floating-point noise.

- **KIDNEY_cross monocyte sink.** BayesPrism, DWLS, MuSiC and SCDC each put more than half the mass into
  monocytes (0.53–0.78, against a true 0.12). This is why BayesPrism's pooled r is 0.163 (0.057 in the earlier
  panel) while its median is 0.973. CIBERSORTx puts 0.16 there and is the only method with a usable pooled r
  (0.659).
- **Cross-set failures are shared.** Every method fails on the same KIDNEY_cross and HIPPOC_cross types. That
  points to the mixture source not matching those reference types, not to one method.
- **LIVER hepatocytes:** 0.993 in the earlier panel, which used a 5-type labelling of the same study; now
  overestimated. DWLS, MuSiC, SCDC and Bisque reach 0.90–0.93, and LIVER is one of BayesPrism's weaker pooled
  sets (0.841).
- **OVARY erythroid cells:** carry about 1% of the RNA in these mixtures.

**Hotspot types.**
- All nine hotspots of the default are benchmarked in their holdout sets, at BayesPrism r 0.983–0.999:
  - BLOOD classical monocytes 0.997, erythroid cells 0.989, megakaryocytes and platelets 0.989, neutrophils 0.995;
  - HYPOTH OPC 0.999; LUNG vascular smooth muscle 0.983;
  - muscle satellite cells 0.996, tenocytes 0.999, vascular smooth muscle 0.998.
- None is benchmarked in a cross set. BLOOD, LUNG and HYPOTH have no cross source, and the muscle cross set
  benchmarks only endothelial cells, macrophages, Schwann cells and skeletal muscle cells. Their evidence is
  within-study only.

## Coverage: what the panel certifies

170 of the 188 types in the 15 references are benchmarked in holdout (`panel_coverage.tsv`). The other 18:
- **Lost at the 70/30 split** (too few cells left for the 70% reference, so absent from both reference and
  mixtures): CORTEX vascular leptomeningeal cells, TESTES Sertoli cells and WATSC mural cells. The TESTES Sertoli
  overestimate seen in production is therefore not tested here.
- **Fewer than 20 pool cells:**
  - CORTEX border-associated macrophages, neutrophils and T cells;
  - HIPPOC mural cells;
  - HYPOTH dopaminergic neurons, endothelial cells and ependymal cells;
  - LIVER mesothelial cells; MUSCLE Schwann cells;
  - OVARY lymphatic endothelial cells and plasmacytoid dendritic cells;
  - SPLEEN endothelial cells;
  - WATSC glandular epithelial cells, mammary luminal epithelial cells and plasmacytoid dendritic cells.
- **Cross sets** benchmark 6 (CORTEX), 6 (HIPPOC), 7 (KIDNEY) and 4 (MUSCLE) types.

## Against the earlier panel

Eight pairs are comparable: the same design and, for cross sets, the same mixture source (identical
source-label sets).

| Set | Earlier set | Earlier | Now | Change |
|---|---|---|---|---|
| BLOOD_holdout | PBMC_holdout | 0.970 | 0.989 | +0.019 |
| HEART_holdout | HRT_holdout | 0.950 | 0.997 | +0.048 |
| LIVER_holdout | LIV_holdout | 0.992 | 0.992 | 0.000 |
| WATSC_holdout | WAT_holdout | 0.972 | 0.994 | +0.023 |
| CORTEX_cross | CTX_union | 0.940 | 0.983 | +0.043 |
| HIPPOC_cross | HIP_merged | 0.994 | 0.996 | +0.002 |
| KIDNEY_cross | KID_cross | 0.852 | 0.973 | +0.122 |
| MUSCLE_cross | SKMVL | 0.996 | 0.986 | −0.010 |

The other rows of `panel_vs_earlier.tsv` set a current holdout set against an earlier cross set (LUNG 0.705 to
0.991, for example). They do not measure a change in the reference.

## Other methods (descriptive)

- **Median per-type r:**
  - BayesPrism is best on 17 of 19 sets and within 0.02 of the best on all 19.
  - CIBERSORTx is the closest (0.944–0.997, within 0.02 of BayesPrism on 17 of 19). It is best on SPLEEN_holdout
    (0.991 against 0.990) and HIPPOC_cross (0.997 against 0.996, over 5 types).
  - Within 0.02 of BayesPrism: DWLS on 11 of 19 sets, MuSiC 8 of 16, SCDC 8 of 19, Bisque 2 of 16.
- **Pooled r:**
  - BayesPrism is best on 13 of 19.
  - MuSiC leads on BAT and LIVER, DWLS on HYPOTH and MUSCLE_holdout, and CIBERSORTx on LUNG_holdout and
    KIDNEY_cross. On KIDNEY_cross, CIBERSORTx reaches 0.659 against BayesPrism's 0.163.
- **Bisque** is the weakest method on most sets (LUNG 0.671, BLOOD 0.771).

## Reading

- **Differential use holds.** The pipeline uses θ differentially (a type's fraction across samples). For that,
  every current reference recovers its benchmarked types, within study and across study.
- **Absolute fractions are less secure.**
  - Across study, kidney fractions are dominated by a monocyte sink for every method except CIBERSORTx and
    Bisque (BayesPrism pooled 0.163), as in the earlier panel.
  - Within study, liver hepatocytes are overestimated.
- **Limits of the certificate.**
  - It covers the benchmarked types only: 170 of 188 in holdout, plus 23 type tests in the cross sets.
  - The hotspot types are covered within study only.
  - Holdout mixtures share a study with their reference, so they are the easier test.
