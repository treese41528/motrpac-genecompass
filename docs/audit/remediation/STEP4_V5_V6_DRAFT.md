# Step 4, V5 and V6: draft

Draft for `STEP4_PREREG.md` sections V5 (SimBu mRNA-content bias, finding 5.2) and V6 (mixture panel, findings
5.5 and 5.6). Both batteries are finished: every registered run was made, none failed and none was retried.

- **Results** come first. They are written from the record tables by one command (see "Commands").
- **Then** what the results mean and do not show, the notes that settled open points before any result
  existed, the settings, the smoke run, the jobs and the code changes.

<!-- RESULTS:BEGIN -->

*This part is written by `analysis/audit_remediation/make_step4_v5v6_report.py` from the record tables (last run 2026-11-15). Paths are relative to `docs/audit/remediation/`.*

## V5 results: mRNA-content bias against a baseline that has no bias

**Result.** Against a baseline with no mRNA bias, BayesPrism's error against cell fractions is higher under the references' native bias on 13 of 15 references and under the added bias on 14 of 15 (mean pooled RMSE over the 15 references: 0.0187 without bias, 0.0460 with the native bias, 0.0720 with the added bias). On BAT and BLOOD it is lower with the native bias than without. The registered corrector rule on arm C minus arm A is **reproduced** over the 12 references every method ran, but not because MuSiC withstands the added bias: MuSiC and DWLS are closest to cell fractions in arm B and furthest in arm A, so their arm C minus arm A difference is negative. From arm B to arm C, the 2026-11-11 reading, the rule is **not reproduced**: MuSiC's error rises by 0.0260 and DWLS's by 0.0284, against 0.0252 for BayesPrism.

### What ran

- 15 references x 3 arms x 6 methods = 270 method runs: 252 scored, 18 not applicable, 0 failed, 0 not run (`record/step4/v5_simbu/v5_method_status.tsv`).
  - Not applicable: Bisque on HYPOTH (arms A, B, C); MuSiC on HYPOTH (arms A, B, C); Bisque on OVARY (arms A, B, C); MuSiC on OVARY (arms A, B, C); Bisque on SPLEEN (arms A, B, C); MuSiC on SPLEEN (arms A, B, C).
    These references hold one sample; MuSiC and Bisque need at least two (the registered case).
- The three arms share one downsampled reference on 15 of 15 references and one cell-fraction truth on 15 of 15; the simulate call recorded in `sim_meta.txt` is the registered one for arm A on 15, arm B on 15 and arm C on 15 of 15 (`record/step4/v5_simbu/v5_status.tsv`).
- On the 13 references that did not change since 2026-11-11, arm B's mixtures are byte-identical to that battery's "nobias" mixtures on 13 and arm C's to its "bias" mixtures on 13 (same table).

### Check: does arm A remove the library-size differences?

Yes. Through SimBu's own summation, the mean library size of 1,000 cells of one type, largest type over smallest type within a reference: arm A 1.0000 to 1.0000; arm B 1.0794 to 28.6686; arm C 1.2006 to 283.2293, over 15 references (`record/step4/v5_simbu/v5_arm_library_size_summary.tsv`; per type in `v5_arm_library_size.tsv`; figure `figures/step4_v5_arm_check.png`). In arm A every cell contributes a total of 1.

### Registered readings

**1. BayesPrism under native bias (RMSE in arm B minus arm A).** Positive on **13 of 15** references; mean +0.0273, range -0.0228 to +0.0934 (`record/step4/v5_simbu/v5_bayesprism.tsv`).

**2. BayesPrism under added bias (arm C minus arm A).** Positive on **14 of 15** references; mean +0.0533, range -0.0162 to +0.1391 (same table; figure `figures/step4_v5_bayesprism_arms.png`).

| Reference | RMSE A | RMSE B | RMSE C | B - A | C - A | C - B |
|---|---|---|---|---|---|---|
| BAT | 0.0373 | 0.0145 | 0.0210 | -0.0228 | -0.0162 | +0.0065 |
| BLOOD | 0.0487 | 0.0467 | 0.0721 | -0.0019 | +0.0234 | +0.0253 |
| CORTEX | 0.0124 | 0.1058 | 0.1515 | +0.0934 | +0.1391 | +0.0457 |
| HEART | 0.0061 | 0.0377 | 0.0755 | +0.0316 | +0.0694 | +0.0378 |
| HIPPOC | 0.0098 | 0.0582 | 0.0856 | +0.0484 | +0.0758 | +0.0274 |
| HYPOTH | 0.0157 | 0.0172 | 0.0395 | +0.0015 | +0.0238 | +0.0222 |
| KIDNEY | 0.0149 | 0.0255 | 0.0423 | +0.0106 | +0.0273 | +0.0167 |
| LIVER | 0.0216 | 0.0555 | 0.0977 | +0.0340 | +0.0762 | +0.0422 |
| LUNG | 0.0147 | 0.0247 | 0.0378 | +0.0100 | +0.0231 | +0.0131 |
| MUSCLE | 0.0100 | 0.0209 | 0.0463 | +0.0109 | +0.0363 | +0.0254 |
| OVARY | 0.0098 | 0.0602 | 0.0930 | +0.0504 | +0.0832 | +0.0329 |
| SMLINT | 0.0275 | 0.0596 | 0.0784 | +0.0321 | +0.0510 | +0.0188 |
| SPLEEN | 0.0210 | 0.0467 | 0.0798 | +0.0258 | +0.0588 | +0.0330 |
| TESTES | 0.0181 | 0.0770 | 0.1028 | +0.0589 | +0.0846 | +0.0257 |
| WATSC | 0.0134 | 0.0395 | 0.0571 | +0.0262 | +0.0437 | +0.0176 |

**3. Correctors: the registered rule on arm C minus arm A.**

Common set (all three arms): 12 references (BAT; BLOOD; CORTEX; HEART; HIPPOC; KIDNEY; LIVER; LUNG; MUSCLE; SMLINT; TESTES; WATSC).

| Method | Mean difference, common set | Mean, all it scored | References scored | of which positive |
|---|---|---|---|---|
| DWLS | -0.0126 | -0.0123 | 15 | 4 |
| MuSiC | -0.0070 | -0.0070 | 12 | 7 |
| Bisque | +0.0049 | +0.0049 | 12 | 12 |
| SCDC | +0.0181 | +0.0179 | 15 | 12 |
| BayesPrism | +0.0528 | +0.0533 | 15 | 14 |
| CIBERSORTx | +0.0618 | +0.0624 | 15 | 14 |

Rule: reproduced if MuSiC and SCDC are both below BayesPrism and CIBERSORTx. MuSiC -0.0070, SCDC +0.0181; BayesPrism +0.0528, CIBERSORTx +0.0618. **Reproduced.** Dropping one reference at a time leaves the verdict unchanged in 12 of 12 cases (descriptive). Tables: `record/step4/v5_simbu/v5_corrector_rule.tsv`, `v5_ranking.tsv`.

**4. The 2026-11-11 readings (arm C minus arm B) on the refs5 references.**

Common set (all three arms): 12 references (BAT; BLOOD; CORTEX; HEART; HIPPOC; KIDNEY; LIVER; LUNG; MUSCLE; SMLINT; TESTES; WATSC).

| Method | Mean difference, common set | Mean, all it scored | References scored | of which positive |
|---|---|---|---|---|
| Bisque | +0.0024 | +0.0024 | 12 | 12 |
| SCDC | +0.0216 | +0.0215 | 15 | 14 |
| CIBERSORTx | +0.0221 | +0.0230 | 15 | 15 |
| BayesPrism | +0.0252 | +0.0260 | 15 | 15 |
| MuSiC | +0.0260 | +0.0260 | 12 | 12 |
| DWLS | +0.0284 | +0.0283 | 15 | 15 |

Rule: reproduced if MuSiC and SCDC are both below BayesPrism and CIBERSORTx. MuSiC +0.0260, SCDC +0.0216; BayesPrism +0.0252, CIBERSORTx +0.0221. **Not reproduced.** Dropping one reference at a time leaves the verdict unchanged in 12 of 12 cases (descriptive). Tables: `record/step4/v5_simbu/v5_corrector_rule.tsv`, `v5_ranking.tsv`.

- BayesPrism's RMSE rises from arm B to arm C on 15 of 15 references (2026-11-11: 15 of 15), by +0.0065 to +0.0457.
- BayesPrism's driver (the type with the largest per-type rise) is the type with the most expressed genes on 10 of 15 references (2026-11-11: 10 of 15); by total counts on 9 of 15 (`record/step4/v5_simbu/v5_driver.tsv`).

| Reference | Driver (C - B) | mRNA rank (genes; counts) | True | Estimate B -> C | Direction |
|---|---|---|---|---|---|
| BAT | Brown adipocytes | 6 of 6; 6 | 0.185 | 0.168 -> 0.215 | inflated |
| BLOOD | Plasmacytoid dendritic cells | 1 of 13; 1 | 0.084 | 0.174 -> 0.230 | inflated |
| CORTEX | Excitatory neurons | 1 of 12; 1 | 0.092 | 0.354 -> 0.493 | inflated |
| HEART | Cardiomyocytes | 1 of 15; 1 | 0.064 | 0.178 -> 0.303 | inflated |
| HIPPOC | Dentate gyrus granule cells | 1 of 11; 1 | 0.111 | 0.218 -> 0.284 | inflated |
| HYPOTH | Glutamatergic neurons | 1 of 11; 1 | 0.107 | 0.132 -> 0.185 | inflated |
| KIDNEY | Proximal tubule cells | 3 of 15; 3 | 0.063 | 0.098 -> 0.149 | inflated |
| LIVER | NK cells | 1 of 8; 1 | 0.142 | 0.268 -> 0.351 | inflated |
| LUNG | Alveolar macrophages | 5 of 21; 3 | 0.045 | 0.074 -> 0.129 | inflated |
| MUSCLE | Skeletal muscle cells | 1 of 9; 1 | 0.113 | 0.159 -> 0.224 | inflated |
| OVARY | Luteal cells | 1 of 18; 1 | 0.059 | 0.250 -> 0.373 | inflated |
| SMLINT | Transit-amplifying cells | 3 of 13; 2 | 0.087 | 0.161 -> 0.213 | inflated |
| SPLEEN | Erythroid progenitors | 1 of 14; 1 | 0.066 | 0.130 -> 0.212 | inflated |
| TESTES | Condensed spermatids | 6 of 6; 4 | 0.179 | 0.133 -> 0.088 | deflated |
| WATSC | Cycling cells | 1 of 13; 2 | 0.080 | 0.130 -> 0.161 | inflated |

- Against the battery of 2026-11-11 (BayesPrism, arm C minus arm B here, bias minus nobias then): the largest absolute change is 0.0074 over 15 references, and 0.0005 over the 13 whose reference did not change (only the exclusion list differs there) (`record/step4/v5_simbu/v5_vs_20261111.tsv`).

### Each method in the three arms (descriptive, not a registered reading)

Mean pooled RMSE against cell fractions over the 12 references every method ran, and the number of those references on which each arm is the method's lowest (`record/step4/v5_simbu/v5_method_arm_rmse.tsv`; figure `figures/step4_v5_methods_arms.png`).

| Method | Arm A | Arm B | Arm C | Lowest in A | Lowest in B | Lowest in C |
|---|---|---|---|---|---|---|
| BayesPrism | 0.0195 | 0.0471 | 0.0723 | 10 | 2 | 0 |
| CIBERSORTx | 0.0255 | 0.0652 | 0.0873 | 11 | 1 | 0 |
| DWLS | 0.0505 | 0.0095 | 0.0379 | 0 | 12 | 0 |
| MuSiC | 0.0531 | 0.0202 | 0.0462 | 1 | 11 | 0 |
| SCDC | 0.0571 | 0.0536 | 0.0752 | 7 | 4 | 1 |
| Bisque | 0.0646 | 0.0671 | 0.0695 | 12 | 0 | 0 |

DWLS and MuSiC are closest to cell fractions in arm B, not in arm A, on most references: for such a method arm A is not the easy case, and its arm C minus arm A difference is small or negative because its arm A error is high, not because it corrects the added bias. BayesPrism, CIBERSORTx, SCDC and Bisque are closest in arm A.

## V6 results: mixture panel on the refs5 references

**Result.** The registered floor is met on every set scored: BayesPrism's median per-type r is 0.990 to 0.999 on the 15 holdout sets; 0.976 to 0.996 on the 4 cross sets, against 0.85 (19 of 19).

### Pass rule (registered, unchanged): BayesPrism's median per-type r of at least 0.85

| Set | BayesPrism median r | Verdict | BayesPrism pooled r | Types in the median | Types below 0.85 | CIBERSORTx | DWLS | MuSiC | SCDC | Bisque |
|---|---|---|---|---|---|---|---|---|---|---|
| BAT_holdout | 0.998 | recovers | 0.958 | 6 of 6 | 0 | 0.990 | 0.996 | 0.997 | 0.991 | 0.959 |
| BLOOD_holdout | 0.990 | recovers | 0.980 | 13 of 13 | 0 | 0.986 | 0.980 | 0.972 | 0.915 | 0.765 |
| CORTEX_holdout | 0.999 | recovers | 0.999 | 8 of 8 | 0 | 0.997 | 0.938 | 0.932 | 0.953 | 0.982 |
| HEART_holdout | 0.994 | recovers | 0.974 | 15 of 15 | 0 | 0.991 | 0.982 | 0.981 | 0.943 | 0.957 |
| HIPPOC_holdout | 0.998 | recovers | 0.997 | 10 of 10 | 0 | 0.993 | 0.980 | 0.972 | 0.970 | 0.965 |
| HYPOTH_holdout | 0.995 | recovers | 0.926 | 10 of 10 | 0 | 0.979 | 0.993 | - | 0.987 | - |
| KIDNEY_holdout | 0.995 | recovers | 0.974 | 15 of 15 | 0 | 0.993 | 0.990 | 0.982 | 0.979 | 0.951 |
| LIVER_holdout | 0.992 | recovers | 0.844 | 7 of 7 | 1 | 0.990 | 0.986 | 0.988 | 0.987 | 0.928 |
| LUNG_holdout | 0.991 | recovers | 0.947 | 21 of 21 | 0 | 0.990 | 0.985 | 0.979 | 0.959 | 0.671 |
| MUSCLE_holdout | 0.997 | recovers | 0.958 | 8 of 8 | 0 | 0.996 | 0.996 | 0.995 | 0.994 | 0.969 |
| OVARY_holdout | 0.996 | recovers | 0.995 | 16 of 16 | 1 | 0.993 | 0.966 | - | 0.950 | - |
| SMLINT_holdout | 0.992 | recovers | 0.979 | 13 of 13 | 0 | 0.984 | 0.966 | 0.964 | 0.951 | 0.890 |
| SPLEEN_holdout | 0.990 | recovers | 0.987 | 13 of 13 | 0 | 0.991 | 0.964 | - | 0.960 | - |
| TESTES_holdout | 0.999 | recovers | 0.996 | 5 of 5 | 0 | 0.996 | 0.983 | 0.982 | 0.977 | 0.983 |
| WATSC_holdout | 0.994 | recovers | 0.950 | 9 of 9 | 0 | 0.983 | 0.977 | 0.985 | 0.974 | 0.920 |
| CORTEX_cross | 0.983 | recovers | 0.897 | 6 of 6 | 0 | 0.973 | 0.922 | 0.927 | 0.964 | 0.960 |
| HIPPOC_cross | 0.996 | recovers | 0.826 | 6 of 6 | 1 | 0.997 | 0.951 | 0.975 | 0.985 | 0.961 |
| KIDNEY_cross | 0.976 | recovers | 0.174 | 7 of 7 | 3 | 0.945 | 0.706 | 0.589 | 0.736 | 0.767 |
| MUSCLE_cross | 0.986 | recovers | 0.832 | 4 of 4 | 0 | 0.945 | 0.934 | 0.962 | 0.965 | 0.900 |

Median per-type r of the other methods is descriptive; "-" is not scored (see the status below). Tables: `record/step4/v6_mixture_panel/v6_floor.tsv`, `panel_median.tsv`, `panel_pooled.tsv`, `panel_per_type.tsv`; figure `figures/step4_v6_median_r.png`.

Per type, BayesPrism has 192 type tests over the 19 sets: 181 at r 0.95 or above, 185 at 0.90 or above and 6 below 0.85 (`record/step4/v6_mixture_panel/v6_floor.tsv`, `panel_per_type.tsv`).

Types below 0.85 for BayesPrism (a set can meet the floor on its median and still have such types): LIVER_holdout: Hepatocytes 0.789; NK cells 0.978; Liver sinusoidal endothelial cells 0.981; OVARY_holdout: Erythroid cells 0.397; Stromal cells 0.976; B cells 0.985; HIPPOC_cross: Excitatory neurons 0.722; Microglia 0.995; Oligodendrocytes 0.995; KIDNEY_cross: Monocytes 0.587; Immune cells 0.698; Distal convoluted tubule cells 0.703. The list gives each set's three lowest types.

### Against the panel of 2026-11-08 (descriptive, as registered)

BayesPrism's median per-type r, 19 sets compared: a drop of more than 0.05 on **0**; largest drop -0.0035 (HEART_holdout), largest rise +0.0022 (KIDNEY_cross) (`record/step4/v6_mixture_panel/v6_vs_20261108.tsv`).
The mixtures themselves are byte-identical to that panel's on 17 of 19 sets; they differ on HEART_holdout and HYPOTH_holdout. The cross harmonization equals the reviewed maps on 4 of 4 cross sets (`record/step4/v6_mixture_panel/v6_build_check.tsv`). Where the mixtures are identical, a change in r comes from the exclusion list alone.

### What the panel covers, and the other methods (descriptive)

- **Coverage.** 169 of the 185 cell types of the 15 references are benchmarked in holdout. 3 are lost at the 70/30 split (CORTEX Vascular leptomeningeal cells; TESTES Sertoli cells; WATSC Mural cells) and 13 have fewer than 20 pool cells. The cross sets benchmark 6 (CORTEX), 6 (HIPPOC), 7 (KIDNEY), 4 (MUSCLE) types (`record/step4/v6_mixture_panel/panel_coverage.tsv`).
- **Against the other methods.** On median per-type r BayesPrism is the best method on 17 of 19 sets and within 0.02 of the best on 19; on pooled r it is the best on 13 of 19 (`record/step4/v6_mixture_panel/panel_median.tsv`, `panel_pooled.tsv`, three decimals).
- **Types left out of a method's median** (the scoring leaves a type out of a method's scores when the method gives no estimate for it, and out of its median when the estimate is a constant): Excitatory neurons in HIPPOC_cross for MuSiC, SCDC and CIBERSORTx (constant estimate); Erythroid cells in BLOOD_holdout for MuSiC (no estimate); Neutrophils in BLOOD_holdout for MuSiC (no estimate); Medium spiny neurons in CORTEX_holdout for MuSiC (no estimate) (`record/step4/v6_mixture_panel/panel_scoring_notes.tsv`). These exclusions favour the other methods, not BayesPrism.

### HEART and HYPOTH, every type (their labels changed)

- **HEART_holdout**: 15 types scored, r 0.894 to 0.999; below 0.85: none. New since 2026-11-08: none. Benchmarked then, not now: CD8+ T cells.
- **HYPOTH_holdout**: 10 types scored, r 0.955 to 0.997; below 0.85: none. New since 2026-11-08: Ependymal cells, Vascular leptomeningeal cells, Pars tuberalis cells. Benchmarked then, not now: GABAergic neurons, Neuroendocrine cells, Radial glial cells.
  Table: `record/step4/v6_mixture_panel/v6_heart_hypoth_types.tsv`; figure `figures/step4_v6_heart_hypoth.png`.

| Set | Type | r now | r 2026-11-08 | True mean now | Estimated mean now |
|---|---|---|---|---|---|
| HEART_holdout | Dendritic cells | 0.894 | 0.900 | 0.056 | 0.062 |
| HEART_holdout | Endothelial cells | 0.984 | 0.982 | 0.051 | 0.064 |
| HEART_holdout | Cardiac fibroblasts | 0.984 | 0.991 | 0.063 | 0.068 |
| HEART_holdout | T cells | 0.988 | 0.971 | 0.047 | 0.057 |
| HEART_holdout | Pericytes | 0.993 | 0.997 | 0.051 | 0.057 |
| HEART_holdout | Macrophages | 0.993 | 0.986 | 0.060 | 0.068 |
| HEART_holdout | Mast cells | 0.994 | 0.994 | 0.081 | 0.049 |
| HEART_holdout | Monocytes | 0.994 | 0.988 | 0.046 | 0.045 |
| HEART_holdout | NK cells | 0.996 | 0.999 | 0.053 | 0.048 |
| HEART_holdout | Cardiac neurons | 0.996 | 0.999 | 0.056 | 0.054 |
| HEART_holdout | Vascular smooth muscle cells | 0.997 | 0.997 | 0.062 | 0.065 |
| HEART_holdout | Lymphatic endothelial cells | 0.997 | 0.998 | 0.061 | 0.059 |
| HEART_holdout | B cells | 0.998 | 0.999 | 0.043 | 0.043 |
| HEART_holdout | Mesothelial cells | 0.998 | 0.999 | 0.080 | 0.065 |
| HEART_holdout | Cardiomyocytes | 0.999 | 0.999 | 0.189 | 0.195 |
| HEART_holdout | CD8+ T cells | n/a | 0.998 | n/a | n/a |
| HYPOTH_holdout | Ependymal cells | 0.955 | n/a | 0.110 | 0.068 |
| HYPOTH_holdout | Neurons | 0.957 | 0.978 | 0.109 | 0.161 |
| HYPOTH_holdout | Tanycytes | 0.982 | 0.997 | 0.111 | 0.118 |
| HYPOTH_holdout | Glutamatergic neurons | 0.991 | 0.850 | 0.119 | 0.139 |
| HYPOTH_holdout | Oligodendrocyte precursor cells | 0.995 | 0.999 | 0.110 | 0.084 |
| HYPOTH_holdout | Microglia | 0.995 | 0.998 | 0.108 | 0.088 |
| HYPOTH_holdout | Pars tuberalis cells | 0.996 | n/a | 0.075 | 0.081 |
| HYPOTH_holdout | Vascular leptomeningeal cells | 0.996 | n/a | 0.089 | 0.081 |
| HYPOTH_holdout | Oligodendrocytes | 0.997 | 0.996 | 0.067 | 0.063 |
| HYPOTH_holdout | Astrocytes | 0.997 | 0.995 | 0.103 | 0.110 |
| HYPOTH_holdout | GABAergic neurons | n/a | 0.940 | n/a | n/a |
| HYPOTH_holdout | Neuroendocrine cells | n/a | 0.994 | n/a | n/a |
| HYPOTH_holdout | Radial glial cells | n/a | 0.996 | n/a | n/a |

### DWLS as the benchmark plan specified

`mast_optimized` at p 0.05 is scored on 19 of 19 sets: median per-type r 0.706 to 0.996. Against `mast` at p 0.01 in the 2026-11-08 panel the change is -0.006 to +0.009 over 19 sets (`record/step4/v6_mixture_panel/v6_dwls_setting.tsv`). The comparison also carries the change of exclusion list and, for HEART and HYPOTH, of labels.
The 19 runs took 319 minutes in all (3.9 to 57.9 per set; same table, column `minutes`).
Setting read back from the provenance files: mast_optimized, p 0.05, fold change 0.5 (19 sets).
The in-memory guard fired on 2 sets: BAT_holdout (Brown adipocytes); OVARY_holdout (Erythroid cells). Those types have no marker genes of their own in the DWLS signature.

### Status of every method run

scored 108, not applicable 6 of 114 (19 sets x 6 methods; `record/step4/v6_mixture_panel/v6_status.tsv`).
- Not applicable: MuSiC on HYPOTH_holdout; Bisque on HYPOTH_holdout; MuSiC on OVARY_holdout; Bisque on OVARY_holdout; MuSiC on SPLEEN_holdout; Bisque on SPLEEN_holdout.

### Provenance records (finding 5.6)

76 records: 57 from the cross-check runner and 19 from the Stage 8 runner. Commits: 631fde0. Records with a modified tracked file: 0. Each record of the cross-check runner holds the commit, the inputs with md5, the methods and their settings, package versions, the cell cap and its seed, and the in-memory patches (`record/step4/v6_mixture_panel/v6_run_settings.tsv`).

<!-- RESULTS:END -->

## Reading the results

### V5: what the three arms say together

- **On 13 of 15 references BayesPrism's error against cell fractions grows with the bias in the mixtures**, as
  it should for a method that estimates shares of RNA: lowest without bias, higher with the references' own
  library-size differences, highest with the added factor. The rise from arm A to arm B is the part the
  battery of 2026-11-11 could not see, because its baseline was arm B.
- **Two references go the other way from arm A to arm B: BAT and BLOOD.** In BAT the cell types differ by
  1.08-fold in library size, so arm B carries almost no bias between types, and what arm A changes is the
  weighting of cells within a type (note N14). There BayesPrism underestimates brown adipocytes in arm A (true
  mean 0.185, estimated 0.111) and less so in arm B (0.168). In BLOOD arm A puts megakaryocytes and platelets
  at 0.192 against a true 0.074 (`record/step4/v5_simbu/v5_bayesprism_per_type.tsv`). The battery cannot say
  how much of any other reference's arm A to arm B difference is this same within-type effect, so reading 1
  is not a clean measure of the cost of the native bias.
- **The corrector rule is met on arm C minus arm A for a reason the rule was not written for.** MuSiC and DWLS
  carry each type's library size in what they fit: MuSiC multiplies a type's profile by its mean library size
  in the reference, and DWLS's signature is the mean raw count per type (both read in the installed packages).
  So they return shares of cells. That is right when the mixture carries the reference's own differences (arm
  B, where their error is lowest), too much when the mixture carries none (arm A, where it is highest), and
  not enough when a further factor is added (arm C). Their arm C minus arm A difference is negative because
  arm A is their worst case. Of the two methods the rule names, SCDC is the one whose difference is positive
  and still below BayesPrism's and CIBERSORTx's.
- **From arm B to arm C nothing changed since 2026-11-11.** Every method but Bisque loses 0.022 to 0.028, the
  rule is not met, and BayesPrism's values on the 13 unchanged references move by at most 0.0005 (the new
  exclusion list).
- **What stands for the pipeline.** BayesPrism does not correct mRNA-content differences, native or added.
  MuSiC and DWLS recover cell fractions under the native differences and are not protected against a factor
  their reference does not contain. "MuSiC and SCDC correct mRNA-content bias and BayesPrism and CIBERSORTx do
  not" holds here only in the first sense, and for SCDC only weakly: its error in arm B is no lower than
  BayesPrism's.

### V6: what the panel says

- **Every set meets the registered floor**, and no set moved by more than 0.004 against the panel of
  2026-11-08. On the 17 sets whose mixtures are byte-identical to that panel's, the only change is the exclusion
  list.
- **HEART and HYPOTH**, the two references whose labels changed, recover every benchmarked type at r 0.89 or
  above. In HYPOTH the re-annotation removed the weakest type of the earlier panel: glutamatergic neurons were
  at 0.850 beside a separate GABAergic label and are at 0.991 now. The broader "Neurons" label is lower than
  before (0.957 against 0.978), and the new ependymal label is the lowest of the reference (0.955).
- **The weak types are the earlier ones.** Six type tests are below 0.85: LIVER hepatocytes, OVARY erythroid
  cells, HIPPOC_cross excitatory neurons, and three KIDNEY_cross types. KIDNEY_cross keeps a pooled r of 0.17
  for BayesPrism: absolute kidney fractions across studies remain unusable, as before.
- **The DWLS correction changes little.** `mast_optimized` at p 0.05 gives the same medians as `mast` at p 0.01
  to within 0.01 on every set. Its 19 runs took 319 minutes in all; the earlier setting's took about 27 hours
  of job time by the scheduler's record of jobs 12192438 and 12192735 (not a table of this record).

## What the results do not show

### V5

- **It is a test of mRNA weighting only.** Each reference is deconvolved against the very cells its mixtures
  were drawn from, so there is no mismatch between reference and mixture. Accuracy against independent
  mixtures is V6's question, and neither battery uses tissue.
- **The truth is cell fractions, and BayesPrism estimates the share of RNA.** In arm A a type's share of the
  RNA equals its share of the cells; in arms B and C it does not. A rise in BayesPrism's error from arm A to
  arm B therefore measures how far RNA shares lie from cell shares on that reference. It is what the method is
  expected to do when its output is read as cell fractions, not an error against its own target.
- **Arm A changes two things** (note N14): the differences in library size between cell types, and the
  weighting of cells within a type. The battery cannot separate them.
- **Arm A has no sampling noise** (note N2): its mixtures are scaled to 1,000,000 reads, while arms B and C
  are subsampled to it.
- **For a method that rescales by library size, arm A is not the easy case** (note N16). The corrector rule
  compares arm C with arm A. A method can pass it because its error in arm A is high, without correcting
  anything in arm C. The table of each method's error in the three arms shows which case applies.
- **The dose-response of the earlier battery was not rerun.** It is not part of V5.

### V6

- **Only the benchmarked types are tested**: those with at least 20 cells in the mixture pool. The coverage
  table lists the others.
- **Holdout mixtures share a study with their reference.** Only four references have a cross-study set.
- **Not changed, as registered:** mixture compositions are flat Dirichlet, so every type averages the same
  share, unlike tissue; a pool as small as 20 cells is resampled, so mixtures share cells; BayesPrism sees every
  reference cell and the other methods at most 2,000 per type.
- **Also not changed** (the rest of finding 5.6): every method gets raw counts (the benchmark plan gave TPM
  mixtures to MuSiC, SCDC and CIBERSORTx and CPM single-cell data to CIBERSORTx; the project has no rat
  gene-length table); BayesPrism runs with cell states and its final update, as in Stage 8; the CIBERSORTx
  container draws samples without a seed (its image is now identified by md5 in every record); KIDNEY_cross and
  MUSCLE_cross pool every arm of their source study.
- **The DWLS comparison with the earlier panel is not a clean test of the setting.** Between the two panels
  the exclusion list also changed, and for HEART and HYPOTH the labels.

## Notes written before the results (2026-11-14)

Each note settles a point the registration leaves open, by the conservative reading, before the output it
affects exists. N1 to N15 were committed before any simulation, mixture or deconvolution of either battery;
N16 after the smoke run on TESTES and before any full run.

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

**N16. V5, a caveat on the corrector rule, written 2026-11-14 after the smoke run on TESTES and before any
full run.** On the smoke reference, MuSiC and DWLS had their lowest error in arm B and a much higher error in
arm A; BayesPrism and CIBERSORTx had their lowest error in arm A. This is expected of a method that converts
RNA shares to cell shares with the reference's library sizes: arm A removes the differences it corrects for.
The registered rule on arm C minus arm A is scored exactly as registered. Beside it, each method's mean error
in the three arms and the arm in which it is lowest are tabled (`v5_method_arm_rmse.tsv`), so a reader can see
whether a small difference comes from a low error in arm C or a high error in arm A. This table is descriptive
and was added after the smoke outputs were seen; no rule, arm or threshold changed.

## Departures, and what was done after the results were seen

**No registered rule, arm, threshold, set, source or method setting changed.** The points the registration
left open are the dated notes N1 to N16 above; N1 to N15 precede every output, and N16 was written after the
smoke run on TESTES and before any full run.

**Scheduling departures** (none changes a computation):
- sets were run several to a job and side by side, with the earlier core counts per run (note N12);
- the first task of a job builds the sets if they are missing, in place of a separate build job in front;
- short packs were queued as a second route and never ran;
- `PROVENANCE_MD5=1` was added to the job environment after the smoke run, so the Stage 8 runner records the
  md5 of its inputs.

**Done after the results were seen** (2026-11-15). The table, figure and report scripts were edited after the
first tables (job 12380299) had been read. The edits change how numbers are written and what is tabled, not
any rule or number:
- cell-type names reported by the DWLS guard are written with their spaces and signs;
- tables hold seven decimals in place of five, so that text, figure and table agree where a value sits on a
  rounding tie (DWLS's mean arm C minus arm A difference, -0.01255);
- added: BayesPrism's per-type table by arm, the counts of per-type r at 0.95 and 0.90, the total DWLS minutes;
- the HEART and HYPOTH figure is drawn as two stacked panels;
- the lead sentence of the V5 results names what the corrector verdict rests on.
The section "Reading the results" was written after the results.

**Not verified.**
- Whether the DWLS guard also fired for BAT brown adipocytes and OVARY erythroid cells in the panel of
  2026-11-08. That run did not log it.
- Run-to-run reproducibility beyond TESTES. The smoke and full runs agree byte for byte there for all six
  methods; no other reference was run twice.
- The mechanism behind BAT and BLOOD in arm A. That the within-type weighting is the cause is the reading that
  fits the numbers; no run isolates it.

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
python analysis/audit_remediation/make_step4_v5v6_report.py   # rewrites the results part of this document from the tables
python analysis/audit_remediation/compare_smoke_full.py       # -> record/step4/smoke/smoke_vs_full_TESTES.tsv
```

`bash slurm/analysis/v56_progress.sh` lists, per battery and method group, the sets not yet scored.

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

## Smoke run (job 12378260, commit `ea44863`, 2026-11-14, 35 minutes)

One job ran the offline test suite and then both batteries on TESTES with the registered settings, into
`simbu_bench_refs5_smoke` and `mixture_panel_refs5_smoke`. Nothing below is part of the result (note N13).

- **Tests.** 337 passed, none failed (`python -m pytest -q`, 107 s). With R on the path the suite includes the
  R test of the runner's settings, the guard and the provenance record, and the check that the installed DWLS
  0.5.2 has exactly one place for the guard in each of its two signature builders.
- **Every task ran and was scored**: BayesPrism, DWLS, MuSiC, SCDC, Bisque and CIBERSORTx on the three V5 arms
  and on TESTES_holdout.
- **Arm A removes the library-size differences** (registered check). Through SimBu's own summation, 1,000
  cells of one type total exactly 1 per cell in arm A for each of the six TESTES types. In arm B the same cells
  total 25,994 (spermatogonia) to 81,336 (spermatocytes) per cell, a 3.13-fold spread; in arm C the spread is
  4.42-fold. Table: `record/step4/smoke/v5_arm_library_size_TESTES_smoke.tsv`.
- **The new arguments do not disturb the old arms.** TESTES arm B's mixtures are byte-identical to the
  "nobias" mixtures of 2026-11-11 and arm C's to the "bias" mixtures; the three arms share one downsampled
  reference and one cell-fraction truth.
- **The panel builder reproduces.** TESTES_holdout's mixtures, truth and 70% reference are byte-identical to
  the 2026-11-08 panel's (`record/step4/smoke/v6_build_check_TESTES_smoke.tsv`).
- **Settings as registered, read back from the provenance files**
  (`record/step4/smoke/v5_run_settings_TESTES_smoke.tsv`, `v6_run_settings_TESTES_smoke.tsv`): V5 DWLS `mast`,
  p 0.01, fold change 0.5, no cap; V6 DWLS `mast_optimized`, p 0.05, fold change 0.5, cap 2,000 with seed 1;
  both DWLS builders patched, guard not fired; every runner on `rat_exclude_genes_v2.tsv` (261 genes removed,
  the same gene count for BayesPrism and the cross-check methods); commit `ea44863`, no modified tracked file.
  The CIBERSORTx record holds the container's md5 and no credential.
- **`mast_optimized` took 6 minutes** on TESTES_holdout; `mast` took 16 on the same set in the earlier panel.
- **The table and figure commands ran on the smoke outputs.** One defect was found and fixed (`7b014d9`): a
  table with no rows was written without its header, which stopped the figure script.

- **The changed summary scripts reproduce the earlier records** (job 12380311, commit `ddccec7`).
  `simbu_summary.py`, which now takes the two arms to compare, rewrites the nine files of
  `deconvolution/simbu_record_refs4all/` byte for byte from the 2026-11-11 battery; `mixture_panel_summary.py`
  rewrites the seven files of `deconvolution/mixture_panel_record/` byte for byte from the 2026-11-08 panel.

After the smoke run one setting of the job scripts changed, before any full run started: `PROVENANCE_MD5=1`,
so that the Stage 8 runner also records the md5 of its inputs. It changes no computation.

## Full batteries: jobs (submitted 2026-11-14 15:23)

Runner code is unchanged since the smoke commit `ea44863`: every later commit of this branch touches only the
table and figure scripts, their tests and this document. Each run's provenance file records the commit it ran
from and any modified tracked file. Three lanes, so at most three of these jobs hold a GPU slot at once.

| Job | ID | Runs |
|---|---|---|
| build | 12380292 | V5: arms A, B, C of the 15 references and the arm check. V6: the 19 sets and `check_panel_build.py` |
| A | 12380293 | BayesPrism, V5 (45 sets) and V6 (19 sets) |
| B | 12380294 | V5 DWLS (19 sets); V5 MuSiC, SCDC, Bisque and CIBERSORTx (45 sets) |
| C | 12380295 | V5 DWLS (26 sets); V6 MuSiC, SCDC, Bisque and CIBERSORTx (19 sets) |
| D, E, F | 12380296, 12380297, 12380298 | V6 DWLS (9, 7 and 3 sets), after A, B and C |
| tables | 12380299 | the three table and figure commands, after D, E and F |
| smoke tables | 12380311 | regression of `simbu_summary.py` and `mixture_panel_summary.py` against their committed records; tables of the smoke runs |

**Outcome.** All of them completed.

| Job | Ran (2026-11-14 to 15) | Tasks, all with exit status 0 | Peak memory |
|---|---|---|---|
| build 12380292 | 17:18 to 17:32 | both batteries built | - |
| A 12380293 | 18:05 to 22:44 | 64 | 46 GB |
| B 12380294 | 17:39 to 22:52 | 109 | 336 GB |
| C 12380295 | 17:42 to 22:56 | 64 | 261 GB |
| D 12380296 | 23:29 to 00:51 | 9 | 347 GB |
| E 12380297 | 23:29 to 00:39 | 7 | 211 GB |
| F 12380298 | 23:29 to 00:17 | 3 | 208 GB |

- 256 deconvolution runs wrote a provenance record (180 in V5, 76 in V6). Every one names commit `631fde0`
  and no modified tracked file (`record/step4/v5_simbu/v5_run_settings.tsv`,
  `record/step4/v6_mixture_panel/v6_run_settings.tsv`).
- No task failed, none ran out of memory or time, and no registered retry was used. MuSiC and Bisque gave no
  output on HYPOTH, OVARY and SPLEEN, whose references hold one sample: the registered "not applicable" case
  (18 runs in V5, 6 in V6).
- The smoke runs and the full runs of TESTES give byte-identical estimates for all six methods on the four
  TESTES sets, CIBERSORTx included (`record/step4/smoke/smoke_vs_full_TESTES.tsv`). This is a run-to-run check
  on one reference only.
- The tables and figures in the record were built by job 12392335 (commit `a65487d`, no modified tracked file,
  2026-11-15 16:30). Jobs 12380299, 12392323 and 12392329 built them before; see "Done after the results were
  seen".
- The full offline test suite passes on the same commit: 340 passed, none failed, 17 deselected by their
  network or data marker (job 12392337).

**Scheduling change, 2026-11-14 16:32, before any of these jobs had started.** The partition was saturated
(about 250 eligible jobs of other groups at the same or higher priority) and the build job had waited an hour.
Jobs A, B and C were made eligible at once instead of after the build job: the first task to run in any of them
now builds the battery's sets if they are missing, under a lock so that only one builder runs
(`slurm/analysis/v56_ensure_build.sh`); the build job, if it starts later, finds the sets built and ends. The
build commands, the sets and every per-task setting are unchanged. Time limits were shortened (A 9 h, B to F
12 h) so the jobs fit more gaps in the schedule.

**Build, 17:18 to 17:32 (job 12380292, commit `c5644d3`, no modified tracked file).** Both batteries were
built. Before any deconvolution:
- the four cross sets' harmonization maps equal the reviewed ones, so the cross sets may run (note N10);
- 17 of the 19 panel sets are byte-identical to the 2026-11-08 panel's (mixtures, truth and, for holdout sets,
  the 70% reference); HEART_holdout and HYPOTH_holdout differ, as they should
  (`data/deconvolution/mixture_panel_refs5/build_check.tsv`);
- the arm check ran on all 15 references (185 cell types).

**Short packs, submitted 17:37 (array 12382533, tasks 0 to 23, two at a time; job 12382534).** Only jobs with
a short time limit were being started: the 40-minute build job was, the 9- and 12-hour jobs were not. The tasks
that fit well inside 50 minutes were therefore also listed in short packs: BayesPrism on both batteries, and
MuSiC, SCDC, Bisque and CIBERSORTx on both; job 12382534 holds the three V5 HEART BayesPrism tasks (85-minute
limit). They call the same task script with the same settings. A task that is done is skipped, and a task that
another job is running is skipped (a lock per set and method group), so no task runs twice. The long jobs stay
queued for the DWLS tasks and for anything the short packs do not finish.

**The short packs never ran.** Jobs B, C and A started at 17:39, 17:42 and 18:05, before any short pack. The
array and job 12382534 were then put on hold, so that three jobs held GPU slots and not five. Every task they
list was done by jobs A to C. They are still in the queue on hold and should be cancelled, not released: a
released pack would skip the finished tasks but would rerun MuSiC, SCDC and Bisque on the nine V5 sets and
three V6 sets of the one-sample references (those have no "done" marker, because two of the three methods give
no output there) and overwrite their records.
