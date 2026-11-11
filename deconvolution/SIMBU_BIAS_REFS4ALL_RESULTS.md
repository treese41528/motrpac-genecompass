# SimBu mRNA-content-bias battery on the 15 current references: results (2026-11-11)

**Question.** The mRNA-bias battery and its dose-response were last run in July, on the earlier references. Do
July's four statements still hold on the 15 references of the 16-tissue default, and which cell type does the
bias distort most in each tissue now?

- Pre-registration: `SIMBU_BIAS_REFS4ALL_PREREG.md` (25b682d), committed before any simulation. Protocol, methods,
  settings and scoring are July's; nothing was changed after results were seen.
- Scripts: `omnideconv_bench/build_simbu_battery.sh`, `simulate_simbu.R` (SimBu 1.8.0), `simbu_summary.py`.
- Record: `simbu_record_refs4all/` (the summary tables and `summary.txt`).
- Data: `data/deconvolution/simbu_bench_refs4all/`.

## What ran

| Item | Count |
|---|---|
| References | 15 (MUSCLE serves both SKMGN and SKMVL) |
| Battery sets | 30 (each reference with and without the added bias) |
| Dose-response sets | 12 (LIVER and BLOOD at alpha 0, 0.5, 1, 2, 3, 4) |
| Method runs | 252 (42 sets x 6 methods) |
| Scored | 240 |
| Not applicable | 12: MuSiC and Bisque on HYPOTH, OVARY and SPLEEN, with and without bias |
| Failed, or retried for memory or time | 0 |

- **The not-applicable runs are the registered case.** The HYPOTH, OVARY and SPLEEN references each hold one
  sample. MuSiC and Bisque need at least two (Bisque's message: "Only one individual detected in single-cell
  data").
- **Sanity checks pass.** The bias and no-bias references are byte-identical and share one cell-fraction truth on
  15 of 15 references. Each dose-response tissue has one reference and one truth across its six alphas.
- **Jobs.**
  - Build 12304323; DWLS 12304324.
  - BayesPrism 12304325 (sets 0-28), then 12315299 and 12315300.
  - MuSiC, SCDC and Bisque 12304326 (sets 0-6), then 12315298.
  - CIBERSORTx 12315301-12315303.
- **A scheduling change, not a protocol change.** The per-set array tasks each waited for their own GPU slot.
  After 16 hours with 90 sets left, the rest were run back to back in six jobs that call the same per-set scripts
  with the same settings (`slurm/analysis/simbu16_pack.slurm`, local).

## Readings, in the registered order

### 1. Corrector ranking: July's finding is not reproduced

Mean ΔRMSE (pooled RMSE with the added bias minus without), lower meaning less affected. The common set is the 12
references where all six methods ran.

| Method | Common set (12) | All scored | References with ΔRMSE > 0 | July (10 tissues) |
|---|---|---|---|---|
| Bisque | +0.0024 | +0.0024 (12) | 12 of 12 | +0.003 |
| SCDC | +0.0215 | +0.0206 (15) | 14 of 15 | +0.016 |
| CIBERSORTx | +0.0220 | +0.0224 (15) | 15 of 15 | +0.024 |
| BayesPrism | +0.0251 | +0.0255 (15) | 15 of 15 | +0.026 |
| MuSiC | +0.0254 | +0.0254 (12) | 12 of 12 | +0.018 |
| DWLS | +0.0283 | +0.0279 (15) | 15 of 15 | +0.024 |

- **Registered rule.** The benchmark paper's finding counts as reproduced if MuSiC and SCDC are both below
  BayesPrism and CIBERSORTx on the common set.
- **Result: not reproduced.** MuSiC (+0.0254) is above both BayesPrism (+0.0251) and CIBERSORTx (+0.0220). SCDC
  (+0.0215) is below both, by 0.0005 against CIBERSORTx.
- **No single reference decides it.** Dropping any one of the 12 references leaves the rule unmet.
- **DWLS** has the largest ΔRMSE of the six, as in July's reading of it as bias-sensitive.
- **Bisque** barely moves, as in July. Its error is already the largest without the bias (see below).

### 2. BayesPrism degrades under the added bias on 15 of 15 references

July: on every tissue. Its ΔRMSE runs from +0.007 (BAT) to +0.046 (CORTEX).

### 3. The driver is the highest-mRNA type on 10 of 15 references, not always

The driver is the cell type with the largest per-type ΔRMSE for BayesPrism. Its mRNA rank is by mean expressed
genes in the reference (rank 1 = most); the rank by total counts is alongside.

- By expressed genes the driver has rank 1 on **10 of 15** references. By total counts it is 9 of 15.
- The five exceptions:
  - **BAT:** brown adipocytes, rank 6 of 6. The six types differ by only 1.1-fold in mRNA content, so rank carries
    little here.
  - **KIDNEY:** proximal tubule cells, rank 3 of 15.
  - **LUNG:** alveolar macrophages, rank 5 of 21 (3 by counts).
  - **SMLINT:** transit-amplifying cells, rank 3 of 13 (2 by counts).
  - **TESTES:** condensed spermatids, rank 6 of 6 (4 by counts). This is the one driver whose estimate falls.

### 4. Dose-response: RMSE rises with alpha, and DWLS has the best baseline and the steepest slope

Pooled RMSE at each alpha, with the composition pinned.

| Tissue | Method | alpha 0 | 0.5 | 1 | 2 | 3 | 4 | Slope per alpha | Non-decreasing |
|---|---|---|---|---|---|---|---|---|---|
| LIVER | BayesPrism | 0.051 | 0.071 | 0.092 | 0.136 | 0.177 | 0.214 | +0.041 | yes |
| LIVER | CIBERSORTx | 0.090 | 0.103 | 0.116 | 0.144 | 0.172 | 0.196 | +0.027 | yes |
| LIVER | DWLS | 0.010 | 0.021 | 0.038 | 0.079 | 0.123 | 0.165 | +0.039 | yes |
| LIVER | MuSiC | 0.011 | 0.021 | 0.038 | 0.079 | 0.122 | 0.165 | +0.038 | yes |
| LIVER | SCDC | 0.033 | 0.036 | 0.048 | 0.084 | 0.125 | 0.166 | +0.033 | yes |
| LIVER | Bisque | 0.109 | 0.110 | 0.110 | 0.113 | 0.116 | 0.119 | +0.002 | yes |
| BLOOD | BayesPrism | 0.052 | 0.069 | 0.083 | 0.108 | 0.130 | 0.147 | +0.024 | yes |
| BLOOD | CIBERSORTx | 0.063 | 0.076 | 0.089 | 0.110 | 0.129 | 0.145 | +0.020 | yes |
| BLOOD | DWLS | 0.009 | 0.023 | 0.041 | 0.070 | 0.096 | 0.117 | +0.027 | yes |
| BLOOD | MuSiC | 0.033 | 0.038 | 0.049 | 0.077 | 0.105 | 0.128 | +0.024 | yes |
| BLOOD | SCDC | 0.069 | 0.046 | 0.044 | 0.070 | 0.097 | 0.118 | +0.012 | no |
| BLOOD | Bisque | 0.099 | 0.104 | 0.109 | 0.114 | 0.117 | 0.114 | +0.004 | no |

- **Monotone rise.** All six methods on LIVER. On BLOOD, four of six: SCDC improves up to alpha 1 before rising,
  and Bisque dips at alpha 4.
- **DWLS has the lowest alpha-0 RMSE of the six** on both tissues, as July stated.
- **DWLS has the steepest slope of the three correctors** (DWLS, MuSiC, SCDC) on both tissues, as July stated. On
  LIVER MuSiC's is nearly the same (+0.0384 against +0.0388).

### 5. Bias drivers on the current references

BayesPrism's driver per reference. "Estimate" is the driver's mean estimated fraction without, then with, the
added bias; "true" is its mean cell fraction.

| Tissue | Driver | mRNA rank (genes; counts) | True | Estimate | Direction |
|---|---|---|---|---|---|
| BAT | Brown adipocytes | 6 of 6; 6 | 0.185 | 0.169 → 0.215 | inflated |
| BLOOD | Plasmacytoid dendritic cells | 1 of 13; 1 | 0.084 | 0.172 → 0.229 | inflated |
| CORTEX | Excitatory neurons | 1 of 12; 1 | 0.092 | 0.354 → 0.492 | inflated |
| HEART | Cardiomyocytes | 1 of 16; 1 | 0.066 | 0.181 → 0.303 | inflated |
| HIPPOC | Dentate gyrus granule cells | 1 of 11; 1 | 0.111 | 0.218 → 0.284 | inflated |
| HYPOTH | Neurons | 1 of 13; 1 | 0.067 | 0.088 → 0.125 | inflated |
| KIDNEY | Proximal tubule cells | 3 of 15; 3 | 0.063 | 0.097 → 0.149 | inflated |
| LIVER | NK cells | 1 of 8; 1 | 0.142 | 0.268 → 0.350 | inflated |
| LUNG | Alveolar macrophages | 5 of 21; 3 | 0.045 | 0.074 → 0.128 | inflated |
| SKMGN, SKMVL (MUSCLE) | Skeletal muscle cells | 1 of 9; 1 | 0.113 | 0.160 → 0.224 | inflated |
| OVARY | Luteal cells | 1 of 18; 1 | 0.059 | 0.247 → 0.370 | inflated |
| SMLINT | Transit-amplifying cells | 3 of 13; 2 | 0.087 | 0.162 → 0.215 | inflated |
| SPLEEN | Erythroid progenitors | 1 of 14; 1 | 0.066 | 0.131 → 0.214 | inflated |
| TESTES | Condensed spermatids | 6 of 6; 4 | 0.179 | 0.132 → 0.087 | deflated |
| WATSC | Cycling cells | 1 of 13; 2 | 0.080 | 0.129 → 0.160 | inflated |

- **The other methods mostly share BayesPrism's driver.** CIBERSORTx has the same driver on 12 of 15 references,
  DWLS on 12 of 15, SCDC on 11 of 15 and MuSiC on 7 of 12.
- **LIVER's driver is the reference's NK cells, not hepatocytes.** In the current liver reference the 181 cells
  labelled NK cells have the most expressed genes (1,134) and counts (2,055) of the eight types. Hepatocytes have
  the fewest expressed genes (542) and rank 3 by counts. An NK label on the most RNA-rich cells of a liver
  reference is unusual and worth a look at that cluster; this battery does not test the label.

## What the "no bias" condition contains (checked in SimBu 1.8.0, not a registered reading)

- **SimBu sums raw counts.** `simulate_bulk` leaves `remove_bias_in_counts` at its default (FALSE), and
  `simulate_simbu.R` does not set it. So without any scaling a sampled cell contributes its own library size.
- **The baseline therefore already carries the references' native differences in mRNA content.** A type with
  larger libraries supplies more of the reads than of the cells.
- **The added bias is a second factor on top.** `--scaling expressed_genes` multiplies each cell's counts by its
  number of expressed genes.
- **ΔRMSE measures only the added factor.** That was equally true in July.
- **BayesPrism's baseline estimate is the driver's share of reads, not of cells.** With equal cell fractions,
  CORTEX excitatory neurons would supply 0.34 of the reads from their library sizes alone. BayesPrism estimates
  0.354 against a true cell fraction of 0.092. HEART cardiomyocytes: 0.18 of the reads, estimate 0.181, true
  0.066.

Mean pooled RMSE against cell fractions, on the 12 common references (descriptive):

| Method | Without the added bias | With it |
|---|---|---|
| DWLS | 0.010 | 0.038 |
| MuSiC | 0.021 | 0.046 |
| BayesPrism | 0.047 | 0.072 |
| SCDC | 0.053 | 0.075 |
| CIBERSORTx | 0.065 | 0.087 |
| Bisque | 0.067 | 0.070 |

- DWLS has the lowest error on 14 of 15 references without the added bias, and on 10 of 15 with it.
- So DWLS and MuSiC are closest to cell fractions in both conditions, while none of the five methods that respond
  at all is protected from the added factor.

## Consequences

- **Withdrawn for the current references:** "MuSiC and SCDC correct mRNA-content bias and BayesPrism and
  CIBERSORTx do not." By the registered statistic all five responsive methods lose a similar amount (+0.021 to
  +0.028).
- **Unchanged:** BayesPrism does not correct mRNA-content bias. Its error against cell fractions rises on every
  reference, and its baseline estimates follow read shares.
- **Narrowed:** "the driver is always the highest-mRNA type" becomes 10 of 15 references.
- **Guidance (`OMNIDECONV_RESULTS.md`):**
  - The driver table above replaces July's.
  - The cross-check for a driver's absolute fraction is DWLS or MuSiC, not MuSiC or SCDC. SCDC's error against
    cell fractions is no lower than BayesPrism's here.
  - The safe and risky classes of claim are unchanged. They rest on within-type contrasts and on target-sum
    tokenization, which this battery does not test.
- **Still not said by this battery:** accuracy against independent mixtures. That is the mixture panel's question
  (`MIXTURE_PANEL_REFS4ALL_RESULTS.md`).
- **Not done:** a baseline with the native bias removed (`remove_bias_in_counts = TRUE`), which would let ΔRMSE
  measure the whole mRNA-content effect. It is a protocol change and would need its own registration.
