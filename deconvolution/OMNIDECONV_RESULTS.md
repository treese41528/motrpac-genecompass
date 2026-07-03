# omnideconv Cross-Method Panel — Results

**Status:** Core fraction benchmark (paper Fig-2 analog) COMPLETE + PRODUCTION-ALIGNED; **SimBu mRNA-bias
battery + dose-response COMPLETE** (all 10 tissues + corrected liver/lung); agreement with the omnideconv
paper documented; **downstream-claims bias guidance** added below. Last updated 2026-07-02.
Companion to `OMNIDECONV_BENCHMARK_PLAN.md` (the execution plan) and
`reference_omnideconv_benchmark` / `omnideconv-crosscheck-status` memories.

## What this is

An independent cross-method check of our patched BayesPrism deconvolution: run the same
known-truth pseudobulk mixtures (`validation_v2/<tissue>/mixtures`, 50 mixtures each, Dirichlet
proportions) through five reference-based methods (**CIBERSORTx, DWLS, MuSiC, SCDC, Bisque**, via
`omnideconv`) plus **BayesPrism**, and score each against ground truth with `score_validation.py`.
Driver: `slurm/analysis/omnideconv_full.slurm` + `dwls_finish.slurm`; aggregators
`tmp/omnideconv_gather.sh` (pooled) and `tmp/omni_pertype_agg.py` (per-type).

## Two metrics, two questions (always report both)

- **`overall_pooled_pearson`** — one correlation over ALL (sample × type) points. Absolute-composition
  accuracy; stricter; sensitive to cross-type scale/bias and to the dominant type.
- **`median_pearson`** — median of the per-type across-sample Pearson r. The **differential** question
  ("does each cell type's fraction track across samples") — what MoTrPAC's relative/differential framing
  actually asks. This is the headline metric.

The earlier recorded WAT numbers (DWLS 0.98/0.99, Bisque 0.73/0.78) were the per-type macro/median and
reproduce exactly; pooled just looks lower because it is the stricter metric. No regression.

## BayesPrism scored on ALL tissues (2026-06-30)

BayesPrism was run through the SAME validation_v2 harness for every tissue on 2026-06-01
(`results/bp_result.rds` + `estimated_fractions.csv`), but only WAT had been scored into the panel.
Scoring the existing outputs (`score_validation.py --est-file results/estimated_fractions.csv --tag
bp_rna`) filled the column: clean label match (0 missing types), and the numbers reproduce the recorded
multi-tissue survey (HRT pooled 0.878≈0.875, HIP_merged 0.817≈0.82, KID_289104 0.642≈0.64) — the same
run, not new numbers.

## Production-aligned panel (10 tissues, 1:1 with `genecompass_input`)

The panel now validates the EXACT reference production embeds for each of its 10 tissues. Seven already
matched; three were aligned on 2026-06-30 (`slurm/analysis/align_deconv.slurm` +
`align_ctx_union_*.slurm`; new dirs `validation_v2/{CTX_union,SKMVL,LIV_holdout}`, two reuse existing
mixtures via symlink — see `reference_omnideconv_validation_vs_production_refs` memory):

| prod tissue | reference validated | val tissue | note |
|---|---|---|---|
| watsc | white adipose GSE137869 (holdout) | WAT_holdout | already matched |
| blood | PBMC GSE285476 (holdout) | PBMC_holdout | already matched |
| heart | GSE280111 LV (holdout) | HRT_holdout | already matched |
| hippoc | hippocampus_GSE305314_WT_merged | HIP_merged | already matched |
| kidney | kidney_GSE240658 | KID_cross | already matched |
| lung | lung_GSE178405 | LNG_cross | already matched |
| skmgn | gastrocnemius_GSE184413 | GAS_cross | already matched |
| **cortex** | **cortex_GSE303115_union_merged (18k genes)** | **CTX_union** | **NEW** — panel had used the 5.5k-gene `_merged` |
| **skmvl** | **skeletal_muscle_GSE254371_muscle_merged** | **SKMVL** | **NEW** — was entirely unvalidated |
| **liver** | **liver_GSE220075** | **LIV_holdout** | **NEW** — reused V0 holdout |

### median per-type r (differential — headline)
```
prod      ref_tissue    BayesPrm CIBERSRTx     DWLS    MuSiC     SCDC   Bisque
watsc     WAT_holdout      0.972    0.953    0.984    0.961    0.925    0.791
blood     PBMC_holdout     0.970    0.960    0.969        -    0.944        -
heart     HRT_holdout      0.950    0.893    0.925    0.921    0.908    0.885
hippoc    HIP_merged       0.994    0.988        -    0.945    0.947    0.973
kidney    KID_cross        0.852    0.887    0.792    0.849    0.826    0.811
lung      LNG_cross        0.705    0.506    0.520    0.579    0.568    0.323
skmgn     GAS_cross        0.948    0.840    0.940    0.820    0.730    0.882
cortex    CTX_union        0.940    0.947    0.827    0.944    0.912    0.833
skmvl     SKMVL            0.996    0.988    0.991    0.925    0.965    0.876
liver     LIV_holdout      0.992    0.985    0.978    0.956    0.901    0.949
```
### pooled r (absolute composition)
```
prod      ref_tissue    BayesPrm CIBERSRTx     DWLS    MuSiC     SCDC   Bisque
watsc     WAT_holdout      0.955    0.907    0.864    0.749    0.574    0.554
blood     PBMC_holdout     0.841    0.739    0.654        -    0.558        -
heart     HRT_holdout      0.878    0.759    0.780    0.702    0.526    0.580
hippoc    HIP_merged       0.817    0.782        -    0.617    0.679    0.233
kidney    KID_cross        0.057    0.635    0.296    0.450    0.424    0.246
lung      LNG_cross        0.156    0.392    0.224    0.204    0.116    0.148
skmgn     GAS_cross        0.419    0.425    0.549    0.151    0.065    0.367
cortex    CTX_union        0.445    0.304    0.412    0.267    0.244    0.541
skmvl     SKMVL            0.987    0.930   (dwls*)   0.449    0.789    0.265
liver     LIV_holdout      0.960    0.848    0.893    0.844    0.735    0.846
```
**Cortex DWLS resolved (2026-07-01):** DWLS/MAST OOM'd on the 18k-gene cortex ref at 32/16/8 cores
but **fit at 2 cores** (job `11165853`, cap 2000, MaxRSS 367 GB, **16.7 h** runtime) → pooled 0.412 /
median 0.827 (now in the tables). So the panel is 10 tissues × 6 methods complete. DWLS on the deep
cortex ref is a memory/time hog, not impossible. Remaining gaps are method-intrinsic, not bugs: PBMC
MuSiC/Bisque (single-donor ref → structural), HIP-DWLS (QP "dataset too small").

### Verdicts on the 3 newly-aligned production references
- **SKMVL** (was unvalidated): BayesPrism **0.987 pooled / 0.996 median — best in the panel.** All 14
  merged labels harmonized from the GSE255196 cross source. The muscle merge production adopted is sound.
- **liver** (was absent): BayesPrism **0.960 / 0.992.** Validates.
- **cortex** (18k-gene `union_merged`): BayesPrism median **0.940**, slightly BEATS the 5.5k `_merged`
  (0.920) — the Fix-2 gene-deepening is confirmed safe; pooled stays low (neuron-dominant collapse).

## Verified interpretation (adversarial workflow 2026-06-30)

1. **Per-type/differential tracking is strong nearly everywhere** (median 0.8–0.99) — CONFIRMED.
2. The grim pooled r has **two** causes: dominant-parenchyma collapse is real but **only for brain-neuronal
   (CTX/HIP) + lung**; in **kidney & gastroc the dominant type tracks fine (0.77–0.94)** yet pooled stays
   low — there it is a **pooling cross-type scale/bias artifact**, not collapse. (Refines
   `reference_dominant_parenchyma_cross_failure`: brain/lung-specific, not a universal parenchyma law.)
3. **Not BayesPrism-specific** (CIBERSORTx collapses the same neurons) BUT **Bisque partially recovers**
   the dominant cross neurons (CTX 0.61, HIP 0.45) — "every method fails the dominant" is too strong.
4. **LNG_cross is the one genuine soft spot** — ~0.5 median across ALL methods, not just the dominant.
5. BayesPrism validates every production reference at **median ≥0.85 except lung (0.705)**.

## Comparison to the omnideconv benchmark paper (Dietrich et al. 2026, Genome Biology 27:6)

We ran the paper's **core fraction benchmark** (Fig 2 analog); we did NOT yet run its SimBu confounder
battery (see below). The one apparent disagreement resolves to a metric choice, and the rest agree:

| Paper finding (human/mouse) | Our rat panel | Verdict |
|---|---|---|
| BayesPrism = **high correlation but high RMSE** (systematic overestimation) | We lead with per-type **correlation** → BayesPrism best/co-best | Same axis; the paper's RMSE penalty is the mRNA-bias we don't lead with |
| **Bisque weakest** (B-cell R=−0.68; large bias) | Bisque our lone outlier everywhere | ✅ strong |
| **Excitatory neurons harder than inhibitory** (Allen brain) | Our cortex/HIP collapse IS the excitatory/glutamatergic neurons | ✅ direct |
| BayesPrism **degrades with coarser types** (T/NK) | We merge only collinear parenchyma, keep immune resolved; HIP failed on an over-coarse "Neurons" catch-all | ✅ (drove our design) |
| **DWLS slowest / highest memory** (>200 GB @100k cells) | DWLS OOM-killed repeatedly on the 18k-gene cortex ref | ✅ experiential |
| **dataset origin / cross-study** a major variance source | holdout pooled ≫ cross pooled | ✅ |
| methods struggle with **abundant + closely-related** types | dominant-parenchyma + collinear-over-split-neuron collapse | ✅ |
| only **DWLS/MuSiC/SCDC(/Scaden) correct mRNA bias**; BayesPrism/CIBERSORTx/AutoGeneS don't | **TESTED** (SimBu battery, all 10 tissues + dose-response, 2026-07-01): MuSiC/SCDC correct; BayesPrism/CIBERSORTx don't — as predicted. **DWLS diverges** (bias-sensitive in rat; dose-response shows its correction is magnitude-limited). | ✅ core agrees; DWLS is the one rat caveat |

Paper's overall recommendation: **DWLS + Scaden most robust** across scenarios (but heaviest compute);
no definitive ranking (context-specific); *"run and compare a few top-performing methods."*

## mRNA-bias battery — RESULTS (SimBu ±expressed-genes, all 10 tissues + dose-response)

Self-simulated ±mRNA-bias per reference (500 cells/type, pinned composition), scored vs cell-fraction
truth. `deconvolution/omnideconv_bench/{simulate_simbu,bias_delta,dose_response}` ; jobs 11167195
(battery), 11167410 (dose-response), 11176839 (corrected liver+lung).

**Corrector ranking (mean ΔRMSE across tissues; lower = corrects the bias):**
```
bisque +0.003  <  scdc +0.016  <  music +0.018  <  dwls +0.024 ≈ cibersortx +0.024 ≈ bp_rna +0.026
```
- **MuSiC + SCDC correct** (low ΔRMSE) — agrees with the paper. **BayesPrism + CIBERSORTx degrade** —
  agrees. **Bisque** has the tiniest ΔRMSE but the dose-response shows that is *insensitivity* (terrible
  baseline, flat-and-wrong), not correction — so we agree it is weak, via a different metric.
- **DWLS is the one divergence:** the paper ranks it a top corrector; in rat it is **bias-sensitive**
  (ΔRMSE ~BayesPrism). The **dose-response** (LIV/PBMC × α) explains it: DWLS has the *best baseline*
  (α=0 near-perfect) but the *steepest slope* — it corrects *small* bias (the paper's moderate regime)
  and *fails at large* bias. So it is magnitude-limited, not a flat contradiction.
- **The bias is DOSE-dependent** (RMSE rises monotonically with amplitude within every tissue) — it is a
  dose effect, **not species-specific**. Assay (sc vs sn) does NOT explain the tissue gradient (snRNA
  tissues span an equal/higher range than scRNA).

**Per-tissue bias driver (the high-mRNA cell type the bias inflates):** liver→Hepatocytes,
heart→Cardiomyocytes, cortex→Cholinergic/Excitatory neurons, hippocampus→Excitatory neurons,
kidney→β-intercalated/Proximal-tubule, blood→Classical monocytes, lung→Alveolar macrophages,
skmvl→Skeletal-muscle/Progenitor, skmgn→Muscle fibroblasts, WAT→Fibroblasts. Always the highest-mRNA type.

**Corrected liver + lung (on the production references, job 11176839):**
- **liver: bp ΔRMSE 0.081 → 0.040 (HALVED)**, driver shifted Hepatocytes→Endothelial. The Visium
  contamination WAS the cause of liver's lone-extreme SimBu bias (high-mRNA multi-cell spots inflated the
  reference's mRNA-content spread). Removed, liver sits in the normal band (0.02–0.04) — no longer an
  outlier. NB: the *real-bulk* Z was unchanged (0.998), so this is a benchmark-metric effect; the
  parenchyma-Z red flag (real bulk) is intrinsic mRNA bias, not the contamination.
- **lung: bp ΔRMSE +0.011 on either reference** (mild), native driver = Alveolar macrophages. Lung is a
  low-bias tissue regardless of reference.

**Net vs omnideconv:** the corrected results **agree with the paper's core bias finding** (MuSiC/SCDC
correct, BayesPrism/CIBERSORTx don't, Bisque weak). Removing the liver Visium artifact *increased*
agreement — it eliminated the one dramatic divergence (an apparent parenchyma-dominance blow-up) and left
rat in the paper's moderate-bias regime. The only genuine rat-specific caveat is **DWLS** (magnitude-
limited correction); the headline "BayesPrism ranks best in our panel" stays the **correlation-vs-RMSE**
metric artifact.

## Guidance — making downstream claims where mRNA bias exists

We are committed to **BayesPrism** (the only panel method that yields the per-cell-type Z the pipeline
needs), and BayesPrism does NOT correct mRNA-content bias. So high-mRNA cell types have over-estimated
RNA-fractions and a distortable Z. Gate claims accordingly:

**SAFE (bias-robust) — state these freely:**
- **Differential / across-condition claims** (does cell type X change with exercise/sex/dose). The
  per-cell-type bias is ~constant across samples, so it **cancels** in a within-type across-sample
  contrast. This is the strong `median_pearson` axis (0.8–0.99 everywhere).
- **Immune / stromal cell types** (not high-mRNA): fractions AND Z reliable (our immune DE recovers
  known Yu/Vetr biology).
- **Embedding-based results** (Stage 9 pseudo-cells → GeneCompass: the exercise hotspots, subspace probe,
  cross-species transfer). Target-sum-6500 tokenization strips the mRNA-content **scale**, insulating
  them from the fraction bias — e.g. lung's 0→3 hotspots and their rat→human preservation are trustworthy.

**RISKY (bias-compromised) — add a caveat, cross-check, or don't claim:**
- **Absolute fraction / abundance of a high-mRNA driver type** (the per-tissue table above): over-
  estimated. Do NOT state "tissue is X% <driver-type>" or rank the driver against other types by fraction.
- **DE on the dominant-parenchyma Z** (hepatocytes / cardiomyocytes / skeletal-muscle): the pre-registered
  red flag — parenchyma expression-change claims are low-confidence and intrinsic to BayesPrism (no
  reference cleanup fixes it; see the liver Visium result).
- **Cross-cell-type composition comparisons** between types that differ in mRNA content.

**To guard a specific claim:** (1) frame it *differentially/relatively*, not absolutely; (2) cross-check
the driver-type fraction against a **corrector** (MuSiC/SCDC — agreement ⇒ confidence); (3) for
parenchyma DE, validate against bulk controls or flag it low-confidence; (4) prefer embedding-based
(Stage 9+) over fraction-based statements for the driver types.

## Open items
- **Cross-method panel + full mRNA-bias battery + dose-response: COMPLETE** (all 10 tissues, corrected
  liver+lung).
- **Not run:** pure/weighted/mirror_db SimBu scenarios (simulator supports them, not driven);
  AutoGeneS + Scaden branches; bulk-TPM (D1); a dedicated Z cross-check (bMIND / CIBERSORTx hi-res GEP);
  native-lung accuracy validation with a non-embryonic cross source (build_datasets LNG source is GSE196313).
