# SimBu mRNA-bias battery on the current references (pre-registered 2026-11-09, before any simulation)

**Why.** The SimBu mRNA-content-bias battery and its dose-response (`OMNIDECONV_RESULTS.md`, "mRNA-bias battery";
jobs 11167195, 11167410, 11176839; 2026-07-01/02) ran on the June/July references of 10 tissues. Its conclusions
feed the downstream claim guidance: BayesPrism does not correct mRNA-content bias, the per-tissue bias-driver
table, the SAFE/RISKY gating, and Fig. 3b of the preprint. The default now deconvolves against 15 other
references, the set the known-truth mixture panel was rerun on (`MIXTURE_PANEL_REFS4ALL_PREREG.md`). This reruns
the battery and the dose-response on them with the July protocol unchanged.

**References.** The `reference_dir` of every ready tissue in `deconvolution/tissue_references.yaml`: 15 references
for the 16 tissues (SKMGN and SKMVL share `references_v4/MUSCLE_GSE137869_Y`). Set codes BAT, BLOOD, CORTEX,
HEART, HIPPOC, HYPOTH, KIDNEY, LIVER, LUNG, MUSCLE, OVARY, SMLINT, SPLEEN, TESTES, WATSC. The builder checks its
table against `tissue_references.yaml` and stops on any difference.

**Battery (primary), per reference.** `omnideconv_bench/build_simbu_battery.sh` runs `simulate_simbu.R`
(SimBu 1.8.0) exactly as `simbu_mrna_bias.slurm` did in July:
- A seeded downsample to at most 500 cells per type. Every type is kept; a type with fewer cells keeps all of them.
- 50 pseudobulks of 1,000 cells, SimBu's `random` composition, 1,000,000 reads, seed 1.
- Simulated twice from the same seed: **bias** (`--scaling expressed_genes`) and **nobias** (`--scaling NONE`).
  The downsample and the compositions are identical, so the cell-fraction truth is the same and only the mRNA
  weighting differs.
- Both are deconvolved against that downsampled reference (no further cap, `REF_MAX_CELLS_PER_TYPE=0`). The
  battery therefore has no reference mismatch: it isolates the mRNA-weighting effect.

**Dose-response (secondary), on LIVER and BLOOD.** These are July's two tissues (LIV_holdout and PBMC_holdout), now
on their current references. `--bias-alpha` takes 0, 0.5, 1, 2, 3 and 4, with the composition pinned (seed 1)
and the same simulation sizes. Per-type mRNA scale = (mean expressed genes of the type)^alpha.

**Methods.**
- BayesPrism (`R/run_deconvolution.sh`, the Stage 8 runner).
- CIBERSORTx, DWLS (`mast`), MuSiC, SCDC and Bisque (`R/run_omnideconv.sh`), on every battery and dose-response
  set. July left CIBERSORTx out of the dose-response to save Apptainer runs; it is included now, so both parts
  use one method set.
- Scoring: `score_validation.py --truth cellfrac` (SimBu's cell fractions), as in July.
- MuSiC and Bisque need a reference with more than one sample. Where the downsampled reference has one, they are
  reported as not applicable.
- A method that fails on a set is reported as failed. The one allowed retry is a task killed for memory or time,
  rerun once with more memory or time (DWLS additionally at 2 cores, the mixture panel's fix). No method setting
  changes.

**Readouts** (`omnideconv_bench/simbu_summary.py`, committed with this file; output
`data/deconvolution/simbu_bench_refs4all/summary/`, record copied to `deconvolution/simbu_record_refs4all/`).
- **Battery, per reference and method:**
  - pooled RMSE under bias and under nobias, and ΔRMSE = RMSE(bias) − RMSE(nobias);
  - pooled Pearson r under both, and Δr.
- **Battery, per method:**
  - mean ΔRMSE over the common set (references where all six methods scored both conditions); this is the
    corrector ranking;
  - also the mean and median over every reference the method scored, and the number of references with
    ΔRMSE > 0.
- **Bias driver, per reference and method:**
  - the type with the largest per-type ΔRMSE, its true mean, and its estimate under nobias and under bias;
  - the median per-type ΔRMSE of the other types.
- **mRNA content, per reference and type:**
  - the mean number of expressed genes (count > 0) per cell in the downsampled reference, ranked within the
    reference (1 = highest). This is the registered rank: it is the quantity SimBu's `expressed_genes` scaling
    uses;
  - the mean total counts per cell and its rank, reported alongside. They can disagree: in July's liver,
    hepatocytes rank 1 by counts and 2 by expressed genes;
  - for each driver, both ranks, and whether its estimate rises under bias (inflated) or falls (deflated).
- **Dose-response, per tissue, alpha and method:** pooled RMSE and r.
- **Dose-response, per tissue and method:**
  - RMSE at alpha 0 (the baseline);
  - the slope (RMSE(4) − RMSE(0)) / 4;
  - whether RMSE is non-decreasing over the six alphas.
- **Sanity checks:**
  - bias and nobias references are byte-identical;
  - the dose-response sets share one reference and one cell-fraction truth.

**Reading.** Each item checks a July statement on the current references.
1. **Corrector ranking.** July (10 tissues): Bisque +0.003, SCDC +0.016, MuSiC +0.018, DWLS +0.024,
   CIBERSORTx +0.024, BayesPrism +0.026.
   - The paper's core finding is **reproduced** if MuSiC's and SCDC's mean ΔRMSE on the common set are both below
     BayesPrism's and CIBERSORTx's. Otherwise it is named as not reproduced.
   - DWLS's place, which July read as bias-sensitive (about BayesPrism's ΔRMSE), is reported.
2. **BayesPrism degrades under bias.** July: ΔRMSE > 0 on every tissue. Reported: on how many of the 15.
3. **The driver is the highest-mRNA type.** July: "always the highest-mRNA type." Reported:
   - on how many references BayesPrism's driver has expressed-genes rank 1;
   - otherwise, its rank;
   - its total-counts rank alongside.
4. **Dose-response.** July: RMSE rises with alpha within each tissue, and DWLS has the best baseline but the
   steepest slope (correction is magnitude-limited). Reported, per tissue:
   - which methods' RMSE is non-decreasing in alpha;
   - whether DWLS has the lowest alpha-0 RMSE of the six methods;
   - whether DWLS has the steepest slope of the correctors (DWLS, MuSiC, SCDC).
5. **Guidance.** BayesPrism's drivers on the current references replace the per-tissue bias-driver table in
   `OMNIDECONV_RESULTS.md` for the 16 tissues.

A comparison with July is reported descriptively. July's sets map onto the current references (LIV → LIVER,
PBMC → BLOOD, WAT → WATSC, HRT → HEART, GAS and SKMVL → MUSCLE, KID → KIDNEY, HIP → HIPPOC, LNG → LUNG,
CTX → CORTEX). Most are different studies or relabelled references, so differences are not read as changes in a
method.

**What the battery does not say.** It measures only how each method responds to mRNA weighting when the
reference is the exact cells the mixtures came from. Accuracy against independent mixtures is the mixture panel's
question (`MIXTURE_PANEL_REFS4ALL_RESULTS.md`).

No threshold, reference, simulation size, seed, method setting or scoring will be changed after any result is
seen.
