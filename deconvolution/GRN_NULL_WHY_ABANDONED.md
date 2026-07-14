# The permutation null was ABANDONED (2026-07-13) — it cannot work for this design

**Goal:** calibrate the `|z| >= 2` "confident edge" threshold with an empirical FDR,
`FDR(t) = P_null(|z|>=t) / P_obs(|z|>=t)`, by shuffling the trained/control labels.

**Why it is invalid.** `label` is a *deterministic function* of `week`:

| week | control | trained |
|---|---|---|
| 0 | 140 | 0 |
| 1 / 2 / 4 / 8 | 0 | 140 each |

Control **is** week 0; trained **is** weeks 1/2/4/8 pooled. Dose and label are perfectly confounded
by design. Consequently:

- A **within-week** stratified permutation is the *identity* — there is nothing to shuffle.
- Any permutation that *does* break the label necessarily mixes weeks into the sham-control arm. The
  observed control arm is dose-**homogeneous** (all week 0); every permuted control arm is dose-
  **heterogeneous**. The 10-cell control arm also carries 4x the denominator weight (1/10 vs 1/40),
  so `boot_sd` differs systematically between null and observed (measured ratio 0.93–1.08), and the
  **sign of the bias flips by tissue**.
- `FDR = P_null/P_obs` requires null and observed `z` to be identically distributed under H0. Here
  they are not, **by construction**. The resulting FDR is biased by a median ~16.5% with no
  consistent sign, so it is not even correctable.

An early partial run suggested ~70-88% FDR at `|z|>=2`. **That number is NOT trustworthy** and must
not be quoted — it is contaminated by exactly this dose/label confound.

**Two further defects in the run as launched** (worth knowing if anyone retries):
1. **Seeding collapse.** `default_rng(permute_seed)` used the same seed for every cell type, and the
   sex vector is in identical dataset row-order within a tissue — so all cell types of a tissue got
   the *same* sham-control rats. "3 permutations x 14 cell types" was only **12 distinct label draws**.
2. **3 draws cannot estimate a tail.** The ~250k edges per run are nowhere near independent
   (TF-clustering alone gives a measured design effect of ~105 -> effective n ~2,577, so the naive
   binomial SE understates the truth by ~10x).

**What to do instead.** `|z|` is still defensible as a *ranking* statistic — it beats `|delta|` at
matched set size for target selection (median rat/human rho 0.451 vs 0.410, and 0.026 for
smallest-|delta|). But it is **not** FDR-calibrated, and it must **not** be used for hub strength
(see `[[project_grn_hub_statistic]]`). A valid null would need a design that is not dose-confounded —
e.g. permuting *within* the trained arm across dose levels, which tests a different (and weaker) null.
