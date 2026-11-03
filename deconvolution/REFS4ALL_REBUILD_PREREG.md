# Clean rebuild of the 16-tissue default: pre-registration (2026-11-01, before any job ran)

Results: `REFS4ALL_REBUILD_RESULTS.md` (2026-11-02). The criteria below are unchanged.

**Question.** Does the committed clean path (`DECONV_REBUILD_RUNBOOK.md` §0.6–0.7, `config/profiles/refs4all_rebuild.yaml`)
reproduce the default tree `results_refs4all/`? That tree was assembled from links to three earlier runs
(July, refs4, in place).

**Scope.**
- **What is rebuilt:** Stages 8–14 from the production references (`references_v3/`, `references_v4/`, unchanged;
  `--force` reference builds would overwrite them), the prepared MoTrPAC bulk, and the v2 model
  (`deconvolution.model_checkpoint`).
- **Outputs:** fresh `_refs4all_rebuild` folders only, so the default tree is not touched.
- **Settings:** BayesPrism with 16 cores, as production, and its default Gibbs seed of 123. The IHW seed is
  `de_seed` = 1.
- **Not covered:** reference builds and joint re-annotation (§0.4–0.5), and the notebooks.

**Comparisons and criteria** (fixed now):

| Layer | Compared | IDENTICAL | EQUIVALENT | DIFFERS |
|---|---|---|---|---|
| Stage 8 θ | `estimated_fractions.csv`, all 16 tissues | max abs diff ≤ 1e-9 | ≤ 0.005 | otherwise |
| Stage 8 Z | `pred_z` per cell type | max rel diff ≤ 1e-9 | ≤ 1e-3 | otherwise |
| Stage 9 embeddings | per-cell CLS vectors | equal | cosine ≥ 0.9999 for every cell (GPU fp nondeterminism) | otherwise |
| Stage 11 gate | blocks / analysable / hotspots | 197 / 151 / the same 9 hotspots, AUCs within 1e-9 | same 9 hotspots | different set |
| Stage 10 DE | dose-significant genes | 53,084, same genes | within ±1% and Jaccard ≥ 0.99 | otherwise |
| Stage 12 transfer | hotspots preserved | 9/9 | 9/9 | fewer |
| Stage 13 GRNs | networks built | 6, edges equal | 6, edge Δ Spearman ≥ 0.99 | otherwise |
| Stage 14 | purity sweep r per tissue | equal | within 0.01 | otherwise |

**Verdict.** The rebuild REPRODUCES the default if every layer is IDENTICAL or EQUIVALENT.
- If anything DIFFERS, the cause is traced and reported. The default is not changed to fit the rebuild.
- If BayesPrism differs at the θ level, the first suspect is the Gibbs RNG and its dependence on the core count.
  BAT, HEART and HYPOTH come from July runs that recorded no provenance, so their core count is not on record.
