# Retrain v2 — results

Scored against `docs/RETRAIN_V2_PREREGISTRATION.md` (registered 2026-10-08; its §6 logs every change
since). C1–C2 ask whether the retrained model learned rat expression without losing what GeneCompass
knew about human and mouse (pass/fail). C3–C5 ask whether the analyses built on it hold up at least as
well as with the April model (comparisons, reported as found).

**v2** = `data/models/rat_genecompass_finetuned_v2/models/rat_phase2_mixed_species/models`, the phase-2
final save (byte-identical to `checkpoint-591763`). Phase 1: job 11917121, 517,793 steps, ended
2026-10-15. Phase 2: job 11961217, 591,763 steps, ended 2026-10-23; the first attempt's log ends at step
474,012 (node failure, 2026-10-21), and the requeued job resumed from `checkpoint-470000`.

## C1, C2 — PASS (2026-10-23, job 12011656)

Registered probe `data/training/eval_probe_v2` (all three md5s match §6), 2,000 cells per species, masks
seeded 20261008. Every model scored the same cells and the same masked tokens (human 443,348, mouse
452,385, rat 317,619).

| Model | Species | Gene-ID loss | Value loss | Top-1 | Gene-ID loss vs base | Value loss vs base |
|---|---|---|---|---|---|---|
| base | human | 3.4444 | 1.728e-4 | 21.25% | | |
| base | mouse | 3.4215 | 1.953e-4 | 21.48% | | |
| v2 init | human | 3.5323 | 1.728e-4 | 20.58% | +2.55% | 0.00% |
| v2 init | mouse | 3.5177 | 1.953e-4 | 20.71% | +2.81% | 0.00% |
| v2 init | rat | 7.1332 | 1.330e-1 | 13.36% | | |
| April | human | 3.4495 | 1.724e-4 | 21.23% | +0.15% | −0.24% |
| April | mouse | 3.4545 | 2.018e-4 | 21.21% | +0.96% | +3.33% |
| April | rat | 5.3102 | 3.916e-4 | 18.80% | | |
| v2 after phase 1 | human | 3.4634 | 1.740e-4 | 21.10% | +0.55% | +0.67% |
| v2 after phase 1 | mouse | 3.4854 | 2.094e-4 | 20.97% | +1.87% | +7.24% |
| v2 after phase 1 | rat | 3.1565 | 5.421e-4 | 24.27% | | |
| **v2** | human | 3.4462 | 1.680e-4 | 21.22% | +0.05% | −2.81% |
| **v2** | mouse | 3.4667 | 2.002e-4 | 21.22% | +1.32% | +2.53% |
| **v2** | rat | 3.0590 | 2.477e-4 | 25.51% | | |

- **C1a PASS.** Every step of both phases is logged exactly once with a finite gene-ID and value loss
  (517,793 and 591,763 steps, from each phase's final `trainer_state.json`). Phase 2's failed first
  attempt logged to wandb up to step 474,012: its 4,013 superseded steps (≥ 470,000) are finite too, and
  its steps before 470,000 equal the final run's.
- **C1b PASS.** Rat probe, v2 vs v2 init: gene-ID loss 3.059 vs 7.133, value loss 2.48e-4 vs 0.133.
- **C2 PASS.** v2 vs base, limit +10%: human +0.05% (gene-ID) and −2.81% (value), mouse +1.32% and
  +2.53%.
- Not gated: April's rat numbers are context only (April was trained on scheme-A values; the probe holds
  scheme-B values). v2 after phase 1 is shown for context and is not a criterion. v2 init equals base on
  human and mouse value loss by construction (the same weights serve those tokens).

**Decision (§4):** C1 and C2 pass, so v2 is the working model for the downstream reruns.

Outputs: `data/results/retrain_v2/mlm/` (one `eval_mlm.py` JSON per model, `score_c1c2.txt`), scored by
`finetune/genecompass/score_prereg_c1c2.py`.

## C4 — gene embeddings (comparison; 2026-10-24, job 12013887)

Input word embeddings, cosine; implementation as logged in pre-registration §6 (2026-10-24).

**C4a — the 255 T4 (rat-only) genes whose symbol matches a human token.** Rank of the human namesake among
all 23,113 human tokens (1 = nearest). Null: 1,000 re-pairings of the 255 counterparts (mean and 95%
interval); p = 0.001 is the smallest possible.

| Model | Median rank (null) | Top-10 (null) | Top-100 (null) | p (median rank) |
|---|---|---|---|---|
| v2 init | 10,976 (11,504; 10,015–13,019) | 0.0% (0.05%) | 0.0% (0.5%) | 0.24 |
| April | 3,298 (12,466; 11,374–13,618) | 5.5% (0.1%) | 12.2% (0.6%) | 0.001 |
| v2 | 6,593 (13,030; 12,054–13,974) | 2.7% (0.06%) | 4.7% (0.5%) | 0.001 |

v2 init sits at the null, as expected for random rows. Both trained models place these genes far nearer
their human namesakes than chance; April more so than v2 (v2 ranks the namesake better for 90 of 255 genes).

**C4b — T3 ortholog retention (drift from init).** Cosine to the ortholog token each T3 row was cloned
from, and whether that ortholog is still the nearest token of its species.

| Model | T3a (1,351; human ortholog): median cosine / nearest / top-10 | T3b (892; mouse ortholog): median cosine / nearest / top-10 |
|---|---|---|
| v2 init | 1.000 / 100% / 100% | 1.000 / 100% / 100% |
| April | 0.810 / 100% / 100% | 0.801 / 96.9% / 99.8% |
| v2 | 0.705 / 75.1% / 82.6% | 0.673 / 55.3% / 71.6% |

v2's T3 rows moved further from their copied ortholog than April's. v2 trained on every row of each phase,
April on a quarter. The 2,243 T3 tokens share 1,094 ortholog tokens, so drift away from a shared ortholog
can also be paralog separation. Pre-registered as drift, not success.

Outputs: `data/results/retrain_v2/c4/` (`c4_summary.json`, per-gene `c4a_genes.tsv`, `c4b_genes.tsv`).

## C3, C5 — pending
