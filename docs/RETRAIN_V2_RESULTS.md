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

## C3–C5 — pending
