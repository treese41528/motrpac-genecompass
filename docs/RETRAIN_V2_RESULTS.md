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

## C3 — cell embeddings of the reference cells (comparison; 2026-10-24, jobs 12013932–12013934)

Do the models place real rat cells with their labelled cell type? The cells of 12 deconvolution
references (TESTES drops out: not in April's corpus), 620,058 cells present in both corpora; each model
embeds its own corpus's tokenization of the same cells. Implementation as logged in pre-registration §6
(2026-10-24). Join control: the 582 samples in both corpora have the same barcode sequence in April's
rebuilt rows as in v2, and every cell of the 12 references was found in both corpora, so none dropped out.

| Metric | References | April median | v2 median | Median v2 − April | v2 higher | Wilcoxon p |
|---|---|---|---|---|---|---|
| Cell type, kNN macro-F1 | 12 | 0.771 | 0.804 | +0.021 | 12 of 12 | 0.0005 |
| Cell type, kNN accuracy | 12 | 0.936 | 0.943 | +0.009 | 12 of 12 | 0.0005 |
| Cell type, silhouette | 12 | 0.045 | 0.062 | +0.040 | 12 of 12 | 0.0005 |
| Sample kNN accuracy within cell type (lower = less batch-driven) | 10 | 0.873 | 0.889 | +0.012 | 10 of 10 | 0.002 |

p = 0.0005 and 0.002 are the smallest two-sided values possible with 12 and 10 pairs.

v2 places the reference cells with their cell type better than April in every reference, on all three
cell-type metrics. It also tells samples apart better within a cell type, in all 10 references where that
is defined, so by the registered reading v2 is the more batch-driven of the two. The metric cannot
separate batch effects from real differences between samples, such as donor or condition, and v2
separates cells more finely on both of the axes measured here.

| Reference | Cells | Cell types | Macro-F1 April → v2 | Accuracy April → v2 | Silhouette April → v2 | Sample kNN accuracy April → v2 |
|---|---|---|---|---|---|---|
| BAT_GSE244451 | 28,246 | 6 | 0.710 → 0.789 | 0.950 → 0.958 | 0.080 → 0.091 | 0.990 → 0.994 |
| BLOOD_GSE285476 | 12,315 | 14 | 0.737 → 0.773 | 0.893 → 0.905 | 0.095 → 0.133 | — (1 sample) |
| CORTEX_GSE303115 | 12,933 | 11 | 0.814 → 0.819 | 0.960 → 0.962 | 0.248 → 0.313 | 0.897 → 0.901 |
| HEART_GSE280111_LV | 135,288 | 16 | 0.861 → 0.893 | 0.963 → 0.968 | 0.232 → 0.351 | 0.270 → 0.297 |
| HIPPOC_GSE295314 | 278,549 | 18 | 0.255 → 0.273 | 0.832 → 0.862 | −0.414 → −0.393 | 0.810 → 0.857 |
| HYPOTH_GSE248413_Y | 8,471 | 13 | 0.695 → 0.712 | 0.686 → 0.702 | 0.010 → 0.032 | — (1 sample) |
| KIDNEY_GSE240658 | 28,626 | 17 | 0.644 → 0.695 | 0.896 → 0.909 | −0.319 → −0.238 | 0.826 → 0.842 |
| LIVER_GSE220075 | 27,041 | 6 | 0.565 → 0.571 | 0.948 → 0.952 | −0.063 → −0.044 | 0.656 → 0.682 |
| LUNG_native_pooled | 46,653 | 28 | 0.811 → 0.830 | 0.927 → 0.933 | −0.152 → −0.061 | 0.958 → 0.961 |
| MUSCLE_GSE137869_Y | 10,763 | 5 | 0.920 → 0.933 | 0.945 → 0.955 | 0.303 → 0.383 | 0.849 → 0.877 |
| SMLINT_GSE272055 | 18,950 | 14 | 0.810 → 0.834 | 0.887 → 0.898 | −0.238 → −0.196 | 0.959 → 0.966 |
| WATSC_GSE137869_Y | 12,223 | 13 | 0.805 → 0.832 | 0.971 → 0.974 | 0.245 → 0.280 | 0.930 → 0.933 |

Silhouettes are negative in five references for both models: at the authors' label granularity those
cell types are not compact clusters in either embedding, while kNN still recovers them.

Outputs: `data/results/retrain_v2/c3/` (`c3_per_reference.tsv`, `c3_summary.json`, the joined cells in
`cells.tsv`, each model's embeddings in `emb/`).

## C5 — downstream analyses (comparison)

Each stage that consumes the model is rerun with v2 on April's inputs (the same pseudo-cells), at v2's
Stage 9 target sum of 6,400 (pre-registration §6). April's values are read from its on-disk outputs.

### Stage 9 — do the pseudo-cell embeddings still separate cell types? (2026-10-24, jobs 12013967, 12014306)

Silhouette by cell type, cosine distance, all pseudo-cells of each tissue (`embed_qc.py`). Control: v2's
rows are April's cells in April's order in all 14 tissues (`cell_id`, `sample`, `cell_type`).

| Tissue | Pseudo-cells | Cell types | April | v2 | v2 − April |
|---|---|---|---|---|---|
| BAT | 300 | 6 | 0.347 | 0.382 | +0.035 |
| BLOOD | 700 | 14 | 0.500 | 0.490 | −0.010 |
| CORTEX | 550 | 11 | 0.893 | 0.901 | +0.008 |
| HEART | 800 | 16 | 0.873 | 0.876 | +0.003 |
| HIPPOC | 900 | 18 | 0.569 | 0.573 | +0.004 |
| HYPOTH | 650 | 13 | 0.729 | 0.765 | +0.036 |
| KIDNEY | 850 | 17 | 0.617 | 0.661 | +0.044 |
| LIVER | 300 | 6 | 0.758 | 0.768 | +0.010 |
| LUNG | 1,400 | 28 | 0.576 | 0.600 | +0.024 |
| SKMGN | 250 | 5 | 0.936 | 0.930 | −0.006 |
| SKMVL | 250 | 5 | 0.882 | 0.879 | −0.003 |
| SMLINT | 700 | 14 | 0.661 | 0.660 | −0.001 |
| TESTES | 150 | 6 | 0.982 | 0.988 | +0.006 |
| WATSC | 650 | 13 | 0.694 | 0.697 | +0.003 |
| **Median** | | | **0.712** | **0.731** | **+0.005** |

v2 separates cell types at least as well as April: higher in 10 tissues (most in KIDNEY, HYPOTH and BAT),
lower in 4 by at most 0.010. The token-value medians follow each corpus as calibrated: liver 0.870 for
April (its corpus 0.869) and 0.920 for v2 (its corpus 0.9145; 6,400 is the rule's 6,352 rounded).

Outputs: `data/results/retrain_v2/c5/stage9_embed_qc.txt`; v2 embeddings in
`data/deconvolution/genecompass_input_v2/<tissue>/`.

### Aim-2 gate — which cell types carry a detectable exercise signal? (2026-10-24, jobs 12014932–12014935)

For each tissue × cell-type block of pseudo-cells (172 blocks in 14 tissues), a supervised PLS-1 probe on
the embedding gives three numbers. Exercise: the held-out AUC for trained vs control. Dose: Spearman ρ
with the training dose. Sex, the positive control: an AUC over 166 blocks, since TESTES is male-only.
Exercise hotspots are the blocks with BH q_sup_trained < 0.05 over all 172. Cell-type separation is the
Stage 9 silhouette above; the gate tables have no cell-type AUC. The C5 method uses 10 stratified 5-fold
splits and 10,000 permutations. It was chosen because with the registered single split and 1,000
permutations the seed decides the hotspot set (pre-registration §6, 2026-10-24). The registered method is
shown alongside.

Controls:
- April's 10-split trained probe equals the stability run's seed set 0 (job 12014932).
- For both models, the Augur exports from the 10-split runs equal the registered runs' in all 14 tissues.
- The PCA-50 and full-gene probes do not use the model. They give the same AUC and p for v2 as for April
  in all 172 blocks, under both methods.

| | Registered: April | Registered: v2 | C5 method: April | C5 method: v2 |
|---|---|---|---|---|
| Exercise hotspots (of 172) | 7 | 11 | 16 | **13** |
| Trained AUC, median / max | 0.5765 / 0.885 | 0.5690 / 0.890 | 0.5575 / 0.870 | 0.5610 / 0.887 |
| Dose ρ, median | 0.0405 | 0.0645 | 0.0290 | 0.0560 |
| Sex AUC, median | 0.7030 | 0.6840 | 0.6885 | 0.6820 |

Paired by block, v2 − April:

| | Registered: median difference / v2 higher | C5 method: median difference / v2 higher |
|---|---|---|
| Trained AUC (172 blocks) | −0.0040 / 82 (3 equal) | −0.0015 / 84 (1 equal) |
| Dose ρ (172) | −0.0105 / 83 | −0.0095 / 79 |
| Sex AUC (166) | 0.0000 / 75 (13 equal) | −0.0010 / 72 (10 equal) |

Under the C5 method the two models share 11 hotspots:
- BLOOD: basophils, classical and non-classical monocytes, ISG-expressing T cells, megakaryocytes, naive
  B cells, NK cells.
- KIDNEY: renal interstitial fibroblasts.
- LUNG: endothelial cells, macrophages.
- SKMGN: skeletal myocytes.

The blocks in only one set:

| Block | Hotspot for | April AUC (q) | v2 AUC (q) |
|---|---|---|---|
| SKMVL Endothelial cells | April | 0.870 (0.011) | 0.787 (0.056) |
| HEART Cardiac neurons | April | 0.831 (0.019) | 0.748 (0.089) |
| SKMVL Skeletal myocytes | April | 0.822 (0.027) | 0.779 (0.059) |
| LUNG Myeloid dendritic cells | April | 0.794 (0.039) | 0.774 (0.065) |
| HEART Monocytes | April | 0.791 (0.039) | 0.685 (0.296) |
| SKMGN Endothelial cells | v2 | 0.706 (0.234) | 0.887 (0.015) |
| HYPOTH Oligodendrocyte precursor cells | v2 | 0.729 (0.122) | 0.798 (0.038) |

What this shows:
- **Exercise.** Across blocks, v2 detects exercise about as well as April: the paired median difference
  is −0.0015, and v2 is higher in 84 of 172 blocks. v2 has 13 hotspots to April's 16.
- **April's five extra hotspots.** Four stay just above the threshold under v2 (q 0.056–0.089). They
  fall out at the cut rather than losing their signal. HEART Monocytes drops to an AUC of 0.685.
- **Where the sets differ.** v2 loses both SKMVL hotspots and both HEART hotspots. It gains SKMGN
  Endothelial cells and HYPOTH OPCs, which April ranks well outside its set.
- **Dose and sex.** v2's median dose ρ is higher (0.056 vs 0.029), but v2 is higher in only 79 of 172
  blocks, so there is no consistent dose gain. Sex is about the same (paired median −0.0010).
- **The registered method** gives the opposite count: v2 11, April 7. Each model's registered hotspots
  lie within its own C5 set, except v2's HEART Dendritic cells (registered q 0.034, C5 q 0.089). The
  registered count is the seed-dependent one that the C5 method replaces.

Context, not registered (C5 method):
- **Embedding vs PCA-50 of the same pseudo-cells.** The embedding's AUC is higher in 95 of 172 blocks for
  both models. The median paired difference is +0.0175 for April (Wilcoxon p 0.17) and +0.0185 for v2
  (p 0.073). Against the full scaled gene space it is +0.0040 (p 0.63) and +0.0180 (p 0.22). Neither
  model's embedding separates exercise significantly better than a plain PCA across blocks.
- **Canonical Augur** (random forest, run once per model on the same exports) gives an AUC of at least
  0.65 in every hotspot: April 16 of 16, v2 13 of 13. Its block AUCs follow the PLS-1 probe's (Spearman
  0.844 for April, 0.822 for v2, over 166 blocks; TESTES has too few control cells for Augur). Augur scores
  PCA-50 at least as high as the embedding: median 0.593 vs 0.556 for April (Wilcoxon p 0.022), 0.593 vs
  0.581 for v2 (p 0.11).

Outputs: `data/results/retrain_v2/c5/gate_summary.json`, from `gate_summary.py` beside it (written before
any gate output existed). The C5-method tables are in `gate_rcv10/` in each model's rat and human root. v2's
registered tables are in `data/deconvolution/genecompass_input_v2/` and `_human/`. April's registered
tables are its on-disk ones (2026-10-24).

### Stage 12 — how far does re-expression as human move each cell? (2026-10-24, job 12014308)

v2's pseudo-cells were re-expressed as human at 6,400 and embedded with species 0. Control
(`stage12_input_control.txt`): the same cells and the same tokens as April's human side in every tissue,
and every token's value equal to April's after rescaling 6,400 → 6,500 (|Δ| ≤ 2e-15), with no token order
contradicting April's values. 930 of 8,450 cells list exactly tied tokens in another order, and one
HYPOTH cell keeps the other of two tied genes at the 2,048 cut. The tokenizer's sort does not fix the
order of ties, and float rounding at another target sum moves where ties land.

Cosine between each pseudo-cell's rat-space and human-space embedding, same model, paired by `cell_id`
(`transfer_cosine.py`; all cells paired in every tissue):

| Tissue | April mean (p05) | v2 mean (p05) | v2 − April |
|---|---|---|---|
| BAT | 0.968 (0.960) | 0.960 (0.953) | −0.008 |
| BLOOD | 0.980 (0.970) | 0.984 (0.968) | +0.004 |
| CORTEX | 0.967 (0.942) | 0.954 (0.908) | −0.013 |
| HEART | 0.968 (0.933) | 0.961 (0.936) | −0.008 |
| HIPPOC | 0.965 (0.949) | 0.952 (0.924) | −0.013 |
| HYPOTH | 0.973 (0.964) | 0.963 (0.937) | −0.010 |
| KIDNEY | 0.976 (0.963) | 0.981 (0.965) | +0.005 |
| LIVER | 0.976 (0.965) | 0.977 (0.966) | +0.000 |
| LUNG | 0.980 (0.968) | 0.983 (0.972) | +0.003 |
| SKMGN | 0.969 (0.947) | 0.954 (0.919) | −0.015 |
| SKMVL | 0.968 (0.947) | 0.956 (0.923) | −0.012 |
| SMLINT | 0.975 (0.957) | 0.975 (0.962) | +0.000 |
| TESTES | 0.972 (0.958) | 0.975 (0.967) | +0.003 |
| WATSC | 0.978 (0.968) | 0.982 (0.971) | +0.004 |
| **Median** | **0.973** | **0.969** | **−0.004** |

v2 moves cells a little more in 7 tissues (most in SKMGN, SKMVL, CORTEX, HIPPOC) and a little less in 7.
Context, not a registered metric: in rat space, two different pseudo-cells of the same tissue have a
median cosine of 0.993 (April) and 0.994 (v2), and two of the same cell type 0.9995 and 0.9996. In every
tissue and for both models, a cell's median cosine to its own human-space embedding is lower than the
median cosine between two different cells of that tissue. So the cosine mostly measures how far apart
the species' regions of the embedding sit, not whether a cell keeps its identity.
The July reading of 0.979 as "the same cell lands near its rat embedding"
(`deconvolution/AIM2_DECONV_RESULTS.md` §4b) does not hold. Whether the exercise signal survives the move
is the transfer comparison below.

Outputs: `data/results/retrain_v2/c5/stage12_cosine_{april,v2}.tsv`, `stage12_input_control.txt`; v2
human-space embeddings in `data/deconvolution/genecompass_input_v2_human/<tissue>/`.

### Stage 12 — does the exercise signal survive re-expression as human? (2026-10-24, jobs 12014932, 12014934, 12014935)

The gate's probe is run on each model's human-space embedding of the same cells, by the same method as
its rat side (`compare_transfer.py`). Each block gets one status:
- PRESERVED: human-space trained AUC ≥ 0.65 with p < 0.05.
- WEAKENED: AUC ≥ 0.65 with p ≥ 0.05.
- LOST: AUC below 0.65.

| | Registered: April | Registered: v2 | C5 method: April | C5 method: v2 |
|---|---|---|---|---|
| Own hotspots: preserved / weakened / lost | 6 / 1 / 0 of 7 | 10 / 0 / 1 of 11 | 14 / 0 / 2 of 16 | **13 / 0 / 0 of 13** |
| All 172 blocks: preserved / weakened / lost | 33 / 19 / 120 | 33 / 18 / 121 | 33 / 13 / 126 | 34 / 12 / 126 |
| Sex AUC median, rat → human | 0.7030 → 0.6970 | 0.6840 → 0.6790 | 0.6885 → 0.6875 | 0.6820 → 0.6710 |
| Trained AUC, Spearman rat ~ human (172) | 0.759 | 0.644 | 0.762 | 0.616 |
| Context: sex AUC, Spearman rat ~ human (166) | 0.913 | 0.893 | 0.936 | 0.904 |
| Context: dose ρ, Spearman rat ~ human (172) | 0.742 | 0.572 | 0.733 | 0.638 |

The lost hotspots, rat → human AUC:
- April, C5 method: HEART Cardiac neurons (0.831 → 0.630) and HEART Monocytes (0.791 → 0.590).
- v2, registered: HEART Dendritic cells (0.843 → 0.602).

What this shows:
- **v2's hotspots all transfer.** Under the C5 method every v2 hotspot keeps its exercise signal in human
  space. April keeps 14 of 16; it loses its two HEART hotspots, which v2 does not call.
- **All blocks.** The preserved counts are about the same for the two models (33 and 34).
- **Block ordering.** v2's human-space AUCs follow its rat-space AUCs less closely than April's (Spearman
  0.616 vs 0.762; registered 0.644 vs 0.759). The same holds for dose, and slightly for sex. v2's hotspots
  carry over, but the other blocks change order more.
- **The script's sex check.** `compare_transfer.py` prints "transfer-validity: FAIL" for both models under
  both methods. The check requires a human-space median sex AUC of at least 0.70. It also failed on April's
  tables from before the 2026-10-24 seed recompute (rat 0.694 → human 0.692). The rat-space median is
  already 0.682–0.703, so the check fails whenever that median is just under 0.70 or drops slightly in
  transfer. Block by block, the sex AUC carries over closely (Spearman 0.89–0.94). The check is not a
  registered C5 criterion and is reported as printed.

Outputs: `transfer_comparison.{tsv,md}` in each model's `genecompass_input*_human/gate_rcv10/` (C5
method), in `genecompass_input_v2_human/` (v2 registered) and in `genecompass_input_human/` (April
registered). Summary values are in `data/results/retrain_v2/c5/gate_summary.json`.

### Stage 13, Module A — four perturbation checks (2026-10-24, jobs 12013970, 12014307)

Both models on the current SKMVL pseudo-cells (cell type "Skeletal myocytes", 50 cells), the frozen
80-gene panel plus Alb and Snap25, `perturb_cells.py --mode delete --targets`, scored by
`score_perturb_prereg.py` (pre-registration §6). 77 of the 80 panel genes are expressed in these cells
under both tokenizations; the other 3 are in 0 cells and drop out.

| Criterion | April | v2 |
|---|---|---|
| #1 Mef2c's cell shift vs expression-matched genes (> 90th percentile) | 96th of 27 — PASS | 100th of 26 — PASS |
| #4 shift vs mean expression value, Spearman (> 0) | ρ = 0.74, p = 1.3e-14 — PASS | ρ = 0.76, p = 1.3e-15 — PASS |
| #2 Mef2c's top-25 targets in the curated muscle/mitochondrial set (p < 0.05) | 4/25 (set 46 of 2,612 expressed genes), p = 8.1e-4 — PASS | 0/25 (48 of 2,597), p = 1.0 — **FAIL** |
| #3 Alb, Snap25: in 0 cells or below the matched-null median | both in 0 cells — PASS | both in 0 cells — PASS |

April passes all four. v2 passes #1, #3 and #4 and fails #2.

The curated set is not neutral between the two models. It was written on 2026-07-03 (ffbebff), after
April's smoke run, and includes "the muscle/mito genes the smoke surfaced" (`score_perturb_prereg.py`):
that run's top Mef2c targets were Perm1, Mlip, Prr33, Ndufa6, Mccc2 and Stbd1. Three of April's four hits
(Perm1, Ndufa6, Stbd1) are among them; the fourth is Nmnat1. v2's top 5 are Afg3l1, Cep131, Tns2, Ppfia1
and Actr8 (full list in `results.tsv`). The registered verdict stands as scored. April's July result on
the pre-rebuild cells ("Skeletal muscle") was 98th of 52, ρ = 0.70, 7/25 (p = 2e-7).

Outputs: `data/results/retrain_v2/c5/module_a/{april,v2}/` (`results.tsv`, `report.md`, `report.json`).
