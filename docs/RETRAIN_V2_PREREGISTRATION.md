# Retrain v2 — pre-registered evaluation

**Registered:** 2026-10-08, before any v2 training step. **Status:** locked — changes after
training starts must be logged in §6 with a date and reason, never edited in place.

## 1. Question and scope

The project asks whether a GeneCompass model extended to rat can carry MoTrPAC rat endurance-training
signals into a human-comparable space. The model is a means; it succeeds when (a) it learned rat
expression without losing what GeneCompass knew about human and mouse, and (b) the downstream analyses
built on it hold up at least as well as with the April model.

There is **no held-out study set** (decided 2026-10-08): the rat corpus is small, and every study usable
as a deconvolution reference must stay in training. Criteria 3–5 therefore use labels or analyses that
training never sees (training is self-supervised; no cell-type, sex or exercise label enters it), and no
MoTrPAC data enters training (checked 2026-10-08: the rat corpus is 80 GEO single-cell studies + OMIX767;
MoTrPAC bulk lives only in `data/motrpac/`).

| Model | What it is |
|---|---|
| **base** | GeneCompass_Base (human + mouse), `vendor/GeneCompass/pretrained_models/GeneCompass_Base` |
| **April** | `rat_phase2_mixed_species/checkpoint-147941` — trained on 25% of its mix per phase (sampler bug), old priors, scheme-A medians, uncleaned corpus |
| **v2 init** | `data/models/rat_genecompass_init_v2` — fixed priors, before training |
| **v2** | phase-2 final model of `finetune/genecompass/configs/rat_finetune_v2.yaml` |

Criteria 1–2 are **pass/fail** checks on training itself. Criteria 3–5 are **comparisons** against
April, reported in full whichever way they fall; they have no pass marks.

## 2. Fixed probe set (criteria 1–2)

`finetune/genecompass/eval_mlm.py build-probe`: 2,000 distinct cells per species drawn from the v2
training mix (seed 20261008; distinct = distinct `(species, source_row)`), saved once and never redrawn.
Masks are GeneCompass's own collator (15%; 80/10/10), seeded per batch, so every model sees the same
masked positions. Metrics per species: gene-ID loss (cross-entropy over masked tokens), value loss
(MSE over masked tokens), and masked top-1 gene-ID accuracy, all token-weighted.

These are in-sample cells for every model (base trained on the human/mouse corpora; v2 on all three).
That is deliberate: criterion 2 asks whether v2 *kept* what base knew on base's own data.

## 3. Criteria

### C1 — GeneCompass's own pretraining losses (pass/fail)
- **C1a.** Every logged training step of both phases has finite `id_loss` and `value_loss`.
- **C1b.** On the rat probe, v2 has lower gene-ID loss **and** lower value loss than v2 init.
- Reported, not gated: training curves per phase; the same probe metrics for April (April was trained on
  scheme-A values, so its rat numbers on scheme-B inputs are context, not a target).

### C2 — no forgetting of human and mouse (pass/fail)
- On the human and on the mouse probe, v2's gene-ID loss and value loss are each **at most 10% above
  base's** on the same cells and masks (four comparisons; all must hold).
- Reported, not gated: April on the same probes; masked top-1 accuracy.

### C3 — cell embeddings (comparison)
- Data: cells of the deconvolution reference datasets with their author / consensus cell-type labels
  (the `cells_meta.tsv` of each reference in `deconvolution/tissue_references.yaml`), joined to tokenized
  rows by `sample_id` + barcode. Each model embeds the tokenization it was trained on (April: v1 corpus;
  v2: v2 corpus); only cells present in both are scored.
- Metrics, per reference: 5-fold kNN (k = 10, cosine) macro-F1 and accuracy for cell type; silhouette
  (cosine) by cell type; study/sample mixing as kNN accuracy for `sample_id` within the same cell type
  (lower = less batch-driven).
- Reported: per-reference table, median across references, v2 − April with a paired Wilcoxon over
  references.

### C4 — gene embeddings (comparison, with nulls)
- **C4a. T4 symbol-matched genes (255).** Rat-only tokens that share a symbol with a human token
  (decision record, 2026-10-06; no warm start). For each, the rank of its human counterpart among all
  human tokens by cosine of the input word embeddings. Statistic: median rank and top-10 / top-100 hit
  rates. Null: the same statistic over 1,000 random re-pairings of the 255 rat tokens to human tokens.
  Reported for v2 init (expected ≈ null), April and v2.
- **C4b. T3 ortholog retention (2,243).** Cosine between each T3 rat token and its ortholog's token, and
  the ortholog's rank among all human tokens. At init these are near-identical by construction, so this
  is reported as **drift**, not success.

### C5 — downstream analyses (comparison; the real judge)
Rerun with v2 the pipeline stages that consume the model, after recalibrating Stage 9's pseudo-cell
target sum for the v2 corpus, and compare with April on the same inputs:

| Stage | Metric |
|---|---|
| 9 (pseudo-cell embedding) | per-tissue cell-type silhouette of pseudo-cell CLS embeddings |
| Aim-2 gate | cell type / sex / exercise separability (AUC), number of exercise hotspots |
| 12 (transfer to human) | rat↔human-re-expressed cosine per tissue; cell types whose exercise signal transfers |
| 13 Module A | the four pre-registered perturbation checks (`score_perturb_prereg.py`) |

April's values are re-read from the on-disk April outputs at evaluation time, not from documents.

## 4. Decision rule
- C1 or C2 fails → v2 is not adopted; the failure is investigated before anything downstream is rerun.
- C1 and C2 pass → v2 becomes the working model for downstream reruns; C3–C5 are reported as found,
  including any where v2 is worse than April.

## 5. What is fixed before training
Probe set file + its md5, the seed, this document's commit. The probe set is built from the final
v2 mix (after the OMIX767 rebuild) and its md5 recorded in §6 before phase 1 starts.

## 6. Change log
- 2026-10-08 — registered.
- 2026-10-08 — evaluator smoke test (job 11911214) on a probe drawn from the PRE-OMIX767 mix (not the
  registered probe). Human / mouse gene-ID loss: base 3.444 / 3.422, v2 init 3.532 / 3.518, April
  3.450 / 3.455; value loss: base 1.73e-4 / 1.95e-4, April 1.72e-4 / 2.02e-4 (v2 init = base exactly on
  human/mouse values, as expected: only shared weights are used). Rat: v2 init 7.141 / 0.120, April
  5.318 / 4.7e-4. April would pass C2 (+0.2% to +3.3%). v2 init's ~2.5% higher human/mouse ID loss comes
  from the 4,717 extra softmax classes. Values are small because GeneCompass orders genes by value, so a
  masked value is bracketed by its neighbours'. No criterion was changed after seeing these.
