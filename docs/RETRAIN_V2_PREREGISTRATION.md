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
- 2026-10-09 — registered probe built (job 11916410) from the rebuilt v2 mix (OMIX767 included; corpus
  6,803,206 cells / 584 samples, mix 33,138,703 rows, verify_mix 21/21 PASS): `data/training/eval_probe_v2`,
  2,000 cells per species, seed 20261008. md5 (also in `data/training/eval_probe_v2.md5`):
  `90aa2d9030f4b769a3388a0b1ea808f9  data-00000-of-00001.arrow`,
  `c6b1b65c0b5e667f07333ff624d2e3ff  dataset_info.json`,
  `76fbde1fbb8cf2d289ec81efeaedf628  state.json`. Fixed before phase 1 starts.
- 2026-10-22 — TESTES inputs corrected; logged before any C3–C5 result exists (phase 2 still training,
  no v2 model evaluated). The TESTES deconvolution reference kept OMIX767's Rnor_6.0 gene IDs while the
  MoTrPAC bulk was lifted to Ensembl 115, so 587 bulk genes (2.3% of TESTES bulk counts) had no reference
  partner. The reference is rebuilt through the bulk's own liftover map
  (`deconvolution/reference/motrpac_bulk_liftover.tsv`) from the same cells and labels, and TESTES is
  deconvolved again. So that C5 compares April and v2 on the same inputs, April's TESTES rows are
  recomputed on the fixed pseudo-cells with the April checkpoint and April's Stage 9 settings (target sum
  6500, April medians and token dictionary) before C5 is scored: the Stage 9 embedding; the Aim-2 gate rows
  (`pheno_merge_test`, `subspace_probe`, `pca_control`), with hotspots re-derived by `corroborate_summary`
  over all blocks; the Stage 12 human embedding, `subspace_probe` and `transfer_comparison`. Recomputed rows
  use the current code (crc32 block seeds). Other tissues keep their on-disk April rows. Superseded TESTES
  outputs are kept in `data/deconvolution/_testes_rawid_backup_20260716/`. TESTES is not in Stage 13.
- 2026-10-22 — that recompute is done (jobs 12007286, 12007401, 12007434), still before any v2 C3–C5
  result. Controls on April's own inputs: the April embedding of the old TESTES pseudo-cells is reproduced
  exactly; `pheno_merge_test` reproduces every non-TESTES row; `corroborate_summary` and `compare_transfer`
  reproduce every value of the 172 current blocks. April's `transfer_comparison.tsv` also held 33 VENACV
  rows left from before VENACV was dropped (2026-07-16), all `no_human_block` with no values. The
  regenerated table has the 172 current blocks only, so April's "blocks compared" goes from 205 to 172 and no
  hotspot or status count changes. April after the recompute: 15 exercise hotspots, as before, with no block
  changing hotspot status; TESTES has none.
- 2026-10-24 — April's seed-dependent Aim-2 gate and Stage 12 rows are recomputed with the current code,
  logged before any v2 C3–C5 result exists (C1/C2 scored 2026-10-23, see `docs/RETRAIN_V2_RESULTS.md`; no
  v2 downstream output exists). April's on-disk `subspace_probe` and `pca_control` rows were computed in
  July with per-block seeds from Python's salted `hash()`, so they cannot be reproduced; since 2026-10-05
  (3a0165f) the code uses stable crc32 seeds, and v2's rows will too. On TESTES the seed change alone moved
  the exercise AUCs by 0.048 per block on average. Recomputed on April's on-disk embeddings with July's
  settings (`subspace_probe` 1,000 permutations, 5 folds; `augur_prep` 500 permutations, PCA-50): rat
  `subspace_probe` and `pca_control` for all 172 blocks, with hotspots re-derived by `corroborate_summary`;
  the human-space `subspace_probe`; `transfer_comparison`. Controls: every seed-independent column (n and
  the eta² values) must equal April's on-disk value, and TESTES's rows, already crc32-seeded on 2026-10-22,
  must be reproduced exactly. `pheno_merge_test` and canonical Augur are deterministically seeded and are
  kept (TESTES has no Augur rows: 5 control pseudo-cells per cell type, below Augur's 10-cell minimum).
  April's human-space Augur and corroboration tables date from 2026-06-25, before the July reference
  rebuild; no C5 metric uses them and they are left as they are. Superseded tables are kept in
  `data/deconvolution/_april_hashseed_backup_20261024/`.
- 2026-10-24 — C4 implementation (`finetune/genecompass/eval_gene_embeddings.py`), fixed before any C4
  number is computed. C4a: the 255 genes are the T4 tokens whose rat symbol, upper-cased, is the symbol of
  a human (ENSG) token in GeneCompass's `Gene_id_name_dict_human_mouse.pickle`; each matches exactly one
  human token. Ranks are among all 23,113 human tokens. The null re-pairs by permuting the 255 human
  counterparts across the 255 rat tokens (1,000 permutations, seed 20261008), so it keeps the counterparts'
  own neighbourhoods; p = (1 + null permutations at least as good) / 1,001. C4b: each T3 token's ortholog is
  the token its init row was cloned from (`rat_model_init.t3_ortholog_token_ids`). The 892 T3b orthologs
  are mouse tokens, so "rank among all human tokens" cannot apply to them: T3a orthologs are ranked among
  the human tokens as registered, T3b orthologs among the 27,443 mouse tokens, and the two are reported
  separately.
