# Reference relabel and analysable-block rule — pre-registration

Registered 2026-10-27, before the label audit below is computed and before any reference is rebuilt. It
fixes how the deconvolution references' cell-type labels are audited and corrected, and which
(tissue × cell type) blocks are analysed downstream. Changes after this date go in §9, dated, before the
result they could affect is read.

## 1. Why

**Near-empty blocks.** In the current v2 run (July references, `genecompass_input_v2/pseudobulk_de/de_summary.tsv`)
49 of 172 DE blocks have a median per-sample library (the sum of BayesPrism's Z over genes) below 50
reads; 70–88% of their values are zero. All 172 are marked `status = ok` and feed the global IHW fit,
which learns its weights by tissue: they contribute 452,390 of 2,099,387 tests and 217 of 28,999 dose calls.
`run_pseudobulk_de.R` records `median_libsize` and `frac_zero` but never acts on them.

**Their main cause is label drift between samples.** References built from the corpus are labelled per
sample: each sample is clustered on its own (Leiden), and each cluster gets one consensus label from
PanglaoDB and ScType resolved by an LLM (`analysis/consensus_annotations.py`; `build_reference.py` joins
barcode → cluster → label). Nothing makes two samples name the same cells the same way. Of the 49
near-empty blocks, 32 have a label that occurs in only one sample of a multi-sample reference; of the 54
such single-sample labels, 32 are near-empty. KIDNEY (4 samples) shows the pattern: sample 8 calls its
β-intercalated cells "Beta-intercalated cells" (37) and has no "β-intercalated cells", which the other three
samples use (57–103); sample 4 has a second fibroblast cluster, "Renal interstitial fibroblasts" (150), next
to its "Renal fibroblasts" (199); sample 8 splits immune cells into "Myeloid cells" (307), "Effector CD8+ T
cells" (58) and "Immune cells" (30), which the other samples call "Immune cells" (103–155); sample 8 has
"Renal vesicle cells" (269), an embryonic structure; sample 6 has "Mesangial cells" (180); sample 15 has
"Intercalated cells" (43). The same cells split across names give BayesPrism collinear profiles and leave
each name a thin share of the reads.

**One exercise hotspot is affected.** KIDNEY "Renal interstitial fibroblasts" (sample 4 only; median
library ~15 reads) is one of v2's 13 hotspots. Its pseudo-cells were scaled from ~15 reads to the 6,400
target before tokenizing.

The other 17 near-empty blocks are not label artifacts: BLOOD's lymphocytes (a one-sample PBMC reference
against whole-blood bulk, §7), HYPOTH's neuron subtypes (a one-sample reference), and rare or low-mRNA
types such as LUNG Club cells.

## 2. What has been seen before registration

Only: the `median_libsize` / `frac_zero` distribution of the 172 v2 DE blocks; the count of hotspots (1),
dose calls (217) and tests among blocks under 50, 100 and 1,000 reads (1 / 1 / 1 hotspots; 217 / 248 / 552
calls); the cross-tabulation of single-sample labels against near-empty blocks above; KIDNEY's
label × sample cell counts. No profile similarity, marker or merge decision has been computed.

## 3. Scope

| Reference | Labels from | Audit |
|---|---|---|
| KIDNEY, LIVER, CORTEX, WATSC, MUSCLE (SKMGN + SKMVL), SMLINT, LUNG, HIPPOC | per-sample consensus | full rule (§4) |
| BAT, HEART | the authors' deposited labels (July route A) | metrics reported; only MERGE applies |
| TESTES | per-cell paper-panel scoring, one fixed class set | metrics reported; only MERGE applies |
| BLOOD, HYPOTH | one sample | not auditable for drift; BLOOD in §7 |

VENACV (dropped 2026-07-16) and the unbuilt references are out of scope.

## 4. The audit

**Profiles.** For each reference, each (sample, label) group's profile is the sum of its raw counts over
cells, as log2(CPM + 1). (`build_reference.clean_cells` already drops clusters under 20 cells, so every
group has at least 20.) Genes: mean log2(CPM + 1) ≥ 1 over the reference's groups, then the 2,000 with the
highest variance across groups. Each gene is centred by its mean over groups (the housekeeping baseline
makes uncentred profiles all correlate, as CDSeq showed on BLOOD). Similarity is Pearson's r between
centred profiles.

**Self-similarity, the yardstick.** For each label present in at least 2 samples, r_self = the median r
over its profiles from different samples. τ = the 10th percentile of r_self over the reference's labels;
if a reference has fewer than 5 such labels, τ is the 10th percentile pooled over all consensus-labelled
references.

**Metric check.** In each reference the median r_self must exceed the median r between profiles of
different labels from different samples. A reference that fails goes to joint re-annotation (§5) instead
of the rule.

**Audit set.** Labels present in at most half of the reference's samples (n = 2 or 3: one sample).
Candidates for a label L: every other label present in at least 2 samples.
r_cross(L, T) = the median r between L's profiles and T's profiles from different samples.
r_within(L, T), the median over samples holding both, is reported but does not decide.

**Decision for each L in the audit set** (T* = the candidate with the highest r_cross):

| Outcome | Condition |
|---|---|
| MERGE into T* | r_cross(L, T*) ≥ τ and at least 0.05 above the next candidate |
| REVIEW | r_cross(L, T*) ≥ τ but within 0.05 of the next: decided from the markers, reason recorded; if the markers do not decide, MERGE into T* |
| KEEP | r_cross(L, T*) < τ, at least 50 cells, and a cell type that plausibly occurs in the tissue |
| DROP | otherwise (distinct but under 50 cells, or implausible in the tissue); its cells leave the reference |

For BAT, HEART and TESTES only MERGE is applied; anything else is KEEP.

**Markers, for the record and REVIEW.** For each audited label, its 10 genes with the highest log2 fold
change against all other groups of the same sample(s), with RGD symbols.

## 5. Applying it

The decisions go into a committed map, one row per (reference, sample, old label): new label, outcome, the
numbers that decided it. `build_reference.py` applies the map when it builds; the shared consensus
annotation files in `data/training/cell_annotations_consensus/` are not rewritten (the training corpus uses
them). References with any changed label are rebuilt and must pass `reference_qc.py`.

Fallback, for a reference that fails the metric check or ends with more than half its audit set at
REVIEW: joint re-annotation (all samples clustered together, one label per cluster from markers). Not
started without a §9 entry.

## 6. Analysable blocks (DE and the Aim-2 gate)

A block is analysed when its median per-sample library (the sum of Z over genes) is at least **1,000**
reads; the threshold sits in the gap between the two modes of the distribution (§2: 65 blocks under 1,000,
4 between 1,000 and 10,000, 103 above). In `run_pseudobulk_de.R` a block under it stops with
`status = "SKIP: degenerate (median libsize …)"`, keeps its `median_libsize` and `frac_zero` in
`de_summary.tsv`, and stays out of the global IHW fit; `de_methods.tsv` records the threshold and the count.
In the gate, `corroborate_summary.py` applies the same rule before the BH correction, so a block under it
cannot be a hotspot. Sensitivity, reported alongside: 100 reads, and no threshold.

## 7. BLOOD

BLOOD's reference has no erythroid or granulocyte profile (CDSeq: ~80% of the whole-blood mRNA mass in
components matching nothing in the PBMC reference), so the label rule cannot fix it. Its reference is
decided separately, from a search for rat references with those compartments; the choice, and how it will
be validated, go in §9 before BLOOD is deconvolved again. Until then BLOOD is rerun under §6 only, and its
results stay flagged.

## 8. What is rerun and reported

With the rebuilt references and v2 (the working model; April is not rerun): Stage 8 for each tissue whose
reference changed; Stage 9 embeddings for those tissues; the gate (10 splits × 10,000 permutations) with
hotspots re-derived over all blocks; Stage 10 DE and the positive controls; Stage 12; Stage 13 for the
hotspots; Stage 14 steps 2–3. Reported against the current July-reference v2 results: blocks analysed,
hotspots (primary rule and sensitivities), transfer status, positive-control tiers, enrichment counts.
The retrain-v2 C5 comparison (April vs v2 on the July references) stays as scored.

## 9. Change log
- 2026-10-27 — registered.
- 2026-10-27 — Audit computed (jobs 12033639, 12033678; `audit_reference_labels.py`), results in
  `REFERENCE_LABEL_AUDIT.md`. Nothing has been applied, rebuilt or deconvolved. Three findings the rule
  does not handle: HIPPOC's labels barely match themselves across samples (median r 0.12, τ −0.07), so its
  merges follow noise; two single-sample labels can't merge with each other (KIDNEY "Renal interstitial
  fibroblasts" with "Renal vesicle cells", r 0.61; WATSC "Pre-B" with "Naive B", r 0.53); and many labels
  carry names their markers contradict, including the KIDNEY hotspot (Havcr1+ injured tubular cells). §4
  also does not say whether plausibility is judged by name or by cells. Pending: applying the rule with
  amendments, or joint re-annotation (§5's fallback) for every consensus-labelled reference, logged here
  before either starts. BLOOD: candidate references found and checked (GSE239996 sorted blood neutrophils,
  GSE306764 rat whole blood scRNA, GSE335156 5-rat PBMC); its reference and validation checks are to be
  logged here before BLOOD is deconvolved again.
