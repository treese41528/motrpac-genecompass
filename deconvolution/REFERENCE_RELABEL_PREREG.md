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
- 2026-10-27 — **Joint re-annotation chosen** (option B in `REFERENCE_LABEL_AUDIT.md` §4) for the eight
  consensus-labelled references, replacing §4–5's merge rule for them; registered in §10 below, before any
  joint clustering was run. The annotation procedure was checked as part of this decision (§10.1): its
  wrong names come from the procedure itself, not from chance.

## 10. Joint re-annotation (registered 2026-10-27)

### 10.1 Why: what the annotation procedure did

The consensus labels come from three steps, all per sample (`analysis/annotate_celltypes.py`,
`annotate_sctype.py`, `consensus_annotations.py`):

1. **Per-sample clustering.** Each sample was clustered alone (Leiden, resolution 0.8). Cluster numbers and
   boundaries mean nothing across samples.
2. **Two automatic annotators per cluster.**
   - PanglaoDB: `score_genes` against every PanglaoDB cell type of every organ (the tissue hint only adds
     organ-specific types to the global set), argmax with no abstain. This is where kidney endothelium was
     called "Glutaminergic neurons", lung cells "Ionocytes", jejunal cells "Merkel cells".
   - ScType: tissue lists looked up from `tissue_normalized` by exact or substring match. "brain"
     (HIPPOC), "skeletal muscle", "white adipose tissue", "proximal jejunum" and "peripheral blood
     mononuclear cells" match nothing and fall back to the **immune-system list only**, so every HIPPOC,
     MUSCLE, WATSC, SMLINT and BLOOD cluster, neurons and myofibres included, was scored only against
     immune cell types. HIPPOC's samples also carry `motrpac_tissue = cortex`.
3. **LLM reconciliation.** Claude Haiku received each unique (PanglaoDB label, ScType label, tissue) triple
   with the top 5 markers of one representative cluster from anywhere in the corpus, and returned one name
   per triple. Names are keyed by the label pair, not the cluster, so two samples' clusters of the same cells
   get different names whenever the PanglaoDB argmax differs (KIDNEY: "Myofibroblasts" + ScType "Renal
   vesicle" → "Renal interstitial fibroblasts" in sample 4; "Transient cells" + the same ScType label →
   "Renal vesicle cells" in sample 8). With the instruction to correct implausible labels, it also invented
   names neither annotator gave (`consensus_source = corrected`): 68% of HIPPOC's cells, 55% of MUSCLE's,
   43% of WATSC's. A failed batch silently fell back to the PanglaoDB label (11% of SMLINT's cells). No step
   checked a final name against the cluster's own markers.

So relabelling within the old clusters (§4–5) cannot fix the names, and per-sample clusters cannot be made
to agree. The labels are re-derived jointly.

### 10.2 Scope

KIDNEY, LIVER, CORTEX, WATSC, MUSCLE (SKMGN = SKMVL), SMLINT, LUNG, HIPPOC. Not BAT and HEART (authors'
labels) or TESTES (paper panel). HYPOTH (one sample, ScType had brain markers, 1% "corrected") and BLOOD
(reference under review, `BLOOD_REFERENCE_PROPOSAL.md`) are not re-annotated now; both carry the same
procedure and can go through the same code later with a panel each.

### 10.3 Method and rule (`deconvolution/joint_reannotate.py`, panels `reference_marker_panels.yaml`)

- **Cells.** Every QC cell of the reference's samples (the `expect` sample list of
  `tissue_references.yaml`), loaded with the production loader and gene join; LUNG as `build_lung_pooled.py`
  (outer join, genes in ≥ 10 cells). The old labels play no part.
- **Clustering.** normalize_total 10⁴ + log1p; 3,000 HVGs (seurat_v3 on counts, `batch_key = sample`);
  scale (max 10), PCA 50; Harmony on sample (harmonypy 0.0.10) when there is more than one sample; kNN 15 on
  30 components; Leiden (igraph, resolution 1.0, seed 0). Clusters under 20 cells leave the reference.
- **Panels.** Written from canonical rodent markers before clustering, 3–6 genes each, organised as
  top-level classes (each with a compartment) and optional sub-classes. Symbols resolve to the reference's
  gene IDs through Ensembl 115, the corpus gene universe, then RGD; panels with fewer than 2 genes found
  are not scored (listed in `panel_genes.tsv`).
- **Score.** Each panel gene's mean log-normalised expression per cluster, min-max scaled across clusters;
  a panel's score is the mean over its genes.
- **Rule.** Top-level: the best class is accepted if its score is ≥ **0.5** and ≥ **0.15** above the
  runner-up; if another class from a different compartment also scores ≥ 0.5 the cluster is a possible
  doublet. An accepted class descends into its sub-classes by the same 0.5 / 0.15 test among siblings and
  stops at the parent when none passes. A cluster failing at top level, or flagged as a doublet, goes to
  REVIEW.
- **Review.** Decided from the cluster's markers (top 25 genes by log2 fold change of pooled CPM against
  all other cells, among genes detected in ≥ 25% of the cluster), its panel scores, per-sample counts and
  median genes per cell; recorded in `deconvolution/joint_reannotation_review.tsv`, one row per cluster,
  with the decision and the genes that decided it. Allowed decisions: a panel label; a name not in the
  panels, supported by at least 3 markers with a literature basis; or DROP (doublet, low quality: few
  genes and no specific markers, or not decidable). The reviewer (Claude) has seen the audit's marker
  lists for the old labels (§9); that is disclosed, not a reason to depart from the markers.
- **Cell states.** The joint cluster replaces `{sample}_c{leiden}` as BayesPrism's `cell.state`, so states
  are shared across samples.

### 10.4 Checks, run on the final labels and reported in full

- **K1, consistency.** §4's profile metric on the new (sample, label) groups: per-label r_self, and the
  metric check (median r_self > median r between different labels). Pass: every multi-sample reference
  passes the check. **HIPPOC** additionally needs a median r_self ≥ 0.5 (other references ranged 0.51–0.97
  on the old labels; HIPPOC had 0.12); if it fails, HIPPOC's labels are not used and HIPPOC is reported as
  without a usable reference.
- **K2, coverage.** Labels present in only one sample, and clusters with ≥ 90% of their cells from one
  sample in a multi-sample reference, are listed with their markers and a reason (biology or batch).
- **K3, old against new.** The cross-tabulation of old and new labels per reference, and where the cells of
  the audit's single-sample pairs went (KIDNEY "Renal interstitial fibroblasts" and "Renal vesicle cells";
  WATSC "Pre-B cells" and "Naive B cells"). Expected, registered here: each pair lands mostly in one joint
  label (Injured proximal tubule cells; B cells). Reported whichever way it falls.
- **K4, losses.** Cells dropped (doublet, low quality, under 20) per reference, and their share.

### 10.5 What follows, not started without the user

`build_reference.py` takes the per-cell labels (`cells_final.tsv.gz`) in place of the consensus join; the
eight references are rebuilt into new directories and must pass `reference_qc.py`; then §6 and §8 apply as
registered. Nothing is deconvolved before that is agreed.

### 10.6 Seen before registration

The audit (§9) and everything in `REFERENCE_LABEL_AUDIT.md`; the consensus annotation table rows for
these references (§10.1); the panel symbols' presence in the rat gene annotations (to fix rat names, not
expression). No joint clustering, panel score or marker of a joint cluster.
