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
- 2026-10-27 — **Deviation from §10.3, logged while reviewing, before the checks (§10.4) were read for
  CORTEX, SMLINT, LUNG and HIPPOC.** Reading the accepted clusters' markers showed that the rule can accept
  wrong labels. The doublet flag only fires across compartments when both panels score ≥ 0.5, and
  min-max scaling lets one extreme small cluster set the scale (MUSCLE: a 40-cell ambient cluster held
  the muscle panel's maximum). Six accepted labels contradicted their markers: KIDNEY j12 (PT–distal
  doublets), WATSC j15 and j16 (stromal–T and stromal–macrophage doublets), WATSC j18 and j21 (glandular
  and milk-protein mammary epithelium accepted as "Mesothelial cells"), and MUSCLE j13 (mitochondrial
  ambient). Change: every accepted label is also checked against its markers. An override is allowed
  only where the markers contradict the label; it is recorded in `joint_reannotation_review.tsv` as
  "OVERRIDE of the accepted label …" with the genes, and the overrides are counted per reference in the
  results. KIDNEY, LIVER, WATSC and MUSCLE had already been finalized once, before this check; their checks
  are rerun after it.
- 2026-10-27 — Two further review rules, same day, after LUNG's clusters were read (HIPPOC not yet
  run). (a) **Under-calls.** A cluster accepted at a parent label ("Immune cells", "Epithelial cells")
  because its sub-panels tied is given the specific type when its markers name it clearly (≥ 3 genes);
  otherwise distinct clusters would be pooled under one uninformative label. Recorded as "Under-call made
  specific". (b) **CD4/CD8.** Cd4 and Il7r are absent from every reference's gene space, and rat NK cells
  carry CD8α, so the CD4/CD8 sub-panel is unreliable. A T cluster keeps "CD4+" or "CD8+" only where the
  CPM of Cd40lg against Cd8a/Cd8b is lopsided; otherwise it is "T cells". (c) **Splits.** A cluster whose
  markers and old labels show two distinct, non-doublet populations (CORTEX j24: T cells and vascular
  cells, median 749 genes) is re-clustered alone (`joint_reannotation_splits.tsv`; Leiden at the same
  resolution and seed on the kNN graph restricted to its cells); its parts go through the rule and the
  review like any other cluster.

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

## 11. Change log after §10 (continued from §9)
- 2026-10-27 — Joint re-annotation run and reviewed for all eight references (jobs 12035670, 12035671,
  12035758, 12035779); results in `JOINT_REANNOTATION_RESULTS.md`. K1 passes for every reference; no label
  is confined to one sample; the K3 expectation held (both single-sample pairs land in one joint label).
  HIPPOC: 255,303 of 279,195 barcodes are low-quality nuclei (≈ 250 genes) and are dropped; the remaining
  23,892 pass with median r_self 0.89 (criterion ≥ 0.5). 100 review decisions (70 reviews, 18 overrides,
  12 under-calls) are in `joint_reannotation_review.tsv` and should be read by someone with tissue
  knowledge before §10.5's rebuild. Nothing has been rebuilt or deconvolved.

## 12. BLOOD reference, option B (registered 2026-10-27, before either dataset was downloaded)

**Decision.** The user chose option B of `BLOOD_REFERENCE_PROPOSAL.md` on 2026-10-27: lymphocytes,
monocytes, dendritic cells and platelets from GSE335156; neutrophils, eosinophils/basophils if present,
and erythroid cells from GSE306764 Blood-CFA, after the checks below. Geyu chose the current PBMC
reference; the proposal is written for Geyu's review and this change has not yet been put to Geyu. The
current reference (GSE285476) is kept as a sensitivity arm. BLOOD results stay flagged until the new
reference passes and BLOOD is rerun.

**Draw site.** MoTrPAC rats were bled by cardiac puncture at harvest (Schenk et al. 2024, *Function*,
PMC11245678). Rat cardiac puncture usually samples the right ventricle, i.e. venous blood (an inference;
the paper does not name the ventricle). So GSE335156's **venous** libraries (V1–V5, GSM9806833–37, inferior
vena cava) are used; the arterial ones are not.

**Annotation.** Each dataset is clustered as in §10.3 (HVGs per library, Harmony on library where there is
more than one, Leiden 1.0, seed 0) and labelled by the BLOOD panel in `reference_marker_panels.yaml` with
the §10.3 rule plus the §9 review rules (markers checked for every cluster; overrides and under-calls
recorded; doublets, low quality and clusters under 20 cells dropped).

**Checks (all must pass before the composite is built; reported in full either way):**
1. *GSE306764 composition.* Cells per compartment in Blood-CFA and Bone-CFA are reported. Neutrophils in
   Blood-CFA: at least 100 cells, or the neutrophil type comes from Bone-CFA's most mature neutrophil
   cluster (Mmp8/Retnlg/Ltf low, S100a8/9 high), recorded as a deviation.
2. *Erythroid cluster is cells, not ambient RNA.* At least 50 cells; the summed CPM of the non-globin
   erythroid genes (Alas2, Slc4a1, Epb42, Car2, Slc25a37, Ahsp) at least 4× that of every other cluster;
   and a median of at least 100 genes per cell. If Blood-CFA fails, late erythroblasts from Bone-CFA
   (erythroid cluster with Mki67/Top2a CPM below the marrow median) are used, recorded as a deviation.
3. *Neutrophil activation.* The CFA control's neutrophils are scored for interferon (Ifit1, Ifit3, Isg15,
   Rsad2) and acute-phase/activation genes (Il1b, Cxcl2, Nfkbia); the share of neutrophils above the 95th
   percentile of the PBMC monocytes' score is reported. No pass/fail threshold (no rat resting reference
   exists); a high share is a stated limitation. The comparison with mouse resting blood neutrophils
   (GSE137540) is deferred.
4. *GSE335156.* Every PBMC type used must be present in at least 4 of the 5 venous rats; per-label
   cross-rat r_self is reported (§4 metric).
5. *Gene space.* Both datasets are put in the BLOOD bulk's gene-ID space; genes not mappable are dropped and
   counted. The composite uses genes present in both.
6. *Recovery on simulated mixtures* (before any real BLOOD run). Reference built from 3 venous rats plus
   half of the GSE306764 cells; mixtures summed from the held-out 2 rats and the other half at known
   proportions (erythroid 0–60% of mRNA, neutrophils 0–40%, the rest PBMC types in their observed ratios;
   20 mixtures); BayesPrism as in Stage 8. Pass: per-type Pearson r ≥ 0.9 between true and estimated
   fractions for erythroid, neutrophils, monocytes, T, B and NK cells.

**Composite.** Cell types and joint clusters (cell states) from both datasets, one reference; the source is
kept per cell. Built with and without the Hba/Hbb paralogs (both reported; the with-globin build is primary
unless check 6 fails for it and not for the other).
- 2026-10-27 — §12 amendment, before any check result was read. (a) Stage 8's runner
  (`run_deconvolution.R`) already removes the hemoglobin genes (`rat_exclude_genes.tsv`, category
  "hemoglobin") for every tissue, so the **no-globin build is primary** (it is what a BLOOD rerun would use)
  and the with-globin build is the sensitivity arm, run by pointing `RAT_EXCLUDE_GENES` at the list without
  that category. (b) The PBMC libraries contain reticulocytes (an erythroid cluster, 1,775 cells, 80% from
  rat V4); as registered, erythroid cells come only from GSE306764, so the PBMC erythroid label is left
  out of the composite. (c) GSE306764 Blood-CFA has no eosinophil or basophil cluster; Bone-CFA has both,
  but §12 allows only Blood-CFA, so the composite has neither (they remain inside other types).
- 2026-10-27 — §12 run (jobs 12036211, 12036261): checks 1, 2, 4, 5 and 6 pass; check 3 reported (53% of
  neutrophils interferon-high relative to PBMC monocytes). Check 6 per-type r 0.91–0.96; erythroid
  under- and neutrophils over-estimated in absolute terms. The with- and without-globin arms are identical
  because BayesPrism's outlier filter (`outlier.cut = 0.01`) removes the globins in both. BLOOD in
  `tissue_references.yaml` now points at `references_v4/BLOOD_composite`; the PBMC reference is kept as
  `previous_reference_dir`. BLOOD has not been deconvolved with it.
- 2026-10-28 — §8 rerun started (user, 2026-10-28), before any result with the new references was seen.
  Ten tissues: the eight rebuilt references cover nine (SKMGN and SKMVL share MUSCLE), plus BLOOD with the
  composite. Profile `config/profiles/refs_v4.yaml`; every output goes to a `_refs4` folder
  (`results_refs4/`, `genecompass_input_v2_refs4/` + `_human/`, `augur_input_v2_refs4/`, `grn_v2_refs4/`,
  `grn_v2_refs4_human/`), so the July-reference v2 results stay as the comparison. BAT, HEART, HYPOTH and
  TESTES are linked in unchanged so the gate's BH and the pooled IHW see all 14 tissues. Stage 9 uses the v2
  model folder (`.../rat_phase2_mixed_species/models`) and target sum 6,400, as the v2 runs did. §6 is now
  code: `run_pseudobulk_de.R` (threshold from `DE_MIN_MEDIAN_LIBSIZE`, default 1,000) and
  `corroborate_summary.py --min-libsize` (default 1,000; the median per-sample sum of Z read from
  `pred_z`). The registered sensitivities (100; 0 = none) are run for both and written beside the primary
  with a `_minlib100` / `_minlib0` suffix; hotspots, Stage 10's ordering and everything downstream use the
  primary. `validate_selection.py` skips the corpus-resolution checks for a reference built by its own
  `builder` (the BLOOD composite) and keeps the post-build sample check. The 100 review decisions are
  being checked against the published marker literature at the same time; if that check changes any
  label, the change is logged here, the reference is rebuilt, and the affected tissues are rerun.
- 2026-10-28 — Literature check of the review decisions (`JOINT_REANNOTATION_LITCHECK.md`,
  `joint_reannotation_litcheck.tsv`). Of 100 tissue-reference decisions, 3 were QUESTIONABLE; each was
  re-checked in the pooled CPM and changed before its tissue was deconvolved with the new reference:
  LUNG j37 Mesothelial cells → DROP (mesothelial–macrophage/AT2 doublets, 444 cells); CORTEX j24.5
  Monocytes → Neutrophils (29 cells); HIPPOC j24 Inhibitory neurons → Thalamic neurons (598 cells, a
  dissection carry-over kept as its own type). LUNG, CORTEX and HIPPOC are rebuilt in `references_v4/`
  (the pre-check builds kept in `tmp/` until the rerun is reported) and their Stage 8 runs wait for the
  rebuild; the other seven tissues ran on references this check did not change. Label refinements that move
  no cells between types are listed for an expert and not applied.
- 2026-10-28 — Name-dependent steps, fixed before any refs4 DE result. The joint labels renamed cell types
  that the frozen positive-control spec targets. Muscle "Skeletal myocytes" is now "Skeletal muscle cells"
  (28 controls). KIDNEY "Distal tubule cells" is now DCT and CNT. BLOOD's naive/memory/ISG/NK labels do
  not exist in the composite. Left alone, those controls would stop `compare_posctrl.py` (it refuses
  targets that match no block). The refs4 run uses its own copy, `posctrl_prereg_refs4.tsv`, built by
  `build_posctrl_prereg.py --celltype-map posctrl_celltype_map_refs4.tsv`. Genes, directions, sexes and
  thresholds are unchanged; 48 target strings are renamed and each renaming is written to `notes`. The
  original spec is regenerated byte-for-byte and stays the July-reference baseline. BLOOD's naive/memory
  targets become the broader T/B types, a loss of resolution reported with the Tier C results.
  `run_stage10.py` reads the spec from `deconvolution.posctrl_prereg` when a profile sets it.
  `validate_parenchyma_dataanchored.py` accepts "Skeletal muscle cells" and now stops when a tissue
  matches no parenchyma block; before, it skipped silently.
- 2026-10-29 — Two failures in the first downstream pass, fixed without reading any DE result. (1) Stage 10
  stopped at the positive-control step. Every BLOOD T- and B-cell block (CD4+, CD8+, T, Treg, B) is
  degenerate under §6, with median library 3–55, so it is skipped in DE. `compare_posctrl.py` lists only
  the blocks that ran, so it reported the targets as stale labels. It now separates a target that is absent
  from `de_summary.tsv`, which still stops the run, from one whose block exists but was skipped. Controls
  of a skipped block score `not_testable` with a warning, as §6 implies. Stage 10 was resumed from step 2.
  (2) Stage 12 crashed writing `subspace_probe.tsv`, because the new human root had no `gate_rcv10/`.
  The transfer and embeddings had finished. `run_stage12.py` now creates the folder, and Stage 12 was
  resumed from step 3.
- 2026-10-29 — A third failure, found while checking the Stage 14 output. `de_technical_covariate_robustness.R`
  hardcoded `data/deconvolution/results/motrpac`, so the first refs4 pass of the RIN/globin check scored the
  July Z matrices. Five hotspots whose names exist in both arms were scored on July Z; six renamed hotspots
  gave "no input". The script now reads `CFG_RESULTS_DIR`, as the DE wrapper does. The invalid table is kept
  in `tmp/refs4_rin_globin_robustness_WRONG_julyZ.tsv` until this is reported, and Stage 14 step 3 was
  rerun. The July-arm table is unaffected, since the default profile reads that folder.
- 2026-10-29 — The rerun of that step still read the July Z. `_config_sh.py` prints plain `CFG_X=...`
  lines, and the wrappers `eval` them without `export`, so the R process never saw `CFG_RESULTS_DIR`.
  This had been true for every non-default profile. Other steps were unaffected, because the DE wrapper
  uses the values in bash and Stage 13/14 pass `DE_DIR` explicitly. `de_technical_covariate_robustness.sh`
  and `run_gene_set_enrichment.sh` now export the two values their R scripts read. The third pass logs
  `Z from: .../results_refs4/motrpac` and scores all 11 hotspots.
- 2026-10-29 — §11 RESULT. The full comparison against the July-reference v2 results is in
  `REFS4_RERUN_RESULTS.md`.
  - Blocks: 124 of 165 analysable.
  - Hotspots: 11 under §6 (11 at a cut of 100, 9 with no rule), against 13 for July and 19 for July + §6.
    The fall comes from the references, not the rule.
  - Kept from July: BLOOD classical monocytes, HYPOTH OPC and LUNG endothelial cells; BLOOD megakaryocytes
    as megakaryocytes and platelets.
  - New: BLOOD erythroid cells and neutrophils; LUNG mesothelial and VSMC; SKMGN tenocytes; SKMVL satellite
    cells and VSMC.
  - Lost: the BLOOD lymphocyte hotspots (blocks now degenerate), muscle myocytes (q 0.061 / 0.089), LUNG
    macrophages (degenerate) and KIDNEY fibroblasts (degenerate in both arms).
  - Transfer: 11/11 preserved.
  - Positive controls: 5 of 81 gene-level outcomes changed. Recovered 12 + 1 confounded → 14; wrong
    direction 2 → 3 (SKMVL Plin4). BLOOD Tier C: cytotoxicity recovered in NK cells (p 0.028); the naive
    and ISG programs are not testable.
  - Enrichment: 5,906 pathways in 106 blocks, against 4,724 in 98.
  - RIN/globin: 11/11 ROBUST.
  - Composition flags on 2 hotspots (LUNG mesothelial, SKMVL VSMC).
  - GRN conservation: median 0.310, against 0.292.
  - Implausible fractions, reported and not adjusted: HIPPOC (endothelial 0.50, excitatory neurons 0.003),
    KIDNEY monocytes 0.16, WATSC pDC 0.05, BLOOD lymphocytes < 0.01.
- 2026-10-29 — HIPPOC diagnosed (`HIPPOC_REFS4_DIAGNOSIS.md`, `diagnostics/hippoc_refs4/`); diagnostic only,
  nothing in the registered run changed. The cause is ambient RNA in the snRNA reference.
  - The soup (the 254,663 dropped barcodes) correlates with the bulk at 0.87, more than any type does
    (≤ 0.82). The endothelial cluster is about 55% ambient and absorbs the bulk.
  - July's "Excitatory neurons" was 91% ambient barcodes, so the July composition was the same artefact.
  - An NNLS proxy (validated on CORTEX) with ambient-corrected profiles and merged vascular types gives
    neurons about two thirds. The neuron subtype split stays unidentifiable.
  - A corrected HIPPOC reference would be a reference change and needs its own entry before any rerun.

## 13. HIPPOC ambient-corrected reference (registered 2026-10-29, before the corrected reference was built)

**Decision.** The user asked on 2026-10-29 to attempt the fix proposed in `HIPPOC_REFS4_DIAGNOSIS.md`
even though it is expected to change little: MoTrPAC itself finds only 244 training-regulated HIPPOC
genes (15th of 19 tissues; Cell Genomics 2024: fewest exercise-responsive genes of 8 tissues), and refs4
HIPPOC has no hotspots and no dose-significant genes. **The refs4 run (§8, §11 RESULT) stays the
registered result.** This is a separate arm, `refs4h`, which differs from refs4 only in the HIPPOC
reference; it is reported beside refs4 and replaces nothing unless the user decides so after reading it.

**What was seen before registration.** Everything in `HIPPOC_REFS4_DIAGNOSIS.md`, including the NNLS-proxy
prediction (neurons ≈ 0.66, vascular 0.13, microglia 0.11 with SoupX-style profile correction and
vascular types merged). No corrected reference has been built and BayesPrism has not been run on one.

**Method** (`deconvolution/R/soupx_correct_reference.R`; SoupX 1.6.2 from CRAN, added to `R_libs/` and the
R manifest).
1. *Soup profile per sample.* No raw (unfiltered) matrices exist for GSE295314, so the soup is estimated
   from the barcodes the authors' matrices contain and refs4 dropped (`references_v3` barcodes absent from
   `references_v4`): per sample, their summed counts, as the diagnosis did for the pool. Passed to SoupX
   with `setSoupProfile`.
2. *Contamination per sample:* `autoEstCont` with the refs4 cell types as clusters (default settings,
   `forceAccept = TRUE` so a high estimate is recorded rather than stopping). The estimates are reported.
   If `autoEstCont` fails for a sample, that sample gets the median rho of the others, recorded.
3. *Correction:* `adjustCounts(roundToInt = TRUE)` with the cell types as clusters.
4. *Labels.* Endothelial cells + Mural cells + Vascular leptomeningeal cells → **Vascular cells**.
   Excitatory, Inhibitory, Thalamic neurons and Dentate gyrus granule cells → **Neurons** (their profiles are
   near-collinear, log-CPM r 0.96–0.99, so their split is not identifiable). The joint clusters stay as
   BayesPrism cell states, so neuronal and vascular subtypes still shape the model; only the reported types
   merge. No other change: same nuclei, genes and samples as refs4.
5. Built to `data/deconvolution/references_v4/HIPPOC_GSE295314_ambcorr` (provenance JSON beside it).

**Rerun (profile `config/profiles/refs_v4h.yaml`; every output in `_refs4h` folders).** The 13 other
tissues are linked from refs4. HIPPOC: Stage 8 from step 2 and Stage 9. Then, because the gate BH and the
pooled DE fit are cross-tissue, the whole gate (10 splits × 10,000 permutations, §6 primary plus 100/0),
Stage 10 (DE, positive controls, sensitivities), Stage 12 (HIPPOC transfer, then the human probe and the
transfer comparison). GRNs are built only for hotspots: if the refs4h hotspot set equals refs4's, the
refs4 networks are linked and only the steps reading DE are rerun (Stage 13 steps 3–4, Stage 14 steps
2–3); otherwise Stage 13 is rerun in full.

**Readouts and criteria (fixed now).**
- *Composition (primary).* Mean BayesPrism fraction over the 50 HIPPOC samples. **The fix works** if
  Neurons ≥ 0.50 and no non-neuronal type exceeds 0.30. Partial: Neurons 0.30–0.50. Fails: Neurons < 0.30
  or a non-neuronal type > 0.40. The NNLS proxy is rerun on the built reference and reported beside it.
- *Ambient check.* Per-type ambient fraction (the diagnosis's script-5 estimator) on the corrected
  reference; correlation of each type with the soup and with the bulk.
- *Downstream (reported, no criterion).* HIPPOC analysable blocks, hotspots, dose-significant genes and
  positive controls (Gpr37l1, Ldlr; cell type ANY); all-tissue hotspot list, compared with refs4. Any
  change outside HIPPOC can only come from the cross-tissue BH/IHW and is reported as such.
- *Expectation stated in advance:* composition improves; HIPPOC hotspots and dose-significant genes stay at
  or near zero, in line with MoTrPAC's weak hippocampal transcriptomic response.

**Known limits not addressed.** Choroid plexus is in the bulk but not in the reference (a few percent);
19 bulk genes are near zero in the reference (Eno2); without empty droplets the soup is a proxy built from
dropped barcodes, which include low-quality nuclei as well as ambient RNA.
- 2026-10-29 — §13 run started, before any output was read. SoupX 1.6.2 was installed into `R_libs/` from
  CRAN and added to `setup/r_packages.yaml`. The links were made by `slurm/analysis/refs4h_setup.sh`, which
  also listed every refs4/v2 file it points to (size and mtime). The last job compares that listing again,
  to catch any write through a link. Jobs: build 12058463 → Stage 8/9 + check 12058464 → gate 12058465 →
  Stage 10 12058466 and Stage 12 12058467 → follow-up 12058468.
  Resubmitted: build 12058507 → 12058508 → gate 12058509 → Stage 10 12058510, Stage 12 12058511 → follow-up 12058512.
- 2026-10-29 — The §13 build (12058463) stopped on a check that every refs4 nucleus exists in the July build.
  Six of 23,892 do not; the diagnosis's script 5 had already matched only 23,886. They have no bearing on
  the soup, which is built from July barcodes absent from refs4, so the check now logs the count. No output
  had been produced. The chain was cancelled and resubmitted.
- 2026-10-29 — §13 primary RESULT (job 12058508, `diagnostics/hippoc_refs4/6_refs4h_check.txt`): **FAILS.**
  - SoupX `autoEstCont` gave rho 0.047–0.167 per sample and removed 7.3% of UMI overall, but only 7.9% of
    the endothelial cluster's UMI (the diagnosis estimated about 55%).
  - BayesPrism mean over 50 samples: Vascular cells 0.504, Neurons 0.402, Astrocytes 0.073 (refs4 with the
    merged names: 0.501 / 0.393 / 0.085). The NNLS proxy agrees: neurons 0.15, against 0.13 uncorrected.
  - The script-5 ambient estimate falls (vascular 0.38 → 0.27, astrocytes 0.40 → 0.20) but stays high.
  - The registered downstream chain (gate 12058509 onward) is left to run as registered.

## 13b. Per-nucleus ambient correction (registered 2026-10-29, after the §13 failure, before this build)

**Why.** SoupX applies one contamination fraction to every nucleus in a sample. Ambient RNA is closer to a
fixed number of UMI per droplet, so its share is largest in shallow nuclei, which is exactly the endothelial
cluster (median 967 UMI). The diagnosis's per-type estimate (endothelial 0.55, neurons 0.07) reflects that.

**Method** (`soupx_correct_reference.R`, `SOUPX_MODE=percell`; everything else as §13: same soup, same
merges, same nuclei).
1. For each refs4 cell type t, a non-expressed gene set G_t: up to 100 of the 2,000 most soup-abundant
   genes that are specific to another type. This is the script-5 rule: another type's CPM ≥ 20 × (t's CPM + 1),
   or another type > 200 CPM while t < 5% of it.
2. Per nucleus c of type t: ambient UMI a_c = Σ_{G_t} counts / Σ_{G_t} soup fraction (an upper bound).
3. Per sample: A_s = median a_c over its nuclei. Per nucleus: rho_c = min(0.9, A_s / UMI_c).
4. `setContaminationFraction(rho_c)` and `adjustCounts(roundToInt = TRUE)` as before.

Output: `references_v4/HIPPOC_GSE295314_ambcorr_percell`, profile `refs_v4h2.yaml`, `_refs4h2` folders.
Same criteria as §13. **The downstream chain runs only if the composition is at least PARTIAL** (Neurons ≥ 0.30
and no non-neuronal type > 0.40). Otherwise the attempt stops here and HIPPOC stays flagged unreliable.
Seen before registration: the §13 result above.
