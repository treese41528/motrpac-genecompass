# Step 5 registration: corpus and model findings are measured and disclosed, not retrained

Registered 2026-11-14, before any of the measurements below was made. The PI's decision is no retrain
(`README.md`, decision 1). So the corpus findings of the audit (Parts 1-3) cannot be repaired in the model.
What can be done is to measure how much each one touches the training data and the results, say so, and fix
the code so a future corpus does not repeat it.

**Why rules first.** Each finding ends in a judgement: small enough to disclose, or large enough to matter.
The thresholds for that judgement are written here, before the numbers exist, so the reading cannot follow
the result [nosek2018].

## M1. Mitochondrial fraction of the training cells and of the reference cells (findings 1.2, 1.3)

**What is known.** No fixed mitochondrial cut was applied. The 3 SD outlier rule on mitochondrial fraction
had no effect in files keyed by Ensembl ID (368 of 555 files by the audit's reading of the report), because
mitochondrial genes were recognised by symbol. GeneCompass removes cells above 15% [yang2024]. High
mitochondrial fraction marks damaged cells and is a standard exclusion in single-cell QC, with the caveat that
a fixed cut is tissue-dependent: cardiomyocytes, hepatocytes and kidney tubule cells carry more mitochondrial
RNA than leukocytes [osorio2021; luecken2019].

**Measurement.** `analysis/audit_remediation/measure_mito_fraction.py`, on every QC file the v2 corpus keeps
(584 files, 6,803,206 cells): each cell's share of counts on the 13 mitochondrial protein-coding genes, found
by Ensembl ID. Per file, per study and overall: cells above 5%, 10%, 15%, 20% and 50%. The same per deconvolution
reference and cell type, for the references built from corpus cells.

**Reading, fixed now.**

| Share of the 6.8 million training cells above 15% | Reading |
|---|---|
| below 1% | The departure from GeneCompass's QC is immaterial for the corpus. Disclosed with the number |
| 1% to 5% | Disclosed as a limitation, with the studies that contribute most |
| above 5% | A material departure. Disclosed as such; a future corpus should apply a cut; the studies concerned are listed for the PI |

| Share of a production reference's cells above 15% | Reading |
|---|---|
| below 5% in every cell type with at least 100 cells | No action |
| 5% or more in some cell type | Reported per cell type and placed in the expert review plan. The reference is not rebuilt in this pass: a mitochondrial cut on references is a new analysis choice and needs its own registration, with tissue in mind |

The second table matters more for the results than the first. A reference profile averaged over damaged cells
is a wrong profile, whatever the model was trained on. Mitochondrial genes themselves are excluded from the
deconvolution, so the concern is the rest of a damaged cell's profile.

## M2. The 79 single-cell series that never entered the corpus (finding 1.1)

**Measurement.** The 35 with a usable matrix on disk are put through the Stage 2 cell QC, into a folder of
their own (never the training tree). Reported: studies, tissues, cells passing QC.

**Use.** They are rat cells the model has not seen. Step 4 registers them as the held-out set for the model
(findings 3.2, 3.6). Using them that way rules out adding them to a retrained corpus later without losing
the held-out set; that trade is the PI's (no retrain) and is recorded here.

**Check before use.** No sample of the 35 may share a GEO sample accession with a training file, and no
series may be a sub-series or super-series of a training series.

## M3. Gene IDs carried over from earlier annotations (finding 1.5)

**What is known.** An ID from an older Ensembl release is accepted as the same gene whenever the ID exists
in release 115. An ID can be reassigned between releases. Step 3's Amendment 1 found the consequence in one
committed list: 39 of 247 IDs carried another symbol in release 115.

**Measurement.** For corpus source files that give both an Ensembl ID and a gene symbol for each feature
(10x `features.tsv`), the share of IDs, and of counts, whose file symbol differs from the release-115 symbol,
split into: a placeholder became a name or the reverse (a rename), and one name became a different name (a
possible reassignment).

**Reading, fixed now.** The count-weighted share in the "different name" class is an upper bound on counts
attributed to a wrong gene. Below 0.5% in every study: disclosed as negligible. Otherwise: the studies and
genes are listed, and genes above the bound in a study that feeds a deconvolution reference go into the review
plan.

## M4. Genes that cannot be tokenized (finding 2.6)

991 genes of the universe have no median and never receive a token. **Measurement:** the share of each
tissue's pseudo-cell counts (refs5) that falls on genes with no token. **Reading:** reported per tissue; no
threshold, because nothing can be done about it without a retrain.

## Code repaired for a future corpus, behind switches that default to v2's behaviour

The v2 corpus is defined by its file list and checksums (`config/corpus_v2_plan.tsv`). Changing Stage 2's
default behaviour would make the committed code describe a corpus that does not exist. So each repair is a
switch, off by default, documented as recommended for any new corpus:

| Finding | Repair | Switch |
|---|---|---|
| 1.3 | Mitochondrial genes recognised by Ensembl ID as well as by symbol | `preprocessing.mito_by_id` |
| 1.2 | The fixed mitochondrial cut (`max_mito_fraction`) is applied | `preprocessing.apply_max_mito_fraction` |
| 1.8 | The 3 SD rule on the number of expressed genes, as GeneCompass applies it | `preprocessing.outlier_on_n_genes` |
| 1.1 | A series returned by both the single-cell and the bulk query keeps its single-cell row | `harvest.prefer_single_cell_on_conflict` |
| 1.6 | The file walk visits `_RAW` once | no switch: the duplicate visit only ever produced copies that v2 drops |

## Disclosed without measurement

| Finding | Statement |
|---|---|
| 1.7 | Stage 2 does not reproduce from raw data; the vocabulary was fixed on the uncleaned corpus |
| 2.3 | The human and mouse sets hold 5 million cells each; each human cell is seen about 3 times and each rat cell about 2 times per phase |
| 2.5 | The promoter builder's defects (April only; v2 does not use it) |
| 3.4 | The retrain's 10% tolerance cannot be shown to predate the smoke-test losses |
| 3.5 | The v2 initial checkpoint was built without a seed and cannot be regenerated; 53 and 35 logged steps have an infinite gradient norm |

## References

- [yang2024] Yang X, et al. GeneCompass. Cell Research 2024;34(12):830-845 (in `docs/audit/REFERENCES.md`).
- [nosek2018] Nosek BA, et al. The preregistration revolution. PNAS 2018;115(11):2600-2606.
- [osorio2021] Osorio D, Cai JJ. Systematic determination of the mitochondrial proportion in human and mice
  tissues for single-cell RNA-sequencing data quality control. Bioinformatics 2021;37(7):963-967.
  doi:10.1093/bioinformatics/btaa751
- [luecken2019] Luecken MD, Theis FJ. Current best practices in single-cell RNA-seq analysis: a tutorial.
  Mol Syst Biol 2019;15(6):e8746. doi:10.15252/msb.20188746

Both looked up on Crossref on 2026-11-14 (`record/crossref_remediation.tsv`).
