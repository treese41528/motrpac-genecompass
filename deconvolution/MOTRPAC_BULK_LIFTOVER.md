# MoTrPAC Bulk Gene-ID Liftover & Deconvolution Prep — Findings & Status

_Status as of 2026-06-09. For collaborator review. Branch `stage8-omnideconv-setup`._
_Commits: `3bd9fb8` (Fix 1 + DWLS), `1bdccd9` (missed-gene record), `fd530e5` (bridge-3),
`05cb123` (Fix 2 cortex), `cde5e00` (ID-space audit); **id_history bridge re-targeted to rel-113 —
the §5 correctness bug is now FIXED, all artifacts regenerated** (this revision)._

## Goal

Prepare the real MoTrPAC rat endurance-training **bulk RNA-seq** (`TRNSCRPT_<TISSUE>_RAW_COUNTS`)
for cell-type **deconvolution** (BayesPrism) so that per-cell-type expression can be fed to
**GeneCompass**. The bulk and our single-cell (SC) references are on **different rat genome
annotations**, which drives most of the work below.

## TL;DR

- The real bulk is **deconv-ready for all 19 tissues** (`deconvolution/motrpac_bulk/<TISSUE>/`).
- A 3-bridge **gene-ID liftover** lands **73.0%** of bulk genes on current **rel-113** IDs; the other
  27% are non-current Rnor_6.0 orphans with no rel-113 home (dropped — lossless for deconv, they never
  intersect the modern references). Primary (training-regulated) **token-vocab coverage 89.5% → 94.8%**.
- **The references — not the bulk or the vocabulary — are the binding constraint** on how many
  genes can be deconvolved. Lifting the bulk helps only where a reference already contains the gene.
- **Fix 2 (cortex) is done**: the cortex reference was gene-poor (5,536 genes) due to a *build bug*,
  not shallow data — fixed to **18,162 genes** (training-regulated coverage 23% → 94%), no new data.
- ✅ **§5 correctness bug FIXED:** bridge 3 previously mapped to RGD's **GRCr8** `ENSEMBL_ID`. Because
  `rowsum` collapses bulk rows by lifted-ID *before* the reference intersection, and GRCr8 accessions
  overlap rel-113 ~60%, that **silently mis-summed ~101 unrelated genes' bulk counts onto a different
  rel-113 gene that *does* enter deconvolution** (e.g. real CD36 → the *Cd36-ps1* pseudogene row) — so
  the earlier "harmless, intersected away" assessment was wrong. The bridge now recovers the **current
  symbol** (RGD Entrez/old-symbol) and resolves it through the rel-113 symbol map, so every lifted ID
  is rel-113 and the corruption is gone. Details in §5.

---

## 1. The real MoTrPAC bulk

- `TRNSCRPT_<TISSUE>_RAW_COUNTS.rda`: data.frame, 4 meta cols (`feature`/`feature_ID`/`tissue`/`assay`)
  + sample columns (viallabels), raw integer counts.
- **32,883 ENSRNOG genes, identical across all 19 tissues** (single annotation, **Rnor_6.0 / Ensembl
  ~rel-96 era**). 50 samples/tissue except OVARY 24, TESTES 25.
- Design (via `PHENO`, join key `viallabel`): **2 sex × 5 group × 5 rep** (1/2/4/8-week programs +
  8-week sedentary control).
- "Genes of primary importance" = **`TRAINING_REGULATED_FEATURES`** → **9,800 unique training-regulated
  transcriptome genes** (union; per-tissue counts match the paper's Fig 1c).

## 2. The gene-ID liftover (3 bridges)

`deconvolution/R/prepare_motrpac_bulk.{R,sh}` lifts each bulk gene ID → current mRatBN7.2 ENSRNOG,
then writes deconv-ready `bulk.mtx` (samples×genes) + `bulk_genes.tsv` + `bulk_samples.tsv` per tissue
(consumed by `run_deconvolution.R … bulk`). Auditable map + report committed under
`deconvolution/reference/motrpac_bulk_liftover.tsv` / `…_report.txt`.

| bridge | rule | count | % of 32,883 |
|---|---|---|---|
| `direct` | bulk ID already current (in vocab ∪ biomart rel-113) | 20,203 | 61.4% |
| `symbol` | `FEATURE_TO_GENE` gene_symbol → vocab/biomart current ENSRNOG | 2,702 | 8.2% |
| `id_history` | Entrez/old-symbol → RGD **current SYMBOL** → rel-113 vocab/biomart ENSRNOG (§5) | 1,098 | 3.3% |
| `unmapped` | no rel-113 home (non-current Rnor_6.0 orphan) — dropped | 8,880 | 27.0% |

The big shift from the earlier draft (`id_history` 9,527→1,098, `unmapped` 451→8,880) **is** the §5 fix:
the ~8.4k genes that previously received a wrong-release **GRCr8** ID — which never validly intersected a
rel-113 reference — are now correctly dropped instead of carried as junk (or, worse, mis-summed; §5).

**Primary-gene token-vocab coverage: 89.5% (8,768/9,800 already in vocab) → 94.6% (+498 symbol) →
94.8% (+29 id_history).** Spot-checks (PLN, EP300, LTB, HINT1, PPP1R18) recover correctly. The report,
map, and missed-gene record are all emitted by `prepare_motrpac_bulk.R` from one recovery pass, so their
numbers cannot drift (the earlier 39-vs-42 `id_history` discrepancy between report and record is gone).

**Reliable rat symbol map = RGD `GENES_RAT.txt`** (`SYMBOL`/`OLD_SYMBOL`/`NCBI_GENE_ID`/`ENSEMBL_ID`).
⚠️ **Do NOT use biomart `rat_gene_info.tsv` "Gene name" for rat symbol matching — it is empty for 43%
of genes** (it lacks even `ACTB`/`CD36`).

## 3. Deconvolution reality: the references are the constraint

The earlier claim that the build mismatch "drops ~12k bulk genes from bulk∩reference" was an
**unmeasured inference and is wrong**. Measured directly, the liftover's effect on the deconvolution
intersection (`bulk ∩ reference`) is **tissue-specific and never negative (Δ≥0)**:

| reference | bulk∩ref (raw bulk) | after lift | Δ |
|---|---|---|---|
| gastrocnemius | 16,940 | 18,580 | **+1,640** |
| cortex (union ref) | 16,170 | 17,012 | +842 |
| hippocampus | 17,599 | 18,175 | +576 |
| skeletal muscle | 17,499 | 18,052 | +553 |
| cortex (old inner ref) | 5,317 | 5,482 | +165 |
| heart / kidney / liver / lung / PBMC / WAT | =ref | =ref | **0** |

The "0" tissues already had their reference IDs covered by the raw bulk. **Takeaway: the bulk/vocab
are nearly complete; what limits deconvolution is which genes each single-cell reference contains.**
(These Δ are *larger* than the earlier draft's because the rel-113 fix lands symbol-recovered genes on
IDs that actually match the references — the old GRCr8 IDs did not intersect, so they added nothing.)

## 4. Fix 2 — cortex reference (DONE)

The cortex reference (`cortex_GSE303115`) had only **5,536 genes** (≈20% primary coverage) — but
**the data is not shallow**. `build_reference.py` concatenated samples with `join="inner"`
(gene-set **intersection**); GSE303115's per-sample depth ranges **9.5k–21k genes**, so the
intersection collapsed to 5,536 (union = 21,248).

Fix: added opt-in `--gene-join {inner,outer}` (default `inner` ⇒ every other reference is byte-identical)
+ `--min-gene-cells`. Rebuilt cortex (`--gene-join outer --min-gene-cells 10`, on a compute node — the
173k-cell union OOMs on login; `slurm/analysis/build_cortex_union.slurm`):

| metric | before | after |
|---|---|---|
| cortex reference genes | 5,536 | **18,162** |
| training-regulated coverage | 22.9% | **94.3%** |
| lifted bulk ∩ reference | 5,482 | **17,012** |

New reference at `deconvolution/reference/cortex_GSE303115_union/` (gitignored); 35 cell types kept.
**No external download needed** — and note the two cortex datasets named in earlier planning notes
were the **wrong species** (`GSE253415` = human pulmonary artery, `GSE271209` = mouse lung). If a
different cortex source is ever wanted, `GSE262970` (rat auditory cortex, 255k nuclei) is a real option.

## 5. ✅ FIXED — the `id_history` bridge now targets rel-113 (was: emitted GRCr8)

**The bug (now fixed).** Bridge 3 used to map to RGD's `ENSEMBL_ID`. An ID-space audit
(`deconvolution/audit_idspace.py` → `deconvolution/reference/idspace_audit/`) showed why that is wrong:

- The **SC corpus** (864 samples, 22,489-gene union, 9.48M cells) is **100% in biomart rel-113**.
- **RGD `ENSEMBL_ID`s are only 60.6% in rel-113** — RGD tracks the newer **GRCr8** assembly.

The earlier draft called this "harmless (wrong-release IDs are intersected away)." **That was wrong.**
`process_tissue` collapses bulk rows with `rowsum(counts, group = lifted_id)` **before** the reference
intersection, keyed on the lifted-ID *string*. Because 60.6% of GRCr8 accessions textually coincide with
a rel-113 ID — often a **different gene** after the Rnor_6.0→mRatBN7.2 rebuild — the bridge summed an
orphan's counts onto that other, real, rel-113 gene, which then **does** intersect the references. Net:
**101 rel-113 genes that appear in our single-cell references had their MoTrPAC bulk counts corrupted**
(126 mis-merged bulk rows), e.g.:

| corrupted rel-113 gene (target) | unrelated bulk gene summed in |
|---|---|
| `ENSRNOG00000005906` = **Cd36-ps1** (CD36 *pseudogene*) | **CD36** (real) |
| `ENSRNOG00000000133` = TXNDC15 | PCBD2 |
| `ENSRNOG00000003703` = MCM6 | DARS1 |
| SURF1, RPL10A, SENP7, … (98 more) | SURF4, LOC680700, IMPG2, … |

**The fix.** Bridge 3 now recovers the **current symbol** (assembly-stable: `FEATURE_TO_GENE.entrez_gene`
→ RGD `NCBI_GENE_ID` → RGD `SYMBOL`, or RGD `OLD_SYMBOL`) and resolves *that* to a rel-113 ENSRNOG through
the **same `sym2ens` map bridge 2 uses**. Every lifted ID is now rel-113, so `rowsum` only ever merges
annotations of the *same* current gene. Verified on the regenerated map: **935 distinct `id_history`
targets, 100% valid rel-113, 0 GRCr8; the 101 cross-gene collisions are gone** — CD36 and PCBD2 now drop
cleanly to `unmapped` rather than corrupting a pseudogene / TXNDC15. (The residual same-ID merges are
legitimate, e.g. bulk `AKAP2`+`PALM2` → rel-113 **Pakap**, whose RGD record lists both as old symbols.)

**No biological prize hiding.** The rel-113 `id_history` bridge lands 1,098 bulk genes and adds **29**
training-regulated genes to the token vocab. As the pre-fix analysis already found, the only
broadly-expressed one is **NDUFS6** (mito complex I, training-regulated in 6 tissues incl. both skeletal
muscles, read in 466/864 SC samples); the rest are sparse/inert and **none are named in the MoTrPAC
papers**. So the value of the fix is **correctness** — it removes count corruption of 101
reference-present genes and drops ~8.4k wrong-release GRCr8 IDs — not a new biological signal.

## 6. What's still missing, and can it be recovered?

`deconvolution/reference/motrpac_missed_genes.tsv` (+ `_summary.txt`, both now emitted by
`prepare_motrpac_bulk.R` from the same recovery pass as the report — so they can't drift): **505**
training-regulated genes remain without a GeneCompass token after the liftover. **503 of the 505 do not
occur in our single-cell corpus** (the 2 that do are `IGHL13` immunoglobulin features, present by
symbol) — so they **cannot be recovered by growing the GeneCompass vocabulary from the current SC
data**; that would require deeper/different single-cell data. Notable members: the `RT1-*` rat-MHC
immune cluster (training-regulated in up to 8 tissues; hyper-polymorphic, excluded from ortholog
vocabularies) and a few real metabolic/immune genes (CD36, NDUFA13, TPI1, GPT, LYVE1, C4A, CFB).

> The count rose 495→505 and `in_current_annot` fell 472→34 **because of the same §5 fix**: the prior
> record used RGD's GRCr8 `ENSEMBL_ID` (so it both spuriously "recovered" ~10 genes via collision and
> marked 472 as having a "current" ID). `current_ensrnog`/`in_current_annot` are now strict **rel-113**
> tests; the 471 with no rel-113 ID are genes rel-113 lacks under that symbol (e.g. NDUFA13 exists only
> as `Ndufa13-ps1`). `category`/`importance` are documented rules in the script (`high 29/med 50/low 426`).

## 7. Artifacts / file map

| path | what | tracked? |
|---|---|---|
| `deconvolution/R/prepare_motrpac_bulk.{R,sh}` | the 3-bridge liftover + bulk writer | yes |
| `deconvolution/reference/motrpac_bulk_liftover.tsv` / `…_report.txt` | per-gene map + coverage report | yes |
| `deconvolution/reference/motrpac_missed_genes.tsv` / `…_summary.txt` | still-missing training-regulated genes (emitted by `prepare_motrpac_bulk.R`) | yes |
| `deconvolution/motrpac_bulk/<TISSUE>/{bulk.mtx,bulk_genes.tsv,bulk_samples.tsv}` | per-tissue deconv-ready bulk | gitignored |
| `deconvolution/build_reference.py` (`--gene-join`) + `build_references_v2.sh` | cortex union fix | yes |
| `deconvolution/reference/cortex_GSE303115_union/` | rebuilt cortex reference (18,162 genes) | gitignored |
| `deconvolution/audit_idspace.py` + `deconvolution/reference/idspace_audit/` | ID-space / release audit + per-gene membership | script yes, outputs gitignored |
| `slurm/analysis/build_cortex_union.slurm` | one-off compute-node build | local only |

## 8. Next stages

0. ✅ **`id_history` correctness bug — DONE** (this revision): bridge 3 re-targeted to rel-113, all 19
   tissues + map/report/missed-gene record regenerated (§5). Every lifted ID is rel-113; ~8.4k inert
   GRCr8 IDs dropped; the 101-gene count corruption removed; NDUFS6 + 28 other training-regulated genes
   correctly recovered into the token vocab.
1. **Deconv-validate the rebuilt cortex reference** (`run_deconvolution.R`, compute node) — gene count is
   fixed (18,162) but cell-fraction recovery has not been re-checked. Optionally apply `--label-scheme brain`
   (merges collinear neuron labels) for cross-dataset validation.
2. **Prepare & run the real MoTrPAC-bulk deconvolution** — for each tissue, pair
   `deconvolution/motrpac_bulk/<TISSUE>` with its single-cell reference and run BayesPrism; then feed the
   per-cell-type expression to GeneCompass. Production considerations (separate notes): relative/differential
   framing, sex-split references (WAT only — the only sex-balanced rat SC reference), per-tissue references,
   immune-label resolution, and the activity-confound threat.
