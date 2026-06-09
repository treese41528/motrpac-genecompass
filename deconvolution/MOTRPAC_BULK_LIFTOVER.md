# MoTrPAC Bulk Gene-ID Liftover & Deconvolution Prep — Findings & Status

_Status as of 2026-06-08. For collaborator review. Branch `stage8-omnideconv-setup`._
_Commits: `3bd9fb8` (Fix 1 + DWLS), `1bdccd9` (missed-gene record), `fd530e5` (bridge-3),
`05cb123` (Fix 2 cortex), `cde5e00` (ID-space audit)._

## Goal

Prepare the real MoTrPAC rat endurance-training **bulk RNA-seq** (`TRNSCRPT_<TISSUE>_RAW_COUNTS`)
for cell-type **deconvolution** (BayesPrism) so that per-cell-type expression can be fed to
**GeneCompass**. The bulk and our single-cell (SC) references are on **different rat genome
annotations**, which drives most of the work below.

## TL;DR

- The real bulk is **deconv-ready for all 19 tissues** (`deconvolution/motrpac_bulk/<TISSUE>/`).
- A 3-bridge **gene-ID liftover** maps 98.6% of bulk genes to (intended) current IDs; primary
  (training-regulated) **token-vocab coverage 89.5% → 95.0%**.
- **The references — not the bulk or the vocabulary — are the binding constraint** on how many
  genes can be deconvolved. Lifting the bulk helps only where a reference already contains the gene.
- **Fix 2 (cortex) is done**: the cortex reference was gene-poor (5,536 genes) due to a *build bug*,
  not shallow data — fixed to **18,162 genes** (training-regulated coverage 23% → 94%), no new data.
- ⚠️ **Known correctness bug (open):** the 3rd liftover bridge (Entrez/RGD) emits **GRCr8-era IDs**,
  not the rel-113 IDs our references use. Harmless to deconvolution (wrong-release IDs are
  intersected away) but incorrect and slightly lossy. Fix is scoped below.

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
| `id_history` | Entrez (`FEATURE_TO_GENE.entrez_gene`) → RGD → RGD `ENSEMBL_ID` (⚠ see §5) | 9,527 | 29.0% |
| `unmapped` | no bridge | 451 | 1.4% |

**Primary-gene token-vocab coverage: 89.5% (8,768/9,800 already in vocab) → 94.6% (+498 symbol) →
95.0% (+39 id_history).** Spot-checks (PLN, EP300, LTB, HINT1, PPP1R18) recover correctly.

**Reliable rat symbol map = RGD `GENES_RAT.txt`** (`SYMBOL`/`OLD_SYMBOL`/`NCBI_GENE_ID`/`ENSEMBL_ID`).
⚠️ **Do NOT use biomart `rat_gene_info.tsv` "Gene name" for rat symbol matching — it is empty for 43%
of genes** (it lacks even `ACTB`/`CD36`).

## 3. Deconvolution reality: the references are the constraint

The earlier claim that the build mismatch "drops ~12k bulk genes from bulk∩reference" was an
**unmeasured inference and is wrong**. Measured directly, the liftover's effect on the deconvolution
intersection (`bulk ∩ reference`) is **tissue-specific and never negative (Δ≥0)**:

| reference | bulk∩ref before | after lift | Δ |
|---|---|---|---|
| gastrocnemius | 16,940 | 17,836 | **+896** |
| hippocampus | 17,599 | 18,045 | +446 |
| skeletal muscle | 17,499 | 17,923 | +424 |
| cortex (old ref) | 5,317 | 5,482 | +165 |
| heart / kidney / liver / lung / PBMC / WAT | =ref | =ref | **0** |

The "0" tissues already had their reference IDs covered by the raw bulk. **Takeaway: the bulk/vocab
are nearly complete; what limits deconvolution is which genes each single-cell reference contains.**

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
| lifted bulk ∩ reference | 5,482 | **16,925** |

New reference at `deconvolution/reference/cortex_GSE303115_union/` (gitignored); 35 cell types kept.
**No external download needed** — and note the two cortex datasets named in earlier planning notes
were the **wrong species** (`GSE253415` = human pulmonary artery, `GSE271209` = mouse lung). If a
different cortex source is ever wanted, `GSE262970` (rat auditory cortex, 255k nuclei) is a real option.

## 5. ⚠️ Known correctness bug — the `id_history` bridge emits the wrong release (OPEN)

The Entrez/RGD `id_history` bridge maps to RGD's `ENSEMBL_ID`. An ID-space audit
(`deconvolution/audit_idspace.py` → `deconvolution/reference/idspace_audit/`) shows:

- The **SC corpus** (864 samples, 22,489-gene union, 9.48M cells) is **100% in biomart rel-113**.
- **RGD `ENSEMBL_ID`s are only 60.6% in rel-113** — RGD tracks the newer **GRCr8** assembly.

So the 9,527 `id_history` genes were lifted to **GRCr8-era IDs that do not exist in our rel-113
references**, reading "absent" by ID (1.2% in corpus) when **by symbol 1,167 are actually present**.
This is harmless to deconvolution (wrong-release IDs simply don't intersect and are dropped) but is
**incorrect** and leaves recoverable genes on the table.

**Re-targeting properly** (`old ID → Entrez → RGD current SYMBOL → rel-113 vocab ENSRNOG`) recovers
**~719 genes** into the references — but the payoff is small and mostly inert:

| of the 719 recovered | count |
|---|---|
| olfactory / chemoreceptors (not expressed in MoTrPAC tissues) | 653 |
| non-receptor (mostly paralog/RIKEN-clone variants) | ~50 |
| **training-regulated** | **~16** (≈15 unique; NDUFS6 counted twice) |

The ~15 training-regulated genes, cross-checked three ways:

| symbol | tissues (train-reg) | SC corpus occurrence | named in MoTrPAC papers |
|---|---|---|---|
| **NDUFS6** (mito complex I) | 6 (incl. both skeletal muscles) | **466/864 samples, 4.7% of cells** | no (pathway OXPHOS *is* discussed) |
| C3H15ORF48 (= C15orf48/NMES1) | 1 | 403 samples, 3.3% | no |
| C7H8ORF82 | 2 | 434 samples, 1.8% | no |
| ZFP534L1 | 1 | 434 samples, 1.3% | no |
| DMBT1 | 4 | **52 samples, 0.017%** (barely read) | no |
| (10 others: C3H15ORF62, PDXKL1, KCTD12B, H3C13, 4930426D05RIKL, LY6M, USP12L1, SULT1C2AL1, NXPE5L3, OR6X1) | 1 each | sparse → inert | no |

**None of the 15 are named in the MoTrPAC papers** (checked readings 20 / **21 = main rat Nature paper** /
22; the extraction does capture named hubs — e.g. HSPA1B appears — so the absence is real). They live
only in the training-regulated supplement by construction. **NDUFS6** (broadly read, OXPHOS/complex-I,
both skeletal muscles) is the single worthwhile recovery; everything else is sparse or inert.

**Conclusion:** fix the bridge for **correctness** (drop ~8,000 wrong-release GRCr8 IDs, land all lifted
IDs on rel-113), not for a biological prize — there is none hiding here.

## 6. What's still missing, and can it be recovered?

`deconvolution/reference/motrpac_missed_genes.tsv` (+ `_summary.txt`): **495** training-regulated genes
remain without a GeneCompass token after the liftover. Using the reliable RGD map, **0 of the 495 occur
in our single-cell corpus** — so they **cannot be recovered by growing the GeneCompass vocabulary from
the current SC data**; that would require deeper/different single-cell data. Notable members: the
`RT1-*` rat-MHC immune cluster (training-regulated in up to 8 tissues; hyper-polymorphic, excluded from
ortholog vocabularies) and a few real metabolic/immune genes (CD36, NDUFA13, TPI1, GPT, LYVE1, C4A, CFB).

## 7. Artifacts / file map

| path | what | tracked? |
|---|---|---|
| `deconvolution/R/prepare_motrpac_bulk.{R,sh}` | the 3-bridge liftover + bulk writer | yes |
| `deconvolution/reference/motrpac_bulk_liftover.tsv` / `…_report.txt` | per-gene map + coverage report | yes |
| `deconvolution/reference/motrpac_missed_genes.tsv` / `…_summary.txt` | still-missing training-regulated genes | yes |
| `deconvolution/motrpac_bulk/<TISSUE>/{bulk.mtx,bulk_genes.tsv,bulk_samples.tsv}` | per-tissue deconv-ready bulk | gitignored |
| `deconvolution/build_reference.py` (`--gene-join`) + `build_references_v2.sh` | cortex union fix | yes |
| `deconvolution/reference/cortex_GSE303115_union/` | rebuilt cortex reference (18,162 genes) | gitignored |
| `deconvolution/audit_idspace.py` + `deconvolution/reference/idspace_audit/` | ID-space / release audit + per-gene membership | script yes, outputs gitignored |
| `slurm/analysis/build_cortex_union.slurm` | one-off compute-node build | local only |

## 8. Next stages

1. **Fix the `id_history` correctness bug** — re-target bridge 3 to rel-113
   (`old ID → Entrez → RGD current/old SYMBOL → vocab rat_gene`), drop the wrong-release GRCr8 IDs, re-run
   `prepare_motrpac_bulk.R` for all 19 tissues, regenerate the map/report (+ the missed-gene record),
   re-commit. Net: every lifted ID on rel-113; ~8k inert GRCr8 IDs removed; NDUFS6 + ~14 training-regulated
   genes correctly recovered.
2. **Deconv-validate the rebuilt cortex reference** (`run_deconvolution.R`, compute node) — gene count is
   fixed (18,162) but cell-fraction recovery has not been re-checked. Optionally apply `--label-scheme brain`
   (merges collinear neuron labels) for cross-dataset validation.
3. **Prepare & run the real MoTrPAC-bulk deconvolution** — for each tissue, pair
   `deconvolution/motrpac_bulk/<TISSUE>` with its single-cell reference and run BayesPrism; then feed the
   per-cell-type expression to GeneCompass. Production considerations (separate notes): relative/differential
   framing, sex-split references (WAT only — the only sex-balanced rat SC reference), per-tissue references,
   immune-label resolution, and the activity-confound threat.
