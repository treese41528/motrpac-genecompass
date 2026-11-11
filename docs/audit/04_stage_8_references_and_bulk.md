# Part 4. Stage 8, part 1: the single-cell deconvolution references and the MoTrPAC bulk target

> One part of the methods audit of 2026-11-11; start at [`README.md`](README.md). The main findings of every
> part are collected in [`FINDINGS.md`](FINDINGS.md) and every reference in [`REFERENCES.md`](REFERENCES.md).
> Citations are `path:line` at commit `8249506` unless the text says otherwise.

Methods audit, 2026-11-11, branch `aim2b-perturbation-engine` (HEAD `e690c22`). Read-only audit: nothing in
the repository was changed and no job was run. The only commands executed were file reads, `grep`, read-only
`git log` / `git show`, small TSV/JSON tabulations, and the two metadata-only validators (`reference_qc.py` without `--deep`,
`validate_selection.py`), both with `PYTHONDONTWRITEBYTECODE=1`.

**Scope.** How the rat single-cell (sc) or single-nucleus (sn) reference of each of the 16 deconvolved
MoTrPAC tissues was selected (8a), labelled (8b), quality-gated (8c), specially constructed where needed
(8d) and written to disk (8e), and how the MoTrPAC bulk RNA-seq was prepared (8f). BayesPrism itself, the
benchmark panels, the purity sweeps and the pseudo-cells are in Part 5 and are mentioned only where
a reference check depends on them.

**What the default run used.** The default configuration deconvolves 16 tissues against 15 reference
directories (SKMGN and SKMVL share one). The reference directory of each tissue is the `reference_dir` of
`deconvolution/tissue_references.yaml`; the Stage 8 provenance files of the default result tree confirm that
these directories were the ones read (`data/deconvolution/results_refs4all_rebuild/motrpac/<TISSUE>/provenance_run_deconvolution.json`,
field `params.ref_dir`, all 16 checked). `config/pipeline_config.yaml:399-406` describes the default;
`config/profiles/*.yaml` change output folders only and never the references
(`config/profiles/refs4all_rebuild.yaml:7-15`, `refs_v4all.yaml:10-18`, `refs4_14.yaml:5-13`,
`july_refs.yaml:6-15`; `july_refs.yaml:6` states that Stage 8 takes `--ref-dir` explicitly).

**Conventions.** Citations are `path:line` relative to the repository root. "NOT FOUND" means the item was
looked for and is absent; where is stated. Bibliographic details first supplied from memory were checked afterwards; the outcome is given in brackets.
Numbers quoted from a result file cite that file. Inside tables, a literal `|` of a regular expression is
written `\|` (Markdown table escaping); the expressions themselves contain a plain `|`.

**Abbreviations.** sc/scRNA = single-cell RNA-seq; sn/snRNA = single-nucleus RNA-seq; GEO = Gene Expression
Omnibus; GSE/GSM = GEO series/sample accession; QC = quality control; HVG = highly variable gene; PCA =
principal-component analysis; kNN = k-nearest-neighbour graph; CPM = counts per million; UMI = unique
molecular identifier; θ = BayesPrism's estimated mRNA fraction of a cell type; Z = BayesPrism's posterior
per-cell-type expression; DE = differential expression; BH = Benjamini-Hochberg; LLM = large language model;
PBMC = peripheral-blood mononuclear cells; RBC = red blood cell; CFA = complete Freund's adjuvant; EDS =
the Leydig-cell toxicant of the testis source study (the repository gives only the abbreviation; ethane
dimethanesulfonate [expansion from general knowledge, not from a project file]); SD = Sprague-Dawley; F344 = Fischer 344; MoTrPAC tissue codes: SKMGN
(gastrocnemius), SKMVL (vastus lateralis), WATSC (subcutaneous white adipose), BAT (brown adipose), HIPPOC
(hippocampus), HYPOTH (hypothalamus), SMLINT (small intestine), ADRNL (adrenal), VENACV (vena cava); cell
types: PT = proximal tubule, DCT/CNT = distal convoluted/connecting tubule, AT1/AT2 = alveolar type I/II,
FAP = fibro-adipogenic progenitor, OPC = oligodendrocyte precursor cell, VLMC = vascular leptomeningeal
cell, BAM = border-associated macrophage, DG = dentate gyrus, NK = natural killer, pDC = plasmacytoid
dendritic cell, LSEC = liver sinusoidal endothelial cell.

## Per-tissue summary (the 15 references of the 16-tissue default)

Cells / types / states / genes are from each reference's `summary.txt` line 1 (path = `data/deconvolution/`
+ reference directory + `/summary.txt`). Samples and titles are from `reports/annotations/annotation_inventory.tsv`
(columns `sample_id`, `gsm`, `geo_title`, `condition_resolved`, `sex_resolved`) for corpus references and from
`deconvolution/tissue_references.yaml` for the external ones. Strain is as recorded in
`deconvolution/REFERENCE_SELECTION_PLAN.md:81-99`; the bulk is F344 (`REFERENCE_SELECTION_PLAN.md:9`) and no
reference is F344 (`REFERENCE_SELECTION_PLAN.md:156-159`; `BLOOD_REFERENCE_PROPOSAL.md:81`).

| Tissue | Reference directory (under `data/deconvolution/`) | Accession, samples used | Assay; strain | Cells / types / states / genes | Label provenance | Known limitations (source) |
|---|---|---|---|---|---|---|
| KIDNEY | `references_v4/KIDNEY_GSE240658` | GSE240658; 4 samples (GSM7707155-58, "healthy control rep1-4"); `tissue_references.yaml:39-65` | snRNA; SD | 24,479 / 15 / 20 / 17,867 | joint re-annotation | Control arm of a fasting-mimicking-diet / nephrosis study (`REFERENCE_ACCESSIONS.md:24`; `REFERENCE_SELECTION_PLAN.md:113-114`); 4,147 of 28,626 QC nuclei (14.5%) dropped in re-annotation (`joint_labels/KIDNEY/checks.json`); "Monocytes" (101 nuclei) take θ 0.161, a suspected sink (`TISSUE_REFERENCE_AND_PROPORTIONS.md:19`; `HIPPOC_REFS4_DIAGNOSIS.md:108-111`) |
| LIVER | `references_v4/LIVER_GSE220075` | GSE220075; 4 snRNA "sham" samples (GSM7699565-68); 2 Visium samples dropped by the QC gate; `tissue_references.yaml:67-91` | snRNA; Lewis + DA, male | 27,059 / 8 / 18 / 17,895 | joint re-annotation | 24,353 of 27,059 nuclei are hepatocytes; three small types have cross-sample r_self < 0.5 (`joint_labels/LIVER/checks.json`); young animals (8-10 wk, `DECONV_REBUILD_RUNBOOK.md:285`) |
| BLOOD | `references_v4/BLOOD_composite` | GSE335156 venous PBMC V1-V5 (GSM9806833-37) + GSE306764 Blood-Ctrl (GSM9207340); `tissue_references.yaml:93-113` | scRNA (10x; Singleron); strain not stated / Lewis female | 67,147 / 13 / 21 / 18,260 | joint re-annotation (BLOOD panel) of external libraries | Erythroid profile = 90 cells; neutrophils from an adjuvant-treated control, 53% interferon-high; no eosinophils or basophils; two platforms; absolute bias on simulated mixtures (erythroid -0.24, neutrophils +0.10); see 8d |
| CORTEX | `references_v4/CORTEX_GSE303115` | GSE303115; the 2 rat libraries (GSM9118171-72); `tissue_references.yaml:115-141` | snRNA (multiome); strain not stated | 12,936 / 12 / 29 / 21,003 | joint re-annotation | No endothelial or mural type (lost to the 20-cell floor, `JOINT_REANNOTATION_RESULTS.md:95-96`); 685 medium spiny neurons from one sample's striatal piece; 2 libraries |
| HEART | `references_v3/HEART_GSE280111_LV` | GSE280111 left ventricle; 19 GSMs (GSM8587480-98); `tissue_references.yaml:143-173` | snRNA; Wistar, male, 17 wk | 135,288 / 16 / 249 / 19,256 | authors' deposited labels (Single Cell Portal SCP2828), merged to 16 types | 30,641 of 166,376 QC barcodes (18.4%) have no deposited label and are dropped; cardiomyocytes are 22,406 nuclei, fewer than fibroblasts (46,451); no build provenance file |
| WATSC | `references_v4/WATSC_GSE137869_Y` | GSE137869 WAT-M-Y, WAT-F-Y (GSM4331846, GSM4331849); `tissue_references.yaml:203-227` | scRNA; SD, 5 mo | 11,984 / 13 / 20 / 17,895 | joint re-annotation | No adipocytes; depot not stated by the source paper; one pooled library per sex; see 8d |
| SKMGN, SKMVL | `references_v4/MUSCLE_GSE137869_Y` (shared) | GSE137869 Muscle-M-Y, Muscle-F-Y (GSM4331858, GSM4331861); `tissue_references.yaml:229-281` | snRNA; SD, 5 mo | 10,740 / 9 / 17 / 17,895 | joint re-annotation | One reference for two muscles; lowest label consistency of the eight (median r_self 0.44; 5 of 9 labels < 0.5, `joint_labels/MUSCLE/checks.json`); replaced the only F344 reference (`REFERENCE_SELECTION_PLAN.md:156-159`) |
| BAT | `references_v3/BAT_GSE244451` | GSE244451 BAT_rep1, BAT_rep2 (GSM7816294, GSM7816296); `tissue_references.yaml:283-309` | snRNA; Dahl SS, male | 28,246 / 6 / 12 / 17,895 | authors' deposited labels (GEO SCP-annotated h5ad), two adipocyte subtypes merged | 37,194 of 65,441 QC barcodes (56.8%) have no deposited label and are dropped; brown adipocytes are 24,832 of 28,246; "unvalidated" (`tissue_references.yaml:309`); no build provenance file |
| HYPOTH | `references_v3/HYPOTH_GSE248413_Y` | GSE248413 sample0 (GSM7913150, title "Y, snRNA-seq"); `tissue_references.yaml:311-335` | snRNA; Brown Norway, male, 4 mo | 8,471 / 13 / 21 / 17,351 | per-sample consensus (PanglaoDB + ScType + LLM), **not re-annotated** | One pooled library; generic "Neurons" (1,792) beside "Glutamatergic" and "GABAergic neurons"; labelling route shown unreliable elsewhere (8b); "provisional" (`tissue_references.yaml:335`); no build provenance file |
| SMLINT | `references_v4/SMLINT_GSE272055` | GSE272055 proximal jejunum replicates 1-2 (GSM8392193-94); `tissue_references.yaml:337-363` | scRNA; SD, male | 16,193 / 13 / 23 / 17,895 | joint re-annotation | Epithelium-only isolation, no muscularis (`tissue_references.yaml:363`); 2,757 of 18,950 cells dropped; B cells 98% from one sample (`JOINT_REANNOTATION_RESULTS.md:102-104`) |
| LUNG | `references_v4/LUNG_native_pooled` | GSE273062 VeNx rep 1-2 (GSM8420505-06) + GSE252844 C3 (GSM8008844); `tissue_references.yaml:365-388` | scRNA; SD / outbred | 44,158 / 21 / 29 / 16,956 | joint re-annotation | Control arms of a pulmonary-hypertension and a blast-injury model (`REFERENCE_ACCESSIONS.md:28-29`); no AT1 cells; 783 endothelial cells; "weakest tissue" (`tissue_references.yaml:386-388`); no build provenance file |
| HIPPOC | `references_v4/HIPPOC_GSE295314` | GSE295314 six "VHPC-FED" libraries (GSM8945533/35/37/39/41/43); `tissue_references.yaml:391-421` | snRNA; strain not stated | 23,892 / 11 / 20 / 17,895 | joint re-annotation | 255,303 of 279,195 barcodes (91.4%) dropped as low quality; ambient-RNA sink; composition flagged **unreliable** (8d) |
| TESTES | `references_v3/TESTES_OMIX767_lifted` | NGDC OMIX767 arms C and E7W; `tissue_references.yaml:424-464` | scRNA (10x); SD | 5,836 / 6 / 12 / 14,482 | per-cell scoring against the source paper's marker lists | E7W is an EDS-treated arm (Leydig cells ablated, then restored); 42 Sertoli cells; Leydig, immune and endothelial cells dropped; differential use only (`tissue_references.yaml:451-460`) |
| SPLEEN | `references_v4/SPLEEN_GSE306764` | GSE306764 Spl-Ctrl (GSM9207348); `tissue_references.yaml:466-492` | scRNA (Singleron); Lewis, female, 8 wk | 13,013 / 14 / 23 / 22,050 | joint re-annotation (SPLEEN panel) | One library from adjuvant-treated controls; no mature erythroid cluster; 100 red-pulp macrophages; 34 endothelial cells; implausible fractions on the real bulk (`REFERENCE_RELABEL_PREREG.md:661-665`) |
| OVARY | `references_v4/OVARY_GSE343214` | GSE343214 "Normal rat" (GSM9947804); `tissue_references.yaml:528-552` | scRNA (10x); strain and age not stated | 9,104 / 18 / 28 / 30,562 | joint re-annotation (OVARY panel) | One rat; no linked paper; GEO text partly copied from another series; no oocytes (`tissue_references.yaml:546-552`) |

Tissues without a reference: ADRNL (`tissue_references.yaml:519-526`), COLON (`:494-516`), VENACV
(`:175-201`). MoTrPAC has 19 tissues with transcriptomics (`deconvolution/reference/motrpac_bulk_liftover_report.txt:34-52`).

---

## Stage 8a - Reference selection per tissue

### Research role
Question: which single-cell data define the cell-type basis that each tissue's bulk is projected onto?
Every downstream per-cell-type statement (fractions, pseudo-cells, the Aim-2 gate, per-cell-type DE)
inherits the basis. The claim resting on this step is that each reference is rat, native tissue, a
healthy or control arm, single cells or nuclei, and built from exactly the samples the project specifies.
It is checked mechanically by `validate_selection.py` against a locked `expect:` list in
`tissue_references.yaml`, and was checked once (2026-07-15, the 13 references of that date) against the GEO
records (`DECONV_REBUILD_RUNBOOK.md:255-288`). Verdict so far: the mechanical check passes for all 16
tissues today (run for this audit: 16 PASS, 0 WARN, 0 FAIL, 3 skipped). The biological adequacy of the
selections is weaker than the mechanical pass suggests: no reference matches the bulk's strain, most are
control arms of disease or intervention studies, nine of fifteen rest on one or two libraries, and three
MoTrPAC tissues have no reference.

### Inputs and outputs
- `reports/annotations/annotation_inventory.tsv` (gitignored; 812 rows; read by
  `deconvolution/build_reference.py:151`): one row per corpus sample with `accession`, `gsm`,
  `tissue_normalized`, `in_corpus`, `geo_organism`, `condition_resolved`, `sex_resolved`, `geo_title`,
  `geo_source_name`, `geo_cell_type`.
- `deconvolution/tissue_references.yaml` (552 lines; schema header `:1-36`): per tissue the study, tissue
  string, filters, the locked `expect:` contract (`n_samples`, `sample_ids`, `qc_dropped`, `organism`) and
  the build command.
- External (non-corpus) inputs for BLOOD, SPLEEN, OVARY, TESTES and the deposited-label objects for BAT and
  HEART: fetched by `deconvolution/fetch_external_refs.sh:63-101`, with byte sizes and md5 in
  `deconvolution/external_refs_md5.tsv:5-57`. HEART's SCP2828 object requires a manual, signed-in download
  (`fetch_external_refs.sh:13-14,91-92`); OMIX767 may also need a manual download (`:94-99`).
- Output: the list of sample IDs each build uses (not written separately; recorded in each reference's
  `cells_meta.tsv` `sample` column and, where present, `provenance_build_reference.json` `params.samples`).

### Procedure
For references built from the corpus, `select_samples()` (`deconvolution/build_reference.py:128-223`)
applies, in this order:
1. Base: inventory rows with `accession == study`, `tissue_normalized` equal to the tissue string
   (case-insensitive) and `in_corpus == true` (`build_reference.py:152-156`).
2. Organism gate: `geo_organism` equal to `Rattus norvegicus` (case-insensitive; default; `"any"` disables
   it) (`:159-160`).
3. Condition filter: `condition_resolved` in the given list (`:161-162`).
4. Sex filter on `sex_resolved` (`:163-164`). No tissue in the configuration uses it.
5. Title include, then title exclude: case-insensitive regular expressions searched in
   `geo_title | geo_source_name | geo_cell_type` (`:167-174`).
6. Explicit `sample_ids`, validated against what survived steps 1-5; a listed ID that did not survive is a
   fatal error (`:175-183`).
7. GSM de-duplication: sort by `sample_id`, keep the first row per `gsm` (`:186-190`).
8. Reference-QC drop (8c): samples whose title text matches the SPATIAL, ENGINEERED or BULK patterns are
   removed unless `ALLOW_NONNATIVE_REF=1`; DEVEL matches only print a warning (`:197-222`).

References not built from the corpus (BLOOD composite, SPLEEN, OVARY) do not pass through
`select_samples()`; their libraries are hard-coded file paths in `deconvolution/joint_reannotate.py:69-74,83-90`.
TESTES was ingested into corpus format first (8d) and then selected with explicit `sample_ids`.

The decision record is `deconvolution/REFERENCE_SELECTION_PLAN.md` (2026-07-15, with later update notes
`:22-39,104-106`). It states that the per-tissue choices came from "a 19-tissue fan-out (one researcher per
tissue, online-accession verification, consolidation)" (`:6-7`), in response to a collaborator's note, and
that restrictions were cross-checked against `deconvolution/StudyDescriptions.docx` (`:111-129`). The
later additions are registered in `deconvolution/REFERENCE_RELABEL_PREREG.md` §12 (BLOOD, `:275-319`) and
§14 (SPLEEN, OVARY, `:542-581`).

### Models and software
No model. Python 3.12.11 environment (`requirements.lock.txt:4`), pandas 2.3.3 (`requirements.lock.txt:155`),
PyYAML; regular expressions via the standard `re` module with `re.I`.

### Parameters, thresholds and flags

Selection filters per tissue (all from `deconvolution/tissue_references.yaml`; "-" = not set). Organism is
`Rattus norvegicus` and `dedup_gsm` is true for every corpus-built tissue.

| Tissue | `study` / `tissue` | `conditions` | `title_include` | `sample_ids` | Samples expected | Where set | Rationale given |
|---|---|---|---|---|---|---|---|
| KIDNEY | GSE240658 / kidney | `["No treatment"]` | - | - | 4 | `:46-61` | control arm; snRNA captures PT; kept over Ma scRNA (`:65`) |
| LIVER | GSE220075 / liver | - | - | - | 4 (+2 QC-dropped Visium) | `:72-87` | 4 snRNA sham samples; Visium dropped by the gate (`:91`) |
| CORTEX | GSE303115 / cortex | - | - | - | 2 | `:122-137` | organism gate drops 5 other species; de-dup 4 rows to 2 (`:124,129`) |
| HEART | GSE280111 / left ventricle | - | - | - | 19 | `:149-168` | healthy Wistar left ventricle; de-dup 38 rows to 19 (`:156,173`) |
| WATSC | GSE137869 / white adipose tissue | - | `[-]Y($\|\b\|_)` | - | 2 | `:208-223` | young arm only; drops old and calorie-restricted arms (`:212,227`) |
| SKMGN, SKMVL | GSE137869 / skeletal muscle | - | `[-]Y($\|\b\|_)` | - | 2 | `:236-251,262-277` | one shared young-arm build (`:231,255,281`) |
| BAT | GSE244451 / subscapular brown adipose tissue | - | - | - | 2 | `:289-304` | snRNA recovers brown adipocytes (`:309`) |
| HYPOTH | GSE248413 / hypothalamus | - | `(^\|[^A-Za-z])Y([^A-Za-z]\|$)` | - | 1 | `:316-331` | young arm; `condition_resolved` reads "no treatment" for young and old, so the title filter is mandatory (`:319-320`) |
| SMLINT | GSE272055 / proximal jejunum | - | - | - | 2 | `:344-359` | healthy adult SD (`:363`) |
| LUNG | pooled: GSE273062 / lung, GSE252844 / lung | `["vehicle and normoxia"]`, `["Control"]` | - | - | 3 | `:373-384` | control arms; GSE242310 dropped (both arms postnatal day 10) (`:377,386-388`) |
| HIPPOC | GSE295314 / brain | `["Fed"]` | - | - | 6 | `:398-414` | "Fed" arm; drops the "Fast" arm (`:401`) |
| TESTES | OMIX767 / testis | - | - | `OMIX767_sampleC, OMIX767_sampleE7W` | 2 | `:429-442` | "normal-testosterone arms" (`:433`) |
| BLOOD | builder `blood_composite` | n/a | n/a | n/a | 6 libraries | `:97-104` | option B of the BLOOD proposal (`:107-113`) |
| SPLEEN | builder `external_reference` | n/a | n/a | n/a | 1 library | `:470-482` | whole spleen, not sorted (`:485-492`) |
| OVARY | builder `external_reference` | n/a | n/a | n/a | 1 library | `:531-543` | saline control of a PCOS model (`:546-552`) |

Other flags:

| Name | Value | Where set | What it controls | Rationale given |
|---|---|---|---|---|
| `--organism` | `Rattus norvegicus` | `deconvolution/build_reference.py:391` | organism gate | a six-species cortex study had seeded an ~85% non-rat reference (`build_reference.py:157-158`) |
| `--no-dedup-gsm` | off (de-dup on) | `build_reference.py:400-402` | one sample per GSM | the same library ingested twice (`build_reference.py:149-150`) |
| `ALLOW_NONNATIVE_REF` | unset | `build_reference.py:214` | bypass of the QC drop | none of the 15 references sets `allow_nonnative` (checked in `tissue_references.yaml`) |
| LUNG sample list | `GSE273062_sample0, _sample2, GSE252844_sample0` | hard-coded, `deconvolution/build_lung_pooled.py:25-28` | which lung samples are pooled | see Deviations |

### Why each reference was chosen and what was rejected
From `deconvolution/REFERENCE_SELECTION_PLAN.md:81-99` unless another source is given.

| Tissue | Chosen because | Rejected or replaced |
|---|---|---|
| KIDNEY | snRNA captures the proximal-tubule parenchyma (`:81`) | Ma 2020 (GSE137869) scRNA kidney: under-captures PT (`:81,145-155`) |
| LIVER | snRNA captures hepatocytes; the only tissue with a cross-method check on real bulk (`:82`) | Ma 2020 scRNA liver (`:82`); the study's 2 Visium samples (QC drop) |
| CORTEX | only native rat cortex snRNA in the corpus (`:83`) | the earlier "union_merged" build, about 85% non-rat (`:83`) |
| HEART | "only clean rat LV atlas" (`:84`) | GSE155699 ("SHR model, no cardiomyocytes") (`:84`) |
| WATSC | only rat white adipose in the corpus (`:86`) | the build pooling young, old and calorie-restricted arms (`:86`) |
| SKMGN | unify both muscles on one study, a collaborator's preference (`:87,165-170`) | GSE184413 (F344/BN gastrocnemius; the only strain-matched and the only previously benchmarked muscle reference) (`:87,156-159`) |
| SKMVL | as SKMGN (`:88`) | GSE254371 (52% rat-mouse chimera from an iPSC study) (`:88`) |
| BLOOD | composite with erythroid and neutrophil profiles (`REFERENCE_RELABEL_PREREG.md:277-282`) | GSE285476 one-rat PBMC (kept as `previous_reference_dir`); GSE239996 sorted bulk neutrophils, proposed in `REFERENCE_LABEL_AUDIT.md:229-231` and not used; others in `BLOOD_REFERENCE_PROPOSAL.md:83-93` |
| BAT | snRNA recovers brown adipocytes (`:90`) | Ma 2020 scRNA BAT (same protocol as WAT, no adipocytes) (`:90`) |
| HYPOTH | only rat hypothalamus in the corpus (`:91`) | Ma 2020 whole brain (not region-specific) (`:91`) |
| SMLINT | "only clean rat small-intestine scRNA" (`:92`) | none named |
| LUNG | native control arms pooled (`:93`) | GSE178405 (in-vitro engineered lung); GSE242310 (both arms neonatal) (`:93`); GSE196313 (embryonic) and GSE312833 (CD31-sorted endothelium) (`REFERENCE_QC.md:54-58`) |
| HIPPOC | native, non-transgenic ventral hippocampus (`:94`) | GSE305314 (wild-type arm of a tauopathy model; a debris cluster took 64% of θ) (`:94`) |
| TESTES | "only rat testis scRNA" (`tissue_references.yaml:452`) | GSE268104 snRNA named as an alternative, not used (`tissue_references.yaml:460`) |
| SPLEEN | whole spleen, RBC-lysed, not sorted (`REFERENCE_RELABEL_PREREG.md:547-551`) | GSE186158 (sorted live immune cells) (`REFERENCE_RELABEL_PREREG.md:550-551`) |
| OVARY | the candidate found in the 2026-10-31 re-search (`REFERENCE_RELABEL_PREREG.md:545-554`) | CRA008987 (spatial only) (`tissue_references.yaml:552`) |
| COLON | none | GSE326856: sorted endothelial, immune and stromal cells, no epithelium (`REFERENCE_RELABEL_PREREG.md:555-556`) |
| ADRNL, VENACV | none | VENACV proxy (GSE280111 pulmonary veins) built and dropped as lung-contaminated (`tissue_references.yaml:176-179`) |

### Statistical procedures
None. Selection is a deterministic filter on metadata.

### Checks, validation and their results
- `deconvolution/validate_selection.py` (`:109-189`): per tissue, (1) resolved selection equals
  `expect.sample_ids` and `n_samples` (`:146-150`); (2) the QC gate dropped exactly `expect.qc_dropped`
  (`:151-155`); (3) every sample's `geo_organism` equals the expected organism (`:156-160`); (4) the h5ad,
  cell-type and consensus files exist (`:161-164`); (5) the built `cells_meta.tsv` used exactly the expected
  samples (`:173-183`); plus a WARN if a selected sample's condition or title matches a treatment/aged
  pattern (`:35-38,165-170`). Exit code 2 on any FAIL (`:220`). **Result when run for this audit:** 16 PASS,
  0 WARN, 0 FAIL, 3 skipped (ADRNL, COLON, VENACV). For BLOOD, SPLEEN and OVARY only check (5) applies
  ("built outside the corpus", `:127-128`).
- The Stage 8/9 driver repeats this check before each tissue and refuses a FAIL
  (`pipeline/run_deconv_all.py:110-113`).
- 2026-07-15 record: selections "verified at three levels", the third an "adversarial GEO-record check (13
  verifiers WebFetched each study + GSMs)", all 13 confirmed (`DECONV_REBUILD_RUNBOOK.md:257-260,268-282`).
  The verifiers' outputs are NOT FOUND in the repository (looked in `deconvolution/`, `docs/`, `reports/`);
  only the summary table exists. BLOOD composite, SPLEEN and OVARY postdate that check; their facts were
  "checked against the GEO FTP family SOFT files" (`REFERENCE_RELABEL_PREREG.md:545-546`).

### Deviations, known defects and caveats
1. **Strain.** The bulk is F344; no reference is. The one F344 reference (GSE184413) was replaced; a
   head-to-head on gastrocnemius was proposed and is recorded as pending (`REFERENCE_SELECTION_PLAN.md:156-159`;
   `tissue_references.yaml:255`). No result of that comparison was found.
2. **Control arms of intervention studies.** KIDNEY (diet/nephrosis study), LUNG (pulmonary-hypertension and
   blast-injury controls), HIPPOC ("Fed" = refed after a 24 h fast, `DECONV_REBUILD_RUNBOOK.md:287`), SPLEEN
   and BLOOD's whole-blood library (rats given complete Freund's adjuvant, `tissue_references.yaml:109,486-488`),
   OVARY (saline control of a PCOS model).
3. **TESTES pools a treated arm.** E7W is the arm in which Leydig cells were ablated and have regrown
   (inventory title "E7W (Leydig restored, normal testosterone)"; study described as an "EDS Leydig-ablation
   study", `REFERENCE_ACCESSIONS.md:49`). The inventory rows for OMIX767 carry `condition_resolved = control`
   for both arms, and the treatment-hint pattern (`validate_selection.py:35-38`) has no term that matches,
   so no WARN is raised.
4. **LUNG: the build ignores the configuration's filters.** `build_lung_pooled.py:25-28` hard-codes the three
   sample IDs; the `pooled_studies` conditions in `tissue_references.yaml:377-379` are used only by the
   validator (`validate_selection.py:83-87`). The two agree today (validator PASS).
5. **Replicate count.** HYPOTH, SPLEEN and OVARY rest on one library; WATSC, MUSCLE, BAT, SMLINT, CORTEX and
   TESTES on two. The Ma 2020 libraries pool 2-3 rats each and there is one library per tissue, sex and arm
   (`REFERENCE_STUDY_GSE137869_MA2020.md:46-50`).
6. **Stale committed manifests.** `deconvolution/reference/canonical_references.tsv:2-11` still lists the
   pre-July references (e.g. LUNG GSE178405, SKMVL GSE254371, BLOOD GSE285476). No code reads it (searched
   `*.py`, `*.R`, `*.sh`, `*.yaml`); older docs call it authoritative
   (`DECONVOLUTION_PIPELINE_REPORT.md:1561-1563`). `tissue_references.yaml:36` still carries
   `version: v3-2026-07-15`, its schema comment says `allow_nonnative` is used for SPLEEN (`:32`) although no
   entry sets it, and the HIPPOC note still describes a 278k-cell reference needing ~500 GB (`:421`).
7. **Docs disagree on one lung candidate.** `REFERENCE_ACCESSIONS.md:51` lists GSE312833 as a candidate with a
   healthy sham arm; `REFERENCE_QC.md:58` lists the same accession as CD31-sorted endothelium, not to be used.
   It is not used.
8. `pipeline/README.md:25-33` names `results_refs4all/` as the default output tree and says there is no
   Stage 11; the configuration's default is `results_refs4all_rebuild/` (`config/pipeline_config.yaml:406`)
   and `pipeline/run_stage11.py` exists (`DECONV_REBUILD_RUNBOOK.md:133`). The references are the same in
   both trees.

### Methods references
- GSE137869: Ma S, et al. Caloric Restriction Reprograms the Single-Cell Transcriptional Landscape of
  Rattus Norvegicus Aging. Cell 2020;180(5):984-1001.e22, PMID 32109414 (as given in
  `deconvolution/REFERENCE_STUDY_GSE137869_MA2020.md`, title block).
- GSE306764: "Que et al., J Neuroinflammation 2026; PMID 41709234"; GSE335156: "Lai et al., FASEB J 2026;
  PMID 42141902" (as given in `deconvolution/BLOOD_REFERENCE_PROPOSAL.md:85-86`).
- OMIX767: "Guan et al., Sci Data 2022, PMID 35338159" (as given in `deconvolution/annotate_omix767_testes.py:8`).
- GSE244451: "Thompson et al. 2024, Single Cell Portal SCP2384"; GSE280111: "Nagai/Ellinor Cell Reports 2024",
  SCP2828 (as given in `deconvolution/adopt_authors_labels.py:33,43,48`).
- Other reference studies (GSE240658, GSE220075, GSE303115, GSE248413, GSE272055, GSE273062, GSE252844,
  GSE295314, GSE343214): no publication is cited in the code or docs read; GSE295314 is linked to "Nat Commun
  2025, PMID 40500290" in `DECONV_REBUILD_RUNBOOK.md:264`; GSE343214 has "no linked paper"
  (`tissue_references.yaml:548`).

---

## Stage 8b - Cell-type labels

### Research role
Question: what are the cell types of each reference, and which cells belong to each? BayesPrism estimates
one fraction and one expression profile per label, so a wrong, duplicated or sample-specific label becomes
a wrong or near-empty (tissue x cell type) block downstream. The claim is that labels name the same cells
in every sample of a reference and are supported by the cluster's own marker genes. It is checked by four
registered checks on the final labels (K1-K4, `deconvolution/REFERENCE_RELABEL_PREREG.md:239-252`) and a
later literature check of the manual decisions. Verdict so far: for the twelve label sets that went
through joint re-annotation (eight corpus references, two blood libraries, SPLEEN, OVARY) the consistency
check passes wherever it is computable; the manual decisions (160) were made and literature-checked by an
AI assistant and have not been read by a domain expert (`JOINT_REANNOTATION_RESULTS.md:173-176,211-212`;
`JOINT_REANNOTATION_LITCHECK.md:3-7`). HYPOTH still carries the older per-sample labels.

Four labelling routes are in the default references:

| Route | References | Code |
|---|---|---|
| 1. Per-sample consensus (PanglaoDB + ScType + LLM) | HYPOTH only | `analysis/annotate_celltypes.py`, `analysis/annotate_sctype.py`, `analysis/consensus_annotations.py` |
| 2. Authors' deposited per-cell labels | BAT, HEART | `deconvolution/adopt_authors_labels.py` |
| 3. Source paper's marker lists, scored per cell | TESTES | `deconvolution/annotate_omix767_testes.py` |
| 4. Joint re-annotation with fixed marker panels | KIDNEY, LIVER, CORTEX, WATSC, MUSCLE, SMLINT, LUNG, HIPPOC, BLOOD (two libraries), SPLEEN, OVARY | `deconvolution/joint_reannotate.py`, `deconvolution/reference_marker_panels.yaml`, `deconvolution/joint_reannotation_review.tsv` |

Not active in the default: the label-merging schemes `brain`, `muscle`, `lung` (`deconvolution/build_reference.py:69-115`;
skipped when joint labels are supplied, `:431-434`; HYPOTH, BAT, HEART and TESTES have `label_scheme: none`);
`deconvolution/merge_consensus_labels.py` (July muscle merge, `:27-38`); `deconvolution/annotate_from_authors.py`
(a marker-panel route for BAT, `:33-44`, superseded by route 2 because "a hand-built marker panel FAILS for
dominant-parenchyma tissues", `adopt_authors_labels.py:6-9`).

### Inputs and outputs
- Corpus QC matrices `data/training/preprocessed/qc_matrices/<sample>.h5ad` (raw counts; `config/pipeline_config.yaml:367`).
- Route 1-3 outputs, in the corpus annotation folders (`config/pipeline_config.yaml:368-369`):
  `data/training/cell_annotations/<sample>/<sample>_celltypes.tsv` (barcode, leiden, cell_type, confidence) and
  `data/training/cell_annotations_consensus/<sample>/<sample>_consensus.tsv` (cluster, n_cells, panglao_label,
  sctype_label, consensus_label, consensus_source).
- Route 4 outputs, `data/deconvolution/joint_labels/<REF>/` (`deconvolution/joint_reannotate.py:34-36`):
  `cells.tsv.gz`, `clusters.tsv`, `scores.tsv`, `panel_genes.tsv`, `group_sums.npz`, `settings.json`,
  `umap.tsv.gz`; after `finalize`: `cells_final.tsv.gz`, `clusters_final.tsv`, `checks.json`,
  `label_checks.tsv`, `old_vs_new.tsv`. Fourteen label sets exist on disk (the twelve above plus BLOOD_BM and
  HEART_GSE307551, which are used for checks only).
- Committed decision files: `deconvolution/reference_marker_panels.yaml` (458 lines);
  `deconvolution/joint_reannotation_review.tsv` (160 decisions; rows per reference: KIDNEY 9, LIVER 6,
  CORTEX 5, WATSC 16, MUSCLE 10, SMLINT 12, LUNG 20, HIPPOC 22, BLOOD_PBMC 17, BLOOD_WB 6, BLOOD_BM 12,
  SPLEEN 11, OVARY 10, HEART_GSE307551 4; tabulated from the file);
  `deconvolution/joint_reannotation_splits.tsv` (2 rows); `deconvolution/joint_reannotation_clusters.tsv`
  (14 SHA-256 records); `deconvolution/joint_reannotation_litcheck.tsv` (156 rows).

### Procedure

**Route 1 - per-sample consensus labels (corpus; used for HYPOTH).** All three steps run one sample at a time.
1. *Clustering* (`analysis/annotate_celltypes.py:326-344`): `normalize_total(target_sum=1e4)`, `log1p`;
   `highly_variable_genes(n_top_genes=min(2000, n_genes), flavor="seurat_v3", layer=None, span=0.3)`;
   `scale(max_value=10)` on the HVG subset; PCA with `min(50, n_cells-1, n_hvg-1)` components; neighbours with
   `n_neighbors=min(15, n_cells-1)`, `n_pcs=min(30, n_pcs)`; Leiden at resolution 0.8 (`:80`). No seed is
   passed (scanpy defaults apply).
2. *PanglaoDB scoring* (`analysis/annotate_celltypes.py`): markers from
   `PanglaoDB_markers_27_Mar_2020.tsv.gz` (`:77`), rows with species `Mm Hs`, `Mm` or `Hs` (`:122`), symbols
   upper-cased and mapped to rat Ensembl IDs through the corpus gene universe (`:105-113,135-147`). Every
   PanglaoDB cell type with at least 3 markers present (`:81,235`) is scored with `sc.tl.score_genes` (`:242`),
   averaged per cluster (`:247`); the cluster takes the highest-scoring type (`:275-281`), with no abstention.
   A tissue hint only adds organ-specific marker sets to the global set (`:353-367`). A score under 0.1 is
   flagged, not rejected (`:82,422`).
3. *ScType scoring* (`analysis/annotate_sctype.py`): marker database `ScTypeDB_full.xlsx` (`:61`); tissue lists
   chosen by exact, then substring, match of `tissue_normalized` against a fixed map, falling back to the
   "Immune system" list (`:87-102,358-375`; hypothalamus maps to Brain + Immune system, `:95`). Clustering as
   in step 1 (`:408-422`). Marker genes are z-scaled (`:461-462`); per cell and type the score is
   `(2 x sum of positive markers - sum of negative markers) / sqrt(number of markers)` for types with at least
   2 positive markers present (`:259-281`); scores are summed over a cluster's cells and the best type is
   assigned, or "Unknown" if the best sum is below `n_cells / 4` (`:318-330`).
4. *LLM reconciliation* (`analysis/consensus_annotations.py`): the unique (PanglaoDB label, ScType label,
   MoTrPAC tissue) triples (`:117-136`), each with the top 5 marker genes of its single largest cluster
   (`:139,156,170`), are sent in batches of 50 (`:272`) to `claude-haiku-4-5-20251001` with `max_tokens=8192`
   (`:283-284`) and a fixed prompt (`:232-262`) that asks for one consensus name and a source tag (`panglao`,
   `sctype`, `merged`, `corrected`). The prompt's eight rules (`:234-250`): (1) synonyms take the most
   standard name; (2) a tissue-inappropriate PanglaoDB label yields to a tissue-appropriate ScType label; (3)
   ScType "Unknown" yields to a plausible PanglaoDB label; (4) a sample not scored by ScType takes the
   PanglaoDB label; (5) two plausible but different labels: the more specific, with the markers as
   tie-breaker; (6) standardised names; (7) immune types without a tissue prefix; (8) at most five words. No temperature or seed is set. A batch whose reply fails
   to parse is skipped (`:308-311`); clusters whose triple has no reply take the PanglaoDB label
   (`fallback_panglao`, `:378-382`). The name is keyed by the label pair, not by the cluster.
5. *Join at build time*: barcode to Leiden cluster to consensus label; cell state = `<sample>_c<leiden>`
   (`deconvolution/build_reference.py:245-250`).

**Route 2 - authors' deposited labels (BAT, HEART)** (`deconvolution/adopt_authors_labels.py`).
- BAT: the GEO supplement `GSE244451_PVAT_BAT_SCPAnnotated.h5ad`, column `celltype`, rows with `tissue == "BAT"`;
  our samples 1 and 3 are mapped to the authors' `Sample` values "1" and "3"; the barcode core is the text
  before the first "-"; `Adipocytes` and `Adipocytes_2` are merged into "Brown adipocytes" (`:31-41,174-185`).
- HEART: the Single Cell Portal object `ratmap_scp.h5ad` (13 GB; only `obs` is read through h5py, `:115-144`),
  column `cell_type__ontology_label`, rows with `tissue == "LV"`; the authors' barcodes are
  `<sample>::<barcode>` and the core is the part after `::`; each of our 19 samples is matched to the authors'
  sample with the largest barcode overlap (`:157-165`; the code comment reports 74-92% overlap, `:46`); the
  ontology labels (33 according to the comment at `:56`) are renamed by a 28-entry table: 25 entries map to
  17 basis names and three (Schwann cell, peripheral nervous system neuron, club cell) to "Unknown"; labels
  not listed are kept as they are (`:56-84`).
- Cells without a deposited label become "Unknown" and leave the reference at build time
  (`:10-11`; `build_reference.py:54,347-350`). The label is written as a per-cell pseudo-cluster into the
  corpus annotation folders (`:95-112`), so the cell state is `<sample>_c<label code>`: one state per sample
  and label (HEART 249 states, BAT 12).

**Route 3 - source-paper markers, per cell (TESTES)** (`deconvolution/annotate_omix767_testes.py`).
For each of seven classes in the authors' table `OMIX767-20-02.xlsx` (`:38,42-47`) the panel is the top 50
genes by positive log2 fold change (`:73-80,147`), extended with markers taken from the paper's text
(`:50-56,81-82`); three further somatic classes (immune, endothelial, peritubular myoid) have marker-only
panels (`:58-62,83-84`). Panels with fewer than 3 genes found are not scored (`:88,101`). After
`normalize_total(1e4)` and `log1p` (`:99`) each cell is scored per class with `sc.tl.score_genes` (`:103`) and
takes the best class (`:106-107`). Gate: a cell called Sertoli, Leydig, immune, endothelial or myoid must
have at least one raw count of that class's gate marker (Sox9; Cyp17a1 or Cyp11a1; Ptprc; Cdh5 or Vwf; Myh11;
`:66-69`), otherwise it is reassigned to its best germ-cell class (`:108-122`). Output is written in the
corpus annotation format with the class code as the cluster (`:130-142`).

**Route 4 - joint re-annotation** (`deconvolution/joint_reannotate.py`; registered in
`REFERENCE_RELABEL_PREREG.md` §10, `:172-258`).
1. *Cells.* Every QC cell of the reference's samples, with the production loader and gene join
   (`:284-297`); LUNG as its build (per-study outer join, then genes detected in at least 10 cells, `:285-291`).
   External libraries (blood, SPLEEN, OVARY, second heart set) are read from matrix files, filtered at
   >= 200 genes and mitochondrial fraction <= 0.15 (`:173,187,249,257`; mitochondrial genes by the `mt-` symbol
   prefix, `:184`, or 13 RefSeq names, `:93,253-254`) and put in the bulk's gene-ID space (`:194-195,264-267`).
   The old labels are carried along for comparison and play no part in assignment (`:300-301`).
2. *Clustering* (`cluster()`, `:307-330`): `normalize_total(1e4)`, `log1p`; 3,000 HVGs with
   `flavor="seurat_v3"` on the raw-count layer, `batch_key="sample"` when there is more than one sample;
   `scale(max_value=10)`; PCA with 50 components, seed 0; Harmony on `sample`
   (`scanpy.external.pp.harmony_integrate`, seed 0, other arguments at their defaults) when there is more than
   one sample; neighbours with `n_neighbors=15`, `n_pcs=30` on the Harmony-adjusted (or plain) PCA, seed 0
   ("kNN 15 on 30 components", `REFERENCE_RELABEL_PREREG.md:216-217`); Leiden with `flavor="igraph"`, `resolution=1.0`, `n_iterations=2`, undirected, seed 0.
   Clusters are named `j0, j1, ...`.
3. *Splits* (`:333-350`): a cluster listed in `joint_reannotation_splits.tsv` is re-clustered alone (Leiden,
   same resolution and seed, `restrict_to` that cluster) into `j<N>.<k>`. Two clusters are listed: CORTEX j24
   and HIPPOC j13 (`joint_reannotation_splits.tsv:3-4`).
4. *Panels* (`reference_marker_panels.yaml`): per reference, top-level classes each with a compartment and
   3-6 marker symbols, and optional sub-classes; shared classes are pulled in by `use:` (`:1-15`). Symbols are
   resolved to the reference's gene IDs through the Ensembl gene names, then the corpus gene universe, then
   the RGD symbol table (`joint_reannotate.py:116-140`). A panel with fewer than 2 symbols found is not scored
   (`:98,157,368`).
5. *Score* (`:360-370,483-486`): per cluster, the mean log-normalised expression of each gene; each gene is
   min-max scaled across clusters; a panel's score is the mean of its genes' scaled values.
6. *Rule* (`assign()`, `:388-420`): among top-level classes, the best is accepted if its score is >= 0.5 and
   it beats the runner-up by >= 0.15; if another top-level class from a different compartment also scores
   >= 0.5 the cluster is flagged "possible doublet"; a cluster with no class >= 0.5, a near-tie, or a doublet
   flag goes to review. An accepted class descends into its sub-classes by the same two tests among siblings
   and stops at the parent when no child passes.
7. *Evidence for review* (`:489-520`): for each cluster the 25 genes with the highest
   `log2(CPM_cluster + 1) - log2(CPM_rest + 1)` on pooled counts among genes detected in >= 25% of its cells;
   panel scores; cells per sample, largest sample share and normalised sample entropy; median counts and
   genes per cell; the four most frequent old labels.
8. *Review decisions* (`joint_reannotation_review.tsv`, applied by `final_labels()`, `:561-575`): a decision
   replaces the automatic label; a cluster with neither is a fatal error; clusters under 20 cells are set to
   DROP (`:99,516,574`). Allowed decisions were registered as a panel label, a name outside the panels
   "supported by at least 3 markers with a literature basis", or DROP (doublet, low quality, not decidable)
   (`REFERENCE_RELABEL_PREREG.md:229-235`). "The reviewer (Claude) has seen the audit's marker lists for the
   old labels" (`:234-235`).
9. *Rules added during review*, logged the same day before the affected checks were read
   (`REFERENCE_RELABEL_PREREG.md:148-170`): (i) every automatically accepted label is also checked against its
   markers and may be overridden only where the markers contradict it, recorded as "OVERRIDE of the accepted
   label ..."; (ii) a cluster accepted at a parent label is made specific when at least 3 marker genes name
   the type ("Under-call made specific"); (iii) a T-cell cluster is called CD4+ or CD8+ only where the CPM of
   Cd40lg against Cd8a/Cd8b is lopsided, because Cd4 and Il7r are absent from every reference's gene space;
   (iv) the cluster splits of step 3.
10. *Finalize* (`:605-646`): writes per-cell final labels and runs K1 and K2. A cluster-ID guard compares a
    SHA-256 of the sorted (sample, barcode, cluster) rows with the recorded one and refuses to apply decisions
    to a different clustering (`:424-464,608-612`).
11. *Literature check* (2026-10-28; `JOINT_REANNOTATION_LITCHECK.md`): "Four agents, one per tissue group"
    searched the literature for each decision and most "also read pooled per-cluster CPM" (`:14-21`). Every
    QUESTIONABLE call and every flagged LUNG cluster was re-checked against pooled CPM (`:23-24`). Three
    labels changed (`:46-50`): LUNG j37 "Mesothelial cells" to DROP (444 cells; mesothelial-macrophage/AT2
    doublets); CORTEX j24.5 "Monocytes" to "Neutrophils" (29 cells); HIPPOC j24 "Inhibitory neurons" to
    "Thalamic neurons" (598 cells). LUNG, CORTEX and HIPPOC were rebuilt before their tissues were deconvolved
    (`REFERENCE_RELABEL_PREREG.md:349-357`). Suggested refinements that move no cells were not applied
    (`JOINT_REANNOTATION_LITCHECK.md:68-90`).

**The audit that led to route 4** (`deconvolution/audit_reference_labels.py`; registered in
`REFERENCE_RELABEL_PREREG.md:56-91`; results `REFERENCE_LABEL_AUDIT.md`). For each (sample, label) group the
summed counts as log2(CPM + 1); genes with mean >= 1 over groups; the 2,000 most variable; each gene centred
over groups; Pearson r between groups (`audit_reference_labels.py:36,67-78`). A label found in at most half
the samples was proposed MERGE into the best-matching multi-sample label if r >= τ (the 10th percentile of
same-label cross-sample r, per reference when it has at least 5 such labels, else pooled) and at least 0.05
above the next candidate (`:36,136-143,174-179`). The audit found 54 single-sample labels
(`REFERENCE_RELABEL_PREREG.md:19-21`), a HIPPOC self-similarity of 0.12, and labels contradicted by their
markers (`REFERENCE_LABEL_AUDIT.md:11-38,46-58`); the
merge rule was replaced by joint re-annotation for the eight corpus references before any joint clustering
(`REFERENCE_RELABEL_PREREG.md:144-147`).

### Models and software
- scanpy 1.12, anndata 0.12.10, igraph 1.0.0, leidenalg 0.11.0, scikit-misc 0.5.2 (`requirements.lock.txt:195,31,91,108,197`;
  the same versions are installed in the project environment); harmonypy 0.0.10 (`requirements.txt:28`;
  `REFERENCE_RELABEL_PREREG.md:216`; absent from `requirements.lock.txt`, which was captured 2026-10-05,
  `requirements.lock.txt:4`). `requirements.txt:25-26` gives only lower bounds for scanpy and anndata.
- PanglaoDB marker table of 27 March 2020 (`analysis/annotate_celltypes.py:77`). ScTypeDB: file
  `data/training/cell_annotations/ScTypeDB_full.xlsx` (`analysis/annotate_sctype.py:61`); its version or date
  is NOT FOUND in the code.
- LLM for route 1: Anthropic `claude-haiku-4-5-20251001` (`analysis/consensus_annotations.py:283,334`),
  `anthropic` client 0.94.0 (`requirements.lock.txt:33`).
- h5py 3.14.0 for the HEART label object (`requirements.lock.txt:84`); openpyxl 3.1.5 for the TESTES table
  (`requirements.lock.txt:151`).
- No other automatic annotator is used: CellTypist 1.7.1 is installed (`requirements.lock.txt:43`) but no
  script under `deconvolution/`, `analysis/`, `pipeline/` or `lib/` calls it (searched for `celltypist`).

### Parameters, thresholds and flags

| Name | Value | Where set (path:line) | What it controls | Rationale given |
|---|---|---|---|---|
| Route 1 Leiden resolution | 0.8 | `analysis/annotate_celltypes.py:80,344`; `analysis/annotate_sctype.py:64,422` | per-sample clusters | none stated |
| Route 1 HVGs | min(2000, n_genes), `seurat_v3`, span 0.3 | `analysis/annotate_celltypes.py:330-332` | features for PCA | none stated |
| Route 1 PCs / neighbours | <= 50 PCs; 15 neighbours on <= 30 PCs | `analysis/annotate_celltypes.py:336-343` | kNN graph | none stated |
| PanglaoDB minimum markers | 3 | `analysis/annotate_celltypes.py:81` | which types are scored | none stated |
| PanglaoDB low-confidence flag | 0.1 | `analysis/annotate_celltypes.py:82` | flag only | none stated |
| ScType weights | +2 positive, -1 negative, / sqrt(n markers) | `analysis/annotate_sctype.py:272-279` | score | "from Ianevski et al. 2022" (`:231-236`) |
| ScType unknown threshold | cluster score < n_cells / 4 | `analysis/annotate_sctype.py:327` | "Unknown" | same |
| LLM model, batch, max tokens | `claude-haiku-4-5-20251001`, 50, 8192 | `analysis/consensus_annotations.py:272,283-284` | consensus names | "Haiku context limit" (`:271`) |
| Markers shown to the LLM | top 5 of 1 cluster per triple | `analysis/consensus_annotations.py:139,170` | LLM context | none stated |
| BAT label merge | `Adipocytes`, `Adipocytes_2` -> Brown adipocytes | `deconvolution/adopt_authors_labels.py:40` | one adipocyte type | collinear subtypes (`:39`) |
| HEART label map | 28 entries: 25 -> 17 basis names (16 survive the build; Adipocytes, 17 cells, does not), 3 -> Unknown | `deconvolution/adopt_authors_labels.py:58-84` | basis types | "immune kept resolved; structural collapsed" (`:56-57`) |
| HEART minimum barcode overlap | 0.30 | `deconvolution/adopt_authors_labels.py:55,166` | warning only (see Deviations) | none stated |
| TESTES top DEGs per class | 50, log2FC > 0 | `deconvolution/annotate_omix767_testes.py:79-80,147` | panels | none stated |
| TESTES minimum panel genes | 3 | `deconvolution/annotate_omix767_testes.py:88,101` | which classes are scored | none stated |
| TESTES somatic gate markers | Sox9; Cyp17a1, Cyp11a1; Ptprc; Cdh5, Vwf; Myh11 | `deconvolution/annotate_omix767_testes.py:66-69` | somatic calls | plain arg-max over-assigns immune (`:11-14,63-65`) |
| `N_HVG` | 3000 | `deconvolution/joint_reannotate.py:97` | joint HVGs | none stated |
| PCA components | 50 | `deconvolution/joint_reannotate.py:318` | PCA | none stated |
| `N_PCS` | 30 | `deconvolution/joint_reannotate.py:97,325` | components used for kNN | none stated |
| `N_NEIGHBORS` | 15 | `deconvolution/joint_reannotate.py:97,325` | kNN | none stated |
| `RESOLUTION` | 1.0 | `deconvolution/joint_reannotate.py:97,326` | Leiden | none stated |
| `SEED` | 0 | `deconvolution/joint_reannotate.py:97` | PCA, Harmony, kNN, Leiden, UMAP | none stated |
| Leiden iterations, flavour | 2, igraph, undirected | `deconvolution/joint_reannotate.py:326-327` | Leiden | none stated |
| External-library QC | >= 200 genes, mito fraction <= 0.15 | `deconvolution/joint_reannotate.py:173,249` | cells kept | "the corpus QC" (`:173`; `config/pipeline_config.yaml:212-214`) |
| `ACCEPT` | 0.5 | `deconvolution/joint_reannotate.py:98` | minimum panel score | none stated; "nothing here is tuned to the data" (`reference_marker_panels.yaml:14-15`) |
| `MARGIN` | 0.15 | `deconvolution/joint_reannotate.py:98` | lead over runner-up | none stated |
| `DOUBLET` | 0.5 | `deconvolution/joint_reannotate.py:98` | second-compartment flag | none stated |
| `MIN_PANEL_GENES` | 2 | `deconvolution/joint_reannotate.py:98` | panel scorable | none stated |
| `MIN_CLUSTER_CELLS` | 20 | `deconvolution/joint_reannotate.py:99` | cluster kept | registered (`REFERENCE_RELABEL_PREREG.md:217`); equals the build's state floor (`:59`) |
| `MARKER_MIN_DETECT`, `N_MARKERS` | 0.25, 25 | `deconvolution/joint_reannotate.py:100` | review marker table | none stated |
| `SINGLE_SAMPLE_SHARE` | 0.9 | `deconvolution/joint_reannotate.py:101,641-642` | K2 listing | registered (`REFERENCE_RELABEL_PREREG.md:246-247`) |
| Audit: genes, mean floor, margin, τ percentile, minimum labels | 2000, 1.0, 0.05, 10, 5 | `deconvolution/audit_reference_labels.py:36` | label audit | registered (`REFERENCE_RELABEL_PREREG.md:58-88`) |

Marker panels (`deconvolution/reference_marker_panels.yaml`; genes are in the file at the cited lines):

| Panel key | Lines | Top-level classes (sub-classes) |
|---|---|---|
| `_shared` | 17-70 | Endothelial; Lymphatic endothelial; Fibroblasts; Mural (Pericytes, Vascular smooth muscle); Mesothelial; Schwann; Erythroid; Immune (Macrophages, Monocytes, Dendritic, Neutrophils, Mast, T (CD4+, CD8+), NK, B, Plasma) |
| KIDNEY | 72-108 | Proximal tubule; Injured proximal tubule; Thin limb; Thick ascending limb; Distal convoluted tubule; Connecting tubule; Principal; Intercalated (Type A, Type B); Podocytes; Mesangial; + shared Endothelial, Fibroblasts, Mural, Immune |
| LIVER | 110-131 | Hepatocytes; Cholangiocytes; Hepatic stellate; Endothelial (LSEC, Vascular endothelial); Kupffer; + shared Fibroblasts, Mural, Immune |
| CORTEX | 133-172 | Excitatory neurons; Inhibitory neurons; Astrocytes; Oligodendrocytes; OPC; Microglia; BAM; Endothelial; Mural; VLMC; Choroid plexus epithelial; Ependymal; T cells |
| HIPPOC | 174-183 | as CORTEX, with Excitatory neurons split into Dentate gyrus granule cells and Pyramidal neurons |
| WATSC | 185-190 | Adipose stromal progenitor; + shared Endothelial, Lymphatic endothelial, Mural, Mesothelial, Schwann, Erythroid, Immune (no adipocyte panel) |
| MUSCLE | 192-206 | Skeletal muscle; Satellite; FAP; Tenocytes; + shared Endothelial, Lymphatic endothelial, Mural, Schwann, Erythroid, Immune |
| SMLINT | 208-229 | Epithelial (Enterocytes, Goblet, Paneth, Enteroendocrine, Tuft, BEST4+ enterocytes, Stem, Transit-amplifying); + shared Endothelial, Fibroblasts, Mural, Immune |
| LUNG | 231-278 | Epithelial (AT1, AT2, Club, Ciliated, Basal, Pulmonary neuroendocrine); Immune (Alveolar macrophages, Interstitial macrophages, Monocytes, Dendritic, Neutrophils, Mast, T (CD4+, CD8+), NK, B, Plasma); + shared Endothelial, Lymphatic endothelial, Fibroblasts, Mural, Mesothelial, Erythroid |
| BLOOD | 283-315 | Immune (Classical monocytes, Non-classical monocytes, Dendritic, pDC, Neutrophils, Eosinophils, Basophils, T, NK, B, Plasma); Megakaryocytes and platelets; Erythroid |
| SPLEEN | 324-367 | Immune (Macrophages (Red-pulp, Marginal zone), Monocytes, Dendritic, pDC, Neutrophils, Eosinophils, Basophils, T (CD4+, CD8+), NK, B, Plasma); Erythroid; Megakaryocytes and platelets; + shared Endothelial, Fibroblasts, Mural |
| OVARY | 374-404 | Granulosa; Luteal; Theca and stromal (Theca, Stromal); Oocytes; Ovarian surface epithelial; Oviduct epithelial; two guard classes (lung epithelium, hepatocytes); + shared Endothelial, Lymphatic endothelial, Mural, Erythroid, Immune |
| HEART_COARSE | 408-437 | used only for the second-heart check (8d) |

### Statistical procedures
- **Panel score and rule** (route 4): deterministic given the clustering; formula in Procedure steps 5-6. No
  test, no multiplicity control.
- **Label audit metric** (registered `REFERENCE_RELABEL_PREREG.md:58-77`): Pearson r between gene-centred
  log2(CPM + 1) profiles of (sample, label) groups on the 2,000 most variable genes;
  r_self(label) = median r over that label's profiles from different samples; τ = 10th percentile of r_self.
  Units are (sample, label) pseudobulk profiles; no inferential test is attached.
- **K1** (`deconvolution/joint_reannotate.py:578-602,637-639`): the same metric on the final labels; pass if
  the median over labels of r_self exceeds the median r between profiles of different labels from different
  samples; HIPPOC additionally needs a median r_self >= 0.5 (`REFERENCE_RELABEL_PREREG.md:241-245`). Not
  computable for one-library references (returned as `null`, `joint_reannotate.py:638`).
- **K2** (`:640-643`): lists labels present in one sample and clusters with >= 90% of their cells from one
  sample; descriptive. **K3**: cross-tabulation of old and new labels, with a registered expectation for two
  label pairs (`REFERENCE_RELABEL_PREREG.md:248-251`). **K4**: cells dropped and their share (`:252`).
- **Route 3**: `sc.tl.score_genes` with scanpy's defaults and per-cell arg-max; no threshold other than the
  marker gate.
- **Seeds**: route 4 uses seed 0 throughout (`joint_reannotate.py:97`); routes 1 and 3 pass no seed.
- **Pre-registration**: §4-§6 committed 2026-10-27 14:44 (commit `8f7acf3`), §10 committed 21:57:14 (commit
  `4e0a40f`) with the panels and the script; the first clustering outputs on disk are timestamped 21:58-22:00
  (`data/deconvolution/joint_labels/KIDNEY/panel_genes.tsv`, `clusters.tsv`). The SPLEEN and OVARY panels were
  committed 2026-10-31 17:40 (commit `77d495d`) and the libraries were downloaded at 23:44
  (`data/raw/geo/GSE306764_spleen/`, file times). Commit times and file times are the only evidence of
  ordering; there is no external timestamp.

### Checks, validation and their results

K1-K4 on the final labels (values from `data/deconvolution/joint_labels/<REF>/checks.json` and
`settings.json`, which reflect the post-literature-check labels):

| Label set | QC cells | Clusters | Dropped (share) | Labels | K1: median r_self vs different labels | Labels with r_self < 0.5 | K2: clusters >= 90% one sample |
|---|---|---|---|---|---|---|---|
| KIDNEY | 28,626 | 24 | 4,147 (14.5%) | 15 | 0.598 vs -0.104, pass | Cycling; Injured PT; Thin limb; Type A and Type B intercalated | none |
| LIVER | 27,059 | 18 | 0 | 8 | 0.569 vs -0.115, pass | Cholangiocytes; Mesothelial; Vascular endothelial | none |
| CORTEX | 12,969 | 31 | 33 (0.25%) | 12 | 0.781 vs -0.118, pass | BAM; Medium spiny neurons; Neutrophils; VLMC | j6, j20 |
| WATSC | 12,240 | 22 | 256 (2.1%) | 13 | 0.643 vs -0.158, pass | Cycling; Mural; pDC | j0, j5, j6, j21 |
| MUSCLE | 10,780 | 18 | 40 (0.37%) | 9 | 0.445 vs -0.135, pass | Pericytes; Satellite; Schwann; Tenocytes; Vascular smooth muscle | none |
| SMLINT | 18,950 | 27 | 2,757 (14.6%) | 13 | 0.645 vs -0.175, pass | B cells; Macrophages; Stem; Transit-amplifying | j2, j17, j23 |
| LUNG | 46,669 | 38 | 2,511 (5.4%) | 21 | 0.731 vs -0.153, pass | none | j21, j34 |
| HIPPOC | 279,195 | 35 | 255,303 (91.4%) | 11 | 0.884 vs -0.105, pass; >= 0.5 met | Mural; Thalamic neurons; VLMC | none |
| BLOOD_PBMC | 76,590 | 29 | 9,926 (13.0%) | 12 | 0.743 vs -0.149, pass | none | none |
| BLOOD_WB | 14,308 | 22 | 0 | 9 | not computable (one library) | - | - |
| SPLEEN | 13,929 | 27 | 916 (6.6%) | 14 | not computable (one library) | - | - |
| OVARY | 9,104 | 28 | 0 | 18 | not computable (one library) | - | - |

- K2: no label is confined to one sample in any multi-sample set (`K2_labels_in_one_sample` is empty in every
  `checks.json`); the single-sample clusters are explained as dissection or biology
  (`JOINT_REANNOTATION_RESULTS.md:115-122`).
- K3: both registered pairs landed mostly in one joint label (KIDNEY "Renal interstitial fibroblasts" 105 of
  150 and "Renal vesicle cells" 179 of 269 in "Injured proximal tubule cells"; WATSC "Pre-B" 68 of 68 and
  "Naive B" 41 of 58 in "B cells") (`JOINT_REANNOTATION_RESULTS.md:126-130`).
- Manual decisions for the eight corpus references: 100 (70 review calls, 18 overrides of an accepted label,
  12 under-calls made specific) (`REFERENCE_RELABEL_PREREG.md:271-272`; the per-reference row counts above
  sum to 100).
- Literature check (`JOINT_REANNOTATION_LITCHECK.md:33-42`): eight tissue references, 100 rows: 53 SUPPORTED,
  19 PLAUSIBLE, 3 QUESTIONABLE, 25 SOUND (DROP); blood libraries, 35 rows: 19 / 5 / 0 / 11; 21 flagged
  auto-accepted clusters: 1 QUESTIONABLE (a bone-marrow cluster not in any reference), 20 PLAUSIBLE.
- Rebuild: the eight references' cell counts equal the kept cells of the joint labels
  (`JOINT_REANNOTATION_RESULTS.md:196-208`, pre-literature-check numbers); the build refuses any mismatch
  between loaded and labelled cells (`deconvolution/build_reference.py:332-336`).
- Unit tests: 12 tests in `tests/test_joint_reannotate.py` (`:57-156`) cover scaling, the accept/margin/doublet
  rule, panel resolution, review application, the split and the joint-label build path. They were not run for
  this audit.
- Deposited-label coverage (tabulated from the `*_consensus.tsv` files of the selected samples): HEART
  135,735 of 166,376 QC barcodes labelled; BAT 28,247 of 65,441.

### Deviations, known defects and caveats
1. **HYPOTH was never re-annotated.** It is the only default reference still labelled by route 1, the
   procedure the project itself documents as producing wrong names (`REFERENCE_RELABEL_PREREG.md:174-201`).
   Its consensus file shows the pattern: a generic "Neurons" type of four clusters (1,792 cells, including a
   cluster PanglaoDB called "Adrenergic neurons" and ScType "Basophils") beside "Glutamatergic neurons" and
   "GABAergic neurons"; "Radial glial cells" from PanglaoDB "Meningeal cells"; "Neuroendocrine cells" invented
   by the LLM (`consensus_source = corrected`) from "Erythroblasts" / "Basophils"
   (`data/training/cell_annotations_consensus/GSE248413_sample0/GSE248413_sample0_consensus.tsv`, rows for
   clusters 2, 4, 10, 14, 15, 18). The same "Radial glial cells" name in CORTEX turned out to be vascular
   cells (`REFERENCE_LABEL_AUDIT.md:100`). The exclusion is stated (`REFERENCE_RELABEL_PREREG.md:205-208`:
   "one sample, ScType had brain markers, 1% corrected ... can go through the same code later"), and
   `label_scheme` is `none` (`tissue_references.yaml:324`), so the collinear neuron labels are not merged.
2. **The manual review is an AI assistant's.** The 160 decisions were made by the assistant from marker
   tables (`REFERENCE_RELABEL_PREREG.md:234-235`); file times show, for example, KIDNEY clusters written at
   22:00:31 and final labels at 22:08:40 on 2026-10-27. The docs state that the decisions "should be read by
   someone who knows the tissue" and that this has not happened (`JOINT_REANNOTATION_RESULTS.md:173-176,211-212`).
3. **The literature check does not supply verified citations.** It was carried out by four agents; "Most were
   read at abstract level, and none has been checked against a local PDF ... Use the citations as pointers,
   not as verified references" (`JOINT_REANNOTATION_LITCHECK.md:26-29`).
4. **The registered rule alone accepts wrong labels.** Eighteen automatically accepted labels were overridden
   after reading their markers (mammary epithelium accepted as "Mesothelial cells", doublets accepted as single
   types; `REFERENCE_RELABEL_PREREG.md:148-159`; `JOINT_REANNOTATION_RESULTS.md:38-50`). Causes given: the doublet
   flag needs both panels >= 0.5, and min-max scaling lets one extreme small cluster set a gene's scale.
5. **Panel genes missing from the gene space.** Cd4, Il7r, Pecam1, Rgs5, Hbb, Ccr2, Cxcr2 and (brain) Cldn5 are
   absent (`JOINT_REANNOTATION_RESULTS.md:170-172`; `data/deconvolution/joint_labels/KIDNEY/panel_genes.tsv`),
   so the CD4/CD8 split rests on surrogate genes (`JOINT_REANNOTATION_LITCHECK.md:94-95`).
6. **Cluster IDs are not guaranteed reproducible.** The guard hashes were recorded on 2026-11-01 (commit
   `d38b83a`), after the reviews of 10-27/28 and 10-31; they fix the on-disk clustering for the future but do
   not show that the decisions were written against it. Exact reproduction relies on the deposited
   `cells_final.tsv.gz` files (`deconvolution/joint_reannotate.py:17-25`; `DECONV_REBUILD_RUNBOOK.md:68-70`).
   The clean rebuild of 2026-11-02 did not cover re-annotation or reference builds
   (`REFS4ALL_REBUILD_PREREG.md:16`; `REFS4ALL_REBUILD_RESULTS.md:111-112`).
7. **Route 1 clustering selects HVGs on log-normalised data.** `flavor="seurat_v3"` is called with `layer=None`
   after `normalize_total` and `log1p` (`analysis/annotate_celltypes.py:326-332`; `analysis/annotate_sctype.py:408-413`),
   whereas route 4 passes the raw-count layer (`deconvolution/joint_reannotate.py:314`). This affects HYPOTH's
   clusters, which are also its BayesPrism cell states.
8. **Route 1 is not reproducible as code.** The LLM call sets no temperature or seed; a failed batch silently
   falls back to the PanglaoDB label; the scripts hard-code `/depot/reese18/apps/motrpac-genecompass`
   (`analysis/annotate_celltypes.py:72`; `analysis/annotate_sctype.py:57`; `analysis/consensus_annotations.py:54`).
   The cached reply map is `data/training/cell_annotations_consensus/consensus_label_map.json`
   (`analysis/consensus_annotations.py:62`).
9. **The ScType step is a re-implementation.** The docstring attributes the weights (+2, -1, / sqrt(n)) to the
   ScType paper (`analysis/annotate_sctype.py:231-236`); agreement with the published algorithm was not
   verified here.
10. **Route 2 rewrites corpus annotation files in place.** `adopt_authors_labels.py:105-112` (and
    `merge_consensus_labels.py:68-69`) overwrite `data/training/cell_annotations*/<sample>/` files that the
    pre-registration says are not rewritten (`REFERENCE_RELABEL_PREREG.md:96-98`). The BAT files on disk were
    written by the first committed version of the script (commit `8dea8f4`): their `consensus_source` is
    `authors_deposited_GSE244451_SCPAnnotated`, a string the current code cannot produce (`:111`). The labels
    are consistent with the current map, but the BAT adoption has not been re-run with the committed code.
11. **HEART weak-overlap branch.** When the best barcode overlap is under 0.30 the code prints "labelling all
    Unknown" but still takes the best match's labels (`adopt_authors_labels.py:166-169`). Not triggered
    according to the comment at `:46`.
12. **A registered merge was not applied.** For HEART (authors' labels) the audit rule proposed MERGE of
    "CD8+ T cells" (1,624 cells, one of 19 samples) into "T cells" (`REFERENCE_LABEL_AUDIT.md:150,181-183`;
    scope `REFERENCE_RELABEL_PREREG.md:50,88`). The reference still has both types
    (`data/deconvolution/references_v3/HEART_GSE280111_LV/summary.txt`), and no change-log entry withdraws the
    merge.
13. **Low-quality barcodes pass the corpus QC.** The 200-gene floor let 255,069 HIPPOC barcodes with a median
    of 243-446 genes into the corpus (`JOINT_REANNOTATION_RESULTS.md:146-149`); KIDNEY and SMLINT lose 3,431 and
    2,740 low-quality barcodes for the same reason (`:72-77`). References not re-annotated (HYPOTH; BAT and HEART
    only through the authors' own QC) have had no such screen.
14. **TESTES.** The panels are built from the authors' differential-expression table for the same data set;
    the ingest uses a mitochondrial cut of 20% and a 3-cell gene filter, not the corpus values
    (`deconvolution/ingest_omix767_testes.py:19,43-45`); one of the two arms is treated (8a, item 3).
15. **Count inconsistencies in the docs.** `REFERENCE_RELABEL_PREREG.md:626` gives "35 decisions" for SPLEEN,
    OVARY and the second heart set; the review file has 25 rows for them (11 + 10 + 4).
    `JOINT_REANNOTATION_RESULTS.md:57-66,199-208` predates the literature check: LUNG is 44,158 cells with
    2,511 dropped (not 44,602 / 2,067) and HIPPOC has 11 labels (not 10) in the current files.
16. A symbol that resolves to more than one gene ID contributes one column per ID to a panel's mean
    (`deconvolution/joint_reannotate.py:149-153,366-370`), so such a marker is weighted more than once.
17. The manuscript says "Labels for lung, cortex and hippocampus were then checked against the literature"
    (`manuscript/motrpac_genecompass.tex:1312-1313`); all 100 decisions of the eight references were checked
    and three labels in those three references changed.

### Methods references
- ScType: "Ianevski et al., Nat Commun 2022" (as given in `analysis/annotate_sctype.py:5`).
- PanglaoDB: only the download URL is given (`analysis/annotate_celltypes.py:77`); Franzén O, Gan L-M,
  Björkegren JLM, Database 2019 [verified against Crossref; see REFERENCES.md].
- Scanpy: Wolf FA, Angerer P, Theis FJ, Genome Biol 2018 [verified against Crossref; see REFERENCES.md]. `seurat_v3` HVG method: Stuart T, et al.,
  Cell 2019 [verified against Crossref; see REFERENCES.md]. Harmony: Korsunsky I, et al., Nat Methods 2019 [verified against Crossref; see REFERENCES.md]. Leiden: Traag VA, Waltman L,
  van Eck NJ, Sci Rep 2019 [verified against Crossref; see REFERENCES.md]. None of these is in the manuscript bibliography
  (`manuscript/motrpac_genecompass.tex:1451-1553`).
- Injured proximal tubule panel: "Kirita 2020" (as given in `deconvolution/reference_marker_panels.yaml:76`).
- Citations used to change labels: "Nagalski 2012; Lipiec 2020" (thalamic markers) and "Torres-Nagel 1992"
  (rat NK cells and CD8α) (as given in `JOINT_REANNOTATION_LITCHECK.md:50,94`); "Lun 2019 EmptyDrops; Fleming
  2023 CellBender; Caglayan 2022" (`:99`). The file itself warns these are unverified (`:26-29`).
- TESTES panels: Guan et al., Sci Data 2022, PMID 35338159 (as given in `deconvolution/annotate_omix767_testes.py:8`).

---

## Stage 8c - Reference QC gate

### Research role
Question: is every sample in a reference a native-tissue, single-cell or single-nucleus library, rather
than a spatial spot array, an engineered or sorted preparation, bulk RNA, or developmental tissue? The gate
exists because two earlier references violated this (Visium spots in the liver reference; an in-vitro
engineered lung reference; `deconvolution/REFERENCE_QC.md:9-22`). The claim that rests on it is the
manuscript's statement that each reference is "admitted only through an automated QC gate blocking
spatial, engineered, developmental, or mixed data" (`manuscript/motrpac_genecompass.tex:1295`). Verdict:
the gate is a keyword screen on GEO title text. It catches the two historical defects and passes the
twelve corpus-built references. It evaluates nothing for the three references built from external
libraries, and it warns but does not block on developmental or mixed-modality samples.

### Inputs and outputs
- A built reference's `cells_meta.tsv` (`sample` column) and, with `--deep`, `reference_counts.mtx`
  (`deconvolution/reference_qc.py:39-60`).
- `reports/annotations/annotation_inventory.tsv`: `geo_title`, `geo_source_name`, `geo_cell_type` per sample
  (`reference_qc.py:28-37`).
- Output: a PASS / WARN / FAIL line per reference on standard output; exit code 2 with `--fail` on any FAIL
  (`reference_qc.py:99-108`). Nothing is written to disk.

### Procedure
The same patterns are applied at three points.
1. **At build time**, inside sample selection: samples matching SPATIAL, ENGINEERED or BULK are dropped (or
   the build stops if every sample matches); DEVEL matches print a warning (`deconvolution/build_reference.py:197-222`).
2. **Standalone**, `reference_qc.py --ref-dir <dir> [--deep] [--fail]` (`reference_qc.py:56-78`): for each
   sample of the built reference, in order: not in the inventory -> WARN "provenance unknown" and no further
   test (`:63-65`); SPATIAL -> FAIL; else ENGINEERED -> FAIL; else BULK -> FAIL (`:66-68`); DEVEL -> WARN (`:69`).
   With `--deep` and at least two samples, the median number of detected genes per cell is computed per
   sample and a WARN is raised if the largest is more than 2.5 times the smallest (`:70-77`).
3. **Before deconvolution**, `pipeline/run_deconv_all.py` calls `reference_qc.py --ref-dir <dir> --fail` for
   each tissue and refuses the tissue on a non-zero exit (`pipeline/run_deconv_all.py:125-130`). It does not
   pass `--deep`. The call is skipped with `--skip-qc`, for a tissue with `allow_nonnative`, or when the
   inventory file is absent (`:66,80-81,126`).

### Models and software
Python standard library (`re`, `csv`); scipy and numpy only for `--deep` (`reference_qc.py:41-43`).

### Parameters, thresholds and flags

| Name | Value | Where set | What it controls | Rationale given |
|---|---|---|---|---|
| `SPATIAL` (FAIL) | `(visium\|_vis\b\|\bvis\b\|spatial\|slide-?seq\|geomx\|cosmx)`, case-insensitive | `deconvolution/reference_qc.py:20` | spatial libraries | multi-cell spots are not single cells (`REFERENCE_QC.md:28`) |
| `ENGINEERED` (FAIL) | `(engineered\|cell isolate\|tri-?culture\|co-?culture\|organoid\|cultured\|\bin[- ]?vitro\|ipsc\|cell line\|passage\|\bday ?\d\|\bd\d+\b\|sorted\|cd\d+\+\|facs\|flow-?sorted\|magnetic)` | `reference_qc.py:21-23` | engineered, cultured, sorted | in-vitro or sorted expression differs from in vivo (`REFERENCE_QC.md:29`) |
| `BULK` (FAIL) | `(bulk rna\|whole[- ]tissue rna(?!-seq single))` | `reference_qc.py:24` | bulk libraries | `REFERENCE_QC.md:28` |
| `DEVEL` (WARN) | `(\bE\d+\.?\d*\b\|\bP\d+\b\|embryo\|embryonic\|fetal\|foetal\|neonat\|postnatal)` | `reference_qc.py:26` | developmental stage | "usable in principle, wrong for ADULT deconvolution" (`reference_qc.py:25`) |
| Depth ratio (WARN, `--deep` only) | max/min of per-sample median genes per cell > 2.5; outliers listed above 2 x min | `reference_qc.py:74-75` | hidden modality mix | caught the liver Visium spots (`reference_qc.py:9-10`) |
| `ALLOW_NONNATIVE_REF` | unset in every current build | `build_reference.py:214`; `build_references_from_config.py:140-141` | bypass of the build-time drop | for an intrinsic sort (`tissue_references.yaml:32`) |
| `--inventory` | `reports/annotations/annotation_inventory.tsv` (relative path) | `reference_qc.py:84` | title source | - |

### Statistical procedures
None. Regular-expression matching and one ratio of medians.

### Checks, validation and their results
- **Run for this audit** (metadata only, no `--deep`), on the 15 default reference directories: PASS for
  KIDNEY, LIVER, CORTEX, HEART, WATSC, MUSCLE, BAT, HYPOTH, SMLINT, LUNG, HIPPOC, TESTES (12); WARN for
  BLOOD_composite (6 libraries), SPLEEN_GSE306764 and OVARY_GSE343214 (1 each), all "not in inventory
  (provenance unknown)"; no FAIL.
- Build-time drop: LIVER's two Visium samples (`GSE220075_sample11`, `_sample2`; titles "Rat_A1_VIS",
  "Rat_B1_VIS") are removed, exactly as `expect.qc_dropped` lists (`tissue_references.yaml:86`; validator
  output "reference_qc dropped (intended)").
- Historical validation: the gate FAILs the retired engineered-lung reference and the old Visium liver "(both
  title + depth)" and passes the corrected references (`REFERENCE_QC.md:45-46`).
- SPLEEN and OVARY: registered check R1 "`reference_qc.py` passes (no `allow_nonnative` bypass)"
  (`REFERENCE_RELABEL_PREREG.md:570`); recorded result "R1 PASS for both (reference_qc --deep --fail: WARN only,
  sample not in the corpus inventory)" (`:648`).
- The `--deep` check was not run for this audit (it loads the count matrices). No stored output of a
  `--deep` run on the current references was found.

### Deviations, known defects and caveats
1. **The gate is vacuous for BLOOD, SPLEEN and OVARY.** Their libraries are not in the corpus inventory, so
   no pattern is evaluated (`reference_qc.py:63-65`). SPLEEN and OVARY have one library each, so the depth
   check, which needs two samples (`:72`), is inactive as well. The registered R1 "PASS" therefore reflects
   only that nothing could be tested.
2. **Developmental and mixed-modality samples are not blocked.** DEVEL is a warning in both code paths
   (`reference_qc.py:69`; `build_reference.py:210-213`); the depth check is a warning and is not part of the
   pre-deconvolution gate (`pipeline/run_deconv_all.py:127-128`). `REFERENCE_QC.md:30` says so ("-> WARN"); the
   manuscript sentence at `manuscript/motrpac_genecompass.tex:1295` says these are blocked. A neonatal lung
   sample did pass the gate in July because its title carried no age token and was removed by hand
   (`TISSUE_REFERENCE_AND_PROPORTIONS.md:1016-1018`; `build_lung_pooled.py:11-13`).
3. **What the gate cannot see** (`TISSUE_REFERENCE_AND_PROPORTIONS.md:1007-1026`, written 2026-07-14): what the
   source study was (control arm of a disease model), donor count, age unless in the title, missing cell
   types, arm pooling, assay-to-bulk compatibility. Organism and duplicate GSMs, also listed there, are now
   handled by sample selection (8a). The same document's summary still applies: "a regression test for the
   two bugs we already found, not a forward-looking QC" (`:1024-1026`).
4. **Ambient RNA is not screened.** The HIPPOC diagnosis recommends that "the ambient fraction of a nuclei
   reference should be part of reference QC" and notes that LIVER, HEART and KIDNEY "have not been checked
   this way" (`HIPPOC_REFS4_DIAGNOSIS.md:108-111`). No such check exists in `reference_qc.py`.
5. `reference_qc.py --all` scans only `data/deconvolution/references` and `references_v2` (`:92-93`), the
   pre-July trees; it does not cover `references_v3` or `references_v4`.
6. Without the inventory (a clone that starts from deposited references) the gate is skipped entirely
   (`pipeline/run_deconv_all.py:80-81,126`).

### Methods references
None (project-specific rule set).

---

## Stage 8d - Special builds

Seven items: (1) the BLOOD whole-blood composite and the CDSeq evidence behind it; (2) the pooled LUNG
reference; (3) TESTES from OMIX767 with lifted gene IDs; (4) the SPLEEN and OVARY additions; (5) the HIPPOC
diagnosis and the two failed ambient corrections; (6) the WATSC caveats; (7) the second-heart cross-study
check.

### Research role
- **BLOOD.** The MoTrPAC blood samples are PAXgene whole blood (`REFERENCE_LABEL_AUDIT.md:191-193`), and the
  July reference was PBMC from one rat, with no erythroid or granulocyte profile. BayesPrism has no
  "unknown" component, so the missing mass was assigned to the types present. The composite supplies the
  two missing compartments. Claim: the composite can recover erythroid, neutrophil and PBMC-type fractions.
  Check: six registered checks, the last a simulated-mixture recovery with a floor of r 0.9. Verdict: all
  checks pass; absolute fractions are biased and BLOOD is to be read differentially
  (`tissue_references.yaml:111-113`).
- **LUNG.** No single clean adult rat lung atlas is in the corpus, so control arms of two studies are pooled.
- **TESTES.** The only rat testis scRNA; its gene IDs are from an older annotation than the lifted bulk.
- **SPLEEN, OVARY.** Added so that two more MoTrPAC tissues can be deconvolved; each is one external library
  admitted under registered checks.
- **HIPPOC.** The reference yields an implausible composition; the question was why, and whether ambient-RNA
  correction repairs it. Verdict: cause identified (ambient RNA); two registered corrections failed; HIPPOC
  composition "stays flagged unreliable" (`REFERENCE_RELABEL_PREREG.md:517-518`).
- **WATSC.** Documented limits of the only available white-adipose reference.
- **Second heart reference.** A cross-study test of the HEART reference, which otherwise has only a
  within-study hold-out validation (`REFERENCE_RELABEL_PREREG.md:585`).

### Inputs and outputs

| Item | Inputs | Outputs |
|---|---|---|
| BLOOD | `data/raw/geo/blood_option_b/GSE335156/GSM98068{33..37}_V{1..5}_*` and `.../GSE306764/{blood/Blood-Ctrl, bone/Bone-Ctrl}` (`joint_reannotate.py:69-74`; md5 in `external_refs_md5.tsv:8-30`); joint labels `joint_labels/{BLOOD_PBMC,BLOOD_WB,BLOOD_BM}/` | `data/deconvolution/references_v4/BLOOD_composite/` (67,147 cells, 18,260 genes, 13 types, 21 states; `summary.txt`), incl. `cells_source.tsv`; `data/deconvolution/blood_option_b/checks.json`; `data/deconvolution/validation/BLOOD_composite_check6/` |
| CDSeq | `data/deconvolution/motrpac_bulk/BLOOD/`; the gene list of the July BLOOD run | `data/deconvolution/results/_deployed_backup_20260715/BLOOD/cdseq_rr/` (`cdseq_proportions.csv`, `cdseq_gep.csv`, `cdseq_result.rds`, `RUN_PARAMS.txt`); the `cdseq/` folder of the full run is empty |
| LUNG | corpus samples `GSE273062_sample0`, `_sample2`, `GSE252844_sample0`; `joint_labels/LUNG/cells_final.tsv.gz` | `references_v4/LUNG_native_pooled/` (44,158 cells, 16,956 genes, 21 types, 29 states) |
| TESTES | `data/raw/ngdc/datasets/OMIX767/Gene_bc_matrices/{C,E7W}_gene_bc_matrices/`, `OMIX767-20-02.xlsx` (`external_refs_md5.tsv:41-57`); `deconvolution/reference/motrpac_bulk_liftover.tsv` | `data/training/preprocessed/qc_matrices/OMIX767_sample{C,E7W}.h5ad`; annotation files; `references_v3/TESTES_OMIX767_lifted/` (5,836 cells, 14,482 genes, 6 types, 12 states). The raw-ID July build remains at `references_v3/TESTES_OMIX767/` |
| SPLEEN | `data/raw/geo/GSE306764_spleen/extracted/Spl-Ctrl/` (`external_refs_md5.tsv:37-40`) | `references_v4/SPLEEN_GSE306764/` (13,013 cells, 22,050 genes, 14 types, 23 states); `s14_checks/SPLEEN.json`; `validation/SPLEEN_R2/` |
| OVARY | `data/raw/geo/GSE343214/GSM9947804_Normal_*` (`external_refs_md5.tsv:34-36`) | `references_v4/OVARY_GSE343214/` (9,104 cells, 30,562 genes, 18 types, 28 states); `s14_checks/OVARY.json`; `validation/OVARY_R2/` |
| HIPPOC | `references_v4/HIPPOC_GSE295314` (refs4) and `references_v3/HIPPOC_GSE295314` (July build, source of the "soup") | corrected references and the `_refs4h` / `_refs4h2` output trees were deleted on 2026-10-31; record tables kept in `deconvolution/diagnostics/hippoc_refs4/refs4h_record/` (`REFERENCE_RELABEL_PREREG.md:534-540`) |
| Second heart | `data/raw/geo/GSE307551/GSM9226643_F3441_*.h5`, `GSM9226644_F3442_*.h5` (`external_refs_md5.tsv:32-33`) | `joint_labels/HEART_GSE307551/`; `validation/HEART_GSE307551_cross/s15_result.json` |

(All output paths are under `data/deconvolution/` unless a full path is given.)

### Procedure

**(1a) CDSeq on the real BLOOD bulk (July; the evidence that a compartment was missing).**
CDSeq is a reference-free method that estimates proportions and profiles from the bulk alone
(`deconvolution/R/run_cdseq_realbulk.R:11-13`). Input: the 50 BLOOD bulk samples, restricted to the genes
BayesPrism used in the then-current run (`data/deconvolution/results/motrpac/BLOOD/pred_z/genes.txt`;
11,319 genes; `run_cdseq_rr.R:51-56`), counts rounded to integers (`:57`).
- Full-fidelity run (`run_cdseq_realbulk.R:48-55`): `cell_type_number = 14`, `mcmc_iterations = 700`,
  `dilution_factor = 1`, `gene_subset_size = NULL`, `block_number = 1`, 16 cores, `set.seed(1)` (`:21`), 12 h
  limit (`slurm/analysis/cdseq_blood.slurm:10`, local file). It produced no output: its folder is empty.
- Reduce-Recover run (`run_cdseq_rr.R:66-73`; `slurm/analysis/cdseq_blood_rr.slurm:33-36`): the same K and
  iterations with `dilution_factor = 25`, `block_number = 16`, `gene_subset_size = 1500`, 16 cores,
  `set.seed(1)` (`:30`). All 11,319 genes were recovered
  (`data/deconvolution/results/_deployed_backup_20260715/BLOOD/cdseq_rr/RUN_PARAMS.txt`).
- Why Reduce-Recover: with one block CDSeq does not parallelise and with dilution 1 the sampler assigns every
  read; the full run was nine hours into a twelve-hour limit, writes nothing until it returns, and its time
  limit could not be raised (`run_cdseq_rr.R:4-11`; `cdseq_blood_rr.slurm:14-22`). The script states that the
  Reduce-Recover result is approximate (read dilution widens the posterior; gene coverage per block is
  random) and that "the exact percentages are not" robust (`run_cdseq_rr.R:13-27`).
- Reported result (`TISSUE_REFERENCE_AND_PROPORTIONS.md:1041-1051`): the 14 components are mutually distinct;
  about 80% of the bulk mRNA mass sits in components that match nothing in the PBMC reference (r about
  0.06-0.10), named by their markers as erythroid (two components, about 21% combined) and granulocyte; the
  platelet/megakaryocyte component matches the reference at r = 0.41.

**(1b) The BLOOD composite** (registered `REFERENCE_RELABEL_PREREG.md:275-319`, amended `:320-327`).
1. *Sources.* PBMC types from the five **venous** libraries of GSE335156 (V1-V5); the arterial libraries are
   not used because MoTrPAC bled rats by cardiac puncture, taken to be venous blood, "an inference; the paper
   does not name the ventricle" (`:284-287`). Neutrophils and erythroid cells (and eosinophils/basophils "if
   present") only from GSE306764 Blood-Ctrl, a whole-blood library after red-cell lysis from Lewis rats given
   adjuvant (`:277-279`; `tissue_references.yaml:108-110`). The bone-marrow library is labelled for check 1
   only.
2. *Labels.* Each set is put through route 4 of 8b with the BLOOD panel
   (`reference_marker_panels.yaml:283-315`): BLOOD_PBMC with Harmony over the five libraries, BLOOD_WB and
   BLOOD_BM as single libraries. GSE306764's gene IDs are lifted with the bulk's map; GSE335156 is already in
   the lifted space (`joint_reannotate.py:165-198`).
3. *Checks 1-5* (`deconvolution/build_blood_composite.py:77-140`), then *check 6* (`:176-239`); see below.
4. *Build* (`:143-163`): PBMC cells whose final label is not Neutrophils, Erythroid cells, Eosinophils,
   Basophils or DROP; whole-blood cells whose final label is one of those four named types (only Neutrophils
   and Erythroid cells exist); genes present in both (inner join); cell state = `PBMC_<cluster>` or
   `WB_<cluster>`; states under 20 cells removed (`build_reference.clean_cells`, 20); written with
   `export_reference` plus `cells_source.tsv`.
5. *Result* (`data/deconvolution/references_v4/BLOOD_composite/summary.txt`): CD4+ T 19,863; B 18,151; CD8+ T
   8,453; non-classical monocytes 6,448; regulatory T 3,211; NK 3,102; classical monocytes 3,050;
   **neutrophils 2,168**; T cells 1,295; cycling 693; megakaryocytes and platelets 533; pDC 90; **erythroid
   cells 90**. The PBMC libraries' own erythroid cluster (1,775 cells, 80% from one rat) is left out by rule
   (`REFERENCE_RELABEL_PREREG.md:324-326`).
6. *Globins.* Stage 8 removes hemoglobin genes for every tissue, so the no-globin build is primary and a
   with-globin arm was run for comparison (`:320-323`).

**(2) LUNG pooled** (`deconvolution/build_lung_pooled.py:31-58`). Each study is loaded with an outer gene join
(`:37`), the two are concatenated with an outer join and zero fill (`:42`), genes detected in fewer than 10
pooled cells are removed (`:45`), joint labels are applied (`:48-49`), states under 20 cells are removed
(`:52`). GSE242310 was dropped because both arms are neonatal; GSE247625 was deferred because it is aligned
to the mouse genome (`:11-14`).

**(3) TESTES** (`tissue_references.yaml:443-464`).
1. *Ingest* (`deconvolution/ingest_omix767_testes.py`): the 10x matrices of arms C and E7W (`:18`) are read;
   cells with at least 200 detected genes and at most 20% mitochondrial counts are kept, genes in fewer than
   3 cells removed (`:19,43-45`); written in corpus format. Kept: C 2,155 and E7W 3,693 cells
   (`tissue_references.yaml:445`).
2. *Annotate*: route 3 of 8b.
3. *Build* with `--sample-ids OMIX767_sampleC,OMIX767_sampleE7W --min-state-cells 20 --gene-map
   deconvolution/reference/motrpac_bulk_liftover.tsv`. `lift_genes()` renames genes through the bulk's
   liftover map, drops genes the map leaves unlifted or does not list, and sums genes that land on the same
   lifted ID, on the concatenated matrix (`deconvolution/build_reference.py:264-290,314-315`).
4. *Result* (`references_v3/TESTES_OMIX767_lifted/summary.txt:1-2`): 16,826 genes in, 14,668 lifted, 14,482
   after summing 186 into shared IDs, 2,158 unmapped; 5,836 cells; round spermatids 2,107, condensed
   spermatids 1,583, spermatocytes 980, elongating spermatids 960, spermatogonia 164, Sertoli cells 42.
   Leydig, immune and endothelial cells fall under 20 cells per state and are dropped
   (`tissue_references.yaml:458-459`).
5. *Why the lift.* OMIX767 ships Rnor_6.0 IDs while the bulk is lifted; the July raw-ID build left 587 bulk
   genes (2.3% of bulk counts) unpaired. After the lift: deconvolution genes 13,124 -> 13,711, bulk counts
   covered 91.5% -> 93.8%, θ shifts <= 1.1 percentage points (`tissue_references.yaml:461-464`).

**(4) SPLEEN and OVARY** (registered `REFERENCE_RELABEL_PREREG.md:542-581`; details fixed `:601-614`).
Each library is fetched (`fetch_external_refs.sh:63-71`), QC-filtered and put in the bulk's gene space
(SPLEEN through the liftover map; OVARY's IDs are already in the lifted space; `joint_reannotate.py:83-86,244-269`),
clustered and labelled by route 4 with a panel committed before download (one library, so no Harmony),
then checked (`deconvolution/build_external_reference.py:71-101`) and built (`:104-115`): cells labelled DROP
or with an OVARY guard class are excluded, cell state = joint cluster, states under 20 cells removed.

**(5) HIPPOC diagnosis and correction attempts.**
- *Diagnosis* (`HIPPOC_REFS4_DIAGNOSIS.md`; scripts `deconvolution/diagnostics/hippoc_refs4/1-5_*.py`). With
  the re-annotated reference BayesPrism gives mean fractions endothelial 0.498, inhibitory neurons 0.358,
  astrocytes 0.085, dentate granule cells 0.028, excitatory neurons 0.003 (`:3-6`). Evidence: the endothelial
  cluster (341 nuclei, median 967 UMI) carries other types' markers (`:27-33`); the "soup" (the summed
  profile of the 254,663 barcodes kept in July and dropped by re-annotation) correlates with the mean bulk at
  0.87, above every reference type (0.66-0.82) (`:46-48`); an estimated ambient fraction per type ranks the
  types as BayesPrism's fractions do (endothelial 0.55 ... excitatory neurons 0.07) (`:50-52`); a
  non-negative-least-squares fit on square-root-scaled profiles, used as a stand-in for BayesPrism and
  checked against it on CORTEX (0.608 vs 0.592 excitatory, `:21-23`), gives neurons about 0.66 after
  profile-level ambient subtraction with vascular types merged (`:53-61`).
- *Attempt 1, §13* (registered `REFERENCE_RELABEL_PREREG.md:421-473`; `deconvolution/R/soupx_correct_reference.R`,
  default mode). Per sample: soup profile = summed counts of that sample's barcodes present in the July build
  and absent from the current one (`:36-37,44-47`; no unfiltered matrices exist for GSE295314);
  `SoupChannel(toc, toc, calcSoupProfile = FALSE)` with `setSoupProfile` (`:45-47`); clusters = the current
  cell types (`:48-49`); `autoEstCont(doPlot = FALSE, forceAccept = TRUE)` (`:50`), a failed sample taking the
  median of the others (`:60-63`); `adjustCounts(roundToInt = TRUE)` (`:95`). After correction Endothelial +
  Mural + VLMC are renamed "Vascular cells" and the four neuron types "Neurons"; cell states are unchanged
  (`:103-109`; `REFERENCE_RELABEL_PREREG.md:444-448`).
- *Attempt 2, §13b* (registered after attempt 1 failed, `REFERENCE_RELABEL_PREREG.md:492-510`; same script with
  `SOUPX_MODE=percell`, `:68-90`). For each type, up to 100 of the 2,000 most soup-abundant genes that are
  specific to another type (another type's CPM >= 20 x (own CPM + 1), or another type > 200 CPM with own < 5%
  of it; `:74-78`). Per nucleus, ambient UMI = counts on those genes / their summed soup fraction (`:81-83`);
  per sample A = the median over its nuclei; per-nucleus contamination = min(0.9, A / UMI) (`:85-86`);
  `setContaminationFraction` and `adjustCounts(roundToInt = TRUE)` (`:93-95`).
- Both corrected references and their outputs were deleted on 2026-10-31 (`REFERENCE_RELABEL_PREREG.md:534-540`).
  The default uses the uncorrected `references_v4/HIPPOC_GSE295314`.

**(6) WATSC.** No special build step; the young-arm filter and joint labels are in 8a and 8b. The caveats are
listed under Deviations.

**(7) Second heart reference** (registered `REFERENCE_RELABEL_PREREG.md:583-599`). The two F344 libraries of
GSE307551 are clustered jointly and labelled with a coarse panel (`reference_marker_panels.yaml:408-437`);
their features are gene symbols, mapped to the HEART bulk's IDs through the panel symbol resolver, keeping a
symbol only if it resolves to exactly one bulk ID (`joint_reannotate.py:224-241`; a forced deviation,
`REFERENCE_RELABEL_PREREG.md:620-624`). Twenty mixtures are drawn from these cells and deconvolved with the
production HEART reference; estimates are summed to coarse classes by a committed map
(`reference_marker_panels.yaml:442-458`; `build_external_reference.py:203-248`).

### Models and software
- CDSeq 1.0.9 (`deconvolution/setup/r_packages.lock.tsv:49`; "not pinned" in the manifest,
  `deconvolution/setup/r_packages.yaml:72`); R 4.4.1.
- SoupX 1.6.2 (`deconvolution/setup/r_packages.yaml:27`; `deconvolution/diagnostics/hippoc_refs4/refs4h_record/refs4h2_provenance_soupx.json`,
  field `soupx_version`); not in `r_packages.lock.tsv`, which was captured 2026-10-05.
- Route-4 software as in 8b. BayesPrism 2.2.3 for the mixture checks (Part 5).

### Parameters, thresholds and flags

| Name | Value | Where set | What it controls | Rationale given |
|---|---|---|---|---|
| CDSeq K | 14 | `deconvolution/R/run_cdseq_rr.R:33`; `run_cdseq_realbulk.R:24` | number of components | none stated (the PBMC reference had 14 labels, `deconvolution/reference/canonical_references.tsv:2`) |
| CDSeq iterations | 700 | `run_cdseq_rr.R:34` | Gibbs iterations | none stated |
| CDSeq dilution / blocks / gene subset | 25 / 16 / 1500 (Reduce-Recover); 1 / 1 / NULL (full) | `run_cdseq_rr.R:35-37,70-72`; `run_cdseq_realbulk.R:52-54` | approximation | runtime (`run_cdseq_rr.R:4-11`) |
| CDSeq seed | `set.seed(1)` | `run_cdseq_rr.R:30` | R RNG | - |
| Whole-blood-only types | Neutrophils, Erythroid cells, Eosinophils, Basophils | `deconvolution/build_blood_composite.py:37` | composite sources | registered (`REFERENCE_RELABEL_PREREG.md:277-279`) |
| Check 1 floor | >= 100 neutrophils in Blood-Ctrl | `build_blood_composite.py:87` | neutrophil source | registered (`:295-297`) |
| Check 2 | >= 50 cells; non-globin erythroid CPM (Alas2, Slc4a1, Epb42, Car2, Slc25a37, Ahsp) >= 4 x every other cluster; median >= 100 genes per cell | `build_blood_composite.py:45,101-102` | erythroid cluster is cells | registered (`:298-301`) |
| Check 3 genes | interferon: Ifit1, Ifit3, Isg15, Rsad2; activation: Il1b, Cxcl2, Nfkbia; threshold = 95th percentile of PBMC monocytes | `build_blood_composite.py:104,119` | neutrophil activation | reported, no pass rule (`:302-306`) |
| Check 4 | every PBMC type in >= 4 of 5 rats | `build_blood_composite.py:131` | PBMC types | registered (`:307-308`) |
| Check 6 design | reference: rats V1-V3 + half of each whole-blood cluster; mixtures: rats V4-V5 + other half; 20 mixtures; 5 x 10^6 reads; erythroid ~ U(0, 0.6); neutrophils ~ U(0, 0.4); PBMC types at their observed mRNA ratios; seed 0 | `build_blood_composite.py:39,166-222` | simulated mixtures | registered (`:311-315`) |
| Check 6 pass | Pearson r >= 0.9 for erythroid, neutrophils, monocytes, T, B, NK (summed groups) | `build_blood_composite.py:41-44,236-237` | recovery floor | "the bar used for the other tissues' purity sweeps" (`BLOOD_REFERENCE_PROPOSAL.md:161-162`) |
| LUNG gene floor | detected in >= 10 pooled cells | `deconvolution/build_lung_pooled.py:45` | union gene set | trims the outer-join tail (`:45`) |
| TESTES ingest QC | >= 200 genes; mito <= 20%; gene in >= 3 cells | `deconvolution/ingest_omix767_testes.py:19` | cells and genes | "standard 10x QC" (`:19`) |
| TESTES state floor | 20 | `tissue_references.yaml:434` | states kept | default |
| S1 (SPLEEN) | B cells, T cells and (monocytes + macrophages, or neutrophils) each >= 100 cells | `deconvolution/build_external_reference.py:42,76-85` | minimum composition | registered (`REFERENCE_RELABEL_PREREG.md:564-565,605-606`) |
| O1 (OVARY) | granulosa and theca/stroma each >= 100 cells | `build_external_reference.py:43,87-89` | minimum composition | registered (`:566-567`) |
| O2 (OVARY) | cells labelled lung epithelium or hepatocyte <= 5% | `build_external_reference.py:41,90-97` | mislabelled library guard | registered (`:568-569`) |
| R2 design | half of each type (seed 0) as reference; 20 mixtures from the other half; Dirichlet(α = 1) over types with >= 20 cells in each half; 5 x 10^6 reads | `build_external_reference.py:40,118-173` | within-library hold-out | registered (`:571-575,610-612`) |
| R2 pass | r >= 0.9 for every type with mean true fraction >= 0.05 | `build_external_reference.py:192,196-197` | recovery floor | registered (`:573-574`) |
| §15 design | 20 mixtures, Dirichlet(1) over coarse classes with >= 50 cells, seed 0 | `build_external_reference.py:207-224` | cross-study mixtures | registered (`:594-596`) |
| §15 pass | cardiomyocyte r >= 0.9 and mean absolute error <= 0.10 | `build_external_reference.py:243-244` | pass rule | "the Chu 2022 bar" (`:597`) |
| SoupX contamination (attempt 1) | `autoEstCont` defaults, `forceAccept = TRUE` | `deconvolution/R/soupx_correct_reference.R:50` | per-sample rho | registered (`:440-442`) |
| SoupX per-nucleus rule (attempt 2) | top 2,000 soup genes; <= 100 marker-free genes per type; rho = min(0.9, A / UMI) | `soupx_correct_reference.R:74-78,86` | per-nucleus rho | registered (`:498-505`) |
| SoupX seed | `set.seed(0)` | `soupx_correct_reference.R:16` | R RNG | - |
| HIPPOC success criterion | Neurons >= 0.50 and no non-neuronal type > 0.30 (partial: Neurons 0.30-0.50; fail: Neurons < 0.30 or a non-neuronal type > 0.40) | `REFERENCE_RELABEL_PREREG.md:459-462` | reading of the corrected composition | registered before the build |

### Statistical procedures
- **CDSeq**: a Gibbs sampler (described in the script as "LDA-style", `run_cdseq_realbulk.R:11-13`) on integer
  counts; K fixed at 14; one chain; no convergence diagnostic is computed in the scripts. Components were
  matched to reference profiles by correlation after the fact (`TISSUE_REFERENCE_AND_PROPORTIONS.md:1044-1051`);
  the script that did so is NOT FOUND (see Deviations).
- **BLOOD check 6**: for each of six groups, Pearson r between true and estimated fraction across the 20
  mixtures; truth = realised read share of the cells drawn (cells drawn with replacement, in batches of at
  most 200, until the type's target reads are reached; `build_blood_composite.py:201-213`); estimate = sum of
  BayesPrism fractions of the group's member types (`:232-236`). One-sided floor at 0.9, no interval, no
  multiplicity adjustment. Units of replication are simulated mixtures built from two held-out rats and half
  of one whole-blood library.
- **R2 (SPLEEN, OVARY)** and **§15 (heart)**: per-type Pearson r across 20 Dirichlet mixtures; mean bias;
  §15 additionally the mean absolute error of the cardiomyocyte fraction (`build_external_reference.py:180-199,227-248`).
  R2 is a within-library test by design (`REFERENCE_RELABEL_PREREG.md:574-575`).
- **SoupX**: `autoEstCont` estimates one contamination fraction per sample; no project-specific statistic.
- **HIPPOC criteria** and the rule that the downstream chain of attempt 2 runs only on at least a PARTIAL
  result were fixed before each build (`REFERENCE_RELABEL_PREREG.md:459-469,508-510`).
- Pre-registration: §12 committed 2026-10-27 23:18:10 (commit `e9bc43a`); the GSE335156 files on disk are
  timestamped 23:18:35. §13 committed 2026-10-29 23:26 (`6fae0bf`); §13b 23:55 (`f52cf03`); §14/§15
  2026-10-31 17:35 (`4895834`).

### Checks, validation and their results

**BLOOD composite** (`data/deconvolution/blood_option_b/checks.json`; `data/deconvolution/validation/BLOOD_composite_check6/check6.json`, `design.json`):

| Check | Result | Verdict |
|---|---|---|
| 1 Composition | Blood-Ctrl: T cells 7,982; neutrophils 2,168; B 1,767; non-classical monocytes 910; NK 823; classical monocytes 283; megakaryocytes/platelets 150; cycling 135; erythroid 90 | pass (>= 100 neutrophils) |
| 2 Erythroid cluster is cells | one cluster (j20), 90 cells, median 357.5 genes; non-globin erythroid CPM 2,811 vs at most 264 in any other cluster (ratio 10.6) | pass |
| 3 Neutrophil activation | 53.1% of neutrophils above the monocyte 95th percentile on interferon genes; 19.0% on activation genes | reported; stated as a limitation |
| 4 PBMC types | all 11 PBMC types used are present in 5 of 5 rats; cross-rat r_self 0.52-0.90 | pass |
| 5 Gene space | PBMC 25,629 genes; whole blood 22,050 after the lift; 18,260 shared, all in the 22,050-gene bulk | pass |
| 6 Simulated mixtures | r: erythroid 0.951, neutrophils 0.952, monocytes 0.957, T 0.914, B 0.946, NK 0.959; mean bias: erythroid -0.237, neutrophils +0.100, T +0.064, B +0.044, NK +0.028, monocytes -0.003 | pass (floor 0.9) |

The with-globin and no-globin arms give identical numbers "because BayesPrism's outlier filter ... removes
the globins in both" (`REFERENCE_RELABEL_PREREG.md:329-331`; both arms in `check6.json`). On the real bulk the
composite gives erythroid cells a mean θ of 0.699 and lymphocytes below 0.01
(`TISSUE_REFERENCE_AND_PROPORTIONS.md:27`; `REFERENCE_RELABEL_PREREG.md:410-411`), and every T- and B-cell
block falls under the analysable-block threshold (`:370-373`).

**SPLEEN and OVARY** (`data/deconvolution/s14_checks/{SPLEEN,OVARY}.json`; `validation/{SPLEEN,OVARY}_R2/R2.json`):

| Check | SPLEEN | OVARY |
|---|---|---|
| Barcodes -> QC cells | 14,527 -> 13,929 (`REFERENCE_RELABEL_PREREG.md:627`) | 10,283 -> 9,104 (`:633`) |
| Dropped in re-annotation | 916 | 0 |
| S1 / O1 | B 3,077; T 5,544; monocytes + macrophages 1,493; neutrophils 319: pass | granulosa 1,477; theca/stroma 2,445: pass |
| O2 | - | 0 guard-class cells: pass (6.6% of cells have any Sftpc/Scgb1a1/Alb/Apoa1 count; reported) |
| R1 | WARN only (see 8c) | WARN only |
| R2 | 12 types scored, r 0.945-0.999: pass; B cells r 0.871 at mean true fraction 0.039, not scored | 13 scored, r 0.940-0.999: pass; macrophages, monocytes, pericytes under 0.05, not scored (r 0.993-0.997) |
| Gene space | 32,883 IDs -> 22,050, all in the bulk | 30,562 IDs, 20,795 in the bulk |
| On the real bulk | T cells about 1%; endothelial cells 0.150 from a 34-cell type; "implausible for spleen" (`:661-665`) | stromal 0.181, luteal 0.174, endothelial 0.160 (`:666-668`) |

**Second heart reference** (`validation/HEART_GSE307551_cross/s15_result.json`): cardiomyocyte r 0.944, mean
absolute error 0.043: PASS. Other classes: endothelial r 0.943, bias +0.151; lymphocytes r 0.970, bias -0.103;
fibroblasts 0.985 / -0.021; mural 0.989 / +0.054; myeloid 0.997 / -0.047; neural 0.997 / -0.033; epicardial
0.999 / -0.043. Gene mapping: 34,034 symbols -> 19,966 bulk IDs (`REFERENCE_RELABEL_PREREG.md:641-643`).

**HIPPOC** (`deconvolution/diagnostics/hippoc_refs4/6_refs4h_check.txt`, `6_refs4h2_check.txt`,
`refs4h_record/*`; `HIPPOC_REFS4_DIAGNOSIS.md:127-134`):

| Mean BayesPrism fraction, 50 samples | Uncorrected (merged names) | Attempt 1 (one rho per sample) | Attempt 2 (rho per nucleus) |
|---|---|---|---|
| Vascular cells | 0.501 | 0.504 | 0.465 |
| Neurons | 0.393 | 0.402 | 0.443 |
| Astrocytes | 0.085 | 0.073 | 0.075 |
| Contamination estimate | - | rho 0.047-0.167 per sample | A = 369-515 UMI per droplet |
| UMI removed, overall / endothelial | - | 7.3% / 7.9% | 8.8% / 23.4% |
| Verdict against the registered criterion | - | FAILS | FAILS |

The downstream chain of attempt 1, registered before its composition was read, reproduced the uncorrected
arm (same 11 hotspots; no HIPPOC block significant; `REFERENCE_RELABEL_PREREG.md:519-533`). Attempt 2's
downstream chain was not run, as registered (`:517-518`).

**TESTES**: validator PASS (8a); gene-lift accounting in `summary.txt` line 2; the before/after comparison is
quoted in `tissue_references.yaml:461-464` (the underlying table was not located for this audit).

**LUNG**: validator PASS; K1 pass (8b).

### Deviations, known defects and caveats
1. **BLOOD check 6 tests three quantities, not six.** In every mixture the PBMC types are mixed at one fixed
   set of ratios scaled by (1 - erythroid - neutrophil) (`build_blood_composite.py:192-198`), so the true
   fractions of monocytes, T, B and NK cells are proportional to one another across the 20 mixtures. Their
   four correlations all measure whether the estimate tracks the total PBMC share; they cannot detect
   mass moving between PBMC types. The registered text says "the rest PBMC types in their observed ratios"
   (`REFERENCE_RELABEL_PREREG.md:313`), so the design is as registered; the manuscript reports the six
   correlations without this qualification (`manuscript/motrpac_genecompass.tex:1320-1322`).
2. **BLOOD: thin and biased basis.** The erythroid profile is 90 cells from one pooled, adjuvant-treated,
   red-cell-lysed Lewis library on a different platform from the PBMC libraries; half of the neutrophils are
   interferon-high; there are no eosinophils or basophils (the marrow library has both but was excluded by
   the registration, `:326-327`). On simulated mixtures the erythroid fraction is under-estimated by 0.24 on
   average; on the real bulk lymphocytes fall below 0.01. A composition check proposed before registration
   (lymphocytes > neutrophils > monocytes; `REFERENCE_LABEL_AUDIT.md:255-257`) was not among the six
   registered checks and would not be met. The change "has not yet been put to" the collaborator who chose
   the PBMC reference (`REFERENCE_RELABEL_PREREG.md:279-281`; `tissue_references.yaml:113`).
3. **CDSeq evidence is approximate and partly untraceable.** Only the Reduce-Recover run produced output;
   `RUN_PARAMS.txt` still points to the full-fidelity folder, which is empty. No committed script computes
   the reported "about 80% unmatched" figure or the component-to-reference correlations (searched `*.py`,
   `*.R`, `*.sh`, `*.ipynb`, `slurm/` for `cdseq_gep`, `cdseq_proportions`, `estGEP`: only the two run scripts
   match). Both run scripts hard-code `data/deconvolution/results/motrpac/<TISSUE>` (`run_cdseq_rr.R:40,51`),
   and the outputs have since been moved to `results/_deployed_backup_20260715/`. K = 14 has no stated
   justification and no sensitivity run.
4. **No provenance file for three builders.** `build_blood_composite.py`, `build_external_reference.py` and
   `build_lung_pooled.py` never call `write_provenance`; BLOOD_composite, SPLEEN, OVARY and LUNG have no
   `provenance_*.json` (directory listings). Their input paths are hard-coded in
   `joint_reannotate.py:69-90` and `build_lung_pooled.py:25-29`, not read from the configuration.
5. **SPLEEN R2: the one type under the floor is not scored.** B cells (3,077 cells, the second-largest type)
   have r 0.871 but a mean true fraction of 0.039, under the registered 5% scoring limit. With Dirichlet(1)
   over 13 types the expected mean fraction of every type is 1/13 (0.077), so which types fall under 5% is
   decided by the 20 random draws. The rule was registered before the result (`:573-574`).
6. **GEO text and shipped IDs disagree for GSE306764.** The registration says the spleen library is "mapped
   to GRCr8 (our gene space)" (`REFERENCE_RELABEL_PREREG.md:547-548`); the files carry the 32,883 Rnor_6.0 IDs
   and went through the liftover (`:618-619`; `references_v4/SPLEEN_GSE306764/summary.txt:2`, `dropped_not_in_map=0`).
   Logged as found.
7. **OVARY reference genes are not restricted to the bulk space** (30,562 genes, 20,795 in the bulk); the
   intersection happens at deconvolution. Harmless, but the gene count in `summary.txt` overstates the usable
   genes.
8. **HIPPOC.** The default still deconvolves HIPPOC with the reference diagnosed as an ambient sink. Limits
   recorded and not addressed: choroid plexus is in the bulk and not in the reference; 19 bulk genes are near
   zero in the reference; the "soup" is built from dropped barcodes, not empty droplets
   (`REFERENCE_RELABEL_PREREG.md:471-473`). The diagnostic scripts hard-code absolute paths and the
   `results_refs4` tree (`deconvolution/diagnostics/hippoc_refs4/5_ambient_soup.py:12-14`). The two corrected
   references no longer exist, so the attempts can be re-examined only by rebuilding them.
9. **TESTES.** One of the two pooled arms is EDS-treated (8a); ingest QC differs from the corpus (20% vs 15%
   mitochondrial ceiling); 42 Sertoli cells carry the whole somatic compartment, and the tissue's dominant
   estimated type on the real bulk is Sertoli cells at 0.372 (`TISSUE_REFERENCE_AND_PROPORTIONS.md:31`), which
   the configuration flags as inflation ("absolute SOMATIC fractions are NOT" trustworthy,
   `tissue_references.yaml:459-460`). Docs give 5,848 or 5,850 cells for the pool
   (`REFERENCE_SELECTION_PLAN.md:96`; `tissue_references.yaml:452`); the built reference has 5,836.
10. **WATSC.** (a) No adipocytes: the source protocol removes them and no cluster expresses adipocyte genes
    (`REFERENCE_STUDY_GSE137869_MA2020.md:146-165`); the WATSC panel has no adipocyte class
    (`reference_marker_panels.yaml:185-190`). For bulk white adipose the dominant mRNA source therefore has no
    basis profile. (b) The depot (subcutaneous vs visceral) is not stated in the paper or GEO and is
    unresolved (`:178-197`). (c) One library per sex, each pooling 2-3 rats (`:46-50`). (d) The reference
    contains mammary epithelium from the female library (62 luminal and 45 glandular cells;
    `JOINT_REANNOTATION_RESULTS.md:97-99,119-120`). (e) Plasmacytoid dendritic cells take θ 0.048, "high for
    adipose" (`TISSUE_REFERENCE_AND_PROPORTIONS.md:28`). Differential use only (`tissue_references.yaml:227`).
11. **LUNG.** No AT1 cells in the reference (`JOINT_REANNOTATION_RESULTS.md:109`); alveolar macrophages are
    15,784 of 44,158 cells and endothelial cells 783, while endothelial cells are the dominant estimated type
    on the real bulk (0.534, `TISSUE_REFERENCE_AND_PROPORTIONS.md:23`).
12. **Second heart check.** Passes on its registered criterion; the endothelial fraction is over-estimated by
    0.15 and lymphocytes under-estimated by 0.10 on average, read by the authors as "consistent with reading
    HEART differentially" (`REFERENCE_RELABEL_PREREG.md:651-655`). It changes no reference.

### Methods references
- CDSeq: "Kang et al., PLoS Comput Biol 2019" (as given in `deconvolution/R/run_cdseq_realbulk.R:12-13`).
- SoupX: no citation in the code; Young MD, Behjati S, GigaScience 2020 [verified against Crossref; see REFERENCES.md].
- BayesPrism (used for the mixture checks): Chu T, Wang Z, Pe'er D, Danko CG, Nat Cancer 2022;3:505-517
  (`manuscript/motrpac_genecompass.tex:1491-1493`).
- MoTrPAC blood draw: "Schenk et al. 2024, Function, PMC11245678" (as given in `REFERENCE_RELABEL_PREREG.md:284-285`).
- GSE307551: "Purdy et al., PNAS 2026" (as given in `REFERENCE_RELABEL_PREREG.md:586`).
- MoTrPAC hippocampus response: "Cell Genomics 2024" companion paper (`HIPPOC_REFS4_DIAGNOSIS.md:104-106`);
  in the manuscript bibliography as Nair VD, et al., Cell Genomics 2024;4(6):100421
  (`manuscript/motrpac_genecompass.tex:1462-1464`). Whether these are the same paper was not verified.

---

## Stage 8e - The reference build

### Research role
Question: are the selected and labelled cells written to the files BayesPrism reads without loss,
duplication or resampling? The claim is that each reference contains every labelled cell of the selected
samples with its raw counts, that cell states are at least 20 cells, and that the build is reproducible
from `tissue_references.yaml`. It is checked by a post-build sample check, by a hard stop when joint labels
and loaded cells differ, and by provenance files where the builder writes them. Verdict: the on-disk
references agree with their specifications and with what Stage 8 read; the builds themselves have each been
executed once and were left out of the 2026-11-02 clean rebuild.

### Inputs and outputs
- Inputs: `data/training/preprocessed/qc_matrices/<sample>.h5ad`; the annotation files or
  `data/deconvolution/joint_labels/<REF>/cells_final.tsv.gz`; for TESTES, SPLEEN and the whole-blood library
  the liftover map `deconvolution/reference/motrpac_bulk_liftover.tsv`.
- Outputs per reference directory (`deconvolution/build_reference.py:359-379`): `reference_counts.mtx` (cells x
  genes, MatrixMarket, written as 32-bit integers, `:364`); `genes.tsv`; `cells_meta.tsv` (barcode, sample,
  leiden, cell_type, cell_state; `:366-367`); `summary.txt` (counts line, optional `gene_map:` line, cells per
  type; `:368-375`); `provenance_build_reference.json` (only from `build_reference.py`, `:439-442`).
- Sizes: see the per-tissue table at the top.

### Procedure
1. **Driver** (`deconvolution/build_references_from_config.py`): tissues with status `ready`, `needs-build` or
   `proposed` and a `reference_dir` are planned (`:30,92-105`); a reference directory shared by two tissues is
   built once (`:101-104`); BAT and HEART first run the deposited-label adoption (`:33-37`); the build command
   depends on the builder: `build_blood_composite.py build`, `build_external_reference.py build --ref`,
   `build_lung_pooled.py`, or `build_reference.py` with the tissue's filters (`:40-70`); an existing reference
   is skipped unless `--force` (`:136-137`); after a build the selection check of 8a is run on the result
   (`:149-155`). `--submit` wraps the whole plan in one batch job (16 cores, 128 GB, 6 h by default;
   `:82-84,116-119`).
2. **Load** (`build_reference.py:226-261,293-320`): per sample, raw counts are taken from `adata.raw` if
   present, else `adata.X`, with a warning if they are not integers (`:237-243`); labels are joined by barcode
   (`:245-256`); samples are concatenated with an inner join of genes (the default; `:312`); the optional
   gene map is applied to the concatenated matrix (`:314-315`); an optional minimum-cells gene filter follows
   (`:316-319`; 0 everywhere except LUNG's own builder).
3. **Labels**: with `--joint-labels`, every loaded cell must be in the label file and vice versa, otherwise the
   build exits; `cell_type` is the joint label, `cell_state` the joint cluster, and cells labelled DROP are
   removed (`:323-342`). Without it, the optional label scheme is applied to the consensus label (`:434`).
4. **Clean** (`:345-356`): cells with a missing label or a label in {`Unknown`, `unknown`, empty, `nan`, `NA`}
   (`:54`) are removed; then every cell state with fewer than `min_state_cells` (20) cells is removed.
5. **Export and provenance** (`:359-379,438-442`).

There is **no cell cap, balancing or subsampling**: "FULL cells with natural counts" (`build_reference.py:6-8`).

**Cell states.** Routes 1-3: one state per sample and cluster or label (`<sample>_c<cluster>`,
`build_reference.py:249`). Route 4: the joint cluster, shared across samples (`:338`); for the BLOOD composite
the cluster name is prefixed with its source (`build_blood_composite.py:150`).

**Gene filters.** The build itself applies none beyond the gene join, LUNG's 10-cell floor and the optional
lift. Four filters are applied to the reference in memory at deconvolution time by
`deconvolution/R/run_deconvolution.R` (covered with BayesPrism in Part 5; listed here because the
lists are reference-preparation artifacts):

| Filter | List / rule | Size | Where applied | Built by |
|---|---|---|---|---|
| Ribosomal, mitochondrial, hemoglobin | `deconvolution/reference/rat_exclude_genes.tsv` | 247 genes (226 "ribosomal", 13 "mito", 8 "hemoglobin") | `run_deconvolution.R:54-56` | generator script NOT FOUND (file added in commit `e4d5c93`) |
| Sex chromosomes | `deconvolution/reference/rat_sex_chrom_genes.tsv`: Ensembl genes on chromosome X or Y | 1,734 genes | `run_deconvolution.R:62-67`; on by default, `config/pipeline_config.yaml:384` | `deconvolution/build_sex_chrom_list.py:32,44` |
| Protein-coding only | `deconvolution/reference/rat_protein_coding_genes.tsv`: BioMart `Gene type == protein_coding` | 22,016 genes | `run_deconvolution.R:79-84`; on by default, `config/pipeline_config.yaml:378` | `deconvolution/build_protein_coding_list.py:41` |
| Expressed in >= 3 reference cells | `colSums(ref > 0) >= 3` | - | `run_deconvolution.R:93` | - |

BayesPrism's own outlier filter on the mixture (`outlier.cut = 0.01`, `outlier.fraction = 0.1`,
`run_deconvolution.R:109`) follows. Reference genes left after the four filters, from each tissue's
`provenance_run_deconvolution.json` (`params.n_ref_genes`): BAT 14,163; BLOOD 14,931; CORTEX 16,030; HEART
16,086; HIPPOC 15,313; HYPOTH 14,061; KIDNEY 14,893; LIVER 12,771; LUNG 15,816; OVARY 14,574; SKMGN and SKMVL
13,151; SMLINT 12,876; SPLEEN 13,182; TESTES 13,711; WATSC 12,870.

### Models and software
Python 3.12.11; scanpy 1.12, anndata 0.12.10, scipy 1.17.1, numpy 2.4.3, pandas 2.3.3
(`requirements.lock.txt:4,195,31,198,134,155`).

### Parameters, thresholds and flags

| Name | Value | Where set | What it controls | Rationale given |
|---|---|---|---|---|
| `--min-state-cells` | 20 | `deconvolution/build_reference.py:403`; `clean_cells(…, 20)` in `build_blood_composite.py:160`, `build_external_reference.py:114`, `build_lung_pooled.py:52` | smallest cell state kept | none stated |
| Unknown labels | `Unknown`, `unknown`, empty, `nan`, `NA` | `build_reference.py:54` | cells dropped | - |
| `--gene-join` | `inner` (all corpus references); `outer` inside the LUNG builder | `build_reference.py:413`; `tissue_references.yaml` per tissue; `build_lung_pooled.py:37,42` | genes kept across samples | cortex outer join was "a cross-species artifact" (`tissue_references.yaml:131`) |
| `--min-gene-cells` | 0 (LUNG: 10) | `build_reference.py:416`; `build_lung_pooled.py:45` | gene floor after concatenation | trims the union's tail |
| `--gene-map` | TESTES only (SPLEEN and blood lift inside their loaders) | `tissue_references.yaml:436` | gene-ID lift | source ships Rnor_6.0 IDs |
| `--label-scheme` | `none` in effect for all 15 references | `build_reference.py:404,431-434` | label merging | see 8b |
| Count type on disk | int32 | `build_reference.py:364` | storage | - |
| Buildable statuses | `ready`, `needs-build`, `proposed` | `build_references_from_config.py:30` | which tissues are planned | - |
| Default output root when `--out` is omitted | `data/deconvolution/references` | `build_reference.py:436`; `config/pipeline_config.yaml:392` | output location | legacy tree; every configured build passes `--out` |

### Statistical procedures
None.

### Checks, validation and their results
- Post-build sample check: all 16 tissues "built reference OK: N sample(s) match expect" (validator run for
  this audit).
- Cells read by Stage 8 equal the reference's cell count for all 16 tissues
  (`data/deconvolution/results_refs4all_rebuild/motrpac/<TISSUE>/provenance_run_deconvolution.json`,
  `params.n_ref_cells`, against `summary.txt`).
- Build provenance, where it exists (`provenance_build_reference.json` in each directory):

| Reference | Built | Commit recorded | Tracked files modified at build time |
|---|---|---|---|
| KIDNEY, LIVER, WATSC, MUSCLE, SMLINT | 2026-10-27 23:21 | `e9bc43a` | `build_reference.py`, `build_lung_pooled.py`, `build_references_from_config.py`, `joint_reannotate.py`, `tissue_references.yaml`, `tests/test_joint_reannotate.py` |
| CORTEX, HIPPOC | 2026-10-28 19:47 | `3115ad0` | none |
| TESTES (lifted) | 2026-10-22 19:07 | `448c3df` | 15 files, including `build_reference.py` and `build_references_from_config.py` |
| LUNG, BLOOD_composite, SPLEEN, OVARY | file dates 2026-10-28, 10-27, 11-01, 11-01 | no provenance file | - |
| HEART, BAT, HYPOTH | file dates 2026-07-16, 07-16, 07-15 | no provenance file | - |

- Unit tests for the joint-label build path exist (`tests/test_joint_reannotate.py:140-156`); not run here.
- The clean rebuild of 2026-11-02 started from the existing references: "Not covered: reference builds and
  joint re-annotation" (`REFS4ALL_REBUILD_PREREG.md:10-12,16`).

### Deviations, known defects and caveats
1. **The exclusion list removes the S6-kinase family.** `rat_exclude_genes.tsv` labels as "ribosomal" ten
   kinase genes and one binding protein: Rps6ka6 (`:12`), Rps6kc1 (`:16`), Rps6kb1 (`:18`), Rps6ka5 (`:22`),
   Rps6kl1 (`:26`), Rps6ka3 (`:28`), Rps6ka2 (`:60`), Rps19bp1 (`:86`), Rps6ka4 (`:117`), Rps6kb2 (`:119`),
   Rps6ka1 (`:198`). These are signalling kinases (Rps6kb1 is p70 S6 kinase), not ribosomal proteins. The
   defect was recorded on 2026-07-14 ("Whether removing this filter would change any DE conclusion is
   UNKNOWN", `TISSUE_REFERENCE_AND_PROPORTIONS.md:287-296,1341`) and is still in effect: Rps6kb1
   (`ENSRNOG00000003919`) is in the bulk gene list and absent from
   `data/deconvolution/results_refs4all_rebuild/motrpac/{SKMGN,LIVER,HEART}/pred_z/genes.txt` (checked for
   this audit). No per-cell-type result can contain these genes.
2. **Builds are single executions.** Seven of fifteen references have no build provenance; six were built
   from a working tree in which build scripts were modified and uncommitted (the five builds of 2026-10-27,
   whose changes were committed eleven minutes later in commit `f687414`, and TESTES); none was rebuilt in
   the clean rebuild. Bit-level
   reproducibility of the references from the committed code is therefore asserted, not demonstrated.
3. **The 20-cell floor is per state, not per type**, and there is no floor on a type's total. Types of 23-42
   cells are in production references: WATSC mural cells 23; CORTEX neutrophils 29 and T cells 30; MUSCLE
   Schwann cells 30; SPLEEN endothelial cells 34; HIPPOC mural cells 40; TESTES Sertoli cells 42 (each
   reference's `summary.txt`). The project's own diagnosis shows very small, low-quality types acting as
   sinks in deconvolution (`HIPPOC_REFS4_DIAGNOSIS.md:64`; SPLEEN's 34-cell endothelial type receives θ 0.150,
   `REFERENCE_RELABEL_PREREG.md:664-665`).
4. **Cells without an annotation match are dropped with a warning only** (`build_reference.py:254-256,347`),
   so a partial annotation file shrinks a reference silently unless the cell count is checked. For BAT and
   HEART this is the intended mechanism for cells the authors did not label (8b).
5. The provenance of MUSCLE, CORTEX and HIPPOC records `label_scheme` = `muscle` / `brain`, which was passed
   but not applied because joint labels were given (`build_reference.py:431-434`).
6. Two tissues (SKMGN, SKMVL) read the same directory; any change to it changes both.
7. Cell-type names become file names downstream through one shared sanitiser that transliterates Greek
   letters first (`deconvolution/celltype_names.py:19-37`); an earlier non-injective sanitiser had merged two
   KIDNEY types (`:4-11`). The current label sets are plain ASCII.

### Methods references
None beyond BayesPrism (Chu et al., Nat Cancer 2022;3:505-517; `manuscript/motrpac_genecompass.tex:1491-1493`),
whose human/mouse-only gene clean-up these rat lists replace (`deconvolution/R/run_deconvolution.R:7-8`;
`deconvolution/build_protein_coding_list.py:6-10`).

---

## Stage 8f - MoTrPAC bulk preparation and gene-ID liftover

### Research role
Question: what exactly is deconvolved, and in which gene-ID space? The bulk counts are the study's data;
the references and the model's vocabulary use a newer Ensembl annotation than the consortium's count
tables, so gene IDs are lifted. The claim is that the bulk enters deconvolution as raw counts for every
sample, with IDs lifted by auditable rules that never merge different genes, and that 94.8% of the
training-regulated genes end up with a model token. It is checked by a committed per-gene map and coverage
report and by a `git diff` reproducibility rule. Verdict: the map and report are internally consistent and
match the bulk on disk; 27% of bulk gene IDs are dropped; no sample is excluded; the annotation release is
mislabelled in the script, a doc and the manuscript.

### Inputs and outputs
- **Source.** R data package `MotrpacRatTraining6moData`, vendored as a git submodule
  (`.gitmodules:8-10`; `git submodule status` reports commit `f831a4f`, described as tag `v2.1.0`). Its
  `DESCRIPTION` says `Version: 2.0.0` (`vendor/MotrpacRatTraining6moData/DESCRIPTION:5`), and 2.0.0 is the
  installed version (`R_libs/MotrpacRatTraining6moData/DESCRIPTION:5`; `deconvolution/setup/r_packages.lock.tsv:185`).
  Installed and mirrored by `deconvolution/R/install_motrpac_data.sh` (package install `:70`; 206 `.rda`
  files mirrored `:76-78`; 58 further files fetched `:93-95`).
- **Objects used** (`config/pipeline_config.yaml:437`, folder `data/motrpac/rat_training_6mo/data`):
  `TRNSCRPT_<TISSUE>_RAW_COUNTS.rda` for 19 tissues (a data frame with four descriptor columns `feature`,
  `feature_ID`, `tissue`, `assay` and one column of raw counts per sample, named by vial label;
  `deconvolution/R/prepare_motrpac_bulk.R:67,162-164`; `MOTRPAC_BULK_LIFTOVER.md:37-38`); `FEATURE_TO_GENE.rda`
  (`:95`); `TRAINING_REGULATED_FEATURES.rda` (`:242`).
- **Annotation inputs**: BioMart `data/references/biomart/rat_gene_info.tsv` (43,360 Ensembl gene IDs); the
  model's token map `data/training/ortholog_mappings/rat_token_mapping.tsv` (22,213 IDs, 20,455 symbols);
  RGD `data/references/biomart/GENES_RAT.txt` (`deconvolution/reference/motrpac_bulk_liftover_report.txt:4-6`;
  `prepare_motrpac_bulk.R:62-65`).
- **Outputs per tissue**, `data/deconvolution/motrpac_bulk/<TISSUE>/`: `bulk.mtx` (samples x genes),
  `bulk_genes.tsv`, `bulk_samples.tsv` (`prepare_motrpac_bulk.R:170-173`). All 19 tissues are present on disk
  with 22,050 genes each.
- **Committed audit artifacts**: `deconvolution/reference/motrpac_bulk_liftover.tsv` (32,883 rows: `old_id`,
  `lifted_id`, `method`, `symbol`; md5 `d773ecdbd29c97a346276ca46f2b97ac`, the value recorded in the TESTES
  and SPLEEN reference summaries), `motrpac_bulk_liftover_report.txt`, `motrpac_missed_genes.tsv` (505 genes),
  `motrpac_missed_genes_summary.txt`.
- **Sample design.** 50 samples per tissue = 2 sexes x 5 groups x 5 animals; groups are the 8-week sedentary
  control and 1, 2, 4 and 8 weeks of training (`MOTRPAC_BULK_LIFTOVER.md:39-42`). TESTES has 25 samples (males)
  and OVARY 24 (females; the 4-week group has 4) (`motrpac_bulk_liftover_report.txt:45,50`; group counts
  tabulated for this audit from each `bulk_samples.tsv` joined to
  `deconvolution/reference/motrpac_sample_pheno.tsv`; every bulk sample is found in that table).

### Procedure
1. `deconvolution/R/prepare_motrpac_bulk.sh <TISSUE|ALL> <out_dir>` sets the paths from the configuration and
   runs the R script with the project R library (`:12-38`).
2. **Lookup tables** (`prepare_motrpac_bulk.R:71-123`).
   - Current-ID universe = token-map IDs united with BioMart IDs (`:81`).
   - Symbol to current ID: per upper-cased symbol, the lexicographically smallest Ensembl ID carrying it, taken
     from BioMart and overridden by the token map (`:85-90`).
   - `FEATURE_TO_GENE`: for each bulk Ensembl ID, the first listed gene symbol and the first listed Entrez
     ID (`:95-102`).
   - RGD: Entrez ID to current symbol, and each old symbol (split on `;`) to current symbol (`:111-121`).
3. **Lift rule, in priority order** (`lift_ids()`, `:140-158`):
   - `direct`: the bulk ID is in the current-ID universe; kept as is.
   - `symbol`: otherwise, the ID's `FEATURE_TO_GENE` symbol is looked up in the symbol-to-current-ID table.
   - `id_history`: otherwise, a current symbol is recovered through the Entrez ID (RGD) or, failing that, by
     treating the `FEATURE_TO_GENE` symbol as an RGD old symbol (`:128-133`), and that symbol is looked up in
     the same table.
   - `unmapped`: otherwise the row is dropped.
   RGD's own Ensembl ID column is deliberately not used (`:105-110`).
4. **Per tissue** (`process_tissue()`, `:161-180`): all sample columns are kept (`:164`); rows with a lifted ID
   are kept; rows that share a lifted ID are summed (`rowsum`, `:168`); the matrix is written samples x genes
   (`:171`).
5. **Audit artifacts** are written only on an `ALL` run or when their paths are passed (`:193`); the map is
   computed from the first tissue's gene list (`:196-203`), the 32,883 IDs being the same in every tissue
   (`motrpac_bulk_liftover_report.txt:34-52`, column `in_genes`).
6. **Report and missed-gene record** (`:225-354`): method tallies; coverage of the training-regulated
   transcript union by the token vocabulary before and after; a per-gene record of training-regulated genes
   still without a token, with a symbol-pattern category and an importance tier (`:289-306`); five spot
   checks (`:340-349`).
7. In the pipeline, Stage 8 step 1 runs the single-tissue form (`pipeline/run_stage8.py:75-82`); the default
   run prepared the bulk first and started Stage 8 at step 2 (`DECONV_REBUILD_RUNBOOK.md:74-76,121`).

**Normalisation and filtering before deconvolution.** None in this step: counts stay raw, no library-size
scaling, no expression filter, no sample filter. Gene filtering happens on the reference side at
deconvolution time (8e), followed by the intersection with the bulk genes and BayesPrism's outlier filter.

### Models and software
R 4.4.1 with `Matrix` 1.7-5 (`deconvolution/setup/r_packages.lock.tsv:2,172`); `MotrpacRatTraining6moData` as
above. No model.

### Parameters, thresholds and flags

| Name | Value | Where set | What it controls | Rationale given |
|---|---|---|---|---|
| Current-ID universe | token map ∪ BioMart (43,360 IDs) | `deconvolution/R/prepare_motrpac_bulk.R:81`; `motrpac_bulk_liftover_report.txt:6` | what counts as a current ID | - |
| Symbol tie-break | lexicographically smallest Ensembl ID; token map overrides BioMart | `prepare_motrpac_bulk.R:83-90` | one current ID per symbol | "stable"; token map preferred so the ID is tokenizable (`:83-84`) |
| First symbol / first Entrez per bulk ID | `!duplicated` | `prepare_motrpac_bulk.R:99-100` | bridge keys | none stated |
| Collapse rule | sum of raw counts over rows with the same lifted ID | `prepare_motrpac_bulk.R:168` | many-to-one lifts | "alternate annotations of one gene" (`:22-23`) |
| Unmapped rows | dropped | `prepare_motrpac_bulk.R:166` | genes without a current ID | "lossless for deconv" (`:24-25`) |
| RGD Ensembl column | not used | `prepare_motrpac_bulk.R:105-110` | avoids cross-annotation ID collisions | a previous version mis-summed 101 genes (`MOTRPAC_BULK_LIFTOVER.md:115-144`) |
| Sample columns | all | `prepare_motrpac_bulk.R:164` | samples kept | none stated |
| Importance tiers of missed genes | MHC genes: high if regulated in >= 2 tissues; curated genes: high >= 4, medium >= 2 tissues; all else low | `prepare_motrpac_bulk.R:303-306` | missed-gene record only | "documented, reproducible" (`:289`) |

### Statistical procedures
None. Deterministic table joins and sums.

### Checks, validation and their results
- **Lift tallies** (`motrpac_bulk_liftover_report.txt:9-13,33-52`): of 32,883 bulk gene IDs, 20,203 direct
  (61.4%), 2,702 by symbol (8.2%), 1,098 by ID history (3.3%), 8,880 unmapped (27.0%); 24,003 rows kept and
  summed into 22,050 distinct IDs (1,953 rows collapsed); identical in all 19 tissues. Distinct lifted IDs per
  method, tabulated from the map: 20,203 direct, 1,506 symbol, 935 ID history (the latter matches
  `MOTRPAC_BULK_LIFTOVER.md:141-142`).
- **Training-regulated genes** (`:15-21`): union of 9,800; 8,768 already in the token vocabulary (89.5%); 498
  recovered by symbol and 29 by ID history; 94.8% after the lift.
- **Still missed** (`:23`; `motrpac_missed_genes_summary.txt:5-11`): 505 genes (importance high 29, medium 50,
  low 426); 2 of the 505 occur in the single-cell corpus.
- **Spot checks** (`:25-30`): PLN, EP300, LTB, HINT1, PPP1R18 are recovered and present in the output.
- **Symbols with more than one token-map ID**: 158, resolved to the smallest ID (`:7`).
- **Reproducibility rule**: an `ALL` run rewrites the committed artifacts and `git diff --stat
  deconvolution/reference/` must be empty (`DECONV_REBUILD_RUNBOOK.md:74-84`). Not re-run for this audit.
- **ID-space audit** (`deconvolution/audit_idspace.py`; `data/deconvolution/idspace_audit/summary.txt:1-4`): the
  corpus gene union (22,489) is 100% in BioMart; 60.6% of RGD's Ensembl IDs are in BioMart.
- **Consistency checked for this audit**: every `bulk_genes.tsv` has 22,050 lines; the map's md5 equals the
  md5 stored in `references_v3/TESTES_OMIX767_lifted/summary.txt:2` and `references_v4/SPLEEN_GSE306764/summary.txt:2`.

### Deviations, known defects and caveats
1. **The annotation release is mislabelled.** The script header, its comments and the liftover document
   describe the target space as "mRatBN7.2 (Ensembl rel-113)" (`prepare_motrpac_bulk.R:2-3,10,80`;
   `MOTRPAC_BULK_LIFTOVER.md:18,48,55-57`), as do a configuration comment (`config/pipeline_config.yaml:430-431`)
   and the manuscript ("lifted to release 113", `manuscript/motrpac_genecompass.tex:1290`; also `:273,386,1252`).
   The configuration states that the BioMart files on disk are Ensembl 115 / GRCr8 and that "the label here
   used to say 113 / mRatBN7.2, which was wrong" (`config/pipeline_config.yaml:31-36`). The lift depends only on
   the files, so the numbers stand, but the release named in the methods text is wrong. The same comments say
   RGD's Ensembl IDs "track the newer GRCr8 assembly" (`prepare_motrpac_bulk.R:19-21,108-110`;
   `MOTRPAC_BULK_LIFTOVER.md:120-121`), which cannot be squared with the project's own files being GRCr8; the
   measured fact is only that 60.6% of RGD's IDs are in the project's BioMart file.
2. **No sample exclusion.** The consortium's outlier list (`OUTLIERS.rda`) is in the data folder and is not
   read by any script under `deconvolution/`, `pipeline/` or `lib/` (searched for "outlier"). All 50 (25, 24)
   samples per tissue are deconvolved. No rationale is stated.
3. **Package version is ambiguous.** The submodule commit is tagged v2.1.0; the package's own version field
   and the installed copy say 2.0.0. Cite the commit hash.
4. **Many-to-one lifts.** 815 lifted IDs receive more than one bulk row (2,768 rows; tabulated from the map),
   769 of them protein-coding (722 with two rows, 47 with three to eleven; e.g. eleven bulk IDs carrying the
   symbol HIST2H4 are summed into one current ID). Because a symbol carried by several current IDs resolves
   to the lexicographically smallest one (`prepare_motrpac_bulk.R:83-84`), the counts of a multi-copy family
   can land on one member, while the references keep per-locus counts. Non-coding families (U6: 481 rows
   into one ID) are removed later by the protein-coding filter. The effect on fractions was not measured.
5. **27% of bulk IDs are dropped.** The share of bulk *counts* lost to the 8,880 unmapped IDs per tissue is
   not in the report (NOT FOUND in `motrpac_bulk_liftover_report.txt` or `MOTRPAC_BULK_LIFTOVER.md`); the
   "lossless for deconv" statement rests on the argument that non-current IDs cannot intersect a reference.
6. **Stale audit output.** `data/deconvolution/idspace_audit/summary.txt` (dated 2026-06-08) reports 9,527
   `id_history` rows, i.e. the map before the bridge was corrected; the committed report has 1,098. The two
   files the bulk preparation actually reads from that folder (`corpus_genes.txt`, `id2symbol.tsv`) do not
   depend on the map. The audit's "built references" column reads the legacy `data/deconvolution/references`
   tree (`deconvolution/audit_idspace.py:21,40-42`).
7. **The sample phenotype table has no generator.** `deconvolution/reference/motrpac_sample_pheno.tsv` (6,156
   vial labels; columns `viallabel`, `sex`, `group`, `key.anirandgroup`, `intervention`, `sacrificetime`) was
   committed as a file (commit `3c2392e`); the script that derived it from the package's `PHENO` object is NOT
   FOUND (searched `*.R`, `*.py`, `*.sh` for `anirandgroup`).
8. **Sample identity is positional downstream.** `run_deconvolution.R:102` renames mixture rows `mix1..N`;
   results are mapped back to vial labels by row order of `bulk_samples.tsv` (`deconvolution/pheno_merge_test.py:8`).
9. Three tissues are prepared and never deconvolved (ADRNL, COLON, VENACV).

### Methods references
- MoTrPAC study and data: MoTrPAC Study Group, Temporal dynamics of the multi-omic response to endurance
  exercise training, Nature 2024;629(8010):174-183; Sanford JA, et al., Cell 2020;181(7):1464-1474
  (`manuscript/motrpac_genecompass.tex:1454-1460`). Data package: MotrpacRatTraining6moData (authors as in
  `vendor/MotrpacRatTraining6moData/DESCRIPTION:6-18`).
- Ensembl BioMart release 115 / GRCr8 and RGD gene table: as configured (`config/pipeline_config.yaml:30-38,63`);
  no publication is cited in the code.

---

## Consolidated audit findings (this section)

Ordered by likely importance to a reviewer. Each is detailed under the stage named.

1. **HYPOTH uses labels from the procedure the project discredited** (8b item 1). `deconvolution/tissue_references.yaml:311-335`;
   `deconvolution/REFERENCE_RELABEL_PREREG.md:174-208`.
2. **The QC gate does less than the manuscript says and tests nothing for BLOOD, SPLEEN, OVARY** (8c items 1-2).
   `deconvolution/reference_qc.py:63-77`; `pipeline/run_deconv_all.py:126-128`; `manuscript/motrpac_genecompass.tex:1295`.
3. **S6-kinase genes are excluded as "ribosomal" in every tissue of the default run** (8e item 1).
   `deconvolution/reference/rat_exclude_genes.tsv:12,16,18,22,26,28,60,117,119,198`.
4. **Bulk: no outlier exclusion; annotation release mislabelled as 113 in script, doc and manuscript; package
   version ambiguous** (8f items 1-3). `deconvolution/R/prepare_motrpac_bulk.R:2-3,164`;
   `config/pipeline_config.yaml:31-36`; `manuscript/motrpac_genecompass.tex:1290`.
5. **BLOOD composite: 90-cell erythroid profile; check 6's six correlations are three independent quantities;
   biased absolute fractions** (8d items 1-2). `deconvolution/build_blood_composite.py:192-198`;
   `data/deconvolution/blood_option_b/checks.json`.
6. **The 160 manual label decisions and their literature check were made by an AI assistant; no expert has
   reviewed them; citations in the check are unverified** (8b items 2-3). `deconvolution/JOINT_REANNOTATION_RESULTS.md:173-176`;
   `deconvolution/JOINT_REANNOTATION_LITCHECK.md:26-29`.
7. **HIPPOC is deconvolved in the default with a reference diagnosed as an ambient-RNA sink; both corrections
   failed** (8d item 8). `deconvolution/REFERENCE_RELABEL_PREREG.md:484-518`.
8. **Reference builds are unreproduced single executions; seven have no provenance; six were built from a
   dirty tree; BAT's annotation files predate the committed script** (8e item 2; 8b item 10).
   `deconvolution/REFS4ALL_REBUILD_PREREG.md:16`.
9. **TESTES pools an EDS-treated arm with the control** and is not flagged by the validator (8a item 3).
   `deconvolution/tissue_references.yaml:433`; `deconvolution/REFERENCE_ACCESSIONS.md:49`.
10. **CDSeq "80% unmatched" is from the approximate run only, and its scoring code is not in the repository**
    (8d item 3). `deconvolution/R/run_cdseq_rr.R:13-27`.
11. **A registered label merge for HEART was never applied** (8b item 12). `deconvolution/REFERENCE_LABEL_AUDIT.md:150`.
12. **No reference matches the bulk's strain; nine of fifteen references rest on one or two libraries**
    (8a items 1, 5).
13. Smaller: stale manifests and counts (`deconvolution/reference/canonical_references.tsv`;
    `REFERENCE_RELABEL_PREREG.md:626`); the LUNG builder hard-codes its samples (`deconvolution/build_lung_pooled.py:25-28`);
    SPLEEN R2 leaves B cells (r 0.871) unscored; 769 protein-coding lifted IDs receive more than one bulk row.

## Items that could not be determined
- Whether a `--deep` (depth-ratio) QC run on the current references shows a warning: not run (it loads the
  matrices) and no stored output found.
- The unit tests in `tests/test_joint_reannotate.py` were not executed.
- The outputs of the July GEO-record verification (13 verifiers) are not in the repository; only its summary.
- The script that scored CDSeq components against the PBMC reference, the generator of
  `rat_exclude_genes.tsv`, and the generator of `motrpac_sample_pheno.tsv`: NOT FOUND.
- Version or date of `ScTypeDB_full.xlsx`; Harmony arguments other than the seed (library defaults, not
  stated in the code).
- The share of bulk counts (as opposed to gene IDs) dropped by the liftover.
- Which Ensembl release RGD's `ENSEMBL_ID` column follows (code comments and the configuration conflict).
- Whether the head-to-head of the Ma muscle reference against the F344 reference on gastrocnemius bulk was
  ever run: no result found.
- Whether the BAT deposited-label adoption reproduces with the committed script (not re-run; the on-disk
  source tag differs from what the current code writes).
- The corrected HIPPOC references and their Stage 8 outputs were deleted; only the record tables could be
  inspected.

