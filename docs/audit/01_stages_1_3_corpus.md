# Part 1. Stages 1-3: corpus harvest, gene universe and cell QC, ortholog mapping

> One part of the methods audit of 2026-11-11; start at [`README.md`](README.md). The main findings of every
> part are collected in [`FINDINGS.md`](FINDINGS.md) and every reference in [`REFERENCES.md`](REFERENCES.md).
> Citations are `path:line` at commit `8249506` unless the text says otherwise.

Audit date 2026-11-11, branch `aim2b-perturbation-engine`, HEAD `8249506`. All paths are relative to
`/depot/reese18/apps/motrpac-genecompass`. Citations are `path:line` for code, config and docs. For on-disk JSON
and TSV result files (under `data/`, which are symlinks into `/depot/reese18/data/`) a line number is not
meaningful, so the citation is the file plus the key or column; where a number was tallied from such a file for
this audit, the text says "counted here".

Abbreviations used: GEO (NCBI Gene Expression Omnibus), GSE (GEO Series accession), GSM (GEO Sample accession),
AE (EBI ArrayExpress), SRA (Sequence Read Archive), RGD (Rat Genome Database), QC (quality control), LLM (large
language model), MTX (Matrix Market sparse matrix, the 10x triplet format), H5 (10x HDF5 matrix), h5ad (AnnData
HDF5 file), GC (GeneCompass), SD (standard deviation), T1-T4 (ortholog tiers defined in Stage 3).

"Canonical run" means the run whose outputs the v2 model was actually built from. Its dates differ by stage:

| Date | Event | Evidence |
|---|---|---|
| 2026-01-17/18 | First GEO harvest, two separate runs: a single-cell query (292 series) and a bulk query (1,573 series) | `data/raw/geo/logs/slurm_single_cell_10199742.log:4-10`; `data/raw/geo/logs/geo_harvest_20260117_225712.log:4-10` |
| 2026-01-18 | `master_catalog.json` written (2,670 rows) | `data/catalog/master_catalog.json` keys `generated_at`, `study_count` |
| 2026-01-19/20 | LLM study analysis (2,562 analyses) | `data/catalog/llm_study_analysis.json` key `analyses[]._meta.analyzed_at` (range counted here) |
| 2026-01-21 | `unified_studies.json` written (the study list Stage 2 step 2 reads) | `data/catalog/unified_studies.json` key `generated_at` |
| 2026-02-18 | A Stage 2 run that was cancelled; it left 54 QC files with log-normalized `X` and counts in `.raw` | `docs/repro_stages1_7/REPRODUCTION.md:101-104`; `/depot/reese18/data/training/preprocessed/raw_count_verification/verification_summary.txt` |
| 2026-02-20/21 | Re-harvest after a storage (RAID) loss: GEO `--mode both` (1,778 series), ArrayExpress (805 studies) | `data/raw/geo/harvest_summary_20260221_134040.txt:8-35`; `data/raw/arrayexpress/harvest_summary_20260220_194033.txt:8-36` |
| 2026-02-22 | Stage 2 step 1 (gene universe), git `14ab1ba` | `data/training/gene_universe/manifest_step1.json` keys `timestamp`, `git_hash` |
| 2026-02-23 | Stage 2 steps 2-3 (cell QC, pruning) | `data/training/gene_universe/preprocessing_report.json` key `timestamp`; `pruning_report.json` |
| 2026-02-24 | Reference folder `data/references/biomart/` rewritten | `docs/repro_stages1_7/stage2_reference_file_mtimes.txt:3-22` |
| 2026-04-01 | Stage 3 (ortholog mapping) | `data/training/ortholog_mappings/mapping_statistics.json` key `timestamp` |
| 2026-10-06 to 10-08 | Clean corpus plan for v2 and the OMIX767 testis addition | `config/profiles/retrain_v2.yaml:5-13`; `pipeline/02_gene_universe/qc_external_mtx.py:1-22` |
| 2026-10-31 to 11-01 | Reproduction of Stages 1-7 | `docs/repro_stages1_7/REPRODUCTION.md:1-10` |

Headline counts for the v2 corpus and vocabulary (details and sources in the stage blocks):

| Quantity | Value | Source |
|---|---|---|
| GEO series returned by the February query | 1,778 | `data/raw/geo/harvest_summary_20260221_134040.txt:27` |
| Catalog entries (January) | 2,670 rows, 2,562 distinct accessions (GEO 1,757, ArrayExpress 805) | `data/catalog/master_catalog.json` (counted here) |
| Catalog entries typed single-cell after the merge | 302 (GEO 184, ArrayExpress 118) | `data/catalog/unified_studies.json` key `studies[].catalog.data_type` (counted here) |
| Studies that entered cell QC | 138 | `data/training/gene_universe/preprocessing_report.json` key `summary.total_studies` |
| Studies with at least one QC-passing matrix | 80 | same file, key `summary.studies_success` |
| Matrix files attempted / passed in the canonical run | 1,038 / 810 | same file, `studies[].samples[].status` (counted here) |
| Cells after QC, canonical run (before cleaning) | 8,930,695 | same file, key `summary.total_cells_after_qc` |
| QC h5ad files listed in the v2 plan | 868 (9,494,923 cells) | `config/corpus_v2_plan.tsv` (counted here) |
| v2 kept corpus | 584 files, 81 studies (80 GEO + OMIX767), 6,803,206 cells | `config/corpus_v2_plan.tsv` rows with `action` starting `keep` (counted here); `docs/repro_stages1_7/REPRODUCTION.md:49` |
| Gene universe after step 1 | 22,488 genes | `data/training/gene_universe/manifest_step1.json` key `outputs.gene_universe.tsv.n_genes` |
| Pruned gene universe (the vocabulary) | 22,213 genes (20,727 protein-coding, 1,275 lncRNA, 211 miRNA) | `data/training/gene_universe/pruning_report.json` keys `summary.pruned_size`, `biotype_kept` |
| Tier counts | T1 13,178; T2a 705; T2b 3,613; T3a 1,351; T3b 892; T4 2,474 | `data/training/ortholog_mappings/mapping_statistics.json` key `tier_distribution` |
| Tokens reused / new | 17,496 / 4,717 | same file, keys `pre_trained_tokens_used`, `new_tokens_created` |

---

## Stage 1 - Data harvesting and study catalog

### Research role
Stage 1 answers: which public rat single-cell RNA-seq studies exist, and which of them have count matrices that
can be used to continue pretraining GeneCompass on rat? Two claims rest on it. (1) The training corpus is rat
single-cell data and contains no MoTrPAC data (the pre-registration states the corpus is "80 GEO single-cell
studies + OMIX767", `docs/RETRAIN_V2_PREREGISTRATION.md:16-17`). (2) The corpus can be re-obtained.

How it is checked: the harvesters' file-filter logic has offline unit tests (`tests/test_harvesters.py:28-55`);
the reproduction did not re-download anything, it verified the md5 and cell count of the 868 QC files that
Stage 2 made from the harvest (`docs/repro_stages1_7/REPRODUCTION.md:49`).

Verdict so far: the harvest itself is recorded (query string, flags, counts) but the list of studies that
entered the corpus depends on three things that cannot be regenerated from the committed code: an LLM
classification, a catalog built from a January directory layout that no longer exists, and a catalog-merge rule
that silently removed 108 series from consideration (see Deviations). The README treats `data/catalog/` as a
deposited input for this reason (`README.md:101-103`).

### Inputs and outputs
Inputs (all external services):
- NCBI GEO through Entrez E-utilities, database `gds` (`pipeline/01_data_harvesting/geo_harvester.py:960`) and
  the GEO FTP site (`geo_harvester.py:1051-1053`, `:1928`).
- EBI ArrayExpress through the EBI Search API, domain `arrayexpress`
  (`pipeline/01_data_harvesting/arrayexpress_harvester.py:169-174`).
- Ensembl BioMart, pinned to the Ensembl 115 archive host (`config/pipeline_config.yaml:120`), and RGD
  `GENES_RAT.txt` (`config/pipeline_config.yaml:123`).
- OMIX767 (rat testis, NGDC): no downloader; fetched by hand into `data/raw/ngdc/datasets/OMIX767/`
  (`README.md:98-100`).
- Anthropic Messages API for the LLM step (`pipeline/01_data_harvesting/llm_study_analyzer.py:533-539`).

Outputs on disk:
- `data/raw/geo/geo_datasets/`: one folder per GSE; 1,778 found and processed, 28,544 files downloaded,
  1,522.66 GB, 0 failed series, 12 SOFT-parse errors (`data/raw/geo/harvest_summary_20260221_134040.txt:27-35`,
  `:40-53`). One of the 12 is GSE143920, a study that the v2 corpus keeps (`:42`;
  `config/corpus_v2_accessions.txt`).
- `data/raw/geo/single_cell/geo_datasets` and `data/raw/geo/bulk/geo_datasets`: since 2026-02-21 both are
  symlinks to the one harvest folder (`README.md:92-94`; `ls -la data/raw/geo/single_cell data/raw/geo/bulk`).
- `data/raw/arrayexpress/datasets/`: 805 studies, 585.32 GB
  (`data/raw/arrayexpress/harvest_summary_20260220_194033.txt:27-36`).
- `data/references/biomart/`: 20 files with md5 and size pinned in `config/reference_md5sums.tsv:9-28`.
- `data/catalog/master_catalog.json` (2026-01-18): 2,670 rows from four sources: GEO single-cell 292, GEO bulk
  1,573, ArrayExpress single-cell 118, ArrayExpress bulk 687; 2,562 distinct accessions because 108 GSEs are in
  both GEO lists (key `sources`, `studies[]`; counted here).
- `data/catalog/matrix_analysis.json` (2026-01-19), `statistics.json` (2026-01-18).
- `data/catalog/llm_study_analysis.json` (2026-01-20): 2,562 analyses, 2,561 parsed successfully (key
  `statistics.total_analyzed`, `statistics.successful`).
- `data/catalog/unified_studies.json` (2026-01-21): 2,562 studies; its `sources.llm` key names
  `llm_test.json`, a file that is no longer in `data/catalog/`. Its per-study LLM fields agree with
  `llm_study_analysis.json` on model, `is_rat` and `is_single_cell` for all 2,562 accessions (compared here).

Note the dates: the catalog, matrix analysis and LLM output all predate the February re-harvest. They describe
the January study set. The February folder holds 21 GSEs that have no catalog entry and therefore could never be
selected for cell QC (counted here: GSE folders in `data/raw/geo/geo_datasets` minus accessions in
`unified_studies.json`).

### Procedure
The orchestrator `pipeline/run_stage1.py` defines ten steps (`run_stage1.py:58-148`) and runs each worker as a
subprocess with no arguments except step 10 (`run_stage1.py:173-176`). What the orchestrator would do and what
was actually run differ, so both are given.

**As coded in the orchestrator**

1. `fetch_biomart_reference_data.py`. For rat, mouse and human (`config/pipeline_config.yaml:125-128`) it sends
   three BioMart XML queries per species: all genes (ID, name, biotype), synonyms, UniProt symbols
   (`fetch_biomart_reference_data.py:175-203`), and writes `<species>_all_genes.tsv`, `_gene_synonyms.tsv`,
   `_uniprot.tsv` and a merged `<species>_symbol_lookup.pickle` (`:405-482`, `:715-720`). For rat it also
   downloads RGD `GENES_RAT.txt` if absent (`:485-491`). It then fetches four tables (`:520-582`):
   `rat_human_orthologs.tsv`, `rat_mouse_orthologs.tsv`, `rat_gene_info.tsv`, `mouse_human_orthologs.tsv`. The
   ortholog queries request the homolog gene ID, homolog gene name, `orthology_type`, `perc_id` and (human
   only) `perc_id_r1` (`:205-228`). No orthology-confidence attribute is requested.
2. `geo_harvester.py`. With no arguments it runs in `--mode single-cell` and without `--download`
   (`geo_harvester.py:2561-2564`), so it only searches and writes metadata.
3. `arrayexpress_harvester.py`. Same: default mode `single-cell`, no download
   (`arrayexpress_harvester.py:2998-3006`).
4. `extract_metadata.py`. Walks every study folder of every configured source that exists
   (`extract_metadata.py:395-404`, `:802-823`), parses the GEO SOFT file or the ArrayExpress IDF/SDRF
   (`:744-765`), and records title, summary, organism list, tissues, platforms, sample count and a file list
   capped at 500 files per study (`:732`; `config/pipeline_config.yaml:187-188`). It computes a 0-100
   "usability score" (+30 processed data, +30 count matrix, +10 each for metadata file, title, summary,
   samples; `:774-782`). Each study is tagged with the `data_type` of the source folder it was found in
   (`single_cell` or `bulk`; `:708`, `:815`). Output `master_catalog.json` (`:826-836`).
5. `analyze_matrices.py`. Header-only inspection of matrix files per study (`analyze_matrices.py:2-9`):
   gene and cell counts, gene-ID type, a data modality derived from the ID type (`:1343-1366`), and an
   unfiltered-10x flag when the barcode count is one of 6,794,880, 737,280 or 3,222,016 (`:623-631`). It
   de-duplicates studies present in both the single-cell and the bulk folder, preferring the single-cell entry
   when that folder holds more than half the bytes of the bulk folder (`:104-169`); this de-duplication applies
   to `matrix_analysis.json` only. It flags studies whose gene IDs are mouse or human Ensembl IDs (`:186-205`).
6. `generate_statistics.py`. Aggregate reporting only (`generate_statistics.py:2-14`).
7. `llm_study_analyzer.py`. See "Models and software" below. Output `llm_study_analysis.json`.
8. `combine_data_sources.py`. Indexes the catalog, the matrix analysis and the LLM output by accession and
   merges them (`combine_data_sources.py:635-683`). The catalog index is a dict comprehension, so when an
   accession occurs twice the later row wins (`:257`). The LLM block is reshaped (`:460-583`):
   `utility_for_motrpac.is_rat` becomes `llm.motrpac_utility.is_rat` (default `False`, `:566`) and
   `study_type.is_single_cell` becomes `llm.study_type.is_single_cell` (default `False`, `:536`). A study with
   no LLM entry gets `llm = {'success': False}` (`:675-676`). Output `unified_studies.json` (`:949`, `:977-979`).
9. `check_matrix_results.py` and 10. `analyze_llm_output.py`. Print-only QC reports
   (`check_matrix_results.py:2-8`; `analyze_llm_output.py:2-4`).

**As actually run (from the logs and outputs)**

- January 2026: GEO was harvested twice by an earlier copy of the script
  (`/depot/reese18/apps/data_fetch/geo_harvester.py`, named in
  `data/raw/geo/logs/slurm_single_cell_10199742.log`), once with the single-cell query (292 hits) and once
  with the bulk query (1,573 hits), into separate `single_cell/` and `bulk/` folders, with generic patterns,
  SRA run tables and GSM-level supplementary files enabled (`slurm_single_cell_10199742.log:4-10`;
  `geo_harvest_20260117_225712.log:4-10`). ArrayExpress logs from January were NOT FOUND (looked in
  `data/raw/arrayexpress/` and `data/raw/geo/logs/`); the catalog lists 118 single-cell and 687 bulk
  ArrayExpress studies. Steps 4-8 then produced the catalog files dated 2026-01-18 to 01-21.
- February 2026, after the storage loss: GEO was re-harvested in one run with `--mode both`,
  microarray patterns, generic patterns, SRA run tables and GSM-level supplementary files
  (`data/raw/geo/harvest_summary_20260221_134040.txt:8-17`). ArrayExpress was re-harvested with mode `both`
  and sample files (`data/raw/arrayexpress/harvest_summary_20260220_194033.txt:8-17`). The catalog was not
  rebuilt.
- The rebuild instruction in the README is to run `geo_harvester.py` directly in replay mode with the 733
  accessions of `config/stage2_step1_accessions.txt`, then `run_stage1.py --from 4` (`README.md:90-101`).

**Search queries.** GEO: the term is built by `build_search_term` (`geo_harvester.py:905-933`) from two
keyword groups, `single cell RNA seq OR scRNA-seq OR single nucleus OR snRNA-seq OR 10x genomics OR drop-seq`
(`:896-899`) and `RNA-seq OR bulk RNA seq OR transcriptome sequencing OR mRNA sequencing` (`:901-903`), plus
`AND <organism>[Organism]` (`:927`) and `AND gse[ETYP]` (`:930-931`). The exact string used on 2026-02-21 was

`(single cell RNA seq OR scRNA-seq OR single nucleus OR snRNA-seq OR 10x genomics OR drop-seq) OR (RNA-seq OR bulk RNA seq OR transcriptome sequencing OR mRNA sequencing) AND Rattus norvegicus[Organism] AND gse[ETYP]`

(`data/raw/geo/harvest_summary_20260221_134040.txt:10`). In mode `both` the two groups are joined by `OR`
without an outer pair of parentheses (`geo_harvester.py:924`), so whether the organism and entry-type filters
bind to both groups depends on Entrez's operator evaluation order [left to right by NCBI's documented rule for Entrez Boolean operators; not re-checked in this audit]. The result is
consistent with the filters applying to both: all 1,757 January GEO accessions are in the February folder, which
has 21 more (counted here). No date limit was used (`--maxdate` exists, `geo_harvester.py:2586-2588`, and is
absent from the recorded parameters). ArrayExpress: free-text query
`(rat OR "Rattus norvegicus") AND (RNA-seq OR transcriptome OR "single cell" OR scRNA-seq OR "single nucleus" OR snRNA-seq OR "10x genomics" OR drop-seq OR smart-seq)`
(`data/raw/arrayexpress/harvest_summary_20260220_194033.txt:10`; built at
`arrayexpress_harvester.py:1098-1101`, `:1139-1164`).

**Which files are downloaded.** A supplementary file is downloaded when its name matches a target regular
expression and no skip pattern (`geo_harvester.py:766-828`). Target lists: single-cell (`:503-553`), bulk
(`:556-596`), microarray (`:599-631`), generic (`:634-669`), metadata (`:672-678`, always on, `:713-714`);
skip list (`:681-688`). Series `filelist.txt` is read first to decide which archives to fetch (`:1213-1251`,
`:1965-1972`); tar, zip and gz files are unpacked in place (`:1737-1765`). md5 sidecars are written by default
since 2026-11-01 (`:2577-2580`); the February harvest predates that.

**Inclusion and exclusion at Stage 1.** The harvesters apply no filter beyond the search query: no organism
check on individual samples, no assay check, no title filter, no de-duplication of GSMs and no handling of
SuperSeries/SubSeries (a search of `pipeline/01_data_harvesting/`, `pipeline/02_gene_universe/` and `lib/` for
"superseries", "subseries", "doublet" returns nothing). Study-level inclusion is decided at Stage 2 step 2 from
`unified_studies.json` (see Stage 2, Procedure, step 2a); the fields it uses are the catalog `data_type` and the
LLM's `is_rat` and `is_single_cell`.

### Models and software
**LLM study analyzer** (`pipeline/01_data_harvesting/llm_study_analyzer.py`).
- What it is: one call to the Anthropic Messages API per study,
  `client.messages.create(model=<model>, max_tokens=8192, temperature=0, system=SYSTEM_PROMPT, messages=[{"role": "user", "content": prompt}])`
  (`:533-539`). Calls are sequential (`:926-927`), rate-limited to 40 per minute by a hard-coded value (`:428`),
  results saved every 10 studies (`:971-972`), resume on by default (`:1169`, `:1196`).
- Model as configured: `claude-haiku-4-5` (`config/pipeline_config.yaml:158`; default `:417`, `:1194`),
  overridable with `--model` (`:1176`).
- Model as recorded in the output: `claude-sonnet-4-5-20250929` for 1,336 analyses, `claude-haiku-4-5` for
  1,216, `claude-3-5-haiku-20241022` for 10 (`data/catalog/llm_study_analysis.json` key
  `analyses[]._meta.model`, counted here). All 138 studies that entered cell QC, and therefore all 80 GEO
  studies of the v2 corpus, were classified by `claude-sonnet-4-5-20250929`; the Haiku analyses that touch the
  single-cell catalog are the 118 ArrayExpress studies (`data/catalog/unified_studies.json` key
  `studies[].llm.model`, counted here).
- System prompt: a curator instruction with MoTrPAC context and the list of 18 MoTrPAC tissues, requiring
  JSON-only output (`:164-198`).
- User prompt (`:362-411`): accession, source, catalog data type; the pre-extracted title, summary, design,
  organisms, tissues, technologies, platforms, sample count, date, PubMed IDs; the matrix-analysis results; a
  directory tree (depth 3, at most 200 files, `:306`); and raw metadata: the GEO family SOFT file truncated at
  100,000 characters (`:204-216`), ArrayExpress SDRF at 50,000 and IDF at 30,000 (`:221-249`), and up to three
  other metadata files at 20,000 each (`:287-296`, `:481`). The reply must follow a fixed JSON schema
  (`:61-158`).
- Output handling: if the reply stops at `max_tokens` the JSON is closed heuristically (`:548-552`, `:692-726`);
  a parse failure is stored as `success: False` (`:599-612`). In the on-disk output no reply was truncated and
  one failed (keys `_meta.truncated`, `_meta.success`, counted here). Token usage summed over the output:
  65,840,162 input and 2,688,623 output tokens (keys `_meta.input_tokens`, `_meta.output_tokens`).
- Pre-filters as configured: only studies whose catalog organism list contains "rattus" (`:884-888`;
  `config/pipeline_config.yaml:165`) and only "usable" studies, meaning 15,000-150,000 genes, modality `mrna`
  or `unknown`, not unfiltered 10x (`:890-898`; `config/pipeline_config.yaml:166`). The on-disk output has an
  analysis for every one of the 2,562 catalog accessions, including 356 with `is_rat = False`, so the January
  run did not use the "usable" filter as now configured. The January command line was NOT FOUND (looked in
  `data/catalog/`, `data/raw/geo/logs/`, `docs/`).
- What the output decides: Stage 2 step 2 drops a catalog single-cell study when `llm.motrpac_utility.is_rat`
  is `False` or `llm.study_type.is_single_cell` is `False`
  (`pipeline/02_gene_universe/preprocess_training_matrices.py:1274-1294`). On the on-disk catalog this removed
  153 of the 302 single-cell-typed studies: 85 as not rat and 68 as not single-cell; by source, 46 GEO studies
  (43 not single-cell, 3 not rat) and 107 ArrayExpress studies (counted here by replaying `:1262-1313` on
  `data/catalog/unified_studies.json`; the replay returns the same 138 accessions as
  `preprocessing_report.json`).
- Other consumers: `analysis/annotation_recovery.py:712` reads `llm_study_analysis.json` for study-level tissue
  in the cell-annotation track (outside Stages 1-3). The other LLM fields (topic, disease, treatments, utility
  flags) are not read by Stages 2-3.
- Determinism: temperature 0 but not reproducible; the README says the step "is not deterministic" and lists
  `data/catalog/` as deposited (`README.md:102-103`).

**Other software.** Versions are from `requirements.lock.txt`, a freeze taken 2026-10-05
(`requirements.lock.txt:4`), months after the Stage 1-3 canonical runs; the lock file says the February harvest
ran from an earlier environment and that GEOparse was added to the lock by hand on 2026-11-01
(`requirements.lock.txt:13-15`). The versions in force in January-April 2026 are not recorded in any manifest.

| Software | Version in lock | Used for | Call |
|---|---|---|---|
| Python | 3.12.11 (`requirements.lock.txt:4`) | all | |
| Biopython `Entrez` | 1.86 (`requirements.lock.txt:40`) | GEO search | `Entrez.esearch(db="gds", term, retmax=500, retstart)` (`geo_harvester.py:960`, `:991`); `Entrez.esummary(db="gds")` in batches of 400 (`:1032`; `config/pipeline_config.yaml:148`); only accessions starting `GSE` kept (`:1042`) |
| GEOparse | 2.0.4 (`requirements.lock.txt:77`) | series metadata | `GEOparse.get_GEO(geo=gse, destdir, silent=True, annotate_gpl=False)` (`geo_harvester.py:2305`) |
| requests | 2.33.0 (`requirements.lock.txt:189`) | BioMart, RGD | `requests.get(BIOMART_SERVER, params={'query': xml}, timeout)` (`fetch_biomart_reference_data.py:266-270`) |
| anthropic SDK | 0.94.0 (`requirements.lock.txt:33`) | LLM step | see above |
| Ensembl BioMart | release 115, GRCr8, host `sep2025.archive.ensembl.org` (`config/pipeline_config.yaml:35-38`, `:120`) | reference tables | datasets `rnorvegicus_gene_ensembl`, `mmusculus_gene_ensembl`, `hsapiens_gene_ensembl` (`fetch_biomart_reference_data.py:166-170`) |
| RGD `GENES_RAT.txt` | header `GENERATED-ON: 2026/02/21` (`data/references/biomart/GENES_RAT.txt:3`), md5 `340a1191...` (`config/reference_md5sums.tsv:10`) | symbol synonyms | no dated archive exists (`config/pipeline_config.yaml:121-123`) |

### Parameters, thresholds and flags
| Name | Value | Where set (path:line) | What it controls | Rationale given |
|---|---|---|---|---|
| GEO mode (February harvest) | `both` | `data/raw/geo/harvest_summary_20260221_134040.txt:8` | Which keyword groups and file patterns are used | none stated |
| GEO organism | `Rattus norvegicus` | `geo_harvester.py:2562`; summary `:9` | `[Organism]` term of the query | none stated |
| GEO include microarray / generic / SRA info / GSM supplementary | True / True / True / True; SRA FASTQ download False | summary `:12-16` | Extra file patterns, SRA run tables, per-sample files | none stated |
| GEO replay command in README | `--mode both --download --resume --batch-esummary 400 --workers 8 --accessions config/stage2_step1_accessions.txt` | `README.md:90-91` | Re-download of the 733 scanned series | "as the February 2026 harvest" (`README.md:95`), but it omits `--include-microarray`, `--include-generic`, `--include-sra`, `--include-gsm-suppl` |
| `harvesting.geo_batch_esummary` | 400 | `config/pipeline_config.yaml:148` | UID-to-GSE batch size | none stated |
| Entrez page size | 500 | `geo_harvester.py:991` | `retmax` per search page | none stated |
| Entrez pause | 0.12 s with API key, 0.34 s without | `geo_harvester.py:495-496` | Request spacing | NCBI rate limits (`:489`) |
| FTP timeouts | 60 s filelist, 1,800 s download, 300 s GSM file | `geo_harvester.py:1177`, `:1911`, `:1641` | Network | none stated |
| `harvesting.geo_query` | `"rattus norvegicus[organism] AND single cell RNA-seq"` | `config/pipeline_config.yaml:147` | Nothing: no code reads it | none stated |
| ArrayExpress mode (February) | `both`, sample files True, microarray/generic/ENA False | `data/raw/arrayexpress/harvest_summary_20260220_194033.txt:8-17` | Query and file patterns | none stated |
| `harvesting.biomart_server` | `https://sep2025.archive.ensembl.org/biomart/martservice` | `config/pipeline_config.yaml:120` | BioMart host | the live host moves with each release (`:118-119`) |
| `harvesting.biomart_timeout` | 600 s | `config/pipeline_config.yaml:124` | Query timeout | none stated |
| `harvesting.fetch_species` | rat, mouse, human | `config/pipeline_config.yaml:125-128` | Species fetched | none stated |
| `harvesting.test_genes` | 14 symbols | `config/pipeline_config.yaml:129-143` | Coverage printout only (`fetch_biomart_reference_data.py:585-598`) | none stated |
| `harvesting.extraction.max_files_per_study` | 500 | `config/pipeline_config.yaml:187-188` | File-list cap in the catalog | none stated |
| `harvesting.sources.*.path` | `geo/single_cell/geo_datasets`, `geo/bulk/geo_datasets`, `arrayexpress/singlecell/datasets`, `arrayexpress/bulk/datasets` | `config/pipeline_config.yaml:171-185` | Folders scanned by steps 4-8 and by Stage 2 | none stated; the two ArrayExpress folders do not exist on disk |
| `harvesting.llm_model` | `claude-haiku-4-5` | `config/pipeline_config.yaml:158` | LLM model default | none stated |
| LLM `max_tokens`, `temperature` | 8192, 0 | `llm_study_analyzer.py:535-536` (hard-coded) | Reply length, sampling | "Increased significantly for detailed JSON" (`:535`) |
| LLM rate limit | 40 calls/min | `llm_study_analyzer.py:428` (hard-coded) | Call spacing | "Slightly lower for safety" (`:428`) |
| `harvesting.llm_rate_limit`, `llm_max_tokens`, `llm_cost_limit_usd` | 40, 8192, 5.0 | `config/pipeline_config.yaml:163-164`, `:159` | Nothing: no code reads them | none stated |
| `harvesting.llm_filter_organism` | `rattus` | `config/pipeline_config.yaml:165` | Substring filter on catalog organisms before the LLM call | none stated |
| `harvesting.llm_filter_usable` | true | `config/pipeline_config.yaml:166` | 15,000-150,000 genes, modality mrna/unknown, not unfiltered 10x (`llm_study_analyzer.py:891-896`) | none stated |
| `harvesting.min_cells_per_study`, `min_genes_per_cell`, `max_mito_fraction`, `matrix_staging_dir`, `catalog_file` | 4, 200, 0.15, paths | `config/pipeline_config.yaml:153-156`, `:168` | Nothing: no code reads the `harvesting.*` copies of these keys | none stated |
| Unfiltered-10x barcode counts (matrix analysis) | 6,794,880; 737,280; 3,222,016 | `analyze_matrices.py:623-627` | Flag used by the LLM "usable" filter | none stated |
| Duplicate-folder preference | keep single-cell entry if its bytes > 0.5 x bulk bytes | `analyze_matrices.py:153-159` | `matrix_analysis.json` only | "prefer single_cell if it has substantial data" (`:153`) |

### Statistical procedures
None. Stage 1 contains no test or estimate. The only quantitative summaries are counts. The LLM output includes
its own judgment of whether the automatically extracted organism was correct (2,122 of 2,561, 82.9%) and whether
the extracted tissues were correct (1,616, 63.1%) (`data/catalog/llm_study_analysis.json` key
`statistics.validation`); these are the model's opinions about the parser, not a measured accuracy of the model.
Nothing is pre-registered for this stage.

### Checks, validation and their results
| Check | Result | Pass rule | Where |
|---|---|---|---|
| Harvest completeness per run | 1,778 of 1,778 series processed, 0 failed, 12 with SOFT parse errors | none defined | `data/raw/geo/harvest_summary_20260221_134040.txt:27-53` |
| File-filter unit tests | present; offline | pytest pass | `tests/test_harvesters.py:28-55` (no stored test log found) |
| Live API checks | present, opt-in | `pytest -m network` | `tests/test_harvesters.py:117-145` |
| "Stage 1" row of the reproduction | 868 of 868 QC files match the plan's md5 and cell count; kept cells 6,803,206 | exact match | `docs/repro_stages1_7/REPRODUCTION.md:49`; `docs/repro_stages1_7/stage1_qc_inputs.tsv` (868 `MATCH`, counted here). This checks Stage 2 outputs, not the download. |
| Reference files | md5 of 20 files pinned | Stage 2 step 1 refuses a mismatch | `config/reference_md5sums.tsv:9-28`; `lib/gene_utils.py:756-790` |
| Accuracy of the LLM's `is_rat` / `is_single_cell` calls | NOT FOUND: no manual review, agreement statistic or audit of the 153 exclusions | | looked in `docs/`, `data/catalog/fix_plan.md`, `tests/` |
| Re-download of the raw data | not done in the reproduction | | `docs/repro_stages1_7/REPRODUCTION.md:40-49` |

### Deviations, known defects and caveats
1. **108 GEO series were removed from consideration by a merge artifact.** The January single-cell and bulk
   queries both returned these 108 series (37% of the 292 single-cell hits). `master_catalog.json` holds each
   twice, single-cell row first; `combine_data_sources.py:257` keeps the last row, so all 108 carry
   `catalog.data_type = "bulk"` in `unified_studies.json` (counted here), and Stage 2 skips every study whose
   catalog type is not `single_cell` (`preprocess_training_matrices.py:1269-1272`). The LLM judged 79 of the 108
   to be rat and single-cell; of those, 52 have a matrix file detected by the Stage 2 step 1 scan, 40 passed its
   BioMart match gate, and 35 have a usable MTX, H5 or h5ad matrix (counted here from
   `data/training/gene_universe/study_coverage.tsv`). None is in `config/corpus_v2_plan.tsv`. The
   de-duplication in `analyze_matrices.py:104-169` does not reach the catalog. This exclusion is not mentioned
   in any document I found. How many cells it cost is not known.
2. **A fresh run of steps 4-8 cannot regenerate the study list.** Besides LLM non-determinism, the study typing
   came from the January two-folder layout. With the layout the README prescribes (both `single_cell/` and
   `bulk/` linked to one folder, `README.md:92-94`), `extract_metadata.py:395-404` would list every GSE twice and
   the last-wins merge would type every GEO series `bulk`, leaving Stage 2 step 2 with no GEO study. This
   follows from the code; it was not executed.
3. **ArrayExpress contributes nothing.** The configured ArrayExpress paths do not exist (data are in
   `data/raw/arrayexpress/datasets/`), so Stage 2 step 1 scans only GEO (`manifest_step1.json` key
   `inputs.source_dirs`) and step 2 drops the 11 ArrayExpress studies that pass every other filter as
   `no_dir` (`preprocess_training_matrices.py:1304-1307`; counted here). The manuscript's corpus sentence counts
   805 ArrayExpress studies in the catalog (`manuscript/motrpac_genecompass.tex:267-268`), which is correct for
   the catalog but none reaches the corpus.
4. **The LLM model is misreported.** Config and README say Claude Haiku (`config/pipeline_config.yaml:158`;
   `README.md:102`); the stored output shows Claude Sonnet 4.5 made every classification that mattered for the
   corpus (above).
5. **The orchestrator does not reproduce the harvest.** Steps 2-3 call the harvesters with defaults (no
   download). The README replay command drops three flags the February run used. A pattern test on the 13 kept
   source files that sit at a study root shows they would still match in replay mode, and the other 569 kept
   sources sit under `<GSE>_RAW/` folders (`config/stage2_sample_order.tsv` column `source`, counted here), so
   the corpus matrices are probably unaffected; the 733-series input of the gene-universe scan could differ.
   Not tested.
6. **Catalog scope is mislabelled in the manuscript.** "rat single-cell RNA-seq metadata for 2,562 studies"
   (`manuscript/motrpac_genecompass.tex:267`): 2,152 of the 2,562 catalog entries came only from the bulk
   queries (1,465 GEO, 687 ArrayExpress; counted here).
7. **Unused config keys** give a false picture of what is controlled: `geo_query`, `llm_rate_limit`,
   `llm_max_tokens`, `llm_cost_limit_usd`, `min_cells_per_study`, `harvesting.min_genes_per_cell`,
   `harvesting.max_mito_fraction`, `matrix_staging_dir`, `catalog_file` (searched all `*.py`, `*.sh`, `*.R`
   outside `vendor/` and `tmp/`).
8. `data/references/biomart/fetch_metadata.json` records only the human fetch (file content; the older script
   overwrote it); the current script merges runs (`fetch_biomart_reference_data.py:103-139`).
9. `docs/BUGS.md` (last updated 2026-02-10, `docs/BUGS.md:2`) and `docs/ARCHITECTURE.md` describe scripts that
   no longer exist (`build_gene_inventory`, `prune_vocabulary`) and settings the code does not use (identity
   thresholds 50/70/80, `docs/ARCHITECTURE.md:243-246`). `README.md:203` still points readers to
   `docs/ARCHITECTURE.md` "for details".

### Methods references
- GEO / Entrez E-utilities: no citation in the repo. [verified against Crossref; see REFERENCES.md] Barrett T et al., NCBI GEO: archive for functional
  genomics data sets - update, Nucleic Acids Res 2013;41:D991-D995.
- ArrayExpress / BioStudies: no citation in the repo. [suggested: Sarkans et al. 2021, verified; see REFERENCES.md]
- Biopython: no citation in the repo. [verified against Crossref; see REFERENCES.md] Cock PJA et al., Bioinformatics 2009;25:1422-1423.
- GEOparse: no citation in the repo (package pinned in `requirements.lock.txt:77`).
- Claude models (Anthropic): no citation in the repo; model IDs as recorded in
  `data/catalog/llm_study_analysis.json`.
- Ensembl BioMart, release 115: no citation in the repo. [suggested: Dyer et al. 2025 and Kinsella et al. 2011, verified; see REFERENCES.md]
- RGD: no citation in the repo. [suggested: Vedi et al. 2023, verified; see REFERENCES.md]

---

## Stage 2 - Gene universe and cell QC

### Research role
Stage 2 answers two questions. Which rat genes form the model's vocabulary, in one fixed ID space? And which
cells, with raw counts, form the training corpus? The claims that rest on it: the vocabulary is the set of
Ensembl rat genes (protein-coding, lncRNA, miRNA) seen in the public data; every corpus cell is a rat cell with
raw integer counts that passed GeneCompass's own cell filters; each cell is counted once.

How it is checked: manifests with input md5s; a data test that every kept file holds integer counts
(`tests/test_gene_ids.py:155-165`); the 2026-10-31 reproduction (`docs/repro_stages1_7/REPRODUCTION.md:85-114`).

Verdict so far: the vocabulary (22,213 genes) and the QC files are fixed and checksummed, and everything
downstream reproduces from them. Stage 2 itself does not reproduce from raw data (22,485 genes against 22,488;
187 of 559 kept QC files differ). The first corpus contained duplicate files, non-rat samples and 54 files with
log-normalized values; v2 removes these with a per-file plan table, not by a change in the QC code. Two parts of
the "GeneCompass-exact" QC do not do what the documents say (mitochondrial filtering; see Deviations 1-2).

### Inputs and outputs
Inputs:
- Raw matrices under `data/raw/geo/single_cell/geo_datasets` and `data/raw/geo/bulk/geo_datasets` (the same
  folder twice; `data/training/gene_universe/manifest_step1.json` key `inputs.source_dirs`).
- `data/references/biomart/rat_gene_info.tsv`: 43,360 Ensembl 115 genes (22,016 protein-coding, 16,149 lncRNA,
  245 miRNA, others; 24,711 with a symbol; counted here), md5 `aa206a03...`
  (`config/pipeline_config.yaml:44`; same md5 in `manifest_step1.json` key `inputs.gene_info_md5`).
- `data/references/biomart/GENES_RAT.txt` (RGD), `config/pipeline_config.yaml:63`.
- `data/catalog/unified_studies.json` (study selection for step 2).
- `config/stage2_sample_order.tsv`: 1,065 rows in 109 studies pinning `<study>_sample<idx>` to a source path
  (`config/stage2_sample_order.tsv:1-8`; counted here).

Outputs (`data/training/gene_universe/`, `data/training/qc_h5ad/`):
- Step 1: `gene_universe.tsv` (22,488 genes; 20,956 protein-coding, 1,287 lncRNA, 245 miRNA),
  `gene_resolution.tsv` (568,594 distinct raw IDs), `study_coverage.tsv` (20,942 file rows covering 733 GSEs),
  `manifest_step1.json` (keys `outputs`, `pipeline.biotype_distribution`).
- Step 2: QC h5ad files, `expression_stats.tsv`, `preprocessing_report.json` (138 studies; 80 success, 35 with
  no matrix found, 23 with all samples failed; 1,038 matrix files: 810 success, 149 no genes mapped, 77 load
  failed, 2 too few cells; 8,930,695 cells; counted here from `studies[]`).
- Step 3: `pruned_gene_universe.tsv` (22,213 genes, md5 `1caa6bda...`), `removed_genes.tsv` (275),
  `pruning_report.json`, `stage2_manifest.json`.
- OMIX767: `OMIX767_grcr8_sampleC.h5ad` (2,081 cells) and `OMIX767_grcr8_sampleE7W.h5ad` (3,574 cells) with
  `.qc.json` reports (`data/training/qc_h5ad/OMIX767_grcr8_sample*.qc.json` key `cells`).
- The corpus definition for v2: `config/corpus_v2_plan.tsv`, one row per QC file with columns `sample_id`,
  `study`, `gsm`, `action`, `reason`, `n_cells`, `md5` (`config/corpus_v2_plan.tsv:1`).

Composition of the 868 planned files (counted here from `config/corpus_v2_plan.tsv`):

| `action` | Files | Cells | Studies | Meaning |
|---|---|---|---|---|
| `keep` | 557 | 6,539,756 | 75 (74 GEO + OMIX767) | 555 files of the 2026-02-23 run plus the 2 OMIX767 files |
| `keep_raw_slot` | 27 | 263,450 | 6 | Files of the cancelled 2026-02-18 run; counts read from `.raw` |
| `drop_duplicate` | 234 | 2,327,913 | 22 | 233 byte-identical copies of a kept file, 1 same-cells copy in another format |
| `drop_nonrat` | 48 | 357,956 | 4 | GEO sample organism is not rat |
| `drop_superseded` | 2 | 5,848 | 1 | July OMIX767 files with unresolved older gene IDs |
| Total | 868 | 9,494,923 | 89 | |
| Kept | 584 | 6,803,206 | 81 | |

GSM accessions are recorded for 544 of the 584 kept files (538 distinct GSMs); 40 are blank (counted here).

### Procedure
Orchestrator `pipeline/run_stage2.py` runs three scripts in order as subprocesses from the repository root
(`run_stage2.py:78-103`, `:193-208`), after checking that the config sections and keys exist and that
`rat_gene_info.tsv` is present (`:106-156`). `--from`, `--to`, `--dry-run`, `-v` and `--allow-reference-drift`
are supported (`:253-266`).

**Step 1. `build_gene_universe.py`: which genes exist in the data.**
1. Reference check: `rat_gene_info.tsv` and `GENES_RAT.txt` must match `config/reference_md5sums.tsv`, else the
   step exits unless drift is allowed (`build_gene_universe.py:675-681`, `:155-158`; `lib/gene_utils.py:756-790`).
   This check was added 2026-11-01; the canonical run recorded only the md5 of `rat_gene_info.tsv`
   (`docs/repro_stages1_7/REPRODUCTION.md:18-22`, `:126-127`).
2. BioMart gate: every Ensembl ID in `rat_gene_info.tsv` that starts `ENSRNOG` is an allowed gene (uppercased,
   version suffix stripped; `:105-116`). A lower-case symbol map is built in which the first gene in file order
   wins (`:118-122`); 233 lower-case symbols name more than one gene (1,580 genes involved; counted here).
3. RGD synonyms: for each RGD row whose `ENSEMBL_ID` field contains exactly one ID that is in the gate
   (`:209-214`), the `SYMBOL` and every `OLD_SYMBOL` alias (split on `;`, lower-cased) map to that gene, first
   occurrence wins (`:216-226`). RGD never adds a gene or a biotype.
4. File discovery: `os.walk` over each source folder (`:513`); files whose name contains any of 23 metadata
   substrings are skipped (`:300-308`, `:520-521`); the format is decided by extension, and a tabular file is a
   matrix only if its name contains one of 10 matrix words or starts with `gsm` (`:310-314`, `:463-479`). One MTX
   per folder, or one per GSM-style prefix, is kept (`:527-545`).
5. For each file, the gene names are read without loading counts: h5ad `var/_index` and alternatives
   (`:317-329`); the first column of the 10x features/genes file (`:355-365`); 10x H5 `matrix/features/id`, then
   `name`, then `matrix/genes` (`:377-381`); loom row attributes (`:398-418`); first column or header of a table
   (`:421-460`).
6. Resolution chain for each raw ID (`:253-292`): reject empty strings and 11 non-gene patterns (pure digits,
   coordinates, `chr...`, Rfam, GenBank-style, scaffolds, mouse and human Ensembl IDs, array probes;
   `:238-250`); if it starts `ENSRNOG`, strip the version and accept it if it is in the gate, else reject as
   `not_in_biomart` (`:270-275`); other `ENS...` IDs are rejected (`:278-279`); else BioMart symbol,
   case-insensitive (`:281-284`); else RGD synonym (`:286-290`); else `unresolved`.
7. Per-file usability gate: a random sample of 1,000 gene names (all, if fewer) is resolved; the file is usable
   if at least 30% resolve (`:749-767`; `config/pipeline_config.yaml:196-197`). One `random.Random(0)` is shared
   by all files in walk order (`:722`; `config/pipeline_config.yaml:198`).
8. For usable files every name is resolved and the study accession (the first parent folder matching an
   accession pattern, `:565-569`; patterns `config/pipeline_config.yaml:204-207`) is added to that gene's study
   set (`:771-794`).
9. Pruning: keep genes resolved in at least 2 distinct studies (`:850-859`; `config/pipeline_config.yaml:199`),
   then keep biotypes `protein_coding`, `lncrna`, `mirna` (lower-cased BioMart "Gene type"; `:861-873`;
   `config/pipeline_config.yaml:200-203`).
10. Outputs and manifest (`:886-998`).

Canonical result (`data/training/gene_universe/manifest_step1.json` key `pipeline`): 20,942 file rows scanned
(8,616 distinct accession-file pairs; every file appears 2 or 4 times because the two source folders are one
folder; counted here from `study_coverage.tsv`); 733 GSEs with a matrix file, 434 usable; 26,412 genes resolved;
14 removed as seen in fewer than 2 studies; 3,910 removed by biotype; 22,488 kept. Name occurrences by
method: direct Ensembl 97,316,648; `not_in_biomart` 50,781,448; BioMart symbol 22,992,680; RGD synonym
4,135,344; unresolved 8,216,402. Distinct raw IDs by method (counted here from `gene_resolution.tsv`): 44,682
`ENSRNOG` IDs are not in Ensembl 115. The 434 usable series include bulk tables: 18,220 of the 20,942 file
rows are tabular.

**Step 2. `preprocess_training_matrices.py`: cell QC.**

a. Study selection (`load_study_catalog`, `:1241-1318`), from `unified_studies.json`:
   - drop accessions in `reference_assembly.exclude_studies` (GSE242354, the MoTrPAC series;
     `config/pipeline_config.yaml:303-304`; `:1254-1267`);
   - drop studies whose catalog `data_type` is not `single_cell` (`:1269-1272`);
   - drop if the LLM says `is_rat` is `False`; if `is_rat` is not `True`, keep only if the LLM species string
     contains "rattus" or "rat", or a catalog organism contains "rattus" or equals "rat" (`:1274-1289`);
   - drop if the LLM says `is_single_cell` is `False` (`:1291-1294`);
   - drop if the study folder does not exist under the configured single-cell source path (`:1296-1307`).
   On the on-disk catalog: 2,259 not single-cell-typed, 85 LLM not rat, 68 LLM not single-cell, 11 no folder,
   1 excluded, 138 selected (counted here; equals `preprocessing_report.json` key `summary.total_studies`).

b. Matrix discovery per study (`find_matrix_files`, `:707-757`): the study folder is walked, and then every
   `*_RAW` subfolder is walked again (`:711-718`, `:723-724`). MTX files are de-duplicated by folder or prefix
   (`:734-751`); files of every other format are appended on each pass (`:752-753`), so an H5, CSV, TSV, h5ad or
   loom file inside `<GSE>_RAW/` is returned twice. Candidates are ordered h5ad, H5, MTX, loom, TSV, CSV
   (`:755-756`).

c. Sample numbering (`number_matrix_files`, `:1032-1066`): each matrix file becomes one "sample"
   `<study>_sample<idx>`. Studies listed in `config/stage2_sample_order.tsv` keep the canonical IDs, matched by
   path, "occurrence by occurrence" so that a file discovered twice keeps both canonical IDs (`:1050-1062`);
   unlisted files are numbered once each in sorted order (`:1063-1065`).

d. Loading (`:783-795`): h5ad by `scanpy.read_h5ad` (`:434`); MTX by
   `scanpy.read_10x_mtx(dir, var_names='gene_ids', make_unique=True)`, falling back to
   `var_names='gene_symbols'`, then to a manual Matrix Market reader (`:573-584`); H5 by `scanpy.read_10x_h5`
   (`:607`); loom by `scanpy.read_loom(sparse=True, cleanup=True)` (`:627`); tables by `pandas.read_csv`
   with an orientation heuristic (`:645-670`).

e. Raw-count test (`detect_is_raw_counts`, `:361-394`): for a sparse matrix, the first 100,000 stored values
   (`:365`); for a dense matrix, the non-zero values of 500 cells drawn with `numpy.random.default_rng(0)`
   (`:372`). If all are within 1e-6 of an integer the matrix is raw (`:382-385`). If not, an h5ad's `.raw` is
   tested and used when it is integer (`:450-456`). Otherwise the matrix is "reverse-normalized": values are
   exponentiated (`expm1`) when the maximum is below 20, and each cell's values are divided by that cell's
   smallest non-zero value and floored (`:397-418`, `:1122-1131`). In the canonical run this path fired for 40
   file entries in 4 studies (GSE144027, GSE289659, GSE292745, GSE307917) and none of those files passed QC, so
   no reverse-normalized matrix is in the corpus (counted here from `preprocessing_report.json` keys
   `studies[].warnings`, `samples[].status`).

f. Unfiltered 10x matrices: if the barcode count is 6,794,880, 737,280 or 2,097,152, or exceeds 500,000
   (`:127-128`, `:421-423`), barcodes with fewer than `min_genes_per_cell` detected genes are removed first
   (`:874-881`, `:1133-1139`). This applied to 140 of the 555 kept canonical files (counted here, key
   `samples[].cells_after_droplet_filter`).

g. Mitochondrial genes are identified on the original gene names by three regular expressions, `^mt-`,
   `^ENSRNOG\d+.*mt`, `^Mt_`, case-insensitive (`:130-134`, `:802-811`, `:1141-1142`), then carried to the
   resolved IDs (`:266-275`, `:1156-1157`).

h. Gene mapping (`map_and_subset`, `:220-264`): each name is looked up in step 1's `gene_resolution.tsv`,
   exact string first, lower-case second (`:228-230`), restricted to genes in `gene_universe.tsv`
   (`:203-206`); unmapped genes are dropped; names that map to the same gene have their counts summed
   (`:248-251`, `:278-309`). Step 2 contains no resolution logic of its own.

i. Cell filters, in this order (`:1159-1177`):
   1. `normal_filter` (`:814-847`): drop cells with zero total counts (`:821-824`); if fewer than 3 cells
      remain, stop (`:826-827`); compute each cell's total counts and mitochondrial fraction; keep cells whose
      z-score is strictly between -3 and 3 on both (`:839-845`). A z-score that is not a number (constant
      column) is set to 0 (`:841-842`).
   2. `gene_number_filter` (`:850-862`): keep cells with more than 6 detected genes among the universe's
      protein-coding and miRNA genes (`:211-215`, `:860-861`).
   3. `min_genes_filter` (`:865-871`): keep cells with at least 200 detected universe genes
      (`config/pipeline_config.yaml:212`; `:1077`, `:1169`).
   4. Drop the sample if fewer than 4 cells remain (`config/pipeline_config.yaml:213`; `:1078`, `:1173-1177`).
   There is no doublet step, no minimum-counts threshold other than "greater than zero", and no absolute
   mitochondrial-fraction threshold.

j. Per-cell `library_size`, `n_genes_detected`, `mito_pct` are stored in `obs` (`:933-963`); per-gene
   `n_cells_nonzero` and `total_raw_counts` are accumulated (`:888-930`, `:1187-1191`); the file is written with
   counts in `X` and the source path and md5 in `uns['source']` (`:1194-1206`). No normalization is applied
   (`:1182`).

k. Corpus mode loops over studies sequentially and writes `expression_stats.tsv` and
   `preprocessing_report.json` (`:1325-1427`). The canonical run took 81,844 s
   (`preprocessing_report.json` key `elapsed_seconds`).

Cell attrition over the 555 kept files of the canonical run (counted here from `preprocessing_report.json`):
247,474,587 barcodes read (including unfiltered barcode spaces); 6,734,360 after the empty-droplet step and
`normal_filter`; 6,734,323 after `gene_number_filter`; 6,534,101 after `min_genes_filter`. Input formats of
those 555 files: MTX 397, H5 150, CSV 6, TSV 2. Gene-name type as classified from the first 100 names
(`:331-358`): Ensembl rat 368 files (4,605,002 cells), symbol 152 (1,540,028), mixed 35 (389,071).

**Step 3. `prune_gene_universe.py`: drop genes never expressed.** Joins `gene_universe.tsv` with
`expression_stats.tsv` and removes genes absent from the stats, or with `n_cells_nonzero` below
`min_cells_nonzero` (1, `config/pipeline_config.yaml:225`; `prune_gene_universe.py:88`), or with total counts
below `min_total_counts` (not in the config; code default 0, `:89`) (`:153-160`). Canonical result: 275 genes
removed, all with zero cells; 22,213 kept; cells per kept gene: minimum 1, median 210,582
(`data/training/gene_universe/pruning_report.json` keys `summary`, `kept_gene_cell_count_distribution`).

**OMIX767 (external matrices). `qc_external_mtx.py`.** Run once per 10x folder (`README.md:111-114`). Gene
resolution differs from step 1: symbol first, if the symbol names exactly one Ensembl 115 gene
(`qc_external_mtx.py:73-82`); else the source gene's own ID, if it exists in Ensembl 115 and no symbol-resolved
gene already claimed it (`:83-86`); only universe genes are kept (`:87-91`); counts of source genes landing on
one ID are summed (`:101-110`). Mitochondrial genes are found by symbol (`:99`). Then the same four filters
(`:115-119`). Result for both samples: 25,340 source genes; 16,079 resolved by symbol, 2,243 by ID, 6,947
unresolved, 71 not in the universe; 18,187 genes out; 13 mitochondrial genes; 91.5% and 90.8% of counts kept;
cells 2,157 to 2,081 and 3,693 to 3,574 (`data/training/qc_h5ad/OMIX767_grcr8_sample{C,E7W}.qc.json`).

**The v2 corpus plan.** `paths.qc_file_list: config/corpus_v2_plan.tsv` (`config/pipeline_config.yaml:86`)
makes Stages 4 and 5 read only rows whose `action` starts with `keep`, and fail if a kept file is missing
(`lib/gene_utils.py:129-154`; callers `pipeline/04_gene_medians/compute_gene_medians.py:561`,
`pipeline/05_tokenization/tokenize_corpus.py:830`). Counts are taken from `X`, or from `.raw` when the first
20,000 stored values of `X` are not integers; a file with non-integer `X` and no integer `.raw` raises an
error (`lib/gene_utils.py:157-177`; callers `compute_gene_medians.py:260`, `tokenize_corpus.py:444`).

**How raw counts are verified as raw.** Three layers exist:
1. At load, the integer test of step e (first 100,000 stored values).
2. At use, `raw_counts_adata` in Stages 4-5 (first 20,000 stored values), which is the gate that applies to
   every v2 file.
3. A data test, `tests/test_gene_ids.py:155-165`, that applies the 20,000-value test to every kept file. No
   stored result of this test was found.
`pipeline/02_gene_universe/verify_raw_counts.py` is a fourth, older check that inspects only the `.raw` slot
(`verify_raw_counts.py:32-37`); it is not called by the orchestrator. Its last output (2026-10-26) reports 54
files OK, 814 with no `.raw`, "All verified: False"
(`/depot/reese18/data/training/preprocessed/raw_count_verification/verification_summary.txt`). It was written
for the 895-file corpus of early February, where every file had a `.raw` slot
(`.../raw_count_verification/verify_10268617.log`). All these tests sample the head of the value array; none
reads a whole matrix.

**The duplicate-file defect and how v2 handled it.** Counted here from `config/corpus_v2_plan.tsv`:
- 247 of the 868 listed files (equivalently 247 of the 866 that existed before the two new OMIX767 files) have
  the same md5 as another listed file.
- 233 are `drop_duplicate` rows marked "byte-identical copy of <sample_id>": 164 are copies of a kept file in
  the same study (1,545,392 cells) and 69 are copies of a kept file in another series (776,202 cells). The
  cross-series pairs are GSE212417 = GSE212415 (19 files), GSE235317 = GSE235314 (2), GSE239558 = GSE239540 (1),
  GSE251835 = GSE251834 (20), GSE253768 = GSE253767 (16), GSE255196 = GSE254371 (2), GSE297628 = GSE297627 (9):
  seven pairs, consistent with SuperSeries/SubSeries copies, although the plan does not name the relationship.
  The other 14 md5-duplicates are among the non-rat files (14 identical pairs, 28 files, all dropped as
  non-rat).
- One further `drop_duplicate` row, GSE143920_sample1, is "same cells as GSE143920_sample0, other file format".
- Same-study copies arise in step b: for example GSE247625's eight H5 files under `GSE247625_RAW/` are listed
  twice, as samples 0-7 and 8-15, with the same paths (`config/stage2_sample_order.tsv:420-435`).
- All 584 kept files have distinct md5s.
The handling is entirely in the plan table. The QC code still returns such files twice and the numbering code
deliberately reproduces the duplicate IDs for pinned studies (`preprocess_training_matrices.py:1050-1062`), so
a rerun of Stage 2 again writes the duplicate, non-rat and (for the six restored studies) differently stored
files. The script that built the plan (md5 comparison, GEO organism lookup, barcode comparison) is not in the
repository; material sits in the git-ignored `tmp/data_audit/` (`.gitignore:58`). The `reason` column is the
only committed record of each decision.

**Non-rat samples.** 48 files in four series were dropped for a non-rat GEO sample organism: GSE262729 (8, Mus
musculus), GSE303115 (18: cat, rabbit, zebrafish, horse, cow), GSE153116 (2, Homo sapiens), GSE266415 (20,
one or two files each of 19 species) (counted here from the `reason` column). They passed Stage 2 because the
study-level LLM flag was `is_rat = True` and because symbol resolution is case-insensitive, so a matrix keyed
by human or mouse symbols resolves to rat genes with the same spelling (`build_gene_universe.py:120-122`,
`:136-137`; `preprocess_training_matrices.py:228-230`) and clears the 30% gate.

### Models and software
No statistical or machine-learning model. Software (versions from the 2026-10-05 lock, with the caveat given
under Stage 1 that the February environment is not recorded): scanpy 1.12 (`requirements.lock.txt:195`),
anndata 0.12.10 (`:31`), h5py 3.14.0 (`:84`), scipy 1.17.1 (`:198`), numpy 2.4.3 (`:134`), pandas 2.3.3
(`:155`). The canonical manifests record the git commit (`14ab1ba`) and input md5s but no package versions
(`data/training/gene_universe/manifest_step1.json`, `stage2_manifest.json`).

The QC is described as "GeneCompass-exact" (`config/pipeline_config.yaml:210`;
`preprocess_training_matrices.py:31-36`). The vendored upstream script (submodule commit `66ab82f`) contains
`normal_filter` with the same +/-3 z-score rule (`vendor/GeneCompass/preprocess/preprocess.py:80-109`) and
`gene_number_filter` with the same "more than 6" rule (`:112-118`). Differences: upstream identifies
mitochondrial genes from a species list file matched on gene symbols (`:34-36`, `:89-92`), not by regular
expression; upstream applies `normal_filter` after name matching and before the gene-ID subset (`:270-291`),
the rat code after the subset; the 200-gene minimum and the 4-cell sample minimum do not appear in the
vendored script, and where they come from is not cited in the repo.

Checked afterwards against the GeneCompass paper's Methods (Yang et al. 2024; `readings/` PDF page 12). Its pipeline is: "filtering cells with less than 200
expressed genes, filtering samples with less than 4 cells, filtering cells with more than 7 expressed
protein-coding or miRNA genes, filtering cells with a proportion of mitochondrial genes exceeding 15%, filtering
cells with an expressed gene number exceeding three standard deviations from the mean number among all cells in
the gene expression matrix for each sample, and dropping genes not in the core gene list". So:
- the 200-gene and 4-cell minima are the paper's;
- the paper applies an absolute 15% mitochondrial cut, which this code does not (Deviation 1);
- the paper's three-standard-deviation rule is on the number of expressed genes, whereas this code, like the
  vendored script, applies it to total counts and to the mitochondrial fraction.

### Parameters, thresholds and flags
| Name | Value | Where set (path:line) | What it controls | Rationale given |
|---|---|---|---|---|
| `biomart.ensembl_release` / `assembly` | `"115"` / `"GRCr8"` | `config/pipeline_config.yaml:35-36` | Label written to manifests; the fetch host | files are byte-identical to the 115 archive; the old label 113 / mRatBN7.2 "was wrong" (`:31-34`) |
| `biomart.rat_gene_info_md5` | `aa206a030b06c5cfe376c1cd211e5a53` | `config/pipeline_config.yaml:44` | Checked by `BioMartReference.from_config` (`lib/gene_utils.py:646-652`), which only `qc_external_mtx.py:72` calls | "Stage 2/3 refuse a different file" (`:43`) |
| `biomart.reference_md5sums` / `allow_reference_drift` | `config/reference_md5sums.tsv` / false | `config/pipeline_config.yaml:49-50` | Step 1 reference check | the RGD file cannot be re-downloaded (`config/reference_md5sums.tsv:1-6`) |
| `rgd.use_for_symbol_synonyms` | true | `config/pipeline_config.yaml:65` | RGD synonym resolver on | `docs/BUGS.md:6` ("RGD: symbol synonyms ONLY") |
| `rgd.use_for_biotype_fallback`, `rgd.require_biomart_resolution` | false, true | `config/pipeline_config.yaml:64`, `:66` | Not read by Stage 2 code (the second is only validated, `lib/gene_utils.py:931`) | none stated |
| `gene_universe.biomart_gate` | true | `config/pipeline_config.yaml:195` | Asserted (`build_gene_universe.py:658`) | "non-negotiable" (`:658`) |
| `gene_universe.min_biomart_match` | 0.30 | `config/pipeline_config.yaml:196` | Fraction of sampled names that must resolve for a file to count | none stated |
| `gene_universe.match_sample_size` / `match_sample_seed` | 1000 / 0 | `config/pipeline_config.yaml:197-198` | Sample for the match rate | none stated |
| `gene_universe.min_studies` | 2 | `config/pipeline_config.yaml:199` | Singleton rule: a gene must resolve in at least 2 distinct GSEs among all usable series (bulk included) | none stated |
| `gene_universe.keep_biotypes` | protein_coding, lncrna, mirna | `config/pipeline_config.yaml:200-203` | Biotype filter | "GeneCompass paper includes lncRNA in core gene list" (`lib/gene_utils.py:200`, `:897-901`) |
| `gene_universe.accession_patterns` | `^GSE\d+$`, `^E-[A-Z]+-\d+$`, `^E-[A-Z]+-[A-Z]+-\d+$` | `config/pipeline_config.yaml:204-207` | Study ID from path | none stated |
| `gene_universe.fail_on_unrecognized_accession` | true | `config/pipeline_config.yaml:208` | File with no accession is skipped with a warning (`build_gene_universe.py:725-732`) | none stated |
| Non-gene name patterns | 11 regular expressions | `build_gene_universe.py:238-250` (hard-coded) | Names rejected before resolution | none stated |
| Skip / matrix file-name substrings | 23 / 10 | `build_gene_universe.py:300-314`; `preprocess_training_matrices.py:136-150` (hard-coded) | File discovery | none stated |
| `preprocessing.min_genes_per_cell` | 200 | `config/pipeline_config.yaml:212` | Minimum detected genes per cell; also the empty-droplet cut | "GeneCompass-exact" (`:210`) |
| `preprocessing.min_cells_per_sample` | 4 | `config/pipeline_config.yaml:213` | Minimum cells for a file to be written | same |
| z-score bounds | -3 < z < 3 on total counts and on mitochondrial fraction | `preprocess_training_matrices.py:844-845` (hard-coded) | Outlier cells | upstream rule (`vendor/GeneCompass/preprocess/preprocess.py:104-105`) |
| `preprocessing.outlier_sd_threshold` | 3.0 | `config/pipeline_config.yaml:215` | Nothing: not read (only validated, `lib/gene_utils.py:924-927`) | none stated |
| Core-gene rule | more than 6 protein-coding or miRNA genes detected | `preprocess_training_matrices.py:861` (hard-coded) | Cell filter | upstream rule (`vendor/GeneCompass/preprocess/preprocess.py:116`) |
| `preprocessing.min_core_genes_per_cell` | 7 | `config/pipeline_config.yaml:216` | Nothing: not read by any script | none stated |
| `preprocessing.max_mito_fraction` | 0.15 | `config/pipeline_config.yaml:214` | Nothing: not read by Stage 2 (only validated, `lib/gene_utils.py:920-923`) | none stated |
| Mitochondrial gene patterns | `^mt-`, `^ENSRNOG\d+.*mt`, `^Mt_` (case-insensitive) | `preprocess_training_matrices.py:130-134` (hard-coded) | Which genes count as mitochondrial | none stated |
| Unfiltered-10x barcode counts | 6,794,880; 737,280; 2,097,152; or more than 500,000 | `preprocess_training_matrices.py:127-128` (hard-coded) | Triggers the empty-droplet cut | none stated; third value differs from Stage 1's 3,222,016 |
| Raw-count test | first 100,000 stored values; tolerance 1e-6 | `preprocess_training_matrices.py:365`, `:382` (hard-coded) | Raw or normalized | none stated |
| Dense raw-count sample | 500 cells, seed 0 | `preprocess_training_matrices.py:361`, `:372` | same | seeded 2026-11-01 (`:370-371`) |
| Reverse-normalization rule | `expm1` if max < 20; divide by the cell's minimum non-zero; floor | `preprocess_training_matrices.py:405-416` (hard-coded) | Recovering counts | "scFoundation method" (`:398`) |
| Gene-name type sample | first 100 names; label if more than 60% agree | `preprocess_training_matrices.py:331-358` | Reporting only | none stated |
| `preprocessing.save_qc_matrices` | true | `config/pipeline_config.yaml:217` | Write h5ad files | none stated |
| `pruning.min_cells_nonzero` | 1 | `config/pipeline_config.yaml:225` | Gene kept if expressed in at least 1 QC'd cell | "only removes genes with zero expression" (`:220-223`) |
| `min_total_counts` | 0 (code default) | `prune_gene_universe.py:89` | Total-count floor | none stated |
| `paths.stage2_sample_order` | `config/stage2_sample_order.tsv` | `config/pipeline_config.yaml:79` | Sample IDs | walk order is not portable (`:77-78`) |
| `paths.expected_qc_h5ad_files` | 868 | `config/pipeline_config.yaml:76` | Nothing: `validate_qc_file_count` (`lib/gene_utils.py:968`) is never called | none stated |
| `paths.qc_file_list` | `config/corpus_v2_plan.tsv` | `config/pipeline_config.yaml:86` | Which QC files Stages 4-5 read | v2 decisions (`config/profiles/retrain_v2.yaml:8-13`) |
| `reference_assembly.exclude_studies` | GSE242354 | `config/pipeline_config.yaml:303-304` | MoTrPAC series excluded | "data leakage prevention" (`:302`) |
| Raw-slot test in Stages 4-5 | first 20,000 stored values; tolerance 1e-6 | `lib/gene_utils.py:168-170` | `X` or `.raw` | 2026-02-18 files hold log-normalized `X` (`:160-162`) |

### Statistical procedures
No hypothesis test. Three numerical rules deserve a precise statement.
- Outlier filter: per matrix file, z = (x - mean) / SD computed by `scipy.stats.zscore` over the cells of that
  file after zero-count cells are removed, for total counts and for mitochondrial fraction; a cell is kept when
  both z-scores lie strictly inside (-3, 3) (`preprocess_training_matrices.py:839-845`). The SD convention is
  scipy's default [confirmed in the project environment: `scipy.stats.zscore(a, axis=0, ddof=0, nan_policy='propagate')`]. The unit is the file, not the study, so the cut is relative
  to each file's own distribution; in an unfiltered matrix the distribution is the one left after the 200-gene
  empty-droplet cut.
- Match rate: resolved fraction of a simple random sample without replacement of 1,000 names
  (`build_gene_universe.py:750-757`), seed 0, one generator for the whole scan (`:722`), so the sample drawn for
  a file depends on the order in which files are discovered.
- Raw-count test: deterministic on the head of the value array; not a statistical test.
Nothing in Stage 2 is pre-registered. The retrain pre-registration fixes the corpus only by reference
("corpus 6,803,206 cells / 584 samples", `docs/RETRAIN_V2_PREREGISTRATION.md:105-106`).

### Checks, validation and their results
| Check | Result | Pass rule | Where |
|---|---|---|---|
| QC files equal the plan | 868 of 868 match md5 and cell count; kept cells 6,803,206 | exact | `docs/repro_stages1_7/REPRODUCTION.md:49`; `docs/repro_stages1_7/stage1_qc_inputs.tsv` |
| Rerun of step 1 | 22,485 genes against 22,488 (28 only in the rerun, 31 only canonical); 3,968 fewer RGD-synonym resolutions | identical md5 | `docs/repro_stages1_7/REPRODUCTION.md:14-17` |
| Cause of the step 1 difference | February code and HEAD give the same rerun result; 214 of 568,594 raw IDs resolve differently, all through RGD synonyms | | `docs/repro_stages1_7/REPRODUCTION.md:118-124` |
| Rerun of step 2 | 80 studies succeed and 58 fail as before; 8,930,742 cells against 8,930,695; of 812 files compared, 421 identical; of the kept files, 372 identical and 187 differ (1-27 fewer genes; 98 with cell count moved by -8 to +13) | identical md5 | `docs/repro_stages1_7/REPRODUCTION.md:93-100`; `docs/repro_stages1_7/stage2_qc_h5ad.tsv` (370 `keep` identical + 2 OMIX767; counted here) |
| The 27 raw-slot files | not produced by the plain rerun; regenerated after unpacking six series, with the same barcodes and identical counts on shared genes; 8 files have 17,351 genes against 17,352 | counts identical | `docs/repro_stages1_7/REPRODUCTION.md:101-104`, `:131-140`; `scripts/restore_rawslot_studies.sh:1-16` |
| Rerun of step 3 | 22,212 genes against 22,213 | identical | `docs/repro_stages1_7/REPRODUCTION.md:107-109` |
| OMIX767 QC | both h5ad files and both reports byte-identical | identical md5 | `docs/repro_stages1_7/REPRODUCTION.md:105-106`; `docs/repro_stages1_7/stage2_omix_qcjson.tsv` |
| Sample numbering | 1,038 of 1,038 IDs correct in native order, 1,037 with listings reversed | all | `docs/repro_stages1_7/REPRODUCTION.md:161-165`; unit tests `tests/test_stage2_numbering.py:19-39` |
| Integer counts in every kept file | data test exists; stored result NOT FOUND | no failing file | `tests/test_gene_ids.py:155-165` |
| `verify_raw_counts.py` | 54 OK, 814 no `.raw`, "All verified: False" (it checks `.raw` only) | all OK | `/depot/reese18/data/training/preprocessed/raw_count_verification/verification_summary.txt` |
| Resolver and biotype unit tests | present | pytest pass | `tests/test_gene_ids.py:23-78` |

### Deviations, known defects and caveats
1. **No absolute mitochondrial threshold is applied, although config and manuscript state one.** The key
   `max_mito_fraction: 0.15` is read by no Stage 2 script; the manuscript says cell QC used
   "<= 0.15 mitochondrial fraction" (`manuscript/motrpac_genecompass.tex:270`, `:1249`), and
   `deconvolution/joint_reannotate.py:173` repeats it as "the corpus QC". The only mitochondrial rule is the
   within-file z-score.
2. **The mitochondrial z-score is inert for most of the corpus.** MTX matrices are loaded with Ensembl IDs as
   gene names (`preprocess_training_matrices.py:575`), and mitochondrial genes are searched for by symbol
   patterns on those names (`:130-134`, `:1142`). A bare `ENSRNOG...` ID matches none of them, so the
   mitochondrial set is empty, every cell's fraction is 0, the z-score is undefined and is set to 0, and the
   filter passes every cell (`:829-842`). 368 of the 555 kept canonical files (4,605,002 cells, 68% of the v2
   corpus) have Ensembl-ID gene names (`preprocessing_report.json` key `samples[].gene_id_format`, counted
   here). This follows from the code and the report; it was not measured on the files (reading h5ad files is
   barred on the login node). The check is `obs['mito_pct'].max() == 0` in any such file. In the H5 and
   symbol-keyed inputs, and in OMIX767 (13 mitochondrial genes found), the filter is active.
3. **Old-release Ensembl IDs are trusted when they exist in release 115.** Source matrices carry annotations
   from several releases (gene counts of 25,399, 32,883, 32,884 and others among the kept Ensembl-keyed files;
   mapping rate median 0.683, minimum 0.104; counted here). The resolver accepts any `ENSRNOG` ID present in
   release 115 as that gene (`build_gene_universe.py:270-275`) and gives the ID priority over a symbol. The
   project's own later script says such IDs "are NOT trusted on their own" because Ensembl reassigns some
   between releases, with an example, and resolves symbol-first (`qc_external_mtx.py:10-15`); it is used only
   for OMIX767. For the 80 GEO studies the number of genes whose ID now names a different gene is not measured
   anywhere I found. IDs absent from release 115 are dropped (44,682 distinct IDs; 50.8 million name
   occurrences). Only 1,287 of release 115's 16,149 lncRNA genes are in the universe, which is consistent with
   most release-115 lncRNA IDs not existing in the older annotations the matrices were built with (not tested).
4. **The release label.** Code, config and README now say Ensembl 115 / GRCr8 (`config/pipeline_config.yaml:31-39`;
   `README.md:203`; `fetch_biomart_reference_data.py:68-73`). The canonical manifest still says
   `"biomart_release": "113"`, `"biomart_assembly": "mRatBN7.2"` (`manifest_step1.json` key `config`), and the
   manuscript still says release 113, mRatBN7.2 (`manuscript/motrpac_genecompass.tex:273`, `:386`, `:1252`).
   The evidence that the files equal the 115 archive is in the git-ignored
   `tmp/data_audit/ENSEMBL_RELEASE_EVIDENCE.txt`; no committed file holds it. Stale wording also remains in the
   deconvolution part of the config ("current mRatBN7.2 ENSRNOG", `config/pipeline_config.yaml:431`).
5. **The vocabulary was fixed on the uncleaned corpus.** Step 3's "expressed in at least one cell" used the
   2026-02-23 run's 810 files, which include the later-dropped duplicate and non-rat files and exclude the 27
   raw-slot files and OMIX767 (`pruning_report.json` key `timestamp`; `config/corpus_v2_plan.tsv`). Stage 3's
   input md5 equals that file (`stage3_manifest.json` key `input_checksums`). Genes whose only expression came
   from dropped files keep a token. Not quantified here.
6. **The singleton rule counts bulk and array tables.** "At least 2 studies" is evaluated over all 434 usable
   series of the harvest, most of whose files are tables, not over single-cell studies
   (`config/stage2_step1_accessions.txt:1-5`). It removed 14 genes.
7. **Stage 2 does not reproduce from raw data**, because the RGD file read on 2026-02-22 was overwritten two
   days later (`docs/repro_stages1_7/REPRODUCTION.md:118-125`); the QC files and the pruned universe are
   therefore deposited (`docs/repro_stages1_7/REPRODUCTION.md:172-175`).
8. **Duplicates, non-rat samples and log-normalized files are fixed by a table, not by code**, and the table's
   builder is not committed (Procedure, above). The first (April) model was trained on the uncleaned corpus
   (`docs/RETRAIN_V2_PREREGISTRATION.md:22`).
9. **27 kept files come from a cancelled run of different code.** They hold log-normalized `X`
   (`config/corpus_v2_plan.tsv` rows `keep_raw_slot`, `reason`), were made with that run's gene set, and have no
   GSM recorded. Their counts match a regeneration (check above).
10. **Symbol ambiguity.** First-wins resolution for the 233 symbols shared by several genes, case-insensitive
    matching, and RGD old symbols can each attach counts to the wrong gene; counts of names landing on one gene
    are summed (187 of 555 kept files had such merges; `preprocessing_report.json` key
    `mapping_stats.duplicates_resolved`, counted here).
11. **Config comment not true of the code.** "Stage 2/3 refuse a different file (BioMartReference.from_config)"
    (`config/pipeline_config.yaml:43`): Stage 2 step 1 uses its own gate plus the md5 table, and Stage 3 checks
    no md5.
12. A "sample" is a matrix file, not a GSM: two GSMs of GSE247625 supply four kept files each
    (`config/stage2_sample_order.tsv:420-427`), and 13 kept files are series-level matrices.
13. The docstring of step 2 lists `cell_stats.tsv` as an output (`preprocess_training_matrices.py:43`); no code
    writes it. `verify_symbol_lookup.py` tests `rat_symbol_lookup.pickle`, a file the resolver does not use,
    and still has the falsy-index bug listed as BUG-10 (`verify_symbol_lookup.py:155`; `docs/BUGS.md:33`).
14. Operational: GSM-prefixed MTX triplets are loaded through `tempfile.mkdtemp`, which writes symlinks under
    the system temp directory (`preprocess_training_matrices.py:563-570`).

### Methods references
- GeneCompass preprocessing rules: Yang X, Liu G, Feng G, et al. GeneCompass: deciphering universal gene
  regulatory mechanisms with a knowledge-informed cross-species foundation model. Cell Research 2024;34:830-845
  (as in `manuscript/motrpac_genecompass.tex:1487-1489`); code `vendor/GeneCompass/preprocess/preprocess.py`.
- Reverse normalization "scFoundation method" (`preprocess_training_matrices.py:398`): Hao M, Gong J, Zeng X,
  et al. Large-scale foundation model on single-cell transcriptomics. Nat Methods 2024;21:1481-1491 (as in
  `manuscript/motrpac_genecompass.tex:1484-1485`). Whether that paper defines this exact procedure: [not verified in this audit].
- Scanpy: no citation in the repo. [verified against Crossref; see REFERENCES.md] Wolf FA, Angerer P, Theis FJ. Genome Biology 2018;19:15.
- AnnData, SciPy (`zscore`), h5py: no citation in the repo. [anndata and SciPy verified, see REFERENCES.md; h5py has no paper]
- Ensembl 115 gene set, RGD synonyms: no citation in the repo. [suggested: Dyer et al. 2025 and Vedi et al. 2023, verified; see REFERENCES.md]

---

## Stage 3 - Ortholog mapping and token assignment

### Research role
Stage 3 answers: which token does each rat gene get, so that the rat model can reuse what GeneCompass learned
about the orthologous human or mouse gene? The claims that rest on it: 17,496 rat genes share a pretrained
human or mouse token through one-to-one orthology; 4,717 get new tokens; no two rat genes share a token; and
the rat-to-human map (15,234 pairs) is the bridge used later to re-express rat cells as human (Stage 12;
`manuscript/motrpac_genecompass.tex:919`).

How it is checked: the tier counts are in `mapping_statistics.json`; the 2026-10-31 rerun reproduced every
output byte for byte (`docs/repro_stages1_7/REPRODUCTION.md:50`). Whether the mapping is biologically right is
tested only downstream, by the retrain criterion C4 on gene embeddings
(`docs/RETRAIN_V2_PREREGISTRATION.md:64-72`), which is outside this section.

Verdict so far: deterministic and reproducible. The orthology calls are taken from Ensembl as given, with no
confidence or identity filter, and the "confidence" label this stage writes is not used downstream.

### Inputs and outputs
Inputs (md5s in `data/training/ortholog_mappings/stage3_manifest.json` key `input_checksums`):
- `data/training/gene_universe/pruned_gene_universe.tsv`, 22,213 genes, md5 `1caa6bda...`.
- `data/references/biomart/rat_human_orthologs.tsv`, md5 `88d44d06...`: 47,126 rows; homology type one2one
  16,426, one2many 3,421, many2many 4,018, empty 23,261; 20,099 of 43,360 rat genes have a human ortholog row
  (counted here).
- `data/references/biomart/rat_mouse_orthologs.tsv`, md5 `1e141550...`: 63,013 rows; one2one 18,742, one2many
  3,154, many2many 20,537, empty 20,580; 22,780 rat genes with a mouse ortholog row (counted here).
- `vendor/GeneCompass/prior_knowledge/human_mouse_tokens.pickle`, md5 `95e86206...`: 50,558 entries (`<pad>` = 0,
  `<mask>` = 1, 23,113 human genes at 2-23,114, 27,443 mouse genes at 23,115-50,557;
  `finetune/genecompass/build_rat_token_dictionary.py:8-12`, `:68-71`).
- `vendor/GeneCompass/prior_knowledge/homologous_hm_token.pickle`, md5 `da950ff9...`: GeneCompass's mouse-token to
  human-token map (`build_ortholog_mapping.py:224-229`).

Outputs (`data/training/ortholog_mappings/`, written 2026-04-01, folder now read-only):
- `rat_token_mapping.tsv`: 22,213 rows with tier, token, orthologs, orthology types, identity, collision count,
  confidence (`build_ortholog_mapping.py:764-776`).
- `rat_tokens.pickle` (22,213), `rat_to_human_mapping.pickle` (15,234), `rat_to_mouse_mapping.pickle` (17,839),
  `new_rat_tokens.txt` (4,717), `tier_diagnostics.json`, `collision_report.tsv` (no collisions),
  `mapping_statistics.json`, `stage3_manifest.json` (key `outputs`).
- `rat_human_mouse_tokens.pickle`: the merged dictionary, 55,275 entries, written by
  `finetune/genecompass/build_rat_token_dictionary.py` (`README.md:122-123`).
- The folder also holds four files dated 2026-02-09 that the current code does not write (`rat_gene_median.pickle`,
  `rat_gene_median_stats.json`, `rat_gene_median_summary.txt`, `rat_mapping_tiers.pickle`;
  `docs/repro_stages1_7/stage3_ortholog_mappings.tsv:12-15`).

Final counts (`data/training/ortholog_mappings/mapping_statistics.json`; token ranges counted here from
`rat_token_mapping.tsv`):

| Tier | Rule (first match wins) | Genes | % | Token given | Token ID range |
|---|---|---|---|---|---|
| T1 tri-species | one2one human ortholog and one2one mouse ortholog, both in the GC vocabulary and linked to each other in GC's homolog map | 13,178 | 59.33 | the human gene's existing token | 3-23,114 |
| T2a human one2one | one2one human ortholog in the GC vocabulary | 705 | 3.17 | the human gene's existing token | 14-23,104 |
| T2b mouse one2one | one2one mouse ortholog in the GC vocabulary | 3,613 | 16.27 | the mouse gene's existing token | 23,116-50,551 |
| T3a human multi | one2many or many2many human ortholog in the GC vocabulary | 1,351 | 6.08 | a new token; best-identity human ortholog recorded | within 50,558-55,274 |
| T3b mouse multi | one2many or many2many mouse ortholog in the GC vocabulary | 892 | 4.02 | a new token; best-identity mouse ortholog recorded | within 50,558-55,274 |
| T4 new | none of the above; biotype protein-coding, lncRNA or miRNA | 2,474 | 11.14 | a new token | within 50,558-55,274 |
| excluded | none of the above and other biotype | 0 | 0 | none | |

By biotype: T1 and T2b are all protein-coding; T2a 589 protein-coding and 116 miRNA; T3a 1,347 and 4; T4 1,108
protein-coding, 1,275 lncRNA, 91 miRNA; no lncRNA is in T1-T3 (`tier_diagnostics.json` key
`biotype_tier_crosstab`). All 22,213 tokens are distinct. The 4,717 new IDs are contiguous from 50,558 to 55,274
and increase with the sorted rat gene ID, with T3a, T3b and T4 interleaved (counted here).

### Procedure
Orchestrator `pipeline/run_stage3.py` validates config keys (`run_stage3.py:76-114`) and input files
(`:117-175`), runs `pipeline/03_ortholog_mapping/build_ortholog_mapping.py` as a subprocess (`:178-208`) and
prints the statistics (`:211-268`). The worker:

1. Loads the pruned universe (`build_ortholog_mapping.py:94-110`).
2. Loads each BioMart table into a list of candidates per rat gene: target gene, orthology type, percent
   identity (`:113-165`). Rows with an empty target are skipped (`:136-138`). Column names are meant to be
   auto-detected (`:168-195`); for the files on disk the detection fails to find the target column and the code
   falls back to column positions, with a warning (`:197-206`; confirmed here by calling the function on both
   headers). The positions give: rat gene = column 1, target gene = column 2, orthology type = column 4,
   percent identity = column 5, "%id. target [Human/Mouse] gene identical to query gene"
   (`data/references/biomart/rat_human_orthologs.tsv:1`; `rat_mouse_orthologs.tsv:1`). The sixth human column
   (query identical to target) is not used.
3. Builds the GC indices: human genes and mouse genes to token, and the set of (mouse gene, human gene) pairs
   linked in GC's homolog map (`:236-273`).
4. Assigns tiers gene by gene in sorted order of rat gene ID (`:306`). Candidates are annotated with "in GC
   vocabulary" and with one2one or multi according to `orthologs.one2one_types` and `orthologs.multi_types`
   (`:293-295`, `:456-474`).
   - T1 (`:334-360`): the first pair (human one2one in vocabulary, mouse one2one in vocabulary) that is linked
     in GC; token = the human token.
   - T2a (`:362-380`): a human one2one in vocabulary; the highest-identity one if several (`:364`); token =
     human token. A mouse one2one, if present, is recorded but not used.
   - T2b (`:382-394`): a mouse one2one in vocabulary; token = mouse token.
   - T3a (`:396-410`): any human non-one2one candidate in vocabulary; the highest-identity one is recorded as
     the ortholog (`:399`, `:477-479`); token = the next new ID (`:402`, `:407`).
   - T3b (`:412-426`): same with mouse.
   - T4 (`:428-437`): biotype in `orthologs.new_token_biotypes`, compared case-insensitively (`:429`); token =
     the next new ID.
   - else `excluded` (`:439-442`).
   The new-ID counter starts at the largest existing GC token ID plus 1 (`:298-301`), which is 50,558.
5. Diagnostics (`:486-533`): token collision counts over T1/T2 only (`:494-499`), an "expansion direction"
   label, and a confidence label per gene (`:536-592`): high = T1/T2, protein-coding, no collision, identity at
   least 70; medium = T1/T2 otherwise, or T3 with identity at least 40 and collision at most 2; low = T4, or T3
   with identity below 40, or non-coding T3.
6. Writes the outputs and a manifest with input md5s and the git commit (`:751-913`, `:1021-1029`).

Token dictionary (`finetune/genecompass/build_rat_token_dictionary.py`): loads GC's 50,558-entry dictionary,
asserts its size and the two special tokens (`:68-71`), takes every rat entry whose token ID is at least 50,558
(`:79`), asserts no key collision and that these IDs are contiguous from 50,558 (`:84-94`), and merges
(`:97-102`). Rat genes in T1/T2 are not added as keys, because they reuse human or mouse tokens (`:19-21`). The
script's text calls all new entries "T4" and gives the old count 2,474 and range 50,558-53,031 (`:14-15`); with
the current Stage 3 the new entries are T3a + T3b + T4 = 4,717.

How T3 tokens are initialized (the ortholog's embedding row plus noise) and how T4 rows are drawn belongs to
Stage 7; the reproduction notes those draws were unseeded in v2's init
(`docs/repro_stages1_7/REPRODUCTION.md:52`).

### Models and software
No model. Pure Python standard library (`csv`, `pickle`, `json`, `hashlib`; `build_ortholog_mapping.py:67-78`).
Orthology source: Ensembl Compara calls as exported by BioMart from the Ensembl 115 archive, attributes
`*_homolog_ensembl_gene`, `*_homolog_associated_gene_name`, `*_homolog_orthology_type`, `*_homolog_perc_id`, and
for human also `*_homolog_perc_id_r1` (`pipeline/01_data_harvesting/fetch_biomart_reference_data.py:205-228`).
GeneCompass vocabulary and homolog map: submodule `vendor/GeneCompass` at commit `66ab82f` (`git submodule
status`), files identified by md5 in `docs/repro_stages1_7/reference_input_md5.txt:6-7`.

### Parameters, thresholds and flags
| Name | Value | Where set (path:line) | What it controls | Rationale given |
|---|---|---|---|---|
| `orthologs.one2one_types` | `ortholog_one2one` | `config/pipeline_config.yaml:248-249` | Which homology type counts as one-to-one (T1, T2) | none stated |
| `orthologs.multi_types` | `ortholog_one2many`, `ortholog_many2many` | `config/pipeline_config.yaml:251-253` | Read and annotated (`build_ortholog_mapping.py:295`, `:472`) but T3 is decided by "not one2one" (`:397`, `:413`), so this list does not gate anything | none stated |
| `orthologs.identity_threshold` | null | `config/pipeline_config.yaml:257` | Nothing: "retained for documentation only; the script ignores it" (`:255-256`) | "GeneCompass empirically used none (min observed: 1.2%)" (`config/pipeline_config.yaml:233-235`, `:244-245`); the computation behind 1.2%, 89.6%, 7.9%, 2.4% was NOT FOUND (searched `*.py`, `*.md`, `*.yaml`, `*.ipynb`) |
| `orthologs.disambiguation` | `highest_identity` | `config/pipeline_config.yaml:260` | Only printed (`build_ortholog_mapping.py:974`); the rule is hard-coded as the maximum of percent identity (`:477-479`), missing identity treated as 0 | none stated |
| `orthologs.new_token_biotypes` | protein_coding, lncRNA, miRNA | `config/pipeline_config.yaml:263-266` | T4 eligibility | none stated |
| Tier order | T1, T2a, T2b, T3a, T3b, T4 | `build_ortholog_mapping.py:334-437` (hard-coded) | Priority | human preferred over mouse because GC pretraining is "heavily human-weighted"; one2one preferred as "unambiguous" (`:32-33`) |
| Orthology confidence filter | none; attribute not fetched | `fetch_biomart_reference_data.py:205-228` | | "Ensembl tree-based orthology IS the quality gate" (`run_stage3.py:27-28`) |
| First new token ID | max existing GC token + 1 = 50,558 | `build_ortholog_mapping.py:298-301` | New-token numbering | none stated |
| Confidence cut-offs | identity 70 (high), 40 (medium/low), collision 1 and 2 | `build_ortholog_mapping.py:565-587` (hard-coded) | The `confidence` column | "Heuristic from implementation plan §6" (`:539`); that plan was NOT FOUND in the repo |
| `paths.genecompass_tokens` / `genecompass_homologs` | `vendor/GeneCompass/prior_knowledge/human_mouse_tokens.pickle` / `homologous_hm_token.pickle` | `config/pipeline_config.yaml:98-99` | GC vocabulary and homolog map | none stated |
| `ORIGINAL_VOCAB_SIZE` | 50,558 | `finetune/genecompass/build_rat_token_dictionary.py:48` | Assertion and split point for new tokens | none stated |

### Statistical procedures
None. Stage 3 is a deterministic rule-based assignment. The identity values are reported as distributions per
tier (minimum, quartiles, mean; `build_ortholog_mapping.py:630-649`): for example T1 human identity median 83.7
(minimum 15.7), T2a minimum 3.3, T3a median 73.4 (minimum 5.9) (`tier_diagnostics.json` key `tier_statistics`).
No uncertainty is attached to an ortholog call. Nothing in this stage is pre-registered; the retrain
pre-registration uses its outputs (the 2,243 T3 and 255 symbol-matched T4 genes,
`docs/RETRAIN_V2_PREREGISTRATION.md:65-72`).

### Checks, validation and their results
| Check | Result | Pass rule | Where |
|---|---|---|---|
| Rerun of Stage 3 plus the token dictionary from the canonical pruned universe | all 10 outputs byte-identical; the two JSON manifests differ only in timestamp and path | identical md5 | `docs/repro_stages1_7/REPRODUCTION.md:50`; `docs/repro_stages1_7/stage3_ortholog_mappings.tsv:2-11` |
| Token collisions | 0 colliding tokens; maximum 1 rat gene per token | reported | `mapping_statistics.json` key `token_collisions`; `collision_report.tsv:1` |
| Every gene mapped | 22,213 of 22,213, 0 excluded | reported | `mapping_statistics.json` keys `total_mapped`, `total_excluded` |
| Dictionary integrity | assertions on size, special tokens, key collisions, contiguity | assertion | `finetune/genecompass/build_rat_token_dictionary.py:68-71`, `:84-102` |
| Confidence distribution | high 14,277; medium 5,283; low 2,653 | reported | `mapping_statistics.json` key `confidence_distribution` |
| Input reference md5s equal the pinned table | `rat_human_orthologs.tsv` and `rat_mouse_orthologs.tsv` md5s in the manifest equal `config/reference_md5sums.tsv:25-26` | compared here | `stage3_manifest.json` key `input_checksums` |
| Unit tests of the tier logic | NOT FOUND | | looked in `tests/` (`test_gene_ids.py` covers ID and biotype helpers only) |

### Deviations, known defects and caveats
1. **The manifest's git commit does not identify the code that made the outputs.** `stage3_manifest.json`
   records `git_hash b9e1d76` (2026-03-29). At that commit T3 genes were given their ortholog's existing token;
   the change that gives T3 genes new tokens was committed on 2026-04-09 as `10bac05` (`git diff b9e1d76 10bac05
   -- pipeline/03_ortholog_mapping/build_ortholog_mapping.py`). The outputs dated 2026-04-01 have 4,717 new
   tokens, so they were made from an uncommitted working tree. `_git_hash` records only `HEAD`, with no
   dirty-tree flag (`build_ortholog_mapping.py:929-937`). The byte-identical rerun at HEAD shows the current
   code is the code that made them.
2. **"T4 = no ortholog" is not exact.** Of the 2,474 T4 genes, 2,237 have no human or mouse ortholog row in
   the Ensembl 115 tables and 237 have one (128 one2one among them) whose gene is not in the GeneCompass
   vocabulary (counted here from `rat_token_mapping.tsv` and the two ortholog tables). The orchestrator's help
   text says "no ortholog" (`run_stage3.py:282`).
3. **Orthology calls are unfiltered.** No Ensembl orthology-confidence flag, no identity floor; one2one calls
   with identity as low as 3.3% (T2a human) and 3.4% (T1 mouse) are accepted (`tier_diagnostics.json`). The
   stated justification rests on figures for GeneCompass's own map whose derivation is not in the repo
   (Parameters table).
4. **Column mapping depends on a positional fallback.** The name-based detection does not recognize BioMart's
   headers, so a re-export with a different column order would silently change which column is read as identity
   or orthology type (`build_ortholog_mapping.py:197-206`). The current files are pinned by md5, which contains
   the risk.
5. **Stage 3 does not verify its reference files.** It records md5s but does not compare them with
   `config/reference_md5sums.tsv` (contrast `build_gene_universe.py:675-681`); the config comment at
   `config/pipeline_config.yaml:43` says it does.
6. **The `confidence` and `expansion_direction` columns are not used to gate anything.** The docstrings say
   they flag GRN edges and guard translation claims (`run_stage3.py:44-45`;
   `build_ortholog_mapping.py:907-908`); the only reader found is
   `finetune/genecompass/build_rat_tf_list.py:70`, which copies the column (searched `finetune/`,
   `deconvolution/`, `translation/`, `analysis/`, `pipeline/04-06`).
7. **"Flagged non-coding" is always empty by a case mismatch**: the test compares the biotype with `'lncRNA'`
   and `'miRNA'` (`build_ortholog_mapping.py:680`) while the universe stores `lncrna` and `mirna`; 120 miRNA
   genes are in T2a/T3a but `flagged_noncoding_in_tiers_1_3.count` is 0 (`tier_diagnostics.json`).
8. **Stale text.** The worker's header says genes with orthologs "inherit pre-trained token embeddings"
   (`build_ortholog_mapping.py:17-19`), which is no longer true of T3 at the token level; the token-dictionary
   script describes the new block as T4 only (`build_rat_token_dictionary.py:14-21`); `docs/BUGS.md` lists
   Stage 3 bugs (BUG-02, -09, -15, -16, -17; `docs/BUGS.md:20`, `:32`, `:38-40`) against a script design (outer
   join, RGD biotype columns) that the current code does not have.
9. **Consequences of Stage 2 carry over.** The tiering is only as good as the gene IDs: a rat gene whose counts
   were attached to a reassigned Ensembl ID (Stage 2, caveat 3) inherits the token of that ID's ortholog.
10. The vocabulary is frozen: new token IDs follow the sorted gene list, so any change to the pruned universe
    renumbers them. The reproduction therefore reran Stage 3 from the canonical universe, not from the rebuilt
    one (`docs/repro_stages1_7/REPRODUCTION.md:42-43`).

### Methods references
- Tiered reuse of human/mouse tokens through homologs: GeneCompass, Yang X, Liu G, Feng G, et al. Cell Research
  2024;34:830-845 (as in `manuscript/motrpac_genecompass.tex:1487-1489`). The tier scheme itself (T1-T4) is this
  project's design (`build_ortholog_mapping.py:21-36`).
- Ensembl Compara orthology types (`ortholog_one2one`, `ortholog_one2many`, `ortholog_many2many`) and percent
  identity, Ensembl release 115: no citation in the repo. [suggested: Dyer et al. 2025, verified; see REFERENCES.md]
- BioMart: no citation in the repo. [suggested: Kinsella et al. 2011, verified; see REFERENCES.md]

---

## What does and does not reproduce byte for byte (Stages 1-3)

| Item | Result | Source |
|---|---|---|
| Raw downloads | not re-tested; GEO files can change after the harvest | `README.md:95-97`; `docs/repro_stages1_7/REPRODUCTION.md:40-49` |
| Study catalog and LLM output | cannot be regenerated (LLM; January layout); deposited | `README.md:101-103`; Stage 1, Deviations 1-2 |
| Reference files | BioMart files equal the Ensembl 115 archive (evidence not committed); RGD file cannot be re-downloaded; all pinned by md5 | `config/reference_md5sums.tsv:1-7` |
| QC h5ad files on disk against the plan | 868 of 868 identical | `docs/repro_stages1_7/REPRODUCTION.md:49` |
| Stage 2 step 1 from raw | differs: 22,485 against 22,488 genes | `docs/repro_stages1_7/REPRODUCTION.md:14-17` |
| Stage 2 step 2 from raw | 372 of 559 kept files identical, 187 differ; the 27 raw-slot files regenerate with identical counts but different storage and md5 | `docs/repro_stages1_7/REPRODUCTION.md:94-104`, `:149-150` |
| Stage 2 step 3 from the rerun | differs: 22,212 against 22,213 | `docs/repro_stages1_7/REPRODUCTION.md:107-109` |
| OMIX767 QC | identical | `docs/repro_stages1_7/REPRODUCTION.md:105-106` |
| Stage 3 and the token dictionary from the canonical universe | identical | `docs/repro_stages1_7/REPRODUCTION.md:50` |

## Items that could not be determined
- The command line, filters and exact script version of the January LLM run, and why Sonnet 4.5 was used for
  GEO single-cell studies and Haiku 4.5 for the rest.
- Package versions in force for the January-April canonical runs.
- Whether `mito_pct` is in fact zero in the Ensembl-keyed QC files (inferred from code, not measured).
- How many cells the 79 excluded rat single-cell series would have added.
- How many corpus genes carry counts from a reassigned Ensembl ID.
- How Entrez grouped the un-parenthesized `OR ... AND` query.
- The source of the "min observed 1.2%" identity figure. (The 200-gene and 4-cell minima were traced afterwards
  to the GeneCompass paper's Methods; see Stage 2.)
- The script and exact rules that produced `config/corpus_v2_plan.tsv` (only the `reason` column is committed).
