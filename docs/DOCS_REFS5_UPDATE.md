# Living documents, notebooks and the preprint brought to refs5 (2026-11-15)

refs5 is the registered rerun of Stages 8-14 that followed the 2026-11-11 audit. It became the default
configuration on 2026-11-15. This page records what was changed so that every living description of the
project says what the refs5 results say, including where they are negative.

- **Source of truth.** `docs/audit/remediation/STEP3_RESULTS.md` (the rerun), `STEP4_RESULTS.md` (the
  validations), `STEP5_RESULTS.md` (corpus and model), `STEP2_STATEMENTS.md`, `STEP6_HYGIENE.md`, and
  `docs/audit/FINDINGS.md`. Nothing under `docs/audit/remediation/` was edited.
- **What was not done.** No analysis was rerun and no pipeline code was changed. Dated documents were not
  rewritten; they got a banner.
- **Who.** An AI coding assistant (Claude), on the branch `audit-remediation-docs`. Nothing was pushed.
- **Start with** section 5, "For the PI's eye", if you read only one part.

## 1. Changes of substance

"Source" names the section of the results files, or the record table, that the new text comes from.
`STEP3`, `STEP4`, `STEP5` are the results files in `docs/audit/remediation/`.

| Document | What it said | What it says now | Source |
|---|---|---|---|
| `README.md` | No result at all; described the project by its aims | Opens with the research question and, per question, the refs5 result and what qualifies it | STEP3 top table; STEP4 top table; STEP5 top table |
| `README.md` | The reference QC gate "blocks spatial/engineered/developmental/mixed samples" | It blocks spatial, engineered or cultured, and bulk samples; it only warns on developmental ages and mixed depth; it cannot test BLOOD, SPLEEN, OVARY | `STEP2_STATEMENTS.md` (finding 4.4) |
| `README.md` | "Which references": `references_v4/`, every output in a `_refs4all_rebuild` folder | The config keys and their refs5 values (`references_v5/`, `rat_exclude_genes_v2.tsv`, target sum 6,700, `_refs5` folders); the earlier default is the profile `refs4all_rebuild.yaml` | STEP3 A1, A3, B3, B9 |
| `README.md` | Orchestrators `run_stage1..14.py (no Stage 11)` | Stage 11 is the gate | `STEP2_STATEMENTS.md` (finding 6.10) |
| `pipeline/README.md` | A results table marked "before the refs5 rerun": 9 hotspots of 151, 53,084 pairs, 9 of 9 transfer, 6 GRNs, 9 of 9 RIN/globin ROBUST | One row per stage: question, refs5 result, what qualifies it. 11 hotspots and 1 borderline of 150; 52,961 pairs; 10 of 11 and 9 of 11; 7 networks; no block robust to technical covariates | STEP3 B1-B8; STEP4 V1-V6; STEP5 M1-M3 |
| `pipeline/README.md` | "Single-sex tissues (TESTES, OVARY) cannot have any" dose-significant gene | Single-sex tissues have a dose test: TESTES 219 pairs, OVARY 8 | STEP3 B5 |
| `pipeline/README.md` | Stage 14 "RIN/globin DE robustness" as the robustness result | Stage 14's check refits a pooled-sex slope, a different model from the headline; the registered check (Stage 10) finds 0 of 150 blocks robust | STEP3 B5 |
| `deconvolution/README.md` | "Current default (2026-11-01): v2 + refs4", folder names, 9 hotspots, 53,084 pairs, "RIN/globin-robust", single-sex zero "is structural" | The question, the chain, and the refs5 result split into what the deconvolution supports and what it does not; folders named by config key | STEP3 B1-B8; STEP4 V1, V5, V6; STEP5 M1 |
| `deconvolution/README.md` | The gate scripts listed under "Not in the pipeline"; `run_deconv_all.py` "for all 14 tissues"; reference paths `references_v4/` | The gate is Stage 11 in the stage table; 16 tissues; `references_v5/` | `pipeline/run_stage11.py`; STEP3 A1 |
| `docs/OPEN_FRONTIER.md` | "Last updated 2026-11-11 ... working references are refs4"; canonical figures 9,232 / 197 / 151 / 9 / 53,084 / 9 of 9 / 6 GRNs / 0.310 | A verdict per aim at the top; canonical figures 9,060 embedded / 194 / 150 / 11 + 1 borderline / 52,961 / 10 of 11 and 9 of 11 / 7 networks / 0.48 | STEP3 top table, B3-B7 |
| `docs/OPEN_FRONTIER.md` | Row 2a: positive controls "Tier A 6/45 ... B 7/22" with the data-anchored check beside it | The gate and the DE are separate rows. DE: no block robust to technical covariates; the red flag stands (5 of 26 testable); the data-anchored check does not lift it | STEP3 B5; STEP4 V2 |
| `docs/OPEN_FRONTIER.md` | Row 3a: "9 of 9 hotspots meet the registered rule ... 7 of 9" | 10 of 11 registered, 9 of 11 corrected, LUNG vascular smooth muscle cells lose the dose trend; not a replication | STEP3 B6 |
| `docs/OPEN_FRONTIER.md` | E.3 "PASS" with no qualification | PASS on the default before refs5; not repeated on refs5 | no refs5 atlas output exists (`data/deconvolution/atlas_tabula_sapiens/` holds `score` and `score_refs4all_rebuild` only) |
| `docs/OPEN_FRONTIER.md` | Module A "DONE ... v2 3/4, #2 FAILS" | v2 on refs5: criteria 1, 3, 4 pass; the myeloid replication fails; criterion 2 withdrawn; what Module A does not establish | STEP4 V4 |
| `docs/OPEN_FRONTIER.md` | Networks: 6, "0 of 723 TFs" | 7 networks, 1,540,997 edges on shared genes, pooled estimate 1.02, 0 of 777 TF x network pairs | STEP3 B7 |
| `docs/OPEN_FRONTIER.md` | Purity sweeps ✅ "meet the Chu-2022 bar on all 12 tissues swept" | ◐: by the registered rule 12 of 15 dominant types pass, 2 fail; called types 4 of 12 validated at their abundance | STEP4 V1 |
| `docs/OPEN_FRONTIER.md` | Technical covariates ✅ "9 of 9 hotspots ROBUST" | ◐: the registered check finds 0 of 150 blocks robust (44% and 33% retained); the older check tests a different model | STEP3 B5 |
| `docs/OPEN_FRONTIER.md` | "Open: a single-sex DE fix ... needs its own registration" | Repaired in refs5 under its registration | STEP3 A4, B5 |
| `docs/OPEN_FRONTIER.md` | "CIBERSORTx: license-gated, Stanford registration pending (the only hard external blocker)" | It ran as one of the six methods in the panels and batteries | STEP4 V5, V6; `record/step4/v6_mixture_panel/panel_median.tsv` |
| `docs/OPEN_FRONTIER.md` | (absent) | Rows for the held-out test and the corpus findings; a list of what the results files leave open for the PI | STEP4 V3; STEP5; STEP3 B1, B2, B4; `STEP6_HYGIENE.md` section 4 |
| `deconvolution/DECONVOLUTION_PIPELINE_REPORT.md` | "Current figures (all 16 tissues, 2026-11-01)" as the authoritative section | A new section "Current figures (refs5, 2026-11-15)" in front of it, with the earlier default as a comparison column; the 2026-11-01 section retitled | STEP3 A3, B1-B8; STEP4 V1, V2, V5, V6 |
| `finetune/genecompass/README.md` | No statement of what was tested | The stage's question, the in-sample check, the held-out test (1.046) and the measured corpus findings | STEP4 V3; STEP5 M1-M3 |
| `config/profiles/refs5.yaml` (comment) | "the current default (`_refs4all_rebuild`) is untouched" | "the default at the time ..., now `config/profiles/refs4all_rebuild.yaml`" | STEP3 B9 |
| `config/pipeline_config.yaml` (comments) | Step 6 listed two stale comments (hotspot rule; "v2 6,400") | Already correct at `59c6121`; not edited | - |
| `notebooks/RERUN_EDITS.md` | Banner pointed to the cells Step 6 listed as stale | A refs5 entry: what prose and code cells changed, the jobs, and how to execute from a worktree | this page, section 3 |

## 2. Documents given a dated banner

Each banner says which run the document describes and points to `docs/audit/remediation/STEP3_RESULTS.md`
(or `STEP4_RESULTS.md`). Nothing below a banner was changed.

| Document | Run it describes | Extra fact stated in the banner |
|---|---|---|
| `deconvolution/AIM2_DECONV_RESULTS.md` | April model, June/July references; its inner banner the default of 2026-11-01 | refs5 headline, with the three negative results |
| `deconvolution/AIM2_DE_MOTRPAC_COMPARISON.md` | June references; §0a the default of 2026-11-01 | the comparison with MoTrPAC has not been redone on refs5 |
| `deconvolution/ALL_TISSUE_RERUN_RESULTS.md` | 16 tissues on refs4 (2026-11-01) | 7 of its 9 hotspots remain |
| `deconvolution/REFS4ALL_REBUILD_RESULTS.md` | clean rebuild of that default (2026-11-02) | refs5's own reproduction record is STEP3 A1, B1 |
| `deconvolution/REFS4_RERUN_RESULTS.md` | 14-tissue refs4 (2026-10-29) | - |
| `deconvolution/TISSUE_REFERENCE_AND_PROPORTIONS.md` | references before 2026-07-16; its banner refs4 | references_v5; fractions shift by at most 2.8 points |
| `deconvolution/OMNIDECONV_RESULTS.md` | June/July references | fractions are mRNA shares (V5); corrector rule reproduced on C minus A only |
| `deconvolution/MIXTURE_PANEL_REFS4ALL_RESULTS.md` | panel of 2026-11-08 on refs4 | refs5 panel: 19 of 19 |
| `deconvolution/SIMBU_BIAS_REFS4ALL_RESULTS.md` | battery of 2026-11-11, whose baseline kept native bias | refs5 battery: 13 of 15, 14 of 15 |
| `deconvolution/GRN_DECOY_NULL_16_RESULTS.md` | six networks, old edge rule | refs5: 1.02, 0 of 777 |
| `deconvolution/GRN_DECOY_NULL_RESULTS.md` | eight networks of the 14-tissue arm | same |
| `deconvolution/REPFDR_STATUS_16_RESULTS.md` | repfdr on every gene-block | refs5: 139 by concordance, 3,623 by repfdr |
| `deconvolution/PERTURB_PREREG_RESULTS.md` | April's scoring | the myeloid replication has been scored and fails |
| `deconvolution/DOWNSTREAM_BUILD_PLAN.md` | status notes of 2026-10-31 and 11-01 | refs5 status line |
| `deconvolution/DECONV_REBUILD_RUNBOOK.md` | §0 describes the default of 2026-11-03 | three ways the refs5 run differs; §0 not rewritten |
| `deconvolution/EMBEDDING_DE_STANDARDS.md` | 2026-06-15, with lines on the default of 2026-11-01 | refs5: 88 of 150 (p 0.013); 80 (p 0.38); Spearman 0.82 |
| `deconvolution/REFERENCE_SELECTION_PLAN.md` | plan of 2026-07-15 with updates | references_v5; TESTES sensitivity |
| `deconvolution/JOINT_REANNOTATION_RESULTS.md` | run of 2026-10-27 | HYPOTH has since been re-annotated |
| `translation/ATLAS_PROJECTION_16_RESULTS.md` | E.3 on the default of 2026-11-04 | not repeated on refs5; which hotspots changed |
| `translation/ATLAS_PROJECTION_RESULTS.md` | E.3 on 14 tissues | not repeated on refs5 |
| `docs/RETRAIN_V2_RESULTS.md` | v2 on the July references | held-out test, Module A on refs5, corpus measurements |
| `docs/audit/README.md`, `docs/audit/FINDINGS.md` | the audit of 2026-11-11 | the findings were acted on in `remediation/` |

## 3. Notebooks 8-12

**Text corrected** (commit `90fe8ac`; script `tmp/docs_refs5/edit_notebooks.py`, every edit an exact single
match):

| Notebook, cells | What it said | What it says now |
|---|---|---|
| 8: 0, 2, 12 | Default = the 2026-11-02 clean rebuild on `references_v4/`; exclusion list `rat_exclude_genes.tsv`; jobs 12121714-29 | refs5: `references_v5/` (13 identical, HYPOTH re-annotated, HEART merge), `rat_exclude_genes_v2.tsv`, jobs 12377805-20, repeat byte-identical; sampler settings recorded |
| 9: 0, 2, 8, 11, 13 | Target sum "calibration job 12013898"; gate "10,000 permutations" | The registered formula and job 12376341; the gate's further batches; refs5 jobs |
| 10: 0, 2, 5, 12, 14 | Single-sex tissues "cannot have dose-significant genes"; "IHW with tissue as the covariate"; sex-consistency "concordance-only, repfdr degenerate"; hotspot = `q_sup_trained < 0.05`; Tier A "must be interpreted alongside the broader evidence" | The one sex's p in a single-sex block; tissue as a nominal covariate; concordance first, repfdr on dose-significant gene-blocks; hotspot = the gate's class; the specification's reading of the pipeline-sanity subset |
| 11: 0, 2, 4, 10, 12, 14 | 10,000 permutations; hotspot = `q < 0.05` "pre-registered primary criterion"; "held-out" AUC; commands without `--sequential`; "Augur > 0.65 (Stage 9 threshold)" | Sequential batches, the three classes, the four sensitivities and the reading rule; "out-of-fold"; the commands as `run_stage11.py` issues them; 0.65 called a descriptive cut |
| 12: 0, 2, 5, 8, 12, 14 | Atlas "not rerun on 16 tissues"; one status rule | Atlas run on the default before refs5, not repeated; the registered rule, the corrected rule and the dose-trend flag; the test is not a replication; the 0.70 floor is set in the script |

**Code cells changed**, because the summaries they render are statements: the cells that flagged hotspots by
`q < 0.05` read the gate's class (and fall back to `q < 0.05` for a table without one); notebook 10's
summary prints pairs with distinct genes, the pipeline-sanity subset, the technical-covariate check and
both sex-consistency counts, and no longer says single-sex tissues have no gene "by construction";
notebook 11's prints the classes and sensitivities; notebook 12's the corrected rule and the dose trend;
notebook 9 no longer prints "(registered threshold 0.80)". Notebook 9 also printed and labelled a
typed-in target sum of 6400 whatever the run; it now reads it from the outputs (commit `cb21e4f`).

**Execution.** The project's engine (`notebooks/run_notebooks.py`: nbclient, the venv kernel, the
`metadata.rerun` stamp), on the default configuration, through SLURM.

| Job | What | Outcome |
|---|---|---|
| 12392385 | Notebooks 8-12 executed inside the worktree | **Not used.** Notebook 8 failed in its last code cell: `ValueError`, a path under the main tree's `data/` "is not in the subpath of" the worktree (the worktree's `data/` is a symlink). Notebooks 9-12 ran but stored the scratch worktree's path in their outputs. All five copies went to scratch only; the tracked notebooks were not touched by this job |
| 12392406 | Notebooks 8-12, files from the worktree, kernel in the main tree's `notebooks/` folder (`analysis/audit_remediation/docs_refs5_exec_notebooks.py`) | **All five OK**: 9, 9, 10, 8 and 10 code cells run, no error. Committed as `4c2e4e9` |
| 12392439 | Notebook 9 again, after the typed-in target sum was removed | **OK**, 9 code cells, no error; prints `target_sum=6700`. Committed as `cb21e4f` |

The executed summaries agree with the results files on every number both give: 194 blocks, 150 analysable,
11 hotspots, 1 borderline; 52,961 pairs, 9,617 genes, 41,401 outside low-fraction blocks; TESTES 219,
OVARY 8; 23,408 and 17,321 pairs retained; 0 of 150 robust; 5 of 26 testable sanity controls; 139 / 3,623
/ 3,629; 10 of 11 and 9 of 11; sex control 0.756 to 0.745; Spearman 0.73 and 0.79; 88 of 150 (p 0.013);
Augur 0.82; 9,060 embedded, 22 LUNG dropped.

## 4. The preprint (`manuscript/motrpac_genecompass.tex`, main tree, gitignored)

- **Backup** before the first edit: `manuscript/_pre_refs5_backup_20261115/` (the `.tex`, `make_figures.py`,
  `figures/`, and the PDF of 2026-11-13).
- **How it was edited.** Section texts in `tmp/docs_refs5/ms/*.tex` (worktree scratch; not committed,
  because manuscript text does not go into tracked files), applied by
  `tmp/docs_refs5/apply_manuscript.py`, which starts from the backup and replaces ranges between markers
  that must each match once.
- **Figures.** `make_figures.py` was changed: paths to the refs5 inputs (mixture panel, SimBu record, decoy
  tables, Module A record), the hotspot flag from the gate's class, Fig. 3b redrawn as BayesPrism's error in
  the three SimBu arms, Fig. 5b marking the hotspots that fail the corrected rule, axis labels ("held-out"
  removed). Jobs 12392435 and 12392479 (the second after moving one label in Fig. 5a): five figures written,
  no error.
- **Compilation.** `pdflatex`, three passes, texlive 20220321.
  - Job 12392476: **failed** (my error: a duplicated `\end{abstract}`); corrected.
  - Job 12392480: **compiled**, 31 pages. Log: no undefined reference, no undefined citation, no missing
    figure. Two overfull boxes above 10 pt: Table "Blocks with the most dose-significant genes" (53 pt) and
    the Data-availability paragraph (25 pt; that paragraph is unchanged from the backup).
  - Job 12392491: **compiled**, 31 pages, after narrowing that table.
  - Job 12392508, **the PDF now in `manuscript/`** (after two last wording changes; `.tex` md5 `3954acbf...`,
    PDF md5 `b363f042...`): **compiled**, 31 pages. Log: no undefined reference, no undefined citation, no
    missing figure, no multiply defined label. One overfull box above 10 pt remains, in the unchanged
    Data-availability paragraph. One font warning remains (bold small caps, from a section title and a
    caption that the backup also has).
  - **The pages were not looked at as rendered pages.** Jobs 12392503 and 12392509 were to render them to
    PNG; the compute nodes have neither `pdftoppm` nor `gs`, so they wrote nothing. Job 12392558 extracted
    the PDF's text with pypdf instead: 31 pages, no "??" (no unresolved reference or citation), captions of
    all 9 tables and 5 figures present, no empty page. I read the extracted text of the three new or rebuilt
    tables (the nine readings, the blocks with the most calls, the perturbation criteria): rows and columns
    come out as written. The five figures were looked at as PNG.
- **Tests.** The offline suite on this branch at `f41702c` (job 12392517): 462 passed, 0 failed, 22
  deselected (the data and network tests). The data tests were not run.
- **Numbers.** Where the results files give a number it is theirs. Numbers they do not print were read from
  the refs5 tables on disk and are listed in section 5.

### Change list (line numbers are those of the edited `.tex`)

**Added**

| Lines | What |
|---|---|
| 105-153 | Abstract rewritten around the registered rerun: held-out test; 11 hotspots and 1 borderline; instability; no block passes all nine readings; DE not robust; red flag; 4 of 12; both transfer rules; failed replication |
| 376-392 | Paragraph: generalisation to 33 held-out studies (1.046; per study; per tier; what the test is not) |
| 394-414 | Paragraph: where the corpus departs from its description (10.4%; 1.04% and 1.09%; 79 series) |
| 512-530 | Paragraph: the analysis is a registered rerun; what changed in the references, exclusion list and target sum; byte-identical repeat |
| 579-594 | mRNA-content bias against a bias-free baseline: 13 of 15, 14 of 15; fractions are mRNA shares; the corrector caveat |
| 596-613 | Purity sweeps by the registered rule: 12 of 15; 4 of 12 called blocks; none of the muscle blocks |
| 615-623 | Reference quality: SMLINT 50.6%, KIDNEY 16.7%, LUNG not measurable; no specialist has read the labels |
| 696-716 | The gate's classes (sequential permutations), the 11 hotspots and the borderline block |
| 718-732 | Paragraph: the list is not stable (7 of 9; AUC shifts; the two changes not separated) |
| 734-750 | Paragraph: sensitivities S1-S4 and the reading rule |
| 767-796 | Table: every called block against the nine readings |
| 860-876 | Paragraph: DE calls not robust to technical covariates (0 of 150; 44%, 33%) |
| 878-899 | Paragraph: the positive-control red flag stands; why controls are untestable |
| 989-997 | The myeloid replication fails (Spi1 81st percentile; Cebpa not testable) |
| 999-1005 | What the perturbation checks do not establish |
| 1052-1054 | "What this test is": the human-space test is not a replication |
| 1062-1066 | The corrected transfer rule (9 of 11) and the lost dose trend |
| 1116-1120 | Network edge rule: genes present in both arms |
| 1223-1241 | Box: what would have to hold before the calls are established |
| 1346-1356 | Methods: audit and registered rerun, and who did it |
| 1379-1384, 1419-1431, 1492-1500 | Methods: corpus measurements; held-out test; registered deconvolution validation |
| 1509-1525 | Methods: sequential Monte Carlo classes and the sensitivities |
| 1765-1778 | Four references (each in `record/crossref_remediation.tsv`) |

**Weakened**

| Lines | Was | Now |
|---|---|---|
| 213-236 | Contributions: "reliably detectable ... axis"; "all 9 hotspots survive"; "claims we make with confidence" | A test with its limits; 10 of 11 and 9 of 11; "we separate what the registered checks support from what they do not" |
| 679 | Section title "The exercise response is a reliably detectable, cell-type-concentrated axis" | "A supervised probe calls eleven exercise hotspots, and the list is not stable" |
| 752-765 | "carries the exercise axis faithfully and compactly" | "carries the training contrast, with a small margin over a principal-component summary and none over the deconvolved expression itself" |
| 830 | Section title "... recovers known exercise biology at new resolution" | "... many calls, none robust to technical covariates" |
| 901-919 | "Concordance with, and extension of, the bulk atlas": calls attributed to named cell types | "Where the calls are": counts described, not attributed to cell types |
| 952-961 | Oxidative phosphorylation in muscle as "the mitochondrial signature of endurance adaptation", including in rare types | The sets lead in every muscle block; not attributed to the rare types |
| 963-1027 | Perturbation engine "passes three of the four" criteria | Three criteria pass, the fourth is withdrawn, the replication fails |
| 1029 | Section title "... survives transfer ..." | "Transfer of the rat exercise axis into human embedding space" |
| 1072-1095 | Atlas projection as a current result | A result of the run before this one, not repeated |
| 1244-1291 | Discussion: feasibility "with calibrated confidence"; three points of emphasis | The chain works; the biological claims are not established; four points |
| 1333-1341 | Outlook: "turns the descriptive map into a mechanistic and clinical one"; "template" | The next steps depend on the open items |

**Removed**

| Where it was (backup line) | Statement | Why |
|---|---|---|
| Abstract (133-144) | "reliably detectable, cell-type-concentrated axis"; "the map concords with the bulk MoTrPAC atlas while resolving the cell type that carries each signal" | No block passes all nine readings; DE not robust; red flag stands |
| Abstract (156-158) | "Gene-set enrichment recovers the expected oxidative-phosphorylation program in muscle" | The sets lead equally in rare types whose expression is not recovered |
| 538-544 | Reporting policy: differential, immune/stromal and embedding-based claims "are bias-robust and are stated directly" | STEP4 V5: such claims are less exposed and are not tested by the battery |
| 549-562 | Purity sweeps "confirm that the dominant parenchyma's inferred expression is faithfully recovered" | STEP4 V1: 12 of 15 by the registered rule; 4 of 12 called blocks |
| 624-629 | The exercise axis "is a diagonal supervised direction ... rather than a genuinely absent signal" | Interpretation the results files do not contain |
| 648-667 | Paragraphs comparing the 14-tissue arm and the July references | Replaced by the refs5 stability result; history kept in one sentence and in the limitations |
| 754-759 | "The immune program that remains testable recovers" as the lead; the Vetr blood genes in the erythroid block | The lead is the registered red flag; the NK result is kept with its block's abundance (0.1% of blood) |
| 767-784 | The three post-hoc analyses "account for the miss"; "The pipeline is therefore sound" | STEP4 V2: the later check does not lift the flag. (i) and (ii) were not recomputed on refs5 |
| 786-816 | "confirms and refines the published MoTrPAC and Vetr findings"; lung "top sex-opposite tissue" localised; "information that the bulk map structurally cannot contain"; the hypothalamic precursor-cell hotspot | No block's calls are robust; rare types not recovered; that hotspot is no longer called |
| 859-868 | "Two pre-registered robustness checks hold ... not a technical-quality artifact" | STEP3 B5: the registered check fails for every block |
| 945-952, 971-973 | The transfer "is a real perturbation rather than a copy"; "This constitutes transferability in the embedding space" | STEP3 B6: the test re-reads the rat data |
| 997-1000 | A "likely cause" for erythroid cells missing their atlas counterpart | Not in the results files |
| 1030-1031, 1051-1052 | Networks "identical, edge for edge, to six of the eight"; decoy tables "reproduce exactly" | Statements about the earlier arms |
| 1059-1060 | Hub strength follows the TF's own expression | Measured on earlier arms only (finding 9.4) |
| 1073-1077 | Anchor rows of the self-consistency (Cebpa, Spi1) | Not in the refs5 results files |
| 1133-1154 | "Robustness and completeness" box, with the single-sex DE item | Single-sex DE is repaired; replaced by the box of open items |
| 1172-1216 | "size and detectability are distinct"; "cell-type resolution changes the biological interpretation"; the positive-control failure "validated in practice"; "self-validating" paragraph | Not supported by the registered readings |
| 1300-1301 | Methods: T4 tokens "receive" family and network rows | 0 of 2,474 family rows and 36 network rows are non-zero (finding 2.1; Results already said so) |
| 1436-1439 | "A clean rebuild ... regenerates every ... output byte for byte" as a statement about this analysis | For refs5: references rebuilt, Stage 8 twice, later stages once |

## 5. For the PI's eye

### Claims I was unsure how to phrase

1. **"Provisional".** The abstract and limitations call the hotspot and gene lists provisional because no
   called block passes all nine readings. The results files state the fact and draw no such conclusion.
2. **Shared signal.** The preprint says that similar gene counts in every block of BLOOD and SPLEEN, and the
   same enriched sets in every block of a tissue, are "what a signal shared by a tissue's deconvolved
   profiles would produce". An earlier version of the preprint made this reading for BLOOD and SPLEEN. It is
   an interpretation, kept in conditional form; the results files do not make it.
3. **Enrichment content.** "Respiratory electron transport and oxidative phosphorylation lead in every
   skeletal-muscle block; RNA metabolism and chromatin organisation, negatively, in the four BLOOD hotspots;
   interferon-gamma response in the two LUNG hotspots" was read from the refs5 enrichment tables
   (`pseudobulk_de/enrichment/gsea__*.tsv`), not from the results files. I checked the 13 blocks named.
4. **Status marks in the tracker.** Purity sweeps and technical covariates went from ✅ to ◐. The work is
   done; the result is mixed. The tracker already uses ◐ that way for an Aim 1 check.
5. **Critical path in the tracker.** I added that Modules D.1 and F take the per-cell-type DE as input, and
   that its two negative results are unresolved. Whether that changes the order of work is yours to say.
6. **The preprint's title and its three planning boxes** (CPA, genetics, synthesis) are unchanged apart from
   three sentences. The title still promises "programs of endurance-exercise adaptation".
7. **Notebook 11, "Augur > 0.65".** Step 6 could not find a source for 0.65. I reworded it as "a descriptive
   cut of this figure". If it has a source, the cell should name it.
8. **SPLEEN "implausible" fractions.** Kept, with the refs5 numbers (T cells about 2%, endothelial cells
   15%; the text said about 1%). The judgement predates refs5 and no results file re-examines it.

### Numbers taken from the refs5 tables on disk, not from the results files

All read by `analysis/audit_remediation/docs_refs5_preprint_numbers.py` (read-only; it prints them) or
printed by the re-executed notebooks.

| Number | Where used | Table |
|---|---|---|
| 9,082 pseudo-cells built (9,060 embedded) | preprint | notebook 8; `pseudocells.h5ad` per tissue |
| Trace screen: 37 of 194 trained, 101 of 170 sex, medians 0.027 / 0.055 / 0.028, none after BH | preprint, Table of regimes | `pheno_merge_test.tsv`, `subspace_probe.tsv` |
| Silhouette median 0.851 (0.382 to 0.988) | preprint | notebook 9 |
| Near misses; 33 blocks at AUC 0.70 to 0.80; cortex and hippocampus medians | preprint | `gate_rcv10/corroboration_merged.tsv` |
| Augur at least 0.746 in every hotspot; Augur medians 0.603 / 0.621 | preprint | same |
| 2,024,970 tests; skipped blocks' library 1 to 733; 56,362 and 49,494 at the two sensitivities | preprint | `de_methods.tsv`, `de_summary.tsv`, `pseudobulk_de_minlib*/` |
| Per-block and per-tissue DE counts; interaction counts; composition flags per tissue | preprint, Table of calls | `de_summary.tsv`, `composition_confound_table.tsv` |
| NK cytotoxicity 5 of 9, 2.7 times, p 0.022, in a block at 0.1% of blood | preprint | `posctrl_responsiveness.tsv` |
| Ortholog mapping 82-84% of genes, 80-87% of count mass; cosines 0.974 and 0.993; 9,041 human-space pseudo-cells | preprint | notebook 12 |
| 100 of 119 lost blocks had a rat AUC below 0.65; per-hotspot AUC pairs | preprint | `transfer_comparison.tsv` |
| Best hub q 0.25; per-network estimate 0.94 to 1.11; share of positive scores 75% to 100% | preprint | `record/step3/downstream/networks_*` |
| 181 of 192 type tests at 0.95 or more; BayesPrism best on 17 of 19 sets | preprint | `record/step4/v6_mixture_panel/` |
| Atlas accuracies of the cell types now called (SKMVL tenocytes 0.44, HEART vascular smooth muscle cells 0.00, ...) | preprint | `translation/atlas_record_16/atlas_projection_per_type.tsv` (the earlier default's cells) |

### Things in the results files that looked inconsistent

1. **`STEP4_RESULTS.md`, V1.** "At 0.5% of a mixture, the lowest point of the grid and still above their
   abundance". SKMVL tenocytes are 0.7% of their tissue in the table just above, so 0.5% is below their
   abundance. It is above the other four muscle blocks' (0.08% to 0.2%).
2. **`STEP5_RESULTS.md`, line 309.** A placeholder is still in the text: "(file-walk check on GSE214388:
   filled in when job 12379033 finishes)". `record/step5/stage2_walk_switch_check.tsv` exists.
3. **`STEP3_RESULTS.md`, B5 table.** The row "Pairs in the 44 low-fraction blocks" gives 9,277 for the
   earlier default, which had 45 such blocks (`STEP2_STATEMENTS.md`).
4. **`STEP3_RESULTS.md`, opening lines.** "This file is written as the stages finish ... Sections that have
   no numbers yet say so." Every section now has its numbers.
5. **`STEP5_RESULTS.md`, references.** osorio2021 is cited as 2021; `record/crossref_remediation.tsv` records
   the year issued as 2020 (volume 37, issue 7). Trivial.
6. **The headline "9,060 pseudo-cells".** It is the number embedded. 9,082 were built; 22 are dropped at
   tokenization. The results files say so in B3 but the top table does not.

### What I did not change, and why

- **Docstrings and help strings in pipeline code that still name the earlier default**: no analysis code was
  to be changed. They are `pipeline/run_stage9.py:12` ("v2 6,400"), `pipeline/run_stage10.py:32`,
  `pipeline/run_stage11.py:17`, `pipeline/run_stage12.py:31,33`, `pipeline/run_deconv_all.py:15`,
  `deconvolution/augur_prep.py:6`, `deconvolution/tokenize_pseudocells.py:95` ("v2 6,400, April 6,500"),
  `translation/transfer_to_human.py:33`.
- **`deconvolution/BLOOD_REFERENCE_PROPOSAL.md`**: addressed to the collaborator. Its note still holds on
  refs5 (the composite is byte-identical; the same four BLOOD hotspots), so it was left alone.
- **`deconvolution/DECONV_REBUILD_RUNBOOK.md` §0**: banner only. Rewriting the commands needs someone to run
  them.
- **Pre-registrations, `docs/audit/0*.md`, `REFERENCE_QC.md`, `REFERENCE_LABEL_AUDIT.md`,
  `HIPPOC_REFS4_DIAGNOSIS.md`, `MOTRPAC_BULK_LIFTOVER.md`, notebooks 1-7, `manuscript/wip/`**: not touched.
- **The atlas projection, the second heart reference check and the BLOOD composite's checks** were not
  rerun; the documents say they describe earlier references.
- **Preprint**: the model tables, Fig. 2 (apart from one axis label), the Data-availability and
  Acknowledgements sections are unchanged. I did not verify the sentence that weights are archived in the
  university repository.
- **The preprint's three post-hoc parenchyma analyses.** I removed (i) the bulk-to-parenchyma slope
  correlation (0.72 to 0.83; computed on the earlier default, not recomputed) and (ii) "the canonical
  controls are flat or down in the rat bulk (Hspa1b -0.96)". For (ii) I could not tell whether the bulk
  identifier printed as Hspa1b is the consortium's gene, given the identifier drift of STEP5 M3. If you want
  either back, it needs recomputing on refs5.
