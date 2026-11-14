# Step 6. Records, tests, hygiene (2026-11-14)

The audit's class C findings and its cross-stage items are about records that are missing, labels that are
wrong, documents that describe a pipeline that no longer exists, and code with no test. This step repairs
them. It is the one step of the remediation that is not allowed to move a result.

- **The rule.** Nothing changed here may alter a number the pipeline produces with its committed defaults.
  A repair that would change a number was not made: the code was left, a switch that is off was added where
  that was useful, and the defect is written down.
- **How the rule was checked.** Not by reading the diff. The changed scripts were run on the data on disk,
  into scratch folders, and their outputs compared by md5 with the same script at the commit this step
  started from (`44332e6`) or with the production output they must reproduce (section 2).
- **Who.** An AI coding assistant (Claude), under the PI's decisions. No domain expert has reviewed this work.
- **Branch.** `audit-remediation-step6`, 29 commits on top of `44332e6`, not pushed.

**Result.** Every item of the step is repaired, guarded, disclosed, or listed with the reason it was left.
Every output-neutrality check came back as expected (8 checks, 97 rows, none unexpected). The offline test
suite passes: 402 tests, 90 of them new. The data tests pass: 10 of 10. One defect was found that the audit
did not list (a bookkeeping error in the purity-sweep generator, section 4); it does not affect the sweep
scores and was left unrepaired for the PI to decide.

Scripts: `analysis/audit_remediation/step6_*`. Record tables: `record/step6/`. Jobs: 12378094 (data tests),
12378421 (neutrality checks), 12379200 (offline tests and two more neutrality checks).

## 1. What was done, finding by finding

"Check" is the evidence that the change is output-neutral. "Unit tests" means `tests/test_step6_hygiene.py`,
`tests/test_stage7.py` or `tests/test_purity_sweep.py`.

| Finding | What was done | Files | Check | Not done, and why |
|---|---|---|---|---|
| **1.10** Stage 3 | **Repaired:** the "flagged non-coding" list was always empty because the biotype was compared in the wrong case; the comparison is now case-insensitive. **Record added:** `stage3_manifest.json` lists modified tracked files, so a build from an uncommitted tree shows. **Disclosed** in the script headers: orthology calls are unfiltered; the canonical outputs record a commit that predates the code that made them; the `confidence` column gates nothing; "T4 = no ortholog" is not exact | `pipeline/03_ortholog_mapping/build_ortholog_mapping.py`, `pipeline/run_stage3.py`, `finetune/genecompass/build_rat_token_dictionary.py` | The list is report-only: no script reads it (searched the repository; notebook 3 reads `tier_statistics` only). Stage 3 rerun into scratch: the 7 data outputs are byte-identical to the canonical ones; `tier_diagnostics.json` differs in that list only (120 genes, all miRNA in T2a and T3a); the manifests differ in time, git and note fields (`neutrality_checks.tsv`, check `stage3`). Unit tests | **ArrayExpress contributes nothing** is disclosed here only (section 4): the keys that point at folders that do not exist are in the `harvesting` block of the config, which belongs to another step |
| **2.4** Scheme B | **Repaired:** the definition is in the repository: [`docs/SCHEME_B_MEDIANS.md`](../../SCHEME_B_MEDIANS.md), linked from `pipeline/README.md` Stage 4. Human median where GeneCompass has one for the rat gene's human ortholog, otherwise the Stage 4 rat median divided by F = 2.052656 | `docs/SCHEME_B_MEDIANS.md`, `analysis/audit_remediation/step6_scheme_b_record.py`, `pipeline/05_tokenization/build_hybrid_medians.py` (header) | Read-only. The dictionary rebuilt from its three inputs with the builder's own functions equals the file on disk in 21,222 of 21,222 values (`scheme_b_medians.tsv`, `scheme_b_source_by_tier.tsv`) | The evidence for choosing B over the other schemes is three files in an uncommitted scratch folder. The page quotes them as read and says so. They were not copied into the repository (section 5) |
| **2.7** Stages 4-6 records | **Record added:** `stage4_manifest.json` and `stage5_manifest.json` carry the commit and the modified files. The release label they copy comes from the config, which has been right since 2026-10-05. **Guarded:** `rat_model_init.py` refuses `t3_prior_source: stage6` (the lookup that leaves every T3a prior row zero) unless `--allow-legacy-t3-priors` is given. **Guarded:** every Stage 6 builder, and `run_stage6.py` before it starts, refuse to replace an existing `<name>_embeddings.pkl` without `--overwrite`. **Disclosed:** `configs/rat_finetune.yaml` says in its header what it does not record of April's build | `lib/provenance.py` (`git_state`), `pipeline/04_gene_medians/gather_gene_medians.py`, `pipeline/05_tokenization/assemble_corpus.py`, `finetune/genecompass/rat_model_init.py`, `pipeline/run_stage7.py`, `lib/gene_utils.py` (`refuse_overwrite`), `pipeline/run_stage6.py`, the four `pipeline/06_prior_knowledge/build_*.py`, `finetune/genecompass/configs/rat_finetune.yaml` | Manifest fields only. The guards sit before any work and the default paths do not pass through them. Both training YAML files and `config/pipeline_config.yaml` parse to the same values as at `44332e6`. Unit tests | The manifests already on disk are not edited. The 26 stale labels in 22 files are tabulated with the right value beside each (`stale_release_labels_on_disk.tsv`). Stages 4-6 were not rerun, so the new manifest fields and the builders' guards are exercised by unit tests only |
| **3.7** Stage 7 | **Disclosed:** `t3_warm_start`, `t3_noise_std` and `species_init` are marked in both training configs as read by no code. **Guarded:** `run_stage7.py --force` was parsed and ignored; it now stops and says what to do. **Disclosed:** `analysis/forgetting_analysis.py` says in its header that it was run for April only and how its losses differ from the registered check. **Stale text corrected** (phase 1 took one job of 4 days 23.6 h; the deposit holds the initial checkpoint and the phase-2 final model; April's vocabulary is 55,275 tokens, not 53,032) | `finetune/genecompass/configs/rat_finetune*.yaml`, `pipeline/run_stage7.py`, `analysis/forgetting_analysis.py`, `scripts/train_stage7.sh`, `README.md`, `finetune/genecompass/README.md`, `finetune/genecompass/rat_pretrain.py`, `finetune/genecompass/build_rat_token_dictionary.py` | Comments, text and one guard. Each corrected figure was read from a file first (job accounting, `config/deposit_md5.tsv`, the checkpoints' `config.json`). Unit tests | `slurm/finetune/stage7.slurm` still describes an old learning rate and step count. It is a local, untracked file and was left |
| **4.10** References | **Repaired:** the LUNG builder takes its samples from `tissue_references.yaml`; `--out` is now required. **Disclosed:** `canonical_references.tsv` says at its top that it is the 10-tissue manifest of June and is read by no code | `deconvolution/build_lung_pooled.py`, `deconvolution/build_references_v2.sh`, `deconvolution/reference/canonical_references.tsv` | LUNG rebuilt into scratch from the yaml: count matrix, gene list, cell table and summary byte-identical to `references_v5/LUNG_native_pooled` (`neutrality_checks.tsv`, check `refs`). The derived sample list equals the list that was typed in (unit test) | **The script that scored the CDSeq components is not in the repository** and was not found. Its result tables are in an uncommitted scratch folder. Disclosed; nothing to commit |
| **5.8** Stage 8 scoring | **Disclosed, switch added, off:** counts are truncated, not rounded, before the variance-stabilising transformation; the comment said "round". `SCORE_Z_VST_ROUND=1` rounds instead. **Disclosed:** the sex-chromosome filter removes 1,574 X-linked and 160 Y-linked genes, 833 of them X-linked protein-coding genes of the vocabulary (`sex_chrom_filter_genes.tsv`); it already has a switch. **Repaired:** SimBu 1.8.0 is in the R package manifest | `deconvolution/R/score_z_vst.R`, `deconvolution/setup/r_packages.yaml`, `analysis/audit_remediation/step6_sex_chrom_filter_record.py` | With the switch off, `score_z_vst.R` gives byte-identical scores to the old script and to the scores on disk in all 11 purity sweeps (`neutrality_checks_vst_sweepgen.tsv`, check `vst`). The filter and the manifest: no code path changed | The SimBu entry was read from `R_libs/SimBu/DESCRIPTION`; no install was run from the manifest |
| **8.6** Atlas | **Record added:** `project_to_atlas.py` writes provenance for `prep` and `score`. **Repaired:** the retrieval of the 11 Tabula Sapiens files is committed, with the sizes and md5s the download jobs recorded. **Wording corrected:** the sex control and its 0.70 floor were set in the script when it was first committed (`fc4a497`); no pre-registration states them, and the script and its report line now say so. The rule is unchanged | `translation/project_to_atlas.py`, `translation/fetch_tabula_sapiens.sh`, `translation/tabula_sapiens_md5.tsv`, `translation/compare_transfer.py` | `score` and `prep` (Kidney), old against new: identical; `score` also equals the committed `atlas_record_16`. The 11 files on disk match the committed table in size and md5 (`neutrality_checks.tsv`, check `atlas`). Unit tests | The download itself was not run (only `--verify`); the URLs are those of the two local job scripts |
| **9.4** Superseded figures | **Corrected** where a figure of a superseded arm was still quoted bare: the tracker's "3,478,564 edges / 398,586 reproducible" (April, pre-July references; "reproducible" withdrawn); the hub-against-expression correlation (July references only); "median 0.297" in `REFERENCE_RELABEL_PREREG.md` (a dated note: it is the median of the per-network medians, 0.310 the median over all scores) | `docs/OPEN_FRONTIER.md`, `deconvolution/REFERENCE_RELABEL_PREREG.md`, `analysis/audit_remediation/step6_self_consistency_medians.py` | Text. Both medians re-derived from the table on disk: 0.3096 over 859 scores, 0.2974 over 8 networks (`self_consistency_two_medians.tsv`) | "0 of 4 myogenic anchors" occurs in no committed document outside the audit. Headline figures of the current arm were not touched (section 4) |
| **9.6** Network records | **Record added:** `perturb_cells.py` writes provenance beside its table, with the md5 of the weights (`build_grn.py`, `build_grn_human.py` and `build_grn_pooled.py` already did, since `d2971dd`). **Disclosed** in `delete_token`: a deletion moves every lower-ranked gene up one position, and the model uses position embeddings, so the reported shifts hold both effects; no control separates them | `finetune/genecompass/perturb_cells.py` | The record is written after the table. Unit test with a stand-in model | "48% of default edges have a stored difference of at most 1e-6" is the audit's count and was not recomputed |
| **Release labels** | **Corrected** in living code, comments and the config's `deconvolution` block; the two label strings `prepare_motrpac_bulk.R` writes now say "release 115". **Dated notes** on `MOTRPAC_BULK_LIFTOVER.md` and `PERTURB_PREREG.md` | `deconvolution/R/prepare_motrpac_bulk.R` and `.sh`, `deconvolution/tokenize_pseudocells.py`, `deconvolution/build_posctrl_prereg.py`, `deconvolution/audit_idspace.py`, `finetune/genecompass/build_rat_tf_list.py`, `config/pipeline_config.yaml`, the two documents | Outside comments, the diff of `prepare_motrpac_bulk.R` is two label strings. The config parses to the same values as at `44332e6` | `prepare_motrpac_bulk.R` was not run. The committed `motrpac_missed_genes_summary.txt` still reads "rel-113" in two lines until the bulk is next prepared. `audit_idspace.py` keeps its column name `in_biomart113`, so the tables already written still match |
| **Provenance** | **Record added** to `pheno_merge_test.py`, `augur_prep.py`, `project_to_atlas.py`, `perturb_cells.py` and the three reference builders that wrote none (LUNG, BLOOD, SPLEEN/OVARY). **Both helpers now store an md5** for every regular input file under 2 GB, beside size and time | `lib/provenance.py`, `deconvolution/R/provenance.R`, the scripts named, `deconvolution/joint_reannotate.py` (`raw_input_files`) | Old against new on the data: `pheno_merge_test.tsv` identical; `pca_control.tsv` and the 48 Augur export files identical; the four references byte-identical (`neutrality_checks.tsv`). A record is written after the outputs and reads none of them. Unit tests | `score_z.py` still writes no record. It belongs with `score_validation.py` and `make_pseudobulk.py`, which another step is changing |
| **Stale documents** | `pipeline/README.md`: Stage 11 is in the stage table, the run order is stated, and the paragraph that named superseded folders is replaced by the config keys that hold each stage's outputs. **Banners** on `docs/ARCHITECTURE.md`, `docs/BUGS.md`, `docs/PROJECT_STRUCTURE.md`, at the start of the body of `DECONVOLUTION_PIPELINE_REPORT.md`, and at the top of `notebooks/RERUN_EDITS.md`. **Notebook 11:** three cell sources corrected | those files, `notebooks/pipeline11_analysis.ipynb`, `deconvolution/README.md` (one sentence) | Text. Notebook 11: five source lines in cells 0, 8 and 13; stored outputs and metadata unchanged (asserted when the file was written); not re-executed | **Notebook 10 was not changed:** every script and file its markdown names exists at this commit. What a refs5 re-execution will make stale in notebooks 10 and 11 is listed in section 4 |
| **Tests** | **90 offline tests added** in three files: purity sweeps and Z scoring (19), Stage 7 without a GPU (43), the guards and records of this step (28). **Data tests run once** | `tests/test_purity_sweep.py`, `tests/test_stage7.py`, `tests/test_step6_hygiene.py` | Offline suite, job 12379200: 402 passed, 0 failed. Data tests, job 12378094: 10 passed (section 3) | The mixture panel is not covered here: its scripts belong to another step |
| `run_stage13._safe` | **Not switched.** The docstring says why | `pipeline/run_stage13.py`, `analysis/audit_remediation/step6_stage13_names_check.py` | Compared on all 191 reference x cell-type names of `references_v5`: the shared `safe()` differs in case for every name and beyond case for 9 (`CD4+ T cells`, `SMCs & Pericytes`, ...). Switching would rename every network file (`stage13_network_names.tsv`) | As stated |
| Positional joins in R | **Repaired:** `de_technical_covariate_robustness.R` takes the animal behind each row from the sample key (`deconvolution/R/sample_key.R`, the R twin of `sample_join.py`) and stops when the key is missing, unless `ALLOW_POSITIONAL_SAMPLE_JOIN=1`. The two CDSeq scripts name their proportions through CDSeq's own sample names and stop on a matrix whose shape does not fit its name files | `deconvolution/R/sample_key.R`, `deconvolution/R/de_technical_covariate_robustness.R`, `deconvolution/R/run_cdseq_rr.R`, `run_cdseq_realbulk.R` | RIN/globin check, old against new on the default tree: identical with the positional pairing allowed and with a key in order; stops without a key and with two vial labels exchanged (`neutrality_checks.tsv`, check `rin`). CDSeq: on the saved July result the new naming equals the positional one and the saved table (`cdseq_naming_check.tsv`) | CDSeq itself was not rerun, so the two scripts were not executed end to end |
| Median-library threshold | **Not changed.** `run_pseudobulk_de.R` already has one definition (`MIN_MEDIAN_LIBSIZE`, line 103). The second one is the gate's constant in `corroborate_summary.py`. A unit test now fails if the two differ | `tests/test_step6_hygiene.py` | Nothing changed in either script | The DE script is in use by the rerun and was not touched |

## 2. The output-neutrality checks

`analysis/audit_remediation/step6_neutrality_checks.py` makes two snapshots of the worktree (the old commit
and the new one), links `data/` into both, and runs each changed script from each. It writes into scratch
only. One row per compared file is in `record/step6/neutrality_checks*.tsv`.

| Check | What ran | Compared with | Result | Job, new commit |
|---|---|---|---|---|
| `refs` | The LUNG, BLOOD, SPLEEN and OVARY builders | `references_v5/` | Count matrix, gene list and cell table byte-identical in all four. SPLEEN's `summary.txt` differs only in the repository root inside a quoted path | 12378421, `82f94a6` |
| `stage3` | The Stage 3 worker | `data/training/ortholog_mappings/` | 7 data outputs identical; 3 JSON files differ in the expected keys only | same |
| `rin` | The RIN/globin check, four ways | The old script | Identical twice; stopped twice, as it must | same |
| `pheno` | `pheno_merge_test.py` | The old script | Identical | same |
| `augur` | `augur_prep.py`, 20 permutations | The old script | 49 of 49 files identical | same |
| `atlas` | `project_to_atlas.py score` and `prep`; the checksum table | The old script; the committed record; the files on disk | Identical; 11 of 11 files verify | same |
| `vst` | `score_z_vst.R` on every purity sweep on disk | The old script; the scores on disk | Identical in 11 of 11 sweeps, to the old script and to the scores on disk | 12379200, `5518dbd` |
| `sweepgen` | `make_purity_sweep.py`, LUNG alveolar type II sweep | The old script; the sweep on disk | 33 of 33 files identical; mixtures, truth and design also equal the sweep on disk. The new note names mix7, mix26 and mix32, the three mixtures with a negative cell fraction | same |

The scripts checked by job 12378421 did not change between `82f94a6` and the end of the step. Three files
they import did: `lib/provenance.py` and `lib/gene_utils.py` each gained one function that none of them
calls, and `config/pipeline_config.yaml` changed in comments only. After `5518dbd` only one line of a test
file changed.

**Not checked by a run on data**, and why: the Stage 4, 5 and 6 writers and guards and the Stage 7 guards
(those stages were not rerun; unit tests); `perturb_cells.py` (needs the model on a GPU; unit test with a
stand-in); `prepare_motrpac_bulk.R` (it would rewrite the prepared bulk that the rerun is reading); the two
CDSeq scripts (a run takes the sampler; the naming was checked on the saved result).

## 3. Tests

**Offline suite** (`python -m pytest -q`, job 12379200, worktree at `d230253`, no tracked file modified):
**402 passed, 0 failed**, 17 deselected (the 10 data tests and the network tests), in 2 min 22 s. The 90 new
tests passed on their first run. They were written and their expected values worked out before any of them
was run, because the suite goes through the queue; one expected value that had been worked out wrongly by
hand (a half-way rounding case) was caught and corrected before the file was committed.

**Data tests** (`python -m pytest -m data`, job 12378094, `PIPELINE_PROFILE` unset, worktree at `a3d0270`):
10 selected, **10 passed, 0 failed, 0 skipped**.

| Test | Result |
|---|---|
| `test_exclude_genes.py::test_committed_v2_list_is_what_the_rule_builds` | passed |
| `test_gene_ids.py::test_real_symbol_lookup_covers_standard_genes` | passed |
| `test_gene_ids.py::test_every_corpus_file_holds_raw_counts` | passed |
| `test_stage10_de.py::test_confound_statuses_on_the_refs4all_rebuild_default` | passed |
| `test_stage13_step3.py::test_enrichment_ortholog_pairs_are_the_stage12_map` | passed |
| `test_v2_outputs.py` (5 tests) | passed |

**The failure that finding 8.4 predicted did not occur, and not because of anything done here.** The audit
expected `test_human_transfer_mirrors_the_rat_run` to fail on the 20 LUNG pseudo-cells that exist in rat
space only. Commit `32bb97b` (Step 3, before this step began) changed that test: it asserted that the human
and rat counts are equal, and now asserts that the human pseudo-cells are a subset of the rat run's, by cell
ID. The subset form passes. No data test was changed in this step.

## 4. For the PI

### Behaviour that changes when this branch is merged

None of these moves a number. Each stops, or records, something that used to pass silently.

| What | Effect |
|---|---|
| `de_technical_covariate_robustness.R` (Stage 14, step 3) | Stops on an output tree that has no `mixture_samples.tsv`. The refs5 tree has the key. The present default tree (`_refs4all_rebuild`) does not: there it needs `ALLOW_POSITIONAL_SAMPLE_JOIN=1` |
| `run_stage6.py` and the four Stage 6 builders | Stop when the output exists, which it does in the default prior-knowledge folder. `--overwrite`, or a fresh folder |
| `run_stage7.py --force` | Exits with instructions |
| `rat_model_init.py --t3-priors stage6` | Needs `--allow-legacy-t3-priors` |
| `build_lung_pooled.py` | Needs `--out` (`build_references_from_config.py` always passed it) |
| Provenance records | Hash every input file under 2 GB. Seconds per record; `PROVENANCE_MD5=0` turns it off |
| A Stage 3 rerun | `tier_diagnostics.json` and `stage3_manifest.json` no longer match the md5s in `docs/repro_stages1_7/stage3_ortholog_mappings.tsv`: the first in the flagged list, the second in the git and note fields. The other outputs match |
| A re-preparation of the bulk | `motrpac_missed_genes_summary.txt` differs from the committed copy in two label lines |

### Headline numbers and method text left for the rewrite after refs5

Not touched, as instructed. Where they are:

- `pipeline/README.md`, the results table (marked "before the refs5 rerun"); `README.md`;
  `deconvolution/README.md` (top section); `docs/OPEN_FRONTIER.md`; the "Current figures" and "refs4, 14
  tissues" sections of `DECONVOLUTION_PIPELINE_REPORT.md`, which also name the `_refs4all` folders.
- **Statements that the code at `44332e6` already contradicts**, because Step 3 changed the method:
  "single-sex tissues (TESTES, OVARY) cannot have a dose-significant gene" (`pipeline/README.md` results
  row 10; `deconvolution/README.md`, top section; notebook 10, cell 2, point 6); "IHW with tissue as the
  covariate" and the 8-week sex-consistency rule (notebook 10, cell 2, points 7 and 8); "a block is a hotspot
  if `q_sup_trained < 0.05`" (notebook 10, cells 2 and 5; notebook 11, cells 2, 4, 10 and 12; the config
  comment above `deconvolution.hotspots_table`).
- **Notebook text tied to the present default's folders and jobs:** notebooks 10 and 11, cell 0
  (`_refs4all_rebuild`, `references_v4/`) and cell 14 (job numbers; in notebook 11 the three commands, which
  have no `--sequential`, and `gate_rcv10/`).
- The config comment at `deconvolution.pseudocell_target_sum` ("v2 6,400").
- Notebook 11, cell 13: its source no longer prints "(registered threshold 0.80)", but its stored output
  still shows the phrase until the notebook is run again.

### Things that looked as if they could change a result

| What | What was found | Status |
|---|---|---|
| **Purity-sweep generator** (`make_purity_sweep.py`), found while writing its tests | The rounding-drift repair can leave a background cell type with a negative cell count. The mixture then holds one or two cells too many, and its `cellfrac__` columns and `focal_cellfrac` are wrong by that much (one entry is negative). Counts, true expression and `rnafrac__` come from the cells actually drawn and are exact, so the sweep scores are not affected. On disk: 3 of 660 mixtures of the default Stage 14 sweeps, all in the LUNG alveolar type II sweep; 12 of 840 in the research-phase folder (`purity_sweep_negative_cellfrac.tsv`) | Left as it is: a repair changes the random stream of every sweep. The run now prints how many mixtures are affected; one test pins the behaviour. **Your call whether to repair it before the next sweeps** |
| **Truncation before the VST** (5.8) | Measured with the switch on, on the 11 sweeps on disk (660 mixtures): rounding instead of truncating moves a sweep's median Pearson-VST by at most 0.0002 (HYPOTH 0.9950 to 0.9952, LUNG alveolar type II 0.9975 to 0.9977) and a single mixture by -0.0006 to +0.0019. No sweep's median is near 0.95 under either rule (the lowest is 0.9950) | Default unchanged; switch off. The effect is in the fourth decimal |
| `run_stage6.py --test` | It passed `--test` to `build_grn_transfer.py`, which has no such option, so `--from 3 --test` would have failed at step 4. And `build_promoter_embedding.py --test` writes the same `promoter_embeddings.pkl` as a full run | The flag now goes to the promoter step only, and the promoter's test mode is under the overwrite guard |
| `build_references_v2.sh` (June) | It called the LUNG builder with no `--out`. The builder's default had since moved to a later production folder (`references_v4/LUNG_native_pooled`), so running the old script would have overwritten that reference with consensus labels | The call now writes into the script's own folder; the builder has no default |
| `analysis/forgetting_analysis.py` | The species-embedding drift is looked up under a tensor name that does not exist, so that column is empty in every row of the April report | Disclosed in the header; not repaired (the script is April-only) |
| Notebook 11, cell 10 | "`augur_embed_trained > 0.65` (Stage 9 threshold)". Augur runs in Stage 11, and no source for 0.65 as a threshold was found | Not changed; I could not establish what it refers to |

### What could not be verified

- **The decision evidence for Scheme B, April's prior rows and the CDSeq component scores** exist only as
  files in an uncommitted scratch folder of the main checkout (`tmp/data_audit/`, `tmp/cdseq_*.tsv`). They
  were read, not copied (section 5). Whoever cleans that folder removes the only copy.
- **Which annotation RGD's `ENSEMBL_ID` column follows.** Two documents explain the 60.6% overlap by "RGD
  tracks the newer GRCr8". That cannot be right, since the project's files are GRCr8. The notes say so and
  leave the question open.
- **"About 80% of BLOOD's mass matches no PBMC type"** rests on the unreproducible CDSeq scoring; nothing
  here changes that.
- The audit's figures quoted in the new header text of Stage 3 (one-to-one calls down to 3.3% and 3.4%
  identity; 237 T4 genes with an ortholog row) and in `forgetting_analysis.py` (4.245 against 3.444) were
  copied from the audit, not recomputed.

### A mistake made and corrected in this step

Commit `532f7cc` added a sentence to the header of `build_rat_token_dictionary.py` saying that April's
Stage 3 gave only T4 genes a new token and that April's dictionary had 53,032 entries. It was written
without checking April's checkpoint. The `config.json` of April's initial checkpoint and of its phase-2
checkpoint both record `vocab_size` 55,275: April's model has the same 4,717 new tokens as v2, and 53,032
was an earlier design that no model was trained on. Commit `5a0c44e` corrects the header, with a dated note,
and the same wrong figure in `finetune/genecompass/README.md` and `rat_pretrain.py`.

### Done beyond the list

- `docs/PROJECT_STRUCTURE.md` got a banner: it lists files that were never created, and the other two
  planning documents are described by it.
- `docs/OPEN_FRONTIER.md` said "there is no `run_stage11.py`" (finding 6.10; Step 2 corrected
  `pipeline/README.md`, not this row).
- Notebook 11 cited a "pre-registered threshold" of 0.80 for the Augur correlation in a markdown cell and in
  the summary its last code cell renders (finding 6.5; Step 2 corrected `AIM2_DECONV_RESULTS.md`).
- `finetune/genecompass/README.md` still called the human and mouse sets "0.5M" (finding 2.3).

## 5. An action that was refused

To put the decision evidence for Scheme B under version control, a command was issued to copy three files
from the main checkout's scratch folder (`tmp/data_audit/median_check/`: a script, its log and its summary
table) into `record/step6/`. The permission system refused it, and it was not retried another way: the
brief asked for numbers taken from files on disk and named, not for scratch material to be imported.
`docs/SCHEME_B_MEDIANS.md` therefore cites those files where they are and says they are uncommitted. The
same holds for `tmp/data_audit/prior_zero_rows_by_tier.tsv` (April's prior rows) and the CDSeq annotation
tables, which were not copied either. **If you want them preserved, copy them yourself or say so.**

## 6. Record tables (`record/step6/`)

| File | Holds | Made by |
|---|---|---|
| `neutrality_checks.tsv`, `neutrality_checks_summary.tsv` | Checks `refs`, `stage3`, `rin`, `pheno`, `augur`, `atlas`: one row per compared file | `step6_neutrality_checks.py`, job 12378421 |
| `neutrality_checks_vst_sweepgen.tsv` and its summary | Checks `vst` and `sweepgen` | same script, job 12379200 |
| `scheme_b_medians.tsv`, `scheme_b_source_by_tier.tsv` | Every figure of the Scheme B page, with its source file | `step6_scheme_b_record.py` |
| `stale_release_labels_on_disk.tsv` | The release and assembly labels in the manifests on disk | `step6_release_labels_on_disk.py` |
| `stage13_network_names.tsv` | Stage 13's file names against the shared sanitiser, 191 names | `step6_stage13_names_check.py` |
| `cdseq_naming_check.tsv` | The CDSeq naming on the saved July result | `step6_cdseq_naming_check.R` |
| `sex_chrom_filter_genes.tsv` | What the sex-chromosome filter removes | `step6_sex_chrom_filter_record.py` |
| `purity_sweep_negative_cellfrac.tsv` | Sweep mixtures with a negative recorded cell fraction | `step6_purity_sweep_cellfrac_scan.py` |
| `self_consistency_two_medians.tsv` | The two "medians" of the rat-human self-consistency | `step6_self_consistency_medians.py` |
