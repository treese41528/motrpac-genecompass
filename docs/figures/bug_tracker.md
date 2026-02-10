# Pipeline Bug & Fix Tracker
**Last updated:** 2026-02-10

## Architectural Decisions (locked)
- **BioMart is canonical gene universe** (Ensembl release pinned in config)
- **RGD: symbol synonyms ONLY** — `use_for_biotype_fallback: false`
- **Policy in YAML, mechanism in gene_utils.py** — no hardcoded keep sets
- **Ortholog merge: separate_best_then_merge** — never outer-join raw tables
- **project_root: "."** — override via PIPELINE_ROOT env var

## Priority: P0 (changes results), P1 (correctness risk), P2 (robustness)

---

## P0: Changes Results

| ID | Script | Bug | Status |
|----|--------|-----|--------|
| BUG-01 | build_gene_inventory | n_studies is actually n_occurrences — all singleton filtering wrong | TODO |
| BUG-02 | build_ortholog_mapping | Outer join creates N×M cartesian products for multi-ortholog genes | TODO |
| BUG-03 | build_gene_inventory | `all_lower` mixes cases — undercounts rat_ref_match | TODO |
| BUG-04 | all scripts | Biotype normalization inconsistent (mirna vs miRNA vs protein-coding) | **DONE** (gene_utils.py) |
| BUG-05 | build_gene_inventory | Pattern exclusion keys by gene_type not pattern_match | TODO |

## P1: Correctness Risk

| ID | Script | Bug | Status |
|----|--------|-----|--------|
| BUG-06 | build_gene_inventory | rat_ref_match samples first N genes (biased) | TODO |
| BUG-07 | prune_vocabulary | Singleton % denominator uses all var genes, not pruned-vocab | TODO |
| BUG-08 | build_ortholog_mapping | Tier keys: "human-rat" vs "human_rat" | **DONE** (Tier constants) |
| BUG-09 | build_ortholog_mapping | RGD biotype by column index [36],[37] — breaks on schema change | TODO |
| BUG-10 | verify_symbol_lookup | Falsy check: `if n_studies_col` fails when index is 0 | TODO |
| BUG-11 | build_gene_inventory | Non-gene patterns case-sensitive (jacyvu vs JACYVU) | **DONE** (re.IGNORECASE) |
| BUG-12 | build_gene_inventory | Version stripping only for ENSRNOG, not other namespaces | TODO |
| BUG-13 | build_gene_inventory | Iterates study_results instead of usable_results | TODO |
| BUG-14 | build_gene_inventory | Dominant gene type misclassified by heavy noncoding content | TODO |
| BUG-15 | build_ortholog_mapping | linked_mouse/linked_human prechecks redundant (depends on BUG-02) | TODO |
| BUG-16 | build_ortholog_mapping | Identity stats may break on string types | TODO |
| BUG-17 | build_ortholog_mapping | corpus_base .upper() creates collisions with non-Ensembl junk | TODO |

## P2: Robustness

| ID | Description | Status |
|----|-------------|--------|
| IMPROVE-01 | Pin BioMart release metadata + checksum validation | **DONE** (config + validate_config) |
| IMPROVE-02 | Multi-namespace resolver (tiers 1+3: Ensembl + symbol) | **DONE** (BioMartReference.resolve) |
| IMPROVE-03 | Per-stage manifests with config snapshot | **DONE** (create_stage_manifest) |
| IMPROVE-04 | gene_resolution.tsv output from Stage 2 | TODO |
| IMPROVE-05 | Unit tests for resolver | TODO |
| IMPROVE-06 | Rename misleading variables (samples→files/matrices) | TODO |
| IMPROVE-07 | Write resolved_ensembl_id into gene_inventory.tsv | TODO |
| IMPROVE-08 | Assert accession pattern (fail loudly) | **DONE** (extract_accession + config patterns) |
| IMPROVE-09 | Expected numbers → targets with ranges | **DONE** (refactoring_plan.md) |
| IMPROVE-10 | Remove unused catalog load in prune_vocabulary | TODO |
| IMPROVE-11 | Apply non-gene patterns BEFORE resolution | TODO |
| IMPROVE-12 | Add token_class coarse field | TODO |
| IMPROVE-13 | Stage A freq dist inflation (removed ≠ never in corpus) | TODO |
| IMPROVE-14 | study_n_files sanity check | TODO |
| IMPROVE-15 | Low-identity ortholog tier (configurable) | **DONE** (config flag + Tier.LOW_IDENTITY) |

---

## Implementation Order

### Phase 1: Shared foundation ✅
1. [x] `gene_utils.py` — policy-parameterized, config-driven
2. [x] `pipeline_config.yaml` — portable, validated, checksummed

### Phase 2: Core script rewrites
3. [ ] `02_gene_inventory.py` — BioMart gate + resolver
       Fixes: BUG-01, BUG-03, BUG-05, BUG-06, BUG-12, BUG-13, BUG-14
       Implements: IMPROVE-04, IMPROVE-07, IMPROVE-11
4. [ ] `03_vocabulary_pruning.py` — BioMart-verified input
       Fixes: BUG-07
       Implements: IMPROVE-06, IMPROVE-10, IMPROVE-13
5. [ ] `04_ortholog_mapping.py` — separate_best_then_merge
       Fixes: BUG-02, BUG-09, BUG-15, BUG-16, BUG-17
6. [ ] `06_create_reference_files.py` — simplified BioMart-only

### Phase 3: Recompute on cluster
7. [ ] Run Stages 2-6 sequentially

### Phase 4: Validation & packaging
8. [ ] Compare old vs new outputs
9. [ ] BUG-10, IMPROVE-05, IMPROVE-12
10. [ ] README, .gitignore, validation notebook