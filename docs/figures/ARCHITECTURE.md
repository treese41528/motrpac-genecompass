# GeneCompass Rat Fine-Tuning Pipeline — Refactoring Plan

## Motivation

The current pipeline accumulated RGD-annotated gene IDs from older Ensembl
assemblies, inflating the gene count from ~20,600 (BioMart-verified) to 27,677.
This creates 4,561 ghost tokens with no current Ensembl annotation, bloating the
vocabulary (64K → 61.5K after partial cleanup) and adding noise to training.

For a shareable, reproducible codebase: **BioMart is the single source of truth
for gene identity.** Any Ensembl ID not in the current BioMart release is excluded.

---

## Current Pipeline (as-built)

```
Stage 1: Data Harvesting
  GEO/ArrayExpress → raw h5ad files → QC → 895 qc_h5ad files
  
Stage 2: Gene Inventory  
  qc_h5ad → gene_inventory.tsv (189,446 genes across all studies)
  
Stage 3: Singleton Filtering
  gene_inventory → pruned gene list (39,187 → 27,677 genes)
  [includes biotype filtering, singleton removal, expansion]
  
Stage 4: Ortholog Mapping
  27,677 genes → rat_token_mapping.tsv, tier assignments, token IDs
  [uses BioMart orthologs + RGD for symbol resolution]
  
Stage 5: Gene Medians
  27,677 genes × 895 h5ad files → rat_gene_median.pickle
  [vectorized sparse computation, 14.4 min on SLURM]
  
Stage 6: Reference Files
  → rat_protein_coding.txt, rat_miRNA.txt, token dicts, etc.
  [current version: 25,164 genes after RGD duplicate collapse]
```

**Problem:** BioMart filtering happens too late (Stage 6 patch). RGD gene IDs
leak into Stages 3-5, inflating counts and wasting compute.

---

## Refactored Pipeline

```
Stage 1: Data Harvesting                    [NO CHANGE]
  GEO/ArrayExpress → raw h5ad → QC → 895 qc_h5ad files
  
Stage 2: Gene Inventory + BioMart Gate      [MODIFIED]
  qc_h5ad → gene_inventory.tsv
  + BioMart intersection → biomart_verified_genes.txt
  + Per-study coverage report (flag studies with low BioMart match)
  
Stage 3: Vocabulary Pruning                 [MODIFIED — uses BioMart set]
  biomart_verified_genes.txt → singleton filter → pruned gene list
  Input: only BioMart-verified genes
  Output: ~20,600 genes (est.)
  
Stage 4: Ortholog Mapping                   [MINOR CHANGE — smaller input]
  ~20,600 genes → rat_token_mapping.tsv
  No RGD fallback needed for ID resolution (all IDs are canonical)
  RGD still used for symbol synonyms in ortholog matching
  
Stage 5: Gene Medians                       [RECOMPUTE — smaller gene set]
  ~20,600 genes × 895 h5ad files → rat_gene_median.pickle
  Faster (fewer genes), same vectorized approach
  
Stage 6: Reference Files                    [SIMPLIFIED — no collapse step]
  Clean output, no exclusion lists needed
  All genes have verified biotype, chromosome, symbol
```

---

## Script Inventory & Changes

### Stage 1: Data Harvesting
**Scripts:** Various harvesting/QC scripts (not changing)
**Data:** `/depot/reese18/data/training/qc_h5ad/` (895 files)
**Status:** ✅ No changes needed

### Stage 2: Gene Inventory + BioMart Gate
**Current script:** (produced gene_inventory.tsv)
**New script:** `02_gene_inventory.py`
**Changes:**
- After building gene_inventory.tsv, intersect with BioMart
- Output `biomart_verified_genes.txt` — the canonical gene universe
- Output `study_biomart_coverage.tsv` — per-study BioMart match rates
- Output `non_biomart_genes.tsv` — excluded genes with reason codes

**Key logic:**
```python
# Load BioMart canonical set
biomart_genes = load_biomart_genes("references/biomart/rat_gene_info.tsv")

# For each gene in inventory:
#   if gene_id in biomart_genes → KEEP
#   else → EXCLUDE (log reason: "not in current Ensembl release")
```

**BioMart reference:** `rat_gene_info.tsv` (43,360 genes)
- Provides: Ensembl ID, symbol, biotype, chromosome
- This is the ONLY authority for gene identity going forward

### Stage 3: Vocabulary Pruning
**Current script:** `analyze_singleton_genes.py`
**New script:** `03_vocabulary_pruning.py`
**Changes:**
- Input: `biomart_verified_genes.txt` instead of raw gene_inventory
- Singleton filtering operates on BioMart-verified set only
- Biotype filtering uses BioMart biotypes directly (no RGD normalization)
- No biotype normalization map needed (BioMart uses consistent labels)

**Expected output:** ~20,600 genes (vs current 27,677)
- protein_coding: ~20,396
- miRNA: ~207 (BioMart-annotated only)
- Singletons removed: TBD (may differ slightly with smaller input set)

### Stage 4: Ortholog Mapping  
**Current script:** (produced rat_token_mapping.tsv)
**New script:** `04_ortholog_mapping.py`
**Changes:**
- Input: pruned BioMart-verified gene list
- All gene IDs are canonical → simpler ortholog lookup
- RGD used only for symbol synonym resolution (not gene ID authority)
- Tier distribution will shift (fewer "new" tier genes expected)

**Expected output:**
- ~20,600 genes with tier assignments
- Vocabulary: ~57,000 tokens (50,558 existing + ~6,500 new)
- Significant reduction in "new" tier (from 9,065 → est. ~4,000-5,000)

### Stage 5: Gene Medians
**Current script:** `compute_gene_medians.py` / SLURM job
**New script:** `05_compute_gene_medians.py`
**Changes:**
- Input: BioMart-verified pruned gene list (~20,600 genes)
- Same vectorized sparse computation
- Faster execution (fewer genes to process)
- Output: rat_gene_median.pickle (all integer medians from raw counts)

### Stage 6: Reference Files
**Current script:** `create_rat_reference_files.py` (v3)
**New script:** `06_create_reference_files.py`
**Changes:**
- MAJOR SIMPLIFICATION: no RGD annotation loading, no exclusion list
- BioMart provides biotype/symbol/chromosome directly
- No biotype normalization needed
- No duplicate collapse needed
- Clean 1:1 relationship between script input and output

**Outputs (unchanged format):**
- rat_protein_coding.txt
- rat_miRNA.txt  
- rat_mitochondria.xlsx
- Gene_id_name_dict_rat.pickle
- Gene_id_name_dict1_rat.pickle
- rat_gene_median_after_filter.pickle
- extended_tokens.pickle
- rat_gene_token_dict.pickle
- rat_token_mapping_annotated.tsv
- rat_reference_summary.json

---

## Directory Structure (GitHub-ready)

```
rat-genecompass/
├── README.md
├── config/
│   └── pipeline_config.yaml       # All paths, thresholds, parameters
├── lib/
│   └── gene_utils.py              # Shared: normalization, BioMart loader, patterns
├── scripts/
│   ├── 01_data_harvesting/        # GEO/ArrayExpress scrapers (existing)
│   ├── 02_gene_inventory.py       # Gene inventory + BioMart gate + resolver
│   ├── 03_vocabulary_pruning.py   # Singleton/biotype filtering
│   ├── 04_ortholog_mapping.py     # Cross-species ortholog assignment
│   ├── 05_compute_gene_medians.py # Corpus-wide median computation
│   └── 06_create_reference_files.py  # GeneCompass-format outputs
├── slurm/
│   ├── 03_vocabulary_pruning.slurm
│   ├── 05_compute_gene_medians.slurm
│   └── run_full_pipeline.slurm    # End-to-end orchestration
├── tests/
│   └── test_gene_utils.py         # Unit tests for resolver, biotype normalization
├── data/                          # .gitignore'd, documented in README
│   ├── references/                # BioMart downloads, RGD (read-only)
│   │   ├── biomart/
│   │   │   ├── rat_gene_info.tsv
│   │   │   ├── rat_all_genes.tsv
│   │   │   ├── rat_human_orthologs.tsv
│   │   │   ├── rat_mouse_orthologs.tsv
│   │   │   └── GENES_RAT.txt     # RGD (symbol synonyms only)
│   │   └── rat_genes_biomart.tsv
│   └── training/                  # Pipeline outputs
│       ├── qc_h5ad/               # 895 h5ad files
│       ├── gene_inventory/        # Stage 2: gene_resolution.tsv, manifests
│       ├── vocabulary/            # Stage 3 (was singleton_analysis)
│       ├── ortholog_mappings/     # Stage 4
│       ├── gene_medians/          # Stage 5 (was in ortholog_mappings)
│       └── reference_files/       # Stage 6 (GeneCompass-ready)
└── notebooks/
    └── pipeline_validation.ipynb  # Validates metrics from stage artifacts
```

---

## Configuration (pipeline_config.yaml)

```yaml
# Pipeline configuration — all tunable parameters in one place

# Paths (relative to project root)
paths:
  references_dir: data/references
  biomart_dir: data/references/biomart
  training_dir: data/training
  qc_h5ad_dir: data/training/qc_h5ad
  output_base: data/training
  genecompass_dir: apps/GeneCompass

# BioMart reference files
biomart:
  rat_gene_info: biomart/rat_gene_info.tsv        # Primary: ID, symbol, biotype, chr
  rat_genes_biomart: rat_genes_biomart.tsv         # Supplement: NCBI IDs
  rat_human_orthologs: biomart/rat_human_orthologs.tsv
  rat_mouse_orthologs: biomart/rat_mouse_orthologs.tsv

# Gene filtering
filtering:
  min_studies: 2                    # Minimum studies for a gene to be retained
  allowed_biotypes:                 # Biotypes to include in final set
    - protein_coding
    - miRNA
  biomart_only: true                # CRITICAL: restrict to BioMart-verified IDs

# Ortholog mapping
orthologs:
  identity_thresholds:
    one2one: 50.0
    one2many: 70.0
    many2many: 80.0
  use_rgd_synonyms: true            # RGD for symbol resolution, not gene identity

# GeneCompass
genecompass:
  existing_tokens: dict/human_mouse_tokens.pickle
  existing_medians: dict/human_gene_median_after_filter.pickle
```

---

## Implementation Priority

### Phase 1: Core refactor (do now)
1. **02_gene_inventory.py** — Add BioMart gate, output verified gene list
2. **03_vocabulary_pruning.py** — Update to use BioMart-verified input
3. **06_create_reference_files.py** — Simplify (remove RGD/collapse logic)
4. **pipeline_config.yaml** — Centralize all paths and parameters

### Phase 2: Recompute (run on cluster)
5. Re-run Stage 3 (vocabulary pruning) with BioMart-only genes
6. Re-run Stage 4 (ortholog mapping) with pruned BioMart set
7. Re-run Stage 5 (gene medians) with final gene set
8. Run Stage 6 (reference files) — should be clean first pass

### Phase 3: Validation & packaging
9. Compare old vs new pipeline outputs (gene counts, tier distributions)
10. Validate reference files against GeneCompass format expectations
11. Write README, add .gitignore, package for GitHub

---

## Target Numbers (Hypotheses — to be validated)

These are estimates based on current BioMart data and corpus composition.
Actual numbers will be determined by pipeline execution and recorded in
stage manifests. A validation notebook will compute metrics from artifacts.

| Metric               | Current (v3) | Target Range (est.) |
|----------------------|--------------|---------------------|
| Input genes          | 27,677       | 19,000 – 22,000     |
| protein_coding       | 24,177       | 18,500 – 20,500     |
| miRNA                | 973          | 150 – 300           |
| Duplicate symbols    | 364          | 80 – 150            |
| New tokens           | 10,962       | 3,000 – 6,000       |
| Total vocabulary     | 61,520       | 53,000 – 57,000     |
| Ghost/retired IDs    | 0 (patched)  | 0 (by design)       |

Notes:
- miRNA drops significantly because BioMart-only excludes RGD-annotated miRNAs
  not in current Ensembl. Verify this is acceptable for downstream analysis.
- protein_coding should land near Ensembl's 22,548 canonical count minus
  genes absent from our 895-study corpus.
- Vocabulary reduction depends on how many "new" tier genes survive pruning.

---

## Key Design Principles

1. **BioMart defines the canonical gene universe.** "The canonical rat gene
   universe is defined as Ensembl BioMart (release X, assembly Y). All pipeline
   outputs are indexed by canonical ENSRNOG IDs from that release. Noncanonical
   identifiers are resolved via deterministic mapping rules; unresolved
   identifiers are excluded with audit logs."
2. **RGD is a synonym dictionary**, not an authority on gene existence
3. **Filter early, not late** — BioMart gate at Stage 2 prevents downstream contamination
4. **Config-driven** — all paths and parameters in one YAML file
5. **Relative paths** — no hardcoded `/depot/reese18/` in committed code
6. **Reproducible** — every stage logs inputs, outputs, parameters, timestamps
7. **Idempotent** — re-running any stage with same inputs produces same outputs