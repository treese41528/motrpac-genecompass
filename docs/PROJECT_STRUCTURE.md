# Project: `motrpac-genecompass`

**Repository:** `treese41528/motrpac-genecompass`
**Description:** Cross-species transcriptomic analysis of exercise adaptations using foundation models — integrating MoTrPAC rat data with GeneCompass

---

## Why this name

| Candidate | Verdict |
|-----------|---------|
| `rat-genecompass` | Too narrow — project includes downstream analysis (Aims 2–3), deconvolution, GRN inference, cross-species translation |
| `exercise-transcriptome-fm` | Vague, not searchable |
| `motrpac-foundation-models` | Implies multiple FMs; we're adapting one (GeneCompass) |
| `xspecies-exercise-genomics` | Accurate but obscure |
| **`motrpac-genecompass`** | ✓ Clear, discoverable, maps directly to the grant's core: adapting GeneCompass for MoTrPAC analysis |

The name pairs the data source (MoTrPAC) with the method (GeneCompass), which is exactly what the project does. Someone searching for either term finds the repo.

---

## GeneCompass integration strategy

GeneCompass (https://github.com/xCompass-AI/GeneCompass) is included as a **git submodule** within the project:

```bash
# Initial setup
git submodule add https://github.com/treese41528/GeneCompass.git vendor/GeneCompass
```

**Why submodule (not fork-only or copy)?**

1. **Contained:** GeneCompass lives inside our project tree — no separate clone needed
2. **Pinned:** Submodule locks to a specific commit — reproducible even if upstream changes
3. **Modifiable:** Our fork (`treese41528/GeneCompass`) carries rat-specific patches (extended tokenizer, species token, rat prior knowledge embeddings) while tracking upstream
4. **Clean boundary:** Our pipeline code and their model code stay clearly separated

**Workflow:**
- Fork `xCompass-AI/GeneCompass` → `treese41528/GeneCompass`
- Submodule that fork into `vendor/GeneCompass/`
- Rat-specific modifications go on a `rat-finetune` branch in the fork
- Upstream improvements can be merged via `git fetch upstream`

---

## Directory structure

```
motrpac-genecompass/
│
├── README.md                          # Project overview, setup, quickstart
├── LICENSE                            # MIT or Apache 2.0
├── .gitignore
├── .gitmodules                        # Points to vendor/GeneCompass
├── environment.yml                    # Conda environment specification
├── requirements.txt                   # pip fallback
│
│── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──
│   CONFIGURATION
│── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──
│
├── config/
│   ├── pipeline_config.yaml           # All paths, thresholds, BioMart metadata
│   └── slurm_profiles/               # Cluster-specific SLURM defaults
│       ├── gilbreth.yaml              # Purdue Gilbreth (A100s, reese18 condo)
│       └── template.yaml             # Generic HPC template
│
│── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──
│   SHARED LIBRARY
│── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──
│
├── lib/
│   ├── __init__.py
│   ├── gene_utils.py                  # Gene normalization, BioMart loader, resolver
│   ├── io_utils.py                    # h5ad reading, sparse matrix helpers
│   └── manifest.py                    # Stage manifest creation, config validation
│
│── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──
│   GENECOMPASS (git submodule)
│── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──
│
├── vendor/
│   └── GeneCompass/                   # Submodule → treese41528/GeneCompass
│       ├── scdata/                    #   Pretrained reference data
│       │   ├── dict/                  #     Token dicts, medians, homologs
│       │   ├── human_protein_coding.txt
│       │   ├── mouse_protein_coding.txt
│       │   └── ...
│       ├── pre_training/              #   Pretraining scripts
│       ├── fine_tuning/               #   Fine-tuning scripts
│       ├── GRN_inference/             #   GRN inference module
│       └── ...                        #   (full upstream structure)
│
│── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──
│   AIM 1: DATA PIPELINE
│   Preprocessing rat scRNA-seq → GeneCompass-ready reference files
│── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──
│
├── pipeline/
│   ├── README.md                      # Pipeline-specific docs and stage descriptions
│   │
│   ├── 01_data_harvesting/            # Stage 1: Corpus assembly
│   │   ├── harvest_geo.py             #   GEO bulk download + metadata extraction
│   │   ├── harvest_arrayexpress.py    #   ArrayExpress scraper
│   │   ├── preprocess_h5ad.py         #   Raw → QC'd h5ad (filtering, format normalization)
│   │   └── validate_metadata.py       #   LLM-powered metadata QC
│   │
│   ├── 02_gene_inventory.py           # Stage 2: Gene inventory + BioMart gate
│   │                                  #   → gene_resolution.tsv, study_biomart_coverage.tsv
│   │
│   ├── 03_vocabulary_pruning.py       # Stage 3: Singleton + biotype filtering
│   │                                  #   → pruned gene list
│   │
│   ├── 04_ortholog_mapping.py         # Stage 4: Cross-species ortholog assignment
│   │                                  #   → rat_token_mapping.tsv, tier assignments
│   │
│   ├── 05_compute_gene_medians.py     # Stage 5: Corpus-wide gene medians
│   │                                  #   → rat_gene_median.pickle
│   │
│   ├── 06_create_reference_files.py   # Stage 6: GeneCompass-format outputs
│   │                                  #   → protein_coding.txt, token dicts, medians
│   │
│   └── 07_export_training_corpus.py   # Stage 7: h5ad → tokenized training data
│                                      #   → ready for GeneCompass fine-tuning
│
│── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──
│   AIM 1 (continued): FINE-TUNING
│   Adapting GeneCompass for rat transcriptomics
│── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──
│
├── finetune/
│   ├── README.md
│   ├── pretrain_rat.py                # Continue pretraining on rat corpus
│   ├── finetune_motrpac.py            # Fine-tune on MoTrPAC bulk RNA-seq
│   ├── embed_cells.py                 # Extract cell/gene embeddings
│   ├── configs/                       # Training hyperparameters
│   │   ├── pretrain_rat.yaml          #   Rat continued pretraining config
│   │   └── finetune_motrpac.yaml      #   MoTrPAC fine-tuning config
│   └── patches/                       # Modifications to GeneCompass source
│       ├── extended_tokenizer.py      #   Support for rat tokens
│       ├── species_token.py           #   Add rat species token
│       └── rat_prior_knowledge.py     #   Rat GRN, promoter, gene family embeddings
│
│── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──
│   AIM 1 (continued): DECONVOLUTION
│   Bridging bulk and single-cell resolution
│── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──
│
├── deconvolution/
│   ├── README.md
│   ├── unicell_wrapper.py             # UniCell deconvolution integration
│   ├── scdeal_wrapper.py              # scDEAL domain transfer learning
│   ├── scissor_wrapper.py             # Scissor phenotype-cell linking
│   └── compare_methods.py             # Benchmark deconvolution approaches
│
│── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──
│   AIM 2: DOWNSTREAM ANALYSIS
│   Multi-task transcriptomic analyses on MoTrPAC
│── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──
│
├── analysis/
│   ├── README.md
│   ├── deg_analysis.py                # Differential expression (tissue × sex × timepoint)
│   ├── grn_inference.py               # Gene regulatory network reconstruction (DeepSEM)
│   ├── pathway_enrichment.py          # Pathway enrichment analysis
│   ├── temporal_modeling.py           # Dynamic trajectory modeling
│   └── embedding_analysis.py          # Cell/gene embedding visualization & clustering
│
│── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──
│   AIM 3: CROSS-SPECIES TRANSLATION
│   Rat → human translation and in silico perturbation
│── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──
│
├── translation/
│   ├── README.md
│   ├── cross_species_embedding.py     # Ortholog-aware embedding alignment
│   ├── conserved_regulators.py        # Identify conserved TF networks
│   ├── perturbation_prediction.py     # In silico perturbation experiments
│   └── validate_human.py             # Validate against human exercise cohorts
│
│── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──
│   CLUSTER JOB SCRIPTS
│── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──
│
├── slurm/
│   ├── pipeline/                      # Data pipeline jobs
│   │   ├── 03_vocabulary_pruning.slurm
│   │   ├── 05_compute_gene_medians.slurm
│   │   ├── 07_export_training_corpus.slurm
│   │   └── run_full_pipeline.slurm
│   ├── finetune/                      # Training jobs
│   │   ├── pretrain_rat.slurm
│   │   └── finetune_motrpac.slurm
│   └── analysis/                      # Analysis jobs
│       ├── grn_inference.slurm
│       └── deg_analysis.slurm
│
│── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──
│   TESTS & VALIDATION
│── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──
│
├── tests/
│   ├── test_gene_utils.py             # Unit tests: resolver, biotype, patterns
│   ├── test_ortholog_mapping.py       # Verify no cartesian joins, tier consistency
│   ├── test_reference_files.py        # Validate GeneCompass format compliance
│   └── fixtures/                      # Test data
│       ├── tricky_gene_ids.tsv        # Edge cases: RGD*, LOC*, versioned, whitespace
│       └── mini_biomart.tsv           # Small BioMart subset for unit tests
│
│── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──
│   NOTEBOOKS & DOCUMENTATION
│── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──
│
├── notebooks/
│   ├── pipeline_validation.ipynb      # Stage-by-stage artifact validation
│   ├── embedding_explorer.ipynb       # Interactive embedding visualization
│   ├── motrpac_overview.ipynb         # MoTrPAC data exploration
│   └── cross_species_analysis.ipynb   # Cross-species comparison figures
│
├── docs/
│   ├── ARCHITECTURE.md                # Pipeline architecture & design decisions
│   ├── BUGS.md                        # Bug tracker (was bug_tracker.md)
│   ├── REFACTORING.md                 # Refactoring plan & history
│   ├── DATA_DICTIONARY.md             # Column definitions for all TSV outputs
│   └── figures/                       # Architecture diagrams, workflow figures
│       └── pipeline_overview.png
│
│── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──
│   DATA (gitignored — documented in README)
│── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──
│
└── data/                              # NOT committed — .gitignore'd
    ├── references/                    # BioMart downloads, RGD (read-only inputs)
    │   ├── biomart/
    │   │   ├── rat_gene_info.tsv      #   Primary gene annotation
    │   │   ├── rat_human_orthologs.tsv
    │   │   ├── rat_mouse_orthologs.tsv
    │   │   └── GENES_RAT.txt          #   RGD (symbol synonyms only)
    │   └── rat_genes_biomart.tsv
    │
    ├── training/                      # Pipeline outputs (stages 1-7)
    │   ├── qc_h5ad/                   #   895 QC'd h5ad files (~27 GB)
    │   ├── gene_inventory/            #   Stage 2 outputs + manifests
    │   ├── vocabulary/                #   Stage 3 outputs
    │   ├── ortholog_mappings/         #   Stage 4 outputs
    │   ├── gene_medians/              #   Stage 5 outputs
    │   ├── reference_files/           #   Stage 6 outputs (GeneCompass-ready)
    │   └── tokenized_corpus/          #   Stage 7 outputs (training-ready)
    │
    ├── motrpac/                       # MoTrPAC-specific data
    │   ├── rat_training_6mo/          #   MotrpacRatTraining6moData package
    │   └── processed/                 #   Processed for analysis
    │
    ├── models/                        # Trained model artifacts
    │   ├── pretrained_rat/            #   Continued pretraining checkpoints
    │   ├── finetuned_motrpac/         #   Fine-tuned model weights (~2 GB)
    │   └── embeddings/                #   Extracted embeddings (<1 GB)
    │
    └── results/                       # Analysis outputs
        ├── deg/                       #   Differential expression results
        ├── grn/                       #   Gene regulatory networks
        ├── deconvolution/             #   Cell-type deconvolution results
        └── cross_species/             #   Translation analysis results
```

---

## Mapping to grant aims

| Grant Aim | Repository Location | Key Scripts |
|-----------|-------------------|-------------|
| **Aim 1a:** Data acquisition & preprocessing | `pipeline/01-07` | Stages 1–7 |
| **Aim 1b:** Fine-tune GeneCompass for rat | `finetune/` + `vendor/GeneCompass/` | `pretrain_rat.py`, `finetune_motrpac.py` |
| **Aim 1c:** Bulk-SC deconvolution | `deconvolution/` | UniCell, scDEAL, Scissor wrappers |
| **Aim 2a:** Differential expression | `analysis/deg_analysis.py` | Tissue × sex × timepoint DEG |
| **Aim 2b:** GRN inference | `analysis/grn_inference.py` | DeepSEM integration |
| **Aim 2c:** Temporal modeling | `analysis/temporal_modeling.py` | Dynamic trajectory |
| **Aim 3a:** Cross-species translation | `translation/cross_species_embedding.py` | Ortholog-aware alignment |
| **Aim 3b:** In silico perturbation | `translation/perturbation_prediction.py` | GeneCompass perturbation |
| **Aim 3c:** Human validation | `translation/validate_human.py` | UK Biobank / All of Us |

---

## .gitignore

```gitignore
# Data (too large for git — documented in README)
data/

# Python
__pycache__/
*.pyc
*.pyo
*.egg-info/
.eggs/
*.egg
dist/
build/
.venv/
venv/

# Jupyter
.ipynb_checkpoints/
*.ipynb_checkpoints

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# SLURM logs
slurm-*.out
*.log

# Model artifacts (tracked via DVC or documented separately)
*.pt
*.pth
*.bin
*.ckpt

# Pickle files in data/ (but NOT in vendor/GeneCompass/scdata/dict/)
data/**/*.pickle
```

---

## .gitmodules

```ini
[submodule "vendor/GeneCompass"]
    path = vendor/GeneCompass
    url = https://github.com/treese41528/GeneCompass.git
    branch = rat-finetune
```

---

## Setup instructions (README excerpt)

```bash
# Clone with submodule
git clone --recurse-submodules https://github.com/treese41528/motrpac-genecompass.git
cd motrpac-genecompass

# Or if already cloned without submodules
git submodule update --init --recursive

# Set project root for HPC
export PIPELINE_ROOT=/depot/reese18

# Validate config
python -c "from lib.gene_utils import load_config, validate_config; validate_config(load_config())"

# Run pipeline (stages 2-7)
# See pipeline/README.md for stage-by-stage instructions
```

---

## Key design decisions

1. **`vendor/` not `external/` or `deps/`** — `vendor` is the standard convention for included third-party code (Go, Ruby, JS all use this). Makes clear that GeneCompass is upstream code we've vendored, not our own.

2. **`pipeline/` not `scripts/`** — the data pipeline is a first-class component, not a grab-bag of scripts. Numbered files enforce execution order.

3. **`finetune/` separate from `pipeline/`** — clear boundary between data preparation (deterministic, CPU-heavy) and model training (GPU-heavy, iterative). Different SLURM profiles, different development cycles.

4. **`deconvolution/` separate from `analysis/`** — deconvolution is part of Aim 1 (preparing data for GeneCompass), not Aim 2 (analyzing results). It feeds upstream into fine-tuning, not downstream from it.

5. **`translation/` as its own module** — Aim 3 is conceptually distinct: it consumes outputs from both the fine-tuned model and the analysis module. Keeping it separate prevents circular dependencies.

6. **`lib/` not `src/`** — this is a research project with scripts, not a Python package. `lib/` signals "shared utilities imported by scripts" without implying pip-installable package structure.

7. **`data/` gitignored but fully documented** — follows FAIR principles from the grant. README documents how to obtain/reconstruct each data directory. Model weights tracked via PURR archival.