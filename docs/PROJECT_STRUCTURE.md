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
# Initial setup (tracks UPSTREAM directly; there is no fork)
git submodule add https://github.com/xCompass-AI/GeneCompass.git vendor/GeneCompass
```

**Why submodule (not fork or copy)?**

1. **Contained:** GeneCompass lives inside our project tree — no separate clone needed
2. **Pinned:** Submodule locks to a specific commit — reproducible even if upstream changes
3. **Clean boundary (pristine):** We use it as an upstream library and keep it unmodified. All
   rat-specific work lives in the parent repo (`finetune/genecompass/` code + `data/` assets),
   so there is no fork to maintain and no patches to rebase onto upstream.
4. **Library, not a patch target:** `finetune/genecompass/` adds `vendor/GeneCompass` to
   `sys.path` and imports `from genecompass import ...`; rat behavior (species token, rat
   tokenizer/dictionary, rat prior-knowledge loader) is implemented in our own modules that
   wrap or replace upstream functions (e.g. `rat_load_prior_embedding.py`).

**Workflow:**
- The submodule tracks `xCompass-AI/GeneCompass`, pinned to a specific commit; bump the pin to
  take upstream improvements.
- Rat code -> `finetune/genecompass/`; rat data (prior-knowledge embeddings, fine-tuned model) ->
  `data/training/prior_knowledge` and `data/models/` (gitignored).
- The large UPSTREAM base assets (the `GeneCompass_Base` weights, the human/mouse prior-knowledge
  embeddings, and the pretraining corpora) are downloaded INTO the submodule tree where upstream's
  code expects them. They are not source, so the submodule is set `ignore = dirty` in `.gitmodules`:
  `git status` does not flag those staged assets, but still surfaces a real change to the pinned commit.

---

## Directory structure

```
motrpac-genecompass/
│
├── README.md                          # Project overview, setup, quickstart
├── LICENSE                            # MIT or Apache 2.0
├── .gitignore
├── .gitmodules                        # Points to vendor/GeneCompass
├── requirements.txt                   # Python deps (python3.12 -m venv; see deconvolution/setup/SETUP.md)
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
│   └── GeneCompass/                   # Submodule → xCompass-AI/GeneCompass (upstream)
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
├── deconvolution/                     # Stages 8–9: BayesPrism + omnideconv panel
│   ├── README.md                      #   per-tissue runbook
│   ├── setup/                         #   R/BayesPrism env, site profile, container (SETUP.md)
│   ├── R/                             #   R wrappers: prepare_motrpac_bulk / run_deconvolution / extract_z / ...
│   ├── build_pseudocells.py           #   bulk → per-cell-type pseudo-cells
│   ├── tokenize_pseudocells.py        #   pseudo-cells → GeneCompass tokens
│   └── ...                            #   (UniCell/scDEAL/Scissor are secondary/planned, not yet wrapped)
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
├── tests/                             # pytest (config in pytest.ini)
│   ├── conftest.py                    # puts the pipeline folders on sys.path
│   ├── test_harvesters.py             # Stage 1: file filters, GEO/ArrayExpress/ENA paths
│   ├── test_gene_ids.py               # Stages 2-3: IDs, biotypes, symbol lookup, raw counts
│   ├── test_tokenization.py           # Stages 5 and 8: tokenizer arithmetic, PA preference
│   ├── test_pseudocells.py            # Stage 8: cell-type filenames, Z -> pseudo-cells
│   ├── test_gate_probe.py             # Stage 10: eta^2, folds, supervised probe
│   ├── test_transfer.py               # Stage 12: rat -> human transfer, orchestration
│   ├── test_config.py                 # config layering, v2 default vs April profile
│   └── test_v2_outputs.py             # data: cross-stage checks on this machine's outputs
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
| **Aim 1c:** Bulk-SC deconvolution | `deconvolution/` | BayesPrism + omnideconv panel (Stages 8–9); UniCell/scDEAL/Scissor secondary/planned |
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
    url = https://github.com/xCompass-AI/GeneCompass
    ignore = dirty   # large upstream base assets are staged into the tree (not source)
```

---

## Setup instructions (README excerpt)

```bash
# Clone with submodule
git clone --recurse-submodules https://github.com/treese41528/motrpac-genecompass.git
cd motrpac-genecompass

# Or if already cloned without submodules
git submodule update --init --recursive

# Set project root = your clone (the repo root, NOT a data root)
export PIPELINE_ROOT=$PWD

# Validate config
python -c "from lib.gene_utils import load_config, validate_config; validate_config(load_config())"

# Run pipeline (stages 1-9; see pipeline/README.md). Deconvolution Stages 8-9 first-time
# setup: deconvolution/setup/SETUP.md

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