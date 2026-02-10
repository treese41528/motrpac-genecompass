# motrpac-genecompass

Cross-species transcriptomic analysis of exercise adaptations using the [GeneCompass](https://github.com/xCompass-AI/GeneCompass) foundation model, applied to [MoTrPAC](https://motrpac.org/) rat endurance exercise training data.

**NIH-funded project** (RFA-RM-24-011) — Purdue University Department of Statistics

---

## Overview

This project adapts GeneCompass — a foundation model pretrained on 120M+ human and mouse single-cell transcriptomes — to analyze rat bulk and single-cell RNA-seq data from the Molecular Transducers of Physical Activity Consortium (MoTrPAC). The goal is to uncover molecular mechanisms of exercise-induced adaptations and translate findings from rat models to human biology.

### What this repository contains

| Component | Location | Description |
|-----------|----------|-------------|
| **Data pipeline** | `pipeline/` | 7-stage preprocessing: corpus assembly → GeneCompass-ready reference files |
| **GeneCompass** | `vendor/GeneCompass/` | Foundation model (git submodule, forked with rat extensions) |
| **Fine-tuning** | `finetune/` | Continued pretraining on rat corpus + MoTrPAC fine-tuning |
| **Deconvolution** | `deconvolution/` | Bulk ↔ single-cell integration (UniCell, scDEAL, Scissor) |
| **Analysis** | `analysis/` | Differential expression, GRN inference, temporal modeling |
| **Translation** | `translation/` | Cross-species embedding alignment, in silico perturbation |
| **Shared library** | `lib/` | Gene normalization, BioMart reference loading, utilities |

### Research aims

1. **Fine-tune GeneCompass for MoTrPAC** — Extend the model to rat transcriptomics with species-specific tokenization, ortholog mapping, and deconvolution-based single-cell resolution
2. **Multi-task transcriptomic analysis** — Differential expression, gene regulatory network reconstruction, and pathway enrichment across 18 rat tissues, 8 timepoints, and both sexes
3. **Cross-species translation** — Align rat exercise biology with human transcriptomes to identify conserved regulatory mechanisms and therapeutic targets

---

## Quick start

```bash
# Clone with GeneCompass submodule
git clone --recurse-submodules https://github.com/treese41528/motrpac-genecompass.git
cd motrpac-genecompass

# Create environment
conda env create -f environment.yml
conda activate motrpac-gc

# Set project root (HPC)
export PIPELINE_ROOT=/depot/reese18    # or your local data root

# Validate configuration
python -c "from lib.gene_utils import load_config, validate_config; c = load_config(); validate_config(c); print('Config OK')"
```

### Running the data pipeline

The pipeline runs in numbered stages. Each stage reads from the previous stage's output directory and writes a JSON manifest for reproducibility.

```bash
# Stage 2: Gene inventory + BioMart gate
python pipeline/02_gene_inventory.py

# Stage 3: Vocabulary pruning (singleton + biotype filtering)
python pipeline/03_vocabulary_pruning.py

# Stage 4: Ortholog mapping (cross-species token assignment)
python pipeline/04_ortholog_mapping.py

# Stage 5: Gene median computation (SLURM recommended)
sbatch slurm/pipeline/05_compute_gene_medians.slurm

# Stage 6: Create GeneCompass reference files
python pipeline/06_create_reference_files.py

# Stage 7: Export tokenized training corpus (SLURM recommended)
sbatch slurm/pipeline/07_export_training_corpus.slurm
```

See `pipeline/README.md` for detailed stage descriptions, expected outputs, and troubleshooting.

---

## Data

Raw data is not committed to this repository. All datasets are publicly available; retrieval scripts are provided in `pipeline/01_data_harvesting/`.

| Dataset | Size | Source | Notes |
|---------|------|--------|-------|
| Rat scRNA-seq corpus | ~27 GB | GEO, ArrayExpress | 895 studies, ~10.2M cells |
| Rat bulk RNA-seq (MoTrPAC) | ~26 GB | [MotrpacRatTraining6moData](https://github.com/MoTrPAC/MotrpacRatTraining6moData) | 18 tissues, 8 timepoints |
| BioMart reference | ~5 MB | [Ensembl BioMart](https://www.ensembl.org/biomart) | Gene annotations, orthologs |
| GeneCompass pretrained | ~2 GB | [xCompass-AI/GeneCompass](https://github.com/xCompass-AI/GeneCompass) | Human + mouse weights |

### Obtaining data

```bash
# Download BioMart reference files
python pipeline/01_data_harvesting/fetch_biomart_reference_data.py

# Download and preprocess rat scRNA-seq from GEO
python pipeline/01_data_harvesting/harvest_geo.py

# Stage 1 output: data/training/qc_h5ad/
```

### Canonical gene universe

All pipeline outputs are indexed by canonical Ensembl rat gene IDs from **BioMart release 113, assembly mRatBN7.2**. Noncanonical identifiers (symbols, retired IDs, RGD numbers) are resolved via deterministic mapping rules in `lib/gene_utils.py`. Unresolved identifiers are excluded with audit logs. See `docs/ARCHITECTURE.md` for details.

---

## Configuration

All tunable parameters live in `config/pipeline_config.yaml`. Scripts read policy from this file; no hardcoded thresholds or paths in code.

```bash
# Override project root via environment variable
export PIPELINE_ROOT=/your/data/root

# Key settings
cat config/pipeline_config.yaml
```

See `config/pipeline_config.yaml` for full documentation of each parameter.

---

## Project structure

```
motrpac-genecompass/
├── config/                     # Pipeline configuration
│   └── pipeline_config.yaml
├── lib/                        # Shared Python utilities
│   ├── gene_utils.py           #   Gene normalization, BioMart loader
│   ├── io_utils.py             #   h5ad/sparse matrix helpers
│   └── manifest.py             #   Stage manifest creation
├── vendor/
│   └── GeneCompass/            # Git submodule (upstream fork)
├── pipeline/                   # Aim 1: Data preprocessing (Stages 1–7)
│   ├── 01_data_harvesting/
│   ├── 02_gene_inventory.py
│   ├── 03_vocabulary_pruning.py
│   ├── 04_ortholog_mapping.py
│   ├── 05_compute_gene_medians.py
│   ├── 06_create_reference_files.py
│   └── 07_export_training_corpus.py
├── finetune/                   # Aim 1: Model fine-tuning
├── deconvolution/              # Aim 1: Bulk-SC integration
├── analysis/                   # Aim 2: Downstream analysis
├── translation/                # Aim 3: Cross-species translation
├── slurm/                      # SLURM job scripts
├── tests/                      # Unit and integration tests
├── notebooks/                  # Jupyter notebooks
├── docs/                       # Architecture docs, bug tracker
└── data/                       # NOT committed (see .gitignore)
```

---

## Dependencies

### External tools (git submodule or pip)

| Tool | Purpose | Integration |
|------|---------|-------------|
| [GeneCompass](https://github.com/xCompass-AI/GeneCompass) | Foundation model | Git submodule in `vendor/` |
| [UniCell](https://github.com/dchary/ucdeconvolve) | Deconvolution | Wrapper in `deconvolution/` |
| [scDEAL](https://github.com/OSU-BMBL/scDEAL) | Domain transfer learning | Wrapper in `deconvolution/` |
| [Scissor](https://github.com/sunduanchen/Scissor) | Phenotype-cell linking | Wrapper in `deconvolution/` |
| [DeepSEM](https://github.com/HantaoShu/DeepSEM) | GRN inference | Wrapper in `analysis/` |

### Python (core)

```
python >= 3.10
torch >= 2.0
scanpy >= 1.9
anndata >= 0.9
pandas >= 2.0
scipy >= 1.10
pyyaml
h5py
```

---

## Team

| Role | Name | Focus |
|------|------|-------|
| PI | Tim Reese | ML/AI development, GeneCompass adaptation, pipeline engineering |
| Co-I | Boran Gao | Human genetics integration (UK Biobank, All of Us) |
| Co-I | Fei Xue | Statistical methodology |
| Co-I | Geyu Zhou | Deconvolution, biological interpretation |
| Advisory | Yu Michael Zhu | ML methodology, experimental design |
| GRA | Yang He | Pathway enrichment, FM output interpretation |
| Consultant | Tim Gavin | Exercise physiology domain expertise |

---

## Citation

If you use this work, please cite:

> Reese TG et al. (2026). Development of Advanced Analytical Tools Integrating External Datasets with MoTrPAC Data. NIH RFA-RM-24-011.

And the GeneCompass paper:

> Yang X et al. (2024). GeneCompass: Deciphering Universal Gene Regulatory Mechanisms with Knowledge-Informed Cross-Species Foundation Model. *Cell Research*.

---

## License

[TBD — MIT or Apache 2.0]

GeneCompass submodule retains its original license. See `vendor/GeneCompass/LICENSE`.