# motrpac-genecompass

Cross-species transcriptomic analysis of exercise adaptations using the [GeneCompass](https://github.com/xCompass-AI/GeneCompass) foundation model, applied to [MoTrPAC](https://motrpac.org/) rat endurance exercise training data.

**NIH-funded project** (RFA-RM-24-011) — Purdue University Department of Statistics

---

## Overview

This project adapts GeneCompass — a foundation model pretrained on 120M+ human and mouse single-cell transcriptomes — to analyze rat bulk and single-cell RNA-seq data from the Molecular Transducers of Physical Activity Consortium (MoTrPAC). The goal is to uncover molecular mechanisms of exercise-induced adaptations and translate findings from rat models to human biology.

### What this repository contains

| Component | Location | Description |
|-----------|----------|-------------|
| **Data pipeline** | `pipeline/` | Staged orchestrators (`run_stageN.py`, see `pipeline/README.md`): corpus → tokens → fine-tuning (Stages 1–7), the deconvolution bridge (8–9), per-cell-type DE (10), rat→human transfer (12), GRN/mechanism (13) and hardening (14) |
| **GeneCompass** | `vendor/GeneCompass/` | Foundation model (upstream git submodule, used as a library; rat adaptations live in `finetune/` + `data/`) |
| **Fine-tuning** | `finetune/` | Continued pretraining on rat corpus |
| **Deconvolution** | `deconvolution/` | Bulk ↔ single-cell integration: **BayesPrism** (vendored) + the **omnideconv** panel are the implemented Stage 8–9 stack; UniCell/scDEAL/Scissor are secondary/planned methods |
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

# Create the Python environment (a venv, not conda)
python3.12 -m venv motrpac-env && source motrpac-env/bin/activate
pip install -U pip && pip install -r requirements.txt

# Set project root = your clone (NOT a data root); all config paths resolve under it
export PIPELINE_ROOT=$PWD

# Machine-specific path overrides (gitignored, deep-merged over the committed template)
cp config/pipeline_config.local.yaml.example config/pipeline_config.local.yaml

# Validate configuration
python -c "from lib.gene_utils import load_config, validate_config; c = load_config(); validate_config(c); print('Config OK')"
```

### Running the data pipeline

The pipeline runs in numbered stages. Each stage reads from the previous stage's output directory and writes a JSON manifest for reproducibility.

```bash
python pipeline/run_stage2.py --dry-run   # gene universe & cell QC
python pipeline/run_stage3.py             # ortholog mapping (cross-species token assignment)
python pipeline/run_stage4.py             # gene medians (scatter / gather; SLURM recommended)
python pipeline/run_stage5.py             # reference assembly & tokenization (SLURM recommended)
python pipeline/run_stage6.py             # prior-knowledge embeddings
python pipeline/run_stage7.py             # fine-tuning (GPU)
```

See `pipeline/README.md` for detailed stage descriptions, expected outputs, and troubleshooting.

### Running deconvolution (Stages 8–9, Aim-2 bridge)

```bash
# Every tissue with a reference, correct-by-construction: reads deconvolution/tissue_references.yaml (the canonical
# tissue→reference map), QC-gates every reference, submits Stage-8+9 with the right --ref-dir.
python pipeline/run_deconv_all.py --dry-run      # validate refs + print the plan
python pipeline/run_deconv_all.py --submit       # sbatch per tissue
# ...or one tissue manually:
python pipeline/run_stage8.py --tissue SKM-GN --ref-dir <built ref dir> --dry-run
python pipeline/run_stage9.py --label skmgn --dry-run
```

**Reference integrity:** references are chosen from `deconvolution/tissue_references.yaml` and gated by
`deconvolution/reference_qc.py` (blocks spatial/engineered/developmental/mixed samples — the class of bug
behind the liver-Visium / engineered-lung fixes). Cross-method validation + the mRNA-bias analysis +
which downstream claims to trust are in `deconvolution/OMNIDECONV_RESULTS.md`.

First-time deconvolution setup — the R/BayesPrism environment, the per-site profile, data
staging, and config — is documented in **`deconvolution/setup/SETUP.md`**; see
`deconvolution/README.md` for the runbook.

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

All pipeline outputs are indexed by canonical Ensembl rat gene IDs from the project's BioMart files. The config and manifests label them release 113 / mRatBN7.2, but they are **Ensembl release 115, GRCr8** (byte-identical to `sep2025.archive.ensembl.org`; 43,360 genes; identified 2026-10-05). The pipeline is internally consistent on those files; never re-fetch from the release-113 archive. Noncanonical identifiers (symbols, retired IDs, RGD numbers) are resolved via deterministic mapping rules in `lib/gene_utils.py`. Unresolved identifiers are excluded with audit logs. See `docs/ARCHITECTURE.md` for details.

---

## Configuration

All tunable parameters live in `config/pipeline_config.yaml`. Scripts read policy from this file; no hardcoded thresholds or paths in code.

```bash
# Override project root via environment variable (the repo/clone root, NOT a data root)
export PIPELINE_ROOT=$PWD

# Key settings
cat config/pipeline_config.yaml
```

See `config/pipeline_config.yaml` for full documentation of each parameter.

**Which model.** The default config uses the retrained **v2** model, its corpus, medians and output folders
(`docs/RETRAIN_V2_RESULTS.md`). To run with the April model and its folders instead, set a profile, which
`lib/gene_utils.load_config` merges over the config (after any `pipeline_config.local.yaml`):

```bash
export PIPELINE_PROFILE=config/profiles/april.yaml
```

**Which references.** Since 2026-10-31 the default deconvolution references are the jointly re-annotated
ones (`references_v4/`, listed per tissue in `deconvolution/tissue_references.yaml`; BLOOD is a whole-blood
composite), and every Stage 8–14 output lives in a `_refs4` folder (`deconvolution/REFS4_RERUN_RESULTS.md`).
The July consensus-label references and their v2 results are `config/profiles/july_refs.yaml`.

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
│   └── GeneCompass/            # Git submodule (tracks upstream xCompass-AI/GeneCompass)
├── pipeline/                   # Stage orchestrators run_stage1..14.py (no Stage 11) + run_deconv_all.py
│   ├── 01_data_harvesting/     #   Stage 1 workers (harvest + QC)
│   ├── 02_gene_universe/       #   Stage 2
│   ├── 03_ortholog_mapping/    #   Stage 3
│   ├── 04_gene_medians/        #   Stage 4
│   ├── 05_tokenization/        #   Stage 5
│   └── 06_prior_knowledge/     #   Stage 6
├── finetune/                   # Aim 1: Model fine-tuning
├── deconvolution/              # Aim 1: Bulk-SC integration
├── analysis/                   # Aim 2: Downstream analysis
├── translation/                # Aim 3: Cross-species translation
├── slurm/                      # SLURM job scripts
├── tests/                      # pytest suite (see Tests)
├── notebooks/                  # Jupyter notebooks
├── docs/                       # Architecture docs, bug tracker
└── data/                       # NOT committed (see .gitignore)
```

---

## Tests

```bash
pytest                 # offline tests, about 10 s; no data or internet needed
pytest -m data         # checks against this machine's pipeline outputs (corpus files, v2 results)
pytest -m network      # live NCBI / EBI / ENA checks (set NCBI_EMAIL for the Entrez one)
```

The offline tests cover each stage's core logic on small made-up inputs: harvester file
filters, gene-ID handling, the tokenizer, pseudo-cell building, the gate statistics, the
rat-to-human transfer and config layering. The data tests read whichever model the config
selects (`PIPELINE_PROFILE=config/profiles/april.yaml` for April).

---

## Dependencies

### External tools (git submodule or pip)

| Tool | Purpose | Integration |
|------|---------|-------------|
| [GeneCompass](https://github.com/xCompass-AI/GeneCompass) | Foundation model | Git submodule in `vendor/` |
| [BayesPrism](https://github.com/Danko-Lab/BayesPrism) | Deconvolution (primary) | Vendored submodule `vendor/BayesPrism` |
| [omnideconv](https://github.com/omnideconv/omnideconv) | Deconvolution cross-check panel | MuSiC/DWLS/SCDC/Bisque + AutoGeneS/Scaden (`deconvolution/`) |
| [UniCell](https://github.com/dchary/ucdeconvolve) | Deconvolution | Secondary / planned |
| [scDEAL](https://github.com/OSU-BMBL/scDEAL) | Domain transfer learning | Secondary / planned |
| [Scissor](https://github.com/sunduanchen/Scissor) | Phenotype-cell linking | Secondary / planned |
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

> Reese TG et al. (2026). Development of Advanced Analytical Tools Integrating External Datasets with MoTrPAC Data: Utilizing Statistical, Machine Learning, and AI Approaches to Uncover Molecular Mechanisms of Physical Activity. NIH RFA-RM-24-011.

And the GeneCompass paper:

> Yang X et al. (2024). GeneCompass: Deciphering Universal Gene Regulatory Mechanisms with Knowledge-Informed Cross-Species Foundation Model. *Cell Research*.

---

## License

This project is licensed under the Apache License 2.0 — see [LICENSE](LICENSE) for details.

GeneCompass submodule retains its original license. See `vendor/GeneCompass/LICENSE`.