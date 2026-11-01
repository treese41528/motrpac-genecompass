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

# Create the Python environment (a venv, not conda) from the EXACT lock (--no-deps: every
# transitive dependency is pinned there; requirements.txt is only the ranged, human-readable list)
python3.12 -m venv motrpac-env && source motrpac-env/bin/activate
pip install -U pip && pip install --no-deps -r requirements.lock.txt

# Set project root = your clone (NOT a data root); all config paths resolve under it
export PIPELINE_ROOT=$PWD

# Machine-specific path overrides (gitignored, deep-merged over the committed template)
cp config/pipeline_config.local.yaml.example config/pipeline_config.local.yaml

# Validate configuration
python -c "from lib.gene_utils import load_config, validate_config; c = load_config(); validate_config(c); print('Config OK')"
```

### Rebuilding the v2 model (Stages 0–7), in order

Every stage reads the previous stage's output under `data/` (paths in `config/pipeline_config.yaml`) and
writes a manifest or a `provenance_*.json` (git commit, config, input sizes and md5s). Runtimes are from the
2026-10-31 reproduction (`docs/repro_stages1_7/REPRODUCTION.md`, which also says what reproduces byte for
byte and what does not). "Deposited" marks the outputs that are deposited with v2 because a rerun cannot
give the same bytes; to reproduce v2 exactly, restore those and start from the next step.

```bash
# ── 0. Clone + environment ────────────────────────────────────────────────────────────────────────────
git clone --recurse-submodules https://github.com/treese41528/motrpac-genecompass.git && cd motrpac-genecompass
python3.12 -m venv motrpac-env && source motrpac-env/bin/activate
pip install --no-deps -r requirements.lock.txt          # or the container: deconvolution/setup/Dockerfile
export PIPELINE_ROOT=$PWD
bash scripts/stage_genecompass_assets.sh                # GeneCompass_Base, prior_knowledge, randsel_500w_human/mouse:
                                                        # prints the manual download links, applies the mouse
                                                        # state.json edit v2 used, checks config/genecompass_assets_md5.tsv

# ── Stage 0. Reference files (data/references/biomart/, ~170 MB, minutes) ──────────────────────────────
python pipeline/01_data_harvesting/fetch_biomart_reference_data.py -v
#   BioMart: the Ensembl 115 / GRCr8 archive (sep2025.archive.ensembl.org). RGD GENES_RAT.txt has no archive:
#   a fetch today gets a newer file. DEPOSITED: the reference folder (md5s: config/reference_md5sums.tsv;
#   Stage 2 refuses other files unless --allow-reference-drift). HGNC (Stage 6) is the dated 2026-03-06
#   archive if absent; v2's local copy (2026-03-17) is not in the archive and is deposited.

# ── Stage 1. Harvest (data/raw/, ~27 GB of matrices; network-bound, many hours) ────────────────────────
python pipeline/01_data_harvesting/geo_harvester.py --mode both --download --resume --batch-esummary 400 \
    --workers 8 --accessions config/stage2_step1_accessions.txt --output_dir data/raw/geo
mkdir -p data/raw/geo/single_cell data/raw/geo/bulk     # Stage 2 reads both config source paths; as for v2,
ln -s ../geo_datasets data/raw/geo/single_cell/geo_datasets   # both point at the one harvest folder
ln -s ../geo_datasets data/raw/geo/bulk/geo_datasets
#   Replay mode (--mode both, as the February 2026 harvest): exactly the 733 GEO series Stage 2 step 1
#   scanned (config/corpus_v2_accessions.txt = the 80 that v2's corpus keeps). md5s are recorded by default.
#   GEO files can change after the February 2026 harvest, so a replay is not guaranteed byte-identical.
#   OMIX767 (rat testis, NGDC) has no downloader: download the 10x matrices by hand from
#   https://ngdc.cncb.ac.cn/omix/release/OMIX767 (files under https://download.cncb.ac.cn/OMIX/OMIX767/)
#   into data/raw/ngdc/datasets/OMIX767/Gene_bc_matrices/{C,E7W}_gene_bc_matrices/.
python pipeline/run_stage1.py --from 4                  # metadata, matrix analysis, LLM study analysis, catalog
#   -> data/catalog/unified_studies.json (the study list Stage 2 reads). The LLM step (Claude Haiku) is not
#   deterministic: DEPOSITED: data/catalog/.

# ── Stage 2. Gene universe + cell QC (data/training/gene_universe/, qc_h5ad/; ~23 h, 16 CPUs, 200 GB) ──
python pipeline/run_stage2.py                           # step 1 universe, step 2 QC h5ads, step 3 pruning
python pipeline/02_gene_universe/qc_external_mtx.py \
    --mtx-dir data/raw/ngdc/datasets/OMIX767/Gene_bc_matrices/C_gene_bc_matrices --sample-id OMIX767_grcr8_sampleC
python pipeline/02_gene_universe/qc_external_mtx.py \
    --mtx-dir data/raw/ngdc/datasets/OMIX767/Gene_bc_matrices/E7W_gene_bc_matrices --sample-id OMIX767_grcr8_sampleE7W
#   -> 868 QC h5ads (config/corpus_v2_plan.tsv: one row each with md5 and n_cells; 584 kept) and
#   pruned_gene_universe.tsv (22,213 genes). DEPOSITED: both. A rerun differs (187 of the kept files and
#   the universe by a few genes; the 27 keep_raw_slot files come from a cancelled February run and are not
#   produced at all). Check a restored set: md5s in config/corpus_v2_plan.tsv.

# ── Stage 3. Ortholog mapping + token dictionary (data/training/ortholog_mappings/; seconds) ───────────
python pipeline/run_stage3.py
python finetune/genecompass/build_rat_token_dictionary.py     # -> rat_human_mouse_tokens.pickle (55,275)
#   Reproduces byte for byte from the deposited pruned universe.

# ── Stage 4. Gene medians (data/training/gene_medians_v2/; scatter 5 min x 16 tasks in parallel, gather 47 min)
python pipeline/run_stage4.py                           # runs medians.n_scatter_tasks = 16 tasks, then gather
python pipeline/05_tokenization/build_hybrid_medians.py # -> hybrid_gene_medians.pickle (seconds)
#   On a cluster, run compute_gene_medians.py --task-id $i (i = 0..15) as an array, then gather_gene_medians.py.

# ── Stage 5. Tokenized corpus (data/training/tokenized_corpus_v2/, 209 GB; 50 tasks ~21 min on 16 CPUs + assemble 22 min)
python pipeline/run_stage5.py                           # reference_assembly.n_tokenize_tasks = 50 tasks, then assemble
#   The task count fixes the row order of the corpus: keep 50. Reproduces byte for byte.

# ── Stage 6. Prior knowledge (data/training/prior_knowledge/) ───────────────────────────────────────────
python pipeline/run_stage6.py --from 2                  # family (8 min; deterministic since 2026-11-01),
                                                        # promoter (GPU, ~7 min; GRCr8 FASTA + DNABERT pinned;
                                                        #   needs `pip install pyfaidx`, not in the lock),
                                                        # GRN transfer (seconds)
python pipeline/run_stage6.py --from 1                  # + co-expression (~49 h; refuses the default config)
#   v2's init takes only 36 T4 GRN rows from Stage 6 (GRN reproduces byte for byte). DEPOSITED: the whole folder
#   (the family vectors were built with 8 gensim threads and cannot be regenerated).

# ── Stage 7. Init, mix, probe, training (data/models/, data/training/) ─────────────────────────────────
python pipeline/run_stage7.py --from 1 --to 1           # init (seconds): rat_model_init.py --seed 0 (initialization.seed)
#   DEPOSITED: data/models/rat_genecompass_init_v2 (built unseeded: 231/234 tensors reproduce, the 3 random ones do not)
python pipeline/run_stage7.py --from 2 --to 2           # training mix (4.7 h, 546 GB; seed 42; 40/45/15 rat/human/mouse)
python pipeline/run_stage7.py --from 3 --to 3           # evaluation probe (1.6 h; seed 20261008; md5s: data/training/eval_probe_v2.md5)
#   Both reproduce byte for byte (state.json differs only in the HF fingerprint).
PHASE=1 bash scripts/train_stage7.sh                    # phase 1: 517,793 steps on 4 x A100-40GB (about a week); rerun to resume
PHASE=2 bash scripts/train_stage7.sh                    # phase 2: 591,763 steps
#   (= python pipeline/run_stage7.py --from 4 --phase N). GPU kernels are nondeterministic: a restart tracks
#   v2's loss to 4 decimals but is not bit-identical. DEPOSITED: the phase 1 and phase 2 checkpoints.
```

`pipeline/README.md` describes each stage. SLURM wrappers for this cluster are kept locally (`slurm/`, not in
git); `scripts/train_stage7.sh` is a scheduler-free copy of the training launch v2 used (wrap it in your own
batch script). `run_stage7.py` checks that all four Stage 6 files and the Stage 5 corpus exist before any step.

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
| Rat scRNA-seq corpus | ~27 GB | GEO (+ OMIX767 from NGDC) | v2: 584 QC files from 81 studies, 6,803,206 cells (`config/corpus_v2_plan.tsv`, 868 files listed) |
| Rat bulk RNA-seq (MoTrPAC) | ~26 GB | [MotrpacRatTraining6moData](https://github.com/MoTrPAC/MotrpacRatTraining6moData) | 18 tissues, 8 timepoints |
| BioMart + RGD reference | ~170 MB | [Ensembl 115 archive](https://sep2025.archive.ensembl.org/biomart), [RGD](https://download.rgd.mcw.edu/data_release/) | Gene annotations, orthologs, symbols (md5s: `config/reference_md5sums.tsv`) |
| GeneCompass base assets | ~2 GB + 0.5M-cell sets | [xCompass-AI/GeneCompass](https://github.com/xCompass-AI/GeneCompass) | Weights, prior knowledge, human/mouse cells (`scripts/stage_genecompass_assets.sh`) |

### Obtaining data

See "Rebuilding the v2 model" above: Stage 0 (`fetch_biomart_reference_data.py`), Stage 1
(`geo_harvester.py --accessions ...`, plus the manual OMIX767 download) and the GeneCompass assets
(`scripts/stage_genecompass_assets.sh`). Stage 1 writes raw matrices to `data/raw/`; the QC h5ads in
`data/training/qc_h5ad/` are Stage 2 outputs.

### Canonical gene universe

All pipeline outputs are indexed by canonical Ensembl rat gene IDs from the project's BioMart files: **Ensembl release 115, GRCr8** (byte-identical to `sep2025.archive.ensembl.org`; 43,360 genes; identified 2026-10-05). Manifests written before 2026-11-01 label them release 113 / mRatBN7.2, which was wrong; the config now says 115 / GRCr8 and the fetcher uses the 115 archive. Never re-fetch from the release-113 archive or the live host. Noncanonical identifiers (symbols, retired IDs, RGD numbers) are resolved via deterministic mapping rules in `lib/gene_utils.py`. Unresolved identifiers are excluded with audit logs. See `docs/ARCHITECTURE.md` for details.

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
composite). Since 2026-11-01 the default covers all 16 deconvolved tissues (SPLEEN and OVARY added) and every
Stage 8–14 output lives in a `_refs4all` folder (`deconvolution/ALL_TISSUE_RERUN_RESULTS.md`); the 14-tissue
refs4 arm (`_refs4` folders, `deconvolution/REFS4_RERUN_RESULTS.md`) is `config/profiles/refs4_14.yaml`. The July consensus-label references and their v2 results are `config/profiles/july_refs.yaml`.

---

## Project structure

```
motrpac-genecompass/
├── config/                     # Pipeline configuration
│   ├── pipeline_config.yaml
│   ├── corpus_v2_plan.tsv      #   the v2 corpus file list (keep/drop, md5, n_cells)
│   ├── corpus_v2_accessions.txt, stage2_step1_accessions.txt   # Stage 1 replay lists
│   ├── reference_md5sums.tsv, genecompass_assets_md5.tsv      # pinned inputs
│   └── profiles/               #   April / reference-arm overlays
├── lib/                        # Shared Python utilities
│   ├── gene_utils.py           #   Config loading, gene normalization, BioMart loader, reference md5 checks
│   └── provenance.py           #   provenance_<script>.json writer (git, config, input md5s)
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
├── scripts/                    # train_stage7.sh (training launch), stage_genecompass_assets.sh
├── slurm/                      # local SLURM wrappers (gitignored, site-specific)
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