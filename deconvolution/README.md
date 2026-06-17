# Deconvolution (grant Aim 2 bridge)

MoTrPAC bulk RNA-seq → **BayesPrism** per-cell-type deconvolution → per-cell-type
**pseudo-cells** → fine-tuned **rat GeneCompass** → 768-d cell embeddings. See
`AIM2_DECONV_RESULTS.md` for the science and `MOTRPAC_BULK_LIFTOVER.md` for the
bulk gene-ID prep.

## Pipeline (production path)

Two orchestrators in `pipeline/` drive the scripts in this folder in place
(nothing is moved), in the same `run_stageN.py` style as the rest of the pipeline:

| Stage | Orchestrator | Chain (scripts in this folder) |
|---|---|---|
| **8 — Deconvolution** | `pipeline/run_stage8.py` | `R/prepare_motrpac_bulk.sh` → `R/run_deconvolution.sh` → `R/extract_z.sh` → `build_pseudocells.py` |
| **9 — Tokenize + Embed** | `pipeline/run_stage9.py` | `tokenize_pseudocells.py` → `finetune/genecompass/embed_cells.py` |

Stage 8 = bulk → `pseudocells.h5ad`; Stage 9 = pseudo-cells → `cell_embeddings.npy`.
Both are per-tissue; each prints its plan with `--dry-run` and supports `--from N`.

```bash
# one tissue, end to end (step 2 of stage 8 is CPU-heavy -> compute node;
# stage 9 embed -> GPU node):
python pipeline/run_stage8.py --tissue SKM-GN \
    --ref-dir "data/deconvolution/references/skeletal muscle_GSE254371"
python pipeline/run_stage9.py --label skmgn
```

You supply `--tissue` (MoTrPAC bulk code, matches `TRNSCRPT_<TISSUE>_RAW_COUNTS.rda`)
and `--ref-dir` (the per-tissue SC reference). The tissue code differs from the reference
dir name (e.g. `SKM-GN` → `skeletal muscle_GSE254371`, `BLOOD` → `peripheral blood
mononuclear cells_GSE285476`, `HEART` → `heart_GSE280111_LV`); the full per-tissue mapping
is the `JOBS` table in `build_all_references.sh`. References are **built, not shipped**
(`build_all_references.sh`, from the SC corpus), so a fresh clone must set up data first —
see [`setup/SETUP.md`](setup/SETUP.md) (Data & config). All other paths default from
`config/pipeline_config.yaml` `[deconvolution]`. Outputs land under `data/deconvolution/`
(gitignored). The per-tissue SLURM job (`slurm/`, local) remains the production driver for
the heavy steps; the orchestrators run the chain serially for one tissue (local /
single-node) and document the SLURM split.

## Not in the pipeline (exploration / analysis)

These are run on demand, separately from the pipeline:

- **Aim-2 analysis:** `pheno_merge_test.py` (gate), `subspace_probe.py`,
  `augur_prep.py` + `R/run_augur.R` + `corroborate_summary.py`, `embed_qc.py`,
  `build_umap_viewer.py` (+ `umap_viewer_template.html`, `umap_embeddings.py`).
- **Deconvolution validation:** `make_pseudobulk.py`, `make_purity_sweep.py`,
  `score_purity_sweep.py`, `score_validation.py`, `score_z.py`, `compute_true_z.py`,
  `R/score_z_vst.R`, `R/run_tutorial_gbm.R`, `build_all_datasets.sh`.
- **Cross-method θ check:** `R/run_omnideconv.R` (see `OMNIDECONV_BENCHMARK_PLAN.md`).
- **Reference / setup:** `build_reference.py`, `build_all_references.sh`,
  `build_protein_coding_list.py`, `build_sex_chrom_list.py`, `audit_idspace.py`,
  `setup/` (container + bootstrap). References are inputs to Stage 8.

## Setup & environment

**First-time setup (env, site profile, data, config): see
[`setup/SETUP.md`](setup/SETUP.md)** — the authoritative Stage 8–9 runbook.

- **Python** steps run under the `motrpac-env` venv (`source` it, or set `DECONV_PYTHON` /
  `MGC_PYTHON`).
- **R** steps run via their `R/*.sh` wrappers, which set `R_LIBS_USER=R_libs/` and source
  `setup/site_env.sh` to reproduce the build env from the per-site `setup/site.env` (load
  `R_MODULES`, strip `STRIP_CONDA`, resolve the python). All of it is **conditional**, so the
  wrappers are portable — a no-op off a module cluster / inside the container. Prefer the
  wrappers over the `.R` files directly.
- **Config** is shared across both languages: Python via `lib/gene_utils.load_config`,
  R/bash via `_config_sh.py` (emits `CFG_*` shell vars from the same YAML). Machine-specific
  path overrides go in the gitignored `config/pipeline_config.local.yaml` (template:
  `config/pipeline_config.local.yaml.example`); cluster/python paths go in `setup/site.env`.
