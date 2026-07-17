# Deconvolution (grant Aim 2 bridge)

MoTrPAC bulk RNA-seq → **BayesPrism** per-cell-type deconvolution → per-cell-type
**pseudo-cells** → fine-tuned **rat GeneCompass** → 768-d cell embeddings. See
`AIM2_DECONV_RESULTS.md` for the science and `MOTRPAC_BULK_LIFTOVER.md` for the
bulk gene-ID prep.

**Reference integrity — start here.** `tissue_references.yaml` is the canonical tissue→reference map
(single source of truth). `REFERENCE_QC.md` documents the build-time QC gate (`reference_qc.py`) and the
two reference bugs it now prevents: the **liver** Visium-spatial contamination (removed) and the
**engineered lung** reference (GSE178405 → native pooled `lung_native_pooled`, which gave lung a real
0→3 exercise-hotspot signal). Cross-method validation, the full **mRNA-bias analysis**, and **which
downstream claims to trust vs treat cautiously** are in `OMNIDECONV_RESULTS.md`. Analysis notebooks
(`../notebooks/pipeline8–12`) still need re-executing per `../notebooks/RERUN_EDITS.md`.

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
    --ref-dir "data/deconvolution/references_v3/MUSCLE_GSE137869_Y"
python pipeline/run_stage9.py --label skmgn
```

You supply `--tissue` (MoTrPAC bulk code, matches `TRNSCRPT_<TISSUE>_RAW_COUNTS.rda`)
and `--ref-dir` (the per-tissue SC reference). The tissue→reference mapping is
**`tissue_references.yaml`** — the canonical single source of truth (each tissue's correct reference +
study + build command). The bulk code differs from the reference dir name (e.g. `LUNG` →
`lung_native_pooled`, `LIVER` → `liver_GSE220075`, `BLOOD` → `peripheral blood mononuclear cells_GSE285476`).

**Recommended — from-scratch, correct-by-construction:** `pipeline/run_deconv_all.py` reads that
manifest, runs the QC gate (`reference_qc.py --fail`) on every reference — refusing a missing or
contaminated one, so a fresh run cannot silently use the wrong study — and submits Stage-8+9 for all 14
tissues with the right `--ref-dir`:

```bash
python pipeline/run_deconv_all.py --dry-run     # validate every reference + print the plan
python pipeline/run_deconv_all.py --submit      # sbatch one job per tissue (refs from the manifest)
# then the cross-tissue layer (once all tissues finish):
sbatch slurm/analysis/redetect_redE.slurm       # Stage 11 probe + Augur + hotspots + Stage 10 DE
sbatch slurm/analysis/run_stage12.slurm         # Stage 12 cross-species transfer
```

References are **built, not shipped** — `build_references_v2.sh` + `build_lung_pooled.py` (native lung),
and liver via `build_reference.py --study GSE220075 --tissue liver` (its QC gate auto-drops the Visium
samples). So a fresh clone must set up data first — see [`setup/SETUP.md`](setup/SETUP.md) (Data & config).
Other paths default from `config/pipeline_config.yaml` `[deconvolution]`; outputs land under
`data/deconvolution/` (gitignored).

## Not in the pipeline (exploration / analysis)

These are run on demand, separately from the pipeline:

- **Aim-2 analysis:** `pheno_merge_test.py` (gate), `subspace_probe.py`,
  `augur_prep.py` + `R/run_augur.R` + `corroborate_summary.py`, `embed_qc.py`,
  `build_umap_viewer.py` (+ `umap_viewer_template.html`, `umap_embeddings.py`).
- **Deconvolution validation:** `make_pseudobulk.py`, `make_purity_sweep.py`,
  `score_purity_sweep.py`, `score_validation.py`, `score_z.py`, `compute_true_z.py`,
  `R/score_z_vst.R`, `R/run_tutorial_gbm.R`, `build_all_datasets.sh`.
- **Cross-method θ check + mRNA-bias battery (COMPLETE):** `R/run_omnideconv.R` +
  `omnideconv_bench/` (the SimBu simulator `simulate_simbu.{R,sh}`, `bias_delta.py`, `dose_response.py`)
  → results + paper comparison + downstream-claims guidance in `OMNIDECONV_RESULTS.md`
  (plan: `OMNIDECONV_BENCHMARK_PLAN.md`).
- **Reference build + QC:** `build_reference.py` (with the built-in QC gate), `build_references_v2.sh`,
  `build_lung_pooled.py` (native-lung pool); **`reference_qc.py`** (the contamination gate — run
  `python deconvolution/reference_qc.py --all --deep` to audit every reference); **`tissue_references.yaml`**
  (the map). Plus `build_protein_coding_list.py`, `build_sex_chrom_list.py`, `audit_idspace.py`,
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

## CIBERSORTx on an HPC cluster (license token + IP binding)

CIBERSORTx is the one license-gated method in the cross-method panel (`R/run_omnideconv.R`,
method `cibersortx`). It is **closed-source** and distributed as a Docker image
([`cibersortx/fractions`](https://hub.docker.com/r/cibersortx/fractions)) gated by registration +
a per-account **token**.

**Links & licensing**
- Register + generate the token: <https://cibersortx.stanford.edu> (log in → *Account* / "Download Token").
- Docker image: `docker pull cibersortx/fractions` ([Docker Hub](https://hub.docker.com/r/cibersortx/fractions)).
- **License:** CIBERSORTx is provided by the Alizadeh/Newman labs (Stanford) and is **free for
  academic / non-profit / government, non-commercial use only**; commercial use requires a separate
  license from Stanford. Review the Terms of Use on the site before use — we redistribute **no**
  CIBERSORTx code or the container image in this repo (only our `run_omnideconv.R` wrapper); each user
  must register and obtain their own token. The `token.txt` and `.sif` are gitignored.
- **Cite:** Newman A.M. *et al.*, "Determining cell type abundance and expression from bulk tissues
  with digital cytometry," *Nat. Biotechnol.* 37, 773–782 (2019); and the original CIBERSORT:
  Newman A.M. *et al.*, *Nat. Methods* 12, 453–457 (2015).

Two HPC-specific gotchas make it awkward — neither is cluster-vendor-specific:

**1. The token is bound to the IP address that contacts the license server *at run time*.**
The CIBERSORTx container phones home to validate `(email, token, IP)` every run. The bound IP is
the **egress (NAT) IP** the license server sees — which on a cluster is usually *not* the IP your
browser used to generate the token. So a token minted on your laptop/login node fails on a compute
node with `Token/username/ip combination is invalid … your IP address has changed`.
To fix it, mint the token from a context that egresses via the **same** IP your jobs will use:
  - Find the compute egress IP from inside a batch job: `curl https://api.ipify.org`. Check it's
    **stable across nodes** (run it on several) — clusters often NAT all compute nodes through one IP,
    but login nodes and compute nodes typically differ, and some clusters round-robin.
  - Generate the token through a browser **tunneled to egress via that IP** — e.g. an SSH dynamic
    SOCKS proxy (`ssh -D <port> …`) through a node whose egress matches where jobs run, with the
    browser pointed at `socks5://127.0.0.1:<port>` (enable remote DNS). Confirm with
    `https://api.ipify.org` in the proxied browser before generating.
  - The token is **reusable as long as that egress IP is stable** (here it's good for ~6 months),
    so this is a one-time setup. Browser notes: Chromium honors the Windows/system cert store (like
    `curl`) and may need `--disable-quic` over a TCP SOCKS proxy; Firefox uses its own cert store
    (`security.enterprise_roots.enabled=true` to trust system roots) and may need OCSP soft-fail.

**2. Run the container with a writable HOME + clean env (Docker is usually unavailable on HPC → use
Apptainer/Singularity).** `apptainer pull docker://cibersortx/fractions` → `fractions_latest.sif`.
The container ships **its own R** for the nu-SVR signature-matrix build, which needs a writable
`$HOME` and an uncontaminated environment; running it `--no-home`/`--contain` (omnideconv 0.1.1's
default) breaks that R (`there is no package called 'e1071'` → empty matrix → abort 134). Run it as:
```
apptainer exec --cleanenv --home <writable_dir> \
  -B <input_dir>:/src/data -B <output_dir>:/src/outdir \
  fractions_latest.sif /src/CIBERSORTxFractions --single_cell TRUE \
  --username <email> --token <token> --refsample <sig> --mixture <bulk> ...
```
`--cleanenv` also drops any host `LD_PRELOAD`/`R_*` vars that would otherwise leak into the
container. `R/run_omnideconv.R` does all of this for you: it reads `CIBERSORTX_EMAIL` and
`CIBERSORTX_TOKEN` from the environment (so the token never lands in code or, with `verbose=FALSE`,
in logs), pulls/uses the `.sif` under `CIBERSORTX_SIF_DIR`, and patches omnideconv's container
command to the `--cleanenv --home` form above. Invoke with
`OMNIDECONV_METHODS=…,cibersortx CIBERSORTX_EMAIL=… CIBERSORTX_TOKEN=$(cat token.txt) … run_omnideconv.sh`.
