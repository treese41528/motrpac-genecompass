# Deconvolution setup & first run (Stages 8–9)

This is the authoritative runbook for the deconvolution **Stages 8–9** (the Aim-2 bridge:
MoTrPAC bulk → BayesPrism → pseudo-cells → fine-tuned rat GeneCompass → embeddings). It
covers the environment, the per-site profile, the data/config a fresh clone needs, and how
to run the orchestrators.

This stage needs an R toolchain: **BayesPrism** (Danko-Lab, vendored) for the primary
deconvolution, the **omnideconv** multi-method panel (MuSiC / DWLS / SCDC / Bisque, plus the
Python methods **AutoGeneS / Scaden** via reticulate) for the cross-check, and **DESeq2** for
the paper-faithful VST scoring metric. The pipeline also has Python pieces (the `*.py`
helpers + the R↔Python `reticulate` bridge); the **container path bundles the full repo
Python env** (`requirements.txt`) alongside R, so it is the entire environment in one image.

The hard part is **not** the R packages — it's the *system libraries* they compile
against (libpng, openssl, libxml2, …). On a laptop with `sudo` that's one `apt install`;
on a locked-down HPC cluster it's the whole battle. So pick your path by how much control
you have over the machine:

| Your situation | Use | Why |
|---|---|---|
| HPC cluster (no root) | **Container — Apptainer** | ships the entire system layer; nothing to install |
| Laptop / workstation / cloud / CI | **Container — Docker** | same image, reproducible anywhere |
| You must run on the bare cluster R | **Bootstrap script** | loads modules, tells you exactly which system libs you need |

The R side is driven by one pinned manifest, [`r_packages.yaml`](r_packages.yaml),
consumed by both the containers and the bootstrap — so all paths install the *same* R
versions (captured 2026-06-02, R 4.4.1 / Bioconductor 3.20). The Python side is the
repo-root [`requirements.txt`](../../requirements.txt) (Python 3.12).

> First, get the vendored submodules (all stages need them):
> ```bash
> git submodule update --init --recursive
> #   BayesPrism -> Stage 8 deconvolution | GeneCompass -> Stage 9 embed
> #   MotrpacRatTraining6moData -> MoTrPAC bulk + training data
> ```

---

## First-time setup checklist

The fastest portable path is the **container** — it bakes in R + every system lib + the
Python env, so there are **zero modules** to configure and the bare-metal `module load`
dance below is an HPC-only escape hatch you skip entirely. Either way:

1. **Clone with submodules** — `git clone --recurse-submodules …`, or in an existing clone
   `git submodule update --init --recursive`.
2. **Environment** — pick ONE:
   - *Container* (recommended; HPC / laptop / CI): build the image (§Path 1) — done, skip to step 4.
   - *Bare metal*: build the Python venv (§Python environment) **and** the R stack
     (`bash deconvolution/setup/install_r_env.sh`, §Path 2).
3. **Site profile** (bare-metal *module cluster* only): `cp deconvolution/setup/site.env.gilbreth.example
   deconvolution/setup/site.env` and edit it for your machine (§Site variables). On a laptop
   with R on PATH, or inside the container, you can skip this — the wrappers use the R/python
   already on PATH.
4. **Config override** — `cp config/pipeline_config.local.yaml.example config/pipeline_config.local.yaml`
   and repoint any data paths to where your inputs live (§Data & config). Set `PIPELINE_ROOT`
   to your **clone root** (not a data root): `export PIPELINE_ROOT=$PWD`.
5. **Data** — stage the MoTrPAC bulk/training data:
   `MGC_MOTRPAC_DATA_STORE=/your/store bash deconvolution/R/install_motrpac_data.sh` (~18 GiB),
   or point `deconvolution.motrpac_bulk_dir` at the vendored `vendor/MotrpacRatTraining6moData/data`.
   Stage 8 references also need the SC corpus, and Stage 9 needs the fine-tuned model — see
   §Data & config for what a fresh clone does and doesn't ship.
6. **Validate + run** — `python -c "from lib.gene_utils import load_config, validate_config;
   validate_config(load_config()); print('Config OK')"`, then
   `python pipeline/run_stage8.py --tissue … --ref-dir … --dry-run` (§Run the pipeline).

---

## Path 1 — Container (recommended)

The image bakes in the system libs, every R package, **and the full repo Python env**
(`requirements.txt`, Python 3.12 via `uv`, with `reticulate` wired to it). One image for
the whole repo — build once, run anywhere.

### Docker (laptop / cloud / CI)
```bash
# from the repo root (submodules must be checked out first)
docker build -f deconvolution/setup/Dockerfile -t motrpac-genecompass:1.0 .

# R step (deconvolution), mounting the repo at /work
docker run --rm -v "$PWD":/work -w /work motrpac-genecompass:1.0 \
  Rscript deconvolution/R/run_omnideconv.R \
    deconvolution/validation_v2/WAT_holdout/reference \
    deconvolution/validation_v2/WAT_holdout/mixtures \
    deconvolution/validation_v2/WAT_holdout/results

# Python step (any repo script) — same image
docker run --rm -v "$PWD":/work -w /work motrpac-genecompass:1.0 \
  python deconvolution/score_validation.py --stage-dir deconvolution/validation_v2/WAT_holdout
```

### Apptainer / Singularity (HPC)
```bash
# build directly (most clusters allow --fakeroot; no Docker needed)
apptainer build --fakeroot genecompass.sif deconvolution/setup/Apptainer.def
#   ...or convert an existing Docker image:
#   apptainer build genecompass.sif docker-daemon://motrpac-genecompass:1.0

# run (Apptainer auto-mounts $PWD and $HOME); R or Python both work
apptainer exec genecompass.sif Rscript deconvolution/R/run_omnideconv.R <ref> <mix> <out>
apptainer exec genecompass.sif python pipeline/01_data_harvesting/geo_harvester.py ...
```
The deconvolution methods are CPU-only. For **Stage-7 GeneCompass training**, the bundled
torch is CUDA-capable — add `--nv` (Apptainer) / `--gpus all` (Docker) on an NVIDIA host.
(If your scheduler *mandates* a GPU allocation even for CPU jobs, that's a property of the
SLURM job, not the container; see Gotchas.)

---

## Path 2 — Bare-metal bootstrap

For running directly against a cluster's module R (or a laptop where you'd rather not
use a container). One command:

```bash
bash deconvolution/setup/install_r_env.sh
```

It sets up the toolchain, then runs [`install_r_packages.R`](install_r_packages.R), which
**prints the exact system-dependency install command for your OS** (via `pak::pkg_sysreqs`)
before installing — so you're never guessing what `png.h` you're missing.

**On a module/conda cluster**, point it at your modules first via the site profile:
```bash
cp deconvolution/setup/site.env.gilbreth.example deconvolution/setup/site.env
$EDITOR deconvolution/setup/site.env     # set R_MODULES / STRIP_CONDA / MGC_PYTHON / ...
bash deconvolution/setup/install_r_env.sh
```
`site.env` is gitignored and cluster-specific; the committed
[`site.env.gilbreth.example`](site.env.gilbreth.example) documents the Purdue Gilbreth
values as a template (see §Site variables for every knob).

**On a laptop/server with R already installed**, no `site.env` is needed — just install
the system libs it reports (e.g. `sudo apt install libpng-dev libxml2-dev libssl-dev …`)
and re-run.

---

## Path 3 — Fully manual

If you want to do it by hand, the installer documents the recipe. The essential moves
(all encoded in `install_r_packages.R`, learned the hard way):

1. Use **`pak`** with `dependencies = NA` (hard deps only). `dependencies = TRUE` pulls
   omnideconv's *Suggests*, one of which — **BisqueRNA** — was removed from CRAN
   2025-06-02; a single unresolvable Suggests aborts pak's whole solve.
2. Install the method packages from the **`omnideconv/*` forks** (not upstream
   `xuranw/MuSiC` etc.) and Bisque from **`cozygene/bisque`** (the CRAN copy is gone;
   `limSolve` is back on CRAN so it resolves).
3. **Patch omnideconv's DESCRIPTION** to drop `devtools/pkgdown/knitr/rmarkdown/testthat`
   from Imports — they're for doc-building, absent from its NAMESPACE, and pull an
   unbuildable `ragg → systemfonts/textshaping` font stack.
4. Get the system libs from your package manager (`pkg_sysreqs` output) or, on a Spack
   cluster, the matching modules.

---

## Python environment

The `*.py` pipeline steps and the R `reticulate`/`anndata` bridge need Python. The
**container already contains the entire repo env** — `requirements.txt` (scanpy/anndata,
the Stage-1 harvesters, torch/transformers for Stage-7, …) installed into a Python 3.12
venv at `/opt/motrpac-venv`, with `RETICULATE_PYTHON` pointed at it. Nothing extra to do.

On **bare metal** (Path 2/3), set Python up next to the R stack:
```bash
python3.12 -m venv motrpac-env && source motrpac-env/bin/activate
pip install -U pip && pip install -r requirements.txt
```
`torch` pulls its default CUDA-capable wheel; for a CPU-only box install it slim first:
`pip install torch --index-url https://download.pytorch.org/whl/cpu`.

Point the R↔Python bridge and the orchestrators at this venv via `MGC_PYTHON` in `site.env`
(below), or simply `source motrpac-env/bin/activate` before running — the wrappers fall back
to the active venv (`$VIRTUAL_ENV`) or `python3` on PATH when `MGC_PYTHON` is unset.

---

## Site variables (`site.env`)

All machine-specific shell/cluster values live in one gitignored file,
`deconvolution/setup/site.env` (copy from `site.env.gilbreth.example`). It is sourced by
`deconvolution/setup/site_env.sh`, which **every** `R/*.sh` wrapper and the bootstrap
installer load. Every variable is optional — leave one empty and that step is a no-op, so the
defaults off a configured cluster are "use the R / python already on PATH" (exactly right for
the container and laptops).

| Variable | Controls | Default if unset |
|---|---|---|
| `R_MODULES` | Full Lmod module list to load (R + build-time libs), e.g. `r/4.4.1 libpng/1.6.37 zlib/1.3.1 cmake/3.30.2`. Loaded only if a `module` command exists. | none → R from PATH |
| `STRIP_CONDA` | Conda prefix to drop from PATH/PKG_CONFIG_PATH (not CPATH) so compiled R deps dlopen against the module toolchain. | none → no strip |
| `MGC_PYTHON` | Python for `reticulate` + the omnideconv Python methods; feeds `ENV_PY` and `RETICULATE_PYTHON`. | active venv → `python3` |
| `MGC_MOTRPAC_DATA_STORE` | Writable path (outside the repo, ≥18 GiB) where `install_motrpac_data.sh` stages the MoTrPAC data; gets symlinked to `data/motrpac/rat_training_6mo`. | author's `/depot` path (set this!) |

The wrappers reproduce the build/run env **conditionally**: modules load only when an Lmod
`module` command is present, the conda strip happens only when `STRIP_CONDA` is set. So the
same scripts work on Gilbreth, a different module cluster (swap the names), a laptop with R
on PATH (no `site.env`), and inside the container.

---

## Data & config

The committed `config/pipeline_config.yaml` is all-relative; machine-specific overrides go in
the gitignored `config/pipeline_config.local.yaml` (deep-merged by `load_config`). Start from
the template:
```bash
cp config/pipeline_config.local.yaml.example config/pipeline_config.local.yaml
export PIPELINE_ROOT=$PWD     # the CLONE root — NOT a data root
```

**What a fresh clone ships vs. needs.** On the original checkout, `data/` is a forest of
symlinks into an external store, so the repo-relative defaults "just work." A fresh clone has
an empty, gitignored `data/` — so you must supply the inputs and either recreate the symlinks
(`ln -s /your/store/... data/...`) or override the path keys in `pipeline_config.local.yaml`:

- **MoTrPAC bulk** (Stage 8 step 1): the `TRNSCRPT_<TISSUE>_RAW_COUNTS.rda` files ship in the
  vendored submodule. Either run `MGC_MOTRPAC_DATA_STORE=/your/store bash
  deconvolution/R/install_motrpac_data.sh` (installs the R package, mirrors ~789 MB, fetches
  ~18 GiB from GCS, and symlinks `data/motrpac/rat_training_6mo`), or just point
  `deconvolution.motrpac_bulk_dir: vendor/MotrpacRatTraining6moData/data`.
- **SC reference** (Stage 8 `--ref-dir`): per-tissue BayesPrism references are *built*
  (`deconvolution/build_all_references.sh`), not shipped, and that build needs the QC'd
  single-cell corpus + annotations (pipeline Stages 1–2 output). A bare clone cannot build a
  reference without first obtaining that corpus.
- **Fine-tuned model** (Stage 9): `deconvolution.genecompass_model_dir` points at the Stage-7
  rat fine-tune (gitignored). Run Stage 7, or pass `--model-dir` to `run_stage9.py`.

---

## Run the pipeline (after setup)

Production Stages 8–9 go through the orchestrators (see `deconvolution/README.md` for the
per-tissue detail and the tissue-code → reference-dir mapping):
```bash
source motrpac-env/bin/activate          # or set MGC_PYTHON in site.env

python pipeline/run_stage8.py --tissue SKM-GN \
    --ref-dir "data/deconvolution/references/skeletal muscle_GSE254371" --dry-run
python pipeline/run_stage9.py --label skmgn --dry-run
```

The individual R steps are also runnable directly via their wrappers (now portable — the
module/conda setup is a no-op off a module cluster):
```bash
# BayesPrism (primary)
bash deconvolution/R/run_deconvolution.sh <ref_dir> <mix_dir> <out_dir>

# omnideconv cross-check panel (default: music,dwls,scdc,bisque)
OMNIDECONV_METHODS=music,scdc,bisque \
  bash deconvolution/R/run_omnideconv.sh <ref_dir> <mix_dir> <out_dir>

# score any method's output vs ground truth
python deconvolution/score_validation.py --stage-dir <stage_dir> \
  --truth cellfrac --est-file results/fractions_music.csv --tag music
```
Inside a container (R + libs already present) you can also call `Rscript run_omnideconv.R …`
directly — the wrapper just adds the (now-conditional) cluster module/conda setup on top.

---

## Reproducibility

- **CRAN** is pinned to a **dated Posit Package Manager snapshot** (`CRAN_REPO`, default
  `…/2026-06-02`) — every CRAN package resolves to its version on that date.
- **Bioconductor** is pinned by the R version (R 4.4.x ⇒ Bioc 3.20 ⇒ DESeq2 1.46).
- **GitHub** packages are pinned by **commit SHA** in the manifest.
- To move forward deliberately: bump the date / SHAs, reinstall, then
  `Rscript deconvolution/setup/snapshot_r_packages.R` to print the new pins.

---

## Gotchas & FAQ

- **`png.h: No such file or directory`** (or any `*.h` not found): a missing system
  `-devel`/`-dev` package. Run the install — `pkg_sysreqs` lists exactly what to install;
  on a module cluster, `module load` the equivalent (add it to `R_MODULES` in `site.env`).
- **`module: command not found` / `/etc/profile.d/modules.sh: No such file`**: you're not on
  an Lmod cluster — leave `R_MODULES` empty (the wrappers then use R from PATH). This is the
  normal laptop/container case, not an error.
- **Killed / OOM during a run**: deconvolution loads the single-cell reference as a dense
  matrix (a 22k-cell ref is ~3 GB, before copies). Don't run on a login node — use a
  compute node / container with enough RAM (≥ 64 GB for large refs).
- **`No GPUs requested` from SLURM**: some clusters (incl. Gilbreth) mandate a GPU
  allocation even for CPU jobs. Attach a dummy `--gres=gpu:1` (see the `slurm/` scripts).
  The methods themselves never use the GPU.
- **DWLS is very slow**: it builds a signature via MAST differential expression per cell
  type — hours on a large reference. Give the SLURM job a long `--time` (e.g. `12:00:00`),
  or subsample the reference. MuSiC/SCDC/Bisque are far faster.
- **Subject-aware methods need ≥ 2 subjects**: MuSiC/SCDC/Bisque estimate cross-subject
  variance and error out on a single-sample reference. Use a reference with multiple
  `sample`s (check the `sample` column of `cells_meta.tsv`).
- **Why are scBio/CPM and MOMF missing?** They pull heavy system deps (terra→gdal/geos/proj,
  rgl→OpenGL) for two non-core extra methods. Deliberately excluded — see the manifest's
  `exclude_methods`. CDSeq is optional (slow; needs `harmony`).
- **Don't install `omnideconv/BayesPrism`**: it installs as package `BayesPrism` and would
  clobber the vendored Danko-Lab build. The panel cross-checks *our* BayesPrism against the
  *other* methods.
- **The image is large (several GB)**: the bundled `torch`/`transformers` (Stage-7
  training) dominate. For a deconvolution-only slim image, drop the
  torch/torchvision/accelerate/transformers lines from `requirements.txt` before building,
  or install CPU-only torch (above).
