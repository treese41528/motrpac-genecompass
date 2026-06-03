# =============================================================================
# Container image for the MoTrPAC GeneCompass repo: the deconvolution R stack
# (BayesPrism + omnideconv panel + DESeq2) AND the full repo Python environment
# (requirements.txt -- scanpy/anndata, the Stage-1 harvesters, torch/transformers
# for Stage-7 GeneCompass training, etc.). One image for the whole repo.
#
# Laptop / cloud / CI path (Docker). For HPC use Apptainer.def (same recipe).
#
# BUILD CONTEXT = repo root, with the BayesPrism submodule checked out:
#   git submodule update --init vendor/BayesPrism
#   docker build -f deconvolution/setup/Dockerfile -t motrpac-genecompass:1.0 .
# RUN (mount the repo; R or Python entry points both work):
#   docker run --rm -v "$PWD":/work -w /work motrpac-genecompass:1.0 \
#     Rscript deconvolution/R/run_omnideconv.R <ref_dir> <mix_dir> <out_dir>
#   docker run --rm -v "$PWD":/work -w /work motrpac-genecompass:1.0 \
#     python pipeline/01_data_harvesting/geo_harvester.py ...
# GPU (Stage-7 training): the default torch wheel is CUDA-capable; run with
#   `--gpus all` on an NVIDIA host. See SETUP.md to build a slimmer CPU-only image.
# =============================================================================
ARG R_VERSION=4.4.1
FROM rocker/r-ver:${R_VERSION}

# --- System libraries (R stack + a few the Python wheels link against) ----------
# Post-patch the R tree needs a modest set (no gdal/geos/proj/OpenGL -- those came
# from the excluded scBio/MOMF). Font -dev pkgs are belt-and-braces from apt.
RUN apt-get update && apt-get install -y --no-install-recommends \
      git cmake pkg-config curl ca-certificates \
      libpng-dev libjpeg-dev libtiff5-dev \
      libxml2-dev libcurl4-openssl-dev libssl-dev \
      zlib1g-dev libicu-dev libglpk-dev libsqlite3-dev libgit2-dev \
      libfontconfig1-dev libfreetype6-dev libharfbuzz-dev libfribidi-dev libx11-dev \
      libhdf5-dev libgsl-dev \
    && rm -rf /var/lib/apt/lists/*

ENV PROJECT_ROOT=/opt/motrpac
WORKDIR /opt/motrpac

# --- R layer ---------------------------------------------------------------------
# Reproducible CRAN: a dated Posit PM *binary* snapshot for this distro (jammy) =>
# no compiling for CRAN/Bioc. Change the date to move the snapshot.
ENV CRAN_REPO=https://packagemanager.posit.co/cran/__linux__/jammy/2026-06-02
COPY deconvolution/setup/ ./deconvolution/setup/
COPY vendor/BayesPrism/   ./vendor/BayesPrism/
RUN Rscript deconvolution/setup/install_r_packages.R \
 && Rscript -e 'for (p in c("BayesPrism","omnideconv","MuSiC","DWLS","SCDC","BisqueRNA")) \
      stopifnot(requireNamespace(p, quietly=TRUE)); cat("DECONV_R_OK\n")'

# --- Python layer (full repo env) ------------------------------------------------
# uv gives a reproducible, base-independent Python 3.12 (matching motrpac-env), so the
# wheels (incl. torch) install regardless of the base image's system Python. reticulate
# is pointed at this interpreter so the R `anndata`/reticulate bridge works.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
COPY requirements.txt ./requirements.txt
RUN uv venv --python 3.12 /opt/motrpac-venv \
 && uv pip install --python /opt/motrpac-venv/bin/python --no-cache -r requirements.txt \
 && uv pip install --python /opt/motrpac-venv/bin/python --no-cache --no-deps \
      "git+https://github.com/omnideconv/scaden.git@03f4ee3e4cce8808a5d1a2467ea5b87f04c90a09"
# Scaden (above) is added --no-deps because its fork pins a stale tensorflow==2.13.0; the modern
# TF + the tf-keras Keras-2 shim come from requirements.txt. TF_USE_LEGACY_KERAS makes
# tf.keras = Keras 2 for Scaden's code (harmless to torch). We do NOT globally disable CUDA here
# (that would break GPU torch); run_omnideconv.* scopes CUDA_VISIBLE_DEVICES=-1 to the
# deconvolution run so TF stays CPU without touching torch.
ENV RETICULATE_PYTHON=/opt/motrpac-venv/bin/python \
    PATH=/opt/motrpac-venv/bin:${PATH} \
    TF_USE_LEGACY_KERAS=1
RUN CUDA_VISIBLE_DEVICES=-1 python -c "import anndata, scanpy, pandas, scipy, numpy, torch, transformers, yaml, tensorflow, autogenes, scaden; print('PY_ENV_OK')"

# Default: drop into R. At runtime mount the repo at /work; `python`/`Rscript` both work.
WORKDIR /work
CMD ["R", "--no-save"]
