#!/bin/bash
# Wrapper for run_omnideconv.R -- reproduces the Gilbreth R env (project-local R_libs/,
# suppressed site Rprofile, TMPDIR=project tmp/, conda stripped from PATH/PKG_CONFIG_PATH
# so compiled deps dlopen against the module toolchain) and runs the multi-method
# omnideconv cross-check. Mirrors run_deconvolution.sh; adds the libpng/zlib modules whose
# runtime libs Seurat/png/data.table were built against (see install_omnideconv.sh).
#
# DWLS signature building is CPU-heavy -- use a COMPUTE node (see run_validation.slurm).
#
# Usage:  [OMNIDECONV_METHODS=music,dwls,scdc,bisque] [N_CORES=16] \
#           run_omnideconv.sh <ref_dir> <mixture_dir> <out_dir> [mtx_basename]
set -eo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PROJECT_ROOT PIPELINE_ROOT="$PROJECT_ROOT"
# Best-effort config read (falls back to repo-relative defaults if python/pyyaml absent)
command -v python3 >/dev/null 2>&1 && \
  eval "$(python3 "${PROJECT_ROOT}/deconvolution/_config_sh.py" 2>/dev/null || true)"
export RAT_EXCLUDE_GENES="${RAT_EXCLUDE_GENES:-${CFG_RAT_EXCLUDE_GENES:-${PROJECT_ROOT}/deconvolution/reference/rat_exclude_genes.tsv}}"
export RAT_PROTEIN_CODING_GENES="${RAT_PROTEIN_CODING_GENES:-${CFG_RAT_PROTEIN_CODING_GENES:-${PROJECT_ROOT}/deconvolution/reference/rat_protein_coding_genes.tsv}}"
export PROTEIN_CODING_ONLY="${PROTEIN_CODING_ONLY:-${CFG_PROTEIN_CODING_ONLY:-1}}"
export RAT_SEX_CHROM_GENES="${RAT_SEX_CHROM_GENES:-${CFG_RAT_SEX_CHROM_GENES:-${PROJECT_ROOT}/deconvolution/reference/rat_sex_chrom_genes.tsv}}"
export EXCLUDE_SEX_CHROMOSOMES="${EXCLUDE_SEX_CHROMOSOMES:-${CFG_EXCLUDE_SEX_CHROMOSOMES:-1}}"
export OMNIDECONV_METHODS="${OMNIDECONV_METHODS:-music,dwls,scdc,bisque}"
export R_LIBS_USER="${PROJECT_ROOT}/R_libs"
export TMPDIR="${PROJECT_ROOT}/tmp"
# Python methods (AutoGeneS, Scaden) run via reticulate against the project venv. Force TF
# onto the Keras-2 shim (tf-keras) and CPU-only so it never contends with torch for the GPU
# or trips over the CUDA-13/TF-2.21 version gap. See deconvolution/R/install_omnideconv_python.sh.
# RETICULATE_PYTHON is resolved AFTER site_env.sh below (site.env MGC_PYTHON -> active venv).
export TF_USE_LEGACY_KERAS="${TF_USE_LEGACY_KERAS:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:--1}"
export N_CORES="${N_CORES:-${CFG_N_CORES:-4}}"
export R_PROFILE=/dev/null R_PROFILE_USER=/dev/null R_ENVIRON=/dev/null R_ENVIRON_USER=/dev/null
mkdir -p "${TMPDIR}"

# Reproduce the build/run env via the shared site profile: load modules (R_MODULES -- on a
# module cluster this MUST include libpng/zlib for Seurat) and strip conda (STRIP_CONDA) only
# where site.env declares them and Lmod exists; a no-op on a laptop/container, where R comes
# from PATH. Also derives ENV_PY/RETICULATE_PYTHON from site.env MGC_PYTHON. See site_env.sh.
source "${PROJECT_ROOT}/deconvolution/setup/site_env.sh"

# reticulate target: site.env MGC_PYTHON (set above) -> active venv -> python3 on PATH.
export RETICULATE_PYTHON="${RETICULATE_PYTHON:-${VIRTUAL_ENV:+${VIRTUAL_ENV}/bin/python}}"
export RETICULATE_PYTHON="${RETICULATE_PYTHON:-$(command -v python3 || true)}"

echo "R       : $(which R)"
echo "N_CORES : ${N_CORES}"
echo "methods : ${OMNIDECONV_METHODS}"
Rscript --no-init-file --no-site-file \
  "${PROJECT_ROOT}/deconvolution/R/run_omnideconv.R" "$@"
