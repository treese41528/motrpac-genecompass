#!/bin/bash
# Wrapper for run_deconvolution.R -- reproduces the Gilbreth R env (project-local
# R_libs/, suppressed site Rprofile, TMPDIR=project tmp/, conda stripped so
# scran/igraph dlopen cleanly), then runs the deconvolution.
#
# run.prism is CPU-heavy -- use a COMPUTE node (see run_validation.slurm).
#
# Usage:  [N_CORES=16] run_deconvolution.sh <ref_dir> <mixture_dir> <out_dir> [mtx_basename]
set -eo pipefail

PROJECT_ROOT="/depot/reese18/apps/motrpac-genecompass"
export R_LIBS_USER="${PROJECT_ROOT}/R_libs"
export TMPDIR="${PROJECT_ROOT}/tmp"
export N_CORES="${N_CORES:-4}"
export R_PROFILE=/dev/null R_PROFILE_USER=/dev/null R_ENVIRON=/dev/null R_ENVIRON_USER=/dev/null
mkdir -p "${TMPDIR}"

source /etc/profile.d/modules.sh
module load r/4.4.1

export PKG_CONFIG_PATH=$(echo "${PKG_CONFIG_PATH:-}" | tr ':' '\n' \
  | grep -v '/apps/external/conda/2025.09' | tr '\n' ':' | sed 's/:$//')
export PATH=$(echo "${PATH}" | tr ':' '\n' \
  | grep -v '/apps/external/conda/2025.09' | tr '\n' ':' | sed 's/:$//')

echo "R       : $(which R)"
echo "N_CORES : ${N_CORES}"
Rscript --no-init-file --no-site-file \
  "${PROJECT_ROOT}/deconvolution/R/run_deconvolution.R" "$@"
