#!/bin/bash
# Wrapper for extract_z.R -- same Gilbreth R env as run_deconvolution.sh (project
# R_libs/, suppressed site Rprofile, conda stripped). Z extraction is light
# (readRDS + indexing), so this can run on a login node.
#
# Usage:  extract_z.sh <bp_result.rds> <out_dir>
set -eo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PROJECT_ROOT PIPELINE_ROOT="$PROJECT_ROOT"
command -v python3 >/dev/null 2>&1 && \
  eval "$(python3 "${PROJECT_ROOT}/deconvolution/_config_sh.py" 2>/dev/null || true)"
R_MODULE="${R_MODULE:-${CFG_R_MODULE:-r/4.4.1}}"
export R_LIBS_USER="${PROJECT_ROOT}/R_libs"
export TMPDIR="${PROJECT_ROOT}/tmp"
export R_PROFILE=/dev/null R_PROFILE_USER=/dev/null R_ENVIRON=/dev/null R_ENVIRON_USER=/dev/null
mkdir -p "${TMPDIR}"

source /etc/profile.d/modules.sh
module load "${R_MODULE}"

export PKG_CONFIG_PATH=$(echo "${PKG_CONFIG_PATH:-}" | tr ':' '\n' \
  | grep -v '/apps/external/conda/2025.09' | tr '\n' ':' | sed 's/:$//')
export PATH=$(echo "${PATH}" | tr ':' '\n' \
  | grep -v '/apps/external/conda/2025.09' | tr '\n' ':' | sed 's/:$//')

Rscript --no-init-file --no-site-file \
  "${PROJECT_ROOT}/deconvolution/R/extract_z.R" "$@"
