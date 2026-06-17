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
export R_LIBS_USER="${PROJECT_ROOT}/R_libs"
export TMPDIR="${PROJECT_ROOT}/tmp"
export R_PROFILE=/dev/null R_PROFILE_USER=/dev/null R_ENVIRON=/dev/null R_ENVIRON_USER=/dev/null
mkdir -p "${TMPDIR}"

# Reproduce the build/run env via the shared site profile: load modules (R_MODULES) and
# strip conda (STRIP_CONDA) only where site.env declares them and Lmod exists; a no-op on a
# laptop/container, where R comes from PATH. See deconvolution/setup/site_env.sh + SETUP.md.
source "${PROJECT_ROOT}/deconvolution/setup/site_env.sh"

Rscript --no-init-file --no-site-file \
  "${PROJECT_ROOT}/deconvolution/R/extract_z.R" "$@"
