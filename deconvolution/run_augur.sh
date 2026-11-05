#!/bin/bash
# run_augur.sh -- R-env wrapper for deconvolution/run_augur.R (canonical Augur corroboration of the Aim-2 gate).
# Same env pattern as deconvolution/R/*.sh: config read via load_config.sh (fatal on failure), project R_libs,
# site modules. Args pass through: run_augur.sh [AUGUR_INPUT_DIR OUT_TSV] (default: from the config).
set -eo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PROJECT_ROOT PIPELINE_ROOT="$PROJECT_ROOT"
source "${PROJECT_ROOT}/deconvolution/load_config.sh"
export R_LIBS_USER="${PROJECT_ROOT}/R_libs"
export TMPDIR="${PROJECT_ROOT}/tmp"
export R_PROFILE=/dev/null R_PROFILE_USER=/dev/null R_ENVIRON=/dev/null R_ENVIRON_USER=/dev/null
mkdir -p "${TMPDIR}"
source "${PROJECT_ROOT}/deconvolution/setup/site_env.sh"
cd "${PROJECT_ROOT}"
Rscript --no-init-file --no-site-file "${PROJECT_ROOT}/deconvolution/run_augur.R" "$@"
