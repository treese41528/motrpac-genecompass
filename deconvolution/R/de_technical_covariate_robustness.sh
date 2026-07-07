#!/bin/bash
# de_technical_covariate_robustness.sh -- R-env wrapper for the RIN / %-globin technical-covariate
# robustness check (Stage 14, step 3). Re-fits the per-cell-type dose slope +/- RIN + pct_globin
# (from MotrpacRatTraining6moData::TRNSCRPT_META). Light (limma on existing pred_z); login-node OK.
# Same R env pattern as run_pseudobulk_de.sh.
set -eo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PROJECT_ROOT PIPELINE_ROOT="$PROJECT_ROOT"
command -v python3 >/dev/null 2>&1 && \
  eval "$(python3 "${PROJECT_ROOT}/deconvolution/_config_sh.py" 2>/dev/null || true)"
export R_LIBS_USER="${PROJECT_ROOT}/R_libs"
export TMPDIR="${PROJECT_ROOT}/tmp"
export R_PROFILE=/dev/null R_PROFILE_USER=/dev/null R_ENVIRON=/dev/null R_ENVIRON_USER=/dev/null
mkdir -p "${TMPDIR}"

source "${PROJECT_ROOT}/deconvolution/setup/site_env.sh"
Rscript --no-init-file --no-site-file \
  "${PROJECT_ROOT}/deconvolution/R/de_technical_covariate_robustness.R"
