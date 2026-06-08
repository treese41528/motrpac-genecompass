#!/bin/bash
# Wrapper for prepare_motrpac_bulk.R -- reproduces the Gilbreth R env (project-local
# R_libs/, suppressed site Rprofile, TMPDIR=project tmp/, conda stripped) then lifts
# the MoTrPAC bulk gene IDs to current mRatBN7.2 ENSRNOG and writes deconv-ready
# samples x genes matrices. Light enough for a login node (reads .rda, builds the
# liftover map, writes mtx) -- no GPU/compute node needed.
#
# Usage:  prepare_motrpac_bulk.sh <TISSUE|ALL> <out_dir> [map_out] [report_out]
#   e.g.  prepare_motrpac_bulk.sh ALL deconvolution/motrpac_bulk
set -eo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PROJECT_ROOT PIPELINE_ROOT="$PROJECT_ROOT"
# Best-effort config read (falls back to repo-relative defaults if python/pyyaml absent)
command -v python3 >/dev/null 2>&1 && \
  eval "$(python3 "${PROJECT_ROOT}/deconvolution/_config_sh.py" 2>/dev/null || true)"

export MOTRPAC_DATA_DIR="${MOTRPAC_DATA_DIR:-${CFG_MOTRPAC_DATA_DIR:-${PROJECT_ROOT}/data/motrpac/rat_training_6mo/data}}"
export RAT_TOKEN_MAPPING="${RAT_TOKEN_MAPPING:-${CFG_RAT_TOKEN_MAPPING:-${PROJECT_ROOT}/data/training/ortholog_mappings/rat_token_mapping.tsv}}"
export RAT_GENE_INFO="${RAT_GENE_INFO:-${CFG_RAT_GENE_INFO:-${PROJECT_ROOT}/data/references/biomart/rat_gene_info.tsv}}"
export RAT_RGD_GENES="${RAT_RGD_GENES:-${CFG_RAT_RGD_GENES:-${PROJECT_ROOT}/data/references/biomart/GENES_RAT.txt}}"
export R_LIBS_USER="${PROJECT_ROOT}/R_libs"
export TMPDIR="${PROJECT_ROOT}/tmp"
R_MODULE="${R_MODULE:-${CFG_R_MODULE:-r/4.4.1}}"
export R_PROFILE=/dev/null R_PROFILE_USER=/dev/null R_ENVIRON=/dev/null R_ENVIRON_USER=/dev/null
mkdir -p "${TMPDIR}"

source /etc/profile.d/modules.sh
module load "${R_MODULE}"

export PKG_CONFIG_PATH=$(echo "${PKG_CONFIG_PATH:-}" | tr ':' '\n' \
  | grep -v '/apps/external/conda/2025.09' | tr '\n' ':' | sed 's/:$//')
export PATH=$(echo "${PATH}" | tr ':' '\n' \
  | grep -v '/apps/external/conda/2025.09' | tr '\n' ':' | sed 's/:$//')

echo "R       : $(which R)"
cd "${PROJECT_ROOT}"
Rscript --no-init-file --no-site-file \
  "${PROJECT_ROOT}/deconvolution/R/prepare_motrpac_bulk.R" "$@"
