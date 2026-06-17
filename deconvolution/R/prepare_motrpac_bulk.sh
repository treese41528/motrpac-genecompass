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
export MOTRPAC_BULK_LIFTOVER="${MOTRPAC_BULK_LIFTOVER:-${CFG_MOTRPAC_BULK_LIFTOVER:-${PROJECT_ROOT}/deconvolution/reference/motrpac_bulk_liftover.tsv}}"
export MOTRPAC_BULK_LIFTOVER_REPORT="${MOTRPAC_BULK_LIFTOVER_REPORT:-${CFG_MOTRPAC_BULK_LIFTOVER_REPORT:-${PROJECT_ROOT}/deconvolution/reference/motrpac_bulk_liftover_report.txt}}"
export SC_CORPUS_GENES="${SC_CORPUS_GENES:-${CFG_IDSPACE_AUDIT_DIR:-${PROJECT_ROOT}/data/deconvolution/idspace_audit}/corpus_genes.txt}"
export SC_ID2SYMBOL="${SC_ID2SYMBOL:-${CFG_IDSPACE_AUDIT_DIR:-${PROJECT_ROOT}/data/deconvolution/idspace_audit}/id2symbol.tsv}"
export R_LIBS_USER="${PROJECT_ROOT}/R_libs"
export TMPDIR="${PROJECT_ROOT}/tmp"
export R_PROFILE=/dev/null R_PROFILE_USER=/dev/null R_ENVIRON=/dev/null R_ENVIRON_USER=/dev/null
mkdir -p "${TMPDIR}"

# Reproduce the build/run env via the shared site profile: load modules (R_MODULES) and
# strip conda (STRIP_CONDA) only where site.env declares them and Lmod exists; a no-op on a
# laptop/container, where R comes from PATH. See deconvolution/setup/site_env.sh + SETUP.md.
source "${PROJECT_ROOT}/deconvolution/setup/site_env.sh"

echo "R       : $(which R)"
cd "${PROJECT_ROOT}"
Rscript --no-init-file --no-site-file \
  "${PROJECT_ROOT}/deconvolution/R/prepare_motrpac_bulk.R" "$@"
