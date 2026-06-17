#!/bin/bash
# Wrapper for run_deconvolution.R -- reproduces the Gilbreth R env (project-local
# R_libs/, suppressed site Rprofile, TMPDIR=project tmp/, conda stripped so
# scran/igraph dlopen cleanly), then runs the deconvolution.
#
# run.prism is CPU-heavy -- use a COMPUTE node (see run_validation.slurm).
#
# Usage:  [N_CORES=16] run_deconvolution.sh <ref_dir> <mixture_dir> <out_dir> [mtx_basename]
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
export R_LIBS_USER="${PROJECT_ROOT}/R_libs"
export TMPDIR="${PROJECT_ROOT}/tmp"
export N_CORES="${N_CORES:-${CFG_N_CORES:-4}}"
export R_PROFILE=/dev/null R_PROFILE_USER=/dev/null R_ENVIRON=/dev/null R_ENVIRON_USER=/dev/null
mkdir -p "${TMPDIR}"

# Reproduce the build/run env via the shared site profile: load modules (R_MODULES) and
# strip conda (STRIP_CONDA) only where site.env declares them and Lmod exists; a no-op on a
# laptop/container, where R comes from PATH. See deconvolution/setup/site_env.sh + SETUP.md.
source "${PROJECT_ROOT}/deconvolution/setup/site_env.sh"

echo "R       : $(which R)"
echo "N_CORES : ${N_CORES}"
Rscript --no-init-file --no-site-file \
  "${PROJECT_ROOT}/deconvolution/R/run_deconvolution.R" "$@"
