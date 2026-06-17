#!/bin/bash
# Run the BayesPrism GBM tutorial replication (Stage 8 validation).
#
# Reproduces the R environment from install_bayesprism.sh (project-local
# R_libs/, suppressed site Rprofile, conda stripped from discovery so
# scran/igraph dlopen cleanly) and runs run_tutorial_gbm.R.
#
# run.prism is CPU-heavy — use a COMPUTE node for the full 169-sample job, e.g.
#   salloc -A <account> -N1 -n16 -t2:00:00
#   N_CORES=16 deconvolution/R/run_tutorial_gbm.sh
#
# Quick login-node sanity check of the whole chain (3 bulk samples):
#   SMOKE=1 deconvolution/R/run_tutorial_gbm.sh

set -eo pipefail   # -u dropped: Gilbreth's profile.d scripts reference unbound vars

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PROJECT_ROOT
export R_LIBS_USER="${PROJECT_ROOT}/R_libs"
export TMPDIR="${PROJECT_ROOT}/tmp"          # project-local scratch (run.prism writes mixtures here)
export N_CORES="${N_CORES:-4}"
export SMOKE="${SMOKE:-0}"

# Suppress Gilbreth's site Rprofile (autoloads Rhipe).
export R_PROFILE=/dev/null
export R_PROFILE_USER=/dev/null
export R_ENVIRON=/dev/null
export R_ENVIRON_USER=/dev/null

mkdir -p "${TMPDIR}"

# Reproduce the build/run env via the shared site profile (modules + conda strip from
# site.env so scran/igraph dlopen cleanly; a no-op off a module cluster). See site_env.sh.
source "${PROJECT_ROOT}/deconvolution/setup/site_env.sh"

echo "R        : $(which R)"
echo "R_LIBS   : ${R_LIBS_USER}"
echo "N_CORES  : ${N_CORES}"
echo "SMOKE    : ${SMOKE}"
echo

Rscript --no-init-file --no-site-file \
  "${PROJECT_ROOT}/deconvolution/R/run_tutorial_gbm.R"
