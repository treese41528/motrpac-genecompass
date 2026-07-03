#!/bin/bash
# Wrapper for simulate_simbu.R -- reproduces the project R env (R_libs/, suppressed site
# Rprofile, TMPDIR=project tmp/, site modules/conda-strip via site_env.sh) exactly like the
# deconv wrappers, then runs the SimBu confounder simulator. h5ad-free (reads the built
# reference matrices) so it is light, but big refs (downsampled to --cells-per-type) still
# want a compute node; smoke runs on small refs are fine on a login node.
#
# Usage:  simulate_simbu.sh --ref-dir <dir> --out <dir> [--scenario random] [--scaling expressed_genes] ...
set -eo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PROJECT_ROOT PIPELINE_ROOT="$PROJECT_ROOT"
export R_LIBS_USER="${PROJECT_ROOT}/R_libs"
export TMPDIR="${PROJECT_ROOT}/tmp"
export R_PROFILE=/dev/null R_PROFILE_USER=/dev/null R_ENVIRON=/dev/null R_ENVIRON_USER=/dev/null
mkdir -p "${TMPDIR}"
# Load site modules (R + libpng/zlib) and strip conda where site.env declares them; no-op on
# a laptop/container. See deconvolution/setup/site_env.sh.
source "${PROJECT_ROOT}/deconvolution/setup/site_env.sh"
echo "R : $(command -v R)"
Rscript --no-init-file --no-site-file \
  "${PROJECT_ROOT}/deconvolution/omnideconv_bench/simulate_simbu.R" "$@"
