#!/bin/bash
# =============================================================================
# Portable bare-metal bootstrap for the deconvolution R stack.
#
# Sets up the toolchain for the host, then hands off to install_r_packages.R (which
# does the actual, environment-agnostic package install). All cluster-specific bits
# (module names, conda quirks) live in an OPTIONAL site profile -- on a laptop or in a
# container base with R already present, this script is a thin no-op wrapper.
#
# Usage:
#   bash deconvolution/setup/install_r_env.sh
#
# Site config (only needed on module/conda clusters): copy the example and edit,
#   cp deconvolution/setup/site.env.gilbreth.example deconvolution/setup/site.env
# or export R_MODULES / STRIP_CONDA yourself. See SETUP.md.
# =============================================================================
set -eo pipefail

SETUP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SETUP_DIR}/../.." && pwd)"
export PROJECT_ROOT

# Optional, gitignored, cluster-specific profile (modules to load, conda to strip).
if [ -f "${SETUP_DIR}/site.env" ]; then
  echo ">> sourcing site profile: ${SETUP_DIR}/site.env"
  # shellcheck disable=SC1091
  source "${SETUP_DIR}/site.env"
fi

export R_LIBS_USER="${R_LIBS_USER:-${PROJECT_ROOT}/R_libs}"
export TMPDIR="${TMPDIR:-${PROJECT_ROOT}/tmp}"
mkdir -p "${R_LIBS_USER}" "${TMPDIR}"

# 1) Toolchain. On a module cluster with R_MODULES set, load them; otherwise assume R
#    and a C/C++ toolchain are already on PATH (system package / conda / rocker base).
if [ -n "${R_MODULES:-}" ] && command -v module >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source /etc/profile.d/modules.sh 2>/dev/null || true
  echo ">> module load ${R_MODULES}"
  module load ${R_MODULES}
fi

# 2) Optional conda strip. On some clusters conda's libstdc++ shadows the module gcc and
#    breaks compiled R deps (scran/igraph). Set STRIP_CONDA=<conda prefix> to remove it
#    from PATH/PKG_CONFIG_PATH -- but NOT from CPATH, which may be the only source of
#    some headers (e.g. openssl/ICU on Gilbreth).
if [ -n "${STRIP_CONDA:-}" ]; then
  echo ">> stripping '${STRIP_CONDA}' from PATH/PKG_CONFIG_PATH (keeping CPATH)"
  export PATH=$(echo "${PATH}" | tr ':' '\n' | grep -v "${STRIP_CONDA}" | tr '\n' ':' | sed 's/:$//')
  export PKG_CONFIG_PATH=$(echo "${PKG_CONFIG_PATH:-}" | tr ':' '\n' | grep -v "${STRIP_CONDA}" | tr '\n' ':' | sed 's/:$//')
fi

# 3) Make the project library authoritative; parallel compiles.
export R_PROFILE=/dev/null R_PROFILE_USER=/dev/null R_ENVIRON=/dev/null R_ENVIRON_USER=/dev/null
export MAKEFLAGS="${MAKEFLAGS:--j$(nproc 2>/dev/null || echo 4)}"

if ! command -v R >/dev/null 2>&1; then
  echo "ERROR: R not found on PATH."
  echo "  - module cluster:  set R_MODULES in site.env (e.g. R_MODULES=\"r/4.4.1 ...\")"
  echo "  - laptop/server:   install R 4.4.x (or use the container -- see SETUP.md)"
  exit 1
fi
echo ">> R          : $(command -v R)"
echo ">> R_LIBS_USER: ${R_LIBS_USER}"
echo ">> handing off to install_r_packages.R ..."
echo

Rscript --no-init-file --no-site-file "${SETUP_DIR}/install_r_packages.R"
