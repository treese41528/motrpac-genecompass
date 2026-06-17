# shellcheck shell=bash
# =============================================================================
# site_env.sh -- shared site/cluster environment loader.  SOURCE this; don't run it.
#
# ONE place that reproduces the build/run environment for every deconvolution
# wrapper (deconvolution/R/*.sh) and the bootstrap installer, in a way that is
# SITE-DRIVEN and CONDITIONAL -- so the SAME scripts work on:
#   * an Lmod/Spack HPC cluster (e.g. Gilbreth) -> loads the modules named in site.env
#   * a different module cluster                -> just swap the module names in site.env
#   * a laptop / workstation with R on PATH     -> no site.env, no modules, uses PATH
#   * inside the container (R + libs baked in)   -> no modules, runs against the image
#
# Every step below is a NO-OP unless its variable is set (in site.env or the
# environment). So off a configured cluster the default is simply "use whatever R /
# python is already on PATH" -- which is exactly right for the container and laptops.
#
# The caller must `export PROJECT_ROOT` before sourcing. Knobs (ALL optional) live in
# deconvolution/setup/site.env  (gitignored; copy from site.env.gilbreth.example):
#
#   R_MODULES   Full module list for THIS cluster, in load order, e.g.
#               "r/4.4.1 libpng/1.6.37 zlib/1.3.1 cmake/3.30.2".
#               (Legacy single-module R_MODULE / config r_module are honored as fallback.)
#   STRIP_CONDA Conda/Anaconda prefix to drop from PATH + PKG_CONFIG_PATH (NOT CPATH):
#               on some clusters its libstdc++ shadows the module gcc and breaks
#               compiled R deps (scran/igraph). Leave empty off such clusters.
#   MGC_PYTHON  Python interpreter for reticulate + the omnideconv Python methods
#               (AutoGeneS / Scaden). Feeds both ENV_PY and RETICULATE_PYTHON.
#   MGC_MOTRPAC_DATA_STORE  Where install_motrpac_data.sh stages the ~18 GB external
#               data (read directly by that script; listed here so it lives with the
#               other site paths).
# =============================================================================

: "${PROJECT_ROOT:?site_env.sh: export PROJECT_ROOT before sourcing}"

# (1) Per-site profile (gitignored; cluster/laptop-specific). Absent => use PATH as-is.
_mgc_site_env="${PROJECT_ROOT}/deconvolution/setup/site.env"
if [ -f "${_mgc_site_env}" ]; then
  # shellcheck disable=SC1090
  source "${_mgc_site_env}"
fi

# (2) Environment modules -- only when a module list is declared AND an Lmod `module`
#     command exists. Prefer R_MODULES (full list); fall back to a single R_MODULE /
#     CFG_R_MODULE (from config). On a laptop/container there is no `module` -> no-op.
_mgc_modules="${R_MODULES:-${R_MODULE:-${CFG_R_MODULE:-}}}"
if [ -n "${_mgc_modules}" ] && command -v module >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source /etc/profile.d/modules.sh 2>/dev/null || true
  echo ">> module load ${_mgc_modules}"
  module load ${_mgc_modules}
fi

# (3) Optional conda strip (PATH + PKG_CONFIG_PATH; NOT CPATH -- some headers, e.g.
#     openssl/ICU on Gilbreth, live only there). No-op unless STRIP_CONDA is set. The
#     `|| true` keeps `set -e` happy when the prefix isn't present in the path.
if [ -n "${STRIP_CONDA:-}" ]; then
  export PATH="$(echo "${PATH}" | tr ':' '\n' | grep -vF "${STRIP_CONDA}" | tr '\n' ':' | sed 's/:$//' || true)"
  export PKG_CONFIG_PATH="$(echo "${PKG_CONFIG_PATH:-}" | tr ':' '\n' | grep -vF "${STRIP_CONDA}" | tr '\n' ':' | sed 's/:$//' || true)"
fi

# (4) Python for reticulate / the omnideconv Python methods. One knob (MGC_PYTHON)
#     feeds both names the scripts read; an already-exported value always wins.
if [ -n "${MGC_PYTHON:-}" ]; then
  export ENV_PY="${ENV_PY:-${MGC_PYTHON}}"
  export RETICULATE_PYTHON="${RETICULATE_PYTHON:-${MGC_PYTHON}}"
fi
