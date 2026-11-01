# load_config.sh -- SOURCE this (do not execute) from a deconvolution wrapper after setting PROJECT_ROOT.
# Runs deconvolution/_config_sh.py with $DECONV_PYTHON (default: python3 on PATH; the orchestrators set it to
# the venv interpreter), evals and EXPORTS every CFG_* value so R scripts can read them with Sys.getenv.
# A config read failure stops the calling script (exit 3): there is no fallback to a hardcoded arm.
#
#   PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
#   export PROJECT_ROOT PIPELINE_ROOT="$PROJECT_ROOT"
#   source "${PROJECT_ROOT}/deconvolution/load_config.sh"
: "${PROJECT_ROOT:?load_config.sh: set PROJECT_ROOT before sourcing}"
export PIPELINE_ROOT="${PIPELINE_ROOT:-$PROJECT_ROOT}"
_cfg_py="${DECONV_PYTHON:-python3}"
if ! command -v "$_cfg_py" >/dev/null 2>&1; then
  echo "FATAL: '$_cfg_py' not found; activate the project venv or export DECONV_PYTHON=<venv>/bin/python" >&2
  exit 3
fi
if ! _cfg_out="$("$_cfg_py" "${PROJECT_ROOT}/deconvolution/_config_sh.py")"; then
  echo "FATAL: config read failed (see the message above); refusing to fall back to default paths" >&2
  exit 3
fi
eval "$_cfg_out"
for _cfg_k in $(printf '%s\n' "$_cfg_out" | sed -n 's/^\(CFG_[A-Z0-9_]*\)=.*/\1/p'); do export "$_cfg_k"; done
unset _cfg_py _cfg_out _cfg_k
