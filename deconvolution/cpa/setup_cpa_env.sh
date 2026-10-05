#!/bin/bash
# ============================================================================
# setup_cpa_env.sh — build the ISOLATED CPA environment for Aim 2c (Module C).
#
# Creates a SEPARATE venv (`cpa-env`, a sibling of `motrpac-env`) and installs
# `cpa-tools` there. It DOES NOT touch the shared `motrpac-env`: `cpa-tools`
# pulls scvi-tools + pytorch-lightning + torch, which could break the pinned
# pipeline environment the whole project depends on. Isolation is by
# construction (fresh venv, no --system-site-packages) and enforced by guards.
#
# Usage:   bash deconvolution/cpa/setup_cpa_env.sh
# Override: CPA_ENV=/path/to/env  PYBASE=python3.12  bash .../setup_cpa_env.sh
# ============================================================================
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"   # .../motrpac-genecompass
APPS="$(cd "$REPO/.." && pwd)"                                # .../apps
CPA_ENV="${CPA_ENV:-$APPS/cpa-env}"                           # sibling of motrpac-env
MOTRPAC_ENV="$APPS/motrpac-env"
REQ="$REPO/deconvolution/cpa/requirements-cpa.txt"
LOCK="$REPO/deconvolution/cpa/requirements-cpa.lock.txt"
# cpa-tools requires Python >=3.9,<3.11 — so cpa-env MUST use a different interpreter
# than motrpac-env's 3.12. /usr/bin/python3.9 is present on Gilbreth (no conda; honors P5).
PYBASE="${PYBASE:-python3.9}"

# --- SAFETY: refuse to ever operate on the shared pipeline env ---
if [ "$CPA_ENV" = "$MOTRPAC_ENV" ]; then
  echo "REFUSING: CPA_ENV must not be motrpac-env ($MOTRPAC_ENV)." >&2; exit 1
fi
if [ -e "$CPA_ENV" ] && [ -e "$MOTRPAC_ENV" ] && [ "$CPA_ENV" -ef "$MOTRPAC_ENV" ]; then
  echo "REFUSING: CPA_ENV resolves to motrpac-env." >&2; exit 1
fi
case "$CPA_ENV" in *motrpac-env*) echo "REFUSING: CPA_ENV path contains 'motrpac-env'." >&2; exit 1;; esac
command -v "$PYBASE" >/dev/null 2>&1 || { echo "No $PYBASE on PATH (need Python 3.9 or 3.10; /usr/bin/python3.9 exists on Gilbreth)." >&2; exit 1; }
PYVER="$("$PYBASE" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
case "$PYVER" in
  3.9|3.10) ;;
  *) echo "REFUSING: cpa-tools requires Python >=3.9,<3.11 but $PYBASE is $PYVER. Use PYBASE=python3.9 (or python3.10)." >&2; exit 1;;
esac

echo "Repo:         $REPO"
echo "Isolated env: $CPA_ENV   (base: $PYBASE $($PYBASE --version 2>&1))"
echo "Shared env:   $MOTRPAC_ENV   (will be left untouched)"
BEFORE="$(ls -la "$MOTRPAC_ENV" 2>/dev/null | md5sum | cut -d' ' -f1)"

# --- fresh, fully-isolated venv ---
if [ ! -x "$CPA_ENV/bin/python" ]; then
  "$PYBASE" -m venv "$CPA_ENV"
fi
"$CPA_ENV/bin/python" -m pip install --upgrade pip wheel setuptools

# --- install cpa-tools + deps into the ISOLATED env only ---
"$CPA_ENV/bin/pip" install -r "$REQ"

# --- verify the modern API is importable ---
"$CPA_ENV/bin/python" - <<'PY'
import cpa, scvi, torch, anndata, scanpy
print(f"cpa-tools={getattr(cpa,'__version__','?')} scvi-tools={scvi.__version__} "
      f"torch={torch.__version__} anndata={anndata.__version__} scanpy={scanpy.__version__}")
assert hasattr(cpa, "CPA") and hasattr(cpa.CPA, "setup_anndata"), "cpa.CPA / setup_anndata missing"
print("OK: cpa.CPA + cpa.CPA.setup_anndata present")
PY

# --- freeze a reproducible lockfile ---
"$CPA_ENV/bin/pip" freeze > "$LOCK"

# --- confirm the shared env was not modified ---
AFTER="$(ls -la "$MOTRPAC_ENV" 2>/dev/null | md5sum | cut -d' ' -f1)"
[ "$BEFORE" = "$AFTER" ] && ISO="UNCHANGED ✓" || ISO="CHANGED (!!)"

echo
echo "DONE. Isolated CPA env ready:  $CPA_ENV"
echo "  motrpac-env: $ISO"
echo "  Lockfile:    deconvolution/cpa/requirements-cpa.lock.txt"
echo "  Activate:    source $CPA_ENV/bin/activate    (NEVER install cpa into motrpac-env)"
