#!/bin/bash
# Install MotrpacRatTraining6moData R package and stage all data.
#
# Three actions:
#   1. Install the R package from vendor/MotrpacRatTraining6moData/ into
#      the project-local R library (R_libs/) so R scripts can do
#      `library(MotrpacRatTraining6moData)`.
#   2. Mirror the 206 in-package .rda files (~789 MB) to the data store
#      so non-R tools (e.g. Python via pyreadr) can read them directly.
#   3. Fetch the 58 GCS-hosted external .rda files (~18.4 GiB:
#      56 epigen-rda ~18 GiB + 2 feature-annot ~362 MB) into the data store.
#
# Data layout produced (single source of truth):
#   /depot/reese18/data/motrpac/rat_training_6mo/
#     data/                    206 .rda files mirrored from package
#     extdata/epigen-rda/      56 GCS-hosted ATAC/METHYL .rda files
#     extdata/feature-annot/   2 GCS-hosted feature-annotation files
#
# Project-side access (symlink, created at end of script):
#   data/motrpac/rat_training_6mo -> /depot/reese18/data/motrpac/rat_training_6mo
#
# Run on a Gilbreth login node. Total time: ~30-60 min on first run
# (mostly the ~18 GiB GCS download). Reruns are idempotent:
#   - R CMD INSTALL re-installs (cheap-ish, ~5 min for lazyload step).
#   - rsync mirrors only changed/missing .rda files.
#   - wget -c skips already-complete files and resumes partial ones.

set -eo pipefail   # -u dropped: Gilbreth's /etc/profile.d/00-modulepath.sh references unbound vars

PROJECT_ROOT="/depot/reese18/apps/motrpac-genecompass"
DATA_STORE="/depot/reese18/data/motrpac/rat_training_6mo"
PROJECT_DATA_LINK="${PROJECT_ROOT}/data/motrpac/rat_training_6mo"
PKG_SRC="${PROJECT_ROOT}/vendor/MotrpacRatTraining6moData"
README="${PKG_SRC}/README.md"

export R_LIBS_USER="${PROJECT_ROOT}/R_libs"
export TMPDIR="${PROJECT_ROOT}/tmp"

# Suppress Gilbreth's site Rprofile (autoloads Rhipe; aborts R CMD INSTALL).
export R_PROFILE=/dev/null
export R_PROFILE_USER=/dev/null
export R_ENVIRON=/dev/null
export R_ENVIRON_USER=/dev/null

mkdir -p "${R_LIBS_USER}" "${TMPDIR}"
mkdir -p "${DATA_STORE}/data" \
         "${DATA_STORE}/extdata/epigen-rda" \
         "${DATA_STORE}/extdata/feature-annot"

source /etc/profile.d/modules.sh
module load r/4.4.1

echo "R           : $(which R)"
echo "R version   : $(R --version | head -1)"
echo "R_LIBS_USER : ${R_LIBS_USER}"
echo "DATA_STORE  : ${DATA_STORE}"
echo "TMPDIR      : ${TMPDIR}"
echo

# -----------------------------------------------------------------------------
# [1/4] Install the R package into the project-local library.
# -----------------------------------------------------------------------------
echo "[1/4] Installing MotrpacRatTraining6moData from vendor/..."
R CMD INSTALL --library="${R_LIBS_USER}" "${PKG_SRC}"

# -----------------------------------------------------------------------------
# [2/4] Mirror in-package .rda files to the data store.
# -----------------------------------------------------------------------------
echo
echo "[2/4] Mirroring 206 in-package .rda files to ${DATA_STORE}/data/..."
rsync -a --info=stats0,progress2 \
  "${PKG_SRC}/data/" "${DATA_STORE}/data/"
n_pkg=$(find "${DATA_STORE}/data" -maxdepth 1 -name '*.rda' | wc -l)
echo "    ${n_pkg} .rda files present in ${DATA_STORE}/data/"

# -----------------------------------------------------------------------------
# [3/4] Fetch GCS-hosted external data.
# -----------------------------------------------------------------------------
echo
echo "[3/4] Fetching GCS-hosted external .rda files..."

# Extract URLs from README, dedup, sort.
mapfile -t urls < <(
  grep -oE "https://storage\.googleapis\.com/motrpac-rat-training-6mo-extdata/[^)]+\.rda" "${README}" \
    | sort -u
)
echo "    Found ${#urls[@]} URLs in README (expected 58)"
if [ "${#urls[@]}" -ne 58 ]; then
  echo "ERROR: expected 58 URLs, got ${#urls[@]} — README format may have changed."
  exit 1
fi

n_skipped=0
n_fetched=0
for url in "${urls[@]}"; do
  # URL: .../extdata/<subdir>/<file.rda>
  subdir=$(echo "${url}" | awk -F'extdata/' '{print $2}' | awk -F'/' '{print $1}')
  fname=$(basename "${url}")
  dest="${DATA_STORE}/extdata/${subdir}/${fname}"

  if [ -f "${dest}" ]; then
    # Verify size matches the remote so we don't keep a truncated file.
    remote_size=$(curl -sIL "${url}" \
      | awk 'BEGIN{IGNORECASE=1} /^content-length:/{gsub(/\r/,""); print $2}' \
      | tail -1)
    local_size=$(stat -c%s "${dest}")
    if [ -n "${remote_size}" ] && [ "${remote_size}" = "${local_size}" ]; then
      n_skipped=$((n_skipped + 1))
      continue
    fi
    echo "    [resume] ${fname} (local ${local_size} / remote ${remote_size:-unknown})"
  else
    echo "    [fetch ] ${fname}"
  fi

  # wget -c resumes partial downloads; quiet progress for log readability.
  wget -q --show-progress -c -O "${dest}" "${url}"
  n_fetched=$((n_fetched + 1))
done
echo "    Fetched ${n_fetched}, skipped ${n_skipped} (already complete)"

# -----------------------------------------------------------------------------
# [4/4] Symlink data store into the project tree.
# -----------------------------------------------------------------------------
echo
echo "[4/4] Creating symlink ${PROJECT_DATA_LINK} -> ${DATA_STORE}..."
if [ -L "${PROJECT_DATA_LINK}" ]; then
  current_target=$(readlink "${PROJECT_DATA_LINK}")
  if [ "${current_target}" = "${DATA_STORE}" ]; then
    echo "    [ok] symlink already correct"
  else
    echo "ERROR: ${PROJECT_DATA_LINK} symlinks to ${current_target}; expected ${DATA_STORE}"
    exit 1
  fi
elif [ -d "${PROJECT_DATA_LINK}" ]; then
  if [ -z "$(ls -A "${PROJECT_DATA_LINK}")" ]; then
    rmdir "${PROJECT_DATA_LINK}"
    ln -s "${DATA_STORE}" "${PROJECT_DATA_LINK}"
    echo "    [created] symlink (empty placeholder dir removed)"
  else
    echo "ERROR: ${PROJECT_DATA_LINK} is a non-empty directory; refusing to replace."
    exit 1
  fi
else
  ln -s "${DATA_STORE}" "${PROJECT_DATA_LINK}"
  echo "    [created] symlink"
fi

# -----------------------------------------------------------------------------
# Verification
# -----------------------------------------------------------------------------
echo
echo "=== Verification ==="
echo "Symlink            : ${PROJECT_DATA_LINK} -> $(readlink ${PROJECT_DATA_LINK})"
echo "Package .rda files : $(find ${DATA_STORE}/data -maxdepth 1 -name '*.rda' | wc -l) / 206"
echo "epigen-rda  files  : $(find ${DATA_STORE}/extdata/epigen-rda -maxdepth 1 -name '*.rda' | wc -l) / 56"
echo "feature-annot files: $(find ${DATA_STORE}/extdata/feature-annot -maxdepth 1 -name '*.rda' | wc -l) / 2"

R --no-init-file --no-site-file --slave <<'RSCRIPT'
.libPaths(c(Sys.getenv("R_LIBS_USER"), .libPaths()))
suppressPackageStartupMessages(library(MotrpacRatTraining6moData))
cat(sprintf("MotrpacRatTraining6moData version: %s\n",
            as.character(packageVersion("MotrpacRatTraining6moData"))))
cat(sprintf("Location: %s\n", find.package("MotrpacRatTraining6moData")))
# Smoke-test: ensure a known data object loads from the lazyload DB.
e <- new.env(); data(PHENO, envir = e)
cat(sprintf("PHENO loaded: %d rows x %d cols\n", nrow(e$PHENO), ncol(e$PHENO)))
RSCRIPT

echo
echo "Install complete."
