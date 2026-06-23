#!/bin/bash
# Install IHW + repfdr into the project-local R library (R_libs/), for the Vetr-faithful
# (MoTrPAC Complex-Traits, Vetr 2024 Nat Commun 15:3346) per-cell-type DE multiple-testing:
#   * IHW  (Bioc) -- Independent Hypothesis Weighting with an informative covariate
#                    (Vetr used a tissue covariate); the upgrade over per-block BH.
#   * repfdr (CRAN) -- replicability / sex-consistency analysis across the two sexes
#                    (Vetr's F1_M1 ... F-1_M-1 sex-consistent states via repfdr).
# limma/edgeR/DESeq2/statmod/locfit are already present; only IHW (+deps) and repfdr are missing.
#
# Run on a Gilbreth login node (needs internet). Reruns are idempotent. ~10-20 min first time.
set -eo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export R_LIBS_USER="${PROJECT_ROOT}/R_libs"
export TMPDIR="${PROJECT_ROOT}/tmp"
export R_PROFILE=/dev/null R_PROFILE_USER=/dev/null R_ENVIRON=/dev/null R_ENVIRON_USER=/dev/null
mkdir -p "${R_LIBS_USER}" "${TMPDIR}"

# Reproduce the build env via the shared site profile (modules + conda strip from site.env so
# compiled deps dlopen cleanly; a no-op off a module cluster). See setup/site_env.sh + SETUP.md.
source "${PROJECT_ROOT}/deconvolution/setup/site_env.sh"

echo "R           : $(which R)"
echo "R_LIBS_USER : ${R_LIBS_USER}"

R --no-init-file --no-site-file --slave <<'RSCRIPT'
.libPaths(c(Sys.getenv("R_LIBS_USER"), .libPaths()))
options(repos = c(CRAN = "https://cloud.r-project.org"))
if (!requireNamespace("BiocManager", quietly = TRUE))
  install.packages("BiocManager", lib = Sys.getenv("R_LIBS_USER"))

if (!requireNamespace("IHW", quietly = TRUE)) {
  cat("Installing IHW (+deps) ...\n")
  BiocManager::install("IHW", lib = Sys.getenv("R_LIBS_USER"),
                       update = FALSE, ask = FALSE, Ncpus = 4)
} else cat("IHW already installed.\n")

if (!requireNamespace("repfdr", quietly = TRUE)) {
  cat("Installing repfdr (+deps) ...\n")
  install.packages("repfdr", lib = Sys.getenv("R_LIBS_USER"), Ncpus = 4)
} else cat("repfdr already installed.\n")

suppressPackageStartupMessages({ library(IHW); library(repfdr) })
cat(sprintf("IHW version    : %s\n", as.character(packageVersion("IHW"))))
cat(sprintf("repfdr version : %s\n", as.character(packageVersion("repfdr"))))
cat(sprintf("ihw() exists       : %s\n", exists("ihw")))
cat(sprintf("repfdr() exists    : %s\n", exists("repfdr")))
RSCRIPT
echo "Install complete."
