#!/bin/bash
# Install DESeq2 into the project-local R library (R_libs/), for the paper-faithful
# expression-correlation metric: Pearson on DESeq2 variance-stabilizing transformed
# (VST) values (Chu 2022, Nat Cancer, Fig. 1h / Methods). edgeR/limma are already
# present; only DESeq2 (+ its deps) is missing.
#
# Run on a Gilbreth login node. Reruns are idempotent. ~10-20 min first time.
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
if (!requireNamespace("DESeq2", quietly = TRUE)) {
  cat("Installing DESeq2 (+deps) ...\n")
  BiocManager::install("DESeq2", lib = Sys.getenv("R_LIBS_USER"),
                       update = FALSE, ask = FALSE, Ncpus = 4)
} else cat("DESeq2 already installed.\n")
suppressPackageStartupMessages(library(DESeq2))
cat(sprintf("DESeq2 version: %s\n", as.character(packageVersion("DESeq2"))))
cat(sprintf("varianceStabilizingTransformation exists: %s\n",
            exists("varianceStabilizingTransformation")))
RSCRIPT
echo "Install complete."
