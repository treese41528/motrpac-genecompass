#!/bin/bash
# Install R BayesPrism v2.2.3+ into the project-local R library.
#
# Rationale: project is locked on the official R BayesPrism from
# Danko-Lab — not pyBayesPrism. See onboarding §9.1 / project memory.
#
# Run on a Gilbreth login node (no compute resources needed for install).
# Total time: ~20-30 min (Bioconductor scran + transitive deps).
#
# Reruns are idempotent — install.packages skips already-installed pkgs,
# except for Matrix which is force-installed to override the stale base
# version that ships with r/4.4.1.

set -eo pipefail   # -u dropped: Gilbreth's /etc/profile.d/00-modulepath.sh references unbound vars

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export R_LIBS_USER="${PROJECT_ROOT}/R_libs"

# Project-local tmp for R's scratch (downloaded_packages, build dirs) — never /tmp.
export TMPDIR="${PROJECT_ROOT}/tmp"

# Suppress Gilbreth's site Rprofile that autoloads Rhipe and aborts under R CMD INSTALL.
export R_PROFILE=/dev/null
export R_PROFILE_USER=/dev/null
export R_ENVIRON=/dev/null
export R_ENVIRON_USER=/dev/null

mkdir -p "${R_LIBS_USER}" "${TMPDIR}"

source /etc/profile.d/modules.sh
module load r/4.4.1

# Strip Gilbreth's system-Python env from build-time discovery.
# /apps/external/conda/2025.09/ is the cluster-supplied Python distribution
# auto-loaded on PATH by default. It sits at the front of PKG_CONFIG_PATH,
# which makes pkg-config hand igraph (and friends) the env's libxml2. That
# libxml2 pulls libicui18n.so.75 as a runtime dependency, but the env's lib
# dir is not on LD_LIBRARY_PATH at R session load time — so igraph.so
# dlopens and fails. Stripping the path lets pkg-config fall back to the
# spack libxml2 module (further down PKG_CONFIG_PATH), which is
# self-contained on standard rpaths.
# (This project's Python lives in a venv at /depot/reese18/apps/motrpac-env/,
# not in this env — that venv stays on PATH and is unaffected.)
export PKG_CONFIG_PATH=$(
  echo "${PKG_CONFIG_PATH:-}" | tr ':' '\n' \
  | grep -v '/apps/external/conda/2025.09' | tr '\n' ':' | sed 's/:$//'
)
export PATH=$(
  echo "${PATH}" | tr ':' '\n' \
  | grep -v '/apps/external/conda/2025.09' | tr '\n' ':' | sed 's/:$//'
)

echo "R           : $(which R)"
echo "R version   : $(R --version | head -1)"
echo "R_LIBS_USER : ${R_LIBS_USER}"
echo "TMPDIR      : ${TMPDIR}"
echo

# Helper: invoke R with site/user profiles suppressed and project libPaths preferred.
run_R() {
  R --no-init-file --no-site-file --slave "$@"
}

run_R <<'RSCRIPT'
options(repos = c(CRAN = "https://cloud.r-project.org"))
.libPaths(c(Sys.getenv("R_LIBS_USER"), .libPaths()))
cat("Active libPaths:\n"); print(.libPaths())

# 0. Matrix — force install to user lib BEFORE any Bioconductor work.
#    The base Matrix shipped with r/4.4.1 is too old for current SparseArray
#    (missing exports `crossprod`, `tcrossprod`), which silently kills the
#    scran cascade later. Forcing a CRAN install puts a newer Matrix at the
#    front of .libPaths() so subsequent installs pick it up. No reload needed
#    in this R session — downstream install.packages() spawns child R
#    processes that re-resolve .libPaths() and see the new Matrix first.
if (packageVersion("Matrix") < "1.7.0" ||
    !"Matrix" %in% rownames(installed.packages(lib.loc = Sys.getenv("R_LIBS_USER")))) {
  cat("\n[0/4] Force-installing Matrix from CRAN (overrides stale base version)...\n")
  install.packages("Matrix", lib = Sys.getenv("R_LIBS_USER"), dependencies = TRUE)
} else {
  cat("\n[0/4] Modern Matrix already installed in user lib.\n")
}
cat(sprintf("Matrix in user lib: %s\n",
            as.character(packageVersion("Matrix",
              lib.loc = Sys.getenv("R_LIBS_USER")))))

# 1. BiocManager
if (!requireNamespace("BiocManager", quietly = TRUE)) {
  cat("\n[1/4] Installing BiocManager...\n")
  install.packages("BiocManager", lib = Sys.getenv("R_LIBS_USER"))
} else {
  cat("\n[1/4] BiocManager already installed.\n")
}

# 2. Pure-CRAN deps (no Bioc dependencies). NMF is intentionally deferred to
#    step 4 because it depends on Biobase, which only arrives via the Bioc
#    step.
cran_deps <- c("snowfall", "gplots")
missing_cran <- cran_deps[!sapply(cran_deps, requireNamespace, quietly = TRUE)]
if (length(missing_cran)) {
  cat(sprintf("\n[2/5] Installing pure-CRAN deps: %s\n", paste(missing_cran, collapse = ", ")))
  install.packages(missing_cran, lib = Sys.getenv("R_LIBS_USER"))
} else {
  cat("\n[2/5] All pure-CRAN deps already installed.\n")
}

# 3. Bioconductor deps (the long pole — scran pulls 50+ transitive packages,
#    including igraph and Biobase. The conda-stripped env above lets igraph
#    compile cleanly against spack libxml2.)
bioc_deps <- c("scran", "BiocParallel")
missing_bioc <- bioc_deps[!sapply(bioc_deps, requireNamespace, quietly = TRUE)]
if (length(missing_bioc)) {
  cat(sprintf("\n[3/5] Installing Bioconductor deps (slow): %s\n",
              paste(missing_bioc, collapse = ", ")))
  BiocManager::install(missing_bioc, lib = Sys.getenv("R_LIBS_USER"),
                       update = FALSE, ask = FALSE, Ncpus = 4)
} else {
  cat("\n[3/5] All Bioconductor deps already installed.\n")
}

# 4. NMF — CRAN package that depends on Biobase (Bioconductor), so it has
#    to come after step 3.
if (!requireNamespace("NMF", quietly = TRUE)) {
  cat("\n[4/5] Installing NMF (depends on Biobase from step 3)...\n")
  install.packages("NMF", lib = Sys.getenv("R_LIBS_USER"))
} else {
  cat("\n[4/5] NMF already installed.\n")
}
RSCRIPT

# 5. BayesPrism from local vendored source (v2.2.3, audited).
#    R CMD INSTALL bypasses --no-site-file flags; suppress via R_PROFILE env.
echo
echo "[5/5] Installing BayesPrism from vendor/BayesPrism/BayesPrism/..."
R CMD INSTALL --library="${R_LIBS_USER}" \
  "${PROJECT_ROOT}/vendor/BayesPrism/BayesPrism"

echo
echo "=== Verification ==="
run_R <<'RSCRIPT'
.libPaths(c(Sys.getenv("R_LIBS_USER"), .libPaths()))
suppressPackageStartupMessages(library(BayesPrism))
cat(sprintf("BayesPrism version : %s\n", as.character(packageVersion("BayesPrism"))))
cat(sprintf("BayesPrism location: %s\n", find.package("BayesPrism")))
cat(sprintf("Matrix version     : %s\n", as.character(packageVersion("Matrix"))))
cat(sprintf("scran version      : %s\n", as.character(packageVersion("scran"))))
cat(sprintf("new.prism exists   : %s\n", exists("new.prism")))
cat(sprintf("run.prism exists   : %s\n", exists("run.prism")))
RSCRIPT

echo
echo "Install complete."
