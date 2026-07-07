#!/bin/bash
# run_gene_set_enrichment.sh -- R-env wrapper for Aim-2 pathway enrichment (Stage 13, step 4).
# fgsea (signed dose statistic) vs MSigDB Hallmark + Reactome over the per-cell-type DE blocks.
# Installs fgsea + msigdbr + msigdbdf into R_libs/ on first use (needs Bioconductor/CRAN/r-universe
# network on the compute node). Same R env pattern as run_pseudobulk_de.sh. Light after install.
set -eo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PROJECT_ROOT PIPELINE_ROOT="$PROJECT_ROOT"
command -v python3 >/dev/null 2>&1 && \
  eval "$(python3 "${PROJECT_ROOT}/deconvolution/_config_sh.py" 2>/dev/null || true)"
export R_LIBS_USER="${PROJECT_ROOT}/R_libs"
export TMPDIR="${PROJECT_ROOT}/tmp"
export R_PROFILE=/dev/null R_PROFILE_USER=/dev/null R_ENVIRON=/dev/null R_ENVIRON_USER=/dev/null
mkdir -p "${TMPDIR}"

# Load modules (R_MODULES) / strip conda (STRIP_CONDA) where site.env declares them; no-op elsewhere.
source "${PROJECT_ROOT}/deconvolution/setup/site_env.sh"
Rscript="Rscript --no-init-file --no-site-file"
command -v Rscript >/dev/null || { echo "FATAL: Rscript not on PATH after site_env.sh" >&2; exit 3; }

echo "== ensure fgsea + msigdbr (+ msigdbdf) present (install to R_libs if missing) =="
$Rscript -e '
  options(repos = c(CRAN = "https://cloud.r-project.org"))
  lib <- Sys.getenv("R_LIBS_USER"); ip <- rownames(installed.packages())
  for (p in c("BiocManager","data.table","babelgene","msigdbr"))
    if (!(p %in% ip)) install.packages(p, lib = lib)
  if (!requireNamespace("msigdbdf", quietly = TRUE))
    try(install.packages("msigdbdf",
        repos = c("https://igordot.r-universe.dev", "https://cloud.r-project.org"), lib = lib))
  if (!requireNamespace("fgsea", quietly = TRUE))
    BiocManager::install("fgsea", update = FALSE, ask = FALSE, lib = lib)
'

echo "== run enrichment =="
$Rscript "${PROJECT_ROOT}/deconvolution/R/run_gene_set_enrichment.R"
