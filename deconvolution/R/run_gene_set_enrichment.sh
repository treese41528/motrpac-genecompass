#!/bin/bash
# run_gene_set_enrichment.sh -- R-env wrapper for Aim-2 pathway enrichment (Stage 13, step 4).
# fgsea (signed dose statistic) vs MSigDB Hallmark + Reactome over the per-cell-type DE blocks.
# Installs fgsea + msigdbr + msigdbdf into R_libs/ on first use (needs Bioconductor/CRAN/r-universe
# network on the compute node). Same R env pattern as run_pseudobulk_de.sh. Light after install.
#
# Environment read by run_gene_set_enrichment.R and passed through unchanged (see its header):
#   DE_DIR          DE folder to analyse       (default: <genecompass_input_dir>/pseudobulk_de of the config)
#   ENRICH_OUT      output folder              (default: <DE_DIR>/enrichment); set it to leave DE_DIR untouched
#   GSEA_GENE_SETS  a saved gene_sets_used.tsv (default: the table in the output folder, else msigdbr)
#   GSEA_SEED       fgsea seed                 (default 1)
# The gene-set table a run used is saved beside its results with its md5 (STEP3_PREREG.md 8.2), so a later
# run that is given that table does not read msigdbr's per-user cache at all.
set -eo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PROJECT_ROOT PIPELINE_ROOT="$PROJECT_ROOT"
# Config read is fatal on failure (no fallback to a hardcoded arm); exports every CFG_*.
source "${PROJECT_ROOT}/deconvolution/load_config.sh"
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
