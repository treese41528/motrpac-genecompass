#!/usr/bin/env Rscript
# extract_z.R -- Extract BayesPrism's per-cell-type, per-sample expression Z from a
# saved bp_result.rds, for scoring against the known pseudobulk ground truth
# (Stage 8 Z validation). No Gibbs re-run: get.exp reads the stored posterior.
#
# get.exp(bp, "type", t) returns a sample x gene matrix for cell type t -- the
# portion of each bulk sample's counts BayesPrism attributes to t
# (bp@posterior.initial.cellType@Z[,,t]). We write one CSV per cell type plus the
# gene/type order into <out_dir>/pred_z/.
#
# Args:  extract_z.R <bp_result.rds> <out_dir>
# Invoke via deconvolution/R/extract_z.sh (sets the Gilbreth r/4.4.1 env).
suppressWarnings(suppressPackageStartupMessages({
  .libPaths(c(Sys.getenv("R_LIBS_USER"), .libPaths()))
  library(BayesPrism)
}))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) stop("usage: extract_z.R <bp_result.rds> <out_dir>")
rds <- args[1]; out_dir <- args[2]
zdir <- file.path(out_dir, "pred_z")
dir.create(zdir, recursive = TRUE, showWarnings = FALSE)

bp <- readRDS(rds)
if (is.null(bp@posterior.initial.cellType) ||
    length(dim(bp@posterior.initial.cellType@Z)) != 3)
  stop("bp_result.rds has no per-sample Z (posterior.initial.cellType@Z); ",
       "run.prism must be re-run with Z stored.")

Z <- bp@posterior.initial.cellType@Z          # sample x gene x cell.type
types <- dimnames(Z)[[3]]; if (is.null(types)) types <- colnames(bp@posterior.theta_f@theta)
genes <- dimnames(Z)[[2]]
writeLines(genes, file.path(zdir, "genes.txt"))
writeLines(types, file.path(zdir, "types.txt"))
cat(sprintf("Z dims: %d samples x %d genes x %d types\n", dim(Z)[1], dim(Z)[2], dim(Z)[3]))

safe <- function(s) gsub("[^A-Za-z0-9]+", "_", s)
for (t in types) {
  m <- get.exp(bp = bp, state.or.type = "type", cell.name = t)   # sample x gene
  write.csv(m, file.path(zdir, paste0("predz__", safe(t), ".csv")))
  cat(sprintf("  wrote %-28s %d x %d\n", t, nrow(m), ncol(m)))
}
cat(sprintf("Wrote pred Z -> %s/\n", zdir))
