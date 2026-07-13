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

# Shared cell-type -> filename contract. This script is where the KIDNEY alpha/beta
# collision destroyed a Z matrix, so the injectivity assert below is load-bearing.
.script_dir <- local({
  a <- commandArgs(trailingOnly = FALSE)
  f <- sub("^--file=", "", a[grep("^--file=", a)])
  if (length(f)) dirname(normalizePath(f[1])) else "."
})
source(file.path(.script_dir, "celltype_names.R"))

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

# One file per cell type -- so two cell types must never claim the same filename.
assert_ct_injective(types, sprintf("%s pred_z", basename(out_dir)))
keep <- sprintf("predz__%s.csv", safe(types))
purge_stale(zdir, "^predz__.*\\.csv$", keep = keep)

for (t in types) {
  m <- get.exp(bp = bp, state.or.type = "type", cell.name = t)   # sample x gene
  write.csv(m, file.path(zdir, paste0("predz__", safe(t), ".csv")))
  cat(sprintf("  wrote %-28s %d x %d\n", t, nrow(m), ncol(m)))
}
write_ct_manifest(file.path(zdir, "celltype_files.tsv"),
                  basename(out_dir), types, "predz__", ".csv")
cat(sprintf("Wrote pred Z (%d types) + celltype_files.tsv -> %s/\n", length(types), zdir))
