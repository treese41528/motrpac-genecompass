#!/usr/bin/env Rscript
# score_z_vst.R -- Paper-faithful expression-accuracy metric for ONE focal cell
# type (Chu 2022, Nat Cancer, Fig. 1h): per bulk sample, Pearson correlation on
# DESeq2 variance-stabilizing transformed (VST) values between BayesPrism's
# inferred focal-type expression (pred Z) and the known ground-truth focal-type
# expression (true Z). Spearman on untransformed values is also reported (the
# paper computes Spearman on raw, Pearson on VST -- see Methods / ExtData Fig 6).
#
# VST is fit ONCE on the column-bind of [true | pred] focal matrices (genes x
# (2*samples)) so both are on a common variance-stabilized scale, then split.
#
# Inputs:
#   <stage_dir>/mixtures/true_z_focal.csv     samples x genes ground-truth focal Z
#   <stage_dir>/results/pred_z/predz__<safe(focal)>.csv  samples x genes pred focal Z
#   <stage_dir>/mixtures/sweep_design.tsv     mixture, target_purity, focal_rnafrac
# Output:
#   <stage_dir>/scores/z_vst_focal.tsv   per sample: purity, focal_rnafrac,
#                                        pearson_vst, spearman_raw, n_genes
#
# Args:  score_z_vst.R <stage_dir> <focal_type>
# Invoke via deconvolution/R/score_z_vst.sh (sets r/4.4.1 env + R_LIBS).
suppressWarnings(suppressPackageStartupMessages({
  .libPaths(c(Sys.getenv("R_LIBS_USER"), .libPaths()))
  library(DESeq2)
}))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) stop("usage: score_z_vst.R <stage_dir> <focal_type>")
sd <- args[1]; focal <- args[2]
# Shared cell-type -> filename contract. resolve_ct_file() prefers the current name but
# falls back to the legacy one, so sweep artifacts written before the 2026-07-12 sanitizer
# fix still load without a rebuild.
source(file.path(local({
  a <- commandArgs(trailingOnly = FALSE)
  f <- sub("^--file=", "", a[grep("^--file=", a)])
  if (length(f)) dirname(normalizePath(f[1])) else "."
}), "celltype_names.R"))

true <- as.matrix(read.csv(file.path(sd, "mixtures", "true_z_focal.csv"),
                           row.names = 1, check.names = FALSE))
predf <- resolve_ct_file(file.path(sd, "results", "pred_z"), focal, "predz__", ".csv")
pred <- as.matrix(read.csv(predf, row.names = 1, check.names = FALSE))
design <- read.delim(file.path(sd, "mixtures", "sweep_design.tsv"),
                     stringsAsFactors = FALSE)

# align genes (pred may drop genes via BayesPrism filtering) and samples
common <- intersect(colnames(true), colnames(pred))
true <- true[, common, drop = FALSE]; pred <- pred[, common, drop = FALSE]
stopifnot(nrow(true) == nrow(pred))
cat(sprintf("focal=%s | %d samples x %d common genes\n",
            focal, nrow(true), length(common)))

# round to integer counts for DESeq2; fit VST once on [true|pred] together
mat <- t(rbind(true, pred))                  # genes x (2*samples)
mode(mat) <- "integer"; mat[is.na(mat)] <- 0L
storage.mode(mat) <- "integer"
mat[mat < 0L] <- 0L
keep <- rowSums(mat) > 0                      # drop all-zero genes (VST needs them out)
mat <- mat[keep, , drop = FALSE]
vsd <- tryCatch(
  varianceStabilizingTransformation(mat, blind = TRUE),
  error = function(e) {                       # fallback: per-column geomean fitType
    dds <- DESeqDataSetFromMatrix(mat, data.frame(s = factor(seq_len(ncol(mat)))), ~1)
    assay(varianceStabilizingTransformation(dds, blind = TRUE))
  })
n <- nrow(true)
vt <- vsd[, 1:n, drop = FALSE]               # true (VST)
vp <- vsd[, (n + 1):(2 * n), drop = FALSE]   # pred (VST)

rows <- lapply(seq_len(n), function(i) {
  pear <- if (sd(vt[, i]) > 0 && sd(vp[, i]) > 0) cor(vt[, i], vp[, i]) else NA
  spear <- if (sd(true[i, ]) > 0 && sd(pred[i, ]) > 0)
    suppressWarnings(cor(true[i, ], pred[i, ], method = "spearman")) else NA
  data.frame(mixture = rownames(true)[i], pearson_vst = pear,
             spearman_raw = spear, n_genes = length(common))
})
res <- do.call(rbind, rows)
res <- merge(design[, c("mixture", "target_purity", "focal_rnafrac")], res,
             by = "mixture", all.y = TRUE)
res <- res[order(res$target_purity, res$mixture), ]

out <- file.path(sd, "scores"); dir.create(out, recursive = TRUE, showWarnings = FALSE)
write.table(res, file.path(out, "z_vst_focal.tsv"), sep = "\t",
            quote = FALSE, row.names = FALSE)
cat("\n=== per-sample focal-expression accuracy (head) ===\n")
print(utils::head(res, 12))
cat(sprintf("\nWrote %s/z_vst_focal.tsv\n", out))
