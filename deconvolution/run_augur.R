#!/usr/bin/env Rscript
# run_augur.R -- CANONICAL Augur (neurorestore/Augur 1.0.3) corroboration of the Aim-2
# supervised-subspace finding (deconvolution/subspace_probe.py). Augur is the published
# standard for cell-type perturbation-responsiveness: per cell type it trains a
# cross-validated Random Forest and reports the AUC of separating the condition, with
# abundance-balanced subsampling (Skinnider/Squair, Nat Biotech 2021).
#
# For each tissue it runs Augur (RF, velocity mode = treat columns as continuous features,
# no gene/count normalization or variance selection) on TWO representations of the same
# pseudo-cells -- the GeneCompass 768-d embedding and PCA-50 of the deconvolved Z -- for
# two conditions: trained-vs-control (exercise) and sex (positive control). This lets the
# published RF method (vs our linear PLS-1 probe) weigh in, and on embedding-vs-PCA too.
#
# Inputs : data/deconvolution/augur_input/<tissue>/{embed.tsv,pca.tsv,meta.tsv}  (augur_prep.py)
# Output : data/deconvolution/genecompass_input/augur_results.tsv  (tissue, representation,
#          condition, cell_type, augur_auc)
#
# Run: R_LIBS_USER=$PWD/R_libs module load r/4.4.1 ; Rscript --no-init-file deconvolution/run_augur.R
suppressMessages({ library(Augur) })

args    <- commandArgs(TRUE)
IN      <- if (length(args) >= 1) args[1] else "data/deconvolution/augur_input"
OUT     <- if (length(args) >= 2) args[2] else "data/deconvolution/genecompass_input/augur_results.tsv"
NTHREAD <- as.integer(Sys.getenv("AUGUR_THREADS", "8"))
# control arm has only 10 pseudo-cells per cell type (2 sex x 5 reps), so Augur's default
# subsample_size=20 (which requires >=20 in EVERY condition) drops all cell types. Balance 10v10.
SS      <- as.integer(Sys.getenv("AUGUR_SUBSAMPLE", "10"))
MC      <- as.integer(Sys.getenv("AUGUR_MINCELLS", "10"))

read_rep <- function(path) {                       # cells x dims TSV -> dims x cells matrix
  m <- as.matrix(read.table(path, sep = "\t", header = TRUE, row.names = 1, check.names = FALSE))
  t(m)
}

run_one <- function(X, meta, mode = "velocity") {
  tryCatch(
    calculate_auc(X, meta, label_col = "label", cell_type_col = "cell_type",
                  augur_mode = mode, classifier = "rf", n_threads = NTHREAD,
                  subsample_size = SS, min_cells = MC, show_progress = FALSE),
    error = function(e) {                          # fallback: default mode, no variance selection
      if (mode == "velocity")
        tryCatch(calculate_auc(X, meta, label_col = "label", cell_type_col = "cell_type",
                               augur_mode = "default", select_var = FALSE, classifier = "rf",
                               n_threads = NTHREAD, subsample_size = SS, min_cells = MC,
                               show_progress = FALSE),
                 error = function(e2) { cat("    FAIL:", conditionMessage(e2), "\n"); NULL })
      else { cat("    FAIL:", conditionMessage(e), "\n"); NULL }
    })
}

res <- list()
for (td in list.dirs(IN, recursive = FALSE)) {
  tis   <- basename(td)
  meta0 <- read.table(file.path(td, "meta.tsv"), sep = "\t", header = TRUE, stringsAsFactors = FALSE)
  rownames(meta0) <- meta0$rowid          # unique positional id (cell_id can collide across subtypes)
  for (rep in c("embed", "pca")) {
    f <- file.path(td, paste0(rep, ".tsv")); if (!file.exists(f)) next
    X <- read_rep(f)
    m0 <- meta0[colnames(X), , drop = FALSE]
    for (cond in c("trained", "sex")) {
      lab  <- if (cond == "trained") m0$label else m0$sex
      meta <- data.frame(cell_type = m0$cell_type, label = lab, row.names = rownames(m0),
                         stringsAsFactors = FALSE)
      meta <- meta[!is.na(meta$label) & meta$label != "None", , drop = FALSE]
      if (length(unique(meta$label)) < 2) next
      au <- run_one(X[, rownames(meta), drop = FALSE], meta)
      if (is.null(au)) next
      a <- au$AUC
      res[[length(res) + 1]] <- data.frame(tissue = tis, representation = rep, condition = cond,
                                           cell_type = a$cell_type, augur_auc = a$auc)
      cat(sprintf("  [%-6s %-5s %-7s] %2d cell types, median AUC %.3f, max %.3f\n",
                  tis, rep, cond, nrow(a), median(a$auc), max(a$auc)))
    }
  }
}
out <- do.call(rbind, res)
write.table(out, OUT, sep = "\t", quote = FALSE, row.names = FALSE)
cat("\nwrote", OUT, "(", nrow(out), "rows )\n")
