#!/usr/bin/env Rscript
# =============================================================================
# check_simbu_arms.R -- what each arm of the V5 battery (docs/audit/remediation/STEP4_PREREG.md) does to the
# library size of the cells SimBu sums. It answers one question with SimBu's own code: does arm A remove the
# differences in library size between cell types, and how large are they in arms B and C?
#
# For one reference, downsampled exactly as simulate_simbu.R does (same seed, same code), and for every cell
# type and arm, SimBu::simulate_bulk draws ONE "pure" sample of --ncells cells of that type with
# total_read_counts = NULL, so nothing is rescaled after the cells are summed. The total of that sample divided
# by --ncells is the mean library size of the type's cells AS SIMBU SUMS THEM:
#   arm A  scaling NONE, remove_bias_in_counts TRUE ("read-number")   -> 1 for every type if the bias is removed
#   arm B  scaling NONE, remove_bias_in_counts FALSE                  -> the type's mean raw library size
#   arm C  scaling expressed_genes, remove_bias_in_counts FALSE       -> mean of (library size x expressed genes)
# The same cells are drawn in the three arms (SimBu seeds the draw per sample).
#
# Output (--out, a TSV), one row per cell type and arm:
#   code, cell_type, n_ref_cells, ref_mean_library_size, ref_mean_expressed_genes   the downsampled reference
#   arm, scaling_factor, remove_bias_in_counts, remove_bias_in_counts_method        the simulate_bulk call
#   mean_library_size_as_summed     total of the pure sample / ncells
#   relative_to_type_mean           that, divided by its mean over the reference's cell types (1 = no difference)
# and a printed line per arm with the largest/smallest ratio across types.
#
# Usage (SIMBU_SCRIPT=check_simbu_arms.R simulate_simbu.sh ...):
#   --ref-dir <dir> --out <file.tsv> [--code NAME] [--cells-per-type 500] [--seed 1] [--ncells 1000]
# =============================================================================
suppressMessages({library(SimBu); library(Matrix)})

args <- commandArgs(trailingOnly = TRUE)
getarg <- function(key, default = NULL) {
  i <- which(args == paste0("--", key))
  if (length(i) == 1L && i < length(args)) args[i + 1L] else default
}
ref_dir <- getarg("ref-dir"); out <- getarg("out")
if (is.null(ref_dir) || is.null(out))
  stop("usage: --ref-dir <dir> --out <file.tsv> [--code NAME] [--cells-per-type 500] [--seed 1] [--ncells 1000]")
code   <- getarg("code", basename(ref_dir))
cap    <- as.integer(getarg("cells-per-type", "500"))
seed   <- as.integer(getarg("seed", "1"))
ncells <- as.integer(getarg("ncells", "1000"))
cat(sprintf("SimBu %s | check of the three arms | ref=%s cap=%d seed=%d ncells=%d\n",
            as.character(packageVersion("SimBu")), ref_dir, cap, seed, ncells))

# ---- reference, downsampled as in simulate_simbu.R ----
ref   <- readMM(file.path(ref_dir, "reference_counts.mtx"))
genes <- readLines(file.path(ref_dir, "genes.tsv"))
meta  <- read.delim(file.path(ref_dir, "cells_meta.tsv"), stringsAsFactors = FALSE)
stopifnot(nrow(ref) == nrow(meta), ncol(ref) == length(genes))
if (!"barcode" %in% colnames(meta)) meta$barcode <- paste0("cell", seq_len(nrow(meta)))
meta$barcode <- make.unique(as.character(meta$barcode))
set.seed(seed)
if (!is.na(cap) && cap > 0L) {
  keep <- sort(unlist(lapply(split(seq_len(nrow(meta)), meta$cell_type), function(ix)
    if (length(ix) > cap) sample(ix, cap) else ix), use.names = FALSE))
  cat(sprintf("downsample: %d -> %d cells (<=%d/type over %d types)\n",
              nrow(ref), length(keep), cap, length(unique(meta$cell_type))))
  ref <- ref[keep, , drop = FALSE]; meta <- meta[keep, , drop = FALSE]
}
ct <- meta$cell_type
lib <- Matrix::rowSums(ref); ngenes <- Matrix::rowSums(ref > 0)

cnt <- as(t(ref), "CsparseMatrix"); rownames(cnt) <- genes; colnames(cnt) <- meta$barcode
anno <- data.frame(ID = meta$barcode, cell_type = ct, stringsAsFactors = FALSE)
ds <- SimBu::dataset(annotation = anno, count_matrix = cnt, name = "arm_check", filter_genes = FALSE)

arms <- list(A = list(scaling = "NONE", remove = TRUE),
             B = list(scaling = "NONE", remove = FALSE),
             C = list(scaling = "expressed_genes", remove = FALSE))
METHOD <- "read-number"
rows <- list()
for (tp in sort(unique(ct))) {
  for (a in names(arms)) {
    res <- suppressMessages(SimBu::simulate_bulk(
      data = ds, scenario = "pure", pure_cell_type = tp, scaling_factor = arms[[a]]$scaling,
      remove_bias_in_counts = arms[[a]]$remove, remove_bias_in_counts_method = METHOD,
      nsamples = 1, ncells = ncells, total_read_counts = NULL, seed = seed, run_parallel = FALSE))
    tot <- sum(SummarizedExperiment::assays(res$bulk)[["bulk_counts"]])
    rows[[length(rows) + 1L]] <- data.frame(
      code = code, cell_type = tp, n_ref_cells = sum(ct == tp),
      ref_mean_library_size = mean(lib[ct == tp]), ref_mean_expressed_genes = mean(ngenes[ct == tp]),
      arm = a, scaling_factor = arms[[a]]$scaling, remove_bias_in_counts = arms[[a]]$remove,
      remove_bias_in_counts_method = METHOD, mean_library_size_as_summed = tot / ncells,
      stringsAsFactors = FALSE)
  }
}
d <- do.call(rbind, rows)
d$relative_to_type_mean <- d$mean_library_size_as_summed / ave(d$mean_library_size_as_summed, d$arm)
dir.create(dirname(out), recursive = TRUE, showWarnings = FALSE)
write.table(d, out, sep = "\t", quote = FALSE, row.names = FALSE)
for (a in names(arms)) {
  x <- d$mean_library_size_as_summed[d$arm == a]
  cat(sprintf("arm %s: mean library size as summed, across %d types: min %.6g max %.6g (max/min %.4f)\n",
              a, length(x), min(x), max(x), max(x) / min(x)))
}
cat(sprintf("wrote %s\n", out))
