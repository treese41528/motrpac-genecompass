#!/usr/bin/env Rscript
# =============================================================================
# simulate_simbu.R -- SimBu-based known-truth mixtures for the omnideconv
# confounder battery (OMNIDECONV_BENCHMARK_PLAN.md Phase 3). Reproduces the
# paper's four confounder scenarios on OUR rat references:
#   --scenario random   --scaling expressed_genes|NONE   (mRNA-content bias, Fig 4)
#   --scenario pure      [--pure-cell-type T]            (spillover, Fig 5A/B)
#   --scenario weighted  --weighted-cell-type T --weighted-amount f  (dominant sweep, Fig 5C)
#   --scenario mirror_db                                 (granularity, Fig 3)
#
# Self-contained: downsamples the reference to --cells-per-type (paper uses 500;
# also keeps DWLS/MAST fast + in-memory), builds a SimBu dataset, simulates bulk,
# and writes BOTH the (downsampled) reference/ AND mixtures/ in the SAME on-disk
# format make_pseudobulk.py emits (pseudobulk_counts.mtx samples x genes,
# pseudobulk_genes.tsv, true_fractions.tsv), so run_omnideconv.sh /
# run_deconvolution.sh / score_validation.py all run on the output UNCHANGED.
# Deconvolve against out/reference so the ONLY moving part is the confounder.
#
# Truth = SimBu cell_fractions (the cell-composition recovery target): written as
# cellfrac__<type>; score with `score_validation.py --truth cellfrac`. Under
# mRNA-bias the bulk is mRNA-weighted while cellfrac is fixed, so methods that do
# not correct the bias show higher RMSE -- the ΔRMSE(bias - no-bias) analysis.
#
# Seeded end-to-end: same --seed => identical downsample AND composition, so a
# bias/no-bias pair (same seed, --scaling expressed_genes vs NONE) is matched.
#
# Usage (via simulate_simbu.sh, which sets the project R env):
#   simulate_simbu.sh --ref-dir <dir> --out <dir> [--scenario random]
#     [--scaling expressed_genes] [--nsamples 50] [--ncells 1000]
#     [--cells-per-type 500] [--seed 1] [--total-reads 1000000]
#     [--weighted-cell-type T --weighted-amount 0.5] [--pure-cell-type T]
# =============================================================================
suppressMessages({library(SimBu); library(Matrix)})

args <- commandArgs(trailingOnly = TRUE)
getarg <- function(key, default = NULL) {
  i <- which(args == paste0("--", key))
  if (length(i) == 1L && i < length(args)) args[i + 1L] else default
}
ref_dir <- getarg("ref-dir"); out <- getarg("out")
if (is.null(ref_dir) || is.null(out))
  stop("usage: --ref-dir <dir> --out <dir> [--scenario random] [--scaling expressed_genes] ",
       "[--nsamples 50] [--ncells 1000] [--cells-per-type 500] [--seed 1] ",
       "[--total-reads 1000000] [--weighted-cell-type T --weighted-amount 0.5] [--pure-cell-type T]")
scenario    <- getarg("scenario", "random")
scaling     <- getarg("scaling", "expressed_genes")
nsamples    <- as.integer(getarg("nsamples", "50"))
ncells      <- as.integer(getarg("ncells", "1000"))
cap         <- as.integer(getarg("cells-per-type", "500"))
seed        <- as.integer(getarg("seed", "1"))
total_reads <- as.numeric(getarg("total-reads", "1000000"))
w_type      <- getarg("weighted-cell-type"); w_amt <- as.numeric(getarg("weighted-amount", "0.5"))
p_type      <- getarg("pure-cell-type")
ba          <- getarg("bias-alpha"); bias_alpha <- if (is.null(ba)) NA_real_ else as.numeric(ba)

dir.create(file.path(out, "mixtures"),  recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(out, "reference"), recursive = TRUE, showWarnings = FALSE)
cat(sprintf("SimBu %s | ref=%s | scenario=%s scaling=%s nsamples=%d ncells=%d cap=%d seed=%d\n",
            as.character(packageVersion("SimBu")), ref_dir, scenario, scaling,
            nsamples, ncells, cap, seed))

# ---- read reference (cells x genes) ----
ref   <- readMM(file.path(ref_dir, "reference_counts.mtx"))
genes <- readLines(file.path(ref_dir, "genes.tsv"))
meta  <- read.delim(file.path(ref_dir, "cells_meta.tsv"), stringsAsFactors = FALSE)
stopifnot(nrow(ref) == nrow(meta), ncol(ref) == length(genes))
if (!"cell_type" %in% colnames(meta)) stop("cells_meta.tsv lacks a cell_type column")
if (!"barcode"  %in% colnames(meta)) meta$barcode <- paste0("cell", seq_len(nrow(meta)))
meta$barcode <- make.unique(as.character(meta$barcode))

# ---- downsample to cap cells/type (seeded, logged) ----
set.seed(seed)
if (!is.na(cap) && cap > 0L) {
  keep <- sort(unlist(lapply(split(seq_len(nrow(meta)), meta$cell_type), function(ix)
    if (length(ix) > cap) sample(ix, cap) else ix), use.names = FALSE))
  cat(sprintf("downsample: %d -> %d cells (<=%d/type over %d types)\n",
              nrow(ref), length(keep), cap, length(unique(meta$cell_type))))
  ref <- ref[keep, , drop = FALSE]; meta <- meta[keep, , drop = FALSE]
}
ct <- meta$cell_type

# ---- SimBu dataset (genes x cells; filter_genes=FALSE keeps the ref gene space) ----
cnt <- as(t(ref), "CsparseMatrix"); rownames(cnt) <- genes; colnames(cnt) <- meta$barcode
anno <- data.frame(ID = meta$barcode, cell_type = ct, stringsAsFactors = FALSE)
ds <- SimBu::dataset(annotation = anno, count_matrix = cnt, name = basename(out),
                     filter_genes = FALSE)

# ---- simulate ----
if (!is.na(bias_alpha)) {
  # DOSE-RESPONSE mode: per-cell-type mRNA scaling s_k = (mean expressed genes_k)^alpha with a
  # PINNED composition (custom_scenario_data) so the cell-fraction truth is IDENTICAL across the
  # alpha sweep -- ONLY the mRNA-bias amplitude varies. seed fixes the downsampled ref AND the
  # composition, so every alpha shares them (a controlled magnitude test). alpha=0 => no bias.
  types <- sort(unique(ct))
  gi <- Matrix::rowSums(ref > 0)                     # expressed genes per cell (ref = cells x genes)
  tmean <- tapply(gi, ct, mean)[types]
  s <- (tmean / mean(tmean)) ^ bias_alpha; s <- s / mean(s); names(s) <- types
  set.seed(seed + 1000L)                             # composition stream, independent of alpha
  comp <- t(vapply(seq_len(nsamples), function(i) { d <- rgamma(length(types), 1); d / sum(d) },
                   numeric(length(types))))
  colnames(comp) <- types
  cat(sprintf("dose-response: alpha=%.2f | per-type mRNA-scale range [%.2f, %.2f] | %d pinned samples\n",
              bias_alpha, min(s), max(s), nsamples))
  res <- SimBu::simulate_bulk(ds, scenario = "custom", custom_scenario_data = as.data.frame(comp),
                              scaling_factor = "custom", custom_scaling_vector = s,
                              ncells = ncells, nsamples = nsamples, total_read_counts = total_reads,
                              seed = seed, run_parallel = FALSE)
} else {
  sim_args <- list(data = ds, scenario = scenario, scaling_factor = scaling,
                   nsamples = nsamples, ncells = ncells, total_read_counts = total_reads,
                   seed = seed, run_parallel = FALSE)
  if (scenario == "weighted" && !is.null(w_type)) {
    sim_args$weighted_cell_type <- w_type; sim_args$weighted_amount <- w_amt
  }
  if (scenario == "pure" && !is.null(p_type)) sim_args$pure_cell_type <- p_type
  res <- do.call(SimBu::simulate_bulk, sim_args)
}

bulk  <- as.matrix(SummarizedExperiment::assays(res$bulk)[["bulk_counts"]])  # genes x samples
fracs <- as.matrix(res$cell_fractions)                                        # samples x types
bgenes <- rownames(bulk)
stopifnot(nrow(fracs) == ncol(bulk))

# ---- write the (downsampled) reference: deconvolve against the SAME cells ----
writeMM(as(ref, "CsparseMatrix"), file.path(out, "reference", "reference_counts.mtx"))
writeLines(genes, file.path(out, "reference", "genes.tsv"))
write.table(meta, file.path(out, "reference", "cells_meta.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE)

# ---- write mixtures (samples x genes) in the make_pseudobulk format ----
pb <- as(Matrix(round(t(bulk)), sparse = TRUE), "CsparseMatrix")             # samples x genes
writeMM(pb, file.path(out, "mixtures", "pseudobulk_counts.mtx"))
writeLines(bgenes, file.path(out, "mixtures", "pseudobulk_genes.tsv"))
truth <- data.frame(mixture = rownames(fracs), check.names = FALSE)
for (tp in colnames(fracs)) truth[[paste0("cellfrac__", tp)]] <- fracs[, tp]  # recovery target
write.table(truth, file.path(out, "mixtures", "true_fractions.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE)
writeLines(c(paste0("scenario=", scenario), paste0("scaling=", scaling),
             paste0("nsamples=", nsamples), paste0("ncells=", ncells),
             paste0("cells_per_type=", cap), paste0("seed=", seed),
             paste0("bias_alpha=", bias_alpha),
             paste0("weighted_cell_type=", ifelse(is.null(w_type), "", w_type)),
             paste0("weighted_amount=", w_amt),
             paste0("pure_cell_type=", ifelse(is.null(p_type), "", p_type)),
             paste0("n_ref_cells=", nrow(ref)), paste0("n_types=", ncol(fracs)),
             paste0("n_genes_bulk=", length(bgenes))),
           file.path(out, "mixtures", "sim_meta.txt"))
cat(sprintf("wrote %d mixtures x %d genes; %d types; ref=%d cells -> %s/{reference,mixtures}\n",
            ncol(bulk), length(bgenes), ncol(fracs), nrow(ref), out))
