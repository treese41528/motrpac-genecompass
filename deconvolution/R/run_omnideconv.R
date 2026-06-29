#!/usr/bin/env Rscript
# Multi-method omnideconv cross-check (Stage 8.5) -- production decision #4: validate our
# patched BayesPrism fractions against an independent panel of reference-based methods.
#
# Consumes the SAME reference + mixture dirs as run_deconvolution.R (BayesPrism), so the
# per-cell-type fractions are directly comparable method-vs-method, vs BayesPrism, and vs
# ground truth:
#   reference dir : reference_counts.mtx (cells x genes), genes.tsv,
#                   cells_meta.tsv (barcode, sample, cell_type, cell_state)
#   mixture dir   : <base>.mtx (samples x genes) + <base less _counts>_genes.tsv
# omnideconv expects genes x cells (reference) and genes x samples (bulk) -- the transpose
# of BayesPrism's layout -- so we transpose both. cells_meta$sample -> batch_ids, which the
# subject-aware methods (MuSiC, SCDC, Bisque) use to model cross-subject variation.
#
# The reference gets the SAME rat gene cleanup as run_deconvolution.R (ribo/mito/hb, sex
# chromosomes, protein-coding, low-expression), driven by the same env vars, so the gene
# universe matches BayesPrism's. (Logic mirrors run_deconvolution.R:48-91; kept inline
# rather than shared to avoid disturbing the validated BayesPrism path.)
#
# Args:  run_omnideconv.R <ref_dir> <mixture_dir> <out_dir> [mtx_basename]
#   mtx_basename defaults to "pseudobulk_counts" (validation mixtures).
# Env:   OMNIDECONV_METHODS  csv of omnideconv internal method names
#                            (default "music,dwls,scdc,bisque"; cdseq/bseqsc are opt-in --
#                            cdseq is a slow Gibbs sampler needing harmony, bseqsc needs the
#                            license-restricted CIBERSORT.R source).
#        N_CORES (default 4); RAT_EXCLUDE_GENES; EXCLUDE_SEX_CHROMOSOMES + RAT_SEX_CHROM_GENES;
#        PROTEIN_CODING_ONLY + RAT_PROTEIN_CODING_GENES; PROJECT_ROOT.
# Invoke via run_omnideconv.sh on a compute node (DWLS signature building is CPU-heavy).

suppressWarnings(suppressPackageStartupMessages({
  .libPaths(c(Sys.getenv("R_LIBS_USER"), .libPaths()))
  library(Matrix); library(omnideconv)
}))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 3) stop("usage: run_omnideconv.R <ref_dir> <mixture_dir> <out_dir> [mtx_basename]")
ref_dir <- args[1]; mix_dir <- args[2]; out_dir <- args[3]
mtx_base   <- if (length(args) >= 4) args[4] else "pseudobulk_counts"
genes_base <- sub("_counts$", "", mtx_base)
n.cores  <- as.integer(Sys.getenv("N_CORES", unset = "4"))
methods  <- strsplit(tolower(Sys.getenv("OMNIDECONV_METHODS", unset = "music,dwls,scdc,bisque")),
                     "[, ]+")[[1]]
methods  <- methods[nzchar(methods)]
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
cat(sprintf("omnideconv %s | n.cores=%d | methods: %s\nref=%s\nmix=%s\n",
            as.character(packageVersion("omnideconv")), n.cores,
            paste(methods, collapse = ", "), ref_dir, mix_dir))

# ---- reference (cells x genes) ----
ref <- as.matrix(readMM(file.path(ref_dir, "reference_counts.mtx")))
genes <- readLines(file.path(ref_dir, "genes.tsv"))
meta <- read.delim(file.path(ref_dir, "cells_meta.tsv"), stringsAsFactors = FALSE)
colnames(ref) <- genes
rownames(ref) <- make.unique(as.character(meta$barcode))
cell_type <- as.character(meta$cell_type)
batch_ids <- as.character(if (!is.null(meta$sample)) meta$sample else meta$barcode)
cat(sprintf("reference: %d cells x %d genes | %d types | %d batches\n",
            nrow(ref), ncol(ref), length(unique(cell_type)), length(unique(batch_ids))))

# ---- rat gene cleanup on the reference (same as run_deconvolution.R) ----
excl_path <- Sys.getenv("RAT_EXCLUDE_GENES")
if (excl_path == "")
  excl_path <- file.path(Sys.getenv("PROJECT_ROOT", unset = "."),
                         "deconvolution/reference/rat_exclude_genes.tsv")
if (file.exists(excl_path)) {
  excl <- read.delim(excl_path, stringsAsFactors = FALSE)$feature_ID
  n_excl <- sum(colnames(ref) %in% excl)
  ref <- ref[, !colnames(ref) %in% excl, drop = FALSE]
  cat(sprintf("ribo/mito/hb filter: removed %d genes\n", n_excl))
}
sex_on <- Sys.getenv("EXCLUDE_SEX_CHROMOSOMES", unset = "1") %in% c("1", "true", "TRUE")
sex_path <- Sys.getenv("RAT_SEX_CHROM_GENES")
if (sex_on && nzchar(sex_path) && file.exists(sex_path)) {
  sexg <- toupper(sub("\\..*$", "", read.delim(sex_path, stringsAsFactors = FALSE)$feature_ID))
  drop_sex <- toupper(sub("\\..*$", "", colnames(ref))) %in% sexg
  cat(sprintf("sex-chromosome filter: removed %d chrX/chrY genes\n", sum(drop_sex)))
  ref <- ref[, !drop_sex, drop = FALSE]
}
pc_only <- Sys.getenv("PROTEIN_CODING_ONLY", unset = "1") %in% c("1", "true", "TRUE")
pc_path <- Sys.getenv("RAT_PROTEIN_CODING_GENES")
if (pc_only && nzchar(pc_path) && file.exists(pc_path)) {
  pc <- toupper(sub("\\..*$", "", read.delim(pc_path, stringsAsFactors = FALSE)$feature_ID))
  keep_pc <- toupper(sub("\\..*$", "", colnames(ref))) %in% pc
  cat(sprintf("protein-coding filter: kept %d/%d genes\n", sum(keep_pc), length(keep_pc)))
  ref <- ref[, keep_pc, drop = FALSE]
}
keepg <- colSums(ref > 0) >= 3
cat(sprintf("low-expression filter: dropped %d genes (<3 cells); %d genes kept\n",
            sum(!keepg), sum(keepg)))
ref <- ref[, keepg, drop = FALSE]

# ---- orient for omnideconv: single cell = genes x cells, bulk = genes x samples ----
sc <- t(ref)                                  # genes x cells
mix <- as.matrix(readMM(file.path(mix_dir, paste0(mtx_base, ".mtx"))))
mgenes <- readLines(file.path(mix_dir, paste0(genes_base, "_genes.tsv")))
colnames(mix) <- mgenes
rownames(mix) <- paste0("mix", seq_len(nrow(mix)))   # match run_deconvolution.R sample labels
bulk <- t(mix)                                # genes x samples
sample_ids <- colnames(bulk)
cat(sprintf("mixture: %d samples x %d genes | shared genes ref<->bulk: %d\n",
            ncol(bulk), nrow(bulk), length(intersect(rownames(sc), rownames(bulk)))))

# Normalize any method's output to rows = samples, cols = cell types.
orient <- function(theta) {
  theta <- as.matrix(theta)
  if (all(sample_ids %in% rownames(theta))) return(theta[sample_ids, , drop = FALSE])
  if (all(sample_ids %in% colnames(theta))) return(t(theta)[sample_ids, , drop = FALSE])
  if (nrow(theta) == length(sample_ids)) return(theta)
  if (ncol(theta) == length(sample_ids)) return(t(theta))
  theta
}

# ---- run each method tolerantly ----
# DWLS: omnideconv's one-shot deconvolute(method="dwls") does NOT forward ncores or
# dwls_method into the signature build, so it silently runs the slowest path (plain
# MAST) on a SINGLE core regardless of N_CORES. Build the DWLS signature explicitly so
# the MAST DE is parallelized across N_CORES, then deconvolute against it. dwls_method
# defaults to "mast" (identical to the unforwarded default) but is overridable via
# DWLS_METHOD (e.g. "mast_optimized"); verbose=TRUE so the long MAST build logs progress.
dwls_method <- Sys.getenv("DWLS_METHOD", unset = "mast")

# ---- CIBERSORTx (license-gated): credentials + Apptainer container (Docker is unavailable on
# this cluster). container_path = directory holding fractions_latest.sif (omnideconv pulls it via
# `apptainer pull docker://cibersortx/fractions` on first use if absent). Credentials come from env
# so no token lands in code/logs. No batch correction (S/B-mode off) for the apples-to-apples panel.
# NB: the CIBERSORTx container validates the token against the Stanford server at RUNTIME -> the
# host must have outbound internet when this method runs. ----
cx_sif_dir <- Sys.getenv("CIBERSORTX_SIF_DIR",
  unset = file.path(Sys.getenv("PROJECT_ROOT", unset = "."), "data/deconvolution/cibersortx"))
if ("cibersortx" %in% methods) {
  cx_email <- Sys.getenv("CIBERSORTX_EMAIL"); cx_token <- Sys.getenv("CIBERSORTX_TOKEN")
  if (!nzchar(cx_email) || !nzchar(cx_token))
    stop("cibersortx requested but CIBERSORTX_EMAIL / CIBERSORTX_TOKEN are not set")
  omnideconv::set_cibersortx_credentials(cx_email, cx_token)
  cat(sprintf("cibersortx: apptainer container_path=%s user=%s\n", cx_sif_dir, cx_email))

  # PATCH omnideconv 0.1.1's hardcoded apptainer invocation. Its `apptainer exec --no-home -c`
  # gives the container no writable HOME, so the CIBERSORTx binary's internal R subprocess (used
  # for the nu-SVR signature-matrix build) can't initialise -> "no package called e1071" -> empty
  # matrix -> Eigen assertion abort (code 134). Replace with `--cleanenv` (drop host env + the
  # host XALT LD_PRELOAD that otherwise breaks the container's glibc) + a writable `--home` so the
  # container's R/MCR run in their intended environment. Keeps omnideconv's input-prep + parsing.
  cx_home <- file.path(tempdir(), "cibersortx_home")   # per-R-session -> safe under parallel array runs
  dir.create(cx_home, showWarnings = FALSE, recursive = TRUE)
  Sys.setenv(CX_HOME = cx_home)
  .cx_cmd <- function(in_dir, out_dir, container, apptainer_container_path,
                      method = c("create_sig", "impute_cell_fractions"), verbose = FALSE, ...) {
    method <- match.arg(method)
    base <- paste0("apptainer exec --cleanenv --home ", Sys.getenv("CX_HOME"),
                   " -B ", in_dir, "/:/src/data -B ", out_dir, "/:/src/outdir ",
                   apptainer_container_path, " /src/CIBERSORTxFractions --single_cell TRUE")
    if (verbose) base <- paste(base, "--verbose TRUE")
    check_credentials()
    credentials <- paste("--username", get("cibersortx_email", envir = config_env),
                         "--token", get("cibersortx_token", envir = config_env))
    paste(base, credentials, get_method_options(method, ...))
  }
  environment(.cx_cmd) <- asNamespace("omnideconv")
  assignInNamespace("create_container_command", .cx_cmd, ns = "omnideconv")
  cat("cibersortx: patched container command -> --cleanenv --home (writable)\n")
}

ok <- character(0)
for (m in methods) {
  cat(sprintf("\n===== %s =====\n", m))
  t0 <- Sys.time()
  theta <- tryCatch(
    if (m == "dwls") {
      sig <- omnideconv::build_model(
        single_cell_object = sc, cell_type_annotations = cell_type,
        method = "dwls", dwls_method = dwls_method, ncores = n.cores, verbose = TRUE)
      orient(omnideconv::deconvolute(
        bulk_gene_expression = bulk, model = sig, method = "dwls", verbose = TRUE))
    } else if (m == "cibersortx") {
      # explicit two-step so BOTH the signature build (create_sig) and the impute run are
      # verbose -> the Apptainer/container stderr is captured (the one-shot path silences the
      # build). NB: verbose prints the apptainer command incl. --token to the log (gitignored).
      # verbose=FALSE in production so the --token never lands in logs (omnideconv only message()s
      # the apptainer command when verbose). Re-run a single tissue with verbose=TRUE to debug.
      cx_model <- omnideconv::build_model(
        single_cell_object = sc, cell_type_annotations = cell_type,
        method = "cibersortx", container = "apptainer", container_path = cx_sif_dir,
        verbose = FALSE)
      orient(omnideconv::deconvolute(
        bulk_gene_expression = bulk, model = cx_model, method = "cibersortx",
        container = "apptainer", container_path = cx_sif_dir,
        rmbatch_B_mode = FALSE, rmbatch_S_mode = FALSE, verbose = FALSE))
    } else {
      orient(omnideconv::deconvolute(
        bulk_gene_expression = bulk, model = NULL, method = m,
        single_cell_object = sc, cell_type_annotations = cell_type,
        batch_ids = batch_ids, verbose = FALSE))
    },
    error = function(e) { cat(sprintf("!! %s FAILED: %s\n", m, conditionMessage(e))); NULL })
  if (is.null(theta)) next
  cat(sprintf("%s wall time: %.2f min\n", m,
              as.numeric(difftime(Sys.time(), t0, units = "mins"))))
  out_csv <- file.path(out_dir, sprintf("fractions_%s.csv", m))
  write.csv(theta, out_csv)
  cat(sprintf("mean fraction (%s):\n", m))
  print(round(sort(colMeans(theta), decreasing = TRUE), 4))
  cat(sprintf("wrote %s\n", out_csv))
  ok <- c(ok, m)
}

cat(sprintf("\n=== omnideconv done: %d/%d methods succeeded (%s) ===\n",
            length(ok), length(methods), paste(ok, collapse = ", ")))
if (length(ok) == 0) stop("all omnideconv methods failed")
