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
#        REF_MAX_CELLS_PER_TYPE (default 2000; 0 = every cell): seeded cap on the signature reference.
#        DWLS_METHOD (default "mast"), DWLS_PVAL_CUTOFF (default 0.01), DWLS_DIFF_CUTOFF (default 0.5): the DWLS
#          signature build. The defaults are what every run before 2026-11-14 used without setting them; the
#          benchmark plan's setting is DWLS_METHOD=mast_optimized DWLS_PVAL_CUTOFF=0.05 (audit finding 5.6).
# Output: <out_dir>/fractions_<method>.csv per method that ran, and a provenance record
#   <out_dir>/provenance_run_omnideconv_<methods>.json (audit finding 5.6): commit and modified tracked files,
#   inputs with md5, the methods and their settings, package versions, the cell cap and its seed, the in-memory
#   patches, and what each method did. No credential is written to it.
# Invoke via run_omnideconv.sh on a compute node (DWLS signature building is CPU-heavy).

suppressWarnings(suppressPackageStartupMessages({
  .libPaths(c(Sys.getenv("R_LIBS_USER"), .libPaths()))
  library(Matrix); library(omnideconv)
}))
.script_dir <- local({
  a <- commandArgs(trailingOnly = FALSE)
  f <- sub("^--file=", "", a[grep("^--file=", a)])
  if (length(f)) dirname(normalizePath(f[1])) else "."
})
source(file.path(.script_dir, "provenance.R"))
source(file.path(.script_dir, "omnideconv_settings.R"))

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
n_excl <- NA_integer_
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

# ---- reference size cap (per cell type) ----
# The cross-dataset references are full SC datasets (up to ~232k cells); at that size the
# signature-building methods (esp. DWLS/MAST) blow memory and gain nothing -- a signature is a
# stable per-type estimate, not improved by 100k+ cells. Downsample each cell type to
# REF_MAX_CELLS_PER_TYPE (seeded -> reproducible; logged exactly what is dropped), which also
# BALANCES the reference (rare types keep all cells; dominant types are capped) for better signatures.
# Set REF_MAX_CELLS_PER_TYPE=0 to use every cell (the prior behaviour).
ref_max <- suppressWarnings(as.integer(Sys.getenv("REF_MAX_CELLS_PER_TYPE", unset = "2000")))
CAP_SEED <- 1L
n_ref_cells_before_cap <- nrow(ref)
cap_dropped <- list()                         # cell type -> cells dropped by the cap (for the provenance record)
if (!is.na(ref_max) && ref_max > 0L && nrow(ref) > 0L) {
  set.seed(CAP_SEED)
  idx_by_type <- split(seq_len(nrow(ref)), cell_type)
  keep <- sort(unlist(lapply(idx_by_type, function(ix)
    if (length(ix) > ref_max) sample(ix, ref_max) else ix), use.names = FALSE))
  if (length(keep) < nrow(ref)) {
    dropped <- vapply(idx_by_type, function(ix) max(0L, length(ix) - ref_max), integer(1))
    cap_dropped <- as.list(dropped[dropped > 0])
    cat(sprintf("reference cap: %d -> %d cells (<=%d/type); downsampled: %s\n",
                nrow(ref), length(keep), ref_max,
                paste(sprintf("%s(-%d)", names(dropped)[dropped > 0], dropped[dropped > 0]),
                      collapse = ", ")))
    ref <- ref[keep, , drop = FALSE]; cell_type <- cell_type[keep]; batch_ids <- batch_ids[keep]
  } else {
    cat(sprintf("reference cap: all %d cell types already <= %d cells; no downsampling\n",
                length(idx_by_type), ref_max))
  }
}

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
# The p-value and fold-change cut-offs of the signature build are passed explicitly as well
# (DWLS_PVAL_CUTOFF, DWLS_DIFF_CUTOFF); unset, they are the values omnideconv used when the
# runner did not pass them (0.01 and 0.5). See omnideconv_settings.R.
dwls <- parse_dwls_settings()
dwls_method <- dwls$dwls_method
if ("dwls" %in% methods)
  cat(sprintf("dwls settings: dwls_method=%s (%s) pval_cutoff=%s (%s) diff_cutoff=%s (%s) submethod=DampenedWLS ncores=%d\n",
              dwls$dwls_method, dwls$dwls_method_set_by, format(dwls$pval_cutoff), dwls$pval_cutoff_set_by,
              format(dwls$diff_cutoff), dwls$diff_cutoff_set_by, n.cores))

# ---- CIBERSORTx (license-gated): credentials + Apptainer container (Docker is unavailable on
# this cluster). container_path = directory holding fractions_latest.sif (omnideconv pulls it via
# `apptainer pull docker://cibersortx/fractions` on first use if absent). Credentials come from env
# so no token lands in code/logs. No batch correction (S/B-mode off) for the apples-to-apples panel.
# NB: the CIBERSORTx container validates the token against the Stanford server at RUNTIME -> the
# host must have outbound internet when this method runs. ----
cx_sif_dir <- Sys.getenv("CIBERSORTX_SIF_DIR",
  unset = file.path(Sys.getenv("PROJECT_ROOT", unset = "."), "data/deconvolution/cibersortx"))
cx_patched <- FALSE
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
  cx_patched <- TRUE
  cat("cibersortx: patched container command -> --cleanenv --home (writable)\n")
}

# ---- DWLS robustness patch. DWLS::DEAnalysisMAST / DEAnalysisMASTOptimized initialise the per-cell-type result
# list to a scalar `0` and only overwrite an entry when a cell type has >=2 markers above the (hardcoded) 0.5
# log2FC cutoff. For a transcriptionally-similar type (e.g. an ISG-expressing T subset sitting next to other T
# cells) <2 genes clear the cutoff, so the entry stays `0`; the signature builder then indexes that scalar
# (buildSignatureMatrixMAST: p.adjust(cluster_lrTest.table[[3]]) -> "subscript out of bounds";
# buildSignatureMatrixMASTOptimized: cluster_lrTest.table$p_value -> "$ operator is invalid for atomic vectors")
# and the ENTIRE DWLS run aborts (observed on PBMC/HIP/LNG references). The downstream signature assembly
# already guards on numberofGenes>0 and still emits a signature COLUMN for every cell type (mean expression over
# the selected genes), so the only missing guard is at that first use. We splice it in for BOTH builders (the
# "mast" and the "mast_optimized" path): a degenerate per-type table is counted as 0 marker genes -- the type
# contributes no markers of its own but remains a deconvolvable signature column. Patched at runtime (same
# assignInNamespace mechanism as CIBERSORTx above) so the installed DWLS in R_libs/ is left untouched. Each
# firing of the guard is printed and recorded in the provenance file. ----
dwls_patch <- list()
if ("dwls" %in% methods && requireNamespace("DWLS", quietly = TRUE)) {
  dwls_patch <- patch_dwls_builders()
  for (fn in names(dwls_patch)) {
    if (isTRUE(dwls_patch[[fn]]$patched)) {
      cat(sprintf("dwls: patched %s -> tolerate cell types with <2 distinct markers\n", fn))
    } else {
      cat(sprintf("dwls: WARNING could not patch %s (%s candidate sites)\n", fn,
                  format(dwls_patch[[fn]]$n_sites)))
    }
  }
}

ok <- character(0)
method_record <- list()                       # per method: status, minutes, error message (provenance)
for (m in methods) {
  cat(sprintf("\n===== %s =====\n", m))
  t0 <- Sys.time()
  err_msg <- NULL
  theta <- tryCatch(
    withCallingHandlers(
    if (m == "dwls") {
      sig <- omnideconv::build_model(
        single_cell_object = sc, cell_type_annotations = cell_type,
        method = "dwls", dwls_method = dwls_method, pval_cutoff = dwls$pval_cutoff,
        diff_cutoff = dwls$diff_cutoff, ncores = n.cores, verbose = TRUE)
      cat(sprintf("dwls signature: %d genes x %d cell types\n", nrow(sig), ncol(sig)))
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
    # METHOD_TRACEBACK=1 dumps the call stack at the point of failure (withCallingHandlers fires
    # before the stack unwinds; the outer tryCatch then swallows the error). Diagnostic only.
    error = function(e) {
      if (nzchar(Sys.getenv("METHOD_TRACEBACK"))) {
        cat("---- call stack at error ----\n")
        cs <- sys.calls(); for (i in seq_along(cs))
          cat(sprintf("%2d: %s\n", i, substr(paste(deparse(cs[[i]]), collapse = " "), 1, 200)))
      }
    }),
    error = function(e) { cat(sprintf("!! %s FAILED: %s\n", m, conditionMessage(e)))
                          err_msg <<- conditionMessage(e); NULL })
  mins <- as.numeric(difftime(Sys.time(), t0, units = "mins"))
  if (is.null(theta)) {
    method_record[[m]] <- list(status = "failed", minutes = round(mins, 2), error = err_msg)
    next
  }
  cat(sprintf("%s wall time: %.2f min\n", m, mins))
  out_csv <- file.path(out_dir, sprintf("fractions_%s.csv", m))
  write.csv(theta, out_csv)
  method_record[[m]] <- list(status = "ok", minutes = round(mins, 2), output = out_csv,
                             n_samples = nrow(theta), n_cell_types = ncol(theta),
                             cell_types_not_estimated = I(setdiff(sort(unique(cell_type)), colnames(theta))))
  cat(sprintf("mean fraction (%s):\n", m))
  print(round(sort(colMeans(theta), decreasing = TRUE), 4))
  cat(sprintf("wrote %s\n", out_csv))
  ok <- c(ok, m)
}

cat(sprintf("\n=== omnideconv done: %d/%d methods succeeded (%s) ===\n",
            length(ok), length(methods), paste(ok, collapse = ", ")))

# ---- provenance (audit finding 5.6). One file per invocation, named by its methods, because the methods of one
# set run in separate jobs that share <out_dir>. Written whether or not a method failed. The CIBERSORTx user and
# token are never written. ----
settings <- list()
for (m in methods) settings[[m]] <- switch(m,
  dwls = c(dwls, list(dwls_submethod = "DampenedWLS", ncores = n.cores, two_step = TRUE,
                      input = "raw counts (reference and mixtures)")),
  cibersortx = list(container = "apptainer", container_dir = cx_sif_dir,
                    container_image = {
                      sif <- list.files(cx_sif_dir, pattern = "\\.sif$", full.names = TRUE)
                      lapply(sif, function(f) list(file = f, bytes = unname(file.info(f)$size),
                                                   md5 = unname(tools::md5sum(f))))
                    },
                    rmbatch_B_mode = FALSE, rmbatch_S_mode = FALSE, two_step = TRUE,
                    container_command_patched = cx_patched, input = "raw counts (reference and mixtures)"),
  list(call = "omnideconv::deconvolute, package defaults", batch_ids = "cells_meta$sample",
       n_batches = length(unique(batch_ids)), input = "raw counts (reference and mixtures)"))
runner_files <- file.path(.script_dir, c("run_omnideconv.R", "omnideconv_settings.R", "provenance.R"))
write_provenance(out_dir, md5 = TRUE,
  name = sprintf("provenance_run_omnideconv_%s.json", paste(methods, collapse = "+")),
  inputs = c(file.path(ref_dir, c("reference_counts.mtx", "cells_meta.tsv", "genes.tsv")),
             file.path(mix_dir, c(paste0(mtx_base, ".mtx"), paste0(genes_base, "_genes.tsv"))),
             excl_path, if (sex_on) sex_path, if (pc_only) pc_path),
  params = list(
    ref_dir = ref_dir, mix_dir = mix_dir, mtx_base = mtx_base,
    methods = I(methods), method_settings = settings, method_result = method_record,
    n_cores = n.cores,
    packages = package_versions(c("omnideconv", "DWLS", "MAST", "MuSiC", "SCDC", "BisqueRNA", "Matrix",
                                  "Seurat", "SingleCellExperiment", "Biobase", "e1071", "quadprog", "jsonlite")),
    reference_cap = list(max_cells_per_type = ref_max, seed = CAP_SEED, sampler = "set.seed(seed); sample() per cell type",
                         n_cells_before = n_ref_cells_before_cap, n_cells_after = nrow(ref),
                         cells_dropped_per_type = cap_dropped),
    gene_cleanup = list(exclude_list = excl_path,
                        exclude_list_md5 = if (file.exists(excl_path)) unname(tools::md5sum(excl_path)) else NA,
                        n_excluded_ribo_mito_hb = n_excl, exclude_sex_chromosomes = sex_on,
                        protein_coding_only = pc_only, min_cells_expressing = 3,
                        n_ref_genes = ncol(ref)),
    n_ref_cell_types = length(unique(cell_type)), n_mixtures = ncol(bulk),
    n_shared_genes = length(intersect(rownames(sc), rownames(bulk))),
    in_memory_patches = list(
      dwls = list(applied = dwls_patch,
                  meaning = paste("a cell type with fewer than 2 genes above DWLS's fold-change cut-off is counted as",
                                  "having 0 marker genes instead of stopping the signature build; it stays a signature column"),
                  guard_fired_for = I(as.character(getOption("dwls_guard_fired", character(0))))),
      cibersortx_container_command = list(applied = cx_patched,
                  meaning = "apptainer exec --cleanenv --home <writable> instead of omnideconv's --no-home -c")),
    runner_md5 = as.list(setNames(unname(tools::md5sum(runner_files)), basename(runner_files))),
    pipeline_profile = Sys.getenv("PIPELINE_PROFILE")))
if (length(ok) == 0) stop("all omnideconv methods failed")
