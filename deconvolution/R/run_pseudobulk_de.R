#!/usr/bin/env Rscript
# run_pseudobulk_de.R -- differential expression per (tissue x cell type) block on BayesPrism's deconvolved
# per-cell-type expression Z, across the MoTrPAC endurance-training design. It adapts the MoTrPAC / Vetr 2024
# (Nat Commun 15:3346) recipe to continuous Z with limma-trend.
#
# "STEP3 §n" below cites docs/audit/remediation/STEP3_PREREG.md (registered 2026-11-14), which fixed the
# single-sex rule, the IHW covariate type, the sex-consistency calls, the fraction tier, the technical-covariate
# reruns and the sample-key check before this code produced a number.
#
# A block is analysed when its median per-sample library (the sum of Z over genes) is at least
# MIN_MEDIAN_LIBSIZE (1,000; REFERENCE_RELABEL_PREREG.md §6; env DE_MIN_MEDIAN_LIBSIZE sets the registered
# sensitivities 100 and 0). A block under it stops with status "SKIP: degenerate (...)", keeps its
# median_libsize and frac_zero in de_summary.tsv, and stays out of the pooled IHW and repfdr fits.
#
# Per analysed block:
#   Genes    Every gene with Z > 0 in at least one sample is tested. All-zero genes are dropped and counted.
#   Models   Four limma-trend fits (eBayes trend = TRUE, robust = TRUE) on log2-CPM (prior count 1, no TMM):
#     combined     ~ sex + factor(week)       (~ factor(week) when the block has one sex)
#                  F_dose_comb / P_dose_comb (omnibus F over the week coefficients), lfc_1w..lfc_8w (log2 fold
#                  change of each time point against control), P_8w_comb, logFC_sex / P_sex
#     ordinal      ~ sex + week, week numeric 0, 1, 2, 4, 8       slope_week / P_week_lin
#     interaction  ~ sex * factor(week), two-sex blocks only      F_sexXweek / P_sexXweek
#     per sex      ~ factor(week) within each sex that is present
#                  P_train_M / P_train_F (omnibus F over the week coefficients); the 8-week contrast as
#                  lfc_8w_*, signed z_8w_* = qnorm(p/2, lower.tail = FALSE) * sign(log2FC) (Vetr's z), and its
#                  BH FDR within the block, fdr_8w_*
#   Dose p   P_fisher is Fisher's combination of the per-sex omnibus p over the sexes present (STEP3 §6.1).
#            With two sexes it is the chi-square on 4 df of -2 (ln p_M + ln p_F). With one sex it is that sex's
#            omnibus p, because -2 ln p on 2 df returns p. Before STEP3 a one-sex block had no dose p, so
#            TESTES and OVARY could never have a call.
#   Fraction The cell type's estimated fraction is regressed on week (and sex when the block has two) to flag
#            a composition trend: frac_week_slope, frac_week_p. mean_fraction below 0.01 labels the block
#            low_fraction (STEP3 §6.3, a tier and not a filter).
#
# After all blocks (pooled over every analysed block of every tissue):
#   IHW      ihw(P_fisher, factor(tissue), alpha, covariate_type = "nominal") gives FDR_IHW (STEP3 §6.1).
#            Tissue has no order, so the weights are not smoothed across alphabetically adjacent tissues.
#            A gene-block is dose-significant when FDR_IHW < alpha. FDR_BH_block is BH within the block.
#   Sex-consistency at 8 weeks (STEP3 §6.2), two-sex blocks only:
#     primary    concordance: both per-sex 8-week BH tests pass within the block and the signs agree.
#                sexcons_8w (up_both / down_both / opposite / M_only / F_only / null) and the summary's
#                up / down / sex-specific / opposite counts are computed from this call alone.
#     secondary  repfdr (replication alternative, tail-area Fdr < alpha), fitted as the consortium fits it:
#                on dose-significant gene-blocks only. The Fdr is stored as repfdr_Fdr.
#     Three logical columns per gene (concordance, repfdr, union) and three counts per block are written.
#     A one-sex block has no sex-consistency call: its state is "single_sex" and its counts are missing.
#
# Technical covariates (STEP3 §6.4): env DE_TECH_COVARIATES names any of rin, pct_globin, pct_umi_dup,
# median_5_3_bias. They are read from MotrpacRatTraining6moData's TRNSCRPT_META and added to every model above,
# the fraction regression included. A sample with a missing covariate is dropped for that run and counted.
# The headline run uses none; the two registered reruns are A = rin, pct_globin and B = all four.
#
# Sample key (STEP3 §2.5): Z row i is sample i of bulk_samples.tsv. Where Stage 8 wrote mixture_samples.tsv
# beside the fractions, its order must equal bulk_samples.tsv; the fraction table and every Z matrix must
# have one row per sample. A mismatch stops the run.
#
# Deliberate differences from Vetr 2024:
#   * limma-trend on log2-CPM of continuous Z, not DESeq2 on counts. Z is BayesPrism's posterior expected
#     count mass and is not integer: in rare cell types nearly every value is below 1, so rounding for a
#     negative-binomial model would remove most of the signal (checked on HEART, BLOOD and LUNG).
#   * repfdr uses two columns (8-week male, 8-week female) pooled over blocks, not the consortium's eight
#     columns per tissue. ztobins has warned of a poor density fit in every run so far; its warnings are
#     written to de_methods.tsv, and a fit that returns Fdr = 1 everywhere is recorded as degenerate.
#   * Effect sizes are expected to be small (Vetr: 56% of bulk fold changes within 0.67-1.5x).
#
# Args:  run_pseudobulk_de.R <results_root> <bulk_root> <pheno_tsv> <hotspots_tsv> <out_dir> [TISSUE ...]
#   (see deconvolution/R/run_pseudobulk_de.sh). Reads pred_z already on disk; no deconvolution is rerun.
# Env:   DE_ALPHA, DE_SEED, DE_MIN_MEDIAN_LIBSIZE, DE_TECH_COVARIATES, MOTRPAC_TRNSCRPT_META.
suppressWarnings(suppressPackageStartupMessages({
  .libPaths(c(Sys.getenv("R_LIBS_USER"), .libPaths()))
  library(limma); library(edgeR); library(IHW); library(repfdr)
}))

# Shared cell-type -> filename contract (safe / assert_ct_injective / purge_stale).
.script_dir <- local({
  a <- commandArgs(trailingOnly = FALSE)
  f <- sub("^--file=", "", a[grep("^--file=", a)])
  if (length(f)) dirname(normalizePath(f[1])) else "."
})
source(file.path(.script_dir, "celltype_names.R"))
source(file.path(.script_dir, "provenance.R"))
source(file.path(.script_dir, "config.R"))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 5)
  stop("usage: run_pseudobulk_de.R <results_root> <bulk_root> <pheno_tsv> <hotspots_tsv> <out_dir> [TISSUE ...]")
results_root <- args[1]; bulk_root <- args[2]; pheno_tsv <- args[3]
hotspots_tsv <- args[4]; out_dir <- args[5]
tissue_filter <- if (length(args) >= 6) toupper(args[6:length(args)]) else NULL

WEEK_NUM  <- c("control" = 0, "1w" = 1, "2w" = 2, "4w" = 4, "8w" = 8)
WEEK_LV   <- c("control", "1w", "2w", "4w", "8w")
TP        <- c("1w", "2w", "4w", "8w")
# Significance level of IHW and of every "significant" count. It comes from env DE_ALPHA, which
# pipeline/run_stage10.py --alpha sets, so the R step and the positive-control scoring use one level
# (STEP3 §6.1, finding 7.10). Unset, it is 0.05.
ALPHA <- Sys.getenv("DE_ALPHA")
ALPHA <- if (nzchar(ALPHA)) suppressWarnings(as.numeric(ALPHA)) else 0.05
if (is.na(ALPHA) || ALPHA <= 0 || ALPHA >= 1) stop("DE_ALPHA is not a number between 0 and 1")
MIN_SAMPLES_BLOCK   <- 12   # combined fit needs sex + 5 week levels estimable + residual df
MIN_SAMPLES_PER_SEX <- 8    # per-sex factor(week) (up to 5 levels) needs residual df
MIN_GENES           <- 50   # eBayes mean-variance trend needs a reasonable gene set
MIN_MEDIAN_LIBSIZE  <- as.numeric(Sys.getenv("DE_MIN_MEDIAN_LIBSIZE", "1000"))   # §6 analysable-block rule
if (is.na(MIN_MEDIAN_LIBSIZE)) stop("DE_MIN_MEDIAN_LIBSIZE is not a number")
# STEP3 §6.3: a block whose mean estimated fraction is below this is labelled low-fraction. It is a label for
# reporting the headline three ways, not a filter, so it has no override.
LOW_FRACTION <- 0.01
# repfdr is not fitted on fewer rows than this. ztobins bins each column into 120 bins and fits a 7-df
# spline to the bin counts; below about 1,000 rows most bins hold fewer than ten z-scores.
MIN_REPFDR_ROWS <- 1000
# Seed for IHW's random fold assignment: env DE_SEED (run_pseudobulk_de.sh / run_stage10.py --seed), else the
# config's deconvolution.de_seed, default 1. IHW's own default is seed = 1L.
DE_SEED <- Sys.getenv("DE_SEED")
DE_SEED <- as.integer(if (nzchar(DE_SEED)) DE_SEED else cfg_value("CFG_DE_SEED", default = "1"))
if (is.na(DE_SEED)) stop("DE_SEED is not an integer")
cat(sprintf("alpha: %g\n", ALPHA))
cat(sprintf("IHW seed: %d\n", DE_SEED))
cat(sprintf("analysable blocks: median per-sample library >= %g\n", MIN_MEDIAN_LIBSIZE))

# ---- technical covariates (STEP3 §6.4) ----
# Names accepted in DE_TECH_COVARIATES and the TRNSCRPT_META column each one reads. The order here is the
# order of the columns in every design, whatever order the variable lists them in.
TECH_COV_COLUMNS <- c(rin = "RIN", pct_globin = "pct_globin", pct_umi_dup = "pct_umi_dup",
                      median_5_3_bias = "median_5_3_bias")
TECH_COV <- trimws(strsplit(Sys.getenv("DE_TECH_COVARIATES"), "[,[:space:]]+")[[1]])
TECH_COV <- unique(TECH_COV[nzchar(TECH_COV)])
if (length(setdiff(TECH_COV, names(TECH_COV_COLUMNS))))
  stop("DE_TECH_COVARIATES names unknown covariates: ",
       paste(setdiff(TECH_COV, names(TECH_COV_COLUMNS)), collapse = ", "),
       " (allowed: ", paste(names(TECH_COV_COLUMNS), collapse = ", "), ")")
TECH_COV <- names(TECH_COV_COLUMNS)[names(TECH_COV_COLUMNS) %in% TECH_COV]
TECH <- NULL; META_RDA <- NA_character_
if (length(TECH_COV)) {
  # The vendored MotrpacRatTraining6moData submodule (under the repository root) holds the table; env
  # MOTRPAC_TRNSCRPT_META overrides the path.
  META_RDA <- Sys.getenv("MOTRPAC_TRNSCRPT_META",
                         file.path(cfg_root(), "vendor/MotrpacRatTraining6moData/data/TRNSCRPT_META.rda"))
  if (!file.exists(META_RDA))
    stop("TRNSCRPT_META.rda not found: ", META_RDA,
         "\n  -> git submodule update --init vendor/MotrpacRatTraining6moData")
  .e <- new.env(); load(META_RDA, envir = .e)
  if (!exists("TRNSCRPT_META", envir = .e)) stop("no object TRNSCRPT_META in ", META_RDA)
  .tm <- as.data.frame(get("TRNSCRPT_META", .e))
  if (length(setdiff(c("viallabel", TECH_COV_COLUMNS[TECH_COV]), names(.tm))))
    stop("TRNSCRPT_META lacks columns: ",
         paste(setdiff(c("viallabel", TECH_COV_COLUMNS[TECH_COV]), names(.tm)), collapse = ", "))
  .tm$viallabel <- as.character(.tm$viallabel); .tm <- .tm[!duplicated(.tm$viallabel), ]
  TECH <- do.call(cbind, lapply(TECH_COV, function(k)
    suppressWarnings(as.numeric(.tm[[TECH_COV_COLUMNS[[k]]]]))))
  dimnames(TECH) <- list(.tm$viallabel, paste0("cov_", TECH_COV))
  rm(.e, .tm)
}
cat(sprintf("technical covariates in every model: %s\n",
            if (length(TECH_COV)) paste(TECH_COV, collapse = ", ") else "none"))

# ---- phenotype (all-character so 11-digit viallabels never go scientific) ----
ph <- read.delim(pheno_tsv, stringsAsFactors = FALSE, colClasses = "character")
ph <- ph[!duplicated(ph$viallabel), ]
rownames(ph) <- ph$viallabel

# ---- hotspot set (a label and a row order; it does not enter any test) ----
# The gate table of the refs5 rerun carries a boolean column is_hotspot (STEP3 §5.2: a call that more
# permutations could not reverse). When the column is present it is the flag. Earlier gate tables have no
# such column; for them the flag is q_sup_trained < 0.05, as before.
# The same table's gate_class (hotspot / borderline / none) is copied into de_summary.tsv, so that borderline
# blocks are carried through the DE tables by name (STEP3 §5.2). It is NA when the gate table has no such
# column or does not list the block. The gate AUC is printed for hotspot and borderline blocks.
as_flag <- function(x) toupper(trimws(as.character(x))) %in% c("TRUE", "T", "1")
hot_key <- character(0); hot_auc <- numeric(0); hotspot_rule <- "no gate table read"
gate_cls <- character(0); gate_auc <- numeric(0)
hs <- tryCatch(read.delim(hotspots_tsv, stringsAsFactors = FALSE), error = function(e) NULL)
if (!is.null(hs) && all(c("tissue", "cell_type") %in% names(hs)) &&
    any(c("is_hotspot", "q_sup_trained") %in% names(hs))) {
  if ("is_hotspot" %in% names(hs)) {
    is_h <- as_flag(hs$is_hotspot); hotspot_rule <- "is_hotspot column of the gate table"
  } else {
    q    <- suppressWarnings(as.numeric(hs$q_sup_trained))
    is_h <- !is.na(q) & q < 0.05; hotspot_rule <- "q_sup_trained < 0.05"
  }
  hh <- hs[is_h, , drop = FALSE]
  hot_key <- paste(toupper(hh$tissue), hh$cell_type, sep = "||")
  hot_auc <- if ("sup_trained_auc" %in% names(hh)) suppressWarnings(as.numeric(hh$sup_trained_auc))
             else rep(NA_real_, nrow(hh))
  names(hot_auc) <- hot_key
  all_key <- paste(toupper(hs$tissue), hs$cell_type, sep = "||")
  if ("gate_class" %in% names(hs)) gate_cls <- setNames(trimws(as.character(hs$gate_class)), all_key)
  if ("sup_trained_auc" %in% names(hs)) gate_auc <- setNames(suppressWarnings(as.numeric(hs$sup_trained_auc)), all_key)
}
# gate class and printed AUC of one block (key = "TISSUE||cell type")
block_gate_class <- function(key) if (key %in% names(gate_cls)) unname(gate_cls[key]) else NA_character_
block_gate_auc   <- function(key, is_hot) {
  if (is_hot) return(unname(hot_auc[key]))
  if (block_gate_class(key) %in% "borderline" && key %in% names(gate_auc)) return(unname(gate_auc[key]))
  NA_real_
}
cat(sprintf("hotspot blocks (%s): %d; borderline blocks (gate_class): %d\n", hotspot_rule, length(hot_key),
            sum(gate_cls %in% "borderline")))

# ---- tissues with Z; hotspot tissues first ----
all_tis <- list.dirs(results_root, recursive = FALSE, full.names = FALSE)
tissues <- all_tis[file.exists(file.path(results_root, all_tis, "pred_z", "genes.txt"))]
if (!is.null(tissue_filter)) tissues <- tissues[toupper(tissues) %in% tissue_filter]
tissues <- toupper(tissues)
hot_tis <- unique(sub("\\|\\|.*", "", hot_key))
tissues <- c(intersect(tissues, hot_tis), setdiff(tissues, hot_tis))
tissues <- tissues[tissues %in% toupper(all_tis)]
if (!length(tissues)) stop("no tissues with pred_z under ", results_root)
cat(sprintf("tissues (%d), hotspots first: %s\n", length(tissues), paste(tissues, collapse = ", ")))

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

# helper: fit limma-trend, return eBayes fit (genes x samples logCPM, design)
fit_trend <- function(logcpm, design)
  eBayes(lmFit(logcpm, design), trend = TRUE, robust = TRUE)

# helper: safe matrix-column getter (returns NA vector if the column is absent)
getcolm <- function(mat, cn) if (cn %in% colnames(mat)) mat[, cn] else rep(NA_real_, nrow(mat))

# helper: append the technical covariates (samples x covariates, NULL when the run has none) to a design.
# It stops when a covariate is constant or collinear in these samples, so that no model is silently fitted
# without a covariate the run was asked to include.
add_cov <- function(design, cov) {
  if (is.null(cov)) return(design)
  if (qr(cbind(design, cov))$rank < qr(design)$rank + ncol(cov))
    stop("technical covariates are collinear with the design in these samples")
  cbind(design, cov)
}

# helper: sample-key failures stop the whole run; the per-block handler passes them on (STEP3 §2.5)
sample_key_stop <- function(msg)
  stop(structure(class = c("sample_key_error", "error", "condition"), list(message = msg, call = NULL)))

# helper: per-sex fit -> list(omni F/p over factor(week), per-timepoint logFC+p, n)
fit_one_sex <- function(logcpm, grp, cov = NULL) {
  wf <- droplevels(factor(grp, levels = WEEK_LV))
  if (ncol(logcpm) < MIN_SAMPLES_PER_SEX || nlevels(wf) < 2) return(NULL)
  d  <- add_cov(model.matrix(~ wf), cov)
  if (ncol(logcpm) <= ncol(d)) return(NULL)                       # no residual degree of freedom
  f  <- fit_trend(logcpm, d)
  wc <- grep("^wf", colnames(d), value = TRUE)
  omni <- topTable(f, coef = wc, number = Inf, sort.by = "none")  # moderated F (or t if 1 coef) + P.Value
  lfc  <- f$coefficients[, wc, drop = FALSE]                      # per-timepoint log2FC vs control
  # per-timepoint p (single-coef moderated t); vapply keeps a genes x coefs matrix even for 1 coef
  pmat <- vapply(wc, function(cc) topTable(f, coef = cc, number = Inf, sort.by = "none")$P.Value,
                 numeric(nrow(lfc)))
  if (is.null(dim(pmat))) pmat <- matrix(pmat, ncol = length(wc))
  colnames(pmat) <- wc; rownames(pmat) <- rownames(lfc)
  list(P = omni$P.Value, lfc = lfc, p = pmat,
       levels = sub("^wf", "", wc), n = ncol(logcpm), genes = rownames(f$coefficients))
}

block_tabs  <- list()   # per-block per-gene data.frames (augmented with FDR_IHW + state in pass 2)
block_nsex  <- list()   # number of sexes in each analysed block
summary_rows <- list()
G_p <- c(); G_tis <- c(); G_key <- c()                 # global IHW inputs (dose p, tissue, block::gene)
G_zM <- c(); G_zF <- c(); G_zkey <- c()                # global repfdr inputs (8w signed z)
key_checked <- character(0)                            # tissues whose mixture_samples.tsv was checked
cov_dropped <- integer(0)                              # per tissue: samples dropped for a missing covariate

for (TIS in tissues) {
  pz    <- file.path(results_root, TIS, "pred_z")
  types <- trimws(readLines(file.path(pz, "types.txt"), encoding = "UTF-8"))
  types <- types[types != ""]
  via   <- trimws(readLines(file.path(bulk_root, TIS, "bulk_samples.tsv"))); via <- via[via != ""]
  n     <- length(via)
  meta  <- data.frame(mix = paste0("mix", seq_len(n)), viallabel = via,
                      sex = ph[via, "sex"], group = ph[via, "group"], stringsAsFactors = FALSE)
  meta$week <- unname(WEEK_NUM[meta$group])

  # STEP3 §2.5: the join of Z rows to animals is positional (mix<i> = line i of bulk_samples.tsv). Where
  # Stage 8 wrote the key it used, the two orders must be identical.
  key_file <- file.path(results_root, TIS, "mixture_samples.tsv")
  if (file.exists(key_file)) {
    kf <- read.delim(key_file, stringsAsFactors = FALSE, colClasses = "character")
    if (!all(c("mix", "viallabel") %in% names(kf)))
      stop(key_file, " has no columns mix and viallabel")
    if (!identical(trimws(kf$viallabel), via))
      stop(sprintf("[%s] the viallabel column of %s (%d rows) is not identical, in order, to %s (%d samples)",
                   TIS, key_file, nrow(kf), file.path(bulk_root, TIS, "bulk_samples.tsv"), n))
    if (!identical(trimws(kf$mix), meta$mix))
      stop(sprintf("[%s] the mix column of %s is not mix1..mix%d in order", TIS, key_file, n))
    key_checked <- c(key_checked, TIS)
  }
  frac_file <- file.path(results_root, TIS, "estimated_fractions.csv")
  fracs <- if (file.exists(frac_file)) read.csv(frac_file, row.names = 1, check.names = FALSE) else NULL
  if (!is.null(fracs) && (nrow(fracs) != n || !all(meta$mix %in% rownames(fracs))))
    stop(sprintf("[%s] %s has %d rows for %d samples of bulk_samples.tsv (rows must be mix1..mix%d)",
                 TIS, frac_file, nrow(fracs), n, n))

  # Usable samples: a known training group and sex and, when the run has technical covariates, every
  # covariate present. The samples dropped for a missing covariate are counted (STEP3 §6.4).
  base_ok <- !is.na(meta$week) & !is.na(meta$sex) & meta$group %in% WEEK_LV
  covt    <- if (length(TECH_COV)) TECH[match(via, rownames(TECH)), , drop = FALSE] else NULL
  cov_ok  <- if (length(TECH_COV)) stats::complete.cases(covt) else rep(TRUE, n)
  ok      <- base_ok & cov_ok
  n_cov_drop <- sum(base_ok & !cov_ok)
  cov_dropped[TIS] <- n_cov_drop

  out_tis <- file.path(out_dir, TIS); dir.create(out_tis, showWarnings = FALSE, recursive = TRUE)
  cat(sprintf("\n[%s] %d samples x %d cell types%s\n", TIS, n, length(types),
              if (n_cov_drop) sprintf(" (%d dropped: missing covariate)", n_cov_drop) else ""))

  # Two cell types must never claim the same de__ filename (see celltype_names.R), and a
  # relabelled tissue must not leave orphan blocks behind for a downstream glob to ingest.
  assert_ct_injective(types, TIS)
  purge_stale(out_tis, "^de__.*\\.tsv$", keep = sprintf("de__%s.tsv", safe(types)))

  for (ct in types) {
    csv    <- file.path(pz, paste0("predz__", safe(ct), ".csv"))
    key    <- paste(TIS, ct, sep = "||")
    is_hot <- key %in% hot_key
    res <- tryCatch({
      if (!file.exists(csv)) stop("no predz csv")
      Z  <- as.matrix(read.csv(csv, row.names = 1, check.names = FALSE))   # samples x genes
      if (nrow(Z) != n || !all(meta$mix %in% rownames(Z)))                 # STEP3 §2.5
        sample_key_stop(sprintf("[%s] %s has %d rows for %d samples of bulk_samples.tsv (rows must be mix1..mix%d)",
                                TIS, csv, nrow(Z), n, n))
      Z  <- Z[meta$mix, , drop = FALSE]
      if (sum(ok) < MIN_SAMPLES_BLOCK) stop(sprintf("only %d usable samples", sum(ok)))
      m  <- meta[ok, ]
      covb <- if (length(TECH_COV)) covt[ok, , drop = FALSE] else NULL    # this block's samples x covariates
      counts <- t(Z[ok, , drop = FALSE]); counts[counts < 0 | is.na(counts)] <- 0  # genes x samples

      # FULL COVERAGE: keep every gene non-zero in >=1 sample; log the all-zero drop
      nz       <- rowSums(counts > 0)
      keep_g   <- nz >= 1
      n_drop   <- sum(!keep_g)
      counts   <- counts[keep_g, , drop = FALSE]; nz <- nz[keep_g]
      if (nrow(counts) < MIN_GENES) stop(sprintf("only %d non-zero genes", nrow(counts)))

      libsize   <- colSums(counts)
      mean_frac <- if (!is.null(fracs) && ct %in% colnames(fracs)) mean(fracs[m$mix, ct], na.rm = TRUE) else NA_real_
      block_zero <- mean(counts == 0)
      if (median(libsize) < MIN_MEDIAN_LIBSIZE)
        stop(structure(class = c("degenerate_block", "error", "condition"), list(
          message = sprintf("degenerate (median libsize %.1f < %g)", median(libsize), MIN_MEDIAN_LIBSIZE),
          call = NULL, n_samples = nrow(m), mean_fraction = mean_frac,
          median_libsize = median(libsize), frac_zero = block_zero)))

      dge    <- DGEList(counts = counts)
      logcpm <- edgeR::cpm(dge, log = TRUE, prior.count = 1)
      genes  <- rownames(logcpm)
      sexf   <- factor(m$sex)
      weekf  <- droplevels(factor(m$group, levels = WEEK_LV))
      weekn  <- as.numeric(m$week)
      two_sex <- nlevels(sexf) >= 2

      # ---- combined factor model: omnibus dose (sex-adjusted) + per-timepoint logFC + sex ----
      des_c <- add_cov(if (two_sex) model.matrix(~ sexf + weekf) else model.matrix(~ weekf), covb)
      fit_c <- fit_trend(logcpm, des_c)
      wc    <- grep("^weekf", colnames(des_c), value = TRUE)
      tt_d  <- topTable(fit_c, coef = wc, number = Inf, sort.by = "none")          # F + P (any dose)
      lfc_tp <- fit_c$coefficients[, wc, drop = FALSE]                             # sex-adj per-tp logFC
      colnames(lfc_tp) <- sub("^weekf", "", wc)
      p_8w_comb <- if ("weekf8w" %in% colnames(des_c))
        topTable(fit_c, coef = "weekf8w", number = Inf, sort.by = "none")$P.Value else rep(NA_real_, length(genes))
      scoef <- grep("^sexf", colnames(des_c), value = TRUE)
      if (length(scoef) == 1) {
        tt_s <- topTable(fit_c, coef = scoef, number = Inf, sort.by = "none")
        logFC_sex <- tt_s$logFC; P_sex <- tt_s$P.Value
      } else { logFC_sex <- rep(NA_real_, length(genes)); P_sex <- rep(NA_real_, length(genes)) }

      # ---- ordinal linear dose slope (continuity with prior run) ----
      des_o <- add_cov(if (two_sex) model.matrix(~ sexf + weekn) else model.matrix(~ weekn), covb)
      fit_o <- fit_trend(logcpm, des_o)
      tt_l  <- topTable(fit_o, coef = "weekn", number = Inf, sort.by = "none")

      # ---- sex x dose interaction omnibus ----
      F_int <- rep(NA_real_, length(genes)); P_int <- rep(NA_real_, length(genes))
      if (two_sex && nlevels(weekf) >= 2) {
        des_i <- model.matrix(~ sexf * weekf)
        icoef <- grep(":", colnames(des_i), value = TRUE)
        if (length(icoef) >= 1) {
          fit_i <- fit_trend(logcpm, add_cov(des_i, covb))
          tt_i  <- topTable(fit_i, coef = icoef, number = Inf, sort.by = "none")
          F_int <- tt_i$F; P_int <- tt_i$P.Value
        }
      }

      # ---- per-sex stratified: one fit for every sex that is present (STEP3 §6.1) ----
      # Before STEP3 the per-sex fits ran only in two-sex blocks, so a one-sex tissue had no per-sex
      # statistic at all. A two-sex block is fitted exactly as before.
      per_sex <- function(s) {
        i <- m$sex == s
        if (!any(i)) return(NULL)
        fit_one_sex(logcpm[, i, drop = FALSE], m$group[i], if (is.null(covb)) NULL else covb[i, , drop = FALSE])
      }
      M  <- per_sex("male")
      Fm <- per_sex("female")
      P_train_M <- if (!is.null(M))  M$P  else rep(NA_real_, length(genes))
      P_train_F <- if (!is.null(Fm)) Fm$P else rep(NA_real_, length(genes))
      lfc_8w_M <- if (!is.null(M)  && "8w" %in% M$levels)  M$lfc[, "wf8w"]  else rep(NA_real_, length(genes))
      lfc_8w_F <- if (!is.null(Fm) && "8w" %in% Fm$levels) Fm$lfc[, "wf8w"] else rep(NA_real_, length(genes))
      p_8w_M  <- if (!is.null(M)  && "8w" %in% M$levels)  M$p[, "wf8w"]  else rep(NA_real_, length(genes))
      p_8w_F  <- if (!is.null(Fm) && "8w" %in% Fm$levels) Fm$p[, "wf8w"] else rep(NA_real_, length(genes))
      # signed z at 8w (Vetr); per-sex BH on the 8w contrast, the input of the concordance call
      z_8w_M  <- qnorm(p_8w_M / 2, lower.tail = FALSE) * sign(lfc_8w_M)
      z_8w_F  <- qnorm(p_8w_F / 2, lower.tail = FALSE) * sign(lfc_8w_F)
      fdr_8w_M <- p.adjust(p_8w_M, "BH"); fdr_8w_F <- p.adjust(p_8w_F, "BH")

      # ---- dose p: Fisher's combination over the sexes present (STEP3 §6.1) ----
      # Two sexes: chi-square on 4 df, missing unless both per-sex p are finite (unchanged). One sex:
      # Fisher's statistic -2 ln p on 2 df returns p, so the dose p is that sex's omnibus p.
      P_fisher <- if (two_sex)
        ifelse(is.finite(P_train_M) & is.finite(P_train_F),
               pchisq(-2 * (log(P_train_M) + log(P_train_F)), df = 4, lower.tail = FALSE),
               NA_real_)
      else {
        p1 <- if (any(m$sex == "male")) P_train_M else P_train_F
        ifelse(is.finite(p1), p1, NA_real_)
      }

      # composition confound: cell-type fraction ~ dose, adjusted for sex when the block has two sexes.
      # With one sex the sex term cannot be fitted; before STEP3 the regression failed there and the
      # p was left missing (finding 7.1).
      frac_p <- NA_real_; frac_slope <- NA_real_
      if (!is.null(fracs) && ct %in% colnames(fracs)) {
        fv <- as.numeric(fracs[m$mix, ct])
        ffrm <- if (two_sex) fv ~ weekn + sexf else fv ~ weekn
        if (!is.null(covb)) ffrm <- update(ffrm, . ~ . + covb)
        lf <- tryCatch(summary(lm(ffrm)), error = function(e) NULL)
        if (!is.null(lf) && "weekn" %in% rownames(lf$coefficients)) {
          frac_slope <- lf$coefficients["weekn", "Estimate"]; frac_p <- lf$coefficients["weekn", "Pr(>|t|)"]
        }
      }

      out <- data.frame(
        gene = genes, n_nonzero = nz, AveExpr = tt_d$AveExpr, mean_fraction = mean_frac,
        F_dose_comb = tt_d$F, P_dose_comb = tt_d$P.Value,
        slope_week = tt_l$logFC, P_week_lin = tt_l$P.Value,
        lfc_1w = getcolm(lfc_tp, "1w"), lfc_2w = getcolm(lfc_tp, "2w"),
        lfc_4w = getcolm(lfc_tp, "4w"), lfc_8w = getcolm(lfc_tp, "8w"), P_8w_comb = p_8w_comb,
        F_sexXweek = F_int, P_sexXweek = P_int,
        P_train_M = P_train_M, P_train_F = P_train_F, P_fisher = P_fisher,
        FDR_BH_block = p.adjust(P_fisher, "BH"),
        lfc_8w_M = lfc_8w_M, lfc_8w_F = lfc_8w_F, z_8w_M = z_8w_M, z_8w_F = z_8w_F,
        fdr_8w_M = fdr_8w_M, fdr_8w_F = fdr_8w_F,
        logFC_sex = logFC_sex, P_sex = P_sex, FDR_sex = p.adjust(P_sex, "BH"),
        stringsAsFactors = FALSE, check.names = FALSE)

      # accumulate for global IHW (dose p ~ tissue) and global repfdr (8w z)
      gk <- paste(key, genes, sep = "||")
      fin <- is.finite(P_fisher)
      G_p   <- c(G_p, P_fisher[fin]); G_tis <- c(G_tis, rep(TIS, sum(fin))); G_key <- c(G_key, gk[fin])
      zin <- is.finite(z_8w_M) & is.finite(z_8w_F)
      G_zM <- c(G_zM, z_8w_M[zin]); G_zF <- c(G_zF, z_8w_F[zin]); G_zkey <- c(G_zkey, gk[zin])

      block_tabs[[key]] <- out
      block_nsex[[key]] <- nlevels(sexf)
      list(tissue = TIS, cell_type = ct, is_hotspot = is_hot, gate_class = block_gate_class(key),
           sup_trained_auc = block_gate_auc(key, is_hot),
           n_samples = nrow(m), n_male = sum(m$sex == "male"), n_female = sum(m$sex == "female"),
           n_genes_tested = nrow(out), n_genes_dropped_allzero = n_drop,
           mean_fraction = mean_frac, low_fraction = mean_frac < LOW_FRACTION,
           median_libsize = median(libsize), frac_zero = block_zero,
           n_dose_tests = sum(fin),
           n_sig_dose_fisher_BHblock = sum(out$FDR_BH_block < ALPHA, na.rm = TRUE),
           n_sig_interaction = sum(p.adjust(out$P_sexXweek, "BH") < ALPHA, na.rm = TRUE),
           n_sig_sex = sum(out$FDR_sex < ALPHA, na.rm = TRUE),
           frac_week_slope = frac_slope, frac_week_p = frac_p,
           n_samples_dropped_covariate = n_cov_drop, status = "ok")
    }, error = function(e) {
      if (inherits(e, "sample_key_error")) stop(e)   # a broken sample key stops the run, not the block
      dg <- inherits(e, "degenerate_block")   # §6: keep the numbers that decided the skip
      list(tissue = TIS, cell_type = ct, is_hotspot = is_hot, gate_class = block_gate_class(key),
           sup_trained_auc = block_gate_auc(key, is_hot),
           n_samples = if (dg) e$n_samples else NA_integer_, n_male = NA_integer_, n_female = NA_integer_,
           n_genes_tested = NA_integer_, n_genes_dropped_allzero = NA_integer_,
           mean_fraction = if (dg) e$mean_fraction else NA_real_,
           low_fraction = NA,                  # the tier labels analysed blocks only (STEP3 §6.3)
           median_libsize = if (dg) e$median_libsize else NA_real_,
           frac_zero = if (dg) e$frac_zero else NA_real_,
           n_dose_tests = NA_integer_,
           n_sig_dose_fisher_BHblock = NA_integer_, n_sig_interaction = NA_integer_,
           n_sig_sex = NA_integer_, frac_week_slope = NA_real_, frac_week_p = NA_real_,
           n_samples_dropped_covariate = n_cov_drop,
           status = paste0("SKIP: ", conditionMessage(e)))
    })
    summary_rows[[length(summary_rows) + 1]] <- res
    cat(sprintf("  %-30s hot=%-5s %s\n", substr(ct, 1, 30), is_hot,
                if (res$status == "ok")
                  sprintf("fisher_sig=%d intxn=%d sex=%d n_genes=%d drop0=%d", res$n_sig_dose_fisher_BHblock,
                          res$n_sig_interaction, res$n_sig_sex, res$n_genes_tested, res$n_genes_dropped_allzero)
                else res$status))
  }
}

# ================= PASS 2: global IHW (tissue covariate) =================
# STEP3 §6.1 (finding 7.8): tissue is a nominal covariate. IHW's default, "ordinal", penalises weight
# differences between neighbouring factor levels, which for tissue means alphabetical neighbours.
# "nominal" penalises deviation from uniform weights and uses no order. This changes power, not error control.
FDR_IHW_map <- setNames(rep(NA_real_, length(G_key)), G_key)
ihw_used <- "none"; ihw_res <- NULL
IHW_COVARIATE_TYPE <- "nominal"
if (length(G_p) > 50) {
  ir <- tryCatch({
    set.seed(DE_SEED)
    r <- ihw(G_p, factor(G_tis), alpha = ALPHA, covariate_type = IHW_COVARIATE_TYPE, seed = DE_SEED)
    ihw_used <<- sprintf("IHW~tissue (covariate_type = %s)", IHW_COVARIATE_TYPE); ihw_res <<- r
    adj_pvalues(r)
  }, error = function(e) {
    ihw_used <<- paste0("BH-global (IHW failed: ", conditionMessage(e), ")"); p.adjust(G_p, "BH")
  })
  FDR_IHW_map <- setNames(ir, G_key)
}
cat(sprintf("\nGlobal multiple-testing: %s over %d dose p-values in %d tissues.\n",
            ihw_used, length(G_p), length(unique(G_tis))))

# ================= PASS 2: repfdr at 8 weeks, the secondary sex-consistency call =================
# STEP3 §6.2: repfdr is fitted as the consortium fits it, on training-regulated features only: the
# gene-blocks with FDR_IHW < alpha whose two per-sex 8-week z are finite. Before STEP3 it was fitted on every
# gene-block, the density fit was poor, and it supplied 97% of the sex-consistency calls. It is now reported
# beside the primary call (concordance, below) and does not enter sexcons_8w.
# ztobins' warnings are kept and written to de_methods.tsv. A fit that returns Fdr = 1 for every row is
# recorded as degenerate and its values are discarded.
repfdr_Fdr <- setNames(rep(NA_real_, length(G_zkey)), G_zkey)   # repfdr's tail-area Fdr, where fitted
repfdr_fitted <- FALSE
repfdr_warnings <- character(0)
in_fit <- { f <- unname(FDR_IHW_map[G_zkey]); is.finite(f) & f < ALPHA }
n_repfdr_in <- sum(in_fit)
if (n_repfdr_in >= MIN_REPFDR_ROWS) {
  repfdr_status <- "failed"
  rf <- tryCatch(withCallingHandlers({
    zmat <- cbind(male = G_zM[in_fit], female = G_zF[in_fit])
    bz   <- ztobins(zmat)
    rr   <- repfdr(bz$pdf.binned.z, bz$binned.z.mat, non.null = "replication")
    rr$mat[, "Fdr"]   # tail-area Bayes FDR per feature for the replication alternative
  }, warning = function(w) {
    repfdr_warnings <<- c(repfdr_warnings, gsub("[[:space:]]+", " ", trimws(conditionMessage(w))))
    invokeRestart("muffleWarning")
  }), error = function(e) {
    repfdr_status <<- paste0("not available: repfdr failed (",
                             gsub("[[:space:]]+", " ", trimws(conditionMessage(e))), ")"); NULL
  })
  if (!is.null(rf)) {
    if (all(rf[is.finite(rf)] >= 1 - 1e-9)) {
      repfdr_status <- sprintf("not available: fit degenerate (Fdr = 1 for all %d finite of %d gene-blocks)",
                               sum(is.finite(rf)), length(rf))
    } else {
      repfdr_status <- sprintf(paste0("fitted on %d dose-significant gene-blocks (FDR_IHW < %g, both 8w z finite); ",
                                      "ztobins defaults; non.null = replication; call = tail-area Fdr < %g and equal sign"),
                               n_repfdr_in, ALPHA, ALPHA)
      repfdr_Fdr[G_zkey[in_fit]] <- rf
      repfdr_fitted <- TRUE
    }
  }
} else {
  repfdr_status <- sprintf("not fitted: %d dose-significant gene-blocks with both 8w z finite, fewer than the minimum of %d",
                           n_repfdr_in, MIN_REPFDR_ROWS)
}
sexcons_primary <- sprintf("concordance: both per-sex 8w BH FDR < %g within the block and equal sign", ALPHA)
cat(sprintf("\nSex-consistency, primary: %s.\nSex-consistency, secondary (repfdr): %s.\n",
            sexcons_primary, repfdr_status))
if (length(repfdr_warnings))
  cat("repfdr/ztobins warnings:\n  ", paste(unique(repfdr_warnings), collapse = "\n   "), "\n")

# assign per-gene FDR_IHW and the sex-consistency calls back into the block tables
for (key in names(block_tabs)) {
  df <- block_tabs[[key]]
  gk <- paste(key, df$gene, sep = "||")
  df$FDR_IHW <- unname(FDR_IHW_map[gk])
  df$repfdr_Fdr <- unname(repfdr_Fdr[gk])
  if (block_nsex[[key]] >= 2) {
    sig_M <- is.finite(df$fdr_8w_M) & df$fdr_8w_M < ALPHA
    sig_F <- is.finite(df$fdr_8w_F) & df$fdr_8w_F < ALPHA
    same_up   <- (df$z_8w_M > 0 & df$z_8w_F > 0) %in% TRUE
    same_down <- (df$z_8w_M < 0 & df$z_8w_F < 0) %in% TRUE
    # primary call (STEP3 §6.2): both per-sex tests pass and the signs agree
    conc <- sig_M & sig_F & (same_up | same_down)
    # secondary call: repfdr replication Fdr below alpha and the signs agree. A gene outside the fit has no
    # repfdr call (FALSE); when no fit is available the column is missing.
    repl <- if (repfdr_fitted) is.finite(df$repfdr_Fdr) & df$repfdr_Fdr < ALPHA & (same_up | same_down)
            else rep(NA, nrow(df))
    # state, from the primary call alone
    st <- rep("null", nrow(df))
    st[sig_M & sig_F & same_up]   <- "up_both"
    st[sig_M & sig_F & same_down] <- "down_both"
    st[sig_M & sig_F & !same_up & !same_down] <- "opposite"
    st[sig_M & !sig_F] <- "M_only"
    st[!sig_M & sig_F] <- "F_only"
    df$sexcons_concordance_8w <- conc
    df$sexcons_repfdr_8w      <- repl
    df$sexcons_union_8w       <- conc | repl
    df$sexcons_8w             <- st
  } else {
    # one sex: there is nothing to be consistent with
    df$sexcons_concordance_8w <- NA; df$sexcons_repfdr_8w <- NA; df$sexcons_union_8w <- NA
    df$sexcons_8w <- "single_sex"
  }
  # order: IHW-significant dose genes first, then by dose p
  df <- df[order(ifelse(is.na(df$FDR_IHW), 1, df$FDR_IHW), df$P_fisher), ]
  block_tabs[[key]] <- df
  TIS <- sub("\\|\\|.*", "", key); ct <- sub(".*\\|\\|", "", key)
  write.table(df, file.path(out_dir, TIS, paste0("de__", safe(ct), ".tsv")),
              sep = "\t", quote = FALSE, row.names = FALSE)
}

# augment summary with global-FDR + sex-consistency counts now available
summ <- do.call(rbind, lapply(summary_rows, function(r) {
  na_counts <- c("n_sexcons_concordance_8w", "n_sexcons_repfdr_8w", "n_sexcons_union_8w",
                 "n_up_both_8w", "n_down_both_8w", "n_sexspecific_8w", "n_opposite_8w")
  r$n_sig_dose_IHW <- NA_integer_
  for (cn in na_counts) r[[cn]] <- NA_integer_
  if (r$status == "ok") {
    key <- paste(r$tissue, r$cell_type, sep = "||")
    df  <- block_tabs[[key]]
    r$n_sig_dose_IHW <- sum(df$FDR_IHW < ALPHA, na.rm = TRUE)
    if (block_nsex[[key]] >= 2) {
      r$n_sexcons_concordance_8w <- sum(df$sexcons_concordance_8w)
      if (repfdr_fitted) {
        r$n_sexcons_repfdr_8w <- sum(df$sexcons_repfdr_8w)
        r$n_sexcons_union_8w  <- sum(df$sexcons_union_8w)
      }
      r$n_up_both_8w     <- sum(df$sexcons_8w == "up_both")
      r$n_down_both_8w   <- sum(df$sexcons_8w == "down_both")
      r$n_sexspecific_8w <- sum(df$sexcons_8w %in% c("M_only", "F_only"))
      r$n_opposite_8w    <- sum(df$sexcons_8w == "opposite")
    }
  }
  as.data.frame(r, stringsAsFactors = FALSE)
}))
# column order: descriptors, then counts
ord <- c("tissue","cell_type","is_hotspot","gate_class","sup_trained_auc","n_samples","n_male","n_female",
         "n_genes_tested","n_genes_dropped_allzero","mean_fraction","low_fraction","median_libsize","frac_zero",
         "n_dose_tests","n_sig_dose_IHW","n_sig_dose_fisher_BHblock","n_sig_interaction",
         "n_sexcons_concordance_8w","n_sexcons_repfdr_8w","n_sexcons_union_8w",
         "n_up_both_8w","n_down_both_8w","n_sexspecific_8w","n_opposite_8w","n_sig_sex","frac_week_slope",
         "frac_week_p","n_samples_dropped_covariate","status")
summ <- summ[, ord]
summ <- summ[order(-summ$is_hotspot,
                   -ifelse(is.na(summ$sup_trained_auc), -1, summ$sup_trained_auc),
                   -ifelse(is.na(summ$n_sig_dose_IHW), -1, summ$n_sig_dose_IHW)), ]
write.table(summ, file.path(out_dir, "de_summary.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)
write.table(summ[summ$is_hotspot %in% TRUE, ], file.path(out_dir, "de_hotspots.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE)

# cell_type -> file manifest for the ok blocks, so consumers join on a table instead of
# re-deriving the filename (which is how the collision went unnoticed for three weeks).
ok <- summ[summ$status == "ok", ]
write.table(data.frame(tissue    = ok$tissue,
                       cell_type = ok$cell_type,
                       file      = file.path(ok$tissue, sprintf("de__%s.tsv", safe(ok$cell_type))),
                       stringsAsFactors = FALSE),
            file.path(out_dir, "celltype_files.tsv"), sep = "\t",
            quote = FALSE, row.names = FALSE)

# ---- headline, three ways (STEP3 §6.3) ----
# Dose-significant gene x block pairs, distinct genes, and both again outside low-fraction blocks; over all
# tissues (scope ALL) and per tissue. A block whose fraction is unknown is in neither fraction tier and is
# counted in n_blocks_fraction_unknown.
sig_genes <- lapply(block_tabs, function(df) df$gene[!is.na(df$FDR_IHW) & df$FDR_IHW < ALPHA])
headline_row <- function(scope, rows) {
  keys <- paste(rows$tissue, rows$cell_type, sep = "||")
  hi   <- keys[rows$low_fraction %in% FALSE]
  data.frame(scope = scope, alpha = ALPHA,
             n_blocks_analysed = nrow(rows),
             n_blocks_low_fraction = sum(rows$low_fraction %in% TRUE),
             n_blocks_fraction_unknown = sum(is.na(rows$low_fraction)),
             n_pairs = sum(lengths(sig_genes[keys])),
             n_genes_distinct = length(unique(unlist(sig_genes[keys], use.names = FALSE))),
             n_pairs_not_low_fraction = sum(lengths(sig_genes[hi])),
             n_genes_distinct_not_low_fraction = length(unique(unlist(sig_genes[hi], use.names = FALSE))),
             stringsAsFactors = FALSE)
}
headline <- do.call(rbind, c(list(headline_row("ALL", ok)),
                             lapply(sort(unique(ok$tissue)), function(t) headline_row(t, ok[ok$tissue == t, ]))))
write.table(headline, file.path(out_dir, "de_headline.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)

# ---- methods sidecar: what this run fitted (STEP3 §6.1, findings 7.7 and 7.8) ----
cov_txt  <- if (length(TECH_COV)) paste0(" + ", paste(TECH_COV, collapse = " + ")) else ""
single   <- vapply(names(block_tabs), function(k) block_nsex[[k]] < 2, logical(1))
ver      <- function(p) as.character(utils::packageVersion(p))
# IHW arguments this script does not set are read from the installed function, not typed in.
ihw_defaults <- tryCatch({
  fm <- formals(IHW:::ihw.default)
  fm <- fm[setdiff(names(fm), c("pvalues", "covariates", "alpha", "covariate_type", "seed", "..."))]
  paste(sprintf("%s = %s", names(fm), vapply(fm, function(x) paste(deparse(x), collapse = ""), "")), collapse = "; ")
}, error = function(e) "not read")
ihw_fit <- if (is.null(ihw_res)) "none" else tryCatch(
  sprintf("nbins = %d; nfolds = %d; penalty = %s; regularization_term per fold = %s; rejections = %d",
          nbins(ihw_res), nfolds(ihw_res), ihw_res@penalty,
          paste(format(regularization_term(ihw_res)), collapse = ", "), rejections(ihw_res)),
  error = function(e) "not read")
writeLines(c(
  sprintf("multiple_testing\t%s", ihw_used),
  sprintf("ihw_version\t%s", ver("IHW")),
  sprintf("ihw_arguments\tpvalues = P_fisher of every analysed gene-block; covariates = factor(tissue), %d levels; alpha = %g; covariate_type = %s; seed = %d",
          length(unique(G_tis)), ALPHA, IHW_COVARIATE_TYPE, DE_SEED),
  sprintf("ihw_arguments_at_package_default\t%s", ihw_defaults),
  sprintf("ihw_fit\t%s", ihw_fit),
  sprintf("package_versions\tlimma %s; edgeR %s; IHW %s; repfdr %s; R %s",
          ver("limma"), ver("edgeR"), ver("IHW"), ver("repfdr"), as.character(getRversion())),
  sprintf("sex_consistency\tprimary = %s; secondary = repfdr, %s", sexcons_primary, repfdr_status),
  sprintf("sex_consistency_primary\t%s", sexcons_primary),
  sprintf("sex_consistency_repfdr\t%s", repfdr_status),
  sprintf("sex_consistency_warnings\t%s",
          if (length(repfdr_warnings)) paste(unique(repfdr_warnings), collapse = " | ") else "none"),
  sprintf("alpha\t%s", ALPHA),
  sprintf("seed\t%d", DE_SEED),
  sprintf("hotspot_rule\t%s", hotspot_rule),
  sprintf("n_blocks_ok\t%d", sum(summ$status == "ok")),
  sprintf("n_blocks_single_sex\t%d", sum(single)),
  sprintf("min_median_libsize\t%g", MIN_MEDIAN_LIBSIZE),
  sprintf("n_blocks_skipped_degenerate\t%d", sum(startsWith(summ$status, "SKIP: degenerate"))),
  sprintf("n_fisher_tests\t%d", length(G_p)),
  sprintf("n_dose_tests_single_sex\t%d", sum(ok$n_dose_tests[ok$n_male == 0 | ok$n_female == 0])),
  sprintf("low_fraction_threshold\t%g", LOW_FRACTION),
  sprintf("technical_covariates\t%s",
          if (length(TECH_COV)) sprintf("%s (TRNSCRPT_META columns %s), in every model",
                                        paste(TECH_COV, collapse = ", "),
                                        paste(TECH_COV_COLUMNS[TECH_COV], collapse = ", ")) else "none"),
  sprintf("n_samples_dropped_missing_covariate\t%d%s", sum(cov_dropped),
          if (sum(cov_dropped)) paste0(" (", paste(sprintf("%s %d", names(cov_dropped)[cov_dropped > 0],
                                                           cov_dropped[cov_dropped > 0]), collapse = ", "), ")") else ""),
  sprintf("sample_key\tmixture_samples.tsv equal in order to bulk_samples.tsv in %d of %d tissues (the file is absent in the others); rows of the fraction table and of every Z matrix checked against the samples in every tissue",
          length(key_checked), length(tissues)),
  sprintf("model_combined\t~ sex + factor(week)%s (~ factor(week)%s in a one-sex block): F_dose_comb, P_dose_comb, lfc_1w..lfc_8w, P_8w_comb, logFC_sex, P_sex",
          cov_txt, cov_txt),
  sprintf("model_ordinal\t~ sex + week%s, week numeric 0,1,2,4,8 (~ week%s in a one-sex block): slope_week, P_week_lin",
          cov_txt, cov_txt),
  sprintf("model_interaction\t~ sex * factor(week)%s, two-sex blocks only: F_sexXweek, P_sexXweek", cov_txt),
  sprintf("model_per_sex\t~ factor(week)%s within each sex present: P_train_M, P_train_F (omnibus F over the week coefficients); 8w contrast: lfc_8w_M/F, z_8w_M/F, fdr_8w_M/F (BH within the block)",
          cov_txt),
  "dose_p\tP_fisher = Fisher's combination of the per-sex omnibus p over the sexes present: chi-square on 4 df with two sexes, the one sex's p with one sex",
  sprintf("model_fraction\tlm(fraction ~ week + sex%s), week numeric (fraction ~ week%s in a one-sex block): frac_week_slope, frac_week_p",
          cov_txt, cov_txt),
  sprintf("design\tlimma-trend (eBayes trend = TRUE, robust = TRUE) on log2-CPM (prior count 1, no TMM) of continuous Z; combined ~ sex + factor(week)%s; ordinal ~ sex + week%s; interaction ~ sex * factor(week)%s; per sex ~ factor(week)%s; dose p = Fisher over the sexes present; IHW ~ tissue (%s); 8w sex-consistency as in sex_consistency",
          cov_txt, cov_txt, cov_txt, cov_txt, IHW_COVARIATE_TYPE),
  sprintf("deviations\tcontinuous Z with limma-trend, not DESeq2 negative binomial; technical covariates: %s; repfdr on two columns (8w male, 8w female) pooled over blocks",
          if (length(TECH_COV)) paste(TECH_COV, collapse = ", ") else "none in this run (see the _covA and _covB reruns)")
), file.path(out_dir, "de_methods.tsv"))
write_provenance(out_dir,
  inputs = c(pheno_tsv, hotspots_tsv, file.path(bulk_root, tissues, "bulk_samples.tsv"),
             file.path(results_root, tissues, "pred_z", "genes.txt"),
             file.path(results_root, key_checked, "mixture_samples.tsv"),
             if (length(TECH_COV)) META_RDA),
  params = list(results_root = results_root, bulk_root = bulk_root, tissues = tissues,
                alpha = ALPHA, seed = DE_SEED, multiple_testing = ihw_used,
                ihw_version = ver("IHW"), ihw_covariate_type = IHW_COVARIATE_TYPE,
                sex_consistency_primary = sexcons_primary, sex_consistency_repfdr = repfdr_status,
                hotspot_rule = hotspot_rule, technical_covariates = I(TECH_COV),
                n_samples_dropped_missing_covariate = sum(cov_dropped),
                min_median_libsize = MIN_MEDIAN_LIBSIZE,
                n_blocks_ok = sum(summ$status == "ok")))

cat("\n=== DE summary (hotspots first) ===\n")
print(utils::head(summ[, c("tissue","cell_type","is_hotspot","n_sig_dose_IHW","n_sexcons_concordance_8w",
                           "n_up_both_8w","n_down_both_8w","n_sig_interaction","frac_week_p","status")], 25))
cat("\n=== headline (STEP3 §6.3) ===\n")
print(headline, row.names = FALSE)
cat(sprintf("\nBlocks: %d total, %d ok (%d one-sex), %d hotspots. Outputs in %s\n",
            nrow(summ), sum(summ$status == "ok"), sum(single), sum(summ$is_hotspot %in% TRUE), out_dir))
