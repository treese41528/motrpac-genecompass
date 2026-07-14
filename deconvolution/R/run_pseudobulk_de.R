#!/usr/bin/env Rscript
# run_pseudobulk_de.R -- EXHAUSTIVE per-(tissue x cell-type) pseudobulk differential
# expression on BayesPrism's deconvolved per-cell-type expression Z, across the MoTrPAC
# endurance-training design. A faithful adaptation of the MoTrPAC PASS1B / Vetr-2024
# (Nat Commun 15:3346) DE recipe to our continuous-Z, limma-trend setting.
#
# WHAT IT COMPUTES, per (tissue x cell-type) block (the four "exhaustive" dimensions):
#   (1) FULL COVERAGE -- every gene expressed (>0) in >=1 sample is tested (NOT the prior
#       >=50%-of-samples filter); only all-zero genes are dropped and the count is logged.
#   (2) RICHER DOSE MODEL (combined, both sexes):
#         * omnibus moderated-F across factor(week) {1,2,4,8w vs control} = "any-timepoint
#           training effect", sex-adjusted   (the limma analog of Vetr's nbinomLRT time-course)
#         * per-timepoint contrasts 1w/2w/4w/8w vs control (sex-adjusted log2FC)
#         * sex x dose INTERACTION omnibus-F (does the dose response differ by sex)
#         * ordinal linear dose slope (~ sex + week_numeric) -- retained for continuity with
#           the prior linear-only run
#   (3) SEX-STRATIFIED (Vetr-faithful):
#         * DE fit SEPARATELY within each sex (~ factor(week)): per-sex training omnibus-F,
#           per-sex per-timepoint contrasts, per-sex signed z at 8w
#           z = qnorm(p/2, lower.tail=FALSE) * sign(log2FC)   (exactly Vetr's z definition)
#         * sexes combined by FISHER's sum-of-logs into a meta training p
#         * sex-consistent 8w states from repfdr (global replicability) x sign concordance:
#           up_both (F1_M1) / down_both (F-1_M-1) / opposite / M_only / F_only / null
#   (4) IHW multiple testing on the Fisher meta-p with TISSUE as covariate (Vetr used IHW
#       with a tissue covariate); per-block BH retained as a local secondary.
#
# DELIBERATE DEVIATIONS from Vetr (documented, not silent):
#   * limma-trend on log2-CPM of CONTINUOUS Z, NOT DESeq2 nbinomLRT on raw counts: Z is
#     BayesPrism get.exp = posterior expected COUNT-MASS. VERIFIED on HEART/BLOOD/LUNG -- Z is
#     continuous (0 integer-valued blocks) and abundance-scaled: abundant parenchyma reach
#     1e2-1e5, but rare-in-tissue cell types (<~1% fraction; the MAJORITY of cell types by
#     count) are ~100% <1 (median nonzero ~0.004) and mid-abundance immune types (~1-20%) are
#     ~25-58% <1. So integer-rounding for an NB model would zero out essentially all rare-type
#     signal AND a quarter-to-half of the mid-abundance immune signal (the hotspot cells). Within
#     a block per-sample libsizes are comparable, so logCPM + eBayes(trend, robust) is justified
#     (Squair 2021 pseudobulk; EMBEDDING_DE_STANDARDS.md).
#   * Technical covariates (RIN, 5'-3' bias, globin %, PCR-dup %) are NOT modeled: they are
#     absent from motrpac_sample_pheno.tsv, and Z is a deconvolved expectation (not raw reads),
#     so per-read technical artifacts are largely upstream/absorbed. Flagged as a limitation.
#   * repfdr is run GLOBALLY on the assembled (z_M_8w, z_F_8w) matrix (more features -> stable
#     EM, and closer to Vetr's single cross-dataset analysis) rather than per block.
#   * Effect MAGNITUDES are expected to be SMALL (Vetr: ~52 genes/tissue >2 SDpheno; 56% of
#     bulk fold-changes within 0.67-1.5x): the signal is reliable separability, not large logFC.
#
# Composition / activity confound: per block we also test whether the cell-type FRACTION
# (estimated_fractions.csv) trends with dose (sex-adjusted). A gene flagged in a block whose
# fraction also shifts with dose may be a composition artifact -> read RELATIVE/differential.
#
# Args:  run_pseudobulk_de.R <results_root> <bulk_root> <pheno_tsv> <hotspots_tsv> <out_dir> [TISSUE ...]
#   (see deconvolution/R/run_pseudobulk_de.sh). Reads pred_z already on disk -- no deconv rerun.
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

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 5)
  stop("usage: run_pseudobulk_de.R <results_root> <bulk_root> <pheno_tsv> <hotspots_tsv> <out_dir> [TISSUE ...]")
results_root <- args[1]; bulk_root <- args[2]; pheno_tsv <- args[3]
hotspots_tsv <- args[4]; out_dir <- args[5]
tissue_filter <- if (length(args) >= 6) toupper(args[6:length(args)]) else NULL

WEEK_NUM  <- c("control" = 0, "1w" = 1, "2w" = 2, "4w" = 4, "8w" = 8)
WEEK_LV   <- c("control", "1w", "2w", "4w", "8w")
TP        <- c("1w", "2w", "4w", "8w")
ALPHA     <- 0.05
MIN_SAMPLES_BLOCK   <- 12   # combined fit needs sex + 5 week levels estimable + residual df
MIN_SAMPLES_PER_SEX <- 8    # per-sex factor(week) (up to 5 levels) needs residual df
MIN_GENES           <- 50   # eBayes mean-variance trend needs a reasonable gene set

# ---- phenotype (all-character so 11-digit viallabels never go scientific) ----
ph <- read.delim(pheno_tsv, stringsAsFactors = FALSE, colClasses = "character")
ph <- ph[!duplicated(ph$viallabel), ]
rownames(ph) <- ph$viallabel

# ---- hotspot set: q_sup_trained < 0.05 (the canonical 22-block gate; for ordering only) ----
hot_key <- character(0); hot_auc <- numeric(0)
hs <- tryCatch(read.delim(hotspots_tsv, stringsAsFactors = FALSE), error = function(e) NULL)
if (!is.null(hs) && all(c("tissue", "cell_type", "q_sup_trained") %in% names(hs))) {
  q  <- suppressWarnings(as.numeric(hs$q_sup_trained))
  hh <- hs[!is.na(q) & q < 0.05, , drop = FALSE]
  hot_key <- paste(toupper(hh$tissue), hh$cell_type, sep = "||")
  hot_auc <- suppressWarnings(as.numeric(hh$sup_trained_auc)); names(hot_auc) <- hot_key
}
cat(sprintf("hotspot blocks (q_sup_trained<0.05): %d\n", length(hot_key)))

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

# helper: per-sex fit -> list(omni F/p over factor(week), per-timepoint logFC+p, n)
fit_one_sex <- function(logcpm, grp) {
  wf <- droplevels(factor(grp, levels = WEEK_LV))
  if (ncol(logcpm) < MIN_SAMPLES_PER_SEX || nlevels(wf) < 2) return(NULL)
  d  <- model.matrix(~ wf)
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
summary_rows <- list()
G_p <- c(); G_tis <- c(); G_key <- c()                 # global IHW inputs (fisher_p, tissue, block::gene)
G_zM <- c(); G_zF <- c(); G_zkey <- c()                # global repfdr inputs (8w signed z)

for (TIS in tissues) {
  pz    <- file.path(results_root, TIS, "pred_z")
  types <- trimws(readLines(file.path(pz, "types.txt"), encoding = "UTF-8"))
  types <- types[types != ""]
  via   <- trimws(readLines(file.path(bulk_root, TIS, "bulk_samples.tsv"))); via <- via[via != ""]
  n     <- length(via)
  meta  <- data.frame(mix = paste0("mix", seq_len(n)), viallabel = via,
                      sex = ph[via, "sex"], group = ph[via, "group"], stringsAsFactors = FALSE)
  frac_file <- file.path(results_root, TIS, "estimated_fractions.csv")
  fracs <- if (file.exists(frac_file)) read.csv(frac_file, row.names = 1, check.names = FALSE) else NULL
  out_tis <- file.path(out_dir, TIS); dir.create(out_tis, showWarnings = FALSE, recursive = TRUE)
  cat(sprintf("\n[%s] %d samples x %d cell types\n", TIS, n, length(types)))

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
      Z  <- Z[meta$mix, , drop = FALSE]
      ok <- !is.na(meta$week <- WEEK_NUM[meta$group]) & !is.na(meta$sex) & meta$group %in% WEEK_LV
      if (sum(ok) < MIN_SAMPLES_BLOCK) stop(sprintf("only %d usable samples", sum(ok)))
      m  <- meta[ok, ]
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

      dge    <- DGEList(counts = counts)
      logcpm <- edgeR::cpm(dge, log = TRUE, prior.count = 1)
      genes  <- rownames(logcpm)
      sexf   <- factor(m$sex)
      weekf  <- droplevels(factor(m$group, levels = WEEK_LV))
      weekn  <- as.numeric(m$week)
      two_sex <- nlevels(sexf) >= 2

      # ---- combined factor model: omnibus dose (sex-adjusted) + per-timepoint logFC + sex ----
      des_c <- if (two_sex) model.matrix(~ sexf + weekf) else model.matrix(~ weekf)
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
      des_o <- if (two_sex) model.matrix(~ sexf + weekn) else model.matrix(~ weekn)
      fit_o <- fit_trend(logcpm, des_o)
      tt_l  <- topTable(fit_o, coef = "weekn", number = Inf, sort.by = "none")

      # ---- sex x dose interaction omnibus ----
      F_int <- rep(NA_real_, length(genes)); P_int <- rep(NA_real_, length(genes))
      if (two_sex && nlevels(weekf) >= 2) {
        des_i <- model.matrix(~ sexf * weekf)
        icoef <- grep(":", colnames(des_i), value = TRUE)
        if (length(icoef) >= 1) {
          fit_i <- fit_trend(logcpm, des_i)
          tt_i  <- topTable(fit_i, coef = icoef, number = Inf, sort.by = "none")
          F_int <- tt_i$F; P_int <- tt_i$P.Value
        }
      }

      # ---- per-sex stratified ----
      M <- if (two_sex) fit_one_sex(logcpm[, m$sex == "male",   drop = FALSE], m$group[m$sex == "male"])   else NULL
      Fm <- if (two_sex) fit_one_sex(logcpm[, m$sex == "female", drop = FALSE], m$group[m$sex == "female"]) else NULL
      getcol <- function(o, mat, lv) if (!is.null(o) && lv %in% o$levels) o[[mat]][, paste0("wf", lv)] else rep(NA_real_, length(genes))
      P_train_M <- if (!is.null(M))  M$P  else rep(NA_real_, length(genes))
      P_train_F <- if (!is.null(Fm)) Fm$P else rep(NA_real_, length(genes))
      lfc_8w_M <- if (!is.null(M)  && "8w" %in% M$levels)  M$lfc[, "wf8w"]  else rep(NA_real_, length(genes))
      lfc_8w_F <- if (!is.null(Fm) && "8w" %in% Fm$levels) Fm$lfc[, "wf8w"] else rep(NA_real_, length(genes))
      p_8w_M  <- if (!is.null(M)  && "8w" %in% M$levels)  M$p[, "wf8w"]  else rep(NA_real_, length(genes))
      p_8w_F  <- if (!is.null(Fm) && "8w" %in% Fm$levels) Fm$p[, "wf8w"] else rep(NA_real_, length(genes))
      # signed z at 8w (Vetr) ; per-sex BH on the 8w contrast for the concordance fallback
      z_8w_M  <- qnorm(p_8w_M / 2, lower.tail = FALSE) * sign(lfc_8w_M)
      z_8w_F  <- qnorm(p_8w_F / 2, lower.tail = FALSE) * sign(lfc_8w_F)
      fdr_8w_M <- p.adjust(p_8w_M, "BH"); fdr_8w_F <- p.adjust(p_8w_F, "BH")

      # ---- Fisher sex-combined training meta-p ----
      P_fisher <- ifelse(is.finite(P_train_M) & is.finite(P_train_F),
                         pchisq(-2 * (log(P_train_M) + log(P_train_F)), df = 4, lower.tail = FALSE),
                         NA_real_)

      # composition confound: cell-type fraction ~ dose (sex-adjusted)
      frac_p <- NA_real_; frac_slope <- NA_real_
      if (!is.null(fracs) && ct %in% colnames(fracs)) {
        fv <- as.numeric(fracs[m$mix, ct])
        lf <- tryCatch(summary(lm(fv ~ weekn + sexf)), error = function(e) NULL)
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

      # accumulate for global IHW (Fisher meta-p ~ tissue) and global repfdr (8w z)
      gk <- paste(key, genes, sep = "||")
      fin <- is.finite(P_fisher)
      G_p   <- c(G_p, P_fisher[fin]); G_tis <- c(G_tis, rep(TIS, sum(fin))); G_key <- c(G_key, gk[fin])
      zin <- is.finite(z_8w_M) & is.finite(z_8w_F)
      G_zM <- c(G_zM, z_8w_M[zin]); G_zF <- c(G_zF, z_8w_F[zin]); G_zkey <- c(G_zkey, gk[zin])

      block_tabs[[key]] <- out
      list(tissue = TIS, cell_type = ct, is_hotspot = is_hot,
           sup_trained_auc = if (is_hot) unname(hot_auc[key]) else NA_real_,
           n_samples = nrow(m), n_male = sum(m$sex == "male"), n_female = sum(m$sex == "female"),
           n_genes_tested = nrow(out), n_genes_dropped_allzero = n_drop,
           mean_fraction = mean_frac, median_libsize = median(libsize), frac_zero = block_zero,
           n_sig_dose_fisher_BHblock = sum(out$FDR_BH_block < ALPHA, na.rm = TRUE),
           n_sig_interaction = sum(p.adjust(out$P_sexXweek, "BH") < ALPHA, na.rm = TRUE),
           n_sig_sex = sum(out$FDR_sex < ALPHA, na.rm = TRUE),
           frac_week_slope = frac_slope, frac_week_p = frac_p, status = "ok")
    }, error = function(e) {
      list(tissue = TIS, cell_type = ct, is_hotspot = is_hot,
           sup_trained_auc = if (is_hot) unname(hot_auc[key]) else NA_real_,
           n_samples = NA_integer_, n_male = NA_integer_, n_female = NA_integer_,
           n_genes_tested = NA_integer_, n_genes_dropped_allzero = NA_integer_,
           mean_fraction = NA_real_, median_libsize = NA_real_, frac_zero = NA_real_,
           n_sig_dose_fisher_BHblock = NA_integer_, n_sig_interaction = NA_integer_,
           n_sig_sex = NA_integer_, frac_week_slope = NA_real_, frac_week_p = NA_real_,
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
FDR_IHW_map <- setNames(rep(NA_real_, length(G_key)), G_key)
ihw_used <- "none"
if (length(G_p) > 50) {
  ir <- tryCatch({
    r <- ihw(G_p, factor(G_tis), alpha = ALPHA); ihw_used <<- "IHW~tissue"; adj_pvalues(r)
  }, error = function(e) {
    ihw_used <<- paste0("BH-global (IHW failed: ", conditionMessage(e), ")"); p.adjust(G_p, "BH")
  })
  FDR_IHW_map <- setNames(ir, G_key)
}
cat(sprintf("\nGlobal multiple-testing: %s over %d Fisher meta-p tests.\n", ihw_used, length(G_p)))

# ================= PASS 2: global repfdr sex-consistency at 8w =================
# state by replicability (repfdr local fdr) x sign concordance; concordance fallback if repfdr fails.
repfdr_localfdr <- setNames(rep(NA_real_, length(G_zkey)), G_zkey)
sexcons_method <- "concordance(per-sex BH 8w + sign)"
if (length(G_zM) > 200) {
  rf <- tryCatch({
    zmat <- cbind(male = G_zM, female = G_zF)
    bz   <- ztobins(zmat)
    rr   <- repfdr(bz$pdf.binned.z, bz$binned.z.mat, non.null = "replication")
    sexcons_method <<- "repfdr(replication) x sign"
    rr$mat[, "Fdr"]   # tail-area Bayes FDR per feature for the replication alternative
  }, error = function(e) { sexcons_method <<- paste0("concordance fallback (repfdr failed: ", conditionMessage(e), ")"); NULL })
  if (!is.null(rf)) repfdr_localfdr <- setNames(rf, G_zkey)
}
cat(sprintf("Sex-consistency caller: %s over %d gene-blocks with both-sex 8w z.\n", sexcons_method, length(G_zM)))

# assign per-gene FDR_IHW + sex-consistency state back into the block tables
state_counts <- function(df) c(
  up_both   = sum(df$sexcons_8w == "up_both",   na.rm = TRUE),
  down_both = sum(df$sexcons_8w == "down_both", na.rm = TRUE),
  sexspec   = sum(df$sexcons_8w %in% c("M_only", "F_only"), na.rm = TRUE),
  opposite  = sum(df$sexcons_8w == "opposite",  na.rm = TRUE))
for (key in names(block_tabs)) {
  df <- block_tabs[[key]]
  gk <- paste(key, df$gene, sep = "||")
  df$FDR_IHW <- unname(FDR_IHW_map[gk])
  lf <- unname(repfdr_localfdr[gk])
  # sex-consistency state
  repl <- is.finite(lf) & lf < ALPHA
  conc_sig <- is.finite(df$fdr_8w_M) & df$fdr_8w_M < ALPHA & is.finite(df$fdr_8w_F) & df$fdr_8w_F < ALPHA
  same_up   <- df$z_8w_M > 0 & df$z_8w_F > 0
  same_down <- df$z_8w_M < 0 & df$z_8w_F < 0
  sig_either <- (is.finite(df$fdr_8w_M) & df$fdr_8w_M < ALPHA) | (is.finite(df$fdr_8w_F) & df$fdr_8w_F < ALPHA)
  # primary = repfdr replicability where available, else per-sex-BH concordance
  base <- ifelse(repl | conc_sig, "sig_consistent", ifelse(sig_either, "sig_one", "null"))
  st <- rep("null", nrow(df))
  st[base == "sig_consistent" & same_up]   <- "up_both"
  st[base == "sig_consistent" & same_down] <- "down_both"
  st[base == "sig_consistent" & !same_up & !same_down] <- "opposite"
  st[base == "sig_one" & is.finite(df$fdr_8w_M) & df$fdr_8w_M < ALPHA & !(repl|conc_sig)] <- "M_only"
  st[base == "sig_one" & is.finite(df$fdr_8w_F) & df$fdr_8w_F < ALPHA & !(repl|conc_sig)] <- "F_only"
  df$repfdr_localfdr <- lf
  df$sexcons_8w <- st
  # order: IHW-significant dose genes first, then by Fisher p
  df <- df[order(ifelse(is.na(df$FDR_IHW), 1, df$FDR_IHW), df$P_fisher), ]
  block_tabs[[key]] <- df
  TIS <- sub("\\|\\|.*", "", key); ct <- sub(".*\\|\\|", "", key)
  write.table(df, file.path(out_dir, TIS, paste0("de__", safe(ct), ".tsv")),
              sep = "\t", quote = FALSE, row.names = FALSE)
}

# augment summary with global-FDR + state counts now available
summ <- do.call(rbind, lapply(summary_rows, function(r) {
  if (r$status == "ok") {
    df <- block_tabs[[paste(r$tissue, r$cell_type, sep = "||")]]
    sc <- state_counts(df)
    r$n_sig_dose_IHW <- sum(df$FDR_IHW < ALPHA, na.rm = TRUE)
    r$n_up_both_8w <- sc["up_both"]; r$n_down_both_8w <- sc["down_both"]
    r$n_sexspecific_8w <- sc["sexspec"]; r$n_opposite_8w <- sc["opposite"]
  } else {
    r$n_sig_dose_IHW <- NA_integer_; r$n_up_both_8w <- NA_integer_; r$n_down_both_8w <- NA_integer_
    r$n_sexspecific_8w <- NA_integer_; r$n_opposite_8w <- NA_integer_
  }
  as.data.frame(r, stringsAsFactors = FALSE)
}))
# column order: descriptors, then counts
ord <- c("tissue","cell_type","is_hotspot","sup_trained_auc","n_samples","n_male","n_female",
         "n_genes_tested","n_genes_dropped_allzero","mean_fraction","median_libsize","frac_zero",
         "n_sig_dose_IHW","n_sig_dose_fisher_BHblock","n_sig_interaction","n_up_both_8w",
         "n_down_both_8w","n_sexspecific_8w","n_opposite_8w","n_sig_sex","frac_week_slope",
         "frac_week_p","status")
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

# provenance sidecar
writeLines(c(
  sprintf("multiple_testing\t%s", ihw_used),
  sprintf("sex_consistency\t%s", sexcons_method),
  sprintf("alpha\t%s", ALPHA),
  sprintf("n_blocks_ok\t%d", sum(summ$status == "ok")),
  sprintf("n_fisher_tests\t%d", length(G_p)),
  sprintf("design\tlimma-trend(robust) on log2CPM of continuous Z; combined ~sex*factor(week) + ordinal ~sex+week_num; per-sex ~factor(week); Fisher sex-combine; IHW~tissue; repfdr 8w sex-consistency"),
  sprintf("deviations\tcontinuous-Z not DESeq2-NB; no RIN/globin/PCRdup covariates (absent + post-deconvolution)")
), file.path(out_dir, "de_methods.tsv"))

cat("\n=== DE summary (hotspots first) ===\n")
print(utils::head(summ[, c("tissue","cell_type","is_hotspot","n_sig_dose_IHW",
                           "n_up_both_8w","n_down_both_8w","n_sig_interaction","frac_week_p","status")], 25))
cat(sprintf("\nBlocks: %d total, %d ok, %d hotspots. Outputs in %s\n",
            nrow(summ), sum(summ$status == "ok"), sum(summ$is_hotspot %in% TRUE), out_dir))
