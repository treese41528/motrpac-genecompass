#!/usr/bin/env Rscript
# check_sample_join.R -- audit finding 6.9: is the positional sample join right in every tissue?
#
# Stages 8-13 name each deconvolved sample mix<i> (run_deconvolution.R) and recover the animal as the i-th line of
# <bulk_root>/<TISSUE>/bulk_samples.tsv. Nothing in those stages checks that the two orders agree. This script
# checks every link of the chain against the data, read-only:
#
#   A. package -> bulk.mtx     Rebuild the tissue's count matrix from the MoTrPAC data package, selecting columns BY
#                              VIALLABEL NAME in bulk_samples.tsv order and applying the committed liftover map.
#                              It must equal bulk.mtx entry for entry.
#   B. bulk.mtx -> theta       estimated_fractions.csv has one row per bulk row, named mix1..mixN in order.
#   C. bulk.mtx -> Z           BayesPrism allocates every read of a gene to one cell type, so the per-type
#                              expression tables (pred_z) summed over cell types must reproduce the bulk row they came
#                              from. For each mix<i> the nearest bulk row (L1 distance) must be row i.
#   D. phenotype table         deconvolution/reference/motrpac_sample_pheno.tsv (no generator in the repository) must
#                              equal the package's PHENO for sex, group and sacrifice time.
#   E. sex from expression     Y-linked genes (Ddx3y, Eif2s3y, Kdm5d, Uty) in bulk row i must agree with the
#                              recorded sex of the i-th viallabel: an independent check of sample to phenotype.
#
# A and C carry a negative control: the same comparison with the sample order shifted by one position.
#
# Usage: check_sample_join.R <bulk_root> <results_root> <out_dir>
#   e.g. data/deconvolution/motrpac_bulk  data/deconvolution/results_refs4all_rebuild/motrpac  data/audit_remediation/step1

.libPaths(c(Sys.getenv("R_LIBS_USER"), .libPaths()))
suppressMessages({library(Matrix); library(data.table)})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 3) stop("usage: check_sample_join.R <bulk_root> <results_root> <out_dir>")
bulk_root <- args[1]; res_root <- args[2]; out_dir <- args[3]
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

DATA_DIR  <- Sys.getenv("MOTRPAC_DATA_DIR", "data/motrpac/rat_training_6mo/data")
MAP_F     <- "deconvolution/reference/motrpac_bulk_liftover.tsv"
PHENO_F   <- "deconvolution/reference/motrpac_sample_pheno.tsv"
SEXCHR_F  <- "deconvolution/reference/rat_sex_chrom_genes.tsv"
Y_SYMBOLS <- c("Ddx3y", "Eif2s3y", "Kdm5d", "Uty")          # fixed before any value was read
META      <- c("feature", "feature_ID", "tissue", "assay")
ld <- function(path) { e <- new.env(); load(path, envir = e); get(ls(e)[1], envir = e) }

map   <- fread(MAP_F, na.strings = c("NA", ""))
pheno <- fread(PHENO_F, colClasses = "character")
pkg   <- as.data.table(ld(file.path(DATA_DIR, "PHENO.rda")))
pkg[, viallabel := as.character(viallabel)]
sexchr <- fread(SEXCHR_F)
y_ids  <- sexchr[gene_symbol %in% Y_SYMBOLS, feature_ID]
cat(sprintf("Y-linked check genes: %s\n", paste(sexchr[gene_symbol %in% Y_SYMBOLS, paste0(gene_symbol, "=", feature_ID)], collapse = ", ")))

# ---- D (whole table): the committed phenotype table against the package ----
pcols <- c("sex", "group", "key.anirandgroup", "intervention", "sacrificetime")
pk <- pkg[match(pheno$viallabel, viallabel)]
whole <- sapply(pcols, function(k) sum(as.character(pk[[k]]) != pheno[[k]] | is.na(pk[[k]])))
cat(sprintf("phenotype table: %d rows; viallabels absent from the package: %d; mismatches per column: %s\n",
            nrow(pheno), sum(is.na(pk$viallabel)), paste(names(whole), whole, sep = "=", collapse = " ")))

shift1 <- function(n) c(2:n, 1)                              # the negative control: every sample moved one place

tissues <- sort(basename(list.dirs(res_root, recursive = FALSE)))
summ <- list(); per <- list()
for (tis in tissues) {
  bdir <- file.path(bulk_root, tis); rdir <- file.path(res_root, tis)
  via <- trimws(readLines(file.path(bdir, "bulk_samples.tsv"))); via <- via[via != ""]
  B   <- as.matrix(readMM(file.path(bdir, "bulk.mtx")))       # samples x genes
  bg  <- readLines(file.path(bdir, "bulk_genes.tsv"))
  n   <- length(via)

  # A. rebuild from the package, columns selected by NAME
  rc  <- ld(file.path(DATA_DIR, sprintf("TRNSCRPT_%s_RAW_COUNTS.rda", tis)))
  ids <- as.character(rc$feature_ID)
  scols <- setdiff(colnames(rc), META)
  lifted <- map$lifted_id[match(ids, map$old_id)]
  keep <- !is.na(lifted)
  a_names <- all(via %in% scols)
  M  <- as.matrix(rc[keep, via, drop = FALSE]); storage.mode(M) <- "double"
  Mc <- rowsum(M, group = lifted[keep])
  a_genes <- setequal(rownames(Mc), bg)
  E  <- t(Mc[bg, , drop = FALSE])                             # expected samples x genes, in bulk_genes order
  a_dim   <- identical(dim(E), dim(B))
  a_exact <- if (a_dim) mean(E == B) else NA
  a_shift <- if (a_dim) mean(E[shift1(n), ] == B) else NA     # negative control
  a_order <- identical(scols, via)                            # the file order is the package's column order

  # B. theta rows
  th <- fread(file.path(rdir, "estimated_fractions.csv"))
  b_n <- nrow(th); b_names <- identical(th[[1]], paste0("mix", seq_len(n)))

  # C. sum of the per-type expression over cell types against the bulk rows
  cf <- fread(file.path(rdir, "pred_z", "celltype_files.tsv"))
  zg <- readLines(file.path(rdir, "pred_z", "genes.txt"))
  S <- NULL; c_rows <- TRUE
  for (f in cf$file) {
    z <- fread(file.path(rdir, "pred_z", f))
    c_rows <- c_rows && identical(z[[1]], paste0("mix", seq_len(n)))
    zm <- as.matrix(z[, -1]); if (!identical(colnames(zm), zg)) zm <- zm[, zg, drop = FALSE]
    S <- if (is.null(S)) zm else S + zm
  }
  Bz <- B[, match(zg, bg), drop = FALSE]
  c_maxabs <- max(abs(S - Bz))
  c_maxrel <- max(abs(rowSums(S) - rowSums(Bz)) / rowSums(Bz))
  # nearest bulk row for each mix<i>, by L1 distance over the Z genes
  near <- sapply(seq_len(n), function(i) which.min(colSums(abs(t(Bz) - S[i, ]))))
  c_self  <- sum(near == seq_len(n))
  c_shift_abs <- mean(abs(S[shift1(n), ] - Bz))               # negative control: mean |difference| when shifted
  c_true_abs  <- mean(abs(S - Bz))

  # D. phenotype rows of this tissue
  ph <- pheno[match(via, viallabel)]; pk <- pkg[match(via, viallabel)]
  d_missing <- sum(is.na(ph$viallabel))
  d_mismatch <- sum(sapply(pcols, function(k) sum(as.character(pk[[k]]) != ph[[k]], na.rm = TRUE)))

  # E. sex from Y-linked expression
  yi <- match(y_ids, bg); yi <- yi[!is.na(yi)]
  ycpm <- rowSums(B[, yi, drop = FALSE]) / rowSums(B) * 1e6
  sex <- ph$sex
  mlo <- if (any(sex == "male")) min(ycpm[sex == "male"]) else NA
  fhi <- if (any(sex == "female")) max(ycpm[sex == "female"]) else NA

  summ[[tis]] <- data.table(
    tissue = tis, n_samples = n, n_bulk_rows = nrow(B), n_bulk_genes = ncol(B),
    A_viallabels_in_package = a_names, A_file_order_is_package_order = a_order, A_gene_sets_equal = a_genes,
    A_share_entries_equal = a_exact, A_share_equal_shifted = a_shift,
    B_theta_rows = b_n, B_theta_names_in_order = b_names,
    C_types = nrow(cf), C_z_genes = length(zg), C_z_rows_in_order = c_rows,
    C_max_abs_diff = c_maxabs, C_max_rel_library_diff = c_maxrel,
    C_nearest_is_self = c_self, C_mean_abs_diff = c_true_abs, C_mean_abs_diff_shifted = c_shift_abs,
    D_pheno_missing = d_missing, D_pheno_mismatch = d_mismatch,
    E_n_y_genes = length(yi), E_n_male = sum(sex == "male"), E_n_female = sum(sex == "female"),
    E_min_male_ycpm = mlo, E_max_female_ycpm = fhi)
  per[[tis]] <- data.table(tissue = tis, i = seq_len(n), mix = paste0("mix", seq_len(n)), viallabel = via,
                           sex = sex, group = ph$group, sacrificetime = ph$sacrificetime,
                           y_cpm = ycpm, library = rowSums(B), z_nearest_bulk_row = near,
                           z_abs_diff = rowSums(abs(S - Bz)))
  cat(sprintf("[%s] n=%d  A equal=%.6f (shifted %.4f)  B rows=%d names=%s  C nearest-self=%d/%d maxabs=%.3g  D missing=%d mismatch=%d  E male>=%.2f female<=%.2f\n",
              tis, n, a_exact, a_shift, b_n, b_names, c_self, n, c_maxabs, d_missing, d_mismatch,
              ifelse(is.na(mlo), NaN, mlo), ifelse(is.na(fhi), NaN, fhi)))
}
summ <- rbindlist(summ); per <- rbindlist(per)
fwrite(summ, file.path(out_dir, "sample_join_summary.tsv"), sep = "\t")
fwrite(per,  file.path(out_dir, "sample_join_per_sample.tsv"), sep = "\t")

ok <- with(summ, all(A_viallabels_in_package) && all(A_gene_sets_equal) && all(A_share_entries_equal == 1) &&
             all(B_theta_rows == n_samples) && all(B_theta_names_in_order) && all(C_z_rows_in_order) &&
             all(C_nearest_is_self == n_samples) && all(D_pheno_missing == 0) && all(D_pheno_mismatch == 0))
allm <- per[sex == "male", min(y_cpm)]; allf <- per[sex == "female", max(y_cpm)]
cat(sprintf("\nALL TISSUES: links A-D %s; sex check: lowest male Y CPM %.2f, highest female Y CPM %.2f -> %s\n",
            if (ok) "PASS" else "FAIL", allm, allf, if (allm > allf) "separated" else "NOT separated"))
writeLines(c(sprintf("tissues\t%d", nrow(summ)), sprintf("samples\t%d", nrow(per)),
             sprintf("links_A_to_D\t%s", if (ok) "PASS" else "FAIL"),
             sprintf("pheno_table_rows\t%d", nrow(pheno)),
             sprintf("pheno_table_mismatches\t%d", sum(whole)),
             sprintf("min_male_y_cpm\t%.4f", allm), sprintf("max_female_y_cpm\t%.4f", allf)),
           file.path(out_dir, "sample_join_verdict.tsv"))
