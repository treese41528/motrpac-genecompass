#!/usr/bin/env Rscript
# step6_cdseq_naming_check.R -- Step 6 (records, tests, hygiene): does naming the CDSeq proportions through
# CDSeq's own sample names (deconvolution/R/run_cdseq_rr.R from 2026-11-14) give the table the positional
# naming gave? Checked on the one CDSeq run kept on disk (BLOOD, Reduce-Recover, 2026-07-14). CDSeq is not
# rerun: the saved result object is read and both namings are applied to it.
#
#   Rscript analysis/audit_remediation/step6_cdseq_naming_check.R [out.tsv]
# Out: docs/audit/remediation/record/step6/cdseq_naming_check.tsv
args <- commandArgs(trailingOnly = TRUE)
out <- if (length(args) >= 1) args[1] else "docs/audit/remediation/record/step6/cdseq_naming_check.tsv"
D <- "data/deconvolution/results/_deployed_backup_20260715/BLOOD/cdseq_rr"
B <- "data/deconvolution/motrpac_bulk/BLOOD"
res <- readRDS(file.path(D, "cdseq_result.rds"))
bs <- trimws(readLines(file.path(B, "bulk_samples.tsv"))); bs <- bs[bs != ""]
est <- res$estProp

# the naming the script used until 2026-11-14 (by position)
old <- est; if (nrow(old) != length(bs)) old <- t(old)
colnames(old) <- paste0("CDSeq_", seq_len(ncol(old))); rownames(old) <- bs
# the naming it uses now (through CDSeq's column names; stops if they are not the samples, in order)
stopifnot(!is.null(colnames(est)), identical(colnames(est), bs))
new <- t(est); colnames(new) <- paste0("CDSeq_", seq_len(ncol(new)))

saved <- as.matrix(read.csv(file.path(D, "cdseq_proportions.csv"), row.names = 1, check.names = FALSE))
rownames(saved) <- as.character(rownames(saved))
rec <- data.frame(
  check = c("estProp shape (cell types x samples)", "estProp columns equal bulk_samples.tsv, in order",
            "new naming identical to positional naming (values and names)",
            "new naming equals the saved cdseq_proportions.csv (names; values to 1e-12)",
            "max abs difference, new naming against the saved table"),
  value = c(sprintf("%d x %d", nrow(est), ncol(est)), as.character(identical(colnames(est), bs)),
            as.character(identical(old, new)),
            as.character(identical(dimnames(saved), dimnames(new)) && max(abs(saved - new)) < 1e-12),
            format(max(abs(saved - new)), digits = 3)),
  source = c(rep(file.path(D, "cdseq_result.rds"), 3), rep(file.path(D, "cdseq_proportions.csv"), 2)),
  stringsAsFactors = FALSE)
dir.create(dirname(out), recursive = TRUE, showWarnings = FALSE)
write.table(rec, out, sep = "\t", quote = FALSE, row.names = FALSE)
print(rec[, 1:2], right = FALSE)
stopifnot(identical(old, new))
