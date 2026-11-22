# ---------------------------------------------------------------------------
# CDSeq on the REAL MoTrPAC bulk -- REFERENCE-FREE complete deconvolution.
#
# WHY: every reference-based method (BayesPrism, MuSiC, DWLS, SCDC, Bisque,
# CIBERSORTx) fits theta on the K-simplex of the REFERENCE's cell types. If the
# bulk contains a compartment the reference lacks, that mass MUST be projected
# onto the existing K -- theta is distorted and no reference-based method can
# escape it. (Confirmed: BayesPrism's theta.first, i.e. NO reference update,
# already gives the 56.4% sink from the raw reference.)
#
# CDSeq estimates BOTH the proportions AND the cell-type expression profiles
# de novo, from the bulk alone (LDA-style Gibbs sampler; Kang et al., PLoS
# Comput Biol 2019). The reference is used only AFTERWARDS, to annotate.
#
# THE TEST: if a large CDSeq component matches NOTHING in the PBMC reference,
# the reference is missing a real compartment -> rebuild it. If CDSeq's
# components map cleanly onto the reference with sane proportions -> BayesPrism
# is misfitting an adequate reference -> fix the method.
# ---------------------------------------------------------------------------
suppressPackageStartupMessages({library(Matrix); library(CDSeq)})
set.seed(1)
args <- commandArgs(trailingOnly=TRUE)
TIS  <- ifelse(length(args)>=1, args[1], "BLOOD")
K    <- as.integer(ifelse(length(args)>=2, args[2], "14"))
ITER <- as.integer(ifelse(length(args)>=3, args[3], "700"))
NC   <- as.integer(Sys.getenv("N_CORES", "16"))
OUT  <- file.path("data/deconvolution/results/motrpac", TIS, "cdseq"); dir.create(OUT, recursive=TRUE, showWarnings=FALSE)

# bulk (genes x samples), on the SAME gene axis BayesPrism used -> directly comparable
B <- file.path("data/deconvolution/motrpac_bulk", TIS)
mat <- readMM(file.path(B,"bulk.mtx"))
bg  <- readLines(file.path(B,"bulk_genes.tsv"))
bs  <- readLines(file.path(B,"bulk_samples.tsv"))
bg <- bg[trimws(bg) != ""]; bs <- trimws(bs); bs <- bs[bs != ""]
# Orientation and names (2026-11-14). The matrix is put genes x samples by comparing its shape with the two
# name files. A shape that fits neither way, or both ways (as many genes as samples), stops the run: the
# rows and columns would be named by position with nothing to check them against.
mat <- as.matrix(mat)
if (length(bg) == length(bs) || !(all(dim(mat) == c(length(bg), length(bs))) || all(dim(mat) == c(length(bs), length(bg)))))
  stop(sprintf("bulk.mtx is %d x %d but bulk_genes.tsv has %d lines and bulk_samples.tsv %d: cannot name its rows and columns",
               nrow(mat), ncol(mat), length(bg), length(bs)))
if (nrow(mat)!=length(bg)) mat <- t(mat)
rownames(mat) <- bg; colnames(mat) <- bs
# Sample key: where the BayesPrism run this is compared with wrote one, it must agree with bulk_samples.tsv
# (deconvolution/R/sample_key.R; audit finding 6.9). A results folder without the key is reported, not fatal,
# because the proportions below are named by vial label through CDSeq's own column names, not by position.
RES_TIS <- file.path("data/deconvolution/results/motrpac", TIS)
if (file.exists(file.path(RES_TIS, "mixture_samples.tsv"))) {
  source(file.path(local({
    a <- commandArgs(trailingOnly = FALSE); f <- sub("^--file=", "", a[grep("^--file=", a)])
    if (length(f)) dirname(normalizePath(f[1])) else "deconvolution/R" }), "sample_key.R"))
  invisible(read_sample_key(file.path(B, "bulk_samples.tsv"), RES_TIS, TIS))
  cat("sample key: mixture_samples.tsv of the BayesPrism run equals bulk_samples.tsv, in order\n")
} else cat(sprintf("sample key: no mixture_samples.tsv in %s; not checked\n", RES_TIS))

gfile <- file.path("data/deconvolution/results/motrpac", TIS, "pred_z", "genes.txt")
if (file.exists(gfile)) {
  keep <- intersect(readLines(gfile), rownames(mat))
  mat  <- mat[keep, , drop=FALSE]
  cat(sprintf("restricted to BayesPrism's gene axis: %d genes\n", nrow(mat)))
} else cat(sprintf("NOTE: %s not found: the bulk is NOT restricted to BayesPrism's gene axis (%d genes kept)\n",
                   gfile, nrow(mat)))
mat <- round(mat)
mode(mat) <- "integer"
cat(sprintf("bulk: %d genes x %d samples | K=%d | iters=%d | cores=%d\n", nrow(mat), ncol(mat), K, ITER, NC))

t0 <- Sys.time()
res <- CDSeq(bulk_data = mat,
             cell_type_number = K,
             mcmc_iterations = ITER,
             cpu_number = NC,
             dilution_factor = 1,
             gene_subset_size = NULL,
             block_number = 1,
             print_progress_msg_to_file = 0)
cat(sprintf("CDSeq wall time: %.1f min\n", as.numeric(difftime(Sys.time(), t0, units="mins"))))

# Name the proportions through CDSeq's own sample names, not by position (2026-11-14). CDSeq returns estProp
# as cell types x samples with the bulk matrix's column names on its columns. Until this date the script
# transposed "if the row count is not the sample count" and then wrote the names on by position, which is
# wrong without a message when K equals the number of samples or CDSeq reorders samples. When the names are
# right the table is the same as before.
cdseq_samples_by_type <- function(est, sample_names) {
  if (is.null(colnames(est)) || !identical(colnames(est), sample_names))
    stop("CDSeq's estProp does not carry the bulk matrix's sample names, in order, on its columns; ",
         "refusing to name the proportions by position")
  t(est)                                    # samples x K, rows = vial labels
}
prop <- cdseq_samples_by_type(res$estProp, colnames(mat))
gep  <- res$estGEP           # genes x K
colnames(prop) <- paste0("CDSeq_", seq_len(ncol(prop)))
colnames(gep)  <- paste0("CDSeq_", seq_len(ncol(gep)))
if (nrow(gep) != nrow(mat) || (!is.null(rownames(gep)) && !identical(rownames(gep), rownames(mat))))
  stop("CDSeq's estGEP rows are not the bulk matrix's genes, in order")
rownames(gep)  <- rownames(mat)

write.csv(prop, file.path(OUT,"cdseq_proportions.csv"))
write.csv(gep,  file.path(OUT,"cdseq_gep.csv"))
saveRDS(res, file.path(OUT,"cdseq_result.rds"))
cat("\n=== CDSeq mean proportions (reference-FREE) ===\n")
print(round(100*sort(colMeans(prop), decreasing=TRUE), 2))
cat("\nwrote:", OUT, "\n")
