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
mat <- as.matrix(mat); if (nrow(mat)!=length(bg)) mat <- t(mat)
rownames(mat) <- bg; colnames(mat) <- bs

gfile <- file.path("data/deconvolution/results/motrpac", TIS, "pred_z", "genes.txt")
if (file.exists(gfile)) {
  keep <- intersect(readLines(gfile), rownames(mat))
  mat  <- mat[keep, , drop=FALSE]
  cat(sprintf("restricted to BayesPrism's gene axis: %d genes\n", nrow(mat)))
}
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

prop <- res$estProp          # samples x K  (or K x samples)
gep  <- res$estGEP           # genes x K
if (nrow(prop) != ncol(mat)) prop <- t(prop)
colnames(prop) <- paste0("CDSeq_", seq_len(ncol(prop)))
rownames(prop) <- colnames(mat)
colnames(gep)  <- paste0("CDSeq_", seq_len(ncol(gep)))
rownames(gep)  <- rownames(mat)

write.csv(prop, file.path(OUT,"cdseq_proportions.csv"))
write.csv(gep,  file.path(OUT,"cdseq_gep.csv"))
saveRDS(res, file.path(OUT,"cdseq_result.rds"))
cat("\n=== CDSeq mean proportions (reference-FREE) ===\n")
print(round(100*sort(colMeans(prop), decreasing=TRUE), 2))
cat("\nwrote:", OUT, "\n")
