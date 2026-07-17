# ---------------------------------------------------------------------------
# CDSeq on the REAL MoTrPAC bulk -- REDUCE-RECOVER mode (INSURANCE RUN).
#
# WHY THIS EXISTS: run_cdseq_realbulk.R (job 11281690) was launched with
# block_number=1 and dilution_factor=1 -- the slowest possible configuration:
#   * dilution_factor=1  -> the Gibbs sampler assigns EVERY read in the bulk
#                           (CDSeq's own example uses 50).
#   * block_number=1     -> CDSeq parallelises ACROSS BLOCKS, so 16 cores idle;
#                           the run is effectively single-threaded.
# It writes NOTHING until CDSeq() returns, so a wall-clock timeout loses it all.
# This run trades exactness for a guaranteed answer.
#
# *** APPROXIMATION DISCLOSURE (this is NOT the full-fidelity estimator) ***
#   1. DILUTION: counts are divided by DIL (default 25) before sampling. This
#      is read-subsampling: unbiased in expectation for the proportions, but it
#      widens the posterior. It does NOT drop genes or samples.
#   2. GENE BLOCKING: BLK blocks x GSS randomly-sampled genes each. With the
#      defaults (16 x 1500 = 24,000 draws over ~11,319 genes) every gene is
#      covered ~2x in expectation, but coverage is RANDOM, not guaranteed --
#      the script logs the realised per-gene coverage and names any gene that
#      landed in zero blocks.
#   3. Cell types are matched across blocks by CDSeq's Hungarian step.
# The full-fidelity run (block=1, dilution=1) remains authoritative if it lands.
#
# THE TEST is unchanged: if a large CDSeq component matches NOTHING in the PBMC
# reference, the reference is missing a real compartment. The qualitative
# verdict is robust to both approximations above; the exact percentages are not.
# ---------------------------------------------------------------------------
suppressPackageStartupMessages({library(Matrix); library(CDSeq)})
set.seed(1)
args <- commandArgs(trailingOnly=TRUE)
TIS  <- ifelse(length(args)>=1, args[1], "BLOOD")
K    <- as.integer(ifelse(length(args)>=2, args[2], "14"))
ITER <- as.integer(ifelse(length(args)>=3, args[3], "700"))
DIL  <- as.integer(Sys.getenv("DILUTION",   "25"))
BLK  <- as.integer(Sys.getenv("BLOCKS",     "16"))
GSS  <- as.integer(Sys.getenv("GENE_SUBSET","1500"))
NC   <- as.integer(Sys.getenv("N_CORES",    "16"))

OUT <- file.path("data/deconvolution/results/motrpac", TIS, "cdseq_rr")
dir.create(OUT, recursive=TRUE, showWarnings=FALSE)

B   <- file.path("data/deconvolution/motrpac_bulk", TIS)
mat <- readMM(file.path(B,"bulk.mtx"))
bg  <- readLines(file.path(B,"bulk_genes.tsv"))
bs  <- readLines(file.path(B,"bulk_samples.tsv"))
mat <- as.matrix(mat); if (nrow(mat)!=length(bg)) mat <- t(mat)
rownames(mat) <- bg; colnames(mat) <- bs

# same gene axis BayesPrism used -> directly comparable, no extra filtering
gfile <- file.path("data/deconvolution/results/motrpac", TIS, "pred_z", "genes.txt")
if (file.exists(gfile)) {
  keep <- intersect(readLines(gfile), rownames(mat))
  mat  <- mat[keep, , drop=FALSE]
  cat(sprintf("restricted to BayesPrism's gene axis: %d genes\n", nrow(mat)))
}
mat <- round(mat); mode(mat) <- "integer"

cat(sprintf("bulk: %d genes x %d samples | K=%d | iters=%d\n", nrow(mat), ncol(mat), K, ITER))
cat(sprintf("REDUCE-RECOVER: blocks=%d gene_subset=%d dilution=%d cores=%d\n", BLK, GSS, DIL, NC))
cat(sprintf("total reads=%.3g -> after dilution ~%.3g\n", sum(as.numeric(mat)), sum(as.numeric(mat))/DIL))
cat(sprintf("expected per-gene block coverage = %.2fx (%d draws / %d genes)\n",
            BLK*GSS/nrow(mat), BLK*GSS, nrow(mat)))

t0 <- Sys.time()
res <- CDSeq(bulk_data       = mat,
             cell_type_number= K,
             mcmc_iterations = ITER,
             cpu_number      = NC,
             dilution_factor = DIL,
             gene_subset_size= GSS,
             block_number    = BLK,
             print_progress_msg_to_file = 1)
cat(sprintf("CDSeq(RR) wall time: %.1f min\n", as.numeric(difftime(Sys.time(), t0, units="mins"))))

prop <- res$estProp; gep <- res$estGEP
if (nrow(prop) != ncol(mat)) prop <- t(prop)
colnames(prop) <- paste0("CDSeq_", seq_len(ncol(prop))); rownames(prop) <- colnames(mat)
colnames(gep)  <- paste0("CDSeq_", seq_len(ncol(gep)))

# genes that survived the random block draw (the ONLY thing this run can drop)
gep_genes <- rownames(res$estGEP)
if (is.null(gep_genes)) gep_genes <- rownames(mat)[seq_len(nrow(gep))]
rownames(gep) <- gep_genes
missed <- setdiff(rownames(mat), gep_genes)
cat(sprintf("\ngenes recovered in GEP: %d / %d  (dropped by random blocking: %d)\n",
            length(gep_genes), nrow(mat), length(missed)))
if (length(missed)) writeLines(missed, file.path(OUT,"genes_missed_by_blocking.txt"))

write.csv(prop, file.path(OUT,"cdseq_proportions.csv"))
write.csv(gep,  file.path(OUT,"cdseq_gep.csv"))
saveRDS(res,    file.path(OUT,"cdseq_result.rds"))
writeLines(c(
  sprintf("mode=reduce-recover blocks=%d gene_subset=%d dilution=%d iters=%d K=%d", BLK,GSS,DIL,ITER,K),
  sprintf("genes_in=%d genes_recovered=%d genes_dropped=%d", nrow(mat), length(gep_genes), length(missed)),
  "APPROXIMATE: read-diluted + gene-blocked. Full-fidelity run = results/motrpac/BLOOD/cdseq/"
), file.path(OUT,"RUN_PARAMS.txt"))

cat("\n=== CDSeq mean proportions, reference-FREE (%) ===\n")
print(round(100*sort(colMeans(prop), decreasing=TRUE), 2))
cat("\nwrote:", OUT, "\n")
