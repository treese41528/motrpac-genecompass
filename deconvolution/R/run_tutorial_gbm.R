#!/usr/bin/env Rscript
# Replicate the BayesPrism GBM tutorial — end-to-end validation of the Stage 8
# install before pointing the pipeline at rat data.
#
# Deconvolves 169 TCGA-GBM bulk RNA-seq samples (bk.dat) against a 23,793-cell
# GBM scRNA reference (sc.dat) and reports per-sample cell-type fractions. This
# is the canonical Danko-Lab workflow:
#     cleanup.genes -> select.gene.type -> new.prism -> run.prism -> get.fraction
# If it runs and the malignant ("tumor") fraction dominates as in
# vendor/BayesPrism/tutorial_deconvolution.pdf, the BayesPrism + scran stack is
# correctly installed and the pipeline is trustworthy.
#
# Invoke via run_tutorial_gbm.sh (sets module env + libPaths + conda strip).
# Do NOT run the full job on a login node — run.prism Gibbs sampling is
# CPU-heavy; submit to a compute node. SMOKE=1 subsets to 3 bulk samples for a
# quick login-node sanity check of the whole chain.

suppressWarnings(suppressPackageStartupMessages({
  .libPaths(c(Sys.getenv("R_LIBS_USER"), .libPaths()))
  library(BayesPrism)
}))

# ---- config ----
proj    <- "/depot/reese18/apps/motrpac-genecompass"
data_f  <- file.path(proj, "vendor/BayesPrism/tutorial.dat/tutorial.gbm.rdata")
out_dir <- file.path(proj, "deconvolution/results/tutorial_gbm")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

smoke   <- Sys.getenv("SMOKE", unset = "0") == "1"
n.cores <- as.integer(Sys.getenv("N_CORES", unset = if (smoke) "2" else "4"))
cat(sprintf("BayesPrism %s | n.cores=%d | smoke=%s\n",
            as.character(packageVersion("BayesPrism")), n.cores, smoke))

# ---- load tutorial data ----
load(data_f)  # provides: bk.dat, sc.dat, cell.type.labels, cell.state.labels
stopifnot(exists("bk.dat"), exists("sc.dat"),
          exists("cell.type.labels"), exists("cell.state.labels"))
cat(sprintf("bk.dat : %d bulk x %d genes\n", nrow(bk.dat), ncol(bk.dat)))
cat(sprintf("sc.dat : %d cells x %d genes\n", nrow(sc.dat), ncol(sc.dat)))
cat("cell types:\n"); print(table(cell.type.labels))

if (smoke) {
  # Subset BOTH inputs so the whole chain fits in login-node memory: the dense
  # 23,793 x 60,294 reference is ~11.5 GB and cleanup.genes peaks at ~2x that.
  n.bulk <- min(3, nrow(bk.dat))
  bk.dat <- bk.dat[seq_len(n.bulk), , drop = FALSE]
  set.seed(1)
  n.cell <- min(4000, nrow(sc.dat))
  keep   <- sort(sample.int(nrow(sc.dat), n.cell))
  sc.dat <- sc.dat[keep, , drop = FALSE]
  cell.type.labels  <- cell.type.labels[keep]
  cell.state.labels <- cell.state.labels[keep]
  gc()
  cat(sprintf("\n*** SMOKE mode: bulk=%d samples, reference=%d cells ***\n",
              n.bulk, n.cell))
}

# ---- [1] gene cleanup on the scRNA reference ----
# Drop ribosomal / mitochondrial / sex-chromosome / MALAT1 genes and genes
# expressed in <5 cells (tutorial defaults).
sc.filt <- cleanup.genes(input = sc.dat,
                         input.type = "count.matrix",
                         species = "hs",
                         gene.group = c("Rb", "Mrp", "other_Rb",
                                        "chrM", "MALAT1", "chrX", "chrY"),
                         exp.cells = 5)
cat(sprintf("after cleanup.genes        : %d genes\n", ncol(sc.filt)))

# ---- [2] restrict to protein-coding genes (recommended for TCGA gencode.v22) ----
sc.filt.pc <- select.gene.type(sc.filt, gene.type = "protein_coding")
cat(sprintf("after select.gene.type(pc) : %d genes\n", ncol(sc.filt.pc)))

# ---- [3] build the prism object ----
# key = the malignant cell type, handled separately (per-sample) by BayesPrism.
myPrism <- new.prism(reference = sc.filt.pc,
                     mixture = bk.dat,
                     input.type = "count.matrix",
                     cell.type.labels = cell.type.labels,
                     cell.state.labels = cell.state.labels,
                     key = "tumor",
                     outlier.cut = 0.01,
                     outlier.fraction = 0.1)

# ---- [4] run deconvolution (the long pole) ----
t0 <- Sys.time()
bp.res <- run.prism(prism = myPrism, n.cores = n.cores)
cat(sprintf("run.prism wall time: %.2f min\n",
            as.numeric(difftime(Sys.time(), t0, units = "mins"))))

# ---- [5] extract + report cell-type fractions ----
theta <- get.fraction(bp = bp.res, which.theta = "final", state.or.type = "type")
cat(sprintf("\n=== mean cell-type fraction across %d GBM bulk sample(s) ===\n",
            nrow(theta)))
print(round(sort(colMeans(theta), decreasing = TRUE), 4))

# sanity: malignant cells should dominate (per tutorial_deconvolution.pdf)
top <- names(which.max(colMeans(theta)))
cat(sprintf("\nDominant cell type: %s  (expected: tumor)\n", top))
if (!identical(top, "tumor"))
  cat("WARNING: tumor is not the dominant fraction — check install/replication.\n")

# ---- save (skip the heavy object in smoke mode) ----
frac_csv <- file.path(out_dir, if (smoke) "gbm_fractions_smoke.csv"
                               else "gbm_celltype_fractions.csv")
write.csv(theta, frac_csv)
cat(sprintf("\nWrote: %s\n", frac_csv))
if (!smoke) {
  rds <- file.path(out_dir, "bp_result.rds")
  saveRDS(bp.res, rds)
  cat(sprintf("Wrote: %s\n", rds))
}
cat("Replication complete.\n")
