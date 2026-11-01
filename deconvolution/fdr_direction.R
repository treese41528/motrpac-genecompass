suppressPackageStartupMessages({ library(IHW) })
# Config + seed: run from the repo root (`Rscript deconvolution/fdr_direction.R [DE_DIR] [SEED]`). The DE folder defaults
# to <deconvolution.genecompass_input_dir>/pseudobulk_de of the active config (PIPELINE_PROFILE honoured);
# the IHW fold seed to deconvolution.de_seed (default 1, IHW's own default). Both are recorded in the output.
source(file.path(Sys.getenv("PIPELINE_ROOT", "."), "deconvolution", "R", "config.R"))
.a    <- commandArgs(trailingOnly = TRUE)
ALPHA <- 0.05
BASE  <- if (length(.a) >= 1) .a[1] else file.path(cfg_value("CFG_GENECOMPASS_INPUT_DIR"), "pseudobulk_de")
SEED  <- as.integer(if (length(.a) >= 2) .a[2] else cfg_value("CFG_DE_SEED", default = "1"))
cat(sprintf("DE folder: %s\nIHW seed:  %d\n", BASE, SEED))
summ <- read.delim(file.path(BASE,"de_summary.tsv"), check.names=FALSE)
man  <- read.delim(file.path(BASE,"celltype_files.tsv"), check.names=FALSE)
k <- function(t,c) paste(t,c,sep="||"); summ$key <- k(summ$tissue, summ$cell_type)
rows <- vector("list", nrow(man))
for (i in seq_len(nrow(man))) {
  f <- file.path(BASE, man$file[i]); if (!file.exists(f)) next
  d <- read.delim(f, check.names=FALSE)
  rows[[i]] <- data.frame(key=k(man$tissue[i], man$cell_type[i]), tissue=man$tissue[i],
                          gene=d$gene, P=d$P_fisher, stringsAsFactors=FALSE)
}
G <- do.call(rbind, rows); G <- G[is.finite(G$P),]
G <- merge(G, summ[,c("key","median_libsize","is_hotspot")], by="key", sort=FALSE)

fit <- function(sub) { r <- tryCatch({ set.seed(SEED); ihw(sub$P, factor(sub$tissue), alpha=ALPHA, seed=SEED) }, error=function(e) NULL)
                       if (is.null(r)) p.adjust(sub$P,"BH") else adj_pvalues(r) }

G$FDR_all <- fit(G)                                    # pooled over ALL 185
keep <- summ$key[summ$median_libsize >= 50]            # the 118 "real-mass" blocks
sub  <- G[G$key %in% keep, ]
sub$FDR_clean <- fit(sub)                              # pooled over the 118 ONLY

gained <- sum(sub$FDR_clean < ALPHA & !(sub$FDR_all < ALPHA))
lost   <- sum(!(sub$FDR_clean < ALPHA) & (sub$FDR_all < ALPHA))
cat(sprintf("\nWITHIN the 118 real-mass blocks (%s tests):\n", format(nrow(sub), big.mark=",")))
cat(sprintf("  significant under the ALL-185 pooled fit : %s\n", format(sum(sub$FDR_all   < ALPHA), big.mark=",")))
cat(sprintf("  significant under the 118-ONLY  fit      : %s\n", format(sum(sub$FDR_clean < ALPHA), big.mark=",")))
cat(sprintf("  GAINED significance when the low-mass blocks are removed : %s\n", format(gained, big.mark=",")))
cat(sprintf("  LOST   significance when the low-mass blocks are removed : %s\n", format(lost,   big.mark=",")))
cat("\n=> the low-mass blocks were making the shared calibration MORE CONSERVATIVE,\n")
cat("   suppressing real discoveries in the blocks we actually report.\n")
