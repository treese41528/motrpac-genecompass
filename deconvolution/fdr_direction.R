suppressPackageStartupMessages({ library(IHW) })
# fdr_direction.R -- in which direction do low-library blocks move the pooled IHW fit for the other blocks?
# It refits IHW on the dose p (P_fisher) of a DE folder twice: over every block of the folder's manifest, and
# over the blocks whose median library is at least a threshold. Within the retained blocks it counts the calls
# under each fit and the calls gained and lost when the low-library blocks are removed, and prints those
# counts and the direction they show. The per-block limma fits are not redone, and nothing is written.
#
# Run from the repo root:  Rscript deconvolution/fdr_direction.R [DE_DIR] [SEED] [MIN_MEDIAN_LIBSIZE]
#   DE_DIR              default <deconvolution.genecompass_input_dir>/pseudobulk_de of the active config
#                       (PIPELINE_PROFILE honoured)
#   SEED                IHW fold seed; default deconvolution.de_seed, else 1 (IHW's own default)
#   MIN_MEDIAN_LIBSIZE  blocks kept in the second fit; default 50, the value of the July 2026 analysis
# A folder made under the analysable-block rule (median library >= 1,000) lists only such blocks in its
# manifest, so a threshold of 1,000 or less keeps all of them and the two fits are one. To ask the question
# of the current default, point DE_DIR at the no-rule sensitivity folder (pseudobulk_de_minlib0) with a
# threshold of 1000.
# IHW is fitted as R/run_pseudobulk_de.R fits it: tissue as a nominal covariate (STEP3_PREREG.md 6.1).
source(file.path(Sys.getenv("PIPELINE_ROOT", "."), "deconvolution", "R", "config.R"))
.a    <- commandArgs(trailingOnly = TRUE)
ALPHA <- 0.05
BASE  <- if (length(.a) >= 1) .a[1] else file.path(cfg_value("CFG_GENECOMPASS_INPUT_DIR"), "pseudobulk_de")
SEED  <- as.integer(if (length(.a) >= 2) .a[2] else cfg_value("CFG_DE_SEED", default = "1"))
MINLIB <- as.numeric(if (length(.a) >= 3) .a[3] else "50")
if (is.na(SEED) || is.na(MINLIB)) stop("SEED must be an integer and MIN_MEDIAN_LIBSIZE a number")
cat(sprintf("DE folder: %s\nIHW seed:  %d\nblocks kept in the second fit: median library >= %g\n", BASE, SEED, MINLIB))
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

# IHW with the DE's settings; BH over the same p-values if IHW fails, and the output says which was used.
fit_method <- character(0)
fit <- function(sub, label) {
  r <- tryCatch({ set.seed(SEED); ihw(sub$P, factor(sub$tissue), alpha=ALPHA, covariate_type="nominal", seed=SEED) },
                error=function(e) NULL)
  fit_method[label] <<- if (is.null(r)) "BH (IHW failed)" else "IHW ~ tissue (nominal)"
  if (is.null(r)) p.adjust(sub$P,"BH") else adj_pvalues(r)
}

n_all  <- length(unique(G$key))
G$FDR_all <- fit(G, "all")                             # pooled over every block of the folder
keep <- summ$key[!is.na(summ$median_libsize) & summ$median_libsize >= MINLIB]
sub  <- G[G$key %in% keep, ]
n_kept <- length(unique(sub$key))
sub$FDR_kept <- fit(sub, "kept")                       # pooled over the retained blocks only

sig_all  <- sub$FDR_all  < ALPHA
sig_kept <- sub$FDR_kept < ALPHA
gained <- sum(sig_kept & !sig_all)
lost   <- sum(!sig_kept & sig_all)
num <- function(x) format(x, big.mark=",")
cat(sprintf("\nblocks with a dose p: %d (%s tests); retained: %d (%s tests); removed: %d\n",
            n_all, num(nrow(G)), n_kept, num(nrow(sub)), n_all - n_kept))
cat(sprintf("fits: all blocks = %s; retained blocks = %s\n", fit_method["all"], fit_method["kept"]))
cat(sprintf("\nWithin the %d retained blocks, at FDR < %g:\n", n_kept, ALPHA))
cat(sprintf("  significant under the fit over all %d blocks       : %s\n", n_all,  num(sum(sig_all))))
cat(sprintf("  significant under the fit over the %d retained only : %s\n", n_kept, num(sum(sig_kept))))
cat(sprintf("  gained when the %d low-library blocks are removed   : %s\n", n_all - n_kept, num(gained)))
cat(sprintf("  lost   when the %d low-library blocks are removed   : %s\n", n_all - n_kept, num(lost)))

# The conclusion is computed from the counts above, not assumed.
net <- sum(sig_kept) - sum(sig_all)
if (n_kept == n_all) {
  cat(sprintf("\n=> no block has a median library under %g: the two fits use the same tests, so there is nothing to compare.\n", MINLIB))
} else if (gained == 0 && lost == 0) {
  cat("\n=> removing the low-library blocks changes no call in the retained blocks.\n")
} else if (net > 0) {
  cat(sprintf("\n=> with the low-library blocks in the fit, the retained blocks have %s fewer calls (%s gained, %s lost on removal):\n   in this folder their presence makes the pooled fit more conservative for the retained blocks.\n",
              num(net), num(gained), num(lost)))
} else if (net < 0) {
  cat(sprintf("\n=> with the low-library blocks in the fit, the retained blocks have %s more calls (%s gained, %s lost on removal):\n   in this folder their presence makes the pooled fit less conservative for the retained blocks.\n",
              num(-net), num(gained), num(lost)))
} else {
  cat(sprintf("\n=> the number of calls in the retained blocks is unchanged, but %s calls are exchanged (%s gained, %s lost).\n",
              num(gained + lost), num(gained), num(lost)))
}
