# ---------------------------------------------------------------------------
# Do the LOW-MASS (rare-cell-type) blocks distort the POOLED FDR for the blocks
# we actually report?  Re-fit only Pass-2 (global IHW ~ tissue). No DE re-fit:
# the per-block limma models are independent; only the pooling is shared.
# ---------------------------------------------------------------------------
suppressPackageStartupMessages({ library(IHW) })
ALPHA <- 0.05
BASE  <- "data/deconvolution/genecompass_input/pseudobulk_de"

summ <- read.delim(file.path(BASE, "de_summary.tsv"),      check.names = FALSE)
man  <- read.delim(file.path(BASE, "celltype_files.tsv"),  check.names = FALSE)  # JOIN, never glob
key  <- function(t, c) paste(t, c, sep = "||")
summ$key <- key(summ$tissue, summ$cell_type)

rows <- vector("list", nrow(man))
for (i in seq_len(nrow(man))) {
  f <- file.path(BASE, man$file[i]); if (!file.exists(f)) next
  d <- read.delim(f, check.names = FALSE)
  rows[[i]] <- data.frame(key = key(man$tissue[i], man$cell_type[i]),
                          tissue = man$tissue[i], gene = d$gene,
                          P = d$P_fisher, FDR_cur = d$FDR_IHW, stringsAsFactors = FALSE)
}
G <- do.call(rbind, rows)
G <- G[is.finite(G$P), ]
G <- merge(G, summ[, c("key","mean_fraction","median_libsize","is_hotspot")], by = "key", sort = FALSE)
cat(sprintf("assembled %s tests over %d blocks\n\n", format(nrow(G), big.mark=","), length(unique(G$key))))

fit_ihw <- function(sub, lbl) {
  r <- tryCatch(ihw(sub$P, factor(sub$tissue), alpha = ALPHA), error = function(e) NULL)
  if (is.null(r)) { cat(lbl, ": IHW FAILED, BH fallback\n"); return(p.adjust(sub$P, "BH")) }
  adj_pvalues(r)
}

scen <- list(
  A_all185      = rep(TRUE, nrow(summ)),
  B_drop_kidney = !(summ$tissue=="KIDNEY" & summ$cell_type %in% c("Intercalated cells","Beta-intercalated cells")),
  C_libsize_50  = summ$median_libsize >= 50,
  D_abundant_P2 = summ$mean_fraction  >= 0.01
)

base_fdr <- NULL
cat(sprintf("%-16s %7s %11s %9s %9s %9s\n", "scenario","blocks","tests","n_sig","flips","hotspots"))
cat(strrep("-", 68), "\n")
for (nm in names(scen)) {
  keep_keys <- summ$key[scen[[nm]]]
  sub <- G[G$key %in% keep_keys, ]
  fdr <- fit_ihw(sub, nm)
  sub$FDR_new <- fdr
  if (nm == "A_all185") { base_fdr <- setNames(sub$FDR_new, paste(sub$key, sub$gene)); }
  # how many RETAINED genes flip significance vs the all-185 pooled fit?
  bk <- base_fdr[paste(sub$key, sub$gene)]
  flips <- sum((sub$FDR_new < ALPHA) != (bk < ALPHA), na.rm = TRUE)
  hs <- length(unique(sub$key[sub$is_hotspot]))
  cat(sprintf("%-16s %7d %11s %9s %9s %9d\n", nm, length(keep_keys),
              format(nrow(sub), big.mark=","), format(sum(sub$FDR_new < ALPHA), big.mark=","),
              format(flips, big.mark=","), hs))
}
cat("\nflips = genes in the RETAINED blocks whose significance call CHANGES when the\n")
cat("low-mass blocks are removed from the pooled calibration (vs the all-185 fit).\n")
