#!/usr/bin/env Rscript
# de_technical_covariate_robustness.R -- Aim 2 hardening: is the per-cell-type exercise (dose)
# signal robust to RNA-quality technical covariates? The main DE (run_pseudobulk_de.R) omitted
# RIN / % globin because motrpac_sample_pheno.tsv lacked them -- but they ARE in the local
# MotrpacRatTraining6moData::TRNSCRPT_META (RIN, pct_globin, per viallabel). For every HOTSPOT
# block we re-fit the ordinal dose slope on log2-CPM of the BayesPrism Z with and without
# {RIN + pct_globin} as covariates and compare the per-gene dose statistic. High agreement =
# the exercise signal is not a technical-quality artifact.
#
# Out: data/deconvolution/genecompass_input/pseudobulk_de/rin_globin_robustness.tsv
suppressWarnings(suppressMessages({ library(limma); library(edgeR); library(data.table) }))

ROOT     <- Sys.getenv("PIPELINE_ROOT", ".")
RESULTS  <- file.path(ROOT, "data/deconvolution/results/motrpac")
BULK     <- file.path(ROOT, "data/deconvolution/motrpac_bulk")
PHENO    <- file.path(ROOT, "deconvolution/reference/motrpac_sample_pheno.tsv")
HOT      <- file.path(ROOT, "data/deconvolution/genecompass_input/pseudobulk_de/de_hotspots.tsv")
OUT      <- file.path(ROOT, "data/deconvolution/genecompass_input/pseudobulk_de/rin_globin_robustness.tsv")
META_RDA <- "/depot/reese18/apps/MotrpacRatTraining6moData/data/TRNSCRPT_META.rda"

# Shared cell-type -> filename contract (see deconvolution/R/celltype_names.R).
source(file.path(local({
  a <- commandArgs(trailingOnly = FALSE)
  f <- sub("^--file=", "", a[grep("^--file=", a)])
  if (length(f)) dirname(normalizePath(f[1])) else "."
}), "celltype_names.R"))

WEEK_NUM <- c("control"=0,"1w"=1,"2w"=2,"4w"=4,"8w"=8); WEEK_LV <- names(WEEK_NUM)
MIN_SAMPLES_BLOCK <- 12; MIN_GENES <- 50

ph <- read.delim(PHENO, stringsAsFactors=FALSE, colClasses="character")
ph <- ph[!duplicated(ph$viallabel), ]; rownames(ph) <- ph$viallabel

e <- new.env(); load(META_RDA, envir=e); tm <- as.data.frame(get(ls(e)[1], e))
tm$viallabel <- as.character(tm$viallabel); tm <- tm[!duplicated(tm$viallabel), ]
RIN <- setNames(suppressWarnings(as.numeric(tm$RIN)),        tm$viallabel)
GLO <- setNames(suppressWarnings(as.numeric(tm$pct_globin)), tm$viallabel)
cat(sprintf("TRNSCRPT_META: %d samples; RIN non-NA=%d, pct_globin non-NA=%d\n",
            nrow(tm), sum(!is.na(RIN)), sum(!is.na(GLO))))

hs <- read.delim(HOT, stringsAsFactors=FALSE); hs <- hs[toupper(hs$is_hotspot)=="TRUE", ]
rows <- list()
for (i in seq_len(nrow(hs))) {
  TIS <- hs$tissue[i]; ct <- hs$cell_type[i]
  bs <- file.path(BULK, TIS, "bulk_samples.tsv"); csv <- file.path(RESULTS, TIS, "pred_z", paste0("predz__", safe(ct), ".csv"))
  base_row <- data.table(tissue=TIS, cell_type=ct)
  if (!file.exists(bs) || !file.exists(csv)) { rows[[length(rows)+1]] <- cbind(base_row, status="no input"); next }
  via <- trimws(readLines(bs)); via <- via[via != ""]
  Z <- as.matrix(read.csv(csv, row.names=1, check.names=FALSE))              # samples(mix1..N) x genes
  m <- data.table(mix=paste0("mix", seq_along(via)), via=via,
                  sex=ph[via,"sex"], group=ph[via,"group"], rin=RIN[via], glo=GLO[via])
  m[, week := WEEK_NUM[group]]
  ok  <- !is.na(m$week) & !is.na(m$sex) & m$group %in% WEEK_LV
  ok2 <- ok & !is.na(m$rin) & !is.na(m$glo)
  if (sum(ok2) < MIN_SAMPLES_BLOCK) { rows[[length(rows)+1]] <- cbind(base_row, n_with_rin=sum(ok2), status="too few w/ RIN"); next }
  m <- m[ok2]; Zb <- Z[m$mix, , drop=FALSE]
  counts <- t(Zb); counts[counts < 0 | is.na(counts)] <- 0
  counts <- counts[rowSums(counts > 0) >= 1, , drop=FALSE]
  if (nrow(counts) < MIN_GENES) { rows[[length(rows)+1]] <- cbind(base_row, status="too few genes"); next }
  logcpm <- edgeR::cpm(DGEList(counts=counts), log=TRUE, prior.count=1)
  sexf <- factor(m$sex); weekn <- as.numeric(m$week); two <- nlevels(sexf) >= 2
  rin <- m$rin; glo <- m$glo
  base_des <- if (two) model.matrix(~ sexf + weekn)             else model.matrix(~ weekn)
  cov_des  <- if (two) model.matrix(~ sexf + weekn + rin + glo) else model.matrix(~ weekn + rin + glo)
  fb <- eBayes(lmFit(logcpm, base_des), trend=TRUE, robust=TRUE)
  fc <- eBayes(lmFit(logcpm, cov_des),  trend=TRUE, robust=TRUE)
  tb <- topTable(fb, coef="weekn", number=Inf, sort.by="none")
  tc <- topTable(fc, coef="weekn", number=Inf, sort.by="none")
  rho_t <- suppressWarnings(cor(tb$t, tc$t, method="spearman", use="complete.obs"))
  r_lfc <- suppressWarnings(cor(tb$logFC, tc$logFC, use="complete.obs"))
  sb <- tb$adj.P.Val < 0.05; sc <- tc$adj.P.Val < 0.05
  nb <- sum(sb, na.rm=TRUE); ncv <- sum(sc, na.rm=TRUE); nboth <- sum(sb & sc, na.rm=TRUE)
  retain <- if (nb > 0) nboth / nb else NA_real_
  verdict <- if (!is.na(rho_t) && rho_t >= 0.9 && (is.na(retain) || retain >= 0.8)) "ROBUST" else "CHECK"
  rows[[length(rows)+1]] <- data.table(tissue=TIS, cell_type=ct, n=sum(ok), n_with_rin=nrow(m),
    mean_rin=round(mean(rin),2), mean_globin=round(mean(glo),3),
    n_dose_sig_base=nb, n_dose_sig_cov=ncv, n_dose_sig_both=nboth, retain_frac=round(retain,3),
    spearman_dose_t=round(rho_t,3), pearson_dose_logfc=round(r_lfc,3), verdict=verdict, status="ok")
  cat(sprintf("  %-7s %-26s n=%d rin=%.2f glo=%.3f  dose-sig %d->%d (retain %.2f)  rho_t=%.3f  %s\n",
              TIS, ct, nrow(m), mean(rin), mean(glo), nb, ncv, ifelse(is.na(retain),NA,retain), rho_t, verdict))
}
res <- rbindlist(rows, fill=TRUE)
fwrite(res, OUT, sep="\t")
ok <- res[status=="ok"]
cat(sprintf("\nwrote %s\n%d hotspots scored: %d ROBUST, %d CHECK; median spearman(dose-t)=%.3f, median retain=%.2f\n",
            OUT, nrow(ok), sum(ok$verdict=="ROBUST"), sum(ok$verdict=="CHECK"),
            median(ok$spearman_dose_t, na.rm=TRUE), median(ok$retain_frac, na.rm=TRUE)))
