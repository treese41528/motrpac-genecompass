#!/usr/bin/env Rscript
# covariates_by_group.R -- are the technical covariates of the DE robustness reruns associated with the training
# groups? A descriptive check made after the reruns were read (NOT registered): if RIN or globin percentage
# differ between training groups, a model that adjusts for them tests something else than the headline model,
# and "calls lost under adjustment" has two readings (lost power, or signal shared with the covariate).
#
# Per tissue, sex and covariate (rin, pct_globin, pct_umi_dup, median_5_3_bias; MotrpacRatTraining6moData's
# TRNSCRPT_META), for the samples of the refs5 bulk: one-way ANOVA of the covariate on the five groups, as the DE
# model has them (control, 1w, 2w, 4w, 8w), with R-squared and F-test p; and the same for trained against control.
#
# Usage: covariates_by_group.R <TRNSCRPT_META.rda> <pheno_tsv> <bulk_root> <out_tsv>
a <- commandArgs(trailingOnly = TRUE)
e <- new.env(); load(a[1], envir = e); tm <- as.data.frame(get("TRNSCRPT_META", e))
tm$viallabel <- as.character(tm$viallabel); tm <- tm[!duplicated(tm$viallabel), ]
ph <- read.delim(a[2], stringsAsFactors = FALSE, colClasses = "character"); ph <- ph[!duplicated(ph$viallabel), ]
cols <- c(rin = "RIN", pct_globin = "pct_globin", pct_umi_dup = "pct_umi_dup", median_5_3_bias = "median_5_3_bias")
rows <- list()
for (tis in sort(list.dirs(a[3], recursive = FALSE, full.names = FALSE))) {
  f <- file.path(a[3], tis, "bulk_samples.tsv"); if (!file.exists(f)) next
  vl <- trimws(readLines(f)); vl <- vl[nzchar(vl) & vl != "viallabel"]      # one vial label per line, no header
  d <- merge(data.frame(viallabel = vl), ph[, c("viallabel", "sex", "group")], by = "viallabel")
  d <- merge(d, tm[, c("viallabel", cols)], by = "viallabel")
  for (sx in sort(unique(d$sex))) for (k in names(cols)) {
    x <- d[d$sex == sx, ]; y <- suppressWarnings(as.numeric(x[[cols[[k]]]])); g <- factor(x$group)
    ok <- is.finite(y); y <- y[ok]; g <- droplevels(g[ok])
    if (length(y) < 10 || nlevels(g) < 2 || sd(y) == 0) next
    fit <- summary(lm(y ~ g)); fs <- fit$fstatistic
    tr <- factor(ifelse(g == "control", "control", "trained"))
    p2 <- if (nlevels(tr) == 2) summary(lm(y ~ tr))$coefficients[2, 4] else NA
    rows[[length(rows) + 1]] <- data.frame(tissue = tis, sex = sx, covariate = k, n = length(y), mean = mean(y), sd = sd(y),
      r2_groups = fit$r.squared, p_groups = pf(fs[1], fs[2], fs[3], lower.tail = FALSE), p_trained_vs_control = p2)
  }
}
t <- do.call(rbind, rows); t$q_groups <- p.adjust(t$p_groups, "BH")
write.table(t, a[4], sep = "\t", quote = FALSE, row.names = FALSE)
cat(sprintf("%d tissue x sex x covariate tests; p < 0.05 in %d; BH q < 0.05 in %d; median R2 %.3f\n", nrow(t), sum(t$p_groups < 0.05),
            sum(t$q_groups < 0.05), median(t$r2_groups)))
for (k in names(cols)) { x <- t[t$covariate == k, ]
  cat(sprintf("  %-16s tests %d; p < 0.05 in %d; q < 0.05 in %d; median R2 %.3f; max R2 %.3f (%s %s)\n", k, nrow(x), sum(x$p_groups < 0.05),
              sum(x$q_groups < 0.05), median(x$r2_groups), max(x$r2_groups), x$tissue[which.max(x$r2_groups)], x$sex[which.max(x$r2_groups)])) }
print(t[t$tissue %in% c("BLOOD", "LUNG", "SKMVL", "SKMGN", "HEART"), c("tissue", "sex", "covariate", "n", "mean", "sd", "r2_groups", "p_groups")], digits = 3, row.names = FALSE)
