#!/usr/bin/env Rscript
# run_gene_set_enrichment.R -- Aim 2 deliverable: pathway / gene-set enrichment
# over the per-(tissue x cell-type) differential-expression blocks.
#
# For every block file  pseudobulk_de/<TISSUE>/de__<cell_type>.tsv  we build a
# SIGNED DOSE statistic  sign(slope_week) * -log10(P_dose_comb)  per gene, map rat
# genes to human symbols (uppercased rat symbol via the Stage-3 ortholog table --
# 1:1 for orthologous genes), and run fgsea against MSigDB Hallmark + Reactome
# (Homo sapiens). Emits one enrichment table per block + a combined summary.
#
# Run via slurm/analysis/run_enrichment.slurm (installs fgsea/msigdbr to R_libs
# on first use). Rscript --no-init-file --no-site-file to dodge the site Rprofile.
suppressWarnings(suppressMessages({
  library(fgsea); library(msigdbr); library(data.table)
}))

# Shared cell-type -> filename contract.
.script_dir <- local({
  a <- commandArgs(trailingOnly = FALSE)
  f <- sub("^--file=", "", a[grep("^--file=", a)])
  if (length(f)) dirname(normalizePath(f[1])) else "."
})
source(file.path(.script_dir, "celltype_names.R"))

ROOT   <- Sys.getenv("PIPELINE_ROOT", unset = normalizePath("."))
DE_DIR <- file.path(ROOT, "data/deconvolution/genecompass_input/pseudobulk_de")
OUT    <- file.path(DE_DIR, "enrichment"); dir.create(OUT, showWarnings = FALSE)
MAP_F  <- file.path(ROOT, "data/training/ortholog_mappings/rat_token_mapping.tsv")
MIN_SET <- 10; MAX_SET <- 500

# ---- rat ENSRNOG -> human symbol (uppercased rat symbol; keep genes w/ human ortholog) ----
tm <- fread(MAP_F)
tm <- tm[nzchar(rat_symbol)]
tm[, hsym := toupper(rat_symbol)]
rat2hsym <- setNames(tm$hsym, tm$rat_gene)

# ---- MSigDB gene sets (human symbols): Hallmark (H) + Reactome (C2 CP:REACTOME) ----
# Schema-agnostic: msigdbr >=10 renamed gs_cat/gs_subcat -> gs_collection/gs_subcollection.
get_sets <- function() {
  m <- as.data.table(msigdbr(species = "Homo sapiens"))
  catcol <- if ("gs_collection" %in% names(m)) "gs_collection" else "gs_cat"
  subcol <- if ("gs_subcollection" %in% names(m)) "gs_subcollection" else "gs_subcat"
  gscol  <- if ("gene_symbol" %in% names(m)) "gene_symbol" else "gs_symbol"
  hallmark <- m[get(catcol) == "H"]
  reactome <- m[get(subcol) == "CP:REACTOME"]
  split_sets <- function(d) split(toupper(d[[gscol]]), d$gs_name)
  c(split_sets(hallmark), split_sets(reactome))
}
cat("loading MSigDB Hallmark + Reactome ...\n"); PATHWAYS <- get_sets()
cat(sprintf("  %d gene sets\n", length(PATHWAYS)))

# ---- per-block fgsea ----
# Drive off de_summary.tsv (status == "ok"), NOT a list.files() glob: a glob also picks up
# orphan de__ files left behind when a tissue is relabelled, and it recovers the cell type
# from the sanitized FILENAME rather than the real label. Both bit us -- the 2026-07-06 run
# scored 10 dead CORTEX/SKMVL blocks and reported cell types as "Pyramidal_neurons".
summ_f <- file.path(DE_DIR, "de_summary.tsv")
if (!file.exists(summ_f)) stop("de_summary.tsv not found -- run Stage 10 first: ", summ_f)
blocks <- fread(summ_f)[status == "ok", .(tissue, cell_type)]
blocks[, file := file.path(DE_DIR, tissue, sprintf("de__%s.tsv", safe(cell_type)))]
missing <- blocks[!file.exists(file)]
if (nrow(missing))
  stop(sprintf("%d block(s) in de_summary have no de__ file (first: %s / %s) -- re-run Stage 10",
               nrow(missing), missing$tissue[1], missing$cell_type[1]))
cat(sprintf("blocks: %d (from de_summary.tsv, status == ok)\n", nrow(blocks)))
purge_stale(OUT, "^gsea__.*\\.tsv$",
            keep = sprintf("gsea__%s__%s.tsv", blocks$tissue, safe(blocks$cell_type)))

all_rows <- list()
for (i in seq_len(nrow(blocks))) {
  tissue <- blocks$tissue[i]
  ct     <- blocks$cell_type[i]          # the REAL label, not the sanitized filename stem
  bf     <- blocks$file[i]
  d <- fread(bf)
  if (!all(c("gene","slope_week","P_dose_comb") %in% names(d))) next
  d <- d[is.finite(slope_week) & is.finite(P_dose_comb)]
  d[, hsym := rat2hsym[gene]]
  d <- d[!is.na(hsym) & nzchar(hsym)]
  d[, stat := sign(slope_week) * -log10(pmax(P_dose_comb, 1e-300))]
  d <- d[order(-abs(stat))][!duplicated(hsym)]          # 1 stat per human symbol
  ranks <- setNames(d$stat, d$hsym)
  ranks <- ranks[is.finite(ranks)]
  if (length(ranks) < 50) next
  res <- tryCatch(
    fgsea(pathways = PATHWAYS, stats = ranks, minSize = MIN_SET, maxSize = MAX_SET),
    error = function(e) NULL)
  if (is.null(res) || nrow(res) == 0) next
  res <- as.data.table(res)[order(padj)]
  res[, `:=`(tissue = tissue, cell_type = ct)]
  res[, leadingEdge := sapply(leadingEdge, function(x) paste(head(x, 20), collapse = ","))]
  fwrite(res, file.path(OUT, sprintf("gsea__%s__%s.tsv", tissue, safe(ct))), sep = "\t")
  all_rows[[length(all_rows)+1]] <- res[padj < 0.05,
      .(tissue, cell_type, pathway, NES, pval, padj, size)]
  cat(sprintf("  %-7s %-28s ranks=%d  sig(padj<.05)=%d\n", tissue, ct, length(ranks),
              sum(res$padj < 0.05, na.rm = TRUE)))
}
summ <- rbindlist(all_rows, fill = TRUE)
setorder(summ, tissue, cell_type, padj)
fwrite(summ, file.path(OUT, "enrichment_summary.tsv"), sep = "\t")
cat(sprintf("\nwrote %s  (%d significant pathway hits across %d blocks)\n",
            file.path(OUT, "enrichment_summary.tsv"), nrow(summ),
            length(unique(paste(summ$tissue, summ$cell_type)))))
