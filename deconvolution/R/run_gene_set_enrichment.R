#!/usr/bin/env Rscript
# run_gene_set_enrichment.R -- Aim 2 deliverable: pathway / gene-set enrichment
# over the per-(tissue x cell-type) differential-expression blocks (Stage 13, step 4).
#
# For every analysed block  pseudobulk_de/<TISSUE>/de__<cell_type>.tsv  the genes are ranked by a signed dose
# statistic and tested with fgsea against MSigDB Hallmark + Reactome (Homo sapiens). One table per block,
# a combined summary of the hits, and a per-block table of what was ranked, dropped and tested.
#
# Four rules were fixed by docs/audit/remediation/STEP3_PREREG.md section 8.2 (audit finding 9.5):
#
#   1. Rat to human. A rat gene is carried to human through the ONE-TO-ONE pairs of the ortholog map that
#      Stage 12 uses (rat_to_human_mapping.pickle; the same 15,234 pairs are the human_ortholog column of
#      rat_token_mapping.tsv, which is what R can read; tests/test_stage13_step3.py checks the two agree).
#      One-to-one = Ensembl orthology type "ortholog_one2one" (orth_type_human; 13,883 pairs). A rat gene
#      without such a pair is dropped and counted. Gene sets are matched on the human Ensembl gene ID, so no
#      gene symbol is compared. Before: the rat symbol was upper-cased and matched to human symbols.
#   2. Ranking statistic  sign(slope_week) * -log10(P_fisher).  P_fisher is the dose p that defines the
#      Stage 10 headline (Fisher's combination of the per-sex training p-values; with one sex, that sex's p).
#      A gene with a missing slope or p is skipped and counted. Before: -log10 of the sex-adjusted omnibus p
#      (P_dose_comb), which is not the p the headline uses.
#   3. Gene sets. The table used is saved beside the results (gene_sets_used.tsv) with its md5, the msigdbr
#      version and the MSigDB release (gene_sets_used.meta.tsv). If a saved table is already there it is
#      read, its md5 is verified, and msigdbr's per-user cache is not consulted.
#   4. Tests that return no p-value are counted per block and in the printed summary.
#
# Environment:
#   DE_DIR          DE folder (pipeline/run_stage13.py sets it); else <genecompass_input_dir>/pseudobulk_de
#   ENRICH_OUT      output folder; default <DE_DIR>/enrichment. Set it to analyse a DE folder without writing
#                   into it.
#   GSEA_GENE_SETS  a gene_sets_used.tsv saved by an earlier run (its .meta.tsv beside it). It is verified
#                   and copied into the output folder, so two runs can be made on the identical gene sets.
#   GSEA_SEED       fgsea seed, default 1 (reset before every block)
#   GSEA_P_COLUMN   p column of the ranking statistic. The registered value, and the default, is P_fisher.
#                   Any other value is for a regression check only and is written into the provenance.
#
# Run via pipeline/run_stage13.py or deconvolution/R/run_gene_set_enrichment.sh.
# Rscript --no-init-file --no-site-file to dodge the site Rprofile.
suppressWarnings(suppressMessages({
  library(fgsea); library(data.table)
}))

# Shared cell-type -> filename contract.
.script_dir <- local({
  a <- commandArgs(trailingOnly = FALSE)
  f <- sub("^--file=", "", a[grep("^--file=", a)])
  if (length(f)) dirname(normalizePath(f[1])) else "."
})
source(file.path(.script_dir, "celltype_names.R"))
source(file.path(.script_dir, "provenance.R"))

source(file.path(.script_dir, "config.R"))
ROOT   <- Sys.getenv("PIPELINE_ROOT", unset = normalizePath("."))
# DE folder: $DE_DIR (pipeline/run_stage13.py sets it), else <genecompass_input_dir>/pseudobulk_de from the
# config (CFG_GENECOMPASS_INPUT_DIR, exported by the .sh wrapper). A config error stops the run.
DE_DIR <- Sys.getenv("DE_DIR")
if (!nzchar(DE_DIR)) DE_DIR <- file.path(cfg_value("CFG_GENECOMPASS_INPUT_DIR"), "pseudobulk_de")
cat(sprintf("DE folder: %s\n", DE_DIR))
OUT    <- Sys.getenv("ENRICH_OUT")
if (!nzchar(OUT)) OUT <- file.path(DE_DIR, "enrichment")
dir.create(OUT, recursive = TRUE, showWarnings = FALSE)
cat(sprintf("output folder: %s\n", OUT))
MAP_F  <- cfg_value("CFG_RAT_TOKEN_MAPPING", default = file.path(ROOT, "data/training/ortholog_mappings/rat_token_mapping.tsv"))
MIN_SET <- 10; MAX_SET <- 500
MIN_RANKED <- 50                                      # a block with fewer ranked genes is not tested
SEED    <- as.integer(Sys.getenv("GSEA_SEED", "1"))   # fgsea draws its sampling seeds from R's RNG
P_COL   <- Sys.getenv("GSEA_P_COLUMN", "P_fisher")    # registration 8.2: the dose p of the headline
if (P_COL != "P_fisher")
  cat(sprintf("NOTE: ranking on %s, not on the registered P_fisher (GSEA_P_COLUMN); regression check only\n", P_COL))

# ---- rat ENSRNOG -> human ENSG: the one-to-one pairs of the Stage 12 ortholog map (registration 8.2) ----
tm <- fread(MAP_F, colClasses = "character")
need <- c("rat_gene", "human_ortholog", "orth_type_human")
if (!all(need %in% names(tm)))
  stop("ortholog table lacks column(s) ", paste(setdiff(need, names(tm)), collapse = ", "), ": ", MAP_F)
tm <- tm[!is.na(human_ortholog) & nzchar(human_ortholog)]
pairs <- tm[orth_type_human == "ortholog_one2one", .(rat_gene, human_ensg = toupper(sub("\\..*$", "", human_ortholog)))]
if (anyDuplicated(pairs$rat_gene) || anyDuplicated(pairs$human_ensg))
  stop("the ortholog_one2one pairs of ", MAP_F, " are not one-to-one; the registered mapping cannot be applied")
rat2ensg <- setNames(pairs$human_ensg, pairs$rat_gene)
other_orth <- setdiff(tm$rat_gene, pairs$rat_gene)     # has a human ortholog in the map, but not one-to-one
cat(sprintf("ortholog map: %d rat genes with a human ortholog; %d one-to-one pairs used; %d not one-to-one, dropped\n",
            nrow(tm), nrow(pairs), length(other_orth)))

# ---- MSigDB gene sets: Hallmark (H) + Reactome (C2 CP:REACTOME), as human Ensembl gene IDs ----
# The table used is kept beside the results, so a rerun reads the same sets whatever is in the user's
# msigdbr cache (registration 8.2). Columns: gs_name, gs_collection, gs_subcollection, gene_symbol, ensembl_gene.
GS_F    <- file.path(OUT, "gene_sets_used.tsv")
GS_META <- file.path(OUT, "gene_sets_used.meta.tsv")
read_meta <- function(f) { m <- fread(f, colClasses = "character"); setNames(m$value, m$field) }
verify_saved <- function(tsv, meta_f) {
  if (!file.exists(meta_f)) stop("saved gene-set table has no md5 record beside it: ", tsv)
  meta <- read_meta(meta_f)
  got <- unname(tools::md5sum(tsv))
  if (!identical(got, unname(meta[["md5"]])))
    stop(sprintf("saved gene-set table %s has md5 %s but its record says %s; refusing to use it",
                 tsv, got, meta[["md5"]]))
  meta
}
build_sets <- function() {
  # Schema-agnostic: msigdbr >=10 renamed gs_cat/gs_subcat -> gs_collection/gs_subcollection.
  suppressWarnings(suppressMessages(library(msigdbr)))
  m <- as.data.table(msigdbr(species = "Homo sapiens"))
  catcol <- if ("gs_collection" %in% names(m)) "gs_collection" else "gs_cat"
  subcol <- if ("gs_subcollection" %in% names(m)) "gs_subcollection" else "gs_subcat"
  gscol  <- if ("gene_symbol" %in% names(m)) "gene_symbol" else "gs_symbol"
  enscol <- intersect(c("ensembl_gene", "human_ensembl_gene"), names(m))[1]
  if (is.na(enscol)) stop("msigdbr table has no Ensembl gene column; cannot match on human gene IDs")
  s <- m[get(catcol) == "H" | get(subcol) == "CP:REACTOME"]
  gs <- unique(data.table(gs_name = s$gs_name, gs_collection = s[[catcol]], gs_subcollection = s[[subcol]],
                          gene_symbol = s[[gscol]], ensembl_gene = toupper(sub("\\..*$", "", s[[enscol]]))))
  gs <- gs[!is.na(ensembl_gene) & nzchar(ensembl_gene)]
  setorder(gs, gs_name, ensembl_gene, gene_symbol)      # fixed row order, so the md5 identifies the content
  release <- if ("db_version" %in% names(s)) paste(sort(unique(s$db_version)), collapse = ",") else NA_character_
  list(gs = gs, release = release)
}
src <- Sys.getenv("GSEA_GENE_SETS")
if (nzchar(src) && normalizePath(src, mustWork = TRUE) != normalizePath(GS_F, mustWork = FALSE)) {
  # A table saved by an earlier run: verify it against its own record, then keep a copy with these results.
  verify_saved(src, sub("\\.tsv$", ".meta.tsv", src))
  file.copy(src, GS_F, overwrite = TRUE)
  file.copy(sub("\\.tsv$", ".meta.tsv", src), GS_META, overwrite = TRUE)
  cat(sprintf("gene sets: copied the saved table %s\n", src))
}
if (file.exists(GS_F)) {
  GS_INFO <- verify_saved(GS_F, GS_META)
  gs <- fread(GS_F, colClasses = "character")
  GS_SOURCE <- "saved table (md5 verified)"
} else {
  cat("loading MSigDB Hallmark + Reactome from msigdbr ...\n")
  b <- build_sets(); gs <- b$gs
  fwrite(gs, GS_F, sep = "\t")
  GS_INFO <- c(file = basename(GS_F), md5 = unname(tools::md5sum(GS_F)),
               msigdbr_version = as.character(packageVersion("msigdbr")), msigdb_release = b$release,
               collections = "H; C2 CP:REACTOME", species = "Homo sapiens",
               n_sets = as.character(uniqueN(gs$gs_name)), n_rows = as.character(nrow(gs)),
               n_ensembl_genes = as.character(uniqueN(gs$ensembl_gene)),
               created = format(Sys.time(), "%Y-%m-%dT%H:%M:%S%z"))
  fwrite(data.table(field = names(GS_INFO), value = unname(GS_INFO)), GS_META, sep = "\t")
  GS_SOURCE <- "msigdbr (table saved by this run)"
}
PATHWAYS <- lapply(split(gs$ensembl_gene, gs$gs_name), unique)
ens2sym  <- setNames(gs$gene_symbol, gs$ensembl_gene)[!duplicated(gs$ensembl_gene)]
SET_GENES <- unique(gs$ensembl_gene)
cat(sprintf("gene sets: %d sets, %d rows, md5 %s, msigdbr %s, MSigDB release %s [%s]\n",
            length(PATHWAYS), nrow(gs), GS_INFO[["md5"]], GS_INFO[["msigdbr_version"]],
            GS_INFO[["msigdb_release"]], GS_SOURCE))

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
block_rows <- vector("list", nrow(blocks))
for (i in seq_len(nrow(blocks))) {
  tissue <- blocks$tissue[i]
  ct     <- blocks$cell_type[i]          # the REAL label, not the sanitized filename stem
  bf     <- blocks$file[i]
  out_f  <- file.path(OUT, sprintf("gsea__%s__%s.tsv", tissue, safe(ct)))
  # One row per block, written whatever happens to it: a block that is not tested says why.
  br <- list(tissue = tissue, cell_type = ct, status = "ok", n_genes = NA_integer_,
             n_missing_stat = NA_integer_, n_no_pair = NA_integer_, n_no_pair_not_one2one = NA_integer_,
             n_ranked = NA_integer_, n_ranked_in_sets = NA_integer_, n_sets_tested = 0L,
             n_tests_no_pvalue = 0L, n_sig = 0L)
  done <- function(status) {             # record a block that is not tested, and drop any older table of it
    br$status <- status
    block_rows[[i]] <<- br
    if (file.exists(out_f)) file.remove(out_f)
    cat(sprintf("  %-7s %-28s NOT TESTED: %s\n", tissue, ct, status))
  }
  d <- fread(bf)
  br$n_genes <- nrow(d)
  if (!all(c("gene", "slope_week", P_COL) %in% names(d))) { done(sprintf("missing column (need gene, slope_week, %s)", P_COL)); next }
  # Registration 8.2: a gene with a missing slope or dose p is skipped and counted. P_fisher is missing for
  # every gene of a single-sex tissue in DE tables written before the Stage 10 repair (finding 7.1).
  has_stat <- is.finite(d$slope_week) & is.finite(d[[P_COL]])
  br$n_missing_stat <- sum(!has_stat)
  d <- d[has_stat]
  # Registration 8.2: rat gene -> human gene through the one-to-one pairs; no pair, no rank.
  d[, ensg := unname(rat2ensg[gene])]
  br$n_no_pair <- sum(is.na(d$ensg))
  br$n_no_pair_not_one2one <- sum(is.na(d$ensg) & d$gene %in% other_orth)
  d <- d[!is.na(ensg)]
  d[, stat := sign(slope_week) * -log10(pmax(get(P_COL), 1e-300))]
  d <- d[order(ensg)]                    # fixed input order, so tied statistics are ordered the same way every run
  ranks <- setNames(d$stat, d$ensg)
  ranks <- ranks[is.finite(ranks)]
  br$n_ranked <- length(ranks)
  br$n_ranked_in_sets <- sum(names(ranks) %in% SET_GENES)
  if (length(ranks) < MIN_RANKED) { done(sprintf("fewer than %d ranked genes (%d)", MIN_RANKED, length(ranks))); next }
  set.seed(SEED)                         # per block, so results don't depend on block order
  err <- NULL
  res <- tryCatch(
    fgsea(pathways = PATHWAYS, stats = ranks, minSize = MIN_SET, maxSize = MAX_SET),
    error = function(e) { err <<- conditionMessage(e); NULL })
  if (!is.null(err)) { done(paste("fgsea error:", err)); next }
  if (is.null(res) || nrow(res) == 0) { done("no gene set within the size limits"); next }
  res <- as.data.table(res)[order(padj)]
  res[, `:=`(tissue = tissue, cell_type = ct)]
  # Leading edge as human gene symbols (first 20), read from the gene-set table; ranks are Ensembl IDs.
  res[, leadingEdge := sapply(leadingEdge, function(x) {
    s <- unname(ens2sym[head(x, 20)]); paste(ifelse(is.na(s), head(x, 20), s), collapse = ",") })]
  fwrite(res, out_f, sep = "\t")
  br$n_sets_tested <- nrow(res)
  br$n_tests_no_pvalue <- sum(is.na(res$pval))   # registration 8.2: counted, not silent
  br$n_sig <- sum(res$padj < 0.05, na.rm = TRUE)
  block_rows[[i]] <- br
  all_rows[[length(all_rows)+1]] <- res[!is.na(padj) & padj < 0.05,
      .(tissue, cell_type, pathway, NES, pval, padj, size)]
  cat(sprintf("  %-7s %-28s ranks=%d  no pair=%d  missing stat=%d  sets=%d  no p=%d  sig(padj<.05)=%d\n",
              tissue, ct, length(ranks), br$n_no_pair, br$n_missing_stat, nrow(res),
              br$n_tests_no_pvalue, br$n_sig))
}
summ <- rbindlist(all_rows, fill = TRUE)
if (nrow(summ)) setorder(summ, tissue, cell_type, padj) else
  summ <- data.table(tissue = character(), cell_type = character(), pathway = character(),
                     NES = numeric(), pval = numeric(), padj = numeric(), size = integer())
fwrite(summ, file.path(OUT, "enrichment_summary.tsv"), sep = "\t")
B <- rbindlist(block_rows, fill = TRUE)
fwrite(B, file.path(OUT, "enrichment_blocks.tsv"), sep = "\t")

tested <- B[status == "ok"]
tot <- list(n_blocks = nrow(B), n_blocks_tested = nrow(tested), n_blocks_not_tested = nrow(B) - nrow(tested),
            n_tests = sum(tested$n_sets_tested), n_tests_no_pvalue = sum(tested$n_tests_no_pvalue),
            n_sig = nrow(summ), n_blocks_with_sig = sum(tested$n_sig > 0),
            n_genes_missing_stat = sum(B$n_missing_stat, na.rm = TRUE),
            n_genes_no_pair = sum(B$n_no_pair, na.rm = TRUE),
            n_genes_ranked = sum(tested$n_ranked))
write_provenance(OUT, inputs = c(summ_f, MAP_F, GS_F),
                 params = c(list(seed = SEED, min_set = MIN_SET, max_set = MAX_SET, min_ranked = MIN_RANKED,
                                 de_dir = DE_DIR, p_column = P_COL,
                                 statistic = sprintf("sign(slope_week) * -log10(%s)", P_COL),
                                 rat_to_human = "ortholog_one2one pairs of the Stage 12 ortholog map, matched on human Ensembl gene ID",
                                 n_one2one_pairs = nrow(pairs),
                                 n_pathways = length(PATHWAYS),
                                 gene_sets_file = basename(GS_F), gene_sets_md5 = GS_INFO[["md5"]],
                                 gene_sets_source = GS_SOURCE,
                                 msigdbr_gene_sets = GS_INFO[["msigdbr_version"]],
                                 msigdb_release = GS_INFO[["msigdb_release"]],
                                 fgsea = as.character(packageVersion("fgsea")),
                                 registration = "docs/audit/remediation/STEP3_PREREG.md section 8.2"),
                            tot))
cat(sprintf("\nwrote %s  (%d significant pathway hits across %d blocks)\n",
            file.path(OUT, "enrichment_summary.tsv"), nrow(summ), tot$n_blocks_with_sig))
cat(sprintf("blocks: %d listed, %d tested, %d not tested (see enrichment_blocks.tsv, column status)\n",
            tot$n_blocks, tot$n_blocks_tested, tot$n_blocks_not_tested))
if (tot$n_blocks_not_tested) print(B[status != "ok", .N, by = status])
cat(sprintf("genes: %d skipped for a missing slope or %s, %d dropped for having no one-to-one human ortholog, %d ranked (summed over blocks)\n",
            tot$n_genes_missing_stat, P_COL, tot$n_genes_no_pair, tot$n_genes_ranked))
cat(sprintf("tests: %d gene-set tests, %d returned no p-value (%.3f%%); these have no adjusted p and cannot be hits\n",
            tot$n_tests, tot$n_tests_no_pvalue, if (tot$n_tests) 100 * tot$n_tests_no_pvalue / tot$n_tests else 0))
cat(sprintf("gene sets: %s, md5 %s\n", GS_F, GS_INFO[["md5"]]))
