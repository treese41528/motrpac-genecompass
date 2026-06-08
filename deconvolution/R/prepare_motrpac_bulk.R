#!/usr/bin/env Rscript
# Fix 1 -- one-time gene-ID LIFTOVER of the real MoTrPAC bulk to current
# mRatBN7.2 (Ensembl rel-113) ENSRNOG, then emit deconv-ready samples x genes
# matrices for BayesPrism (run_deconvolution.R).
#
# WHY: the consortium's TRNSCRPT_<TISSUE>_RAW_COUNTS are annotated on an OLDER
# Ensembl rat build (~Rnor_6.0). Only ~61% of the 32,883 bulk ENSRNOG IDs are
# current; the rest are ID orphans that (a) silently drop out of bulk∩reference
# at deconvolution and (b) miss the GeneCompass token vocab. We lift them:
#   (a) DIRECT  -- IDs already current (in biomart rel-113 / token vocab) pass through;
#   (b) SYMBOL  -- orphans are bridged via MoTrPAC FEATURE_TO_GENE
#                  (ensembl_gene -> gene_symbol) to the current ENSRNOG carrying
#                  that symbol (token vocab preferred over biomart so the lifted
#                  ID is tokenizable). Old IDs that collapse onto the same current
#                  ID have their raw counts SUMMED (alternate annotations of one gene).
#   (c) UNMAPPED-- no bridge: dropped (non-current by definition; never intersects
#                  the modern single-cell reference, so dropping is lossless for deconv).
#
# NOTE: in FEATURE_TO_GENE the `feature_ID` column is RefSeq (NP_/XM_...), NOT
# ENSRNOG -- the ENSRNOG is in `ensembl_gene`; the symbol bridge keys on that.
#
# Outputs per tissue under <out_dir>/<TISSUE>/:
#   bulk.mtx        samples x genes (MatrixMarket)   -- the pair run_deconvolution.R
#   bulk_genes.tsv  current ENSRNOG, column order    -- consumes (mtx_basename "bulk")
#   bulk_samples.tsv viallabels (PHENO join key)
# Plus two small AUDITABLE artifacts (committed):
#   <map_out>     per-gene liftover map (old_id, lifted_id, method, symbol)
#   <report_out> coverage report (direct/bridge/drop + primary-gene vocab coverage)
#
# Usage: prepare_motrpac_bulk.R <TISSUE|ALL> <out_dir> [map_out] [report_out]
#   env overrides: MOTRPAC_DATA_DIR, RAT_TOKEN_MAPPING, RAT_GENE_INFO

.libPaths(c(Sys.getenv("R_LIBS_USER"), .libPaths()))
suppressMessages(library(Matrix))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2)
  stop("usage: prepare_motrpac_bulk.R <TISSUE|ALL> <out_dir> [map_out] [report_out]")
which_tissue <- toupper(args[1])
out_dir      <- args[2]
MAP_OUT    <- if (length(args) >= 3) args[3] else "deconvolution/reference/motrpac_bulk_liftover.tsv"
REPORT_OUT <- if (length(args) >= 4) args[4] else "deconvolution/reference/motrpac_bulk_liftover_report.txt"

DATA_DIR <- Sys.getenv("MOTRPAC_DATA_DIR",  "data/motrpac/rat_training_6mo/data")
VOCAB_F  <- Sys.getenv("RAT_TOKEN_MAPPING", "data/training/ortholog_mappings/rat_token_mapping.tsv")
BIOMART_F<- Sys.getenv("RAT_GENE_INFO",     "data/references/biomart/rat_gene_info.tsv")
RGD_F    <- Sys.getenv("RAT_RGD_GENES",     "data/references/biomart/GENES_RAT.txt")

META  <- c("feature", "feature_ID", "tissue", "assay")
upper <- function(x) toupper(trimws(x))
ld <- function(path) { e <- new.env(); load(path, envir = e); get(ls(e)[1], envir = e) }

# ---- build the (tissue-independent) liftover lookups ----
cat("Loading token vocab + biomart + FEATURE_TO_GENE ...\n")
vocab <- read.delim(VOCAB_F, stringsAsFactors = FALSE)
vocab_ens <- vocab$rat_gene
vocab_sym <- upper(vocab$rat_symbol)
bm <- read.delim(BIOMART_F, stringsAsFactors = FALSE, check.names = FALSE)
bm_ens <- bm[["Gene stable ID"]]
bm_sym <- upper(bm[["Gene name"]])

# "is this ENSRNOG current?" = present in rel-113 biomart or the token vocab
current_ids <- unique(c(vocab_ens, bm_ens))

# symbol -> current ENSRNOG. vocab preferred (tokenizable); biomart fallback. For a
# symbol carried by >1 current ID, take the lexicographically-smallest ENSRNOG (stable).
v_ok <- nzchar(vocab_sym) & nzchar(vocab_ens)
b_ok <- nzchar(bm_sym)    & nzchar(bm_ens)
vmap <- tapply(vocab_ens[v_ok], vocab_sym[v_ok], function(x) sort(x)[1])
bmap <- tapply(bm_ens[b_ok],    bm_sym[b_ok],    function(x) sort(x)[1])
sym2ens <- bmap
sym2ens[names(vmap)] <- vmap                       # vocab overrides biomart
n_sym_multi <- sum(tapply(vocab_ens[v_ok], vocab_sym[v_ok],
                          function(x) length(unique(x))) > 1)

# old/any ENSRNOG -> current gene_symbol AND -> Entrez, via FEATURE_TO_GENE
fg <- ld(file.path(DATA_DIR, "FEATURE_TO_GENE.rda"))
ens_ok <- !is.na(fg$ensembl_gene) & nzchar(fg$ensembl_gene) & grepl("^ENSRNOG", fg$ensembl_gene)
fs <- data.frame(ens = fg$ensembl_gene, sym = upper(fg$gene_symbol),
                 entrez = as.character(fg$entrez_gene), stringsAsFactors = FALSE)[ens_ok, ]
sb <- fs[!is.na(fs$sym)    & nzchar(fs$sym), ];    sb <- sb[!duplicated(sb$ens), ]
eb <- fs[!is.na(fs$entrez) & nzchar(fs$entrez), ]; eb <- eb[!duplicated(eb$ens), ]
sym_by_ens    <- setNames(sb$sym,    sb$ens)       # ENSRNOG -> symbol (first)
entrez_by_ens <- setNames(eb$entrez, eb$ens)       # ENSRNOG -> Entrez (first)
rm(fg, fs, sb, eb); invisible(gc())

# RGD (GENES_RAT.txt): Entrez/symbol -> CURRENT mRatBN7.2 ENSRNOG -- the symbol-INDEPENDENT
# ID-history bridge. RGD's ENSEMBL_ID is the current annotation; NCBI_GENE_ID (Entrez) and
# OLD_SYMBOL are assembly-stable, so this recovers genes whose ENSRNOG/symbol changed across
# the Rnor_6.0 -> mRatBN7.2 rebuild (the gap the direct + symbol bridges leave).
rgd <- read.delim(RGD_F, comment.char = "#", header = TRUE, quote = "", fill = TRUE,
                  sep = "\t", check.names = FALSE, stringsAsFactors = FALSE)
rgd_ens <- vapply(rgd[["ENSEMBL_ID"]], function(s) {
  if (is.na(s)) return(NA_character_)
  m <- regmatches(s, regexpr("ENSRNOG[0-9]+", s)); if (length(m)) m else NA_character_ },
  character(1), USE.NAMES = FALSE)
rok <- !is.na(rgd_ens)
rg_ncbi <- as.character(rgd[["NCBI_GENE_ID"]]); rg_sym <- upper(rgd[["SYMBOL"]]); rg_old <- upper(rgd[["OLD_SYMBOL"]])
ncbi2ens    <- tapply(rgd_ens[rok & nzchar(rg_ncbi)], rg_ncbi[rok & nzchar(rg_ncbi)], function(x) x[1])
sym2ens_rgd <- tapply(rgd_ens[rok & nzchar(rg_sym)],  rg_sym[rok & nzchar(rg_sym)],   function(x) x[1])
old2ens_rgd <- tapply(rgd_ens[rok & nzchar(rg_old)],  rg_old[rok & nzchar(rg_old)],   function(x) x[1])
cat(sprintf("RGD: %d genes with an ENSRNOG (Entrez/symbol -> current-ID bridge)\n", sum(rok)))

# ---- the lift function (vectorised over a vector of bulk ENSRNOG IDs) ----
# priority: direct (already current) > symbol (FEATURE_TO_GENE symbol -> vocab/biomart) >
# id_history (Entrez or RGD current/old symbol -> current ENSRNOG, independent of symbol drift)
lift_ids <- function(ids) {
  n <- length(ids)
  lifted <- rep(NA_character_, n)
  method <- rep("unmapped", n)
  sym    <- unname(sym_by_ens[ids])                # symbol for every id (NA if none)
  direct <- ids %in% current_ids
  lifted[direct] <- ids[direct]; method[direct] <- "direct"
  bridge <- which(!direct & !is.na(sym) & (sym %in% names(sym2ens)))
  lifted[bridge] <- unname(sym2ens[sym[bridge]]); method[bridge] <- "symbol"
  rest <- which(is.na(lifted))                     # ID-history (Entrez/RGD) bridge (vectorised)
  if (length(rest)) {
    o   <- ids[rest]
    e   <- unname(ncbi2ens[unname(entrez_by_ens[o])])      # old ENSRNOG -> Entrez -> current
    so  <- unname(sym_by_ens[o])
    l3  <- ifelse(!is.na(e), e,
            ifelse(!is.na(sym2ens_rgd[so]), unname(sym2ens_rgd[so]), unname(old2ens_rgd[so])))
    h3  <- which(!is.na(l3))
    lifted[rest[h3]] <- l3[h3]; method[rest[h3]] <- "id_history"
  }
  data.frame(old_id = ids, lifted_id = lifted, method = method, symbol = sym,
             stringsAsFactors = FALSE)
}

# ---- per-tissue: read .rda, lift, collapse dup targets (sum), write mtx ----
process_tissue <- function(tis) {
  rc <- ld(file.path(DATA_DIR, sprintf("TRNSCRPT_%s_RAW_COUNTS.rda", tis)))
  ids   <- as.character(rc$feature_ID)
  scols <- setdiff(colnames(rc), META)
  lm <- lift_ids(ids)
  keep <- !is.na(lm$lifted_id)
  M <- as.matrix(rc[keep, scols, drop = FALSE]); storage.mode(M) <- "double"
  Mc <- rowsum(M, group = lm$lifted_id[keep])      # unique lifted ENSRNOG x samples
  genes <- rownames(Mc)
  out <- file.path(out_dir, tis); dir.create(out, recursive = TRUE, showWarnings = FALSE)
  writeMM(Matrix(t(Mc), sparse = TRUE), file.path(out, "bulk.mtx"))   # samples x genes
  writeLines(genes, file.path(out, "bulk_genes.tsv"))
  writeLines(scols, file.path(out, "bulk_samples.tsv"))
  data.frame(tissue = tis, in_genes = length(ids), kept = sum(keep),
             out_genes = length(genes), direct = sum(lm$method == "direct"),
             symbol = sum(lm$method == "symbol"), idhist = sum(lm$method == "id_history"),
             dropped = sum(lm$method == "unmapped"),
             collapsed = sum(keep) - length(genes), samples = length(scols),
             stringsAsFactors = FALSE)
}

# ---- select tissues ----
all_files <- list.files(DATA_DIR, pattern = "^TRNSCRPT_.*_RAW_COUNTS\\.rda$")
all_tis   <- sub("^TRNSCRPT_(.*)_RAW_COUNTS\\.rda$", "\\1", all_files)
tissues <- if (which_tissue == "ALL") sort(all_tis) else {
  if (!which_tissue %in% all_tis) stop("unknown tissue: ", which_tissue,
                                       " (have: ", paste(sort(all_tis), collapse = ", "), ")")
  which_tissue
}

# ---- write the auditable liftover map (canonical bulk gene set = first tissue) ----
canon_ids <- as.character(ld(file.path(DATA_DIR,
                  sprintf("TRNSCRPT_%s_RAW_COUNTS.rda", tissues[1])))$feature_ID)
lm_canon <- lift_ids(canon_ids)
dir.create(dirname(MAP_OUT), recursive = TRUE, showWarnings = FALSE)
write.table(lm_canon, MAP_OUT, sep = "\t", quote = FALSE, row.names = FALSE)
cat(sprintf("Wrote liftover map: %s (%d genes)\n", MAP_OUT, nrow(lm_canon)))

# ---- process tissues ----
cat(sprintf("\nProcessing %d tissue(s) -> %s\n", length(tissues), out_dir))
summ <- do.call(rbind, lapply(tissues, function(t) {
  s <- process_tissue(t)
  cat(sprintf("  %-7s in=%d kept=%d out=%d (direct=%d symbol=%d idhist=%d drop=%d collapse=%d) samples=%d\n",
              s$tissue, s$in_genes, s$kept, s$out_genes, s$direct, s$symbol, s$idhist,
              s$dropped, s$collapsed, s$samples))
  s
}))

# ---- coverage report (primary = training-regulated transcriptome genes) ----
sink(REPORT_OUT)
cat("MoTrPAC bulk gene-ID liftover -- coverage report\n")
cat("=================================================\n\n")
cat(sprintf("token vocab: %s (%d current ENSRNOG, %d symbols)\n",
            VOCAB_F, length(vocab_ens), sum(v_ok)))
cat(sprintf("biomart    : %s (%d current ENSRNOG)\n", BIOMART_F, length(bm_ens)))
cat(sprintf("current-ID universe (vocab ∪ biomart): %d\n", length(current_ids)))
cat(sprintf("symbols with >1 vocab ENSRNOG (resolved to smallest): %d\n\n", n_sym_multi))

cat("-- bulk gene liftover (canonical 32,883-gene set) --\n")
mt <- table(factor(lm_canon$method, levels = c("direct", "symbol", "id_history", "unmapped")))
cat(sprintf("  direct    : %5d (%.1f%%)\n", mt["direct"],     100 * mt["direct"]     / nrow(lm_canon)))
cat(sprintf("  symbol    : %5d (%.1f%%)\n", mt["symbol"],     100 * mt["symbol"]     / nrow(lm_canon)))
cat(sprintf("  id_history: %5d (%.1f%%)\n", mt["id_history"], 100 * mt["id_history"] / nrow(lm_canon)))
cat(sprintf("  unmapped  : %5d (%.1f%%)\n", mt["unmapped"],   100 * mt["unmapped"]   / nrow(lm_canon)))

trf <- file.path(DATA_DIR, "TRAINING_REGULATED_FEATURES.rda")
if (file.exists(trf)) {
  trt <- (function(){ tr <- ld(trf); tr[tr$assay == "TRNSCRPT", ] })()
  union_tr <- sort(unique(trt$feature_ID[grepl("^ENSRNOG", trt$feature_ID)]))
  orphans  <- setdiff(union_tr, vocab_ens)
  orph_sym <- unname(sym_by_ens[orphans])
  recov    <- !is.na(orph_sym) & (orph_sym %in% vocab_sym)        # symbol bridge
  # id_history bridge: orphans the symbol bridge missed whose Entrez/RGD current ID is in vocab
  o3  <- orphans[!recov]
  e3  <- unname(ncbi2ens[unname(entrez_by_ens[o3])])
  s3  <- unname(sym_by_ens[o3])
  l3  <- ifelse(!is.na(e3), e3,
          ifelse(!is.na(sym2ens_rgd[s3]), unname(sym2ens_rgd[s3]), unname(old2ens_rgd[s3])))
  recov_idh  <- !is.na(l3) & (l3 %in% vocab_ens)
  n_rec      <- sum(recov) + sum(recov_idh)
  cov_before <- mean(union_tr %in% vocab_ens)
  cov_after  <- (sum(union_tr %in% vocab_ens) + n_rec) / length(union_tr)
  cat(sprintf("\n-- primary genes (training-regulated transcriptome union = %d) --\n",
              length(union_tr)))
  cat(sprintf("  in vocab directly       : %5d  (%.1f%%)\n",
              sum(union_tr %in% vocab_ens), 100 * cov_before))
  cat(sprintf("  orphans (not in vocab)  : %5d\n", length(orphans)))
  cat(sprintf("  recovered via symbol    : %5d\n", sum(recov)))
  cat(sprintf("  recovered via id_history: %5d\n", sum(recov_idh)))
  cat(sprintf("  vocab coverage  BEFORE  : %.1f%%\n", 100 * cov_before))
  cat(sprintf("  vocab coverage  AFTER   : %.1f%%\n", 100 * cov_after))

  cat("\n-- spot checks (orphan recovered -> lifted ID present in output?) --\n")
  spot <- c("PLN", "EP300", "LTB", "HINT1", "PPP1R18")
  out_genes1 <- readLines(file.path(out_dir, tissues[1], "bulk_genes.tsv"))
  for (s in spot) {
    is_recov <- s %in% orph_sym[recov]
    tgt <- if (s %in% names(sym2ens)) sym2ens[[s]] else NA
    in_out <- !is.na(tgt) && tgt %in% out_genes1
    cat(sprintf("  %-8s recovered=%-5s lifted_id=%-20s in_output=%s\n",
                s, is_recov, ifelse(is.na(tgt), "-", tgt), in_out))
  }
}

cat("\n-- per-tissue summary --\n")
print(summ, row.names = FALSE)
sink()

cat(sprintf("\nWrote coverage report: %s\n", REPORT_OUT))
cat("Done.\n")
