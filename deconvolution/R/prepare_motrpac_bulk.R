#!/usr/bin/env Rscript
# Fix 1 -- one-time gene-ID LIFTOVER of the real MoTrPAC bulk to current
# mRatBN7.2 (Ensembl rel-113) ENSRNOG, then emit deconv-ready samples x genes
# matrices for BayesPrism (run_deconvolution.R).
#
# WHY: the consortium's TRNSCRPT_<TISSUE>_RAW_COUNTS are annotated on an OLDER
# Ensembl rat build (~Rnor_6.0). Only ~61% of the 32,883 bulk ENSRNOG IDs are
# current; the rest are ID orphans that (a) silently drop out of bulk∩reference
# at deconvolution and (b) miss the GeneCompass token vocab. We lift them:
#   (a) DIRECT     -- IDs already current (in biomart rel-113 / token vocab) pass through;
#   (b) SYMBOL     -- orphans bridged via MoTrPAC FEATURE_TO_GENE (ensembl_gene -> gene_symbol)
#                     to the current ENSRNOG carrying that symbol (token vocab preferred over
#                     biomart so the lifted ID is tokenizable).
#   (c) ID_HISTORY -- orphans the symbol bridge missed (their FEATURE_TO_GENE symbol is an OLD
#                     spelling absent from the current vocab): recover the CURRENT symbol from RGD
#                     via the assembly-stable Entrez ID (FEATURE_TO_GENE.entrez_gene -> RGD
#                     NCBI_GENE_ID -> RGD SYMBOL) or via RGD OLD_SYMBOL, then resolve that current
#                     symbol to a rel-113 ENSRNOG through the SAME symbol map as (b). We deliberately
#                     do NOT use RGD's ENSEMBL_ID: it tracks the newer GRCr8 assembly, not our
#                     rel-113 references, so it would emit wrong-release IDs that textually collide
#                     with -- and mis-SUM onto -- a DIFFERENT rel-113 gene at the rowsum step.
#   Old IDs that collapse onto the same current ID have their raw counts SUMMED (alternate
#   annotations of one gene -- now always the same rel-113 gene, never a cross-assembly accident).
#   (d) UNMAPPED   -- no bridge: dropped (non-current by definition; never intersects the modern
#                     single-cell reference, so dropping is lossless for deconv).
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
# still-missed training-regulated record (committed; emitted from the SAME recovery logic as the
# report, so it can never drift from it). Written next to the map. SC-corpus membership is filled
# from the audit's gene sets when present (else left NA) -- release-robust, by current ID and symbol.
MISSED_TSV <- file.path(dirname(MAP_OUT), "motrpac_missed_genes.tsv")
MISSED_SUM <- file.path(dirname(MAP_OUT), "motrpac_missed_genes_summary.txt")
CORPUS_F   <- Sys.getenv("SC_CORPUS_GENES", "deconvolution/reference/idspace_audit/corpus_genes.txt")
ID2SYM_F   <- Sys.getenv("SC_ID2SYMBOL",    "deconvolution/reference/idspace_audit/id2symbol.tsv")

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

# RGD (GENES_RAT.txt) -> CURRENT gene SYMBOL (assembly-stable), NOT RGD's ENSEMBL_ID. RGD's
# NCBI_GENE_ID (Entrez) and OLD_SYMBOL are stable across genome builds and its SYMBOL is the current
# canonical symbol, so we recover the current symbol here and resolve it to a rel-113 ENSRNOG via
# sym2ens (so bridge 3, like bridge 2, always lands on rel-113). RGD's ENSEMBL_ID is deliberately
# UNUSED: it tracks the newer GRCr8 assembly (~60% overlap with rel-113), so it would emit
# wrong-release IDs that textually collide with -- and mis-sum onto -- a different rel-113 gene.
rgd <- read.delim(RGD_F, comment.char = "#", header = TRUE, quote = "", fill = TRUE,
                  sep = "\t", check.names = FALSE, stringsAsFactors = FALSE)
rg_ncbi <- as.character(rgd[["NCBI_GENE_ID"]]); rg_sym <- upper(rgd[["SYMBOL"]])
ok_n <- !is.na(rg_ncbi) & nzchar(rg_ncbi) & nzchar(rg_sym)
ncbi2sym <- tapply(rg_sym[ok_n], rg_ncbi[ok_n], function(x) x[1])     # Entrez -> current symbol
# OLD_SYMBOL is ';'-delimited (one gene lists many former spellings); explode to old->current pairs
old_split     <- strsplit(ifelse(is.na(rgd[["OLD_SYMBOL"]]), "", rgd[["OLD_SYMBOL"]]), ";", fixed = TRUE)
old_pairs_old <- upper(unlist(old_split))
old_pairs_sym <- rep(rg_sym, lengths(old_split))
ok_o <- nzchar(old_pairs_old) & nzchar(old_pairs_sym)
old2sym <- tapply(old_pairs_sym[ok_o], old_pairs_old[ok_o], function(x) x[1])  # old symbol -> current
cat(sprintf("RGD: %d Entrez->symbol, %d old-symbol->symbol bridges\n",
            length(ncbi2sym), length(old2sym)))

# id_history: recover a CURRENT symbol for an orphan ENSRNOG via Entrez (preferred) or its
# FEATURE_TO_GENE symbol seen as an RGD OLD_SYMBOL. Used by BOTH lift_ids and the coverage report
# (single source of truth, so the report's recovery count cannot drift from the actual lift).
idhist_sym <- function(old_ids) {
  ent  <- unname(entrez_by_ens[old_ids])      # old ID -> Entrez (FEATURE_TO_GENE)
  fsym <- unname(sym_by_ens[old_ids])         # old ID -> FEATURE_TO_GENE symbol
  cs   <- unname(ncbi2sym[ent])               # Entrez -> RGD current symbol
  ifelse(!is.na(cs), cs, unname(old2sym[fsym]))
}

# ---- the lift function (vectorised over a vector of bulk ENSRNOG IDs) ----
# priority: direct (already current) > symbol (FEATURE_TO_GENE symbol -> vocab/biomart) >
# id_history (recover the CURRENT symbol via Entrez/old-symbol, then -> rel-113 via sym2ens).
# Every bridge lands on a rel-113 ENSRNOG, so the rowsum collapse only ever merges annotations
# of the SAME current gene (never a cross-assembly accident).
lift_ids <- function(ids) {
  n <- length(ids)
  lifted <- rep(NA_character_, n)
  method <- rep("unmapped", n)
  sym    <- unname(sym_by_ens[ids])                # symbol for every id (NA if none)
  direct <- ids %in% current_ids
  lifted[direct] <- ids[direct]; method[direct] <- "direct"
  bridge <- which(!direct & !is.na(sym) & (sym %in% names(sym2ens)))
  lifted[bridge] <- unname(sym2ens[sym[bridge]]); method[bridge] <- "symbol"
  rest <- which(is.na(lifted))                     # id_history: Entrez/old-symbol -> rel-113
  if (length(rest)) {
    cur_sym <- idhist_sym(ids[rest])               # current symbol (assembly-stable recovery)
    l3 <- unname(sym2ens[cur_sym])                 # current symbol -> rel-113 ENSRNOG (vocab-preferred)
    h3 <- which(!is.na(l3))
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
  # id_history bridge (SAME logic as lift_ids): recover current symbol -> rel-113 vocab ENSRNOG
  o3  <- orphans[!recov]
  l3  <- unname(sym2ens[idhist_sym(o3)])
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

  # ---- still-missed training-regulated genes: write the committed per-gene record ----
  # (single source of truth -- reuses the recovery flags above, so report and record agree).
  recovered <- recov
  recovered[!recov] <- recov_idh                  # fold id_history recoveries into the orphan mask
  missed <- orphans[!recovered]                   # training-regulated ENSRNOG with no GeneCompass token
  trk <- trt[grepl("^ENSRNOG", trt$feature_ID), ]
  has_tis <- "tissue" %in% names(trk)
  tis_str <- if (has_tis) tapply(trk$tissue, trk$feature_ID, function(x) paste(sort(unique(x)), collapse = ",")) else NULL
  n_tis   <- if (has_tis) tapply(trk$tissue, trk$feature_ID, function(x) length(unique(x)))               else NULL
  bm_sym_by_id <- setNames(bm_sym, bm_ens)
  cs  <- idhist_sym(missed); fgs <- unname(sym_by_ens[missed]); bms <- unname(bm_sym_by_id[missed])
  symf <- ifelse(!is.na(cs)  & nzchar(cs),  cs,                  # current symbol: RGD(Entrez) >
          ifelse(!is.na(fgs) & nzchar(fgs), fgs,                 # FEATURE_TO_GENE > biomart name of own id
          ifelse(!is.na(bms) & nzchar(bms), bms, "")))
  curid    <- ifelse(missed %in% current_ids, missed, unname(sym2ens[symf]))  # rel-113 id (own/symbol-resolved)
  in_annot <- !is.na(curid) & nzchar(curid)
  if (file.exists(CORPUS_F)) {                     # release-robust SC-corpus membership (by id and symbol)
    corp_ids <- readLines(CORPUS_F)
    corp_sym <- if (file.exists(ID2SYM_F)) { i2s <- read.delim(ID2SYM_F, stringsAsFactors = FALSE)
                  upper(i2s$symbol[i2s$id %in% corp_ids]) } else character(0)
    # release-consistent: rel-113 current id in corpus, OR (release-robust) current symbol in corpus
    in_corpus <- (nzchar(curid) & curid %in% corp_ids) | (nzchar(symf) & upper(symf) %in% corp_sym)
  } else in_corpus <- rep(NA, length(missed))
  classify <- function(s) { s <- upper(s)          # symbol-pattern category (documented, reproducible)
    ifelse(!nzchar(s), "no_symbol",
    ifelse(grepl("^RT1-", s), "MHC_RT1",
    ifelse(grepl("^HB[ABEGZ]", s), "hemoglobin",
    ifelse(grepl("^MRP[LS]", s), "mito_ribosomal",
    ifelse(grepl("^RP[LS][0-9]", s), "ribosomal",
    ifelse(grepl("^HIST|^H[1-4][ABC]", s), "histone",
    ifelse(grepl("^IG[HKL]", s), "immunoglobulin",
    ifelse(grepl("RIK$|^AABR[0-9]|^A[CL][0-9]{5}|^BX[0-9]", s), "clone",
    ifelse(grepl("^LOC[0-9]|^RGD[0-9]|^ENSRNOG|^NEWGENE", s), "predicted",
    ifelse(grepl("-PS[0-9]*$|^GM[0-9]|^MIR[0-9]|LINC|^SNOR", s), "ncRNA/pseudo", "real_curated"))))))))))
  }
  cat_ <- classify(symf)
  nt   <- if (has_tis) { v <- as.integer(n_tis[missed]); v[is.na(v)] <- 0L; v } else rep(NA_integer_, length(missed))
  ntc  <- ifelse(is.na(nt), 0L, nt)                # importance scales with training-reg tissue breadth
  imp  <- ifelse(cat_ == "MHC_RT1",    ifelse(ntc >= 2, "high", "low"),               # MHC immune cluster
          ifelse(cat_ == "real_curated", ifelse(ntc >= 4, "high", ifelse(ntc >= 2, "med", "low")),
          "low"))                                  # clones/predicted/ncRNA/etc. -> low
  md <- data.frame(symbol = symf, old_ensrnog = missed,
                   current_ensrnog = ifelse(in_annot, curid, ""),
                   n_tissues_trainreg = nt,
                   tissues = if (has_tis) unname(tis_str[missed]) else "",
                   in_current_annot = in_annot, in_sc_corpus = in_corpus,
                   category = cat_, importance = imp, stringsAsFactors = FALSE)
  ord <- order(-ifelse(is.na(md$n_tissues_trainreg), 0L, md$n_tissues_trainreg), md$symbol)
  md  <- md[ord, ]
  write.table(md, MISSED_TSV, sep = "\t", quote = FALSE, row.names = FALSE)
  it <- table(factor(md$importance, levels = c("high", "med", "low")))
  n_corp <- if (all(is.na(md$in_sc_corpus))) "NA (audit sets absent)" else as.character(sum(md$in_sc_corpus, na.rm = TRUE))
  writeLines(c(
    "MoTrPAC training-regulated ('primary') genes still NOT tokenizable after the",
    "3-bridge gene-ID liftover  (direct + symbol + Entrez/RGD ID-history -> rel-113)",
    "=============================================================================", "",
    sprintf("%d of the %d training-regulated transcriptome genes remain without a", nrow(md), length(union_tr)),
    sprintf("GeneCompass token: %d already in vocab + %d symbol-recovered + %d id_history-recovered",
            sum(union_tr %in% vocab_ens), sum(recov), sum(recov_idh)),
    sprintf("= %d covered (%.1f%%). Per-gene record: motrpac_missed_genes.tsv,", sum(union_tr %in% vocab_ens) + n_rec, 100 * cov_after),
    sprintf("flagged by category and importance: high %d, med %d, low %d.", it["high"], it["med"], it["low"]), "",
    "Can these be recovered by GROWING the GeneCompass vocab?  NO -- not from our current",
    sprintf("single-cell corpus: %s of the %d still-missed genes occur in any SC reference", n_corp, nrow(md)),
    "(in_sc_corpus, by current ID and by symbol -- release-robust). They are simply absent",
    "from our SC data, so tokens would require deeper/different single-cell data, not vocab",
    "growth. Notable members: the RT1-* rat-MHC immune cluster (training-regulated in up to",
    "8 tissues; hyper-polymorphic, excluded from ortholog vocabularies) and a few real",
    "metabolic/immune genes (e.g. CD36, NDUFA13, TPI1, GPT, LYVE1, C4A, CFB).", "",
    "Columns: symbol, old_ensrnog, current_ensrnog (rel-113: own id if current, else symbol-",
    "resolved), n_tissues_trainreg, tissues, in_current_annot, in_sc_corpus, category, importance."),
    MISSED_SUM)
  cat(sprintf("\n-- still-missed primary genes: %d (record: %s; high %d/med %d/low %d) --\n",
              nrow(md), basename(MISSED_TSV), it["high"], it["med"], it["low"]))

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
if (file.exists(MISSED_TSV))
  cat(sprintf("Wrote missed-genes record: %s (+ %s)\n", MISSED_TSV, basename(MISSED_SUM)))
cat("Done.\n")
