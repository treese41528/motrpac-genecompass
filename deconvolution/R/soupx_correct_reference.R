# SoupX ambient-RNA correction of a nuclei reference (REFERENCE_RELABEL_PREREG.md §13; HIPPOC_REFS4_DIAGNOSIS.md).
#
#   Rscript soupx_correct_reference.R <ref_dir> <soup_source_dir> <out_dir> [merge ...]
#
#   ref_dir          reference to correct (reference_counts.mtx cells x genes, genes.tsv, cells_meta.tsv)
#   soup_source_dir  an earlier build of the same study whose extra barcodes (absent from ref_dir) are the soup:
#                    no raw/unfiltered matrices exist, so the dropped barcodes stand in for empty droplets
#   merge            "Type A,Type B=New type" relabels applied after correction (cell_state is kept, so the
#                    merged subtypes stay separate BayesPrism cell states)
#
# Per sample: soup profile = summed counts of that sample's dropped barcodes (setSoupProfile); contamination
# from autoEstCont with the cell types as clusters (forceAccept); adjustCounts(roundToInt = TRUE). A sample
# where autoEstCont fails gets the median rho of the others. Writes the corrected reference in the same
# format plus soupx_rho.tsv, soupx_per_type.tsv and provenance_soupx.json.
suppressPackageStartupMessages({ library(Matrix); library(SoupX); library(jsonlite) })
set.seed(0)
args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) >= 3)
ref_dir <- args[1]; soup_dir <- args[2]; out_dir <- args[3]; merges <- args[-(1:3)]
if (dir.exists(out_dir)) stop("FATAL: ", out_dir, " exists")

read_ref <- function(d) {
  genes <- readLines(file.path(d, "genes.tsv"))
  meta <- read.delim(file.path(d, "cells_meta.tsv"), colClasses = "character")
  t0 <- Sys.time(); X <- as(readMM(file.path(d, "reference_counts.mtx")), "CsparseMatrix")
  if (nrow(X) == length(genes) && ncol(X) == nrow(meta)) X <- t(X)
  stopifnot(nrow(X) == nrow(meta), ncol(X) == length(genes))
  message(sprintf("read %s: %d cells x %d genes (%.0fs)", d, nrow(X), ncol(X), as.numeric(Sys.time() - t0, units = "secs")))
  list(genes = genes, meta = meta, X = t(X))   # genes x cells, as SoupX wants
}
ref <- read_ref(ref_dir); src <- read_ref(soup_dir)
stopifnot(identical(ref$genes, src$genes))
key_ref <- paste(ref$meta$barcode, ref$meta$sample); key_src <- paste(src$meta$barcode, src$meta$sample)
stopifnot(!anyDuplicated(key_ref))
message(sprintf("reference nuclei absent from the source build: %d (no effect on the soup)", sum(!(key_ref %in% key_src))))
dropped <- !(key_src %in% key_ref)
message(sprintf("soup barcodes: %d of %d in the source build", sum(dropped), length(key_src)))

samples <- sort(unique(ref$meta$sample))
rows <- list(); fits <- list()
for (s in samples) {
  ci <- which(ref$meta$sample == s); di <- which(dropped & src$meta$sample == s)
  toc <- ref$X[, ci, drop = FALSE]; colnames(toc) <- key_ref[ci]; rownames(toc) <- ref$genes
  soup_counts <- Matrix::rowSums(src$X[, di, drop = FALSE])
  sc <- SoupChannel(toc, toc, calcSoupProfile = FALSE)
  sc <- setSoupProfile(sc, data.frame(est = soup_counts / sum(soup_counts), counts = soup_counts,
                                      row.names = ref$genes))
  cl <- setNames(ref$meta$cell_type[ci], colnames(toc))
  sc <- setClusters(sc, cl)
  est <- tryCatch(autoEstCont(sc, doPlot = FALSE, forceAccept = TRUE, verbose = FALSE),
                  error = function(e) { message("  autoEstCont failed for ", s, ": ", conditionMessage(e)); NULL })
  rho <- if (is.null(est)) NA_real_ else unique(est$metaData$rho)[1]
  fits[[s]] <- list(sc = sc, ci = ci, rho = rho)
  rows[[s]] <- data.frame(sample = s, n_cells = length(ci), n_soup_barcodes = length(di),
                          soup_umi = sum(soup_counts), median_cell_umi = median(Matrix::colSums(toc)),
                          rho_autoEstCont = rho)
  message(sprintf("  %s: %d nuclei, %d soup barcodes, rho %.3f", s, length(ci), length(di), rho))
}
rt <- do.call(rbind, rows)
med <- median(rt$rho_autoEstCont, na.rm = TRUE)
if (!is.finite(med)) stop("FATAL: autoEstCont failed for every sample")
rt$rho_used <- ifelse(is.na(rt$rho_autoEstCont), med, rt$rho_autoEstCont)
rt$rho_source <- ifelse(is.na(rt$rho_autoEstCont), "median of other samples", "autoEstCont")

# SOUPX_MODE=percell (REFERENCE_RELABEL_PREREG.md §13b): ambient as a fixed number of UMI per droplet.
# Per type, non-expressed genes = soup-abundant genes specific to another type; per nucleus, ambient UMI =
# counts on those genes / their soup fraction; per sample A_s = median; rho_c = min(0.9, A_s / UMI_c).
mode <- Sys.getenv("SOUPX_MODE", "global"); stopifnot(mode %in% c("global", "percell"))
if (mode == "percell") {
  types <- sort(unique(ref$meta$cell_type))
  cpm <- sapply(types, function(t) { v <- Matrix::rowSums(ref$X[, ref$meta$cell_type == t, drop = FALSE]); 1e6 * v / sum(v) })
  rownames(cpm) <- ref$genes
  soup_all <- Matrix::rowSums(src$X[, dropped, drop = FALSE]); soup_all <- soup_all / sum(soup_all)
  top <- order(-soup_all)[1:2000]
  G <- lapply(setNames(types, types), function(t) {
    oth <- apply(cpm[, setdiff(types, t), drop = FALSE], 1, max)
    keep <- top[oth[top] >= 20 * (cpm[top, t] + 1) | (oth[top] > 200 & cpm[top, t] < 0.05 * oth[top])]
    head(keep, 100)
  })
  message("non-expressed genes per type: ", paste(sprintf("%s %d", types, lengths(G)), collapse = "; "))
  umi <- Matrix::colSums(ref$X); a <- numeric(ncol(ref$X))
  for (t in types) { ci <- which(ref$meta$cell_type == t); g <- G[[t]]
    a[ci] <- Matrix::colSums(ref$X[g, ci, drop = FALSE]) / sum(soup_all[g]) }
  rho_cell <- numeric(ncol(ref$X))
  for (s in samples) { ci <- which(ref$meta$sample == s); A <- median(a[ci])
    rt$ambient_umi_per_droplet[rt$sample == s] <- A; rho_cell[ci] <- pmin(0.9, A / pmax(umi[ci], 1)) }
  rt$rho_source <- "percell: min(0.9, A_s / UMI)"
  ty <- aggregate(rho_cell, list(cell_type = ref$meta$cell_type), median); names(ty)[2] <- "median_rho_cell"
  print(ty, row.names = FALSE)
}
adjs <- lapply(samples, function(s) {
  f <- fits[[s]]
  r <- if (mode == "percell") setNames(rho_cell[f$ci], colnames(f$sc$toc)) else rt$rho_used[rt$sample == s]
  sc <- setContaminationFraction(f$sc, r, forceAccept = TRUE)
  adj <- adjustCounts(sc, roundToInt = TRUE, verbose = 0)
  stopifnot(identical(colnames(adj), colnames(f$sc$toc)))
  as(adj, "CsparseMatrix")
})
Y <- do.call(cbind, adjs)[, key_ref]   # back to the reference's cell order
stopifnot(identical(colnames(Y), key_ref), all(Y@x >= 0))
Y <- drop0(Y)

meta <- ref$meta; meta$cell_type_refs4 <- meta$cell_type
for (m in merges) {
  p <- strsplit(m, "=", fixed = TRUE)[[1]]; from <- trimws(strsplit(p[1], ",", fixed = TRUE)[[1]])
  miss <- setdiff(from, meta$cell_type); if (length(miss)) stop("FATAL: merge names absent: ", paste(miss, collapse = "; "))
  meta$cell_type[meta$cell_type %in% from] <- trimws(p[2])
}
stopifnot(all(tapply(meta$cell_type, meta$cell_state, function(x) length(unique(x))) == 1))

umi0 <- Matrix::colSums(ref$X); umi1 <- Matrix::colSums(Y)
pt <- aggregate(data.frame(umi_before = umi0, umi_after = umi1), list(cell_type_refs4 = meta$cell_type_refs4), sum)
pt$fraction_removed <- round(1 - pt$umi_after / pt$umi_before, 4)
pt$n_cells <- as.integer(table(meta$cell_type_refs4)[pt$cell_type_refs4])
pt$median_umi_after <- as.numeric(tapply(umi1, meta$cell_type_refs4, median)[pt$cell_type_refs4])
pt$cell_type <- meta$cell_type[match(pt$cell_type_refs4, meta$cell_type_refs4)]
message(sprintf("UMI removed overall: %.1f%%", 100 * (1 - sum(umi1) / sum(umi0))))
print(pt[order(-pt$fraction_removed), ], row.names = FALSE)
print(rt, row.names = FALSE)
if (any(umi1 == 0)) message(sprintf("WARNING: %d nuclei have no counts left (kept)", sum(umi1 == 0)))

dir.create(out_dir, recursive = TRUE)
writeMM(as(t(Y), "CsparseMatrix"), file.path(out_dir, "reference_counts.mtx"))   # cells x genes, integer
writeLines(ref$genes, file.path(out_dir, "genes.tsv"))
write.table(meta, file.path(out_dir, "cells_meta.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)
write.table(rt, file.path(out_dir, "soupx_rho.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)
write.table(pt, file.path(out_dir, "soupx_per_type.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)
tab <- sort(table(meta$cell_type), decreasing = TRUE)
writeLines(c(sprintf("cells=%d genes=%d cell_types=%d cell_states=%d", ncol(Y), nrow(Y), length(tab),
                     length(unique(meta$cell_state))), "", "cell_type",
             sprintf("%-35s %d", names(tab), as.integer(tab))), file.path(out_dir, "summary.txt"))
git <- function(...) tryCatch(system2("git", c(...), stdout = TRUE), error = function(e) NA)
write_json(list(script = "deconvolution/R/soupx_correct_reference.R", created = format(Sys.time(), "%Y-%m-%dT%H:%M:%S%z"),
                git_commit = git("rev-parse", "HEAD"), git_modified_tracked_files = git("diff", "--name-only", "HEAD"),
                ref_dir = ref_dir, soup_source_dir = soup_dir, merges = merges, mode = mode, soupx_version = as.character(packageVersion("SoupX")),
                r_version = R.version.string, n_soup_barcodes = sum(dropped), umi_removed = 1 - sum(umi1) / sum(umi0),
                host = Sys.info()[["nodename"]]), file.path(out_dir, "provenance_soupx.json"), auto_unbox = TRUE, pretty = TRUE)
message("wrote ", out_dir)
