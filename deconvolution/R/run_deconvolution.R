#!/usr/bin/env Rscript
# Generalized rat BayesPrism deconvolution (Stage 8.4) -- used for the V0/V1
# validation mixtures AND the real MoTrPAC bulk.
#
# Reads a reference dir (reference_counts.mtx cells x genes, genes.tsv,
# cells_meta.tsv) and a mixture dir (a *.mtx samples x genes + matching *_genes.tsv),
# applies the rat-specific gene cleanup (ribo/mito/hb -- since BayesPrism's
# cleanup.genes is human/mouse only; see deconvolution/reference/rat_exclude_genes.tsv),
# then new.prism(key=NULL) -> run.prism -> get.fraction. Writes estimated_fractions.csv.
#
# Args:  run_deconvolution.R <ref_dir> <mixture_dir> <out_dir> [mixture_mtx_basename]
#   mixture_mtx_basename defaults to "pseudobulk_counts" (validation); for real
#   bulk pass the basename of the <name>.mtx / <name>_genes.tsv pair.
# Env:   N_CORES (default 4)
# Invoke via run_deconvolution.sh on a compute node (run.prism is CPU-heavy).

suppressWarnings(suppressPackageStartupMessages({
  .libPaths(c(Sys.getenv("R_LIBS_USER"), .libPaths()))
  library(Matrix); library(BayesPrism)
}))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 3) stop("usage: run_deconvolution.R <ref_dir> <mixture_dir> <out_dir> [mtx_basename]")
ref_dir <- args[1]; mix_dir <- args[2]; out_dir <- args[3]
mtx_base <- if (length(args) >= 4) args[4] else "pseudobulk_counts"
genes_base <- sub("_counts$", "", mtx_base)   # pseudobulk_counts -> pseudobulk_genes
n.cores <- as.integer(Sys.getenv("N_CORES", unset = "4"))
excl_path <- Sys.getenv("RAT_EXCLUDE_GENES")
if (excl_path == "")   # standalone fallback when not launched via run_deconvolution.sh
  excl_path <- file.path(Sys.getenv("PROJECT_ROOT", unset = "."),
                         "deconvolution/reference/rat_exclude_genes.tsv")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
cat(sprintf("BayesPrism %s | n.cores=%d\nref=%s\nmix=%s\n",
            as.character(packageVersion("BayesPrism")), n.cores, ref_dir, mix_dir))

# ---- reference (cells x genes) ----
ref <- as.matrix(readMM(file.path(ref_dir, "reference_counts.mtx")))
genes <- readLines(file.path(ref_dir, "genes.tsv"))
meta <- read.delim(file.path(ref_dir, "cells_meta.tsv"), stringsAsFactors = FALSE)
colnames(ref) <- genes
rownames(ref) <- make.unique(as.character(meta$barcode))
cell.type.labels  <- as.character(meta$cell_type)
cell.state.labels <- as.character(meta$cell_state)
cat(sprintf("reference: %d cells x %d genes | %d types | %d states\n",
            nrow(ref), ncol(ref), length(unique(cell.type.labels)),
            length(unique(cell.state.labels))))

# ---- rat gene cleanup (replaces hs/mm-only cleanup.genes) ----
excl <- read.delim(excl_path, stringsAsFactors = FALSE)$feature_ID
n_excl <- sum(colnames(ref) %in% excl)
ref <- ref[, !colnames(ref) %in% excl, drop = FALSE]

# ---- sex-chromosome removal (replaces cleanup.genes chrX/chrY groups) ----
# chrX/chrY genes from biomart rat_gene_info.tsv; default-on so sex composition
# does not confound cell-type fractions in mixed-sex bulk. Gated by
# EXCLUDE_SEX_CHROMOSOMES; skips cleanly if the list/flag is absent.
sex_on <- Sys.getenv("EXCLUDE_SEX_CHROMOSOMES", unset = "1") %in% c("1", "true", "TRUE")
sex_path <- Sys.getenv("RAT_SEX_CHROM_GENES")
if (sex_on && nzchar(sex_path) && file.exists(sex_path)) {
  sexg <- toupper(sub("\\..*$", "", read.delim(sex_path, stringsAsFactors = FALSE)$feature_ID))
  drop_sex <- toupper(sub("\\..*$", "", colnames(ref))) %in% sexg
  cat(sprintf("sex-chromosome filter: removed %d chrX/chrY genes\n", sum(drop_sex)))
  ref <- ref[, !drop_sex, drop = FALSE]
} else {
  cat(sprintf("sex-chromosome filter: SKIPPED (EXCLUDE_SEX_CHROMOSOMES=%s, list=%s)\n",
              Sys.getenv("EXCLUDE_SEX_CHROMOSOMES", unset = "1"),
              if (nzchar(sex_path) && file.exists(sex_path)) "present" else "absent"))
}

# ---- protein-coding subset (replaces hs/mm-only select.gene.type="protein_coding") ----
# Rat protein-coding genes from biomart rat_gene_info.tsv (Gene type == protein_coding),
# the same biotype source GeneCompass Stage 2 uses. Gated by PROTEIN_CODING_ONLY so the
# pre-protein-coding behavior stays reproducible if the list/flag is absent.
pc_only <- Sys.getenv("PROTEIN_CODING_ONLY", unset = "1") %in% c("1", "true", "TRUE")
pc_path <- Sys.getenv("RAT_PROTEIN_CODING_GENES")
if (pc_only && nzchar(pc_path) && file.exists(pc_path)) {
  pc <- toupper(sub("\\..*$", "", read.delim(pc_path, stringsAsFactors = FALSE)$feature_ID))
  keep_pc <- toupper(sub("\\..*$", "", colnames(ref))) %in% pc
  cat(sprintf("protein-coding filter: %d/%d reference genes are protein_coding (%d non-coding/unknown dropped)\n",
              sum(keep_pc), length(keep_pc), sum(!keep_pc)))
  ref <- ref[, keep_pc, drop = FALSE]
} else {
  cat(sprintf("protein-coding filter: SKIPPED (PROTEIN_CODING_ONLY=%s, list=%s)\n",
              Sys.getenv("PROTEIN_CODING_ONLY", unset = "1"),
              if (nzchar(pc_path) && file.exists(pc_path)) "present" else "absent"))
}

keepg <- colSums(ref > 0) >= 3                      # species-agnostic low-expression filter
cat(sprintf("removed %d ribo/mito/hb genes; %d genes expressed in <3 cells; %d genes kept\n",
            n_excl, sum(!keepg), sum(keepg)))
ref <- ref[, keepg, drop = FALSE]

# ---- mixture (samples x genes) ----
mix <- as.matrix(readMM(file.path(mix_dir, paste0(mtx_base, ".mtx"))))
mgenes <- readLines(file.path(mix_dir, paste0(genes_base, "_genes.tsv")))
colnames(mix) <- mgenes
rownames(mix) <- paste0("mix", seq_len(nrow(mix)))
cat(sprintf("mixture: %d samples x %d genes\n", nrow(mix), ncol(mix)))

# ---- BayesPrism (key=NULL: normal tissue, no malignant reference) ----
prism <- new.prism(reference = ref, mixture = mix, input.type = "count.matrix",
                   cell.type.labels = cell.type.labels,
                   cell.state.labels = cell.state.labels,
                   key = NULL, outlier.cut = 0.01, outlier.fraction = 0.1)
t0 <- Sys.time()
bp <- run.prism(prism = prism, n.cores = n.cores)
cat(sprintf("run.prism wall time: %.2f min\n",
            as.numeric(difftime(Sys.time(), t0, units = "mins"))))

theta <- get.fraction(bp = bp, which.theta = "final", state.or.type = "type")
write.csv(theta, file.path(out_dir, "estimated_fractions.csv"))
saveRDS(bp, file.path(out_dir, "bp_result.rds"))
cat("\n=== mean estimated cell-type fraction ===\n")
print(round(sort(colMeans(theta), decreasing = TRUE), 4))
cat(sprintf("\nWrote: %s/estimated_fractions.csv\n", out_dir))
