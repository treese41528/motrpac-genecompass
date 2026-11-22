# omnideconv_settings.R -- the settings of the cross-check runner (run_omnideconv.R) that used to be implicit,
# as plain functions so they can be tested without loading omnideconv (tests/r/test_omnideconv_settings.R).
#
#   parse_dwls_settings()   DWLS signature method and its two cut-offs, from DWLS_METHOD / DWLS_PVAL_CUTOFF /
#                           DWLS_DIFF_CUTOFF. Unset means the values every run before 2026-11-14 used without
#                           setting them: "mast", p 0.01, log2 fold change 0.5 (the defaults of omnideconv 0.1.1's
#                           build_model_dwls). The benchmark plan's setting is DWLS_METHOD=mast_optimized
#                           DWLS_PVAL_CUTOFF=0.05 (OMNIDECONV_BENCHMARK_PLAN.md; audit finding 5.6).
#   insert_guard()          the text edit behind the in-memory DWLS patch.
#   patch_dwls_builders()   applies it to both DWLS signature builders and says what it did.
#   package_versions()      installed versions of the packages a run can touch.

DWLS_METHODS <- c("mast", "mast_optimized", "seurat")
DWLS_DEFAULTS <- list(dwls_method = "mast", pval_cutoff = 0.01, diff_cutoff = 0.5)

parse_dwls_settings <- function(method = Sys.getenv("DWLS_METHOD"), pval = Sys.getenv("DWLS_PVAL_CUTOFF"),
                                diff = Sys.getenv("DWLS_DIFF_CUTOFF")) {
  num <- function(x, name, default, lo, hi) {
    if (!nzchar(x)) return(default)
    v <- suppressWarnings(as.numeric(x))
    if (is.na(v) || v <= lo || v > hi)
      stop(sprintf("%s must be a number in (%s, %s], got: %s", name, lo, hi, x))
    v
  }
  m <- if (nzchar(method)) method else DWLS_DEFAULTS$dwls_method
  if (!m %in% DWLS_METHODS)
    stop(sprintf("DWLS_METHOD must be one of %s, got: %s", paste(DWLS_METHODS, collapse = ", "), method))
  list(dwls_method = m,
       pval_cutoff = num(pval, "DWLS_PVAL_CUTOFF", DWLS_DEFAULTS$pval_cutoff, 0, 1),
       diff_cutoff = num(diff, "DWLS_DIFF_CUTOFF", DWLS_DEFAULTS$diff_cutoff, 0, Inf),
       dwls_method_set_by = if (nzchar(method)) "DWLS_METHOD" else "default",
       pval_cutoff_set_by = if (nzchar(pval)) "DWLS_PVAL_CUTOFF" else "default",
       diff_cutoff_set_by = if (nzchar(diff)) "DWLS_DIFF_CUTOFF" else "default")
}

# The guard. DWLS's differential-expression step leaves a cell type's table as the scalar 0 when fewer than two
# genes pass its fold-change cut-off; the signature builders then index that scalar and stop the whole run. The
# guard counts such a type as having no marker genes of its own (numberofGenes 0). The builders already skip a
# type with numberofGenes 0 when they pick genes and still give it a column of the signature, so the type stays
# deconvolvable. Every firing is printed and kept in options("dwls_guard_fired") for the provenance record.
DWLS_GUARD <- c(
  "        if (is.null(dim(cluster_lrTest.table)) || NCOL(cluster_lrTest.table) < 3) {",
  "            cat(sprintf(\"dwls guard: cell type '%s' has fewer than 2 candidate marker genes; counted as 0\\n\", i))",
  "            options(dwls_guard_fired = c(getOption(\"dwls_guard_fired\"), as.character(i)))",
  "            numberofGenes <- c(numberofGenes, 0)",
  "            next",
  "        }")

# Put `guard` before the one line of `lines` that matches `pattern`. More or fewer than one match is not
# patched: list(lines = <unchanged>, patched = FALSE, n_sites = <matches>).
insert_guard <- function(lines, pattern, guard = DWLS_GUARD) {
  hit <- grep(pattern, lines)
  if (length(hit) != 1L) return(list(lines = lines, patched = FALSE, n_sites = length(hit)))
  list(lines = append(lines, guard, after = hit - 1L), patched = TRUE, n_sites = 1L)
}

# function name in the DWLS namespace -> the line the guard goes before (the first use of the type's table)
DWLS_GUARD_SITES <- list(
  buildSignatureMatrixMAST          = "p\\.adjust\\(cluster_lrTest\\.table\\[\\[3\\]\\]",
  buildSignatureMatrixMASTOptimized = "as\\.numeric\\(cluster_lrTest\\.table\\$p_value\\)")

# Patch the installed DWLS in memory (R_libs/ is not touched). Returns, per builder, patched TRUE/FALSE and the
# number of candidate sites found.
patch_dwls_builders <- function(ns = "DWLS", sites = DWLS_GUARD_SITES) {
  options(dwls_guard_fired = character(0))
  out <- list()
  for (fn in names(sites)) {
    res <- tryCatch({
      g <- insert_guard(deparse(getFromNamespace(fn, ns)), sites[[fn]])
      if (g$patched) {
        f <- eval(parse(text = paste(g$lines, collapse = "\n")))
        environment(f) <- asNamespace(ns)
        assignInNamespace(fn, f, ns = ns)
      }
      list(patched = g$patched, n_sites = g$n_sites)
    }, error = function(e) list(patched = FALSE, n_sites = NA_integer_, error = conditionMessage(e)))
    out[[fn]] <- res
  }
  out
}

package_versions <- function(pkgs) {
  v <- lapply(pkgs, function(p) tryCatch(as.character(utils::packageVersion(p)), error = function(e) NA_character_))
  names(v) <- pkgs
  v
}
