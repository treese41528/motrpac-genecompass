# Run by tests/test_omnideconv_runner.py:  Rscript tests/r/test_omnideconv_settings.R <repo root> <scratch dir>
# Tests deconvolution/R/omnideconv_settings.R and the provenance record the cross-check runner writes, without
# loading omnideconv. The last block needs the DWLS package and is skipped (and says so) where it is absent.
args <- commandArgs(trailingOnly = TRUE)
root <- args[1]; tmp <- args[2]
.libPaths(c(Sys.getenv("R_LIBS_USER"), .libPaths()))
source(file.path(root, "deconvolution/R/omnideconv_settings.R"))
source(file.path(root, "deconvolution/R/provenance.R"))
fails <- function(expr) inherits(tryCatch(expr, error = function(e) e), "error")

# ---- DWLS settings: unset = what the earlier runs used; the benchmark plan's values pass through ----
s <- parse_dwls_settings("", "", "")
stopifnot(identical(s$dwls_method, "mast"), s$pval_cutoff == 0.01, s$diff_cutoff == 0.5,
          s$dwls_method_set_by == "default", s$pval_cutoff_set_by == "default", s$diff_cutoff_set_by == "default")
s <- parse_dwls_settings("mast_optimized", "0.05", "")
stopifnot(identical(s$dwls_method, "mast_optimized"), s$pval_cutoff == 0.05, s$diff_cutoff == 0.5,
          s$dwls_method_set_by == "DWLS_METHOD", s$pval_cutoff_set_by == "DWLS_PVAL_CUTOFF")
stopifnot(fails(parse_dwls_settings("mastoptimized", "", "")), fails(parse_dwls_settings("", "five", "")),
          fails(parse_dwls_settings("", "0", "")), fails(parse_dwls_settings("", "1.5", "")),
          fails(parse_dwls_settings("", "", "-1")))
stopifnot(parse_dwls_settings("", "1", "")$pval_cutoff == 1)

# ---- the guard: one site is patched; none or several are left alone ----
optimized <- c("function (tables, ids) ", "{", "    numberofGenes <- c()", "    for (i in ids) {",
               "        cluster_lrTest.table <- tables[[i]]",
               "        pvals <- as.numeric(cluster_lrTest.table$p_value)",
               "        numberofGenes <- c(numberofGenes, length(pvals))", "    }", "    numberofGenes", "}")
mast <- sub("pvals <- as.numeric\\(cluster_lrTest.table\\$p_value\\)",
            "pvals <- stats::p.adjust(cluster_lrTest.table[[3]], method = \"fdr\")", optimized)
tables <- list(a = data.frame(gene = c("g1", "g2"), test.type = "hurdle", p_value = c(0.1, 0.2)), b = 0)
for (case in list(list(lines = optimized, site = DWLS_GUARD_SITES$buildSignatureMatrixMASTOptimized),
                  list(lines = mast, site = DWLS_GUARD_SITES$buildSignatureMatrixMAST))) {
  before <- eval(parse(text = paste(case$lines, collapse = "\n")))
  stopifnot(fails(before(tables, c("a", "b"))))                 # the unpatched builder stops on type b
  g <- insert_guard(case$lines, case$site)
  stopifnot(g$patched, g$n_sites == 1L, length(g$lines) == length(case$lines) + length(DWLS_GUARD))
  after <- eval(parse(text = paste(g$lines, collapse = "\n")))
  options(dwls_guard_fired = character(0))
  out <- utils::capture.output(n <- after(tables, c("a", "b")))
  stopifnot(identical(n, c(2, 0)), identical(getOption("dwls_guard_fired"), "b"), any(grepl("dwls guard", out)))
  stopifnot(identical(after(tables["a"], "a"), before(tables["a"], "a")))   # no effect on a normal type
}
g <- insert_guard(c("x <- 1", "y <- 2"), "p\\.adjust")
stopifnot(!g$patched, g$n_sites == 0L, identical(g$lines, c("x <- 1", "y <- 2")))
g <- insert_guard(c(optimized, optimized), DWLS_GUARD_SITES$buildSignatureMatrixMASTOptimized)
stopifnot(!g$patched, g$n_sites == 2L)

# ---- package versions: NA for a package that is not installed ----
v <- package_versions(c("stats", "no_such_package_xyz"))
stopifnot(is.character(v$stats), is.na(v$no_such_package_xyz))

# ---- the provenance record, with md5 of the inputs ----
f <- file.path(tmp, "input.txt"); writeLines(c("a", "b"), f)
Sys.setenv(PROJECT_ROOT = root)
p <- write_provenance(tmp, inputs = c(f, file.path(tmp, "missing.txt")), md5 = TRUE, name = "provenance_test.json",
                      params = list(methods = I("dwls"),
                                    method_settings = list(dwls = parse_dwls_settings("mast_optimized", "0.05", "")),
                                    reference_cap = list(max_cells_per_type = 2000L, seed = 1L),
                                    in_memory_patches = list(dwls = list(guard_fired_for = I(character(0))))))
stopifnot(!is.null(p), file.exists(p))
cat("input_md5", unname(tools::md5sum(f)), "\n")

# ---- the installed DWLS: each builder has exactly one guard site ----
if (requireNamespace("DWLS", quietly = TRUE)) {
  r <- patch_dwls_builders()
  stopifnot(identical(names(r), c("buildSignatureMatrixMAST", "buildSignatureMatrixMASTOptimized")))
  for (fn in names(r)) {
    stopifnot(isTRUE(r[[fn]]$patched), r[[fn]]$n_sites == 1L)
    stopifnot(any(grepl("dwls guard", deparse(getFromNamespace(fn, "DWLS")))))
  }
  cat("dwls_installed TRUE", as.character(utils::packageVersion("DWLS")), "\n")
} else {
  cat("dwls_installed FALSE\n")
}
cat("ALL OK\n")
