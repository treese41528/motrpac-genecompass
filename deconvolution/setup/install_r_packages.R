#!/usr/bin/env Rscript
# =============================================================================
# Environment-agnostic installer for the MoTrPAC GeneCompass deconvolution R stack.
# Reads setup/r_packages.yaml and installs every root package (CRAN / Bioconductor /
# GitHub / patched-GitHub / vendored), resolving the dependency tree with pak.
#
# This is the SHARED CORE: the containers (setup/Dockerfile, setup/Apptainer.def) and
# the bare-metal bootstrap (setup/install_r_env.sh) all just call this script. It makes
# NO assumptions about modules/conda/cluster -- the *system libraries* are the caller's
# job (the container apt-installs them; the bootstrap loads modules or prints what to
# install). To make that tractable everywhere, this script first prints the exact
# system-dependency commands for the host OS via pak::pkg_sysreqs().
#
# Reproducibility:
#   * CRAN  -> a DATED Posit Package Manager snapshot (CRAN_REPO env), so every CRAN
#             package resolves to its version on that date. Default = capture date.
#   * Bioc  -> pinned by the R version (R 4.4.x => Bioconductor 3.20 => DESeq2 1.46).
#   * GitHub-> pinned by commit SHA from the manifest.
#
# Env:
#   R_LIBS_USER   where to install (default: first writable .libPaths()).
#   R_MANIFEST    manifest path (default: r_packages.yaml next to this script).
#   PROJECT_ROOT  repo root, for the vendored BayesPrism (default: two dirs up).
#   CRAN_REPO     CRAN repo URL (default: dated PPM snapshot; containers override with
#                 the distro BINARY endpoint for speed).
#   NCPUS         parallel compile jobs (default: detected cores).
# =============================================================================

## ---- locate self, manifest, project root ----------------------------------
args_all  <- commandArgs(FALSE)
self_path <- sub("^--file=", "", grep("^--file=", args_all, value = TRUE))
self_dir  <- if (length(self_path)) normalizePath(dirname(self_path)) else getwd()
manifest_path <- Sys.getenv("R_MANIFEST", unset = file.path(self_dir, "r_packages.yaml"))
project_root  <- Sys.getenv("PROJECT_ROOT",
                            unset = normalizePath(file.path(self_dir, "..", ".."), mustWork = FALSE))

lib <- Sys.getenv("R_LIBS_USER")
if (nzchar(lib)) {
  dir.create(lib, recursive = TRUE, showWarnings = FALSE)
  .libPaths(c(lib, .libPaths()))
} else {
  lib <- .libPaths()[1]
}
ncpus <- as.integer(Sys.getenv("NCPUS", unset = as.character(max(1L, parallel::detectCores()))))
cran  <- Sys.getenv("CRAN_REPO", unset = "https://packagemanager.posit.co/cran/2026-06-02")
options(repos = c(CRAN = cran), Ncpus = ncpus, timeout = 3600)

cat(sprintf("install_r_packages.R\n  R         : %s\n  lib       : %s\n  manifest  : %s\n  CRAN repo : %s\n  Ncpus     : %d\n\n",
            getRversion(), lib, manifest_path, cran, ncpus))

## ---- pak + yaml bootstrap ---------------------------------------------------
if (!requireNamespace("pak", quietly = TRUE)) {
  cat(">> installing pak ...\n"); install.packages("pak", lib = lib)
}
if (!requireNamespace("yaml", quietly = TRUE)) {
  cat(">> installing yaml ...\n"); pak::pkg_install("yaml", lib = lib, ask = FALSE)
}
man <- yaml::read_yaml(manifest_path)
pkgs <- man$packages

src_of <- function(p) p$source
by_src <- function(s) Filter(function(p) src_of(p) == s, pkgs)
ref_github <- function(p) sprintf("%s@%s", p$repo, p$sha)

## ---- 1) report system dependencies for THIS OS ------------------------------
# pak::pkg_sysreqs prints the platform-specific install command (apt/dnf/...). On HPC
# without root these are module/admin hints; the containers turn them into apt lines.
cat("================ SYSTEM DEPENDENCIES (host OS) ================\n")
sysreq_refs <- c(vapply(by_src("cran"),  function(p) p$name, ""),
                 vapply(by_src("github"),function(p) ref_github(p), ""),
                 "bioc::DESeq2")
tryCatch(print(pak::pkg_sysreqs(sysreq_refs, dependencies = NA)),
         error = function(e) cat("  (pkg_sysreqs unavailable:", conditionMessage(e), ")\n"))
cat("NOTE: pak detects required system packages from an rpm/apt database; on a module\n",
    "     cluster these may already be provided by modules (load them / see SETUP.md).\n",
    "==============================================================\n\n", sep = "")

## ---- 2) patched omnideconv: clone @sha, demote dev/doc Imports, local install
patch_omnideconv <- function(p) {
  src <- file.path(Sys.getenv("TMPDIR", unset = tempdir()), "omnideconv_src_patched")
  url <- sprintf("https://github.com/%s.git", p$repo)
  if (!dir.exists(file.path(src, ".git"))) {
    unlink(src, recursive = TRUE)
    system2("git", c("clone", "-q", url, shQuote(src)))
  }
  system2("git", c("-C", shQuote(src), "checkout", "-q", p$sha))
  system2("git", c("-C", shQuote(src), "checkout", "-q", "--", "DESCRIPTION"))  # pristine
  d <- readLines(file.path(src, "DESCRIPTION"))
  # devtools/pkgdown/knitr/rmarkdown/testthat: in Imports for doc-building only, absent
  # from NAMESPACE; they pull the unbuildable ragg->systemfonts/textshaping font stack.
  d <- d[!grepl("^[[:space:]]+(devtools|knitr|pkgdown|rmarkdown|testthat[^,]*),[[:space:]]*$", d)]
  vb <- grep("^VignetteBuilder:", d)
  if (length(vb)) d <- if (grepl("knitr", d[vb])) d[-vb] else d[-c(vb, vb + 1L)]
  writeLines(d, file.path(src, "DESCRIPTION"))
  cat(sprintf(">> omnideconv %s @ %s patched at %s\n", p$version, p$sha, src))
  pak::pkg_install(paste0("local::", src), dependencies = NA, ask = FALSE)
}

## ---- 3) vendored / fallback BayesPrism --------------------------------------
install_vendor <- function(p) {
  vpath <- file.path(project_root, p$path)
  if (dir.exists(vpath) && file.exists(file.path(vpath, "DESCRIPTION"))) {
    cat(sprintf(">> BayesPrism %s from vendored submodule %s\n", p$version, vpath))
    pak::pkg_install(paste0("local::", vpath), dependencies = NA, ask = FALSE)
  } else if (!is.null(p$github_fallback)) {
    ref <- if (!is.null(p$subdir)) sprintf("%s/%s", p$github_fallback, p$subdir) else p$github_fallback
    cat(sprintf(">> BayesPrism: vendored path missing (%s); falling back to github::%s\n",
                vpath, ref))
    cat("   (less reproducible -- prefer `git submodule update --init vendor/BayesPrism`)\n")
    pak::pkg_install(paste0("github::", ref), dependencies = NA, ask = FALSE)
  } else stop("BayesPrism vendored path missing and no github_fallback in manifest")
}

## ---- 4) install everything, fault-tolerantly --------------------------------
install_one <- function(p) {
  if (requireNamespace(p$name, quietly = TRUE)) { cat(sprintf(">> %-13s already present\n", p$name)); return(invisible()) }
  cat(sprintf(">> installing %-13s (%s)\n", p$name, p$source))
  tryCatch(
    switch(p$source,
      cran           = pak::pkg_install(p$name, dependencies = NA, ask = FALSE),
      bioc           = pak::pkg_install(paste0("bioc::", p$name), dependencies = NA, ask = FALSE),
      github         = pak::pkg_install(paste0("github::", ref_github(p)), dependencies = NA, ask = FALSE),
      github_patched = patch_omnideconv(p),
      vendor         = install_vendor(p),
      stop("unknown source: ", p$source)),
    error = function(e) cat(sprintf("   !! %s FAILED: %s\n", p$name, conditionMessage(e))))
}
# Order: CRAN+Bioc (deps) -> GitHub methods -> patched omnideconv -> vendored BayesPrism.
for (s in c("cran", "bioc", "github", "github_patched", "vendor"))
  for (p in by_src(s)) install_one(p)

## ---- 5) verify --------------------------------------------------------------
cat("\n==================== INSTALLED ====================\n")
ok <- TRUE
for (p in pkgs) {
  have <- requireNamespace(p$name, quietly = TRUE)
  ok <- ok && have
  cat(sprintf("  %-14s %s\n", p$name,
              if (have) as.character(packageVersion(p$name)) else "MISSING"))
}
cat(sprintf("\nexcluded by design: %s\n", paste(man$exclude_methods, collapse = ", ")))
if (ok) cat("\nR_ENV_INSTALL_OK\n") else {
  cat("\nR_ENV_INSTALL_INCOMPLETE (some roots missing -- check the FAILED lines above)\n")
  quit(status = 1)
}
