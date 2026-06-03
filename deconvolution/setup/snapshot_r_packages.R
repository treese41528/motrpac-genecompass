#!/usr/bin/env Rscript
# Print the currently-installed version + GitHub SHA of each manifest root, as a
# copy-pasteable block for r_packages.yaml. Run after a deliberate upgrade to refresh
# the pins:  Rscript deconvolution/setup/snapshot_r_packages.R
# (Prints rather than rewrites, to preserve the YAML's comments.)

args_all  <- commandArgs(FALSE)
self_path <- sub("^--file=", "", grep("^--file=", args_all, value = TRUE))
self_dir  <- if (length(self_path)) normalizePath(dirname(self_path)) else getwd()
lib <- Sys.getenv("R_LIBS_USER"); if (nzchar(lib)) .libPaths(c(lib, .libPaths()))

man <- yaml::read_yaml(file.path(self_dir, "r_packages.yaml"))
cat("# current pins (", as.character(getRversion()), "):\n", sep = "")
for (p in man$packages) {
  if (!requireNamespace(p$name, quietly = TRUE)) { cat(sprintf("  %-14s MISSING\n", p$name)); next }
  ver <- as.character(packageVersion(p$name))
  d   <- packageDescription(p$name)
  sha <- d$RemoteSha
  pin <- if (!is.null(sha)) sprintf("sha: %s", substr(sha, 1, 8)) else "(cran/bioc/vendor)"
  cat(sprintf("  %-14s version: \"%s\"  %s\n", p$name, ver, pin))
}
