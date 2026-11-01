# config.R -- read pipeline config values from R. The R twin of deconvolution/load_config.sh.
#
# cfg_value("CFG_GENECOMPASS_INPUT_DIR") returns, in order: the environment variable of that name (the .sh
# wrappers export every CFG_* through load_config.sh), else the value emitted by deconvolution/_config_sh.py
# (run once with $DECONV_PYTHON, default python3; honours PIPELINE_PROFILE). A config read failure, or a
# required key that is absent, STOPS the script: there is no fallback to a hardcoded results arm.
#
#   source(file.path(.script_dir, "config.R"))      # or file.path(<repo>, "deconvolution/R/config.R")
#   GC_DIR <- cfg_value("CFG_GENECOMPASS_INPUT_DIR")
#   SEED   <- as.integer(cfg_value("CFG_DE_SEED", default = "1"))

.cfg_env <- new.env(parent = emptyenv())

cfg_root <- function() {
  for (v in c("PIPELINE_ROOT", "PROJECT_ROOT")) {
    r <- Sys.getenv(v)
    if (nzchar(r) && file.exists(file.path(r, "deconvolution", "_config_sh.py"))) return(normalizePath(r))
  }
  if (file.exists(file.path("deconvolution", "_config_sh.py"))) return(normalizePath("."))
  stop("config.R: cannot locate the repository root; run from the repo root or set PIPELINE_ROOT")
}

.cfg_load <- function() {
  if (!is.null(.cfg_env$values)) return(.cfg_env$values)
  root <- cfg_root()
  py <- Sys.getenv("DECONV_PYTHON", "python3")
  old <- Sys.getenv("PIPELINE_ROOT", unset = NA)
  Sys.setenv(PIPELINE_ROOT = root)
  on.exit(if (is.na(old)) Sys.unsetenv("PIPELINE_ROOT") else Sys.setenv(PIPELINE_ROOT = old), add = TRUE)
  out <- tryCatch(suppressWarnings(system2(py, file.path(root, "deconvolution", "_config_sh.py"),
                                           stdout = TRUE, stderr = "")),
                  error = function(e) structure(character(), status = 127L))
  st <- attr(out, "status")
  if (!is.null(st) && st != 0)
    stop(sprintf("config.R: '%s deconvolution/_config_sh.py' failed (exit %s); set DECONV_PYTHON to the venv python",
                 py, st))
  kv <- regmatches(out, regexpr("=", out), invert = TRUE)
  vals <- vapply(kv, function(x) if (length(x) == 2) x[2] else NA_character_, "")
  vals <- sub("^'(.*)'$", "\\1", vals)                     # undo shlex.quote for the simple case
  names(vals) <- vapply(kv, `[`, "", 1)
  .cfg_env$values <- vals[!is.na(vals)]
  .cfg_env$values
}

cfg_value <- function(name, default = NULL) {
  v <- Sys.getenv(name)
  if (nzchar(v)) return(v)
  vals <- .cfg_load()
  if (name %in% names(vals)) return(unname(vals[[name]]))
  if (!is.null(default)) return(default)
  stop(sprintf("config.R: %s is not set in the pipeline config (deconvolution section)", name))
}
