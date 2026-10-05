# provenance.R -- write a small JSON record of how an output was produced. The R twin of
# lib/provenance.py, with the same fields: git commit (plus modified tracked files), command
# line, time, host, run parameters, and the size and mtime of each input file.
#
# It never stops the run: a failure only prints a warning.
#
#   source(file.path(.script_dir, "provenance.R"))
#   write_provenance(out_dir, inputs = c(pheno_tsv), params = list(alpha = ALPHA))
#   # -> <out_dir>/provenance_<script>.json

write_provenance <- function(out_dir, inputs = character(), params = list(), name = NULL) {
  tryCatch({
    args_all <- commandArgs(trailingOnly = FALSE)
    script <- sub("^--file=", "", grep("^--file=", args_all, value = TRUE))
    script <- if (length(script)) tools::file_path_sans_ext(basename(script[1])) else "R"
    repo <- Sys.getenv("PROJECT_ROOT", Sys.getenv("PIPELINE_ROOT", getwd()))
    git <- function(...) {
      out <- tryCatch(suppressWarnings(system2("git", c("-C", shQuote(repo), ...),
                                               stdout = TRUE, stderr = FALSE)),
                      error = function(e) character())
      if (!is.null(attr(out, "status"))) character() else out
    }
    stamp <- function(p) {
      if (!file.exists(p)) return(list(path = p, exists = FALSE))
      i <- file.info(p)
      list(path = p, exists = TRUE, bytes = unname(i$size),
           mtime = format(i$mtime, "%Y-%m-%dT%H:%M:%S"))
    }
    record <- list(
      script = script,
      created = format(Sys.time(), "%Y-%m-%dT%H:%M:%S%z"),
      git_commit = paste(git("rev-parse", "HEAD"), collapse = ""),
      git_modified_tracked_files = I(substring(git("status", "--porcelain", "--untracked-files=no"), 4)),
      command = I(args_all),
      cwd = getwd(),
      host = unname(Sys.info()[["nodename"]]),
      r_version = as.character(getRversion()),
      inputs = lapply(unname(as.character(inputs)), stamp),
      params = params)
    dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
    path <- file.path(out_dir, if (is.null(name)) sprintf("provenance_%s.json", script) else name)
    jsonlite::write_json(record, path, auto_unbox = TRUE, pretty = TRUE, null = "null", na = "null")
    invisible(path)
  }, error = function(e) {
    message("WARNING: provenance not written for ", out_dir, ": ", conditionMessage(e))
    invisible(NULL)
  })
}
