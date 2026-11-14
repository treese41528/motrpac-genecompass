# provenance.R -- write a small JSON record of how an output was produced. The R twin of
# lib/provenance.py, with the same fields: git commit (plus modified tracked files), command
# line, time, host, run parameters, and the size, mtime and md5 of each input file.
#
# It never stops the run: a failure only prints a warning. It is called after the outputs it describes are
# written and never reads or changes them.
#
# md5. Since 2026-11-14 each regular input file under 2 GB also gets its md5 (tools::md5sum), as in
# lib/provenance.py; size and mtime are kept. Larger files carry md5_skipped, directories carry neither.
# Before that date the md5 was off unless PROVENANCE_MD5=1, so the records held size and time only (audit,
# cross-stage "provenance"). md5 = FALSE, or env PROVENANCE_MD5=0, turns it off.
#
#   source(file.path(.script_dir, "provenance.R"))
#   write_provenance(out_dir, inputs = c(pheno_tsv), params = list(alpha = ALPHA))
#   # -> <out_dir>/provenance_<script>.json

PROVENANCE_MD5_MAX_BYTES <- 2 * 1024^3   # as MD5_MAX_BYTES in lib/provenance.py

write_provenance <- function(out_dir, inputs = character(), params = list(), name = NULL,
                             md5 = !identical(Sys.getenv("PROVENANCE_MD5"), "0")) {
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
      rec <- list(path = p, exists = TRUE, bytes = unname(i$size),
                  mtime = format(i$mtime, "%Y-%m-%dT%H:%M:%S"))
      if (isTRUE(md5) && !isTRUE(i$isdir)) {
        if (isTRUE(i$size < PROVENANCE_MD5_MAX_BYTES)) {
          rec$md5 <- tryCatch(unname(tools::md5sum(p)), error = function(e) NA_character_)
        } else {
          rec["md5"] <- list(NULL)
          rec$md5_skipped <- sprintf("file >= %.0f bytes", PROVENANCE_MD5_MAX_BYTES)
        }
      }
      rec
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
