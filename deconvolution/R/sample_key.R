# sample_key.R -- read and check the sample key of one tissue. The R twin of deconvolution/sample_join.py.
#
# Stage 8 names the bulk samples of a tissue mix1 ... mixN in the order of <bulk>/<TISSUE>/bulk_samples.tsv and,
# from the refs5 rerun on, writes that pairing beside its fractions: <results>/<TISSUE>/mixture_samples.tsv
# (mix, viallabel; docs/audit/remediation/STEP3_PREREG.md 2.5, audit finding 6.9). A script that needs the
# animal behind a row of a Stage 8 matrix reads the pairing here, from the key, and no longer builds it itself
# from the order of the samples file.
#
#   key <- read_sample_key(samples_file, tissue_results_dir, tissue)   # data.frame(mix, viallabel)
#
# What it checks, and when it stops:
#   * the key exists: its viallabel column must equal the samples file line for line and its mix column must be
#     mix1 ... mixN in order (the checks of run_pseudobulk_de.R). Any difference stops the script.
#   * the key is missing: the script stops. An output tree made before the key was written (everything up to
#     the refs4all rebuild) can still be read by setting ALLOW_POSITIONAL_SAMPLE_JOIN=1, which pairs mix<i> with
#     line i of the samples file, as every run before 2026-11-14 did, and says so.
# When the order is right the returned table is the one the positional join built, so no result changes.

read_sample_key <- function(samples_file, tissue_results_dir, tissue = "") {
  tag <- if (nzchar(tissue)) sprintf("[%s] ", tissue) else ""
  via <- trimws(readLines(samples_file)); via <- via[via != ""]
  if (!length(via)) stop(tag, "no vial labels in ", samples_file)
  positional <- data.frame(mix = paste0("mix", seq_along(via)), viallabel = via, stringsAsFactors = FALSE)
  key_file <- file.path(tissue_results_dir, "mixture_samples.tsv")
  if (!file.exists(key_file)) {
    if (!identical(Sys.getenv("ALLOW_POSITIONAL_SAMPLE_JOIN"), "1"))
      stop(tag, "no sample key ", key_file, ". Stage 8 writes it from the refs5 rerun on ",
           "(docs/audit/remediation/STEP3_PREREG.md 2.5). For an output tree made before that, set ",
           "ALLOW_POSITIONAL_SAMPLE_JOIN=1 to pair mix<i> with line i of ", samples_file, ".")
    message(tag, "no sample key in ", tissue_results_dir, ": samples joined by the order of ", samples_file,
            " (ALLOW_POSITIONAL_SAMPLE_JOIN=1)")
    return(positional)
  }
  kf <- read.delim(key_file, stringsAsFactors = FALSE, colClasses = "character")
  if (!all(c("mix", "viallabel") %in% names(kf)))
    stop(tag, key_file, " has no columns mix and viallabel")
  key <- data.frame(mix = trimws(kf$mix), viallabel = trimws(kf$viallabel), stringsAsFactors = FALSE)
  if (!identical(key$viallabel, via))
    stop(sprintf("%sthe viallabel column of %s (%d rows) is not identical, in order, to %s (%d samples)",
                 tag, key_file, nrow(key), samples_file, length(via)))
  if (!identical(key$mix, positional$mix))
    stop(sprintf("%sthe mix column of %s is not mix1..mix%d in order", tag, key_file, length(via)))
  key
}
