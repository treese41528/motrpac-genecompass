# celltype_names.R -- the ONE definition of the cell-type -> filename contract.
# source() this; do not re-implement safe().
#
# WHY THIS FILE EXISTS
# --------------------
# Until 2026-07-12 every script carried its own copy of
#     safe <- function(s) gsub("[^A-Za-z0-9]+", "_", s)
# which is lossy and NOT injective. KIDNEY's reference contains both
# "alpha-intercalated cells" and "beta-intercalated cells" written with the Greek
# letters, and BOTH sanitized to "_intercalated_cells". Consequences:
#   * extract_z.R wrote ONE predz CSV where there should have been two, so one cell
#     type's BayesPrism Z was silently destroyed;
#   * run_pseudobulk_de.R then read that same Z for BOTH blocks -- running a duplicated
#     block, which also perturbed the GLOBAL IHW/repfdr fit shared by all 185 blocks;
#   * the surviving de__ / gsea__ file carried an ambiguous name.
# augur_prep.py hit the same collision, worked around it locally, and the fix was never
# propagated. Hence: one shared definition, and a hard assert.
#
# THE CONTRACT
#   safe()                -- cell type -> filename stem. Transliterates Unicode first, so
#                            the map is injective over the corpus. Byte-identical to the
#                            legacy sanitizer for every pure-ASCII label (only the 3
#                            non-ASCII labels in the corpus change name).
#   assert_ct_injective() -- hard-fail on a colliding label set instead of silently
#                            overwriting a file. Call this in every writer.
#   write_ct_manifest()   -- emit cell_type -> file so consumers JOIN instead of guessing.
#   purge_stale()         -- delete outputs from a previous run whose labels no longer
#                            exist (the other half of the bug: writers never cleaned up,
#                            so relabelled tissues left orphan blocks behind that any
#                            list.files() glob would happily ingest).

CT_TRANSLIT <- c(
  "α" = "alpha", "β" = "beta",  "γ" = "gamma", "δ" = "delta",
  "ε" = "epsilon", "κ" = "kappa", "λ" = "lambda", "μ" = "mu",
  "π" = "pi",    "σ" = "sigma", "τ" = "tau",   "ω" = "omega",
  "ä" = "a", "å" = "a", "ç" = "c", "è" = "e", "é" = "e",
  "í" = "i", "ñ" = "n", "ö" = "o", "ü" = "u",
  "Ä" = "A", "Ö" = "O", "Ü" = "U"
)

ct_translit <- function(s) {
  s <- enc2utf8(as.character(s))
  for (k in names(CT_TRANSLIT))
    s <- gsub(k, CT_TRANSLIT[[k]], s, fixed = TRUE, useBytes = FALSE)
  s
}

safe <- function(s) {
  s <- gsub("[^A-Za-z0-9]+", "_", ct_translit(s))
  gsub("^_+|_+$", "", s)
}

# The pre-2026-07-12 sanitizer. Only for locating artifacts written before the fix.
legacy_safe <- function(s) gsub("[^A-Za-z0-9]+", "_", enc2utf8(as.character(s)))

# Path to a per-cell-type artifact, tolerating files written by the legacy sanitizer.
# Prefers the current name; falls back only if the legacy file actually exists. Lets
# readers open pre-fix artifacts (e.g. the purity-sweep Z) without forcing a rebuild.
resolve_ct_file <- function(dir, cell_type, prefix, ext) {
  p <- file.path(dir, paste0(prefix, safe(cell_type), ext))
  if (file.exists(p)) return(p)
  legacy <- file.path(dir, paste0(prefix, legacy_safe(cell_type), ext))
  if (file.exists(legacy)) legacy else p
}

assert_ct_injective <- function(types, context = "") {
  k   <- safe(types)
  dup <- unique(k[duplicated(k)])
  if (length(dup)) {
    detail <- paste(vapply(dup, function(d) sprintf(
      "    %-30s <- %s", d, paste(sprintf("'%s'", types[k == d]), collapse = ", ")),
      character(1)), collapse = "\n")
    stop(sprintf(paste0(
      "cell-type name collision%s: %d filename(s) claimed by more than one cell type.\n%s\n",
      "  A collision silently overwrites one cell type's data. Add the offending\n",
      "  character(s) to CT_TRANSLIT in deconvolution/R/celltype_names.R."),
      if (nzchar(context)) paste0(" in ", context) else "", length(dup), detail),
      call. = FALSE)
  }
  invisible(TRUE)
}

write_ct_manifest <- function(path, tissue, types, prefix, ext) {
  df <- data.frame(tissue    = tissue,
                   cell_type = types,
                   file      = sprintf("%s%s%s", prefix, safe(types), ext),
                   stringsAsFactors = FALSE)
  write.table(df, path, sep = "\t", quote = FALSE, row.names = FALSE)
  invisible(df)
}

purge_stale <- function(dir, pattern, keep = character(0)) {
  have <- list.files(dir, pattern = pattern)
  dead <- setdiff(have, keep)
  if (length(dead)) {
    file.remove(file.path(dir, dead))
    cat(sprintf("  purged %d stale file(s) in %s: %s%s\n", length(dead), basename(dir),
                paste(head(dead, 4), collapse = ", "),
                if (length(dead) > 4) sprintf(", +%d more", length(dead) - 4) else ""))
  }
  invisible(dead)
}
