#!/bin/bash
# Install the omnideconv multi-method deconvolution framework (Dietrich et al. 2026,
# Genome Biol 27:6) into the project-local R library (R_libs/), for the production
# decision #4 multi-method theta cross-check vs our patched BayesPrism.
#
# Installs a *patched* omnideconv core + the R-native reference-based method packages
# (MuSiC, DWLS, SCDC, Bisque, CDSeq, BSeq-sc) via pak. Two deliberate departures from a
# naive `pak::pkg_install("omnideconv/omnideconv", dependencies=TRUE)`, both learned the
# hard way (see git log / commit message):
#
#   1. dependencies = NA (hard deps only). dependencies=TRUE resolves omnideconv's
#      Suggests, one of which is BisqueRNA -- removed from CRAN 2025-06-02, no GitHub
#      remote in omnideconv's DESCRIPTION -- and one unresolvable Suggests aborts pak's
#      ENTIRE solve. We install the method packages explicitly instead (step 3).
#
#   2. We demote omnideconv's dev/doc-only Imports (devtools, pkgdown, knitr, rmarkdown,
#      testthat) to nothing, by patching a local clone's DESCRIPTION. None appear in
#      omnideconv's NAMESPACE; they exist only to build its pkgdown site. Left in, they
#      drag in the pkgdown -> ragg -> {systemfonts, textshaping} font stack
#      (fontconfig/freetype/harfbuzz/fribidi/jpeg/webp) which has neither a Spack module
#      nor headers in the bare conda env -- unbuildable here, and never used at runtime.
#
# scBio (CPM) and MOMF are intentionally NOT installed: CPM pulls terra (gdal/geos/proj)
# and MOMF pulls rgl (OpenGL/glu/X11) -- heavy geospatial/3D system libs with no clean
# source on this cluster, for two non-core extra methods. The MuSiC/DWLS/SCDC/Bisque
# panel is untouched and is the core cross-check.
#
# Run on a Gilbreth login node (needs internet; compute nodes are offline). Reruns are
# idempotent. First run is LONG (~20-40 min: Seurat + many Bioc pkgs from source).
set -eo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export R_LIBS_USER="${PROJECT_ROOT}/R_libs"
export TMPDIR="${PROJECT_ROOT}/tmp"
export R_PROFILE=/dev/null R_PROFILE_USER=/dev/null R_ENVIRON=/dev/null R_ENVIRON_USER=/dev/null
mkdir -p "${R_LIBS_USER}" "${TMPDIR}"

source /etc/profile.d/modules.sh
# r/4.4.1 is the toolchain; the rest provide build-time system headers/libs that the
# bare conda env lacks: libpng (png.h, for Seurat's png dep), zlib (zlib.h), cmake (for
# nloptr). libxml2/curl/libtiff come in with the r module already. openssl + ICU headers
# come from conda's include dir, which we keep on CPATH (we only strip conda from PATH /
# PKG_CONFIG_PATH below, not CPATH). Verified: png/openssl/stringi/igraph/curl/nloptr all
# build AND load under exactly this set.
module load r/4.4.1 libpng/1.6.37 zlib/1.3.1 cmake/3.30.2

# Conda strip (same as install_bayesprism.sh / install_deseq2.sh): keep the module gcc /
# pkg-config in front so compiled deps build+dlopen against the module toolchain, not the
# conda one (conda's libstdc++ otherwise shadows it and breaks scran/igraph). NB: we do
# NOT strip CPATH -- conda's include dir is the only source of openssl/ICU headers here.
export PKG_CONFIG_PATH=$(echo "${PKG_CONFIG_PATH:-}" | tr ':' '\n' \
  | grep -v '/apps/external/conda/2025.09' | tr '\n' ':' | sed 's/:$//')
export PATH=$(echo "${PATH}" | tr ':' '\n' \
  | grep -v '/apps/external/conda/2025.09' | tr '\n' ':' | sed 's/:$//')
# Let pak compile in parallel.
export MAKEFLAGS="-j8"

# ---- fetch + patch omnideconv source (see departure #2 above) ----
OMNIDECONV_SRC="${TMPDIR}/omnideconv_src"
OMNIDECONV_PIN="2dcd4d83bf9813409673636a6ac22c54da142c5e"   # omnideconv 0.1.1, main @ 2026-06-02
if [ ! -d "${OMNIDECONV_SRC}/.git" ]; then
  rm -rf "${OMNIDECONV_SRC}"
  echo ">> cloning omnideconv ..."
  git clone -q https://github.com/omnideconv/omnideconv.git "${OMNIDECONV_SRC}"
fi
git -C "${OMNIDECONV_SRC}" checkout -q "${OMNIDECONV_PIN}" 2>/dev/null \
  || echo "WARN: pin ${OMNIDECONV_PIN} unavailable; using HEAD $(git -C "${OMNIDECONV_SRC}" rev-parse --short HEAD)"
git -C "${OMNIDECONV_SRC}" checkout -q -- DESCRIPTION 2>/dev/null || true   # pristine before re-patch
# Drop the dev/doc-only Imports lines + the knitr VignetteBuilder. (check_and_install()'s
# "devtools::install_github(...)" strings are just message text; the real on-demand
# installer is remotes::install_github, which we keep -- and which is a no-op anyway since
# every method below is pre-installed.)
sed -i -E '/^[[:space:]]+(devtools|knitr|pkgdown|rmarkdown|testthat[^,]*),[[:space:]]*$/d' \
  "${OMNIDECONV_SRC}/DESCRIPTION"
perl -0pi -e 's/\nVignetteBuilder:[^\n]*\n[[:space:]]+knitr\n/\n/' "${OMNIDECONV_SRC}/DESCRIPTION"
export OMNIDECONV_SRC

echo "R           : $(which R)"
echo "R_LIBS_USER : ${R_LIBS_USER}"
echo "omnideconv  : ${OMNIDECONV_SRC} ($(git -C "${OMNIDECONV_SRC}" rev-parse --short HEAD), patched)"
echo "started     : $(date)"
echo "NOTE: pak may print 'Missing system package: libpng-devel/libxml2-devel/...'. Those"
echo "      are FALSE NEGATIVES -- pak probes rpm/dnf and can't see Spack modules or conda."
echo "      The compiles still succeed (verified). Judge success by OMNIDECONV_INSTALL_OK."

R --no-init-file --no-site-file --slave <<'RSCRIPT'
.libPaths(c(Sys.getenv("R_LIBS_USER"), .libPaths()))
options(repos = c(CRAN = "https://cloud.r-project.org"),
        Ncpus = 8, timeout = 3600)

lib <- Sys.getenv("R_LIBS_USER")
omnideconv_src <- Sys.getenv("OMNIDECONV_SRC")

# 1) pak -- fast resolver for the CRAN+Bioc+GitHub mix omnideconv pulls.
if (!requireNamespace("pak", quietly = TRUE)) {
  cat(">> installing pak ...\n")
  install.packages("pak", lib = lib)
}
library(pak)
cat(sprintf(">> pak %s\n", as.character(packageVersion("pak"))))

# 2) omnideconv CORE from the patched local source, hard dependencies only. The patched
#    DESCRIPTION still pulls Seurat + the Bioconductor stack and the GitHub-only Imports
#    SCOPfunctions/bbplot (via omnideconv's Remotes), minus the dev/doc font-stack chain.
if (!requireNamespace("omnideconv", quietly = TRUE)) {
  cat(">> installing patched omnideconv core (hard deps) -- this is the long part ...\n")
  pak::pkg_install(paste0("local::", omnideconv_src), dependencies = NA, ask = FALSE)
} else {
  cat(">> omnideconv already present; ensuring method packages ...\n")
}

# 3) Reference-based method packages for the multi-method theta cross-check, installed
#    one-at-a-time and fault-tolerantly so a single bad build cannot sink the rest. Slugs
#    are the GitHub remotes omnideconv's own DESCRIPTION points at -- these forks are
#    patched to match omnideconv::deconvolute()'s calling convention. limSolve is back on
#    CRAN (2.0.1, 2025-06-24), so cozygene/bisque (pkg name "BisqueRNA") resolves cleanly.
#    EXCLUDED: omnideconv/BayesPrism -- it installs as package "BayesPrism" and would
#    CLOBBER our Danko-Lab-patched BayesPrism 2.2.3 in R_libs/ (this panel cross-checks
#    OUR BayesPrism against these *other* methods); scBio (CPM, pulls terra) and MOMF
#    (pulls rgl) -- heavy geospatial/3D system deps, non-core extras (see header).
method_pkgs <- c(
  MuSiC     = "omnideconv/MuSiC",
  DWLS      = "omnideconv/DWLS",
  SCDC      = "omnideconv/SCDC",
  CDSeq     = "omnideconv/CDSeq",
  bseqsc    = "omnideconv/bseqsc",   # installs clean; running it also needs CIBERSORT.R source
  BisqueRNA = "cozygene/bisque"      # CRAN copy removed; GitHub pkg name is "BisqueRNA"
)
for (nm in names(method_pkgs)) {
  if (!requireNamespace(nm, quietly = TRUE)) {
    cat(sprintf(">> installing method pkg: %-9s (%s)\n", nm, method_pkgs[[nm]]))
    tryCatch(pak::pkg_install(method_pkgs[[nm]], dependencies = NA, ask = FALSE),
             error = function(e) cat(sprintf("   !! %s FAILED: %s\n", nm, conditionMessage(e))))
  } else {
    cat(sprintf(">> %-9s already present\n", nm))
  }
}

# 4) Report what landed.
cat("\n==== METHOD PACKAGE AVAILABILITY ====\n")
probe <- c("omnideconv","MuSiC","DWLS","SCDC","BisqueRNA","CDSeq","bseqsc",
           "Biobase","Seurat","Matrix","reticulate")
for (p in probe)
  cat(sprintf("  %-12s %s\n", p,
      if (requireNamespace(p, quietly = TRUE)) as.character(packageVersion(p)) else "MISSING"))

cross_methods <- c("MuSiC","DWLS","SCDC","BisqueRNA","CDSeq","bseqsc")
methods_ok <- sum(vapply(cross_methods,
                         function(p) requireNamespace(p, quietly = TRUE), logical(1)))
cat(sprintf("\n%d/%d cross-check methods installed\n", methods_ok, length(cross_methods)))

suppressPackageStartupMessages(ok <- requireNamespace("omnideconv", quietly = TRUE))
if (ok) cat("\nOMNIDECONV_INSTALL_OK\n") else stop("omnideconv did NOT install")
RSCRIPT

echo "finished    : $(date)"
echo "OMNIDECONV_INSTALL_DONE"
