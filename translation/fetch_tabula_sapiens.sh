#!/bin/bash
# fetch_tabula_sapiens.sh -- download the Tabula Sapiens files the atlas anchoring (E.3) reads, and check them
# against translation/tabula_sapiens_md5.tsv.
#
# What it fetches. One h5ad per tissue from the CELLxGENE Discover data portal (public, no account), collection
# e5f58829-1a66-40b5-a624-9046778e74f5 (Tabula Sapiens):
#   first registration  (ATLAS_PROJECTION_PREREG.md):     Blood Lung Muscle Heart Liver Kidney Fat
#                                                          Small_Intestine Testis
#   second registration (ATLAS_PROJECTION_16_PREREG.md):  Spleen Ovary
# into data/external/tabula_sapiens/<Tissue>.h5ad, the folder translation/project_to_atlas.py prep reads
# (--atlas-dir). About 20.6 GB in all.
#
# Where this script comes from. The files on disk were fetched on the cluster by two local job scripts
# (ts_download.slurm, ts_download16.slurm) that were never committed (audit finding 8.6). This is the same
# download (same URLs and dataset ids; curl now with --fail, so an error page is never saved as a file) with
# the sizes and md5s those jobs recorded turned
# into a check. The download jobs named the collection revision of 2026-06-11; the URL carries only the
# dataset id, so a file that CELLxGENE has since revised will fail the md5 check here rather than pass
# silently.
#
# Usage (repo root; network needed; a login node is fine for downloads):
#   bash translation/fetch_tabula_sapiens.sh                 # every tissue (skips files that already verify)
#   bash translation/fetch_tabula_sapiens.sh Spleen Ovary    # a subset, by the tissue names of the table
#   bash translation/fetch_tabula_sapiens.sh --verify        # only check what is on disk; exit 1 on a mismatch
#   ATLAS_DIR=/some/folder bash translation/fetch_tabula_sapiens.sh
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MD5_TSV=translation/tabula_sapiens_md5.tsv
ATLAS_DIR="${ATLAS_DIR:-data/external/tabula_sapiens}"
BASE=https://datasets.cellxgene.cziscience.com

wanted() {   # wanted <tissue> [subset...]: true when no subset was given or the tissue is in it
  local t="$1"; shift
  [ $# -eq 0 ] && return 0
  local x; for x in "$@"; do [ "$x" = "$t" ] && return 0; done
  return 1
}

check() {    # check <tissue> <bytes> <md5>: prints a status line; returns 1 unless the file verifies
  local f="$ATLAS_DIR/$1.h5ad"
  if [ ! -s "$f" ]; then echo "MISSING  $f"; return 1; fi
  if [ "$(stat -c %s "$f")" != "$2" ]; then echo "SIZE     $f ($(stat -c %s "$f") != $2)"; return 1; fi
  if [ "$(md5sum "$f" | cut -d' ' -f1)" != "$3" ]; then echo "MD5      $f"; return 1; fi
  echo "ok       $f"
}

verify_only=0
if [ "${1:-}" = "--verify" ]; then verify_only=1; shift; fi
mkdir -p "$ATLAS_DIR"
bad=0
while IFS=$'\t' read -r tissue id bytes md5 _; do
  [[ "$tissue" == \#* || "$tissue" == tissue || -z "$tissue" ]] && continue
  wanted "$tissue" "$@" || continue
  if [ $verify_only = 1 ]; then check "$tissue" "$bytes" "$md5" || bad=1; continue; fi
  if check "$tissue" "$bytes" "$md5" >/dev/null 2>&1; then echo "have     $ATLAS_DIR/$tissue.h5ad"; continue; fi
  curl -sS -L --fail --retry 5 -o "$ATLAS_DIR/$tissue.h5ad.part" "$BASE/$id.h5ad" \
    && mv "$ATLAS_DIR/$tissue.h5ad.part" "$ATLAS_DIR/$tissue.h5ad" \
    || { echo "FAILED   $BASE/$id.h5ad" >&2; rm -f "$ATLAS_DIR/$tissue.h5ad.part"; bad=1; continue; }
  check "$tissue" "$bytes" "$md5" || bad=1
done < "$MD5_TSV"
[ $bad = 0 ] && echo "all files verify against $MD5_TSV" || { echo "some files are missing or differ" >&2; exit 1; }
