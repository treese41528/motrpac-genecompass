#!/usr/bin/env bash
# ============================================================================
# Unpack the six studies behind the 27 keep_raw_slot files of config/corpus_v2_plan.tsv, so Stage 2
# regenerates them. Run after Stage 1 (harvest) and before Stage 2 step 2.
#
#   GSE253767 GSE253672 GSE275999 GSE235314 GSE186158  series <GSE>_RAW.tar -> <GSE>_RAW/, then each nested
#                                                     GSM*.tar.gz unpacked into <GSE>_RAW/<name>/ and removed
#   GSE127248                                         its series RAW.tar on GEO is truncated (and now 404s), so
#                                                     the 24 per-sample files are downloaded into <GSE>_RAW/
#
# Every archive and file is checked against config/rawslot_restore_md5.tsv; a missing tar is downloaded from
# the URL listed there. This is the procedure verified on 2026-11-01: Stage 2 then produced all 27 files
# with the canonical cells and counts, and config/stage2_sample_order.tsv pins their ids
# (docs/repro_stages1_7/REPRODUCTION.md). The regenerated files hold counts in X, where the canonical ones
# held log-normalized X with counts in .raw, so their md5s differ from the plan's; the corpus stages read
# counts from either (lib/gene_utils.py raw_counts_adata).
#
# Usage:  bash scripts/restore_rawslot_studies.sh            (data root from config: harvesting.data_root)
#         bash scripts/restore_rawslot_studies.sh <data_root>
# ============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TABLE="$ROOT/config/rawslot_restore_md5.tsv"
if [ $# -ge 1 ]; then
    DATA_ROOT="$1"
else
    DATA_ROOT="$(cd "$ROOT" && python -c "import sys; sys.path.insert(0, 'lib')
from gene_utils import load_config, resolve_path
c = load_config(); print(resolve_path(c, c.get('harvesting', {}).get('data_root', 'data/raw')))")"
fi
G="$DATA_ROOT/geo/single_cell/geo_datasets"
[ -d "$G" ] || { echo "ERROR: $G not found (run Stage 1 first)" >&2; exit 1; }

check_md5() {  # <path> <md5>
    local got; got=$(md5sum "$1" | cut -d' ' -f1)
    [ "$got" = "$2" ] || { echo "ERROR: md5 mismatch for $1 ($got, expected $2)" >&2; return 1; }
}

fetch() {  # <url> <dest>: GEO's FTP-over-HTTPS drops connections now and then, so retry
    local i
    for i in 1 2 3 4 5 6; do
        curl -fsSL --retry 3 -o "$2.part" "$1" && mv "$2.part" "$2" && return 0
        echo "  retry $i: $1" >&2; sleep $((i * 10))
    done
    rm -f "$2.part"; echo "ERROR: could not download $1" >&2; return 1
}

grep -v '^#' "$TABLE" | tail -n +2 | while IFS=$'\t' read -r rel md5 url; do
    path="$G/$rel"
    mkdir -p "$(dirname "$path")"
    if [ ! -s "$path" ] || ! check_md5 "$path" "$md5" 2>/dev/null; then
        echo "download $rel"; fetch "$url" "$path"
    fi
    check_md5 "$path" "$md5"
    case "$rel" in
        *_RAW.tar)
            gse=$(basename "$rel" _RAW.tar); raw="$G/$gse/${gse}_RAW"
            echo "unpack $gse"; mkdir -p "$raw"; tar xf "$path" -C "$raw"
            for t in "$raw"/*.tar.gz; do
                [ -e "$t" ] || continue
                d="${t%.tar.gz}"; mkdir -p "$d"; tar xzf "$t" -C "$d" && rm "$t"
            done ;;
    esac
done
echo "Done. The six studies are unpacked under $G; run Stage 2 step 2 (preprocess_training_matrices.py)."
