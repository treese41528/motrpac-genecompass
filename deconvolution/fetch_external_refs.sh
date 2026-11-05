#!/bin/bash
# fetch_external_refs.sh -- download the deconvolution reference inputs that are NOT in the single-cell corpus,
# into the exact data/raw/ layout the builders read, and check them against deconvolution/external_refs_md5.tsv.
#
#   SPLEEN  GSE306764 Spl-Ctrl  (GSM9207348)  -> data/raw/geo/GSE306764_spleen/          build_external_reference.py
#   OVARY   GSE343214 Normal    (GSM9947804)  -> data/raw/geo/GSE343214/                 build_external_reference.py
#   BLOOD   option B: GSE335156 V1-V5 (GSM9806833-37) + GSE306764 Blood-Ctrl (GSM9207340) and Bone-Ctrl
#           (GSM9207344)                      -> data/raw/geo/blood_option_b/            joint_reannotate.py + build_blood_composite.py
#   HEART2  GSE307551 F3441/F3442 (GSM9226643/4; the §15 second-heart check, not a production reference)
#                                             -> data/raw/geo/GSE307551/
#   BAT     GSE244451_PVAT_BAT_SCPAnnotated.h5ad(.gz) (authors' deposited labels; adopt_authors_labels.py)
#                                             -> data/raw/geo/GSE244451/                 (~4 GB download, ~9 GB unpacked)
#   HEART   SCP2828 ratmap_scp.h5ad (authors' labels; Single Cell Portal, sign-in required) -> data/raw/geo/GSE280111/
#           MANUAL: https://singlecell.broadinstitute.org/single_cell/study/SCP2828 -> Download -> ratmap_scp.h5ad
#   TESTES  OMIX767 (NGDC/CNCB) OMIX767-20-01.zip + OMIX767-20-02.xlsx -> data/raw/ngdc/datasets/OMIX767/
#           https://ngdc.cncb.ac.cn/omix/release/OMIX767 ; tried from https://download.cncb.ac.cn/OMIX/OMIX767/ --
#           if that fails (NGDC may require the browser), download the two files by hand into that folder and rerun.
#
# Usage (repo root; network needed; ~3.5 GB + 4 GB for BAT; a login node is fine for downloads):
#   bash deconvolution/fetch_external_refs.sh                 # everything (skips files whose md5 already matches)
#   bash deconvolution/fetch_external_refs.sh SPLEEN OVARY    # a subset: SPLEEN OVARY BLOOD HEART2 BAT TESTES
#   bash deconvolution/fetch_external_refs.sh --verify        # only check sizes/md5 of what is on disk
# The corpus references (KIDNEY, LIVER, CORTEX, ...) come from the GEO corpus harvest (Stages 1-2); HEART's and
# BAT's corpus samples too -- only their deposited label objects are fetched here.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MD5_TSV=deconvolution/external_refs_md5.tsv
GEO=https://ftp.ncbi.nlm.nih.gov/geo
export TMPDIR="${TMPDIR:-$PWD/tmp}"; mkdir -p "$TMPDIR"

verify() {   # verify [tissue...]: compare bytes + md5 of every listed file (md5 TODO -> size only)
  local bad=0 t p b m k s want
  while IFS=$'\t' read -r t p b m k s; do
    [[ "$t" == \#* || "$t" == tissue ]] && continue
    if [ $# -gt 0 ]; then want=0; for x in "$@"; do [ "$x" = "$t" ] && want=1; done; [ $want = 1 ] || continue; fi
    if [ ! -f "$p" ]; then echo "MISSING  $p"; bad=1; continue; fi
    if [ "$(stat -c %s "$p")" != "$b" ]; then echo "SIZE     $p ($(stat -c %s "$p") != $b)"; bad=1; continue; fi
    if [ "$m" = TODO ]; then echo "size-ok  $p (md5 not recorded)"; continue; fi
    if [ "$(md5sum "$p" | cut -d' ' -f1)" != "$m" ]; then echo "MD5      $p"; bad=1; else echo "ok       $p"; fi
  done < "$MD5_TSV"
  return $bad
}

md5_ok() {   # md5_ok <path>: true if the file exists and matches its recorded md5
  local m; m=$(awk -F'\t' -v p="$1" '$2==p{print $4}' "$MD5_TSV")
  [ -f "$1" ] && [ -n "$m" ] && [ "$m" != TODO ] && [ "$(md5sum "$1" | cut -d' ' -f1)" = "$m" ]
}

get() {      # get <dest_dir> <url>: download with retries unless the file already verifies
  local dir="$1" url="$2" out i; out="$dir/$(basename "$url")"; mkdir -p "$dir"
  if md5_ok "$out"; then echo "have     $out"; return 0; fi
  for i in 1 2 3 4 5 6; do
    curl -sS --fail --retry 5 --retry-all-errors -o "$out.part" "$url" && { mv "$out.part" "$out"; echo "got      $out"; return 0; }
    echo "retry $i $url"; rm -f "$out.part"; sleep 20
  done
  echo "FAILED   $url" >&2; return 1
}

gsm() {      # gsm <GSM> -> the GEO samples/ folder URL
  local g="$1"; echo "$GEO/samples/${g:0:${#g}-3}nnn/$g/suppl"
}

fetch_SPLEEN() {
  local d=data/raw/geo/GSE306764_spleen
  get $d "$(gsm GSM9207348)/GSM9207348_Spl-Ctrl.tar.gz"
  mkdir -p $d/extracted && tar -xzf $d/GSM9207348_Spl-Ctrl.tar.gz -C $d/extracted      # -> extracted/Spl-Ctrl/
}
fetch_OVARY() {
  for f in barcodes.tsv.gz features.tsv.gz matrix.mtx.gz; do
    get data/raw/geo/GSE343214 "$(gsm GSM9947804)/GSM9947804_Normal_$f"; done
}
fetch_HEART2() {
  get data/raw/geo/GSE307551 "$(gsm GSM9226643)/GSM9226643_F3441_feature_bc_matrix.h5"
  get data/raw/geo/GSE307551 "$(gsm GSM9226644)/GSM9226644_F3442_feature_bc_matrix.h5"
}
fetch_BLOOD() {
  local d=data/raw/geo/blood_option_b i g
  for i in 1 2 3 4 5; do g=GSM98068$((32 + i))
    for f in barcodes.tsv.gz features.tsv.gz matrix.mtx.gz; do get $d/GSE335156 "$(gsm $g)/${g}_V${i}_$f"; done
  done
  get $d/GSE306764 "$(gsm GSM9207340)/GSM9207340_Blood-Ctrl.tar.gz"
  get $d/GSE306764 "$(gsm GSM9207344)/GSM9207344_Bone-Ctrl.tar.gz"
  mkdir -p $d/GSE306764/blood $d/GSE306764/bone
  tar -xzf $d/GSE306764/GSM9207340_Blood-Ctrl.tar.gz -C $d/GSE306764/blood              # -> blood/Blood-Ctrl/
  tar -xzf $d/GSE306764/GSM9207344_Bone-Ctrl.tar.gz -C $d/GSE306764/bone                # -> bone/Bone-Ctrl/
}
fetch_BAT() {
  local d=data/raw/geo/GSE244451
  get $d "$GEO/series/GSE244nnn/GSE244451/suppl/GSE244451_PVAT_BAT_SCPAnnotated.h5ad.gz"
  [ -f $d/GSE244451_PVAT_BAT_SCPAnnotated.h5ad ] || gunzip -k $d/GSE244451_PVAT_BAT_SCPAnnotated.h5ad.gz
  local h=data/raw/geo/GSE280111/ratmap_scp.h5ad
  [ -f $h ] || echo "MANUAL   HEART deposited labels: download SCP2828 ratmap_scp.h5ad (sign-in) to $h"
}
fetch_TESTES() {
  local d=data/raw/ngdc/datasets/OMIX767 f
  for f in OMIX767-20-01.zip OMIX767-20-02.xlsx; do
    get $d "https://download.cncb.ac.cn/OMIX/OMIX767/$f" || \
      echo "MANUAL   download $f from https://ngdc.cncb.ac.cn/omix/release/OMIX767 into $d/, then rerun" >&2
  done
  [ -f $d/OMIX767-20-01.zip ] && unzip -n -q $d/OMIX767-20-01.zip -d $d                  # -> Gene_bc_matrices/
}

if [ "${1:-}" = "--verify" ]; then shift; verify "$@"; exit $?; fi
ALL=(SPLEEN OVARY BLOOD HEART2 BAT TESTES)
SEL=("${@:-${ALL[@]}}")
for t in "${SEL[@]}"; do
  case "$t" in SPLEEN|OVARY|BLOOD|HEART2|BAT|TESTES) echo "=== $t ==="; "fetch_$t" ;;
    *) echo "unknown: $t (choose from ${ALL[*]})" >&2; exit 2 ;; esac
done
echo "=== verify ==="
map=(); for t in "${SEL[@]}"; do case $t in HEART2) map+=(HEART_GSE307551);; BAT) map+=(BAT HEART);; *) map+=("$t");; esac; done
verify "${map[@]}"
