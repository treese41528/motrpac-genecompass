#!/usr/bin/env bash
# ============================================================================
# Stage the GeneCompass base assets under vendor/GeneCompass/ and check them against
# config/genecompass_assets_md5.tsv (the files the v2 build used).
#
# vendor/GeneCompass is the pristine upstream submodule (github.com/xCompass-AI/GeneCompass). Its large
# assets are not in git; upstream distributes them by browser download only (ScienceDB / Google Drive), so
# this script cannot fetch them for you. It tells you what to put where, applies the one local edit v2
# depends on, and verifies md5s (small files) and sizes (arrow shards).
#
#   What                               Where (under vendor/GeneCompass/)        Upstream download
#   GeneCompass_Base checkpoint        pretrained_models/GeneCompass_Base/      https://www.scidb.cn/en/anonymous/SUZOdk1y
#   prior-knowledge embeddings/tokens  prior_knowledge/                         https://drive.google.com/drive/folders/1a8jhpyw4jmAifK6Ui6u1RJtRFlbsdx08
#     (the PECA2vec, co-expression, family and promoter pickles are also in the submodule's git tree)
#   0.5M human pretraining cells       data/randsel_500w_human/                 https://www.scidb.cn/file?fid=d46129ce14de622c4a07a1d1574dddaa&mode=front
#   0.5M mouse pretraining cells       data/randsel_500w_mouse/                 https://www.scidb.cn/file?fid=65ff1a08ea3292dbbb2ca113f5041ae8&mode=front
#   human gene medians + token dict    scdata/dict/                             in the submodule's git tree
#   (links as listed in vendor/GeneCompass/README.md, "Download Checkpoints" and the 0.5M data row)
#
# The mouse-shard edit (2026-03-31): the downloaded randsel_500w_mouse lacks data-00107-of-00165.arrow
# (164 of 165 shards arrived), so its state.json was edited to drop that entry; the upstream original is
# kept as state.json.bak. v2's training mix was built from the edited set, so a rebuild must apply the
# same edit (this script does, byte for byte: md5 a8afcfe7674a6e3adfa484061a1cb00d). If your download
# does include shard 00107, the mix will not match v2's.
#
# Usage:
#   bash scripts/stage_genecompass_assets.sh             # check presence, apply the mouse edit, verify
#   bash scripts/stage_genecompass_assets.sh --verify    # verify only (no edits)
#   bash scripts/stage_genecompass_assets.sh --no-md5    # sizes only (skip hashing, ~1.6 GB of small files)
# ============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GC="$ROOT/vendor/GeneCompass"
TABLE="$ROOT/config/genecompass_assets_md5.tsv"
EDIT=1; MD5=1
for a in "$@"; do
    case "$a" in
        --verify) EDIT=0 ;;
        --no-md5) MD5=0 ;;
        -h|--help) sed -n '2,33p' "$0"; exit 0 ;;
        *) echo "unknown option: $a" >&2; exit 2 ;;
    esac
done

[ -d "$GC/.git" ] || [ -f "$GC/.git" ] || {
    echo "vendor/GeneCompass is not checked out: git submodule update --init vendor/GeneCompass" >&2; exit 1; }

missing=0
for d in pretrained_models/GeneCompass_Base prior_knowledge scdata/dict data/randsel_500w_human data/randsel_500w_mouse; do
    if [ ! -d "$GC/$d" ]; then echo "MISSING: vendor/GeneCompass/$d (see the download table in this script)"; missing=1; fi
done
[ "$missing" -eq 0 ] || exit 1

# ── The mouse state.json edit ──────────────────────────────────────────────
MOUSE="$GC/data/randsel_500w_mouse"
ORIG_MD5=7ad1fe1a7cb84d77adae26cac9a6afd5      # upstream state.json (165 shards)
EDIT_MD5=a8afcfe7674a6e3adfa484061a1cb00d      # v2's edited state.json (164 shards)
cur=$(md5sum "$MOUSE/state.json" | cut -d' ' -f1)
if [ "$cur" = "$EDIT_MD5" ]; then
    echo "randsel_500w_mouse/state.json: already edited (shard 00107 dropped) [OK]"
elif [ "$cur" = "$ORIG_MD5" ]; then
    if [ -f "$MOUSE/data-00107-of-00165.arrow" ]; then
        echo "NOTE: shard data-00107-of-00165.arrow is present; v2 was trained WITHOUT it. Applying the edit"
        echo "      anyway so the mix matches v2 (the shard is then unused)."
    fi
    if [ "$EDIT" -eq 1 ]; then
        cp -p "$MOUSE/state.json" "$MOUSE/state.json.bak"
        python3 - "$MOUSE/state.json.bak" "$MOUSE/state.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
d["_data_files"] = [x for x in d["_data_files"] if x["filename"] != "data-00107-of-00165.arrow"]
open(sys.argv[2], "w").write(json.dumps(d, indent=2))   # no trailing newline, as in the v2 file
PY
        new=$(md5sum "$MOUSE/state.json" | cut -d' ' -f1)
        [ "$new" = "$EDIT_MD5" ] || { echo "edited state.json md5 $new != $EDIT_MD5" >&2; exit 1; }
        echo "randsel_500w_mouse/state.json: edited (original kept as state.json.bak) [OK]"
    else
        echo "randsel_500w_mouse/state.json: upstream original; run without --verify to apply the v2 edit"
    fi
else
    echo "WARNING: randsel_500w_mouse/state.json md5 $cur is neither the upstream original nor v2's edit"
fi

# ── Verify against the table ──────────────────────────────────────────────
python3 - "$ROOT" "$TABLE" "$MD5" <<'PY'
import csv, hashlib, os, sys
root, table, do_md5 = sys.argv[1], sys.argv[2], sys.argv[3] == "1"
rows = [l for l in open(table) if l.strip() and not l.startswith("#")]
bad = n = 0
for r in csv.DictReader(rows, delimiter="\t"):
    p = os.path.join(root, r["path"]); n += 1
    if not os.path.isfile(p):
        if r["path"].endswith("state.json.bak"):
            continue                       # only present after the edit has been applied once
        print(f"MISSING  {r['path']}"); bad += 1; continue
    if os.path.getsize(p) != int(r["bytes"]):
        print(f"SIZE     {r['path']}: {os.path.getsize(p)} != {r['bytes']}"); bad += 1; continue
    if do_md5 and r["md5"] != "-":
        h = hashlib.md5()
        with open(p, "rb") as f:
            for b in iter(lambda: f.read(1 << 20), b""):
                h.update(b)
        if h.hexdigest() != r["md5"]:
            print(f"MD5      {r['path']}: {h.hexdigest()} != {r['md5']}"); bad += 1
print(f"{n - bad}/{n} files match config/genecompass_assets_md5.tsv"
      + ("" if do_md5 else " (sizes only)"))
sys.exit(1 if bad else 0)
PY
