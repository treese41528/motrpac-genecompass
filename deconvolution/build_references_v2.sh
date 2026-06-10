#!/bin/bash
# Stage 8 multi-tissue references — V2 (re-audited 2026-06-01).
# Fixes from the full-corpus healthy-arm re-audit (condition_resolved is unreliable;
# select on geo_title — see reports/deconvolution/multitissue_validation.md):
#   heart: GSE155699 (SHR, NO cardiomyocytes) -> GSE280111 left ventricle (healthy, CM 20%)
#   hippocampus: GSE305314 ALL-12 (pooled 6 WT + 6 Tau) -> GSE305314 WT-only (6 samples)
# Only the CROSS tissues need a prebuilt reference here; the holdout tissues
# (heart, white adipose, PBMC) build their reference inside make_pseudobulk.
# Output: deconvolution/reference_v2/<tag>/   (parallel to v1 reference/, kept for comparison)
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PIPELINE_ROOT="$PWD"; command -v python3 >/dev/null && eval "$(python3 deconvolution/_config_sh.py 2>/dev/null || true)"
PY=/depot/reese18/apps/motrpac-env/bin/python3
OUT="${CFG_REFERENCE_V2_DIR:-data/deconvolution/references_v2}"
mkdir -p tmp/refbuild_v2 "$OUT"
WT_HIPPO="GSE305314_sample2,GSE305314_sample5,GSE305314_sample7,GSE305314_sample8,GSE305314_sample12,GSE305314_sample14"

launch () {  # tag  -- then build_reference.py args
  local tag="$1"; shift
  local log="tmp/refbuild_v2/${tag}.log"
  echo "[launch] $tag -> $log"
  $PY deconvolution/build_reference.py "$@" --out "$OUT/$tag" > "$log" 2>&1 &
}

# CROSS-tissue references (healthy arms)
# Fix 2 (2026-06-08): GSE303115 cortex samples have uneven gene depth (per-sample 9.5k-21k);
# the default inner (intersection) join collapsed the reference to 5,536 genes (~20% primary
# coverage). --gene-join outer + --min-gene-cells 10 recovers the 21,248-gene union -> 18,162
# genes, lifting training-regulated coverage to ~94%. (173k-cell union OOMs on login -> SLURM.)
launch cortex_GSE303115        --study GSE303115 --tissue cortex --gene-join outer --min-gene-cells 10
launch hippocampus_GSE305314_WT --study GSE305314 --tissue hippocampus --sample-ids "$WT_HIPPO"
launch kidney_GSE240658        --study GSE240658 --tissue kidney --conditions "No treatment"
launch lung_GSE178405          --study GSE178405 --tissue lung
launch gastrocnemius_GSE184413 --study GSE184413 --tissue gastrocnemius --conditions "Normal ambulation"

pids=($(jobs -p)); echo "launched ${#pids[@]} ref builds; waiting..."
rc=0; for p in "${pids[@]}"; do wait "$p" || rc=1; done
echo "ref builds done (rc=$rc):"
for f in tmp/refbuild_v2/*.log; do echo "=== $f ==="; tail -3 "$f"; done
exit $rc
