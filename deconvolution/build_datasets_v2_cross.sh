#!/bin/bash
# Rebuild the 5 CROSS validation datasets with the FIXED token-level harmonization
# (build_harmonization whole-token + best-overlap; substring 't'/'b' bug fixed).
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PIPELINE_ROOT="$PWD"; command -v python3 >/dev/null && eval "$(python3 deconvolution/_config_sh.py 2>/dev/null || true)"
PY=/depot/reese18/apps/motrpac-env/bin/python3
REF="${CFG_REFERENCE_V2_DIR:-data/deconvolution/references_v2}"; VAL="${CFG_VALIDATION_V2_DIR:-data/deconvolution/validation_v2}"
mkdir -p tmp/dsbuild_v2
run(){ local tag="$1"; shift; local log="tmp/dsbuild_v2/${tag}.log"; echo "[launch] $tag"; \
  $PY deconvolution/make_pseudobulk.py "$@" --out "$VAL/$tag" > "$log" 2>&1 & }
run CTX_cross --mode cross --tissue "cerebral cortex"           --source-study GSE213978 --conditions sham        --ref-dir "$REF/cortex_GSE303115"
run HIP_cross --mode cross --tissue "hippocampal dentate gyrus" --source-study GSE307917 --conditions "naïve,sham" --ref-dir "$REF/hippocampus_GSE305314_WT"
run KID_cross --mode cross --tissue kidney                      --source-study GSE137869                          --ref-dir "$REF/kidney_GSE240658"
run LNG_cross --mode cross --tissue lung                        --source-study GSE196313 --conditions control     --ref-dir "$REF/lung_GSE178405"
run GAS_cross --mode cross --tissue "skeletal muscle"           --source-study GSE137869                          --ref-dir "$REF/gastrocnemius_GSE184413"
pids=($(jobs -p)); rc=0; for p in "${pids[@]}"; do wait "$p" || rc=1; done
echo "cross rebuilds done (rc=$rc)"; for f in tmp/dsbuild_v2/{CTX,HIP,KID,LNG,GAS}*.log; do echo "=== $f ==="; tail -3 "$f"; done
exit $rc
