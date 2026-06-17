#!/bin/bash
# Stage 8 multi-tissue validation datasets — V2 (re-audited 2026-06-01).
# Proper healthy sources + adequate sample sizes (see multitissue_validation.md):
#   CROSS (ref_v2 vs distinct healthy source study):
#     cortex      ref GSE303115        <- source GSE213978 (cerebral cortex, sham arm, neuron-rich)
#     hippocampus ref GSE305314 WT     <- source GSE307917 (dentate gyrus, naive/sham, neuron-rich)
#     kidney      ref GSE240658        <- source GSE137869 (kidney, healthy)
#     lung        ref GSE178405        <- source GSE196313 (control arm, balanced parenchyma)
#     gastroc     ref GSE184413        <- source GSE137869 (skeletal muscle, healthy)
#   HOLDOUT (single healthy study, ref built in-place):
#     heart  GSE280111 left ventricle (healthy, cardiomyocyte-containing)
#     white adipose GSE137869 ; PBMC GSE285476 healthy control group
# NB: --tissue selects the SOURCE samples (must match the source's tissue_normalized);
#     the reference comes from --ref-dir, independent of --tissue.
# Output: deconvolution/validation_v2/<TAG>/
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PIPELINE_ROOT="$PWD"; command -v python3 >/dev/null && eval "$(python3 deconvolution/_config_sh.py 2>/dev/null || true)"
PY="${DECONV_PYTHON:-python3}"
REF="${CFG_REFERENCE_V2_DIR:-data/deconvolution/references_v2}"
VAL="${CFG_VALIDATION_V2_DIR:-data/deconvolution/validation_v2}"
mkdir -p tmp/dsbuild_v2 "$VAL"

run () { local tag="$1"; shift
  local log="tmp/dsbuild_v2/${tag}.log"; echo "[launch] $tag -> $log"
  $PY deconvolution/make_pseudobulk.py "$@" --out "$VAL/$tag" > "$log" 2>&1 & }

# CROSS  (--tissue = SOURCE tissue_normalized)
run CTX_cross  --mode cross --tissue "cerebral cortex"          --source-study GSE213978 --conditions sham        --ref-dir "$REF/cortex_GSE303115"
run HIP_cross  --mode cross --tissue "hippocampal dentate gyrus" --source-study GSE307917 --conditions "naïve,sham" --ref-dir "$REF/hippocampus_GSE305314_WT"
run KID_cross  --mode cross --tissue kidney                      --source-study GSE137869                          --ref-dir "$REF/kidney_GSE240658"
run LNG_cross  --mode cross --tissue lung                        --source-study GSE196313 --conditions control     --ref-dir "$REF/lung_GSE178405"
run GAS_cross  --mode cross --tissue "skeletal muscle"           --source-study GSE137869                          --ref-dir "$REF/gastrocnemius_GSE184413"

# HOLDOUT (ref built in-place from the one healthy study)
run HRT_holdout  --mode holdout --tissue "left ventricle" --study GSE280111
run WAT_holdout  --mode holdout --tissue "white adipose tissue" --study GSE137869
run PBMC_holdout --mode holdout --tissue "peripheral blood mononuclear cells" --study GSE285476 --conditions "healthy control group"

pids=($(jobs -p)); echo "launched ${#pids[@]} dataset builds; waiting..."
rc=0; for p in "${pids[@]}"; do wait "$p" || rc=1; done
echo "dataset builds done (rc=$rc):"
for f in tmp/dsbuild_v2/*.log; do echo "=== $f ==="; tail -4 "$f"; done
exit $rc
