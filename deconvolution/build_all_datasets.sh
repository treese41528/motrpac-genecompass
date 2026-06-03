#!/bin/bash
# Build validation pseudobulk datasets for every tissue with a built reference.
# Cross-dataset (V1-style) where a distinct source study exists; holdout (V0) for
# single-study tissues. Mirrors the liver V0/V1 harness (make_pseudobulk.py).
# Output: deconvolution/validation/<TAG>/mixtures/  (+ reference/ for holdout).
#
# Source studies VERIFIED present in annotation_inventory.tsv (tmp/source_verify).
# Run on a compute node (loads many h5ads): sbatch slurm/analysis/build_datasets.slurm
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY=/depot/reese18/apps/motrpac-env/bin/python3
REFROOT=deconvolution/reference
VAL=deconvolution/validation
mkdir -p tmp/dsbuild

# CROSS jobs: tag<TAB>tissue_normalized<TAB>ref_study<TAB>source_study<TAB>source_conditions(optional)
CROSS=$(cat <<'EOF'
KID_cross	kidney	GSE240658	GSE289104
LNG_cross	lung	GSE178405	GSE247625
CTX_cross	cortex	GSE303115	GSE181979
HIP_cross	hippocampus	GSE305314	GSE232936
HRT_cross	heart	GSE155699	GSE237527
SKM_cross	skeletal muscle	GSE254371	GSE255196
EOF
)

# HOLDOUT jobs (single-study tissues): tag<TAB>tissue_normalized<TAB>study<TAB>conditions(optional)
HOLD=$(cat <<'EOF'
GAS_holdout	gastrocnemius	GSE184413	Normal ambulation
WAT_holdout	white adipose tissue	GSE137869
PBMC_holdout	peripheral blood mononuclear cells	GSE285476	healthy control group
EOF
)

pids=()
while IFS=$'\t' read -r tag tissue ref src conds; do
  [ -z "$tag" ] && continue
  log="tmp/dsbuild/${tag}.log"
  args=(--mode cross --tissue "$tissue" --source-study "$src"
        --ref-dir "$REFROOT/${tissue}_${ref}" --out "$VAL/$tag")
  [ -n "${conds// }" ] && args+=(--conditions "$conds")
  echo "[cross]  $tag  $tissue  src=$src vs ref=$ref ${conds:+(cond=$conds)} -> $log"
  $PY deconvolution/make_pseudobulk.py "${args[@]}" > "$log" 2>&1 &
  pids+=($!)
done <<< "$CROSS"

while IFS=$'\t' read -r tag tissue study conds; do
  [ -z "$tag" ] && continue
  log="tmp/dsbuild/${tag}.log"
  args=(--mode holdout --tissue "$tissue" --study "$study" --out "$VAL/$tag")
  [ -n "${conds// }" ] && args+=(--conditions "$conds")
  echo "[holdout] $tag  $tissue  study=$study ${conds:+(cond=$conds)} -> $log"
  $PY deconvolution/make_pseudobulk.py "${args[@]}" > "$log" 2>&1 &
  pids+=($!)
done <<< "$HOLD"

echo "launched ${#pids[@]} dataset builds; waiting..."
rc=0; for p in "${pids[@]}"; do wait "$p" || rc=1; done
echo "all dataset builds finished (rc=$rc). Tails:"
for f in tmp/dsbuild/*.log; do echo "=== $f ==="; tail -4 "$f"; done
exit $rc
