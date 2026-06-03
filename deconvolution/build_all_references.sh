#!/bin/bash
# Build single-study BayesPrism references for all MoTrPAC-relevant rat tissues,
# in parallel. Each reference = the richest-cell-type study per tissue, filtered to
# its healthy/control/baseline arm (see deconvolution survey). Output dirs:
#   deconvolution/reference/<tissue_normalized w/ spaces->as-is>_<STUDY>/
# Liver (GSE220075) is already built; not repeated here.
#
# Run on a compute node (loads many h5ads): sbatch slurm/analysis/build_references.slurm
# Or directly:  deconvolution/build_all_references.sh
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY=/depot/reese18/apps/motrpac-env/bin/python3
mkdir -p tmp/refbuild

# tissue<TAB>study<TAB>conditions(optional)  -- tissue must match tissue_normalized exactly
JOBS=$(cat <<'EOF'
kidney	GSE240658	No treatment
lung	GSE178405
cortex	GSE303115
hippocampus	GSE305314
heart	GSE155699	control
skeletal muscle	GSE254371
gastrocnemius	GSE184413	Normal ambulation
white adipose tissue	GSE137869
peripheral blood mononuclear cells	GSE285476	healthy control group
EOF
)

pids=()
while IFS=$'\t' read -r tissue study conds; do
  [ -z "$tissue" ] && continue
  log="tmp/refbuild/${study}.log"
  args=(--study "$study" --tissue "$tissue")
  [ -n "${conds// }" ] && args+=(--conditions "$conds")
  echo "[launch] $tissue / $study ${conds:+(cond=$conds)} -> $log"
  $PY deconvolution/build_reference.py "${args[@]}" > "$log" 2>&1 &
  pids+=($!)
done <<< "$JOBS"

echo "launched ${#pids[@]} builds; waiting..."
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
echo "all builds finished (rc=$rc). Per-build tails:"
for f in tmp/refbuild/*.log; do echo "=== $f ==="; tail -3 "$f"; done
exit $rc
