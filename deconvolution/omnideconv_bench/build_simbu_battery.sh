#!/bin/bash
# Build the SimBu mRNA-bias battery on the CURRENT production references (SIMBU_BIAS_REFS4ALL_PREREG.md).
#   battery: per reference, 50 random-composition pseudobulks x 1,000 cells from a seeded <= 500-cells/type
#            downsample, simulated twice with seed 1: bias (scaling expressed_genes) and nobias (scaling NONE)
#   dose   : LIVER and BLOOD, composition pinned, mRNA-bias amplitude alpha in 0 0.5 1 2 3 4
# Output: $ROOT/<CODE>/{bias,nobias}/{reference,mixtures}/  and  $ROOT/doseresp/<CODE>/a<alpha>/{reference,mixtures}/
#         (ROOT default data/deconvolution/simbu_bench_refs4all)
# Run on a compute node: every simulation reads the full production reference.
#   bash deconvolution/omnideconv_bench/build_simbu_battery.sh [CODE ... | dose:CODE ...]   (no args = all 15 + dose:LIVER dose:BLOOD)
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${DECONV_PYTHON:-python3}"
ROOT="${ROOT:-data/deconvolution/simbu_bench_refs4all}"
SIM=deconvolution/omnideconv_bench/simulate_simbu.sh
COMMON=(--nsamples 50 --ncells 1000 --cells-per-type 500 --seed 1)
ALPHAS=(0 0.5 1 2 3 4)
DOSE=(LIVER BLOOD)
mkdir -p "$ROOT/logs"

# CODE<TAB>reference_dir: the reference_dir of every ready tissue in tissue_references.yaml (SKMGN/SKMVL share MUSCLE)
REFS=$(cat <<'EOF'
BAT	data/deconvolution/references_v3/BAT_GSE244451
BLOOD	data/deconvolution/references_v4/BLOOD_composite
CORTEX	data/deconvolution/references_v4/CORTEX_GSE303115
HEART	data/deconvolution/references_v3/HEART_GSE280111_LV
HIPPOC	data/deconvolution/references_v4/HIPPOC_GSE295314
HYPOTH	data/deconvolution/references_v3/HYPOTH_GSE248413_Y
KIDNEY	data/deconvolution/references_v4/KIDNEY_GSE240658
LIVER	data/deconvolution/references_v4/LIVER_GSE220075
LUNG	data/deconvolution/references_v4/LUNG_native_pooled
MUSCLE	data/deconvolution/references_v4/MUSCLE_GSE137869_Y
OVARY	data/deconvolution/references_v4/OVARY_GSE343214
SMLINT	data/deconvolution/references_v4/SMLINT_GSE272055
SPLEEN	data/deconvolution/references_v4/SPLEEN_GSE306764
TESTES	data/deconvolution/references_v3/TESTES_OMIX767_lifted
WATSC	data/deconvolution/references_v4/WATSC_GSE137869_Y
EOF
)

# the table must be exactly the ready reference_dirs of tissue_references.yaml
REFS_TABLE="$REFS" "$PY" - <<'PYEOF' || exit 2
import os, sys, yaml
y = yaml.safe_load(open("deconvolution/tissue_references.yaml"))
tis = y.get("tissues", y)
ready = {d["reference_dir"] for d in tis.values()
         if isinstance(d, dict) and d.get("status") == "ready" and d.get("reference_dir")}
table = {line.split("\t")[1] for line in os.environ["REFS_TABLE"].splitlines() if line.strip()}
if ready != table:
    sys.exit(f"FATAL: reference table differs from tissue_references.yaml: "
             f"missing {sorted(ready - table)}, extra {sorted(table - ready)}")
print(f"reference table matches tissue_references.yaml ({len(table)} ready references)")
PYEOF

ref_of(){ awk -F'\t' -v c="$1" '$1 == c {print $2}' <<< "$REFS"; }
ITEMS=("$@")
if [ ${#ITEMS[@]} -eq 0 ]; then
  while IFS=$'\t' read -r c _; do ITEMS+=("$c"); done <<< "$REFS"
  for c in "${DOSE[@]}"; do ITEMS+=("dose:$c"); done
fi

rc=0; LOGS=()
for it in "${ITEMS[@]}"; do
  code="${it#dose:}"; ref="$(ref_of "$code")"
  [ -n "$ref" ] && [ -f "$ref/reference_counts.mtx" ] || { echo "FAIL $it: no reference for $code"; rc=1; continue; }
  if [ "$it" = "$code" ]; then
    for cond in bias nobias; do
      scal=expressed_genes; [ "$cond" = nobias ] && scal=NONE
      echo "[battery] $code/$cond (scaling=$scal) <- $ref  $(date)"; LOGS+=("$ROOT/logs/${code}_${cond}.log")
      bash "$SIM" --ref-dir "$ref" --out "$ROOT/$code/$cond" --scenario random --scaling "$scal" "${COMMON[@]}" \
        > "$ROOT/logs/${code}_${cond}.log" 2>&1 || { echo "  FAIL $code/$cond (see $ROOT/logs/${code}_${cond}.log)"; rc=1; }
    done
  else
    for a in "${ALPHAS[@]}"; do
      echo "[dose]    $code/a$a <- $ref  $(date)"; LOGS+=("$ROOT/logs/dose_${code}_a${a}.log")
      bash "$SIM" --ref-dir "$ref" --out "$ROOT/doseresp/$code/a$a" --bias-alpha "$a" "${COMMON[@]}" \
        > "$ROOT/logs/dose_${code}_a${a}.log" 2>&1 || { echo "  FAIL dose $code/a$a (see $ROOT/logs/dose_${code}_a${a}.log)"; rc=1; }
    done
  fi
done
for f in "${LOGS[@]}"; do echo "=== $(basename "$f" .log): $(grep -E '^downsample|^wrote|Error|ERROR' "$f" | tr '\n' ' ')"; done
exit $rc
