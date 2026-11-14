#!/bin/bash
# Build the SimBu mRNA-bias battery on a set of production references.
#
#   REFSET=refs5 (default)  docs/audit/remediation/STEP4_PREREG.md V5: the references_v5 builds, three arms per
#                           reference, simulated from one seed so the downsample, the composition and the sampled
#                           cells are the same in all three:
#                             A  scaling NONE, remove_bias_in_counts TRUE (read-number)   no mRNA bias
#                             B  scaling NONE, remove_bias_in_counts FALSE                native bias only
#                             C  scaling expressed_genes, remove_bias_in_counts FALSE     added bias
#                           Output $ROOT/<CODE>/{A,B,C}/{reference,mixtures}/, ROOT default
#                           data/deconvolution/simbu_bench_refs5. No dose-response.
#   REFSET=refs4all         SIMBU_BIAS_REFS4ALL_PREREG.md, as run on 2026-11-11: the references in use until
#                           refs5, arms bias (scaling expressed_genes) and nobias (scaling NONE; the same call
#                           as arm B), plus the dose-response on LIVER and BLOOD (alpha 0 0.5 1 2 3 4).
#                           Output $ROOT/<CODE>/{bias,nobias}/ and $ROOT/doseresp/<CODE>/a<alpha>/, ROOT default
#                           data/deconvolution/simbu_bench_refs4all.
#   ARMS="..."              overrides the arms of the REFSET (any of A B C bias nobias).
#
# Per reference: 50 random-composition pseudobulks x 1,000 cells from a seeded <= 500-cells/type downsample, seed 1.
# Items: CODE (the battery arms), dose:CODE (the dose-response), check:CODE (check_simbu_arms.R: the mean library
# size of each cell type's cells as SimBu sums them in arms A, B and C -> $ROOT/arm_check/<CODE>.tsv).
# Run on a compute node: every simulation reads the full production reference.
#   bash deconvolution/omnideconv_bench/build_simbu_battery.sh [CODE ... | dose:CODE ... | check:CODE ...]
#   (no args = every reference of the REFSET, plus its dose-response tissues)
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${DECONV_PYTHON:-python3}"
REFSET="${REFSET:-refs5}"
SIM=deconvolution/omnideconv_bench/simulate_simbu.sh
COMMON=(--nsamples 50 --ncells 1000 --cells-per-type 500 --seed 1)
ALPHAS=(0 0.5 1 2 3 4)

# CODE<TAB>reference_dir: the reference of every ready tissue in tissue_references.yaml (SKMGN/SKMVL share MUSCLE)
case "$REFSET" in
refs5)
  ROOT="${ROOT:-data/deconvolution/simbu_bench_refs5}"; ARMS="${ARMS:-A B C}"; DOSE=(); REF_KEY=reference_dir
  REFS=$(cat <<'EOF'
BAT	data/deconvolution/references_v5/BAT_GSE244451
BLOOD	data/deconvolution/references_v5/BLOOD_composite
CORTEX	data/deconvolution/references_v5/CORTEX_GSE303115
HEART	data/deconvolution/references_v5/HEART_GSE280111_LV
HIPPOC	data/deconvolution/references_v5/HIPPOC_GSE295314
HYPOTH	data/deconvolution/references_v5/HYPOTH_GSE248413_Y
KIDNEY	data/deconvolution/references_v5/KIDNEY_GSE240658
LIVER	data/deconvolution/references_v5/LIVER_GSE220075
LUNG	data/deconvolution/references_v5/LUNG_native_pooled
MUSCLE	data/deconvolution/references_v5/MUSCLE_GSE137869_Y
OVARY	data/deconvolution/references_v5/OVARY_GSE343214
SMLINT	data/deconvolution/references_v5/SMLINT_GSE272055
SPLEEN	data/deconvolution/references_v5/SPLEEN_GSE306764
TESTES	data/deconvolution/references_v5/TESTES_OMIX767_lifted
WATSC	data/deconvolution/references_v5/WATSC_GSE137869_Y
EOF
) ;;
refs4all)
  ROOT="${ROOT:-data/deconvolution/simbu_bench_refs4all}"; ARMS="${ARMS:-bias nobias}"; DOSE=(LIVER BLOOD)
  REF_KEY=refs4_reference_dir
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
) ;;
*) echo "FATAL: REFSET must be refs5 or refs4all, got '$REFSET'"; exit 2 ;;
esac
mkdir -p "$ROOT/logs"

# arm -> the simulate_simbu.sh arguments that define it
arm_args(){
  case "$1" in
    A)      echo "--scaling NONE --remove-bias-in-counts TRUE --remove-bias-method read-number" ;;
    B)      echo "--scaling NONE --remove-bias-in-counts FALSE" ;;
    C)      echo "--scaling expressed_genes --remove-bias-in-counts FALSE" ;;
    bias)   echo "--scaling expressed_genes" ;;
    nobias) echo "--scaling NONE" ;;
    *)      return 1 ;;
  esac
}
for arm in $ARMS; do
  arm_args "$arm" > /dev/null || { echo "FATAL: unknown arm '$arm' (A B C bias nobias)"; exit 2; }
done

# the table must be exactly the ready references of tissue_references.yaml
# (refs5: each tissue's reference_dir; refs4all: its refs4_reference_dir, the build in use until refs5)
REFS_TABLE="$REFS" REF_KEY="$REF_KEY" "$PY" - <<'PYEOF' || exit 2
import os, sys, yaml
y = yaml.safe_load(open("deconvolution/tissue_references.yaml"))
tis = y.get("tissues", y); key = os.environ["REF_KEY"]
ready = {d[key] for d in tis.values()
         if isinstance(d, dict) and d.get("status") == "ready" and d.get(key)}
table = {line.split("\t")[1] for line in os.environ["REFS_TABLE"].splitlines() if line.strip()}
if ready != table:
    sys.exit(f"FATAL: reference table differs from tissue_references.yaml ({key}): "
             f"missing {sorted(ready - table)}, extra {sorted(table - ready)}")
print(f"reference table matches tissue_references.yaml ({len(table)} ready references, key {key})")
PYEOF

ref_of(){ awk -F'\t' -v c="$1" '$1 == c {print $2}' <<< "$REFS"; }
ITEMS=("$@")
if [ ${#ITEMS[@]} -eq 0 ]; then
  while IFS=$'\t' read -r c _; do ITEMS+=("$c"); done <<< "$REFS"
  for c in ${DOSE[@]+"${DOSE[@]}"}; do ITEMS+=("dose:$c"); done
fi

rc=0; LOGS=()
for it in "${ITEMS[@]}"; do
  code="${it#dose:}"; code="${code#check:}"; ref="$(ref_of "$code")"
  [ -n "$ref" ] && [ -f "$ref/reference_counts.mtx" ] || { echo "FAIL $it: no reference for $code"; rc=1; continue; }
  if [ "$it" = "$code" ]; then
    for cond in $ARMS; do
      read -r -a AARGS <<< "$(arm_args "$cond")"
      echo "[battery] $code/$cond (${AARGS[*]}) <- $ref  $(date)"; LOGS+=("$ROOT/logs/${code}_${cond}.log")
      bash "$SIM" --ref-dir "$ref" --out "$ROOT/$code/$cond" --scenario random "${AARGS[@]}" "${COMMON[@]}" \
        > "$ROOT/logs/${code}_${cond}.log" 2>&1 || { echo "  FAIL $code/$cond (see $ROOT/logs/${code}_${cond}.log)"; rc=1; }
    done
  elif [ "$it" = "check:$code" ]; then
    mkdir -p "$ROOT/arm_check"
    echo "[check]   $code <- $ref  $(date)"; LOGS+=("$ROOT/logs/check_${code}.log")
    SIMBU_SCRIPT=check_simbu_arms.R bash "$SIM" --ref-dir "$ref" --out "$ROOT/arm_check/$code.tsv" --code "$code" \
      --cells-per-type 500 --seed 1 --ncells 1000 \
      > "$ROOT/logs/check_${code}.log" 2>&1 || { echo "  FAIL check $code (see $ROOT/logs/check_${code}.log)"; rc=1; }
  else
    for a in "${ALPHAS[@]}"; do
      echo "[dose]    $code/a$a <- $ref  $(date)"; LOGS+=("$ROOT/logs/dose_${code}_a${a}.log")
      bash "$SIM" --ref-dir "$ref" --out "$ROOT/doseresp/$code/a$a" --bias-alpha "$a" "${COMMON[@]}" \
        > "$ROOT/logs/dose_${code}_a${a}.log" 2>&1 || { echo "  FAIL dose $code/a$a (see $ROOT/logs/dose_${code}_a${a}.log)"; rc=1; }
    done
  fi
done
for f in ${LOGS[@]+"${LOGS[@]}"}; do
  echo "=== $(basename "$f" .log): $(grep -E '^downsample|^wrote|^arm [ABC]:|Error|ERROR' "$f" | tr '\n' ' ')"
done
exit $rc
