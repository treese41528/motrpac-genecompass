#!/bin/bash
# Build the known-truth mixture panel on the CURRENT production references (MIXTURE_PANEL_REFS4ALL_PREREG.md).
#   holdout: split each production reference_dir (tissue_references.yaml) 70/30 by cell type (make_pseudobulk.py --ref-source)
#   cross  : the earlier panel's cross sources that are still a different study from the reference
# 50 mixtures x 1,000 cells, Dirichlet alpha 1, seed 1, >= 20 pool cells per benchmarked type.
# Output: $PANEL/<SET>/{reference,mixtures}/   (PANEL default data/deconvolution/mixture_panel_refs4all)
# Cross harmonization overrides (names only, before deconvolution): deconvolution/reference/mixture_panel_harmonization_overrides.tsv
# Run on a compute node: loads every production reference and four source studies.
#   bash deconvolution/build_mixture_panel.sh [SET ...]      (no args = all 19 sets)
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PROJECT_ROOT="$PWD" PIPELINE_ROOT="$PWD"; source deconvolution/load_config.sh   # fatal on a config error
PY="${DECONV_PYTHON:-python3}"
PANEL="${PANEL:-data/deconvolution/mixture_panel_refs4all}"
OVR=deconvolution/reference/mixture_panel_harmonization_overrides.tsv
COMMON=(--n-mixtures 50 --cells-per-mixture 1000 --alpha 1.0 --seed 1 --min-pool-cells 20)
mkdir -p "$PANEL/logs"

# SET<TAB>tissue<TAB>reference_dir
HOLD=$(cat <<'EOF'
BAT_holdout	subscapular brown adipose tissue	data/deconvolution/references_v3/BAT_GSE244451
BLOOD_holdout	blood	data/deconvolution/references_v4/BLOOD_composite
CORTEX_holdout	cortex	data/deconvolution/references_v4/CORTEX_GSE303115
HEART_holdout	left ventricle	data/deconvolution/references_v3/HEART_GSE280111_LV
HIPPOC_holdout	brain	data/deconvolution/references_v4/HIPPOC_GSE295314
HYPOTH_holdout	hypothalamus	data/deconvolution/references_v3/HYPOTH_GSE248413_Y
KIDNEY_holdout	kidney	data/deconvolution/references_v4/KIDNEY_GSE240658
LIVER_holdout	liver	data/deconvolution/references_v4/LIVER_GSE220075
LUNG_holdout	lung	data/deconvolution/references_v4/LUNG_native_pooled
MUSCLE_holdout	skeletal muscle	data/deconvolution/references_v4/MUSCLE_GSE137869_Y
OVARY_holdout	ovary	data/deconvolution/references_v4/OVARY_GSE343214
SMLINT_holdout	proximal jejunum	data/deconvolution/references_v4/SMLINT_GSE272055
SPLEEN_holdout	spleen	data/deconvolution/references_v4/SPLEEN_GSE306764
TESTES_holdout	testis	data/deconvolution/references_v3/TESTES_OMIX767_lifted
WATSC_holdout	white adipose tissue	data/deconvolution/references_v4/WATSC_GSE137869_Y
EOF
)
# SET<TAB>tissue<TAB>source study<TAB>conditions (or -)<TAB>label scheme<TAB>reference_dir
CROSS=$(cat <<'EOF'
KIDNEY_cross	kidney	GSE137869	-	none	data/deconvolution/references_v4/KIDNEY_GSE240658
CORTEX_cross	cerebral cortex	GSE213978	sham	brain	data/deconvolution/references_v4/CORTEX_GSE303115
HIPPOC_cross	hippocampal dentate gyrus	GSE307917	naïve,sham	brain	data/deconvolution/references_v4/HIPPOC_GSE295314
MUSCLE_cross	skeletal muscle	GSE255196	-	muscle	data/deconvolution/references_v4/MUSCLE_GSE137869_Y
EOF
)

want(){ [ $# -eq 0 ] && return 0; local s; for s in "${SETS[@]}"; do [ "$s" = "$1" ] && return 0; done; return 1; }
SETS=("$@")
rc=0
while IFS=$'\t' read -r set tissue ref; do
  [ -z "$set" ] && continue
  if [ ${#SETS[@]} -gt 0 ]; then want "$set" || continue; fi
  echo "[holdout] $set <- $ref"
  $PY deconvolution/make_pseudobulk.py --mode holdout --tissue "$tissue" --ref-source "$ref" \
      --holdout-frac 0.30 "${COMMON[@]}" --out "$PANEL/$set" > "$PANEL/logs/$set.log" 2>&1 || { echo "  FAIL $set"; rc=1; }
done <<< "$HOLD"
while IFS=$'\t' read -r set tissue src conds scheme ref; do
  [ -z "$set" ] && continue
  if [ ${#SETS[@]} -gt 0 ]; then want "$set" || continue; fi
  args=(--mode cross --tissue "$tissue" --source-study "$src" --ref-dir "$ref" --label-scheme "$scheme")
  [ "$conds" != "-" ] && args+=(--conditions "$conds")
  [ -s "$OVR" ] && args+=(--harmonization-override "$OVR" --override-set "$set")
  echo "[cross]   $set <- $src vs $ref"
  $PY deconvolution/make_pseudobulk.py "${args[@]}" "${COMMON[@]}" --out "$PANEL/$set" > "$PANEL/logs/$set.log" 2>&1 || { echo "  FAIL $set"; rc=1; }
done <<< "$CROSS"
for f in "$PANEL"/logs/*.log; do echo "=== $(basename "$f" .log)"; grep -E "^Wrote|ERROR|not benchmarked" "$f" | tail -3; done
exit $rc
