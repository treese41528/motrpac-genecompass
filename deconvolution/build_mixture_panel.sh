#!/bin/bash
# Build the known-truth mixture panel on a set of production references (MIXTURE_PANEL_REFS4ALL_PREREG.md).
#   REFSET=refs5 (default)  docs/audit/remediation/STEP4_PREREG.md V6: the references_v5 builds.
#                           PANEL default data/deconvolution/mixture_panel_refs5
#   REFSET=refs4all         the references in use until refs5, as built on 2026-11-04.
#                           PANEL default data/deconvolution/mixture_panel_refs4all
#   holdout: split each production reference 70/30 by cell type (make_pseudobulk.py --ref-source)
#   cross  : the earlier panel's cross sources that are still a different study from the reference
# 50 mixtures x 1,000 cells, Dirichlet alpha 1, seed 1, >= 20 pool cells per benchmarked type.
# Output: $PANEL/<SET>/{reference,mixtures}/
# Cross harmonization overrides (names only, before deconvolution): deconvolution/reference/mixture_panel_harmonization_overrides.tsv
# Run on a compute node: loads every production reference and four source studies.
#   bash deconvolution/build_mixture_panel.sh [SET ...]      (no args = all 19 sets)
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PROJECT_ROOT="$PWD" PIPELINE_ROOT="$PWD"; source deconvolution/load_config.sh   # fatal on a config error
PY="${DECONV_PYTHON:-python3}"
REFSET="${REFSET:-refs5}"
OVR=deconvolution/reference/mixture_panel_harmonization_overrides.tsv
COMMON=(--n-mixtures 50 --cells-per-mixture 1000 --alpha 1.0 --seed 1 --min-pool-cells 20)

# HOLD:  SET<TAB>tissue<TAB>reference_dir
# CROSS: SET<TAB>tissue<TAB>source study<TAB>conditions (or -)<TAB>label scheme<TAB>reference_dir
case "$REFSET" in
refs5)
  PANEL="${PANEL:-data/deconvolution/mixture_panel_refs5}"; REF_KEY=reference_dir
  HOLD=$(cat <<'EOF'
BAT_holdout	subscapular brown adipose tissue	data/deconvolution/references_v5/BAT_GSE244451
BLOOD_holdout	blood	data/deconvolution/references_v5/BLOOD_composite
CORTEX_holdout	cortex	data/deconvolution/references_v5/CORTEX_GSE303115
HEART_holdout	left ventricle	data/deconvolution/references_v5/HEART_GSE280111_LV
HIPPOC_holdout	brain	data/deconvolution/references_v5/HIPPOC_GSE295314
HYPOTH_holdout	hypothalamus	data/deconvolution/references_v5/HYPOTH_GSE248413_Y
KIDNEY_holdout	kidney	data/deconvolution/references_v5/KIDNEY_GSE240658
LIVER_holdout	liver	data/deconvolution/references_v5/LIVER_GSE220075
LUNG_holdout	lung	data/deconvolution/references_v5/LUNG_native_pooled
MUSCLE_holdout	skeletal muscle	data/deconvolution/references_v5/MUSCLE_GSE137869_Y
OVARY_holdout	ovary	data/deconvolution/references_v5/OVARY_GSE343214
SMLINT_holdout	proximal jejunum	data/deconvolution/references_v5/SMLINT_GSE272055
SPLEEN_holdout	spleen	data/deconvolution/references_v5/SPLEEN_GSE306764
TESTES_holdout	testis	data/deconvolution/references_v5/TESTES_OMIX767_lifted
WATSC_holdout	white adipose tissue	data/deconvolution/references_v5/WATSC_GSE137869_Y
EOF
)
  CROSS=$(cat <<'EOF'
KIDNEY_cross	kidney	GSE137869	-	none	data/deconvolution/references_v5/KIDNEY_GSE240658
CORTEX_cross	cerebral cortex	GSE213978	sham	brain	data/deconvolution/references_v5/CORTEX_GSE303115
HIPPOC_cross	hippocampal dentate gyrus	GSE307917	naïve,sham	brain	data/deconvolution/references_v5/HIPPOC_GSE295314
MUSCLE_cross	skeletal muscle	GSE255196	-	muscle	data/deconvolution/references_v5/MUSCLE_GSE137869_Y
EOF
) ;;
refs4all)
  PANEL="${PANEL:-data/deconvolution/mixture_panel_refs4all}"; REF_KEY=refs4_reference_dir
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
  CROSS=$(cat <<'EOF'
KIDNEY_cross	kidney	GSE137869	-	none	data/deconvolution/references_v4/KIDNEY_GSE240658
CORTEX_cross	cerebral cortex	GSE213978	sham	brain	data/deconvolution/references_v4/CORTEX_GSE303115
HIPPOC_cross	hippocampal dentate gyrus	GSE307917	naïve,sham	brain	data/deconvolution/references_v4/HIPPOC_GSE295314
MUSCLE_cross	skeletal muscle	GSE255196	-	muscle	data/deconvolution/references_v4/MUSCLE_GSE137869_Y
EOF
) ;;
*) echo "FATAL: REFSET must be refs5 or refs4all, got '$REFSET'"; exit 2 ;;
esac
mkdir -p "$PANEL/logs"

# the holdout table must be exactly the ready references of tissue_references.yaml, and every cross reference one
# of them (refs5: each tissue's reference_dir; refs4all: its refs4_reference_dir, the build in use until refs5)
HOLD_TABLE="$HOLD" CROSS_TABLE="$CROSS" REF_KEY="$REF_KEY" "$PY" - <<'PYEOF' || exit 2
import os, sys, yaml
y = yaml.safe_load(open("deconvolution/tissue_references.yaml"))
tis = y.get("tissues", y); key = os.environ["REF_KEY"]
ready = {d[key] for d in tis.values()
         if isinstance(d, dict) and d.get("status") == "ready" and d.get(key)}
rows = lambda name: [l.split("\t") for l in os.environ[name].splitlines() if l.strip()]
hold = {r[2] for r in rows("HOLD_TABLE")}; cross = {r[5] for r in rows("CROSS_TABLE")}
if ready != hold:
    sys.exit(f"FATAL: holdout table differs from tissue_references.yaml ({key}): "
             f"missing {sorted(ready - hold)}, extra {sorted(hold - ready)}")
if not cross <= ready:
    sys.exit(f"FATAL: cross references not in tissue_references.yaml ({key}): {sorted(cross - ready)}")
print(f"panel tables match tissue_references.yaml ({len(hold)} holdout references, {len(cross)} cross, key {key})")
PYEOF

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
