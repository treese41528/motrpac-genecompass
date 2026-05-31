#!/bin/bash
# Driver for the BayesPrism-Fig-1h purity-sweep comparison on rat liver data.
# Builds sweep mixtures (both paper-faithful holdout + our cross-dataset design),
# runs BayesPrism, extracts pred Z, and scores focal-type expression vs purity
# with the paper's Pearson-on-VST metric.
#
# Prereqs (one-time): deconvolution/R/install_deseq2.sh
# BayesPrism (run.prism) is CPU-heavy -> run STEP 2 on a compute node, or submit
# slurm/analysis/run_purity_sweep.slurm. Steps 1/3/4 are light (login node OK).
#
# Usage:
#   deconvolution/run_purity_sweep.sh build     # step 1: make both sweep stages
#   deconvolution/run_purity_sweep.sh deconv    # step 2: BayesPrism on both (heavy)
#   deconvolution/run_purity_sweep.sh score     # steps 3-4: extract Z + VST score
#   deconvolution/run_purity_sweep.sh all       # build -> deconv -> score
set -eo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # project root
PY=/depot/reese18/apps/motrpac-env/bin/python3
REF=deconvolution/reference/liver_GSE220075
FOCAL="Hepatocytes"
HOLD=deconvolution/validation/SWEEP_hepato_holdout
CROSS=deconvolution/validation/SWEEP_hepato_cross
STAGES=("$HOLD" "$CROSS")
# Sweep by CELL fraction but denser at the high end: hepatocyte RNA fraction lags
# cell fraction in the snRNA cross source, so we need high cell-frac points to
# populate the paper's >50% RNA-purity regime. Scorer bins by realized RNA frac.
GRID="${GRID:-0.1,0.3,0.5,0.7,0.85,0.95}"
REPS="${REPS:-10}"

build() {
  echo "## build holdout (paper-faithful: focal in-reference) ##"
  $PY deconvolution/make_purity_sweep.py --mode holdout --study GSE220075 \
    --tissue liver --focal-type "$FOCAL" --out "$HOLD" \
    --purity-grid "$GRID" --reps "$REPS"
  echo "## build cross (our real MoTrPAC setting) ##"
  $PY deconvolution/make_purity_sweep.py --mode cross --source-study GSE245240 \
    --tissue liver --conditions Nave --focal-type "$FOCAL" \
    --ref-dir "$REF" --out "$CROSS" \
    --purity-grid "$GRID" --reps "$REPS"
}

deconv() {
  echo "## BayesPrism: holdout (its own disjoint reference) ##"
  bash deconvolution/R/run_deconvolution.sh "$HOLD/reference" "$HOLD/mixtures" "$HOLD/results"
  echo "## BayesPrism: cross (fixed liver_GSE220075 reference) ##"
  bash deconvolution/R/run_deconvolution.sh "$REF" "$CROSS/mixtures" "$CROSS/results"
}

score() {
  for s in "${STAGES[@]}"; do
    echo "## extract pred Z + VST score: $s ##"
    bash deconvolution/R/extract_z.sh "$s/results/bp_result.rds" "$s/results"
    bash deconvolution/R/score_z_vst.sh "$s" "$FOCAL"
    $PY deconvolution/score_purity_sweep.py --stage-dir "$s"
  done
}

case "${1:-all}" in
  build)  build ;;
  deconv) deconv ;;
  score)  score ;;
  all)    build; deconv; score ;;
  *) echo "usage: run_purity_sweep.sh [build|deconv|score|all]"; exit 1 ;;
esac
