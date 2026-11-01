#!/bin/bash
#SBATCH --job-name=repro_stage2
#SBATCH --account=reese18
#SBATCH --partition=a100-40gb
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --gres=gpu:1
#SBATCH --time=3-12:00:00
#SBATCH --output=logs/repro_%x_%j.out
# Reproducibility check 2026-10-31, Stage 2 (all 3 steps, committed run_stage2.py, sequential as the Feb-23 run)
# + OMIX767 qc_external_mtx (C, E7W) into scratch; then compare with the canonical outputs.
S=/scratch/gilbreth/reese18/motrpac_repro_20261031
cd /depot/reese18/apps/motrpac-genecompass
source /depot/reese18/apps/motrpac-env/bin/activate
export TMPDIR=$S/tmp TEMP=$S/tmp TMP=$S/tmp HF_HOME=$S/tmp/hf HF_DATASETS_CACHE=$S/tmp/hf_ds XDG_CACHE_HOME=$S/tmp/xdg \
       MPLCONFIGDIR=$S/tmp/mpl NUMBA_CACHE_DIR=$S/tmp/numba
export PIPELINE_PROFILE=$S/profiles/repro_s2.yaml
echo "start $(date) $(hostname)"
T0=$(date +%s)
python pipeline/run_stage2.py; echo "run_stage2 exit $? after $(( $(date +%s) - T0 ))s"
T1=$(date +%s)
for s in C E7W; do
  python pipeline/02_gene_universe/qc_external_mtx.py --mtx-dir data/raw/ngdc/datasets/OMIX767/Gene_bc_matrices/${s}_gene_bc_matrices \
     --sample-id OMIX767_grcr8_sample$s > $S/s2/omix_$s.log 2>&1; echo "omix $s exit $?"
done
echo "omix done after $(( $(date +%s) - T1 ))s"
mkdir -p $S/cmp
python $S/scripts/cmp_files.py $S/s2/gene_universe /depot/reese18/data/training/gene_universe $S/cmp/stage2_gene_universe.tsv
python $S/scripts/cmp_h5ad.py $S/s2/qc_h5ad /depot/reese18/data/training/preprocessed/qc_matrices $S/cmp/stage2_qc_h5ad.tsv 12
mkdir -p $S/s2/omix_json && cp $S/s2/qc_h5ad/OMIX767_grcr8_*.qc.json $S/s2/omix_json/ && mkdir -p $S/s2/omix_json_canon && cp /depot/reese18/data/training/preprocessed/qc_matrices/OMIX767_grcr8_*.qc.json $S/s2/omix_json_canon/
python $S/scripts/cmp_files.py $S/s2/omix_json $S/s2/omix_json_canon $S/cmp/stage2_omix_qcjson.tsv
echo "end $(date) total $(( $(date +%s) - T0 ))s"
