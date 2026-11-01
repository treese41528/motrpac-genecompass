#!/bin/bash
#SBATCH --job-name=repro_train
#SBATCH --account=reese18
#SBATCH --partition=a100-40gb
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:4
#SBATCH --time=5:00:00
#SBATCH --output=logs/repro_%x_%j.out
# Reproducibility check 2026-10-31, Stage 7: restart v2 phase 1 from the v2 init with the committed config
# and default seeds (seed_num 0 / seed_val 42 / TrainingArguments.seed 42), same launch as slurm/finetune/stage7.slurm
# (torchrun 4 GPUs, --fp16 --gradient_checkpointing --save_model), output to scratch only. Stopped after 2,000 steps.
S=/scratch/gilbreth/reese18/motrpac_repro_20261031
cd /depot/reese18/apps/motrpac-genecompass
source /depot/reese18/apps/motrpac-env/bin/activate
export TMPDIR=$S/tmp TEMP=$S/tmp TMP=$S/tmp HF_HOME=$S/tmp/hf HF_DATASETS_CACHE=$S/tmp/hf_ds XDG_CACHE_HOME=$S/tmp/xdg \
       MPLCONFIGDIR=$S/tmp/mpl NUMBA_CACHE_DIR=$S/tmp/numba TRITON_CACHE_DIR=$S/tmp/triton TORCHINDUCTOR_CACHE_DIR=$S/tmp/inductor \
       WANDB_MODE=disabled WANDB_DIR=$S/tmp/wandb WANDB_CACHE_DIR=$S/tmp/wandb WANDB_CONFIG_DIR=$S/tmp/wandb
mkdir -p $TMPDIR $WANDB_DIR $S/s7 logs
OUT=$S/s7/rat_genecompass_finetuned_v2_repro
LOG=$S/s7/train_stdout.log; ERR=$S/s7/train_stderr.log
echo "node $(hostname) start $(date)"; nvidia-smi -L
torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0 --master_port=12391 \
    finetune/genecompass/rat_pretrain.py \
    --config finetune/genecompass/configs/rat_finetune_v2.yaml --phase 1 \
    --output_directory $OUT --do_train --save_model --fp16 --gradient_checkpointing > $LOG 2> $ERR &
PID=$!
while kill -0 $PID 2>/dev/null; do
  nl=$(grep -c "^{'loss'" $LOG); nid=$(grep -c "^{'id_loss'" $LOG)
  if [ "$nl" -ge 40 ]; then echo "stopping at loss-lines=$nl id-lines=$nid $(date)"; kill -TERM $PID; sleep 60; pkill -KILL -f "rat_pretrain.py.*$OUT"; break; fi
  sleep 15
done
wait $PID; echo "torchrun exit $? end $(date)"
