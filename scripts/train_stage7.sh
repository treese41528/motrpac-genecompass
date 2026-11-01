#!/usr/bin/env bash
# ============================================================================
# Stage 7 training launch (committed, scheduler-free copy of the launch v2 used).
#
# Reproduces the torchrun launch in the local SLURM wrapper used for v2 (slurm/finetune/stage7.slurm,
# gitignored): 1 node, torchrun with one process per GPU, fp16, gradient checkpointing, and automatic
# resume from the newest checkpoint-* of the phase. Wrap it in your scheduler's batch script (v2: one
# node, 4 x A100-40GB, 32 CPUs, 256 GB RAM, 6-day jobs; phase 1 = 517,793 steps, ~10 days over 2 jobs).
#
# Usage (from anywhere; paths are resolved from the repository root):
#   PHASE=1 bash scripts/train_stage7.sh      # phase 1 (frozen encoder); rerun the same command to resume
#   PHASE=2 bash scripts/train_stage7.sh      # phase 2 (unfrozen), after phase 1 has saved its final model
#   python pipeline/run_stage7.py --from 4 --phase 1   # the same, via the orchestrator
#
# Environment (all optional):
#   PHASE        1 or 2 (default 1)
#   CONFIG       training YAML, repo-relative (default finetune/genecompass/configs/rat_finetune_v2.yaml)
#   OUT          output root (default: output.output_directory in CONFIG)
#   NPROC        processes per node (default: distributed.nproc_per_node in CONFIG; v2: 4)
#   NNODES       nodes (default: distributed.nnodes in CONFIG; v2: 1)
#   MASTER_PORT  torchrun rendezvous port (default 12348, as for v2)
#   VENV         a venv activate script to source first (default: none; activate your env yourself)
#
# Seeds are rat_pretrain.py's defaults (seed_num 0, seed_val 42; Trainer seed 42), as for v2. GPU kernels
# are not deterministic: a restart tracks v2's loss within fp16 nondeterminism (docs/repro_stages1_7/
# REPRODUCTION.md) but is not bit-identical. The v2 checkpoints are deposited.
# ============================================================================
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PIPELINE_ROOT="$ROOT"
[ -n "${VENV:-}" ] && source "$VENV"

PHASE="${PHASE:-1}"
CONFIG="${CONFIG:-finetune/genecompass/configs/rat_finetune_v2.yaml}"
case "$PHASE" in 1|2) ;; *) echo "PHASE must be 1 or 2, got '$PHASE'" >&2; exit 2 ;; esac
[ -f "$CONFIG" ] || { echo "Training config not found: $CONFIG" >&2; exit 2; }

# Read the launch parameters from the training YAML
cfg() {  # cfg <section> <key> <default>
    python - "$CONFIG" "$1" "$2" "$3" <<'PY'
import sys, yaml
c = yaml.safe_load(open(sys.argv[1])) or {}
v = (c.get(sys.argv[2]) or {}).get(sys.argv[3])
print(sys.argv[4] if v is None else v)
PY
}
OUT="${OUT:-$(cfg output output_directory data/models/rat_genecompass_finetuned_v2)}"
NPROC="${NPROC:-$(cfg distributed nproc_per_node 4)}"
NNODES="${NNODES:-$(cfg distributed nnodes 1)}"
MASTER_PORT="${MASTER_PORT:-12348}"
if [ "$PHASE" -eq 1 ]; then
    RUN_NAME="$(cfg phase1 run_name rat_phase1_freeze_mixed)"
else
    RUN_NAME="$(cfg phase2 run_name rat_phase2_mixed_species)"
fi
PHASE_DIR="$OUT/models/$RUN_NAME"

echo "=================================================="
echo "STAGE 7: GeneCompass Rat Fine-Tuning — Phase $PHASE"
echo "Date:      $(date)"
echo "Host:      $(hostname)"
echo "GPUs:      $(nvidia-smi -L 2>/dev/null | wc -l) visible, nproc_per_node=$NPROC, nnodes=$NNODES"
echo "Root:      $ROOT"
echo "Python:    $(command -v python)"
echo "Config:    $CONFIG"
echo "Output:    $OUT  (phase dir $PHASE_DIR)"
echo "Git:       $(git -C "$ROOT" rev-parse HEAD 2>/dev/null)"
echo "=================================================="

# ── Resume from the newest checkpoint of this phase, if any ────────────────
RESUME_ARG=()
if [ -d "$PHASE_DIR" ]; then
    LATEST_CKPT=$(ls -d "$PHASE_DIR"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
    if [ -n "$LATEST_CKPT" ]; then
        echo "Resuming from checkpoint: $LATEST_CKPT"
        RESUME_ARG=(--resume_from "$LATEST_CKPT")
    fi
fi

torchrun \
    --nproc_per_node="$NPROC" \
    --nnodes="$NNODES" \
    --node_rank=0 \
    --master_port="$MASTER_PORT" \
    finetune/genecompass/rat_pretrain.py \
    --config "$CONFIG" \
    --phase "$PHASE" \
    --output_directory "$OUT" \
    --do_train \
    --save_model \
    --fp16 \
    --gradient_checkpointing \
    "${RESUME_ARG[@]}"
EXIT_CODE=$?

echo "=================================================="
echo "Stage 7 Phase $PHASE finished at $(date), exit code $EXIT_CODE"
echo "=================================================="
exit $EXIT_CODE
