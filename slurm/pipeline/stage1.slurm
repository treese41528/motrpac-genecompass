#!/bin/bash
#SBATCH --job-name=stage1_harvesting
#SBATCH --account=reese18
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=logs/stage1_%j.out
#SBATCH --error=logs/stage1_%j.err

# ============================================================================
# Stage 1: Data Harvesting & QC
#
# Runs the full Stage 1 pipeline via run_stage1.py orchestrator.
# All paths and settings come from config/pipeline_config.yaml.
#
# Usage:
#   sbatch slurm/stage1.slurm                       # full run
#   sbatch slurm/stage1.slurm --from 4              # start from step 4
#   sbatch slurm/stage1.slurm --skip-llm            # skip LLM steps
#   sbatch slurm/stage1.slurm --skip-existing       # skip completed steps
#   sbatch slurm/stage1.slurm --only 5              # run only step 5
#
# Environment:
#   Set PIPELINE_ENV to your Python environment activate script:
#     export PIPELINE_ENV=/path/to/your/venv/bin/activate
# ============================================================================

# Project root — derive from script location (slurm/ is one level below root)
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

# Activate environment
source "${PIPELINE_ENV:?Set PIPELINE_ENV to your Python venv activate script}"

# Create log directory if needed
mkdir -p logs

echo "=================================================="
echo "STAGE 1: Data Harvesting & QC"
echo "Date:    $(date)"
echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $SLURM_NODELIST"
echo "CPUs:    $SLURM_CPUS_PER_TASK"
echo "Memory:  $SLURM_MEM_PER_NODE"
echo "Root:    $PROJECT_ROOT"
echo "Python:  $(which python)"
echo "Args:    $@"
echo "=================================================="

# Run orchestrator — pass through any SLURM script arguments
python pipeline/run_stage1.py "$@"

EXIT_CODE=$?

echo ""
echo "=================================================="
echo "Stage 1 completed at: $(date)"
echo "Exit code: $EXIT_CODE"
echo "=================================================="

exit $EXIT_CODE
