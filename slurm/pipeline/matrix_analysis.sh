#!/bin/bash
#SBATCH --job-name=matrix_analysis_fast
#SBATCH --output=/depot/reese18/data/logs/matrix_analysis_%j.out
#SBATCH --error=/depot/reese18/data/logs/matrix_analysis_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --partition=a100-40gb
#SBATCH --account=reese18
#SBATCH --gres=gpu:1
#SBATCH --qos=normal

# Fast parallel matrix analysis - should complete in ~1-2 hours for 2000+ studies

echo "=================================================="
echo "FAST Matrix Analysis Job Started: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "=================================================="

cd /depot/reese18/apps/data_exploration
source /depot/reese18/apps/foundational_models_env/bin/activate


# Configuration
CONFIG="/depot/reese18/apps/data_exploration/config.yaml"
WORKERS=${SLURM_CPUS_PER_TASK:-8}

echo ""
echo "Configuration: $CONFIG"
echo "Workers: $WORKERS"
echo ""

# Parse additional arguments
EXTRA_ARGS=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --organism)
            EXTRA_ARGS="$EXTRA_ARGS --organism $2"
            shift 2
            ;;
        --data-type)
            EXTRA_ARGS="$EXTRA_ARGS --data-type $2"
            shift 2
            ;;
        --max-studies)
            EXTRA_ARGS="$EXTRA_ARGS --max-studies $2"
            shift 2
            ;;
        --resume)
            EXTRA_ARGS="$EXTRA_ARGS --resume"
            shift
            ;;
        --extract-tar)
            EXTRA_ARGS="$EXTRA_ARGS --extract-tar"
            shift
            ;;
        *)
            shift
            ;;
    esac
done

echo "Extra arguments: $EXTRA_ARGS"
echo ""

# Run analysis
python scripts/analyze_matrices.py \
    --config "$CONFIG" \
    --workers "$WORKERS" \
    $EXTRA_ARGS

EXIT_CODE=$?

echo ""
echo "=================================================="
echo "Job Completed: $(date)"
echo "Exit Code: $EXIT_CODE"

# Show results summary
if [ -f "/depot/reese18/data/catalog/matrix_analysis.json" ]; then
    echo ""
    echo "Results Summary:"
    python3 -c "
import json
with open('/depot/reese18/data/catalog/matrix_analysis.json') as f:
    d = json.load(f)
    a = d.get('aggregate', {})
    print(f\"  Studies analyzed: {a.get('studies_analyzed', 0)}\")
    print(f\"  Studies with matrices: {a.get('studies_with_matrices', 0)}\")
    print(f\"  Total cells: {a.get('total_cells', 0):,}\")
    print(f\"  Total samples: {a.get('total_samples', 0):,}\")
    print(f\"  Max genes: {a.get('max_genes', 0):,}\")
    print(f\"  Formats: {a.get('formats', {})}\")
    rat = a.get('by_organism', {}).get('Rattus norvegicus', {})
    if rat:
        print(f\"  Rat studies: {rat.get('studies', 0)}\")
        print(f\"  Rat cells: {rat.get('cells', 0):,}\")
"
fi

echo "=================================================="

exit $EXIT_CODE