#!/bin/bash
#SBATCH --job-name=extract_metadata
#SBATCH --account=reese18
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --gpus-per-task=0
#SBATCH --output=extract_metadata_%j.out
#SBATCH --error=extract_metadata_%j.err

# ============================================================================
# Metadata Extraction Job (CONFIGURABLE)
# 
# This job extracts metadata from harvested datasets and generates
# catalog files for the Study Explorer.
#
# Usage:
#   sbatch run_extraction.slurm                          # Use defaults
#   sbatch run_extraction.slurm /path/to/data            # Custom data root
#   sbatch run_extraction.slurm /path/to/data /path/out  # Custom data + output
#
# Or set environment variables before submitting:
#   export MOTRPAC_DATA_ROOT=/my/data
#   export MOTRPAC_CATALOG_DIR=/my/output
#   sbatch run_extraction.slurm
# ============================================================================

echo "============================================"
echo "Starting metadata extraction"
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "============================================"

# Load Python environment

# Activate your environment if you have one
cd /depot/reese18/apps/data_fetch
source /depot/reese18/apps/foundational_models_env/bin/activate
cd /depot/reese18/apps/data_exploration

# ============================================================================
# CONFIGURATION - Customize these or pass as arguments
# ============================================================================

# Data root: where your harvested data lives
# Priority: 1) command line arg, 2) environment variable, 3) default
DATA_ROOT="${1:-${MOTRPAC_DATA_ROOT:-/depot/reese18/data}}"

# Output directory: where to save catalog files  
OUTPUT_DIR="${2:-${MOTRPAC_CATALOG_DIR:-${DATA_ROOT}/catalog}}"

# Script directory - assumes slurm/ is inside motrpac_explorer/
SLURM_DIR="$SLURM_SUBMIT_DIR"
SCRIPT_DIR="$(dirname "$SLURM_DIR")/scripts"

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "Data root: $DATA_ROOT"
echo "Output dir: $OUTPUT_DIR"
echo ""

# Run metadata extraction with auto-detection
echo "Running metadata extraction..."
python "$SCRIPT_DIR/extract_metadata.py" \
    --data-root "$DATA_ROOT" \
    --output-dir "$OUTPUT_DIR" \
    --auto-detect

EXTRACT_STATUS=$?

if [ $EXTRACT_STATUS -eq 0 ]; then
    echo ""
    echo "Extraction completed successfully!"
    
    # Run statistics generation
    echo ""
    echo "Generating statistics..."
    python "$SCRIPT_DIR/generate_statistics.py" \
        --catalog "$OUTPUT_DIR/master_catalog.json" \
        --output "$OUTPUT_DIR/statistics.json" \
        --report
    
    STATS_STATUS=$?
    
    if [ $STATS_STATUS -eq 0 ]; then
        echo ""
        echo "Generating static HTML explorer..."
        python "$SCRIPT_DIR/generate_static_explorer.py" \
            --catalog "$OUTPUT_DIR/master_catalog.json" \
            --stats "$OUTPUT_DIR/statistics.json" \
            --output "$OUTPUT_DIR/explorer.html"
        
        STATIC_STATUS=$?
        
        echo ""
        echo "============================================"
        echo "All tasks completed successfully!"
        echo "============================================"
        echo ""
        echo "Output files:"
        ls -lh "$OUTPUT_DIR"
        echo ""
        echo "=== USAGE OPTIONS ==="
        echo ""
        echo "Option 1: Static HTML (no server needed)"
        echo "  - Copy explorer.html to your local machine"
        echo "  - Open directly in any web browser"
        echo ""
        echo "Option 2: CLI Query Tool (headless)"
        echo "  python $SCRIPT_DIR/query_catalog.py -c $OUTPUT_DIR/master_catalog.json summary"
        echo "  python $SCRIPT_DIR/query_catalog.py -c $OUTPUT_DIR/master_catalog.json search 'rat exercise'"
        echo "  python $SCRIPT_DIR/query_catalog.py -c $OUTPUT_DIR/master_catalog.json list --organism rattus"
        echo ""
        echo "Option 3: HTTP Server (if you have browser access)"
        echo "  python -m http.server 8000 --directory $OUTPUT_DIR"
        echo ""
    else
        echo "ERROR: Statistics generation failed with status $STATS_STATUS"
        exit $STATS_STATUS
    fi
else
    echo "ERROR: Metadata extraction failed with status $EXTRACT_STATUS"
    exit $EXTRACT_STATUS
fi

echo ""
echo "Job completed at: $(date)"
