#!/bin/bash
# =============================================================================
# PTRM: Run All Benchmarks
#
# Runs PTRM inference and evaluation on all 4 benchmarks:
#   1. Sudoku-Extreme
#   2. Maze-Hard 30x30
#   3. ARC-AGI-2
#
# Usage:
#   ./scripts/run_all_benchmarks.sh                  # Run all
#   ./scripts/run_all_benchmarks.sh sudoku            # Run one
#   ./scripts/run_all_benchmarks.sh --download-only   # Just download data
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

log_info()  { echo -e "${BLUE}[INFO]${NC} $*"; }
log_ok()    { echo -e "${GREEN}[OK]${NC} $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*"; }

# Defaults
BENCHMARKS=("sudoku" "maze" "arc")
DOWNLOAD_ONLY=false
RESULTS_DIR="results"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --download-only)
            DOWNLOAD_ONLY=true
            shift
            ;;
        --results-dir)
            RESULTS_DIR="$2"
            shift 2
            ;;
        sudoku|maze|arc)
            BENCHMARKS=("$1")
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [benchmark] [options]"
            echo ""
            echo "Benchmarks: sudoku, maze, arc"
            echo ""
            echo "Options:"
            echo "  --download-only   Only download models and data"
            echo "  --results-dir DIR Output directory for results (default: results/)"
            echo "  -h, --help        Show this help"
            exit 0
            ;;
        *)
            log_error "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# =============================================================================
# Step 1: Download models
# =============================================================================

log_info "Step 1: Downloading model checkpoints..."
python scripts/download_models.py
log_ok "Models ready."

# =============================================================================
# Step 2: Download/build datasets
# =============================================================================

log_info "Step 2: Building evaluation datasets..."
python scripts/download_data.py --datasets "${BENCHMARKS[@]}"
log_ok "Datasets ready."

if [ "$DOWNLOAD_ONLY" = true ]; then
    log_ok "Download-only mode. Exiting."
    exit 0
fi

# =============================================================================
# Step 3: Run benchmarks
# =============================================================================

mkdir -p "$RESULTS_DIR"

CONFIGS_DIR="config/inference"
BENCHMARK_CONFIG_MAP=(
    "sudoku:sudoku_extreme"
    "maze:maze_hard"
    "arc:arc_agi2"
)

for entry in "${BENCHMARK_CONFIG_MAP[@]}"; do
    bench="${entry%%:*}"
    config="${entry##*:}"

    # Skip if not in selected benchmarks
    if [[ ! " ${BENCHMARKS[*]} " =~ " ${bench} " ]]; then
        continue
    fi

    log_info "========================================="
    log_info "Running benchmark: $bench"
    log_info "Config: $CONFIGS_DIR/$config.yaml"
    log_info "========================================="

    python -m evaluation.evaluate \
        --config "$CONFIGS_DIR/$config.yaml" \
        --output-dir "$RESULTS_DIR/$bench" \
        2>&1 | tee "$RESULTS_DIR/${bench}_log.txt" || {
        log_warn "Benchmark '$bench' failed. Check $RESULTS_DIR/${bench}_log.txt"
        continue
    }

    log_ok "Benchmark '$bench' complete."
done

# =============================================================================
# Step 4: Summary
# =============================================================================

echo ""
log_info "========================================="
log_info "All benchmarks complete!"
log_info "Results saved to: $RESULTS_DIR/"
log_info "========================================="
