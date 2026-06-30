#!/bin/bash
# =============================================================================
# PPBench TRM-Att Training Pipeline
# =============================================================================
#
# End-to-end pipeline: dataset build → training → evaluation → HF upload
#
# Replicates arXiv:2605.19943 (Probabilistic Tiny Recursive Models)
# Paper Appendix A.2-A.3 training specification
#
# Requirements:
#   - CUDA GPU with ≥24GB VRAM (trained on H100 80GB in the paper)
#   - Python 3.10+ with PyTorch 2.0+
#   - W&B account (for training tracking)
#   - HuggingFace account (for model upload)
#
# Usage:
#   bash scripts/train_ppbench.sh                    # Full pipeline
#   bash scripts/train_ppbench.sh --data-only        # Build dataset only
#   bash scripts/train_ppbench.sh --skip-data        # Skip dataset build
#   bash scripts/train_ppbench.sh --skip-upload      # Skip HF upload
#   bash scripts/train_ppbench.sh --gpus 4           # Multi-GPU training
#
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Defaults
DATA_DIR="${PROJECT_ROOT}/data/ppbench"
CHECKPOINT_DIR="${PROJECT_ROOT}/checkpoints/ptrm-ppbench-training"
WANDB_PROJECT="ptrm-ppbench-training"
HF_REPO_NAME=""  # Set via --hf-repo or prompted
NUM_GPUS=1
SEED=42

# Flags
DATA_ONLY=false
SKIP_DATA=false
SKIP_UPLOAD=false
EPOCHS=""
EVAL_INTERVAL=""

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

while [[ $# -gt 0 ]]; do
    case $1 in
        --data-only)
            DATA_ONLY=true
            shift
            ;;
        --skip-data)
            SKIP_DATA=true
            shift
            ;;
        --skip-upload)
            SKIP_UPLOAD=true
            shift
            ;;
        --gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --hf-repo)
            HF_REPO_NAME="$2"
            shift 2
            ;;
        --wandb-project)
            WANDB_PROJECT="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --eval-interval)
            EVAL_INTERVAL="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --help)
            echo "Usage: bash scripts/train_ppbench.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --data-only        Build dataset only, don't train"
            echo "  --skip-data        Skip dataset build (already built)"
            echo "  --skip-upload      Skip HuggingFace upload"
            echo "  --gpus N           Number of GPUs for DDP training (default: 1)"
            echo "  --hf-repo NAME     HuggingFace repo name (e.g., user/model-name)"
            echo "  --wandb-project P  W&B project name (default: ptrm-ppbench-training)"
            echo "  --epochs N         Override number of epochs"
            echo "  --eval-interval N  Override evaluation interval"
            echo "  --seed N           Random seed (default: 42)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

log_info()  { echo -e "\033[1;34m[INFO]\033[0m  $*"; }
log_ok()    { echo -e "\033[1;32m[OK]\033[0m    $*"; }
log_warn()  { echo -e "\033[1;33m[WARN]\033[0m  $*"; }
log_error() { echo -e "\033[1;31m[ERROR]\033[0m $*"; }

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------

log_info "PPBench TRM-Att Training Pipeline"
log_info "================================="
log_info "Project root: ${PROJECT_ROOT}"
log_info "Data dir: ${DATA_DIR}"
log_info "Checkpoint dir: ${CHECKPOINT_DIR}"
log_info "GPUs: ${NUM_GPUS}"
echo ""

# GPU/W&B checks are deferred until training step (below)
# so that --data-only works on CPU-only machines

# =============================================================================
# Step 1: Build PPBench Dataset
# =============================================================================

if [ "$SKIP_DATA" = false ]; then
    log_info "Step 1: Building PPBench dataset..."
    log_info "  Downloading from HuggingFace (bluecoconut/pencil-puzzle-bench)"
    log_info "  Filtering to 6 types, building vocab, augmenting..."

    python3 "${PROJECT_ROOT}/dataset/build_ppbench_dataset.py" \
        --output-dir "${DATA_DIR}" \
        --seed "${SEED}"

    log_ok "Dataset built: ${DATA_DIR}"

    # Verify counts
    if [ -f "${DATA_DIR}/train/dataset.json" ]; then
        TRAIN_SIZE=$(python3 -c "import json; print(json.load(open('${DATA_DIR}/train/dataset.json'))['total_puzzles'])")
        log_info "  Train examples: ${TRAIN_SIZE}"
    fi
else
    log_info "Step 1: Skipping dataset build (--skip-data)"
fi

if [ "$DATA_ONLY" = true ]; then
    log_ok "Data-only mode. Exiting."
    exit 0
fi

# =============================================================================
# Pre-flight: GPU & W&B checks (only needed for training)
# =============================================================================

if python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    GPU_NAME=$(python3 -c "import torch; print(torch.cuda.get_device_name(0))")
    GPU_MEM=$(python3 -c "import torch; print(f'{torch.cuda.get_device_properties(0).total_mem / 1e9:.1f}GB')")
    log_ok "CUDA available: ${GPU_NAME} (${GPU_MEM})"
else
    log_error "CUDA not available! Training requires a GPU."
    log_info "Run 'bash scripts/train_ppbench.sh --data-only' to build the dataset without a GPU."
    exit 1
fi

if ! python3 -c "import wandb; wandb.Api()" 2>/dev/null; then
    log_warn "W&B not logged in. Run 'wandb login' first."
    read -p "Continue without W&B? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# =============================================================================
# Step 2: Train TRM-Att on PPBench
# =============================================================================

log_info "Step 2: Training TRM-Att on PPBench..."
log_info "  Architecture: TRM-Att (7M params)"
log_info "  Training recipe: per arXiv:2605.19943"
log_info "  W&B project: ${WANDB_PROJECT}"

TRAIN_CMD="python3 ${PROJECT_ROOT}/scripts/train_ppbench.py"
TRAIN_ARGS="--wandb-project ${WANDB_PROJECT}"

if [ -n "$EPOCHS" ]; then
    TRAIN_ARGS="${TRAIN_ARGS} --epochs ${EPOCHS}"
fi

if [ -n "$EVAL_INTERVAL" ]; then
    TRAIN_ARGS="${TRAIN_ARGS} --eval-interval ${EVAL_INTERVAL}"
fi

if [ "$NUM_GPUS" -gt 1 ]; then
    log_info "  Multi-GPU training with ${NUM_GPUS} GPUs (DDP)"
    torchrun --nproc_per_node="${NUM_GPUS}" \
        "${PROJECT_ROOT}/scripts/train_ppbench.py" \
        ${TRAIN_ARGS}
else
    log_info "  Single-GPU training"
    ${TRAIN_CMD} ${TRAIN_ARGS}
fi

log_ok "Training complete!"

# =============================================================================
# Step 3: Upload to HuggingFace
# =============================================================================

if [ "$SKIP_UPLOAD" = false ]; then
    log_info "Step 3: Uploading best checkpoint to HuggingFace..."

    # Find best checkpoint directory
    BEST_DIR=$(find "${CHECKPOINT_DIR}" -name "best" -type d | head -1)
    if [ -z "$BEST_DIR" ]; then
        log_error "No best checkpoint found in ${CHECKPOINT_DIR}"
        exit 1
    fi

    # Prompt for HF repo name if not set
    if [ -z "$HF_REPO_NAME" ]; then
        read -p "Enter HuggingFace repo name (e.g., username/ptrm-ppbench-trm-att-7m): " HF_REPO_NAME
        if [ -z "$HF_REPO_NAME" ]; then
            log_error "No repo name provided. Skipping upload."
            exit 1
        fi
    fi

    # Get W&B run URL
    WANDB_URL=$(python3 -c "
import wandb
api = wandb.Api()
runs = api.runs('${WANDB_PROJECT}', order='-created_at')
if runs:
    print(runs[0].url)
" 2>/dev/null || echo "")

    python3 "${PROJECT_ROOT}/scripts/upload_to_hf.py" \
        --checkpoint-dir "${BEST_DIR}" \
        --repo-name "${HF_REPO_NAME}" \
        ${WANDB_URL:+--wandb-url "${WANDB_URL}"} \
        --github-url "https://github.com/JerMa88/PTRM"

    log_ok "Upload complete: https://huggingface.co/${HF_REPO_NAME}"
else
    log_info "Step 3: Skipping HuggingFace upload (--skip-upload)"
fi

# =============================================================================
# Done
# =============================================================================

echo ""
log_ok "========================================="
log_ok "PPBench Training Pipeline Complete!"
log_ok "========================================="
log_info "Checkpoint dir: ${CHECKPOINT_DIR}"
if [ -n "$HF_REPO_NAME" ] && [ "$SKIP_UPLOAD" = false ]; then
    log_info "HuggingFace: https://huggingface.co/${HF_REPO_NAME}"
fi
log_info "W&B: https://wandb.ai/${WANDB_PROJECT}"
