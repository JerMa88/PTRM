# PTRM PPBench Training — Agent Handoff

## Project Overview

**Repository**: `/Users/zma/Documents/programs/PTRM` (GitHub: `JerMa88/PTRM`)  
**Paper**: arXiv:2605.19943 — "Tiny Recursive Models"  
**Goal**: Train a TRM-Att model on the PPBench dataset, replicating the original paper's results exactly.

---

## What's Been Done ✅

### 1. Dataset Built Successfully
- **Script**: [build_ppbench_dataset.py](file:///Users/zma/Documents/programs/PTRM/dataset/build_ppbench_dataset.py)
- **Output**: `data/ppbench/` with train/test/golden splits
- **Stats**:
  | Split | Examples | Base Puzzles |
  |-------|----------|-------------|
  | Train | 91,490 | 9,149 (×10 augmentation) |
  | Val | 550 | 550 |
  | Golden | 61 | 61 |
- **Vocab**: 152 tokens, seq_len=100
- **6 puzzle types**: sudoku, lightup, nurikabe, shakashaka, heyawake, tapa
- Uses `ppbench` library (pzpr.js engine) for proper puzzle decoding
- Raw JSONL downloaded from HuggingFace at `data/ppbench/_raw/`

### 2. Training Pipeline Implemented
All scripts are written and ready. They just need a CUDA GPU to run.

#### Key Files:
- [config/train/ppbench.yaml](file:///Users/zma/Documents/programs/PTRM/config/train/ppbench.yaml) — Training hyperparameters (per paper Appendix A.3)
- [scripts/train_ppbench.py](file:///Users/zma/Documents/programs/PTRM/scripts/train_ppbench.py) — Training wrapper with granular W&B telemetry
- [scripts/train_ppbench.sh](file:///Users/zma/Documents/programs/PTRM/scripts/train_ppbench.sh) — Orchestration shell script
- [scripts/upload_to_hf.py](file:///Users/zma/Documents/programs/PTRM/scripts/upload_to_hf.py) — HuggingFace model upload script

#### Training Hyperparameters (from paper):
- **Architecture**: TRM-Att, ~7M parameters
- **Batch size**: 768
- **Learning rate**: 3e-4 (AdamW with atan2 variant)
- **Epochs**: Configured in `ppbench.yaml`
- **seq_len**: 100 (10×10 grid)
- **VRAM needed**: Only ~4-8 GB (model is tiny at 28MB FP32)

### 3. W&B Integration
- W&B project name: `ptrm-ppbench-training`
- Tracks per-step loss, per-epoch metrics, eval accuracy, learning rate, GPU stats
- Custom hooks in `train_ppbench.py` for maximum telemetry

### 4. HuggingFace Upload Script
- [scripts/upload_to_hf.py](file:///Users/zma/Documents/programs/PTRM/scripts/upload_to_hf.py) — Creates repo, uploads checkpoint + manifest + model card
- Target repo: `JerMa88/ptrm-ppbench-trm-att-7m`

### 5. Demo Notebook
- [demo_ptrm.ipynb](file:///Users/zma/Documents/programs/PTRM/demo_ptrm.ipynb) — Interactive notebook for inference demos
- Has pip install cell, MPS/CUDA/CPU fallback, model architecture comparison, width/depth scaling sweeps

### 6. Requirements Updated
- [requirements.txt](file:///Users/zma/Documents/programs/PTRM/requirements.txt) includes: `adam-atan2`, `coolname`, `wandb`, `numba`, `ppbench`

---

## What Needs To Be Done 🔲

### Step 1: Run Training on a CUDA GPU Machine

The user's Mac has no CUDA. Two options:

**Option A: Cloud GPU** (recommended — cheapest T4/L4/A10 will work fine)
```bash
# On the GPU machine, with the repo cloned and data transferred:
bash scripts/train_ppbench.sh --skip-data --hf-repo JerMa88/ptrm-ppbench-trm-att-7m
```

**Option B: Patch for Apple Silicon MPS** (slower, may have compatibility issues)
The user asked about this. To make it work:
1. Modify `scripts/train_ppbench.sh`: Change the CUDA check to also accept MPS
2. Modify `scripts/train_ppbench.py`: Replace `torch.device("cuda")` with MPS fallback
3. Modify the underlying TRM training code at `TinyRecursiveModels/pretrain.py` if it hardcodes CUDA
4. Disable `torch.compile` (not fully supported on MPS)
5. Reduce batch size (768 → 128 or lower for memory)

### Step 2: Monitor Training
```bash
tail -f logs/train_ppbench_*.log
# Or check W&B dashboard: wandb.ai project ptrm-ppbench-training
```

### Step 3: Upload to HuggingFace
After training completes:
```bash
python scripts/upload_to_hf.py \
  --checkpoint checkpoints/ptrm-ppbench-training/best.pt \
  --hf-repo JerMa88/ptrm-ppbench-trm-att-7m \
  --wandb-project ptrm-ppbench-training
```

---

## Key Architecture Details

### File Organization
```
PTRM/
├── TinyRecursiveModels/      # Core TRM library (upstream, don't modify)
│   ├── pretrain.py           # Main training loop
│   └── ...
├── config/train/ppbench.yaml # Training config
├── data/ppbench/             # Built dataset (91k examples)
│   ├── train/                # all__inputs.npy, all__labels.npy, etc.
│   ├── test/                 # val split (TRM expects "test" dir name)
│   └── golden/
├── dataset/
│   └── build_ppbench_dataset.py  # Dataset builder (uses ppbench lib)
├── inference/                # PTRM inference engine
│   └── checkpoint_loader.py
├── scripts/
│   ├── train_ppbench.py      # Training wrapper with W&B hooks
│   ├── train_ppbench.sh      # Shell orchestrator
│   ├── upload_to_hf.py       # HF upload
│   └── download_models.py    # Model downloader
├── demo_ptrm.ipynb           # Interactive demo
└── requirements.txt
```

### How Training Works
1. `train_ppbench.sh` orchestrates: data check → CUDA check → W&B check → launch training
2. `train_ppbench.py` wraps `TinyRecursiveModels/pretrain.py` with W&B hooks
3. `pretrain.py` does the actual training loop (AdamW-atan2, cosine LR schedule)
4. Checkpoints saved to `checkpoints/ptrm-ppbench-training/`

### The TRM Codebase
- The core TRM library lives in `TinyRecursiveModels/` — this is the upstream code from the paper authors
- Don't modify it unless absolutely necessary
- Our scripts wrap it via config files and monkey-patching for W&B hooks

---

## User Preferences & Constraints

- **Exact replication**: User wants training settings and accuracy to match the paper exactly
- **Max telemetry**: "I WANT MAX TRACKING AT EVERY STEP AND EPOCH AND EVAL AND TESTING"
- **HuggingFace**: Upload best checkpoint with model card referencing paper, GitHub repo, and W&B project
- **No training of other models**: "Do not train anything [else], strictly only pull models from existing repos"
- **Hardware**: User is on a Mac (Apple Silicon, no CUDA). Training must happen on a CUDA machine or the scripts need MPS patching.

---

## Previous Artifacts

- [Implementation Plan](file:///Users/zma/.gemini/antigravity-ide/brain/a87d3a02-21ba-4f9b-ae3c-522de3c4b226/implementation_plan.md)
- [Task List](file:///Users/zma/.gemini/antigravity-ide/brain/a87d3a02-21ba-4f9b-ae3c-522de3c4b226/task.md)
- [Walkthrough](file:///Users/zma/.gemini/antigravity-ide/brain/a87d3a02-21ba-4f9b-ae3c-522de3c4b226/walkthrough.md)
- [Reference Paper PDF](file:///Users/zma/Documents/programs/PTRM/references/2605.19943v1.pdf)
