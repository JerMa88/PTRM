# PTRM — Probabilistic Tiny Recursive Models

Community implementation of [Probabilistic Tiny Recursive Models](https://arxiv.org/abs/2605.19943) (arXiv:2605.19943).

PTRM transforms pre-trained Tiny Recursive Models (TRM) into probabilistic reasoners by injecting calibrated noise into their latent states during inference, enabling diverse solution exploration without any additional training.

## Key Idea

TRM models recursively refine latent states `z_H` and `z_L` to solve structured puzzles. PTRM adds noise (`z_H += ε, ε ~ N(0, σ²I)`) before each supervision step, then runs **K parallel rollouts** and selects the best solution using the model's own Q-head (**best-Q@K**).

## TRM vs PTRM Architecture

PTRM is an **inference-time only** wrapper around a TRM model. The underlying model weights are identical; the difference is entirely in how the latent space is traversed during inference.

| Feature | TRM (Baseline) | PTRM |
|:---|:---|:---|
| **Inference Path** | Single, deterministic trajectory | $K$ parallel, stochastic trajectories |
| **Latent State ($z_H$)** | $z_H^{(t)} = f(z_H^{(t-1)}, x)$ | $z_H^{(t)} = f(z_H^{(t-1)} + \varepsilon, x)$ where $\varepsilon \sim \mathcal{N}(0, \sigma^2 I)$ |
| **Output Selection** | Single fixed-point attractor | **best-Q@K** or **mode@K** |
| **Output Confidence** | Single Q-head logit | Selection based on maximum Q-head logit across $K$ rollouts |
| **Training Required?**| Yes | **No** (Zero-shot application to pre-trained TRMs) |
| **Compute Cost** | $O(D)$ | $O(K \times D)$ (Embarrassingly parallelizable) |

## Setup

```bash
# Clone with submodule
git clone --recurse-submodules https://github.com/your-repo/PTRM.git
cd PTRM

# Install dependencies
pip install -r requirements.txt

# Download model checkpoints
python scripts/download_models.py

# Download/build evaluation datasets
python scripts/download_data.py
```

## Quick Start

```python
from inference.checkpoint_loader import load_model_from_manifest
from inference.ptrm_inference import PTRMInference
from evaluation.metrics import compute_metrics_from_result, format_metrics

# Load model
model, meta = load_model_from_manifest("models/sudoku/manifest.yaml", device="cuda")
engine = PTRMInference(model, device="cuda")

# Run PTRM inference
result = engine.run(batch, K=25, D=16, sigma=0.3, seed=42)

# Evaluate
metrics = compute_metrics_from_result(result, labels)
print(format_metrics(metrics))
```

## Run All Benchmarks

```bash
# Run everything
./scripts/run_all_benchmarks.sh

# Run specific benchmark
./scripts/run_all_benchmarks.sh sudoku

# Download only (no inference)
./scripts/run_all_benchmarks.sh --download-only
```

## Paper Results (Table 1)

| Benchmark       | Model    | pass@25 | best-Q@25 | mode@25 |
|:----------------|:---------|--------:|----------:|--------:|
| Sudoku-Extreme  | TRM-MLP  |   92.3% |     65.3% |   63.8% |
| Maze-Hard 30×30 | TRM-Att  |   85.2% |     71.3% |   68.5% |
| ARC-AGI-2       | TRM-Att  |   35.4% |     24.1% |   22.7% |

## Project Structure

```
PTRM/
├── inference/
│   ├── checkpoint_loader.py   # Unified model loading from manifests
│   └── ptrm_inference.py      # Core Algorithm 1: K-parallel stochastic rollouts
├── evaluation/
│   ├── metrics.py             # pass@K, best-Q@K, mode@K metrics
│   ├── evaluate.py            # End-to-end evaluation runner
│   └── evaluators/
│       └── puzzle_evaluator.py # Dataset loading (Sudoku, Maze, ARC)
├── visualization/
│   ├── pca_latent.py          # PCA latent dynamics (Figure 1)
│   ├── q_tracking.py          # Q-value evolution across steps
│   ├── basin_escape.py        # Basin escape analysis
│   ├── width_scaling.py       # Accuracy vs K curves (Figure 2)
│   └── noise_ablation.py      # Accuracy vs σ sweep
├── config/
│   ├── config_loader.py       # YAML config loader
│   └── inference/             # Per-benchmark YAML configs
│       ├── sudoku_extreme.yaml
│       ├── maze_hard.yaml
│       └── arc_agi2.yaml
├── scripts/
│   ├── download_models.py     # Download HF checkpoints
│   ├── download_data.py       # Build evaluation datasets
│   └── run_all_benchmarks.sh  # Full pipeline runner
├── tests/                     # 161 tests across 7 test files
│   ├── test_checkpoint_loader.py   # 42 tests
│   ├── test_ptrm_inference.py      # 29 tests
│   ├── test_metrics.py             # 32 tests
│   ├── test_evaluation.py          # 15 tests
│   ├── test_download_data.py       # 16 tests
│   ├── test_configs.py             # 25 tests
│   └── test_visualization.py       # 22 tests
├── TinyRecursiveModels/       # Git submodule (original TRM codebase)
├── models/                    # Downloaded checkpoints (git-ignored)
├── data/                      # Built datasets (git-ignored)
└── requirements.txt
```

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific component
python -m pytest tests/test_ptrm_inference.py -v
```

## Visualization

```python
from visualization.pca_latent import plot_pca_latent_dynamics
from visualization.width_scaling import plot_width_scaling

# After running inference with collect_trajectories=True
result = engine.run(batch, K=25, D=16, sigma=0.3, collect_trajectories=True)

# Plot latent dynamics
plot_pca_latent_dynamics(
    result.latent_trajectories,
    result.all_q_values,
    output_path="plots/pca_dynamics.png"
)
```

## Citation

```bibtex
@article{ptrm2025,
  title={Probabilistic Tiny Recursive Models},
  author={...},
  journal={arXiv preprint arXiv:2605.19943},
  year={2025}
}
```

## License

This is a community implementation for research purposes. See the original [TinyRecursiveModels](https://github.com/SamsungSAILMontreal/TinyRecursiveModels) repository for the base model license.
