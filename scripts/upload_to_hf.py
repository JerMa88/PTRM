"""
Upload best PPBench TRM-Att checkpoint to HuggingFace Hub.

Creates a model repo with:
  - model.pt: Best checkpoint (compile prefix stripped)
  - manifest.yaml: Architecture + dataset metadata for PTRM inference
  - config.yaml: Full training config
  - README.md: Model card with training details, paper reference, W&B link

Usage:
    python scripts/upload_to_hf.py \\
        --checkpoint-dir checkpoints/ptrm-ppbench-training/best/ \\
        --repo-name ptrm-ppbench-trm-att-7m \\
        --wandb-url https://wandb.ai/...

    python scripts/upload_to_hf.py --help
"""

import argparse
import os
import sys
import yaml
from datetime import datetime
from pathlib import Path

import torch
from huggingface_hub import HfApi, create_repo


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def strip_compile_prefix(state_dict: dict) -> dict:
    """Remove torch.compile '_orig_mod.' prefixes from state dict keys."""
    new_sd = {}
    for k, v in state_dict.items():
        new_key = k.replace("_orig_mod.", "")
        new_sd[new_key] = v
    return new_sd


def count_parameters(state_dict: dict) -> int:
    """Count total parameters from a state dict."""
    return sum(v.numel() for v in state_dict.values())


def create_manifest(checkpoint_dir: str, output_path: str) -> dict:
    """Create a manifest.yaml for PTRM inference loader compatibility."""
    # Load training config if available
    config_path = os.path.join(checkpoint_dir, "..", "all_config.yaml")
    if not os.path.exists(config_path):
        config_path = str(PROJECT_ROOT / "config" / "train" / "ppbench.yaml")

    arch_config = {
        "name": "recursive_reasoning.trm@TinyRecursiveReasoningModel_ACTV1",
        "H_cycles": 3,
        "L_cycles": 6,
        "H_layers": 0,
        "L_layers": 2,
        "hidden_size": 512,
        "num_heads": 8,
        "expansion": 4,
        "pos_encodings": "rope",
        "forward_dtype": "bfloat16",
        "mlp_t": False,  # Attention variant
        "puzzle_emb_len": 16,
        "puzzle_emb_ndim": 512,
        "halt_exploration_prob": 0.1,
        "halt_max_steps": 16,
        "no_ACT_continue": True,
        "loss": {
            "name": "losses@ACTLossHead",
            "loss_type": "stablemax_cross_entropy",
        },
    }

    dataset_meta = {
        "vocab_size": 294,
        "seq_len": 100,
        "pad_id": 0,
        "ignore_label_id": 0,
        "blank_identifier_id": 0,
        "num_puzzle_identifiers": 6,
    }

    manifest = {
        "benchmark": "ppbench",
        "model_type": "TRM-Att",
        "hf_repo": "",  # Will be filled in by caller
        "arch_config": arch_config,
        "dataset_meta": dataset_meta,
        "checkpoints": [
            {"local_filename": "model.pt", "description": "Best TRM-Att checkpoint"},
        ],
        "default_checkpoint": "model.pt",
    }

    with open(output_path, "w") as f:
        yaml.dump(manifest, f, default_flow_style=False, sort_keys=False)

    return manifest


def create_model_card(
    repo_name: str,
    num_params: int,
    best_accuracy: float,
    best_step: int,
    wandb_url: str = None,
    github_url: str = None,
    training_config: dict = None,
) -> str:
    """Generate a comprehensive HuggingFace model card README.md."""

    wandb_section = ""
    if wandb_url:
        wandb_section = f"""
## Training Logs

Full training curves, per-step metrics, and system utilization are available on W&B:

🔗 **[W&B Training Dashboard]({wandb_url})**

Logged metrics include:
- Every training step: loss, cell accuracy, learning rate, gradient norm, throughput
- Every evaluation: validation accuracy, per-puzzle-type accuracy
- System: GPU memory usage, utilization
- Q-head bias tracking (monitors halt-exploration dynamics)
"""

    github_section = ""
    if github_url:
        github_section = f"""
## Source Code

This model was trained using the PTRM community implementation:

🔗 **[GitHub Repository]({github_url})**
"""

    config_section = ""
    if training_config:
        config_yaml = yaml.dump(training_config, default_flow_style=False, sort_keys=False)
        config_section = f"""
## Training Configuration

```yaml
{config_yaml}
```
"""

    card = f"""---
language:
- en
license: gpl-3.0
tags:
- puzzle-solving
- recursive-model
- TRM
- PTRM
- pencil-puzzle
- constraint-satisfaction
library_name: pytorch
pipeline_tag: other
---

# TRM-Att PPBench ({num_params / 1e6:.1f}M)

**Pre-trained Tiny Recursive Model (Attention variant)** for 6 types of constraint-satisfaction pencil puzzles from the [Pencil Puzzle Bench](https://github.com/approximatelabs/pencil-puzzle-bench).

Trained as part of the **[Probabilistic Tiny Recursive Models (PTRM)](https://arxiv.org/abs/2605.19943)** community replication.

## Model Details

| Property | Value |
|:---------|:------|
| Architecture | TRM-Att (Tiny Recursive Reasoning Model, Attention variant) |
| Parameters | {num_params:,} |
| Hidden size | 512 |
| H-cycles | 3 |
| L-cycles | 6 |
| L-layers | 2 (Attention) |
| Positional encoding | RoPE |
| Precision | bfloat16 |
| Vocab size | 294 tokens |
| Sequence length | 100 (10×10 grid) |

## Training Data

Trained on **PPBench** (Pencil Puzzle Bench) — 6 puzzle types with the following splits:

| Puzzle Type | Train | Validation | Golden |
|:------------|------:|-----------:|-------:|
| Sudoku (9×9→10×10) | 7,810 | 97 | 15 |
| Lightup (10×10) | 9,504 | 65 | 8 |
| Nurikabe (10×10) | 15,180 | 55 | 9 |
| Heyawake (10×10) | 42,108 | 70 | 7 |
| Tapa (10×10) | 3,663 | 26 | 10 |
| Shakashaka (10×10) | 20,702 | 62 | 12 |
| **Total** | **98,967** | **375** | **61** |

### Data Augmentation
Each training puzzle is expanded into 10 examples:
1. Original (initial state → solved) pair
2. 9 augmented examples via trajectory sampling × dihedral transforms

## Evaluation Results

**Best validation accuracy: {best_accuracy:.4f}** (at training step {best_step:,})

### Paper Reference Results (arXiv:2605.19943, Table 5)

| Method | Sudoku | Lightup | Nurikabe | Heyawake | Tapa | Aggregate |
|:-------|-------:|--------:|---------:|---------:|-----:|----------:|
| TRM (K=1, D=16) | 68.7% | 83.3% | 76.0% | 96.7% | 39.7% | 76.4% |
| TRM (K=1, D=48) | 74.0% | 84.0% | 76.7% | 98.0% | 41.0% | 78.3% |
| PTRM best-Q@K (K=100, D=48) | 93.3% | 93.3% | 84.7% | 100% | 71.8% | 90.4% |

## How to Use

### With PTRM Inference
```python
from inference.checkpoint_loader import load_model_from_manifest
from inference.ptrm_inference import PTRMInference

# Load model
model, meta = load_model_from_manifest("models/ppbench/manifest.yaml", device="cuda")

# Create PTRM engine
engine = PTRMInference(model, device="cuda")

# Run PTRM inference (K=100 rollouts, D=48 steps, σ=0.2 noise)
result = engine.run(batch, K=100, D=48, sigma=0.2)
```

### Direct Download
```python
from huggingface_hub import hf_hub_download
checkpoint = hf_hub_download("{repo_name}", "model.pt")
```

## Citation

```bibtex
@article{{ptrm2025,
  title={{Probabilistic Tiny Recursive Models}},
  author={{...}},
  journal={{arXiv preprint arXiv:2605.19943}},
  year={{2025}}
}}

@article{{trm2025,
  title={{Less is More: Recursive Reasoning with Tiny Networks}},
  author={{Alexia Jolicoeur-Martineau}},
  journal={{arXiv preprint arXiv:2510.04871}},
  year={{2025}}
}}

@article{{ppbench2026,
  title={{Pencil Puzzle Bench: A Benchmark for Multi-Step Verifiable Reasoning}},
  author={{Justin Waugh}},
  journal={{arXiv preprint arXiv:2603.02119}},
  year={{2026}}
}}
```
{wandb_section}
{github_section}
{config_section}
"""
    return card


def upload_to_hf(
    checkpoint_dir: str,
    repo_name: str,
    wandb_url: str = None,
    github_url: str = None,
    private: bool = False,
):
    """
    Upload the best PPBench checkpoint to HuggingFace Hub.
    """
    api = HfApi()

    # Determine checkpoint path
    model_path = os.path.join(checkpoint_dir, "model.pt")
    if not os.path.exists(model_path):
        # Try to find any .pt file
        for f in os.listdir(checkpoint_dir):
            if f.endswith(".pt") or (not f.endswith(".yaml") and not f.endswith(".json")):
                model_path = os.path.join(checkpoint_dir, f)
                break

    if not os.path.exists(model_path):
        print(f"Error: No checkpoint found in {checkpoint_dir}")
        sys.exit(1)

    print(f"Loading checkpoint: {model_path}")
    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)

    # Strip compile prefixes
    state_dict = strip_compile_prefix(state_dict)
    num_params = count_parameters(state_dict)
    print(f"  Parameters: {num_params:,}")

    # Load best info if available
    best_info_path = os.path.join(checkpoint_dir, "best_info.yaml")
    best_accuracy = 0.0
    best_step = 0
    if os.path.exists(best_info_path):
        with open(best_info_path, "r") as f:
            best_info = yaml.safe_load(f)
        best_accuracy = best_info.get("best_accuracy", 0.0)
        best_step = best_info.get("best_step", 0)
        print(f"  Best accuracy: {best_accuracy:.4f} (step {best_step})")

    # Create temporary upload directory
    upload_dir = os.path.join(checkpoint_dir, "_upload")
    os.makedirs(upload_dir, exist_ok=True)

    # Save stripped checkpoint
    stripped_path = os.path.join(upload_dir, "model.pt")
    torch.save(state_dict, stripped_path)
    print(f"  Saved stripped checkpoint: {stripped_path}")

    # Create manifest
    manifest_path = os.path.join(upload_dir, "manifest.yaml")
    manifest = create_manifest(checkpoint_dir, manifest_path)
    manifest["hf_repo"] = repo_name

    with open(manifest_path, "w") as f:
        yaml.dump(manifest, f, default_flow_style=False, sort_keys=False)

    # Create model card
    card_content = create_model_card(
        repo_name=repo_name,
        num_params=num_params,
        best_accuracy=best_accuracy,
        best_step=best_step,
        wandb_url=wandb_url,
        github_url=github_url,
    )
    readme_path = os.path.join(upload_dir, "README.md")
    with open(readme_path, "w") as f:
        f.write(card_content)

    # Create HF repo
    print(f"\nCreating HuggingFace repo: {repo_name}")
    try:
        create_repo(repo_name, private=private, exist_ok=True)
    except Exception as e:
        print(f"  Repo creation: {e}")

    # Upload all files
    print("Uploading files...")
    api.upload_folder(
        folder_path=upload_dir,
        repo_id=repo_name,
        commit_message=f"Upload PPBench TRM-Att best checkpoint (acc={best_accuracy:.4f}, step={best_step})",
    )

    print(f"\n✅ Upload complete!")
    print(f"   Repo: https://huggingface.co/{repo_name}")
    print(f"   Model: {num_params:,} parameters")
    print(f"   Best accuracy: {best_accuracy:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Upload best PPBench TRM-Att checkpoint to HuggingFace"
    )
    parser.add_argument(
        "--checkpoint-dir", required=True,
        help="Path to directory containing the best checkpoint"
    )
    parser.add_argument(
        "--repo-name", required=True,
        help="HuggingFace repo name (e.g., username/ptrm-ppbench-trm-att-7m)"
    )
    parser.add_argument(
        "--wandb-url", default=None,
        help="URL to the W&B training run"
    )
    parser.add_argument(
        "--github-url", default=None,
        help="URL to the GitHub repository"
    )
    parser.add_argument(
        "--private", action="store_true",
        help="Create a private HuggingFace repo"
    )
    args = parser.parse_args()

    upload_to_hf(
        checkpoint_dir=args.checkpoint_dir,
        repo_name=args.repo_name,
        wandb_url=args.wandb_url,
        github_url=args.github_url,
        private=args.private,
    )


if __name__ == "__main__":
    main()
