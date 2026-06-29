"""
Download pre-trained TRM checkpoints from HuggingFace Hub.

Supports three benchmark models:
  - Sudoku-Extreme (TRM-MLP): alphaXiv/trm-model-sudoku
  - Maze-Hard (TRM-Att):      Sanjin2024/TinyRecursiveModel-Maze-Hard
  - ARC-AGI (TRM-Att):        arcprize/trm_arc_prize_verification (v1 + v2)

Usage:
    python scripts/download_models.py                  # Download all models
    python scripts/download_models.py --benchmarks sudoku maze  # Selective
    python scripts/download_models.py --output-dir ./my_models  # Custom dir
"""

import argparse
import os
import sys
import yaml

from huggingface_hub import hf_hub_download


# Registry of available benchmarks and their HuggingFace sources.
# Each entry maps a benchmark name to its HF repo, the files to download,
# and the architecture config embedded in the checkpoint's all_config.yaml
# (or inferred from the original TRM README).
MODEL_REGISTRY = {
    "sudoku": {
        "hf_repo": "alphaXiv/trm-model-sudoku",
        "files": [
            # This repo has 4 checkpoint files (2 MLP, 2 Att variants at different epochs).
            # The best MLP model is step_39060_sudoku_60k_epoch_attn_type (confusingly named;
            # it's the MLP model trained longer). We download both MLP checkpoints.
            {"hf_filename": "step_39060_sudoku_60k_epoch_attn_type", "local_filename": "model_mlp.pt"},
            {"hf_filename": "step_39060_sudoku_epoch_60k", "local_filename": "model_att.pt"},
        ],
        # Architecture config for the TRM-MLP variant (matches paper: 5M params)
        "arch_config": {
            "name": "recursive_reasoning.trm@TinyRecursiveReasoningModel_ACTV1",
            "H_cycles": 3,
            "L_cycles": 6,
            "H_layers": 0,
            "L_layers": 2,
            "hidden_size": 512,
            "num_heads": 8,
            "expansion": 4,
            "pos_encodings": "none",
            "forward_dtype": "bfloat16",
            "mlp_t": True,
            "puzzle_emb_len": 16,
            "puzzle_emb_ndim": 512,
            "halt_exploration_prob": 0.1,
            "halt_max_steps": 16,
            "no_ACT_continue": True,
            "loss": {"name": "losses@ACTLossHead", "loss_type": "stablemax_cross_entropy"},
        },
        # Dataset metadata needed to instantiate the model
        "dataset_meta": {
            "vocab_size": 11,   # PAD + digits 0-9
            "seq_len": 81,      # 9x9 grid
            "num_puzzle_identifiers": 1,
            "pad_id": 0,
            "ignore_label_id": 0,
            "blank_identifier_id": 0,
        },
        "default_checkpoint": "model_mlp.pt",
    },
    "maze": {
        "hf_repo": "Sanjin2024/TinyRecursiveModel-Maze-Hard",
        "files": [
            {"hf_filename": "step_9765", "local_filename": "model.pt"},
            {"hf_filename": "all_config.yaml", "local_filename": "all_config.yaml"},
        ],
        "arch_config": {
            "name": "recursive_reasoning.trm@TinyRecursiveReasoningModel_ACTV1",
            "H_cycles": 3,
            "L_cycles": 4,
            "H_layers": 0,
            "L_layers": 2,
            "hidden_size": 512,
            "num_heads": 8,
            "expansion": 4,
            "pos_encodings": "rope",
            "forward_dtype": "bfloat16",
            "mlp_t": False,
            "puzzle_emb_len": 16,
            "puzzle_emb_ndim": 512,
            "halt_exploration_prob": 0.1,
            "halt_max_steps": 16,
            "no_ACT_continue": True,
            "loss": {"name": "losses@ACTLossHead", "loss_type": "stablemax_cross_entropy"},
        },
        "dataset_meta": {
            "vocab_size": 5,    # PAD + 4 maze chars (# SGo)
            "seq_len": 900,     # 30x30 grid
            "num_puzzle_identifiers": 1,
            "pad_id": 0,
            "ignore_label_id": 0,
            "blank_identifier_id": 0,
        },
        "default_checkpoint": "model.pt",
    },
    "arc_v1": {
        "hf_repo": "arcprize/trm_arc_prize_verification",
        "files": [
            {"hf_filename": "arc_v1_public/step_518071", "local_filename": "model.pt"},
            {"hf_filename": "arc_v1_public/all_config.yaml", "local_filename": "all_config.yaml"},
        ],
        "arch_config": {
            "name": "recursive_reasoning.trm@TinyRecursiveReasoningModel_ACTV1",
            "H_cycles": 3,
            "L_cycles": 4,
            "H_layers": 0,
            "L_layers": 2,
            "hidden_size": 512,
            "num_heads": 8,
            "expansion": 4,
            "pos_encodings": "rope",
            "forward_dtype": "bfloat16",
            "mlp_t": False,
            "puzzle_emb_len": 16,
            "puzzle_emb_ndim": 512,
            "halt_exploration_prob": 0.1,
            "halt_max_steps": 16,
            "no_ACT_continue": True,
            "loss": {"name": "losses@ACTLossHead", "loss_type": "stablemax_cross_entropy"},
        },
        "dataset_meta": {
            "vocab_size": 12,   # PAD + EOS + 10 ARC colors
            "seq_len": 900,     # 30x30 grid
            "num_puzzle_identifiers": 1,  # Will be expanded by puzzle_emb
            "pad_id": 0,
            "ignore_label_id": 0,
            "blank_identifier_id": 0,
        },
        "default_checkpoint": "model.pt",
    },
    "arc_v2": {
        "hf_repo": "arcprize/trm_arc_prize_verification",
        "files": [
            {"hf_filename": "arc_v2_public/step_723914", "local_filename": "model.pt"},
            {"hf_filename": "arc_v2_public/all_config.yaml", "local_filename": "all_config.yaml"},
        ],
        "arch_config": {
            "name": "recursive_reasoning.trm@TinyRecursiveReasoningModel_ACTV1",
            "H_cycles": 3,
            "L_cycles": 4,
            "H_layers": 0,
            "L_layers": 2,
            "hidden_size": 512,
            "num_heads": 8,
            "expansion": 4,
            "pos_encodings": "rope",
            "forward_dtype": "bfloat16",
            "mlp_t": False,
            "puzzle_emb_len": 16,
            "puzzle_emb_ndim": 512,
            "halt_exploration_prob": 0.1,
            "halt_max_steps": 16,
            "no_ACT_continue": True,
            "loss": {"name": "losses@ACTLossHead", "loss_type": "stablemax_cross_entropy"},
        },
        "dataset_meta": {
            "vocab_size": 12,
            "seq_len": 900,
            "num_puzzle_identifiers": 1,
            "pad_id": 0,
            "ignore_label_id": 0,
            "blank_identifier_id": 0,
        },
        "default_checkpoint": "model.pt",
    },
}


def download_benchmark(benchmark_name: str, output_dir: str, force: bool = False) -> str:
    """
    Download a single benchmark's TRM checkpoint from HuggingFace.

    Args:
        benchmark_name: Key in MODEL_REGISTRY (e.g. 'sudoku', 'maze', 'arc_v2').
        output_dir: Root directory for all models (e.g. './models').
        force: If True, re-download even if files already exist.

    Returns:
        Path to the benchmark's model directory.
    """
    if benchmark_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown benchmark '{benchmark_name}'. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )

    entry = MODEL_REGISTRY[benchmark_name]
    model_dir = os.path.join(output_dir, benchmark_name)
    os.makedirs(model_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Downloading {benchmark_name} from {entry['hf_repo']}")
    print(f"  Target: {model_dir}")
    print(f"{'='*60}")

    for file_info in entry["files"]:
        local_path = os.path.join(model_dir, file_info["local_filename"])

        if os.path.exists(local_path) and not force:
            print(f"  [SKIP] {file_info['local_filename']} (already exists)")
            continue

        print(f"  [DOWNLOAD] {file_info['hf_filename']} -> {file_info['local_filename']}")
        downloaded_path = hf_hub_download(
            repo_id=entry["hf_repo"],
            filename=file_info["hf_filename"],
            local_dir=model_dir,
        )

        # hf_hub_download may place the file in a subdirectory; move it to the expected location
        if os.path.abspath(downloaded_path) != os.path.abspath(local_path):
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            os.rename(downloaded_path, local_path)
            # Clean up any empty subdirectories left behind
            _cleanup_empty_dirs(model_dir)

    # Save arch_config and dataset_meta as a manifest for the checkpoint loader
    manifest_path = os.path.join(model_dir, "manifest.yaml")
    manifest = {
        "benchmark": benchmark_name,
        "hf_repo": entry["hf_repo"],
        "arch_config": entry["arch_config"],
        "dataset_meta": entry["dataset_meta"],
        "default_checkpoint": entry["default_checkpoint"],
    }
    with open(manifest_path, "w") as f:
        yaml.dump(manifest, f, default_flow_style=False, sort_keys=False)
    print(f"  [SAVED] manifest.yaml")

    return model_dir


def _cleanup_empty_dirs(root: str):
    """Remove empty subdirectories created by hf_hub_download."""
    for dirpath, _dirnames, _filenames in os.walk(root, topdown=False):
        if dirpath == root:
            continue
        # Check actual contents at removal time (os.walk pre-computes listings,
        # so after removing children the parent may now be empty too).
        try:
            if not os.listdir(dirpath):
                os.rmdir(dirpath)
        except OSError:
            pass


def download_all(output_dir: str, benchmarks: list[str] | None = None, force: bool = False):
    """
    Download all (or selected) benchmark checkpoints.

    Args:
        output_dir: Root directory for all models.
        benchmarks: List of benchmark names to download, or None for all.
        force: If True, re-download even if files exist.
    """
    targets = benchmarks if benchmarks else list(MODEL_REGISTRY.keys())

    for name in targets:
        try:
            download_benchmark(name, output_dir, force=force)
        except Exception as e:
            print(f"\n  [ERROR] Failed to download {name}: {e}", file=sys.stderr)
            continue

    print(f"\n{'='*60}")
    print(f"Download complete. Models saved to: {output_dir}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Download pre-trained TRM checkpoints for PTRM inference."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help="Root directory for downloaded models (default: models/)",
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        choices=list(MODEL_REGISTRY.keys()),
        default=None,
        help="Specific benchmarks to download (default: all).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if files already exist.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        dest="list_models",
        help="List available benchmarks and exit.",
    )

    args = parser.parse_args()

    if args.list_models:
        print("Available benchmarks:")
        for name, entry in MODEL_REGISTRY.items():
            arch_type = "TRM-MLP" if entry["arch_config"]["mlp_t"] else "TRM-Att"
            print(f"  {name:12s}  {arch_type:8s}  {entry['hf_repo']}")
        return

    download_all(args.output_dir, args.benchmarks, args.force)


if __name__ == "__main__":
    main()
