"""
Download and build evaluation datasets for PTRM benchmarks.

This script wraps the TinyRecursiveModels submodule's dataset builders
to produce ready-to-evaluate test sets for each benchmark.

Supported datasets:
  - sudoku: Sudoku-Extreme test set from sapientinc/sudoku-extreme (HuggingFace)
  - maze:   Maze-Hard 30x30 from Sanjin2024/Maze-Hard-30x30 (HuggingFace)
  - arc:    ARC-AGI from fchollet/ARC-AGI (GitHub)

Usage:
    python scripts/download_data.py                     # Build all datasets
    python scripts/download_data.py --datasets sudoku    # Selective
    python scripts/download_data.py --output-dir ./data  # Custom dir
"""

import argparse
import json
import os
import sys
import subprocess

import numpy as np


# Path to TRM submodule
TRM_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "TinyRecursiveModels"))


# =============================================================================
# Dataset builders
# =============================================================================

def build_sudoku_dataset(output_dir: str, force: bool = False) -> str:
    """
    Build Sudoku-Extreme evaluation dataset.

    Downloads from HuggingFace (sapientinc/sudoku-extreme) and converts to
    TRM .npy format using the submodule's build_sudoku_dataset.py.

    Only builds the test set (training data not needed for PTRM inference).
    """
    dataset_dir = os.path.join(output_dir, "sudoku")
    test_dir = os.path.join(dataset_dir, "test")
    marker = os.path.join(test_dir, "dataset.json")

    if os.path.exists(marker) and not force:
        print(f"  [SKIP] Sudoku dataset already exists at {dataset_dir}")
        return dataset_dir

    print(f"  [BUILD] Sudoku-Extreme test set -> {dataset_dir}")
    os.makedirs(dataset_dir, exist_ok=True)

    # Use TRM's dataset builder
    cmd = [
        sys.executable,
        os.path.join(TRM_ROOT, "dataset", "build_sudoku_dataset.py"),
        "--source-repo", "sapientinc/sudoku-extreme",
        "--output-dir", dataset_dir,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=TRM_ROOT)

    if result.returncode != 0:
        print(f"  [ERROR] Sudoku build failed: {result.stderr}")
        raise RuntimeError(f"Sudoku dataset build failed:\n{result.stderr}")

    print(f"  [OK] Sudoku dataset built at {dataset_dir}")
    return dataset_dir


def build_maze_dataset(output_dir: str, force: bool = False) -> str:
    """
    Build Maze-Hard 30x30 evaluation dataset.

    Downloads from HuggingFace (Sanjin2024/Maze-Hard-30x30) and converts.
    """
    dataset_dir = os.path.join(output_dir, "maze")
    test_dir = os.path.join(dataset_dir, "test")
    marker = os.path.join(test_dir, "dataset.json")

    if os.path.exists(marker) and not force:
        print(f"  [SKIP] Maze dataset already exists at {dataset_dir}")
        return dataset_dir

    print(f"  [BUILD] Maze-Hard 30x30 test set -> {dataset_dir}")
    os.makedirs(dataset_dir, exist_ok=True)

    cmd = [
        sys.executable,
        os.path.join(TRM_ROOT, "dataset", "build_maze_dataset.py"),
        "--source-repo", "Sanjin2024/Maze-Hard-30x30",
        "--output-dir", dataset_dir,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=TRM_ROOT)

    if result.returncode != 0:
        print(f"  [ERROR] Maze build failed: {result.stderr}")
        raise RuntimeError(f"Maze dataset build failed:\n{result.stderr}")

    print(f"  [OK] Maze dataset built at {dataset_dir}")
    return dataset_dir


def build_arc_dataset(output_dir: str, force: bool = False) -> str:
    """
    Build ARC-AGI evaluation dataset.

    Downloads from GitHub (fchollet/ARC-AGI) and converts.
    """
    dataset_dir = os.path.join(output_dir, "arc")
    test_dir = os.path.join(dataset_dir, "test")
    marker = os.path.join(test_dir, "dataset.json")

    if os.path.exists(marker) and not force:
        print(f"  [SKIP] ARC dataset already exists at {dataset_dir}")
        return dataset_dir

    print(f"  [BUILD] ARC-AGI dataset -> {dataset_dir}")
    os.makedirs(dataset_dir, exist_ok=True)

    # ARC dataset builder requires input_file_prefix pointing to the ARC JSON tasks
    # We need to clone/download the ARC repo first
    arc_repo_dir = os.path.join(output_dir, "_arc_raw")
    if not os.path.isdir(os.path.join(arc_repo_dir, "data")):
        print(f"  [CLONE] Cloning ARC-AGI repository...")
        clone_result = subprocess.run(
            ["git", "clone", "--depth", "1", "https://github.com/fchollet/ARC-AGI.git", arc_repo_dir],
            capture_output=True, text=True,
        )
        if clone_result.returncode != 0:
            raise RuntimeError(f"ARC clone failed:\n{clone_result.stderr}")

    cmd = [
        sys.executable,
        os.path.join(TRM_ROOT, "dataset", "build_arc_dataset.py"),
        "--input-file-prefix", os.path.join(arc_repo_dir, "data"),
        "--output-dir", dataset_dir,
        "--subsets", "training", "evaluation",
        "--test-set-name", "evaluation",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=TRM_ROOT)

    if result.returncode != 0:
        print(f"  [ERROR] ARC build failed: {result.stderr}")
        raise RuntimeError(f"ARC dataset build failed:\n{result.stderr}")

    print(f"  [OK] ARC dataset built at {dataset_dir}")
    return dataset_dir


# =============================================================================
# Dataset loading utilities
# =============================================================================

def load_trm_dataset(dataset_dir: str, split: str = "test"):
    """
    Load a TRM-format dataset from .npy files.

    TRM stores each field as a separate .npy file in:
        dataset_dir/split/all__inputs.npy
        dataset_dir/split/all__labels.npy
        dataset_dir/split/all__puzzle_identifiers.npy

    Returns a dict with numpy arrays.
    """
    split_dir = os.path.join(dataset_dir, split)
    if not os.path.isdir(split_dir):
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    data = {}
    for field in ["inputs", "labels", "puzzle_identifiers"]:
        path = os.path.join(split_dir, f"all__{field}.npy")
        if os.path.exists(path):
            data[field] = np.load(path)

    if "inputs" not in data:
        raise FileNotFoundError(f"No inputs found in {split_dir}")

    # Load metadata if available
    meta_path = os.path.join(split_dir, "dataset.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            data["metadata"] = json.load(f)

    return data


def convert_trm_to_npz(dataset_dir: str, split: str = "test", output_path: str | None = None):
    """
    Convert TRM .npy format to a single .npz file for use with PuzzleDataset.

    Args:
        dataset_dir: Directory containing the TRM-format dataset.
        split: Split to convert.
        output_path: Where to save the .npz file. Defaults to dataset_dir/split.npz.
    """
    data = load_trm_dataset(dataset_dir, split)

    if output_path is None:
        output_path = os.path.join(dataset_dir, f"{split}.npz")

    save_dict = {}
    for key in ["inputs", "labels", "puzzle_identifiers"]:
        if key in data:
            save_dict[key] = data[key]

    if "metadata" in data:
        save_dict["metadata"] = np.array(json.dumps(data["metadata"]))

    np.savez(output_path, **save_dict)
    print(f"  [SAVED] {output_path} ({save_dict['inputs'].shape[0]} samples)")

    return output_path


# =============================================================================
# Registry and CLI
# =============================================================================

DATASET_REGISTRY = {
    "sudoku": build_sudoku_dataset,
    "maze": build_maze_dataset,
    "arc": build_arc_dataset,
}


def download_all(output_dir: str, datasets: list[str] | None = None, force: bool = False):
    """Download and build all (or selected) datasets."""
    targets = datasets if datasets else list(DATASET_REGISTRY.keys())

    for name in targets:
        if name not in DATASET_REGISTRY:
            print(f"  [ERROR] Unknown dataset: {name}")
            continue

        print(f"\n{'='*60}")
        print(f"Building {name} dataset")
        print(f"{'='*60}")

        try:
            dataset_dir = DATASET_REGISTRY[name](output_dir, force=force)
            # Convert to .npz for convenience
            for split in ["test", "train"]:
                split_dir = os.path.join(dataset_dir, split)
                if os.path.isdir(split_dir):
                    npz_path = os.path.join(dataset_dir, f"{split}.npz")
                    if not os.path.exists(npz_path) or force:
                        convert_trm_to_npz(dataset_dir, split)
        except Exception as e:
            print(f"  [ERROR] Failed to build {name}: {e}")
            continue

    print(f"\n{'='*60}")
    print(f"Dataset build complete. Data saved to: {output_dir}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Download and build evaluation datasets for PTRM benchmarks."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data",
        help="Root directory for datasets (default: data/)",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=list(DATASET_REGISTRY.keys()),
        default=None,
        help="Specific datasets to build (default: all).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force rebuild even if datasets already exist.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        dest="list_datasets",
        help="List available datasets and exit.",
    )

    args = parser.parse_args()

    if args.list_datasets:
        print("Available datasets:")
        for name in DATASET_REGISTRY:
            print(f"  {name}")
        return

    download_all(args.output_dir, args.datasets, args.force)


if __name__ == "__main__":
    main()
