"""
Puzzle evaluator for grid-based benchmarks (Sudoku, Maze).

Handles the data format used by TRM's puzzle datasets:
  - inputs: (seq_len,) flattened grid with clue cells filled, blanks as blank_id
  - labels: (seq_len,) full solution grid
  - puzzle_identifiers: scalar puzzle group ID

For Sudoku: 9x9 grid, vocab {0=PAD, 1-9=digits}, blank_id=0
For Maze: 30x30 grid, vocab {0=PAD, 1-4=maze chars}, blank_id=0
"""

import os
import sys
from typing import Optional

import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset


class PuzzleDataset(Dataset):
    """
    Generic puzzle dataset loader for .npz files produced by TRM's
    dataset builders (build_sudoku_dataset.py, build_maze_dataset.py).

    Expected .npz format:
      - inputs: (N, seq_len) int array
      - labels: (N, seq_len) int array
      - puzzle_identifiers: (N,) int array
      - metadata: JSON string with vocab_size, seq_len, etc.
    """

    def __init__(self, data_path: str, split: str = "test", max_samples: Optional[int] = None):
        """
        Args:
            data_path: Path to .npz file or directory containing either:
                       - split .npz files (e.g., test.npz)
                       - TRM-format .npy directories (e.g., test/all__inputs.npy)
            split: Dataset split to load ('train', 'val', 'test').
            max_samples: If set, limit to first N samples.
        """
        loaded = False

        if os.path.isfile(data_path) and data_path.endswith(".npz"):
            # Direct .npz file path
            self._load_from_npz(data_path)
            loaded = True
        elif os.path.isdir(data_path):
            # Try .npz first, then TRM .npy format
            npz_path = os.path.join(data_path, f"{split}.npz")
            npy_dir = os.path.join(data_path, split)

            if os.path.exists(npz_path):
                self._load_from_npz(npz_path)
                loaded = True
            elif os.path.isdir(npy_dir) and os.path.exists(os.path.join(npy_dir, "all__inputs.npy")):
                self._load_from_npy_dir(npy_dir)
                loaded = True

        if not loaded:
            if os.path.isdir(data_path):
                raise FileNotFoundError(
                    f"No dataset found for split '{split}' in {data_path}. "
                    f"Expected either {data_path}/{split}.npz or {data_path}/{split}/all__inputs.npy"
                )
            else:
                raise FileNotFoundError(f"Dataset not found: {data_path}")

        if max_samples is not None:
            self.inputs = self.inputs[:max_samples]
            self.labels = self.labels[:max_samples]
            self.puzzle_identifiers = self.puzzle_identifiers[:max_samples]

    def _load_from_npz(self, npz_path: str):
        """Load from a single .npz file."""
        data = np.load(npz_path, allow_pickle=True)

        self.inputs = torch.from_numpy(data["inputs"]).long()
        self.labels = torch.from_numpy(data["labels"]).long()

        if "puzzle_identifiers" in data:
            self.puzzle_identifiers = torch.from_numpy(data["puzzle_identifiers"]).long()
        else:
            self.puzzle_identifiers = torch.zeros(len(self.inputs), dtype=torch.long)

        self.metadata = {}
        if "metadata" in data:
            import json
            self.metadata = json.loads(str(data["metadata"]))

    def _load_from_npy_dir(self, npy_dir: str):
        """Load from TRM's .npy directory format (all__inputs.npy, etc.)."""
        self.inputs = torch.from_numpy(np.load(os.path.join(npy_dir, "all__inputs.npy"))).long()
        self.labels = torch.from_numpy(np.load(os.path.join(npy_dir, "all__labels.npy"))).long()

        pi_path = os.path.join(npy_dir, "all__puzzle_identifiers.npy")
        if os.path.exists(pi_path):
            self.puzzle_identifiers = torch.from_numpy(np.load(pi_path)).long()
        else:
            self.puzzle_identifiers = torch.zeros(len(self.inputs), dtype=torch.long)

        self.metadata = {}
        meta_path = os.path.join(npy_dir, "dataset.json")
        if os.path.exists(meta_path):
            import json
            with open(meta_path, "r") as f:
                self.metadata = json.load(f)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return {
            "inputs": self.inputs[idx],
            "labels": self.labels[idx],
            "puzzle_identifiers": self.puzzle_identifiers[idx],
        }


def create_puzzle_dataloader(
    data_path: str,
    split: str = "test",
    batch_size: int = 32,
    max_samples: Optional[int] = None,
    num_workers: int = 0,
) -> DataLoader:
    """
    Create a DataLoader for a puzzle dataset.

    Args:
        data_path: Path to .npz file or directory.
        split: Dataset split.
        batch_size: Batch size for evaluation.
        max_samples: Limit number of samples.
        num_workers: Number of data loading workers.

    Returns:
        DataLoader yielding batches of {inputs, labels, puzzle_identifiers}.
    """
    dataset = PuzzleDataset(data_path, split=split, max_samples=max_samples)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )
