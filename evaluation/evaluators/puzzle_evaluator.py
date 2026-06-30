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
            data_path: Path to .npz file or directory containing split .npz files.
            split: Dataset split to load ('train', 'val', 'test').
            max_samples: If set, limit to first N samples.
        """
        if os.path.isdir(data_path):
            npz_path = os.path.join(data_path, f"{split}.npz")
        else:
            npz_path = data_path

        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"Dataset not found: {npz_path}")

        data = np.load(npz_path, allow_pickle=True)

        self.inputs = torch.from_numpy(data["inputs"]).long()
        self.labels = torch.from_numpy(data["labels"]).long()

        if "puzzle_identifiers" in data:
            self.puzzle_identifiers = torch.from_numpy(data["puzzle_identifiers"]).long()
        else:
            self.puzzle_identifiers = torch.zeros(len(self.inputs), dtype=torch.long)

        if max_samples is not None:
            self.inputs = self.inputs[:max_samples]
            self.labels = self.labels[:max_samples]
            self.puzzle_identifiers = self.puzzle_identifiers[:max_samples]

        # Load metadata if available
        self.metadata = {}
        if "metadata" in data:
            import json
            self.metadata = json.loads(str(data["metadata"]))

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
