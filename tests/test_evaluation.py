"""
Tests for Component 3 (continued): Puzzle evaluator and evaluation runner.

Test categories:
  1. PuzzleDataset — loading .npz files, metadata, max_samples
  2. create_puzzle_dataloader — DataLoader creation and batching
  3. EvaluationConfig — config defaults and validation
  4. run_evaluation integration — synthetic end-to-end test
"""

import os
import sys
import json

import pytest
import torch
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from evaluation.evaluators.puzzle_evaluator import PuzzleDataset, create_puzzle_dataloader
from evaluation.evaluate import EvaluationConfig


# =============================================================================
# Helper: create a synthetic .npz dataset
# =============================================================================

def create_synthetic_npz(path: str, N: int = 20, seq_len: int = 81, vocab_size: int = 11):
    """Create a minimal synthetic puzzle dataset in .npz format."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    inputs = np.random.randint(0, vocab_size, (N, seq_len)).astype(np.int64)
    labels = np.random.randint(0, vocab_size, (N, seq_len)).astype(np.int64)
    puzzle_identifiers = np.zeros(N, dtype=np.int64)
    metadata = json.dumps({
        "vocab_size": vocab_size,
        "seq_len": seq_len,
        "num_puzzle_identifiers": 1,
    })
    np.savez(
        path,
        inputs=inputs,
        labels=labels,
        puzzle_identifiers=puzzle_identifiers,
        metadata=np.array(metadata),
    )
    return path


# =============================================================================
# 1. PuzzleDataset tests
# =============================================================================

class TestPuzzleDataset:

    def test_load_from_file(self, tmp_path):
        npz_path = create_synthetic_npz(str(tmp_path / "test.npz"), N=10, seq_len=81)
        dataset = PuzzleDataset(npz_path)

        assert len(dataset) == 10
        sample = dataset[0]
        assert "inputs" in sample
        assert "labels" in sample
        assert "puzzle_identifiers" in sample
        assert sample["inputs"].shape == (81,)
        assert sample["labels"].shape == (81,)

    def test_load_from_directory(self, tmp_path):
        create_synthetic_npz(str(tmp_path / "test.npz"), N=15, seq_len=81)
        dataset = PuzzleDataset(str(tmp_path), split="test")
        assert len(dataset) == 15

    def test_max_samples(self, tmp_path):
        npz_path = create_synthetic_npz(str(tmp_path / "test.npz"), N=20, seq_len=81)
        dataset = PuzzleDataset(npz_path, max_samples=5)
        assert len(dataset) == 5

    def test_max_samples_larger_than_dataset(self, tmp_path):
        npz_path = create_synthetic_npz(str(tmp_path / "test.npz"), N=5, seq_len=81)
        dataset = PuzzleDataset(npz_path, max_samples=100)
        assert len(dataset) == 5

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            PuzzleDataset("/nonexistent/path/test.npz")

    def test_missing_split_raises(self, tmp_path):
        create_synthetic_npz(str(tmp_path / "test.npz"))
        with pytest.raises(FileNotFoundError):
            PuzzleDataset(str(tmp_path), split="train")

    def test_metadata_loaded(self, tmp_path):
        npz_path = create_synthetic_npz(str(tmp_path / "test.npz"), vocab_size=11)
        dataset = PuzzleDataset(npz_path)
        assert dataset.metadata["vocab_size"] == 11

    def test_no_puzzle_identifiers_defaults_to_zeros(self, tmp_path):
        """Test dataset without puzzle_identifiers key."""
        path = str(tmp_path / "test.npz")
        N, seq_len = 5, 10
        np.savez(
            path,
            inputs=np.random.randint(0, 5, (N, seq_len)),
            labels=np.random.randint(0, 5, (N, seq_len)),
        )
        dataset = PuzzleDataset(path)
        assert torch.all(dataset.puzzle_identifiers == 0)

    def test_tensor_dtypes(self, tmp_path):
        npz_path = create_synthetic_npz(str(tmp_path / "test.npz"))
        dataset = PuzzleDataset(npz_path)
        sample = dataset[0]
        assert sample["inputs"].dtype == torch.long
        assert sample["labels"].dtype == torch.long
        assert sample["puzzle_identifiers"].dtype == torch.long


# =============================================================================
# 2. create_puzzle_dataloader tests
# =============================================================================

class TestCreatePuzzleDataloader:

    def test_creates_dataloader(self, tmp_path):
        create_synthetic_npz(str(tmp_path / "test.npz"), N=10)
        dl = create_puzzle_dataloader(str(tmp_path), split="test", batch_size=4)

        assert len(dl.dataset) == 10
        batch = next(iter(dl))
        assert batch["inputs"].shape[0] == 4

    def test_batch_size_larger_than_dataset(self, tmp_path):
        create_synthetic_npz(str(tmp_path / "test.npz"), N=3)
        dl = create_puzzle_dataloader(str(tmp_path), split="test", batch_size=10)

        batch = next(iter(dl))
        assert batch["inputs"].shape[0] == 3  # No drop_last, so partial batch

    def test_max_samples_in_dataloader(self, tmp_path):
        create_synthetic_npz(str(tmp_path / "test.npz"), N=20)
        dl = create_puzzle_dataloader(str(tmp_path), split="test", batch_size=4, max_samples=8)

        assert len(dl.dataset) == 8
        total_samples = sum(batch["inputs"].shape[0] for batch in dl)
        assert total_samples == 8

    def test_not_shuffled(self, tmp_path):
        create_synthetic_npz(str(tmp_path / "test.npz"), N=10)
        dl = create_puzzle_dataloader(str(tmp_path), split="test", batch_size=10)

        batch = next(iter(dl))
        # Load raw data for comparison
        dataset = PuzzleDataset(str(tmp_path), split="test")
        assert torch.equal(batch["inputs"][0], dataset[0]["inputs"])
        assert torch.equal(batch["inputs"][-1], dataset[len(dataset) - 1]["inputs"])


# =============================================================================
# 3. EvaluationConfig tests
# =============================================================================

class TestEvaluationConfig:

    def test_default_values(self):
        config = EvaluationConfig(manifest_path="test.yaml")
        assert config.K == 25
        assert config.D == 16
        assert config.sigma == 0.3
        assert config.batch_size == 32
        assert config.split == "test"
        assert config.ignore_id == 0

    def test_custom_values(self):
        config = EvaluationConfig(
            manifest_path="test.yaml",
            K=100, D=64, sigma=0.5,
            batch_size=16, seed=42,
        )
        assert config.K == 100
        assert config.D == 64
        assert config.sigma == 0.5
        assert config.seed == 42


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
