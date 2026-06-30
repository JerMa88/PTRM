"""
Tests for Component 4: Dataset Download Scripts & TRM Format Support.

Test categories:
  1. load_trm_dataset — loading .npy directory format
  2. convert_trm_to_npz — .npy to .npz conversion
  3. PuzzleDataset TRM format — loading from .npy directories
  4. DATASET_REGISTRY — registry validation
"""

import os
import sys
import json

import pytest
import torch
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from scripts.download_data import (
    load_trm_dataset,
    convert_trm_to_npz,
    DATASET_REGISTRY,
)
from evaluation.evaluators.puzzle_evaluator import PuzzleDataset


# =============================================================================
# Helper: create synthetic TRM-format .npy directory
# =============================================================================

def create_trm_npy_dir(base_dir: str, split: str = "test", N: int = 10, seq_len: int = 81):
    """Create a minimal TRM-format .npy dataset directory."""
    split_dir = os.path.join(base_dir, split)
    os.makedirs(split_dir, exist_ok=True)

    inputs = np.random.randint(0, 10, (N, seq_len)).astype(np.int32)
    labels = np.random.randint(1, 10, (N, seq_len)).astype(np.int32)
    puzzle_ids = np.zeros(N, dtype=np.int32)

    np.save(os.path.join(split_dir, "all__inputs.npy"), inputs)
    np.save(os.path.join(split_dir, "all__labels.npy"), labels)
    np.save(os.path.join(split_dir, "all__puzzle_identifiers.npy"), puzzle_ids)

    metadata = {
        "vocab_size": 11,
        "seq_len": seq_len,
        "num_puzzle_identifiers": 1,
        "pad_id": 0,
        "ignore_label_id": 0,
        "blank_identifier_id": 0,
        "total_groups": N,
        "mean_puzzle_examples": 1,
        "total_puzzles": N,
        "sets": ["all"],
    }
    with open(os.path.join(split_dir, "dataset.json"), "w") as f:
        json.dump(metadata, f)

    return base_dir


# =============================================================================
# 1. load_trm_dataset tests
# =============================================================================

class TestLoadTrmDataset:

    def test_loads_npy_files(self, tmp_path):
        create_trm_npy_dir(str(tmp_path), split="test", N=10, seq_len=81)
        data = load_trm_dataset(str(tmp_path), split="test")

        assert "inputs" in data
        assert "labels" in data
        assert "puzzle_identifiers" in data
        assert data["inputs"].shape == (10, 81)
        assert data["labels"].shape == (10, 81)

    def test_loads_metadata(self, tmp_path):
        create_trm_npy_dir(str(tmp_path), split="test")
        data = load_trm_dataset(str(tmp_path), split="test")

        assert "metadata" in data
        assert data["metadata"]["vocab_size"] == 11

    def test_missing_split_raises(self, tmp_path):
        create_trm_npy_dir(str(tmp_path), split="test")
        with pytest.raises(FileNotFoundError):
            load_trm_dataset(str(tmp_path), split="train")

    def test_missing_inputs_raises(self, tmp_path):
        split_dir = tmp_path / "test"
        split_dir.mkdir(parents=True)
        # Create labels but no inputs
        np.save(str(split_dir / "all__labels.npy"), np.zeros((5, 10)))
        with pytest.raises(FileNotFoundError):
            load_trm_dataset(str(tmp_path), split="test")


# =============================================================================
# 2. convert_trm_to_npz tests
# =============================================================================

class TestConvertTrmToNpz:

    def test_creates_npz_file(self, tmp_path):
        create_trm_npy_dir(str(tmp_path), split="test", N=10, seq_len=81)
        output_path = convert_trm_to_npz(str(tmp_path), split="test")

        assert os.path.exists(output_path)
        assert output_path.endswith(".npz")

    def test_npz_contents_match_npy(self, tmp_path):
        create_trm_npy_dir(str(tmp_path), split="test", N=10, seq_len=81)
        output_path = convert_trm_to_npz(str(tmp_path), split="test")

        # Load both and compare
        npy_data = load_trm_dataset(str(tmp_path), split="test")
        npz_data = np.load(output_path, allow_pickle=True)

        np.testing.assert_array_equal(npy_data["inputs"], npz_data["inputs"])
        np.testing.assert_array_equal(npy_data["labels"], npz_data["labels"])

    def test_custom_output_path(self, tmp_path):
        create_trm_npy_dir(str(tmp_path), split="test")
        custom_path = str(tmp_path / "custom_output.npz")
        output_path = convert_trm_to_npz(str(tmp_path), split="test", output_path=custom_path)

        assert output_path == custom_path
        assert os.path.exists(custom_path)

    def test_roundtrip_npy_to_npz_to_dataset(self, tmp_path):
        """Full roundtrip: create .npy dir -> convert to .npz -> load via PuzzleDataset."""
        create_trm_npy_dir(str(tmp_path), split="test", N=15, seq_len=81)
        npz_path = convert_trm_to_npz(str(tmp_path), split="test")

        dataset = PuzzleDataset(npz_path)
        assert len(dataset) == 15
        assert dataset[0]["inputs"].shape == (81,)


# =============================================================================
# 3. PuzzleDataset TRM format tests
# =============================================================================

class TestPuzzleDatasetTrmFormat:

    def test_load_from_npy_directory(self, tmp_path):
        """PuzzleDataset should load directly from TRM .npy format."""
        create_trm_npy_dir(str(tmp_path), split="test", N=10, seq_len=81)
        dataset = PuzzleDataset(str(tmp_path), split="test")

        assert len(dataset) == 10
        sample = dataset[0]
        assert sample["inputs"].shape == (81,)
        assert sample["labels"].shape == (81,)

    def test_prefers_npz_over_npy(self, tmp_path):
        """When both .npz and .npy exist, .npz should be preferred."""
        create_trm_npy_dir(str(tmp_path), split="test", N=10, seq_len=81)

        # Create a .npz with different size to verify which is loaded
        npz_path = str(tmp_path / "test.npz")
        np.savez(
            npz_path,
            inputs=np.zeros((5, 81), dtype=np.int64),
            labels=np.zeros((5, 81), dtype=np.int64),
        )

        dataset = PuzzleDataset(str(tmp_path), split="test")
        assert len(dataset) == 5  # Should load from .npz (5 samples) not .npy (10 samples)

    def test_npy_metadata_loaded(self, tmp_path):
        create_trm_npy_dir(str(tmp_path), split="test")
        dataset = PuzzleDataset(str(tmp_path), split="test")
        assert dataset.metadata["vocab_size"] == 11

    def test_npy_max_samples(self, tmp_path):
        create_trm_npy_dir(str(tmp_path), split="test", N=20)
        dataset = PuzzleDataset(str(tmp_path), split="test", max_samples=5)
        assert len(dataset) == 5

    def test_npy_without_puzzle_identifiers(self, tmp_path):
        """Test .npy format without puzzle_identifiers file."""
        split_dir = tmp_path / "test"
        split_dir.mkdir(parents=True)
        np.save(str(split_dir / "all__inputs.npy"), np.zeros((5, 10), dtype=np.int32))
        np.save(str(split_dir / "all__labels.npy"), np.ones((5, 10), dtype=np.int32))

        dataset = PuzzleDataset(str(tmp_path), split="test")
        assert len(dataset) == 5
        assert torch.all(dataset.puzzle_identifiers == 0)

    def test_error_message_includes_both_formats(self, tmp_path):
        """Error message should mention both .npz and .npy paths."""
        (tmp_path / "dummy").mkdir()  # Make it a valid directory
        with pytest.raises(FileNotFoundError, match="all__inputs.npy"):
            PuzzleDataset(str(tmp_path), split="val")


# =============================================================================
# 4. DATASET_REGISTRY tests
# =============================================================================

class TestDatasetRegistry:

    def test_registry_has_all_benchmarks(self):
        assert "sudoku" in DATASET_REGISTRY
        assert "maze" in DATASET_REGISTRY
        assert "arc" in DATASET_REGISTRY

    def test_registry_values_are_callable(self):
        for name, builder in DATASET_REGISTRY.items():
            assert callable(builder), f"Builder for '{name}' is not callable"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
