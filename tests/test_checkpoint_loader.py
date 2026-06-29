"""
Tests for Component 1: Checkpoint Download & Loading.

Test categories:
  1. download_models.py — MODEL_REGISTRY validity, download_benchmark, CLI
  2. checkpoint_loader.py — prefix stripping, config building, model loading, manifest loading
"""

import os
import sys
import tempfile
import shutil

import pytest
import yaml
import torch

# Ensure project root is on path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from scripts.download_models import MODEL_REGISTRY, download_benchmark, _cleanup_empty_dirs
from inference.checkpoint_loader import (
    _strip_compile_prefix,
    _build_model_config,
    _detect_num_puzzle_identifiers,
    load_model,
    load_model_from_manifest,
    list_available_models,
    _add_trm_to_path,
)


# =============================================================================
# 1. download_models.py tests
# =============================================================================

class TestModelRegistry:
    """Validate the MODEL_REGISTRY data structure."""

    def test_registry_not_empty(self):
        assert len(MODEL_REGISTRY) > 0

    def test_all_entries_have_required_fields(self):
        required_fields = {"hf_repo", "files", "arch_config", "dataset_meta", "default_checkpoint"}
        for name, entry in MODEL_REGISTRY.items():
            missing = required_fields - set(entry.keys())
            assert not missing, f"'{name}' missing fields: {missing}"

    def test_all_files_have_required_keys(self):
        for name, entry in MODEL_REGISTRY.items():
            for i, file_info in enumerate(entry["files"]):
                assert "hf_filename" in file_info, f"'{name}' files[{i}] missing 'hf_filename'"
                assert "local_filename" in file_info, f"'{name}' files[{i}] missing 'local_filename'"

    def test_arch_config_has_name(self):
        for name, entry in MODEL_REGISTRY.items():
            assert "name" in entry["arch_config"], f"'{name}' arch_config missing 'name'"
            assert "@" in entry["arch_config"]["name"], (
                f"'{name}' arch_config name should be 'module@ClassName' format"
            )

    def test_dataset_meta_has_required_fields(self):
        required = {"vocab_size", "seq_len", "num_puzzle_identifiers"}
        for name, entry in MODEL_REGISTRY.items():
            missing = required - set(entry["dataset_meta"].keys())
            assert not missing, f"'{name}' dataset_meta missing: {missing}"

    def test_default_checkpoint_is_in_files(self):
        for name, entry in MODEL_REGISTRY.items():
            local_filenames = {f["local_filename"] for f in entry["files"]}
            assert entry["default_checkpoint"] in local_filenames, (
                f"'{name}' default_checkpoint '{entry['default_checkpoint']}' "
                f"not found in files: {local_filenames}"
            )

    def test_sudoku_is_mlp(self):
        assert MODEL_REGISTRY["sudoku"]["arch_config"]["mlp_t"] is True

    def test_maze_is_attention(self):
        assert MODEL_REGISTRY["maze"]["arch_config"]["mlp_t"] is False

    def test_arc_v2_is_attention(self):
        assert MODEL_REGISTRY["arc_v2"]["arch_config"]["mlp_t"] is False

    def test_sudoku_seq_len_is_81(self):
        assert MODEL_REGISTRY["sudoku"]["dataset_meta"]["seq_len"] == 81

    def test_maze_seq_len_is_900(self):
        assert MODEL_REGISTRY["maze"]["dataset_meta"]["seq_len"] == 900


class TestCleanupEmptyDirs:
    """Test _cleanup_empty_dirs helper."""

    def test_removes_empty_nested_dirs(self, tmp_path):
        nested = tmp_path / "a" / "b" / "c"
        nested.mkdir(parents=True)
        assert nested.exists()

        _cleanup_empty_dirs(str(tmp_path))
        assert not (tmp_path / "a").exists()

    def test_preserves_dirs_with_files(self, tmp_path):
        sub = tmp_path / "keep_me"
        sub.mkdir()
        (sub / "file.txt").write_text("hello")

        _cleanup_empty_dirs(str(tmp_path))
        assert sub.exists()
        assert (sub / "file.txt").exists()


class TestDownloadBenchmark:
    """Test download_benchmark function (offline, mocked)."""

    def test_invalid_benchmark_raises(self):
        with pytest.raises(ValueError, match="Unknown benchmark"):
            download_benchmark("nonexistent_benchmark", "/tmp/test_models")

    def test_creates_output_directory(self, tmp_path):
        model_dir = os.path.join(str(tmp_path), "test_model")
        # This will fail on the actual download (no network), but should create the dir
        try:
            download_benchmark("sudoku", str(tmp_path))
        except Exception:
            pass
        # The directory should have been created before the download attempt
        assert os.path.isdir(os.path.join(str(tmp_path), "sudoku"))


# =============================================================================
# 2. checkpoint_loader.py tests
# =============================================================================

class TestStripCompilePrefix:
    """Test _strip_compile_prefix state dict key transformation."""

    def test_strips_orig_mod_prefix(self):
        state_dict = {"_orig_mod.model.inner.weight": torch.tensor([1.0])}
        result = _strip_compile_prefix(state_dict)
        assert "inner.weight" in result
        assert "_orig_mod.model.inner.weight" not in result

    def test_strips_model_prefix_only(self):
        state_dict = {"model.inner.weight": torch.tensor([1.0])}
        result = _strip_compile_prefix(state_dict)
        assert "inner.weight" in result

    def test_no_prefix_unchanged(self):
        state_dict = {"inner.weight": torch.tensor([1.0])}
        result = _strip_compile_prefix(state_dict)
        assert "inner.weight" in result

    def test_preserves_values(self):
        t = torch.randn(3, 4)
        state_dict = {"_orig_mod.model.layer.weight": t}
        result = _strip_compile_prefix(state_dict)
        assert torch.equal(result["layer.weight"], t)

    def test_handles_empty_dict(self):
        result = _strip_compile_prefix({})
        assert result == {}

    def test_handles_multiple_keys(self):
        state_dict = {
            "_orig_mod.model.inner.embed_tokens.weight": torch.tensor([1.0]),
            "_orig_mod.model.inner.lm_head.weight": torch.tensor([2.0]),
            "_orig_mod.model.inner.q_head.weight": torch.tensor([3.0]),
        }
        result = _strip_compile_prefix(state_dict)
        expected_keys = {
            "inner.embed_tokens.weight",
            "inner.lm_head.weight",
            "inner.q_head.weight",
        }
        assert set(result.keys()) == expected_keys

    def test_double_model_prefix(self):
        """Edge case: key like _orig_mod.model.model.something"""
        state_dict = {"_orig_mod.model.model.weight": torch.tensor([1.0])}
        result = _strip_compile_prefix(state_dict)
        # After stripping _orig_mod. -> model.model.weight
        # After stripping model. -> model.weight
        assert "model.weight" in result


class TestBuildModelConfig:
    """Test _build_model_config merging logic."""

    def test_basic_merge(self):
        arch = {
            "name": "test@Model",
            "loss": {"name": "test"},
            "hidden_size": 512,
            "num_heads": 8,
            "mlp_t": False,
        }
        meta = {"vocab_size": 11, "seq_len": 81, "num_puzzle_identifiers": 1}
        result = _build_model_config(arch, meta)

        assert result["hidden_size"] == 512
        assert result["vocab_size"] == 11
        assert result["seq_len"] == 81
        assert result["batch_size"] == 1  # default
        assert "name" not in result  # excluded
        assert "loss" not in result  # excluded

    def test_custom_batch_size(self):
        arch = {"name": "test@Model", "loss": {}, "hidden_size": 256}
        meta = {"vocab_size": 5, "seq_len": 100}
        result = _build_model_config(arch, meta, batch_size=32)
        assert result["batch_size"] == 32

    def test_dataset_meta_overrides(self):
        arch = {"name": "test@Model", "loss": {}}
        meta = {"vocab_size": 294, "seq_len": 100, "num_puzzle_identifiers": 50}
        result = _build_model_config(arch, meta)
        assert result["num_puzzle_identifiers"] == 50


class TestDetectNumPuzzleIdentifiers:
    """Test _detect_num_puzzle_identifiers from state dict inspection."""

    def test_detects_from_weights(self):
        state_dict = {
            "inner.puzzle_emb.weights": torch.randn(42, 512),
        }
        result = _detect_num_puzzle_identifiers(state_dict, {})
        assert result == 42

    def test_falls_back_to_config(self):
        state_dict = {"inner.other_param": torch.randn(10)}
        result = _detect_num_puzzle_identifiers(state_dict, {"num_puzzle_identifiers": 7})
        assert result == 7

    def test_falls_back_to_default_1(self):
        state_dict = {"inner.other_param": torch.randn(10)}
        result = _detect_num_puzzle_identifiers(state_dict, {})
        assert result == 1


class TestAddTrmToPath:
    """Test that _add_trm_to_path correctly adds the submodule."""

    def test_adds_trm_root(self):
        _add_trm_to_path()
        trm_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "TinyRecursiveModels")
        )
        assert trm_root in sys.path

    def test_idempotent(self):
        _add_trm_to_path()
        count_before = sys.path.count(
            os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "TinyRecursiveModels")
            )
        )
        _add_trm_to_path()
        count_after = sys.path.count(
            os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "TinyRecursiveModels")
            )
        )
        assert count_after == count_before


class TestLoadModelFromManifest:
    """Test manifest-based model loading."""

    def test_missing_manifest_raises(self):
        with pytest.raises(FileNotFoundError):
            load_model_from_manifest("/nonexistent/path/manifest.yaml")

    def test_missing_checkpoint_raises(self, tmp_path):
        manifest = {
            "benchmark": "test",
            "hf_repo": "test/test",
            "arch_config": {"name": "test@Model", "mlp_t": False},
            "dataset_meta": {"vocab_size": 11, "seq_len": 81},
            "default_checkpoint": "model.pt",
        }
        manifest_path = tmp_path / "manifest.yaml"
        with open(manifest_path, "w") as f:
            yaml.dump(manifest, f)

        with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
            load_model_from_manifest(str(manifest_path))


class TestListAvailableModels:
    """Test list_available_models scanner."""

    def test_empty_directory(self, tmp_path):
        result = list_available_models(str(tmp_path))
        assert result == []

    def test_nonexistent_directory(self):
        result = list_available_models("/nonexistent/path")
        assert result == []

    def test_finds_manifests(self, tmp_path):
        # Create a fake model directory with manifest
        model_dir = tmp_path / "test_model"
        model_dir.mkdir()
        manifest = {
            "benchmark": "test_bench",
            "hf_repo": "user/repo",
            "arch_config": {"mlp_t": False},
            "dataset_meta": {},
            "default_checkpoint": "model.pt",
        }
        with open(model_dir / "manifest.yaml", "w") as f:
            yaml.dump(manifest, f)

        result = list_available_models(str(tmp_path))
        assert len(result) == 1
        assert result[0]["benchmark"] == "test_bench"
        assert result[0]["arch_type"] == "TRM-Att"

    def test_identifies_mlp_type(self, tmp_path):
        model_dir = tmp_path / "mlp_model"
        model_dir.mkdir()
        manifest = {
            "benchmark": "sudoku",
            "hf_repo": "user/repo",
            "arch_config": {"mlp_t": True},
            "dataset_meta": {},
            "default_checkpoint": "model.pt",
        }
        with open(model_dir / "manifest.yaml", "w") as f:
            yaml.dump(manifest, f)

        result = list_available_models(str(tmp_path))
        assert result[0]["arch_type"] == "TRM-MLP"


# =============================================================================
# Integration test: full model loading with TRM submodule
# =============================================================================

class TestIntegrationModelInstantiation:
    """
    Integration tests that instantiate real TRM models (without weights).
    These verify the full pipeline from config -> model object.
    Requires the TinyRecursiveModels submodule.
    """

    @pytest.fixture(autouse=True)
    def setup_trm_path(self):
        _add_trm_to_path()

    def _can_import_trm(self):
        try:
            from utils.functions import load_model_class
            return True
        except ImportError:
            return False

    @pytest.mark.skipif(
        not os.path.isdir(os.path.join(PROJECT_ROOT, "TinyRecursiveModels", "models")),
        reason="TinyRecursiveModels submodule not available"
    )
    def test_instantiate_sudoku_mlp_model(self):
        """Verify the Sudoku TRM-MLP config produces a valid model."""
        from utils.functions import load_model_class

        entry = MODEL_REGISTRY["sudoku"]
        model_cfg = _build_model_config(entry["arch_config"], entry["dataset_meta"], batch_size=1)
        model_cls = load_model_class(entry["arch_config"]["name"])

        with torch.device("cpu"):
            model = model_cls(model_cfg)

        # Verify model has the expected structure
        assert hasattr(model, "inner")
        assert hasattr(model.inner, "embed_tokens")
        assert hasattr(model.inner, "lm_head")
        assert hasattr(model.inner, "q_head")
        assert hasattr(model.inner, "L_level")

        # Check parameter count is in the right ballpark (should be ~5M for MLP)
        num_params = sum(p.numel() for p in model.parameters())
        assert 3_000_000 < num_params < 8_000_000, f"Unexpected param count: {num_params:,}"

    @pytest.mark.skipif(
        not os.path.isdir(os.path.join(PROJECT_ROOT, "TinyRecursiveModels", "models")),
        reason="TinyRecursiveModels submodule not available"
    )
    def test_instantiate_maze_att_model(self):
        """Verify the Maze TRM-Att config produces a valid model."""
        from utils.functions import load_model_class

        entry = MODEL_REGISTRY["maze"]
        model_cfg = _build_model_config(entry["arch_config"], entry["dataset_meta"], batch_size=1)
        model_cls = load_model_class(entry["arch_config"]["name"])

        with torch.device("cpu"):
            model = model_cls(model_cfg)

        assert hasattr(model, "inner")
        num_params = sum(p.numel() for p in model.parameters())
        assert 5_000_000 < num_params < 10_000_000, f"Unexpected param count: {num_params:,}"

    @pytest.mark.skipif(
        not os.path.isdir(os.path.join(PROJECT_ROOT, "TinyRecursiveModels", "models")),
        reason="TinyRecursiveModels submodule not available"
    )
    def test_instantiate_arc_v2_model(self):
        """Verify the ARC-AGI-2 TRM-Att config produces a valid model."""
        from utils.functions import load_model_class

        entry = MODEL_REGISTRY["arc_v2"]
        model_cfg = _build_model_config(entry["arch_config"], entry["dataset_meta"], batch_size=1)
        model_cls = load_model_class(entry["arch_config"]["name"])

        with torch.device("cpu"):
            model = model_cls(model_cfg)

        assert hasattr(model, "inner")
        num_params = sum(p.numel() for p in model.parameters())
        assert 5_000_000 < num_params < 10_000_000, f"Unexpected param count: {num_params:,}"

    @pytest.mark.skipif(
        not os.path.isdir(os.path.join(PROJECT_ROOT, "TinyRecursiveModels", "models")),
        reason="TinyRecursiveModels submodule not available"
    )
    def test_model_forward_pass_sudoku(self):
        """Verify a forward pass works with dummy data for Sudoku-MLP."""
        from utils.functions import load_model_class

        entry = MODEL_REGISTRY["sudoku"]
        meta = entry["dataset_meta"]
        model_cfg = _build_model_config(entry["arch_config"], meta, batch_size=2)
        model_cls = load_model_class(entry["arch_config"]["name"])

        with torch.device("cpu"):
            model = model_cls(model_cfg)
        model.eval()

        # Create dummy batch
        batch = {
            "inputs": torch.randint(0, meta["vocab_size"], (2, meta["seq_len"])),
            "labels": torch.randint(0, meta["vocab_size"], (2, meta["seq_len"])),
            "puzzle_identifiers": torch.zeros(2, dtype=torch.long),
        }

        # Run initial_carry -> forward
        with torch.no_grad():
            carry = model.initial_carry(batch)
            new_carry, outputs = model(carry=carry, batch=batch)

        # Check outputs contain expected keys
        assert "logits" in outputs
        assert "q_halt_logits" in outputs
        assert outputs["logits"].shape == (2, meta["seq_len"], meta["vocab_size"])

    @pytest.mark.skipif(
        not os.path.isdir(os.path.join(PROJECT_ROOT, "TinyRecursiveModels", "models")),
        reason="TinyRecursiveModels submodule not available"
    )
    def test_round_trip_save_load(self, tmp_path):
        """Test saving and loading a model through our checkpoint_loader."""
        from utils.functions import load_model_class

        entry = MODEL_REGISTRY["sudoku"]
        model_cfg = _build_model_config(entry["arch_config"], entry["dataset_meta"], batch_size=1)
        model_cls = load_model_class(entry["arch_config"]["name"])

        with torch.device("cpu"):
            original_model = model_cls(model_cfg)

        # Save with the _orig_mod.model. prefix pattern (simulating torch.compile + loss head)
        state_dict = {}
        for key, value in original_model.state_dict().items():
            state_dict[f"_orig_mod.model.{key}"] = value

        checkpoint_path = str(tmp_path / "checkpoint.pt")
        torch.save(state_dict, checkpoint_path)

        # Load through our loader
        loaded_model, metadata = load_model(
            checkpoint_path=checkpoint_path,
            arch_config=entry["arch_config"],
            dataset_meta=entry["dataset_meta"],
            device="cpu",
        )

        # Verify parameters match
        for (name_orig, param_orig), (name_loaded, param_loaded) in zip(
            original_model.named_parameters(), loaded_model.named_parameters()
        ):
            assert name_orig == name_loaded, f"Name mismatch: {name_orig} vs {name_loaded}"
            assert torch.equal(param_orig, param_loaded), f"Value mismatch for {name_orig}"

    @pytest.mark.skipif(
        not os.path.isdir(os.path.join(PROJECT_ROOT, "TinyRecursiveModels", "models")),
        reason="TinyRecursiveModels submodule not available"
    )
    def test_manifest_round_trip(self, tmp_path):
        """Test full manifest-based save/load cycle."""
        from utils.functions import load_model_class

        entry = MODEL_REGISTRY["sudoku"]
        model_cfg = _build_model_config(entry["arch_config"], entry["dataset_meta"], batch_size=1)
        model_cls = load_model_class(entry["arch_config"]["name"])

        with torch.device("cpu"):
            original_model = model_cls(model_cfg)

        # Save checkpoint with compile prefix
        state_dict = {f"_orig_mod.model.{k}": v for k, v in original_model.state_dict().items()}
        torch.save(state_dict, str(tmp_path / "model_mlp.pt"))

        # Save manifest
        manifest = {
            "benchmark": "sudoku",
            "hf_repo": "test/test",
            "arch_config": entry["arch_config"],
            "dataset_meta": entry["dataset_meta"],
            "default_checkpoint": "model_mlp.pt",
        }
        manifest_path = str(tmp_path / "manifest.yaml")
        with open(manifest_path, "w") as f:
            yaml.dump(manifest, f)

        # Load via manifest
        loaded_model, metadata = load_model_from_manifest(manifest_path, device="cpu")
        assert metadata["num_params"] > 0
        assert metadata["checkpoint_path"] == str(tmp_path / "model_mlp.pt")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
