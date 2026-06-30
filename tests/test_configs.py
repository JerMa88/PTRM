"""
Tests for Component 5: Inference Configs.

Test categories:
  1. YAML config loading
  2. Config overrides
  3. Path resolution
  4. list_available_configs
  5. All actual config files validate correctly
"""

import os
import sys

import pytest
import yaml

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from config.config_loader import load_inference_config, list_available_configs
from evaluation.evaluate import EvaluationConfig


# =============================================================================
# 1. YAML config loading tests
# =============================================================================

class TestLoadInferenceConfig:

    def test_loads_sudoku_config(self):
        config_path = os.path.join(PROJECT_ROOT, "config", "inference", "sudoku_extreme.yaml")
        config = load_inference_config(config_path)

        assert isinstance(config, EvaluationConfig)
        assert config.K == 25
        assert config.D == 16
        assert config.sigma == 0.3
        assert config.split == "test"
        assert config.ignore_id == 0

    def test_loads_maze_config(self):
        config_path = os.path.join(PROJECT_ROOT, "config", "inference", "maze_hard.yaml")
        config = load_inference_config(config_path)

        assert config.K == 25
        assert config.D == 16
        assert config.batch_size == 16  # Smaller for longer sequences

    def test_loads_arc_config(self):
        config_path = os.path.join(PROJECT_ROOT, "config", "inference", "arc_agi2.yaml")
        config = load_inference_config(config_path)

        assert config.K == 25
        assert config.D == 64  # ARC uses more steps
        assert config.batch_size == 8
        assert config.k_chunk_size == 5

    def test_manifest_path_resolved(self):
        config_path = os.path.join(PROJECT_ROOT, "config", "inference", "sudoku_extreme.yaml")
        config = load_inference_config(config_path)

        assert os.path.isabs(config.manifest_path)
        assert "models/sudoku/manifest.yaml" in config.manifest_path

    def test_data_path_resolved(self):
        config_path = os.path.join(PROJECT_ROOT, "config", "inference", "sudoku_extreme.yaml")
        config = load_inference_config(config_path)

        assert os.path.isabs(config.data_path)
        assert "data/sudoku" in config.data_path


# =============================================================================
# 2. Config overrides tests
# =============================================================================

class TestConfigOverrides:

    def test_override_k(self):
        config_path = os.path.join(PROJECT_ROOT, "config", "inference", "sudoku_extreme.yaml")
        config = load_inference_config(config_path, overrides={"ptrm.K": 100})
        assert config.K == 100

    def test_override_sigma(self):
        config_path = os.path.join(PROJECT_ROOT, "config", "inference", "sudoku_extreme.yaml")
        config = load_inference_config(config_path, overrides={"ptrm.sigma": 0.5})
        assert config.sigma == 0.5

    def test_override_batch_size(self):
        config_path = os.path.join(PROJECT_ROOT, "config", "inference", "sudoku_extreme.yaml")
        config = load_inference_config(config_path, overrides={"evaluation.batch_size": 64})
        assert config.batch_size == 64

    def test_override_device(self):
        config_path = os.path.join(PROJECT_ROOT, "config", "inference", "sudoku_extreme.yaml")
        config = load_inference_config(config_path, overrides={"device": "cpu"})
        assert config.device == "cpu"

    def test_multiple_overrides(self):
        config_path = os.path.join(PROJECT_ROOT, "config", "inference", "sudoku_extreme.yaml")
        config = load_inference_config(config_path, overrides={
            "ptrm.K": 50,
            "ptrm.D": 32,
            "ptrm.sigma": 0.1,
        })
        assert config.K == 50
        assert config.D == 32
        assert config.sigma == 0.1


# =============================================================================
# 3. Custom YAML config tests
# =============================================================================

class TestCustomConfig:

    def test_minimal_config(self, tmp_path):
        config_data = {
            "model": {"manifest_path": "/abs/path/manifest.yaml"},
            "data": {"path": "/abs/path/data"},
            "ptrm": {"K": 10, "D": 8, "sigma": 0.2},
        }
        config_path = str(tmp_path / "test.yaml")
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        config = load_inference_config(config_path)
        assert config.K == 10
        assert config.D == 8
        assert config.sigma == 0.2
        assert config.manifest_path == "/abs/path/manifest.yaml"

    def test_defaults_applied(self, tmp_path):
        config_data = {
            "model": {"manifest_path": "/test/manifest.yaml"},
            "data": {"path": "/test/data"},
        }
        config_path = str(tmp_path / "minimal.yaml")
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        config = load_inference_config(config_path)
        assert config.K == 25  # default
        assert config.D == 16  # default
        assert config.sigma == 0.3  # default
        assert config.batch_size == 32  # default


# =============================================================================
# 4. list_available_configs tests
# =============================================================================

class TestListAvailableConfigs:

    def test_finds_all_configs(self):
        config_dir = os.path.join(PROJECT_ROOT, "config", "inference")
        configs = list_available_configs(config_dir)

        names = [c["name"] for c in configs]
        assert "sudoku_extreme" in names
        assert "maze_hard" in names
        assert "arc_agi2" in names

    def test_config_metadata(self):
        config_dir = os.path.join(PROJECT_ROOT, "config", "inference")
        configs = list_available_configs(config_dir)

        for c in configs:
            assert "name" in c
            assert "path" in c
            assert "K" in c
            assert "D" in c
            assert "sigma" in c

    def test_empty_directory(self, tmp_path):
        configs = list_available_configs(str(tmp_path))
        assert configs == []

    def test_nonexistent_directory(self):
        configs = list_available_configs("/nonexistent/path")
        assert configs == []


# =============================================================================
# 5. Validate all actual config files
# =============================================================================

class TestAllConfigsValid:

    @pytest.fixture(params=["sudoku_extreme", "maze_hard", "arc_agi2"])
    def config_name(self, request):
        return request.param

    def test_config_loads_without_error(self, config_name):
        config_path = os.path.join(PROJECT_ROOT, "config", "inference", f"{config_name}.yaml")
        config = load_inference_config(config_path)
        assert isinstance(config, EvaluationConfig)

    def test_config_has_valid_ptrm_params(self, config_name):
        config_path = os.path.join(PROJECT_ROOT, "config", "inference", f"{config_name}.yaml")
        config = load_inference_config(config_path)

        assert config.K > 0
        assert config.D > 0
        assert config.sigma >= 0
        assert config.batch_size > 0

    def test_config_yaml_is_valid(self, config_name):
        config_path = os.path.join(PROJECT_ROOT, "config", "inference", f"{config_name}.yaml")
        with open(config_path, "r") as f:
            raw = yaml.safe_load(f)

        assert "model" in raw
        assert "data" in raw
        assert "ptrm" in raw


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
