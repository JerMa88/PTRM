"""
PTRM inference configuration loader.

Loads YAML inference configs and converts them to EvaluationConfig objects,
supporting the YAML format defined in config/inference/*.yaml.
"""

import os
from typing import Optional

import yaml

from evaluation.evaluate import EvaluationConfig


def load_inference_config(
    config_path: str,
    overrides: Optional[dict] = None,
) -> EvaluationConfig:
    """
    Load an inference configuration from a YAML file.

    Args:
        config_path: Path to the YAML config file.
        overrides: Optional dict of key-value overrides (e.g., {"ptrm.K": 100}).

    Returns:
        EvaluationConfig ready for run_evaluation().
    """
    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)

    # Apply overrides
    if overrides:
        for key, value in overrides.items():
            parts = key.split(".")
            d = raw
            for part in parts[:-1]:
                d = d.setdefault(part, {})
            d[parts[-1]] = value

    # Resolve paths relative to the config file's directory
    config_dir = os.path.dirname(os.path.abspath(config_path))
    project_root = os.path.abspath(os.path.join(config_dir, "..", ".."))

    def resolve_path(path: str) -> str:
        if os.path.isabs(path):
            return path
        return os.path.join(project_root, path)

    model = raw.get("model", {})
    data = raw.get("data", {})
    ptrm = raw.get("ptrm", {})
    evaluation = raw.get("evaluation", {})

    return EvaluationConfig(
        manifest_path=resolve_path(model.get("manifest_path", "")),
        checkpoint_override=model.get("checkpoint_override"),
        data_path=resolve_path(data.get("path", "")),
        split=data.get("split", "test"),
        max_samples=data.get("max_samples"),
        batch_size=evaluation.get("batch_size", 32),
        K=ptrm.get("K", 25),
        D=ptrm.get("D", 16),
        sigma=ptrm.get("sigma", 0.3),
        seed=ptrm.get("seed"),
        ignore_id=evaluation.get("ignore_id", 0),
        collect_trajectories=evaluation.get("collect_trajectories", False),
        k_chunk_size=evaluation.get("k_chunk_size"),
        device=raw.get("device", "cuda"),
    )


def list_available_configs(config_dir: str = "config/inference") -> list[dict]:
    """
    List all available inference configs.

    Returns:
        List of dicts with name, path, and key parameters.
    """
    configs = []
    if not os.path.isdir(config_dir):
        return configs

    for filename in sorted(os.listdir(config_dir)):
        if filename.endswith(".yaml") or filename.endswith(".yml"):
            path = os.path.join(config_dir, filename)
            try:
                with open(path, "r") as f:
                    raw = yaml.safe_load(f)
                ptrm = raw.get("ptrm", {})
                configs.append({
                    "name": os.path.splitext(filename)[0],
                    "path": path,
                    "K": ptrm.get("K", 25),
                    "D": ptrm.get("D", 16),
                    "sigma": ptrm.get("sigma", 0.3),
                })
            except Exception:
                continue

    return configs
