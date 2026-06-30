"""
End-to-end PTRM evaluation runner.

Orchestrates:
  1. Load model from manifest
  2. Load dataset from .npz
  3. Run PTRM inference on batches
  4. Compute and aggregate metrics
  5. Print and optionally save results

Usage:
    from evaluation.evaluate import run_evaluation

    results = run_evaluation(
        manifest_path="models/sudoku/manifest.yaml",
        data_path="data/sudoku/test.npz",
        K=25, D=16, sigma=0.3,
    )
"""

from typing import Optional
from dataclasses import dataclass, field

import torch
from tqdm import tqdm

from inference.checkpoint_loader import load_model_from_manifest
from inference.ptrm_inference import PTRMInference, PTRMBatchResult
from evaluation.metrics import compute_metrics_from_result, format_metrics, MetricsResult
from evaluation.evaluators.puzzle_evaluator import create_puzzle_dataloader


@dataclass
class EvaluationConfig:
    """Configuration for a PTRM evaluation run."""
    # Model
    manifest_path: str
    checkpoint_override: Optional[str] = None

    # Data
    data_path: str = ""
    split: str = "test"
    max_samples: Optional[int] = None
    batch_size: int = 32

    # PTRM parameters
    K: int = 25          # Number of parallel rollouts
    D: int = 16          # Number of supervision steps
    sigma: float = 0.3   # Noise standard deviation
    seed: Optional[int] = None

    # Evaluation
    ignore_id: int = 0
    collect_trajectories: bool = False
    k_chunk_size: Optional[int] = None

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Output
    output_path: Optional[str] = None


@dataclass
class EvaluationResult:
    """Full result of a PTRM evaluation run."""
    config: EvaluationConfig
    metrics: MetricsResult
    # Per-batch results (for further analysis if needed)
    batch_results: list[PTRMBatchResult] = field(default_factory=list)
    batch_labels: list[torch.Tensor] = field(default_factory=list)


def run_evaluation(
    manifest_path: str,
    data_path: str,
    K: int = 25,
    D: int = 16,
    sigma: float = 0.3,
    batch_size: int = 32,
    device: Optional[str] = None,
    seed: Optional[int] = None,
    split: str = "test",
    max_samples: Optional[int] = None,
    ignore_id: int = 0,
    collect_trajectories: bool = False,
    k_chunk_size: Optional[int] = None,
    checkpoint_override: Optional[str] = None,
    save_batch_results: bool = False,
) -> EvaluationResult:
    """
    Run a complete PTRM evaluation.

    Args:
        manifest_path: Path to model manifest.yaml.
        data_path: Path to dataset .npz file or directory.
        K: Number of parallel rollouts.
        D: Number of supervision steps.
        sigma: Noise standard deviation.
        batch_size: Evaluation batch size.
        device: Target device (auto-detected if None).
        seed: Random seed for reproducibility.
        split: Dataset split to evaluate.
        max_samples: Limit number of evaluation samples.
        ignore_id: Token ID to ignore in accuracy computation.
        collect_trajectories: Whether to collect latent trajectories.
        k_chunk_size: Chunk size for memory-efficient rollouts.
        checkpoint_override: Override the default checkpoint file.
        save_batch_results: Whether to keep per-batch PTRMBatchResults.

    Returns:
        EvaluationResult with aggregated metrics and optional batch details.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    config = EvaluationConfig(
        manifest_path=manifest_path,
        data_path=data_path,
        split=split,
        max_samples=max_samples,
        batch_size=batch_size,
        K=K, D=D, sigma=sigma, seed=seed,
        ignore_id=ignore_id,
        collect_trajectories=collect_trajectories,
        k_chunk_size=k_chunk_size,
        device=device,
        checkpoint_override=checkpoint_override,
    )

    # 1. Load model
    print(f"Loading model from {manifest_path}...")
    model, model_meta = load_model_from_manifest(
        manifest_path,
        device=device,
        batch_size=batch_size,
        checkpoint_override=checkpoint_override,
    )

    # 2. Create inference engine
    engine = PTRMInference(model, device=device)

    # 3. Load dataset
    print(f"Loading dataset from {data_path} (split={split})...")
    dataloader = create_puzzle_dataloader(
        data_path,
        split=split,
        batch_size=batch_size,
        max_samples=max_samples,
    )
    print(f"  {len(dataloader.dataset)} samples, {len(dataloader)} batches")

    # 4. Run inference and collect metrics
    all_pass_exact = []
    all_bestq_exact = []
    all_mode_exact = []
    all_pass_cell = []
    all_bestq_cell = []
    all_mode_cell = []

    batch_results = []
    batch_labels = []

    print(f"\nRunning PTRM inference (K={K}, D={D}, σ={sigma})...")
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
        labels = batch["labels"]

        # Run PTRM inference
        ptrm_result = engine.run(
            batch,
            K=K, D=D, sigma=sigma,
            seed=(seed + batch_idx) if seed is not None else None,
            collect_trajectories=collect_trajectories,
            k_chunk_size=k_chunk_size,
        )

        # Compute per-batch metrics
        batch_metrics = compute_metrics_from_result(
            ptrm_result,
            labels.to(device),
            ignore_id=ignore_id,
        )

        # Accumulate per-puzzle metrics
        all_pass_exact.append(batch_metrics.pass_at_k_exact.cpu())
        all_bestq_exact.append(batch_metrics.best_q_at_k_exact.cpu())
        all_mode_exact.append(batch_metrics.mode_at_k_exact.cpu())
        all_pass_cell.append(batch_metrics.pass_at_k_cell.cpu())
        all_bestq_cell.append(batch_metrics.best_q_at_k_cell.cpu())
        all_mode_cell.append(batch_metrics.mode_at_k_cell.cpu())

        if save_batch_results:
            batch_results.append(ptrm_result)
            batch_labels.append(labels)

    # 5. Aggregate metrics
    pass_exact = torch.cat(all_pass_exact)
    bestq_exact = torch.cat(all_bestq_exact)
    mode_exact = torch.cat(all_mode_exact)
    pass_cell = torch.cat(all_pass_cell)
    bestq_cell = torch.cat(all_bestq_cell)
    mode_cell = torch.cat(all_mode_cell)

    N = len(pass_exact)
    metrics = MetricsResult(
        pass_at_k_exact=pass_exact,
        best_q_at_k_exact=bestq_exact,
        mode_at_k_exact=mode_exact,
        pass_at_k_cell=pass_cell,
        best_q_at_k_cell=bestq_cell,
        mode_at_k_cell=mode_cell,
        mean_pass_at_k_exact=pass_exact.mean().item(),
        mean_best_q_at_k_exact=bestq_exact.mean().item(),
        mean_mode_at_k_exact=mode_exact.mean().item(),
        mean_pass_at_k_cell=pass_cell.mean().item(),
        mean_best_q_at_k_cell=bestq_cell.mean().item(),
        mean_mode_at_k_cell=mode_cell.mean().item(),
        num_puzzles=N,
        K=K,
    )

    # 6. Print results
    print(f"\n{format_metrics(metrics)}")

    result = EvaluationResult(
        config=config,
        metrics=metrics,
        batch_results=batch_results if save_batch_results else [],
        batch_labels=batch_labels if save_batch_results else [],
    )

    return result
