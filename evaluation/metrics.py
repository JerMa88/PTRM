"""
PTRM Evaluation Metrics.

Implements the three selection/evaluation metrics from the paper:
  - pass@K:   Oracle upper bound — any of K rollouts is correct
  - best-Q@K: Rollout with highest Q-value is correct (inference-time selectable)
  - mode@K:   Most frequent answer among K rollouts is correct (majority vote)

All metrics operate on the outputs of PTRMBatchResult and ground truth labels.

Correctness is measured two ways:
  - Cell accuracy: fraction of tokens matching ground truth (ignoring pad/ignore)
  - Exact accuracy: entire prediction matches ground truth exactly (ignoring pad/ignore)
"""

from dataclasses import dataclass

import torch


@dataclass
class MetricsResult:
    """Container for all PTRM evaluation metrics on a batch."""
    # Per-puzzle metrics (shape: (B,) for each)
    pass_at_k_exact: torch.Tensor       # 1.0 if any rollout is exactly correct
    best_q_at_k_exact: torch.Tensor     # 1.0 if best-Q rollout is exactly correct
    mode_at_k_exact: torch.Tensor       # 1.0 if mode rollout is exactly correct

    pass_at_k_cell: torch.Tensor        # Max cell accuracy across K rollouts
    best_q_at_k_cell: torch.Tensor      # Cell accuracy of best-Q rollout
    mode_at_k_cell: torch.Tensor        # Cell accuracy of mode rollout

    # Aggregated scalars
    mean_pass_at_k_exact: float
    mean_best_q_at_k_exact: float
    mean_mode_at_k_exact: float
    mean_pass_at_k_cell: float
    mean_best_q_at_k_cell: float
    mean_mode_at_k_cell: float

    # Metadata
    num_puzzles: int
    K: int


def compute_cell_accuracy(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    ignore_id: int = 0,
) -> torch.Tensor:
    """
    Compute per-token accuracy, ignoring tokens where label == ignore_id.

    Args:
        predictions: (..., seq_len) predicted token IDs.
        labels: (..., seq_len) ground truth token IDs.
        ignore_id: Token ID to ignore in accuracy computation (typically 0 = PAD).

    Returns:
        (...) cell accuracy as a float tensor (0.0 to 1.0).
    """
    mask = labels != ignore_id                          # (..., seq_len)
    correct = (predictions == labels) & mask            # (..., seq_len)
    # Avoid division by zero for fully-padded sequences
    num_valid = mask.float().sum(dim=-1).clamp(min=1)   # (...)
    return correct.float().sum(dim=-1) / num_valid      # (...)


def compute_exact_accuracy(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    ignore_id: int = 0,
) -> torch.Tensor:
    """
    Compute exact-match accuracy: 1.0 if ALL non-ignored tokens match.

    Args:
        predictions: (..., seq_len) predicted token IDs.
        labels: (..., seq_len) ground truth token IDs.
        ignore_id: Token ID to ignore.

    Returns:
        (...) boolean tensor (0.0 or 1.0) indicating full match.
    """
    mask = labels != ignore_id                          # (..., seq_len)
    # A prediction is "exactly correct" if every non-ignored token matches.
    correct = (predictions == labels) | ~mask            # (..., seq_len)
    return correct.all(dim=-1).float()                  # (...)


def compute_metrics(
    all_predictions: torch.Tensor,
    all_q_values: torch.Tensor,
    best_q_predictions: torch.Tensor,
    mode_predictions: torch.Tensor,
    labels: torch.Tensor,
    ignore_id: int = 0,
) -> MetricsResult:
    """
    Compute all PTRM evaluation metrics.

    Args:
        all_predictions: (B, K, seq_len) predictions from all K rollouts.
        all_q_values: (B, K) Q-halt logits from all K rollouts.
        best_q_predictions: (B, seq_len) best-Q selected prediction per puzzle.
        mode_predictions: (B, seq_len) mode-selected prediction per puzzle.
        labels: (B, seq_len) ground truth labels.
        ignore_id: Token ID to ignore in accuracy computation.

    Returns:
        MetricsResult with all per-puzzle and aggregated metrics.
    """
    B, K, seq_len = all_predictions.shape

    # Expand labels for per-rollout comparison: (B, seq_len) -> (B, K, seq_len)
    labels_expanded = labels.unsqueeze(1).expand_as(all_predictions)

    # === Per-rollout metrics ===
    # Cell accuracy per rollout: (B, K)
    rollout_cell_acc = compute_cell_accuracy(all_predictions, labels_expanded, ignore_id)
    # Exact accuracy per rollout: (B, K)
    rollout_exact_acc = compute_exact_accuracy(all_predictions, labels_expanded, ignore_id)

    # === pass@K ===
    # Any rollout exactly correct
    pass_at_k_exact = rollout_exact_acc.max(dim=1).values      # (B,)
    # Max cell accuracy across rollouts (oracle best)
    pass_at_k_cell = rollout_cell_acc.max(dim=1).values        # (B,)

    # === best-Q@K ===
    best_q_cell = compute_cell_accuracy(best_q_predictions, labels, ignore_id)  # (B,)
    best_q_exact = compute_exact_accuracy(best_q_predictions, labels, ignore_id)  # (B,)

    # === mode@K ===
    mode_cell = compute_cell_accuracy(mode_predictions, labels, ignore_id)  # (B,)
    mode_exact = compute_exact_accuracy(mode_predictions, labels, ignore_id)  # (B,)

    return MetricsResult(
        pass_at_k_exact=pass_at_k_exact,
        best_q_at_k_exact=best_q_exact,
        mode_at_k_exact=mode_exact,
        pass_at_k_cell=pass_at_k_cell,
        best_q_at_k_cell=best_q_cell,
        mode_at_k_cell=mode_cell,
        mean_pass_at_k_exact=pass_at_k_exact.mean().item(),
        mean_best_q_at_k_exact=best_q_exact.mean().item(),
        mean_mode_at_k_exact=mode_exact.mean().item(),
        mean_pass_at_k_cell=pass_at_k_cell.mean().item(),
        mean_best_q_at_k_cell=best_q_cell.mean().item(),
        mean_mode_at_k_cell=mode_cell.mean().item(),
        num_puzzles=B,
        K=K,
    )


def compute_metrics_from_result(
    result,  # PTRMBatchResult (avoid circular import)
    labels: torch.Tensor,
    ignore_id: int = 0,
) -> MetricsResult:
    """
    Convenience wrapper: compute metrics directly from a PTRMBatchResult.

    Args:
        result: PTRMBatchResult from PTRMInference.run().
        labels: (B, seq_len) ground truth labels.
        ignore_id: Token ID to ignore.

    Returns:
        MetricsResult with all metrics.
    """
    return compute_metrics(
        all_predictions=result.all_predictions,
        all_q_values=result.all_q_values,
        best_q_predictions=result.best_q_predictions,
        mode_predictions=result.mode_predictions,
        labels=labels,
        ignore_id=ignore_id,
    )


def format_metrics(metrics: MetricsResult) -> str:
    """Format metrics as a human-readable string."""
    lines = [
        f"PTRM Evaluation Results (K={metrics.K}, N={metrics.num_puzzles})",
        f"{'='*50}",
        f"{'Metric':<25s} {'Cell Acc':>10s} {'Exact Acc':>10s}",
        f"{'-'*50}",
        f"{'pass@K (oracle):':<25s} {metrics.mean_pass_at_k_cell:>9.2%} {metrics.mean_pass_at_k_exact:>9.2%}",
        f"{'best-Q@K:':<25s} {metrics.mean_best_q_at_k_cell:>9.2%} {metrics.mean_best_q_at_k_exact:>9.2%}",
        f"{'mode@K:':<25s} {metrics.mean_mode_at_k_cell:>9.2%} {metrics.mean_mode_at_k_exact:>9.2%}",
        f"{'='*50}",
    ]
    return "\n".join(lines)
