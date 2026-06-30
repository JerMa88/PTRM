"""
Tests for Component 3: Evaluation & Metrics.

Test categories:
  1. compute_cell_accuracy — token-level accuracy with ignore masking
  2. compute_exact_accuracy — full-sequence match with ignore masking
  3. compute_metrics — full metrics pipeline with synthetic data
  4. compute_metrics_from_result — PTRMBatchResult integration
  5. format_metrics — human-readable output formatting
  6. Edge cases — all correct, all wrong, single token, all ignored
"""

import os
import sys

import pytest
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from evaluation.metrics import (
    compute_cell_accuracy,
    compute_exact_accuracy,
    compute_metrics,
    compute_metrics_from_result,
    format_metrics,
    MetricsResult,
)


# =============================================================================
# 1. compute_cell_accuracy tests
# =============================================================================

class TestCellAccuracy:

    def test_perfect_match(self):
        preds = torch.tensor([1, 2, 3, 4, 5])
        labels = torch.tensor([1, 2, 3, 4, 5])
        acc = compute_cell_accuracy(preds, labels, ignore_id=0)
        assert acc.item() == pytest.approx(1.0)

    def test_all_wrong(self):
        preds = torch.tensor([5, 4, 3, 2, 1])
        labels = torch.tensor([1, 2, 7, 8, 9])
        acc = compute_cell_accuracy(preds, labels, ignore_id=0)
        assert acc.item() == pytest.approx(0.0)

    def test_partial_match(self):
        preds = torch.tensor([1, 2, 9, 9, 5])
        labels = torch.tensor([1, 2, 3, 4, 5])
        acc = compute_cell_accuracy(preds, labels, ignore_id=0)
        assert acc.item() == pytest.approx(3.0 / 5.0)

    def test_ignore_id_excluded(self):
        # Labels with ignore_id=0 at positions 2 and 3
        preds = torch.tensor([1, 2, 9, 9, 5])
        labels = torch.tensor([1, 2, 0, 0, 5])
        acc = compute_cell_accuracy(preds, labels, ignore_id=0)
        # 3 valid tokens (1, 2, 5), all correct → 100%
        assert acc.item() == pytest.approx(1.0)

    def test_ignore_id_with_wrong_predictions(self):
        preds = torch.tensor([1, 9, 9, 9, 9])
        labels = torch.tensor([1, 2, 0, 0, 5])
        acc = compute_cell_accuracy(preds, labels, ignore_id=0)
        # 3 valid tokens, 1 correct (pos 0) → 1/3
        assert acc.item() == pytest.approx(1.0 / 3.0)

    def test_all_ignored(self):
        preds = torch.tensor([1, 2, 3])
        labels = torch.tensor([0, 0, 0])
        acc = compute_cell_accuracy(preds, labels, ignore_id=0)
        # No valid tokens — should return 0.0 (clamped denominator prevents NaN)
        assert acc.item() == pytest.approx(0.0)

    def test_batched_input(self):
        preds = torch.tensor([[1, 2, 3], [4, 5, 6]])
        labels = torch.tensor([[1, 2, 3], [4, 9, 9]])
        acc = compute_cell_accuracy(preds, labels, ignore_id=0)
        assert acc.shape == (2,)
        assert acc[0].item() == pytest.approx(1.0)
        assert acc[1].item() == pytest.approx(1.0 / 3.0)

    def test_3d_input(self):
        # (B, K, seq_len) shape
        preds = torch.tensor([[[1, 2, 3], [1, 9, 3]]])
        labels = torch.tensor([[[1, 2, 3], [1, 2, 3]]])
        acc = compute_cell_accuracy(preds, labels, ignore_id=0)
        assert acc.shape == (1, 2)
        assert acc[0, 0].item() == pytest.approx(1.0)
        assert acc[0, 1].item() == pytest.approx(2.0 / 3.0)

    def test_custom_ignore_id(self):
        preds = torch.tensor([1, 2, 3, 4])
        labels = torch.tensor([1, 2, -100, -100])
        acc = compute_cell_accuracy(preds, labels, ignore_id=-100)
        assert acc.item() == pytest.approx(1.0)


# =============================================================================
# 2. compute_exact_accuracy tests
# =============================================================================

class TestExactAccuracy:

    def test_perfect_match(self):
        preds = torch.tensor([1, 2, 3, 4, 5])
        labels = torch.tensor([1, 2, 3, 4, 5])
        acc = compute_exact_accuracy(preds, labels, ignore_id=0)
        assert acc.item() == pytest.approx(1.0)

    def test_single_token_wrong(self):
        preds = torch.tensor([1, 2, 9, 4, 5])
        labels = torch.tensor([1, 2, 3, 4, 5])
        acc = compute_exact_accuracy(preds, labels, ignore_id=0)
        assert acc.item() == pytest.approx(0.0)

    def test_all_wrong(self):
        preds = torch.tensor([9, 9, 9, 9, 9])
        labels = torch.tensor([1, 2, 3, 4, 5])
        acc = compute_exact_accuracy(preds, labels, ignore_id=0)
        assert acc.item() == pytest.approx(0.0)

    def test_ignored_positions_dont_count(self):
        preds = torch.tensor([1, 2, 99, 99, 5])
        labels = torch.tensor([1, 2, 0, 0, 5])
        acc = compute_exact_accuracy(preds, labels, ignore_id=0)
        # Positions 2,3 are ignored; positions 0,1,4 match → exact
        assert acc.item() == pytest.approx(1.0)

    def test_ignored_but_wrong_elsewhere(self):
        preds = torch.tensor([9, 2, 99, 99, 5])
        labels = torch.tensor([1, 2, 0, 0, 5])
        acc = compute_exact_accuracy(preds, labels, ignore_id=0)
        # Position 0 is wrong → not exact
        assert acc.item() == pytest.approx(0.0)

    def test_all_ignored(self):
        preds = torch.tensor([9, 9, 9])
        labels = torch.tensor([0, 0, 0])
        acc = compute_exact_accuracy(preds, labels, ignore_id=0)
        # All ignored → vacuously true, exact match
        assert acc.item() == pytest.approx(1.0)

    def test_batched_input(self):
        preds = torch.tensor([[1, 2, 3], [1, 9, 3]])
        labels = torch.tensor([[1, 2, 3], [1, 2, 3]])
        acc = compute_exact_accuracy(preds, labels, ignore_id=0)
        assert acc.shape == (2,)
        assert acc[0].item() == pytest.approx(1.0)
        assert acc[1].item() == pytest.approx(0.0)


# =============================================================================
# 3. compute_metrics tests
# =============================================================================

class TestComputeMetrics:

    def _make_synthetic_data(self):
        """
        Create synthetic data where:
          B=2 puzzles, K=3 rollouts, seq_len=4

          Puzzle 0: label = [1, 2, 3, 4]
            Rollout 0: [1, 2, 3, 4] (exact match)      Q = 0.5
            Rollout 1: [1, 2, 9, 9] (2/4 correct)      Q = 0.8 (highest Q but wrong)
            Rollout 2: [1, 2, 3, 9] (3/4 correct)      Q = 0.3

          Puzzle 1: label = [5, 6, 7, 8]
            Rollout 0: [5, 6, 7, 9] (3/4 correct)      Q = 0.1
            Rollout 1: [5, 6, 7, 9] (3/4 correct)      Q = 0.9 (highest Q)
            Rollout 2: [5, 6, 7, 9] (3/4 correct)      Q = 0.2
        """
        all_predictions = torch.tensor([
            [[1, 2, 3, 4], [1, 2, 9, 9], [1, 2, 3, 9]],
            [[5, 6, 7, 9], [5, 6, 7, 9], [5, 6, 7, 9]],
        ])  # (2, 3, 4)

        all_q_values = torch.tensor([
            [0.5, 0.8, 0.3],
            [0.1, 0.9, 0.2],
        ])  # (2, 3)

        labels = torch.tensor([
            [1, 2, 3, 4],
            [5, 6, 7, 8],
        ])  # (2, 4)

        # best-Q selections: puzzle 0 → rollout 1 (Q=0.8), puzzle 1 → rollout 1 (Q=0.9)
        best_q_predictions = torch.tensor([
            [1, 2, 9, 9],
            [5, 6, 7, 9],
        ])

        # mode selections: puzzle 0 → first unique = [1,2,3,4], puzzle 1 → [5,6,7,9] (all same)
        mode_predictions = torch.tensor([
            [1, 2, 3, 4],   # From the synthetic test, the mode could vary
            [5, 6, 7, 9],
        ])

        return all_predictions, all_q_values, best_q_predictions, mode_predictions, labels

    def test_result_shape(self):
        all_preds, all_q, best_q, mode, labels = self._make_synthetic_data()
        result = compute_metrics(all_preds, all_q, best_q, mode, labels, ignore_id=0)

        assert result.pass_at_k_exact.shape == (2,)
        assert result.best_q_at_k_exact.shape == (2,)
        assert result.mode_at_k_exact.shape == (2,)
        assert result.num_puzzles == 2
        assert result.K == 3

    def test_pass_at_k_exact(self):
        all_preds, all_q, best_q, mode, labels = self._make_synthetic_data()
        result = compute_metrics(all_preds, all_q, best_q, mode, labels, ignore_id=0)

        # Puzzle 0: rollout 0 is exact match → pass@K = 1.0
        assert result.pass_at_k_exact[0].item() == pytest.approx(1.0)
        # Puzzle 1: no exact match → pass@K = 0.0
        assert result.pass_at_k_exact[1].item() == pytest.approx(0.0)

    def test_pass_at_k_cell(self):
        all_preds, all_q, best_q, mode, labels = self._make_synthetic_data()
        result = compute_metrics(all_preds, all_q, best_q, mode, labels, ignore_id=0)

        # Puzzle 0: best cell acc = 4/4 = 1.0 (rollout 0)
        assert result.pass_at_k_cell[0].item() == pytest.approx(1.0)
        # Puzzle 1: best cell acc = 3/4 = 0.75
        assert result.pass_at_k_cell[1].item() == pytest.approx(0.75)

    def test_best_q_exact(self):
        all_preds, all_q, best_q, mode, labels = self._make_synthetic_data()
        result = compute_metrics(all_preds, all_q, best_q, mode, labels, ignore_id=0)

        # Puzzle 0: best-Q rollout is [1,2,9,9] → not exact → 0.0
        assert result.best_q_at_k_exact[0].item() == pytest.approx(0.0)
        # Puzzle 1: best-Q rollout is [5,6,7,9] → not exact → 0.0
        assert result.best_q_at_k_exact[1].item() == pytest.approx(0.0)

    def test_best_q_cell(self):
        all_preds, all_q, best_q, mode, labels = self._make_synthetic_data()
        result = compute_metrics(all_preds, all_q, best_q, mode, labels, ignore_id=0)

        # Puzzle 0: best-Q = [1,2,9,9] vs [1,2,3,4] → 2/4 = 0.5
        assert result.best_q_at_k_cell[0].item() == pytest.approx(0.5)
        # Puzzle 1: best-Q = [5,6,7,9] vs [5,6,7,8] → 3/4 = 0.75
        assert result.best_q_at_k_cell[1].item() == pytest.approx(0.75)

    def test_mode_exact(self):
        all_preds, all_q, best_q, mode, labels = self._make_synthetic_data()
        result = compute_metrics(all_preds, all_q, best_q, mode, labels, ignore_id=0)

        # Puzzle 0: mode = [1,2,3,4] → exact match → 1.0
        assert result.mode_at_k_exact[0].item() == pytest.approx(1.0)
        # Puzzle 1: mode = [5,6,7,9] → not exact → 0.0
        assert result.mode_at_k_exact[1].item() == pytest.approx(0.0)

    def test_mean_aggregation(self):
        all_preds, all_q, best_q, mode, labels = self._make_synthetic_data()
        result = compute_metrics(all_preds, all_q, best_q, mode, labels, ignore_id=0)

        assert result.mean_pass_at_k_exact == pytest.approx(0.5)  # (1 + 0) / 2
        assert result.mean_best_q_at_k_exact == pytest.approx(0.0)  # (0 + 0) / 2
        assert result.mean_mode_at_k_exact == pytest.approx(0.5)  # (1 + 0) / 2

    def test_with_ignore_id(self):
        """Test that ignore_id properly masks tokens."""
        all_preds = torch.tensor([[[1, 2, 9, 9]]])  # (1, 1, 4)
        all_q = torch.tensor([[0.5]])                 # (1, 1)
        labels = torch.tensor([[1, 2, 0, 0]])         # (1, 4) — last 2 ignored
        best_q = torch.tensor([[1, 2, 9, 9]])
        mode = torch.tensor([[1, 2, 9, 9]])

        result = compute_metrics(all_preds, all_q, best_q, mode, labels, ignore_id=0)

        # Only positions 0,1 are valid, both correct → exact match
        assert result.pass_at_k_exact[0].item() == pytest.approx(1.0)
        assert result.best_q_at_k_cell[0].item() == pytest.approx(1.0)


class TestComputeMetricsFromResult:

    def test_from_batch_result(self):
        """Test the convenience wrapper using a mock PTRMBatchResult."""
        from inference.ptrm_inference import PTRMBatchResult

        result = PTRMBatchResult(
            all_predictions=torch.tensor([[[1, 2, 3], [1, 2, 9]]]),
            all_q_values=torch.tensor([[0.5, 0.8]]),
            best_q_predictions=torch.tensor([[1, 2, 9]]),
            mode_predictions=torch.tensor([[1, 2, 3]]),
        )
        labels = torch.tensor([[1, 2, 3]])

        metrics = compute_metrics_from_result(result, labels, ignore_id=0)
        assert metrics.mean_pass_at_k_exact == pytest.approx(1.0)
        assert metrics.mean_mode_at_k_exact == pytest.approx(1.0)
        assert metrics.mean_best_q_at_k_exact == pytest.approx(0.0)


# =============================================================================
# 4. format_metrics tests
# =============================================================================

class TestFormatMetrics:

    def test_format_contains_key_info(self):
        metrics = MetricsResult(
            pass_at_k_exact=torch.tensor([1.0, 0.0]),
            best_q_at_k_exact=torch.tensor([0.0, 0.0]),
            mode_at_k_exact=torch.tensor([1.0, 0.0]),
            pass_at_k_cell=torch.tensor([1.0, 0.75]),
            best_q_at_k_cell=torch.tensor([0.5, 0.75]),
            mode_at_k_cell=torch.tensor([1.0, 0.75]),
            mean_pass_at_k_exact=0.5,
            mean_best_q_at_k_exact=0.0,
            mean_mode_at_k_exact=0.5,
            mean_pass_at_k_cell=0.875,
            mean_best_q_at_k_cell=0.625,
            mean_mode_at_k_cell=0.875,
            num_puzzles=2,
            K=3,
        )
        output = format_metrics(metrics)
        assert "K=3" in output
        assert "N=2" in output
        assert "pass@K" in output
        assert "best-Q@K" in output
        assert "mode@K" in output
        assert "50.00%" in output   # pass@K exact = 0.5

    def test_format_is_multiline(self):
        metrics = MetricsResult(
            pass_at_k_exact=torch.tensor([1.0]),
            best_q_at_k_exact=torch.tensor([1.0]),
            mode_at_k_exact=torch.tensor([1.0]),
            pass_at_k_cell=torch.tensor([1.0]),
            best_q_at_k_cell=torch.tensor([1.0]),
            mode_at_k_cell=torch.tensor([1.0]),
            mean_pass_at_k_exact=1.0,
            mean_best_q_at_k_exact=1.0,
            mean_mode_at_k_exact=1.0,
            mean_pass_at_k_cell=1.0,
            mean_best_q_at_k_cell=1.0,
            mean_mode_at_k_cell=1.0,
            num_puzzles=1,
            K=5,
        )
        output = format_metrics(metrics)
        lines = output.strip().split("\n")
        assert len(lines) >= 7  # Header + separator + 3 metric lines + separators


# =============================================================================
# 5. Edge case tests
# =============================================================================

class TestEdgeCases:

    def test_all_correct_all_rollouts(self):
        B, K, seq_len = 1, 3, 5
        label = torch.tensor([[1, 2, 3, 4, 5]])
        all_preds = label.unsqueeze(1).expand(B, K, seq_len)
        all_q = torch.tensor([[0.5, 0.8, 0.3]])
        best_q = label.clone()
        mode = label.clone()

        result = compute_metrics(all_preds, all_q, best_q, mode, label, ignore_id=0)
        assert result.mean_pass_at_k_exact == pytest.approx(1.0)
        assert result.mean_best_q_at_k_exact == pytest.approx(1.0)
        assert result.mean_mode_at_k_exact == pytest.approx(1.0)
        assert result.mean_pass_at_k_cell == pytest.approx(1.0)

    def test_all_wrong_all_rollouts(self):
        B, K, seq_len = 1, 3, 5
        label = torch.tensor([[1, 2, 3, 4, 5]])
        all_preds = torch.tensor([[[9, 9, 9, 9, 9], [8, 8, 8, 8, 8], [7, 7, 7, 7, 7]]])
        all_q = torch.tensor([[0.5, 0.8, 0.3]])
        best_q = torch.tensor([[8, 8, 8, 8, 8]])
        mode = torch.tensor([[9, 9, 9, 9, 9]])

        result = compute_metrics(all_preds, all_q, best_q, mode, label, ignore_id=0)
        assert result.mean_pass_at_k_exact == pytest.approx(0.0)
        assert result.mean_best_q_at_k_exact == pytest.approx(0.0)
        assert result.mean_mode_at_k_exact == pytest.approx(0.0)
        assert result.mean_pass_at_k_cell == pytest.approx(0.0)

    def test_k_equals_1(self):
        label = torch.tensor([[1, 2, 3]])
        all_preds = torch.tensor([[[1, 2, 3]]])  # K=1
        all_q = torch.tensor([[0.5]])
        best_q = torch.tensor([[1, 2, 3]])
        mode = torch.tensor([[1, 2, 3]])

        result = compute_metrics(all_preds, all_q, best_q, mode, label, ignore_id=0)
        assert result.K == 1
        # All three metrics should agree for K=1
        assert result.mean_pass_at_k_exact == result.mean_best_q_at_k_exact
        assert result.mean_pass_at_k_exact == result.mean_mode_at_k_exact

    def test_single_token_sequence(self):
        label = torch.tensor([[5]])
        all_preds = torch.tensor([[[5], [3], [5]]])  # K=3
        all_q = torch.tensor([[0.1, 0.9, 0.2]])
        best_q = torch.tensor([[3]])   # Highest Q but wrong
        mode = torch.tensor([[5]])     # Mode is correct

        result = compute_metrics(all_preds, all_q, best_q, mode, label, ignore_id=0)
        assert result.mean_pass_at_k_exact == pytest.approx(1.0)   # Rollout 0 or 2
        assert result.mean_best_q_at_k_exact == pytest.approx(0.0)  # Best Q is wrong
        assert result.mean_mode_at_k_exact == pytest.approx(1.0)    # Mode is correct

    def test_large_batch(self):
        B, K, seq_len = 100, 10, 20
        labels = torch.randint(1, 10, (B, seq_len))  # No zeros (no ignoring)
        all_preds = torch.randint(1, 10, (B, K, seq_len))
        all_q = torch.randn(B, K)
        best_q = torch.randint(1, 10, (B, seq_len))
        mode = torch.randint(1, 10, (B, seq_len))

        result = compute_metrics(all_preds, all_q, best_q, mode, labels, ignore_id=0)
        # Just verify shapes and ranges
        assert 0.0 <= result.mean_pass_at_k_exact <= 1.0
        assert 0.0 <= result.mean_best_q_at_k_cell <= 1.0
        assert result.num_puzzles == 100
        assert result.K == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
