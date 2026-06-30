"""
Tests for Component 6: Visualization Suite.

Tests that all 5 visualization functions produce valid matplotlib Figures
with correct structure, using synthetic data (no model required).

Test categories:
  1. PCA Latent Dynamics
  2. Q-Value Tracking
  3. Width Scaling Curves
  4. Noise Ablation Sweep
  5. Basin Escape Analysis
"""

import os
import sys

import pytest
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for tests
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from visualization.pca_latent import plot_pca_latent_dynamics
from visualization.q_tracking import plot_q_tracking
from visualization.width_scaling import plot_width_scaling
from visualization.noise_ablation import plot_noise_ablation
from visualization.basin_escape import compute_basin_statistics, plot_basin_escape


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def synthetic_trajectories():
    """Create synthetic latent trajectories: (B=2, K=5, D=4, latent_dim=16)."""
    torch.manual_seed(42)
    B, K, D, latent_dim = 2, 5, 4, 16
    traj = torch.randn(B, K, D, latent_dim)
    q_vals = torch.randn(B, K)
    return traj, q_vals


@pytest.fixture
def synthetic_step_q_values():
    """Create synthetic per-step Q-values: (B=2, K=5, D=4)."""
    torch.manual_seed(42)
    return torch.randn(2, 5, 4)


# =============================================================================
# 1. PCA Latent Dynamics
# =============================================================================

class TestPCALatentDynamics:

    def test_returns_figure(self, synthetic_trajectories):
        traj, q_vals = synthetic_trajectories
        fig = plot_pca_latent_dynamics(traj, q_vals)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_saves_to_file(self, synthetic_trajectories, tmp_path):
        traj, q_vals = synthetic_trajectories
        output = str(tmp_path / "pca.png")
        fig = plot_pca_latent_dynamics(traj, q_vals, output_path=output)
        assert os.path.exists(output)
        assert os.path.getsize(output) > 0
        plt.close(fig)

    def test_custom_title(self, synthetic_trajectories):
        traj, q_vals = synthetic_trajectories
        fig = plot_pca_latent_dynamics(traj, q_vals, title="Custom Title")
        ax = fig.axes[0]
        assert ax.get_title() == "Custom Title"
        plt.close(fig)

    def test_max_rollouts(self, synthetic_trajectories):
        traj, q_vals = synthetic_trajectories
        fig = plot_pca_latent_dynamics(traj, q_vals, max_rollouts=2)
        # Should complete without error
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_puzzle_idx(self, synthetic_trajectories):
        traj, q_vals = synthetic_trajectories
        fig = plot_pca_latent_dynamics(traj, q_vals, puzzle_idx=1)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


# =============================================================================
# 2. Q-Value Tracking
# =============================================================================

class TestQTracking:

    def test_returns_figure(self, synthetic_step_q_values):
        fig = plot_q_tracking(synthetic_step_q_values)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_saves_to_file(self, synthetic_step_q_values, tmp_path):
        output = str(tmp_path / "q_tracking.png")
        fig = plot_q_tracking(synthetic_step_q_values, output_path=output)
        assert os.path.exists(output)
        plt.close(fig)

    def test_has_legend(self, synthetic_step_q_values):
        fig = plot_q_tracking(synthetic_step_q_values)
        ax = fig.axes[0]
        legend = ax.get_legend()
        assert legend is not None
        labels = [t.get_text() for t in legend.get_texts()]
        assert any("Best Q" in l for l in labels)
        assert any("Mean Q" in l for l in labels)
        plt.close(fig)

    def test_max_rollouts(self, synthetic_step_q_values):
        fig = plot_q_tracking(synthetic_step_q_values, max_rollouts=3)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


# =============================================================================
# 3. Width Scaling Curves
# =============================================================================

class TestWidthScaling:

    def test_returns_figure(self):
        K_vals = [1, 5, 10, 25, 50]
        fig = plot_width_scaling(
            K_vals,
            pass_at_k=[0.2, 0.5, 0.7, 0.85, 0.92],
            best_q_at_k=[0.15, 0.35, 0.50, 0.65, 0.73],
            mode_at_k=[0.18, 0.40, 0.55, 0.70, 0.78],
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_saves_to_file(self, tmp_path):
        output = str(tmp_path / "width_scaling.png")
        fig = plot_width_scaling(
            [1, 5, 10],
            pass_at_k=[0.2, 0.5, 0.7],
            best_q_at_k=[0.15, 0.35, 0.50],
            mode_at_k=[0.18, 0.40, 0.55],
            output_path=output,
        )
        assert os.path.exists(output)
        plt.close(fig)

    def test_baseline_line(self):
        fig = plot_width_scaling(
            [1, 5, 10],
            pass_at_k=[0.2, 0.5, 0.7],
            best_q_at_k=[0.15, 0.35, 0.50],
            mode_at_k=[0.18, 0.40, 0.55],
            baseline=0.10,
        )
        ax = fig.axes[0]
        legend_labels = [t.get_text() for t in ax.get_legend().get_texts()]
        assert any("Baseline" in l for l in legend_labels)
        plt.close(fig)

    def test_log_scale_for_large_range(self):
        K_vals = [1, 2, 4, 8, 16, 32, 64]
        fig = plot_width_scaling(
            K_vals,
            pass_at_k=[0.1, 0.2, 0.3, 0.5, 0.6, 0.8, 0.9],
            best_q_at_k=[0.08, 0.15, 0.25, 0.4, 0.5, 0.65, 0.75],
            mode_at_k=[0.09, 0.18, 0.28, 0.45, 0.55, 0.70, 0.80],
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


# =============================================================================
# 4. Noise Ablation Sweep
# =============================================================================

class TestNoiseAblation:

    def test_returns_figure(self):
        sigmas = [0.0, 0.1, 0.2, 0.3, 0.5, 1.0]
        fig = plot_noise_ablation(
            sigmas,
            pass_at_k=[0.3, 0.5, 0.7, 0.85, 0.75, 0.4],
            best_q_at_k=[0.25, 0.40, 0.55, 0.65, 0.50, 0.20],
            mode_at_k=[0.28, 0.45, 0.60, 0.70, 0.55, 0.25],
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_saves_to_file(self, tmp_path):
        output = str(tmp_path / "noise_ablation.png")
        fig = plot_noise_ablation(
            [0.0, 0.3, 1.0],
            pass_at_k=[0.3, 0.85, 0.4],
            best_q_at_k=[0.25, 0.65, 0.20],
            mode_at_k=[0.28, 0.70, 0.25],
            output_path=output,
        )
        assert os.path.exists(output)
        plt.close(fig)

    def test_optimal_sigma_marker(self):
        fig = plot_noise_ablation(
            [0.0, 0.3, 1.0],
            pass_at_k=[0.3, 0.85, 0.4],
            best_q_at_k=[0.25, 0.65, 0.20],
            mode_at_k=[0.28, 0.70, 0.25],
        )
        ax = fig.axes[0]
        legend_labels = [t.get_text() for t in ax.get_legend().get_texts()]
        assert any("σ=0.3" in l for l in legend_labels)
        plt.close(fig)


# =============================================================================
# 5. Basin Escape Analysis
# =============================================================================

class TestBasinEscape:

    def test_compute_statistics(self, synthetic_trajectories):
        traj, q_vals = synthetic_trajectories
        stats = compute_basin_statistics(traj, q_vals)

        assert "endpoint_distances" in stats
        assert "q_spread" in stats
        assert "trajectory_divergence" in stats
        assert "endpoint_cluster_count" in stats

        K = traj.shape[1]
        assert stats["endpoint_distances"].shape == (K, K)
        assert stats["q_spread"] >= 0
        assert stats["trajectory_divergence"] >= 0
        assert stats["endpoint_cluster_count"] >= 1

    def test_distance_matrix_symmetric(self, synthetic_trajectories):
        traj, q_vals = synthetic_trajectories
        stats = compute_basin_statistics(traj, q_vals)
        dists = stats["endpoint_distances"]
        np.testing.assert_allclose(dists, dists.T, atol=1e-6)

    def test_distance_matrix_diagonal_zero(self, synthetic_trajectories):
        traj, q_vals = synthetic_trajectories
        stats = compute_basin_statistics(traj, q_vals)
        dists = stats["endpoint_distances"]
        np.testing.assert_allclose(np.diag(dists), 0, atol=1e-6)

    def test_plot_returns_figure(self, synthetic_trajectories):
        traj, q_vals = synthetic_trajectories
        fig = plot_basin_escape(traj, q_vals)
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) >= 2  # Two subplots
        plt.close(fig)

    def test_plot_saves_to_file(self, synthetic_trajectories, tmp_path):
        traj, q_vals = synthetic_trajectories
        output = str(tmp_path / "basin.png")
        fig = plot_basin_escape(traj, q_vals, output_path=output)
        assert os.path.exists(output)
        plt.close(fig)

    def test_single_rollout(self):
        traj = torch.randn(1, 1, 4, 16)
        q_vals = torch.tensor([[0.5]])
        stats = compute_basin_statistics(traj, q_vals)
        assert stats["endpoint_cluster_count"] == 1
        assert stats["q_spread"] == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
