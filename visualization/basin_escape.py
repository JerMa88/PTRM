"""
Basin Escape Analysis Visualization.

Visualizes how noise injection enables the model to escape local minima
(fixed-point "basins") in latent space. Computes pairwise distances between
rollout endpoints and correlates with Q-value diversity, showing that noise
creates meaningfully different solution trajectories rather than just
adding random perturbation.
"""

import os
from typing import Optional

import torch
import numpy as np
import matplotlib.pyplot as plt


def compute_basin_statistics(
    latent_trajectories: torch.Tensor,
    q_values: torch.Tensor,
    puzzle_idx: int = 0,
) -> dict:
    """
    Compute basin escape statistics for a single puzzle.

    Args:
        latent_trajectories: (B, K, D, latent_dim) latent states.
        q_values: (B, K) final Q-halt logits.
        puzzle_idx: Which puzzle to analyze.

    Returns:
        Dict with keys:
          - endpoint_distances: (K, K) pairwise L2 distances between final states
          - q_spread: Standard deviation of Q-values across rollouts
          - trajectory_divergence: Mean pairwise distance between full trajectories
          - endpoint_cluster_count: Number of distinct clusters (heuristic)
    """
    traj = latent_trajectories[puzzle_idx].float()  # (K, D, latent_dim)
    q_vals = q_values[puzzle_idx].float()  # (K,)

    K, D, latent_dim = traj.shape

    # Endpoint pairwise L2 distances
    endpoints = traj[:, -1, :]  # (K, latent_dim)
    diffs = endpoints.unsqueeze(0) - endpoints.unsqueeze(1)  # (K, K, latent_dim)
    endpoint_distances = torch.norm(diffs, dim=-1)  # (K, K)

    # Q-value spread
    q_spread = q_vals.std(correction=0).item()

    # Trajectory divergence: mean pairwise L2 across all steps
    traj_flat = traj.reshape(K, -1)  # (K, D*latent_dim)
    traj_diffs = traj_flat.unsqueeze(0) - traj_flat.unsqueeze(1)
    trajectory_divergence = torch.norm(traj_diffs, dim=-1).mean().item()

    # Cluster count heuristic: number of endpoints farther than median distance
    median_dist = endpoint_distances[endpoint_distances > 0].median().item() if K > 1 else 0
    if median_dist > 0:
        # Simple thresholding: count groups of nearby endpoints
        cluster_count = min(K, max(1, int(endpoint_distances.max().item() / median_dist)))
    else:
        cluster_count = 1

    return {
        "endpoint_distances": endpoint_distances.cpu().numpy(),
        "q_spread": q_spread,
        "trajectory_divergence": trajectory_divergence,
        "endpoint_cluster_count": cluster_count,
    }


def plot_basin_escape(
    latent_trajectories: torch.Tensor,
    q_values: torch.Tensor,
    puzzle_idx: int = 0,
    output_path: Optional[str] = None,
    title: Optional[str] = None,
    figsize: tuple = (12, 5),
) -> plt.Figure:
    """
    Plot basin escape analysis: distance heatmap + Q-value distribution.

    Args:
        latent_trajectories: (B, K, D, latent_dim) latent states.
        q_values: (B, K) final Q-halt logits.
        puzzle_idx: Which puzzle to visualize.
        output_path: If set, save figure.
        title: Custom title.
        figsize: Figure size.

    Returns:
        matplotlib Figure.
    """
    stats = compute_basin_statistics(latent_trajectories, q_values, puzzle_idx)
    q_vals = q_values[puzzle_idx].float().cpu().numpy()
    K = len(q_vals)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Left: Endpoint pairwise distance heatmap
    im = ax1.imshow(stats["endpoint_distances"], cmap='viridis', aspect='auto')
    plt.colorbar(im, ax=ax1, label='L2 Distance')
    ax1.set_xlabel('Rollout k')
    ax1.set_ylabel('Rollout k')
    ax1.set_title('Endpoint Pairwise Distances')

    # Right: Q-value distribution with diversity stats
    sorted_q = np.sort(q_vals)[::-1]
    colors = plt.cm.RdYlGn(np.linspace(0, 1, K))
    ax2.bar(range(K), sorted_q, color=colors)
    ax2.set_xlabel('Rollout (sorted by Q)')
    ax2.set_ylabel('Q-halt logit')
    ax2.set_title(f'Q-Value Distribution\n'
                  f'(σ_Q={stats["q_spread"]:.3f}, '
                  f'divergence={stats["trajectory_divergence"]:.1f})')
    ax2.grid(True, alpha=0.3, axis='y')

    fig.suptitle(title or 'Basin Escape Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches='tight')

    return fig
