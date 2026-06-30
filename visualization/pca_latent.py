"""
PCA Latent Dynamics Visualization.

Implements Figure 1 from the paper: 2D PCA projection of z_H[:, 0, :]
latent trajectories across D supervision steps for K rollouts, showing
how noise injection creates diverse exploration paths through latent space.

Color coding: trajectories colored by final Q-value (warm = high Q = confident).
"""

import os
from typing import Optional

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def plot_pca_latent_dynamics(
    latent_trajectories: torch.Tensor,
    q_values: torch.Tensor,
    puzzle_idx: int = 0,
    max_rollouts: int = 25,
    output_path: Optional[str] = None,
    title: Optional[str] = None,
    figsize: tuple = (10, 8),
) -> plt.Figure:
    """
    Plot 2D PCA projection of latent trajectories.

    Args:
        latent_trajectories: (B, K, D, latent_dim) from PTRMBatchResult.
        q_values: (B, K) Q-halt logits per rollout.
        puzzle_idx: Which puzzle in the batch to visualize.
        max_rollouts: Max number of rollouts to plot (for clarity).
        output_path: If set, save figure to this path.
        title: Custom plot title.
        figsize: Figure size.

    Returns:
        matplotlib Figure.
    """
    # Extract single puzzle's trajectories: (K, D, latent_dim)
    traj = latent_trajectories[puzzle_idx].float().cpu().numpy()
    q_vals = q_values[puzzle_idx].float().cpu().numpy()

    K, D, latent_dim = traj.shape
    K_plot = min(K, max_rollouts)

    # Fit PCA on all latent vectors across all rollouts and steps
    all_points = traj[:K_plot].reshape(-1, latent_dim)  # (K_plot * D, latent_dim)
    pca = PCA(n_components=2)
    projected = pca.fit_transform(all_points)  # (K_plot * D, 2)
    projected = projected.reshape(K_plot, D, 2)

    # Normalize Q-values for colormap
    q_min, q_max = q_vals[:K_plot].min(), q_vals[:K_plot].max()
    if q_max - q_min > 1e-6:
        q_norm = (q_vals[:K_plot] - q_min) / (q_max - q_min)
    else:
        q_norm = np.ones(K_plot) * 0.5

    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    cmap = plt.cm.RdYlGn  # Red (low Q) -> Green (high Q)

    for k in range(K_plot):
        color = cmap(q_norm[k])
        ax.plot(projected[k, :, 0], projected[k, :, 1],
                color=color, alpha=0.6, linewidth=0.8)
        # Start marker
        ax.scatter(projected[k, 0, 0], projected[k, 0, 1],
                   color=color, marker='o', s=20, zorder=5)
        # End marker
        ax.scatter(projected[k, -1, 0], projected[k, -1, 1],
                   color=color, marker='s', s=30, zorder=5, edgecolors='black', linewidths=0.5)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(q_min, q_max))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, label='Q-halt logit')

    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var)')
    ax.set_title(title or f'PTRM Latent Dynamics (K={K_plot}, D={D})')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches='tight')

    return fig
