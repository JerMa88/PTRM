"""
Q-Value Tracking Visualization.

Plots Q-halt logit trajectories across D supervision steps for K rollouts,
showing how the Q-head's confidence evolves as recursive processing deepens.

This corresponds to the paper's analysis of Q-value dynamics and the
observation that high-Q rollouts tend to converge to correct solutions.
"""

import os
from typing import Optional

import torch
import numpy as np
import matplotlib.pyplot as plt


def plot_q_tracking(
    step_q_values: torch.Tensor,
    puzzle_idx: int = 0,
    max_rollouts: int = 25,
    output_path: Optional[str] = None,
    title: Optional[str] = None,
    figsize: tuple = (10, 6),
) -> plt.Figure:
    """
    Plot Q-halt logit trajectories across supervision steps.

    Args:
        step_q_values: (B, K, D) Q-halt logits per step from PTRMBatchResult.
        puzzle_idx: Which puzzle to visualize.
        max_rollouts: Max rollouts to plot.
        output_path: If set, save figure.
        title: Custom title.
        figsize: Figure size.

    Returns:
        matplotlib Figure.
    """
    q_vals = step_q_values[puzzle_idx].float().cpu().numpy()  # (K, D)
    K, D = q_vals.shape
    K_plot = min(K, max_rollouts)

    # Final Q-values for coloring
    final_q = q_vals[:K_plot, -1]
    q_min, q_max = final_q.min(), final_q.max()
    if q_max - q_min > 1e-6:
        q_norm = (final_q - q_min) / (q_max - q_min)
    else:
        q_norm = np.ones(K_plot) * 0.5

    fig, ax = plt.subplots(figsize=figsize)
    cmap = plt.cm.RdYlGn
    steps = np.arange(1, D + 1)

    for k in range(K_plot):
        color = cmap(q_norm[k])
        ax.plot(steps, q_vals[k], color=color, alpha=0.5, linewidth=0.8)

    # Highlight best-Q trajectory
    best_k = np.argmax(final_q)
    ax.plot(steps, q_vals[best_k], color='black', linewidth=2.0,
            label=f'Best Q (rollout {best_k})', zorder=10)

    # Mean Q trajectory
    mean_q = q_vals[:K_plot].mean(axis=0)
    ax.plot(steps, mean_q, color='blue', linewidth=1.5,
            linestyle='--', label='Mean Q', zorder=9)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(q_min, q_max))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label='Final Q-halt logit')

    ax.set_xlabel('Supervision Step d')
    ax.set_ylabel('Q-halt logit')
    ax.set_title(title or f'Q-Value Dynamics (K={K_plot}, D={D})')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches='tight')

    return fig
