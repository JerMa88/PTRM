"""
Noise Ablation Sweep Visualization.

Plots accuracy metrics as a function of noise σ (sigma), showing the
optimal noise level for each benchmark. The paper demonstrates that
moderate noise (σ ≈ 0.3) provides the best balance between diversity
and quality.
"""

import os
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt


def plot_noise_ablation(
    sigma_values: list[float],
    pass_at_k: list[float],
    best_q_at_k: list[float],
    mode_at_k: list[float],
    K: int = 25,
    metric_type: str = "exact",
    output_path: Optional[str] = None,
    title: Optional[str] = None,
    figsize: tuple = (10, 6),
) -> plt.Figure:
    """
    Plot accuracy vs noise σ (noise ablation sweep).

    Args:
        sigma_values: List of σ values tested.
        pass_at_k: pass@K accuracy for each σ.
        best_q_at_k: best-Q@K accuracy for each σ.
        mode_at_k: mode@K accuracy for each σ.
        K: Number of rollouts used (for title).
        metric_type: "exact" or "cell" for y-axis label.
        output_path: If set, save figure.
        title: Custom title.
        figsize: Figure size.

    Returns:
        matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(sigma_values, pass_at_k, 'o-', color='#2ecc71', linewidth=2,
            markersize=6, label='pass@K')
    ax.plot(sigma_values, best_q_at_k, 's-', color='#3498db', linewidth=2,
            markersize=6, label='best-Q@K')
    ax.plot(sigma_values, mode_at_k, '^-', color='#e74c3c', linewidth=2,
            markersize=6, label='mode@K')

    # Mark optimal σ for best-Q
    best_idx = np.argmax(best_q_at_k)
    ax.axvline(x=sigma_values[best_idx], color='#3498db', linestyle=':',
               alpha=0.5, label=f'Best σ={sigma_values[best_idx]}')

    ax.set_xlabel('Noise σ')
    ax.set_ylabel(f'{"Exact" if metric_type == "exact" else "Cell"} Accuracy')
    ax.set_title(title or f'Noise Ablation (K={K})')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)

    plt.tight_layout()
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches='tight')

    return fig
