"""
Width Scaling Curves Visualization.

Plots accuracy metrics (pass@K, best-Q@K, mode@K) as a function of K,
showing how increasing the number of parallel rollouts improves performance.
This is the central result from the paper (Figure 2, Table 1).
"""

import os
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt


def plot_width_scaling(
    K_values: list[int],
    pass_at_k: list[float],
    best_q_at_k: list[float],
    mode_at_k: list[float],
    metric_type: str = "exact",
    output_path: Optional[str] = None,
    title: Optional[str] = None,
    figsize: tuple = (10, 6),
    baseline: Optional[float] = None,
    baseline_label: str = "Baseline (K=1)",
) -> plt.Figure:
    """
    Plot accuracy vs K (width scaling curves).

    Args:
        K_values: List of K values tested.
        pass_at_k: pass@K accuracy for each K.
        best_q_at_k: best-Q@K accuracy for each K.
        mode_at_k: mode@K accuracy for each K.
        metric_type: "exact" or "cell" for y-axis label.
        output_path: If set, save figure.
        title: Custom title.
        figsize: Figure size.
        baseline: Optional baseline accuracy to show as horizontal line.
        baseline_label: Label for baseline line.

    Returns:
        matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(K_values, pass_at_k, 'o-', color='#2ecc71', linewidth=2,
            markersize=6, label='pass@K (oracle)')
    ax.plot(K_values, best_q_at_k, 's-', color='#3498db', linewidth=2,
            markersize=6, label='best-Q@K')
    ax.plot(K_values, mode_at_k, '^-', color='#e74c3c', linewidth=2,
            markersize=6, label='mode@K')

    if baseline is not None:
        ax.axhline(y=baseline, color='gray', linestyle=':', linewidth=1.5,
                   label=baseline_label)

    ax.set_xlabel('K (number of parallel rollouts)')
    ax.set_ylabel(f'{"Exact" if metric_type == "exact" else "Cell"} Accuracy')
    ax.set_title(title or 'PTRM Width Scaling')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)

    # Log scale for K if range is large
    if len(K_values) > 1 and max(K_values) / min(K_values) > 10:
        ax.set_xscale('log', base=2)
        ax.set_xticks(K_values)
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())

    plt.tight_layout()
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches='tight')

    return fig
