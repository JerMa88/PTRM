"""
PTRM Core Inference Engine.

Implements the Probabilistic Tiny Recursive Model algorithm (Algorithm 1,
Figure 4 from arXiv:2605.19943):

    For k = 1..K in parallel:
        Initialize z_k from model's init state
        For t = 1..D:
            z_k.z_H += ε,  ε ~ N(0, σ²I)     # Noise injection into latent
            z_k, y_k = inner(z_k, x)           # One supervision step
        ŷ_k = argmax f_O(y_k)                  # Decode output
        q̂_k = f_Q(y_k)                         # Q-head score
    Return ŷ_{k*}, where k* = argmax_k q̂_k     # Best-Q selection

The noise is injected into z_H (the H-level latent state) at each
supervision step (= one call to inner.forward), NOT at each L-cycle.

Usage:
    from inference.ptrm_inference import PTRMInference

    engine = PTRMInference(model, device="cuda")
    results = engine.run(batch, K=100, D=64, sigma=0.3)
"""

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class PTRMRolloutResult:
    """Result of a single PTRM inference call on one batch element."""
    # Shape: (K, seq_len) — predicted token IDs per rollout
    predictions: torch.Tensor
    # Shape: (K,) — Q-halt logit per rollout (higher = model more confident)
    q_values: torch.Tensor
    # Shape: (K, D, latent_dim) — z_H[:, 0, :] trajectory per rollout per step (for viz)
    # Only populated if collect_trajectories=True
    latent_trajectories: Optional[torch.Tensor] = None
    # Shape: (K, D, seq_len) — per-step predicted token IDs (for viz)
    step_predictions: Optional[torch.Tensor] = None
    # Shape: (K, D) — per-step Q values (for viz)
    step_q_values: Optional[torch.Tensor] = None


@dataclass
class PTRMBatchResult:
    """Result of PTRM inference on a full batch."""
    # Shape: (B, K, seq_len) — predictions for each batch element
    all_predictions: torch.Tensor
    # Shape: (B, K) — Q values for each batch element × rollout
    all_q_values: torch.Tensor
    # Selected predictions per selection method
    # Shape: (B, seq_len) — best-Q selected predictions
    best_q_predictions: torch.Tensor
    # Shape: (B, seq_len) — mode (majority vote) selected predictions
    mode_predictions: torch.Tensor
    # Per-rollout detail for visualization (only if collect_trajectories=True)
    latent_trajectories: Optional[torch.Tensor] = None  # (B, K, D, latent_dim)
    step_predictions: Optional[torch.Tensor] = None     # (B, K, D, seq_len)
    step_q_values: Optional[torch.Tensor] = None        # (B, K, D)


class PTRMInference:
    """
    PTRM inference engine.

    Wraps a TRM model (TinyRecursiveReasoningModel_ACTV1) and provides
    stochastic rollout inference with Q-head selection.
    """

    def __init__(self, model: nn.Module, device: str = "cpu"):
        """
        Args:
            model: A TinyRecursiveReasoningModel_ACTV1 instance in eval mode.
            device: Target device for inference.
        """
        self.model = model
        self.device = device
        self.inner = model.inner  # Direct access to the inner model

        # Cache architecture properties
        self.hidden_size = self.inner.config.hidden_size
        self.seq_len = self.inner.config.seq_len
        self.puzzle_emb_len = self.inner.puzzle_emb_len
        self.forward_dtype = self.inner.forward_dtype

    def _expand_batch_for_rollouts(
        self, batch: dict[str, torch.Tensor], K: int
    ) -> dict[str, torch.Tensor]:
        """
        Expand a batch of size B to B*K by repeating each element K times.
        After expansion, elements [0:K] correspond to K rollouts of the
        first batch element, [K:2K] to the second, etc.
        """
        expanded = {}
        for key, tensor in batch.items():
            # (B, ...) -> (B, K, ...) -> (B*K, ...)
            expanded[key] = tensor.unsqueeze(1).expand(
                tensor.shape[0], K, *tensor.shape[1:]
            ).reshape(tensor.shape[0] * K, *tensor.shape[1:])
        return expanded

    def _run_supervision_step(
        self,
        carry_z_H: torch.Tensor,
        carry_z_L: torch.Tensor,
        batch: dict[str, torch.Tensor],
        sigma: float,
        generator: Optional[torch.Generator] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Execute one supervision step with noise injection.

        This is the core PTRM operation:
            1. Add Gaussian noise to z_H
            2. Run inner model forward (H_cycles × L_cycles of recursive reasoning)
            3. Collect outputs

        Args:
            carry_z_H: (B*K, seq_len+puzzle_emb_len, hidden_size) H-level state
            carry_z_L: (B*K, seq_len+puzzle_emb_len, hidden_size) L-level state
            batch: Expanded batch dict
            sigma: Noise standard deviation
            generator: Optional torch Generator for reproducibility

        Returns:
            new_z_H: Updated H-level state
            new_z_L: Updated L-level state
            logits: (B*K, seq_len, vocab_size) output logits
            q_halt: (B*K,) Q-halt logit
            q_continue: (B*K,) Q-continue logit
        """
        # 1. Inject noise into z_H
        if sigma > 0:
            noise = torch.randn(
                carry_z_H.shape,
                dtype=carry_z_H.dtype,
                device=carry_z_H.device,
                generator=generator,
            ) * sigma
            carry_z_H = carry_z_H + noise

        # 2. Build inner carry and run forward pass
        # We directly use inner.forward() to bypass the ACT halting logic,
        # since PTRM controls the number of steps externally via D.
        from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1InnerCarry
        inner_carry = TinyRecursiveReasoningModel_ACTV1InnerCarry(
            z_H=carry_z_H, z_L=carry_z_L
        )

        with torch.no_grad():
            new_carry, logits, (q_halt, q_continue) = self.inner(inner_carry, batch)

        return new_carry.z_H, new_carry.z_L, logits, q_halt, q_continue

    def _select_best_q(
        self, predictions: torch.Tensor, q_values: torch.Tensor, B: int, K: int
    ) -> torch.Tensor:
        """
        Select the prediction with the highest Q-value per batch element.

        Args:
            predictions: (B*K, seq_len) predicted tokens
            q_values: (B*K,) Q-halt logits
            B: Original batch size
            K: Number of rollouts

        Returns:
            (B, seq_len) best-Q selected predictions
        """
        # Reshape to (B, K, ...)
        preds_bk = predictions.view(B, K, -1)
        q_bk = q_values.view(B, K)

        # Select argmax Q per batch element
        best_k = q_bk.argmax(dim=1)  # (B,)
        best_preds = preds_bk[torch.arange(B, device=predictions.device), best_k]
        return best_preds

    def _select_mode(
        self, predictions: torch.Tensor, B: int, K: int
    ) -> torch.Tensor:
        """
        Select the most frequent prediction (majority vote) per batch element.

        Compares full prediction vectors for equality. If there's a tie,
        the first (lowest index) mode is selected.

        Args:
            predictions: (B*K, seq_len) predicted tokens
            B: Original batch size
            K: Number of rollouts

        Returns:
            (B, seq_len) mode-selected predictions
        """
        preds_bk = predictions.view(B, K, -1)
        result = torch.empty(B, preds_bk.shape[2], dtype=predictions.dtype, device=predictions.device)

        for b in range(B):
            # Count occurrences of each unique prediction
            unique_preds, inverse_indices = torch.unique(
                preds_bk[b], dim=0, return_inverse=True
            )
            # Count each unique prediction's frequency
            counts = torch.bincount(inverse_indices, minlength=unique_preds.shape[0])
            # Select the most frequent
            most_common_idx = counts.argmax()
            result[b] = unique_preds[most_common_idx]

        return result

    @torch.no_grad()
    def run(
        self,
        batch: dict[str, torch.Tensor],
        K: int = 25,
        D: int = 16,
        sigma: float = 0.3,
        seed: Optional[int] = None,
        collect_trajectories: bool = False,
        k_chunk_size: Optional[int] = None,
    ) -> PTRMBatchResult:
        """
        Run PTRM inference on a batch.

        Args:
            batch: Dict with keys 'inputs', 'puzzle_identifiers', optionally 'labels'.
                   Shape: inputs (B, seq_len), puzzle_identifiers (B,).
            K: Number of parallel rollouts per puzzle.
            D: Number of supervision (deep recursion) steps.
            sigma: Noise standard deviation for z_H perturbation.
            seed: Random seed for reproducibility.
            collect_trajectories: If True, collect per-step latent states and
                                  predictions for visualization.
            k_chunk_size: If set, process rollouts in chunks of this size to
                          reduce peak GPU memory. None = all K at once.

        Returns:
            PTRMBatchResult with predictions, Q-values, and selection results.
        """
        B = batch["inputs"].shape[0]

        # Ensure batch is on the correct device
        batch = {k: v.to(self.device) for k, v in batch.items()}

        # Setup random generator for reproducibility
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(seed)

        if k_chunk_size is None or k_chunk_size >= K:
            return self._run_all_rollouts(batch, B, K, D, sigma, generator, collect_trajectories)
        else:
            return self._run_chunked_rollouts(batch, B, K, D, sigma, generator, collect_trajectories, k_chunk_size)

    def _run_all_rollouts(
        self,
        batch: dict[str, torch.Tensor],
        B: int,
        K: int,
        D: int,
        sigma: float,
        generator: Optional[torch.Generator],
        collect_trajectories: bool,
    ) -> PTRMBatchResult:
        """Run all K rollouts simultaneously (memory intensive but fast)."""

        # Expand batch: (B, ...) -> (B*K, ...)
        expanded_batch = self._expand_batch_for_rollouts(batch, K)

        # Initialize carries from the model's init state
        # z_H and z_L are initialized to the model's learned init vectors
        total = B * K
        latent_len = self.seq_len + self.puzzle_emb_len
        z_H = self.inner.H_init.unsqueeze(0).expand(total, latent_len, -1).clone()
        z_L = self.inner.L_init.unsqueeze(0).expand(total, latent_len, -1).clone()

        # Trajectory collection storage
        if collect_trajectories:
            all_latents = torch.empty(total, D, self.hidden_size, device=self.device)
            all_step_preds = torch.empty(total, D, self.seq_len, dtype=torch.long, device=self.device)
            all_step_qs = torch.empty(total, D, device=self.device)

        # Run D supervision steps
        for t in range(D):
            z_H, z_L, logits, q_halt, q_continue = self._run_supervision_step(
                z_H, z_L, expanded_batch, sigma, generator
            )

            if collect_trajectories:
                # Store latent from position 0 (puzzle_emb position, where Q-head reads)
                all_latents[:, t] = z_H[:, 0].float()
                all_step_preds[:, t] = logits.argmax(dim=-1)
                all_step_qs[:, t] = q_halt.float()

        # Final predictions and Q-values from the last step
        final_predictions = logits.argmax(dim=-1)  # (B*K, seq_len)
        final_q_values = q_halt.float()             # (B*K,)

        # Selection
        best_q_preds = self._select_best_q(final_predictions, final_q_values, B, K)
        mode_preds = self._select_mode(final_predictions, B, K)

        result = PTRMBatchResult(
            all_predictions=final_predictions.view(B, K, -1),
            all_q_values=final_q_values.view(B, K),
            best_q_predictions=best_q_preds,
            mode_predictions=mode_preds,
        )

        if collect_trajectories:
            result.latent_trajectories = all_latents.view(B, K, D, -1)
            result.step_predictions = all_step_preds.view(B, K, D, -1)
            result.step_q_values = all_step_qs.view(B, K, D)

        return result

    def _run_chunked_rollouts(
        self,
        batch: dict[str, torch.Tensor],
        B: int,
        K: int,
        D: int,
        sigma: float,
        generator: Optional[torch.Generator],
        collect_trajectories: bool,
        k_chunk_size: int,
    ) -> PTRMBatchResult:
        """Run rollouts in chunks of k_chunk_size to reduce peak memory."""

        all_preds_list = []
        all_q_list = []
        all_latents_list = [] if collect_trajectories else None
        all_step_preds_list = [] if collect_trajectories else None
        all_step_qs_list = [] if collect_trajectories else None

        remaining = K
        while remaining > 0:
            chunk_k = min(k_chunk_size, remaining)
            chunk_result = self._run_all_rollouts(
                batch, B, chunk_k, D, sigma, generator, collect_trajectories
            )

            all_preds_list.append(chunk_result.all_predictions)
            all_q_list.append(chunk_result.all_q_values)

            if collect_trajectories:
                all_latents_list.append(chunk_result.latent_trajectories)
                all_step_preds_list.append(chunk_result.step_predictions)
                all_step_qs_list.append(chunk_result.step_q_values)

            remaining -= chunk_k

        # Concatenate chunks along K dimension
        all_predictions = torch.cat(all_preds_list, dim=1)  # (B, K, seq_len)
        all_q_values = torch.cat(all_q_list, dim=1)          # (B, K)

        # Flatten for selection
        flat_preds = all_predictions.view(B * K, -1)
        flat_q = all_q_values.view(B * K)

        best_q_preds = self._select_best_q(flat_preds, flat_q, B, K)
        mode_preds = self._select_mode(flat_preds, B, K)

        result = PTRMBatchResult(
            all_predictions=all_predictions,
            all_q_values=all_q_values,
            best_q_predictions=best_q_preds,
            mode_predictions=mode_preds,
        )

        if collect_trajectories:
            result.latent_trajectories = torch.cat(all_latents_list, dim=1)
            result.step_predictions = torch.cat(all_step_preds_list, dim=1)
            result.step_q_values = torch.cat(all_step_qs_list, dim=1)

        return result
