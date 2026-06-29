"""
Tests for Component 2: PTRM Core Inference Engine.

Test categories:
  1. PTRMInference helper methods (batch expansion, selection methods)
  2. Supervision step mechanics (noise injection, carry propagation)
  3. Full PTRM run (deterministic σ=0, stochastic σ>0, trajectory collection)
  4. Chunked rollout equivalence
  5. Edge cases (K=1, D=1, σ=0)
"""

import os
import sys

import pytest
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from inference.ptrm_inference import PTRMInference, PTRMBatchResult, PTRMRolloutResult
from inference.checkpoint_loader import _build_model_config, _add_trm_to_path
from scripts.download_models import MODEL_REGISTRY

# Skip all tests if TRM submodule is not available
TRM_AVAILABLE = os.path.isdir(os.path.join(PROJECT_ROOT, "TinyRecursiveModels", "models"))
pytestmark = pytest.mark.skipif(not TRM_AVAILABLE, reason="TinyRecursiveModels submodule not available")


@pytest.fixture(scope="module")
def sudoku_model():
    """Create a Sudoku TRM-MLP model (untrained) for testing."""
    _add_trm_to_path()
    from utils.functions import load_model_class

    entry = MODEL_REGISTRY["sudoku"]
    model_cfg = _build_model_config(entry["arch_config"], entry["dataset_meta"], batch_size=4)
    model_cls = load_model_class(entry["arch_config"]["name"])

    with torch.device("cpu"):
        model = model_cls(model_cfg)
    model.eval()
    return model


@pytest.fixture(scope="module")
def sudoku_meta():
    return MODEL_REGISTRY["sudoku"]["dataset_meta"]


@pytest.fixture
def dummy_batch(sudoku_meta):
    """Create a dummy Sudoku batch (B=2)."""
    B = 2
    meta = sudoku_meta
    return {
        "inputs": torch.randint(0, meta["vocab_size"], (B, meta["seq_len"])),
        "labels": torch.randint(0, meta["vocab_size"], (B, meta["seq_len"])),
        "puzzle_identifiers": torch.zeros(B, dtype=torch.long),
    }


@pytest.fixture
def engine(sudoku_model):
    """Create a PTRMInference engine."""
    return PTRMInference(sudoku_model, device="cpu")


# =============================================================================
# 1. Batch expansion tests
# =============================================================================

class TestBatchExpansion:

    def test_expansion_shapes(self, engine, dummy_batch):
        K = 5
        expanded = engine._expand_batch_for_rollouts(dummy_batch, K)
        B = dummy_batch["inputs"].shape[0]

        assert expanded["inputs"].shape == (B * K, dummy_batch["inputs"].shape[1])
        assert expanded["puzzle_identifiers"].shape == (B * K,)

    def test_expansion_repeats_correctly(self, engine, dummy_batch):
        K = 3
        expanded = engine._expand_batch_for_rollouts(dummy_batch, K)

        # First K elements should all be copies of batch[0]
        for k in range(K):
            assert torch.equal(expanded["inputs"][k], dummy_batch["inputs"][0])

        # Next K elements should all be copies of batch[1]
        for k in range(K):
            assert torch.equal(expanded["inputs"][K + k], dummy_batch["inputs"][1])

    def test_expansion_k1_is_identity(self, engine, dummy_batch):
        expanded = engine._expand_batch_for_rollouts(dummy_batch, K=1)
        assert torch.equal(expanded["inputs"], dummy_batch["inputs"])


# =============================================================================
# 2. Selection method tests
# =============================================================================

class TestBestQSelection:

    def test_selects_highest_q(self, engine):
        B, K, seq_len = 2, 4, 10
        predictions = torch.arange(B * K * seq_len).view(B * K, seq_len)
        q_values = torch.tensor([0.1, 0.9, 0.3, 0.5, 0.2, 0.4, 0.8, 0.6])  # B=2, K=4

        result = engine._select_best_q(predictions, q_values, B, K)

        # For B=0: best K=1 (q=0.9), for B=1: best K=2 (q=0.8)
        assert result.shape == (B, seq_len)
        assert torch.equal(result[0], predictions[1])   # K=1 for batch 0
        assert torch.equal(result[1], predictions[6])    # K=2 for batch 1

    def test_selects_single_rollout(self, engine):
        B, K, seq_len = 1, 1, 5
        predictions = torch.tensor([[1, 2, 3, 4, 5]])
        q_values = torch.tensor([0.5])

        result = engine._select_best_q(predictions, q_values, B, K)
        assert torch.equal(result, predictions)


class TestModeSelection:

    def test_selects_majority(self, engine):
        B, K, seq_len = 1, 5, 3
        # 3 rollouts predict [1,2,3], 2 predict [4,5,6]
        predictions = torch.tensor([
            [1, 2, 3],
            [1, 2, 3],
            [4, 5, 6],
            [1, 2, 3],
            [4, 5, 6],
        ])  # shape: (5, 3) = (B*K, seq_len)

        result = engine._select_mode(predictions, B, K)
        assert result.shape == (1, 3)
        assert torch.equal(result[0], torch.tensor([1, 2, 3]))

    def test_single_rollout(self, engine):
        B, K, seq_len = 1, 1, 5
        predictions = torch.tensor([[1, 2, 3, 4, 5]])
        result = engine._select_mode(predictions, B, K)
        assert torch.equal(result[0], predictions[0])

    def test_all_unique(self, engine):
        B, K, seq_len = 1, 3, 2
        predictions = torch.tensor([[1, 2], [3, 4], [5, 6]])
        result = engine._select_mode(predictions, B, K)
        # With all unique, first one should be selected (argmax of all-ones counts)
        assert result.shape == (1, 2)

    def test_multi_batch(self, engine):
        B, K, seq_len = 2, 3, 2
        predictions = torch.tensor([
            [1, 1], [2, 2], [1, 1],  # B=0: mode is [1,1]
            [3, 3], [3, 3], [4, 4],  # B=1: mode is [3,3]
        ])
        result = engine._select_mode(predictions, B, K)
        assert torch.equal(result[0], torch.tensor([1, 1]))
        assert torch.equal(result[1], torch.tensor([3, 3]))


# =============================================================================
# 3. Supervision step tests
# =============================================================================

class TestSupervisionStep:

    def test_output_shapes(self, engine, dummy_batch):
        B = dummy_batch["inputs"].shape[0]
        K = 3
        expanded = engine._expand_batch_for_rollouts(dummy_batch, K)

        total = B * K
        latent_len = engine.seq_len + engine.puzzle_emb_len
        z_H = engine.inner.H_init.unsqueeze(0).expand(total, latent_len, -1).clone()
        z_L = engine.inner.L_init.unsqueeze(0).expand(total, latent_len, -1).clone()

        new_z_H, new_z_L, logits, q_halt, q_continue = engine._run_supervision_step(
            z_H, z_L, expanded, sigma=0.3
        )

        assert new_z_H.shape == z_H.shape
        assert new_z_L.shape == z_L.shape
        assert logits.shape == (total, engine.seq_len, engine.inner.config.vocab_size)
        assert q_halt.shape == (total,)
        assert q_continue.shape == (total,)

    def test_zero_sigma_is_deterministic(self, engine, dummy_batch):
        B = dummy_batch["inputs"].shape[0]
        K = 2
        expanded = engine._expand_batch_for_rollouts(dummy_batch, K)

        total = B * K
        latent_len = engine.seq_len + engine.puzzle_emb_len
        z_H = engine.inner.H_init.unsqueeze(0).expand(total, latent_len, -1).clone()
        z_L = engine.inner.L_init.unsqueeze(0).expand(total, latent_len, -1).clone()

        # Run twice with sigma=0
        _, _, logits1, q1, _ = engine._run_supervision_step(z_H.clone(), z_L.clone(), expanded, sigma=0.0)
        _, _, logits2, q2, _ = engine._run_supervision_step(z_H.clone(), z_L.clone(), expanded, sigma=0.0)

        assert torch.equal(logits1, logits2)
        assert torch.equal(q1, q2)

    def test_nonzero_sigma_adds_variation(self, engine, dummy_batch):
        B = dummy_batch["inputs"].shape[0]
        K = 2
        expanded = engine._expand_batch_for_rollouts(dummy_batch, K)

        total = B * K
        latent_len = engine.seq_len + engine.puzzle_emb_len
        z_H = engine.inner.H_init.unsqueeze(0).expand(total, latent_len, -1).clone()
        z_L = engine.inner.L_init.unsqueeze(0).expand(total, latent_len, -1).clone()

        # Run twice with sigma=1.0 (high noise)
        _, _, logits1, q1, _ = engine._run_supervision_step(z_H.clone(), z_L.clone(), expanded, sigma=1.0)
        _, _, logits2, q2, _ = engine._run_supervision_step(z_H.clone(), z_L.clone(), expanded, sigma=1.0)

        # With high noise, outputs should differ (very high probability)
        assert not torch.equal(logits1, logits2)

    def test_seeded_noise_is_reproducible(self, engine, dummy_batch):
        B = dummy_batch["inputs"].shape[0]
        K = 2
        expanded = engine._expand_batch_for_rollouts(dummy_batch, K)

        total = B * K
        latent_len = engine.seq_len + engine.puzzle_emb_len
        z_H = engine.inner.H_init.unsqueeze(0).expand(total, latent_len, -1).clone()
        z_L = engine.inner.L_init.unsqueeze(0).expand(total, latent_len, -1).clone()

        gen1 = torch.Generator(device="cpu")
        gen1.manual_seed(42)
        gen2 = torch.Generator(device="cpu")
        gen2.manual_seed(42)

        _, _, logits1, q1, _ = engine._run_supervision_step(
            z_H.clone(), z_L.clone(), expanded, sigma=0.5, generator=gen1
        )
        _, _, logits2, q2, _ = engine._run_supervision_step(
            z_H.clone(), z_L.clone(), expanded, sigma=0.5, generator=gen2
        )

        assert torch.equal(logits1, logits2)
        assert torch.equal(q1, q2)


# =============================================================================
# 4. Full PTRM run tests
# =============================================================================

class TestPTRMRun:

    def test_basic_run_shapes(self, engine, dummy_batch):
        B = dummy_batch["inputs"].shape[0]
        K, D = 5, 4

        result = engine.run(dummy_batch, K=K, D=D, sigma=0.3, seed=42)

        assert isinstance(result, PTRMBatchResult)
        assert result.all_predictions.shape == (B, K, engine.seq_len)
        assert result.all_q_values.shape == (B, K)
        assert result.best_q_predictions.shape == (B, engine.seq_len)
        assert result.mode_predictions.shape == (B, engine.seq_len)

    def test_trajectory_collection(self, engine, dummy_batch):
        B = dummy_batch["inputs"].shape[0]
        K, D = 3, 4

        result = engine.run(dummy_batch, K=K, D=D, sigma=0.3, seed=42, collect_trajectories=True)

        assert result.latent_trajectories is not None
        assert result.latent_trajectories.shape == (B, K, D, engine.hidden_size)
        assert result.step_predictions is not None
        assert result.step_predictions.shape == (B, K, D, engine.seq_len)
        assert result.step_q_values is not None
        assert result.step_q_values.shape == (B, K, D)

    def test_no_trajectory_by_default(self, engine, dummy_batch):
        result = engine.run(dummy_batch, K=3, D=2, sigma=0.3)
        assert result.latent_trajectories is None
        assert result.step_predictions is None
        assert result.step_q_values is None

    def test_sigma_zero_all_rollouts_identical(self, engine, dummy_batch):
        """With σ=0, all K rollouts should produce identical results."""
        B = dummy_batch["inputs"].shape[0]
        K, D = 5, 3

        result = engine.run(dummy_batch, K=K, D=D, sigma=0.0, seed=42)

        # All K rollouts should be the same for each batch element
        for b in range(B):
            for k in range(1, K):
                assert torch.equal(result.all_predictions[b, 0], result.all_predictions[b, k]), \
                    f"Batch {b}: rollout 0 != rollout {k} with sigma=0"

    def test_sigma_zero_q_values_identical(self, engine, dummy_batch):
        """With σ=0, all Q-values should be identical within each batch element."""
        B = dummy_batch["inputs"].shape[0]
        K = 5

        result = engine.run(dummy_batch, K=K, D=3, sigma=0.0)

        for b in range(B):
            for k in range(1, K):
                assert torch.isclose(result.all_q_values[b, 0], result.all_q_values[b, k]), \
                    f"Batch {b}: Q[0]={result.all_q_values[b, 0]} != Q[{k}]={result.all_q_values[b, k]}"

    def test_seeded_run_is_reproducible(self, engine, dummy_batch):
        K, D, sigma = 5, 4, 0.3

        r1 = engine.run(dummy_batch, K=K, D=D, sigma=sigma, seed=123)
        r2 = engine.run(dummy_batch, K=K, D=D, sigma=sigma, seed=123)

        assert torch.equal(r1.all_predictions, r2.all_predictions)
        assert torch.equal(r1.all_q_values, r2.all_q_values)
        assert torch.equal(r1.best_q_predictions, r2.best_q_predictions)
        assert torch.equal(r1.mode_predictions, r2.mode_predictions)

    def test_different_seeds_give_different_results(self, engine, dummy_batch):
        K, D, sigma = 10, 4, 0.5

        r1 = engine.run(dummy_batch, K=K, D=D, sigma=sigma, seed=42)
        r2 = engine.run(dummy_batch, K=K, D=D, sigma=sigma, seed=99)

        # Different seeds should produce different predictions.
        # (Q-values may be identical in untrained models since the Q-head
        # is zero-weight initialized with bias=-5, making all outputs -5.0.)
        assert not torch.equal(r1.all_predictions, r2.all_predictions)


# =============================================================================
# 5. Edge case tests
# =============================================================================

class TestEdgeCases:

    def test_k_equals_1(self, engine, dummy_batch):
        result = engine.run(dummy_batch, K=1, D=4, sigma=0.3, seed=42)
        B = dummy_batch["inputs"].shape[0]

        assert result.all_predictions.shape == (B, 1, engine.seq_len)
        # best_q and mode should equal the single rollout
        assert torch.equal(result.best_q_predictions, result.all_predictions[:, 0])
        assert torch.equal(result.mode_predictions, result.all_predictions[:, 0])

    def test_d_equals_1(self, engine, dummy_batch):
        result = engine.run(dummy_batch, K=5, D=1, sigma=0.3, seed=42)
        B = dummy_batch["inputs"].shape[0]

        assert result.all_predictions.shape == (B, 5, engine.seq_len)

    def test_batch_size_1(self, engine, sudoku_meta):
        """Test with a single puzzle."""
        meta = sudoku_meta
        batch = {
            "inputs": torch.randint(0, meta["vocab_size"], (1, meta["seq_len"])),
            "puzzle_identifiers": torch.zeros(1, dtype=torch.long),
        }
        result = engine.run(batch, K=5, D=3, sigma=0.3, seed=42)

        assert result.all_predictions.shape == (1, 5, meta["seq_len"])
        assert result.best_q_predictions.shape == (1, meta["seq_len"])

    def test_large_sigma(self, engine, dummy_batch):
        """Large σ should still produce valid output shapes."""
        result = engine.run(dummy_batch, K=3, D=2, sigma=10.0, seed=42)
        assert result.all_predictions.shape[0] == dummy_batch["inputs"].shape[0]


# =============================================================================
# 6. Chunked rollout tests
# =============================================================================

class TestChunkedRollouts:

    def test_chunked_shapes_match(self, engine, dummy_batch):
        B = dummy_batch["inputs"].shape[0]
        K, D = 6, 3

        result = engine.run(dummy_batch, K=K, D=D, sigma=0.3, seed=42, k_chunk_size=2)

        assert result.all_predictions.shape == (B, K, engine.seq_len)
        assert result.all_q_values.shape == (B, K)

    def test_chunked_trajectory_shapes(self, engine, dummy_batch):
        B = dummy_batch["inputs"].shape[0]
        K, D = 6, 3

        result = engine.run(
            dummy_batch, K=K, D=D, sigma=0.3, seed=42,
            k_chunk_size=2, collect_trajectories=True
        )

        assert result.latent_trajectories.shape == (B, K, D, engine.hidden_size)
        assert result.step_predictions.shape == (B, K, D, engine.seq_len)
        assert result.step_q_values.shape == (B, K, D)

    def test_chunk_size_equals_k(self, engine, dummy_batch):
        """chunk_size == K should behave identically to no chunking."""
        K, D = 4, 3
        # Note: can't compare values since generator state differs between chunked paths.
        # But shapes and types should match.
        result = engine.run(dummy_batch, K=K, D=D, sigma=0.3, seed=42, k_chunk_size=K)

        assert result.all_predictions.shape[1] == K

    def test_chunk_size_1(self, engine, dummy_batch):
        """Extreme: process one rollout at a time."""
        B = dummy_batch["inputs"].shape[0]
        K, D = 4, 2

        result = engine.run(dummy_batch, K=K, D=D, sigma=0.3, seed=42, k_chunk_size=1)

        assert result.all_predictions.shape == (B, K, engine.seq_len)


# =============================================================================
# 7. Best-Q selection correctness
# =============================================================================

class TestBestQCorrectness:

    def test_best_q_in_predictions(self, engine, dummy_batch):
        """The best-Q prediction should be one of the K rollout predictions."""
        B = dummy_batch["inputs"].shape[0]
        K = 5

        result = engine.run(dummy_batch, K=K, D=3, sigma=0.3, seed=42)

        for b in range(B):
            best_k = result.all_q_values[b].argmax()
            assert torch.equal(
                result.best_q_predictions[b],
                result.all_predictions[b, best_k]
            ), f"Batch {b}: best-Q prediction doesn't match argmax rollout"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
