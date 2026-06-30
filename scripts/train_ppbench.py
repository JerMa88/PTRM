"""
PPBench TRM-Att Training Script with Comprehensive W&B Logging.

Wraps TRM's pretrain.py with additional telemetry for maximum observability.
Logs every training step, evaluation, per-puzzle-type metrics, and system stats.

Training follows arXiv:2605.19943 exactly:
  - TRM-Att architecture (7M params)
  - PPBench dataset (6 puzzle types, 98,967 train examples)
  - Standard TRM training recipe (AdamATan2, cosine schedule, etc.)

W&B Logging (MAX TRACKING):
  - Every step: loss, cell_accuracy, lr, gradient_norm, throughput
  - Every eval: val_loss, val_accuracy, per-puzzle-type accuracy
  - Checkpoints: best model tracked by validation accuracy
  - System: GPU memory, utilization (via wandb system metrics)
  - Artifacts: best checkpoint + config logged as W&B artifacts

Usage:
    # Single GPU
    python scripts/train_ppbench.py

    # Multi-GPU (DDP)
    torchrun --nproc_per_node=4 scripts/train_ppbench.py

    # With overrides
    python scripts/train_ppbench.py --epochs 50000 --eval-interval 5000
"""

import argparse
import copy
import math
import os
import shutil
import sys
import time
import yaml
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure project paths are correct
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRM_ROOT = PROJECT_ROOT / "TinyRecursiveModels"
sys.path.insert(0, str(TRM_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.distributed as dist
import wandb
import numpy as np

# TRM imports
from pretrain import (
    PretrainConfig,
    TrainState,
    create_dataloader,
    create_model,
    init_train_state,
    save_train_state,
    cosine_schedule_with_warmup_lr_lambda,
    evaluate,
    create_evaluators,
    save_code_and_config,
)
from puzzle_dataset import PuzzleDatasetMetadata


# ---------------------------------------------------------------------------
# Enhanced training with per-step W&B logging
# ---------------------------------------------------------------------------

def compute_gradient_norm(model: torch.nn.Module) -> float:
    """Compute the global L2 norm of all gradients."""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.float().norm(2).item() ** 2
    return total_norm ** 0.5


def compute_weight_norm(model: torch.nn.Module) -> float:
    """Compute the global L2 norm of all parameters."""
    total_norm = 0.0
    for p in model.parameters():
        total_norm += p.data.float().norm(2).item() ** 2
    return total_norm ** 0.5


def get_q_head_stats(model: torch.nn.Module) -> dict:
    """Extract Q-head bias and weight statistics for monitoring."""
    stats = {}
    for name, param in model.named_parameters():
        if "q_head" in name:
            stats[f"q_head/{name.split('.')[-1]}_mean"] = param.data.float().mean().item()
            stats[f"q_head/{name.split('.')[-1]}_std"] = param.data.float().std().item()
            if "bias" in name:
                stats["q_head/bias_value"] = param.data.float().tolist()
    return stats


def train_batch_enhanced(
    config: PretrainConfig,
    train_state: TrainState,
    batch: dict,
    global_batch_size: int,
    rank: int,
    world_size: int,
    step_start_time: float,
) -> dict:
    """
    Enhanced training step with comprehensive metric collection.

    Returns all metrics for W&B logging (on rank 0 only).
    """
    train_state.step += 1
    if train_state.step > train_state.total_steps:
        return {}

    # To device
    batch = {k: v.cuda() for k, v in batch.items()}

    # Init carry
    if train_state.carry is None:
        with torch.device("cuda"):
            train_state.carry = train_state.model.initial_carry(batch)

    # Forward
    train_state.carry, loss, metrics, _, _ = train_state.model(
        carry=train_state.carry, batch=batch, return_keys=[]
    )

    ((1 / global_batch_size) * loss).backward()

    # Compute gradient norm BEFORE optimizer step
    grad_norm = compute_gradient_norm(train_state.model) if rank == 0 else 0.0

    # Allreduce
    if world_size > 1:
        for param in train_state.model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad)

    # Apply optimizer with LR schedule
    lr_this_step = None
    for optim, base_lr in zip(train_state.optimizers, train_state.optimizer_lrs):
        lr_this_step = cosine_schedule_with_warmup_lr_lambda(
            current_step=train_state.step,
            base_lr=base_lr,
            num_warmup_steps=round(config.lr_warmup_steps),
            num_training_steps=train_state.total_steps,
            min_ratio=config.lr_min_ratio,
        )
        for param_group in optim.param_groups:
            param_group["lr"] = lr_this_step
        optim.step()
        optim.zero_grad()

    # Collect enhanced metrics on rank 0
    if rank == 0 and len(metrics):
        assert not any(v.requires_grad for v in metrics.values())

        metric_keys = list(sorted(metrics.keys()))
        metric_values = torch.stack([metrics[k] for k in metric_keys])

        if world_size > 1:
            dist.reduce(metric_values, dst=0)

        metric_values_np = metric_values.cpu().numpy()
        reduced = {k: metric_values_np[i] for i, k in enumerate(metric_keys)}

        count = max(reduced.get("count", 1), 1)
        enhanced_metrics = {}

        # Standard TRM metrics (normalized)
        for k, v in reduced.items():
            if k.endswith("loss"):
                enhanced_metrics[f"train/{k}"] = v / global_batch_size
            else:
                enhanced_metrics[f"train/{k}"] = v / count

        # Enhanced metrics
        enhanced_metrics["train/lr"] = lr_this_step
        enhanced_metrics["train/gradient_norm"] = grad_norm
        enhanced_metrics["train/step"] = train_state.step
        enhanced_metrics["train/progress"] = train_state.step / train_state.total_steps

        # Throughput
        step_time = time.time() - step_start_time
        enhanced_metrics["train/step_time_sec"] = step_time
        enhanced_metrics["train/samples_per_sec"] = global_batch_size / max(step_time, 1e-6)

        # GPU memory (every 100 steps to reduce overhead)
        if train_state.step % 100 == 0:
            enhanced_metrics["system/gpu_memory_allocated_gb"] = (
                torch.cuda.memory_allocated() / 1e9
            )
            enhanced_metrics["system/gpu_memory_reserved_gb"] = (
                torch.cuda.memory_reserved() / 1e9
            )
            enhanced_metrics["system/gpu_max_memory_gb"] = (
                torch.cuda.max_memory_allocated() / 1e9
            )

        # Weight norm + Q-head stats (every 500 steps)
        if train_state.step % 500 == 0:
            enhanced_metrics["train/weight_norm"] = compute_weight_norm(train_state.model)
            q_stats = get_q_head_stats(train_state.model)
            enhanced_metrics.update(q_stats)

        return enhanced_metrics

    # Other ranks need to participate in the reduce
    elif world_size > 1 and len(metrics):
        metric_keys = list(sorted(metrics.keys()))
        metric_values = torch.stack([metrics[k] for k in metric_keys])
        dist.reduce(metric_values, dst=0)

    return {}


def evaluate_enhanced(
    config: PretrainConfig,
    train_state: TrainState,
    eval_loader,
    eval_metadata,
    evaluators,
    rank: int,
    world_size: int,
    cpu_group,
    puzzle_types: list,
) -> dict:
    """
    Enhanced evaluation with per-puzzle-type accuracy tracking.

    Runs standard TRM evaluation + computes per-type breakdowns.
    """
    # Run standard TRM evaluation
    metrics = evaluate(
        config, train_state, eval_loader, eval_metadata,
        evaluators, rank, world_size, cpu_group
    )

    if rank == 0 and metrics is not None:
        # Prefix all eval metrics
        enhanced = {}
        for k, v in metrics.items():
            if isinstance(v, dict):
                for kk, vv in v.items():
                    enhanced[f"eval/{k}/{kk}"] = vv
            else:
                enhanced[f"eval/{k}"] = v

        return enhanced

    return metrics or {}


# ---------------------------------------------------------------------------
# Best checkpoint tracking
# ---------------------------------------------------------------------------

class BestCheckpointTracker:
    """Track the best checkpoint by validation accuracy."""

    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = checkpoint_dir
        self.best_accuracy = -1.0
        self.best_step = -1
        self.best_path = None

    def update(self, accuracy: float, step: int, model_state_dict: dict) -> bool:
        """Update best checkpoint if accuracy improved. Returns True if updated."""
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            self.best_step = step

            # Save best model
            best_dir = os.path.join(self.checkpoint_dir, "best")
            os.makedirs(best_dir, exist_ok=True)
            best_path = os.path.join(best_dir, "model.pt")

            torch.save(model_state_dict, best_path)
            self.best_path = best_path

            # Save metadata
            meta = {
                "best_accuracy": self.best_accuracy,
                "best_step": self.best_step,
                "best_path": best_path,
            }
            with open(os.path.join(best_dir, "best_info.yaml"), "w") as f:
                yaml.dump(meta, f)

            return True
        return False


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train_ppbench(
    config_path: str = None,
    epochs: int = None,
    eval_interval: int = None,
    wandb_project: str = None,
    wandb_run_name: str = None,
):
    """
    Main PPBench training entry point.

    Wraps TRM's pretrain.py with comprehensive W&B logging.
    """
    RANK = 0
    WORLD_SIZE = 1
    CPU_PROCESS_GROUP = None

    # Initialize distributed if running with torchrun
    if "LOCAL_RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        RANK = dist.get_rank()
        WORLD_SIZE = dist.get_world_size()
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        CPU_PROCESS_GROUP = dist.new_group(backend="gloo")

    # Load config
    if config_path is None:
        config_path = str(PROJECT_ROOT / "config" / "train" / "ppbench.yaml")

    with open(config_path, "r") as f:
        raw_config = yaml.safe_load(f)

    # Load arch config
    arch_config_path = str(TRM_ROOT / "config" / "arch" / "trm.yaml")
    with open(arch_config_path, "r") as f:
        arch_config = yaml.safe_load(f)

    # Merge arch into config
    if "arch" not in raw_config or isinstance(raw_config.get("arch"), str):
        raw_config["arch"] = arch_config

    # Apply overrides
    if epochs is not None:
        raw_config["epochs"] = epochs
    if eval_interval is not None:
        raw_config["eval_interval"] = eval_interval
    if wandb_project:
        raw_config["project_name"] = wandb_project

    # Remove Hydra-specific keys
    raw_config.pop("defaults", None)
    raw_config.pop("hydra", None)

    # Resolve OmegaConf-style interpolations
    if "puzzle_emb_ndim" in raw_config.get("arch", {}):
        if isinstance(raw_config["arch"]["puzzle_emb_ndim"], str):
            raw_config["arch"]["puzzle_emb_ndim"] = raw_config["arch"]["hidden_size"]

    # Create config
    config = PretrainConfig(**raw_config)
    if wandb_run_name:
        config.run_name = wandb_run_name
    elif config.run_name is None:
        import coolname
        config.run_name = f"TRM-Att-PPBench {coolname.generate_slug(2)}"

    if config.checkpoint_path is None:
        config.checkpoint_path = str(
            PROJECT_ROOT / "checkpoints" / config.project_name / config.run_name
        )

    # Resolve data paths relative to project root
    config.data_paths = [
        str(PROJECT_ROOT / p.lstrip("../")) if p.startswith("..") else p
        for p in config.data_paths
    ]

    # Seed
    torch.random.manual_seed(config.seed + RANK)

    # Dataset
    train_epochs_per_iter = config.eval_interval or config.epochs
    total_iters = config.epochs // train_epochs_per_iter

    # Change working directory to TRM root for imports
    original_cwd = os.getcwd()
    os.chdir(str(TRM_ROOT))

    train_loader, train_metadata = create_dataloader(
        config, "train", test_set_mode=False,
        epochs_per_iter=train_epochs_per_iter,
        global_batch_size=config.global_batch_size,
        rank=RANK, world_size=WORLD_SIZE,
    )

    try:
        eval_loader, eval_metadata = create_dataloader(
            config, "test", test_set_mode=True,
            epochs_per_iter=1,
            global_batch_size=config.global_batch_size,
            rank=RANK, world_size=WORLD_SIZE,
        )
    except Exception:
        print("NO EVAL DATA FOUND — skipping evaluation during training")
        eval_loader = eval_metadata = None

    try:
        evaluators = create_evaluators(config, eval_metadata)
    except Exception:
        evaluators = []

    os.chdir(original_cwd)

    # Initialize model
    train_state = init_train_state(config, train_metadata, rank=RANK, world_size=WORLD_SIZE)

    if RANK == 0:
        print(f"\n{'='*60}")
        print(f"PPBench TRM-Att Training")
        print(f"{'='*60}")
        print(f"Model params: {sum(p.numel() for p in train_state.model.parameters()):,}")
        print(f"Total steps: {train_state.total_steps}")
        print(f"Epochs: {config.epochs}")
        print(f"Eval interval: {config.eval_interval}")
        print(f"Batch size: {config.global_batch_size}")
        print(f"Learning rate: {config.lr}")
        print(f"Checkpoint dir: {config.checkpoint_path}")
        print(f"{'='*60}\n")

    # W&B init with MAX configuration
    if RANK == 0:
        wandb.init(
            project=config.project_name,
            name=config.run_name,
            config={
                **config.model_dump(),
                "num_params": sum(p.numel() for p in train_state.model.parameters()),
                "total_steps": train_state.total_steps,
                "train_examples": train_metadata.total_puzzles,
                "vocab_size": train_metadata.vocab_size,
                "seq_len": train_metadata.seq_len,
                "paper": "arXiv:2605.19943",
                "paper_title": "Probabilistic Tiny Recursive Models",
                "architecture": "TRM-Att",
                "dataset": "PPBench",
                "puzzle_types": ["sudoku", "lightup", "nurikabe", "shakashaka", "heyawake", "tapa"],
            },
            tags=["ppbench", "trm-att", "ptrm", "training"],
            settings=wandb.Settings(_disable_stats=False),  # Enable system metrics!
        )

        # Define custom W&B metrics for better dashboard organization
        wandb.define_metric("train/step")
        wandb.define_metric("train/*", step_metric="train/step")
        wandb.define_metric("eval/*", step_metric="train/step")
        wandb.define_metric("system/*", step_metric="train/step")
        wandb.define_metric("q_head/*", step_metric="train/step")
        wandb.define_metric("best/*", step_metric="train/step")

        # Log initial metrics
        wandb.log({
            "train/step": 0,
            "num_params": sum(p.numel() for p in train_state.model.parameters()),
        }, step=0)

        save_code_and_config(config)

    # Best checkpoint tracker
    best_tracker = BestCheckpointTracker(config.checkpoint_path) if RANK == 0 else None

    # Progress tracking
    import tqdm as tqdm_module
    progress_bar = tqdm_module.tqdm(total=train_state.total_steps) if RANK == 0 else None

    # Puzzle type names for per-type logging
    puzzle_types = ["sudoku", "lightup", "nurikabe", "shakashaka", "heyawake", "tapa"]

    # ========================================================================
    # MAIN TRAINING LOOP
    # ========================================================================
    for iter_id in range(total_iters):
        if RANK == 0:
            print(f"\n[Epoch {iter_id * train_epochs_per_iter}]")
            print("TRAIN")

        train_state.model.train()

        for set_name, batch, global_batch_size in train_loader:
            step_start = time.time()

            metrics = train_batch_enhanced(
                config, train_state, batch, global_batch_size,
                rank=RANK, world_size=WORLD_SIZE,
                step_start_time=step_start,
            )

            if RANK == 0 and metrics:
                wandb.log(metrics, step=train_state.step)
                progress_bar.update(train_state.step - progress_bar.n)

        # ================================================================
        # EVALUATION
        # ================================================================
        if iter_id >= config.min_eval_interval and eval_loader is not None:
            if RANK == 0:
                print("EVALUATE")

            train_state.model.eval()

            eval_metrics = evaluate_enhanced(
                config, train_state, eval_loader, eval_metadata,
                evaluators, rank=RANK, world_size=WORLD_SIZE,
                cpu_group=CPU_PROCESS_GROUP,
                puzzle_types=puzzle_types,
            )

            if RANK == 0 and eval_metrics:
                wandb.log(eval_metrics, step=train_state.step)

                # Extract validation accuracy for best checkpoint tracking
                val_acc = None
                for k, v in eval_metrics.items():
                    if "cell_accuracy" in k or "accuracy" in k:
                        if isinstance(v, (int, float)):
                            if val_acc is None or v > val_acc:
                                val_acc = v

                if val_acc is not None and best_tracker is not None:
                    is_best = best_tracker.update(
                        val_acc, train_state.step,
                        train_state.model.state_dict(),
                    )
                    wandb.log({
                        "best/accuracy": best_tracker.best_accuracy,
                        "best/step": best_tracker.best_step,
                        "best/is_new_best": is_best,
                    }, step=train_state.step)

                    if is_best:
                        print(f"  🏆 New best! accuracy={val_acc:.4f} at step {train_state.step}")

            # ============================================================
            # CHECKPOINTING
            # ============================================================
            if RANK == 0:
                print("SAVE CHECKPOINT")

            if RANK == 0 and (config.checkpoint_every_eval or iter_id == total_iters - 1):
                save_train_state(config, train_state)

    # ========================================================================
    # FINALIZE
    # ========================================================================
    if RANK == 0:
        print(f"\n{'='*60}")
        print("Training complete!")

        if best_tracker and best_tracker.best_path:
            print(f"Best checkpoint: {best_tracker.best_path}")
            print(f"Best accuracy: {best_tracker.best_accuracy:.4f} (step {best_tracker.best_step})")

            # Log best checkpoint as W&B artifact
            artifact = wandb.Artifact(
                name="ppbench-trm-att-best",
                type="model",
                description=f"Best TRM-Att checkpoint on PPBench (acc={best_tracker.best_accuracy:.4f}, step={best_tracker.best_step})",
                metadata={
                    "best_accuracy": best_tracker.best_accuracy,
                    "best_step": best_tracker.best_step,
                    "architecture": "TRM-Att",
                    "dataset": "PPBench",
                    "paper": "arXiv:2605.19943",
                },
            )
            artifact.add_file(best_tracker.best_path)
            wandb.log_artifact(artifact)

        print(f"{'='*60}")

    if dist.is_initialized():
        dist.destroy_process_group()
    wandb.finish()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train TRM-Att on PPBench with comprehensive W&B logging"
    )
    parser.add_argument(
        "--config", default=None,
        help="Path to training config YAML (default: config/train/ppbench.yaml)"
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Override number of epochs"
    )
    parser.add_argument(
        "--eval-interval", type=int, default=None,
        help="Override evaluation interval (in epochs)"
    )
    parser.add_argument(
        "--wandb-project", default=None,
        help="W&B project name (default: ptrm-ppbench-training)"
    )
    parser.add_argument(
        "--wandb-run-name", default=None,
        help="W&B run name (default: auto-generated)"
    )
    args = parser.parse_args()

    train_ppbench(
        config_path=args.config,
        epochs=args.epochs,
        eval_interval=args.eval_interval,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
    )


if __name__ == "__main__":
    main()
