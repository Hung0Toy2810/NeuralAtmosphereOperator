"""Versioned, atomic, and exactly restartable checkpoints."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from .dependencies import runtime_fingerprint


CHECKPOINT_VERSION = 1


def capture_rng_state(device: torch.device | None = None) -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if device is not None and device.type == "cuda":
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    if (
        device is not None
        and device.type == "mps"
        and hasattr(torch.mps, "get_rng_state")
    ):
        state["torch_mps"] = torch.mps.get_rng_state()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    if "torch_cuda" in state:
        if not torch.cuda.is_available():
            raise RuntimeError("Checkpoint contains CUDA RNG but CUDA is unavailable")
        torch.cuda.set_rng_state_all(
            [value.cpu() for value in state["torch_cuda"]]
        )
    if "torch_mps" in state:
        if not torch.backends.mps.is_available():
            raise RuntimeError("Checkpoint contains MPS RNG but MPS is unavailable")
        torch.mps.set_rng_state(state["torch_mps"].cpu())


def save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    scaler: Any,
    train_generator: torch.Generator,
    epoch: int,
    best_validation_loss: float,
    epochs_without_improvement: int,
    early_stopped: bool,
    model_config: dict[str, Any],
    training_config: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    device = next(model.parameters()).device
    torch.save(
        {
            "version": CHECKPOINT_VERSION,
            "epoch": epoch,
            "best_validation_loss": best_validation_loss,
            "epochs_without_improvement": epochs_without_improvement,
            "early_stopped": early_stopped,
            "model_config": model_config,
            "training_config": training_config,
            "runtime": runtime_fingerprint(device),
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict(),
            "rng_state": capture_rng_state(device),
            "train_generator_state": train_generator.get_state(),
        },
        temporary,
    )
    os.replace(temporary, path)


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("version") != CHECKPOINT_VERSION:
        raise ValueError(f"Unsupported checkpoint version in {path}")
    return checkpoint
