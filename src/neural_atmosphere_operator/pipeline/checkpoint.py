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


CHECKPOINT_VERSION = 2

_REQUIRED_KEYS = frozenset(
    {
        "version",
        "resume_boundary",
        "epoch",
        "completed_updates",
        "best_validation_loss",
        "epochs_without_improvement",
        "early_stopped",
        "model_config",
        "training_config",
        "runtime",
        "model_state",
        "optimizer_state",
        "scheduler_state",
        "scaler_state",
        "rng_state",
        "train_generator_state",
        "state_classes",
    }
)


def _class_path(value: Any) -> str:
    cls = type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def _validate_checkpoint_payload(checkpoint: Any, path: Path) -> dict[str, Any]:
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Checkpoint payload in {path} is not a dictionary")
    if checkpoint.get("version") != CHECKPOINT_VERSION:
        raise ValueError(
            f"Unsupported checkpoint version in {path}: "
            f"expected {CHECKPOINT_VERSION}, received {checkpoint.get('version')!r}"
        )
    missing = sorted(_REQUIRED_KEYS - checkpoint.keys())
    if missing:
        raise ValueError(f"Checkpoint in {path} is missing: {', '.join(missing)}")
    if checkpoint["resume_boundary"] != "epoch_end":
        raise ValueError(f"Checkpoint in {path} is not an epoch-end checkpoint")
    if int(checkpoint["epoch"]) < 0 or int(checkpoint["completed_updates"]) < 0:
        raise ValueError(f"Checkpoint in {path} has invalid progress counters")
    optimizer_state = checkpoint["optimizer_state"]
    if not isinstance(optimizer_state, dict) or not {
        "state",
        "param_groups",
    }.issubset(optimizer_state):
        raise ValueError(f"Checkpoint in {path} has an invalid optimizer state")
    mapping_states = (
        "model_state",
        "scheduler_state",
        "scaler_state",
        "rng_state",
        "state_classes",
    )
    if any(not isinstance(checkpoint[name], dict) for name in mapping_states):
        raise ValueError(f"Checkpoint in {path} contains an invalid state mapping")
    if not {"python", "numpy", "torch_cpu"}.issubset(checkpoint["rng_state"]):
        raise ValueError(f"Checkpoint in {path} has an incomplete RNG state")
    if not {"optimizer", "scheduler", "scaler"}.issubset(
        checkpoint["state_classes"]
    ):
        raise ValueError(f"Checkpoint in {path} has incomplete state class metadata")
    if not isinstance(checkpoint["train_generator_state"], torch.Tensor):
        raise ValueError(f"Checkpoint in {path} has an invalid loader generator state")
    return checkpoint


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
    completed_updates: int,
    best_validation_loss: float,
    epochs_without_improvement: int,
    early_stopped: bool,
    model_config: dict[str, Any],
    training_config: dict[str, Any],
) -> None:
    if epoch < 0 or completed_updates < 0:
        raise ValueError("Checkpoint progress counters cannot be negative")
    if scheduler.last_epoch != completed_updates:
        raise ValueError(
            "Scheduler progress must match the completed optimizer-update count"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    device = next(model.parameters()).device
    torch.save(
        {
            "version": CHECKPOINT_VERSION,
            # No gradients are pending here: the checkpoint is written only
            # after the final optimizer update and validation of an epoch.
            "resume_boundary": "epoch_end",
            "epoch": epoch,
            "completed_updates": completed_updates,
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
            "state_classes": {
                "optimizer": _class_path(optimizer),
                "scheduler": _class_path(scheduler),
                "scaler": _class_path(scaler),
            },
        },
        temporary,
    )
    os.replace(temporary, path)


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    return _validate_checkpoint_payload(checkpoint, path)


def restore_training_state(
    checkpoint: dict[str, Any],
    *,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    scaler: Any,
    train_generator: torch.Generator,
) -> None:
    """Restore every state that affects the next epoch's numerical trajectory."""
    expected_classes = {
        "optimizer": _class_path(optimizer),
        "scheduler": _class_path(scheduler),
        "scaler": _class_path(scaler),
    }
    mismatches = [
        name
        for name, value in expected_classes.items()
        if checkpoint["state_classes"].get(name) != value
    ]
    if mismatches:
        raise ValueError(
            "Resume state class mismatch in: " + ", ".join(mismatches)
        )
    model.load_state_dict(checkpoint["model_state"], strict=True)
    # Adam/AdamW moments, per-parameter step counters and parameter-group
    # hyperparameters all live inside optimizer_state.
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    scheduler.load_state_dict(checkpoint["scheduler_state"])
    scaler.load_state_dict(checkpoint["scaler_state"])
    train_generator.set_state(checkpoint["train_generator_state"].cpu())
    restore_rng_state(checkpoint["rng_state"])
