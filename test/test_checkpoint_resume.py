"""Focused regression tests for exact epoch-boundary checkpoint resume."""

from __future__ import annotations

import copy
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from torch import Tensor, nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT_ROOT), str(PROJECT_ROOT / "src")]

import neural_atmosphere_operator.pipeline.checkpoint as checkpoint_module
from neural_atmosphere_operator.pipeline.checkpoint import (
    load_checkpoint,
    restore_training_state,
    save_checkpoint,
)


def _components(seed: int) -> tuple[
    nn.Module,
    AdamW,
    LambdaLR,
    torch.amp.GradScaler,
    torch.Generator,
]:
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(4, 8), nn.Dropout(0.25), nn.Linear(8, 2))
    optimizer = AdamW(model.parameters(), lr=2e-3, betas=(0.9, 0.95))
    scheduler = LambdaLR(optimizer, lambda update: 0.97**update)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    generator = torch.Generator().manual_seed(seed + 1)
    return model, optimizer, scheduler, scaler, generator


def _update(
    model: nn.Module,
    optimizer: AdamW,
    scheduler: LambdaLR,
    scaler: torch.amp.GradScaler,
    inputs: Tensor,
    targets: Tensor,
) -> None:
    optimizer.zero_grad(set_to_none=True)
    loss = (model(inputs) - targets).square().mean()
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    scheduler.step()


def _assert_nested_equal(actual: Any, expected: Any) -> None:
    if isinstance(expected, Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_nested_equal(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected):
            _assert_nested_equal(actual_item, expected_item)
    else:
        assert actual == expected


def test_adam_moments_and_next_update_are_exact_after_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        checkpoint_module,
        "runtime_fingerprint",
        lambda device: {"device_type": device.type, "test_runtime": True},
    )
    random.seed(41)
    np.random.seed(41)
    torch.manual_seed(41)
    model, optimizer, scheduler, scaler, generator = _components(41)
    inputs = torch.arange(20, dtype=torch.float32).reshape(5, 4) / 10
    targets = torch.arange(10, dtype=torch.float32).reshape(5, 2) / 7
    _update(model, optimizer, scheduler, scaler, inputs, targets)

    path = tmp_path / "last.pt"
    save_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        train_generator=generator,
        epoch=0,
        completed_updates=1,
        best_validation_loss=0.5,
        epochs_without_improvement=0,
        early_stopped=False,
        model_config={"name": "resume-test"},
        training_config={"epochs": 2},
    )
    checkpoint = load_checkpoint(path, torch.device("cpu"))
    adam_state = next(iter(checkpoint["optimizer_state"]["state"].values()))
    assert {"step", "exp_avg", "exp_avg_sq"}.issubset(adam_state)

    expected_python = random.random()
    expected_numpy = float(np.random.random())
    expected_torch = torch.rand(3)
    expected_loader = torch.rand(3, generator=generator)
    _update(model, optimizer, scheduler, scaler, inputs, targets)
    expected_model = copy.deepcopy(model.state_dict())
    expected_optimizer = copy.deepcopy(optimizer.state_dict())
    expected_scheduler = copy.deepcopy(scheduler.state_dict())

    random.seed(999)
    np.random.seed(999)
    torch.manual_seed(999)
    resumed = _components(999)
    (
        resumed_model,
        resumed_optimizer,
        resumed_scheduler,
        resumed_scaler,
        resumed_gen,
    ) = resumed
    restore_training_state(
        checkpoint,
        model=resumed_model,
        optimizer=resumed_optimizer,
        scheduler=resumed_scheduler,
        scaler=resumed_scaler,
        train_generator=resumed_gen,
    )

    assert random.random() == expected_python
    assert float(np.random.random()) == expected_numpy
    torch.testing.assert_close(torch.rand(3), expected_torch, rtol=0, atol=0)
    torch.testing.assert_close(
        torch.rand(3, generator=resumed_gen), expected_loader, rtol=0, atol=0
    )
    _update(
        resumed_model,
        resumed_optimizer,
        resumed_scheduler,
        resumed_scaler,
        inputs,
        targets,
    )
    _assert_nested_equal(resumed_model.state_dict(), expected_model)
    _assert_nested_equal(resumed_optimizer.state_dict(), expected_optimizer)
    _assert_nested_equal(resumed_scheduler.state_dict(), expected_scheduler)


def test_load_rejects_checkpoint_missing_resume_state(tmp_path: Path) -> None:
    path = tmp_path / "broken.pt"
    torch.save({"version": checkpoint_module.CHECKPOINT_VERSION}, path)
    with pytest.raises(ValueError, match="missing"):
        load_checkpoint(path, torch.device("cpu"))
