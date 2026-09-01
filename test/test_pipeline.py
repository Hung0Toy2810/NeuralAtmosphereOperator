"""Tests for restartability, scheduling, rollout, and atmospheric metrics."""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT_ROOT), str(PROJECT_ROOT / "src")]

from configs.model_config import AtmosphereModelConfig
from configs.pipeline_config import DataPaths, TrainingConfig
from neural_atmosphere_operator.models.model import AtmosphereNeuralOperator
from neural_atmosphere_operator.pipeline import dependencies
from neural_atmosphere_operator.pipeline.checkpoint import (
    CHECKPOINT_VERSION,
    capture_rng_state,
    load_checkpoint,
    restore_rng_state,
    save_checkpoint,
)
from neural_atmosphere_operator.pipeline.dependencies import (
    enforce_checkpoint_runtime,
    runtime_fingerprint,
)
from neural_atmosphere_operator.pipeline.forecast import rollout_loss
from neural_atmosphere_operator.pipeline.metrics import LeadMetricAccumulator
from neural_atmosphere_operator.pipeline.runtime import (
    accumulation_bucket_sample_count,
    checkpoint_time_step,
)
from neural_atmosphere_operator.pipeline.schedule import warmup_cosine_multiplier
from scripts.evaluate import build_milestone_report


def test_warmup_cosine_schedule_hits_boundaries() -> None:
    values = [
        warmup_cosine_multiplier(
            update,
            total_updates=10,
            warmup_updates=2,
            warmup_start_factor=0.1,
            min_factor=0.01,
        )
        for update in range(11)
    ]
    assert values[0] == pytest.approx(0.1)
    assert values[2] == pytest.approx(1.0)
    assert values[-1] == pytest.approx(0.01)
    assert all(left >= right for left, right in zip(values[2:], values[3:]))


def test_long_validation_and_test_ranges_have_four_rollout_starts() -> None:
    paths = DataPaths()
    assert paths.split_time_range("valid") == ("2019-01-01", "2019-01-16")
    assert paths.split_time_range("test") == ("2019-01-17", "2019-02-16")
    assert TrainingConfig().validation_rollout_steps == 60
    assert TrainingConfig().validation_batch_size == 1
    assert 64 - 60 == 4
    assert 124 - 120 == 4


def test_accumulation_counts_short_final_batch_by_sample() -> None:
    arguments = {
        "total_batches": 3,
        "total_samples": 11,
        "batch_size": 4,
        "accumulation_steps": 2,
    }
    assert accumulation_bucket_sample_count(batch_index=0, **arguments) == 8
    assert accumulation_bucket_sample_count(batch_index=1, **arguments) == 8
    assert accumulation_bucket_sample_count(batch_index=2, **arguments) == 3

    mixed_final_bucket = {
        "total_batches": 4,
        "total_samples": 15,
        "batch_size": 4,
        "accumulation_steps": 2,
    }
    assert accumulation_bucket_sample_count(batch_index=2, **mixed_final_bucket) == 7
    assert accumulation_bucket_sample_count(batch_index=3, **mixed_final_bucket) == 7


def test_checkpoint_time_step_rejects_a_mislabeled_forecast_stride() -> None:
    training = {"time_step": 1}
    assert checkpoint_time_step(training) == 1
    assert checkpoint_time_step(training, 1) == 1
    with pytest.raises(ValueError, match="trained with time_step=1"):
        checkpoint_time_step(training, 2)
    with pytest.raises(ValueError, match="invalid"):
        checkpoint_time_step({"time_step": 0})


def test_three_day_milestone_report_groups_channels() -> None:
    rows = [
        {"lead": lead, "lead_hours": lead * 6.0, "normalized_rmse": lead / 10}
        for lead in range(1, 61)
    ]
    channels = [
        {"lead": lead, "lead_hours": lead * 6.0, "channel": channel}
        for lead in range(1, 61)
        for channel in range(2)
    ]
    levels = build_milestone_report(rows, channels, every_days=3)
    assert [level["lead_steps"] for level in levels] == [12, 24, 36, 48, 60]
    assert [level["name"] for level in levels] == [
        "day_03",
        "day_06",
        "day_09",
        "day_12",
        "day_15",
    ]
    assert all(len(level["channels"]) == 2 for level in levels)


def test_rng_state_round_trip() -> None:
    random.seed(17)
    np.random.seed(17)
    torch.manual_seed(17)
    state = capture_rng_state()
    expected = (random.random(), np.random.random(), torch.rand(3))
    random.random()
    np.random.random()
    torch.rand(3)
    restore_rng_state(state)
    actual = (random.random(), np.random.random(), torch.rand(3))
    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    torch.testing.assert_close(actual[2], expected[2], rtol=0, atol=0)


def test_atomic_checkpoint_contains_resume_state(tmp_path: Path) -> None:
    model = torch.nn.Linear(3, 2)
    optimizer = AdamW(model.parameters(), lr=1e-3)
    scheduler = LambdaLR(optimizer, lambda update: 0.9**update)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    generator = torch.Generator().manual_seed(123)
    model(torch.ones(1, 3)).square().mean().backward()
    optimizer.step()
    scheduler.step()
    path = tmp_path / "last.pt"
    save_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        train_generator=generator,
        epoch=3,
        best_validation_loss=0.25,
        epochs_without_improvement=2,
        early_stopped=False,
        model_config={"name": "test"},
        training_config={"epochs": 10},
    )
    checkpoint = load_checkpoint(path, torch.device("cpu"))
    assert checkpoint["version"] == CHECKPOINT_VERSION
    assert checkpoint["epoch"] == 3
    assert checkpoint["optimizer_state"]["state"]
    assert "rng_state" in checkpoint
    assert checkpoint["runtime"]["operator_source"].endswith("models.sfno")


def test_runtime_guard_detects_operator_change() -> None:
    current = runtime_fingerprint(torch.device("cpu"))
    assert len(current["source_sha256"]) == 64
    checkpoint = {"runtime": dict(current)}
    assert enforce_checkpoint_runtime(checkpoint, current) == ()
    checkpoint["runtime"]["torch_harmonics"] = "different"
    with pytest.raises(RuntimeError, match="torch_harmonics"):
        enforce_checkpoint_runtime(checkpoint, current)
    assert enforce_checkpoint_runtime(checkpoint, current, allow_mismatch=True) == (
        "torch_harmonics",
    )


def test_exact_operator_runtime_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    actual = runtime_fingerprint(torch.device("cpu"))
    monkeypatch.setattr(
        dependencies,
        "runtime_fingerprint",
        lambda _device: {**actual, "torch_harmonics": "0.1.0"},
    )
    with pytest.raises(RuntimeError, match="torch-harmonics==0.7.4"):
        dependencies.validate_supported_runtime(torch.device("cpu"))


def test_gradient_checkpointed_rollout_retains_temporal_credit() -> None:
    config = AtmosphereModelConfig(
        img_size=(13, 24),
        in_channels=4,
        out_channels=4,
        scale_factor=3,
        embed_dim=8,
        num_layers=1,
    )
    model = AtmosphereNeuralOperator(config).train()
    model_input = torch.randn(1, 4, 13, 24, requires_grad=True)
    batch = {
        "input": model_input,
        "target": torch.randn(1, 2, 4, 13, 24),
        "time_index": torch.zeros(1, dtype=torch.long),
    }
    loss, losses, predictions = rollout_loss(
        model,
        batch,
        2,
        gradient_checkpointing=True,
        discount_factor=0.8,
    )
    loss.backward()
    assert len(losses) == len(predictions) == 2
    assert model_input.grad is not None
    assert torch.isfinite(model_input.grad).all()


def test_physical_metric_uses_channel_standard_deviation() -> None:
    target = torch.zeros(1, 2, 5, 8)
    prediction = torch.ones_like(target)
    accumulator = LeadMetricAccumulator()
    accumulator.update(
        prediction,
        target,
        torch.tensor([2.0, 4.0]).view(1, 2, 1, 1),
        torch.linspace(90.0, -90.0, 5),
        torch.zeros(1, 2, 5, 8),
    )
    result = accumulator.result(1)
    channels = accumulator.channel_results(
        1, ("a", "b"), units=("m s**-1", "K")
    )
    assert result["normalized_mse"] == pytest.approx(1.0)
    assert result["normalized_rmse"] == pytest.approx(1.0)
    assert channels[0]["physical_rmse"] == pytest.approx(2.0)
    assert channels[1]["physical_rmse"] == pytest.approx(4.0)
    assert [row["unit"] for row in channels] == ["m s**-1", "K"]


def test_acc_is_mean_of_per_initialization_spatial_correlations() -> None:
    target_pattern = torch.tensor([1.0, -1.0]).view(1, 1, 1, 2)
    target = target_pattern.expand(2, 1, 3, 2).clone()
    prediction = target.clone()
    prediction[0] *= 100.0  # energetic but perfectly correlated
    prediction[1] *= -1.0   # quiet and perfectly anti-correlated

    accumulator = LeadMetricAccumulator()
    accumulator.update(
        prediction,
        target,
        torch.ones(1, 1, 1, 1),
        torch.linspace(90.0, -90.0, 3),
        torch.zeros(1, 1, 3, 2),
    )
    assert accumulator.result(1)["normalized_acc"] == pytest.approx(0.0, abs=1e-7)
    assert accumulator.channel_results(1, ("test",))[0][
        "normalized_acc"
    ] == pytest.approx(0.0, abs=1e-7)


def test_acc_is_invariant_to_batch_partitioning() -> None:
    generator = torch.Generator().manual_seed(31)
    target = torch.randn(5, 2, 5, 8, generator=generator)
    prediction = target + 0.7 * torch.randn(5, 2, 5, 8, generator=generator)
    climatology = torch.randn(1, 2, 5, 8, generator=generator)
    arguments = (
        torch.ones(1, 2, 1, 1),
        torch.linspace(90.0, -90.0, 5),
        climatology,
    )

    whole = LeadMetricAccumulator()
    whole.update(prediction, target, *arguments)
    partitioned = LeadMetricAccumulator()
    partitioned.update(prediction[:2], target[:2], *arguments)
    partitioned.update(prediction[2:], target[2:], *arguments)

    assert partitioned.result(1)["normalized_acc"] == pytest.approx(
        whole.result(1)["normalized_acc"], abs=1e-12
    )
    for left, right in zip(
        partitioned.channel_results(1, ("a", "b")),
        whole.channel_results(1, ("a", "b")),
    ):
        assert left["normalized_acc"] == pytest.approx(
            right["normalized_acc"], abs=1e-12
        )
