"""Autoregressive rollout shared by training, evaluation, and inference."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import cast

import torch
import numpy as np
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from neural_atmosphere_operator.models.loss import latitude_weighted_mse
from neural_atmosphere_operator.models.model import AtmosphereNeuralOperator


def autoregressive_predictions(
    model: AtmosphereNeuralOperator,
    batch: dict[str, Tensor],
    steps: int,
    *,
    gradient_checkpointing: bool = False,
    target_provider: Callable[[int], Tensor] | None = None,
) -> Iterator[tuple[Tensor, Tensor]]:
    """Yield prediction/target pairs while shifting the history window."""
    if steps < 1 or (target_provider is None and batch["target"].shape[1] < steps):
        raise ValueError("Batch does not contain enough rollout targets")
    for step, prediction in enumerate(
        forecast_states(
            model, batch["input"], steps, gradient_checkpointing=gradient_checkpointing
        )
    ):
        target = (
            target_provider(step)
            if target_provider is not None
            else batch["target"][:, step]
        )
        yield prediction, target.to(prediction.device, non_blocking=True)


def lazy_target_provider(dataset, raw_batch) -> Callable[[int], Tensor] | None:
    """Load only one verification lead per batch in the main process.

    Workers fetch input histories and indices; CPU and device truth storage no
    longer scale with rollout horizon. Each process keeps its own Zarr handle.
    """
    if not dataset.config.lazy_targets:
        return None
    indices = raw_batch["target_time_indices"].cpu().numpy()

    def read(lead):
        if lead < 0 or lead >= indices.shape[1]:
            raise IndexError("Verification lead outside available target indices")
        fields = [
            dataset.normalizer.normalize(
                dataset._read_timestep_channels(dataset._get_dataset(), int(index))
            )
            for index in indices[:, lead]
        ]
        return torch.from_numpy(np.stack(fields).astype(np.float32, copy=False))

    return read


def forecast_states(
    model: AtmosphereNeuralOperator,
    state: Tensor,
    steps: int,
    *,
    gradient_checkpointing: bool = False,
) -> Iterator[Tensor]:
    """Forecast solely from observed history; keep the autoregressive graph intact."""
    if steps < 1:
        raise ValueError("Forecast steps must be positive")
    output_channels = model.config.out_channels
    for _ in range(steps):
        if gradient_checkpointing and model.training:
            prediction = cast(Tensor, checkpoint(model, state, use_reentrant=False))
        else:
            prediction = model(state)
        yield prediction
        state = (
            prediction
            if model.config.in_channels == output_channels
            else torch.cat((state[:, output_channels:], prediction), dim=1)
        )


def rollout_loss(
    model: AtmosphereNeuralOperator,
    batch: dict[str, Tensor],
    steps: int,
    *,
    gradient_checkpointing: bool = False,
    discount_factor: float = 1.0,
    loss_fn: Callable[[Tensor, Tensor], Tensor] | None = None,
) -> tuple[Tensor, list[Tensor], list[Tensor]]:
    """Return discounted rollout loss, per-lead losses, and predictions."""
    if not 0.0 < discount_factor <= 1.0:
        raise ValueError("discount_factor must be in (0, 1]")
    criterion = loss_fn or latitude_weighted_mse
    predictions: list[Tensor] = []
    losses: list[Tensor] = []
    weights: list[float] = []
    for lead, (prediction, target) in enumerate(
        autoregressive_predictions(
            model,
            batch,
            steps,
            gradient_checkpointing=gradient_checkpointing,
        )
    ):
        predictions.append(prediction)
        losses.append(criterion(prediction, target))
        weights.append(discount_factor**lead)
    weight_tensor = losses[0].new_tensor(weights)
    total = (torch.stack(losses) * weight_tensor).sum() / weight_tensor.sum()
    return total, losses, predictions
