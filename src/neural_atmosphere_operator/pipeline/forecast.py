"""Autoregressive rollout shared by training, evaluation, and inference."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import cast

import torch
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
) -> Iterator[tuple[Tensor, Tensor]]:
    """Yield prediction/target pairs while shifting the history window."""
    if steps < 1 or batch["target"].shape[1] < steps:
        raise ValueError("Batch does not contain enough rollout targets")
    state = batch["input"]
    output_channels = model.config.out_channels
    for step in range(steps):
        if gradient_checkpointing and model.training:
            prediction = cast(
                Tensor,
                checkpoint(model, state, use_reentrant=False),
            )
        else:
            prediction = model(state)
        yield prediction, batch["target"][:, step]
        if model.config.in_channels == output_channels:
            state = prediction
        else:
            state = torch.cat((state[:, output_channels:], prediction), dim=1)


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
