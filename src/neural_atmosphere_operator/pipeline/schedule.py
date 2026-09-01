"""Update-level warmup and cosine learning-rate scheduling."""

from __future__ import annotations

import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def warmup_cosine_multiplier(
    update: int,
    *,
    total_updates: int,
    warmup_updates: int,
    warmup_start_factor: float,
    min_factor: float,
) -> float:
    if total_updates < 1:
        raise ValueError("total_updates must be positive")
    if not 0 <= warmup_updates < total_updates:
        raise ValueError("warmup_updates must be in [0, total_updates)")
    if not 0 < warmup_start_factor <= 1:
        raise ValueError("warmup_start_factor must be in (0, 1]")
    if not 0 <= min_factor <= 1:
        raise ValueError("min_factor must be in [0, 1]")

    bounded = min(max(update, 0), total_updates)
    if warmup_updates and bounded < warmup_updates:
        progress = bounded / warmup_updates
        return warmup_start_factor + (1.0 - warmup_start_factor) * progress
    progress = (bounded - warmup_updates) / (total_updates - warmup_updates)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_factor + (1.0 - min_factor) * cosine


def create_warmup_cosine_scheduler(
    optimizer: Optimizer,
    *,
    total_updates: int,
    warmup_updates: int,
    warmup_start_factor: float,
    min_learning_rate: float,
    base_learning_rate: float,
) -> LambdaLR:
    if base_learning_rate <= 0:
        raise ValueError("base_learning_rate must be positive")
    if not 0 <= min_learning_rate <= base_learning_rate:
        raise ValueError("min_learning_rate must be in [0, base_learning_rate]")
    min_factor = min_learning_rate / base_learning_rate
    return LambdaLR(
        optimizer,
        lr_lambda=lambda update: warmup_cosine_multiplier(
            update,
            total_updates=total_updates,
            warmup_updates=warmup_updates,
            warmup_start_factor=warmup_start_factor,
            min_factor=min_factor,
        ),
    )
