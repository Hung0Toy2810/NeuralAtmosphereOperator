"""Probe SFNO temporal-gradient growth for candidate rollout curricula."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT_ROOT), str(PROJECT_ROOT / "src")]

import torch

from configs.model_config import AtmosphereModelConfig
from neural_atmosphere_operator.models.model import AtmosphereNeuralOperator
from neural_atmosphere_operator.pipeline.forecast import rollout_loss


def gradient_norm(model: torch.nn.Module) -> float:
    return math.sqrt(
        sum(
            float(parameter.grad.norm()) ** 2
            for parameter in model.parameters()
            if parameter.grad is not None
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, nargs="+", default=(1, 2, 4))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if min(args.steps) < 1:
        raise ValueError("rollout lengths must be positive")
    config = AtmosphereModelConfig(
        img_size=(13, 24),
        in_channels=4,
        out_channels=4,
        scale_factor=3,
        embed_dim=8,
        num_layers=2,
    )
    print("steps | loss | parameter_grad_norm | initial_state_grad_norm")
    for steps in args.steps:
        torch.manual_seed(args.seed)
        model = AtmosphereNeuralOperator(config).train()
        initial = torch.randn(1, 4, 13, 24, requires_grad=True)
        batch = {
            "input": initial,
            "target": torch.randn(1, steps, 4, 13, 24),
            "time_index": torch.zeros(1, dtype=torch.long),
        }
        loss, _, _ = rollout_loss(model, batch, steps)
        loss.backward()
        assert initial.grad is not None
        print(
            f"{steps:5d} | {float(loss.detach()):.6g} | {gradient_norm(model):.6g} "
            f"| {float(initial.grad.norm()):.6g}"
        )
    print(
        "Large growth across steps indicates that the next curriculum stage "
        "needs a lower learning rate, gradient clipping, or fewer rollout steps."
    )


if __name__ == "__main__":
    main()
