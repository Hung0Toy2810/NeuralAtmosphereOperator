"""Verify SFNO forward/backward on CPU, Apple MPS, or NVIDIA CUDA."""

from __future__ import annotations

import argparse
import platform
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT_ROOT), str(PROJECT_ROOT / "src")]

import torch

from configs.model_config import AtmosphereModelConfig
from neural_atmosphere_operator.models.loss import latitude_weighted_mse
from neural_atmosphere_operator.models.model import AtmosphereNeuralOperator
from neural_atmosphere_operator.pipeline.dependencies import validate_supported_runtime
from neural_atmosphere_operator.pipeline.runtime import choose_device


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--forward-only", action="store_true")
    args = parser.parse_args()
    device = choose_device(args.device)
    runtime = validate_supported_runtime(device)
    print(f"platform: {platform.platform()}")
    print(f"torch: {torch.__version__}")
    print(
        "torch-harmonics: "
        f"{runtime['torch_harmonics_distribution']}=={runtime['torch_harmonics']}"
    )
    print(f"operator: {runtime['operator_implementation']}")
    print(f"device: {device}")

    config = AtmosphereModelConfig(
        img_size=(13, 24),
        in_channels=4,
        out_channels=4,
        scale_factor=3,
        embed_dim=8,
        num_layers=2,
    )
    model = AtmosphereNeuralOperator(config).to(device)
    model.train(not args.forward_only)
    state = torch.randn(1, 4, 13, 24, device=device)
    target = torch.randn_like(state)
    prediction = model(state)
    loss = latitude_weighted_mse(prediction, target)
    if not args.forward_only:
        loss.backward()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()
    if not bool(torch.isfinite(prediction).all().cpu()):
        raise RuntimeError("Backend produced a non-finite prediction")
    if not args.forward_only and not all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all().cpu())
        for parameter in model.parameters()
    ):
        raise RuntimeError("Backend produced a non-finite gradient")
    print(f"forward: OK {tuple(prediction.shape)}")
    print(f"backward: {'SKIPPED' if args.forward_only else 'OK'}")
    print(f"loss: {float(loss.detach().cpu()):.6g}")


if __name__ == "__main__":
    main()
