"""Measure SFNO CUDA training memory and throughput on a realistic rollout."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT_ROOT), str(PROJECT_ROOT / "src")]

import torch
from torch.optim import AdamW

from configs.model_config import AtmosphereModelConfig
from neural_atmosphere_operator.models.model import AtmosphereNeuralOperator
from neural_atmosphere_operator.pipeline.dependencies import validate_supported_runtime
from neural_atmosphere_operator.pipeline.forecast import rollout_loss
from neural_atmosphere_operator.pipeline.runtime import amp_context


def main() -> None:
    defaults = AtmosphereModelConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--embed-dim", type=int, default=defaults.embed_dim)
    parser.add_argument("--num-layers", type=int, default=defaults.num_layers)
    parser.add_argument("--scale-factor", type=int, default=defaults.scale_factor)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--rollout-steps", type=int, default=1)
    parser.add_argument(
        "--amp-dtype", choices=("bfloat16", "float16"), default="bfloat16"
    )
    parser.add_argument("--gradient-checkpointing", action="store_true")
    args = parser.parse_args()
    if min(args.batch_size, args.steps, args.rollout_steps) < 1:
        parser.error("batch-size, steps, and rollout-steps must be positive")
    if args.warmup_steps < 0:
        parser.error("warmup-steps cannot be negative")
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA")
    device = torch.device("cuda")
    validate_supported_runtime(device)
    config = AtmosphereModelConfig(
        embed_dim=args.embed_dim,
        num_layers=args.num_layers,
        scale_factor=args.scale_factor,
    )
    model = AtmosphereNeuralOperator(config).to(device)
    optimizer = AdamW(model.parameters(), lr=1e-3)
    state = torch.randn(
        args.batch_size, config.in_channels, *config.img_size, device=device
    )
    target = torch.randn(
        args.batch_size,
        args.rollout_steps,
        config.out_channels,
        *config.img_size,
        device=device,
    )

    def step() -> None:
        optimizer.zero_grad(set_to_none=True)
        with amp_context(device, True, args.amp_dtype):
            loss, _, _ = rollout_loss(
                model,
                {"input": state, "target": target},
                args.rollout_steps,
                gradient_checkpointing=args.gradient_checkpointing,
            )
        loss.backward()
        optimizer.step()

    if args.warmup_steps == 0:
        torch.cuda.synchronize()
        started = time.perf_counter()
    else:
        started = 0.0
    torch.cuda.reset_peak_memory_stats()
    for index in range(args.warmup_steps + args.steps):
        step()
        if index == args.warmup_steps - 1:
            torch.cuda.synchronize()
            started = time.perf_counter()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    print(
        json.dumps(
            {
                "gpu": torch.cuda.get_device_name(),
                "parameters": sum(
                    parameter.numel() for parameter in model.parameters()
                ),
                "seconds_per_step": elapsed / args.steps,
                "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
                "rollout_steps": args.rollout_steps,
                "batch_size": args.batch_size,
                "gradient_checkpointing": args.gradient_checkpointing,
                "amp_dtype": args.amp_dtype,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
