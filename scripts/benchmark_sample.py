"""Benchmark candidate SFNO profiles on the real one-day ERA5 sample.

The source grid is uniformly downsampled over the complete sphere, rather than
cropped, so the benchmark retains both poles and longitude periodicity. This is
a small-data capacity/timing check; authoritative CUDA timing still requires
running the same script on the target GPU.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT_ROOT), str(PROJECT_ROOT / "src")]

import numpy as np
import torch
from torch import Tensor
from torch.optim import AdamW
import xarray as xr

from configs.download_data_config import channel_names
from configs.model_config import AtmosphereModelConfig
from configs.pipeline_config import TrainingConfig
from neural_atmosphere_operator.models.loss import latitude_weighted_mse
from neural_atmosphere_operator.models.model import AtmosphereNeuralOperator
from neural_atmosphere_operator.pipeline.runtime import choose_device, seed_everything


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def load_real_states(path: Path, height: int, width: int) -> tuple[Tensor, Tensor]:
    """Load raw states [T,C,H,W] and their actual latitudes."""
    ds = xr.open_zarr(str(path))
    try:
        source_height = int(ds.sizes["latitude"])
        source_width = int(ds.sizes["longitude"])
        if (source_height - 1) % (height - 1) != 0:
            raise ValueError("height must uniformly divide the pole-to-pole grid")
        if source_width % width != 0:
            raise ValueError("width must uniformly divide the longitude grid")
        latitude_stride = (source_height - 1) // (height - 1)
        longitude_stride = source_width // width
        spatial_index = {
            "latitude": slice(None, None, latitude_stride),
            "longitude": slice(None, None, longitude_stride),
        }
        if "state" not in ds:
            raise ValueError(
                "Dataset must contain the canonical flattened 71-channel state"
            )
        stored_channels = tuple(str(value) for value in ds.channel.values)
        if stored_channels != channel_names():
            raise ValueError("Stored state channels do not match the 71-channel contract")
        values = np.asarray(ds["state"].isel(spatial_index).values, dtype=np.float32)
        latitudes = np.asarray(
            ds.latitude.isel({"latitude": spatial_index["latitude"]}).values,
            dtype=np.float32,
        ).copy()
    finally:
        ds.close()

    states = torch.from_numpy(np.ascontiguousarray(values))
    return states, torch.from_numpy(latitudes)


def projected_full_grid_parameters(embed_dim: int, num_layers: int) -> int:
    """Return PyTorch tensor elements for the configured full-grid model."""
    config = AtmosphereModelConfig(embed_dim=embed_dim, num_layers=num_layers)
    internal_height = (config.img_size[0] - 1) // config.scale_factor + 1
    internal_width = config.img_size[1] // config.scale_factor
    modes = int(
        min(internal_height, internal_width // 2) * config.hard_thresholding_fraction
    )
    hidden = 2 * embed_dim
    spectral = num_layers * embed_dim * embed_dim * modes
    mlp = num_layers * (embed_dim * hidden + hidden + hidden * embed_dim)
    projections = config.in_channels * embed_dim + embed_dim * config.out_channels
    norms = num_layers * 4 * embed_dim
    return spectral + mlp + projections + norms


def benchmark_profile(
    states: Tensor,
    latitudes: Tensor,
    *,
    embed_dim: int,
    num_layers: int,
    batch_size: int,
    steps: int,
    learning_rate: float,
    device: torch.device,
    seed: int,
) -> dict[str, float | int | str]:
    if not 1 <= batch_size <= states.shape[0] - 2:
        raise ValueError("Training must leave at least one held-out transition")
    seed_everything(seed)
    height, width = states.shape[-2:]
    config = AtmosphereModelConfig(
        img_size=(height, width),
        in_channels=states.shape[1],
        out_channels=states.shape[1],
        embed_dim=embed_dim,
        num_layers=num_layers,
    )
    model = AtmosphereNeuralOperator(config).to(device).train()
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    inputs = states[:-1][:batch_size].to(device)
    targets = states[1:][:batch_size].to(device)
    validation_input = states[-2:-1].to(device)
    validation_target = states[-1:].to(device)
    device_latitudes = latitudes.to(device)

    def loss_value(model_input: Tensor, model_target: Tensor) -> Tensor:
        return latitude_weighted_mse(
            model(model_input), model_target, latitudes=device_latitudes
        )

    with torch.no_grad():
        initial_loss = float(loss_value(inputs, targets).cpu())
        initial_validation_loss = float(
            loss_value(validation_input, validation_target).cpu()
        )
    elapsed = 0.0
    for step in range(steps + 1):
        optimizer.zero_grad(set_to_none=True)
        synchronize(device)
        started = time.perf_counter()
        loss = loss_value(inputs, targets)
        loss.backward()
        optimizer.step()
        synchronize(device)
        duration = time.perf_counter() - started
        if step > 0:  # first update initializes kernels and AdamW state
            elapsed += duration
    with torch.no_grad():
        final_loss = float(loss_value(inputs, targets).cpu())
        final_validation_loss = float(
            loss_value(validation_input, validation_target).cpu()
        )
    result: dict[str, float | int | str] = {
        "profile": f"E{embed_dim}-L{num_layers}",
        "seed": seed,
        "device": str(device),
        "benchmark_grid": f"{height}x{width}",
        "actual_benchmark_parameters": sum(
            parameter.numel() for parameter in model.parameters()
        ),
        "projected_full_grid_parameters": projected_full_grid_parameters(
            embed_dim, num_layers
        ),
        "seconds_per_training_update": elapsed / steps,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "loss_ratio": final_loss / initial_loss,
        "initial_validation_loss": initial_validation_loss,
        "final_validation_loss": final_validation_loss,
        "validation_loss_ratio": final_validation_loss / initial_validation_loss,
    }
    if device.type == "mps":
        result["mps_allocated_gib_after_update"] = (
            torch.mps.current_allocated_memory() / 2**30
        )
    del optimizer, inputs, targets, validation_input, validation_target
    if device.type == "mps":
        torch.mps.empty_cache()
    return result


def main() -> None:
    model_defaults = AtmosphereModelConfig()
    training_defaults = TrainingConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=PROJECT_ROOT / "data" / "dataset" / "era5_sample_0p5_71ch.zarr",
    )
    parser.add_argument(
        "--device", choices=("auto", "cpu", "mps", "cuda"), default="auto"
    )
    parser.add_argument("--height", type=int, default=61)
    parser.add_argument("--width", type=int, default=120)
    parser.add_argument(
        "--embed-dims",
        type=int,
        nargs="+",
        default=(model_defaults.embed_dim,),
        help="Embedding widths to benchmark; defaults to the current E192 baseline",
    )
    parser.add_argument(
        "--num-layers", type=int, default=model_defaults.num_layers
    )
    parser.add_argument(
        "--batch-size", type=int, default=training_defaults.batch_size
    )
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument(
        "--learning-rate", type=float, default=training_defaults.learning_rate
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=(17, 42, 73))
    args = parser.parse_args()
    if (
        min(
            args.height,
            args.width,
            *args.embed_dims,
            args.num_layers,
            args.batch_size,
            args.steps,
            *args.seeds,
        )
        < 1
    ):
        parser.error("dimensions and counts must be positive")
    if args.learning_rate <= 0:
        parser.error("learning-rate must be positive")

    data_path = args.data.expanduser().resolve()
    states, latitudes = load_real_states(data_path, args.height, args.width)
    if args.batch_size > states.shape[0] - 2:
        parser.error("batch-size must leave the final transition outside training")
    # Fit normalization strictly on states participating in the training
    # transitions. The final held-out target must not influence statistics.
    training_states = states[: args.batch_size + 1]
    means = training_states.mean(dim=(0, 2, 3), keepdim=True)
    stds = training_states.std(dim=(0, 2, 3), keepdim=True).clamp_min(1e-6)
    states = (states - means) / stds
    device = choose_device(args.device)
    profiles = [
        benchmark_profile(
            states,
            latitudes,
            embed_dim=embed_dim,
            num_layers=args.num_layers,
            batch_size=args.batch_size,
            steps=args.steps,
            learning_rate=args.learning_rate,
            device=device,
            seed=seed,
        )
        for embed_dim in args.embed_dims
        for seed in args.seeds
    ]
    aggregates = []
    for embed_dim in args.embed_dims:
        selected = [
            row
            for row in profiles
            if row["profile"] == f"E{embed_dim}-L{args.num_layers}"
        ]
        aggregate: dict[str, float | int | str] = {
            "profile": f"E{embed_dim}-L{args.num_layers}",
            "runs": len(selected),
            "projected_full_grid_parameters": int(
                selected[0]["projected_full_grid_parameters"]
            ),
        }
        for metric in (
            "seconds_per_training_update",
            "final_loss",
            "loss_ratio",
            "final_validation_loss",
            "validation_loss_ratio",
        ):
            values = [float(row[metric]) for row in selected]
            aggregate[f"mean_{metric}"] = statistics.fmean(values)
            aggregate[f"std_{metric}"] = statistics.pstdev(values)
        aggregates.append(aggregate)
    print(
        json.dumps(
            {
                "data": str(data_path),
                "states": list(states.shape),
                "device": str(device),
                "profiles": profiles,
                "aggregates": aggregates,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
