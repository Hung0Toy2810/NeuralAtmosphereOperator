"""Shared runtime helpers for reproducible pipeline commands."""

from __future__ import annotations

import json
import hashlib
import os
import random
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Any, ContextManager

import numpy as np
import torch
from torch import Tensor, nn

from configs.model_config import AtmosphereModelConfig
from configs.pipeline_config import DataPaths
from neural_atmosphere_operator.data.loader import (
    AtmosphereDatasetConfig,
    AtmosphereZarrDataset,
    create_data_loader,
)
from neural_atmosphere_operator.models.model import AtmosphereNeuralOperator


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        device = torch.device(requested)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    if device.type not in {"cpu", "cuda", "mps"}:
        raise ValueError(f"Unsupported device type: {device.type}")
    return device


def amp_context(
    device: torch.device, enabled: bool, dtype_name: str
) -> ContextManager[Any]:
    if not enabled or device.type != "cuda":
        return nullcontext()
    dtype = torch.bfloat16 if dtype_name == "bfloat16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def move_batch(
    batch: dict[str, Tensor], device: torch.device, *, keep_targets_cpu: bool = False
) -> dict[str, Tensor]:
    return {
        key: value
        if key in {"time_index", "target_time_indices"}
        or (keep_targets_cpu and key == "target")
        else value.to(device, non_blocking=device.type == "cuda")
        for key, value in batch.items()
    }


def resolve_data_path(
    paths: DataPaths,
    split: str,
    override: Path | None = None,
) -> Path:
    if split not in {"train", "valid", "test"}:
        raise ValueError(f"Unknown split: {split}")
    return (override or getattr(paths, split)).expanduser().resolve()


def build_loader(
    paths: DataPaths,
    split: str,
    *,
    rollout_steps: int,
    history: int,
    time_step: int,
    batch_size: int,
    num_workers: int,
    data_path: Path | None = None,
    means_path: Path | None = None,
    stds_path: Path | None = None,
    max_samples: int | None = None,
    spatial_crop: tuple[int, int] | None = None,
    generator: torch.Generator | None = None,
    input_noise_std: float = 0.0,
    pin_memory: bool = True,
    prefetch_factor: int = 2,
    persistent_workers: bool = True,
    sample_selection: str = "first",
    require_targets: bool = True,
    lazy_targets: bool = False,
) -> tuple[Any, AtmosphereZarrDataset]:
    resolved_data_path = resolve_data_path(paths, split, data_path)
    use_logical_range = resolved_data_path == paths.dataset.expanduser().resolve()
    start_time, end_time = (
        paths.split_time_range(split) if use_logical_range else (None, None)
    )
    config = AtmosphereDatasetConfig(
        data_path=resolved_data_path,
        means_path=(means_path or paths.means).expanduser().resolve(),
        stds_path=(stds_path or paths.stds).expanduser().resolve(),
        start_time=start_time,
        end_time=end_time,
        history=history,
        rollout_steps=rollout_steps,
        time_step=time_step,
        normalize=True,
        max_samples=max_samples,
        sample_selection=sample_selection,
        require_targets=require_targets,
        lazy_targets=lazy_targets,
        spatial_crop=spatial_crop,
        add_noise=split == "train" and input_noise_std > 0.0,
        noise_std=input_noise_std,
    )
    dataset = AtmosphereZarrDataset(config, training=split == "train")
    if use_logical_range:
        if (
            str(dataset.times[0].astype("datetime64[D]")) != start_time
            or str(dataset.times[-1].astype("datetime64[D]")) != end_time
        ):
            dataset.close()
            raise ValueError(
                f"Canonical {split} split requires complete interval {start_time}..{end_time}; download missing periods"
            )
    loader = create_data_loader(
        dataset,
        batch_size=batch_size,
        shuffle=split == "train",
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        # Persistent worker state is not represented in an epoch-boundary
        # checkpoint.  Recreate training workers every epoch so uninterrupted
        # and resumed runs consume the checkpointed generator identically.
        # Evaluation remains free to keep workers alive because it has no
        # restartable stochastic trajectory.
        persistent_workers=persistent_workers and split != "train",
        # Every valid temporal window is part of every epoch.  The training
        # loop sample-weights a short final batch during gradient accumulation.
        drop_last=False,
        generator=generator,
    )
    return loader, dataset


def checkpoint_time_step(
    training_config: dict[str, Any], requested: int | None = None
) -> int:
    """Return the trained forecast stride and reject incompatible overrides.

    One model application represents exactly the temporal stride used during
    training.  Merely selecting a farther target in the data loader does not
    turn a six-hour model into a twelve-hour model.
    """
    saved = int(training_config.get("time_step", 1))
    if saved < 1:
        raise ValueError("Checkpoint contains an invalid training time_step")
    if requested is not None and requested != saved:
        raise ValueError(
            f"Checkpoint was trained with time_step={saved}; requested "
            f"time_step={requested} would mislabel one model application"
        )
    return saved


def accumulation_bucket_sample_count(
    *,
    batch_index: int,
    total_batches: int,
    total_samples: int,
    batch_size: int,
    accumulation_steps: int,
) -> int:
    """Return the number of samples represented by an accumulation bucket.

    This keeps the accumulated gradient equal to the sample mean when the
    final DataLoader batch is shorter than ``batch_size``.
    """
    if min(total_batches, total_samples, batch_size, accumulation_steps) < 1:
        raise ValueError("batch and sample counts must be positive")
    if not 0 <= batch_index < total_batches:
        raise ValueError("batch_index is outside the DataLoader")
    bucket_start = (batch_index // accumulation_steps) * accumulation_steps
    bucket_end = min(bucket_start + accumulation_steps, total_batches)
    first_sample = bucket_start * batch_size
    last_sample = min(bucket_end * batch_size, total_samples)
    return last_sample - first_sample


def build_model(
    config: AtmosphereModelConfig, device: torch.device
) -> AtmosphereNeuralOperator:
    return AtmosphereNeuralOperator(config).to(device)


def model_config_from_dict(values: dict[str, Any]) -> AtmosphereModelConfig:
    copied = dict(values)
    if "img_size" in copied:
        copied["img_size"] = tuple(copied["img_size"])
    return AtmosphereModelConfig(**copied)


def config_dict(config: AtmosphereModelConfig) -> dict[str, Any]:
    return asdict(config)


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def dataset_signature(dataset: AtmosphereZarrDataset) -> dict[str, Any]:
    """Return a cheap but meaningful identity for exact resume checks."""
    path = dataset.data_path
    metadata_candidates = (
        path / "zarr.json",
        path / ".zmetadata",
        path / ".zgroup",
    )
    metadata = next((item for item in metadata_candidates if item.exists()), None)
    stat = metadata.stat() if metadata is not None else path.stat()
    # Catch payload edits even when .zmetadata is unchanged. This inexpensive
    # file inventory is distinct from the full decoded-content checksum made
    # by compute_stats/validate_data --full-scan.
    inventory = hashlib.sha256()
    for item in sorted(path.rglob("*")):
        if item.is_file():
            payload_stat = item.stat()
            inventory.update(
                f"{item.relative_to(path)}:{payload_stat.st_size}:{payload_stat.st_mtime_ns}\n".encode()
            )
    return {
        "path": str(path),
        "metadata_size": stat.st_size,
        "payload_inventory_sha256": inventory.hexdigest(),
        "metadata_mtime_ns": stat.st_mtime_ns,
        "timesteps": dataset.total_timesteps,
        "first_time": dataset.first_time,
        "last_time": dataset.last_time,
        "cadence_hours": dataset.cadence_hours,
        "spatial_shape": dataset.spatial_shape,
        "channels": dataset.channel_names,
        "channel_units": dataset.channel_units,
        "samples": len(dataset),
        "sample_indices_sha256": hashlib.sha256(
            dataset.sample_indices.tobytes()
        ).hexdigest(),
        "latitude": dataset.latitudes.tolist(),
        "longitude": dataset.longitudes.tolist(),
    }


def statistics_signature(path: Path) -> dict[str, Any]:
    """Fingerprint a small normalization array by content, not only its path."""
    resolved = path.expanduser().resolve()
    digest = hashlib.sha256()
    with resolved.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size": stat.st_size,
        "sha256": digest.hexdigest(),
    }
