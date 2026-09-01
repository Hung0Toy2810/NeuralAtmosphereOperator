"""Worker-safe Zarr and array data pipeline for atmospheric weather forecasting."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, TypedDict

import numpy as np
import torch
import xarray as xr
import zarr
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from .normalization import AtmosphereNormalizer
from configs.download_data_config import (
    DEFAULT_LEVEL_SELECTIONS,
    DEFAULT_SURFACE_VARIABLES,
    channel_names as configured_channel_names,
)

LOGGER = logging.getLogger(__name__)

DEFAULT_SURFACE_VARS = DEFAULT_SURFACE_VARIABLES
DEFAULT_LEVEL_SPECS = DEFAULT_LEVEL_SELECTIONS


class AtmosphereSample(TypedDict):
    """Standard training sample contract for atmospheric forecasting models."""

    input: Tensor  # [C_in, H, W] where C_in = (history + 1) * C
    target: Tensor  # [rollout_steps, C_out, H, W]
    time_index: Tensor  # [1] scalar index of base time t


@dataclass(frozen=True, slots=True)
class AtmosphereDatasetConfig:
    """Configuration for :class:`AtmosphereZarrDataset`."""

    data_path: str | Path
    means_path: str | Path | np.ndarray | Tensor | None = None
    stds_path: str | Path | np.ndarray | Tensor | None = None
    surface_variables: tuple[str, ...] = DEFAULT_SURFACE_VARS
    level_selections: tuple[tuple[str, tuple[int, ...]], ...] = DEFAULT_LEVEL_SPECS
    start_time: str | None = None
    end_time: str | None = None
    history: int = 0
    rollout_steps: int = 1
    time_step: int = 1
    normalize: bool = True
    add_noise: bool = False
    noise_std: float = 0.0
    spatial_crop: tuple[int, int] | None = None
    max_samples: int | None = None

    def __post_init__(self) -> None:
        if self.history < 0:
            raise ValueError("history cannot be negative")
        if self.rollout_steps < 1:
            raise ValueError("rollout_steps must be at least 1")
        if self.time_step < 1:
            raise ValueError("time_step must be at least 1")
        if not self.surface_variables and not self.level_selections:
            raise ValueError("At least one variable must be specified")
        if any(not levels for _, levels in self.level_selections):
            raise ValueError("Every level variable requires selected pressure levels")
        if self.start_time and self.end_time and self.start_time > self.end_time:
            raise ValueError("start_time must not be later than end_time")
        if self.add_noise and self.noise_std <= 0:
            raise ValueError("noise_std must be positive when add_noise=True")

    @property
    def channel_count(self) -> int:
        return len(self.channel_names)

    @property
    def channel_names(self) -> tuple[str, ...]:
        return configured_channel_names(self.surface_variables, self.level_selections)


class AtmosphereZarrDataset(Dataset[AtmosphereSample]):
    """Worker-safe Zarr dataset for global atmospheric forecasting."""

    def __init__(
        self, config: AtmosphereDatasetConfig, *, training: bool = True
    ) -> None:
        self.config = config
        self.training = training
        self.data_path = Path(config.data_path).expanduser().resolve()
        if not self.data_path.exists():
            raise FileNotFoundError(f"Dataset path not found: {self.data_path}")

        self.normalizer = AtmosphereNormalizer(
            means=config.means_path,
            stds=config.stds_path,
            normalize=config.normalize,
        )
        if config.normalize:
            assert self.normalizer.means is not None
            if self.normalizer.means.size != config.channel_count:
                raise ValueError(
                    "Normalization statistics must contain exactly "
                    f"{config.channel_count} channels"
                )

        # Inspect dimensions and coordinates from metadata lazily
        self._inspect_dataset()

        # Cache for worker-local zarr/xarray handles
        self._worker_ds: xr.Dataset | None = None
        self._worker_zarr: zarr.Group | None = None

    def _inspect_dataset(self) -> None:
        ds = self._select_time_range(xr.open_zarr(str(self.data_path), consolidated=True))
        try:
            if "state" in ds:
                if "channel" not in ds.state.dims:
                    raise ValueError("state array must have a channel dimension")
                stored_channels = tuple(str(value) for value in ds.channel.values)
                if stored_channels != self.config.channel_names:
                    raise ValueError("Stored state channels do not match configured channels")
                self.channel_units = self._string_channel_coordinate(
                    ds, "channel_units", stored_channels, fallback=""
                )
                self.channel_long_names = self._string_channel_coordinate(
                    ds, "channel_long_name", stored_channels, fallback=""
                )
                self.channel_source_variables = self._string_channel_coordinate(
                    ds, "channel_source_variable", stored_channels, fallback=""
                )
                if "channel_pressure_level_hpa" in ds.coords:
                    pressure_levels = np.asarray(
                        ds["channel_pressure_level_hpa"].values, dtype=np.float32
                    )
                    if pressure_levels.shape != (len(stored_channels),):
                        raise ValueError(
                            "channel_pressure_level_hpa must align with channel"
                        )
                    self.channel_pressure_levels_hpa = tuple(
                        float(value) for value in pressure_levels
                    )
                else:
                    self.channel_pressure_levels_hpa = (float("nan"),) * len(
                        stored_channels
                    )
                source = str(ds.attrs.get("source", ""))
                if source.startswith("gs://weatherbench2/") and int(
                    ds.attrs.get("regridding_version", -1)
                ) != 2:
                    raise ValueError(
                        "WeatherBench2 state store uses an unaudited regridding "
                        "version; regenerate it with the current downloader"
                    )
                self._state_layout = True
            else:
                required_variables = self.config.surface_variables + tuple(
                    variable for variable, _ in self.config.level_selections
                )
                missing_variables = [name for name in required_variables if name not in ds]
                if missing_variables:
                    raise ValueError(f"Dataset is missing variables: {missing_variables}")
                self._state_layout = False
                metadata = [
                    (name, None) for name in self.config.surface_variables
                ] + [
                    (variable, level)
                    for variable, levels in self.config.level_selections
                    for level in levels
                ]
                self.channel_units = tuple(
                    str(ds[variable].attrs.get("units", "1" if variable == "relative_humidity" else ""))
                    for variable, _ in metadata
                )
                self.channel_long_names = tuple(
                    str(ds[variable].attrs.get("long_name", variable))
                    for variable, _ in metadata
                )
                self.channel_source_variables = tuple(
                    variable for variable, _ in metadata
                )
                self.channel_pressure_levels_hpa = tuple(
                    float("nan") if level is None else float(level)
                    for _, level in metadata
                )
            for coordinate in ("time", "latitude", "longitude"):
                if coordinate not in ds.coords:
                    raise ValueError(
                        f"Dataset is missing required coordinate {coordinate!r}"
                    )
            self.total_timesteps = len(ds.time)
            time_values = np.asarray(ds.time.values)
            if self.total_timesteps < 2:
                raise ValueError("Dataset must contain at least two timesteps")
            all_latitudes = np.asarray(ds.latitude.values)
            all_longitudes = np.asarray(ds.longitude.values)
            self.first_time = str(time_values[0])
            self.last_time = str(time_values[-1])

            if not np.issubdtype(time_values.dtype, np.datetime64):
                raise ValueError("ERA5 time coordinate must use datetime64 values")
            time_nanoseconds = time_values.astype("datetime64[ns]").astype(np.int64)
            time_differences = np.diff(time_nanoseconds)
            if not np.all(time_differences > 0):
                raise ValueError("Time coordinates must be strictly increasing")
            if not np.all(time_differences == time_differences[0]):
                raise ValueError("Time coordinates must have a regular cadence")
            self.cadence_hours = float(time_differences[0] / 3_600_000_000_000)
            if not np.isfinite(self.cadence_hours) or self.cadence_hours <= 0:
                raise ValueError("Time cadence must be positive")
            self.times = time_values.astype("datetime64[ns]").copy()

            if not np.all(np.isfinite(all_latitudes)) or not np.all(
                np.isfinite(all_longitudes)
            ):
                raise ValueError("Latitude and longitude coordinates must be finite")
            if len(all_latitudes) < 3 or len(all_longitudes) < 4:
                raise ValueError("Grid is too small for spherical forecasting")
            latitude_steps = np.diff(all_latitudes.astype(np.float64))
            longitude_steps = np.diff(all_longitudes.astype(np.float64))
            if not (np.all(latitude_steps > 0) or np.all(latitude_steps < 0)):
                raise ValueError("Latitude coordinates must be strictly monotonic")
            if not np.all(longitude_steps > 0):
                raise ValueError("Longitude coordinates must be strictly increasing")
            if not np.allclose(
                np.abs(latitude_steps),
                np.abs(latitude_steps[0]),
                rtol=1e-5,
                atol=1e-6,
            ):
                raise ValueError("SFNO requires regularly spaced latitudes")
            if not np.allclose(
                longitude_steps,
                longitude_steps[0],
                rtol=1e-5,
                atol=1e-6,
            ):
                raise ValueError("SFNO requires regularly spaced longitudes")

            if not self._state_layout and self.config.level_selections:
                available_levels = list(ds.level.values) if "level" in ds else []
                for variable, levels in self.config.level_selections:
                    for level in levels:
                        if level not in available_levels:
                            raise ValueError(
                                f"Level {level} hPa for {variable} not present in "
                                f"dataset levels {available_levels}"
                            )
        finally:
            ds.close()

        full_h, full_w = len(all_latitudes), len(all_longitudes)
        if self.config.spatial_crop is not None:
            crop_h, crop_w = self.config.spatial_crop
            if crop_h < 1 or crop_w < 1:
                raise ValueError("spatial_crop dimensions must be positive")
            if crop_h > full_h or crop_w > full_w:
                raise ValueError(
                    f"Crop {self.config.spatial_crop} exceeds dataset shape "
                    f"{(full_h, full_w)}"
                )
            self.spatial_shape = (crop_h, crop_w)
        else:
            self.spatial_shape = (full_h, full_w)
        self.latitudes = np.asarray(
            all_latitudes[: self.spatial_shape[0]], dtype=np.float32
        ).copy()
        self.longitudes = np.asarray(
            all_longitudes[: self.spatial_shape[1]], dtype=np.float32
        ).copy()

        # Compute valid sample count
        context = self.config.history * self.config.time_step
        horizon = self.config.rollout_steps * self.config.time_step
        valid_count = max(0, self.total_timesteps - context - horizon)

        if valid_count == 0:
            raise ValueError(
                f"Dataset with {self.total_timesteps} steps is too short for history={self.config.history}, "
                f"rollout_steps={self.config.rollout_steps}, time_step={self.config.time_step}"
            )

        if self.config.max_samples is not None:
            self.sample_count = min(valid_count, self.config.max_samples)
        else:
            self.sample_count = valid_count

    @staticmethod
    def _string_channel_coordinate(
        ds: xr.Dataset,
        name: str,
        channels: tuple[str, ...],
        *,
        fallback: str,
    ) -> tuple[str, ...]:
        """Read and validate a string coordinate aligned with ``channel``."""
        if name not in ds.coords:
            return (fallback,) * len(channels)
        values = tuple(str(value) for value in ds[name].values)
        if len(values) != len(channels):
            raise ValueError(f"{name} must align with the channel dimension")
        return values

    def _get_dataset(self) -> xr.Dataset:
        if self._worker_ds is None:
            self._worker_ds = self._select_time_range(
                xr.open_zarr(str(self.data_path), consolidated=True)
            )
        return self._worker_ds

    def _select_time_range(self, ds: xr.Dataset) -> xr.Dataset:
        if self.config.start_time is None and self.config.end_time is None:
            return ds
        return ds.sel(time=slice(self.config.start_time, self.config.end_time))

    @property
    def channel_names(self) -> tuple[str, ...]:
        """Return the exact flattened channel order used by every sample."""
        return self.config.channel_names

    def close(self) -> None:
        """Close this process' lazy xarray/Zarr handle."""
        worker_dataset = getattr(self, "_worker_ds", None)
        if worker_dataset is not None:
            worker_dataset.close()
            self._worker_ds = None

    def __del__(self) -> None:
        self.close()

    def __len__(self) -> int:
        return self.sample_count

    def _read_timestep_channels(self, ds: xr.Dataset, t_idx: int) -> np.ndarray:
        """Extract all 2D and 3D variables at timestep t_idx as [C, H, W]."""
        h, w = self.spatial_shape
        if self._state_layout:
            return np.asarray(
                ds["state"].isel(
                    time=t_idx,
                    latitude=slice(0, h),
                    longitude=slice(0, w),
                ),
                dtype=np.float32,
            )
        channel_slices: list[np.ndarray] = []

        # 1. Surface variables: [1, H, W]
        for var in self.config.surface_variables:
            arr = np.asarray(ds[var].isel(time=t_idx)[:h, :w], dtype=np.float32)
            channel_slices.append(arr[None, ...])

        # 2. Upper-air variables: [len(levels), H, W]. WeatherBench/Zarr
        # stores the levels of one variable in the same chunk; loading them
        # together avoids repeatedly fetching and decompressing that chunk.
        for var, levels in self.config.level_selections:
            arr = np.asarray(
                ds[var]
                .sel(level=list(levels))
                .isel(
                    time=t_idx,
                    latitude=slice(0, h),
                    longitude=slice(0, w),
                ),
                dtype=np.float32,
            )
            channel_slices.append(arr)

        return np.concatenate(channel_slices, axis=0)  # [C, H, W]

    def __getitem__(self, index: int) -> AtmosphereSample:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(
                f"Index {index} out of bounds for dataset of length {len(self)}"
            )

        ds = self._get_dataset()
        step = self.config.time_step
        base_t = self.config.history * step + index

        # Extract history inputs [t - history * step, ..., t]
        input_timesteps: list[np.ndarray] = []
        for h_step in range(self.config.history, -1, -1):
            t = base_t - h_step * step
            data = self._read_timestep_channels(ds, t)
            data = self.normalizer.normalize(data)
            input_timesteps.append(data)

        # Concatenate history along channel dimension: [(history + 1) * C, H, W]
        model_input = np.concatenate(input_timesteps, axis=0)

        # Add Gaussian noise during training if requested
        if self.training and self.config.add_noise and self.config.noise_std > 0:
            noise = np.random.normal(
                0.0, self.config.noise_std, size=model_input.shape
            ).astype(np.float32)
            model_input = model_input + noise

        # Extract rollout targets [t + 1 * step, ..., t + rollout * step]
        target_timesteps: list[np.ndarray] = []
        for r_step in range(1, self.config.rollout_steps + 1):
            t = base_t + r_step * step
            target_data = self._read_timestep_channels(ds, t)
            target_data = self.normalizer.normalize(target_data)
            target_timesteps.append(target_data)

        # Target tensor of shape [rollout_steps, C, H, W]
        model_target = np.stack(target_timesteps, axis=0)

        return {
            "input": torch.from_numpy(
                np.ascontiguousarray(model_input, dtype=np.float32)
            ),
            "target": torch.from_numpy(
                np.ascontiguousarray(model_target, dtype=np.float32)
            ),
            "time_index": torch.tensor(base_t, dtype=torch.long),
        }


def create_data_loader(
    dataset: Dataset[AtmosphereSample],
    *,
    batch_size: int = 1,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    prefetch_factor: int = 2,
    persistent_workers: bool = False,
    distributed: bool = False,
    drop_last: bool = False,
    generator: torch.Generator | None = None,
) -> DataLoader[AtmosphereSample]:
    """Create a worker-safe loader with optional asynchronous prefetching."""
    if num_workers < 0:
        raise ValueError("num_workers cannot be negative")
    if prefetch_factor < 1:
        raise ValueError("prefetch_factor must be positive")
    sampler = DistributedSampler(dataset, shuffle=shuffle) if distributed else None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(shuffle if sampler is None else False),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        drop_last=drop_last,
        # PyTorch only accepts prefetching/persistence with worker processes.
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=persistent_workers and num_workers > 0,
        generator=generator,
    )
