"""Compute train-only channel z-score statistics from an ERA5 Zarr store."""

from __future__ import annotations

import argparse
import json
import hashlib
import sys
import os
import tempfile
from collections import deque
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT_ROOT), str(PROJECT_ROOT / "src")]

import numpy as np
import xarray as xr

from configs.pipeline_config import DataPaths
from configs.download_data_config import channel_names
from neural_atmosphere_operator.data.normalization import latitude_cell_weights
from neural_atmosphere_operator.pipeline.runtime import statistics_signature


def spherical_mean_and_std(
    field: xr.DataArray,
    latitude_weights: xr.DataArray,
) -> tuple[float, float]:
    """Compute population statistics with exact latitude-cell area weights."""
    # The public helper only needs state moments; reuse a bounded two-pass
    # reduction even for a single time slice.
    dimensions = ("time", "latitude", "longitude")
    mean = (
        field.astype("float64")
        .weighted(latitude_weights)
        .mean(dim=dimensions, skipna=False)
        .compute()
    )
    variance = (
        ((field.astype("float64") - mean) ** 2)
        .weighted(latitude_weights)
        .mean(dim=dimensions, skipna=False)
        .compute()
    )
    return float(mean), float(np.sqrt(variance))


def positional_time_difference(field: xr.DataArray, time_step: int) -> xr.DataArray:
    """Subtract samples by position without xarray timestamp re-alignment."""
    count = field.sizes["time"] - time_step
    coordinate = np.arange(count, dtype=np.int64)
    later = field.isel(time=slice(time_step, None)).assign_coords(time=coordinate)
    earlier = field.isel(time=slice(None, -time_step)).assign_coords(time=coordinate)
    return later - earlier


def spherical_channel_statistics(
    state: xr.DataArray,
    latitude_weights: xr.DataArray,
    time_step: int,
) -> xr.Dataset:
    """Stream joint-channel float64 Chan moments with O(stride * C * H * W) RAM.

    Each time slice is materialized separately. No global Dask reduction keeps
    chunks alive across the entire corpus. Delta buffers preserve positional
    stride; climatology and both moments share each decompression.
    """
    required = {"time", "channel", "latitude", "longitude"}
    if set(state.dims) != required:
        raise ValueError("State must have time/channel/latitude/longitude dimensions")
    state = state.transpose("time", "channel", "latitude", "longitude")
    if not 1 <= time_step < state.sizes["time"]:
        raise ValueError("time_step requires at least one temporal difference")
    weights = np.asarray(latitude_weights.values, dtype=np.float64)
    weights = weights / weights.sum() / state.sizes["longitude"]
    weight = weights[None, :, None]
    channels = state.sizes["channel"]
    means = [np.zeros(channels), np.zeros(channels)]
    m2 = [np.zeros(channels), np.zeros(channels)]
    counts = [0, 0]
    climate = np.zeros(tuple(state.shape[1:]), dtype=np.float64)
    previous = deque()
    payload_hash = hashlib.sha256()

    def update(values, branch):
        mean = np.sum(values * weight, axis=(1, 2))
        variance = np.sum((values - mean[:, None, None]) ** 2 * weight, axis=(1, 2))
        counts[branch] += 1
        difference = mean - means[branch]
        means[branch] += difference / counts[branch]
        m2[branch] += variance + difference**2 * (counts[branch] - 1) / counts[branch]

    for index in range(state.sizes["time"]):
        values = np.asarray(state.isel(time=index).values, dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(
                f"Training data contains non-finite values at time index {index}"
            )
        payload_hash.update(values.astype("<f4").tobytes())
        update(values, 0)
        climate += (values - climate) / (index + 1)
        if len(previous) == time_step:
            update(values - previous.popleft(), 1)
        previous.append(values)
    coords = {name: state[name] for name in ("channel", "latitude", "longitude")}
    return xr.Dataset(
        {
            "mean": ("channel", means[0]),
            "variance": ("channel", m2[0] / counts[0]),
            "time_diff_mean": ("channel", means[1]),
            "time_diff_variance": ("channel", m2[1] / counts[1]),
            "time_mean": (("channel", "latitude", "longitude"), climate),
        },
        coords=coords,
        attrs={"training_state_float32_sha256": payload_hash.hexdigest()},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = DataPaths()
    parser.add_argument("--train-data", type=Path, default=defaults.train)
    parser.add_argument("--train-start", default=defaults.split_time_range("train")[0])
    parser.add_argument("--train-end", default=defaults.split_time_range("train")[1])
    parser.add_argument("--output-dir", type=Path, default=defaults.means.parent)
    parser.add_argument(
        "--time-step",
        type=int,
        default=1,
        help="Temporal stride used by the forecast target and delta statistics",
    )
    args = parser.parse_args()
    if args.time_step < 1:
        raise ValueError("time-step must be at least 1")
    train_path = args.train_data.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(
            f"Statistics bundles are immutable; choose a new output directory: {output_dir}"
        )
    ds = xr.open_zarr(str(train_path), consolidated=True).sel(
        time=slice(args.train_start, args.train_end)
    )
    means: list[float] = []
    stds: list[float] = []
    time_diff_means: list[float] = []
    time_diff_stds: list[float] = []
    time_means: list[np.ndarray] = []
    names: list[str] = []
    units: list[str] = []
    if ds.sizes["time"] <= args.time_step:
        ds.close()
        raise ValueError("Training split is empty or too short for time-step")
    first_time = str(ds.time.values[0])
    last_time = str(ds.time.values[-1])
    timesteps = int(ds.sizes["time"])
    if timesteps <= args.time_step:
        ds.close()
        raise ValueError("Training split is too short for the requested time-step")
    time_values = np.asarray(ds.time.values)
    if not np.issubdtype(time_values.dtype, np.datetime64):
        ds.close()
        raise ValueError("ERA5 time coordinate must use datetime64 values")
    time_nanoseconds = time_values.astype("datetime64[ns]").astype(np.int64)
    time_differences = np.diff(time_nanoseconds)
    if not np.all(time_differences > 0) or not np.all(
        time_differences == time_differences[0]
    ):
        ds.close()
        raise ValueError("Training timestamps must be strictly increasing and regular")
    cadence_hours = float(time_differences[0] / 3_600_000_000_000)
    latitudes = np.asarray(ds.latitude.values, dtype=np.float64)
    longitudes = np.asarray(ds.longitude.values, dtype=np.float64)
    area_weights = np.asarray(latitude_cell_weights(latitudes), dtype=np.float64)
    latitude_weights = xr.DataArray(
        area_weights,
        dims=("latitude",),
        coords={"latitude": ds.latitude},
    )
    try:
        if "state" not in ds:
            raise ValueError(
                "Dataset must contain the canonical flattened 71-channel state"
            )
        names = [str(name) for name in ds.channel.values]
        if tuple(names) != channel_names():
            raise ValueError("Stored state channels do not match the 71-channel contract")
        if "channel_units" not in ds.coords:
            raise ValueError(
                "Flattened state is missing per-channel units; regenerate with the current downloader"
            )
        units = [str(unit) for unit in ds.channel_units.values]
        state = ds["state"]
        if len(units) != len(names) or any(not unit.strip() for unit in units):
            raise ValueError("channel_units must be non-empty and align with channel")
        computed = spherical_channel_statistics(state, latitude_weights, args.time_step)
        means = np.asarray(computed["mean"]).tolist()
        stds = np.sqrt(np.asarray(computed["variance"])).tolist()
        time_diff_means = np.asarray(computed["time_diff_mean"]).tolist()
        time_diff_stds = np.sqrt(np.asarray(computed["time_diff_variance"])).tolist()
        time_means = list(np.asarray(computed["time_mean"]))
        payload_sha256 = computed.attrs["training_state_float32_sha256"]
        for index, name in enumerate(names):
            print(
                f"{name}: mean={means[index]:.7g} std={stds[index]:.7g} "
                f"delta_mean={time_diff_means[index]:.7g} delta_std={time_diff_stds[index]:.7g}"
            )
    finally:
        ds.close()
    means_array = np.asarray(means, dtype=np.float32)
    stds_array = np.asarray(stds, dtype=np.float32)
    time_diff_means_array = np.asarray(time_diff_means, dtype=np.float32)
    time_diff_stds_array = np.asarray(time_diff_stds, dtype=np.float32)
    time_means_array = np.stack(time_means, axis=0).astype(np.float32)
    arrays = (
        means_array,
        stds_array,
        time_diff_means_array,
        time_diff_stds_array,
        time_means_array,
    )
    if not all(np.all(np.isfinite(array)) for array in arrays):
        raise ValueError("Training data contains non-finite values")
    if np.any(stds_array <= 0) or np.any(time_diff_stds_array <= 0):
        raise ValueError("Every training channel must have positive variance")
    destination = output_dir
    if destination.exists():
        raise FileExistsError(
            f"Statistics bundles are immutable; choose a new output directory: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    output_dir = Path(
        tempfile.mkdtemp(prefix=destination.name + ".part-", dir=destination.parent)
    )
    np.save(output_dir / "means.npy", means_array)
    np.save(output_dir / "stds.npy", stds_array)
    np.save(output_dir / "time_means.npy", time_means_array)
    np.save(
        output_dir / f"time_diff_means_dt{args.time_step}.npy",
        time_diff_means_array,
    )
    np.save(
        output_dir / f"time_diff_stds_dt{args.time_step}.npy",
        time_diff_stds_array,
    )
    (output_dir / "channels.txt").write_text("\n".join(names) + "\n", encoding="utf-8")
    (output_dir / "stats.json").write_text(
        json.dumps(
            {
                "source": str(train_path),
                "training_state_float32_sha256": payload_sha256,
                "latitude": latitudes.astype(np.float32).tolist(),
                "longitude": longitudes.astype(np.float32).tolist(),
                "artifacts": {
                    name: statistics_signature(output_dir / filename)["sha256"]
                    for name, filename in {
                        "means": "means.npy",
                        "stds": "stds.npy",
                        "climatology": "time_means.npy",
                        "time_diff_stds": f"time_diff_stds_dt{args.time_step}.npy",
                        "time_diff_means": f"time_diff_means_dt{args.time_step}.npy",
                    }.items()
                },
                "first_time": first_time,
                "last_time": last_time,
                "timesteps": timesteps,
                "channels": names,
                "channel_units": units,
                "spatial_weighting": "spherical_latitude_cell_area",
                "statistics_version": 4,
                "computation": "streaming_joint_channel_float64_chan",
                "climatology": "training_long_term_mean_by_grid_cell",
                "time_difference_step": args.time_step,
                "cadence_hours": cadence_hours,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    os.rename(output_dir, destination)
    print(f"Statistics written to {destination}")


if __name__ == "__main__":
    main()
