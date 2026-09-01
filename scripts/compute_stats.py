"""Compute train-only channel z-score statistics from an ERA5 Zarr store."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT_ROOT), str(PROJECT_ROOT / "src")]

import numpy as np
import xarray as xr

from configs.pipeline_config import DataPaths
from neural_atmosphere_operator.data.loader import (
    DEFAULT_LEVEL_SPECS,
    DEFAULT_SURFACE_VARS,
)
from neural_atmosphere_operator.data.normalization import latitude_cell_weights


def spherical_mean_and_std(
    field: xr.DataArray,
    latitude_weights: xr.DataArray,
) -> tuple[float, float]:
    """Compute population statistics with exact latitude-cell area weights."""
    dimensions = ("time", "latitude", "longitude")
    missing = set(dimensions).difference(field.dims)
    if missing:
        raise ValueError(f"Field {field.name!r} is missing dimensions {sorted(missing)}")
    mean = field.weighted(latitude_weights).mean(dim=dimensions, skipna=False)
    variance = ((field - mean) ** 2).weighted(latitude_weights).mean(
        dim=dimensions,
        skipna=False,
    )
    computed = xr.Dataset({"mean": mean, "variance": variance}).compute()
    return (
        float(computed["mean"].values),
        float(np.sqrt(computed["variance"].values)),
    )


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
    """Build one shared Dask graph for all statistics of a channel tensor.

    The production Zarr store keeps all channels in the same time chunk.  A
    channel-by-channel loop would therefore decompress that same chunk once per
    channel. Computing the vector reductions together lets Dask share each read
    across global, tendency and climatological statistics.
    """
    required = {"time", "channel", "latitude", "longitude"}
    missing = required.difference(state.dims)
    if missing:
        raise ValueError(f"State tensor is missing dimensions {sorted(missing)}")
    reductions = ("time", "latitude", "longitude")
    mean = state.weighted(latitude_weights).mean(dim=reductions, skipna=False)
    variance = ((state - mean) ** 2).weighted(latitude_weights).mean(
        dim=reductions, skipna=False
    )
    delta = positional_time_difference(state, time_step)
    delta_mean = delta.weighted(latitude_weights).mean(
        dim=reductions, skipna=False
    )
    delta_variance = ((delta - delta_mean) ** 2).weighted(latitude_weights).mean(
        dim=reductions, skipna=False
    )
    return xr.Dataset(
        {
            "mean": mean,
            "variance": variance,
            "time_diff_mean": delta_mean,
            "time_diff_variance": delta_variance,
            "time_mean": state.mean(dim="time", skipna=False),
        }
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
    area_weights = np.asarray(latitude_cell_weights(latitudes), dtype=np.float64)
    latitude_weights = xr.DataArray(
        area_weights,
        dims=("latitude",),
        coords={"latitude": ds.latitude},
    )
    try:
        if "state" in ds:
            names = [str(name) for name in ds.channel.values]
            if "channel_units" not in ds.coords:
                raise ValueError(
                    "Flattened state is missing per-channel units; regenerate it "
                    "with the current downloader before computing statistics"
                )
            units = [str(unit) for unit in ds.channel_units.values]
            if len(units) != len(names) or any(not unit.strip() for unit in units):
                raise ValueError("channel_units must be non-empty and align with channel")
            computed = spherical_channel_statistics(
                ds["state"], latitude_weights, args.time_step
            ).compute()
            means.extend(np.asarray(computed["mean"].values).tolist())
            stds.extend(np.sqrt(np.asarray(computed["variance"].values)).tolist())
            time_diff_means.extend(
                np.asarray(computed["time_diff_mean"].values).tolist()
            )
            time_diff_stds.extend(
                np.sqrt(np.asarray(computed["time_diff_variance"].values)).tolist()
            )
            time_means.extend(np.asarray(computed["time_mean"].values))
            for index, name in enumerate(names):
                print(
                    f"{name}: mean={means[index]:.7g} std={stds[index]:.7g} "
                    f"delta_mean={time_diff_means[index]:.7g} "
                    f"delta_std={time_diff_stds[index]:.7g}"
                )
        else:
            selections = [(name, ds[name]) for name in DEFAULT_SURFACE_VARS] + [
                (f"{name}@{level}hPa", ds[name].sel(level=level))
                for name, levels in DEFAULT_LEVEL_SPECS
                for level in levels
            ]
            for name, field in selections:
                mean, std = spherical_mean_and_std(field, latitude_weights)
                delta = positional_time_difference(field, args.time_step)
                delta_mean, delta_std = spherical_mean_and_std(delta, latitude_weights)
                time_mean = np.asarray(
                    field.mean(dim="time", skipna=False).compute().values,
                    dtype=np.float32,
                )
                means.append(mean)
                stds.append(std)
                time_diff_means.append(delta_mean)
                time_diff_stds.append(delta_std)
                time_means.append(time_mean)
                names.append(name)
                unit = str(
                    field.attrs.get(
                        "units", "1" if field.name == "relative_humidity" else ""
                    )
                )
                if not unit.strip():
                    raise ValueError(f"Field {name!r} is missing its physical unit")
                units.append(unit)
                print(
                    f"{name}: mean={mean:.7g} std={std:.7g} "
                    f"delta_mean={delta_mean:.7g} delta_std={delta_std:.7g}"
                )
    finally:
        ds.close()
    means_array = np.asarray(means, dtype=np.float32)
    stds_array = np.asarray(stds, dtype=np.float32)
    time_diff_means_array = np.asarray(time_diff_means, dtype=np.float32)
    time_diff_stds_array = np.asarray(time_diff_stds, dtype=np.float32)
    time_means_array = np.stack(time_means, axis=0)
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
    output_dir.mkdir(parents=True, exist_ok=True)
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
                "first_time": first_time,
                "last_time": last_time,
                "timesteps": timesteps,
                "channels": names,
                "channel_units": units,
                "spatial_weighting": "spherical_latitude_cell_area",
                "statistics_version": 3,
                "computation": "joint_channel_dask_graph",
                "climatology": "training_long_term_mean_by_grid_cell",
                "time_difference_step": args.time_step,
                "cadence_hours": cadence_hours,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Statistics written to {output_dir}")


if __name__ == "__main__":
    main()
