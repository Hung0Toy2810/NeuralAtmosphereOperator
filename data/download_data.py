"""Download the fixed 71-channel ERA5 state from public WeatherBench2 Zarr.

Requires: pip install xarray zarr gcsfs dask

Usage:
    python data/download_data.py

The channel and pressure-level contract is immutable. Configuration controls
the date range, batching and output path; field values are conservatively
regridded from 0.25 to 0.5 degrees.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import json
import sys
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Mapping, cast
from uuid import uuid4

import numpy as np
import xarray as xr
import zarr

# Ensure project root is in sys.path when running script directly
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from configs.download_data_config import WeatherBenchDownloadConfig
from configs.download_data_config import (
    AVAILABLE_PRESSURE_LEVELS,
    PRESSURE_VARIABLES,
    SURFACE_VARIABLES,
)

REGRIDDING_VERSION = 2
REGRIDDING_METHOD = "first_order_conservative_area_overlap_aligned_periodic"
DOWNLOAD_CONTRACT_VERSION = 2


def download_contract(config: WeatherBenchDownloadConfig) -> dict[str, Any]:
    """Return the canonical data contract that must remain fixed on resume."""
    return {
        "schema_version": DOWNLOAD_CONTRACT_VERSION,
        "source": config.zarr_url,
        "surface_variables": list(SURFACE_VARIABLES),
        "pressure_variables": list(PRESSURE_VARIABLES),
        "pressure_levels": list(AVAILABLE_PRESSURE_LEVELS),
        "channel_names": list(config.channel_names),
        "start_date": config.start_date,
        "end_date": config.end_date,
        "time_stride_hours": config.time_stride_hours,
        "target_resolution_degrees": config.target_resolution_degrees,
        "regridding_version": REGRIDDING_VERSION,
        "regridding_method": REGRIDDING_METHOD,
    }


def contract_json(contract: dict[str, Any]) -> str:
    """Serialize a download contract deterministically for storage and hashing."""
    return json.dumps(contract, sort_keys=True, separators=(",", ":"))


def contract_fingerprint(contract: dict[str, Any]) -> str:
    return hashlib.sha256(contract_json(contract).encode("utf-8")).hexdigest()


def validate_resume_contract(
    progress: Mapping[str, Any],
    store_attrs: Mapping[str, Any],
    config: WeatherBenchDownloadConfig,
) -> None:
    """Reject partial stores created for any different semantic data contract."""
    expected = download_contract(config)
    fingerprint = contract_fingerprint(expected)
    if (
        progress.get("download_contract") != expected
        or progress.get("download_contract_sha256") != fingerprint
        or store_attrs.get("download_contract_json") != contract_json(expected)
        or store_attrs.get("download_contract_sha256") != fingerprint
    ):
        raise RuntimeError(
            "Partial store download contract does not match the requested "
            "source/date/cadence/channel configuration. Preserve it for "
            "inspection and choose a new output path."
        )


def _channel_metadata(subset: xr.Dataset) -> dict[str, tuple[str, np.ndarray]]:
    """Build metadata coordinates aligned exactly with the flattened channels."""
    units: list[str] = []
    long_names: list[str] = []
    source_variables: list[str] = []
    pressure_levels: list[float] = []
    selections = [(name, None) for name in SURFACE_VARIABLES] + [
        (variable, level)
        for variable in PRESSURE_VARIABLES
        for level in AVAILABLE_PRESSURE_LEVELS
    ]
    for variable, level in selections:
        attrs = subset[variable].attrs
        unit = str(attrs.get("units", "")).strip()
        # WB2 relative humidity is a dimensionless fraction but some archive
        # revisions omit its CF units attribute.
        if not unit and variable == "relative_humidity":
            unit = "1"
        units.append(unit)
        long_names.append(str(attrs.get("long_name", variable)))
        source_variables.append(variable)
        pressure_levels.append(float("nan") if level is None else float(level))
    return {
        "channel_units": ("channel", np.asarray(units, dtype=str)),
        "channel_long_name": ("channel", np.asarray(long_names, dtype=str)),
        "channel_source_variable": (
            "channel",
            np.asarray(source_variables, dtype=str),
        ),
        "channel_pressure_level_hpa": (
            "channel",
            np.asarray(pressure_levels, dtype=np.float32),
        ),
    }


def _cell_bounds(centres: np.ndarray) -> np.ndarray:
    """Return latitude-cell edges, clipped to the two poles."""
    ascending = np.asarray(centres, dtype=np.float64)
    if ascending.ndim != 1 or len(ascending) < 2 or not np.all(np.diff(ascending) > 0):
        raise ValueError("latitude centres must be a strictly ascending vector")
    bounds = np.empty(len(ascending) + 1, dtype=np.float64)
    bounds[1:-1] = 0.5 * (ascending[:-1] + ascending[1:])
    bounds[0], bounds[-1] = -90.0, 90.0
    return bounds


def _latitude_overlap_weights(
    source_latitudes: np.ndarray, target_latitudes: np.ndarray
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    """Build exact first-order conservative latitude overlap weights."""
    source = np.asarray(source_latitudes, dtype=np.float64)
    target = np.asarray(target_latitudes, dtype=np.float64)
    source_order = np.argsort(source)
    target_order = np.argsort(target)
    source_sorted, target_sorted = source[source_order], target[target_order]
    source_bounds = _cell_bounds(source_sorted)
    target_bounds = _cell_bounds(target_sorted)
    rows: list[tuple[np.ndarray, np.ndarray] | None] = [None] * len(target)
    for sorted_target_index in range(len(target_sorted)):
        lower, upper = target_bounds[sorted_target_index : sorted_target_index + 2]
        overlap_lower = np.maximum(source_bounds[:-1], lower)
        overlap_upper = np.minimum(source_bounds[1:], upper)
        valid = overlap_upper > overlap_lower
        sorted_indices = np.flatnonzero(valid)
        weights = np.sin(np.deg2rad(overlap_upper[valid])) - np.sin(
            np.deg2rad(overlap_lower[valid])
        )
        weights /= weights.sum()
        original_target_index = int(target_order[sorted_target_index])
        rows[original_target_index] = (source_order[sorted_indices], weights)
    return tuple(row for row in rows if row is not None)


def _conservative_half_degree(
    values: np.ndarray,
    latitude_rows: tuple[tuple[np.ndarray, np.ndarray], ...],
) -> np.ndarray:
    """Area-average a 0.25-degree field to the aligned 0.5-degree grid.

    Both grids use cell centres at integer multiples of their resolution.  A
    0.5-degree longitude cell therefore contains all of the source cell at the
    same centre and half of each neighbouring 0.25-degree source cell.  The
    longitude axis is periodic, so the cell centred at zero also overlaps the
    source cell centred at 359.75 degrees.
    """
    if values.shape[-2:] != (721, 1440):
        raise ValueError(f"expected a 721x1440 source grid, got {values.shape[-2:]}")
    centred = values[..., ::2]
    left = np.roll(values, shift=1, axis=-1)[..., ::2]
    right = np.roll(values, shift=-1, axis=-1)[..., ::2]
    longitude_averaged = 0.25 * left + 0.5 * centred + 0.25 * right
    output = np.empty((*values.shape[:-2], 361, 720), dtype=np.float32)
    for target_index, (indices, weights) in enumerate(latitude_rows):
        output[..., target_index, :] = np.sum(
            longitude_averaged[..., indices, :]
            * weights.reshape(*([1] * (values.ndim - 2)), -1, 1),
            axis=-2,
        )
    return output


def _regrid_data_array(field: xr.DataArray) -> xr.DataArray:
    source_latitudes = np.asarray(field.latitude.values, dtype=np.float64)
    source_longitudes = np.asarray(field.longitude.values, dtype=np.float64)
    if len(source_latitudes) != 721 or len(source_longitudes) != 1440:
        raise ValueError("the conservative 0.5-degree path requires a 721x1440 source")
    target_latitudes = np.linspace(
        float(source_latitudes[0]), float(source_latitudes[-1]), 361, dtype=np.float32
    )
    target_longitudes = source_longitudes[::2].astype(np.float32)
    latitude_rows = _latitude_overlap_weights(source_latitudes, target_latitudes)
    result = xr.apply_ufunc(
        _conservative_half_degree,
        field,
        input_core_dims=[["latitude", "longitude"]],
        output_core_dims=[["target_latitude", "target_longitude"]],
        kwargs={"latitude_rows": latitude_rows},
        dask="parallelized",
        output_dtypes=[np.float32],
        dask_gufunc_kwargs={
            "output_sizes": {"target_latitude": 361, "target_longitude": 720},
            "allow_rechunk": True,
        },
    )
    return result.rename(
        target_latitude="latitude", target_longitude="longitude"
    ).assign_coords(latitude=target_latitudes, longitude=target_longitudes)


def validate_source_channels(source: xr.Dataset) -> None:
    """Check the complete fixed 71-channel state using only source metadata."""
    surface_dims = {"time", "latitude", "longitude"}
    selections = [(name, surface_dims) for name in SURFACE_VARIABLES] + [
        (name, surface_dims | {"level"}) for name in PRESSURE_VARIABLES
    ]
    missing = [name for name, _ in selections if name not in source.data_vars]
    if missing:
        raise ValueError(f"Source is missing requested variables: {missing}")
    for name, dimensions in selections:
        if set(source[name].dims) != dimensions:
            raise ValueError(f"Unexpected source dimensions for {name}: {source[name].dims}")
    if "level" not in source.coords:
        raise ValueError("Source is missing the pressure-level coordinate")
    available = set(source.level.values.tolist())
    missing_levels = set(AVAILABLE_PRESSURE_LEVELS) - available
    if missing_levels:
        raise ValueError(f"Source is missing pressure levels: {sorted(missing_levels)}")


def build_state(
    config: WeatherBenchDownloadConfig, source: xr.Dataset | None = None
) -> xr.Dataset:
    """Build a lazy, conservatively regridded ``state[time,channel,lat,lon]``."""
    if source is None:
        print(f"Opening {config.zarr_url}")
        full = xr.open_zarr(
            config.zarr_url, storage_options={"token": "anon"}, consolidated=True
        )
    else:
        full = source

    try:
        validate_source_channels(full)
    except Exception:
        if source is None:
            full.close()
        raise

    variables = list(SURFACE_VARIABLES + PRESSURE_VARIABLES)
    step = config.time_stride_hours // 6  # dataset is natively 6-hourly
    subset = full[variables].sel(time=slice(config.start_date, config.end_date))
    subset = subset.isel(time=slice(None, None, step))

    channels: list[xr.DataArray] = []
    for variable in SURFACE_VARIABLES:
        channels.append(_regrid_data_array(subset[variable]))
    for variable in PRESSURE_VARIABLES:
        regridded = _regrid_data_array(
            subset[variable].sel(level=list(AVAILABLE_PRESSURE_LEVELS))
        )
        channels.extend(
            regridded.sel(level=level, drop=True)
            for level in AVAILABLE_PRESSURE_LEVELS
        )
    state = xr.concat(
        channels, dim=xr.IndexVariable("channel", list(config.channel_names))
    )
    state = state.transpose("time", "channel", "latitude", "longitude")
    # xr.concat otherwise copies the first variable's attrs onto the complete
    # state tensor, incorrectly labelling every channel as (for example) m/s.
    state.attrs = {
        "long_name": "flattened multivariate atmospheric state",
        "channel_metadata": "coordinates aligned with the channel dimension",
    }
    state = state.assign_coords(_channel_metadata(subset))
    state = state.chunk(
        {"time": 1, "channel": config.channel_count, "latitude": 361, "longitude": 720}
    )
    result = state.to_dataset(name="state")
    result.attrs.update(
        source=config.zarr_url,
        regridding=REGRIDDING_METHOD,
        regridding_version=REGRIDDING_VERSION,
        temporal_cadence_hours=config.time_stride_hours,
        channel_metadata_version=1,
    )

    print(
        f"Fixed {config.channel_count}-channel state, {result.sizes['time']} timesteps, "
        f"grid={result.sizes['latitude']}x{result.sizes['longitude']}"
    )
    return result


def _truncate_partial_store(path: Path, timesteps: int) -> None:
    """Discard an interrupted append beyond the last committed batch."""
    # zarr 2.x exposes these runtime methods, but some bundled type stubs model
    # Group.arrays/Array.resize as attributes. Keep that incompatibility at
    # this library boundary rather than suppressing checking in the pipeline.
    group = zarr.open_group(cast(Any, str(path)), mode="a")
    for _, untyped_array in cast(Any, group).arrays():
        array = cast(Any, untyped_array)
        dimensions = tuple(array.attrs.get("_ARRAY_DIMENSIONS", ()))
        if "time" not in dimensions:
            continue
        axis = dimensions.index("time")
        shape = list(array.shape)
        shape[axis] = timesteps
        array.resize(tuple(shape))


def _write_progress(
    path: Path,
    *,
    next_date: str,
    timesteps: int,
    contract: dict[str, Any],
    fingerprint: str,
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            {
                "next_date": next_date,
                "timesteps": timesteps,
                "download_contract": contract,
                "download_contract_sha256": fingerprint,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _validate_timestamps(
    dataset: xr.Dataset, config: WeatherBenchDownloadConfig
) -> None:
    """Require every requested forecast time, in order and without duplicates."""
    expected = np.arange(
        np.datetime64(config.start_date, "ns"),
        np.datetime64(config.end_date, "ns") + np.timedelta64(1, "D"),
        np.timedelta64(config.time_stride_hours, "h"),
    )
    if "time" not in dataset.coords or not np.array_equal(
        dataset.time.values, expected
    ):
        raise ValueError(
            "Dataset timestamps do not match the complete requested interval/cadence"
        )


def main(config: WeatherBenchDownloadConfig | None = None) -> None:
    config = config or WeatherBenchDownloadConfig()
    output_path = Path(config.output_zarr_path)
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = output_path.with_name(output_path.name + ".partial")
    progress_path = output_path.with_name(output_path.name + ".progress.json")
    full_contract = download_contract(config)
    full_contract_json = contract_json(full_contract)
    full_contract_fingerprint = contract_fingerprint(full_contract)
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite completed dataset: {output_path}")

    cursor = datetime.fromisoformat(config.start_date)
    committed_steps = 0
    if partial_path.exists() and not progress_path.is_file():
        raise RuntimeError(
            f"Partial store has no progress record; inspect it before removing: {partial_path}"
        )
    if progress_path.is_file():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        # Validate the journal before inspecting a possibly incomplete first write.
        validate_resume_contract(
            progress,
            {
                "download_contract_json": full_contract_json,
                "download_contract_sha256": full_contract_fingerprint,
            },
            config,
        )
        cursor = datetime.fromisoformat(str(progress["next_date"]))
        committed_steps = int(progress["timesteps"])
        expected_steps = len(
            np.arange(
                np.datetime64(config.start_date),
                np.datetime64(cursor),
                np.timedelta64(config.time_stride_hours, "h"),
            )
        )
        if (
            cursor < datetime.fromisoformat(config.start_date)
            or cursor > datetime.fromisoformat(config.end_date) + timedelta(days=1)
            or committed_steps != expected_steps
        ):
            raise RuntimeError("Invalid download progress interval or timestep count")
        if committed_steps == 0:
            if partial_path.exists():
                # Preserve incomplete metadata/chunks; restart the uncommitted batch.
                os.replace(
                    partial_path,
                    partial_path.with_name(
                        partial_path.name + ".interrupted-" + uuid4().hex
                    ),
                )
        else:
            partial_group = zarr.open_group(cast(Any, str(partial_path)), mode="r")
            validate_resume_contract(progress, partial_group.attrs, config)
            _truncate_partial_store(partial_path, committed_steps)
        print(f"Resuming at {cursor.date()} after {committed_steps:,} timesteps")

    if committed_steps == 0:
        _write_progress(
            progress_path,
            next_date=config.start_date,
            timesteps=0,
            contract=full_contract,
            fingerprint=full_contract_fingerprint,
        )

    final_day = datetime.fromisoformat(config.end_date)
    print(f"Opening {config.zarr_url}")
    source = xr.open_zarr(
        config.zarr_url, storage_options={"token": "anon"}, consolidated=True
    )
    try:
        while cursor <= final_day:
            batch_end = min(
                cursor + timedelta(days=config.download_batch_days - 1), final_day
            )
            batch_config = replace(
                config,
                start_date=cursor.date().isoformat(),
                end_date=batch_end.date().isoformat(),
            )
            batch_state = build_state(batch_config, source)
            _validate_timestamps(batch_state, batch_config)
            # Batch-local date bounds are an implementation detail. Persist the
            # complete requested contract on every append target.
            batch_state.attrs.update(
                download_contract_json=full_contract_json,
                download_contract_sha256=full_contract_fingerprint,
            )
            print(f"Writing {cursor.date()}..{batch_end.date()} to {partial_path}")
            if committed_steps == 0:
                batch_state.to_zarr(
                    cast(Any, str(partial_path)), mode="w", consolidated=False
                )
            else:
                batch_state.to_zarr(
                    cast(Any, str(partial_path)),
                    mode="a",
                    append_dim="time",
                    consolidated=False,
                )
            committed_steps += int(batch_state.sizes["time"])
            cursor = batch_end + timedelta(days=1)
            _write_progress(
                progress_path,
                next_date=cursor.date().isoformat(),
                timesteps=committed_steps,
                contract=full_contract,
                fingerprint=full_contract_fingerprint,
            )
    finally:
        source.close()
    zarr.consolidate_metadata(cast(Any, str(partial_path)))
    with xr.open_zarr(str(partial_path), consolidated=True) as completed:
        _validate_timestamps(completed, config)
    os.replace(partial_path, output_path)
    progress_path.unlink(missing_ok=True)
    print(f"Done: {output_path}")


def parse_args() -> argparse.Namespace:
    defaults = WeatherBenchDownloadConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", default=defaults.start_date)
    parser.add_argument("--end-date", default=defaults.end_date)
    parser.add_argument("--output", default=defaults.output_zarr_path)
    parser.add_argument("--batch-days", type=int, default=defaults.download_batch_days)
    parser.add_argument(
        "--confirm-full-download",
        action="store_true",
        help="Required safety acknowledgement before starting the large download",
    )
    args = parser.parse_args()
    if not args.confirm_full_download:
        parser.error(
            "refusing to start a large download without --confirm-full-download"
        )
    return args


if __name__ == "__main__":
    cli = parse_args()
    main(
        replace(
            WeatherBenchDownloadConfig(),
            start_date=cli.start_date,
            end_date=cli.end_date,
            output_zarr_path=cli.output,
            download_batch_days=cli.batch_days,
        )
    )
