"""Inspect remote WeatherBench2 ERA5 store and verify local data slice."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, cast
import numpy as np
import xarray as xr

os.environ["GRPC_ENABLE_FORK_SUPPORT"] = "0"

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from configs.download_data_config import (
    DEFAULT_ZARR_URL,
    WeatherBenchDownloadConfig,
)
from data.download_data import build_subset


def inspect_remote_store(zarr_url: str = DEFAULT_ZARR_URL) -> None:
    """Print metadata for remote WeatherBench2 zarr store."""
    print(f"Connecting to {zarr_url}")
    ds = xr.open_zarr(zarr_url, storage_options={"token": "anon"})
    try:
        lat, lon = ds.latitude.values, ds.longitude.values
        levels = ds.level.values if "level" in ds else []
        print(f"Grid: {len(lat)}x{len(lon)} (lat: [{lat[0]:.2f}, {lat[-1]:.2f}], lon: [{lon[0]:.2f}, {lon[-1]:.2f}])")
        print(f"Levels ({len(levels)}): {list(levels)}")
        print(f"Timesteps: {len(ds.time)} ({str(ds.time.values[0])[:10]} to {str(ds.time.values[-1])[:10]})")
        print(f"Variables ({len(ds.data_vars)}): {sorted(str(k) for k in ds.data_vars)}")
    finally:
        ds.close()


def download_sample(
    output_path: str = "data/dataset/era5_sample_0p5_26ch.zarr",
    sample_date: str = "2018-01-01",
) -> None:
    """Download a one-day production-format 26-channel slice."""
    config = WeatherBenchDownloadConfig(
        start_date=sample_date,
        end_date=sample_date,
        output_zarr_path=output_path,
    )
    subset = build_subset(config)

    out = Path(config.output_zarr_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing sample ({config.channel_count} channels, {subset.sizes['time']} timesteps) to {out}")
    # xarray accepts filesystem paths here, although some releases expose a
    # narrower StoreLike annotation that omits str.
    try:
        subset.to_zarr(cast(Any, str(out)), mode="w", consolidated=True)
    finally:
        subset.close()
    print("Download completed.")


def inspect_local_zarr(zarr_path: str = "data/dataset/era5_sample_0p5_26ch.zarr") -> None:
    """Compute and print basic statistics for a local zarr dataset."""
    path = Path(zarr_path)
    if not path.exists():
        path = project_root / zarr_path
    if not path.exists():
        print(f"Path does not exist: {zarr_path}")
        return

    ds = xr.open_zarr(str(path), consolidated=True)
    try:
        total_bytes = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
        print(f"Loaded: {zarr_path} ({total_bytes / (1024 * 1024):.2f} MB)")
        print(f"Timesteps: {len(ds.time)} -> {[str(t)[:16] for t in ds.time.values]}")

        print(f"{'Variable':<32} | {'Min':>10} | {'Max':>10} | {'Mean':>10} | {'Std':>10} | {'NaNs':>5}")
        print("-" * 85)

        for name in ds.channel.values:
            arr = ds.state.sel(channel=name).values
            print(f"{str(name):<32} | {np.nanmin(arr):10.2f} | {np.nanmax(arr):10.2f} | {np.nanmean(arr):10.2f} | {np.nanstd(arr):10.2f} | {int(np.isnan(arr).sum()):>5}")
    finally:
        ds.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="WeatherBench2 data exploration utility.")
    parser.add_argument("--remote", action="store_true", help="Inspect remote store metadata")
    parser.add_argument("--download-sample", action="store_true", help="Download 1-day sample slice")
    parser.add_argument("--local-path", type=str, default="data/dataset/era5_sample_0p5_26ch.zarr", help="Inspect local zarr store")
    args = parser.parse_args()

    if args.remote:
        inspect_remote_store()
    elif args.download_sample:
        download_sample(args.local_path)
        inspect_local_zarr(args.local_path)
    else:
        inspect_local_zarr(args.local_path)
