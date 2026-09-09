"""Download and verify a small sample (1-day slice) of WeatherBench2 ERA5 data."""

from __future__ import annotations

import argparse
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

from configs.download_data_config import DEFAULT_SAMPLE_FILENAME, WeatherBenchDownloadConfig
from data.download_data import build_state


def download_sample(
    sample_date: str = "2018-01-01",
    output_path: str = f"data/dataset/{DEFAULT_SAMPLE_FILENAME}",
) -> Path:
    """Download a one-day slice using the configured production contract."""
    config = WeatherBenchDownloadConfig(
        start_date=sample_date,
        end_date=sample_date,
        output_zarr_path=output_path,
    )
    out = Path(config.output_zarr_path)
    if not out.is_absolute():
        out = project_root / out
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite existing sample: {out}")

    state = build_state(config)
    print(
        f"Writing {state.sizes['time']} timesteps ({config.channel_count} channels) to {out}"
    )
    # xarray accepts filesystem paths here, although some releases expose a
    # narrower StoreLike annotation that omits str.
    try:
        state.to_zarr(cast(Any, str(out)), mode="w", consolidated=True)
    finally:
        state.close()
    print("Download finished.")
    return out


def verify_sample(path: Path) -> None:
    """Verify integrity and display summary statistics for the downloaded sample."""
    ds = xr.open_zarr(str(path), consolidated=True)
    try:
        total_bytes = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())

        print(f"Sample path: {path} ({total_bytes / (1024 * 1024):.2f} MB)")
        print(f"Timesteps: {len(ds.time)} -> {[str(t)[:16] for t in ds.time.values]}")

        print(
            f"{'Variable':<32} | {'Min':>10} | {'Max':>10} | {'Mean':>10} | {'Std':>10} | {'NaNs':>5}"
        )
        print("-" * 85)

        for name in ds.channel.values:
            arr = ds.state.sel(channel=name).values
            print(
                f"{str(name):<32} | {np.nanmin(arr):10.2f} | {np.nanmax(arr):10.2f} | {np.nanmean(arr):10.2f} | {np.nanstd(arr):10.2f} | {int(np.isnan(arr).sum()):>5}"
            )
    finally:
        ds.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-date", default="2018-01-01")
    parser.add_argument("--output", default=f"data/dataset/{DEFAULT_SAMPLE_FILENAME}")
    args = parser.parse_args()
    path = download_sample(args.sample_date, args.output)
    verify_sample(path)


if __name__ == "__main__":
    main()
