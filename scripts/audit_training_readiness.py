"""Audit ERA5 storage and temporal sample capacity before training."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
import math
from pathlib import Path
import shutil
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT_ROOT), str(PROJECT_ROOT / "src")]

from configs.download_data_config import WeatherBenchDownloadConfig
from configs.model_config import AtmosphereModelConfig
from configs.pipeline_config import TrainingConfig

GIB = 2**30


def directory_bytes(path: Path) -> int:
    """Return the exact logical byte size of regular files below ``path``."""
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def count_timesteps(start_date: str, end_date: str, stride_hours: int) -> int:
    """Count samples at a regular cadence over inclusive calendar dates."""
    start = datetime.fromisoformat(start_date)
    end_exclusive = datetime.fromisoformat(end_date) + timedelta(days=1)
    seconds = (end_exclusive - start).total_seconds()
    return math.ceil(seconds / (stride_hours * 3600))


def add_years(value: datetime, years: int) -> datetime:
    """Advance a date by whole years, mapping leap day to February 28."""
    try:
        return value.replace(year=value.year + years)
    except ValueError:
        return value.replace(month=2, day=28, year=value.year + years)


def main() -> None:
    data_defaults = WeatherBenchDownloadConfig()
    model_defaults = AtmosphereModelConfig()
    train_defaults = TrainingConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", default=data_defaults.start_date)
    parser.add_argument("--end-date", default=data_defaults.end_date)
    parser.add_argument(
        "--stride-hours", type=int, default=data_defaults.time_stride_hours
    )
    parser.add_argument("--channels", type=int, default=data_defaults.channel_count)
    parser.add_argument("--height", type=int, default=model_defaults.img_size[0])
    parser.add_argument("--width", type=int, default=model_defaults.img_size[1])
    parser.add_argument(
        "--sample-zarr",
        type=Path,
        default=PROJECT_ROOT / "data" / "dataset" / "era5_sample_0p5_26ch.zarr",
    )
    parser.add_argument("--sample-timesteps", type=int, default=4)
    parser.add_argument(
        "--train-years",
        type=int,
        nargs="+",
        default=(4, 8, 12),
        help="Training-window lengths to compare",
    )
    parser.add_argument("--rollout-steps", type=int, default=1)
    parser.add_argument("--history", type=int, default=train_defaults.history)
    parser.add_argument("--time-step", type=int, default=train_defaults.time_step)
    parser.add_argument("--batch-size", type=int, default=train_defaults.batch_size)
    parser.add_argument(
        "--gradient-accumulation",
        type=int,
        default=train_defaults.gradient_accumulation,
    )
    parser.add_argument("--epochs", type=int, default=train_defaults.epochs)
    args = parser.parse_args()
    positive = (
        args.stride_hours,
        args.channels,
        args.height,
        args.width,
        args.sample_timesteps,
        args.rollout_steps,
        args.time_step,
        args.batch_size,
        args.gradient_accumulation,
        args.epochs,
        *args.train_years,
    )
    if min(positive) < 1 or args.history < 0:
        parser.error("counts must be positive and history cannot be negative")

    total_steps = count_timesteps(args.start_date, args.end_date, args.stride_hours)
    raw_per_step = args.channels * args.height * args.width * 4
    raw_total = total_steps * raw_per_step
    sample_path = args.sample_zarr.expanduser().resolve()
    compressed_total: int | None = None
    if sample_path.is_dir():
        sample_bytes = directory_bytes(sample_path)
        compressed_total = round(sample_bytes / args.sample_timesteps * total_steps)
        compression_ratio = raw_total / compressed_total
    else:
        sample_bytes = 0
        compression_ratio = 0.0

    available = shutil.disk_usage(PROJECT_ROOT).free
    print("ERA5 training-readiness audit")
    print(
        f"period: {args.start_date} .. {args.end_date}, "
        f"cadence={args.stride_hours}h, timestamps={total_steps:,}"
    )
    print(
        f"tensor per timestamp: {args.channels}x{args.height}x{args.width} "
        f"float32 = {raw_per_step / 2**20:.2f} MiB"
    )
    print(f"uncompressed 4D fields: {raw_total / GIB:.2f} GiB")
    if compressed_total is not None:
        print(
            f"sample Zarr: {sample_bytes / GIB:.3f} GiB / "
            f"{args.sample_timesteps} timestamps"
        )
        print(f"observed compression ratio: {compression_ratio:.2f}x")
        print(f"projected downloaded Zarr: {compressed_total / GIB:.2f} GiB")
        print("logical train/valid/test slices: no additional field-data copy")
        recommended_free = 1.25 * compressed_total
        print(
            "recommended free disk during split/stats: "
            f"{recommended_free / GIB:.2f} GiB"
        )
        disk_verdict = (
            "PASS"
            if available >= recommended_free
            else "FAIL: choose a larger volume before downloading"
        )
    else:
        print(f"sample Zarr not found: {sample_path}")
        disk_verdict = "UNKNOWN: compressed size cannot be projected"
    print(f"currently available on project volume: {available / GIB:.2f} GiB")
    print(f"disk verdict: {disk_verdict}")

    start = datetime.fromisoformat(args.start_date)
    end_exclusive = datetime.fromisoformat(args.end_date) + timedelta(days=1)
    print("temporal training capacity:")
    for years in args.train_years:
        train_end = min(add_years(start, years), end_exclusive)
        train_steps = math.ceil(
            (train_end - start).total_seconds() / (args.stride_hours * 3600)
        )
        samples = max(
            0,
            train_steps - (args.history + args.rollout_steps) * args.time_step,
        )
        mini_batches = math.ceil(samples / args.batch_size)
        updates = math.ceil(mini_batches / args.gradient_accumulation)
        print(
            f"  {years} year(s): {train_steps:,} timestamps, "
            f"{samples:,} training windows, {updates:,} updates/epoch, "
            f"{updates * args.epochs:,} updates/{args.epochs} epochs"
        )
    print(
        "Caution: six-hourly windows and neighboring grid cells are strongly "
        "correlated; timestamp/pixel counts are not IID sample counts."
    )


if __name__ == "__main__":
    main()
