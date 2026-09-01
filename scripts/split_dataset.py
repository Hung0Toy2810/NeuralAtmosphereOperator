"""Chronologically split a local ERA5 Zarr store into train/valid/test."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, cast

import numpy as np
import xarray as xr


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/dataset"))
    parser.add_argument("--train-end", required=True, help="Inclusive ISO date")
    parser.add_argument("--valid-end", required=True, help="Inclusive ISO date")
    args = parser.parse_args()
    if args.train_end >= args.valid_end:
        raise ValueError("train-end must be earlier than valid-end")
    train_end = np.datetime64(
        args.train_end + "T23:59:59" if len(args.train_end) == 10 else args.train_end
    )
    valid_end = np.datetime64(
        args.valid_end + "T23:59:59" if len(args.valid_end) == 10 else args.valid_end
    )
    source = args.source.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    destinations = {
        "train": output / "era5_train.zarr",
        "valid": output / "era5_valid.zarr",
        "test": output / "era5_test.zarr",
    }
    existing = [str(path) for path in destinations.values() if path.exists()]
    if existing:
        raise FileExistsError("Refusing to overwrite split(s): " + ", ".join(existing))
    ds = xr.open_zarr(str(source))
    try:
        time = ds.time
        train = ds.where(time <= train_end, drop=True)
        valid = ds.where((time > train_end) & (time <= valid_end), drop=True)
        test = ds.where(time > valid_end, drop=True)
        splits = {"train": train, "valid": valid, "test": test}
        if any(split.sizes.get("time", 0) < 2 for split in splits.values()):
            raise ValueError("Every chronological split must contain at least two steps")
        output.mkdir(parents=True, exist_ok=True)
        for name, split in splits.items():
            print(
                f"Writing {name}: {split.sizes['time']} steps "
                f"({str(split.time.values[0])} .. {str(split.time.values[-1])})"
            )
            split.to_zarr(cast(Any, str(destinations[name])), mode="w")
    finally:
        ds.close()


if __name__ == "__main__":
    main()
