"""Validate Zarr split geometry, statistics, and one sample contract."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import cast

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT_ROOT), str(PROJECT_ROOT / "src")]

import numpy as np

from configs.pipeline_config import DataPaths
from neural_atmosphere_operator.pipeline.runtime import build_loader
from neural_atmosphere_operator.utils.logger import setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DataPaths().root)
    parser.add_argument("--split", choices=("all", "train", "valid", "test"), default="all")
    parser.add_argument("--rollout-steps", type=int, default=1)
    parser.add_argument("--history", type=int, default=0)
    parser.add_argument("--time-step", type=int, default=1)
    parser.add_argument(
        "--temp-diff-normalization",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = setup_logger()
    paths = DataPaths(args.data_dir.expanduser().resolve())
    for path in (paths.means, paths.stds):
        if not path.is_file():
            raise FileNotFoundError(f"Missing normalization file: {path}")
    means, stds = np.load(paths.means), np.load(paths.stds)
    if means.shape != stds.shape or not np.all(np.isfinite(means)):
        raise ValueError("Invalid normalization statistics")
    if not np.all(np.isfinite(stds)) or np.any(stds <= 0):
        raise ValueError("Every channel standard deviation must be finite and positive")
    if not paths.climatology.is_file():
        raise FileNotFoundError(f"Missing training climatology: {paths.climatology}")
    climatology = np.load(paths.climatology, mmap_mode="r")
    if climatology.ndim != 3 or climatology.shape[0] != means.size:
        raise ValueError("Training climatology must have shape [C, H, W]")
    if not np.all(np.isfinite(climatology)):
        raise ValueError("Training climatology must be finite")
    if args.temp_diff_normalization:
        delta_path = paths.time_diff_stds(args.time_step)
        if not delta_path.is_file():
            raise FileNotFoundError(f"Missing time-difference statistics: {delta_path}")
        delta_stds = np.load(delta_path)
        if delta_stds.shape != stds.shape:
            raise ValueError("Time-difference statistics must match global statistics")
        if not np.all(np.isfinite(delta_stds)) or np.any(delta_stds <= 0):
            raise ValueError(
                "Every time-difference standard deviation must be finite and positive"
            )

    splits = ("train", "valid", "test") if args.split == "all" else (args.split,)
    checked = 0
    records: dict[str, dict[str, object]] = {}
    reference_grid: tuple[np.ndarray, np.ndarray, float] | None = None
    for split in splits:
        data_path = getattr(paths, split)
        if not data_path.exists():
            logger.warning("Skipping missing split: %s", data_path)
            continue
        loader, dataset = build_loader(
            paths,
            split,
            rollout_steps=args.rollout_steps,
            history=args.history,
            time_step=args.time_step,
            batch_size=1,
            num_workers=0,
        )
        try:
            batch = next(iter(loader))
            if not all(np.isfinite(value.numpy()).all() for value in (batch["input"], batch["target"])):
                raise ValueError(f"Non-finite data found in {split}")
            if dataset.config.spatial_crop is None:
                if not np.isclose(np.abs(dataset.latitudes[[0, -1]]), 90.0).all():
                    raise ValueError("Full SFNO grid must include both poles")
            if means.size != dataset.config.channel_count:
                raise ValueError("Normalization statistics do not match dataset channels")
            if len(dataset.channel_units) != dataset.config.channel_count or any(
                not unit.strip() for unit in dataset.channel_units
            ):
                raise ValueError(
                    "Every channel must carry a non-empty physical unit; regenerate "
                    "legacy flattened stores with the current downloader"
                )
            if tuple(climatology.shape[-2:]) != dataset.spatial_shape:
                raise ValueError("Training climatology does not match dataset grid")
            grid = (
                dataset.latitudes.copy(),
                dataset.longitudes.copy(),
                dataset.cadence_hours,
            )
            if reference_grid is None:
                reference_grid = grid
            elif (
                not np.array_equal(grid[0], reference_grid[0])
                or not np.array_equal(grid[1], reference_grid[1])
                or grid[2] != reference_grid[2]
            ):
                raise ValueError("All dataset splits must use the same grid and cadence")
            records[split] = {
                "path": dataset.data_path,
                "first_time": dataset.first_time,
                "last_time": dataset.last_time,
                "channels": dataset.channel_names,
                "channel_units": dataset.channel_units,
                "cadence_hours": dataset.cadence_hours,
            }
            logger.info(
                "%s: samples=%d time=%s..%s grid=%s input=%s target=%s",
                split,
                len(dataset),
                dataset.first_time,
                dataset.last_time,
                dataset.spatial_shape,
                tuple(batch["input"].shape),
                tuple(batch["target"].shape),
            )
            checked += 1
        finally:
            dataset.close()
    if checked == 0:
        raise FileNotFoundError("No dataset split could be validated")
    ordered = [split for split in ("train", "valid", "test") if split in records]
    for left, right in zip(ordered, ordered[1:]):
        if np.datetime64(str(records[left]["last_time"])) >= np.datetime64(
            str(records[right]["first_time"])
        ):
            raise ValueError(f"Temporal overlap between {left} and {right} splits")

    if "train" in records:
        metadata_path = paths.means.parent / "stats.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(
                f"Missing statistics provenance metadata: {metadata_path}"
            )
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        train = records["train"]
        expected = {
            "source": str(train["path"]),
            "first_time": train["first_time"],
            "last_time": train["last_time"],
            "channels": list(cast(tuple[str, ...], train["channels"])),
            "channel_units": list(
                cast(tuple[str, ...], train["channel_units"])
            ),
            "spatial_weighting": "spherical_latitude_cell_area",
            "time_difference_step": args.time_step,
            "cadence_hours": train["cadence_hours"],
        }
        mismatches = [key for key, value in expected.items() if metadata.get(key) != value]
        if mismatches:
            raise ValueError(
                "Statistics provenance does not match training data: "
                + ", ".join(mismatches)
            )


if __name__ == "__main__":
    main()
