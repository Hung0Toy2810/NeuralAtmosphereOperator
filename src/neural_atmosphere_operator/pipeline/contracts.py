"""Semantic forecast and immutable statistics contracts, independent of paths."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .runtime import statistics_signature


def forecast_contract(dataset, time_step: int) -> dict[str, Any]:
    return {
        "version": 1,
        "channels": list(dataset.channel_names),
        "channel_units": list(dataset.channel_units),
        "latitude": dataset.latitudes.tolist(),
        "longitude": dataset.longitudes.tolist(),
        "cadence_hours": dataset.cadence_hours,
        "forecast_step_hours": time_step * dataset.cadence_hours,
    }


def validate_global_grid(dataset) -> None:
    lat, lon = dataset.latitudes, dataset.longitudes
    if not np.allclose(lat, np.linspace(90, -90, len(lat)), atol=1e-4, rtol=0):
        raise ValueError(
            "SFNO requires a global north-to-south equiangular latitude grid"
        )
    if not np.isclose(float(lon[1] - lon[0]) * len(lon), 360, atol=1e-4):
        raise ValueError("SFNO longitude grid must cover exactly 360 degrees")
    if any(not unit.strip() for unit in dataset.channel_units):
        raise ValueError("Missing physical channel units")


def enforce_forecast_contract(
    training: dict, dataset, time_step: int, split: str, *, diagnostic: bool = False
) -> None:
    """Never allow cadence/grid/leakage overrides to masquerade as valid scores."""
    validate_global_grid(dataset)
    saved = training.get("forecast_contract")
    if saved is None:
        raise ValueError(
            "Checkpoint lacks forecast_contract; migrate using verified training metadata"
        )
    actual = forecast_contract(dataset, time_step)
    differences = [key for key in actual if actual[key] != saved.get(key)]
    if differences:
        raise ValueError("Forecast data contract mismatch: " + ", ".join(differences))
    if training.get("diagnostic_run") and not diagnostic:
        raise ValueError(
            "Diagnostic checkpoint requires explicit diagnostic evaluation"
        )
    interval = training.get("data_signature", {}).get("train", {})
    if split in {"valid", "test"}:
        if not interval.get("first_time") or not interval.get("last_time"):
            raise ValueError("Checkpoint lacks training interval provenance")
        overlap = np.datetime64(dataset.first_time) <= np.datetime64(
            interval["last_time"]
        ) and np.datetime64(dataset.last_time) >= np.datetime64(interval["first_time"])
        if overlap:
            raise ValueError(
                "Held-out evaluation overlaps checkpoint training interval"
            )
    if split == "test":
        valid = training.get("data_signature", {}).get("valid", {})
        if valid.get("first_time") and valid.get("last_time"):
            overlap = np.datetime64(dataset.first_time) <= np.datetime64(
                valid["last_time"]
            ) and np.datetime64(dataset.last_time) >= np.datetime64(valid["first_time"])
            if overlap:
                raise ValueError(
                    "Test evaluation overlaps checkpoint validation interval"
                )


def validate_statistics_bundle(dataset, paths: dict[str, Path], time_step: int) -> dict:
    metadata_path = paths["means"].expanduser().resolve().parent / "stats.json"
    if not metadata_path.is_file():
        raise ValueError(
            "stats.json is required; recompute train-only statistics with compute_stats.py"
        )
    metadata = json.loads(metadata_path.read_text())
    expected = {
        "first_time": dataset.first_time,
        "last_time": dataset.last_time,
        "channels": list(dataset.channel_names),
        "channel_units": list(dataset.channel_units),
        "cadence_hours": dataset.cadence_hours,
        "latitude": dataset.latitudes.tolist(),
        "longitude": dataset.longitudes.tolist(),
        "spatial_weighting": "spherical_latitude_cell_area",
        "time_difference_step": time_step,
    }
    if metadata.get("statistics_version") != 4:
        raise ValueError(
            "Statistics bundle requires version 4 with artifact hashes; recompute statistics"
        )
    mismatch = [key for key, value in expected.items() if metadata.get(key) != value]
    if mismatch:
        raise ValueError(
            "Normalization metadata does not match training: " + ", ".join(mismatch)
        )
    for name, path in paths.items():
        if statistics_signature(path)["sha256"] != metadata.get("artifacts", {}).get(
            name
        ):
            raise ValueError(f"Statistics bundle hash mismatch: {name}")
    return metadata
