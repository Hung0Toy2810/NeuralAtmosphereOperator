"""Small real-ERA5 learning check; not a held-out forecast-skill experiment."""

import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np
import xarray as xr

root = Path.cwd()
work = Path(tempfile.mkdtemp(prefix="nao-real-pilot-"))
evidence = root / "audit/remediation"
source_path = root / "data/dataset/era5_sample_audited.zarr"
with xr.open_zarr(source_path, consolidated=True) as source:
    assert source.state.shape == (4, 26, 361, 720)
    assert all(str(unit).strip() for unit in source.channel_units.values)
    assert np.isfinite(source.state.values).all()
    source.isel(time=slice(0, 2)).to_zarr(
        work / "train.zarr", mode="w", consolidated=True
    )
    source.isel(time=slice(2, 4)).to_zarr(
        work / "valid.zarr", mode="w", consolidated=True
    )
    units = source.channel_units.values.tolist()


def run(name, args):
    result = subprocess.run(
        [sys.executable] + args,
        cwd=root,
        env={**os.environ, "OMP_NUM_THREADS": "1"},
        capture_output=True,
        text=True,
        timeout=300,
    )
    (evidence / f"real_sample_{name}.txt").write_text(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)


run(
    "stats",
    [
        "scripts/compute_stats.py",
        "--train-data",
        str(work / "train.zarr"),
        "--output-dir",
        str(work / "stats"),
    ],
)
run(
    "train",
    [
        "scripts/train.py",
        "--data-dir",
        str(work),
        "--train-data",
        str(work / "train.zarr"),
        "--valid-data",
        str(work / "valid.zarr"),
        "--run-dir",
        str(work / "run"),
        "--device",
        "cpu",
        "--embed-dim",
        "8",
        "--num-layers",
        "2",
        "--epochs",
        "12",
        "--warmup-epochs",
        "1",
        "--batch-size",
        "1",
        "--gradient-accumulation",
        "1",
        "--num-workers",
        "0",
        "--validation-rollout-steps",
        "1",
        "--max-validation-samples",
        "1",
        "--learning-rate",
        "0.001",
    ],
)
with (work / "run/history.csv").open() as file:
    rows = list(csv.DictReader(file))
result = {
    "scope": "real-data tiny-model learning sanity check; adjacent within-day validation is not independent forecast evidence",
    "source": str(source_path),
    "temporary_work_directory": str(work),
    "shape": [4, 26, 361, 720],
    "channel_units": units,
    "model": "E8/L2, full 361x720 grid, stabilized SHT, CPU",
    "train_period": "2018-01-01 00:00 to 06:00",
    "valid_period": "2018-01-01 12:00 to 18:00",
    "statistics_train_only": True,
    "updates": len(rows),
    "first_train_loss": float(rows[0]["train_loss"]),
    "last_train_loss": float(rows[-1]["train_loss"]),
    "first_validation_loss": float(rows[0]["validation_loss"]),
    "last_validation_loss": float(rows[-1]["validation_loss"]),
    "maximum_preclip_gradient_norm": max(
        float(row["max_preclip_gradient_norm"]) for row in rows
    ),
    "nonfinite_gradient_norms": sum(
        int(row["nonfinite_gradient_norms"]) for row in rows
    ),
}
assert result["last_train_loss"] < result["first_train_loss"]
assert result["nonfinite_gradient_norms"] == 0
(evidence / "real_sample_probe.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
