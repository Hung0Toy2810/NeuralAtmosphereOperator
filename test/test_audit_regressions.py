"""Reproduce audit failures through the production contracts and CLIs."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import torch
import xarray as xr
from torch.amp.grad_scaler import GradScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from configs.download_data_config import WeatherBenchDownloadConfig, channel_names
from configs.model_config import AtmosphereModelConfig
from neural_atmosphere_operator.data.loader import (
    AtmosphereDatasetConfig,
    AtmosphereZarrDataset,
)
from neural_atmosphere_operator.models.loss import ChannelRelativeAtmosphereLoss
from neural_atmosphere_operator.pipeline.metrics import LeadMetricAccumulator
from scripts.compute_stats import spherical_channel_statistics

UNITS = (
    ["m s**-1"] * 2
    + ["K", "Pa", "Pa", "kg m**-2"]
    + ["m**2 s**-2"] * 5
    + ["m s**-1"] * 8
    + ["K"] * 4
    + ["kg kg**-1"] * 2
    + ["1"]
)


def make_store(
    path, year, *, cadence=6, latitude=None, longitude=None, transpose=False
):
    data = np.random.default_rng(17).normal(size=(24, 26, 13, 24)).astype("float32")
    ds = xr.Dataset(
        {"state": (("time", "channel", "latitude", "longitude"), data)},
        coords={
            "time": np.datetime64(f"{year}-01-01", "h")
            + np.arange(24) * np.timedelta64(cadence, "h"),
            "channel": list(channel_names()),
            "channel_units": ("channel", UNITS),
            "latitude": np.linspace(90, -90, 13) if latitude is None else latitude,
            "longitude": np.arange(24) * 15.0 if longitude is None else longitude,
        },
    )
    if transpose:
        ds = ds.transpose("time", "latitude", "channel", "longitude")
    ds.chunk({"time": 1}).to_zarr(path, mode="w", consolidated=True)
    return data


def cli(args, *, code=None, success=True):
    command = [sys.executable] + (["-c", code] if code else []) + args
    result = subprocess.run(
        command,
        cwd=ROOT,
        env={
            **os.environ,
            "OMP_NUM_THREADS": "1",
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        },
        text=True,
        capture_output=True,
        timeout=180,
    )
    output = result.stdout + result.stderr
    assert (result.returncode == 0) == success, output
    return output


@pytest.fixture(scope="module")
def experiment(tmp_path_factory):
    work = tmp_path_factory.mktemp("audit-regression")
    for split, year in [("train", 1995), ("valid", 2019), ("test", 2020)]:
        make_store(work / f"{split}.zarr", year)
    stats = work / "stats"
    cli(
        [
            "scripts/compute_stats.py",
            "--train-data",
            str(work / "train.zarr"),
            "--output-dir",
            str(stats),
        ]
    )
    base = [
        "--train-data",
        str(work / "train.zarr"),
        "--valid-data",
        str(work / "valid.zarr"),
        "--means-path",
        str(stats / "means.npy"),
        "--stds-path",
        str(stats / "stds.npy"),
        "--time-diff-stds-path",
        str(stats / "time_diff_stds_dt1.npy"),
        "--epochs",
        "2",
        "--scheduler-epochs",
        "5",
        "--warmup-epochs",
        "2",
        "--batch-size",
        "4",
        "--gradient-accumulation",
        "2",
        "--rollout-steps",
        "2",
        "--num-workers",
        "0",
        "--embed-dim",
        "8",
        "--num-layers",
        "2",
        "--device",
        "cpu",
        "--max-samples",
        "17",
    ]
    snapshot = work / "epoch0.pt"
    code = (
        "import runpy,sys,shutil;sys.path[:0]=['.','src'];import neural_atmosphere_operator.pipeline.checkpoint as cp;original=cp.save_checkpoint\n"
        "def save(path,**kw):\n original(path,**kw)\n if kw['epoch']==0 and path.name=='last.pt':shutil.copyfile(path,"
        + repr(str(snapshot))
        + ")\n"
        "cp.save_checkpoint=save\nrunpy.run_path('scripts/train.py',run_name='__main__')"
    )
    cli(base + ["--run-dir", str(work / "full")], code=code)
    return work, stats, base, snapshot


def test_exact_resume_and_changed_arguments(experiment):
    work, stats, base, snapshot = experiment
    cli(
        ["scripts/train.py"]
        + base
        + ["--run-dir", str(work / "resume"), "--resume", str(snapshot)]
    )
    left = torch.load(
        work / "full/checkpoints/last.pt", map_location="cpu", weights_only=False
    )
    right = torch.load(
        work / "resume/checkpoints/last.pt", map_location="cpu", weights_only=False
    )
    for key in left["model_state"]:
        torch.testing.assert_close(
            left["model_state"][key], right["model_state"][key], rtol=0, atol=0
        )
    for flag, value in [("--warmup-start-factor", "0.9"), ("--max-samples", "18")]:
        output = cli(
            ["scripts/train.py"]
            + base
            + [
                "--run-dir",
                str(work / flag[2:]),
                "--resume",
                str(snapshot),
                flag,
                value,
            ],
            success=False,
        )
        assert "Resume must preserve" in output


def test_validation_artifact_records_every_target_lead(experiment):
    work, _, _, _ = experiment
    config = json.loads((work / "full/config.json").read_text())
    report = json.loads((work / "full/validation/epoch_0001.json").read_text())
    assert config["training"]["rollout_steps"] == 2
    assert config["training"]["validation_rollout_steps"] == 2
    assert config["valid_samples"] == 22
    assert report["selection_horizon_hours"] == 12
    assert report["selection_objective"] == {
        "name": "standardized_tendency_mse",
        "rollout_discount": 1.0,
    }
    assert report["selection_score"] == pytest.approx(report["mean_loss"])
    assert [item["lead_hours"] for item in report["lead_metrics"]] == [6, 12]


def evaluation_args(work, stats):
    return [
        "scripts/evaluate.py",
        "--checkpoint",
        str(work / "full/checkpoints/last.pt"),
        "--means-path",
        str(stats / "means.npy"),
        "--stds-path",
        str(stats / "stds.npy"),
        "--num-workers",
        "0",
        "--device",
        "cpu",
        "--rollout-steps",
        "2",
        "--max-samples",
        "1",
    ]


def test_valid_evaluation_baselines_and_forecast_beyond_observations(experiment):
    work, stats, _, _ = experiment
    cli(
        evaluation_args(work, stats)
        + [
            "--data-path",
            str(work / "test.zarr"),
            "--output-dir",
            str(work / "evaluation"),
        ]
    )
    report = json.loads((work / "evaluation/metrics.json").read_text())
    assert report["checkpoint_sha256"] and len(report["initializations"]) == 1
    assert set(report["baselines"]) == {"persistence", "training_climatology"}
    assert report["metrics"][0]["lead_hours"] == 6
    assert report["rollout_steps"] == 2
    assert report["objective"]["name"] == "standardized_tendency_mse"
    assert report["objective"]["loss"] == pytest.approx(
        sum(report["objective"]["loss_by_lead"]) / 2
    )
    cli(
        [
            "scripts/rollout.py",
            "--checkpoint",
            str(work / "full/checkpoints/last.pt"),
            "--data-path",
            str(work / "test.zarr"),
            "--means-path",
            str(stats / "means.npy"),
            "--stds-path",
            str(stats / "stds.npy"),
            "--device",
            "cpu",
            "--forecast-only",
            "--start-index",
            "23",
            "--rollout-steps",
            "3",
            "--output",
            str(work / "forecast.zarr"),
        ]
    )
    import zarr

    forecast = zarr.open_group(work / "forecast.zarr", mode="r")
    assert "target" not in forecast
    valid_times = np.asarray(forecast["valid_time_unix_ns"])
    initial_times = np.asarray(forecast["initial_time_unix_ns"])
    assert valid_times[0, -1] - initial_times[0] == 18 * 3_600_000_000_000


@pytest.mark.parametrize(
    "kind,expected",
    [
        ("cadence", "cadence_hours"),
        ("shifted", "longitude"),
        ("regional", "global north-to-south"),
        ("train", "training interval"),
        ("valid", "validation interval"),
        ("empty", "max_samples"),
    ],
)
def test_invalid_evaluation_is_rejected(experiment, kind, expected):
    work, stats, _, _ = experiment
    path = work / f"{kind}.zarr"
    if kind == "cadence":
        make_store(path, 2020, cadence=12)
    if kind == "shifted":
        make_store(path, 2020, longitude=np.arange(24) * 15.0 + 15)
    if kind == "regional":
        make_store(path, 2020, latitude=np.linspace(20, 0, 13))
    if kind == "empty":
        path = work / "test.zarr"
    command = evaluation_args(work, stats) + [
        "--data-path",
        str(path),
        "--output-dir",
        str(work / f"eval_{kind}"),
    ]
    if kind == "empty":
        command += ["--max-samples", "0"]
    assert expected in cli(command, success=False)


def test_mixed_statistics_are_rejected(experiment):
    work, stats, base, _ = experiment
    wrong = work / "wrong-stds.npy"
    np.save(wrong, np.load(stats / "stds.npy") * 2)
    output = cli(
        ["scripts/train.py"]
        + base
        + ["--run-dir", str(work / "wrong_stats"), "--stds-path", str(wrong)],
        success=False,
    )
    assert "Statistics bundle hash mismatch: stds" in output


def test_nonfinite_training_stops_before_checkpoint(experiment):
    work, _, base, _ = experiment
    code = "import runpy,sys,torch;sys.path[:0]=['.','src'];import neural_atmosphere_operator.pipeline.forecast as f;f.rollout_loss=lambda *a,**k:(torch.tensor(float('nan'),requires_grad=True),[],[]);runpy.run_path('scripts/train.py',run_name='__main__')"
    output = cli(base + ["--run-dir", str(work / "nan")], code=code, success=False)
    assert "Non-finite training loss" in output
    assert not (work / "nan/checkpoints/last.pt").exists()


def test_relative_loss_has_finite_zero_subgradient():
    for target_value in (0.0, 1.0):
        prediction = torch.full((2, 3, 5, 8), target_value, requires_grad=True)
        value, _ = ChannelRelativeAtmosphereLoss()(
            prediction, prediction.detach().clone()
        )
        value.backward()
        assert value.item() == 0
        torch.testing.assert_close(prediction.grad, torch.zeros_like(prediction))


def test_dimension_order_and_spread_sampling(tmp_path):
    values = make_store(tmp_path / "transposed.zarr", 2020, transpose=True)
    dataset = AtmosphereZarrDataset(
        AtmosphereDatasetConfig(
            data_path=tmp_path / "transposed.zarr",
            normalize=False,
            max_samples=3,
            sample_selection="evenly_spaced",
        )
    )
    np.testing.assert_array_equal(dataset[0]["input"], values[0])
    assert dataset.sample_indices.tolist() == [0, 11, 22]
    dataset.close()


def test_streaming_statistics_float64_stride_and_bounded_scheduler_cache():
    from dask.callbacks import Callback

    rng = np.random.default_rng(4)
    values = (1e5 + rng.normal(size=(64, 2, 5, 8))).astype(np.float64)
    state = xr.DataArray(
        values, dims=("time", "channel", "latitude", "longitude")
    ).chunk({"time": 1})
    weights = xr.DataArray(np.array([0.1, 0.5, 1.0, 0.5, 0.1]), dims="latitude")
    peaks = []

    def record(key, result, dsk, state, worker):
        peaks.append(sum(getattr(v, "nbytes", 0) for v in state["cache"].values()))

    with Callback(posttask=record):
        result = spherical_channel_statistics(state, weights, 3)
    area = np.asarray(weights)[None, None, :, None]
    area = area / area.mean()
    delta = values[3:] - values[:-3]
    mean = (values * area).mean(axis=(0, 2, 3))
    delta_mean = (delta * area).mean(axis=(0, 2, 3))
    np.testing.assert_allclose(result["mean"], mean, atol=1e-9)
    np.testing.assert_allclose(
        result["variance"],
        ((values - mean[None, :, None, None]) ** 2 * area).mean(axis=(0, 2, 3)),
        rtol=1e-10,
    )
    np.testing.assert_allclose(result["time_diff_mean"], delta_mean, atol=1e-11)
    assert max(peaks, default=0) < values.nbytes / 4


@pytest.mark.parametrize(
    "config",
    [
        dict(use_complex_kernels=False),
        dict(scale_factor=1),
        dict(num_layers=1, normalization_layer="layer_norm"),
        dict(hard_thresholding_fraction=1e-9),
    ],
)
def test_unsupported_kernel_configurations_fail(config):
    with pytest.raises(ValueError):
        AtmosphereModelConfig(**config)


def test_download_phase_and_empty_metrics_rejected():
    with pytest.raises(ValueError, match="cadence phase"):
        WeatherBenchDownloadConfig(time_stride_hours=18)
    with pytest.raises(ValueError, match="empty evaluation"):
        LeadMetricAccumulator().result(1)


def test_preflight_runs_actual_updates_and_validation(experiment):
    work, _, _, _ = experiment
    cli(
        [
            "scripts/preflight_gpu.py",
            "--data-dir",
            str(work),
            "--train-data",
            str(work / "train.zarr"),
            "--valid-data",
            str(work / "valid.zarr"),
            "--device",
            "cpu",
            "--embed-dim",
            "8",
            "--num-layers",
            "2",
            "--batch-size",
            "2",
            "--gradient-accumulation",
            "2",
            "--updates",
            "2",
            "--rollout-steps",
            "2",
            "--output",
            str(work / "preflight.json"),
        ]
    )
    result = json.loads((work / "preflight.json").read_text())
    assert result["status"] == "passed" and len(result["losses"]) == 2
    assert result["peak_allocated_gib"] is None


def test_nonfinite_checkpoint_preserves_last_good(tmp_path, monkeypatch):
    from neural_atmosphere_operator.pipeline import checkpoint as cp

    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    monkeypatch.setattr(cp, "runtime_fingerprint", lambda device: {})
    path = tmp_path / "last.pt"
    scaler = GradScaler("cuda", enabled=False)
    generator = torch.Generator()

    def save() -> None:
        cp.save_checkpoint(
            path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            train_generator=generator,
            epoch=0,
            completed_updates=0,
            best_validation_loss=1.0,
            epochs_without_improvement=0,
            early_stopped=False,
            model_config={},
            training_config={},
        )

    save()
    original = path.read_bytes()
    with torch.no_grad():
        model.weight.fill_(float("nan"))
    with pytest.raises(FloatingPointError):
        save()
    assert path.read_bytes() == original


def test_lazy_truth_matches_eager_and_does_not_materialize_horizon(tmp_path):
    from torch.utils.data import DataLoader

    from neural_atmosphere_operator.pipeline.forecast import lazy_target_provider

    make_store(tmp_path / "data.zarr", 2020)
    config = AtmosphereDatasetConfig(
        data_path=tmp_path / "data.zarr", normalize=False, rollout_steps=20
    )
    eager = AtmosphereZarrDataset(config)
    lazy = AtmosphereZarrDataset(replace(config, lazy_targets=True))
    raw = next(iter(DataLoader(lazy, batch_size=2)))
    assert raw["target"].numel() == 0
    read = lazy_target_provider(lazy, raw)
    assert read is not None
    expected = next(iter(DataLoader(eager, batch_size=2)))["target"]
    for lead in (0, 9, 19):
        torch.testing.assert_close(read(lead), expected[:, lead], rtol=0, atol=0)
    eager.close()
    lazy.close()


def test_nonfinite_gradient_never_updates_optimizer(experiment):
    work, _, base, _ = experiment
    marker = work / "unsafe_update"
    code = (
        "import runpy,sys,torch;sys.path[:0]=['.','src'];import neural_atmosphere_operator.pipeline.runtime as rt;original=rt.build_model\n"
        "def build(*a,**kw):\n m=original(*a,**kw)\n next(m.parameters()).register_hook(lambda g:g*float('nan'))\n return m\n"
        "rt.build_model=build\n"
        "def step(*a,**kw):\n open(" + repr(str(marker)) + ",'w').write('unsafe')\n"
        "torch.optim.AdamW.step=step\nrunpy.run_path('scripts/train.py',run_name='__main__')"
    )
    output = cli(base + ["--run-dir", str(work / "nan_grad")], code=code, success=False)
    assert "non-finite" in output
    assert not marker.exists()


def test_payload_edit_changes_resume_identity(tmp_path):
    import zarr

    from neural_atmosphere_operator.pipeline.runtime import dataset_signature

    make_store(tmp_path / "data.zarr", 2020)
    dataset = AtmosphereZarrDataset(
        AtmosphereDatasetConfig(data_path=tmp_path / "data.zarr", normalize=False)
    )
    before = dataset_signature(dataset)
    group = zarr.open_group(tmp_path / "data.zarr", mode="a")
    group["state"][0, 0, 0, 0] = 42.0
    after = dataset_signature(dataset)
    assert before["payload_inventory_sha256"] != after["payload_inventory_sha256"]
    dataset.close()


def test_full_scan_verifies_payload_and_rejects_mutation(experiment):
    import zarr

    from configs.pipeline_config import DataPaths

    work, _, _, _ = experiment
    stores = [
        xr.open_zarr(work / f"{split}.zarr", consolidated=True)
        for split in ("train", "valid", "test")
    ]
    canonical = DataPaths(work).dataset
    try:
        xr.concat(stores, dim="time").to_zarr(
            cast(Any, str(canonical)), mode="w", consolidated=True
        )
    finally:
        for store in stores:
            store.close()
    code = (
        "import sys,runpy;sys.path[:0]=['.','src'];from configs.pipeline_config import DataPaths;"
        "DataPaths.split_time_range=lambda self,split:({'train':('1995-01-01','1995-01-06'),"
        "'valid':('2019-01-01','2019-01-06'),'test':('2020-01-01','2020-01-06')}[split]);"
        "runpy.run_path('scripts/validate_data.py',run_name='__main__')"
    )
    cli(
        [
            "--data-dir",
            str(work),
            "--full-scan",
            "--output",
            str(work / "data_validation.json"),
        ],
        code=code,
    )
    report = json.loads((work / "data_validation.json").read_text())
    assert report["full_scan"] and all(
        value["state_float32_sha256"] for value in report["splits"].values()
    )
    group = zarr.open_group(canonical, mode="a")
    state = cast(Any, group["state"])
    state[0, 0, 0, 0] = float(state[0, 0, 0, 0]) + 1.0
    result = cli(["--data-dir", str(work), "--full-scan"], code=code, success=False)
    assert "Training payload differs" in result


def test_fp16_complex_training_profile_fails_fast():
    output = cli(["scripts/train.py", "--amp-dtype", "float16"], success=False)
    assert "FP16 GradScaler with complex spectral gradients" in output


def test_constant_preserving_sht_formula_and_gradient():
    from torch_harmonics import RealSHT

    from neural_atmosphere_operator.models.transforms import ConstantPreservingRealSHT

    native = RealSHT(13, 24, lmax=4, mmax=4).double()
    stable = ConstantPreservingRealSHT(native)
    value = torch.randn(1, 2, 13, 24, dtype=torch.float64, requires_grad=True)
    torch.testing.assert_close(stable(value), native(value), rtol=1e-11, atol=1e-12)
    assert torch.autograd.gradcheck(stable, (value,), fast_mode=True)
    constant = torch.full_like(value, 3.0)
    coefficients = stable(constant)
    torch.testing.assert_close(
        coefficients[..., 0, 0].real,
        torch.full((1, 2), 3 * np.sqrt(4 * np.pi), dtype=torch.float64),
    )
    assert coefficients[..., 1:, :].abs().max() == 0
    assert coefficients[..., :, 1:].abs().max() == 0


def test_model_constant_fields_and_transform_aliases():
    from neural_atmosphere_operator.models.model import AtmosphereNeuralOperator
    from neural_atmosphere_operator.models.transforms import ConstantPreservingRealSHT

    config = AtmosphereModelConfig(img_size=(13, 24), embed_dim=8, num_layers=2)
    torch.manual_seed(42)
    model = AtmosphereNeuralOperator(config)
    output = model(torch.ones(1, 26, 13, 24))
    assert output.std(dim=(-2, -1)).max() < 1e-5
    output.square().mean().backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()
    )
    forward_transforms = [
        m for m in model.modules() if isinstance(m, ConstantPreservingRealSHT)
    ]
    assert len(forward_transforms) == 2  # shared references, not per-block copies
