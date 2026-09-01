"""Unit tests for AtmosphereNormalizer and AtmosphereZarrDataset."""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import torch
import xarray as xr
from torch.utils.data import TensorDataset

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
src_root = project_root / "src"
if str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))

from neural_atmosphere_operator.data.normalization import AtmosphereNormalizer
from neural_atmosphere_operator.data.loader import (
    AtmosphereDatasetConfig,
    AtmosphereZarrDataset,
    create_data_loader,
)
from data.download_data import (
    _conservative_half_degree,
    _latitude_overlap_weights,
    build_subset,
    contract_fingerprint,
    contract_json,
    download_contract,
    validate_resume_contract,
)
from configs.download_data_config import WeatherBenchDownloadConfig
from scripts.compute_stats import spherical_channel_statistics


def test_half_degree_regrid_preserves_constants_and_global_integral():
    source_latitudes = np.linspace(90.0, -90.0, 721)
    target_latitudes = np.linspace(90.0, -90.0, 361)
    rows = _latitude_overlap_weights(source_latitudes, target_latitudes)
    constant = np.full((721, 1440), 7.0, dtype=np.float32)
    np.testing.assert_array_equal(_conservative_half_degree(constant, rows), 7.0)

    field = np.random.default_rng(4).normal(size=(721, 1440)).astype(np.float32)
    regridded = _conservative_half_degree(field, rows)
    source_bounds = np.linspace(-90.125, 90.125, 722).clip(-90.0, 90.0)
    target_bounds = np.linspace(-90.25, 90.25, 362).clip(-90.0, 90.0)
    source_weights = np.diff(np.sin(np.deg2rad(source_bounds)))[::-1]
    target_weights = np.diff(np.sin(np.deg2rad(target_bounds)))[::-1]
    source_mean = np.sum(field.mean(axis=1) * source_weights) / source_weights.sum()
    target_mean = (
        np.sum(regridded.mean(axis=1) * target_weights) / target_weights.sum()
    )
    assert target_mean == pytest.approx(source_mean, abs=1e-8)


def test_half_degree_regrid_uses_periodic_aligned_longitude_overlap():
    source_latitudes = np.linspace(90.0, -90.0, 721)
    target_latitudes = np.linspace(90.0, -90.0, 361)
    rows = _latitude_overlap_weights(source_latitudes, target_latitudes)

    # A source cell centred at 359.75 degrees overlaps equally with the target
    # cells centred at 359.5 and 0 degrees.  Its contribution is one quarter of
    # either target-cell average because the target cell is twice as wide.
    field = np.zeros((721, 1440), dtype=np.float32)
    field[:, -1] = 1.0
    regridded = _conservative_half_degree(field, rows)
    np.testing.assert_allclose(regridded[:, 0], 0.25, rtol=0, atol=1e-7)
    np.testing.assert_allclose(regridded[:, -1], 0.25, rtol=0, atol=1e-7)
    np.testing.assert_allclose(regridded[:, 1:-1], 0.0, rtol=0, atol=1e-7)

    # Verify the complete longitude stencil independently on a smooth wave;
    # this catches the former +0.125-degree phase shift while remaining exact
    # for the piecewise-constant first-order conservative scheme.
    longitude = np.deg2rad(np.arange(1440, dtype=np.float64) * 0.25)
    wave = np.cos(17.0 * longitude).astype(np.float32)
    field = np.broadcast_to(wave, (721, 1440)).copy()
    regridded = _conservative_half_degree(field, rows)
    expected = (
        0.25 * np.roll(wave, 1)[::2]
        + 0.5 * wave[::2]
        + 0.25 * np.roll(wave, -1)[::2]
    )
    np.testing.assert_allclose(
        regridded,
        np.broadcast_to(expected, regridded.shape),
        rtol=2e-6,
        atol=2e-7,
    )


def test_download_contract_fingerprint_covers_semantic_configuration():
    config = WeatherBenchDownloadConfig()
    fingerprint = contract_fingerprint(download_contract(config))
    assert len(fingerprint) == 64
    assert contract_fingerprint(download_contract(config)) == fingerprint
    assert contract_fingerprint(
        download_contract(replace(config, end_date="2019-02-17"))
    ) != fingerprint
    progress = {
        "download_contract": download_contract(config),
        "download_contract_sha256": fingerprint,
    }
    attrs = {
        "download_contract_json": contract_json(download_contract(config)),
        "download_contract_sha256": fingerprint,
    }
    validate_resume_contract(progress, attrs, config)
    with pytest.raises(RuntimeError, match="download contract"):
        validate_resume_contract(progress, attrs, replace(config, end_date="2019-02-17"))
    assert contract_fingerprint(
        download_contract(
            replace(config, surface_variables=config.surface_variables[:-1])
        )
    ) != fingerprint


def test_flattened_state_keeps_per_channel_metadata(tmp_path: Path):
    latitude = np.linspace(90.0, -90.0, 721)
    longitude = np.arange(1440, dtype=np.float64) * 0.25
    shape = (1, 721, 1440)
    source = xr.Dataset(
        {
            "10m_u_component_of_wind": xr.DataArray(
                np.zeros(shape, dtype=np.float32),
                dims=("time", "latitude", "longitude"),
                attrs={"units": "m s**-1", "long_name": "10 metre U wind"},
            ),
            "2m_temperature": xr.DataArray(
                np.ones(shape, dtype=np.float32),
                dims=("time", "latitude", "longitude"),
                attrs={"units": "K", "long_name": "2 metre temperature"},
            ),
        },
        coords={
            "time": np.asarray(["2019-01-01T00"], dtype="datetime64[h]"),
            "latitude": latitude,
            "longitude": longitude,
        },
    )
    config = replace(
        WeatherBenchDownloadConfig(),
        surface_variables=("10m_u_component_of_wind", "2m_temperature"),
        level_selections=(),
        start_date="2019-01-01",
        end_date="2019-01-01",
    )
    subset = build_subset(config, source)
    assert tuple(str(value) for value in subset.channel_units.values) == (
        "m s**-1",
        "K",
    )
    assert tuple(str(value) for value in subset.channel_source_variable.values) == (
        "10m_u_component_of_wind",
        "2m_temperature",
    )
    assert "units" not in subset.state.attrs
    assert subset.attrs["channel_metadata_version"] == 1
    path = tmp_path / "metadata.zarr"
    subset.to_zarr(cast(Any, str(path)), mode="w", consolidated=True)
    reopened = xr.open_zarr(path, consolidated=True)
    try:
        assert tuple(str(value) for value in reopened.channel_units.values) == (
            "m s**-1",
            "K",
        )
        assert "units" not in reopened.state.attrs
    finally:
        reopened.close()


def test_atmosphere_normalizer_numpy_and_torch():
    c, h, w = 25, 32, 64
    means = np.random.uniform(200.0, 300.0, size=(c,)).astype(np.float32)
    stds = np.random.uniform(5.0, 20.0, size=(c,)).astype(np.float32)

    normalizer = AtmosphereNormalizer(means=means, stds=stds, normalize=True)

    # Test NumPy
    x_np = np.random.normal(250.0, 10.0, size=(c, h, w)).astype(np.float32)
    norm_np = normalizer.normalize(x_np)
    denorm_np = normalizer.denormalize(norm_np)
    assert np.allclose(x_np, denorm_np, atol=1e-5)

    # Test PyTorch Tensor
    x_torch = torch.from_numpy(x_np).unsqueeze(0)  # [1, C, H, W]
    norm_torch = normalizer.normalize(x_torch)
    denorm_torch = normalizer.denormalize(norm_torch)
    assert torch.allclose(x_torch, denorm_torch, atol=1e-5)

    # The documented [..., C, H, W] contract must also preserve 3D and 5D
    # tensor shapes (the old implementation accidentally added a batch axis
    # to 3D tensors and could not broadcast over rollout tensors).
    for shape in ((c, h, w), (2, 3, c, h, w)):
        values = torch.randn(shape)
        transformed = normalizer.normalize(values)
        assert transformed.shape == values.shape
        torch.testing.assert_close(
            normalizer.denormalize(transformed), values, rtol=1e-5, atol=5e-5
        )


def test_vectorized_channel_statistics_match_direct_weighted_reductions():
    generator = np.random.default_rng(12)
    values = generator.normal(size=(5, 2, 3, 4)).astype(np.float32)
    state = xr.DataArray(
        values,
        dims=("time", "channel", "latitude", "longitude"),
        coords={"latitude": [90.0, 0.0, -90.0], "channel": ["a", "b"]},
    )
    weights = xr.DataArray(
        np.asarray([0.25, 1.0, 0.25]),
        dims=("latitude",),
        coords={"latitude": state.latitude},
    )
    statistics = spherical_channel_statistics(state, weights, 1).compute()
    reductions = (0, 2, 3)
    broadcast_weights = np.broadcast_to(
        np.asarray(weights)[None, None, :, None], values.shape
    )
    expected_mean = (values * broadcast_weights).sum(axis=reductions) / (
        broadcast_weights.sum(axis=reductions)
    )
    expected_variance = (
        (values - expected_mean[None, :, None, None]) ** 2 * broadcast_weights
    ).sum(axis=reductions) / broadcast_weights.sum(axis=reductions)
    delta = values[1:] - values[:-1]
    delta_weights = broadcast_weights[:-1]
    expected_delta_mean = (delta * delta_weights).sum(axis=reductions) / (
        delta_weights.sum(axis=reductions)
    )
    expected_delta_variance = (
        (delta - expected_delta_mean[None, :, None, None]) ** 2 * delta_weights
    ).sum(axis=reductions) / delta_weights.sum(axis=reductions)
    np.testing.assert_allclose(statistics["mean"], expected_mean, rtol=2e-6)
    np.testing.assert_allclose(
        statistics["variance"], expected_variance, rtol=2e-6
    )
    np.testing.assert_allclose(
        statistics["time_diff_mean"], expected_delta_mean, rtol=2e-6
    )
    np.testing.assert_allclose(
        statistics["time_diff_variance"], expected_delta_variance, rtol=2e-6
    )
    np.testing.assert_allclose(statistics["time_mean"], values.mean(axis=0), rtol=2e-6)


def test_normalizer_validates_statistics_and_channel_selection():
    with pytest.raises(ValueError, match="same number"):
        AtmosphereNormalizer(np.zeros(2), np.ones(3))
    with pytest.raises(ValueError, match="strictly positive"):
        AtmosphereNormalizer(np.zeros(2), np.array([1.0, -1.0]))

    normalizer = AtmosphereNormalizer(np.array([10.0, 20.0]), np.array([2.0, 5.0]))
    values = np.full((1, 2, 2), 25.0, dtype=np.float32)
    transformed = normalizer.normalize(values, channels=(1,))
    np.testing.assert_allclose(transformed, 1.0)


def test_latitude_weights():
    lats = np.linspace(90.0, -90.0, 721, dtype=np.float32)
    weights_np = AtmosphereNormalizer.get_latitude_weights(lats)
    assert isinstance(weights_np, np.ndarray)
    assert len(weights_np) == 721
    assert np.isclose(np.mean(weights_np), 1.0, atol=1e-4)
    assert weights_np[0] > 0.0
    assert weights_np[-1] > 0.0
    assert weights_np[0] / weights_np[1] == pytest.approx(0.125, rel=1e-3)

    weights_torch = AtmosphereNormalizer.get_latitude_weights(torch.from_numpy(lats))
    assert isinstance(weights_torch, torch.Tensor)
    assert torch.isclose(weights_torch.mean(), torch.tensor(1.0), atol=1e-4)
    assert weights_torch[0] > 0.0


def test_atmosphere_zarr_dataset():
    sample_zarr = project_root / "data" / "dataset" / "era5_sample_0p5_26ch.zarr"
    if not sample_zarr.exists():
        pytest.skip("Sample dataset not found, skipping dataset test")

    # Mock statistics
    means = np.zeros((26,), dtype=np.float32)
    stds = np.ones((26,), dtype=np.float32)

    config = AtmosphereDatasetConfig(
        data_path=sample_zarr,
        means_path=means,
        stds_path=stds,
        history=0,
        rollout_steps=1,
        time_step=1,
        normalize=True,
    )

    dataset = AtmosphereZarrDataset(config, training=True)
    assert len(dataset) > 0

    sample = dataset[0]
    assert "input" in sample
    assert "target" in sample
    assert "time_index" in sample

    assert sample["input"].shape == (26, 361, 720)
    assert sample["target"].shape == (1, 26, 361, 720)

    # Test DataLoader
    loader = create_data_loader(dataset, batch_size=1, shuffle=False, num_workers=0)
    for batch in loader:
        assert batch["input"].shape == (1, 26, 361, 720)
        assert batch["target"].shape == (1, 1, 26, 361, 720)
        break


def test_loader_rejects_legacy_weatherbench_regridding(tmp_path: Path):
    path = tmp_path / "legacy.zarr"
    times = np.asarray(["2018-01-01T00", "2018-01-01T06"], dtype="datetime64[h]")
    state = xr.DataArray(
        np.zeros((2, 26, 3, 4), dtype=np.float32),
        dims=("time", "channel", "latitude", "longitude"),
        coords={
            "time": times,
            "channel": np.asarray(
                AtmosphereDatasetConfig(data_path=path).channel_names, dtype=str
            ),
            "latitude": [90.0, 0.0, -90.0],
            "longitude": [0.0, 90.0, 180.0, 270.0],
        },
        name="state",
    ).to_dataset()
    state.attrs["source"] = "gs://weatherbench2/datasets/era5/example.zarr"
    state.to_zarr(cast(Any, str(path)), mode="w", consolidated=True)
    with pytest.raises(ValueError, match="unaudited regridding version"):
        AtmosphereZarrDataset(
            AtmosphereDatasetConfig(data_path=path, normalize=False),
            training=False,
        )


def test_multistep_rollout_dataset():
    sample_zarr = project_root / "data" / "dataset" / "era5_sample_0p5_26ch.zarr"
    if not sample_zarr.exists():
        pytest.skip("Sample dataset not found, skipping dataset test")

    means = np.zeros((26,), dtype=np.float32)
    stds = np.ones((26,), dtype=np.float32)

    config = AtmosphereDatasetConfig(
        data_path=sample_zarr,
        means_path=means,
        stds_path=stds,
        history=1,  # 2 timesteps input [t-1, t]
        rollout_steps=2,  # 2 timesteps target [t+1, t+2]
        time_step=1,
        normalize=True,
    )

    dataset = AtmosphereZarrDataset(config, training=False)
    assert len(dataset) >= 1

    sample = dataset[0]
    assert sample["input"].shape == (52, 361, 720)
    assert sample["target"].shape == (2, 26, 361, 720)


def test_spatial_crop_exposes_matching_coordinates():
    sample_zarr = project_root / "data" / "dataset" / "era5_sample_0p5_26ch.zarr"
    if not sample_zarr.exists():
        pytest.skip("Sample dataset not found, skipping dataset test")

    config = AtmosphereDatasetConfig(
        data_path=sample_zarr,
        normalize=False,
        spatial_crop=(32, 64),
    )
    dataset = AtmosphereZarrDataset(config, training=False)
    assert dataset.spatial_shape == (32, 64)
    assert dataset.latitudes.shape == (32,)
    assert dataset.longitudes.shape == (64,)


def test_training_noise_only_changes_input_and_workers_are_restartable():
    sample_zarr = project_root / "data" / "dataset" / "era5_sample_0p5_26ch.zarr"
    if not sample_zarr.exists():
        pytest.skip("Sample dataset not found, skipping dataset test")
    config = AtmosphereDatasetConfig(
        data_path=sample_zarr,
        normalize=False,
        add_noise=True,
        noise_std=0.1,
        spatial_crop=(13, 24),
    )
    dataset = AtmosphereZarrDataset(config, training=True)
    first = dataset[0]
    second = dataset[0]
    assert not torch.equal(first["input"], second["input"])
    torch.testing.assert_close(first["target"], second["target"])

    loader = create_data_loader(dataset, num_workers=1, persistent_workers=False)
    assert loader.persistent_workers is False
    dataset.close()


def test_loader_exposes_asynchronous_prefetch_controls():
    sample_zarr = project_root / "data" / "dataset" / "era5_sample_0p5_26ch.zarr"
    if not sample_zarr.exists():
        pytest.skip("Sample dataset not found, skipping dataset test")
    dataset = AtmosphereZarrDataset(
        AtmosphereDatasetConfig(
            data_path=sample_zarr,
            normalize=False,
            spatial_crop=(13, 24),
        ),
        training=False,
    )
    loader = create_data_loader(
        dataset,
        num_workers=1,
        prefetch_factor=3,
        persistent_workers=True,
    )
    assert loader.prefetch_factor == 3
    assert loader.persistent_workers is True
    batch = next(iter(loader))
    assert batch["input"].shape == (1, 26, 13, 24)
    del loader
    dataset.close()


def test_nonpersistent_loader_reproduces_next_epoch_after_resume():
    dataset = TensorDataset(torch.arange(19))
    generator = torch.Generator().manual_seed(1234)
    loader = create_data_loader(
        dataset,  # type: ignore[arg-type]
        batch_size=4,
        shuffle=True,
        num_workers=1,
        persistent_workers=False,
        generator=generator,
    )
    list(loader)  # finish epoch one before saving its generator state
    saved_state = generator.get_state()
    uninterrupted = torch.cat([batch[0] for batch in loader]).tolist()

    resumed_generator = torch.Generator()
    resumed_generator.set_state(saved_state)
    resumed_loader = create_data_loader(
        dataset,  # type: ignore[arg-type]
        batch_size=4,
        shuffle=True,
        num_workers=1,
        persistent_workers=False,
        generator=resumed_generator,
    )
    resumed = torch.cat([batch[0] for batch in resumed_loader]).tolist()
    assert resumed == uninterrupted
