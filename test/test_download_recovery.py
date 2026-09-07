"""Exercise downloader journal recovery and publication with local Zarr I/O."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from configs.download_data_config import WeatherBenchDownloadConfig
from data import download_data as downloader


@pytest.fixture
def download_case(tmp_path, monkeypatch):
    config = WeatherBenchDownloadConfig(
        start_date="2018-01-01",
        end_date="2018-01-02",
        download_batch_days=1,
        output_zarr_path=str(tmp_path / "result.zarr"),
    )
    original_open = xr.open_zarr

    def open_store(path, *args, **kwargs):
        if path == config.zarr_url:
            return xr.Dataset()
        return original_open(path, *args, **kwargs)

    def subset(batch, source):
        times = np.arange(
            np.datetime64(batch.start_date),
            np.datetime64(batch.end_date) + np.timedelta64(1, "D"),
            np.timedelta64(6, "h"),
        ).astype("datetime64[ns]")
        return xr.Dataset(
            {"state": ("time", np.arange(len(times), dtype=np.float32))},
            coords={"time": times},
            attrs={"regridding_version": downloader.REGRIDDING_VERSION},
        )

    monkeypatch.setattr(xr, "open_zarr", open_store)
    monkeypatch.setattr(downloader, "build_subset", subset)
    return config


@pytest.mark.parametrize("after_write", [False, True])
def test_first_batch_interruption_can_restart(download_case, monkeypatch, after_write):
    config = download_case
    original_write = xr.Dataset.to_zarr

    def interrupted(self, *args, **kwargs):
        if after_write:
            original_write(self, *args, **kwargs)
        raise OSError("injected interruption")

    with monkeypatch.context() as patch:
        patch.setattr(xr.Dataset, "to_zarr", interrupted)
        with pytest.raises(OSError, match="injected"):
            downloader.main(config)
    with pytest.raises(RuntimeError, match="contract"):
        downloader.main(replace(config, end_date="2018-01-03"))
    downloader.main(config)
    output = Path(config.output_zarr_path)
    with xr.open_zarr(str(output), consolidated=True) as ds:
        assert ds.sizes["time"] == 8
        np.testing.assert_array_equal(ds.state.values, [0, 1, 2, 3, 0, 1, 2, 3])
    assert not output.with_name(output.name + ".progress.json").exists()
    if after_write:
        assert len(list(output.parent.glob("*.interrupted-*"))) == 1


@pytest.mark.parametrize("indices", [[0, 1, 3], [0, 1, 1, 3], [1, 0, 2, 3]])
def test_incomplete_duplicate_or_reordered_source_is_rejected(
    download_case, monkeypatch, indices
):
    original = downloader.build_subset
    monkeypatch.setattr(
        downloader, "build_subset", lambda c, s: original(c, s).isel(time=indices)
    )
    with pytest.raises(ValueError, match="timestamps"):
        downloader.main(download_case)
    assert not Path(download_case.output_zarr_path).exists()


def test_final_stored_timestamps_are_verified(download_case, monkeypatch):
    original_write = xr.Dataset.to_zarr

    def corrupt(self, *args, **kwargs):
        return original_write(
            self.assign_coords(time=self.time.values + np.timedelta64(1, "h")),
            *args,
            **kwargs,
        )

    monkeypatch.setattr(xr.Dataset, "to_zarr", corrupt)
    with pytest.raises(ValueError, match="timestamps"):
        downloader.main(download_case)
    output = Path(download_case.output_zarr_path)
    assert not output.exists()
    assert output.with_name(output.name + ".partial").exists()
