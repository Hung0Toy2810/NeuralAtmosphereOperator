"""Configuration for the 24-year, 0.5-degree ERA5 training corpus."""

from __future__ import annotations

from dataclasses import dataclass

# Official 0.25-degree, 13-level, 6-hourly WeatherBench2 ERA5 archive. The
# downloader conservatively regrids this source to the configured target grid.
DEFAULT_ZARR_URL = (
    "gs://weatherbench2/datasets/era5/"
    "1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr"
)

AVAILABLE_PRESSURE_LEVELS: tuple[int, ...] = (
    50,
    100,
    150,
    200,
    250,
    300,
    400,
    500,
    600,
    700,
    850,
    925,
    1000,
)

# All channels of the SFNO 73-channel state available in this WB2 archive:
# six surface fields plus five variables on all thirteen pressure levels.
# The source has no 100-m u/v winds, so this is a 71-channel state. Derived
# diagnostics and static fields are not additional prognostic channels.
SURFACE_VARIABLES: tuple[str, ...] = (
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "2m_temperature",
    "surface_pressure",
    "mean_sea_level_pressure",
    "total_column_water_vapour",
)
PRESSURE_VARIABLES: tuple[str, ...] = (
    "u_component_of_wind",
    "v_component_of_wind",
    "geopotential",
    "temperature",
    "specific_humidity",
)


def channel_names() -> tuple[str, ...]:
    """Return the immutable flattened state-channel order."""
    return SURFACE_VARIABLES + tuple(
        f"{variable}@{level}hPa"
        for variable in PRESSURE_VARIABLES
        for level in AVAILABLE_PRESSURE_LEVELS
    )


DEFAULT_CHANNEL_COUNT = len(channel_names())
DEFAULT_DATASET_FILENAME = f"era5_1995_2020_0p5deg_{DEFAULT_CHANNEL_COUNT}ch.zarr"
DEFAULT_SAMPLE_FILENAME = f"era5_sample_0p5_{DEFAULT_CHANNEL_COUNT}ch.zarr"


@dataclass(frozen=True, slots=True)
class WeatherBenchDownloadConfig:
    """Conservatively regrid and store the fixed 71-channel ERA5 state."""

    zarr_url: str = DEFAULT_ZARR_URL
    start_date: str = "1995-01-01"
    # Full, disjoint years for seasonal validation and held-out testing.
    # Training remains 1995-2018; validation 2019; untouched test 2020.
    end_date: str = "2020-12-31"
    time_stride_hours: int = 6
    target_resolution_degrees: float = 0.5
    download_batch_days: int = 7
    output_zarr_path: str = f"data/dataset/{DEFAULT_DATASET_FILENAME}"

    def __post_init__(self) -> None:
        if self.start_date > self.end_date:
            raise ValueError("start_date must be before or equal to end_date")
        if self.time_stride_hours < 6 or self.time_stride_hours % 6:
            raise ValueError("time_stride_hours must be a positive multiple of 6")
        if (self.download_batch_days * 24) % self.time_stride_hours:
            raise ValueError(
                "download batch duration must be divisible by time_stride_hours to preserve cadence phase"
            )
        if self.target_resolution_degrees != 0.5:
            raise ValueError("the validated downloader currently supports 0.5 degrees")
        if self.download_batch_days < 1:
            raise ValueError("download_batch_days must be positive")

    @property
    def channel_names(self) -> tuple[str, ...]:
        return channel_names()

    @property
    def channel_count(self) -> int:
        return len(self.channel_names)
