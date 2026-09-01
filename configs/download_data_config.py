"""Configuration for the 12-year, 0.5-degree ERA5 training corpus."""

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

# WB2-native approximation of the compact 26-channel SFNO set. WeatherBench2
# does not contain the paper's 100-m winds, so q1000/q850 supply lower-
# tropospheric moisture instead. Each variable has its own selected levels.
DEFAULT_SURFACE_VARIABLES: tuple[str, ...] = (
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "2m_temperature",
    "surface_pressure",
    "mean_sea_level_pressure",
    "total_column_water_vapour",
)
DEFAULT_LEVEL_SELECTIONS: tuple[tuple[str, tuple[int, ...]], ...] = (
    ("geopotential", (1000, 850, 500, 250, 50)),
    ("u_component_of_wind", (1000, 850, 500, 250)),
    ("v_component_of_wind", (1000, 850, 500, 250)),
    ("temperature", (850, 500, 250, 100)),
    ("specific_humidity", (1000, 850)),
    ("relative_humidity", (500,)),
)


def channel_names(
    surface_variables: tuple[str, ...] = DEFAULT_SURFACE_VARIABLES,
    level_selections: tuple[tuple[str, tuple[int, ...]], ...] = DEFAULT_LEVEL_SELECTIONS,
) -> tuple[str, ...]:
    """Return the canonical flattened state-channel order."""
    return surface_variables + tuple(
        f"{variable}@{level}hPa"
        for variable, levels in level_selections
        for level in levels
    )


@dataclass(frozen=True, slots=True)
class WeatherBenchDownloadConfig:
    """Select, conservatively regrid and store an ERA5 state tensor."""

    zarr_url: str = DEFAULT_ZARR_URL
    surface_variables: tuple[str, ...] = DEFAULT_SURFACE_VARIABLES
    level_selections: tuple[tuple[str, tuple[int, ...]], ...] = DEFAULT_LEVEL_SELECTIONS
    start_date: str = "2007-01-01"
    # Train: 2007-2018; validation: 2019-01-01..16; test: 2019-01-17..02-16.
    # Sixteen validation days provide four independent UTC phases for a
    # 60-step (15-day) rollout.  The 31-day test range similarly provides the
    # initial state plus 120 six-hour targets for a complete 30-day rollout.
    end_date: str = "2019-02-16"
    time_stride_hours: int = 6
    target_resolution_degrees: float = 0.5
    download_batch_days: int = 7
    output_zarr_path: str = "data/dataset/era5_2007_2019_0p5deg_26ch.zarr"

    def __post_init__(self) -> None:
        if not self.surface_variables and not self.level_selections:
            raise ValueError("at least one channel must be selected")
        variables = [name for name, _ in self.level_selections]
        if len(variables) != len(set(variables)):
            raise ValueError("level variables must not be repeated")
        for variable, levels in self.level_selections:
            if not variable or not levels:
                raise ValueError("each level variable requires at least one level")
            if len(levels) != len(set(levels)):
                raise ValueError(f"duplicate pressure level for {variable}")
            unknown = set(levels) - set(AVAILABLE_PRESSURE_LEVELS)
            if unknown:
                raise ValueError(f"levels not in wb13 for {variable}: {sorted(unknown)}")
        if self.start_date > self.end_date:
            raise ValueError("start_date must be before or equal to end_date")
        if self.time_stride_hours < 6 or self.time_stride_hours % 6:
            raise ValueError("time_stride_hours must be a positive multiple of 6")
        if self.target_resolution_degrees != 0.5:
            raise ValueError("the validated downloader currently supports 0.5 degrees")
        if self.download_batch_days < 1:
            raise ValueError("download_batch_days must be positive")

    @property
    def channel_names(self) -> tuple[str, ...]:
        return channel_names(self.surface_variables, self.level_selections)

    @property
    def channel_count(self) -> int:
        return len(self.channel_names)
