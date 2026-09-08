"""Shared defaults for the atmospheric training and evaluation pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True, slots=True)
class DataPaths:
    """Conventional filesystem layout used by pipeline commands."""

    root: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "dataset")

    @property
    def dataset(self) -> Path:
        return self.root / "era5_1995_2020_0p5deg_26ch.zarr"

    @property
    def train(self) -> Path:
        return self.dataset

    @property
    def valid(self) -> Path:
        return self.dataset

    @property
    def test(self) -> Path:
        return self.dataset

    def split_time_range(self, split: str) -> tuple[str, str]:
        """Return non-overlapping inclusive logical ranges in the single store."""
        ranges = {
            "train": ("1995-01-01", "2018-12-31"),
            "valid": ("2019-01-01", "2019-12-31"),
            "test": ("2020-01-01", "2020-12-31"),
        }
        try:
            return ranges[split]
        except KeyError as error:
            raise ValueError(f"Unknown split: {split}") from error

    @property
    def means(self) -> Path:
        return self.root / "stats" / "means.npy"

    @property
    def stds(self) -> Path:
        return self.root / "stats" / "stds.npy"

    @property
    def climatology(self) -> Path:
        return self.root / "stats" / "time_means.npy"

    def time_diff_means(self, time_step: int = 1) -> Path:
        return self.root / "stats" / f"time_diff_means_dt{time_step}.npy"

    def time_diff_stds(self, time_step: int = 1) -> Path:
        return self.root / "stats" / f"time_diff_stds_dt{time_step}.npy"


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Conservative defaults for the 24-year 0.5-degree SFNO corpus."""

    # A finite initial budget; zero remains an explicit opt-in for unlimited runs.
    epochs: int = 25
    # An infinite run still needs a finite cosine-decay horizon. After this
    # many epochs the scheduler remains at min_learning_rate.
    scheduler_epochs: int = 25
    batch_size: int = 4
    validation_batch_size: int = 1
    gradient_accumulation: int = 2
    learning_rate: float = 5e-4
    weight_decay: float = 1e-4
    min_learning_rate: float = 1e-6
    warmup_epochs: int = 3
    warmup_start_factor: float = 0.1
    gradient_clip: float = 1.0
    rollout_steps: int = 1
    history: int = 0
    time_step: int = 1
    num_workers: int = 4
    prefetch_factor: int = 2
    # Training workers are recreated at epoch boundaries so checkpointed
    # sampler/worker RNG state produces the same next epoch after a restart.
    persistent_workers: bool = False
    # Bound spend once the stage-matched validation objective stops improving.
    patience: int = 5
    early_stopping_min_delta: float = 0.0
    rollout_discount: float = 1.0
    channel_weighting: str = "graphcast"
    input_noise_std: float = 0.0
    gradient_checkpointing: bool = False
    seed: int = 42
    amp_dtype: str = "bfloat16"
