"""Neural Atmosphere Operator for data-driven weather forecasting."""

from .data import (
    AtmosphereDatasetConfig,
    AtmosphereNormalizer,
    AtmosphereSample,
    AtmosphereZarrDataset,
    create_data_loader,
)
from .models import (
    AtmosphereNeuralOperator,
    ChannelRelativeAtmosphereLoss,
    CombinedAtmosphereLoss,
    LossScaler,
    RolloutLoss,
    SpectralLoss,
    latitude_weighted_l1,
    latitude_weighted_mse,
    makani_auto_channel_weights,
)

__version__ = "0.1.0"

__all__ = [
    "AtmosphereDatasetConfig",
    "AtmosphereNormalizer",
    "AtmosphereSample",
    "AtmosphereZarrDataset",
    "create_data_loader",
    "AtmosphereNeuralOperator",
    "ChannelRelativeAtmosphereLoss",
    "CombinedAtmosphereLoss",
    "LossScaler",
    "RolloutLoss",
    "SpectralLoss",
    "latitude_weighted_l1",
    "latitude_weighted_mse",
    "makani_auto_channel_weights",
]
