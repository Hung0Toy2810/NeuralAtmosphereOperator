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
    StandardizedTendencyLoss,
    graphcast_channel_weights,
    latitude_weighted_l1,
    latitude_weighted_mse,
    makani_auto_channel_weights,
    standardized_tendency_scale,
)

__version__ = "0.1.0"

__all__ = [
    "AtmosphereDatasetConfig",
    "AtmosphereNeuralOperator",
    "AtmosphereNormalizer",
    "AtmosphereSample",
    "AtmosphereZarrDataset",
    "ChannelRelativeAtmosphereLoss",
    "CombinedAtmosphereLoss",
    "LossScaler",
    "RolloutLoss",
    "SpectralLoss",
    "StandardizedTendencyLoss",
    "create_data_loader",
    "graphcast_channel_weights",
    "latitude_weighted_l1",
    "latitude_weighted_mse",
    "makani_auto_channel_weights",
    "standardized_tendency_scale",
]
