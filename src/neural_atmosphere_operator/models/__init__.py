"""Models and loss functions for atmospheric forecasting."""

from .loss import (
    ChannelRelativeAtmosphereLoss,
    CombinedAtmosphereLoss,
    LossScaler,
    RolloutLoss,
    SpectralLoss,
    latitude_weighted_l1,
    latitude_weighted_mse,
    makani_auto_channel_weights,
)
from .model import AtmosphereNeuralOperator

__all__ = [
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
