"""Loss functions for atmospheric forecasting models.

Atmospheric fields cover the entire globe without coastline/land-mask discontinuities.
Area weighting is derived from spherical latitude-cell boundaries.
"""

from __future__ import annotations

from typing import Literal, Sequence, cast

import torch
from torch import Tensor, nn
from torch.autograd import Function

from neural_atmosphere_operator.data.normalization import latitude_cell_weights

_MAKANI_SURFACE_CHANNEL_WEIGHTS = {
    "10m_u_component_of_wind": 0.1,
    "10m_v_component_of_wind": 0.1,
    "100m_u_component_of_wind": 0.1,
    "100m_v_component_of_wind": 0.1,
    "2m_temperature": 1.0,
    "2m_dewpoint_temperature": 1.0,
    "total_precipitation": 0.1,
    "surface_pressure": 0.1,
    "mean_sea_level_pressure": 0.1,
    "total_column_water_vapour": 0.1,
    "sea_surface_temperature": 0.1,
}

_MAKANI_PRESSURE_VARIABLES = {
    "geopotential",
    "specific_humidity",
    "relative_humidity",
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
}


def makani_auto_channel_weights(
    channel_names: Sequence[str],
    *,
    normalization_stds: Tensor | Sequence[float] | None = None,
    time_difference_stds: Tensor | Sequence[float] | None = None,
    eps: float = 1e-6,
) -> Tensor:
    """Return Makani's atmospheric channel weights in local channel order.

    Pressure-level fields receive ``0.001 * pressure_hPa``. Surface winds and
    pressure-like fields receive 0.1, while 2 m temperature receives 1.0.
    The base weights are normalized to sum to one. When time-difference
    statistics are supplied, the result is additionally multiplied by
    ``normalization_std / time_difference_std``, matching Makani's
    ``temp_diff_normalization`` for z-score-normalized model states.
    """
    if not channel_names:
        raise ValueError("channel_names cannot be empty")
    if eps <= 0:
        raise ValueError("eps must be positive")

    values: list[float] = []
    for name in channel_names:
        if name in _MAKANI_SURFACE_CHANNEL_WEIGHTS:
            values.append(_MAKANI_SURFACE_CHANNEL_WEIGHTS[name])
            continue
        if "@" in name:
            variable, level_label = name.rsplit("@", 1)
            if variable in _MAKANI_PRESSURE_VARIABLES and level_label.endswith("hPa"):
                try:
                    pressure_level = float(level_label[:-3])
                except ValueError as error:
                    raise ValueError(
                        f"Invalid pressure-level channel {name!r}"
                    ) from error
                if pressure_level <= 0:
                    raise ValueError(f"Pressure level must be positive in {name!r}")
                values.append(0.001 * pressure_level)
                continue
        # This is Makani's fallback for an otherwise unclassified channel.
        values.append(0.01)

    weights = torch.tensor(values, dtype=torch.float32)
    weights = weights / weights.sum()
    if (normalization_stds is None) != (time_difference_stds is None):
        raise ValueError(
            "normalization_stds and time_difference_stds must be supplied together"
        )
    if normalization_stds is not None and time_difference_stds is not None:
        global_stds = torch.as_tensor(normalization_stds, dtype=torch.float32).flatten()
        delta_stds = torch.as_tensor(
            time_difference_stds, dtype=torch.float32
        ).flatten()
        expected = len(channel_names)
        if global_stds.numel() != expected or delta_stds.numel() != expected:
            raise ValueError(
                "Temporal-difference statistics must match the channel count "
                f"{expected}"
            )
        if (
            not torch.isfinite(global_stds).all()
            or not torch.isfinite(delta_stds).all()
        ):
            raise ValueError("Channel standard deviations must be finite")
        if (global_stds <= 0).any() or (delta_stds <= 0).any():
            raise ValueError("Channel standard deviations must be positive")
        weights = weights * global_stds / delta_stds.clamp_min(eps)
    return weights


def _normalized_latitude_weights(
    prediction: Tensor,
    latitudes: Tensor | Sequence[float] | None,
    eps: float,
) -> Tensor:
    """Build validated unit-mean area weights of shape ``[1, 1, H, 1]``."""
    height = prediction.shape[-2]
    # MPS does not implement float64 tensors. Float32 is amply precise for the
    # latitude coordinates; the exact cell-boundary calculation itself still
    # runs in NumPy float64 inside latitude_cell_weights.
    coordinate_dtype = (
        torch.float32 if prediction.device.type == "mps" else torch.float64
    )
    if latitudes is None:
        coordinate = torch.linspace(
            90.0,
            -90.0,
            height,
            device=prediction.device,
            dtype=coordinate_dtype,
        )
    elif isinstance(latitudes, Tensor):
        coordinate = latitudes.to(device=prediction.device)
    else:
        coordinate = torch.as_tensor(
            latitudes, device=prediction.device, dtype=coordinate_dtype
        )

    if coordinate.ndim != 1 or coordinate.numel() != height:
        raise ValueError(f"latitudes must be one-dimensional with length {height}")
    weights = latitude_cell_weights(coordinate).to(
        device=prediction.device, dtype=prediction.dtype
    )
    return (weights / weights.mean().clamp_min(eps)).view(1, 1, height, 1)


def latitude_weighted_mse(
    prediction: Tensor,
    target: Tensor,
    latitudes: Tensor | Sequence[float] | None = None,
    *,
    channel_weights: Tensor | Sequence[float] | None = None,
    eps: float = 1e-8,
) -> Tensor:
    """Area-weighted Mean Squared Error on spherical equiangular grid.

    Args:
        prediction: Predicted field of shape ``[B, C, H, W]``.
        target: Target field of shape ``[B, C, H, W]``.
        latitudes: Optional 1D latitude coordinates of length ``H``. If None,
            interpolates linearly from +90 to -90 degrees.
        eps: Small constant for numerical stability.

    Returns:
        Scalar area-weighted MSE loss.
    """
    if prediction.shape != target.shape or prediction.ndim != 4:
        raise ValueError(
            "prediction and target must share shape [B, C, H, W], got "
            f"{prediction.shape} vs {target.shape}"
        )

    weights = _normalized_latitude_weights(prediction, latitudes, eps)
    squared_error = (prediction - target).square()
    per_channel = (squared_error * weights).mean(dim=(0, 2, 3))
    if channel_weights is None:
        return per_channel.mean()
    channel_weight_tensor = torch.as_tensor(
        channel_weights, device=prediction.device, dtype=prediction.dtype
    ).flatten()
    if channel_weight_tensor.numel() != prediction.shape[1]:
        raise ValueError(
            "channel_weights must contain exactly " f"{prediction.shape[1]} values"
        )
    if (
        not torch.isfinite(channel_weight_tensor).all()
        or (channel_weight_tensor < 0).any()
    ):
        raise ValueError("channel_weights must be finite and non-negative")
    if channel_weight_tensor.sum() <= 0:
        raise ValueError("channel_weights must contain a positive value")
    return torch.sum(per_channel * channel_weight_tensor)


def latitude_weighted_l1(
    prediction: Tensor,
    target: Tensor,
    latitudes: Tensor | Sequence[float] | None = None,
    *,
    eps: float = 1e-8,
) -> Tensor:
    """Area-weighted L1 Error on spherical equiangular grid."""
    if prediction.shape != target.shape or prediction.ndim != 4:
        raise ValueError(
            "prediction and target must share shape [B, C, H, W], got "
            f"{prediction.shape} vs {target.shape}"
        )

    weights = _normalized_latitude_weights(prediction, latitudes, eps)
    abs_error = (prediction - target).abs()
    return (abs_error * weights).mean()


class ChannelRelativeAtmosphereLoss(nn.Module):
    """Per-channel relative L2 loss for unmasked atmospheric fields.

    Calculates ``||pred_c - tar_c||_2 / ||tar_c||_2`` per channel, keeping
    different physical fields equally weighted during optimization.
    """

    def __init__(
        self,
        eps: float = 1e-8,
        latitudes: Tensor | Sequence[float] | None = None,
    ) -> None:
        super().__init__()
        if eps <= 0:
            raise ValueError("eps must be positive")
        self.eps = eps
        self.latitudes = latitudes

    def forward(self, prediction: Tensor, target: Tensor) -> tuple[Tensor, Tensor]:
        """Return ``(scalar_loss, per_channel_loss)``."""
        if prediction.shape != target.shape or prediction.ndim != 4:
            raise ValueError("prediction and target must share shape [B, C, H, W]")

        weights = _normalized_latitude_weights(prediction, self.latitudes, self.eps)
        diff_norm = torch.sqrt(
            ((prediction - target).square() * weights).sum(dim=(-2, -1))
        )
        tar_norm = torch.sqrt((target.square() * weights).sum(dim=(-2, -1))).clamp_min(
            self.eps
        )
        rel = diff_norm / tar_norm

        per_channel = rel.mean(dim=0)
        scalar = per_channel.mean()
        return scalar, per_channel


class SpectralLoss(nn.Module):
    """Optional Fourier-coefficient discrepancy on the planar lat-lon grid.

    This is not part of the standard FourCastNet training objective and does
    not isolate high frequencies.  It is retained as an explicit experimental
    regularizer; the default atmospheric objective leaves it disabled.
    """

    def __init__(self, loss_type: Literal["l1", "l2"] = "l1") -> None:
        super().__init__()
        if loss_type not in ("l1", "l2"):
            raise ValueError("loss_type must be 'l1' or 'l2'")
        self.loss_type = loss_type

    def forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        if prediction.shape != target.shape or prediction.ndim != 4:
            raise ValueError("prediction and target must share shape [B, C, H, W]")

        pred_fft = torch.fft.rfft2(prediction, norm="ortho")
        tar_fft = torch.fft.rfft2(target, norm="ortho")

        difference = (pred_fft - tar_fft).abs()
        # rfft stores only non-negative longitudinal frequencies.  Count the
        # omitted conjugate coefficients so the reduction represents the full
        # 2D spectrum; for L2 this also preserves Parseval equivalence.
        width = prediction.shape[-1]
        multiplicity = torch.full(
            (difference.shape[-1],),
            2.0,
            device=prediction.device,
            dtype=difference.dtype,
        )
        multiplicity[0] = 1.0
        if width % 2 == 0:
            multiplicity[-1] = 1.0
        coefficient_error = (
            difference if self.loss_type == "l1" else difference.square()
        )
        per_field = (coefficient_error * multiplicity.view(1, 1, 1, -1)).sum(
            dim=(-2, -1)
        ) / (prediction.shape[-2] * width)
        return per_field.mean()


class _ChannelGradientScaleFunction(Function):
    """Identity forward with per-sample, per-channel gradient balancing."""

    @staticmethod
    def forward(ctx, values: Tensor, eps: float) -> Tensor:  # type: ignore[override]
        if values.ndim != 4:
            raise ValueError("LossScaler expects input shaped [B, C, H, W]")
        ctx.eps = eps
        return values

    @staticmethod
    def backward(ctx, gradients: Tensor) -> tuple[Tensor, None]:  # type: ignore[override]
        channel_count = gradients.shape[1]
        inverse_norm = (
            gradients.norm(p=2, dim=(-2, -1), keepdim=True)
            .clamp_min(ctx.eps)
            .reciprocal()
        )
        normalized = inverse_norm / inverse_norm.sum(dim=1, keepdim=True).clamp_min(
            ctx.eps
        )
        return channel_count * normalized * gradients, None


class LossScaler(nn.Module):
    """Optional NeuralOceanOperator/MetNet-style gradient re-balancing.

    The forward value is unchanged. During backward, each output channel is
    scaled inversely to its spatial gradient norm. It is experimental for this
    atmospheric recipe and therefore disabled by default; Makani's explicit
    channel and temporal-difference weights remain the primary mechanism.
    """

    def __init__(self, eps: float = 1e-5) -> None:
        super().__init__()
        if eps <= 0:
            raise ValueError("eps must be positive")
        self.eps = eps

    def forward(self, values: Tensor) -> Tensor:
        return _ChannelGradientScaleFunction.apply(values, self.eps)


class CombinedAtmosphereLoss(nn.Module):
    """Latitude-area-weighted MSE with an optional spectral regularizer."""

    def __init__(
        self,
        spectral_weight: float = 0.0,
        latitudes: Tensor | Sequence[float] | None = None,
        channel_weights: Tensor | Sequence[float] | None = None,
        use_loss_scaler: bool = False,
    ) -> None:
        super().__init__()
        if spectral_weight < 0:
            raise ValueError("spectral_weight must be non-negative")
        self.spectral_weight = spectral_weight
        self.latitudes = latitudes
        weights = (
            None
            if channel_weights is None
            else torch.as_tensor(channel_weights, dtype=torch.float32).flatten()
        )
        self.register_buffer("channel_weights", weights)
        self.loss_scaler = LossScaler() if use_loss_scaler else None
        self.spectral_loss = (
            SpectralLoss(loss_type="l1") if spectral_weight > 0.0 else None
        )

    def forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        scaled_prediction = (
            self.loss_scaler(prediction) if self.loss_scaler is not None else prediction
        )
        spatial_mse = latitude_weighted_mse(
            scaled_prediction,
            target,
            self.latitudes,
            channel_weights=cast(Tensor | None, self.channel_weights),
        )
        if self.spectral_loss is not None and self.spectral_weight > 0.0:
            spec = self.spectral_loss(prediction, target)
            return spatial_mse + self.spectral_weight * spec
        return spatial_mse


class RolloutLoss(nn.Module):
    """Multi-step autoregressive rollout loss with discount factor."""

    def __init__(
        self,
        base_loss: nn.Module | None = None,
        discount_factor: float = 0.9,
        time_dim: Literal[0, 1] = 1,
    ) -> None:
        super().__init__()
        if not 0.0 <= discount_factor <= 1.0:
            raise ValueError("discount_factor must be in [0, 1]")
        if time_dim not in (0, 1):
            raise ValueError("time_dim must be 0 or 1")
        self.base_loss = (
            base_loss if base_loss is not None else CombinedAtmosphereLoss()
        )
        self.discount_factor = discount_factor
        self.time_dim = time_dim

    def forward(
        self, predictions: Sequence[Tensor] | Tensor, targets: Tensor
    ) -> Tensor:
        """Compute discounted multi-step loss.

        Args:
            predictions: A 5D tensor, or a list of ``K`` tensors shaped
                ``[B, C, H, W]``.
            targets: A 5D tensor with the same layout. ``time_dim`` declares
                whether its rollout axis is 0 (``[K, B, ...]``) or 1
                (``[B, K, ...]``), avoiding ambiguous shape inference when
                batch size equals rollout length.
        """
        if targets.ndim != 5:
            raise ValueError("targets must be a 5D rollout tensor")

        tar_list = list(targets.unbind(dim=self.time_dim))
        if isinstance(predictions, Tensor):
            if predictions.ndim != 5 or predictions.shape != targets.shape:
                raise ValueError("tensor predictions and targets must share a 5D shape")
            pred_list = list(predictions.unbind(dim=self.time_dim))
        else:
            pred_list = list(predictions)

        if not pred_list or len(pred_list) != len(tar_list):
            raise ValueError(
                "predictions and targets must contain the same non-zero steps"
            )
        for prediction, target in zip(pred_list, tar_list):
            if prediction.shape != target.shape or prediction.ndim != 4:
                raise ValueError("each rollout step must share shape [B, C, H, W]")

        k_steps = len(pred_list)

        total_loss = torch.tensor(
            0.0, device=pred_list[0].device, dtype=pred_list[0].dtype
        )
        total_weight = 0.0

        for k in range(k_steps):
            weight = self.discount_factor**k
            step_loss = self.base_loss(pred_list[k], tar_list[k])
            total_loss = total_loss + weight * step_loss
            total_weight += weight

        return total_loss / max(total_weight, 1e-8)
