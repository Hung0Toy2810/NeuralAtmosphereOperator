"""Streaming spherical metrics by forecast lead and atmospheric channel."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from neural_atmosphere_operator.models.loss import _normalized_latitude_weights


@dataclass(slots=True)
class LeadMetricAccumulator:
    """Keep small channel reductions on-device until a report is requested."""

    samples: int = 0
    _totals: Tensor | None = None
    _host: np.ndarray | None = None

    @torch.no_grad()
    def update(
        self,
        prediction: Tensor,
        target: Tensor,
        channel_stds: Tensor,
        latitudes: Tensor | np.ndarray,
        normalized_climatology: Tensor | None = None,
    ) -> None:
        if prediction.shape != target.shape or prediction.ndim != 4:
            raise ValueError("prediction and target must share [B, C, H, W]")
        if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
            raise FloatingPointError(
                "Metrics require finite prediction and target fields"
            )
        if not torch.isfinite(channel_stds).all() or (channel_stds <= 0).any():
            raise ValueError("Channel standard deviations must be finite and positive")
        if normalized_climatology is None:
            raise ValueError(
                "normalized_climatology is required for scientifically valid ACC"
            )
        climate = normalized_climatology.to(prediction)
        if climate.ndim == 3:
            climate = climate.unsqueeze(0)
        if climate.shape[1:] != prediction.shape[1:] or climate.shape[0] not in (
            1,
            prediction.shape[0],
        ):
            raise ValueError("Climatology must match prediction [C, H, W] and batch")
        area = _normalized_latitude_weights(prediction, latitudes, 1e-8)
        weights = torch.broadcast_to(area, prediction.shape)
        error = prediction - target
        physical_error = error * channel_stds
        a, b = prediction - climate, target - climate
        cross = (a * b * weights).sum(dim=(2, 3))
        denominator = torch.sqrt(
            (a.square() * weights).sum(dim=(2, 3))
            * (b.square() * weights).sum(dim=(2, 3))
        )
        valid = denominator > 1e-12
        sample_acc = torch.where(
            valid, cross / denominator.clamp_min(1e-12), torch.zeros_like(cross)
        )
        spatial_weight = weights.sum(dim=(2, 3), keepdim=True)
        pred_mean = (prediction * weights).sum(
            dim=(2, 3), keepdim=True
        ) / spatial_weight
        truth_mean = (target * weights).sum(dim=(2, 3), keepdim=True) / spatial_weight
        # ACC remains a mean of per-initialization spatial correlations, never
        # a correlation pooled across weather cases.
        reductions = torch.stack(
            [
                (error.square() * weights).sum(dim=(0, 2, 3)),
                (error.abs() * weights).sum(dim=(0, 2, 3)),
                weights.sum(dim=(0, 2, 3)),
                (physical_error.square() * weights).sum(dim=(0, 2, 3)),
                sample_acc.sum(dim=0),
                valid.sum(dim=0).to(prediction.dtype),
                (physical_error * weights).sum(dim=(0, 2, 3)),
                ((prediction - pred_mean).square() * weights).sum(dim=(0, 2, 3)),
                ((target - truth_mean).square() * weights).sum(dim=(0, 2, 3)),
            ]
        )
        # MPS does not support float64. CPU/CUDA accumulate small reductions
        # in float64, preserving the established per-initialization averaging.
        reductions = reductions.to(
            torch.float32 if prediction.device.type == "mps" else torch.float64
        )
        if self._totals is None:
            self._totals = torch.zeros_like(reductions)
        if (
            self._totals.shape != reductions.shape
            or self._totals.device != reductions.device
        ):
            raise ValueError("Accumulator channel count/device must remain fixed")
        self._totals += reductions
        self.samples += prediction.shape[0]
        self._host = None

    def _snapshot(self) -> np.ndarray:
        if self.samples < 1 or self._totals is None:
            raise ValueError("Cannot report metrics for an empty evaluation")
        if self._host is None:
            self._host = self._totals.cpu().numpy().astype(np.float64)
        if not np.isfinite(self._host).all():
            raise FloatingPointError("Metric reductions overflowed")
        return self._host

    def _channel_acc(self) -> np.ndarray:
        data = self._snapshot()
        return np.divide(
            data[4], data[5], out=np.full_like(data[4], np.nan), where=data[5] > 0
        )

    def result(self, lead: int, time_step_hours: float = 6.0) -> dict[str, float | int]:
        data = self._snapshot()
        weight = float(data[2].sum())
        acc = self._channel_acc()
        finite = acc[np.isfinite(acc)]
        mse = float(data[0].sum()) / weight
        return {
            "lead": lead,
            "lead_hours": lead * time_step_hours,
            "normalized_mse": mse,
            "normalized_rmse": math.sqrt(mse),
            "normalized_mae": float(data[1].sum()) / weight,
            "normalized_acc": float(finite.mean()) if finite.size else float("nan"),
            "samples": self.samples,
            "acc_valid_channels": int(finite.size),
            "acc_total_channels": int(acc.size),
        }

    def channel_results(
        self,
        lead: int,
        names: tuple[str, ...],
        time_step_hours: float = 6.0,
        units: tuple[str, ...] | None = None,
    ) -> list[dict[str, float | int | str]]:
        data = self._snapshot()
        if len(names) != data.shape[1] or (
            units is not None and len(units) != len(names)
        ):
            raise ValueError("names and units must align with channel count")
        acc = self._channel_acc()
        return [
            {
                "lead": lead,
                "lead_hours": lead * time_step_hours,
                "channel": channel,
                "variable": name,
                "unit": units[channel] if units else "",
                "physical_rmse": math.sqrt(float(data[3, channel] / data[2, channel])),
                "normalized_acc": float(acc[channel]),
                "acc_valid_initializations": int(data[5, channel]),
                "physical_bias": float(data[6, channel] / data[2, channel]),
                "spatial_variance_ratio": float(data[7, channel] / data[8, channel])
                if data[8, channel] > 0
                else float("nan"),
            }
            for channel, name in enumerate(names)
        ]


def channel_stds_tensor(stds: np.ndarray, device: torch.device) -> Tensor:
    values = np.squeeze(np.asarray(stds, dtype=np.float32))
    if values.ndim != 1:
        raise ValueError("stds must be a channel vector")
    return torch.as_tensor(values, device=device).view(1, -1, 1, 1)


def normalized_climatology_tensor(
    climatology: np.ndarray,
    means: np.ndarray,
    stds: np.ndarray,
    device: torch.device,
) -> Tensor:
    """Convert physical training climatology to normalized ``[1,C,H,W]``."""
    climate = np.asarray(climatology, dtype=np.float32)
    mean_values = np.squeeze(np.asarray(means, dtype=np.float32))
    std_values = np.squeeze(np.asarray(stds, dtype=np.float32))
    if climate.ndim != 3:
        raise ValueError("climatology must have shape [C, H, W]")
    if (
        mean_values.ndim != 1
        or std_values.ndim != 1
        or climate.shape[0] != mean_values.size
        or mean_values.shape != std_values.shape
    ):
        raise ValueError("Climatology and channel statistics are incompatible")
    if not np.all(np.isfinite(climate)):
        raise ValueError("Climatology must be finite")
    normalized = (climate - mean_values[:, None, None]) / std_values[:, None, None]
    return torch.as_tensor(normalized, device=device).unsqueeze(0)
