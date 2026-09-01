"""Streaming spherical metrics by forecast lead and atmospheric channel."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from neural_atmosphere_operator.data.normalization import latitude_cell_weights


@dataclass(slots=True)
class LeadMetricAccumulator:
    normalized_squared_error: float = 0.0
    normalized_absolute_error: float = 0.0
    normalized_weight: float = 0.0
    samples: int = 0
    physical_squared_error: np.ndarray | None = None
    channel_weight: np.ndarray | None = None
    channel_acc_sum: np.ndarray | None = None
    channel_acc_count: np.ndarray | None = None

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
        latitude = torch.as_tensor(
            latitudes, device=prediction.device, dtype=prediction.dtype
        )
        area = latitude_cell_weights(latitude).view(1, 1, -1, 1)
        weights = torch.broadcast_to(area, prediction.shape)
        error = prediction - target
        physical_error = error * channel_stds
        self.normalized_squared_error += float((error.square() * weights).sum())
        self.normalized_absolute_error += float((error.abs() * weights).sum())
        self.normalized_weight += float(weights.sum())

        channel_reductions = (0, 2, 3)
        physical_squared_error = (
            (physical_error.square() * weights)
            .sum(dim=channel_reductions)
            .double()
            .cpu()
            .numpy()
        )
        channel_weight = weights.sum(dim=channel_reductions).double().cpu().numpy()
        if normalized_climatology is None:
            raise ValueError(
                "normalized_climatology is required for scientifically valid ACC"
            )
        climatology = normalized_climatology.to(
            device=prediction.device, dtype=prediction.dtype
        )
        if climatology.ndim == 3:
            climatology = climatology.unsqueeze(0)
        if climatology.shape[1:] != prediction.shape[1:]:
            raise ValueError("Climatology must match prediction [C, H, W]")
        if climatology.shape[0] not in (1, prediction.shape[0]):
            raise ValueError("Climatology batch must be one or match prediction")
        prediction_anomaly = prediction - climatology
        target_anomaly = target - climatology
        # ACC is a spatial correlation for each forecast initialization and
        # channel.  Average those correlations afterwards; pooling anomaly
        # energies over the batch would incorrectly give energetic weather
        # states more influence than quiet ones.
        spatial_reductions = (2, 3)
        cross = (prediction_anomaly * target_anomaly * weights).sum(
            dim=spatial_reductions
        )
        pred_energy = (prediction_anomaly.square() * weights).sum(
            dim=spatial_reductions
        )
        target_energy = (target_anomaly.square() * weights).sum(
            dim=spatial_reductions
        )
        acc_denominator = torch.sqrt(pred_energy * target_energy)
        valid_acc = acc_denominator > 1e-12
        sample_acc = torch.where(
            valid_acc,
            cross / acc_denominator.clamp_min(1e-12),
            torch.zeros_like(cross),
        )
        acc_sum = sample_acc.sum(dim=0).double().cpu().numpy()
        acc_count = valid_acc.sum(dim=0).double().cpu().numpy()
        if self.physical_squared_error is None:
            self.physical_squared_error = np.zeros_like(physical_squared_error)
            self.channel_weight = np.zeros_like(channel_weight)
            self.channel_acc_sum = np.zeros_like(acc_sum)
            self.channel_acc_count = np.zeros_like(acc_count)
        assert self.channel_weight is not None
        assert self.channel_acc_sum is not None
        assert self.channel_acc_count is not None
        self.physical_squared_error += physical_squared_error
        self.channel_weight += channel_weight
        self.channel_acc_sum += acc_sum
        self.channel_acc_count += acc_count
        self.samples += prediction.shape[0]

    def _channel_acc(self) -> np.ndarray:
        if self.channel_acc_sum is None:
            return np.asarray([], dtype=np.float64)
        assert self.channel_acc_count is not None
        return np.divide(
            self.channel_acc_sum,
            self.channel_acc_count,
            out=np.full_like(self.channel_acc_sum, np.nan),
            where=self.channel_acc_count > 0,
        )

    def result(self, lead: int, time_step_hours: float = 6.0) -> dict[str, float | int]:
        denominator = max(self.normalized_weight, 1e-12)
        channel_acc = self._channel_acc()
        finite = channel_acc[np.isfinite(channel_acc)]
        normalized_mse = self.normalized_squared_error / denominator
        return {
            "lead": lead,
            "lead_hours": lead * time_step_hours,
            "normalized_mse": normalized_mse,
            "normalized_rmse": math.sqrt(normalized_mse),
            "normalized_mae": self.normalized_absolute_error / denominator,
            "normalized_acc": float(finite.mean()) if finite.size else float("nan"),
            "samples": self.samples,
        }

    def channel_results(
        self,
        lead: int,
        names: tuple[str, ...],
        time_step_hours: float = 6.0,
        units: tuple[str, ...] | None = None,
    ) -> list[dict[str, float | int | str]]:
        if self.physical_squared_error is None or self.channel_weight is None:
            return []
        if units is not None and len(units) != len(names):
            raise ValueError("units must align with channel names")
        acc = self._channel_acc()
        return [
            {
                "lead": lead,
                "lead_hours": lead * time_step_hours,
                "channel": channel,
                "variable": names[channel],
                "unit": units[channel] if units is not None else "",
                "physical_rmse": math.sqrt(
                    float(self.physical_squared_error[channel])
                    / max(float(self.channel_weight[channel]), 1e-12)
                ),
                "normalized_acc": float(acc[channel]),
            }
            for channel in range(len(names))
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
