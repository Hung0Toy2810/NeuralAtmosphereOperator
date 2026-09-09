"""Per-channel normalization and spherical latitude weighting for atmospheric fields."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence, overload

import numpy as np
import torch
from torch import Tensor


def _validate_latitudes_np(latitudes: np.ndarray) -> np.ndarray:
    """Return a validated one-dimensional floating-point latitude vector."""
    values = np.asarray(latitudes, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("latitudes must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(values)):
        raise ValueError("latitudes must be finite")
    if np.any(np.abs(values) > 90.0):
        raise ValueError("latitudes must lie in [-90, 90] degrees")
    if values.size > 1:
        differences = np.diff(values)
        if not (np.all(differences > 0.0) or np.all(differences < 0.0)):
            raise ValueError("latitudes must be strictly monotonic")
    return values


def _latitude_cell_weights_np(latitudes: np.ndarray) -> np.ndarray:
    """Calculate exact zonal cell-area factors from latitude cell boundaries."""
    values = _validate_latitudes_np(latitudes)
    if values.size == 1:
        return np.ones(1, dtype=np.float64)

    edges = np.empty(values.size + 1, dtype=np.float64)
    edges[1:-1] = 0.5 * (values[:-1] + values[1:])
    edges[0] = values[0] + 0.5 * (values[0] - values[1])
    edges[-1] = values[-1] + 0.5 * (values[-1] - values[-2])
    edges = np.clip(edges, -90.0, 90.0)
    sin_edges = np.sin(np.deg2rad(edges))
    return np.abs(np.diff(sin_edges))


@overload
def latitude_cell_weights(latitudes: Tensor) -> Tensor: ...


@overload
def latitude_cell_weights(
    latitudes: np.ndarray | Sequence[float],
) -> np.ndarray: ...


def latitude_cell_weights(
    latitudes: np.ndarray | Tensor | Sequence[float],
) -> np.ndarray | Tensor:
    """Return unit-mean weights proportional to spherical grid-cell area.

    Cell boundaries are placed halfway between adjacent latitude coordinates
    and clipped at the poles.  Unlike a pointwise ``cos(latitude)``
    approximation, this gives the pole cells of a 721-point ERA5 grid their
    small but non-zero area.
    """
    if isinstance(latitudes, Tensor):
        output_dtype = (
            latitudes.dtype if latitudes.is_floating_point() else torch.float32
        )
        values_np = latitudes.detach().cpu().numpy()
        weights_np = _latitude_cell_weights_np(values_np)
        weights = torch.as_tensor(
            weights_np, device=latitudes.device, dtype=output_dtype
        )
        return weights / weights.mean()

    weights = _latitude_cell_weights_np(np.asarray(latitudes))
    weights = weights / weights.mean()
    return weights.astype(np.float32)


class AtmosphereNormalizer:
    """Channel-wise z-score normalization and latitude-weight generator.

    Provides forward and inverse normalization for both NumPy arrays and
    PyTorch tensors of shape ``[..., C, H, W]``.
    """

    def __init__(
        self,
        means: np.ndarray | Tensor | str | Path | None = None,
        stds: np.ndarray | Tensor | str | Path | None = None,
        normalize: bool = True,
    ) -> None:
        self.normalize_enabled = normalize
        self._means_np: np.ndarray | None = (
            self._to_numpy(means) if means is not None else None
        )
        self._stds_np: np.ndarray | None = (
            self._to_numpy(stds) if stds is not None else None
        )

        if normalize and (self._means_np is None or self._stds_np is None):
            raise ValueError("means and stds must be provided when normalize=True")

        if self._means_np is not None:
            self._means_np = self._as_channel_vector(self._means_np, "means")
        if self._stds_np is not None:
            self._stds_np = self._as_channel_vector(self._stds_np, "stds")

        if self._means_np is not None and not np.all(np.isfinite(self._means_np)):
            raise ValueError("Means must be finite")
        if self._stds_np is not None:
            if not np.all(np.isfinite(self._stds_np)):
                raise ValueError("Standard deviations must be finite")
            if np.any(self._stds_np <= 0):
                raise ValueError("Standard deviations must be strictly positive")
        if (
            self._means_np is not None
            and self._stds_np is not None
            and self._means_np.size != self._stds_np.size
        ):
            raise ValueError("means and stds must contain the same number of channels")

    @staticmethod
    def _to_numpy(data: np.ndarray | Tensor | str | Path | None) -> np.ndarray | None:
        if data is None:
            return None
        if isinstance(data, (str, Path)):
            path = Path(data).expanduser().resolve()
            if not path.exists():
                raise FileNotFoundError(f"Statistics file not found: {path}")
            return np.load(path, mmap_mode="r").astype(np.float32)
        if isinstance(data, Tensor):
            return data.detach().cpu().numpy().astype(np.float32)
        return np.asarray(data, dtype=np.float32)

    @staticmethod
    def _as_channel_vector(data: np.ndarray, name: str) -> np.ndarray:
        squeezed = np.squeeze(np.asarray(data, dtype=np.float32))
        if squeezed.ndim == 0:
            squeezed = squeezed.reshape(1)
        if squeezed.ndim != 1 or squeezed.size == 0:
            raise ValueError(
                f"{name} must be a channel vector or singleton-expanded vector, "
                f"got shape {data.shape}"
            )
        return np.ascontiguousarray(squeezed)

    def _stats_for_channel_count(
        self, channel_count: int
    ) -> tuple[np.ndarray, np.ndarray]:
        if self._means_np is None or self._stds_np is None:
            raise ValueError("Statistics are not initialized")

        if self._means_np.size != channel_count:
            raise ValueError(
                f"Expected {self._means_np.size} channels, received {channel_count}"
            )
        return self._means_np, self._stds_np

    @property
    def means(self) -> np.ndarray | None:
        return self._means_np

    @property
    def stds(self) -> np.ndarray | None:
        return self._stds_np

    def get_stats_tensor(
        self, device: torch.device | str = "cpu", dtype: torch.dtype = torch.float32
    ) -> tuple[Tensor, Tensor]:
        """Return means and stds as PyTorch tensors reshaped to ``[1, C, 1, 1]``."""
        if self._means_np is None or self._stds_np is None:
            raise ValueError("Statistics are not initialized")
        means = torch.from_numpy(self._means_np).to(device=device, dtype=dtype)
        stds = torch.from_numpy(self._stds_np).to(device=device, dtype=dtype)
        return means.view(1, -1, 1, 1), stds.view(1, -1, 1, 1)

    @overload
    def normalize(self, values: Tensor) -> Tensor: ...

    @overload
    def normalize(self, values: np.ndarray) -> np.ndarray: ...

    def normalize(
        self,
        values: np.ndarray | Tensor,
    ) -> np.ndarray | Tensor:
        """Apply channel-wise z-score normalization: ``(x - mean) / std``."""
        if (
            not self.normalize_enabled
            or self._means_np is None
            or self._stds_np is None
        ):
            return values

        if values.ndim < 3:
            raise ValueError("values must have shape [..., C, H, W]")
        means_np, stds_np = self._stats_for_channel_count(values.shape[-3])

        if isinstance(values, Tensor):
            broadcast_shape = (1,) * (values.ndim - 3) + (-1, 1, 1)
            means = torch.as_tensor(means_np, device=values.device, dtype=values.dtype)
            stds = torch.as_tensor(stds_np, device=values.device, dtype=values.dtype)
            means = means.view(broadcast_shape)
            stds = stds.view(broadcast_shape)
            return (values - means) / stds

        val_np = np.asarray(values, dtype=np.float32)
        means = means_np.reshape(-1, 1, 1)
        stds = stds_np.reshape(-1, 1, 1)
        return (val_np - means) / stds

    @overload
    def denormalize(self, values: Tensor) -> Tensor: ...

    @overload
    def denormalize(self, values: np.ndarray) -> np.ndarray: ...

    def denormalize(
        self,
        values: np.ndarray | Tensor,
    ) -> np.ndarray | Tensor:
        """Reverse z-score normalization: ``x * std + mean``."""
        if (
            not self.normalize_enabled
            or self._means_np is None
            or self._stds_np is None
        ):
            return values

        if values.ndim < 3:
            raise ValueError("values must have shape [..., C, H, W]")
        means_np, stds_np = self._stats_for_channel_count(values.shape[-3])

        if isinstance(values, Tensor):
            broadcast_shape = (1,) * (values.ndim - 3) + (-1, 1, 1)
            means = torch.as_tensor(means_np, device=values.device, dtype=values.dtype)
            stds = torch.as_tensor(stds_np, device=values.device, dtype=values.dtype)
            means = means.view(broadcast_shape)
            stds = stds.view(broadcast_shape)
            return values * stds + means

        val_np = np.asarray(values, dtype=np.float32)
        means = means_np.reshape(-1, 1, 1)
        stds = stds_np.reshape(-1, 1, 1)
        return val_np * stds + means

    @overload
    @staticmethod
    def get_latitude_weights(latitudes: Tensor) -> Tensor: ...

    @overload
    @staticmethod
    def get_latitude_weights(latitudes: np.ndarray | Sequence[float]) -> np.ndarray: ...

    @staticmethod
    def get_latitude_weights(
        latitudes: np.ndarray | Tensor | Sequence[float],
    ) -> np.ndarray | Tensor:
        """Compute unit-mean spherical cell-area weights."""
        return latitude_cell_weights(latitudes)

    @classmethod
    def compute_stats_from_dataset(
        cls,
        data: np.ndarray,
        save_dir: str | Path | None = None,
        *,
        latitudes: np.ndarray | Sequence[float] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute per-channel training-set statistics from ``[T, C, H, W]``.

        ``data`` must contain only the training split. This in-memory helper
        uses the same spherical area measure as the production CLI. It does not
        create a provenance bundle; use compute_stats.py for training artifacts.
        """
        if data.ndim != 4:
            raise ValueError(f"Expected 4D array [T, C, H, W], got {data.shape}")
        latitude = (
            np.linspace(90, -90, data.shape[2])
            if latitudes is None
            else np.asarray(latitudes)
        )
        if latitude.shape != (data.shape[2],):
            raise ValueError("latitudes must match the spatial height")
        weights = np.asarray(latitude_cell_weights(latitude), dtype=np.float64)[
            None, None, :, None
        ]
        values = np.asarray(data, dtype=np.float64)
        means64 = (values * weights).mean(axis=(0, 2, 3))
        means = means64.astype(np.float32)
        stds = np.sqrt(
            ((values - means64[None, :, None, None]) ** 2 * weights).mean(
                axis=(0, 2, 3)
            )
        ).astype(np.float32)
        if not np.all(np.isfinite(means)) or not np.all(np.isfinite(stds)):
            raise ValueError("Cannot compute normalization from non-finite data")
        if np.any(stds <= 0):
            raise ValueError("Every channel must have positive variance")

        if save_dir is not None:
            out = Path(save_dir)
            out.mkdir(parents=True, exist_ok=True)
            np.save(out / "means.npy", means)
            np.save(out / "stds.npy", stds)

        return means, stds
