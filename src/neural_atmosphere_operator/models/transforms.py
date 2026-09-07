"""Numerically preserve constant fields in the orthonormal real SHT."""

from __future__ import annotations

import math
from typing import cast
from torch import Tensor, nn
from torch_harmonics import RealSHT


class ConstantPreservingRealSHT(RealSHT):
    """Evaluate SHT(x-c) + c*sqrt(4*pi)*e_00 using existing quadrature buffers.

    This is the same continuous linear transform: an orthonormal scalar
    constant has only coefficient (l,m)=(0,0). Removing its large offset before
    quadrature prevents float32 constant leakage into nonzero degrees from
    being amplified by subsequent InstanceNorm layers. Anchored centering
    gives exactly zero residual for an exactly constant floating-point input.
    """

    def __init__(self, transform: RealSHT):
        nn.Module.__init__(self)
        if transform.norm != "ortho":
            raise ValueError("Constant-preserving SHT requires orthonormal harmonics")
        for name in ("nlat", "nlon", "lmax", "mmax", "grid", "norm", "csphase"):
            setattr(self, name, getattr(transform, name))
        # Preserve the original buffer, dtype, device and state_dict contract.
        self.register_buffer(
            "weights", cast(Tensor, transform.weights), persistent=False
        )

    def forward(self, x: Tensor) -> Tensor:
        anchor = x[..., :1, :1]
        center = anchor + (x - anchor).mean(dim=(-2, -1), keepdim=True)
        coefficients = super().forward(x - center)
        coefficients[..., 0, 0] = coefficients[..., 0, 0] + center[
            ..., 0, 0
        ] * math.sqrt(4.0 * math.pi)
        return coefficients


def stabilize_sht_constants(module: nn.Module) -> None:
    """Replace shared forward-transform references once, preserving aliasing."""
    replacements: dict[int, ConstantPreservingRealSHT] = {}

    def visit(parent: nn.Module) -> None:
        for name, child in tuple(parent._modules.items()):
            if isinstance(child, ConstantPreservingRealSHT):
                continue
            if isinstance(child, RealSHT):
                if id(child) not in replacements:
                    replacements[id(child)] = ConstantPreservingRealSHT(child)
                parent._modules[name] = replacements[id(child)]
            elif child is not None:
                visit(child)

    visit(module)
