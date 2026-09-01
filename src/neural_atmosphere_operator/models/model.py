"""Spherical Fourier Neural Operator for global atmospheric forecasting."""

from __future__ import annotations

import inspect
from importlib import import_module
from typing import Any, cast

import torch
from torch import Tensor, nn

from configs.model_config import AtmosphereModelConfig


# Follow NeuralOceanOperator's dependency boundary: the reference spherical
# architectures are examples in torch-harmonics rather than stable core API.
# Keeping the import and API adaptation here prevents version-specific details
# from leaking into the rest of the project.
_sfno_module = import_module("torch_harmonics.examples.models.sfno")
_MODERN_SFNO_API = hasattr(_sfno_module, "SphericalFourierNeuralOperator")
_SFNO_CLASS: Any = getattr(
    _sfno_module,
    (
        "SphericalFourierNeuralOperator"
        if _MODERN_SFNO_API
        else "SphericalFourierNeuralOperatorNet"
    ),
)


def _build_sfno(config: AtmosphereModelConfig) -> nn.Module:
    """Build the official torch-harmonics SFNO for the installed API.

    ``torch-harmonics`` renamed the reference class and some residual/position
    options between releases. Only supported constructor arguments are passed,
    while the scientific architecture remains the same. The large internal
    residual is deliberately disabled because :class:`AtmosphereNeuralOperator`
    performs the physically interpretable current-state plus tendency update.
    """

    options: dict[str, Any] = {
        "img_size": config.img_size,
        "operator_type": config.operator_type,
        "grid": config.grid,
        "grid_internal": config.grid_internal,
        "scale_factor": config.scale_factor,
        "in_chans": config.in_channels,
        "out_chans": config.out_channels,
        "embed_dim": config.embed_dim,
        "num_layers": config.num_layers,
        "activation_function": config.activation_function,
        "use_mlp": config.use_mlp,
        "mlp_ratio": config.mlp_ratio,
        "drop_rate": config.drop_rate,
        "drop_path_rate": config.drop_path_rate,
        "normalization_layer": config.normalization_layer,
        "hard_thresholding_fraction": config.hard_thresholding_fraction,
        "use_complex_kernels": config.use_complex_kernels,
    }

    if config.pos_embed == "none":
        position_embedding: bool | str = False
    elif _MODERN_SFNO_API:
        position_embedding = "learnable lat"
    else:
        # torch-harmonics 0.7 accepts named modes although its annotation in
        # some releases incorrectly declares this parameter as bool.
        position_embedding = cast(Any, "lat")

    constructor_parameters = inspect.signature(_SFNO_CLASS).parameters
    accepts_extra_options = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in constructor_parameters.values()
    )
    required_options = {
        "img_size",
        "scale_factor",
        "in_chans",
        "out_chans",
        "embed_dim",
        "num_layers",
        "activation_function",
        "use_mlp",
        "mlp_ratio",
        "normalization_layer",
        "hard_thresholding_fraction",
    }
    unsupported_required = required_options - constructor_parameters.keys()
    if unsupported_required and not accepts_extra_options:
        raise RuntimeError(
            "Installed torch-harmonics SFNO API cannot represent the requested "
            "architecture; missing constructor options: "
            + ", ".join(sorted(unsupported_required))
        )
    if "pos_embed" in constructor_parameters or accepts_extra_options:
        options["pos_embed"] = position_embedding
    if _MODERN_SFNO_API and (
        "residual_prediction" in constructor_parameters or accepts_extra_options
    ):
        options["residual_prediction"] = False
    elif "big_skip" in constructor_parameters or accepts_extra_options:
        options["big_skip"] = False
    if not accepts_extra_options and not {
        "residual_prediction", "big_skip"
    }.intersection(constructor_parameters):
        raise RuntimeError(
            "Installed SFNO API does not expose a residual control; external "
            "current-state addition could otherwise be applied twice"
        )
    if "bias" in constructor_parameters:
        options["bias"] = False

    supported_options = (
        options
        if accepts_extra_options
        else {
            name: value
            for name, value in options.items()
            if name in constructor_parameters
        }
    )
    return _SFNO_CLASS(**supported_options)


class AtmosphereNeuralOperator(nn.Module):
    """Predict the next normalized atmospheric state with an SFNO tendency.

    The backbone is NVIDIA's reference Spherical Fourier Neural Operator from
    ``torch-harmonics``. Unlike planar AFNO, its spectral mixing uses spherical
    harmonic transforms and therefore respects the geometry of the global
    latitude-longitude grid.

    As in ``NeuralOceanOperator``, the backbone predicts a normalized tendency
    and this wrapper adds it to the most recent state. The operation remains
    differentiable through an autoregressive rollout.
    """

    def __init__(self, config: AtmosphereModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or AtmosphereModelConfig()
        self.backbone = _build_sfno(self.config)

    def forward(self, x: Tensor) -> Tensor:
        """Return the next state for an input shaped ``[B, C_in, H, W]``."""
        self._validate_input(x)
        tendency = self.backbone(x)
        expected_output_shape = (
            x.shape[0],
            self.config.out_channels,
            *self.config.img_size,
        )
        if tendency.shape != expected_output_shape:
            raise RuntimeError(
                "SFNO backbone returned shape "
                f"{tuple(tendency.shape)}, expected {expected_output_shape}"
            )

        if not self.config.use_residual_connection:
            return tendency

        # AtmosphereZarrDataset concatenates history oldest-to-newest, making
        # the final out_channels the current normalized atmospheric state.
        current_state = x[:, -self.config.out_channels :]
        return current_state + tendency

    def rollout(self, x: Tensor, steps: int = 1) -> list[Tensor]:
        """Perform a differentiable autoregressive rollout for ``steps`` leads."""
        self._validate_input(x)
        if steps < 1:
            raise ValueError("steps must be at least 1")
        if self.config.in_channels % self.config.out_channels != 0:
            raise ValueError(
                "Autoregressive rollout requires in_channels to be a multiple "
                "of out_channels"
            )

        forecasts: list[Tensor] = []
        current = x
        for _ in range(steps):
            prediction = self.forward(current)
            forecasts.append(prediction)
            if self.config.in_channels == self.config.out_channels:
                current = prediction
            else:
                current = torch.cat(
                    (current[:, self.config.out_channels :], prediction), dim=1
                )
        return forecasts

    def _validate_input(self, x: Tensor) -> None:
        expected_shape = (self.config.in_channels, *self.config.img_size)
        if x.ndim != 4 or x.shape[1:] != expected_shape:
            raise ValueError(
                f"x must have shape [B, {expected_shape[0]}, "
                f"{expected_shape[1]}, {expected_shape[2]}]"
            )
