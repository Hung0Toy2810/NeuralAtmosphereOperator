"""Configuration schema for the spherical atmospheric operator."""

from __future__ import annotations

from dataclasses import dataclass

PILOT_EMBED_DIM = 128
PILOT_NUM_LAYERS = 6
MAKANI_REFERENCE_EMBED_DIM = 384
MAKANI_REFERENCE_NUM_LAYERS = 8


@dataclass(frozen=True, slots=True)
class AtmosphereModelConfig:
    """Configuration for a residual Spherical Fourier Neural Operator.

    The default is the locked compact SFNO-SC2-L6-E128 adapted to the selected
    0.5-degree grid. It
        follows the same SFNO family while reducing statistical capacity and
        memory for this project's 26-channel, twenty-four-year dataset. It is an
        adaptation, not an exact reproduction of Makani's SC3/73-channel setup. Residual
    prediction is applied by the project wrapper, not inside torch-harmonics.
    """

    img_size: tuple[int, int] = (361, 720)
    in_channels: int = 26
    out_channels: int = 26

    scale_factor: int = 2
    embed_dim: int = PILOT_EMBED_DIM
    num_layers: int = PILOT_NUM_LAYERS
    activation_function: str = "gelu"
    normalization_layer: str = "instance_norm"
    use_mlp: bool = True
    mlp_ratio: float = 2.0
    drop_rate: float = 0.0
    drop_path_rate: float = 0.0
    hard_thresholding_fraction: float = 1.0
    use_complex_kernels: bool = True
    stabilize_sht_constants: bool = True

    operator_type: str = "driscoll-healy"
    grid: str = "equiangular"
    grid_internal: str = "legendre-gauss"
    pos_embed: str = "none"
    use_residual_connection: bool = True
    initialization: str = "torch_harmonics_native"

    def __post_init__(self) -> None:
        height, width = self.img_size
        if height < 3 or width < 4:
            raise ValueError("img_size is too small for spherical transforms")
        if self.scale_factor < 1:
            raise ValueError("scale_factor must be at least 1")
        if (height - 1) % self.scale_factor != 0:
            raise ValueError(
                "spatial height must satisfy (height - 1) % scale_factor == 0"
            )
        if width % self.scale_factor != 0:
            raise ValueError("spatial width must be divisible by scale_factor")
        if self.in_channels < 1 or self.out_channels < 1:
            raise ValueError("channel counts must be positive")
        if self.embed_dim < 1 or self.num_layers < 1:
            raise ValueError("embed_dim and num_layers must be positive")
        if self.activation_function not in {"gelu", "relu", "identity"}:
            raise ValueError("unsupported activation_function")
        if self.normalization_layer not in {
            "instance_norm",
            "layer_norm",
            "none",
        }:
            raise ValueError("unsupported normalization_layer")
        if self.mlp_ratio <= 0:
            raise ValueError("mlp_ratio must be positive")
        if not 0.0 <= self.drop_rate < 1.0:
            raise ValueError("drop_rate must be in [0, 1)")
        if not 0.0 <= self.drop_path_rate < 1.0:
            raise ValueError("drop_path_rate must be in [0, 1)")
        if not 0.0 < self.hard_thresholding_fraction <= 1.0:
            raise ValueError("hard_thresholding_fraction must be in (0, 1]")
        if not self.use_complex_kernels:
            raise ValueError(
                "torch-harmonics 0.7.4 ignores use_complex_kernels=False; real kernels are unsupported"
            )
        if self.scale_factor == 1 and self.grid_internal != self.grid:
            raise ValueError(
                "scale_factor=1 requires matching grids to avoid incorrect residual resampling"
            )
        if self.num_layers == 1 and self.normalization_layer == "layer_norm":
            raise ValueError(
                "num_layers=1 with layer_norm is unsupported by the locked backend"
            )
        retained = int(
            min((height - 1) // self.scale_factor + 1, width // self.scale_factor // 2)
            * self.hard_thresholding_fraction
        )
        if retained < 1:
            raise ValueError(
                "hard_thresholding_fraction must retain at least one harmonic degree/order"
            )
        if self.operator_type not in {"driscoll-healy", "diagonal"}:
            raise ValueError("unsupported operator_type")
        if self.grid != "equiangular":
            raise ValueError("atmospheric input grid must be equiangular")
        if self.grid_internal not in {"legendre-gauss", "equiangular"}:
            raise ValueError("unsupported internal spherical grid")
        if self.pos_embed not in {"none", "lat"}:
            raise ValueError("pos_embed must be 'none' or 'lat'")
        if self.initialization != "torch_harmonics_native":
            raise ValueError(
                "Only torch_harmonics_native initialization is supported because "
                "generic reinitialization would overwrite transform-specific scaling"
            )
        if self.use_residual_connection and (
            self.in_channels < self.out_channels
            or self.in_channels % self.out_channels != 0
        ):
            raise ValueError(
                "Residual prediction requires in_channels to be a positive "
                "multiple of out_channels (one or more stacked states)"
            )


def large_e384_l8_ablation_config() -> AtmosphereModelConfig:
    """Return an E384/L8 capacity ablation while retaining this project's SC2 contract."""
    return AtmosphereModelConfig(
        embed_dim=MAKANI_REFERENCE_EMBED_DIM,
        num_layers=MAKANI_REFERENCE_NUM_LAYERS,
    )


def makani_reference_model_config() -> AtmosphereModelConfig:
    """Compatibility alias for the E384/L8 ablation; not an exact Makani config."""
    return large_e384_l8_ablation_config()
