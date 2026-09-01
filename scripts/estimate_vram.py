"""Estimate SFNO CUDA training memory without allocating the full model.

This is a capacity-planning estimate, not a replacement for a CUDA benchmark.
It accounts for complex spectral parameters, AdamW state, precomputed spherical
transform buffers, the full-resolution first/last SFNO blocks, and rollout
activation retention.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT_ROOT), str(PROJECT_ROOT / "src")]

from configs.model_config import AtmosphereModelConfig

GIB = 2**30


@dataclass(frozen=True, slots=True)
class MemoryEstimate:
    """Analytical components of an SFNO training-memory estimate."""

    internal_height: int
    internal_width: int
    modes: int
    spectral_parameters: int
    real_parameters: int
    real_scalar_dof: int
    parameter_bytes: int
    transform_buffer_bytes: int
    training_state_bytes: int
    optimizer_temporary_bytes: int
    one_forward_activation_bytes: int
    retained_activation_bytes: int
    analytical_peak_bytes: int
    planning_low_bytes: int
    planning_high_bytes: int


def estimate_memory(
    config: AtmosphereModelConfig,
    *,
    batch_size: int,
    rollout_steps: int,
    gradient_checkpointing: bool,
    activation_element_bytes: int,
) -> MemoryEstimate:
    """Return a conservative component-wise SFNO memory estimate.

    The project's checkpointing wraps one complete SFNO invocation. It avoids
    retaining every invocation during a rollout, but backward still has to
    materialize one complete invocation. Consequently, it should not be
    treated as layer-wise checkpointing for a one-step training pass.
    """
    if batch_size < 1 or rollout_steps < 1:
        raise ValueError("batch_size and rollout_steps must be positive")
    if activation_element_bytes not in {2, 4}:
        raise ValueError("activation_element_bytes must be 2 or 4")

    height, width = config.img_size
    internal_height = (height - 1) // config.scale_factor + 1
    internal_width = width // config.scale_factor
    modes = int(
        min(internal_height, internal_width // 2) * config.hard_thresholding_fraction
    )
    channels = config.embed_dim

    if config.operator_type == "driscoll-healy":
        spectral_parameters = config.num_layers * channels * channels * modes
    elif config.operator_type == "diagonal":
        spectral_parameters = config.num_layers * channels * channels * modes * modes
    else:  # guarded by AtmosphereModelConfig; retained for future extensions
        raise ValueError(f"Unsupported operator_type: {config.operator_type}")

    hidden_channels = int(channels * config.mlp_ratio)
    # torch-harmonics MLP: C->hidden with bias, then hidden->C without bias.
    mlp_parameters = (
        config.num_layers
        * (channels * hidden_channels + hidden_channels + hidden_channels * channels)
        if config.use_mlp
        else 0
    )
    projection_parameters = (
        config.in_channels * channels + channels * config.out_channels
    )
    if config.normalization_layer == "instance_norm":
        norm_parameters = config.num_layers * 4 * channels
    elif config.normalization_layer == "layer_norm":
        if config.num_layers == 1:
            norm_parameters = 4 * internal_height * internal_width
        else:
            norm_parameters = 4 * (
                (config.num_layers - 1) * internal_height * internal_width
                + height * width
            )
    else:
        norm_parameters = 0
    position_parameters = channels * height if config.pos_embed == "lat" else 0
    real_parameters = (
        mlp_parameters + projection_parameters + norm_parameters + position_parameters
    )

    # Spectral weights are complex64; all other learned tensors are float32.
    parameter_bytes = spectral_parameters * 8 + real_parameters * 4
    real_scalar_dof = spectral_parameters * 2 + real_parameters

    # torch-harmonics 0.7.x keeps one forward and one inverse Legendre table at
    # both the external and internal latitude grids, all in float32:
    # [modes, modes, nlat]. These buffers move to CUDA with the model.
    transform_buffer_bytes = 2 * modes * modes * (height + internal_height) * 4

    # FP32/complex64 weights + gradients + two AdamW moments, plus persistent
    # non-parameter transform tables. AMP does not shrink these tensors.
    training_state_bytes = 4 * parameter_bytes + transform_buffer_bytes

    # The current non-fused AdamW may temporarily allocate tensor-list-sized
    # intermediates during optimizer.step(). Treat one parameter copy as a
    # planning reserve; benchmark_model.py measures the real implementation.
    optimizer_temporary_bytes = parameter_bytes

    full_hidden = batch_size * channels * height * width * activation_element_bytes
    internal_hidden = (
        batch_size
        * channels
        * internal_height
        * internal_width
        * activation_element_bytes
    )
    input_state = (
        batch_size * config.in_channels * height * width * activation_element_bytes
    )
    output_state = (
        batch_size * config.out_channels * height * width * activation_element_bytes
    )
    spectral_activation = batch_size * channels * modes * modes * 8
    mlp_internal = (
        batch_size
        * hidden_channels
        * internal_height
        * internal_width
        * activation_element_bytes
        if config.use_mlp
        else 0
    )
    mlp_full = (
        batch_size * hidden_channels * height * width * activation_element_bytes
        if config.use_mlp
        else 0
    )

    # Unique large tensors along one SFNO call: encoder state, one spectral
    # tensor per block, block outputs, MLP hidden tensors, and decoded output.
    # The 1.35 factor covers normalization/activation/autograd saved tensors
    # that cannot be derived reliably without executing the installed kernels.
    intermediate_blocks = max(config.num_layers - 1, 0)
    large_activation_bytes = (
        input_state
        + full_hidden
        + config.num_layers * spectral_activation
        + intermediate_blocks * internal_hidden
        + full_hidden
        + intermediate_blocks * mlp_internal
        + mlp_full
        + output_state
    )
    one_forward_activation_bytes = int(1.35 * large_activation_bytes)

    if gradient_checkpointing:
        # A single call is recomputed at backward peak. Other rollout leads
        # retain only state/target-sized boundaries rather than full internals.
        rollout_boundaries = rollout_steps * (input_state + 2 * output_state)
        retained_activation_bytes = one_forward_activation_bytes + rollout_boundaries
    else:
        retained_activation_bytes = one_forward_activation_bytes * rollout_steps

    analytical_peak_bytes = (
        training_state_bytes + optimizer_temporary_bytes + retained_activation_bytes
    )
    # CUDA context, cuFFT/SHT workspaces, allocator fragmentation and backend
    # selection are device/runtime dependent. Give a range rather than false
    # precision and keep at least 1--2 GiB outside the analytical allocation.
    planning_low_bytes = int(analytical_peak_bytes * 1.15 + 1 * GIB)
    planning_high_bytes = int(analytical_peak_bytes * 1.35 + 2 * GIB)

    return MemoryEstimate(
        internal_height=internal_height,
        internal_width=internal_width,
        modes=modes,
        spectral_parameters=spectral_parameters,
        real_parameters=real_parameters,
        real_scalar_dof=real_scalar_dof,
        parameter_bytes=parameter_bytes,
        transform_buffer_bytes=transform_buffer_bytes,
        training_state_bytes=training_state_bytes,
        optimizer_temporary_bytes=optimizer_temporary_bytes,
        one_forward_activation_bytes=one_forward_activation_bytes,
        retained_activation_bytes=retained_activation_bytes,
        analytical_peak_bytes=analytical_peak_bytes,
        planning_low_bytes=planning_low_bytes,
        planning_high_bytes=planning_high_bytes,
    )


def _gib(value: int) -> str:
    return f"{value / GIB:.2f} GiB"


def main() -> None:
    defaults = AtmosphereModelConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embed-dim", type=int, default=defaults.embed_dim)
    parser.add_argument("--num-layers", type=int, default=defaults.num_layers)
    parser.add_argument("--scale-factor", type=int, default=defaults.scale_factor)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--rollout-steps", type=int, default=1)
    parser.add_argument(
        "--amp-dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument(
        "--gpu-vram-gib",
        type=float,
        help="Optional installed VRAM for a coarse fit verdict",
    )
    args = parser.parse_args()
    if args.gpu_vram_gib is not None and args.gpu_vram_gib <= 0:
        parser.error("--gpu-vram-gib must be positive")

    config = AtmosphereModelConfig(
        embed_dim=args.embed_dim,
        num_layers=args.num_layers,
        scale_factor=args.scale_factor,
    )
    estimate = estimate_memory(
        config,
        batch_size=args.batch_size,
        rollout_steps=args.rollout_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        activation_element_bytes=4 if args.amp_dtype == "float32" else 2,
    )

    print("SFNO CUDA memory capacity estimate")
    print(
        f"architecture: E{config.embed_dim}-L{config.num_layers}-SC{config.scale_factor}, "
        f"batch={args.batch_size}, rollout={args.rollout_steps}, "
        f"amp={args.amp_dtype}, checkpoint={args.gradient_checkpointing}"
    )
    print(
        f"grid: {config.img_size[0]}x{config.img_size[1]} -> "
        f"{estimate.internal_height}x{estimate.internal_width}; "
        f"retained modes={estimate.modes}"
    )
    print(f"spectral parameters: {estimate.spectral_parameters:,} complex64")
    print(f"other parameters: {estimate.real_parameters:,} float32")
    print(f"equivalent real scalar degrees of freedom: {estimate.real_scalar_dof:,}")
    print(f"model parameter storage: {_gib(estimate.parameter_bytes)}")
    print(f"spherical-transform buffers: {_gib(estimate.transform_buffer_bytes)}")
    print(f"weights+gradients+AdamW+buffers: {_gib(estimate.training_state_bytes)}")
    print(
        "possible non-fused AdamW temporary: "
        f"{_gib(estimate.optimizer_temporary_bytes)}"
    )
    print(
        "one-SFNO-call activation heuristic: "
        f"{_gib(estimate.one_forward_activation_bytes)}"
    )
    print(
        "retained/recomputed rollout activations: "
        f"{_gib(estimate.retained_activation_bytes)}"
    )
    print(f"analytical peak heuristic: {_gib(estimate.analytical_peak_bytes)}")
    print(
        "recommended planning range: "
        f"{_gib(estimate.planning_low_bytes)} .. "
        f"{_gib(estimate.planning_high_bytes)}"
    )

    if args.gpu_vram_gib is not None:
        installed = args.gpu_vram_gib * GIB
        if estimate.planning_high_bytes <= 0.9 * installed:
            verdict = "comfortable analytical margin"
        elif estimate.planning_low_bytes <= 0.95 * installed:
            verdict = "borderline; run the CUDA benchmark before training"
        else:
            verdict = "likely insufficient without reducing model/rollout/batch"
        print(f"fit verdict for {args.gpu_vram_gib:g} GiB: {verdict}")

    print(
        "The planning range is deliberately conservative. "
        "Run benchmark_model.py on the target CUDA GPU for the authoritative peak."
    )


if __name__ == "__main__":
    main()
