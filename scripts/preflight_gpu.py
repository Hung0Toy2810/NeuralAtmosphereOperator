"""Measure real training updates and target-horizon validation on a CUDA GPU."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
import numpy as np
import torch

from configs.model_config import AtmosphereModelConfig
from configs.pipeline_config import DataPaths, TrainingConfig
from neural_atmosphere_operator.models.loss import (
    StandardizedTendencyLoss,
    graphcast_channel_weights,
    standardized_tendency_scale,
)
from neural_atmosphere_operator.pipeline.contracts import (
    enforce_forecast_contract,
    forecast_contract,
    validate_global_grid,
    validate_statistics_bundle,
)
from neural_atmosphere_operator.pipeline.dependencies import validate_supported_runtime
from neural_atmosphere_operator.pipeline.forecast import rollout_loss
from neural_atmosphere_operator.pipeline.runtime import (
    amp_context,
    build_loader,
    build_model,
    choose_device,
    dataset_signature,
    move_batch,
    save_json,
    seed_everything,
)
from scripts.train import validate


def main():
    training_defaults = TrainingConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DataPaths().root)
    parser.add_argument("--train-data", type=Path)
    parser.add_argument("--valid-data", type=Path)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--updates", type=int, default=3)
    parser.add_argument(
        "--rollout-steps", type=int, default=training_defaults.rollout_steps
    )
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument(
        "--stabilize-sht-constants", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, default=Path("runs/gpu_preflight.json"))
    args = parser.parse_args()
    if (
        min(
            args.batch_size,
            args.gradient_accumulation,
            args.updates,
            args.rollout_steps,
        )
        < 1
    ):
        raise ValueError("Counts must be positive")
    if args.output.exists():
        raise FileExistsError(
            "Choose a new preflight output to preserve prior measurements"
        )
    seed_everything(42)
    device = choose_device(args.device)
    if device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("This BF16 training profile requires native BF16 support")
    paths = DataPaths(args.data_dir.expanduser().resolve())
    runtime = validate_supported_runtime(device)
    train_loader, train = build_loader(
        paths,
        "train",
        rollout_steps=args.rollout_steps,
        batch_size=args.batch_size,
        data_path=args.train_data,
        max_samples=args.batch_size * args.gradient_accumulation,
        history=0,
        time_step=1,
        num_workers=0,
        pin_memory=False,
    )
    valid_loader, valid = build_loader(
        paths,
        "valid",
        rollout_steps=args.rollout_steps,
        batch_size=1,
        data_path=args.valid_data,
        max_samples=1,
        sample_selection="first",
        lazy_targets=True,
        history=0,
        time_step=1,
        num_workers=0,
        pin_memory=False,
    )
    try:
        validate_global_grid(train)
        bundle = validate_statistics_bundle(
            train,
            {
                "means": paths.means,
                "stds": paths.stds,
                "climatology": paths.climatology,
                "time_diff_stds": paths.time_diff_stds(),
            },
            1,
        )
        training = {
            "forecast_contract": forecast_contract(train, 1),
            "data_signature": {"train": dataset_signature(train)},
        }
        enforce_forecast_contract(training, valid, 1, "valid")
        tendency_scale = standardized_tendency_scale(
            np.load(paths.stds), np.load(paths.time_diff_stds())
        )
        config = AtmosphereModelConfig(
            img_size=train.spatial_shape,
            embed_dim=args.embed_dim,
            num_layers=args.num_layers,
            stabilize_sht_constants=args.stabilize_sht_constants,
            tendency_scale=tuple(float(value) for value in tendency_scale.tolist()),
        )
        model = build_model(config, device)
        latitudes = torch.as_tensor(train.latitudes, device=device)
        weights = graphcast_channel_weights(train.channel_names).to(device)
        criterion = StandardizedTendencyLoss(
            tendency_scale=tendency_scale,
            channel_weights=weights,
            latitudes=latitudes,
        ).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=5e-4, betas=(0.9, 0.95), weight_decay=1e-4
        )
        amp = device.type == "cuda"
        if amp:
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        first = move_batch(next(iter(train_loader)), device, keep_targets_cpu=True)
        with torch.no_grad():
            reference = model(first["input"]).float()
            with amp_context(device, amp, "bfloat16"):
                reduced = model(first["input"]).float()
            precision_error = float((reference - reduced).square().mean().sqrt())
            precision_relative = precision_error / max(
                float(reference.square().mean().sqrt()), 1e-12
            )
            probes = {}
            for name, value in [
                ("zero", 0.0),
                ("constant", 1.0),
                ("near_constant", 1.0),
            ]:
                probe_input = torch.full_like(first["input"][:1], value)
                if name == "near_constant":
                    probe_input += 1e-5 * torch.randn_like(probe_input)
                output = model(probe_input).float()
                if not torch.isfinite(output).all():
                    raise FloatingPointError(f"{name} probe is non-finite")
                probes[name] = {
                    "rms": float(output.square().mean().sqrt()),
                    "spatial_std": float(output.std(dim=(-2, -1)).mean()),
                }
            if config.stabilize_sht_constants and probes["constant"][
                "spatial_std"
            ] > 1e-5 * max(1.0, probes["constant"]["rms"]):
                raise FloatingPointError(
                    "Constant-preserving SHT failed the full-grid constant-field check on this backend"
                )
        del reference, reduced, first, output, probe_input
        losses, norms = [], []
        start = time.perf_counter()
        for update in range(args.updates):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            total = 0.0
            for raw in train_loader:
                batch = move_batch(raw, device)
                with amp_context(device, amp, "bfloat16"):
                    loss, _, _ = rollout_loss(
                        model,
                        batch,
                        args.rollout_steps,
                        gradient_checkpointing=args.gradient_checkpointing,
                        loss_fn=criterion,
                    )
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite preflight loss")
                (loss * (batch["input"].shape[0] / len(train))).backward()
                total += float(loss.detach()) * batch["input"].shape[0] / len(train)
            norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0, error_if_nonfinite=True
            )
            optimizer.step()
            if any(not torch.isfinite(value).all() for value in model.parameters()):
                raise FloatingPointError("Optimizer produced non-finite parameters")
            losses.append(total)
            norms.append(float(norm))
        if amp:
            torch.cuda.synchronize(device)
        update_seconds = (time.perf_counter() - start) / args.updates
        validation = validate(
            model,
            valid_loader,
            device,
            args.rollout_steps,
            amp,
            "bfloat16",
            1.0,
            latitudes,
            criterion,
        )
        if amp:
            torch.cuda.synchronize(device)
        report = {
            "status": "passed",
            "scope": "hardware/data smoke test, not forecast skill",
            "runtime": runtime,
            "arguments": vars(args),
            "training_contract": training,
            "statistics_artifacts": bundle["artifacts"],
            "losses": losses,
            "preclip_gradient_norms": norms,
            "seconds_per_update": update_seconds,
            "fp32_bf16_output_rmse": precision_error,
            "fp32_bf16_relative_rmse": precision_relative,
            "constant_probes": probes,
            "validation_lead_rmse": validation.lead_rmse,
            "validation_selection": validation.selection_score,
            "validation_mean_loss": validation.mean_loss,
            "validation_final_loss": validation.final_loss,
            "peak_allocated_gib": (
                torch.cuda.max_memory_allocated(device) / 2**30 if amp else None
            ),
            "peak_reserved_gib": (
                torch.cuda.max_memory_reserved(device) / 2**30 if amp else None
            ),
            "gpu_total_gib": (
                torch.cuda.get_device_properties(device).total_memory / 2**30
                if amp
                else None
            ),
        }
        save_json(args.output, report)
        print(f"Preflight passed; measurements: {args.output}")
    finally:
        train.close()
        valid.close()


if __name__ == "__main__":
    main()
