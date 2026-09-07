"""Train the atmospheric SFNO with restartable autoregressive stages."""

from __future__ import annotations

import argparse
import csv
import itertools
import math
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, cast

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT_ROOT), str(PROJECT_ROOT / "src")]

import numpy as np
import torch
from torch import Tensor
from torch.amp.grad_scaler import GradScaler
from torch.optim import AdamW

from configs.model_config import AtmosphereModelConfig
from configs.pipeline_config import DataPaths, TrainingConfig
from neural_atmosphere_operator.models.loss import (
    ChannelRelativeAtmosphereLoss,
    CombinedAtmosphereLoss,
    latitude_weighted_mse,
    makani_auto_channel_weights,
)
from neural_atmosphere_operator.models.model import AtmosphereNeuralOperator
from neural_atmosphere_operator.pipeline.checkpoint import (
    load_checkpoint,
    restore_training_state,
    save_checkpoint,
)
from neural_atmosphere_operator.pipeline.dependencies import (
    enforce_checkpoint_runtime,
    validate_supported_runtime,
)
from neural_atmosphere_operator.pipeline.forecast import (
    autoregressive_predictions,
    lazy_target_provider,
    rollout_loss,
)
from neural_atmosphere_operator.pipeline.runtime import (
    accumulation_bucket_sample_count,
    amp_context,
    build_loader,
    build_model,
    choose_device,
    config_dict,
    count_parameters,
    dataset_signature,
    model_config_from_dict,
    move_batch,
    save_json,
    seed_everything,
    statistics_signature,
)
from neural_atmosphere_operator.pipeline.contracts import (
    forecast_contract,
    validate_global_grid,
    validate_statistics_bundle,
)
from neural_atmosphere_operator.pipeline.schedule import (
    create_warmup_cosine_scheduler,
)
from neural_atmosphere_operator.utils.logger import setup_logger


def parse_args() -> argparse.Namespace:
    defaults = TrainingConfig()
    model = AtmosphereModelConfig()
    paths = DataPaths()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=paths.root)
    parser.add_argument("--train-data", type=Path)
    parser.add_argument("--valid-data", type=Path)
    parser.add_argument("--means-path", type=Path)
    parser.add_argument("--stds-path", type=Path)
    parser.add_argument("--climatology-path", type=Path)
    parser.add_argument("--time-diff-stds-path", type=Path)
    parser.add_argument(
        "--run-dir", type=Path, default=PROJECT_ROOT / "runs" / "default"
    )
    parser.add_argument("--stage-name", default="sfno_1step")
    checkpoint_group = parser.add_mutually_exclusive_group()
    checkpoint_group.add_argument("--resume", type=Path)
    checkpoint_group.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--allow-runtime-mismatch", action="store_true")
    parser.add_argument("--reset-early-stopping", action="store_true")

    parser.add_argument(
        "--epochs",
        type=int,
        default=defaults.epochs,
        help="Maximum epochs; 0 runs indefinitely until manually stopped",
    )
    parser.add_argument(
        "--scheduler-epochs",
        type=int,
        help=(
            "Cosine-decay horizon; defaults to --epochs for a finite run and "
            f"{defaults.scheduler_epochs} for an unlimited run"
        ),
    )
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument(
        "--validation-batch-size",
        type=int,
        default=defaults.validation_batch_size,
        help="Smaller batch for long validation rollouts to bound target memory",
    )
    parser.add_argument(
        "--gradient-accumulation",
        type=int,
        default=defaults.gradient_accumulation,
    )
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=defaults.weight_decay)
    parser.add_argument(
        "--min-learning-rate", type=float, default=defaults.min_learning_rate
    )
    parser.add_argument("--warmup-epochs", type=int, default=defaults.warmup_epochs)
    parser.add_argument(
        "--warmup-start-factor", type=float, default=defaults.warmup_start_factor
    )
    parser.add_argument("--gradient-clip", type=float, default=defaults.gradient_clip)
    parser.add_argument("--rollout-steps", type=int, default=defaults.rollout_steps)
    parser.add_argument(
        "--validation-rollout-steps",
        type=int,
        default=defaults.validation_rollout_steps,
        help=(
            "Autoregressive horizon used for checkpoint selection, independent "
            "of the shorter training curriculum horizon"
        ),
    )
    parser.add_argument(
        "--rollout-discount", type=float, default=defaults.rollout_discount
    )
    parser.add_argument("--history", type=int, default=defaults.history)
    parser.add_argument("--time-step", type=int, default=defaults.time_step)
    parser.add_argument("--num-workers", type=int, default=defaults.num_workers)
    parser.add_argument("--prefetch-factor", type=int, default=defaults.prefetch_factor)
    parser.add_argument(
        "--persistent-workers",
        action=argparse.BooleanOptionalAction,
        default=defaults.persistent_workers,
        help=(
            "Unsupported for restartable training; workers must be recreated "
            "at epoch boundaries (asynchronous prefetching remains enabled)"
        ),
    )
    parser.add_argument("--patience", type=int, default=defaults.patience)
    parser.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=defaults.early_stopping_min_delta,
    )
    parser.add_argument(
        "--terminal-loss-weight",
        type=float,
        default=defaults.terminal_loss_weight,
    )
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--amp-dtype",
        choices=("bfloat16", "float16"),
        default=defaults.amp_dtype,
    )
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=defaults.gradient_checkpointing,
        help="Checkpoint each SFNO rollout call (disabled for one-step by default)",
    )
    parser.add_argument(
        "--input-noise-std",
        type=float,
        default=defaults.input_noise_std,
        help="Optional Gaussian noise in normalized input space (training only)",
    )

    parser.add_argument(
        "--stabilize-sht-constants",
        action=argparse.BooleanOptionalAction,
        default=model.stabilize_sht_constants,
    )
    parser.add_argument("--embed-dim", type=int, default=model.embed_dim)
    parser.add_argument("--num-layers", type=int, default=model.num_layers)
    parser.add_argument("--scale-factor", type=int, default=model.scale_factor)
    parser.add_argument("--drop-rate", type=float, default=model.drop_rate)
    parser.add_argument("--drop-path-rate", type=float, default=model.drop_path_rate)
    parser.add_argument(
        "--hard-thresholding-fraction",
        type=float,
        default=model.hard_thresholding_fraction,
    )
    parser.add_argument("--pos-embed", choices=("none", "lat"), default=model.pos_embed)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument(
        "--max-validation-samples",
        type=int,
        default=32,
        help="Evenly spaced initialization panel; fixed across model candidates",
    )
    parser.add_argument(
        "--validation-num-workers",
        type=int,
        default=0,
        help="Separate worker budget for large future-target windows",
    )
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--spatial-crop", type=int, nargs=2, metavar=("H", "W"))
    parser.add_argument(
        "--allow-overlapping-splits",
        action="store_true",
        help="Diagnostic only: permit temporal overlap between train and validation",
    )

    parser.add_argument(
        "--spectral-loss-weight",
        type=float,
        default=defaults.spectral_loss_weight,
    )
    parser.add_argument(
        "--channel-relative-weight",
        type=float,
        default=defaults.channel_relative_weight,
    )
    parser.add_argument(
        "--channel-weighting",
        choices=("auto", "constant"),
        default=defaults.channel_weighting,
        help="Makani auto pressure-level weights or equal channel weights",
    )
    parser.add_argument(
        "--temp-diff-normalization",
        action=argparse.BooleanOptionalAction,
        default=defaults.temp_diff_normalization,
        help="Scale loss channels by global_std/time_diff_std as in Makani",
    )
    parser.add_argument(
        "--use-loss-scaler",
        action="store_true",
        default=defaults.use_loss_scaler,
        help="Experimental per-channel backward gradient re-balancing",
    )
    return parser.parse_args()


@dataclass(frozen=True, slots=True)
class ValidationSummary:
    selection_score: float
    mean_loss: float
    final_loss: float
    lead_losses: tuple[float, ...]
    lead_rmse: tuple[float, ...]


@torch.no_grad()
def validate(
    model: AtmosphereNeuralOperator,
    loader,
    device: torch.device,
    steps: int,
    amp: bool,
    amp_dtype: str,
    terminal_loss_weight: float,
    latitudes: Tensor,
    loss_fn: Callable[[Tensor, Tensor], Tensor],
) -> ValidationSummary:
    model.eval()
    reduction_dtype = torch.float32 if device.type == "mps" else torch.float64
    step_loss_sums = torch.zeros(steps, device=device, dtype=reduction_dtype)
    step_mse_sums = torch.zeros_like(step_loss_sums)
    sample_count = 0
    for raw_batch in loader:
        batch = move_batch(raw_batch, device, keep_targets_cpu=True)
        with amp_context(device, amp, amp_dtype):
            pairs = autoregressive_predictions(
                model,
                batch,
                steps,
                target_provider=lazy_target_provider(loader.dataset, raw_batch),
            )
            for lead, (prediction, target) in enumerate(pairs):
                lead_loss = loss_fn(prediction, target)
                if not torch.isfinite(lead_loss).all():
                    raise FloatingPointError(
                        f"Non-finite validation loss at lead {lead + 1}"
                    )
                size = batch["input"].shape[0]
                step_loss_sums[lead] += lead_loss * size
                step_mse_sums[lead] += (
                    latitude_weighted_mse(prediction.float(), target.float(), latitudes)
                    * size
                )
        size = batch["input"].shape[0]
        sample_count += size
    if sample_count < 1:
        raise ValueError("Validation requires at least one sample")
    denominator = sample_count
    if not torch.isfinite(step_mse_sums).all():
        raise FloatingPointError("Validation metric reductions overflowed")
    step_losses = tuple((step_loss_sums / denominator).cpu().tolist())
    lead_rmse = tuple((step_mse_sums / denominator).sqrt().cpu().tolist())
    mean_loss = sum(step_losses) / len(step_losses)
    final_loss = step_losses[-1]
    selection = (
        1.0 - terminal_loss_weight
    ) * mean_loss + terminal_loss_weight * final_loss
    return ValidationSummary(
        selection_score=selection,
        mean_loss=mean_loss,
        final_loss=final_loss,
        lead_losses=step_losses,
        lead_rmse=lead_rmse,
    )


def append_history(path: Path, row: dict[str, float | int]) -> None:
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def same_model_architecture(
    saved: AtmosphereModelConfig, current: AtmosphereModelConfig
) -> bool:
    return saved == current


def _validate_cli(args: argparse.Namespace) -> None:
    if args.amp_dtype == "float16" and not args.no_amp:
        raise ValueError(
            "FP16 GradScaler with complex spectral gradients is unsupported by this training profile; use bfloat16 or --no-amp"
        )
    positive = (
        args.batch_size,
        args.validation_batch_size,
        args.gradient_accumulation,
        args.rollout_steps,
        args.validation_rollout_steps,
        args.time_step,
    )
    if min(positive) < 1 or args.history < 0:
        raise ValueError(
            "positive training counts are required; history cannot be negative"
        )
    if args.epochs < 0:
        raise ValueError("epochs must be zero (unlimited) or positive")
    if args.scheduler_epochs is not None and args.scheduler_epochs < 1:
        raise ValueError("scheduler epochs must be positive")
    if args.warmup_epochs < 0 or args.patience < 0:
        raise ValueError("warmup epochs and patience cannot be negative")
    if not 0 < args.warmup_start_factor <= 1:
        raise ValueError("warmup_start_factor must be in (0, 1]")
    if not 0 <= args.min_learning_rate <= args.learning_rate:
        raise ValueError("min learning rate must be in [0, learning rate]")
    if args.gradient_clip <= 0:
        raise ValueError("gradient clip must be positive")
    if not 0 < args.rollout_discount <= 1:
        raise ValueError("rollout discount must be in (0, 1]")
    if not 0 <= args.terminal_loss_weight <= 1:
        raise ValueError("terminal loss weight must be in [0, 1]")
    if min(args.spectral_loss_weight, args.channel_relative_weight) < 0:
        raise ValueError("auxiliary loss weights cannot be negative")
    if args.input_noise_std < 0:
        raise ValueError("input noise standard deviation cannot be negative")
    if args.num_workers < 0 or args.prefetch_factor < 1:
        raise ValueError(
            "num_workers cannot be negative; prefetch_factor must be positive"
        )
    if args.persistent_workers:
        raise ValueError(
            "persistent training workers cannot be checkpointed exactly; use "
            "--no-persistent-workers (prefetching still remains asynchronous)"
        )
    if args.resume and args.allow_runtime_mismatch:
        raise ValueError("Exact resume cannot allow a runtime mismatch")
    if not args.stage_name.strip():
        raise ValueError("stage name cannot be empty")


def main() -> None:
    args = parse_args()
    _validate_cli(args)
    run_dir = args.run_dir.expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(log_file=run_dir / "train.log")
    seed_everything(args.seed)
    device = choose_device(args.device)
    torch.use_deterministic_algorithms(args.deterministic)
    runtime = validate_supported_runtime(device)
    amp_enabled = not args.no_amp and device.type == "cuda"
    paths = DataPaths(args.data_dir.expanduser().resolve())
    means_path = args.means_path or paths.means
    stds_path = args.stds_path or paths.stds
    climatology_path = args.climatology_path or means_path.parent / "time_means.npy"
    if not climatology_path.expanduser().resolve().is_file():
        raise FileNotFoundError(f"Missing training climatology: {climatology_path}")
    time_diff_stds_path = args.time_diff_stds_path or paths.time_diff_stds(
        args.time_step
    )
    crop = tuple(args.spatial_crop) if args.spatial_crop else None
    train_generator = torch.Generator().manual_seed(args.seed)

    common_loader = dict(
        history=args.history,
        time_step=args.time_step,
        num_workers=args.num_workers,
        means_path=means_path,
        stds_path=stds_path,
        max_samples=args.max_samples,
        spatial_crop=cast(tuple[int, int] | None, crop),
        input_noise_std=args.input_noise_std,
        pin_memory=device.type == "cuda",
        prefetch_factor=args.prefetch_factor,
        persistent_workers=args.persistent_workers,
    )
    train_loader, train_dataset = build_loader(
        paths,
        "train",
        rollout_steps=args.rollout_steps,
        batch_size=args.batch_size,
        data_path=args.train_data,
        generator=train_generator,
        **common_loader,
    )
    valid_loader, valid_dataset = build_loader(
        paths,
        "valid",
        rollout_steps=args.validation_rollout_steps,
        batch_size=args.validation_batch_size,
        data_path=args.valid_data,
        **{
            **common_loader,
            "max_samples": args.max_validation_samples,
            "sample_selection": "evenly_spaced",
            "num_workers": args.validation_num_workers,
            "prefetch_factor": 1,
            "lazy_targets": True,
        },
    )
    if any(not unit.strip() for unit in train_dataset.channel_units):
        raise ValueError(
            "Training data is missing per-channel physical units; regenerate "
            "legacy flattened stores with the current downloader"
        )
    if train_dataset.channel_units != valid_dataset.channel_units:
        raise ValueError("Train and validation channel units must match")
    statistic_paths = {
        "means": means_path,
        "stds": stds_path,
        "climatology": climatology_path,
    }
    if args.temp_diff_normalization:
        statistic_paths["time_diff_stds"] = time_diff_stds_path
    validate_statistics_bundle(train_dataset, statistic_paths, args.time_step)
    if train_dataset.spatial_shape != valid_dataset.spatial_shape:
        raise ValueError("Train and validation grids must match")
    if train_dataset.cadence_hours != valid_dataset.cadence_hours:
        raise ValueError("Train and validation temporal cadence must match")
    if not np.array_equal(train_dataset.longitudes, valid_dataset.longitudes):
        raise ValueError("Train and validation longitude coordinates must match")
    if not np.array_equal(train_dataset.latitudes, valid_dataset.latitudes):
        raise ValueError("Train and validation latitude coordinates must match")
    split_overlap = np.datetime64(train_dataset.last_time) >= np.datetime64(
        valid_dataset.first_time
    )
    if split_overlap and not args.allow_overlapping_splits:
        raise ValueError(
            "Train and validation periods overlap. Use chronological splits; "
            "the override is intended only for pipeline diagnostics."
        )
    if split_overlap:
        logger.warning("Train/validation overlap enabled for a diagnostic run")
    if crop is not None:
        raise ValueError(
            "Spatial cropping is incompatible with spherical forecasting; use a globally regridded diagnostic store"
        )
    validate_global_grid(train_dataset)
    validate_global_grid(valid_dataset)

    channels = train_dataset.config.channel_count
    model_config = AtmosphereModelConfig(
        img_size=train_dataset.spatial_shape,
        in_channels=channels * (args.history + 1),
        out_channels=channels,
        scale_factor=args.scale_factor,
        embed_dim=args.embed_dim,
        num_layers=args.num_layers,
        drop_rate=args.drop_rate,
        drop_path_rate=args.drop_path_rate,
        hard_thresholding_fraction=args.hard_thresholding_fraction,
        pos_embed=args.pos_embed,
        stabilize_sht_constants=args.stabilize_sht_constants,
    )
    latitudes = torch.as_tensor(train_dataset.latitudes, device=device)
    global_stds = np.load(stds_path)
    delta_stds = None
    if args.temp_diff_normalization:
        if not time_diff_stds_path.expanduser().resolve().is_file():
            raise FileNotFoundError(
                "Missing time-difference statistics: "
                f"{time_diff_stds_path}. Run scripts/compute_stats.py "
                f"--time-step {args.time_step} on the training split."
            )
        delta_stds = np.load(time_diff_stds_path)
    if args.channel_weighting == "auto":
        channel_weights = makani_auto_channel_weights(
            train_dataset.channel_names,
            normalization_stds=global_stds if delta_stds is not None else None,
            time_difference_stds=delta_stds,
        )
    else:
        channel_weights = torch.full((channels,), 1.0 / channels)
        if delta_stds is not None:
            global_stds_tensor = torch.as_tensor(global_stds).flatten()
            delta_stds_tensor = torch.as_tensor(delta_stds).flatten()
            if (
                global_stds_tensor.numel() != channels
                or delta_stds_tensor.numel() != channels
            ):
                raise ValueError("Loss statistics must match the output channel count")
            if (
                not torch.isfinite(global_stds_tensor).all()
                or not torch.isfinite(delta_stds_tensor).all()
                or (global_stds_tensor <= 0).any()
                or (delta_stds_tensor <= 0).any()
            ):
                raise ValueError("Loss standard deviations must be finite and positive")
            channel_weights *= global_stds_tensor / delta_stds_tensor.clamp_min(1e-6)
    channel_weights = channel_weights.to(device=device)
    spatial_loss = CombinedAtmosphereLoss(
        spectral_weight=args.spectral_loss_weight,
        latitudes=latitudes,
        channel_weights=channel_weights,
        use_loss_scaler=args.use_loss_scaler,
    )
    relative_loss = ChannelRelativeAtmosphereLoss(latitudes=latitudes)

    def training_loss(prediction: Tensor, target: Tensor) -> Tensor:
        total = spatial_loss(prediction, target)
        if args.channel_relative_weight:
            relative, _ = relative_loss(prediction, target)
            total = total + args.channel_relative_weight * relative
        return total

    checkpoint = None
    checkpoint_source = args.resume or args.init_checkpoint
    if checkpoint_source is not None:
        checkpoint = load_checkpoint(checkpoint_source.expanduser().resolve(), device)
        mismatches = enforce_checkpoint_runtime(
            checkpoint,
            runtime,
            allow_mismatch=bool(args.init_checkpoint and args.allow_runtime_mismatch),
        )
        if mismatches:
            logger.warning("Runtime mismatch: %s", ", ".join(mismatches))
        saved_model_config = model_config_from_dict(checkpoint["model_config"])
        if not same_model_architecture(saved_model_config, model_config):
            raise ValueError("Checkpoint model architecture does not match this stage")

    model = build_model(model_config, device)
    if checkpoint is not None and args.init_checkpoint:
        model.load_state_dict(checkpoint["model_state"])
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )
    updates_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation)
    scheduler_epochs = (
        args.scheduler_epochs
        if args.scheduler_epochs is not None
        else (args.epochs if args.epochs > 0 else TrainingConfig().scheduler_epochs)
    )
    total_updates = updates_per_epoch * scheduler_epochs
    warmup_updates = min(
        updates_per_epoch * args.warmup_epochs,
        max(total_updates - 1, 0),
    )
    scheduler = create_warmup_cosine_scheduler(
        optimizer,
        total_updates=total_updates,
        warmup_updates=warmup_updates,
        warmup_start_factor=args.warmup_start_factor,
        min_learning_rate=args.min_learning_rate,
        base_learning_rate=args.learning_rate,
    )
    scaler = GradScaler(
        device.type,
        enabled=amp_enabled and args.amp_dtype == "float16",
    )
    resume_signature = {
        "stage_name": args.stage_name,
        "epochs": args.epochs,
        "scheduler_epochs": scheduler_epochs,
        "batch_size": args.batch_size,
        "validation_batch_size": args.validation_batch_size,
        "gradient_accumulation": args.gradient_accumulation,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "min_learning_rate": args.min_learning_rate,
        "warmup_updates": warmup_updates,
        "warmup_start_factor": args.warmup_start_factor,
        "max_samples": args.max_samples,
        "max_validation_samples": args.max_validation_samples,
        "validation_num_workers": args.validation_num_workers,
        "deterministic": args.deterministic,
        "forecast_contract": forecast_contract(train_dataset, args.time_step),
        "diagnostic_run": bool(args.allow_overlapping_splits),
        "total_updates": total_updates,
        "gradient_clip": args.gradient_clip,
        "rollout_steps": args.rollout_steps,
        "validation_rollout_steps": args.validation_rollout_steps,
        "rollout_discount": args.rollout_discount,
        "history": args.history,
        "time_step": args.time_step,
        "num_workers": args.num_workers,
        "prefetch_factor": args.prefetch_factor,
        "persistent_workers": args.persistent_workers,
        "seed": args.seed,
        "device": str(device),
        "amp_enabled": amp_enabled,
        "amp_dtype": args.amp_dtype,
        "gradient_checkpointing": args.gradient_checkpointing,
        "allow_overlapping_splits": args.allow_overlapping_splits,
        "patience": args.patience,
        "early_stopping_min_delta": args.early_stopping_min_delta,
        "terminal_loss_weight": args.terminal_loss_weight,
        "spectral_loss_weight": args.spectral_loss_weight,
        "channel_relative_weight": args.channel_relative_weight,
        "channel_weighting": args.channel_weighting,
        "temp_diff_normalization": args.temp_diff_normalization,
        "use_loss_scaler": args.use_loss_scaler,
        "input_noise_std": args.input_noise_std,
        "runtime": runtime,
        "data_signature": {
            "train": dataset_signature(train_dataset),
            "valid": dataset_signature(valid_dataset),
            "means": statistics_signature(means_path),
            "stds": statistics_signature(stds_path),
            "climatology": statistics_signature(climatology_path),
            "time_diff_stds": (
                statistics_signature(time_diff_stds_path)
                if args.temp_diff_normalization
                else None
            ),
        },
    }

    start_epoch = 0
    completed_updates = 0
    best_loss = float("inf")
    epochs_without_improvement = 0
    if checkpoint is not None and args.resume:
        saved_training = checkpoint.get("training_config", {})
        mismatches = [
            key
            for key, value in resume_signature.items()
            if saved_training.get(key) != value
        ]
        if mismatches:
            raise ValueError(
                "Resume must preserve the numerical trajectory; mismatched: "
                + ", ".join(mismatches)
            )
        restore_training_state(
            checkpoint,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            train_generator=train_generator,
        )
        start_epoch = int(checkpoint["epoch"]) + 1
        completed_updates = int(checkpoint["completed_updates"])
        if scheduler.last_epoch != completed_updates:
            raise ValueError(
                "Checkpoint scheduler progress does not match completed_updates"
            )
        best_loss = float(checkpoint["best_validation_loss"])
        epochs_without_improvement = int(checkpoint["epochs_without_improvement"])
        if checkpoint["early_stopped"]:
            if not args.reset_early_stopping:
                raise ValueError(
                    "Checkpoint already stopped early; use "
                    "--reset-early-stopping to continue intentionally"
                )
            epochs_without_improvement = 0
    if args.epochs > 0 and start_epoch >= args.epochs:
        raise ValueError("Checkpoint has already reached the requested epochs")

    history_path = run_dir / "history.csv"
    if history_path.exists() and not args.resume:
        raise FileExistsError(
            f"{history_path} exists; choose a new run directory or resume"
        )
    if args.resume:
        for ledger in (history_path, run_dir / "validation_by_lead.csv"):
            if ledger.exists():
                with ledger.open(newline="") as file:
                    reader = csv.DictReader(file)
                    fields = reader.fieldnames
                    if fields is None:
                        raise ValueError(f"History ledger has no header: {ledger}")
                    rows = [row for row in reader if int(row["epoch"]) <= start_epoch]
                shutil.copy2(
                    ledger,
                    ledger.with_name(ledger.name + f".before-resume-{time.time_ns()}"),
                )
                with ledger.open("w", newline="") as file:
                    writer = csv.DictWriter(file, fieldnames=fields)
                    writer.writeheader()
                    writer.writerows(rows)
        for stale in (run_dir / "validation").glob("epoch_*.json"):
            if int(stale.stem.split("_")[-1]) > start_epoch:
                stale.rename(
                    stale.with_name(stale.name + f".before-resume-{time.time_ns()}")
                )
    training_config = vars(args).copy()
    training_config.update(resume_signature)
    save_json(
        run_dir / "config.json",
        {
            "model": config_dict(model_config),
            "training": training_config,
            "train_samples": len(train_dataset),
            "valid_samples": len(valid_dataset),
            "channel_names": train_dataset.channel_names,
            "loss_channel_weights": channel_weights.tolist(),
        },
    )
    logger.info(
        "stage=%s device=%s parameters=%d train=%d valid=%d "
        "effective_batch=%d validation_batch=%d train_rollout=%d "
        "validation_rollout=%d history=%d",
        args.stage_name,
        device,
        count_parameters(model),
        len(train_dataset),
        len(valid_dataset),
        args.batch_size * args.gradient_accumulation,
        args.validation_batch_size,
        args.rollout_steps,
        args.validation_rollout_steps,
        args.history,
    )
    logger.info(
        "data_loader workers=%d prefetch_factor=%d persistent=%s pin_memory=%s",
        args.num_workers,
        args.prefetch_factor,
        bool(train_loader.persistent_workers),
        bool(train_loader.pin_memory),
    )
    logger.info(
        "loss channel_weighting=%s temp_diff_normalization=%s "
        "loss_scaler=%s input_noise_std=%.4g range=[%.4g, %.4g]",
        args.channel_weighting,
        args.temp_diff_normalization,
        args.use_loss_scaler,
        args.input_noise_std,
        float(channel_weights.min()),
        float(channel_weights.max()),
    )

    try:
        epoch_iterator = (
            range(start_epoch, args.epochs)
            if args.epochs > 0
            else itertools.count(start_epoch)
        )
        for epoch in epoch_iterator:
            started = time.perf_counter()
            model.train()
            optimizer.zero_grad(set_to_none=True)
            train_loss_sum = 0.0
            train_samples = 0
            data_wait_seconds = 0.0
            gradient_norm_sum = 0.0
            gradient_norm_max = 0.0
            gradient_norm_measurements = 0
            clipped_updates = 0
            nonfinite_gradient_norms = 0
            previous_batch_finished = time.perf_counter()
            for batch_index, raw_batch in enumerate(train_loader):
                batch_received = time.perf_counter()
                data_wait_seconds += batch_received - previous_batch_finished
                batch = move_batch(raw_batch, device)
                size = batch["input"].shape[0]
                bucket_start = (
                    batch_index // args.gradient_accumulation
                ) * args.gradient_accumulation
                bucket_size = min(
                    args.gradient_accumulation,
                    len(train_loader) - bucket_start,
                )
                with amp_context(device, amp_enabled, args.amp_dtype):
                    loss, _, _ = rollout_loss(
                        model,
                        batch,
                        args.rollout_steps,
                        gradient_checkpointing=args.gradient_checkpointing,
                        discount_factor=args.rollout_discount,
                        loss_fn=training_loss,
                    )
                    # A final partial accumulation bucket must not be divided
                    # equally by mini-batch count: its final mini-batch may be
                    # shorter. Weight by samples so every temporal window has
                    # exactly the same contribution.
                    bucket_samples = accumulation_bucket_sample_count(
                        batch_index=batch_index,
                        total_batches=len(train_loader),
                        total_samples=len(train_dataset),
                        batch_size=args.batch_size,
                        accumulation_steps=args.gradient_accumulation,
                    )
                    scaled_loss = loss * (size / bucket_samples)
                if not torch.isfinite(loss.detach()).all():
                    optimizer.zero_grad(set_to_none=True)
                    raise FloatingPointError(
                        f"Non-finite training loss at epoch {epoch}, batch {batch_index}; last checkpoint preserved"
                    )
                cast(Tensor, scaler.scale(scaled_loss)).backward()
                should_step = batch_index + 1 == bucket_start + bucket_size
                if should_step:
                    scaler.unscale_(optimizer)
                    preclip_norm = float(
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(),
                            args.gradient_clip,
                            error_if_nonfinite=True,
                        )
                    )
                    if math.isfinite(preclip_norm):
                        gradient_norm_sum += preclip_norm
                        gradient_norm_max = max(gradient_norm_max, preclip_norm)
                        gradient_norm_measurements += 1
                        clipped_updates += int(preclip_norm > args.gradient_clip)
                    else:
                        nonfinite_gradient_norms += 1
                    previous_scale = scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer_ran = (
                        not scaler.is_enabled() or scaler.get_scale() >= previous_scale
                    )
                    if optimizer_ran:
                        scheduler.step()
                        completed_updates += 1
                    optimizer.zero_grad(set_to_none=True)
                train_loss_sum += float(loss.detach()) * size
                train_samples += size
                previous_batch_finished = time.perf_counter()

            train_loss = train_loss_sum / max(train_samples, 1)
            training_seconds = time.perf_counter() - started
            data_wait_fraction = data_wait_seconds / max(training_seconds, 1e-9)
            mean_gradient_norm = gradient_norm_sum / max(gradient_norm_measurements, 1)
            gradient_clip_fraction = clipped_updates / max(
                gradient_norm_measurements, 1
            )
            validation = validate(
                model,
                valid_loader,
                device,
                args.validation_rollout_steps,
                amp_enabled,
                args.amp_dtype,
                args.terminal_loss_weight,
                latitudes,
                training_loss,
            )
            improved = (
                validation.selection_score < best_loss - args.early_stopping_min_delta
            )
            if improved:
                best_loss = validation.selection_score
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            early_stopped = (
                args.patience > 0 and epochs_without_improvement >= args.patience
            )
            save_checkpoint(
                run_dir / "checkpoints" / "last.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                train_generator=train_generator,
                epoch=epoch,
                completed_updates=completed_updates,
                best_validation_loss=best_loss,
                epochs_without_improvement=epochs_without_improvement,
                early_stopped=early_stopped,
                model_config=config_dict(model_config),
                training_config=training_config,
            )
            if improved:
                save_checkpoint(
                    run_dir / "checkpoints" / "best.pt",
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    train_generator=train_generator,
                    epoch=epoch,
                    completed_updates=completed_updates,
                    best_validation_loss=best_loss,
                    epochs_without_improvement=epochs_without_improvement,
                    early_stopped=early_stopped,
                    model_config=config_dict(model_config),
                    training_config=training_config,
                )

            elapsed = time.perf_counter() - started
            append_history(
                history_path,
                {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "validation_loss": validation.selection_score,
                    "validation_mean_loss": validation.mean_loss,
                    "validation_final_loss": validation.final_loss,
                    "validation_final_rmse": validation.lead_rmse[-1],
                    "validation_rmse_growth": (
                        validation.lead_rmse[-1] - validation.lead_rmse[0]
                    ),
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "seconds": elapsed,
                    "train_data_wait_seconds": data_wait_seconds,
                    "train_data_wait_fraction": data_wait_fraction,
                    "mean_preclip_gradient_norm": mean_gradient_norm,
                    "max_preclip_gradient_norm": gradient_norm_max,
                    "gradient_clip_fraction": gradient_clip_fraction,
                    "nonfinite_gradient_norms": nonfinite_gradient_norms,
                },
            )
            for lead, rmse in enumerate(validation.lead_rmse, start=1):
                append_history(
                    run_dir / "validation_by_lead.csv",
                    {
                        "epoch": epoch + 1,
                        "lead": lead,
                        "lead_hours": (
                            lead * args.time_step * train_dataset.cadence_hours
                        ),
                        "normalized_rmse": rmse,
                        "rmse_growth_from_lead1": rmse - validation.lead_rmse[0],
                    },
                )
            lead_hours = args.time_step * train_dataset.cadence_hours
            validation_milestones = [
                {
                    "lead": lead,
                    "lead_hours": lead * lead_hours,
                    "lead_days": lead * lead_hours / 24.0,
                    "loss": validation.lead_losses[lead - 1],
                    "normalized_rmse": validation.lead_rmse[lead - 1],
                }
                for lead in range(1, args.validation_rollout_steps + 1)
                if math.isclose((lead * lead_hours) % 72.0, 0.0, abs_tol=1e-8)
            ]
            save_json(
                run_dir / "validation" / f"epoch_{epoch + 1:04d}.json",
                {
                    "epoch": epoch + 1,
                    "validation_rollout_steps": args.validation_rollout_steps,
                    "forecast_step_hours": lead_hours,
                    "selection_score": validation.selection_score,
                    "mean_loss": validation.mean_loss,
                    "final_loss": validation.final_loss,
                    "milestones_every_days": 3,
                    "milestones": validation_milestones,
                },
            )
            logger.info(
                "epoch=%d train=%.6g selection=%.6g valid_final=%.6g "
                "final_rmse=%.6g lr=%.3g time=%.1fs data_wait=%.1f%% "
                "grad_norm=%.4g clip=%.1f%% nonfinite=%d%s",
                epoch + 1,
                train_loss,
                validation.selection_score,
                validation.final_loss,
                validation.lead_rmse[-1],
                optimizer.param_groups[0]["lr"],
                elapsed,
                100.0 * data_wait_fraction,
                mean_gradient_norm,
                100.0 * gradient_clip_fraction,
                nonfinite_gradient_norms,
                " best" if improved else "",
            )
            if early_stopped:
                logger.info("Early stopping after %d stale epochs", args.patience)
                break
    except KeyboardInterrupt:
        logger.warning(
            "Training interrupted manually; resume from %s",
            run_dir / "checkpoints" / "last.pt",
        )
        raise
    finally:
        train_dataset.close()
        valid_dataset.close()


if __name__ == "__main__":
    main()
