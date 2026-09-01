"""Evaluate an SFNO checkpoint with per-lead and per-channel metrics."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT_ROOT), str(PROJECT_ROOT / "src")]

import numpy as np
import torch

from configs.pipeline_config import DataPaths
from neural_atmosphere_operator.pipeline.checkpoint import load_checkpoint
from neural_atmosphere_operator.pipeline.dependencies import (
    enforce_checkpoint_runtime,
    validate_supported_runtime,
)
from neural_atmosphere_operator.pipeline.forecast import autoregressive_predictions
from neural_atmosphere_operator.pipeline.metrics import (
    LeadMetricAccumulator,
    channel_stds_tensor,
    normalized_climatology_tensor,
)
from neural_atmosphere_operator.pipeline.runtime import (
    amp_context,
    build_loader,
    build_model,
    checkpoint_time_step,
    choose_device,
    model_config_from_dict,
    move_batch,
    save_json,
    statistics_signature,
)
from neural_atmosphere_operator.utils.logger import setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=DataPaths().root)
    parser.add_argument("--data-path", type=Path)
    parser.add_argument("--means-path", type=Path)
    parser.add_argument("--stds-path", type=Path)
    parser.add_argument("--climatology-path", type=Path)
    parser.add_argument("--split", choices=("train", "valid", "test"), default="test")
    parser.add_argument("--rollout-steps", type=int, default=1)
    parser.add_argument(
        "--report-every-days",
        type=int,
        help="Also write milestone JSON at this many forecast days",
    )
    parser.add_argument("--time-step", type=int)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument(
        "--persistent-workers",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp-dtype", choices=("bfloat16", "float16"))
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--allow-runtime-mismatch", action="store_true")
    parser.add_argument("--allow-data-mismatch", action="store_true")
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_milestone_report(
    rows: list[dict[str, Any]],
    channel_rows: list[dict[str, Any]],
    every_days: int,
) -> list[dict[str, Any]]:
    """Group aggregate and per-channel metrics at fixed forecast-day levels."""
    if every_days < 1:
        raise ValueError("report interval must be positive")
    interval_hours = every_days * 24
    selected = [
        row
        for row in rows
        if np.isclose(
            float(row["lead_hours"]) % interval_hours,
            0.0,
            atol=1e-6,
        )
    ]
    levels: list[dict[str, Any]] = []
    for level, row in enumerate(selected, start=1):
        lead = int(row["lead"])
        lead_days = float(row["lead_hours"]) / 24.0
        levels.append(
            {
                "level": level,
                "name": f"day_{int(round(lead_days)):02d}",
                "lead_steps": lead,
                "lead_hours": float(row["lead_hours"]),
                "lead_days": lead_days,
                "aggregate": row,
                "channels": [
                    channel for channel in channel_rows if int(channel["lead"]) == lead
                ],
            }
        )
    return levels


def main() -> None:
    args = parse_args()
    if min(args.rollout_steps, args.batch_size) < 1:
        raise ValueError("rollout steps and batch size must be positive")
    if args.report_every_days is not None and args.report_every_days < 1:
        raise ValueError("report-every-days must be positive")
    checkpoint_path = args.checkpoint.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else checkpoint_path.parents[1] / f"evaluation_{args.split}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(log_file=output_dir / "evaluate.log")
    device = choose_device(args.device)
    runtime = validate_supported_runtime(device)
    checkpoint = load_checkpoint(checkpoint_path, device)
    mismatches = enforce_checkpoint_runtime(
        checkpoint, runtime, allow_mismatch=args.allow_runtime_mismatch
    )
    if mismatches:
        logger.warning("Runtime mismatch: %s", ", ".join(mismatches))
    model_config = model_config_from_dict(checkpoint["model_config"])
    model = build_model(model_config, device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    paths = DataPaths(args.data_dir.expanduser().resolve())
    training_config = checkpoint.get("training_config", {})
    amp_dtype = args.amp_dtype or str(training_config.get("amp_dtype", "bfloat16"))
    history = model_config.in_channels // model_config.out_channels - 1
    time_step = checkpoint_time_step(training_config, args.time_step)
    means_path = args.means_path or paths.means
    stds_path = args.stds_path or paths.stds
    climatology_path = args.climatology_path or means_path.parent / "time_means.npy"
    saved_data_signature = training_config.get("data_signature", {})
    for name, path in (
        ("means", means_path),
        ("stds", stds_path),
        ("climatology", climatology_path),
    ):
        saved = saved_data_signature.get(name)
        current = statistics_signature(path)
        if isinstance(saved, dict) and saved.get("sha256") != current["sha256"]:
            if not args.allow_data_mismatch:
                raise RuntimeError(
                    f"{name} statistics differ from checkpoint training data"
                )
            logger.warning("Using mismatched %s statistics", name)
    loader, dataset = build_loader(
        paths,
        args.split,
        rollout_steps=args.rollout_steps,
        history=history,
        time_step=time_step,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        prefetch_factor=args.prefetch_factor,
        persistent_workers=args.persistent_workers,
        data_path=args.data_path,
        means_path=means_path,
        stds_path=stds_path,
        max_samples=args.max_samples,
        spatial_crop=model_config.img_size,
    )
    channel_stds = channel_stds_tensor(np.load(stds_path), device)
    normalized_climatology = normalized_climatology_tensor(
        np.load(climatology_path),
        np.load(means_path),
        np.load(stds_path),
        device,
    )
    if tuple(normalized_climatology.shape[-2:]) != dataset.spatial_shape:
        normalized_climatology = normalized_climatology[
            ..., : dataset.spatial_shape[0], : dataset.spatial_shape[1]
        ]
    latitudes = torch.as_tensor(dataset.latitudes, device=device)
    accumulators = [LeadMetricAccumulator() for _ in range(args.rollout_steps)]
    amp_enabled = (
        not args.no_amp
        and not bool(training_config.get("no_amp", False))
        and device.type == "cuda"
    )
    try:
        with torch.inference_mode():
            for batch_index, raw_batch in enumerate(loader):
                batch = move_batch(raw_batch, device)
                with amp_context(device, amp_enabled, amp_dtype):
                    pairs = autoregressive_predictions(model, batch, args.rollout_steps)
                    for lead, (prediction, target) in enumerate(pairs):
                        accumulators[lead].update(
                            prediction.float(),
                            target.float(),
                            channel_stds,
                            latitudes,
                            normalized_climatology,
                        )
                if (batch_index + 1) % 20 == 0:
                    logger.info("Processed %d/%d batches", batch_index + 1, len(loader))
    finally:
        dataset.close()

    lead_hours = time_step * dataset.cadence_hours
    rows = [
        accumulator.result(lead, lead_hours)
        for lead, accumulator in enumerate(accumulators, start=1)
    ]
    first_rmse = float(rows[0]["normalized_rmse"])
    for row in rows:
        row["rmse_growth_from_lead1"] = float(row["normalized_rmse"]) - first_rmse
    channel_rows = [
        row
        for lead, accumulator in enumerate(accumulators, start=1)
        for row in accumulator.channel_results(
            lead, dataset.channel_names, lead_hours, dataset.channel_units
        )
    ]
    write_csv(output_dir / "metrics_by_lead.csv", rows)
    write_csv(output_dir / "metrics_by_channel.csv", channel_rows)
    summary = {
        "checkpoint": str(checkpoint_path),
        "split": args.split,
        "rollout_steps": args.rollout_steps,
        "samples": len(dataset),
        "data_first_time": dataset.first_time,
        "data_last_time": dataset.last_time,
        "terminal_normalized_rmse": float(rows[-1]["normalized_rmse"]),
        "terminal_normalized_acc": float(rows[-1]["normalized_acc"]),
        "acc_definition": (
            "mean_per_initialization_spatial_acc_with_training_long_term_climatology"
        ),
        "metrics": rows,
    }
    save_json(output_dir / "metrics.json", summary)
    if args.report_every_days is not None:
        levels = build_milestone_report(rows, channel_rows, args.report_every_days)
        if not levels:
            raise ValueError(
                "rollout horizon is shorter than the requested report interval"
            )
        save_json(
            output_dir / "milestones.json",
            {
                "schema_version": 1,
                "checkpoint": str(checkpoint_path),
                "split": args.split,
                "data_first_time": dataset.first_time,
                "data_last_time": dataset.last_time,
                "report_every_days": args.report_every_days,
                "rollout_steps": args.rollout_steps,
                "samples": len(dataset),
                "levels": levels,
            },
        )
    logger.info(
        "Evaluation complete: terminal RMSE=%.6g ACC=%.6g",
        summary["terminal_normalized_rmse"],
        summary["terminal_normalized_acc"],
    )


if __name__ == "__main__":
    main()
