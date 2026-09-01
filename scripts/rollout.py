"""Run autoregressive inference and stream physical fields to a Zarr store."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT_ROOT), str(PROJECT_ROOT / "src")]

import numpy as np
import torch
import zarr

from configs.pipeline_config import DataPaths
from neural_atmosphere_operator.pipeline.checkpoint import load_checkpoint
from neural_atmosphere_operator.pipeline.dependencies import (
    enforce_checkpoint_runtime,
    validate_supported_runtime,
)
from neural_atmosphere_operator.pipeline.forecast import autoregressive_predictions
from neural_atmosphere_operator.pipeline.runtime import (
    amp_context,
    build_loader,
    build_model,
    checkpoint_time_step,
    choose_device,
    model_config_from_dict,
    move_batch,
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
    parser.add_argument("--split", choices=("train", "valid", "test"), default="test")
    parser.add_argument("--rollout-steps", type=int, default=4)
    parser.add_argument("--time-step", type=int)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp-dtype", choices=("bfloat16", "float16"))
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--allow-runtime-mismatch", action="store_true")
    parser.add_argument("--allow-data-mismatch", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if min(args.rollout_steps, args.samples) < 1 or args.start_index < 0:
        raise ValueError("Positive rollout/samples and non-negative start are required")
    checkpoint_path = args.checkpoint.expanduser().resolve()
    output = (
        args.output.expanduser().resolve()
        if args.output
        else checkpoint_path.parents[1] / f"rollout_{args.split}.zarr"
    )
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing rollout: {output}")
    temporary = output.with_name(output.name + ".part")
    if temporary.exists():
        raise FileExistsError(f"Stale partial rollout exists: {temporary}")
    logger = setup_logger(log_file=output.with_suffix(".log"))
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
    training = checkpoint.get("training_config", {})
    amp_dtype = args.amp_dtype or str(training.get("amp_dtype", "bfloat16"))
    history = model_config.in_channels // model_config.out_channels - 1
    time_step = checkpoint_time_step(training, args.time_step)
    means_path = args.means_path or paths.means
    stds_path = args.stds_path or paths.stds
    saved_data_signature = training.get("data_signature", {})
    for name, path in (("means", means_path), ("stds", stds_path)):
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
        batch_size=1,
        num_workers=0,
        data_path=args.data_path,
        means_path=means_path,
        stds_path=stds_path,
        spatial_crop=model_config.img_size,
    )
    del loader
    if args.start_index + args.samples > len(dataset):
        raise IndexError("Requested rollout samples exceed the dataset")

    channels = model_config.out_channels
    height, width = dataset.spatial_shape
    group = zarr.open_group(str(temporary), mode="w")
    shape = (args.samples, args.rollout_steps, channels, height, width)
    chunks = (1, 1, 1, height, width)
    predictions = group.create_dataset(
        "prediction", shape=shape, chunks=chunks, dtype="f4"
    )
    targets = group.create_dataset(
        "target", shape=shape, chunks=chunks, dtype="f4"
    )
    time_indices = group.create_dataset(
        "initial_time_index", shape=(args.samples,), dtype="i8"
    )
    initial_times = group.create_dataset(
        "initial_time_unix_ns", shape=(args.samples,), dtype="i8"
    )
    valid_times = group.create_dataset(
        "valid_time_unix_ns",
        shape=(args.samples, args.rollout_steps),
        chunks=(1, args.rollout_steps),
        dtype="i8",
    )
    group.create_dataset("latitude", data=dataset.latitudes)
    group.create_dataset("longitude", data=dataset.longitudes)
    group.attrs.update(
        checkpoint=str(checkpoint_path),
        split=args.split,
        rollout_steps=args.rollout_steps,
        time_step_hours=time_step * dataset.cadence_hours,
        time_encoding="nanoseconds since 1970-01-01T00:00:00",
        channel_names=list(dataset.channel_names),
        channel_units=list(dataset.channel_units),
    )
    amp_enabled = (
        not args.no_amp
        and not bool(training.get("no_amp", False))
        and device.type == "cuda"
    )
    try:
        with torch.inference_mode():
            for output_index, sample_index in enumerate(
                range(args.start_index, args.start_index + args.samples)
            ):
                raw = dataset[sample_index]
                batch = move_batch(
                    {
                        "input": raw["input"].unsqueeze(0),
                        "target": raw["target"].unsqueeze(0),
                        "time_index": raw["time_index"].unsqueeze(0),
                    },
                    device,
                )
                with amp_context(device, amp_enabled, amp_dtype):
                    for lead, (prediction, target) in enumerate(
                        autoregressive_predictions(
                            model, batch, args.rollout_steps
                        )
                    ):
                        prediction_raw = dataset.normalizer.denormalize(
                            prediction.float().cpu()
                        )
                        target_raw = dataset.normalizer.denormalize(
                            target.float().cpu()
                        )
                        predictions[output_index, lead] = np.asarray(prediction_raw[0])
                        targets[output_index, lead] = np.asarray(target_raw[0])
                time_indices[output_index] = int(raw["time_index"])
                base_index = int(raw["time_index"])
                initial_times[output_index] = int(
                    dataset.times[base_index].astype(np.int64)
                )
                for lead in range(args.rollout_steps):
                    valid_index = base_index + (lead + 1) * time_step
                    valid_times[output_index, lead] = int(
                        dataset.times[valid_index].astype(np.int64)
                    )
                logger.info("Saved sample %d", sample_index)
        os.replace(temporary, output)
    finally:
        dataset.close()
    logger.info("Rollout written to %s", output)


if __name__ == "__main__":
    main()
