"""Plot training history and evaluation lead-time metrics."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def read_csv(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"No rows in {path}")
    return {
        key: np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        for key in rows[0]
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--evaluation-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    import matplotlib.pyplot as plt

    run_dir = args.run_dir.expanduser().resolve()
    output = (
        args.output_dir.expanduser().resolve()
        if args.output_dir else run_dir / "figures"
    )
    output.mkdir(parents=True, exist_ok=True)
    history_path = run_dir / "history.csv"
    if history_path.exists():
        history = read_csv(history_path)
        figure, axis = plt.subplots(figsize=(8, 4.5))
        axis.plot(history["epoch"], history["train_loss"], label="Train")
        axis.plot(history["epoch"], history["validation_loss"], label="Validation")
        axis.set(xlabel="Epoch", ylabel="Loss", title="SFNO training history")
        axis.set_yscale("log")
        axis.grid(alpha=0.25)
        lr_axis = axis.twinx()
        lr_axis.plot(
            history["epoch"], history["learning_rate"],
            color="tab:gray", linestyle="--", label="Learning rate",
        )
        lr_axis.set_ylabel("Learning rate")
        lines = axis.lines + lr_axis.lines
        axis.legend(lines, [str(line.get_label()) for line in lines])
        figure.tight_layout()
        figure.savefig(output / "training_history.png", dpi=180)
        plt.close(figure)

    evaluation = (
        args.evaluation_dir.expanduser().resolve()
        if args.evaluation_dir else run_dir / "evaluation_test"
    )
    metrics_path = evaluation / "metrics_by_lead.csv"
    if metrics_path.exists():
        metrics = read_csv(metrics_path)
        figure, axes = plt.subplots(2, 1, figsize=(8, 7), constrained_layout=True)
        axes[0].plot(metrics["lead_hours"], metrics["normalized_rmse"], marker="o")
        axes[0].set(xlabel="Lead time (hours)", ylabel="Normalized RMSE")
        axes[1].plot(
            metrics["lead_hours"], metrics["normalized_acc"],
            marker="o", color="tab:green",
        )
        axes[1].set(xlabel="Lead time (hours)", ylabel="ACC", ylim=(-1, 1))
        for axis in axes:
            axis.grid(alpha=0.25)
        figure.savefig(output / "metrics_by_lead.png", dpi=180)
        plt.close(figure)
    print(f"Charts written to {output}")


if __name__ == "__main__":
    main()
