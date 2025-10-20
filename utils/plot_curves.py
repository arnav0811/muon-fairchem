from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd


def plot_metric(df: pd.DataFrame, metric: str, label: str, out_dir: Path) -> None:
    plt.figure(figsize=(8, 5))
    for run, group in df.groupby("run"):
        group = group.sort_values("epoch")
        plt.plot(group["epoch"], group[metric], label=run)
    plt.xlabel("Epoch")
    plt.ylabel(label)
    if metric in {"val_mae", "val_rmse", "train_loss"}:
        plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    outfile = out_dir / f"curves_{metric}.png"
    plt.savefig(outfile, dpi=200)
    plt.close()
    print(f"Saved {outfile}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot training/validation curves.")
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path("results/metrics_clean.csv"),
        help="CSV produced by utils.parse_logs.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/figs"),
        help="Directory to write figures to.",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.metrics)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    metrics_to_plot: Dict[str, str] = {
        "train_loss": "Training Loss",
        "train_mae": "Training MAE",
        "val_mae": "Validation MAE",
        "val_rmse": "Validation RMSE",
    }

    for metric, label in metrics_to_plot.items():
        if metric not in df.columns:
            print(f"Skipping {metric}: column missing")
            continue
        plot_metric(df, metric, label, args.out_dir)


if __name__ == "__main__":
    main()
