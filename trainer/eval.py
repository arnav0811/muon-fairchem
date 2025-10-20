"""Entry point for evaluating a trained FAIR-Chem model checkpoint."""
from __future__ import annotations

import argparse
import math
import pathlib
import sys
from typing import Any, Dict

import torch
import torch.nn.functional as F
from .train import _epoch_metric  # Reuse metric calculation
from utils.datamodule import build_dataloaders
from models.e3_model import build_model

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained FAIR-Chem model.")
    parser.add_argument(
        "--checkpoint",
        type=pathlib.Path,
        required=True,
        help="Path to the .pt checkpoint file.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for evaluation.",
    )
    return parser.parse_args()


def evaluate(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    running_mae = 0.0
    running_mse = 0.0
    n_samples = 0

    # Per-structure metrics
    running_mae_low = 0.0
    n_low = 0
    running_mae_medium = 0.0
    n_medium = 0
    running_mae_high = 0.0
    n_high = 0

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            target = batch.energy
            prediction = model(batch)["energy"]["energy"]

            batch_size = target.numel()
            n_samples += batch_size
            running_mae += F.l1_loss(prediction, target, reduction="sum").item()
            running_mse += F.mse_loss(prediction, target, reduction="sum").item()

            if hasattr(batch, "prototype_error"):
                errors = batch.prototype_error.view(-1)
                low_mask = errors < 0.01
                medium_mask = (errors >= 0.01) & (errors < 0.1)
                high_mask = errors >= 0.1

                if low_mask.any():
                    running_mae_low += F.l1_loss(
                        prediction[low_mask], target[low_mask], reduction="sum"
                    ).item()
                    n_low += low_mask.sum().item()
                if medium_mask.any():
                    running_mae_medium += F.l1_loss(
                        prediction[medium_mask], target[medium_mask], reduction="sum"
                    ).item()
                    n_medium += medium_mask.sum().item()
                if high_mask.any():
                    running_mae_high += F.l1_loss(
                        prediction[high_mask], target[high_mask], reduction="sum"
                    ).item()
                    n_high += high_mask.sum().item()

    return {
        "mae": _epoch_metric(running_mae, n_samples),
        "rmse": math.sqrt(_epoch_metric(running_mse, n_samples)),
        "mae_low_rattle": _epoch_metric(running_mae_low, n_low),
        "mae_medium_rattle": _epoch_metric(running_mae_medium, n_medium),
        "mae_high_rattle": _epoch_metric(running_mae_high, n_high),
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    print(f"Loading checkpoint from: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint["config"]

    print("Building test dataloader...")
    loaders = build_dataloaders(config["dataset"], device=args.device)
    test_loader = loaders.get("test")
    if test_loader is None:
        raise RuntimeError("No 'test' split found in the dataset.")

    print("Building model...")
    model = build_model(config["model"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    print("Running evaluation on the test set...")
    metrics = evaluate(model, test_loader, device)

    print("\n--- Final Test Set Results ---")
    print(f"  Overall MAE:         {metrics['mae']:.4f}")
    print(f"  Overall RMSE:        {metrics['rmse']:.4f}")
    print("-" * 30)
    print(f"  MAE (Low Rattle):    {metrics['mae_low_rattle']:.4f}")
    print(f"  MAE (Medium Rattle): {metrics['mae_medium_rattle']:.4f}")
    print(f"  MAE (High Rattle):   {metrics['mae_high_rattle']:.4f}")
    print("----------------------------\n")


if __name__ == "__main__":
    main()
