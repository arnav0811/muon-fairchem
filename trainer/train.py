"""Entry point for training FAIR-Chem models with Muon or AdamW."""
from __future__ import annotations

import argparse
import csv
import math
import os
import pathlib
import sys
from typing import Any, Dict, Iterable, List, Optional

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import torch
import torch.nn.functional as F
import yaml

from models.e3_model import build_model
from optim.muon import Muon
from optim.routing import RoutedParams, split_params_for_muon
from utils.datamodule import build_dataloaders
from utils.logging import create_logger
from utils.svd_capture import save_svd_snapshot, select_layers_for_svd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train FAIR-Chem model with Muon")
    parser.add_argument(
        "--config",
        type=pathlib.Path,
        default=pathlib.Path("configs/omat24_baseline.yaml"),
        help="Path to a YAML config file.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for training.",
    )
    return parser.parse_args()


def load_config(path: pathlib.Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _zero_grad(optimizers: Iterable[torch.optim.Optimizer]) -> None:
    for opt in optimizers:
        opt.zero_grad(set_to_none=True)


def _step(optimizers: Iterable[torch.optim.Optimizer]) -> None:
    for opt in optimizers:
        opt.step()


def _epoch_metric(total: float, count: int) -> float:
    return total / max(count, 1)


class CSVLogger:
    header = (
        "run",
        "epoch",
        "train_loss",
        "train_mae",
        "val_mae",
        "val_rmse",
        "val_mae_low_rattle",
        "val_mae_medium_rattle",
        "val_mae_high_rattle",
    )

    def __init__(self, path: pathlib.Path, run_name: str) -> None:
        self.path = path
        self.run_name = run_name
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Always create a new file; do not append
        with self.path.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(self.header)

    def log(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Optional[Dict[str, float]],
    ) -> None:
        row = [
            self.run_name,
            epoch,
            train_metrics["loss"],
            train_metrics["mae"],
            val_metrics.get("mae", ""),
            val_metrics.get("rmse", ""),
            val_metrics.get("mae_low_rattle", ""),
            val_metrics.get("mae_medium_rattle", ""),
            val_metrics.get("mae_high_rattle", ""),
        ]
        with self.path.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(row)


def train_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizers: List[torch.optim.Optimizer],
    device: torch.device,
    grad_clip: float,
    log_interval: int,
    logger,
    svd_interval: int,
    svd_out_dir: pathlib.Path,
    svd_target_params: Dict[str, torch.nn.Parameter],
) -> Dict[str, float]:
    model.train()
    running_loss = 0.0
    running_mae = 0.0
    n_samples = 0

    for step, batch in enumerate(loader, start=1):
        batch = batch.to(device)
        target = batch.energy

        _zero_grad(optimizers)
        outputs = model(batch)
        prediction = outputs["energy"]["energy"]
        loss = F.mse_loss(prediction, target)
        loss.backward()

        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        _step(optimizers)

        if svd_interval and step % svd_interval == 0:
            for name, param in svd_target_params.items():
                if param.grad is None:
                    continue

                # Find the momentum buffer for this parameter across all optimizers
                update = None
                for opt in optimizers:
                    if param in opt.state:
                        state = opt.state[param]
                        if "momentum_buffer" in state:  # Muon
                            update = state["momentum_buffer"]
                            break
                        if "exp_avg" in state:  # Adam / AdamW
                            update = state["exp_avg"]
                            break

                if update is not None:
                    save_svd_snapshot(
                        layer_name=name,
                        weights=param.data,
                        updates=update,
                        step=step,
                        out_dir=svd_out_dir,
                    )

        batch_size = target.numel()
        n_samples += batch_size
        running_loss += loss.item() * batch_size
        running_mae += F.l1_loss(
            prediction.detach(), target.detach(), reduction="sum"
        ).item()

        if log_interval and step % log_interval == 0:
            logger.info(
                "train step %d | loss=%.4f | mae=%.4f",
                step,
                running_loss / n_samples,
                running_mae / n_samples,
            )

    return {
        "loss": _epoch_metric(running_loss, n_samples),
        "mae": _epoch_metric(running_mae, n_samples),
    }


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


def _save_checkpoint(
    path: pathlib.Path,
    epoch: int,
    model: torch.nn.Module,
    optimizers: List[torch.optim.Optimizer],
    train_metrics: Dict[str, float],
    val_metrics: Dict[str, float],
    config: Dict[str, Any],
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optim_states": [opt.state_dict() for opt in optimizers],
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "config": config,
        },
        path,
    )


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    logger = create_logger()
    logger.info("Loaded config from %s", args.config)

    set_seed(cfg.get("seed", 42))
    log_cfg = cfg.get("log", {})
    run_name = cfg.get("experiment_name", args.config.stem)

    loaders = build_dataloaders(cfg["dataset"], device=args.device)
    model = build_model(cfg["model"]).to(args.device)

    optim_cfg = cfg["optim"]
    routed: RoutedParams = split_params_for_muon(
        model,
        apply_layers=optim_cfg.get("apply_layers", "all"),
        first_k_blocks=optim_cfg.get("first_k_blocks"),
    )
    muon_params = list(routed.muon)
    adamw_params = list(routed.adamw)

    optimizers: List[torch.optim.Optimizer] = []
    use_muon = optim_cfg.get("use_muon", False)

    if use_muon:
        if len(muon_params) == 0:
            logger.warning(
                "Muon selected but no eligible parameters found; falling back to AdamW only"
            )
            use_muon = False
        else:
            optimizers.append(
                Muon(
                    muon_params,
                    lr=optim_cfg.get("muon_lr", optim_cfg.get("adamw_lr", 2e-3)),
                    weight_decay=optim_cfg.get("muon_wd", 0.0),
                    eps=optim_cfg.get("muon_eps", 1e-8),
                    momentum=optim_cfg.get("muon_momentum", 0.9),
                    orthogonalize=optim_cfg.get("ortho", True),
                )
            )
    if not use_muon:
        adamw_params.extend(muon_params)

    optimizers.append(
        torch.optim.AdamW(
            adamw_params,
            lr=optim_cfg.get("adamw_lr", 2e-3),
            weight_decay=optim_cfg.get("adamw_wd", 0.01),
            betas=(0.9, 0.999),
        )
    )

    train_loader = loaders["train"]
    val_loader = loaders.get("val")

    epochs = cfg["train"].get("epochs", 1)
    grad_clip = optim_cfg.get("grad_clip", 0.0)
    log_interval = cfg["log"].get("log_interval", 50)
    svd_interval = cfg["log"].get("svd_interval", 0)
    svd_layers = cfg["log"].get("svd_layers", [])
    svd_out_dir = pathlib.Path(f"results/svd_{run_name}")

    svd_target_params = {}
    if svd_interval > 0 and svd_layers:
        svd_target_params = select_layers_for_svd(model.named_parameters(), svd_layers)
        if not svd_target_params:
            logger.warning("SVD logging enabled, but no target layers found or specified.")
            svd_interval = 0  # Disable SVD logging if no layers match
        else:
            logger.info("SVD snapshots will be saved for: %s", list(svd_target_params.keys()))
            svd_out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    csv_path = pathlib.Path(cfg.get("log", {}).get("csv_path", "results/metrics.csv"))
    csv_logger = CSVLogger(csv_path, run_name)

    ckpt_cfg = cfg.get("checkpoint", {})
    ckpt_dir = pathlib.Path(ckpt_cfg.get("save_dir", "results/checkpoints"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    monitor_raw = ckpt_cfg.get("monitor", "val_mae")
    monitor = monitor_raw.replace("/", "_")
    mode = ckpt_cfg.get("mode", "min").lower()
    if monitor not in {"val_mae", "val/rmse"}:
        raise ValueError(
            "checkpoint.monitor must be 'val_mae', 'val/rmse', or their underscore forms"
        )
    if mode not in {"min", "max"}:
        raise ValueError("checkpoint.mode must be 'min' or 'max'")

    best_score: Optional[float] = None

    for epoch in range(1, epochs + 1):
        train_metrics = train_epoch(
            model=model,
            loader=train_loader,
            optimizers=optimizers,
            device=device,
            grad_clip=grad_clip,
            log_interval=log_interval,
            logger=logger,
            svd_interval=svd_interval,
            svd_out_dir=svd_out_dir,
            svd_target_params=svd_target_params,
        )
        logger.info(
            "epoch %d | train_loss=%.4f | train_mae=%.4f",
            epoch,
            train_metrics["loss"],
            train_metrics["mae"],
        )

        val_metrics = None
        if val_loader is not None and epoch % cfg["log"].get("eval_interval", 1) == 0:
            val_metrics = evaluate(
                model=model,
                loader=val_loader,
                device=device,
            )
            logger.info(
                "epoch %d | val_mae=%.4f | val_rmse=%.4f",
                epoch,
                val_metrics["mae"],
                val_metrics["rmse"],
            )
            logger.info(
                "epoch %d | val_mae_rattle (low/med/high)=%.4f/%.4f/%.4f",
                epoch,
                val_metrics["mae_low_rattle"],
                val_metrics["mae_medium_rattle"],
                val_metrics["mae_high_rattle"],
            )

        csv_logger.log(epoch, train_metrics, val_metrics)

        if val_metrics is not None:
            score = val_metrics["mae"] if monitor == "val_mae" else val_metrics["rmse"]
            improved = (
                best_score is None
                or (mode == "min" and score < best_score)
                or (mode == "max" and score > best_score)
            )
            if improved:
                best_score = score
                ckpt_path = ckpt_dir / f"epoch{epoch:03d}_val{score:.4f}.pt"
                _save_checkpoint(
                    path=ckpt_path,
                    epoch=epoch,
                    model=model,
                    optimizers=optimizers,
                    train_metrics=train_metrics,
                    val_metrics=val_metrics,
                    config=cfg,
                )
                logger.info("Checkpoint saved to %s", ckpt_path)


if __name__ == "__main__":
    main()
