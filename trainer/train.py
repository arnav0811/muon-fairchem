"""Entry point for training FAIR-Chem models with Muon or AdamW."""
from __future__ import annotations

import argparse
import math
import pathlib
import sys
from typing import Any, Dict, Iterable, List

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import torch
import torch.nn.functional as F
import yaml

from models.e3_model import build_model
from optim.muon import Muon
from optim.routing import split_params_for_muon
from utils.datamodule import build_dataloaders
from utils.logging import create_logger


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


def train_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizers: List[torch.optim.Optimizer],
    device: torch.device,
    grad_clip: float,
    log_interval: int,
    logger,
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

        batch_size = target.numel()
        n_samples += batch_size
        running_loss += loss.item() * batch_size
        running_mae += F.l1_loss(prediction.detach(), target.detach(), reduction="sum").item()

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

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            target = batch.energy
            prediction = model(batch)["energy"]["energy"]

            batch_size = target.numel()
            n_samples += batch_size
            running_mae += F.l1_loss(prediction, target, reduction="sum").item()
            running_mse += F.mse_loss(prediction, target, reduction="sum").item()

    return {
        "mae": _epoch_metric(running_mae, n_samples),
        "rmse": math.sqrt(_epoch_metric(running_mse, n_samples)),
    }


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    logger = create_logger()
    logger.info("Loaded config from %s", args.config)

    set_seed(cfg.get("seed", 42))

    loaders = build_dataloaders(cfg["dataset"], device=args.device)
    model = build_model(cfg["model"]).to(args.device)

    optim_cfg = cfg["optim"]
    routed = split_params_for_muon(
        model,
        apply_layers=optim_cfg.get("apply_layers", "all"),
        first_k_blocks=optim_cfg.get("first_k_blocks"),
    )
    muon_params = list(routed.muon)
    adamw_params = list(routed.adamw)

    optimizers: List[torch.optim.Optimizer] = []
    use_muon = optim_cfg.get("use_muon", False)

    if use_muon:
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
    else:
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

    device = torch.device(args.device)

    for epoch in range(1, epochs + 1):
        train_metrics = train_epoch(
            model=model,
            loader=train_loader,
            optimizers=optimizers,
            device=device,
            grad_clip=grad_clip,
            log_interval=log_interval,
            logger=logger,
        )
        logger.info(
            "epoch %d | train_loss=%.4f | train_mae=%.4f",
            epoch,
            train_metrics["loss"],
            train_metrics["mae"],
        )

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


if __name__ == "__main__":
    main()
