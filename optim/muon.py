from __future__ import annotations

from typing import Iterable, Optional

import torch
from torch.optim import Optimizer


class Muon(Optimizer):
    """Muon optimizer (Im et al., 2024) with orthogonalized momentum updates."""

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 2.0e-3,
        weight_decay: float = 0.0,
        eps: float = 1.0e-8,
        momentum: float = 0.9,
        orthogonalize: bool = True,
    ) -> None:
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            eps=eps,
            momentum=momentum,
            orthogonalize=orthogonalize,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[callable] = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]
            orthogonalize = group["orthogonalize"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError("Muon does not support sparse gradients")

                grad = p.grad
                if weight_decay != 0.0:
                    grad = grad.add(p, alpha=weight_decay)

                state = self.state[p]
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(p)

                buf: torch.Tensor = state["momentum_buffer"]
                buf.mul_(momentum).add_(grad)

                update = buf
                if orthogonalize:
                    if p.dim() == 1:
                        denom = torch.dot(p, p).add_(eps)
                        proj = torch.dot(update, p) / denom
                        update = update - proj * p
                    elif p.dim() >= 2:
                        w = p.view(p.shape[0], -1)
                        u = update.view_as(w)
                        denom = (w * w).sum(dim=1, keepdim=True).add_(eps)
                        proj = (u * w).sum(dim=1, keepdim=True) / denom
                        update = (u - proj * w).view_as(p)

                p.add_(update, alpha=-lr)

        return loss
