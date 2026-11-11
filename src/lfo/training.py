"""Training utilities shared by all projects.

``ComplexAdam`` is the Adam variant shipped with the original FNO and LNO code
(``Adam.py``). It differs from ``torch.optim.Adam`` on complex parameters:
the second moment is ``|g|^2`` per complex entry, whereas ``torch.optim.Adam``
views a complex tensor as (real, imag) pairs and keeps a separate second moment
for each. Weight decay is the coupled L2 form (added to the gradient). Use it
when reproducing the original LNO results; new models may use either.

``relative_l2_loss`` is the differentiable training loss used by the original
code (``LpLoss.rel``: per-sample relative L2, averaged over the batch). It is a
loss, not a reported metric: report errors with ``lfo.metrics.relative_l2``.
"""
from __future__ import annotations

import math
import random

import numpy as np
import torch
from torch.optim.optimizer import Optimizer

__all__ = ["ComplexAdam", "relative_l2_loss", "seed_everything", "count_params"]


class ComplexAdam(Optimizer):
    """Port of ``Adam.py`` from the original FNO/LNO repositories (amsgrad omitted)."""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        if lr < 0 or eps < 0 or weight_decay < 0 or not (0 <= betas[0] < 1 and 0 <= betas[1] < 1):
            raise ValueError("invalid Adam hyperparameters")
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                state["step"] += 1
                step = state["step"]
                grad = p.grad
                if group["weight_decay"] != 0:
                    grad = grad.add(p, alpha=group["weight_decay"])
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad.conj(), value=1 - beta2)
                bc1 = 1 - beta1 ** step
                bc2 = 1 - beta2 ** step
                denom = (exp_avg_sq.sqrt() / math.sqrt(bc2)).add_(group["eps"])
                p.addcdiv_(exp_avg, denom, value=-group["lr"] / bc1)
        return loss


def relative_l2_loss(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    """Mean over the batch of ||pred_i - true_i||_2 / ||true_i||_2 (original ``LpLoss.rel``)."""
    b = pred.shape[0]
    diff = torch.linalg.vector_norm(pred.reshape(b, -1) - true.reshape(b, -1), dim=1)
    ref = torch.linalg.vector_norm(true.reshape(b, -1), dim=1)
    return torch.mean(diff / ref)


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and PyTorch (CPU and CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_params(model: torch.nn.Module) -> int:
    """Number of real parameters; a complex entry counts as two."""
    return sum(p.numel() * (2 if p.is_complex() else 1) for p in model.parameters())
