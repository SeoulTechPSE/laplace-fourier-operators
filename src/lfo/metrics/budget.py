"""Parameter budgets in real degrees of freedom.

`sum(p.numel())` counts a complex weight once, so a Fourier-type model with complex spectral weights looks about
half its size next to a real-valued model. (C) E1 matched budgets this way from the start
(`lfo.training.count_params`, which now calls this function); (A) deep and (D) used `numel`. Budgets across models are matched on `n_real_params`: a complex
parameter counts as two (decision 2026-11-22 in `STATUS.md`)."""
from __future__ import annotations

import torch.nn as nn


def n_real_params(model: nn.Module, trainable_only: bool = False) -> int:
    """Number of real degrees of freedom: complex parameters count twice."""
    return sum(p.numel() * (2 if p.is_complex() else 1) for p in model.parameters()
               if p.requires_grad or not trainable_only)
