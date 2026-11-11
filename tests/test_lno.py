"""Tests for the LNO port (lfo.models.lno) and lfo.training.

The equivalence tests load the class definitions from the original repository
(only the classes; the scripts themselves train at import time) and check that,
with identical weights, the port gives the same output. They need the original
code in ``runs/ref/LNO`` (or ``$LFO_LNO_REF``) and a CUDA device, because the
original hard-codes ``.cuda()``; otherwise they are skipped.
"""
import ast
import os
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from lfo.models import LNO1d, LNO2d, LNO2dV2
from lfo.training import ComplexAdam, count_params, relative_l2_loss

REF = Path(os.environ.get("LFO_LNO_REF", Path(__file__).resolve().parents[1] / "runs" / "ref" / "LNO"))
needs_ref = pytest.mark.skipif(
    not REF.is_dir() or not torch.cuda.is_available(),
    reason="original LNO code (runs/ref/LNO) or CUDA not available",
)


def _load_original(example: str, globals_: dict) -> dict:
    """Exec only the class definitions of <example>/main.py into a namespace."""
    tree = ast.parse((REF / example / "main.py").read_text())
    classes = [n for n in tree.body if isinstance(n, ast.ClassDef)]
    ns = {"torch": torch, "nn": nn, "F": F, "np": np, "time": __import__("time"), **globals_}
    exec(compile(ast.Module(body=classes, type_ignores=[]), str(REF / example / "main.py"), "exec"), ns)
    return ns


def _copy_weights(src: nn.Module, dst: nn.Module) -> None:
    missing, unexpected = dst.load_state_dict(src.state_dict(), strict=False)
    assert not unexpected
    assert all(k.endswith((".t", ".x")) for k in missing), missing  # only the coordinate buffers


# ---------------------------------------------------------------- CPU checks
def test_lno1d_shapes_and_gradients_on_cpu():
    t = torch.linspace(0, 1, 64)
    model = LNO1d(width=4, modes=8, t=t)
    x = torch.randn(3, 64, 1)
    y = model(x)
    assert y.shape == (3, 64, 1) and y.dtype == torch.float32
    relative_l2_loss(y, torch.randn_like(y)).backward()
    assert all(p.grad is not None for p in model.parameters())


def test_lno2d_and_v2_shapes_on_cpu():
    xg, tg = torch.linspace(0, 1, 16), torch.linspace(0, 2, 12)
    f = torch.randn(2, 16, 12, 1)
    assert LNO2d(8, 3, 3, xg, tg)(f).shape == (2, 16, 12, 1)
    assert LNO2dV2(8, 8, 8, 2, 2, 2, 2, xg, tg)(f).shape == (2, 16, 12, 1)


def test_count_params_counts_complex_twice():
    model = LNO1d(width=4, modes=16, t=torch.linspace(0, 1, 8))
    pr = 2 * (4 * 4 * 16) * 2  # pole + residue, complex
    dense = (1 * 4 + 4) + (4 * 4 + 4) + (4 * 128 + 128) + (128 + 1)
    assert count_params(model) == pr + dense


def test_relative_l2_loss_matches_metric():
    from lfo.metrics import relative_l2

    a, b = torch.randn(5, 10, 3), torch.randn(5, 10, 3)
    per_sample = relative_l2(a, b, axis=(1, 2))
    assert relative_l2_loss(a, b).item() == pytest.approx(float(np.mean(per_sample)), rel=1e-5)


# ------------------------------------------------- equivalence with original
@needs_ref
@pytest.mark.parametrize("example,use_grid,activation", [
    ("1D_Duffing_c0", False, "sin"),
    ("1D_Duffing_c05", False, "sin"),
    ("1D_Lorenz_rho5", True, "tanh"),
    ("1D_Pendulum_c0", True, "sin"),
])
def test_lno1d_matches_original(example, use_grid, activation):
    torch.manual_seed(0)
    t = torch.linspace(0, 20, 256).reshape(-1, 1)  # original grid shape (n, 1)
    ns = _load_original(example, {"grid_x_train": t})
    ref = ns["LNO1d"](4, 16).cuda()
    port = LNO1d(4, 16, t, use_grid=use_grid, activation=activation).cuda()
    _copy_weights(ref, port)
    x = torch.randn(5, 256, 1, device="cuda")
    with torch.no_grad():
        torch.testing.assert_close(port(x), ref(x), rtol=0, atol=0)


@needs_ref
@pytest.mark.parametrize("example", ["2D_Diffusion", "2D_Reac_diffusion", "2D_Beam"])
def test_lno2d_matches_original(example):
    torch.manual_seed(0)
    X = torch.linspace(0, 1, 20).reshape(1, -1)
    T = torch.linspace(0, 2, 14).reshape(1, -1)
    ns = _load_original(example, {"T": T, "X": X})
    ref = ns["LNO2d"](8, 4, 4).cuda()
    port = LNO2d(8, 4, 4, X, T).cuda()
    _copy_weights(ref, port)
    f = torch.randn(3, 20, 14, 1, device="cuda")
    with torch.no_grad():
        torch.testing.assert_close(port(f), ref(f), rtol=0, atol=0)


@needs_ref
def test_lno2d_v2_matches_original_burgers():
    torch.manual_seed(0)
    X = torch.linspace(0, 1, 16).reshape(1, -1)
    T = torch.linspace(0, 1, 12).reshape(1, -1)
    ns = _load_original("2D_Burger", {"T": T, "X": X})
    ref = ns["LNO2d"](8, 8, 8, 2, 2, 2, 2).cuda()
    port = LNO2dV2(8, 8, 8, 2, 2, 2, 2, X, T).cuda()
    _copy_weights(ref, port)
    f = torch.randn(2, 16, 12, 1, device="cuda")
    with torch.no_grad():
        torch.testing.assert_close(port(f), ref(f), rtol=0, atol=0)


@needs_ref
def test_complex_adam_matches_original():
    ns = {}
    exec(compile((REF / "1D_Duffing_c0" / "Adam.py").read_text(), "Adam.py", "exec"), ns)
    torch.manual_seed(0)
    p0 = [torch.randn(4, 3, dtype=torch.cfloat), torch.randn(5)]
    pa = [nn.Parameter(p.clone()) for p in p0]
    pb = [nn.Parameter(p.clone()) for p in p0]
    opt_a = ns["Adam"](pa, lr=0.01, weight_decay=1e-4)
    opt_b = ComplexAdam(pb, lr=0.01, weight_decay=1e-4)
    for k in range(5):
        torch.manual_seed(k)
        grads = [torch.randn_like(p) for p in p0]
        for p, g in zip(pa, grads):
            p.grad = g.clone()
        for p, g in zip(pb, grads):
            p.grad = g.clone()
        opt_a.step()
        opt_b.step()
    for a, b in zip(pa, pb):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
