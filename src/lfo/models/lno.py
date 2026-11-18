"""Laplace Neural Operator (LNO), ported from the original code.

Source: github.com/qianyingcao/Laplace-Neural-Operator, commit 78c64ef
(Cao, Goswami, Karniadakis, arXiv:2303.10528).

The port keeps the original arithmetic exactly (same einsum contractions, same
normalisation, same parameter shapes and initialisation) so that, for equal
weights, outputs agree with the original to float32 round-off. That equivalence
is checked in ``tests/test_lno.py``. The only changes are:

- coordinate grids are passed to the constructor and stored as buffers instead
  of being read from module-level globals (``grid_x_train``, ``T``, ``X``);
- no hard-coded ``.cuda()``: modules follow ``.to(device)``;
- per-example switches of the original scripts are constructor arguments.

Two quirks of the original are kept on purpose, because reproduction is the goal:

1. In the transient term the output channel of the residues is contracted again
   with the *input* channel index of the poles (``"bix,ioxz->boz"`` in 1D,
   ``"kbpq,bipqzx->kizx"`` in 2D). This is a second channel mixing, not the
   textbook pole-residue form; it only type-checks because in == out channels.
2. The sign of the transient residues differs by example: ``-Hw`` in the 1D ODE
   examples and ``+Hw`` in the 2D PDE examples (see the comment in the original
   2D code). Use ``transient_sign``.

Versions:
  v1  ``LNO1d``, ``LNO2d``: transient and steady terms from the same poles/residues.
  v2  ``LNO2dV2`` (Burgers example): pole-residue transient branch plus an
      FNO-style steady branch with free Fourier weights.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn

__all__ = [
    "PoleResidue1d",
    "PoleResidue2d",
    "LNO1d",
    "LNO2d",
    "Transient2d",
    "Steady2d",
    "LNO2dV2",
]


def _as_grid(x) -> torch.Tensor:
    """Coordinates as a 1D float32 tensor (accepts (n,), (n,1), (1,n))."""
    return torch.as_tensor(x, dtype=torch.float32).reshape(-1).clone()


def _unit_grid(shape, device) -> torch.Tensor:
    """Normalised coordinates in [0, 1] appended as input features (original get_grid)."""
    b = shape[0]
    axes = []
    for d, n in enumerate(shape[1:-1]):
        g = torch.tensor(np.linspace(0, 1, n), dtype=torch.float32)  # as original (not torch.linspace)
        view = [1] * (len(shape) - 1)
        view[d + 1] = n
        reps = [b] + list(shape[1:-1])
        reps[d + 1] = 1
        axes.append(g.reshape(*view, 1).repeat(*reps, 1))
    return torch.cat(axes, dim=-1).to(device)


# ----------------------------------------------------------------------------
# v1, 1D
# ----------------------------------------------------------------------------
class PoleResidue1d(nn.Module):
    """1D pole-residue layer (``PR`` in the original 1D examples).

    Input and output have shape (batch, channels, n_t); ``t`` is the time grid.
    Learned poles and residues: ``weights_pole``, ``weights_residue``, both
    complex with shape (in, out, modes).
    """

    def __init__(self, in_channels: int, out_channels: int, modes: int, t, transient_sign: float = -1.0):
        super().__init__()
        self.modes1 = modes
        self.transient_sign = transient_sign
        self.scale = 1 / (in_channels * out_channels)
        self.weights_pole = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, modes, dtype=torch.cfloat))
        self.weights_residue = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, modes, dtype=torch.cfloat))
        self.register_buffer("t", _as_grid(t))

    def output_PR(self, lambda1, alpha):
        Hw = self.weights_residue * (1 / (lambda1 - self.weights_pole))
        out_steady = torch.einsum("bix,xiok->box", alpha, Hw)
        out_transient = torch.einsum("bix,xiok->bok", alpha, self.transient_sign * Hw)
        return out_steady, out_transient

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = self.t
        dt = (t[1] - t[0]).item()
        alpha = torch.fft.fft(x)
        lambda1 = (torch.fft.fftfreq(t.shape[0], dt) * 2 * math.pi * 1j).to(x.device)
        lambda1 = lambda1[:, None, None, None]
        out_steady, out_transient = self.output_PR(lambda1, alpha)

        x1 = torch.real(torch.fft.ifft(out_steady, n=x.size(-1)))
        term = torch.exp(torch.einsum("bix,kz->bixz", self.weights_pole, t.type(torch.complex64).reshape(1, -1)))
        x2 = torch.real(torch.einsum("bix,ioxz->boz", out_transient, term)) / x.size(-1)
        return x1 + x2


class LNO1d(nn.Module):
    """LNO v1 for 1D ODE examples (Duffing, Lorenz, Pendulum).

    Input (batch, n_t, 1) -> output (batch, n_t, 1).

    Per-example switches in the original scripts:
      Duffing c=0:   use_grid=False, activation="sin"
      Duffing c=0.5: use_grid=False, activation="sin"
      Lorenz:        use_grid=True,  activation="tanh"
      Pendulum:      use_grid=True,  activation="sin"
    All 1D examples use transient_sign=-1.

    ``textbook=True`` replaces the pole-residue layer by ``lfo.models.hybrid.PoleResidueTS1d`` (same shapes and
    initialisation distributions; poles in ``conv0.mu``, residues in ``conv0.beta``). The original layer multiplies
    each transient coefficient by another channel's pole (quirk 1 in the module docstring), so the periodic steady
    term's wrap-around is not cancelled and the layer is non-causal; the textbook wiring is causal up to
    discretisation ((C) E5, eleventh pass). The default keeps the original for reproduction.
    """

    def __init__(self, width: int, modes: int, t, use_grid: bool = False,
                 activation: str = "sin", transient_sign: float = -1.0, textbook: bool = False):
        super().__init__()
        self.width = width
        self.modes1 = modes
        self.use_grid = use_grid
        self.act = {"sin": torch.sin, "tanh": torch.tanh}[activation]
        self.fc0 = nn.Linear(2 if use_grid else 1, width)
        if textbook:
            if transient_sign != -1.0:
                raise ValueError("the textbook wiring uses transient_sign = -1 (all 1D examples)")
            from .hybrid import PoleResidueTS1d       # local import: hybrid imports this module
            self.conv0 = PoleResidueTS1d(width, width, modes, t)
        else:
            self.conv0 = PoleResidue1d(width, width, modes, t, transient_sign)
        self.w0 = nn.Conv1d(width, width, 1)
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_grid:
            x = torch.cat((x, _unit_grid(x.shape, x.device)), dim=-1)
        x = self.fc0(x).permute(0, 2, 1)
        x = self.conv0(x) + self.w0(x)
        x = x.permute(0, 2, 1)
        x = self.act(self.fc1(x))
        return self.fc2(x)


# ----------------------------------------------------------------------------
# v1, 2D
# ----------------------------------------------------------------------------
def _freqs(grid: torch.Tensor, device, rfft: bool = False) -> torch.Tensor:
    d = (grid[1] - grid[0]).item()
    f = torch.fft.rfftfreq if rfft else torch.fft.fftfreq
    return (f(grid.shape[0], d) * 2 * math.pi * 1j).to(device)


class PoleResidue2d(nn.Module):
    """2D pole-residue layer (``PR2d`` in the original 2D examples).

    Input (batch, channels, n_x, n_t). ``x`` is the first (spatial) grid and
    ``t`` the second (time) grid. Poles are separable: ``weights_pole1`` on the
    first axis, ``weights_pole2`` on the second; residues have shape
    (in, out, modes1, modes2).
    """

    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int, x, t,
                 transient_sign: float = 1.0):
        super().__init__()
        self.modes1, self.modes2 = modes1, modes2
        self.transient_sign = transient_sign
        self.scale = 1 / (in_channels * out_channels)
        self.weights_pole1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, modes1, dtype=torch.cfloat))
        self.weights_pole2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, modes2, dtype=torch.cfloat))
        self.weights_residue = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))
        self.register_buffer("x", _as_grid(x))
        self.register_buffer("t", _as_grid(t))

    def output_PR(self, lambda1, lambda2, alpha):
        term1 = 1 / torch.einsum("pbix,qbik->pqbixk", lambda1 - self.weights_pole1, lambda2 - self.weights_pole2)
        Hw = torch.einsum("bixk,pqbixk->pqbixk", self.weights_residue, term1)
        out_steady = torch.einsum("biox,oxikpq->bkox", alpha, Hw)
        out_transient = torch.einsum("biox,oxikpq->bkpq", alpha, self.transient_sign * Hw)
        return out_steady, out_transient

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        alpha = torch.fft.fft2(u, dim=[-2, -1])
        lambda1 = _freqs(self.x, u.device)[:, None, None, None]
        lambda2 = _freqs(self.t, u.device)[:, None, None, None]
        out_steady, out_transient = self.output_PR(lambda1, lambda2, alpha)

        x1 = torch.real(torch.fft.ifft2(out_steady, s=(u.size(-2), u.size(-1))))
        term1 = torch.einsum("bip,kz->bipz", self.weights_pole1, self.x.type(torch.complex64).reshape(1, -1))
        term2 = torch.einsum("biq,kx->biqx", self.weights_pole2, self.t.type(torch.complex64).reshape(1, -1))
        term3 = torch.einsum("bipz,biqx->bipqzx", torch.exp(term1), torch.exp(term2))
        x2 = torch.real(torch.einsum("kbpq,bipqzx->kizx", out_transient, term3)) / u.size(-1) / u.size(-2)
        return x1 + x2


class LNO2d(nn.Module):
    """LNO v1 for 2D examples (Diffusion, Reaction-diffusion, Beam).

    Input (batch, n_x, n_t, 1) -> output (batch, n_x, n_t, 1). The original
    appends normalised (x, t) coordinates and applies InstanceNorm around the
    pole-residue layer.
    """

    def __init__(self, width: int, modes1: int, modes2: int, x, t, transient_sign: float = 1.0):
        super().__init__()
        self.width = width
        self.modes1, self.modes2 = modes1, modes2
        self.fc0 = nn.Linear(3, width)
        self.conv0 = PoleResidue2d(width, width, modes1, modes2, x, t, transient_sign)
        self.w0 = nn.Conv2d(width, width, 1)
        self.norm = nn.InstanceNorm2d(width)
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        f = torch.cat((f, _unit_grid(f.shape, f.device)), dim=-1)
        x = self.fc0(f).permute(0, 3, 1, 2)
        x = self.norm(self.conv0(self.norm(x))) + self.w0(x)
        x = x.permute(0, 2, 3, 1)
        x = torch.sin(self.fc1(x))
        return self.fc2(x)


# ----------------------------------------------------------------------------
# v2, 2D (Burgers example)
# ----------------------------------------------------------------------------
class Transient2d(nn.Module):
    """v2 transient branch: pole-residue terms only (``Transient`` in the original).

    Residues are computed on the input subsampled by ``stride`` along both axes
    and evaluated on the full output grid.
    """

    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int, x, t, stride: int = 2):
        super().__init__()
        self.modes11, self.modes12 = modes1, modes2
        self.stride = stride
        self.scale = 1 / (in_channels * out_channels)
        self.weights_pole1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, modes1, dtype=torch.cfloat))
        self.weights_pole2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, modes2, dtype=torch.cfloat))
        self.weights_residue = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))
        self.register_buffer("x", _as_grid(x))
        self.register_buffer("t", _as_grid(t))

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        s = self.stride
        u = u[:, :, ::s, ::s]
        alpha = torch.fft.fft2(u, dim=[-2, -1])
        lambda1 = _freqs(self.x[::s], u.device)[:, None, None, None]
        lambda2 = _freqs(self.t[::s], u.device)[:, None, None, None]
        term1 = 1 / torch.einsum("pbix,qbik->pqbixk", self.weights_pole1 - lambda1, self.weights_pole2 - lambda2)
        Pk = torch.einsum("bixk,pqbixk->pqbixk", self.weights_residue, term1)
        res = torch.einsum("biox,oxikpq->bkpq", alpha, Pk)

        term1 = torch.einsum("bip,kz->bipz", self.weights_pole1, self.x.type(torch.complex64).reshape(1, -1))
        term2 = torch.einsum("biq,kx->biqx", self.weights_pole2, self.t.type(torch.complex64).reshape(1, -1))
        term3 = torch.einsum("bipz,biqx->bipqzx", torch.exp(term1), torch.exp(term2))
        return torch.real(torch.einsum("kbpq,bipqzx->kizx", res, term3)) / u.size(-1) / u.size(-2)


class Steady2d(nn.Module):
    """v2 steady branch: FNO-style free Fourier weights (``Steady`` in the original).

    Like ``Transient2d`` it reads the input subsampled by ``stride`` and
    evaluates the inverse transform explicitly on the full output grid.
    """

    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int, x, t, stride: int = 2):
        super().__init__()
        self.out_channels = out_channels
        self.modes21, self.modes22 = modes1, modes2
        self.stride = stride
        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))
        self.register_buffer("x", _as_grid(x))
        self.register_buffer("t", _as_grid(t))

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        s = self.stride
        u = u[:, :, ::s, ::s]
        FW1 = torch.fft.rfft2(u)
        lambda1 = _freqs(self.x[::s], u.device)
        lambda2 = _freqs(self.t[::s], u.device, rfft=True)
        m1, m2 = self.modes21, self.modes22
        out = torch.zeros(u.shape[0], self.out_channels, u.size(-2), u.size(-1) // 2 + 1,
                          dtype=torch.cfloat, device=u.device)
        out[:, :, :m1, :m2] = torch.einsum("bixy,ioxy->boxy", FW1[:, :, :m1, :m2], self.weights1)
        out[:, :, -m1:, :m2] = torch.einsum("bixy,ioxy->boxy", FW1[:, :, -m1:, :m2], self.weights2)

        term4 = torch.einsum("xk,kz->xz", lambda1.reshape(-1, 1), self.x.type(torch.complex64).reshape(1, -1))
        term5 = torch.einsum("bk,ki->bi", lambda2.reshape(-1, 1), self.t.type(torch.complex64).reshape(1, -1))
        term6 = torch.einsum("xz,bi->xbzi", torch.exp(term4), torch.exp(term5))
        return torch.real(torch.einsum("pqxb,xbzi->pqzi", out, term6)) / u.size(-1) / u.size(-2)


class LNO2dV2(nn.Module):
    """LNO v2 (original Burgers example): transient branch + steady branch, summed.

    Transient branch: 4 ``Transient2d`` layers with sin activations.
    Steady branch: 4 ``Steady2d`` layers plus 1x1 convolutions on layers 1-3
    (``w1``..``w3``; ``w0`` exists in the original but is unused).
    """

    def __init__(self, width1: int, width2: int, width3: int, modes11: int, modes12: int,
                 modes21: int, modes22: int, x, t, stride: int = 2):
        super().__init__()
        self.fc2 = nn.Linear(3, width2)
        self.conv_s0 = Steady2d(width2, width2, modes21, modes22, x, t, stride)
        self.conv_s1 = Steady2d(width2, width2, modes21, modes22, x, t, stride)
        self.conv_s2 = Steady2d(width2, width2, modes21, modes22, x, t, stride)
        self.conv_s3 = Steady2d(width2, width2, modes21, modes22, x, t, stride)
        self.w0 = nn.Conv2d(width3, width3, 1)
        self.w1 = nn.Conv2d(width3, width3, 1)
        self.w2 = nn.Conv2d(width3, width3, 1)
        self.w3 = nn.Conv2d(width3, width3, 1)
        self.norm2 = nn.InstanceNorm2d(width2)
        self.fc5 = nn.Linear(width2, 128)
        self.fc8 = nn.Linear(128, 1)
        self.fc1 = nn.Linear(3, width1)
        self.conv_t0 = Transient2d(width1, width1, modes11, modes12, x, t, stride)
        self.conv_t1 = Transient2d(width1, width1, modes11, modes12, x, t, stride)
        self.conv_t2 = Transient2d(width1, width1, modes11, modes12, x, t, stride)
        self.conv_t3 = Transient2d(width1, width1, modes11, modes12, x, t, stride)
        self.norm1 = nn.InstanceNorm2d(width1)
        self.fc4 = nn.Linear(width1, 128)
        self.fc7 = nn.Linear(128, 1)

    def transient(self, f: torch.Tensor) -> torch.Tensor:
        x = self.fc1(f).permute(0, 3, 1, 2)
        for i, conv in enumerate((self.conv_t0, self.conv_t1, self.conv_t2, self.conv_t3)):
            x = self.norm1(conv(self.norm1(x)))
            if i < 3:
                x = torch.sin(x)
        x = torch.sin(self.fc4(x.permute(0, 2, 3, 1)))
        return self.fc7(x)

    def steady(self, f: torch.Tensor) -> torch.Tensor:
        x = self.fc2(f).permute(0, 3, 1, 2)
        x = torch.sin(self.norm2(self.conv_s0(self.norm2(x))))
        for i, (conv, w) in enumerate(((self.conv_s1, self.w1), (self.conv_s2, self.w2), (self.conv_s3, self.w3))):
            x = self.norm2(conv(self.norm2(x))) + w(x)
            if i < 2:
                x = torch.sin(x)
        x = torch.sin(self.fc5(x.permute(0, 2, 3, 1)))
        return self.fc8(x)

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        f = torch.cat((f, _unit_grid(f.shape, f.device)), dim=-1)
        return self.transient(f) + self.steady(f)
