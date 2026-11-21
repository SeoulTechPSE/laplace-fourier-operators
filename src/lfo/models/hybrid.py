"""Dual-branch Laplace–Fourier operators re-implemented from the papers (1D, time axis).

No public code exists for either model (checked 2026-11-12: papers, author
GitHub accounts, lab pages, GitHub search), so both are re-implementations.

ALNO  Kim, Cao, Shin, Lee, Karniadakis, Choi, "Physics-Informed Laplace Neural
      Operator", arXiv:2602.12706, Eq. (15). Each layer is
          v <- sigma( T[v] + F^-1[ H(omega) F[v] ] + W v ),
      a pole-residue transient term T plus a free FNO multiplier H on the lowest
      modes (the steady term is decoupled from the poles). This is the
      "advanced LNO" of the LNO Supplementary Information, written per layer.
      The original Burgers code (``LNO2dV2``) instead runs a transient stack and
      a steady stack in parallel and adds them at the end.

LFNO  Ha, Yoon, Lee, "LFNO: Bridging Laplace and Fourier via Transient-Steady
      Decomposition", arXiv:2606.07601, Eqs. (2)-(7). The steady coefficients
      come from the *same* poles, lambda_l = alpha_l sum_n beta_n/(i omega_l - mu_n),
      and are then mixed by a learned R_l on the lowest modes (Eq. 5-6):
          v <- sigma( T[v] + F^-1[ R_l lambda_l ] + W v ).
      The paper does not specify how its "6 layers" (more transient than steady)
      are arranged. We use ``n_layers`` pole-residue layers of which the first
      ``n_steady`` also carry the R-mixed steady term. With the paper's ODE
      setting (width 4, 16 modes, n_layers=4, n_steady=2) this reproduces its
      reported parameter count of 3,417 exactly (complex entries counted once, as
      in the paper; see tests/test_hybrid.py). This arrangement is our
      reconstruction, not confirmed by the authors.

Unlike the LNO port (``lfo.models.lno``), the pole-residue term here uses the
textbook contraction (no second channel mixing): poles and residues are indexed
by (in, out, n) and

    y_o(t) = Re sum_{i,n} gamma_{ion} exp(mu_{ion} t),
    gamma_{ion} = beta_{ion} / N * sum_l alpha_{il} / (mu_{ion} - i omega_l),

which is the residue of K(s) V(s) at s = mu (sign as in the original 1D code).
``poles()`` returns the learned (mu, beta) for the pole audit.

Initialization (not given in either paper). With the original LNO pole
initialization (``pole_init="lno"``: Re, Im uniform in [0, 1/(in*out)), i.e.
slightly unstable poles near 0) a 4-layer stack does not train: the transient
term grows layer by layer (relu: diverges; sin: saturates). Two remedies work
(Duffing c=0, iid, 300 epochs, 1 seed; STATUS.md 2026-11-12):

- ``pole_init="stable"`` (default): Re mu = -exp(U[log 1/T, log 10/T]),
  Im mu ~ U[0, pi/(4 dt)];
- ``norm=True``: InstanceNorm around each pole-residue layer, as in the
  original v2 code (``LNO2dV2``).

The default is the stable initialization without normalization, because a
normalization between layers makes the learned poles harder to interpret.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from .lno import _as_grid, _unit_grid

__all__ = ["PoleResidueTS1d", "SpectralConv1d", "ALNO1d", "LFNO1d"]

_ACT = {"sin": torch.sin, "tanh": torch.tanh, "relu": torch.relu, "gelu": nn.functional.gelu}


def _omega(t: torch.Tensor, device) -> torch.Tensor:
    """Angular DFT frequencies i*omega_l for the grid ``t`` (fft order)."""
    dt = (t[1] - t[0]).item()
    f = torch.fft.fftfreq(t.numel(), dt)
    return (f * 2 * math.pi * 1j).to(device)


class PoleResidueTS1d(nn.Module):
    """Pole-residue layer with separate transient and steady outputs (1D, time axis).

    ``transient(x)``      time signal sum_n gamma_n exp(mu_n t), shape (b, out, n_t).
    ``steady_coeffs(x)``  rfft-ordered steady coefficients lambda_l = alpha_l K(i omega_l)
                          for l < ``modes_steady``, shape (b, out, modes_steady).
    ``forward(x)``        transient + full steady term (the LNO v1 layer, textbook form).
    """

    def __init__(self, in_channels: int, out_channels: int, n_poles: int, t, pole_init: str = "lno"):
        super().__init__()
        scale = 1 / (in_channels * out_channels)
        shape = (in_channels, out_channels, n_poles)
        if pole_init == "lno":        # original LNO: real and imaginary parts uniform in [0, scale)
            mu = scale * torch.rand(*shape, dtype=torch.cfloat)
        elif pole_init == "stable":   # decay rates 1/T..10/T, frequencies up to the Nyquist/4
            tt = _as_grid(t)
            T, dt = (tt[-1] - tt[0]).item(), (tt[1] - tt[0]).item()
            re = -torch.exp(torch.empty(shape).uniform_(math.log(1 / T), math.log(10 / T)))
            im = torch.empty(shape).uniform_(0, math.pi / (4 * dt))
            mu = torch.complex(re, im)
        else:
            raise ValueError("pole_init must be 'lno' or 'stable'")
        self.mu = nn.Parameter(mu)
        self.beta = nn.Parameter(scale * torch.rand(*shape, dtype=torch.cfloat))
        self.register_buffer("t", _as_grid(t))

    def poles(self):
        """Learned poles and residues, each (in, out, n_poles), detached."""
        return self.mu.detach(), self.beta.detach()

    def _H(self, iw: torch.Tensor) -> torch.Tensor:
        """beta / (i omega - mu), shape (L, in, out, n)."""
        return self.beta / (iw[:, None, None, None] - self.mu)

    def transient(self, x: torch.Tensor) -> torch.Tensor:
        n_t = x.size(-1)
        alpha = torch.fft.fft(x)                                       # (b, i, L)
        H = self._H(_omega(self.t, x.device))                          # (L, i, o, n)
        gamma = -torch.einsum("bil,lion->bion", alpha, H) / n_t
        e = torch.exp(torch.einsum("ion,z->ionz", self.mu, self.t.to(torch.complex64)))
        return torch.real(torch.einsum("bion,ionz->boz", gamma, e))

    def steady_coeffs(self, x: torch.Tensor, modes: int | None = None) -> torch.Tensor:
        """Steady term Re sum_l lambda_l e^{i omega_l t} as rfft coefficients.

        The poles need not be conjugate-symmetric, so lambda_{-l} is not
        conj(lambda_l). Folding c_l = (lambda_l + conj(lambda_{-l})) / 2 gives
        irfft(c) == Re(ifft(lambda)) exactly (the original LNO steady term).
        """
        n_t = x.size(-1)
        alpha = torch.fft.fft(x)                                       # (b, i, L)
        H = self._H(_omega(self.t, x.device)).sum(-1)                  # (L, i, o)
        lam = torch.einsum("bil,lio->bol", alpha, H)                   # (b, o, L)
        pos = torch.arange(n_t // 2 + 1, device=x.device)
        c = (lam[..., pos] + lam[..., (-pos) % n_t].conj()) / 2
        return c if modes is None else c[..., :modes]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        steady = torch.fft.irfft(self.steady_coeffs(x), n=x.size(-1))
        return self.transient(x) + steady


class SpectralConv1d(nn.Module):
    """FNO spectral convolution: free complex multiplier on the lowest ``modes`` rfft modes."""

    def __init__(self, in_channels: int, out_channels: int, modes: int):
        super().__init__()
        self.modes = modes
        scale = 1 / (in_channels * out_channels)
        self.weight = nn.Parameter(scale * torch.rand(in_channels, out_channels, modes, dtype=torch.cfloat))

    def mix(self, coeffs: torch.Tensor) -> torch.Tensor:
        """Apply the multiplier to rfft coefficients (b, in, >=modes) -> (b, out, modes)."""
        return torch.einsum("bil,iol->bol", coeffs[..., :self.modes], self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.fft.irfft(self.mix(torch.fft.rfft(x)), n=x.size(-1))


class _DualBranch1d(nn.Module):
    """Shared lifting / projection; subclasses define ``layer(k, v)``."""

    def __init__(self, width: int, n_layers: int, t, use_grid: bool, activation: str,
                 norm: bool = False, proj_width: int = 128):
        super().__init__()
        self.norm = nn.InstanceNorm1d(width) if norm else nn.Identity()
        self.use_grid = use_grid
        self.act = _ACT[activation]
        self.n_layers = n_layers
        self.fc0 = nn.Linear(2 if use_grid else 1, width)
        self.ws = nn.ModuleList(nn.Conv1d(width, width, 1) for _ in range(n_layers))
        self.fc1 = nn.Linear(width, proj_width)
        self.fc2 = nn.Linear(proj_width, 1)
        self.register_buffer("t", _as_grid(t))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(batch, n_t, 1) -> (batch, n_t, 1)."""
        if self.use_grid:
            x = torch.cat((x, _unit_grid(x.shape, x.device)), dim=-1)
        v = self.fc0(x).permute(0, 2, 1)
        for k in range(self.n_layers):
            v = self.norm(self.layer(k, self.norm(v))) + self.ws[k](v)   # norm as in LNO2dV2
            if k < self.n_layers - 1:
                v = self.act(v)
        v = self.act(self.fc1(v.permute(0, 2, 1)))
        return self.fc2(v)

    def poles(self):
        """List over layers of (mu, beta), each (in, out, n_poles)."""
        return [p.poles() for p in self.pr]


class ALNO1d(_DualBranch1d):
    """ALNO (PILNO paper, Eq. 15): pole-residue transient + free Fourier steady per layer."""

    def __init__(self, width: int, n_poles: int, modes: int, t, n_layers: int = 4,
                 use_grid: bool = False, activation: str = "sin", norm: bool = False, pole_init: str = "stable"):
        super().__init__(width, n_layers, t, use_grid, activation, norm)
        self.pr = nn.ModuleList(PoleResidueTS1d(width, width, n_poles, t, pole_init) for _ in range(n_layers))
        self.spec = nn.ModuleList(SpectralConv1d(width, width, modes) for _ in range(n_layers))

    def layer(self, k, v):
        return self.pr[k].transient(v) + self.spec[k](v)


class LFNO1d(_DualBranch1d):
    """LFNO (Ha et al., Eq. 7): transient from the poles; steady = R_l * (pole-residue
    steady coefficients) on the lowest modes, in the first ``n_steady`` layers."""

    def __init__(self, width: int, modes: int, t, n_layers: int = 4, n_steady: int = 2,
                 use_grid: bool = False, activation: str = "relu", norm: bool = False, pole_init: str = "stable"):
        super().__init__(width, n_layers, t, use_grid, activation, norm)
        if not 0 <= n_steady <= n_layers:
            raise ValueError("need 0 <= n_steady <= n_layers")
        self.modes = modes
        self.pr = nn.ModuleList(PoleResidueTS1d(width, width, modes, t, pole_init) for _ in range(n_layers))
        self.R = nn.ModuleList(SpectralConv1d(width, width, modes) for _ in range(n_steady))

    def layer(self, k, v):
        out = self.pr[k].transient(v)
        if k < len(self.R):
            lam = self.pr[k].steady_coeffs(v, self.modes)              # (b, width, modes)
            out = out + torch.fft.irfft(self.R[k].mix(lam), n=v.size(-1))
        return out
