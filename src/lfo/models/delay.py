"""Pole-residue-delay layer (project A, Application 1: process dynamics with time delay; plan Section 5.1).

    K(s) = sum_n beta_n e^{-theta_n s} / (s - mu_n),       theta_n >= 0.

**Exact delayed first-order hold.** The input is linear between samples (the D1 time axis). Without delay, the
mode state x_j = x(t_j) follows the exact FOH recursion (`wavenumber.foh_response`). With a delay theta, the
output at t_j is the undelayed state at tau = t_j - theta, which generally falls between samples. Write
tau = t_m + r with 0 <= r < dt. Then the state follows from x_m by a partial FOH step of length r with the same
input ramp:

    x(tau) = e^{mu r} x_m + I0(r) u_m + I1(r) (u_{m+1} - u_m) / dt,
    I0(r) = (e^{mu r} - 1) / mu,    I1(r) = (e^{mu r} - 1 - mu r) / mu^2,

and x(tau) = 0 for tau < 0 (rest). This is exact for piecewise-linear inputs, causal, and continuous and
differentiable in theta (m jumps where r wraps, and the formula is continuous there).

`PoleResidueDelay` is a SISO operator on (batch, n_t) signals. Real outputs: real poles with real residues, or
complex poles whose output is 2 Re(.) (conjugate pairs implied). Delays are softplus-parameterised; `shared_delay`
uses one theta for all terms (the plan's first stage).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .wavenumber import foh_kernel, foh_response

__all__ = ["partial_step", "delayed_foh", "PoleResidueDelay"]


def _series(z, n_terms=6):
    """(e^z - 1) / z and (e^z - 1 - z) / z^2 by their Taylor series, for small |z|."""
    a = sum(z ** j / torch.exp(torch.lgamma(torch.tensor(j + 2.0, dtype=torch.float64))) for j in range(n_terms))
    b = sum(z ** j / torch.exp(torch.lgamma(torch.tensor(j + 3.0, dtype=torch.float64))) for j in range(n_terms))
    return a, b


def partial_step(mu: torch.Tensor, r: torch.Tensor):
    """(e^{mu r}, I0(r), I1(r)) for complex mu and real r >= 0 (broadcast)."""
    mu = mu.to(torch.complex128)
    z = mu * r
    small = z.abs() < 1e-3
    zs = torch.where(small, torch.ones_like(z), z)
    a_s, b_s = _series(z)
    E = torch.exp(z)
    I0 = torch.where(small, r * a_s, (E - 1) / zs * r)
    I1 = torch.where(small, r ** 2 * b_s, (E - 1 - zs) / zs ** 2 * r ** 2)
    return E, I0, I1


def delayed_foh(mu: torch.Tensor, theta: torch.Tensor, u: torch.Tensor, dt: float) -> torch.Tensor:
    """Response of 1 / (s - mu) e^{-theta s} to u (linear between samples), from rest.

    mu, theta: (P,) (theta real >= 0); u: (B, n_t) real or complex. Returns (B, P, n_t) complex."""
    n_t = u.shape[-1]
    mu = mu.to(torch.complex128)
    uc = u.to(torch.complex128)
    g, c = foh_kernel(mu, n_t, dt)                                       # (P, n_t)
    x = foh_response(g[None], c[None], uc[:, None])                      # (B, P, n_t): undelayed state at t_j
    q = theta / dt                                                        # delay in samples (P,)
    shift = torch.floor(q).detach()                                       # whole samples
    frac = q - shift                                                      # in [0, 1)
    # tau_j = t_j - theta = t_{j - shift - 1} + (1 - frac) dt  (when frac > 0); use m = j - shift - 1, r = (1 - frac) dt
    r = (1 - frac) * dt                                                   # (P,) in (0, dt]
    E, I0, I1 = partial_step(mu, r)                                       # (P,)
    j = torch.arange(n_t, device=u.device)
    m = j[None, :] - shift.long()[:, None] - 1                           # (P, n_t)
    valid = m >= 0
    mc = m.clamp(0, n_t - 2)
    xm = torch.gather(x, 2, mc[None].expand(x.shape[0], -1, -1))
    um = uc[:, mc]                                                        # (B, P, n_t)
    up = uc[:, (mc + 1).clamp(max=n_t - 1)]
    y = E[None, :, None] * xm + I0[None, :, None] * um + I1[None, :, None] * (up - um) / dt
    # tau in [t_{-1}, t_0): before the input starts (rest); covered by valid = m >= 0
    return torch.where(valid[None], y, torch.zeros_like(y))


class PoleResidueDelay(nn.Module):
    """SISO pole-residue-delay operator, sum_n beta_n e^{-theta_n s} / (s - mu_n), on (batch, n_t) signals.

    n_poles     N
    real        real poles (mu = -softplus(.) if stable) with real residues; else complex poles, output 2 Re(.)
    shared_delay one delay for all terms
    """

    def __init__(self, n_poles: int, dt: float, real: bool = True, stable: bool = True, shared_delay: bool = True,
                 theta0: float = 1.0, init_scale: float = 1.0):
        super().__init__()
        self.dt, self.real, self.stable = float(dt), real, stable
        self.mu_re = nn.Parameter(init_scale * torch.randn(n_poles, dtype=torch.float64))
        self.mu_im = None if real else nn.Parameter(init_scale * torch.randn(n_poles, dtype=torch.float64))
        self.beta_re = nn.Parameter(torch.full((n_poles,), 1.0 / n_poles, dtype=torch.float64))
        self.beta_im = None if real else nn.Parameter(torch.zeros(n_poles, dtype=torch.float64))
        raw = torch.log(torch.expm1(torch.tensor(float(theta0), dtype=torch.float64)))   # softplus^-1
        self.theta_raw = nn.Parameter(raw.repeat(1 if shared_delay else n_poles))

    def poles(self):
        re = -F.softplus(self.mu_re) if self.stable else self.mu_re
        mu = torch.complex(re, torch.zeros_like(re) if self.real else self.mu_im)
        beta = torch.complex(self.beta_re, torch.zeros_like(self.beta_re) if self.real else self.beta_im)
        theta = F.softplus(self.theta_raw).expand(mu.shape[0])
        return mu, beta, theta

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        mu, beta, theta = self.poles()
        y = (beta[None, :, None] * delayed_foh(mu, theta, u, self.dt)).sum(1)
        return (y.real if self.real else 2 * y.real).to(u.dtype)
