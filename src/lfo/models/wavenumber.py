"""Wavenumber-dependent pole layer (project A), 1D space x time.

Plan: `docs/proposal_A_wavenumber_pole_operator.qmd`, Section 4. For a linear PDE u_t = L u + f with a periodic
spatial axis, every spatial Fourier mode k obeys d u_k / dt = lambda(k) u_k + f_k. The layer learns the temporal
kernel of each mode as a pole-residue sum whose poles and residues are functions of k:

    K(k, t) = sum_n beta_n(k) e^{mu_n(k) t},    v_k(t) = int_0^t K(k, t - s) f_k(s) ds   (zero initial state).

**Time axis** (D1 decision, `experiments/D/d1_inputs/README.md`): the first-order-hold exponential-integrator
recursion. The response is exact for an input that is linear between samples; it is causal by construction and makes
no periodicity assumption in time. It is computed as a causal convolution with the discrete FOH impulse response
(FFT with zero padding, O(n log n)) plus a correction for the first sample; `foh_recursion` is the O(n) loop
form, used to check it.

**Pole functions** (Section 4.3, polynomial form). Over the wavenumber k of the spatial grid:

    mu_n(k)   = sum_j a_nj k^{2j}  +  i sum_j c_nj k^{2j+1}      (even real part, odd imaginary part)
    beta_n(k) = sum_j (b_nj + i e_nj) k^{2j}

For real fields this gives mu(-k) = conj(mu(k)), as needed. `stable=True` maps the real part through
-softplus(.) so Re mu_n(k) <= 0. Parameters are continuous in k, so the layer can be evaluated on a finer spatial
grid than it was trained on.

Input / output: (batch, n_x, n_t) real fields on a uniform periodic x grid of length `length` and a uniform t grid.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["foh_coefficients", "foh_kernel", "foh_recursion", "foh_response", "WavenumberPoleLayer1d"]


def foh_coefficients(mu: torch.Tensor, dt: float):
    """(E, b0, b1) of the exact FOH step x_{j+1} = E x_j + b0 u_j + b1 u_{j+1} for the mode 1/(s - mu).
    Computed in complex128; I1 cancels for small mu dt, so a series is used there."""
    mu = mu.to(torch.complex128)
    z = mu * dt
    E = torch.exp(z)
    small = z.abs() < 1e-3
    zs = torch.where(small, torch.ones_like(z), z)
    # I0 = (E - 1) / mu, I1 = (E - 1 - z) / mu^2, written in z to avoid 0 / 0
    I0 = torch.where(small, dt * (1 + z / 2 + z ** 2 / 6 + z ** 3 / 24), (E - 1) / zs * dt)
    I1 = torch.where(small, dt ** 2 * (0.5 + z / 6 + z ** 2 / 24 + z ** 3 / 120), (E - 1 - zs) / zs ** 2 * dt ** 2)
    return E, I0 - I1 / dt, I1 / dt


def foh_kernel(mu: torch.Tensor, n_t: int, dt: float):
    """Discrete FOH impulse responses of 1/(s - mu), each of shape mu.shape + (n_t,):

    - g: response to a unit sample u_k at an interior index k >= 1, shifted to start at k (g_0 = b1,
      g_j = E^{j-1} (E b1 + b0));
    - c: the correction for the first sample. From x_0 = 0 the ramp before t_0 does not exist, so the response
      to u_0 is g_j - b1 E^j."""
    E, b0, b1 = foh_coefficients(mu, dt)
    j = torch.arange(n_t, device=mu.device, dtype=torch.float64)
    Ej = torch.exp(mu.to(torch.complex128)[..., None] * dt * j)                  # E^j
    Ejm1 = torch.exp(mu.to(torch.complex128)[..., None] * dt * (j - 1))           # E^{j-1}
    g = torch.where(j == 0, b1[..., None], (E * b1 + b0)[..., None] * Ejm1)
    return g, b1[..., None] * Ej


def foh_recursion(mu: torch.Tensor, u: torch.Tensor, dt: float) -> torch.Tensor:
    """O(n) loop form, for tests: x_{j+1} = E x_j + b0 u_j + b1 u_{j+1}, x_0 = 0. mu: (...), u: (..., n_t)."""
    E, b0, b1 = foh_coefficients(mu, dt)
    u = u.to(torch.complex128)
    x = torch.zeros(u.shape[:-1], dtype=torch.complex128, device=u.device)
    out = [x]
    for j in range(u.shape[-1] - 1):
        x = E * x + b0 * u[..., j] + b1 * u[..., j + 1]
        out.append(x)
    return torch.stack(out, dim=-1)


def causal_conv(g: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """y_j = sum_{k <= j} g_{j-k} u_k along the last axis (complex), by FFT with zero padding."""
    n = u.shape[-1]
    return torch.fft.ifft(torch.fft.fft(g, 2 * n) * torch.fft.fft(u, 2 * n))[..., :n]


def foh_response(g: torch.Tensor, c: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """Exact FOH response from rest: causal convolution with g, minus the first-sample correction u_0 c."""
    return causal_conv(g, u) - u[..., :1] * c


class WavenumberPoleLayer1d(nn.Module):
    """K(k, s) = sum_n beta_n(k) / (s - mu_n(k)), polynomial in k, FOH time axis. See the module docstring.

    n_poles     N
    mu_degree   number of even (real) and odd (imaginary) coefficients of mu_n(k): (n_re, n_im)
    beta_degree number of coefficients of beta_n(k) in k^2
    """

    def __init__(self, n_poles: int, t, length: float = 2 * math.pi, mu_degree=(3, 2), beta_degree: int = 2,
                 stable: bool = False, init_scale: float = 0.1):
        super().__init__()
        t = torch.as_tensor(t, dtype=torch.float64).reshape(-1)
        self.register_buffer("t", t)
        self.length, self.stable = float(length), stable
        n_re, n_im = mu_degree
        self.mu_re = nn.Parameter(init_scale * torch.randn(n_poles, n_re, dtype=torch.float64))
        self.mu_im = nn.Parameter(init_scale * torch.randn(n_poles, n_im, dtype=torch.float64))
        self.beta = nn.Parameter(torch.zeros(n_poles, beta_degree, 2, dtype=torch.float64))
        with torch.no_grad():
            self.beta[:, 0, 0] = 1.0 / n_poles

    def wavenumbers(self, n_x: int) -> torch.Tensor:
        """Angular wavenumbers of the rfft of an n_x-point periodic grid of length `length`."""
        return 2 * math.pi * torch.fft.rfftfreq(n_x, self.length / n_x, device=self.t.device, dtype=torch.float64)

    def poles(self, k: torch.Tensor):
        """(mu, beta), each (N, len(k)) complex128."""
        k2 = k ** 2
        re = sum(self.mu_re[:, j, None] * k2 ** j for j in range(self.mu_re.shape[1]))
        if self.stable:
            re = -F.softplus(re)
        im = sum(self.mu_im[:, j, None] * k * k2 ** j for j in range(self.mu_im.shape[1]))
        b = sum(torch.complex(self.beta[:, j, 0, None], self.beta[:, j, 1, None]) * k2 ** j
                for j in range(self.beta.shape[1]))
        return torch.complex(re, im), b

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        b_, n_x, n_t = f.shape
        dt = float(self.t[1] - self.t[0])
        fk = torch.fft.rfft(f.to(torch.float64), dim=1)                       # (b, K, n_t)
        mu, beta = self.poles(self.wavenumbers(n_x))                          # (N, K)
        g, c = foh_kernel(mu, n_t, dt)                                         # (N, K, n_t) each
        g, c = (beta[..., None] * g).sum(0), (beta[..., None] * c).sum(0)     # (K, n_t): sum_n beta_n (.)
        vk = foh_response(g[None], c[None], fk)                               # (b, K, n_t)
        return torch.fft.irfft(vk, n=n_x, dim=1).to(f.dtype)
