"""Forced 1D Cahn-Hilliard equation on a periodic domain (project A, RQ3: an unstable band).

    u_t = M d_xx ( u^3 - u - kappa d_xx u ) + f(x, t),      x in [0, 2 pi) periodic,  u(x, 0) = 0.

Linearised at u = 0, every mode has the pole

    lambda(k) = M k^2 (1 - kappa k^2),

positive for 0 < k < k_c = 1 / sqrt(kappa) (spinodal band), largest at k_max = k_c / sqrt(2). The cubic term
saturates the growth (phase separation towards u = +-1).

**Initial state and mean composition** (project D2 / A application 2). `solve(..., u0k=...)` starts from a given
field instead of rest. Its mean (mode 0, rfft normalisation: n_x * phi0) is conserved. Linearised about a uniform
phi0, the cubic term adds 3 phi0^2 to f'' = -1, so each mode has

    lambda(k; phi0) = M k^2 (1 - 3 phi0^2 - kappa k^2)      (`growth_rate(k, M, kappa, phi0)`),

unstable for 0 < k < k_c(phi0) = sqrt((1 - 3 phi0^2) / kappa) inside the spinodal |phi0| < 1 / sqrt(3).

Numerics as `lfo.data.burgers`: pseudo-spectral in x, integrating-factor RK4 in t (the linear part, including the
stiff kappa k^4 term, is exact). The cubic term is dealiased by the 1/2 rule (modes with k > n_x / 4 are dropped
from u^3), so the solution lives on k <= n_x / 4 once the forcing does.

`solve2d` also takes an initial field, and runs unforced (fk = None with n_steps) for free evolution such as
coarsening (application 2, stage 2), without holding a forcing array of length n_t.
"""
from __future__ import annotations

import numpy as np

__all__ = ["solve", "solve2d", "growth_rate"]


def growth_rate(k, M: float, kappa: float, phi0: float = 0.0):
    k = np.asarray(k, dtype=float)
    return M * k ** 2 * (1 - 3 * phi0 ** 2 - kappa * k ** 2)


def solve(fk: np.ndarray, dt: float, M: float, kappa: float, n_x: int, stride: int = 1,
          u0k: np.ndarray | None = None) -> np.ndarray:
    """Spectral solution. fk (B, K, n): rfft of the forcing at t_j = j dt (linear between samples),
    K = n_x // 2 + 1; u0k: optional (B, K) rfft of the initial field (default: rest).
    Returns uk (B, K, ceil(n / stride)) at t_0, t_stride, ..."""
    B, K, n = fk.shape
    if K != n_x // 2 + 1:
        raise ValueError("fk must hold the rfft of an n_x-point grid")
    k = np.arange(K, dtype=float)
    L = growth_rate(k, M, kappa)
    E, E2 = np.exp(L * dt / 2), np.exp(L * dt)
    keep = k <= n_x / 4
    nk = -M * k ** 2 * keep

    def N(u, f):
        w = np.fft.irfft(u, n=n_x, axis=1)
        return nk * np.fft.rfft(w ** 3, axis=1) + f

    u = np.zeros((B, K), complex) if u0k is None else np.array(u0k, dtype=complex)
    out = [u.copy()]
    for j in range(n - 1):
        f0, f1 = fk[..., j], fk[..., j + 1]
        fm = 0.5 * (f0 + f1)
        a = N(u, f0)
        u1 = E * (u + 0.5 * dt * a)
        b = N(u1, fm)
        u2 = E * u + 0.5 * dt * b
        c = N(u2, fm)
        u3 = E2 * u + dt * E * c
        d = N(u3, f1)
        u = E2 * u + dt / 6 * (E2 * a + 2 * E * (b + c) + d)
        if (j + 1) % stride == 0:
            out.append(u.copy())
    return np.stack(out, axis=-1)


def solve2d(fk: np.ndarray | None, dt: float, M: float, kappa: float, n: int, stride: int = 1,
            u0k: np.ndarray | None = None, n_steps: int | None = None) -> np.ndarray:
    """2D version of `solve`. fk (B, n, n // 2 + 1, n_t): rfft2 of the forcing. Returns uk (B, n, n // 2 + 1, .).

    u0k: optional (B, n, n // 2 + 1) rfft2 of the initial field (default: rest). Unforced runs (free evolution,
    e.g. coarsening) pass fk = None with u0k and n_steps, so no forcing array of length n_t is held in memory."""
    if fk is None:
        if u0k is None or n_steps is None:
            raise ValueError("fk = None needs u0k and n_steps")
        B, nx, nyh = u0k.shape
        nt = n_steps + 1
    else:
        B, nx, nyh, nt = fk.shape
    if (nx, nyh) != (n, n // 2 + 1):
        raise ValueError("fk / u0k must hold the rfft2 of an n x n grid")
    kx = np.fft.fftfreq(n, 1 / n)[:, None]
    ky = np.fft.rfftfreq(n, 1 / n)[None, :]
    k2 = kx ** 2 + ky ** 2
    L = M * k2 * (1 - kappa * k2)
    E, E2 = np.exp(L * dt / 2), np.exp(L * dt)
    nk = -M * k2 * ((np.abs(kx) <= n / 4) & (ky <= n / 4))

    def N(u, f):
        w = np.fft.irfft2(u, s=(n, n), axes=(1, 2))
        r = nk * np.fft.rfft2(w ** 3, axes=(1, 2))
        return r if f is None else r + f

    zero = lambda j: None
    F = zero if fk is None else (lambda j: fk[..., j])
    u = np.zeros((B, n, n // 2 + 1), complex) if u0k is None else np.array(u0k, dtype=complex)
    out = [u.copy()]
    for j in range(nt - 1):
        f0, f1 = F(j), F(j + 1)
        fm = None if f0 is None else 0.5 * (f0 + f1)
        a = N(u, f0)
        u1 = E * (u + 0.5 * dt * a)
        b = N(u1, fm)
        u2 = E * u + 0.5 * dt * b
        c = N(u2, fm)
        u3 = E2 * u + dt * E * c
        d = N(u3, f1)
        u = E2 * u + dt / 6 * (E2 * a + 2 * E * (b + c) + d)
        if (j + 1) % stride == 0:
            out.append(u.copy())
    return np.stack(out, axis=-1)
