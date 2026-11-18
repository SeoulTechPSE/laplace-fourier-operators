"""Forced 1D Cahn-Hilliard equation on a periodic domain (project A, RQ3: an unstable band).

    u_t = M d_xx ( u^3 - u - kappa d_xx u ) + f(x, t),      x in [0, 2 pi) periodic,  u(x, 0) = 0.

Linearised at u = 0, every mode has the pole

    lambda(k) = M k^2 (1 - kappa k^2),

positive for 0 < k < k_c = 1 / sqrt(kappa) (spinodal band), largest at k_max = k_c / sqrt(2). The cubic term
saturates the growth (phase separation towards u = +-1).

Numerics as `lfo.data.burgers`: pseudo-spectral in x, integrating-factor RK4 in t (the linear part, including the
stiff kappa k^4 term, is exact). The cubic term is dealiased by the 1/2 rule (modes with k > n_x / 4 are dropped
from u^3), so the solution lives on k <= n_x / 4 once the forcing does.
"""
from __future__ import annotations

import numpy as np

__all__ = ["solve", "solve2d", "growth_rate"]


def growth_rate(k, M: float, kappa: float):
    k = np.asarray(k, dtype=float)
    return M * k ** 2 * (1 - kappa * k ** 2)


def solve(fk: np.ndarray, dt: float, M: float, kappa: float, n_x: int, stride: int = 1) -> np.ndarray:
    """Spectral solution from rest. fk (B, K, n): rfft of the forcing at t_j = j dt (linear between samples),
    K = n_x // 2 + 1. Returns uk (B, K, ceil(n / stride)) at t_0, t_stride, ..."""
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

    u = np.zeros((B, K), complex)
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


def solve2d(fk: np.ndarray, dt: float, M: float, kappa: float, n: int, stride: int = 1) -> np.ndarray:
    """2D version of `solve`. fk (B, n, n // 2 + 1, n_t): rfft2 of the forcing. Returns uk (B, n, n // 2 + 1, .)."""
    B, nx, nyh, nt = fk.shape
    if (nx, nyh) != (n, n // 2 + 1):
        raise ValueError("fk must hold the rfft2 of an n x n grid")
    kx = np.fft.fftfreq(n, 1 / n)[:, None]
    ky = np.fft.rfftfreq(n, 1 / n)[None, :]
    k2 = kx ** 2 + ky ** 2
    L = M * k2 * (1 - kappa * k2)
    E, E2 = np.exp(L * dt / 2), np.exp(L * dt)
    nk = -M * k2 * ((np.abs(kx) <= n / 4) & (ky <= n / 4))

    def N(u, f):
        w = np.fft.irfft2(u, s=(n, n), axes=(1, 2))
        return nk * np.fft.rfft2(w ** 3, axes=(1, 2)) + f

    u = np.zeros((B, n, n // 2 + 1), complex)
    out = [u.copy()]
    for j in range(nt - 1):
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
