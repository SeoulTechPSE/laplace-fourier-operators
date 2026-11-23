"""Forced viscous Burgers equation on a periodic domain (project A, first nonlinear benchmark).

    u_t + u u_x = nu u_xx + f(x, t),      x in [0, 2 pi) periodic,  u(x, 0) = 0.

Pseudo-spectral in x (rfft, 2/3-rule dealiasing of u^2), integrating-factor RK4 in t (the diffusion is exact; the
nonlinear term and the forcing are integrated by RK4). The forcing is given on the solver's time grid and taken as
linear between its samples (so midpoints use the average of neighboring samples).

Linearized at u = 0 the equation is diffusion: each mode has the pole lambda(k) = -nu k^2. For small forcing the
solution tends to the linear response (tested).

With `delta` != 0 the solver integrates the forced KdV-Burgers equation u_t + u u_x = nu u_xx - delta u_xxx + f,
whose linearization is dispersive: lambda(k) = -nu k^2 + i delta k^3 (exact in the integrating factor).
"""
from __future__ import annotations

import numpy as np

__all__ = ["solve"]


def solve(fk: np.ndarray, dt: float, nu: float, n_x: int, stride: int = 1, u0k: np.ndarray | None = None,
          delta: float = 0.0) -> np.ndarray:
    """Spectral solution of the forced Burgers equation from rest.

    fk     (B, K, n) complex: rfft (over n_x points, numpy normalization) of the forcing at t_j = j dt, K = n_x // 2 + 1
    dt     solver step = forcing sample spacing
    stride return every `stride`-th time sample
    u0k    optional (B, K) rfft of the initial field (default: rest)
    delta  dispersion coefficient of -delta u_xxx (default 0: Burgers)

    Returns uk (B, K, ceil(n / stride)) complex, the rfft of u at t_0, t_stride, ...
    """
    B, K, n = fk.shape
    if K != n_x // 2 + 1:
        raise ValueError("fk must hold the rfft of an n_x-point grid")
    k = np.arange(K, dtype=float)
    L = -nu * k ** 2 + 1j * delta * k ** 3
    E, E2 = np.exp(L * dt / 2), np.exp(L * dt)
    keep = k <= n_x / 3                                   # 2/3 rule for the quadratic term
    ik2 = -0.5j * k * keep

    def N(u, f):
        w = np.fft.irfft(u, n=n_x, axis=1)
        return ik2 * np.fft.rfft(w * w, axis=1) + f

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
