"""Synthetic linear time-invariant (LTI) systems with known poles.

Used for the (C) audit and as linear benchmarks for (A) and (D). The response
y(t) = sum_n beta_n * int_0^t exp(mu_n (t - s)) u(s) ds  (+ homogeneous part)
is computed exactly for piecewise-linear inputs (first-order hold), so the
ground truth carries no discretization error from the solver.
"""
from __future__ import annotations

import numpy as np


def random_poles(n: int, rng: np.random.Generator, re_range=(-3.0, -0.1),
                 im_range=(0.5, 10.0), complex_fraction: float = 0.5):
    """Draw ``n`` stable poles closed under conjugation, with residues.

    Complex poles come in conjugate pairs, so the realized count may be ``n``
    or ``n + 1``; the returned arrays are always conjugate-symmetric.
    Returns (mu, beta) as complex arrays.
    """
    mu, beta = [], []
    while len(mu) < n:
        re = rng.uniform(*re_range)
        if rng.random() < complex_fraction and len(mu) + 2 <= n + 1:
            im = rng.uniform(*im_range)
            b = complex(rng.normal(), rng.normal())
            mu += [complex(re, im), complex(re, -im)]
            beta += [b, np.conj(b)]
        else:
            mu.append(complex(re, 0.0))
            beta.append(complex(rng.normal(), 0.0))
    return np.array(mu), np.array(beta)


def response(t, u, mu, beta, x0=None) -> np.ndarray:
    """Exact response of a pole-residue system to a piecewise-linear input.

    Parameters
    ----------
    t : (T,) uniform time grid starting at 0.
    u : (T,) or (B, T) input samples (linear between samples).
    mu, beta : (N,) complex poles and residues (conjugate-symmetric for real output).
    x0 : optional (N,) or (B, N) homogeneous-solution coefficients c_n; adds
         sum_n c_n exp(mu_n t) to the forced response (conjugate-symmetric for
         a real result).

    Returns
    -------
    y : real array with the shape of ``u``.
    """
    t = np.asarray(t, dtype=float)
    u = np.asarray(u, dtype=float)
    squeeze = u.ndim == 1
    if squeeze:
        u = u[None, :]
    mu = np.asarray(mu, dtype=complex)
    beta = np.asarray(beta, dtype=complex)
    dt = t[1] - t[0]
    if not np.allclose(np.diff(t), dt):
        raise ValueError("time grid must be uniform")

    z = mu * dt
    E = np.exp(z)
    small = np.abs(z) < 1e-6
    safe_mu = np.where(small, 1.0, mu)
    I0 = np.where(small, dt, (E - 1.0) / safe_mu)                     # int_0^dt e^{mu(dt-s)} ds
    I1 = np.where(small, dt**2 / 2, (E - 1.0 - z) / safe_mu**2)        # int_0^dt e^{mu(dt-s)} s ds

    B, T = u.shape
    # forced response, zero initial state
    x = np.zeros((B, mu.size), dtype=complex)
    xs = np.empty((B, T, mu.size), dtype=complex)
    xs[:, 0] = x
    for j in range(T - 1):
        du = (u[:, j + 1] - u[:, j])[:, None] / dt
        x = E * x + u[:, j][:, None] * I0 + du * I1
        xs[:, j + 1] = x
    y = np.real(np.einsum("btn,n->bt", xs, beta))

    # homogeneous response: sum_n c_n exp(mu_n t), with c_n = x0_n
    if x0 is not None:
        c = np.broadcast_to(np.asarray(x0, dtype=complex), (B, mu.size))
        y = y + np.real(c @ np.exp(np.outer(mu, t)))
    return y[0] if squeeze else y


# ---- input families -------------------------------------------------------

def step(t, t0: float = 0.0, amplitude: float = 1.0):
    return amplitude * (np.asarray(t) >= t0).astype(float)


def ramp(t, t0: float = 0.0, slope: float = 1.0):
    t = np.asarray(t)
    return slope * np.clip(t - t0, 0.0, None)


def prbs(t, rng: np.random.Generator, hold: float = 0.5, amplitude: float = 1.0):
    """Pseudo-random binary sequence switching between ±amplitude every ``hold`` seconds."""
    t = np.asarray(t)
    n = int(np.ceil(t[-1] / hold)) + 1
    levels = amplitude * rng.choice([-1.0, 1.0], size=n)
    return levels[(t // hold).astype(int)]


def smooth_random(t, rng: np.random.Generator, n_modes: int = 8, decay: float = 1.5):
    """Smooth random signal with non-matching endpoints (non-periodic on [0, T])."""
    t = np.asarray(t)
    T = t[-1] if t[-1] > 0 else 1.0
    k = np.arange(1, n_modes + 1)
    a = rng.normal(size=n_modes) / k**decay
    ph = rng.uniform(0, 2 * np.pi, size=n_modes)
    # half-integer frequencies break periodicity on [0, T]
    return np.sum(a[:, None] * np.sin((k[:, None] - 0.5) * np.pi * t[None, :] / T + ph[:, None]), axis=0)
