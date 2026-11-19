"""(A) Application 1: a dispersion reactor with real inlet / outlet boundaries (Danckwerts), inlet -> outlet.

a24 used a periodic analogue. A real tubular reactor has an inlet and an outlet, so Fourier modes do not
diagonalise it. Its modes are the eigenfunctions of the dispersion operator with Danckwerts boundary conditions,
and its transfer function is still a sum of poles, one per mode. This run asks whether a "dispersion relation over
the mode index" captures it, with the mode number n in place of the wavenumber.

**System.** c_t = -v c_x + D c_xx - k_r c on [0, 1]. Inlet (Danckwerts): v c_in(t) = v c - D c_x at x = 0; outlet:
c_x = 0 at x = 1. v = 1, D = 0.05 (Peclet 20), k_r = 0.5; mean residence time 1. Input: the inlet concentration
c_in(t); output: the outlet concentration c(1, t). Truth: central finite differences on 400 cells, Crank-Nicolson on
a 32x finer time grid. The reference eigenvalues are the least-negative eigenvalues of that matrix (all real). T = 6,
dt = 0.02 (n_t = 301); E1-family inputs; splits 200 / 50 / 100; data seed 0. Deviation variables, from rest.

**Models** (output error, L-BFGS 300, 10 starts, selection by validation):
- `fopdt`, `sopdt`: `PoleResidueDelay`, real poles, one delay;
- `poles6`: 6 free real poles, free residues, no delay;
- `modal8`: K(s) = sum_{n=1..8} beta_n / (s - lambda_n) with lambda_n = -(a0 + a1 n^2), a0, a1 >= 0 (softplus): 2 pole
  parameters and 8 free residues.

**Predictions** (written before running).

1. `modal8` has the lowest test error of the four, and FOPDT / SOPDT are at least 3x above it.
2. `modal8`'s lambda_1..lambda_3 match the three least-negative reference eigenvalues (-5.846, -6.909, -8.753)
   within 8 %, 5 % and 5 %. Measured before any fit: the Danckwerts eigenvalues do not follow n^2 exactly, and the
   best least-squares a0 + a1 n^2 over n = 1..8 is itself 5.4 %, 2.2 % and 0.4 % off there.
3. `poles6`'s slowest pole is within 5 % of lambda_1.

    python experiments/A/process/a27_danckwerts.py      # results/a27_danckwerts.md / .json
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import a20_fopdt as P  # noqa: E402
from lfo.models import PoleResidueDelay  # noqa: E402
from lfo.models.wavenumber import foh_kernel, foh_response  # noqa: E402

V, DIF, KR, M = 1.0, 0.05, 0.5, 400
T, NT, FINE, SPLIT = 6.0, 301, 32, (200, 50, 100)
DT = T / (NT - 1)
OUT = HERE / "results" / "a27_danckwerts.json"


def operator():
    """c' = A c + b c_in on nodes x_0..x_M (central differences, ghost nodes for the boundary conditions)."""
    h = 1.0 / M
    n = M + 1
    A = np.zeros((n, n))
    b = np.zeros(n)
    lo, di, up = V / (2 * h) + DIF / h ** 2, -2 * DIF / h ** 2 - KR, -V / (2 * h) + DIF / h ** 2
    for i in range(n):
        A[i, i] = di
        if i > 0:
            A[i, i - 1] += lo
        if i < M:
            A[i, i + 1] += up
    # inlet ghost: c_{-1} = c_1 - 2 h v (c_0 - c_in) / D
    A[0, 1] += lo
    A[0, 0] += -lo * 2 * h * V / DIF
    b[0] = lo * 2 * h * V / DIF
    # outlet ghost: c_{M+1} = c_{M-1}
    A[M, M - 1] += up
    return A, b


def make_data():
    A_, b_ = operator()
    rng = np.random.default_rng(0)
    n = sum(SPLIT)
    tf = np.linspace(0, T, (NT - 1) * FINE + 1)
    dtf = tf[1] - tf[0]
    u = P.A.profiles(tf, rng, n)
    I = np.eye(M + 1)
    Lm = np.linalg.inv(I - dtf / 2 * A_)
    R = Lm @ (I + dtf / 2 * A_)
    Bv = Lm @ (dtf * b_)
    c = np.zeros((M + 1, n))
    y = np.zeros((n, tf.size))
    for j in range(tf.size - 1):
        c = R @ c + Bv[:, None] * (0.5 * (u[:, j] + u[:, j + 1]))[None]
        y[:, j + 1] = c[M]
    lam = np.sort(np.linalg.eigvals(A_).real)[::-1]
    u, y = torch.tensor(u[:, ::FINE]), torch.tensor(y[:, ::FINE])
    cut = np.cumsum([0, *SPLIT])
    return {k: (u[a:b], y[a:b]) for k, a, b in zip(("train", "vali", "test"), cut[:-1], cut[1:])}, lam


class Modal(nn.Module):
    """sum_n beta_n / (s - lambda_n), lambda_n = -(a0 + a1 n^2), n = 1..N, real."""

    def __init__(self, n_modes=8, a0=1.0, a1=0.5):
        super().__init__()
        inv = lambda x: math.log(math.expm1(x))
        self.a0 = nn.Parameter(torch.tensor(inv(a0), dtype=torch.float64))
        self.a1 = nn.Parameter(torch.tensor(inv(a1), dtype=torch.float64))
        self.beta = nn.Parameter(torch.randn(n_modes, dtype=torch.float64) * 0.5)
        self.register_buffer("n2", torch.arange(1, n_modes + 1, dtype=torch.float64) ** 2)

    def lam(self):
        return -(F.softplus(self.a0) + F.softplus(self.a1) * self.n2)

    def forward(self, u):
        mu = self.lam().to(torch.complex128)
        g, c = foh_kernel(mu, u.shape[-1], DT)
        x = foh_response(g[None], c[None], u.to(torch.complex128)[:, None])     # (B, N, n_t)
        return (self.beta[None, :, None] * x.real).sum(1)


def fit(make, data, device, n_starts=10):
    best = None
    for s in range(n_starts):
        torch.manual_seed(s)
        m = make(s).to(device)
        v = P.lbfgs(m, data, device)
        if math.isfinite(v) and (best is None or v < best[0]):
            best = (v, m)
    return best[1]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data, lam = make_data()
    P.DT = DT
    print("reference eigenvalues", np.round(lam[:8], 4), flush=True)
    xte, yte = (x.to(device) for x in data["test"])
    makers = {
        "fopdt": lambda s: PoleResidueDelay(1, DT, theta0=(0.2, 0.5, 1.0, 1.5, 2.0)[s % 5], init_scale=1.0),
        "sopdt": lambda s: PoleResidueDelay(2, DT, theta0=(0.2, 0.5, 1.0, 1.5, 2.0)[s % 5], init_scale=1.0),
        "poles6": lambda s: PoleResidueDelay(6, DT, theta0=1e-6, init_scale=2.0),
        "modal8": lambda s: Modal(8, a0=float(np.exp(np.random.default_rng(s).uniform(-1, 2.5))),
                                  a1=float(np.exp(np.random.default_rng(100 + s).uniform(-2, 1)))),
    }
    out = dict(lam_ref=lam[:8].tolist(), models={})
    for name, mk in makers.items():
        if name == "poles6":
            def mk2(s, mk=mk):
                m = mk(s)
                m.theta_raw.requires_grad_(False)
                return m
            m = fit(mk2, data, device)
        else:
            m = fit(mk, data, device)
        with torch.no_grad():
            test = float(P.rel(m(xte), yte))
            if name == "modal8":
                poles = sorted(m.lam().cpu().numpy().tolist(), reverse=True)
                extra = dict(a0=float(F.softplus(m.a0)), a1=float(F.softplus(m.a1)))
            else:
                mu, beta, theta = m.poles()
                poles = sorted(mu.real.cpu().numpy().tolist(), reverse=True)
                extra = dict(theta=float(theta[0]))
        out["models"][name] = dict(test=test, poles=poles, **extra)
        print(name, f"test {test:.3e}", "poles", np.round(poles[:4], 4), extra, flush=True)
        OUT.parent.mkdir(exist_ok=True)
        OUT.write_text(json.dumps(out, indent=1))
    lines = ["# (A) Danckwerts dispersion reactor, inlet → outlet", "",
             f"v {V}, D {DIF} (Pe 20), k_r {KR}. Reference eigenvalues (FD, 400 cells): " + ", ".join(f"{x:.3f}" for x in lam[:5]) + ".", "",
             "| model | test | slowest poles | other |", "|---|---|---|---|"]
    for name, r in out["models"].items():
        other = ", ".join(f"{k} {v:.4f}" for k, v in r.items() if k in ("a0", "a1", "theta"))
        lines.append(f"| {name} | {r['test']:.3e} | " + ", ".join(f"{p:.3f}" for p in r["poles"][:3]) + f" | {other} |")
    (OUT.with_suffix(".md")).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
