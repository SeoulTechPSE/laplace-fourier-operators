"""(D) Non-periodic space: diffusion with time-varying Dirichlet boundary values, as a boundary channel that shares
the poles.

Plan D, Table "Problems": diffusion with a time-varying boundary or source. The periodic layer cannot represent
boundaries. On [0, L] with Dirichlet values u(0, t) = a(t), u(L, t) = b(t), the discrete sine transform (DST-I) of
the interior points diagonalises the operator. The boundary values then enter every mode as a forcing:

    d u_m / dt = lambda(k_m) u_m + f_m(t) + c(k_m) (a(t) - (-1)^m b(t)),    k_m = m pi / L,

with c(k) = 2 D k / L in the sine-Galerkin model (an odd function of k). The model is the wavenumber
pole layer on the sine modes, with a boundary channel whose coefficient c(k) = c1 k + c3 k^3 is learned. The boundary
signals reach every mode through the same poles mu(k).

**System.** u_t = D u_xx + f on [0, 1], D = 0.02. n = 63 interior points (h = 1 / 64), T = 5, n_t = 201.
- Forcing: 4 sine modes m in 1..16 with E1 profiles, amplitude ~ 1 / m.
- Boundary values a, b: E1 profiles minus their value at t = 0, so they start at rest.
- Truth: the 63-mode sine-Galerkin solution. Per mode, lambda_m = -D k_m^2 and boundary coefficient
  c(k_m) = 2 D k_m / L. It is integrated with the exact first-order hold on a 32x finer time grid and sampled at the
  interior points, where the DST-I inverts exactly.
- A finite-difference truth was considered first. Its discrete eigenvalues -(4 D / h^2) sin^2(k h / 2) are not
  polynomial over all 63 modes (a {1, k^2, k^4} fit is 1.4 % off, and its k^2 coefficient gives D = 0.0181), so
  "D within 1 %" would have tested the discretisation rather than the model. This was decided before any fit.
- Splits 200 / 50 / 100; data seeds 0-4.

**Models** (modes 1..63, Re mu on {1, k^2, k^4}, Im mu free, k_ref = 63 pi; leaky Re cap; the a7 joint fit with
8 starts, L-BFGS, selection by validation):
- `sine_bc`: sine modes with the boundary channel;
- `sine_nobc`: sine modes, forcing only;
- `periodic`: the periodic (rfft) layer on the interior field, forcing only.

**Oracle, measured before any fit** (seed 0): 1.03e-2 (the sampling floor of steps and PRBS, as in a1).

**Predictions** (written before running).

1. `sine_bc`: test error <= 1.2x the oracle, D within 1 % (from the k^2 coefficient) and c1 within 2 % of 2 D / L,
   in every seed.
2. `sine_nobc` and `periodic`: test error >= 3x the oracle in every seed.

    python experiments/D/d3_boundary/d3_dirichlet.py      # results/d3_dirichlet.md / .json
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "A" / "layer"))
import a1_dispersion as A  # noqa: E402
from lfo.models.wavenumber import WavenumberPoleLayer1d, foh_kernel, foh_response  # noqa: E402

DIF, N, T, NT, FINE, SPLIT = 0.02, 63, 5.0, 201, 32, (200, 50, 100)
H = 1.0 / (N + 1)
KM = np.arange(1, N + 1) * np.pi                     # k_m = m pi / L, L = 1
K_REF = float(KM[-1])
OUT = HERE / "results" / "d3_dirichlet.json"
S = np.sin(np.outer(np.arange(1, N + 1), np.arange(1, N + 1)) * np.pi / (N + 1))   # DST-I matrix (symmetric)


def dst(u):
    """Interior field (..., N, n_t) -> sine coefficients (..., N, n_t)."""
    return torch.einsum("mi,...it->...mt", torch.tensor(S * 2 / (N + 1)), u)


def idst(c):
    return torch.einsum("im,...mt->...it", torch.tensor(S), c)


def c_true(k):
    return 2 * DIF * k


def make_data(seed):
    rng = np.random.default_rng(500 + seed)
    n = sum(SPLIT)
    tf = np.linspace(0, T, (NT - 1) * FINE + 1)
    fm = np.zeros((n, N, tf.size))
    for _ in range(4):
        prof = A.profiles(tf, rng, n)
        m = rng.integers(1, 17, size=n)
        fm[np.arange(n), m - 1] += (rng.normal(size=n) / m)[:, None] * prof
    ab = np.stack([A.profiles(tf, rng, n), A.profiles(tf, rng, n)], 1)
    ab = ab - ab[..., :1]
    sgn = (-1.0) ** np.arange(1, N + 1)
    sm = fm + c_true(KM)[None, :, None] * (ab[:, 0, None, :] - sgn[None, :, None] * ab[:, 1, None, :])
    um = A.foh_np((-DIF * KM ** 2).astype(complex), sm.astype(complex), tf[1] - tf[0]).real
    f = np.einsum("im,nmt->nit", S, fm)
    u = np.einsum("im,nmt->nit", S, um)
    to = lambda a: torch.tensor(a[..., ::FINE])
    f, ab, u = to(f), to(ab), to(u)
    c = np.cumsum([0, *SPLIT])
    return {k: (f[p:q], ab[p:q], u[p:q]) for k, p, q in zip(("train", "vali", "test"), c[:-1], c[1:])}


class SineLayer(nn.Module):
    """Wavenumber pole layer on DST-I modes, with an optional boundary channel c(k) (a - (-1)^m b)."""

    def __init__(self, t, bc=True):
        super().__init__()
        self.core = WavenumberPoleLayer1d(1, t, re_max=0.5, re_leak=0.05, k_ref=K_REF)
        self.bc = bc
        self.c = nn.Parameter(torch.tensor([0.1, 0.0], dtype=torch.float64)) if bc else None
        self.register_buffer("k", torch.tensor(KM))
        self.register_buffer("sgn", torch.tensor((-1.0) ** np.arange(1, N + 1)))

    def forward(self, f, ab):
        fk = dst(f.to(torch.float64))                                     # (b, N, n_t)
        if self.bc:
            kk = self.k / K_REF
            cb = (self.c[0] * kk + self.c[1] * kk ** 3) * K_REF
            fk = fk + cb[None, :, None] * (ab[:, 0, None, :] - self.sgn[None, :, None] * ab[:, 1, None, :])
        mu, beta = self.core.poles(self.k)
        g, c = foh_kernel(mu, f.shape[-1], float(self.core.t[1] - self.core.t[0]))
        g, c = (beta[..., None] * g).sum(0), (beta[..., None] * c).sum(0)
        return idst(foh_response(g[None], c[None], fk.to(torch.complex128)).real)


class Periodic(nn.Module):
    def __init__(self, t):
        super().__init__()
        self.core = WavenumberPoleLayer1d(1, t, modes=33, re_max=0.5, re_leak=0.05, k_ref=K_REF)

    def forward(self, f, ab):
        return self.core(f)


def build(kind, t, device):
    return (SineLayer(t, bc=True) if kind == "sine_bc" else SineLayer(t, bc=False) if kind == "sine_nobc" else Periodic(t)).to(device)


def core_of(m):
    return m.core


def fit(kind, data, t, device):
    (ftr, abtr, ytr), (fva, abva, yva) = ((x.to(device) for x in data[s]) for s in ("train", "vali"))
    best = None
    for s in range(8):
        torch.manual_seed(s)
        m = build(kind, t, device)
        with torch.no_grad():
            m.core.mu_re.normal_(0, 3.0), m.core.mu_im.normal_(0, 3.0)
        opt = torch.optim.LBFGS(m.parameters(), max_iter=300, line_search_fn="strong_wolfe", tolerance_grad=1e-12,
                                tolerance_change=1e-15, history_size=50)

        def closure():
            opt.zero_grad()
            loss = A.rel(m(ftr, abtr), ytr)
            if torch.isfinite(loss):
                loss.backward()
            return loss

        try:
            opt.step(closure)
        except (RuntimeError, IndexError):
            pass
        with torch.no_grad():
            v = float(A.rel(m(fva, abva), yva))
        if math.isfinite(v) and (best is None or v < best[0]):
            best = (v, m)
    return best[1]


@torch.no_grad()
def oracle(data, t, device):
    m = SineLayer(t, bc=True).to(device)
    lam = torch.tensor(-DIF * KM ** 2, device=device)
    fte, abte, yte = (x.to(device) for x in data["test"])
    fk = dst(fte) + torch.tensor(c_true(KM), device=device)[None, :, None] * (abte[:, 0, None, :] - m.sgn[None, :, None] * abte[:, 1, None, :])
    g, c = foh_kernel(lam.to(torch.complex128), fte.shape[-1], float(t[1] - t[0]))
    y = idst(foh_response(g[None], c[None], fk.to(torch.complex128)).real)
    return float(A.rel(y, yte))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t = np.linspace(0, T, NT)
    out = []
    for seed in range(5):
        data = make_data(seed)
        floor = oracle(data, t, device)
        fte, abte, yte = (x.to(device) for x in data["test"])
        for kind in ("sine_bc", "sine_nobc", "periodic"):
            t0 = time.time()
            m = fit(kind, data, t, device)
            with torch.no_grad():
                test = float(A.rel(m(fte, abte), yte))
            r = dict(seed=seed, model=kind, oracle=floor, ratio=test / floor, seconds=round(time.time() - t0, 1))
            if kind == "sine_bc":
                ar = m.core.mu_re[0].detach().cpu().numpy()
                r.update(D=float(-ar[1] / K_REF ** 2), c1=float(m.c[0].detach()), c1_true=2 * DIF)
            out.append(r)
            print(seed, kind, f"ratio {r['ratio']:.2f} (oracle {floor:.2e})", {k: round(r[k], 5) for k in ("D", "c1") if k in r},
                  f"{r['seconds']} s", flush=True)
            OUT.parent.mkdir(exist_ok=True)
            OUT.write_text(json.dumps(out, indent=1))
    lines = ["# (D) Diffusion with time-varying Dirichlet boundaries: a boundary channel on sine modes", "",
             f"D = {DIF}, c1 = 2D/L = {2 * DIF}. Range over data seeds 0–4.", "",
             "| model | test / oracle | D̂ | ĉ₁ |", "|---|---|---|---|"]
    for kind in ("sine_bc", "sine_nobc", "periodic"):
        rs = [r for r in out if r["model"] == kind]
        g = lambda q: f"{min(r[q] for r in rs):.5f}–{max(r[q] for r in rs):.5f}" if q in rs[0] else "–"
        lines.append(f"| {kind} | {min(r['ratio'] for r in rs):.2f}–{max(r['ratio'] for r in rs):.2f} | {g('D')} | {g('c1')} |")
    (OUT.with_suffix(".md")).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
