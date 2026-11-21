"""(D) Robin boundaries: the basis itself depends on an unknown boundary coefficient, learned through its eigenvalues.

d3b covered Dirichlet, Neumann and mixed boundaries, whose bases are known in advance (sin / cos with k = m pi or
(m - 1/2) pi). A Robin boundary D u_x(1) + H u(1) = F1(t) (a flux with heat transfer H) has the basis sin(k_m x) with
k_m the roots of

    k cos k + beta sin k = 0,        beta = H / D,        k_m in ((m - 1/2) pi, m pi),

so the wavenumbers are not equally spaced, and they are unknown when H is. By Green's identity, with
N_m = int_0^1 sin^2(k_m x) dx = 1/2 - sin(2 k_m) / (4 k_m):

    u_m' = -D k_m^2 u_m + f_m + (D k_m / N_m) a(t) + (sin k_m / N_m) F1(t),

with a(t) the Dirichlet value at x = 0. The value channel is again proportional to k and the flux channel constant,
once the basis factors (1 / N_m and sin k_m) are applied.

**Model** (`robin`): the d3b basis layer, but the basis is built from a learned beta. k_m(beta) is solved by
bisection and made differentiable through the implicit function theorem (dk/dbeta = -sin k / f_k). The grid basis
Phi(beta), its inverse, N_m and sin k_m are recomputed in each forward pass. The free cubic channel laws and the
pole law (Re mu on {1, k^2, k^4}) are as in d3b. Each start draws beta0 ~ U[0.25, 1] (truth 2).

**Controls.** `fixed`: beta given (= 2), the ceiling. `dn`: beta = 0, i.e. d3b's Dirichlet–Neumann basis, the wrong
boundary type.

**System.** D = 0.02, H = 0.04 (beta = 2), N = 32 modes, grid x_i = i / N. Forcing on modes 1..12 (E1 profiles);
a and F1 E1 profiles starting at rest. Truth: the 32-mode Galerkin solution with the exact first-order hold on a 32x
finer time grid. T = 5, n_t = 201, splits 200 / 50 / 100, data seeds 0-4. Fit: the a7 joint fit with 8 starts.

**Checks before any fit.** Roots satisfy the equation to 1e-12; dk_1/dbeta from autograd 0.2036559 against
0.2036559 by finite differences; cond(Phi) = 1.4. Oracle: 5.6e-3 (seed 0), 5.9e-3 (seed 1).

**Fix before any result** (first launch crashed in seed 0 `robin` on a singular basis). beta is now
parameterized as softplus(rho) > 0, the physical range (H > 0): for -1 < beta < 0 the first root leaves the bracket
((1/2) pi, pi). A start whose basis is singular is skipped.
The second launch completed `robin` and `fixed`, but every `dn` start failed: at beta = 0 the root is the bracket's
left end, which the bisection missed (a bug in the control, fixed afterwards; `dn` rerun alone).

**Predictions** (written before running).

1. `robin`: in every seed, test error <= 1.2x the oracle, beta within 2 %, D within 1 %, and each channel law within
   2 % on the excited modes m <= 8 (d3b's lesson: laws are claimed only where the data excite them).
2. `fixed`: the same accuracy criteria (test, D, laws) in every seed.
3. `dn`: test error >= 3x the oracle in every seed.

    python experiments/D/d3_boundary/d3d_robin.py      # results/d3d_robin.md / .json
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
sys.path.insert(0, str(HERE))
import d3b_bc_types as B  # noqa: E402

A = B.A
DIF, H, N, T, NT, FINE, SPLIT = 0.02, 0.04, 32, 5.0, 201, 32, (200, 50, 100)
BETA = H / DIF
X = np.arange(1, N + 1) / N
OUT = HERE / "results" / "d3d_robin.json"


def roots_np(beta):
    """k_m, m = 1..N, roots of k cos k + beta sin k in ((m - 1/2) pi, m pi) (beta > -1/2 keeps one root there)."""
    m = np.arange(1, N + 1)
    if beta == 0:                                   # the root is the bracket's left end; bisection misses it
        return (m - 0.5) * np.pi
    lo, hi = (m - 0.5) * np.pi, m * np.pi - 1e-12
    f = lambda k: k * np.cos(k) + beta * np.sin(k)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        same = np.sign(f(mid)) == np.sign(f(lo))
        lo, hi = np.where(same, mid, lo), np.where(same, hi, mid)
    return 0.5 * (lo + hi)


def roots(beta):
    """Differentiable k_m(beta): bisection value plus the implicit-function gradient."""
    k0 = torch.tensor(roots_np(float(beta.detach())), dtype=torch.float64, device=beta.device)
    fk = torch.cos(k0) - k0 * torch.sin(k0) + beta.detach() * torch.cos(k0)
    return k0 - (k0 * torch.cos(k0) + beta * torch.sin(k0)) / fk


def basis(k, x):
    phi = torch.sin(x[:, None] * k[None])
    nm = 0.5 - torch.sin(2 * k) / (4 * k)
    return phi, nm


def true_laws(k):
    return np.stack([DIF * k, np.ones_like(k)])          # value channel D k, flux channel 1 (after sgn, 1 / N_m)


def make_data(seed):
    k = roots_np(BETA)
    phi = np.sin(np.outer(X, k))
    nm = 0.5 - np.sin(2 * k) / (4 * k)
    sgn = np.stack([np.ones(N), np.sin(k)])
    rng = np.random.default_rng(900 + seed)
    n = sum(SPLIT)
    tf = np.linspace(0, T, (NT - 1) * FINE + 1)
    fm = np.zeros((n, N, tf.size))
    for _ in range(4):
        prof = A.profiles(tf, rng, n)
        mm = rng.integers(1, 13, size=n)
        fm[np.arange(n), mm - 1] += (rng.normal(size=n) / mm)[:, None] * prof
    bc = np.stack([A.profiles(tf, rng, n), A.profiles(tf, rng, n)], 1)
    bc = bc - bc[..., :1]
    src = fm + np.einsum("jm,njt->nmt", true_laws(k) * sgn / nm, bc)
    um = A.foh_np((-DIF * k ** 2).astype(complex), src.astype(complex), tf[1] - tf[0]).real
    f, u = np.einsum("xm,nmt->nxt", phi, fm), np.einsum("xm,nmt->nxt", phi, um)
    to = lambda a: torch.tensor(a[..., ::FINE])
    f, bc, u = to(f), to(bc), to(u)
    c = np.cumsum([0, *SPLIT])
    return {s: (f[p:q], bc[p:q], u[p:q]) for s, p, q in zip(("train", "vali", "test"), c[:-1], c[1:])}


class RobinLayer(nn.Module):
    def __init__(self, t, beta0, learn=True):
        super().__init__()
        self.kref = float(np.pi * N)
        self.core = B.WavenumberPoleLayer1d(1, t, re_max=0.5, re_leak=0.05, k_ref=self.kref)
        self.c = nn.Parameter(torch.zeros(2, 4, dtype=torch.float64))
        # beta = softplus(rho) > 0 (H > 0: heat loss); for -1 < beta < 0 the first root leaves ((1/2) pi, pi)
        rho0 = math.log(math.expm1(beta0)) if beta0 > 0 else -50.0
        self.rho = nn.Parameter(torch.tensor(rho0, dtype=torch.float64), requires_grad=learn)
        self.register_buffer("zero", torch.tensor(beta0 == 0))
        self.register_buffer("x", torch.tensor(X, dtype=torch.float64))

    @property
    def beta(self):
        return torch.zeros_like(self.rho) if bool(self.zero) else torch.nn.functional.softplus(self.rho)

    def law(self, k):
        kk = k / self.kref
        return sum(self.c[:, p, None] * kk[None] ** p for p in range(4))

    def forward(self, f, bc, law=None):
        k = roots(self.beta)
        phi, nm = basis(k, self.x)
        sgn = torch.stack([torch.ones_like(k), torch.sin(k)])
        fk = torch.linalg.solve(phi, f.to(torch.float64))
        lw = self.law(k) if law is None else law(k)
        fk = fk + torch.einsum("jm,bjt->bmt", lw * sgn / nm, bc.to(torch.float64))
        mu, beta = self.core.poles(k)
        g, c = B.foh_kernel(mu, f.shape[-1], float(self.core.t[1] - self.core.t[0]))
        g, c = (beta[..., None] * g).sum(0), (beta[..., None] * c).sum(0)
        return torch.einsum("xm,bmt->bxt", phi, B.foh_response(g[None], c[None], fk.to(torch.complex128)).real)


def fit(kind, data, t, device):
    (ftr, btr, ytr), (fva, bva, yva) = ((z.to(device) for z in data[s]) for s in ("train", "vali"))
    best = None
    for s in range(8):
        torch.manual_seed(s)
        b0 = {"robin": float(np.random.default_rng(s).uniform(0.25, 1.0)), "fixed": BETA, "dn": 0.0}[kind]
        m = RobinLayer(t, b0, learn=kind == "robin").to(device)
        with torch.no_grad():
            m.core.mu_re.normal_(0, 3.0), m.core.mu_im.normal_(0, 3.0)
        opt = torch.optim.LBFGS([p for p in m.parameters() if p.requires_grad], max_iter=300,
                                line_search_fn="strong_wolfe", tolerance_grad=1e-12, tolerance_change=1e-15,
                                history_size=50)

        def closure():
            opt.zero_grad()
            loss = A.rel(m(ftr, btr), ytr)
            if torch.isfinite(loss):
                loss.backward()
            return loss

        try:
            opt.step(closure)
        except (RuntimeError, IndexError):
            pass
        with torch.no_grad():
            if kind == "robin" and not 0 < float(m.beta) < 50:
                continue
            try:
                v = float(A.rel(m(fva, bva), yva))
            except RuntimeError:                     # singular basis: start failed
                continue
        if math.isfinite(v) and (best is None or v < best[0]):
            best = (v, m)
    return None if best is None else best[1]


@torch.no_grad()
def oracle(data, t, device):
    m = RobinLayer(t, BETA, learn=False).to(device)
    core = m.core
    core.mu_re.zero_(), core.mu_im.zero_(), core.beta.zero_()
    core.mu_re[0, 1] = -DIF * m.kref ** 2
    core.beta[0, 0, 0] = 1.0
    fte, bte, yte = (z.to(device) for z in data["test"])
    law = lambda k: torch.stack([DIF * k, torch.ones_like(k)])
    return float(A.rel(m(fte, bte, law), yte))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t = np.linspace(0, T, NT)
    out = json.loads(OUT.read_text()) if OUT.exists() else []
    done = {(r["seed"], r["model"]) for r in out}
    for seed in range(5):
        data = make_data(seed)
        floor = oracle(data, t, device)
        print("seed", seed, "oracle", f"{floor:.2e}", flush=True)
        fte, bte, yte = (z.to(device) for z in data["test"])
        for kind in ("robin", "fixed", "dn"):
            if (seed, kind) in done:
                continue
            t0 = time.time()
            m = fit(kind, data, t, device)
            if m is None:
                out.append(dict(seed=seed, model=kind, oracle=floor, ratio=float("inf"), failed=True))
                OUT.write_text(json.dumps(out, indent=1))
                continue
            with torch.no_grad():
                test = float(A.rel(m(fte, bte), yte))
                k = roots(m.beta).cpu().numpy()
                lw = m.law(roots(m.beta)).cpu().numpy()
            tl = true_laws(k)
            r = dict(seed=seed, model=kind, oracle=floor, ratio=test / floor, beta=float(m.beta),
                     D=float(-m.core.mu_re[0, 1] / m.kref ** 2),
                     law_err8=[float(np.linalg.norm(lw[j, :8] - tl[j, :8]) / np.linalg.norm(tl[j, :8])) for j in range(2)],
                     law_err=[float(np.linalg.norm(lw[j] - tl[j]) / np.linalg.norm(tl[j])) for j in range(2)],
                     seconds=round(time.time() - t0, 1))
            out.append(r)
            print({q: (round(v, 5) if isinstance(v, float) else v) for q, v in r.items()}, flush=True)
            OUT.parent.mkdir(exist_ok=True)
            OUT.write_text(json.dumps(out, indent=1))
    lines = ["# (D) Robin boundary with a learned coefficient: basis from learned eigenvalues", "",
             f"D = {DIF}, H = {H}: beta = H / D = {BETA}. Range over data seeds 0–4.", "",
             "| model | test / oracle | β̂ | D̂ | value-law error m ≤ 8 | flux-law error m ≤ 8 | law errors, all modes |",
             "|---|---|---|---|---|---|---|"]
    for kind in ("robin", "fixed", "dn"):
        rs = [r for r in out if r["model"] == kind and not r.get("failed")]
        rg = lambda v, f: f"{min(v):{f}}–{max(v):{f}}"
        lines.append(f"| {kind} | {rg([r['ratio'] for r in rs], '.2f')} | {rg([r['beta'] for r in rs], '.4f')} | "
                     f"{rg([r['D'] for r in rs], '.5f')} | {rg([r['law_err8'][0] for r in rs], '.1e')} | "
                     f"{rg([r['law_err8'][1] for r in rs], '.1e')} | "
                     f"{rg([max(r['law_err']) for r in rs], '.1e')} |")
    (OUT.with_suffix(".md")).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
