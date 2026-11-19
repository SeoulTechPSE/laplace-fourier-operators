"""(D) Boundary types: does the boundary channel learn the right law in k for Dirichlet, Neumann and mixed boundaries?

d3 (`d3_dirichlet.py`) showed that Dirichlet values enter the sine modes with c(k) = 2 D k / L. By Green's identity,
a boundary *value* (Dirichlet) enters mode m through the basis derivative phi_m'(boundary), which is proportional to
k. A boundary *flux* (Neumann) enters through phi_m(boundary), which does not depend on k. Here the channel
coefficient is a free cubic c_j(k) = c_j0 + c_j1 k + c_j2 k^2 + c_j3 k^3 per boundary j, and the question is whether
each channel learns its own law.

**Cases** (u_t = D u_xx + f on [0, 1], D = 0.02, n = 32 modes and grid points):
- `DD`: values a(t), b(t). Basis sin(k x), k = m pi, m = 1..32. Mode forcing (2 / L) D k (a - (-1)^m b).
- `NN`: influxes F0 = -D u_x(0), F1 = D u_x(1). Basis cos(k x), k = m pi, m = 0..31. Mode forcing
  w_m (2 / L)(F0 + (-1)^m F1), with w_0 = 1/2.
- `DN`: value a(t) at 0 and influx F1 at 1. Basis sin(k x), k = (m - 1/2) pi, m = 1..32. Mode forcing
  (2 / L)(D k a + (-1)^(m+1) F1).

The sign pattern and w_m come from the basis (phi_m at the boundary) and are given to the model. The k-law is
learned. Truth: the 32-mode Galerkin solution with the exact first-order hold on a 32x finer time grid, sampled on
grid points where the basis matrix is invertible (the model projects with its inverse). Forcing on modes 1..12 with
E1 profiles; boundary signals E1 profiles starting at rest. T = 5, n_t = 201, splits 200 / 50 / 100, data seeds 0-4.

**Models** (Re mu on {1, k^2, k^4}, Im mu free, k_ref = 32 pi, leaky cap; the a7 joint fit with 8 starts):
`bc` (both channels) and `nobc` (forcing only).

**Oracle, measured before any fit** (seed 0): DD 1.02e-2, NN 3.9e-3, DN 4.4e-3. In NN the k = 0 mode is an
integrator, since the mean is driven by the net influx.

**Predictions** (written before running).

1. `bc`: in every case and seed, test error <= 1.2x the oracle, D within 1 %, and each learned c_j(k) within 2 %
   (relative L2 over the modes) of its true law. Dirichlet channels are proportional to k; Neumann channels are
   constant.
2. `nobc`: test error >= 3x the oracle in every case and seed.

    python experiments/D/d3_boundary/d3b_bc_types.py      # results/d3b_bc_types.md / .json
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

DIF, N, T, NT, FINE, SPLIT, L = 0.02, 32, 5.0, 201, 32, (200, 50, 100), 1.0
OUT = HERE / "results" / "d3b_bc_types.json"


def case(name):
    """(k (N,), basis matrix Phi (grid x modes), true channel laws (2, N) including sign pattern and w_m,
    sign patterns (2, N) and weights w (N,) given to the model, and a callable for the true k-laws (2, N))."""
    m = np.arange(1, N + 1)
    if name == "DD":
        k, x = m * np.pi, np.arange(1, N + 1) / (N + 1)
        phi = np.sin(np.outer(x, k))
        sgn, w = np.stack([np.ones(N), -(-1.0) ** m]), np.ones(N)
        law = np.stack([2 / L * DIF * k, 2 / L * DIF * k])
    elif name == "NN":
        k, x = (m - 1) * np.pi, (np.arange(N) + 0.5) / N
        phi = np.cos(np.outer(x, k))
        sgn, w = np.stack([np.ones(N), (-1.0) ** (m - 1)]), np.where(m == 1, 0.5, 1.0)
        law = np.stack([2 / L * np.ones(N), 2 / L * np.ones(N)])
    else:                                                   # DN
        k, x = (m - 0.5) * np.pi, np.arange(1, N + 1) / N
        phi = np.sin(np.outer(x, k))
        sgn, w = np.stack([np.ones(N), (-1.0) ** (m + 1)]), np.ones(N)
        law = np.stack([2 / L * DIF * k, 2 / L * np.ones(N)])
    return k, phi, sgn, w, law


def make_data(name, seed):
    k, phi, sgn, w, law = case(name)
    rng = np.random.default_rng(600 + seed + 10 * ["DD", "NN", "DN"].index(name))
    n = sum(SPLIT)
    tf = np.linspace(0, T, (NT - 1) * FINE + 1)
    fm = np.zeros((n, N, tf.size))
    for _ in range(4):
        prof = A.profiles(tf, rng, n)
        mm = rng.integers(1, 13, size=n)
        fm[np.arange(n), mm - 1] += (rng.normal(size=n) / mm)[:, None] * prof
    bc = np.stack([A.profiles(tf, rng, n), A.profiles(tf, rng, n)], 1)
    bc = bc - bc[..., :1]
    src = fm + np.einsum("jm,njt->nmt", law * sgn * w, bc)
    um = A.foh_np((-DIF * k ** 2).astype(complex), src.astype(complex), tf[1] - tf[0]).real
    f, u = np.einsum("xm,nmt->nxt", phi, fm), np.einsum("xm,nmt->nxt", phi, um)
    to = lambda a: torch.tensor(a[..., ::FINE])
    f, bc, u = to(f), to(bc), to(u)
    c = np.cumsum([0, *SPLIT])
    return {s: (f[p:q], bc[p:q], u[p:q]) for s, p, q in zip(("train", "vali", "test"), c[:-1], c[1:])}


class BasisLayer(nn.Module):
    def __init__(self, name, t, bc=True):
        super().__init__()
        k, phi, sgn, w, _ = case(name)
        self.kref = float(np.pi * N)
        self.core = WavenumberPoleLayer1d(1, t, re_max=0.5, re_leak=0.05, k_ref=self.kref)
        self.bc = bc
        self.c = nn.Parameter(torch.zeros(2, 4, dtype=torch.float64)) if bc else None
        for nm, v in (("k", k), ("phi", phi), ("phinv", np.linalg.inv(phi)), ("sgn", sgn), ("w", w)):
            self.register_buffer(nm, torch.tensor(v, dtype=torch.float64))

    def law(self):
        kk = self.k / self.kref
        return sum(self.c[:, p, None] * kk[None] ** p for p in range(4))       # (2, N)

    def forward(self, f, bc):
        fk = torch.einsum("mx,bxt->bmt", self.phinv, f.to(torch.float64))
        if self.bc:
            fk = fk + torch.einsum("jm,bjt->bmt", self.law() * self.sgn * self.w, bc.to(torch.float64))
        mu, beta = self.core.poles(self.k)
        g, c = foh_kernel(mu, f.shape[-1], float(self.core.t[1] - self.core.t[0]))
        g, c = (beta[..., None] * g).sum(0), (beta[..., None] * c).sum(0)
        return torch.einsum("xm,bmt->bxt", self.phi, foh_response(g[None], c[None], fk.to(torch.complex128)).real)


def fit(name, bc, data, t, device):
    (ftr, btr, ytr), (fva, bva, yva) = ((x.to(device) for x in data[s]) for s in ("train", "vali"))
    best = None
    for s in range(8):
        torch.manual_seed(s)
        m = BasisLayer(name, t, bc).to(device)
        with torch.no_grad():
            m.core.mu_re.normal_(0, 3.0), m.core.mu_im.normal_(0, 3.0)
        opt = torch.optim.LBFGS(m.parameters(), max_iter=300, line_search_fn="strong_wolfe", tolerance_grad=1e-12,
                                tolerance_change=1e-15, history_size=50)

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
            v = float(A.rel(m(fva, bva), yva))
        if math.isfinite(v) and (best is None or v < best[0]):
            best = (v, m)
    return best[1]


@torch.no_grad()
def oracle(name, data, t, device):
    m = BasisLayer(name, t, True).to(device)
    _, _, _, _, law = case(name)
    m.core.mu_re.zero_(), m.core.mu_im.zero_(), m.core.beta.zero_()
    m.core.mu_re[0, 1] = -DIF * m.kref ** 2
    m.core.beta[0, 0, 0] = 1.0
    fte, bte, yte = (x.to(device) for x in data["test"])
    fk = torch.einsum("mx,bxt->bmt", m.phinv, fte) + torch.einsum("jm,bjt->bmt", torch.tensor(law, device=device) * m.sgn * m.w, bte)
    mu = m.core.mu(m.k)
    g, c = foh_kernel(mu, fte.shape[-1], float(t[1] - t[0]))
    y = torch.einsum("xm,bmt->bxt", m.phi, foh_response(g, c, fk.to(torch.complex128)).real)
    return float(A.rel(y, yte))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t = np.linspace(0, T, NT)
    out = []
    for name in ("DD", "NN", "DN"):
        law_true = case(name)[4]
        for seed in range(5):
            data = make_data(name, seed)
            floor = oracle(name, data, t, device)
            fte, bte, yte = (x.to(device) for x in data["test"])
            for bc in (True, False):
                t0 = time.time()
                m = fit(name, bc, data, t, device)
                with torch.no_grad():
                    test = float(A.rel(m(fte, bte), yte))
                r = dict(case=name, seed=seed, model="bc" if bc else "nobc", oracle=floor, ratio=test / floor,
                         seconds=round(time.time() - t0, 1))
                if bc:
                    lw = m.law().detach().cpu().numpy()
                    r.update(D=float(-m.core.mu_re[0, 1].detach() / m.kref ** 2),
                             law_err=[float(np.linalg.norm(lw[j] - law_true[j]) / np.linalg.norm(law_true[j])) for j in range(2)],
                             coef=m.c.detach().cpu().numpy().tolist())
                out.append(r)
                print(name, seed, r["model"], f"ratio {r['ratio']:.2f} (oracle {floor:.1e})",
                      {k: (round(v, 5) if isinstance(v, float) else [round(x, 4) for x in v]) for k, v in r.items() if k in ("D", "law_err")},
                      f"{r['seconds']} s", flush=True)
                OUT.parent.mkdir(exist_ok=True)
                OUT.write_text(json.dumps(out, indent=1))
    lines = ["# (D) Boundary types: learned boundary-channel laws in k", "",
             f"D = {DIF}. True laws: Dirichlet channel 2Dk/L (∝ k), Neumann flux channel 2/L (constant). Range over data seeds 0–4.", "",
             "| case | model | test / oracle | D̂ | channel 0 law error | channel 1 law error |", "|---|---|---|---|---|---|"]
    for name in ("DD", "NN", "DN"):
        for model in ("bc", "nobc"):
            rs = [r for r in out if r["case"] == name and r["model"] == model]
            rg = lambda v, f: f"{min(v):{f}}–{max(v):{f}}"
            row = f"| {name} | {model} | {rg([r['ratio'] for r in rs], '.2f')} |"
            row += (f" {rg([r['D'] for r in rs], '.5f')} | {rg([r['law_err'][0] for r in rs], '.1e')} | {rg([r['law_err'][1] for r in rs], '.1e')} |"
                    if model == "bc" else " – | – | – |")
            lines.append(row)
    (OUT.with_suffix(".md")).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
