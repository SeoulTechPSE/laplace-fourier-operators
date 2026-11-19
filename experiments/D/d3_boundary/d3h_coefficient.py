"""(D) A variable coefficient learned as a function: D(x) on a cosine basis, the modal basis from its eigenproblem.

d3f learned the eigen-decomposition of u_t = (D(x) u_x)_x + f (Dirichlet) as a free orthogonal Q and free poles,
528 reals, but not D(x) itself. Here the parameter is D(x) = sum_{j<6} d_j cos(j pi x). The Galerkin operator
A(d) = sum_j d_j A_j (A_j[m, n] = -int cos(j pi x) phi_m' phi_n' dx) is linear in d, and the layer's basis and
poles come from its eigen-decomposition (torch.linalg.eigh, differentiable for distinct eigenvalues). 6 reals.

**Cases** (same system, grid, forcing, splits and data seeds as d3f, `d3f_variable.py`):
- `cos`: D(x) = 0.02 (1 + 0.5 cos pi x), in the span (= d3f's system).
- `exp`: D(x) = 0.01 * 3^x (0.01 to 0.03), not in the span. Its best 6-term cosine approximation has a
  relative L2 error of 1.06e-2 (computed before any fit).

**Checks before any fit** (seed 0). `cos` data are identical to d3f's. Oracle test error 6.0e-3 in both cases. The
layer at the true d (`cos`) reproduces the oracle to 1e-16. At the best cosine approximation (`exp`) it gives
6.8e-3 (1.12x).

**Fit.** 4 starts (d_0 log-uniform in [0.01, 0.04], the rest 0), each run to convergence with the d3g rule
(L-BFGS chunks of 300 until the training loss drops by less than a relative 1e-4; cap 6000), pick by validation.

**Predictions** (written before running).

1. `cos`, every seed: test error <= 1.2x the oracle, D(x) within 1 % (relative L2 on [0, 1]), lambda_1..8 within
   1 %.
2. `exp`, every seed: test error <= 1.2x the oracle and D(x) within 2 %.
3. Iterations to convergence: the median over runs is below d3f `mix`'s budget plus its post hoc continuation
   (2400 + 1800 on the selected start).

    python experiments/D/d3_boundary/d3h_coefficient.py      # results/d3h_coefficient.md / .json
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
import d3f_variable as V  # noqa: E402
import d3g_converged as G  # noqa: E402

A = V.A
N, J = V.N, 6
OUT = HERE / "results" / "d3h_coefficient.json"
XQ, WQ = np.polynomial.legendre.leggauss(2000)
XQ, WQ = (XQ + 1) / 2, WQ / 2
CASES = {"cos": lambda x: 0.02 * (1 + 0.5 * np.cos(math.pi * x)), "exp": lambda x: 0.01 * 3.0 ** x}


def stiffness_basis():
    dphi = math.sqrt(2) * np.cos(np.outer(XQ, V.M_IDX) * math.pi) * (V.M_IDX * math.pi)
    return np.stack([-(dphi * (np.cos(j * math.pi * XQ) * WQ)[:, None]).T @ dphi for j in range(J)])


def operator(dfun):
    dphi = math.sqrt(2) * np.cos(np.outer(XQ, V.M_IDX) * math.pi) * (V.M_IDX * math.pi)
    Am = -(dphi * (dfun(XQ) * WQ)[:, None]).T @ dphi
    lam, Vm = np.linalg.eigh(Am)
    o = np.argsort(-lam)
    return lam[o], Vm[:, o]


def best_cosine(dfun):
    C = np.cos(np.outer(XQ, np.arange(J)) * math.pi)
    w = np.sqrt(WQ)
    c = np.linalg.lstsq(C * w[:, None], dfun(XQ) * w, rcond=None)[0]
    return c, float(np.sqrt(np.sum(WQ * (C @ c - dfun(XQ)) ** 2) / np.sum(WQ * dfun(XQ) ** 2)))


def make_data(case, seed):
    lam, Vm = operator(CASES[case])
    rng = np.random.default_rng(1100 + seed)                         # d3f's data seeds: identical forcing
    n = sum(A.SPLIT)
    tf = np.linspace(0, A.T_END, (A.N_T - 1) * A.FINE + 1)
    fh = np.zeros((n, N, tf.size))
    for m in range(1, V.N_EXC + 1):
        fh[:, m - 1] = rng.normal(size=n)[:, None] / m * A.profiles(tf, rng, n)
    g = np.einsum("mk,bmt->bkt", Vm, fh).astype(complex)
    z = A.foh_np(lam.astype(complex), g, tf[1] - tf[0]).real
    c = np.einsum("mk,bkt->bmt", Vm, z)
    f = np.einsum("im,bmt->bit", V.PHI, fh[..., ::A.FINE])
    u = np.einsum("im,bmt->bit", V.PHI, c[..., ::A.FINE])
    s = np.cumsum([0, *A.SPLIT])
    to = lambda a: torch.tensor(a)
    return {k: (to(f[a:b]), to(u[a:b])) for k, a, b in zip(("train", "vali", "test"), s[:-1], s[1:])}, lam, Vm


class CoefLayer(V.VarLayer):
    """VarLayer whose Q and poles come from eigh of A(d) = sum_j d_j A_j."""

    def __init__(self, t):
        super().__init__(t, "diag")
        self.register_buffer("Aj", torch.tensor(stiffness_basis()))
        self.d = nn.Parameter(torch.zeros(J, dtype=torch.float64))
        del self.rho

    def eig(self):
        lam, Q = torch.linalg.eigh(torch.einsum("j,jmn->mn", self.d, self.Aj))
        return lam, Q

    def Q(self):
        return self.eig()[1]

    def lam(self):
        return self.eig()[0]

    def forward(self, f):
        lam, Q = self.eig()
        g = torch.einsum("mk,im,bit->bkt", Q, self.phinv.T, f)
        Gk, c = V.foh_kernel(lam.to(torch.complex128), self.n_t, self.dt)
        z = V.foh_response(Gk, c, g.to(torch.complex128)).real
        return torch.einsum("im,mk,bkt->bit", self.phi, Q, z)


def fit(data, t, device):
    tr, va = ([z.to(device) for z in data[s]] for s in ("train", "vali"))
    best = None
    for st in range(4):
        g = np.random.default_rng(st)
        m = CoefLayer(t).to(device)
        with torch.no_grad():
            m.d[0] = math.exp(g.uniform(math.log(0.01), math.log(0.04)))
        iters, capped = G.converge(m, lambda: A.rel(m(tr[0]), tr[1]), [m.d])
        v = V.val(m, va)
        if best is None or v < best[0]:
            best = (v, m, iters, capped)
    return best[1:]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t = np.linspace(0, A.T_END, A.N_T)
    C = np.cos(np.outer(XQ, np.arange(J)) * math.pi)
    out = json.loads(OUT.read_text()) if OUT.exists() else []
    for case in CASES:
        for seed in range(5):
            if any(r["case"] == case and r["seed"] == seed for r in out):
                continue
            data, lam, Vm = make_data(case, seed)
            te = [z.to(device) for z in data["test"]]
            orc = V.VarLayer(t, "diag").to(device)
            with torch.no_grad():
                orc.rho.copy_(torch.tensor(np.log(-lam)))
                orc.Qfix.copy_(torch.tensor(Vm))
            floor = V.val(orc, te)
            t0 = time.time()
            m, iters, capped = fit(data, t, device)
            with torch.no_grad():
                lh = m.lam().cpu().numpy()
                d = m.d.cpu().numpy()
            lh = lh[np.argsort(-lh)]
            Dt, Dh = CASES[case](XQ), C @ d
            r = dict(case=case, seed=seed, oracle=floor, ratio=V.val(m, te) / floor, d=d.tolist(),
                     D_err=float(np.sqrt(np.sum(WQ * (Dh - Dt) ** 2) / np.sum(WQ * Dt ** 2))),
                     lam_err8=[float(abs(lh[i] / lam[i] - 1)) for i in range(V.N_CLAIM)], iters=iters, capped=capped,
                     seconds=round(time.time() - t0, 1))
            out.append(r)
            print({q: (round(v, 6) if isinstance(v, float) else v) for q, v in r.items()}, flush=True)
            OUT.parent.mkdir(exist_ok=True)
            OUT.write_text(json.dumps(out, indent=1))
    lines = ["# (D) Variable coefficient learned as D(x) on 6 cosines; basis from the eigenproblem", "",
             f"Best 6-cosine approximation error of the `exp` truth: {best_cosine(CASES['exp'])[1]:.1e}.", "",
             "| case | test / oracle | D(x) error (rel. L2) | max eigenvalue error m ≤ 8 | iterations (selected start) |",
             "|---|---|---|---|---|"]
    for case in CASES:
        rs = [r for r in out if r["case"] == case]
        rg = lambda xs, f: f"{min(xs):{f}}–{max(xs):{f}}"
        lines.append(f"| {case} | {rg([r['ratio'] for r in rs], '.3f')} | {rg([r['D_err'] for r in rs], '.1e')} | "
                     f"{rg([max(r['lam_err8']) for r in rs], '.1e')} | {rg([r['iters'] for r in rs], 'd')} |")
    (OUT.with_suffix(".md")).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
