"""(D) A variable coefficient: u_t = (D(x) u_x)_x + f on [0, 1] with u = 0 at both ends. The basis is no longer sines.

d3–d3e kept constant coefficients, where a known basis (sines, cosines, or sines with learned wavenumbers for Robin)
diagonalises the operator. With D(x) varying, the eigenfunctions of the Sturm–Liouville operator are not sines, and
a diagonal layer in the sine basis is the wrong model class. The question is whether the layer can learn the basis
as well as the poles.

**System.** D(x) = 0.02 (1 + 0.5 cos(pi x)) (0.01 to 0.03, not symmetric about x = 1/2, so all modes couple).
Galerkin in N = 32 sine modes phi_m = sqrt(2) sin(m pi x): c' = A c + f_hat with
A_mn = -int D phi_m' phi_n' dx (symmetric; Gauss–Legendre quadrature), so A = V Lambda V^T. Truth: exact
first-order hold in the eigenbasis on a 32x finer time grid, sampled on the interior grid x_i = i / 33 (DST-I, exact
for 32 modes). The truth and the models share the 32-mode space, so the only floor is the time axis. Forcing on
sine modes 1..12 with E1 profiles (amplitude ~ 1 / m). T = 5, n_t = 201, splits 200 / 50 / 100, data seeds 0-4.

**Models** (each mode a pole lambda_m = -exp(rho_m) < 0, free per mode; output Phi Q z, z_m' = lambda_m z_m +
(Q^T f_hat)_m, FOH time axis):
- `mix`: Q = exp(S - S^T), a learned orthogonal basis change (496 + 32 reals). Starts at the sine basis (S = 0 plus
  noise 1e-3).
- `diag`: Q = I, the sine basis (32 reals): the best constant-coefficient-style model.
- `oracle`: Q = V, lambda = Lambda.
Fit: L-BFGS (600 iterations) from 4 starts, lambda_m = -Dbar (m pi)^2 with Dbar log-uniform in [0.01, 0.04]; pick
by validation.

**Claims** are made on the excited eigenmodes m <= 8 (as in d3b): eigenvalues sorted by decay rate, eigenvectors
compared by |<v_m, v_hat_m>|.

**Checks before any fit** (seed 0). lambda_1..8 = -0.180, -0.712, -1.60, -2.84, -4.44, -6.39, -8.70, -11.4
(lambda_32 = -263). The eigenvectors are far from sines: |<v_m, e_m>| = 0.98, 0.93, 0.84, 0.73, 0.59, 0.44, 0.28,
0.13 for m = 1..8. Oracle test error 6.0e-3. The sine-diagonal model at the true diag(A) gives 0.147 (24x).

**Predictions** (written before running).

1. `mix`, in every seed: test error <= 1.2x the oracle, lambda_1..8 within 1 %, and |<v_m, v_hat_m>| >= 0.999 for
   m = 1..8.
2. `diag`: test error >= 3x the oracle in every seed.

    python experiments/D/d3_boundary/d3f_variable.py      # results/d3f_variable.md / .json
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
from lfo.models.wavenumber import foh_kernel, foh_response  # noqa: E402

N, D0, EPS, N_EXC, N_CLAIM = 32, 0.02, 0.5, 12, 8
X = np.arange(1, N + 1) / (N + 1)
M_IDX = np.arange(1, N + 1)
PHI = math.sqrt(2) * np.sin(np.outer(X, M_IDX) * math.pi)          # (grid, mode)
OUT = HERE / "results" / "d3f_variable.json"


def dcoef(x):
    return D0 * (1 + EPS * np.cos(math.pi * x))


def operator():
    xq, wq = np.polynomial.legendre.leggauss(2000)
    xq, wq = (xq + 1) / 2, wq / 2
    dphi = math.sqrt(2) * np.cos(np.outer(xq, M_IDX) * math.pi) * (M_IDX * math.pi)
    Amat = -(dphi * (dcoef(xq) * wq)[:, None]).T @ dphi
    lam, V = np.linalg.eigh(Amat)
    order = np.argsort(-lam)                                          # slowest first
    return Amat, lam[order], V[:, order]


def make_data(seed):
    Amat, lam, V = operator()
    rng = np.random.default_rng(1100 + seed)
    n = sum(A.SPLIT)
    tf = np.linspace(0, A.T_END, (A.N_T - 1) * A.FINE + 1)
    fh = np.zeros((n, N, tf.size))
    for m in range(1, N_EXC + 1):
        fh[:, m - 1] = rng.normal(size=n)[:, None] / m * A.profiles(tf, rng, n)
    g = np.einsum("mk,bmt->bkt", V, fh).astype(complex)                # eigen-coordinates of the forcing
    z = A.foh_np(lam.astype(complex), g, tf[1] - tf[0]).real
    c = np.einsum("mk,bkt->bmt", V, z)
    f = np.einsum("im,bmt->bit", PHI, fh[..., ::A.FINE])
    u = np.einsum("im,bmt->bit", PHI, c[..., ::A.FINE])
    s = np.cumsum([0, *A.SPLIT])
    t = lambda a: torch.tensor(a)
    return {k: (t(f[a:b]), t(u[a:b])) for k, a, b in zip(("train", "vali", "test"), s[:-1], s[1:])}


class VarLayer(nn.Module):
    def __init__(self, t, kind):
        super().__init__()
        self.kind, self.n_t, self.dt = kind, len(t), float(t[1] - t[0])
        self.register_buffer("phi", torch.tensor(PHI))
        self.register_buffer("phinv", torch.linalg.inv(torch.tensor(PHI)))
        self.rho = nn.Parameter(torch.zeros(N, dtype=torch.float64))
        self.S = nn.Parameter(torch.zeros(N, N, dtype=torch.float64)) if kind == "mix" else None
        self.register_buffer("Qfix", torch.eye(N, dtype=torch.float64))

    def Q(self):
        return torch.linalg.matrix_exp(self.S - self.S.T) if self.S is not None else self.Qfix

    def lam(self):
        return -torch.exp(self.rho)

    def forward(self, f):
        Q = self.Q()
        g = torch.einsum("mk,im,bit->bkt", Q, self.phinv.T, f)          # Q^T Phi^-1 f
        G, c = foh_kernel(self.lam().to(torch.complex128), self.n_t, self.dt)
        z = foh_response(G, c, g.to(torch.complex128)).real
        return torch.einsum("im,mk,bkt->bit", self.phi, Q, z)


def lbfgs(m, d, iters=600):
    opt = torch.optim.LBFGS(m.parameters(), max_iter=iters, line_search_fn="strong_wolfe", tolerance_grad=1e-12,
                            tolerance_change=1e-15, history_size=50)

    def closure():
        opt.zero_grad()
        loss = A.rel(m(d[0]), d[1])
        if torch.isfinite(loss):
            loss.backward()
        return loss

    try:
        opt.step(closure)
    except (RuntimeError, IndexError):
        pass


def val(m, d):
    with torch.no_grad():
        v = float(A.rel(m(d[0]), d[1]))
    return v if math.isfinite(v) else float("inf")


def fit(kind, data, t, device):
    tr, va = ([z.to(device) for z in data[s]] for s in ("train", "vali"))
    best = None
    for st in range(4):
        g = np.random.default_rng(st)
        m = VarLayer(t, kind).to(device)
        dbar = math.exp(g.uniform(math.log(0.01), math.log(0.04)))
        with torch.no_grad():
            m.rho.copy_(torch.tensor(np.log(dbar * (M_IDX * math.pi) ** 2)))
            if m.S is not None:
                m.S.copy_(torch.tensor(g.normal(0, 1e-3, size=(N, N))))
        lbfgs(m, tr)
        v = val(m, va)
        if best is None or v < best[0]:
            best = (v, m)
    return best[1]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t = np.linspace(0, A.T_END, A.N_T)
    _, lam, V = operator()
    out = json.loads(OUT.read_text()) if OUT.exists() else []
    for seed in range(5):
        data = make_data(seed)
        te = [z.to(device) for z in data["test"]]
        orc = VarLayer(t, "diag").to(device)
        with torch.no_grad():
            orc.rho.copy_(torch.tensor(np.log(-lam)))
            orc.Qfix.copy_(torch.tensor(V))
        floor = val(orc, te)
        for kind in ("mix", "diag"):
            if any(r["seed"] == seed and r["model"] == kind for r in out):
                continue
            t0 = time.time()
            m = fit(kind, data, t, device)
            test = val(m, te)
            with torch.no_grad():
                lh = m.lam().cpu().numpy()
                Q = m.Q().cpu().numpy()
            o = np.argsort(-lh)
            lh, Q = lh[o], Q[:, o]
            r = dict(model=kind, seed=seed, oracle=floor, ratio=test / floor,
                     lam_err8=[float(abs(lh[i] / lam[i] - 1)) for i in range(N_CLAIM)],
                     vec_dot8=[float(abs(Q[:, i] @ V[:, i])) for i in range(N_CLAIM)],
                     seconds=round(time.time() - t0, 1))
            out.append(r)
            print({q: (round(v, 5) if isinstance(v, float) else v) for q, v in r.items()}, flush=True)
            OUT.parent.mkdir(exist_ok=True)
            OUT.write_text(json.dumps(out, indent=1))
    lines = ["# (D) Variable coefficient D(x) = 0.02 (1 + 0.5 cos(pi x)), Dirichlet: learned basis against sines", "",
             "Range over data seeds 0–4. Eigenmodes sorted by decay rate; claims on m ≤ 8.", "",
             "| model | test / oracle | max eigenvalue error m ≤ 8 | min abs(<v_m, v̂_m>) m ≤ 8 |", "|---|---|---|---|"]
    for kind in ("mix", "diag"):
        rs = [r for r in out if r["model"] == kind]
        lines.append(f"| {kind} | {min(r['ratio'] for r in rs):.2f}–{max(r['ratio'] for r in rs):.2f} | "
                     f"{max(max(r['lam_err8']) for r in rs):.1e} | {min(min(r['vec_dot8']) for r in rs):.5f} |")
    lines += ["", f"Oracle test error {min(r['oracle'] for r in out):.2e}–{max(r['oracle'] for r in out):.2e}."]
    (OUT.with_suffix(".md")).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
