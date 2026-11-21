"""(A) Application 2, stage 1: the linear regime of spinodal decomposition across mean compositions.

Plan (`docs/proposal_A_wavenumber_pole_operator.qmd`, Section 6.2, stage 1): train on early-time data while varying
the mean composition phi0 inside and outside the spinodal, and compare the learned mu(k) with the analytical
lambda(k), with emphasis on k_c and k_max. 1D here (the plan's 2D solver is the next step); the D2 initial-condition
path carries the data, since spinodal decomposition is free evolution from a perturbed uniform state.

**System** (`lfo.data.cahn_hilliard`, `u0k`; 954e509): u_t = M d_xx (u^3 - u - kappa d_xx u), M = 0.1,
kappa = 0.04, periodic, n_x = 64, no forcing. Linearized about phi0:

    lambda(k; phi0) = M k^2 (1 - 3 phi0^2) - M kappa k^4,   k_c = sqrt((1 - 3 phi0^2) / kappa),   k_max = k_c / sqrt 2,

so the spinodal is |phi0| < phi_s = 1 / sqrt 3 = 0.5774 (k_c(0) = 5, k_max(0) = 3.54).

**Data.** Initial field phi0 + a perturbation on modes 1..16, each mode complex normal times eps, eps log-uniform
in [1e-4, 1e-3]. T = 5, n_t = 201, truth from the solver on a 32x finer grid.
- Training compositions phi0 in {0, 0.1, ..., 0.7}, 25 windows each (200).
- Validation: 50 windows at the training compositions.
- Test (held out): phi0 in {0.15, 0.35, 0.55, 0.65, 0.8}, 20 windows each. 0.55 is inside the spinodal with only
  k = 1 unstable; 0.65 and 0.8 are outside, 0.8 beyond the training range.
- Data seeds 0-4.
Errors are relative L2 of the fluctuation u - phi0 (the mean is conserved and known from u0).

**Models.** The IC path alone: u_hat(k, t) = gamma(k) u0_hat(k) e^{mu(k, phi0) t}, k = 1..16, with phi0 read from
the mean of u0. Laws in q = (k / 16)^2; gamma on {1, q} (true 1, 0).
- `joint`: mu = a10 q + a11 q phi0^2 + a20 q^2 + a21 q^2 phi0^2 (true a10 = 25.6, a11 = -76.8, a20 = -262.1,
  a21 = 0). 6 reals.
- `stable`: the same with mu passed through -softplus(-mu) <= 0 (the LNO's stability constraint).
- `nophi`: mu = a10 q + a20 q^2, with no composition dependence.
- `oracle`: the true linearization (gamma = 1).
Fit: 4 starts (a10 log-uniform in [5, 50], a20 = -a10, the rest 0), each run to convergence with the d3g rule
(`experiments/D/d3_boundary/d3g_converged.py`); pick by validation.

**Derived.** phi_s = sqrt(-a10 / a11); k_c(phi0) = 16 sqrt(-(a10 + a11 phi0^2) / (a20 + a21 phi0^2)),
k_max = k_c / sqrt 2.

**Checks before any fit** (seed 0). The oracle (the linearization) leaves 1.7e-3 on the training set and
2.8e-3, 2.6e-3, 1.3e-3, 5.4e-4, 8.2e-4 on the held-out compositions (the nonlinear residual). Max |u - phi0| is 0.045
(phi0 = 0.15), so the data stay in the linear regime.

**Predictions** (written before running).

1. `joint`, every seed: held-out test error <= 1.2x the oracle's, phi_s within 1 %, and k_c within 1 % at
   phi0 = 0, 0.3 and 0.5.
2. `stable`: >= 10x the oracle on the held-out compositions inside the spinodal (0.15, 0.35), every seed.
3. `nophi`: >= 10x the oracle on the whole held-out set, every seed.

    python experiments/A/layer/a29_spinodal.py      # results/a29_spinodal.md / .json
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
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "D" / "d3_boundary"))
import d3g_converged as G  # noqa: E402
from lfo.data import cahn_hilliard as ch  # noqa: E402

MOB, KAPPA, NX, T, NT, FINE, KM, KREF = 0.1, 0.04, 64, 5.0, 201, 32, 16, 16.0
TRAIN_PHI = tuple(round(0.1 * i, 1) for i in range(8))
TEST_PHI = (0.15, 0.35, 0.55, 0.65, 0.8)
PHI_S = 1 / math.sqrt(3)
OUT = HERE / "results" / "a29_spinodal.json"
TRUE_A = dict(a10=MOB * KREF ** 2, a11=-3 * MOB * KREF ** 2, a20=-MOB * KAPPA * KREF ** 4, a21=0.0)


def kc_true(phi0):
    v = (1 - 3 * phi0 ** 2) / KAPPA
    return math.sqrt(v) if v > 0 else float("nan")


def windows(rng, phis, n_each):
    K = NX // 2 + 1
    rows = []
    for p in phis:
        u0k = np.zeros((n_each, K), complex)
        u0k[:, 0] = p * NX
        eps = np.exp(rng.uniform(np.log(1e-4), np.log(1e-3), size=n_each))
        u0k[:, 1:KM + 1] = (rng.normal(size=(n_each, KM)) + 1j * rng.normal(size=(n_each, KM))) * NX / 2 * eps[:, None]
        fk = np.zeros((n_each, K, (NT - 1) * FINE + 1), complex)
        uk = ch.solve(fk, T / ((NT - 1) * FINE), MOB, KAPPA, NX, stride=FINE, u0k=u0k)
        rows.append(np.fft.irfft(uk, n=NX, axis=1))
    u = np.concatenate(rows)
    return torch.tensor(u[..., 0]), torch.tensor(u)


def make_data(seed):
    rng = np.random.default_rng(1300 + seed)
    tr = windows(rng, TRAIN_PHI, 25)
    va = windows(rng, TRAIN_PHI, 7)
    te = {p: windows(rng, (p,), 20) for p in TEST_PHI}
    return dict(train=tr, vali=va, test=te)


class Spinodal(nn.Module):
    def __init__(self, t, kind):
        super().__init__()
        self.kind = kind
        self.register_buffer("t", torch.tensor(t))
        self.register_buffer("q", (torch.arange(1, KM + 1, dtype=torch.float64) / KREF) ** 2)
        self.a = nn.Parameter(torch.zeros(4, dtype=torch.float64))            # a10, a11, a20, a21
        self.g = nn.Parameter(torch.tensor([1.0, 0.0], dtype=torch.float64))

    def mu(self, phi0):
        a10, a11, a20, a21 = self.a
        p2 = phi0[:, None] ** 2 if self.kind != "nophi" else torch.zeros_like(phi0)[:, None]
        m = a10 * self.q + a11 * self.q * p2 + a20 * self.q ** 2 + a21 * self.q ** 2 * p2
        return -F.softplus(-m) if self.kind == "stable" else m

    def forward(self, u0):
        u0k = torch.fft.rfft(u0, dim=1)
        phi0 = u0k[:, 0].real / NX
        mu = self.mu(phi0)                                                      # (b, KM)
        gam = self.g[0] + self.g[1] * self.q
        z = torch.zeros(u0.shape[0], NX // 2 + 1, len(self.t), dtype=torch.complex128, device=u0.device)
        z[:, 1:KM + 1] = (gam * u0k[:, 1:KM + 1])[..., None] * torch.exp(mu[..., None] * self.t)
        return torch.fft.irfft(z, n=NX, dim=1)                                  # fluctuation (mean removed)

    def set_true(self):
        with torch.no_grad():
            self.a.copy_(torch.tensor([TRUE_A[k] for k in ("a10", "a11", "a20", "a21")]))
            self.g.copy_(torch.tensor([1.0, 0.0]))


def rel(m, d):
    u0, u = d
    w = u - u.mean(1, keepdim=True)
    p = m(u0)
    return ((p - w).flatten(1).norm(dim=1) / w.flatten(1).norm(dim=1)).mean()


def val(m, d):
    with torch.no_grad():
        v = float(rel(m, d))
    return v if math.isfinite(v) else float("inf")


def fit(kind, data, t, device):
    tr, va = ([z.to(device) for z in data[s]] for s in ("train", "vali"))
    best = None
    for st in range(4):
        g = np.random.default_rng(st)
        m = Spinodal(t, kind).to(device)
        a10 = math.exp(g.uniform(math.log(5), math.log(50)))
        with torch.no_grad():
            m.a.copy_(torch.tensor([a10, 0.0, -a10, 0.0]))
        iters, capped = G.converge(m, lambda: rel(m, tr), list(m.parameters()))
        v = val(m, va)
        if best is None or v < best[0]:
            best = (v, m, iters, capped)
    return best[1:]


def derived(a):
    a10, a11, a20, a21 = a
    phis = math.sqrt(-a10 / a11) if a10 * a11 < 0 else float("nan")
    kc = {}
    for p in (0.0, 0.3, 0.5):
        v = -(a10 + a11 * p ** 2) / (a20 + a21 * p ** 2)
        kc[str(p)] = KREF * math.sqrt(v) if v > 0 else float("nan")
    return phis, kc


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t = np.linspace(0, T, NT)
    out = json.loads(OUT.read_text()) if OUT.exists() else []
    for seed in range(5):
        data = make_data(seed)
        te = {p: [z.to(device) for z in d] for p, d in data["test"].items()}
        orc = Spinodal(t, "joint").to(device)
        orc.set_true()
        floor = {p: val(orc, d) for p, d in te.items()}
        for kind in ("joint", "stable", "nophi"):
            if any(r["seed"] == seed and r["model"] == kind for r in out):
                continue
            t0 = time.time()
            m, iters, capped = fit(kind, data, t, device)
            errs = {p: val(m, d) for p, d in te.items()}
            a = m.a.detach().cpu().numpy().tolist()
            phis, kc = derived(a)
            r = dict(model=kind, seed=seed, oracle={str(p): v for p, v in floor.items()},
                     test={str(p): v for p, v in errs.items()}, ratio={str(p): errs[p] / floor[p] for p in TEST_PHI},
                     a=a, g=m.g.detach().cpu().numpy().tolist(), phi_s=phis, kc=kc, iters=iters, capped=capped,
                     seconds=round(time.time() - t0, 1))
            out.append(r)
            print({q: v for q, v in r.items() if q not in ("oracle", "test")}, flush=True)
            OUT.parent.mkdir(exist_ok=True)
            OUT.write_text(json.dumps(out, indent=1))
    lines = ["# (A) Spinodal decomposition, linear regime across mean compositions (1D Cahn–Hilliard)", "",
             f"Truth: phi_s = {PHI_S:.4f}; k_c(0, 0.3, 0.5) = {kc_true(0):.3f}, {kc_true(0.3):.3f}, {kc_true(0.5):.3f}. "
             "Range over data seeds 0–4.", "",
             "| model | " + " | ".join(f"test / oracle, φ0 = {p}" for p in TEST_PHI) + " | φ̂_s | k̂_c(0), k̂_c(0.3), k̂_c(0.5) |",
             "|---|" + "---|" * len(TEST_PHI) + "---|---|"]
    rg = lambda xs, f: f"{min(xs):{f}}–{max(xs):{f}}"
    for kind in ("joint", "stable", "nophi"):
        rs = [r for r in out if r["model"] == kind]
        if not rs:
            continue
        cells = [rg([r["ratio"][str(p)] for r in rs], ".2f") for p in TEST_PHI]
        lines.append(f"| {kind} | " + " | ".join(cells) + f" | {rg([r['phi_s'] for r in rs], '.4f')} | "
                     + ", ".join(rg([r["kc"][p] for r in rs], ".3f") for p in ("0.0", "0.3", "0.5")) + " |")
    o = [r for r in out if r["model"] == "joint"]
    if o:
        lines += ["", "Oracle (linearization) error by φ0: " + ", ".join(
            f"{p}: {rg([r['oracle'][str(p)] for r in o], '.1e')}" for p in TEST_PHI) + "."]
    (OUT.with_suffix(".md")).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
