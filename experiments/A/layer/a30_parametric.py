"""(A) Application 2, stage 3 (linear regime): mobility M, gradient energy kappa and composition phi0 as inputs.

Plan (`docs/proposal_A_wavenumber_pole_operator.qmd`, Section 6.2, stage 3): feed M, kappa and phi0 as operator
inputs and test extrapolation outside the training range. a29 (stage 1) fixed M and kappa and learned the law in
phi0. Here the same IC-path layer gets M and kappa as inputs and is tested on (M, kappa) pairs outside the training
grid. 1D, linear regime (early time), as a29.

**System.** u_t = M d_xx (u^3 - u - kappa d_xx u), periodic, n_x = 64, no forcing, solver as a29. Linearised:

    lambda(k; M, kappa, phi0) = M k^2 (1 - 3 phi0^2) - M kappa k^4.

**Data** (a29's windows: phi0 + perturbation on modes 1..16, eps log-uniform in [1e-4, 1e-3], T = 5, n_t = 201).
- Training grid: M in {0.05, 0.1}, kappa in {0.03, 0.06}, phi0 in {0, 0.2, 0.4, 0.6}; 15 windows per cell (240).
  Fastest growth M / (4 kappa) <= 0.83, so the perturbation stays small over T.
- Validation: 5 windows per training cell.
- Test, 20 windows per (point, phi0), phi0 in {0.15, 0.45, 0.7}:
  `interp` (0.075, 0.045); `hiM` (0.2, 0.08) (M 2x above the grid); `lowk` (0.025, 0.015) (kappa 2x below,
  k_c(0) = 8.2 against 4.1-5.8 on the grid); `hiMk` (0.15, 0.045).
- Data seeds 0-4. Errors as a29 (relative L2 of u - phi0).

**Models.** The IC path u_hat(k, t) = gamma(k) u0_hat(k) e^{mu t}, gamma on {1, q}, q = (k / 16)^2, phi0 from the
mean of u0; inputs m = M / 0.1, c = kappa / 0.04.
- `phys`: mu = m (a q + b q phi0^2) + m c d q^2 (the form of the linearisation; true a = 25.6, b = -76.8,
  d = -262.1). 3 reals.
- `bilin`: mu = sum over m^i c^j (i, j in {0, 1}) of (b1 q + b2 q phi0^2 + b3 q^2 + b4 q^2 phi0^2): a29's law
  with each coefficient bilinear in (m, c). 16 reals; contains the truth but does not know how M and kappa enter.
- `mlp`: mu = MLP(q, phi0^2, m, c), 2 hidden layers of 32 (tanh), no form. 1249 reals.
- `fixed`: a29's `joint` law, ignoring M and kappa. 4 reals.
- `oracle`: the linearisation.
Fit: 4 starts, each to convergence with the d3g rule; pick by validation. Starts: `phys`/`bilin`/`fixed`: the q
coefficient a1 log-uniform in [5, 50], the q^2 coefficient -10 a1, the rest 0; `mlp` torch seeds 0-3.

**Checks before any fit** (seed 0 unless stated). Max |u - phi0| 0.12 (train), <= 0.05 (test). Oracle error
1.3e-3 (train), 4.9e-4 to 4.4e-3 on the test cells. `phys` at the true coefficients reproduces the oracle and gives
k_c(0) exact to the grid step (1e-3). Start rule: with a29's starts (q^2 coefficient -a1) only 1 of 4 starts of
`phys` and `bilin` reached the oracle level; the others stopped at a training loss of 1 (prediction decayed to
zero), 2, or overflowed. With -10 a1 (checked on training loss only, seeds 0 and 1): every `phys` start reaches the
oracle's training loss; `bilin`'s best start reaches 1.00x `phys`'s training loss (seed 0) and 1.20x (seed 1), one
start in four overflowing; `fixed` stops at 0.33. No test errors were looked at.

**Derived** (`phys`, `bilin`): k_c(0) at each test point.

**Predictions** (written before running).

1. `phys`, every seed: test error <= 1.2x the oracle at every (point, phi0), and k_c(0) within 1 % at every point.
2. `bilin`, every seed: <= 1.2x the oracle on `interp` and <= 2x at every other (point, phi0).
3. `mlp`, every seed: <= 3x the oracle on `interp` (mean over phi0) and >= 5x on at least one extrapolation point
   (mean over phi0).
4. `fixed`, every seed: >= 10x the oracle at every extrapolation point (mean over phi0).

    python experiments/A/layer/a30_parametric.py      # results/a30_parametric.md / .json
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
import a29_spinodal as S  # noqa: E402

G = S.G
ch = S.ch
NX, T, NT, FINE, KM, KREF = S.NX, S.T, S.NT, S.FINE, S.KM, S.KREF
TRAIN_M, TRAIN_K, TRAIN_PHI = (0.05, 0.1), (0.03, 0.06), (0.0, 0.2, 0.4, 0.6)
POINTS = dict(interp=(0.075, 0.045), hiM=(0.2, 0.08), lowk=(0.025, 0.015), hiMk=(0.15, 0.045))
TEST_PHI = (0.15, 0.45, 0.7)
KINDS = ("phys", "bilin", "mlp", "fixed")
OUT = HERE / "results" / "a30_parametric.json"


def windows(rng, M, kappa, phis, n_each):
    K = NX // 2 + 1
    rows = []
    for p in phis:
        u0k = np.zeros((n_each, K), complex)
        u0k[:, 0] = p * NX
        eps = np.exp(rng.uniform(np.log(1e-4), np.log(1e-3), size=n_each))
        u0k[:, 1:KM + 1] = (rng.normal(size=(n_each, KM)) + 1j * rng.normal(size=(n_each, KM))) * NX / 2 * eps[:, None]
        fk = np.zeros((n_each, K, (NT - 1) * FINE + 1), complex)
        uk = ch.solve(fk, T / ((NT - 1) * FINE), M, kappa, NX, stride=FINE, u0k=u0k)
        rows.append(np.fft.irfft(uk, n=NX, axis=1))
    u = np.concatenate(rows)
    par = torch.tensor([[M / 0.1, kappa / 0.04]] * len(u))
    return torch.tensor(u[..., 0]), par, torch.tensor(u)


def cat(parts):
    return tuple(torch.cat(z) for z in zip(*parts))


def make_data(seed):
    rng = np.random.default_rng(1400 + seed)
    grid = [(M, k) for M in TRAIN_M for k in TRAIN_K]
    tr = cat([windows(rng, M, k, TRAIN_PHI, 15) for M, k in grid])
    va = cat([windows(rng, M, k, TRAIN_PHI, 5) for M, k in grid])
    te = {(n, p): windows(rng, *POINTS[n], (p,), 20) for n in POINTS for p in TEST_PHI}
    return dict(train=tr, vali=va, test=te)


class Param(nn.Module):
    def __init__(self, t, kind):
        super().__init__()
        self.kind = kind
        self.register_buffer("t", torch.tensor(t))
        self.register_buffer("q", (torch.arange(1, KM + 1, dtype=torch.float64) / KREF) ** 2)
        self.g = nn.Parameter(torch.tensor([1.0, 0.0], dtype=torch.float64))
        if kind == "mlp":
            self.net = nn.Sequential(nn.Linear(4, 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh(),
                                     nn.Linear(32, 1)).double()
        else:
            self.a = nn.Parameter(torch.zeros({"phys": 3, "bilin": 16, "fixed": 4}[kind], dtype=torch.float64))

    def init(self, st):
        g = np.random.default_rng(st)
        with torch.no_grad():
            if self.kind == "mlp":
                torch.manual_seed(st)
                for l in self.net:
                    if isinstance(l, nn.Linear):
                        l.reset_parameters()
                return
            a1 = math.exp(g.uniform(math.log(5), math.log(50)))
            self.a.zero_()
            self.a[0], self.a[2] = a1, -10 * a1     # bilin: the (i, j) = (0, 0) block; fixed: a29's layout

    def mu(self, phi0, par):
        p2, m, c, q = phi0[:, None] ** 2, par[:, :1], par[:, 1:], self.q
        if self.kind == "phys":
            a, b, d = self.a
            return m * (a * q + b * q * p2) + m * c * d * q ** 2
        shape = lambda w: w[0] * q + w[1] * q * p2 + w[2] * q ** 2 + w[3] * q ** 2 * p2
        if self.kind == "fixed":
            return shape(self.a)
        if self.kind == "bilin":
            w = self.a.view(4, 4)                   # rows: 1, m, c, m c
            return shape(w[0]) + m * shape(w[1]) + c * shape(w[2]) + m * c * shape(w[3])
        x = torch.stack(torch.broadcast_tensors(q.expand_as(p2 * q), p2 * torch.ones_like(q), m * torch.ones_like(q),
                                                c * torch.ones_like(q)), -1)
        return self.net(x)[..., 0] * 10.0

    def forward(self, u0, par):
        u0k = torch.fft.rfft(u0, dim=1)
        mu = self.mu(u0k[:, 0].real / NX, par)
        gam = self.g[0] + self.g[1] * self.q
        z = torch.zeros(u0.shape[0], NX // 2 + 1, len(self.t), dtype=torch.complex128, device=u0.device)
        z[:, 1:KM + 1] = (gam * u0k[:, 1:KM + 1])[..., None] * torch.exp(mu[..., None] * self.t)
        return torch.fft.irfft(z, n=NX, dim=1)


class Oracle(nn.Module):
    def __init__(self, t):
        super().__init__()
        self.register_buffer("t", torch.tensor(t))
        self.register_buffer("k", torch.arange(1, KM + 1, dtype=torch.float64))

    def forward(self, u0, par):
        u0k = torch.fft.rfft(u0, dim=1)
        p2, M, kap = (u0k[:, :1].real / NX) ** 2, 0.1 * par[:, :1], 0.04 * par[:, 1:]
        mu = M * self.k ** 2 * (1 - 3 * p2) - M * kap * self.k ** 4
        z = torch.zeros(u0.shape[0], NX // 2 + 1, len(self.t), dtype=torch.complex128, device=u0.device)
        z[:, 1:KM + 1] = u0k[:, 1:KM + 1, None] * torch.exp(mu[..., None] * self.t)
        return torch.fft.irfft(z, n=NX, dim=1)


def rel(m, d):
    u0, par, u = d
    w = u - u.mean(1, keepdim=True)
    p = m(u0, par)
    return ((p - w).flatten(1).norm(dim=1) / w.flatten(1).norm(dim=1)).mean()


def val(m, d):
    with torch.no_grad():
        v = float(rel(m, d))
    return v if math.isfinite(v) else float("inf")


def fit(kind, data, t, device):
    tr, va = ([z.to(device) for z in data[s]] for s in ("train", "vali"))
    best = None
    for st in range(4):
        m = Param(t, kind).to(device)
        m.init(st)
        iters, capped = G.converge(m, lambda: rel(m, tr), list(m.parameters()))
        v = val(m, va)
        if best is None or v < best[0]:
            best = (v, m, iters, capped)
    return best[1:]


@torch.no_grad()
def kc0(m, name, device):
    """k_c(0) from the learned law: the largest k (fine grid in [0.5, 16]) with mu > 0, phi0 = 0."""
    M, kap = POINTS[name]
    k = torch.linspace(0.5, 16, 15501, dtype=torch.float64, device=device)
    q = (k / KREF) ** 2
    saved = m.q
    m.q = q
    mu = m.mu(torch.zeros(1, dtype=torch.float64, device=device),
              torch.tensor([[M / 0.1, kap / 0.04]], device=device))[0]
    m.q = saved
    pos = (mu > 0).nonzero()
    return float(k[pos[-1]]) if len(pos) else float("nan")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t = np.linspace(0, T, NT)
    out = json.loads(OUT.read_text()) if OUT.exists() else []
    for seed in range(5):
        data = make_data(seed)
        te = {key: [z.to(device) for z in d] for key, d in data["test"].items()}
        orc = Oracle(t).to(device)
        floor = {key: val(orc, d) for key, d in te.items()}
        for kind in KINDS:
            if any(r["seed"] == seed and r["model"] == kind for r in out):
                continue
            t0 = time.time()
            m, iters, capped = fit(kind, data, t, device)
            ratio = {f"{n}/{p}": val(m, te[(n, p)]) / floor[(n, p)] for n, p in te}
            r = dict(model=kind, seed=seed, oracle={f"{n}/{p}": v for (n, p), v in floor.items()}, ratio=ratio,
                     kc0={n: kc0(m, n, device) for n in POINTS} if kind in ("phys", "bilin") else None,
                     a=m.a.detach().cpu().numpy().tolist() if kind != "mlp" else None,
                     iters=iters, capped=capped, seconds=round(time.time() - t0, 1))
            out.append(r)
            print({q: v for q, v in r.items() if q not in ("oracle",)}, flush=True)
            OUT.parent.mkdir(exist_ok=True)
            OUT.write_text(json.dumps(out, indent=1))
    kc_true = {n: math.sqrt(1 / POINTS[n][1]) for n in POINTS}
    rg = lambda xs, f: f"{min(xs):{f}}–{max(xs):{f}}"
    lines = ["# (A) Cahn–Hilliard linear regime with M, kappa, phi0 as inputs: extrapolation in (M, kappa)", "",
             "Test error / oracle, mean over φ0 ∈ {0.15, 0.45, 0.7}; range over data seeds 0–4. Points (M, κ): "
             + ", ".join(f"{n} {POINTS[n]}" for n in POINTS) + ".", "",
             "| model | " + " | ".join(POINTS) + " | worst (point, φ0) |", "|---|" + "---|" * len(POINTS) + "---|"]
    for kind in KINDS:
        rs = [r for r in out if r["model"] == kind]
        if not rs:
            continue
        cells = [rg([np.mean([r["ratio"][f"{n}/{p}"] for p in TEST_PHI]) for r in rs], ".2f") for n in POINTS]
        lines.append(f"| {kind} | " + " | ".join(cells) + f" | {max(max(r['ratio'].values()) for r in rs):.2f} |")
    lines += ["", "k_c(0) relative error: " + "; ".join(
        f"{kind}: " + ", ".join(f"{n} {rg([abs(r['kc0'][n] / kc_true[n] - 1) for r in out if r['model'] == kind], '.1e')}"
                                for n in POINTS)
        for kind in ("phys", "bilin") if any(r["model"] == kind for r in out)) + "."]
    (OUT.with_suffix(".md")).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
