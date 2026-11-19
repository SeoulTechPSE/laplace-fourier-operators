"""(A) Application 1: a nonlinear CSTR. Do probes of deep operators recover the linearised poles at a steady state?

**System** (dimensionless exothermic CSTR, Uppal-Ray-Poore form):

    x1' = -x1 + Da (1 - x1) exp(x2 / (1 + x2 / gamma))
    x2' = -x2 + B Da (1 - x1) exp(x2 / (1 + x2 / gamma)) - beta (x2 - u)

Da = 0.072, B = 8, beta = 0.3, gamma = 20. Three steady states at u = 0. We use the upper, stable one
(x1 = 0.7646, x2 = 4.7050), whose Jacobian has the complex pair mu = -0.7694 +- 0.9597 i. Input: the coolant-
temperature deviation u(t). Output: the reactor-temperature deviation y = x2 - x2_s. Large inputs can extinguish
the reactor (a jump towards the lower steady state).

**Data.** T = 20, dt = 0.05 (n_t = 401); RK4 on a 32x finer grid, input linear between fine samples. Inputs are
E1-family profiles times s, with s log-uniform in [0.015, 0.5]. Splits 200 / 50 / 100. Measured before training
(`--describe`): the part of y not explained by the linearisation is 1 % / 7 % / 44 % at the 10th / 50th / 90th
percentile, up to 86 %.

**Models** (3 seeds; relative L2; Adam with cosine decay, 6000 steps, batch 25):
- `wpo`: `WavenumberPoleOperator1d` on n_x = 1 (time only; k = 0), width 16, 4 layers, 4 poles, Im mu constant
  (`im_powers=(0,)`), stable; lr 3e-3 with pole ratio 30.
- `fno`: neuralop FNO on t (n_modes 64, domain padding 0.25), width chosen for the parameter count closest to the
  WPO's; lr 1e-3.
- `linear`: the best linear model, `PoleResidueDelay` with one complex pole pair and no delay. L-BFGS from 10 starts,
  selection by validation.

**Probe.** Small-amplitude inputs (64 E1 profiles, eps = 0.005), odd part (G(eps u) - G(-eps u)) / 2. One complex
pole pair is fitted, y = 2 Re(b x_mu), with b by least squares, by grid search plus Nelder-Mead. The solver probed
the same way is the method check.

**Method check, measured before training:** the solver's probe gives -0.7672 +- 0.9568 i (0.3 % from the
Jacobian), residual 5e-3. Parameters: WPO 18,689; FNO 17,746 (closest width).

**Predictions** (written before running).

1. `wpo`: probe pole within 5 % of the Jacobian pair in at least 2 of 3 seeds.
2. `linear`: its pole is more than 10 % off (it absorbs the nonlinear part as extra damping or frequency).
3. `wpo` mean test error at or below the FNO's.

    python experiments/A/process/a23_cstr.py [--describe]      # results/a23_cstr.md / .json
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import fsolve, minimize

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "layer"))
sys.path.insert(0, str(HERE))
import a1_dispersion as A  # noqa: E402
from lfo.models import PoleResidueDelay, WavenumberPoleOperator1d  # noqa: E402
from lfo.models.wavenumber import foh_kernel, foh_response  # noqa: E402

DA, BB, BETA, GAM = 0.072, 8.0, 0.3, 20.0
T, NT, FINE, SPLIT, S_RANGE, EPS = 20.0, 401, 32, (200, 50, 100), (0.015, 0.5), 0.005
DT = T / (NT - 1)
SEEDS, STEPS, BATCH = (0, 1, 2), 6000, 25
WPO_KW = dict(width=16, n_layers=4, n_poles=4, modes=1, stable=True, im_powers=(0,), mu_degree=(1, 1), k_ref=1.0)
OUT = HERE / "results" / "a23_cstr.json"
CKPT = HERE.parents[2] / "runs" / "A" / "a23"


def rate(x1, x2):
    return DA * (1 - x1) * np.exp(x2 / (1 + x2 / GAM))


X1S, X2S = fsolve(lambda X: [-X[0] + rate(*X), -X[1] + BB * rate(*X) - BETA * X[1]], [0.7646, 4.705])
_e = np.exp(X2S / (1 + X2S / GAM))
_de = _e / (1 + X2S / GAM) ** 2
JAC = np.array([[-1 - DA * _e, DA * (1 - X1S) * _de], [-BB * DA * _e, -1 + BB * DA * (1 - X1S) * _de - BETA]])
MU_TRUE = complex(sorted(np.linalg.eigvals(JAC), key=lambda z: -z.imag)[0])


def simulate(u_fine, dtf):
    n, nt = u_fine.shape
    X = np.tile([X1S, X2S], (n, 1))
    out = np.zeros((n, nt))
    f = lambda X, uu: np.stack([-X[:, 0] + rate(X[:, 0], X[:, 1]),
                                -X[:, 1] + BB * rate(X[:, 0], X[:, 1]) - BETA * (X[:, 1] - uu)], 1)
    for j in range(nt - 1):
        u0, u1 = u_fine[:, j], u_fine[:, j + 1]
        um = 0.5 * (u0 + u1)
        k1 = f(X, u0)
        k2 = f(X + dtf / 2 * k1, um)
        k3 = f(X + dtf / 2 * k2, um)
        k4 = f(X + dtf * k3, u1)
        X = X + dtf / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        out[:, j + 1] = X[:, 1] - X2S
    return out


def make_data():
    rng = np.random.default_rng(0)
    n = sum(SPLIT)
    tf = np.linspace(0, T, (NT - 1) * FINE + 1)
    s = np.exp(rng.uniform(*np.log(S_RANGE), size=n))
    u = A.profiles(tf, rng, n) * s[:, None]
    y = simulate(u, tf[1] - tf[0])
    lin = simulate(u * 1e-4, tf[1] - tf[0]) * 1e4
    u, y, lin = (torch.tensor(a[:, ::FINE]) for a in (u, y, lin))
    c = np.cumsum([0, *SPLIT])
    d = {k: (u[a:b], y[a:b]) for k, a, b in zip(("train", "vali", "test"), c[:-1], c[1:])}
    d["nonlinear"] = ((y - lin).norm(dim=1) / y.norm(dim=1)).numpy()
    return d


def probe_inputs():
    rng = np.random.default_rng(2)
    tf = np.linspace(0, T, (NT - 1) * FINE + 1)
    return A.profiles(tf, rng, 64) * EPS, tf[1] - tf[0]


def fit_pair(u, y):
    """One complex pole pair, y = 2 Re(b x_mu), b by least squares; returns (mu, relative residual)."""
    u = torch.as_tensor(u, dtype=torch.float64)
    y = torch.as_tensor(y, dtype=torch.float64)

    def resid(mu):
        g, c = foh_kernel(torch.tensor([mu], dtype=torch.complex128), NT, DT)
        x = foh_response(g, c, u.to(torch.complex128))                    # (B, n_t)
        Am = torch.stack([2 * x.real, -2 * x.imag], -1).reshape(-1, 2)
        sol = torch.linalg.lstsq(Am, y.reshape(-1, 1)).solution
        return float((Am @ sol - y.reshape(-1, 1)).norm() / y.norm())

    best = min(((resid(complex(-re, im)), complex(-re, im)) for re in np.logspace(-2, 1, 40) for im in np.linspace(0, 5, 41)),
               key=lambda z: z[0])
    o = minimize(lambda q: resid(complex(*q)), [best[1].real, best[1].imag], method="Nelder-Mead",
                 options=dict(xatol=1e-10, fatol=1e-14, maxiter=3000))
    return complex(*o.x), float(o.fun)


def odd_probe(G, uf, dtf):
    up, yp = G(uf)
    _, ym = G(-uf)
    return fit_pair(up, 0.5 * (yp - ym))


class FNO1(torch.nn.Module):
    def __init__(self, width):
        super().__init__()
        from neuralop.models import FNO
        self.net = FNO(n_modes=(64,), in_channels=1, out_channels=1, hidden_channels=width, n_layers=4, domain_padding=0.25)

    def forward(self, u):
        return self.net(u.float()[:, None])[:, 0].to(u.dtype)


class WPO0(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = WavenumberPoleOperator1d(np.linspace(0, T, NT), **WPO_KW)

    def forward(self, u):
        return self.net(u[:, None, :])[:, 0]


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def build(kind, seed, device):
    torch.manual_seed(seed)
    if kind == "wpo":
        return WPO0().to(device)
    target = n_params(WPO0())
    w = min(range(4, 64), key=lambda w: abs(n_params(FNO1(w)) - target))
    torch.manual_seed(seed)
    return FNO1(w).to(device)


def rel(a, b):
    return ((a - b).norm(dim=-1) / b.norm(dim=-1)).mean()


def train(m, data, device, lr, plr=1.0):
    xtr, ytr = data["train"]
    is_pole = lambda n: n.endswith("mu_re") or n.endswith("mu_im")
    groups = [dict(params=[p for n, p in m.named_parameters() if not is_pole(n)], lr=lr)]
    pole = [p for n, p in m.named_parameters() if is_pole(n)]
    if pole:
        groups.append(dict(params=pole, lr=lr * plr))
    opt = torch.optim.Adam(groups)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS)
    g = torch.Generator().manual_seed(0)
    for _ in range(STEPS):
        idx = torch.randint(0, xtr.shape[0], (BATCH,), generator=g)
        opt.zero_grad()
        loss = rel(m(xtr[idx].to(device)), ytr[idx].to(device))
        loss.backward()
        opt.step()
        sch.step()


def fit_linear(data, device):
    (xtr, ytr), (xva, yva) = ((x.to(device) for x in data[s]) for s in ("train", "vali"))
    best = None
    for s in range(10):
        torch.manual_seed(s)
        m = PoleResidueDelay(1, DT, real=False, theta0=1e-6).to(device)
        m.theta_raw.requires_grad_(False)
        opt = torch.optim.LBFGS([q for q in m.parameters() if q.requires_grad], max_iter=300, line_search_fn="strong_wolfe",
                                tolerance_grad=1e-12, tolerance_change=1e-15, history_size=50)

        def closure():
            opt.zero_grad()
            loss = rel(m(xtr), ytr)
            if torch.isfinite(loss):
                loss.backward()
            return loss

        try:
            opt.step(closure)
        except (RuntimeError, IndexError):
            pass
        with torch.no_grad():
            v = float(rel(m(xva), yva))
        if math.isfinite(v) and (best is None or v < best[0]):
            best = (v, m)
    return best[1]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = make_data()
    nl = data["nonlinear"]
    uf, dtf = probe_inputs()
    solver = odd_probe(lambda x: (torch.tensor(x[:, ::FINE]), torch.tensor(simulate(x, dtf)[:, ::FINE])), uf, dtf)
    print(f"mu_true {MU_TRUE:.4f}", "nonlinear fraction p10/50/90/max", np.round(np.percentile(nl, [10, 50, 90, 100]), 3),
          f"solver probe {solver[0]:.4f} (err {abs(solver[0] - MU_TRUE) / abs(MU_TRUE):.2e}, resid {solver[1]:.1e})", flush=True)
    if "--describe" in sys.argv:
        return
    CKPT.mkdir(parents=True, exist_ok=True)
    out = dict(mu_true=[MU_TRUE.real, MU_TRUE.imag], solver_probe=[solver[0].real, solver[0].imag], runs=[])
    ufc = torch.tensor(uf[:, ::FINE])
    for kind, lr, plr in (("wpo", 3e-3, 30.0), ("fno", 1e-3, 1.0)):
        for seed in SEEDS:
            t0 = time.time()
            m = build(kind, seed, device)
            train(m, data, device, lr, plr)
            m.eval()
            with torch.no_grad():
                test = float(rel(m(data["test"][0].to(device)), data["test"][1].to(device)))
                G = lambda x: (torch.tensor(x[:, ::FINE]), m(torch.tensor(x[:, ::FINE]).to(device)).cpu().double())
                mu, res = odd_probe(G, uf, dtf)
            r = dict(model=kind, seed=seed, params=n_params(m), test=test, probe=[mu.real, mu.imag], probe_resid=res,
                     probe_err=abs(mu - MU_TRUE) / abs(MU_TRUE), seconds=round(time.time() - t0, 1))
            out["runs"].append(r)
            print(kind, seed, f"params {r['params']} test {test:.3e} probe {mu:.4f} err {r['probe_err']:.2e} resid {res:.1e}",
                  f"{r['seconds']} s", flush=True)
            OUT.parent.mkdir(exist_ok=True)
            OUT.write_text(json.dumps(out, indent=1))
    lin = fit_linear(data, device)
    with torch.no_grad():
        mu_l = complex(lin.poles()[0][0].cpu())
        test_l = float(rel(lin(data["test"][0].to(device)), data["test"][1].to(device)))
    out["linear"] = dict(test=test_l, pole=[mu_l.real, mu_l.imag], err=abs(mu_l - MU_TRUE) / abs(MU_TRUE))
    print("linear", f"test {test_l:.3e} pole {mu_l:.4f} err {out['linear']['err']:.2e}", flush=True)
    OUT.write_text(json.dumps(out, indent=1))
    lines = ["# (A) Nonlinear CSTR (upper steady state): probe poles against the Jacobian", "",
             f"Jacobian pair: {MU_TRUE.real:.4f} ± {MU_TRUE.imag:.4f} i. Solver probe: {solver[0].real:.4f} ± {abs(solver[0].imag):.4f} i "
             f"(error {abs(solver[0] - MU_TRUE) / abs(MU_TRUE):.1e}).", "",
             "| model | seed | params | test | probe pole | probe error | probe residual |", "|---|---|---|---|---|---|---|"]
    for r in out["runs"]:
        lines.append(f"| {r['model']} | {r['seed']} | {r['params']} | {r['test']:.3e} | {r['probe'][0]:.4f} ± {abs(r['probe'][1]):.4f} i | "
                     f"{r['probe_err']:.1e} | {r['probe_resid']:.1e} |")
    L = out["linear"]
    lines.append(f"| best linear (1 pair) | – | – | {L['test']:.3e} | {L['pole'][0]:.4f} ± {abs(L['pole'][1]):.4f} i | {L['err']:.1e} | – |")
    (OUT.with_suffix(".md")).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
