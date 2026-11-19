"""(A) Application 1, first step: delay identification on FOPDT / SOPDT (plan Section 5.1, Table "Process-dynamics").

**Systems** (SISO, from rest):
- FOPDT: G(s) = K e^{-theta s} / (tau s + 1), K = 2, tau = 1.5, theta = 1.37;
- SOPDT: G(s) = K e^{-theta s} / ((tau1 s + 1)(tau2 s + 1)), K = 1, tau1 = 2.0, tau2 = 0.5, theta = 0.83.
Neither delay is a multiple of the sample time.

**Data.** T = 20, dt = 0.05 (n_t = 401). Inputs from the E1 families (step, ramp, PRBS, smooth random) on a 32x finer
grid. Truth: the undelayed exact response on the fine grid (`lfo.data.lti.response`), shifted by theta with linear
interpolation on the fine grid (independent of the layer), then sampled. Splits 200 / 50 / 100; data seeds 0-4;
noise-free.

**Models.**
- `pdelay`: `lfo.models.PoleResidueDelay` with the true order (real stable poles, real residues, one shared delay).
  Joint fit: L-BFGS 300 iterations from 10 starts (initial theta in {0.2, 0.5, 1, 2, 4} x 2 pole draws); selection
  by validation error.
- `pole_only`: the same layer without delay (theta fixed at 0) and 4 complex poles (output 2 Re): the LNO-type
  representation, which must approximate e^{-theta s} by poles. Same fit protocol (10 starts).
- `arx`: discrete ARX, y_t = sum_{i<=na} a_i y_{t-i} + sum_{j<=nb} b_j u_{t-nk-j+1}, least squares, na = order,
  nb = order + 1. nk is chosen from 0..100 by validation error. theta_hat = nk dt; K and time constants from the
  discrete poles (mu = log z / dt).

**Oracle:** the true parameters in `PoleResidueDelay`. Measured before any fit (seed 0): 1.04e-2 (FOPDT), 9.7e-3
(SOPDT), set by the inputs' sub-sample features (steps and PRBS jumps between samples), as in a1.

**Predictions** (written before running).

1. `pdelay`: theta, K and every tau within 1 % in every data seed and both systems, with a test error <= 1.2x the
   oracle.
2. `pole_only`: test error >= 5x the oracle in every seed and both systems.
3. `arx`: theta_hat within one sample (0.05) of theta in every seed, with a test error above `pdelay`'s.

    python experiments/A/process/a20_fopdt.py      # results/a20_fopdt.md / .json
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "A" / "layer"))
import a1_dispersion as A  # noqa: E402
from lfo.data import lti  # noqa: E402
from lfo.models import PoleResidueDelay  # noqa: E402

T, NT, FINE, SPLIT = 20.0, 401, 32, (200, 50, 100)
DT = T / (NT - 1)
SYSTEMS = {"fopdt": dict(K=2.0, taus=(1.5,), theta=1.37), "sopdt": dict(K=1.0, taus=(2.0, 0.5), theta=0.83)}
THETA0 = (0.2, 0.5, 1.0, 2.0, 4.0)
OUT = HERE / "results" / "a20_fopdt.json"


def pole_residue(K, taus):
    """Poles and residues of K / prod(tau_i s + 1) (distinct taus)."""
    p = np.array([-1 / t for t in taus])
    r = np.array([K / np.prod([taus[i] * (p[j] - p[i]) for i in range(len(taus)) if i != j]) / taus[j]
                  if len(taus) > 1 else K / taus[0] for j in range(len(taus))])
    return p, r


def make_data(system, seed):
    s = SYSTEMS[system]
    rng = np.random.default_rng(seed)
    tf = np.linspace(0, T, (NT - 1) * FINE + 1)
    n = sum(SPLIT)
    u = A.profiles(tf, rng, n)
    p, r = pole_residue(s["K"], s["taus"])
    y0 = lti.response(tf, u, p.astype(complex), r.astype(complex))
    y = np.stack([np.interp(tf - s["theta"], tf, row, left=0.0) for row in y0])
    u, y = torch.tensor(u[:, ::FINE]), torch.tensor(y[:, ::FINE])
    c = np.cumsum([0, *SPLIT])
    return {k: (u[a:b], y[a:b]) for k, a, b in zip(("train", "vali", "test"), c[:-1], c[1:])}


def rel(a, b):
    return ((a - b).norm(dim=-1) / b.norm(dim=-1)).mean()


def lbfgs(m, data, device):
    (xtr, ytr), (xva, yva) = ((x.to(device) for x in data[s]) for s in ("train", "vali"))
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
        return float(rel(m(xva), yva))


def fit_layer(system, data, device, kind):
    order = len(SYSTEMS[system]["taus"])
    best = None
    for i, th0 in enumerate(THETA0):
        for draw in range(2):
            torch.manual_seed(10 * i + draw)
            if kind == "pdelay":
                m = PoleResidueDelay(order, DT, theta0=th0).to(device)
            else:
                m = PoleResidueDelay(4, DT, real=False, theta0=1e-6).to(device)
                m.theta_raw.requires_grad_(False)
                with torch.no_grad():
                    m.mu_im.mul_(2.0)
            v = lbfgs(m, data, device)
            if math.isfinite(v) and (best is None or v < best[0]):
                best = (v, m)
    return best[1]


def layer_params(m):
    with torch.no_grad():
        mu, beta, theta = m.poles()
    p, r = mu.real.cpu().numpy(), beta.real.cpu().numpy()
    taus = sorted((-1 / p).tolist(), reverse=True)
    return dict(theta=float(theta[0]), K=float(-(r / p).sum()), taus=taus)


def arx_fit(system, data):
    order = len(SYSTEMS[system]["taus"])
    na, nb = order, order + 1

    def design(u, y, nk):
        u, y = u.numpy(), y.numpy()
        start = max(na, nk + nb)
        rows, tgt = [], []
        for t in range(start, y.shape[1]):
            rows.append(np.concatenate([y[:, t - na:t][:, ::-1], u[:, t - nk - nb + 1:t - nk + 1][:, ::-1]], 1))
            tgt.append(y[:, t])
        return np.concatenate(rows), np.concatenate(tgt)

    def simulate(theta_ab, nk, u):
        a, b = theta_ab[:na], theta_ab[na:]
        u = u.numpy()
        y = np.zeros_like(u)
        for t in range(u.shape[1]):
            acc = sum(a[i] * y[:, t - 1 - i] for i in range(na) if t - 1 - i >= 0)
            acc = acc + sum(b[j] * u[:, t - nk - j] for j in range(nb) if t - nk - j >= 0)
            y[:, t] = acc
        return torch.tensor(y)

    best = None
    for nk in range(0, 101):
        X, z = design(*data["train"], nk)
        coef, *_ = np.linalg.lstsq(X, z, rcond=None)
        v = float(rel(simulate(coef, nk, data["vali"][0]), data["vali"][1]))
        if best is None or v < best[0]:
            best = (v, nk, coef)
    _, nk, coef = best
    test = float(rel(simulate(coef, nk, data["test"][0]), data["test"][1]))
    zp = np.roots(np.concatenate([[1.0], -coef[:na]]))
    mu = np.log(zp.astype(complex)) / DT
    K = float(coef[na:].sum() / (1 - coef[:na].sum()))
    return dict(theta=nk * DT, K=K, taus=sorted((-1 / mu.real).tolist(), reverse=True), test=test)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = []
    for system, s in SYSTEMS.items():
        for seed in range(5):
            data = make_data(system, seed)
            orc = PoleResidueDelay(len(s["taus"]), DT).to(device)
            p, r = pole_residue(s["K"], s["taus"])
            with torch.no_grad():
                orc.mu_re.copy_(torch.log(torch.expm1(torch.tensor(-p))))
                orc.beta_re.copy_(torch.tensor(r))
                orc.theta_raw.fill_(math.log(math.expm1(s["theta"])))
                floor = float(rel(orc(data["test"][0].to(device)), data["test"][1].to(device)))
            for kind in ("pdelay", "pole_only", "arx"):
                t0 = time.time()
                if kind == "arx":
                    res = arx_fit(system, data)
                else:
                    m = fit_layer(system, data, device, kind)
                    with torch.no_grad():
                        res = dict(test=float(rel(m(data["test"][0].to(device)), data["test"][1].to(device))))
                    if kind == "pdelay":
                        res.update(layer_params(m))
                r_ = dict(system=system, seed=seed, kind=kind, oracle=floor, ratio=res["test"] / floor, **res,
                          seconds=round(time.time() - t0, 1))
                if "theta" in res:
                    r_["err"] = dict(theta=abs(res["theta"] - s["theta"]) / s["theta"], K=abs(res["K"] - s["K"]) / s["K"],
                                     taus=[abs(a - b) / b for a, b in zip(res["taus"], s["taus"])])
                out.append(r_)
                print(system, seed, kind, f"test {res['test']:.2e} ({r_['ratio']:.2f}x oracle {floor:.1e})",
                      {k: (round(v, 4) if isinstance(v, float) else [round(x, 4) for x in v]) for k, v in res.items()
                       if k in ("theta", "K", "taus")}, f"{r_['seconds']} s", flush=True)
                OUT.parent.mkdir(exist_ok=True)
                OUT.write_text(json.dumps(out, indent=1))
    lines = ["# (A) Delay identification on FOPDT / SOPDT", "",
             "FOPDT: K 2, τ 1.5, θ 1.37. SOPDT: K 1, τ 2.0 / 0.5, θ 0.83. dt 0.05, T 20, noise-free, data seeds 0–4. "
             "Ranges over seeds.", "",
             "| system | model | test / oracle | θ̂ | K̂ | τ̂ |", "|---|---|---|---|---|---|"]
    for system in SYSTEMS:
        for kind in ("pdelay", "pole_only", "arx"):
            rs = [r for r in out if r["system"] == system and r["kind"] == kind]
            rg = lambda v, f=".4f": f"{min(v):{f}}–{max(v):{f}}"
            row = f"| {system} | {kind} | {rg([r['ratio'] for r in rs], '.2f')} |"
            if "theta" in rs[0]:
                row += f" {rg([r['theta'] for r in rs])} | {rg([r['K'] for r in rs])} | " + "; ".join(
                    rg([r['taus'][i] for r in rs]) for i in range(len(rs[0]['taus']))) + " |"
            else:
                row += " – | – | – |"
            lines.append(row)
    (OUT.with_suffix(".md")).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
