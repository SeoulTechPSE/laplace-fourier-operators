"""(A) Application 1: CSTR identification from closed-loop data with measurement noise.

Plant: the a23 CSTR at its upper steady state (Jacobian pair -0.7694 +- 0.9597 i). Four data regimes, 350 signals
each (200 / 50 / 100), data seed 0:

- `open`, `open_noisy`: u = E1 profiles x s, s log-uniform in [0.015, 0.5] (a23);
- `closed`, `closed_noisy`: a PI controller (Kc = 1, Ki = 0.5) drives the coolant temperature, u = Kc e + Ki int e,
  with e = r - y_meas. The set-point r is the E1 profiles x s. In closed loop the noise enters u through the
  feedback, so u is correlated with the output noise.

Noise: white measurement noise on y at the sample rate (held between samples). Its standard deviation is 2 % of
the median rms of y in the noise-free open-loop data. The recorded pairs are (u, y_meas) at dt = 0.05. Each model
is tested on a clean, open-loop test set (u -> y of the plant), so the test error measures the plant model. The RK4
solver holds u over each fine step (dt / 32), so in closed loop u is only approximately linear between samples.
This is the same for every model.

**Models** (a23 settings): `wpo` (time-only WPO, 3 seeds, its probe pole); `linear` (one complex pair, output-error
L-BFGS, 10 starts); `arx` (ARX(2, 2), equation error, least squares; its discrete poles mapped by log z / dt).

**Predictions** (written before running).

1. `wpo`: probe pole within 2 % of the Jacobian pair in every regime, in at least 2 of 3 seeds.
2. `linear` (output error): pole within 2 % in every regime. With white measurement noise, the output-error model
   is consistent, even in closed loop.
3. `arx` (equation error): pole within 2 % in the noise-free regimes and more than 10 % off in `closed_noisy`.

    python experiments/A/process/a26_cstr_closed_loop.py      # results/a26_cstr_closed_loop.md / .json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import a23_cstr as C  # noqa: E402

KC, KI, NOISE = 1.0, 0.5, 0.02
OUT = HERE / "results" / "a26_cstr_closed_loop.json"


def plant_f(X, uu):
    r = C.rate(X[:, 0], X[:, 1])
    return np.stack([-X[:, 0] + r, -X[:, 1] + C.BB * r - C.BETA * (X[:, 1] - uu)], 1)


def run(sig_fine, dtf, closed, noise):
    """sig_fine: u (open) or r (closed) on the fine grid; noise: (n, NT) at the coarse samples. Returns u, y_meas, y."""
    n, nt = sig_fine.shape
    X = np.tile([C.X1S, C.X2S], (n, 1))
    I = np.zeros(n)
    u = np.zeros((n, nt))
    y = np.zeros((n, nt))
    for j in range(nt - 1):
        nz = noise[:, j // C.FINE]
        if closed:
            e = sig_fine[:, j] - (X[:, 1] - C.X2S + nz)
            uu = KC * e + KI * I
            I = I + dtf * e
        else:
            uu = sig_fine[:, j]
        u[:, j] = uu
        k1 = plant_f(X, uu)
        k2 = plant_f(X + dtf / 2 * k1, uu)
        k3 = plant_f(X + dtf / 2 * k2, uu)
        k4 = plant_f(X + dtf * k3, uu)
        X = X + dtf / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        y[:, j + 1] = X[:, 1] - C.X2S
    u[:, -1] = u[:, -2] if closed else sig_fine[:, -1]
    us, ys = u[:, ::C.FINE], y[:, ::C.FINE]
    return us, ys + noise, ys


def make_regimes():
    rng = np.random.default_rng(0)
    n = sum(C.SPLIT)
    tf = np.linspace(0, C.T, (C.NT - 1) * C.FINE + 1)
    dtf = tf[1] - tf[0]
    s = np.exp(rng.uniform(*np.log(C.S_RANGE), size=n))
    sig = C.A.profiles(tf, rng, n) * s[:, None]
    zero = np.zeros((n, C.NT))
    _, _, y_clean = run(sig, dtf, False, zero)
    sigma = NOISE * float(np.median(np.sqrt((y_clean ** 2).mean(1))))
    noise = np.random.default_rng(5).normal(size=(n, C.NT)) * sigma
    test_sig = C.A.profiles(tf, np.random.default_rng(7), C.SPLIT[2]) * np.exp(
        np.random.default_rng(8).uniform(*np.log(C.S_RANGE), size=C.SPLIT[2]))[:, None]
    tu, _, ty = run(test_sig, dtf, False, np.zeros((C.SPLIT[2], C.NT)))
    test = (torch.tensor(tu), torch.tensor(ty))
    regimes = {}
    for name, closed, nz in (("open", False, zero), ("open_noisy", False, noise), ("closed", True, zero), ("closed_noisy", True, noise)):
        u, ym, _ = run(sig, dtf, closed, nz)
        u, ym = torch.tensor(u), torch.tensor(ym)
        regimes[name] = dict(train=(u[:200], ym[:200]), vali=(u[200:250], ym[200:250]), test=test)
    return regimes, sigma


def arx_pole(data):
    u, y = (a.numpy() for a in data["train"])
    rows, tgt = [], []
    for t in range(2, y.shape[1]):
        rows.append(np.stack([y[:, t - 1], y[:, t - 2], u[:, t - 1], u[:, t - 2]], 1))
        tgt.append(y[:, t])
    coef, *_ = np.linalg.lstsq(np.concatenate(rows), np.concatenate(tgt), rcond=None)
    z = np.roots([1.0, -coef[0], -coef[1]])
    mu = np.log(z.astype(complex)) / C.DT
    return complex(sorted(mu, key=lambda m: -m.imag)[0])


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    regimes, sigma = make_regimes()
    uf, dtf = C.probe_inputs()
    err = lambda mu: abs(mu - C.MU_TRUE) / abs(C.MU_TRUE)
    out = dict(sigma=sigma, runs=[])
    print(f"noise sigma {sigma:.4f}", flush=True)
    for name, data in regimes.items():
        for seed in C.SEEDS:
            t0 = time.time()
            m = C.build("wpo", seed, device)
            C.train(m, data, device, 3e-3, 30.0)
            m.eval()
            with torch.no_grad():
                test = float(C.rel(m(data["test"][0].to(device)), data["test"][1].to(device)))
                G = lambda x: (torch.tensor(x[:, ::C.FINE]), m(torch.tensor(x[:, ::C.FINE]).to(device)).cpu().double())
                mu, res = C.odd_probe(G, uf, dtf)
            r = dict(regime=name, model="wpo", seed=seed, test=test, pole=[mu.real, mu.imag], err=err(mu), seconds=round(time.time() - t0, 1))
            out["runs"].append(r)
            print(name, "wpo", seed, f"test {test:.3e} pole {mu:.4f} err {r['err']:.2e}", flush=True)
            OUT.parent.mkdir(exist_ok=True)
            OUT.write_text(json.dumps(out, indent=1))
        lin = C.fit_linear(data, device)
        with torch.no_grad():
            mu_l = complex(lin.poles()[0][0].cpu())
            test_l = float(C.rel(lin(data["test"][0].to(device)), data["test"][1].to(device)))
        mu_a = arx_pole(data)
        out["runs"].append(dict(regime=name, model="linear", test=test_l, pole=[mu_l.real, mu_l.imag], err=err(mu_l)))
        out["runs"].append(dict(regime=name, model="arx", pole=[mu_a.real, mu_a.imag], err=err(mu_a)))
        print(name, f"linear pole {mu_l:.4f} err {err(mu_l):.2e} test {test_l:.3e}; arx pole {mu_a:.4f} err {err(mu_a):.2e}", flush=True)
        OUT.write_text(json.dumps(out, indent=1))
    lines = ["# (A) CSTR from closed-loop data with measurement noise", "",
             f"Jacobian pair {C.MU_TRUE.real:.4f} ± {C.MU_TRUE.imag:.4f} i. PI Kc {KC}, Ki {KI}. Noise σ {sigma:.4f} (2 % of the median rms of y). "
             "Test: clean open-loop data.", "",
             "| regime | model | pole (seeds) | pole error | clean test error |", "|---|---|---|---|---|"]
    for name in regimes:
        for model in ("wpo", "linear", "arx"):
            rs = [r for r in out["runs"] if r["regime"] == name and r["model"] == model]
            poles = "; ".join(f"{r['pole'][0]:.3f} ± {abs(r['pole'][1]):.3f} i" for r in rs)
            tests = ", ".join(f"{r['test']:.3f}" for r in rs if "test" in r) or "–"
            lines.append(f"| {name} | {model} | {poles} | {', '.join(f'{r['err']:.1e}' for r in rs)} | {tests} |")
    (OUT.with_suffix(".md")).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
