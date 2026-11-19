"""(A) Delay and dispersion together: the wavenumber pole layer with a shared input delay.

**System.** ADR with a delayed input, u_t = L u + f(x, t - theta): lambda(k) = -i v k - D k^2 - k_r with v = 0.5,
D = 0.05, k_r = 0.2 (a1's `adr`), and theta = 0.73 (not a multiple of dt = 0.025). Data as a1 (n_x = 64, T = 5,
4 forced modes in k = 1..16, E1 profiles). The truth is the undelayed exact response on a 32x finer time grid,
shifted by theta with linear interpolation on that grid. Splits 200 / 50 / 100; data seeds 0-4.

**Models** (layer modes 17, Re cap 0.5 with leak 0.05, k_ref 16; joint fit, L-BFGS 300, selection by validation):
- `wpo_delay`: N = 1 and a shared delay (`delay=` option). 10 starts: initial theta in {0.2, 0.5, 1, 2, 4} x 2 pole
  draws (N(0, 3^2) in k / k_ref units).
- `wpo_n1`, `wpo_n3`: no delay, N = 1 or 3 poles; 10 pole draws each. The extra poles may approximate the delay
  mode by mode.

**Oracle:** the true lambda(k) and theta in `wpo_delay`'s layer; measured before any fit (seed 0): 1.03e-2.

**Predictions** (written before running).

1. `wpo_delay`: test error <= 1.2x the oracle, theta within 1 %, and D, v within 1 % (k_r within 0.005), in at least
   4 of 5 seeds.
2. `wpo_n1` and `wpo_n3`: test error >= 2x the oracle in every seed.

    python experiments/A/process/a25_wpo_delay.py      # results/a25_wpo_delay.md / .json
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
sys.path.insert(0, str(HERE.parents[0] / "layer"))
import a1_dispersion as A  # noqa: E402
from lfo.models.wavenumber import WavenumberPoleLayer1d  # noqa: E402

THETA, K_REF = 0.73, 16.0
P = A.SYSTEMS["adr"]
OUT = HERE / "results" / "a25_wpo_delay.json"


def make_data(seed):
    rng = np.random.default_rng(300 + seed)
    n = sum(A.SPLIT)
    tf = np.linspace(0, A.T_END, (A.N_T - 1) * A.FINE + 1)
    K = A.N_X // 2 + 1
    fk = np.zeros((n, K, tf.size), complex)
    for _ in range(4):
        prof = A.profiles(tf, rng, n)
        km = rng.integers(1, 17, size=n)
        amp = (rng.normal(size=n) + 1j * rng.normal(size=n)) / km * A.N_X / 2
        fk[np.arange(n), km] += amp[:, None] * prof
    uk = A.foh_np(A.lam(np.arange(K), **P), fk, tf[1] - tf[0])
    shift = lambda a: np.interp(tf - THETA, tf, a.real, left=0.0) + 1j * np.interp(tf - THETA, tf, a.imag, left=0.0)
    uk = np.apply_along_axis(shift, -1, uk)
    f = np.fft.irfft(fk[..., ::A.FINE], n=A.N_X, axis=1)
    u = np.fft.irfft(uk[..., ::A.FINE], n=A.N_X, axis=1)
    c = np.cumsum([0, *A.SPLIT])
    return {k: (torch.tensor(f[a:b]), torch.tensor(u[a:b])) for k, a, b in zip(("train", "vali", "test"), c[:-1], c[1:])}


def build(kind, t, device, theta0=None):
    n = 3 if kind == "wpo_n3" else 1
    return WavenumberPoleLayer1d(n, t, modes=A.MODES, re_max=A.RE_MAX, re_leak=0.05, k_ref=K_REF,
                                 delay=theta0 if kind == "wpo_delay" else None).to(device)


def fit(kind, data, t, device):
    (xtr, ytr), (xva, yva) = ((x.to(device) for x in data[s]) for s in ("train", "vali"))
    starts = [(th, d) for th in (0.2, 0.5, 1.0, 2.0, 4.0) for d in range(2)] if kind == "wpo_delay" else [(None, d) for d in range(10)]
    best = None
    for i, (th, d) in enumerate(starts):
        torch.manual_seed(i)
        m = build(kind, t, device, th)
        with torch.no_grad():
            m.mu_re.normal_(0, 3.0), m.mu_im.normal_(0, 3.0)
        opt = torch.optim.LBFGS(m.parameters(), max_iter=300, line_search_fn="strong_wolfe", tolerance_grad=1e-12,
                                tolerance_change=1e-15, history_size=50)

        def closure():
            opt.zero_grad()
            loss = A.rel(m(xtr), ytr)
            if torch.isfinite(loss):
                loss.backward()
            return loss

        try:
            opt.step(closure)
        except (RuntimeError, IndexError):
            pass
        with torch.no_grad():
            v = float(A.rel(m(xva), yva))
        if math.isfinite(v) and (best is None or v < best[0]):
            best = (v, m)
    return best[1]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t = np.linspace(0, A.T_END, A.N_T)
    out = []
    for seed in range(5):
        data = make_data(seed)
        xte, yte = (x.to(device) for x in data["test"])
        orc = build("wpo_delay", t, device, THETA)
        with torch.no_grad():
            orc.mu_re.zero_(), orc.mu_im.zero_(), orc.beta.zero_()
            orc.mu_re[0, 0], orc.mu_re[0, 1], orc.mu_im[0, 0] = -P["kr"], -P["D"] * K_REF ** 2, -P["v"] * K_REF
            orc.beta[0, 0, 0] = 1.0
            floor = float(A.rel(orc(xte), yte))
        for kind in ("wpo_delay", "wpo_n1", "wpo_n3"):
            t0 = time.time()
            m = fit(kind, data, t, device)
            with torch.no_grad():
                test = float(A.rel(m(xte), yte))
                ar, ci = m.mu_re[0].cpu().numpy(), m.mu_im[0].cpu().numpy()
            r = dict(seed=seed, kind=kind, oracle=floor, ratio=test / floor, seconds=round(time.time() - t0, 1))
            if kind == "wpo_delay":
                r.update(theta=float(torch.nn.functional.softplus(m.theta_raw)), D=float(-ar[1] / K_REF ** 2),
                         v=float(-ci[0] / K_REF), kr=float(-ar[0]))
            out.append(r)
            print(seed, kind, f"ratio {r['ratio']:.2f} (oracle {floor:.2e})",
                  {k: round(r[k], 5) for k in ("theta", "D", "v", "kr") if k in r}, f"{r['seconds']} s", flush=True)
            OUT.parent.mkdir(exist_ok=True)
            OUT.write_text(json.dumps(out, indent=1))
    lines = ["# (A) ADR with an input delay (θ = 0.73): wavenumber pole layer with and without a delay", "",
             "Truth: v 0.5, D 0.05, k_r 0.2, θ 0.73. Range over data seeds 0–4.", "",
             "| model | test / oracle | θ̂ | D̂ | v̂ | k̂_r |", "|---|---|---|---|---|---|"]
    for kind in ("wpo_delay", "wpo_n1", "wpo_n3"):
        rs = [r for r in out if r["kind"] == kind]
        g = lambda q: f"{min(r[q] for r in rs):.4f}–{max(r[q] for r in rs):.4f}" if q in rs[0] else "–"
        lines.append(f"| {kind} | {min(r['ratio'] for r in rs):.2f}–{max(r['ratio'] for r in rs):.2f} | {g('theta')} | {g('D')} | "
                     f"{g('v')} | {g('kr')} |")
    (OUT.with_suffix(".md")).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
