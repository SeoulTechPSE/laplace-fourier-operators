"""(D2) Initial conditions as an explicit input, and rollout by chaining windows (plan D, Section 5).

The wavenumber pole layer gets an initial-condition path (`ic=True`, 7b70758): the free response
u0_hat(k) sum_n gamma_n(k) e^{mu_n(k) t} shares the poles with the forced response, so it adds only gamma_n(k).
Semigroup consistency is then structural: the end state of one window is the next window's initial field.

**System.** a1's ADR (lambda(k) = -0.5 i k - 0.05 k^2 - 0.2), periodic, n_x = 64, window T = 5, n_t = 201.
- Each window has a random initial field: modes 1..16, amplitudes ~ N(0, 1) / k.
- The forcing is a1's family at half amplitude.
- Truth: the exact forced response on a 32x finer time grid (a1), plus the exact free response
  u0_hat(k) e^{lambda(k) t}.
- Splits 200 / 50 / 100 windows; data seeds 0-4.
- Rollout set: 50 trajectories on [0, 20] (4 windows), with one initial field at t = 0 and forcing over the whole
  interval.

**Models** (N = 1, modes 17, leaky Re cap, k_ref 16; the a7 / a8 joint fit with 8 starts, L-BFGS, selection by
validation):
- `ic`: the layer with the initial-condition path;
- `noic`: the same layer without it; it sees only the forcing.

**Rollout.** The trained `ic` layer is run window by window. Each window's last field is the next window's u0, and
its forcing is that window's segment.

**Oracle, measured before any fit** (seed 0): the true parameters with gamma = 1 give 1.48e-3 on one window and
2.0e-3 over the 4-window rollout (ratio 1.33, from the sampling floor accumulating across windows).

**Predictions** (written before running).

1. `ic`: test error <= 1.2x the oracle, D, v, k_r within 1 % and gamma(k) within 1 % of 1 over k = 1..16, in every
   seed.
2. `noic`: test error >= 5x the oracle in every seed.
3. The 4-window rollout error of `ic` over [0, 20] is <= 1.5x its single-window test error, in every seed.

    python experiments/D/d2_initial_conditions/d2_ic_rollout.py      # results/d2_ic_rollout.md / .json
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
from lfo.models.wavenumber import WavenumberPoleLayer1d  # noqa: E402

P = A.SYSTEMS["adr"]
K_REF, N_WIN = 16.0, 4
OUT = HERE / "results" / "d2_ic_rollout.json"


def sample(rng, n, T, n_t):
    """Forcing (fine spectrum), initial fields and the exact solution sampled to n_t points on [0, T]."""
    tf = np.linspace(0, T, (n_t - 1) * A.FINE + 1)
    K = A.N_X // 2 + 1
    fk = np.zeros((n, K, tf.size), complex)
    for _ in range(4):
        prof = A.profiles(tf, rng, n)
        km = rng.integers(1, 17, size=n)
        amp = 0.5 * (rng.normal(size=n) + 1j * rng.normal(size=n)) / km * A.N_X / 2
        fk[np.arange(n), km] += amp[:, None] * prof
    u0k = np.zeros((n, K), complex)
    ks = np.arange(1, 17)
    u0k[:, 1:17] = (rng.normal(size=(n, 16)) + 1j * rng.normal(size=(n, 16))) / ks * A.N_X / 2
    lam = A.lam(np.arange(K), **P)
    uk = A.foh_np(lam, fk, tf[1] - tf[0]) + u0k[..., None] * np.exp(lam[None, :, None] * tf[None, None])
    f = np.fft.irfft(fk[..., ::A.FINE], n=A.N_X, axis=1)
    u = np.fft.irfft(uk[..., ::A.FINE], n=A.N_X, axis=1)
    u0 = np.fft.irfft(u0k, n=A.N_X, axis=1)
    return torch.tensor(f), torch.tensor(u0), torch.tensor(u)


def make_data(seed):
    rng = np.random.default_rng(400 + seed)
    f, u0, u = sample(rng, sum(A.SPLIT), A.T_END, A.N_T)
    c = np.cumsum([0, *A.SPLIT])
    d = {k: (f[a:b], u0[a:b], u[a:b]) for k, a, b in zip(("train", "vali", "test"), c[:-1], c[1:])}
    d["roll"] = sample(rng, 50, N_WIN * A.T_END, N_WIN * (A.N_T - 1) + 1)
    return d


def build(ic, t, device):
    return WavenumberPoleLayer1d(1, t, modes=A.MODES, re_max=A.RE_MAX, re_leak=0.05, k_ref=K_REF, ic=ic).to(device)


def run(m, f, u0, ic):
    return m(f, u0) if ic else m(f)


def fit(ic, data, t, device):
    (ftr, u0tr, ytr), (fva, u0va, yva) = ((x.to(device) for x in data[s]) for s in ("train", "vali"))
    best = None
    for s in range(8):
        torch.manual_seed(s)
        m = build(ic, t, device)
        with torch.no_grad():
            m.mu_re.normal_(0, 3.0), m.mu_im.normal_(0, 3.0)
        opt = torch.optim.LBFGS(m.parameters(), max_iter=300, line_search_fn="strong_wolfe", tolerance_grad=1e-12,
                                tolerance_change=1e-15, history_size=50)

        def closure():
            opt.zero_grad()
            loss = A.rel(run(m, ftr, u0tr, ic), ytr)
            if torch.isfinite(loss):
                loss.backward()
            return loss

        try:
            opt.step(closure)
        except (RuntimeError, IndexError):
            pass
        with torch.no_grad():
            v = float(A.rel(run(m, fva, u0va, ic), yva))
        if math.isfinite(v) and (best is None or v < best[0]):
            best = (v, m)
    return best[1]


@torch.no_grad()
def rollout(m, f, u0):
    w = A.N_T - 1
    out, state = [u0[..., None]], u0
    for i in range(N_WIN):
        seg = m(f[..., i * w:(i + 1) * w + 1], state)
        out.append(seg[..., 1:])
        state = seg[..., -1]
    return torch.cat(out, -1)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t = np.linspace(0, A.T_END, A.N_T)
    out = []
    for seed in range(5):
        data = make_data(seed)
        fte, u0te, yte = (x.to(device) for x in data["test"])
        orc = build(True, t, device)
        with torch.no_grad():
            orc.mu_re.zero_(), orc.mu_im.zero_(), orc.beta.zero_(), orc.gamma.zero_()
            orc.mu_re[0, 0], orc.mu_re[0, 1], orc.mu_im[0, 0] = -P["kr"], -P["D"] * K_REF ** 2, -P["v"] * K_REF
            orc.beta[0, 0, 0], orc.gamma[0, 0, 0] = 1.0, 1.0
            floor = float(A.rel(orc(fte, u0te), yte))
        for ic in (True, False):
            t0 = time.time()
            m = fit(ic, data, t, device)
            with torch.no_grad():
                test = float(A.rel(run(m, fte, u0te, ic), yte))
            r = dict(seed=seed, model="ic" if ic else "noic", oracle=floor, ratio=test / floor, test=test,
                     seconds=round(time.time() - t0, 1))
            if ic:
                ar, ci = m.mu_re[0].detach().cpu().numpy(), m.mu_im[0].detach().cpu().numpy()
                k2 = (torch.arange(1, 17, dtype=torch.float64) / K_REF) ** 2
                gm = m.gamma.detach().cpu()
                gam = torch.complex(gm[0, 0, 0] + gm[0, 1, 0] * k2, gm[0, 0, 1] + gm[0, 1, 1] * k2)
                fr, u0r, yr = (x.to(device) for x in data["roll"])
                roll = float(A.rel(rollout(m, fr, u0r), yr))
                r.update(D=float(-ar[1] / K_REF ** 2), v=float(-ci[0] / K_REF), kr=float(-ar[0]),
                         gamma_err=float((gam - 1).abs().max()), rollout=roll, rollout_ratio=roll / test)
            out.append(r)
            print(seed, r["model"], f"ratio {r['ratio']:.2f}", {k: round(r[k], 5) for k in ("D", "v", "kr", "gamma_err", "rollout", "rollout_ratio") if k in r},
                  f"{r['seconds']} s", flush=True)
            OUT.parent.mkdir(exist_ok=True)
            OUT.write_text(json.dumps(out, indent=1))
    lines = ["# (D2) Initial-condition path and window chaining (ADR, periodic)", "",
             "Range over data seeds 0–4. Rollout: 4 chained windows on [0, 20]; ratio to the single-window test error.", "",
             "| model | test / oracle | D̂ | v̂ | k̂_r | max abs(γ(k) − 1) | rollout error | rollout / test |", "|---|---|---|---|---|---|---|---|"]
    for model in ("ic", "noic"):
        rs = [r for r in out if r["model"] == model]
        g = lambda q, f=".4f": f"{min(r[q] for r in rs):{f}}–{max(r[q] for r in rs):{f}}" if q in rs[0] else "–"
        lines.append(f"| {model} | {g('ratio', '.2f')} | {g('D')} | {g('v')} | {g('kr')} | {g('gamma_err', '.1e')} | {g('rollout', '.2e')} | "
                     f"{g('rollout_ratio', '.2f')} |")
    (OUT.with_suffix(".md")).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
