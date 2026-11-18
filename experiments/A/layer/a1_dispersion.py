"""(A) Pole layer, first linear validation (RQ1): does the wavenumber pole layer recover a dispersion relation?

Plan: `docs/proposal_A_wavenumber_pole_operator.qmd`, Sections 4 and 7 (RQ1, gate "dispersion relations recovered",
month 10). Layer: `lfo.models.wavenumber.WavenumberPoleLayer1d` (FOH time axis, D1 decision 2026-11-18).

**Systems** (periodic x in [0, 2 pi), zero initial state, u_t = L u + f, output u):

- `diff`: diffusion, lambda(k) = -D k^2, D = 0.1;
- `adr`: advection-diffusion-reaction, lambda(k) = -i v k - D k^2 - k_r, v = 0.5, D = 0.05, k_r = 0.2.

**Data.**

- Grid: n_x = 64, T = 5, n_t = 201.
- Forcing f(x, t) = sum_m a_m(x) p_m(t): 4 random spatial modes, wavenumbers 1..16, amplitudes ~ 1 / m.
  Temporal profiles come from the E1 families (step, ramp, PRBS, smooth random) on [0, T].
- Truth on a 32x finer time grid, per spatial mode: exact FOH response on that grid, in numpy and independent of
  the layer. Everything is then sampled to n_t. Following D1, neither the data nor the layer shares a hold.
- Splits 200 / 50 / 100.

**Model.**

- One layer, N poles, mu_n(k) = a0 + a1 k^2 + a2 k^4 + i (c1 k + c3 k^3), beta_n(k) = b0 + b1 k^2.
- Not stabilised; init scale 0.1; float64.
- Adam, lr 1e-2 with cosine decay, 3000 steps, full batch, relative L2 loss. N in {1, 2}, seeds 0-4.

**Metrics.**

- Test relative L2.
- Dispersion error: for each k in the forced band 1..16, the pole with the largest |beta_n(k)| against lambda(k),
  as || mu - lambda || / || lambda || over the band.
- Recovered physical constants (N = 1): D = -a1, v = c1, k_r = -a0.

**Predictions** (written before running).

1. N = 1: in every seed, test error <= 1.2x the oracle floor, the dispersion error <= 1 %, and D (and v, k_r for
   adr) within 1 % of the truth.

The oracle floor (true lambda(k) in the layer, on the sampled data) is about 9e-3 on `adr`; it was measured before
the runs and is set by the sampling of steps and PRBS, as in D1. An earlier draft of prediction 1 asked for a test
error <= 1e-3, which this floor rules out; it was corrected before any training.
2. N = 2: the dominant pole's dispersion error is <= 5 % in at least 4 of 5 seeds per system. A spare pole either
   carries little residue or splits the mode; the latter is Proposition 8's risk, which applies here only if the
   two poles share a mode.

    python experiments/A/layer/a1_dispersion.py      # results/a1_dispersion.md / .json
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
sys.path.insert(0, str(ROOT / "src"))
from lfo.data import lti  # noqa: E402
from lfo.models.wavenumber import WavenumberPoleLayer1d  # noqa: E402

N_X, T_END, N_T, FINE = 64, 5.0, 201, 32
SPLIT = (200, 50, 100)
BAND = np.arange(1, 17)
SYSTEMS = {"diff": dict(v=0.0, D=0.1, kr=0.0), "adr": dict(v=0.5, D=0.05, kr=0.2)}
STEPS, LR = 3000, 1e-2
OUT = HERE / "results" / "a1_dispersion.json"


def lam(k, v, D, kr):
    return -1j * v * k - D * k ** 2 - kr


def profiles(tf, rng, n):
    T = tf[-1]
    fams = [lambda: lti.step(tf, rng.uniform(0, T / 2), rng.uniform(0.5, 2) * rng.choice([-1, 1])),
            lambda: lti.ramp(tf, rng.uniform(0, T / 2), rng.uniform(0.2, 1) * rng.choice([-1, 1])),
            lambda: lti.prbs(tf, rng, hold=rng.uniform(0.2, 1.0)),
            lambda: lti.smooth_random(tf, rng)]
    return np.stack([fams[rng.integers(4)]() for _ in range(n)])


def foh_np(lmb, u, dt):
    """Exact FOH response from rest, per mode: lmb (K,), u (B, K, n) complex -> (B, K, n)."""
    z = lmb * dt
    E = np.exp(z)
    small = np.abs(z) < 1e-6
    s = np.where(small, 1.0, lmb)
    I0 = np.where(small, dt, (E - 1) / s)
    I1 = np.where(small, dt ** 2 / 2, (E - 1 - z) / s ** 2)
    x = np.zeros(u.shape[:2], complex)
    out = np.zeros(u.shape, complex)
    for j in range(u.shape[2] - 1):
        x = E * x + u[..., j] * I0 + (u[..., j + 1] - u[..., j]) / dt * I1
        out[..., j + 1] = x
    return out


def make_data(system, seed=0):
    p = SYSTEMS[system]
    rng = np.random.default_rng(seed)
    n = sum(SPLIT)
    t = np.linspace(0, T_END, N_T)
    tf = np.linspace(0, T_END, (N_T - 1) * FINE + 1)
    x = np.arange(N_X) * 2 * math.pi / N_X
    K = N_X // 2 + 1
    fk_fine = np.zeros((n, K, tf.size), complex)
    for m in range(4):
        prof = profiles(tf, rng, n)                                      # (n, n_tf)
        km = rng.integers(1, 17, size=n)
        amp = (rng.normal(size=n) + 1j * rng.normal(size=n)) / km * N_X / 2
        fk_fine[np.arange(n), km] += amp[:, None] * prof
    uk_fine = foh_np(lam(np.arange(K), **p), fk_fine, tf[1] - tf[0])
    f = np.fft.irfft(fk_fine[..., ::FINE], n=N_X, axis=1)
    u = np.fft.irfft(uk_fine[..., ::FINE], n=N_X, axis=1)
    s = np.cumsum([0, *SPLIT])
    to = lambda a: torch.tensor(a)
    return {k: (to(f[a:b]), to(u[a:b])) for k, a, b in zip(("train", "vali", "test"), s[:-1], s[1:])}, t


def rel(a, b):
    return ((a - b).flatten(1).norm(dim=1) / b.flatten(1).norm(dim=1)).mean()


@torch.no_grad()
def oracle(system, device):
    data, t = make_data(system)
    p = SYSTEMS[system]
    m = WavenumberPoleLayer1d(1, t).to(device)
    m.mu_re.zero_(), m.mu_im.zero_(), m.beta.zero_()
    m.mu_re[0, 0], m.mu_re[0, 1], m.mu_im[0, 0], m.beta[0, 0, 0] = -p["kr"], -p["D"], -p["v"], 1.0
    xte, yte = (a.to(device) for a in data["test"])
    return float(rel(m(xte), yte))


def run(system, n_poles, seed, device):
    data, t = make_data(system)
    torch.manual_seed(seed)
    model = WavenumberPoleLayer1d(n_poles, t).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS)
    (xtr, ytr), (xte, yte) = ((a.to(device) for a in data[k]) for k in ("train", "test"))
    t0 = time.time()
    for _ in range(STEPS):
        opt.zero_grad()
        loss = rel(model(xtr), ytr)
        loss.backward()
        opt.step()
        sch.step()
    with torch.no_grad():
        test = float(rel(model(xte), yte))
        k = torch.tensor(BAND, dtype=torch.float64, device=device)
        mu, beta = model.poles(k)
        dom = beta.abs().argmax(0)
        mu_dom = mu[dom, torch.arange(k.numel())].cpu().numpy()
    true = lam(BAND, **SYSTEMS[system])
    res = dict(system=system, n_poles=n_poles, seed=seed, test=test, seconds=round(time.time() - t0, 1),
               disp_err=float(np.linalg.norm(mu_dom - true) / np.linalg.norm(true)),
               mu_re=model.mu_re.detach().cpu().tolist(), mu_im=model.mu_im.detach().cpu().tolist(),
               beta=model.beta.detach().cpu().tolist())
    if n_poles == 1:
        a, c = model.mu_re.detach().cpu().numpy()[0], model.mu_im.detach().cpu().numpy()[0]
        res.update(D=float(-a[1]), v=float(-c[0]), kr=float(-a[0]))
    return res


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = []
    floors = {s: oracle(s, device) for s in SYSTEMS}
    print("oracle floors", floors, flush=True)
    for system in SYSTEMS:
        for n_poles in (1, 2):
            for seed in range(5):
                r = run(system, n_poles, seed, device)
                r["oracle"] = floors[system]
                out.append(r)
                print(system, n_poles, seed, f"test {r['test']:.2e} disp {r['disp_err']:.3f}",
                      {k: round(r[k], 4) for k in ("D", "v", "kr") if k in r}, flush=True)
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1))
    summarize(out)


def summarize(out):
    lines = ["# (A) First linear validation: dispersion recovery by the wavenumber pole layer", "",
             "One layer, FOH time axis, polynomial μ(k), truth on a 32× finer time grid. Dispersion error over the "
             "forced band k = 1..16 (dominant pole). 5 seeds.", "",
             "| system | N | oracle floor | test rel. L2 (range) | dispersion error (range) | D | v | k_r |", "|---|---|---|---|---|---|---|---|"]
    for system, p in SYSTEMS.items():
        for n in (1, 2):
            rs = [r for r in out if r["system"] == system and r["n_poles"] == n]
            rg = lambda key: f"{min(r[key] for r in rs):.1e}–{max(r[key] for r in rs):.1e}"
            phys = (" | ".join(f"{min(r[q] for r in rs):.4f}–{max(r[q] for r in rs):.4f} ({p[q]:g})" for q in ("D", "v", "kr"))
                    if n == 1 else "– | – | –")
            lines.append(f"| {system} | {n} | {rs[0]['oracle']:.1e} | {rg('test')} | {rg('disp_err')} | {phys} |")
    (OUT.with_suffix(".md")).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
