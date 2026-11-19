"""(A) RQ4: polynomial against MLP pole functions. Does the learned dispersion relation extend beyond the forced band?

The plan (Section 4.3) offers two parameterisations of mu_n(k): a polynomial (strong prior, interpretable
coefficients) and an MLP (flexible, interpreted post hoc). Inside the band the data excite, both can fit. The test
here is the band the data do not excite: training data are forced at k <= 8 only, and the learned mu(k) is checked
at k = 9..16, both directly (dispersion error) and on test data forced at k = 9..16.

**Systems and data** (as a1 / a4: n_x = 64, T = 5, n_t = 201, truth on a 32x finer time grid, data seeds 0-4):
`diff` (lambda = -0.1 k^2), `adr` (-0.5 i k - 0.05 k^2 - 0.2) and `beam` (velocity; poles +-0.04 i k^2, residues
1/2). Train / validation forced at k in 1..8 (200 / 50 samples). Two test sets of 100: forced at k in 1..8
(in-band), and forced at k in 9..16 (out-of-band).

**Models** (layer modes 17, Re cap 0.5 with leak 0.05, k_ref 16; N = 1, or N = 2 with Im powers (1, 2, 3) for the
beam):
- `poly`: the default polynomial (Re: 1, k^2, k^4; Im: k, k^3, or k, k^2, k^3 for the beam);
- `mlp`: `mu_mlp = 32` (two tanh layers of 32), residues polynomial as before.
Both: the a7 / a8 joint fit, 8 random starts (polynomial coefficients N(0, 3^2); the MLP at PyTorch's default
init, per start seed), L-BFGS 300 iterations, selection by in-band validation error.

**Predictions** (written before running).

1. In-band: both `poly` and `mlp` reach <= 1.2x the oracle on the in-band test set, in every system and seed.
2. Out-of-band dispersion error (k = 9..16): `poly` <= 5 % in at least 4 of 5 seeds per system; `mlp` > 10 % in at
   least 4 of 5 seeds per system.
3. Out-of-band test set: `poly` <= 1.5x the oracle in at least 4 of 5 seeds per system; `mlp` > 3x in at least 4 of
   5 seeds per system.

    python experiments/A/layer/a19_mlp_band.py      # results/a19_mlp_band.md / .json
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
sys.path.insert(0, str(HERE))
import a1_dispersion as A  # noqa: E402
from lfo.models.wavenumber import WavenumberPoleLayer1d  # noqa: E402

SYSTEMS = ("diff", "adr", "beam")
STARTS, SIGMA, ITERS, K_REF = 8, 3.0, 300, 16.0
KS_IN, KS_OUT = np.arange(1, 9), np.arange(9, 17)
OUT = HERE / "results" / "a19_mlp_band.json"


def poles(system, k):
    """True poles (P, len(k)) and residues (P,)."""
    k = np.asarray(k, float)
    if system == "beam":
        return np.stack([0.04j * k ** 2, -0.04j * k ** 2]), np.array([0.5, 0.5])
    return A.lam(k, **A.SYSTEMS[system])[None], np.array([1.0])


def make_set(system, rng, n, kmin, kmax):
    tf = np.linspace(0, A.T_END, (A.N_T - 1) * A.FINE + 1)
    K = A.N_X // 2 + 1
    fk = np.zeros((n, K, tf.size), complex)
    for _ in range(4):
        prof = A.profiles(tf, rng, n)
        km = rng.integers(kmin, kmax + 1, size=n)
        amp = (rng.normal(size=n) + 1j * rng.normal(size=n)) / km * A.N_X / 2
        fk[np.arange(n), km] += amp[:, None] * prof
    mu, r = poles(system, np.arange(K))
    uk = sum(r[p] * A.foh_np(mu[p], fk, tf[1] - tf[0]) for p in range(len(r)))
    f = np.fft.irfft(fk[..., ::A.FINE], n=A.N_X, axis=1)
    u = np.fft.irfft(uk[..., ::A.FINE], n=A.N_X, axis=1)
    return torch.tensor(f), torch.tensor(u)


def make_data(system, seed):
    rng = np.random.default_rng(100 + seed)
    return dict(train=make_set(system, rng, 200, 1, 8), vali=make_set(system, rng, 50, 1, 8),
                test_in=make_set(system, rng, 100, 1, 8), test_out=make_set(system, rng, 100, 9, 16))


def build(system, t, device, kind):
    kw = dict(modes=A.MODES, re_max=A.RE_MAX, re_leak=0.05, k_ref=K_REF)
    if kind == "mlp":
        kw["mu_mlp"] = 32
    if system == "beam":
        return WavenumberPoleLayer1d(2, t, mu_degree=(3, 3), beta_degree=2, im_powers=(1, 2, 3), **kw).to(device)
    return WavenumberPoleLayer1d(1, t, **kw).to(device)


@torch.no_grad()
def oracle(system, data, t, device):
    m = build(system, t, device, "poly")
    for q in m.parameters():
        q.zero_()
    if system == "beam":
        m.mu_im[0, 1], m.mu_im[1, 1] = 0.04 * K_REF ** 2, -0.04 * K_REF ** 2
        m.beta[:, 0, 0] = 0.5
    else:
        p = A.SYSTEMS[system]
        m.mu_re[0, 0], m.mu_re[0, 1], m.mu_im[0, 0] = -p["kr"], -p["D"] * K_REF ** 2, -p["v"] * K_REF
        m.beta[0, 0, 0] = 1.0
    return {s: float(A.rel(m(data[s][0].to(device)), data[s][1].to(device))) for s in ("test_in", "test_out")}


def disp_err(system, m, ks, device):
    with torch.no_grad():
        mu = m.mu(torch.tensor(ks, dtype=torch.float64, device=device)).cpu().numpy()
    tr, _ = poles(system, ks)
    if system == "beam":                                   # match labels by the sign of Im at the top of the band
        mu = mu if mu[0, -1].imag > 0 else mu[::-1]
    return float(np.linalg.norm(mu - tr) / np.linalg.norm(tr))


def fit(system, data, t, device, kind):
    (xtr, ytr), (xva, yva) = ((x.to(device) for x in data[s]) for s in ("train", "vali"))
    best = None
    for s in range(STARTS):
        torch.manual_seed(s)
        m = build(system, t, device, kind)
        if kind == "poly":
            with torch.no_grad():
                m.mu_re.normal_(0, SIGMA), m.mu_im.normal_(0, SIGMA)
        opt = torch.optim.LBFGS(m.parameters(), max_iter=ITERS, line_search_fn="strong_wolfe", tolerance_grad=1e-12,
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
    for system in SYSTEMS:
        for seed in range(5):
            data = make_data(system, seed)
            orc = oracle(system, data, t, device)
            for kind in ("poly", "mlp"):
                t0 = time.time()
                m = fit(system, data, t, device, kind)
                with torch.no_grad():
                    err = {s: float(A.rel(m(data[s][0].to(device)), data[s][1].to(device))) for s in ("test_in", "test_out")}
                r = dict(system=system, seed=seed, kind=kind, ratio_in=err["test_in"] / orc["test_in"],
                         ratio_out=err["test_out"] / orc["test_out"], disp_in=disp_err(system, m, KS_IN, device),
                         disp_out=disp_err(system, m, KS_OUT, device), oracle=orc, seconds=round(time.time() - t0, 1))
                out.append(r)
                print(system, seed, kind, f"in {r['ratio_in']:.2f}x out {r['ratio_out']:.2f}x",
                      f"disp in {r['disp_in']:.2e} out {r['disp_out']:.2e}", f"{r['seconds']} s", flush=True)
                OUT.write_text(json.dumps(out, indent=1))
    lines = ["# (A) RQ4: polynomial against MLP pole functions; training forced at k ≤ 8", "",
             "Test / oracle on data forced in-band (k ≤ 8) and out-of-band (k = 9..16); dispersion error over k = 1..8 "
             "and 9..16. Range over data seeds 0–4.", "",
             "| system | μ(k) | in-band test / oracle | out-of-band test / oracle | dispersion error k ≤ 8 | k = 9..16 |",
             "|---|---|---|---|---|---|"]
    for system in SYSTEMS:
        for kind in ("poly", "mlp"):
            rs = [r for r in out if r["system"] == system and r["kind"] == kind]
            g = lambda q, f: f"{min(r[q] for r in rs):{f}}–{max(r[q] for r in rs):{f}}"
            lines.append(f"| {system} | {kind} | {g('ratio_in', '.2f')} | {g('ratio_out', '.2f')} | {g('disp_in', '.1e')} | "
                         f"{g('disp_out', '.1e')} |")
    (OUT.with_suffix(".md")).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
