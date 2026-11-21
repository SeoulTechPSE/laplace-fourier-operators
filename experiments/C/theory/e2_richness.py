"""(C) E2: data richness and practical identifiability of the poles (RQ2, H2).

Plan: docs/proposal_C_identifiability.qmd, H2 and E2 ("vary the number of input
frequencies and the window T to test the persistent-excitation condition of H2").

Phase 1 (this script): local identifiability of the tied (LNO v1) layer at the truth,
swept over the data. For each configuration (system, input family, number of input
frequencies K, window T) it reports

  floor    e at the true parameters: the layer's own representation error (DFT
           discretization) on these inputs; e = RMS relative L2 over the samples
  cond     condition number of the Gauss-Newton matrix F
  s_k      profiled error caused by a 10 % change of Re mu_k (other parameters
           re-fitted, linearized), as in ``e1_audit/identifiability.py``
  ratio_k  s_k / floor; a pole counts as practically identifiable when ratio >= 10
           (its 10 % change is visible an order of magnitude above the floor)

The tied layer, its float64 arithmetic and the Gauss-Newton set-up are taken from
``experiments/C/e1_audit/identifiability.py`` unchanged. The time step is fixed
(dt = 20 / 1024, the E1 grid); T changes the number of samples.

Input families (amplitudes random per sample, data seed 0, 200 samples):
  e1        the E1 mix: step, ramp, PRBS, smooth random (equal shares)
  step, ramp, prbs, smooth   one E1 family alone
  multisine sum of K sinusoids at fixed frequencies log-spaced in [0.1, 10] rad/s,
            random amplitudes and phases, switched on at t = 0
  pulse     1-3 rectangular pulses per sample, width U[0.05, 0.5], random sign/height

    python experiments/C/theory/e2_richness.py          # writes results/e2_richness.md (+ .json)
    python experiments/C/theory/e2_richness.py --hold foh   # results/e2_richness_foh.md (+ .json)

``--hold foh``: the same analysis with the layer's 1 / (i omega - mu) replaced by the exact
sampled response to a first-order-hold input (the data generator's input model, as
``FOHPoleResidue`` in e2_train.py). The floor then drops to the float64 rounding of the data,
so the ratio s_k / floor is no longer the limit; the table reports s_k / 10 instead: the
output-noise level (relative L2 per sample) up to which a 10 % change of Re mu_k stays an
order of magnitude above the noise.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "e1_audit"))
import e1_audit as E  # noqa: E402
import identifiability as ID  # noqa: E402

from lfo.data import lti  # noqa: E402

RESULTS = HERE / "results"
DT = E.T_END / E.N_T
N_SAMPLES = 200
RATIO_OK = 10.0


# ---- inputs ---------------------------------------------------------------------

def multisine(t, rng, K: int, w_lo=0.1, w_hi=10.0):
    w = np.geomspace(w_lo, w_hi, K) if K > 1 else np.array([np.sqrt(w_lo * w_hi)])
    a = rng.normal(size=K) / np.sqrt(K)
    ph = rng.uniform(0, 2 * np.pi, size=K)
    return np.sum(a[:, None] * np.sin(w[:, None] * t[None, :] + ph[:, None]), axis=0)


def pulses(t, rng):
    u = np.zeros_like(t)
    for _ in range(rng.integers(1, 4)):
        t0, w = rng.uniform(0, 0.8 * t[-1]), rng.uniform(0.05, 0.5)
        u += rng.choice([-1, 1]) * rng.uniform(0.5, 2.0) * ((t >= t0) & (t < t0 + w))
    return u


def inputs(family: str, t, rng, K: int = 0) -> np.ndarray:
    T = t[-1] + DT
    one = {
        "step": lambda: lti.step(t, rng.uniform(0, T / 2), rng.uniform(0.5, 2.0) * rng.choice([-1, 1])),
        "ramp": lambda: lti.ramp(t, rng.uniform(0, T / 2), rng.uniform(0.05, 0.3) * rng.choice([-1, 1])),
        "prbs": lambda: lti.prbs(t, rng, hold=rng.uniform(0.3, 2.0)),
        "smooth": lambda: lti.smooth_random(t, rng),
        "multisine": lambda: multisine(t, rng, K),
        "pulse": lambda: pulses(t, rng),
    }
    if family == "e1":
        fams = ["step", "ramp", "prbs", "smooth"]
        return np.stack([one[fams[i % 4]]() for i in range(N_SAMPLES)])
    return np.stack([one[family]() for _ in range(N_SAMPLES)])


FINE = 32     # families ending in "fine" (D1 third pass): inputs and truth on a FINE x finer grid, then sampled


def make(system: str, family: str, T: float, K: int = 0, seed: int = 0):
    mu, beta = (np.array(a, dtype=complex) for a in E.SYSTEMS[system])
    n = int(round(T / DT))
    t = np.arange(n) * DT
    if family.endswith("fine"):
        tf = np.arange(n * FINE) * DT / FINE
        uf = inputs(family[:-4], tf, np.random.default_rng(seed), K)
        u, y = uf[:, ::FINE], lti.response(tf, uf, mu, beta)[:, ::FINE]
    else:
        u = inputs(family, t, np.random.default_rng(seed), K)
        y = lti.response(t, u, mu, beta)
    m, b = ID.minimal_poles(mu, beta)
    return torch.tensor(t), torch.tensor(u), torch.tensor(y), m, b


# ---- analysis -------------------------------------------------------------------

def layer_out_foh(mu, beta, x, t):
    """``identifiability.layer_out`` with the first-order-hold response
    H_l = (b0 + b1 z_l) / (z_l - E) per pole (E = e^{mu dt}, z_l = e^{i omega_l dt})."""
    n = x.shape[-1]
    dt = float(t[1] - t[0])
    alpha = torch.fft.fft(x)
    iw = 2j * math.pi * torch.fft.fftfreq(n, dt, dtype=torch.float64)
    z = mu * dt
    Ek = torch.exp(z)
    I0, I1 = (Ek - 1) / mu, (Ek - 1 - z) / mu ** 2
    zl = torch.exp(iw * dt)[:, None]
    H = beta[None, :] * ((I0 - I1 / dt)[None, :] + (I1 / dt)[None, :] * zl) / (zl - Ek[None, :])
    gamma = -(alpha @ H) / n
    tr = torch.real(gamma @ torch.exp(mu[:, None] * t[None, :].to(torch.complex128)))
    lam = alpha * H.sum(1)[None, :]
    pos = torch.arange(n // 2 + 1)
    c = (lam[:, pos] + torch.conj_physical(lam[:, (-pos) % n])) / 2
    return tr + torch.fft.irfft(c, n=n)


def local(system, family, T, K=0) -> dict:
    t, x, y, m, b = make(system, family, T, K)
    keep = y.norm(dim=1) > 1e-8 * y.norm(dim=1).max()        # drop all-zero outputs (e.g. late pulses)
    x, y = x[keep], y[keep]
    P = ID.Params(m, b)
    v0 = torch.tensor(P.vector())
    J = torch.autograd.functional.jacobian(lambda v: ID.residual(P, v, x, y, t).reshape(-1), v0,
                                           vectorize=True, strategy="forward-mode")   # few params, many outputs
    F = (J.T @ J) / x.shape[0]
    ev = torch.linalg.eigvalsh(F)
    Finv = torch.linalg.pinv(F, hermitian=True)
    floor = ID.rms_error(P, v0, x, y, t)
    poles = []
    for j, z in enumerate(m):
        row = dict(pole=[z.real, z.imag])
        for part in ("Re mu", "Im mu"):
            if (j, part) not in P.names:
                continue
            k = P.names.index((j, part))
            s = 0.1 * abs(float(v0[k])) / float(Finv[k, k].sqrt())
            row[part] = dict(profiled_e=s, ratio=s / floor)
        poles.append(row)
    return dict(system=system, family=family, T=T, K=K, n_t=int(t.numel()), samples=int(x.shape[0]),
                floor=floor, cond=float(ev.max() / ev.min()), min_eig=float(ev.min()), poles=poles)


def sweeps():
    cfgs = []
    for s in ("p4", "p6"):
        cfgs += [(s, "multisine", 20.0, K) for K in (1, 2, 3, 4, 6, 8, 12, 16, 32)]     # richness
        cfgs += [(s, fam, T, 16 if fam == "multisine" else 0)                           # window
                 for fam, T in itertools.product(("e1", "multisine"), (5.0, 10.0, 40.0, 80.0))]
        cfgs += [(s, fam, 20.0, 0) for fam in ("e1", "step", "ramp", "prbs", "smooth", "pulse")]  # family
    return cfgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hold", choices=("band", "foh"), default="band")
    hold = ap.parse_args().hold
    if hold == "foh":
        ID.layer_out = layer_out_foh                     # ID.residual looks it up at call time
        return main_foh()
    out = []
    for cfg in sweeps():
        r = local(*cfg)
        out.append(r)
        RESULTS.mkdir(exist_ok=True)
        (RESULTS / "e2_richness.json").write_text(json.dumps(out, indent=1))   # save as we go
        worst = min(p["Re mu"]["ratio"] for p in r["poles"])
        print(f"{r['system']} {r['family']:9s} K={r['K']:2d} T={r['T']:4.0f}: floor {r['floor']:.1e} "
              f"cond {r['cond']:.1e} min Re-ratio {worst:.1f}", flush=True)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "e2_richness.json").write_text(json.dumps(out, indent=1))
    write_md(out)


def main_foh():
    band = {(r["system"], r["family"], r["T"], r["K"]): r for r in json.loads((RESULTS / "e2_richness.json").read_text())}
    out = []
    for cfg in sweeps():
        r = local(*cfg)
        out.append(r)
        (RESULTS / "e2_richness_foh.json").write_text(json.dumps(out, indent=1))
    fmt = lambda z: f"{z[0]:+g}{z[1]:+g}i" if z[1] else f"{z[0]:+g}"
    lines = ["# E2 phase 1 with the first-order-hold layer (at the truth)", "",
             "As `e2_richness.md`, with the exact first-order-hold response in the tied layer. Floor: band layer → "
             "FOH layer. Per pole: band ratio (profiled error of a 10 % change in Re μ / band floor; **bold** ≥ "
             f"{RATIO_OK:g}) / η* = profiled error / 10 with the FOH layer: the output-noise level (relative L2 per "
             "sample) below which the 10 % change stays ≥ 10 × the noise.", ""]
    for system in ("p4", "p6"):
        rs = [r for r in out if r["system"] == system]
        poles = [fmt(p["pole"]) for p in rs[0]["poles"]]
        lines += [f"## {system}", "",
                  "| family | K | T | floor (band → FOH) | " + " | ".join(f"Re {p}: band ratio / η*" for p in poles) + " |",
                  "|---|---|---|---|" + "---|" * len(poles)]
        for r in rs:
            b = band[(r["system"], r["family"], r["T"], r["K"])]
            cells = []
            for pb, pf in zip(b["poles"], r["poles"]):
                q = pb["Re mu"]["ratio"]
                cells.append((f"**{q:.1f}**" if q >= RATIO_OK else f"{q:.1f}") + f" / {pf['Re mu']['profiled_e'] / 10:.1e}")
            lines.append(f"| {r['family']} | {r['K'] or '–'} | {r['T']:g} | {b['floor']:.1e} → {r['floor']:.1e} | "
                         + " | ".join(cells) + " |")
        lines.append("")
    (RESULTS / "e2_richness_foh.md").write_text("\n".join(lines))
    print("\n".join(lines))


def write_md(out):
    fmt = lambda z: f"{z[0]:+g}{z[1]:+g}i" if z[1] else f"{z[0]:+g}"
    lines = ["# E2 phase 1: local identifiability vs data (tied layer, at the truth)", "",
             "ratio = profiled error of a 10 % change in Re μ / representation floor; "
             f"**bold** = practically identifiable (ratio ≥ {RATIO_OK:g}). "
             "Im μ of the complex poles is listed in the JSON (always ≥ the Re ratio here).", ""]
    for system in ("p4", "p6"):
        rs = [r for r in out if r["system"] == system]
        poles = [fmt(p["pole"]) for p in rs[0]["poles"]]
        lines += [f"## {system}", "",
                  "| sweep | family | K | T | floor | cond(F) | " + " | ".join(f"Re {p}" for p in poles) + " |",
                  "|---|---|---|---|---|---|" + "---|" * len(poles)]
        for r in rs:
            sweep = ("richness" if r["family"] == "multisine" and r["T"] == 20.0 else
                     "window" if r["T"] != 20.0 else "family")
            cells = []
            for p in r["poles"]:
                q = p["Re mu"]["ratio"]
                cells.append(f"**{q:.1f}**" if q >= RATIO_OK else f"{q:.1f}")
            lines.append(f"| {sweep} | {r['family']} | {r['K'] or '–'} | {r['T']:g} | {r['floor']:.1e} | "
                         f"{r['cond']:.1e} | " + " | ".join(cells) + " |")
        lines.append("")
    (RESULTS / "e2_richness.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
