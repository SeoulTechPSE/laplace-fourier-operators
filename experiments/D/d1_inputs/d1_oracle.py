"""(D1) Oracle comparison of input representations: how much error does each time-axis computation add, with the
true poles? (No training.)

Plan: `docs/proposal_D_nonperiodic_initial_conditions.qmd`, Section 4 and RQ1; first step of the D1 timeline
("quantify the endpoint error of the Fourier path"). The pole-residue structure is fixed at the truth, so every
error below comes from the representation of the input alone.

Systems: E1's p2, p4, p6 (`experiments/C/e1_audit/e1_audit.py`). Window T = 20, n = 1024 (E1 grid); the Fourier path
also at n = 256 and 4096. Inputs, 100 per family: step, ramp, PRBS and smooth random with mismatched endpoints
(`lfo.data.lti`, E1 parameter ranges), plus a periodic control (integer harmonics, matched endpoints).

Truth: `lfo.data.lti.response` (exact for the piecewise-linear interpolant of the samples).

Paths:

- `fourier`      LNO v1 layer in the textbook wiring (`PoleResidueTS1d`) with the true poles and residues: the
                 response to the trigonometric interpolant (Lemma 1 of `experiments/C/theory/H1_H2.md`);
- `fourier_pad2`, `fourier_pad4`  the same on the input zero-padded to 2n / 4n, cropped to [0, T];
- `zoh`          exponential-integrator recursion with a zero-order hold, O(n);
- `foh`          the same with a first-order hold (exact for the truth's interpolant; a sanity check);
- `cheb16`, `cheb64`, `cheb256`  least-squares Chebyshev projection of the samples (degree d on [0, T]), evaluated
                 on a 16x finer grid and integrated exactly (FOH on the fine grid): the representation error of a
                 global Chebyshev path, without its recursion.

Metrics (per sample, then median over the family): relative L2 error on the window; endpoint error = relative L2
error over the first and the last 5 % of the window (each against the output there); max error / max |output|.

Predictions (written before running):

1. `fourier`: on the four mismatched-endpoint families the median relative error is at least 10x that on the
   periodic control, and the last-5 % endpoint error is at least 3x the whole-window error.
2. `fourier` error falls with n, with a log-log slope between -0.5 and -1.5 from n = 256 to 4096.
3. `foh` reproduces the truth to <= 1e-10; `zoh` error is O(dt): halving dt halves it (not run at other n here,
   reported at n = 1024 only).
4. A global Chebyshev path (`cheb64`) is better than `fourier` on smooth random inputs but worse on step and PRBS,
   whose discontinuities lie inside the window.

    python experiments/D/d1_inputs/d1_oracle.py      # writes results/d1_oracle.md / .json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "src"))
from lfo.data import lti  # noqa: E402
from lfo.models.hybrid import PoleResidueTS1d  # noqa: E402

SYSTEMS = {
    "p2": ([-0.5 + 2j, -0.5 - 2j], [0.5 - 0.3j, 0.5 + 0.3j]),
    "p4": ([-0.4 + 3j, -0.4 - 3j, -1.0, -0.2], [0.3 - 0.4j, 0.3 + 0.4j, 1.0, -0.5]),
    "p6": ([-0.5 + 1.5j, -0.5 - 1.5j, -0.3 + 4j, -0.3 - 4j, -1.0, -2.0],
           [0.5 - 0.3j, 0.5 + 0.3j, 0.4 + 0.2j, 0.4 - 0.2j, 1.0, -0.8]),
}                                                    # = e1_audit.SYSTEMS
T_END, N_DEF, N_PER = 20.0, 1024, 100
FAMILIES = ("step", "ramp", "prbs", "smooth", "periodic")
END = 0.05
OUT = HERE / "results" / "d1_oracle.json"


def grid(n):
    return np.linspace(0.0, T_END, n)


def inputs(family, t, rng):
    T = t[-1]
    if family == "step":
        return np.stack([lti.step(t, rng.uniform(0, T / 2), rng.uniform(0.5, 2.0) * rng.choice([-1, 1]))
                         for _ in range(N_PER)])
    if family == "ramp":
        return np.stack([lti.ramp(t, rng.uniform(0, T / 2), rng.uniform(0.05, 0.3) * rng.choice([-1, 1]))
                         for _ in range(N_PER)])
    if family == "prbs":
        return np.stack([lti.prbs(t, rng, hold=rng.uniform(0.3, 2.0)) for _ in range(N_PER)])
    if family == "smooth":
        return np.stack([lti.smooth_random(t, rng) for _ in range(N_PER)])
    k = np.arange(1, 9)                              # periodic control: integer harmonics of the window t[-1] + dt
    P = t[-1] + (t[1] - t[0])
    out = []
    for _ in range(N_PER):
        a, ph = rng.normal(size=8) / k ** 1.5, rng.uniform(0, 2 * np.pi, 8)
        out.append(np.sum(a[:, None] * np.sin(2 * np.pi * k[:, None] * t[None] / P + ph[:, None]), 0))
    return np.stack(out)


def fourier(u, t, mu, beta, pad=1):
    n = t.size
    m = n * pad
    tt = np.arange(m) * (t[1] - t[0])
    up = np.zeros((u.shape[0], m))
    up[:, :n] = u
    layer = PoleResidueTS1d(1, 1, len(mu), tt).double()
    with torch.no_grad():
        layer.mu.copy_(torch.tensor(np.asarray(mu), dtype=torch.complex128).reshape(1, 1, -1))
        layer.beta.copy_(torch.tensor(np.asarray(beta), dtype=torch.complex128).reshape(1, 1, -1))
        y = layer(torch.tensor(up)[:, None, :])[:, 0].numpy()
    return y[:, :n]


def hold(u, t, mu, beta, order):
    """Exponential-integrator recursion, zero- (order 0) or first-order (1) hold; real output."""
    dt = t[1] - t[0]
    mu, beta = np.asarray(mu, complex), np.asarray(beta, complex)
    e = np.exp(mu * dt)
    a0 = (e - 1) / mu                                         # int_0^dt e^{mu (dt - s)} ds
    a1 = (e - 1 - mu * dt) / (mu ** 2 * dt)                    # int_0^dt e^{mu (dt - s)} s / dt ds
    x = np.zeros((u.shape[0], mu.size), complex)
    y = np.zeros(u.shape)
    for j in range(1, t.size):
        if order == 0:
            x = e * x + a0 * u[:, j - 1, None]
        else:
            x = e * x + a0 * u[:, j - 1, None] + a1 * (u[:, j, None] - u[:, j - 1, None])
        y[:, j] = (x @ beta).real
    return y


def chebyshev(u, t, mu, beta, deg, fine=16):
    s = 2 * t / t[-1] - 1
    V = np.polynomial.chebyshev.chebvander(s, deg)
    c = np.linalg.lstsq(V, u.T, rcond=None)[0]                # (deg + 1, B)
    tf = np.linspace(0, t[-1], (t.size - 1) * fine + 1)
    uf = (np.polynomial.chebyshev.chebvander(2 * tf / t[-1] - 1, deg) @ c).T
    return lti.response(tf, uf, mu, beta)[:, ::fine]


def metrics(y, y_ref):
    n = y.shape[1]
    k = max(1, int(END * n))
    rel = lambda a, b: np.linalg.norm(a - b, axis=1) / np.maximum(np.linalg.norm(b, axis=1), 1e-300)
    return dict(rel=float(np.median(rel(y, y_ref))),
                first=float(np.median(rel(y[:, :k], y_ref[:, :k]))),
                last=float(np.median(rel(y[:, -k:], y_ref[:, -k:]))),
                maxerr=float(np.median(np.abs(y - y_ref).max(1) / np.abs(y_ref).max(1))))


def main():
    res = {}
    for system, (mu, beta) in SYSTEMS.items():
        for fam in FAMILIES:
            for n in (256, N_DEF, 4096):
                t = grid(n)
                u = inputs(fam, t, np.random.default_rng([n, FAMILIES.index(fam)]))
                y = lti.response(t, u, mu, beta)
                paths = {"fourier": lambda: fourier(u, t, mu, beta)}
                if n == N_DEF:
                    paths.update({"fourier_pad2": lambda: fourier(u, t, mu, beta, 2),
                                  "fourier_pad4": lambda: fourier(u, t, mu, beta, 4),
                                  "zoh": lambda: hold(u, t, mu, beta, 0), "foh": lambda: hold(u, t, mu, beta, 1),
                                  "cheb16": lambda: chebyshev(u, t, mu, beta, 16),
                                  "cheb64": lambda: chebyshev(u, t, mu, beta, 64),
                                  "cheb256": lambda: chebyshev(u, t, mu, beta, 256)})
                if n == 256 or n == 4096:
                    paths["zoh"] = lambda: hold(u, t, mu, beta, 0)
                for p, fn in paths.items():
                    res[f"{system}|{fam}|{n}|{p}"] = dict(system=system, family=fam, n=n, path=p, **metrics(fn(), y))
            print(system, fam, {p: f"{res[f'{system}|{fam}|{N_DEF}|{p}']['rel']:.1e}" for p in
                                ("fourier", "fourier_pad2", "zoh", "foh", "cheb64")}, flush=True)
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1))
    summarize(res)


def summarize(res):
    paths = ("fourier", "fourier_pad2", "fourier_pad4", "zoh", "foh", "cheb16", "cheb64", "cheb256")
    lines = ["# D1 oracle: error added by each input representation, true poles", "",
             f"Median over {N_PER} inputs per family; T = {T_END:g}. Truth: exact response to the piecewise-linear "
             "interpolant (`lfo.data.lti.response`).", ""]
    for system in SYSTEMS:
        lines += [f"**{system}, n = {N_DEF}: relative L2 error (last-5 % endpoint error)**", "",
                  "| family | " + " | ".join(paths) + " |", "|---|" + "---|" * len(paths)]
        for fam in FAMILIES:
            cells = [res[f"{system}|{fam}|{N_DEF}|{p}"] for p in paths]
            lines.append(f"| {fam} | " + " | ".join(f"{c['rel']:.1e} ({c['last']:.1e})" for c in cells) + " |")
        lines.append("")
    lines += ["**Fourier and ZOH paths against n (p6): relative L2 error; log-log slope 256 → 4096**", "",
              "| family | fourier 256 | 1024 | 4096 | slope | zoh 256 | 1024 | 4096 | slope |", "|---|---|---|---|---|---|---|---|---|"]
    for fam in FAMILIES:
        f = [res[f"p6|{fam}|{n}|fourier"]["rel"] for n in (256, N_DEF, 4096)]
        z = [res[f"p6|{fam}|{n}|zoh"]["rel"] for n in (256, N_DEF, 4096)]
        sl = lambda v: np.log(v[2] / v[0]) / np.log(16)
        lines.append(f"| {fam} | {f[0]:.1e} | {f[1]:.1e} | {f[2]:.1e} | {sl(f):.2f} | {z[0]:.1e} | {z[1]:.1e} | "
                     f"{z[2]:.1e} | {sl(z):.2f} |")
    (OUT.with_suffix(".md")).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
