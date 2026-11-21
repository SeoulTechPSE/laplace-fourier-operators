"""(C) Theory, Proposition 8 (`H1_H2.md`, Section 6): a damped mode is a Cauchy mixture of less damped modes.

e^{(-a + i w) t} = int c_{a-a'}(nu) e^{(-a' + i (w + nu)) t} d nu for t >= 0, with c_g the Cauchy density of width g
(because e^{-g t} is its characteristic function). So K modes with the smaller decay rate a' and spread frequencies
approximate one mode with rate a on a window [0, T], with an error that falls as K grows. With spare slots, a decay
rate is therefore identifiable only up to such substitutions, and only toward smaller rates: frequencies cannot
stand in for faster decay than the slots have (a' > a gives a lower bound on the error, see `over`).

Check: target e^{-a t} cos(w t), a = 0.25, w = 0.9682 (damped Duffing / pendulum small-signal pole), on the E5
probe window t in (0, 8] (h_1 ... h_799) and on the full window (0, 20.47]. For K = 1 ... 16 modes of rate a' = r a,
r in {0.2, 0.4, 0.6, 0.8, 1.2, 1.5}, frequencies and real cos / sin weights are fitted by least squares (frequencies
by a few starts of a bounded optimizer, warm-started from Cauchy quantiles); reported: the relative L2 error and
the order-2 matrix-pencil pole of the best K-mode sum (what the E5 probe would read).

    python experiments/C/theory/h8_dephase.py      # writes results/h8_dephase.md / .json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares, nnls

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "e5_probe"))
import e5_probe as P  # noqa: E402

A, W, DT = 0.25, 0.9682, 0.01
RATIOS = (0.2, 0.4, 0.6, 0.8, 1.2, 1.5)
KS = (1, 2, 3, 4, 6, 8, 12, 16)
WINDOWS = {"probe (0, 8]": 799, "full (0, 20.47]": 2047}
OUT = HERE / "results" / "h8_dephase.json"


def fit(t, y, a2, K):
    """Best sum of K modes e^{-a2 t} (c cos + s sin)(w_j t): frequencies by nonlinear LS, weights by linear LS."""
    def basis(w):
        e = np.exp(-a2 * t)[:, None]
        return np.concatenate([e * np.cos(np.outer(t, w)), e * np.sin(np.outer(t, w))], 1)

    def resid(w):
        B = basis(w)
        c = np.linalg.lstsq(B, y, rcond=None)[0]
        return B @ c - y

    g = max(A - a2, 1e-3)
    starts = [np.clip(W + g * np.tan(np.pi * ((np.arange(K) + 0.5) / K - 0.5) * s), 1e-3, 3 * W - 1e-3)
              for s in (0.5, 0.8, 0.95)]
    best = None
    for w0 in starts:
        r = least_squares(resid, w0, bounds=(0, 3 * W))
        if best is None or r.cost < best.cost:
            best = r
    B = basis(best.x)
    c = np.linalg.lstsq(B, y, rcond=None)[0]
    y_hat = B @ c
    mass = float(np.hypot(c[:K], c[K:]).sum())             # sum of mode amplitudes (target amplitude: 1)
    return float(np.linalg.norm(y_hat - y) / np.linalg.norm(y)), y_hat, best.x, mass


def positive_fit(t, y, a2, n_grid=400):
    """Best positive mixture: non-negative weights on e^{-a2 t} cos(w t), w on a grid in (0, 3 W] (NNLS)."""
    w = np.linspace(3 * W / n_grid, 3 * W, n_grid)
    B = np.exp(-a2 * t)[:, None] * np.cos(np.outer(t, w))
    c, _ = nnls(B, y, maxiter=50 * n_grid)
    return float(np.linalg.norm(B @ c - y) / np.linalg.norm(y)), float(c.sum()), int((c > 1e-6 * c.max()).sum())


def main():
    res = {}
    for wname, m in WINDOWS.items():
        t = np.arange(1, m + 1) * DT
        y = np.exp(-A * t) * np.cos(W * t)
        for r in RATIOS:
            for K in KS:
                err, y_hat, w, mass = fit(t, y, r * A, K)
                z = P.pencil_poles(y_hat, DT, 2)
                pole = complex(z[0]) if z else complex("nan")
                res[f"{wname}|{r}|{K}"] = dict(window=wname, ratio=r, K=K, err=err, pole=[pole.real, pole.imag],
                                              pole_err=abs(pole - complex(-A, W)) / abs(complex(-A, W)),
                                              freq_spread=float(np.std(w) / W) if K > 1 else 0.0, mass=mass)
                print(wname, r, K, f"err {err:.2e} pole {pole:.3f}", flush=True)
            pe, pm, pn = positive_fit(t, y, r * A)
            res[f"{wname}|{r}|pos"] = dict(window=wname, ratio=r, err=pe, mass=pm, n_active=pn)
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1))
    summarize(res)


def summarize(res):
    lines = ["# Proposition 8: a damped mode as a sum of K modes with a different decay rate", "",
             f"Target e^(-{A} t) cos({W} t); K modes with rate a' = r·{A}, frequencies and weights fitted. Cell: relative "
             "L2 error / relative error of the order-2 pencil pole of the fitted sum (what the E5 probe reads) / weight "
             "mass Σ|c_j| (target: 1).", ""]
    for wname in WINDOWS:
        lines += [f"**Window {wname}**", "", "| r = a'/a | " + " | ".join(f"K = {k}" for k in KS) + " |",
                  "|---|" + "---|" * len(KS)]
        for r in RATIOS:
            cells = [res[f"{wname}|{r}|{k}"] for k in KS]
            lines.append(f"| {r} | " + " | ".join(f"{c['err']:.1e} / {c['pole_err']:.2f} / {c['mass']:.3g}" for c in cells) + " |")
        lines += ["", "Positive mixture (non-negative weights, 400 frequencies in (0, 3ω], NNLS): relative L2 error / weight "
                  "mass / active modes", "", "| r = a'/a | error | mass | active |", "|---|---|---|---|"]
        for r in RATIOS:
            c = res[f"{wname}|{r}|pos"]
            lines.append(f"| {r} | {c['err']:.1e} | {c['mass']:.3g} | {c['n_active']} |")
        lines.append("")
    (OUT.with_suffix(".md")).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
