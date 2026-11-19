"""(D) Robin boundaries, second pass: find beta by a profile over a grid, then release it.

In d3d (`d3d_robin.py`) the joint fit with a learned beta collapsed to beta ~ 0 in all 5 seeds (truth 2), 6–8x the
oracle, while the same model with beta given was at the oracle (1.00–1.01, D within 0.8 %). The model is right; the
joint fit does not find beta. With the poles and laws starting at random values, beta is pushed to 0 early. With
beta = softplus(rho) its gradient then vanishes there.

**Method** (`profile`). For each beta on a log grid that does not contain the truth, {0.3, 0.6, 1.2, 2.4, 4.8, 9.6},
fit the rest with beta fixed (2 starts each, the d3d fit). Take the grid point with the lowest validation error,
release beta and continue with L-BFGS (300 iterations). Same data, oracle and seeds as d3d.

**Predictions** (written before running).

1. In every seed: beta within 2 %, D within 1 %, test error <= 1.2x the oracle, and each channel law within 2 % on
   modes m <= 8.
2. The profile's validation error is lowest at the grid point 2.4 (nearest the truth on a log scale) in every seed.

    python experiments/D/d3_boundary/d3e_robin_profile.py      # results/d3e_robin_profile.md / .json
"""
from __future__ import annotations

import copy
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import d3d_robin as R  # noqa: E402

A = R.A
GRID = (0.3, 0.6, 1.2, 2.4, 4.8, 9.6)
OUT = HERE / "results" / "d3e_robin_profile.json"


def lbfgs(m, ftr, btr, ytr):
    opt = torch.optim.LBFGS([p for p in m.parameters() if p.requires_grad], max_iter=300, line_search_fn="strong_wolfe",
                            tolerance_grad=1e-12, tolerance_change=1e-15, history_size=50)

    def closure():
        opt.zero_grad()
        loss = A.rel(m(ftr, btr), ytr)
        if torch.isfinite(loss):
            loss.backward()
        return loss

    try:
        opt.step(closure)
    except (RuntimeError, IndexError):
        pass


def val(m, d):
    with torch.no_grad():
        try:
            v = float(A.rel(m(d[0], d[1]), d[2]))
        except RuntimeError:
            return float("inf")
    return v if math.isfinite(v) else float("inf")


def fit(data, t, device):
    tr, va = ([z.to(device) for z in data[s]] for s in ("train", "vali"))
    prof, best = {}, None
    for bg in GRID:
        for s in range(2):
            torch.manual_seed(s)
            m = R.RobinLayer(t, bg, learn=False).to(device)
            with torch.no_grad():
                m.core.mu_re.normal_(0, 3.0), m.core.mu_im.normal_(0, 3.0)
            lbfgs(m, *tr)
            v = val(m, va)
            prof[bg] = min(prof.get(bg, float("inf")), v)
            if best is None or v < best[0]:
                best = (v, bg, m)
    m = copy.deepcopy(best[2])
    m.rho.requires_grad_(True)
    lbfgs(m, *tr)
    if val(m, va) > best[0]:                     # release made it worse: keep the grid fit
        m = best[2]
    return m, prof, best[1]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t = np.linspace(0, R.T, R.NT)
    out = json.loads(OUT.read_text()) if OUT.exists() else []
    for seed in range(5):
        if any(r["seed"] == seed for r in out):
            continue
        data = R.make_data(seed)
        floor = R.oracle(data, t, device)
        fte, bte, yte = (z.to(device) for z in data["test"])
        t0 = time.time()
        m, prof, bg = fit(data, t, device)
        with torch.no_grad():
            test = float(A.rel(m(fte, bte), yte))
            k = R.roots(m.beta)
            lw = m.law(k).cpu().numpy()
            k = k.cpu().numpy()
        tl = R.true_laws(k)
        r = dict(seed=seed, oracle=floor, ratio=test / floor, beta=float(m.beta), beta_grid=bg,
                 D=float(-m.core.mu_re[0, 1] / m.kref ** 2), profile={str(b): v for b, v in prof.items()},
                 law_err8=[float(np.linalg.norm(lw[j, :8] - tl[j, :8]) / np.linalg.norm(tl[j, :8])) for j in range(2)],
                 seconds=round(time.time() - t0, 1))
        out.append(r)
        print({q: (round(v, 5) if isinstance(v, float) else v) for q, v in r.items()}, flush=True)
        OUT.parent.mkdir(exist_ok=True)
        OUT.write_text(json.dumps(out, indent=1))
    lines = ["# (D) Robin boundary: beta by a profile over a grid, then released", "",
             f"Truth beta = {R.BETA}, D = {R.DIF}. Grid {GRID}.", "",
             "| seed | test / oracle | grid pick | β̂ | D̂ | law errors m ≤ 8 | profile validation error by grid β |",
             "|---|---|---|---|---|---|---|"]
    for r in out:
        lines.append(f"| {r['seed']} | {r['ratio']:.2f} | {r['beta_grid']} | {r['beta']:.4f} | {r['D']:.5f} | "
                     f"{r['law_err8'][0]:.1e}, {r['law_err8'][1]:.1e} | "
                     + ", ".join(f"{float(v):.1e}" for v in r["profile"].values()) + " |")
    (OUT.with_suffix(".md")).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
