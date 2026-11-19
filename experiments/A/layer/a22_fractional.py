"""(A) RQ4, second test: a dispersion relation that is not polynomial (fractional diffusion). Where does the MLP pay off?

a19 (`a19_mlp_band.py`) showed that, when lambda(k) is polynomial, the polynomial pole function wins in band and
extrapolates out of band, while the MLP fails out of band. This run uses lambda(k) = -D |k|^alpha with D = 0.15 and
alpha = 1.5 (fractional diffusion; |lambda| dt <= 0.24 at k = 16).

The best least-squares fit of {1, k^2, k^4} to -0.15 k^1.5 is 2.3 % off over k = 1..16 (equal weights). Fitted over
k = 1..8 it is 1.9 % off there and 71 % off over k = 9..16. These are representation floors computed before any
training.

**Setup** (as a19: n_x = 64, T = 5, truth on a 32x finer grid, 5 data seeds, joint fit with 8 starts and L-BFGS,
leaky cap, k_ref 16). Two training regimes:
- `full`: training and validation forced at k = 1..16; test forced at k = 1..16;
- `band`: training forced at k = 1..8; tests at k = 1..8 and k = 9..16 (as a19).

Oracle: a layer whose mu(k) is the exact lambda(k).

**Predictions** (written before running).

1. `full`: the MLP reaches <= 1.2x the oracle and a dispersion error <= 1 % in at least 4 of 5 seeds. The polynomial
   has a dispersion error >= 1.5 % and a test error above the MLP's, in at least 4 of 5 seeds.
2. `band`: both have a dispersion error > 10 % over k = 9..16 in at least 4 of 5 seeds; neither extrapolates.

    python experiments/A/layer/a22_fractional.py      # results/a22_fractional.md / .json
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
import a1_dispersion as A  # noqa: E402
import a19_mlp_band as R  # noqa: E402
from lfo.models.wavenumber import WavenumberPoleLayer1d  # noqa: E402

D_F, ALPHA = 0.15, 1.5
OUT = HERE / "results" / "a22_fractional.json"


def lam(k):
    return -D_F * np.abs(np.asarray(k, float)) ** ALPHA


_orig_poles = R.poles
R.poles = lambda system, k: (lam(k)[None].astype(complex), np.array([1.0])) if system == "frac" else _orig_poles(system, k)


class Exact(WavenumberPoleLayer1d):
    def mu(self, k):
        return torch.complex(-D_F * k.abs() ** ALPHA, torch.zeros_like(k))[None]


def make_data(seed, regime):
    rng = np.random.default_rng(200 + seed)
    hi = 16 if regime == "full" else 8
    d = dict(train=R.make_set("frac", rng, 200, 1, hi), vali=R.make_set("frac", rng, 50, 1, hi),
             test_in=R.make_set("frac", rng, 100, 1, hi))
    if regime == "band":
        d["test_out"] = R.make_set("frac", rng, 100, 9, 16)
    return d


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t = np.linspace(0, A.T_END, A.N_T)
    out = []
    for regime in ("full", "band"):
        for seed in range(5):
            data = make_data(seed, regime)
            orc = Exact(1, t, modes=A.MODES, beta_degree=1).to(device)
            with torch.no_grad():
                orc.beta.zero_()
                orc.beta[0, 0, 0] = 1.0
                floor = {s: float(A.rel(orc(data[s][0].to(device)), data[s][1].to(device))) for s in data if s.startswith("test")}
            for kind in ("poly", "mlp"):
                t0 = time.time()
                m = R.fit("frac", data, t, device, kind)
                with torch.no_grad():
                    ratio = {s: float(A.rel(m(data[s][0].to(device)), data[s][1].to(device))) / floor[s] for s in floor}
                ks_in = np.arange(1, 17) if regime == "full" else np.arange(1, 9)
                r = dict(regime=regime, seed=seed, kind=kind, ratio=ratio, disp_in=R.disp_err("frac", m, ks_in, device),
                         disp_out=R.disp_err("frac", m, np.arange(9, 17), device) if regime == "band" else None,
                         oracle=floor, seconds=round(time.time() - t0, 1))
                out.append(r)
                print(regime, seed, kind, {k: round(v, 2) for k, v in ratio.items()}, f"disp in {r['disp_in']:.2e}",
                      f"out {r['disp_out']:.2e}" if r["disp_out"] is not None else "", flush=True)
                OUT.write_text(json.dumps(out, indent=1))
    lines = ["# (A) RQ4: fractional diffusion λ = −0.15 |k|^1.5, polynomial against MLP μ(k)", "",
             "Range over data seeds 0–4. `full`: trained and tested on k = 1..16. `band`: trained on k ≤ 8, tested in and out "
             "of band.", "",
             "| regime | μ(k) | test / oracle (in) | test / oracle (out) | dispersion error (in) | dispersion error k = 9..16 |",
             "|---|---|---|---|---|---|"]
    for regime in ("full", "band"):
        for kind in ("poly", "mlp"):
            rs = [r for r in out if r["regime"] == regime and r["kind"] == kind]
            g = lambda v, f: f"{min(v):{f}}–{max(v):{f}}"
            lines.append(f"| {regime} | {kind} | {g([r['ratio']['test_in'] for r in rs], '.2f')} | "
                         + (g([r['ratio']['test_out'] for r in rs], '.2f') if regime == "band" else "–") + " | "
                         + f"{g([r['disp_in'] for r in rs], '.1e')} | "
                         + (g([r['disp_out'] for r in rs], '.1e') if regime == "band" else "–") + " |")
    (OUT.with_suffix(".md")).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
