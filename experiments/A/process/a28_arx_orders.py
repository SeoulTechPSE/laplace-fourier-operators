"""(A) Why did ARX fail on the CSTR in a26, even without noise? Orders and a small-amplitude data gate.

In a26 (`a26_cstr_closed_loop.py`) the equation-error ARX(2, 2) returned the wrong poles in every regime, noise-free
open loop included, while output-error fits (linear pair, WPO probe) were within about 1 %. Two candidate causes:
the order is too low for a sampled, nonlinear plant; or the large, nonlinear samples dominate the equation-error
least squares. This run separates them on the a26 data (same regimes, same seed):

- orders na = nb in {2, 4, 6} (regressors y_{t-1..t-na}, u_{t-1..t-nb});
- training on all 200 samples, or only on the quarter with the smallest input rms (the "gate", as in (C)).

The reported pole is the continuous-time pole mu = log z / dt with Im mu > 0.05 closest to the origin (the slowest
oscillatory mode), or none.

**Predictions** (written before running).

1. Noise-free open loop: with the gate, some order gives the pair within 5 %; without the gate, every order is more
   than 10 % off.
2. `closed_noisy`: with the gate, every order is more than 10 % off (lower signal-to-noise on the small samples).

    python experiments/A/process/a28_arx_orders.py      # results/a28_arx_orders.md / .json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import a23_cstr as C  # noqa: E402
import a26_cstr_closed_loop as L  # noqa: E402

ORDERS = (2, 4, 6)
OUT = HERE / "results" / "a28_arx_orders.json"


def arx(u, y, n):
    rows, tgt = [], []
    for t in range(n, y.shape[1]):
        rows.append(np.concatenate([y[:, t - n:t][:, ::-1], u[:, t - n:t][:, ::-1]], 1))
        tgt.append(y[:, t])
    coef, *_ = np.linalg.lstsq(np.concatenate(rows), np.concatenate(tgt), rcond=None)
    z = np.roots(np.concatenate([[1.0], -coef[:n]])).astype(complex)
    mu = np.log(z) / C.DT
    osc = [m for m in mu if m.imag > 0.05]
    return complex(min(osc, key=abs)) if osc else None


def main():
    regimes, _ = L.make_regimes()
    out = []
    for name, data in regimes.items():
        u, y = (a.numpy() for a in data["train"])
        gate = np.argsort(np.sqrt((u ** 2).mean(1)))[: len(u) // 4]
        for gated in (False, True):
            uu, yy = (u[gate], y[gate]) if gated else (u, y)
            for n in ORDERS:
                mu = arx(uu, yy, n)
                e = L.pole_err(mu) if mu is not None else float("inf")
                out.append(dict(regime=name, gated=gated, order=n, pole=None if mu is None else [mu.real, mu.imag], err=e))
                print(name, "gated" if gated else "all", n, mu, f"{e:.2e}", flush=True)
    OUT.write_text(json.dumps(out, indent=1))
    lines = ["# (A) ARX on the a26 CSTR data: orders and a small-amplitude gate", "",
             f"Jacobian pair {C.MU_TRUE.real:.4f} ± {C.MU_TRUE.imag:.4f} i. Relative pole error; '–' = no oscillatory pole.", "",
             "| regime | training samples | ARX(2, 2) | ARX(4, 4) | ARX(6, 6) |", "|---|---|---|---|---|"]
    for name in regimes:
        for gated in (False, True):
            rs = {r["order"]: r for r in out if r["regime"] == name and r["gated"] == gated}
            cell = lambda r: "–" if r["pole"] is None else f"{r['err']:.1e}"
            lines.append(f"| {name} | {'smallest quarter' if gated else 'all'} | " + " | ".join(cell(rs[n]) for n in ORDERS) + " |")
    (OUT.with_suffix(".md")).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
