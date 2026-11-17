"""(C) E5, fourth pass: new damped-Duffing data sets with diverse waveforms, to test the RQ4 trust rule out of sample.

The LNO data `1D_Duffing_c05` follow x'' + 0.5 x' + x + k3 x^3 = f with k3 = 1 (checked here: a solve_ivp
simulation reproduces stored training samples to 2.5e-4 (A = 0.05), 4.5e-3 (A = 3) and 1.2e-2 (A = 10), and a
test sample to 4.3e-3, relative L2). Small-signal poles: -0.25 +- 0.9682i.

New sets, same grid (T = 20.47, dt = 0.01, n = 2048), zero initial state, inputs of the four E1 families (step,
ramp, PRBS, smooth random; `e1_audit.make_inputs`, times scaled to this window), each scaled to a random peak |f|:

- `small`: peak |f| ~ U[0.02, 0.1], so peak |x| <= 0.23 and k3 x^2 / k <= 0.05 (nearly linear; a first version
  with |f| <= 0.5 reached |x| = 0.88, where the cubic term is 0.8 k x);
- `mixed`: peak |f| ~ U[0.05, 5];
- fifth pass, data seed 1 (bands chosen from a data-only scan of the best-linear-fit residual, before any training):
  - `medium`: peak |f| ~ U[0.1, 0.5], data residual just under the linearity gate (0.1);
  - `moderate`: peak |f| ~ U[0.2, 1], data residual above the gate.

Sixth pass, driven pendulum x'' + 0.5 x' + sin x = f (the LNO data `1D_Pendulum_c05`, checked here: a simulation
reproduces stored samples to 2.4e-4 (A = 0.05) ... 1.4e-2 (A = 10); same small-signal pole), data seed 2, bands from a
data-only gate scan: `pend_medium` peak |f| ~ U[0.1, 0.5] (gate residual 0.04), `pend_large` U[0.3, 1] (0.26; the
pendulum goes over the top).

Seventh pass, data seed 3, bands where the v2 and v3 gates disagree in a data-only scan: `wide` (Duffing, peak |f| ~
U[0.05, 0.6]) and `pend_wide` (pendulum, U[0.05, 1]).

Eighth pass, undamped Duffing x'' + x + x^3 = f (the LNO data `1D_Duffing_c0`, checked here: c = 0, k3 = 1 reproduce
stored samples to 6.6e-4 (A = 0.05) ... 1.2e-1 (A = 10); small-signal poles +-i, on the imaginary axis), data seed 4,
bands from a data-only gate scan: `c0_small` peak |f| ~ U[0.02, 0.1] (both gates pass), `c0_medium` U[0.05, 0.3]
(both fail, but the equivalent pole is within 6 % of i).

Splits 200 / 50 / 130 (train / vali / test); data seed 0 for `small` / `mixed`, 1 for `medium` / `moderate`. Stored in runs/C/e5_data/<name>.npz (git-ignored).

    python experiments/C/e5_probe/e5_data.py
"""
from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE.parent / "e1_audit"))
sys.path.insert(0, str(ROOT / "src"))
import e1_audit as E  # noqa: E402

C, K, K3 = 0.5, 1.0, 1.0
N_T, DT = 2048, 0.01
SETS = {"small": (0.02, 0.1, 0), "mixed": (0.05, 5.0, 0), "medium": (0.1, 0.5, 1), "moderate": (0.2, 1.0, 1),
        "pend_medium": (0.1, 0.5, 2), "pend_large": (0.3, 1.0, 2), "wide": (0.05, 0.6, 3), "pend_wide": (0.05, 1.0, 3),
        "c0_small": (0.02, 0.1, 4), "c0_medium": (0.05, 0.3, 4)}
SPLIT = (200, 50, 130)
OUT = ROOT / "runs" / "C" / "e5_data"


def grid():
    return np.arange(N_T) * DT


def system_of(name):
    """pend_*: driven pendulum; c0_*: undamped Duffing (c = 0); otherwise damped Duffing."""
    return "pendulum" if name.startswith("pend") else "duffing_c0" if name.startswith("c0_") else "duffing"


def simulate(f, system="duffing"):
    t = grid()
    spring = np.sin if system == "pendulum" else (lambda x: K * x + K3 * x ** 3)
    c = 0.0 if system == "duffing_c0" else C
    rhs = lambda s, z: [z[1], np.interp(s, t, f) - c * z[1] - spring(z[0])]
    return solve_ivp(rhs, (0, t[-1]), [0.0, 0.0], t_eval=t, rtol=1e-9, atol=1e-11, max_step=DT).y[0]


def inputs(n, rng):
    """E1 input families on this window: E1's times scale with T_END (20), so pass a rescaled grid."""
    t = grid()
    u = E.make_inputs(t * (E.T_END / t[-1]), n, rng)
    return u


def make(name):
    lo, hi, seed = SETS[name]
    rng = np.random.default_rng(seed)
    n = sum(SPLIT)
    f = inputs(n, rng)
    f = f / np.abs(f).max(1, keepdims=True) * rng.uniform(lo, hi, (n, 1))
    with ProcessPoolExecutor(16) as ex:
        system = system_of(name)
        u = np.array(list(ex.map(simulate, f, [system] * n, chunksize=8)))
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / f"{name}.npz", f=f, u=u, t=grid(), split=np.array(SPLIT))
    print(name, f.shape, "peak |f|", f"{np.abs(f).max(1).min():.2f}-{np.abs(f).max(1).max():.2f}",
          "peak |u|", f"{np.abs(u).max(1).min():.3f}-{np.abs(u).max(1).max():.3f}", flush=True)


if __name__ == "__main__":
    for s in sys.argv[1:] or ("small", "mixed"):
        make(s)
