"""(D2) A second nonlinear rollout system: forced 1D Cahn–Hilliard with initial fields (the d2e protocol).

d2b–d2e drew their rollout conclusions from one system, Burgers (dissipative). They were: developed-start training
matters most; the pole structure helps with unseen initial modes but gives no edge at large amplitudes or over long
rollouts; per-window errors saturate. Cahn–Hilliard has an unstable band (k < k_c = 5), and the field saturates
at +-1 (phase separation). This is a different regime for chained windows: small errors in the band are amplified
within a window.

**System.** `lfo.data.cahn_hilliard`, M = 0.1, kappa = 0.04, n_x = 64, the a16 forcing (scale 0.3), T = 5 windows
(n_t = 201), truth on a 32x finer grid. Initial fields as d2b (modes 1..8, amplitude ~ s0 / k, s0 log-uniform in
[0.03, 1]).

**Training** (d2c's developed-start protocol): both windows of 200 two-window trajectories (400 windows), 6000
steps, batch 25, 3 seeds.

**Models.** As d2b, but the WPOs allow Re mu > 0 (a16's `wpo`: Re cap 1.0, leak 0.05), since the band is
unstable:
- `wpo_ic`: IC path sharing the poles, width 23;
- `wpo_chan`: u0 as an input channel;
- `tfno_chan`: the a13 TFNO with two input channels.

**Evaluation sets** (as d2e):
- `id`: fresh windows;
- `amp`: s0 in [1, 2] (d2e used [1, 3]; the solver diverged on that set, whose initial fields reach |u| ~ 9:
  the explicit cubic term is too stiff there; found before any fit);
- `high`: initial field on modes 9..16 only;
- `roll4`: one forcing profile over [0, 20];
- `roll10`: 10 windows with forcing drawn per window.

**Checks before any fit.** All sets finite; max |u| 4.4 (train), 7.8 (`amp`). Smoke test: 5 steps per model,
finite 10-window rollouts; 61.1 k / 59.6 k / 59.1 k parameters.

**Predictions** (written before running). Each tests whether a Burgers finding carries over.

1. Unseen initial modes: the TFNO's ratio high / id is at least 2x `wpo_ic`'s (Burgers: 3.3 against 0.72).
2. Larger amplitudes: every model's ratio amp / id >= 2 in every seed.
3. Saturation: for every model, the mean per-window error in window 10 is at most 2x that in window 3.
4. No clear structural edge in rollout: `wpo_ic`'s mean `roll4` error is within +-10 % of the TFNO's.

    python experiments/D/d2_initial_conditions/d2g_ch_rollout.py      # results/d2g_ch_rollout.md / .json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import d2b_burgers_rollout as R  # noqa: E402
import d2e_ic_generalisation as E  # noqa: E402

M = R.M
from lfo.data import cahn_hilliard as ch  # noqa: E402
from lfo.models import WavenumberPoleOperator1d  # noqa: E402

MOB, KAPPA, SCALE = 0.1, 0.04, 0.3
KINDS, SEEDS, N10 = ("wpo_ic", "wpo_chan", "tfno_chan"), (0, 1, 2), 10
WPO_KW = dict(M.WPO_KW, re_max=1.0)
OUT = HERE / "results" / "d2g_ch_rollout.json"


def solve(fk, dtf, u0k):
    uk = ch.solve(fk, dtf, MOB, KAPPA, M.NX, stride=M.FINE, u0k=u0k)
    f = np.fft.irfft(fk[..., ::M.FINE], n=M.NX, axis=1)
    u = np.fft.irfft(uk, n=M.NX, axis=1)
    return torch.tensor(f), torch.tensor(u[..., 0]), torch.tensor(u)


def forcing(rng, n, T, n_t):
    fk, dtf = M.forcing(rng, n, T, n_t)
    return fk * SCALE, dtf


def sample(rng, n, T, n_t, modes=(1, 8), s_range=(0.03, 1.0)):
    fk, dtf = forcing(rng, n, T, n_t)
    return solve(fk, dtf, E.initial(rng, n, modes, s_range))


def long_set(rng, n):
    parts = [forcing(rng, n, M.T1, M.NT1) for _ in range(N10)]
    fk = np.concatenate([parts[0][0]] + [p[0][..., 1:] for p in parts[1:]], -1)
    return solve(fk, parts[0][1], E.initial(rng, n, (1, 8), (0.03, 1.0)))


def make_sets():
    w = M.NT1 - 1
    f, u0, u = sample(np.random.default_rng(10), 250, 2 * M.T1, 2 * w + 1)
    single = lambda d: (torch.cat([d[0][..., :w + 1], d[0][..., w:]]), torch.cat([d[1], d[2][..., w]]),
                        torch.cat([d[2][..., :w + 1], d[2][..., w:]]))
    return dict(train=single((f[:200], u0[:200], u[:200])),
                id=sample(np.random.default_rng(11), 100, M.T1, M.NT1),
                amp=sample(np.random.default_rng(21), 100, M.T1, M.NT1, s_range=(1.0, 2.0)),
                high=sample(np.random.default_rng(22), 100, M.T1, M.NT1, modes=(9, 16)),
                roll4=sample(np.random.default_rng(1), 50, 4 * M.T1, 4 * w + 1),
                roll10=long_set(np.random.default_rng(23), 50))


class WPOChanU(R.WPOChan):
    """d2b's channel WPO, without the stability constraint."""

    def __init__(self, t):
        nn.Module.__init__(self)
        self.net = WavenumberPoleOperator1d(t, **WPO_KW)
        self.net.lift = nn.Linear(2, self.net.lift.out_features, dtype=torch.float64)


class WPOICU(R.WPOIC):
    def __init__(self, t):
        nn.Module.__init__(self)
        self.net = WavenumberPoleOperator1d(t, **dict(WPO_KW, width=23), ic=True)


def build(kind, seed, t, device):
    torch.manual_seed(seed)
    return {"wpo_ic": lambda: WPOICU(t), "wpo_chan": lambda: WPOChanU(t), "tfno_chan": R.TFNOChan}[kind]().to(device)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sets = make_sets()
    t = np.linspace(0, M.T1, M.NT1)
    w = M.NT1 - 1
    out = json.loads(OUT.read_text()) if OUT.exists() else []
    for kind in KINDS:
        for seed in SEEDS:
            if any(r["model"] == kind and r["seed"] == seed for r in out):
                continue
            t0 = time.time()
            m = build(kind, seed, t, device)
            R.train(m, dict(train=sets["train"]), device, 30.0 if kind.startswith("wpo") else 1.0)
            m.eval()
            r = dict(model=kind, seed=seed, params=sum(p.numel() for p in m.parameters()))
            for s in ("id", "amp", "high"):
                f, u0, y = sets[s]
                r[s] = float(M.A.rel(R.predict(m, f, u0, device), y))
            f, u0, y = sets["roll4"]
            r["roll4"] = float(M.A.rel(E.rollout(m, f, u0, device, 4), y))
            f, u0, y = sets["roll10"]
            p = E.rollout(m, f, u0, device, N10)
            r["roll10"] = float(M.A.rel(p, y))
            r["roll10_per_window"] = [float(M.A.rel(p[..., i * w + 1:(i + 1) * w + 1], y[..., i * w + 1:(i + 1) * w + 1]))
                                      for i in range(N10)]
            r["seconds"] = round(time.time() - t0, 1)
            out.append(r)
            print({q: (round(v, 5) if isinstance(v, float) else v) for q, v in r.items() if q != "roll10_per_window"},
                  flush=True)
            OUT.parent.mkdir(exist_ok=True)
            OUT.write_text(json.dumps(out, indent=1))
            del m
            torch.cuda.empty_cache()
    mean = lambda rs, q: np.mean([r[q] for r in rs])
    rng_ = lambda xs: f"{min(xs):.2f}–{max(xs):.2f}"
    lines = ["# (D2) Forced Cahn–Hilliard with initial fields: generalisation and 10-window rollout", "",
             "| model | params | id | amp (ratio to id) | high (ratio to id) | roll4 | roll10 [0, 50] |",
             "|---|---|---|---|---|---|---|"]
    for kind in KINDS:
        rs = [r for r in out if r["model"] == kind]
        if rs:
            lines.append(f"| {kind} | {rs[0]['params']} | {mean(rs, 'id'):.4f} | {mean(rs, 'amp'):.4f} "
                         f"({rng_([r['amp'] / r['id'] for r in rs])}) | {mean(rs, 'high'):.4f} "
                         f"({rng_([r['high'] / r['id'] for r in rs])}) | {mean(rs, 'roll4'):.4f} | {mean(rs, 'roll10'):.4f} |")
    lines += ["", "Per-window error on roll10 (mean over seeds):", "",
              "| model | " + " | ".join(str(i + 1) for i in range(N10)) + " |", "|---|" + "---|" * N10]
    for kind in KINDS:
        rs = [r for r in out if r["model"] == kind]
        if rs:
            pw = np.mean([r["roll10_per_window"] for r in rs], 0)
            lines.append(f"| {kind} | " + " | ".join(f"{v:.3f}" for v in pw) + " |")
    (OUT.with_suffix(".md")).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
