"""(A) Deep operator on forced 1D Cahn-Hilliard: can the WPO and its probe recover an unstable band (RQ3)?

**System** (`lfo.data.cahn_hilliard`): u_t = M d_xx (u^3 - u - kappa d_xx u) + f, periodic, zero initial state,
M = 0.1, kappa = 0.04. Linearized at u = 0: lambda(k) = M k^2 (1 - kappa k^2), positive for k < k_c = 5, largest at
k_max = 3.54 (lambda = 0.625). Over T = 5 the fastest mode grows by e^{3.1} = 22x before the cubic term saturates
it.

**Data.** As a9 (n_x = 64, T = 5, n_t = 201, 4 forced modes k in 1..8 with E1-family profiles, truth on a 32x finer
time grid; splits 200 / 50 / 100; extrapolation set on T = 10), with the forcing scale s log-uniform in
[0.009, 0.3]. Measured before any training: the part of u not explained by the linear response is 0.1 % / 3 % /
42 % at the 10th / 50th / 90th percentile, up to 217 %.

**Models** (3 seeds, 12000 steps, batch 25, relative L2; configurations from a11 / a13, no new sweep):

- `wpo`: the a11 WPO (lr 3e-3, pole ratio 30) without `stable`, so Re mu may be positive: Re cap 1.0 with leak 0.05;
- `wpo_stable`: the same with `stable=True` (Re mu <= 0), a control that cannot represent growth;
- `tfno`: the a13 TFNO (Tucker, n_modes (32, 64), 59 k parameters, lr 3e-3).

**Probe.** The a9 probe at eps = 0.003 (inside the training scales), per-mode identification, resolution and
observability cuts. Re mu is fitted by weighted least squares on {1, k^2, k^4}. From the fit:
k_c = sqrt(-a1 / a2) and k_max = k_c / sqrt(2). Also checked: the sign of Re mu per kept mode against lambda(k).

**Method check, measured before training:** the solver's own probe keeps k = 3, 4, 6, 7 (k = 1, 2 fail the
observability cut, k >= 8 the resolution cut), gets every sign right and gives k_c = 5.107 (2.1 %), k_max = 3.61.
The per-mode poles at k = 1, 2 are exact (0.096, 0.336) but cut.

**Predictions** (written before running).

1. `wpo`: in at least 2 of 3 seeds, the probe's k_c is within 5 % of 5, and the sign of Re mu is right for every
   kept mode.
2. `wpo_stable`: its test error is above `wpo`'s in every seed.
3. `wpo`'s mean test error is at or below the TFNO's.
4. TFNO: in at least 2 of 3 seeds, its probe's k_c is within 10 % of 5.
5. Extrapolation ((5, 10] error / test error): `wpo` below the TFNO in every seed.

    python experiments/A/deep/a16_cahn_hilliard.py      # results/a16_cahn_hilliard.md / .json
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
sys.path.insert(0, str(HERE.parent / "layer"))
import a2_twostage as B  # noqa: E402
import a9_burgers as M  # noqa: E402
import a11_fair as F  # noqa: E402
import a13_baselines as B13  # noqa: E402
from lfo.data import cahn_hilliard as ch  # noqa: E402
from lfo.models import WavenumberPoleOperator1d  # noqa: E402

MOB, KAPPA, SCALE, EPS = 0.1, 0.04, 0.3, 0.003
KC, KMAX = 1 / math.sqrt(KAPPA), 1 / math.sqrt(2 * KAPPA)
VARIANTS = {"wpo": dict(lr=3e-3, plr=30.0), "wpo_stable": dict(lr=3e-3, plr=30.0), "tfno": dict(lr=3e-3)}
WPO_KW = dict(M.WPO_KW, re_max=1.0)
OUT = HERE / "results" / "a16_cahn_hilliard.json"
CKPT = M.ROOT / "runs" / "A" / "a16"


def fields(fk, dtf, stride=M.FINE):
    uk = ch.solve(fk, dtf, MOB, KAPPA, M.NX, stride=stride)
    return torch.tensor(np.fft.irfft(fk[..., ::stride], n=M.NX, axis=1)), torch.tensor(np.fft.irfft(uk, n=M.NX, axis=1))


def forcing(rng, n, T, n_t):
    fk, dtf = M.forcing(rng, n, T, n_t)
    return fk * SCALE, dtf


def make_data():
    f, u = fields(*forcing(np.random.default_rng(0), sum(M.SPLIT), M.T1, M.NT1))
    s = np.cumsum([0, *M.SPLIT])
    data = {k: (f[a:b], u[a:b]) for k, a, b in zip(("train", "vali", "test"), s[:-1], s[1:])}
    data["ext"] = fields(*forcing(np.random.default_rng(1), M.N_EXT, M.T2, M.NT2))
    return data


def probe(G, fk, device):
    f_plus, g_plus = G(fk)
    _, g_minus = G(-fk)
    dt = M.T1 / (M.NT1 - 1)
    modes = B.per_mode({"train": (f_plus, 0.5 * (g_plus - g_minus))}, dt, device)
    kept = [m for m in modes if abs(complex(*m["mu"])) * dt <= M.CUT and max(map(abs, m["mu"])) * M.T1 >= M.OBS]
    res = dict(modes=modes, kept=[m["k"] for m in kept])
    if len(kept) < 3:
        return dict(res, k_c=float("nan"), k_max=float("nan"), signs_ok=False, disp_err=float("inf"))
    ks = np.array([m["k"] for m in kept], float)
    mu = np.array([complex(*m["mu"]) for m in kept])
    w = 1 / np.array([max(m["resid"], 1e-6) for m in kept]) ** 2
    V = np.stack([ks ** 0, ks ** 2, ks ** 4], 1)
    sw = np.sqrt(w)
    a, *_ = np.linalg.lstsq(V * sw[:, None], mu.real * sw, rcond=None)
    kc = math.sqrt(-a[1] / a[2]) if a[1] > 0 > a[2] else float("nan")
    true = ch.growth_rate(ks, MOB, KAPPA)
    return dict(res, coef=[float(x) for x in a], k_c=kc, k_max=kc / math.sqrt(2),
                signs_ok=bool(np.all(np.sign(mu.real) == np.sign(true))),
                disp_err=float(np.linalg.norm(mu - true) / np.linalg.norm(true)),
                resid_median=float(np.median([m["resid"] for m in kept])))


def build(kind, seed, t, device):
    torch.manual_seed(seed)
    if kind.startswith("wpo"):
        return WavenumberPoleOperator1d(t, **dict(WPO_KW, stable=kind == "wpo_stable")).to(device)
    return B13.build("tfno", seed, device)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CKPT.mkdir(parents=True, exist_ok=True)
    data = make_data()
    t = np.linspace(0, M.T1, M.NT1)
    eps0, M.EPS = M.EPS, EPS                         # the a9 probe inputs at this problem's amplitude
    pfk, pdt = M.probe_inputs()
    M.EPS = eps0
    out = json.loads(OUT.read_text()) if OUT.exists() else dict(runs=[])
    if "solver_probe" not in out:
        out["solver_probe"] = probe(lambda spec: fields(spec, pdt), pfk, device)
        sp = out["solver_probe"]
        print("solver probe", f"k_c {sp['k_c']:.3f} signs {sp['signs_ok']} disp {sp['disp_err']:.2e} kept {sp['kept']}", flush=True)
    n_ext = M.NT1 - 1
    for kind, cfg in VARIANTS.items():
        for seed in M.SEEDS:
            if any(r["model"] == kind and r["seed"] == seed for r in out["runs"]):
                continue
            t1 = time.time()
            m = build(kind, seed, t, device)
            F.train(m, data, device, steps=F.FINAL_STEPS, **cfg)
            torch.save(m.state_dict(), CKPT / f"{kind}_{seed}.pt")
            m.eval()
            xe, ye = data["ext"]
            pe = M.batched(m, xe.to(device)).cpu().to(torch.float64)

            def G(spec):
                f = torch.tensor(np.fft.irfft(spec[..., ::M.FINE], n=M.NX, axis=1))
                return f, M.batched(m, f.to(device)).cpu().to(torch.float64)

            r = dict(model=kind, seed=seed, params=M.n_params(m), train=F.err(m, data, "train", device),
                     test=F.err(m, data, "test", device),
                     ext_late=float(M.A.rel(pe[..., n_ext + 1:], ye[..., n_ext + 1:])), probe=probe(G, pfk, device),
                     seconds=round(time.time() - t1, 1))
            out["runs"].append(r)
            p = r["probe"]
            print(kind, seed, f"train {r['train']:.3e} test {r['test']:.3e} ext {r['ext_late']:.3e}",
                  f"probe k_c {p['k_c']:.3f} signs {p['signs_ok']} disp {p['disp_err']:.2e} kept {p['kept']}",
                  f"{r['seconds']} s", flush=True)
            OUT.write_text(json.dumps(out, indent=1, default=float))
            del m
            torch.cuda.empty_cache()
    OUT.write_text(json.dumps(out, indent=1, default=float))
    summarize(out)


def summarize(out):
    sp = out["solver_probe"]
    lines = ["# (A) Deep operator on forced 1D Cahn–Hilliard (M = 0.1, κ = 0.04; k_c = 5, k_max = 3.54)", "",
             f"Solver probe (method check): k_c = {sp['k_c']:.3f}, signs right: {sp['signs_ok']}, dispersion error "
             f"{sp['disp_err']:.1e}, kept k = {sp['kept']}.", "",
             "| model | seed | train | test | (5, 10] / test | probe k_c | k_max | signs right | dispersion error | kept k |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in out["runs"]:
        p = r["probe"]
        lines.append(f"| {r['model']} | {r['seed']} | {r['train']:.3e} | {r['test']:.3e} | {r['ext_late'] / r['test']:.2f} | "
                     f"{p['k_c']:.3f} | {p['k_max']:.3f} | {p['signs_ok']} | {p['disp_err']:.1e} | {p['kept']} |")
    (OUT.with_suffix(".md")).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
