"""(A) Deep operator in 2D: forced 2D Cahn-Hilliard, WPO-2D against a matched TFNO (RQ3 in the plan's setting).

**System** (`lfo.data.cahn_hilliard.solve2d`): u_t = M lap (u^3 - u - kappa lap u) + f on a periodic 32 x 32 grid,
zero initial state, M = 0.1, kappa = 0.04. Linearised: lambda(k) = M |k|^2 (1 - kappa |k|^2), unstable for
|k| < k_c = 5, fastest at k_max = 3.54.

**Data.** T = 5, n_t = 201; truth on a 32x finer time grid. Each sample is forced at 4 wavevectors from the half
plane of |k_x|, k_y <= 6, with amplitude s / |k| and E1-family time profiles; s log-uniform in [0.009, 0.3] (as a16).
Splits 200 / 50 / 100.

**Models** (3 seeds; Adam with cosine decay, batch 10; relative L2):

- WPO-2D: `lfo.models.WavenumberPoleOperator2d`, width 16, 4 layers, 4 poles. Re mu is isotropic in |k|^2 (3
  coefficients), with advection; modes 9 (|k_x|, k_y <= 8); Re cap 1.0 with leak 0.05; k_ref 8. lr 3e-3 with
  pole ratio 30 (a11).
- TFNO: neuraloperator FNO on (x, y, t), Tucker, n_modes (16, 16, 64), domain padding 0.25 in t, rank chosen
  for the parameter count closest to the WPO's. lr 3e-3 (a13).

Timing run (before any training, `--time`): WPO-2D 18,737 parameters, 0.42 s/step; TFNO 14,965 parameters (the
closest rank among 0.002-0.05; 20 % fewer), 0.17 s/step. Both get 6000 steps (about 3 h in total for 3 seeds each).

**Method check, measured before training:** the solver's own probe keeps (3,0), (4,0), (6,0), (2,2), (3,3), (4,4),
(5,5), gets every sign right and gives k_c = 5.003.

**Probe.** Single-wavevector inputs at k = (k, 0) and (k, k), k = 1..6 (8 inputs per wavevector, eps = 0.003),
odd part, per-mode identification, the a5 cuts with T = 5. Re mu fitted on {1, |k|^2, |k|^4}, giving k_c. Also
the isotropy check: Re mu at (k, 0) against (k, k) is compared through |k|.

**Predictions** (written before running).

1. WPO-2D: probe k_c within 5 % of 5 and every kept sign right, in at least 2 of 3 seeds.
2. WPO-2D mean test error at or below the TFNO's.
3. TFNO: probe k_c within 10 % in at most 1 of 3 seeds (spurious out-of-band poles, as in 1D).

    python experiments/A/deep/a17_ch2d.py [--time]      # results/a17_ch2d.md / .json
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
import a1_dispersion as A  # noqa: E402
import a2_twostage as B  # noqa: E402
import a11_fair as F  # noqa: E402
from lfo.data import cahn_hilliard as ch  # noqa: E402
from lfo.models import WavenumberPoleOperator2d  # noqa: E402

N, T, NT, FINE = 32, 5.0, 201, 32
MOB, KAPPA, S_RANGE, BANDMAX, EPS = 0.1, 0.04, (0.009, 0.3), 6, 0.003
SPLIT, SEEDS, BATCH, LR, PLR = (200, 50, 100), (0, 1, 2), 10, 3e-3, 30.0
STEPS = 6000
H = [(kx, ky) for ky in range(0, BANDMAX + 1) for kx in range(-BANDMAX, BANDMAX + 1) if ky > 0 or kx > 0]
PROBE_K = [(k, 0) for k in range(1, 7)] + [(k, k) for k in range(1, 7)]
WPO_KW = dict(width=16, n_layers=4, n_poles=4, modes=9, re_max=1.0, re_leak=0.05, k_ref=8.0)
OUT = HERE / "results" / "a17_ch2d.json"
CKPT = F.M.ROOT / "runs" / "A" / "a17"


def lam(kx, ky):
    k2 = np.asarray(kx, float) ** 2 + np.asarray(ky, float) ** 2
    return MOB * k2 * (1 - KAPPA * k2)


def solve_fields(Fk, dtf):
    Uk = ch.solve2d(Fk, dtf, MOB, KAPPA, N, stride=FINE)
    return (torch.tensor(np.fft.irfft2(Fk[..., ::FINE], s=(N, N), axes=(1, 2))),
            torch.tensor(np.fft.irfft2(Uk, s=(N, N), axes=(1, 2))))


def spectrum(ks, amps, profs, nt_f):
    """Fine-grid rfft2 forcing from per-sample lists of wavevectors, complex amplitudes and profiles."""
    n = len(ks)
    Fk = np.zeros((n, N, N // 2 + 1, nt_f), complex)
    for i in range(n):
        for (kx, ky), a, p in zip(ks[i], amps[i], profs[i]):
            Fk[i, kx % N, ky] += a * p
            if ky == 0:                                             # conjugate partner on the k_y = 0 row
                Fk[i, (-kx) % N, 0] += np.conj(a) * p
    return Fk


def make_data():
    rng = np.random.default_rng(0)
    n = sum(SPLIT)
    tf = np.linspace(0, T, (NT - 1) * FINE + 1)
    s = np.exp(rng.uniform(*np.log(S_RANGE), size=n))
    profs = np.stack([A.profiles(tf, rng, n) for _ in range(4)], 1)          # (n, 4, n_tf)
    idx = rng.integers(len(H), size=(n, 4))
    ks = [[H[j] for j in row] for row in idx]
    amps = [[(rng.normal() + 1j * rng.normal()) / math.hypot(*k) * N * N / 2 * s[i] for k in ks[i]] for i in range(n)]
    f, u = solve_fields(spectrum(ks, amps, profs, tf.size), tf[1] - tf[0])
    c = np.cumsum([0, *SPLIT])
    return {k: (f[a:b], u[a:b]) for k, a, b in zip(("train", "vali", "test"), c[:-1], c[1:])}


def probe_inputs():
    rng = np.random.default_rng(2)
    tf = np.linspace(0, T, (NT - 1) * FINE + 1)
    ks, amps, profs = [], [], []
    for k in PROBE_K:
        p = A.profiles(tf, rng, 8)
        for j in range(8):
            ks.append([k]), amps.append([EPS / math.hypot(*k) * N * N / 2 * np.exp(2j * np.pi * rng.uniform())])
            profs.append([p[j]])
    return spectrum(ks, amps, profs, tf.size), tf[1] - tf[0]


class TFNO3(torch.nn.Module):
    def __init__(self, rank):
        super().__init__()
        from neuralop.models import FNO
        self.net = FNO(n_modes=(16, 16, 64), in_channels=1, out_channels=1, hidden_channels=16, n_layers=4,
                       domain_padding=[0.0, 0.0, 0.25], factorization="tucker", rank=rank)

    def forward(self, f):
        return self.net(f.float()[:, None])[:, 0].to(f.dtype)


def build(kind, seed, t, device):
    torch.manual_seed(seed)
    if kind == "wpo":
        return WavenumberPoleOperator2d(t, **WPO_KW).to(device)
    target = F.M.n_params(WavenumberPoleOperator2d(t, **WPO_KW))
    rank = min((0.002, 0.005, 0.01, 0.02, 0.05), key=lambda r: abs(F.M.n_params(TFNO3(r)) - target))
    torch.manual_seed(seed)
    return TFNO3(rank).to(device)


def batched(m, x, device, bs=5):
    with torch.no_grad():
        return torch.cat([m(x[i:i + bs].to(device)).cpu().to(torch.float64) for i in range(0, len(x), bs)])


def train(m, data, device, steps, lr, plr=1.0):
    xtr, ytr = data["train"]
    is_pole = lambda n: n.endswith("mu_re") or n.endswith("mu_im") or n.endswith("adv")
    groups = [dict(params=[p for n, p in m.named_parameters() if not is_pole(n)], lr=lr)]
    pole = [p for n, p in m.named_parameters() if is_pole(n)]
    if pole:
        groups.append(dict(params=pole, lr=lr * plr))
    opt = torch.optim.Adam(groups)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    g = torch.Generator().manual_seed(0)
    for _ in range(steps):
        idx = torch.randint(0, xtr.shape[0], (BATCH,), generator=g)
        opt.zero_grad()
        loss = A.rel(m(xtr[idx].to(device)), ytr[idx].to(device))
        loss.backward()
        opt.step()
        sch.step()


def probe(G, fk, device):
    f_plus, g_plus = G(fk)
    _, g_minus = G(-fk)
    g = 0.5 * (g_plus - g_minus)
    dt = T / (NT - 1)
    Fk, Gk = torch.fft.rfft2(f_plus, dim=(1, 2)), torch.fft.rfft2(g, dim=(1, 2))
    modes = []
    for i, (kx, ky) in enumerate(PROBE_K):
        sl = slice(8 * i, 8 * i + 8)
        fk, uk = Fk[sl, kx % N, ky].to(device), Gk[sl, kx % N, ky].to(device)
        best = (np.inf, None)
        for re in B.GRID_RE:
            mu = torch.tensor(re + 1j * B.GRID_IM, dtype=torch.complex128, device=device)
            r, _ = B.residual(mu, fk, uk, dt)
            j = int(r.argmin())
            if float(r[j]) < best[0]:
                best = (float(r[j]), complex(mu[j]))
        from scipy.optimize import minimize
        fn = lambda q: float(B.residual(torch.tensor([complex(*q)], dtype=torch.complex128, device=device), fk, uk, dt)[0][0])
        o = minimize(fn, [best[1].real, best[1].imag], method="Nelder-Mead", options=dict(xatol=1e-10, fatol=1e-14, maxiter=4000))
        r, _ = B.residual(torch.tensor([complex(*o.x)], dtype=torch.complex128, device=device), fk, uk, dt)
        modes.append(dict(k=[kx, ky], mu=[float(o.x[0]), float(o.x[1])], resid=float(r[0]), truth=float(lam(kx, ky))))
    kept = [m for m in modes if math.hypot(*m["mu"]) * dt <= 0.25 and max(map(abs, m["mu"])) * T >= 2.0]
    res = dict(modes=modes, kept=[m["k"] for m in kept])
    if len(kept) < 3:
        return dict(res, k_c=float("nan"), signs_ok=False)
    k2 = np.array([m["k"][0] ** 2 + m["k"][1] ** 2 for m in kept], float)
    re = np.array([m["mu"][0] for m in kept])
    w = np.sqrt(1 / np.array([max(m["resid"], 1e-6) for m in kept]) ** 2)
    a, *_ = np.linalg.lstsq(np.stack([k2 ** 0, k2, k2 ** 2], 1) * w[:, None], re * w, rcond=None)
    kc = math.sqrt(-a[1] / a[2]) if a[1] > 0 > a[2] else float("nan")
    return dict(res, coef=[float(x) for x in a], k_c=kc,
                signs_ok=bool(all(np.sign(m["mu"][0]) == np.sign(m["truth"]) for m in kept)))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CKPT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    data = make_data()
    t = np.linspace(0, T, NT)
    pfk, pdt = probe_inputs()
    print(f"data {time.time() - t0:.0f} s", flush=True)
    if "--time" in sys.argv:
        for kind in ("wpo", "tfno"):
            m = build(kind, 0, t, device)
            t1 = time.time()
            train(m, data, device, 20, LR, PLR if kind == "wpo" else 1.0)
            torch.cuda.synchronize()
            print(kind, F.M.n_params(m), f"{(time.time() - t1) / 20:.3f} s/step",
                  f"mem {torch.cuda.max_memory_allocated() / 1e9:.1f} GB", flush=True)
        sp = probe(lambda spec: solve_fields(spec, pdt), pfk, device)
        print("solver probe", sp["k_c"], sp["signs_ok"], sp["kept"], [(m["k"], round(m["mu"][0], 3), round(m["truth"], 3)) for m in sp["modes"]])
        return
    out = json.loads(OUT.read_text()) if OUT.exists() else dict(runs=[])
    if "solver_probe" not in out:
        out["solver_probe"] = probe(lambda spec: solve_fields(spec, pdt), pfk, device)
    for kind in ("wpo", "tfno"):
        for seed in SEEDS:
            if any(r["model"] == kind and r["seed"] == seed for r in out["runs"]):
                continue
            t1 = time.time()
            m = build(kind, seed, t, device)
            train(m, data, device, STEPS, LR, PLR if kind == "wpo" else 1.0)
            torch.save(m.state_dict(), CKPT / f"{kind}_{seed}.pt")
            m.eval()
            ev = lambda s: float(A.rel(batched(m, data[s][0], device), data[s][1]))

            def G(spec):
                f = torch.tensor(np.fft.irfft2(spec[..., ::FINE], s=(N, N), axes=(1, 2)))
                return f, batched(m, f, device)

            r = dict(model=kind, seed=seed, params=F.M.n_params(m), train=ev("train"), test=ev("test"),
                     probe=probe(G, pfk, device), seconds=round(time.time() - t1, 1))
            out["runs"].append(r)
            print(kind, seed, f"train {r['train']:.3e} test {r['test']:.3e}", f"k_c {r['probe']['k_c']:.3f}",
                  f"signs {r['probe']['signs_ok']}", f"kept {r['probe']['kept']}", f"{r['seconds']} s", flush=True)
            OUT.write_text(json.dumps(out, indent=1, default=float))
            del m
            torch.cuda.empty_cache()
    summarize(out)


def summarize(out):
    sp = out["solver_probe"]
    lines = ["# (A) Deep operator on forced 2D Cahn–Hilliard (32², M = 0.1, κ = 0.04; k_c = 5)", "",
             f"Solver probe: k_c = {sp['k_c']:.3f}, signs right {sp['signs_ok']}, kept {sp['kept']}.", "",
             "| model | seed | params | train | test | probe k_c | signs right | kept |", "|---|---|---|---|---|---|---|---|"]
    for r in out["runs"]:
        p = r["probe"]
        lines.append(f"| {r['model']} | {r['seed']} | {r['params']} | {r['train']:.3e} | {r['test']:.3e} | {p['k_c']:.3f} | "
                     f"{p['signs_ok']} | {p['kept']} |")
    lines += ["", "Per-mode Re μ̂ (truth) at k = (k, 0) / (k, k), k = 1..6:", ""]
    for name, p in [("solver", sp)] + [(f"{r['model']} {r['seed']}", r["probe"]) for r in out["runs"]]:
        lines.append(f"- {name}: " + ", ".join(f"{tuple(m['k'])} {m['mu'][0]:.3f} ({m['truth']:.3f})" for m in p["modes"]))
    (OUT.with_suffix(".md")).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
