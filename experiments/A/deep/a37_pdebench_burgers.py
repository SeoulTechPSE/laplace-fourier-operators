"""(A) Next-level step 3: the core (A) claims on a public benchmark, PDEBench 1D Burgers (nu = 0.01).

So far every (A) result used data generated here. PDEBench (Takamoto et al., NeurIPS 2022 Datasets and Benchmarks;
DaRUS doi:10.18419/darus-2986, file `1D_Burgers_Sols_Nu0.01.hdf5`) gives an independent data set: 10,000
trajectories of u_t + (u^2 / 2)_x = (nu / pi) u_xx on x in (0, 1) periodic, t in [0, 2], 1024 x 201 samples,
random initial fields, no forcing.

**Task.** Initial field -> trajectory on [0, 2], at PDEBench's reduced resolution for 1D (x every 4th point, t every
5th: 256 x 41). Splits: the first 9,000 trajectories for training (the last 500 of them for validation), the last
1,000 for test (PDEBench's 90 / 10). Metric: PDEBench's nRMSE (per-trajectory relative L2 over x and t > 0,
averaged).

**Exact map to this repository's solver.** With X = 2 pi x and w = 2 pi u the equation becomes w_t + w w_X =
nu' w_XX on [0, 2 pi) with nu' = 4 pi^2 (nu / pi) = 0.04 pi = 0.1257. Linearised at u = 0 each mode m decays at
lambda_m = -nu' m^2 (the same number in physical units: (nu / pi) (2 pi m)^2). Checked before any fit (below).

**Models** (3 seeds; relative L2 loss; Adam with cosine decay; batch 25; lr sweep at 4000 steps with model seed 0,
selection by validation; final 12000 steps):
- `wpo`: `WavenumberPoleOperator1d(ic=True)` on the physical domain (length 1), the d2b configuration (width 23,
  4 layers, 6 poles, stable, pole lr x30), modes 64 (m <= 63), k_ref = 2 pi 16. The forcing input is zero; the
  initial field enters the first pole layer as a free response. Sweep lr 1e-3, 3e-3.
- `tfno_r`: neuraloperator TFNO (Tucker, 4 layers, width 24, domain padding 0.25 in t), input the initial field
  repeated over t; n_modes (64, 21) (|m| <= 32, all temporal modes); rank chosen so that its real parameter count
  (`lfo.metrics.n_real_params`) is closest to the WPO's. Sweep lr 1e-3, 3e-3, 1e-2.

**Metrics.**
1. nRMSE on the test set (256 x 41).
2. Linear probe at u = 0: initial fields eps cos(2 pi m x + phi) / m, eps = 0.01, m = 1..10, 8 phases each; odd
   part (G(u0) - G(-u0)) / 2; per mode, the decay rate from a least-squares fit of log |mode amplitude| over the
   samples above 1e-3 of the first; nu'_hat from lambda_m = -nu' m^2 (weighted by the number of samples); and the
   dispersion error ||lambda_hat - lambda|| / ||lambda|| over m = 1..10. The same probe on the true solver checks the
   method.
3. Resolution transfer: nRMSE on the test set at the full 1024 x 41 grid, without retraining.

**Checks before any fit** (run 2026-11-22, before this commit). The file holds 201 time samples (t-coordinate
202 entries: 0 to 2.01) and 1024 points; reduced arrays 256 x 41, max |u| 3.41 in training. The repository solver
through the 2 pi map reproduces PDEBench test trajectories to nRMSE 2.9e-4 (1024 points). The probe on the solver
gives nu'_hat = 0.12566 (truth 0.12566), dispersion error 3e-5. Budgets: WPO 61,130 real parameters (width 23,
the IC path included); TFNO rank 0.015, 64,745 (+6 %; the next rank down is further away).

**Predictions** (written before running).

1. `wpo`: probe nu'_hat within 5 % of 0.1257 in every seed; `tfno_r`'s dispersion error >= 3x the WPO's (mean).
2. Accuracy: `tfno_r`'s mean nRMSE is below the WPO's (on strongly nonlinear data the TFNO has been the more
   accurate surrogate in (A) and (D)), and the WPO's is <= 2x `tfno_r`'s.
3. Resolution transfer: both models' 1024-point nRMSE is <= 1.3x their 256-point nRMSE in every seed.

    python experiments/A/deep/a37_pdebench_burgers.py      # results/a37_pdebench_burgers.md / .json
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
import a11_fair as F  # noqa: E402
from lfo.data import burgers  # noqa: E402
from lfo.metrics import n_real_params  # noqa: E402
from lfo.models import WavenumberPoleOperator1d  # noqa: E402

FILE = ROOT / "runs" / "data" / "pdebench" / "1D_Burgers_Sols_Nu0.01.hdf5"
CACHE = ROOT / "runs" / "A" / "a37" / "data.pt"
OUT = HERE / "results" / "a37_pdebench_burgers.json"
NU_P = 0.04 * math.pi                                     # nu' on [0, 2 pi): 4 pi^2 (0.01 / pi)
RX, RT, NTR, NVA = 4, 5, 9000, 500
SEEDS, BATCH, SWEEP_STEPS, FINAL_STEPS = (0, 1, 2), 25, 4000, 12000
EPS, MS, NPH = 0.01, np.arange(1, 11), 8
WPO_KW = dict(width=23, n_layers=4, n_poles=6, modes=64, stable=True, k_ref=2 * math.pi * 16, length=1.0)
SWEEP = {"wpo": [dict(lr=1e-3, plr=30.0), dict(lr=3e-3, plr=30.0)], "tfno_r": [dict(lr=lr) for lr in (1e-3, 3e-3, 1e-2)]}


def load():
    if CACHE.exists():
        return torch.load(CACHE)
    with h5py.File(FILE, "r") as h:
        u = h["tensor"][:, ::RT, :]                                   # (N, 41, 1024) float32
        t = np.asarray(h["t-coordinate"])[:-1:RT] if len(h["t-coordinate"]) > u.shape[1] * RT else np.asarray(h["t-coordinate"])[::RT]
    u = torch.tensor(u).permute(0, 2, 1).contiguous()                 # (N, 1024, 41)
    d = dict(full=u, t=torch.tensor(t[:u.shape[2]], dtype=torch.float64))
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    torch.save(d, CACHE)
    return d


def splits(d, rx=RX):
    u = d["full"][:, ::rx].to(torch.float64)
    return dict(train=u[:NTR - NVA], vali=u[NTR - NVA:NTR], test=u[NTR:])


class WPO(nn.Module):
    def __init__(self, t):
        super().__init__()
        self.net = WavenumberPoleOperator1d(t, ic=True, **WPO_KW)

    def forward(self, u0, n_t):
        return self.net(torch.zeros(*u0.shape, n_t, dtype=torch.float64, device=u0.device), u0)


class TFNO(nn.Module):
    def __init__(self, rank):
        super().__init__()
        from neuralop.models import FNO
        self.net = FNO(n_modes=(64, 21), in_channels=1, out_channels=1, hidden_channels=24, n_layers=4,
                       domain_padding=[0.0, 0.25], factorization="tucker", rank=rank)

    def forward(self, u0, n_t):
        return self.net(u0.float()[:, None, :, None].expand(-1, 1, -1, n_t).contiguous())[:, 0].to(torch.float64)


def tfno_rank(target):
    cands = [r / 1000 for r in range(2, 40)]
    return min(cands, key=lambda r: abs(n_real_params(TFNO(r)) - target))


def build(kind, seed, t, device, rank=None):
    torch.manual_seed(seed)
    return (WPO(t) if kind == "wpo" else TFNO(rank)).to(device)


def nrmse(p, y):
    """PDEBench nRMSE: per trajectory ||p - y|| / ||y|| over x and t > 0, averaged."""
    p, y = p[..., 1:], y[..., 1:]
    return float(((p - y).flatten(1).norm(dim=1) / y.flatten(1).norm(dim=1)).mean())


def predict(model, u, device, bs=50):
    with torch.no_grad():
        return torch.cat([model(u[i:i + bs, :, 0].to(device), u.shape[2]).cpu() for i in range(0, len(u), bs)])


def train(model, u, device, lr, steps, plr=1.0):
    is_pole = lambda n: n.endswith("mu_re") or n.endswith("mu_im")
    groups = [dict(params=[p for n, p in model.named_parameters() if not is_pole(n)], lr=lr)]
    pole = [p for n, p in model.named_parameters() if is_pole(n)]
    if pole:
        groups.append(dict(params=pole, lr=lr * plr))
    opt = torch.optim.Adam(groups)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    g = torch.Generator().manual_seed(0)
    loss = torch.tensor(0.0)
    for _ in range(steps):
        idx = torch.randint(0, u.shape[0], (BATCH,), generator=g)
        y = u[idx].to(device)
        opt.zero_grad()
        p = model(y[:, :, 0], y.shape[2])
        loss = ((p - y).flatten(1).norm(dim=1) / y.flatten(1).norm(dim=1)).mean()
        if not torch.isfinite(loss):
            return float("nan")
        loss.backward()
        opt.step()
        sch.step()
    return float(loss)


def probe_inputs(n_x):
    rng = np.random.default_rng(5)
    x = np.arange(n_x) / n_x
    ph = rng.uniform(0, 2 * np.pi, size=(len(MS), NPH))
    u0 = np.stack([EPS / m * np.cos(2 * np.pi * m * x + p) for m, row in zip(MS, ph) for p in row])
    return torch.tensor(u0)


def probe_fit(g, t):
    """Decay rate per probed mode from the odd-part response g (len(MS) * NPH, n_x, n_t)."""
    gk = torch.fft.rfft(g, dim=1).numpy()
    lam, w = [], []
    for i, m in enumerate(MS):
        rates, cnt = [], 0
        for j in range(NPH):
            a = np.abs(gk[i * NPH + j, m])
            ok = a > 1e-3 * a[0]
            ok &= np.cumprod(ok).astype(bool)
            if ok.sum() >= 3:
                rates.append(np.polyfit(t[ok], np.log(a[ok]), 1)[0])
                cnt += int(ok.sum())
        lam.append(float(np.mean(rates)) if rates else float("nan"))
        w.append(cnt)
    lam, w = np.array(lam), np.array(w, float)
    ok = np.isfinite(lam)
    nu_hat = float(-(w[ok] * lam[ok] * MS[ok] ** 2).sum() / (w[ok] * MS[ok] ** 4).sum())
    true = -NU_P * MS ** 2
    disp = float(np.linalg.norm(np.where(ok, lam, 0) - true) / np.linalg.norm(true))
    return dict(lam=lam.tolist(), nu_hat=nu_hat, disp_err=disp, n_ok=int(ok.sum()))


def model_probe(model, t, n_x, n_t, device):
    u0 = probe_inputs(n_x).to(device)
    with torch.no_grad():
        g = 0.5 * (model(u0, n_t) - model(-u0, n_t)).cpu()
    return probe_fit(g, t)


def solver_traj(u0, t, n_x, fine=40):
    """True trajectories from initial fields u0 (B, n_x) on (0, 1) via the 2 pi map (w = 2 pi u)."""
    dt = float(t[1] - t[0]) / fine
    n = (len(t) - 1) * fine + 1
    w0k = np.fft.rfft(2 * np.pi * np.asarray(u0, float), axis=1)
    fk = np.zeros((u0.shape[0], n_x // 2 + 1, n), complex)
    wk = burgers.solve(fk, dt, NU_P, n_x, stride=fine, u0k=w0k)
    return torch.tensor(np.fft.irfft(wk, n=n_x, axis=1) / (2 * np.pi))


def checks(d):
    """Before any fit: the map to the repository's solver, and the probe on the solver."""
    t = d["t"].numpy()
    u = d["full"][NTR:NTR + 8].to(torch.float64)                       # 1024 points
    sol = solver_traj(u[:, :, 0].numpy(), t, u.shape[1])
    c1 = nrmse(sol, u)
    u0 = probe_inputs(256).numpy()
    g = 0.5 * (solver_traj(u0, t, 256) - solver_traj(-u0, t, 256))
    return dict(solver_vs_pdebench=c1, solver_probe=probe_fit(g, t))


def main():
    device = torch.device("cuda")
    d = load()
    t = d["t"].numpy()
    S = splits(d)
    n_t = S["train"].shape[2]
    out = json.loads(OUT.read_text()) if OUT.exists() else dict(checks=None, sweep=[], final=[])
    save = lambda: OUT.write_text(json.dumps(out, indent=1, default=float))
    if out["checks"] is None:
        out["checks"] = checks(d)
        save()
    print("checks", out["checks"], flush=True)
    target = n_real_params(WPO(t))
    rank = tfno_rank(target)
    out["budget"] = dict(wpo=target, tfno_r=n_real_params(TFNO(rank)), tfno_rank=rank)
    save()
    print("budget", out["budget"], flush=True)
    for kind, cfgs in SWEEP.items():
        for cfg in cfgs:
            if any(r["model"] == kind and r["cfg"] == cfg for r in out["sweep"]):
                continue
            t1 = time.time()
            m = build(kind, 0, t, device, rank)
            last = train(m, S["train"], device, steps=SWEEP_STEPS, **cfg)
            m.eval()
            vali = nrmse(predict(m, S["vali"], device), S["vali"]) if np.isfinite(last) else float("inf")
            out["sweep"].append(dict(model=kind, cfg=cfg, vali=vali, seconds=round(time.time() - t1, 1)))
            print("sweep", kind, cfg, f"vali {vali:.4f}", f"{time.time() - t1:.0f} s", flush=True)
            save()
            del m
            torch.cuda.empty_cache()
    full_test = d["full"][NTR:].to(torch.float64)
    for kind in SWEEP:
        best = min((r for r in out["sweep"] if r["model"] == kind), key=lambda r: r["vali"])["cfg"]
        for seed in SEEDS:
            if any(r["model"] == kind and r["seed"] == seed for r in out["final"]):
                continue
            t1 = time.time()
            m = build(kind, seed, t, device, rank)
            train(m, S["train"], device, steps=FINAL_STEPS, **best)
            m.eval()
            r = dict(model=kind, seed=seed, cfg=best, real_params=n_real_params(m),
                     test=nrmse(predict(m, S["test"], device), S["test"]),
                     test_1024=nrmse(predict(m, full_test, device), full_test),
                     probe=model_probe(m, t, 256, n_t, device), seconds=round(time.time() - t1, 1))
            out["final"].append(r)
            print("final", kind, seed, best, f"test {r['test']:.4f} 1024 {r['test_1024']:.4f}",
                  f"nu' {r['probe']['nu_hat']:.4f} disp {r['probe']['disp_err']:.3f}", f"{r['seconds']} s", flush=True)
            save()
            del m
            torch.cuda.empty_cache()
    summarize(out)


def summarize(out):
    c = out["checks"]
    rg = lambda xs, f: f"{np.mean(xs):{f}} ({min(xs):{f}}–{max(xs):{f}})"
    lines = ["# (A) PDEBench 1D Burgers (nu = 0.01): initial field -> trajectory, 256 x 41", "",
             f"Checks: repository solver vs PDEBench, nRMSE {c['solver_vs_pdebench']:.2e}; solver probe nu' "
             f"{c['solver_probe']['nu_hat']:.4f} (truth {NU_P:.4f}), dispersion error {c['solver_probe']['disp_err']:.3f}. "
             f"Budget (real): {out['budget']}. Means over 3 seeds (range).", "",
             "| model | real params | nRMSE 256 | nRMSE 1024 | 1024 / 256 | probe ν' | dispersion error |",
             "|---|---|---|---|---|---|---|"]
    for kind in SWEEP:
        rs = [r for r in out["final"] if r["model"] == kind]
        if rs:
            lines.append(f"| {kind} | {rs[0]['real_params']} | {rg([r['test'] for r in rs], '.4f')} | "
                         f"{rg([r['test_1024'] for r in rs], '.4f')} | {rg([r['test_1024'] / r['test'] for r in rs], '.2f')} | "
                         f"{rg([r['probe']['nu_hat'] for r in rs], '.4f')} | {rg([r['probe']['disp_err'] for r in rs], '.3f')} |")
    lines += ["", "| sweep | cfg | validation |", "|---|---|---|"] + [
        f"| {r['model']} | {r['cfg']} | {r['vali']:.4f} |" for r in out["sweep"]]
    (OUT.with_suffix(".md")).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
