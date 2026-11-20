"""(A) Application 2, stage 2: rollout into the coarsening regime of 2D Cahn–Hilliard.

Plan (`docs/proposal_A_wavenumber_pole_operator.qmd`, Section 6.2, stage 2): roll out into coarsening and compare
the structure factor S(k, t) and the growth of the characteristic length L(t) (theory t^{1/3}) against baselines.
Stage 1 (a29) and stage 3 (a30) were the linear regime; this is the nonlinear one.

**System.** Unforced u_t = M lap (u^3 - u - kappa lap u), M = 1, kappa = 0.02 (k_c = 7.1, interface width
sqrt(2 kappa) = 0.2), periodic [0, 2 pi)^2, critical quench (mean 0). Truth from `lfo.data.cahn_hilliard.
solve2d_free_torch` on 128^2 (dealiased band |k| <= 32), dt = 4e-4, from white noise of standard deviation 0.05
on that band. Fields are stored every 0.1 up to t = 21 and spectrally truncated to 64^2 (|k_x|, k_y <= 31) for the
models.

**Task.** A window maps the state at t0 to the states over [t0, t0 + 1] (11 frames). Training windows start at
t0 = 1, 2, ..., 20 (after the spinodal stage, so in the coarsening regime). Rollout: from the true state at t = 1,
20 chained windows to t = 21.

**Data.** Trajectories 48 / 8 / 16 (train / validation / test), data seed 1500: 960 training windows.

**Models** (3 seeds; Adam, cosine decay, 6000 steps, batch 10, lr 3e-3; relative L2 loss):
- `wpo`: `lfo.models.WavenumberPoleOperator2d` (a17's settings: width 16, 4 layers, 4 poles, Re cap 1.0, leak 0.05,
  k_ref 8; modes 12), the state at t0 broadcast over the window as its input (unforced, so the only input);
  pole lr ratio 30. Output: the 11 frames.
- `tfno`: neuraloperator FNO on (x, y, t), Tucker, n_modes (16, 16, 6), same input and output, rank chosen for the
  parameter count closest to the WPO's (a17's rule).
- `fno2d`: neuraloperator FNO on (x, y), n_modes (16, 16), dense, hidden width chosen for the closest parameter
  count; maps the state at t0 to the state at t0 + 1 (the usual autoregressive surrogate). Loss on that frame only.

**Metrics** (test trajectories).
- `win`: relative L2 of the window-end state from true starts (16 x 20 windows).
- Rollout: per-window end-state error; L(t) = 2 pi / <k>, <k> the first moment of S(k, t) averaged over the 16
  trajectories; the exponent n from a least-squares fit of log L against log t on t = 2..21; S(k, 21) radially
  averaged on integer shells 1..16, relative L2 against the truth; max |u| at t = 21.

**Checks before any fit.**
- Solver: lfo's 2D solvers needed a real-field projection per step (1a7deee); without it, round-off in the
  non-Hermitian part of the k_y = 0 column grew in the unstable band and the run diverged at t ~ 5.
- Resolution (8 trajectories, seed 0): 128^2 against 256^2 (dt / 4) agree to 3.7e-5 in the field at t = 21 and
  4.5e-6 in L(t). Energy beyond |k| = 12 at t = 21: 3e-4 of the total.
- Truth on the test set: L(1) = 1.51, L(21) = 3.52, n = 0.271 (the t^{1/3} law with early-time and finite-box
  corrections; L(21) is 0.56 of the box); max |u| 1.04.
- References: persistence (u(t0 + 1) = u(t0)) has window-end error 0.192; a rollout frozen at t = 1 has
  L(21) / truth 0.43 and S(k, 21) error 1.12.
- Parameters (the closest available): `wpo` 18,737; `tfno` 15,473 (rank grid, 17 % fewer); `fno2d` 21,331 (width
  grid, 14 % more). 0.117 / 0.037 / 0.014 s per step.

**Predictions** (written before running).

1. Every model beats persistence on the window-end error (0.192) in every seed.
2. Mean window-end error: `tfno` <= `wpo` (as in D2's d2g on 1D Cahn–Hilliard).
3. `wpo`: L(21) within 10 % of the truth and n within +-0.05 of 0.271, in at least 2 of 3 seeds.
4. `fno2d`: the same criteria as 3, in at least 2 of 3 seeds.
5. Every model, every seed: S(k, 21) error <= 0.25.

    python experiments/A/deep/a31_coarsening.py [--time]      # results/a31_coarsening.md / .json
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
ROOT = HERE.parents[2]
from lfo.data import cahn_hilliard as ch  # noqa: E402
from lfo.models import WavenumberPoleOperator2d  # noqa: E402

MOB, KAPPA, NS, NM, DT, T_END, FR = 1.0, 0.02, 128, 64, 4e-4, 21.0, 10       # FR frames per unit time
SPLIT, SEEDS, BATCH, LR, PLR, STEPS = (48, 8, 16), (0, 1, 2), 10, 3e-3, 30.0, 6000
NWIN, KINDS = 20, ("wpo", "tfno", "fno2d")
WPO_KW = dict(width=16, n_layers=4, n_poles=4, modes=12, re_max=1.0, re_leak=0.05, k_ref=8.0)
OUT = HERE / "results" / "a31_coarsening.json"
CACHE = ROOT / "runs" / "A" / "a31" / "data.pt"
TW = np.linspace(0, 1, FR + 1)


def initial(rng, b):
    u0k = np.fft.rfft2(0.05 * rng.normal(size=(b, NS, NS)), axes=(1, 2))
    kx = np.fft.fftfreq(NS, 1 / NS)[:, None]
    ky = np.fft.rfftfreq(NS, 1 / NS)[None, :]
    u0k[:, ~((np.abs(kx) <= NS / 4) & (ky <= NS / 4))] = 0
    return u0k


def truncate(uk):
    """rfft2 on NS^2 (torch, (b, NS, NS/2+1, m)) -> real fields on NM^2 with |k_x|, k_y <= NM/2 - 1."""
    h = NM // 2 - 1
    out = torch.zeros(uk.shape[0], NM, NM // 2 + 1, uk.shape[-1], dtype=uk.dtype, device=uk.device)
    out[:, :h + 1, :h + 1] = uk[:, :h + 1, :h + 1]
    out[:, NM - h:, :h + 1] = uk[:, NS - h:, :h + 1]
    return torch.fft.irfft2(out * (NM / NS) ** 2, s=(NM, NM), dim=(1, 2))


def make_data():
    if CACHE.exists():
        return torch.load(CACHE)
    rng = np.random.default_rng(1500)
    steps, stride = int(round(T_END / DT)), int(round(1 / (FR * DT)))
    parts = []
    for _ in range(sum(SPLIT) // 8):
        uk = ch.solve2d_free_torch(initial(rng, 8), DT, MOB, KAPPA, NS, steps, stride=stride, device="cuda")
        parts.append(truncate(uk).cpu())
    u = torch.cat(parts)                                                     # (n, NM, NM, 211)
    c = np.cumsum([0, *SPLIT])
    d = {k: u[a:b] for k, a, b in zip(("train", "vali", "test"), c[:-1], c[1:])}
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    torch.save(d, CACHE)
    return d


def windows(u):
    """All windows starting at t0 = 1..NWIN: (n * NWIN, NM, NM, FR + 1)."""
    return torch.cat([u[..., FR * s:FR * s + FR + 1] for s in range(1, NWIN + 1)])


class WPO(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = WavenumberPoleOperator2d(TW, **WPO_KW)

    def forward(self, u0):
        return self.net(u0[..., None].expand(*u0.shape, FR + 1))


class TFNO3(torch.nn.Module):
    def __init__(self, rank):
        super().__init__()
        from neuralop.models import FNO
        self.net = FNO(n_modes=(16, 16, 6), in_channels=1, out_channels=1, hidden_channels=16, n_layers=4,
                       domain_padding=[0.0, 0.0, 0.25], factorization="tucker", rank=rank)

    def forward(self, u0):
        x = u0[..., None].expand(*u0.shape, FR + 1)
        return self.net(x.float()[:, None])[:, 0].to(u0.dtype)


class FNO2(torch.nn.Module):
    def __init__(self, width):
        super().__init__()
        from neuralop.models import FNO
        self.net = FNO(n_modes=(16, 16), in_channels=1, out_channels=1, hidden_channels=width, n_layers=4)

    def forward(self, u0):
        return self.net(u0.float()[:, None])[:, 0].to(u0.dtype)


n_params = lambda m: sum(p.numel() for p in m.parameters())


def build(kind, seed, device):
    torch.manual_seed(seed)
    target = n_params(WPO())
    if kind == "wpo":
        m = WPO()
    elif kind == "tfno":
        rank = min((0.002, 0.005, 0.01, 0.02, 0.05, 0.1), key=lambda r: abs(n_params(TFNO3(r)) - target))
        torch.manual_seed(seed)
        m = TFNO3(rank)
    else:
        w = min(range(4, 33), key=lambda w: abs(n_params(FNO2(w)) - target))
        torch.manual_seed(seed)
        m = FNO2(w)
    return m.to(device)


def rel(p, y):
    return ((p - y).flatten(1).norm(dim=1) / y.flatten(1).norm(dim=1)).mean()


def step_end(m, kind, u0):
    return m(u0) if kind == "fno2d" else m(u0)[..., -1]


def train(m, kind, xw, device, steps=STEPS):
    is_pole = lambda n: n.endswith("mu_re") or n.endswith("mu_im") or n.endswith("adv")
    groups = [dict(params=[p for n, p in m.named_parameters() if not is_pole(n)], lr=LR)]
    pole = [p for n, p in m.named_parameters() if is_pole(n)]
    if pole:
        groups.append(dict(params=pole, lr=LR * PLR))
    opt = torch.optim.Adam(groups)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    g = torch.Generator().manual_seed(0)
    for _ in range(steps):
        idx = torch.randint(0, xw.shape[0], (BATCH,), generator=g)
        w = xw[idx].to(device)
        opt.zero_grad()
        p = m(w[..., 0])
        loss = rel(p, w[..., -1]) if kind == "fno2d" else rel(p, w)
        loss.backward()
        opt.step()
        sch.step()


@torch.no_grad()
def batched_end(m, kind, u0, device, bs=16):
    return torch.cat([step_end(m, kind, u0[i:i + bs].to(device)).cpu().to(torch.float64)
                      for i in range(0, len(u0), bs)])


@torch.no_grad()
def rollout(m, kind, u, device):
    """From the true state at t = 1, NWIN chained windows; returns the end states (n, NM, NM, NWIN + 1) at t = 1..21."""
    s = u[..., FR]
    out = [s]
    for _ in range(NWIN):
        s = batched_end(m, kind, s, device)
        out.append(s)
    return torch.stack(out, -1)


KM = torch.sqrt(torch.fft.fftfreq(NM, 1 / NM)[:, None] ** 2 + torch.fft.rfftfreq(NM, 1 / NM)[None, :] ** 2)
WK = torch.full(KM.shape, 2.0)
WK[:, 0] = 1


def stats(u):
    """u (n, NM, NM, m) -> L(t) (m,), radial S on shells 1..16 (16, m); S averaged over samples."""
    S = (torch.fft.rfft2(u, dim=(1, 2)).abs() ** 2 * WK[..., None]).mean(0)
    S[0, 0] = 0
    L = 2 * math.pi * S.sum((0, 1)) / (KM[..., None] * S).sum((0, 1))
    shell = torch.round(KM).long()
    rad = torch.stack([S[shell == k].sum(0) for k in range(1, 17)])
    return L, rad


def exponent(L):
    t = np.arange(2, NWIN + 2)
    return float(np.polyfit(np.log(t), np.log(np.asarray(L[1:])), 1)[0])


def evaluate(m, kind, data, device):
    te = data["test"]
    xw = windows(te)
    win = float(rel(batched_end(m, kind, xw[..., 0], device), xw[..., -1]))
    p = rollout(m, kind, te, device)
    y = te[..., FR::FR]                                                      # t = 1..21
    per = [float(rel(p[..., i], y[..., i])) for i in range(1, NWIN + 1)]
    Lp, Sp = stats(p)
    Ly, Sy = stats(y)
    return dict(win=win, roll_per_window=per, L=Lp.tolist(), L_true=Ly.tolist(), n=exponent(Lp), n_true=exponent(Ly),
                L21_ratio=float(Lp[-1] / Ly[-1]), S21_err=float((Sp[:, -1] - Sy[:, -1]).norm() / Sy[:, -1].norm()),
                maxabs21=float(p[..., -1].abs().max()))


def main():
    device = torch.device("cuda")
    data = make_data()
    if "--time" in sys.argv:
        xw = windows(data["train"])
        for kind in KINDS:
            m = build(kind, 0, device)
            t0 = time.time()
            train(m, kind, xw, device, steps=20)
            torch.cuda.synchronize()
            print(kind, n_params(m), f"{(time.time() - t0) / 20:.3f} s/step")
        Ly, _ = stats(data["test"][..., FR::FR])
        print("truth L(t=1..21)", np.round(Ly.numpy(), 3), "n", exponent(Ly))
        return
    xw = windows(data["train"])
    out = json.loads(OUT.read_text()) if OUT.exists() else []
    for kind in KINDS:
        for seed in SEEDS:
            if any(r["model"] == kind and r["seed"] == seed for r in out):
                continue
            t0 = time.time()
            m = build(kind, seed, device)
            train(m, kind, xw, device)
            m.eval()
            r = dict(model=kind, seed=seed, params=n_params(m), **evaluate(m, kind, data, device),
                     seconds=round(time.time() - t0, 1))
            out.append(r)
            print({q: (round(v, 4) if isinstance(v, float) else v) for q, v in r.items()
                   if q not in ("roll_per_window", "L", "L_true")}, flush=True)
            OUT.parent.mkdir(exist_ok=True)
            OUT.write_text(json.dumps(out, indent=1))
            del m
            torch.cuda.empty_cache()
    rg = lambda xs, f: f"{min(xs):{f}}–{max(xs):{f}}"
    r0 = out[0]
    lines = ["# (A) 2D Cahn–Hilliard coarsening: 20-window rollouts from t = 1 to 21", "",
             f"Truth: L(1) = {r0['L_true'][0]:.3f}, L(21) = {r0['L_true'][-1]:.3f}, exponent n on t = 2..21: "
             f"{r0['n_true']:.3f}. Ranges over 3 seeds.", "",
             "| model | params | window-end error | rollout error, window 1 / 5 / 20 | L(21) / truth | n | S(k, 21) error | max abs u(21) |",
             "|---|---|---|---|---|---|---|---|"]
    for kind in KINDS:
        rs = [r for r in out if r["model"] == kind]
        if rs:
            pw = np.mean([r["roll_per_window"] for r in rs], 0)
            lines.append(f"| {kind} | {rs[0]['params']} | {rg([r['win'] for r in rs], '.4f')} | "
                         f"{pw[0]:.3f} / {pw[4]:.3f} / {pw[-1]:.3f} | {rg([r['L21_ratio'] for r in rs], '.3f')} | "
                         f"{rg([r['n'] for r in rs], '.3f')} | {rg([r['S21_err'] for r in rs], '.3f')} | "
                         f"{rg([r['maxabs21'] for r in rs], '.2f')} |")
    (OUT.with_suffix(".md")).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
