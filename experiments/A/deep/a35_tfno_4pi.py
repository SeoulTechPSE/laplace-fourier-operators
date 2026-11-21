"""(A) Application 2, stage 2e: a TFNO trained on the 4 pi box (the accuracy reference for a34's transfer).

a34: a WPO-2D trained on the 2 pi box runs on a 4 pi box at the same spacing without retraining (window-end error
0.038, 1.02–1.03x its 2 pi error); the TFNO trained on 2 pi does not (0.69). That tests transfer. The accuracy
reference is a TFNO trained on the 4 pi box itself, which this script provides.

**Data on 4 pi** (a34's truth generator: 256^2 with M / 4, kappa / 4, truncated to 128^2). Training: 12
trajectories (data seed 1700), the same total area as a31's 48 on 2 pi, so 240 windows, plus one amplitude-scaled
copy each as a33 (a ~ U[0.7, 1.6], rng seed 1701; truth from a s on 256^2, dt / 2). Test: a34's 16 trajectories.

**Models** (a33's training: 6000 steps, batch 10, lr 3e-3; 3 seeds):
- `tfno4`: a31's TFNO unchanged (n_modes (16, 16, 6), rank 0.01, 15,473 parameters). On 4 pi its spectral path
  covers physical |k| < 4, half a31's band.
- `tfno4_m32`: n_modes (32, 32, 6), the same physical band as a31's TFNO on 2 pi; rank 0.005 (19,109 parameters,
  the closest to the WPO-2D's 18,737).
Reference: a34's transferred `wpo_aug` (trained on 2 pi only).

**Checks before any fit.** Re-solving one window from a 4 pi test state (a = 1) against the stored truth: 3.2e-5
(the truncation). Training set 480 windows, max |u| 1.63. 0.30 / 0.27 s per step (`tfno4` / `tfno4_m32`, with
another job on the GPU).

**Metrics** (a34's): window-end error on the 4 pi test windows, 20-window rollout from t = 1, L(21) / truth, n.

**Predictions** (written before running).

1. `tfno4_m32`: mean 4 pi window-end error <= 0.025 (1.3x a33's `tfno_aug` on 2 pi, 0.019).
2. `tfno4_m32`: mean 4 pi window-end error below the transferred `wpo_aug`'s (0.038).
3. `tfno4`: mean window-end error >= 1.2x `tfno4_m32`'s (the band of the spectral path matters).
4. `tfno4_m32`: L(21) within 10 % and n within +-0.05 of the truth (0.261) in every seed.

    python experiments/A/deep/a35_tfno_4pi.py      # results/a35_tfno_4pi.md / .json
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
import a31_coarsening as C  # noqa: E402
import a34_domain as D  # noqa: E402

KINDS = {"tfno4": ((16, 16, 6), 0.01), "tfno4_m32": ((32, 32, 6), 0.005)}
NTRAIN, SEED_TR, SEED_AUG = 12, 1700, 1701
CACHE = D.CKPT / "train4pi.pt"
OUT = HERE / "results" / "a35_tfno_4pi.json"


class TFNO(torch.nn.Module):
    def __init__(self, n_modes, rank):
        super().__init__()
        from neuralop.models import FNO
        self.net = FNO(n_modes=n_modes, in_channels=1, out_channels=1, hidden_channels=16, n_layers=4,
                       domain_padding=[0.0, 0.0, 0.25], factorization="tucker", rank=rank)

    def forward(self, u0):
        x = u0[..., None].expand(*u0.shape, C.FR + 1)
        return self.net(x.float()[:, None])[:, 0].to(u0.dtype)


def solve_trajectories(seed, n):
    rng = np.random.default_rng(seed)
    steps, stride = int(round(C.T_END / C.DT)), int(round(1 / (C.FR * C.DT)))
    parts = []
    for _ in range(n // 4):
        u0k = np.fft.rfft2(0.05 * rng.normal(size=(4, D.NS4, D.NS4)), axes=(1, 2))
        kx = np.fft.fftfreq(D.NS4, 1 / D.NS4)[:, None]
        ky = np.fft.rfftfreq(D.NS4, 1 / D.NS4)[None, :]
        u0k[:, ~((np.abs(kx) <= D.NS4 / 4) & (ky <= D.NS4 / 4))] = 0
        uk = C.ch.solve2d_free_torch(u0k, C.DT, C.MOB / 4, C.KAPPA / 4, D.NS4, steps, stride=stride, device="cuda")
        parts.append(down(uk).cpu())
        del uk
        torch.cuda.empty_cache()
    return torch.cat(parts)


def down(uk):
    h = D.NM4 // 2 - 1
    out = torch.zeros(uk.shape[0], D.NM4, D.NM4 // 2 + 1, uk.shape[-1], dtype=uk.dtype, device=uk.device)
    out[:, :h + 1, :h + 1] = uk[:, :h + 1, :h + 1]
    out[:, D.NM4 - h:, :h + 1] = uk[:, D.NS4 - h:, :h + 1]
    return torch.fft.irfft2(out * (D.NM4 / D.NS4) ** 2, s=(D.NM4, D.NM4), dim=(1, 2))


def true_window(s, bs=16):
    """Truth over one window (11 frames) from 128^2 states on the 4 pi box."""
    n, h, dt = D.NS4, D.NM4 // 2, C.DT / 2
    steps = int(round(1.0 / dt))
    outs = []
    for i in range(0, len(s), bs):
        sk = torch.fft.rfft2(s[i:i + bs], dim=(1, 2))
        big = torch.zeros(sk.shape[0], n, n // 2 + 1, dtype=sk.dtype)
        big[:, :h, :h], big[:, -h:, :h] = sk[:, :h, :h], sk[:, -h:, :h]
        uk = C.ch.solve2d_free_torch(big * (n / D.NM4) ** 2, dt, C.MOB / 4, C.KAPPA / 4, n, steps,
                                     stride=steps // C.FR, device="cuda")
        o = torch.zeros(sk.shape[0], D.NM4, D.NM4 // 2 + 1, uk.shape[-1], dtype=uk.dtype)
        o[:, :h, :h], o[:, -h:, :h] = uk[:, :h, :h].cpu(), uk[:, -h:, :h].cpu()
        outs.append(torch.fft.irfft2(o / (n / D.NM4) ** 2, s=(D.NM4, D.NM4), dim=(1, 2)))
        del uk
        torch.cuda.empty_cache()
    return torch.cat(outs)


def training_set():
    if CACHE.exists():
        return torch.load(CACHE)
    xw = C.windows(solve_trajectories(SEED_TR, NTRAIN))
    a = torch.tensor(np.random.default_rng(SEED_AUG).uniform(0.7, 1.6, size=len(xw)))
    aw = true_window(a[:, None, None] * xw[..., 0])
    train = torch.cat([xw, aw])
    torch.save(train, CACHE)
    return train


def main():
    device = torch.device("cuda")
    u4 = D.truth4()
    L4 = D.stats(u4[..., C.FR::C.FR], 4 * math.pi)
    train = training_set()
    print("train", tuple(train.shape), "max |u|", float(train.abs().max()), flush=True)
    out = json.loads(OUT.read_text()) if OUT.exists() else []
    for kind, (nm, rank) in KINDS.items():
        for seed in C.SEEDS:
            if any(r["model"] == kind and r["seed"] == seed for r in out):
                continue
            t0 = time.time()
            torch.manual_seed(seed)
            m = TFNO(nm, rank).to(device)
            C.train(m, "tfno", train, device)
            m.eval()
            r = dict(model=kind, seed=seed, params=C.n_params(m), win_4pi=D.win_error(m, "tfno", u4, device, 4))
            p = D.rollout(m, "tfno", u4, device, 4)
            y = u4[..., C.FR::C.FR]
            Lp = D.stats(p, 4 * math.pi)
            r.update(roll_per_window=[float(C.rel(p[..., i], y[..., i])) for i in range(1, C.NWIN + 1)],
                     L=Lp.tolist(), n=C.exponent(Lp), n_true=C.exponent(L4), L21_ratio=float(Lp[-1] / L4[-1]),
                     maxabs21=float(p[..., -1].abs().max()), seconds=round(time.time() - t0, 1))
            out.append(r)
            print({q: (round(v, 4) if isinstance(v, float) else v) for q, v in r.items()
                   if q not in ("roll_per_window", "L")}, flush=True)
            OUT.parent.mkdir(exist_ok=True)
            OUT.write_text(json.dumps(out, indent=1))
            del m
            torch.cuda.empty_cache()
    ref = [dict(r, model="wpo_aug (2 pi, transferred)") for r in json.loads((HERE / "results" / "a34_domain.json").read_text())
           if r["model"] == "wpo_aug"]
    rows = [(k, [r for r in out if r["model"] == k]) for k in KINDS] + [("wpo_aug (2 pi, transferred; a34)", ref)]
    rg = lambda xs, f: f"{min(xs):{f}}–{max(xs):{f}}"
    lines = ["# (A) 2D Cahn–Hilliard coarsening on the 4 pi box: TFNO trained there against the transferred WPO-2D", "",
             f"Truth on 4 pi: L(21) = {L4[-1]:.3f}, n = {C.exponent(L4):.3f}. Ranges over 3 seeds.", "",
             "| model | params | window-end error 4 pi | rollout, window 1 / 5 / 20 | L(21) / truth | n | max abs u(21) |",
             "|---|---|---|---|---|---|---|"]
    for name, rs in rows:
        pw = np.mean([r["roll_per_window"] for r in rs], 0)
        par = rs[0].get("params", 18737)
        lines.append(f"| {name} | {par} | {rg([r['win_4pi'] for r in rs], '.4f')} | {pw[0]:.3f} / {pw[4]:.3f} / "
                     f"{pw[-1]:.3g} | {rg([r['L21_ratio'] for r in rs], '.3f')} | {rg([r['n'] for r in rs], '.3f')} | "
                     f"{rg([r['maxabs21'] for r in rs], '.3g')} |")
    (OUT.with_suffix(".md")).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
