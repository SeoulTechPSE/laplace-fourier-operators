"""(A) Application 2, stage 2d: transfer to a larger box (4 pi, same grid spacing) without retraining.

The pole operator's poles mu_n(k) are functions of the physical wavenumber, so a WPO-2D trained on the 2 pi box
can be evaluated on a 4 pi box at the same grid spacing by giving its layers the new length and twice the mode
count (the same physical band |k| < 12). The TFNO's spectral weights are indexed by mode number: on the larger box
mode m is the physical wavenumber m / 2, so the same weights act on a different band; there is no consistent way
to transfer them. A larger box is also the cleaner test of the coarsening law (a31: L(21) is 0.56 of the 2 pi box).

**Models.** a33's `wpo_aug` and `tfno_aug` (amplitude-perturbed training starts; a33 had no divergence), retrained
with a33's data and seeds (3 each) and saved (`runs/A/a34/`). On the 4 pi box: `wpo_aug` with length 4 pi and
modes 24 in every layer; `tfno_aug` unchanged.

**Truth on the 4 pi box.** Rescaling x by 1/2 maps the box to 2 pi with M' = M / 4, kappa' = kappa / 4, so a31's
solver runs unchanged: 256^2 (the same spacing and dealiased physical band |k| <= 32 as a31's 128^2), dt 4e-4,
white noise of standard deviation 0.05 on that band (the same physical statistics), frames every 0.1 to t = 21,
truncated to 128^2 for the models (|k| <= 31.5). 16 trajectories, data seed 1600.

**Metrics** (as a31, with physical wavenumbers): window-end error from true starts (16 x 20 windows) on both boxes;
on the 4 pi box the 20-window rollout from t = 1, L(t) = 2 pi / <k>, n on t = 2..21, max |u(21)|.

**Predictions** (written before running).

1. `wpo_aug`: window-end error on the 4 pi box <= 1.3x its 2 pi error, in every seed.
2. `tfno_aug`: window-end error on the 4 pi box >= 2x its 2 pi error, in every seed.
3. On the 4 pi box, `wpo_aug`'s mean window-end error is below `tfno_aug`'s.
4. `wpo_aug` on the 4 pi box: no divergence (max |u(21)| <= 2), L(21) within 10 % and n within +-0.05 of the
   truth, in every seed.

A fifth prediction drafted before the truth was generated ("n on the 4 pi box above a31's 0.271") is withdrawn:
the truth was computed as a check before this commit, and its n is known (below), so it is no longer a prediction.

**Checks before any fit.**
- Transfer: a 2 pi test state tiled 2 x 2 is a 4 pi field with the same physical content. The transferred
  (untrained) WPO-2D reproduces the tiled 2 pi output to 3.9e-16; the TFNO differs by 2.4 % (untrained weights).
- Truth on 4 pi: L(1) = 1.51, L(21) = 3.45, n = 0.261 (2 pi box: 3.52, 0.271); max |u| 1.04. The 2 pi box's L(21)
  is not visibly limited by its size.

    python experiments/A/deep/a34_domain.py [--check | --truth]      # results/a34_domain.md / .json
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
import a33_amplitude_aug as A33  # noqa: E402

KINDS = {"wpo_aug": "wpo", "tfno_aug": "tfno"}
NS4, NM4, N4, SEED4 = 256, 128, 16, 1600
CKPT = C.ROOT / "runs" / "A" / "a34"
CACHE4 = CKPT / "data4pi.pt"
OUT = HERE / "results" / "a34_domain.json"


def truth4():
    if CACHE4.exists():
        return torch.load(CACHE4)
    rng = np.random.default_rng(SEED4)
    steps, stride = int(round(C.T_END / C.DT)), int(round(1 / (C.FR * C.DT)))
    parts = []
    for _ in range(N4 // 4):
        u0k = np.fft.rfft2(0.05 * rng.normal(size=(4, NS4, NS4)), axes=(1, 2))
        kx = np.fft.fftfreq(NS4, 1 / NS4)[:, None]
        ky = np.fft.rfftfreq(NS4, 1 / NS4)[None, :]
        u0k[:, ~((np.abs(kx) <= NS4 / 4) & (ky <= NS4 / 4))] = 0
        uk = C.ch.solve2d_free_torch(u0k, C.DT, C.MOB / 4, C.KAPPA / 4, NS4, steps, stride=stride, device="cuda")
        h = NM4 // 2 - 1
        out = torch.zeros(uk.shape[0], NM4, NM4 // 2 + 1, uk.shape[-1], dtype=uk.dtype, device=uk.device)
        out[:, :h + 1, :h + 1] = uk[:, :h + 1, :h + 1]
        out[:, NM4 - h:, :h + 1] = uk[:, NS4 - h:, :h + 1]
        parts.append(torch.fft.irfft2(out * (NM4 / NS4) ** 2, s=(NM4, NM4), dim=(1, 2)).cpu())
        del uk, out
        torch.cuda.empty_cache()
    u = torch.cat(parts)
    CKPT.mkdir(parents=True, exist_ok=True)
    torch.save(u, CACHE4)
    return u


def stats(u, box):
    """L(t) = 2 pi / <k> with physical k = 2 pi / box * index; u (n, N, N, m)."""
    n = u.shape[1]
    kx = torch.fft.fftfreq(n, 1 / n)[:, None]
    ky = torch.fft.rfftfreq(n, 1 / n)[None, :]
    km = torch.sqrt(kx ** 2 + ky ** 2) * (2 * math.pi / box)
    wk = torch.full(km.shape, 2.0)
    wk[:, 0] = 1
    S = (torch.fft.rfft2(u, dim=(1, 2)).abs() ** 2 * wk[..., None]).mean(0)
    S[0, 0] = 0
    return 2 * math.pi * S.sum((0, 1)) / (km[..., None] * S).sum((0, 1))


def to_box(m, kind, box):
    if kind == "wpo":
        for K in m.net.K:
            K.length = (box, box)
            K.modes = int(round(C.WPO_KW["modes"] * box / (2 * math.pi)))
    return m


@torch.no_grad()
def batched_end(m, kind, u0, device, bs):
    return torch.cat([C.step_end(m, kind, u0[i:i + bs].to(device)).cpu().to(torch.float64)
                      for i in range(0, len(u0), bs)])


def win_error(m, kind, u, device, bs):
    xw = C.windows(u)
    return float(C.rel(batched_end(m, kind, xw[..., 0], device, bs), xw[..., -1]))


@torch.no_grad()
def rollout(m, kind, u, device, bs):
    s = u[..., C.FR]
    out = [s]
    for _ in range(C.NWIN):
        s = batched_end(m, kind, s, device, bs)
        out.append(s)
    return torch.stack(out, -1)


def tile_check(device):
    """A 2 pi field tiled 2 x 2 is a 4 pi field with the same physical content: the transferred model must give the
    tiled 2 pi output. Untrained models (seed 0)."""
    data = C.make_data()
    s = data["test"][:2, ..., C.FR * 5]
    for kind in KINDS.values():
        m = C.build(kind, 0, device).eval()
        a = batched_end(m, kind, s, device, 2)
        to_box(m, kind, 4 * math.pi)
        b = batched_end(m, kind, s.repeat(1, 2, 2), device, 2)
        print(kind, "tiled 4 pi output against tiled 2 pi output: rel diff",
              float((b - a.repeat(1, 2, 2)).norm() / b.norm()), flush=True)


def main():
    device = torch.device("cuda")
    if "--check" in sys.argv:
        tile_check(device)
        return
    u4 = truth4()
    L4 = stats(u4[..., C.FR::C.FR], 4 * math.pi)
    n4 = C.exponent(L4)
    print("truth 4pi: L(1..21)", np.round(L4.numpy()[::5], 3), "n", round(n4, 4), "max |u|", float(u4.abs().max()),
          flush=True)
    if "--truth" in sys.argv:
        return
    data = C.make_data()
    xw = C.windows(data["train"])
    train = torch.cat([xw, A33.augmented(xw)])
    out = json.loads(OUT.read_text()) if OUT.exists() else []
    for kind, base in KINDS.items():
        for seed in C.SEEDS:
            if any(r["model"] == kind and r["seed"] == seed for r in out):
                continue
            t0 = time.time()
            m = C.build(base, seed, device)
            path = CKPT / f"{kind}_{seed}.pt"
            if path.exists():
                m.load_state_dict(torch.load(path))
            else:
                C.train(m, base, train, device)
                torch.save(m.state_dict(), path)
            m.eval()
            r = dict(model=kind, seed=seed, win_2pi=win_error(m, base, data["test"], device, 16))
            to_box(m, base, 4 * math.pi)
            r["win_4pi"] = win_error(m, base, u4, device, 4)
            p = rollout(m, base, u4, device, 4)
            y = u4[..., C.FR::C.FR]
            Lp = stats(p, 4 * math.pi)
            r.update(roll_per_window=[float(C.rel(p[..., i], y[..., i])) for i in range(1, C.NWIN + 1)],
                     L=Lp.tolist(), L_true=L4.tolist(), n=C.exponent(Lp), n_true=n4,
                     L21_ratio=float(Lp[-1] / L4[-1]), maxabs21=float(p[..., -1].abs().max()),
                     seconds=round(time.time() - t0, 1))
            out.append(r)
            print({q: (round(v, 4) if isinstance(v, float) else v) for q, v in r.items()
                   if q not in ("roll_per_window", "L", "L_true")}, flush=True)
            OUT.parent.mkdir(exist_ok=True)
            OUT.write_text(json.dumps(out, indent=1))
            del m
            torch.cuda.empty_cache()
    rg = lambda xs, f: f"{min(xs):{f}}–{max(xs):{f}}"
    lines = ["# (A) 2D Cahn–Hilliard coarsening: transfer from the 2 pi box to a 4 pi box (same spacing)", "",
             f"Truth on 4 pi: L(1) = {L4[0]:.3f}, L(21) = {L4[-1]:.3f}, n = {n4:.3f} (a31, 2 pi box: L(21) = 3.522, "
             "n = 0.271). Ranges over 3 seeds.", "",
             "| model | window-end error 2 pi | window-end error 4 pi | ratio | rollout 4 pi, window 1 / 5 / 20 | "
             "L(21) / truth | n | max abs u(21) |", "|---|---|---|---|---|---|---|---|"]
    for kind in KINDS:
        rs = [r for r in out if r["model"] == kind]
        pw = np.mean([r["roll_per_window"] for r in rs], 0)
        lines.append(f"| {kind} | {rg([r['win_2pi'] for r in rs], '.4f')} | {rg([r['win_4pi'] for r in rs], '.4f')} | "
                     f"{rg([r['win_4pi'] / r['win_2pi'] for r in rs], '.2f')} | {pw[0]:.3f} / {pw[4]:.3f} / {pw[-1]:.3g} | "
                     f"{rg([r['L21_ratio'] for r in rs], '.3f')} | {rg([r['n'] for r in rs], '.3f')} | "
                     f"{rg([r['maxabs21'] for r in rs], '.3g')} |")
    (OUT.with_suffix(".md")).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
