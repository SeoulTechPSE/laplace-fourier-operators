"""(A) Application 2, stage 2c: training WPO-2D on amplitude-perturbed starts (a restoring force for coarsening).

a32 / a32b: the WPO-2D window map has no restoring force on the amplitude. Scaled by a = 2, a true state comes back
at 0.50x its maximum under the truth, 0.76x under the TFNO and 1.5–4.8x under every WPO-2D run (stable poles
included), so overshoots past +-1 persist, and 1–3 of 3 seeds diverge in the 20-window coarsening rollout whatever
the growth cap. The training starts all have |u| <~ 1.05, so the data never show the map pulling an overshoot back.

**Remedy tested.** Each of a31's 960 training windows gets one scaled copy: start a s, a ~ U[0.7, 1.6] (rng seed
1501, one draw per window), target the true solution from a s over the window (11 frames). The truth is computed
by embedding the 64^2 start in 128^2 and running `solve2d_free_torch` with dt = a31's dt / 2 (the cubic term is
stiffer at a = 1.6), then truncating back. Training set: 960 original + 960 scaled windows; everything else as
a31 (6000 steps, batch 10, same models, seeds and parameter counts).
- `wpo_aug`: a31's `wpo` (cap 1.0, leak 0.05);
- `tfno_aug`: a31's `tfno`.
References: a31's `wpo` and `tfno` (`results/a31_coarsening.json`) and a32b's gains (`results/a32b_amplitude.json`).

**Checks before any fit** (8 test states, t0 = 5). dt / 2 against dt / 4: 4–5e-6 at a = 0.7, 1, 1.6. Re-solving
from a = 1 against the stored truth one unit later: 3.3e-5 (the 64^2 truncation). One window from a = 1.6 brings
max |u| back to 1.03.

**Metrics.** a31's (window-end error on the unscaled test windows, 20-window rollout, L(21), n, S(k, 21), max |u|)
plus a32b's gain max |M(a s)| / max |a s| for a in {1, 1.1, 1.25, 1.5, 2} on test states at t0 = 5, 10, 15
(a = 2 is outside the training range).

**Predictions** (written before running).

1. `wpo_aug`: no divergence (max |u(21)| <= 2) in any of 3 seeds.
2. `wpo_aug`: gain < 1 at a = 1.25 and at a = 1.5 in every seed.
3. `wpo_aug`: mean window-end error <= 1.2x a31's `wpo` (0.034), i.e. <= 0.041.
4. `wpo_aug`: mean window-20 rollout error <= 0.74 (a31's stable `wpo` seeds gave 0.67 and 0.74).
5. `tfno_aug` stays the best pointwise model: its mean window-20 rollout error is below `wpo_aug`'s.

    python experiments/A/deep/a33_amplitude_aug.py      # results/a33_amplitude_aug.md / .json
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
import a31_coarsening as C  # noqa: E402

KINDS = {"wpo_aug": "wpo", "tfno_aug": "tfno"}
SCALES, STARTS, A_RANGE = (1.0, 1.1, 1.25, 1.5, 2.0), (5, 10, 15), (0.7, 1.6)
OUT = HERE / "results" / "a33_amplitude_aug.json"
CACHE = C.CACHE.parent / "aug.pt"


def true_window(s, n=128, bs=64):
    """The truth over one window (11 frames) from 64^2 states s (b, NM, NM): (b, NM, NM, FR + 1)."""
    h, dt = C.NM // 2, C.DT / 2
    steps = int(round(1.0 / dt))
    outs = []
    for i in range(0, len(s), bs):
        sk = torch.fft.rfft2(s[i:i + bs], dim=(1, 2))
        big = torch.zeros(sk.shape[0], n, n // 2 + 1, dtype=sk.dtype)
        big[:, :h, :h], big[:, -h:, :h] = sk[:, :h, :h], sk[:, -h:, :h]
        uk = C.ch.solve2d_free_torch(big * (n / C.NM) ** 2, dt, C.MOB, C.KAPPA, n, steps, stride=steps // C.FR,
                                     device="cuda").cpu()
        o = torch.zeros(sk.shape[0], C.NM, C.NM // 2 + 1, uk.shape[-1], dtype=uk.dtype)
        o[:, :h, :h], o[:, -h:, :h] = uk[:, :h, :h], uk[:, -h:, :h]
        outs.append(torch.fft.irfft2(o / (n / C.NM) ** 2, s=(C.NM, C.NM), dim=(1, 2)))
    return torch.cat(outs)


def augmented(xw):
    if CACHE.exists():
        return torch.load(CACHE)
    a = torch.tensor(np.random.default_rng(1501).uniform(*A_RANGE, size=len(xw)))
    aw = true_window(a[:, None, None] * xw[..., 0])
    torch.save(aw, CACHE)
    return aw


@torch.no_grad()
def gains(m, kind, states, device):
    return {str(a): float(C.batched_end(m, kind, a * states, device).abs().amax((1, 2)).mean()
                          / (a * states).abs().amax((1, 2)).mean()) for a in SCALES}


def main():
    device = torch.device("cuda")
    data = C.make_data()
    xw = C.windows(data["train"])
    aw = augmented(xw)
    print("augmented", tuple(aw.shape), "max |u| start / end", float(aw[..., 0].abs().max()),
          float(aw[..., -1].abs().max()), flush=True)
    train = torch.cat([xw, aw])
    te = data["test"]
    states = torch.cat([te[..., C.FR * t0] for t0 in STARTS])
    out = json.loads(OUT.read_text()) if OUT.exists() else []
    for kind, base in KINDS.items():
        for seed in C.SEEDS:
            if any(r["model"] == kind and r["seed"] == seed for r in out):
                continue
            t0 = time.time()
            m = C.build(base, seed, device)
            C.train(m, base, train, device)
            m.eval()
            r = dict(model=kind, seed=seed, params=C.n_params(m), gain=gains(m, base, states, device),
                     **C.evaluate(m, base, data, device), seconds=round(time.time() - t0, 1))
            out.append(r)
            print({q: (round(v, 4) if isinstance(v, float) else v) for q, v in r.items()
                   if q not in ("roll_per_window", "L", "L_true")}, flush=True)
            OUT.parent.mkdir(exist_ok=True)
            OUT.write_text(json.dumps(out, indent=1))
            del m
            torch.cuda.empty_cache()
    ref = json.loads((HERE / "results" / "a31_coarsening.json").read_text())
    rows = [("a31 " + k, [r for r in ref if r["model"] == k]) for k in ("wpo", "tfno")]
    rows += [(k, [r for r in out if r["model"] == k]) for k in KINDS]
    rg = lambda xs, f: f"{min(xs):{f}}–{max(xs):{f}}"
    lines = ["# (A) 2D Cahn–Hilliard coarsening: training on amplitude-perturbed starts", "",
             f"Truth: L(21) = {ref[0]['L_true'][-1]:.3f}, n = {ref[0]['n_true']:.3f}. Ranges over 3 seeds; rollout "
             "errors are means over seeds.", "",
             "| model | window-end error | rollout error, window 1 / 5 / 20 | L(21) / truth | n | S(k, 21) error | "
             "max abs u(21) |", "|---|---|---|---|---|---|---|"]
    for name, rs in rows:
        pw = np.mean([r["roll_per_window"] for r in rs], 0)
        lines.append(f"| {name} | {rg([r['win'] for r in rs], '.4f')} | {pw[0]:.3f} / {pw[4]:.3f} / {pw[-1]:.3g} | "
                     f"{rg([r['L21_ratio'] for r in rs], '.3f')} | {rg([r['n'] for r in rs], '.3f')} | "
                     f"{rg([r['S21_err'] for r in rs], '.3g')} | {rg([r['maxabs21'] for r in rs], '.3g')} |")
    lines += ["", "Gain max |M(a s)| / max |a s| (per seed; truth 1.00 / 0.91 / 0.80 / 0.67 / 0.50):", "",
              "| run | " + " | ".join(f"a = {a}" for a in SCALES) + " |", "|---|" + "---|" * len(SCALES)]
    for r in out:
        lines.append(f"| {r['model']} s{r['seed']} | " + " | ".join(f"{r['gain'][str(a)]:.3f}" for a in SCALES) + " |")
    lines += ["", "Per seed (window-end error, window-20 error, max abs u(21)):", ""]
    for name, rs in rows:
        lines.append(f"- {name}: " + "; ".join(f"s{r['seed']} {r['win']:.4f}, {r['roll_per_window'][-1]:.3g}, "
                                              f"{r['maxabs21']:.3g}" for r in rs))
    (OUT.with_suffix(".md")).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
