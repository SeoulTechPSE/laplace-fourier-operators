"""(A) a32 post hoc: does the WPO-2D window map restore the amplitude of the separated phases?

Written after a32's first results (no predictions): with stable poles (Re mu < 0 in every layer) 2 of 3 WPO-2D seeds
still diverge in the coarsening rollout, so the growth comes from the pointwise path, not from the poles. The true
window map saturates the field at +-1 (u^3 - u); a learned map that has seen only |u| <~ 1.05 may not.

Test: true test states s at t0 = 5, 10, 15, scaled by a in {1, 1.1, 1.25, 1.5, 2}; gain(a) = max |M(a s)| /
max |a s| at the window end, and the truth's gain from the solver (128^2 is not needed: the scaled 64^2 states are
resolved on 64^2 over one window, checked against 128^2). A gain >= 1 at a > 1 means no restoring force, so an
overshoot persists or grows. Also max |u| after each rollout window. Runs retrained with a31 / a32 settings and
seeds (training is deterministic: a31's seed 0 reproduced exactly).

    python experiments/A/deep/a32b_amplitude.py      # results/a32b_amplitude.md / .json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import a31_coarsening as C  # noqa: E402
import a32_growth_cap as G  # noqa: E402

RUNS = (("wpo", 0), ("wpo", 1), ("wpo_stable", 0), ("wpo_stable", 1), ("tfno", 0))
SCALES, STARTS = (1.0, 1.1, 1.25, 1.5, 2.0), (5, 10, 15)
OUT = HERE / "results" / "a32b_amplitude.json"


def true_end(s, n):
    """The solver over one window from (b, NM, NM) states, on an n^2 grid; window-end state on 64^2."""
    sk = torch.fft.rfft2(s, dim=(1, 2))
    big = torch.zeros(s.shape[0], n, n // 2 + 1, dtype=sk.dtype)
    h = C.NM // 2
    big[:, :h, :h], big[:, -h:, :h] = sk[:, :h, :h], sk[:, -h:, :h]
    big *= (n / C.NM) ** 2
    uk = ch_free(big, n)
    out = torch.zeros_like(sk)
    out[:, :h, :h], out[:, -h:, :h] = uk[:, :h, :h], uk[:, -h:, :h]
    return torch.fft.irfft2(out / (n / C.NM) ** 2, s=(C.NM, C.NM), dim=(1, 2))


def ch_free(u0k, n):
    dt = C.DT / 4                                                            # a = 2 makes the cubic term stiff
    steps = int(round(1.0 / dt))
    uk = C.ch.solve2d_free_torch(u0k, dt, C.MOB, C.KAPPA, n, steps, stride=steps, device=torch.device("cuda"))
    return uk[..., -1].cpu()


def model(kind, seed, device, xw):
    if kind in G.VARIANTS:
        torch.manual_seed(seed)
        m = G.WPOv(G.VARIANTS[kind]).to(device)
    else:
        m = C.build(kind, seed, device)
    C.train(m, "wpo" if kind != "fno2d" else kind, xw, device)
    return m.eval()


def main():
    device = torch.device("cuda")
    data = C.make_data()
    te = data["test"]
    states = torch.cat([te[..., C.FR * t0] for t0 in STARTS])
    truth = {}
    for a in SCALES:
        e64, e128 = true_end(a * states, 64), true_end(a * states, 128)
        truth[str(a)] = dict(gain=float(e128.abs().amax((1, 2)).mean() / (a * states).abs().amax((1, 2)).mean()),
                        res_check=float((e64 - e128).norm() / e128.norm()))
    print("truth", truth, flush=True)
    out = json.loads(OUT.read_text()) if OUT.exists() else dict(truth=truth, runs=[])
    xw = C.windows(data["train"])
    for kind, seed in RUNS:
        if any(r["model"] == kind and r["seed"] == seed for r in out["runs"]):
            continue
        m = model(kind, seed, device, xw)
        gain = {}
        for a in SCALES:
            p = C.batched_end(m, kind, a * states, device)
            gain[str(a)] = float(p.abs().amax((1, 2)).mean() / (a * states).abs().amax((1, 2)).mean())
        roll = C.rollout(m, kind, te, device)
        r = dict(model=kind, seed=seed, gain=gain, maxabs=[float(roll[..., i].abs().max()) for i in range(roll.shape[-1])])
        out["runs"].append(r)
        print(kind, seed, {a: round(v, 3) for a, v in gain.items()}, np.round(r["maxabs"][::5], 2), flush=True)
        OUT.write_text(json.dumps(out, indent=1))
        del m
        torch.cuda.empty_cache()
    lines = ["# (A) a32 post hoc: amplitude gain of the window map (max |M(a s)| / max |a s|)", "",
             "| run | " + " | ".join(f"a = {a}" for a in SCALES) + " | max abs u after window 0 / 5 / 10 / 15 / 20 |",
             "|---|" + "---|" * (len(SCALES) + 1),
             "| truth (solver, 128^2) | " + " | ".join(f"{out['truth'][str(a)]['gain']:.3f}" for a in SCALES) + " | — |"]
    for r in out["runs"]:
        g = r["gain"]
        lines.append(f"| {r['model']} s{r['seed']} | " + " | ".join(f"{g[str(a)]:.3f}" for a in SCALES)
                     + " | " + " / ".join(f"{v:.3g}" for v in r["maxabs"][::5]) + " |")
    (OUT.with_suffix(".md")).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
