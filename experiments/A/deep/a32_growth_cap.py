"""(A) Application 2, stage 2b: the growth cap of WPO-2D on 2D Cahn–Hilliard coarsening.

In a31 one WPO-2D seed (0) diverged in the 20-window rollout (max |u(21)| 4e5) with layers 2–4 at the growth cap
(max Re mu 1.10 / 1.01 / 1.01 with re_max = 1.0 and leak 0.05, so the leak lets Re mu pass the cap). About the
separated phases u = +-1 the linearization is stable, lambda = -M k^2 (2 + kappa k^2), so in the coarsening regime the
data need no growing pole. This script asks whether a tighter cap removes the divergence and what it costs per window.

Everything (data, windows, training, metrics) is a31's (`a31_coarsening.py`, imported); only the WPO-2D pole
constraint changes:
- `wpo_stable`: `stable=True`, Re mu = -softplus(.) < 0 in every layer (no growth at all);
- `wpo_cap03`: re_max = 0.3, no leak (growth at most e^0.3 per unit time in a layer);
- `wpo_cap1`: re_max = 1.0, no leak (a31's cap without the leak, to separate the leak from the cap level).
a31's `wpo` (re_max 1.0, leak 0.05) and `tfno` rows are the references (`results/a31_coarsening.json`).
Same parameter count as a31's `wpo` (18,737; the constraint adds no parameters). 3 seeds.

Extra diagnostic per run: max Re mu over the kept modes, per layer.

**Predictions** (written before running).

1. `wpo_stable`: no divergence (max |u(21)| <= 2) in every seed, and L(21) within 10 % of the truth and n within
   +-0.05 of 0.271 in every seed.
2. `wpo_stable`: mean window-end error within 1.25x of a31's `wpo` (0.034), i.e. <= 0.043.
3. `wpo_cap03`: no divergence in every seed.
4. `wpo_cap1`: at least one of 3 seeds reaches max Re mu >= 0.95 in some layer (the fit wants growth when allowed).
5. The TFNO stays better pointwise: every variant's mean rollout error at window 20 is above a31's `tfno` mean
   (0.37).

    python experiments/A/deep/a32_growth_cap.py      # results/a32_growth_cap.md / .json
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
from lfo.models import WavenumberPoleOperator2d  # noqa: E402

BASE = dict(C.WPO_KW)
VARIANTS = {
    "wpo_stable": dict(BASE, re_max=None, re_leak=0.0, stable=True),
    "wpo_cap03": dict(BASE, re_max=0.3, re_leak=0.0),
    "wpo_cap1": dict(BASE, re_max=1.0, re_leak=0.0),
}
OUT = HERE / "results" / "a32_growth_cap.json"


class WPOv(C.WPO):
    def __init__(self, kw):
        torch.nn.Module.__init__(self)
        self.net = WavenumberPoleOperator2d(C.TW, **kw)


@torch.no_grad()
def max_re(m, device):
    kx, ky = m.net.K[0].wavenumbers(C.NM, C.NM)
    h = BASE["modes"]
    sel = ((kx.abs() <= h) & (ky <= h)).to(device)
    return [round(float(K.mu(kx.to(device), ky.to(device)).real[..., sel].max()), 3) for K in m.net.K]


def main():
    device = torch.device("cuda")
    data = C.make_data()
    xw = C.windows(data["train"])
    out = json.loads(OUT.read_text()) if OUT.exists() else []
    for kind, kw in VARIANTS.items():
        for seed in C.SEEDS:
            if any(r["model"] == kind and r["seed"] == seed for r in out):
                continue
            t0 = time.time()
            torch.manual_seed(seed)
            m = WPOv(kw).to(device)
            C.train(m, "wpo", xw, device)
            m.eval()
            r = dict(model=kind, seed=seed, params=C.n_params(m), max_re=max_re(m, device),
                     **C.evaluate(m, "wpo", data, device), seconds=round(time.time() - t0, 1))
            out.append(r)
            print({q: (round(v, 4) if isinstance(v, float) else v) for q, v in r.items()
                   if q not in ("roll_per_window", "L", "L_true")}, flush=True)
            OUT.parent.mkdir(exist_ok=True)
            OUT.write_text(json.dumps(out, indent=1))
            del m
            torch.cuda.empty_cache()
    ref = json.loads((HERE / "results" / "a31_coarsening.json").read_text())
    rows = [("a31 " + k, [r for r in ref if r["model"] == k]) for k in ("wpo", "tfno")]
    rows += [(k, [r for r in out if r["model"] == k]) for k in VARIANTS]
    rg = lambda xs, f: f"{min(xs):{f}}–{max(xs):{f}}"
    lines = ["# (A) 2D Cahn–Hilliard coarsening: the WPO-2D growth cap", "",
             f"Truth: L(21) = {ref[0]['L_true'][-1]:.3f}, n = {ref[0]['n_true']:.3f}. Ranges over 3 seeds; rollout "
             "errors are means over seeds.", "",
             "| model | window-end error | rollout error, window 1 / 5 / 20 | L(21) / truth | n | S(k, 21) error | "
             "max abs u(21) | max Re mu per layer (per seed) |",
             "|---|---|---|---|---|---|---|---|"]
    for name, rs in rows:
        if not rs:
            continue
        pw = np.mean([r["roll_per_window"] for r in rs], 0)
        mr = "; ".join("/".join(f"{v:.2f}" for v in r["max_re"]) for r in rs) if "max_re" in rs[0] else "—"
        lines.append(f"| {name} | {rg([r['win'] for r in rs], '.4f')} | {pw[0]:.3f} / {pw[4]:.3f} / {pw[-1]:.3g} | "
                     f"{rg([r['L21_ratio'] for r in rs], '.3f')} | {rg([r['n'] for r in rs], '.3f')} | "
                     f"{rg([r['S21_err'] for r in rs], '.3g')} | {rg([r['maxabs21'] for r in rs], '.3g')} | {mr} |")
    lines += ["", "Per seed (window-end error, window-20 error, max abs u(21)):", ""]
    for name, rs in rows:
        lines.append(f"- {name}: " + "; ".join(f"s{r['seed']} {r['win']:.4f}, {r['roll_per_window'][-1]:.3g}, "
                                              f"{r['maxabs21']:.3g}" for r in rs))
    (OUT.with_suffix(".md")).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
