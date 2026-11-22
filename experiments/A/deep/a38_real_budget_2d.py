"""(A) 2D Cahn–Hilliard coarsening: the TFNO at the WPO-2D's real parameter budget (a33 setup).

a31 / a33 matched the TFNO to the WPO-2D by `numel` (15,473 against 18,737). The TFNO's Tucker factors are complex,
so in real degrees of freedom it had 27,409, 1.46x the WPO-2D's 18,737 (all real); see a36 and
`lfo.metrics.n_real_params`. a33's result "the TFNO is 2x more accurate per window (0.019 against 0.037) and in
rollout (window 20: 0.41 against 0.81)" was therefore measured with more parameters. This pass reruns the TFNO at
the WPO-2D's real budget, otherwise exactly as a33 (amplitude-scaled training starts, 6000 steps, batch 10,
lr 3e-3, 3 seeds, a31's evaluation).

- `tfno_r`: a31's TFNO3 (Tucker, n_modes (16, 16, 6)) with width 12 and rank 0.01: 19,037 real parameters (the
  closest over widths 12–19 and ranks 0.004–0.013).
- Reference: a33's `wpo_aug` and `tfno_aug` (not rerun).

**Predictions** (written before running).

1. `tfno_r`'s mean window-end error is below 0.028 (it keeps at least most of its edge over `wpo_aug`, 0.037).
2. `tfno_r`'s mean window-20 rollout error is below `wpo_aug`'s (0.81).

    python experiments/A/deep/a38_real_budget_2d.py      # results/a38_real_budget_2d.md / .json
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
import a33_amplitude_aug as A33  # noqa: E402
from lfo.metrics import n_real_params  # noqa: E402

OUT = HERE / "results" / "a38_real_budget_2d.json"


class TFNOReal(C.TFNO3):
    def __init__(self):
        torch.nn.Module.__init__(self)
        from neuralop.models import FNO
        self.net = FNO(n_modes=(16, 16, 6), in_channels=1, out_channels=1, hidden_channels=12, n_layers=4,
                       domain_padding=[0.0, 0.0, 0.25], factorization="tucker", rank=0.01)


def main():
    device = torch.device("cuda")
    data = C.make_data()
    xw = C.windows(data["train"])
    train = torch.cat([xw, A33.augmented(xw)])
    out = json.loads(OUT.read_text()) if OUT.exists() else []
    for seed in C.SEEDS:
        if any(r["seed"] == seed for r in out):
            continue
        t0 = time.time()
        torch.manual_seed(seed)
        m = TFNOReal().to(device)
        C.train(m, "tfno", train, device)
        m.eval()
        r = dict(model="tfno_r", seed=seed, params=n_real_params(m), **C.evaluate(m, "tfno", data, device),
                 seconds=round(time.time() - t0, 1))
        out.append(r)
        print({q: (round(v, 4) if isinstance(v, float) else v) for q, v in r.items()
               if q not in ("roll_per_window", "L", "L_true")}, flush=True)
        OUT.parent.mkdir(exist_ok=True)
        OUT.write_text(json.dumps(out, indent=1))
        del m
        torch.cuda.empty_cache()
    ref = json.loads((HERE / "results" / "a33_amplitude_aug.json").read_text())
    real = {"wpo_aug": 18737, "tfno_aug": 27409}
    rows = [(k, [r for r in ref if r["model"] == k], real[k]) for k in ("wpo_aug", "tfno_aug")]
    rows.append(("tfno_r", out, out[0]["params"]))
    rg = lambda xs, f: f"{min(xs):{f}}–{max(xs):{f}}"
    lines = ["# (A) 2D Cahn–Hilliard coarsening: the TFNO at the WPO-2D's real budget (a33 setup)", "",
             "Ranges over 3 seeds; rollout errors are means over seeds.", "",
             "| model | real params | window-end error (mean) | rollout error, window 1 / 5 / 20 | L(21) / truth | n |",
             "|---|---|---|---|---|---|"]
    for name, rs, par in rows:
        pw = np.mean([r["roll_per_window"] for r in rs], 0)
        lines.append(f"| {name} | {par} | {rg([r['win'] for r in rs], '.4f')} ({np.mean([r['win'] for r in rs]):.4f}) | "
                     f"{pw[0]:.3f} / {pw[4]:.3f} / {pw[-1]:.3g} | {rg([r['L21_ratio'] for r in rs], '.3f')} | "
                     f"{rg([r['n'] for r in rs], '.3f')} |")
    (OUT.with_suffix(".md")).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
