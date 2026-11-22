"""(D2) Cahn–Hilliard with initial fields: the TFNO at the WPO's real parameter budget.

d2g–d2l compared `wpo_ic` (61,130 parameters, all real) with `tfno_chan` (59,113 by `numel`). The TFNO's Tucker
factors are complex, and `numel` counts a complex weight once: in real degrees of freedom `tfno_chan` has 110,617,
1.81x the WPO's (found in (A) a36; budgets are now matched with `lfo.metrics.n_real_params`). Its one-window edge
(0.0126 against 0.0255, d2g; the "2x gap" that d2i–d2l traced to the WPO backbone) was therefore measured with
almost twice the parameters. This pass reruns the TFNO at the WPO's real budget.

- `tfno_r`: d2b's TFNO (Tucker, n_modes (32, 64), two input channels, lr 3e-3, d2g's 6000 steps) with width 21 and
  rank 0.014: 61,320 real parameters (the closest over widths 18–27 and ranks 0.006–0.019).
- References: `wpo_ic` and `tfno_chan` from d2g / d2j (not rerun).
- Evaluation sets as d2k / d2l: `id`, `high` (unseen initial-field modes), `roll4`, `force_only`, `ic_only`.
  3 seeds.

**Predictions** (written before running).

1. `tfno_r`'s mean `id` error is above `tfno_chan`'s (0.0126) but below 0.019: the TFNO keeps more than half of
   its edge over the WPO (0.0255) at an equal real budget.
2. `tfno_r`'s `high` / `id` ratio is >= 5 (the unseen-mode weakness does not depend on the budget; `tfno_chan`
   10.4, `wpo_ic` 2.35).

    python experiments/D/d2_initial_conditions/d2m_real_budget.py      # results/d2m_real_budget.md / .json
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
import d2g_ch_rollout as G  # noqa: E402
import d2k_ic_capacity as K  # noqa: E402
from lfo.metrics import n_real_params  # noqa: E402

R, E, M = G.R, G.E, G.M
SEEDS = (0, 1, 2)
OUT = HERE / "results" / "d2m_real_budget.json"


class TFNOReal(R.TFNOChan):
    def __init__(self):
        torch.nn.Module.__init__(self)
        from neuralop.models import FNO
        self.net = FNO(n_modes=(32, 64), in_channels=2, out_channels=1, hidden_channels=21, n_layers=4,
                       domain_padding=[0.0, 0.25], factorization="tucker", rank=0.014)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sets = K.make_sets()
    out = json.loads(OUT.read_text()) if OUT.exists() else []
    for seed in SEEDS:
        if any(r["seed"] == seed for r in out):
            continue
        t0 = time.time()
        torch.manual_seed(seed)
        m = TFNOReal().to(device)
        R.train(m, dict(train=sets["train"]), device, 30.0)
        m.eval()
        r = dict(model="tfno_r", seed=seed, params=n_real_params(m))
        for s in ("id", "high", "force_only", "ic_only"):
            f, u0, y = sets[s]
            r[s] = float(M.A.rel(R.predict(m, f, u0, device), y))
        f, u0, y = sets["roll4"]
        r["roll4"] = float(M.A.rel(E.rollout(m, f, u0, device, 4), y))
        r["seconds"] = round(time.time() - t0, 1)
        out.append(r)
        print(r, flush=True)
        OUT.parent.mkdir(exist_ok=True)
        OUT.write_text(json.dumps(out, indent=1))
        del m
        torch.cuda.empty_cache()
    g = json.loads((HERE / "results" / "d2g_ch_rollout.json").read_text())
    j = json.loads((HERE / "results" / "d2j_ch_paths.json").read_text())
    ref = {k: [dict(rg, **{s: rj[s] for s in ("force_only", "ic_only")}) for rg in g if rg["model"] == k
               for rj in j if rj["model"] == k and rj["seed"] == rg["seed"]] for k in ("wpo_ic", "tfno_chan")}
    real = {"wpo_ic": 61130, "tfno_chan": 110617}
    rows = [(k, ref[k], real[k]) for k in ref] + [("tfno_r", out, out[0]["params"])]
    mean = lambda rs, q: np.mean([r[q] for r in rs])
    lines = ["# (D2) Forced Cahn–Hilliard: the TFNO at the WPO's real budget (d2g setup, 6000 steps)", "",
             "| model | real params | id | high (ratio to id) | roll4 | force_only | ic_only |", "|---|---|---|---|---|---|---|"]
    for name, rs, par in rows:
        lines.append(f"| {name} | {par} | {mean(rs, 'id'):.4f} ({min(r['id'] for r in rs):.4f}–{max(r['id'] for r in rs):.4f}) | "
                     f"{mean(rs, 'high'):.4f} ({mean(rs, 'high') / mean(rs, 'id'):.2f}) | {mean(rs, 'roll4'):.4f} | "
                     f"{mean(rs, 'force_only'):.4f} | {mean(rs, 'ic_only'):.4f} |")
    (OUT.with_suffix(".md")).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
