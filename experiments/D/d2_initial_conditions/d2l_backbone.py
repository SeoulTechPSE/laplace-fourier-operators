"""(D2) Cahn–Hilliard with initial fields: is the WPO backbone limited by its temporal or by its nonlinear depth?

d2k: four ways of feeding the initial field to the WPO give the same one-window error (0.0255–0.0259) against the
TFNO's 0.0126, so the gap is in the WPO backbone. Two parts of the backbone could be the limit:
- the temporal expressivity: each layer's time response is a sum of N = 6 poles per wavenumber, shared by all
  channel pairs, while the TFNO keeps 64 temporal Fourier modes with a weight per channel pair;
- the nonlinear depth: 4 layers of pointwise GELU between pole layers.
Three variants of d2g's `wpo_ic` at the closest parameter count to its 61,130 (same training, 3 seeds):
- `poles12`: 12 poles, width 16 (57,825; the next width overshoots by more);
- `layers6`: 6 layers, width 19 (60,178);
- `layers3`: 3 layers, width 26 (60,851).
Evaluation sets as d2k (`id`, `high`, `roll4`, `force_only`, `ic_only`); references `wpo_ic`, `tfno_chan` (d2g, d2j).

**Predictions** (written before running).

1. `poles12`: mean `id` error <= 0.9x `wpo_ic`'s (<= 0.0230): temporal expressivity limits the backbone.
2. `layers6`: mean `id` within +-10 % of `wpo_ic`'s.
3. `layers3`: mean `id` >= 1.1x `wpo_ic`'s.
4. No variant closes half the gap to the TFNO (every mean `id` > 0.019).

    python experiments/D/d2_initial_conditions/d2l_backbone.py      # results/d2l_backbone.md / .json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import d2g_ch_rollout as G  # noqa: E402
import d2k_ic_capacity as K  # noqa: E402
from lfo.models import WavenumberPoleOperator1d  # noqa: E402

R, E, M = G.R, G.E, G.M
VARIANTS = {"poles12": dict(n_poles=12, width=16), "layers6": dict(n_layers=6, width=19),
            "layers3": dict(n_layers=3, width=26)}
SEEDS = (0, 1, 2)
OUT = HERE / "results" / "d2l_backbone.json"


class WPOICv(nn.Module):
    def __init__(self, t, kw):
        super().__init__()
        self.net = WavenumberPoleOperator1d(t, **dict(G.WPO_KW, **kw), ic=True)

    def forward(self, f, u0):
        return self.net(f, u0)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t = np.linspace(0, M.T1, M.NT1)
    sets = K.make_sets()
    out = json.loads(OUT.read_text()) if OUT.exists() else []
    for kind, kw in VARIANTS.items():
        for seed in SEEDS:
            if any(r["model"] == kind and r["seed"] == seed for r in out):
                continue
            t0 = time.time()
            torch.manual_seed(seed)
            m = WPOICv(t, kw).to(device)
            R.train(m, dict(train=sets["train"]), device, 30.0)
            m.eval()
            r = dict(model=kind, seed=seed, params=K.n_params(m))
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
    rows = [(k, ref[k]) for k in ref] + [(k, [r for r in out if r["model"] == k]) for k in VARIANTS]
    mean = lambda rs, q: np.mean([r[q] for r in rs])
    lines = ["# (D2) Forced Cahn–Hilliard: WPO backbone variants at a matched budget (d2g setup, 6000 steps)", "",
             "| model | params | id | high (ratio to id) | roll4 | force_only | ic_only |", "|---|---|---|---|---|---|---|"]
    for name, rs in rows:
        lines.append(f"| {name} | {rs[0]['params']} | {mean(rs, 'id'):.4f} ({min(r['id'] for r in rs):.4f}–"
                     f"{max(r['id'] for r in rs):.4f}) | {mean(rs, 'high'):.4f} ({mean(rs, 'high') / mean(rs, 'id'):.2f}) | "
                     f"{mean(rs, 'roll4'):.4f} | {mean(rs, 'force_only'):.4f} | {mean(rs, 'ic_only'):.4f} |")
    (OUT.with_suffix(".md")).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
