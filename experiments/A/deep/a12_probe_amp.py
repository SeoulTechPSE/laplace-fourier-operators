"""(A) Deep operator: is the probe bias of a10 an amplitude effect? A probe-amplitude scan (no training).

a10 found that the probe of the stable WPO gives nu_hat = 0.111-0.114 (truth 0.1) in all three seeds. One candidate
explanation is that the probe amplitude eps = 0.01 lies below the smallest training scale (s = 0.03), so the probe
reads the model where it was never trained. If so, nu_hat should change with eps and approach 0.1 inside the
training range. If nu_hat is flat in eps, the bias sits in the model's linearization itself (for example an
effective viscosity absorbed from the nonlinear samples).

**Scan.** The a9 probe (single-mode inputs k = 1..16, odd part, per-mode identification, cuts, WLS for nu) at
eps in {0.003, 0.01, 0.03, 0.1, 0.3}, applied to:
- the solver (reference: the odd part picks up cubic terms as eps grows);
- the three a10 `plr30_stable` checkpoints (`runs/A/a10`).

**Predictions** (written before running).

1. Solver: nu_hat within 1 % of 0.1 for eps <= 0.03.
2. WPO: in every seed, nu_hat varies by less than 3 % (relative) across eps in {0.003, 0.01, 0.03}, so the bias is
   not an amplitude extrapolation.

    python experiments/A/deep/a12_probe_amp.py      # results/a12_probe_amp.md / .json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import a9_burgers as M  # noqa: E402
import a10_polelr as P  # noqa: E402
from lfo.models import WavenumberPoleOperator1d  # noqa: E402

EPSS = (0.003, 0.01, 0.03, 0.1, 0.3)
OUT = HERE / "results" / "a12_probe_amp.json"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t = np.linspace(0, M.T1, M.NT1)
    base, pdt = M.probe_inputs()                         # built at M.EPS
    models = {}
    for seed in M.SEEDS:
        m = WavenumberPoleOperator1d(t, **P.VARIANTS["plr30_stable"]).to(device)
        m.load_state_dict(torch.load(P.CKPT / f"plr30_stable_{seed}.pt", map_location=device))
        models[seed] = m.eval()
    out = []
    for eps in EPSS:
        fk = base * (eps / M.EPS)
        r = dict(eps=eps, solver=M.probe(lambda spec: M.fields(spec, pdt), fk, pdt, device))
        r["wpo"] = {s: M.model_probe(m, fk, pdt, device) for s, m in models.items()}
        out.append(r)
        print(f"eps {eps:g}", f"solver nu {r['solver']['nu_hat']:.5f} disp {r['solver']['disp_err']:.2e}",
              "wpo nu", [round(r["wpo"][s]["nu_hat"], 5) for s in models],
              "disp", [f"{r['wpo'][s]['disp_err']:.2e}" for s in models], flush=True)
        OUT.write_text(json.dumps(out, indent=1, default=float))
    summarize(out)


def summarize(out):
    seeds = list(out[0]["wpo"])
    lines = ["# (A) Probe-amplitude scan: solver and a10 stable WPO", "",
             "ν̂ (dispersion error) per probe amplitude ε; training scales s ∈ [0.03, 1].", "",
             "| ε | solver | " + " | ".join(f"WPO seed {s}" for s in seeds) + " |", "|---|---|" + "---|" * len(seeds)]
    for r in out:
        cell = lambda p: f"{p['nu_hat']:.4f} ({p['disp_err']:.1e})"
        lines.append(f"| {r['eps']:g} | {cell(r['solver'])} | " + " | ".join(cell(r["wpo"][s]) for s in seeds) + " |")
    (OUT.with_suffix(".md")).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
