"""(A) Application 1: delay identification under output noise (FOPDT / SOPDT).

a20 (`a20_fopdt.py`) was noise-free. Here white Gaussian noise is added to the training and validation outputs,
with a standard deviation of 1 % or 5 % of each signal's rms. Test outputs stay clean, so the test error measures
the model against the truth. Systems, inputs, models (`pdelay`, `arx`) and fit protocols are a20's; data seeds 0-4.
The noise draw for seed s is seeded with 1000 + s.

Equation-error ARX is known to be biased under output noise (the regressors contain noisy past outputs). The
delay layer is an output-error fit (simulation of the model against the data), which is not.

**Predictions** (written before running).

1. `pdelay`: theta, K and every tau within 2 % at 1 % noise and within 5 % at 5 % noise, in every seed and both
   systems; at 1 % noise, the clean test error is <= 1.2x the oracle.
2. `arx` at 5 % noise: some tau is off by > 5 % in at least 4 of 5 seeds per system.

    python experiments/A/process/a21_noise.py      # results/a21_noise.md / .json
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
import a20_fopdt as P  # noqa: E402

LEVELS = (0.01, 0.05)
OUT = HERE / "results" / "a21_noise.json"


def noisy(data, level, seed):
    rng = np.random.default_rng(1000 + seed)
    out = dict(data)
    for s in ("train", "vali"):
        u, y = data[s]
        rms = y.pow(2).mean(dim=-1, keepdim=True).sqrt()
        out[s] = (u, y + level * rms * torch.tensor(rng.normal(size=y.shape)))
    return out


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = []
    for system, s in P.SYSTEMS.items():
        for seed in range(5):
            clean = P.make_data(system, seed)
            orc = P.PoleResidueDelay(len(s["taus"]), P.DT).to(device)
            p, r = P.pole_residue(s["K"], s["taus"])
            with torch.no_grad():
                orc.mu_re.copy_(torch.log(torch.expm1(torch.tensor(-p))))
                orc.beta_re.copy_(torch.tensor(r))
                orc.theta_raw.fill_(math.log(math.expm1(s["theta"])))
                floor = float(P.rel(orc(clean["test"][0].to(device)), clean["test"][1].to(device)))
            for level in LEVELS:
                data = noisy(clean, level, seed)
                for kind in ("pdelay", "arx"):
                    t0 = time.time()
                    if kind == "arx":
                        res = P.arx_fit(system, data)
                    else:
                        m = P.fit_layer(system, data, device, kind)
                        with torch.no_grad():
                            res = dict(test=float(P.rel(m(data["test"][0].to(device)), data["test"][1].to(device))))
                        res.update(P.layer_params(m))
                    err = dict(theta=abs(res["theta"] - s["theta"]) / s["theta"], K=abs(res["K"] - s["K"]) / s["K"],
                               taus=[abs(a - b) / b for a, b in zip(res["taus"], s["taus"])])
                    row = dict(system=system, seed=seed, level=level, kind=kind, ratio=res["test"] / floor, err=err, **res,
                               seconds=round(time.time() - t0, 1))
                    out.append(row)
                    print(system, seed, level, kind, f"ratio {row['ratio']:.2f}",
                          {k: (round(v, 4) if isinstance(v, float) else [round(x, 4) for x in v]) for k, v in err.items()}, flush=True)
                    OUT.write_text(json.dumps(out, indent=1))
    lines = ["# (A) Delay identification under output noise", "",
             "Noise σ = 1 % or 5 % of each signal's rms on training / validation outputs; clean test. Max relative error over "
             "seeds 0–4.", "", "| system | noise | model | test / oracle | θ error | K error | τ error (max) |",
             "|---|---|---|---|---|---|---|"]
    for system in P.SYSTEMS:
        for level in LEVELS:
            for kind in ("pdelay", "arx"):
                rs = [r for r in out if r["system"] == system and r["level"] == level and r["kind"] == kind]
                lines.append(f"| {system} | {level:.0%} | {kind} | {min(r['ratio'] for r in rs):.2f}–{max(r['ratio'] for r in rs):.2f} | "
                             f"{max(r['err']['theta'] for r in rs):.2%} | {max(r['err']['K'] for r in rs):.2%} | "
                             f"{max(max(r['err']['taus']) for r in rs):.2%} |")
    (OUT.with_suffix(".md")).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
