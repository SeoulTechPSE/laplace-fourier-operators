"""(A) Deep operator: zero-shot resolution transfer on forced Burgers (no training).

The a11 WPO and the a13 TFNO / FNO were trained at n_x = 64, dt = 0.025 (T = 5, n_t = 201). They are evaluated
unchanged on new test samples (100, the a9 distribution, new random stream) at other resolutions:

- space: n_x in {32, 64, 128};
- time: n_t = 401 (dt / 2) at n_x = 64.

Truth: the Burgers solver on a 128-point grid (2/3 dealiasing, k <= 42), 32 substeps per coarse step of the
dt = 0.025 grid, sampled to each target grid. The forcing band (k <= 8) is representable on all grids. The WPO's
time grid is set to the new t (its layers only use dt). The FNO-type models run on the new grid as they are.

**Predictions** (written before running).

1. WPO: error <= 1.2x its error at the training resolution (n_x = 64 on the same new samples) at n_x = 128 and at
   dt / 2, in every seed.
2. TFNO: the same (<= 1.2x at n_x = 128 and at dt / 2), in every seed.
3. At n_x = 32, both WPO and TFNO exceed 1.2x in every seed: the pointwise nonlinearities then act on a grid that
   aliases the harmonics they create.

    python experiments/A/deep/a15_resolution.py      # results/a15_resolution.md / .json
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
import a11_fair as F  # noqa: E402
import a13_baselines as B13  # noqa: E402
from lfo.data import burgers  # noqa: E402

N_FINE, N_TEST = 128, 100
OUT = HERE / "results" / "a15_resolution.json"


def test_sets():
    rng = np.random.default_rng(3)
    nx0 = M.NX
    M.NX = N_FINE                                         # forcing on the 128 grid (same band, amplitudes per N)
    fk, dtf = M.forcing(rng, N_TEST, M.T1, M.NT1)
    M.NX = nx0
    out = {}
    for stride, nt_name in ((M.FINE, "dt"), (M.FINE // 2, "dt/2")):
        uk = burgers.solve(fk, dtf, M.NU, N_FINE, stride=stride)
        fs = fk[..., ::stride]
        for nx in ((32, 64, 128) if nt_name == "dt" else (64,)):
            f = np.fft.irfft(fs, n=N_FINE, axis=1)[:, :: N_FINE // nx]
            u = np.fft.irfft(uk, n=N_FINE, axis=1)[:, :: N_FINE // nx]
            out[(nx, nt_name)] = (torch.tensor(f), torch.tensor(u))
    return out


def set_time(model, n_t):
    t = torch.linspace(0, M.T1, n_t, dtype=torch.float64, device=next(model.parameters()).device)
    for K in getattr(model, "K", []):
        K.t = t


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sets = test_sets()
    t = np.linspace(0, M.T1, M.NT1)
    out = []
    for kind in ("wpo", "tfno", "fno_big"):
        for seed in M.SEEDS:
            if kind == "wpo":
                m = F.build("wpo", seed, t, device)
                m.load_state_dict(torch.load(F.CKPT / f"wpo_{seed}.pt", map_location=device))
            else:
                m = B13.build(kind, seed, device)
                sd = torch.load(B13.CKPT / f"{kind}_{seed}.pt", map_location=device, weights_only=False)
                sd.pop("_metadata", None)
                m.load_state_dict(sd, strict=False)
            m.eval()
            r = dict(model=kind, seed=seed)
            for (nx, nt), (x, y) in sets.items():
                set_time(m, x.shape[-1])
                p = M.batched(m, x.to(device)).cpu().to(torch.float64)
                r[f"{nx}_{nt}"] = float(M.A.rel(p, y))
            set_time(m, M.NT1)
            out.append(r)
            print(r, flush=True)
    OUT.write_text(json.dumps(out, indent=1))
    keys = [k for k in out[0] if k not in ("model", "seed")]
    lines = ["# (A) Zero-shot resolution transfer on forced Burgers (trained at n_x = 64, dt = 0.025)", "",
             "Relative L2 on 100 new samples; truth from a 128-point solve. Ratio to the training resolution in brackets.", "",
             "| model | seed | " + " | ".join(k.replace("_", ", ") for k in keys) + " |", "|---|---|" + "---|" * len(keys)]
    for r in out:
        base = r["64_dt"]
        lines.append(f"| {r['model']} | {r['seed']} | " + " | ".join(f"{r[k]:.3e} ({r[k] / base:.2f})" for k in keys) + " |")
    (OUT.with_suffix(".md")).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
