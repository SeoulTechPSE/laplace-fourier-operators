"""(A) Deep operator: does the Burgers data itself carry an effective viscosity? Best linear fits by amplitude.

a12 (`a12_probe_amp.py`) showed that the stable WPO's probe gives nu_hat = 0.111-0.114 at every probe amplitude
from 0.003 to 0.03: the bias is in the model's linearisation, not an amplitude extrapolation. Remaining candidate:
the best *linear* description of the data is itself more damped than the linearisation at u = 0, because in the
larger samples the nonlinear term moves energy out of the forced modes (an effective viscosity), and a model fitted
to all samples inherits it.

**Test.** Fit the single wavenumber pole layer (N = 1; the a7 joint fit: 8 random starts, k_ref 16, L-BFGS 300,
leaky Re cap, selection by validation) to the a9 Burgers data:
- all training samples;
- each quartile of the forcing amplitude (rms of f; train / validation split by the same edges).
nu_lin = -(k^2 coefficient). Linear test error per subset is reported, as a measure of how linear the subset is.

**Predictions** (written before running).

1. All samples: nu_lin in [0.105, 0.13].
2. nu_lin increases from the lowest to the highest amplitude quartile, and the lowest quartile gives nu_lin within
   3 % of 0.1.

    python experiments/A/deep/a12b_linear_fit.py      # results/a12b_linear_fit.md / .json
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "layer"))
import a9_burgers as M  # noqa: E402
from lfo.models.wavenumber import WavenumberPoleLayer1d  # noqa: E402

STARTS, SIGMA, ITERS = 8, 3.0, 300
OUT = HERE / "results" / "a12b_linear_fit.json"


def fit(xtr, ytr, xva, yva, t, device):
    best = None
    for s in range(STARTS):
        torch.manual_seed(s)
        m = WavenumberPoleLayer1d(1, t, modes=M.WPO_KW["modes"], re_max=0.5, re_leak=0.05, k_ref=16.0).to(device)
        with torch.no_grad():
            m.mu_re.normal_(0, SIGMA), m.mu_im.normal_(0, SIGMA)
        opt = torch.optim.LBFGS(m.parameters(), max_iter=ITERS, line_search_fn="strong_wolfe", tolerance_grad=1e-12,
                                tolerance_change=1e-15, history_size=50)

        def closure():
            opt.zero_grad()
            loss = M.A.rel(m(xtr), ytr)
            if torch.isfinite(loss):
                loss.backward()
            return loss

        try:
            opt.step(closure)
        except (RuntimeError, IndexError):
            pass
        with torch.no_grad():
            v = float(M.A.rel(m(xva), yva))
        if math.isfinite(v) and (best is None or v < best[0]):
            best = (v, m)
    return best


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = M.make_data()
    t = np.linspace(0, M.T1, M.NT1)
    (xtr, ytr), (xva, yva), (xte, yte) = (data[k] for k in ("train", "vali", "test"))
    rms = lambda x: x.flatten(1).pow(2).mean(1).sqrt().numpy()
    edges = np.quantile(rms(xtr), [0, 0.25, 0.5, 0.75, 1.0])
    edges[0], edges[-1] = 0, np.inf
    subsets = {"all": (np.ones(len(xtr), bool), np.ones(len(xva), bool), np.ones(len(xte), bool))}
    for q in range(4):
        sel = lambda x: (rms(x) >= edges[q]) & (rms(x) < edges[q + 1])
        subsets[f"Q{q + 1}"] = (sel(xtr), sel(xva), sel(xte))
    out = []
    for name, (a, b, c) in subsets.items():
        g = lambda x, i: x[torch.tensor(i)].to(device)
        v, m = fit(g(xtr, a), g(ytr, a), g(xva, b), g(yva, b), t, device)
        with torch.no_grad():
            test = float(M.A.rel(m(g(xte, c)), g(yte, c)))
            ar, ci = m.mu_re[0].cpu().numpy(), m.mu_im[0].cpu().numpy()
        r = dict(subset=name, n_train=int(a.sum()), vali=v, test=test, nu_lin=float(-ar[1] / 256),
                 k4=float(ar[2] / 16 ** 4), offset=float(-ar[0]), v=float(-ci[0] / 16))
        out.append(r)
        print(name, r, flush=True)
    OUT.write_text(json.dumps(dict(edges=edges[1:-1].tolist(), fits=out), indent=1, default=float))
    lines = ["# (A) Best linear fits to the Burgers data, by forcing amplitude", "",
             "Single wavenumber pole layer (N = 1), a7 joint fit. ν_lin = −(k² coefficient); truth at u = 0: 0.1.", "",
             "| subset | train samples | linear test error | ν_lin | k⁴ coefficient | offset (−a0) |", "|---|---|---|---|---|---|"]
    for r in out:
        lines.append(f"| {r['subset']} | {r['n_train']} | {r['test']:.3e} | {r['nu_lin']:.5f} | {r['k4']:.2e} | {r['offset']:.4f} |")
    (OUT.with_suffix(".md")).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
