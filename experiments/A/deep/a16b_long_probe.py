"""(A) Cahn-Hilliard follow-up: can a longer probe window tell true growth from a stable cascade? (no training)

In a16 (`a16_cahn_hilliard.py`) the probe over T = 5 gave the same per-mode poles for the WPO with Re mu > 0 allowed
(`wpo`) and for the stable control (`wpo_stable`, every layer pole Re mu <= 0): k = 1..4 at +0.096, +0.335, +0.579,
+0.576, residuals ~0.002. A feed-forward cascade of stable layers is stable, so the control can only be emulating
growth over the finite window, e.g. by sums of t^j e^{mu t} terms with large cancelling residues. Over a longer
window that emulation must fail, while true unstable poles keep fitting.

**Test.** The a16 probe (single-mode inputs, odd part, eps = 0.003, per-mode identification) on a T = 10 window
(n_t = 401, same dt), on the a16 checkpoints and the solver. The models were trained on T = 5. The WPO runs on any
window; the TFNO's temporal modes rescale with it and are reported for completeness. The observability cut uses
T = 10.

**Predictions** (written before running).

1. Solver and `wpo`: per-mode Re mu for k = 1..4 within 5 % of lambda(k), and per-mode residual <= 0.05, in every
   seed.
2. `wpo_stable`: for at least one of k = 1..4 in every seed, Re mu is off by > 20 % or the residual is > 0.2, so
   over T = 10 it no longer looks like the growing modes.

    python experiments/A/deep/a16b_long_probe.py      # results/a16b_long_probe.md / .json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import a16_cahn_hilliard as C  # noqa: E402
import a9_burgers as M  # noqa: E402
import a2_twostage as B  # noqa: E402

OUT = HERE / "results" / "a16b_long_probe.json"


def probe_inputs_long():
    rng = np.random.default_rng(2)
    n = len(M.KS) * M.N_PER_K
    tf = np.linspace(0, M.T2, (M.NT2 - 1) * M.FINE + 1)
    fk = np.zeros((n, M.NX // 2 + 1, tf.size), complex)
    prof = M.A.profiles(tf, rng, n)
    ks = np.repeat(M.KS, M.N_PER_K)
    phase = np.exp(1j * rng.uniform(0, 2 * np.pi, size=n))
    fk[np.arange(n), ks] = (C.EPS / ks * M.NX / 2 * phase)[:, None] * prof
    return fk, tf[1] - tf[0]


def per_mode(G, fk, device):
    f_plus, g_plus = G(fk)
    _, g_minus = G(-fk)
    modes = B.per_mode({"train": (f_plus, 0.5 * (g_plus - g_minus))}, M.T2 / (M.NT2 - 1), device)
    return [dict(k=m["k"], re=m["mu"][0], im=m["mu"][1], resid=m["resid"],
                 rel_err=abs(complex(*m["mu"]) - C.ch.growth_rate(m["k"], C.MOB, C.KAPPA)) / abs(C.ch.growth_rate(m["k"], C.MOB, C.KAPPA) or 1))
            for m in modes[:8]]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fk, dtf = probe_inputs_long()
    t = np.linspace(0, M.T1, M.NT1)
    out = dict(solver=per_mode(lambda s: C.fields(s, dtf), fk, device), runs=[])
    print("solver", [(m["k"], round(m["re"], 3), round(m["resid"], 3)) for m in out["solver"][:5]], flush=True)
    for kind in C.VARIANTS:
        for seed in M.SEEDS:
            m = C.build(kind, seed, t, device)
            sd = torch.load(C.CKPT / f"{kind}_{seed}.pt", map_location=device, weights_only=False)
            sd.pop("_metadata", None)
            m.load_state_dict(sd, strict=False)
            m.eval()

            def G(spec):
                f = torch.tensor(np.fft.irfft(spec[..., ::M.FINE], n=M.NX, axis=1))
                return f, M.batched(m, f.to(device)).cpu().to(torch.float64)

            r = dict(model=kind, seed=seed, modes=per_mode(G, fk, device))
            out["runs"].append(r)
            print(kind, seed, [(x["k"], round(x["re"], 3), round(x["resid"], 3)) for x in r["modes"][:5]], flush=True)
    OUT.write_text(json.dumps(out, indent=1, default=float))
    lam = [C.ch.growth_rate(k, C.MOB, C.KAPPA) for k in range(1, 7)]
    lines = ["# (A) Cahn–Hilliard: per-mode probe over T = 10 (models trained on T = 5)", "",
             "Re μ̂ (residual) per mode; truth λ(k) = " + ", ".join(f"{x:.3f}" for x in lam) + " for k = 1..6.", "",
             "| model | seed | " + " | ".join(f"k = {k}" for k in range(1, 7)) + " |", "|---|---|" + "---|" * 6]
    rows = [("solver", "–", out["solver"])] + [(r["model"], r["seed"], r["modes"]) for r in out["runs"]]
    for name, seed, ms in rows:
        lines.append(f"| {name} | {seed} | " + " | ".join(f"{x['re']:.3f} ({x['resid']:.3f})" for x in ms[:6]) + " |")
    (OUT.with_suffix(".md")).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
