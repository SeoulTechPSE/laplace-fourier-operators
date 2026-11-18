"""(D1) Second pass: trained single pole-residue layers with the Fourier and the FOH time axis.

The first pass (`d1_oracle.py`) measured the error each representation adds with the true poles. Here a single tied
pole-residue layer is trained with each time axis, on data whose truth is computed on a 32x finer grid (inputs
defined there; the layer sees the n = 1024 samples), so neither path is exact by construction.

- `fourier`  the E1 tied layer (`e1_audit.Layer1("tied")`, textbook LNO v1 layer: response to the trigonometric
             interpolant);
- `foh`      the same layer with the exact sampled first-order-hold response (`e2_train.FOHPoleResidue`).

Systems p2, p4, p6 (E1), E1 input families (step, ramp, PRBS, smooth; labels kept for a per-family test error),
splits 200 / 50 / 100, E1 `stable` init, `common` protocol with pole lr x 10 (the E1 setting in which the complex
poles are found), 10 seeds, number of slots = number of true poles. 60 runs.

Predictions (written before running):

1. On the ramp and smooth test samples, the median (over seeds) test error of `foh` is at most half that of
   `fourier`, in all three systems. On step and PRBS samples the two differ by less than 30 % (sampling-limited).
2. Complex-pole recovery (error < 0.1) differs by at most 1 seed in 10 per pole between the paths.
3. Real poles (p4 -1, -0.2; p6 -1, -2): `foh` recovers each at least as often as `fourier`.

    python experiments/D/d1_inputs/d1_train.py --system p2 p4 p6 --seeds 0 ... 9   # results/d1_train/*.json
    python experiments/D/d1_inputs/d1_train.py --summarize                          # results/d1_train.md
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "C" / "e1_audit"))
sys.path.insert(0, str(ROOT / "experiments" / "C" / "theory"))
import e1_audit as E  # noqa: E402
from e2_train import FOHPoleResidue  # noqa: E402
from lfo.data import lti  # noqa: E402
from lfo.metrics import pole_error  # noqa: E402
from lfo.training import seed_everything  # noqa: E402

FINE, SPLIT = 32, (200, 50, 100)
FAMS = ("step", "ramp", "prbs", "smooth")
PATHS = ("fourier", "foh")
OUT = HERE / "results" / "d1_train"


def make_data(system, data_seed=0):
    """E1 input families on a 32x finer grid, truth there, everything subsampled to the E1 grid."""
    mu, beta = (np.array(a, dtype=complex) for a in E.SYSTEMS[system])
    t = E.time_grid()
    tf = np.arange(t.size * FINE) * (t[1] - t[0]) / FINE
    rng = np.random.default_rng(data_seed)
    n = sum(SPLIT)
    fams = [
        lambda: lti.step(tf, rng.uniform(0, E.T_END / 2), rng.uniform(0.5, 2.0) * rng.choice([-1, 1])),
        lambda: lti.ramp(tf, rng.uniform(0, E.T_END / 2), rng.uniform(0.05, 0.3) * rng.choice([-1, 1])),
        lambda: lti.prbs(tf, rng, hold=rng.uniform(0.3, 2.0)),
        lambda: lti.smooth_random(tf, rng),
    ]
    lab = np.arange(n) % 4
    uf = np.stack([fams[k]() for k in lab])
    perm = rng.permutation(n)
    uf, lab = uf[perm], lab[perm]
    yf = lti.response(tf, uf, mu, beta)
    u, y = uf[:, ::FINE], yf[:, ::FINE]
    f = lambda a: torch.tensor(a, dtype=torch.float32)
    s = np.cumsum([0, *SPLIT])
    data = {k: (f(u[a:b]), f(y[a:b])) for k, a, b in zip(("train", "vali", "test"), s[:-1], s[1:])}
    return data, lab[s[2]:s[3]], t, mu, beta


def run(system, path, seed, device):
    out = OUT / f"{system}_{path}_s{seed}.json"
    if out.exists():
        return
    data, lab, t, mu_true, beta_true = make_data(system)
    seed_everything(seed)
    model = E.Layer1("tied", mu_true.size, t, "stable")
    if path == "foh":
        model.pr.__class__ = FOHPoleResidue
    model.to(device)
    t0 = time.time()
    hist = E.train(model, data, E.PROTOCOLS["common"], device, pole_lr_mult=10.0)
    xte, yte = (a.to(device)[..., None] for a in data["test"])
    with torch.no_grad():
        pred = model(xte)
        rel = ((pred - yte).flatten(1).norm(dim=1) / yte.flatten(1).norm(dim=1)).cpu().numpy()
    mu_l, beta_l = (a.cpu().numpy() for a in model.pole_sets())
    mf, _ = E.fold(mu_l, beta_l)
    mt, _ = E.true_folded(mu_true, beta_true)
    res = dict(system=system, path=path, seed=seed, seconds=round(time.time() - t0, 1), history=hist,
               test=float(rel.mean()), test_family={f: float(np.median(rel[lab == k])) for k, f in enumerate(FAMS)},
               true_poles=[[float(z.real), float(z.imag)] for z in mt],
               pole_error=pole_error(mf, mt, reduce="none").tolist())
    OUT.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    print(system, path, seed, f"test {res['test']:.2e}", {k: f"{v:.1e}" for k, v in res["test_family"].items()},
          "pole err", [round(e, 3) for e in res["pole_error"]], flush=True)


def summarize():
    rows = [json.loads(f.read_text()) for f in sorted(OUT.glob("*.json"))]
    lines = ["# D1 second pass: trained tied layer, Fourier vs FOH time axis", "",
             "Truth on a 32× finer grid; E1 families; `stable` init, `common` + pole lr × 10; slots = true poles. "
             "Test error: median over seeds of the per-family median relative L2. Recovery: seeds with pole error "
             "< 0.1.", "",
             "| system | path | seeds | test (all) | step | ramp | PRBS | smooth | recovered per true pole |",
             "|---|---|---|---|---|---|---|---|---|"]
    for system in E.SYSTEMS:
        for path in PATHS:
            rs = [r for r in rows if r["system"] == system and r["path"] == path]
            if not rs:
                continue
            med = lambda f: np.median([r["test_family"][f] for r in rs])
            tp = rs[0]["true_poles"]
            rec = [sum(r["pole_error"][i] < 0.1 for r in rs) for i in range(len(tp))]
            lab = ", ".join(f"{a:+g}{b:+g}i: {k}" if b else f"{a:+g}: {k}" for (a, b), k in zip(tp, rec))
            lines.append(f"| {system} | {path} | {len(rs)} | {np.median([r['test'] for r in rs]):.1e} | "
                         + " | ".join(f"{med(f):.1e}" for f in FAMS) + f" | {lab} |")
    (HERE / "results" / "d1_train.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", nargs="+", default=list(E.SYSTEMS))
    ap.add_argument("--path", nargs="+", default=list(PATHS))
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    ap.add_argument("--summarize", action="store_true")
    a = ap.parse_args()
    if a.summarize:
        summarize()
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for s in a.seeds:
        for system in a.system:
            for path in a.path:
                run(system, path, s, device)


if __name__ == "__main__":
    main()
