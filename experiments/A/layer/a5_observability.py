"""(A) Pole layer, fifth validation: an observability cut in stage 2.

The beam validation (`a4_beam.py`) recovered c in 3 of 5 seeds. The two failures came from slow modes: for
k = 1..3 the poles turn by c k^2 T = 0.2, 0.8, 1.8 rad over the window. Over less than about one period (or one
e-fold), e^{mu t} is close to 1 + mu t, so mu and the residue trade off: wrong poles (or a slow pole plus a spurious
one with zero residue) fit the data with residuals as small as the good modes. Stage 2 weights modes by
1 / residual^2, so these wrong poles entered the polynomial with full weight and spoiled the initial point.

**Change** (one change; everything else as in a3 / a4): before the resolution cut, stage 2 keeps only modes whose
stage-1 poles are all observable over the window,

    max(|Re mu_hat|, |Im mu_hat|) * T >= 2      for every pole of the mode (T = 5),

computed from the stage-1 estimates only (no truth). The threshold 2 (two e-folds or two radians) is where the
linearization e^{mu t} ~ 1 + mu t is off by order 1 over the window. The fine-tune still sees all modes 0..16; only
the polynomial initialization changes.

**Stage 1 is reused**, not rerun: it is deterministic and independent of stage 2, so the per-mode poles are read
from `results/a3_twostage.json` and `results/a4_beam.json` (data regenerated from the same seeds). The comparison
with a3 / a4 therefore isolates the cut.

What the cut removes, from those stage-1 estimates (known before running):

- beam: k = 1, 2, 3 in every seed (min |mu_hat| T over the two poles: k = 3 gives 0.8-1.9, k = 4 gives 3.2);
- diff (lambda = -0.1 k^2): k = 1 (0.49); k = 2 sits on the threshold (1.99-2.01) and is kept or not by seed;
- adr: nothing (k = 1 gives 2.8).

Predictions (written before running):

1. Beam: tuned c within 1 % of 0.04 in at least 4 of 5 seeds (a4: 3 of 5), test error <= 1.2x the oracle floor and
   max |Re mu| / |Im mu| <= 0.01 over k = 3..16 in the same seeds.
2. diff and adr: no loss against a3. In all 10 runs the tuned test error is <= 1.2x the oracle floor, the
   dispersion error over k = 1..16 is <= 2 %, and the tuned D, v, k_r are within 1 % (k_r within 0.002 absolute).

    python experiments/A/layer/a5_observability.py      # results/a5_observability.md / .json
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
import a1_dispersion as A  # noqa: E402
import a2_twostage as B  # noqa: E402
import a3_twostage as C  # noqa: E402
import a4_beam as D  # noqa: E402

OBS = 2.0
OUT = HERE / "results" / "a5_observability.json"


def observable(poles, T=A.T_END):
    """True if every pole [re, im] of a mode has max(|Re|, |Im|) T >= OBS."""
    return min(max(abs(p[0]), abs(p[1])) for p in poles) * T >= OBS


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = []
    a3 = json.loads((HERE / "results" / "a3_twostage.json").read_text())
    floors = a3["floors"]
    for r3 in a3["runs"]:
        system, ds = r3["system"], r3["data_seed"]
        t0 = time.time()
        data, t = A.make_data(system, seed=ds)
        dt = float(t[1] - t[0])
        modes = [m for m in r3["modes"] if observable([m["mu"]])]
        s2 = C.stage2(modes, dt)
        test0, test, mu_ft, tuned = C.finetune(data, t, s2, device)
        true = A.lam(B.KS, **A.SYSTEMS[system])
        r = dict(system=system, data_seed=ds, kept=s2["kept"], stage2=s2, test_init=test0, test=test, oracle=floors[system],
                 tuned=tuned, disp_err=float(np.linalg.norm(mu_ft - true) / np.linalg.norm(true)),
                 a3=dict(kept=r3["stage2"]["kept"], test=r3["test"], disp_err=r3["disp_err"], tuned=r3["tuned"]),
                 seconds=round(time.time() - t0, 1))
        out.append(r)
        print(system, ds, f"kept {s2['kept'][0]}..{s2['kept'][-1]}", f"test {test0:.2e} -> {test:.2e}",
              f"disp {r['disp_err']:.1e}", tuned, flush=True)
    a4 = json.loads((HERE / "results" / "a4_beam.json").read_text())
    for r4 in a4:
        ds = r4["data_seed"]
        t0 = time.time()
        data, t = D.make_data(ds)
        dt = float(t[1] - t[0])
        kept = [m for m in r4["modes"] if observable(m["mu"])
                and max(abs(complex(*z)) for z in m["mu"]) * dt <= D.CUT]
        fit = D.fit_beam(data, t, kept, device)
        r = dict(system="beam", data_seed=ds, **fit, a4=dict(kept=r4["kept"], c=r4["c"], test=r4["test"],
                                                              re_ratio=r4["re_ratio"]),
                 seconds=round(time.time() - t0, 1))
        out.append(r)
        print("beam", ds, f"kept {fit['kept'][0]}..{fit['kept'][-1]}", f"c {fit['c_init']:.5f} -> {fit['c']:.5f}",
              f"re/im {fit['re_ratio']:.1e}", f"test {fit['test_init']:.2e} -> {fit['test']:.2e} ({fit['oracle']:.2e})",
              flush=True)
    OUT.write_text(json.dumps(out, indent=1, default=float))
    summarize(out)


def summarize(out):
    lines = ["# (A) Observability cut in stage 2", "",
             f"Stage 2 keeps modes with max(|Re μ̂|, |Im μ̂|) T ≥ {OBS:g} for every pole (stage-1 estimates, no truth), then "
             "the resolution cut. Stage 1 reused from a3 / a4. Data seeds 0–4.", "",
             "| system | seed | kept k (a3/a4 → now) | test: init → tuned (oracle) | a3/a4 tuned test | dispersion error | "
             "tuned constants (now; a3/a4) |", "|---|---|---|---|---|---|---|"]
    for r in out:
        prev = r.get("a3") or r.get("a4")
        rng = lambda ks: f"{ks[0]}–{ks[-1]}"
        if r["system"] == "beam":
            const = f"c {r['c']:.5f}, Re/Im {r['re_ratio']:.1e}; c {prev['c']:.5f}"
            disp = f"{r['disp_err']:.1e}"
        else:
            tu, tp = r["tuned"], prev["tuned"]
            const = (f"{tu['D']:.5f}, {tu['v']:.5f}, {tu['kr']:.5f}; {tp['D']:.5f}, {tp['v']:.5f}, {tp['kr']:.5f}")
            disp = f"{r['disp_err']:.1e} (a3 {prev['disp_err']:.1e})"
        lines.append(f"| {r['system']} | {r['data_seed']} | {rng(prev['kept'])} → {rng(r['kept'])} | "
                     f"{r['test_init']:.2e} → {r['test']:.2e} ({r['oracle']:.2e}) | {prev['test']:.2e} | {disp} | {const} |")
    (OUT.with_suffix(".md")).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
