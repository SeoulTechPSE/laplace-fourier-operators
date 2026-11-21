"""(A) Pole layer, eighth validation: removing the Re-cap trap of the joint fit.

In a7 (`a7_joint.py`) the joint fit from random starts recovered every system, but a fixed set of starts failed in
every data seed (1D diffusion / ADR: 5 of 16; 2D: 3 of 8). They end with Re mu at the soft cap 0.5 for every k,
where the soft minimum has almost no gradient. In the beam, a7's failures are mixed (inspected after a7, seed 0):
- cap-trapped starts (2, 5);
- a near-canceling pole pair with large opposite residues (0);
- an over-damped fit (1);
- a fit close to the truth except at high k (3).

Two remedies, each run on all starts of a7 (same draws, data seeds 0-4, systems as in a7):

- `leak`: `re_leak = 0.05`, a slope of 0.05 above the cap (layer option, 6dcc57d);
- `stable`: no leak; the Re coefficients start at -|draw|, so Re mu <= 0 at every k at the start (all powers of
  k^2 are non-negative). The Im and residue draws are unchanged.

Predictions (written before running):

1. diff, adr, diff2d, adr2d, with each remedy: in every data seed at least 15 of 16 (1D) or 7 of 8 (2D) starts
   succeed (a7: 11 of 16, 5 of 8).
2. beam: with each remedy, the per-seed success count rises over a7 by 1 to 4 starts. The cap-trapped starts
   recover; the other failure types remain.
3. With each remedy, the start selected by validation error is at <= 1.2x the oracle in all 25 data seeds.

The first launch stopped at beam seed 3 (`leak`): torch's strong-Wolfe line search raised an IndexError on a
non-finite loss. `a7_joint.fit` now treats it like a non-finite loss (the start keeps its parameters at that point
and is evaluated); the run resumes from the finished entries.

    python experiments/A/layer/a8_captrap.py      # results/a8_captrap.md / .json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import a7_joint as J  # noqa: E402

VARIANTS = {"leak": dict(leak=0.05), "stable": dict(stable_start=True)}
OUT = HERE / "results" / "a8_captrap.json"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    a7 = {(r["system"], r["data_seed"]): r for r in json.loads((HERE / "results" / "a7_joint.json").read_text())}
    out = json.loads(OUT.read_text()) if OUT.exists() else []      # resume: keep finished (system, seed, variant)
    done = {(r["system"], r["data_seed"], r["variant"]) for r in out}
    for system in J.STARTS:
        for ds in range(5):
            if all((system, ds, v) in done for v in VARIANTS):
                continue
            data, t = J.data_for(system, ds)
            floor = a7[(system, ds)]["oracle"]
            for name, kw in VARIANTS.items():
                if (system, ds, name) in done:
                    continue
                t0 = time.time()
                starts = [J.fit(system, data, t, s, device, **kw) for s in range(J.STARTS[system])]
                for s in starts:
                    s["ratio"] = s["test"] / floor
                best = min(starts, key=lambda s: s["vali"])
                r = dict(system=system, data_seed=ds, variant=name, oracle=floor, starts=starts, best=best,
                         n_success=sum(s["ratio"] <= 1.2 for s in starts), a7_success=a7[(system, ds)]["n_success"],
                         failed=[s["start"] for s in starts if not s["ratio"] <= 1.2], seconds=round(time.time() - t0, 1))
                out.append(r)
                print(system, ds, name, f"success {r['n_success']}/{len(starts)} (a7 {r['a7_success']})",
                      f"failed {r['failed']}", f"best ratio {best['ratio']:.3f}", best["const"], f"{r['seconds']} s",
                      flush=True)
                OUT.write_text(json.dumps(out, indent=1, default=float))
            del data
            torch.cuda.empty_cache()
    summarize(out)


def summarize(out):
    lines = ["# (A) Removing the Re-cap trap of the joint fit", "",
             "a7 protocol (random starts, k_ref, L-BFGS 300, selection by validation). `leak`: re_leak = 0.05. `stable`: "
             "Re coefficients start at −|draw|. Success: test ≤ 1.2× oracle.", "",
             "| system | seed | variant | starts succeeded (a7) | failed starts | selected: test / oracle | selected: constants |",
             "|---|---|---|---|---|---|---|"]
    fmt = lambda c: ("–" if c is None else ", ".join(
        f"{k} {v:.5f}" if not isinstance(v, list) else f"{k} ({v[0]:.4f}, {v[1]:.4f})" for k, v in c.items()))
    for r in out:
        lines.append(f"| {r['system']} | {r['data_seed']} | {r['variant']} | {r['n_success']} / {len(r['starts'])} "
                     f"({r['a7_success']}) | {', '.join(map(str, r['failed'])) or '–'} | {r['best']['ratio']:.3f} | "
                     f"{fmt(r['best']['const'])} |")
    (OUT.with_suffix(".md")).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
