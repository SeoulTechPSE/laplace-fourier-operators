"""(D2) Cahn–Hilliard with initial fields: is the one-window gap in the initial-field or in the forcing response?

d2i: the TFNO's rollout edge over the IC-path WPO on forced 1D Cahn–Hilliard is its one-window edge (local error
0.56x in every window, in the same bands). In project A (a16), on the same system with zero initial states, the two
were at parity. So the gap may live in the response to the initial field, the part the IC path adds. This script
splits the one-window error by which input dominates.

**Sets** (100 windows each, fresh seeds; forcing and initial-field generators of d2g with their amplitude ranges
narrowed; both families of d2g's training set span s, s0 in [0.03, 1], log-uniform):
- `id`: d2g's in-distribution set (for reference);
- `force_dom`: forcing s in [0.3, 1], initial field s0 in [0.03, 0.04] (in the training range, forcing-dominated);
- `ic_dom`: forcing s in [0.03, 0.04], initial field s0 in [0.3, 1] (in range, initial-field-dominated);
- `force_only`: initial field exactly 0, forcing as `id` (just outside the training range);
- `ic_only`: forcing exactly 0, initial field as `id` (just outside).

**Checks before any fit.** All sets finite. max |u| (window): `id` 3.2, `force_dom` 2.1, `ic_dom` 3.0,
`force_only` 1.6, `ic_only` 3.2; max |f|: 0.83, 2.0, 0.07, 1.3, 0.

**Setup.** d2g as it stands: 6000 steps, `wpo_ic` and `tfno_chan`, 3 seeds. Metric: d2g's relative L2 over the
window.

**Predictions** (written before running).

1. `force_dom`: TFNO / WPO error ratio (means over seeds) >= 0.8 (near a16's parity).
2. `ic_dom`: ratio <= 0.6 (the gap is in the initial-field response).
3. The exact-path sets order the same way: ratio(`force_only`) > ratio(`ic_only`).

    python experiments/D/d2_initial_conditions/d2j_ch_paths.py      # results/d2j_ch_paths.md / .json
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

R, E, M = G.R, G.E, G.M
KINDS, SEEDS = ("wpo_ic", "tfno_chan"), (0, 1, 2)
SETS = ("id", "force_dom", "ic_dom", "force_only", "ic_only")
OUT = HERE / "results" / "d2j_ch_paths.json"


def sample(rng, n, s_force, s_ic, zero_force=False, zero_ic=False):
    old = M.S_RANGE
    M.S_RANGE = s_force
    try:
        fk, dtf = G.forcing(rng, n, M.T1, M.NT1)
    finally:
        M.S_RANGE = old
    u0k = E.initial(rng, n, (1, 8), s_ic)
    return G.solve(fk * 0 if zero_force else fk, dtf, u0k * 0 if zero_ic else u0k)


def make_sets():
    base = G.make_sets()
    full = (0.03, 1.0)
    return dict(train=base["train"], id=base["id"],
                force_dom=sample(np.random.default_rng(31), 100, (0.3, 1.0), (0.03, 0.04)),
                ic_dom=sample(np.random.default_rng(32), 100, (0.03, 0.04), (0.3, 1.0)),
                force_only=sample(np.random.default_rng(33), 100, full, full, zero_ic=True),
                ic_only=sample(np.random.default_rng(34), 100, full, full, zero_force=True))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sets = make_sets()
    for s in SETS:
        print(s, "max |u|", round(float(sets[s][2].abs().max()), 3), "mean |y|",
              round(float(sets[s][2].flatten(1).norm(dim=1).mean()), 2), flush=True)
    t = np.linspace(0, M.T1, M.NT1)
    out = json.loads(OUT.read_text()) if OUT.exists() else []
    for kind in KINDS:
        for seed in SEEDS:
            if any(r["model"] == kind and r["seed"] == seed for r in out):
                continue
            t0 = time.time()
            m = G.build(kind, seed, t, device)
            R.train(m, dict(train=sets["train"]), device, 30.0 if kind.startswith("wpo") else 1.0)
            m.eval()
            r = dict(model=kind, seed=seed)
            for s in SETS:
                f, u0, y = sets[s]
                r[s] = float(M.A.rel(R.predict(m, f, u0, device), y))
            r["seconds"] = round(time.time() - t0, 1)
            out.append(r)
            print(r, flush=True)
            OUT.parent.mkdir(exist_ok=True)
            OUT.write_text(json.dumps(out, indent=1))
            del m
            torch.cuda.empty_cache()
    mean = {k: {s: np.mean([r[s] for r in out if r["model"] == k]) for s in SETS} for k in KINDS}
    lines = ["# (D2) Forced Cahn–Hilliard: one-window error by dominant input (d2g setup, 6000 steps)", "",
             "| set | " + " | ".join(KINDS) + " | TFNO / WPO |", "|---|---|---|---|"]
    for s in SETS:
        ranges = " | ".join(f"{mean[k][s]:.4f} ({min(r[s] for r in out if r['model'] == k):.4f}–"
                            f"{max(r[s] for r in out if r['model'] == k):.4f})" for k in KINDS)
        lines.append(f"| {s} | {ranges} | {mean['tfno_chan'][s] / mean['wpo_ic'][s]:.2f} |")
    (OUT.with_suffix(".md")).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
