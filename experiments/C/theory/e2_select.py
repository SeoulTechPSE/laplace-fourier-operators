"""(C) E2: best-of-k selection by validation loss, checked on independent runs.

The E2 remedy analysis estimated by bootstrap (10 seeds) that picking the lowest final
validation loss among 5 independent ``mixed_off_ms`` runs recovers all p4 poles in ~95 %
of cases. Here seeds 10-59 (new; multi-start candidates 100 * seed + c, also new) are
split into 10 disjoint groups of 5 consecutive seeds; per group the run with the lowest
validation loss is selected and scored. Seeds 0-9 (used for the estimate) are excluded.

    python experiments/C/theory/e2_select.py      # writes results/e2_select.md (+ .json)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
RUNS = OUT / "e2_train"
VARIANT, SEEDS, K = "mixed_off_ms", range(10, 60), 5
RECOVERED = 0.1


def load(cond):
    rs = {}
    for s in SEEDS:
        f = RUNS / f"{cond}_{VARIANT}_s{s}.json"
        if f.exists():
            rs[s] = json.loads(f.read_text())
    return rs


def analyse(cond):
    rs = load(cond)
    ok = lambda r: max(r["pole_error"]) < RECOVERED
    val = lambda r: r["history"][-1][2]
    groups = []
    for g0 in range(SEEDS.start, SEEDS.stop, K):
        g = [rs[s] for s in range(g0, g0 + K) if s in rs]
        if len(g) < K:
            continue
        best = min(g, key=val)
        groups.append(dict(seeds=[r["seed"] for r in g], selected=best["seed"], selected_ok=ok(best),
                           selected_test=best["test_rel_l2"], any_ok=any(ok(r) for r in g),
                           n_ok=sum(ok(r) for r in g)))
    single = [ok(r) for r in rs.values()]
    per_pole = np.array([r["pole_error"] for r in rs.values()])
    sel_pe = np.array([rs[g["selected"]]["pole_error"] for g in groups])
    return dict(cond=cond, runs=len(rs), single_rate=float(np.mean(single)),
                per_pole_single=(per_pole < RECOVERED).mean(0).tolist(),
                groups=groups, selected_rate=float(np.mean([g["selected_ok"] for g in groups])),
                per_pole_selected=(sel_pe < RECOVERED).mean(0).tolist(),
                any_rate=float(np.mean([g["any_ok"] for g in groups])),
                true_poles=next(iter(rs.values()))["true_poles"])


def main():
    out = [analyse(c) for c in ("p4_ms16_T20", "p6_ms16_T80")]
    (OUT / "e2_select.json").write_text(json.dumps(out, indent=1))
    lab = lambda z: f"{z[0]:+g}{z[1]:+g}i" if z[1] else f"{z[0]:+g}"
    lines = ["# E2: best-of-5 selection by validation loss, independent runs", "",
             f"`{VARIANT}`, seeds 10-59 (not used for the bootstrap estimate), 10 disjoint groups of 5. "
             "'all recovered' = every true pole within relative error 0.1.", "",
             "| condition | runs | single run: all recovered | best of 5 by validation: all recovered | "
             "groups with ≥ 1 good run | per pole, single → selected |",
             "|---|---|---|---|---|---|"]
    for r in out:
        pp = ", ".join(f"{lab(z)} {a:.0%} → {b:.0%}" for z, a, b in
                       zip(r["true_poles"], r["per_pole_single"], r["per_pole_selected"]))
        lines.append(f"| {r['cond']} | {r['runs']} | {r['single_rate']:.0%} | "
                     f"{r['selected_rate']:.0%} ({sum(g['selected_ok'] for g in r['groups'])}/{len(r['groups'])}) | "
                     f"{r['any_rate']:.0%} | {pp} |")
    lines += ["", "Per group (seeds, good runs, selected seed, selected ok, selected test rel. L2):", ""]
    for r in out:
        lines.append(f"- {r['cond']}: " + "; ".join(
            f"{g['seeds'][0]}-{g['seeds'][-1]}: {g['n_ok']}/5, s{g['selected']} "
            f"{'ok' if g['selected_ok'] else 'miss'} {g['selected_test']:.1e}" for g in r["groups"]))
    (OUT / "e2_select.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
