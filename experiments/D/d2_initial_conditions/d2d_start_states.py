"""(D2) Why did d2c's `base` roll out better than d2b's? Developed start states against window count.

d2c (`d2c_rollout_training.py`) trained the IC-path WPO on 400 windows: both halves of 200 two-window trajectories,
so half of the windows start from a developed Burgers field (the state at t = 5). Its rollout error was 0.042,
against 0.054 for d2b's 200 windows that all start from a fresh random field. Two things changed at once. This
2 x 2 design separates them. All conditions use the same model, 6000 steps and batch 25, so the window count changes
data diversity, not the number of updates.

| condition | windows | start states |
|---|---|---|
| `fresh200` | 200 single windows | fresh random fields (as d2b) |
| `fresh400` | 400 single windows | fresh random fields |
| `dev200` | both windows of 100 two-window trajectories | half fresh, half developed |
| `dev400` | both windows of 200 two-window trajectories | half fresh, half developed (= d2c `base`, reused) |

Fresh windows: `d2b_burgers_rollout.sample` with rng 12. Two-window trajectories, test set (fresh windows, rng 11)
and rollout set (4 windows over [0, 20], rng 1): exactly d2c's. 3 seeds each. `dev400` is not retrained; its d2c
results are copied in (same data, model and training code).

**Predictions** (written before running).

1. `dev200` has a lower mean rollout error than `fresh400`: developed start states matter more than twice the data.
2. At equal count, developed start states lower the mean rollout error by at least 10 %: dev200 <= 0.9 x fresh200
   and dev400 <= 0.9 x fresh400.
3. On the fresh-window test set, `fresh400` has a mean window error at or below `dev400`'s.

    python experiments/D/d2_initial_conditions/d2d_start_states.py      # results/d2d_start_states.md / .json
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
import d2c_rollout_training as C  # noqa: E402

R, M = C.R, C.M
OUT = HERE / "results" / "d2d_start_states.json"


def single(d):
    w = M.NT1 - 1
    return (torch.cat([d[0][..., :w + 1], d[0][..., w:]]), torch.cat([d[1], d[2][..., w]]),
            torch.cat([d[2][..., :w + 1], d[2][..., w:]]))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base = C.make_data()
    fresh = R.sample(np.random.default_rng(12), 400, M.T1, M.NT1)
    traj = base["traj"]
    sets = {"fresh200": tuple(a[:200] for a in fresh), "fresh400": fresh,
            "dev200": single(tuple(a[:100] for a in traj))}
    t = np.linspace(0, M.T1, M.NT1)
    out = json.loads(OUT.read_text()) if OUT.exists() else []
    if not any(r["cond"] == "dev400" for r in out):
        for r in json.loads(C.OUT.read_text()):
            if r["mode"] == "base":
                out.append(dict(cond="dev400", seed=r["seed"], test=r["test"], rollout=r["rollout"], ratio=r["ratio"],
                                seconds=r["seconds"], source="d2c base"))
    for cond, single_set in sets.items():
        for seed in C.SEEDS:
            if any(r["cond"] == cond and r["seed"] == seed for r in out):
                continue
            t0 = time.time()
            m = R.build("wpo_ic", seed, t, device)
            C.train(m, dict(base, single=single_set), device, "base")
            m.eval()
            f, u0, y = base["test"]
            test = float(M.A.rel(R.predict(m, f, u0, device), y))
            fr, u0r, yr = base["roll"]
            roll = float(M.A.rel(R.rollout(m, fr, u0r, device), yr))
            r = dict(cond=cond, seed=seed, n=int(single_set[0].shape[0]), test=test, rollout=roll, ratio=roll / test,
                     seconds=round(time.time() - t0, 1))
            out.append(r)
            print(r, flush=True)
            OUT.parent.mkdir(exist_ok=True)
            OUT.write_text(json.dumps(out, indent=1))
            del m
            torch.cuda.empty_cache()
    lines = ["# (D2) Start states against window count (wpo_ic, forced Burgers, 3 seeds)", "",
             "| condition | window test (mean) | rollout over [0, 20] (mean; per seed) | rollout / test |", "|---|---|---|---|"]
    for cond in ("fresh200", "fresh400", "dev200", "dev400"):
        rs = sorted([r for r in out if r["cond"] == cond], key=lambda r: r["seed"])
        lines.append(f"| {cond} | {np.mean([r['test'] for r in rs]):.4f} | {np.mean([r['rollout'] for r in rs]):.4f} "
                     f"({', '.join(f'{r['rollout']:.4f}' for r in rs)}) | "
                     f"{min(r['ratio'] for r in rs):.2f}–{max(r['ratio'] for r in rs):.2f} |")
    (OUT.with_suffix(".md")).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
