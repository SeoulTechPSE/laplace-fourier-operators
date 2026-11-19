"""(D2) Training for rollout: pushforward and the semigroup regulariser (plan D, Section 5) on forced Burgers.

d2b (`d2b_burgers_rollout.py`) found that every model roughly doubles its error over 4 chained windows. Even the
shared-pole initial-state path (`wpo_ic`, lowest rollout error) does, because in a nonlinear system each window
starts from a predicted field the model was never trained on. Two remedies from the plan, on the d2b `wpo_ic`
model:

- `push` (pushforward): training samples are two consecutive windows of one trajectory. The first window starts
  from the true field. The second starts from the model's own prediction at t = 5 (detached, no gradient through
  it), so the model learns to start from its own outputs. Loss: mean of the two windows' relative L2.
- `semi` (semigroup regulariser): on a single window, the prediction over [0, T] should equal chaining two
  half-windows, G(f[T/2:], G(f[:T/2], u0)[end]). The penalty ||G_full[T/2:] - G_chained[T/2:]|| / ||G_full[T/2:]|| is
  added with weight 1.

`base` is trained on single windows (as d2b). All variants use the same trajectories: 200 training trajectories of
two windows ([0, 10], n_t = 401). `base` and `semi` use both windows as independent samples starting from their true
fields. Validation: 50 trajectories; rollout set: 50 trajectories on [0, 20] (4 windows). The a9 forcing and the
d2b initial fields; nu = 0.1. 3 seeds, 6000 steps, a11 settings at width 23.

**Predictions** (written before running).

1. `push` and `semi` each have a lower mean rollout error (4 windows) than `base`.
2. `push` has a rollout / window-test ratio <= 1.7 in every seed (`base` about 2.1 in d2b).
3. Neither remedy raises the mean single-window test error by more than 10 % over `base`.

    python experiments/D/d2_initial_conditions/d2c_rollout_training.py      # results/d2c_rollout_training.md / .json
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
import d2b_burgers_rollout as R  # noqa: E402

M = R.M
STEPS, SEEDS, BATCH = 6000, (0, 1, 2), 25
OUT = HERE / "results" / "d2c_rollout_training.json"


def make_data():
    rng = np.random.default_rng(10)
    f, u0, u = R.sample(rng, 250, 2 * M.T1, 2 * (M.NT1 - 1) + 1)           # two-window trajectories
    w = M.NT1 - 1
    tr = (f[:200], u0[:200], u[:200])
    va = (f[200:], u0[200:], u[200:])
    single = lambda d: (torch.cat([d[0][..., :w + 1], d[0][..., w:]]), torch.cat([d[1], d[2][..., w]]),
                        torch.cat([d[2][..., :w + 1], d[2][..., w:]]))
    test = R.sample(np.random.default_rng(11), 100, M.T1, M.NT1)
    return dict(traj=tr, single=single(tr), vali=single(va), test=test,
                roll=R.sample(np.random.default_rng(1), 50, R.N_WIN * M.T1, R.N_WIN * (M.NT1 - 1) + 1))


def train(m, data, device, mode):
    is_pole = lambda n: n.endswith("mu_re") or n.endswith("mu_im")
    opt = torch.optim.Adam([dict(params=[p for n, p in m.named_parameters() if not is_pole(n)], lr=3e-3),
                            dict(params=[p for n, p in m.named_parameters() if is_pole(n)], lr=9e-2)])
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS)
    g = torch.Generator().manual_seed(0)
    w, h = M.NT1 - 1, (M.NT1 - 1) // 2
    for _ in range(STEPS):
        opt.zero_grad()
        if mode == "push":
            f, u0, u = data["traj"]
            idx = torch.randint(0, f.shape[0], (BATCH,), generator=g)
            f, u0, u = f[idx].to(device), u0[idx].to(device), u[idx].to(device)
            p1 = m(f[..., :w + 1], u0)
            p2 = m(f[..., w:], p1[..., -1].detach())
            loss = 0.5 * (M.A.rel(p1, u[..., :w + 1]) + M.A.rel(p2, u[..., w:]))
        else:
            f, u0, u = data["single"]
            idx = torch.randint(0, f.shape[0], (BATCH,), generator=g)
            f, u0, u = f[idx].to(device), u0[idx].to(device), u[idx].to(device)
            p = m(f, u0)
            loss = M.A.rel(p, u)
            if mode == "semi":
                a = m(f[..., :h + 1], u0)
                b = m(f[..., h:], a[..., -1])
                loss = loss + M.A.rel(b, p[..., h:])
        loss.backward()
        opt.step()
        sch.step()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = make_data()
    t = np.linspace(0, M.T1, M.NT1)
    out = json.loads(OUT.read_text()) if OUT.exists() else []
    for mode in ("base", "push", "semi"):
        for seed in SEEDS:
            if any(r["mode"] == mode and r["seed"] == seed for r in out):
                continue
            t0 = time.time()
            m = R.build("wpo_ic", seed, t, device)
            train(m, data, device, mode)
            m.eval()
            f, u0, y = data["test"]
            test = float(M.A.rel(R.predict(m, f, u0, device), y))
            fr, u0r, yr = data["roll"]
            roll = float(M.A.rel(R.rollout(m, fr, u0r, device), yr))
            r = dict(mode=mode, seed=seed, test=test, rollout=roll, ratio=roll / test, seconds=round(time.time() - t0, 1))
            out.append(r)
            print(r, flush=True)
            OUT.parent.mkdir(exist_ok=True)
            OUT.write_text(json.dumps(out, indent=1))
            del m
            torch.cuda.empty_cache()
    lines = ["# (D2) Training for rollout on forced Burgers: pushforward and semigroup regulariser (wpo_ic)", "",
             "| training | window test (mean) | rollout over [0, 20] (mean) | rollout / test (range) |", "|---|---|---|---|"]
    for mode in ("base", "push", "semi"):
        rs = [r for r in out if r["mode"] == mode]
        lines.append(f"| {mode} | {np.mean([r['test'] for r in rs]):.4f} | {np.mean([r['rollout'] for r in rs]):.4f} | "
                     f"{min(r['ratio'] for r in rs):.2f}–{max(r['ratio'] for r in rs):.2f} |")
    (OUT.with_suffix(".md")).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
