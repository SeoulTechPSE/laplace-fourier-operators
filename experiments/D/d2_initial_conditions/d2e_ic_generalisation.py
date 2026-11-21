"""(D2) Initial fields outside the training range, and rollout over 10 windows (RQ2 and the plan's 10x metric).

d2b compared three ways for the initial field to enter (IC path sharing the poles, input channel in the WPO,
input channel in a TFNO) on windows with fresh start states. d2c / d2d showed that training on windows of which half
start from developed fields lowers the rollout error by ~30 %. Here all three models are trained that way (d2c
`base` data: both windows of 200 two-window trajectories; 6000 steps, batch 25, 3 seeds; d2b architectures and
learning rates), and evaluated on what the plan asks for and the earlier passes did not measure.

**Evaluation sets** (forcing from the training family throughout; 100 windows each unless stated).
- `id`: fresh windows as in training (d2c's test set, rng 11).
- `amp`: initial-field scale s0 log-uniform in [1, 3] (training: [0.03, 1]), modes 1..8 (rng 21).
- `high`: initial field on modes 9..16 only (training: 1..8), amplitude ~ s0 / k with s0 in [0.03, 1] (rng 22).
- `roll4`: d2c's rollout set (4 windows over [0, 20], one forcing profile over [0, 20]; rng 1).
- `roll10`: 10 windows over [0, 50] on 50 trajectories (rng 23). The forcing is a concatenation of independent
  window-length draws, so each window's forcing is from the training family. (A single profile over [0, 50] would
  let ramps grow 10x beyond the training range.) Reported: error over [0, 50], over the first 4 windows, and per
  window.

Generalization is reported as the ratio of the set's error to the model's own `id` error.

**Predictions** (written before running).

1. With developed-start training, `wpo_ic` still has the lowest mean `roll4` error, at most 0.9x `tfno_chan`'s
   (d2b, fresh starts: 0.89x).
2. `high`: the `wpo_ic` ratio high / id is at most half of `tfno_chan`'s and of `wpo_chan`'s. Reason: the IC path
   propagates each mode of the initial field with the learned pole law, while for the channel models a field on
   unseen modes is a new constant input.
3. `amp`: every model's error rises at least 2x (ratio amp / id >= 2 in every seed), and `wpo_ic`'s ratio is not
   below `wpo_chan`'s by more than 10 %: the amplitude dependence is nonlinear and comes from the later layers,
   which the two WPOs share.
4. `roll10`: for `wpo_ic` in every seed, the error over 10 windows is at most 1.5x its error over the first 4 windows
   of the same trajectories (Burgers is dissipative; per-window errors should saturate, not grow).

    python experiments/D/d2_initial_conditions/d2e_ic_generalisation.py      # results/d2e_ic_generalisation.md / .json
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
import d2c_rollout_training as C  # noqa: E402

M = R.M
burgers = R.burgers
KINDS, SEEDS, N10 = ("wpo_ic", "wpo_chan", "tfno_chan"), (0, 1, 2), 10
OUT = HERE / "results" / "d2e_ic_generalisation.json"


def initial(rng, n, modes, s_range):
    K = M.NX // 2 + 1
    s0 = np.exp(rng.uniform(np.log(s_range[0]), np.log(s_range[1]), size=n))
    u0k = np.zeros((n, K), complex)
    ks = np.arange(modes[0], modes[1] + 1)
    u0k[:, ks] = (rng.normal(size=(n, ks.size)) + 1j * rng.normal(size=(n, ks.size))) / ks * M.NX / 2 * s0[:, None]
    return u0k


def solve(fk, dtf, u0k):
    uk = burgers.solve(fk, dtf, M.NU, M.NX, stride=M.FINE, u0k=u0k)
    f = np.fft.irfft(fk[..., ::M.FINE], n=M.NX, axis=1)
    u = np.fft.irfft(uk, n=M.NX, axis=1)
    return torch.tensor(f), torch.tensor(u[..., 0]), torch.tensor(u)


def ood(rng, n, modes, s_range):
    fk, dtf = M.forcing(rng, n, M.T1, M.NT1)
    return solve(fk, dtf, initial(rng, n, modes, s_range))


def long_set(rng, n):
    parts = [M.forcing(rng, n, M.T1, M.NT1) for _ in range(N10)]
    dtf = parts[0][1]
    fk = np.concatenate([parts[0][0]] + [p[0][..., 1:] for p in parts[1:]], -1)
    return solve(fk, dtf, initial(rng, n, (1, 8), (0.03, 1.0)))


@torch.no_grad()
def rollout(m, f, u0, device, n_win):
    w = M.NT1 - 1
    out, state = [u0[..., None]], u0
    for i in range(n_win):
        seg = R.predict(m, f[..., i * w:(i + 1) * w + 1], state, device)
        out.append(seg[..., 1:])
        state = seg[..., -1]
    return torch.cat(out, -1)


def make_sets():
    base = C.make_data()
    return dict(train=base["single"], id=base["test"], roll4=base["roll"],
                amp=ood(np.random.default_rng(21), 100, (1, 8), (1.0, 3.0)),
                high=ood(np.random.default_rng(22), 100, (9, 16), (0.03, 1.0)),
                roll10=long_set(np.random.default_rng(23), 50))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sets = make_sets()
    t = np.linspace(0, M.T1, M.NT1)
    w = M.NT1 - 1
    out = json.loads(OUT.read_text()) if OUT.exists() else []
    for kind in KINDS:
        for seed in SEEDS:
            if any(r["model"] == kind and r["seed"] == seed for r in out):
                continue
            t0 = time.time()
            m = R.build(kind, seed, t, device)
            R.train(m, dict(train=sets["train"]), device, 30.0 if kind.startswith("wpo") else 1.0)
            m.eval()
            r = dict(model=kind, seed=seed, params=sum(p.numel() for p in m.parameters()))
            for s in ("id", "amp", "high"):
                f, u0, y = sets[s]
                r[s] = float(M.A.rel(R.predict(m, f, u0, device), y))
            f, u0, y = sets["roll4"]
            r["roll4"] = float(M.A.rel(rollout(m, f, u0, device, 4), y))
            f, u0, y = sets["roll10"]
            p = rollout(m, f, u0, device, N10)
            r["roll10"] = float(M.A.rel(p, y))
            r["roll10_first4"] = float(M.A.rel(p[..., :4 * w + 1], y[..., :4 * w + 1]))
            r["roll10_per_window"] = [float(M.A.rel(p[..., i * w + 1:(i + 1) * w + 1], y[..., i * w + 1:(i + 1) * w + 1]))
                                      for i in range(N10)]
            r["seconds"] = round(time.time() - t0, 1)
            out.append(r)
            print({q: (round(v, 5) if isinstance(v, float) else v) for q, v in r.items() if q != "roll10_per_window"},
                  flush=True)
            OUT.parent.mkdir(exist_ok=True)
            OUT.write_text(json.dumps(out, indent=1))
            del m
            torch.cuda.empty_cache()
    mean = lambda rs, q: np.mean([r[q] for r in rs])
    rng_ = lambda xs: f"{min(xs):.2f}–{max(xs):.2f}"
    lines = ["# (D2) Initial fields outside the training range; rollout over 10 windows (developed-start training)", "",
             "| model | params | id | amp (ratio to id) | high (ratio to id) | roll4 | roll10 [0, 50] | roll10 / its first 4 |",
             "|---|---|---|---|---|---|---|---|"]
    for kind in KINDS:
        rs = [r for r in out if r["model"] == kind]
        if not rs:
            continue
        lines.append(f"| {kind} | {rs[0]['params']} | {mean(rs, 'id'):.4f} | {mean(rs, 'amp'):.4f} "
                     f"({rng_([r['amp'] / r['id'] for r in rs])}) | {mean(rs, 'high'):.4f} "
                     f"({rng_([r['high'] / r['id'] for r in rs])}) | {mean(rs, 'roll4'):.4f} | {mean(rs, 'roll10'):.4f} | "
                     f"{rng_([r['roll10'] / r['roll10_first4'] for r in rs])} |")
    lines += ["", "Per-window error on roll10 (mean over seeds):", "",
              "| model | " + " | ".join(str(i + 1) for i in range(N10)) + " |", "|---|" + "---|" * N10]
    for kind in KINDS:
        rs = [r for r in out if r["model"] == kind]
        if rs:
            pw = np.mean([r["roll10_per_window"] for r in rs], 0)
            lines.append(f"| {kind} | " + " | ".join(f"{v:.3f}" for v in pw) + " |")
    (OUT.with_suffix(".md")).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
