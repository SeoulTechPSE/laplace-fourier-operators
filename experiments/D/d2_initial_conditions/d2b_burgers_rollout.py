"""(D2) Deep nonlinear rollout: forced Burgers with initial fields, chained over 4 windows.

The linear pass (`d2_ic_rollout.py`) chained windows exactly. Here the operator is deep and the system nonlinear,
so chaining can accumulate error. The question is how the initial field should enter.

**System.** Forced viscous Burgers (`lfo.data.burgers`, nu = 0.1, n_x = 64), windows of T = 5 (n_t = 201).
- Initial fields: modes 1..8 with amplitude ~ s0 / k, s0 log-uniform in [0.03, 1].
- Forcing: the a9 family (scale log-uniform in [0.03, 1]).
- Splits 200 / 50 / 100 windows.
- Rollout set: 50 trajectories solved continuously on [0, 20] (4 windows). Each starts from an initial field drawn
  as above, with forcing over the whole interval.

**Models** (3 seeds; relative L2; Adam with cosine decay, 6000 steps, batch 25):
- `wpo_ic`: `WavenumberPoleOperator1d(ic=True)`, the a11 configuration (4 layers, 6 poles, stable, lr 3e-3, pole
  ratio 30) at width 23. The initial field is lifted and enters the first pole layer as a free response that shares
  its poles. Its gamma matrices are constant in k, and the width is 23 instead of 24, to keep the budget: 61.1 k
  parameters against 59.6 k / 59.1 k for the others.
- `wpo_chan`: the same WPO without the IC path. The initial field is a second input channel, repeated over time.
- `tfno_chan`: the a13 TFNO (Tucker, n_modes 32 x 64, lr 3e-3) with the same two input channels.

**Rollout.** Each window's last predicted field is the next window's initial field; the error is relative L2 over
[0, 20].

**Predictions** (written before running).

1. `wpo_ic`'s mean window test error is at or below `wpo_chan`'s and `tfno_chan`'s.
2. Rollout error / window test error is <= 2 for `wpo_ic` in every seed and > 2 for `tfno_chan` in every seed.
3. `wpo_ic` has the lowest mean rollout error of the three.

    python experiments/D/d2_initial_conditions/d2b_burgers_rollout.py      # results/d2b_burgers_rollout.md / .json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "A" / "deep"))
import a9_burgers as M  # noqa: E402
from lfo.data import burgers  # noqa: E402
from lfo.models import WavenumberPoleOperator1d  # noqa: E402

N_WIN, STEPS, SEEDS, BATCH = 4, 6000, (0, 1, 2), 25
OUT = HERE / "results" / "d2b_burgers_rollout.json"


def initial(rng, n):
    K = M.NX // 2 + 1
    s0 = np.exp(rng.uniform(np.log(0.03), 0.0, size=n))
    u0k = np.zeros((n, K), complex)
    ks = np.arange(1, 9)
    u0k[:, 1:9] = (rng.normal(size=(n, 8)) + 1j * rng.normal(size=(n, 8))) / ks * M.NX / 2 * s0[:, None]
    return u0k


def sample(rng, n, T, n_t):
    fk, dtf = M.forcing(rng, n, T, n_t)
    u0k = initial(rng, n)
    uk = burgers.solve(fk, dtf, M.NU, M.NX, stride=M.FINE, u0k=u0k)
    f = np.fft.irfft(fk[..., ::M.FINE], n=M.NX, axis=1)
    u = np.fft.irfft(uk, n=M.NX, axis=1)
    return torch.tensor(f), torch.tensor(u[..., 0]), torch.tensor(u)


def make_data():
    rng = np.random.default_rng(0)
    f, u0, u = sample(rng, sum(M.SPLIT), M.T1, M.NT1)
    c = np.cumsum([0, *M.SPLIT])
    d = {k: (f[a:b], u0[a:b], u[a:b]) for k, a, b in zip(("train", "vali", "test"), c[:-1], c[1:])}
    d["roll"] = sample(np.random.default_rng(1), 50, N_WIN * M.T1, N_WIN * (M.NT1 - 1) + 1)
    return d


class WPOChan(nn.Module):
    """The deep WPO with a 2-channel lift (forcing, initial field repeated over time)."""

    def __init__(self, t):
        super().__init__()
        self.net = WavenumberPoleOperator1d(t, **dict(M.WPO_KW, stable=True))
        self.net.lift = nn.Linear(2, self.net.lift.out_features, dtype=torch.float64)

    def forward(self, f, u0):
        x = torch.stack([f.to(torch.float64), u0.to(torch.float64)[..., None].expand_as(f)], -1)
        v = self.net.lift(x).permute(0, 3, 1, 2)
        for i, (K, W) in enumerate(zip(self.net.K, self.net.W)):
            v = K(v) + W(v.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            if i < len(self.net.K) - 1:
                v = F.gelu(v)
        return self.net.proj(v.permute(0, 2, 3, 1))[..., 0]


class WPOIC(nn.Module):
    def __init__(self, t):
        super().__init__()
        self.net = WavenumberPoleOperator1d(t, **dict(M.WPO_KW, stable=True, width=23), ic=True)

    def forward(self, f, u0):
        return self.net(f, u0)


class TFNOChan(nn.Module):
    def __init__(self):
        super().__init__()
        from neuralop.models import FNO
        self.net = FNO(n_modes=(32, 64), in_channels=2, out_channels=1, hidden_channels=24, n_layers=4,
                       domain_padding=[0.0, 0.25], factorization="tucker", rank=0.02)

    def forward(self, f, u0):
        x = torch.stack([f.float(), u0.float()[..., None].expand_as(f)], 1)
        return self.net(x)[:, 0].to(f.dtype)


def build(kind, seed, t, device):
    torch.manual_seed(seed)
    return {"wpo_ic": lambda: WPOIC(t), "wpo_chan": lambda: WPOChan(t), "tfno_chan": TFNOChan}[kind]().to(device)


def train(m, data, device, plr):
    f, u0, y = data["train"]
    is_pole = lambda n: n.endswith("mu_re") or n.endswith("mu_im")
    groups = [dict(params=[p for n, p in m.named_parameters() if not is_pole(n)], lr=3e-3)]
    pole = [p for n, p in m.named_parameters() if is_pole(n)]
    if pole:
        groups.append(dict(params=pole, lr=3e-3 * plr))
    opt = torch.optim.Adam(groups)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS)
    g = torch.Generator().manual_seed(0)
    for _ in range(STEPS):
        idx = torch.randint(0, f.shape[0], (BATCH,), generator=g)
        opt.zero_grad()
        loss = M.A.rel(m(f[idx].to(device), u0[idx].to(device)), y[idx].to(device))
        loss.backward()
        opt.step()
        sch.step()


@torch.no_grad()
def predict(m, f, u0, device, bs=10):
    return torch.cat([m(f[i:i + bs].to(device), u0[i:i + bs].to(device)).cpu().to(torch.float64) for i in range(0, len(f), bs)])


@torch.no_grad()
def rollout(m, f, u0, device):
    w = M.NT1 - 1
    out, state = [u0[..., None]], u0
    for i in range(N_WIN):
        seg = predict(m, f[..., i * w:(i + 1) * w + 1], state, device)
        out.append(seg[..., 1:])
        state = seg[..., -1]
    return torch.cat(out, -1)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = make_data()
    t = np.linspace(0, M.T1, M.NT1)
    out = json.loads(OUT.read_text()) if OUT.exists() else []
    for kind in ("wpo_ic", "wpo_chan", "tfno_chan"):
        for seed in SEEDS:
            if any(r["model"] == kind and r["seed"] == seed for r in out):
                continue
            t0 = time.time()
            m = build(kind, seed, t, device)
            train(m, data, device, 30.0 if kind.startswith("wpo") else 1.0)
            m.eval()
            f, u0, y = data["test"]
            test = float(M.A.rel(predict(m, f, u0, device), y))
            fr, u0r, yr = data["roll"]
            roll = float(M.A.rel(rollout(m, fr, u0r, device), yr))
            r = dict(model=kind, seed=seed, params=sum(p.numel() for p in m.parameters()), test=test, rollout=roll,
                     ratio=roll / test, seconds=round(time.time() - t0, 1))
            out.append(r)
            print(r, flush=True)
            OUT.parent.mkdir(exist_ok=True)
            OUT.write_text(json.dumps(out, indent=1))
            del m
            torch.cuda.empty_cache()
    lines = ["# (D2) Deep rollout on forced Burgers with initial fields: 4 chained windows", "",
             "| model | params | window test (3 seeds) | rollout error over [0, 20] | rollout / test |", "|---|---|---|---|---|"]
    for kind in ("wpo_ic", "wpo_chan", "tfno_chan"):
        rs = [r for r in out if r["model"] == kind]
        rg = lambda q, f: f"{min(r[q] for r in rs):{f}}–{max(r[q] for r in rs):{f}} (mean {np.mean([r[q] for r in rs]):{f}})"
        lines.append(f"| {kind} | {rs[0]['params']} | {rg('test', '.4f')} | {rg('rollout', '.4f')} | {rg('ratio', '.2f')} |")
    (OUT.with_suffix(".md")).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
