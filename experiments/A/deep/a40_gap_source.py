"""(A) Next-level step 4b: where the TFNO's nonlinear edge on PDEBench Burgers comes from.

a39 ruled out the linear symbol: odd residue powers or local mixing leave the WPO at 0.047 / 0.054 against the
TFNO's 0.026 (a37, equal real budgets). Reading the neuraloperator FNO used as the baseline shows two structural
differences beyond the spectral weights:

- **Nonlinear block.** Each FNO block is `gelu(spectral(x) + skip(x))`, then a channel MLP (width -> width / 2 ->
  width, GELU) plus a soft-gated skip of the block input, then `gelu`; the lifting is a two-layer MLP. The WPO
  block is only `gelu(K x + W x)` with W linear: one nonlinearity per layer against two.
- **Time axis.** The TFNO's spectral weights span 21 temporal modes over the padded window (non-causal mixing of
  the whole trajectory), and it gets (x, t) as grid channels. The WPO's FOH recursion is causal and gets no t.

Three variants on a37's task, metrics and splits (nRMSE 256 x 41; 1024-point zero-shot; linear probe at u = 0),
each at the WPO's real budget (61,130; `lfo.metrics.n_real_params`):

- `wpo_mlp`: the WPO with neuraloperator's block order: v <- gelu(K v + W v); v <- gelu(MLP(v) + g * v_in), MLP
  width -> width / 2 -> width with GELU, g a per-channel gate (as SoftGating); no activation after the last
  layer. Width reduced to the budget. Poles, residues and the time axis unchanged.
- `tfno_nomlp`: the a37 TFNO without channel MLPs (`use_channel_mlp=False`), rank re-matched to the budget.
- `fno_x`: an FNO over x only (Tucker, 64 modes, width 24, 4 layers), applied to each time sample independently
  with inputs (u0, t) and the x grid: u(., t) = G(u0, t). No mixing across time at all.

Budgets (checked before this commit): `wpo_mlp` width 23, 63,382 (the MLPs cost less than one width step);
`tfno_nomlp` rank 0.015, 62,201; `fno_x` rank 0.346, 62,257.

Training: a37's protocol (relative L2, Adam with cosine decay, batch 25, 3 seeds, 12000 steps), with a short lr
sweep at 4000 steps (model seed 0, selection by validation): `wpo_mlp` lr 1e-3, 3e-3 (pole lr x30, as the WPO);
`tfno_nomlp` and `fno_x` lr 3e-3, 1e-2.

**Predictions** (written before running).

1. `wpo_mlp`'s mean nRMSE <= 0.036 (closes at least half of the gap from 0.045 to 0.026).
2. `tfno_nomlp`'s mean nRMSE >= 0.032 (removing the MLP costs the TFNO at least a third of its edge).
3. `fno_x`'s mean nRMSE <= 0.030 (temporal mixing is not what gives the TFNO its edge).
4. `wpo_mlp` keeps the WPO's properties: mean 1024 / 256 ratio <= 1.1 and mean dispersion error <= 0.3.

    python experiments/A/deep/a40_gap_source.py [wpo_mlp|tfno_nomlp|fno_x ...]   # results/a40_gap_source.md / .json
"""
from __future__ import annotations

import fcntl
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import a37_pdebench_burgers as X37  # noqa: E402
from lfo.metrics import n_real_params  # noqa: E402
from lfo.models import WavenumberPoleOperator1d  # noqa: E402

OUT = HERE / "results" / "a40_gap_source.json"
VARIANTS = ("wpo_mlp", "tfno_nomlp", "fno_x")
SWEEP = {"wpo_mlp": [dict(lr=1e-3, plr=30.0), dict(lr=3e-3, plr=30.0)],
         "tfno_nomlp": [dict(lr=3e-3), dict(lr=1e-2)], "fno_x": [dict(lr=3e-3), dict(lr=1e-2)]}
T = None   # time samples, set in main


class WPOMLP(nn.Module):
    """WPO with neuraloperator's block: gelu(K v + W v), then MLP + gated skip of the block input."""

    def __init__(self, t, width):
        super().__init__()
        kw = {k: v for k, v in X37.WPO_KW.items() if k != "width"}
        self.net = WavenumberPoleOperator1d(t, ic=True, width=width, **kw)
        f64 = dict(dtype=torch.float64)
        h = max(width // 2, 1)
        self.mlp = nn.ModuleList(nn.Sequential(nn.Linear(width, h, **f64), nn.GELU(), nn.Linear(h, width, **f64))
                                 for _ in self.net.K)
        self.gate = nn.ParameterList(nn.Parameter(torch.ones(width, **f64)) for _ in self.net.K)

    def forward(self, u0, n_t):
        n = self.net
        v = n.lift(torch.zeros(*u0.shape, n_t, 1, dtype=torch.float64, device=u0.device)).permute(0, 3, 1, 2)
        v0 = n.lift0(u0.to(torch.float64)[..., None]).permute(0, 2, 1)
        last = len(n.K) - 1
        for i, (K, W) in enumerate(zip(n.K, n.W)):
            vin = v.permute(0, 2, 3, 1)                                         # (b, n_x, n_t, c)
            y = (K(v, v0) if i == 0 else K(v)).permute(0, 2, 3, 1) + W(vin)
            if i < last:
                y = Fn.gelu(y)
            y = self.mlp[i](y) + self.gate[i] * vin
            if i < last:
                y = Fn.gelu(y)
            v = y.permute(0, 3, 1, 2)
        return n.proj(v.permute(0, 2, 3, 1))[..., 0]


class TFNONoMLP(X37.TFNO):
    def __init__(self, rank):
        nn.Module.__init__(self)
        from neuralop.models import FNO
        self.net = FNO(n_modes=(64, 21), in_channels=1, out_channels=1, hidden_channels=24, n_layers=4,
                       domain_padding=[0.0, 0.25], factorization="tucker", rank=rank, use_channel_mlp=False)


class FNOX(nn.Module):
    """FNO over x per time sample, inputs (u0, t): no mixing across time."""

    def __init__(self, rank, t):
        super().__init__()
        from neuralop.models import FNO
        self.net = FNO(n_modes=(64,), in_channels=2, out_channels=1, hidden_channels=24, n_layers=4,
                       factorization="tucker", rank=rank)
        self.register_buffer("t", torch.tensor(np.asarray(t), dtype=torch.float32))

    def forward(self, u0, n_t):
        b, n_x = u0.shape
        uu = u0.float()[:, None, :].expand(b, n_t, n_x)
        tt = self.t[:n_t][None, :, None].expand(b, n_t, n_x)
        x = torch.stack([uu, tt], 2).reshape(b * n_t, 2, n_x)
        return self.net(x).reshape(b, n_t, n_x).permute(0, 2, 1).to(torch.float64)


def best(build, cands, target):
    return min(cands, key=lambda c: abs(n_real_params(build(c)) - target))


def builder(v, t, target):
    if v == "wpo_mlp":
        w = best(lambda w: WPOMLP(t, w), range(8, 32), target)
        return (lambda: WPOMLP(t, w)), dict(width=w)
    ranks = [r / 1000 for r in range(2, 60)]
    if v == "tfno_nomlp":
        r = best(TFNONoMLP, ranks, target)
        return (lambda: TFNONoMLP(r)), dict(rank=r)
    r = best(lambda r: FNOX(r, t), [r / 1000 for r in range(100, 500, 2)], target)
    return (lambda: FNOX(r, t)), dict(rank=r)


def locked_update(fn):
    """Read-modify-write the shared results file under a lock (variants run as parallel processes)."""
    OUT.parent.mkdir(exist_ok=True)
    with open(OUT.with_suffix(".lock"), "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        out = json.loads(OUT.read_text()) if OUT.exists() else dict(sweep=[], runs=[])
        fn(out)
        OUT.write_text(json.dumps(out, indent=1, default=float))
        return out


def run(v, device):
    d = X37.load()
    t = d["t"].numpy()
    S = X37.splits(d)
    n_t = S["train"].shape[2]
    full_test = d["full"][X37.NTR:].to(torch.float64)
    target = n_real_params(X37.WPO(t))
    make, info = builder(v, t, target)
    cur = locked_update(lambda o: None)
    for cfg in SWEEP[v]:
        if any(s["model"] == v and s["cfg"] == cfg for s in cur["sweep"]):
            continue
        t1 = time.time()
        torch.manual_seed(0)
        m = make().to(device)
        X37.train(m, S["train"], device, steps=X37.SWEEP_STEPS, **cfg)
        m.eval()
        s = dict(model=v, cfg=cfg, vali=X37.nrmse(X37.predict(m, S["vali"], device), S["vali"]),
                 seconds=round(time.time() - t1, 1))
        print("sweep", s, flush=True)
        cur = locked_update(lambda o: o["sweep"].append(s))
        del m
        torch.cuda.empty_cache()
    sw = [s for s in cur["sweep"] if s["model"] == v and np.isfinite(s["vali"])]
    cfg = min(sw, key=lambda s: s["vali"])["cfg"]
    for seed in X37.SEEDS:
        if any(r["model"] == v and r["seed"] == seed for r in cur["runs"]):
            continue
        t1 = time.time()
        torch.manual_seed(seed)
        m = make().to(device)
        X37.train(m, S["train"], device, steps=X37.FINAL_STEPS, **cfg)
        m.eval()
        r = dict(model=v, seed=seed, cfg=cfg, **info, real_params=n_real_params(m),
                 test=X37.nrmse(X37.predict(m, S["test"], device), S["test"]),
                 test_1024=X37.nrmse(X37.predict(m, full_test, device), full_test),
                 probe=X37.model_probe(m, t, 256, n_t, device), seconds=round(time.time() - t1, 1))
        print(v, seed, f"test {r['test']:.4f} 1024 {r['test_1024']:.4f} nu' {r['probe']['nu_hat']:.4f}",
              f"{r['seconds']} s", flush=True)
        cur = locked_update(lambda o: o["runs"].append(r))
        del m
        torch.cuda.empty_cache()


def summarize():
    out = json.loads(OUT.read_text())
    a37 = json.loads((HERE / "results" / "a37_pdebench_burgers.json").read_text())
    rg = lambda xs, f: f"{np.mean(xs):{f}} ({min(xs):{f}}–{max(xs):{f}})"
    lines = ["# (A) Where the TFNO's edge on PDEBench Burgers comes from (a37 setup)", "",
             "Means over 3 seeds (range); all models at the WPO's real budget.", "",
             "| model | lr | real params | nRMSE 256 | nRMSE 1024 / 256 | probe ν' (truth 0.1257) | dispersion error |",
             "|---|---|---|---|---|---|---|"]
    rows = [("wpo (a37 rerun)", a37.get("rerun", [])), ("tfno_r (a37)", [r for r in a37["final"] if r["model"] == "tfno_r"])]
    rows += [(v, [r for r in out["runs"] if r["model"] == v]) for v in VARIANTS]
    for name, rs in rows:
        if rs:
            lines.append(f"| {name} | {rs[0].get('cfg', {}).get('lr', '')} | {rs[0]['real_params']} | "
                         f"{rg([r['test'] for r in rs], '.4f')} | {rg([r['test_1024'] / r['test'] for r in rs], '.2f')} | "
                         f"{rg([r['probe']['nu_hat'] for r in rs], '.4f')} | {rg([r['probe']['disp_err'] for r in rs], '.3f')} |")
    lines += ["", "Sweep (4000 steps, seed 0, validation nRMSE): " +
              "; ".join(f"{s['model']} lr {s['cfg']['lr']}: {s['vali']:.4f}" for s in out["sweep"])]
    OUT.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    which = [a for a in sys.argv[1:] if a in VARIANTS]
    if "--summary" not in sys.argv:
        for v in which or VARIANTS:
            run(v, torch.device("cuda"))
    summarize()


if __name__ == "__main__":
    main()
