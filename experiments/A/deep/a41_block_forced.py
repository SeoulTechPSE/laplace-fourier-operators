"""(A) Next-level step 4c: does the FNO block keep the physical-wavenumber form's benefits (forced Burgers)?

a40: on PDEBench the WPO with neuraloperator's block order (`wpo_mlp`: after gelu(K v + W v), a channel MLP
width -> width / 2 -> width with a gated skip of the block input, then gelu) closes half of the gap to the TFNO
(0.036 against 0.045; TFNO 0.026) and has the best linear probe. Before it becomes the default deep block it must
keep what the form is for. The MLP is pointwise in x and t, so it should, but it adds nonlinear capacity that could
also fit in-band structure that does not continue outside the band.

Setup: a36's forced Burgers (a11 data and training: lr 3e-3, pole lr x30, 12000 steps, 3 seeds; no sweep, the
WPO's selected configuration, which a40's sweep also selected), a36's evaluation: test, (5, 10] extrapolation,
linear probe outside the forced band (k = 9..16), unseen forcing modes k = 9..12. Width reduced to the WPO's real
budget (59,529): width 23, 57,011, the closest (checked before this commit). References: a36's `wpo` (a11) and `tfno_r`.

**Predictions** (written before running).

1. `wpo_mlp`'s mean test error <= 0.0215 (at least 10 % below the WPO's 0.0237; a40 gave 21 % on PDEBench).
2. The form's benefits are kept in every seed: unseen-mode ratio <= 3 and probe error outside the band <= 0.3.
3. Mean (5, 10] / test ratio <= 2.5 (WPO 1.83, TFNO 10.3).

    python experiments/A/deep/a41_block_forced.py [seed ...]     # results/a41_block_forced.md / .json
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
import a9_burgers as M  # noqa: E402
import a11_fair as F  # noqa: E402
import a36_index_poles as X36  # noqa: E402
from lfo.metrics import n_real_params  # noqa: E402
from lfo.models import WavenumberPoleOperator1d  # noqa: E402

OUT = HERE / "results" / "a41_block_forced.json"
CFG = dict(lr=3e-3, plr=30.0)


class WPOMLP(nn.Module):
    """Forced WPO with neuraloperator's block (as a40's `wpo_mlp`); input (b, n_x, n_t) forcing."""

    def __init__(self, t, width):
        super().__init__()
        kw = {k: v for k, v in F.WPO_KW.items() if k != "width"}
        self.net = WavenumberPoleOperator1d(t, width=width, **kw)
        f64 = dict(dtype=torch.float64)
        h = max(width // 2, 1)
        self.mlp = nn.ModuleList(nn.Sequential(nn.Linear(width, h, **f64), nn.GELU(), nn.Linear(h, width, **f64))
                                 for _ in self.net.K)
        self.gate = nn.ParameterList(nn.Parameter(torch.ones(width, **f64)) for _ in self.net.K)

    def forward(self, f):
        n = self.net
        v = n.lift(f.to(torch.float64)[..., None]).permute(0, 3, 1, 2)
        last = len(n.K) - 1
        for i, (K, W) in enumerate(zip(n.K, n.W)):
            vin = v.permute(0, 2, 3, 1)
            y = K(v).permute(0, 2, 3, 1) + W(vin)
            if i < last:
                y = Fn.gelu(y)
            y = self.mlp[i](y) + self.gate[i] * vin
            if i < last:
                y = Fn.gelu(y)
            v = y.permute(0, 3, 1, 2)
        return n.proj(v.permute(0, 2, 3, 1))[..., 0].to(f.dtype)


def width_for(t):
    target = n_real_params(WavenumberPoleOperator1d(t, **F.WPO_KW))
    return min(range(8, 32), key=lambda w: abs(n_real_params(WPOMLP(t, w)) - target))


def locked_update(fn):
    OUT.parent.mkdir(exist_ok=True)
    with open(f"/tmp/{OUT.stem}.lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        out = json.loads(OUT.read_text()) if OUT.exists() else dict(runs=[])
        fn(out)
        OUT.write_text(json.dumps(out, indent=1, default=float))
        return out


def run(seeds, device):
    data = M.make_data()
    data["high"] = X36.high_set()
    t = np.linspace(0, M.T1, M.NT1)
    pfk, pdt = M.probe_inputs()
    w = width_for(t)
    for seed in seeds:
        if any(r["seed"] == seed for r in locked_update(lambda o: None)["runs"]):
            continue
        t1 = time.time()
        torch.manual_seed(seed)
        m = WPOMLP(t, w).to(device)
        F.train(m, data, device, steps=F.FINAL_STEPS, **CFG)
        r = X36.evaluate(m, "wpo_mlp", seed, CFG, data, t, pfk, pdt, device, t1)
        r["width"] = w
        print(seed, f"w {w} test {r['test']:.4f} ext/test {r['ext_late'] / r['test']:.2f}",
              f"high/test {r['high'] / r['test']:.2f} oob {r['oob']:.3f} {r['seconds']} s", flush=True)
        locked_update(lambda o: o["runs"].append(r))
        del m
        torch.cuda.empty_cache()


def summarize():
    out = json.loads(OUT.read_text())
    a36 = json.loads((HERE / "results" / "a36_index_poles.json").read_text())
    rg = lambda xs, f: f"{np.mean(xs):{f}} ({min(xs):{f}}–{max(xs):{f}})"
    rows = [("wpo (a11)", a36["ref"]), ("tfno_r (a36)", [r for r in a36["final"] if r["model"] == "tfno_r"]),
            ("wpo_mlp", sorted(out["runs"], key=lambda r: r["seed"]))]
    lines = ["# (A) The FNO block on the WPO, forced Burgers (a36 setup)", "", "Means over 3 seeds (range).", "",
             "| model | real params | test | (5, 10] / test | probe outside the band | unseen / test |",
             "|---|---|---|---|---|---|"]
    for name, rs in rows:
        if rs:
            lines.append(f"| {name} | {rs[0]['real_params']} | {rg([r['test'] for r in rs], '.4f')} | "
                         f"{rg([r['ext_late'] / r['test'] for r in rs], '.2f')} | {rg([r['oob'] for r in rs], '.3f')} | "
                         f"{rg([r['high'] / r['test'] for r in rs], '.2f')} |")
    OUT.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    seeds = [int(a) for a in sys.argv[1:] if a.isdigit()] or list(M.SEEDS)
    if "--summary" not in sys.argv:
        run(seeds, torch.device("cuda"))
    summarize()


if __name__ == "__main__":
    main()
