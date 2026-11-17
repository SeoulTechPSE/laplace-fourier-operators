"""(C) E5, second pass: Duffing models whose poles can move, for the linear-probe diagnostic.

Damped Duffing (`1D_Duffing_c05`, original split), E1 training protocol `common` (the original LNO one), 5 seeds.

| variant | model | init | pole lr |
|---|---|---|---|
| lno_plr10 | original 1-layer LNO (256 poles) | original (Re, Im mu in [0, 1/16)) | x 10 |
| lno_stable_plr10 | original 1-layer LNO | E1 "stable": Re mu = -exp(U[log 1/T, log 10/T]), Im mu ~ U[0, pi / (4 dt)] | x 10 |
| tied4_stable_plr10 | E1 tied single layer, 4 slots (LNO v1 layer, linear) | E1 "stable" | x 10 |

The base variant (original init, pole lr x 1) is the first pass (`reproduce_lno.py --tag e5`).

    python experiments/C/e5_probe/e5_train.py --variant lno_plr10 --seeds 0 1 2 3 4
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE.parent / "e1_audit"))
sys.path.insert(0, str(ROOT / "experiments" / "foundation"))
sys.path.insert(0, str(ROOT / "src"))
import e1_audit as E  # noqa: E402
import reproduce_lno as RL  # noqa: E402

CASE = "duffing_c05"
VARIANTS = {"lno_plr10": ("lno", "original", 10.0),
            "lno_stable_plr10": ("lno", "stable", 10.0),
            "tied4_stable_plr10": ("tied4", "stable", 10.0)}
CKPT = ROOT / "runs" / "C" / "e5"
RESULTS = HERE / "results" / "e5_train"


def data_and_grid():
    split, grids = RL.load(RL.CASES[CASE], None, "ood")
    data = {k: tuple(a.reshape(a.shape[0], -1).float() for a in v) for k, v in split.items()}
    return data, grids


@torch.no_grad()
def stable_init(mu_param, t, gen):
    """E1 "stable" init in place: Re mu = -exp(U[log 1/T, log 10/T]), Im mu ~ U[0, pi / (4 dt)]."""
    T, dt = float(t[-1] - t[0]), float(t[1] - t[0])
    re = -torch.exp(torch.empty(mu_param.shape).uniform_(math.log(1 / T), math.log(10 / T), generator=gen))
    im = torch.empty(mu_param.shape).uniform_(0, math.pi / (4 * dt), generator=gen)
    mu_param.copy_(torch.complex(re, im).to(mu_param.dtype))


def build(variant, grids, seed):
    kind, init, _ = VARIANTS[variant]
    torch.manual_seed(seed)
    t = torch.as_tensor(np.asarray(grids["t"], float).ravel(), dtype=torch.float32)
    gen = torch.Generator().manual_seed(1000 + seed)
    if kind == "lno":
        model, _ = RL.build("lno", RL.CASES[CASE], grids)
        if init == "stable":
            stable_init(model.conv0.weights_pole, t, gen)
    else:
        model = E.Layer1("tied", 4, t, "stable")
    return model


def poles_of(model):
    """Stored poles (folded to Im >= 0) and, for the LNO, the residues of the same entries."""
    if hasattr(model, "conv0"):
        mu = model.conv0.weights_pole.detach().cpu().numpy()
    else:
        mu = model.pr.mu.detach().cpu().numpy()
    mu = mu.ravel()
    return np.where(mu.imag < 0, mu.conj(), mu)


def run(variant, seed, device):
    out = RESULTS / f"{variant}_seed{seed}.json"
    if out.exists():
        return
    data, grids = data_and_grid()
    model = build(variant, grids, seed).to(device)
    mu0 = poles_of(model)
    t0 = time.time()
    hist = E.train(model, data, E.PROTOCOLS["common"], device, pole_lr_mult=VARIANTS[variant][2])
    xte, yte = (a.to(device)[..., None] for a in data["test"])
    test = E.evaluate(model, xte, yte)
    CKPT.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), CKPT / f"{variant}_seed{seed}.pt")
    mu = poles_of(model)
    res = dict(variant=variant, seed=seed, test_rel_l2=test, train_hist=hist, seconds=round(time.time() - t0, 1),
               travel_median=float(np.median(np.abs(mu - mu0))), n_poles=int(mu.size))
    RESULTS.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    print(f"{variant} s{seed}: test {test:.3f}, median pole travel {res['travel_median']:.3f} ({res['seconds']:.0f}s)",
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", nargs="+", default=list(VARIANTS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for v in a.variant:
        for s in a.seeds:
            run(v, s, device)


if __name__ == "__main__":
    main()
