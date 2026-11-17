"""(C) E5, second pass: Duffing models whose poles can move, for the linear-probe diagnostic.

Damped Duffing (`1D_Duffing_c05`, original split), E1 training protocol `common` (the original LNO one), 5 seeds.

| variant | model | init | pole lr |
|---|---|---|---|
| lno_plr10 | original 1-layer LNO (256 poles) | original (Re, Im mu in [0, 1/16)) | x 10 |
| lno_stable_plr10 | original 1-layer LNO | E1 "stable": Re mu = -exp(U[log 1/T, log 10/T]), Im mu ~ U[0, pi / (4 dt)] | x 10 |
| tied4_stable_plr10 | E1 tied single layer, 4 slots (LNO v1 layer, linear) | E1 "stable" | x 10 |

On the LNO data the base variant (original init, pole lr x 1) is the first pass (`reproduce_lno.py --tag e5`);
on the e5_data.py sets (`--data small | mixed | ...`) `base` is trained here, with the same protocol. Pendulum tags
(`pend_*`, sixth pass) use the pendulum LNO architecture (grid input, 8 modes) with the same `common` protocol.

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
ORIG = {"orig": "duffing_c05", "pend_orig": "pendulum_c05", "c0_orig": "duffing_c0", "lz_orig": "lorenz_rho5", "lz_r10_orig": "lorenz_rho10"}     # data tags that are original LNO data


def case_of(data_tag):
    """LNO case for a data tag: pendulum sets use the pendulum LNO (grid input, 8 modes); c0_* the undamped Duffing
    case (same architecture as duffing_c05, its own data)."""
    return ("pendulum_c05" if data_tag.startswith("pend") else "duffing_c0" if data_tag.startswith("c0_")
            else "lorenz_rho10" if data_tag.startswith("lz_r10") else "lorenz_rho5" if data_tag.startswith("lz_")
            else CASE)
VARIANTS = {"base": ("lno", "original", 1.0),
            "lno_plr10": ("lno", "original", 10.0),
            "lno_stable_plr10": ("lno", "stable", 10.0),
            "tied4_stable_plr10": ("tied4", "stable", 10.0)}
CKPT = ROOT / "runs" / "C" / "e5"
RESULTS = HERE / "results" / "e5_train"


def data_and_grid(data_tag="orig"):
    """orig: the LNO data (original split); small / mixed / medium / moderate: e5_data.py sets (same grid). Tensors (b, n)."""
    split, grids = RL.load(RL.CASES[case_of(data_tag)], None, "ood")
    if data_tag in ORIG:
        data = {k: tuple(a.reshape(a.shape[0], -1).float() for a in v) for k, v in split.items()}
    else:
        z = np.load(ROOT / "runs" / "C" / "e5_data" / f"{data_tag}.npz")
        s = np.cumsum([0, *z["split"]])
        data = {k: (torch.tensor(z["f"][a:b]).float(), torch.tensor(z["u"][a:b]).float())
                for k, a, b in zip(("train", "vali", "test"), s[:-1], s[1:])}
    return data, grids


def tag(variant, data_tag):
    return variant if data_tag == "orig" else f"{data_tag}_{variant}"


class Offset(torch.nn.Module):
    """Tied layer + a learned output offset b(t) (ninth pass, Lorenz): the data are x(t) about a nonzero operating
    point, which a linear layer cannot output. b does not depend on the input, so the probes (odd part, Jacobian at
    f = 0) see only the layer."""

    def __init__(self, layer, n):
        super().__init__()
        self.layer, self.pr = layer, layer.pr
        self.offset = torch.nn.Parameter(torch.zeros(n))

    def forward(self, x):
        return self.layer(x) + self.offset[None, :, None]


@torch.no_grad()
def stable_init(mu_param, t, gen):
    """E1 "stable" init in place: Re mu = -exp(U[log 1/T, log 10/T]), Im mu ~ U[0, pi / (4 dt)]."""
    T, dt = float(t[-1] - t[0]), float(t[1] - t[0])
    re = -torch.exp(torch.empty(mu_param.shape).uniform_(math.log(1 / T), math.log(10 / T), generator=gen))
    im = torch.empty(mu_param.shape).uniform_(0, math.pi / (4 * dt), generator=gen)
    mu_param.copy_(torch.complex(re, im).to(mu_param.dtype))


def build(variant, grids, seed, data_tag="orig"):
    kind, init, _ = VARIANTS[variant]
    torch.manual_seed(seed)
    t = torch.as_tensor(np.asarray(grids["t"], float).ravel(), dtype=torch.float32)
    gen = torch.Generator().manual_seed(1000 + seed)
    if kind == "lno":
        model, _ = RL.build("lno", RL.CASES[case_of(data_tag)], grids)
        if init == "stable":
            stable_init(model.conv0.weights_pole, t, gen)
    else:
        model = E.Layer1("tied", 4, t, "stable")
        if data_tag.startswith("lz_"):
            model = Offset(model, t.numel())
    return model


def poles_of(model):
    """Stored poles (folded to Im >= 0) and, for the LNO, the residues of the same entries."""
    if hasattr(model, "conv0"):
        mu = model.conv0.weights_pole.detach().cpu().numpy()
    else:
        mu = model.pr.mu.detach().cpu().numpy()
    mu = mu.ravel()
    return np.where(mu.imag < 0, mu.conj(), mu)


def run(variant, seed, device, data_tag="orig"):
    name = tag(variant, data_tag)
    out = RESULTS / f"{name}_seed{seed}.json"
    if out.exists():
        return
    data, grids = data_and_grid(data_tag)
    model = build(variant, grids, seed, data_tag).to(device)
    mu0 = poles_of(model)
    t0 = time.time()
    if isinstance(model, Offset) and data_tag.startswith("lz_r10"):
        # tenth pass (rule v4): start the offset at the training-mean trajectory; under the `common` lr schedule a
        # zero-initialised offset stopped at ~1.5 of 3.27 in the ninth pass (the travel bound)
        with torch.no_grad():
            model.offset.copy_(data["train"][1].mean(0).to(device))
    hist = E.train(model, data, E.PROTOCOLS["common"], device, pole_lr_mult=VARIANTS[variant][2])
    xte, yte = (a.to(device)[..., None] for a in data["test"])
    test = E.evaluate(model, xte, yte)
    CKPT.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), CKPT / f"{name}_seed{seed}.pt")
    mu = poles_of(model)
    res = dict(variant=variant, data=data_tag, seed=seed, test_rel_l2=test, train_hist=hist, seconds=round(time.time() - t0, 1),
               travel_median=float(np.median(np.abs(mu - mu0))), n_poles=int(mu.size))
    RESULTS.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    print(f"{name} s{seed}: test {test:.3f}, median pole travel {res['travel_median']:.3f} ({res['seconds']:.0f}s)",
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", nargs="+", default=list(VARIANTS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--data", default="orig", choices=["orig", "small", "mixed", "medium", "moderate", "pend_orig", "pend_medium", "pend_large", "wide", "pend_wide", "c0_orig", "c0_small", "c0_medium", "lz_orig", "lz_small", "lz_wide", "lz_r10_orig", "lz_r10_small", "lz_r10_wide"])
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for v in a.variant:
        for s in a.seeds:
            run(v, s, device, a.data)


if __name__ == "__main__":
    main()
