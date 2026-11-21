"""(D2) Cahn–Hilliard with initial fields: does the IC path lack capacity? Two variants at a matched budget.

d2g–d2j: on forced 1D Cahn–Hilliard with initial fields, the TFNO is 2x better than the IC-path WPO in one window,
and the rollout inherits that (d2i). The gap is largest in the initial-field response (TFNO / WPO 0.42 with the
forcing zero) but also present in the forcing response (0.73 with the initial field zero; d2j). The WPO trained on
initial fields is worse on forcing alone (0.046) than a16's WPO trained without them (0.026, same architecture),
while the TFNO barely changes (0.033 against 0.029). So learning the initial-field response seems to cost the WPO
capacity that the forcing response needed. The channel WPO of d2g (u0 as an input channel) was no better (0.0259),
so the entry point alone does not matter.

In `wpo_ic` the initial field enters once: lifted pointwise, it is the initial state of the first pole layer, whose
free response sum_n gamma_n e^{mu_n(k) t} uses that layer's poles and a k-independent gamma. Variants:
- `wpo_ic_sep`: the first layer's free response gets its own pole set (same parametrization and constraint as the
  forced poles; +60 parameters). Tests whether sharing the poles constrains the fit.
- `wpo_ic_all`: every pole layer adds a free response from its own input at t = 0 (layer 1: the lifted initial
  field as in `wpo_ic`; layers 2..4: the hidden state v_l(x, 0)), each with its own gamma, on the layer's poles.
  The width is reduced to keep the parameter count closest to `wpo_ic`'s 61,130. Tests whether the initial state
  needs to reach the deeper layers.
References (same protocol and seeds): `wpo_ic`, `tfno_chan` from d2g (`id`, `high`, `roll4`) and d2j (`force_only`,
`ic_only`).

**Setup.** d2g as it stands (6000 steps, batch 25, pole lr x30, 3 seeds); evaluation sets of d2g (`id`, `high`,
`roll4`) and d2j (`force_only`, `ic_only`, regenerated with their seeds).

**Checks before any fit.** The wrapper with neither change reproduces `WavenumberPoleOperator1d(ic=True)` exactly
(max difference 4.8e-15). Parameters: `wpo_ic` 61,130; `wpo_ic_sep` 61,160 (width 23); `wpo_ic_all` 60,869
(width 20).

**Predictions** (written before running).

1. `wpo_ic_sep`: mean `id` error within +-10 % of `wpo_ic`'s (0.0255): the shared poles are not the constraint.
2. `wpo_ic_all`: mean `id` error <= 0.85x `wpo_ic`'s (<= 0.0217).
3. Neither variant closes half the gap to the TFNO: both mean `id` errors > 0.019.
4. `wpo_ic_all` keeps the unseen-mode tolerance: mean `high` / `id` <= 4 (`wpo_ic` 2.35, TFNO 10.5).

    python experiments/D/d2_initial_conditions/d2k_ic_capacity.py [--check]      # results/d2k_ic_capacity.md / .json
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import d2g_ch_rollout as G  # noqa: E402
import d2j_ch_paths as J  # noqa: E402
from lfo.models import WavenumberPoleOperator1d  # noqa: E402
from lfo.models.wavenumber import WavenumberPoleLayer1d  # noqa: E402

R, E, M = G.R, G.E, G.M
KINDS, SEEDS = ("wpo_ic_sep", "wpo_ic_all"), (0, 1, 2)
SETS = ("id", "high", "roll4", "force_only", "ic_only")
OUT = HERE / "results" / "d2k_ic_capacity.json"
TARGET = 61130


def free(gamma, mu, v0, n_x, n_t, dt, K):
    """Free response sum_n gamma_n e^{mu_n t} of the initial channel field v0 (b, c, n_x): (b, c_out, n_x, n_t).
    gamma (N, 1, c_out, c_in, 2) constant in k, mu (N, K)."""
    gam = torch.complex(gamma[:, 0, ..., 0], gamma[:, 0, ..., 1])                         # (N, o, i)
    v0k = torch.fft.rfft(v0.to(torch.float64), dim=2)[:, :, :K]
    ej = torch.exp(mu[..., None] * dt * torch.arange(n_t, device=v0.device, dtype=torch.float64))
    return torch.fft.irfft(torch.einsum("noi,bik,nkt->bokt", gam, v0k, ej), n=n_x, dim=2)


class WPOICVar(nn.Module):
    """`WavenumberPoleOperator1d(ic=True)` with optional separate IC poles (`sep`) or a free response in every
    layer (`all`). mode None reproduces the original."""

    def __init__(self, t, mode=None, width=23):
        super().__init__()
        kw = dict(G.WPO_KW, width=width)
        self.mode = mode
        self.net = WavenumberPoleOperator1d(t, **kw, ic=True)
        pole_kw = {k: v for k, v in kw.items() if k not in ("width", "n_layers", "n_poles", "hidden")}
        self.ic_poles = WavenumberPoleLayer1d(kw["n_poles"], t, **pole_kw) if mode == "sep" else None
        if self.ic_poles is not None:
            self.ic_poles.beta.requires_grad_(False)                         # only its poles are used
        self.gammas = None
        if mode == "all":
            n, w = kw["n_poles"], width
            self.gammas = nn.ParameterList()
            for _ in range(len(self.net.K) - 1):
                g = torch.zeros(n, 1, w, w, 2, dtype=torch.float64)
                g[:, 0, :, :, 0] = torch.randn(n, w, w, dtype=torch.float64) / math.sqrt(w * n)
                self.gammas.append(nn.Parameter(g))

    def forward(self, f, u0):
        net = self.net
        n_x, n_t = f.shape[1], f.shape[2]
        dt = float(net.K[0].t[1] - net.K[0].t[0])
        v = net.lift(f.to(torch.float64)[..., None]).permute(0, 3, 1, 2)
        v0 = net.lift0(u0.to(torch.float64)[..., None]).permute(0, 2, 1)
        for i, (K, W) in enumerate(zip(net.K, net.W)):
            Km = n_x // 2 + 1 if K.modes is None else min(K.modes, n_x // 2 + 1)
            k = K.wavenumbers(n_x)[:Km]
            out = K(v)
            if i == 0:
                mu = (self.ic_poles if self.mode == "sep" else K).mu(k)
                out = out + free(K.gamma, mu, v0, n_x, n_t, dt, Km)
            elif self.mode == "all":
                out = out + free(self.gammas[i - 1], K.mu(k), v[..., 0], n_x, n_t, dt, Km)
            v = out + W(v.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            if i < len(net.K) - 1:
                v = F.gelu(v)
        return net.proj(v.permute(0, 2, 3, 1))[..., 0].to(f.dtype)


n_params = lambda m: sum(p.numel() for p in m.parameters() if p.requires_grad)


def build(kind, seed, t, device):
    mode = kind.removeprefix("wpo_ic_")
    width = 23
    if mode == "all":
        width = min(range(12, 24), key=lambda w: abs(n_params(WPOICVar(t, "all", w)) - TARGET))
    torch.manual_seed(seed)
    return WPOICVar(t, mode, width).to(device)


def make_sets():
    s = G.make_sets()
    j = dict(force_only=J.sample(np.random.default_rng(33), 100, (0.03, 1.0), (0.03, 1.0), zero_ic=True),
             ic_only=J.sample(np.random.default_rng(34), 100, (0.03, 1.0), (0.03, 1.0), zero_force=True))
    return dict(train=s["train"], id=s["id"], high=s["high"], roll4=s["roll4"], **j)


def check(t, device):
    torch.manual_seed(0)
    ref = G.build("wpo_ic", 0, t, device)
    torch.manual_seed(0)
    m = WPOICVar(t).to(device)
    m.net.load_state_dict(ref.net.state_dict())
    f = torch.randn(3, M.NX, M.NT1, dtype=torch.float64, device=device)
    u0 = torch.randn(3, M.NX, dtype=torch.float64, device=device)
    with torch.no_grad():
        d = float((m(f, u0) - ref(f, u0)).abs().max())
    print("wrapper against WavenumberPoleOperator1d(ic=True): max diff", d)
    for kind in KINDS:
        mm = build(kind, 0, t, device)
        print(kind, "width", mm.net.lift.out_features, "params", n_params(mm))
    print("wpo_ic params", n_params(ref))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t = np.linspace(0, M.T1, M.NT1)
    if "--check" in sys.argv:
        check(t, device)
        return
    sets = make_sets()
    out = json.loads(OUT.read_text()) if OUT.exists() else []
    for kind in KINDS:
        for seed in SEEDS:
            if any(r["model"] == kind and r["seed"] == seed for r in out):
                continue
            t0 = time.time()
            m = build(kind, seed, t, device)
            R.train(m, dict(train=sets["train"]), device, 30.0)
            m.eval()
            r = dict(model=kind, seed=seed, params=n_params(m), width=m.net.lift.out_features)
            for s in ("id", "high", "force_only", "ic_only"):
                f, u0, y = sets[s]
                r[s] = float(M.A.rel(R.predict(m, f, u0, device), y))
            f, u0, y = sets["roll4"]
            r["roll4"] = float(M.A.rel(E.rollout(m, f, u0, device, 4), y))
            r["seconds"] = round(time.time() - t0, 1)
            out.append(r)
            print(r, flush=True)
            OUT.parent.mkdir(exist_ok=True)
            OUT.write_text(json.dumps(out, indent=1))
            del m
            torch.cuda.empty_cache()
    g = json.loads((HERE / "results" / "d2g_ch_rollout.json").read_text())
    j = json.loads((HERE / "results" / "d2j_ch_paths.json").read_text())
    ref = {k: [dict(rg, **{s: rj[s] for s in ("force_only", "ic_only")}) for rg in g if rg["model"] == k
               for rj in j if rj["model"] == k and rj["seed"] == rg["seed"]] for k in ("wpo_ic", "tfno_chan")}
    rows = [(k, ref[k]) for k in ref] + [(k, [r for r in out if r["model"] == k]) for k in KINDS]
    mean = lambda rs, q: np.mean([r[q] for r in rs])
    lines = ["# (D2) Forced Cahn–Hilliard: IC-path capacity at a matched budget (d2g setup, 6000 steps)", "",
             "| model | params | id | high (ratio to id) | roll4 | force_only | ic_only |", "|---|---|---|---|---|---|---|"]
    for name, rs in rows:
        lines.append(f"| {name} | {rs[0]['params']} | {mean(rs, 'id'):.4f} ({min(r['id'] for r in rs):.4f}–"
                     f"{max(r['id'] for r in rs):.4f}) | {mean(rs, 'high'):.4f} ({mean(rs, 'high') / mean(rs, 'id'):.2f}) | "
                     f"{mean(rs, 'roll4'):.4f} | {mean(rs, 'force_only'):.4f} | {mean(rs, 'ic_only'):.4f} |")
    (OUT.with_suffix(".md")).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
