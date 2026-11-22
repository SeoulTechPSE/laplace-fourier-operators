"""(A) Next-level step 4: closing the nonlinear gap without giving up the physical-wavenumber form.

At an equal real budget the TFNO is 1.8x more accurate than the WPO on PDEBench Burgers (a37: nRMSE 0.026 against
0.045). A structural reason, found while designing this step: the WPO's residues beta_n(k) are polynomials in k^2,
so every pole layer applies a symbol that is even in k, and the pointwise channel mixes apply none. No part of the
operator can apply the derivative i k (odd), which Burgers' convection (u^2 / 2)_x needs at every layer; the FNO's
free complex weights per mode can. Two minimal changes that keep the poles as functions of k:

- `wpo_odd`: residues with powers (0, 1, 2, 3) of k / k_ref instead of (0, 2) (`beta_powers`, lfo.models,
  this commit's parent). Still a function of the physical wavenumber, so domain transfer and unseen modes should
  be kept.
- `wpo_local`: the pointwise channel mix W_l replaced by a periodic convolution over x with 5 taps (per time
  sample, so still causal). Its symbol is a function of k dx: right under domain transfer at the same spacing,
  not under resolution change.

Widths are reduced so that each variant's real parameter count is the closest to the WPO's in that setup.
Resulting widths (checked before this commit): P `wpo_odd` 17 (61,568 against 61,130), `wpo_local` 22 (63,763);
F `wpo_odd` 17 (58,083 against 59,529), `wpo_local` 22 (57,933).
Training as the WPO's selected configuration (lr 3e-3, pole lr x30, 12000 steps, 3 seeds; no new sweep).

**Part P: PDEBench 1D Burgers** (a37's setup, task and metrics; the `foh_kernel` fix applies).
References: a37's WPO rerun (0.045) and TFNO at the real budget (0.026).

**Part F: forced Burgers** (a36's setup: test, (5, 10] extrapolation, probe outside the band k = 9..16, unseen
modes k = 9..12). References: a36's WPO (a11) and `tfno_r`.

**Predictions** (written before running).

1. P: `wpo_odd`'s mean nRMSE <= 0.036 (0.8x the WPO's).
2. P: `wpo_local`'s mean nRMSE <= 0.036.
3. P: neither reaches the TFNO: both means > 0.026.
4. P: `wpo_local`'s mean 1024 / 256 nRMSE ratio > 1.3 (its stencil is tied to the grid spacing); `wpo_odd`'s <= 1.1.
5. F: `wpo_odd` keeps the form's benefits: unseen-mode ratio <= 3 and out-of-band probe error <= 0.3 in every seed.
6. F: `wpo_odd`'s mean test error <= the WPO's (0.0237).

    python experiments/A/deep/a39_odd_local.py      # results/a39_odd_local.md / .json
"""
from __future__ import annotations

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
import a37_pdebench_burgers as X37  # noqa: E402
from lfo.metrics import n_real_params  # noqa: E402
from lfo.models import WavenumberPoleOperator1d  # noqa: E402

OUT = HERE / "results" / "a39_odd_local.json"
SEEDS = (0, 1, 2)
CFG = dict(lr=3e-3, plr=30.0)
VARIANTS = ("wpo_odd", "wpo_local")
TAPS = 5


class LocalMix(nn.Module):
    """Periodic convolution over x (TAPS taps), applied at every time sample; (b, n_x, n_t, c) -> same."""

    def __init__(self, width):
        super().__init__()
        self.conv = nn.Conv1d(width, width, TAPS, padding=TAPS // 2, padding_mode="circular", dtype=torch.float64)

    def forward(self, v):
        b, n_x, n_t, c = v.shape
        y = self.conv(v.permute(0, 2, 3, 1).reshape(b * n_t, c, n_x))
        return y.reshape(b, n_t, c, n_x).permute(0, 3, 1, 2)


def make_op(t, variant, base_kw, width, ic=False):
    kw = dict(base_kw, width=width)
    if variant == "wpo_odd":
        kw["beta_powers"] = (0, 1, 2, 3)
    m = WavenumberPoleOperator1d(t, ic=ic, **kw)
    if variant == "wpo_local":
        m.W = nn.ModuleList(LocalMix(width) for _ in m.W)
    return m


def match_width(t, variant, base_kw, target, ic=False):
    return min(range(8, 32), key=lambda w: abs(n_real_params(make_op(t, variant, base_kw, w, ic)) - target))


class PWrap(nn.Module):
    """a37's interface: initial field (b, n_x) and n_t -> trajectory."""

    def __init__(self, op):
        super().__init__()
        self.net = op

    def forward(self, u0, n_t):
        return self.net(torch.zeros(*u0.shape, n_t, dtype=torch.float64, device=u0.device), u0)


def part_p(out, save, device):
    d = X37.load()
    t = d["t"].numpy()
    S = X37.splits(d)
    n_t = S["train"].shape[2]
    full_test = d["full"][X37.NTR:].to(torch.float64)
    base = {k: v for k, v in X37.WPO_KW.items() if k != "width"}
    target = n_real_params(X37.WPO(t))
    for v in VARIANTS:
        w = match_width(t, v, base, target, ic=True)
        for seed in SEEDS:
            if any(r["part"] == "P" and r["model"] == v and r["seed"] == seed for r in out["runs"]):
                continue
            t1 = time.time()
            torch.manual_seed(seed)
            m = PWrap(make_op(t, v, base, w, ic=True)).to(device)
            X37.train(m, S["train"], device, steps=X37.FINAL_STEPS, **CFG)
            m.eval()
            r = dict(part="P", model=v, seed=seed, width=w, real_params=n_real_params(m),
                     test=X37.nrmse(X37.predict(m, S["test"], device), S["test"]),
                     test_1024=X37.nrmse(X37.predict(m, full_test, device), full_test),
                     probe=X37.model_probe(m, t, 256, n_t, device), seconds=round(time.time() - t1, 1))
            out["runs"].append(r)
            print("P", v, seed, f"w {w} test {r['test']:.4f} 1024 {r['test_1024']:.4f} nu' {r['probe']['nu_hat']:.4f}",
                  f"{r['seconds']} s", flush=True)
            save()
            del m
            torch.cuda.empty_cache()


def part_f(out, save, device):
    data = M.make_data()
    data["high"] = X36.high_set()
    t = np.linspace(0, M.T1, M.NT1)
    pfk, pdt = M.probe_inputs()
    base = {k: v for k, v in F.WPO_KW.items() if k != "width"}
    target = n_real_params(WavenumberPoleOperator1d(t, **F.WPO_KW))
    for v in VARIANTS:
        w = match_width(t, v, base, target)
        for seed in SEEDS:
            if any(r["part"] == "F" and r["model"] == v and r["seed"] == seed for r in out["runs"]):
                continue
            t1 = time.time()
            torch.manual_seed(seed)
            m = make_op(t, v, base, w).to(device)
            F.train(m, data, device, steps=F.FINAL_STEPS, **CFG)
            r = X36.evaluate(m, v, seed, CFG, data, t, pfk, pdt, device, t1)
            r.update(part="F", width=w)
            out["runs"].append(r)
            print("F", v, seed, f"w {w} test {r['test']:.4f} ext/test {r['ext_late'] / r['test']:.2f}",
                  f"high/test {r['high'] / r['test']:.2f} oob {r['oob']:.3f}", f"{r['seconds']} s", flush=True)
            save()
            del m
            torch.cuda.empty_cache()


def main():
    device = torch.device("cuda")
    out = json.loads(OUT.read_text()) if OUT.exists() else dict(runs=[])
    save = lambda: OUT.write_text(json.dumps(out, indent=1, default=float))
    which = sys.argv[1:] or ["P", "F"]
    if "P" in which:
        part_p(out, save, device)
    if "F" in which:
        part_f(out, save, device)
    summarize(out)


def summarize(out):
    rg = lambda xs, f: f"{np.mean(xs):{f}} ({min(xs):{f}}–{max(xs):{f}})"
    a37 = json.loads((HERE / "results" / "a37_pdebench_burgers.json").read_text())
    a36 = json.loads((HERE / "results" / "a36_index_poles.json").read_text())
    lines = ["# (A) Odd residues and local mixing: the nonlinear gap with the poles kept in k", "",
             "Means over 3 seeds (range).", "", "## P: PDEBench 1D Burgers (a37 setup)", "",
             "| model | real params | nRMSE 256 | nRMSE 1024 / 256 | probe ν' (truth 0.1257) | dispersion error |",
             "|---|---|---|---|---|---|"]
    rowsP = [("wpo (a37 rerun)", a37.get("rerun", []))] + [
        ("tfno_r (a37)", [r for r in a37["final"] if r["model"] == "tfno_r"])] + [
        (v, [r for r in out["runs"] if r["part"] == "P" and r["model"] == v]) for v in VARIANTS]
    for name, rs in rowsP:
        if rs:
            lines.append(f"| {name} | {rs[0]['real_params']} | {rg([r['test'] for r in rs], '.4f')} | "
                         f"{rg([r['test_1024'] / r['test'] for r in rs], '.2f')} | {rg([r['probe']['nu_hat'] for r in rs], '.4f')} | "
                         f"{rg([r['probe']['disp_err'] for r in rs], '.3f')} |")
    lines += ["", "## F: forced Burgers (a36 setup)", "",
              "| model | real params | test | (5, 10] / test | probe outside the band | unseen / test |", "|---|---|---|---|---|---|"]
    rowsF = [("wpo (a11)", a36["ref"]), ("tfno_r (a36)", [r for r in a36["final"] if r["model"] == "tfno_r"])] + [
        (v, [r for r in out["runs"] if r["part"] == "F" and r["model"] == v]) for v in VARIANTS]
    for name, rs in rowsF:
        if rs:
            lines.append(f"| {name} | {rs[0]['real_params']} | {rg([r['test'] for r in rs], '.4f')} | "
                         f"{rg([r['ext_late'] / r['test'] for r in rs], '.2f')} | {rg([r['oob'] for r in rs], '.3f')} | "
                         f"{rg([r['high'] / r['test'] for r in rs], '.2f')} |")
    (OUT.with_suffix(".md")).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
